// connect.rs — 手机互联模块
// 提供局域网 HTTP 服务器，支持遥控指令、文件传输、摄像头流（P3）
// 与 lib.rs 完全解耦，仅通过 Tauri event 与前端通信

use std::net::UdpSocket;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::body::Bytes;
use axum::extract::{Multipart, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter};
use tokio::sync::Notify;
use tower_http::cors::CorsLayer;
use uuid::Uuid;

// ==================== 常量 ====================

/// 心跳超时时间（秒）：设备超过此时间未发送心跳则认为离线
/// 前端每 30 秒发送一次心跳，90 秒 = 连续 3 次未收到心跳
const HEARTBEAT_TIMEOUT_SECS: u64 = 90;
/// 心跳检查间隔（秒）：后台任务检查心跳的间隔
const HEARTBEAT_CHECK_INTERVAL_SECS: u64 = 15;

/// UDP 多播地址（LocalSend 协议默认）
const MULTICAST_GROUP: &str = "224.0.0.167";
/// UDP 多播端口
const MULTICAST_PORT: u16 = 53317;
/// 多播广播间隔（秒）
const MULTICAST_BROADCAST_INTERVAL_SECS: u64 = 5;
/// 普通 UDP 广播地址（fallback，部分网络不支持多播）
const BROADCAST_ADDR: &str = "255.255.255.255";

// ==================== 公开 API ====================

/// 启动手机互联 HTTP 服务器。
/// 在 Tauri setup() 中调用，服务启动后通过 `phone-server-ready` event 通知前端。
/// setup() 是同步上下文，没有 tokio runtime，因此在独立线程中创建新 runtime。
pub fn init_server(app: &AppHandle) -> Result<(), Box<dyn std::error::Error>> {
    let app_clone = app.clone();

    std::thread::Builder::new()
        .name("phone-connect-server".into())
        .spawn(move || {
            let rt = match tokio::runtime::Runtime::new() {
                Ok(rt) => rt,
                Err(e) => {
                    log::error!("[connect] 无法创建 tokio 运行时: {}", e);
                    return;
                }
            };

            rt.block_on(async {
                let token = generate_token();
                *ACTIVE_TOKEN.lock().unwrap() = Some(token.clone());
                SERVER_RUNNING.store(true, Ordering::SeqCst);

                run_server(app_clone, token).await;
            });
        })
        .map_err(|e| format!("无法启动服务器线程: {}", e))?;

    Ok(())
}

/// 停止服务器（可选，由设置开关调用）
#[allow(dead_code)]
pub fn stop_server() {
    SHUTDOWN_FLAG.store(true, Ordering::SeqCst);
}

/// Tauri IPC 命令：启动手机互联服务。
#[tauri::command]
pub fn phone_server_start(app: tauri::AppHandle) -> Result<(), String> {
    if SERVER_RUNNING.load(Ordering::SeqCst) {
        return Ok(());
    }
    SHUTDOWN_FLAG.store(false, Ordering::SeqCst);
    init_server(&app).map_err(|e| e.to_string())
}

/// Tauri IPC 命令：停止手机互联服务。
#[tauri::command]
pub fn phone_server_stop() {
    stop_server();
}

/// Tauri IPC 命令：查询手机互联服务状态。
/// 前端刷新后可主动调用此命令获取服务地址，无需等待事件。
#[tauri::command]
pub fn phone_server_status() -> Option<serde_json::Value> {
    if !SERVER_RUNNING.load(Ordering::SeqCst) {
        return None;
    }
    let ip = SERVER_IP.lock().unwrap().clone();
    let port = *SERVER_PORT.lock().unwrap();
    let token = ACTIVE_TOKEN.lock().unwrap().clone();
    let all_ips = get_all_local_ips();
    match (ip, port, token) {
        (Some(ip), Some(port), Some(token)) => Some(serde_json::json!({
            "ip": ip,
            "port": port,
            "token": token,
            "ips": all_ips,
        })),
        _ => None,
    }
}

/// Tauri IPC 命令：查询手机摄像头状态。
#[tauri::command]
pub fn phone_camera_status() -> serde_json::Value {
    let active = CAMERA_ACTIVE.load(Ordering::SeqCst);
    let session = CAMERA_SESSION.lock().unwrap().clone();
    let codec = match *CAMERA_CODEC.lock().unwrap() {
        CameraCodec::H264 => "h264",
        CameraCodec::Jpeg => "jpeg",
    };
    serde_json::json!({
        "active": active,
        "session": session,
        "codec": codec,
    })
}

/// Tauri IPC 命令：启动手机摄像头。
#[tauri::command]
pub fn phone_camera_start() -> Result<bool, String> {
    // 这个命令主要由前端调用，实际的摄像头启动是通过HTTP API触发的
    // 这里只是返回当前状态
    let active = CAMERA_ACTIVE.load(Ordering::SeqCst);
    Ok(active)
}

/// Tauri IPC 命令：停止手机摄像头。
#[tauri::command]
pub fn phone_camera_stop() -> Result<bool, String> {
    // 这个命令主要由前端调用，实际的摄像头停止是通过HTTP API触发的
    // 这里只是返回当前状态
    let active = CAMERA_ACTIVE.load(Ordering::SeqCst);
    Ok(!active)
}

// ==================== 内部状态 ====================

static SHUTDOWN_FLAG: AtomicBool = AtomicBool::new(false);
static ACTIVE_TOKEN: Lazy<Mutex<Option<String>>> = Lazy::new(|| Mutex::new(None));
static SERVER_RUNNING: AtomicBool = AtomicBool::new(false);
static SERVER_IP: Lazy<Mutex<Option<String>>> = Lazy::new(|| Mutex::new(None));
static SERVER_PORT: Lazy<Mutex<Option<u16>>> = Lazy::new(|| Mutex::new(None));
static CAMERA_ACTIVE: AtomicBool = AtomicBool::new(false);
static CAMERA_SESSION: Lazy<Mutex<Option<String>>> = Lazy::new(|| Mutex::new(None));
static CAMERA_CODEC: Lazy<Mutex<CameraCodec>> = Lazy::new(|| Mutex::new(CameraCodec::Jpeg));

#[derive(Clone, Serialize, Deserialize)]
struct SessionInfo {
    session_id: String,
    device_name: String,
    connected_at: u64,
    last_heartbeat: u64,
}

struct CameraFrame {
    jpeg: Bytes,
}

/// 摄像头编码类型
#[derive(Clone, Copy, PartialEq)]
enum CameraCodec {
    Jpeg,
    H264,
}

struct ServerState {
    app: AppHandle,
    token: String,
    sessions: Arc<Mutex<Vec<SessionInfo>>>,
    camera_active: Arc<AtomicBool>,
    camera_session: Arc<Mutex<Option<String>>>,
    camera_codec: Arc<Mutex<CameraCodec>>,
    camera_frame: Arc<Mutex<Option<CameraFrame>>>,
    camera_notify: Arc<Notify>,
    /// H.264 fMP4 segment 广播通道（init segment + video segments）
    h264_tx: tokio::sync::broadcast::Sender<Bytes>,
}

// ==================== 数据结构 ====================

#[derive(Deserialize)]
struct ConnectQuery {
    token: String,
    #[serde(default)]
    device_name: Option<String>,
}

#[derive(Serialize)]
struct ConnectResponse {
    session: String,
    server_version: String,
    capabilities: Vec<String>,
}

#[derive(Deserialize)]
struct ControlBody {
    #[serde(default)]
    params: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct ControlResponse {
    success: bool,
    action: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Serialize)]
struct StatusResponse {
    server_running: bool,
    connected_devices: usize,
    sessions: Vec<SessionInfo>,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

#[derive(Serialize)]
struct CameraStatusResponse {
    active: bool,
    device_name: Option<String>,
}

// ==================== UDP 多播广播 ====================

/// 获取所有非回环的本地 IPv4 地址
fn get_all_local_ips() -> Vec<String> {
    let mut ips = Vec::new();
    if let Ok(ifaces) = local_ip_address::list_afinet_netifas() {
        for (_name, ip) in ifaces {
            if let std::net::IpAddr::V4(v4) = ip {
                if !v4.is_loopback() && !v4.is_unspecified() {
                    ips.push(v4.to_string());
                }
            }
        }
    }
    if ips.is_empty() {
        // fallback
        if let Some(ip) = get_local_ip() {
            ips.push(ip);
        }
    }
    ips
}

/// 启动 UDP 广播，让局域网内的手机端自动发现本机。
/// 在每个网络接口上都发送广播（包括热点接口）。
fn start_multicast_broadcast(port: u16, token: String) {
    std::thread::Builder::new()
        .name("multicast-broadcast".into())
        .spawn(move || {
            let fingerprint = Uuid::new_v4().to_string();
            let device_model = format!(
                "{} ({}-bit)",
                sysinfo::System::host_name().unwrap_or_else(|| "ViewStage".into()),
                if cfg!(target_pointer_width = "64") { 64 } else { 32 }
            );

            log::info!("[connect] 广播线程已启动");

            // 预创建 UDP socket，循环内复用
            let mcast_socket = UdpSocket::bind("0.0.0.0:0").ok();
            if let Some(ref sock) = mcast_socket {
                let _ = sock.set_multicast_ttl_v4(32);
            }
            let bcast_socket = UdpSocket::bind("0.0.0.0:0").ok();
            if let Some(ref sock) = bcast_socket {
                let _ = sock.set_broadcast(true);
            }

            let mcast_dest = format!("{}:{}", MULTICAST_GROUP, MULTICAST_PORT);
            let global_bcast_dest = format!("{}:{}", BROADCAST_ADDR, MULTICAST_PORT);

            // 缓存接口列表，每 60 秒刷新一次（而非每 5 秒）
            let mut cached_ips: Vec<String> = Vec::new();
            let mut last_ip_refresh = std::time::Instant::now() - Duration::from_secs(60);

            let mut send_count: u32 = 0;
            loop {
                if SHUTDOWN_FLAG.load(Ordering::SeqCst) {
                    log::info!("[connect] 广播收到关闭信号，停止");
                    break;
                }

                // 每 60 秒刷新接口列表
                if last_ip_refresh.elapsed() >= Duration::from_secs(60) {
                    cached_ips = get_all_local_ips();
                    last_ip_refresh = std::time::Instant::now();
                }

                for ip in &cached_ips {
                    let message = serde_json::json!({
                        "alias": "ViewStage",
                        "version": "2.0",
                        "deviceModel": device_model,
                        "deviceType": "desktop",
                        "fingerprint": fingerprint,
                        "port": port,
                        "protocol": "http",
                        "announce": true,
                        "token": token,
                        "ip": ip,
                    });
                    let payload = message.to_string().into_bytes();

                    // 多播
                    if let Some(ref sock) = mcast_socket {
                        let _ = sock.send_to(&payload, &mcast_dest);
                    }

                    // 子网广播（/24 假设）
                    if let Ok(v4) = ip.parse::<std::net::Ipv4Addr>() {
                        let bcast_ip = std::net::Ipv4Addr::new(v4.octets()[0], v4.octets()[1], v4.octets()[2], 255);
                        let bcast_dest = format!("{}:{}", bcast_ip, MULTICAST_PORT);
                        if let Some(ref sock) = bcast_socket {
                            let _ = sock.send_to(&payload, &bcast_dest);
                        }
                    }

                    // 全局广播
                    if let Some(ref sock) = bcast_socket {
                        let _ = sock.send_to(&payload, &global_bcast_dest);
                    }
                }

                send_count += 1;
                if send_count <= 3 || send_count % 12 == 0 {
                    log::info!(
                        "[connect] 广播 #{} 已发送 ({} 个接口: {:?})",
                        send_count, cached_ips.len(), cached_ips
                    );
                }

                std::thread::sleep(Duration::from_secs(MULTICAST_BROADCAST_INTERVAL_SECS));
            }
        })
        .ok();
}

// ==================== 服务器运行 ====================

async fn run_server(app: AppHandle, token: String) {
    let sessions: Arc<Mutex<Vec<SessionInfo>>> = Arc::new(Mutex::new(Vec::new()));
    let camera_active = Arc::new(AtomicBool::new(false));
    let camera_session: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let camera_codec = Arc::new(Mutex::new(CameraCodec::Jpeg));
    let camera_frame: Arc<Mutex<Option<CameraFrame>>> = Arc::new(Mutex::new(None));
    let camera_notify = Arc::new(Notify::new());
    let (h264_tx, _) = tokio::sync::broadcast::channel::<Bytes>(64);

    let state = Arc::new(ServerState {
        app: app.clone(),
        token,
        sessions: sessions.clone(),
        camera_active: camera_active.clone(),
        camera_session: camera_session.clone(),
        camera_codec: camera_codec.clone(),
        camera_frame: camera_frame.clone(),
        camera_notify: camera_notify.clone(),
        h264_tx: h264_tx.clone(),
    });

    let router = Router::new()
        .route("/connect", get(handle_connect))
        .route("/disconnect", post(handle_disconnect))
        .route("/control/{action}", post(handle_control))
        .route("/file/upload", post(handle_file_upload))
        .route("/file/download", get(handle_file_download))
        .route("/status", get(handle_status))
        .route("/heartbeat", post(handle_heartbeat))
        .route("/camera/status", get(handle_camera_status))
        .route("/camera/start", post(handle_camera_start))
        .route("/camera/stop", post(handle_camera_stop))
        .route("/camera/stream", get(handle_camera_stream))
        .route("/camera/mjpeg", get(handle_camera_mjpeg))
        .route("/camera/h264", get(handle_camera_h264))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let listener = match tokio::net::TcpListener::bind("0.0.0.0:0").await {
        Ok(l) => l,
        Err(e) => {
            log::error!("[connect] 无法绑定端口: {}", e);
            let _ = app.emit("phone-server-error", format!("无法绑定端口: {}", e));
            return;
        }
    };

    let addr = match listener.local_addr() {
        Ok(a) => a,
        Err(e) => {
            log::error!("[connect] 无法获取本地地址: {}", e);
            return;
        }
    };

    let local_ip = get_local_ip().unwrap_or_else(|| "127.0.0.1".to_string());
    let port = addr.port();
    let token_ref = ACTIVE_TOKEN.lock().unwrap().clone().unwrap_or_default();

    // 保存服务地址供前端查询
    *SERVER_IP.lock().unwrap() = Some(local_ip.clone());
    *SERVER_PORT.lock().unwrap() = Some(port);

    let all_ips = get_all_local_ips();
    log::info!(
        "[connect] 服务已启动 http://{}:{} (token={}, 所有接口: {:?})",
        local_ip,
        port,
        token_ref,
        all_ips
    );

    log::info!("[connect] 发射 phone-server-ready 事件");
    let emit_result = app.emit(
        "phone-server-ready",
        serde_json::json!({
            "ip": local_ip,
            "port": port,
            "token": token_ref,
            "ips": all_ips,
        }),
    );
    if let Err(e) = &emit_result {
        log::warn!("[connect] phone-server-ready 事件发射失败: {:?}", e);
    }

    // 启动 UDP 多播广播，让手机端自动发现本机
    start_multicast_broadcast(port, token_ref.clone());

    // 运行服务器，定期检查关闭标志
    let shutdown_future = async {
        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
            if SHUTDOWN_FLAG.load(Ordering::SeqCst) {
                break;
            }
        }
    };

    // 心跳检测任务
    let sessions_clone = sessions.clone();
    let app_clone = app.clone();
    let camera_active_clone = camera_active.clone();
    let camera_session_clone = camera_session.clone();
    let camera_frame_clone = camera_frame.clone();
    let heartbeat_future = async move {
        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(HEARTBEAT_CHECK_INTERVAL_SECS)).await;
            
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            
            let mut sessions = sessions_clone.lock().unwrap();
            let mut expired_sessions = Vec::new();
            
            // 找出超时的session
            for session in sessions.iter() {
                // 安全处理：如果 last_heartbeat > now（时钟回拨），视为超时
                let elapsed = if now >= session.last_heartbeat {
                    now - session.last_heartbeat
                } else {
                    log::warn!(
                        "[connect] 检测到时钟回拨: now={}, last_heartbeat={}, 设备={}",
                        now, session.last_heartbeat, session.device_name
                    );
                    HEARTBEAT_TIMEOUT_SECS + 1 // 视为超时
                };

                if elapsed > HEARTBEAT_TIMEOUT_SECS {
                    expired_sessions.push(session.clone());
                }
            }
            
            // 移除超时的session并发送断开事件
            for expired in expired_sessions {
                log::info!(
                    "[connect] 设备心跳超时离线: {} (session={}, 超时{}秒)",
                    expired.device_name,
                    expired.session_id,
                    now.saturating_sub(expired.last_heartbeat)
                );
                
                // 从列表中移除
                sessions.retain(|s| s.session_id != expired.session_id);

                // 如果超时设备是摄像头持有者，清理摄像头状态
                let is_camera_owner = camera_session_clone.lock().unwrap()
                    .as_ref() == Some(&expired.session_id);
                if is_camera_owner {
                    log::info!("[connect] 摄像头持有者离线，清理摄像头状态");
                    camera_active_clone.store(false, Ordering::SeqCst);
                    *camera_session_clone.lock().unwrap() = None;
                    *camera_frame_clone.lock().unwrap() = None;
                    CAMERA_ACTIVE.store(false, Ordering::SeqCst);
                    *CAMERA_SESSION.lock().unwrap() = None;
                }
                
                // 发送断开事件
                let _ = app_clone.emit(
                    "phone-device-disconnected",
                    serde_json::json!({
                        "session": expired.session_id,
                        "device_name": expired.device_name,
                    }),
                );
            }
        }
    };

    tokio::select! {
        result = axum::serve(listener, router) => {
            if let Err(e) = result {
                log::error!("[connect] 服务器错误: {}", e);
            }
        }
        _ = shutdown_future => {
            log::info!("[connect] 收到关闭信号，服务停止");
        }
        _ = heartbeat_future => {
            log::info!("[connect] 心跳检测任务结束");
        }
    }

    SERVER_RUNNING.store(false, Ordering::SeqCst);
    *ACTIVE_TOKEN.lock().unwrap() = None;
    log::info!("[connect] 服务已停止");
}

// ==================== 路由处理器 ====================

/// GET /connect?token=xxx&device_name=iPhone
/// 握手接口：校验 token，返回 session ID
async fn handle_connect(
    State(state): State<Arc<ServerState>>,
    Query(query): Query<ConnectQuery>,
) -> impl IntoResponse {
    if query.token != state.token {
        return (
            StatusCode::FORBIDDEN,
            Json(ErrorResponse {
                error: "invalid_token".to_string(),
            }),
        )
            .into_response();
    }

    let session_id = Uuid::new_v4().to_string();
    let device_name = query.device_name.unwrap_or_else(|| "未知设备".to_string());
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let session = SessionInfo {
        session_id: session_id.clone(),
        device_name: device_name.clone(),
        connected_at: now,
        last_heartbeat: now,
    };

    state.sessions.lock().unwrap().push(session);

    log::info!("[connect] 设备已连接: {} (session={})", device_name, session_id);
    let _ = state.app.emit(
        "phone-device-connected",
        serde_json::json!({
            "session": session_id,
            "device_name": device_name,
        }),
    );

    (
        StatusCode::OK,
        Json(ConnectResponse {
            session: session_id,
            server_version: "1.0".to_string(),
            capabilities: vec![
                "control".to_string(),
                "file_upload".to_string(),
                "camera".to_string(),
            ],
        }),
    )
        .into_response()
}

/// POST /disconnect
/// 主动断开连接：手机端点击"断开"时调用
async fn handle_disconnect(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if !validate_session(&headers, &state) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(ErrorResponse {
                error: "unauthorized".to_string(),
            }),
        )
            .into_response();
    }

    // 提取 session_id
    let session_id = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
        .map(|s| s.to_string());

    if let Some(sid) = &session_id {
        let mut sessions = state.sessions.lock().unwrap();
        let device_name = sessions
            .iter()
            .find(|s| s.session_id == *sid)
            .map(|s| s.device_name.clone())
            .unwrap_or_else(|| "未知设备".to_string());

        sessions.retain(|s| s.session_id != *sid);

        // 如果是摄像头持有者，清理摄像头状态
        let is_camera_owner = state.camera_session.lock().unwrap().as_ref() == Some(sid);
        if is_camera_owner {
            log::info!("[connect] 摄像头持有者主动断开，清理摄像头状态");
            state.camera_active.store(false, Ordering::SeqCst);
            *state.camera_session.lock().unwrap() = None;
            *state.camera_frame.lock().unwrap() = None;
            CAMERA_ACTIVE.store(false, Ordering::SeqCst);
            *CAMERA_SESSION.lock().unwrap() = None;
        }

        log::info!("[connect] 设备主动断开: {} (session={})", device_name, sid);
        let _ = state.app.emit(
            "phone-device-disconnected",
            serde_json::json!({
                "session": sid,
                "device_name": device_name,
            }),
        );
    }

    (StatusCode::OK, Json(serde_json::json!({ "success": true }))).into_response()
}

/// POST /control/{action}
/// 遥控指令：翻页、切换工具、缩放等
async fn handle_control(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    axum::extract::Path(action): axum::extract::Path<String>,
    Json(body): Json<ControlBody>,
) -> impl IntoResponse {
    if !validate_session(&headers, &state) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(ErrorResponse {
                error: "unauthorized".to_string(),
            }),
        )
            .into_response();
    }

    let valid_actions = [
        "next", "prev", "first-page", "last-page", "goto-page",
        "annotate", "move", "eraser", "screenshot",
        "zoom-in", "zoom-out", "zoom-reset", "toggle-blackboard",
        "toggle-camera", "mirror", "clear-annotations", "undo", "settings",
    ];

    if !valid_actions.contains(&action.as_str()) {
        return (
            StatusCode::BAD_REQUEST,
            Json(ControlResponse {
                success: false,
                action,
                error: Some("unknown_action".to_string()),
            }),
        )
            .into_response();
    }

    log::info!("[connect] 遥控指令: {}", action);

    let _ = state.app.emit(
        "phone-control",
        serde_json::json!({
            "action": action,
            "params": body.params,
        }),
    );

    (
        StatusCode::OK,
        Json(ControlResponse {
            success: true,
            action,
            error: None,
        }),
    )
        .into_response()
}

/// POST /file/upload
/// 文件上传：multipart form-data
async fn handle_file_upload(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
    multipart: Multipart,
) -> impl IntoResponse {
    if !validate_session(&headers, &state) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(ErrorResponse {
                error: "unauthorized".to_string(),
            }),
        )
            .into_response();
    }

    match save_upload(multipart).await {
        Ok((path, name, size)) => {
            log::info!("[connect] 文件已接收: {} ({} bytes)", name, size);

            let _ = state.app.emit(
                "phone-file-received",
                serde_json::json!({
                    "path": path,
                    "name": name,
                    "size": size,
                }),
            );

            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "success": true,
                    "path": path,
                    "name": name,
                    "size": size,
                })),
            )
                .into_response()
        }
        Err(e) => {
            log::error!("[connect] 文件上传失败: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("upload_failed: {}", e),
                }),
            )
                .into_response()
        }
    }
}

/// GET /file/download?path=xxx
/// 下载已上传的文件（供前端用 fetch 获取 blob URL）
/// 无需认证：仅桌面端自身调用，从 phone-uploads 读取文件
async fn handle_file_download(
    Query(query): Query<std::collections::HashMap<String, String>>,
) -> impl IntoResponse {
    let Some(path) = query.get("path") else {
        return (StatusCode::BAD_REQUEST, Bytes::new()).into_response();
    };

    // 安全检查：只允许访问 phone-uploads 目录
    let cache_base = dirs::cache_dir()
        .map(|p| p.join("SECTL").join("ViewStage").join("phone-uploads"));
    let Ok(file_path) = std::path::Path::new(path).canonicalize() else {
        return (StatusCode::NOT_FOUND, Bytes::new()).into_response();
    };
    if let Some(base) = &cache_base {
        match base.canonicalize() {
            Ok(base) => {
                if !file_path.starts_with(&base) {
                    return (StatusCode::FORBIDDEN, Bytes::new()).into_response();
                }
            }
            Err(_) => {
                // 目录不存在，没有任何文件应该可访问
                return (StatusCode::FORBIDDEN, Bytes::new()).into_response();
            }
        }
    }

    let mime = match file_path.extension().and_then(|e| e.to_str()) {
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("png") => "image/png",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        Some("bmp") => "image/bmp",
        Some("pdf") => "application/pdf",
        _ => "application/octet-stream",
    };

    // 文件读取放到阻塞线程，不阻塞 Tokio worker
    match tokio::task::spawn_blocking(move || std::fs::read(&file_path)).await {
        Ok(Ok(data)) => {
            let mut h = HeaderMap::new();
            h.insert("Content-Type", mime.parse().unwrap());
            h.insert("Cache-Control", "no-store".parse().unwrap());
            (StatusCode::OK, h, Bytes::from(data)).into_response()
        }
        _ => (StatusCode::NOT_FOUND, Bytes::new()).into_response(),
    }
}

/// GET /status
/// 查询服务器状态
async fn handle_status(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if !validate_session(&headers, &state) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(ErrorResponse {
                error: "unauthorized".to_string(),
            }),
        )
            .into_response();
    }

    let sessions = state.sessions.lock().unwrap().clone();

    (
        StatusCode::OK,
        Json(StatusResponse {
            server_running: SERVER_RUNNING.load(Ordering::SeqCst),
            connected_devices: sessions.len(),
            sessions,
        }),
    )
        .into_response()
}

/// POST /heartbeat
/// 心跳接口：设备定期调用以更新最后心跳时间
async fn handle_heartbeat(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if !validate_session(&headers, &state) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(ErrorResponse {
                error: "unauthorized".to_string(),
            }),
        )
            .into_response();
    }

    // 获取session_id
    if let Some(auth) = headers.get("authorization") {
        if let Ok(auth_str) = auth.to_str() {
            let prefix = "Bearer ";
            if auth_str.starts_with(prefix) {
                let session_id = &auth_str[prefix.len()..];
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                
                // 更新最后心跳时间
                let mut sessions = state.sessions.lock().unwrap();
                if let Some(session) = sessions.iter_mut().find(|s| s.session_id == session_id) {
                    session.last_heartbeat = now;
                    log::debug!("[connect] 心跳更新: {} (session={})", session.device_name, session_id);
                }
            }
        }
    }

    (StatusCode::OK, Json(serde_json::json!({ "success": true }))).into_response()
}

/// GET /camera/status
/// 查询手机摄像头状态
async fn handle_camera_status(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if !validate_session(&headers, &state) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(ErrorResponse {
                error: "unauthorized".to_string(),
            }),
        )
            .into_response();
    }

    let active = state.camera_active.load(Ordering::SeqCst);
    let session = state.camera_session.lock().unwrap().clone();
    let device_name = if let Some(ref session_id) = session {
        state.sessions.lock().unwrap()
            .iter()
            .find(|s| s.session_id == *session_id)
            .map(|s| s.device_name.clone())
    } else {
        None
    };

    (
        StatusCode::OK,
        Json(CameraStatusResponse {
            active,
            device_name,
        }),
    )
        .into_response()
}

/// POST /camera/start
/// 开始手机摄像头推流
async fn handle_camera_start(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if !validate_session(&headers, &state) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(ErrorResponse {
                error: "unauthorized".to_string(),
            }),
        )
            .into_response();
    }

    // 获取session_id
    let session_id = if let Some(auth) = headers.get("authorization") {
        if let Ok(auth_str) = auth.to_str() {
            let prefix = "Bearer ";
            if auth_str.starts_with(prefix) {
                auth_str[prefix.len()..].to_string()
            } else {
                return (
                    StatusCode::UNAUTHORIZED,
                    Json(ErrorResponse {
                        error: "unauthorized".to_string(),
                    }),
                )
                    .into_response();
            }
        } else {
            return (
                StatusCode::UNAUTHORIZED,
                Json(ErrorResponse {
                    error: "unauthorized".to_string(),
                }),
            )
                .into_response();
        }
    } else {
        return (
            StatusCode::UNAUTHORIZED,
            Json(ErrorResponse {
                error: "unauthorized".to_string(),
            }),
        )
            .into_response();
    };

    // 检查是否已有其他设备在使用摄像头
    if state.camera_active.load(Ordering::SeqCst) {
        let current_session = state.camera_session.lock().unwrap().clone();
        if current_session.as_deref() != Some(&session_id) {
            return (
                StatusCode::CONFLICT,
                Json(ErrorResponse {
                    error: "camera_in_use".to_string(),
                }),
            )
                .into_response();
        }
    }

    // 激活摄像头
    state.camera_active.store(true, Ordering::SeqCst);
    *state.camera_session.lock().unwrap() = Some(session_id.clone());
    CAMERA_ACTIVE.store(true, Ordering::SeqCst);
    *CAMERA_SESSION.lock().unwrap() = Some(session_id.clone());

    // 获取设备名称
    let device_name = state.sessions.lock().unwrap()
        .iter()
        .find(|s| s.session_id == session_id)
        .map(|s| s.device_name.clone())
        .unwrap_or_else(|| "未知设备".to_string());

    log::info!("[connect] 手机摄像头已激活: {} (session={})", device_name, session_id);

    // 通知前端
    let _ = state.app.emit(
        "phone-camera-started",
        serde_json::json!({
            "session": session_id,
            "device_name": device_name,
        }),
    );

    (StatusCode::OK, Json(serde_json::json!({ "success": true }))).into_response()
}

/// POST /camera/stop
/// 停止手机摄像头推流
async fn handle_camera_stop(
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
) -> impl IntoResponse {
    if !validate_session(&headers, &state) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(ErrorResponse {
                error: "unauthorized".to_string(),
            }),
        )
            .into_response();
    }

    // 获取session_id
    let session_id = if let Some(auth) = headers.get("authorization") {
        if let Ok(auth_str) = auth.to_str() {
            let prefix = "Bearer ";
            if auth_str.starts_with(prefix) {
                auth_str[prefix.len()..].to_string()
            } else {
                return (
                    StatusCode::UNAUTHORIZED,
                    Json(ErrorResponse {
                        error: "unauthorized".to_string(),
                    }),
                )
                    .into_response();
            }
        } else {
            return (
                StatusCode::UNAUTHORIZED,
                Json(ErrorResponse {
                    error: "unauthorized".to_string(),
                }),
            )
                .into_response();
        }
    } else {
        return (
            StatusCode::UNAUTHORIZED,
            Json(ErrorResponse {
                error: "unauthorized".to_string(),
            }),
        )
            .into_response();
    };

    // 检查是否是摄像头持有者
    let current_session = state.camera_session.lock().unwrap().clone();
    if current_session.as_deref() != Some(&session_id) {
        return (
            StatusCode::FORBIDDEN,
            Json(ErrorResponse {
                error: "not_camera_owner".to_string(),
            }),
        )
            .into_response();
    }

    // 停止摄像头
    state.camera_active.store(false, Ordering::SeqCst);
    *state.camera_session.lock().unwrap() = None;
    CAMERA_ACTIVE.store(false, Ordering::SeqCst);
    *CAMERA_SESSION.lock().unwrap() = None;
    *state.camera_frame.lock().unwrap() = None;

    log::info!("[connect] 手机摄像头已停止: session={}", session_id);

    // 通知前端
    let _ = state.app.emit("phone-camera-stopped", serde_json::json!({}));

    (StatusCode::OK, Json(serde_json::json!({ "success": true }))).into_response()
}

/// GET /camera/stream
/// WebSocket端点，用于接收手机摄像头视频帧
async fn handle_camera_stream(
    ws: axum::extract::ws::WebSocketUpgrade,
    State(state): State<Arc<ServerState>>,
    headers: HeaderMap,
) -> impl IntoResponse {
    // WebSocket 握手前验证 session
    if !validate_session(&headers, &state) {
        return (StatusCode::UNAUTHORIZED, Bytes::new()).into_response();
    }
    ws.on_upgrade(move |socket| handle_camera_ws(socket, state))
}

async fn handle_camera_ws(
    mut socket: axum::extract::ws::WebSocket,
    state: Arc<ServerState>,
) {
    use axum::extract::ws::Message;
    use futures::StreamExt;

    log::info!("[connect] 手机摄像头WebSocket连接已建立");

    while let Some(msg) = socket.next().await {
        let msg = match msg {
            Ok(msg) => msg,
            Err(e) => {
                log::error!("[connect] WebSocket错误: {}", e);
                break;
            }
        };

        match msg {
            Message::Binary(data) => {
                if !state.camera_active.load(Ordering::SeqCst) || data.is_empty() {
                    continue;
                }

                // 协议：首字节标识帧类型
                // 0x01 = H.264 init segment (fMP4 moov)
                // 0x02 = H.264 video segment (fMP4 moof+mdat)
                // 0x03 = JPEG 帧
                let frame_type = data[0];
                let payload = &data[1..];

                match frame_type {
                    0x01 | 0x02 => {
                        // H.264 fMP4 segment → 广播给所有 SSE 客户端
                        *state.camera_codec.lock().unwrap() = CameraCodec::H264;
                        *CAMERA_CODEC.lock().unwrap() = CameraCodec::H264;
                        let segment = Bytes::from(payload.to_vec());
                        let _ = state.h264_tx.send(segment);
                    }
                    0x03 => {
                        // JPEG 帧 → 存入单帧槽位（兼容旧流程）
                        *state.camera_codec.lock().unwrap() = CameraCodec::Jpeg;
                        *CAMERA_CODEC.lock().unwrap() = CameraCodec::Jpeg;
                        let mut frame = state.camera_frame.lock().unwrap();
                        *frame = Some(CameraFrame {
                            jpeg: Bytes::from(payload.to_vec()),
                        });
                        drop(frame);
                        state.camera_notify.notify_waiters();
                    }
                    _ => {
                        log::warn!("[connect] 未知帧类型: 0x{:02X}", frame_type);
                    }
                }
            }
            Message::Close(_) => {
                log::info!("[connect] 手机摄像头WebSocket连接已关闭");
                break;
            }
            _ => {}
        }
    }

    // 连接断开，清理状态
    state.camera_active.store(false, Ordering::SeqCst);
    *state.camera_session.lock().unwrap() = None;
    *state.camera_frame.lock().unwrap() = None;
    CAMERA_ACTIVE.store(false, Ordering::SeqCst);
    *CAMERA_SESSION.lock().unwrap() = None;
    let _ = state.app.emit("phone-camera-stopped", serde_json::json!({}));
    log::info!("[connect] 手机摄像头已断开");
}

/// GET /camera/mjpeg
/// 返回 MJPEG 流（multipart/x-mixed-replace）
/// 前端直接用 <img src="http://ip:port/camera/mjpeg"> 显示，零 JS 开销
async fn handle_camera_mjpeg(
    State(state): State<Arc<ServerState>>,
) -> impl IntoResponse {
    if !state.camera_active.load(Ordering::SeqCst) {
        return (StatusCode::SERVICE_UNAVAILABLE, Bytes::new()).into_response();
    }

    let boundary = "jpgboundary";
    let stream = async_stream::stream! {
        loop {
            // 等待新帧通知或超时（防止连接永久挂起）
            let notify_future = state.camera_notify.notified();
            let timeout_future = tokio::time::sleep(tokio::time::Duration::from_secs(5));

            tokio::select! {
                _ = notify_future => {}
                _ = timeout_future => {}
            }

            if !state.camera_active.load(Ordering::SeqCst) {
                break;
            }

            let frame_data = {
                let frame = state.camera_frame.lock().unwrap();
                frame.as_ref().map(|f| f.jpeg.clone())
            };

            if let Some(jpeg) = frame_data {
                let header = format!(
                    "--{}\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n",
                    boundary,
                    jpeg.len()
                );
                yield Ok::<_, std::convert::Infallible>(Bytes::from(header));
                yield Ok(jpeg);
                yield Ok(Bytes::from("\r\n"));
            }
        }
    };

    let mut headers = HeaderMap::new();
    headers.insert(
        "Content-Type",
        format!("multipart/x-mixed-replace; boundary={}", boundary)
            .parse()
            .unwrap(),
    );
    headers.insert("Cache-Control", "no-store, no-cache".parse().unwrap());
    headers.insert("Access-Control-Allow-Origin", "*".parse().unwrap());

    (StatusCode::OK, headers, axum::body::Body::from_stream(stream)).into_response()
}

/// GET /camera/h264
/// 返回 H.264 fMP4 二进制流，前端通过 MSE 播放。
/// 协议：每个 segment 前加 4 字节大端长度头
///   [4 bytes length][segment bytes][4 bytes length][segment bytes]...
/// 第一个 segment 一定是 init segment (moov)，后续是 video segments (moof+mdat)
async fn handle_camera_h264(
    State(state): State<Arc<ServerState>>,
) -> impl IntoResponse {
    if !state.camera_active.load(Ordering::SeqCst) {
        return (StatusCode::SERVICE_UNAVAILABLE, Bytes::new()).into_response();
    }

    let mut rx = state.h264_tx.subscribe();

    let stream = async_stream::stream! {
        loop {
            if !state.camera_active.load(Ordering::SeqCst) {
                break;
            }

            let segment = tokio::select! {
                result = rx.recv() => {
                    match result {
                        Ok(seg) => seg,
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                            log::warn!("[connect] H.264 客户端落后 {} 个 segment", n);
                            continue;
                        }
                        Err(_) => break,
                    }
                }
                _ = tokio::time::sleep(Duration::from_secs(5)) => {
                    // 超时：发送心跳保持连接
                    yield Ok::<Bytes, std::convert::Infallible>(Bytes::new());
                    continue;
                }
            };

            if segment.is_empty() {
                continue;
            }

            // 长度前缀协议：4 字节大端 + segment 数据
            let len = (segment.len() as u32).to_be_bytes();
            yield Ok(Bytes::from(len.to_vec()));
            yield Ok(segment);
        }
    };

    let mut headers = HeaderMap::new();
    headers.insert("Content-Type", "application/octet-stream".parse().unwrap());
    headers.insert("Cache-Control", "no-store, no-cache".parse().unwrap());
    headers.insert("Access-Control-Allow-Origin", "*".parse().unwrap());
    headers.insert("X-Accel-Buffering", "no".parse().unwrap());

    (StatusCode::OK, headers, axum::body::Body::from_stream(stream)).into_response()
}

// ==================== 工具函数 ====================

fn validate_session(headers: &HeaderMap, state: &ServerState) -> bool {
    let sessions = state.sessions.lock().unwrap();

    // 没有已连接的 session，拒绝
    if sessions.is_empty() {
        return false;
    }

    // 检查 Authorization header
    if let Some(auth) = headers.get("authorization") {
        if let Ok(auth_str) = auth.to_str() {
            let prefix = "Bearer ";
            if auth_str.starts_with(prefix) {
                let session_id = &auth_str[prefix.len()..];
                return sessions.iter().any(|s| s.session_id == session_id);
            }
        }
    }

    false
}

fn generate_token() -> String {
    let uuid = Uuid::new_v4().to_string();
    uuid.replace('-', "")[..8].to_uppercase()
}

fn get_local_ip() -> Option<String> {
    match local_ip_address::local_ip() {
        Ok(ip) => Some(ip.to_string()),
        Err(e) => {
            log::warn!("[connect] 无法获取本机 IP: {}", e);
            None
        }
    }
}

/// 保存上传的文件到临时目录
async fn save_upload(mut multipart: Multipart) -> Result<(String, String, u64), String> {
    let cache_dir = dirs::cache_dir()
        .ok_or("无法获取缓存目录")?
        .join("SECTL")
        .join("ViewStage")
        .join("phone-uploads");

    // 目录创建放到阻塞线程
    let cache_dir_clone = cache_dir.clone();
    tokio::task::spawn_blocking(move || std::fs::create_dir_all(&cache_dir_clone))
        .await
        .map_err(|e| format!("创建目录任务失败: {}", e))?
        .map_err(|e| format!("创建目录失败: {}", e))?;

    let mut file_name = String::new();
    let mut file_data: Vec<u8> = Vec::new();

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| format!("读取字段失败: {}", e))?
    {
        let name = field.name().unwrap_or("").to_string();

        if name == "file" {
            file_name = field
                .file_name()
                .unwrap_or("upload")
                .to_string();

            let data = field
                .bytes()
                .await
                .map_err(|e| format!("读取文件数据失败: {}", e))?;

            // 限制 50MB
            if data.len() > 50 * 1024 * 1024 {
                return Err("文件超过 50MB 限制".to_string());
            }

            file_data = data.to_vec();
        }
    }

    if file_data.is_empty() {
        return Err("未收到文件数据".to_string());
    }

    // 生成唯一文件名
    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
    let ext = std::path::Path::new(&file_name)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("bin");
    let safe_name = format!("{}_{}.{}", timestamp, Uuid::new_v4().to_string()[..8].to_uppercase(), ext);

    let file_path = cache_dir.join(&safe_name);
    let file_path_clone = file_path.clone();
    let size = file_data.len() as u64;

    // 文件写入放到阻塞线程，不阻塞 Tokio worker
    tokio::task::spawn_blocking(move || std::fs::write(&file_path_clone, &file_data))
        .await
        .map_err(|e| format!("写入文件任务失败: {}", e))?
        .map_err(|e| format!("写入文件失败: {}", e))?;

    let path_str = file_path.to_string_lossy().to_string();

    Ok((path_str, file_name, size))
}
