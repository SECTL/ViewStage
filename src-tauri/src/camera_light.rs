use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use once_cell::sync::Lazy;
use tauri::Emitter;

const SEEWO_VID: u16 = 0x1FF7;
const SEEWO_PIDS: &[u16] = &[
    0x0F16, 0x0F18, 0x0F6E, 0x0F56,
    0x0F59, 0x0F63, 0x0F6F, 0x0F81,
    0x0F05, 0x0F51, 0x0F4E, 0x0F64,
];

const CMD_LIGHT_ON: [u8; 9] = [0xAA, 0xBB, 0xCC, 0x02, 0x02, 0x02, 0x00, 0x01, 0x32];
const CMD_LIGHT_OFF: [u8; 9] = [0xAA, 0xBB, 0xCC, 0x02, 0x02, 0x02, 0x00, 0x00, 0x00];
const CMD_GET_LIGHT_STATE: [u8; 4] = [0xAA, 0xBB, 0xCC, 0x02];

static APP_HANDLE: OnceLock<tauri::AppHandle> = OnceLock::new();

struct LightState {
    is_on: bool,
    level: u8,
}

struct Monitor {
    running: Arc<AtomicBool>,
    state: Arc<Mutex<LightState>>,
    pid: u16,
    _thread: thread::JoinHandle<()>,
}

static MONITOR: Lazy<Mutex<Option<Monitor>>> = Lazy::new(|| Mutex::new(None));

/// 查找并打开第一个匹配的展台 HID 设备，返回 (device, pid)
fn device_find_and_open() -> Result<(hidapi::HidDevice, u16), String> {
    let api = hidapi::HidApi::new().map_err(|e| format!("HID init failure: {}", e))?;
    for &pid in SEEWO_PIDS {
        if let Ok(device) = api.open(SEEWO_VID, pid) {
            return Ok((device, pid));
        }
    }
    Err("No supported camera found".to_string())
}

fn emit_light_changed(is_on: bool) {
    if let Some(app) = APP_HANDLE.get() {
        let _ = app.emit("camera-light-changed", serde_json::json!({"isOn": is_on}));
    }
}

/// 后台监控线程：持续读取 HID 报告，解析灯状态并缓存，状态变化时发射事件
fn monitor_thread_loop(
    device: hidapi::HidDevice,
    running: Arc<AtomicBool>,
    state: Arc<Mutex<LightState>>,
) {
    let mut buf = [0u8; 64];
    let result = loop {
        if !running.load(Ordering::Relaxed) {
            break Ok(());
        }
        match device.read(&mut buf) {
            Ok(n) if n >= 2 && buf[0] == 0xBB && buf[1] == 0xCC => {
                if n >= 8 {
                    match buf[2] {
                        2 | 5 => {
                            let is_on = buf[6] == 0x01;
                            let level = buf[7];
                            let changed = if let Ok(mut s) = state.lock() {
                                let changed = s.is_on != is_on || s.level != level;
                                s.is_on = is_on;
                                s.level = level;
                                changed
                            } else {
                                false
                            };
                            if changed {
                                emit_light_changed(is_on);
                            }
                        }
                        _ => {}
                    }
                }
            }
            Ok(_) => {}
            Err(e) => {
                break Err(e);
            }
        }
    };

    // 标记线程已退出，使 camera_light_start() 能检测到并重建
    running.store(false, Ordering::Relaxed);
    if result.is_err() {
        log::warn!("[camera-light] monitor thread exited due to read error");
    }
}

/// 发送 HID 命令（临时打开设备）
fn send_raw_command(cmd: &[u8]) -> Result<(), String> {
    let guard = MONITOR.lock().map_err(|e| format!("Lock failure: {}", e))?;
    let m = guard.as_ref().ok_or("Monitor not started")?;
    if !m.running.load(Ordering::Relaxed) {
        return Err("Monitor thread stopped".to_string());
    }
    let pid = m.pid;
    drop(guard);

    let api = hidapi::HidApi::new().map_err(|e| format!("HID init failure: {}", e))?;
    let device = api
        .open(SEEWO_VID, pid)
        .map_err(|e| format!("Open device for write failed: {}", e))?;
    device.write(cmd).map_err(|e| format!("Write failed: {}", e))?;
    Ok(())
}

/// 启动后台监控（自动检测设备）
fn camera_light_start() -> Result<(), String> {
    let mut guard = MONITOR.lock().map_err(|e| format!("Lock failure: {}", e))?;

    // 检查现有监控线程是否存活
    if let Some(ref m) = *guard {
        if m.running.load(Ordering::Relaxed) {
            return Ok(()); // 监控线程存活，无需重建
        }
        // 线程已死，清理旧 Monitor
        *guard = None;
    }

    let (device, pid) = device_find_and_open()?;
    let running = Arc::new(AtomicBool::new(true));
    let state = Arc::new(Mutex::new(LightState { is_on: false, level: 0 }));

    let t_running = running.clone();
    let t_state = state.clone();
    let thread = thread::Builder::new()
        .name("camera-light-monitor".into())
        .spawn(move || {
            monitor_thread_loop(device, t_running, t_state);
        })
        .map_err(|e| format!("Thread spawn failed: {}", e))?;

    *guard = Some(Monitor {
        running,
        state,
        pid,
        _thread: thread,
    });
    drop(guard);

    // 立即查询当前补光灯状态，监控线程的 read 循环会接收到响应并更新缓存
    thread::sleep(std::time::Duration::from_millis(20));
    let _ = send_raw_command(&CMD_GET_LIGHT_STATE);
    Ok(())
}

/// 初始化 AppHandle（由 lib.rs setup 调用）
pub fn camera_light_init_app(app: tauri::AppHandle) {
    let _ = APP_HANDLE.set(app);
}

/// 开灯
pub fn camera_light_set_on(app: &tauri::AppHandle) -> Result<(), String> {
    let _ = APP_HANDLE.set(app.clone());
    camera_light_start()?;
    send_raw_command(&CMD_LIGHT_ON)
}

/// 关灯
pub fn camera_light_set_off(app: &tauri::AppHandle) -> Result<(), String> {
    let _ = APP_HANDLE.set(app.clone());
    camera_light_start()?;
    send_raw_command(&CMD_LIGHT_OFF)
}

/// 获取缓存的最新灯状态（硬件通知，无需查询）
pub fn camera_light_get_state() -> Result<(bool, u8), String> {
    let guard = MONITOR.lock().map_err(|e| format!("Lock failure: {}", e))?;
    if let Some(ref m) = *guard {
        if let Ok(s) = m.state.lock() {
            return Ok((s.is_on, s.level));
        }
    }
    Err("Monitor not started".to_string())
}

/// 检测设备是否存在
pub fn camera_light_detect() -> bool {
    if let Ok(guard) = MONITOR.lock() {
        if let Some(ref m) = *guard {
            if m.running.load(Ordering::Relaxed) {
                return true;
            }
        }
    }
    if let Ok(api) = hidapi::HidApi::new() {
        SEEWO_PIDS.iter().any(|&pid| {
            api.device_list()
                .any(|d| d.vendor_id() == SEEWO_VID && d.product_id() == pid)
        })
    } else {
        false
    }
}
