/**
 * phone-events.js — 手机互联事件监听与指令分发
 * 监听 Rust connect 模块发出的 Tauri 事件，分发给对应功能模块
 */

let _phone_server_info = null;
let _phone_connected_devices = [];
let _phone_heartbeat_timers = {}; // 存储每个设备的心跳定时器

// ==================== 事件注册 ====================

export function phone_events_init() {
    console.log('[phone] phone_events_init 被调用');
    console.log('[phone] window.__TAURI__:', !!window.__TAURI__);
    console.log('[phone] window.__TAURI__.event:', !!window.__TAURI__?.event);

    if (!window.__TAURI__?.event?.listen) {
        console.error('[phone] Tauri event API 不可用，跳过事件注册');
        return;
    }

    const { listen } = window.__TAURI__.event;

    console.log('[phone] 开始注册 phone-server-ready 监听...');
    listen('phone-server-ready', (event) => {
        console.log('[phone] 收到 phone-server-ready 事件:', JSON.stringify(event.payload));
        const { ip, port, token } = event.payload;
        _phone_server_info = { ip, port, token };
        if (window.phone_connect_ui_update) {
            console.log('[phone] 调用 phone_connect_ui_update');
            window.phone_connect_ui_update('server_ready', _phone_server_info);
        } else {
            console.warn('[phone] phone_connect_ui_update 未定义');
        }
    }).then(() => {
        console.log('[phone] phone-server-ready 监听注册成功');
    }).catch(err => {
        console.error('[phone] phone-server-ready 监听注册失败:', err);
    });

    listen('phone-device-connected', (event) => {
        const { session, device_name } = event.payload;
        _phone_connected_devices.push({ session, device_name });
        console.log(`[phone] 设备已连接: ${device_name}`);
        if (window.phone_connect_ui_update) {
            window.phone_connect_ui_update('device_connected', { session, device_name });
        }
        
        // 启动心跳定时器
        _phone_start_heartbeat(session);
    });

    listen('phone-device-disconnected', (event) => {
        const { session } = event.payload;
        _phone_connected_devices = _phone_connected_devices.filter(d => d.session !== session);
        console.log(`[phone] 设备已断开: ${session}`);
        if (window.phone_connect_ui_update) {
            window.phone_connect_ui_update('device_disconnected', { session });
        }
        
        // 停止心跳定时器
        _phone_stop_heartbeat(session);
    });

    listen('phone-control', (event) => {
        const { action, params } = event.payload;
        console.log(`[phone] 遥控指令: ${action}`, params || '');
        phone_dispatch_control(action, params);
    });

    listen('phone-file-received', (event) => {
        const { path, name, size } = event.payload;
        console.log(`[phone] 文件已接收: ${name} (${size} bytes)`);
        phone_handle_file(path, name);
    });

    listen('phone-server-error', (event) => {
        console.error('[phone] 服务错误:', event.payload);
        if (window.phone_connect_ui_update) {
            window.phone_connect_ui_update('server_error', { error: event.payload });
        }
    });

    listen('phone-camera-started', (event) => {
        const { session, device_name } = event.payload;
        console.log(`[phone] 手机摄像头已启动: ${device_name}`);
        if (window.phone_connect_ui_update) {
            window.phone_connect_ui_update('camera_started', { session, device_name });
        }
    });

    listen('phone-camera-stopped', (event) => {
        console.log('[phone] 手机摄像头已停止');
        if (window.phone_connect_ui_update) {
            window.phone_connect_ui_update('camera_stopped', {});
        }
    });

    console.log('[phone] 事件监听已注册');

    // 主动查询服务状态（处理前端刷新后后端已启动的情况）
    window.__TAURI__.core.invoke('phone_server_status').then(result => {
        if (result) {
            console.log('[phone] 查询到服务已运行:', result);
            _phone_server_info = result;
            if (window.phone_connect_ui_update) {
                window.phone_connect_ui_update('server_ready', result);
            }
        } else {
            console.log('[phone] 服务未运行，等待 phone-server-ready 事件');
        }
    }).catch(err => {
        console.warn('[phone] 查询服务状态失败:', err);
    });
}

// ==================== 指令分发 ====================

function phone_dispatch_control(action, params) {
    switch (action) {
        case 'next':
            phone_doc_next();
            break;
        case 'prev':
            phone_doc_prev();
            break;
        case 'first-page':
            phone_doc_first();
            break;
        case 'last-page':
            phone_doc_last();
            break;
        case 'goto-page':
            phone_doc_goto(params?.page);
            break;
        case 'annotate':
            phone_set_draw_mode('comment');
            break;
        case 'move':
            phone_set_draw_mode('move');
            break;
        case 'eraser':
            phone_set_draw_mode('eraser');
            break;
        case 'screenshot':
            phone_take_screenshot();
            break;
        case 'zoom-in':
            phone_zoom(1.25);
            break;
        case 'zoom-out':
            phone_zoom(1 / 1.25);
            break;
        case 'zoom-reset':
            phone_zoom_reset();
            break;
        case 'toggle-blackboard':
            phone_toggle_blackboard();
            break;
        case 'toggle-camera':
            phone_toggle_camera();
            break;
        case 'mirror':
            phone_toggle_mirror();
            break;
        case 'clear-annotations':
            phone_clear_annotations();
            break;
        case 'undo':
            phone_undo();
            break;
        case 'settings':
            window.__TAURI__?.core.invoke('window_show_settings');
            break;
        default:
            console.warn(`[phone] 未知指令: ${action}`);
    }
}

// ==================== 指令实现 ====================

// --- 文档翻页（提交笔画后再翻页） ---

function phone_doc_next() {
    const dr = window.documentReaderManager;
    if (dr?.is_open) {
        dr.handle_page_nav_next?.();
    } else if (window.blackboardManager?.is_open) {
        window.blackboardManager.handle_page_nav_next?.();
    }
}

function phone_doc_prev() {
    const dr = window.documentReaderManager;
    if (dr?.is_open) {
        dr.handle_page_nav_prev?.();
    } else if (window.blackboardManager?.is_open) {
        window.blackboardManager.handle_page_nav_prev?.();
    }
}

function phone_doc_first() {
    const dr = window.documentReaderManager;
    if (dr?.is_open) {
        dr._scroll_to_page(0);
    }
}

function phone_doc_last() {
    const dr = window.documentReaderManager;
    if (dr?.is_open) {
        const last = dr.page_manager.get_page_count() - 1;
        if (last >= 0) dr._scroll_to_page(last);
    }
}

function phone_doc_goto(page) {
    const dr = window.documentReaderManager;
    if (dr?.is_open && typeof page === 'number') {
        const idx = page - 1; // 手机端传 1-based 页码
        if (idx >= 0 && idx < dr.page_manager.get_page_count()) {
            dr._scroll_to_page(idx);
        }
    }
}

// --- 画笔模式（适配阅读器） ---

function phone_set_draw_mode(mode) {
    const dr = window.documentReaderManager;
    if (dr?.is_open) {
        dr._set_draw_mode(mode);
    } else {
        window.main_update_mode?.(mode);
    }
}

// --- 缩放（适配阅读器） ---

function phone_zoom(factor) {
    const dr = window.documentReaderManager;
    if (dr?.is_open) {
        const delta = factor > 1 ? 0.15 : -0.15;
        dr._dr_zoom_by_step(delta);
    } else {
        const state = window.state;
        if (!state) return;
        const newScale = Math.max(0.1, Math.min(10, state.scale * factor));
        state.scale = newScale;
        window.main_update_canvas_transform?.();
    }
}

function phone_zoom_reset() {
    const dr = window.documentReaderManager;
    if (dr?.is_open) {
        dr.dr_scale = 1;
        dr._dr_apply_scale?.();
    } else {
        const state = window.state;
        if (!state) return;
        state.scale = 1;
        state.canvasX = 0;
        state.canvasY = 0;
        window.main_update_canvas_transform?.();
    }
}

// --- 清除批注（适配阅读器） ---

function phone_clear_annotations() {
    const dr = window.documentReaderManager;
    if (dr?.is_open) {
        dr.handle_clear?.();
    } else {
        window.main_delete_all_drawings?.();
    }
}

// --- 其他 ---

function phone_toggle_blackboard() {
    const bb = window.blackboardManager;
    if (!bb) return;
    if (bb.is_open) {
        bb.close?.();
    } else {
        bb.open?.();
    }
}

function phone_toggle_camera() {
    const phoneCam = window.phoneCameraManager;
    if (phoneCam) {
        phoneCam.togglePhoneCamera();
    }
}

function phone_toggle_mirror() {
    const state = window.state;
    if (!state) return;
    state.isMirrored = !state.isMirrored;
    window.main_update_camera_video_style?.();
    window.__TAURI__?.core.invoke('mirror_update_state', { enabled: state.isMirrored });
}

function phone_undo() {
    const dr = window.documentReaderManager;
    if (dr?.is_open) {
        dr.handle_undo?.();
    } else if (window.blackboardManager?.is_open) {
        window.blackboardManager.handle_undo?.();
    }
}

function phone_take_screenshot() {
    const dom = window.dom;
    if (!dom) return;

    try {
        const canvas = document.createElement('canvas');
        const wrapper = dom.canvasWrapper;
        const rect = wrapper.getBoundingClientRect();
        canvas.width = rect.width * window.devicePixelRatio;
        canvas.height = rect.height * window.devicePixelRatio;
        const ctx = canvas.getContext('2d');
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

        // 绘制图片层
        const img = dom.imageElement;
        if (img && img.src) {
            ctx.drawImage(img, 0, 0, rect.width, rect.height);
        }

        // 绘制摄像头视频
        const video = dom.cameraVideo;
        if (video && video.srcObject && video.readyState >= 2) {
            ctx.drawImage(video, 0, 0, rect.width, rect.height);
        }

        canvas.toBlob((blob) => {
            if (!blob) return;
            const reader = new FileReader();
            reader.onload = () => {
                const base64 = reader.result.split(',')[1];
                window.__TAURI__?.core.invoke('image_save_file', {
                    image_data: `data:image/png;base64,${base64}`,
                    prefix: 'screenshot'
                });
            };
            reader.readAsDataURL(blob);
        }, 'image/png');
    } catch (e) {
        console.error('[phone] 截图失败:', e);
    }
}

// ==================== 文件处理 ====================

function phone_handle_file(path, name) {
    const ext = name.split('.').pop()?.toLowerCase();
    const docExts = ['pdf', 'doc', 'docx'];
    const imgExts = ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'webp'];

    if (docExts.includes(ext)) {
        window.main_load_pdf_from_path?.(path, true);
    } else if (imgExts.includes(ext)) {
        const info = window.phone_server_info?.();
        if (!info) {
            console.error('[phone] 服务信息不可用，无法加载图片');
            return;
        }
        const url = `http://${info.ip}:${info.port}/file/download?path=${encodeURIComponent(path)}`;
        fetch(url)
        .then(res => {
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            return res.blob();
        })
        .then(blob => {
            const blobUrl = URL.createObjectURL(blob);
            const img = new Image();
            img.onload = () => {
                window.main_render_image_centered?.(img);
                // 添加到侧边栏图片列表
                window.main_save_image_to_list_no_highlight?.(img, name);
                // 释放原始 blob URL（save_image 会创建自己的 blob URL）
                URL.revokeObjectURL(blobUrl);
            };
            img.onerror = () => {
                URL.revokeObjectURL(blobUrl);
            };
            img.src = blobUrl;
        })
        .catch(err => {
            console.error('[phone] 加载图片失败:', err);
        });
    } else {
        console.warn(`[phone] 不支持的文件类型: ${ext}`);
    }
}

// ==================== 心跳管理 ====================

function _phone_start_heartbeat(session) {
    // 如果已有定时器，先停止
    _phone_stop_heartbeat(session);
    
    let failCount = 0;
    const MAX_FAILS = 3; // 连续失败 3 次才判定离线
    
    // 每30秒发送一次心跳
    const timer = setInterval(async () => {
        try {
            if (!_phone_server_info) {
                console.warn('[phone] 服务信息不可用，停止心跳');
                _phone_stop_heartbeat(session);
                _phone_emit_disconnect(session);
                return;
            }
            
            const url = `http://${_phone_server_info.ip}:${_phone_server_info.port}/heartbeat`;
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${session}`,
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                failCount++;
                console.warn(`[phone] 心跳请求失败 (${failCount}/${MAX_FAILS}): ${response.status}`);
                if (failCount >= MAX_FAILS) {
                    _phone_stop_heartbeat(session);
                    _phone_emit_disconnect(session);
                }
            } else {
                failCount = 0; // 成功则重置
            }
        } catch (error) {
            failCount++;
            console.error(`[phone] 心跳请求错误 (${failCount}/${MAX_FAILS}):`, error);
            if (failCount >= MAX_FAILS) {
                _phone_stop_heartbeat(session);
                _phone_emit_disconnect(session);
            }
        }
    }, 30000); // 30秒
    
    _phone_heartbeat_timers[session] = timer;
    console.log(`[phone] 启动心跳定时器: ${session}`);
}

function _phone_stop_heartbeat(session) {
    if (_phone_heartbeat_timers[session]) {
        clearInterval(_phone_heartbeat_timers[session]);
        delete _phone_heartbeat_timers[session];
        console.log(`[phone] 停止心跳定时器: ${session}`);
    }
}

function _phone_emit_disconnect(session) {
    const device = _phone_connected_devices.find(d => d.session === session);
    if (device) {
        _phone_connected_devices = _phone_connected_devices.filter(d => d.session !== session);
        console.log(`[phone] 心跳失败，设备离线: ${device.device_name} (${session})`);
        if (window.phone_connect_ui_update) {
            window.phone_connect_ui_update('device_disconnected', { session, device_name: device.device_name });
        }
    }
}

// ==================== 公开 API ====================

window.phone_events_init = phone_events_init;
window.phone_server_info = () => _phone_server_info;
window.phone_connected_devices = () => _phone_connected_devices;

// 自动初始化
console.log('[phone] phone-events.js 模块加载完成');
if (window.__TAURI__) {
    console.log('[phone] 检测到 Tauri 环境，自动初始化');
    phone_events_init();
} else {
    console.warn('[phone] 未检测到 Tauri 环境，跳过自动初始化');
}
