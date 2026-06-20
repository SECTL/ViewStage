use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use once_cell::sync::Lazy;

const SEEWO_VID: u16 = 0x1FF7;
const SEEWO_PIDS: &[u16] = &[
    0x0F16, 0x0F18, 0x0F6E, 0x0F56,
    0x0F59, 0x0F63, 0x0F6F, 0x0F81,
    0x0F05, 0x0F51, 0x0F4E, 0x0F64,
];

const CMD_LIGHT_ON: [u8; 9] = [0xAA, 0xBB, 0xCC, 0x02, 0x02, 0x02, 0x00, 0x01, 0x32];
const CMD_LIGHT_OFF: [u8; 9] = [0xAA, 0xBB, 0xCC, 0x02, 0x02, 0x02, 0x00, 0x00, 0x00];

struct LightState {
    is_on: bool,
    level: u8,
}

#[allow(dead_code)]
struct Monitor {
    running: Arc<AtomicBool>,
    state: Arc<Mutex<LightState>>,
    pid: u16,
    _thread: thread::JoinHandle<()>,
}

static MONITOR: Lazy<Mutex<Option<Monitor>>> = Lazy::new(|| Mutex::new(None));

/// 查找并打开第一个匹配的希沃展台 HID 设备，返回 (device, pid)
fn device_find_and_open() -> Result<(hidapi::HidDevice, u16), String> {
    let api = hidapi::HidApi::new().map_err(|e| format!("HID init failure: {}", e))?;
    for &pid in SEEWO_PIDS {
        if let Ok(device) = api.open(SEEWO_VID, pid) {
            return Ok((device, pid));
        }
    }
    Err("No supported camera found".to_string())
}

/// 后台监控线程：持续读取 HID 报告，解析灯状态并缓存
fn monitor_thread_loop(
    device: hidapi::HidDevice,
    running: Arc<AtomicBool>,
    state: Arc<Mutex<LightState>>,
) {
    let mut buf = [0u8; 64];
    while running.load(Ordering::Relaxed) {
        match device.read(&mut buf) {
            Ok(n) if n >= 2 && buf[0] == 0xBB && buf[1] == 0xCC => {
                if n >= 8 {
                    match buf[2] {
                        2 | 5 => {
                            let is_on = buf[6] == 0x01;
                            let level = buf[7];
                            if let Ok(mut s) = state.lock() {
                                s.is_on = is_on;
                                s.level = level;
                            }
                        }
                        _ => {}
                    }
                }
            }
            Ok(_) => {}
            Err(_) => {
                break;
            }
        }
    }
}

/// 发送 HID 命令（临时打开设备）
fn send_raw_command(cmd: &[u8]) -> Result<(), String> {
    let guard = MONITOR.lock().map_err(|e| format!("Lock failure: {}", e))?;
    let pid = guard.as_ref().ok_or("Monitor not started")?.pid;
    drop(guard);

    let api = hidapi::HidApi::new().map_err(|e| format!("HID init failure: {}", e))?;
    let device = api
        .open(SEEWO_VID, pid)
        .map_err(|e| format!("Open device for write failed: {}", e))?;
    device.write(cmd).map_err(|e| format!("Write failed: {}", e))?;
    Ok(())
}

/// 启动后台监控（自动检测设备）
pub fn camera_light_start() -> Result<(), String> {
    let mut guard = MONITOR.lock().map_err(|e| format!("Lock failure: {}", e))?;
    if guard.is_some() {
        return Ok(());
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
    Ok(())
}

/// 停止后台监控
#[allow(dead_code)]
pub fn camera_light_stop() {
    if let Ok(mut guard) = MONITOR.lock() {
        if let Some(m) = guard.take() {
            m.running.store(false, Ordering::Relaxed);
            // 线程会在下次 read 出错或检查 running 后退出
        }
    }
}

/// 开灯
pub fn camera_light_set_on() -> Result<(), String> {
    camera_light_start()?;
    send_raw_command(&CMD_LIGHT_ON)
}

/// 关灯
pub fn camera_light_set_off() -> Result<(), String> {
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
        if guard.is_some() {
            return true;
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
