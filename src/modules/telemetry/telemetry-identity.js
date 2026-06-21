/**
 * 设备身份管理：UUID 生成/读取、device_type 判定
 * UUID 由 Rust 后端持久化到 identity.json（%APPDATA%/SECTL/ViewStage/identity.json），
 * 设备重置后保持不变。
 */

/**
 * 读取或生成本机设备 UUID，由 Rust 后端持久化
 */
export async function getInstallUUID() {
    try {
        const invoke = window.__TAURI__?.core?.invoke;
        if (invoke) {
            const uuid = await invoke('get_device_uuid');
            if (uuid) return uuid;
        }
    } catch (e) {
        console.warn('[telemetry] failed to get device UUID from backend, using fallback:', e);
    }
    // fallback: localStorage
    try {
        let id = localStorage.getItem('viewstage_install_id');
        if (!id) {
            id = crypto.randomUUID();
            localStorage.setItem('viewstage_install_id', id);
        }
        return id;
    } catch (e) {
        console.warn('[telemetry] localStorage unavailable, using temp UUID');
        return crypto.randomUUID();
    }
}

/**
 * 根据 navigator.platform 返回 device_type 枚举值
 */
export function getDeviceType() {
    const platform = navigator.platform || '';
    if (platform.includes('Win')) {
        return 'windows-desktop';
    }
    if (platform.includes('Linux')) {
        return 'linux-desktop';
    }
    if (platform.includes('Mac')) {
        return 'macos-desktop';
    }
    return 'unknown-desktop';
}
