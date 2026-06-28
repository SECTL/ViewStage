/**
 * phone-connect.js — 手机互联 UI 面板
 * 工具栏按钮 + 连接面板（地址显示、状态显示）
 */

const _pc_dom = {};
let _pc_visible = false;
let _pc_server_info = null;
let _pc_devices = [];

function _pc_escapeHtml(str) {
    const s = String(str);
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// ==================== 初始化 ====================

export function phone_connect_init() {
    console.log('[phone-connect] phone_connect_init 被调用');
    _pc_create_button();
    _pc_create_panel();
    _pc_bind_events();
    window.phone_connect_ui_update = _pc_handle_update;
    console.log('[phone-connect] 初始化完成，phone_connect_ui_update 已注册');
}

function _pc_create_button() {
    // 监听菜单弹出，添加手机互联菜单项
    const observer = new MutationObserver((mutations) => {
        for (const mutation of mutations) {
            for (const node of mutation.addedNodes) {
                if (node.id === 'menuPopup' && node.classList.contains('menu-popup')) {
                    _pc_add_menu_item(node);
                }
            }
        }
    });
    observer.observe(document.body, { childList: true, subtree: true });
    _pc_dom.observer = observer;
}

function _pc_check_status() {
    const loading = document.getElementById('pcLoading');
    const urlSection = document.getElementById('pcUrlSection');
    const notRunning = document.getElementById('pcNotRunning');

    if (!window.__TAURI__) return;

    window.__TAURI__.core.invoke('phone_server_status').then(result => {
        if (result) {
            if (loading) loading.style.display = 'none';
            if (notRunning) notRunning.style.display = 'none';
            if (urlSection) urlSection.style.display = 'flex';
            _pc_show_server_ready(result);
        } else {
            if (loading) loading.style.display = 'none';
            if (urlSection) urlSection.style.display = 'none';
            if (notRunning) notRunning.style.display = 'flex';
        }
    }).catch(() => {
        if (loading) loading.style.display = 'none';
        if (urlSection) urlSection.style.display = 'none';
        if (notRunning) notRunning.style.display = 'flex';
    });
}

function _pc_add_menu_item(menuPopup) {
    const closeBtn = document.getElementById('menuClose');
    if (!closeBtn) return;

    // 检查服务是否已启用，未启用则不显示菜单项
    if (!window.__TAURI__) return;
    window.__TAURI__.core.invoke('phone_server_status').then(result => {
        if (!result) return; // 服务未运行，不添加菜单项

        const btn = document.createElement('button');
        btn.className = 'menu-item';
        btn.id = 'menuPhoneConnect';
        btn.innerHTML = `
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="5" y="2" width="14" height="20" rx="2" ry="2"/>
                <line x1="12" y1="18" x2="12.01" y2="18"/>
            </svg>
            手机互联
        `;
        btn.addEventListener('click', () => {
            const menuPopup = document.getElementById('menuPopup');
            if (menuPopup) menuPopup.remove();
            _pc_visible = true;
            _pc_dom.panel.style.display = 'flex';
        });
        closeBtn.parentNode.insertBefore(btn, closeBtn);
    }).catch(() => {});
}

function _pc_create_panel() {
    const panel = document.createElement('div');
    panel.className = 'phone-connect-panel';
    panel.id = 'phoneConnectPanel';
    panel.style.display = 'none';
    panel.innerHTML = `
        <div class="pc-header">
            <span class="pc-title">手机互联</span>
            <button class="pc-close-btn" id="pcCloseBtn">&times;</button>
        </div>
        <div class="pc-body">
            <div class="pc-loading" id="pcLoading">
                <div class="pc-spinner"></div>
                <div class="pc-loading-text">正在启动服务...</div>
            </div>
            <div class="pc-not-running" id="pcNotRunning" style="display:none">
                <div class="pc-not-running-text">服务未启动</div>
                <button class="pc-start-btn" id="pcStartBtn">开启服务</button>
            </div>
            <div class="pc-url-section" id="pcUrlSection" style="display:none">
                <div class="pc-ready-status" id="pcReadyStatus">
                    <span class="pc-ready-dot"></span>
                    <span class="pc-ready-text">服务已就绪，等待手机连接</span>
                </div>
                <button class="pc-toggle-info-btn" id="pcToggleInfoBtn">显示连接信息</button>
                <div class="pc-url-info" id="pcUrlInfo" style="display:none">
                    <div class="pc-url-label">连接地址</div>
                    <div class="pc-url-value" id="pcUrlValue"></div>
                    <div class="pc-token-label">验证码</div>
                    <div class="pc-token-value" id="pcTokenValue"></div>
                </div>
            </div>
            <div class="pc-devices-section" id="pcDevicesSection" style="display:none">
                <div class="pc-devices-title">已连接设备</div>
                <div class="pc-devices-list" id="pcDevicesList"></div>
            </div>
        </div>
    `;
    document.querySelector('.container')?.appendChild(panel);
    _pc_dom.panel = panel;
}

function _pc_bind_events() {
    _pc_dom.btn?.addEventListener('click', () => {
        _pc_visible = !_pc_visible;
        _pc_dom.panel.style.display = _pc_visible ? 'flex' : 'none';
        if (_pc_visible) {
            _pc_check_status();
        }
    });

    document.getElementById('pcCloseBtn')?.addEventListener('click', () => {
        _pc_visible = false;
        _pc_dom.panel.style.display = 'none';
    });

    document.getElementById('pcToggleInfoBtn')?.addEventListener('click', () => {
        const info = document.getElementById('pcUrlInfo');
        const btn = document.getElementById('pcToggleInfoBtn');
        if (!info || !btn) return;
        const hidden = info.style.display === 'none';
        info.style.display = hidden ? 'block' : 'none';
        btn.textContent = hidden ? '隐藏连接信息' : '显示连接信息';
    });

    document.getElementById('pcStartBtn')?.addEventListener('click', async () => {
        const btn = document.getElementById('pcStartBtn');
        const notRunning = document.getElementById('pcNotRunning');
        const loading = document.getElementById('pcLoading');
        if (btn) btn.disabled = true;
        if (btn) btn.textContent = '启动中...';
        try {
            await window.__TAURI__.core.invoke('phone_server_start');
            if (notRunning) notRunning.style.display = 'none';
            if (loading) loading.style.display = 'flex';
            // 保存设置，下次启动时自动恢复
            await window.__TAURI__.core.invoke('settings_save_all', { settings: { phoneServerEnabled: true } });
        } catch (e) {
            console.error('[phone] 启动服务失败:', e);
            if (btn) btn.disabled = false;
            if (btn) btn.textContent = '开启服务';
        }
    });

    // 标题栏拖动
    _pc_setup_drag();
}

function _pc_setup_drag() {
    const panel = _pc_dom.panel;
    const header = panel?.querySelector('.pc-header');
    if (!panel || !header) return;

    let isDragging = false;
    let startX = 0, startY = 0;
    let origLeft = 0, origTop = 0;

    header.style.cursor = 'move';

    header.addEventListener('mousedown', (e) => {
        if (e.target.closest('.pc-close-btn')) return;
        isDragging = true;
        startX = e.clientX;
        startY = e.clientY;
        const rect = panel.getBoundingClientRect();
        origLeft = rect.left;
        origTop = rect.top;
        panel.style.transform = 'none';
        panel.style.left = origLeft + 'px';
        panel.style.top = origTop + 'px';
        e.preventDefault();
    });

    document.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        panel.style.left = (origLeft + e.clientX - startX) + 'px';
        panel.style.top = (origTop + e.clientY - startY) + 'px';
    });

    document.addEventListener('mouseup', () => {
        isDragging = false;
    });
}

// ==================== UI 更新 ====================

function _pc_handle_update(type, data) {
    console.log(`[phone-connect] _pc_handle_update: type=${type}`, data);
    switch (type) {
        case 'server_ready':
            _pc_server_info = data;
            _pc_show_server_ready(data);
            break;
        case 'device_connected':
            _pc_devices.push(data);
            _pc_update_devices();
            break;
        case 'device_disconnected':
            _pc_devices = _pc_devices.filter(d => d.session !== data.session);
            _pc_update_devices();
            break;
        case 'camera_started':
            _pc_update_camera_status(true, data.device_name);
            break;
        case 'camera_stopped':
            _pc_update_camera_status(false, null);
            break;
        case 'server_error':
            console.error('[phone-connect] 服务错误:', data.error);
            break;
    }
}

function _pc_show_server_ready(info) {
    console.log('[phone-connect] _pc_show_server_ready:', info);
    const loading = document.getElementById('pcLoading');
    const urlSection = document.getElementById('pcUrlSection');
    const urlValue = document.getElementById('pcUrlValue');
    const tokenValue = document.getElementById('pcTokenValue');

    if (loading) loading.style.display = 'none';
    if (urlSection) urlSection.style.display = 'flex';

    // 显示所有可用 IP（热点 + WiFi + 有线等）
    const ips = info.ips && info.ips.length > 0 ? info.ips : [info.ip];
    if (urlValue) {
        urlValue.innerHTML = ips.map(ip =>
            `<div style="margin-bottom:2px">http://${ip}:${info.port}</div>`
        ).join('');
    }
    if (tokenValue) tokenValue.textContent = info.token;
}

function _pc_update_devices() {
    const section = document.getElementById('pcDevicesSection');
    const list = document.getElementById('pcDevicesList');

    if (_pc_devices.length > 0) {
        if (section) section.style.display = 'block';
        if (list) {
            list.innerHTML = _pc_devices.map(d =>
                `<div class="pc-device-item">
                    <span class="pc-device-icon">📱</span>
                    <span class="pc-device-name">${_pc_escapeHtml(d.device_name)}</span>
                </div>`
            ).join('');
        }
    } else {
        if (section) section.style.display = 'none';
    }
}

function _pc_update_camera_status(active, deviceName) {
    const section = document.getElementById('pcDevicesSection');
    const list = document.getElementById('pcDevicesList');

    // 更新状态对象
    if (window.state) {
        window.state.phoneCameraActive = active;
    }

    // 更新UI显示
    if (active && deviceName) {
        if (section) section.style.display = 'block';
        if (list) {
            // 检查是否已经有摄像头状态显示
            let cameraItem = document.getElementById('pcCameraStatus');
            if (!cameraItem) {
                cameraItem = document.createElement('div');
                cameraItem.id = 'pcCameraStatus';
                cameraItem.className = 'pc-device-item pc-camera-active';
                list.insertBefore(cameraItem, list.firstChild);
            }
            cameraItem.innerHTML = `
                <span class="pc-device-icon">📹</span>
                <span class="pc-device-name">${_pc_escapeHtml(deviceName)} (摄像头)</span>
            `;
        }
    } else {
        // 移除摄像头状态显示
        const cameraItem = document.getElementById('pcCameraStatus');
        if (cameraItem) {
            cameraItem.remove();
        }
        
        // 如果没有其他设备，隐藏设备区域
        if (_pc_devices.length === 0 && !active) {
            if (section) section.style.display = 'none';
        }
    }
}

// ==================== 公开 API ====================

window.phone_connect_init = phone_connect_init;

// 自动初始化（等待 DOM 就绪）
console.log('[phone-connect] phone-connect.js 模块加载完成');
if (document.readyState === 'loading') {
    console.log('[phone-connect] DOM 仍在加载，等待 DOMContentLoaded');
    document.addEventListener('DOMContentLoaded', () => {
        console.log('[phone-connect] DOMContentLoaded 触发，调用 phone_connect_init');
        phone_connect_init();
    });
} else {
    console.log('[phone-connect] DOM 已加载，立即调用 phone_connect_init');
    phone_connect_init();
}
