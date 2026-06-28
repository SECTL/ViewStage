/**
 * ViewStage 开发者选项
 * 通过关于页面图标点击5次打开
 */

async function developer_options_init() {
    // 从后端加载已保存的值（设置面板无 DRAW_CONFIG，必须读后端）
    const invoke = window.__TAURI__?.core?.invoke;
    let savedWidthRatio = 0.4;
    let savedMaxScale = 4;
    let savedPerfMonitor = false;
    let savedPerfInterval = 200;
    let savedDevMode = true;
    let savedFrameDelta = 60;
    let savedTailDuration = 30;
    let savedEllipseStroke = false;

    if (invoke) {
        try {
            const result = await invoke('settings_fetch_all');
            const s = result?.settings || {};
            // 优先使用 DRAW_CONFIG（主窗口），否则用后端保存值，最后用硬编码默认
            savedWidthRatio = window.DRAW_CONFIG?.penMinWidthRatio
                ?? s.penMinWidthRatio
                ?? 0.4;
            savedMaxScale = window.DRAW_CONFIG?.maxScaleImage
                ?? s.maxScaleImage
                ?? 4;
            savedPerfMonitor = s.perfMonitorEnabled === true;
            savedPerfInterval = s.perfMonitorInterval ?? 200;
            savedDevMode = s.developerMode !== false;
            savedFrameDelta = window.DRAW_CONFIG?.gestureFrameDelta
                ?? s.gestureFrameDelta
                ?? 60;
            savedTailDuration = window.DRAW_CONFIG?.penTailDuration
                ?? s.penTailDuration
                ?? 30;
            savedEllipseStroke = s.ellipseStrokeEnabled === true;
        } catch (_) {
            savedWidthRatio = window.DRAW_CONFIG?.penMinWidthRatio ?? 0.4;
            savedMaxScale = window.DRAW_CONFIG?.maxScaleImage ?? 4;
        }
    } else {
        savedWidthRatio = window.DRAW_CONFIG?.penMinWidthRatio ?? 0.4;
        savedMaxScale = window.DRAW_CONFIG?.maxScaleImage ?? 4;
    }

    developer_options_show_main(savedWidthRatio, savedMaxScale, savedPerfMonitor, savedPerfInterval, savedDevMode, savedFrameDelta, savedTailDuration, savedEllipseStroke);
}

const PERF_INTERVAL_OPTIONS = [
    { value: '100', i18nKey: 'developer.perfIntervalFast' },
    { value: '200', i18nKey: 'developer.perfIntervalNormal' },
    { value: '500', i18nKey: 'developer.perfIntervalSlow' },
];

function _tk(key) { return window.i18n?.format_translate(key) ?? key; }
function _perf_interval_label(ms) {
    const opt = PERF_INTERVAL_OPTIONS.find(p => parseInt(p.value) === ms);
    return opt ? `${_tk(opt.i18nKey)}（${ms}ms）` : `${ms}ms`;
}

function developer_options_show_main(currentWidthRatio, currentMaxScale, perfMonitorEnabled, perfMonitorInterval, devModeEnabled, currentFrameDelta, currentTailDuration, ellipseStrokeEnabled) {
    const invoke = window.__TAURI__?.core?.invoke;
    const page = document.getElementById('pageDevOptions');
    if (!page) return;
    const devModeOn = devModeEnabled !== false;

    const _presetDefault = _tk('developer.presetDefault');
    const _presetDisabled = _tk('developer.presetDisabled');
    const _displayDefault = _tk('developer.displayDefault');

    const widthPresets = [
        { value: '0.05', label: '0.05' },
        { value: '0.1', label: '0.10' },
        { value: '0.15', label: '0.15' },
        { value: '0.2', label: '0.20' },
        { value: '0.25', label: '0.25' },
        { value: '0.3', label: '0.30' },
        { value: '0.4', label: `0.40${_presetDefault}` },
        { value: '0.5', label: '0.50' },
        { value: '0.75', label: '0.75' },
        { value: '1', label: '1.00' },
    ];
    const currentWidthLabel = widthPresets.find(p => parseFloat(p.value) === currentWidthRatio)?.label
        || currentWidthRatio.toFixed(2);

    const scalePresets = [
        { value: '2', label: '2x' },
        { value: '3', label: '3x' },
        { value: '4', label: `4x${_presetDefault}` },
        { value: '5', label: '5x' },
        { value: '6', label: '6x' },
        { value: '8', label: '8x' },
        { value: '10', label: '10x' },
    ];
    const currentScaleLabel = scalePresets.find(p => parseInt(p.value) === currentMaxScale)?.label
        || `${currentMaxScale}x`;

    const frameDeltaPresets = [
        { value: '60', label: `60px${_presetDefault}` },
        { value: '100', label: '100px' },
        { value: '200', label: '200px' },
        { value: '500', label: '500px' },
        { value: '1000', label: '1000px' },
    ];
    const currentFrameDeltaLabel = frameDeltaPresets.find(p => parseInt(p.value) === currentFrameDelta)?.label
        || `${currentFrameDelta}px`;

    const currentTailDurationVal = currentTailDuration ?? 30;

    page.innerHTML = `
        <h2 class="page-title">${_tk('developer.title')}</h2>
        <div class="setting-item" style="border-bottom-color:var(--color-hairline, rgba(255,255,255,0.08));">
            <span class="setting-label">${_tk('developer.devMode')}</span>
            <label class="toggle-switch">
                <input type="checkbox" id="devModeToggle"${devModeOn ? ' checked' : ''}>
                <span class="toggle-slider"></span>
            </label>
        </div>
        <div class="setting-item">
            <span class="setting-label">${_tk('developer.perfMonitor')}</span>
            <label class="toggle-switch">
                <input type="checkbox" id="devPerfMonitorToggle"${perfMonitorEnabled ? ' checked' : ''}>
                <span class="toggle-slider"></span>
            </label>
        </div>
        <div class="setting-item">
            <span class="setting-label">${_tk('developer.perfInterval')}</span>
            <div class="custom-select" id="devPerfIntervalSelect">
                <div class="select-selected" id="devPerfIntervalSelected">${_perf_interval_label(perfMonitorInterval)}</div>
                <div class="select-options" id="devPerfIntervalOptions">
                    ${PERF_INTERVAL_OPTIONS.map(p => `
                        <div class="select-option${parseInt(p.value) === perfMonitorInterval ? ' selected' : ''}" data-value="${p.value}">${_tk(p.i18nKey)}（${p.value}ms）</div>
                    `).join('')}
                </div>
            </div>
        </div>
        <div class="setting-item">
            <span class="setting-label">${_tk('developer.widthRatio')}</span>
            <div class="custom-select" id="devWidthRatioSelect">
                <div class="select-selected" id="devWidthRatioSelected">${currentWidthLabel}</div>
                <div class="select-options" id="devWidthRatioOptions">
                    ${widthPresets.map(p => `
                        <div class="select-option${parseFloat(p.value) === currentWidthRatio ? ' selected' : ''}" data-value="${p.value}">${p.label}</div>
                    `).join('')}
                </div>
            </div>
        </div>
        <div class="setting-item">
            <span class="setting-label">${_tk('developer.maxScale')}</span>
            <div class="custom-select" id="devMaxScaleSelect">
                <div class="select-selected" id="devMaxScaleSelected">${currentScaleLabel}</div>
                <div class="select-options" id="devMaxScaleOptions">
                    ${scalePresets.map(p => `
                        <div class="select-option${parseInt(p.value) === currentMaxScale ? ' selected' : ''}" data-value="${p.value}">${p.label}</div>
                    `).join('')}
                </div>
            </div>
        </div>
        <div class="setting-item">
            <span class="setting-label">${_tk('developer.frameDelta')}</span>
            <div class="custom-select" id="devFrameDeltaSelect">
                <div class="select-selected" id="devFrameDeltaSelected">${currentFrameDeltaLabel}</div>
                <div class="select-options" id="devFrameDeltaOptions">
                    ${frameDeltaPresets.map(p => `
                        <div class="select-option${parseInt(p.value) === currentFrameDelta ? ' selected' : ''}" data-value="${p.value}">${p.label}</div>
                    `).join('')}
                </div>
            </div>
        </div>
        <div class="setting-item">
            <span class="setting-label">${_tk('developer.tailDuration')}</span>
            <div style="display:flex;align-items:center;gap:6px;">
                <input type="number" id="devTailDurationInput" class="dev-number-input" value="${currentTailDurationVal}" min="0" max="500" step="1">
                <span style="font-size:12px;color:var(--color-muted, #888);">ms</span>
            </div>
        </div>
        <div class="setting-item">
            <span class="setting-label">${_tk('memclean.title')}</span>
            <span id="devGoMemclean" style="cursor:pointer;font-size:18px;color:var(--color-muted, #888);padding:4px;">→</span>
        </div>
        <div class="setting-item">
            <span class="setting-label">${_tk('developer.ellipseStroke')}<span class="experimental-badge">${_tk('developer.experimental')}</span></span>
            <label class="toggle-switch">
                <input type="checkbox" id="devEllipseStrokeToggle"${ellipseStrokeEnabled ? ' checked' : ''}>
                <span class="toggle-slider"></span>
            </label>
        </div>
        <div class="setting-item">
            <span class="setting-label">${_tk('developer.phoneConnect')}<span class="experimental-badge">${_tk('developer.experimental')}</span></span>
            <label class="toggle-switch">
                <input type="checkbox" id="devPhoneConnectToggle">
                <span class="toggle-slider"></span>
            </label>
        </div>
        <div class="setting-item" style="margin-top:-8px;opacity:0.6">
            <span class="setting-label" style="font-size:11px">切换后需重启应用生效</span>
        </div>
        <div class="setting-item">
            <span class="setting-label">${_tk('developer.docDetection')}</span>
            <span id="devGoDetection" style="cursor:pointer;font-size:18px;color:var(--color-muted, #888);padding:4px;">→</span>
        </div>
    `;

    document.getElementById('devGoDetection')?.addEventListener('click', developer_options_show_detection);
    document.getElementById('devGoMemclean')?.addEventListener('click', developer_options_show_memclean);

    // 初始化下拉框（复用 settings.js 的统一逻辑）
    if (typeof window.settings_init_all_selects === 'function') {
        window.settings_init_all_selects();
    }

    // 开发者模式开关
    (function setup_dev_mode_toggle() {
        const toggle = document.getElementById('devModeToggle');
        if (!toggle) return;
        toggle.addEventListener('change', () => {
            const enabled = toggle.checked;
            if (invoke) {
                invoke('settings_save_all', { settings: { developerMode: enabled } });
            }
            /* 同步隐藏/显示侧边栏按钮 */
            const devBtn = document.getElementById('btnDevOptions');
            if (devBtn) devBtn.style.display = enabled ? '' : 'none';
        });
    })();

    // 性能监视器开关
    (function setup_perf_monitor_toggle() {
        const toggle = document.getElementById('devPerfMonitorToggle');
        if (!toggle) return;
        toggle.addEventListener('change', () => {
            const enabled = toggle.checked;
            if (invoke) {
                invoke('settings_save_all', { settings: { perfMonitorEnabled: enabled, developerMode: true } });
            }
        });
    })();

    // 监视器更新频率选择器
    (function setup_perf_interval_select() {
        const select = document.getElementById('devPerfIntervalSelect');
        const selected = document.getElementById('devPerfIntervalSelected');
        const options = document.querySelectorAll('#devPerfIntervalOptions .select-option');

        if (!select || !selected) return;

        options.forEach(opt => {
            opt.addEventListener('click', (e) => {
                e.stopPropagation();
                const v = parseInt(opt.dataset.value);
                selected.textContent = _perf_interval_label(v);
                options.forEach(o => o.classList.remove('selected'));
                opt.classList.add('selected');
                select.classList.remove('open');
                devClearSelectOpts(select);

                if (invoke) {
                    invoke('settings_save_all', { settings: { perfMonitorInterval: v, developerMode: true } });
                }
            });
        });
    })();

    // 宽度比例选择器
    (function setup_width_ratio_select() {
        const select = document.getElementById('devWidthRatioSelect');
        const selected = document.getElementById('devWidthRatioSelected');
        const options = document.querySelectorAll('#devWidthRatioOptions .select-option');

        if (!select || !selected) return;

        options.forEach(opt => {
            opt.addEventListener('click', (e) => {
                e.stopPropagation();
                const v = parseFloat(opt.dataset.value);
                selected.textContent = opt.textContent;
                options.forEach(o => o.classList.remove('selected'));
                opt.classList.add('selected');
                select.classList.remove('open');
                devClearSelectOpts(select);

                if (window.DRAW_CONFIG) {
                    window.DRAW_CONFIG.penMinWidthRatio = v;
                }
                if (invoke) {
                    invoke('settings_save_all', { settings: { penMinWidthRatio: v, developerMode: true } });
                }
            });
        });
    })();

    // 最大缩放选择器
    (function setup_max_scale_select() {
        const select = document.getElementById('devMaxScaleSelect');
        const selected = document.getElementById('devMaxScaleSelected');
        const options = document.querySelectorAll('#devMaxScaleOptions .select-option');

        if (!select || !selected) return;

        options.forEach(opt => {
            opt.addEventListener('click', (e) => {
                e.stopPropagation();
                const v = parseInt(opt.dataset.value);
                selected.textContent = opt.textContent;
                options.forEach(o => o.classList.remove('selected'));
                opt.classList.add('selected');
                select.classList.remove('open');
                devClearSelectOpts(select);

                if (window.DRAW_CONFIG) {
                    window.DRAW_CONFIG.maxScaleImage = v;
                }
                if (invoke) {
                    invoke('settings_save_all', { settings: { maxScaleImage: v, developerMode: true } });
                }
            });
        });
    })();

    // 单帧手势位移上限选择器
    (function setup_frame_delta_select() {
        const select = document.getElementById('devFrameDeltaSelect');
        const selected = document.getElementById('devFrameDeltaSelected');
        const options = document.querySelectorAll('#devFrameDeltaOptions .select-option');

        if (!select || !selected) return;

        options.forEach(opt => {
            opt.addEventListener('click', (e) => {
                e.stopPropagation();
                const v = parseInt(opt.dataset.value);
                selected.textContent = opt.textContent;
                options.forEach(o => o.classList.remove('selected'));
                opt.classList.add('selected');
                select.classList.remove('open');
                devClearSelectOpts(select);

                if (window.DRAW_CONFIG) {
                    window.DRAW_CONFIG.gestureFrameDelta = v;
                }
                if (invoke) {
                    invoke('settings_save_all', { settings: { gestureFrameDelta: v, developerMode: true } });
                }
            });
        });
    })();

    // 收尾时长输入框
    (function setup_tail_duration_input() {
        const input = document.getElementById('devTailDurationInput');
        if (!input) return;

        let saveTimer = null;
        input.addEventListener('input', () => {
            clearTimeout(saveTimer);
            saveTimer = setTimeout(() => {
                const v = parseInt(input.value);
                if (isNaN(v) || v < 0) return;
                const val = Math.min(500, Math.max(0, v));
                input.value = val;
                if (window.DRAW_CONFIG) {
                    window.DRAW_CONFIG.penTailDuration = val;
                }
                if (invoke) {
                    invoke('settings_save_all', { settings: { penTailDuration: val, developerMode: true } });
                }
            }, 300);
        });
    })();

    // 椭圆笔迹渲染开关
    (function setup_ellipse_stroke_toggle() {
        const toggle = document.getElementById('devEllipseStrokeToggle');
        if (!toggle) return;
        toggle.addEventListener('change', () => {
            const enabled = toggle.checked;
            if (window.DRAW_CONFIG) {
                window.DRAW_CONFIG.ellipseStrokeEnabled = enabled;
            }
            if (window.batchDrawManager) {
                window.batchDrawManager.ellipseMode = enabled;
            }
            // 同步到阅读器和黑板的独立 batch_draw 实例
            const docReader = window.__documentReaderManager;
            if (docReader?.batch_draw) {
                docReader.batch_draw.ellipseMode = enabled;
            }
            const bb = window.blackboardManager;
            if (bb?.drawing_engine?.batch_draw) {
                bb.drawing_engine.batch_draw.ellipseMode = enabled;
            }
            if (invoke) {
                invoke('settings_save_all', { settings: { ellipseStrokeEnabled: enabled } });
            }
        });
    })();

    // 手机互联开关
    (function setup_phone_connect_toggle() {
        const toggle = document.getElementById('devPhoneConnectToggle');
        if (!toggle) return;

        // 从 config 读取持久化设置，同时检查运行状态
        if (invoke) {
            Promise.all([
                invoke('settings_fetch_all'),
                invoke('phone_server_status')
            ]).then(([settingsResult, statusResult]) => {
                const settings = settingsResult?.settings || {};
                // 优先看运行状态，其次看 config
                toggle.checked = !!statusResult || !!settings.phoneServerEnabled;
            }).catch(() => {});
        }

        toggle.addEventListener('change', async () => {
            const enabled = toggle.checked;
            if (!invoke) return;
            try {
                if (enabled) {
                    await invoke('phone_server_start');
                } else {
                    await invoke('phone_server_stop');
                }
                // 保存设置，下次启动时自动恢复
                await invoke('settings_save_all', { settings: { phoneServerEnabled: enabled } });
            } catch (e) {
                console.error('[phone] 切换服务状态失败:', e);
                toggle.checked = !enabled;
            }
        });
    })();

    // 动态加载 memclean 模块（供子页面使用）
    if (typeof memclean_init !== 'function') {
        const script = document.createElement('script');
        script.src = './modules/memclean/memclean.js';
        document.body.appendChild(script);
    }
}

function developer_options_show_memclean() {
    const page = document.getElementById('pageDevOptions');
    if (!page) return;

    page.innerHTML = `
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:16px;">
            <span id="devBackFromMemclean" style="cursor:pointer;font-size:18px;color:var(--color-muted, #888);padding:4px;">${_tk('developer.backToMain')}</span>
            <h2 class="page-title" style="margin:0;">${_tk('memclean.title')}</h2>
        </div>
        <div class="memclean-status" id="memcleanStatusRow">
            <span class="memclean-status-dot inactive" id="memcleanStatusDot"></span>
            <span class="memclean-status-text" id="memcleanStatusText">${_tk('memclean.statusChecking')}</span>
        </div>
        <h3 class="memclean-section-title" style="margin-top:12px;">${_tk('memclean.regionHeader')}</h3>
        <div class="memclean-regions" id="memcleanRegions"></div>
        <hr class="memclean-divider">
        <div class="memclean-btn-row">
            <button class="btn-action" id="memcleanCleanBtn">${_tk('memclean.cleanNow')}</button>
            <div id="memcleanSetupRow">
                <button class="btn-action" id="memcleanSetupBtn">${_tk('memclean.setupTask')}</button>
            </div>
            <div id="memcleanUninstallRow" style="display:none;">
                <button class="btn-action" id="memcleanUninstallBtn" style="color:var(--color-muted,#888);border-color:rgba(128,128,128,0.2);font-size:12px;padding:6px 14px;">${_tk('memclean.uninstallTask')}</button>
            </div>
        </div>
        <div class="memclean-hint">${_tk('memclean.hint')}</div>
    `;

    document.getElementById('devBackFromMemclean')?.addEventListener('click', developer_options_init);

    // 加载并初始化 memclean 模块
    if (typeof memclean_init === 'function') {
        memclean_init();
    } else {
        const script = document.createElement('script');
        script.src = './modules/memclean/memclean.js';
        script.onload = () => { if (typeof memclean_init === 'function') memclean_init(); };
        document.body.appendChild(script);
    }
}

function developer_options_show_detection() {
    const page = document.getElementById('pageDevOptions');
    if (!page) return;

    page.innerHTML = `
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:16px;">
            <span id="devBackToMain" style="cursor:pointer;font-size:18px;color:var(--color-muted, #888);padding:4px;">${_tk('developer.backToMain')}</span>
            <h2 class="page-title" style="margin:0;">${_tk('developer.detectionTitle')}</h2>
        </div>
        <div class="setting-item">
            <span class="setting-label">${_tk('developer.detectMsOffice')}</span>
            <div style="display:flex;align-items:center;gap:8px;">
                <span id="devWordStatus" style="font-size:13px;color:var(--color-muted, #888);">${_tk('developer.notDetected')}</span>
                <button class="btn-action" data-check="word">${_tk('developer.docDetection')}</button>
            </div>
        </div>
        <div class="setting-item">
            <span class="setting-label">${_tk('developer.detectWps')}</span>
            <div style="display:flex;align-items:center;gap:8px;">
                <span id="devWpsStatus" style="font-size:13px;color:var(--color-muted, #888);">${_tk('developer.notDetected')}</span>
                <button class="btn-action" data-check="wps">${_tk('developer.docDetection')}</button>
            </div>
        </div>
        <div class="setting-item">
            <span class="setting-label">${_tk('developer.detectLibreOffice')}</span>
            <div style="display:flex;align-items:center;gap:8px;">
                <span id="devLibreStatus" style="font-size:13px;color:var(--color-muted, #888);">${_tk('developer.notDetected')}</span>
                <button class="btn-action" data-check="libreoffice">${_tk('developer.docDetection')}</button>
            </div>
        </div>
    `;

    document.getElementById('devBackToMain')?.addEventListener('click', developer_options_init);

    const statusIds = { word: 'devWordStatus', wps: 'devWpsStatus', libreoffice: 'devLibreStatus' };
    const invoke = window.__TAURI__?.core?.invoke;
    if (!invoke) return;

    document.querySelectorAll('[data-check]').forEach(btn => {
        btn.addEventListener('click', () => {
            const check = btn.dataset.check;
            const statusEl = document.getElementById(statusIds[check]);
            if (!statusEl) return;

            statusEl.textContent = _tk('developer.detecting');
            statusEl.style.color = 'var(--color-muted, #888)';
            statusEl.style.fontSize = '13px';
            statusEl.style.fontWeight = 'normal';

            invoke('office_check_runtime').then(r => r[check])
                .then(ok => {
                    statusEl.textContent = ok ? '✓' : '✗';
                    statusEl.style.color = ok ? '#2ecc71' : '#e74c3c';
                    statusEl.style.fontSize = '20px';
                    statusEl.style.fontWeight = 'bold';
                })
                .catch(() => {
                    statusEl.textContent = '✗';
                    statusEl.style.color = '#e74c3c';
                    statusEl.style.fontSize = '20px';
                    statusEl.style.fontWeight = 'bold';
                });
        });
    });
}
