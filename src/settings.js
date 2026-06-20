/**
 * ViewStage 设置窗口脚本
 * 
 * 功能模块：
 * - 应用设置：语言、主题、启动选项
 * - Canvas调节：画布尺寸、帧率
 * - 信号源调节：摄像头选择、分辨率、镜像
 * - 关于：版本信息、检查更新
 */

import { checkForUpdate, startDownload, installDownload, onProgress, offProgress } from './modules/update/update.js';

document.addEventListener('DOMContentLoaded', async () => {
    await i18n.init_start();

    // Linux平台隐藏内存清理相关UI
    try {
        const { invoke } = window.__TAURI__.core;
        const platform = await invoke('app_fetch_platform');
        if (platform === 'linux') {
            const el = document.getElementById('memreductCleanItem');
            if (el) el.style.display = 'none';
        }
    } catch (_) {}
    
    // ==================== 自定义弹窗函数 ====================
    function settings_show_dialog(title, message, type = 'info') {
        const existing = document.getElementById('settingsDialog');
        if (existing) existing.remove();
        
        const dialog = document.createElement('div');
        dialog.id = 'settingsDialog';
        dialog.className = 'settings-dialog-overlay';
        
        const icon = type === 'error' ? '❌' : type === 'success' ? '✅' : 'ℹ️';
        
        dialog.innerHTML = `
            <div class="settings-dialog">
                <div class="settings-dialog-icon">${icon}</div>
                <div class="settings-dialog-title">${title}</div>
                <div class="settings-dialog-message">${message}</div>
                <button class="settings-dialog-btn" id="settingsDialogClose">${window.i18n?.format_translate('common.confirm') || '确定'}</button>
            </div>
        `;
        document.body.appendChild(dialog);
        
        const closeBtn = document.getElementById('settingsDialogClose');
        closeBtn?.addEventListener('click', () => dialog.remove());
        dialog.addEventListener('click', (e) => {
            if (e.target === dialog) dialog.remove();
        });
    }
    
    /**
     * 显示自定义确认弹窗（确定/取消）
     * @param {string} title - 标题
     * @param {string} message - 消息内容
     * @returns {Promise<boolean>} true=确认, false=取消
     */
    function settings_show_confirm(title, message) {
        return new Promise((resolve) => {
            const existing = document.querySelector('.modal-overlay.confirm-dialog');
            if (existing) existing.remove();
            
            const overlay = document.createElement('div');
            overlay.className = 'modal-overlay confirm-dialog';
            overlay.innerHTML = `
                <div class="modal-dialog">
                    <div class="modal-title">${title}</div>
                    <div class="modal-message">${message}</div>
                    <div class="modal-buttons">
                        <button class="modal-btn modal-btn-cancel" id="confirmCancel">${window.i18n?.format_translate('common.cancel') || '取消'}</button>
                        <button class="modal-btn modal-btn-confirm" id="confirmOk">${window.i18n?.format_translate('common.confirm') || '确认'}</button>
                    </div>
                </div>
            `;
            document.body.appendChild(overlay);
            requestAnimationFrame(() => overlay.classList.add('active'));
            
            const cleanup = (result) => {
                overlay.classList.remove('active');
                setTimeout(() => overlay.remove(), 300);
                resolve(result);
            };
            
            overlay.querySelector('#confirmOk').addEventListener('click', () => cleanup(true));
            overlay.querySelector('#confirmCancel').addEventListener('click', () => cleanup(false));
            overlay.addEventListener('click', (e) => {
                if (e.target === overlay) cleanup(false);
            });
        });
    }
    
    // ==================== DOM 元素引用 ====================
    const btnClose = document.getElementById('btnClose');
    const auroraBg = document.getElementById('auroraBg');
    
    // ==================== 版本信息加载 ====================
    /**
     * 加载应用版本号和版权年份
     */
    async function settings_load_version() {
        if (window.__TAURI__) {
            try {
                const { invoke } = window.__TAURI__.core;
                const version = await invoke('app_fetch_version');
                
                const versionNumber = document.getElementById('versionNumber');
                const currentVersion = document.getElementById('currentVersion');
                const latestVersion = document.getElementById('latestVersion');
                
                if (versionNumber) versionNumber.textContent = version;
                if (currentVersion) currentVersion.textContent = version;
                if (latestVersion) latestVersion.textContent = version;
            } catch (error) {
                console.error('获取版本号失败:', error);
            }
        }
        
        const copyrightYear = document.getElementById('copyrightYear');
        if (copyrightYear) {
            copyrightYear.textContent = new Date().getFullYear();
        }
    }
    
    // ==================== 摄像头设置禁用 ====================
    function settings_hide_camera() {
        const cameraSettingItems = [
            document.querySelector('#cameraSelect')?.closest('.setting-item'),
            document.querySelector('#cameraResolutionSelect')?.closest('.setting-item'),
            document.querySelector('#mirrorToggle')?.closest('.setting-item'),
        ];
        
        cameraSettingItems.forEach(item => {
            if (item) {
                item.classList.add('disabled');
            }
        });
    }
    
    // ==================== 设置加载 ====================
    /**
     * 从后端加载设置并更新UI
     */
    async function settings_load_all() {
        if (window.__TAURI__) {
            try {
                const { invoke } = window.__TAURI__.core;
                const result = await invoke('settings_fetch_all');
                const settings = (result && typeof result === 'object' && result.settings)
                    ? result.settings : {};
                
                const selectSelected = document.getElementById('selectSelected');
                const languageOptions = document.querySelectorAll('#selectOptions .select-option');
                
                if (settings.language && selectSelected) {
                    languageOptions.forEach(option => {
                        if (option.dataset.value === settings.language) {
                            selectSelected.textContent = option.textContent;
                            option.classList.add('selected');
                        } else {
                            option.classList.remove('selected');
                        }
                    });
                }
                
                const cameraSelected = document.getElementById('cameraSelected');
                const cameraOptionsContainer = document.getElementById('cameraOptions');
                const cameraResolutionSelected = document.getElementById('cameraResolutionSelected');
                
                let hasCameraPermission = false;
                let hasCamera = false;
                let cameraStream = null;
                
                try {
                    cameraStream = await navigator.mediaDevices.getUserMedia({ video: true });
                    hasCameraPermission = true;
                    cameraStream.getTracks().forEach(t => t.stop());
                } catch (e) {
                    console.log('摄像头权限检测:', e.name);
                    hasCameraPermission = false;
                }
                
                if (cameraSelected && cameraOptionsContainer) {
                    try {
                        const devices = await navigator.mediaDevices.enumerateDevices();
                        const videoDevices = devices.filter(device => device.kind === 'videoinput');
                        hasCamera = videoDevices.length > 0;
                        
                        cameraOptionsContainer.innerHTML = '';
                        
                            if (!hasCameraPermission) {
                                cameraSelected.textContent = window.i18n?.format_translate('settings.noCameraPermission') || '无摄像头权限';
                                cameraResolutionSelected.textContent = '-';
                                settings_hide_camera();
                            } else if (videoDevices.length === 0) {
                                cameraSelected.textContent = window.i18n?.format_translate('settings.noCameraDetected') || '未检测到摄像头';
                                cameraResolutionSelected.textContent = '-';
                                settings_hide_camera();
                            } else {
                                videoDevices.forEach((device, index) => {
                                    const option = document.createElement('div');
                                    option.className = 'select-option';
                                    option.dataset.value = device.deviceId;
                                    
                                    const cameraText = window.i18n?.format_translate('camera.camera') || '摄像头';
                                    let label = device.label || `${cameraText} ${index + 1}`;
                                    if (label.includes('back') || label.includes('后置') || label.includes('rear')) {
                                        label = `${window.i18n?.format_translate('camera.rearCamera') || '后置'}: ${label}`;
                                    } else if (label.includes('front') || label.includes('前置') || label.includes('user')) {
                                        label = `${window.i18n?.format_translate('camera.frontCamera') || '前置'}: ${label}`;
                                    }
                                    
                                    option.textContent = label;
                                    cameraOptionsContainer.appendChild(option);
                                });
                                
                                if (settings.defaultCamera) {
                                    const cameraOptions = cameraOptionsContainer.querySelectorAll('.select-option');
                                    let found = false;
                                    cameraOptions.forEach(option => {
                                        if (option.dataset.value === settings.defaultCamera) {
                                            cameraSelected.textContent = option.textContent;
                                            option.classList.add('selected');
                                            found = true;
                                        } else {
                                            option.classList.remove('selected');
                                        }
                                    });
                                    
                                    if (!found && cameraOptions.length > 0) {
                                        cameraSelected.textContent = cameraOptions[0].textContent;
                                        cameraOptions[0].classList.add('selected');
                                    }
                                } else {
                                    const cameraOptions = cameraOptionsContainer.querySelectorAll('.select-option');
                                    if (cameraOptions.length > 0) {
                                        cameraSelected.textContent = cameraOptions[0].textContent;
                                        cameraOptions[0].classList.add('selected');
                                    }
                                }
                        }
                    } catch (error) {
                        console.error('获取摄像头列表失败:', error);
                        cameraSelected.textContent = window.i18n?.format_translate('settings.getFailed') || '获取失败';
                        cameraResolutionSelected.textContent = '-';
                        settings_hide_camera();
                    }
                }
                
                const cameraResolutionOptionsContainer = document.getElementById('cameraResolutionOptions');
                
                if (cameraResolutionSelected && cameraResolutionOptionsContainer && hasCameraPermission && hasCamera) {
                    const selectedCameraOption = cameraOptionsContainer.querySelector('.select-option.selected');
                    const selectedCameraId = selectedCameraOption ? selectedCameraOption.dataset.value : null;
                    try {
                        await settings_update_camera_resolution_options(selectedCameraId, settings.cameraWidth, settings.cameraHeight);
                    } catch (e) {
                        console.error('初始化分辨率选择失败:', e);
                    }
                }
                
                const dprLimitSelected = document.getElementById('dprLimitSelected');
                const dprLimitOptionsContainer = document.getElementById('dprLimitOptions');

                if (dprLimitSelected && dprLimitOptionsContainer) {
                    const savedDprLimit = settings.dprLimit !== undefined ? settings.dprLimit : 2;
                    const dprLimitOptions = dprLimitOptionsContainer.querySelectorAll('.select-option');
                    dprLimitOptions.forEach(option => {
                        if (parseFloat(option.dataset.value) === savedDprLimit) {
                            dprLimitSelected.textContent = option.textContent;
                            option.classList.add('selected');
                        } else {
                            option.classList.remove('selected');
                        }
                    });
                }

                const dynamicDprEnabled = settings.dynamicDprEnabled !== undefined ? settings.dynamicDprEnabled : true;
                const dprToggle = document.getElementById('dynamicDprToggle');
                if (dprToggle) {
                    dprToggle.checked = dynamicDprEnabled;
                }

                const eraserSpeedEnabled = settings.eraserSpeedEnabled !== undefined ? settings.eraserSpeedEnabled : false;
                const eraserSpeedToggle = document.getElementById('eraserSpeedToggle');
                if (eraserSpeedToggle) {
                    eraserSpeedToggle.checked = eraserSpeedEnabled;
                }
                const eraserPresetsItem = document.getElementById('eraserPresetsItem');
                if (eraserPresetsItem) {
                    eraserPresetsItem.style.display = eraserSpeedEnabled ? 'none' : '';
                }
                const momentumEnabled = settings.momentumEnabled !== undefined ? settings.momentumEnabled : false;
                const momentumToggle = document.getElementById('momentumToggle');
                if (momentumToggle) {
                    momentumToggle.checked = momentumEnabled;
                }
                const palmEraserEnabled = settings.palmEraserEnabled !== undefined ? settings.palmEraserEnabled : (window.DRAW_CONFIG?.palmEraserEnabled ?? false);
                const palmEraserToggle = document.getElementById('palmEraserToggle');
                if (palmEraserToggle) {
                    palmEraserToggle.checked = palmEraserEnabled;
                }
                if (window.DRAW_CONFIG) {
                    window.DRAW_CONFIG.palmEraserEnabled = palmEraserEnabled;
                }
                const dprRangeItem = document.getElementById('dprRangeItem');
                if (dprRangeItem) {
                    dprRangeItem.style.display = dynamicDprEnabled ? '' : 'none';
                }
                const dprLimitItem = document.getElementById('dprLimitItem');
                if (dprLimitItem) {
                    dprLimitItem.style.display = dynamicDprEnabled ? 'none' : '';
                }

                function settings_load_dpr_select(id, key, defaultVal) {
                    const selSelected = document.getElementById(id.replace('Select', 'Selected'));
                    const selOptions = document.getElementById(id.replace('Select', 'Options'));
                    if (selSelected && selOptions) {
                        const saved = settings[key] !== undefined ? parseFloat(settings[key]) : defaultVal;
                        selOptions.querySelectorAll('.select-option').forEach(opt => {
                            if (parseFloat(opt.dataset.value) === saved) {
                                selSelected.textContent = opt.textContent;
                                opt.classList.add('selected');
                            } else {
                                opt.classList.remove('selected');
                            }
                        });
                    }
                }
                settings_load_dpr_select('dprMinSelect', 'dprMin', 1);
                settings_load_dpr_select('dprMaxSelect', 'dprMax', 4);

                const DEFAULT_COLORS = [
                    '#ef4444', '#f97316', '#eab308', '#22c55e', '#06b6d4',
                    '#3b82f6', '#6366f1', '#a855f7', '#ec4899', '#f43f5e',
                    '#14b8a6', '#64748b', '#1e293b', '#000000', '#ffffff'
                ];
                const savedColors = settings.penColors || DEFAULT_COLORS;
                
                // 颜色格式转换函数
                function settings_calc_color_to_hex(color) {
                    if (typeof color === 'string') {
                        return color;
                    }
                    if (typeof color === 'object' && color.r !== undefined) {
                        return settings_calc_rgb_to_hex(color.r, color.g, color.b);
                    }
                    return '#000000';
                }
                
                function settings_calc_rgb_to_hex(r, g, b) {
                    return '#' + [r, g, b].map(x => {
                        const hex = x.toString(16);
                        return hex.length === 1 ? '0' + hex : hex;
                    }).join('');
                }
                
                for (let i = 1; i <= 15; i++) {
                    const colorBtn = document.querySelector(`.color-edit-item[data-index="${i - 1}"] .color-edit-btn`);
                    if (colorBtn) {
                        const color = savedColors[i - 1] || DEFAULT_COLORS[i - 1];
                        const hexColor = settings_calc_color_to_hex(color);
                        colorBtn.style.backgroundColor = hexColor;
                        colorBtn.dataset.color = hexColor;
                    }
                }
                
                const DEFAULT_PRESETS = [2, 5, 10, 15, 21];
                const savedPresets = settings.penSizePresets || DEFAULT_PRESETS;
                for (let i = 0; i < 5; i++) {
                    const input = document.getElementById('penPreset' + i);
                    if (input) {
                        input.value = savedPresets[i] !== undefined ? savedPresets[i] : DEFAULT_PRESETS[i];
                    }
                }

                const DEFAULT_ERASER_PRESETS = [5, 15, 25, 38, 50];
                const savedEraserPresets = settings.eraserSizePresets || DEFAULT_ERASER_PRESETS;
                for (let i = 0; i < 5; i++) {
                    const input = document.getElementById('eraserPreset' + i);
                    if (input) {
                        input.value = savedEraserPresets[i] !== undefined ? savedEraserPresets[i] : DEFAULT_ERASER_PRESETS[i];
                    }
                }

                function settings_update_default_dots(group) {
                    const saved = group === 'pen'
                        ? (settings.penWidth !== undefined ? settings.penWidth : 5)
                        : (settings.eraserSize !== undefined ? settings.eraserSize : 15);
                    document.querySelectorAll(`.preset-default-dot[data-default="${group}"]`).forEach(dot => {
                        const idx = parseInt(dot.dataset.index);
                        const input = document.getElementById((group === 'pen' ? 'penPreset' : 'eraserPreset') + idx);
                        const val = input ? parseInt(input.value) : 0;
                        dot.classList.toggle('active', val === saved);
                    });
                }
                settings_update_default_dots('pen');
                settings_update_default_dots('eraser');

                const mirrorToggle = document.getElementById('mirrorToggle');
                if (mirrorToggle) {
                    try {
                        const isMirrored = await invoke('mirror_fetch_state');
                        mirrorToggle.checked = isMirrored;
                    } catch (error) {
                        console.error('获取镜像状态失败:', error);
                        mirrorToggle.checked = false;
                    }
                }

                const restoreLastDocToggle = document.getElementById('restoreLastDocToggle');
                if (restoreLastDocToggle) {
                    restoreLastDocToggle.checked = settings.restoreLastDoc !== false;
                }

                const frameRateModeGroup = document.getElementById('frameRateModeGroup');
                if (frameRateModeGroup) {
                    const mode = settings.frameRateMode || 'adaptive';
                    frameRateModeGroup.dataset.active = mode;
                    const buttons = frameRateModeGroup.querySelectorAll('.option-btn');
                    buttons.forEach(btn => {
                        btn.classList.toggle('active', btn.dataset.value === mode);
                    });
                    if (window.batchDrawManager) {
                        window.batchDrawManager.batch_draw_update_frame_rate(mode);
                    }
                }
                
                const penEffectModeGroup = document.getElementById('penEffectModeGroup');
                if (penEffectModeGroup) {
                    const mode = settings.penEffectMode || 'limited';
                    penEffectModeGroup.dataset.active = mode;
                    const buttons = penEffectModeGroup.querySelectorAll('.option-btn');
                    buttons.forEach(btn => {
                        btn.classList.toggle('active', btn.dataset.value === mode);
                    });
                    if (window.DRAW_CONFIG) {
                        window.DRAW_CONFIG.penEffectMode = mode;
                        if (window.realPenManager) {
                            window.realPenManager.invalidate_cache();
                        }
                    }
                }

                const docReaderZoomGroup = document.getElementById('docReaderZoomGroup');
                if (docReaderZoomGroup) {
                    const zoom = settings.docReaderDefaultZoom || 'fitWidth';
                    docReaderZoomGroup.dataset.active = zoom;
                    const buttons = docReaderZoomGroup.querySelectorAll('.option-btn');
                    buttons.forEach(btn => {
                        btn.classList.toggle('active', btn.dataset.value === zoom);
                    });
                }

                const docReaderShowMinimizeToggle = document.getElementById('docReaderShowMinimizeToggle');
                if (docReaderShowMinimizeToggle) {
                    docReaderShowMinimizeToggle.checked = settings.docReaderShowMinimize !== false;
                }
                
                // 主题设置 — 卡片模式
                const savedTheme = settings.theme || 'com.viewstage.theme.simplify';
                settings_load_user_themes(savedTheme);
                if (window.ThemeManager) {
                    window.ThemeManager.theme_update_active(savedTheme);
                }
                
                const defaultRotationSelected = document.getElementById('defaultRotationSelected');
                const defaultRotationOptionsContainer = document.getElementById('defaultRotationOptions');
                if (defaultRotationSelected && defaultRotationOptionsContainer) {
                    const savedRotation = settings.defaultRotation || 0;
                    const rotationOptions = defaultRotationOptionsContainer.querySelectorAll('.select-option');
                    rotationOptions.forEach(option => {
                        if (parseInt(option.dataset.value) === savedRotation) {
                            defaultRotationSelected.textContent = option.textContent;
                            option.classList.add('selected');
                        } else {
                            option.classList.remove('selected');
                        }
                    });
                }
                
                const autoClearCacheDays = settings.autoClearCacheDays ?? 15;
                const autoClearCacheSelected = document.getElementById('autoClearCacheSelected');
                const autoClearCacheOptions = document.getElementById('autoClearCacheOptions');
                if (autoClearCacheSelected && autoClearCacheOptions) {
                    const options = autoClearCacheOptions.querySelectorAll('.select-option');
                    options.forEach(opt => {
                        if (parseInt(opt.dataset.value) === autoClearCacheDays) {
                            autoClearCacheSelected.textContent = opt.textContent;
                        }
                    });
                }

                const blackboardEnabled = settings.blackboardEnabled !== false;
                const blackboardToggle = document.getElementById('blackboardEnabledToggle');
                if (blackboardToggle) {
                    blackboardToggle.checked = blackboardEnabled;
                }

                const telemetryEnabled = settings.telemetryEnabled !== false;
                const telemetryToggle = document.getElementById('telemetryToggle');
                if (telemetryToggle) {
                    telemetryToggle.checked = telemetryEnabled;
                }

                const blurEnabled = settings.blurEnabled === true;
                const blurToggle = document.getElementById('blurToggle');
                if (blurToggle) {
                    blurToggle.checked = blurEnabled;
                }
                document.body.classList.toggle('blur-enabled', blurEnabled);

                // 文档关联状态检测（功能检测）
                async function checkAssociation(ext, statusElId) {
                    const statusEl = document.getElementById(statusElId);
                    if (!statusEl || !window.__TAURI__) return;
                    try {
                        const { invoke } = window.__TAURI__.core;
                        const cmd = ext === 'pdf' ? 'filetype_validate_pdf_default' : 'filetype_validate_word_default';
                        const ok = await invoke(cmd);
                        statusEl.textContent = ok ? '✓' : '✗';
                        statusEl.className = 'association-status ' + (ok ? 'associated' : 'not-associated');
                        statusEl.title = ok
                            ? (window.i18n?.format_translate('settings.defaultSet') || '已设为默认')
                            : (window.i18n?.format_translate('settings.defaultNotSet') || '未关联');
                    } catch (_) {
                        statusEl.textContent = '✗';
                        statusEl.className = 'association-status not-associated';
                    }
                }
                checkAssociation('pdf', 'pdfDefaultStatus');
                checkAssociation('word', 'wordDefaultStatus');
                
                return settings;
            } catch (error) {
                console.error('加载设置失败:', error);
                return {};
            }
        }
        return {};
    }
    
    async function settings_save_all_local(settings) {
        if (window.__TAURI__) {
            try {
                const { invoke } = window.__TAURI__.core;
                const { emit } = window.__TAURI__.event;
                await invoke('settings_save_all', { settings });
                
                await emit('settings-changed', settings);
                
                return true;
            } catch (error) {
                console.error('保存设置失败:', error);
                return false;
            }
        }
        return false;
    }

    function settings_load_user_themes(savedTheme) {
        if (!window.__TAURI__) return;

        const { invoke } = window.__TAURI__.core;

        invoke('theme_list_user').then(themes => {
            const builtinSuffix = window.i18n?.format_translate('settings.themeBuiltinSuffix') || '（内置主题）';
            const allThemes = [
                {
                    name: 'com.viewstage.theme.dark',
                    display_name: (window.i18n?.format_translate('settings.themeDark') || '深色') + builtinSuffix,
                    canvas_bg: '#1a1a1a',
                    text_color: '#ffffff'
                },
                {
                    name: 'com.viewstage.theme.simplify',
                    display_name: (window.i18n?.format_translate('settings.themeSimplify') || '浅色') + builtinSuffix,
                    canvas_bg: '#ffffff',
                    text_color: '#1a1a1a'
                },
                ...themes
            ];
            settings_render_all_themes(allThemes, savedTheme);
        }).catch(e => {
            console.error('Failed to load user themes:', e);
        });
    }

    function settings_render_all_themes(themes, selectedName) {
        const grid = document.getElementById('themeGrid');
        if (!grid) return;

        grid.innerHTML = '';

        const builtinNames = ['com.viewstage.theme.dark', 'com.viewstage.theme.simplify'];
        const userThemes = [];

        themes.forEach(({ name, display_name, canvas_bg, text_color }) => {
            const card = document.createElement('div');
            card.className = 'theme-card' + (name === selectedName ? ' selected' : '');
            card.dataset.value = name;

            const isLight = canvas_bg === '#ffffff' || canvas_bg === '#fff';
            const dotColor = isLight ? '#1a1a1a' : canvas_bg;

            const isBuiltin = builtinNames.includes(name);
            const previewImg = isBuiltin
                ? `<img class="theme-card-preview-img" src="themes/${name}/preview.png" alt="${display_name}" loading="lazy">`
                : '';

            const fallbackHtml = !previewImg ? `
                <div class="theme-card-preview-bar" style="background: ${text_color}"></div>
                <span class="theme-card-preview-text" style="color: ${text_color}">Aa</span>
                <div class="theme-card-preview-dot" style="background: ${dotColor}"></div>
            ` : '';

            card.innerHTML = `
                <div class="theme-card-preview" style="background: ${canvas_bg}">
                    ${previewImg}
                    ${fallbackHtml}
                </div>
                <div class="theme-card-info">
                    <div class="theme-card-info-main">
                        <span class="theme-card-name">${display_name}</span>
                        <span class="theme-card-check">✓</span>
                    </div>
                </div>
                <div class="theme-card-actions">
                    <button class="theme-card-btn theme-card-btn-apply" data-action="apply">${window.i18n?.format_translate('settings.apply') || '应用'}</button>
                    ${isBuiltin ? '' : `<button class="theme-card-btn theme-card-btn-delete" data-action="delete">${window.i18n?.format_translate('settings.delete') || '删除'}</button>`}
                </div>
            `;

            grid.appendChild(card);

            if (!isBuiltin) {
                userThemes.push({ card, name });
            }
        });

        // 异步加载用户主题预览图
        if (userThemes.length > 0 && window.__TAURI__) {
            const { invoke } = window.__TAURI__.core;
            userThemes.forEach(({ card, name }) => {
                invoke('theme_get_preview', { name }).then(b64 => {
                    if (!b64) return;
                    const preview = card.querySelector('.theme-card-preview');
                    if (!preview) return;
                    preview.style.background = '';
                    preview.innerHTML = `<img class="theme-card-preview-img" src="${b64}" alt="" loading="lazy">`;
                }).catch(e => console.warn('加载主题预览失败:', e));
            });
        }
    }

    async function settings_fetch_supported_resolutions(deviceId) {
        const commonResolutions = [
            { w: 640, h: 480, label: '640 x 480 (VGA)', aspectRatio: '4:3' },
            { w: 800, h: 600, label: '800 x 600 (SVGA)', aspectRatio: '4:3' },
            { w: 1280, h: 720, label: '1280 x 720 (720p)', aspectRatio: '16:9' },
            { w: 1280, h: 960, label: '1280 x 960', aspectRatio: '4:3' },
            { w: 1600, h: 1200, label: '1600 x 1200', aspectRatio: '4:3' },
            { w: 1920, h: 1080, label: '1920 x 1080 (1080p)', aspectRatio: '16:9' },
            { w: 2560, h: 1440, label: '2560 x 1440 (2K)', aspectRatio: '16:9' },
            { w: 3840, h: 2160, label: '3840 x 2160 (4K)', aspectRatio: '16:9' }
        ];
        
        let stream = null;
        let track = null;
        const supportedResolutions = [];
        
        try {
            let constraints;
            if (deviceId && deviceId !== '') {
                constraints = { video: { deviceId: { exact: deviceId } } };
            } else {
                constraints = { video: true };
            }
            
            stream = await navigator.mediaDevices.getUserMedia(constraints);
            track = stream.getVideoTracks()[0];
            
            const capabilities = track.getCapabilities();
            const maxWidth = capabilities.width?.max || 1920;
            const maxHeight = capabilities.height?.max || 1080;
            
            for (const res of commonResolutions) {
                if (res.w <= maxWidth && res.h <= maxHeight) {
                    supportedResolutions.push({ ...res, actual: true });
                }
            }
            
            const maxText = window.i18n?.format_translate('settings.maximum') || '最大';
            supportedResolutions.push({
                w: maxWidth,
                h: maxHeight,
                label: `${maxWidth} x ${maxHeight} (${maxText})`,
                actual: true
            });
            
        } catch (error) {
            console.error('检测摄像头分辨率失败:', error);
        } finally {
            if (track) {
                track.stop();
            }
            if (stream) {
                stream.getTracks().forEach(t => t.stop());
            }
        }
        
        const uniqueResolutions = [];
        const seen = new Set();
        for (const res of supportedResolutions) {
            const key = `${res.w}x${res.h}`;
            if (!seen.has(key)) {
                seen.add(key);
                uniqueResolutions.push(res);
            }
        }
        
        return uniqueResolutions.sort((a, b) => (b.w * b.h) - (a.w * a.h));
    }
    
    async function settings_update_camera_resolution_options(deviceId, savedWidth, savedHeight) {
        const cameraResolutionOptionsContainer = document.getElementById('cameraResolutionOptions');
        const cameraResolutionSelected = document.getElementById('cameraResolutionSelected');
        
        if (!cameraResolutionOptionsContainer || !cameraResolutionSelected) return false;
        
        cameraResolutionOptionsContainer.innerHTML = '';
        
        const resolutions = await settings_fetch_supported_resolutions(deviceId);
        
        if (resolutions.length === 0) {
            cameraResolutionSelected.textContent = window.i18n?.format_translate('settings.cannotGet') || '无法获取';
            return false;
        }
        
        resolutions.forEach(res => {
            const option = document.createElement('div');
            option.className = 'select-option';
            option.dataset.width = res.w;
            option.dataset.height = res.h;
            option.dataset.value = `${res.w}x${res.h}`;
            option.textContent = res.label;
            cameraResolutionOptionsContainer.appendChild(option);
        });
        
        const targetWidth = savedWidth || (resolutions[0]?.w || 0);
        const targetHeight = savedHeight || (resolutions[0]?.h || 0);
        const targetRes = `${targetWidth}x${targetHeight}`;
        
        const resOptions = cameraResolutionOptionsContainer.querySelectorAll('.select-option');
        let found = false;
        
        resOptions.forEach(option => {
            if (option.dataset.value === targetRes) {
                cameraResolutionSelected.textContent = option.textContent;
                option.classList.add('selected');
                found = true;
            }
        });
        
        if (!found && resOptions.length > 0) {
            const defaultOption = resOptions[0];
            cameraResolutionSelected.textContent = defaultOption.textContent;
            defaultOption.classList.add('selected');
        }
        
        return true;
    }
    
    settings_load_version();
    settings_load_all().then(settings => {
        if (settings.developerMode) {
            developer_options_activate();
        }
    });
    
    // 统一接管所有自定义下拉框的展开/关闭（stopPropagation 防止事件冲突）
    function settings_init_all_selects() {
        function closeOneSelect(s) {
            s.classList.remove('open');
            const opts = document.querySelector('body > .select-options[data-owner="' + s.dataset.selectId + '"]');
            if (opts) {
                opts.style.opacity = '0';
                opts.style.visibility = 'hidden';
                setTimeout(() => {
                    if (!s.classList.contains('open')) {
                        s.appendChild(opts);
                        opts.classList.remove('up');
                        opts.style.position = '';
                        opts.style.left = '';
                        opts.style.top = '';
                        opts.style.bottom = '';
                        opts.style.minWidth = '';
                        opts.style.transform = '';
                    }
                }, 200);
            }
        }
        function closeAllSelects() {
            document.querySelectorAll('.custom-select.open').forEach(s => closeOneSelect(s));
        }
        document.addEventListener('click', closeAllSelects);
        let selectId = 0;
        document.querySelectorAll('.custom-select').forEach(select => {
            const selected = select.querySelector('.select-selected');
            if (!selected || select.dataset.selectInitialized) return;
            select.dataset.selectInitialized = 'true';
            const id = 'sel_' + (selectId++);
            select.dataset.selectId = id;
            const opts = select.querySelector('.select-options');
            if (opts) opts.dataset.owner = id;
            selected.addEventListener('click', (e) => {
                e.stopPropagation();
                document.querySelectorAll('.custom-select.open').forEach(s => {
                    if (s !== select) closeOneSelect(s);
                });
                const isOpen = select.classList.toggle('open');
                if (opts) {
                    if (isOpen) {
                        const rect = selected.getBoundingClientRect();
                        if (opts.parentNode !== document.body) {
                            document.body.appendChild(opts);
                        }
                        opts.style.visibility = 'hidden';
                        opts.style.opacity = '1';
                        void opts.offsetHeight;
                        const spaceBelow = window.innerHeight - rect.bottom;
                        const spaceAbove = rect.top;
                        const needed = opts.scrollHeight || 180;
                        const showUp = spaceBelow < needed && spaceAbove > spaceBelow;
                        opts.classList.toggle('up', showUp);
                        opts.style.position = 'fixed';
                        opts.style.left = rect.left + 'px';
                        if (showUp) {
                            opts.style.top = '';
                            opts.style.bottom = (window.innerHeight - rect.top + 4) + 'px';
                        } else {
                            opts.style.top = (rect.bottom + 4) + 'px';
                            opts.style.bottom = '';
                        }
                        opts.style.minWidth = rect.width + 'px';
                        opts.style.opacity = '1';
                        opts.style.visibility = 'visible';
                        opts.style.transform = 'translateY(0)';
                    } else {
                        if (opts.parentNode !== select) {
                            select.appendChild(opts);
                        }
                        opts.classList.remove('up');
                        opts.style.position = '';
                        opts.style.left = '';
                        opts.style.top = '';
                        opts.style.bottom = '';
                        opts.style.minWidth = '';
                        opts.style.opacity = '';
                        opts.style.visibility = '';
                        opts.style.transform = '';
                    }
                }
            });
        });
    }
    settings_init_all_selects();

    // 辅助函数：关闭下拉框并归位 portal 元素
    function closeSelect(selectEl) {
        selectEl.classList.remove('open');
        const opts = selectEl.querySelector('.select-options') || document.querySelector('body > .select-options[data-owner="' + selectEl.dataset.selectId + '"]');
        if (opts && opts.parentNode !== selectEl) {
            selectEl.appendChild(opts);
        }
        if (opts) {
            opts.classList.remove('up');
            opts.style.position = '';
            opts.style.left = '';
            opts.style.top = '';
            opts.style.bottom = '';
            opts.style.minWidth = '';
            opts.style.opacity = '';
            opts.style.visibility = '';
            opts.style.transform = '';
        }
    }

    // 语言选择
    const languageSelect = document.getElementById('languageSelect');
    const selectSelected = document.getElementById('selectSelected');
    const languageOptions = document.querySelectorAll('#selectOptions .select-option');
    if (languageSelect && selectSelected) {
        languageOptions.forEach(option => {
            option.addEventListener('click', async () => {
                const value = option.dataset.value;
                selectSelected.textContent = option.textContent;
                languageOptions.forEach(opt => opt.classList.remove('selected'));
                option.classList.add('selected');
                closeSelect(languageSelect);
                const saved = await settings_save_all_local({ language: value });
                if (saved) {
                    if (window.i18n) {
                        await window.i18n.load_messages(value);
                        window.i18n.render_page_texts();
                    }
                    const restartModal = document.getElementById('restartModal');
                    if (restartModal) restartModal.classList.add('active');
                } else {
                    settings_show_dialog(window.i18n?.format_translate('settings.saveFailed') || '保存失败', window.i18n?.format_translate('settings.saveFailedRetry') || '保存设置失败，请重试', 'error');
                }
            });
        });
    }
    
    // 摄像头选择
    const cameraSelect = document.getElementById('cameraSelect');
    const cameraSelected = document.getElementById('cameraSelected');
    const cameraOptions = document.getElementById('cameraOptions');
    if (cameraSelect && cameraSelected && cameraOptions) {
        cameraOptions.addEventListener('click', async (e) => {
            const option = e.target.closest('.select-option');
            if (!option) return;
            const value = option.dataset.value;
            cameraSelected.textContent = option.textContent;
            cameraOptions.querySelectorAll('.select-option').forEach(opt => opt.classList.remove('selected'));
            option.classList.add('selected');
            closeSelect(cameraSelect);
            try {
                const saved = await settings_save_all_local({ defaultCamera: value });
                if (saved) {
                    const currentResolutionOption = document.querySelector('#cameraResolutionOptions .select-option.selected');
                    const currentWidth = currentResolutionOption ? parseInt(currentResolutionOption.dataset.width) : null;
                    const currentHeight = currentResolutionOption ? parseInt(currentResolutionOption.dataset.height) : null;
                    const success = await settings_update_camera_resolution_options(value, currentWidth, currentHeight);
                    if (success) {
                        const newResolutionOption = document.querySelector('#cameraResolutionOptions .select-option.selected');
                        if (newResolutionOption) {
                            const newWidth = parseInt(newResolutionOption.dataset.width);
                            const newHeight = parseInt(newResolutionOption.dataset.height);
                            if (newWidth && newHeight) {
                                await settings_save_all_local({ cameraWidth: newWidth, cameraHeight: newHeight });
                            }
                        }
                    }
                } else {
                    settings_show_dialog(window.i18n?.format_translate('settings.saveFailed') || '保存失败', window.i18n?.format_translate('settings.saveFailedRetry') || '保存设置失败，请重试', 'error');
                }
            } catch (error) {
                console.error('切换摄像头失败:', error);
                settings_show_dialog(window.i18n?.format_translate('settings.saveFailed') || '保存失败', String(error), 'error');
            }
        });
    }
    
    // 摄像头分辨率选择
    const cameraResolutionSelect = document.getElementById('cameraResolutionSelect');
    const cameraResolutionSelected = document.getElementById('cameraResolutionSelected');
    const cameraResolutionOptions = document.getElementById('cameraResolutionOptions');
    if (cameraResolutionSelect && cameraResolutionSelected && cameraResolutionOptions) {
        cameraResolutionOptions.addEventListener('click', async (e) => {
            const option = e.target.closest('.select-option');
            if (!option) return;
            const width = parseInt(option.dataset.width);
            const height = parseInt(option.dataset.height);
            cameraResolutionSelected.textContent = option.textContent;
            cameraResolutionOptions.querySelectorAll('.select-option').forEach(opt => opt.classList.remove('selected'));
            option.classList.add('selected');
            closeSelect(cameraResolutionSelect);
            try {
                await settings_save_all_local({ cameraWidth: width, cameraHeight: height });
            } catch (error) {
                console.error('保存分辨率设置失败:', error);
            }
        });
    }
    
    // DPR 限制选择
    const dprLimitSelect = document.getElementById('dprLimitSelect');
    const dprLimitSelected = document.getElementById('dprLimitSelected');
    const dprLimitOptions = document.getElementById('dprLimitOptions');
    if (dprLimitSelect && dprLimitSelected && dprLimitOptions) {
        dprLimitOptions.addEventListener('click', async (e) => {
            const option = e.target.closest('.select-option');
            if (!option) return;
            const value = parseFloat(option.dataset.value);
            dprLimitSelected.textContent = option.textContent;
            dprLimitOptions.querySelectorAll('.select-option').forEach(opt => opt.classList.remove('selected'));
            option.classList.add('selected');
            closeSelect(dprLimitSelect);
            const saved = await settings_save_all_local({ dprLimit: value });
            if (saved) {
                const restartModal = document.getElementById('restartModal');
                const modalMessage = restartModal?.querySelector('.modal-message');
                if (modalMessage) modalMessage.textContent = window.i18n?.format_translate('settings.dprChanged') || '画面精度已更改，建议重启应用以确保完全生效。';
                if (restartModal) restartModal.classList.add('active');
            } else {
                settings_show_dialog(window.i18n?.format_translate('settings.saveFailed') || '保存失败', window.i18n?.format_translate('settings.saveFailedRetry') || '保存设置失败，请重试', 'error');
            }
        });
    }

    // 通用选择保存（dprMin / dprMax）
    function settings_bind_select_save(id, settingsKey) {
        const sel = document.getElementById(id);
        const selSelected = document.getElementById(id.replace('Select', 'Selected'));
        const selOptions = document.getElementById(id.replace('Select', 'Options'));
        if (!sel || !selSelected || !selOptions) return;
        selOptions.addEventListener('click', async (e) => {
            const option = e.target.closest('.select-option');
            if (!option) return;
            const value = parseFloat(option.dataset.value);
            selSelected.textContent = option.textContent;
            selOptions.querySelectorAll('.select-option').forEach(opt => opt.classList.remove('selected'));
            option.classList.add('selected');
            closeSelect(sel);
            await settings_save_all_local({ [settingsKey]: value });
        });
    }
    settings_bind_select_save('dprMinSelect', 'dprMin');
    settings_bind_select_save('dprMaxSelect', 'dprMax');

    const dprToggle = document.getElementById('dynamicDprToggle');
    if (dprToggle) {
        dprToggle.addEventListener('change', async () => {
            const value = dprToggle.checked;
            const dprRangeItem = document.getElementById('dprRangeItem');
            if (dprRangeItem) dprRangeItem.style.display = value ? '' : 'none';
            const dprLimitItem = document.getElementById('dprLimitItem');
            if (dprLimitItem) dprLimitItem.style.display = value ? 'none' : '';
            await settings_save_all_local({ dynamicDprEnabled: value });
        });
    }

    const eraserSpeedToggle = document.getElementById('eraserSpeedToggle');
    if (eraserSpeedToggle) {
        eraserSpeedToggle.addEventListener('change', async () => {
            await settings_save_all_local({ eraserSpeedEnabled: eraserSpeedToggle.checked });
            const restartModal = document.getElementById('restartModal');
            const modalMessage = restartModal?.querySelector('.modal-message');
            if (modalMessage) {
                modalMessage.textContent = window.i18n?.format_translate('settings.languageChanged') || '需要重启应用才能生效。';
            }
            if (restartModal) restartModal.classList.add('active');
        });
    }

    const momentumToggle = document.getElementById('momentumToggle');
    if (momentumToggle) {
        momentumToggle.addEventListener('change', async () => {
            await settings_save_all_local({ momentumEnabled: momentumToggle.checked });
        });
    }

    const palmEraserToggle = document.getElementById('palmEraserToggle');
    if (palmEraserToggle) {
        palmEraserToggle.addEventListener('change', async () => {
            await settings_save_all_local({ palmEraserEnabled: palmEraserToggle.checked });
            const restartModal = document.getElementById('restartModal');
            const modalMessage = restartModal?.querySelector('.modal-message');
            if (modalMessage) {
                modalMessage.textContent = window.i18n?.format_translate('settings.languageChanged') || '需要重启应用才能生效。';
            }
            if (restartModal) restartModal.classList.add('active');
        });
    }
    
    // 自定义颜色选择器
    const colorPickerPopup = document.getElementById('colorPickerPopup');
    const colorPickerSV = document.getElementById('colorPickerSV');
    const colorPickerSVCursor = document.getElementById('colorPickerSVCursor');
    const colorPickerHue = document.getElementById('colorPickerHue');
    const colorPickerHueCursor = document.getElementById('colorPickerHueCursor');
    const colorPickerPresets = document.getElementById('colorPickerPresets');
    const colorPickerPreview = document.getElementById('colorPickerPreview');
    const colorPickerInput = document.getElementById('colorPickerInput');
    const colorPickerConfirm = document.getElementById('colorPickerConfirm');
    const colorPickerCancel = document.getElementById('colorPickerCancel');
    
    let current_color_index = 0;
    let current_hue = 0;
    let current_saturation = 100;
    let current_value = 100;
    let color_picker_overlay = null;
    
    const PRESET_COLORS = [
        '#e74c3c', '#e91e63', '#9b59b6', '#673ab7',
        '#3498db', '#00bcd4', '#1abc9c', '#2ecc71',
        '#8bc34a', '#f39c12', '#ff5722', '#795548',
        '#34495e', '#000000', '#ffffff'
    ];
    
    function settings_init_color_picker_presets() {
        if (!colorPickerPresets) return;
        colorPickerPresets.innerHTML = '';
        PRESET_COLORS.forEach(color => {
            const preset = document.createElement('div');
            preset.className = 'color-picker-preset';
            preset.style.backgroundColor = color;
            preset.addEventListener('click', () => {
                const rgb = settings_calc_hex_to_rgb(color);
                if (rgb) {
                    const hsv = settings_calc_rgb_to_hsv(rgb.r, rgb.g, rgb.b);
                    current_hue = hsv.h;
                    current_saturation = hsv.s;
                    current_value = hsv.v;
                    settings_update_color_picker_ui();
                }
            });
            colorPickerPresets.appendChild(preset);
        });
    }
    
    function settings_calc_hsv_to_rgb(h, s, v) {
        s /= 100;
        v /= 100;
        const c = v * s;
        const x = c * (1 - Math.abs((h / 60) % 2 - 1));
        const m = v - c;
        let r = 0, g = 0, b = 0;
        if (h < 60) { r = c; g = x; b = 0; }
        else if (h < 120) { r = x; g = c; b = 0; }
        else if (h < 180) { r = 0; g = c; b = x; }
        else if (h < 240) { r = 0; g = x; b = c; }
        else if (h < 300) { r = x; g = 0; b = c; }
        else { r = c; g = 0; b = x; }
        return {
            r: Math.round((r + m) * 255),
            g: Math.round((g + m) * 255),
            b: Math.round((b + m) * 255)
        };
    }
    
    function settings_calc_rgb_to_hsv(r, g, b) {
        r /= 255; g /= 255; b /= 255;
        const max = Math.max(r, g, b), min = Math.min(r, g, b);
        let h = 0, s = max === 0 ? 0 : (max - min) / max, v = max;
        if (max !== min) {
            const d = max - min;
            if (max === r) h = ((g - b) / d + (g < b ? 6 : 0)) * 60;
            else if (max === g) h = ((b - r) / d + 2) * 60;
            else h = ((r - g) / d + 4) * 60;
        }
        return { h, s: s * 100, v: v * 100 };
    }
    
    function settings_calc_rgb_to_hex(r, g, b) {
        return '#' + [r, g, b].map(x => {
            const hex = x.toString(16);
            return hex.length === 1 ? '0' + hex : hex;
        }).join('');
    }
    
    function settings_calc_hex_to_rgb(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : null;
    }
    
    function settings_calc_current_hex_color() {
        const rgb = settings_calc_hsv_to_rgb(current_hue, current_saturation, current_value);
        return settings_calc_rgb_to_hex(rgb.r, rgb.g, rgb.b);
    }
    
    function settings_update_color_picker_ui() {
        const rgb = settings_calc_hsv_to_rgb(current_hue, current_saturation, current_value);
        const hex = settings_calc_rgb_to_hex(rgb.r, rgb.g, rgb.b);
        
        if (colorPickerSVCursor) {
            const x = (current_saturation / 100) * 240;
            const y = (1 - current_value / 100) * 180;
            colorPickerSVCursor.style.left = x + 'px';
            colorPickerSVCursor.style.top = y + 'px';
        }
        
        if (colorPickerHueCursor) {
            const hueX = (current_hue / 360) * 240;
            colorPickerHueCursor.style.left = hueX + 'px';
        }
        
        if (colorPickerSV) {
            const hueRgb = settings_calc_hsv_to_rgb(current_hue, 100, 100);
            const hueHex = settings_calc_rgb_to_hex(hueRgb.r, hueRgb.g, hueRgb.b);
            colorPickerSV.style.backgroundColor = hueHex;
        }
        
        if (colorPickerPreview) {
            colorPickerPreview.style.backgroundColor = hex;
        }
        
        if (colorPickerInput) {
            colorPickerInput.value = hex;
        }
    }
    
    function settings_show_color_picker(index) {
        current_color_index = index;
        const colorBtn = document.querySelector(`.color-edit-item[data-index="${index}"] .color-edit-btn`);
        if (colorBtn) {
            const hex = colorBtn.dataset.color || '#3498db';
            const rgb = settings_calc_hex_to_rgb(hex);
            if (rgb) {
                const hsv = settings_calc_rgb_to_hsv(rgb.r, rgb.g, rgb.b);
                current_hue = hsv.h;
                current_saturation = hsv.s;
                current_value = hsv.v;
            }
        }
        
        settings_update_color_picker_ui();
        settings_init_color_picker_presets();
        
        if (colorPickerPopup) {
            colorPickerPopup.classList.add('active');
        }
        
        if (!color_picker_overlay) {
            color_picker_overlay = document.createElement('div');
            color_picker_overlay.className = 'color-picker-overlay';
            color_picker_overlay.addEventListener('click', settings_hide_color_picker);
            document.body.appendChild(color_picker_overlay);
        }
        color_picker_overlay.style.display = 'block';
    }
    
    function settings_hide_color_picker() {
        if (colorPickerPopup) {
            colorPickerPopup.classList.remove('active');
        }
        if (color_picker_overlay) {
            color_picker_overlay.style.display = 'none';
        }
    }
    
    function settings_handle_sv_drag(e) {
        if (!colorPickerSV) return;
        const rect = colorPickerSV.getBoundingClientRect();
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        let x = Math.max(0, Math.min(clientX - rect.left, rect.width));
        let y = Math.max(0, Math.min(clientY - rect.top, rect.height));
        current_saturation = (x / rect.width) * 100;
        current_value = (1 - y / rect.height) * 100;
        settings_update_color_picker_ui();
    }
    
    function settings_handle_hue_drag(e) {
        if (!colorPickerHue) return;
        const rect = colorPickerHue.getBoundingClientRect();
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        let x = Math.max(0, Math.min(clientX - rect.left, rect.width));
        current_hue = (x / rect.width) * 360;
        settings_update_color_picker_ui();
    }
    
    if (colorPickerSV) {
        let is_sv_dragging = false;
        colorPickerSV.addEventListener('mousedown', (e) => { is_sv_dragging = true; settings_handle_sv_drag(e); });
        colorPickerSV.addEventListener('touchstart', (e) => { is_sv_dragging = true; settings_handle_sv_drag(e); }, { passive: true });
        document.addEventListener('mousemove', (e) => { if (is_sv_dragging) settings_handle_sv_drag(e); });
        document.addEventListener('touchmove', (e) => { if (is_sv_dragging) settings_handle_sv_drag(e); }, { passive: true });
        document.addEventListener('mouseup', () => { is_sv_dragging = false; });
        document.addEventListener('touchend', () => { is_sv_dragging = false; });
    }
    
    if (colorPickerHue) {
        let is_hue_dragging = false;
        colorPickerHue.addEventListener('mousedown', (e) => { is_hue_dragging = true; settings_handle_hue_drag(e); });
        colorPickerHue.addEventListener('touchstart', (e) => { is_hue_dragging = true; settings_handle_hue_drag(e); }, { passive: true });
        document.addEventListener('mousemove', (e) => { if (is_hue_dragging) settings_handle_hue_drag(e); });
        document.addEventListener('touchmove', (e) => { if (is_hue_dragging) settings_handle_hue_drag(e); }, { passive: true });
        document.addEventListener('mouseup', () => { is_hue_dragging = false; });
        document.addEventListener('touchend', () => { is_hue_dragging = false; });
    }
    
    if (colorPickerInput) {
        colorPickerInput.addEventListener('input', () => {
            const hex = colorPickerInput.value;
            if (/^#[0-9A-Fa-f]{6}$/.test(hex)) {
                const rgb = settings_calc_hex_to_rgb(hex);
                if (rgb) {
                    const hsv = settings_calc_rgb_to_hsv(rgb.r, rgb.g, rgb.b);
                    current_hue = hsv.h;
                    current_saturation = hsv.s;
                    current_value = hsv.v;
                    settings_update_color_picker_ui();
                }
            }
        });
    }
    
    if (colorPickerConfirm) {
        colorPickerConfirm.addEventListener('click', async () => {
            const hex = settings_calc_current_hex_color();
            const colorBtn = document.querySelector(`.color-edit-item[data-index="${current_color_index}"] .color-edit-btn`);
            if (colorBtn) {
                colorBtn.style.backgroundColor = hex;
                colorBtn.dataset.color = hex;
            }
            
            const colors = [];
            for (let i = 0; i < 15; i++) {
                const btn = document.querySelector(`.color-edit-item[data-index="${i}"] .color-edit-btn`);
                const hexColor = btn ? btn.dataset.color : '#000000';
                const rgb = settings_calc_hex_to_rgb(hexColor);
                colors.push(rgb || { r: 0, g: 0, b: 0 });
            }
            await settings_save_all_local({ penColors: colors });
            
            settings_hide_color_picker();
        });
    }
    
    if (colorPickerCancel) {
        colorPickerCancel.addEventListener('click', settings_hide_color_picker);
    }
    
    document.querySelectorAll('.color-edit-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const item = btn.closest('.color-edit-item');
            if (item) {
                const index = parseInt(item.dataset.index);
                settings_show_color_picker(index);
            }
        });
    });
    
    // 默认粗细选择点
    document.querySelectorAll('.preset-default-dot').forEach(dot => {
        dot.addEventListener('click', async () => {
            const group = dot.dataset.default;
            const idx = parseInt(dot.dataset.index);
            const input = document.getElementById((group === 'pen' ? 'penPreset' : 'eraserPreset') + idx);
            if (!input) return;
            const val = parseInt(input.value);
            if (isNaN(val)) return;
            const key = group === 'pen' ? 'penWidth' : 'eraserSize';
            document.querySelectorAll(`.preset-default-dot[data-default="${group}"]`).forEach(d => d.classList.remove('active'));
            dot.classList.add('active');
            await settings_save_all_local({ [key]: val });
        });
    });

    // 画笔预设粗细
    const DEFAULT_PRESETS_BIND = [2, 5, 10, 15, 21];
    function settings_read_presets_from_ui() {
        const values = [];
        for (let i = 0; i < 5; i++) {
            const input = document.getElementById('penPreset' + i);
            values.push(input ? parseInt(input.value) || DEFAULT_PRESETS_BIND[i] : DEFAULT_PRESETS_BIND[i]);
        }
        return values;
    }
    for (let i = 0; i < 5; i++) {
        const input = document.getElementById('penPreset' + i);
        if (input) {
            input.addEventListener('change', async () => {
                let val = parseInt(input.value);
                if (isNaN(val) || val < 0) val = 0;
                if (val > 100) val = 100;
                input.value = val;
                const presets = settings_read_presets_from_ui();
                await settings_save_all_local({ penSizePresets: presets });
                settings_update_default_dots('pen');
            });
        }
    }
    const presetRestore = document.getElementById('penPresetRestore');
    if (presetRestore) {
        presetRestore.addEventListener('click', async () => {
            const defaults = DEFAULT_PRESETS_BIND;
            for (let i = 0; i < 5; i++) {
                const input = document.getElementById('penPreset' + i);
                if (input) input.value = defaults[i];
            }
            await settings_save_all_local({ penSizePresets: defaults });
            settings_update_default_dots('pen');
        });
    }

    // 橡皮擦预设粗细
    const DEFAULT_ERASER_PRESETS_BIND = [5, 15, 25, 38, 50];
    function settings_read_eraser_presets_from_ui() {
        const values = [];
        for (let i = 0; i < 5; i++) {
            const input = document.getElementById('eraserPreset' + i);
            values.push(input ? parseInt(input.value) || DEFAULT_ERASER_PRESETS_BIND[i] : DEFAULT_ERASER_PRESETS_BIND[i]);
        }
        return values;
    }
    for (let i = 0; i < 5; i++) {
        const input = document.getElementById('eraserPreset' + i);
        if (input) {
            input.addEventListener('change', async () => {
                let val = parseInt(input.value);
                if (isNaN(val) || val < 0) val = 0;
                if (val > 200) val = 200;
                input.value = val;
                const presets = settings_read_eraser_presets_from_ui();
                await settings_save_all_local({ eraserSizePresets: presets });
                settings_update_default_dots('eraser');
            });
        }
    }
    const eraserPresetRestore = document.getElementById('eraserPresetRestore');
    if (eraserPresetRestore) {
        eraserPresetRestore.addEventListener('click', async () => {
            const defaults = DEFAULT_ERASER_PRESETS_BIND;
            for (let i = 0; i < 5; i++) {
                const input = document.getElementById('eraserPreset' + i);
                if (input) input.value = defaults[i];
            }
            await settings_save_all_local({ eraserSizePresets: defaults });
            settings_update_default_dots('eraser');
        });
    }

    // 镜像开关
    const mirrorToggle = document.getElementById('mirrorToggle');
    if (mirrorToggle) {
        mirrorToggle.addEventListener('change', async () => {
            try {
                const { invoke } = window.__TAURI__.core;
                await invoke('mirror_update_state', { enabled: mirrorToggle.checked });
            } catch (error) {
                console.error('设置镜像状态失败:', error);
            }
        });
    }

    // 展台灯开关
    const cameraLightToggle = document.getElementById('cameraLightToggle');
    const cameraLightItem = document.getElementById('cameraLightItem');
    if (cameraLightToggle && cameraLightItem) {
        (async () => {
            try {
                const { invoke } = window.__TAURI__.core;
                const seewoDetected = await invoke('camera_light_detect_and_save');
                if (!seewoDetected) return;
                cameraLightItem.style.display = '';
                try {
                    const [isOn] = await invoke('camera_light_get_state');
                    cameraLightToggle.checked = isOn;
                } catch (_) {}
            } catch (_) {}
        })();
        cameraLightToggle.addEventListener('change', async () => {
            try {
                const { invoke } = window.__TAURI__.core;
                if (cameraLightToggle.checked) {
                    await invoke('camera_light_on');
                } else {
                    await invoke('camera_light_off');
                }
            } catch (error) {
                console.error('展台灯控制失败:', error);
                cameraLightToggle.checked = !cameraLightToggle.checked;
            }
        });
    }

    // 恢复上次文档状态开关
    const restoreLastDocToggle = document.getElementById('restoreLastDocToggle');
    if (restoreLastDocToggle) {
        restoreLastDocToggle.addEventListener('change', async () => {
            const enabled = restoreLastDocToggle.checked;
            await settings_save_all_local({ restoreLastDoc: enabled });
            window.__restoreLastDocEnabled = enabled;
            // 关闭时立即清理保存的文档状态和缓存
            if (!enabled) {
                await settings_save_all_local({ lastOpenDoc: null });
                if (window.documentReaderManager) {
                    await window.documentReaderManager.delete_annotation_cache_files?.();
                }
            }
        });
    }

    // 帧率模式选择
    const frameRateModeGroup = document.getElementById('frameRateModeGroup');
    if (frameRateModeGroup) {
        const buttons = frameRateModeGroup.querySelectorAll('.option-btn');
        buttons.forEach(btn => {
            btn.addEventListener('click', async () => {
                const mode = btn.dataset.value;
                frameRateModeGroup.dataset.active = mode;
                buttons.forEach(b => b.classList.toggle('active', b === btn));
                await settings_save_all_local({ frameRateMode: mode });
                if (window.batchDrawManager) {
                    window.batchDrawManager.batch_draw_update_frame_rate(mode);
                }
            });
        });
    }
    
    // 钢笔效果模式选择
    const penEffectModeGroup = document.getElementById('penEffectModeGroup');
    if (penEffectModeGroup) {
        const buttons = penEffectModeGroup.querySelectorAll('.option-btn');
        buttons.forEach(btn => {
            btn.addEventListener('click', async () => {
                const mode = btn.dataset.value;
                penEffectModeGroup.dataset.active = mode;
                buttons.forEach(b => b.classList.toggle('active', b === btn));
                await settings_save_all_local({ penEffectMode: mode });
                if (window.DRAW_CONFIG) {
                    window.DRAW_CONFIG.penEffectMode = mode;
                    if (window.realPenManager) {
                        window.realPenManager.invalidate_cache();
                    }
                }
            });
        });
    }

    // 文档阅读器默认缩放
    const docReaderZoomGroup = document.getElementById('docReaderZoomGroup');
    if (docReaderZoomGroup) {
        const buttons = docReaderZoomGroup.querySelectorAll('.option-btn');
        buttons.forEach(btn => {
            btn.addEventListener('click', async () => {
                const zoom = btn.dataset.value;
                docReaderZoomGroup.dataset.active = zoom;
                buttons.forEach(b => b.classList.toggle('active', b === btn));
                await settings_save_all_local({ docReaderDefaultZoom: zoom });
            });
        });
    }

    // 文档阅读器显示最小化按钮
    const docReaderShowMinimizeToggle = document.getElementById('docReaderShowMinimizeToggle');
    if (docReaderShowMinimizeToggle) {
        docReaderShowMinimizeToggle.addEventListener('change', async () => {
            await settings_save_all_local({ docReaderShowMinimize: docReaderShowMinimizeToggle.checked });
        });
    }
    
    // 主题卡片选择
    const themeGrid = document.getElementById('themeGrid');
    if (themeGrid) {
        themeGrid.addEventListener('click', async (e) => {
            const btn = e.target.closest('.theme-card-btn');
            const card = e.target.closest('.theme-card');
            if (!card) return;

            // 处理删除
            if (btn?.dataset.action === 'delete') {
                const name = card.dataset.value;
                const displayName = card.querySelector('.theme-card-name')?.textContent || name;
                if (!await settings_show_confirm(
                    window.i18n?.format_translate('settings.deleteTheme') || '删除主题',
                    window.i18n?.format_translate('settings.deleteThemeConfirm')?.replace('{name}', displayName) || `确定要删除主题"${displayName}"吗？`
                )) return;

                const { invoke } = window.__TAURI__.core;
                try {
                    await invoke('theme_delete', { name });
                    card.remove();
                    // 如果删除的是当前选中的主题，切回 simplify
                    const selectedCard = themeGrid.querySelector('.theme-card.selected');
                    if (!selectedCard) {
                        const simplifyCard = themeGrid.querySelector('.theme-card[data-value="com.viewstage.theme.simplify"]');
                        if (simplifyCard) {
                            themeGrid.querySelectorAll('.theme-card').forEach(c => c.classList.remove('selected'));
                            simplifyCard.classList.add('selected');
                            await settings_save_all_local({ theme: 'com.viewstage.theme.simplify' });
                        }
                    }
                } catch (e) {
                    console.error('Failed to delete theme:', e);
                    settings_show_dialog(
                        window.i18n?.format_translate('settings.deleteFailed') || '删除失败',
                        window.i18n?.format_translate('settings.deleteThemeError')?.replace('{error}', e) || `删除主题失败: ${e}`,
                        'error'
                    );
                }
                return;
            }

            // 处理应用 — 仅点击"应用"按钮
            if (!btn || btn.dataset.action !== 'apply') return;
            if (card.classList.contains('selected')) return;

            const value = card.dataset.value;

            themeGrid.querySelectorAll('.theme-card').forEach(c => c.classList.remove('selected'));
            card.classList.add('selected');

            const saved = await settings_save_all_local({ theme: value });
            if (saved) {
                const restartModal = document.getElementById('restartModal');
                const modalMessage = restartModal?.querySelector('.modal-message');
                if (modalMessage) {
                    modalMessage.textContent = window.i18n?.format_translate('settings.themeChanged') || 'Theme changed, restart to apply.';
                }
                if (restartModal) {
                    restartModal.classList.add('active');
                }
            }
        });
    }
    
    // 默认旋转角度选择
    const defaultRotationSelect = document.getElementById('defaultRotationSelect');
    const defaultRotationSelected = document.getElementById('defaultRotationSelected');
    if (defaultRotationSelect && defaultRotationSelected) {
        document.querySelectorAll('#defaultRotationOptions .select-option').forEach(option => {
            option.addEventListener('click', async () => {
                const value = parseInt(option.dataset.value);
                defaultRotationSelected.textContent = option.textContent;
                document.querySelectorAll('#defaultRotationOptions .select-option').forEach(opt => opt.classList.remove('selected'));
                option.classList.add('selected');
                closeSelect(defaultRotationSelect);
                await settings_save_all_local({ defaultRotation: value });
            });
        });
    }
    
    // 默认打开方式按钮
    const btnSetPdf = document.getElementById('btnSetPdf');
    if (btnSetPdf) {
        btnSetPdf.addEventListener('click', async () => {
            if (!window.__TAURI__) return;
            const { invoke } = window.__TAURI__.core;
            const origText = btnSetPdf.textContent;
            btnSetPdf.disabled = true;
            btnSetPdf.textContent = window.i18n?.format_translate('settings.setSetting') || '设置中...';
            try {
                await invoke('filetype_set_icons');
                settings_show_dialog(
                    window.i18n?.format_translate('common.success') || '成功',
                    window.i18n?.format_translate('settings.pdfDefaultSetSuccess') || 'PDF 已设置为默认打开方式',
                    'success'
                );
            } catch (e) {
                console.error('设置 PDF 默认打开方式:', e);
                settings_show_dialog(
                    window.i18n?.format_translate('common.success') || '成功',
                    window.i18n?.format_translate('settings.setDefaultManual') || '部分关联已注册，请在系统设置中手动设置默认程序',
                    'info'
                );
            } finally {
                btnSetPdf.disabled = false;
                btnSetPdf.textContent = origText;
            }
        });
    }

    const btnSetWord = document.getElementById('btnSetWord');
    if (btnSetWord) {
        btnSetWord.addEventListener('click', async () => {
            if (!window.__TAURI__) return;
            const { invoke } = window.__TAURI__.core;
            const origText = btnSetWord.textContent;
            btnSetWord.disabled = true;
            btnSetWord.textContent = window.i18n?.format_translate('settings.setSetting') || '设置中...';
            try {
                await invoke('filetype_set_icons');
                settings_show_dialog(
                    window.i18n?.format_translate('common.success') || '成功',
                    window.i18n?.format_translate('settings.wordDefaultSetSuccess') || 'Word 文档已设置为默认打开方式',
                    'success'
                );
            } catch (e) {
                console.error('设置 Word 默认打开方式:', e);
                settings_show_dialog(
                    window.i18n?.format_translate('common.success') || '成功',
                    window.i18n?.format_translate('settings.setDefaultManual') || '部分关联已注册，请在系统设置中手动设置默认程序',
                    'info'
                );
            } finally {
                btnSetWord.disabled = false;
                btnSetWord.textContent = origText;
            }
        });
    }
    
    const btnReset = document.getElementById('btnReset');
    const modalOverlay = document.getElementById('modalOverlay');
    const modalCancel = document.getElementById('modalCancel');
    const modalConfirm = document.getElementById('modalConfirm');
    
    // 导出设置
    const btnExportSettings = document.getElementById('btnExportSettings');
    if (btnExportSettings && window.__TAURI__) {
        btnExportSettings.addEventListener('click', async () => {
            try {
                const { invoke } = window.__TAURI__.core;
                const { save } = window.__TAURI__.dialog;
                const { writeTextFile } = window.__TAURI__.fs;
                
                const result = await invoke('settings_fetch_all');
                const jsonStr = JSON.stringify(result.settings, null, 2);
                
                const filePath = await save({
                    defaultPath: 'viewstage-settings.json',
                    filters: [{ name: 'JSON', extensions: ['json'] }]
                });
                
                if (filePath) {
                    await writeTextFile(filePath, jsonStr);
                    console.log('设置已导出:', filePath);
                }
            } catch (error) {
                console.error('导出设置失败:', error);
                settings_show_dialog(window.i18n?.format_translate('settings.exportFailed') || '导出失败', String(error), 'error');
            }
        });
    }
    
    // 导入设置
    const btnImportSettings = document.getElementById('btnImportSettings');
    if (btnImportSettings && window.__TAURI__) {
        btnImportSettings.addEventListener('click', async () => {
            try {
                const { invoke } = window.__TAURI__.core;
                const { open } = window.__TAURI__.dialog;
                const { readTextFile } = window.__TAURI__.fs;
                
                const filePath = await open({
                    filters: [{ name: 'JSON', extensions: ['json'] }]
                });
                
                if (filePath) {
                    const jsonStr = await readTextFile(filePath);
                    const settings = JSON.parse(jsonStr);
                    
                await invoke('settings_save_all', { settings });
                    console.log('设置已导入:', filePath);
                    
                    // 重新加载页面以应用新设置
                    location.reload();
                }
            } catch (error) {
                console.error('导入设置失败:', error);
                settings_show_dialog(window.i18n?.format_translate('settings.importFailed') || '导入失败', String(error), 'error');
            }
        });
    }

    // 导入 .vst 主题
    const btnImportTheme = document.getElementById('btnImportTheme');
    if (btnImportTheme && window.__TAURI__) {
        btnImportTheme.addEventListener('click', async () => {
            try {
                const { invoke } = window.__TAURI__.core;
                const { open } = window.__TAURI__.dialog;

                const filePath = await open({
                    filters: [{ name: 'ViewStage Theme', extensions: ['vst'] }]
                });

                if (filePath) {
                    let themeResult;
                    try {
                        themeResult = await invoke('theme_import_vst', { filePath, force: false });
                    } catch (err) {
                        // 主题已存在，询问是否覆盖
                        if (String(err).includes('already exists')) {
                            if (!await settings_show_confirm(
                                window.i18n?.format_translate('settings.importTheme') || '导入主题',
                                window.i18n?.format_translate('settings.themeOverwriteConfirm') || '主题已存在，是否覆盖？'
                            )) {
                                return;
                            }
                            themeResult = await invoke('theme_import_vst', { filePath, force: true });
                        } else {
                            throw err;
                        }
                    }

                    const themeName = themeResult.name;
                    const displayName = themeResult.display_name;
                    const canvasBg = themeResult.canvas_bg || '#1a1a1a';
                    const textColor = themeResult.text_color || '#ffffff';
                    console.log('主题已导入:', themeName);

                    // 刷新所有主题卡片
                    const savedTheme = themeName;
                    invoke('theme_list_user').then(themes => {
                        const builtinSuffix = window.i18n?.format_translate('settings.themeBuiltinSuffix') || '（内置主题）';
                        const allThemes = [
                            {
                            name: 'com.viewstage.theme.dark',
                            display_name: (window.i18n?.format_translate('settings.themeDark') || '深色') + builtinSuffix,
                            canvas_bg: '#1a1a1a',
                            text_color: '#ffffff'
                        },
                        {
                            name: 'com.viewstage.theme.simplify',
                                display_name: (window.i18n?.format_translate('settings.themeSimplify') || '浅色') + builtinSuffix,
                                canvas_bg: '#ffffff',
                                text_color: '#1a1a1a'
                            },
                            ...themes
                        ];
                        settings_render_all_themes(allThemes, savedTheme);
                    }).catch(() => {
                        // 失败时至少添加当前导入的主题卡片，并尝试加载预览图
                        const grid = document.getElementById('themeGrid');
                        if (grid) {
                            const existing = grid.querySelector(`.theme-card[data-value="${themeName}"]`);
                            if (!existing) {
                                const card = document.createElement('div');
                                card.className = 'theme-card selected';
                                card.dataset.value = themeName;
                                card.innerHTML = `
                                    <div class="theme-card-preview" style="background: ${canvasBg}">
                                        <div class="theme-card-preview-bar" style="background: ${textColor}"></div>
                                        <span class="theme-card-preview-text" style="color: ${textColor}">Aa</span>
                                        <div class="theme-card-preview-dot" style="${canvasBg === '#ffffff' ? 'background: #1a1a1a' : 'background: ' + canvasBg}"></div>
                                    </div>
                                    <div class="theme-card-info">
                                        <div class="theme-card-info-main">
                                            <span class="theme-card-name">${displayName}</span>
                                            <span class="theme-card-check">✓</span>
                                        </div>
                                    </div>
                                    <div class="theme-card-actions">
                                        <button class="theme-card-btn theme-card-btn-apply" data-action="apply">${window.i18n?.format_translate('settings.apply') || '应用'}</button>
                                        <button class="theme-card-btn theme-card-btn-delete" data-action="delete">${window.i18n?.format_translate('settings.delete') || '删除'}</button>
                                    </div>
                                `;
                                grid.appendChild(card);
                                // 异步加载预览图
                                invoke('theme_get_preview', { name: themeName }).then(b64 => {
                                    if (!b64) return;
                                    const preview = card.querySelector('.theme-card-preview');
                                    if (!preview) return;
                                    preview.style.background = '';
                                    preview.innerHTML = `<img class="theme-card-preview-img" src="${b64}" alt="" loading="lazy">`;
                                }).catch(e => console.warn('加载主题预览失败:', e));
                            }
                            grid.querySelectorAll('.theme-card').forEach(c => c.classList.remove('selected'));
                            if (existing) existing.classList.add('selected');
                        }
                    });

                    // 自动选中并提示重启
                    await settings_save_all_local({ theme: themeName });
                    const restartModal = document.getElementById('restartModal');
                    const modalMessage = restartModal?.querySelector('.modal-message');
                    if (modalMessage) {
                        modalMessage.textContent = window.i18n?.format_translate('settings.themeChanged') || 'Theme changed, restart to apply.';
                    }
                    if (restartModal) restartModal.classList.add('active');

                    settings_show_dialog(
                        window.i18n?.format_translate('settings.themeImportSuccess') || '主题导入成功，重启后生效。',
                        '',
                        'success'
                    );
                }
            } catch (error) {
                console.error('导入主题失败:', error);
                settings_show_dialog(
                    window.i18n?.format_translate('settings.themeImportFailed') || '主题导入失败',
                    String(error),
                    'error'
                );
            }
        });
    }

    if (btnReset && modalOverlay && window.__TAURI__) {
        btnReset.addEventListener('click', () => {
            const modalTitle = modalOverlay.querySelector('.modal-title');
            const modalMessage = modalOverlay.querySelector('.modal-message');
            if (modalTitle && modalMessage) {
                modalTitle.textContent = window.i18n?.format_translate('settings.confirmReset') || '确认重置';
                modalMessage.textContent = window.i18n?.format_translate('settings.resetWarning') || '确定要重置应用吗？这将删除所有设置并重启应用。';
            }
            modalConfirm.dataset.action = 'reset';
            modalOverlay.classList.add('active');
        });
        
        modalCancel.addEventListener('click', () => {
            modalOverlay.classList.remove('active');
            delete modalConfirm.dataset.action;
        });
        
        modalOverlay.addEventListener('click', (e) => {
            if (e.target === modalOverlay) {
                modalOverlay.classList.remove('active');
                delete modalConfirm.dataset.action;
            }
        });
        
        modalConfirm.addEventListener('click', async () => {
            try {
                const { invoke } = window.__TAURI__.core;
                const { getCurrentWebview } = window.__TAURI__.webview;
                const webview = getCurrentWebview();
                await webview.clearAllBrowsingData();
                await invoke('settings_delete_all');
            } catch (error) {
                console.error('重置失败:', error);
                settings_show_dialog(window.i18n?.format_translate('settings.saveFailed') || '保存失败', String(error), 'error');
                modalOverlay.classList.remove('active');
            }
        });
    }
    
    // 缓存管理
    const cacheSizeEl = document.getElementById('cacheSize');
    const btnClearCache = document.getElementById('btnClearCache');
    
    async function settings_update_cache_size() {
        if (!window.__TAURI__) return;
        try {
            const { invoke } = window.__TAURI__.core;
            const size = await invoke('cache_fetch_size');
            if (cacheSizeEl) {
                if (size === 0) {
                    cacheSizeEl.textContent = '(0 B)';
                } else if (size < 1024) {
                    cacheSizeEl.textContent = `(${size} B)`;
                } else if (size < 1024 * 1024) {
                    cacheSizeEl.textContent = `(${(size / 1024).toFixed(1)} KB)`;
                } else if (size < 1024 * 1024 * 1024) {
                    cacheSizeEl.textContent = `(${(size / 1024 / 1024).toFixed(1)} MB)`;
                } else {
                    cacheSizeEl.textContent = `(${(size / 1024 / 1024 / 1024).toFixed(2)} GB)`;
                }
            }
        } catch (error) {
            console.error('获取缓存大小失败:', error);
        }
    }
    
    settings_update_cache_size();
    
    if (btnClearCache && window.__TAURI__) {
        btnClearCache.addEventListener('click', async () => {
            try {
                const { invoke } = window.__TAURI__.core;
                const result = await invoke('cache_delete_all');
                settings_show_dialog(window.i18n?.format_translate('settings.clearComplete') || '清除完成', result, 'success');
                settings_update_cache_size();
            } catch (error) {
                console.error('清除缓存失败:', error);
                settings_show_dialog(window.i18n?.format_translate('settings.clearFailed') || '清除失败', String(error), 'error');
            }
        });
    }
    
    // 自动清除缓存设置
    const autoClearCacheSelect = document.getElementById('autoClearCacheSelect');
    const autoClearCacheSelected = document.getElementById('autoClearCacheSelected');
    const autoClearCacheOptions = document.getElementById('autoClearCacheOptions');
    
    if (autoClearCacheSelect && autoClearCacheSelected && autoClearCacheOptions && window.__TAURI__) {
        autoClearCacheOptions.querySelectorAll('.select-option').forEach(option => {
            option.addEventListener('click', async () => {
                const days = parseInt(option.dataset.value);
                autoClearCacheSelected.textContent = option.textContent;
                closeSelect(autoClearCacheSelect);
                
                if (days === 0) {
                    settings_show_dialog(window.i18n?.format_translate('common.warning') || '警告', window.i18n?.format_translate('errors.autoClearWarning') || '若关闭自动清理可能导致C盘异常，强烈建议打开自动清理功能', 'error');
                }
                await settings_save_all_local({ autoClearCacheDays: days });
            });
        });
    }
    
    // 打开日志目录
    const btnOpenLogDir = document.getElementById('btnOpenLogDir');

    // 小黑板开关
    const blackboardToggle = document.getElementById('blackboardEnabledToggle');
    if (blackboardToggle) {
        blackboardToggle.addEventListener('change', async () => {
            await settings_save_all_local({ blackboardEnabled: blackboardToggle.checked });
            const restartModal = document.getElementById('restartModal');
            const modalMessage = restartModal?.querySelector('.modal-message');
            if (modalMessage) {
                modalMessage.textContent = window.i18n?.format_translate('settings.languageChanged') || '需要重启应用才能生效。';
            }
            if (restartModal) restartModal.classList.add('active');
        });
    }

    // 遥测功能开关
    const telemetryToggle = document.getElementById('telemetryToggle');
    if (telemetryToggle) {
        telemetryToggle.addEventListener('change', async () => {
            await settings_save_all_local({ telemetryEnabled: telemetryToggle.checked });
        });
    }

    // 界面模糊效果开关
    const blurToggle = document.getElementById('blurToggle');
    if (blurToggle) {
        blurToggle.addEventListener('change', async () => {
            document.body.classList.toggle('blur-enabled', blurToggle.checked);
            await settings_save_all_local({ blurEnabled: blurToggle.checked });
        });
    }

    // 内存自动清理开关
    const memreductCleanToggle = document.getElementById('memreductCleanToggle');
    if (memreductCleanToggle) {
        const invoke = window.__TAURI__?.core?.invoke;
        if (invoke) {
            (async () => {
                try {
                    const exists = await invoke('memreduct_check_skipuac');
                    if (exists) {
                        const r = await invoke('settings_fetch_all');
                        memreductCleanToggle.checked = r.settings?.memreductCleanEnabled !== false;
                    } else {
                        memreductCleanToggle.checked = false;
                        await invoke('settings_save_all', { settings: { memreductCleanEnabled: false } });
                    }
                } catch (_) {
                    memreductCleanToggle.checked = false;
                }
            })();
        }
        memreductCleanToggle.addEventListener('change', async () => {
            if (!invoke) return;
            if (memreductCleanToggle.checked) {
                try {
                    const exists = await invoke('memreduct_check_skipuac');
                    if (!exists) {
                        await invoke('memreduct_setup');
                    }
                    await invoke('settings_save_all', { settings: { memreductCleanEnabled: true } });
                } catch (e) {
                    console.warn('memory auto clean: 计划任务设置失败', e);
                    memreductCleanToggle.checked = false;
                }
            } else {
                await invoke('settings_save_all', { settings: { memreductCleanEnabled: false } });
            }
        });
    }

    if (btnOpenLogDir && window.__TAURI__) {
        btnOpenLogDir.addEventListener('click', async () => {
            try {
                const { invoke } = window.__TAURI__.core;
                const { openPath } = window.__TAURI__.opener;
                
                const logDir = await invoke('dir_fetch_log');
                
                await openPath(logDir);
            } catch (error) {
                console.error('打开日志目录失败:', error);
                settings_show_dialog(window.i18n?.format_translate('common.error') || '错误', window.i18n?.format_translate('settings.openLogDirFailed') || '打开日志目录失败', 'error');
            }
        });
    }
    
    const restartModal = document.getElementById('restartModal');
    const restartLater = document.getElementById('restartLater');
    const restartNow = document.getElementById('restartNow');
    
    if (restartModal && window.__TAURI__) {
        restartLater.addEventListener('click', () => {
            restartModal.classList.remove('active');
        });
        
        restartModal.addEventListener('click', (e) => {
            if (e.target === restartModal) {
                restartModal.classList.remove('active');
            }
        });
        
        restartNow.addEventListener('click', async () => {
            try {
                const { invoke } = window.__TAURI__.core;
                await invoke('app_restart_process');
            } catch (error) {
                console.error('重启失败:', error);
                settings_show_dialog(window.i18n?.format_translate('settings.saveFailed') || '保存失败', String(error), 'error');
            }
        });
    }
    
    let blobs = [];
    let animationId = null;
    let lastFrameTime = 0;
    const frameInterval = 33; // ~30 FPS
    
    function settings_calc_random_color() {
        const hue = Math.floor(Math.random() * 360);
        const saturation = 55 + Math.floor(Math.random() * 25);
        const lightness = 45 + Math.floor(Math.random() * 20);
        return `hsla(${hue}, ${saturation}%, ${lightness}%, 0.6)`;
    }
    
    function settings_create_blobs() {
        if (!auroraBg) return;
        
        auroraBg.innerHTML = '';
        blobs = [];
        
        const blobCount = 5;
        const width = window.innerWidth;
        const height = window.innerHeight;
        
        for (let i = 0; i < blobCount; i++) {
            const blob = document.createElement('div');
            blob.className = 'aurora-blob';
            
            const size = 400 + Math.random() * 300;
            blob.style.width = size + 'px';
            blob.style.height = size + 'px';
            blob.style.background = settings_calc_random_color();
            
            auroraBg.appendChild(blob);
            
            const x = Math.random() * width;
            const y = Math.random() * height;
            const angle = Math.random() * Math.PI * 2;
            const speed = 0.5 + Math.random() * 1.5;
            
            blobs.push({
                element: blob,
                x: x,
                y: y,
                vx: Math.cos(angle) * speed,
                vy: Math.sin(angle) * speed,
                speed: speed
            });
        }
    }
    
    function settings_update_blobs(currentTime) {
        if (currentTime - lastFrameTime < frameInterval) {
            animationId = requestAnimationFrame(settings_update_blobs);
            return;
        }
        lastFrameTime = currentTime;
        
        const width = window.innerWidth;
        const height = window.innerHeight;
        
        blobs.forEach(blob => {
            blob.x += blob.vx;
            blob.y += blob.vy;
            
            // 边界反弹
            if (blob.x < -200 || blob.x > width + 200) {
                blob.vx = -blob.vx;
                blob.x = Math.max(-200, Math.min(width + 200, blob.x));
            }
            if (blob.y < -200 || blob.y > height + 200) {
                blob.vy = -blob.vy;
                blob.y = Math.max(-200, Math.min(height + 200, blob.y));
            }
            
            blob.element.style.transform = `translate(${blob.x}px, ${blob.y}px)`;
        });
        
        animationId = requestAnimationFrame(settings_update_blobs);
    }
    
    function settings_start_aurora() {
        if (blobs.length === 0) {
            settings_create_blobs();
        }
        if (!animationId) {
            lastFrameTime = 0;
            settings_update_blobs(performance.now());
        }
    }
    
    function settings_hide_aurora() {
        if (animationId) {
            cancelAnimationFrame(animationId);
            animationId = null;
        }
    }
    
    if (btnClose) {
        btnClose.addEventListener('click', async () => {
            if (window.__TAURI__) {
                try {
                    const { getCurrentWindow } = window.__TAURI__.window;
                    const appWindow = getCurrentWindow();
                    await appWindow.close();
                } catch (error) {
                    console.error('关闭窗口失败:', error);
                }
            }
        });
    }

    const sidebarBtns = document.querySelectorAll('.sidebar-btn');
    const pages = document.querySelectorAll('.page');
    
    function settings_show_page(pageId) {
        pages.forEach(page => page.classList.remove('active'));
        const targetPage = document.getElementById(pageId);
        if (targetPage) {
            targetPage.classList.add('active');
        }
        
        if (auroraBg) {
            const showAurora = window.ThemeManager?.theme_fetch_aurora_effect?.() ?? true;
            if ((pageId === 'pageAbout' || pageId === 'pageUpdate') && showAurora) {
                settings_start_aurora();
                auroraBg.classList.add('active');
            } else {
                auroraBg.classList.remove('active');
                settings_hide_aurora();
            }
        }

        const btnAbout = document.getElementById('btnAbout');
        if (btnAbout) {
            if (pageId === 'pageUpdate') {
                const icon = btnAbout.querySelector('.btn-icon img');
                if (icon) icon.setAttribute('data-icon', 'arrow-left-short');
                const text = btnAbout.querySelector('.btn-text');
                if (text) text.textContent = '返回';
            } else {
                const icon = btnAbout.querySelector('.btn-icon img');
                if (icon) icon.setAttribute('data-icon', 'about');
                const text = btnAbout.querySelector('.btn-text');
                if (text) {
                    text.textContent = window.i18n?.format_translate('settings.about') || '关于';
                }
            }
            window.ThemeManager?.theme_load_icons?.();
        }
    }

    sidebarBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            sidebarBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            const pageMap = {
                'btnApp': 'pageApp',
                'btnStorage': 'pageStorage',
                'btnCanvas': 'pageCanvas',
                'btnSource': 'pageSource',
                'btnDocReader': 'pageDocReader',
                'btnTheme': 'pageTheme',
                'btnAbout': 'pageAbout'
            };
            
            const pageId = pageMap[btn.id];
            if (pageId) {
                settings_show_page(pageId);
            }
        });
    });

    const btnUpdate = document.getElementById('btnUpdate');
    if (btnUpdate) {
        btnUpdate.addEventListener('click', () => {
            settings_show_page('pageUpdate');
            setupCheckUpdate();
        });
    }

    // ==================== 更新功能（OOBE 风格） ====================

    let _latestRelease = null;
    let _downloadFilePath = null;
    let _downloadCancelled = false;
    let _updateChecked = false;
    let _updateResult = null;

    const _upd = {
        banner: document.getElementById('updateBanner'),
        notes: document.getElementById('updateNotes'),
        status: document.getElementById('updateStatus'),
        progress: document.getElementById('updateDownloadProgress'),
        progressBar: document.getElementById('updateProgressBar'),
        progressText: document.getElementById('updateProgressText'),
        btnDownload: document.getElementById('btnUpdateDownload'),
        appInfo: document.getElementById('updateAppInfo'),
    };

    function _resetUpdateUI() {
        _upd.banner.style.display = 'none';
        _upd.banner.className = 'update-banner';
        _upd.notes.style.display = 'none';
        _upd.notes.innerHTML = '';
        _upd.progress.style.display = 'none';
        _upd.btnDownload.style.display = 'none';
        _upd.appInfo.style.display = 'none';
        _downloadCancelled = false;
        _latestRelease = null;
        _downloadFilePath = null;
        _upd.btnDownload.onclick = _startDownload;
    }

    function setupCheckUpdate() {
        if (_updateChecked) {
            _showUpdateResult(_updateResult);
        } else {
            _upd.status.innerHTML = '<div class="spinner"></div><div class="update-text">' + (i18n.format_translate('settings.checkingForUpdates') || '检查更新...') + '</div>';
            _doCheckUpdate();
        }
    }

    async function _doCheckUpdate() {
        try {
            const { result } = await checkForUpdate();
            _updateChecked = true;
            _updateResult = result;
            _showUpdateResult(result);
        } catch (err) {
            console.warn('Update check failed:', err);
            _updateChecked = true;
            _updateResult = null;
            _upd.status.innerHTML = '';
            _upd.banner.className = 'update-banner banner-error';
            _upd.banner.textContent = i18n.format_translate('settings.updateCheckFailedDetail') || '检查更新失败，请稍后重试';
            _upd.banner.style.display = '';
        }
    }

    function _showUpdateResult(result) {
        _upd.status.innerHTML = '';

        if (!result) {
            _upd.banner.className = 'update-banner banner-error';
            _upd.banner.textContent = i18n.format_translate('settings.updateCheckFailedDetail') || '检查更新失败，请稍后重试';
            _upd.banner.style.display = '';
            return;
        }

        _upd.appInfo.style.display = '';

        if (result.has_update) {
            _upd.banner.style.display = 'none';
            if (result.release?.body) {
                _upd.notes.innerHTML = renderMarkdownSimple(result.release.body);
                _upd.notes.style.display = '';
            } else {
                _upd.notes.style.display = 'none';
            }
            _upd.btnDownload.style.display = '';
            _latestRelease = result.release;
        } else {
            _upd.banner.className = 'update-banner banner-latest';
            _upd.banner.textContent = i18n.format_translate('settings.alreadyLatest') || '当前已是最新版本';
            _upd.banner.style.display = '';
            if (result.current_release?.body) {
                _upd.notes.innerHTML = renderMarkdownSimple(result.current_release.body);
                _upd.notes.style.display = '';
            } else {
                _upd.notes.style.display = 'none';
            }
        }
    }

    _upd.btnDownload?.addEventListener('click', async () => {
        if (!_latestRelease || !_latestRelease.assets?.length) return;

        if (_downloadFilePath) {
            _startInstall();
            return;
        }

        try {
            const platform = await window.__TAURI__.core.invoke('app_fetch_platform');

            _upd.btnDownload.style.display = 'none';
            _upd.progress.style.display = '';
            _upd.progressBar.style.width = '0%';
            _upd.progressText.textContent = '0%';

            offProgress();
            await onProgress((p) => {
                if (_downloadCancelled) return;
                _upd.progressBar.style.width = p + '%';
                _upd.progressText.textContent = Math.round(p) + '%';
            });

            _downloadFilePath = await startDownload(_latestRelease, platform, '');

            offProgress();

            _upd.progress.style.display = 'none';
            _upd.btnDownload.style.display = '';
            _upd.btnDownload.disabled = false;
            _upd.btnDownload.textContent = i18n.format_translate('settings.installNow') || '立即安装';
        } catch (err) {
            console.error('Download failed:', err);
            _upd.btnDownload.style.display = '';
            _upd.btnDownload.disabled = false;
            _upd.btnDownload.textContent = i18n.format_translate('settings.downloadUpdate') || '下载更新';
            _upd.progress.style.display = 'none';
        }
    });

    async function _startDownload() {
        _upd.btnDownload.click();
    }

    async function _startInstall() {
        if (!_downloadFilePath) return;
        try {
            _upd.btnDownload.disabled = true;
            _upd.btnDownload.textContent = i18n.format_translate('settings.installing') || '正在安装...';
            await installDownload(_downloadFilePath);
        } catch (error) {
            console.error('安装更新失败:', error);
            _upd.btnDownload.disabled = false;
            _upd.btnDownload.textContent = i18n.format_translate('settings.installNow') || '立即安装';
            settings_show_dialog(i18n.format_translate('settings.installFailed') || '安装失败', String(error), 'error');
        }
    }

    const linkGithub = document.getElementById('linkGithub');
    if (linkGithub && window.__TAURI__) {
        linkGithub.addEventListener('click', (e) => {
            e.preventDefault();
            window.__TAURI__.opener.openUrl('https://github.com/ospneam/ViewStage');
        });
    }

    const linkLicense = document.getElementById('linkLicense');
    if (linkLicense && window.__TAURI__) {
        linkLicense.addEventListener('click', (e) => {
            e.preventDefault();
            window.__TAURI__.opener.openUrl('https://github.com/ospneam/ViewStage?tab=Apache-2.0-1-ov-file');
        });
    }

    // ==================== 开发者选项 ====================
    let dev_loaded = false;

    function developer_options_activate(navigate) {
        if (dev_loaded) return;
        dev_loaded = true;

        const devBtn = document.getElementById('btnDevOptions');
        const devPage = document.getElementById('pageDevOptions');
        if (!devBtn || !devPage) return;

        devBtn.style.display = '';

        devBtn.addEventListener('click', () => {
            sidebarBtns.forEach(b => b.classList.remove('active'));
            devBtn.classList.add('active');
            settings_show_page('pageDevOptions');
        });

        const script = document.createElement('script');
        script.src = './modules/developer/developer-options.js';
        script.onload = () => {
            if (typeof developer_options_init === 'function') {
                developer_options_init();
            }
            if (navigate) {
                devBtn.click();
            }
        };
        document.body.appendChild(script);
    }

    // 点击关于页图标5次打开
    let dev_click_count = 0;
    const logoIcon = document.querySelector('.logo-icon');
    if (logoIcon) {
        logoIcon.style.cursor = 'pointer';
        logoIcon.addEventListener('click', () => {
            dev_click_count++;
            if (dev_click_count >= 5) {
                dev_click_count = 0;
                developer_options_activate(true);
                settings_save_all_local({ developerMode: true });
            }
        });
    }

    settings_show_page('pageApp');
    document.getElementById('btnApp')?.classList.add('active');
});
