/**
 * 应用初始化流程 - DOM 构建、画布设置、摄像头初始化、事件绑定
 */
import ThemeManager from './themes/theme.js';
import { history_init_manager, history_validate_undo } from './modules/history.js';
import './modules/canvas/tile-renderer.js';
console.log('[init] module loaded, readyState:', document.readyState);

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        console.log('[init] DOMContentLoaded -> initPdfJs');
        window.main_init_pdfjs();
    });
} else {
    console.log('[init] already loaded, calling initPdfJs immediately');
    window.main_init_pdfjs();
}

// 通过 Tauri 后端初始化缓存、配置、图片保存等目录路径
async function dir_init_cache_path() {
    if (window.__TAURI__) {
        try {
            window.cacheDir = await window.__TAURI__.core.invoke('dir_fetch_cache');
            window.configDir = await window.__TAURI__.core.invoke('dir_fetch_config');
            window.cdsDir = await window.__TAURI__.core.invoke('dir_fetch_pictures_viewstage');
        } catch (error) {
            console.error('获取缓存目录失败:', error);
        }
    }
}

// 发射启动进度事件到 splashscreen 窗口
function app_emit_splash_progress(step, message) {
    if (window.__TAURI__) {
        const { emit } = window.__TAURI__.event;
        emit('splash-progress', { step, message }).catch(e => console.warn('发射启动进度失败:', e));
    }
}

// 缓存所有 DOM 元素引用，验证关键节点存在性
function dom_init_all() {
    const dom = window.dom;
    dom.canvasContainer = document.getElementById('canvasContainer');
    dom.canvasWrapper = document.getElementById('canvasWrapper');
    dom.imageElement = document.getElementById('imageElement');
    dom.cameraVideo = document.getElementById('cameraVideo');
    dom.eraserHint = document.getElementById('eraserHint');
    dom.palmEraserHint = document.getElementById('palmEraserHint');
    dom.penControlPanel = document.getElementById('penControlPanel');
    dom.settingsPanel = document.getElementById('settingsPanel');

    dom.penSizePresets = document.getElementById('penSizePresets');
    dom.penSizeValue = document.getElementById('penSizeValue');
    dom.penColorPicker = document.getElementById('penColorPicker');
    dom.eraserSizePresets = document.getElementById('eraserSizePresets');
    dom.eraserSizeValue = document.getElementById('eraserSizeValue');

    dom.btnMove = document.getElementById('btnMove');
    dom.btnComment = document.getElementById('btnComment');
    dom.btnEraser = document.getElementById('btnEraser');
    dom.btnUndo = document.getElementById('btnUndo');
    dom.btnPhoto = document.getElementById('btnPhoto');
    dom.btnLight = document.getElementById('btnLight');
    dom.btnSettings = document.getElementById('btnSettings');
    dom.btnExpand = document.getElementById('btnExpand');
    dom.btnSave = document.getElementById('btnSave');
    dom.btnMinimize = document.getElementById('btnMinimize');
    dom.btnMenu = document.getElementById('btnMenu');
    dom.btnBlackboard = document.getElementById('btnBlackboard');
    dom.blackboardPanel = document.getElementById('blackboardPanel');
    dom.blackboardCanvasWrap = document.getElementById('blackboardCanvasWrap');
    dom.blackboardCanvas = document.getElementById('blackboardCanvas');
    dom.bbClose = document.getElementById('bbClose');

    dom.documentReaderPanel = document.getElementById('documentReaderPanel');
    dom.docReaderScrollContainer = document.getElementById('docReaderScrollContainer');
    dom.drPagePrev = document.getElementById('drPagePrev');
    dom.drPageNext = document.getElementById('drPageNext');
    dom.drPageIndicator = document.getElementById('drPageIndicator');
    dom.drBtnMove = document.getElementById('drBtnMove');
    dom.drBtnComment = document.getElementById('drBtnComment');
    dom.drBtnEraser = document.getElementById('drBtnEraser');
    dom.drBtnUndo = document.getElementById('drBtnUndo');
    dom.drBtnClose = document.getElementById('drBtnClose');
    dom.drToolGroup = document.getElementById('drToolGroup');

    if (!dom.imageElement || !dom.canvasContainer) {
        console.error('必需的元素未找到');
        return false;
    }

    return true;
}

/**
 * 初始化画布：设置尺寸（屏幕 2 倍）、DPR 缩放、渲染上下文属性
 * 计算画布偏移使画布居中屏幕
 */
function canvas_init_all() {
    const dom = window.dom;
    const state = window.state;
    const DRAW_CONFIG = window.DRAW_CONFIG;
    const container = dom.canvasContainer;
    const screenW = Math.max(1, container.clientWidth);
    const screenH = Math.max(1, container.clientHeight);

    DRAW_CONFIG.screenW = screenW;
    DRAW_CONFIG.screenH = screenH;
    DRAW_CONFIG.canvasW = Math.max(1, Math.floor(screenW * 2));
    DRAW_CONFIG.canvasH = Math.max(1, Math.floor(screenH * 2));

    DRAW_CONFIG.baseDpr = window.devicePixelRatio || 1;
    DRAW_CONFIG.dpr = window.main_calc_capped_dpr(DRAW_CONFIG.baseDpr, DRAW_CONFIG.dprLimit);

    window.main_update_move_bound();

    state.canvasX = -(DRAW_CONFIG.canvasW - DRAW_CONFIG.screenW) / 2;
    state.canvasY = -(DRAW_CONFIG.canvasH - DRAW_CONFIG.screenH) / 2;

    state.cameraViewState = {
        scale: 1,
        canvasX: state.canvasX,
        canvasY: state.canvasY,
        strokeHistory: [],
        baseImageURL: null
    };

    dom.imageElement.style.width = DRAW_CONFIG.canvasW + 'px';
    dom.imageElement.style.height = DRAW_CONFIG.canvasH + 'px';
    dom.canvasWrapper.style.width = DRAW_CONFIG.canvasW + 'px';
    dom.canvasWrapper.style.height = DRAW_CONFIG.canvasH + 'px';

    window.tileRenderer.init_tiles(dom.canvasWrapper);

    if (window.batchDrawManager) {
        window.batchDrawManager.init_overlay(container, screenW, screenH, DRAW_CONFIG.dpr);
    }

    window.main_update_pen_style();
    window.main_update_eraser_hint_size();
    window.main_update_canvas_transform();
    window.main_update_canvas_bg_color(DRAW_CONFIG.canvasBgColor);

    dom.btnMove.classList.add('primary-btn');

    console.log(`画布初始化: 屏幕 ${screenW}x${screenH}, 画布 ${DRAW_CONFIG.canvasW}x${DRAW_CONFIG.canvasH}`);
}

// 从后端加载摄像头、渲染尺寸、主题等设置，并应用到当前状态
async function settings_load_camera_config() {
    if (window.__TAURI__) {
        try {
            const { invoke } = window.__TAURI__.core;
            const state = window.state;
            const DRAW_CONFIG = window.DRAW_CONFIG;
            const result = await invoke('settings_fetch_all');

            // 容错：后端返回异常时使用空对象，避免 settings 访问崩溃
            const settings = (result && typeof result === 'object' && result.settings)
                ? result.settings : {};

            if (result && result.recovered?.length) {
                show_config_recovery_dialog(result.recovered);
            }

            if (settings.defaultCamera) {
                state.defaultCameraId = settings.defaultCamera;
            }

            if (settings.cameraWidth && settings.cameraHeight) {
                state.cameraWidth = Number(settings.cameraWidth) || null;
                state.cameraHeight = Number(settings.cameraHeight) || null;
            }

            if (settings.defaultRotation !== undefined) {
                state.cameraRotation = Number(settings.defaultRotation) || 0;
            }

            // Do not load brightness/contrast from saved settings — session-only controls

            if (settings.dprLimit !== undefined) {
                DRAW_CONFIG.dprLimit = Number(settings.dprLimit) || 2;
                DRAW_CONFIG.baseDpr = window.devicePixelRatio || 1;
                DRAW_CONFIG.dpr = window.main_calc_capped_dpr(DRAW_CONFIG.baseDpr, DRAW_CONFIG.dprLimit);
            }

            if (settings.dynamicDprEnabled !== undefined) {
                DRAW_CONFIG.dynamicDprEnabled = !!settings.dynamicDprEnabled;
            }
            if (settings.dprMin !== undefined) {
                DRAW_CONFIG.dprMin = Number(settings.dprMin) || 1;
            }
            if (settings.dprMax !== undefined) {
                DRAW_CONFIG.dprMax = Number(settings.dprMax) || 4;
            }
            if (settings.dprStep !== undefined) {
                DRAW_CONFIG.dprStep = Number(settings.dprStep) || 0.5;
            }
            if (settings.overlayDpr !== undefined) {
                DRAW_CONFIG.overlayDpr = Number(settings.overlayDpr) || 1;
            }

            if (settings.penWidth !== undefined) {
                DRAW_CONFIG.penWidth = Number(settings.penWidth) || 5;
            }
            if (settings.eraserSize !== undefined) {
                DRAW_CONFIG.eraserSize = Number(settings.eraserSize) || 15;
            }

            if (settings.penColors && Array.isArray(settings.penColors)) {
                DRAW_CONFIG.penColors = settings.penColors.map(color => {
                    if (typeof color === 'object' && color !== null) {
                        const r = Number(color.r) || 0;
                        const g = Number(color.g) || 0;
                        const b = Number(color.b) || 0;
                        return window.main_calc_rgb_to_hex(r, g, b);
                    }
                    return typeof color === 'string' ? color : '#000000';
                });
                window.main_update_color_buttons();
            }
            if (settings.penSizePresets && Array.isArray(settings.penSizePresets)) {
                DRAW_CONFIG.penSizePresets = settings.penSizePresets
                    .map(v => Number(v))
                    .filter(v => v > 0 && v < 500);
                if (DRAW_CONFIG.penSizePresets.length === 0) {
                    DRAW_CONFIG.penSizePresets = [2, 5, 10, 15, 21];
                }
            }

            if (settings.eraserSizePresets && Array.isArray(settings.eraserSizePresets)) {
                DRAW_CONFIG.eraserSizePresets = settings.eraserSizePresets
                    .map(v => Number(v))
                    .filter(v => v > 0 && v < 500);
                if (DRAW_CONFIG.eraserSizePresets.length === 0) {
                    DRAW_CONFIG.eraserSizePresets = [5, 15, 25, 38, 50];
                }
            }

            if (settings.frameRateMode !== undefined) {
                if (window.batchDrawManager) {
                    window.batchDrawManager.batch_draw_update_frame_rate(settings.frameRateMode);
                }
            }

            if (settings.penEffectMode !== undefined) {
                DRAW_CONFIG.penEffectMode = settings.penEffectMode;
            }

            if (settings.eraserSpeedEnabled !== undefined) {
                DRAW_CONFIG.eraserSpeedEnabled = !!settings.eraserSpeedEnabled;
            }

            if (settings.momentumEnabled !== undefined) {
                DRAW_CONFIG.momentumEnabled = !!settings.momentumEnabled;
            }

            const themeName = settings.theme || 'com.viewstage.theme.simplify';
            if (typeof themeName === 'string' && themeName) {
                await ThemeManager.theme_update_active(themeName);
            }

            if (settings.blurEnabled === true) {
                document.body.classList.add('blur-enabled');
            }

            const canvasBgColor = ThemeManager.theme_fetch_canvas_bg_color();
            DRAW_CONFIG.canvasBgColor = canvasBgColor;
            window.main_update_canvas_bg_color(canvasBgColor);

            const blackboardEnabled = settings.blackboardEnabled !== false;
            window.__blackboardEnabled = blackboardEnabled;
            if (!blackboardEnabled && dom.btnBlackboard) {
                dom.btnBlackboard.style.display = 'none';
            } else if (dom.btnBlackboard) {
                dom.btnBlackboard.style.display = '';
            }

            const palmEraserEnabled = settings.palmEraserEnabled === true;
            DRAW_CONFIG.palmEraserEnabled = palmEraserEnabled;
            if (palmEraserEnabled) {
                try {
                    const mod = await import('./modules/palm-eraser/palm-eraser.js');
                    window.__palmEraser = mod;
                } catch (e) {
                    console.error('[init] palm eraser load error:', e);
                }
            } else {
                window.__palmEraser = null;
            }

            try {
                window.__eraser = await import('./modules/eraser/eraser.js');
            } catch (e) {
                console.error('[init] eraser load error:', e);
            }

            if (DRAW_CONFIG.eraserSpeedEnabled) {
                try {
                    window.__eraserSpeed = await import('./modules/eraser/eraser_speed.js');
                } catch (e) {
                    console.error('[init] eraser speed load error:', e);
                }
            } else {
                window.__eraserSpeed = null;
            }

            DRAW_CONFIG.developerMode = !!settings.developerMode;
            if (settings.developerMode && settings.penMinWidthRatio !== undefined) {
                DRAW_CONFIG.penMinWidthRatio = Number(settings.penMinWidthRatio) || 0.2;
            }
            if (settings.developerMode && settings.maxScaleImage !== undefined) {
                DRAW_CONFIG.maxScaleImage = Number(settings.maxScaleImage) || 3;
            }
            if (settings.developerMode && settings.gestureFrameDelta !== undefined) {
                DRAW_CONFIG.gestureFrameDelta = Number(settings.gestureFrameDelta) || 60;
            }
            if (settings.developerMode && settings.penTailDuration !== undefined) {
                DRAW_CONFIG.penTailDuration = Number(settings.penTailDuration) || 0;
            }
            // 仅在开发者模式下才检查并加载性能监视器
            if (settings.developerMode && settings.perfMonitorEnabled) {
                try {
                    window.perfMonitor = await import('./modules/developer/perf-monitor.js');
                    window.perfMonitor.perf_monitor_init(Number(settings.perfMonitorInterval) || 200);
                } catch (e) {
                    console.error('[init] perf monitor load error:', e);
                }
            }
        } catch (error) {
            console.error('加载摄像头设置失败:', error);
        }
    }
}

// 提交当前笔画到历史快照
async function draw_save_snapshot() {
    await window.main_submit_stroke();
}

// 根据可撤销状态更新撤销按钮
function history_update_button_status() {
    window.dom.btnUndo.disabled = !history_validate_undo();
}

/**
 * 主初始化入口，按顺序执行：
 * 国际化 → OOBE 检查 → PDF 文件关联 → DOM 构建 → 设置加载 → 画布初始化 →
 * 历史管理器 → 事件绑定 → 快照保存 → 摄像头检测与初始化 → 关闭启动屏
 */
async function main_init_all() {
    console.log('[init] main_init_all start');
    try {
        app_emit_splash_progress(0, '正在初始化...');
        console.log('[init] progress 0 emitted');

        if (window.i18n) {
            app_emit_splash_progress(0, '正在初始化多语言...');
            console.log('[init] init_start begin');
            await window.i18n.init_start();
            console.log('[init] init_start done');
        }

        if (window.__TAURI__) {
            app_emit_splash_progress(0, '正在检查运行环境...');
            console.log('[init] checking oobe_active');
            const isOobeActive = await window.__TAURI__.core.invoke('oobe_check_active');
            console.log('[init] oobe_active:', isOobeActive);
            if (isOobeActive) {
                console.log('[init] OOBE active, returning');
                return;
            }
            app_emit_splash_progress(0, '正在注册文件关联...');
            console.log('[init] setup_pdf_file_open');
            window.main_setup_pdf_file_open();
        }

        app_emit_splash_progress(0, '正在构建界面...');
        if (!dom_init_all()) {
            console.error('[init] dom_init_all failed');
            throw new Error('DOM 初始化失败');
        }
        console.log('[init] dom_init_all ok');

        app_emit_splash_progress(1, '正在加载设置...');
        console.log('[init] progress 1 emitted');
        app_emit_splash_progress(1, '正在初始化缓存路径...');
        console.log('[init] dir_init_cache_path begin');
        await dir_init_cache_path();
        console.log('[init] dir_init_cache_path done');

        app_emit_splash_progress(1, '正在加载摄像头配置...');
        console.log('[init] settings_load_camera_config begin');
        await settings_load_camera_config();
        console.log('[init] settings_load_camera_config done');

        app_emit_splash_progress(2, '正在加载组件...');
        console.log('[init] progress 2 emitted');
        app_emit_splash_progress(2, '正在初始化画布...');
        console.log('[init] calling canvas_init_all');
        canvas_init_all();
        app_emit_splash_progress(2, '正在加载白板...');
        console.log('[init] blackboard init');
        if (window.__blackboardEnabled !== false) {
            try {
                await window.blackboard_ensure_loaded(dom.canvasContainer);
            } catch (e) {
                console.error('[init] blackboard lazy load error:', e);
            }
        }
        app_emit_splash_progress(2, '正在加载文档阅读器...');
        console.log('[init] document reader init');
        if (window.documentReaderManager) {
            window.documentReaderManager.init();
        }
        app_emit_splash_progress(2, '正在加载历史记录...');
        console.log('[init] history_init_manager');
        history_init_manager({
            on_state_change: () => {
                history_update_button_status();
            }
        });
        app_emit_splash_progress(2, '正在绑定事件...');
        console.log('[init] setup_all_events');
        window.main_setup_all_events();
        console.log('[init] draw_save_snapshot');
        await draw_save_snapshot();

        // 展台灯可用性检测（非阻塞，检测结果写入 device.json 后不再重复 HID 枚举）
        window.__TAURI__?.core?.invoke('camera_light_detect_and_save').then(detected => {
            window.__lightAvailable = detected;
            if (!detected && dom.btnLight) {
                dom.btnLight.classList.add('no-seewo');
            }
        }).catch(() => {});

        console.log('[init] resize listener');
        window.addEventListener('resize', window.main_handle_resize);
        let blackboard_resize_timer;
        window.__handle_secondary_resize = () => {
            // blackboard resize 去抖（tile_renderer.rebuild_all 较重）
            if (window.blackboardManager && dom.canvasContainer) {
                clearTimeout(blackboard_resize_timer);
                blackboard_resize_timer = setTimeout(() => {
                    const w = Math.max(1, dom.canvasContainer.clientWidth);
                    const h = Math.max(1, dom.canvasContainer.clientHeight);
                    window.blackboardManager.resize(w, h);
                }, 100);
            }
            if (window.documentReaderManager && dom.canvasContainer) {
                const w = Math.max(1, dom.canvasContainer.clientWidth);
                const h = Math.max(1, dom.canvasContainer.clientHeight);
                window.documentReaderManager.resize(w, h);
            }
        };
        window.addEventListener('resize', window.__handle_secondary_resize);

        app_emit_splash_progress(3, '正在加载主题...');
        console.log('[init] progress 3 emitted');

        app_emit_splash_progress(4, '正在初始化摄像头...');
        console.log('[init] progress 4 emitted');

        // 摄像头检测与初始化：先枚举设备，无摄像头则直接跳过
        let is_camera_handled = false;
        let has_video_device = false;

        if (navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
            try {
                app_emit_splash_progress(4, '正在检测摄像头设备...');
                console.log('[init] enumerateDevices');
                const devices = await navigator.mediaDevices.enumerateDevices();
                console.log('[init] devices found:', devices.length);
                has_video_device = devices.some(d => d.kind === 'videoinput');
                if (!has_video_device) {
                    app_emit_splash_progress(4, '未检测到摄像头，跳过');
                    console.log('[init] no video devices, initWithoutCamera');
                    await window.main_init_without_camera(window.i18n?.format_translate('camera.notDetected') || '未检测到摄像头');
                    is_camera_handled = true;
                }
            } catch (e) {
                app_emit_splash_progress(4, '摄像头检测失败');
                console.warn('[init] enumerateDevices error:', e);
            }
        }

        if (!is_camera_handled) {
            try {
                app_emit_splash_progress(4, '正在连接摄像头...');
                console.log('[init] init_camera');
                await window.main_init_camera();
                app_emit_splash_progress(4, '摄像头已连接');
                console.log('[init] init_camera done');
            } catch (error) {
                const err_name = error?.name || '';
                const handled_codes = ['NotFoundError', 'DevicesNotFoundError', 'NotAllowedError', 'PermissionDeniedError'];
                if (handled_codes.includes(err_name)) {
                    const is_not_found = (err_name === 'NotFoundError' || err_name === 'DevicesNotFoundError');
                    const msg = is_not_found ? '未检测到摄像头' : '无摄像头权限';
                    app_emit_splash_progress(4, msg);
                    console.warn('[init] init_camera handled:', is_not_found ? 'no device' : 'no permission', error?.message);
                    const msg_key = is_not_found ? 'camera.notDetected' : 'camera.noPermission';
                    const fallback = is_not_found ? '未检测到摄像头' : '无摄像头权限';
                    await window.main_init_without_camera(
                        window.i18n?.format_translate(msg_key) || fallback
                    );
                    // NotAllowedError: 权限未缓存，延迟到首次用户交互再尝试 getUserMedia
                    if (!is_not_found && window.main_setup_deferred_camera) {
                        window.main_setup_deferred_camera();
                    }
                } else {
                    app_emit_splash_progress(4, '摄像头初始化失败');
                    console.error('[init] 摄像头初始化失败:', error?.name, error?.message);
                    await window.main_init_without_camera(
                        window.i18n?.format_translate('camera.initFailed') || '摄像头初始化失败'
                    );
                }
            }
        }

        app_emit_splash_progress(5, '正在完成...');
        console.log('[init] progress 5 emitted');

        app_emit_splash_progress(6, '');
        console.log('[init] progress 6 emitted');

        // 关闭启动屏
        if (window.__TAURI__) {
            try {
                console.log('[init] invoking window_hide_splashscreen');
                await window.__TAURI__.core.invoke('window_hide_splashscreen');
                console.log('[init] window_hide_splashscreen done');
            } catch (e) {
                console.log('[init] 关闭启动界面失败:', e);
            }
        }

        // 遥测初始化（延迟执行，不阻塞启动）
        import('./modules/telemetry/telemetry.js').then(m => {
            m.telemetryInit();
        }).catch(e => {
            console.warn('[init] telemetry init failed:', e);
        });

        // 缓存清理验证（延迟执行，不阻塞启动）
        setTimeout(() => {
            window.__TAURI__?.core?.invoke('cache_validate_auto_clear').catch(() => {});
        }, 2000);

        // 内存自动清理（延迟执行，不阻塞启动）
        setTimeout(async () => {
            const invoke = window.__TAURI__?.core?.invoke;
            if (!invoke) return;
            try {
                const usage = await invoke('memreduct_get_usage');
                if (usage > 80) {
                    console.log(`[memclean] 启动时内存使用率 ${usage}%，自动清理`);
                    await invoke('memreduct_clean_now', { mask: null });
                }
            } catch (_) {}
        }, 3000);

        // 恢复上次打开的文档（延迟执行，确保主窗口已完全加载）
        if (window.documentReaderManager) {
            setTimeout(() => {
                // 检查是否启用了恢复上次文档状态
                window.__TAURI__.core.invoke('settings_fetch_all').then(result => {
                    const settings = (result && typeof result === 'object' && result.settings)
                        ? result.settings : {};
                    window.__restoreLastDocEnabled = settings.restoreLastDoc !== false;
                    if (window.__restoreLastDocEnabled) {
                        window.documentReaderManager.restore_last_document().catch(e => {
                            console.log('[init] 恢复上次文档失败:', e);
                        });
                    }
                }).catch(e => {
                    console.log('[init] 读取设置失败:', e);
                });
            }, 500);
        }
    } catch (error) {
        console.error('初始化失败:', error);
        window.main_show_error_dialog(
            window.i18n?.format_translate('errors.initFailed') || '初始化失败',
            window.i18n?.format_translate('errors.initFailedDesc') || '应用初始化失败，请刷新页面重试'
        );
    }
}

// 配置恢复提示弹窗：当后端检测到配置异常并重置为默认值时显示
function show_config_recovery_dialog(recoveredFields) {
    const fieldLabels = {
        theme: window.i18n?.format_translate('settings.theme') || '主题',
        language: window.i18n?.format_translate('settings.language') || '语言',
        width: window.i18n?.format_translate('settings.resolution') || '分辨率',
        height: window.i18n?.format_translate('settings.resolution') || '分辨率',
        defaultCamera: window.i18n?.format_translate('settings.camera') || '摄像头',
        penColors: window.i18n?.format_translate('settings.penColors') || '画笔颜色',
        autoClearCacheDays: window.i18n?.format_translate('settings.autoClearCache') || '自动清理缓存',
    };
    const labels = recoveredFields.map(f => fieldLabels[f] || f);
    const listHtml = labels.map(l => `<div class="recovery-item">${l}</div>`).join('');

    const existing = document.getElementById('configRecoveryDialog');
    if (existing) existing.remove();

    const title = window.i18n?.format_translate('config.recoveryTitle');
    const message = window.i18n?.format_translate('config.recoveryMessage');
    const btnText = window.i18n?.format_translate('common.confirm') || '确认';
    const finalTitle = (title && !title.includes('config.')) ? title : '配置已恢复';
    const finalMessage = (message && !message.includes('config.')) ? message : '以下配置项存在异常，已自动恢复为默认值：';
    const finalBtn = btnText || '知道了';

    const dialog = document.createElement('div');
    dialog.id = 'configRecoveryDialog';
    dialog.className = 'error-dialog-overlay';
    dialog.innerHTML = `
        <div class="error-dialog">
            <div class="error-icon">🛡️</div>
            <div class="error-title">${finalTitle}</div>
            <div class="error-message">${finalMessage}</div>
            <div class="recovery-list">${listHtml}</div>
            <div class="error-buttons">
                <button class="error-btn error-btn-close" id="configRecoveryClose">${finalBtn}</button>
            </div>
        </div>
    `;
    document.body.appendChild(dialog);

    document.getElementById('configRecoveryClose')?.addEventListener('click', () => dialog.remove());
    dialog.addEventListener('click', (e) => {
        if (e.target === dialog) dialog.remove();
    });
}

window.blackboard_ensure_loaded = (async (container) => {
    if (window.blackboardManager) {
        if (!window.blackboardManager.bb_wrapper) {
            window.blackboardManager.init(container);
        }
        return window.blackboardManager;
    }
    try {
        await import('./modules/blackboard/blackboard.js');
        if (!window.blackboardManager.bb_wrapper) {
            window.blackboardManager.init(container);
        }
        return window.blackboardManager;
    } catch (e) {
        console.error('[blackboard] failed to load:', e);
        return null;
    }
});

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => main_init_all());
} else {
    main_init_all();
}

document.addEventListener('beforeunload', () => {
    window.main_delete_image_blob_urls?.();
    window.main_delete_all_pdf_blob_urls?.();
    // 如果启用了恢复上次文档状态，保存状态但不清理缓存
    if (window.documentReaderManager) {
        if (window.__restoreLastDocEnabled) {
            // 保存状态到缓存和 config
            window.documentReaderManager._save_annotations_to_cache?.();
            window.documentReaderManager._save_last_doc_state?.();
        } else {
            window.documentReaderManager.destroy?.();
            window.documentReaderManager.delete_annotation_cache_files?.();
        }
    }
    // 清理 resize 监听器
    if (window.__handle_secondary_resize) {
        window.removeEventListener('resize', window.__handle_secondary_resize);
    }
});

// 为按钮添加触摸缩放反馈，并初始化窗口最小化监听
function main_setup_touch_events() {
    const buttons = document.querySelectorAll('button');
    buttons.forEach(button => {
        button.addEventListener('touchstart', function(e) {
            this.style.transform = 'scale(0.95)';
            this.style.transition = 'transform 0.1s ease';
        }, { passive: true });

        button.addEventListener('touchend', function(e) {
            this.style.transform = '';
        }, { passive: true });

        button.addEventListener('touchcancel', function(e) {
            this.style.transform = '';
        }, { passive: true });
    });

    window.main_setup_minimize_listeners();
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', main_setup_touch_events);
} else {
    main_setup_touch_events();
}
