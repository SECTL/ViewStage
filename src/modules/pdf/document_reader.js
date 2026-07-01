/**
 * 文档阅读器管理器 —— 连续滚动、逐页批注、懒加载 + Canvas 分块渲染
 * 从顶部弹出的独立阅读面板，复用 main.js 的绘制管道与撤销系统
 * 工具栏全部在右侧，支持 IntersectionObserver 懒加载
 */

import { InputSource, PinchZoomSource } from '../gesture/index.js';
import { DocumentReaderPageManager } from './document_reader_page.js';
import {
    history_execute_command,
    history_init_manager,
    history_validate_undo,
    history_handle_undo,
    history_handle_state_change,
    history_trim_undo_front,
    history_peek_undo,
    DrawCommand,
    ClearCommand,
    history_state
} from '../history.js';

class DocumentReaderManager {
    constructor() {
        this.is_open = false;
        this.page_manager = new DocumentReaderPageManager();

        this.draw_mode = 'comment';
        this.is_drawing = false;
        this.current_stroke = null;

        this.active_page_index = -1;
        this.saved_history_state = null;
        this.folder_index = -1;

        this.last_x = 0;
        this.last_y = 0;
        this.cached_draw_type = null;
        this.cached_draw_color = null;
        this.cached_draw_line_width = null;
        this.current_pressure = 0.5;
        this.current_line_width = 5;
        this.last_line_width = 5;

        this._scroll_container = null;
        this._dr_tool_group = null;
        this._eraser_hint = null;
        this._eraser_hint_raf_id = null;
        this._eraser_hint_pending_pos = null;
        this._palm_eraser_hint = null;
        this.isPalmErasing = false;
        this.savedDrawMode = null;
        this.palmEraserSize = 60;
        this._was_camera_open_before = false;
        this._last_loaded_index = -1;
        this._page_visible_timeout_id = null;
        this._resize_raf_id = null;
        this._wheel_raf_id = null;               // 滚轮缩放 rAF 节流
        this._smooth_transform_timeout_id = null; // will-change 延迟移除
        this._gpu_cleanup_delay_ms = 800;
        this._tile_keep_distance = 5;
        this._image_keep_distance = 5;
        this._blob_keep_distance = 6;
        this._prerender_distance = 5;           // 预渲染距离：提前渲染前后5页
        this._prerender_enabled = true;         // 预渲染开关
        this._prerender_queue = [];             // 预渲染队列
        this._prerender_raf_id = null;          // 预渲染 rAF ID
        this._is_prerendering = false;          // 是否正在预渲染
        this._sidebar_virtual_threshold = 160;
        this._sidebar_item_height = 128;
        this._sidebar_overscan = 8;
        this._max_history_steps = 15;
        this._sidebar_thumbnail_cache = new Map(); // 缩略图缓存：page_index -> blob URL
        this._thumbnail_observer = null;            // 侧边栏缩略图 IntersectionObserver

        // 分块渲染相关
        this.batch_draw = null;
        this.draw_canvas_rect = null;
        this._window_resize_handler = null;

        // 缩放状态（Blackboard 风格：CSS transform translate3d + scale）
        this.dr_scale = 1;
        this.dr_canvas_x = 0;
        this.dr_canvas_y = 0;
        this.dr_move_bound = { min_x: 0, max_x: 0, min_y: 0, max_y: 0 };
        this.dr_is_dragging = false;
        this.dr_is_scaling = false;
        this.dr_start_drag_x = 0;
        this.dr_start_drag_y = 0;
        this.dr_start_scale = 1;
        this.dr_start_finger0_cx = 0;
        this.dr_start_finger0_cy = 0;
        this.dr_start_canvas_x = 0;
        this.dr_start_canvas_y = 0;
        this.dr_start_distance_sq = 0;
        this._dragFingerId = null;
        this._pinchProcessedFirstDelta = false;
        this.dr_min_scale = 0.25;
        this.dr_max_scale = 4;
        this.dr_cached_inv_scale = 1;

        // _dr_update_move_bound 缓存（避免每帧读 scrollWidth/scrollHeight 触发布局）
        this._dr_mb_cache_cw = -1;
        this._dr_mb_cache_ch = -1;
        this._dr_mb_cache_vw = -1;
        this._dr_mb_cache_vh = -1;
        this._dr_mb_cache_scale = -1;

        // 弹性 overscroll 状态
        this._dr_is_overscrolling = false;
        this._dr_overscroll_display_x = 0;
        this._dr_overscroll_display_y = 0;
        this._zoom_wrapper = null;
        this._dr_is_zooming = false;            // 缩放进行中标记，缩放结束后延迟批量重绘
        this._zoom_complete_timer = null;        // 缩放结束延迟触发重绘

        // 触摸手势优化状态
        this._touch_raf_id = null;               // 捏合缩放 rAF 节流 ID
        this._dr_touch_pending = null;            // 捏合缩放最新触摸数据（rAF 节流用）

        // transform rAF 节流（拖拽/捏合共用）
        this._dr_pending_transform = null;
        this._dr_transform_raf_id = null;
        this._dr_last_transform = { x: 0, y: 0, scale: 1 };

        // 惯性（动量）系统
        this._dr_momentum_raf = null;
        this._dr_gesture_vx = 0;
        this._dr_gesture_vy = 0;
        this._dr_last_canvas_x = 0;
        this._dr_last_canvas_y = 0;

        // 自适应 DPR（按缩放级别 + 内存压力动态降级，减少 4K 屏幕 GPU 显存占用）
        this._adaptive_dpr_enabled = true;

        // gesture 模块实例
        this._input_source = null;
        this._pinch_source = null;


        // 已初始化 tile 的页面索引集合（_dr_apply_scale 仅遍历此集合，跳过无 tile 页面）
        this._pages_with_tiles = new Set();

        // 首个页面渲染完成标记（用于隐藏加载指示）
        this._has_rendered_first_page = false;

        // 容器矩形缓存（_check_page_visibility 中避免反复 getBoundingClientRect 触发布局）
        this._cached_container_rect = null;

        // 离屏 canvas 池（复用双缓冲临时 canvas，避免频繁 GC）
        this._canvas_pool = [];
        this._canvas_pool_max = 3;

        // PDFPage 对象缓存（避免每次渲染重调 folder.pdfDoc.getPage()）
        this._pdf_page_cache = new Map();

        // 文档级 cleanup 去抖定时器
        this._doc_cleanup_timer = null;

        // 页面 offsetTop/offsetHeight 缓存（避免 _check_page_visibility 重复触发布局）
        this._page_positions = { tops: [], heights: [], stale: true };
    }

    // ====== 初始化 ======

    init(container) {
        this._scroll_container = document.getElementById('docReaderScrollContainer');
        this._dr_tool_group = document.getElementById('drToolGroup');

        this._setup_toolbar_events();
    }

    // ====== 面板管理 ======

    async open(folder_index, page_index = 0) {
        // 如果已打开其他文档，先关闭再打开新的
        if (this.is_open) {
            await this.close();
        }

        if (window.main_update_camera_state && window.state.isCameraOpen) {
            this._was_camera_open_before = true;
            await window.main_update_camera_state(false);
        } else {
            this._was_camera_open_before = false;
        }

        if (window.main_submit_stroke) {
            await window.main_submit_stroke();
        }
        if (window.batchDrawManager) {
            window.batchDrawManager.batch_draw_delete_all();
        }
        if (window.main_update_mode) {
            window.main_update_mode('move');
        }

        // 历史隔离
        window.__HISTORY_ISOLATED = true;
        this.saved_history_state = {
            undo_list: [...history_state.undo_list],
            redo_list: [...history_state.redo_list],
            on_state_change: history_state.on_state_change
        };
        history_init_manager({
            on_state_change: () => this._update_button_status()
        });

        this.folder_index = folder_index;
        const folder = window.state.fileList[folder_index];
        if (!folder || !folder.pages || folder.pages.length === 0) return;

        this.page_manager.init_from_folder_pages(folder.pages);
        this.active_page_index = page_index;
        this.page_manager.current_index = page_index;

        // 显示加载指示（首个页面渲染完成后移除）
        this._show_reader_loading();

        // 从缓存恢复批注（与 DOM 构建并行 I/O）
        const cachePromise = this._load_annotations_from_cache();

        this._build_page_dom();

        // 确保滚动容器事件已绑定（close() 可能已移除）
        this._setup_events();
        this._setup_keyboard_events();

        // 创建文档阅读器专用的橡皮擦提示元素（与 blackboard 模式一致）
        this._create_eraser_hint();

        // 默认启用移动模式，允许立即拖拽平移（不设为批注模式）
        this._set_draw_mode('move');

        // 等待缓存就绪
        const saved_state = await cachePromise;

        // 窗口 resize 时同步页面布局、批注坐标与 overlay canvas 尺寸
        this._window_resize_handler = () => {
            this._schedule_reader_resize();
            if (!this.batch_draw?._overlayCanvas) return;
            const overlay = this.batch_draw._overlayCanvas;
            const overlay_dpr = this.batch_draw._calc_overlay_dpr(this.dr_scale || 1);
            this.batch_draw._overlayDpr = overlay_dpr;
            const target_w = Math.ceil(window.innerWidth * overlay_dpr);
            const target_h = Math.ceil(window.innerHeight * overlay_dpr);
            if (overlay.width !== target_w || overlay.height !== target_h) {
                overlay.width = target_w;
                overlay.height = target_h;
                overlay.style.width = window.innerWidth + 'px';
                overlay.style.height = window.innerHeight + 'px';
            }
        };
        window.addEventListener('resize', this._window_resize_handler);

        // 恢复上次的缩放/位置/页码，或使用传入的 page_index
        let target_page = page_index;
        if (saved_state && saved_state.active_page_index >= 0) {
            target_page = saved_state.active_page_index;
            this.dr_scale = saved_state.dr_scale;
            this.dr_canvas_x = saved_state.dr_canvas_x;
            this.dr_canvas_y = saved_state.dr_canvas_y;
            this.dr_cached_inv_scale = 1 / this.dr_scale;
        }

        // 滚动到初始页面（会触发 _dr_apply_scale → _check_page_visibility → _on_page_visible → 首批页面渲染）
        await this._scroll_to_page(target_page);

        // 恢复缩放 transform（_scroll_to_page 内部会调 _dr_apply_scale，此处需重新设置）
        if (saved_state && saved_state.active_page_index >= 0) {
            this.dr_scale = saved_state.dr_scale;
            this.dr_canvas_x = saved_state.dr_canvas_x;
            this.dr_canvas_y = saved_state.dr_canvas_y;
            this.dr_cached_inv_scale = 1 / this.dr_scale;
            this._dr_update_move_bound();
            this._dr_update_canvas_position();
            this._dr_sync_transform();
        }

        this.is_open = true;

        // 空闲时间预加载附近页面的 PDFPage 到缓存（提升后续滚动/缩放渲染速度）
        this._idle_preload_pages(target_page);

        const panel = document.getElementById('documentReaderPanel');
        if (panel) panel.classList.add('active');

        this._switch_toolbar(true);
        this._update_page_indicator();
        this._sync_page_buttons();
        this._update_button_status();
    }

    async close() {
        if (!this.is_open) return;

        this._hide_reader_loading();

        // 清理 resize handler
        if (this._window_resize_handler) {
            window.removeEventListener('resize', this._window_resize_handler);
            this._window_resize_handler = null;
        }
        if (this._resize_raf_id !== null) {
            cancelAnimationFrame(this._resize_raf_id);
            this._resize_raf_id = null;
        }

        // 清理 gesture 模块
        this._teardown_gesture();

        if (this._wheel_raf_id !== null) {
            cancelAnimationFrame(this._wheel_raf_id);
            this._wheel_raf_id = null;
        }
        if (this._smooth_transform_timeout_id !== null) {
            clearTimeout(this._smooth_transform_timeout_id);
            this._smooth_transform_timeout_id = null;
        }
        if (this._zoom_complete_timer !== null) {
            clearTimeout(this._zoom_complete_timer);
            this._zoom_complete_timer = null;
        }
        this._dr_is_zooming = false;

        // 清理预渲染队列
        this._cancel_prerender();

        // 移除键盘事件监听器
        if (this._bound_handle_keydown) {
            document.removeEventListener('keydown', this._bound_handle_keydown);
            this._bound_handle_keydown = null;
        }

        this.is_open = false;
        window.__HISTORY_ISOLATED = false;

        await this._submit_stroke();
        this._hide_eraser_hint();
        this._hide_palm_eraser_hint();

        // 移除橡皮擦提示元素
        if (this._eraser_hint && this._eraser_hint.parentNode) {
            this._eraser_hint.parentNode.removeChild(this._eraser_hint);
            this._eraser_hint = null;
        }
        if (this._palm_eraser_hint && this._palm_eraser_hint.parentNode) {
            this._palm_eraser_hint.parentNode.removeChild(this._palm_eraser_hint);
            this._palm_eraser_hint = null;
        }

        // 保存所有页的批注到缓存（含全局 undo/redo 历史）
        await this._save_annotations_to_cache();

        // 保存最后打开的文档信息到 config（用于重启后恢复）
        await this._save_last_doc_state();

        // 恢复主画面历史
        if (this.saved_history_state) {
            history_state.undo_list = this.saved_history_state.undo_list;
            history_state.redo_list = this.saved_history_state.redo_list;
            history_state.on_state_change = this.saved_history_state.on_state_change;
            this.saved_history_state = null;
            history_handle_state_change();
        }

        this._destroy_lazy_loading();
        this._destroy_all_tiles();
        this._pages_with_tiles.clear();
        this.page_manager.destroy();

        // 清理 batch_draw 和 overlay_canvas
        if (this.batch_draw) {
            // 显式清空 overlay canvas 释放 GPU 纹理
            if (this.batch_draw._overlayCanvas) {
                const ctx = this.batch_draw._overlayCanvas.getContext('2d');
                if (ctx) {
                    ctx.clearRect(0, 0, this.batch_draw._overlayCanvas.width, this.batch_draw._overlayCanvas.height);
                }
                this.batch_draw._overlayCanvas.width = 0;
                this.batch_draw._overlayCanvas.height = 0;
                if (this.batch_draw._overlayCanvas.parentNode) {
                    this.batch_draw._overlayCanvas.parentNode.removeChild(this.batch_draw._overlayCanvas);
                }
            }
            this.batch_draw.batch_draw_delete_all();
            this.batch_draw = null;
        }

        // 重置缩放状态
        this.dr_scale = 1;
        this.dr_canvas_x = 0;
        this.dr_canvas_y = 0;
        this.dr_is_scaling = false;
        this.dr_is_dragging = false;
        this.dr_move_bound = { min_x: 0, max_x: 0, min_y: 0, max_y: 0 };
        this.dr_cached_inv_scale = 1;
        this._dr_mb_cache_cw = -1;
        this._dr_mb_cache_ch = -1;
        this._dr_mb_cache_vw = -1;
        this._dr_mb_cache_vh = -1;
        this._dr_last_transform = { x: 0, y: 0, scale: 1 };
        this._dr_mb_cache_scale = -1;
        this._zoom_wrapper = null;
        this._cached_container_rect = null;

        const panel = document.getElementById('documentReaderPanel');
        if (panel) panel.classList.remove('active');

        // 清理页面侧边栏
        const page_sidebar = document.getElementById('drPageSidebar');
        if (page_sidebar) page_sidebar.remove();

        // 断开缩略图 IntersectionObserver
        if (this._thumbnail_observer) {
            this._thumbnail_observer.disconnect();
            this._thumbnail_observer = null;
        }
        // 释放缩略图缓存
        this._release_sidebar_thumbnail_cache();

        // 释放 canvas 池
        this._canvas_pool.forEach(c => {
            c.width = 0;
            c.height = 0;
        });
        this._canvas_pool = [];

        // 释放缓存的 PDFPage 对象
        for (const pdf_page of this._pdf_page_cache.values()) {
            pdf_page.cleanup?.();
        }
        this._pdf_page_cache.clear();

        // 清理文档级 cleanup 定时器
        if (this._doc_cleanup_timer !== null) {
            clearTimeout(this._doc_cleanup_timer);
            this._doc_cleanup_timer = null;
        }

        if (this._scroll_container) {
            this._scroll_container.innerHTML = '';
        }

        this._switch_toolbar(false);

        if (this._was_camera_open_before && window.main_update_camera_state) {
            this._was_camera_open_before = false;
            await window.main_update_camera_state(true);
        }
    }

    /** 将所有页的批注序列化写入缓存文件（含全局 undo/redo 历史） */
    async _save_annotations_to_cache() {
        if (this.folder_index < 0) return;
        const config_dir = window.configDir;
        if (!config_dir) return;
        const cache_id = this._get_annotations_cache_id();
        if (!cache_id) return;

        const pages = this.page_manager.pages_list;
        const folder = window.state.fileList[this.folder_index];

        // 序列化全局 undo/redo 栈（保留 page_index 和 stroke _cache_uid 用于重建）
        const serialize_cmd = (cmd) => {
            if (cmd.type === 'draw') {
                return { type: 'draw', page_index: cmd.page_index, stroke_uid: cmd.stroke?._cache_uid || null };
            } else if (cmd.type === 'clear') {
                return {
                    type: 'clear',
                    page_index: cmd.page_index,
                    saved_strokes: (cmd.savedStrokeHistory || []).map(s => ({
                        _cache_uid: s._cache_uid,
                        points: s.points,
                        color: s.color,
                        lineWidth: s.lineWidth,
                        eraserSize: s.eraserSize,
                        eraserSizeRaw: s.eraserSizeRaw,
                        storedWidths: s.storedWidths,
                        bounds: s.bounds,
                        type: s.type
                    }))
                };
            }
            return null;
        };

        const today = new Date().toISOString().split('T')[0];
        const cache_data = {
            version: 4,
            folder_index: this.folder_index,
            file_md5: folder?.fileMd5 || null,
            active_page_index: this.active_page_index,
            dr_scale: this.dr_scale,
            dr_canvas_x: this.dr_canvas_x,
            dr_canvas_y: this.dr_canvas_y,
            last_open_date: today,
            pages: pages.map(p => ({
                stroke_history: p.stroke_history
            })),
            undo_stack: history_state.undo_list.map(serialize_cmd).filter(Boolean),
            redo_stack: history_state.redo_list.map(serialize_cmd).filter(Boolean)
        };

        try {
            const { writeTextFile, mkdir } = window.__TAURI__.fs;
            const doc_state_dir = `${config_dir}/doc_state`;
            try { await mkdir(doc_state_dir, { recursive: true }); } catch (_) {}
            const file_path = `${doc_state_dir}/doc_annotations_${cache_id}.json`;
            await writeTextFile(file_path, JSON.stringify(cache_data));
        } catch (err) {
            console.error('[document_reader] 保存批注缓存失败:', err);
        }
    }

    /** 从缓存文件恢复所有页的批注和全局 undo/redo 历史 */
    async _load_annotations_from_cache() {
        if (this.folder_index < 0) return null;
        const config_dir = window.configDir;
        const cache_dir = window.cacheDir;
        if (!config_dir && !cache_dir) return null;
        const cache_id = this._get_annotations_cache_id();
        if (!cache_id) return null;

        try {
            const { readTextFile } = window.__TAURI__.fs;
            let json_str;
            // 优先从 config_dir/doc_state 读取
            if (config_dir) {
                try {
                    json_str = await readTextFile(`${config_dir}/doc_state/doc_annotations_${cache_id}.json`);
                } catch (_) {}
            }
            // 兼容旧版：从缓存目录读取
            if (!json_str && cache_dir) {
                try {
                    json_str = await readTextFile(`${cache_dir}/doc_annotations_${cache_id}.json`);
                } catch (_) {}
            }
            if (!json_str) return null;
            const cache_data = JSON.parse(json_str);
            if (!cache_data || !cache_data.pages) return null;

            const pages = this.page_manager.pages_list;
            const len = Math.min(cache_data.pages.length, pages.length);

            // 恢复每页的 stroke_history
            for (let i = 0; i < len; i++) {
                const src = cache_data.pages[i];
                const dst = pages[i];
                if (src.stroke_history) dst.stroke_history = src.stroke_history;
            }

            // v3 格式：重建全局 undo/redo 栈
            if (cache_data.version >= 3 && cache_data.undo_stack) {
                this._rebuild_history_from_cache(cache_data.undo_stack, pages, history_state.undo_list);
            }
            if (cache_data.version >= 3 && cache_data.redo_stack) {
                this._rebuild_history_from_cache(cache_data.redo_stack, pages, history_state.redo_list);
            }

            // v4 格式：返回保存的视图状态
            if (cache_data.version >= 4) {
                return {
                    active_page_index: cache_data.active_page_index ?? -1,
                    dr_scale: cache_data.dr_scale ?? 1,
                    dr_canvas_x: cache_data.dr_canvas_x ?? 0,
                    dr_canvas_y: cache_data.dr_canvas_y ?? 0
                };
            }
            return null;
        } catch (err) {
            // 文件不存在或解析失败 → 无缓存，忽略
            if (err && err.code !== 'ENOENT' && !err.message?.includes('No such file')) {
                console.error('[document_reader] 恢复批注缓存失败:', err);
            }
            return null;
        }
    }

    /**
     * 从缓存数据重建 undo/redo 栈命令
     * 通过 stroke._cache_uid 匹配还原后的 stroke_history 中的对象引用
     */
    _rebuild_history_from_cache(serialized_list, pages, target_stack) {
        for (const entry of serialized_list) {
            const page = pages[entry.page_index];
            if (!page) continue;

            if (entry.type === 'draw' && entry.stroke_uid) {
                const stroke = page.stroke_history.find(s => s._cache_uid === entry.stroke_uid);
                if (stroke) {
                    const cmd = new DrawCommand({
                        stroke,
                        strokeHistoryRef: page.stroke_history,
                        redrawFn: () => this._render_all_strokes(stroke.bounds)
                    });
                    cmd.page_index = entry.page_index;
                    target_stack.push(cmd);
                }
            } else if (entry.type === 'clear' && entry.saved_strokes) {
                // 重建 ClearCommand：saved_strokes 为清空前的笔画快照
                const saved_strokes = entry.saved_strokes.map(s_data => {
                    // 尝试匹配 stroke_history 中的对象（若笔画未被清除）
                    const existing = page.stroke_history.find(s => s._cache_uid === s_data._cache_uid);
                    return existing || s_data;
                });
                const cmd = new ClearCommand({
                    savedStrokeHistory: saved_strokes,
                    strokeHistoryRef: page.stroke_history,
                    baseImageURLRef: { get value() { return null; }, set value(v) {} },
                    baseImageObjRef: { get value() { return null; }, set value(v) {} },
                    redrawFn: () => this._render_all_strokes(),
                    loadBaseImageFn: () => Promise.resolve()
                });
                cmd.page_index = entry.page_index;
                target_stack.push(cmd);
            }
        }
    }

    _get_annotations_cache_id() {
        const folder = window.state?.fileList?.[this.folder_index];
        if (folder?.fileMd5) {
            return `md5_${folder.fileMd5}`;
        }
        return this.folder_index >= 0 ? `index_${this.folder_index}` : null;
    }

    /** 保存最后打开的文档信息到 config.json（用于重启后恢复） */
    async _save_last_doc_state() {
        if (this.folder_index < 0) return;
        const folder = window.state?.fileList?.[this.folder_index];
        if (!folder) return;

        const today = new Date().toISOString().split('T')[0];
        const lastDoc = {
            folder_index: this.folder_index,
            file_name: folder.name || null,
            file_md5: folder.fileMd5 || null,
            page_index: this.active_page_index,
            dr_scale: this.dr_scale,
            dr_canvas_x: this.dr_canvas_x,
            dr_canvas_y: this.dr_canvas_y,
            last_open_date: today
        };

        try {
            if (window.__TAURI__?.core?.invoke) {
                await window.__TAURI__.core.invoke('settings_save_all', {
                    settings: { lastOpenDoc: lastDoc }
                });
            }
        } catch (err) {
            console.error('[document_reader] 保存最后文档状态失败:', err);
        }
    }

    /** 从 config.json 读取最后打开的文档信息 */
    async _load_last_doc_state() {
        try {
            if (window.__TAURI__?.core?.invoke) {
                const result = await window.__TAURI__.core.invoke('settings_fetch_all');
                return result?.settings?.lastOpenDoc || null;
            }
        } catch (err) {
            console.error('[document_reader] 读取最后文档状态失败:', err);
        }
        return null;
    }

    /** 恢复上次打开的文档（重启后调用） */
    async restore_last_document() {
        const lastDoc = await this._load_last_doc_state();
        if (!lastDoc) return false;

        const { folder_index, file_md5, page_index, dr_scale, dr_canvas_x, dr_canvas_y } = lastDoc;

        // 查找匹配的文件夹（优先用 md5 匹配，其次用 index）
        let target_index = -1;
        const fileList = window.state?.fileList;
        if (!fileList || fileList.length === 0) return false;

        if (file_md5) {
            target_index = fileList.findIndex(f => f?.fileMd5 === file_md5);
        }
        if (target_index < 0 && folder_index >= 0 && folder_index < fileList.length) {
            target_index = folder_index;
        }
        if (target_index < 0) return false;

        // 打开文档并恢复状态
        await this.open(target_index, page_index || 0);

        // open() 内部已从缓存恢复了缩放/位置，但如果缓存不存在则使用 config 中的值
        if (this.dr_scale === 1 && this.dr_canvas_x === 0 && this.dr_canvas_y === 0 &&
            (dr_scale !== 1 || dr_canvas_x !== 0 || dr_canvas_y !== 0)) {
            this.dr_scale = dr_scale || 1;
            this.dr_canvas_x = dr_canvas_x || 0;
            this.dr_canvas_y = dr_canvas_y || 0;
            this.dr_cached_inv_scale = 1 / this.dr_scale;
            this._dr_update_move_bound();
            this._dr_update_canvas_position();
            this._dr_sync_transform();
            // 恢复缩放后同步瓦片 DPR，否则已初始化的瓦片仍使用缩放=1 时的 DPR
            for (const i of this._pages_with_tiles) {
                const pd = this.page_manager.pages_list[i];
                if (pd?.tile_renderer) {
                    pd.tile_renderer.update_visible_tile_dpr(this.dr_scale, false, true);
                }
            }
        }

        return true;
    }

    async delete_annotation_cache_files() {
        if (!window.__TAURI__?.core?.invoke) return;
        try {
            await window.__TAURI__.core.invoke('cache_delete_doc_annotations');
        } catch (error) {
            console.error('[document_reader] 删除批注缓存失败:', error);
        }
    }

    // ====== DOM 构建 ======

    _build_page_dom() {
        if (!this._scroll_container) return;
        this._scroll_container.innerHTML = '';

        // 缩放包装器（transform translate3d + scale 统一缩放）
        const wrapper = document.createElement('div');
        wrapper.className = 'dr-zoom-wrapper';
        this._zoom_wrapper = wrapper;
        this._scroll_container.appendChild(wrapper);

        // 基准页面宽度（容器可见宽度减 padding），后续 resize 会动态重算
        const base_w = this._get_page_base_width();

        // ［性能］DocumentFragment 批量挂载，避免逐 page appendChild 触发布局
        const fragment = document.createDocumentFragment();

        for (let i = 0; i < this.page_manager.pages_list.length; i++) {
            const page_data = this.page_manager.pages_list[i];
            const page_div = document.createElement('div');
            page_div.className = 'doc-reader-page';
            page_div.dataset.page = i;
            page_data.page_element = page_div;

            // 页面基准尺寸（wrapper transform 负责缩放）
            this._set_page_box_size(page_data, base_w);
            page_div.style.touchAction = 'none';

            if (page_data.render_mode === 'pdfjs') {
                this._create_pdf_page_layers(page_data);

                const tiles_container = document.createElement('div');
                tiles_container.className = 'doc-reader-page-tiles';
                page_div.appendChild(tiles_container);
            } else if (page_data.loaded || page_data.image_url) {
                // 图片层（懒加载：data-src 替代 src）
                const img = document.createElement('img');
                img.alt = `第 ${page_data.page_num} 页`;
                img.loading = 'lazy';
                img.decoding = 'async';
                if (page_data.image_url) {
                    img.dataset.src = page_data.image_url;
                }
                page_div.appendChild(img);

                // Tile 容器（wrapper transform 统一缩放，tiles 不再单独 scale）
                const tiles_container = document.createElement('div');
                tiles_container.className = 'doc-reader-page-tiles';
                page_div.appendChild(tiles_container);
            } else {
                const placeholder = document.createElement('div');
                placeholder.className = 'doc-reader-page-placeholder doc-reader-page-virtual-placeholder';
                placeholder.textContent = `第 ${page_data.page_num} 页`;
                page_div.appendChild(placeholder);
                page_div.classList.add('virtualized');
                page_data.is_virtualized = true;
            }

            // overlay canvas 延迟到 _on_page_visible 创建（节省大量 getContext 开销）
            fragment.appendChild(page_div);
            page_data._visible_init_timeout = null;
        }

        wrapper.appendChild(fragment);

        // 页面已挂载到 DOM，缓存初始位置（首次 layout 后 offsetTop 可用）
        this._batch_read_page_positions();

        // 重置缩放状态，直接设置初始 transform（不触发 layout-heavy _dr_apply_scale）
        this.dr_scale = 1;
        this.dr_canvas_x = 0;
        this.dr_canvas_y = 0;
        this.dr_cached_inv_scale = 1;
        this._dr_sync_transform();
    }

    // ====== 页面位置缓存（避免 _check_page_visibility 中反复读 offsetTop 触发布局） ======

    /** 标记页面位置缓存为脏（页面尺寸变更后调用） */
    _invalidate_page_positions() {
        this._page_positions.stale = true;
    }

    /** 批量读取所有页面 offsetTop/offsetHeight, 仅触发一次布局 */
    _batch_read_page_positions() {
        const pages = this.page_manager.pages_list;
        const len = pages.length;
        const tops = new Array(len);
        const heights = new Array(len);
        for (let i = 0; i < len; i++) {
            const el = pages[i]?.page_element;
            if (el) {
                tops[i] = el.offsetTop;
                heights[i] = el.offsetHeight;
            }
        }
        this._page_positions.tops = tops;
        this._page_positions.heights = heights;
        this._page_positions.stale = false;
    }

    // ====== 懒加载（手动可见性检查，transform 替代 IntersectionObserver） ======

    _destroy_lazy_loading() {
        // 清理延迟销毁定时器
        if (this._page_visible_timeout_id !== null) {
            clearTimeout(this._page_visible_timeout_id);
            this._page_visible_timeout_id = null;
        }
    }

    /** 手动检查每页是否在视口中（用 offsetTop 数学推算，避免 getBoundingClientRect 布局抖动） */
    _check_page_visibility() {
        if (!this._scroll_container || !this.page_manager || !this._zoom_wrapper) return;

        // 容器位置在滚动/缩放中不变，缓存复用避免触发布局
        if (!this._cached_container_rect) {
            const cr = this._scroll_container.getBoundingClientRect();
            this._cached_container_rect = { top: cr.top, bottom: cr.bottom, left: cr.left };
        }
        const container_top = this._cached_container_rect.top;
        const container_bottom = this._cached_container_rect.bottom;

        // wrapper 的 getBoundingClientRect 会因 css transform 变化，需要实时获取
        const wrapper_top = this._zoom_wrapper.getBoundingClientRect().top;
        const s = this.dr_scale;

        let nearest_page = -1;
        let nearest_dist = Infinity;
        const viewport_center = (container_top + container_bottom) / 2;
        const viewport_height = container_bottom - container_top;

        // 预渲染范围：视口上下扩展 viewport_height * _prerender_distance
        const prerender_margin = viewport_height * this._prerender_distance;
        const prerender_top = container_top - prerender_margin;
        const prerender_bottom = container_bottom + prerender_margin;

        // ［性能］页面位置缓存未命中时批量读取（仅触发一次布局，避免与后续 DOM 写入交插）
        if (this._page_positions.stale) {
            this._batch_read_page_positions();
        }
        const page_tops = this._page_positions.tops;
        const page_heights = this._page_positions.heights;

        const visible_pages = [];
        const prerender_pages = [];

        for (let i = 0; i < this.page_manager.pages_list.length; i++) {
            const page_data = this.page_manager.pages_list[i];
            if (!page_data?.page_element) continue;

            const page_top = page_tops[i] ?? 0;
            const page_bottom = page_top + (page_heights[i] ?? 0);

            // visual_y = wrapper_top + offsetTop * scale（canvas_y 已包含在 wrapper_top 中）
            const visual_top = wrapper_top + page_top * s;
            const visual_bottom = wrapper_top + page_bottom * s;

            // 页面按 offsetTop 递增排列，越过预渲染下边界后不再有可见页，直接 break
            if (visual_top > prerender_bottom) break;

            const is_intersecting = visual_bottom > container_top && visual_top < container_bottom;
            const is_in_prerender_range = visual_bottom > prerender_top && visual_top < prerender_bottom;

            if (is_intersecting) {
                visible_pages.push(i);
                this._on_page_visible(i);
            } else if (is_in_prerender_range && this._prerender_enabled) {
                // 在预渲染范围内但不在视口中 → 添加到预渲染队列
                prerender_pages.push(i);
                this._on_page_hidden(i);
            } else {
                this._on_page_hidden(i);
            }

            // 找距离视口中心最近的页（无论是否可见，用于翻页同步）
            const visual_center = (visual_top + visual_bottom) / 2;
            const dist = Math.abs(visual_center - viewport_center);
            if (dist < nearest_dist) {
                nearest_dist = dist;
                nearest_page = i;
            }
        }

        // 主动预加载 keep_distance 范围内的图片（避免滚动到时白屏）
        this._preload_nearby_images(nearest_page);

        // 触发预渲染（按距离排序，优先渲染最近的页面）
        if (prerender_pages.length > 0) {
            this._schedule_prerender(prerender_pages, nearest_page);
        }

        // 同步翻页器到距离视口中心最近的页
        if (nearest_page >= 0 && nearest_page !== this.active_page_index) {
            this.active_page_index = nearest_page;
            this.page_manager.current_index = nearest_page;
            this._update_page_indicator();
            this._sync_page_buttons();

            // 切换 batch_draw 的 tileRenderer 引用到新页
            if (this.batch_draw && nearest_page < this.page_manager.pages_list.length) {
                const pd = this.page_manager.pages_list[nearest_page];
                if (pd.tile_renderer) {
                    this.batch_draw._tileRenderer = pd.tile_renderer;
                }
            }
        }

        // 去抖文档级 cleanup（滚动停止约 3 秒后释放 PDF.js 内部缓存）
        const active_folder = window.state?.fileList?.[this.folder_index];
        if (active_folder?.pdfDoc?.cleanup) {
            if (this._doc_cleanup_timer !== null) {
                clearTimeout(this._doc_cleanup_timer);
            }
            this._doc_cleanup_timer = setTimeout(() => {
                this._doc_cleanup_timer = null;
                try {
                    active_folder.pdfDoc.cleanup();
                } catch (_) {}
            }, 3000);
        }
    }

    /** 主动预加载 keep_distance 范围内的图片（避免滚动到时白屏） */
    _preload_nearby_images(active_page) {
        if (active_page < 0) return;
        const pages = this.page_manager.pages_list;
        const start = Math.max(0, active_page - this._image_keep_distance);
        const end = Math.min(pages.length - 1, active_page + this._image_keep_distance);

        for (let i = start; i <= end; i++) {
            if (i === active_page) continue;
            const pd = pages[i];
            if (!pd || pd.is_virtualized) continue;

            const img = pd.page_element?.querySelector('img');
            if (img && !img.hasAttribute('src') && img.dataset.src) {
                img.src = img.dataset.src;
            }

            // PDF 页面预加载
            if (!pd.loaded && this.folder_index >= 0) {
                this._load_pdf_page(i);
            }
        }
    }

    // ====== 预渲染调度 ======

    /** 调度预渲染任务（按距离排序，使用 requestIdleCallback 或 rAF） */
    _schedule_prerender(page_indices, active_page) {
        // 按距离当前页排序（优先渲染最近的页面）
        const sorted = page_indices
            .filter(i => {
                const pd = this.page_manager.pages_list[i];
                return pd && !pd.is_visible && !pd.pdf_render_promise;
            })
            .sort((a, b) => Math.abs(a - active_page) - Math.abs(b - active_page));

        // 限制预渲染队列长度（最多5页）
        this._prerender_queue = sorted.slice(0, 5);

        // 如果没有正在预渲染的任务，启动预渲染
        if (!this._is_prerendering && this._prerender_queue.length > 0) {
            this._process_prerender_queue();
        }
    }

    /** 处理预渲染队列（使用 requestIdleCallback 避免阻塞主线程） */
    _process_prerender_queue() {
        if (this._prerender_queue.length === 0 || !this._prerender_enabled) {
            this._is_prerendering = false;
            return;
        }

        this._is_prerendering = true;

        const process_next = () => {
            if (this._prerender_queue.length === 0 || !this._prerender_enabled) {
                this._is_prerendering = false;
                return;
            }

            const page_index = this._prerender_queue.shift();
            const page_data = this.page_manager.pages_list[page_index];

            // 如果页面已可见或正在渲染，跳过
            if (!page_data || page_data.is_visible || page_data.pdf_render_promise) {
                this._process_prerender_queue();
                return;
            }

            // 使用 requestIdleCallback 在空闲时预渲染
            const prerender_fn = () => {
                this._prerender_page(page_index).then(() => {
                    // 继续处理下一个
                    this._process_prerender_queue();
                });
            };

            if (window.requestIdleCallback) {
                window.requestIdleCallback(prerender_fn, { timeout: 1000 });
            } else {
                // 降级：使用 setTimeout
                setTimeout(prerender_fn, 50);
            }
        };

        process_next();
    }

    /** 预渲染单个页面（PDF 渲染 / 图片预加载） */
    async _prerender_page(page_index) {
        const page_data = this.page_manager.pages_list[page_index];
        if (!page_data || page_data.is_visible || page_data.pdf_render_promise) return;

        // 确保页面 DOM 已创建
        this._ensure_page_runtime_dom(page_index);

        // 如果是 PDF 页面，提前渲染
        if (page_data.render_mode === 'pdfjs') {
            await this._render_pdf_page_direct(page_index, false, true);
        } else {
            // 图片页面：预加载图片（不初始化 tiles，留给 _on_page_visible 处理）
            const img = page_data.page_element?.querySelector('img');
            if (img && !img.hasAttribute('src') && img.dataset.src) {
                img.src = img.dataset.src;
            }
        }
    }

    /** 翻页时预渲染目标页及相邻页 */
    _prerender_for_navigation(target_index) {
        if (!this._prerender_enabled) return;

        const pages = this.page_manager.pages_list;
        const prerender_indices = [];

        // 预渲染目标页的前后各5页
        for (let offset = -5; offset <= 5; offset++) {
            const idx = target_index + offset;
            if (idx >= 0 && idx < pages.length && idx !== this.active_page_index) {
                const pd = pages[idx];
                if (pd && !pd.is_visible && !pd.pdf_render_promise) {
                    prerender_indices.push(idx);
                }
            }
        }

        if (prerender_indices.length > 0) {
            this._schedule_prerender(prerender_indices, target_index);
        }
    }

    /** 取消所有预渲染任务 */
    _cancel_prerender() {
        this._prerender_queue = [];
        this._is_prerendering = false;
    }

    _on_page_visible(page_index) {
        const page_data = this.page_manager.pages_list[page_index];
        if (!page_data) return;
        page_data.is_visible = true;
        this._ensure_page_runtime_dom(page_index);

        // 取消待销毁的 tiles（页面快速滚回可见区域时避免闪烁）
        if (this._page_visible_timeout_id !== null) {
            clearTimeout(this._page_visible_timeout_id);
            this._page_visible_timeout_id = null;
        }

        if (page_data.render_mode === 'pdfjs') {
            this._render_pdf_page_direct(page_index);
            if (!page_data.is_tiles_initialized) {
                this._init_page_tiles(page_index);
                this._update_overlay_size(page_index);
            }
            return;
        }

        // 懒加载图片
        const img = page_data.page_element?.querySelector('img');
        const has_img_src = img?.hasAttribute('src') && img.getAttribute('src');
        if (img && !has_img_src && img.dataset.src) {
            img.src = img.dataset.src;
            img.onload = () => {
                // 图片加载后设置页面尺寸并初始化 tiles
                page_data.page_width = img.naturalWidth || img.clientWidth;
                page_data.page_height = img.naturalHeight || img.clientHeight;
                this._refresh_page_aspect(page_data);
                this._resize_page_layout(page_index, this._get_page_base_width());
                if (!page_data.is_tiles_initialized) {
                    this._init_page_tiles(page_index);
                    this._update_overlay_size(page_index);
                }
            };
        } else if (img && has_img_src) {
            // 已有图片 → 立即初始化 tiles（无延迟，避免白屏）
            page_data.page_width = img.naturalWidth || img.clientWidth;
            page_data.page_height = img.naturalHeight || img.clientHeight;
            this._refresh_page_aspect(page_data);
            this._resize_page_layout(page_index, this._get_page_base_width());
            if (!page_data.is_tiles_initialized) {
                this._init_page_tiles(page_index);
                this._update_overlay_size(page_index);
            }
        }

        // PDF 懒加载：如果页面未加载，则加载
        if (!page_data.loaded && this.folder_index >= 0) {
            this._load_pdf_page(page_index);
        }
    }

    _on_page_hidden(page_index) {
        const page_data = this.page_manager.pages_list[page_index];
        if (!page_data) return;
        page_data.is_visible = false;

        // 取消待处理的初始化定时器（快速滚动跳过该页）
        if (page_data._visible_init_timeout !== null) {
            clearTimeout(page_data._visible_init_timeout);
            page_data._visible_init_timeout = null;
        }

        // 不再取消正在进行的 PDF 渲染任务（取消会导致 tempCanvas 泄露 + 竞态条件）
        // 改为由渲染完成后的 is_visible 检查跳过 swap，渲染自然结束并归还 canvas
        page_data.pdf_render_promise = null;

        // 离开视口后延迟释放页面 GPU 资源（防抖动 + requestIdleCallback 降 GPU 峰值）
        if (this._page_visible_timeout_id !== null) {
            clearTimeout(this._page_visible_timeout_id);
        }
        this._page_visible_timeout_id = setTimeout(() => {
            this._page_visible_timeout_id = null;
            const destroy_fn = () => this._cleanup_hidden_page_gpu();
            if (window.requestIdleCallback) {
                window.requestIdleCallback(destroy_fn, { timeout: 2000 });
            } else {
                destroy_fn();
            }
        }, this._gpu_cleanup_delay_ms);
    }

    // ====== PDF 懒加载 ======

    async _load_pdf_page(page_index) {
        const page_data = this.page_manager.pages_list[page_index];
        if (!page_data || page_data.loaded) return;
        if (page_data.render_mode === 'pdfjs') {
            page_data.loaded = true;
            return this._render_pdf_page_direct(page_index);
        }
        if (page_data.loading_promise) return page_data.loading_promise;

        const folder = window.state.fileList[this.folder_index];
        if (!folder || !folder.pdfDoc) return;

        page_data.loading_promise = (async () => {
            const page_num = page_data.page_num;
            const doc_number = folder.docNumber ?? null;

            const { get_pdf_page_info } = await import('./document_loader.js');
            const result = await get_pdf_page_info(folder.pdfDoc, page_num, doc_number);

            // 更新页面数据
            page_data.image_url = null;
            page_data.thumbnail_url = null;
            page_data.render_mode = 'pdfjs';
            page_data.loaded = true;
            page_data.page_width = result.width;
            page_data.page_height = result.height;
            this._refresh_page_aspect(page_data);

            // 移除占位符
            const placeholder = page_data.page_element?.querySelector('.doc-reader-page-placeholder:not(.doc-reader-page-virtual-placeholder)');
            if (placeholder) {
                placeholder.remove();
            }

            page_data.page_element?.querySelector('img')?.remove();
            this._create_pdf_page_layers(page_data);
            if (!page_data.page_element?.querySelector('.doc-reader-page-tiles')) {
                const tiles_container = document.createElement('div');
                tiles_container.className = 'doc-reader-page-tiles';
                page_data.page_element?.appendChild(tiles_container);
            }
            this._resize_page_layout(page_index, this._get_page_base_width());

            // 更新侧边栏中的页面数据
            if (folder.pages[page_index]) {
                folder.pages[page_index].full = null;
                folder.pages[page_index].thumbnail = null;
                folder.pages[page_index].loaded = true;
                folder.pages[page_index].width = result.width;
                folder.pages[page_index].height = result.height;
                folder.pages[page_index].renderMode = 'pdfjs';
            }

            await this._render_pdf_page_direct(page_index);
        })();

        try {
            return await page_data.loading_promise;
        } catch (error) {
            console.error(`加载 PDF 页面 ${page_index + 1} 失败:`, error);
        } finally {
            page_data.loading_promise = null;
        }
    }

    // ====== TileRenderer 集成 ======

    _is_page_near_active(page_index, distance) {
        if (this.active_page_index < 0) return false;
        return Math.abs(page_index - this.active_page_index) <= distance;
    }

    _cleanup_hidden_page_gpu() {
        if (!this.page_manager?.pages_list) return;

        const pages = this.page_manager.pages_list;

        // ［性能］tile 销毁仅遍历实际有 tile 的页面，跳过大量无 tile 页
        for (const i of this._pages_with_tiles) {
            const pd = pages[i];
            if (!pd || pd.is_visible || i === this.active_page_index) continue;
            if (!this._is_page_near_active(i, this._tile_keep_distance)) {
                this._destroy_page_tiles(i);
            }
        }

        // 虚拟化 / blob 释放仍需遍历全部页（可能有图片但没有 tile）
        for (let i = 0; i < pages.length; i++) {
            const pd = pages[i];
            if (!pd || pd.is_visible || i === this.active_page_index) continue;

            if (!this._is_page_near_active(i, this._image_keep_distance)) {
                this._virtualize_page(i);
            } else if (!this._is_page_near_active(i, this._blob_keep_distance)) {
                this._release_page_blob_url(i);
            }
        }
    }

    _ensure_page_runtime_dom(page_index) {
        const page_data = this.page_manager.pages_list[page_index];
        const page_el = page_data?.page_element;
        if (!page_el) return;

        page_el.classList.remove('virtualized');
        page_data.is_virtualized = false;

        page_el.querySelectorAll('.doc-reader-page-virtual-placeholder').forEach(el => el.remove());

        if (page_data.render_mode === 'pdfjs') {
            if (!page_el.querySelector('.doc-reader-pdf-canvas')) {
                this._create_pdf_page_layers(page_data);
            }
            if (!page_el.querySelector('.doc-reader-page-tiles')) {
                const tiles_container = document.createElement('div');
                tiles_container.className = 'doc-reader-page-tiles';
                page_el.appendChild(tiles_container);
            }
            this._set_page_box_size(page_data, page_data.coord_width || this._get_page_base_width());
            return;
        }

        let img = page_el.querySelector('img');
        if (!img) {
            img = document.createElement('img');
            img.alt = `第 ${page_data.page_num} 页`;
            img.loading = 'lazy';
            img.decoding = 'async';
            page_el.prepend(img);
        }
        if (page_data.image_url) {
            img.dataset.src = page_data.image_url;
        }

        const existing_placeholder = page_el.querySelector('.doc-reader-page-placeholder:not(.doc-reader-page-virtual-placeholder)');
        if (!page_data.loaded && !existing_placeholder) {
            const placeholder = document.createElement('div');
            placeholder.className = 'doc-reader-page-placeholder';
            placeholder.textContent = `第 ${page_data.page_num} 页`;
            page_el.appendChild(placeholder);
        } else if (page_data.loaded && existing_placeholder) {
            existing_placeholder.remove();
        }

        if (!page_el.querySelector('.doc-reader-page-tiles')) {
            const tiles_container = document.createElement('div');
            tiles_container.className = 'doc-reader-page-tiles';
            page_el.appendChild(tiles_container);
        }

        this._set_page_box_size(page_data, page_data.coord_width || this._get_page_base_width());
    }

    _virtualize_page(page_index) {
        const page_data = this.page_manager.pages_list[page_index];
        const page_el = page_data?.page_element;
        if (!page_el || page_data.is_virtualized || page_data.is_visible) return;

        this._destroy_page_tiles(page_index);
        if (page_data.render_mode === 'pdfjs') {
            this._release_pdf_page_render(page_index);
        } else {
            this._release_page_image(page_index);
        }

        const placeholder = document.createElement('div');
        placeholder.className = 'doc-reader-page-placeholder doc-reader-page-virtual-placeholder';
        placeholder.textContent = `第 ${page_data.page_num} 页`;

        page_el.replaceChildren(placeholder);
        page_el.classList.add('virtualized');
        page_data.is_virtualized = true;
    }

    _release_page_image(page_index) {
        const page_data = this.page_manager.pages_list[page_index];
        const img = page_data?.page_element?.querySelector('img');
        if (!img || !img.hasAttribute('src')) return;

        img.onload = null;
        img.removeAttribute('src');
        // blob URL 保留在 dataset.src，页面再次可见时复用；移除 src 后浏览器可回收解码纹理。
    }

    // ====== Canvas 池（复用离屏 canvas，减少 GC 压力） ======

    _acquire_temp_canvas(width, height) {
        let canvas = this._canvas_pool.pop();
        if (!canvas) {
            canvas = document.createElement('canvas');
        }
        canvas.width = Math.ceil(width);
        canvas.height = Math.ceil(height);
        return canvas;
    }

    _release_temp_canvas(canvas) {
        if (!canvas) return;
        if (this._canvas_pool.length < this._canvas_pool_max) {
            this._canvas_pool.push(canvas);
        }
    }

    _create_pdf_page_layers(page_data) {
        const page_el = page_data?.page_element;
        if (!page_el) return;

        if (!page_el.querySelector('.doc-reader-pdf-canvas')) {
            const canvas = document.createElement('canvas');
            canvas.className = 'doc-reader-pdf-canvas';
            canvas.setAttribute('aria-label', `第 ${page_data.page_num} 页`);
            page_el.appendChild(canvas);
            page_data.pdf_canvas = canvas;
        } else {
            page_data.pdf_canvas = page_el.querySelector('.doc-reader-pdf-canvas');
        }

        // 文本层已移除，清理遗留 DOM 节点
        const existing = page_el.querySelector('.doc-reader-text-layer');
        if (existing) existing.remove();
        page_data.pdf_text_layer = null;
    }

    /**
     * 根据缩放级别和内存压力计算自适应 DPR
     * @param {number} base_dpr - 基础设备像素比
     * @param {number} scale - 当前缩放级别
     * @returns {number} 降级后的 DPR
     */
    _calculate_adaptive_dpr(base_dpr, scale, is_active_page = true) {
        if (!this._adaptive_dpr_enabled) return Math.min(base_dpr, 2);

        // 极低缩放（<0.3）时才强制降 DPR=1，避免 0.5x 附近的模糊
        if (scale < 0.3) return 1;

        // 内存压力检测：堆内存超 500MB 时降级 DPR
        if (performance.memory?.usedJSHeapSize > 500 * 1024 * 1024) return 1;

        // 非当前页只在大幅放大（>3x）且不可见时才降级
        if (!is_active_page && scale > 3) return 1;

        // 使用 ceil 语义避免向下取整导致的模糊，上限 4x 防止 OOM
        const dpr = base_dpr * scale;
        const step = 0.25;
        return Math.min(Math.ceil(dpr / step) * step, 4);
    }

    async _render_pdf_page_direct(page_index, force = false, is_prerender = false) {
        const page_data = this.page_manager.pages_list[page_index];
        if (!page_data || page_data.render_mode !== 'pdfjs') return;

        // 缩放进行中不触发 PDF 重绘，由缩放结束后的批量刷新处理
        if (this._dr_is_zooming && !force) return;

        if (page_data.pdf_render_promise && !force) return page_data.pdf_render_promise;

        const folder = window.state.fileList[this.folder_index];
        if (!folder?.pdfDoc || !page_data.page_element) return;

        this._create_pdf_page_layers(page_data);
        const css_w = Math.round(parseFloat(page_data.page_element.style.width)) || page_data.page_element.clientWidth || 800;

        // 计算目标 DPR，若与上次缓存不同则强制重绘（如翻页后活跃页变更）
        const target_dpr = is_prerender ? 1 : this._calculate_adaptive_dpr(
            window.devicePixelRatio || window.DRAW_CONFIG?.dpr || 1,
            this.dr_scale,
            page_index === this.active_page_index
        );
        if (!force &&
            page_data.pdf_render_css_width === css_w &&
            page_data.pdf_render_dpr === target_dpr &&
            page_data.pdf_canvas?.width > 0) {
            return;
        }

        page_data.pdf_render_promise = (async () => {
            if (force && page_data.pdf_render_task) {
                page_data.pdf_render_task.cancel?.();
                page_data.pdf_render_task = null;
            }

            let pdf_page = this._pdf_page_cache.get(page_index);
            if (!pdf_page) {
                pdf_page = await folder.pdfDoc.getPage(page_data.page_num);
                this._pdf_page_cache.set(page_index, pdf_page);
            }
            try {
                const base_viewport = pdf_page.getViewport({ scale: 1 });
                const css_scale = css_w / base_viewport.width;
                const css_viewport = pdf_page.getViewport({ scale: css_scale });

                // 预渲染固定 1x，普通渲染沿用缓存检查阶段已算好的 target_dpr
                const render_dpr = is_prerender ? 1 : target_dpr;
                const render_viewport = pdf_page.getViewport({ scale: css_scale * render_dpr });

                page_data.page_width = base_viewport.width;
                page_data.page_height = base_viewport.height;
                this._refresh_page_aspect(page_data);

                const canvas = page_data.pdf_canvas;
                if (!canvas) return;

                const render_w = Math.ceil(render_viewport.width);
                const render_h = Math.ceil(render_viewport.height);
                const css_w_px = Math.ceil(css_viewport.width) + 'px';
                const css_h_px = Math.ceil(css_viewport.height) + 'px';

                // 离屏预渲染：先渲染到临时 canvas，保留显示 canvas 旧内容避免白屏
                const tempCanvas = this._acquire_temp_canvas(render_w, render_h);
                try {
                    const tempCtx = tempCanvas.getContext('2d', { alpha: false });
                    tempCtx.setTransform(1, 0, 0, 1, 0, 0);
                    tempCtx.fillStyle = '#fff';
                    tempCtx.fillRect(0, 0, render_w, render_h);

                    const render_task = pdf_page.render({
                        canvasContext: tempCtx,
                        viewport: render_viewport,
                        annotationMode: 0
                    });
                    page_data.pdf_render_task = render_task;
                    await render_task.promise;
                    page_data.pdf_render_task = null;

                    // 页面已不在视口 → 跳过 swap，保留旧内容（后续 virtualization 会清理）
                    if (!page_data.is_visible && !is_prerender) return;

                    // 渲染完成后原子交换到显示 canvas（resize + drawImage 在同一帧完成）
                    canvas.width = render_w;
                    canvas.height = render_h;
                    const ctx = canvas.getContext('2d', { alpha: false });
                    ctx.setTransform(1, 0, 0, 1, 0, 0);
                    ctx.drawImage(tempCanvas, 0, 0);
                    canvas.style.width = css_w_px;
                    canvas.style.height = css_h_px;

                    // 首个页面渲染完成时隐藏加载指示
                    if (!this._has_rendered_first_page) {
                        this._has_rendered_first_page = true;
                        this._hide_reader_loading();
                    }

                    page_data.pdf_render_css_width = css_w;
                    page_data.pdf_render_dpr = target_dpr;
                } finally {
                    // 确保 tempCanvas 始终归还池中，避免泄露
                    this._release_temp_canvas(tempCanvas);
                }
            } finally {
                // 缓存页不 cleanup，保持内部数据热就绪；移除缓存时才释放
            }
        })();

        try {
            return await page_data.pdf_render_promise;
        } catch (error) {
            this._hide_reader_loading();
            if (error?.name !== 'RenderingCancelledException') {
                console.error(`直接渲染 PDF 页面 ${page_index + 1} 失败:`, error);
            }
        } finally {
            page_data.pdf_render_promise = null;
        }
    }

    /** 空闲时间预加载附近页面的 PDFPage 到缓存，减少后续滚动/缩放时的 getPage() 延迟 */
    _idle_preload_pages(center_page) {
        const folder = window.state?.fileList?.[this.folder_index];
        if (!folder?.pdfDoc?.getPage) return;

        const preload_fn = () => {
            const total = this.page_manager?.pages_list?.length || 0;
            if (total === 0) return;

            const radius = 10; // 前后各 10 页，共 21 页
            const start = Math.max(0, center_page - radius);
            const end = Math.min(total - 1, center_page + radius);
            const pages_to_load = [];
            for (let i = start; i <= end; i++) {
                if (!this._pdf_page_cache.has(i)) {
                    const pd = this.page_manager.pages_list[i];
                    if (pd?.render_mode === 'pdfjs') {
                        pages_to_load.push(i);
                    }
                }
            }

            // 分批加载，每批并发 3 个，避免阻塞主线程
            const batch_size = 3;
            const load_batch = async (indices) => {
                for (let i = 0; i < indices.length; i += batch_size) {
                    const batch = indices.slice(i, i + batch_size);
                    await Promise.all(batch.map(async (idx) => {
                        if (this._pdf_page_cache.has(idx)) return;
                        if (!this.is_open) return;
                        const pd = this.page_manager.pages_list[idx];
                        if (!pd) return;
                        try {
                            const pdf_page = await folder.pdfDoc.getPage(pd.page_num);
                            if (this.is_open) {
                                this._pdf_page_cache.set(idx, pdf_page);
                            } else {
                                pdf_page.cleanup?.();
                            }
                        } catch (_) {
                            // 单个页面加载失败不影响其他页
                        }
                    }));
                }
            };

            load_batch(pages_to_load);
        };

        if (window.requestIdleCallback) {
            window.requestIdleCallback(preload_fn, { timeout: 3000 });
        } else {
            setTimeout(preload_fn, 500);
        }
    }

    _release_pdf_page_render(page_index) {
        const page_data = this.page_manager.pages_list[page_index];
        if (!page_data) return;

        if (page_data.pdf_render_task) {
            page_data.pdf_render_task.cancel?.();
            page_data.pdf_render_task = null;
        }
        if (page_data.pdf_canvas) {
            page_data.pdf_canvas.width = 0;
            page_data.pdf_canvas.height = 0;
        }
        page_data.pdf_render_css_width = 0;

        // 释放缓存的 PDFPage 对象
        const cached = this._pdf_page_cache.get(page_index);
        if (cached) {
            cached.cleanup?.();
            this._pdf_page_cache.delete(page_index);
        }
    }

    _release_page_blob_url(page_index) {
        const page_data = this.page_manager.pages_list[page_index];
        if (!page_data || page_data.is_visible || page_index === this.active_page_index) return;
        if (page_data.render_mode === 'pdfjs') return;

        const image_url = page_data.image_url;
        const thumbnail_url = page_data.thumbnail_url;
        if (!image_url && !thumbnail_url) return;

        const revoke_urls = new Set();
        if (image_url?.startsWith('blob:')) revoke_urls.add(image_url);
        if (thumbnail_url?.startsWith('blob:')) revoke_urls.add(thumbnail_url);
        if (revoke_urls.size === 0) return;

        const img = page_data.page_element?.querySelector('img');
        if (img) {
            img.onload = null;
            img.removeAttribute('src');
            img.removeAttribute('data-src');
        }

        const sidebar_img = document.querySelector(`#drPageSidebar .dr-page-sidebar-thumb[data-page="${page_index}"]`);
        if (sidebar_img) {
            sidebar_img.removeAttribute('src');
            sidebar_img.classList.add('is-loading');
            sidebar_img.closest('.dr-page-sidebar-item')?.classList.add('loading');
        }

        revoke_urls.forEach(url => URL.revokeObjectURL(url));

        page_data.image_url = null;
        page_data.thumbnail_url = null;
        page_data.loaded = false;

        const folder_page = window.state?.fileList?.[this.folder_index]?.pages?.[page_index];
        if (folder_page) {
            folder_page.full = null;
            folder_page.thumbnail = null;
            folder_page.loaded = false;
        }

        const has_placeholder = page_data.page_element?.querySelector('.doc-reader-page-placeholder');
        if (page_data.page_element && !has_placeholder) {
            const placeholder = document.createElement('div');
            placeholder.className = 'doc-reader-page-placeholder';
            placeholder.textContent = `第 ${page_data.page_num} 页`;
            page_data.page_element.appendChild(placeholder);
        }
    }

    _get_page_base_width() {
        if (!this._scroll_container) return 800;
        return Math.max(200, this._scroll_container.clientWidth - 32);
    }

    _refresh_page_aspect(page_data) {
        if (!page_data?.page_width || !page_data.page_height) return;
        page_data.aspect_ratio = page_data.page_width / page_data.page_height;
    }

    _get_page_aspect(page_data) {
        const ratio = page_data?.aspect_ratio;
        return (ratio && ratio > 0 && isFinite(ratio)) ? ratio : 0.70710678;
    }

    _set_page_box_size(page_data, width) {
        if (!page_data?.page_element) return;
        const safe_w = Math.max(200, Math.round(width));
        const aspect = this._get_page_aspect(page_data);
        const safe_h = Math.max(200, Math.round(safe_w / aspect));

        page_data.page_element.style.width = safe_w + 'px';
        page_data.page_element.style.height = safe_h + 'px';

        // 页面尺寸变更影响所有后续页面的 offsetTop，标记缓存脏
        this._invalidate_page_positions();

        const img = page_data.page_element.querySelector('img');
        if (img) {
            img.style.width = '100%';
            img.style.height = '100%';
        }

        if (page_data.pdf_canvas) {
            page_data.pdf_canvas.style.width = '100%';
            page_data.pdf_canvas.style.height = '100%';
        }

        const tiles_container = page_data.page_element.querySelector('.doc-reader-page-tiles');
        if (tiles_container) {
            tiles_container.style.width = safe_w + 'px';
            tiles_container.style.height = safe_h + 'px';
        }
    }

    _schedule_reader_resize() {
        if (!this.is_open || this._resize_raf_id !== null) return;
        this._resize_raf_id = requestAnimationFrame(() => {
            this._resize_raf_id = null;
            this._handle_reader_resize();
        });
    }

    _handle_reader_resize() {
        if (!this.is_open || !this._zoom_wrapper || !this._scroll_container) return;

        const new_w = this._get_page_base_width();
        const active = this.page_manager.pages_list[this.active_page_index]
            || this.page_manager.get_current_page();
        const active_offset = active?.page_element
            ? active.page_element.offsetTop * this.dr_scale + this.dr_canvas_y
            : null;

        const pages = this.page_manager.pages_list;

        // ［性能］全部页仅更新 DOM box 尺寸（纯 style 赋值，不触发布局），
        // 不覆盖 coord_width/coord_height —— 留给 _resize_page_layout 判断尺寸变化
        for (let i = 0; i < pages.length; i++) {
            const pd = pages[i];
            if (!pd?.page_element) continue;
            this._set_page_box_size(pd, new_w);
        }

        // 仅对已有 tile 的页执行完整 resize（含注解缩放 + tile 重建 + 重绘）
        for (const i of this._pages_with_tiles) {
            this._resize_page_layout(i, new_w);
        }

        if (active?.page_element && active_offset !== null) {
            this.dr_canvas_y = active_offset - active.page_element.offsetTop * this.dr_scale;
        }

        // 容器 rect 缓存失效，下次 _check_page_visibility 重新获取
        this._cached_container_rect = null;

        this._dr_apply_scale();
    }

    _resize_page_layout(page_index, new_w) {
        const page_data = this.page_manager.pages_list[page_index];
        if (!page_data?.page_element) return;

        const old_w = page_data.coord_width || Math.round(parseFloat(page_data.page_element.style.width)) || 0;
        const old_h = page_data.coord_height || Math.round(parseFloat(page_data.page_element.style.height)) || 0;
        this._set_page_box_size(page_data, new_w);
        const new_h = Math.round(parseFloat(page_data.page_element.style.height)) || 0;

        if (old_w > 0 && old_h > 0 && (Math.abs(old_w - new_w) >= 1 || Math.abs(old_h - new_h) >= 1)) {
            this._scale_page_annotations(page_data, new_w / old_w, old_h > 0 ? new_h / old_h : new_w / old_w);
        }

        page_data.coord_width = new_w;
        page_data.coord_height = new_h;

        if (page_data.is_tiles_initialized) {
            this._destroy_page_tiles(page_index);
            this._init_page_tiles(page_index);
        }
        if (page_data.render_mode === 'pdfjs' && page_data.is_visible) {
            this._render_pdf_page_direct(page_index, true);
        }
        this._update_overlay_size(page_index);
    }

    _scale_page_annotations(page_data, sx, sy) {
        if (!page_data || sx === 1 && sy === 1) return;
        // 实例级 WeakSet：跨调用持久化去重，防止同一 stroke 被多次缩放
        if (!this._scaled_strokes) this._scaled_strokes = new WeakSet();
        const seen = this._scaled_strokes;
        const scale_stroke = (stroke) => {
            if (!stroke || seen.has(stroke)) return;
            seen.add(stroke);
            const sw = (sx + sy) / 2;
            if (Array.isArray(stroke.points)) {
                for (const p of stroke.points) {
                    if (typeof p.fromX === 'number') p.fromX *= sx;
                    if (typeof p.toX === 'number') p.toX *= sx;
                    if (typeof p.fromY === 'number') p.fromY *= sy;
                    if (typeof p.toY === 'number') p.toY *= sy;
                }
            }
            if (stroke.bounds) {
                if (typeof stroke.bounds.minX === 'number') stroke.bounds.minX *= sx;
                if (typeof stroke.bounds.maxX === 'number') stroke.bounds.maxX *= sx;
                if (typeof stroke.bounds.minY === 'number') stroke.bounds.minY *= sy;
                if (typeof stroke.bounds.maxY === 'number') stroke.bounds.maxY *= sy;
            }
            if (typeof stroke.lineWidth === 'number') stroke.lineWidth *= sw;
            if (typeof stroke.eraserSize === 'number') stroke.eraserSize *= sw;
            if (typeof stroke.eraserSizeRaw === 'number') stroke.eraserSizeRaw *= sw;
            if (Array.isArray(stroke.storedWidths)) {
                stroke.storedWidths = stroke.storedWidths.map(w => typeof w === 'number' ? w * sw : w);
            }
        };
        const scale_command = (cmd) => {
            if (!cmd) return;
            scale_stroke(cmd.stroke);
            if (Array.isArray(cmd.savedStrokeHistory)) cmd.savedStrokeHistory.forEach(scale_stroke);
            if (Array.isArray(cmd.beforeStrokes)) cmd.beforeStrokes.forEach(scale_stroke);
            if (Array.isArray(cmd.afterStrokes)) cmd.afterStrokes.forEach(scale_stroke);
        };

        page_data.stroke_history.forEach(scale_stroke);

        // 全局 undo/redo 栈包含所有页面的命令，仅在首次调用时缩放（通过 WeakSet 去重）
        history_state.undo_list.forEach(scale_command);
        history_state.redo_list.forEach(scale_command);
    }

    /** 确保页面 tile 已初始化（延迟创建的页面在首次绘制前调用） */
    _ensure_page_tiles(page_index) {
        const page_data = this.page_manager.pages_list[page_index];
        if (!page_data?._tiles_deferred) return;
        page_data._tiles_deferred = false;
        page_data.is_tiles_initialized = false;
        page_data._tiles_force = true;
        this._init_page_tiles(page_index);
        page_data._tiles_force = false;
    }

    _init_page_tiles(page_index) {
        const page_data = this.page_manager.pages_list[page_index];
        if (!page_data || page_data.is_tiles_initialized) return;

        // 无批注时延迟创建 tile（节省 GPU 显存和 DOM 节点），首次落笔时真正初始化
        if (page_data.stroke_history.length === 0 && !page_data._tiles_force) {
            page_data.is_tiles_initialized = true;
            page_data._tiles_deferred = true;
            return;
        }

        const tiles_container = page_data.page_element?.querySelector('.doc-reader-page-tiles');
        if (!tiles_container) return;

        // tile 坐标系使用页面的 CSS 宽度（固定基准，wrapper transform 负责缩放）
        const page_el = page_data.page_element;
        const tile_w = Math.round(parseFloat(page_el.style.width) || page_el.clientWidth || 800);
        const tile_h = Math.round(parseFloat(page_el.style.height) || page_el.clientHeight || (tile_w / this._get_page_aspect(page_data)));

        tiles_container.style.width = tile_w + 'px';
        tiles_container.style.height = tile_h + 'px';
        page_data.coord_width = tile_w;
        page_data.coord_height = tile_h;

        const tile_renderer = new TileRenderer({
            canvasW: tile_w,
            canvasH: tile_h,
            strokeHistoryRef: page_data.stroke_history,
            getVisibleRect: () => this._get_page_visible_rect(page_index),
            skipBaseCache: true
        });

        tile_renderer.init_tiles(tiles_container, this.dr_scale || 1);
        page_data.tile_renderer = tile_renderer;
        page_data.is_tiles_initialized = true;
        this._pages_with_tiles.add(page_index);

        // 初始化 batch_draw（如果还没有初始化）
        if (!this.batch_draw) {
            this._init_batch_draw();
        }

        this._render_page_strokes(page_index);
    }

    _init_batch_draw() {
        // 创建覆盖层用于实时预览（固定在视口中央，不跟随滚动）
        const overlay_canvas = document.createElement('canvas');
        overlay_canvas.className = 'doc-reader-overlay-global';
        overlay_canvas.style.position = 'fixed';
        overlay_canvas.style.top = '0';
        overlay_canvas.style.left = '0';
        overlay_canvas.style.pointerEvents = 'none';
        overlay_canvas.style.zIndex = '100';
        overlay_canvas.style.willChange = 'transform';

        // 先创建 batch_draw 实例，复用 DPR 计算逻辑
        this.batch_draw = new window.RealtimeBatchDrawManager();
        const init_overlay_dpr = this.batch_draw._calc_overlay_dpr(this.dr_scale || 1);

        // 设置初始尺寸为视口大小（含 DPR，确保清晰）
        overlay_canvas.width = Math.ceil(window.innerWidth * init_overlay_dpr);
        overlay_canvas.height = Math.ceil(window.innerHeight * init_overlay_dpr);
        overlay_canvas.style.width = window.innerWidth + 'px';
        overlay_canvas.style.height = window.innerHeight + 'px';

        document.body.appendChild(overlay_canvas);

        const overlay_ctx = overlay_canvas.getContext('2d');
        overlay_ctx.imageSmoothingEnabled = false;

        // 初始化 batch_draw
        this.batch_draw._overlayDpr = init_overlay_dpr;
        this.batch_draw._overlayCanvas = overlay_canvas;
        this.batch_draw._overlayCtx = overlay_ctx;
        this.batch_draw._overlayTransformScale = 0;
        this.batch_draw._overlayTransformX = 0;
        this.batch_draw._overlayTransformY = 0;
        this.batch_draw._overlay_cached_rect_left = null;
        this.batch_draw._overlay_cached_rect_top = null;
        this.batch_draw._sync_overlay_transform = () => {
            // 文档阅读器需要根据页面位置设置变换（含缩放因子）
            if (!this.batch_draw._overlayCtx) return;
            if (this.active_page_index < 0) return;

            const page_data = this.page_manager.pages_list[this.active_page_index];
            if (!page_data?.page_element) return;

            // 使用 overlay 自身动态 DPR，替代固定 Math.min(dpr, 1) 导致的模糊
            const overlay_dpr = this.batch_draw._overlayDpr || 1;

            // 缓存 rect 避免每次绘制都调用 getBoundingClientRect
            const now = performance.now();
            if (this.batch_draw._overlay_cached_rect_left !== null &&
                now - (this.batch_draw._overlay_last_rect_time || 0) < 16) {
                // 16ms 内复用缓存（约 60fps）
                const s = this.dr_scale;
                this.batch_draw._overlayCtx.setTransform(
                    overlay_dpr * s, 0, 0, overlay_dpr * s,
                    this.batch_draw._overlay_cached_rect_left * overlay_dpr,
                    this.batch_draw._overlay_cached_rect_top * overlay_dpr
                );
                return;
            }

            const rect = page_data.page_element.getBoundingClientRect();

            this.batch_draw._overlay_cached_rect_left = rect.left;
            this.batch_draw._overlay_cached_rect_top = rect.top;
            this.batch_draw._overlay_last_rect_time = now;

            // 设置变换，将页面坐标转换为视口坐标（含缩放因子）
            const s = this.dr_scale;
            this.batch_draw._overlayCtx.setTransform(
                overlay_dpr * s, 0, 0, overlay_dpr * s,
                rect.left * overlay_dpr, rect.top * overlay_dpr
            );
        };

        if (window.DRAW_CONFIG.frameRateMode) {
            this.batch_draw.batch_draw_update_frame_rate(window.DRAW_CONFIG.frameRateMode);
        }
    }

    _destroy_page_tiles(page_index) {
        const page_data = this.page_manager.pages_list[page_index];
        if (!page_data) return;

        if (page_data.tile_renderer) {
            // 显式清空每个 tile canvas 的 context，释放 GPU 纹理
            for (const info of page_data.tile_renderer.tileInfos || []) {
                if (info.ctx) {
                    info.ctx.clearRect(0, 0, info.canvas?.width || 0, info.canvas?.height || 0);
                }
                if (info.canvas) {
                    info.canvas.width = 0;
                    info.canvas.height = 0;
                }
            }
            page_data.tile_renderer.destroy();
            page_data.tile_renderer = null;
        }

        // 清理历史版本可能已创建的 per-page overlay canvas，避免滚动大量页面后驻留纹理
        if (page_data.overlay_canvas) {
            const ctx = page_data.overlay_canvas.getContext('2d');
            if (ctx) {
                ctx.clearRect(0, 0, page_data.overlay_canvas.width, page_data.overlay_canvas.height);
            }
            page_data.overlay_canvas.width = 0;
            page_data.overlay_canvas.height = 0;
            if (page_data.overlay_canvas.parentNode) {
                page_data.overlay_canvas.parentNode.removeChild(page_data.overlay_canvas);
            }
        }
        page_data.overlay_canvas = null;
        page_data.overlay_ctx = null;
        page_data._overlay_cached_w = 0;
        page_data._overlay_cached_h = 0;

        const tiles_container = page_data.page_element?.querySelector('.doc-reader-page-tiles');
        if (tiles_container) tiles_container.innerHTML = '';
        page_data.is_tiles_initialized = false;
        this._pages_with_tiles.delete(page_index);
    }

    _destroy_all_tiles() {
        for (let i = 0; i < this.page_manager.pages_list.length; i++) {
            this._destroy_page_tiles(i);
        }
    }

    _get_page_visible_rect(page_index) {
        const page_data = this.page_manager.pages_list[page_index];
        if (!page_data?.page_element) {
            return { x: 0, y: 0, width: page_data?.page_width || 800, height: page_data?.page_height || 600 };
        }
        const rect = page_data.page_element.getBoundingClientRect();
        const container_rect = this._scroll_container?.getBoundingClientRect();
        if (!container_rect) {
            return { x: 0, y: 0, width: rect.width, height: rect.height };
        }

        const visible_left = Math.max(0, container_rect.left - rect.left);
        const visible_top = Math.max(0, container_rect.top - rect.top);
        const visible_right = Math.min(rect.width, container_rect.right - rect.left);
        const visible_bottom = Math.min(rect.height, container_rect.bottom - rect.top);
        const inv = this.dr_cached_inv_scale || 1;

        return {
            x: visible_left * inv,
            y: visible_top * inv,
            width: Math.max(0, visible_right - visible_left) * inv,
            height: Math.max(0, visible_bottom - visible_top) * inv
        };
    }

    _update_overlay_size(page_index) {
        const page_data = this.page_manager.pages_list[page_index];
        if (!page_data?.overlay_canvas || !page_data.page_element) return;

        const rect = page_data.page_element.getBoundingClientRect();
        const w = Math.ceil(rect.width);
        const h = Math.ceil(rect.height);

        // 缓存尺寸，避免重复 resize 触发 GPU 纹理重建
        if (page_data._overlay_cached_w === w && page_data._overlay_cached_h === h) return;
        page_data._overlay_cached_w = w;
        page_data._overlay_cached_h = h;

        // overlay 仅用于实时预览，DPR=1 足够，节省 GPU 显存
        page_data.overlay_canvas.width = w;
        page_data.overlay_canvas.height = h;
        page_data.overlay_canvas.style.width = w + 'px';
        page_data.overlay_canvas.style.height = h + 'px';
        page_data.overlay_ctx.imageSmoothingEnabled = false;
    }

    // ====== 批注渲染 ======

    _render_page_strokes(page_index) {
        const page_data = this.page_manager.pages_list[page_index];
        if (!page_data?.tile_renderer) return;

        page_data.tile_renderer._strokeHistoryRef = page_data.stroke_history;

        // 无笔画时跳过全量重建，减少 GPU 开销（skipBaseCache=true 时 tiles 只有笔画内容）
        if (page_data.stroke_history.length === 0) {
            return;
        }

        page_data.tile_renderer.mark_strokes_changed();
        page_data.tile_renderer.mark_all();
        page_data.tile_renderer.rebuild_all();
    }

    async _render_all_strokes(bounds) {
        const page = this.page_manager.get_current_page();
        if (!page || !page.tile_renderer) return;

        const orig_scale = window.state?.scale;
        if (window.state) window.state.scale = this.dr_scale;

        window.main_reset_context_state?.();
        page.tile_renderer._strokeHistoryRef = page.stroke_history;
        page.tile_renderer.mark_strokes_changed();

        // 优先只重建脏区域涉及的 tiles
        if (bounds && isFinite(bounds.minX) && isFinite(bounds.minY) &&
            isFinite(bounds.maxX) && isFinite(bounds.maxY)) {
            const infos = page.tile_renderer.infos_for_segment(
                bounds.minX, bounds.minY, bounds.maxX, bounds.maxY
            );
            for (const info of infos) {
                page.tile_renderer.dirty.add(info.key);
            }
        } else {
            page.tile_renderer.mark_all();
        }

        try {
            page.tile_renderer.rebuild_all();
        } finally {
            if (window.state) window.state.scale = orig_scale;
        }
    }

    // ====== 绘制事件 ======

    _setup_events() {
        if (!this._scroll_container) return;

        const input = new InputSource(this._scroll_container);
        this._input_source = input;
        input.attach();

        // ====== 输入事件（绘制、手掌擦除、拖拽平移） ======
        input.on('inputDown', async (ev) => {
            if (!this.is_open) return;
            this._dr_cancel_momentum();

            if (this.isPalmErasing) return;

            // 缩放中不处理任何状态切换，直到手势结束重置
            if (this.dr_is_scaling) return;

            // 手掌擦除 — PointerEvent 路径（触点宽高检测）
            if (window.PointerEvent && ev.originEvent?.pointerType) {
                const palmEraser = window.__palmEraser;
                const palmResult = palmEraser ? palmEraser.is_palm_by_pointer(ev.originEvent) : { isPalm: false, width: 0, height: 0 };
                if (palmResult.isPalm && (window.DRAW_CONFIG?.palmEraserEnabled !== false)) {
                    if (this.is_drawing) {
                        this.is_drawing = false;
                        this.draw_canvas_rect = null;
                        await this._submit_stroke();
                    }
                    const size = palmEraser.compute_palm_eraser_size_from_pointer(palmResult.width, palmResult.height);
                    this._start_palm_erase(ev.position.x, ev.position.y, size);
                    return;
                }
            }

            // 4+ 触点手掌检测（所有设备路径统一处理）
            if ((window.DRAW_CONFIG?.palmEraserEnabled !== false) && input.activeCount >= 4) {
                const palmEraser = window.__palmEraser;
                if (palmEraser) {
                    let isPalm = false;
                    let centerX = 0, centerY = 0;

                    if (ev.originEvent?.touches) {
                        // TouchEvent 路径
                        isPalm = palmEraser.is_palm_by_touch_count(ev.originEvent.touches);
                        if (isPalm) {
                            const c = palmEraser.get_palm_center(ev.originEvent.touches);
                            centerX = c.x;
                            centerY = c.y;
                        }
                    } else {
                        // PointerEvent 路径
                        const positions = input.getActivePositions();
                        isPalm = palmEraser.is_palm_by_positions(positions);
                        if (isPalm) {
                            const c = palmEraser.get_palm_center_from_positions(positions);
                            centerX = c.x;
                            centerY = c.y;
                        }
                    }

                    if (isPalm) {
                        if (this.is_drawing) {
                            this.is_drawing = false;
                            this.draw_canvas_rect = null;
                            await this._submit_stroke();
                        }
                        this._start_palm_erase(centerX, centerY, window.DRAW_CONFIG?.palmEraserSize || 60);
                        return;
                    }
                }
            }

            // 非第一指不处理（后续手指留给 PinchZoomSource）
            if (input.activeCount > 1) return;

            // 页面选择
            const target = ev.originEvent?.target?.closest?.('.doc-reader-page');
            if (!target) return;
            const page_index = parseInt(target.dataset.page);
            if (isNaN(page_index)) return;

            const page_data = this.page_manager.pages_list[page_index];
            if (!page_data?.is_tiles_initialized) {
                this._on_page_visible(page_index);
            }

            this.active_page_index = page_index;
            this.page_manager.switch_page(page_index);

            if (this.batch_draw) {
                this.batch_draw._overlay_cached_rect_left = null;
                this.batch_draw._overlay_cached_rect_top = null;
                this.batch_draw._tileRenderer = page_data?.tile_renderer;
            }

            this._update_page_indicator();
            this._sync_page_buttons();

            // 拖拽平移（move 模式）
            if (this.draw_mode === 'move') {
                this.dr_is_dragging = true;
                this._dragFingerId = ev.id;
                this.dr_start_drag_x = ev.position.x - this.dr_canvas_x;
                this.dr_start_drag_y = ev.position.y - this.dr_canvas_y;
                this._dr_last_canvas_x = this.dr_canvas_x;
                this._dr_last_canvas_y = this.dr_canvas_y;
                this._dr_gesture_vx = 0;
                this._dr_gesture_vy = 0;
                this._dr_enable_smooth_transform();
                return;
            }

            // 批注 / 擦除模式
            if (this.draw_mode === 'comment' || this.draw_mode === 'eraser') {
                this.is_drawing = true;
                if (this._scroll_container) {
                    this._scroll_container.style.touchAction = 'none';
                }
                const rect = target.getBoundingClientRect();
                this.draw_canvas_rect = rect;
                const inv = this.dr_cached_inv_scale;
                this.last_x = (ev.position.x - rect.left) * inv;
                this.last_y = (ev.position.y - rect.top) * inv;
                this._ensure_page_tiles(this.active_page_index);
                this._start_stroke(this.draw_mode === 'comment' ? 'draw' : 'erase');
                if (this.draw_mode === 'eraser') {
                    this._show_eraser_hint();
                    this._update_eraser_hint_position(ev.position.x, ev.position.y);
                }
            }
        });

        input.on('inputMove', (ev) => {
            if (!this.is_open) return;

            if (this.isPalmErasing) {
                this._update_palm_erase(ev.position.x, ev.position.y);
                return;
            }

            // 拖拽平移
            if (this.dr_is_dragging) {
                // 缩放进行中：拖拽与缩放锚点冲突，完全交由 pinch 处理位置
                if (this.dr_is_scaling) return;

                this.dr_canvas_x = ev.position.x - this.dr_start_drag_x;
                this.dr_canvas_y = ev.position.y - this.dr_start_drag_y;
                this._dr_update_canvas_position();
                this._dr_update_gesture_velocity();
                // 脏检查 + rAF 节流
                if (this._dr_last_transform.x !== this.dr_canvas_x ||
                    this._dr_last_transform.y !== this.dr_canvas_y ||
                    this._dr_last_transform.scale !== this.dr_scale) {
                    this._dr_sync_transform_schedule(this.dr_canvas_x, this.dr_canvas_y, this.dr_scale);
                }
                return;
            }

            if (!this.is_drawing || this.active_page_index < 0) return;

            this.current_pressure = ev.originEvent?.pressure || 0.5;

            if (this.draw_mode === 'eraser') {
                this._update_eraser_hint_position(ev.position.x, ev.position.y);
            }

            const page_data = this.page_manager.pages_list[this.active_page_index];
            if (!page_data?.page_element) return;

            const rect = this.draw_canvas_rect || page_data.page_element.getBoundingClientRect();
            const inv = this.dr_cached_inv_scale;
            const x = (ev.position.x - rect.left) * inv;
            const y = (ev.position.y - rect.top) * inv;
            const dx = x - this.last_x;
            const dy = y - this.last_y;

            if (dx * dx + dy * dy > 1) {
                this._save_stroke_point(this.last_x, this.last_y, x, y, this.current_pressure);
                if (this.batch_draw && page_data.tile_renderer) {
                    this.batch_draw.batch_draw_create_command(
                        this.cached_draw_type, this.last_x, this.last_y, x, y,
                        this.cached_draw_color, this.cached_draw_line_width
                    );
                }
                this.last_x = x;
                this.last_y = y;
            }
        });

        input.on('inputUp', async (ev) => {
            if (this.isPalmErasing) {
                if (input.activeCount < 4) {
                    await this._end_palm_erase();
                }
                return;
            }

            if (this.dr_is_dragging) {
                // 缩放进行中：拖拽位置已不准确，跳过 momentum 避免抖动
                if (this.dr_is_scaling) {
                    this.dr_is_dragging = false;
                    this._dr_schedule_disable_smooth_transform();
                    return;
                }
                this.dr_is_dragging = false;
                if (this.draw_mode === 'move' && (Math.abs(this._dr_gesture_vx) > 2 || Math.abs(this._dr_gesture_vy) > 2)) {
                    this._dr_update_move_bound();
                    this._dr_update_canvas_position();
                    this._dr_start_momentum();
                } else {
                    this._dr_schedule_disable_smooth_transform();
                    this._check_page_visibility();
                }
                return;
            }

            if (!this.is_drawing) return;
            this.is_drawing = false;
            this.draw_canvas_rect = null;
            if (this.draw_mode === 'eraser') this._hide_eraser_hint();
            await this._submit_stroke();
        });

        // ====== 两指捏合缩放 ======
        const pinch = new PinchZoomSource(input);
        this._pinch_source = pinch;
        pinch.startDelayMs = (this.draw_mode !== 'move') ? 200 : 0;

        pinch.onPinchStarted = () => {
            if (!this.is_open) return;

            this._dr_cancel_momentum();
            this._dr_last_canvas_x = this.dr_canvas_x;
            this._dr_last_canvas_y = this.dr_canvas_y;
            this._dr_gesture_vx = 0;
            this._dr_gesture_vy = 0;

            // 取消当前笔画
            if (this.is_drawing) {
                this.is_drawing = false;
                this.draw_canvas_rect = null;
                this._submit_stroke();
                if (this.batch_draw) {
                    this.batch_draw.batch_draw_delete_all();
                }
            }

            this.dr_is_scaling = true;
            // 不立即杀死拖拽：fingers 可能在 tolerance 阈值内，
            // 让原始拖拽手指继续移动直到收到第一个缩放 delta
            this.dr_start_scale = this.dr_scale;
            this._pinchProcessedFirstDelta = false;

            // 以两指中点为缩放中心（替代只用 finger0，两指操作更自然）
            const positions = input.getActivePositions();
            if (positions.length >= 2) {
                const midX = (positions[0].x + positions[1].x) / 2;
                const midY = (positions[0].y + positions[1].y) / 2;
                this.dr_start_finger0_cx = (midX - this.dr_canvas_x) / this.dr_scale;
                this.dr_start_finger0_cy = (midY - this.dr_canvas_y) / this.dr_scale;
            }
            this.dr_start_canvas_x = this.dr_canvas_x;
            this.dr_start_canvas_y = this.dr_canvas_y;

            this._dr_enable_smooth_transform();
            this._dr_set_zooming();
        };

        pinch.onPinchDelta = (ev) => {
            if (!this.is_open || !this.dr_is_scaling) return;

            // 收到第一个缩放 delta：结束拖拽，切换到缩放模式
            if (!this._pinchProcessedFirstDelta) {
                this._pinchProcessedFirstDelta = true;
                this.dr_is_dragging = false;
                // 此时手指可能已移动，以当前中点为基准重算缩放中心
                this.dr_start_finger0_cx = (ev.centerX - this.dr_canvas_x) / this.dr_scale;
                this.dr_start_finger0_cy = (ev.centerY - this.dr_canvas_y) / this.dr_scale;
                this.dr_start_scale = this.dr_scale;
            }

            const unclamped_s = this.dr_start_scale * ev.scale;
            this.dr_scale = Math.max(this.dr_min_scale, Math.min(this.dr_max_scale, unclamped_s));
            this.dr_cached_inv_scale = 1 / this.dr_scale;
            if (this.batch_draw) {
                this.batch_draw._overlay_cached_rect_left = null;
                this.batch_draw._overlay_cached_rect_top = null;
            }

            if (this.dr_scale !== unclamped_s) {
                this.dr_start_finger0_cx = (ev.centerX - this.dr_canvas_x) / this.dr_scale;
                this.dr_start_finger0_cy = (ev.centerY - this.dr_canvas_y) / this.dr_scale;
                this.dr_start_scale = this.dr_scale;
            }

            this.dr_canvas_x = ev.centerX - this.dr_start_finger0_cx * this.dr_scale;
            this.dr_canvas_y = ev.centerY - this.dr_start_finger0_cy * this.dr_scale;

            this._dr_update_move_bound();
            this._dr_update_canvas_position();

            // 弹性 overscroll（仅显示层）
            const mb = this.dr_move_bound;
            this._dr_is_overscrolling = false;
            let display_x = this.dr_canvas_x;
            let display_y = this.dr_canvas_y;

            if (this.dr_canvas_x < mb.min_x) {
                display_x = mb.min_x + (this.dr_canvas_x - mb.min_x) * 0.3;
                this._dr_is_overscrolling = true;
            } else if (this.dr_canvas_x > mb.max_x) {
                display_x = mb.max_x + (this.dr_canvas_x - mb.max_x) * 0.3;
                this._dr_is_overscrolling = true;
            }

            if (this.dr_canvas_y < mb.min_y) {
                display_y = mb.min_y + (this.dr_canvas_y - mb.min_y) * 0.3;
                this._dr_is_overscrolling = true;
            } else if (this.dr_canvas_y > mb.max_y) {
                display_y = mb.max_y + (this.dr_canvas_y - mb.max_y) * 0.3;
                this._dr_is_overscrolling = true;
            }

            if (this._dr_is_overscrolling) {
                this._dr_overscroll_display_x = display_x;
                this._dr_overscroll_display_y = display_y;
            }

            this._dr_set_zooming();
            this._dr_update_gesture_velocity();

            // rAF 节流更新 transform（缩放中跳过 tile/overlay，仅写 DOM）
            this._dr_sync_transform_schedule(display_x, display_y, this.dr_scale);
        };

        pinch.onPinchCompleted = () => {
            if (!this.is_open) return;
            this.dr_is_scaling = false;
            this._dr_cancel_zoom_debounce();

            // 没有处理过缩放 delta（误触或两指平移刚过 tolerance 但未缩放）
            // 拖拽状态未丢失，无需额外处理
            if (!this._pinchProcessedFirstDelta) {
                this._pinchProcessedFirstDelta = false;
                this._dragFingerId = null;
                // 缩放结束后：如果剩一指且在 move 模式，继续拖拽
                if (input.activeCount === 1 && this.draw_mode === 'move') {
                    const ev = input.activeEvents[0];
                    if (ev) {
                        this.dr_is_dragging = true;
                        this.dr_start_drag_x = ev.position.x - this.dr_canvas_x;
                        this.dr_start_drag_y = ev.position.y - this.dr_canvas_y;
                    }
                }
                return;
            }
            this._pinchProcessedFirstDelta = false;

            // 缩放结束后：如果剩一指且在 move 模式，继续拖拽
            if (input.activeCount === 1 && this.draw_mode === 'move') {
                const ev = input.activeEvents[0];
                if (ev) {
                    this.dr_is_dragging = true;
                    this.dr_start_drag_x = ev.position.x - this.dr_canvas_x;
                    this.dr_start_drag_y = ev.position.y - this.dr_canvas_y;
                }
            } else if (input.activeCount === 0) {
                if (this._dr_is_overscrolling) {
                    this._dr_is_overscrolling = false;
                    const mb = this.dr_move_bound;
                    const snap_x = Math.max(mb.min_x, Math.min(mb.max_x, this._dr_overscroll_display_x));
                    const snap_y = Math.max(mb.min_y, Math.min(mb.max_y, this._dr_overscroll_display_y));
                    this.dr_canvas_x = snap_x;
                    this.dr_canvas_y = snap_y;
                    if (this._zoom_wrapper) {
                        this._zoom_wrapper.style.transitionDuration = '250ms';
                        this._zoom_wrapper.classList.add('smooth-transform');
                    }
                    this._dr_apply_scale();
                    if (this._zoom_wrapper) {
                        setTimeout(() => {
                            this._zoom_wrapper.classList.remove('smooth-transform');
                            this._zoom_wrapper.style.transitionDuration = '';
                        }, 250);
                    }
                    this._dr_schedule_disable_smooth_transform();
                } else if (Math.abs(this._dr_gesture_vx) > 2 || Math.abs(this._dr_gesture_vy) > 2) {
                    this._dr_update_move_bound();
                    this._dr_update_canvas_position();
                    this._dr_start_momentum();
                } else {
                    this._check_page_visibility();
                    this._dr_schedule_disable_smooth_transform();
                }
            }
        };

        // ====== 滚轮缩放（独立于 gesture 模块） ======
        this._bound_dr_handle_wheel = (e) => this._dr_handle_wheel(e);
        this._scroll_container.addEventListener('wheel', this._bound_dr_handle_wheel, { passive: false });
    }

    _teardown_gesture() {
        if (this._pinch_source) {
            this._pinch_source.destroy();
            this._pinch_source = null;
        }
        if (this._input_source) {
            this._input_source.detach();
            this._input_source = null;
        }
        if (this._bound_dr_handle_wheel) {
            this._scroll_container?.removeEventListener('wheel', this._bound_dr_handle_wheel);
            this._bound_dr_handle_wheel = null;
        }
    }

    _setup_keyboard_events() {
        this._bound_handle_keydown = (e) => this._handle_keydown(e);
        document.addEventListener('keydown', this._bound_handle_keydown);
    }

    _handle_keydown(e) {
        if (!this.is_open) return;

        // 输入框聚焦时跳过快捷键（避免干扰页面跳转输入）
        const active_tag = document.activeElement?.tagName;
        const is_input_focused = active_tag === 'INPUT' || active_tag === 'TEXTAREA';

        if (e.key === 'Escape') {
            e.preventDefault();
            // 若页面跳转输入框聚焦，由输入框自身处理 Escape
            if (is_input_focused && document.activeElement?.classList?.contains('dr-page-jump-input')) return;
            this.close();
        }

        // Ctrl+0 / Cmd+0 重置缩放
        if ((e.ctrlKey || e.metaKey) && e.key === '0') {
            e.preventDefault();
            if (this.dr_scale !== 1 || this.dr_canvas_x !== 0 || this.dr_canvas_y !== 0) {
                this.dr_scale = 1;
                this.dr_canvas_x = 0;
                this.dr_canvas_y = 0;
                this._dr_apply_scale();
            }
            return;
        }

        if (is_input_focused) return;

        // Home → 第一页，End → 最后一页
        if (e.key === 'Home') {
            e.preventDefault();
            this._scroll_to_page(0);
            this.page_manager.current_index = 0;
            this.active_page_index = 0;
            this._update_page_indicator();
            this._sync_page_buttons();
            return;
        }
        if (e.key === 'End') {
            e.preventDefault();
            const last = this.page_manager.get_page_count() - 1;
            this._scroll_to_page(last);
            this.page_manager.current_index = last;
            this.active_page_index = last;
            this._update_page_indicator();
            this._sync_page_buttons();
            return;
        }

        // PageUp → 上一页，PageDown → 下一页
        if (e.key === 'PageUp') {
            e.preventDefault();
            this.handle_page_nav_prev();
            return;
        }
        if (e.key === 'PageDown') {
            e.preventDefault();
            this.handle_page_nav_next();
            return;
        }

        // +/- 缩放（0.15 步长）
        if (e.key === '+' || e.key === '=') {
            e.preventDefault();
            this._dr_zoom_by_step(0.15);
            return;
        }
        if (e.key === '-' || e.key === '_') {
            e.preventDefault();
            this._dr_zoom_by_step(-0.15);
            return;
        }
    }

    /** 以视口中心为基准缩放指定步长 */
    _dr_zoom_by_step(delta) {
        const new_s = Math.max(this.dr_min_scale, Math.min(this.dr_max_scale, this.dr_scale + delta));
        if (new_s === this.dr_scale) return;

        const ratio = new_s / this.dr_scale;
        const cx = this._scroll_container?.clientWidth / 2 || 0;
        const cy = this._scroll_container?.clientHeight / 2 || 0;

        this.dr_canvas_x = cx - (cx - this.dr_canvas_x) * ratio;
        this.dr_canvas_y = cy - (cy - this.dr_canvas_y) * ratio;
        this.dr_scale = new_s;
        this._dr_apply_scale();
    }

    // ====== 笔画生命周期 — 复制自 main.js ======

    _start_stroke(type) {
        const DRAW_CONFIG = window.DRAW_CONFIG;
        const baseEraserSize = DRAW_CONFIG.eraserSize;
        this.current_stroke = {
            type: type,
            points: [],
            color: type === 'draw' ? DRAW_CONFIG.penColor : '#000000',
            lineWidth: type === 'draw' ? DRAW_CONFIG.penWidth : baseEraserSize,
            eraserSize: baseEraserSize,
            eraserSizeRaw: DRAW_CONFIG.eraserSize,
            eraserShape: 'square',
            ...(window.__eraserSpeed ? window.__eraserSpeed.eraser_speed_build_config(DRAW_CONFIG, 1) : { eraserSpeedEnabled: false }),
            scale: 1,
            bounds: { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity },
            variableWidths: [],
            _cache_uid: `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
        };

        this.current_pressure = 0.5;
        this.current_line_width = DRAW_CONFIG.penWidth;
        this.last_line_width = DRAW_CONFIG.penWidth;

        this.cached_draw_type = type;
        this.cached_draw_color = type === 'draw' ? DRAW_CONFIG.penColor : '#000000';
        this.cached_draw_line_width = type === 'draw' ? DRAW_CONFIG.penWidth / Math.max(0.001, this.dr_scale || 1) : baseEraserSize / Math.max(0.001, this.dr_scale || 1);

        this._eraser_speed_state = window.__eraserSpeed?.eraser_speed_create_state() ?? null;

        if (this.batch_draw) {
            this.batch_draw.batch_draw_init_start();
        }
    }

    _save_stroke_point(from_x, from_y, to_x, to_y, pressure) {
        const stroke = this.current_stroke;
        if (!stroke) return;

        const bounds = stroke.bounds;
        if (from_x < bounds.minX) bounds.minX = from_x;
        if (to_x < bounds.minX) bounds.minX = to_x;
        if (from_y < bounds.minY) bounds.minY = from_y;
        if (to_y < bounds.minY) bounds.minY = to_y;
        if (from_x > bounds.maxX) bounds.maxX = from_x;
        if (to_x > bounds.maxX) bounds.maxX = to_x;
        if (from_y > bounds.maxY) bounds.maxY = from_y;
        if (to_y > bounds.maxY) bounds.maxY = to_y;

        let currentWidth = stroke.lineWidth;
        const DRAW_CONFIG = window.DRAW_CONFIG;
        const currentScale = Math.max(0.001, this.dr_scale || 1);

        if (stroke.type === 'draw') {
            this.current_pressure = pressure;
            this.last_line_width = this.current_line_width;
            currentWidth = stroke.lineWidth * (0.9 + pressure * 0.2);
            this.current_line_width = currentWidth;
            this.cached_draw_line_width = DRAW_CONFIG.penWidth / currentScale;
        } else if (stroke.type === 'erase' && stroke.eraserSpeedEnabled) {
            currentWidth = window.__eraserSpeed.eraser_speed_update(this._eraser_speed_state, stroke, to_x, to_y);
            this.cached_draw_line_width = currentWidth;
        } else if (stroke.type === 'erase') {
            this.cached_draw_line_width = DRAW_CONFIG.eraserSize / currentScale;
        }

        stroke.variableWidths.push(currentWidth);

        stroke.points.push({ fromX: from_x, fromY: from_y, toX: to_x, toY: to_y });
    }

    async _submit_stroke() {
        if (this.current_stroke && this.current_stroke.points.length > 0) {
            if (this.batch_draw) {
                this.batch_draw.batch_draw_handle_flush();
                const penMode = window.get_pen_effect_mode ? window.get_pen_effect_mode() : 'off';
                if (penMode === 'limited' && this.batch_draw._storedWidths.length > 0) {
                    const baseW = this.current_stroke.lineWidth || 5;
                    this.batch_draw._apply_speed_taper(this.batch_draw._storedWidths, this.current_stroke.points, baseW);
                }
                const stored_widths = this.batch_draw._storedWidths;
                if (stored_widths &&
                    stored_widths.length === this.current_stroke.points.length) {
                    this.current_stroke.storedWidths = [...stored_widths];
                }
            }

            const page = this.page_manager.get_current_page();
            if (page) {
                const hw = Math.max(this.current_stroke.lineWidth || 5, this.current_stroke.eraserSize || 5) / 2;
                const raw = this.current_stroke.bounds;
                const stroke_bounds = raw ? {
                    minX: raw.minX - hw, minY: raw.minY - hw,
                    maxX: raw.maxX + hw, maxY: raw.maxY + hw
                } : null;
                const cmd = new DrawCommand({
                    stroke: this.current_stroke,
                    strokeHistoryRef: page.stroke_history,
                    redrawFn: () => this._render_all_strokes(stroke_bounds)
                });
                cmd.page_index = this.active_page_index;
                await history_execute_command(cmd, false);
                this._trim_undo_stack();

                if (page.tile_renderer) {
                    const tr = page.tile_renderer;
                    tr._strokeHistoryRef = page.stroke_history;
                    const orig_scale = window.state?.scale;
                    if (window.state) window.state.scale = this.dr_scale;
                    tr.add_stroke(this.current_stroke);
                    if (window.state) window.state.scale = orig_scale;
                } else if (page._tiles_deferred) {
                    // 延迟创建的 tile 在落笔后初始化
                    this._ensure_page_tiles(this.active_page_index);
                    if (page.tile_renderer) {
                        page.tile_renderer._strokeHistoryRef = page.stroke_history;
                        const orig_scale = window.state?.scale;
                        if (window.state) window.state.scale = this.dr_scale;
                        page.tile_renderer.add_stroke(this.current_stroke);
                        if (window.state) window.state.scale = orig_scale;
                    }
                }
            }
        }

        this.current_stroke = null;
        if (this.batch_draw) {
            await this.batch_draw.batch_draw_handle_end();
            this.batch_draw.batch_draw_delete_all();
        }
        this._update_button_status();
    }

    // ====== 撤销与清空 ======

    /** 裁剪全局 undo 栈至 _max_history_steps 上限 */
    _trim_undo_stack() {
        history_trim_undo_front(this._max_history_steps);
    }

    async handle_undo() {
        if (!history_validate_undo()) return;
        if (this.is_drawing) return;

        // 检查栈顶命令所属页面，若与当前页不同则先切换
        const top_cmd = history_peek_undo();
        if (top_cmd && typeof top_cmd.page_index === 'number' &&
            top_cmd.page_index !== this.active_page_index) {
            this.active_page_index = top_cmd.page_index;
            this.page_manager.current_index = top_cmd.page_index;
            await this._scroll_to_page(top_cmd.page_index);
            this._update_page_indicator();
            this._sync_page_buttons();
        }

        await history_handle_undo();
        this._update_button_status();
    }

    async handle_clear() {
        if (this.is_drawing) return;

        const page = this.page_manager.get_current_page();
        if (!page || page.stroke_history.length === 0) return;

        const cmd = new ClearCommand({
            savedStrokeHistory: [...page.stroke_history],
            savedBaseImageURL: null,
            strokeHistoryRef: page.stroke_history,
            baseImageURLRef: {
                get value() { return null; },
                set value(v) {}
            },
            baseImageObjRef: {
                get value() { return null; },
                set value(v) {}
            },
            redrawFn: () => this._render_all_strokes(),
            loadBaseImageFn: () => Promise.resolve()
        });
        cmd.page_index = this.active_page_index;
        await history_execute_command(cmd, false);
        this._trim_undo_stack();

        this._update_button_status();
    }

    // ====== 页面导航 ======

    async handle_page_nav_prev() {
        if (this.is_drawing) return;
        await this._submit_stroke();
        const moved = this.page_manager.nav_prev();
        if (moved) {
            await this._scroll_to_page(this.page_manager.current_index);
            this._update_page_indicator();
            this._sync_page_buttons();
            this._update_button_status();
        }
    }

    async handle_page_nav_next() {
        if (this.is_drawing) return;
        await this._submit_stroke();
        const moved = this.page_manager.nav_next();
        if (moved) {
            await this._scroll_to_page(this.page_manager.current_index);
            this._update_page_indicator();
            this._sync_page_buttons();
            this._update_button_status();
        }
    }

    async _scroll_to_page(page_index) {
        if (!this._scroll_container || !this._zoom_wrapper) return;
        const page_data = this.page_manager.pages_list[page_index];
        if (!page_data?.page_element) return;

        const page_el = page_data.page_element;
        const container = this._scroll_container;
        const s = this.dr_scale;

        // 计算页面中心在 wrapper 中的偏移（CSS 像素，transform 前）
        const page_center_y = page_el.offsetTop + page_el.offsetHeight / 2;
        const viewport_center_y = container.clientHeight / 2;

        // 设置 canvas_y 使页面中心居中视口（无需动画，_dr_apply_scale 会 clamp 边界）
        this.dr_canvas_x = 0;
        this.dr_canvas_y = viewport_center_y - page_center_y * s;
        this._dr_apply_scale();

        // 翻页后预渲染相邻页面
        this._prerender_for_navigation(page_index);
    }

    _update_page_indicator() {
        const el = document.getElementById('drPageIndicator');
        if (el) {
            el.textContent = `${this.page_manager.current_index + 1} / ${this.page_manager.get_page_count()}`;
        }
    }

    /** 点击页码指示器时显示页码跳转输入框 */
    _show_page_jump_input() {
        const el = document.getElementById('drPageIndicator');
        if (!el || el.querySelector('input')) return;

        const max_page = this.page_manager.get_page_count();
        const current_page = this.page_manager.current_index + 1;

        const input = document.createElement('input');
        input.type = 'number';
        input.className = 'dr-page-jump-input';
        input.min = 1;
        input.max = max_page;
        input.value = current_page;
        input.setAttribute('aria-label', '跳转页码');

        el.textContent = '';
        el.appendChild(input);
        input.focus();
        input.select();

        let cleaned = false;
        const cleanup = () => {
            if (cleaned) return;
            cleaned = true;
            input.removeEventListener('keydown', key_handler);
            input.removeEventListener('blur', blur_handler);
        };

        const jump_to_page = (page_num) => {
            cleanup();
            const index = page_num - 1;
            if (index >= 0 && index < max_page) {
                this.page_manager.current_index = index;
                this.active_page_index = index;
                this._scroll_to_page(index);
                this._sync_page_buttons();
            }
            this._update_page_indicator();
        };

        const key_handler = (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                const val = parseInt(input.value, 10);
                if (!isNaN(val)) jump_to_page(val);
                else this._update_page_indicator();
            } else if (e.key === 'Escape') {
                e.preventDefault();
                cleanup();
                this._update_page_indicator();
            }
        };

        const blur_handler = () => {
            // 延迟执行，避免与 keydown Enter 冲突
            setTimeout(() => {
                if (cleaned) return;
                const val = parseInt(input.value, 10);
                if (!isNaN(val)) jump_to_page(val);
                else this._update_page_indicator();
            }, 100);
        };

        input.addEventListener('keydown', key_handler);
        input.addEventListener('blur', blur_handler);
    }

    _sync_page_buttons() {
        const prev_btn = document.getElementById('drPagePrev');
        const next_btn = document.getElementById('drPageNext');
        if (prev_btn) prev_btn.disabled = this.page_manager.current_index <= 0;
        if (next_btn) next_btn.disabled = this.page_manager.current_index >= this.page_manager.get_page_count() - 1;
    }

    // ====== 工具栏切换 ======

    _switch_toolbar(reader_active) {
        const toolbar = document.querySelector('.toolbar');
        const left_section = document.querySelector('.toolbar-left');
        const center_section = document.querySelector('.toolbar-center');
        const right_section = document.querySelector('.toolbar-right');
        const btn_expand = document.getElementById('btnExpand');
        const btn_save = document.getElementById('btnSave');

        if (reader_active) {
            // 文档阅读器激活时：隐藏左侧和中间，右侧独占
            if (left_section) left_section.style.display = 'none';
            if (center_section) center_section.style.display = 'none';
            if (right_section) {
                // 用 margin-left:auto 在 flex 流中推到右侧，避免 position:absolute 导致高度/对齐问题
                right_section.style.marginLeft = 'auto';
            }
        } else {
            // 恢复原始布局
            if (left_section) left_section.style.display = '';
            if (center_section) center_section.style.display = '';
            if (right_section) {
                right_section.style.marginLeft = '';
            }
        }

        // 隐藏右侧的图片/文件按钮
        if (btn_expand) btn_expand.style.display = reader_active ? 'none' : '';
        if (btn_save) btn_save.style.display = reader_active ? 'none' : '';

        // 显示/隐藏文档阅读器控件（CSS 已定义 .toolbar-dr-group { display: inline-flex }）
        if (this._dr_tool_group) {
            this._dr_tool_group.style.display = reader_active ? '' : 'none';
        }
    }

    // ====== 工具栏事件 ======

    async _setup_toolbar_events() {
        const close_btn = document.getElementById('drBtnClose');
        if (close_btn) close_btn.addEventListener('click', () => this.close());

        const minimize_btn = document.getElementById('drBtnMinimize');
        if (minimize_btn) {
            minimize_btn.addEventListener('click', async () => {
                if (window.__TAURI__?.window?.getCurrentWindow) {
                    await window.__TAURI__.window.getCurrentWindow().minimize();
                }
            });
            try {
                const result = await window.__TAURI__?.core?.invoke('settings_fetch_all');
                if (result?.settings?.docReaderShowMinimize === false) {
                    minimize_btn.style.display = 'none';
                }
            } catch (_) {}
        }

        const prev_btn = document.getElementById('drPagePrev');
        const next_btn = document.getElementById('drPageNext');
        if (prev_btn) prev_btn.addEventListener('click', () => this.handle_page_nav_prev());
        if (next_btn) next_btn.addEventListener('click', () => this.handle_page_nav_next());

        const page_indicator = document.getElementById('drPageIndicator');
        if (page_indicator) {
            page_indicator.style.cursor = 'pointer';
            // 单击 → 页面侧边栏
            page_indicator.addEventListener('click', () => this._toggle_page_sidebar());
            // 双击 → 页码跳转输入框
            page_indicator.addEventListener('dblclick', () => this._show_page_jump_input());
        }

        const move_btn = document.getElementById('drBtnMove');
        const comment_btn = document.getElementById('drBtnComment');
        const eraser_btn = document.getElementById('drBtnEraser');
        const undo_btn = document.getElementById('drBtnUndo');

        if (move_btn) move_btn.addEventListener('click', () => this._set_draw_mode('move'));
        if (comment_btn) comment_btn.addEventListener('click', () => {
            if (this.draw_mode === 'comment') {
                if (window.main_show_pen_control_panel) {
                    window.main_show_pen_control_panel(comment_btn, 'comment');
                }
            } else {
                this._set_draw_mode('comment');
            }
        });
        if (eraser_btn) eraser_btn.addEventListener('click', () => {
            if (this.draw_mode === 'eraser' && !window.DRAW_CONFIG.eraserSpeedEnabled) {
                if (window.main_show_pen_control_panel) {
                    window.main_show_pen_control_panel(eraser_btn, 'eraser');
                }
            } else {
                this._set_draw_mode('eraser');
            }
        });
        if (undo_btn) undo_btn.addEventListener('click', () => this.handle_undo());

        // 小黑板按钮：在阅读器上方打开小黑板面板
        const bb_btn = document.getElementById('drBtnBlackboard');
        if (bb_btn) {
            bb_btn.addEventListener('click', async () => {
                const bb = await window.blackboard_ensure_loaded(window.dom.canvasContainer);
                if (bb && !bb.is_open) {
                    // 在阅读器面板上方打开，不关闭阅读器
                    bb.open();
                }
            });
        }
    }

    async _set_draw_mode(mode) {
        if (this.is_drawing) {
            this.is_drawing = false;
            if (this.current_stroke) {
                await this._submit_stroke();
            }
            if (this.batch_draw) {
                this.batch_draw.batch_draw_delete_all();
            }
        }
        if (this.isPalmErasing) {
            await this._end_palm_erase();
        }

        this.draw_mode = mode;

        const move_btn = document.getElementById('drBtnMove');
        const comment_btn = document.getElementById('drBtnComment');
        const eraser_btn = document.getElementById('drBtnEraser');

        if (move_btn) move_btn.classList.toggle('active', mode === 'move');
        if (comment_btn) comment_btn.classList.toggle('active', mode === 'comment');
        if (eraser_btn) eraser_btn.classList.toggle('active', mode === 'eraser');

        // 无原生滚动条，touch-action 仅用于控制双指手势由 touch handler 接管
        if (this._scroll_container) {
            this._scroll_container.style.touchAction = 'none';
        }

        if (mode !== 'eraser') {
            this._hide_eraser_hint();
        }
    }

    // ====== 阅读器加载指示 ======

    _show_reader_loading() {
        if (!this._scroll_container) return;
        if (this._reader_loading_el) return;
        const el = document.createElement('div');
        el.className = 'doc-reader-loading';
        el.innerHTML = '<div class="loading-spinner"></div><div class="loading-message">'
            + (window.i18n?.format_translate('loading.rendering') || '正在渲染...')
            + '</div>';
        this._scroll_container.appendChild(el);
        this._reader_loading_el = el;
        this._has_rendered_first_page = false;
    }

    _hide_reader_loading() {
        if (this._reader_loading_el) {
            this._reader_loading_el.remove();
            this._reader_loading_el = null;
        }
    }

    // ====== 橡皮擦提示 ======

    _create_eraser_hint() {
        // 移除旧的橡皮擦提示（如果有）
        if (this._eraser_hint && this._eraser_hint.parentNode) {
            this._eraser_hint.parentNode.removeChild(this._eraser_hint);
        }

        // 创建文档阅读器专用的橡皮擦提示元素
        this._eraser_hint = document.createElement('div');
        this._eraser_hint.className = 'eraser-hint';
        this._eraser_hint.style.width = (window.DRAW_CONFIG?.eraserSize || 15) + 'px';
        this._eraser_hint.style.height = (window.DRAW_CONFIG?.eraserSize || 15) + 'px';
        this._scroll_container.appendChild(this._eraser_hint);

        this._palm_eraser_hint = document.createElement('div');
        this._palm_eraser_hint.className = 'palm-eraser-hint';
        this._palm_eraser_hint.style.width = '60px';
        this._palm_eraser_hint.style.height = '60px';
        this._scroll_container.appendChild(this._palm_eraser_hint);
    }

    _show_eraser_hint() {
        if (!this._eraser_hint) return;
        // 更新橡皮擦尺寸
        const eraser_size = window.DRAW_CONFIG?.eraserSize || 15;
        this._eraser_hint.style.width = eraser_size + 'px';
        this._eraser_hint.style.height = eraser_size + 'px';
        this._eraser_hint.classList.add('active');
    }

    _hide_eraser_hint() {
        if (!this._eraser_hint) return;
        this._eraser_hint.classList.remove('active');
        if (this._eraser_hint_raf_id !== null) {
            cancelAnimationFrame(this._eraser_hint_raf_id);
            this._eraser_hint_raf_id = null;
        }
        this._eraser_hint_pending_pos = null;
    }

    _update_eraser_hint_position(clientX, clientY) {
        if (!this._eraser_hint) return;
        this._eraser_hint_pending_pos = { clientX, clientY };
        if (this._eraser_hint_raf_id !== null) return;

        this._eraser_hint_raf_id = requestAnimationFrame(() => {
            this._eraser_hint_raf_id = null;
            if (!this._eraser_hint_pending_pos) return;
            const pos = this._eraser_hint_pending_pos;
            this._eraser_hint_pending_pos = null;

            const scale = Math.max(0.001, this.dr_scale || 1);
            const eraser_size = (this.cached_draw_line_width || window.DRAW_CONFIG?.eraserSize || 15) * scale;
            this._eraser_hint.style.width = eraser_size + 'px';
            this._eraser_hint.style.height = eraser_size + 'px';

            // 计算相对于滚动容器的位置
            if (this._scroll_container) {
                const rect = this._scroll_container.getBoundingClientRect();
                const x = pos.clientX - rect.left;
                const y = pos.clientY - rect.top;
                this._eraser_hint.style.left = x + 'px';
                this._eraser_hint.style.top = y + 'px';
                this._eraser_hint.style.transform = 'translate(-50%, -50%)';
            }
        });
    }

    // ====== 文档阅读器手掌擦除 ======

    _show_palm_eraser_hint() {
        if (!this._palm_eraser_hint) return;
        this._palm_eraser_hint.classList.add('active');
    }

    _hide_palm_eraser_hint() {
        if (!this._palm_eraser_hint) return;
        this._palm_eraser_hint.classList.remove('active');
    }

    _update_palm_eraser_hint(clientX, clientY, size) {
        if (!this._palm_eraser_hint || !this._scroll_container) return;
        const rect = this._scroll_container.getBoundingClientRect();
        const x = clientX - rect.left;
        const y = clientY - rect.top;
        const visualSize = size * Math.max(0.001, this.dr_scale || 1);
        this._palm_eraser_hint.style.width = visualSize + 'px';
        this._palm_eraser_hint.style.height = visualSize + 'px';
        this._palm_eraser_hint.style.left = x + 'px';
        this._palm_eraser_hint.style.top = y + 'px';
        this._palm_eraser_hint.style.transform = 'translate(-50%, -50%)';
    }

    _init_palm_session() {
        if (this._palmSession || !window.__palmEraser) return;
        const self = this;
        this._palmSession = new window.__palmEraser.PalmEraserSession({
            getCanvasRect: () => self.draw_canvas_rect,
            getScale: () => Math.max(0.001, self.dr_scale || 1),
            showHint: () => self._show_palm_eraser_hint(),
            updateHint: (cx, cy, size) => self._update_palm_eraser_hint(cx, cy, size),
            hideHint: () => self._hide_palm_eraser_hint(),
            saveStrokePoint: (fromX, fromY, toX, toY, pressure) => self._save_stroke_point(fromX, fromY, toX, toY, pressure),
            submitStroke: () => self._submit_stroke(),
            onSessionStart(stroke, session) {
                self.isPalmErasing = true;
                self.savedDrawMode = self.draw_mode;
                self.draw_mode = 'eraser';
                self.palmEraserSize = session.palmEraserSize;
                self.is_drawing = true;
                self.current_stroke = stroke;
                self.cached_draw_type = 'erase';
                self.cached_draw_color = '#000000';
                self.cached_draw_line_width = session.palmEraserSize / Math.max(0.001, self.dr_scale || 1);
                const page_data = self.page_manager.pages_list[self.active_page_index];
                if (page_data && !page_data.is_tiles_initialized) {
                    self._on_page_visible(self.active_page_index);
                }
                if (self.batch_draw) {
                    self.batch_draw.batch_draw_init_start();
                    self.batch_draw.eraserShape = 'square';
                    if (page_data?.tile_renderer) {
                        self.batch_draw._tileRenderer = page_data.tile_renderer;
                    }
                }
            },
            onSessionEnd() {
                self.isPalmErasing = false;
                self.is_drawing = false;
                self.draw_canvas_rect = null;
                self.draw_mode = self.savedDrawMode || 'comment';
                self.savedDrawMode = null;
                self.current_stroke = null;
            }
        });
    }

    _start_palm_erase(clientX, clientY, eraserWidth) {
        const target = document.elementFromPoint(clientX, clientY);
        if (!target) return;
        const page_index = parseInt(target.dataset.page);
        if (isNaN(page_index)) return;
        const page_data = this.page_manager.pages_list[page_index];
        if (!page_data?.page_element) return;
        this.active_page_index = page_index;
        this.draw_canvas_rect = page_data.page_element.getBoundingClientRect();

        this._init_palm_session();
        if (this._palmSession) this._palmSession.start(clientX, clientY, eraserWidth);
    }

    _update_palm_erase(clientX, clientY) {
        if (this._palmSession) this._palmSession.update(clientX, clientY);
    }

    async _end_palm_erase() {
        if (this._palmSession) await this._palmSession.end();
    }

    // ====== 页面侧边栏 ======

    _toggle_page_sidebar() {
        const existing_sidebar = document.getElementById('drPageSidebar');
        if (existing_sidebar) {
            existing_sidebar.remove();
            return;
        }

        const sidebar = document.createElement('div');
        sidebar.id = 'drPageSidebar';
        sidebar.className = 'dr-page-sidebar';

        const pages = this.page_manager.pages_list;
        const current_index = this.page_manager.current_index;

        // 创建头部
        const header = document.createElement('div');
        header.className = 'dr-page-sidebar-header';
        header.textContent = `页面 (${current_index + 1}/${pages.length})`;
        sidebar.appendChild(header);

        // 创建内容区域
        const content = document.createElement('div');
        content.className = 'dr-page-sidebar-content';

        const use_virtual_sidebar = pages.length > this._sidebar_virtual_threshold;
        if (use_virtual_sidebar) {
            content.classList.add('virtualized');
        } else {
            pages.forEach((page, index) => {
                content.appendChild(this._create_page_sidebar_item(page, index, current_index));
            });
        }

        sidebar.appendChild(content);
        document.body.appendChild(sidebar);
        if (use_virtual_sidebar) {
            this._setup_virtual_page_sidebar(content, pages, current_index);
        } else {
            this._setup_page_sidebar_thumbnail_loading(content);
        }

        // 绑定点击事件
        content.addEventListener('click', (event) => {
            const item = event.target.closest('.dr-page-sidebar-item');
            if (!item || !content.contains(item)) return;
            const page_index = parseInt(item.dataset.page);
            this._scroll_to_page(page_index);
            this.page_manager.current_index = page_index;
            this.active_page_index = page_index;
            this._update_page_indicator();
            this._sync_page_buttons();
            sidebar.remove();
        });

        // 点击外部关闭
        const close_handler = (e) => {
            if (!sidebar.contains(e.target) && e.target.id !== 'drPageIndicator') {
                sidebar.remove();
                document.removeEventListener('click', close_handler);
            }
        };
        setTimeout(() => document.addEventListener('click', close_handler), 100);
    }

    _create_page_sidebar_item(page, index, current_index) {
        const is_active = index === current_index;
        const page_label = `第 ${page.page_num || index + 1} 页`;
        const item = document.createElement('div');
        item.className = `dr-page-sidebar-item ${is_active ? 'active' : ''}`;
        item.dataset.page = index;

        const thumbnail_src = page.image_url || page.thumbnail_url;
        let thumb_el;
        if (thumbnail_src) {
            const img = document.createElement('img');
            img.className = 'dr-page-sidebar-thumb';
            img.dataset.page = index;
            img.src = thumbnail_src;
            img.alt = page_label;
            img.loading = 'lazy';
            thumb_el = img;
        } else if (page.render_mode === 'pdfjs') {
            const canvas = document.createElement('canvas');
            canvas.className = 'dr-page-sidebar-thumb dr-page-sidebar-pdf-thumb is-loading';
            canvas.dataset.page = index;
            canvas.setAttribute('role', 'img');
            canvas.setAttribute('aria-label', page_label);
            thumb_el = canvas;
            item.classList.add('loading');
        } else {
            thumb_el = document.createElement('div');
            thumb_el.className = 'dr-page-sidebar-thumb is-loading';
            thumb_el.dataset.page = index;
            thumb_el.setAttribute('role', 'img');
            thumb_el.setAttribute('aria-label', page_label);
            item.classList.add('loading');
        }

        const label = document.createElement('span');
        label.textContent = page_label;

        item.appendChild(thumb_el);
        item.appendChild(label);
        return item;
    }

    _setup_virtual_page_sidebar(content, pages, current_index) {
        let render_raf = null;
        const render_window = () => {
            render_raf = null;
            const item_h = this._sidebar_item_height;
            const viewport_h = content.clientHeight || 480;
            const start = Math.max(0, Math.floor(content.scrollTop / item_h) - this._sidebar_overscan);
            const end = Math.min(
                pages.length,
                Math.ceil((content.scrollTop + viewport_h) / item_h) + this._sidebar_overscan
            );

            const spacer = document.createElement('div');
            spacer.className = 'dr-page-sidebar-virtual-spacer';
            spacer.style.height = `${pages.length * item_h}px`;

            for (let i = start; i < end; i++) {
                const item = this._create_page_sidebar_item(pages[i], i, current_index);
                item.style.top = `${i * item_h}px`;
                item.style.height = `${item_h - 6}px`;
                spacer.appendChild(item);
            }

            content.replaceChildren(spacer);
            this._setup_page_sidebar_thumbnail_loading(content);
        };

        const schedule_render = () => {
            if (render_raf !== null) return;
            render_raf = requestAnimationFrame(render_window);
        };

        content.addEventListener('scroll', schedule_render, { passive: true });
        content.scrollTop = Math.max(0, current_index * this._sidebar_item_height - this._sidebar_item_height);
        render_window();
    }

    _setup_page_sidebar_thumbnail_loading(content) {
        const unloaded_imgs = Array.from(content.querySelectorAll('.dr-page-sidebar-thumb.is-loading'));
        if (unloaded_imgs.length === 0) return;

        // 优先加载可见页面和当前活动页面附近的缩略图
        const priority_imgs = unloaded_imgs
            .filter(img => {
                const page_index = parseInt(img.dataset.page);
                const page = this.page_manager.pages_list[page_index];
                return page?.is_visible || page_index === this.active_page_index;
            })
            .sort((a, b) => {
                const ai = parseInt(a.dataset.page);
                const bi = parseInt(b.dataset.page);
                return Math.abs(ai - this.active_page_index) - Math.abs(bi - this.active_page_index);
            });

        // 批量加载优先图片（限制并发数为3）
        const max_concurrent = 3;
        let loading_count = 0;
        const load_next = () => {
            while (loading_count < max_concurrent && priority_imgs.length > 0) {
                const img = priority_imgs.shift();
                loading_count++;
                this._load_page_sidebar_thumbnail(parseInt(img.dataset.page), img).finally(() => {
                    loading_count--;
                    load_next();
                });
            }
        };
        load_next();

        const deferred_imgs = unloaded_imgs.filter(img => !priority_imgs.includes(img));
        if (deferred_imgs.length === 0) return;

        if (!window.IntersectionObserver) {
            // 不支持IntersectionObserver时，加载前8个
            deferred_imgs.slice(0, 8).forEach(img => {
                this._load_page_sidebar_thumbnail(parseInt(img.dataset.page), img);
            });
            return;
        }

        // 断开前一个 observer（防止多次打开侧边栏导致泄漏）
        if (this._thumbnail_observer) {
            this._thumbnail_observer.disconnect();
            this._thumbnail_observer = null;
        }
        // 使用IntersectionObserver加载可见区域的缩略图
        this._thumbnail_observer = new IntersectionObserver((entries) => {
            for (const entry of entries) {
                if (!entry.isIntersecting) continue;
                const img = entry.target;
                this._thumbnail_observer?.unobserve(img);
                this._load_page_sidebar_thumbnail(parseInt(img.dataset.page), img);
            }
        }, {
            root: content,
            rootMargin: '200px 0px', // 增加预加载区域
            threshold: 0.01
        });

        deferred_imgs.forEach(img => this._thumbnail_observer.observe(img));
    }

    async _load_page_sidebar_thumbnail(page_index, img) {
        const page = this.page_manager.pages_list[page_index];
        if (!page || !img) return;

        // 检查缓存
        if (this._sidebar_thumbnail_cache.has(page_index)) {
            const cached_url = this._sidebar_thumbnail_cache.get(page_index);
            this._set_sidebar_thumbnail_src(img, page_index, cached_url);
            return;
        }

        if (page.render_mode === 'pdfjs') {
            if (img.dataset.rendered === 'true') return;
            if (page.sidebar_thumbnail_loading) return;

            page.sidebar_thumbnail_loading = true;
            try {
                await this._render_page_sidebar_pdf_thumbnail(page_index, img);
            } catch (error) {
                console.error(`渲染 PDF 缩略图 ${page_index + 1} 失败:`, error);
            } finally {
                page.sidebar_thumbnail_loading = false;
            }
            return;
        }

        const existing_src = page.image_url || page.thumbnail_url;
        if (existing_src) {
            this._set_sidebar_thumbnail_src(img, page_index, existing_src);
            return;
        }
        if (page.sidebar_thumbnail_loading) return;

        page.sidebar_thumbnail_loading = true;
        try {
            await this._load_pdf_page(page_index);
            const loaded_src = page.image_url || page.thumbnail_url;
            if (loaded_src) {
                this._update_page_sidebar_thumbnail(page_index, loaded_src);
            }
        } catch (error) {
            console.error(`加载侧边栏原图 ${page_index + 1} 失败:`, error);
        } finally {
            page.sidebar_thumbnail_loading = false;
        }
    }

    async _render_page_sidebar_pdf_thumbnail(page_index, canvas) {
        const page = this.page_manager.pages_list[page_index];
        const folder = window.state.fileList[this.folder_index];
        if (!page || !folder?.pdfDoc || !canvas || canvas.tagName !== 'CANVAS') return;

        // 检查缓存
        if (this._sidebar_thumbnail_cache.has(page_index)) {
            const cached_url = this._sidebar_thumbnail_cache.get(page_index);
            this._set_sidebar_thumbnail_src(canvas, page_index, cached_url);
            return;
        }

        let pdf_page = this._pdf_page_cache.get(page_index);
        if (!pdf_page) {
            pdf_page = await folder.pdfDoc.getPage(page.page_num);
        }
        try {
            const base_viewport = pdf_page.getViewport({ scale: 1 });
            // 使用更小的渲染尺寸以提高性能
            const css_w = Math.max(120, Math.round(canvas.clientWidth || canvas.closest('.dr-page-sidebar-item')?.clientWidth || 180));
            const css_h = Math.round(css_w * 9 / 16);
            const dpr = Math.min(window.devicePixelRatio || window.DRAW_CONFIG?.dpr || 1, 2);
            const canvas_w = Math.ceil(css_w * dpr);
            const canvas_h = Math.ceil(css_h * dpr);
            const page_scale = Math.min(canvas_w / base_viewport.width, canvas_h / base_viewport.height);
            const viewport = pdf_page.getViewport({ scale: page_scale });
            const offset_x = Math.round((canvas_w - viewport.width) / 2);
            const offset_y = Math.round((canvas_h - viewport.height) / 2);

            canvas.width = canvas_w;
            canvas.height = canvas_h;
            canvas.style.width = '100%';
            canvas.style.height = css_h + 'px';

            const ctx = canvas.getContext('2d', { alpha: false });
            ctx.setTransform(1, 0, 0, 1, 0, 0);
            ctx.fillStyle = '#fff';
            ctx.fillRect(0, 0, canvas_w, canvas_h);

            const task = pdf_page.render({
                canvasContext: ctx,
                viewport,
                transform: [1, 0, 0, 1, offset_x, offset_y]
            });
            await task.promise;

            // 将渲染结果转换为blob URL并缓存
            const blob_url = await new Promise(resolve => {
                canvas.toBlob(blob => {
                    if (blob) {
                        resolve(URL.createObjectURL(blob));
                    } else {
                        resolve(null);
                    }
                }, 'image/jpeg', 0.7);
            });

            if (blob_url) {
                this._sidebar_thumbnail_cache.set(page_index, blob_url);
                // 使用缓存的blob URL替换canvas
                this._set_sidebar_thumbnail_src(canvas, page_index, blob_url);
            } else {
                canvas.dataset.rendered = 'true';
                canvas.classList.remove('is-loading');
                canvas.closest('.dr-page-sidebar-item')?.classList.remove('loading');
            }
        } finally {
            // 只在非缓存页 cleanup（缓存页保持内部数据就绪）
            if (!this._pdf_page_cache.has(page_index)) {
                pdf_page.cleanup?.();
            }
        }
    }

    _update_page_sidebar_thumbnail(page_index, src) {
        const img = document.querySelector(`#drPageSidebar .dr-page-sidebar-thumb[data-page="${page_index}"]`);
        if (!img || !src) return;
        this._set_sidebar_thumbnail_src(img, page_index, src);
    }

    _set_sidebar_thumbnail_src(thumb_el, page_index, src) {
        if (!thumb_el || !src) return;
        let img = thumb_el;
        if (thumb_el.tagName !== 'IMG') {
            img = document.createElement('img');
            img.className = 'dr-page-sidebar-thumb';
            img.dataset.page = page_index;
            img.alt = `第 ${page_index + 1} 页`;
            img.loading = 'lazy';
            thumb_el.replaceWith(img);
        }
        img.src = src;
        img.classList.remove('is-loading');
        img.closest('.dr-page-sidebar-item')?.classList.remove('loading');
    }

    _release_sidebar_thumbnail_cache() {
        // 释放所有缓存的blob URL
        for (const blob_url of this._sidebar_thumbnail_cache.values()) {
            if (blob_url && blob_url.startsWith('blob:')) {
                URL.revokeObjectURL(blob_url);
            }
        }
        this._sidebar_thumbnail_cache.clear();
    }

    // ====== 缩放与 LOD ======

    /** 计算平移边界：缩放后内容是否超出视口，决定可拖动范围 */
    _dr_update_move_bound() {
        if (!this._zoom_wrapper || !this._scroll_container) return;
        const wrapper = this._zoom_wrapper;
        const container = this._scroll_container;
        const mb = this.dr_move_bound;
        const s = this.dr_scale;

        // 仅在内容/视口尺寸变化时重新读取 DOM（避免每帧 layout thrashing）
        const content_w = wrapper.scrollWidth;
        const content_h = wrapper.scrollHeight;
        const viewport_w = container.clientWidth;
        const viewport_h = container.clientHeight;

        const needRecompute = this._dr_mb_cache_cw !== content_w ||
            this._dr_mb_cache_ch !== content_h ||
            this._dr_mb_cache_vw !== viewport_w ||
            this._dr_mb_cache_vh !== viewport_h;
        if (needRecompute) {
            this._dr_mb_cache_cw = content_w;
            this._dr_mb_cache_ch = content_h;
            this._dr_mb_cache_vw = viewport_w;
            this._dr_mb_cache_vh = viewport_h;
        }
        // scale 每帧变化时仍需重算 bounds，但跳过 DOM 读取（数学计算极快）

        // X 方向：内容 + padding 始终居中
        const bounded_w = content_w * s;
        if (bounded_w >= viewport_w) {
            mb.min_x = -(bounded_w - viewport_w);
            mb.max_x = 0;
        } else {
            mb.min_x = (viewport_w - bounded_w) / 2;
            mb.max_x = (viewport_w - bounded_w) / 2;
        }

        // Y 方向
        const bounded_h = content_h * s;
        if (bounded_h >= viewport_h) {
            mb.min_y = -(bounded_h - viewport_h);
            mb.max_y = 0;
        } else {
            mb.min_y = (viewport_h - bounded_h) / 2;
            mb.max_y = (viewport_h - bounded_h) / 2;
        }
    }

    /** 将 canvas_x/y 钳制在 move_bound 内 */
    _dr_update_canvas_position() {
        const eps = 0.001;
        const mb = this.dr_move_bound;
        this.dr_canvas_x = Math.max(mb.min_x - eps, Math.min(mb.max_x + eps, this.dr_canvas_x));
        this.dr_canvas_y = Math.max(mb.min_y - eps, Math.min(mb.max_y + eps, this.dr_canvas_y));
    }

    /** 仅同步 transform（无 LOD 更新，用于高频拖拽） */
    _dr_sync_transform() {
        if (!this._zoom_wrapper) return;
        this._zoom_wrapper.style.transform = 'translate3d(' + this.dr_canvas_x + 'px, ' + this.dr_canvas_y + 'px, 0) scale(' + this.dr_scale + ')';
        if (this.batch_draw) {
            this.batch_draw._overlay_cached_rect_left = null;
            this.batch_draw._overlay_cached_rect_top = null;
        }
    }

    /** rAF 节流版 sync_transform：合并多帧调用，每帧最多一次 DOM 写入 */
    _dr_sync_transform_schedule(x, y, scale) {
        this._dr_pending_transform = { x, y, scale };
        if (this._dr_transform_raf_id === null) {
            this._dr_transform_raf_id = requestAnimationFrame(() => {
                const pt = this._dr_pending_transform;
                this._dr_pending_transform = null;
                this._dr_transform_raf_id = null;
                if (pt && this._zoom_wrapper) {
                    this._zoom_wrapper.style.transform = 'translate3d(' + pt.x + 'px, ' + pt.y + 'px, 0) scale(' + pt.scale + ')';
                    this._dr_last_transform.x = pt.x;
                    this._dr_last_transform.y = pt.y;
                    this._dr_last_transform.scale = pt.scale;
                    if (this.batch_draw) {
                        this.batch_draw._overlay_cached_rect_left = null;
                        this.batch_draw._overlay_cached_rect_top = null;
                    }
                }
            });
        }
    }

    /** 应用当前缩放比到 wrapper transform + TileRenderer LOD */
    _dr_apply_scale() {
        const s = this.dr_scale;
        this.dr_cached_inv_scale = 1 / s;

        // 使 overlay 缓存失效
        if (this.batch_draw) {
            this.batch_draw._overlay_cached_rect_left = null;
            this.batch_draw._overlay_cached_rect_top = null;
        }

        this._dr_update_move_bound();
        this._dr_update_canvas_position();
        this._dr_sync_transform();

        // 缩放进行中跳过 tile DPR 更新和可见页重绘，由缩放结束后批量刷新
        if (this._dr_is_zooming) return;

        // 仅遍历已初始化 tile 的页面（跳过无 tile 页面，200+ 页文档性能提升显著）
        for (const i of this._pages_with_tiles) {
            const pd = this.page_manager.pages_list[i];
            if (pd && (pd.is_visible || this._is_page_near_active(i, this._tile_keep_distance))) {
                pd.tile_renderer.update_visible_tile_dpr(s, false, true);
            }
        }

        // 更新 active 页面的 overlay 尺寸（getBoundingClientRect 已包含 transform 缩放）
        if (this.active_page_index >= 0) {
            this._update_overlay_size(this.active_page_index);
        }

        // 缩放/平移后检查页面可见性（已优化为纯数学计算，不触发每页布局）
        this._check_page_visibility();
    }

    /** 标记缩放进行中，延迟 300ms 后触发批量重绘 */
    _dr_set_zooming() {
        if (!this._dr_is_zooming) {
            this._dr_is_zooming = true;
            if (this.batch_draw) {
                this.batch_draw.hide_overlay();
            }
        }
        if (this._zoom_complete_timer !== null) clearTimeout(this._zoom_complete_timer);
        this._zoom_complete_timer = setTimeout(() => {
            this._zoom_complete_timer = null;
            this._dr_is_zooming = false;
            if (this.batch_draw) {
                this.batch_draw.show_overlay();
            }
            // 缩放结束后批量重绘可见页 + 更新 tile DPR
            this._check_page_visibility();
            for (const i of this._pages_with_tiles) {
                const pd = this.page_manager.pages_list[i];
                if (pd && (pd.is_visible || this._is_page_near_active(i, this._tile_keep_distance))) {
                    pd.tile_renderer?.update_visible_tile_dpr(this.dr_scale, false, true);
                }
            }
        }, 300);
    }

    /** 撤销、翻页等操作应强制立即重绘，取消缩放延迟 */
    _dr_cancel_zoom_debounce() {
        if (this._zoom_complete_timer !== null) {
            clearTimeout(this._zoom_complete_timer);
            this._zoom_complete_timer = null;
        }
        this._dr_is_zooming = false;
        if (this.batch_draw) {
            this.batch_draw.show_overlay();
        }
    }
    _dr_enable_smooth_transform() {
        if (this._smooth_transform_timeout_id !== null) {
            clearTimeout(this._smooth_transform_timeout_id);
            this._smooth_transform_timeout_id = null;
        }
        if (this._zoom_wrapper) {
            this._zoom_wrapper.classList.add('smooth-transform');
        }
    }

    /** 延迟移除 will-change: transform（交互结束后 150ms 释放 GPU 资源） */
    _dr_schedule_disable_smooth_transform() {
        if (this._smooth_transform_timeout_id !== null) {
            clearTimeout(this._smooth_transform_timeout_id);
        }
        this._smooth_transform_timeout_id = setTimeout(() => {
            this._smooth_transform_timeout_id = null;
            if (this._zoom_wrapper) {
                this._zoom_wrapper.classList.remove('smooth-transform');
            }
        }, 150);
    }

    // ====== 惯性系统 ======

    _dr_cancel_momentum() {
        if (this._dr_momentum_raf !== null) {
            cancelAnimationFrame(this._dr_momentum_raf);
            this._dr_momentum_raf = null;
        }
    }

    _dr_start_momentum() {
        if (window.DRAW_CONFIG && !window.DRAW_CONFIG.momentumEnabled) return;
        this._dr_cancel_momentum();
        if (this._dr_momentum_raf !== null) return;
        this._dr_momentum_raf = requestAnimationFrame(() => this._dr_momentum_tick());
    }

    _dr_momentum_tick() {
        let vx = this._dr_gesture_vx;
        let vy = this._dr_gesture_vy;
        const speed = Math.sqrt(vx * vx + vy * vy);
        const friction = 0.85 - 0.20 * Math.exp(-speed / 8);
        vx *= friction;
        vy *= friction;
        this._dr_gesture_vx = vx;
        this._dr_gesture_vy = vy;

        const prevX = this.dr_canvas_x;
        const prevY = this.dr_canvas_y;
        this.dr_canvas_x += vx;
        this.dr_canvas_y += vy;

        this._dr_update_move_bound();
        this._dr_update_canvas_position();

        // 边界碰撞处理：速度归零（防止贴边滑行）
        if (this.dr_canvas_x === prevX && vx !== 0) {
            this._dr_gesture_vx = 0;
            vx = 0;
        }
        if (this.dr_canvas_y === prevY && vy !== 0) {
            this._dr_gesture_vy = 0;
            vy = 0;
        }

        this._dr_sync_transform();

        if (Math.abs(vx) > 0.5 || Math.abs(vy) > 0.5) {
            this._dr_momentum_raf = requestAnimationFrame(() => this._dr_momentum_tick());
        } else {
            this._dr_momentum_raf = null;
            // 延迟重量操作到浏览器空闲时，避免最后一帧 jank
            const doFinal = () => {
                this._dr_apply_scale();
                this._dr_schedule_disable_smooth_transform();
                this._check_page_visibility();
            };
            if (window.requestIdleCallback) {
                window.requestIdleCallback(doFinal, { timeout: 500 });
            } else {
                doFinal();
            }
        }
    }

    _dr_update_gesture_velocity() {
        const dx = this.dr_canvas_x - this._dr_last_canvas_x;
        const dy = this.dr_canvas_y - this._dr_last_canvas_y;
        const alpha = 0.5;
        this._dr_gesture_vx = this._dr_gesture_vx * (1 - alpha) + dx * alpha;
        this._dr_gesture_vy = this._dr_gesture_vy * (1 - alpha) + dy * alpha;
        this._dr_last_canvas_x = this.dr_canvas_x;
        this._dr_last_canvas_y = this.dr_canvas_y;
    }

    /** 滚轮缩放（以鼠标位置为中心，rAF 节流重计算） */
    _dr_handle_wheel(e) {
        if (!this.is_open) return;
        if (this.is_drawing) return;

        // 阅读器内部无原生滚动条，滚轮直接用于缩放。
        e.preventDefault();

        const max_s = this.dr_max_scale;
        const min_s = this.dr_min_scale;
        const delta = e.deltaY > 0 ? -0.15 : 0.15;
        const new_s = Math.max(min_s, Math.min(max_s, this.dr_scale + delta));

        if (new_s !== this.dr_scale) {
            const old_s = this.dr_scale;
            const ratio = new_s / old_s;

            // 以鼠标位置为中心缩放（Blackboard 风格）
            if (!this._cached_container_rect) {
                const cr = this._scroll_container.getBoundingClientRect();
                this._cached_container_rect = { top: cr.top, bottom: cr.bottom, left: cr.left };
            }
            const container_rect = this._cached_container_rect;
            const mouse_x = e.clientX - container_rect.left;
            const mouse_y = e.clientY - container_rect.top;

            this.dr_canvas_x = mouse_x - (mouse_x - this.dr_canvas_x) * ratio;
            this.dr_canvas_y = mouse_y - (mouse_y - this.dr_canvas_y) * ratio;
            this.dr_scale = new_s;
            this.dr_cached_inv_scale = 1 / new_s;
            if (this.batch_draw) {
                this.batch_draw._overlay_cached_rect_left = null;
                this.batch_draw._overlay_cached_rect_top = null;
            }

            // will-change: 按需启用 GPU 合成层
            this._dr_enable_smooth_transform();

            // 标记缩放进行中，合并多帧事件并延迟批量重绘
            this._dr_set_zooming();

            // rAF 节流：合并多帧滚轮事件的 _dr_apply_scale 调用
            if (this._wheel_raf_id !== null) {
                cancelAnimationFrame(this._wheel_raf_id);
            }
            this._wheel_raf_id = requestAnimationFrame(() => {
                this._wheel_raf_id = null;
                this._dr_apply_scale();
                this._dr_schedule_disable_smooth_transform();
            });
        }
    }

    // ====== 按钮状态 ======

    _update_button_status() {
        const undo_btn = document.getElementById('drBtnUndo');
        if (undo_btn) undo_btn.disabled = !history_validate_undo();
    }
}

const documentReaderManager = new DocumentReaderManager();
window.documentReaderManager = documentReaderManager;
export default documentReaderManager;
