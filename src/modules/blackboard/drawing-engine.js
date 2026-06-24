/**
 * ViewStage 公共绘制引擎
 * 封装笔画生命周期、batch_draw 集成、橡皮擦、历史管理
 * 通过 CoordinateProvider 抽象与具体坐标系解耦
 *
 * CoordinateProvider 接口:
 *   get_rect()              -> { left, top }      绘制区域的屏幕 bounding rect
 *   get_scale()             -> number              当前缩放比
 *   get_origin()            -> { x, y }            画布平移偏移
 *   set_origin(x, y)        -> void                设置画布平移偏移（move 模式用）
 *   get_stroke_history()    -> Array               笔画历史引用
 *   get_eraser_hint_rect()  -> { left, top }       橡皮擦提示定位用 rect（可选）
 *   on_stroke_finalized(stroke, bounds?) -> void   笔画提交回调
 *   render_all_strokes(bounds?)          -> void   全量/局部重绘
 */

import {
    history_execute_command,
    history_init_manager,
    history_validate_undo,
    history_handle_undo,
    history_handle_state_change,
    DrawCommand,
    ClearCommand,
    history_state
} from '../history.js';

import {
    is_palm_by_pointer,
    is_palm_by_touch_count,
    get_palm_center,
    compute_palm_eraser_size_from_pointer,
    PALM_CONFIG
} from '../palm-eraser/palm-eraser.js';

export class DrawingEngine {
    /**
     * @param {Object} coord - CoordinateProvider
     */
    constructor(coord) {
        this.coord = coord;

        // 绘制模式
        this.draw_mode = 'move'; // 'move' | 'comment' | 'eraser'

        // 当前笔画状态
        this.is_drawing = false;
        this.current_stroke = null;
        this.last_x = 0;
        this.last_y = 0;
        this.cached_draw_type = null;
        this.cached_draw_color = null;
        this.cached_draw_line_width = null;
        this.current_pressure = 0.5;
        this.current_line_width = 5;
        this.last_line_width = 5;

        // 缓存鼠标/触摸位置的 rect（减少 getBoundingClientRect 调用）
        this.draw_canvas_rect = null;

        // 速度擦除状态
        this._eraser_speed_state = null;
        this._last_draw_time = 0;
        this._last_draw_x = null;
        this._last_draw_y = null;
        this._speed_buffer = new Array(5);
        this._speed_buffer_idx = 0;
        this._speed_buffer_count = 0;

        // batch_draw
        this.batch_draw = null;

        // 面板过渡标记：过渡中禁止绘制（防止坐标偏移）
        this._painting_allowed = false;

        // 保存的历史快照（用于隔离）
        this.saved_history_state = null;

        // 橡皮擦提示
        this._eraser_hint = null;
        this._eraser_hint_raf_id = null;
        this._eraser_hint_pending_pos = null;

        // 手掌擦除
        this.isPalmErasing = false;
        this.savedDrawMode = null;
        this.palmEraserSize = 60;
        this._palm_eraser_hint = null;
    }

    /**
     * 设置为可绘制状态（面板过渡完成后调用，修复坐标偏移 bug）
     */
    set_painting_allowed(allowed) {
        this._painting_allowed = allowed;
    }

    // ====== 初始化 ======

    init_batch_draw(overlay_canvas, overlay_ctx) {
        this.batch_draw = new window.RealtimeBatchDrawManager();
        this.batch_draw._overlayCanvas = overlay_canvas;
        this.batch_draw._overlayCtx = overlay_ctx;
        this.batch_draw._overlayDpr = this.batch_draw._calc_overlay_dpr(this.coord.get_scale() || 1);
        this.batch_draw._overlayTransformScale = 0;
        this.batch_draw._overlayTransformX = 0;
        this.batch_draw._overlayTransformY = 0;
        this.batch_draw._sync_overlay_transform = () => this._sync_overlay_transform();

        if (window.DRAW_CONFIG?.frameRateMode) {
            this.batch_draw.batch_draw_update_frame_rate(window.DRAW_CONFIG.frameRateMode);
        }
    }

    _sync_overlay_transform() {
        if (!this.batch_draw?._overlayCtx) return;
        const ctx = this.batch_draw._overlayCtx;
        const s = this.coord.get_scale();
        const origin = this.coord.get_origin();
        const dpr = this.batch_draw._overlayDpr || 1;
        const scale = s || 1;
        const ox = origin?.x || 0;
        const oy = origin?.y || 0;
        if (this.batch_draw._overlayTransformScale === scale &&
            this.batch_draw._overlayTransformX === ox &&
            this.batch_draw._overlayTransformY === oy) return;
        this.batch_draw._overlayTransformScale = scale;
        this.batch_draw._overlayTransformX = ox;
        this.batch_draw._overlayTransformY = oy;
        ctx.setTransform(scale * dpr, 0, 0, scale * dpr, ox * dpr, oy * dpr);
    }

    init_history(on_state_change) {
        history_init_manager({ on_state_change });
    }

    /** 保存全局历史快照并创建隔离历史 */
    push_history_isolate(on_state_change) {
        window.__HISTORY_ISOLATED = true;
        this.saved_history_state = {
            undo_list: [...history_state.undo_list],
            redo_list: [...history_state.redo_list],
            on_state_change: history_state.on_state_change
        };
        this.init_history(on_state_change);
    }

    /** 恢复全局历史 */
    pop_history_isolate() {
        window.__HISTORY_ISOLATED = false;
        if (this.saved_history_state) {
            history_state.undo_list = this.saved_history_state.undo_list;
            history_state.redo_list = this.saved_history_state.redo_list;
            history_state.on_state_change = this.saved_history_state.on_state_change;
            this.saved_history_state = null;
            history_handle_state_change();
        }
    }

    // ====== 工具方法 ======

    _fetch_safe_scale() {
        return Math.max(0.001, this.coord.get_scale() || 1);
    }

    _get_canvas_rect() {
        return this.coord.get_rect();
    }

    // ====== 设置绘制模式 ======

    set_draw_mode(mode) {
        this.draw_mode = mode;
    }

    // ====== 笔画生命周期 ======

    _start_stroke(type) {
        const DRAW_CONFIG = window.DRAW_CONFIG || {};
        const inv_scale = 1 / this._fetch_safe_scale();
        const baseEraserSize = DRAW_CONFIG.eraserSize * inv_scale;
        this.current_stroke = {
            type: type,
            points: [],
            color: type === 'draw' ? DRAW_CONFIG.penColor : '#000000',
            lineWidth: type === 'draw' ? DRAW_CONFIG.penWidth * inv_scale : baseEraserSize,
            eraserSize: baseEraserSize,
            eraserSizeRaw: DRAW_CONFIG.eraserSize,
            eraserSpeedEnabled: DRAW_CONFIG.eraserSpeedEnabled,
            eraserSpeedMinSize: (DRAW_CONFIG.eraserSpeedMinSize || 0) * inv_scale,
            eraserSpeedMaxSize: (DRAW_CONFIG.eraserSpeedMaxSize || 0) * inv_scale,
            eraserSpeedFactor: DRAW_CONFIG.eraserSpeedFactor,
            scale: this.coord.get_scale() || 1,
            bounds: {
                minX: Infinity, minY: Infinity,
                maxX: -Infinity, maxY: -Infinity
            },
            variableWidths: []
        };

        this.current_pressure = 0.5;
        this.current_line_width = DRAW_CONFIG.penWidth * inv_scale;
        this.last_line_width = DRAW_CONFIG.penWidth * inv_scale;

        this.cached_draw_type = type;
        this.cached_draw_color = type === 'draw' ? DRAW_CONFIG.penColor : '#000000';
        const currentScale = this._fetch_safe_scale();
        this.cached_draw_line_width = type === 'draw' ? DRAW_CONFIG.penWidth / currentScale : DRAW_CONFIG.eraserSize / currentScale;

        this._last_draw_time = performance.now();
        this._last_draw_x = null;
        this._last_draw_y = null;
        this._speed_buffer = new Array(5);
        this._speed_buffer_idx = 0;
        this._speed_buffer_count = 0;
        this._eraser_speed_state = window.__eraserSpeed?.eraser_speed_create_state() ?? null;

        if (this.batch_draw) {
            this.batch_draw.batch_draw_init_start();
        }
    }

    _save_stroke_point(from_x, from_y, to_x, to_y, pressure) {
        const stroke = this.current_stroke;
        if (!stroke) return;
        const DRAW_CONFIG = window.DRAW_CONFIG || {};

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
        const currentScale = this._fetch_safe_scale();

        if (stroke.type === 'draw') {
            this.current_pressure = pressure;
            this.last_line_width = this.current_line_width;
            currentWidth = stroke.lineWidth * (0.9 + pressure * 0.2);
            this.current_line_width = currentWidth;
            this.cached_draw_line_width = DRAW_CONFIG.penWidth / currentScale;
        } else if (stroke.type === 'erase' && stroke.eraserSpeedEnabled) {
            if (this._eraser_speed_state && window.__eraserSpeed) {
                currentWidth = window.__eraserSpeed.eraser_speed_update(this._eraser_speed_state, stroke, to_x, to_y);
            } else {
                const now = performance.now();
                const dt = now - this._last_draw_time;
                if (this._last_draw_x !== null && dt > 0) {
                    const dx = to_x - this._last_draw_x;
                    const dy = to_y - this._last_draw_y;
                    const speed = Math.sqrt(dx * dx + dy * dy) / dt;
                    // 环形缓冲区替代 push/shift，避免数组扩容和 O(n) 移位
                    this._speed_buffer[this._speed_buffer_idx] = speed;
                    this._speed_buffer_idx = (this._speed_buffer_idx + 1) % 5;
                    if (this._speed_buffer_count < 5) this._speed_buffer_count++;
                    let speed_sum = 0;
                    for (let i = 0; i < this._speed_buffer_count; i++) speed_sum += this._speed_buffer[i];
                    const avgSpeed = speed_sum / this._speed_buffer_count;
                    const sizeRange = stroke.eraserSpeedMaxSize - stroke.eraserSpeedMinSize;
                    currentWidth = stroke.eraserSpeedMinSize + Math.min(avgSpeed * stroke.eraserSpeedFactor * 100, sizeRange);
                    currentWidth = Math.max(stroke.eraserSpeedMinSize, Math.min(stroke.eraserSpeedMaxSize, currentWidth));
                }
                this._last_draw_time = now;
                this._last_draw_x = to_x;
                this._last_draw_y = to_y;
            }
            this.cached_draw_line_width = currentWidth;
        } else if (stroke.type === 'erase') {
            this.cached_draw_line_width = DRAW_CONFIG.eraserSize / currentScale;
        }

        stroke.variableWidths.push(currentWidth);
        stroke.points.push({
            fromX: from_x, fromY: from_y,
            toX: to_x, toY: to_y
        });
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

            const stroke_history = this.coord.get_stroke_history();
            if (stroke_history) {
                const hw = Math.max(
                    this.current_stroke.lineWidth || 5,
                    this.current_stroke.eraserSize || 5
                ) / 2;
                const raw = this.current_stroke.bounds;
                const stroke_bounds = raw ? {
                    minX: raw.minX - hw, minY: raw.minY - hw,
                    maxX: raw.maxX + hw, maxY: raw.maxY + hw
                } : null;
                const cmd = new DrawCommand({
                    stroke: this.current_stroke,
                    strokeHistoryRef: stroke_history,
                    redrawFn: () => this.coord.render_all_strokes(stroke_bounds)
                });
                await history_execute_command(cmd, false);

                // 通知消费者笔画已提交
                this.coord.on_stroke_finalized?.(this.current_stroke, stroke_bounds);
            }
        }

        this.current_stroke = null;
        if (this.batch_draw) {
            await this.batch_draw.batch_draw_handle_end();
            this.batch_draw.batch_draw_delete_all();
        }
    }

    // ====== 撤销 & 清空 ======

    async handle_undo() {
        if (history_validate_undo() && !this.is_drawing) {
            await history_handle_undo();
            await this.coord.render_all_strokes();
        }
    }

    async handle_clear(strokeHistoryRef) {
        if (this.is_drawing) return;
        if (!strokeHistoryRef || strokeHistoryRef.length === 0) return;

        const cmd = new ClearCommand({
            savedStrokeHistory: [...strokeHistoryRef],
            savedBaseImageURL: null,
            strokeHistoryRef: strokeHistoryRef,
            baseImageURLRef: {
                get value() { return null; },
                set value(v) {}
            },
            baseImageObjRef: {
                get value() { return null; },
                set value(v) {}
            },
            redrawFn: () => this.coord.render_all_strokes(),
            loadBaseImageFn: () => Promise.resolve()
        });
        await history_execute_command(cmd, false);
        await this.coord.render_all_strokes();
    }

    // ====== 事件处理器 ======

    // --- PointerEvent ---

    handle_pointer_down(e) {
        e.preventDefault();
        if (!this._painting_allowed) return;
        this.draw_canvas_rect = this._get_canvas_rect();
        if (!this.draw_canvas_rect) return;

        const palmResult = is_palm_by_pointer(e);
        if (palmResult.isPalm && (window.DRAW_CONFIG?.palmEraserEnabled !== false)) {
            const size = compute_palm_eraser_size_from_pointer(palmResult.width, palmResult.height);
            this._start_palm_erase(e.clientX, e.clientY, size);
            return;
        }

        const inv = 1 / this._fetch_safe_scale();

        if (this.draw_mode === 'move') {
            const s_origin = this.coord.get_origin();
            this._move_state = {
                start_x: e.clientX - (s_origin?.x || 0),
                start_y: e.clientY - (s_origin?.y || 0)
            };
        } else if (this.draw_mode === 'comment' || this.draw_mode === 'eraser') {
            this.is_drawing = true;
            this.last_x = (e.clientX - this.draw_canvas_rect.left) * inv;
            this.last_y = (e.clientY - this.draw_canvas_rect.top) * inv;
            this._start_stroke(this.draw_mode === 'comment' ? 'draw' : 'erase');
            if (this.draw_mode === 'eraser') {
                this._show_eraser_hint();
                this._update_eraser_hint_position(e.clientX, e.clientY);
            }
        }
    }

    handle_pointer_move(e) {
        e.preventDefault();
        if (!this._painting_allowed) return;

        if (this.isPalmErasing) {
            this._update_palm_erase(e.clientX, e.clientY);
            return;
        }

        if (this.draw_mode === 'eraser' && this.is_drawing) {
            this._update_eraser_hint_position(e.clientX, e.clientY);
        }

        if (this._move_state) {
            const origin = this.coord.get_origin();
            const new_x = e.clientX - this._move_state.start_x;
            const new_y = e.clientY - this._move_state.start_y;
            if (origin.x !== new_x || origin.y !== new_y) {
                this.coord.set_origin?.(new_x, new_y);
            }
            return;
        }

        if (!this.is_drawing || !this.draw_canvas_rect) return;

        const inv = 1 / this._fetch_safe_scale();
        const x = (e.clientX - this.draw_canvas_rect.left) * inv;
        const y = (e.clientY - this.draw_canvas_rect.top) * inv;
        const dx = x - this.last_x;
        const dy = y - this.last_y;

        if (dx * dx + dy * dy > 1) {
            this._save_stroke_point(this.last_x, this.last_y, x, y, e.pressure || 0.5);
            if (this.batch_draw) {
                this.batch_draw.batch_draw_create_command(
                    this.cached_draw_type,
                    this.last_x, this.last_y,
                    x, y,
                    this.cached_draw_color,
                    this.cached_draw_line_width
                );
            }
            this.last_x = x;
            this.last_y = y;
        }
    }

    async handle_pointer_up(e) {
        if (this.isPalmErasing) {
            await this._end_palm_erase();
            return;
        }
        if (this._move_state) {
            this._move_state = null;
            return;
        }
        if (!this.is_drawing) return;
        this.is_drawing = false;
        this.draw_canvas_rect = null;
        if (this.draw_mode === 'eraser') this._hide_eraser_hint();
        await this._submit_stroke();
    }

    // --- MouseEvent fallback ---

    handle_mouse_down(e) {
        this.handle_pointer_down(Object.assign(e, { pressure: 0.5 }));
    }

    handle_mouse_move(e) {
        e.preventDefault();
        if (!this._painting_allowed) return;
        if (this.draw_mode === 'eraser' && this.is_drawing) {
            this._update_eraser_hint_position(e.clientX, e.clientY);
        }
        if (this._move_state) {
            const origin = this.coord.get_origin();
            const new_x = e.clientX - this._move_state.start_x;
            const new_y = e.clientY - this._move_state.start_y;
            if (origin.x !== new_x || origin.y !== new_y) {
                this.coord.set_origin?.(new_x, new_y);
            }
            return;
        }
        if (!this.is_drawing || !this.draw_canvas_rect) return;
        const inv = 1 / this._fetch_safe_scale();
        const x = (e.clientX - this.draw_canvas_rect.left) * inv;
        const y = (e.clientY - this.draw_canvas_rect.top) * inv;
        const dx = x - this.last_x;
        const dy = y - this.last_y;
        if (dx * dx + dy * dy > 1) {
            this._save_stroke_point(this.last_x, this.last_y, x, y, 0.5);
            if (this.batch_draw) {
                this.batch_draw.batch_draw_create_command(
                    this.cached_draw_type,
                    this.last_x, this.last_y,
                    x, y,
                    this.cached_draw_color,
                    this.cached_draw_line_width
                );
            }
            this.last_x = x;
            this.last_y = y;
        }
    }

    /** 为消费者提供的单指绘制处理（用于 touch 事件） */
    _handle_single_touch_draw(touch) {
        if (!this._painting_allowed || !this.draw_canvas_rect) return;
        const inv = 1 / this._fetch_safe_scale();
        const x = (touch.clientX - this.draw_canvas_rect.left) * inv;
        const y = (touch.clientY - this.draw_canvas_rect.top) * inv;
        const pressure = (touch.force > 0) ? touch.force : 0.5;
        const dx = x - this.last_x;
        const dy = y - this.last_y;
        if (dx * dx + dy * dy > 1) {
            this._save_stroke_point(this.last_x, this.last_y, x, y, pressure);
            if (this.batch_draw) {
                this.batch_draw.batch_draw_create_command(
                    this.cached_draw_type,
                    this.last_x, this.last_y,
                    x, y,
                    this.cached_draw_color,
                    this.cached_draw_line_width
                );
            }
            this.last_x = x;
            this.last_y = y;
        }
    }

    // ====== 橡皮擦提示 ======

    init_eraser_hint(container) {
        this._eraser_hint = document.createElement('div');
        this._eraser_hint.className = 'eraser-hint';
        this._eraser_hint.style.width = (window.DRAW_CONFIG?.eraserSize || 15) + 'px';
        this._eraser_hint.style.height = (window.DRAW_CONFIG?.eraserSize || 15) + 'px';
        container.appendChild(this._eraser_hint);

        this._palm_eraser_hint = document.createElement('div');
        this._palm_eraser_hint.className = 'palm-eraser-hint';
        this._palm_eraser_hint.style.width = '60px';
        this._palm_eraser_hint.style.height = '60px';
        container.appendChild(this._palm_eraser_hint);
    }

    _show_eraser_hint() {
        if (!this._eraser_hint) return;
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

            const { clientX, clientY } = this._eraser_hint_pending_pos;
            this._eraser_hint_pending_pos = null;

            const scale = this._fetch_safe_scale();
            const eraser_size = (this.cached_draw_line_width
                || window.DRAW_CONFIG?.eraserSize || 15) * scale;
            this._eraser_hint.style.width = eraser_size + 'px';
            this._eraser_hint.style.height = eraser_size + 'px';

            const rect = this.coord.get_eraser_hint_rect
                ? this.coord.get_eraser_hint_rect()
                : this._get_canvas_rect();
            if (!rect) return;

            const x = clientX - rect.left;
            const y = clientY - rect.top;
            this._eraser_hint.style.left = `${x}px`;
            this._eraser_hint.style.top = `${y}px`;
            this._eraser_hint.style.transform = 'translate(-50%, -50%)';
        });
    }

    /**
     * 外部切换橡皮擦尺寸后立即刷新提示大小和缓存。
     * 由 main_build_eraser_presets 在点击预设按钮时调用。
     */
    refresh_eraser_hint_size() {
        if (!this._eraser_hint) return;
        const size = window.DRAW_CONFIG?.eraserSize || 15;
        this.cached_draw_line_width = size;
        this._eraser_hint.style.width = size + 'px';
        this._eraser_hint.style.height = size + 'px';
    }

    // ====== 手掌擦除 ======

    _show_palm_eraser_hint() {
        if (!this._palm_eraser_hint) return;
        this._palm_eraser_hint.classList.add('active');
    }

    _hide_palm_eraser_hint() {
        if (!this._palm_eraser_hint) return;
        this._palm_eraser_hint.classList.remove('active');
    }

    _update_palm_eraser_hint(clientX, clientY, size) {
        if (!this._palm_eraser_hint) return;
        const rect = this.coord.get_eraser_hint_rect
            ? this.coord.get_eraser_hint_rect()
            : this._get_canvas_rect();
        if (!rect) return;
        const x = clientX - rect.left;
        const y = clientY - rect.top;
        this._palm_eraser_hint.style.width = size + 'px';
        this._palm_eraser_hint.style.height = size + 'px';
        this._palm_eraser_hint.style.left = `${x}px`;
        this._palm_eraser_hint.style.top = `${y}px`;
        this._palm_eraser_hint.style.transform = 'translate(-50%, -50%)';
    }

    _init_palm_session() {
        if (this._palmSession || !window.__palmEraser) return;
        const self = this;
        this._palmSession = new window.__palmEraser.PalmEraserSession({
            getCanvasRect: () => self.draw_canvas_rect,
            getScale: () => self._fetch_safe_scale(),
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
                self.cached_draw_line_width = session.palmEraserSize / Math.max(0.001, self._fetch_safe_scale());
                if (self.batch_draw) {
                    self.batch_draw.batch_draw_init_start();
                    self.batch_draw.eraserShape = 'square';
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
        this.draw_canvas_rect = this._get_canvas_rect();
        if (!this.draw_canvas_rect) return;
        this._init_palm_session();
        if (this._palmSession) this._palmSession.start(clientX, clientY, eraserWidth);
    }

    _update_palm_erase(clientX, clientY) {
        if (this._palmSession) this._palmSession.update(clientX, clientY);
    }

    async _end_palm_erase() {
        if (this._palmSession) await this._palmSession.end();
    }

    // ====== 清理 ======

    destroy() {
        if (this._eraser_hint_raf_id !== null) {
            cancelAnimationFrame(this._eraser_hint_raf_id);
            this._eraser_hint_raf_id = null;
        }
        if (this._eraser_hint?.parentNode) {
            this._eraser_hint.parentNode.removeChild(this._eraser_hint);
        }
        if (this._palm_eraser_hint?.parentNode) {
            this._palm_eraser_hint.parentNode.removeChild(this._palm_eraser_hint);
        }
        this._eraser_hint = null;
        this._palm_eraser_hint = null;
        this._eraser_hint_pending_pos = null;
        this.batch_draw = null;
        this.coord = null;
    }
}
