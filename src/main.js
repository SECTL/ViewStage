/**
 * ViewStage 主逻辑 —— 摄像头及展台应用核心
 * 架构: 图像层(img) + 批注层(canvas)，批注系统含笔画记录/压缩/撤销，图像处理由Rust后端并行
 * 性能: RAF批量绘制减少重绘；Blob URL替代Data URL节省内存
 */

import './modules/canvas/batch-draw.js';
import ThemeManager from './themes/theme.js';
import {
    history_execute_command,
    DrawCommand,
    ClearCommand,
    SnapshotCommand,
    history_validate_undo,
    history_handle_undo,
    history_delete_all,
    history_validate_compact,
    history_fetch_undo_stack,
    history_fetch_commands_to_compact,
    history_format_compact,
    MAX_HISTORY_STEPS
} from './modules/history.js';
import { DocLoader } from './modules/pdf/document_loader.js';
import { InputSource, DragTapSource, PinchZoomSource, PinchZoomSourceV2, ZoomWallDamper, TOLERANCE } from './modules/gesture/index.js';
import { CameraManager, camera_format_blob_to_data_url } from './modules/camera/camera.js';
import { resetContextState, updateContextState } from './modules/canvas/context-state.js';
import { renderStrokesToContext, getPenEffectMode } from './modules/canvas/stroke-renderer.js';
import { createHistoryCompactor } from './modules/canvas/history-compactor.js';

// === 全局变量 ===
let last_canvas_transform = { x: null, y: null, scale: null };
let currentAnimationId = null;
let pending_transform = null;
let transform_raf_id = null;
let wheel_smooth_timer_id = null;
let zoom_complete_timer_id = null;   // 双指缩放结束后延迟批量更新 tile 的定时器


/** 标记缩放进行中，重置延迟批量更新定时器（缩放期间跳过 tile/overlay 逐帧更新） */
function main_set_zooming() {
    if (!state.isZooming) {
        state.isZooming = true;
        // 缩放开始，隐藏 overlay 释放 GPU 合成层
        if (window.batchDrawManager) window.batchDrawManager.hide_overlay();
    }
    if (zoom_complete_timer_id !== null) clearTimeout(zoom_complete_timer_id);
    zoom_complete_timer_id = setTimeout(() => {
        zoom_complete_timer_id = null;
        state.isZooming = false;
        // 缩放结束，恢复 overlay
        if (window.batchDrawManager) window.batchDrawManager.show_overlay();
        // 延迟 tile 重建到浏览器空闲时，避免最后一帧 jank
        const doTileUpdate = () => {
            if (window.tileRenderer) {
                window.tileRenderer.update_visible_tile_dpr(state.scale, false, true);
            }
            if (window.batchDrawManager) {
                window.batchDrawManager.update_overlay_dpr(state.scale);
            }
        };
        if (window.requestIdleCallback) {
            window.requestIdleCallback(doTileUpdate, { timeout: 500 });
        } else {
            doTileUpdate();
        }
    }, 300);
}

/** 取消缩放延迟更新（缩放结束时立即更新） */
function main_cancel_zoom_debounce() {
    if (zoom_complete_timer_id !== null) {
        clearTimeout(zoom_complete_timer_id);
        zoom_complete_timer_id = null;
    }
    state.isZooming = false;
    // 立即恢复 overlay
    if (window.batchDrawManager) window.batchDrawManager.show_overlay();
}

/** 取消缓动动画，移除 CSS transition，防止与新手势冲突 */
function main_cancel_smooth_transform() {
    if (currentAnimationId !== null) {
        clearTimeout(currentAnimationId);
        currentAnimationId = null;
    }
    dom.canvasWrapper.classList.remove('smooth-transform');
    dom.canvasWrapper.style.transitionDuration = '';
    main_cancel_momentum();
}

/** 取消等待中的 transform rAF，防止缩放结束后旧值覆盖新位置 */
function main_cancel_pending_transform() {
    if (transform_raf_id !== null) {
        cancelAnimationFrame(transform_raf_id);
        transform_raf_id = null;
    }
    pending_transform = null;
}

function main_update_transform_schedule(x, y, scale) {
    if (!pending_transform) {
        pending_transform = { x: 0, y: 0, scale: 1 };
    }
    pending_transform.x = x;
    pending_transform.y = y;
    pending_transform.scale = scale;
    
    if (transform_raf_id === null) {
        transform_raf_id = requestAnimationFrame(() => {
            if (pending_transform) {
                const pt = pending_transform;
                dom.canvasWrapper.style.transform = `translate3d(${pt.x}px, ${pt.y}px, 0) scale(${pt.scale})`;
                last_canvas_transform.x = pt.x;
                last_canvas_transform.y = pt.y;
                last_canvas_transform.scale = pt.scale;
                // 缩放进行中时跳过 tile/overlay 更新，由缩放结束定时器统一处理
                if (!state.isZooming) {
                    if (window.tileRenderer) {
                        window.tileRenderer.cancel_idle_shrink();
                        window.tileRenderer.update_visible_tile_dpr(pt.scale);
                    }
                    if (window.batchDrawManager) {
                        window.batchDrawManager.update_overlay_dpr(pt.scale);
                    }
                }
            }
            transform_raf_id = null;
        });
    }
}

// ===== 惯性系统（动量/平滑减速）=====

/** 取消惯性动画 */
function main_cancel_momentum() {
    if (state._momentumRaf !== null) {
        cancelAnimationFrame(state._momentumRaf);
        state._momentumRaf = null;
    }
}

/**
 * 启动惯性动画
 * @param {'xy'|'xyv'} mode - 'xy': 仅平移动量，'xyv': 平移+缩放动量
 */
function main_start_momentum(mode = 'xy') {
    if (!DRAW_CONFIG.momentumEnabled) return;
    main_cancel_smooth_transform();
    if (state._momentumRaf !== null) return;
    state._momentumRaf = requestAnimationFrame(() => main_momentum_tick(mode));
}

const MOMENTUM_FRICTION_BASE = 0.65; // 基础摩擦（低速 ≈ 急停）
const MOMENTUM_FRICTION_MAX = 0.85;  // 极速时摩擦（再快也不会低于此值）
const MOMENTUM_SPEED_SMOOTH = 8;     // 速度特征尺度（px/帧），控制阻力随速度下降的快慢
const MOMENTUM_STOP_THRESHOLD = 0.5; // 速度低于此值停止

function main_calc_adaptive_friction(speed) {
    // 速度越高阻力越小：高速 → 接近 1.0（几乎无衰减），低速 → 接近 BASE（快速停止）
    return MOMENTUM_FRICTION_MAX - (MOMENTUM_FRICTION_MAX - MOMENTUM_FRICTION_BASE) * Math.exp(-speed / MOMENTUM_SPEED_SMOOTH);
}

function main_momentum_tick(mode) {
    let vx = state._gestureVx;
    let vy = state._gestureVy;

    const speed = Math.sqrt(vx * vx + vy * vy);
    const friction = main_calc_adaptive_friction(speed);

    // 应用自适应摩擦衰减
    vx *= friction;
    vy *= friction;
    state._gestureVx = vx;
    state._gestureVy = vy;

    // 应用速度到画布位置
    const prevX = state.canvasX;
    const prevY = state.canvasY;
    state.canvasX += vx;
    state.canvasY += vy;

    // 边界钳制
    main_update_move_bound();
    main_update_canvas_position();

    // 边界碰撞处理：速度归零（防止贴边滑行）
    if (state.canvasX === prevX && vx !== 0) {
        state._gestureVx = 0;
        vx = 0;
    }
    if (state.canvasY === prevY && vy !== 0) {
        state._gestureVy = 0;
        vy = 0;
    }

    // 更新渲染
    main_update_canvas_transform();

    if (Math.abs(vx) > MOMENTUM_STOP_THRESHOLD || Math.abs(vy) > MOMENTUM_STOP_THRESHOLD) {
        state._momentumRaf = requestAnimationFrame(() => main_momentum_tick(mode));
    } else {
        state._momentumRaf = null;
        // 速度归零后做一次缓动到位
        main_update_canvas_transform_smooth(state.canvasX, state.canvasY, state.scale, 150);
    }
}

/** 在 touchmove 中更新速度追踪（每次调用记录当前帧速度） */
function main_update_gesture_velocity(isTwoFinger) {
    const dx = state.canvasX - state._lastCanvasX;
    const dy = state.canvasY - state._lastCanvasY;

    // EMA 平滑：两指用较大惯性（0.4），单指用较小（0.6）
    const alpha = isTwoFinger ? 0.4 : 0.6;
    state._gestureVx = state._gestureVx * (1 - alpha) + dx * alpha;
    state._gestureVy = state._gestureVy * (1 - alpha) + dy * alpha;

    state._lastCanvasX = state.canvasX;
    state._lastCanvasY = state.canvasY;
}

// === PDF.js 配置 ===
function main_init_pdfjs() {
    return DocLoader.init_pdfjs();
}

async function main_wait_pdfjs(maxWait = 5000) {
    return DocLoader.wait_pdfjs(maxWait);
}

// === 全局配置 ===

const DRAW_CONFIG = {
    penColor: null,
    penWidth: 5,
    penSizePresets: [2, 5, 10, 15, 21],
    eraserSize: 15,
    eraserSizePresets: [5, 15, 25, 38, 50],
    eraserSpeedEnabled: false,
    eraserSpeedMinSize: 5,
    eraserSpeedMaxSize: 120,
    eraserSpeedFactor: 0.3,
    palmEraserEnabled: false,
    palmEraserSize: 60,
    momentumEnabled: false,
    pinchZoomV2: true,
    minScale: 0.5,
    maxScale: 3,
    maxScaleImage: 4,
    canvasFill2x: false,
    canvasW: 1000,
    canvasH: 600,
    screenW: 0,
    screenH: 0,
    dprLimit: 2,
    dpr: 1,
    dynamicDprEnabled: true,
    dprMin: 1,
    dprMax: 4,
    dprStep: 0.25,
    imageSmoothingQuality: 'high',
    baseDpr: window.devicePixelRatio || 1,
    canvasBgColor: '#2a2a2a',
    penColors: [
        '#ef4444', '#f97316', '#eab308', '#22c55e', '#06b6d4',
        '#3b82f6', '#6366f1', '#a855f7', '#ec4899', '#f43f5e',
        '#14b8a6', '#64748b', '#1e293b', '#000000', '#ffffff'
    ],
    penSmoothness: 0.8,
    penEffectMode: 'limited',
    penMinWidthRatio: 0.4,
    gestureFrameDelta: 60
};

// 选中第一号颜色作为默认笔色
if (DRAW_CONFIG.penColor === null && DRAW_CONFIG.penColors.length > 0) {
    DRAW_CONFIG.penColor = DRAW_CONFIG.penColors[0];
}

// 将配置暴露到全局，供 batch-draw.js 使用
window.DRAW_CONFIG = DRAW_CONFIG;

// 应用 DPR 限制（0=自动无限制）
function main_calc_capped_dpr(rawDpr, limit) {
    return limit > 0 ? Math.min(rawDpr, limit) : rawDpr;
}
window.main_calc_capped_dpr = main_calc_capped_dpr;

function main_fetch_safe_scale() {
    return Math.max(0.001, state.scale || 1);
}
window.main_fetch_safe_scale = main_fetch_safe_scale;

// === 钢笔笔锋效果管理器 ===
// 使用曲面细分算法，根据速度和压感动态调整线宽
// 效果分级: full(完整) | limited(限制) | off(关闭)

class RealPenManager {
    constructor() {
        this.tessellator = null;
        this.cached_tessellated = new WeakMap();
    }

    init_tessellator() {
        if (!this.tessellator && window.penTessellator) {
            this.tessellator = window.penTessellator;
        }
    }
    
    reset() {
        this.cached_tessellated = new WeakMap();
        this.init_tessellator();
    }
    
    update_position(x, y, timestamp) {
        return 0;
    }
    
    build_tessellated_stroke(stroke, mode = null) {
        this.init_tessellator();
        if (!this.tessellator) return null;
        
        const effect_mode = mode || DRAW_CONFIG.penEffectMode || 'off';
        if (effect_mode === 'off') return null;
        
        if (this.cached_tessellated.has(stroke)) {
            return this.cached_tessellated.get(stroke);
        }
        
        const points = stroke.points;
        const base_width = stroke.lineWidth || DRAW_CONFIG.penWidth;
        const color = stroke.color || DRAW_CONFIG.penColor;
        const storedWidths = stroke.storedWidths;
        
        if (!points || points.length < 1) return null;

        // 有存储宽度时：直接构建 segments，绕过 tessellator 的速度重算
        // 存储宽度来自 batch-draw 的实时计算（含真实指针时序），确保与预览完全一致
        if (storedWidths && storedWidths.length === points.length) {
            const raw = [{ x: points[0].fromX, y: points[0].fromY }];
            for (let i = 0; i < points.length; i++) {
                raw.push({ x: points[i].toX, y: points[i].toY });
            }
            if (raw.length < 2) return null;

            const segments = [];
            for (let i = 0; i < storedWidths.length; i++) {
                segments.push({
                    x1: raw[i].x, y1: raw[i].y,
                    x2: raw[i + 1].x, y2: raw[i + 1].y,
                    line_width: storedWidths[i]
                });
            }

            const result = { segments, color };
            if (result) {
                this.cached_tessellated.set(stroke, result);
            }
            return result;
        }

        // 无存储宽度：走标准 tessellator 速度重算路径
        const raw = [{ x: points[0].fromX, y: points[0].fromY }];
        for (let i = 0; i < points.length; i++) {
            raw.push({ x: points[i].toX, y: points[i].toY });
        }
        if (raw.length < 2) return null;

        const filtered = [raw[0]];
        for (let i = 1; i < raw.length; i++) {
            const prev = filtered[filtered.length - 1];
            const curr = raw[i];
            const dx = curr.x - prev.x;
            const dy = curr.y - prev.y;
            if (dx * dx + dy * dy >= 1) {
                filtered.push(curr);
            }
        }
        if (filtered.length < 2) return null;

        const input_points = filtered;
        if (input_points.length < 2) return null;

        // limited 模式回退：使用常量宽度，不经过速度重算
        if (effect_mode === 'limited') {
            const segments = [];
            for (let i = 0; i < input_points.length - 1; i++) {
                segments.push({
                    x1: input_points[i].x, y1: input_points[i].y,
                    x2: input_points[i + 1].x, y2: input_points[i + 1].y,
                    line_width: base_width
                });
            }
            if (segments.length > 0) {
                const result = { segments, color };
                this.cached_tessellated.set(stroke, result);
                return result;
            }
            return null;
        }

        const stroke_data = [];
        for (let i = 0; i < input_points.length; i++) {
            if (i === 0) {
                stroke_data.push({ fromX: input_points[i].x, fromY: input_points[i].y, toX: input_points[i].x, toY: input_points[i].y });
            } else {
                const prev = input_points[i - 1];
                stroke_data.push({ fromX: prev.x, fromY: prev.y, toX: input_points[i].x, toY: input_points[i].y });
            }
        }

        const result = this.tessellator.tessellator_build_stroke_from_stroke_data(
            { points: stroke_data, lineWidth: base_width, color },
            { density: 1, noStartTaper: stroke.noStartTaper }
        );
        
        if (result) {
            this.cached_tessellated.set(stroke, result);
        }
        return result;
    }

    render_tessellated_stroke(ctx, tessellated_stroke, scaleRatio = 1) {
        this.init_tessellator();
        if (!this.tessellator || !tessellated_stroke) return false;
        
        this.tessellator.tessellator_render_stroke(ctx, tessellated_stroke, scaleRatio);
        return true;
    }
    
    invalidate_cache() {
        this.cached_tessellated = new WeakMap();
    }
}

const realPenManager = new RealPenManager();

function main_stroke_clone(strokes, deep = false) {
    if (!strokes || strokes.length === 0) return [];
    if (deep) {
        return strokes.map(stroke => ({
            type: stroke.type,
            points: stroke.points ? stroke.points.map(p => ({ ...p })) : [],
            color: stroke.color,
            lineWidth: stroke.lineWidth,
            eraserSize: stroke.eraserSize,
            eraserSizeRaw: stroke.eraserSizeRaw,
            scale: stroke.scale,
            bounds: stroke.bounds ? { ...stroke.bounds } : undefined,
            variableWidths: stroke.variableWidths ? [...stroke.variableWidths] : null,
            storedWidths: stroke.storedWidths ? [...stroke.storedWidths] : undefined,
            noStartTaper: stroke.noStartTaper,
            savedStrokeHistory: stroke.savedStrokeHistory ? main_stroke_clone(stroke.savedStrokeHistory, true) : undefined,
            savedBaseImageURL: stroke.savedBaseImageURL
        }));
    }
    return strokes.map(stroke => ({
        type: stroke.type,
        points: stroke.points,
        color: stroke.color,
        lineWidth: stroke.lineWidth,
        eraserSize: stroke.eraserSize,
        eraserSizeRaw: stroke.eraserSizeRaw,
        scale: stroke.scale,
        bounds: stroke.bounds,
        variableWidths: stroke.variableWidths,
        storedWidths: stroke.storedWidths,
        noStartTaper: stroke.noStartTaper,
        savedStrokeHistory: stroke.savedStrokeHistory,
        savedBaseImageURL: stroke.savedBaseImageURL
    }));
}

function main_main_stroke_clone_deep(strokes) {
    return main_stroke_clone(strokes, true);
}

// StrokeQuadTree —— 四叉树空间索引，用于快速查找与脏区域相交的笔画
class StrokeQuadTree {
    constructor(boundary, capacity = 8, maxDepth = 6, depth = 0) {
        this.boundary = boundary;
        this.capacity = capacity;
        this.maxDepth = maxDepth;
        this.depth = depth;
        this.strokes = [];
        this.children = null;
    }
    
    insert(stroke) {
        if (!stroke.bounds) return false;
        
        if (!this.intersects(stroke.bounds)) return false;
        
        if (this.children) {
            return this.insert_to_children(stroke);
        }
        
        this.strokes.push(stroke);
        
        if (this.strokes.length > this.capacity && this.depth < this.maxDepth) {
            this.subdivide();
        }
        
        return true;
    }
    
    insert_to_children(stroke) {
        let inserted = false;
        for (const child of this.children) {
            if (child.insert(stroke)) {
                inserted = true;
            }
        }
        return inserted;
    }
    
    subdivide() {
        const { x, y, width, height } = this.boundary;
        const hw = width / 2;
        const hh = height / 2;
        
        this.children = [
            new StrokeQuadTree({ x, y, width: hw, height: hh }, this.capacity, this.maxDepth, this.depth + 1),
            new StrokeQuadTree({ x: x + hw, y, width: hw, height: hh }, this.capacity, this.maxDepth, this.depth + 1),
            new StrokeQuadTree({ x, y: y + hh, width: hw, height: hh }, this.capacity, this.maxDepth, this.depth + 1),
            new StrokeQuadTree({ x: x + hw, y: y + hh, width: hw, height: hh }, this.capacity, this.maxDepth, this.depth + 1)
        ];
        
        for (const stroke of this.strokes) {
            this.insert_to_children(stroke);
        }
        this.strokes = [];
    }
    
    query(range, found = new Set()) {
        if (!this.intersects(range)) return found;
        
        for (const stroke of this.strokes) {
            if (this.stroke_intersects(stroke, range)) {
                found.add(stroke);
            }
        }
        
        if (this.children) {
            for (const child of this.children) {
                child.query(range, found);
            }
        }
        
        return found;
    }
    
    intersects(bounds) {
        const padding = 5;
        const bMinX = bounds.minX != null ? bounds.minX : bounds.x;
        const bMaxX = bounds.maxX != null ? bounds.maxX : bounds.x + bounds.width;
        const bMinY = bounds.minY != null ? bounds.minY : bounds.y;
        const bMaxY = bounds.maxY != null ? bounds.maxY : bounds.y + bounds.height;
        return !(bMaxX + padding < this.boundary.x ||
                 bMinX - padding > this.boundary.x + this.boundary.width ||
                 bMaxY + padding < this.boundary.y ||
                 bMinY - padding > this.boundary.y + this.boundary.height);
    }
    
    stroke_intersects(stroke, range) {
        if (!stroke.bounds) return true;
        const padding = Math.max(stroke.lineWidth || 5, stroke.eraserSize || 5);
        return !(stroke.bounds.maxX + padding < range.x ||
                 stroke.bounds.minX - padding > range.x + range.width ||
                 stroke.bounds.maxY + padding < range.y ||
                 stroke.bounds.minY - padding > range.y + range.height);
    }
    
    clear() {
        this.strokes = [];
        this.children = null;
    }
    
    build(strokes) {
        this.clear();
        for (const stroke of strokes) {
            this.insert(stroke);
        }
    }
}

// 全局四叉树索引
let strokeQuadTree = null;

// === 全局状态 ===

let state = {
    drawMode: 'move',
    isDrawing: false,
    isDragging: false,
    isScaling: false,
    isZooming: false,         // 双指缩放进行中，用于延迟 tile/overlay 更新
    canvasX: 0,
    canvasY: 0,
    scale: 1,
    lastX: 0,
    lastY: 0,
    cameraViewState: {
        scale: 1,
        canvasX: 0,
        canvasY: 0,
        strokeHistory: [],
        baseImageURL: null
    },
    startDragX: 0,
    startDragY: 0,
    _pinchResidualDrag: false,
    _pinchResidualDragFingerId: null,
    startScale: 1,
    startDistanceSq: 0,
    startScaleX: 0,
    startScaleY: 0,
    startCanvasX: 0,
    startCanvasY: 0,
    startFinger0CX: 0,
    startFinger0CY: 0,

    // 惯性（动量）系统
    _gestureVx: 0,
    _gestureVy: 0,
    _lastCanvasX: 0,
    _lastCanvasY: 0,
    _momentumRaf: null,

    // 缩放边界阻尼器（pinch 开始时创建/重置，见 ZoomWallDamper）
    _zoomDamper: null,

    strokeHistory: [],
    baseImageURL: null,
    baseImageObj: null,
    baseImageLoadId: 0,
    currentStroke: null,
    moveBound: {
        minX: 0, maxX: 0,
        minY: 0, maxY: 0
    },
    cameraStream: null,
    isCameraOpen: false,
    isCameraReady: false,
    cameraAvailable: true,
    isMirrored: false,
    cameraRotation: 0,
    camera_brightness: 10,
    camera_contrast: 1.4,
    useFrontCamera: false,
    defaultCameraId: null,
    cameraWidth: null,
    cameraHeight: null,
    phoneCameraActive: false,
    phoneCameraReady: false,
    wasCameraOpenBeforeMinimize: false,
    currentImage: null,
    imageList: [],
    currentImageIndex: -1,
    fileList: [],
    currentFolderIndex: -1,
    currentFolderPageIndex: -1,
    pdfDocuments: new Map(),
    loadedPages: new Set(),
    currentPressure: 0.5,
    currentVelocity: 0,
    isPalmErasing: false,
    savedDrawMode: null,
    palmEraserSize: 60
};

const MAX_PDF_CACHE = 10;

// === 源ID管理系统 ===
// 统一管理所有源（摄像头、图片、文档）的缩放和批注数据

let sourceIdCounters = {
    pic: 0,
    doc: 0
};

function main_calculate_md5(bytes) {
    const data = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    const original_len = data.length;
    const padded_len = (((original_len + 8) >> 6) + 1) << 6;
    const buffer = new Uint8Array(padded_len);
    buffer.set(data);
    buffer[original_len] = 0x80;

    const bit_len = original_len * 8;
    for (let i = 0; i < 8; i++) {
        buffer[padded_len - 8 + i] = Math.floor(bit_len / Math.pow(256, i)) & 0xff;
    }

    const shifts = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
        5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
        4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
        6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21
    ];
    const table = Array.from({ length: 64 }, (_, i) =>
        Math.floor(Math.abs(Math.sin(i + 1)) * 0x100000000) >>> 0
    );

    let a0 = 0x67452301;
    let b0 = 0xefcdab89;
    let c0 = 0x98badcfe;
    let d0 = 0x10325476;

    const add32 = (a, b) => (a + b) >>> 0;
    const left_rotate = (x, c) => ((x << c) | (x >>> (32 - c))) >>> 0;

    for (let offset = 0; offset < padded_len; offset += 64) {
        const m = new Uint32Array(16);
        for (let i = 0; i < 16; i++) {
            const j = offset + i * 4;
            m[i] = (buffer[j] | (buffer[j + 1] << 8) | (buffer[j + 2] << 16) | (buffer[j + 3] << 24)) >>> 0;
        }

        let a = a0;
        let b = b0;
        let c = c0;
        let d = d0;

        for (let i = 0; i < 64; i++) {
            let f;
            let g;
            if (i < 16) {
                f = (b & c) | (~b & d);
                g = i;
            } else if (i < 32) {
                f = (d & b) | (~d & c);
                g = (5 * i + 1) % 16;
            } else if (i < 48) {
                f = b ^ c ^ d;
                g = (3 * i + 5) % 16;
            } else {
                f = c ^ (b | ~d);
                g = (7 * i) % 16;
            }
            const tmp = d;
            d = c;
            c = b;
            b = add32(b, left_rotate(add32(add32(a, f >>> 0), add32(table[i], m[g])), shifts[i]));
            a = tmp;
        }

        a0 = add32(a0, a);
        b0 = add32(b0, b);
        c0 = add32(c0, c);
        d0 = add32(d0, d);
    }

    const word_to_hex = (word) => {
        let out = '';
        for (let i = 0; i < 4; i++) {
            out += ((word >>> (i * 8)) & 0xff).toString(16).padStart(2, '0');
        }
        return out;
    };

    return word_to_hex(a0) + word_to_hex(b0) + word_to_hex(c0) + word_to_hex(d0);
}

let currentSourceId = null;

let sourceDataStore = {};

const MAX_SOURCE_CACHE = 50;

// 生成源ID
function main_create_source_id(type, pageIndex = null) {
    if (type === 'cam') {
        return 'cam';
    } else if (type === 'pic') {
        sourceIdCounters.pic++;
        return `pic-${sourceIdCounters.pic}`;
    } else if (type === 'doc') {
        if (pageIndex !== null && pageIndex !== undefined) {
            return `doc-${sourceIdCounters.doc}-${pageIndex}`;
        } else {
            console.error('[错误] main_create_source_id: 文档类型必须提供pageIndex参数');
            sourceIdCounters.doc++;
            return `doc-${sourceIdCounters.doc}-unknown`;
        }
    }
    
    console.error(`[错误] main_create_source_id: 未知的类型参数: ${type}`);
    return `unknown-${Date.now()}`;
}

// 保存当前源数据
function main_save_current_source_data() {
    if (!currentSourceId) return;
    
    const keys = Object.keys(sourceDataStore);
    if (keys.length >= MAX_SOURCE_CACHE && !sourceDataStore[currentSourceId]) {
        let oldestKey = null;
        let oldestTime = Infinity;
        
        for (const key of keys) {
            if (sourceDataStore[key].timestamp < oldestTime) {
                oldestTime = sourceDataStore[key].timestamp;
                oldestKey = key;
            }
        }
        
        if (oldestKey) {
            delete sourceDataStore[oldestKey];
            console.log(`[源管理] 缓存已满,移除最旧的源: ${oldestKey}`);
        }
    }
    
    sourceDataStore[currentSourceId] = {
        scale: state.scale,
        canvasX: state.canvasX,
        canvasY: state.canvasY,
        strokeHistory: main_main_stroke_clone_deep(state.strokeHistory),
        baseImageURL: state.baseImageURL,
        timestamp: Date.now()
    };
    
    console.log(`[源管理] 保存数据: ${currentSourceId}, 缩放: ${state.scale.toFixed(2)}, 笔画: ${state.strokeHistory.length}`);
}

// 加载指定源数据
function main_load_source_data(sourceId) {
    if (!sourceId) {
        console.warn('[源管理] main_load_source_data: sourceId为空,跳过加载');
        return;
    }
    
    const data = sourceDataStore[sourceId];
    if (data) {
        state.scale = data.scale || 1;
        state.canvasX = data.canvasX || -(DRAW_CONFIG.canvasW - DRAW_CONFIG.screenW) / 2;
        state.canvasY = data.canvasY || -(DRAW_CONFIG.canvasH - DRAW_CONFIG.screenH) / 2;
        state.strokeHistory = main_main_stroke_clone_deep(data.strokeHistory || []);
        state.baseImageURL = data.baseImageURL || null;
        state.baseImageObj = null;
        history_delete_all();
        
        data.timestamp = Date.now();
        
        console.log(`[源管理] 加载数据: ${sourceId}, 缩放: ${state.scale.toFixed(2)}, 笔画: ${state.strokeHistory.length}`);
    } else {
        // 新源，使用默认值
        state.scale = main_default_scale();
        state.canvasX = -(DRAW_CONFIG.canvasW - DRAW_CONFIG.screenW) / 2;
        state.canvasY = -(DRAW_CONFIG.canvasH - DRAW_CONFIG.screenH) / 2;
        state.strokeHistory = [];
        state.baseImageURL = null;
        state.baseImageObj = null;
        history_delete_all();
        
        console.log(`[源管理] 新源初始化: ${sourceId}`);
    }
    
    currentSourceId = sourceId;
}

// 切换到新源：保存当前源 → 加载目标源 → 重绘 → 刷新UI
async function main_update_source(newSourceId) {
    main_save_current_source_data();
    main_load_source_data(newSourceId);
    main_delete_draw_canvas();
    if (state.strokeHistory.length > 0) {
        await main_render_all_strokes();
    }
    main_update_move_bound();
    main_update_canvas_position();
    main_update_canvas_transform();
    main_update_history_button_status();
}

let dom = {};  // DOM 元素引用缓存

// 将 dom 暴露到全局，供 batch-draw.js 使用
window.dom = dom;
window.state = state;

const cameraManager = new CameraManager({
    state, dom, DRAW_CONFIG, ThemeManager,
    saveCurrentSourceData: () => main_save_current_source_data(),
    updateSource: (id) => main_update_source(id),
    updateSettingsControlsState: () => main_update_settings_controls_state(),
    deleteSidebarSelection: () => main_delete_sidebar_selection(),
    renderImageCentered: (img) => main_render_image_centered(img),
    deleteImageLayer: () => main_delete_image_layer(),
    deleteDrawCanvas: () => main_delete_draw_canvas(),
    historyDeleteAll: () => history_delete_all(),
    resetSourceId: () => { currentSourceId = null; },
    showErrorDialog: (t, d) => main_show_error_dialog(t, d),
    showSidebarIfHidden: () => main_show_sidebar_if_hidden(),
    saveImageToList: (img, name, filter) => main_save_image_to_list_no_highlight(img, name, filter),
    updateSidebarContent: () => main_update_sidebar_content(),
    updateCanvasBgColor: (c) => main_update_canvas_bg_color(c),
    updateCanvasTransform: () => main_update_canvas_transform(),
    updateMoveBound: () => main_update_move_bound(),
    updateCanvasPosition: () => main_update_canvas_position(),
    updatePhotoButtonState: () => cameraManager.updatePhotoButtonState(),
});
window.cameraManager = cameraManager;

const historyCompactor = createHistoryCompactor({
    state,
    cloneStrokeDeep: (strokes) => main_main_stroke_clone_deep(strokes),
    fetchOffscreenCanvas: () => main_fetch_offscreen_canvas(),
    releaseOffscreenCanvas: (c) => main_release_offscreen_canvas(c),
    renderAllStrokes: (bounds) => main_render_all_strokes(bounds),
    loadBaseImage: (url) => main_load_base_image(url),
    safeScaleFn: main_fetch_safe_scale,
    penManager: () => realPenManager,
    historyValidateCompact: history_validate_compact,
    historyFetchUndoStack: history_fetch_undo_stack,
    historyFetchCommandsToCompact: history_fetch_commands_to_compact,
    historyFormatCompact: history_format_compact,
    SnapshotCommand,
});

let cachedCanvasRect = null;
let cachedVisibleRect = null;
let cachedVisibleRectScale = null;
let cachedVisibleRectX = null;
let cachedVisibleRectY = null;

const OFFSCREEN_MAX_PHYSICAL = 3840;
const OFFSCREEN_POOL_MAX = 2;
const OFFSCREEN_POOL_IDLE_MS = 30000;
const _offscreenPool = [];
let _offscreenPoolTimer = null;

function main_clear_offscreen_pool() {
    for (const entry of _offscreenPool) {
        entry.canvas = null;
        entry.ctx = null;
    }
    _offscreenPool.length = 0;
}

function main_schedule_offscreen_pool_evict() {
    clearTimeout(_offscreenPoolTimer);
    _offscreenPoolTimer = setTimeout(() => {
        _offscreenPoolTimer = null;
        main_clear_offscreen_pool();
    }, OFFSCREEN_POOL_IDLE_MS);
}

function main_fetch_offscreen_canvas() {
    clearTimeout(_offscreenPoolTimer);
    let w = DRAW_CONFIG.canvasW * DRAW_CONFIG.dpr;
    let h = DRAW_CONFIG.canvasH * DRAW_CONFIG.dpr;
    if (w > OFFSCREEN_MAX_PHYSICAL || h > OFFSCREEN_MAX_PHYSICAL) {
        const s = OFFSCREEN_MAX_PHYSICAL / Math.max(w, h);
        w = Math.round(w * s);
        h = Math.round(h * s);
    }
    let entry;
    for (let i = _offscreenPool.length - 1; i >= 0; i--) {
        if (_offscreenPool[i].canvas.width >= w && _offscreenPool[i].canvas.height >= h) {
            entry = _offscreenPool.splice(i, 1)[0];
            break;
        }
    }
    if (!entry) {
        const canvas = document.createElement('canvas');
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext('2d', { alpha: true });
        entry = { canvas, ctx };
    }
    entry.canvas.width = w;
    entry.canvas.height = h;
    entry.ctx.setTransform(1, 0, 0, 1, 0, 0);
    entry.ctx.scale(w / DRAW_CONFIG.canvasW, h / DRAW_CONFIG.canvasH);
    return entry;
}

function main_release_offscreen_canvas(offscreen) {
    if (_offscreenPool.length < OFFSCREEN_POOL_MAX) {
        _offscreenPool.push(offscreen);
    }
    main_schedule_offscreen_pool_evict();
}

function main_delete_cached_rect() {
    cachedCanvasRect = null;
}

function main_fetch_cached_canvas_rect() {
    if (!cachedCanvasRect) {
        cachedCanvasRect = dom.canvasContainer.getBoundingClientRect();
    }
    return cachedCanvasRect;
}

// 监听系统关联打开的PDF文件
function main_setup_pdf_file_open() {
    if (!window.__TAURI__) {
        console.log('非 Tauri 环境，跳过文件打开监听');
        return;
    }
    
    console.log('开始注册文件打开事件监听...');
    
    const { listen } = window.__TAURI__.event;
    
    listen('file-opened', (event) => {
        console.log('========== 收到文件打开事件 ==========');
        console.log('完整事件对象:', JSON.stringify(event, null, 2));
        console.log('Payload 类型:', typeof event.payload);
        console.log('Payload 内容:', event.payload);
        
        let filePath = event.payload;
        
        if (typeof filePath === 'string') {
            if (filePath.startsWith('file://')) {
                filePath = decodeURIComponent(filePath.replace('file://', ''));
            }
            console.log('最终文件路径:', filePath);
            main_load_pdf_from_path(filePath, true);
        } else {
            console.error('无法解析文件路径，payload:', event.payload);
            main_show_error_dialog(
                window.i18n?.format_translate('errors.fileError') || '文件错误',
                window.i18n?.format_translate('errors.fileParseError') || '无法解析文件路径'
            );
        }
    }).then(() => {
        console.log('file-opened 事件监听注册成功');
    }).catch(err => {
        console.error('注册 file-opened 事件监听失败:', err);
    });
    
    listen('opener://open-file', (event) => {
        console.log('========== 收到 opener 事件 ==========');
        console.log('完整事件对象:', JSON.stringify(event, null, 2));
        
        let filePath = null;
        
        if (typeof event.payload === 'string') {
            filePath = event.payload;
        } else if (event.payload && typeof event.payload === 'object') {
            filePath = event.payload.path || event.payload.url || event.payload.filePath || event.payload.uri;
        }
        
        if (filePath) {
            if (filePath.startsWith('file://')) {
                filePath = decodeURIComponent(filePath.replace('file://', ''));
            }
            console.log('最终文件路径:', filePath);
            main_load_pdf_from_path(filePath, true);
        }
    }).catch(err => {
        console.log('opener 事件监听可选:', err);
    });
    
    listen('rotate-image', (event) => {
        const direction = event.payload;
        main_update_image_rotation(direction);
    }).catch(err => {
        console.error('rotate-image 事件监听失败:', err);
    });
    
    listen('mirror-changed', (event) => {
        state.isMirrored = event.payload;
        if (state.isCameraOpen) {
            main_update_camera_video_style();
        }
        console.log('镜像状态已更改:', state.isMirrored);
    }).catch(err => {
        console.error('mirror-changed 事件监听失败:', err);
    });
    
    listen('switch-camera', () => {
        main_update_camera();
        console.log('切换摄像头');
    }).catch(err => {
        console.error('switch-camera 事件监听失败:', err);
    });
    
    listen('settings-changed', (event) => {
        const settings = event.payload;
        console.log('收到设置更改通知:', settings);
        
        let needRestartCamera = false;
        
        if (settings.defaultCamera !== undefined) {
            state.defaultCameraId = settings.defaultCamera;
            console.log('默认摄像头已更改:', settings.defaultCamera);
            needRestartCamera = true;
        }
        
        if (settings.cameraWidth !== undefined && settings.cameraHeight !== undefined) {
            state.cameraWidth = settings.cameraWidth;
            state.cameraHeight = settings.cameraHeight;
            console.log('摄像头分辨率已更改:', settings.cameraWidth, 'x', settings.cameraHeight);
            needRestartCamera = true;
        }
        
        if (settings.dynamicDprEnabled !== undefined) {
            DRAW_CONFIG.dynamicDprEnabled = settings.dynamicDprEnabled;
        }
        if (settings.dprMin !== undefined) {
            DRAW_CONFIG.dprMin = settings.dprMin;
        }
        if (settings.dprMax !== undefined) {
            DRAW_CONFIG.dprMax = settings.dprMax;
        }
        if (settings.dprStep !== undefined) {
            DRAW_CONFIG.dprStep = settings.dprStep;
        }
        if (settings.overlayDpr !== undefined) {
            DRAW_CONFIG.overlayDpr = settings.overlayDpr;
        }
        if (settings.dynamicDprEnabled !== undefined || settings.dprMin !== undefined ||
            settings.dprMax !== undefined || settings.dprStep !== undefined ||
            settings.overlayDpr !== undefined) {
            if (window.tileRenderer) {
                window.tileRenderer.update_visible_tile_dpr(state.scale, true, true);
            }
            if (window.batchDrawManager) {
                window.batchDrawManager.update_overlay_dpr(state.scale, true);
            }
            // 同步阅读器和黑板
            window.sync_all_overlay_dpr?.();
        }

        if (settings.penColors && Array.isArray(settings.penColors)) {
            DRAW_CONFIG.penColors = settings.penColors.map(color => {
                if (typeof color === 'object' && color.r !== undefined) {
                    return main_calc_rgb_to_hex(color.r, color.g, color.b);
                }
                return color;
            });
            main_update_color_buttons();
            console.log('画笔颜色已更改:', DRAW_CONFIG.penColors);
        }
        
        if (settings.penWidth !== undefined) {
            DRAW_CONFIG.penWidth = settings.penWidth;
            main_build_pen_presets(DRAW_CONFIG.penSizePresets);
        }
        if (settings.eraserSize !== undefined) {
            DRAW_CONFIG.eraserSize = settings.eraserSize;
            main_update_eraser_hint_size();
            if (window.blackboardManager?.drawing_engine) {
                window.blackboardManager.drawing_engine.refresh_eraser_hint_size();
            }
            main_build_eraser_presets(DRAW_CONFIG.eraserSizePresets);
        }
        
        if (settings.penSizePresets && Array.isArray(settings.penSizePresets)) {
            DRAW_CONFIG.penSizePresets = settings.penSizePresets;
            main_build_pen_presets(settings.penSizePresets);
            console.log('画笔预设已更改:', settings.penSizePresets);
        }
        
        if (settings.eraserSizePresets && Array.isArray(settings.eraserSizePresets)) {
            DRAW_CONFIG.eraserSizePresets = settings.eraserSizePresets;
            main_build_eraser_presets(settings.eraserSizePresets);
            if (window.blackboardManager?.drawing_engine) {
                window.blackboardManager.drawing_engine.refresh_eraser_hint_size();
            }
            console.log('橡皮擦预设已更改:', settings.eraserSizePresets);
        }
        
        if (settings.theme !== undefined) {
            ThemeManager.theme_update_active(settings.theme).then(() => {
                const canvasBgColor = ThemeManager.theme_fetch_canvas_bg_color();
                DRAW_CONFIG.canvasBgColor = canvasBgColor;
                main_update_canvas_bg_color(canvasBgColor);
                
                const noCameraMsg = document.getElementById('noCameraMessage');
                if (noCameraMsg && noCameraMsg.style.display !== 'none') {
                    const style = ThemeManager.theme_fetch_no_camera_style();
                    noCameraMsg.innerHTML = `
                        <div style="font-size: 2.5vw; color: ${style.textColor}; margin-bottom: 2vh; text-shadow: ${style.textShadow};">( $ _ $ )</div>
                        <div style="font-size: 1.2vw; color: ${style.secondaryTextColor}; margin-bottom: 1vh; text-shadow: ${style.textShadow};">${window.i18n?.format_translate('camera.deviceNotFound') || '找不到展台设备'}</div>
                        <div style="font-size: 0.9vw; color: ${style.tertiaryTextColor}; text-shadow: ${style.textShadow};">${noCameraMsg.dataset.message || ''}</div>
                    `;
                }
                
                console.log('主题已更改:', settings.theme);
            });
        }

        if (settings.blurEnabled !== undefined) {
            document.body.classList.toggle('blur-enabled', settings.blurEnabled === true);
        }

        if (settings.penMinWidthRatio !== undefined && DRAW_CONFIG.developerMode) {
            DRAW_CONFIG.penMinWidthRatio = settings.penMinWidthRatio;
        }
        if (settings.maxScaleImage !== undefined && DRAW_CONFIG.developerMode) {
            DRAW_CONFIG.maxScaleImage = settings.maxScaleImage;
        }
        if (settings.gestureFrameDelta !== undefined && DRAW_CONFIG.developerMode) {
            DRAW_CONFIG.gestureFrameDelta = settings.gestureFrameDelta;
        }

        // 性能监视器动态开关（仅在开发者模式下生效）
        if (settings.perfMonitorEnabled !== undefined && DRAW_CONFIG.developerMode) {
            DRAW_CONFIG.perfMonitorEnabled = settings.perfMonitorEnabled;
            const interval = settings.perfMonitorInterval || 200;
            if (settings.perfMonitorEnabled) {
                if (!window.perfMonitor) {
                    import('./modules/developer/perf-monitor.js').then(mod => {
                        window.perfMonitor = mod;
                        mod.perf_monitor_init(interval);
                    }).catch(e => {
                        console.error('动态加载 perf monitor 失败:', e);
                    });
                } else {
                    window.perfMonitor.perf_monitor_set_enabled(true, interval);
                }
            } else {
                if (window.perfMonitor) {
                    window.perfMonitor.perf_monitor_set_enabled(false);
                }
            }
        } else if (settings.perfMonitorInterval !== undefined && DRAW_CONFIG.developerMode && window.perfMonitor) {
            // 仅更新频率（不改变开关状态）
            window.perfMonitor.perf_monitor_set_interval(settings.perfMonitorInterval);
        }

        if (needRestartCamera && state.isCameraOpen) {
            console.log('摄像头设置已更改，重新初始化摄像头...');
            main_update_camera_state(false).then(() => {
                setTimeout(() => {
                    main_update_camera_state(true);
                }, 300);
            });
        }
    }).catch(err => {
        console.error('settings-changed 事件监听失败:', err);
    });
    
}

async function main_render_pdf_pages_lazy(pdf, totalPages, initialPages = 3, docNumber = null) {
    return DocLoader.render_pdf_pages_lazy(pdf, totalPages, initialPages, docNumber);
}

const PDF_INITIAL_RENDER_PAGES = 20;

async function main_load_pdf_from_path(filePath, autoOpen = false) {
    if (currentSourceId) {
        main_save_current_source_data();
    }
    
    const wasCameraOpen = state.isCameraOpen;
    
    if (state.isCameraOpen) {
        await main_update_camera_state(false);
    }
    
    console.log('开始加载文件:', filePath);
    
    const fileName_lower = filePath.toLowerCase();
    const isWord = fileName_lower.endsWith('.docx') || fileName_lower.endsWith('.doc');
    
    if (isWord) {
        main_show_loading_overlay(window.i18n?.format_translate('loading.detectingOffice') || '正在检测 Office 软件...');
        
        const { invoke } = window.__TAURI__.core;
        const { fs } = window.__TAURI__;
        
        let detection;
        try {
            detection = await invoke('office_detect_all');
            console.log('Office 检测结果:', detection);
            if (detection.recommended === 'None') {
                main_hide_loading_overlay();
                main_show_error_dialog(
                    window.i18n?.format_translate('errors.officeNotInstalled') || 'Office 未安装',
                    window.i18n?.format_translate('errors.officeNotInstalledDesc') || '未检测到可用的 Office 软件\n\n请安装以下软件之一：\n• Microsoft Word\n• WPS Office\n• LibreOffice\n\n或将 Word 文档另存为 PDF 后导入'
                );
                if (wasCameraOpen) await main_update_camera_state(true);
                return;
            }
        } catch (e) {
            main_hide_loading_overlay();
            console.log('检测 Office 失败:', e);
            main_show_error_dialog(
                window.i18n?.format_translate('errors.officeDetectFailed') || '检测失败',
                window.i18n?.format_translate('errors.officeDetectFailedDesc') || '检测 Office 软件失败，请重试'
            );
            if (wasCameraOpen) await main_update_camera_state(true);
            return;
        }
        
        main_update_loading_progress(window.i18n?.format_translate('loading.readingFile') || '正在读取文件...');
        
        let fileData;
        try {
            fileData = await fs.readFile(filePath);
        } catch (readError) {
            main_hide_loading_overlay();
            console.error('文件读取失败:', readError);
            main_show_error_dialog(
                window.i18n?.format_translate('errors.readFailed') || '读取失败',
                window.i18n?.format_translate('errors.readFailedDesc') || '无法读取文件'
            );
            if (wasCameraOpen) await main_update_camera_state(true);
            return;
        }
        
        let uint8Array;
        if (Array.isArray(fileData)) {
            uint8Array = new Uint8Array(fileData);
        } else {
            uint8Array = new Uint8Array(fileData);
        }
        
        console.log('文件大小:', uint8Array.length, '字节');
        const fileMd5 = main_calculate_md5(uint8Array);
        
        main_update_loading_progress(window.i18n?.format_translate('loading.processingWord') || '正在处理 Word 文档...');
        
        const fileName = filePath.split(/[\\/]/).pop();
        const fileDataForConvert = Array.from(uint8Array);
        fileData = null;
        uint8Array = null;
        
        let pdfPath = null;
        try {
            pdfPath = await invoke('office_convert_docx_to_pdf_bytes', {
                fileData: fileDataForConvert,
                fileName: fileName
            });
            console.log('Word 文档已转换为 PDF:', pdfPath);
        } catch (convertError) {
            main_hide_loading_overlay();
            console.error('Word 转换失败:', convertError);
            const errorMsg = String(convertError);
            let friendlyMsg = window.i18n?.format_translate('errors.wordConvertFailed') || 'Word 文档转换失败';
            
            if (errorMsg.includes('Office') || errorMsg.includes('Word') || errorMsg.includes('WPS')) {
                friendlyMsg = window.i18n?.format_translate('errors.officeCallFailed') || 'Office 软件调用失败\n\n可能的原因：\n• Office 软件未正确安装\n• 文件被其他程序占用\n• 文件格式不支持';
            }
            
            main_show_error_dialog(
                window.i18n?.format_translate('errors.convertFailed') || '转换失败',
                friendlyMsg,
                () => {
                    main_load_pdf_from_path(filePath);
                }
            );
            if (wasCameraOpen) await main_update_camera_state(true);
            return;
        }
        
        main_update_loading_progress(window.i18n?.format_translate('loading.renderingPage') || '正在渲染页面...');
        
        try {
            const pdfReady = await main_wait_pdfjs();
            if (!pdfReady) {
                main_hide_loading_overlay();
                console.error('PDF.js 库加载超时');
                main_show_error_dialog(
                    window.i18n?.format_translate('errors.loadFailed') || '加载失败',
                    window.i18n?.format_translate('errors.pdfLoadTimeout') || 'PDF 库加载超时\n\n请重启应用后重试'
                );
                if (wasCameraOpen) await main_update_camera_state(true);
                return;
            }
            
            let pdfBytes = await fs.readFile(pdfPath);
            let pdfArrayBuffer = pdfBytes.buffer;
            const pdf = await pdfjsLib.getDocument({
                data: pdfArrayBuffer,
                enableXfa: false,
                useSystemFonts: false,
                isEvalSupported: false
            }).promise;
            pdfBytes = null;
            pdfArrayBuffer = null;
            console.log('PDF加载成功，页数:', pdf.numPages);
            
            const totalPages = pdf.numPages;
            const fileName = filePath.split(/[/\\]/).pop().replace(/\.(pdf|docx|doc)$/i, '');
            const docNumber = sourceIdCounters.doc++;
            
            const folder = {
                name: fileName,
                pages: [],
                isPdf: true,
                pdfDoc: pdf,
                totalPages: totalPages,
                docNumber: docNumber,
                fileMd5: fileMd5
            };
            
            if (state.pdfDocuments.size >= MAX_PDF_CACHE) {
                const firstKey = state.pdfDocuments.keys().next().value;
                main_delete_pdf_blob_urls(firstKey);
                state.pdfDocuments.delete(firstKey);
                console.log(`[PDF缓存] 缓存已满,移除文档: ${firstKey}`);
            }
            
            state.pdfDocuments.set(docNumber, pdf);
            
            const processedPages = await main_render_pdf_pages_lazy(pdf, totalPages, PDF_INITIAL_RENDER_PAGES, docNumber);
            folder.pages = processedPages;
            
            state.fileList.push(folder);
            main_update_file_sidebar_content();
            main_show_file_sidebar();
            
            main_hide_loading_overlay();
            console.log(`文件已导入: ${folder.name}，共${folder.pages.length}页`);

            // PDF 加载后内存占用可能较高，自动清理
            const invoke = window.__TAURI__?.core?.invoke;
            if (invoke) {
                const now = Date.now();
                if (!window.__memclean_last_auto || now - window.__memclean_last_auto >= 600000) {
                    invoke('memreduct_get_usage').then(usage => {
                        if (usage > 80) {
                            console.log(`[memclean] PDF加载后内存使用率 ${usage}%，自动清理`);
                            window.__memclean_last_auto = Date.now();
                            invoke('memreduct_clean_now', { mask: null }).catch(() => {});
                        }
                    }).catch(() => {});
                }
            }
            
            if (autoOpen && window.documentReaderManager) {
                const fileIndex = state.fileList.length - 1;
                main_hide_file_sidebar();
                window.documentReaderManager.open(fileIndex);
            }
            
            if (wasCameraOpen) await main_update_camera_state(true);
            
            try {
                await fs.remove(pdfPath);
            } catch (e) {
                console.log('清理转换的 PDF 失败:', e);
            }
        } catch (error) {
            main_hide_loading_overlay();
            console.error('文件导入失败:', error);
            main_show_error_dialog(
                window.i18n?.format_translate('errors.importFailed') || '导入失败',
                window.i18n?.format_translate('errors.importFailedDesc') || '文件导入失败，请确保文件格式正确'
            );
            if (wasCameraOpen) await main_update_camera_state(true);
        }
        
        return;
    }
    
    main_show_loading_overlay(window.i18n?.format_translate('loading.importingFile') || '正在导入文件...');
    
    try {
        const pdfReady = await main_wait_pdfjs();
        if (!pdfReady) {
            main_hide_loading_overlay();
            console.error('PDF.js 库加载超时');
            main_show_error_dialog(
                window.i18n?.format_translate('errors.loadFailed') || '加载失败',
                window.i18n?.format_translate('errors.pdfLoadTimeout') || 'PDF 库加载超时\n\n请重启应用后重试'
            );
            if (wasCameraOpen) await main_update_camera_state(true);
            return;
        }
        
        const { fs } = window.__TAURI__;
        
        let fileData;
        try {
            fileData = await fs.readFile(filePath);
            console.log('文件读取成功，数据类型:', typeof fileData, '是否数组:', Array.isArray(fileData));
        } catch (readError) {
            console.error('文件读取失败:', readError);
            main_hide_loading_overlay();
            main_show_error_dialog(
                window.i18n?.format_translate('errors.readFailed') || '读取失败',
                window.i18n?.format_translate('errors.readFailedDesc') || '无法读取文件'
            );
            if (wasCameraOpen) await main_update_camera_state(true);
            return;
        }
        
        let uint8Array;
        if (Array.isArray(fileData)) {
            uint8Array = new Uint8Array(fileData);
        } else if (fileData instanceof ArrayBuffer) {
            uint8Array = new Uint8Array(fileData);
        } else {
            uint8Array = new Uint8Array(fileData);
        }
        
        console.log('PDF数据大小:', uint8Array.length);

        // PDF 解析（Worker 线程）与 MD5（主线程）并发执行
        const pdfPromise = pdfjsLib.getDocument({
            data: uint8Array,
            enableXfa: false,
            useSystemFonts: false,
            isEvalSupported: false
        }).promise;
        const fileMd5 = main_calculate_md5(uint8Array);
        const pdf = await pdfPromise;
        fileData = null;
        uint8Array = null;
        console.log('PDF加载成功，页数:', pdf.numPages);
        
        const totalPages = pdf.numPages;
        const fileName = filePath.split(/[/\\]/).pop().replace(/\.(pdf|docx|doc)$/i, '');
        const docNumber = sourceIdCounters.doc++;
        
        const folder = {
            name: fileName,
            pages: [],
            isWord: false,
            pdfDoc: pdf,
            totalPages: totalPages,
            docNumber: docNumber,
            fileMd5: fileMd5
        };
        
        const processedPages = await main_render_pdf_pages_lazy(pdf, totalPages, PDF_INITIAL_RENDER_PAGES, docNumber);
        folder.pages = processedPages;
        
        state.fileList.push(folder);
        main_update_file_sidebar_content();
        main_show_file_sidebar();
        
        main_hide_loading_overlay();
        console.log(`文件已导入: ${folder.name}，共${folder.pages.length}页`);
        
        if (autoOpen && window.documentReaderManager) {
            const fileIndex = state.fileList.length - 1;
            main_hide_file_sidebar();
            window.documentReaderManager.open(fileIndex);
        }
        
        if (wasCameraOpen) await main_update_camera_state(true);
    } catch (error) {
        main_hide_loading_overlay();
        console.error('文件导入失败:', error);
        main_show_error_dialog(
            window.i18n?.format_translate('errors.importFailed') || '导入失败',
            window.i18n?.format_translate('errors.importFailedDesc') || '文件导入失败，请确保文件格式正确'
        );
        if (wasCameraOpen) await main_update_camera_state(true);
    }
}

// 处理窗口大小变化（防抖 150ms）
let resizeTimeout = null;

function main_handle_resize() {
    main_delete_cached_rect();
    if (resizeTimeout) clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
        resizeTimeout = null;
        const container = dom.canvasContainer;
    const newScreenW = Math.max(1, container.clientWidth);
    const newScreenH = Math.max(1, container.clientHeight);
        
        if (newScreenW !== DRAW_CONFIG.screenW || newScreenH !== DRAW_CONFIG.screenH) {
            main_update_canvas_size(newScreenW, newScreenH);
        }
    }, 150);
}

// 调整画布大小
async function main_update_canvas_size(newScreenW, newScreenH) {
    const oldScale = state.scale;
    const oldCanvasX = state.canvasX;
    const oldCanvasY = state.canvasY;
    
    if (window.tileRenderer) {
        window.tileRenderer.destroy_all();
    }
    
    DRAW_CONFIG.screenW = Math.max(1, newScreenW);
    DRAW_CONFIG.screenH = Math.max(1, newScreenH);
    
    DRAW_CONFIG.canvasW = Math.max(1, Math.floor(newScreenW * 2));
    DRAW_CONFIG.canvasH = Math.max(1, Math.floor(newScreenH * 2));
    
    DRAW_CONFIG.dpr = window.main_calc_capped_dpr(DRAW_CONFIG.baseDpr, DRAW_CONFIG.dprLimit);
    
    main_update_move_bound();
    
    dom.imageElement.style.width = DRAW_CONFIG.canvasW + 'px';
    dom.imageElement.style.height = DRAW_CONFIG.canvasH + 'px';
    dom.canvasWrapper.style.width = DRAW_CONFIG.canvasW + 'px';
    dom.canvasWrapper.style.height = DRAW_CONFIG.canvasH + 'px';
    
    window.tileRenderer.init_tiles(dom.canvasWrapper);
    
    if (window.batchDrawManager) {
        window.batchDrawManager.resize_overlay(newScreenW, newScreenH, DRAW_CONFIG.dpr);
    }
    
    if (state.currentImage) {
        main_render_image_centered(state.currentImage);
    }

    // 修复：窗口缩放时摄像头未重排（开启 2x 满铺后会错位）
    if (state.isCameraOpen) main_update_camera_video_style();

    if (state.strokeHistory.length > 0 || state.baseImageObj) {
        await main_render_all_strokes();
    }
    
    state.scale = oldScale;
    state.canvasX = oldCanvasX;
    state.canvasY = oldCanvasY;
    
    main_update_move_bound();
    main_update_canvas_position();
    main_update_canvas_transform();
    
    console.log(`窗口调整: 屏幕 ${newScreenW}x${newScreenH}, 画布 ${DRAW_CONFIG.canvasW}x${DRAW_CONFIG.canvasH}, DPR ${DRAW_CONFIG.dpr.toFixed(2)}`);
}

// 更新画布背景颜色
function main_update_canvas_bg_color(color) {
    if (dom.canvasContainer) {
        dom.canvasContainer.style.backgroundColor = color;
    }
    if (dom.canvasWrapper) {
        dom.canvasWrapper.style.backgroundColor = color;
    }
}

let cachedMoveBoundScale = null;
let cachedMoveBoundScreenW = -1;
let cachedMoveBoundScreenH = -1;

function main_update_move_bound() {
    // 缓存键必须包含屏幕尺寸：resize 只改 screenW/H 而不动 scale，
    // 若仅按 scale 缓存，resize 后边界过期 → 画布可被拖出屏幕外。
    // 用数值分量比较而非字符串拼接，避免 pinch/惯性每帧分配字符串
    if (cachedMoveBoundScale === state.scale &&
        cachedMoveBoundScreenW === DRAW_CONFIG.screenW &&
        cachedMoveBoundScreenH === DRAW_CONFIG.screenH) {
        return;
    }
    cachedMoveBoundScale = state.scale;
    cachedMoveBoundScreenW = DRAW_CONFIG.screenW;
    cachedMoveBoundScreenH = DRAW_CONFIG.screenH;
    
    const screenW = DRAW_CONFIG.screenW;
    const screenH = DRAW_CONFIG.screenH;
    const scaledW = DRAW_CONFIG.canvasW * state.scale;
    const scaledH = DRAW_CONFIG.canvasH * state.scale;
    
    if (scaledW >= screenW) {
        state.moveBound.minX = -(scaledW - screenW);
        state.moveBound.maxX = 0;
    } else {
        state.moveBound.minX = (screenW - scaledW) / 2;
        state.moveBound.maxX = (screenW - scaledW) / 2;
    }
    
    if (scaledH >= screenH) {
        state.moveBound.minY = -(scaledH - screenH);
        state.moveBound.maxY = 0;
    } else {
        state.moveBound.minY = (screenH - scaledH) / 2;
        state.moveBound.maxY = (screenH - scaledH) / 2;
    }
}

function main_update_canvas_position() {
    const eps = 0.001;
    state.canvasX = Math.max(state.moveBound.minX - eps, Math.min(state.moveBound.maxX + eps, state.canvasX));
    state.canvasY = Math.max(state.moveBound.minY - eps, Math.min(state.moveBound.maxY + eps, state.canvasY));
}

function main_fetch_visible_rect() {
    if (cachedVisibleRectScale === state.scale && 
        cachedVisibleRectX === state.canvasX && 
        cachedVisibleRectY === state.canvasY && 
        cachedVisibleRect) {
        return cachedVisibleRect;
    }
    
    cachedVisibleRectScale = state.scale;
    cachedVisibleRectX = state.canvasX;
    cachedVisibleRectY = state.canvasY;
    
    // 确保缩放系数 > 0，防止除以零
    const scale = Math.max(0.01, state.scale || 1);
    const screenW = DRAW_CONFIG.screenW || 1;
    const screenH = DRAW_CONFIG.screenH || 1;
    
    let visibleX = Math.max(0, -state.canvasX / scale);
    let visibleY = Math.max(0, -state.canvasY / scale);
    let visibleW = Math.min(DRAW_CONFIG.canvasW - visibleX, screenW / scale);
    let visibleH = Math.min(DRAW_CONFIG.canvasH - visibleY, screenH / scale);
    
    const padding = 10;
    visibleX = Math.max(0, visibleX - padding);
    visibleY = Math.max(0, visibleY - padding);
    visibleW = Math.min(DRAW_CONFIG.canvasW - visibleX, visibleW + padding * 2);
    visibleH = Math.min(DRAW_CONFIG.canvasH - visibleY, visibleH + padding * 2);
    
    cachedVisibleRect = {
        x: visibleX,
        y: visibleY,
        width: visibleW,
        height: visibleH
    };
    
    return cachedVisibleRect;
}

// 绑定所有事件
function main_setup_all_events() {
    main_setup_mode_events();
    main_setup_tool_events();
    main_setup_pen_control_events();
    main_setup_gesture_system();
    main_setup_settings_events();
    main_setup_click_outside();
    // 摄像头预览自适应帧率：订阅批注绘制耗时采样
    window.batchDrawManager?.set_perf_hook(main_handle_draw_perf_sample);
    if (window.blackboardManager) {
        window.blackboardManager.setup_toolbar_events();
    }
}

// ===== 基于手势模块的统一输入系统 =====

/** 基于 InputSource/DragTapSource/PinchZoomSource 的手势处理 */
function main_setup_gesture_system() {
    const input = new InputSource(dom.canvasWrapper);
    input.attach();
    window._gestureInput = input;

    // ------- 输入事件（用于绘制/手掌擦除） -------
    input.on('inputDown', async (ev) => {
        if (window.tileRenderer) window.tileRenderer.cancel_idle_shrink();
        main_cancel_smooth_transform();

        if (state.isPalmErasing) return;

        // 缩放中不处理任何状态切换（手掌检测/绘制），直到手势结束重置
        if (state.isScaling) return;

        state.drawCanvasRect = dom.canvasWrapper.getBoundingClientRect();
        state.currentPressure = ev.originEvent?.pressure || 0.5;

        // PointerEvent 路径：通过触点宽高检测手掌（单个触点面积大 → 手掌）
        if (window.PointerEvent && DRAW_CONFIG.palmEraserEnabled && ev.originEvent?.pointerType) {
            const palmResult = main_is_palm_pointer(ev.originEvent);
            if (palmResult.isPalm) {
                if (state.isDrawing) {
                    state.isDrawing = false;
                    main_hide_drawing_mode();
                }
                const size = window.__palmEraser.compute_palm_eraser_size_from_pointer(palmResult.width, palmResult.height);
                main_start_palm_erase(ev.position.x, ev.position.y, size);
                return;
            }
        }

        // 4+ 触点手掌检测（所有设备路径统一处理）
        // 在 PointerEvent 设备上，4 指同时触摸也可能误触；需要与宽度/高度检测互补
        if (DRAW_CONFIG.palmEraserEnabled && input.activeCount >= 4) {
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
                    // PointerEvent 路径：使用已注册的活跃触点位置
                    const positions = input.getActivePositions();
                    isPalm = palmEraser.is_palm_by_positions(positions);
                    if (isPalm) {
                        const c = palmEraser.get_palm_center_from_positions(positions);
                        centerX = c.x;
                        centerY = c.y;
                    }
                }

                if (isPalm) {
                    if (state.isDrawing) {
                        state.isDrawing = false;
                        main_hide_drawing_mode();
                    }
                    main_start_palm_erase(centerX, centerY, DRAW_CONFIG.palmEraserSize);
                    return;
                }
            }
        }

        // 绘制模式（comment / eraser）
        if (state.drawMode !== 'move' && !state.isScaling) {
            main_hide_pen_control_panel();
            state.isDrawing = true;
            main_start_drawing_mode();
            state.cachedInvScale = 1 / main_fetch_safe_scale();
            state.lastX = (ev.position.x - state.drawCanvasRect.left) * state.cachedInvScale;
            state.lastY = (ev.position.y - state.drawCanvasRect.top) * state.cachedInvScale;
            main_start_stroke(state.drawMode === 'eraser' ? 'erase' : 'draw');
            if (state.drawMode === 'eraser') {
                main_show_eraser_hint();
                main_update_eraser_hint_position(ev.position.x, ev.position.y);
            }
        }
    });

    input.on('inputMove', (ev) => {
        if (state._pinchResidualDrag) {
            if (state._pinchResidualDragFingerId !== null && state._pinchResidualDragFingerId !== ev.id) return;
            state.canvasX = ev.position.x - state.startDragX;
            state.canvasY = ev.position.y - state.startDragY;
            main_update_canvas_position();
            main_update_gesture_velocity(false);
            main_update_transform_schedule(state.canvasX, state.canvasY, state.scale);
            return;
        }
        if (state.isPalmErasing) {
            main_update_palm_erase(ev.position.x, ev.position.y);
            return;
        }

        state.currentPressure = ev.originEvent?.pressure || 0.5;

        if (state.drawMode === 'eraser' && state.isDrawing) {
            main_update_eraser_hint_position(ev.position.x, ev.position.y);
        }

        if (state.isDrawing && state.drawMode !== 'move') {
            const rect = state.drawCanvasRect;
            if (!rect) return;
            const invScale = state.cachedInvScale;
            const x = (ev.position.x - rect.left) * invScale;
            const y = (ev.position.y - rect.top) * invScale;

            const dx = x - state.lastX;
            const dy = y - state.lastY;
            if (dx * dx + dy * dy > 1) {
                main_save_stroke_point(state.lastX, state.lastY, x, y);
                window.batchDrawManager.batch_draw_create_command(
                    state.cachedDrawType,
                    state.lastX, state.lastY,
                    x, y,
                    state.cachedDrawColor,
                    state.cachedDrawLineWidth
                );
                state.lastX = x;
                state.lastY = y;
            }
        }
    });

    input.on('inputUp', async (ev) => {
        if (state._pinchResidualDrag) {
            if (state._pinchResidualDragFingerId !== null && state._pinchResidualDragFingerId !== ev.id) return;
            state._pinchResidualDrag = false;
            state._pinchResidualDragFingerId = null;
            dom.canvasWrapper.classList.remove('dragging');
            if (state.drawMode === 'move' && (Math.abs(state._gestureVx) > 2 || Math.abs(state._gestureVy) > 2)) {
                main_update_move_bound();
                main_update_canvas_position();
                main_start_momentum('xy');
            } else {
                main_update_canvas_transform_smooth(state.canvasX, state.canvasY, state.scale, 200);
            }
            return;
        }
        if (state.isPalmErasing) {
            if (input.activeCount < 4) {
                await main_end_palm_erase();
            }
            return;
        }
        if (state.isDrawing) {
            main_flush_last_segment(ev.position.x, ev.position.y);
            main_hide_drawing_mode();
            await main_submit_stroke();
            if (state.drawMode === 'eraser') main_hide_eraser_hint();
        }
        if (!state.isScaling) {
            dom.canvasWrapper.classList.remove('dragging');
        }
    });

    // ------- 拖拽手势（用于 move 模式平移） -------
    const drag = new DragTapSource(input, { toleranceSet: TOLERANCE.DRAG });
    window._gestureDrag = drag;

    drag.onDragStarted = () => {
        if (state.isPalmErasing || state.isDrawing || state.isScaling) return;
        // 两指及以上交给 pinch 处理，单指拖拽不介入（防止与缩放锚点冲突）
        if (input.activeCount >= 2) return;
        state.isDragging = true;
        const lastEv = input.activeEvents[0];
        state.startDragX = (lastEv?.position.x || 0) - state.canvasX;
        state.startDragY = (lastEv?.position.y || 0) - state.canvasY;
        state._lastCanvasX = state.canvasX;
        state._lastCanvasY = state.canvasY;
        state._gestureVx = 0;
        state._gestureVy = 0;
        dom.canvasWrapper.classList.add('dragging');
    };

    drag.onDragDelta = (ev) => {
        if (!state.isDragging || state.isPalmErasing || state.isScaling) return;
        // 两指手势期间 pinch 全权负责位置，拖拽增量一律忽略
        if (input.activeCount >= 2) return;

        state.canvasX = ev.position.x - state.startDragX;
        state.canvasY = ev.position.y - state.startDragY;
        main_update_canvas_position();
        main_update_gesture_velocity(false);
        main_update_transform_schedule(state.canvasX, state.canvasY, state.scale);
    };

    drag.onDragCompleted = () => {
        if (!state.isDragging) return;
        state.isDragging = false;
        dom.canvasWrapper.classList.remove('dragging');

        if (state.drawMode === 'move' && (Math.abs(state._gestureVx) > 2 || Math.abs(state._gestureVy) > 2)) {
            main_update_move_bound();
            main_update_canvas_position();
            main_start_momentum('xy');
        } else {
            main_update_canvas_transform_smooth(state.canvasX, state.canvasY, state.scale, 200);
        }
    };

    // ------- 两指捏合（缩放 + 平移） -------
    const useV2 = DRAW_CONFIG.pinchZoomV2 === true;
    const pinch = useV2 ? new PinchZoomSourceV2(input) : new PinchZoomSource(input);
    window._gesturePinch = pinch;

    pinch.onPinchStarted = (ev) => {
        main_cancel_smooth_transform();
        main_cancel_pending_transform();

        // 清除上一轮缩放/拖拽残留状态，防止与新 pinch 冲突
        state._pinchResidualDrag = false;
        state._pinchResidualDragFingerId = null;
        // 终止单指拖拽跟踪：若 pinch 前首指正在拖拽，其过期的抓取偏移
        // 会在 pinch 结束（isScaling=false）后覆写画布位置导致跳变
        state.isDragging = false;
        drag.cancelDrag();

        if (state.isDrawing) {
            state.isDrawing = false;
            main_hide_drawing_mode();
            // flush 最后一段并提交到历史，保留已有批注内容不丢失
            if (ev.finger0) {
                main_flush_last_segment(ev.finger0.x, ev.finger0.y);
            }
            main_submit_stroke();
        }

        state.isScaling = true;
        state.startScale = state.scale;

        // 缩放边界阻尼器：消除贴墙时的触控噪声"呼吸"抖动
        state._zoomDamper ??= new ZoomWallDamper();
        state._zoomDamper.reset(state.scale, DRAW_CONFIG.minScale, DRAW_CONFIG.maxScaleImage);

        if (useV2) {
            // V2: 以两指中点为缩放锚点
            state.startMidCX = (ev.centerX - state.canvasX) / state.scale;
            state.startMidCY = (ev.centerY - state.canvasY) / state.scale;
        } else {
            // V1: 使用 PinchZoomSource 传入的 finger0，而非 getActivePositions()[0]，
            // 确保锚点计算与追踪手指完全一致
            if (ev.finger0) {
                state.startFinger0CX = (ev.finger0.x - state.canvasX) / state.scale;
                state.startFinger0CY = (ev.finger0.y - state.canvasY) / state.scale;
            }
        }
        state.startCanvasX = state.canvasX;
        state.startCanvasY = state.canvasY;
        dom.canvasWrapper.classList.add('dragging');
        dom.canvasWrapper.style.willChange = 'transform';
        state._lastCanvasX = state.canvasX;
        state._lastCanvasY = state.canvasY;
        state._gestureVx = 0;
        state._gestureVy = 0;
    };

    pinch.onPinchDelta = (ev) => {
        if (!state.isScaling) return;
        const maxScale = DRAW_CONFIG.maxScaleImage;

        if (useV2) {
            // V2: 增量式缩放 + 中点锚点 + 边界阻尼（贴墙吸收噪声，累计越界 2% 才脱离）
            state.scale = state._zoomDamper.update(ev.scale);
            // 中点锚点：保持画布上初始中点位置跟随当前中点
            state.canvasX = ev.centerX - state.startMidCX * state.scale;
            state.canvasY = ev.centerY - state.startMidCY * state.scale;
        } else {
            // V1: 累积式缩放 + finger0 锚点
            const unclampedScale = state.startScale * ev.scale;
            state.scale = Math.max(DRAW_CONFIG.minScale, Math.min(maxScale, unclampedScale));

            if (state.scale !== unclampedScale) {
                // 缩放到达边界时同步重置 PinchZoomSource 内部参考距离，
                // 使后续 ev.scale 相对于当前手指距离而非 pinch 起始距离，
                // 消除边界处缩放死区（缩放回退时立即响应）
                const fdx = ev.finger0.x - ev.finger1.x;
                const fdy = ev.finger0.y - ev.finger1.y;
                pinch.resetScaleReference(Math.sqrt(fdx * fdx + fdy * fdy));
                state.startFinger0CX = (ev.finger0.x - state.canvasX) / state.scale;
                state.startFinger0CY = (ev.finger0.y - state.canvasY) / state.scale;
                state.startScale = state.scale;
            }

            state.canvasX = ev.finger0.x - state.startFinger0CX * state.scale;
            state.canvasY = ev.finger0.y - state.startFinger0CY * state.scale;
        }

        main_update_move_bound();
        main_update_canvas_position();

        main_update_gesture_velocity(true);

        // 脏检查 + rAF 节流（与拖拽路径对齐，避免每帧直接写 DOM）
        if (last_canvas_transform.x !== state.canvasX ||
            last_canvas_transform.y !== state.canvasY ||
            last_canvas_transform.scale !== state.scale) {
            main_update_transform_schedule(state.canvasX, state.canvasY, state.scale);
        }

        main_set_zooming();
    };

    pinch.onPinchCompleted = () => {
        state.isScaling = false;
        dom.canvasWrapper.style.willChange = '';
        dom.canvasWrapper.classList.remove('dragging');
        main_cancel_zoom_debounce();

        // 缩放结束后：仍有手指在屏幕上的，进入残余拖拽模式，
        // 记录当前手指 ID，后续只接受该手指的 inputMove/inputUp
        if (input.activeCount >= 1 && state.drawMode === 'move') {
            const ev = input.activeEvents[0];
            if (ev) {
                state.isDragging = false;
                drag.cancelDrag();
                // 残余拖拽开始前清掉 CSS transition，避免后续逐帧直写 transform 变成补间
                main_cancel_smooth_transform();
                state._pinchResidualDrag = true;
                state._pinchResidualDragFingerId = ev.id;
                state.startDragX = ev.position.x - state.canvasX;
                state.startDragY = ev.position.y - state.canvasY;
                state._lastCanvasX = state.canvasX;
                state._lastCanvasY = state.canvasY;
                state._gestureVx = 0;
                state._gestureVy = 0;
                dom.canvasWrapper.classList.add('dragging');
            }
        } else if (input.activeCount === 0) {
            main_update_move_bound();
            main_update_canvas_position();

            if (Math.abs(state._gestureVx) > 2 || Math.abs(state._gestureVy) > 2) {
                main_start_momentum('xy');
            } else {
                main_update_canvas_transform_smooth(state.canvasX, state.canvasY, state.scale, 200);
            }
        }
    };

    // ------- 鼠标滚轮缩放 -------
    dom.canvasWrapper.addEventListener('wheel', main_handle_wheel, { passive: true });
}

// 设置面板事件
function main_setup_settings_events() {
    document.getElementById('btnRotateLeft')?.addEventListener('click', () => {
        main_update_image_rotation('left');
    });
    
    document.getElementById('btnRotateRight')?.addEventListener('click', () => {
        main_update_image_rotation('right');
    });

    // 亮度 / 对比度 / 黑白 控件
    const brightnessEl = document.getElementById('cameraBrightness');
    const brightnessVal = document.getElementById('cameraBrightnessValue');
    const contrastEl = document.getElementById('cameraContrast');
    const contrastVal = document.getElementById('cameraContrastValue');
    const grayscaleGroup = document.getElementById('cameraGrayscaleGroup');

    // brightness / contrast / grayscale input: only apply to current session (do not persist)
    brightnessEl?.addEventListener('input', (e) => {
        const v = parseInt(e.target.value, 10);
        state.camera_brightness = v;
        if (brightnessVal) brightnessVal.textContent = String(v);
        main_apply_camera_filters();
    });

    contrastEl?.addEventListener('input', (e) => {
        const v = parseInt(e.target.value, 10) / 100.0;
        state.camera_contrast = v;
        if (contrastVal) contrastVal.textContent = v.toFixed(2);
        main_apply_camera_filters();
    });

    grayscaleGroup?.addEventListener('click', (e) => {
        const btn = e.target.closest('.option-btn');
        if (!btn) return;
        const value = btn.dataset.value;
        grayscaleGroup.querySelectorAll('.option-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        grayscaleGroup.dataset.active = value;
        state.camera_grayscale = value === 'on' ? 1 : 0;
        main_apply_camera_filters();
    });

    // reset sliders
    document.getElementById('btnResetSliders')?.addEventListener('click', () => {
        const brightnessInput = document.getElementById('cameraBrightness');
        const brightnessVal = document.getElementById('cameraBrightnessValue');
        const contrastInput = document.getElementById('cameraContrast');
        const contrastVal = document.getElementById('cameraContrastValue');
        const grayscaleGroup = document.getElementById('cameraGrayscaleGroup');

        if (brightnessInput) {
            brightnessInput.value = '10';
            state.camera_brightness = 10;
            if (brightnessVal) brightnessVal.textContent = '10';
        }
        if (contrastInput) {
            contrastInput.value = '140';
            state.camera_contrast = 1.4;
            if (contrastVal) contrastVal.textContent = '1.40';
        }
        if (grayscaleGroup) {
            grayscaleGroup.querySelectorAll('.option-btn').forEach(b => b.classList.remove('active'));
            const offBtn = grayscaleGroup.querySelector('.option-btn[data-value="off"]');
            if (offBtn) {
                offBtn.classList.add('active');
                grayscaleGroup.dataset.active = 'off';
            }
            state.camera_grayscale = 0;
        }
        main_apply_camera_filters();
    });
}

// 点击外部关闭面板
function main_setup_click_outside() {
    document.addEventListener('click', (e) => {
        const panel = dom.penControlPanel;
        const isClickInsidePanel = panel.contains(e.target);
        const isClickOnBtnComment = dom.btnComment.contains(e.target);
        const isClickOnBtnEraser = dom.btnEraser.contains(e.target);
        // 阅读器模式也有自己的笔/橡皮按钮
        const drCommentBtn = document.getElementById('drBtnComment');
        const drEraserBtn = document.getElementById('drBtnEraser');
        const isClickOnDrBtnComment = drCommentBtn?.contains(e.target);
        const isClickOnDrBtnEraser = drEraserBtn?.contains(e.target);
        // 小黑板自己的笔/橡皮按钮
        const bbCommentBtn = document.getElementById('bbBtnComment');
        const bbEraserBtn = document.getElementById('bbBtnEraser');
        const isClickOnBbBtnComment = bbCommentBtn?.contains(e.target);
        const isClickOnBbBtnEraser = bbEraserBtn?.contains(e.target);
        
        if (!isClickInsidePanel && !isClickOnBtnComment && !isClickOnBtnEraser
            && !isClickOnDrBtnComment && !isClickOnDrBtnEraser
            && !isClickOnBbBtnComment && !isClickOnBbBtnEraser) {
            main_hide_pen_control_panel();
        }
        
        const settingsPanel = dom.settingsPanel;
        const isClickInsideSettings = settingsPanel.contains(e.target);
        const isClickOnBtnSettings = dom.btnSettings.contains(e.target);
        
        if (!isClickInsideSettings && !isClickOnBtnSettings) {
            main_hide_settings_panel();
        }
    });
}

// 模式切换事件
function main_setup_mode_events() {
    dom.btnMove.addEventListener('click', () => {
        main_update_mode('move');
    });
    dom.btnComment.addEventListener('click', () => {
        if (state.drawMode === 'comment') {
            main_show_pen_control_panel(dom.btnComment, 'comment');
        } else {
            main_update_mode('comment');
        }
    });
    dom.btnEraser.addEventListener('click', () => {
        if (state.drawMode === 'eraser' && !DRAW_CONFIG.eraserSpeedEnabled) {
            main_show_pen_control_panel(dom.btnEraser, 'eraser');
        } else {
            main_update_mode('eraser');
        }
    });
}

// 切换模式
async function main_update_mode(mode) {
    const bb = window.blackboardManager;
    if (bb?.is_open) {
        // 通过 DrawingEngine 提交未完成笔画
        if (bb.drawing_engine?.is_drawing) {
            bb.drawing_engine.is_drawing = false;
            if (bb.drawing_engine.current_stroke) {
                await bb.drawing_engine._submit_stroke();
            }
        }
        bb.draw_mode = mode;
        bb.drawing_engine?.set_draw_mode(mode);

        [dom.btnMove, dom.btnComment, dom.btnEraser].forEach(btn => {
            btn.classList.remove('primary-btn');
        });

        switch (mode) {
            case 'move':
                if (dom.btnMove) dom.btnMove.classList.add('primary-btn');
                if (bb.bb_wrapper) bb.bb_wrapper.style.cursor = 'grab';
                bb.drawing_engine?._hide_eraser_hint();
                main_update_camera_frame_rate(null);
                break;
            case 'comment':
                if (dom.btnComment) dom.btnComment.classList.add('primary-btn');
                if (bb.bb_wrapper) bb.bb_wrapper.style.cursor = 'crosshair';
                bb.drawing_engine?._hide_eraser_hint();
                main_update_pen_style();
                main_enter_annotate_camera_fps();
                break;
            case 'eraser':
                if (dom.btnEraser) dom.btnEraser.classList.add('primary-btn');
                bb.drawing_engine?.set_draw_mode(mode);
                main_update_eraser_style();
                main_enter_annotate_camera_fps();
                break;
        }

        console.log(`[黑板] 切换到 ${mode} 模式`);
        return;
    }

    // 切换模式前提交当前未完成的笔画并重置绘制状态
    if (state.isDrawing) {
        state.isDrawing = false;
        main_hide_drawing_mode();
        await main_submit_stroke();
        batchDrawManager.batch_draw_delete_all();
    }
    state.isDragging = false;
    state.isScaling = false;
    main_cancel_zoom_debounce();
    
    main_hide_pen_control_panel();
    
    [dom.btnMove, dom.btnComment, dom.btnEraser].forEach(btn => {
        btn.classList.remove('primary-btn');
    });
    
    dom.canvasWrapper.classList.remove('drawing', 'dragging');
    
    state.drawMode = mode;
    
    // 批注模式下延迟缩放启动，防止绘画中误触
    if (window._gesturePinch) {
        window._gesturePinch.startDelayMs = (mode !== 'move') ? 200 : 0;
    }
    
    switch (mode) {
        case 'move':
            dom.btnMove.classList.add('primary-btn');
            dom.canvasWrapper.style.cursor = 'grab';
            main_hide_eraser_hint();
            main_update_camera_frame_rate(null);
            break;
        case 'comment':
            dom.btnComment.classList.add('primary-btn');
            dom.canvasWrapper.classList.add('drawing');
            dom.canvasWrapper.style.cursor = 'crosshair';
            main_hide_eraser_hint();
            main_update_pen_style();
            main_enter_annotate_camera_fps();
            break;
        case 'eraser':
            dom.btnEraser.classList.add('primary-btn');
            main_update_eraser_style();
            main_enter_annotate_camera_fps();
            break;
    }
    
    console.log(`切换到 ${mode} 模式`);
}

// 工具按钮事件
function main_setup_tool_events() {
    dom.btnUndo.addEventListener('click', () => {
        if (window.blackboardManager?.is_open) {
            window.blackboardManager.handle_undo();
        } else {
            main_handle_undo();
        }
    });
    dom.btnPhoto.addEventListener('click', main_save_photo);
    dom.btnLight.addEventListener('click', main_handle_light_toggle);
    // 展台灯状态同步（来自设置页或其它窗口的开关操作）
    if (window.__TAURI__) {
        try {
            window.__TAURI__.event.listen('camera-light-changed', (event) => {
                const isOn = event.payload?.isOn;
                if (isOn === undefined) return;
                window.__lightOn = isOn;
                window.__savedLightState = isOn;
                const img = dom.btnLight?.querySelector('img');
                if (isOn) {
                    dom.btnLight?.classList.add('active');
                    if (img) img.src = ThemeManager.theme_fetch_icon_path('light');
                } else {
                    dom.btnLight?.classList.remove('active');
                    if (img) img.src = ThemeManager.theme_fetch_icon_path('light-off');
                }
            });
        } catch (_) {}
    }
    dom.btnSettings.addEventListener('click', main_show_settings);
    dom.btnSave.addEventListener('click', main_handle_file_sidebar_toggle);
    dom.btnMinimize.addEventListener('click', main_hide_window);
    dom.btnMenu.addEventListener('click', main_handle_menu_toggle);
    dom.btnExpand.addEventListener('click', main_handle_sidebar_toggle);
    dom.btnBlackboard.addEventListener('click', async () => {
        // 如果阅读器已打开，先关闭阅读器再打开小黑板
        if (window.documentReaderManager?.is_open) {
            await window.documentReaderManager.close();
        }
        const bb = await window.blackboard_ensure_loaded(dom.canvasContainer);
        if (bb) {
            if (bb.is_open) {
                bb.close();
            } else {
                bb.open();
            }
        }
    });
}

// 菜单弹出
function main_handle_menu_toggle() {
    const existingMenu = document.getElementById('menuPopup');
    if (existingMenu) {
        main_hide_menu();
    } else {
        main_show_menu();
    }
}

function main_show_menu() {
    const menuPopup = document.createElement('div');
    menuPopup.id = 'menuPopup';
    menuPopup.className = 'menu-popup';
    menuPopup.innerHTML = `
        <button class="menu-item" id="menuSettings">
            ${ThemeManager.theme_fetch_icon('settings', { alt: window.i18n?.format_translate('toolbar.settings') || '设置' })}
            ${window.i18n?.format_translate('toolbar.settings') || '设置'}
        </button>
        <button class="menu-item" id="menuClose">
            ${ThemeManager.theme_fetch_icon('close', { alt: window.i18n?.format_translate('common.close') || '关闭' })}
            ${window.i18n?.format_translate('common.close') || '关闭'}
        </button>
    `;
    
    dom.canvasContainer.appendChild(menuPopup);
    
    document.getElementById('menuSettings').addEventListener('click', () => {
        main_hide_menu();
        main_show_settings_window();
    });
    
    document.getElementById('menuClose').addEventListener('click', () => {
        main_hide_menu();
        main_submit_close_window();
    });
    
    setTimeout(() => {
        document.addEventListener('click', main_handle_menu_outside_click);
    }, 0);
}

function main_hide_menu() {
    const menuPopup = document.getElementById('menuPopup');
    if (menuPopup) {
        menuPopup.remove();
    }
    document.removeEventListener('click', main_handle_menu_outside_click);
}

function main_handle_menu_outside_click(e) {
    const menuPopup = document.getElementById('menuPopup');
    const btnMenu = dom.btnMenu;
    
    if (menuPopup && !menuPopup.contains(e.target) && !btnMenu.contains(e.target)) {
        main_hide_menu();
    }
}

// 展台灯开关
// 状态统一存放在 window 上：camera.js / document_reader.js 通过
// window.__savedLightState / window.__lightOn 读写同一份状态，
// 模块级变量会与之割裂导致重开摄像头/关闭阅读器后灯状态无法恢复
window.__lightOn = window.__lightOn === true;
window.__savedLightState = window.__savedLightState === true;
let __lightToggleBusy = false;

async function main_handle_light_toggle() {
    if (!window.__TAURI__ || !window.__lightAvailable) return;
    // in-flight 防抖：上次 HID 写入未完成前忽略重复点击，防止并发 invoke 状态翻转
    if (__lightToggleBusy) return;
    __lightToggleBusy = true;
    try {
        const { invoke } = window.__TAURI__.core;
        const img = dom.btnLight?.querySelector('img');
        if (window.__lightOn) {
            await invoke('camera_light_off');
            window.__lightOn = false;
            dom.btnLight?.classList.remove('active');
            if (img) img.src = ThemeManager.theme_fetch_icon_path('light-off');
        } else {
            await invoke('camera_light_on');
            window.__lightOn = true;
            dom.btnLight?.classList.add('active');
            if (img) img.src = ThemeManager.theme_fetch_icon_path('light');
        }
        window.__savedLightState = window.__lightOn;
    } catch (e) {
        console.error('展台灯控制失败:', e);
    } finally {
        __lightToggleBusy = false;
    }
}

async function main_light_off() {
    if (!window.__TAURI__ || !window.__lightAvailable || !window.__lightOn) return;
    try {
        await window.__TAURI__.core.invoke('camera_light_off');
        window.__lightOn = false;
        dom.btnLight?.classList.remove('active');
        const img = dom.btnLight?.querySelector('img');
        if (img) img.src = ThemeManager.theme_fetch_icon_path('light-off');
    } catch (e) {
        console.error('展台灯关闭失败:', e);
    }
}

// 最小化窗口
async function main_hide_window() {
    if (window.__TAURI__?.window?.getCurrentWindow) {
        const appWindow = window.__TAURI__.window.getCurrentWindow();
        
        // 如果摄像头开启，先关闭摄像头
        if (state.isCameraOpen) {
            await main_update_camera_state(false);
            state.wasCameraOpenBeforeMinimize = true;
            console.log('摄像头已关闭（最小化）');
        }
        
        await appWindow.minimize();
        console.log('窗口已最小化');
    } else {
        console.log('Tauri API 不可用');
    }
}

// 监听窗口最小化和恢复事件
function main_setup_minimize_listeners() {
    if (window.__TAURI__) {
        const { getCurrentWindow } = window.__TAURI__.window;
        
        let isRestoring = false;
        
        const main_handle_restore = async () => {
            if (isRestoring) return;
            isRestoring = true;
            try {
                await main_init_camera_if_needed();
            } finally {
                setTimeout(() => {
                    isRestoring = false;
                }, 300);
            }
        };
        
        getCurrentWindow().listen('tauri://restore', main_handle_restore);
        getCurrentWindow().listen('tauri://show', main_handle_restore);
        getCurrentWindow().listen('tauri://focus', main_handle_restore);
    }
}

// 恢复摄像头（如果需要）
async function main_init_camera_if_needed() {
    // 如果之前摄像头是开启的，重新开启摄像头
    if (state.wasCameraOpenBeforeMinimize && !state.isCameraOpen) {
        try {
            await main_update_camera_state(true);
            console.log('摄像头已重新开启');
            // 只有在成功开启后才重置状态
            state.wasCameraOpenBeforeMinimize = false;
        } catch (error) {
            console.error('重新开启摄像头失败:', error);
            // 开启失败时保持状态，以便下次尝试
        }
    }
}

// 关闭窗口
async function main_submit_close_window() {
    if (window.__TAURI__?.window?.getCurrentWindow) {
        await window.documentReaderManager?.delete_annotation_cache_files?.();
        const appWindow = window.__TAURI__.window.getCurrentWindow();
        await appWindow.close();
        console.log('窗口已关闭');
    } else {
        console.log('Tauri API 不可用');
    }
}

// 动态构建画笔预设按钮
function main_build_pen_presets(presets) {
    const container = dom.penSizePresets;
    container.querySelectorAll('.size-preset-btn').forEach(b => b.remove());
    const valueSpan = container.querySelector('.pen-size-label');
    presets.forEach(value => {
        const btn = document.createElement('button');
        btn.className = 'size-preset-btn';
        btn.dataset.value = value;
        btn.style.setProperty('--dot-size', Math.round(value + 4) + 'px');
        btn.addEventListener('click', () => {
            DRAW_CONFIG.penWidth = value;
            container.querySelectorAll('.size-preset-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            dom.penSizeValue.textContent = `${value}px`;
            if (state.drawMode === 'comment') {
                main_update_pen_style();
            }
        });
        container.insertBefore(btn, valueSpan);
    });
    const active = container.querySelector(`[data-value="${DRAW_CONFIG.penWidth}"]`);
    if (active) active.classList.add('active');
    main_update_pen_preset_dot_color();
    dom.penSizeValue.textContent = `${DRAW_CONFIG.penWidth}px`;
}

// 动态构建橡皮擦预设按钮
function main_build_eraser_presets(presets) {
    const container = dom.eraserSizePresets;
    container.querySelectorAll('.size-preset-btn').forEach(b => b.remove());
    const valueSpan = container.querySelector('.pen-size-label');
    presets.forEach(value => {
        const btn = document.createElement('button');
        btn.className = 'size-preset-btn';
        btn.dataset.value = value;
        const maxVal = Math.max(...presets);
        const minVal = Math.min(...presets);
        const range = maxVal - minVal || 1;
        const dot = 4 + (value - minVal) / range * 24;
        btn.style.setProperty('--dot-size', Math.round(dot) + 'px');
        btn.addEventListener('click', () => {
            DRAW_CONFIG.eraserSize = value;
            container.querySelectorAll('.size-preset-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            dom.eraserSizeValue.textContent = `${value}px`;
            main_update_eraser_hint_size();
            if (window.blackboardManager?.drawing_engine) {
                window.blackboardManager.drawing_engine.refresh_eraser_hint_size();
            }
            if (state.drawMode === 'eraser') {
                main_update_eraser_style();
            }
        });
        container.insertBefore(btn, valueSpan);
    });
    const active = container.querySelector(`[data-value="${DRAW_CONFIG.eraserSize}"]`);
    if (active) active.classList.add('active');
    dom.eraserSizeValue.textContent = `${DRAW_CONFIG.eraserSize}px`;
}

// 笔触控制事件
function main_setup_pen_control_events() {
    main_build_pen_presets(DRAW_CONFIG.penSizePresets);
    main_build_eraser_presets(DRAW_CONFIG.eraserSizePresets);
    
    // 颜色按钮点击事件
    const colorButtons = document.querySelectorAll('.pen-color-btn');
    colorButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const index = parseInt(btn.dataset.index);
            const color = DRAW_CONFIG.penColors[index];
            if (color) {
                DRAW_CONFIG.penColor = color;
                main_update_pen_preset_dot_color();
                
                // 更新选中状态
                colorButtons.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                
                if (state.drawMode === 'comment') {
                    main_update_pen_style();
                }
            }
        });
    });
    
    // 橡皮清空滑块 — 滑到最右触发清空
    const clearSlider = document.getElementById('eraserClearSlider');
    const hintText = document.getElementById('eraserHintText');
    if (clearSlider) {
        const onInput = () => {
            const pct = clearSlider.value;
            clearSlider.style.setProperty('--fill', pct + '%');
            if (hintText) hintText.classList.toggle('hidden', pct !== '0');
            if (pct === '100') {
                if (window.blackboardManager?.is_open) {
                    window.blackboardManager.handle_clear();
                } else if (window.documentReaderManager?.is_open) {
                    window.documentReaderManager.handle_clear();
                } else {
                    main_delete_all_drawings();
                }
                clearSlider.value = '0';
                clearSlider.style.setProperty('--fill', '0%');
            }
        };
        clearSlider.addEventListener('input', onInput);
        // 离手后自动退回
        clearSlider.addEventListener('pointerup', () => {
            if (clearSlider.value === '100') return;
            clearSlider.value = '0';
            clearSlider.style.setProperty('--fill', '0%');
            if (hintText) hintText.classList.remove('hidden');
        });
        // 初始化
        clearSlider.style.setProperty('--fill', '0%');
    }

    // 初始化颜色按钮
    main_update_color_buttons();
}

// RGB转十六进制颜色
function main_calc_rgb_to_hex(r, g, b) {
    return '#' + [r, g, b].map(x => {
        const hex = x.toString(16);
        return hex.length === 1 ? '0' + hex : hex;
    }).join('');
}

function main_update_color_buttons() {
    const colorButtons = document.querySelectorAll('.pen-color-btn');
    colorButtons.forEach((btn, index) => {
        if (DRAW_CONFIG.penColors[index]) {
            btn.dataset.color = DRAW_CONFIG.penColors[index];
            btn.style.backgroundColor = DRAW_CONFIG.penColors[index];
            btn.title = window.i18n?.format_translate('settings.colorN', { n: index + 1 }) || `颜色${index + 1}`;
            
            if (DRAW_CONFIG.penColors[index].toLowerCase() === '#000000') {
                btn.classList.add('dark-color');
            } else {
                btn.classList.remove('dark-color');
            }
            
            if (DRAW_CONFIG.penColors[index].toLowerCase() === '#ffffff') {
                btn.classList.add('light-color');
            } else {
                btn.classList.remove('light-color');
            }
        }
    });
    main_update_color_button_active();
}

function main_update_color_button_active() {
    const colorButtons = document.querySelectorAll('.pen-color-btn');
    colorButtons.forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.color === DRAW_CONFIG.penColor) {
            btn.classList.add('active');
        }
    });
    main_update_pen_preset_dot_color();
}

function main_update_pen_preset_dot_color() {
    dom.penSizePresets.querySelectorAll('.size-preset-btn').forEach(btn => {
        btn.style.setProperty('--dot-color', DRAW_CONFIG.penColor);
    });
}

// 设置笔触样式
function main_update_pen_style() {
    main_reset_context_state();
}

function main_update_eraser_style() {
    main_reset_context_state();
}

function main_start_drawing_mode() {
    dom.canvasWrapper.classList.add('drawing');
}

function main_hide_drawing_mode() {
    dom.canvasWrapper.classList.remove('drawing');
}

// 橡皮提示框 — 固定屏幕像素尺寸，不随缩放变化
function main_update_eraser_hint_size() {
    dom.eraserHint.style.width = `${DRAW_CONFIG.eraserSize}px`;
    dom.eraserHint.style.height = `${DRAW_CONFIG.eraserSize}px`;
}

function main_show_eraser_hint() {
    dom.eraserHint.classList.add('active');
}

function main_hide_eraser_hint() {
    dom.eraserHint.classList.remove('active');
    if (eraserHintRafId !== null) {
        cancelAnimationFrame(eraserHintRafId);
        eraserHintRafId = null;
    }
    eraserHintPendingPos = null;
}

function main_show_pen_control_panel(targetBtn, mode) {
    const panel = dom.penControlPanel;
    if (!panel) return;
    
    const btnRect = targetBtn.getBoundingClientRect();
    const containerRect = document.querySelector('.main-function').getBoundingClientRect();
    
    const penSizeControl = panel.querySelector('.pen-size-presets:nth-child(1)');
    const colorButtons = panel.querySelector('.pen-color-buttons');
    const eraserSizeControl = panel.querySelector('.pen-size-presets:nth-child(3)');
    
    if (mode === 'comment') {
        if (penSizeControl) penSizeControl.style.display = 'flex';
        if (colorButtons) colorButtons.style.display = 'grid';
        if (eraserSizeControl) eraserSizeControl.style.display = 'none';
    } else if (mode === 'eraser') {
        if (penSizeControl) penSizeControl.style.display = 'none';
        if (colorButtons) colorButtons.style.display = 'none';
        if (eraserSizeControl) eraserSizeControl.style.display = DRAW_CONFIG.eraserSpeedEnabled ? 'none' : 'flex';
    }
    
    // 重置面板布局 → 强制重排获取准确尺寸 → 计算并约束位置
    panel.style.position = 'absolute';
    panel.style.bottom = 'auto';
    panel.style.top = 'auto';
    panel.style.right = 'auto';
    panel.style.left = 'auto';
    panel.style.visibility = 'hidden';
    panel.style.opacity = '0';
    panel.classList.remove('visible');
    
    panel.offsetHeight;
    
    const panelWidth = panel.offsetWidth || (mode === 'comment' ? 380 : 240);
    const panelHeight = panel.offsetHeight || 120;
    
    let left = btnRect.left - containerRect.left + (btnRect.width / 2) - (panelWidth / 2);
    let top = btnRect.top - containerRect.top - panelHeight - 15;
    
    const containerPadding = 10;
    left = Math.max(containerPadding, Math.min(left, containerRect.width - panelWidth - containerPadding));
    
    // 面板顶部超出容器时改显示在按钮下方
    if (top < containerPadding) {
        top = btnRect.bottom - containerRect.top + 15;
    }
    
    // 设置最终位置并显示面板
    panel.style.left = `${left}px`;
    panel.style.top = `${top}px`;
    panel.style.visibility = 'visible';
    panel.style.opacity = '1';
    panel.classList.add('visible');
}

function main_hide_pen_control_panel() {
    const panel = dom.penControlPanel;
    if (!panel.classList.contains('visible')) return;
    panel.classList.remove('visible');
    panel.style.opacity = '0';
    panel.style.visibility = 'hidden';
}

let eraserHintRafId = null;
let eraserHintPendingPos = null;

function main_update_eraser_hint_position(clientX, clientY) {
    eraserHintPendingPos = { clientX, clientY };

    if (eraserHintRafId !== null) return;

    eraserHintRafId = requestAnimationFrame(() => {
        eraserHintRafId = null;
        if (!eraserHintPendingPos) return;

        const { clientX, clientY } = eraserHintPendingPos;
        eraserHintPendingPos = null;

        const rect = main_fetch_cached_canvas_rect();
        const x = clientX - rect.left;
        const y = clientY - rect.top;
        dom.eraserHint.style.left = `${x}px`;
        dom.eraserHint.style.top = `${y}px`;
        dom.eraserHint.style.transform = 'translate(-50%, -50%)';

        if (state.drawMode === 'eraser' && state.isDrawing) {
            const scale = state.scale || 1;
            const hintSize = (state.cachedDrawLineWidth || DRAW_CONFIG.eraserSize) * scale;
            dom.eraserHint.style.width = `${hintSize}px`;
            dom.eraserHint.style.height = `${hintSize}px`;
        }
    });
}

// ====== 手掌擦除 ======

function main_is_palm_pointer(e) {
    const mod = window.__palmEraser;
    if (!mod) return { isPalm: false, width: 0, height: 0 };
    return mod.is_palm_by_pointer(e);
}

function main_show_palm_eraser_hint() {
    if (!dom.palmEraserHint) return;
    dom.palmEraserHint.classList.add('active');
}

function main_hide_palm_eraser_hint() {
    if (!dom.palmEraserHint) return;
    dom.palmEraserHint.classList.remove('active');
}

function main_update_palm_eraser_hint(clientX, clientY, size) {
    if (!dom.palmEraserHint) return;
    const rect = main_fetch_cached_canvas_rect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    const visualSize = size * main_fetch_safe_scale();
    dom.palmEraserHint.style.width = `${visualSize}px`;
    dom.palmEraserHint.style.height = `${visualSize}px`;
    dom.palmEraserHint.style.left = `${x}px`;
    dom.palmEraserHint.style.top = `${y}px`;
    dom.palmEraserHint.style.transform = 'translate(-50%, -50%)';
}

let _palmSession = null;
function main_get_palm_session() {
    if (!_palmSession && window.__palmEraser) {
        _palmSession = new window.__palmEraser.PalmEraserSession({
            defaultEraserSize: DRAW_CONFIG.palmEraserSize,
            getCanvasRect: () => dom.canvasWrapper.getBoundingClientRect(),
            getScale: main_fetch_safe_scale,
            showHint: main_show_palm_eraser_hint,
            updateHint: main_update_palm_eraser_hint,
            hideHint: main_hide_palm_eraser_hint,
            saveStrokePoint: (fromX, fromY, toX, toY) => main_save_stroke_point(fromX, fromY, toX, toY),
            submitStroke: () => main_submit_stroke(),
            onSessionStart(stroke, session) {
                state.isPalmErasing = true;
                state.savedDrawMode = state.drawMode;
                state.drawMode = 'eraser';
                state.palmEraserSize = session.palmEraserSize;
                state.currentStroke = stroke;
                state.isDrawing = true;
                main_start_drawing_mode();
                state.cachedDrawType = 'erase';
                state.cachedDrawColor = '#000000';
                state.cachedDrawLineWidth = session.palmEraserSize / main_fetch_safe_scale();
                batchDrawManager.eraserShape = 'square';
                batchDrawManager.batch_draw_init_start();
            },
            createCommand: (fromX, fromY, toX, toY) => {
                batchDrawManager.batch_draw_create_command(
                    state.cachedDrawType, fromX, fromY, toX, toY,
                    state.cachedDrawColor, state.cachedDrawLineWidth
                );
            },
            onSessionEnd() {
                state.isPalmErasing = false;
                state.isDrawing = false;
                main_hide_drawing_mode();
                state.drawMode = state.savedDrawMode || 'move';
                state.savedDrawMode = null;
                state.currentStroke = null;
            }
        });
    }
    return _palmSession;
}

function main_start_palm_erase(clientX, clientY, eraserWidth) {
    const session = main_get_palm_session();
    if (session) session.start(clientX, clientY, eraserWidth);
}

function main_update_palm_erase(clientX, clientY) {
    if (_palmSession) _palmSession.update(clientX, clientY);
}

async function main_end_palm_erase() {
    if (_palmSession) await _palmSession.end();
}

function main_flush_last_segment(clientX, clientY) {
    if (!state.drawCanvasRect) return;
    const invScale = state.cachedInvScale;
    const x = (clientX - state.drawCanvasRect.left) * invScale;
    const y = (clientY - state.drawCanvasRect.top) * invScale;
    const dx = x - state.lastX;
    const dy = y - state.lastY;
    if (dx !== 0 || dy !== 0) {
        main_save_stroke_point(state.lastX, state.lastY, x, y);
        batchDrawManager.batch_draw_create_command(
            state.cachedDrawType,
            state.lastX,
            state.lastY,
            x,
            y,
            state.cachedDrawColor,
            state.cachedDrawLineWidth
        );
        state.lastX = x;
        state.lastY = y;
    }
}

function main_handle_wheel(e) {
    if (window.tileRenderer) window.tileRenderer.cancel_idle_shrink();
    // 步长随滚动增量自适应：鼠标滚轮每格 ≈0.1（与原固定步长一致），
    // 触控板的高频小增量事件获得细腻的渐进缩放而非跳变
    const step = Math.min(0.1, Math.max(0.01, Math.abs(e.deltaY) * 0.001));
    const delta = e.deltaY > 0 ? -step : step;
    const maxScale = DRAW_CONFIG.maxScaleImage;
    const newScale = Math.max(DRAW_CONFIG.minScale, Math.min(maxScale, state.scale + delta));
    
    if (newScale !== state.scale) {
        const containerRect = main_fetch_cached_canvas_rect();
        const mouseX = e.clientX - containerRect.left;
        const mouseY = e.clientY - containerRect.top;
        
        const scaleRatio = newScale / state.scale;
        const targetX = mouseX - (mouseX - state.canvasX) * scaleRatio;
        const targetY = mouseY - (mouseY - state.canvasY) * scaleRatio;
        
        state.scale = newScale;
        state.canvasX = targetX;
        state.canvasY = targetY;
        
        // 标记缩放进行中，抑制 main_update_transform_schedule 中逐帧 DPR 更新
        state.isZooming = true;
        if (zoom_complete_timer_id !== null) {
            clearTimeout(zoom_complete_timer_id);
        }
        
        // wheel 缩放期间启用 GPU 合成层
        dom.canvasWrapper.style.willChange = 'transform';
        
        main_update_move_bound();
        main_update_canvas_position();
        // 连续缩放用 RAF 直写 transform，避免 CSS transition 每 tick 重算
        main_update_transform_schedule(state.canvasX, state.canvasY, state.scale);
        
        main_update_eraser_hint_size();
        if (window.tileRenderer) window.tileRenderer.mark_visible();
        
        // wheel 滚动停稳后缓动归位，防抖 150ms
        if (wheel_smooth_timer_id !== null) clearTimeout(wheel_smooth_timer_id);
        wheel_smooth_timer_id = setTimeout(() => {
            wheel_smooth_timer_id = null;
            state.isZooming = false;
            main_update_canvas_transform_smooth(state.canvasX, state.canvasY, state.scale, 150);
        }, 150);
    }
}

function main_update_canvas_transform() {
    if (last_canvas_transform.x === state.canvasX && 
        last_canvas_transform.y === state.canvasY && 
        last_canvas_transform.scale === state.scale) {
        return;
    }
    
    const scaleChanged = last_canvas_transform.scale !== state.scale;
    last_canvas_transform.x = state.canvasX;
    last_canvas_transform.y = state.canvasY;
    last_canvas_transform.scale = state.scale;
    
    dom.canvasWrapper.style.transform = 'translate3d(' + state.canvasX + 'px, ' + state.canvasY + 'px, 0) scale(' + state.scale + ')';

    // 仅 scale 变化时更新 tile DPR（平移/惯性期间跳过冗余调用）
    if (scaleChanged && window.tileRenderer) {
        window.tileRenderer.update_visible_tile_dpr(state.scale, false, true);
    }
    if (window.batchDrawManager) {
        window.batchDrawManager.update_overlay_dpr(state.scale);
    }
}

function main_update_canvas_transform_smooth(targetX, targetY, targetScale, duration = 200) {
    if (currentAnimationId !== null) {
        clearTimeout(currentAnimationId);
        currentAnimationId = null;
    }
    
    state.canvasX = targetX;
    state.canvasY = targetY;
    state.scale = targetScale;
    
    main_update_move_bound();
    main_update_canvas_position();
    
    last_canvas_transform.x = state.canvasX;
    last_canvas_transform.y = state.canvasY;
    last_canvas_transform.scale = state.scale;
    
    dom.canvasWrapper.style.transitionDuration = duration + 'ms';
    dom.canvasWrapper.classList.add('smooth-transform');
    dom.canvasWrapper.style.transform = 'translate3d(' + state.canvasX + 'px, ' + state.canvasY + 'px, 0) scale(' + state.scale + ')';

    // 更新瓦片与覆盖层（内部有 hysteresis 保护，不会冗余重建）
    if (window.tileRenderer) {
        window.tileRenderer.mark_visible();
        window.tileRenderer.update_visible_tile_dpr(state.scale, false, true);
    }
    if (window.batchDrawManager) {
        window.batchDrawManager.update_overlay_dpr(state.scale);
    }
    
    currentAnimationId = setTimeout(() => {
        currentAnimationId = null;
        dom.canvasWrapper.classList.remove('smooth-transform');
        dom.canvasWrapper.style.transitionDuration = '';
        dom.canvasWrapper.style.willChange = '';
    }, duration);
}

// 撤销功能 - 混合方案：路径记录 + ImageData 压缩
function main_start_stroke(type, eraserShape) {
    const invScale = 1 / main_fetch_safe_scale();
    const baseEraserSize = DRAW_CONFIG.eraserSize * invScale;
    state.currentStroke = {
        type: type,
        points: [],
        color: type === 'draw' ? DRAW_CONFIG.penColor : '#000000',
        lineWidth: (type === 'draw' ? DRAW_CONFIG.penWidth : DRAW_CONFIG.eraserSize) * invScale,
        eraserSize: baseEraserSize,
        eraserSizeRaw: DRAW_CONFIG.eraserSize,
        eraserShape: eraserShape || 'square',
        ...(window.__eraserSpeed ? window.__eraserSpeed.eraser_speed_build_config(DRAW_CONFIG, invScale) : { eraserSpeedEnabled: false }),
        scale: state.scale,
        bounds: {
            minX: Infinity,
            minY: Infinity,
            maxX: -Infinity,
            maxY: -Infinity
        },
        variableWidths: []
    };
    
    state.currentPressure = 0.5;

    state.cachedDrawType = type;
    state.cachedDrawColor = type === 'draw' ? DRAW_CONFIG.penColor : '#000000';
    const startScale = main_fetch_safe_scale();
    state.cachedDrawLineWidth = type === 'draw' ? DRAW_CONFIG.penWidth / startScale : DRAW_CONFIG.eraserSize / startScale;
    
    state.eraserSpeedState = window.__eraserSpeed?.eraser_speed_create_state() ?? null;
    
    batchDrawManager.eraserShape = state.currentStroke.eraserShape;
    batchDrawManager.batch_draw_init_start();
}

function main_save_stroke_point(fromX, fromY, toX, toY) {
    const stroke = state.currentStroke;
    if (!stroke) return;
    
    const bounds = stroke.bounds;
    if (fromX < bounds.minX) bounds.minX = fromX;
    if (toX < bounds.minX) bounds.minX = toX;
    if (fromY < bounds.minY) bounds.minY = fromY;
    if (toY < bounds.minY) bounds.minY = toY;
    if (fromX > bounds.maxX) bounds.maxX = fromX;
    if (toX > bounds.maxX) bounds.maxX = toX;
    if (fromY > bounds.maxY) bounds.maxY = fromY;
    if (toY > bounds.maxY) bounds.maxY = toY;
    
    let currentWidth = stroke.lineWidth;
    const currentScale = main_fetch_safe_scale();
    
    if (stroke.type === 'draw') {
        state.cachedDrawLineWidth = DRAW_CONFIG.penWidth / currentScale;
    } else if (stroke.type === 'erase' && stroke.eraserSpeedEnabled) {
        currentWidth = window.__eraserSpeed.eraser_speed_update(state.eraserSpeedState, stroke, toX, toY);
        state.cachedDrawLineWidth = currentWidth;
    } else if (stroke.type === 'erase') {
        state.cachedDrawLineWidth = DRAW_CONFIG.eraserSize / currentScale;
    }
    
    stroke.variableWidths.push(currentWidth);
    
    const points = stroke.points;
    points.push({ fromX, fromY, toX, toY });
}

async function main_submit_stroke() {
    if (state.currentStroke && state.currentStroke.points.length > 0) {
        // 强制刷新待处理命令，确保 _storedWidths 包含所有段的线宽
        batchDrawManager.batch_draw_handle_flush();
        // limited 模式：末尾添加收尾渐变
        const penMode = window.get_pen_effect_mode ? window.get_pen_effect_mode() : 'off';
        if (penMode === 'limited' && batchDrawManager._storedWidths.length > 0) {
            const baseW = state.currentStroke.lineWidth || DRAW_CONFIG.penWidth || 5;
            batchDrawManager._apply_speed_taper(batchDrawManager._storedWidths, state.currentStroke.points, baseW);
        }
        // 捕获实时绘制的逐段宽度，确保离线渲染与实时预览一致
        const storedWidths = batchDrawManager._storedWidths;
        if (storedWidths && storedWidths.length === state.currentStroke.points.length) {
            state.currentStroke.storedWidths = [...storedWidths];
        }
        
        const halfWidth = Math.max(state.currentStroke.lineWidth || 5, state.currentStroke.eraserSize || 5) / 2;
        const strokeBounds = state.currentStroke && state.currentStroke.bounds
            ? {
                minX: state.currentStroke.bounds.minX - halfWidth,
                minY: state.currentStroke.bounds.minY - halfWidth,
                maxX: state.currentStroke.bounds.maxX + halfWidth,
                maxY: state.currentStroke.bounds.maxY + halfWidth
            } : null;
        
        const cmd = new DrawCommand({
            stroke: state.currentStroke,
            strokeHistoryRef: state.strokeHistory,
            redrawFn: () => main_render_all_strokes(strokeBounds)
        });
        await history_execute_command(cmd, false);

        if (state.currentStroke.type === 'erase') {
            if (window.tileRenderer) {
                await main_render_all_strokes(strokeBounds);
            }
        } else {
            if (window.tileRenderer) {
                await window.tileRenderer.add_stroke(state.currentStroke);
            }
        }

        if (history_validate_compact()) {
            main_init_compact();
        }
    }
    state.currentStroke = null;

    await batchDrawManager.batch_draw_handle_end();

    batchDrawManager.batch_draw_delete_all();
}

async function main_render_all_strokes(bounds) {
    main_reset_context_state();
    const tr = window.tileRenderer;
    if (!tr) return;

    if (state.strokeHistory.length === 0 && !state.baseImageObj) {
        tr.mark_strokes_changed();
        tr.for_each((info) => {
            const ctx = info.ctx;
            const dpr = info.dpr;
            ctx.save();
            ctx.setTransform(dpr, 0, 0, dpr, -info.rect.x * dpr, -info.rect.y * dpr);
            ctx.clearRect(info.rect.x, info.rect.y, info.rect.width, info.rect.height);
            ctx.restore();
        });
        tr.dirty.clear();
        return;
    }

    tr.mark_strokes_changed();

    if (bounds && isFinite(bounds.minX) && isFinite(bounds.minY) &&
                  isFinite(bounds.maxX) && isFinite(bounds.maxY)) {
        const infos = tr.infos_for_segment(
            bounds.minX, bounds.minY,
            bounds.maxX, bounds.maxY
        );
        for (const info of infos) {
            tr.dirty.add(info.key);
        }
    } else {
        tr.mark_all();
    }

    tr.rebuild_all();
}

function get_pen_effect_mode() { return getPenEffectMode(); }
window.get_pen_effect_mode = get_pen_effect_mode;

function main_reset_context_state() { resetContextState(); }
window.main_reset_context_state = main_reset_context_state;

function main_update_context_state(ctx, s) { updateContextState(ctx, s); }
window.main_update_context_state = main_update_context_state;

/**
 * 按原始顺序逐个绘制笔画：draw/comment 用 source-over，erase 用 destination-out
 * @param {CanvasRenderingContext2D} ctx
 * @param {Array} strokes - 笔画数组
 */
async function main_render_strokes_to_context(ctx, strokes) {
    return renderStrokesToContext(ctx, strokes, {
        renderScale: main_fetch_safe_scale(),
        penManager: realPenManager
    });
}
window.main_render_strokes_to_context = main_render_strokes_to_context;

function main_init_compact() { historyCompactor.initCompaction(); }

async function main_handle_undo() {
    historyCompactor.cancelCompaction();
    state.baseImageLoadId++;
    state.compactSnapshotId = (state.compactSnapshotId || 0) + 1;
    realPenManager.invalidate_cache();
    await history_handle_undo();
    console.log('撤销操作');
}

function main_update_history_button_status() {
    dom.btnUndo.disabled = !history_validate_undo();
}

// 清空画布
function main_delete_draw_canvas() {
    if (window.tileRenderer) {
        window.tileRenderer.destroy_all();
        window.tileRenderer.init_tiles(dom.canvasWrapper);
    }
    if (window.batchDrawManager) {
        window.batchDrawManager.clear_overlay();
    }
    main_reset_context_state();
}

async function main_delete_all_drawings() {
    if (state.strokeHistory.length === 0 && !state.baseImageObj) return;
    
    const cmd = new ClearCommand({
        savedStrokeHistory: [...state.strokeHistory],
        savedBaseImageURL: state.baseImageURL,
        strokeHistoryRef: state.strokeHistory,
        baseImageURLRef: { get value() { return state.baseImageURL; }, set value(v) { state.baseImageURL = v; } },
        baseImageObjRef: { get value() { return state.baseImageObj; }, set value(v) { state.baseImageObj = v; } },
        redrawFn: () => main_render_all_strokes(),
        loadBaseImageFn: (url) => main_load_base_image(url)
    });
    await history_execute_command(cmd);
    
    main_delete_draw_canvas();
    
    if (currentSourceId) {
        main_save_current_source_data();
    }
    
    if (state.drawMode === 'eraser') {
        main_update_mode('comment');
    }
    
    console.log('清空所有批注');
}

function main_load_base_image(url) {
    const loadId = ++state.baseImageLoadId;
    const img = new Image();
    img.onload = () => {
        if (loadId === state.baseImageLoadId) {
            state.baseImageObj = img;
            if (window.tileRenderer) window.tileRenderer.mark_all();
            main_render_all_strokes();
        }
    };
    img.onerror = () => {
        console.error('base image 加载失败:', url ? url.substring(0, 50) + '...' : 'null');
        if (loadId === state.baseImageLoadId) {
            state.baseImageObj = null;
            if (window.tileRenderer) window.tileRenderer.mark_all();
            main_render_all_strokes();
        }
    };
    img.src = url;
}

// 拍照/切换回摄像头/保存画布截图
function main_save_photo() {
    if (state.isCameraOpen) {
        main_save_camera_image();
        return;
    }

    // 从图片或文档源切换回摄像头：优先打开摄像头（利用用户手势瞬态激活），再切换源
    if (state.currentImageIndex >= 0 || state.currentFolderIndex >= 0) {
        (async () => {
            // 记录当前源类型和清理回调
            const is_image = state.currentImageIndex >= 0 && state.imageList.length > 0;
            try {
                // 1. 先尝试打开摄像头（此时用户手势激活尚未过期）
                if (!state.isCameraOpen) {
                    await main_update_camera_state(true);
                }
                // 2. 摄像头已打开，清理旧源状态
                // 注意：main_update_camera_state 已通过 main_save_current_source_data + main_update_source('cam')
                // 完成源保存和切换，此处只需清理额外 UI 状态
                if (is_image) {
                    state.currentImageIndex = -1;
                    state.currentImage = null;
                } else {
                    state.currentFolderIndex = -1;
                    state.currentFolderPageIndex = -1;
                    state.currentImage = null;
                }
                main_delete_image_layer();
                main_delete_draw_canvas();
                main_update_sidebar_selection();
                main_update_photo_button_state();
            } catch (error) {
                console.error('返回摄像头失败:', error?.name, error?.message);
                if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
                    main_show_no_camera_message(window.i18n?.format_translate('camera.notDetected') || '未检测到摄像头');
                } else if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
                    main_show_no_camera_message(window.i18n?.format_translate('camera.noPermission') || '无摄像头权限');
                } else if (error.name !== 'AbortError') {
                    main_show_no_camera_message(window.i18n?.format_translate('camera.initFailed') || '摄像头初始化失败');
                }
            }
        })();
        return;
    }

    // 无摄像头/图片/文档源时：保存画布截图
    main_save_merged_canvas();
}

function main_save_merged_canvas() {
    console.log('执行拍照功能');
    const offscreen = main_fetch_offscreen_canvas();
    const mergedCtx = offscreen.ctx;
    
    mergedCtx.fillStyle = '#3a3a3a';
    mergedCtx.fillRect(0, 0, DRAW_CONFIG.canvasW, DRAW_CONFIG.canvasH);
    
    if (dom.imageElement.src) {
        mergedCtx.drawImage(dom.imageElement, 
            parseFloat(dom.imageElement.style.left) || 0, 
            parseFloat(dom.imageElement.style.top) || 0, 
            parseFloat(dom.imageElement.style.width) || DRAW_CONFIG.canvasW, 
            parseFloat(dom.imageElement.style.height) || DRAW_CONFIG.canvasH
        );
    }
    const tr = window.tileRenderer;
    if (tr) {
        for (const info of tr.tileInfos) {
            if (info.canvas) {
                mergedCtx.drawImage(
                    info.canvas,
                    0, 0,
                    info.canvas.width, info.canvas.height,
                    info.rect.x, info.rect.y,
                    info.rect.width, info.rect.height
                );
            }
        }
    }
    
    const link = document.createElement('a');
    link.download = `photo_${Date.now()}.png`;
    link.href = offscreen.canvas.toDataURL('image/png');
    link.click();
    
    main_release_offscreen_canvas(offscreen);
}

function main_update_photo_button_state() {
    cameraManager.updatePhotoButtonState();
}

// 设置功能
function main_show_settings() {
    const existingPanel = dom.settingsPanel.classList.contains('visible');
    if (existingPanel) {
        main_hide_settings_panel();
    } else {
        main_show_settings_panel();
    }
}

function main_show_settings_panel() {
    main_hide_pen_control_panel();
    
    const panel = dom.settingsPanel;
    const btnRect = dom.btnSettings.getBoundingClientRect();
    const containerRect = document.querySelector('.main-function').getBoundingClientRect();
    
    const panelWidth = 130;
    const panelHeight = panel.offsetHeight || 50;
    
    let left = btnRect.left - containerRect.left + (btnRect.width / 2) - (panelWidth / 2);
    let top = btnRect.top - containerRect.top - panelHeight - 10;
    
    if (left < 10) left = 10;
    if (left + panelWidth > containerRect.width - 10) {
        left = containerRect.width - panelWidth - 10;
    }
    
    if (top < 10) {
        top = btnRect.bottom - containerRect.top + 10;
    }
    
    panel.style.left = `${left}px`;
    panel.style.top = `${top}px`;
    panel.classList.add('visible');

    main_update_settings_controls_state();
}

function main_hide_settings_panel() {
    dom.settingsPanel.classList.remove('visible');
}

function main_update_settings_controls_state() {
    const brightnessRow = dom.settingsPanel?.querySelector('.settings-slider-row:has(#cameraBrightness)');
    const contrastRow = dom.settingsPanel?.querySelector('.settings-slider-row:has(#cameraContrast)');
    const brightnessInput = document.getElementById('cameraBrightness');
    const contrastInput = document.getElementById('cameraContrast');

    const disabled = !state.isCameraOpen;

    if (brightnessRow) {
        brightnessRow.classList.toggle('settings-controls-disabled', disabled);
    }
    if (contrastRow) {
        contrastRow.classList.toggle('settings-controls-disabled', disabled);
    }
    if (brightnessInput) {
        brightnessInput.disabled = disabled;
    }
    if (contrastInput) {
        contrastInput.disabled = disabled;
    }
}

function main_show_settings_window() {
    if (window.__TAURI__) {
        const { invoke } = window.__TAURI__.core;
        invoke('window_show_settings').catch(error => {
            console.error('打开设置窗口失败:', error);
        });
    }
}

async function main_update_image_rotation(direction) {
    if (state.isCameraOpen) {
        if (direction === 'left') {
            state.cameraRotation = (state.cameraRotation - 90 + 360) % 360;
        } else {
            state.cameraRotation = (state.cameraRotation + 90) % 360;
        }
        main_update_camera_video_style();
        console.log(`摄像头画面已旋转到 ${state.cameraRotation}°`);
        return;
    }
    
    if (!state.currentImage) {
        console.log('没有图片可旋转');
        return;
    }
    
    let rotatedDataUrl;
    
    if (window.__TAURI__) {
        try {
            const { invoke } = window.__TAURI__.core;
            rotatedDataUrl = await invoke('image_update_rotation', { 
                imageData: state.currentImage.src, 
                direction: direction 
            });
            console.log('Rust 图片旋转完成');
        } catch (error) {
            console.error('Rust 图片旋转失败，使用前端降级方案:', error);
            rotatedDataUrl = main_update_image_rotation_fallback(state.currentImage, direction);
        }
    } else {
        rotatedDataUrl = main_update_image_rotation_fallback(state.currentImage, direction);
    }
    
    const rotatedImg = new Image();
    rotatedImg.onload = async () => {
        state.currentImage = rotatedImg;
        
        if (state.currentImageIndex >= 0 && state.currentImageIndex < state.imageList.length) {
            state.imageList[state.currentImageIndex].full = rotatedImg.src;
            state.imageList[state.currentImageIndex].thumbnail = rotatedImg.src;
            state.imageList[state.currentImageIndex].width = rotatedImg.width;
            state.imageList[state.currentImageIndex].height = rotatedImg.height;
            
            main_update_sidebar_content();
        }
        
        main_render_image_centered(rotatedImg);
        console.log(`图片已向${direction === 'left' ? '左' : '右'}旋转`);
    };
    rotatedImg.src = rotatedDataUrl;
}

function main_update_image_rotation_fallback(img, direction) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    if (direction === 'left') {
        canvas.width = img.height;
        canvas.height = img.width;
        ctx.translate(canvas.width / 2, canvas.height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.drawImage(img, -img.width / 2, -img.height / 2);
    } else {
        canvas.width = img.height;
        canvas.height = img.width;
        ctx.translate(canvas.width / 2, canvas.height / 2);
        ctx.rotate(Math.PI / 2);
        ctx.drawImage(img, -img.width / 2, -img.height / 2);
    }
    
    return canvas.toDataURL('image/png');
}

const SIDEBAR_PLACEHOLDER = 'data:image/svg+xml,' + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="200" height="113"><rect fill="#2a2a2e" width="200" height="113"/></svg>');

let sidebarObserver = null;

function main_destroy_sidebar_lazy_loader() {
    if (sidebarObserver) {
        sidebarObserver.disconnect();
        sidebarObserver = null;
    }
}

function main_setup_sidebar_lazy_loader(sidebarContent) {
    main_destroy_sidebar_lazy_loader();

    sidebarObserver = new IntersectionObserver((entries) => {
        for (const entry of entries) {
            const item = entry.target;
            const img = item.querySelector('.sidebar-thumbnail');
            if (!img) continue;

            if (entry.isIntersecting) {
                if (img.src === SIDEBAR_PLACEHOLDER) {
                    const index = parseInt(item.dataset.index);
                    const imgData = state.imageList[index];
                    if (imgData && imgData.thumbnail) {
                        img.src = imgData.thumbnail;
                    }
                }
            }
        }
    }, {
        root: sidebarContent,
        rootMargin: '300px 0px'
    });

    sidebarContent.querySelectorAll('.sidebar-image-item').forEach(item => {
        sidebarObserver.observe(item);
    });
}

function main_handle_sidebar_toggle() {
    const existingSidebar = document.querySelector('.sidebar:not(.file-sidebar)');
    const existingFileSidebar = document.querySelector('.file-sidebar');
    
    if (existingFileSidebar) {
        main_hide_file_sidebar();
    }
    
    if (existingSidebar) {
        main_hide_sidebar();
    } else {
        main_show_sidebar();
    }
}

function main_show_sidebar() {
    const sidebarElement = document.createElement('div');
    sidebarElement.classList.add('sidebar');
    
    const noImagesText = window.i18n?.format_translate('common.noImages') || '暂无图片';
    const imageListText = window.i18n?.format_translate('sidebar.imageList') || '图片列表';
    const importImageText = window.i18n?.format_translate('sidebar.importImage') || '导入图片';
    const deleteText = window.i18n?.format_translate('common.delete') || '删除';
    const collapseText = window.i18n?.format_translate('common.collapse') || '收起';
    
    let imageListHTML = '';
    if (state.imageList.length === 0) {
        imageListHTML = `<div class="sidebar-empty">${noImagesText}</div>`;
    } else {
        state.imageList.forEach((imgData, index) => {
            const isActive = (state.currentImageIndex >= 0 && index === state.currentImageIndex) ? 'active' : '';
            const imageAlt = window.i18n?.format_translate('sidebar.imageAlt', { n: index + 1 }) || `图片${index + 1}`;
            const filterStyle = imgData.captureFilter ? ` style="filter: ${imgData.captureFilter.replace(/"/g, '&quot;')}"` : '';
            imageListHTML += `
                <div class="sidebar-image-item ${isActive}" data-index="${index}">
                    <img src="${imgData.thumbnail}" class="sidebar-thumbnail" alt="${imageAlt}" loading="lazy"${filterStyle}>
                    <div class="sidebar-image-actions">
                        <button class="sidebar-btn-delete" title="${deleteText}">✕</button>
                    </div>
                </div>
            `;
        });
    }
    
    sidebarElement.innerHTML = `
        <div class="sidebar-header"><span class="sidebar-header-text">${imageListText}</span></div>
        <div class="sidebar-content">
            ${imageListHTML}
        </div>
        <button class="sidebar-import-btn" id="btnImportImageSidebar">
            ${ThemeManager.theme_fetch_icon('image', { alt: importImageText })}
            ${importImageText}
        </button>
    `;
    dom.canvasContainer.appendChild(sidebarElement);
    
    document.getElementById('btnImportImageSidebar')?.addEventListener('click', main_load_image);
    
    document.querySelectorAll('.sidebar-image-item').forEach(item => {
        const index = parseInt(item.dataset.index);
        
        item.querySelector('.sidebar-btn-delete')?.addEventListener('click', (e) => {
            e.stopPropagation();
            main_delete_image(index);
        });
        
        item.addEventListener('click', () => main_update_image_selection(index));
    });
    
    const sidebarContent = sidebarElement.querySelector('.sidebar-content');
    if (sidebarContent && state.imageList.length > 0) {
        main_setup_sidebar_lazy_loader(sidebarContent);
    }
    
    const showText = ThemeManager.theme_fetch_toolbar_text();
    dom.btnExpand.innerHTML = `
        ${ThemeManager.theme_fetch_icon('collapse', { alt: collapseText })}
        ${showText ? collapseText : ''}
    `;
    console.log('展开侧边栏');
}

async function main_update_image_selection(index) {
    if (index < 0 || index >= state.imageList.length) return;
    
    if (index === state.currentImageIndex && state.currentImage) {
        (async () => {
            try {
                main_save_current_source_data();
                
                state.currentImageIndex = -1;
                state.currentImage = null;
                currentSourceId = null;
                main_delete_image_layer();
                main_delete_draw_canvas();
                
                if (state.isCameraOpen) {
                    await main_update_camera_state(false);
                }
                
                await main_update_source('cam');
                
                if (!state.isCameraOpen) {
                    await main_update_camera_state(true);
                }
                
                main_update_sidebar_selection();
                main_update_photo_button_state();
                console.log('返回摄像头');
            } catch (error) {
                console.error('返回摄像头失败:', error);
            }
        })();
        return;
    }
    
    state.currentImageIndex = index;
    state.currentFolderIndex = -1;
    state.currentFolderPageIndex = -1;
    
    // 使用源ID切换源（自动保存当前并加载目标源的数据）
    const imgData = state.imageList[index];
    
    if (!imgData.sourceId) {
        imgData.sourceId = main_create_source_id('pic');
    }
    
    await main_update_source(imgData.sourceId);
    
    const img = new Image();
    img.onload = async () => {
        state.currentImage = img;
        
        if (state.isCameraOpen) {
            await main_update_camera_state(false);
        }
        if (window.__lightAvailable) {
            window.__savedLightState = window.__lightOn;
            main_light_off();
            dom.btnLight.style.display = 'none';
        }
        main_render_image_centered(img);
        
        // 恢复拍摄时的 CSS filter，保证与实际效果一致
        if (imgData.captureFilter) {
            dom.imageElement.style.filter = imgData.captureFilter;
        }
        
        await main_render_all_strokes();
        main_update_sidebar_selection();
        main_update_photo_button_state();
    };
    img.onerror = () => {
        console.error(`加载图片 ${index + 1} 失败`);
    };
    img.src = imgData.full;
    
    console.log(`切换到图片 ${index + 1}`);
}

function main_delete_image(index) {
    if (index < 0 || index >= state.imageList.length) return;
    
    const imgData = state.imageList[index];
    if (imgData.full && imgData.full.startsWith('blob:')) {
        URL.revokeObjectURL(imgData.full);
    }
    if (imgData.thumbnail && imgData.thumbnail.startsWith('blob:')) {
        URL.revokeObjectURL(imgData.thumbnail);
    }
    
    state.imageList.splice(index, 1);
    
    if (state.currentImageIndex === index) {
        if (state.imageList.length > 0) {
            const newIndex = Math.min(index, state.imageList.length - 1);
            state.currentImageIndex = -1;
            main_update_image_selection(newIndex);
        } else {
            state.currentImageIndex = -1;
            state.currentImage = null;
            main_delete_image_layer();
            main_delete_draw_canvas();
            main_update_photo_button_state();
            main_init_camera();
        }
    } else if (state.currentImageIndex > index) {
        state.currentImageIndex--;
    }
    
    main_update_sidebar_content();
    console.log(`删除图片 ${index + 1}`);
}

let lastSidebarSelection = -2;

function main_update_sidebar_selection() {
    if (lastSidebarSelection === state.currentImageIndex) return;
    
    const sidebarContent = document.querySelector('.sidebar:not(.file-sidebar) .sidebar-content');
    if (!sidebarContent) return;
    
    const items = sidebarContent.querySelectorAll('.sidebar-image-item');
    
    if (lastSidebarSelection >= 0 && lastSidebarSelection < items.length) {
        const prevItem = items[lastSidebarSelection];
        prevItem.classList.remove('active');
    }
    
    if (state.currentImageIndex >= 0 && state.currentImageIndex < items.length) {
        const curItem = items[state.currentImageIndex];
        curItem.classList.add('active');
        const curImg = curItem.querySelector('.sidebar-thumbnail');
        if (curImg && curImg.src === SIDEBAR_PLACEHOLDER) {
            const imgData = state.imageList[state.currentImageIndex];
            if (imgData && imgData.thumbnail) {
                curImg.src = imgData.thumbnail;
            }
        }
    }
    
    lastSidebarSelection = state.currentImageIndex;
    
    document.querySelectorAll('.file-sidebar .sidebar-image-item').forEach(item => {
        item.classList.remove('active');
    });
}

function main_update_sidebar_content() {
    const sidebarContent = document.querySelector('.sidebar:not(.file-sidebar) .sidebar-content');
    if (!sidebarContent) return;
    
    const noImagesText = window.i18n?.format_translate('common.noImages') || '暂无图片';
    const deleteText = window.i18n?.format_translate('common.delete') || '删除';
    
    let imageListHTML = '';
    if (state.imageList.length === 0) {
        imageListHTML = `<div class="sidebar-empty">${noImagesText}</div>`;
    } else {
        state.imageList.forEach((imgData, index) => {
            const isActive = (state.currentImageIndex >= 0 && index === state.currentImageIndex) ? 'active' : '';
            const imageAlt = window.i18n?.format_translate('sidebar.imageAlt', { n: index + 1 }) || `图片${index + 1}`;
            const filterStyle = imgData.captureFilter ? ` style="filter: ${imgData.captureFilter.replace(/"/g, '&quot;')}"` : '';
            imageListHTML += `
                <div class="sidebar-image-item ${isActive}" data-index="${index}">
                    <img src="${imgData.thumbnail}" class="sidebar-thumbnail" alt="${imageAlt}" loading="lazy"${filterStyle}>
                    <div class="sidebar-image-actions">
                        <button class="sidebar-btn-delete" title="${deleteText}">✕</button>
                    </div>
                </div>
            `;
        });
    }
    
    sidebarContent.innerHTML = imageListHTML;
    
    document.querySelectorAll('.sidebar-image-item').forEach(item => {
        const index = parseInt(item.dataset.index);
        
        item.querySelector('.sidebar-btn-delete')?.addEventListener('click', (e) => {
            e.stopPropagation();
            main_delete_image(index);
        });
        
        item.addEventListener('click', () => main_update_image_selection(index));
    });
    
    if (state.imageList.length > 0) {
        main_setup_sidebar_lazy_loader(sidebarContent);
        const activeItem = sidebarContent.querySelector('.sidebar-image-item.active');
        if (activeItem) {
            activeItem.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
        }
    }
}

function main_hide_sidebar() {
    const sidebar = document.querySelector('.sidebar');
    if (sidebar) {
        sidebar.classList.add('collapse');
        sidebar.addEventListener('animationend', function() {
            sidebar.remove();
        }, { once: true });
    }
    
    main_destroy_sidebar_lazy_loader();
    
    const imageText = window.i18n?.format_translate('toolbar.image') || '图片';
    const showText = ThemeManager.theme_fetch_toolbar_text();
    dom.btnExpand.innerHTML = `
        ${ThemeManager.theme_fetch_icon('image', { alt: imageText })}
        ${showText ? imageText : ''}
    `;
    console.log('收起侧边栏');
}

// 文件侧边栏
function main_handle_file_sidebar_toggle() {
    const existingFileSidebar = document.querySelector('.file-sidebar');
    const existingSidebar = document.querySelector('.sidebar:not(.file-sidebar)');
    
    if (existingSidebar) {
        main_hide_sidebar();
    }
    
    if (existingFileSidebar) {
        main_hide_file_sidebar();
    } else {
        main_show_file_sidebar();
    }
}

function main_show_file_sidebar() {
    const existingSidebar = document.querySelector('.file-sidebar');
    if (existingSidebar) {
        main_update_file_sidebar_content();
        return;
    }
    
    const noFilesText = window.i18n?.format_translate('common.noFiles') || '暂无文件';
    const fileListText = window.i18n?.format_translate('sidebar.fileList') || '文件列表';
    const addFileText = window.i18n?.format_translate('sidebar.addFile') || '添加文件';
    const collapseText = window.i18n?.format_translate('common.collapse') || '收起';
    
    const fileSidebarElement = document.createElement('div');
    fileSidebarElement.classList.add('sidebar', 'file-sidebar');
    
    let contentHTML = '';
    if (state.fileList.length === 0) {
        contentHTML = `<div class="sidebar-empty">${noFilesText}</div>`;
    } else {
        state.fileList.forEach((folder, index) => {
            const isWord = folder.isWord === true;
            const iconName = isWord ? 'word' : 'pdf';
            const fileAlt = window.i18n?.format_translate('toolbar.file') || '文件';
            const pagesText = window.i18n?.format_translate('sidebar.pages', { n: folder.pages.length }) || `${folder.pages.length}页`;
            contentHTML += `
                <div class="sidebar-folder-item" data-index="${index}">
                    ${ThemeManager.theme_fetch_icon(iconName, { alt: fileAlt })}
                    <span class="folder-name">${folder.name}</span>
                    <span class="folder-count">${pagesText}</span>
                </div>
            `;
        });
    }
    
    fileSidebarElement.innerHTML = `
        <div class="sidebar-header"><span class="sidebar-header-text">${fileListText}</span></div>
        <div class="sidebar-content">
            ${contentHTML}
        </div>
        <button class="sidebar-import-btn" id="btnAddFile">
            ${ThemeManager.theme_fetch_icon('addFile', { alt: addFileText })}
            ${addFileText}
        </button>
    `;
    
    dom.canvasContainer.appendChild(fileSidebarElement);
    
    document.getElementById('btnAddFile')?.addEventListener('click', () => {
        main_load_pdf();
    });
    
    document.querySelectorAll('.sidebar-folder-item').forEach(item => {
        item.addEventListener('click', () => {
            const index = parseInt(item.dataset.index);
            if (window.documentReaderManager) {
                main_hide_file_sidebar();
                window.documentReaderManager.open(index);
            }
        });
    });
    
    const showText = ThemeManager.theme_fetch_toolbar_text();
    dom.btnSave.innerHTML = `
        ${ThemeManager.theme_fetch_icon('collapse', { alt: collapseText })}
        ${showText ? collapseText : ''}
    `;
    console.log('展开文件侧边栏');
}



function main_update_file_sidebar_content() {
    const sidebarContent = document.querySelector('.file-sidebar .sidebar-content');
    if (!sidebarContent) return;
    
    const noFilesText = window.i18n?.format_translate('common.noFiles') || '暂无文件';
    
    let contentHTML = '';
    if (state.fileList.length === 0) {
        contentHTML = `<div class="sidebar-empty">${noFilesText}</div>`;
    } else {
        state.fileList.forEach((folder, index) => {
            const isWord = folder.isWord === true;
            const iconName = isWord ? 'word' : 'pdf';
            console.log(`文件夹 ${folder.name}: isWord=${isWord}, iconName=${iconName}`);
            const fileAlt = window.i18n?.format_translate('toolbar.file') || '文件';
            const pagesText = window.i18n?.format_translate('sidebar.pages', { n: folder.pages.length }) || `${folder.pages.length}页`;
            contentHTML += `
                <div class="sidebar-folder-item" data-index="${index}">
                    ${ThemeManager.theme_fetch_icon(iconName, { alt: fileAlt })}
                    <span class="folder-name">${folder.name}</span>
                    <span class="folder-count">${pagesText}</span>
                </div>
            `;
        });
    }
    
    sidebarContent.innerHTML = contentHTML;
    
    document.querySelectorAll('.sidebar-folder-item').forEach(item => {
        item.addEventListener('click', () => {
            const index = parseInt(item.dataset.index);
            if (window.documentReaderManager) {
                main_hide_file_sidebar();
                window.documentReaderManager.open(index);
            }
        });
    });
}

function main_load_pdf() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.pdf,.docx,.doc';
    
    input.onchange = async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        
        if (currentSourceId) {
            main_save_current_source_data();
        }
        
        const wasCameraOpen = state.isCameraOpen;
        
        if (state.isCameraOpen) {
            await main_update_camera_state(false);
        }
        
        const fileName = file.name.toLowerCase();
        const isWord = fileName.endsWith('.docx') || fileName.endsWith('.doc');
        
        if (isWord) {
            main_show_loading_overlay(window.i18n?.format_translate('loading.detectingOffice') || '正在检测 Office 软件...');
            
            const { invoke } = window.__TAURI__.core;
            
            let detection;
            try {
                detection = await invoke('office_detect_all');
                console.log('Office 检测结果:', detection);
                if (detection.recommended === 'None') {
                    main_hide_loading_overlay();
                    main_show_error_dialog(
                        window.i18n?.format_translate('errors.officeNotInstalled') || 'Office 未安装',
                        window.i18n?.format_translate('errors.officeNotInstalledDesc') || '未检测到可用的 Office 软件\n\n请安装以下软件之一：\n• Microsoft Word\n• WPS Office\n• LibreOffice\n\n或将 Word 文档另存为 PDF 后导入'
                    );
                    if (wasCameraOpen) await main_update_camera_state(true);
                    return;
                }
            } catch (e) {
                main_hide_loading_overlay();
                console.log('检测 Office 失败:', e);
                main_show_error_dialog(
                    window.i18n?.format_translate('errors.officeDetectFailed') || '检测失败',
                    window.i18n?.format_translate('errors.officeDetectFailedDesc') || '检测 Office 软件失败，请重试'
                );
                if (wasCameraOpen) await main_update_camera_state(true);
                return;
            }
            
            main_update_loading_progress(window.i18n?.format_translate('loading.readingFile') || '正在读取文件...');
            
            let arrayBuffer = await file.arrayBuffer();
            let uint8Array = new Uint8Array(arrayBuffer);
            const fileMd5 = main_calculate_md5(uint8Array);
            
            console.log('文件大小:', uint8Array.length, '字节');
            
            main_update_loading_progress(window.i18n?.format_translate('loading.processingWord') || '正在处理 Word 文档...');
            
            let pdfPath = null;
            try {
                const fileDataForConvert = Array.from(uint8Array);
                arrayBuffer = null;
                uint8Array = null;
                pdfPath = await invoke('office_convert_docx_to_pdf_bytes', {
                    fileData: fileDataForConvert,
                    fileName: file.name
                });
                console.log('Word 文档已转换为 PDF:', pdfPath);
            } catch (convertError) {
                main_hide_loading_overlay();
                console.error('Word 转换失败:', convertError);
                const errorMsg = String(convertError);
                let friendlyMsg = window.i18n?.format_translate('errors.wordConvertFailed') || 'Word 文档转换失败';
                
                if (errorMsg.includes('Office') || errorMsg.includes('Word') || errorMsg.includes('WPS')) {
                    friendlyMsg = window.i18n?.format_translate('errors.officeCallFailed') || 'Office 软件调用失败\n\n可能的原因：\n• Office 软件未正确安装\n• 文件被其他程序占用\n• 文件格式不支持';
                }
                
                main_show_error_dialog(
                    window.i18n?.format_translate('errors.convertFailed') || '转换失败',
                    friendlyMsg,
                    () => {
                        main_load_pdf();
                    }
                );
                if (wasCameraOpen) await main_update_camera_state(true);
                return;
            }
            
            main_update_loading_progress(window.i18n?.format_translate('loading.renderingPage') || '正在渲染页面...');
            
            try {
                const pdfReady = await main_wait_pdfjs();
                if (!pdfReady) {
                    main_hide_loading_overlay();
                    main_show_error_dialog(
                        window.i18n?.format_translate('errors.loadFailed') || '加载失败',
                        window.i18n?.format_translate('errors.pdfLoadTimeout') || 'PDF 库加载超时\n\n请重启应用后重试'
                    );
                    if (wasCameraOpen) await main_update_camera_state(true);
                    return;
                }
                
                const { readFile, remove } = window.__TAURI__.fs;
                let pdfBytes = await readFile(pdfPath);
                let pdfArrayBuffer = pdfBytes.buffer;
                const pdf = await pdfjsLib.getDocument({
                    data: pdfArrayBuffer,
                    enableXfa: false,
                    useSystemFonts: false,
                    isEvalSupported: false
                }).promise;
                pdfBytes = null;
                pdfArrayBuffer = null;
                
                const totalPages = pdf.numPages;
                const docNumber = sourceIdCounters.doc++;
                const folder = {
                    name: file.name.replace(/\.(pdf|docx|doc)$/i, ''),
                    pages: [],
                    isWord: true,
                    pdfDoc: pdf,
                    totalPages: totalPages,
                    docNumber: docNumber,
                    fileMd5: fileMd5
                };
                
                folder.pages = await main_render_pdf_pages_lazy(pdf, totalPages, PDF_INITIAL_RENDER_PAGES, docNumber);
                
                state.fileList.push(folder);
                main_update_file_sidebar_content();
                
                const existingFileSidebar = document.querySelector('.file-sidebar');
                if (!existingFileSidebar) {
                    main_show_file_sidebar();
                }
                
                main_hide_loading_overlay();
                console.log(`文件已导入: ${folder.name}，共${folder.pages.length}页`);

                // PDF 加载后内存占用可能较高，自动清理
                const invoke = window.__TAURI__?.core?.invoke;
                if (invoke) {
                    const now = Date.now();
                    if (!window.__memclean_last_auto || now - window.__memclean_last_auto >= 600000) {
                        invoke('memreduct_get_usage').then(usage => {
                            if (usage > 80) {
                                console.log(`[memclean] PDF加载后内存使用率 ${usage}%，自动清理`);
                                window.__memclean_last_auto = Date.now();
                                invoke('memreduct_clean_now', { mask: null }).catch(() => {});
                            }
                        }).catch(() => {});
                    }
                }

                if (wasCameraOpen) await main_update_camera_state(true);
                
                try {
                    await remove(pdfPath);
                } catch (e) {
                    console.log('清理转换的 PDF 失败:', e);
                }
            } catch (error) {
                main_hide_loading_overlay();
                console.error('文件导入失败:', error);
                main_show_error_dialog(
                    window.i18n?.format_translate('errors.importFailed') || '导入失败',
                    window.i18n?.format_translate('errors.importFailedDesc') || '文件导入失败，请确保文件格式正确'
                );
                if (wasCameraOpen) await main_update_camera_state(true);
            }
        } else {
            main_show_loading_overlay(window.i18n?.format_translate('loading.importingFile') || '正在导入文件...');
            
            try {
                const pdfReady = await main_wait_pdfjs();
                if (!pdfReady) {
                    main_hide_loading_overlay();
                    main_show_error_dialog(
                        window.i18n?.format_translate('errors.loadFailed') || '加载失败',
                        window.i18n?.format_translate('errors.pdfLoadTimeout') || 'PDF 库加载超时\n\n请重启应用后重试'
                    );
                    if (wasCameraOpen) await main_update_camera_state(true);
                    return;
                }
                
                let pdfArrayBuffer = await file.arrayBuffer();

                // PDF 解析（Worker 线程）与 MD5（主线程）并发执行
                const pdfPromise = pdfjsLib.getDocument({
                    data: pdfArrayBuffer,
                    enableXfa: false,
                    useSystemFonts: false,
                    isEvalSupported: false
                }).promise;
                const fileMd5 = main_calculate_md5(new Uint8Array(pdfArrayBuffer));
                const pdf = await pdfPromise;
                pdfArrayBuffer = null;
                
                const totalPages = pdf.numPages;
                const docNumber = sourceIdCounters.doc++;
                const folder = {
                    name: file.name.replace('.pdf', ''),
                    pages: [],
                    pdfDoc: pdf,
                    totalPages: totalPages,
                    docNumber: docNumber,
                    fileMd5: fileMd5
                };
                
                folder.pages = await main_render_pdf_pages_lazy(pdf, totalPages, PDF_INITIAL_RENDER_PAGES, docNumber);
                
                state.fileList.push(folder);
                main_update_file_sidebar_content();
                
                const existingFileSidebar = document.querySelector('.file-sidebar');
                if (!existingFileSidebar) {
                    main_show_file_sidebar();
                }
                
                main_hide_loading_overlay();
                console.log(`文件已导入: ${folder.name}，共${folder.pages.length}页`);

                // PDF 加载后内存占用可能较高，自动清理
                (() => {
                    const invoke = window.__TAURI__?.core?.invoke;
                    if (!invoke) return;
                    const now = Date.now();
                    if (window.__memclean_last_auto && now - window.__memclean_last_auto < 600000) return;
                    invoke('memreduct_get_usage').then(usage => {
                        if (usage > 80) {
                            console.log(`[memclean] PDF加载后内存使用率 ${usage}%，自动清理`);
                            window.__memclean_last_auto = Date.now();
                            invoke('memreduct_clean_now', { mask: null }).catch(() => {});
                        }
                    }).catch(() => {});
                })();

                if (wasCameraOpen) await main_update_camera_state(true);
            } catch (error) {
                main_hide_loading_overlay();
                console.error('文件导入失败:', error);
                main_show_error_dialog(
                    window.i18n?.format_translate('errors.importFailed') || '导入失败',
                    window.i18n?.format_translate('errors.importFailedDesc') || '文件导入失败，请确保文件格式正确'
                );
                if (wasCameraOpen) await main_update_camera_state(true);
            }
        }
    };
    
    input.click();
}

function main_show_loading_overlay(message) {
    DocLoader.show_loading_overlay(message);
}

function main_update_loading_progress(message) {
    DocLoader.update_loading_progress(message);
}

function main_hide_loading_overlay() {
    DocLoader.hide_loading_overlay();
}

function main_show_error_dialog(title, message, retryCallback = null) {
    DocLoader.show_error_dialog(title, message, retryCallback);
}

function main_hide_file_sidebar() {
    const fileSidebar = document.querySelector('.file-sidebar');
    if (fileSidebar) {
        fileSidebar.classList.add('collapse');
        fileSidebar.addEventListener('animationend', function() {
            fileSidebar.remove();
        }, { once: true });
    }
    
    const fileText = window.i18n?.format_translate('toolbar.file') || '文件';
    const showText = ThemeManager.theme_fetch_toolbar_text();
    dom.btnSave.innerHTML = `
        ${ThemeManager.theme_fetch_icon('file', { alt: fileText })}
        ${showText ? fileText : ''}
    `;
    console.log('收起文件侧边栏');
}

// === 摄像头功能（委托给 cameraManager） ===

async function main_update_camera_state(open, options = {}) {
    if (open) {
        await cameraManager.open();
    } else {
        await cameraManager.close();
    }
}

async function main_init_camera() { await cameraManager.toggle(); }
function main_setup_deferred_camera() { cameraManager.setupDeferred(); }
async function main_init_without_camera(message) { await cameraManager.initWithoutCamera(message); }
function main_show_no_camera_message(message) { cameraManager.showNoCameraMessage(message); }
function main_hide_no_camera_message() { cameraManager.hideNoCameraMessage(); }
async function main_update_camera() { await cameraManager.switchCamera(); }
function main_update_camera_video_style() { cameraManager.updateVideoStyle(); }
function main_apply_camera_filters() { cameraManager.applyFilters(); }
async function main_save_camera_image() { await cameraManager.saveImage(); }
function main_update_camera_frame_rate(idealFps) { cameraManager.updateFrameRate(idealFps); }

// ===== 摄像头预览自适应帧率 =====
// 批注模式下预览以 30fps 起步；绘制 flush 平均耗时持续超预算时降回 15fps，
// 负载消失并保持一段时间后恢复（滞回 + 冷却防抖动）。
let _camPreviewFpsHigh = true;
let _camFpsRecoverAt = 0;

/** 进入批注模式：重置自适应状态并以 30fps 预览 */
function main_enter_annotate_camera_fps() {
    _camPreviewFpsHigh = true;
    _camFpsRecoverAt = 0;
    main_update_camera_frame_rate(30);
}

/** 批注绘制性能采样回调（最近 20 次 overlay flush 平均耗时，节流 500ms） */
function main_handle_draw_perf_sample(avgMs) {
    if (!state.isCameraOpen || state.drawMode === 'move') return;
    const now = performance.now();
    if (_camPreviewFpsHigh) {
        if (avgMs > 10) {
            _camPreviewFpsHigh = false;
            _camFpsRecoverAt = 0;
            main_update_camera_frame_rate(15);
        }
    } else if (avgMs <= 5) {
        if (_camFpsRecoverAt === 0) {
            _camFpsRecoverAt = now;
        } else if (now - _camFpsRecoverAt >= 1000) {
            _camPreviewFpsHigh = true;
            _camFpsRecoverAt = 0;
            main_update_camera_frame_rate(30);
        }
    } else {
        // 仍高负载：重置恢复计时
        _camFpsRecoverAt = 0;
    }
}

function main_delete_sidebar_selection() {
    document.querySelectorAll('.sidebar:not(.file-sidebar) .sidebar-image-item').forEach(item => {
        item.classList.remove('active');
    });
    document.querySelectorAll('.file-sidebar .sidebar-image-item').forEach(item => {
        item.classList.remove('active');
    });
    state.currentImageIndex = -1;
    state.currentFolderPageIndex = -1;
}

function main_show_sidebar_if_hidden() {
    const sidebar = document.querySelector('.sidebar');
    if (!sidebar) { main_show_sidebar(); }
    else if (sidebar.classList.contains('file-sidebar')) { sidebar.remove(); main_show_sidebar(); }
}

// === 图像导入功能 ===
// 图片导入、拍照保存、PDF处理

/**
 * 导入图片文件（支持多选，批量导入时用 Rust 并行生成缩略图）
 */
async function main_load_image() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.multiple = true;
    
    input.onchange = async (e) => {
        const files = Array.from(e.target.files);
        if (files.length === 0) return;
        
        // 保存当前源数据，确保切换前批注不丢失
        if (currentSourceId) {
            main_save_current_source_data();
        }
        
        if (state.isCameraOpen) {
            await main_update_camera_state(false);
        }
        
        const hasLargeImage = files.some(file => file.size > 2.5 * 1024 * 1024);
        
        // 如果有大图片或者多个文件，显示加载动画
        if (files.length > 1 || hasLargeImage) {
            main_show_loading_overlay(window.i18n?.format_translate('loading.readingImages') || '正在读取图片...');
        }
        
        const imageDataList = [];
        
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            
            if (files.length > 1 || file.size > 2.5 * 1024 * 1024) {
                main_update_loading_progress(window.i18n?.format_translate('loading.readingImage', { current: i + 1, total: files.length }) || `正在读取图片 ${i + 1}/${files.length}...`);
            }
            
            const blobUrl = URL.createObjectURL(file);
            
            const imageName = file.name || window.i18n?.format_translate('sidebar.imageAlt', { n: state.imageList.length + imageDataList.length + 1 }) || `图片${state.imageList.length + imageDataList.length + 1}`;
            imageDataList.push({
                data: blobUrl,
                blob: file,
                name: imageName
            });
        }
        
        for (let i = 0; i < imageDataList.length; i++) {
            const imgData = imageDataList[i];
            const isLast = (i === imageDataList.length - 1);
            
            const img = new Image();
            await new Promise((resolve) => {
                img.onload = () => resolve();
                img.onerror = () => {
                    console.error(`加载图片失败: ${imgData.name}`);
                    resolve();
                };
                img.src = imgData.data;
            });
            
            const newImgData = {
                full: imgData.data,
                thumbnail: imgData.data,
                name: imgData.name,
                width: img.width,
                height: img.height,
                strokeHistory: [],
                baseImageURL: null,
                viewState: {
                    scale: main_default_scale(),
                    canvasX: -(DRAW_CONFIG.canvasW - DRAW_CONFIG.screenW) / 2,
                    canvasY: -(DRAW_CONFIG.canvasH - DRAW_CONFIG.screenH) / 2
                },
                sourceId: main_create_source_id('pic')
            };
            
            state.imageList.push(newImgData);
            state.currentImageIndex = state.imageList.length - 1;
            state.currentImage = img;
            state.currentFolderIndex = -1;
            state.currentFolderPageIndex = -1;
            
            main_delete_draw_canvas();
            state.strokeHistory = [];
            state.baseImageURL = null;
            state.baseImageObj = null;
            history_delete_all();
            state.scale = main_default_scale();
            state.canvasX = -(DRAW_CONFIG.canvasW - DRAW_CONFIG.screenW) / 2;
            state.canvasY = -(DRAW_CONFIG.canvasH - DRAW_CONFIG.screenH) / 2;
            main_update_move_bound();
            main_update_canvas_transform();
            main_update_history_button_status();
            
            if (isLast) {
                main_render_image_centered(img);
                main_update_sidebar_content();
                main_update_photo_button_state();
            }
        }
        
        // 如果显示了加载动画，无论文件数量多少，都需要隐藏
        if (files.length > 1 || hasLargeImage) {
            main_hide_loading_overlay();
        }
        
        console.log(`已导入 ${imageDataList.length} 张图片`);
    };
    
    input.click();
}

async function main_save_image_to_list_no_highlight(img, name, captureFilter) {
    const blob = await fetch(img.src).then(r => r.blob());
    const blobUrl = URL.createObjectURL(blob);
    
    const imgData = {
        full: blobUrl,
        thumbnail: blobUrl,
        name: name,
        width: img.width,
        height: img.height,
        sourceId: main_create_source_id('pic'),
        captureFilter: captureFilter || null
    };
    
    state.imageList.push(imgData);
    
    main_update_sidebar_content();
}

window.main_save_image_to_list_no_highlight = main_save_image_to_list_no_highlight;
window.main_update_sidebar_content = main_update_sidebar_content;
window.main_light_off = main_light_off;
window.main_delete_all_drawings = main_delete_all_drawings;

// 默认缩放：仅在「2x 满铺模式」开启时为 0.5，否则为 1x
function main_default_scale() {
    return (window.DRAW_CONFIG && window.DRAW_CONFIG.canvasFill2x) ? 0.5 : 1;
}
window.main_default_scale = main_default_scale;

function main_render_image_centered(img) {
    main_delete_image_layer();
    main_hide_no_camera_message();
    
    const fill2x = DRAW_CONFIG.canvasFill2x;
    const baseW = fill2x ? DRAW_CONFIG.canvasW : DRAW_CONFIG.screenW;
    const baseH = fill2x ? DRAW_CONFIG.canvasH : DRAW_CONFIG.screenH;

    const imgRatio = img.width / img.height;
    const baseRatio = baseW / baseH;

    let drawW, drawH, drawX, drawY;

    if (imgRatio > baseRatio) {
        drawW = baseW;
        drawH = baseW / imgRatio;
    } else {
        drawH = baseH;
        drawW = baseH * imgRatio;
    }

    const canvasW = DRAW_CONFIG.canvasW;
    const canvasH = DRAW_CONFIG.canvasH;

    drawX = (canvasW - drawW) / 2;
    drawY = (canvasH - drawH) / 2;
    
    dom.imageElement.src = img.src;
    dom.imageElement.style.left = drawX + 'px';
    dom.imageElement.style.top = drawY + 'px';
    dom.imageElement.style.width = drawW + 'px';
    dom.imageElement.style.height = drawH + 'px';
}

function main_delete_image_layer() {
    dom.imageElement.src = '';
    dom.imageElement.style.left = '0';
    dom.imageElement.style.top = '0';
    dom.imageElement.style.width = DRAW_CONFIG.canvasW + 'px';
    dom.imageElement.style.height = DRAW_CONFIG.canvasH + 'px';
}

function main_delete_image_blob_urls() {
    state.imageList.forEach(imgData => {
        if (imgData.full && imgData.full.startsWith('blob:')) {
            URL.revokeObjectURL(imgData.full);
        }
        if (imgData.thumbnail && imgData.thumbnail.startsWith('blob:') && imgData.thumbnail !== imgData.full) {
            URL.revokeObjectURL(imgData.thumbnail);
        }
    });
}

function main_delete_pdf_blob_urls(docNumber) {
    const folder = state.fileList.find(f => f.docNumber === docNumber);
    if (folder) {
        folder.pages.forEach(page => {
            if (page.full && page.full.startsWith('blob:')) {
                URL.revokeObjectURL(page.full);
            }
            if (page.thumbnail && page.thumbnail.startsWith('blob:') && page.thumbnail !== page.full) {
                URL.revokeObjectURL(page.thumbnail);
            }
        });
    }
}

function main_delete_all_pdf_blob_urls() {
    DocLoader.revoke_all_document_blob_urls();
}

window.main_setup_all_events = main_setup_all_events;
window.main_setup_pdf_file_open = main_setup_pdf_file_open;
window.main_init_camera = main_init_camera;
window.main_update_camera_state = main_update_camera_state;
window.main_init_without_camera = main_init_without_camera;
window.main_setup_deferred_camera = main_setup_deferred_camera;
window.main_show_error_dialog = main_show_error_dialog;
window.main_handle_resize = main_handle_resize;
window.main_submit_stroke = main_submit_stroke;
window.main_update_mode = main_update_mode;
window.main_update_canvas_bg_color = main_update_canvas_bg_color;
window.main_calc_rgb_to_hex = main_calc_rgb_to_hex;
window.main_update_color_buttons = main_update_color_buttons;
window.main_delete_image_blob_urls = main_delete_image_blob_urls;
window.main_delete_all_pdf_blob_urls = main_delete_all_pdf_blob_urls;
window.main_setup_minimize_listeners = main_setup_minimize_listeners;
window.main_update_move_bound = main_update_move_bound;
window.main_update_pen_style = main_update_pen_style;
window.main_update_eraser_hint_size = main_update_eraser_hint_size;
window.main_update_canvas_transform = main_update_canvas_transform;
window.main_init_pdfjs = main_init_pdfjs;
window.main_wait_pdfjs = main_wait_pdfjs;
window.main_show_pen_control_panel = main_show_pen_control_panel;
window.main_hide_pen_control_panel = main_hide_pen_control_panel;
window.main_hide_settings_panel = main_hide_settings_panel;
window.main_render_image_centered = main_render_image_centered;
window.main_render_all_strokes = main_render_all_strokes;
window.main_fetch_visible_rect = main_fetch_visible_rect;
window.main_load_pdf_from_path = main_load_pdf_from_path;
window.main_update_camera_video_style = main_update_camera_video_style;
window.main_update_settings_controls_state = main_update_settings_controls_state;
window.StrokeQuadTree = StrokeQuadTree;

/** 同步所有 overlay DPR（主界面 + 阅读器 + 黑板） */
window.sync_all_overlay_dpr = function () {
    const dpr = window.DRAW_CONFIG?.overlayDpr;
    if (dpr == null || dpr <= 0) return;
    // 主界面
    if (window.batchDrawManager) {
        window.batchDrawManager.resize_overlay(
            DRAW_CONFIG.screenW || 800,
            DRAW_CONFIG.screenH || 600
        );
    }
    // 阅读器
    const reader = window.documentReaderManager;
    if (reader?.batch_draw?._overlayCanvas) {
        const overlay = reader.batch_draw._overlayCanvas;
        reader.batch_draw._overlayDpr = dpr;
        overlay.width = Math.ceil(window.innerWidth * dpr);
        overlay.height = Math.ceil(window.innerHeight * dpr);
        overlay.style.width = window.innerWidth + 'px';
        overlay.style.height = window.innerHeight + 'px';
    }
    // 黑板
    const bb = window.blackboardManager;
    if (bb?.overlay_canvas && bb.drawing_engine?.batch_draw) {
        bb.drawing_engine.batch_draw._overlayDpr = dpr;
        bb.overlay_canvas.width = Math.ceil(bb.screen_w * dpr);
        bb.overlay_canvas.height = Math.ceil(bb.screen_h * dpr);
        bb.overlay_canvas.style.width = bb.screen_w + 'px';
        bb.overlay_canvas.style.height = bb.screen_h + 'px';
    }
};