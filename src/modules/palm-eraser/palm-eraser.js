export const PALM_CONFIG = {
    penThreshold: 5,
    fingerThreshold: 15,
    palmMultiplier: 2.5,
    palmSizeMultiplier: 1.2,
    eraserSizeK: 1.0,
    fallbackTouchCount: 4,
    fallbackSpread: 300,
    candidateDelay: 150,
    palmEraserSize: 60,
};

export function is_palm_by_pointer(e) {
    if (typeof e.width !== 'number' || e.width <= 0 || typeof e.height !== 'number' || e.height <= 0) {
        return { isPalm: false, width: 0, height: 0 };
    }
    const w = e.width;
    const h = e.height;
    const threshold = PALM_CONFIG.fingerThreshold;
    return {
        isPalm: w > threshold * PALM_CONFIG.palmMultiplier,
        width: w,
        height: h
    };
}

/** 根据 PointerEvent 触点宽高计算手掌擦除大小（取 max 以覆盖手掌接触面），首次检测后固定 */
export function compute_palm_eraser_size_from_pointer(width, height) {
    const dim = Math.max(width, height);
    const size = dim * PALM_CONFIG.palmSizeMultiplier * PALM_CONFIG.eraserSizeK;
    return Math.max(40, Math.min(150, size));
}

export function is_palm_by_touch_count(touches) {
    if (touches.length < PALM_CONFIG.fallbackTouchCount) return false;
    return is_palm_by_positions(touches, 'clientX', 'clientY', PALM_CONFIG.fallbackSpread);
}

export function get_palm_center(touches) {
    let cx = 0, cy = 0;
    for (const t of touches) {
        cx += t.clientX;
        cy += t.clientY;
    }
    return { x: cx / touches.length, y: cy / touches.length };
}

export function is_palm_by_positions(positions, keyX = 'x', keyY = 'y', spreadThreshold = PALM_CONFIG.fallbackSpread) {
    if (positions.length < PALM_CONFIG.fallbackTouchCount) return false;
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const p of positions) {
        const px = typeof p[keyX] === 'number' ? p[keyX] : p.x;
        const py = typeof p[keyY] === 'number' ? p[keyY] : p.y;
        if (px < minX) minX = px;
        if (px > maxX) maxX = px;
        if (py < minY) minY = py;
        if (py > maxY) maxY = py;
    }
    return Math.max(maxX - minX, maxY - minY) < spreadThreshold;
}

export function get_palm_center_from_positions(positions, keyX = 'x', keyY = 'y') {
    let cx = 0, cy = 0;
    for (const p of positions) {
        cx += typeof p[keyX] === 'number' ? p[keyX] : p.x;
        cy += typeof p[keyY] === 'number' ? p[keyY] : p.y;
    }
    return { x: cx / positions.length, y: cy / positions.length };
}

/**
 * PalmEraserSession — 手掌擦除会话。
 *
 * 职责：手掌检测触发 + 计算擦除大小，擦除逻辑完全由宿主的橡皮擦路径处理。
 *
 * host 接口：
 *   getCanvasRect()           → { left, top }
 *   getScale()                → number（canvas 坐标缩放）
 *   defaultEraserSize?        → number（默认手掌擦除大小）
 *   showHint?()
 *   updateHint?(clientX, clientY, size)
 *   hideHint?()
 *   saveStrokePoint(fromX, fromY, toX, toY, pressure?)  → 宿主橡皮擦逻辑（bounds / variableWidths / points / batch_draw）
 *   submitStroke()            → 提交笔画
 *   onSessionStart?(stroke, session)
 *   onSessionEnd?()
 */
export class PalmEraserSession {
    constructor(host) {
        this.host = host;
        this.isErasing = false;
        this.lastX = 0;
        this.lastY = 0;
        this.palmEraserSize = 60;
    }

    start(clientX, clientY, eraserWidth) {
        if (this.isErasing) return null;
        const h = this.host;
        this.isErasing = true;
        this.palmEraserSize = eraserWidth || h.defaultEraserSize || 60;

        const rect = h.getCanvasRect();
        const scale = h.getScale();
        const inv = 1 / Math.max(0.001, scale);
        this.lastX = (clientX - rect.left) * inv;
        this.lastY = (clientY - rect.top) * inv;

        const baseSize = this.palmEraserSize * inv;
        const stroke = {
            type: 'erase',
            points: [],
            color: '#000000',
            lineWidth: baseSize,
            eraserSize: baseSize,
            eraserSizeRaw: this.palmEraserSize,
            eraserShape: 'square',
            eraserSpeedEnabled: false,
            scale: scale || 1,
            bounds: { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity },
            variableWidths: []
        };

        h.showHint?.();
        h.updateHint?.(clientX, clientY, this.palmEraserSize);
        h.onSessionStart?.(stroke, this);

        return stroke;
    }

    update(clientX, clientY) {
        if (!this.isErasing) return;
        const h = this.host;
        const rect = h.getCanvasRect();
        const inv = 1 / Math.max(0.001, h.getScale());
        const x = (clientX - rect.left) * inv;
        const y = (clientY - rect.top) * inv;
        const dx = x - this.lastX;
        const dy = y - this.lastY;

        h.updateHint?.(clientX, clientY, this.palmEraserSize);

        if (dx !== 0 || dy !== 0) {
            h.saveStrokePoint(this.lastX, this.lastY, x, y, 0.5);
            this.lastX = x;
            this.lastY = y;
        }
    }

    async end() {
        if (!this.isErasing) return;
        this.isErasing = false;
        const h = this.host;
        try {
            h.hideHint?.();
            await h.submitStroke();
        } catch (e) {
            console.error('[PalmEraser] submitStroke failed:', e);
        } finally {
            h.onSessionEnd?.();
        }
    }
}
