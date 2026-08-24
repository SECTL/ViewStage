/**
 * 拖拽/轻触状态机
 *
 * 输入 down/move/up 事件流，通过距离容差自动区分 tap 和 drag：
 * - 移动距离 < tolerance → 触发 tap
 * - 移动距离 >= tolerance → 触发 dragStarted → dragDelta* → dragCompleted
 */
export class DragTapConverter {
    constructor() {
        this._ox = 0;
        this._oy = 0;
        this._lx = 0;
        this._ly = 0;
        this._tolerance = 0;
        this._dragging = false;

        /** @type {function(number, number, number, number)|null} tap(originX, originY, currentX, currentY) */
        this.onTap = null;

        /** @type {function(number, number, number, number, number, number, number, number)|null} dragStarted(ox, oy, cx, cy, dxo, dyo, dxl, dyl) */
        this.onDragStarted = null;

        /** @type {function(number, number, number, number, number, number, number, number)|null} dragDelta(ox, oy, cx, cy, dxo, dyo, dxl, dyl) */
        this.onDragDelta = null;

        /** @type {function(number, number, number, number, number, number, number, number)|null} dragCompleted(ox, oy, cx, cy, dxo, dyo, dxl, dyl) */
        this.onDragCompleted = null;
    }

    /**
     * 按下事件
     * @param {number} x
     * @param {number} y
     * @param {number} tolerance - 该设备的容差阈值（px），低于此值视为 tap
     */
    down(x, y, tolerance) {
        this._tolerance = tolerance;
        this._ox = x;
        this._oy = y;
        this._lx = x;
        this._ly = y;
        this._dragging = false;
    }

    /**
     * 移动事件
     * @param {number} x
     * @param {number} y
     */
    move(x, y) {
        const ddx = x - this._lx;
        const ddy = y - this._ly;
        this._lx = x;
        this._ly = y;

        if (!this._dragging) {
            const dxo = x - this._ox;
            const dyo = y - this._oy;
            const dist = Math.sqrt(dxo * dxo + dyo * dyo);
            if (dist >= this._tolerance) {
                this._dragging = true;
                if (this.onDragStarted) {
                    this.onDragStarted(this._ox, this._oy, x, y, 0, 0, 0, 0);
                }
                if (this.onDragDelta) {
                    this.onDragDelta(this._ox, this._oy, x, y, dxo, dyo, ddx, ddy);
                }
            }
        } else {
            const dxo = x - this._ox;
            const dyo = y - this._oy;
            if (this.onDragDelta) {
                this.onDragDelta(this._ox, this._oy, x, y, dxo, dyo, ddx, ddy);
            }
        }
    }

    /**
     * 抬起事件
     * @param {number} x
     * @param {number} y
     */
    up(x, y) {
        const dxo = x - this._ox;
        const dyo = y - this._oy;
        const ddx = x - this._lx;
        const ddy = y - this._ly;
        this._lx = x;
        this._ly = y;

        if (!this._dragging) {
            if (this.onTap) {
                this.onTap(this._ox, this._oy, x, y);
            }
        } else {
            if (this.onDragCompleted) {
                this.onDragCompleted(this._ox, this._oy, x, y, dxo, dyo, ddx, ddy);
            }
        }

        this._dragging = false;
    }

    /** 是否正在拖拽中 */
    get isDragging() {
        return this._dragging;
    }

    /**
     * 强制清除拖拽状态（不触发任何回调）。
     * 用于两指手势（pinch）开始时接管输入，防止残留的拖拽状态
     * 在手势结束后用过期的起点坐标覆写位置
     */
    reset() {
        this._dragging = false;
    }
}
