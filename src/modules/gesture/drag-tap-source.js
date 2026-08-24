import { DeviceInputDragEvent, VirtualDeviceType } from './types.js';
import { getTolerance, TOLERANCE, detectDeviceType } from './tolerance.js';
import { DragTapConverter } from './drag-tap-converter.js';
import { InputSource } from './input-source.js';

/**
 * 拖拽/轻触输入源
 *
 * 组合 InputSource + DragTapConverter，直接输出高阶语义事件：
 * - Tap: 用户轻触（无拖拽）
 * - DragStarted / DragDelta / DragCompleted: 完整的拖拽生命周期
 *
 * 对应 DeviceInputDragTapSource
 */
export class DragTapSource {
    /**
     * @param {InputSource} inputSource - 已 attach 的 InputSource
     * @param {object} [options]
     * @param {object} [options.toleranceSet] - 容差配置，默认 TOLERANCE.DRAG
     */
    constructor(inputSource, options = {}) {
        this._input = inputSource;
        this._toleranceSet = options.toleranceSet || TOLERANCE.DRAG;
        this._converter = new DragTapConverter();
        this._deviceId = null;
        this._deviceType = null;
        this._deviceButton = null;
        this._lastInput = null;
        this._wasCanceled = false;

        /** @type {function(DeviceInputDragEvent)|null} */
        this.onTap = null;

        /** @type {function(DeviceInputDragEvent)|null} */
        this.onDragStarted = null;

        /** @type {function(DeviceInputDragEvent)|null} */
        this.onDragDelta = null;

        /** @type {function(DeviceInputDragEvent)|null} */
        this.onDragCompleted = null;

        this._converter.onTap = (ox, oy, cx, cy) => {
            if (this.onTap) {
                this.onTap(new DeviceInputDragEvent(
                    this._deviceId,
                    { x: cx, y: cy },
                    this._deviceType,
                    this._deviceButton,
                    { x: ox, y: oy },
                    { x: 0, y: 0 },
                    { x: 0, y: 0 }
                ));
            }
        };

        this._converter.onDragStarted = (ox, oy, cx, cy) => {
            if (this.onDragStarted) {
                this.onDragStarted(new DeviceInputDragEvent(
                    this._deviceId,
                    { x: cx, y: cy },
                    this._deviceType,
                    this._deviceButton,
                    { x: ox, y: oy },
                    { x: 0, y: 0 },
                    { x: 0, y: 0 }
                ));
            }
        };

        this._converter.onDragDelta = (ox, oy, cx, cy, dxo, dyo, dxl, dyl) => {
            if (this.onDragDelta) {
                this.onDragDelta(new DeviceInputDragEvent(
                    this._deviceId,
                    { x: cx, y: cy },
                    this._deviceType,
                    this._deviceButton,
                    { x: ox, y: oy },
                    { x: dxo, y: dyo },
                    { x: dxl, y: dyl }
                ));
            }
        };

        this._converter.onDragCompleted = (ox, oy, cx, cy, dxo, dyo, dxl, dyl) => {
            if (this.onDragCompleted) {
                const virtualType = this._wasCanceled
                    ? VirtualDeviceType.LostCapture
                    : VirtualDeviceType.Device;
                const ev = new DeviceInputDragEvent(
                    this._deviceId,
                    { x: cx, y: cy },
                    this._deviceType,
                    this._deviceButton,
                    { x: ox, y: oy },
                    { x: dxo, y: dyo },
                    { x: dxl, y: dyl }
                );
                ev.virtualType = virtualType;
                this.onDragCompleted(ev);
            }
        };

        this._onInputStarting = (ev) => {
            this._wasCanceled = false;
        };

        this._onInputStarted = (ev) => {
            this._deviceId = ev.id;
            this._deviceType = ev.type;
            this._deviceButton = ev.button;
        };

        this._onInputDown = (ev) => {
            if (this._deviceId !== null && this._deviceId !== ev.id) return;

            this._deviceId = ev.id;
            this._deviceType = ev.type;
            this._deviceButton = ev.button;
            this._lastInput = ev;

            const tolerance = getTolerance(this._toleranceSet, ev.type);
            this._converter.down(ev.position.x, ev.position.y, tolerance);
        };

        this._onInputMove = (ev) => {
            if (this._deviceId !== ev.id) return;
            this._lastInput = ev;
            this._converter.move(ev.position.x, ev.position.y);
        };

        this._onInputUp = (ev) => {
            if (this._deviceId !== ev.id) return;

            if (ev.virtualType === VirtualDeviceType.LostCapture) {
                this._wasCanceled = true;
            }

            this._lastInput = ev;
            this._converter.up(ev.position.x, ev.position.y);
        };

        this._onInputCompleted = (ev) => {
            if (this._converter.isDragging && ev.virtualType === VirtualDeviceType.LostCapture) {
                this._wasCanceled = true;
                const lastPos = this._lastInput ? this._lastInput.position : { x: 0, y: 0 };
                this._converter.up(lastPos.x, lastPos.y);
            }
            this._deviceId = null;
        };

        inputSource.on('inputStarting', this._onInputStarting);
        inputSource.on('inputStarted', this._onInputStarted);
        inputSource.on('inputDown', this._onInputDown);
        inputSource.on('inputMove', this._onInputMove);
        inputSource.on('inputUp', this._onInputUp);
        inputSource.on('inputCompleted', this._onInputCompleted);
    }

    destroy() {
        this._input.off('inputStarting', this._onInputStarting);
        this._input.off('inputStarted', this._onInputStarted);
        this._input.off('inputDown', this._onInputDown);
        this._input.off('inputMove', this._onInputMove);
        this._input.off('inputUp', this._onInputUp);
        this._input.off('inputCompleted', this._onInputCompleted);
    }

    /** 是否正在拖拽中 */
    get isDragging() {
        return this._converter.isDragging;
    }

    /**
     * 取消当前拖拽跟踪（不触发 onDragCompleted/onTap）。
     * 两指手势（pinch）开始时调用，清除首指残留的拖拽状态，
     * 防止其后续 move 事件用 pinch 前的过期抓取偏移覆写画布位置
     */
    cancelDrag() {
        this._converter.reset();
    }
}
