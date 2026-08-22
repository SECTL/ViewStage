import { DeviceType, VirtualDeviceType, DeviceInputEvent, DeviceInputStartingEvent, DeviceInputStartedEvent, DeviceInputCompletedEvent } from './types.js';
import { getTolerance, TOLERANCE, detectDeviceType, PINCH_MIN_DISTANCE, PINCH_FRAME_RATIO_MAX, PINCH_FRAME_RATIO_MIN } from './tolerance.js';

/** 微动死区：比值变化低于该阈值视为触控噪声 */
const EMIT_DEAD_BAND_RATIO = 0.001;
/** 微动死区：中点位移平方低于该阈值视为触控噪声（0.75px） */
const EMIT_DEAD_BAND_MID_SQ = 0.75 * 0.75;

/**
 * 两指捏合识别器 V2 — 增量式缩放 + 中点锚点 + 帧对齐发射
 *
 * 与 V1 的区别：
 * - ev.scale 为增量比（每帧相对于上一帧的距离比），非累积比
 * - 无缩放死区：到达边界后反向操作立即生效，无需 resetScaleReference
 * - centerX/Y 始终为两指中点（V1 由消费者自行用 finger0 计算锚点）
 *
 * 发射模型：inputMove 仅做调度，实际计算在 requestAnimationFrame 回调中以
 * 两指最新位置执行 —— 同一显示帧内的多个事件合并为一次 delta：
 * - 配对始终为帧内最新值，消除逐事件发射的新旧配对交替抖动
 * - 单指静止（锚指+滑动指手势）时直接复用其存储位置，不会冻结
 * - 发射频率与屏幕刷新对齐，消费者 DOM 写入天然合并
 */
export class PinchZoomSourceV2 {
    /**
     * @param {InputSource} inputSource - 已绑定的 InputSource 实例
     * @param {object} [options]
     * @param {number} [options.minScale=0.1] - 最小缩放限制
     * @param {number} [options.maxScale=10] - 最大缩放限制
     * @param {number} [options.toleranceSet] - 容差配置，默认 TOLERANCE.PINCH
     */
    constructor(inputSource, options = {}) {
        this._input = inputSource;
        this._minScale = options.minScale ?? 0.1;
        this._maxScale = options.maxScale ?? 10;
        this._toleranceSet = options.toleranceSet || TOLERANCE.PINCH;

        this._isPinching = false;
        this._pinchIds = [];
        this._prevDistance = 0;
        this._initialDistance = 0;
        this._startMidX = 0;
        this._startMidY = 0;
        this._toleranceSq = 0;
        this._beyondTolerance = false;

        this._startDelayMs = 0;
        this._firstFingerTime = 0;

        this._isPending = false;
        this._pendingPinchIds = [];
        this._pendingStartPos0 = { x: 0, y: 0 };
        this._pendingStartPos1 = { x: 0, y: 0 };

        // 帧对齐发射状态
        this._emitRafId = null;
        this._lastEmitMidX = 0;
        this._lastEmitMidY = 0;

        this._finger0 = { x: 0, y: 0 };
        this._finger1 = { x: 0, y: 0 };
        this._deltaPayload = {
            scale: 1, centerX: 0, centerY: 0,
            originScale: 1, deltaScale: 0,
            startMidX: 0, startMidY: 0,
            finger0: this._finger0, finger1: this._finger1,
        };

        this.onPinchStarted = null;
        this.onPinchDelta = null;
        this.onPinchCompleted = null;

        this._onInputDown = this._onInputDown.bind(this);
        this._onInputMove = this._onInputMove.bind(this);
        this._onInputUp = this._onInputUp.bind(this);

        inputSource.on('inputDown', this._onInputDown);
        inputSource.on('inputMove', this._onInputMove);
        inputSource.on('inputUp', this._onInputUp);
    }

    get isPinching() {
        return this._isPinching;
    }

    cancelPinch() {
        if (this._isPinching) {
            this._finishPinch(VirtualDeviceType.LostCapture);
        }
    }

    get startDelayMs() {
        return this._startDelayMs;
    }
    set startDelayMs(v) {
        this._startDelayMs = Math.max(0, v);
    }

    destroy() {
        this._input.off('inputDown', this._onInputDown);
        this._input.off('inputMove', this._onInputMove);
        this._input.off('inputUp', this._onInputUp);
        this._cancel_pinch_emit();
        this._isPinching = false;
        this._isPending = false;
        this._pendingPinchIds = [];
    }

    _onInputDown(ev) {
        if (this._isPinching || this._isPending) return;

        const count = this._input.activeCount;

        if (count === 1) {
            this._firstFingerTime = performance.now();
            return;
        }

        if (count < 2) return;

        const events = this._input.activeEvents;
        if (events.length < 2) return;

        if (this._startDelayMs > 0) {
            const elapsed = performance.now() - this._firstFingerTime;
            if (elapsed > this._startDelayMs) {
                this._isPending = true;
                this._pendingPinchIds = [events[0].id, events[1].id];
                this._pendingStartPos0.x = events[0].position.x;
                this._pendingStartPos0.y = events[0].position.y;
                this._pendingStartPos1.x = events[1].position.x;
                this._pendingStartPos1.y = events[1].position.y;
                return;
            }
        }

        this._pinchIds = [events[0].id, events[1].id];
        this._startPinch(events[0].position, events[1].position);
    }

    _startPinch(pos0, pos1) {
        const dx = pos0.x - pos1.x;
        const dy = pos0.y - pos1.y;
        this._initialDistance = Math.sqrt(dx * dx + dy * dy);
        this._prevDistance = this._initialDistance;
        this._startMidX = (pos0.x + pos1.x) / 2;
        this._startMidY = (pos0.y + pos1.y) / 2;
        this._lastEmitMidX = this._startMidX;
        this._lastEmitMidY = this._startMidY;
        this._beyondTolerance = false;

        const tol = getTolerance(this._toleranceSet, DeviceType.Touch);
        this._toleranceSq = tol * tol;

        this._isPinching = true;

        if (this.onPinchStarted) {
            this.onPinchStarted({
                scale: 1,
                centerX: this._startMidX,
                centerY: this._startMidY,
                originScale: 1,
                deltaScale: 0,
                finger0: { x: pos0.x, y: pos0.y },
                finger1: { x: pos1.x, y: pos1.y },
            });
        }
    }

    _onInputMove(ev) {
        if (this._isPending) {
            const events = this._input.activeEvents;
            let f0 = null, f1 = null;
            for (let i = 0; i < events.length; i++) {
                const e = events[i];
                if (e.id === this._pendingPinchIds[0]) f0 = e;
                if (e.id === this._pendingPinchIds[1]) f1 = e;
            }
            if (!f0 || !f1) {
                this._isPending = false;
                this._pendingPinchIds = [];
                return;
            }

            const tol = getTolerance(this._toleranceSet, DeviceType.Touch);
            // BOTH 语义：两指都超过容差才激活缩放。
            // 防止批注模式下一指书写、另一指搭扶时，书写指的移动误触发缩放切断笔画
            const f0Moved = Math.abs(f0.position.x - this._pendingStartPos0.x) > tol ||
                Math.abs(f0.position.y - this._pendingStartPos0.y) > tol;
            const f1Moved = Math.abs(f1.position.x - this._pendingStartPos1.x) > tol ||
                Math.abs(f1.position.y - this._pendingStartPos1.y) > tol;
            if (f0Moved && f1Moved) {
                this._isPending = false;
                this._pendingPinchIds = [];
                this._pinchIds = [f0.id, f1.id];
                this._startPinch(f0.position, f1.position);
            }
            return;
        }

        if (!this._isPinching) return;
        if (this._input.activeCount < 2) {
            this._finishPinch(VirtualDeviceType.Device);
            return;
        }

        // 帧对齐发射：本帧稍后用两指最新位置统一计算
        this._schedule_pinch_emit();
    }

    _cancel_pinch_emit() {
        if (this._emitRafId !== null) {
            cancelAnimationFrame(this._emitRafId);
            this._emitRafId = null;
        }
    }

    _schedule_pinch_emit() {
        if (this._emitRafId !== null) return;
        this._emitRafId = requestAnimationFrame(() => this._emit_pinch_frame());
    }

    _emit_pinch_frame() {
        this._emitRafId = null;
        if (!this._isPinching) return;
        if (this._input.activeCount < 2) return;

        const events = this._input.activeEvents;
        let f0Ev = null, f1Ev = null;
        for (let i = 0; i < events.length; i++) {
            const e = events[i];
            if (e.id === this._pinchIds[0]) f0Ev = e;
            else if (e.id === this._pinchIds[1]) f1Ev = e;
        }
        if (!f0Ev || !f1Ev) {
            this._finishPinch(VirtualDeviceType.Device);
            return;
        }

        const dx = f0Ev.position.x - f1Ev.position.x;
        const dy = f0Ev.position.y - f1Ev.position.y;
        const currentDist = Math.sqrt(dx * dx + dy * dy);
        const midX = (f0Ev.position.x + f1Ev.position.x) / 2;
        const midY = (f0Ev.position.y + f1Ev.position.y) / 2;

        // 最小距离冻结：两指几乎并拢/交叉时距离不可信（微小抖动即产生巨大增量比），
        // 冻结期间不发射 delta、不更新参考距离，跨越该区间后自动恢复链式一致。
        // 注：prevDistance===0（起始即并拢）无需特判——下方激活分支会以当前距离
        // 重建参考值实现自愈；比值分母的 Math.max(prev, MIN) 已保证无除零
        if (currentDist < PINCH_MIN_DISTANCE) return;

        if (!this._beyondTolerance) {
            // 用初始距离和初始中点判断是否超过容差
            const distFromInitial = Math.abs(currentDist - this._initialDistance);
            const midDxFromStart = midX - this._startMidX;
            const midDyFromStart = midY - this._startMidY;
            const midMoveSq = midDxFromStart * midDxFromStart + midDyFromStart * midDyFromStart;
            if (distFromInitial < getTolerance(this._toleranceSet, DeviceType.Touch) &&
                midMoveSq < this._toleranceSq) {
                return;
            }
            this._beyondTolerance = true;
            // 首次超过容差，将 prevDistance 设为当前距离，后续帧用增量比
            this._prevDistance = currentDist;
        }

        // V2 核心：增量式缩放，每帧相对于上一帧。
        // 参考距离钳制下限：起始即并拢时（prevDistance < MIN）防止除以极小值；
        // 单帧比值夹取上下限，防御指针事件丢失/跳变导致的异常大步长
        let incrementalRatio = currentDist / Math.max(this._prevDistance, PINCH_MIN_DISTANCE);
        if (incrementalRatio > PINCH_FRAME_RATIO_MAX) incrementalRatio = PINCH_FRAME_RATIO_MAX;
        else if (incrementalRatio < PINCH_FRAME_RATIO_MIN) incrementalRatio = PINCH_FRAME_RATIO_MIN;

        // 微动死区：比值变化与中点位移同时低于阈值视为触控噪声，跳过本次发射。
        // 不更新 prevDistance —— 真实慢速缩放无损累积，往复噪声相互抵消
        const emitMidDx = midX - this._lastEmitMidX;
        const emitMidDy = midY - this._lastEmitMidY;
        if (Math.abs(incrementalRatio - 1) < EMIT_DEAD_BAND_RATIO &&
            emitMidDx * emitMidDx + emitMidDy * emitMidDy < EMIT_DEAD_BAND_MID_SQ) {
            return;
        }

        // 更新参考距离。同样钳制下限，保证下一帧分母不会是极小值
        this._prevDistance = Math.max(currentDist, PINCH_MIN_DISTANCE);
        this._lastEmitMidX = midX;
        this._lastEmitMidY = midY;

        if (this.onPinchDelta) {
            this._finger0.x = f0Ev.position.x;
            this._finger0.y = f0Ev.position.y;
            this._finger1.x = f1Ev.position.x;
            this._finger1.y = f1Ev.position.y;
            const p = this._deltaPayload;
            p.scale = incrementalRatio;
            p.centerX = midX;
            p.centerY = midY;
            p.originScale = incrementalRatio;
            p.deltaScale = incrementalRatio - 1.0;
            p.startMidX = this._startMidX;
            p.startMidY = this._startMidY;
            this.onPinchDelta(p);
        }
    }

    _onInputUp(ev) {
        if (this._isPending && this._pendingPinchIds.indexOf(ev.id) !== -1) {
            this._isPending = false;
            this._pendingPinchIds = [];
            return;
        }

        if (!this._isPinching && this._input.activeCount === 1) {
            this._firstFingerTime = performance.now();
        }

        if (!this._isPinching) return;

        if (this._pinchIds.indexOf(ev.id) !== -1) {
            this._finishPinch(VirtualDeviceType.Device);
            if (this._input.activeCount >= 2) {
                const events = this._input.activeEvents;
                if (events.length >= 2) {
                    this._pinchIds = [events[0].id, events[1].id];
                    this._startPinch(events[0].position, events[1].position);
                }
            }
        }
    }

    _finishPinch(virtualType) {
        if (!this._isPinching) return;
        this._isPinching = false;
        this._pinchIds = [];
        this._cancel_pinch_emit();

        if (this.onPinchCompleted) {
            this.onPinchCompleted({
                scale: 1,
                centerX: this._startMidX,
                centerY: this._startMidY,
                originScale: 1,
                virtualType: virtualType,
            });
        }
    }
}
