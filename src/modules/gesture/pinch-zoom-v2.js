import { DeviceType, VirtualDeviceType, DeviceInputEvent, DeviceInputStartingEvent, DeviceInputStartedEvent, DeviceInputCompletedEvent } from './types.js';
import { getTolerance, TOLERANCE, detectDeviceType } from './tolerance.js';

/**
 * 两指捏合识别器 V2 — 增量式缩放 + 中点锚点
 *
 * 与 V1 的区别：
 * - ev.scale 为增量比（每帧相对于上一帧的距离比），非累积比
 * - 无缩放死区：到达边界后反向操作立即生效，无需 resetScaleReference
 * - centerX/Y 始终为两指中点（V1 由消费者自行用 finger0 计算锚点）
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

        this._movedThisBatch = [];

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
        this._beyondTolerance = false;
        this._movedThisBatch = [];

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
            if (Math.abs(f0.position.x - this._pendingStartPos0.x) > tol ||
                Math.abs(f0.position.y - this._pendingStartPos0.y) > tol ||
                Math.abs(f1.position.x - this._pendingStartPos1.x) > tol ||
                Math.abs(f1.position.y - this._pendingStartPos1.y) > tol) {
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

        const events = this._input.activeEvents;
        let f0Ev = null, f1Ev = null;
        for (let i = 0; i < events.length; i++) {
            const e = events[i];
            if (e.id === this._pinchIds[0]) f0Ev = e;
            if (e.id === this._pinchIds[1]) f1Ev = e;
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

        if (this._prevDistance === 0) return;

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

        // V2 核心：增量式缩放，每帧相对于上一帧
        const incrementalRatio = currentDist / this._prevDistance;

        if (this._movedThisBatch.indexOf(ev.id) === -1) {
            this._movedThisBatch.push(ev.id);
        }
        if (!this._pinchIds.every(id => this._movedThisBatch.indexOf(id) !== -1)) return;
        this._movedThisBatch = [];

        // batch 检查通过后才更新 prevDistance，避免单指先更新时破坏参考值
        this._prevDistance = currentDist;

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
        this._movedThisBatch = [];

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
