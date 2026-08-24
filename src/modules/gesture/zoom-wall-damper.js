/**
 * 缩放边界阻尼器 —— 消除缩放到 min/max 边界后的"呼吸"抖动
 *
 * 问题：V2 增量式缩放的候选值 = 当前缩放 × 单帧比值，钉在边界时
 * 触控噪声（±0.5%~2%）使候选值反复跨越边界：
 *   噪声向下 → 未钳制值 < 边界 → 立即脱离边界
 *   噪声向上 → 重新钳回边界
 * 表现为缩放值在边界两侧逐帧弹跳，内容可见地"呼吸"。
 *
 * 机制：
 * - 内部维护累计缩放 _acc，贴墙期间持续复利累积手指真实意图，
 *   并钳制在有界窗口 [min×(1-esc), max×(1+esc)] 内（防止深度过冲后反向出现巨大死区）；
 * - 显示值钉死在边界，直到累计值以超过 escape 余量明确越过边界才脱离；
 * - 自由区间内 _acc 与显示值保持同步，脱离/再贴墙无缝，无额外死区。
 */
export class ZoomWallDamper {
    /**
     * @param {number} [escapeRatio=0.02] - 脱离边界所需的累计越界余量（相对边界值的比例）
     */
    constructor(escapeRatio = 0.02) {
        this.escape = escapeRatio;
        this._min = null;
        this._max = null;
        this._acc = 1;
        this._value = 1;
    }

    /**
     * 开始/重置一次捏合会话
     * @param {number} value - 当前缩放值
     * @param {number} min - 最小缩放限制
     * @param {number} max - 最大缩放限制
     */
    reset(value, min, max) {
        this._min = min;
        this._max = max;
        this._value = Math.max(min, Math.min(max, value));
        this._acc = this._value;
        return this._value;
    }

    /**
     * 喂入一帧增量比，返回应用边界阻尼后的缩放值
     * @param {number} ratio - 相对上一帧的增量比
     * @returns {number}
     */
    update(ratio) {
        if (this._min == null || this._max == null) return this._value;
        if (!Number.isFinite(ratio) || ratio <= 0) return this._value;

        const min = this._min;
        const max = this._max;
        const eps = 1e-9;

        this._acc = this._acc * ratio;

        // 累计窗口钳制：贴墙期间 _acc 无界复利会造成巨大脱离死区
        // （如过冲到 acc=max×2 后反向，需要数十个百分点行程才响应）。
        // 钳制后最坏死区收敛为约 ±2×escape 的手指行程
        const overCap = max * (1 + this.escape);
        const underCap = min * (1 - this.escape);
        if (this._acc > overCap) {
            this._acc = overCap;
        } else if (this._acc < underCap) {
            this._acc = underCap;
        }

        let next = this._acc;
        if (this._value >= max - eps) {
            // 钉在上边界：累计值回落不足 escape 余量则保持钉住
            if (next >= max * (1 - this.escape)) next = max;
        } else if (this._value <= min + eps) {
            // 钉在下边界：累计值抬升不足 escape 余量则保持钉住
            if (next <= min * (1 + this.escape)) next = min;
        }

        this._value = Math.max(min, Math.min(max, next));

        // 自由区间内同步累计值：保证脱离边界与再次贴墙时无跳变
        if (this._value > min + eps && this._value < max - eps) {
            this._acc = this._value;
        }
        return this._value;
    }
}
