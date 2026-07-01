class RealtimeBatchDrawManager {
    constructor() {
        this.pendingCommands = [];
        this.pendingCount = 0;
        this.drawRafId = null;
        this.drawInterval = 1000 / 60;
        this.lastDrawTime = 0;
        this.lastType = null;
        this.lastColor = null;
        this.lastLineWidth = null;

        this.currentFps = 60;
        this.minFps = 15;
        this.maxFps = 60;
        this.fpsStep = 5;

        this.drawTimes = [];
        this.drawTimesMax = 10;

        this.commandCounts = [];
        this.commandCountsMax = 5;

        this.frameRateMode = 'adaptive';
        this.lastAdjustTime = 0;
        this.adjustCooldown = 100;

        this.LOW_LOAD_FPS = 60;
        this.MEDIUM_LOAD_FPS = 45;
        this.HIGH_LOAD_FPS = 30;
        this.CRITICAL_LOAD_FPS = 20;

        this.LOW_LOAD_THRESHOLD = 10;
        this.MEDIUM_LOAD_THRESHOLD = 30;
        this.HIGH_LOAD_THRESHOLD = 50;

        this.lastBatchMoveTime = 0;

        this._strokeStart = true;
        this._totalSegments = 0;
        this._lastMidX = null;
        this._lastMidY = null;
        this._lastToX = null;
        this._lastToY = null;
        this._speedBuffer = [];
        this._storedWidths = [];
        this._segmentTimes = [];
        this._penEffectMode = 'off';
        this._dirtyBoundsCanvas = null;
        this._limitedTailWidth = null;

        this._overlayCanvas = null;
        this._overlayCtx = null;
        this._overlayTransformScale = 0;
        this._overlayTransformX = 0;
        this._overlayTransformY = 0;
        this._overlayDpr = 1;
        this._overlayDprSettleTimerId = null;
        this._overlayDprSettleMs = 300;

        this._tileRenderer = null;
        this.eraserShape = 'square';
        this.ellipseMode = false;
    }

    /**
     * 计算覆盖层 DPR。
     * 覆盖层是屏幕空间画布（position: fixed 覆盖视口），DPR 超过 devicePixelRatio
     * 对显示无增益，仅浪费显存。因此以 display_dpr 作为硬上限。
     * @param {number} scale - 当前画布缩放比例
     * @returns {number} 覆盖层 DPR
     */
    _calc_overlay_dpr(scale) {
        const cfg = window.DRAW_CONFIG;
        if (cfg.overlayDpr != null && cfg.overlayDpr > 0) return cfg.overlayDpr;
        if (cfg.dynamicDprEnabled === false) return Math.min(cfg.dpr, 2);
        return 1;
    }

    update_overlay_dpr(scale, force) {
        if (this._overlayDprSettleTimerId != null) {
            clearTimeout(this._overlayDprSettleTimerId);
            this._overlayDprSettleTimerId = null;
        }
        this._overlayDprSettleTimerId = setTimeout(() => {
            this._overlayDprSettleTimerId = null;
            const targetDpr = this._calc_overlay_dpr(scale);
            if (targetDpr !== this._overlayDpr || force) {
                this._apply_overlay_dpr(targetDpr);
            }
        }, this._overlayDprSettleMs);
    }

    _apply_overlay_dpr(newDpr) {
        if (!this._overlayCanvas) return;
        const screenW = window.DRAW_CONFIG?.screenW || 1;
        const screenH = window.DRAW_CONFIG?.screenH || 1;
        this._overlayDpr = newDpr;
        this._overlayCanvas.width = Math.ceil(Math.max(1, screenW * newDpr));
        this._overlayCanvas.height = Math.ceil(Math.max(1, screenH * newDpr));
        this._overlayTransformScale = 0;
        this._overlayTransformX = 0;
        this._overlayTransformY = 0;
    }

    init_overlay(container, screenW, screenH, dpr) {
        this._overlayCanvas = document.createElement('canvas');
        this._overlayCanvas.className = 'canvas-tile draw-overlay';
        this._overlayDpr = this._calc_overlay_dpr(window.state.scale || 1);
        this._overlayCanvas.width = Math.ceil(screenW * this._overlayDpr);
        this._overlayCanvas.height = Math.ceil(screenH * this._overlayDpr);
        this._overlayCanvas.style.width = screenW + 'px';
        this._overlayCanvas.style.height = screenH + 'px';
        container.appendChild(this._overlayCanvas);
        this._overlayCtx = this._overlayCanvas.getContext('2d', { willReadFrequently: false });
        if (!this._overlayCtx) {
            console.error('batch-draw: 无法获取 overlay canvas 上下文');
            return;
        }
        this._overlayCtx.imageSmoothingEnabled = false;
        this._overlayTransformScale = 0;
        this._overlayTransformX = 0;
        this._overlayTransformY = 0;
    }

    resize_overlay(screenW, screenH, dpr) {
        if (this._overlayCanvas) {
        this._overlayDpr = this._calc_overlay_dpr(window.state.scale || 1);
            this._overlayCanvas.width = Math.ceil(screenW * this._overlayDpr);
            this._overlayCanvas.height = Math.ceil(screenH * this._overlayDpr);
            this._overlayCanvas.style.width = screenW + 'px';
            this._overlayCanvas.style.height = screenH + 'px';
        }
        this._overlayTransformScale = 0;
        this._overlayTransformX = 0;
        this._overlayTransformY = 0;
    }

    destroy_overlay() {
        if (this._overlayCanvas && this._overlayCanvas.parentNode) {
            this._overlayCanvas.parentNode.removeChild(this._overlayCanvas);
        }
        this._overlayCanvas = null;
        this._overlayCtx = null;
    }

    /** 缩放期间隐藏 overlay，释放 GPU 合成层 */
    hide_overlay() {
        if (this._overlayCanvas) this._overlayCanvas.style.visibility = 'hidden';
    }

    /** 缩放结束后恢复 overlay */
    show_overlay() {
        if (this._overlayCanvas) this._overlayCanvas.style.visibility = '';
    }

    _sync_overlay_transform() {
        if (!this._overlayCtx) return;
        const dpr = this._overlayDpr;
        const scale = window.state.scale || 1;
        const canvasX = window.state.canvasX || 0;
        const canvasY = window.state.canvasY || 0;
        if (this._overlayTransformScale === scale &&
            this._overlayTransformX === canvasX &&
            this._overlayTransformY === canvasY) {
            return;
        }
        this._overlayTransformScale = scale;
        this._overlayTransformX = canvasX;
        this._overlayTransformY = canvasY;
        this._overlayCtx.setTransform(scale * dpr, 0, 0, scale * dpr, canvasX * dpr, canvasY * dpr);
    }

    clear_overlay() {
        if (!this._overlayCtx) return;
        this._overlayCtx.setTransform(1, 0, 0, 1, 0, 0);

        if (this._dirtyBoundsCanvas) {
            const s = window.state.scale || 1;
            const dpr = this._overlayDpr;
            const ox = (window.state.canvasX || 0) * dpr;
            const oy = (window.state.canvasY || 0) * dpr;
            const x = Math.floor(this._dirtyBoundsCanvas.x * s * dpr + ox - 1);
            const y = Math.floor(this._dirtyBoundsCanvas.y * s * dpr + oy - 1);
            const w = Math.ceil((this._dirtyBoundsCanvas.x2 - this._dirtyBoundsCanvas.x) * s * dpr + 2);
            const h = Math.ceil((this._dirtyBoundsCanvas.y2 - this._dirtyBoundsCanvas.y) * s * dpr + 2);
            const cw = this._overlayCanvas.width;
            const ch = this._overlayCanvas.height;
            const clampX = Math.max(0, Math.min(x, cw));
            const clampY = Math.max(0, Math.min(y, ch));
            const clampW = Math.max(0, Math.min(w, cw - clampX));
            const clampH = Math.max(0, Math.min(h, ch - clampY));
            if (clampW > 0 && clampH > 0) {
                this._overlayCtx.clearRect(clampX, clampY, clampW, clampH);
            }
        } else {
            this._overlayCtx.clearRect(0, 0, this._overlayCanvas.width, this._overlayCanvas.height);
        }
        this._dirtyBoundsCanvas = null;
        this._overlayTransformScale = 0;
        this._overlayTransformX = 0;
        this._overlayTransformY = 0;
    }

    _each_visible_tile(fn) {
        const tr = this._tileRenderer || window.tileRenderer;
        if (!tr) return;
        const keys = tr.get_visible_keys();
        for (const info of tr.tileInfos) {
            if (keys.has(info.key)) {
                const ctx = info.ctx;
                const dpr = info.dpr;
                ctx.save();
                ctx.setTransform(dpr, 0, 0, dpr,
                    -info.rect.x * dpr, -info.rect.y * dpr);
                ctx.imageSmoothingEnabled = false;
                ctx.lineCap = 'round';
                ctx.lineJoin = 'round';
                fn(ctx, info);
                ctx.restore();
            }
        }
    }

    _each_tile(x1, y1, x2, y2, fn, padding = 0) {
        const tr = this._tileRenderer || window.tileRenderer;
        if (!tr) return;
        const infos = tr.infos_for_segment(x1, y1, x2, y2, padding);
        for (const info of infos) {
            const ctx = info.ctx;
            const dpr = info.dpr;
            ctx.save();
            ctx.setTransform(dpr, 0, 0, dpr,
                -info.rect.x * dpr, -info.rect.y * dpr);
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            fn(ctx);
            ctx.restore();
        }
    }

    batch_draw_update_frame_rate(mode) {
        this.frameRateMode = mode;

        if (mode === 'low') {
            this.currentFps = 30;
            this.drawInterval = 1000 / 30;
        } else if (mode === 'high') {
            this.currentFps = 60;
            this.drawInterval = 1000 / 60;
        } else {
            this.currentFps = 60;
            this.drawInterval = 1000 / 60;
        }
    }

    get is_adaptive() {
        return this.frameRateMode === 'adaptive';
    }

    batch_draw_calc_target_fps(commandCount) {
        if (commandCount < this.LOW_LOAD_THRESHOLD) {
            return this.LOW_LOAD_FPS;
        } else if (commandCount < this.MEDIUM_LOAD_THRESHOLD) {
            return this.MEDIUM_LOAD_FPS;
        } else if (commandCount < this.HIGH_LOAD_THRESHOLD) {
            return this.HIGH_LOAD_FPS;
        } else {
            return this.CRITICAL_LOAD_FPS;
        }
    }

    batch_draw_calc_adjust_fps(drawTime, commandCount) {
        const now = performance.now();
        if (now - this.lastAdjustTime < this.adjustCooldown) {
            return;
        }
        this.lastAdjustTime = now;

        this.drawTimes.push(drawTime);
        if (this.drawTimes.length > this.drawTimesMax) {
            this.drawTimes.shift();
        }

        this.commandCounts.push(commandCount);
        if (this.commandCounts.length > this.commandCountsMax) {
            this.commandCounts.shift();
        }

        const avgDrawTime = this.drawTimes.reduce((a, b) => a + b, 0) / this.drawTimes.length;
        const avgCommandCount = this.commandCounts.reduce((a, b) => a + b, 0) / this.commandCounts.length;

        const targetFps = this.batch_draw_calc_target_fps(avgCommandCount);
        const currentFrameTime = 1000 / this.currentFps;

        if (avgDrawTime > currentFrameTime * 1.5) {
            const newFps = Math.max(this.minFps, this.currentFps - this.fpsStep);
            if (newFps !== this.currentFps) {
                this.currentFps = newFps;
                this.drawInterval = 1000 / this.currentFps;
            }
        } else if (this.currentFps < targetFps && avgDrawTime < currentFrameTime * 0.7) {
            const newFps = Math.min(targetFps, this.currentFps + this.fpsStep);
            if (newFps !== this.currentFps) {
                this.currentFps = newFps;
                this.drawInterval = 1000 / this.currentFps;
            }
        }
    }

    batch_draw_fetch_stats() {
        return {
            currentFps: this.currentFps,
            targetFps: this.batch_draw_calc_target_fps(this.pendingCount),
            pendingCount: this.pendingCount,
            avgDrawTime: this.drawTimes.length > 0
                ? this.drawTimes.reduce((a, b) => a + b, 0) / this.drawTimes.length
                : 0,
            frameRateMode: this.frameRateMode
        };
    }

    batch_draw_create_command(type, fromX, fromY, toX, toY, color, lineWidth) {
        const idx = this.pendingCount++;
        if (idx >= this.pendingCommands.length) {
            this.pendingCommands.push({ type, fromX, fromY, toX, toY, color, lineWidth, originalLineWidth: lineWidth });
        } else {
            const cmd = this.pendingCommands[idx];
            cmd.type = type;
            cmd.fromX = fromX;
            cmd.fromY = fromY;
            cmd.toX = toX;
            cmd.toY = toY;
            cmd.color = color;
            cmd.lineWidth = lineWidth;
            cmd.originalLineWidth = lineWidth;
        }

        if (this.is_adaptive && this.pendingCount === 1) {
            const targetFps = this.batch_draw_calc_target_fps(1);
            if (this.currentFps > targetFps) {
                this.currentFps = targetFps;
                this.drawInterval = 1000 / this.currentFps;
            } else if (this.currentFps < targetFps) {
                /* 负载已降为单笔，快速恢复帧率降低感知延迟 */
                this.currentFps = Math.min(targetFps, this.currentFps + this.fpsStep * 2);
                this.drawInterval = 1000 / this.currentFps;
            }
        }

        this.batch_draw_setup_schedule();
    }

    batch_draw_setup_schedule() {
        if (this.drawRafId !== null) return;

        const now = performance.now();
        const timeSinceLastDraw = now - this.lastDrawTime;

        if (timeSinceLastDraw >= this.drawInterval) {
            this.batch_draw_handle_flush();
        } else {
            this.drawRafId = requestAnimationFrame(() => {
                this.drawRafId = null;
                this.batch_draw_handle_flush();
            });
        }
    }

    _calc_pen_line_width(speed, baseWidth, lastLineWidth, dist) {
        const minRatio = window.DRAW_CONFIG?.penMinWidthRatio ?? 0.4;
        const speedScale = Math.max(0.4, Math.min(2.5, baseWidth / 4));
        const maxSpeed = 2.5 * speedScale;
        const minSpeed = 0.2 * speedScale;

        let lineWidth;
        if (speed >= maxSpeed) {
            lineWidth = baseWidth * minRatio;
        } else if (speed <= minSpeed) {
            lineWidth = baseWidth;
        } else {
            const ratio = (speed - minSpeed) / (maxSpeed - minSpeed);
            const eased = ratio * ratio * (3 - 2 * ratio);
            lineWidth = baseWidth - eased * (baseWidth * minRatio);
        }

        const blend = Math.max(0.3, Math.min(0.85, 1 - dist / (baseWidth * 3)));
        lineWidth = lineWidth * (1 - blend) + lastLineWidth * blend;

        const maxDelta = baseWidth * 0.12;
        lineWidth = Math.min(lastLineWidth + maxDelta, Math.max(lastLineWidth - maxDelta, lineWidth));

        return Math.max(0.5, lineWidth);
    }

    _apply_start_taper(lineWidth, baseWidth, segmentIndex, totalInBatch) {
        const globalIndex = this._totalSegments + segmentIndex;
        if (globalIndex < 4) {
            const taperT = (globalIndex + 1) / 4;
            const eased = taperT * taperT * (3 - 2 * taperT);
            const minStart = baseWidth * 0.2;
            return minStart + (lineWidth - minStart) * eased;
        }
        return lineWidth;
    }


    _draw_segment_ellipse(ctx, fromX, fromY, toX, toY, lineWidth, color) {
        const dx = toX - fromX;
        const dy = toY - fromY;
        const segLen = Math.sqrt(dx * dx + dy * dy);
        const halfW = Math.max(0.5, lineWidth) / 2;
        const cx = (fromX + toX) / 2;
        const cy = (fromY + toY) / 2;
        ctx.fillStyle = color;
        ctx.beginPath();
        if (segLen < 0.5) {
            ctx.arc(cx, cy, halfW, 0, Math.PI * 2);
        } else {
            ctx.ellipse(cx, cy, segLen / 2, halfW, Math.atan2(dy, dx), 0, Math.PI * 2);
        }
        ctx.fill();
    }

    batch_draw_handle_flush() {
        const count = this.pendingCount;
        if (count === 0) return;
        this.pendingCount = 0;

        const drawStart = performance.now();

        this._sync_overlay_transform();

        const commands = this.pendingCommands;
        let currentType = this.lastType;
        let currentColor = this.lastColor;
        let currentLineWidth = this.lastLineWidth;

        const getPenEffect = window.get_pen_effect_mode;
        this._penEffectMode = getPenEffect ? getPenEffect() : 'off';

        let lastLineWidth = currentLineWidth || 5;
        let lastMoveTime = this.lastBatchMoveTime || performance.now() - 16;

        const curTime = performance.now();
        const batchTimeSpan = Math.max(1, curTime - lastMoveTime);
        const perSegTime = Math.min(batchTimeSpan / count, 8);
        lastMoveTime = curTime;

        if (this._overlayCtx) {
            this._overlayCtx.globalCompositeOperation = 'source-over';
            this._overlayCtx.lineCap = 'round';
            this._overlayCtx.lineJoin = 'round';
        }

        let eraseByTile = null;
        let batchFirst = true;
        let batchWidth = 0;
        let batchColor = null;

        if (this._penEffectMode === 'limited') {
            const prev = this._segmentTimes.length > 0 ? this._segmentTimes[this._segmentTimes.length - 1] : 0;
            for (let i = 0; i < count; i++) {
                this._segmentTimes.push(prev + perSegTime * (i + 1));
            }
        }

        for (let i = 0; i < count; i++) {
            const cmd = commands[i];

            const fromX = cmd.fromX, fromY = cmd.fromY;
            const toX = cmd.toX, toY = cmd.toY;

            let lineWidth = cmd.lineWidth;
            if (this._penEffectMode === 'full' && cmd.type === 'draw') {
                const dx = toX - fromX;
                const dy = toY - fromY;
                const dist = Math.sqrt(dx * dx + dy * dy);
                const rawSpeed = dist / perSegTime;
                this._speedBuffer.push(rawSpeed);
                if (this._speedBuffer.length > 3) {
                    this._speedBuffer.shift();
                }
                const speed = this._speedBuffer.reduce((a, b) => a + b, 0) / this._speedBuffer.length;
                lineWidth = this._calc_pen_line_width(speed, cmd.lineWidth, lastLineWidth, dist);

                if (this._strokeStart) {
                    lineWidth = this._apply_start_taper(lineWidth, cmd.lineWidth, i, count);
                }
            }

            if (cmd.type === 'draw') {
                this._storedWidths.push(lineWidth);
            } else if (cmd.type === 'erase') {
                this._storedWidths.push(lineWidth);
            }
            lastLineWidth = lineWidth;

            if (cmd.type !== currentType || cmd.color !== currentColor) {
                currentType = cmd.type;
                currentColor = cmd.color;
            }

            if (cmd.type === 'erase') {
                /* 刷出上一个绘制批处理 */
                if (this._overlayCtx && !batchFirst) {
                    this._overlayCtx.stroke();
                    batchFirst = true;
                }
                const tr = this._tileRenderer || window.tileRenderer;
                if (tr) {
                    if (!eraseByTile) eraseByTile = new Map();
                    const halfWidth = (lineWidth || 20) / 2;
                    const infos = tr.infos_for_segment(fromX, fromY, toX, toY, halfWidth);
                    for (const info of infos) {
                        let entry = eraseByTile.get(info.key);
                        if (!entry) {
                            entry = { paths: [], lineWidths: [] };
                            eraseByTile.set(info.key, entry);
                        }
                        entry.paths.push({ fromX, fromY, toX, toY });
                        entry.lineWidths.push(lineWidth);
                    }
                }
            } else if (this._overlayCtx) {
                const ctx = this._overlayCtx;

                if (this.ellipseMode) {
                    this._draw_segment_ellipse(ctx, fromX, fromY, toX, toY, lineWidth, cmd.color || currentColor);
                } else {
                    const needsBreak = batchFirst ||
                        (cmd.color && cmd.color !== batchColor) ||
                        Math.abs(lineWidth - batchWidth) >= 0.5;

                    if (needsBreak) {
                        if (!batchFirst) ctx.stroke();
                        if (cmd.color) ctx.strokeStyle = cmd.color;
                        ctx.lineWidth = Math.max(0.5, lineWidth);
                        batchColor = cmd.color;
                        batchWidth = lineWidth;
                        ctx.beginPath();

                        const midX = (fromX + toX) / 2;
                        const midY = (fromY + toY) / 2;
                        const isFirstSeg = (i === 0 && (this._strokeStart || this._lastMidX === null));
                        if (isFirstSeg) {
                            ctx.moveTo(fromX, fromY);
                            ctx.lineTo(midX, midY);
                        } else if (batchFirst) {
                            const prevMx = (i === 0 ? this._lastMidX : ((commands[i - 1].fromX + commands[i - 1].toX) / 2));
                            const prevMy = (i === 0 ? this._lastMidY : ((commands[i - 1].fromY + commands[i - 1].toY) / 2));
                            ctx.moveTo(prevMx, prevMy);
                            ctx.quadraticCurveTo(fromX, fromY, midX, midY);
                        } else {
                            ctx.quadraticCurveTo(fromX, fromY, midX, midY);
                        }
                        batchFirst = false;
                    } else {
                        const midX = (fromX + toX) / 2;
                        const midY = (fromY + toY) / 2;
                        ctx.quadraticCurveTo(fromX, fromY, midX, midY);
                    }
                }
            }

            const halfW = (lineWidth || 5) / 2;
            const minX = Math.min(fromX, toX) - halfW;
            const maxX = Math.max(fromX, toX) + halfW;
            const minY = Math.min(fromY, toY) - halfW;
            const maxY = Math.max(fromY, toY) + halfW;
            if (!this._dirtyBoundsCanvas) {
                this._dirtyBoundsCanvas = { x: minX, y: minY, x2: maxX, y2: maxY };
            } else {
                if (minX < this._dirtyBoundsCanvas.x) this._dirtyBoundsCanvas.x = minX;
                if (minY < this._dirtyBoundsCanvas.y) this._dirtyBoundsCanvas.y = minY;
                if (maxX > this._dirtyBoundsCanvas.x2) this._dirtyBoundsCanvas.x2 = maxX;
                if (maxY > this._dirtyBoundsCanvas.y2) this._dirtyBoundsCanvas.y2 = maxY;
            }

            if (i === count - 1) {
                this._lastMidX = (fromX + toX) / 2;
                this._lastMidY = (fromY + toY) / 2;
                this._lastToX = toX;
                this._lastToY = toY;
            }
        }

        /* 提交最后一个批处理路径 */
        if (this._overlayCtx && !batchFirst) {
            this._overlayCtx.stroke();
        }

        if (eraseByTile && eraseByTile.size > 0) {
            const tr = this._tileRenderer || window.tileRenderer;
            if (tr) {
                for (const info of tr.tileInfos) {
                    const entry = eraseByTile.get(info.key);
                    if (!entry) continue;
                    const ctx = info.ctx;
                    const dpr = info.dpr;
                    ctx.save();
                    ctx.setTransform(dpr, 0, 0, dpr, -info.rect.x * dpr, -info.rect.y * dpr);
                    ctx.globalCompositeOperation = 'destination-out';
                    ctx.strokeStyle = 'rgba(0,0,0,1)';
                    
                    const eraser = window.__eraser;
                    if (eraser) {
                        ctx.beginPath();
                        for (let j = 0; j < entry.paths.length; j++) {
                            const w = entry.lineWidths && entry.lineWidths.length > 0
                                ? entry.lineWidths[j] || 20
                                : entry.lineWidth || 20;
                            const p = entry.paths[j];
                            eraser.renderEraseSegment(ctx, p.fromX, p.fromY, p.toX, p.toY, w);
                        }
                        ctx.fill();
                    }
                    ctx.restore();
                }
            }
        }

        this._totalSegments += count;
        this._strokeStart = false;

        this.lastBatchMoveTime = curTime;

        const drawEnd = performance.now();
        const drawTime = drawEnd - drawStart;
        this.lastDrawTime = drawEnd;

        this.lastType = currentType;
        this.lastColor = currentColor;
        this.lastLineWidth = lastLineWidth;

        if (this.is_adaptive) {
            this.batch_draw_calc_adjust_fps(drawTime, count);
        }
    }

    _apply_speed_taper(widths, points, baseWidth) {
        if (!points || points.length < 2 || widths.length !== points.length) return;
        const tailDuration = window.DRAW_CONFIG?.penTailDuration ?? 30;
        const totalDuration = this._segmentTimes.length > 0 ? this._segmentTimes[this._segmentTimes.length - 1] : 0;
        if (totalDuration <= 0) return;
        const tailStartTime = Math.max(0, totalDuration - tailDuration);
        let tailStart = -1;
        for (let i = 0; i < this._segmentTimes.length; i++) {
            if (this._segmentTimes[i] >= tailStartTime) { tailStart = i; break; }
        }
        if (tailStart < 0 || tailStart >= widths.length) return;
        let lastWidth = tailStart > 0 ? widths[tailStart - 1] : baseWidth;
        for (let i = tailStart; i < widths.length; i++) {
            const pt = points[i];
            const dx = pt.toX - pt.fromX;
            const dy = pt.toY - pt.fromY;
            const dist = Math.sqrt(dx * dx + dy * dy);
            const prevTime = i > 0 ? this._segmentTimes[i - 1] : 0;
            const segTime = Math.max(1, this._segmentTimes[i] - prevTime);
            const speed = dist / segTime;
            widths[i] = this._calc_pen_line_width(speed, baseWidth, lastWidth, dist);
            lastWidth = widths[i];
        }
    }

    reset_state() {
        this.pendingCount = 0;
        this.pendingCommands.length = 0;
        if (this.drawRafId !== null) {
            cancelAnimationFrame(this.drawRafId);
            this.drawRafId = null;
        }
        this.lastType = null;
        this.lastColor = null;
        this.lastLineWidth = null;
        this.lastBatchMoveTime = 0;
        this._penEffectMode = 'off';
        this._strokeStart = true;
        this._totalSegments = 0;
        this._lastMidX = null;
        this._lastMidY = null;
        this._lastToX = null;
        this._lastToY = null;
        this._speedBuffer = [];
        this._storedWidths = [];
        this._segmentTimes = [];
        this._dirtyBoundsCanvas = null;
        this._limitedTailWidth = null;
        this.clear_overlay();
    }

    batch_draw_init_start() {
        this.pendingCount = 0;
        this.pendingCommands.length = 0;
        this.lastDrawTime = performance.now();
        this._penEffectMode = 'off';
        this._strokeStart = true;
        this.ellipseMode = window.DRAW_CONFIG?.ellipseStrokeEnabled === true;
        this._totalSegments = 0;
        this._lastMidX = null;
        this._lastMidY = null;
        this._lastToX = null;
        this._lastToY = null;
        this._speedBuffer = [];
        this._storedWidths = [];
        this._segmentTimes = [];
        this._dirtyBoundsCanvas = null;
        this.eraserShape = 'square';

        if (this.is_adaptive) {
            this.currentFps = this.LOW_LOAD_FPS;
            this.drawInterval = 1000 / this.currentFps;
        }

        this._each_visible_tile((ctx, info) => {
            ctx.imageSmoothingEnabled = false;
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
        });

        this.show_overlay();
    }

    batch_draw_handle_end() {
        if (this.drawRafId !== null) {
            cancelAnimationFrame(this.drawRafId);
            this.drawRafId = null;
        }

        this.batch_draw_handle_flush();
        this._sync_overlay_transform();

        if (this._lastMidX !== null && this._lastToX !== null) {
            if (this._overlayCtx) {
                const ctx = this._overlayCtx;
                const cfg = window.DRAW_CONFIG || {};
                ctx.globalCompositeOperation = 'source-over';
                this._limitedTailWidth = this.lastLineWidth || 5;
                if (this._penEffectMode === 'limited') {
                    const baseW = cfg.penWidth || 5;
                    const minRatio = cfg.penMinWidthRatio ?? 0.4;
                    const fromX = 2 * this._lastMidX - this._lastToX;
                    const fromY = 2 * this._lastMidY - this._lastToY;
                    const dx = this._lastToX - fromX;
                    const dy = this._lastToY - fromY;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist > 0 && this._segmentTimes.length >= 2) {
                        const prevTime = this._segmentTimes.length > 1
                            ? this._segmentTimes[this._segmentTimes.length - 2] : 0;
                        const segTime = Math.max(1, this._segmentTimes[this._segmentTimes.length - 1] - prevTime);
                        const speed = dist / segTime;
                        this._limitedTailWidth = this._calc_pen_line_width(speed, baseW, this.lastLineWidth || baseW, dist);
                    } else {
                        this._limitedTailWidth = Math.max(this._limitedTailWidth, baseW * minRatio);
                    }
                }
                if (this.ellipseMode) {
                    this._draw_segment_ellipse(ctx, this._lastMidX, this._lastMidY, this._lastToX, this._lastToY,
                        this._limitedTailWidth, cfg.penColor || '#3498db');
                } else {
                    ctx.strokeStyle = cfg.penColor || '#3498db';
                    ctx.lineWidth = Math.max(0.5, this._limitedTailWidth);
                    ctx.lineCap = 'round';
                    ctx.lineJoin = 'round';
                    ctx.beginPath();
                    ctx.moveTo(this._lastMidX, this._lastMidY);
                    const endCpx = 2 * this._lastMidX - this._lastToX;
                    const endCpy = 2 * this._lastMidY - this._lastToY;
                    ctx.quadraticCurveTo(endCpx, endCpy, this._lastToX, this._lastToY);
                    ctx.stroke();
                }
            }
        }

        const tailW = this._limitedTailWidth || this.lastLineWidth || 5;
        const tailHalf = tailW / 2;
        if (this._lastMidX !== null && this._lastToX !== null) {
            const tMinX = Math.min(this._lastMidX, this._lastToX) - tailHalf;
            const tMaxX = Math.max(this._lastMidX, this._lastToX) + tailHalf;
            const tMinY = Math.min(this._lastMidY, this._lastToY) - tailHalf;
            const tMaxY = Math.max(this._lastMidY, this._lastToY) + tailHalf;
            if (!this._dirtyBoundsCanvas) {
                this._dirtyBoundsCanvas = { x: tMinX, y: tMinY, x2: tMaxX, y2: tMaxY };
            } else {
                if (tMinX < this._dirtyBoundsCanvas.x) this._dirtyBoundsCanvas.x = tMinX;
                if (tMinY < this._dirtyBoundsCanvas.y) this._dirtyBoundsCanvas.y = tMinY;
                if (tMaxX > this._dirtyBoundsCanvas.x2) this._dirtyBoundsCanvas.x2 = tMaxX;
                if (tMaxY > this._dirtyBoundsCanvas.y2) this._dirtyBoundsCanvas.y2 = tMaxY;
            }
        }

        this.clear_overlay();

        if (this.is_adaptive) {
            this.drawTimes = [];
            this.commandCounts = [];
        }
    }

    batch_draw_delete_all() {
        this.reset_state();
    }
}

window.RealtimeBatchDrawManager = RealtimeBatchDrawManager;
window.batchDrawManager = new RealtimeBatchDrawManager();
