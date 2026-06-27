import { resetContextState, updateContextState } from './context-state.js';

export function getPenEffectMode() {
    return window.DRAW_CONFIG?.penEffectMode || 'off';
}

/* FuzzyContains：两矩形间交面积 ≥ min(面积)×percent% 时认为重叠 */
function _fuzzy_contains(r1, r2, percent) {
    const ix = Math.max(r1.x, r2.x);
    const iy = Math.max(r1.y, r2.y);
    const ix2 = Math.min(r1.x + r1.w, r2.x + r2.w);
    const iy2 = Math.min(r1.y + r1.h, r2.y + r2.h);
    const iw = Math.max(0, ix2 - ix);
    const ih = Math.max(0, iy2 - iy);
    if (iw === 0 || ih === 0) return null;
    const a1 = r1.w * r1.h;
    const a2 = r2.w * r2.h;
    const minA = Math.min(a1, a2);
    if ((iw * ih) / minA * 100 < percent) return null;
    return a1 >= a2 ? 'r1cr2' : 'r2cr1';
}

/* Outer Tangent 外切线法 — 每节点独立宽度 */
function _build_ellipse_outline(ctx, segments) {
    /* FuzzyContains 节点消除 */
    const filtered = [];
    let prevRect = null;
    for (let i = 0; i < segments.length; i++) {
        const s = segments[i];
        const hw = Math.max(0.5, s.lineWidth) / 2;
        const rect = { x: Math.min(s.fromX, s.toX) - hw, y: Math.min(s.fromY, s.toY) - hw,
            w: Math.abs(s.toX - s.fromX) + hw * 2, h: Math.abs(s.toY - s.fromY) + hw * 2 };
        const fc = prevRect ? _fuzzy_contains(prevRect, rect, 95) : null;
        if (fc === 'r1cr2') continue;
        if (fc === 'r2cr1') filtered.pop();
        filtered.push(s);
        prevRect = rect;
    }

    const n_seg = filtered.length;
    if (n_seg === 0) return;

    /* 构建节点数组 [{x, y, w}] — 每个滤波后 segment 贡献一个 from 节点 + 最后一个 to 节点 */
    let nodes;
    if (n_seg < 3) {
        nodes = [];
        for (let i = 0; i < n_seg; i++) {
            if (i === 0) nodes.push({ x: filtered[i].fromX, y: filtered[i].fromY, w: filtered[i].lineWidth });
            nodes.push({ x: filtered[i].toX, y: filtered[i].toY, w: filtered[i].lineWidth });
        }
    } else {
        const raw = filtered.map(s => ({
            x: (s.fromX + s.toX) / 2, y: (s.fromY + s.toY) / 2, w: s.lineWidth
        }));
        raw[0] = { x: filtered[0].fromX, y: filtered[0].fromY, w: filtered[0].lineWidth };
        raw[raw.length - 1] = {
            x: filtered[filtered.length - 1].toX,
            y: filtered[filtered.length - 1].toY,
            w: filtered[filtered.length - 1].lineWidth
        };
        const smoothPts = [];
        for (let i = 0; i < raw.length - 1; i++) {
            const p0 = raw[Math.max(0, i - 1)];
            const p1 = raw[i];
            const p2 = raw[i + 1];
            const p3 = raw[Math.min(raw.length - 1, i + 2)];
            const c1 = { x: p1.x + (p2.x - p0.x) / 6, y: p1.y + (p2.y - p0.y) / 6 };
            const c2 = { x: p2.x - (p3.x - p1.x) / 6, y: p2.y - (p3.y - p1.y) / 6 };
            const mx = (p1.x + 3 * c1.x + 3 * c2.x + p2.x) / 8;
            const my = (p1.y + 3 * c1.y + 3 * c2.y + p2.y) / 8;
            const mw = p1.w * 0.75 + p2.w * 0.25;
            if (i === 0) {
                smoothPts.push({ x: p1.x, y: p1.y, w: p1.w });
                smoothPts.push({ x: mx, y: my, w: mw });
            } else {
                smoothPts.push({ x: mx, y: my, w: mw });
            }
        }
        smoothPts.push({ x: raw[raw.length - 1].x, y: raw[raw.length - 1].y, w: raw[raw.length - 1].w });
        nodes = smoothPts;
    }

    const m = nodes.length;
    if (m < 2) {
        if (m === 1) {
            const r = Math.max(0.5, nodes[0].w) / 2;
            ctx.beginPath();
            ctx.arc(nodes[0].x, nodes[0].y, r, 0, Math.PI * 2);
            ctx.fill();
        }
        return;
    }

    /* Outer Tangent 外切线法构建 quads */
    const segs = [];
    for (let i = 0; i < m - 1; i++) {
        const n0 = nodes[i], n1 = nodes[i + 1];
        const dx = n1.x - n0.x, dy = n1.y - n0.y;
        const len = Math.hypot(dx, dy);
        if (len < 0.5) continue;
        const r0 = Math.max(0.5, n0.w) / 2;
        const r1 = Math.max(0.5, n1.w) / 2;
        const a = Math.atan2(dy, dx);

        /* Outer tangent 方向 */
        const Vx = dx / len, Vy = dy / len;
        const Px = -Vy, Py = Vx;  // perp (CCW)

            const dr = r1 - r0;
            const drSqOverL = (dr * dr) / (len * len);
            let t1x, t1y, t2x, t2y;
            if (drSqOverL < 1) {
                const k = Math.sqrt(1 - drSqOverL);
                const pComp = -(dr / len);
                t1x = Vx * pComp + Px * k;
                t1y = Vy * pComp + Py * k;
                t2x = Vx * pComp - Px * k;
                t2y = Vy * pComp - Py * k;
            } else {
                t1x = Px; t1y = Py;
                t2x = -Px; t2y = -Py;
            }

            segs.push({
                fx: n0.x, fy: n0.y, tx: n1.x, ty: n1.y,
                hw0: r0, hw1: r1, angle: a,
                A: { x: n0.x + t1x * r0, y: n0.y + t1y * r0 },
                B: { x: n1.x + t1x * r1, y: n1.y + t1y * r1 },
                C: { x: n1.x + t2x * r1, y: n1.y + t2y * r1 },
                D: { x: n0.x + t2x * r0, y: n0.y + t2y * r0 }
            });
    }

    const k = segs.length;
    if (k === 0) return;

    /* Polyline 轮廓（平端 cap + 节点填充圆补圆） */
    ctx.beginPath();
    ctx.moveTo(segs[0].D.x, segs[0].D.y);
    for (let i = 0; i < k; i++) {
        ctx.lineTo(segs[i].A.x, segs[i].A.y);
        ctx.lineTo(segs[i].B.x, segs[i].B.y);
    }
    for (let i = k - 1; i >= 0; i--) {
        ctx.lineTo(segs[i].C.x, segs[i].C.y);
        ctx.lineTo(segs[i].D.x, segs[i].D.y);
    }
    ctx.closePath();
    ctx.fill();

    /* 节点填充圆（所有节点都补圆消除缺口，防止锯齿） */
    const patchIdx = new Set();
    for (let i = 0; i < m; i++) patchIdx.add(i);
    for (const idx of patchIdx) {
        const n = nodes[idx];
        const r = Math.max(0.5, n.w) / 2;
        ctx.beginPath();
        ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
        ctx.fill();
    }
}

function _draw_capsule(ctx, fromX, fromY, toX, toY, lineWidth) {
    _build_ellipse_outline(ctx, [{ fromX, fromY, toX, toY, lineWidth }]);
}

/**
 * 按原始顺序逐个绘制笔画：draw/comment 用 source-over，erase 用 destination-out
 * 优化：可变宽度段按线宽分组合并连续段到同一条路径，减少 stroke() 调用
 * @param {CanvasRenderingContext2D} ctx
 * @param {Array} strokes - 笔画数组
 * @param {Object} options
 * @param {number} options.renderScale - 当前 canvas 缩放比
 * @param {Object} [options.penManager] - RealPenManager 实例（笔锋渲染）
 */
export async function renderStrokesToContext(ctx, strokes, options = {}) {
    if (strokes.length === 0) return;

    const DRAW_CONFIG = window.DRAW_CONFIG;
    const penManager = options.penManager || null;

    resetContextState();

    updateContextState(ctx, {
        lineCap: 'round',
        lineJoin: 'round'
    });

    let currentEraserShape = 'round';
    const pen_effect = getPenEffectMode();

    let batchActive = false;
    let batchColor = null;
    let batchLineWidth = 0;
    let batchIsErase = false;
    let batchPrevMidX = 0;
    let batchPrevMidY = 0;

    const batch_flush = () => {
        if (batchActive) {
            ctx.stroke();
            batchActive = false;
        }
    };

    for (const stroke of strokes) {
        if (!stroke.points || stroke.points.length < 1) continue;

        const hasStoredWidths = stroke.storedWidths && stroke.storedWidths.length > 0;
        const hasVariableWidths = stroke.variableWidths && stroke.variableWidths.length > 0;
        const strokeColor = stroke.color || DRAW_CONFIG.penColor;
        let baseLineWidth;
        if (stroke.type === 'erase') {
            baseLineWidth = stroke.eraserSize || (stroke.eraserSizeRaw / (stroke.scale || 1));
        } else if (stroke.type === 'draw') {
            baseLineWidth = stroke.lineWidth || DRAW_CONFIG.penWidth;
        } else {
            baseLineWidth = stroke.lineWidth || (stroke.type === 'erase' ? DRAW_CONFIG.eraserSize : DRAW_CONFIG.penWidth);
        }

        if (stroke.type === 'erase') {
            batch_flush();
            updateContextState(ctx, {
                globalCompositeOperation: 'destination-out',
                fillStyle: '#000000',
                strokeStyle: '#000000'
            });
        } else {
            if (batchIsErase) batch_flush();
            updateContextState(ctx, {
                globalCompositeOperation: 'source-over'
            });

            if (pen_effect !== 'off' && stroke.type === 'draw' && penManager) {
                batch_flush();
                if (!window.batchDrawManager?.ellipseMode) {
                    const tessellated = penManager.build_tessellated_stroke(stroke, pen_effect);
                    if (tessellated) {
                        penManager.render_tessellated_stroke(ctx, tessellated, 1);
                        continue;
                    }
                }
            }

            updateContextState(ctx, {
                strokeStyle: strokeColor
            });
            batchColor = strokeColor;
            batchIsErase = false;
        }

        if (hasStoredWidths || hasVariableWidths) {
            batch_flush();
            if (stroke.type === 'erase') {
                const eraser = window.__eraser;
                if (eraser) eraser.renderEraseStroke(ctx, stroke, baseLineWidth);
                continue;
            }

            if (window.batchDrawManager?.ellipseMode) {
                ctx.fillStyle = strokeColor;
                const segs = [];
                for (let i = 0; i < stroke.points.length; i++) {
                    const point = stroke.points[i];
                    let lw;
                    if (hasStoredWidths && stroke.storedWidths[i] !== undefined) {
                        lw = stroke.storedWidths[i];
                    } else if (hasVariableWidths && stroke.variableWidths[i] !== undefined) {
                        lw = stroke.variableWidths[i];
                    } else {
                        lw = baseLineWidth;
                    }
                    segs.push({ fromX: point.fromX, fromY: point.fromY, toX: point.toX, toY: point.toY, lineWidth: lw });
                }
                _build_ellipse_outline(ctx, segs);
                continue;
            }

            let varBatchActive = false;
            let varBatchWidth = 0;
            let varPrevMidX = 0, varPrevMidY = 0;

            for (let i = 0; i < stroke.points.length; i++) {
                const point = stroke.points[i];
                let lineWidth;
                if (hasStoredWidths && stroke.storedWidths[i] !== undefined) {
                    lineWidth = stroke.storedWidths[i];
                } else if (hasVariableWidths && stroke.variableWidths[i] !== undefined) {
                    lineWidth = stroke.variableWidths[i];
                } else {
                    lineWidth = baseLineWidth;
                }
                const midX = (point.fromX + point.toX) / 2;
                const midY = (point.fromY + point.toY) / 2;

                if (!varBatchActive || Math.abs(lineWidth - varBatchWidth) >= 0.5) {
                    if (varBatchActive) ctx.stroke();
                    updateContextState(ctx, { lineWidth });
                    varBatchWidth = lineWidth;
                    ctx.beginPath();
                    if (!varBatchActive) {
                        ctx.moveTo(point.fromX, point.fromY);
                        ctx.lineTo(midX, midY);
                    } else {
                        ctx.moveTo(varPrevMidX, varPrevMidY);
                        ctx.quadraticCurveTo(point.fromX, point.fromY, midX, midY);
                    }
                    varBatchActive = true;
                } else {
                    ctx.quadraticCurveTo(point.fromX, point.fromY, midX, midY);
                }
                varPrevMidX = midX;
                varPrevMidY = midY;
            }
            if (varBatchActive) ctx.stroke();
            continue;
        }

        if (stroke.type === 'erase') {
            batch_flush();
            const eraser = window.__eraser;
            if (eraser) eraser.renderEraseStroke(ctx, stroke, baseLineWidth);
            continue;
        }

        if (window.batchDrawManager?.ellipseMode) {
            if (batchActive) batch_flush();
            batchColor = strokeColor;
            batchIsErase = (stroke.type === 'erase');
            ctx.fillStyle = strokeColor;
            const segs = stroke.points.map(p => ({
                fromX: p.fromX, fromY: p.fromY, toX: p.toX, toY: p.toY, lineWidth: baseLineWidth
            }));
            _build_ellipse_outline(ctx, segs);
            continue;
        }

        if (!batchActive ||
            batchIsErase !== (stroke.type === 'erase') ||
            batchColor !== strokeColor ||
            Math.abs(baseLineWidth - batchLineWidth) >= 0.5) {
            batch_flush();
            updateContextState(ctx, { lineWidth: baseLineWidth });
            batchLineWidth = baseLineWidth;
            batchColor = strokeColor;
            batchIsErase = (stroke.type === 'erase');

            const pts = stroke.points;
            const path = new Path2D();
            path.moveTo(pts[0].fromX, pts[0].fromY);
            path.lineTo(pts[0].toX, pts[0].toY);
            for (let i = 1; i < pts.length; i++) {
                const p = pts[i];
                path.lineTo(p.fromX, p.fromY);
                path.lineTo(p.toX, p.toY);
            }
            ctx.stroke(path);
            const lastPt = pts[pts.length - 1];
            batchPrevMidX = (lastPt.fromX + lastPt.toX) / 2;
            batchPrevMidY = (lastPt.fromY + lastPt.toY) / 2;
        } else {
            const pts = stroke.points;
            if (!batchActive) {
                batchActive = true;
                ctx.beginPath();
                ctx.moveTo(batchPrevMidX, batchPrevMidY);
            }
            ctx.lineTo(pts[0].fromX, pts[0].fromY);
            let midX = (pts[0].fromX + pts[0].toX) / 2;
            let midY = (pts[0].fromY + pts[0].toY) / 2;
            ctx.lineTo(midX, midY);
            for (let i = 1; i < pts.length; i++) {
                const nmidX = (pts[i].fromX + pts[i].toX) / 2;
                const nmidY = (pts[i].fromY + pts[i].toY) / 2;
                ctx.moveTo(midX, midY);
                ctx.quadraticCurveTo(pts[i].fromX, pts[i].fromY, nmidX, nmidY);
                midX = nmidX;
                midY = nmidY;
            }
            batchPrevMidX = midX;
            batchPrevMidY = midY;
        }
    }

    batch_flush();

    updateContextState(ctx, {
        globalCompositeOperation: 'source-over',
        lineCap: 'round',
        lineJoin: 'round'
    });
}
