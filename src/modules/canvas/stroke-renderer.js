import { resetContextState, updateContextState } from './context-state.js';

export function getPenEffectMode() {
    return window.DRAW_CONFIG?.penEffectMode || 'off';
}

function _render_segment_ellipse(ctx, fromX, fromY, toX, toY, lineWidth, color) {
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
                    _render_segment_ellipse(ctx, point.fromX, point.fromY, point.toX, point.toY, lineWidth, strokeColor);
                }
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
            for (let i = 0; i < stroke.points.length; i++) {
                const p = stroke.points[i];
                _render_segment_ellipse(ctx, p.fromX, p.fromY, p.toX, p.toY, baseLineWidth, strokeColor);
            }
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
