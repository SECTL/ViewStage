export function renderEraseSegment(ctx, fromX, fromY, toX, toY, lineWidth) {
    const hw = lineWidth / 2;
    ctx.rect(fromX - hw, fromY - hw, lineWidth, lineWidth);
    ctx.rect(toX - hw, toY - hw, lineWidth, lineWidth);
    const x1 = fromX + hw, y1 = fromY - hw;
    const x2 = toX + hw, y2 = toY - hw;
    const x3 = toX - hw, y3 = toY + hw;
    const x4 = fromX - hw, y4 = fromY + hw;
    const area = (x1 * y2 - x2 * y1) + (x2 * y3 - x3 * y2) + (x3 * y4 - x4 * y3) + (x4 * y1 - x1 * y4);
    if (area >= 0) {
        ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.lineTo(x3, y3); ctx.lineTo(x4, y4);
    } else {
        ctx.moveTo(x1, y1); ctx.lineTo(x4, y4); ctx.lineTo(x3, y3); ctx.lineTo(x2, y2);
    }
    ctx.closePath();
}

export function renderEraseStroke(ctx, stroke, baseLineWidth) {
    const hasStoredWidths = stroke.storedWidths && stroke.storedWidths.length > 0;
    const hasVariableWidths = stroke.variableWidths && stroke.variableWidths.length > 0;
    ctx.beginPath();
    for (let i = 0; i < stroke.points.length; i++) {
        const pt = stroke.points[i];
        let w;
        if (hasStoredWidths && stroke.storedWidths[i] !== undefined) {
            w = stroke.storedWidths[i];
        } else if (hasVariableWidths && stroke.variableWidths[i] !== undefined) {
            w = stroke.variableWidths[i];
        } else {
            w = baseLineWidth;
        }
        renderEraseSegment(ctx, pt.fromX, pt.fromY, pt.toX, pt.toY, w);
    }
    ctx.fill();
}
