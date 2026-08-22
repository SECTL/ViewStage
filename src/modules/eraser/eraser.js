export function renderEraseSegment(ctx, fromX, fromY, toX, toY, lineWidth) {
    ctx.lineWidth = lineWidth;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(fromX, fromY);
    ctx.lineTo(toX, toY);
    ctx.stroke();
}

export function renderEraseStroke(ctx, stroke, baseLineWidth) {
    const hasStoredWidths = stroke.storedWidths && stroke.storedWidths.length > 0;
    const hasVariableWidths = stroke.variableWidths && stroke.variableWidths.length > 0;

    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

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
        ctx.lineWidth = w;
        ctx.beginPath();
        ctx.moveTo(pt.fromX, pt.fromY);
        ctx.lineTo(pt.toX, pt.toY);
        ctx.stroke();
    }
}
