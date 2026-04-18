class RealtimeBatchDrawManager {
    constructor() {
        this.ctx = null;
        this.pendingCommands = [];
        this.pendingCount = 0;
        this.drawRafId = null;
        this.highFrameRate = false;
        this.drawInterval = 1000 / 30;
        this.lastDrawTime = 0;
        this.lastType = null;
        this.lastColor = null;
        this.lastLineWidth = null;
    }

    getCtx() {
        if (!this.ctx) {
            this.ctx = window.dom?.drawCtx;
        }
        return this.ctx;
    }

    setFrameRate(highFrameRate) {
        this.highFrameRate = highFrameRate;
        this.drawInterval = highFrameRate ? 1000 / 60 : 1000 / 30;
    }

    addCommand(type, fromX, fromY, toX, toY, color, lineWidth) {
        const idx = this.pendingCount++;
        if (idx >= this.pendingCommands.length) {
            this.pendingCommands.push({ type, fromX, fromY, toX, toY, color, lineWidth });
        } else {
            const cmd = this.pendingCommands[idx];
            cmd.type = type;
            cmd.fromX = fromX;
            cmd.fromY = fromY;
            cmd.toX = toX;
            cmd.toY = toY;
            cmd.color = color;
            cmd.lineWidth = lineWidth;
        }

        this.scheduleBatchDraw();
    }

    scheduleBatchDraw() {
        if (this.drawRafId !== null) return;

        const now = performance.now();
        const timeSinceLastDraw = now - this.lastDrawTime;

        if (timeSinceLastDraw >= this.drawInterval) {
            this.flushPending();
        } else {
            this.drawRafId = requestAnimationFrame(() => {
                this.drawRafId = null;
                this.flushPending();
            });
        }
    }

    flushPending() {
        const count = this.pendingCount;
        if (count === 0) return;
        this.pendingCount = 0;

        const ctx = this.getCtx();
        if (!ctx) return;

        this.lastDrawTime = performance.now();

        const commands = this.pendingCommands;
        let currentType = this.lastType;
        let currentColor = this.lastColor;
        let currentLineWidth = this.lastLineWidth;
        let currentPath = null;

        for (let i = 0; i < count; i++) {
            const cmd = commands[i];
            
            if (cmd.type !== currentType ||
                (cmd.type !== 'erase' && cmd.color !== currentColor) ||
                cmd.lineWidth !== currentLineWidth) {

                if (currentPath) {
                    ctx.stroke(currentPath);
                    currentPath = null;
                }

                currentType = cmd.type;
                currentColor = cmd.color;
                currentLineWidth = cmd.lineWidth;

                if (cmd.type === 'erase') {
                    ctx.globalCompositeOperation = 'destination-out';
                    ctx.strokeStyle = 'rgba(0,0,0,1)';
                    ctx.lineWidth = cmd.lineWidth;
                } else {
                    ctx.globalCompositeOperation = 'source-over';
                    ctx.strokeStyle = cmd.color || '#3498db';
                    const scale = window.getSafeScale ? window.getSafeScale() : 1;
                    ctx.lineWidth = cmd.lineWidth / scale;
                }
            }

            if (!currentPath) {
                currentPath = new Path2D();
            }
            currentPath.moveTo(cmd.fromX, cmd.fromY);
            currentPath.lineTo(cmd.toX, cmd.toY);
        }

        if (currentPath) {
            ctx.stroke(currentPath);
        }

        this.lastType = currentType;
        this.lastColor = currentColor;
        this.lastLineWidth = currentLineWidth;
    }

    _resetState() {
        this.pendingCount = 0;
        if (this.drawRafId !== null) {
            cancelAnimationFrame(this.drawRafId);
            this.drawRafId = null;
        }
        this.lastType = null;
        this.lastColor = null;
        this.lastLineWidth = null;
    }

    startDrawing() {
        this.pendingCount = 0;
        this.lastDrawTime = performance.now();
        
        const ctx = this.getCtx();
        if (ctx) {
            ctx.imageSmoothingEnabled = true;
            ctx.imageSmoothingQuality = 'high';
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
        }
    }

    endDrawing() {
        if (this.drawRafId !== null) {
            cancelAnimationFrame(this.drawRafId);
            this.drawRafId = null;
        }

        this.flushPending();
    }

    clear() {
        this._resetState();
    }
}

window.batchDrawManager = new RealtimeBatchDrawManager();
