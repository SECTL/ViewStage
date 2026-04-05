class RealtimeBatchDrawManager {
    constructor() {
        this.ctx = null;
        this.pendingCommands = [];
        this.drawRafId = null;
        this.highFrameRate = false;
        this.lastDrawTime = 0;
    }

    getCtx() {
        if (!this.ctx) {
            this.ctx = window.dom?.drawCtx;
        }
        return this.ctx;
    }

    setFrameRate(highFrameRate) {
        this.highFrameRate = highFrameRate;
    }

    getDrawInterval() {
        return this.highFrameRate ? 1000 / 60 : 1000 / 30;
    }

    addCommand(type, fromX, fromY, toX, toY, color, lineWidth) {
        this.pendingCommands.push({
            type, fromX, fromY, toX, toY, color, lineWidth
        });

        this.scheduleBatchDraw();
    }

    scheduleBatchDraw() {
        if (this.drawRafId !== null) return;

        const now = performance.now();
        const timeSinceLastDraw = now - this.lastDrawTime;
        const drawInterval = this.getDrawInterval();

        if (timeSinceLastDraw >= drawInterval) {
            this.flushPending();
        } else {
            this.drawRafId = requestAnimationFrame(() => {
                this.drawRafId = null;
                this.flushPending();
            });
        }
    }

    flushPending() {
        if (this.pendingCommands.length === 0) return;

        const ctx = this.getCtx();
        if (!ctx) {
            this.pendingCommands = [];
            return;
        }

        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        const commands = this.pendingCommands;
        this.pendingCommands = [];
        this.lastDrawTime = performance.now();

        let currentType = null;
        let currentColor = null;
        let currentLineWidth = null;
        let currentPath = null;

        const flushPath = () => {
            if (currentPath) {
                ctx.stroke(currentPath);
                currentPath = null;
            }
        };

        for (const cmd of commands) {
            if (cmd.type !== currentType ||
                (cmd.type !== 'erase' && cmd.color !== currentColor) ||
                cmd.lineWidth !== currentLineWidth) {

                flushPath();

                currentType = cmd.type;
                currentColor = cmd.color;
                currentLineWidth = cmd.lineWidth;

                if (cmd.type === 'erase') {
                    ctx.globalCompositeOperation = 'destination-out';
                    ctx.strokeStyle = 'rgba(0,0,0,1)';
                } else {
                    ctx.globalCompositeOperation = 'source-over';
                    ctx.strokeStyle = cmd.color || '#3498db';
                }
                ctx.lineWidth = cmd.lineWidth;
            }

            if (!currentPath) {
                currentPath = new Path2D();
            }
            currentPath.moveTo(cmd.fromX, cmd.fromY);
            currentPath.lineTo(cmd.toX, cmd.toY);
        }

        flushPath();
    }

    _resetState() {
        this.pendingCommands = [];
        if (this.drawRafId !== null) {
            cancelAnimationFrame(this.drawRafId);
            this.drawRafId = null;
        }
    }

    startDrawing() {
        this.pendingCommands = [];
        this.lastDrawTime = performance.now();
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
