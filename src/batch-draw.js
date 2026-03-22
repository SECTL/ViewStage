/**
 * 实时绘制管理器
 * 简单直接绘制，减少GPU开销
 */
class RealtimeBatchDrawManager {
    constructor() {
        this.ctx = null;
        this.lastType = null;
        this.lastColor = null;
        this.lastLineWidth = null;
    }
    
    getCtx() {
        if (!this.ctx) {
            this.ctx = window.dom.drawCtx;
        }
        return this.ctx;
    }
    
    /**
     * 添加绘制命令 - 直接绘制
     */
    addCommand(type, fromX, fromY, toX, toY, color, lineWidth) {
        const ctx = this.getCtx();
        if (!ctx) return;
        
        // 只在状态变化时设置
        if (type !== this.lastType) {
            if (type === 'erase') {
                ctx.globalCompositeOperation = 'destination-out';
                ctx.strokeStyle = 'rgba(0,0,0,1)';
            } else {
                ctx.globalCompositeOperation = 'source-over';
                ctx.strokeStyle = color || '#3498db';
            }
            this.lastType = type;
            this.lastColor = null; // 重置颜色缓存
        }
        
        // 只在颜色变化时设置
        if (type !== 'erase' && color !== this.lastColor) {
            ctx.strokeStyle = color;
            this.lastColor = color;
        }
        
        // 只在线宽变化时设置
        if (lineWidth !== this.lastLineWidth) {
            ctx.lineWidth = lineWidth;
            this.lastLineWidth = lineWidth;
        }
        
        // 直接绘制线段
        ctx.beginPath();
        ctx.moveTo(fromX, fromY);
        ctx.lineTo(toX, toY);
        ctx.stroke();
    }
    
    startDrawing() {
        const ctx = this.getCtx();
        if (ctx) {
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
        }
    }
    
    async endDrawing() {
        this.lastType = null;
        this.lastColor = null;
        this.lastLineWidth = null;
    }
    
    clear() {
        this.lastType = null;
        this.lastColor = null;
        this.lastLineWidth = null;
    }
}

window.batchDrawManager = new RealtimeBatchDrawManager();
