/**
 * 实时绘制管理器
 * 直接在 Canvas 上绘制，无缓冲无延迟
 */
class RealtimeBatchDrawManager {
    constructor() {
        this.ctx = null;  // 延迟初始化
        // 钢笔效果：更小的点间距，使线条更流畅
        this.minDistance = 0.3;  // 最小点间距过滤（降低以提高流畅度）
    }
    
    /**
     * 获取 ctx，延迟初始化
     */
    getCtx() {
        if (!this.ctx) {
            this.ctx = dom.drawCtx;
        }
        return this.ctx;
    }
    
    /**
     * 添加绘制命令 - 钢笔模式使用固定线宽
     */
    addCommand(type, fromX, fromY, toX, toY, color, lineWidth, fromWidth = null, toWidth = null) {
        // 计算距离，过滤太近的点
        const distance = Math.sqrt(Math.pow(toX - fromX, 2) + Math.pow(toY - fromY, 2));
        if (distance < this.minDistance) {
            return;
        }
        
        // 钢笔模式：使用固定线宽（忽略 fromWidth 和 toWidth）
        this.drawSingleLine(type, fromX, fromY, toX, toY, color, lineWidth);
    }
    
    /**
     * 绘制单条固定线宽线段
     */
    drawSingleLine(type, fromX, fromY, toX, toY, color, lineWidth) {
        const ctx = this.getCtx();
        const isErase = type === 'erase';
        
        if (isErase) {
            ctx.globalCompositeOperation = 'destination-out';
            ctx.strokeStyle = 'rgba(0, 0, 0, 1)';
        } else {
            ctx.globalCompositeOperation = 'source-over';
            ctx.strokeStyle = color;
        }
        
        ctx.lineWidth = lineWidth;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        
        ctx.beginPath();
        ctx.moveTo(fromX, fromY);
        ctx.lineTo(toX, toY);
        ctx.stroke();
    }
    
    /**
     * 开始绘制（空操作，保持接口兼容）
     */
    startDrawing() {}
    
    /**
     * 结束绘制（空操作，保持接口兼容）
     */
    async endDrawing() {}
    
    /**
     * 清空（空操作）
     */
    clear() {}
}

// 全局实时绘制管理器 - 挂载到 window 对象以便 main.js 访问
window.batchDrawManager = new RealtimeBatchDrawManager();
