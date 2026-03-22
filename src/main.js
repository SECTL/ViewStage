/**
 * ViewStage - 摄像头及PDF展台应用
 * 
 * 架构说明：
 * - 三层Canvas：背景层(bgCanvas) → 图像层(imageCanvas) → 批注层(drawCanvas)
 * - 批注系统：笔画记录 + 压缩存储 + 撤销支持
 * - 图像处理：Rust后端并行处理（增强、缩略图、旋转）
 * - 点处理：WASM高性能计算（距离计算、点简化、坐标量化、碰撞检测）
 * 
 * 性能优化策略：
 * - 智能调度：根据帧率动态调整绘制参数
 * - 批量处理：使用RAF批量绘制减少重绘次数
 * - 内存优化：使用Blob URL替代Data URL存储图片
 * - WASM加速：复杂计算移至WebAssembly
 */

// 导入WASM点处理器
import wasmPointProcessor from './wasm-processor.js';
import './batch-draw.js';

// ==================== 全局变量 ====================
let lastCanvasTransform = { x: null, y: null, scale: null };
let currentAnimationId = null;
let pendingTransform = null;
let transformRafId = null;

// 节流更新 transform（减少 DOM 操作频率）
function scheduleTransformUpdate(x, y, scale) {
    pendingTransform = { x, y, scale };
    
    if (transformRafId === null) {
        transformRafId = requestAnimationFrame(() => {
            if (pendingTransform) {
                const { x, y, scale } = pendingTransform;
                dom.canvasWrapper.style.transform = `translate(${x}px, ${y}px) scale(${scale})`;
                lastCanvasTransform.x = x;
                lastCanvasTransform.y = y;
                lastCanvasTransform.scale = scale;
            }
            transformRafId = null;
        });
    }
}

// ==================== PDF.js 配置 ====================
// PDF.js 库初始化和等待加载

function initPdfJs() {
    if (window.pdfjsLib) {
        pdfjsLib.GlobalWorkerOptions.workerSrc = 'JS/pdf.worker.min.js';
        console.log('PDF.js Worker 已配置');
        return true;
    }
    console.warn('PDF.js 库未加载');
    return false;
}

async function waitForPdfJs(maxWait = 5000) {
    const startTime = Date.now();
    while (!window.pdfjsLib && (Date.now() - startTime) < maxWait) {
        await new Promise(resolve => setTimeout(resolve, 100));
    }
    if (window.pdfjsLib) {
        initPdfJs();
        return true;
    }
    return false;
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initPdfJs);
} else {
    initPdfJs();
}

// ==================== Tauri API ====================
// Tauri 后端 API 接口封装

const { invoke } = window.__TAURI__?.core || {};
const getCurrentWindow = window.__TAURI__?.window?.getCurrentWindow;

let cacheDir = null;
let configDir = null;
let cdsDir = null;

async function initCacheDir() {
    if (window.__TAURI__) {
        try {
            cacheDir = await invoke('get_cache_dir');
            configDir = await invoke('get_config_dir');
            cdsDir = await invoke('get_cds_dir');
            console.log('缓存目录:', cacheDir);
            console.log('配置目录:', configDir);
            console.log('ViewStage目录:', cdsDir);
        } catch (error) {
            console.error('获取缓存目录失败:', error);
        }
    }
}

// ==================== 全局配置 ====================
// 绘制参数、画布尺寸、缩放限制等全局配置

const DRAW_CONFIG = {
    penColor: '#3498db',           // 默认笔色
    penWidth: 5,                   // 默认笔宽 (px)
    eraserSize: 15,                // 橡皮大小 (px)
    minScale: 0.5,                 // 最小缩放比例
    maxScale: 3,                   // 默认最大缩放比例
    maxScaleCamera: 2,             // 摄像头模式最大缩放比例
    maxScaleImage: 4,              // 图片/文档模式最大缩放比例
    canvasW: 1000,                 // 画布宽度 (逻辑像素)
    canvasH: 600,                  // 画布高度 (逻辑像素)
    screenW: 0,                    // 屏幕宽度
    screenH: 0,                    // 屏幕高度
    renderW: 1920,                 // 渲染分辨率宽度
    renderH: 1080,                 // 渲染分辨率高度
    canvasScale: 2,                // 画布相对屏幕的缩放倍数
    dpr: Math.min(window.devicePixelRatio || 1, 2),  // 设备像素比
    pdfScale: 1.5,                 // PDF 渲染缩放比例
    enhanceContrast: 1.4,          // 增强对比度
    enhanceBrightness: 10,         // 增强亮度
    enhanceSaturation: 1.2,        // 增强饱和度
    enhanceSharpen: 0,             // 增强锐化 (0-100)
    imageSmoothingQuality: 'high', // 图像平滑质量
    baseDpr: Math.min(window.devicePixelRatio || 1, 2), // 基础设备像素比
    penColors: [                   // 画笔颜色列表
        '#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6',
        '#1abc9c', '#34495e', '#e91e63', '#00bcd4', '#8bc34a',
        '#ff5722', '#673ab7', '#795548', '#000000', '#ffffff'
    ],
    // 钢笔效果配置
    penSmoothness: 0.8             // 钢笔平滑度 (0-1, 越高越平滑)
};

// 将配置暴露到全局，供 batch-draw.js 使用
window.DRAW_CONFIG = DRAW_CONFIG;

function getSafeScale() {
    return Math.max(0.001, state.scale || 1);
}

// ==================== 真实笔触效果管理器 ====================
// 根据速度和压感动态调整线宽，模拟真实笔触效果

class RealPenManager {
    constructor() {
        // 钢笔模式不需要速度计算
    }
    
    reset() {
        // 钢笔模式不需要重置
    }
    
    updatePosition(x, y, timestamp) {
        // 钢笔模式不需要速度计算
        return 0;
    }
    
    calculateLineWidth(baseWidth, velocity, pressure = 0.5) {
        // 钢笔模式：固定线宽 + 轻微压感 (0.9-1.1 倍)
        const pressureFactor = 0.9 + (pressure * 0.2);
        return baseWidth * pressureFactor;
    }
}

const realPenManager = new RealPenManager();

// ==================== 智能绘制调度器 ====================
// 根据实时帧率动态调整绘制参数，优化性能

class SmartDrawScheduler {
    constructor() {
        this.performanceHistory = []; // 性能历史记录
        this.maxHistorySize = 8;      // 最大历史记录数（减少以更快响应性能变化）
        this.targetFps = 60;          // 目标帧率
        this.minDistance = 0.7;       // 最小点间距（增加以减少点数量）
        this.maxPointsPerFlush = 40;  // 每次flush的最大点数（减少以降低单次负载）
        this.adaptationInterval = 800; // 自适应调整间隔 (ms)（减少以更快调整）
        this.lastAdaptationTime = 0;  // 上次自适应调整时间
        this.drawQuality = 'medium';  // 绘制质量设置
    }
    
    /**
     * 记录绘制性能
     */
    recordPerformance(drawTime) {
        this.performanceHistory.push(drawTime);
        if (this.performanceHistory.length > this.maxHistorySize) {
            this.performanceHistory.shift();
        }
        
        // 定期自适应调整
        const now = Date.now();
        if (now - this.lastAdaptationTime >= this.adaptationInterval) {
            this.adaptParameters();
            this.lastAdaptationTime = now;
        }
    }
    
    /**
     * 自适应调整参数
     */
    adaptParameters() {
        if (this.performanceHistory.length === 0) return;
        
        // 计算平均绘制时间
        const avgDrawTime = this.performanceHistory.reduce((sum, time) => sum + time, 0) / this.performanceHistory.length;
        const currentFps = 1000 / avgDrawTime;
        
        // 根据帧率调整参数
        if (currentFps < this.targetFps * 0.6) {
            // 性能很差，显著降低绘制质量
            this.minDistance = Math.min(this.minDistance + 0.2, 2.0);
            this.maxPointsPerFlush = Math.max(this.maxPointsPerFlush - 20, 10);
            this.drawQuality = 'low';
        } else if (currentFps < this.targetFps * 0.75) {
            // 性能较差，明显降低绘制质量
            this.minDistance = Math.min(this.minDistance + 0.15, 1.5);
            this.maxPointsPerFlush = Math.max(this.maxPointsPerFlush - 15, 15);
            this.drawQuality = 'low';
        } else if (currentFps < this.targetFps * 0.9) {
            // 性能一般，适度降低绘制质量
            this.minDistance = Math.min(this.minDistance + 0.1, 1.0);
            this.maxPointsPerFlush = Math.max(this.maxPointsPerFlush - 10, 25);
            this.drawQuality = 'medium';
        } else if (currentFps > this.targetFps * 0.95) {
            // 性能较好，提高绘制质量
            this.minDistance = Math.max(this.minDistance - 0.1, 0.5);
            this.maxPointsPerFlush = Math.min(this.maxPointsPerFlush + 10, 60);
            this.drawQuality = 'medium';
        }
        
        // 更新批处理管理器的参数
        if (batchDrawManager) {
            batchDrawManager.minDistance = this.minDistance;
        }
        
        // 根据绘制质量调整画布设置
        this.adjustDrawQuality();
    }
    
    /**
     * 根据绘制质量调整画布设置
     */
    adjustDrawQuality() {
        if (!dom.drawCtx) return;
        
        const dc = dom.drawCtx;
        
        switch (this.drawQuality) {
            case 'low':
                dc.imageSmoothingQuality = 'medium';
                break;
            case 'medium':
                dc.imageSmoothingQuality = 'high';
                break;
            case 'high':
                dc.imageSmoothingQuality = 'high';
                break;
        }
    }
    
    /**
     * 获取当前性能状态
     */
    getPerformanceState() {
        if (this.performanceHistory.length === 0) {
            return { fps: 60, status: 'normal' };
        }
        
        const avgDrawTime = this.performanceHistory.reduce((sum, time) => sum + time, 0) / this.performanceHistory.length;
        const fps = 1000 / avgDrawTime;
        
        if (fps < this.targetFps * 0.6) {
            return { fps, status: 'poor' };
        } else if (fps < this.targetFps * 0.9) {
            return { fps, status: 'fair' };
        } else {
            return { fps, status: 'good' };
        }
    }
    
    /**
     * 获取当前的最大点数限制
     */
    getMaxPointsPerFlush() {
        return this.maxPointsPerFlush;
    }
    
    /**
     * 获取当前的最小点间距
     */
    getMinDistance() {
        return this.minDistance;
    }
}

// 脏区域管理器 - 用于增量渲染
class DirtyRegionManager {
    constructor() {
        this.dirtyRects = [];
        this.maxRects = 10;
    }
    
    addRect(x, y, width, height, padding = 5) {
        const rect = {
            x: Math.max(0, x - padding),
            y: Math.max(0, y - padding),
            width: width + padding * 2,
            height: height + padding * 2
        };
        this.dirtyRects.push(rect);
        
        if (this.dirtyRects.length > this.maxRects) {
            this.mergeRects();
        }
    }
    
    addPoint(x, y, size = 5) {
        this.addRect(x - size/2, y - size/2, size, size, size/2);
    }
    
    addLine(x1, y1, x2, y2, lineWidth = 2) {
        const minX = Math.min(x1, x2);
        const minY = Math.min(y1, y2);
        const maxX = Math.max(x1, x2);
        const maxY = Math.max(y1, y2);
        this.addRect(minX, minY, maxX - minX, maxY - minY, lineWidth);
    }
    
    addStrokePoints(points, lineWidth = 2) {
        if (!points || points.length === 0) return;
        
        for (const point of points) {
            if (point.x !== undefined) {
                this.addPoint(point.x, point.y, lineWidth);
            } else {
                this.addLine(point.fromX, point.fromY, point.toX, point.toY, lineWidth);
            }
        }
    }
    
    mergeRects() {
        if (this.dirtyRects.length <= 1) return;
        
        const merged = [];
        const used = new Set();
        
        for (let i = 0; i < this.dirtyRects.length; i++) {
            if (used.has(i)) continue;
            
            let rect = { ...this.dirtyRects[i] };
            
            for (let j = i + 1; j < this.dirtyRects.length; j++) {
                if (used.has(j)) continue;
                
                const other = this.dirtyRects[j];
                if (this.rectsOverlap(rect, other)) {
                    rect = this.mergeTwoRects(rect, other);
                    used.add(j);
                }
            }
            
            merged.push(rect);
            used.add(i);
        }
        
        this.dirtyRects = merged;
    }
    
    rectsOverlap(a, b) {
        return !(a.x + a.width < b.x || 
                 b.x + b.width < a.x || 
                 a.y + a.height < b.y || 
                 b.y + b.height < a.y);
    }
    
    mergeTwoRects(a, b) {
        const x = Math.min(a.x, b.x);
        const y = Math.min(a.y, b.y);
        const width = Math.max(a.x + a.width, b.x + b.width) - x;
        const height = Math.max(a.y + a.height, b.y + b.height) - y;
        return { x, y, width, height };
    }
    
    getMergedRect() {
        if (this.dirtyRects.length === 0) return null;
        
        this.mergeRects();
        
        if (this.dirtyRects.length === 1) {
            return this.dirtyRects[0];
        }
        
        let result = this.dirtyRects[0];
        for (let i = 1; i < this.dirtyRects.length; i++) {
            result = this.mergeTwoRects(result, this.dirtyRects[i]);
        }
        return result;
    }
    
    getClipRects() {
        this.mergeRects();
        return this.dirtyRects;
    }
    
    clear() {
        this.dirtyRects = [];
    }
    
    isEmpty() {
        return this.dirtyRects.length === 0;
    }
    
    getBoundingRect() {
        if (this.dirtyRects.length === 0) return null;
        
        let minX = Infinity, minY = Infinity;
        let maxX = -Infinity, maxY = -Infinity;
        
        for (const rect of this.dirtyRects) {
            minX = Math.min(minX, rect.x);
            minY = Math.min(minY, rect.y);
            maxX = Math.max(maxX, rect.x + rect.width);
            maxY = Math.max(maxY, rect.y + rect.height);
        }
        
        return {
            x: minX,
            y: minY,
            width: maxX - minX,
            height: maxY - minY
        };
    }
}

// 全局脏区域管理器
const dirtyRegionManager = new DirtyRegionManager();

// 浅拷贝笔画数组（替代 structuredClone 提升性能）
function cloneStrokes(strokes) {
    if (!strokes || strokes.length === 0) return [];
    return strokes.map(stroke => ({
        type: stroke.type,
        points: stroke.points ? [...stroke.points] : [],
        color: stroke.color,
        lineWidth: stroke.lineWidth,
        eraserSize: stroke.eraserSize,
        bounds: stroke.bounds ? { ...stroke.bounds } : undefined,
        savedStrokeHistory: stroke.savedStrokeHistory,
        savedBaseImageURL: stroke.savedBaseImageURL
    }));
}

// 四叉树空间索引（用于快速查找与脏区域相交的笔画）
class StrokeQuadTree {
    constructor(boundary, capacity = 8, maxDepth = 6, depth = 0) {
        this.boundary = boundary;
        this.capacity = capacity;
        this.maxDepth = maxDepth;
        this.depth = depth;
        this.strokes = [];
        this.children = null;
    }
    
    insert(stroke) {
        if (!stroke.bounds) return false;
        
        if (!this.intersects(stroke.bounds)) return false;
        
        if (this.children) {
            return this.insertToChildren(stroke);
        }
        
        this.strokes.push(stroke);
        
        if (this.strokes.length > this.capacity && this.depth < this.maxDepth) {
            this.subdivide();
        }
        
        return true;
    }
    
    insertToChildren(stroke) {
        let inserted = false;
        for (const child of this.children) {
            if (child.insert(stroke)) {
                inserted = true;
            }
        }
        return inserted;
    }
    
    subdivide() {
        const { x, y, width, height } = this.boundary;
        const hw = width / 2;
        const hh = height / 2;
        
        this.children = [
            new StrokeQuadTree({ x, y, width: hw, height: hh }, this.capacity, this.maxDepth, this.depth + 1),
            new StrokeQuadTree({ x: x + hw, y, width: hw, height: hh }, this.capacity, this.maxDepth, this.depth + 1),
            new StrokeQuadTree({ x, y: y + hh, width: hw, height: hh }, this.capacity, this.maxDepth, this.depth + 1),
            new StrokeQuadTree({ x: x + hw, y: y + hh, width: hw, height: hh }, this.capacity, this.maxDepth, this.depth + 1)
        ];
        
        for (const stroke of this.strokes) {
            this.insertToChildren(stroke);
        }
        this.strokes = [];
    }
    
    query(range, found = new Set()) {
        if (!this.intersects(range)) return found;
        
        for (const stroke of this.strokes) {
            if (this.strokeIntersects(stroke, range)) {
                found.add(stroke);
            }
        }
        
        if (this.children) {
            for (const child of this.children) {
                child.query(range, found);
            }
        }
        
        return found;
    }
    
    intersects(bounds) {
        const padding = 5;
        return !(bounds.maxX + padding < this.boundary.x ||
                 bounds.minX - padding > this.boundary.x + this.boundary.width ||
                 bounds.maxY + padding < this.boundary.y ||
                 bounds.minY - padding > this.boundary.y + this.boundary.height);
    }
    
    strokeIntersects(stroke, range) {
        if (!stroke.bounds) return true;
        const padding = Math.max(stroke.lineWidth || 5, stroke.eraserSize || 5);
        return !(stroke.bounds.maxX + padding < range.x ||
                 stroke.bounds.minX - padding > range.x + range.width ||
                 stroke.bounds.maxY + padding < range.y ||
                 stroke.bounds.minY - padding > range.y + range.height);
    }
    
    clear() {
        this.strokes = [];
        this.children = null;
    }
    
    build(strokes) {
        this.clear();
        for (const stroke of strokes) {
            this.insert(stroke);
        }
    }
}

// 全局四叉树索引
let strokeQuadTree = null;

// ==================== 全局状态 ====================
// 应用状态管理：模式、画布变换、批注历史、摄像头、图像管理等

let state = {
    // 模式状态
    drawMode: 'move',              // 当前模式: 'move' | 'comment' | 'eraser'
    isDrawing: false,              // 是否正在绘制
    isDragging: false,             // 是否正在拖拽
    isScaling: false,              // 是否正在缩放 (触控)
    
    // 画布变换
    canvasX: 0,                    // 画布X偏移
    canvasY: 0,                    // 画布Y偏移
    scale: 1,                      // 当前缩放比例
    lastX: 0,                      // 上一个绘制点X
    lastY: 0,                      // 上一个绘制点Y
    
    // 摄像头视图状态（独立保存）
    cameraViewState: {
        scale: 1,
        canvasX: 0,
        canvasY: 0,
        strokeHistory: [],
        baseImageURL: null
    },
    
    // 拖拽状态
    startDragX: 0,                 // 拖拽起始X
    startDragY: 0,                 // 拖拽起始Y
    
    // 缩放状态 (触控双指)
    startScale: 1,                 // 缩放起始比例
    startDistance: 0,              // 双指起始距离
    startScaleX: 0,                // 缩放中心X
    startScaleY: 0,                // 缩放中心Y
    startCanvasX: 0,               // 缩放起始画布X
    startCanvasY: 0,               // 缩放起始画布Y
    
    // 批注历史 (撤销系统)
    strokeHistory: [],             // 笔画历史数组
    baseImageURL: null,            // 压缩后的基础图片 (base64)
    baseImageObj: null,            // 基础图片 Image 对象
    baseImageLoadId: 0,            // 用于跟踪 baseImageObj 加载状态
    currentStroke: null,           // 当前正在绘制的笔画
    MAX_UNDO_STEPS: 10,            // 最大可撤销步数
    STROKE_COMPACT_THRESHOLD: 30,  // 触发压缩的笔画阈值
    
    // 移动边界
    moveBound: {
        minX: 0, maxX: 0,
        minY: 0, maxY: 0
    },
    
    // 摄像头状态
    cameraStream: null,            // MediaStream 对象
    isCameraOpen: false,           // 摄像头是否开启
    isCameraReady: false,          // 摄像头视频是否就绪（有有效尺寸）
    isMirrored: false,             // 是否镜像 (前置摄像头)
    cameraAnimationId: null,       // requestAnimationFrame ID
    cameraRotation: 0,             // 摄像头旋转角度
    useFrontCamera: false,         // 是否使用前置摄像头
    defaultCameraId: null,         // 默认摄像头设备ID
    cameraWidth: 1280,             // 摄像头宽度
    cameraHeight: 720,             // 摄像头高度
    wasCameraOpenBeforeMinimize: false, // 最小化前摄像头是否开启
    
    // 图像增强
    enhanceEnabled: false,         // 是否启用文档增强
    
    // 图像管理
    currentImage: null,            // 当前显示的图像 Image 对象
    imageList: [],                 // 图片列表
    currentImageIndex: -1,         // 当前图片索引
    
    // PDF/文件管理
    fileList: [],                  // 文件列表 (PDF等)
    currentFolderIndex: -1,        // 当前文件夹索引
    currentFolderPageIndex: -1,    // 当前页索引
    
    // 真实笔触效果
    currentPressure: 0.5,          // 当前压感值 (0-1)
    currentVelocity: 0,            // 当前速度（钢笔模式不使用）
    currentLineWidth: 0,           // 当前动态线宽
    lastLineWidth: 0               // 上一个点的线宽
};

// ==================== 源ID管理系统 ====================
// 统一管理所有源（摄像头、图片、文档）的缩放和批注数据

// 源ID计数器
let sourceIdCounters = {
    pic: 0,      // 图片计数器
    doc: 0       // 文档计数器
};

// 当前源ID
let currentSourceId = null;

// 统一存储结构
let sourceDataStore = {};

// 生成源ID
function generateSourceId(type, pageIndex = null) {
    if (type === 'cam') {
        return 'cam';
    } else if (type === 'pic') {
        sourceIdCounters.pic++;
        return `pic-${sourceIdCounters.pic}`;
    } else if (type === 'doc') {
        if (pageIndex !== null) {
            return `doc-${sourceIdCounters.doc}-${pageIndex}`;
        }
    }
    return null;
}

// 保存当前源数据
function saveCurrentSourceData() {
    if (!currentSourceId) return;
    
    sourceDataStore[currentSourceId] = {
        scale: state.scale,
        canvasX: state.canvasX,
        canvasY: state.canvasY,
        strokeHistory: cloneStrokes(state.strokeHistory),
        baseImageURL: state.baseImageURL
    };
    
    console.log(`[源管理] 保存数据: ${currentSourceId}, 缩放: ${state.scale.toFixed(2)}, 笔画: ${state.strokeHistory.length}`);
}

// 加载指定源数据
function loadSourceData(sourceId) {
    if (!sourceId) return;
    
    const data = sourceDataStore[sourceId];
    if (data) {
        state.scale = data.scale;
        state.canvasX = data.canvasX;
        state.canvasY = data.canvasY;
        state.strokeHistory = data.strokeHistory || [];
        state.baseImageURL = data.baseImageURL;
        state.baseImageObj = null;
        
        console.log(`[源管理] 加载数据: ${sourceId}, 缩放: ${state.scale.toFixed(2)}, 笔画: ${state.strokeHistory.length}`);
    } else {
        // 新源，使用默认值
        state.scale = 1;
        state.canvasX = -(DRAW_CONFIG.canvasW - DRAW_CONFIG.screenW) / 2;
        state.canvasY = -(DRAW_CONFIG.canvasH - DRAW_CONFIG.screenH) / 2;
        state.strokeHistory = [];
        state.baseImageURL = null;
        state.baseImageObj = null;
        
        console.log(`[源管理] 新源初始化: ${sourceId}`);
    }
    
    currentSourceId = sourceId;
}

// 切换到新源
async function switchToSource(newSourceId) {
    // 保存当前源数据
    saveCurrentSourceData();
    
    // 加载新源数据
    loadSourceData(newSourceId);
    
    // 清除画布并重新渲染
    clearDrawCanvas();
    if (state.strokeHistory.length > 0) {
        await redrawAllStrokes();
    }
    
    // 更新UI
    updateMoveBound();
    clampCanvasPosition();
    updateCanvasTransform();
    updateUndoBtnStatus();
}

let dom = {};  // DOM 元素引用缓存

// 将 dom 暴露到全局，供 batch-draw.js 使用
window.dom = dom;

let cachedCanvasRect = null;  // 缓存的画布边界矩形

let offscreenCanvasPool = [];
const MAX_OFFSCREEN_CANVAS = 2;

function getOffscreenCanvas() {
    if (offscreenCanvasPool.length > 0) {
        return offscreenCanvasPool.pop();
    }
    
    const canvas = document.createElement('canvas');
    canvas.width = DRAW_CONFIG.canvasW * DRAW_CONFIG.dpr;
    canvas.height = DRAW_CONFIG.canvasH * DRAW_CONFIG.dpr;
    const ctx = canvas.getContext('2d', { alpha: true });
    ctx.scale(DRAW_CONFIG.dpr, DRAW_CONFIG.dpr);
    return { canvas, ctx };
}

function releaseOffscreenCanvas(offscreen) {
    if (offscreenCanvasPool.length < MAX_OFFSCREEN_CANVAS) {
        offscreen.ctx.setTransform(1, 0, 0, 1, 0, 0);
        offscreen.ctx.clearRect(0, 0, offscreen.canvas.width, offscreen.canvas.height);
        offscreen.ctx.scale(DRAW_CONFIG.dpr, DRAW_CONFIG.dpr);
        offscreenCanvasPool.push(offscreen);
    }
}

function invalidateCachedRect() {
    cachedCanvasRect = null;
}

function getCachedCanvasRect() {
    if (!cachedCanvasRect) {
        cachedCanvasRect = dom.canvasContainer.getBoundingClientRect();
    }
    return cachedCanvasRect;
}

// ==================== 初始化 ====================
// 应用启动入口：DOM初始化、画布初始化、事件绑定、配置加载

window.addEventListener('DOMContentLoaded', async () => {
    try {
        if (window.i18n) {
            await window.i18n.init();
        }
        
        if (window.__TAURI__) {
            const isOobeActive = await invoke('is_oobe_active');
            if (isOobeActive) {
                console.log('OOBE 激活中，跳过主窗口初始化');
                return;
            }
            listenForPdfFileOpen();
        }
        
        if (!initDOM()) {
            throw new Error('DOM 初始化失败');
        }
        initCanvas();
        bindAllEvents();
        saveSnapshot();
        
        window.addEventListener('resize', handleResize);
        
        await initCacheDir();
        
        // 检查并执行自动清除缓存
        try {
            const { invoke } = window.__TAURI__.core;
            const cleared = await invoke('check_auto_clear_cache');
            if (cleared) {
                console.log('自动清除缓存已执行');
            }
        } catch (e) {
            console.log('检查自动清除缓存失败:', e);
        }
        
        await loadCameraSetting();
        
        // 检测摄像头是否存在
        let hasCamera = false;
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            hasCamera = devices.some(device => device.kind === 'videoinput');
        } catch (e) {
            console.log('无法枚举设备:', e.name);
        }
        
        if (hasCamera) {
            try {
                await openCamera();
            } catch (error) {
                if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
                    console.log('摄像头权限被拒绝，跳过摄像头初始化');
                    showNoCameraMessage(window.i18n?.t('camera.noPermission') || '无摄像头权限');
                } else {
                    console.error('摄像头初始化失败:', error);
                    showNoCameraMessage(window.i18n?.t('camera.initFailed') || '摄像头初始化失败');
                }
            }
        } else {
            console.log('未检测到摄像头，跳过摄像头初始化');
            showNoCameraMessage(window.i18n?.t('camera.notDetected') || '未检测到摄像头');
        }
        
        console.log('画布初始化完成');
    } catch (error) {
        console.error('初始化失败:', error);
        showErrorDialog(
            window.i18n?.t('errors.initFailed') || '初始化失败',
            window.i18n?.t('errors.initFailedDesc') || '应用初始化失败，请刷新页面重试'
        );
    }
});

async function loadCameraSetting() {
    if (window.__TAURI__) {
        try {
            const { invoke } = window.__TAURI__.core;
            const settings = await invoke('get_settings');
            
            if (settings.defaultCamera) {
                state.defaultCameraId = settings.defaultCamera;
                console.log('已加载摄像头设置:', settings.defaultCamera);
            }
            
            // 加载摄像头分辨率设置
            if (settings.cameraWidth && settings.cameraHeight) {
                state.cameraWidth = settings.cameraWidth;
                state.cameraHeight = settings.cameraHeight;
                console.log('已加载摄像头分辨率:', settings.cameraWidth, 'x', settings.cameraHeight);
            }
            
            // 加载默认旋转角度
            if (settings.defaultRotation !== undefined) {
                state.cameraRotation = settings.defaultRotation;
                console.log('已加载默认旋转角度:', settings.defaultRotation, '°');
            }
            
            // 加载渲染分辨率设置
            if (settings.width && settings.height) {
                DRAW_CONFIG.renderW = settings.width;
                DRAW_CONFIG.renderH = settings.height;
                console.log('已加载渲染分辨率:', settings.width, 'x', settings.height);
            }
            
            if (settings.dprLimit) {
                DRAW_CONFIG.dpr = Math.min(window.devicePixelRatio || 1, settings.dprLimit);
                console.log('已加载设备像素比限制:', settings.dprLimit);
            }
            
            if (settings.pdfScale) {
                DRAW_CONFIG.pdfScale = settings.pdfScale;
                console.log('已加载 PDF 输出分辨率:', settings.pdfScale);
            }
            
            if (settings.contrast) {
                DRAW_CONFIG.enhanceContrast = settings.contrast;
                console.log('已加载增强对比度:', settings.contrast);
            }
            
            if (settings.brightness) {
                DRAW_CONFIG.enhanceBrightness = settings.brightness;
                console.log('已加载增强亮度:', settings.brightness);
            }
            
            if (settings.saturation) {
                DRAW_CONFIG.enhanceSaturation = settings.saturation;
                console.log('已加载增强饱和度:', settings.saturation);
            }
            
            if (settings.sharpen !== undefined) {
                DRAW_CONFIG.enhanceSharpen = settings.sharpen;
                console.log('已加载增强锐化:', settings.sharpen);
            }
            
            if (settings.penColors && Array.isArray(settings.penColors)) {
                DRAW_CONFIG.penColors = settings.penColors.map(color => {
                    if (typeof color === 'object' && color.r !== undefined) {
                        return rgbToHex(color.r, color.g, color.b);
                    }
                    return color;
                });
                console.log('已加载画笔颜色:', DRAW_CONFIG.penColors);
                updateColorButtons();
            }
        } catch (error) {
            console.error('加载摄像头设置失败:', error);
        }
    }
}

// 监听系统关联打开的PDF文件
function listenForPdfFileOpen() {
    if (!window.__TAURI__) {
        console.log('非 Tauri 环境，跳过文件打开监听');
        return;
    }
    
    console.log('开始注册文件打开事件监听...');
    
    const { listen } = window.__TAURI__.event;
    
    listen('file-opened', (event) => {
        console.log('========== 收到文件打开事件 ==========');
        console.log('完整事件对象:', JSON.stringify(event, null, 2));
        console.log('Payload 类型:', typeof event.payload);
        console.log('Payload 内容:', event.payload);
        
        let filePath = event.payload;
        
        if (typeof filePath === 'string') {
            if (filePath.startsWith('file://')) {
                filePath = decodeURIComponent(filePath.replace('file://', ''));
            }
            console.log('最终文件路径:', filePath);
            loadPdfFromPath(filePath);
        } else {
            console.error('无法解析文件路径，payload:', event.payload);
            showErrorDialog(
                window.i18n?.t('errors.fileError') || '文件错误',
                window.i18n?.t('errors.fileParseError') || '无法解析文件路径'
            );
        }
    }).then(() => {
        console.log('file-opened 事件监听注册成功');
    }).catch(err => {
        console.error('注册 file-opened 事件监听失败:', err);
    });
    
    listen('opener://open-file', (event) => {
        console.log('========== 收到 opener 事件 ==========');
        console.log('完整事件对象:', JSON.stringify(event, null, 2));
        
        let filePath = null;
        
        if (typeof event.payload === 'string') {
            filePath = event.payload;
        } else if (event.payload && typeof event.payload === 'object') {
            filePath = event.payload.path || event.payload.url || event.payload.filePath || event.payload.uri;
        }
        
        if (filePath) {
            if (filePath.startsWith('file://')) {
                filePath = decodeURIComponent(filePath.replace('file://', ''));
            }
            console.log('最终文件路径:', filePath);
            loadPdfFromPath(filePath);
        }
    }).catch(err => {
        console.log('opener 事件监听可选:', err);
    });
    
    listen('rotate-image', (event) => {
        const direction = event.payload;
        rotateImage(direction);
    }).catch(err => {
        console.error('rotate-image 事件监听失败:', err);
    });
    
    listen('mirror-changed', (event) => {
        state.isMirrored = event.payload;
        if (state.isCameraOpen) {
            updateCameraVideoStyle();
        }
        console.log('镜像状态已更改:', state.isMirrored);
    }).catch(err => {
        console.error('mirror-changed 事件监听失败:', err);
    });
    
    listen('enhance-changed', (event) => {
        state.enhanceEnabled = event.payload;
        console.log('增强状态已更改:', state.enhanceEnabled);
    }).catch(err => {
        console.error('enhance-changed 事件监听失败:', err);
    });
    
    listen('switch-camera', () => {
        switchCamera();
        console.log('切换摄像头');
    }).catch(err => {
        console.error('switch-camera 事件监听失败:', err);
    });
    
    listen('settings-changed', (event) => {
        const settings = event.payload;
        console.log('收到设置更改通知:', settings);
        
        let needRestartCamera = false;
        
        if (settings.defaultCamera !== undefined) {
            state.defaultCameraId = settings.defaultCamera;
            console.log('默认摄像头已更改:', settings.defaultCamera);
            needRestartCamera = true;
        }
        
        if (settings.cameraWidth !== undefined && settings.cameraHeight !== undefined) {
            state.cameraWidth = settings.cameraWidth;
            state.cameraHeight = settings.cameraHeight;
            console.log('摄像头分辨率已更改:', settings.cameraWidth, 'x', settings.cameraHeight);
            needRestartCamera = true;
        }
        
        if (settings.dprLimit !== undefined) {
            DRAW_CONFIG.dpr = Math.min(window.devicePixelRatio || 1, settings.dprLimit);
            console.log('设备像素比限制已更改:', settings.dprLimit);
        }
        
        if (settings.pdfScale !== undefined) {
            DRAW_CONFIG.pdfScale = settings.pdfScale;
            console.log('PDF 输出分辨率已更改:', settings.pdfScale);
        }
        
        if (settings.contrast !== undefined) {
            DRAW_CONFIG.enhanceContrast = settings.contrast;
            console.log('增强对比度已更改:', settings.contrast);
        }
        
        if (settings.brightness !== undefined) {
            DRAW_CONFIG.enhanceBrightness = settings.brightness;
            console.log('增强亮度已更改:', settings.brightness);
        }
        
        if (settings.saturation !== undefined) {
            DRAW_CONFIG.enhanceSaturation = settings.saturation;
            console.log('增强饱和度已更改:', settings.saturation);
        }
        
        if (settings.sharpen !== undefined) {
            DRAW_CONFIG.enhanceSharpen = settings.sharpen;
            console.log('增强锐化已更改:', settings.sharpen);
        }
        
        if (settings.penColors && Array.isArray(settings.penColors)) {
            DRAW_CONFIG.penColors = settings.penColors.map(color => {
                if (typeof color === 'object' && color.r !== undefined) {
                    return rgbToHex(color.r, color.g, color.b);
                }
                return color;
            });
            updateColorButtons();
            console.log('画笔颜色已更改:', DRAW_CONFIG.penColors);
        }
        
        if (needRestartCamera && state.isCameraOpen) {
            console.log('摄像头设置已更改，重新初始化摄像头...');
            setCameraState(false).then(() => {
                setTimeout(() => {
                    setCameraState(true);
                }, 300);
            });
        }
    }).catch(err => {
        console.error('settings-changed 事件监听失败:', err);
    });
}

async function processPdfPagesParallel(pdf, totalPages, batchSize = 4, docNumber = null) {
    const pages = [];
    let processedCount = 0;
    
    async function processPage(pageNum) {
        const page = await pdf.getPage(pageNum);
        const viewport = page.getViewport({ scale: DRAW_CONFIG.pdfScale });
        
        const canvas = document.createElement('canvas');
        canvas.width = viewport.width;
        canvas.height = viewport.height;
        const ctx = canvas.getContext('2d');
        
        await page.render({
            canvasContext: ctx,
            viewport: viewport
        }).promise;
        
        const fullBlob = await new Promise((resolve, reject) => {
            canvas.toBlob(blob => {
                if (blob) resolve(blob);
                else reject(new Error('Failed to create blob'));
            }, 'image/jpeg', 0.85);
        });
        const fullUrl = URL.createObjectURL(fullBlob);
        
        const thumbnail = await generateThumbnailFromCanvas(canvas, 150);
        
        processedCount++;
        updateLoadingProgress(window.i18n?.t('loading.processingPage', { current: processedCount, total: totalPages }) || `正在处理 ${processedCount}/${totalPages} 页`);
        
        // 生成源ID
        let sourceId = null;
        if (docNumber !== null) {
            sourceId = `doc-${docNumber}-${pageNum}`;
        }
        
        return {
            full: fullUrl,
            fullBlob: fullBlob,
            thumbnail: thumbnail,
            pageNum: pageNum,
            strokeHistory: null,
            baseImageURL: null,
            sourceId: sourceId
        };
    }
    
    for (let i = 1; i <= totalPages; i += batchSize) {
        const batch = [];
        for (let j = i; j <= Math.min(i + batchSize - 1, totalPages); j++) {
            batch.push(processPage(j));
        }
        const batchResults = await Promise.all(batch);
        pages.push(...batchResults);
        
        await new Promise(resolve => setTimeout(resolve, 0));
    }
    
    pages.sort((a, b) => a.pageNum - b.pageNum);
    
    return pages;
}

async function generateThumbnailFromCanvas(canvas, maxSize = 150) {
    const img = new Image();
    const blob = await new Promise((resolve, reject) => {
        canvas.toBlob(b => {
            if (b) resolve(b);
            else reject(new Error('Failed to create blob'));
        }, 'image/jpeg', 0.85);
    });
    const blobUrl = URL.createObjectURL(blob);
    
    await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
        img.src = blobUrl;
    });
    
    let thumbW, thumbH;
    if (img.width > img.height) {
        thumbW = maxSize;
        thumbH = (img.height / img.width) * maxSize;
    } else {
        thumbH = maxSize;
        thumbW = (img.width / img.height) * maxSize;
    }
    
    const thumbCanvas = document.createElement('canvas');
    thumbCanvas.width = thumbW;
    thumbCanvas.height = thumbH;
    const thumbCtx = thumbCanvas.getContext('2d');
    thumbCtx.drawImage(img, 0, 0, thumbW, thumbH);
    
    URL.revokeObjectURL(blobUrl);
    
    const thumbBlob = await new Promise((resolve, reject) => {
        thumbCanvas.toBlob(blob => {
            if (blob) resolve(blob);
            else reject(new Error('Failed to create thumbnail blob'));
        }, 'image/jpeg', 0.7);
    });
    
    return URL.createObjectURL(thumbBlob);
}

async function generateThumbnailBlob(blob, maxSize = 150) {
    const img = new Image();
    const url = URL.createObjectURL(blob);
    
    await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
        img.src = url;
    });
    
    let thumbW, thumbH;
    if (img.width > img.height) {
        thumbW = maxSize;
        thumbH = (img.height / img.width) * maxSize;
    } else {
        thumbH = maxSize;
        thumbW = (img.width / img.height) * maxSize;
    }
    
    const thumbCanvas = document.createElement('canvas');
    thumbCanvas.width = thumbW;
    thumbCanvas.height = thumbH;
    const thumbCtx = thumbCanvas.getContext('2d');
    thumbCtx.drawImage(img, 0, 0, thumbW, thumbH);
    
    URL.revokeObjectURL(url);
    
    const thumbBlob = await new Promise((resolve, reject) => {
        thumbCanvas.toBlob(blob => {
            if (blob) resolve(blob);
            else reject(new Error('Failed to create thumbnail blob'));
        }, 'image/jpeg', 0.7);
    });
    
    return URL.createObjectURL(thumbBlob);
}

async function loadPdfFromPath(filePath) {
    // 先保存当前批注数据，再关闭摄像头
    saveCurrentDrawData();
    saveCurrentFolderPageDrawData();
    
    const wasCameraOpen = state.isCameraOpen;
    
    if (state.isCameraOpen) {
        await setCameraState(false);
    }
    
    console.log('开始加载文件:', filePath);
    
    const fileName_lower = filePath.toLowerCase();
    const isWord = fileName_lower.endsWith('.docx') || fileName_lower.endsWith('.doc');
    
    if (isWord) {
        showLoadingOverlay(window.i18n?.t('loading.detectingOffice') || '正在检测 Office 软件...');
        
        const { invoke } = window.__TAURI__.core;
        const { fs } = window.__TAURI__;
        
        let detection;
        try {
            detection = await invoke('detect_office');
            console.log('Office 检测结果:', detection);
            if (detection.recommended === 'None') {
                hideLoadingOverlay();
                showErrorDialog(
                    window.i18n?.t('errors.officeNotInstalled') || 'Office 未安装',
                    window.i18n?.t('errors.officeNotInstalledDesc') || '未检测到可用的 Office 软件\n\n请安装以下软件之一：\n• Microsoft Word\n• WPS Office\n• LibreOffice\n\n或将 Word 文档另存为 PDF 后导入'
                );
                if (wasCameraOpen) await setCameraState(true);
                return;
            }
        } catch (e) {
            hideLoadingOverlay();
            console.log('检测 Office 失败:', e);
            showErrorDialog(
                window.i18n?.t('errors.officeDetectFailed') || '检测失败',
                window.i18n?.t('errors.officeDetectFailedDesc') || '检测 Office 软件失败，请重试'
            );
            if (wasCameraOpen) await setCameraState(true);
            return;
        }
        
        updateLoadingProgress(window.i18n?.t('loading.readingFile') || '正在读取文件...');
        
        let fileData;
        try {
            fileData = await fs.readFile(filePath);
        } catch (readError) {
            hideLoadingOverlay();
            console.error('文件读取失败:', readError);
            showErrorDialog(
                window.i18n?.t('errors.readFailed') || '读取失败',
                window.i18n?.t('errors.readFailedDesc') || '无法读取文件'
            );
            if (wasCameraOpen) await setCameraState(true);
            return;
        }
        
        let uint8Array;
        if (Array.isArray(fileData)) {
            uint8Array = new Uint8Array(fileData);
        } else {
            uint8Array = new Uint8Array(fileData);
        }
        
        console.log('文件大小:', uint8Array.length, '字节');
        
        updateLoadingProgress(window.i18n?.t('loading.processingWord') || '正在处理 Word 文档...');
        
        let pdfPath = null;
        try {
            pdfPath = await invoke('convert_docx_to_pdf_from_bytes', { 
                fileData: Array.from(uint8Array),
                fileName: filePath.split(/[/\\]/).pop()
            });
            console.log('Word 文档已转换为 PDF:', pdfPath);
        } catch (convertError) {
            hideLoadingOverlay();
            console.error('Word 转换失败:', convertError);
            const errorMsg = String(convertError);
            let friendlyMsg = window.i18n?.t('errors.wordConvertFailed') || 'Word 文档转换失败';
            
            if (errorMsg.includes('Office') || errorMsg.includes('Word') || errorMsg.includes('WPS')) {
                friendlyMsg = window.i18n?.t('errors.officeCallFailed') || 'Office 软件调用失败\n\n可能的原因：\n• Office 软件未正确安装\n• 文件被其他程序占用\n• 文件格式不支持';
            }
            
            showErrorDialog(
                window.i18n?.t('errors.convertFailed') || '转换失败',
                friendlyMsg,
                () => {
                    loadPdfFromPath(filePath);
                }
            );
            if (wasCameraOpen) await setCameraState(true);
            return;
        }
        
        updateLoadingProgress(window.i18n?.t('loading.renderingPage') || '正在渲染页面...');
        
        try {
            const pdfReady = await waitForPdfJs();
            if (!pdfReady) {
                hideLoadingOverlay();
                console.error('PDF.js 库加载超时');
                showErrorDialog(
                    window.i18n?.t('errors.loadFailed') || '加载失败',
                    window.i18n?.t('errors.pdfLoadTimeout') || 'PDF 库加载超时\n\n请重启应用后重试'
                );
                if (wasCameraOpen) await setCameraState(true);
                return;
            }
            
            const pdfBytes = await fs.readFile(pdfPath);
            const pdfArrayBuffer = pdfBytes.buffer;
            const pdf = await pdfjsLib.getDocument({ data: pdfArrayBuffer }).promise;
            console.log('PDF加载成功，页数:', pdf.numPages);
            
            const totalPages = pdf.numPages;
            const fileName = filePath.split(/[/\\]/).pop().replace(/\.(pdf|docx|doc)$/i, '');
            
            const folder = {
                name: fileName,
                pages: [],
                isPdf: true
            };
            
            sourceIdCounters.doc++;  // 增加文档计数器
            const docNumber = sourceIdCounters.doc;
            
            const processedPages = await processPdfPagesParallel(pdf, totalPages, 4, docNumber);
            folder.pages = processedPages;
            
            state.fileList.push(folder);
            updateFileSidebarContent();
            expandFileSidebar();
            
            if (folder.pages.length > 0) {
                const firstPage = folder.pages[0];
                const img = new Image();
                img.onload = async () => {
                    state.currentImage = img;
                    state.currentFolderIndex = state.fileList.length - 1;
                    state.currentFolderPageIndex = 0;
                    
                    // 切换到新源ID
                    if (firstPage.sourceId) {
                        await switchToSource(firstPage.sourceId);
                    }
                    
                    drawImageToCenter(img);
                    updatePhotoButtonState();
                    updateEnhanceButtonState();
                };
                img.src = firstPage.full;
            }
            
            hideLoadingOverlay();
            console.log(`文件已导入: ${folder.name}，共${folder.pages.length}页`);
            
            try {
                await fs.remove(pdfPath);
            } catch (e) {
                console.log('清理转换的 PDF 失败:', e);
            }
        } catch (error) {
            hideLoadingOverlay();
            console.error('文件导入失败:', error);
            showErrorDialog(
                window.i18n?.t('errors.importFailed') || '导入失败',
                window.i18n?.t('errors.importFailedDesc') || '文件导入失败，请确保文件格式正确'
            );
            if (wasCameraOpen) await setCameraState(true);
        }
        
        return;
    }
    
    showLoadingOverlay(window.i18n?.t('loading.importingFile') || '正在导入文件...');
    
    try {
        const pdfReady = await waitForPdfJs();
        if (!pdfReady) {
            hideLoadingOverlay();
            console.error('PDF.js 库加载超时');
            showErrorDialog(
                window.i18n?.t('errors.loadFailed') || '加载失败',
                window.i18n?.t('errors.pdfLoadTimeout') || 'PDF 库加载超时\n\n请重启应用后重试'
            );
            if (wasCameraOpen) await setCameraState(true);
            return;
        }
        
        const { fs } = window.__TAURI__;
        
        let fileData;
        try {
            fileData = await fs.readFile(filePath);
            console.log('文件读取成功，数据类型:', typeof fileData, '是否数组:', Array.isArray(fileData));
        } catch (readError) {
            console.error('文件读取失败:', readError);
            hideLoadingOverlay();
            showErrorDialog(
                window.i18n?.t('errors.readFailed') || '读取失败',
                window.i18n?.t('errors.readFailedDesc') || '无法读取文件'
            );
            if (wasCameraOpen) await setCameraState(true);
            return;
        }
        
        let uint8Array;
        if (Array.isArray(fileData)) {
            uint8Array = new Uint8Array(fileData);
        } else if (fileData instanceof ArrayBuffer) {
            uint8Array = new Uint8Array(fileData);
        } else {
            uint8Array = new Uint8Array(fileData);
        }
        
        console.log('PDF数据大小:', uint8Array.length);
        
        const pdf = await pdfjsLib.getDocument({ data: uint8Array }).promise;
        console.log('PDF加载成功，页数:', pdf.numPages);
        
        const totalPages = pdf.numPages;
        const fileName = filePath.split(/[/\\]/).pop().replace(/\.(pdf|docx|doc)$/i, '');
        
        const folder = {
            name: fileName,
            pages: [],
            isWord: false
        };
        
        sourceIdCounters.doc++;  // 增加文档计数器
        const docNumber = sourceIdCounters.doc;
        
        const processedPages = await processPdfPagesParallel(pdf, totalPages, 4, docNumber);
        folder.pages = processedPages;
        
        state.fileList.push(folder);
        updateFileSidebarContent();
        expandFileSidebar();
        
        if (folder.pages.length > 0) {
            const firstPage = folder.pages[0];
            const img = new Image();
            img.onload = async () => {
                state.currentImage = img;
                state.currentFolderIndex = state.fileList.length - 1;
                state.currentFolderPageIndex = 0;
                
                // 切换到新源ID
                if (firstPage.sourceId) {
                    await switchToSource(firstPage.sourceId);
                }
                
                drawImageToCenter(img);
                updatePhotoButtonState();
                updateEnhanceButtonState();
            };
            img.src = firstPage.full;
        }
        
        hideLoadingOverlay();
        console.log(`文件已导入: ${folder.name}，共${folder.pages.length}页`);
    } catch (error) {
        hideLoadingOverlay();
        console.error('文件导入失败:', error);
        showErrorDialog(
            window.i18n?.t('errors.importFailed') || '导入失败',
            window.i18n?.t('errors.importFailedDesc') || '文件导入失败，请确保文件格式正确'
        );
        if (wasCameraOpen) await setCameraState(true);
    }
}

// 处理窗口大小变化
async function handleResize() {
    invalidateCachedRect();
    const container = dom.canvasContainer;
    const newScreenW = container.clientWidth;
    const newScreenH = container.clientHeight;
    
    if (newScreenW !== DRAW_CONFIG.screenW || newScreenH !== DRAW_CONFIG.screenH) {
        await resizeCanvas(newScreenW, newScreenH);
    }
}

// 调整画布大小
async function resizeCanvas(newScreenW, newScreenH) {
    const oldCanvasW = DRAW_CONFIG.canvasW;
    const oldCanvasH = DRAW_CONFIG.canvasH;
    
    // 保存当前状态
    const oldScale = state.scale;
    const oldCanvasX = state.canvasX;
    const oldCanvasY = state.canvasY;
    
    DRAW_CONFIG.screenW = newScreenW;
    DRAW_CONFIG.screenH = newScreenH;
    
    // 动态调整画布缩放倍数，根据屏幕尺寸和性能状态
    let adaptiveCanvasScale = DRAW_CONFIG.canvasScale;
    if (newScreenW > 1920 || newScreenH > 1080) {
        // 大屏幕，减少画布缩放
        adaptiveCanvasScale = Math.max(1.5, DRAW_CONFIG.canvasScale * 0.8);
    } else if (newScreenW > 1366 || newScreenH > 768) {
        // 中等屏幕，保持默认缩放
        adaptiveCanvasScale = DRAW_CONFIG.canvasScale;
    } else {
        // 小屏幕，适当增加缩放
        adaptiveCanvasScale = Math.min(2.5, DRAW_CONFIG.canvasScale * 1.2);
    }
    
    DRAW_CONFIG.canvasW = Math.floor(newScreenW * adaptiveCanvasScale);
    DRAW_CONFIG.canvasH = Math.floor(newScreenH * adaptiveCanvasScale);
    
    // 使用固定的 DPR
    DRAW_CONFIG.dpr = DRAW_CONFIG.baseDpr;
    
    updateMoveBound();
    
    // 图像层：摄像头模式下不使用，跳过以提升性能
    if (!state.isCameraOpen) {
        dom.imageCanvas.width = DRAW_CONFIG.canvasW * DRAW_CONFIG.dpr;
        dom.imageCanvas.height = DRAW_CONFIG.canvasH * DRAW_CONFIG.dpr;
    }
    
    // 批注层：摄像头模式下如果没有批注，可以跳过尺寸设置
    const hasStrokes = state.strokeHistory && state.strokeHistory.length > 0;
    const hasBaseImage = state.baseImageObj !== null;
    
    if (!state.isCameraOpen || hasStrokes || hasBaseImage) {
        dom.drawCanvas.width = DRAW_CONFIG.canvasW * DRAW_CONFIG.dpr;
        dom.drawCanvas.height = DRAW_CONFIG.canvasH * DRAW_CONFIG.dpr;
    }
    
    // 所有层 CSS 尺寸相同
    dom.drawCanvas.style.width = DRAW_CONFIG.canvasW + 'px';
    dom.drawCanvas.style.height = DRAW_CONFIG.canvasH + 'px';
    
    // imageCanvas CSS 尺寸：摄像头模式下跳过
    if (!state.isCameraOpen) {
        dom.imageCanvas.style.width = DRAW_CONFIG.canvasW + 'px';
        dom.imageCanvas.style.height = DRAW_CONFIG.canvasH + 'px';
    }
    
    // 图像层上下文：摄像头模式下跳过
    if (!state.isCameraOpen) {
        dom.imageCtx.setTransform(1, 0, 0, 1, 0, 0);
        dom.imageCtx.scale(DRAW_CONFIG.dpr, DRAW_CONFIG.dpr);
    }
    
    // 批注层上下文：只在有内容时设置
    if (!state.isCameraOpen || hasStrokes || hasBaseImage) {
        dom.drawCtx.setTransform(1, 0, 0, 1, 0, 0);
        dom.drawCtx.scale(DRAW_CONFIG.dpr, DRAW_CONFIG.dpr);
    }
    
    if (!state.isCameraOpen) {
        dom.imageCtx.imageSmoothingEnabled = true;
        dom.imageCtx.imageSmoothingQuality = 'high';
    }
    
    if (!state.isCameraOpen || hasStrokes || hasBaseImage) {
        dom.drawCtx.imageSmoothingEnabled = true;
        dom.drawCtx.imageSmoothingQuality = 'high';
        dom.drawCtx.lineCap = 'round';
        dom.drawCtx.lineJoin = 'round';
        dom.drawCtx.miterLimit = 10;
    }
    
    // 重新绘制内容
    if (state.currentImage) {
        drawImageToCenter(state.currentImage);
    }
    // 摄像头模式下不需要重绘 imageCanvas，video 元素直接显示
    
    // 重新绘制批注
    if (state.strokeHistory.length > 0 || state.baseImageObj) {
        await redrawAllStrokes();
    }
    
    // 恢复画布位置和缩放
    state.scale = oldScale;
    state.canvasX = oldCanvasX;
    state.canvasY = oldCanvasY;
    
    updateMoveBound();
    clampCanvasPosition();
    updateCanvasTransform();
    
    console.log(`窗口调整: 屏幕 ${newScreenW}x${newScreenH}, 画布 ${DRAW_CONFIG.canvasW}x${DRAW_CONFIG.canvasH}, DPR ${DRAW_CONFIG.dpr.toFixed(2)}`);
}

// 初始化 DOM 元素引用
function initDOM() {
    dom.canvasContainer = document.getElementById('canvasContainer');
    dom.canvasWrapper = document.getElementById('canvasWrapper');
    dom.imageCanvas = document.getElementById('imageCanvas');
    dom.drawCanvas = document.getElementById('drawCanvas');
    dom.cameraVideo = document.getElementById('cameraVideo');
    dom.eraserHint = document.getElementById('eraserHint');
    dom.penControlPanel = document.getElementById('penControlPanel');
    dom.settingsPanel = document.getElementById('settingsPanel');
    
    dom.penSizeSliderWrapper = document.getElementById('penSizeSliderWrapper');
    dom.penSizeThumb = document.getElementById('penSizeThumb');
    dom.penSizeValue = document.getElementById('penSizeValue');
    dom.penColorPicker = document.getElementById('penColorPicker');
    dom.eraserSizeSliderWrapper = document.getElementById('eraserSizeSliderWrapper');
    dom.eraserSizeThumb = document.getElementById('eraserSizeThumb');
    dom.eraserSizeValue = document.getElementById('eraserSizeValue');
    
    dom.btnMove = document.getElementById('btnMove');
    dom.btnComment = document.getElementById('btnComment');
    dom.btnEraser = document.getElementById('btnEraser');
    dom.btnUndo = document.getElementById('btnUndo');
    dom.btnClear = document.getElementById('btnClear');
    dom.btnPhoto = document.getElementById('btnPhoto');
    dom.btnSettings = document.getElementById('btnSettings');
    dom.btnExpand = document.getElementById('btnExpand');
    dom.btnSave = document.getElementById('btnSave');
    dom.btnMinimize = document.getElementById('btnMinimize');
    dom.btnMenu = document.getElementById('btnMenu');
    dom.btnEnhance = document.getElementById('btnEnhance');
    
    if (!dom.imageCanvas || !dom.drawCanvas || !dom.canvasContainer) {
        console.error('必需的 Canvas 元素未找到');
        return false;
    }
    
    dom.imageCtx = dom.imageCanvas.getContext('2d', { alpha: true, desynchronized: true });
    dom.drawCtx = dom.drawCanvas.getContext('2d', { alpha: true, desynchronized: true });
    
    return true;
}

// ==================== 画布初始化 ====================
// 两层Canvas初始化：图像层、批注层

/**
 * 初始化画布
 * - 物理尺寸 = 逻辑尺寸 × DPR (Retina适配)
 * - CSS尺寸 = 逻辑尺寸
 */
function initCanvas() {
    const container = dom.canvasContainer;
    const screenW = container.clientWidth;
    const screenH = container.clientHeight;
    
    DRAW_CONFIG.screenW = screenW;
    DRAW_CONFIG.screenH = screenH;
    DRAW_CONFIG.canvasW = Math.floor(screenW * DRAW_CONFIG.canvasScale);
    DRAW_CONFIG.canvasH = Math.floor(screenH * DRAW_CONFIG.canvasScale);
    
    updateMoveBound();
    
    state.canvasX = -(DRAW_CONFIG.canvasW - screenW) / 2;
    state.canvasY = -(DRAW_CONFIG.canvasH - screenH) / 2;
    
    // 初始化摄像头视图状态
    state.cameraViewState = {
        scale: 1,
        canvasX: state.canvasX,
        canvasY: state.canvasY,
        strokeHistory: [],
        baseImageURL: null
    };
    
    // 图像层：物理尺寸 = CSS尺寸 × dpr
    dom.imageCanvas.width = DRAW_CONFIG.canvasW * DRAW_CONFIG.dpr;
    dom.imageCanvas.height = DRAW_CONFIG.canvasH * DRAW_CONFIG.dpr;
    
    // 批注层：物理尺寸 = CSS尺寸 × dpr
    dom.drawCanvas.width = DRAW_CONFIG.canvasW * DRAW_CONFIG.dpr;
    dom.drawCanvas.height = DRAW_CONFIG.canvasH * DRAW_CONFIG.dpr;
    
    // 所有层 CSS 尺寸相同
    dom.imageCanvas.style.width = DRAW_CONFIG.canvasW + 'px';
    dom.imageCanvas.style.height = DRAW_CONFIG.canvasH + 'px';
    dom.drawCanvas.style.width = DRAW_CONFIG.canvasW + 'px';
    dom.drawCanvas.style.height = DRAW_CONFIG.canvasH + 'px';
    
    dom.imageCtx.scale(DRAW_CONFIG.dpr, DRAW_CONFIG.dpr);
    dom.drawCtx.scale(DRAW_CONFIG.dpr, DRAW_CONFIG.dpr);
    
    dom.imageCtx.imageSmoothingEnabled = true;
    dom.imageCtx.imageSmoothingQuality = 'high';
    
    const dc = dom.drawCtx;
    dc.imageSmoothingEnabled = true;
    dc.imageSmoothingQuality = 'high';
    dc.lineCap = 'round';
    dc.lineJoin = 'round';
    dc.miterLimit = 10;
    
    setPenStyle();
    updateEraserHintSize();
    updateCanvasTransform();
    
    dom.btnMove.classList.add('primary-btn');
    
    console.log(`画布初始化: 屏幕 ${screenW}x${screenH}, 画布 ${DRAW_CONFIG.canvasW}x${DRAW_CONFIG.canvasH}`);
}

// 计算画布移动边界 (缩放后)
function updateMoveBound() {
    const screenW = DRAW_CONFIG.screenW;
    const screenH = DRAW_CONFIG.screenH;
    const scaledW = DRAW_CONFIG.canvasW * state.scale;
    const scaledH = DRAW_CONFIG.canvasH * state.scale;
    
    if (scaledW >= screenW) {
        state.moveBound.minX = -(scaledW - screenW);
        state.moveBound.maxX = 0;
    } else {
        state.moveBound.minX = (screenW - scaledW) / 2;
        state.moveBound.maxX = (screenW - scaledW) / 2;
    }
    
    if (scaledH >= screenH) {
        state.moveBound.minY = -(scaledH - screenH);
        state.moveBound.maxY = 0;
    } else {
        state.moveBound.minY = (screenH - scaledH) / 2;
        state.moveBound.maxY = (screenH - scaledH) / 2;
    }
}

function clampCanvasPosition() {
    const eps = 0.001;
    state.canvasX = Math.max(state.moveBound.minX - eps, Math.min(state.moveBound.maxX + eps, state.canvasX));
    state.canvasY = Math.max(state.moveBound.minY - eps, Math.min(state.moveBound.maxY + eps, state.canvasY));
}

// 获取当前可见区域（Canvas 坐标系）
function getVisibleRect() {
    const scale = state.scale || 1;
    const screenW = DRAW_CONFIG.screenW;
    const screenH = DRAW_CONFIG.screenH;
    
    // 计算可见区域在 Canvas 坐标系中的位置
    let visibleX = Math.max(0, -state.canvasX / scale);
    let visibleY = Math.max(0, -state.canvasY / scale);
    let visibleW = Math.min(DRAW_CONFIG.canvasW - visibleX, screenW / scale);
    let visibleH = Math.min(DRAW_CONFIG.canvasH - visibleY, screenH / scale);
    
    // 添加边距，避免边缘裁剪问题
    const padding = 10;
    visibleX = Math.max(0, visibleX - padding);
    visibleY = Math.max(0, visibleY - padding);
    visibleW = Math.min(DRAW_CONFIG.canvasW - visibleX, visibleW + padding * 2);
    visibleH = Math.min(DRAW_CONFIG.canvasH - visibleY, visibleH + padding * 2);
    
    return {
        x: visibleX,
        y: visibleY,
        width: visibleW,
        height: visibleH
    };
}

// 检查笔画是否在可见区域内
function isStrokeVisible(stroke, visibleRect) {
    if (!stroke.bounds) return true;
    
    return !(stroke.bounds.maxX < visibleRect.x ||
             stroke.bounds.minX > visibleRect.x + visibleRect.width ||
             stroke.bounds.maxY < visibleRect.y ||
             stroke.bounds.minY > visibleRect.y + visibleRect.height);
}

// 绑定所有事件
function bindAllEvents() {
    bindModeEvents();
    bindToolEvents();
    bindPenControlEvents();
    bindCanvasMouseEvents();
    bindCanvasTouchEvents();
    bindSidebarEvents();
    bindSettingsEvents();
    bindClickOutside();
}

// 设置面板事件
function bindSettingsEvents() {
    document.getElementById('btnRotateLeft')?.addEventListener('click', () => {
        rotateImage('left');
    });
    
    document.getElementById('btnRotateRight')?.addEventListener('click', () => {
        rotateImage('right');
    });
    
    dom.btnEnhance?.addEventListener('click', () => {
        toggleEnhance();
    });
}

// 点击外部关闭面板
function bindClickOutside() {
    document.addEventListener('click', (e) => {
        const panel = dom.penControlPanel;
        const isClickInsidePanel = panel.contains(e.target);
        const isClickOnBtnComment = dom.btnComment.contains(e.target);
        const isClickOnBtnEraser = dom.btnEraser.contains(e.target);
        
        if (!isClickInsidePanel && !isClickOnBtnComment && !isClickOnBtnEraser) {
            hidePenControlPanel();
        }
        
        const settingsPanel = dom.settingsPanel;
        const isClickInsideSettings = settingsPanel.contains(e.target);
        const isClickOnBtnSettings = dom.btnSettings.contains(e.target);
        
        if (!isClickInsideSettings && !isClickOnBtnSettings) {
            hideSettingsPanel();
        }
    });
}

// 模式切换事件
function bindModeEvents() {
    dom.btnMove.addEventListener('click', () => switchMode('move'));
    dom.btnComment.addEventListener('click', () => switchMode('comment'));
    dom.btnEraser.addEventListener('click', () => switchMode('eraser'));
    
    dom.btnComment.addEventListener('dblclick', (e) => {
        e.preventDefault();
        showPenControlPanel(dom.btnComment, 'comment');
    });
    
    dom.btnEraser.addEventListener('dblclick', (e) => {
        e.preventDefault();
        showPenControlPanel(dom.btnEraser, 'eraser');
    });
}

// 切换模式
function switchMode(mode) {
    state.drawMode = mode;
    
    hidePenControlPanel();
    
    [dom.btnMove, dom.btnComment, dom.btnEraser].forEach(btn => {
        btn.classList.remove('primary-btn');
    });
    
    dom.drawCanvas.classList.remove('drawing', 'eraser', 'dragging');
    
    switch (mode) {
        case 'move':
            dom.btnMove.classList.add('primary-btn');
            dom.drawCanvas.style.cursor = 'grab';
            hideEraserHint();
            break;
        case 'comment':
            dom.btnComment.classList.add('primary-btn');
            dom.drawCanvas.classList.add('drawing');
            dom.drawCanvas.style.cursor = 'crosshair';
            hideEraserHint();
            setPenStyle();
            break;
        case 'eraser':
            dom.btnEraser.classList.add('primary-btn');
            dom.drawCanvas.classList.add('eraser');
            dom.drawCanvas.style.cursor = 'none';
            showEraserHint();
            setEraserStyle();
            break;
    }
    
    console.log(`切换到 ${mode} 模式`);
}

// 工具按钮事件
function bindToolEvents() {
    dom.btnUndo.addEventListener('click', undo);
    dom.btnClear.addEventListener('click', clearAllDrawings);
    dom.btnPhoto.addEventListener('click', takePhoto);
    dom.btnSettings.addEventListener('click', openSettings);
    dom.btnSave.addEventListener('click', toggleFileSidebar);
    dom.btnMinimize.addEventListener('click', minimizeWindow);
    dom.btnMenu.addEventListener('click', toggleMenu);
}

// 菜单弹出
function toggleMenu() {
    const existingMenu = document.getElementById('menuPopup');
    if (existingMenu) {
        closeMenu();
    } else {
        showMenu();
    }
}

function showMenu() {
    const menuPopup = document.createElement('div');
    menuPopup.id = 'menuPopup';
    menuPopup.className = 'menu-popup';
    menuPopup.innerHTML = `
        <button class="menu-item" id="menuSettings">
            <img src="assets/icon/gear.svg" width="16" height="16" alt="${window.i18n?.t('toolbar.settings') || '设置'}" style="filter: invert(1);">
            ${window.i18n?.t('toolbar.settings') || '设置'}
        </button>
        <button class="menu-item menu-item-danger" id="menuClose">
            <img src="assets/icon/arrow-bar-left.svg" width="16" height="16" alt="${window.i18n?.t('common.close') || '关闭'}" style="filter: invert(1);">
            ${window.i18n?.t('common.close') || '关闭'}
        </button>
    `;
    
    dom.canvasContainer.appendChild(menuPopup);
    
    document.getElementById('menuSettings').addEventListener('click', () => {
        closeMenu();
        openSettingsWindow();
    });
    
    document.getElementById('menuClose').addEventListener('click', () => {
        closeMenu();
        closeWindow();
    });
    
    setTimeout(() => {
        document.addEventListener('click', handleMenuOutsideClick);
    }, 0);
}

function closeMenu() {
    const menuPopup = document.getElementById('menuPopup');
    if (menuPopup) {
        menuPopup.remove();
    }
    document.removeEventListener('click', handleMenuOutsideClick);
}

function handleMenuOutsideClick(e) {
    const menuPopup = document.getElementById('menuPopup');
    const btnMenu = dom.btnMenu;
    
    if (menuPopup && !menuPopup.contains(e.target) && !btnMenu.contains(e.target)) {
        closeMenu();
    }
}

// 最小化窗口
async function minimizeWindow() {
    if (getCurrentWindow) {
        const appWindow = getCurrentWindow();
        
        // 如果摄像头开启，先关闭摄像头
        if (state.isCameraOpen) {
            await setCameraState(false);
            state.wasCameraOpenBeforeMinimize = true;
            console.log('摄像头已关闭（最小化）');
        }
        
        await appWindow.minimize();
        console.log('窗口已最小化');
    } else {
        console.log('Tauri API 不可用');
    }
}

// 监听窗口最小化和恢复事件
function setupWindowMinimizeListeners() {
    if (window.__TAURI__) {
        const { getCurrentWindow } = window.__TAURI__.window;
        
        let isRestoring = false;
        
        const handleRestore = async () => {
            if (isRestoring) return;
            isRestoring = true;
            try {
                await restoreCameraIfNeeded();
            } finally {
                setTimeout(() => {
                    isRestoring = false;
                }, 300);
            }
        };
        
        getCurrentWindow().listen('tauri://restore', handleRestore);
        getCurrentWindow().listen('tauri://show', handleRestore);
        getCurrentWindow().listen('tauri://focus', handleRestore);
    }
}

// 恢复摄像头（如果需要）
async function restoreCameraIfNeeded() {
    // 如果之前摄像头是开启的，重新开启摄像头
    if (state.wasCameraOpenBeforeMinimize && !state.isCameraOpen) {
        try {
            await setCameraState(true);
            console.log('摄像头已重新开启');
            // 只有在成功开启后才重置状态
            state.wasCameraOpenBeforeMinimize = false;
        } catch (error) {
            console.error('重新开启摄像头失败:', error);
            // 开启失败时保持状态，以便下次尝试
        }
    }
}

// 关闭窗口
async function closeWindow() {
    if (getCurrentWindow) {
        const appWindow = getCurrentWindow();
        await appWindow.close();
        console.log('窗口已关闭');
    } else {
        console.log('Tauri API 不可用');
    }
}

// 笔触控制事件
function bindPenControlEvents() {
    initTriangleSlider(dom.penSizeSliderWrapper, dom.penSizeThumb, dom.penSizeValue, 2, 21, DRAW_CONFIG.penWidth, (value) => {
        DRAW_CONFIG.penWidth = value;
        if (state.drawMode === 'comment') {
            setPenStyle();
        }
    });
    
    initTriangleSlider(dom.eraserSizeSliderWrapper, dom.eraserSizeThumb, dom.eraserSizeValue, 5, 50, DRAW_CONFIG.eraserSize, (value) => {
        DRAW_CONFIG.eraserSize = value;
        updateEraserHintSize();
        if (state.drawMode === 'eraser') {
            setEraserStyle();
        }
    });
    
    // 颜色按钮点击事件
    const colorButtons = document.querySelectorAll('.pen-color-btn');
    colorButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const index = parseInt(btn.dataset.index);
            const color = DRAW_CONFIG.penColors[index];
            if (color) {
                DRAW_CONFIG.penColor = color;
                
                // 更新选中状态
                colorButtons.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                
                if (state.drawMode === 'comment') {
                    setPenStyle();
                }
            }
        });
    });
    
    // 初始化颜色按钮
    updateColorButtons();
}

// 初始化三角形滑块
function initTriangleSlider(wrapper, thumb, valueLabel, minValue, maxValue, initialValue, onChange) {
    const wrapperHeight = 50;
    const thumbHeight = 18;
    const validHeight = wrapperHeight - thumbHeight;
    
    let currentValue = initialValue;
    let isDragging = false;
    
    function updateThumbPosition() {
        const ratio = (currentValue - minValue) / (maxValue - minValue);
        const top = (1 - ratio) * validHeight;
        thumb.style.top = `${top}px`;
        valueLabel.textContent = `${currentValue}px`;
    }
    
    function onDrag(e) {
        if (!isDragging) return;
        const mouseY = e.clientY - wrapper.getBoundingClientRect().top;
        const clampedY = Math.max(0, Math.min(mouseY, validHeight));
        const ratio = 1 - (clampedY / validHeight);
        currentValue = Math.round(minValue + ratio * (maxValue - minValue));
        updateThumbPosition();
        if (onChange) onChange(currentValue);
    }
    
    function stopDrag() {
        isDragging = false;
        document.removeEventListener('mousemove', onDrag);
        document.removeEventListener('mouseup', stopDrag);
    }
    
    thumb.addEventListener('mousedown', (e) => {
        e.preventDefault();
        isDragging = true;
        document.addEventListener('mousemove', onDrag);
        document.addEventListener('mouseup', stopDrag);
    });
    
    wrapper.addEventListener('click', (e) => {
        if (isDragging) return;
        const clickY = e.clientY - wrapper.getBoundingClientRect().top;
        const ratio = 1 - Math.max(0, Math.min(clickY / validHeight, 1));
        currentValue = Math.round(minValue + ratio * (maxValue - minValue));
        updateThumbPosition();
        if (onChange) onChange(currentValue);
    });
    
    updateThumbPosition();
}

// RGB转十六进制颜色
function rgbToHex(r, g, b) {
    return '#' + [r, g, b].map(x => {
        const hex = x.toString(16);
        return hex.length === 1 ? '0' + hex : hex;
    }).join('');
}

// 十六进制颜色转RGB
function hexToRgb(hex) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16)
    } : null;
}

function updateColorButtons() {
    const colorButtons = document.querySelectorAll('.pen-color-btn');
    colorButtons.forEach((btn, index) => {
        if (DRAW_CONFIG.penColors[index]) {
            btn.dataset.color = DRAW_CONFIG.penColors[index];
            btn.style.backgroundColor = DRAW_CONFIG.penColors[index];
            btn.title = window.i18n?.t('settings.colorN', { n: index + 1 }) || `颜色${index + 1}`;
            
            // 为黑色和白色添加边框
            if (DRAW_CONFIG.penColors[index].toLowerCase() === '#000000' || 
                DRAW_CONFIG.penColors[index].toLowerCase() === '#ffffff') {
                btn.style.border = '1px solid #555';
            } else {
                btn.style.border = 'none';
            }
        }
    });
    updateColorButtonActive();
}

function updateColorButtonActive() {
    const colorButtons = document.querySelectorAll('.pen-color-btn');
    colorButtons.forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.color === DRAW_CONFIG.penColor) {
            btn.classList.add('active');
        }
    });
}

// 设置笔触样式
function setPenStyle() {
    const dc = dom.drawCtx;
    dc.strokeStyle = DRAW_CONFIG.penColor;
    dc.lineWidth = DRAW_CONFIG.penWidth;
    dc.lineCap = 'round';
    dc.lineJoin = 'round';
    dc.miterLimit = 10;
    dc.globalCompositeOperation = 'source-over';
}

// 设置橡皮样式
function setEraserStyle() {
    const dc = dom.drawCtx;
    dc.lineWidth = DRAW_CONFIG.eraserSize;
    dc.lineCap = 'round';
    dc.lineJoin = 'round';
    dc.miterLimit = 10;
    dc.globalCompositeOperation = 'destination-out';
}

// 橡皮提示框
function updateEraserHintSize() {
    const size = DRAW_CONFIG.eraserSize;
    // 橡皮擦大小基于 Canvas 坐标系，显示时需要考虑缩放
    dom.eraserHint.style.width = `${size}px`;
    dom.eraserHint.style.height = `${size}px`;
}

function showEraserHint() {
    dom.eraserHint.classList.add('active');
}

function hideEraserHint() {
    dom.eraserHint.classList.remove('active');
}

function showPenControlPanel(targetBtn, mode) {
    const panel = dom.penControlPanel;
    const btnRect = targetBtn.getBoundingClientRect();
    const containerRect = document.querySelector('.main-function').getBoundingClientRect();
    
    const penSizeControl = panel.querySelector('.pen-size-vertical:nth-child(1)');
    const colorButtons = panel.querySelector('.pen-color-buttons');
    const eraserSizeControl = panel.querySelector('.pen-size-vertical:nth-child(3)');
    
    if (mode === 'comment') {
        if (penSizeControl) penSizeControl.style.display = 'flex';
        if (colorButtons) colorButtons.style.display = 'grid';
        if (eraserSizeControl) eraserSizeControl.style.display = 'none';
    } else if (mode === 'eraser') {
        if (penSizeControl) penSizeControl.style.display = 'none';
        if (colorButtons) colorButtons.style.display = 'none';
        if (eraserSizeControl) eraserSizeControl.style.display = 'flex';
    }
    
    // 重置面板位置和可见性，确保尺寸计算准确
    panel.style.position = 'absolute';
    panel.style.bottom = 'auto';
    panel.style.top = 'auto';
    panel.style.right = 'auto';
    panel.style.left = 'auto';
    panel.style.visibility = 'hidden';
    panel.style.opacity = '0';
    panel.classList.remove('visible');
    
    // 强制浏览器重排，确保获取准确的尺寸
    panel.offsetHeight;
    
    // 获取实际面板尺寸
    const panelWidth = panel.offsetWidth || (mode === 'comment' ? 240 : 120);
    const panelHeight = panel.offsetHeight || 120;
    
    // 计算面板位置，确保居中对齐按钮
    let left = btnRect.left - containerRect.left + (btnRect.width / 2) - (panelWidth / 2);
    let top = btnRect.top - containerRect.top - panelHeight - 15;
    
    // 边界检查，确保面板不超出容器
    const containerPadding = 10;
    left = Math.max(containerPadding, Math.min(left, containerRect.width - panelWidth - containerPadding));
    
    // 如果面板顶部超出容器，显示在按钮下方
    if (top < containerPadding) {
        top = btnRect.bottom - containerRect.top + 15;
    }
    
    // 设置最终位置并显示面板
    panel.style.left = `${left}px`;
    panel.style.top = `${top}px`;
    panel.style.visibility = 'visible';
    panel.style.opacity = '1';
    panel.classList.add('visible');
}

function hidePenControlPanel() {
    dom.penControlPanel.classList.remove('visible');
    dom.penControlPanel.style.opacity = '0';
    dom.penControlPanel.style.visibility = 'hidden';
}

function updateEraserHintPos(clientX, clientY) {
    const rect = getCachedCanvasRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    dom.eraserHint.style.left = `${x}px`;
    dom.eraserHint.style.top = `${y}px`;
    dom.eraserHint.style.transform = `translate(-50%, -50%) scale(${state.scale})`;
}

// ==================== 画布交互事件 ====================
// 鼠标、触控事件处理：绘制、拖拽、缩放

function bindCanvasMouseEvents() {
    // 优先使用 Pointer Events（支持压感）
    if (window.PointerEvent) {
        dom.drawCanvas.addEventListener('pointerdown', handlePointerDown);
        dom.drawCanvas.addEventListener('pointermove', handlePointerMove);
        dom.drawCanvas.addEventListener('pointerup', handlePointerUp);
        dom.drawCanvas.addEventListener('pointerleave', handlePointerLeave);
        dom.drawCanvas.addEventListener('pointercancel', handlePointerUp);
    } else {
        // 降级到传统鼠标事件
        dom.drawCanvas.addEventListener('mousedown', handleMouseDown);
        dom.drawCanvas.addEventListener('mousemove', handleMouseMove);
        dom.drawCanvas.addEventListener('mouseup', handleMouseUp);
        dom.drawCanvas.addEventListener('mouseleave', handleMouseLeave);
    }
    dom.drawCanvas.addEventListener('wheel', handleWheel, { passive: false });
}

/**
 * Pointer 按下处理
 */
function handlePointerDown(e) {
    e.preventDefault();
    const rect = dom.drawCanvas.getBoundingClientRect();
    
    // 保存压感值
    state.currentPressure = e.pressure || 0.5;
    
    if (state.drawMode === 'move') {
        state.isDragging = true;
        state.startDragX = e.clientX - state.canvasX;
        state.startDragY = e.clientY - state.canvasY;
        dom.canvasWrapper.classList.add('dragging');
        dom.drawCanvas.classList.add('dragging');
    } else if (state.drawMode === 'comment') {
        hidePenControlPanel();
        state.isDrawing = true;
        state.lastX = (e.clientX - rect.left) / getSafeScale();
        state.lastY = (e.clientY - rect.top) / getSafeScale();
        startStroke('draw');
    } else if (state.drawMode === 'eraser') {
        hidePenControlPanel();
        state.isDrawing = true;
        state.lastX = (e.clientX - rect.left) / getSafeScale();
        state.lastY = (e.clientY - rect.top) / getSafeScale();
        startStroke('erase');
    }
}

/**
 * Pointer 移动处理
 */
function handlePointerMove(e) {
    e.preventDefault();
    
    // 更新压感值
    state.currentPressure = e.pressure || 0.5;
    
    if (state.drawMode === 'eraser') {
        updateEraserHintPos(e.clientX, e.clientY);
    }
    
    if (state.isDragging) {
        state.canvasX = e.clientX - state.startDragX;
        state.canvasY = e.clientY - state.startDragY;
        clampCanvasPosition();
        
        const transform = `translate(${state.canvasX}px, ${state.canvasY}px) scale(${state.scale})`;
        dom.canvasWrapper.style.transform = transform;
        
        if (dom.cameraVideo && state.isCameraOpen && state.isCameraReady) {
            updateCameraVideoStyle();
        }
        
        lastCanvasTransform.x = state.canvasX;
        lastCanvasTransform.y = state.canvasY;
        lastCanvasTransform.scale = state.scale;
    } else if (state.isDrawing) {
        const rect = dom.drawCanvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) / getSafeScale();
        const y = (e.clientY - rect.top) / getSafeScale();
        
        const dx = x - state.lastX;
        const dy = y - state.lastY;
        const distSq = dx * dx + dy * dy;
        
        // 增大距离阈值到2px，减少GPU绘制调用
        if (distSq > 4) {
            // 先调用 addStrokePoint 来计算动态线宽
            addStrokePoint(state.lastX, state.lastY, x, y, state.currentPressure);
            
            // 直接绘制，不收集点
            const type = state.drawMode === 'eraser' ? 'erase' : 'draw';
            const color = state.drawMode === 'comment' ? DRAW_CONFIG.penColor : '#000000';
            const lineWidth = state.drawMode === 'comment' ? DRAW_CONFIG.penWidth : DRAW_CONFIG.eraserSize;
            
            batchDrawManager.addCommand(
                type, 
                state.lastX, 
                state.lastY, 
                x, 
                y, 
                color, 
                lineWidth
            );
            
            state.lastX = x;
            state.lastY = y;
        }
    }
}

/**
 * Pointer 抬起处理
 */
async function handlePointerUp(e) {
    if (state.isDragging) {
        state.isDragging = false;
        dom.canvasWrapper.classList.remove('dragging');
        dom.drawCanvas.classList.remove('dragging');
    }
    if (state.isDrawing) {
        state.isDrawing = false;
        if (state.drawRafId) {
            cancelAnimationFrame(state.drawRafId);
            state.drawRafId = null;
        }
        await flushDrawPoints();
        await endStroke();
    }
}

/**
 * Pointer 离开处理
 */
async function handlePointerLeave(e) {
    if (state.isDragging) {
        state.isDragging = false;
        dom.drawCanvas.classList.remove('dragging');
    }
    if (state.isDrawing) {
        state.isDrawing = false;
        if (state.drawRafId) {
            cancelAnimationFrame(state.drawRafId);
            state.drawRafId = null;
        }
        await flushDrawPoints();
        await endStroke();
    }
}

/**
 * 鼠标按下处理
 * - move模式: 开始拖拽
 * - comment模式: 开始绘制笔画
 * - eraser模式: 开始擦除
 */
function handleMouseDown(e) {
    e.preventDefault();
    const rect = dom.drawCanvas.getBoundingClientRect();
    
    if (state.drawMode === 'move') {
        state.isDragging = true;
        state.startDragX = e.clientX - state.canvasX;
        state.startDragY = e.clientY - state.canvasY;
        dom.canvasWrapper.classList.add('dragging');
        dom.drawCanvas.classList.add('dragging');
    } else if (state.drawMode === 'comment') {
        hidePenControlPanel();
        state.isDrawing = true;
        state.lastX = (e.clientX - rect.left) / getSafeScale();
        state.lastY = (e.clientY - rect.top) / getSafeScale();
        startStroke('draw');
    } else if (state.drawMode === 'eraser') {
        hidePenControlPanel();
        state.isDrawing = true;
        state.lastX = (e.clientX - rect.left) / getSafeScale();
        state.lastY = (e.clientY - rect.top) / getSafeScale();
        startStroke('erase');
    }
}

/**
 * 鼠标移动处理
 * - 拖拽: 更新画布位置
 * - 绘制: 收集点并批量绘制 (RAF优化)
 */
function handleMouseMove(e) {
    e.preventDefault();
    
    if (state.drawMode === 'eraser') {
        updateEraserHintPos(e.clientX, e.clientY);
    }
    
    if (state.isDragging) {
        state.canvasX = e.clientX - state.startDragX;
        state.canvasY = e.clientY - state.startDragY;
        clampCanvasPosition();
        
        // 直接更新容器 transform，提高跟手性
        const transform = `translate(${state.canvasX}px, ${state.canvasY}px) scale(${state.scale})`;
        dom.canvasWrapper.style.transform = transform;
        
        // 更新 video 元素
        if (dom.cameraVideo && state.isCameraOpen && state.isCameraReady) {
            updateCameraVideoStyle();
        }
        
        // 更新 lastCanvasTransform
        lastCanvasTransform.x = state.canvasX;
        lastCanvasTransform.y = state.canvasY;
        lastCanvasTransform.scale = state.scale;
    } else if (state.isDrawing) {
        const rect = dom.drawCanvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) / getSafeScale();
        const y = (e.clientY - rect.top) / getSafeScale();
        
        const dx = x - state.lastX;
        const dy = y - state.lastY;
        const distSq = dx * dx + dy * dy;
        
        // 钢笔效果：更小的距离阈值，使线条更流畅（0.3px）
        if (distSq > 0.09) {
            // 先调用 addStrokePoint 来计算动态线宽
            addStrokePoint(state.lastX, state.lastY, x, y);
            
            // 直接绘制，不收集点
            const type = state.drawMode === 'eraser' ? 'erase' : 'draw';
            const color = state.drawMode === 'comment' ? DRAW_CONFIG.penColor : '#000000';
            const lineWidth = state.drawMode === 'comment' ? DRAW_CONFIG.penWidth : DRAW_CONFIG.eraserSize;
            
            batchDrawManager.addCommand(
                type, 
                state.lastX, 
                state.lastY, 
                x, 
                y, 
                color, 
                lineWidth,
                state.lastLineWidth,
                state.currentLineWidth
            );
            
            state.lastX = x;
            state.lastY = y;
        }
    }
}

async function flushDrawPoints() {
    // 不再需要此函数，绘制已即时完成
}

async function handleMouseUp(e) {
    if (state.isDragging) {
        state.isDragging = false;
        dom.canvasWrapper.classList.remove('dragging');
        dom.drawCanvas.classList.remove('dragging');
    }
    if (state.isDrawing) {
        state.isDrawing = false;
        await batchDrawManager.endDrawing();
        await endStroke();
    }
}

async function handleMouseLeave(e) {
    if (state.isDragging) {
        state.isDragging = false;
        dom.canvasWrapper.classList.remove('dragging');
        dom.drawCanvas.classList.remove('dragging');
    }
    if (state.isDrawing) {
        state.isDrawing = false;
        await batchDrawManager.endDrawing();
        await endStroke();
    }
}

function handleWheel(e) {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.1 : 0.1;
    const maxScale = state.isCameraOpen ? DRAW_CONFIG.maxScaleCamera : DRAW_CONFIG.maxScaleImage;
    const newScale = Math.max(DRAW_CONFIG.minScale, Math.min(maxScale, state.scale + delta));
    
    if (newScale !== state.scale) {
        const containerRect = dom.canvasContainer.getBoundingClientRect();
        const mouseX = e.clientX - containerRect.left;
        const mouseY = e.clientY - containerRect.top;
        
        const oldScale = state.scale;
        
        updateMoveBound();
        
        const scaleRatio = newScale / oldScale;
        const targetX = mouseX - (mouseX - state.canvasX) * scaleRatio;
        const targetY = mouseY - (mouseY - state.canvasY) * scaleRatio;
        
        state.scale = newScale;
        state.canvasX = targetX;
        state.canvasY = targetY;
        
        clampCanvasPosition();
        animateCanvasTransform(state.canvasX, state.canvasY, state.scale, 100);
    }
}

// 画布触控事件
function bindCanvasTouchEvents() {
    dom.drawCanvas.addEventListener('touchstart', handleTouchStart, { passive: false });
    dom.drawCanvas.addEventListener('touchmove', handleTouchMove, { passive: false });
    dom.drawCanvas.addEventListener('touchend', handleTouchEnd, { passive: false });
    dom.drawCanvas.addEventListener('touchcancel', handleTouchEnd, { passive: false });
}

function handleTouchStart(e) {
    e.preventDefault();
    const touches = e.touches;
    const rect = dom.drawCanvas.getBoundingClientRect();
    
    if (touches.length === 1) {
        const touch = touches[0];
        if (state.drawMode === 'move') {
            state.isDragging = true;
            state.startDragX = touch.clientX - state.canvasX;
            state.startDragY = touch.clientY - state.canvasY;
            dom.canvasWrapper.classList.add('dragging');
            dom.drawCanvas.classList.add('dragging');
        } else if (state.drawMode === 'comment') {
            state.isDrawing = true;
            state.lastX = (touch.clientX - rect.left) / getSafeScale();
            state.lastY = (touch.clientY - rect.top) / getSafeScale();
            startStroke('draw');
        } else if (state.drawMode === 'eraser') {
            state.isDrawing = true;
            updateEraserHintPos(touch.clientX, touch.clientY);
            state.lastX = (touch.clientX - rect.left) / getSafeScale();
            state.lastY = (touch.clientY - rect.top) / getSafeScale();
            startStroke('erase');
        }
    } else if (touches.length === 2) {
        state.isScaling = true;
        state.isDragging = false;
        state.isDrawing = false;
        state.startDistance = getTouchDistance(touches[0], touches[1]);
        state.startScale = state.scale;
        state.startScaleX = (touches[0].clientX + touches[1].clientX) / 2;
        state.startScaleY = (touches[0].clientY + touches[1].clientY) / 2;
        state.startCanvasX = state.canvasX;
        state.startCanvasY = state.canvasY;
        dom.canvasWrapper.classList.add('dragging');
    }
}

function handleTouchMove(e) {
    e.preventDefault();
    const touches = e.touches;
    const rect = dom.drawCanvas.getBoundingClientRect();
    
    if (touches.length === 1 && state.isDragging) {
        const touch = touches[0];
        state.canvasX = touch.clientX - state.startDragX;
        state.canvasY = touch.clientY - state.startDragY;
        clampCanvasPosition();
        
        // 直接更新容器 transform，提高跟手性
        const transform = `translate(${state.canvasX}px, ${state.canvasY}px) scale(${state.scale})`;
        dom.canvasWrapper.style.transform = transform;
        
        // 更新 video 元素
        if (dom.cameraVideo && state.isCameraOpen && state.isCameraReady) {
            updateCameraVideoStyle();
        }
        
        // 更新 lastCanvasTransform
        lastCanvasTransform.x = state.canvasX;
        lastCanvasTransform.y = state.canvasY;
        lastCanvasTransform.scale = state.scale;
    } else if (touches.length === 1 && state.isDrawing) {
        const touch = touches[0];
        if (state.drawMode === 'eraser') {
            updateEraserHintPos(touch.clientX, touch.clientY);
        }
        
        const x = (touch.clientX - rect.left) / getSafeScale();
        const y = (touch.clientY - rect.top) / getSafeScale();
        
        // 获取触控压感（如果有的话）
        const pressure = (touch.force > 0) ? touch.force : 0.5;
        
        const dx = x - state.lastX;
        const dy = y - state.lastY;
        const distSq = dx * dx + dy * dy;
        
        // 钢笔效果：更小的距离阈值，使线条更流畅（0.3px）
        if (distSq > 0.09) {
            // 先调用 addStrokePoint 来计算动态线宽
            addStrokePoint(state.lastX, state.lastY, x, y, pressure);
            
            // 直接绘制，不收集点
            const type = state.drawMode === 'eraser' ? 'erase' : 'draw';
            const color = state.drawMode === 'comment' ? DRAW_CONFIG.penColor : '#000000';
            const lineWidth = state.drawMode === 'comment' ? DRAW_CONFIG.penWidth : DRAW_CONFIG.eraserSize;
            
            batchDrawManager.addCommand(
                type, 
                state.lastX, 
                state.lastY, 
                x, 
                y, 
                color, 
                lineWidth,
                state.lastLineWidth,
                state.currentLineWidth
            );
            
            state.lastX = x;
            state.lastY = y;
        }
    } else if (touches.length === 2 && state.isScaling) {
        const currentDistance = getTouchDistance(touches[0], touches[1]);
        const scaleRatio = currentDistance / state.startDistance;
        let newScale = state.startScale * scaleRatio;
        const maxScale = state.isCameraOpen ? DRAW_CONFIG.maxScaleCamera : DRAW_CONFIG.maxScaleImage;
        newScale = Math.max(DRAW_CONFIG.minScale, Math.min(maxScale, newScale));
        
        const centerX = (touches[0].clientX + touches[1].clientX) / 2;
        const centerY = (touches[0].clientY + touches[1].clientY) / 2;
        
        const finalRatio = newScale / state.startScale;
        state.canvasX = centerX - (state.startScaleX - state.startCanvasX) * finalRatio;
        state.canvasY = centerY - (state.startScaleY - state.startCanvasY) * finalRatio;
        state.scale = newScale;
        
        updateMoveBound();
        clampCanvasPosition();
        
        // 使用节流更新 transform
        scheduleTransformUpdate(state.canvasX, state.canvasY, state.scale);
        
        // 更新 video 元素
        if (dom.cameraVideo && state.isCameraOpen && state.isCameraReady) {
            updateCameraVideoStyle();
        }
    }
}

async function handleTouchEnd(e) {
    e.preventDefault();
    
    if (e.touches.length === 0) {
        state.isDragging = false;
        state.isScaling = false;
        dom.canvasWrapper.classList.remove('dragging');
        dom.drawCanvas.classList.remove('dragging');
        
        if (state.isDrawing) {
            state.isDrawing = false;
            if (state.drawRafId) {
                cancelAnimationFrame(state.drawRafId);
                state.drawRafId = null;
            }
            await flushDrawPoints();
            await endStroke();
        }
    } else if (e.touches.length === 1) {
        state.isScaling = false;
        const touch = e.touches[0];
        if (state.drawMode === 'move') {
            state.isDragging = true;
            state.startDragX = touch.clientX - state.canvasX;
            state.startDragY = touch.clientY - state.canvasY;
        }
    }
}

function getTouchDistance(touch1, touch2) {
    const dx = touch2.clientX - touch1.clientX;
    const dy = touch2.clientY - touch1.clientY;
    return Math.max(1, Math.sqrt(dx * dx + dy * dy));
}

function updateCanvasTransform() {
    if (lastCanvasTransform.x === state.canvasX && 
        lastCanvasTransform.y === state.canvasY && 
        lastCanvasTransform.scale === state.scale) {
        return;
    }
    
    lastCanvasTransform.x = state.canvasX;
    lastCanvasTransform.y = state.canvasY;
    lastCanvasTransform.scale = state.scale;
    
    // 只对容器设置 transform，减少 DOM 操作
    const transform = `translate(${state.canvasX}px, ${state.canvasY}px) scale(${state.scale})`;
    dom.canvasWrapper.style.transform = transform;
    
    // video 元素需要单独处理（因为它有额外的偏移）
    if (dom.cameraVideo && state.isCameraOpen && state.isCameraReady) {
        updateCameraVideoStyle();
    }
}

function animateCanvasTransform(targetX, targetY, targetScale, duration = 250) {
    // 取消之前的动画
    if (currentAnimationId !== null) {
        cancelAnimationFrame(currentAnimationId);
        currentAnimationId = null;
    }
    
    const startX = state.canvasX;
    const startY = state.canvasY;
    const startScale = state.scale;
    
    const deltaX = targetX - startX;
    const deltaY = targetY - startY;
    const deltaScale = targetScale - startScale;
    
    const startTime = performance.now();
    
    dom.canvasWrapper.classList.add('smooth-transform');
    
    function animate(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        const easeProgress = 1 - Math.pow(1 - progress, 3);
        
        state.canvasX = startX + deltaX * easeProgress;
        state.canvasY = startY + deltaY * easeProgress;
        state.scale = startScale + deltaScale * easeProgress;
        
        updateMoveBound();
        clampCanvasPosition();
        
        lastCanvasTransform.x = state.canvasX;
        lastCanvasTransform.y = state.canvasY;
        lastCanvasTransform.scale = state.scale;
        
        // 只对容器设置 transform
        const transform = `translate(${state.canvasX}px, ${state.canvasY}px) scale(${state.scale})`;
        dom.canvasWrapper.style.transform = transform;
        
        // 更新 video 元素
        if (dom.cameraVideo && state.isCameraOpen && state.isCameraReady) {
            updateCameraVideoStyle();
        }
        
        if (progress < 1) {
            currentAnimationId = requestAnimationFrame(animate);
        } else {
            currentAnimationId = null;
            dom.canvasWrapper.classList.remove('smooth-transform');
        }
    }
    
    currentAnimationId = requestAnimationFrame(animate);
}

// 撤销功能 - 混合方案：路径记录 + ImageData 压缩
function startStroke(type) {
    state.currentStroke = {
        type: type,
        points: [],
        color: DRAW_CONFIG.penColor,
        lineWidth: DRAW_CONFIG.penWidth,
        eraserSize: DRAW_CONFIG.eraserSize,
        // 边界框（用于脏区域渲染）
        bounds: {
            minX: Infinity,
            minY: Infinity,
            maxX: -Infinity,
            maxY: -Infinity
        },
        // 钢笔模式：不需要可变线宽数据
        variableWidths: null
    };
    
    // 重置状态
    state.currentPressure = 0.5;
    state.currentLineWidth = DRAW_CONFIG.penWidth;
    state.lastLineWidth = DRAW_CONFIG.penWidth;
    
    // 开始批处理绘制
    batchDrawManager.startDrawing();
}

function addStrokePoint(fromX, fromY, toX, toY, pressure = 0.5) {
    if (state.currentStroke) {
        // 更新边界框
        const bounds = state.currentStroke.bounds;
        bounds.minX = Math.min(bounds.minX, fromX, toX);
        bounds.minY = Math.min(bounds.minY, fromY, toY);
        bounds.maxX = Math.max(bounds.maxX, fromX, toX);
        bounds.maxY = Math.max(bounds.maxY, fromY, toY);
        
        // 钢笔模式：计算轻微压感变化的线宽
        if (state.currentStroke.type === 'draw') {
            state.currentPressure = pressure;
            const baseWidth = state.currentStroke.lineWidth;
            
            state.lastLineWidth = state.currentLineWidth;
            state.currentLineWidth = realPenManager.calculateLineWidth(
                baseWidth,
                0,  // 钢笔模式不需要速度
                pressure
            );
        }
        
        // 检查是否需要添加连接点
        if (state.currentStroke.points.length > 0) {
            const lastPoint = state.currentStroke.points[state.currentStroke.points.length - 1];
            const gapDistance = distance(lastPoint.toX, lastPoint.toY, fromX, fromY);
            
            // 如果距离过大，添加连接点
            if (gapDistance > 1.5) { // 1.5px 阈值
                state.currentStroke.points.push({
                    fromX: lastPoint.toX,
                    fromY: lastPoint.toY,
                    toX: fromX,
                    toY: fromY
                });
            }
        }
        
        state.currentStroke.points.push({ fromX, fromY, toX, toY });
    }
}

async function endStroke() {
    if (state.currentStroke && state.currentStroke.points.length > 0) {
        if (state.currentStroke.type === 'erase') {
            await processEraserStroke(state.currentStroke);
        } else {
            // 钢笔模式：直接保存笔画，不需要复杂的点优化
            state.strokeHistory.push(state.currentStroke);
            
            if (state.strokeHistory.length > state.STROKE_COMPACT_THRESHOLD) {
                compactStrokes();
            }
            
            updateUndoBtnStatus();
        }
    }
    state.currentStroke = null;
    
    await batchDrawManager.endDrawing();
    
    batchDrawManager.clear();
}

/**
 * 处理橡皮擦笔画 - 只在实际擦除到内容时记录步骤
 */
async function processEraserStroke(eraserStroke) {
    // 检测橡皮擦路径是否与现有笔画相交
    const hasIntersection = checkEraserIntersection(eraserStroke);
    
    // 只有实际擦除到内容时才记录撤销步骤
    if (hasIntersection) {
        state.strokeHistory.push(eraserStroke);
        updateUndoBtnStatus();
        console.log('橡皮擦擦除了内容，记录撤销步骤');
    } else {
        console.log('橡皮擦未擦除到内容，不记录撤销步骤');
    }
}

/**
 * 检测橡皮擦是否与现有笔画相交
 */
function checkEraserIntersection(eraserStroke) {
    if (state.strokeHistory.length === 0) {
        return false; // 没有任何笔画，肯定不相交
    }
    
    const eraserPoints = eraserStroke.points;
    if (!eraserPoints || eraserPoints.length === 0) {
        return false;
    }
    
    const eraserSize = eraserStroke.eraserSize || DRAW_CONFIG.eraserSize;
    const eraserRadius = eraserSize / 2;
    
    // 遍历所有现有笔画（不包括橡皮擦笔画）
    for (const stroke of state.strokeHistory) {
        if (stroke.type === 'erase' || stroke.type === 'clear') {
            continue; // 跳过橡皮擦和清空操作
        }
        
        const points = stroke.points;
        if (!points || points.length === 0) {
            continue;
        }
        
        // 检查橡皮擦路径上的每个点
        for (const eraserPoint of eraserPoints) {
            const ex = eraserPoint.fromX || eraserPoint.x;
            const ey = eraserPoint.fromY || eraserPoint.y;
            
            // 检查笔画的每个线段
            for (const point of points) {
                const x1 = point.fromX || point.x;
                const y1 = point.fromY || point.y;
                const x2 = point.toX || point.x;
                const y2 = point.toY || point.y;
                
                // 检测点到线段的距离
                const distance = pointToSegmentDistance(ex, ey, x1, y1, x2, y2);
                
                // 考虑橡皮擦的半径和笔画的线宽
                const strokeWidth = (stroke.lineWidth || DRAW_CONFIG.penWidth) / 2;
                const maxDistance = eraserRadius + strokeWidth;
                
                if (distance <= maxDistance) {
                    return true; // 检测到相交
                }
            }
        }
    }
    
    return false; // 没有检测到相交
}

/**
 * 计算点到线段的最短距离
 */
function pointToSegmentDistance(px, py, x1, y1, x2, y2) {
    const dx = x2 - x1;
    const dy = y2 - y1;
    
    if (dx === 0 && dy === 0) {
        // 线段退化为点
        return Math.sqrt((px - x1) ** 2 + (py - y1) ** 2);
    }
    
    // 计算投影参数 t
    const t = Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)));
    
    // 计算投影点
    const projX = x1 + t * dx;
    const projY = y1 + t * dy;
    
    // 返回距离
    return Math.sqrt((px - projX) ** 2 + (py - projY) ** 2);
}

async function redrawAllStrokes(dirtyRect = null) {
    const startTime = performance.now();
    const ctx = dom.drawCtx;
    
    // 获取可见区域，用于过滤不可见的笔画
    const visibleRect = getVisibleRect();
    
    console.log(`[重绘] 可见区域: (${visibleRect.x.toFixed(0)}, ${visibleRect.y.toFixed(0)}) ${visibleRect.width.toFixed(0)}x${visibleRect.height.toFixed(0)}, 缩放: ${state.scale.toFixed(2)}`);
    
    // 如果有脏区域，只清除和重绘该区域
    if (dirtyRect) {
        const { x, y, width, height } = dirtyRect;
        ctx.save();
        ctx.beginPath();
        ctx.rect(x, y, width, height);
        ctx.clip();
        ctx.clearRect(x, y, width, height);
    } else {
        // 优化：只清除可见区域
        ctx.clearRect(visibleRect.x, visibleRect.y, visibleRect.width, visibleRect.height);
    }
    
    if (state.baseImageObj) {
        if (dirtyRect) {
            ctx.drawImage(state.baseImageObj, 0, 0, DRAW_CONFIG.canvasW, DRAW_CONFIG.canvasH);
        } else {
            // 优化：只绘制可见区域
            ctx.drawImage(
                state.baseImageObj,
                visibleRect.x, visibleRect.y, visibleRect.width, visibleRect.height,
                visibleRect.x, visibleRect.y, visibleRect.width, visibleRect.height
            );
        }
    }
    
    if (state.strokeHistory.length === 0) {
        if (dirtyRect) {
            ctx.restore();
        }
        return;
    }
    
    // 过滤出需要重绘的笔画（在脏区域内或有交集，且在可见区域内）
    let strokesToRedraw = state.strokeHistory;
    if (dirtyRect) {
        // 使用四叉树空间索引快速查询
        if (!strokeQuadTree || strokeQuadTree.strokes.length === 0 && !strokeQuadTree.children) {
            strokeQuadTree = new StrokeQuadTree({ x: 0, y: 0, width: DRAW_CONFIG.canvasW, height: DRAW_CONFIG.canvasH });
            strokeQuadTree.build(state.strokeHistory);
        }
        strokesToRedraw = Array.from(strokeQuadTree.query(dirtyRect));
    }
    
    // 进一步过滤：只保留可见区域内的笔画
    strokesToRedraw = strokesToRedraw.filter(stroke => isStrokeVisible(stroke, visibleRect));
    
    const totalStrokes = state.strokeHistory.length;
    const visibleStrokes = strokesToRedraw.length;
    const savedPercent = totalStrokes > 0 ? ((1 - visibleStrokes / totalStrokes) * 100).toFixed(1) : 0;
    
    console.log(`[重绘] 笔画: ${visibleStrokes}/${totalStrokes} (节省 ${savedPercent}%)`);
    
    // 分离可变线宽笔画和固定线宽笔画
    const eraseStrokes = [];
    const variableWidthStrokes = [];
    const fixedWidthStrokes = new Map();
    
    for (const stroke of strokesToRedraw) {
        if (stroke.type === 'erase') {
            eraseStrokes.push(stroke);
        } else if (stroke.type === 'draw' || stroke.type === 'comment') {
            // 检查是否有可变线宽数据
            if (stroke.variableWidths && stroke.variableWidths.length > 0) {
                variableWidthStrokes.push(stroke);
            } else {
                const stateKey = `${stroke.color || DRAW_CONFIG.penColor}-${stroke.lineWidth || DRAW_CONFIG.penWidth}`;
                if (!fixedWidthStrokes.has(stateKey)) {
                    fixedWidthStrokes.set(stateKey, {
                        color: stroke.color || DRAW_CONFIG.penColor,
                        lineWidth: stroke.lineWidth || DRAW_CONFIG.penWidth,
                        strokes: []
                    });
                }
                fixedWidthStrokes.get(stateKey).strokes.push(stroke);
            }
        }
    }
    
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    // 批量绘制橡皮擦笔画
    for (const stroke of eraseStrokes) {
        const eraserSize = stroke.eraserSize || DRAW_CONFIG.eraserSize;
        ctx.globalCompositeOperation = 'destination-out';
        ctx.strokeStyle = 'rgba(0, 0, 0, 1)';
        ctx.lineWidth = eraserSize;
        
        const erasePath = new Path2D();
        if (stroke.points && stroke.points.length >= 1) {
            const firstPoint = stroke.points[0];
            if (firstPoint.x !== undefined) {
                erasePath.moveTo(firstPoint.x, firstPoint.y);
                for (let i = 1; i < stroke.points.length; i++) {
                    erasePath.lineTo(stroke.points[i].x, stroke.points[i].y);
                }
            } else {
                erasePath.moveTo(firstPoint.fromX, firstPoint.fromY);
                erasePath.lineTo(firstPoint.toX, firstPoint.toY);
                for (let i = 1; i < stroke.points.length; i++) {
                    erasePath.moveTo(stroke.points[i].fromX, stroke.points[i].fromY);
                    erasePath.lineTo(stroke.points[i].toX, stroke.points[i].toY);
                }
            }
        }
        ctx.stroke(erasePath);
    }
    
    ctx.globalCompositeOperation = 'source-over';
    
    // 绘制可变线宽笔画
    for (const stroke of variableWidthStrokes) {
        if (!stroke.points || stroke.points.length === 0) continue;
        
        ctx.fillStyle = stroke.color || DRAW_CONFIG.penColor;
        
        for (let i = 0; i < stroke.points.length && i < stroke.variableWidths.length; i++) {
            const point = stroke.points[i];
            const widthInfo = stroke.variableWidths[i];
            
            // 使用多边形绘制可变线宽线段
            const x1 = point.fromX, y1 = point.fromY;
            const x2 = point.toX, y2 = point.toY;
            const w1 = widthInfo.fromWidth, w2 = widthInfo.toWidth;
            
            const angle = Math.atan2(y2 - y1, x2 - x1);
            const perpAngle = angle + Math.PI / 2;
            const hw1 = w1 / 2, hw2 = w2 / 2;
            const cos = Math.cos(perpAngle), sin = Math.sin(perpAngle);
            
            ctx.beginPath();
            ctx.moveTo(x1 + cos * hw1, y1 + sin * hw1);
            ctx.lineTo(x2 + cos * hw2, y2 + sin * hw2);
            ctx.lineTo(x2 - cos * hw2, y2 - sin * hw2);
            ctx.lineTo(x1 - cos * hw1, y1 - sin * hw1);
            ctx.closePath();
            ctx.fill();
            
            // 绘制两端圆形
            ctx.beginPath();
            ctx.arc(x1, y1, hw1, 0, Math.PI * 2);
            ctx.fill();
            ctx.beginPath();
            ctx.arc(x2, y2, hw2, 0, Math.PI * 2);
            ctx.fill();
        }
    }
    
    // 批量绘制固定线宽笔画
    for (const [stateKey, group] of fixedWidthStrokes) {
        ctx.strokeStyle = group.color;
        ctx.lineWidth = group.lineWidth;
        
        const drawPath = new Path2D();
        for (const stroke of group.strokes) {
            if (!stroke.points || stroke.points.length < 1) continue;
            
            const firstPoint = stroke.points[0];
            if (firstPoint.x !== undefined) {
                drawPath.moveTo(firstPoint.x, firstPoint.y);
                for (let i = 1; i < stroke.points.length; i++) {
                    drawPath.lineTo(stroke.points[i].x, stroke.points[i].y);
                }
            } else {
                drawPath.moveTo(firstPoint.fromX, firstPoint.fromY);
                drawPath.lineTo(firstPoint.toX, firstPoint.toY);
                for (let i = 1; i < stroke.points.length; i++) {
                    drawPath.moveTo(stroke.points[i].fromX, stroke.points[i].fromY);
                    drawPath.lineTo(stroke.points[i].toX, stroke.points[i].toY);
                }
            }
        }
        ctx.stroke(drawPath);
    }
    
    if (dirtyRect) {
        ctx.restore();
    }
    
    const endTime = performance.now();
    console.log(`[重绘] 完成，耗时: ${(endTime - startTime).toFixed(2)}ms`);
}

async function drawEraserStroke(stroke) {
    if (!stroke.points || stroke.points.length < 1) return;
    
    const ctx = dom.drawCtx;
    setContextState(dom.drawCtx, {
        globalCompositeOperation: 'destination-out',
        strokeStyle: 'rgba(0, 0, 0, 1)',
        lineWidth: stroke.eraserSize || DRAW_CONFIG.eraserSize,
        lineCap: 'round',
        lineJoin: 'round'
    });
    
    const path = new Path2D();
    
    const firstPoint = stroke.points[0];
    if (firstPoint.x !== undefined) {
        path.moveTo(firstPoint.x, firstPoint.y);
        for (let i = 1; i < stroke.points.length; i++) {
            path.lineTo(stroke.points[i].x, stroke.points[i].y);
        }
    } else {
        path.moveTo(firstPoint.fromX, firstPoint.fromY);
        path.lineTo(firstPoint.toX, firstPoint.toY);
        for (let i = 1; i < stroke.points.length; i++) {
            path.moveTo(stroke.points[i].fromX, stroke.points[i].fromY);
            path.lineTo(stroke.points[i].toX, stroke.points[i].toY);
        }
    }
    
    dom.drawCtx.stroke(path);
}

async function drawStroke(stroke) {
    if (!stroke.points || stroke.points.length < 1) return;
    
    setContextState(dom.drawCtx, {
        strokeStyle: stroke.color || DRAW_CONFIG.penColor,
        lineWidth: stroke.lineWidth || DRAW_CONFIG.penWidth,
        lineCap: 'round',
        lineJoin: 'round',
        globalCompositeOperation: 'source-over'
    });
    
    const path = new Path2D();
    
    const firstPoint = stroke.points[0];
    if (firstPoint.x !== undefined) {
        path.moveTo(firstPoint.x, firstPoint.y);
        for (let i = 1; i < stroke.points.length; i++) {
            path.lineTo(stroke.points[i].x, stroke.points[i].y);
        }
    } else {
        path.moveTo(firstPoint.fromX, firstPoint.fromY);
        path.lineTo(firstPoint.toX, firstPoint.toY);
        for (let i = 1; i < stroke.points.length; i++) {
            path.moveTo(stroke.points[i].fromX, stroke.points[i].fromY);
            path.lineTo(stroke.points[i].toX, stroke.points[i].toY);
        }
    }
    
    dom.drawCtx.stroke(path);
}

let compactIdleId = null;

// ==================== 批注绘制系统 ====================
// Canvas上下文状态管理、笔画绘制、批注压缩

// 上下文状态缓存
let currentContextState = {
    strokeStyle: null,
    lineWidth: null,
    lineCap: null,
    lineJoin: null,
    globalCompositeOperation: null
};

// ==================== 线段点优化 ====================
// 点简化配置：用于优化笔画数据，减少冗余点

// 点简化配置
const POINT_OPTIMIZATION = {
    epsilon: 0.3,  // 简化阈值 (像素) - 减小以保留更多点
    minDistance: 1, // 最小点间距 (像素) - 减小以减少断点
    quantization: 0.25, // 坐标量化步长 (像素) - 提高精度
    maxPointsPerStroke: 1500, // 每笔画最大点数 - 增加以容纳更多点
    batchInterval: 30, // 批量收集间隔 (ms) - 减小以提高响应速度
};

// ==================== WASM 降级方案 ====================
// 以下函数仅在 WASM 加载失败时使用

/**
 * 坐标量化 (降级方案)
 */
function quantizeCoord(coord) {
    return Math.round(coord / POINT_OPTIMIZATION.quantization) * POINT_OPTIMIZATION.quantization;
}

/**
 * 计算两点之间的距离 (降级方案)
 */
function distance(x1, y1, x2, y2) {
    const dx = x2 - x1;
    const dy = y2 - y1;
    return Math.sqrt(dx * dx + dy * dy);
}

/**
 * Douglas-Peucker 点简化算法（迭代实现，避免递归调用栈溢出）
 */
function simplifyPoints(points, epsilon) {
    if (points.length <= 2) return points;
    
    // 使用迭代实现，避免递归调用栈溢出
    const result = new Array(points.length);
    result[0] = points[0];
    result[points.length - 1] = points[points.length - 1];
    
    // 使用栈来存储待处理的区间
    const stack = [{ start: 0, end: points.length - 1 }];
    
    while (stack.length > 0) {
        const { start, end } = stack.pop();
        
        if (end - start <= 1) continue;
        
        let maxDist = 0;
        let maxIndex = start;
        
        // 找到距离最远的点
        for (let i = start + 1; i < end; i++) {
            const dist = perpendicularDistance(
                points[i].fromX, points[i].fromY,
                points[start].fromX, points[start].fromY,
                points[end].toX, points[end].toY
            );
            if (dist > maxDist) {
                maxDist = dist;
                maxIndex = i;
            }
        }
        
        // 如果最大距离大于阈值，保留该点并继续处理子区间
        if (maxDist > epsilon) {
            result[maxIndex] = points[maxIndex];
            stack.push({ start: maxIndex, end: end });
            stack.push({ start: start, end: maxIndex });
        }
    }
    
    // 过滤掉未标记的点，保持原始顺序
    const simplified = [];
    for (let i = 0; i < points.length; i++) {
        if (result[i] !== undefined) {
            simplified.push(result[i]);
        }
    }
    
    return simplified;
}

/**
 * 计算点到线段的垂直距离 (降级方案)
 */
function perpendicularDistance(px, py, x1, y1, x2, y2) {
    const A = px - x1;
    const B = py - y1;
    const C = x2 - x1;
    const D = y2 - y1;
    
    const dot = A * C + B * D;
    const lenSq = C * C + D * D;
    let param = -1;
    
    if (lenSq !== 0) {
        param = dot / lenSq;
    }
    
    let xx, yy;
    if (param < 0) {
        xx = x1;
        yy = y1;
    } else if (param > 1) {
        xx = x2;
        yy = y2;
    } else {
        xx = x1 + param * C;
        yy = y1 + param * D;
    }
    
    return distance(px, py, xx, yy);
}

/**
 * 批量收集线段点
 */
class PointCollector {
    constructor() {
        this.points = [];
        this.lastTime = Date.now();
        this.lastX = 0;
        this.lastY = 0;
    }
    
    async addPoint(fromX, fromY, toX, toY) {
        try {
            const config = {
                epsilon: POINT_OPTIMIZATION.epsilon,
                minDistance: POINT_OPTIMIZATION.minDistance,
                quantization: POINT_OPTIMIZATION.quantization
            };
            
            const points = [{
                fromX: fromX,
                fromY: fromY,
                toX: toX,
                toY: toY
            }];
            
            const result = await wasmPointProcessor.collectPoints(
                points,
                config,
                this.lastTime,
                this.lastX,
                this.lastY
            );
            
            if (result.error) {
                throw new Error(result.error);
            }
            
            if (result.collectedPoints && result.collectedPoints.length > 0) {
                const collectedPoint = result.collectedPoints[0];
                this.points.push({
                    fromX: collectedPoint.fromX,
                    fromY: collectedPoint.fromY,
                    toX: collectedPoint.toX,
                    toY: collectedPoint.toY
                });
                
                this.lastTime = result.lastTime;
                this.lastX = result.lastX;
                this.lastY = result.lastY;
                
                return true;
            }
        } catch (error) {
            console.warn('WASM点收集失败，使用前端降级方案:', error);
            const qFromX = quantizeCoord(fromX);
            const qFromY = quantizeCoord(fromY);
            const qToX = quantizeCoord(toX);
            const qToY = quantizeCoord(toY);
            
            if (distance(qFromX, qFromY, qToX, qToY) < POINT_OPTIMIZATION.minDistance) {
                return false;
            }
            
            const now = Date.now();
            if (now - this.lastTime < POINT_OPTIMIZATION.batchInterval) {
                return false;
            }
            
            this.lastTime = now;
            this.lastX = qToX;
            this.lastY = qToY;
            
            this.points.push({
                fromX: qFromX,
                fromY: qFromY,
                toX: qToX,
                toY: qToY
            });
            
            if (this.points.length > POINT_OPTIMIZATION.maxPointsPerStroke) {
                this.points = simplifyPoints(this.points, POINT_OPTIMIZATION.epsilon);
            }
            
            return true;
        }
        
        return false;
    }
    
    getPoints() {
        return this.points;
    }
    
    clear() {
        this.points = [];
        this.lastTime = Date.now();
    }
}



// ==================== 批处理绘制系统 ====================
// 批量绘制命令管理：减少Canvas状态切换，提高绘制效率

/**
 * 批处理绘制管理器
 * 按状态分组批量处理绘制命令
 */
class BatchDrawManager {
    constructor() {
        this.batches = new Map(); // 按状态分组的批处理
        this.maxBatchSize = 300;   // 每批最大命令数（进一步增加以减少flush次数）
        this.batchInterval = 16;   // 批处理间隔 (ms)
        this.lastBatchTime = 0;     // 上次批处理时间
        this.isDrawing = false;     // 是否正在绘制中
        this.pendingFlush = false;  // 是否有待处理的flush
        this.minDistance = 0.5;     // 最小点间距，用于过滤冗余点（增加以减少点数量）
        this.variableWidthCommands = []; // 可变线宽命令队列
    }
    
    /**
     * 添加绘制命令（支持可变线宽）
     */
    addCommand(type, fromX, fromY, toX, toY, color, lineWidth, fromWidth = null, toWidth = null) {
        // 计算距离，过滤太近的点
        const distance = Math.sqrt(Math.pow(toX - fromX, 2) + Math.pow(toY - fromY, 2));
        if (distance < this.minDistance) {
            return; // 跳过太近的点
        }
        
        // 生成状态键
        const stateKey = `${type}-${color}-${lineWidth}`;
        
        // 获取或创建批处理
        if (!this.batches.has(stateKey)) {
            this.batches.set(stateKey, {
                type,
                color,
                lineWidth,
                commands: []
            });
        }
        
        const batch = this.batches.get(stateKey);
        batch.commands.push({ fromX, fromY, toX, toY });
        
        // 检查批处理大小
        if (batch.commands.length >= this.maxBatchSize) {
            this.flushBatch(stateKey);
        }
    }
    
    /**
     * 使用WASM优化绘制命令
     */
    async optimizeCommands() {
        try {
            // 收集所有命令
            let allCommands = [];
            for (const [stateKey, batch] of this.batches) {
                for (const cmd of batch.commands) {
                    allCommands.push({
                        type: batch.type,
                        fromX: cmd.fromX,
                        fromY: cmd.fromY,
                        toX: cmd.toX,
                        toY: cmd.toY,
                        color: batch.color,
                        lineWidth: batch.lineWidth
                    });
                }
            }
            
            // 使用WASM优化命令
            const optimizedCommands = await wasmPointProcessor.batchProcessDrawCommands(
                allCommands,
                this.minDistance,
                this.maxBatchSize
            );
            
            // 清空现有批处理
            this.batches.clear();
            
            // 重新分组优化后的命令
            for (const cmd of optimizedCommands) {
                const stateKey = `${cmd.type}-${cmd.color}-${cmd.line_width}`;
                if (!this.batches.has(stateKey)) {
                    this.batches.set(stateKey, {
                        type: cmd.type,
                        color: cmd.color,
                        lineWidth: cmd.line_width,
                        commands: []
                    });
                }
                const batch = this.batches.get(stateKey);
                batch.commands.push({
                    fromX: cmd.fromX,
                    fromY: cmd.fromY,
                    toX: cmd.toX,
                    toY: cmd.toY
                });
            }
            
            console.log('使用WASM优化绘制命令，从', allCommands.length, '个命令优化到', optimizedCommands.length, '个命令');
        } catch (error) {
            console.warn('WASM命令优化失败，使用原始命令:', error);
            // 继续使用原始命令
        }
    }
    
    /**
     * 开始绘制
     */
    startDrawing() {
        this.isDrawing = true;
        this.pendingFlush = false;
    }
    
    /**
     * 结束绘制
     */
    async endDrawing() {
        this.isDrawing = false;
        await this.optimizeCommands();
        this.flushAll();
    }
    
    /**
     * 绘制可变线宽线段（使用多边形模拟）
     */
    drawVariableWidthLine(ctx, x1, y1, x2, y2, w1, w2, color, isErase) {
        if (isErase) {
            ctx.globalCompositeOperation = 'destination-out';
            ctx.fillStyle = 'rgba(0, 0, 0, 1)';
        } else {
            ctx.globalCompositeOperation = 'source-over';
            ctx.fillStyle = color;
        }
        
        const angle = Math.atan2(y2 - y1, x2 - x1);
        const perpAngle = angle + Math.PI / 2;
        
        const hw1 = w1 / 2;
        const hw2 = w2 / 2;
        
        const cos = Math.cos(perpAngle);
        const sin = Math.sin(perpAngle);
        
        ctx.beginPath();
        ctx.moveTo(x1 + cos * hw1, y1 + sin * hw1);
        ctx.lineTo(x2 + cos * hw2, y2 + sin * hw2);
        ctx.lineTo(x2 - cos * hw2, y2 - sin * hw2);
        ctx.lineTo(x1 - cos * hw1, y1 - sin * hw1);
        ctx.closePath();
        ctx.fill();
        
        // 绘制两端圆形
        ctx.beginPath();
        ctx.arc(x1, y1, hw1, 0, Math.PI * 2);
        ctx.fill();
        
        ctx.beginPath();
        ctx.arc(x2, y2, hw2, 0, Math.PI * 2);
        ctx.fill();
    }
    
    /**
     * 执行单个批处理
     */
    flushBatch(stateKey) {
        const batch = this.batches.get(stateKey);
        if (!batch || batch.commands.length === 0) return;
        
        const ctx = dom.drawCtx;
        const isErase = batch.type === 'erase';
        
        if (isErase) {
            ctx.globalCompositeOperation = 'destination-out';
            ctx.strokeStyle = 'rgba(0, 0, 0, 1)';
        } else {
            ctx.globalCompositeOperation = 'source-over';
            ctx.strokeStyle = batch.color;
        }
        
        ctx.lineWidth = batch.lineWidth;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        
        const path = new Path2D();
        for (const cmd of batch.commands) {
            path.moveTo(cmd.fromX, cmd.fromY);
            path.lineTo(cmd.toX, cmd.toY);
        }
        ctx.stroke(path);
        
        batch.commands = [];
    }
    
    /**
     * 执行所有批处理
     */
    flushAll() {
        const ctx = dom.drawCtx;
        
        // 先绘制可变线宽命令
        if (this.variableWidthCommands.length > 0) {
            for (const cmd of this.variableWidthCommands) {
                this.drawVariableWidthLine(
                    ctx,
                    cmd.fromX, cmd.fromY,
                    cmd.toX, cmd.toY,
                    cmd.fromWidth, cmd.toWidth,
                    cmd.color,
                    cmd.type === 'erase'
                );
            }
            this.variableWidthCommands = [];
        }
        
        if (this.batches.size === 0) {
            ctx.globalCompositeOperation = 'source-over';
            return;
        }
        
        // 不变的状态设置移到循环外部
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        
        for (const [stateKey, batch] of this.batches) {
            if (batch.commands.length === 0) continue;
            
            const isErase = batch.type === 'erase';
            
            // 只设置变化的状态
            if (isErase) {
                ctx.globalCompositeOperation = 'destination-out';
                ctx.strokeStyle = 'rgba(0, 0, 0, 1)';
            } else {
                ctx.globalCompositeOperation = 'source-over';
                ctx.strokeStyle = batch.color;
            }
            
            ctx.lineWidth = batch.lineWidth;
            
            const path = new Path2D();
            for (const cmd of batch.commands) {
                path.moveTo(cmd.fromX, cmd.fromY);
                path.lineTo(cmd.toX, cmd.toY);
            }
            ctx.stroke(path);
            
            batch.commands = [];
        }
        
        // 恢复默认状态
        ctx.globalCompositeOperation = 'source-over';
        
        this.batches.clear();
    }
    
    /**
     * 清空所有批处理
     */
    clear() {
        this.batches.clear();
        this.variableWidthCommands = [];
    }
    
    /**
     * 获取批处理数量
     */
    getBatchCount() {
        return this.batches.size;
    }
}

/**
 * 设置上下文状态（只更新变化的属性）
 * @param {CanvasRenderingContext2D} ctx - 目标上下文
 * @param {Object} state - 新状态
 */
function setContextState(ctx, state) {
    if (currentContextState.strokeStyle !== state.strokeStyle) {
        ctx.strokeStyle = state.strokeStyle;
        currentContextState.strokeStyle = state.strokeStyle;
    }
    
    if (currentContextState.lineWidth !== state.lineWidth) {
        ctx.lineWidth = state.lineWidth;
        currentContextState.lineWidth = state.lineWidth;
    }
    
    if (currentContextState.lineCap !== state.lineCap) {
        ctx.lineCap = state.lineCap;
        currentContextState.lineCap = state.lineCap;
    }
    
    if (currentContextState.lineJoin !== state.lineJoin) {
        ctx.lineJoin = state.lineJoin;
        currentContextState.lineJoin = state.lineJoin;
    }
    
    if (currentContextState.globalCompositeOperation !== state.globalCompositeOperation) {
        ctx.globalCompositeOperation = state.globalCompositeOperation;
        currentContextState.globalCompositeOperation = state.globalCompositeOperation;
    }
}

/**
 * 按时间顺序逐个绘制笔画
 * 必须按原始顺序绘制，确保橡皮擦的destination-out在正确时机执行
 * @param {CanvasRenderingContext2D} ctx - 目标上下文
 * @param {Array} strokes - 笔画数组
 */
async function drawStrokes(ctx, strokes) {
    if (strokes.length === 0) return;
    
    const totalStrokes = strokes.length;
    
    // 视口基于 Canvas 尺寸
    const viewport = {
        x: 0,
        y: 0,
        width: DRAW_CONFIG.canvasW,
        height: DRAW_CONFIG.canvasH
    };
    
    let visibleStrokes = strokes;
    
    if (totalStrokes > 20) {
        try {
            visibleStrokes = await wasmPointProcessor.cullStrokesByViewport(strokes, viewport);
            if (visibleStrokes.length < strokes.length) {
                console.log(`视口裁剪: ${strokes.length} -> ${visibleStrokes.length} 笔画`);
            }
        } catch (error) {
            console.warn('WASM视口裁剪失败，使用全部笔画:', error);
            visibleStrokes = strokes;
        }
    }
    
    // 按状态分组，使用 Path2D 批量绘制
    const eraseStrokes = new Map();
    const drawStrokes = new Map();
    
    for (const stroke of visibleStrokes) {
        if (stroke.type === 'erase') {
            const sizeKey = stroke.eraserSize || DRAW_CONFIG.eraserSize;
            if (!eraseStrokes.has(sizeKey)) {
                eraseStrokes.set(sizeKey, []);
            }
            eraseStrokes.get(sizeKey).push(stroke);
        } else if (stroke.type === 'draw' || stroke.type === 'comment') {
            const stateKey = `${stroke.color || DRAW_CONFIG.penColor}-${stroke.lineWidth || DRAW_CONFIG.penWidth}`;
            if (!drawStrokes.has(stateKey)) {
                drawStrokes.set(stateKey, {
                    color: stroke.color || DRAW_CONFIG.penColor,
                    lineWidth: stroke.lineWidth || DRAW_CONFIG.penWidth,
                    strokes: []
                });
            }
            drawStrokes.get(stateKey).strokes.push(stroke);
        }
    }
    
    // 设置通用状态
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    // 批量绘制橡皮擦笔画
    for (const [eraserSize, strokes] of eraseStrokes) {
        ctx.globalCompositeOperation = 'destination-out';
        ctx.strokeStyle = 'rgba(0, 0, 0, 1)';
        ctx.lineWidth = eraserSize;
        
        const path = new Path2D();
        for (const stroke of strokes) {
            if (!stroke.points || stroke.points.length < 1) continue;
            
            const firstPoint = stroke.points[0];
            if (firstPoint.x !== undefined) {
                path.moveTo(firstPoint.x, firstPoint.y);
                for (let i = 1; i < stroke.points.length; i++) {
                    path.lineTo(stroke.points[i].x, stroke.points[i].y);
                }
            } else {
                for (const point of stroke.points) {
                    path.moveTo(point.fromX, point.fromY);
                    path.lineTo(point.toX, point.toY);
                }
            }
        }
        ctx.stroke(path);
    }
    
    // 批量绘制普通笔画
    ctx.globalCompositeOperation = 'source-over';
    for (const [stateKey, group] of drawStrokes) {
        ctx.strokeStyle = group.color;
        ctx.lineWidth = group.lineWidth;
        
        const path = new Path2D();
        for (const stroke of group.strokes) {
            if (!stroke.points || stroke.points.length < 1) continue;
            
            const firstPoint = stroke.points[0];
            if (firstPoint.x !== undefined) {
                path.moveTo(firstPoint.x, firstPoint.y);
                for (let i = 1; i < stroke.points.length; i++) {
                    path.lineTo(stroke.points[i].x, stroke.points[i].y);
                }
            } else {
                for (const point of stroke.points) {
                    path.moveTo(point.fromX, point.fromY);
                    path.lineTo(point.toX, point.toY);
                }
            }
        }
        ctx.stroke(path);
    }
}

/**
 * 调度笔画压缩 (空闲时执行)
 * 当笔画数超过阈值时，将旧笔画压缩为图片
 */
function scheduleCompact() {
    if (state.strokeHistory.length <= state.MAX_UNDO_STEPS) return;
    if (compactIdleId !== null) return;
    
    const strokesToCompact = state.strokeHistory.slice(0, state.strokeHistory.length - state.MAX_UNDO_STEPS);
    state.strokeHistory = state.strokeHistory.slice(state.strokeHistory.length - state.MAX_UNDO_STEPS);
    
    if (strokesToCompact.length === 0) return;
    
    compactIdleId = requestIdleCallback((deadline) => {
        compactIdleId = null;
        doCompactStrokes(strokesToCompact);
    }, { timeout: 2000 });
}

/**
 * 执行笔画压缩
 * 优先使用Rust并行处理，失败时降级到前端Canvas
 */
async function doCompactStrokes(strokesToCompact) {
    const loadId = ++state.baseImageLoadId;
    
    if (window.__TAURI__) {
        try {
            const { invoke } = window.__TAURI__.core;
            
            const request = {
                baseImage: state.baseImageURL,
                strokes: strokesToCompact,
                canvasWidth: DRAW_CONFIG.canvasW,
                canvasHeight: DRAW_CONFIG.canvasH
            };
            
            const result = await invoke('compact_strokes', { request });
            
            if (loadId !== state.baseImageLoadId) return;
            
            state.baseImageURL = result;
            state.baseImageObj = null;
            const img = new Image();
            img.onload = () => {
                if (loadId === state.baseImageLoadId) {
                    state.baseImageObj = img;
                }
            };
            img.src = result;
            
            console.log('Rust 笔画已压缩，保留最近', state.strokeHistory.length, '笔可撤销');
            return;
        } catch (error) {
            console.error('Rust 笔画压缩失败，使用前端降级方案:', error);
        }
    }
    
    const offscreen = getOffscreenCanvas();
    const tempCtx = offscreen.ctx;
    
    if (state.baseImageObj) {
        tempCtx.drawImage(state.baseImageObj, 0, 0, DRAW_CONFIG.canvasW, DRAW_CONFIG.canvasH);
    }
    
    await drawStrokes(tempCtx, strokesToCompact);
    
    if (loadId !== state.baseImageLoadId) {
        releaseOffscreenCanvas(offscreen);
        return;
    }
    
    state.baseImageURL = offscreen.canvas.toDataURL('image/png');
    state.baseImageObj = null;
    const img = new Image();
    img.onload = () => {
        if (loadId === state.baseImageLoadId) {
            state.baseImageObj = img;
        }
        releaseOffscreenCanvas(offscreen);
    };
    img.onerror = () => {
        releaseOffscreenCanvas(offscreen);
    };
    img.src = state.baseImageURL;
    
    console.log('笔画已异步压缩，保留最近', state.strokeHistory.length, '笔可撤销');
}

function compactStrokes() {
    scheduleCompact();
}

async function saveSnapshot() {
    await endStroke();
}

async function undo() {
    if (state.strokeHistory.length === 0) return;
    
    const lastStroke = state.strokeHistory[state.strokeHistory.length - 1];
    
    if (lastStroke.type === 'clear') {
        state.strokeHistory = lastStroke.savedStrokeHistory || [];
        state.baseImageURL = lastStroke.savedBaseImageURL;
        state.baseImageObj = null;
        
        if (state.baseImageURL) {
            const img = new Image();
            img.onload = async () => {
                state.baseImageObj = img;
                await redrawAllStrokes();
            };
            img.src = state.baseImageURL;
        } else {
            await redrawAllStrokes();
        }
    } else {
        state.strokeHistory.pop();
        // 撤销时重绘整个批注层，不使用局部重绘
        await redrawAllStrokes();
    }
    
    updateUndoBtnStatus();
    console.log('撤销操作');
}

function updateUndoBtnStatus() {
    dom.btnUndo.disabled = state.strokeHistory.length === 0;
}

// 清空画布
function clearDrawCanvas() {
    dom.drawCtx.clearRect(0, 0, DRAW_CONFIG.canvasW, DRAW_CONFIG.canvasH);
    setContextState(dom.drawCtx, {
        strokeStyle: DRAW_CONFIG.penColor,
        lineWidth: DRAW_CONFIG.penWidth,
        lineCap: 'round',
        lineJoin: 'round',
        globalCompositeOperation: 'source-over'
    });
}

function clearAllDrawings() {
    if (state.strokeHistory.length === 0 && !state.baseImageObj) return;
    
    const clearStroke = {
        type: 'clear',
        savedStrokeHistory: cloneStrokes(state.strokeHistory),
        savedBaseImageURL: state.baseImageURL
    };
    
    state.strokeHistory = [clearStroke];
    
    clearDrawCanvas();
    state.baseImageURL = null;
    state.baseImageObj = null;
    updateUndoBtnStatus();
    
    if (state.currentImageIndex >= 0 && state.currentImageIndex < state.imageList.length) {
        state.imageList[state.currentImageIndex].strokeHistory = cloneStrokes(state.strokeHistory);
        state.imageList[state.currentImageIndex].baseImageURL = null;
    }
    
    if (state.currentFolderIndex >= 0 && state.currentFolderPageIndex >= 0) {
        if (state.currentFolderIndex < state.fileList.length) {
            const folder = state.fileList[state.currentFolderIndex];
            if (state.currentFolderPageIndex < folder.pages.length) {
                folder.pages[state.currentFolderPageIndex].strokeHistory = cloneStrokes(state.strokeHistory);
                folder.pages[state.currentFolderPageIndex].baseImageURL = null;
            }
        }
    }
    
    if (state.drawMode === 'eraser') {
        switchMode('comment');
    }
    
    console.log('清空所有批注');
}

// 拍照功能
function takePhoto() {
    if (state.isCameraOpen) {
        captureCamera();
    } else if (state.currentImageIndex >= 0 && state.imageList.length > 0) {
        (async () => {
            try {
                saveCurrentDrawData();
                saveCurrentFolderPageDrawData();
                state.currentImageIndex = -1;
                state.currentImage = null;
                clearImageLayer();
                clearDrawCanvas();
                
                // 恢复摄像头视图状态和批注
                state.scale = state.cameraViewState.scale;
                state.canvasX = state.cameraViewState.canvasX;
                state.canvasY = state.cameraViewState.canvasY;
                state.strokeHistory = cloneStrokes(state.cameraViewState.strokeHistory);
                state.baseImageURL = state.cameraViewState.baseImageURL;
                state.baseImageObj = null;
                updateMoveBound();
                updateCanvasTransform();
                
                // 恢复批注
                if (state.strokeHistory.length > 0) {
                    await redrawAllStrokes();
                } else {
                    clearDrawCanvas();
                }
                updateUndoBtnStatus();
                
                await openCamera();
                updateSidebarSelection();
                updatePhotoButtonState();
                updateEnhanceButtonState();
                console.log('返回摄像头');
            } catch (error) {
                console.error('返回摄像头失败:', error);
                if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
                    showNoCameraMessage(window.i18n?.t('camera.notDetected') || '未检测到摄像头');
                } else if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
                    showNoCameraMessage(window.i18n?.t('camera.noPermission') || '无摄像头权限');
                } else {
                    showNoCameraMessage(window.i18n?.t('camera.initFailed') || '摄像头初始化失败');
                }
            }
        })();
    } else if (state.currentFolderIndex >= 0 && state.currentFolderPageIndex >= 0) {
        (async () => {
            try {
                saveCurrentDrawData();
                saveCurrentFolderPageDrawData();
                state.currentFolderIndex = -1;
                state.currentFolderPageIndex = -1;
                state.currentImage = null;
                clearImageLayer();
                clearDrawCanvas();
                
                // 恢复摄像头视图状态和批注
                state.scale = state.cameraViewState.scale;
                state.canvasX = state.cameraViewState.canvasX;
                state.canvasY = state.cameraViewState.canvasY;
                state.strokeHistory = cloneStrokes(state.cameraViewState.strokeHistory);
                state.baseImageURL = state.cameraViewState.baseImageURL;
                state.baseImageObj = null;
                updateMoveBound();
                updateCanvasTransform();
                
                // 恢复批注
                if (state.strokeHistory.length > 0) {
                    await redrawAllStrokes();
                } else {
                    clearDrawCanvas();
                }
                updateUndoBtnStatus();
                
                await openCamera();
                updateFolderPageSelection(-1, -1);
                updatePhotoButtonState();
                updateEnhanceButtonState();
                console.log('返回摄像头');
            } catch (error) {
                console.error('返回摄像头失败:', error);
            }
        })();
    } else {
        saveMergedCanvas();
    }
}

function saveMergedCanvas() {
    console.log('执行拍照功能');
    const offscreen = getOffscreenCanvas();
    const mergedCtx = offscreen.ctx;
    
    // 填充背景色
    mergedCtx.fillStyle = '#3a3a3a';
    mergedCtx.fillRect(0, 0, DRAW_CONFIG.canvasW, DRAW_CONFIG.canvasH);
    
    mergedCtx.drawImage(dom.imageCanvas, 0, 0, DRAW_CONFIG.canvasW, DRAW_CONFIG.canvasH);
    mergedCtx.drawImage(dom.drawCanvas, 0, 0, DRAW_CONFIG.canvasW, DRAW_CONFIG.canvasH);
    
    const link = document.createElement('a');
    link.download = `photo_${Date.now()}.png`;
    link.href = offscreen.canvas.toDataURL('image/png');
    link.click();
    
    releaseOffscreenCanvas(offscreen);
}

let lastPhotoButtonState = null;

function updatePhotoButtonState() {
    const btnPhoto = dom.btnPhoto;
    if (!btnPhoto) return;
    
    let newState;
    let html, title, addClass, removeClass;
    
    const photoText = window.i18n?.t('toolbar.photo') || '拍照';
    const switchToCameraText = window.i18n?.t('camera.switchToCamera') || '切换到摄像头';
    
    if (state.isCameraOpen) {
        newState = 'camera';
        html = `<img src="assets/icon/camera.svg" width="16" height="16" alt="${photoText}" style="filter: invert(1);">${photoText}`;
        title = window.i18n?.t('camera.captureFrame') || '捕获摄像头画面';
        addClass = 'camera-active';
        removeClass = '';
    } else if ((state.currentImageIndex >= 0 && state.imageList.length > 0) || 
               (state.currentFolderIndex >= 0 && state.currentFolderPageIndex >= 0)) {
        newState = 'switch';
        html = `<img src="assets/icon/camera-fill.svg" width="16" height="16" alt="${switchToCameraText}" style="filter: invert(1);">${switchToCameraText}`;
        title = window.i18n?.t('camera.switchToCamera') || '返回摄像头';
        addClass = '';
        removeClass = 'camera-active';
    } else {
        newState = 'save';
        html = `<img src="assets/icon/camera.svg" width="16" height="16" alt="${photoText}" style="filter: invert(1);">${photoText}`;
        title = window.i18n?.t('camera.saveScreenshot') || '保存画布截图';
        addClass = '';
        removeClass = 'camera-active';
    }
    
    if (lastPhotoButtonState === newState) return;
    lastPhotoButtonState = newState;
    
    btnPhoto.innerHTML = html;
    btnPhoto.title = title;
    if (addClass) btnPhoto.classList.add(addClass);
    if (removeClass) btnPhoto.classList.remove(removeClass);
}

// 设置功能
function openSettings() {
    const existingPanel = dom.settingsPanel.classList.contains('visible');
    if (existingPanel) {
        hideSettingsPanel();
    } else {
        showSettingsPanel();
    }
}

function showSettingsPanel() {
    hidePenControlPanel();
    
    const panel = dom.settingsPanel;
    const btnRect = dom.btnSettings.getBoundingClientRect();
    const containerRect = document.querySelector('.main-function').getBoundingClientRect();
    
    const panelWidth = 130;
    const panelHeight = panel.offsetHeight || 50;
    
    let left = btnRect.left - containerRect.left + (btnRect.width / 2) - (panelWidth / 2);
    let top = btnRect.top - containerRect.top - panelHeight - 10;
    
    if (left < 10) left = 10;
    if (left + panelWidth > containerRect.width - 10) {
        left = containerRect.width - panelWidth - 10;
    }
    
    if (top < 10) {
        top = btnRect.bottom - containerRect.top + 10;
    }
    
    panel.style.left = `${left}px`;
    panel.style.top = `${top}px`;
    panel.classList.add('visible');
}

function hideSettingsPanel() {
    dom.settingsPanel.classList.remove('visible');
}

function openSettingsWindow() {
    if (window.__TAURI__) {
        const { invoke } = window.__TAURI__.core;
        invoke('open_settings_window').catch(error => {
            console.error('打开设置窗口失败:', error);
        });
    }
}

async function rotateImage(direction) {
    if (state.isCameraOpen) {
        if (direction === 'left') {
            state.cameraRotation = (state.cameraRotation - 90 + 360) % 360;
        } else {
            state.cameraRotation = (state.cameraRotation + 90) % 360;
        }
        updateCameraVideoStyle();
        console.log(`摄像头画面已旋转到 ${state.cameraRotation}°`);
        return;
    }
    
    if (!state.currentImage) {
        console.log('没有图片可旋转');
        return;
    }
    
    let rotatedDataUrl;
    
    if (window.__TAURI__) {
        try {
            const { invoke } = window.__TAURI__.core;
            rotatedDataUrl = await invoke('rotate_image', { 
                imageData: state.currentImage.src, 
                direction: direction 
            });
            console.log('Rust 图片旋转完成');
        } catch (error) {
            console.error('Rust 图片旋转失败，使用前端降级方案:', error);
            rotatedDataUrl = rotateImageFallback(state.currentImage, direction);
        }
    } else {
        rotatedDataUrl = rotateImageFallback(state.currentImage, direction);
    }
    
    const rotatedImg = new Image();
    rotatedImg.onload = async () => {
        state.currentImage = rotatedImg;
        
        if (state.currentImageIndex >= 0 && state.currentImageIndex < state.imageList.length) {
            const thumbnail = await generateThumbnail(rotatedImg.src, 150);
            
            state.imageList[state.currentImageIndex].full = rotatedImg.src;
            state.imageList[state.currentImageIndex].thumbnail = thumbnail;
            state.imageList[state.currentImageIndex].width = rotatedImg.width;
            state.imageList[state.currentImageIndex].height = rotatedImg.height;
            
            updateSidebarContent();
        }
        
        drawImageToCenter(rotatedImg);
        console.log(`图片已向${direction === 'left' ? '左' : '右'}旋转`);
    };
    rotatedImg.src = rotatedDataUrl;
}

function rotateImageFallback(img, direction) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    if (direction === 'left') {
        canvas.width = img.height;
        canvas.height = img.width;
        ctx.translate(canvas.width / 2, canvas.height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.drawImage(img, -img.width / 2, -img.height / 2);
    } else {
        canvas.width = img.height;
        canvas.height = img.width;
        ctx.translate(canvas.width / 2, canvas.height / 2);
        ctx.rotate(Math.PI / 2);
        ctx.drawImage(img, -img.width / 2, -img.height / 2);
    }
    
    return canvas.toDataURL('image/png');
}

function toggleEnhance() {
    state.enhanceEnabled = !state.enhanceEnabled;
    
    if (dom.btnEnhance) {
        dom.btnEnhance.classList.toggle('primary-btn', state.enhanceEnabled);
    }
    
    console.log(`文档增强已${state.enhanceEnabled ? '开启' : '关闭'}`);
}

let lastEnhanceButtonState = null;

function updateEnhanceButtonState() {
    if (!dom.btnEnhance) return;
    
    const newState = state.isCameraOpen ? 'visible' : 'hidden';
    
    if (lastEnhanceButtonState === newState) return;
    lastEnhanceButtonState = newState;
    
    const toolbarCenter = document.querySelector('.toolbar-center');
    
    if (state.isCameraOpen) {
        if (toolbarCenter) {
            toolbarCenter.classList.remove('compact');
        }
        
        setTimeout(() => {
            dom.btnEnhance.classList.remove('fade-out');
            dom.btnEnhance.classList.add('fade-in');
        }, 400);
    } else {
        dom.btnEnhance.classList.remove('fade-in');
        dom.btnEnhance.classList.add('fade-out');
        
        setTimeout(() => {
            if (toolbarCenter) {
                toolbarCenter.classList.add('compact');
            }
        }, 500);
    }
}

async function generateThumbnail(imageData, maxSize = 150, fixedRatio = true) {
    if (window.__TAURI__) {
        try {
            const { invoke } = window.__TAURI__.core;
            const thumbnail = await invoke('generate_thumbnail', { 
                imageData: imageData, 
                maxSize: maxSize,
                fixedRatio: fixedRatio
            });
            return thumbnail;
        } catch (error) {
            console.error('Rust 缩略图生成失败，使用前端降级方案:', error);
        }
    }
    
    return generateThumbnailFallback(imageData, maxSize, fixedRatio);
}

function generateThumbnailFallback(imageData, maxSize = 150, fixedRatio = true) {
    const img = new Image();
    img.src = imageData;
    
    let thumbW, thumbH, scaledW, scaledH, offsetX, offsetY;
    
    if (fixedRatio) {
        thumbW = maxSize;
        thumbH = Math.round(maxSize * 9 / 16);
        
        const imgRatio = img.width / img.height;
        const canvasRatio = 16 / 9;
        
        if (imgRatio > canvasRatio) {
            scaledW = thumbW;
            scaledH = thumbW / imgRatio;
        } else {
            scaledH = thumbH;
            scaledW = thumbH * imgRatio;
        }
        
        offsetX = (thumbW - scaledW) / 2;
        offsetY = (thumbH - scaledH) / 2;
    } else {
        if (img.width > img.height) {
            thumbW = maxSize;
            thumbH = (img.height / img.width) * maxSize;
        } else {
            thumbH = maxSize;
            thumbW = (img.width / img.height) * maxSize;
        }
        scaledW = thumbW;
        scaledH = thumbH;
        offsetX = 0;
        offsetY = 0;
    }
    
    const thumbCanvas = document.createElement('canvas');
    thumbCanvas.width = thumbW;
    thumbCanvas.height = thumbH;
    const thumbCtx = thumbCanvas.getContext('2d');
    
    thumbCtx.fillStyle = '#000000';
    thumbCtx.fillRect(0, 0, thumbW, thumbH);
    
    thumbCtx.drawImage(img, offsetX, offsetY, scaledW, scaledH);
    
    return thumbCanvas.toDataURL('image/jpeg', 0.7);
}

// 侧边栏事件
function bindSidebarEvents() {
    dom.btnExpand.addEventListener('click', toggleSidebar);
}

function toggleSidebar() {
    const existingSidebar = document.querySelector('.sidebar:not(.file-sidebar)');
    const existingFileSidebar = document.querySelector('.file-sidebar');
    
    if (existingFileSidebar) {
        collapseFileSidebar();
    }
    
    if (existingSidebar) {
        collapseSidebar();
    } else {
        expandSidebar();
    }
}

function expandSidebar() {
    const sidebarElement = document.createElement('div');
    sidebarElement.classList.add('sidebar');
    
    const noImagesText = window.i18n?.t('common.noImages') || '暂无图片';
    const imageListText = window.i18n?.t('sidebar.imageList') || '图片列表';
    const importImageText = window.i18n?.t('sidebar.importImage') || '导入图片';
    const deleteText = window.i18n?.t('common.delete') || '删除';
    const collapseText = window.i18n?.t('common.collapse') || '收起';
    
    let imageListHTML = '';
    if (state.imageList.length === 0) {
        imageListHTML = `<div class="sidebar-empty">${noImagesText}</div>`;
    } else {
        state.imageList.forEach((imgData, index) => {
            const isActive = (state.currentImageIndex >= 0 && index === state.currentImageIndex) ? 'active' : '';
            const imageAlt = window.i18n?.t('sidebar.imageAlt', { n: index + 1 }) || `图片${index + 1}`;
            imageListHTML += `
                <div class="sidebar-image-item ${isActive}" data-index="${index}">
                    <img src="${imgData.thumbnail}" class="sidebar-thumbnail" alt="${imageAlt}">
                    <div class="sidebar-image-actions">
                        <button class="sidebar-btn-delete" title="${deleteText}">✕</button>
                    </div>
                </div>
            `;
        });
    }
    
    sidebarElement.innerHTML = `
        <div class="sidebar-header"><span class="sidebar-header-text">${imageListText}</span></div>
        <div class="sidebar-content">
            ${imageListHTML}
        </div>
        <button class="sidebar-import-btn" id="btnImportImageSidebar">
            <img src="assets/icon/file-earmark-medical.svg" width="16" height="16" alt="${importImageText}" style="filter: invert(1);">
            ${importImageText}
        </button>
    `;
    dom.canvasContainer.appendChild(sidebarElement);
    
    document.getElementById('btnImportImageSidebar')?.addEventListener('click', importImage);
    
    document.querySelectorAll('.sidebar-image-item').forEach(item => {
        const index = parseInt(item.dataset.index);
        
        item.querySelector('.sidebar-btn-delete')?.addEventListener('click', (e) => {
            e.stopPropagation();
            deleteImage(index);
        });
        
        item.addEventListener('click', () => selectImage(index));
    });
    
    dom.btnExpand.innerHTML = `
        <img src="assets/icon/caret-down-fill.svg" width="16" height="16" alt="${collapseText}" style="filter: invert(1);">
        ${collapseText}
    `;
    console.log('展开侧边栏');
}

async function selectImage(index) {
    if (index < 0 || index >= state.imageList.length) return;
    
    if (index === state.currentImageIndex && state.currentImage) {
        (async () => {
            try {
                // 保存摄像头视图状态和批注
                if (state.isCameraOpen) {
                    state.cameraViewState = {
                        scale: state.scale,
                        canvasX: state.canvasX,
                        canvasY: state.canvasY,
                        strokeHistory: cloneStrokes(state.strokeHistory),
                        baseImageURL: state.baseImageURL
                    };
                }
                
                saveCurrentDrawData();
                saveCurrentFolderPageDrawData();
                state.currentImageIndex = -1;
                state.currentImage = null;
                clearImageLayer();
                clearDrawCanvas();
                if (state.isCameraOpen) {
                    await setCameraState(false);
                }
                
                // 恢复摄像头视图状态和批注
                state.scale = state.cameraViewState.scale;
                state.canvasX = state.cameraViewState.canvasX;
                state.canvasY = state.cameraViewState.canvasY;
                state.strokeHistory = cloneStrokes(state.cameraViewState.strokeHistory);
                state.baseImageURL = state.cameraViewState.baseImageURL;
                state.baseImageObj = null;
                updateMoveBound();
                updateCanvasTransform();
                
                // 恢复批注
                if (state.strokeHistory.length > 0) {
                    await redrawAllStrokes();
                } else {
                    clearDrawCanvas();
                }
                updateUndoBtnStatus();
                
                await setCameraState(true);
                updateSidebarSelection();
                updatePhotoButtonState();
                updateEnhanceButtonState();
                console.log('返回摄像头');
            } catch (error) {
                console.error('返回摄像头失败:', error);
            }
        })();
        return;
    }
    
    // 保存摄像头视图状态和批注
    if (state.isCameraOpen) {
        state.cameraViewState = {
            scale: state.scale,
            canvasX: state.canvasX,
            canvasY: state.canvasY,
            strokeHistory: cloneStrokes(state.strokeHistory),
            baseImageURL: state.baseImageURL
        };
    }
    
    // 使用源ID管理系统切换
    const imgData = state.imageList[index];
    if (imgData.sourceId) {
        await switchToSource(imgData.sourceId);
    }
    
    saveCurrentDrawData();
    saveCurrentFolderPageDrawData();
    
    state.currentImageIndex = index;
    state.currentFolderIndex = -1;
    state.currentFolderPageIndex = -1;
    
    const img = new Image();
    img.onload = async () => {
        state.currentImage = img;
        
        if (state.isCameraOpen) {
            await setCameraState(false);
        }
        drawImageToCenter(img);
        
        await redrawAllStrokes();
        updateSidebarSelection();
        updatePhotoButtonState();
        updateEnhanceButtonState();
    };
    img.onerror = () => {
        console.error(`加载图片 ${index + 1} 失败`);
    };
    img.src = imgData.full;
    
    console.log(`切换到图片 ${index + 1}`);
}

function saveCurrentDrawData() {
    if (state.currentImageIndex >= 0 && state.currentImageIndex < state.imageList.length) {
        state.imageList[state.currentImageIndex].strokeHistory = cloneStrokes(state.strokeHistory);
        state.imageList[state.currentImageIndex].baseImageURL = state.baseImageURL;
        state.imageList[state.currentImageIndex].viewState = {
            scale: state.scale,
            canvasX: state.canvasX,
            canvasY: state.canvasY
        };
    }
}

async function restoreDrawData(index) {
    if (index >= 0 && index < state.imageList.length) {
        const imgData = state.imageList[index];
        if (imgData.strokeHistory && imgData.strokeHistory.length > 0) {
            state.strokeHistory = cloneStrokes(imgData.strokeHistory);
        } else {
            state.strokeHistory = [];
        }
        
        state.baseImageURL = imgData.baseImageURL || null;
        state.baseImageObj = null;
        
        if (state.baseImageURL) {
            const img = new Image();
            img.onload = async () => {
                state.baseImageObj = img;
                await redrawAllStrokes();
                updateUndoBtnStatus();
            };
            img.src = state.baseImageURL;
        } else {
            if (state.strokeHistory.length > 0) {
                await redrawAllStrokes();
            } else {
                clearDrawCanvas();
            }
            updateUndoBtnStatus();
        }
    }
}

function deleteImage(index) {
    if (index < 0 || index >= state.imageList.length) return;
    
    const imgData = state.imageList[index];
    if (imgData.full && imgData.full.startsWith('blob:')) {
        URL.revokeObjectURL(imgData.full);
    }
    if (imgData.thumbnail && imgData.thumbnail.startsWith('blob:')) {
        URL.revokeObjectURL(imgData.thumbnail);
    }
    
    state.imageList.splice(index, 1);
    
    if (state.currentImageIndex === index) {
        if (state.imageList.length > 0) {
            const newIndex = Math.min(index, state.imageList.length - 1);
            state.currentImageIndex = -1;
            selectImage(newIndex);
        } else {
            state.currentImageIndex = -1;
            state.currentImage = null;
            clearImageLayer();
            clearDrawCanvas();
            updatePhotoButtonState();
            openCamera();
        }
    } else if (state.currentImageIndex > index) {
        state.currentImageIndex--;
    }
    
    updateSidebarContent();
    console.log(`删除图片 ${index + 1}`);
}

let lastSidebarSelection = -2;

function updateSidebarSelection() {
    if (lastSidebarSelection === state.currentImageIndex) return;
    
    const sidebarContent = document.querySelector('.sidebar:not(.file-sidebar) .sidebar-content');
    if (!sidebarContent) return;
    
    const items = sidebarContent.querySelectorAll('.sidebar-image-item');
    
    if (lastSidebarSelection >= 0 && lastSidebarSelection < items.length) {
        items[lastSidebarSelection].classList.remove('active');
    }
    
    if (state.currentImageIndex >= 0 && state.currentImageIndex < items.length) {
        items[state.currentImageIndex].classList.add('active');
    }
    
    lastSidebarSelection = state.currentImageIndex;
    
    document.querySelectorAll('.file-sidebar .sidebar-image-item').forEach(item => {
        item.classList.remove('active');
    });
}

function updateSidebarContent() {
    const sidebarContent = document.querySelector('.sidebar-content');
    if (!sidebarContent) return;
    
    const noImagesText = window.i18n?.t('common.noImages') || '暂无图片';
    const deleteText = window.i18n?.t('common.delete') || '删除';
    
    let imageListHTML = '';
    if (state.imageList.length === 0) {
        imageListHTML = `<div class="sidebar-empty">${noImagesText}</div>`;
    } else {
        state.imageList.forEach((imgData, index) => {
            const isActive = (state.currentImageIndex >= 0 && index === state.currentImageIndex) ? 'active' : '';
            const imageAlt = window.i18n?.t('sidebar.imageAlt', { n: index + 1 }) || `图片${index + 1}`;
            imageListHTML += `
                <div class="sidebar-image-item ${isActive}" data-index="${index}">
                    <img src="${imgData.thumbnail}" class="sidebar-thumbnail" alt="${imageAlt}">
                    <div class="sidebar-image-actions">
                        <button class="sidebar-btn-delete" title="${deleteText}">✕</button>
                    </div>
                </div>
            `;
        });
    }
    
    sidebarContent.innerHTML = imageListHTML;
    
    document.querySelectorAll('.sidebar-image-item').forEach(item => {
        const index = parseInt(item.dataset.index);
        
        item.querySelector('.sidebar-btn-delete')?.addEventListener('click', (e) => {
            e.stopPropagation();
            deleteImage(index);
        });
        
        item.addEventListener('click', () => selectImage(index));
    });
}

function collapseSidebar() {
    const sidebar = document.querySelector('.sidebar');
    if (sidebar) {
        sidebar.classList.add('collapse');
        sidebar.addEventListener('animationend', function() {
            sidebar.remove();
        }, { once: true });
    }
    
    const imageText = window.i18n?.t('toolbar.image') || '图片';
    dom.btnExpand.innerHTML = `
        <img src="assets/icon/file-earmark-medical.svg" width="16" height="16" alt="${imageText}" style="filter: invert(1);">
        ${imageText}
    `;
    console.log('收起侧边栏');
}

// 文件侧边栏
function toggleFileSidebar() {
    const existingFileSidebar = document.querySelector('.file-sidebar');
    const existingSidebar = document.querySelector('.sidebar:not(.file-sidebar)');
    
    if (existingSidebar) {
        collapseSidebar();
    }
    
    if (existingFileSidebar) {
        collapseFileSidebar();
    } else {
        expandFileSidebar();
    }
}

function expandFileSidebar() {
    const existingSidebar = document.querySelector('.file-sidebar');
    if (existingSidebar) {
        updateFileSidebarContent();
        return;
    }
    
    const noFilesText = window.i18n?.t('common.noFiles') || '暂无文件';
    const fileListText = window.i18n?.t('sidebar.fileList') || '文件列表';
    const addFileText = window.i18n?.t('sidebar.addFile') || '添加文件';
    const collapseText = window.i18n?.t('common.collapse') || '收起';
    
    const fileSidebarElement = document.createElement('div');
    fileSidebarElement.classList.add('sidebar', 'file-sidebar');
    
    let contentHTML = '';
    if (state.fileList.length === 0) {
        contentHTML = `<div class="sidebar-empty">${noFilesText}</div>`;
    } else {
        state.fileList.forEach((folder, index) => {
            const isWord = folder.isWord === true;
            const iconSrc = isWord 
                ? 'assets/icon/file-earmark-word-fill.svg' 
                : 'assets/icon/pdf.svg';
            const fileAlt = window.i18n?.t('toolbar.file') || '文件';
            const pagesText = window.i18n?.t('sidebar.pages', { n: folder.pages.length }) || `${folder.pages.length}页`;
            contentHTML += `
                <div class="sidebar-folder-item" data-index="${index}">
                    <img src="${iconSrc}" width="16" height="16" alt="${fileAlt}" style="filter: invert(1);">
                    <span class="folder-name">${folder.name}</span>
                    <span class="folder-count">${pagesText}</span>
                </div>
            `;
        });
    }
    
    fileSidebarElement.innerHTML = `
        <div class="sidebar-header"><span class="sidebar-header-text">${fileListText}</span></div>
        <div class="sidebar-content">
            ${contentHTML}
        </div>
        <button class="sidebar-import-btn" id="btnAddFile">
            <img src="assets/icon/file-earmark.svg" width="16" height="16" alt="${addFileText}" style="filter: invert(1);">
            ${addFileText}
        </button>
    `;
    
    dom.canvasContainer.appendChild(fileSidebarElement);
    
    document.getElementById('btnAddFile')?.addEventListener('click', () => {
        importPDF();
    });
    
    document.querySelectorAll('.sidebar-folder-item').forEach(item => {
        item.addEventListener('click', () => {
            const index = parseInt(item.dataset.index);
            openFolder(index);
        });
    });
    
    dom.btnSave.innerHTML = `
        <img src="assets/icon/caret-down-fill.svg" width="16" height="16" alt="${collapseText}" style="filter: invert(1);">
        ${collapseText}
    `;
    console.log('展开文件侧边栏');
    
    if (state.currentFolderIndex >= 0 && state.currentFolderPageIndex >= 0) {
        openFolder(state.currentFolderIndex);
    }
}

function openFolder(folderIndex) {
    if (folderIndex < 0 || folderIndex >= state.fileList.length) return;
    
    const folder = state.fileList[folderIndex];
    
    const sidebarContent = document.querySelector('.file-sidebar .sidebar-content');
    const sidebarHeader = document.querySelector('.file-sidebar .sidebar-header');
    
    if (!sidebarContent || !sidebarHeader) return;
    
    sidebarHeader.innerHTML = `
        <button class="folder-back-btn" id="btnBackFolder">←</button>
        <span class="sidebar-header-text">${folder.name}</span>
    `;
    sidebarHeader.classList.add('has-back');
    
    let pagesHTML = '';
    folder.pages.forEach((page, index) => {
        const isActive = (state.currentFolderIndex === folderIndex && state.currentFolderPageIndex === index) ? 'active' : '';
        const pageLabel = window.i18n?.t('sidebar.page', { n: index + 1 }) || `第${index + 1}页`;
        pagesHTML += `
            <div class="sidebar-image-item ${isActive}" data-folder="${folderIndex}" data-page="${index}">
                <img src="${page.thumbnail}" class="sidebar-thumbnail" alt="${pageLabel}">
                <div class="sidebar-page-label">${pageLabel}</div>
            </div>
        `;
    });
    
    sidebarContent.innerHTML = pagesHTML;
    
    document.getElementById('btnBackFolder')?.addEventListener('click', () => {
        closeFolder();
    });
    
    document.querySelectorAll('.file-sidebar .sidebar-image-item').forEach(item => {
        item.addEventListener('click', () => {
            const folderIdx = parseInt(item.dataset.folder);
            const pageIdx = parseInt(item.dataset.page);
            selectFolderPage(folderIdx, pageIdx);
        });
    });
    
    console.log(`打开文件夹: ${folder.name}`);
}

function closeFolder() {
    const fileListText = window.i18n?.t('sidebar.fileList') || '文件列表';
    const sidebarHeader = document.querySelector('.file-sidebar .sidebar-header');
    if (sidebarHeader) {
        sidebarHeader.innerHTML = `<span class="sidebar-header-text">${fileListText}</span>`;
        sidebarHeader.classList.remove('has-back');
    }
    
    updateFileSidebarContent();
    console.log('关闭文件夹');
}

function selectFolderPage(folderIndex, pageIndex) {
    if (folderIndex < 0 || folderIndex >= state.fileList.length) return;
    
    const folder = state.fileList[folderIndex];
    if (pageIndex < 0 || pageIndex >= folder.pages.length) return;
    
    (async () => {
        try {
            // 保存摄像头视图状态和批注
            if (state.isCameraOpen) {
                state.cameraViewState = {
                    scale: state.scale,
                    canvasX: state.canvasX,
                    canvasY: state.canvasY,
                    strokeHistory: cloneStrokes(state.strokeHistory),
                    baseImageURL: state.baseImageURL
                };
                await setCameraState(false);
            }
            
            saveCurrentDrawData();
            saveCurrentFolderPageDrawData();
            
            const page = folder.pages[pageIndex];
            
            // 使用源ID管理系统切换
            if (page.sourceId) {
                await switchToSource(page.sourceId);
            } else {
                // 兼容旧数据
                if (page.viewState) {
                    state.scale = page.viewState.scale;
                    state.canvasX = page.viewState.canvasX;
                    state.canvasY = page.viewState.canvasY;
                    updateMoveBound();
                    updateCanvasTransform();
                } else {
                    state.scale = 1;
                    state.canvasX = -(DRAW_CONFIG.canvasW - DRAW_CONFIG.screenW) / 2;
                    state.canvasY = -(DRAW_CONFIG.canvasH - DRAW_CONFIG.screenH) / 2;
                    updateMoveBound();
                    updateCanvasTransform();
                }
            }
            
            const img = new Image();
            img.onload = async () => {
                state.currentImage = img;
                state.currentImageIndex = -1;
                state.currentFolderIndex = folderIndex;
                state.currentFolderPageIndex = pageIndex;
                drawImageToCenter(img);
                
                // 如果有源ID，数据已经通过 switchToSource 加载
                if (!page.sourceId) {
                    await restoreFolderPageDrawData(folderIndex, pageIndex);
                } else {
                    await redrawAllStrokes();
                }
                
                updateFolderPageSelection(folderIndex, pageIndex);
                updatePhotoButtonState();
                updateEnhanceButtonState();
            };
            img.src = page.full;
            
            console.log(`选择: ${folder.name} 第${pageIndex + 1}页`);
        } catch (error) {
            console.error('选择文件夹页面失败:', error);
        }
    })();
}

function updateFolderPageSelection(folderIndex, pageIndex) {
    document.querySelectorAll('.file-sidebar .sidebar-image-item').forEach((item, idx) => {
        const itemFolder = parseInt(item.dataset.folder);
        const itemPage = parseInt(item.dataset.page);
        if (itemFolder === folderIndex && itemPage === pageIndex) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    });
    
    document.querySelectorAll('.sidebar:not(.file-sidebar) .sidebar-image-item').forEach(item => {
        item.classList.remove('active');
    });
    state.currentImageIndex = -1;
}

function saveCurrentFolderPageDrawData() {
    if (state.currentFolderIndex >= 0 && state.currentFolderPageIndex >= 0) {
        if (state.currentFolderIndex < state.fileList.length) {
            const folder = state.fileList[state.currentFolderIndex];
            if (state.currentFolderPageIndex < folder.pages.length) {
                folder.pages[state.currentFolderPageIndex].strokeHistory = cloneStrokes(state.strokeHistory);
                folder.pages[state.currentFolderPageIndex].baseImageURL = state.baseImageURL;
                folder.pages[state.currentFolderPageIndex].viewState = {
                    scale: state.scale,
                    canvasX: state.canvasX,
                    canvasY: state.canvasY
                };
            }
        }
    }
}

async function restoreFolderPageDrawData(folderIndex, pageIndex) {
    if (folderIndex >= 0 && folderIndex < state.fileList.length) {
        const folder = state.fileList[folderIndex];
        if (pageIndex >= 0 && pageIndex < folder.pages.length) {
            const page = folder.pages[pageIndex];
            if (page.strokeHistory && page.strokeHistory.length > 0) {
                state.strokeHistory = cloneStrokes(page.strokeHistory);
            } else {
                state.strokeHistory = [];
            }
            
            state.baseImageURL = page.baseImageURL || null;
            state.baseImageObj = null;
            
            if (state.baseImageURL) {
                const img = new Image();
                img.onload = async () => {
                    state.baseImageObj = img;
                    await redrawAllStrokes();
                    updateUndoBtnStatus();
                };
                img.src = state.baseImageURL;
            } else {
                if (state.strokeHistory.length > 0) {
                    await redrawAllStrokes();
                } else {
                    clearDrawCanvas();
                }
                updateUndoBtnStatus();
            }
        }
    }
}

function updateFileSidebarContent() {
    const sidebarContent = document.querySelector('.file-sidebar .sidebar-content');
    if (!sidebarContent) return;
    
    const noFilesText = window.i18n?.t('common.noFiles') || '暂无文件';
    
    let contentHTML = '';
    if (state.fileList.length === 0) {
        contentHTML = `<div class="sidebar-empty">${noFilesText}</div>`;
    } else {
        state.fileList.forEach((folder, index) => {
            const isWord = folder.isWord === true;
            const iconSrc = isWord 
                ? 'assets/icon/file-earmark-word-fill.svg' 
                : 'assets/icon/pdf.svg';
            console.log(`文件夹 ${folder.name}: isWord=${isWord}, iconSrc=${iconSrc}`);
            const fileAlt = window.i18n?.t('toolbar.file') || '文件';
            const pagesText = window.i18n?.t('sidebar.pages', { n: folder.pages.length }) || `${folder.pages.length}页`;
            contentHTML += `
                <div class="sidebar-folder-item" data-index="${index}">
                    <img src="${iconSrc}" width="16" height="16" alt="${fileAlt}" style="filter: invert(1);" onerror="this.style.display='none'">
                    <span class="folder-name">${folder.name}</span>
                    <span class="folder-count">${pagesText}</span>
                </div>
            `;
        });
    }
    
    sidebarContent.innerHTML = contentHTML;
    
    document.querySelectorAll('.sidebar-folder-item').forEach(item => {
        item.addEventListener('click', () => {
            const index = parseInt(item.dataset.index);
            openFolder(index);
        });
    });
}

function importPDF() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.pdf,.docx,.doc';
    
    input.onchange = async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        
        // 先保存当前批注数据，再关闭摄像头
        saveCurrentDrawData();
        saveCurrentFolderPageDrawData();
        
        const wasCameraOpen = state.isCameraOpen;
        
        if (state.isCameraOpen) {
            await setCameraState(false);
        }
        
        const fileName = file.name.toLowerCase();
        const isWord = fileName.endsWith('.docx') || fileName.endsWith('.doc');
        
        if (isWord) {
            showLoadingOverlay(window.i18n?.t('loading.detectingOffice') || '正在检测 Office 软件...');
            
            const { invoke } = window.__TAURI__.core;
            
            let detection;
            try {
                detection = await invoke('detect_office');
                console.log('Office 检测结果:', detection);
                if (detection.recommended === 'None') {
                    hideLoadingOverlay();
                    showErrorDialog(
                        window.i18n?.t('errors.officeNotInstalled') || 'Office 未安装',
                        window.i18n?.t('errors.officeNotInstalledDesc') || '未检测到可用的 Office 软件\n\n请安装以下软件之一：\n• Microsoft Word\n• WPS Office\n• LibreOffice\n\n或将 Word 文档另存为 PDF 后导入'
                    );
                    if (wasCameraOpen) await setCameraState(true);
                    return;
                }
            } catch (e) {
                hideLoadingOverlay();
                console.log('检测 Office 失败:', e);
                showErrorDialog(
                    window.i18n?.t('errors.officeDetectFailed') || '检测失败',
                    window.i18n?.t('errors.officeDetectFailedDesc') || '检测 Office 软件失败，请重试'
                );
                if (wasCameraOpen) await setCameraState(true);
                return;
            }
            
            updateLoadingProgress(window.i18n?.t('loading.readingFile') || '正在读取文件...');
            
            const arrayBuffer = await file.arrayBuffer();
            const uint8Array = new Uint8Array(arrayBuffer);
            
            console.log('文件大小:', uint8Array.length, '字节');
            
            updateLoadingProgress(window.i18n?.t('loading.processingWord') || '正在处理 Word 文档...');
            
            let pdfPath = null;
            try {
                pdfPath = await invoke('convert_docx_to_pdf_from_bytes', { 
                    fileData: Array.from(uint8Array),
                    fileName: file.name
                });
                console.log('Word 文档已转换为 PDF:', pdfPath);
            } catch (convertError) {
                hideLoadingOverlay();
                console.error('Word 转换失败:', convertError);
                const errorMsg = String(convertError);
                let friendlyMsg = window.i18n?.t('errors.wordConvertFailed') || 'Word 文档转换失败';
                
                if (errorMsg.includes('Office') || errorMsg.includes('Word') || errorMsg.includes('WPS')) {
                    friendlyMsg = window.i18n?.t('errors.officeCallFailed') || 'Office 软件调用失败\n\n可能的原因：\n• Office 软件未正确安装\n• 文件被其他程序占用\n• 文件格式不支持';
                }
                
                showErrorDialog(
                    window.i18n?.t('errors.convertFailed') || '转换失败',
                    friendlyMsg,
                    () => {
                        importPDF();
                    }
                );
                if (wasCameraOpen) await setCameraState(true);
                return;
            }
            
            updateLoadingProgress(window.i18n?.t('loading.renderingPage') || '正在渲染页面...');
            
            try {
                const pdfReady = await waitForPdfJs();
                if (!pdfReady) {
                    hideLoadingOverlay();
                    showErrorDialog(
                        window.i18n?.t('errors.loadFailed') || '加载失败',
                        window.i18n?.t('errors.pdfLoadTimeout') || 'PDF 库加载超时\n\n请重启应用后重试'
                    );
                    if (wasCameraOpen) await setCameraState(true);
                    return;
                }
                
                const { readFile, remove } = window.__TAURI__.fs;
                const pdfBytes = await readFile(pdfPath);
                const pdfArrayBuffer = pdfBytes.buffer;
                const pdf = await pdfjsLib.getDocument({ data: pdfArrayBuffer }).promise;
                
                const totalPages = pdf.numPages;
                const folder = {
                    name: file.name.replace(/\.(pdf|docx|doc)$/i, ''),
                    pages: [],
                    isWord: true
                };
                
                sourceIdCounters.doc++;  // 增加文档计数器
                const docNumber = sourceIdCounters.doc;
                
                folder.pages = await processPdfPagesParallel(pdf, totalPages, 4, docNumber);
                
                state.fileList.push(folder);
                updateFileSidebarContent();
                
                const existingFileSidebar = document.querySelector('.file-sidebar');
                if (!existingFileSidebar) {
                    expandFileSidebar();
                }
                
                if (folder.pages.length > 0) {
                    const firstPage = folder.pages[0];
                    const img = new Image();
                    img.onload = async () => {
                        state.currentImage = img;
                        state.currentFolderIndex = state.fileList.length - 1;
                        state.currentFolderPageIndex = 0;
                        
                        // 切换到新源ID
                        if (firstPage.sourceId) {
                            await switchToSource(firstPage.sourceId);
                        }
                        
                        drawImageToCenter(img);
                        updatePhotoButtonState();
                        updateEnhanceButtonState();
                    };
                    img.src = firstPage.full;
                }
                
                hideLoadingOverlay();
                console.log(`文件已导入: ${folder.name}，共${folder.pages.length}页`);
                
                try {
                    await remove(pdfPath);
                } catch (e) {
                    console.log('清理转换的 PDF 失败:', e);
                }
            } catch (error) {
                hideLoadingOverlay();
                console.error('文件导入失败:', error);
                showErrorDialog(
                    window.i18n?.t('errors.importFailed') || '导入失败',
                    window.i18n?.t('errors.importFailedDesc') || '文件导入失败，请确保文件格式正确'
                );
                if (wasCameraOpen) await setCameraState(true);
            }
        } else {
            showLoadingOverlay(window.i18n?.t('loading.importingFile') || '正在导入文件...');
            
            try {
                const pdfReady = await waitForPdfJs();
                if (!pdfReady) {
                    hideLoadingOverlay();
                    showErrorDialog(
                        window.i18n?.t('errors.loadFailed') || '加载失败',
                        window.i18n?.t('errors.pdfLoadTimeout') || 'PDF 库加载超时\n\n请重启应用后重试'
                    );
                    if (wasCameraOpen) await setCameraState(true);
                    return;
                }
                
                const pdfArrayBuffer = await file.arrayBuffer();
                const pdf = await pdfjsLib.getDocument({ data: pdfArrayBuffer }).promise;
                
                const totalPages = pdf.numPages;
                const folder = {
                    name: file.name.replace('.pdf', ''),
                    pages: []
                };
                
                sourceIdCounters.doc++;  // 增加文档计数器
                const docNumber = sourceIdCounters.doc;
                
                folder.pages = await processPdfPagesParallel(pdf, totalPages, 4, docNumber);
                
                state.fileList.push(folder);
                updateFileSidebarContent();
                
                const existingFileSidebar = document.querySelector('.file-sidebar');
                if (!existingFileSidebar) {
                    expandFileSidebar();
                }
                
                if (folder.pages.length > 0) {
                    const firstPage = folder.pages[0];
                    const img = new Image();
                    img.onload = async () => {
                        state.currentImage = img;
                        state.currentFolderIndex = state.fileList.length - 1;
                        state.currentFolderPageIndex = 0;
                        
                        // 切换到新源ID
                        if (firstPage.sourceId) {
                            await switchToSource(firstPage.sourceId);
                        }
                        
                        drawImageToCenter(img);
                        updatePhotoButtonState();
                        updateEnhanceButtonState();
                    };
                    img.src = firstPage.full;
                }
                
                hideLoadingOverlay();
                console.log(`文件已导入: ${folder.name}，共${folder.pages.length}页`);
            } catch (error) {
                hideLoadingOverlay();
                console.error('文件导入失败:', error);
                showErrorDialog(
                    window.i18n?.t('errors.importFailed') || '导入失败',
                    window.i18n?.t('errors.importFailedDesc') || '文件导入失败，请确保文件格式正确'
                );
                if (wasCameraOpen) await setCameraState(true);
            }
        }
    };
    
    input.click();
}

function showLoadingOverlay(message) {
    const overlay = document.createElement('div');
    overlay.id = 'loadingOverlay';
    overlay.className = 'loading-overlay';
    overlay.innerHTML = `
        <div class="loading-content">
            <div class="loading-spinner"></div>
            <div class="loading-message" id="loadingMessage">${message}</div>
        </div>
    `;
    document.body.appendChild(overlay);
}

function updateLoadingProgress(message) {
    const loadingMessage = document.getElementById('loadingMessage');
    if (loadingMessage) {
        loadingMessage.textContent = message;
    }
}

function hideLoadingOverlay() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.remove();
    }
}

function showErrorDialog(title, message, retryCallback = null) {
    const existing = document.getElementById('errorDialog');
    if (existing) existing.remove();
    
    const retryText = window.i18n?.t('common.retry') || '重试';
    const closeText = window.i18n?.t('common.close') || '关闭';
    
    const dialog = document.createElement('div');
    dialog.id = 'errorDialog';
    dialog.className = 'error-dialog-overlay';
    dialog.innerHTML = `
        <div class="error-dialog">
            <div class="error-icon">⚠️</div>
            <div class="error-title">${title}</div>
            <div class="error-message">${message}</div>
            <div class="error-buttons">
                ${retryCallback ? `<button class="error-btn error-btn-retry" id="errorRetry">${retryText}</button>` : ''}
                <button class="error-btn error-btn-close" id="errorClose">${closeText}</button>
            </div>
        </div>
    `;
    document.body.appendChild(dialog);
    
    const closeBtn = document.getElementById('errorClose');
    const retryBtn = document.getElementById('errorRetry');
    
    closeBtn?.addEventListener('click', () => {
        dialog.remove();
    });
    
    retryBtn?.addEventListener('click', () => {
        dialog.remove();
        if (retryCallback) retryCallback();
    });
    
    dialog.addEventListener('click', (e) => {
        if (e.target === dialog) {
            dialog.remove();
        }
    });
}

function collapseFileSidebar() {
    const fileSidebar = document.querySelector('.file-sidebar');
    if (fileSidebar) {
        fileSidebar.classList.add('collapse');
        fileSidebar.addEventListener('animationend', function() {
            fileSidebar.remove();
        }, { once: true });
    }
    
    const fileText = window.i18n?.t('toolbar.file') || '文件';
    dom.btnSave.innerHTML = `
        <img src="assets/icon/File.svg" width="16" height="16" alt="${fileText}" style="filter: invert(1);">
        ${fileText}
    `;
    console.log('收起文件侧边栏');
}

// 触控反馈
document.addEventListener('DOMContentLoaded', function() {
    const buttons = document.querySelectorAll('button');
    buttons.forEach(button => {
        button.addEventListener('touchstart', function(e) {
            this.style.transform = 'scale(0.95)';
            this.style.transition = 'transform 0.1s ease';
        }, { passive: true });
        
        button.addEventListener('touchend', function(e) {
            this.style.transform = '';
        }, { passive: true });
        
        button.addEventListener('touchcancel', function(e) {
            this.style.transform = '';
        }, { passive: true });
    });
    
    // 设置窗口最小化监听器
    setupWindowMinimizeListeners();
});

// ==================== 摄像头功能 ====================
// 摄像头开启/关闭、帧渲染、拍照、旋转、镜像等

/**
 * 统一的摄像头状态管理函数
 * @param {boolean} open - true: 开启摄像头, false: 关闭摄像头
 * @param {Object} options - 可选参数
 * @param {boolean} options.forceClose - 强制关闭（不保存状态用于恢复）
 */
async function setCameraState(open, options = {}) {
    const { forceClose = false } = options;
    
    if (open) {
        if (state.isCameraOpen) {
            return;
        }
        
        try {
            // 保存当前源数据
            saveCurrentSourceData();
            
            let constraints;
            
            if (state.defaultCameraId) {
                constraints = {
                    video: {
                        deviceId: { ideal: state.defaultCameraId },
                        width: { ideal: state.cameraWidth || 1280 },
                        height: { ideal: state.cameraHeight || 720 }
                    },
                    audio: false
                };
            } else {
                constraints = {
                    video: {
                        width: { ideal: state.cameraWidth || 1280 },
                        height: { ideal: state.cameraHeight || 720 },
                        facingMode: state.useFrontCamera ? 'user' : 'environment'
                    },
                    audio: false
                };
            }
            
            try {
                state.cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
            } catch (constraintError) {
                if (constraintError.name === 'OverconstrainedError') {
                    console.warn('摄像头不支持请求的分辨率，使用默认设置');
                    const fallbackConstraints = {
                        video: {
                            facingMode: state.useFrontCamera ? 'user' : 'environment'
                        },
                        audio: false
                    };
                    state.cameraStream = await navigator.mediaDevices.getUserMedia(fallbackConstraints);
                } else {
                    throw constraintError;
                }
            }
            
            state.isCameraOpen = true;
            
            // 切换到摄像头源ID
            await switchToSource('cam');
            
            // 重置图片索引，避免摄像头的批注被错误保存到图片
            state.currentImageIndex = -1;
            state.currentFolderIndex = -1;
            state.currentFolderPageIndex = -1;
            
            hideNoCameraMessage();
            
            const videoTrack = state.cameraStream.getVideoTracks()[0];
            const settings = videoTrack.getSettings();
            const label = videoTrack.label.toLowerCase();
            state.isMirrored = label.includes('front') || label.includes('user') || label.includes('前置') || settings.facingMode === 'user';
            
            createCameraVideo();
            createCameraControls();
            clearSidebarSelection();
            updateEnhanceButtonState();
            
            console.log('摄像头已打开:', videoTrack.label || '未知设备', '分辨率:', settings.width, 'x', settings.height);
        } catch (error) {
            if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
                console.log('未检测到摄像头');
            } else if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
                console.log('摄像头权限被拒绝');
            } else {
                console.error('无法访问摄像头:', error);
            }
            throw error;
        }
    } else {
        // 关闭摄像头
        if (state.cameraAnimationId) {
            cancelAnimationFrame(state.cameraAnimationId);
            state.cameraAnimationId = null;
        }
        
        if (state.cameraStream) {
            state.cameraStream.getTracks().forEach(track => track.stop());
            state.cameraStream = null;
        }
        
        state.isCameraOpen = false;
        state.isCameraReady = false;
        
        // 隐藏 video 元素
        if (dom.cameraVideo) {
            dom.cameraVideo.style.display = 'none';
            dom.cameraVideo.srcObject = null;
        }
        
        updatePhotoButtonState();
        updateEnhanceButtonState();
        
        // 保存摄像头数据
        saveCurrentSourceData();
        
        // 恢复之前的数据
        if (state.currentImage && state.currentImageIndex >= 0) {
            const imgData = state.imageList[state.currentImageIndex];
            if (imgData && imgData.sourceId) {
                await switchToSource(imgData.sourceId);
            }
            drawImageToCenter(state.currentImage);
        } else if (state.currentImage && state.currentFolderIndex >= 0 && state.currentFolderPageIndex >= 0) {
            const folder = state.fileList[state.currentFolderIndex];
            const page = folder.pages[state.currentFolderPageIndex];
            if (page && page.sourceId) {
                await switchToSource(page.sourceId);
            }
            drawImageToCenter(state.currentImage);
        } else {
            clearImageLayer();
            clearDrawCanvas();
            state.strokeHistory = [];
        }
        
        console.log('摄像头已关闭');
    }
}

/**
 * 打开/关闭摄像头（用户交互入口）
 */
async function openCamera() {
    if (state.isCameraOpen) {
        // 关闭摄像头前保存批注数据
        saveCurrentDrawData();
        saveCurrentFolderPageDrawData();
        await setCameraState(false);
    } else {
        await setCameraState(true);
    }
}

/**
 * 显示无摄像头提示信息（绘制到画布上）
 */
function showNoCameraMessage(message) {
    if (!dom.imageCanvas || !dom.imageCtx) return;
    
    const ctx = dom.imageCtx;
    const width = DRAW_CONFIG.canvasW;
    const height = DRAW_CONFIG.canvasH;
    
    ctx.clearRect(0, 0, width, height);
    
    ctx.fillStyle = 'rgba(30, 30, 30, 0.9)';
    ctx.fillRect(0, 0, width, height);
    
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    
    const centerX = width / 2;
    const centerY = height / 2;
    
    ctx.font = `bold ${Math.round(width * 0.035)}px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`;
    ctx.fillStyle = '#ffffff';
    ctx.fillText('( $ _ $ )', centerX, centerY - height * 0.06);
    
    ctx.font = `${Math.round(width * 0.018)}px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`;
    ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
    ctx.fillText(window.i18n?.t('camera.deviceNotFound') || '找不到展台设备', centerX, centerY + height * 0.015);
    
    ctx.font = `${Math.round(width * 0.012)}px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`;
    ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
    ctx.fillText(message, centerX, centerY + height * 0.045);
}

/**
 * 隐藏无摄像头提示信息
 */
function hideNoCameraMessage() {
    if (dom.imageCtx) {
        dom.imageCtx.clearRect(0, 0, dom.imageCanvas.width, dom.imageCanvas.height);
    }
}

function clearSidebarSelection() {
    document.querySelectorAll('.sidebar:not(.file-sidebar) .sidebar-image-item').forEach(item => {
        item.classList.remove('active');
    });
    document.querySelectorAll('.file-sidebar .sidebar-image-item').forEach(item => {
        item.classList.remove('active');
    });
    state.currentImageIndex = -1;
    state.currentFolderPageIndex = -1;
}

async function switchCamera() {
    state.useFrontCamera = !state.useFrontCamera;
    
    if (state.isCameraOpen) {
        await setCameraState(false);
        await setCameraState(true);
    }
    
    console.log(state.useFrontCamera ? '已切换到前置摄像头' : '已切换到后置摄像头');
}

function createCameraVideo() {
    const video = dom.cameraVideo;
    if (!video) {
        console.error('找不到 video 元素');
        return;
    }
    
    video.srcObject = state.cameraStream;
    video.play();
    
    video.onloadedmetadata = () => {
        state.isCameraReady = true;
        console.log('摄像头视频就绪:', video.videoWidth, 'x', video.videoHeight);
        updateCameraVideoStyle();
        video.style.display = 'block';
    };
}

// 缓存上次 video 样式的值，避免不必要的 DOM 更新
let lastVideoStyleCache = {
    drawW: 0, drawH: 0, offsetX: 0, offsetY: 0,
    rotation: -1, isMirrored: null
};

function updateCameraVideoStyle() {
    const video = dom.cameraVideo;
    if (!video) return;
    
    const videoW = video.videoWidth;
    const videoH = video.videoHeight;
    
    if (!videoW || !videoH) return;
    
    const rotation = state.cameraRotation;
    const isRotated = rotation === 90 || rotation === 270;
    
    // 旋转时交换宽高
    const effectiveVideoW = isRotated ? videoH : videoW;
    const effectiveVideoH = isRotated ? videoW : videoH;
    
    // 在 100% 缩放时，video 适应屏幕高度
    const screenW = DRAW_CONFIG.screenW;
    const screenH = DRAW_CONFIG.screenH;
    const videoRatio = effectiveVideoW / effectiveVideoH;
    const screenRatio = screenW / screenH;
    
    let drawW, drawH;
    if (videoRatio > screenRatio) {
        drawW = screenW;
        drawH = screenW / videoRatio;
    } else {
        drawH = screenH;
        drawW = screenH * videoRatio;
    }
    
    // 计算 video 在 canvas 中的居中偏移
    const canvasW = DRAW_CONFIG.canvasW;
    const canvasH = DRAW_CONFIG.canvasH;
    const offsetX = (canvasW - drawW) / 2;
    const offsetY = (canvasH - drawH) / 2;
    
    // 检查是否有变化，避免不必要的 DOM 更新
    const styleChanged = 
        lastVideoStyleCache.drawW !== drawW ||
        lastVideoStyleCache.drawH !== drawH ||
        lastVideoStyleCache.offsetX !== offsetX ||
        lastVideoStyleCache.offsetY !== offsetY ||
        lastVideoStyleCache.rotation !== rotation ||
        lastVideoStyleCache.isMirrored !== state.isMirrored;
    
    if (!styleChanged) return;
    
    // 更新缓存
    lastVideoStyleCache = { drawW, drawH, offsetX, offsetY, rotation, isMirrored: state.isMirrored };
    
    // 只在值变化时更新 DOM
    video.style.width = drawW + 'px';
    video.style.height = drawH + 'px';
    video.style.left = '0px';
    video.style.top = '0px';
    
    // video 只需要设置额外偏移，容器已经应用了主要的 transform
    let videoTransform = `translate(${offsetX}px, ${offsetY}px)`;
    
    // 镜像
    if (state.isMirrored) {
        videoTransform += ' scaleX(-1)';
    }
    
    // 旋转
    if (rotation !== 0) {
        videoTransform += ` rotate(${rotation}deg)`;
    }
    
    video.style.transform = videoTransform;
    video.style.transformOrigin = '0 0';
}

function startCameraPreview() {
    const video = dom.cameraVideo;
    if (!video) return;
    
    updateCameraVideoStyle();
    video.style.display = 'block';
}

function createCameraControls() {
    updatePhotoButtonState();
}

async function captureCamera() {
    const video = document.getElementById('cameraVideo');
    if (!video) {
        console.error('找不到视频元素');
        return;
    }
    
    if (!state.isCameraReady) {
        console.error('摄像头尚未就绪');
        showErrorDialog(
            window.i18n?.t('camera.notReady') || '摄像头未就绪',
            window.i18n?.t('camera.notReadyDesc') || '摄像头尚未就绪，请稍后再试'
        );
        return;
    }
    
    const videoW = video.videoWidth;
    const videoH = video.videoHeight;
    
    if (!videoW || !videoH) {
        console.error('视频尺寸无效:', videoW, videoH);
        showErrorDialog(
            window.i18n?.t('camera.notReady') || '摄像头未就绪',
            window.i18n?.t('camera.notReadyDesc') || '摄像头尚未就绪，请稍后再试'
        );
        return;
    }
    
    console.log('捕获摄像头画面:', videoW, 'x', videoH);
    
    saveCurrentDrawData();
    saveCurrentFolderPageDrawData();
    
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = videoW;
    tempCanvas.height = videoH;
    const tempCtx = tempCanvas.getContext('2d');
    
    if (state.isMirrored) {
        tempCtx.translate(tempCanvas.width, 0);
        tempCtx.scale(-1, 1);
    }
    
    tempCtx.drawImage(video, 0, 0);
    
    if (state.isMirrored) {
        tempCtx.setTransform(1, 0, 0, 1, 0, 0);
    }
    
    let blob = await new Promise((resolve, reject) => {
        tempCanvas.toBlob(b => {
            if (b) resolve(b);
            else reject(new Error('Failed to create blob'));
        }, 'image/png');
    });
    
    if (window.__TAURI__) {
        try {
            const { invoke } = window.__TAURI__.core;
            const dataUrl = await blobToDataUrl(blob);
            
            if (state.enhanceEnabled) {
                const result = await invoke('save_image_with_enhance', { 
                    imageData: dataUrl,
                    prefix: 'photo',
                    contrast: DRAW_CONFIG.enhanceContrast,
                    brightness: DRAW_CONFIG.enhanceBrightness,
                    saturation: DRAW_CONFIG.enhanceSaturation,
                    sharpen: DRAW_CONFIG.enhanceSharpen
                });
                console.log('图片已保存(增强):', result.path);
                if (result.enhanced_data) {
                    blob = await dataUrlToBlob(result.enhanced_data);
                }
            } else {
                const result = await invoke('save_image', { 
                    imageData: dataUrl,
                    prefix: 'photo'
                });
                console.log('图片已保存:', result.path);
            }
        } catch (error) {
            console.error('保存图片失败:', error);
        }
    }
    
    const blobUrl = URL.createObjectURL(blob);
    const img = new Image();
    img.src = blobUrl;
    img.onload = () => {
        const photoName = window.i18n?.t('camera.photoName', { n: state.imageList.length + 1 }) || `拍摄${state.imageList.length + 1}`;
        addImageToListNoHighlight(img, photoName);
        expandSidebarIfCollapsed();
        console.log('已捕获摄像头画面并保存到图片列表');
    };
}

async function blobToDataUrl(blob) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
    });
}

async function dataUrlToBlob(dataUrl) {
    const response = await fetch(dataUrl);
    return response.blob();
}

function expandSidebarIfCollapsed() {
    const sidebar = document.querySelector('.sidebar');
    if (!sidebar) {
        expandSidebar();
    } else if (sidebar.classList.contains('file-sidebar')) {
        sidebar.remove();
        expandSidebar();
    }
}

// ==================== 图像导入功能 ====================
// 图片导入、拍照保存、PDF处理等

/**
 * 导入图片文件
 * - 支持多选
 * - 批量导入时使用Rust并行生成缩略图
 */
async function importImage() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.multiple = true;
    
    input.onchange = async (e) => {
        const files = Array.from(e.target.files);
        if (files.length === 0) return;
        
        saveCurrentDrawData();
        saveCurrentFolderPageDrawData();
        
        if (state.isCameraOpen) {
            await setCameraState(false);
        }
        
        // 检查是否有大图片（大于2.5MB）
        const hasLargeImage = files.some(file => file.size > 2.5 * 1024 * 1024);
        
        // 如果有大图片或者多个文件，显示加载动画
        if (files.length > 1 || hasLargeImage) {
            showLoadingOverlay(window.i18n?.t('loading.readingImages') || '正在读取图片...');
        }
        
        const imageDataList = [];
        
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            
            if (files.length > 1 || file.size > 2.5 * 1024 * 1024) {
                updateLoadingProgress(window.i18n?.t('loading.readingImage', { current: i + 1, total: files.length }) || `正在读取图片 ${i + 1}/${files.length}...`);
            }
            
            const blobUrl = URL.createObjectURL(file);
            
            const imageName = file.name || window.i18n?.t('sidebar.imageAlt', { n: state.imageList.length + imageDataList.length + 1 }) || `图片${state.imageList.length + imageDataList.length + 1}`;
            imageDataList.push({
                data: blobUrl,
                blob: file,
                name: imageName
            });
        }
        
        let thumbnails = [];
        
        if (window.__TAURI__ && imageDataList.length > 1) {
            try {
                updateLoadingProgress(window.i18n?.t('loading.generatingThumbnails') || '正在并行生成缩略图...');
                const { invoke } = window.__TAURI__.core;
                
                const base64Promises = imageDataList.map(async (imgData) => {
                    return await new Promise((resolve) => {
                        const reader = new FileReader();
                        reader.onload = (e) => resolve(e.target.result);
                        reader.onerror = () => resolve('');
                        reader.readAsDataURL(imgData.blob);
                    });
                });
                const base64Images = await Promise.all(base64Promises);
                
                thumbnails = await invoke('generate_thumbnails_batch', {
                    images: base64Images.map((data, i) => ({
                        data: data,
                        name: imageDataList[i].name
                    })),
                    maxSize: 150,
                    fixedRatio: false
                });
                console.log('Rust 批量缩略图生成完成');
            } catch (error) {
                console.error('Rust 批量缩略图生成失败，使用前端降级方案:', error);
            }
        }
        
        for (let i = 0; i < imageDataList.length; i++) {
            const imgData = imageDataList[i];
            const isLast = (i === imageDataList.length - 1);
            
            const img = new Image();
            await new Promise((resolve) => {
                img.onload = () => resolve();
                img.onerror = () => {
                    console.error(`加载图片失败: ${imgData.name}`);
                    resolve();
                };
                img.src = imgData.data;
            });
            
            let thumbnail;
            if (thumbnails[i] && thumbnails[i].length > 0) {
                const thumbBlob = await fetch(thumbnails[i]).then(r => r.blob());
                thumbnail = URL.createObjectURL(thumbBlob);
            } else {
                thumbnail = await generateThumbnailBlob(imgData.blob, 150);
            }
            
            const newImgData = {
                full: imgData.data,
                thumbnail: thumbnail,
                name: imgData.name,
                width: img.width,
                height: img.height,
                strokeHistory: null,
                baseImageURL: null,
                viewState: null
            };
            
            state.imageList.push(newImgData);
            state.currentImageIndex = state.imageList.length - 1;
            state.currentImage = img;
            state.currentFolderIndex = -1;
            state.currentFolderPageIndex = -1;
            
            clearDrawCanvas();
            state.strokeHistory = [];
            state.baseImageURL = null;
            state.baseImageObj = null;
            state.scale = 1;
            state.canvasX = -(DRAW_CONFIG.canvasW - DRAW_CONFIG.screenW) / 2;
            state.canvasY = -(DRAW_CONFIG.canvasH - DRAW_CONFIG.screenH) / 2;
            updateMoveBound();
            updateCanvasTransform();
            updateUndoBtnStatus();
            
            if (isLast) {
                drawImageToCenter(img);
                updateSidebarContent();
                updatePhotoButtonState();
                updateEnhanceButtonState();
            }
        }
        
        // 如果显示了加载动画，无论文件数量多少，都需要隐藏
        if (files.length > 1 || hasLargeImage) {
            hideLoadingOverlay();
        }
        
        console.log(`已导入 ${imageDataList.length} 张图片`);
    };
    
    input.click();
}

async function addImageToList(img, name, isLast = true) {
    const blob = await fetch(img.src).then(r => r.blob());
    const blobUrl = URL.createObjectURL(blob);
    const thumbnail = await generateThumbnailBlob(blob, 150);
    
    const imgData = {
        full: blobUrl,
        thumbnail: thumbnail,
        name: name,
        width: img.width,
        height: img.height,
        strokeHistory: null,
        baseImageURL: null,
        viewState: null,
        sourceId: generateSourceId('pic')  // 分配源ID
    };
    
    state.imageList.push(imgData);
    state.currentImageIndex = state.imageList.length - 1;
    state.currentImage = img;
    state.currentFolderIndex = -1;
    state.currentFolderPageIndex = -1;
    
    // 切换到新源ID
    await switchToSource(imgData.sourceId);
    
    clearDrawCanvas();
    state.strokeHistory = [];
    state.baseImageURL = null;
    state.baseImageObj = null;
    state.scale = 1;
    state.canvasX = -(DRAW_CONFIG.canvasW - DRAW_CONFIG.screenW) / 2;
    state.canvasY = -(DRAW_CONFIG.canvasH - DRAW_CONFIG.screenH) / 2;
    updateMoveBound();
    updateCanvasTransform();
    updateUndoBtnStatus();
    
    if (isLast) {
        if (state.isCameraOpen) {
            await setCameraState(false);
        }
        img.src = blobUrl;
        drawImageToCenter(img);
        
        updateSidebarContent();
        updatePhotoButtonState();
        updateEnhanceButtonState();
    }
}

async function addImageToListNoHighlight(img, name) {
    const blob = await fetch(img.src).then(r => r.blob());
    const blobUrl = URL.createObjectURL(blob);
    const thumbnail = await generateThumbnailBlob(blob, 150);
    
    const imgData = {
        full: blobUrl,
        thumbnail: thumbnail,
        name: name,
        width: img.width,
        height: img.height,
        strokeHistory: null,
        baseImageURL: null,
        viewState: null
    };
    
    state.imageList.push(imgData);
    
    updateSidebarContent();
}

function drawImageToCenter(img) {
    clearImageLayer();
    hideNoCameraMessage();
    
    const screenW = DRAW_CONFIG.screenW;
    const screenH = DRAW_CONFIG.screenH;
    
    const imgRatio = img.width / img.height;
    const screenRatio = screenW / screenH;
    
    let drawW, drawH, drawX, drawY;
    
    if (imgRatio > screenRatio) {
        drawW = screenW;
        drawH = screenW / imgRatio;
    } else {
        drawH = screenH;
        drawW = screenH * imgRatio;
    }
    
    const canvasW = DRAW_CONFIG.canvasW;
    const canvasH = DRAW_CONFIG.canvasH;
    
    drawX = (canvasW - drawW) / 2;
    drawY = (canvasH - drawH) / 2;
    
    dom.imageCtx.drawImage(img, drawX, drawY, drawW, drawH);
}

function clearImageLayer() {
    dom.imageCtx.clearRect(0, 0, DRAW_CONFIG.canvasW, DRAW_CONFIG.canvasH);
}
