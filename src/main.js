/**
 * ViewStage - 摄像头及PDF展台应用
 * 
 * 架构说明：
 * - 图像层(imageElement) + 批注层(drawCanvas)
 * - 批注系统：笔画记录 + 压缩存储 + 撤销支持
 * - 图像处理：Rust后端并行处理（增强、缩略图、旋转）
 * 
 * 性能优化策略：
 * - 批量处理：使用RAF批量绘制减少重绘次数
 * - 内存优化：使用Blob URL替代Data URL存储图片
 */

import './batch-draw.js';
import ThemeManager from './themes/theme.js';

// ==================== 全局变量 ====================
let lastCanvasTransform = { x: null, y: null, scale: null };
let currentAnimationId = null;
let pendingTransform = null;
let transformRafId = null;

function scheduleTransformUpdate(x, y, scale) {
    if (!pendingTransform) {
        pendingTransform = { x: 0, y: 0, scale: 1 };
    }
    pendingTransform.x = x;
    pendingTransform.y = y;
    pendingTransform.scale = scale;
    
    if (transformRafId === null) {
        transformRafId = requestAnimationFrame(() => {
            if (pendingTransform) {
                const pt = pendingTransform;
                dom.canvasWrapper.style.transform = `translate3d(${pt.x}px, ${pt.y}px, 0) scale(${pt.scale})`;
                lastCanvasTransform.x = pt.x;
                lastCanvasTransform.y = pt.y;
                lastCanvasTransform.scale = pt.scale;
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
    pdfScale: 2,                   // PDF 渲染缩放比例
    imageSmoothingQuality: 'high', // 图像平滑质量
    baseDpr: Math.min(window.devicePixelRatio || 1, 2), // 基础设备像素比
    canvasBgColor: '#2a2a2a',      // 画布背景颜色
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
window.getSafeScale = getSafeScale;

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
        savedStrokeHistory: stroke.savedStrokeHistory ? cloneStrokes(stroke.savedStrokeHistory) : undefined,
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
    
    // 图像管理
    currentImage: null,            // 当前显示的图像 Image 对象
    imageList: [],                 // 图片列表
    currentImageIndex: -1,         // 当前图片索引
    
    // PDF/文件管理
    fileList: [],                  // 文件列表 (PDF等)
    currentFolderIndex: -1,        // 当前文件夹索引
    currentFolderPageIndex: -1,    // 当前页索引
    pdfDocuments: new Map(),       // PDF文档对象缓存 (热加载)
    loadedPages: new Set(),        // 已加载的页面ID集合
    
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
        state.strokeHistory = cloneStrokes(data.strokeHistory || []);
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
window.state = state;

let cachedCanvasRect = null;
let cachedVisibleRect = null;
let cachedVisibleRectScale = null;
let cachedVisibleRectX = null;
let cachedVisibleRectY = null;

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
        
        const startupOverlay = document.getElementById('startupOverlay');
        if (startupOverlay) {
            startupOverlay.classList.add('hidden');
            setTimeout(() => startupOverlay.remove(), 300);
        }
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
                DRAW_CONFIG.baseDpr = DRAW_CONFIG.dpr;
                console.log('已加载设备像素比限制:', settings.dprLimit);
            }
            
            if (settings.pdfScale) {
                DRAW_CONFIG.pdfScale = settings.pdfScale;
                console.log('已加载 PDF 输出分辨率:', settings.pdfScale);
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
            
            // 加载高帧率绘制设置
            if (settings.highFrameRate !== undefined) {
                if (window.batchDrawManager) {
                    window.batchDrawManager.setFrameRate(settings.highFrameRate);
                }
                console.log('已加载高帧率绘制设置:', settings.highFrameRate);
            }
            
            // 加载文档扫描按钮显示设置
            if (settings.showDocScanButton !== undefined) {
                dom.btnDocScan.style.display = settings.showDocScanButton ? '' : 'none';
                console.log('已加载文档扫描按钮显示设置:', settings.showDocScanButton);
            }
            
            // 加载主题设置
            const themeName = settings.theme || 'simplify';
            await ThemeManager.setTheme(themeName);
            console.log('已加载主题设置:', themeName);
            
            // 从主题获取画布背景颜色
            const canvasBgColor = ThemeManager.getCanvasBgColor();
            DRAW_CONFIG.canvasBgColor = canvasBgColor;
            updateCanvasBgColor(canvasBgColor);
            console.log('已加载画布背景颜色:', canvasBgColor);
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
            DRAW_CONFIG.baseDpr = DRAW_CONFIG.dpr;
            console.log('设备像素比限制已更改:', settings.dprLimit);
        }
        
        if (settings.pdfScale !== undefined) {
            DRAW_CONFIG.pdfScale = settings.pdfScale;
            console.log('PDF 输出分辨率已更改:', settings.pdfScale);
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
        
        if (settings.showDocScanButton !== undefined) {
            dom.btnDocScan.style.display = settings.showDocScanButton ? '' : 'none';
            console.log('文档扫描按钮显示已更改:', settings.showDocScanButton);
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
    
    listen('doc-scan-save-image', async (event) => {
        console.log('收到文档扫描保存图片事件');
        const { imageData, name } = event.payload;
        
        if (imageData) {
            const img = new Image();
            img.onload = async () => {
                await addImageToListNoHighlight(img, name);
            };
            img.src = imageData;
        }
    }).catch(err => {
        console.error('doc-scan-save-image 事件监听失败:', err);
    });
}

async function processPdfPagesLazy(pdf, totalPages, initialPages = 3, docNumber = null) {
    const pages = [];
    
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
        
        let sourceId = null;
        if (docNumber !== null) {
            sourceId = `doc-${docNumber}-${pageNum}`;
        }
        
        return {
            full: fullUrl,
            fullBlob: fullBlob,
            thumbnail: fullUrl,
            pageNum: pageNum,
            strokeHistory: [],
            baseImageURL: null,
            viewState: {
                scale: 1,
                canvasX: -(DRAW_CONFIG.canvasW - DRAW_CONFIG.screenW) / 2,
                canvasY: -(DRAW_CONFIG.canvasH - DRAW_CONFIG.screenH) / 2
            },
            sourceId: sourceId,
            loaded: true
        };
    }
    
    const pagesToLoad = Math.min(initialPages, totalPages);
    for (let i = 1; i <= pagesToLoad; i++) {
        updateLoadingProgress(window.i18n?.t('loading.processingPage', { current: i, total: totalPages }) || `正在处理 ${i}/${totalPages} 页`);
        const pageData = await processPage(i);
        pages.push(pageData);
        state.loadedPages.add(pageData.sourceId);
    }
    
    for (let i = pagesToLoad + 1; i <= totalPages; i++) {
        let sourceId = null;
        if (docNumber !== null) {
            sourceId = `doc-${docNumber}-${i}`;
        }
        pages.push({
            full: null,
            fullBlob: null,
            thumbnail: null,
            pageNum: i,
            strokeHistory: [],
            baseImageURL: null,
            viewState: {
                scale: 1,
                canvasX: -(DRAW_CONFIG.canvasW - DRAW_CONFIG.screenW) / 2,
                canvasY: -(DRAW_CONFIG.canvasH - DRAW_CONFIG.screenH) / 2
            },
            sourceId: sourceId,
            loaded: false
        });
    }
    
    return pages;
}

async function loadPdfPage(pdf, pageNum, docNumber) {
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
    
    let sourceId = null;
    if (docNumber !== null) {
        sourceId = `doc-${docNumber}-${pageNum}`;
    }
    
    return {
        full: fullUrl,
        fullBlob: fullBlob,
        thumbnail: fullUrl,
        pageNum: pageNum,
        loaded: true
    };
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
        
        processedCount++;
        updateLoadingProgress(window.i18n?.t('loading.processingPage', { current: processedCount, total: totalPages }) || `正在处理 ${processedCount}/${totalPages} 页`);
        
        let sourceId = null;
        if (docNumber !== null) {
            sourceId = `doc-${docNumber}-${pageNum}`;
        }
        
        return {
            full: fullUrl,
            fullBlob: fullBlob,
            thumbnail: fullUrl,
            pageNum: pageNum,
            strokeHistory: [],
            baseImageURL: null,
            viewState: {
                scale: 1,
                canvasX: -(DRAW_CONFIG.canvasW - DRAW_CONFIG.screenW) / 2,
                canvasY: -(DRAW_CONFIG.canvasH - DRAW_CONFIG.screenH) / 2
            },
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
                isPdf: true,
                pdfDoc: pdf,
                totalPages: totalPages,
                docNumber: sourceIdCounters.doc
            };
            
            sourceIdCounters.doc++;
            const docNumber = sourceIdCounters.doc;
            
            state.pdfDocuments.set(docNumber, pdf);
            
            const processedPages = await processPdfPagesLazy(pdf, totalPages, 3, docNumber);
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
    
    // 批注层：摄像头模式下如果没有批注，可以跳过尺寸设置
    const hasStrokes = state.strokeHistory && state.strokeHistory.length > 0;
    const hasBaseImage = state.baseImageObj !== null;
    
    if (!state.isCameraOpen || hasStrokes || hasBaseImage) {
        dom.drawCanvas.width = DRAW_CONFIG.canvasW * DRAW_CONFIG.dpr;
        dom.drawCanvas.height = DRAW_CONFIG.canvasH * DRAW_CONFIG.dpr;
    }
    
    // 所有层 CSS 尺寸相同
    dom.imageElement.style.width = DRAW_CONFIG.canvasW + 'px';
    dom.imageElement.style.height = DRAW_CONFIG.canvasH + 'px';
    dom.drawCanvas.style.width = DRAW_CONFIG.canvasW + 'px';
    dom.drawCanvas.style.height = DRAW_CONFIG.canvasH + 'px';
    
    // 批注层上下文：只在有内容时设置
    if (!state.isCameraOpen || hasStrokes || hasBaseImage) {
        dom.drawCtx.setTransform(1, 0, 0, 1, 0, 0);
        dom.drawCtx.scale(DRAW_CONFIG.dpr, DRAW_CONFIG.dpr);
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
    dom.imageElement = document.getElementById('imageElement');
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
    
    dom.btnDocScan = document.getElementById('btnDocScan');
    dom.docScanPanel = document.getElementById('docScanPanel');
    dom.scanAutoDetect = document.getElementById('scanAutoDetect');
    dom.scanEnhance = document.getElementById('scanEnhance');
    dom.scanDenoise = document.getElementById('scanDenoise');
    dom.scanBinarize = document.getElementById('scanBinarize');
    dom.btnApplyScan = document.getElementById('btnApplyScan');
    dom.btnCancelScan = document.getElementById('btnCancelScan');
    
    if (!dom.imageElement || !dom.drawCanvas || !dom.canvasContainer) {
        console.error('必需的元素未找到');
        return false;
    }
    
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
    
    // 批注层：物理尺寸 = CSS尺寸 × dpr
    dom.drawCanvas.width = DRAW_CONFIG.canvasW * DRAW_CONFIG.dpr;
    dom.drawCanvas.height = DRAW_CONFIG.canvasH * DRAW_CONFIG.dpr;
    
    // 所有层 CSS 尺寸相同
    dom.imageElement.style.width = DRAW_CONFIG.canvasW + 'px';
    dom.imageElement.style.height = DRAW_CONFIG.canvasH + 'px';
    dom.drawCanvas.style.width = DRAW_CONFIG.canvasW + 'px';
    dom.drawCanvas.style.height = DRAW_CONFIG.canvasH + 'px';
    
    dom.drawCtx.scale(DRAW_CONFIG.dpr, DRAW_CONFIG.dpr);
    
    const dc = dom.drawCtx;
    dc.imageSmoothingEnabled = true;
    dc.imageSmoothingQuality = 'high';
    dc.lineCap = 'round';
    dc.lineJoin = 'round';
    dc.miterLimit = 10;
    
    setPenStyle();
    updateEraserHintSize();
    updateCanvasTransform();
    updateCanvasBgColor(DRAW_CONFIG.canvasBgColor);
    
    dom.btnMove.classList.add('primary-btn');
    
    console.log(`画布初始化: 屏幕 ${screenW}x${screenH}, 画布 ${DRAW_CONFIG.canvasW}x${DRAW_CONFIG.canvasH}`);
}

// 更新画布背景颜色
function updateCanvasBgColor(color) {
    if (dom.canvasContainer) {
        dom.canvasContainer.style.backgroundColor = color;
    }
    if (dom.canvasWrapper) {
        dom.canvasWrapper.style.backgroundColor = color;
    }
}

let cachedMoveBoundScale = null;

function updateMoveBound() {
    if (cachedMoveBoundScale === state.scale) {
        return;
    }
    cachedMoveBoundScale = state.scale;
    
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

function getVisibleRect() {
    if (cachedVisibleRectScale === state.scale && 
        cachedVisibleRectX === state.canvasX && 
        cachedVisibleRectY === state.canvasY && 
        cachedVisibleRect) {
        return cachedVisibleRect;
    }
    
    cachedVisibleRectScale = state.scale;
    cachedVisibleRectX = state.canvasX;
    cachedVisibleRectY = state.canvasY;
    
    const scale = state.scale || 1;
    const screenW = DRAW_CONFIG.screenW;
    const screenH = DRAW_CONFIG.screenH;
    
    let visibleX = Math.max(0, -state.canvasX / scale);
    let visibleY = Math.max(0, -state.canvasY / scale);
    let visibleW = Math.min(DRAW_CONFIG.canvasW - visibleX, screenW / scale);
    let visibleH = Math.min(DRAW_CONFIG.canvasH - visibleY, screenH / scale);
    
    const padding = 10;
    visibleX = Math.max(0, visibleX - padding);
    visibleY = Math.max(0, visibleY - padding);
    visibleW = Math.min(DRAW_CONFIG.canvasW - visibleX, visibleW + padding * 2);
    visibleH = Math.min(DRAW_CONFIG.canvasH - visibleY, visibleH + padding * 2);
    
    cachedVisibleRect = {
        x: visibleX,
        y: visibleY,
        width: visibleW,
        height: visibleH
    };
    
    return cachedVisibleRect;
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
    dom.btnDocScan.addEventListener('click', openDocScanWindow);
}

// 打开文档扫描窗口
async function openDocScanWindow() {
    try {
        if (!window.__TAURI__) {
            alert('文档扫描功能需要Tauri环境');
            return;
        }
        
        const { invoke } = window.__TAURI__.core;
        await invoke('open_doc_scan_window');
    } catch (error) {
        console.error('打开文档扫描窗口失败:', error);
        alert('打开文档扫描窗口失败: ' + error.message);
    }
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
            ${ThemeManager.getIcon('settings', { alt: window.i18n?.t('toolbar.settings') || '设置' })}
            ${window.i18n?.t('toolbar.settings') || '设置'}
        </button>
        <button class="menu-item" id="menuClose">
            ${ThemeManager.getIcon('close', { alt: window.i18n?.t('common.close') || '关闭' })}
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
    const scale = getSafeScale();
    dc.strokeStyle = DRAW_CONFIG.penColor;
    dc.lineWidth = DRAW_CONFIG.penWidth / scale;
    dc.lineCap = 'round';
    dc.lineJoin = 'round';
    dc.miterLimit = 10;
    dc.globalCompositeOperation = 'source-over';
}

function setEraserStyle() {
    const dc = dom.drawCtx;
    dc.lineWidth = DRAW_CONFIG.eraserSize;
    dc.lineCap = 'round';
    dc.lineJoin = 'round';
    dc.miterLimit = 10;
    dc.globalCompositeOperation = 'destination-out';
}

function startDrawingMode() {
    dom.canvasWrapper.classList.add('drawing');
}

function endDrawingMode() {
    dom.canvasWrapper.classList.remove('drawing');
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
    const panel = dom.penControlPanel;
    if (!panel.classList.contains('visible')) return;
    panel.classList.remove('visible');
    panel.style.opacity = '0';
    panel.style.visibility = 'hidden';
}

let eraserHintRafId = null;
let eraserHintPendingPos = null;

function updateEraserHintPos(clientX, clientY) {
    eraserHintPendingPos = { clientX, clientY };
    
    if (eraserHintRafId !== null) return;
    
    eraserHintRafId = requestAnimationFrame(() => {
        eraserHintRafId = null;
        if (!eraserHintPendingPos) return;
        
        const { clientX, clientY } = eraserHintPendingPos;
        eraserHintPendingPos = null;
        
        const rect = getCachedCanvasRect();
        const x = clientX - rect.left;
        const y = clientY - rect.top;
        dom.eraserHint.style.left = `${x}px`;
        dom.eraserHint.style.top = `${y}px`;
        dom.eraserHint.style.transform = `translate(-50%, -50%) scale(${state.scale})`;
    });
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
    state.drawCanvasRect = dom.drawCanvas.getBoundingClientRect();
    
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
        startDrawingMode();
        state.cachedInvScale = 1 / getSafeScale();
        state.lastX = (e.clientX - state.drawCanvasRect.left) * state.cachedInvScale;
        state.lastY = (e.clientY - state.drawCanvasRect.top) * state.cachedInvScale;
        startStroke('draw');
    } else if (state.drawMode === 'eraser') {
        hidePenControlPanel();
        state.isDrawing = true;
        startDrawingMode();
        state.cachedInvScale = 1 / getSafeScale();
        state.lastX = (e.clientX - state.drawCanvasRect.left) * state.cachedInvScale;
        state.lastY = (e.clientY - state.drawCanvasRect.top) * state.cachedInvScale;
        startStroke('erase');
    }
}

/**
 * Pointer 移动处理
 */
function handlePointerMove(e) {
    e.preventDefault();
    
    state.currentPressure = e.pressure || 0.5;
    
    if (state.drawMode === 'eraser') {
        updateEraserHintPos(e.clientX, e.clientY);
    }
    
    if (state.isDragging) {
        state.canvasX = e.clientX - state.startDragX;
        state.canvasY = e.clientY - state.startDragY;
        clampCanvasPosition();
        
        const transform = `translate3d(${state.canvasX}px, ${state.canvasY}px, 0) scale(${state.scale})`;
        dom.canvasWrapper.style.transform = transform;
        
        lastCanvasTransform.x = state.canvasX;
        lastCanvasTransform.y = state.canvasY;
        lastCanvasTransform.scale = state.scale;
    } else if (state.isDrawing) {
        const rect = state.drawCanvasRect;
        const invScale = state.cachedInvScale;
        const x = (e.clientX - rect.left) * invScale;
        const y = (e.clientY - rect.top) * invScale;
        
        const dx = x - state.lastX;
        const dy = y - state.lastY;
        const distSq = dx * dx + dy * dy;
        
        if (distSq > 4) {
            addStrokePoint(state.lastX, state.lastY, x, y, state.currentPressure);
            
            batchDrawManager.addCommand(
                state.cachedDrawType, 
                state.lastX, 
                state.lastY, 
                x, 
                y, 
                state.cachedDrawColor, 
                state.cachedDrawLineWidth
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
        endDrawingMode();
        if (state.drawRafId) {
            cancelAnimationFrame(state.drawRafId);
            state.drawRafId = null;
        }
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
        endDrawingMode();
        if (state.drawRafId) {
            cancelAnimationFrame(state.drawRafId);
            state.drawRafId = null;
        }
        await endStroke();
    }
}

/**
 * 鼠标按下处理
 */
function handleMouseDown(e) {
    e.preventDefault();
    state.drawCanvasRect = dom.drawCanvas.getBoundingClientRect();
    
    if (state.drawMode === 'move') {
        state.isDragging = true;
        state.startDragX = e.clientX - state.canvasX;
        state.startDragY = e.clientY - state.canvasY;
        dom.canvasWrapper.classList.add('dragging');
        dom.drawCanvas.classList.add('dragging');
    } else if (state.drawMode === 'comment') {
        hidePenControlPanel();
        state.isDrawing = true;
        startDrawingMode();
        state.cachedInvScale = 1 / getSafeScale();
        state.lastX = (e.clientX - state.drawCanvasRect.left) * state.cachedInvScale;
        state.lastY = (e.clientY - state.drawCanvasRect.top) * state.cachedInvScale;
        startStroke('draw');
    } else if (state.drawMode === 'eraser') {
        hidePenControlPanel();
        state.isDrawing = true;
        startDrawingMode();
        state.cachedInvScale = 1 / getSafeScale();
        state.lastX = (e.clientX - state.drawCanvasRect.left) * state.cachedInvScale;
        state.lastY = (e.clientY - state.drawCanvasRect.top) * state.cachedInvScale;
        startStroke('erase');
    }
}

/**
 * 鼠标移动处理
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
        
        const transform = `translate3d(${state.canvasX}px, ${state.canvasY}px, 0) scale(${state.scale})`;
        dom.canvasWrapper.style.transform = transform;
        
        lastCanvasTransform.x = state.canvasX;
        lastCanvasTransform.y = state.canvasY;
        lastCanvasTransform.scale = state.scale;
    } else if (state.isDrawing) {
        const rect = state.drawCanvasRect;
        const invScale = state.cachedInvScale;
        const x = (e.clientX - rect.left) * invScale;
        const y = (e.clientY - rect.top) * invScale;
        
        const dx = x - state.lastX;
        const dy = y - state.lastY;
        const distSq = dx * dx + dy * dy;
        
        if (distSq > 0.09) {
            addStrokePoint(state.lastX, state.lastY, x, y);
            
            batchDrawManager.addCommand(
                state.cachedDrawType, 
                state.lastX, 
                state.lastY, 
                x, 
                y, 
                state.cachedDrawColor, 
                state.cachedDrawLineWidth
            );
            
            state.lastX = x;
            state.lastY = y;
        }
    }
}

async function handleMouseUp(e) {
    if (state.isDragging) {
        state.isDragging = false;
        dom.canvasWrapper.classList.remove('dragging');
        dom.drawCanvas.classList.remove('dragging');
    }
    if (state.isDrawing) {
        state.isDrawing = false;
        endDrawingMode();
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
        endDrawingMode();
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
        const containerRect = getCachedCanvasRect();
        const mouseX = e.clientX - containerRect.left;
        const mouseY = e.clientY - containerRect.top;
        
        const oldScale = state.scale;
        
        const scaleRatio = newScale / oldScale;
        const targetX = mouseX - (mouseX - state.canvasX) * scaleRatio;
        const targetY = mouseY - (mouseY - state.canvasY) * scaleRatio;
        
        state.scale = newScale;
        state.canvasX = targetX;
        state.canvasY = targetY;
        
        updateMoveBound();
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
    state.drawCanvasRect = dom.drawCanvas.getBoundingClientRect();
    
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
            startDrawingMode();
            state.cachedInvScale = 1 / getSafeScale();
            state.lastX = (touch.clientX - state.drawCanvasRect.left) * state.cachedInvScale;
            state.lastY = (touch.clientY - state.drawCanvasRect.top) * state.cachedInvScale;
            startStroke('draw');
        } else if (state.drawMode === 'eraser') {
            state.isDrawing = true;
            startDrawingMode();
            updateEraserHintPos(touch.clientX, touch.clientY);
            state.cachedInvScale = 1 / getSafeScale();
            state.lastX = (touch.clientX - state.drawCanvasRect.left) * state.cachedInvScale;
            state.lastY = (touch.clientY - state.drawCanvasRect.top) * state.cachedInvScale;
            startStroke('erase');
        }
    } else if (touches.length === 2) {
        state.isScaling = true;
        state.isDragging = false;
        state.isDrawing = false;
        state.startDistanceSq = getTouchDistanceSq(touches[0], touches[1]);
        state.startScale = state.scale;
        state.startScaleX = (touches[0].clientX + touches[1].clientX) / 2;
        state.startScaleY = (touches[0].clientY + touches[1].clientY) / 2;
        state.startCanvasX = state.canvasX;
        state.startCanvasY = state.canvasY;
        dom.canvasWrapper.classList.add('dragging');
        
        if (state.isCameraOpen && dom.cameraVideo) {
            dom.cameraVideo.classList.add('scaling');
        }
    }
}

function handleTouchMove(e) {
    e.preventDefault();
    const touches = e.touches;
    
    if (touches.length === 1 && state.isDragging) {
        const touch = touches[0];
        state.canvasX = touch.clientX - state.startDragX;
        state.canvasY = touch.clientY - state.startDragY;
        clampCanvasPosition();
        
        const transform = `translate3d(${state.canvasX}px, ${state.canvasY}px, 0) scale(${state.scale})`;
        dom.canvasWrapper.style.transform = transform;
        
        lastCanvasTransform.x = state.canvasX;
        lastCanvasTransform.y = state.canvasY;
        lastCanvasTransform.scale = state.scale;
    } else if (touches.length === 1 && state.isDrawing) {
        const touch = touches[0];
        if (state.drawMode === 'eraser') {
            updateEraserHintPos(touch.clientX, touch.clientY);
        }
        
        const invScale = state.cachedInvScale;
        const x = (touch.clientX - state.drawCanvasRect.left) * invScale;
        const y = (touch.clientY - state.drawCanvasRect.top) * invScale;
        
        const pressure = (touch.force > 0) ? touch.force : 0.5;
        
        const dx = x - state.lastX;
        const dy = y - state.lastY;
        const distSq = dx * dx + dy * dy;
        
        if (distSq > 0.09) {
            addStrokePoint(state.lastX, state.lastY, x, y, pressure);
            
            batchDrawManager.addCommand(
                state.cachedDrawType, 
                state.lastX, 
                state.lastY, 
                x, 
                y, 
                state.cachedDrawColor, 
                state.cachedDrawLineWidth
            );
            
            state.lastX = x;
            state.lastY = y;
        }
    } else if (touches.length === 2 && state.isScaling) {
        const currentDistanceSq = getTouchDistanceSq(touches[0], touches[1]);
        const scaleRatio = Math.sqrt(currentDistanceSq / state.startDistanceSq);
        let newScale = state.startScale * scaleRatio;
        const maxScale = state.isCameraOpen ? DRAW_CONFIG.maxScaleCamera : DRAW_CONFIG.maxScaleImage;
        newScale = newScale < DRAW_CONFIG.minScale ? DRAW_CONFIG.minScale : (newScale > maxScale ? maxScale : newScale);
        
        const centerX = (touches[0].clientX + touches[1].clientX) / 2;
        const centerY = (touches[0].clientY + touches[1].clientY) / 2;
        
        const finalRatio = newScale / state.startScale;
        state.canvasX = centerX - (state.startScaleX - state.startCanvasX) * finalRatio;
        state.canvasY = centerY - (state.startScaleY - state.startCanvasY) * finalRatio;
        state.scale = newScale;
        
        scheduleTransformUpdate(state.canvasX, state.canvasY, state.scale);
    }
}

async function handleTouchEnd(e) {
    e.preventDefault();
    
    if (e.touches.length === 0) {
        state.isDragging = false;
        state.isScaling = false;
        dom.canvasWrapper.classList.remove('dragging');
        dom.drawCanvas.classList.remove('dragging');
        
        if (state.isCameraOpen && dom.cameraVideo) {
            dom.cameraVideo.classList.remove('scaling');
        }
        
        updateMoveBound();
        clampCanvasPosition();
        dom.canvasWrapper.style.transform = `translate3d(${state.canvasX}px, ${state.canvasY}px, 0) scale(${state.scale})`;
        
        if (state.isDrawing) {
            state.isDrawing = false;
            endDrawingMode();
            if (state.drawRafId) {
                cancelAnimationFrame(state.drawRafId);
                state.drawRafId = null;
            }
            await endStroke();
        }
    } else if (e.touches.length === 1) {
        state.isScaling = false;
        if (state.isCameraOpen && dom.cameraVideo) {
            dom.cameraVideo.classList.remove('scaling');
        }
        
        updateMoveBound();
        clampCanvasPosition();
        dom.canvasWrapper.style.transform = `translate3d(${state.canvasX}px, ${state.canvasY}px, 0) scale(${state.scale})`;
        
        const touch = e.touches[0];
        if (state.drawMode === 'move') {
            state.isDragging = true;
            state.startDragX = touch.clientX - state.canvasX;
            state.startDragY = touch.clientY - state.canvasY;
        }
    }
}

function getTouchDistanceSq(touch1, touch2) {
    const dx = touch2.clientX - touch1.clientX;
    const dy = touch2.clientY - touch1.clientY;
    return dx * dx + dy * dy;
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
    
    const transform = `translate3d(${state.canvasX}px, ${state.canvasY}px, 0) scale(${state.scale})`;
    dom.canvasWrapper.style.transform = transform;
}

function animateCanvasTransform(targetX, targetY, targetScale, duration = 250) {
    if (currentAnimationId !== null) {
        clearTimeout(currentAnimationId);
        currentAnimationId = null;
    }
    
    state.canvasX = targetX;
    state.canvasY = targetY;
    state.scale = targetScale;
    
    updateMoveBound();
    clampCanvasPosition();
    
    lastCanvasTransform.x = state.canvasX;
    lastCanvasTransform.y = state.canvasY;
    lastCanvasTransform.scale = state.scale;
    
    dom.canvasWrapper.classList.add('smooth-transform');
    dom.canvasWrapper.style.transform = `translate3d(${state.canvasX}px, ${state.canvasY}px, 0) scale(${state.scale})`;
    
    currentAnimationId = setTimeout(() => {
        currentAnimationId = null;
        dom.canvasWrapper.classList.remove('smooth-transform');
    }, duration);
}

// 撤销功能 - 混合方案：路径记录 + ImageData 压缩
function startStroke(type) {
    state.currentStroke = {
        type: type,
        points: [],
        color: DRAW_CONFIG.penColor,
        lineWidth: DRAW_CONFIG.penWidth,
        eraserSize: DRAW_CONFIG.eraserSize,
        bounds: {
            minX: Infinity,
            minY: Infinity,
            maxX: -Infinity,
            maxY: -Infinity
        },
        variableWidths: null
    };
    
    state.currentPressure = 0.5;
    state.currentLineWidth = DRAW_CONFIG.penWidth;
    state.lastLineWidth = DRAW_CONFIG.penWidth;
    
    state.cachedDrawType = type;
    state.cachedDrawColor = type === 'draw' ? DRAW_CONFIG.penColor : '#000000';
    state.cachedDrawLineWidth = type === 'draw' ? DRAW_CONFIG.penWidth : DRAW_CONFIG.eraserSize;
    
    batchDrawManager.startDrawing();
}

function addStrokePoint(fromX, fromY, toX, toY, pressure = 0.5) {
    const stroke = state.currentStroke;
    if (!stroke) return;
    
    const bounds = stroke.bounds;
    if (fromX < bounds.minX) bounds.minX = fromX;
    if (toX < bounds.minX) bounds.minX = toX;
    if (fromY < bounds.minY) bounds.minY = fromY;
    if (toY < bounds.minY) bounds.minY = toY;
    if (fromX > bounds.maxX) bounds.maxX = fromX;
    if (toX > bounds.maxX) bounds.maxX = toX;
    if (fromY > bounds.maxY) bounds.maxY = fromY;
    if (toY > bounds.maxY) bounds.maxY = toY;
    
    if (stroke.type === 'draw') {
        state.currentPressure = pressure;
        state.lastLineWidth = state.currentLineWidth;
        state.currentLineWidth = stroke.lineWidth * (0.9 + pressure * 0.2);
    }
    
    const points = stroke.points;
    const len = points.length;
    if (len > 0) {
        const lastPoint = points[len - 1];
        const dx = fromX - lastPoint.toX;
        const dy = fromY - lastPoint.toY;
        if (dx * dx + dy * dy > 2.25) {
            points.push({
                fromX: lastPoint.toX,
                fromY: lastPoint.toY,
                toX: fromX,
                toY: fromY
            });
        }
    }
    
    points.push({ fromX, fromY, toX, toY });
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
    const ctx = dom.drawCtx;
    const scale = getSafeScale();
    
    const visibleRect = getVisibleRect();
    
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
    
    ctx.globalCompositeOperation = 'destination-out';
    ctx.strokeStyle = 'rgba(0, 0, 0, 1)';
    
    const erasePath = new Path2D();
    for (const stroke of eraseStrokes) {
        const eraserSize = stroke.eraserSize || DRAW_CONFIG.eraserSize;
        ctx.lineWidth = eraserSize;
        
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
    
    // 绘制可变线宽笔画 - 优化：合并路径减少 GPU 调用
    for (const stroke of variableWidthStrokes) {
        if (!stroke.points || stroke.points.length === 0) continue;
        
        ctx.fillStyle = stroke.color || DRAW_CONFIG.penColor;
        
        const polygonPath = new Path2D();
        const circlePath = new Path2D();
        
        for (let i = 0; i < stroke.points.length && i < stroke.variableWidths.length; i++) {
            const point = stroke.points[i];
            const widthInfo = stroke.variableWidths[i];
            
            const x1 = point.fromX, y1 = point.fromY;
            const x2 = point.toX, y2 = point.toY;
            const w1 = widthInfo.fromWidth / scale, w2 = widthInfo.toWidth / scale;
            
            const angle = Math.atan2(y2 - y1, x2 - x1);
            const perpAngle = angle + Math.PI / 2;
            const hw1 = w1 / 2, hw2 = w2 / 2;
            const cos = Math.cos(perpAngle), sin = Math.sin(perpAngle);
            
            polygonPath.moveTo(x1 + cos * hw1, y1 + sin * hw1);
            polygonPath.lineTo(x2 + cos * hw2, y2 + sin * hw2);
            polygonPath.lineTo(x2 - cos * hw2, y2 - sin * hw2);
            polygonPath.lineTo(x1 - cos * hw1, y1 - sin * hw1);
            polygonPath.closePath();
            
            circlePath.moveTo(x1 + hw1, y1);
            circlePath.arc(x1, y1, hw1, 0, Math.PI * 2);
            circlePath.moveTo(x2 + hw2, y2);
            circlePath.arc(x2, y2, hw2, 0, Math.PI * 2);
        }
        
        ctx.fill(polygonPath);
        ctx.fill(circlePath);
    }
    
    // 批量绘制固定线宽笔画
    for (const [stateKey, group] of fixedWidthStrokes) {
        ctx.strokeStyle = group.color;
        ctx.lineWidth = group.lineWidth / scale;
        
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
    
    const scale = getSafeScale();
    setContextState(dom.drawCtx, {
        strokeStyle: stroke.color || DRAW_CONFIG.penColor,
        lineWidth: (stroke.lineWidth || DRAW_CONFIG.penWidth) / scale,
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

// ==================== 点处理工具函数 ====================

/**
 * 坐标量化
 */
function quantizeCoord(coord) {
    return Math.round(coord / POINT_OPTIMIZATION.quantization) * POINT_OPTIMIZATION.quantization;
}

/**
 * 计算两点之间的距离
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
    
    const scale = getSafeScale();
    
    // 按状态分组，使用 Path2D 批量绘制
    const eraseStrokes = new Map();
    const drawStrokesMap = new Map();
    
    for (const stroke of strokes) {
        if (stroke.type === 'erase') {
            const sizeKey = stroke.eraserSize || DRAW_CONFIG.eraserSize;
            if (!eraseStrokes.has(sizeKey)) {
                eraseStrokes.set(sizeKey, []);
            }
            eraseStrokes.get(sizeKey).push(stroke);
        } else if (stroke.type === 'draw' || stroke.type === 'comment') {
            const stateKey = `${stroke.color || DRAW_CONFIG.penColor}-${stroke.lineWidth || DRAW_CONFIG.penWidth}`;
            if (!drawStrokesMap.has(stateKey)) {
                drawStrokesMap.set(stateKey, {
                    color: stroke.color || DRAW_CONFIG.penColor,
                    lineWidth: stroke.lineWidth || DRAW_CONFIG.penWidth,
                    strokes: []
                });
            }
            drawStrokesMap.get(stateKey).strokes.push(stroke);
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
    for (const [stateKey, group] of drawStrokesMap) {
        ctx.strokeStyle = group.color;
        ctx.lineWidth = group.lineWidth / scale;
        
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
    
    const hasClearStroke = state.strokeHistory.some(stroke => stroke.type === 'clear');
    if (hasClearStroke) {
        console.log('检测到清空操作，跳过压缩以保留撤销能力');
        return;
    }
    
    const hasEraserStroke = state.strokeHistory.some(stroke => stroke.type === 'erase');
    if (hasEraserStroke) {
        console.log('检测到橡皮擦操作，跳过压缩以保留撤销能力');
        return;
    }
    
    compactIdleId = requestIdleCallback((deadline) => {
        compactIdleId = null;
        doCompactStrokes();
    }, { timeout: 2000 });
}

/**
 * 执行笔画压缩
 * 优先使用Rust并行处理，失败时降级到前端Canvas
 */
async function doCompactStrokes() {
    if (state.strokeHistory.length <= state.MAX_UNDO_STEPS) return;
    
    const hasClearStroke = state.strokeHistory.some(stroke => stroke.type === 'clear');
    if (hasClearStroke) {
        console.log('压缩执行前检测到清空操作，取消压缩');
        return;
    }
    
    const hasEraserStroke = state.strokeHistory.some(stroke => stroke.type === 'erase');
    if (hasEraserStroke) {
        console.log('压缩执行前检测到橡皮擦操作，取消压缩');
        return;
    }
    
    const strokesToCompact = state.strokeHistory.slice(0, state.strokeHistory.length - state.MAX_UNDO_STEPS);
    
    if (strokesToCompact.length === 0) return;
    
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
            
            state.strokeHistory = state.strokeHistory.slice(state.strokeHistory.length - state.MAX_UNDO_STEPS);
            
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
    
    state.strokeHistory = state.strokeHistory.slice(state.strokeHistory.length - state.MAX_UNDO_STEPS);
    
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
    if (compactIdleId !== null) {
        cancelIdleCallback(compactIdleId);
        compactIdleId = null;
        console.log('撤销操作：取消正在进行的压缩任务');
    }
    
    state.baseImageLoadId++;
    
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
    const scale = getSafeScale();
    setContextState(dom.drawCtx, {
        strokeStyle: DRAW_CONFIG.penColor,
        lineWidth: DRAW_CONFIG.penWidth / scale,
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
        state.imageList[state.currentImageIndex].strokeHistory = [];
        state.imageList[state.currentImageIndex].baseImageURL = null;
    }
    
    if (state.currentFolderIndex >= 0 && state.currentFolderPageIndex >= 0) {
        if (state.currentFolderIndex < state.fileList.length) {
            const folder = state.fileList[state.currentFolderIndex];
            if (state.currentFolderPageIndex < folder.pages.length) {
                folder.pages[state.currentFolderPageIndex].strokeHistory = [];
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
                
                await openCamera();
                updateSidebarSelection();
                updatePhotoButtonState();
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
                
                await openCamera();
                updateFolderPageSelection(-1, -1);
                updatePhotoButtonState();
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
    
    // 绘制图片层
    if (dom.imageElement.src) {
        mergedCtx.drawImage(dom.imageElement, 
            parseFloat(dom.imageElement.style.left) || 0, 
            parseFloat(dom.imageElement.style.top) || 0, 
            parseFloat(dom.imageElement.style.width) || DRAW_CONFIG.canvasW, 
            parseFloat(dom.imageElement.style.height) || DRAW_CONFIG.canvasH
        );
    }
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
    let html, title;
    
    const photoText = window.i18n?.t('toolbar.photo') || '拍照';
    const switchToCameraText = window.i18n?.t('camera.switchToCamera') || '切换到摄像头';
    const showText = ThemeManager.getShowToolbarText();
    
    if (state.isCameraOpen) {
        newState = 'camera';
        html = `${ThemeManager.getIcon('camera', { alt: photoText })}${showText ? photoText : ''}`;
        title = window.i18n?.t('camera.captureFrame') || '捕获摄像头画面';
    } else if ((state.currentImageIndex >= 0 && state.imageList.length > 0) || 
               (state.currentFolderIndex >= 0 && state.currentFolderPageIndex >= 0)) {
        newState = 'switch';
        html = `${ThemeManager.getIcon('camera-fill', { alt: switchToCameraText })}${showText ? switchToCameraText : ''}`;
        title = window.i18n?.t('camera.switchToCamera') || '返回摄像头';
    } else {
        newState = 'save';
        html = `${ThemeManager.getIcon('camera', { alt: photoText })}${showText ? photoText : ''}`;
        title = window.i18n?.t('camera.saveScreenshot') || '保存画布截图';
    }
    
    if (lastPhotoButtonState === newState) return;
    lastPhotoButtonState = newState;
    
    btnPhoto.innerHTML = html;
    btnPhoto.title = title;
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
            state.imageList[state.currentImageIndex].full = rotatedImg.src;
            state.imageList[state.currentImageIndex].thumbnail = rotatedImg.src;
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
            ${ThemeManager.getIcon('image', { alt: importImageText })}
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
    
    const showText = ThemeManager.getShowToolbarText();
    dom.btnExpand.innerHTML = `
        ${ThemeManager.getIcon('collapse', { alt: collapseText })}
        ${showText ? collapseText : ''}
    `;
    console.log('展开侧边栏');
}

async function selectImage(index) {
    if (index < 0 || index >= state.imageList.length) return;
    
    if (index === state.currentImageIndex && state.currentImage) {
        (async () => {
            try {
                saveCurrentDrawData();
                saveCurrentFolderPageDrawData();
                state.currentImageIndex = -1;
                state.currentImage = null;
                clearImageLayer();
                clearDrawCanvas();
                
                if (state.isCameraOpen) {
                    await setCameraState(false);
                }
                
                await setCameraState(true);
                updateSidebarSelection();
                updatePhotoButtonState();
                console.log('返回摄像头');
            } catch (error) {
                console.error('返回摄像头失败:', error);
            }
        })();
        return;
    }
    
    // 先保存当前数据（使用旧索引）
    saveCurrentDrawData();
    saveCurrentFolderPageDrawData();
    
    // 更新索引
    state.currentImageIndex = index;
    state.currentFolderIndex = -1;
    state.currentFolderPageIndex = -1;
    
    // 使用源ID管理系统切换
    const imgData = state.imageList[index];
    if (imgData.sourceId) {
        await switchToSource(imgData.sourceId);
    } else {
        // 兼容旧数据：为没有 sourceId 的图片分配新ID
        imgData.sourceId = generateSourceId('pic');
        
        // 初始化数据（如果不存在）
        if (!imgData.strokeHistory) {
            imgData.strokeHistory = [];
        }
        if (!imgData.viewState) {
            imgData.viewState = {
                scale: 1,
                canvasX: -(DRAW_CONFIG.canvasW - DRAW_CONFIG.screenW) / 2,
                canvasY: -(DRAW_CONFIG.canvasH - DRAW_CONFIG.screenH) / 2
            };
        }
        
        // 保存当前摄像头数据
        saveCurrentSourceData();
        
        // 切换到新源
        currentSourceId = imgData.sourceId;
        sourceDataStore[imgData.sourceId] = {
            scale: imgData.viewState.scale,
            canvasX: imgData.viewState.canvasX,
            canvasY: imgData.viewState.canvasY,
            strokeHistory: cloneStrokes(imgData.strokeHistory),
            baseImageURL: imgData.baseImageURL
        };
        
        // 加载数据
        state.scale = imgData.viewState.scale;
        state.canvasX = imgData.viewState.canvasX;
        state.canvasY = imgData.viewState.canvasY;
        state.strokeHistory = cloneStrokes(imgData.strokeHistory);
        state.baseImageURL = imgData.baseImageURL;
        state.baseImageObj = null;
        
        updateMoveBound();
        updateCanvasTransform();
        clearDrawCanvas();
        
        if (state.strokeHistory.length > 0) {
            await redrawAllStrokes();
        }
        updateUndoBtnStatus();
    }
    
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
            await new Promise((resolve, reject) => {
                const img = new Image();
                img.onload = async () => {
                    state.baseImageObj = img;
                    await redrawAllStrokes();
                    updateUndoBtnStatus();
                    resolve();
                };
                img.onerror = () => {
                    console.error('加载基础图片失败');
                    reject(new Error('Failed to load base image'));
                };
                img.src = state.baseImageURL;
            });
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
    const showText = ThemeManager.getShowToolbarText();
    dom.btnExpand.innerHTML = `
        ${ThemeManager.getIcon('image', { alt: imageText })}
        ${showText ? imageText : ''}
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
            const iconName = isWord ? 'word' : 'pdf';
            const fileAlt = window.i18n?.t('toolbar.file') || '文件';
            const pagesText = window.i18n?.t('sidebar.pages', { n: folder.pages.length }) || `${folder.pages.length}页`;
            contentHTML += `
                <div class="sidebar-folder-item" data-index="${index}">
                    ${ThemeManager.getIcon(iconName, { alt: fileAlt })}
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
            ${ThemeManager.getIcon('addFile', { alt: addFileText })}
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
    
    const showText = ThemeManager.getShowToolbarText();
    dom.btnSave.innerHTML = `
        ${ThemeManager.getIcon('collapse', { alt: collapseText })}
        ${showText ? collapseText : ''}
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
            if (state.isCameraOpen) {
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
                await new Promise((resolve, reject) => {
                    const img = new Image();
                    img.onload = async () => {
                        state.baseImageObj = img;
                        await redrawAllStrokes();
                        updateUndoBtnStatus();
                        resolve();
                    };
                    img.onerror = () => {
                        console.error('加载基础图片失败');
                        reject(new Error('Failed to load base image'));
                    };
                    img.src = state.baseImageURL;
                });
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
            const iconName = isWord ? 'word' : 'pdf';
            console.log(`文件夹 ${folder.name}: isWord=${isWord}, iconName=${iconName}`);
            const fileAlt = window.i18n?.t('toolbar.file') || '文件';
            const pagesText = window.i18n?.t('sidebar.pages', { n: folder.pages.length }) || `${folder.pages.length}页`;
            contentHTML += `
                <div class="sidebar-folder-item" data-index="${index}">
                    ${ThemeManager.getIcon(iconName, { alt: fileAlt })}
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
    const showText = ThemeManager.getShowToolbarText();
    dom.btnSave.innerHTML = `
        ${ThemeManager.getIcon('file', { alt: fileText })}
        ${showText ? fileText : ''}
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
 * 显示无摄像头提示信息
 */
function showNoCameraMessage(message) {
    if (!dom.canvasContainer) return;
    
    let msgElement = document.getElementById('noCameraMessage');
    if (!msgElement) {
        msgElement = document.createElement('div');
        msgElement.id = 'noCameraMessage';
        msgElement.style.cssText = `
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background: rgba(30, 30, 30, 0.9);
            z-index: 10;
            pointer-events: none;
        `;
        dom.canvasContainer.appendChild(msgElement);
    }
    
    msgElement.innerHTML = `
        <div style="font-size: 2.5vw; color: #fff; margin-bottom: 2vh;">( $ _ $ )</div>
        <div style="font-size: 1.2vw; color: rgba(255,255,255,0.8); margin-bottom: 1vh;">${window.i18n?.t('camera.deviceNotFound') || '找不到展台设备'}</div>
        <div style="font-size: 0.9vw; color: rgba(255,255,255,0.5);">${message}</div>
    `;
    msgElement.style.display = 'flex';
}

/**
 * 隐藏无摄像头提示信息
 */
function hideNoCameraMessage() {
    const msgElement = document.getElementById('noCameraMessage');
    if (msgElement) {
        msgElement.style.display = 'none';
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
    
    const videoRatio = videoW / videoH;
    
    const screenW = DRAW_CONFIG.screenW;
    const screenH = DRAW_CONFIG.screenH;
    const screenRatio = screenW / screenH;
    
    let drawW, drawH;
    if (videoRatio > screenRatio) {
        drawW = screenW;
        drawH = screenW / videoRatio;
    } else {
        drawH = screenH;
        drawW = screenH * videoRatio;
    }
    
    const canvasW = DRAW_CONFIG.canvasW;
    const canvasH = DRAW_CONFIG.canvasH;
    const offsetX = (canvasW - drawW) / 2;
    const offsetY = (canvasH - drawH) / 2;
    
    const styleChanged = 
        lastVideoStyleCache.drawW !== drawW ||
        lastVideoStyleCache.drawH !== drawH ||
        lastVideoStyleCache.offsetX !== offsetX ||
        lastVideoStyleCache.offsetY !== offsetY ||
        lastVideoStyleCache.rotation !== rotation ||
        lastVideoStyleCache.isMirrored !== state.isMirrored;
    
    if (!styleChanged) return;
    
    lastVideoStyleCache = { drawW, drawH, offsetX, offsetY, rotation, isMirrored: state.isMirrored };
    
    let transforms = [];
    
    if (rotation !== 0) {
        transforms.push(`rotate(${rotation}deg)`);
    }
    
    if (state.isMirrored) {
        transforms.push('scaleX(-1)');
    }
    
    const transformStr = transforms.join(' ');
    
    video.style.cssText = `
        width: ${drawW}px;
        height: ${drawH}px;
        left: ${offsetX}px;
        top: ${offsetY}px;
        transform: ${transformStr};
        transform-origin: center center;
        display: block;
    `;
}

function startCameraPreview() {
    const video = dom.cameraVideo;
    if (!video) return;
    
    updateCameraVideoStyle();
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
    
    // 保存摄像头批注数据
    saveCurrentSourceData();
    
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
            
            const result = await invoke('save_image', { 
                imageData: dataUrl,
                prefix: 'photo'
            });
            console.log('图片已保存:', result.path);
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
        URL.revokeObjectURL(blobUrl);
        console.log('已捕获摄像头画面并保存到图片列表');
    };
    img.onerror = () => {
        URL.revokeObjectURL(blobUrl);
        console.error('加载拍摄的图片失败');
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
            
            const newImgData = {
                full: imgData.data,
                thumbnail: imgData.data,
                name: imgData.name,
                width: img.width,
                height: img.height,
                strokeHistory: [],
                baseImageURL: null,
                viewState: {
                    scale: 1,
                    canvasX: -(DRAW_CONFIG.canvasW - DRAW_CONFIG.screenW) / 2,
                    canvasY: -(DRAW_CONFIG.canvasH - DRAW_CONFIG.screenH) / 2
                },
                sourceId: generateSourceId('pic')
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
    
    const imgData = {
        full: blobUrl,
        thumbnail: blobUrl,
        name: name,
        width: img.width,
        height: img.height,
        strokeHistory: [],
        baseImageURL: null,
        viewState: {
            scale: 1,
            canvasX: -(DRAW_CONFIG.canvasW - DRAW_CONFIG.screenW) / 2,
            canvasY: -(DRAW_CONFIG.canvasH - DRAW_CONFIG.screenH) / 2
        },
        sourceId: generateSourceId('pic')
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
    }
}

async function addImageToListNoHighlight(img, name) {
    const blob = await fetch(img.src).then(r => r.blob());
    const blobUrl = URL.createObjectURL(blob);
    
    const imgData = {
        full: blobUrl,
        thumbnail: blobUrl,
        name: name,
        width: img.width,
        height: img.height,
        strokeHistory: [],
        baseImageURL: null,
        viewState: {
            scale: 1,
            canvasX: -(DRAW_CONFIG.canvasW - DRAW_CONFIG.screenW) / 2,
            canvasY: -(DRAW_CONFIG.canvasH - DRAW_CONFIG.screenH) / 2
        },
        sourceId: generateSourceId('pic')
    };
    
    state.imageList.push(imgData);
    
    updateSidebarContent();
}

// 暴露必要的函数到 window 对象，供 doc-scan.js 使用
window.addImageToListNoHighlight = addImageToListNoHighlight;
window.updateSidebarContent = updateSidebarContent;
window.clearAllDrawings = clearAllDrawings;

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
    
    dom.imageElement.src = img.src;
    dom.imageElement.style.left = drawX + 'px';
    dom.imageElement.style.top = drawY + 'px';
    dom.imageElement.style.width = drawW + 'px';
    dom.imageElement.style.height = drawH + 'px';
}

function clearImageLayer() {
    dom.imageElement.src = '';
    dom.imageElement.style.left = '0';
    dom.imageElement.style.top = '0';
    dom.imageElement.style.width = DRAW_CONFIG.canvasW + 'px';
    dom.imageElement.style.height = DRAW_CONFIG.canvasH + 'px';
}
