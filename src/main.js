/**
 * ViewStage - 摄像头及PDF展台应用
 * 
 * 架构说明：
 * - 三层Canvas：背景层(bgCanvas) → 图像层(imageCanvas) → 批注层(drawCanvas)
 * - 批注系统：笔画记录 + 压缩存储 + 撤销支持
 * - 图像处理：Rust后端并行处理（增强、缩略图、旋转）
 */

// ==================== PDF.js 配置 ====================

function initPdfJs() {
    if (window.pdfjsLib) {
        pdfjsLib.GlobalWorkerOptions.workerSrc = 'assets/pdf.worker.min.js';
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

const DRAW_CONFIG = {
    penColor: '#3498db',           // 默认笔色
    penWidth: 2,                   // 默认笔宽 (px)
    eraserSize: 15,                // 橡皮大小 (px)
    minScale: 0.5,                 // 最小缩放比例
    maxScale: 5,                   // 最大缩放比例
    canvasW: 1000,                 // 画布宽度 (逻辑像素)
    canvasH: 600,                  // 画布高度 (逻辑像素)
    screenW: 0,                    // 屏幕宽度
    screenH: 0,                    // 屏幕高度
    canvasScale: 2,                // 画布相对屏幕的缩放倍数
    dpr: Math.min(window.devicePixelRatio || 1, 2),  // 设备像素比 (限制最大为2，减少GPU负担)
    cameraFrameInterval: 33,       // 摄像头帧间隔 (ms) - 30fps
    cameraFrameIntervalLow: 100    // 低帧率模式 (绘制时)
};

// ==================== 智能绘制调度器 ====================

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
                dc.imageSmoothingQuality = 'low';
                break;
            case 'medium':
                dc.imageSmoothingQuality = 'medium';
                break;
            case 'high':
                dc.imageSmoothingQuality = 'medium'; // 保持medium以避免GPU负载过高
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

// 全局智能绘制调度器
const smartDrawScheduler = new SmartDrawScheduler();

// ==================== 全局状态 ====================

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
    isMirrored: false,             // 是否镜像 (前置摄像头)
    cameraAnimationId: null,       // requestAnimationFrame ID
    cameraRotation: 0,             // 摄像头旋转角度
    useFrontCamera: false,         // 是否使用前置摄像头
    
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
    
    // 绘制优化
    pendingDrawPoints: [],         // 待绘制点队列 (RAF批量处理)
    drawRafId: null                // requestAnimationFrame ID
};

let dom = {};  // DOM 元素引用缓存

// ==================== 初始化 ====================

window.addEventListener('DOMContentLoaded', async () => {
    initDOM();
    initCanvas();
    bindAllEvents();
    saveSnapshot();
    
    window.addEventListener('resize', handleResize);
    
    await initCacheDir();
    
    openCamera();
    
    if (window.__TAURI__) {
        listenForPdfFileOpen();
    }
    
    console.log('画布初始化完成');
});

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
            alert('无法解析文件路径');
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
}

async function processPdfPagesParallel(pdf, totalPages, batchSize = 4) {
    const pages = [];
    let processedCount = 0;
    
    async function processPage(pageNum) {
        const page = await pdf.getPage(pageNum);
        const viewport = page.getViewport({ scale: 1.5 });
        
        const canvas = document.createElement('canvas');
        canvas.width = viewport.width;
        canvas.height = viewport.height;
        const ctx = canvas.getContext('2d');
        
        await page.render({
            canvasContext: ctx,
            viewport: viewport
        }).promise;
        
        const fullImage = canvas.toDataURL('image/jpeg', 0.85);
        const thumbnail = await generateThumbnail(fullImage, 150, false);
        
        processedCount++;
        updateLoadingProgress(`正在处理 ${processedCount}/${totalPages} 页`);
        
        return {
            full: fullImage,
            thumbnail: thumbnail,
            pageNum: pageNum,
            strokeHistory: null,
            baseImageURL: null
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
    if (state.isCameraOpen) {
        closeCamera();
    }
    
    showLoadingOverlay('正在导入文件...');
    
    try {
        const pdfReady = await waitForPdfJs();
        if (!pdfReady) {
            hideLoadingOverlay();
            console.error('PDF.js 库加载超时');
            alert('PDF库加载超时，请重启应用后重试');
            return;
        }
        
        console.log('开始加载PDF:', filePath);
        
        const { fs } = window.__TAURI__;
        
        let fileData;
        try {
            fileData = await fs.readFile(filePath);
            console.log('文件读取成功，数据类型:', typeof fileData, '是否数组:', Array.isArray(fileData));
        } catch (readError) {
            console.error('文件读取失败:', readError);
            hideLoadingOverlay();
            alert('无法读取文件: ' + readError.message);
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
        const fileName = filePath.split(/[/\\]/).pop().replace('.pdf', '');
        
        const folder = {
            name: fileName,
            pages: []
        };
        
        const processedPages = await processPdfPagesParallel(pdf, totalPages);
        folder.pages = processedPages;
        
        state.fileList.push(folder);
        updateFileSidebarContent();
        expandFileSidebar();
        
        if (folder.pages.length > 0) {
            const firstPage = folder.pages[0];
            const img = new Image();
            img.onload = () => {
                state.currentImage = img;
                state.currentFolderIndex = state.fileList.length - 1;
                state.currentFolderPageIndex = 0;
                drawImageToCenter(img);
                updatePhotoButtonState();
                updateEnhanceButtonState();
            };
            img.src = firstPage.full;
        }
        
        hideLoadingOverlay();
        console.log(`PDF已导入: ${folder.name}，共${folder.pages.length}页`);
    } catch (error) {
        hideLoadingOverlay();
        console.error('PDF导入失败:', error);
        alert('PDF导入失败，请确保文件格式正确');
    }
}

// 处理窗口大小变化
function handleResize() {
    const container = dom.canvasContainer;
    const newScreenW = container.clientWidth;
    const newScreenH = container.clientHeight;
    
    if (newScreenW !== DRAW_CONFIG.screenW || newScreenH !== DRAW_CONFIG.screenH) {
        resizeCanvas(newScreenW, newScreenH);
    }
}

// 调整画布大小
function resizeCanvas(newScreenW, newScreenH) {
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
    
    updateMoveBound();
    
    // 调整画布大小
    dom.bgCanvas.width = DRAW_CONFIG.canvasW * DRAW_CONFIG.dpr;
    dom.bgCanvas.height = DRAW_CONFIG.canvasH * DRAW_CONFIG.dpr;
    dom.imageCanvas.width = DRAW_CONFIG.canvasW * DRAW_CONFIG.dpr;
    dom.imageCanvas.height = DRAW_CONFIG.canvasH * DRAW_CONFIG.dpr;
    dom.drawCanvas.width = DRAW_CONFIG.canvasW * DRAW_CONFIG.dpr;
    dom.drawCanvas.height = DRAW_CONFIG.canvasH * DRAW_CONFIG.dpr;
    
    dom.bgCanvas.style.width = DRAW_CONFIG.canvasW + 'px';
    dom.bgCanvas.style.height = DRAW_CONFIG.canvasH + 'px';
    dom.imageCanvas.style.width = DRAW_CONFIG.canvasW + 'px';
    dom.imageCanvas.style.height = DRAW_CONFIG.canvasH + 'px';
    dom.drawCanvas.style.width = DRAW_CONFIG.canvasW + 'px';
    dom.drawCanvas.style.height = DRAW_CONFIG.canvasH + 'px';
    
    // 重置上下文
    dom.bgCtx.setTransform(1, 0, 0, 1, 0, 0);
    dom.imageCtx.setTransform(1, 0, 0, 1, 0, 0);
    dom.drawCtx.setTransform(1, 0, 0, 1, 0, 0);
    dom.bgCtx.scale(DRAW_CONFIG.dpr, DRAW_CONFIG.dpr);
    dom.imageCtx.scale(DRAW_CONFIG.dpr, DRAW_CONFIG.dpr);
    dom.drawCtx.scale(DRAW_CONFIG.dpr, DRAW_CONFIG.dpr);
    
    // 设置绘制质量
    dom.bgCtx.imageSmoothingEnabled = true;
    dom.bgCtx.imageSmoothingQuality = 'medium';
    dom.imageCtx.imageSmoothingEnabled = true;
    dom.imageCtx.imageSmoothingQuality = 'medium';
    dom.drawCtx.imageSmoothingEnabled = true;
    dom.drawCtx.imageSmoothingQuality = 'medium';
    dom.drawCtx.lineCap = 'round';
    dom.drawCtx.lineJoin = 'round';
    dom.drawCtx.miterLimit = 10;
    
    // 重置背景
    resetBgCanvas();
    
    // 重新绘制内容（避免使用昂贵的getImageData/putImageData）
    if (state.currentImage) {
        drawImageToCenter(state.currentImage);
    } else if (state.isCameraOpen) {
        // 摄像头画面会通过renderFrame重新绘制
    }
    
    // 重新绘制批注
    if (state.strokeHistory.length > 0 || state.baseImageObj) {
        redrawAllStrokes();
    }
    
    // 恢复画布位置和缩放
    state.scale = oldScale;
    state.canvasX = oldCanvasX;
    state.canvasY = oldCanvasY;
    
    clampCanvasPosition();
    updateCanvasTransform();
    
    console.log(`窗口调整: 屏幕 ${newScreenW}x${newScreenH}, 画布 ${DRAW_CONFIG.canvasW}x${DRAW_CONFIG.canvasH}`);
}

// 初始化 DOM 元素引用
function initDOM() {
    dom.canvasContainer = document.getElementById('canvasContainer');
    dom.bgCanvas = document.getElementById('bgCanvas');
    dom.imageCanvas = document.getElementById('imageCanvas');
    dom.drawCanvas = document.getElementById('drawCanvas');
    dom.eraserHint = document.getElementById('eraserHint');
    dom.penControlPanel = document.getElementById('penControlPanel');
    dom.settingsPanel = document.getElementById('settingsPanel');
    
    dom.penSizeSlider = document.getElementById('penSizeSlider');
    dom.penSizeValue = document.getElementById('penSizeValue');
    dom.penColorPicker = document.getElementById('penColorPicker');
    dom.eraserSizeSlider = document.getElementById('eraserSizeSlider');
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
    
    dom.bgCtx = dom.bgCanvas.getContext('2d');
    dom.imageCtx = dom.imageCanvas.getContext('2d', { desynchronized: true });
    dom.drawCtx = dom.drawCanvas.getContext('2d', { willReadFrequently: true });
}

// ==================== 画布初始化 ====================

/**
 * 初始化三层画布
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
    
    dom.bgCanvas.width = DRAW_CONFIG.canvasW * DRAW_CONFIG.dpr;
    dom.bgCanvas.height = DRAW_CONFIG.canvasH * DRAW_CONFIG.dpr;
    dom.imageCanvas.width = DRAW_CONFIG.canvasW * DRAW_CONFIG.dpr;
    dom.imageCanvas.height = DRAW_CONFIG.canvasH * DRAW_CONFIG.dpr;
    dom.drawCanvas.width = DRAW_CONFIG.canvasW * DRAW_CONFIG.dpr;
    dom.drawCanvas.height = DRAW_CONFIG.canvasH * DRAW_CONFIG.dpr;
    
    dom.bgCanvas.style.width = DRAW_CONFIG.canvasW + 'px';
    dom.bgCanvas.style.height = DRAW_CONFIG.canvasH + 'px';
    dom.imageCanvas.style.width = DRAW_CONFIG.canvasW + 'px';
    dom.imageCanvas.style.height = DRAW_CONFIG.canvasH + 'px';
    dom.drawCanvas.style.width = DRAW_CONFIG.canvasW + 'px';
    dom.drawCanvas.style.height = DRAW_CONFIG.canvasH + 'px';
    
    dom.bgCtx.scale(DRAW_CONFIG.dpr, DRAW_CONFIG.dpr);
    dom.imageCtx.scale(DRAW_CONFIG.dpr, DRAW_CONFIG.dpr);
    dom.drawCtx.scale(DRAW_CONFIG.dpr, DRAW_CONFIG.dpr);
    
    dom.imageCtx.imageSmoothingEnabled = true;
    dom.imageCtx.imageSmoothingQuality = 'medium';
    
    const dc = dom.drawCtx;
    // 降低图像平滑质量，减少GPU负载
    dc.imageSmoothingEnabled = true;
    dc.imageSmoothingQuality = 'medium'; // 从'high'降低到'medium'
    dc.lineCap = 'round';
    dc.lineJoin = 'round';
    dc.miterLimit = 10;
    
    resetBgCanvas();
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
        state.moveBound.minX = 0;
        state.moveBound.maxX = 0;
    }
    
    if (scaledH >= screenH) {
        state.moveBound.minY = -(scaledH - screenH);
        state.moveBound.maxY = 0;
    } else {
        state.moveBound.minY = 0;
        state.moveBound.maxY = 0;
    }
}

function clampCanvasPosition() {
    state.canvasX = Math.max(state.moveBound.minX, Math.min(state.moveBound.maxX, state.canvasX));
    state.canvasY = Math.max(state.moveBound.minY, Math.min(state.moveBound.maxY, state.canvasY));
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
        <button class="menu-item menu-item-danger" id="menuClose">
            <img src="assets/javascript.svg" width="16" height="16" alt="关闭" style="filter: invert(1);">
            关闭
        </button>
    `;
    
    dom.canvasContainer.appendChild(menuPopup);
    
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
        await appWindow.minimize();
        console.log('窗口已最小化');
    } else {
        console.log('Tauri API 不可用');
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
    dom.penSizeSlider.addEventListener('input', (e) => {
        DRAW_CONFIG.penWidth = Number(e.target.value);
        dom.penSizeValue.textContent = DRAW_CONFIG.penWidth;
        if (state.drawMode === 'comment') {
            setPenStyle();
        }
    });
    
    dom.penColorPicker.addEventListener('input', (e) => {
        DRAW_CONFIG.penColor = e.target.value;
        if (state.drawMode === 'comment') {
            setPenStyle();
        }
    });
    
    dom.eraserSizeSlider.addEventListener('input', (e) => {
        DRAW_CONFIG.eraserSize = Number(e.target.value);
        dom.eraserSizeValue.textContent = DRAW_CONFIG.eraserSize;
        updateEraserHintSize();
        if (state.drawMode === 'eraser') {
            setEraserStyle();
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
    
    const penRow = panel.querySelector('.pen-control-row:nth-child(1)');
    const colorRow = panel.querySelector('.pen-control-row:nth-child(2)');
    const eraserRow = panel.querySelector('.pen-control-row:nth-child(3)');
    
    if (mode === 'comment') {
        if (penRow) penRow.style.display = 'flex';
        if (colorRow) colorRow.style.display = 'flex';
        if (eraserRow) eraserRow.style.display = 'none';
    } else if (mode === 'eraser') {
        if (penRow) penRow.style.display = 'none';
        if (colorRow) colorRow.style.display = 'none';
        if (eraserRow) eraserRow.style.display = 'flex';
    }
    
    panel.style.position = 'absolute';
    panel.style.bottom = 'auto';
    panel.style.top = 'auto';
    panel.style.right = 'auto';
    panel.style.left = 'auto';
    
    const panelWidth = 180;
    const panelHeight = panel.offsetHeight || 100;
    
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

function hidePenControlPanel() {
    dom.penControlPanel.classList.remove('visible');
}

function updateEraserHintPos(clientX, clientY) {
    const rect = dom.canvasContainer.getBoundingClientRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;
    dom.eraserHint.style.left = `${x}px`;
    dom.eraserHint.style.top = `${y}px`;
    dom.eraserHint.style.transform = `translate(-50%, -50%) scale(${state.scale})`;
}

// ==================== 画布交互事件 ====================

function bindCanvasMouseEvents() {
    dom.drawCanvas.addEventListener('mousedown', handleMouseDown);
    dom.drawCanvas.addEventListener('mousemove', handleMouseMove);
    dom.drawCanvas.addEventListener('mouseup', handleMouseUp);
    dom.drawCanvas.addEventListener('mouseleave', handleMouseLeave);
    dom.drawCanvas.addEventListener('wheel', handleWheel, { passive: false });
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
        dom.drawCanvas.classList.add('dragging');
    } else if (state.drawMode === 'comment') {
        state.isDrawing = true;
        state.lastX = (e.clientX - rect.left) / state.scale;
        state.lastY = (e.clientY - rect.top) / state.scale;
        startStroke('draw');
    } else if (state.drawMode === 'eraser') {
        state.isDrawing = true;
        state.lastX = (e.clientX - rect.left) / state.scale;
        state.lastY = (e.clientY - rect.top) / state.scale;
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
        updateCanvasTransform();
    } else if (state.isDrawing) {
        const rect = dom.drawCanvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) / state.scale;
        const y = (e.clientY - rect.top) / state.scale;
        
        // 计算距离，只添加足够远的点
        const distance = Math.sqrt(Math.pow(x - state.lastX, 2) + Math.pow(y - state.lastY, 2));
        
        // 使用智能调度器的最小点间距参数
        const minDistance = smartDrawScheduler.getMinDistance();
        
        // 只在距离足够大时添加点，减少点数量
        if (distance > minDistance) {
            // 添加到绘制队列
            state.pendingDrawPoints.push({ fromX: state.lastX, fromY: state.lastY, toX: x, toY: y });
            
            // 使用优化的点收集（只影响历史存储）
            pointCollector.addPoint(state.lastX, state.lastY, x, y);
            
            // 总是添加到笔画历史
            addStrokePoint(state.lastX, state.lastY, x, y);
            
            state.lastX = x;
            state.lastY = y;
            
            // 调度绘制，但限制频率
            if (!state.drawRafId) {
                state.drawRafId = requestAnimationFrame(flushDrawPoints);
            }
        }
    }
}

function flushDrawPoints() {
    if (state.pendingDrawPoints.length === 0) {
        state.drawRafId = null;
        return;
    }
    
    const startTime = performance.now();
    
    // 使用智能调度器的参数
    const maxPointsPerFlush = smartDrawScheduler.getMaxPointsPerFlush();
    const pointsToProcess = state.pendingDrawPoints.slice(0, maxPointsPerFlush);
    const remainingPoints = state.pendingDrawPoints.slice(maxPointsPerFlush);
    
    // 开始绘制
    batchDrawManager.startDrawing();
    
    // 使用批处理管理器
    for (const point of pointsToProcess) {
        const type = state.drawMode === 'eraser' ? 'erase' : 'draw';
        const color = state.drawMode === 'comment' ? DRAW_CONFIG.penColor : '#000000';
        const lineWidth = state.drawMode === 'comment' ? DRAW_CONFIG.penWidth : DRAW_CONFIG.eraserSize;
        
        batchDrawManager.addCommand(type, point.fromX, point.fromY, point.toX, point.toY, color, lineWidth);
    }
    
    // 执行批处理
    batchDrawManager.flushAll();
    
    // 结束绘制
    batchDrawManager.endDrawing();
    
    // 计算绘制时间并记录性能
    const drawTime = performance.now() - startTime;
    smartDrawScheduler.recordPerformance(drawTime);
    
    // 更新待处理点
    state.pendingDrawPoints = remainingPoints;
    
    // 如果还有剩余点，继续调度绘制
    if (state.pendingDrawPoints.length > 0) {
        state.drawRafId = requestAnimationFrame(flushDrawPoints);
    } else {
        state.drawRafId = null;
    }
}

function handleMouseUp(e) {
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
        flushDrawPoints();
        endStroke();
    }
}

function handleMouseLeave(e) {
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
        flushDrawPoints();
        endStroke();
    }
}

function handleWheel(e) {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.1 : 0.1;
    const newScale = Math.max(DRAW_CONFIG.minScale, Math.min(DRAW_CONFIG.maxScale, state.scale + delta));
    
    if (newScale !== state.scale) {
        const containerRect = dom.canvasContainer.getBoundingClientRect();
        const mouseX = e.clientX - containerRect.left;
        const mouseY = e.clientY - containerRect.top;
        
        const oldScale = state.scale;
        state.scale = newScale;
        
        updateMoveBound();
        
        const scaleRatio = newScale / oldScale;
        state.canvasX = mouseX - (mouseX - state.canvasX) * scaleRatio;
        state.canvasY = mouseY - (mouseY - state.canvasY) * scaleRatio;
        
        clampCanvasPosition();
        updateCanvasTransform();
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
        } else if (state.drawMode === 'comment') {
            state.isDrawing = true;
            state.lastX = (touch.clientX - rect.left) / state.scale;
            state.lastY = (touch.clientY - rect.top) / state.scale;
            startStroke('draw');
        } else if (state.drawMode === 'eraser') {
            state.isDrawing = true;
            updateEraserHintPos(touch.clientX, touch.clientY);
            state.lastX = (touch.clientX - rect.left) / state.scale;
            state.lastY = (touch.clientY - rect.top) / state.scale;
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
        updateCanvasTransform();
    } else if (touches.length === 1 && state.isDrawing) {
        const touch = touches[0];
        if (state.drawMode === 'eraser') {
            updateEraserHintPos(touch.clientX, touch.clientY);
        }
        
        const x = (touch.clientX - rect.left) / state.scale;
        const y = (touch.clientY - rect.top) / state.scale;
        
        // 总是添加到绘制队列以确保流畅绘制
        state.pendingDrawPoints.push({ fromX: state.lastX, fromY: state.lastY, toX: x, toY: y });
        
        // 使用优化的点收集（只影响历史存储）
        pointCollector.addPoint(state.lastX, state.lastY, x, y);
        
        // 总是添加到笔画历史
        addStrokePoint(state.lastX, state.lastY, x, y);
        
        state.lastX = x;
        state.lastY = y;
        
        // 立即调度绘制
        if (!state.drawRafId) {
            state.drawRafId = requestAnimationFrame(flushDrawPoints);
        }
    } else if (touches.length === 2 && state.isScaling) {
        const currentDistance = getTouchDistance(touches[0], touches[1]);
        const scaleRatio = currentDistance / state.startDistance;
        let newScale = state.startScale * scaleRatio;
        newScale = Math.max(DRAW_CONFIG.minScale, Math.min(DRAW_CONFIG.maxScale, newScale));
        
        const centerX = (touches[0].clientX + touches[1].clientX) / 2;
        const centerY = (touches[0].clientY + touches[1].clientY) / 2;
        
        const finalRatio = newScale / state.startScale;
        state.canvasX = centerX - (state.startScaleX - state.startCanvasX) * finalRatio;
        state.canvasY = centerY - (state.startScaleY - state.startCanvasY) * finalRatio;
        state.scale = newScale;
        
        clampCanvasPosition();
        updateCanvasTransform();
    }
}

function handleTouchEnd(e) {
    e.preventDefault();
    
    if (e.touches.length === 0) {
        state.isDragging = false;
        state.isScaling = false;
        
        if (state.isDrawing) {
            state.isDrawing = false;
            if (state.drawRafId) {
                cancelAnimationFrame(state.drawRafId);
                state.drawRafId = null;
            }
            flushDrawPoints();
            endStroke();
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
    return Math.sqrt(dx * dx + dy * dy);
}

// 更新画布变换
function updateCanvasTransform() {
    const transform = `translate(${state.canvasX}px, ${state.canvasY}px) scale(${state.scale})`;
    dom.bgCanvas.style.transform = transform;
    dom.imageCanvas.style.transform = transform;
    dom.drawCanvas.style.transform = transform;
    dom.bgCanvas.style.transformOrigin = '0 0';
    dom.imageCanvas.style.transformOrigin = '0 0';
    dom.drawCanvas.style.transformOrigin = '0 0';
}

// 撤销功能 - 混合方案：路径记录 + ImageData压缩
function startStroke(type) {
    state.currentStroke = {
        type: type,
        points: [],
        color: DRAW_CONFIG.penColor,
        lineWidth: DRAW_CONFIG.penWidth,
        eraserSize: DRAW_CONFIG.eraserSize
    };
    
    // 开始批处理绘制
    batchDrawManager.startDrawing();
}

function addStrokePoint(fromX, fromY, toX, toY) {
    if (state.currentStroke) {
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

function endStroke() {
    if (state.currentStroke && state.currentStroke.points.length > 0) {
        // 只在点数较多时简化，避免过度简化导致断点
        if (state.currentStroke.points.length > 50) {
            // 使用更小的 epsilon 值以保留更多点
            state.currentStroke.points = simplifyPoints(state.currentStroke.points, POINT_OPTIMIZATION.epsilon * 0.8);
        }
        
        state.strokeHistory.push(state.currentStroke);
        
        if (state.strokeHistory.length > state.STROKE_COMPACT_THRESHOLD) {
            compactStrokes();
        }
        
        updateUndoBtnStatus();
    }
    state.currentStroke = null;
    
    // 结束批处理绘制
    batchDrawManager.endDrawing();
    
    // 清理点收集器
    pointCollector.clear();
    
    // 清理批处理管理器
    batchDrawManager.clear();
}

let compactIdleId = null;

// ==================== 批注绘制系统 ====================

// 上下文状态缓存
let currentContextState = {
    strokeStyle: null,
    lineWidth: null,
    lineCap: null,
    lineJoin: null,
    globalCompositeOperation: null
};

// ==================== 线段点优化 ====================

// 点简化配置
const POINT_OPTIMIZATION = {
    epsilon: 0.3,  // 简化阈值 (像素) - 减小以保留更多点
    minDistance: 1, // 最小点间距 (像素) - 减小以减少断点
    quantization: 0.25, // 坐标量化步长 (像素) - 提高精度
    maxPointsPerStroke: 1500, // 每笔画最大点数 - 增加以容纳更多点
    batchInterval: 30, // 批量收集间隔 (ms) - 减小以提高响应速度
};

/**
 * 坐标量化
 * 将坐标值量化到指定步长，减少精度
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
 * Douglas-Peucker 点简化算法
 * 减少冗余点，保留关键特征点
 */
function simplifyPoints(points, epsilon) {
    if (points.length <= 2) return points;
    
    let maxDist = 0;
    let maxIndex = 0;
    
    // 找到距离最远的点
    for (let i = 1; i < points.length - 1; i++) {
        const dist = perpendicularDistance(
            points[i].fromX, points[i].fromY,
            points[0].fromX, points[0].fromY,
            points[points.length - 1].toX, points[points.length - 1].toY
        );
        if (dist > maxDist) {
            maxDist = dist;
            maxIndex = i;
        }
    }
    
    // 如果最远点距离大于阈值，递归简化
    if (maxDist > epsilon) {
        const left = simplifyPoints(points.slice(0, maxIndex + 1), epsilon);
        const right = simplifyPoints(points.slice(maxIndex), epsilon);
        return [...left.slice(0, -1), ...right];
    } else {
        // 否则只保留起点和终点
        return [points[0], points[points.length - 1]];
    }
}

/**
 * 计算点到线段的垂直距离
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
    
    addPoint(fromX, fromY, toX, toY) {
        // 量化坐标
        const qFromX = quantizeCoord(fromX);
        const qFromY = quantizeCoord(fromY);
        const qToX = quantizeCoord(toX);
        const qToY = quantizeCoord(toY);
        
        // 检查距离阈值
        if (distance(qFromX, qFromY, qToX, qToY) < POINT_OPTIMIZATION.minDistance) {
            return false; // 距离太小，跳过
        }
        
        // 检查时间间隔
        const now = Date.now();
        if (now - this.lastTime < POINT_OPTIMIZATION.batchInterval) {
            return false; // 时间间隔太小，跳过
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
        
        // 限制每笔画最大点数
        if (this.points.length > POINT_OPTIMIZATION.maxPointsPerStroke) {
            this.points = simplifyPoints(this.points, POINT_OPTIMIZATION.epsilon);
        }
        
        return true;
    }
    
    getPoints() {
        return this.points;
    }
    
    clear() {
        this.points = [];
        this.lastTime = Date.now();
    }
}

// 全局点收集器
const pointCollector = new PointCollector();

// ==================== 批处理绘制系统 ====================

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
    }
    
    /**
     * 添加绘制命令
     */
    addCommand(type, fromX, fromY, toX, toY, color, lineWidth) {
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
     * 开始绘制
     */
    startDrawing() {
        this.isDrawing = true;
        this.pendingFlush = false;
    }
    
    /**
     * 结束绘制
     */
    endDrawing() {
        this.isDrawing = false;
        this.flushAll();
    }
    
    /**
     * 执行单个批处理
     */
    flushBatch(stateKey) {
        const batch = this.batches.get(stateKey);
        if (!batch || batch.commands.length === 0) return;
        
        const ctx = dom.drawCtx;
        
        // 限制单次批处理的命令数量，避免GPU负载过高
        const maxCommandsPerBatch = 100;
        const commandsToProcess = batch.commands.slice(0, maxCommandsPerBatch);
        const remainingCommands = batch.commands.slice(maxCommandsPerBatch);
        
        // 设置上下文状态
        setContextState(ctx, {
            strokeStyle: batch.color,
            lineWidth: batch.lineWidth,
            lineCap: 'round',
            lineJoin: 'round',
            globalCompositeOperation: batch.type === 'erase' ? 'destination-out' : 'source-over'
        });
        
        // 批量绘制
        ctx.beginPath();
        for (const cmd of commandsToProcess) {
            ctx.moveTo(cmd.fromX, cmd.fromY);
            ctx.lineTo(cmd.toX, cmd.toY);
        }
        ctx.stroke();
        
        // 更新批处理命令
        batch.commands = remainingCommands;
    }
    
    /**
     * 执行所有批处理
     */
    flushAll() {
        for (const stateKey of this.batches.keys()) {
            this.flushBatch(stateKey);
        }
        
        // 清理空批处理
        for (const [stateKey, batch] of this.batches.entries()) {
            if (batch.commands.length === 0) {
                this.batches.delete(stateKey);
            }
        }
    }
    
    /**
     * 清空所有批处理
     */
    clear() {
        this.batches.clear();
    }
    
    /**
     * 获取批处理数量
     */
    getBatchCount() {
        return this.batches.size;
    }
}

// 全局批处理管理器
const batchDrawManager = new BatchDrawManager();

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
 * 按状态分组批量绘制笔画
 * @param {CanvasRenderingContext2D} ctx - 目标上下文
 * @param {Array} strokes - 笔画数组
 */
function drawStrokes(ctx, strokes) {
    // 清空现有批处理
    batchDrawManager.clear();
    
    // 限制单次绘制的笔画数量，避免GPU负载过高
    const maxStrokesPerBatch = 50;
    const totalStrokes = strokes.length;
    
    for (let i = 0; i < totalStrokes; i += maxStrokesPerBatch) {
        const batchStrokes = strokes.slice(i, i + maxStrokesPerBatch);
        
        // 按状态分组添加到批处理
        for (const stroke of batchStrokes) {
            if (stroke.type === 'draw') {
                for (const point of stroke.points) {
                    batchDrawManager.addCommand(
                        'draw',
                        point.fromX, point.fromY,
                        point.toX, point.toY,
                        stroke.color,
                        stroke.lineWidth
                    );
                }
            } else if (stroke.type === 'erase') {
                for (const point of stroke.points) {
                    batchDrawManager.addCommand(
                        'erase',
                        point.fromX, point.fromY,
                        point.toX, point.toY,
                        '#000000',
                        stroke.eraserSize
                    );
                }
            }
        }
        
        // 执行批处理
        batchDrawManager.flushAll();
    }
    
    // 恢复默认合成模式
    setContextState(ctx, {
        globalCompositeOperation: 'source-over'
    });
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
            
            state.baseImageURL = result;
            state.baseImageObj = null;
            const img = new Image();
            img.onload = () => {
                state.baseImageObj = img;
            };
            img.src = result;
            
            console.log('Rust 笔画已压缩，保留最近', state.strokeHistory.length, '笔可撤销');
            return;
        } catch (error) {
            console.error('Rust 笔画压缩失败，使用前端降级方案:', error);
        }
    }
    
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = DRAW_CONFIG.canvasW * DRAW_CONFIG.dpr;
    tempCanvas.height = DRAW_CONFIG.canvasH * DRAW_CONFIG.dpr;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.scale(DRAW_CONFIG.dpr, DRAW_CONFIG.dpr);
    
    if (state.baseImageObj) {
        tempCtx.drawImage(state.baseImageObj, 0, 0, DRAW_CONFIG.canvasW, DRAW_CONFIG.canvasH);
    }
    
    drawStrokes(tempCtx, strokesToCompact);
    
    state.baseImageURL = tempCanvas.toDataURL('image/png');
    state.baseImageObj = null;
    const img = new Image();
    img.onload = () => {
        state.baseImageObj = img;
    };
    img.src = state.baseImageURL;
    
    console.log('笔画已异步压缩，保留最近', state.strokeHistory.length, '笔可撤销');
}

function compactStrokes() {
    scheduleCompact();
}

function saveSnapshot() {
    endStroke();
}

function undo() {
    if (state.strokeHistory.length === 0) return;
    
    state.strokeHistory.pop();
    redrawAllStrokes();
    updateUndoBtnStatus();
    console.log('撤销操作');
}

function redrawAllStrokes() {
    clearDrawCanvas();
    
    if (state.baseImageObj) {
        dom.drawCtx.drawImage(state.baseImageObj, 0, 0, DRAW_CONFIG.canvasW, DRAW_CONFIG.canvasH);
    }
    
    drawStrokes(dom.drawCtx, state.strokeHistory);
}

function updateUndoBtnStatus() {
    dom.btnUndo.disabled = state.strokeHistory.length === 0;
}

// 清空画布
function clearDrawCanvas() {
    dom.drawCtx.clearRect(0, 0, DRAW_CONFIG.canvasW, DRAW_CONFIG.canvasH);
    setContextState(dom.drawCtx, {
        globalCompositeOperation: 'source-over'
    });
}

function clearAllDrawings() {
    clearDrawCanvas();
    state.strokeHistory = [];
    state.baseImageURL = null;
    state.baseImageObj = null;
    updateUndoBtnStatus();
    
    if (state.currentImageIndex >= 0 && state.currentImageIndex < state.imageList.length) {
        state.imageList[state.currentImageIndex].strokeHistory = null;
        state.imageList[state.currentImageIndex].baseImageURL = null;
    }
    
    if (state.currentFolderIndex >= 0 && state.currentFolderPageIndex >= 0) {
        if (state.currentFolderIndex < state.fileList.length) {
            const folder = state.fileList[state.currentFolderIndex];
            if (state.currentFolderPageIndex < folder.pages.length) {
                folder.pages[state.currentFolderPageIndex].strokeHistory = null;
                folder.pages[state.currentFolderPageIndex].baseImageURL = null;
            }
        }
    }
    
    if (state.drawMode === 'eraser') {
        switchMode('comment');
    }
    
    console.log('清空所有批注');
}

function resetBgCanvas() {
    dom.bgCtx.clearRect(0, 0, DRAW_CONFIG.canvasW, DRAW_CONFIG.canvasH);
    dom.bgCtx.fillStyle = '#3a3a3a';
    dom.bgCtx.fillRect(0, 0, DRAW_CONFIG.canvasW, DRAW_CONFIG.canvasH);
}

// 拍照功能
function takePhoto() {
    if (state.isCameraOpen) {
        captureCamera();
    } else if (state.currentImageIndex >= 0 && state.imageList.length > 0) {
        saveCurrentDrawData();
        saveCurrentFolderPageDrawData();
        state.currentImageIndex = -1;
        state.currentImage = null;
        clearImageLayer();
        clearDrawCanvas();
        openCamera();
        updateSidebarSelection();
        updatePhotoButtonState();
        updateEnhanceButtonState();
        console.log('返回摄像头');
    } else if (state.currentFolderIndex >= 0 && state.currentFolderPageIndex >= 0) {
        saveCurrentDrawData();
        saveCurrentFolderPageDrawData();
        state.currentFolderIndex = -1;
        state.currentFolderPageIndex = -1;
        state.currentImage = null;
        clearImageLayer();
        clearDrawCanvas();
        openCamera();
        updateFolderPageSelection(-1, -1);
        updatePhotoButtonState();
        updateEnhanceButtonState();
        console.log('返回摄像头');
    } else {
        saveMergedCanvas();
    }
}

function saveMergedCanvas() {
    console.log('执行拍照功能');
    const mergedCanvas = document.createElement('canvas');
    mergedCanvas.width = DRAW_CONFIG.canvasW * DRAW_CONFIG.dpr;
    mergedCanvas.height = DRAW_CONFIG.canvasH * DRAW_CONFIG.dpr;
    const mergedCtx = mergedCanvas.getContext('2d');
    mergedCtx.scale(DRAW_CONFIG.dpr, DRAW_CONFIG.dpr);
    
    mergedCtx.drawImage(dom.bgCanvas, 0, 0, DRAW_CONFIG.canvasW, DRAW_CONFIG.canvasH);
    mergedCtx.drawImage(dom.imageCanvas, 0, 0, DRAW_CONFIG.canvasW, DRAW_CONFIG.canvasH);
    mergedCtx.drawImage(dom.drawCanvas, 0, 0, DRAW_CONFIG.canvasW, DRAW_CONFIG.canvasH);
    
    const link = document.createElement('a');
    link.download = `photo_${Date.now()}.png`;
    link.href = mergedCanvas.toDataURL('image/png');
    link.click();
}

function updatePhotoButtonState() {
    const btnPhoto = dom.btnPhoto;
    if (!btnPhoto) return;
    
    if (state.isCameraOpen) {
        btnPhoto.classList.add('camera-active');
        btnPhoto.innerHTML = `
            <img src="assets/tauri.svg" width="16" height="16" alt="拍照" style="filter: invert(1);">
            拍照
        `;
        btnPhoto.title = '捕获摄像头画面';
    } else if ((state.currentImageIndex >= 0 && state.imageList.length > 0) || 
               (state.currentFolderIndex >= 0 && state.currentFolderPageIndex >= 0)) {
        btnPhoto.classList.remove('camera-active');
        btnPhoto.innerHTML = `
            <img src="assets/javascript.svg" width="16" height="16" alt="切换到摄像头" style="filter: invert(1);">
            切换到摄像头
        `;
        btnPhoto.title = '返回摄像头';
    } else {
        btnPhoto.classList.remove('camera-active');
        btnPhoto.innerHTML = `
            <img src="assets/tauri.svg" width="16" height="16" alt="拍照" style="filter: invert(1);">
            拍照
        `;
        btnPhoto.title = '保存画布截图';
    }
    
    const btnCamera = document.getElementById('btnCamera');
    if (btnCamera) {
        if (state.isCameraOpen) {
            btnCamera.classList.add('camera-active');
            btnCamera.innerHTML = `
                <img src="assets/javascript.svg" width="16" height="16" alt="关闭摄像头" style="filter: invert(1);">
                关闭
            `;
        } else {
            btnCamera.classList.remove('camera-active');
            btnCamera.innerHTML = `
                <img src="assets/javascript.svg" width="16" height="16" alt="摄像头" style="filter: invert(1);">
                摄像头
            `;
        }
    }
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

async function rotateImage(direction) {
    if (state.isCameraOpen) {
        if (direction === 'left') {
            state.cameraRotation = (state.cameraRotation - 90 + 360) % 360;
        } else {
            state.cameraRotation = (state.cameraRotation + 90) % 360;
        }
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

function updateEnhanceButtonState() {
    if (!dom.btnEnhance) return;
    
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
    
    let imageListHTML = '';
    if (state.imageList.length === 0) {
        imageListHTML = '<div class="sidebar-empty">暂无图片</div>';
    } else {
        state.imageList.forEach((imgData, index) => {
            const isActive = (state.currentImageIndex >= 0 && index === state.currentImageIndex) ? 'active' : '';
            imageListHTML += `
                <div class="sidebar-image-item ${isActive}" data-index="${index}">
                    <img src="${imgData.thumbnail}" class="sidebar-thumbnail" alt="图片${index + 1}">
                    <div class="sidebar-image-actions">
                        <button class="sidebar-btn-delete" title="删除">✕</button>
                    </div>
                </div>
            `;
        });
    }
    
    sidebarElement.innerHTML = `
        <div class="sidebar-header"><span class="sidebar-header-text">图片列表</span></div>
        <div class="sidebar-content">
            ${imageListHTML}
        </div>
        <button class="sidebar-import-btn" id="btnImportImageSidebar">
            <img src="assets/javascript.svg" width="16" height="16" alt="导入" style="filter: invert(1);">
            导入图片
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
        <img src="assets/move.svg" width="16" height="16" alt="收起" style="filter: invert(1);">
        收起
    `;
    console.log('展开侧边栏');
}

function selectImage(index) {
    if (index < 0 || index >= state.imageList.length) return;
    
    if (index === state.currentImageIndex && state.currentImage) {
        saveCurrentDrawData();
        saveCurrentFolderPageDrawData();
        state.currentImageIndex = -1;
        state.currentImage = null;
        clearImageLayer();
        clearDrawCanvas();
        if (state.isCameraOpen) {
            closeCamera();
        }
        openCamera();
        updateSidebarSelection();
        updatePhotoButtonState();
        updateEnhanceButtonState();
        console.log('返回摄像头');
        return;
    }
    
    saveCurrentDrawData();
    saveCurrentFolderPageDrawData();
    
    state.currentImageIndex = index;
    state.currentFolderIndex = -1;
    state.currentFolderPageIndex = -1;
    
    const imgData = state.imageList[index];
    
    if (imgData.viewState) {
        state.scale = imgData.viewState.scale;
        state.canvasX = imgData.viewState.canvasX;
        state.canvasY = imgData.viewState.canvasY;
        updateMoveBound();
        updateCanvasTransform();
    } else {
        state.scale = 1;
        state.canvasX = -(DRAW_CONFIG.canvasW - DRAW_CONFIG.screenW) / 2;
        state.canvasY = -(DRAW_CONFIG.canvasH - DRAW_CONFIG.screenH) / 2;
        updateMoveBound();
        updateCanvasTransform();
    }
    
    const img = new Image();
    img.onload = () => {
        state.currentImage = img;
        
        if (state.isCameraOpen) {
            closeCamera();
        } else {
            drawImageToCenter(img);
        }
        
        restoreDrawData(index);
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
        state.imageList[state.currentImageIndex].strokeHistory = [...state.strokeHistory];
        state.imageList[state.currentImageIndex].baseImageURL = state.baseImageURL;
        state.imageList[state.currentImageIndex].viewState = {
            scale: state.scale,
            canvasX: state.canvasX,
            canvasY: state.canvasY
        };
    }
}

function restoreDrawData(index) {
    if (index >= 0 && index < state.imageList.length) {
        const imgData = state.imageList[index];
        if (imgData.strokeHistory && imgData.strokeHistory.length > 0) {
            state.strokeHistory = [...imgData.strokeHistory];
        } else {
            state.strokeHistory = [];
        }
        
        state.baseImageURL = imgData.baseImageURL || null;
        state.baseImageObj = null;
        
        if (state.baseImageURL) {
            const img = new Image();
            img.onload = () => {
                state.baseImageObj = img;
                redrawAllStrokes();
                updateUndoBtnStatus();
            };
            img.src = state.baseImageURL;
        } else {
            if (state.strokeHistory.length > 0) {
                redrawAllStrokes();
            } else {
                clearDrawCanvas();
            }
            updateUndoBtnStatus();
        }
    }
}

function deleteImage(index) {
    if (index < 0 || index >= state.imageList.length) return;
    
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

function updateSidebarSelection() {
    const sidebarContent = document.querySelector('.sidebar:not(.file-sidebar) .sidebar-content');
    if (!sidebarContent) return;
    
    document.querySelectorAll('.sidebar:not(.file-sidebar) .sidebar-image-item').forEach((item, idx) => {
        if (state.currentImageIndex >= 0 && idx === state.currentImageIndex) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    });
    
    document.querySelectorAll('.file-sidebar .sidebar-image-item').forEach(item => {
        item.classList.remove('active');
    });
}

function updateSidebarContent() {
    const sidebarContent = document.querySelector('.sidebar-content');
    if (!sidebarContent) return;
    
    let imageListHTML = '';
    if (state.imageList.length === 0) {
        imageListHTML = '<div class="sidebar-empty">暂无图片</div>';
    } else {
        state.imageList.forEach((imgData, index) => {
            const isActive = (state.currentImageIndex >= 0 && index === state.currentImageIndex) ? 'active' : '';
            imageListHTML += `
                <div class="sidebar-image-item ${isActive}" data-index="${index}">
                    <img src="${imgData.thumbnail}" class="sidebar-thumbnail" alt="图片${index + 1}">
                    <div class="sidebar-image-actions">
                        <button class="sidebar-btn-delete" title="删除">✕</button>
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
    
    dom.btnExpand.innerHTML = `
        <img src="assets/move.svg" width="16" height="16" alt="图片" style="filter: invert(1);">
        图片
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
    const fileSidebarElement = document.createElement('div');
    fileSidebarElement.classList.add('sidebar', 'file-sidebar');
    
    let contentHTML = '';
    if (state.fileList.length === 0) {
        contentHTML = '<div class="sidebar-empty">暂无文件</div>';
    } else {
        state.fileList.forEach((folder, index) => {
            contentHTML += `
                <div class="sidebar-folder-item" data-index="${index}">
                    <img src="assets/javascript.svg" width="16" height="16" alt="文件夹" style="filter: invert(1);">
                    <span class="folder-name">${folder.name}</span>
                    <span class="folder-count">${folder.pages.length}页</span>
                </div>
            `;
        });
    }
    
    fileSidebarElement.innerHTML = `
        <div class="sidebar-header"><span class="sidebar-header-text">文件列表</span></div>
        <div class="sidebar-content">
            ${contentHTML}
        </div>
        <button class="sidebar-import-btn" id="btnAddFile">
            <img src="assets/javascript.svg" width="16" height="16" alt="添加" style="filter: invert(1);">
            添加文件
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
        <img src="assets/javascript.svg" width="16" height="16" alt="收起" style="filter: invert(1);">
        收起
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
        pagesHTML += `
            <div class="sidebar-image-item ${isActive}" data-folder="${folderIndex}" data-page="${index}">
                <img src="${page.thumbnail}" class="sidebar-thumbnail" alt="第${index + 1}页">
                <div class="sidebar-page-label">第${index + 1}页</div>
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
    const sidebarHeader = document.querySelector('.file-sidebar .sidebar-header');
    if (sidebarHeader) {
        sidebarHeader.innerHTML = '<span class="sidebar-header-text">文件列表</span>';
        sidebarHeader.classList.remove('has-back');
    }
    
    updateFileSidebarContent();
    console.log('关闭文件夹');
}

function selectFolderPage(folderIndex, pageIndex) {
    if (folderIndex < 0 || folderIndex >= state.fileList.length) return;
    
    const folder = state.fileList[folderIndex];
    if (pageIndex < 0 || pageIndex >= folder.pages.length) return;
    
    if (state.isCameraOpen) {
        closeCamera();
    }
    
    saveCurrentDrawData();
    saveCurrentFolderPageDrawData();
    
    const page = folder.pages[pageIndex];
    
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
    
    const img = new Image();
    img.onload = () => {
        state.currentImage = img;
        state.currentImageIndex = -1;
        state.currentFolderIndex = folderIndex;
        state.currentFolderPageIndex = pageIndex;
        drawImageToCenter(img);
        restoreFolderPageDrawData(folderIndex, pageIndex);
        updateFolderPageSelection(folderIndex, pageIndex);
        updatePhotoButtonState();
        updateEnhanceButtonState();
    };
    img.src = page.full;
    
    console.log(`选择: ${folder.name} 第${pageIndex + 1}页`);
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
                folder.pages[state.currentFolderPageIndex].strokeHistory = [...state.strokeHistory];
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

function restoreFolderPageDrawData(folderIndex, pageIndex) {
    if (folderIndex >= 0 && folderIndex < state.fileList.length) {
        const folder = state.fileList[folderIndex];
        if (pageIndex >= 0 && pageIndex < folder.pages.length) {
            const page = folder.pages[pageIndex];
            if (page.strokeHistory && page.strokeHistory.length > 0) {
                state.strokeHistory = [...page.strokeHistory];
            } else {
                state.strokeHistory = [];
            }
            
            state.baseImageURL = page.baseImageURL || null;
            state.baseImageObj = null;
            
            if (state.baseImageURL) {
                const img = new Image();
                img.onload = () => {
                    state.baseImageObj = img;
                    redrawAllStrokes();
                    updateUndoBtnStatus();
                };
                img.src = state.baseImageURL;
            } else {
                if (state.strokeHistory.length > 0) {
                    redrawAllStrokes();
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
    
    let contentHTML = '';
    if (state.fileList.length === 0) {
        contentHTML = '<div class="sidebar-empty">暂无文件</div>';
    } else {
        state.fileList.forEach((folder, index) => {
            contentHTML += `
                <div class="sidebar-folder-item" data-index="${index}">
                    <img src="assets/javascript.svg" width="16" height="16" alt="文件夹" style="filter: invert(1);">
                    <span class="folder-name">${folder.name}</span>
                    <span class="folder-count">${folder.pages.length}页</span>
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
    input.accept = '.pdf';
    
    input.onchange = async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        
        if (state.isCameraOpen) {
            closeCamera();
        }
        
        showLoadingOverlay('正在导入文件...');
        
        try {
            const pdfReady = await waitForPdfJs();
            if (!pdfReady) {
                hideLoadingOverlay();
                alert('PDF库加载超时，请重启应用后重试');
                return;
            }
            
            const arrayBuffer = await file.arrayBuffer();
            const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
            
            const totalPages = pdf.numPages;
            const folder = {
                name: file.name.replace('.pdf', ''),
                pages: []
            };
            
            folder.pages = await processPdfPagesParallel(pdf, totalPages);
            
            state.fileList.push(folder);
            updateFileSidebarContent();
            
            const existingFileSidebar = document.querySelector('.file-sidebar');
            if (!existingFileSidebar) {
                expandFileSidebar();
            }
            
            if (folder.pages.length > 0) {
                const firstPage = folder.pages[0];
                const img = new Image();
                img.onload = () => {
                    state.currentImage = img;
                    state.currentFolderIndex = state.fileList.length - 1;
                    state.currentFolderPageIndex = 0;
                    drawImageToCenter(img);
                    updatePhotoButtonState();
                    updateEnhanceButtonState();
                };
                img.src = firstPage.full;
            }
            
            hideLoadingOverlay();
            console.log(`PDF已导入: ${folder.name}，共${folder.pages.length}页`);
        } catch (error) {
            hideLoadingOverlay();
            console.error('PDF导入失败:', error);
            alert('PDF导入失败，请确保文件格式正确');
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

function collapseFileSidebar() {
    const fileSidebar = document.querySelector('.file-sidebar');
    if (fileSidebar) {
        fileSidebar.classList.add('collapse');
        fileSidebar.addEventListener('animationend', function() {
            fileSidebar.remove();
        }, { once: true });
    }
    
    dom.btnSave.innerHTML = `
        <img src="assets/javascript.svg" width="16" height="16" alt="文件" style="filter: invert(1);">
        文件
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
        });
        
        button.addEventListener('touchend', function(e) {
            this.style.transform = '';
        });
        
        button.addEventListener('touchcancel', function(e) {
            this.style.transform = '';
        });
    });
});

// ==================== 摄像头功能 ====================

/**
 * 打开/关闭摄像头
 * - 支持前置/后置摄像头切换
 * - 自动渲染到 imageCanvas
 */
async function openCamera() {
    if (state.isCameraOpen) {
        closeCamera();
        return;
    }
    
    try {
        const constraints = {
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: state.useFrontCamera ? 'user' : 'environment'
            },
            audio: false
        };
        
        state.cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
        state.isCameraOpen = true;
        state.isMirrored = state.useFrontCamera;
        
        createCameraVideo();
        createCameraControls();
        clearSidebarSelection();
        updateEnhanceButtonState();
        
        console.log('摄像头已打开');
    } catch (error) {
        console.error('无法访问摄像头:', error);
        alert('无法访问摄像头，请确保已授权摄像头权限');
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
        closeCamera();
        await openCamera();
    }
    
    console.log(state.useFrontCamera ? '已切换到前置摄像头' : '已切换到后置摄像头');
}

function createCameraVideo() {
    const video = document.createElement('video');
    video.id = 'cameraVideo';
    video.style.display = 'none';
    video.autoplay = true;
    video.playsInline = true;
    video.muted = true;
    document.body.appendChild(video);
    
    video.srcObject = state.cameraStream;
    video.play();
    
    video.onloadedmetadata = () => {
        startCameraPreview();
    };
}

function startCameraPreview() {
    const video = document.getElementById('cameraVideo');
    if (!video) return;
    
    let lastFrameTime = 0;
    let cachedDrawParams = null;
    
    function updateDrawParams() {
        const screenW = DRAW_CONFIG.screenW;
        const screenH = DRAW_CONFIG.screenH;
        const videoW = video.videoWidth;
        const videoH = video.videoHeight;
        
        if (!videoW || !videoH) return null;
        
        const canvasW = DRAW_CONFIG.canvasW;
        const canvasH = DRAW_CONFIG.canvasH;
        
        const rotation = state.cameraRotation;
        const isRotated = rotation === 90 || rotation === 270;
        
        const effectiveVideoW = isRotated ? videoH : videoW;
        const effectiveVideoH = isRotated ? videoW : videoH;
        
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
        
        const drawX = (canvasW - drawW) / 2;
        const drawY = (canvasH - drawH) / 2;
        
        return {
            canvasW, canvasH, drawW, drawH, drawX, drawY,
            centerX: canvasW / 2,
            centerY: canvasH / 2,
            rotation,
            isRotated
        };
    }
    
    function renderFrame(currentTime) {
        if (!state.isCameraOpen) return;
        
        // 在绘制时完全暂停摄像头处理，减少GPU占用
        if (state.drawMode === 'comment' || state.drawMode === 'eraser') {
            // 只在需要时更新（例如旋转或镜像变化）
            if (!cachedDrawParams || cachedDrawParams.rotation !== state.cameraRotation) {
                cachedDrawParams = updateDrawParams();
            }
            state.cameraAnimationId = requestAnimationFrame(renderFrame);
            return;
        }
        
        // 根据性能状态调整摄像头帧率
        const currentInterval = smartDrawScheduler ? 
            (smartDrawScheduler.getPerformanceState().status === 'poor' ? 
                DRAW_CONFIG.cameraFrameIntervalLow : 
                DRAW_CONFIG.cameraFrameInterval) : 
            DRAW_CONFIG.cameraFrameInterval;
        
        if (currentTime - lastFrameTime >= currentInterval) {
            lastFrameTime = currentTime;
            
            if (!cachedDrawParams || cachedDrawParams.rotation !== state.cameraRotation) {
                cachedDrawParams = updateDrawParams();
            }
            
            if (cachedDrawParams) {
                const { canvasW, canvasH, drawW, drawH, drawX, drawY, centerX, centerY, rotation, isRotated } = cachedDrawParams;
                
                // 只清除需要更新的区域，减少绘制操作
                dom.imageCtx.clearRect(drawX, drawY, drawW, drawH);
                
                dom.imageCtx.save();
                dom.imageCtx.translate(centerX, centerY);
                
                if (state.isMirrored) {
                    dom.imageCtx.scale(-1, 1);
                }
                
                dom.imageCtx.rotate(rotation * Math.PI / 180);
                
                // 降低摄像头画面的绘制质量，减少GPU负担
                const originalQuality = dom.imageCtx.imageSmoothingQuality;
                dom.imageCtx.imageSmoothingQuality = 'low';
                
                if (isRotated) {
                    dom.imageCtx.drawImage(video, -drawH / 2, -drawW / 2, drawH, drawW);
                } else {
                    dom.imageCtx.drawImage(video, -drawW / 2, -drawH / 2, drawW, drawH);
                }
                
                dom.imageCtx.imageSmoothingQuality = originalQuality;
                dom.imageCtx.restore();
            }
        }
        
        state.cameraAnimationId = requestAnimationFrame(renderFrame);
    }
    
    renderFrame(0);
}

function createCameraControls() {
    updatePhotoButtonState();
}

function closeCamera() {
    if (state.cameraAnimationId) {
        cancelAnimationFrame(state.cameraAnimationId);
        state.cameraAnimationId = null;
    }
    
    if (state.cameraStream) {
        state.cameraStream.getTracks().forEach(track => track.stop());
        state.cameraStream = null;
    }
    
    state.isCameraOpen = false;
    
    const video = document.getElementById('cameraVideo');
    if (video) {
        video.remove();
    }
    
    updatePhotoButtonState();
    updateEnhanceButtonState();
    
    if (state.currentImage && state.currentImageIndex >= 0) {
        drawImageToCenter(state.currentImage);
        restoreDrawData(state.currentImageIndex);
    } else if (state.currentImage && state.currentFolderIndex >= 0 && state.currentFolderPageIndex >= 0) {
        drawImageToCenter(state.currentImage);
        restoreFolderPageDrawData(state.currentFolderIndex, state.currentFolderPageIndex);
    } else {
        clearImageLayer();
    }
    
    console.log('摄像头已关闭');
}

/**
 * 拍照捕获
 * - 捕获当前摄像头画面
 * - 支持镜像和旋转
 * - 调用Rust保存图片（可选增强）
 */
async function captureCamera() {
    const video = document.getElementById('cameraVideo');
    if (!video) return;
    
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    const tempCtx = tempCanvas.getContext('2d');
    
    if (state.isMirrored) {
        tempCtx.translate(tempCanvas.width, 0);
        tempCtx.scale(-1, 1);
    }
    
    tempCtx.drawImage(video, 0, 0);
    
    if (state.isMirrored) {
        tempCtx.setTransform(1, 0, 0, 1, 0, 0);
    }
    
    let dataUrl = tempCanvas.toDataURL('image/png');
    
    if (window.__TAURI__) {
        try {
            const { invoke } = window.__TAURI__.core;
            
            if (state.enhanceEnabled) {
                const result = await invoke('save_image_with_enhance', { 
                    imageData: dataUrl,
                    prefix: 'photo'
                });
                console.log('图片已保存到:', result.path);
                if (result.enhanced_data) {
                    dataUrl = result.enhanced_data;
                }
            } else {
                const result = await invoke('save_image', { 
                    imageData: dataUrl,
                    prefix: 'photo'
                });
                console.log('图片已保存到:', result.path);
            }
        } catch (error) {
            console.error('保存图片失败:', error);
        }
    }
    
    const img = new Image();
    img.src = dataUrl;
    img.onload = () => {
        addImageToListNoHighlight(img, `拍摄${state.imageList.length + 1}`);
        expandSidebarIfCollapsed();
        console.log('已捕获摄像头画面并保存到图片列表');
    };
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
            closeCamera();
        }
        
        if (files.length > 1) {
            showLoadingOverlay(`正在读取图片...`);
        }
        
        const imageDataList = [];
        
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            
            if (files.length > 1) {
                updateLoadingProgress(`正在读取图片 ${i + 1}/${files.length}...`);
            }
            
            const dataUrl = await new Promise((resolve) => {
                const reader = new FileReader();
                reader.onload = (event) => resolve(event.target.result);
                reader.onerror = () => {
                    console.error(`读取图片失败: ${file.name}`);
                    resolve(null);
                };
                reader.readAsDataURL(file);
            });
            
            if (dataUrl) {
                imageDataList.push({
                    data: dataUrl,
                    name: file.name || `图片${state.imageList.length + imageDataList.length + 1}`
                });
            }
        }
        
        let thumbnails = [];
        
        if (window.__TAURI__ && imageDataList.length > 1) {
            try {
                updateLoadingProgress(`正在并行生成缩略图...`);
                const { invoke } = window.__TAURI__.core;
                thumbnails = await invoke('generate_thumbnails_batch', {
                    images: imageDataList,
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
                thumbnail = thumbnails[i];
            } else {
                thumbnail = await generateThumbnail(img.src, 150, false);
            }
            
            const newImgData = {
                full: img.src,
                thumbnail: thumbnail,
                name: imgData.name,
                width: img.width,
                height: img.height,
                strokeHistory: null,
                baseImageURL: null
            };
            
            state.imageList.push(newImgData);
            state.currentImageIndex = state.imageList.length - 1;
            state.currentImage = img;
            state.currentFolderIndex = -1;
            state.currentFolderPageIndex = -1;
            
            if (isLast) {
                clearDrawCanvas();
                state.strokeHistory = [];
                state.baseImageURL = null;
                state.baseImageObj = null;
                updateUndoBtnStatus();
                drawImageToCenter(img);
                updateSidebarContent();
                updatePhotoButtonState();
                updateEnhanceButtonState();
            }
        }
        
        if (files.length > 1) {
            hideLoadingOverlay();
        }
        
        console.log(`已导入 ${imageDataList.length} 张图片`);
    };
    
    input.click();
}

async function addImageToList(img, name, isLast = true) {
    const thumbnail = await generateThumbnail(img.src, 150);
    
    const imgData = {
        full: img.src,
        thumbnail: thumbnail,
        name: name,
        width: img.width,
        height: img.height,
        strokeHistory: null,
        baseImageURL: null
    };
    
    state.imageList.push(imgData);
    state.currentImageIndex = state.imageList.length - 1;
    state.currentImage = img;
    state.currentFolderIndex = -1;
    state.currentFolderPageIndex = -1;
    
    if (isLast) {
        clearDrawCanvas();
        state.strokeHistory = [];
        state.baseImageURL = null;
        state.baseImageObj = null;
        updateUndoBtnStatus();
        
        if (state.isCameraOpen) {
            closeCamera();
        } else {
            drawImageToCenter(img);
        }
        
        updateSidebarContent();
        updatePhotoButtonState();
        updateEnhanceButtonState();
    }
}

async function addImageToListNoHighlight(img, name) {
    const thumbnail = await generateThumbnail(img.src, 150);
    
    const imgData = {
        full: img.src,
        thumbnail: thumbnail,
        name: name,
        width: img.width,
        height: img.height,
        strokeHistory: null,
        baseImageURL: null
    };
    
    state.imageList.push(imgData);
    
    updateSidebarContent();
}

function drawImageToCenter(img) {
    clearImageLayer();
    
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
