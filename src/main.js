// PDF.js Worker配置
function initPdfJs() {
    if (window.pdfjsLib) {
        pdfjsLib.GlobalWorkerOptions.workerSrc = 'assets/pdf.worker.min.js';
        console.log('PDF.js Worker 已配置');
        return true;
    }
    console.warn('PDF.js 库未加载');
    return false;
}

// 等待 PDF.js 加载
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

// 确保在 DOM 加载后初始化 PDF.js
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initPdfJs);
} else {
    initPdfJs();
}

// Tauri API引用
const { invoke } = window.__TAURI__?.core || {};
const getCurrentWindow = window.__TAURI__?.window?.getCurrentWindow;

// 缓存目录路径
let cacheDir = null;
let configDir = null;
let cdsDir = null;

// 初始化缓存目录
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

// 全局配置
const DRAW_CONFIG = {
    penColor: '#3498db',
    penWidth: 2,
    eraserSize: 15,
    minScale: 0.5,
    maxScale: 5,
    canvasW: 1000,
    canvasH: 600,
    screenW: 0,
    screenH: 0,
    canvasScale: 2,
    dpr: window.devicePixelRatio || 1,
    cameraFrameInterval: 16,
    cameraFrameIntervalLow: 100
};

// 全局状态
let state = {
    drawMode: 'move',
    isDrawing: false,
    isDragging: false,
    isScaling: false,
    canvasX: 0,
    canvasY: 0,
    scale: 1,
    lastX: 0,
    lastY: 0,
    startDragX: 0,
    startDragY: 0,
    startScale: 1,
    startDistance: 0,
    startScaleX: 0,
    startScaleY: 0,
    startCanvasX: 0,
    startCanvasY: 0,
    historyStack: [],
    currentStep: -1,
    MAX_UNDO_STEPS: 15,
    moveBound: {
        minX: 0,
        maxX: 0,
        minY: 0,
        maxY: 0
    },
    cameraStream: null,
    isCameraOpen: false,
    isMirrored: false,
    cameraAnimationId: null,
    cameraRotation: 0,
    enhanceEnabled: false,
    currentImage: null,
    useFrontCamera: false,
    imageList: [],
    currentImageIndex: -1,
    fileList: [],
    currentFolderIndex: -1,
    currentFolderPageIndex: -1,
    enhanceRequestQueue: [],
    lastEnhanceTime: 0
};

// DOM 元素引用
let dom = {};

// 页面初始化
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

// 缓存数据保存和加载
async function saveCacheData(key, data) {
    if (!cacheDir || !window.__TAURI__) return;
    
    try {
        const { fs } = window.__TAURI__;
        const filePath = `${cacheDir}/${key}.json`;
        const jsonData = JSON.stringify(data);
        await fs.writeTextFile(filePath, jsonData);
        console.log(`缓存已保存: ${key}`);
    } catch (error) {
        console.error(`保存缓存失败: ${key}`, error);
    }
}

async function loadCacheData(key) {
    if (!cacheDir || !window.__TAURI__) return null;
    
    try {
        const { fs } = window.__TAURI__;
        const filePath = `${cacheDir}/${key}.json`;
        const jsonData = await fs.readTextFile(filePath);
        console.log(`缓存已加载: ${key}`);
        return JSON.parse(jsonData);
    } catch (error) {
        console.log(`加载缓存失败或不存在: ${key}`);
        return null;
    }
}

async function clearCacheData(key) {
    if (!cacheDir || !window.__TAURI__) return;
    
    try {
        const { fs } = window.__TAURI__;
        const filePath = `${cacheDir}/${key}.json`;
        await fs.removeFile(filePath);
        console.log(`缓存已清除: ${key}`);
    } catch (error) {
        console.log(`清除缓存失败或不存在: ${key}`);
    }
}

// 保存应用状态到缓存
async function saveAppState() {
    const appState = {
        imageList: state.imageList.map(img => ({
            ...img,
            drawData: null
        })),
        fileList: state.fileList.map(folder => ({
            ...folder,
            pages: folder.pages.map(page => ({
                ...page,
                drawData: null
            }))
        })),
        currentImageIndex: state.currentImageIndex,
        currentFolderIndex: state.currentFolderIndex,
        currentFolderPageIndex: state.currentFolderPageIndex
    };
    
    await saveCacheData('app_state', appState);
}

// 从缓存加载应用状态
async function loadAppState() {
    const appState = await loadCacheData('app_state');
    if (appState) {
        state.imageList = appState.imageList || [];
        state.fileList = appState.fileList || [];
        state.currentImageIndex = appState.currentImageIndex ?? -1;
        state.currentFolderIndex = appState.currentFolderIndex ?? -1;
        state.currentFolderPageIndex = appState.currentFolderPageIndex ?? -1;
        return true;
    }
    return false;
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
            alert('PDF库加载超时，请检查网络连接后重试');
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
        
        for (let i = 1; i <= totalPages; i++) {
            updateLoadingProgress(`正在处理第 ${i}/${totalPages} 页`);
            
            const page = await pdf.getPage(i);
            const viewport = page.getViewport({ scale: 1.5 });
            
            const canvas = document.createElement('canvas');
            canvas.width = viewport.width;
            canvas.height = viewport.height;
            const ctx = canvas.getContext('2d');
            
            await page.render({
                canvasContext: ctx,
                viewport: viewport
            }).promise;
            
            const maxSize = 150;
            let thumbW, thumbH;
            if (viewport.width > viewport.height) {
                thumbW = maxSize;
                thumbH = (viewport.height / viewport.width) * maxSize;
            } else {
                thumbH = maxSize;
                thumbW = (viewport.width / viewport.height) * maxSize;
            }
            
            const thumbCanvas = document.createElement('canvas');
            thumbCanvas.width = thumbW;
            thumbCanvas.height = thumbH;
            const thumbCtx = thumbCanvas.getContext('2d');
            thumbCtx.drawImage(canvas, 0, 0, thumbW, thumbH);
            
            folder.pages.push({
                full: canvas.toDataURL('image/png'),
                thumbnail: thumbCanvas.toDataURL('image/jpeg', 0.7),
                pageNum: i,
                drawData: null
            });
        }
        
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
    
    const bgImageData = dom.bgCtx.getImageData(0, 0, oldCanvasW * DRAW_CONFIG.dpr, oldCanvasH * DRAW_CONFIG.dpr);
    const imageImageData = dom.imageCtx.getImageData(0, 0, oldCanvasW * DRAW_CONFIG.dpr, oldCanvasH * DRAW_CONFIG.dpr);
    const drawImageData = dom.drawCtx.getImageData(0, 0, oldCanvasW * DRAW_CONFIG.dpr, oldCanvasH * DRAW_CONFIG.dpr);
    
    DRAW_CONFIG.screenW = newScreenW;
    DRAW_CONFIG.screenH = newScreenH;
    DRAW_CONFIG.canvasW = Math.floor(newScreenW * DRAW_CONFIG.canvasScale);
    DRAW_CONFIG.canvasH = Math.floor(newScreenH * DRAW_CONFIG.canvasScale);
    
    updateMoveBound();
    
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
    
    dom.bgCtx.setTransform(1, 0, 0, 1, 0, 0);
    dom.imageCtx.setTransform(1, 0, 0, 1, 0, 0);
    dom.drawCtx.setTransform(1, 0, 0, 1, 0, 0);
    dom.bgCtx.scale(DRAW_CONFIG.dpr, DRAW_CONFIG.dpr);
    dom.imageCtx.scale(DRAW_CONFIG.dpr, DRAW_CONFIG.dpr);
    dom.drawCtx.scale(DRAW_CONFIG.dpr, DRAW_CONFIG.dpr);
    
    dom.bgCtx.imageSmoothingEnabled = true;
    dom.bgCtx.imageSmoothingQuality = 'high';
    dom.imageCtx.imageSmoothingEnabled = true;
    dom.imageCtx.imageSmoothingQuality = 'high';
    dom.drawCtx.imageSmoothingEnabled = true;
    dom.drawCtx.imageSmoothingQuality = 'high';
    dom.drawCtx.lineCap = 'round';
    dom.drawCtx.lineJoin = 'round';
    dom.drawCtx.miterLimit = 10;
    
    resetBgCanvas();
    dom.bgCtx.putImageData(bgImageData, 0, 0);
    dom.imageCtx.putImageData(imageImageData, 0, 0);
    dom.drawCtx.putImageData(drawImageData, 0, 0);
    
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
    dom.imageCtx = dom.imageCanvas.getContext('2d');
    dom.drawCtx = dom.drawCanvas.getContext('2d');
}

// 初始化画布
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
    dom.imageCtx.imageSmoothingQuality = 'high';
    
    const dc = dom.drawCtx;
    dc.imageSmoothingEnabled = true;
    dc.imageSmoothingQuality = 'high';
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

// 画布鼠标事件
function bindCanvasMouseEvents() {
    dom.drawCanvas.addEventListener('mousedown', handleMouseDown);
    dom.drawCanvas.addEventListener('mousemove', handleMouseMove);
    dom.drawCanvas.addEventListener('mouseup', handleMouseUp);
    dom.drawCanvas.addEventListener('mouseleave', handleMouseLeave);
    dom.drawCanvas.addEventListener('wheel', handleWheel, { passive: false });
}

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
        dom.drawCtx.beginPath();
        dom.drawCtx.moveTo(state.lastX, state.lastY);
    } else if (state.drawMode === 'eraser') {
        state.isDrawing = true;
        state.lastX = (e.clientX - rect.left) / state.scale;
        state.lastY = (e.clientY - rect.top) / state.scale;
        dom.drawCtx.beginPath();
        dom.drawCtx.moveTo(state.lastX, state.lastY);
    }
}

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
        
        dom.drawCtx.beginPath();
        dom.drawCtx.moveTo(state.lastX, state.lastY);
        dom.drawCtx.lineTo(x, y);
        dom.drawCtx.stroke();
        
        state.lastX = x;
        state.lastY = y;
    }
}

function handleMouseUp(e) {
    if (state.isDragging) {
        state.isDragging = false;
        dom.drawCanvas.classList.remove('dragging');
    }
    if (state.isDrawing) {
        state.isDrawing = false;
        saveSnapshot();
    }
}

function handleMouseLeave(e) {
    if (state.isDragging) {
        state.isDragging = false;
        dom.drawCanvas.classList.remove('dragging');
    }
    if (state.isDrawing) {
        state.isDrawing = false;
        saveSnapshot();
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
            dom.drawCtx.beginPath();
            dom.drawCtx.moveTo(state.lastX, state.lastY);
        } else if (state.drawMode === 'eraser') {
            state.isDrawing = true;
            updateEraserHintPos(touch.clientX, touch.clientY);
            state.lastX = (touch.clientX - rect.left) / state.scale;
            state.lastY = (touch.clientY - rect.top) / state.scale;
            dom.drawCtx.beginPath();
            dom.drawCtx.moveTo(state.lastX, state.lastY);
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
        
        dom.drawCtx.beginPath();
        dom.drawCtx.moveTo(state.lastX, state.lastY);
        dom.drawCtx.lineTo(x, y);
        dom.drawCtx.stroke();
        
        state.lastX = x;
        state.lastY = y;
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
            saveSnapshot();
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

// 撤销功能
function saveSnapshot() {
    const snapshot = dom.drawCanvas.toDataURL('image/png');
    if (state.currentStep < state.historyStack.length - 1) {
        state.historyStack = state.historyStack.slice(0, state.currentStep + 1);
    }
    state.historyStack.push(snapshot);
    state.currentStep++;
    if (state.historyStack.length > state.MAX_UNDO_STEPS) {
        state.historyStack.shift();
        state.currentStep--;
    }
    updateUndoBtnStatus();
}

function restoreSnapshot(step) {
    if (step < 0 || step >= state.historyStack.length) return;
    const img = new Image();
    img.src = state.historyStack[step];
    img.onload = () => {
        clearDrawCanvas();
        dom.drawCtx.drawImage(img, 0, 0, DRAW_CONFIG.canvasW, DRAW_CONFIG.canvasH);
        state.currentStep = step;
        updateUndoBtnStatus();
    };
}

function undo() {
    if (state.currentStep <= 0) return;
    restoreSnapshot(state.currentStep - 1);
    console.log('撤销操作');
}

function updateUndoBtnStatus() {
    dom.btnUndo.disabled = state.currentStep <= 0;
}

// 清空画布
function clearDrawCanvas() {
    dom.drawCtx.clearRect(0, 0, DRAW_CONFIG.canvasW, DRAW_CONFIG.canvasH);
    dom.drawCtx.globalCompositeOperation = 'source-over';
}

function clearAllDrawings() {
    clearDrawCanvas();
    
    if (state.currentImageIndex >= 0 && state.currentImageIndex < state.imageList.length) {
        state.imageList[state.currentImageIndex].drawData = null;
    }
    
    if (state.currentFolderIndex >= 0 && state.currentFolderPageIndex >= 0) {
        if (state.currentFolderIndex < state.fileList.length) {
            const folder = state.fileList[state.currentFolderIndex];
            if (state.currentFolderPageIndex < folder.pages.length) {
                folder.pages[state.currentFolderPageIndex].drawData = null;
            }
        }
    }
    
    saveSnapshot();
    
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

function rotateImage(direction) {
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
    
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    if (direction === 'left') {
        canvas.width = state.currentImage.height;
        canvas.height = state.currentImage.width;
        ctx.translate(canvas.width / 2, canvas.height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.drawImage(state.currentImage, -state.currentImage.width / 2, -state.currentImage.height / 2);
    } else {
        canvas.width = state.currentImage.height;
        canvas.height = state.currentImage.width;
        ctx.translate(canvas.width / 2, canvas.height / 2);
        ctx.rotate(Math.PI / 2);
        ctx.drawImage(state.currentImage, -state.currentImage.width / 2, -state.currentImage.height / 2);
    }
    
    const rotatedImg = new Image();
    rotatedImg.onload = () => {
        state.currentImage = rotatedImg;
        
        if (state.currentImageIndex >= 0 && state.currentImageIndex < state.imageList.length) {
            const maxSize = 150;
            let thumbW, thumbH;
            if (rotatedImg.width > rotatedImg.height) {
                thumbW = maxSize;
                thumbH = (rotatedImg.height / rotatedImg.width) * maxSize;
            } else {
                thumbH = maxSize;
                thumbW = (rotatedImg.width / rotatedImg.height) * maxSize;
            }
            
            const thumbCanvas = document.createElement('canvas');
            thumbCanvas.width = thumbW;
            thumbCanvas.height = thumbH;
            const thumbCtx = thumbCanvas.getContext('2d');
            thumbCtx.drawImage(rotatedImg, 0, 0, thumbW, thumbH);
            
            state.imageList[state.currentImageIndex].full = rotatedImg.src;
            state.imageList[state.currentImageIndex].thumbnail = thumbCanvas.toDataURL('image/jpeg', 0.7);
            state.imageList[state.currentImageIndex].width = rotatedImg.width;
            state.imageList[state.currentImageIndex].height = rotatedImg.height;
            
            updateSidebarContent();
        }
        
        drawImageToCenter(rotatedImg);
        console.log(`图片已向${direction === 'left' ? '左' : '右'}旋转`);
    };
    rotatedImg.src = canvas.toDataURL('image/png');
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

function applyEnhanceFilter(imageData) {
    if (!state.enhanceEnabled) return imageData;
    
    const data = imageData.data;
    
    for (let i = 0; i < data.length; i += 4) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        
        const contrast = 1.4;
        const brightness = 10;
        
        let newR = ((r - 128) * contrast) + 128 + brightness;
        let newG = ((g - 128) * contrast) + 128 + brightness;
        let newB = ((b - 128) * contrast) + 128 + brightness;
        
        const saturation = 1.2;
        const gray = 0.299 * newR + 0.587 * newG + 0.114 * newB;
        newR = gray + (newR - gray) * saturation;
        newG = gray + (newG - gray) * saturation;
        newB = gray + (newB - gray) * saturation;
        
        data[i] = Math.max(0, Math.min(255, newR));
        data[i + 1] = Math.max(0, Math.min(255, newG));
        data[i + 2] = Math.max(0, Math.min(255, newB));
    }
    
    return imageData;
}

function enhanceCanvas(sourceCanvas) {
    if (!state.enhanceEnabled) return sourceCanvas;
    
    const canvas = document.createElement('canvas');
    canvas.width = sourceCanvas.width;
    canvas.height = sourceCanvas.height;
    const ctx = canvas.getContext('2d');
    
    ctx.drawImage(sourceCanvas, 0, 0);
    
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const enhancedData = applyEnhanceFilter(imageData);
    ctx.putImageData(enhancedData, 0, 0);
    
    return canvas;
}

// 保存画布
function saveCanvas() {
    console.log('保存画布');
    const mergedCanvas = document.createElement('canvas');
    mergedCanvas.width = DRAW_CONFIG.canvasW * DRAW_CONFIG.dpr;
    mergedCanvas.height = DRAW_CONFIG.canvasH * DRAW_CONFIG.dpr;
    const mergedCtx = mergedCanvas.getContext('2d');
    mergedCtx.scale(DRAW_CONFIG.dpr, DRAW_CONFIG.dpr);
    
    mergedCtx.drawImage(dom.bgCanvas, 0, 0, DRAW_CONFIG.canvasW, DRAW_CONFIG.canvasH);
    mergedCtx.drawImage(dom.imageCanvas, 0, 0, DRAW_CONFIG.canvasW, DRAW_CONFIG.canvasH);
    mergedCtx.drawImage(dom.drawCanvas, 0, 0, DRAW_CONFIG.canvasW, DRAW_CONFIG.canvasH);
    
    const link = document.createElement('a');
    link.download = `canvas_${Date.now()}.png`;
    link.href = mergedCanvas.toDataURL('image/png');
    link.click();
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
        state.currentImageIndex = -1;
        state.currentImage = null;
        clearImageLayer();
        clearDrawCanvas();
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
        const drawData = dom.drawCtx.getImageData(
            0, 0, 
            DRAW_CONFIG.canvasW * DRAW_CONFIG.dpr, 
            DRAW_CONFIG.canvasH * DRAW_CONFIG.dpr
        );
        state.imageList[state.currentImageIndex].drawData = drawData;
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
        if (imgData.drawData) {
            dom.drawCtx.putImageData(imgData.drawData, 0, 0);
        } else {
            clearDrawCanvas();
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
                const drawData = dom.drawCtx.getImageData(
                    0, 0,
                    DRAW_CONFIG.canvasW * DRAW_CONFIG.dpr,
                    DRAW_CONFIG.canvasH * DRAW_CONFIG.dpr
                );
                folder.pages[state.currentFolderPageIndex].drawData = drawData;
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
            if (page.drawData) {
                dom.drawCtx.putImageData(page.drawData, 0, 0);
            } else {
                clearDrawCanvas();
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
                alert('PDF库加载超时，请检查网络连接后重试');
                return;
            }
            
            const arrayBuffer = await file.arrayBuffer();
            const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
            
            const totalPages = pdf.numPages;
            const folder = {
                name: file.name.replace('.pdf', ''),
                pages: []
            };
            
            for (let i = 1; i <= totalPages; i++) {
                updateLoadingProgress(`正在处理第 ${i}/${totalPages} 页`);
                
                const page = await pdf.getPage(i);
                const viewport = page.getViewport({ scale: 1.5 });
                
                const canvas = document.createElement('canvas');
                canvas.width = viewport.width;
                canvas.height = viewport.height;
                const ctx = canvas.getContext('2d');
                
                await page.render({
                    canvasContext: ctx,
                    viewport: viewport
                }).promise;
                
                const maxSize = 150;
                let thumbW, thumbH;
                if (viewport.width > viewport.height) {
                    thumbW = maxSize;
                    thumbH = (viewport.height / viewport.width) * maxSize;
                } else {
                    thumbH = maxSize;
                    thumbW = (viewport.width / viewport.height) * maxSize;
                }
                
                const thumbCanvas = document.createElement('canvas');
                thumbCanvas.width = thumbW;
                thumbCanvas.height = thumbH;
                const thumbCtx = thumbCanvas.getContext('2d');
                thumbCtx.drawImage(canvas, 0, 0, thumbW, thumbH);
                
                folder.pages.push({
                    full: canvas.toDataURL('image/png'),
                    thumbnail: thumbCanvas.toDataURL('image/jpeg', 0.7),
                    pageNum: i,
                    drawData: null
                });
            }
            
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

// 摄像头功能
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
    const frameInterval = DRAW_CONFIG.cameraFrameInterval;
    
    function renderFrame(currentTime) {
        if (!state.isCameraOpen) return;
        
        const currentInterval = state.drawMode === 'comment' || state.drawMode === 'eraser' 
            ? DRAW_CONFIG.cameraFrameIntervalLow 
            : DRAW_CONFIG.cameraFrameInterval;
        
        if (currentTime - lastFrameTime >= currentInterval) {
            lastFrameTime = currentTime;
            
            clearImageLayer();
            
            const screenW = DRAW_CONFIG.screenW;
            const screenH = DRAW_CONFIG.screenH;
            const videoW = video.videoWidth;
            const videoH = video.videoHeight;
            
            if (videoW && videoH) {
                const canvasW = DRAW_CONFIG.canvasW;
                const canvasH = DRAW_CONFIG.canvasH;
                
                const rotation = state.cameraRotation;
                const isRotated = rotation === 90 || rotation === 270;
                
                let effectiveVideoW = isRotated ? videoH : videoW;
                let effectiveVideoH = isRotated ? videoW : videoH;
                
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
                
                dom.imageCtx.save();
                dom.imageCtx.translate(canvasW / 2, canvasH / 2);
                
                if (state.isMirrored) {
                    dom.imageCtx.scale(-1, 1);
                }
                
                dom.imageCtx.rotate(rotation * Math.PI / 180);
                
                if (isRotated) {
                    dom.imageCtx.drawImage(video, -drawH / 2, -drawW / 2, drawH, drawW);
                } else {
                    dom.imageCtx.drawImage(video, -drawW / 2, -drawH / 2, drawW, drawH);
                }
                
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
    } else {
        clearImageLayer();
    }
    
    console.log('摄像头已关闭');
}

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
    
    if (state.enhanceEnabled && window.__TAURI__) {
        const currentTime = Date.now();
        const timeSinceLastEnhance = currentTime - state.lastEnhanceTime;
        const shouldShowLoading = timeSinceLastEnhance < 2000 || state.enhanceRequestQueue.length > 0;
        
        const requestId = Date.now() + Math.random();
        state.enhanceRequestQueue.push(requestId);
        state.lastEnhanceTime = currentTime;
        
        if (shouldShowLoading) {
            showLoadingOverlay('正在处理图像...');
        }
        
        try {
            const { invoke } = window.__TAURI__.core;
            dataUrl = await invoke('enhance_image', { imageData: dataUrl });
            console.log('Rust 图像增强完成');
        } catch (error) {
            console.error('Rust 图像增强失败，使用前端降级方案:', error);
            const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
            const enhancedData = applyEnhanceFilter(imageData);
            tempCtx.putImageData(enhancedData, 0, 0);
            dataUrl = tempCanvas.toDataURL('image/png');
        } finally {
            const index = state.enhanceRequestQueue.indexOf(requestId);
            if (index > -1) {
                state.enhanceRequestQueue.splice(index, 1);
            }
            
            if (state.enhanceRequestQueue.length === 0) {
                hideLoadingOverlay();
            }
        }
    } else if (state.enhanceEnabled) {
        const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
        const enhancedData = applyEnhanceFilter(imageData);
        tempCtx.putImageData(enhancedData, 0, 0);
        dataUrl = tempCanvas.toDataURL('image/png');
    }
    
    if (window.__TAURI__ && cdsDir) {
        try {
            const { fs } = window.__TAURI__;
            const now = new Date();
            const dateStr = now.toISOString().slice(0, 10);
            const timeStr = now.toTimeString().slice(0, 8).replace(/:/g, '-');
            
            const dateDir = `${cdsDir}/${dateStr}`;
            
            try {
                await fs.mkdir(dateDir, { recursive: true });
            } catch (e) {
            }
            
            const fileName = `photo_${timeStr}.png`;
            const filePath = `${dateDir}/${fileName}`;
            
            const base64Data = dataUrl.split(',')[1];
            const binaryString = atob(base64Data);
            const bytes = new Uint8Array(binaryString.length);
            for (let i = 0; i < binaryString.length; i++) {
                bytes[i] = binaryString.charCodeAt(i);
            }
            
            await fs.writeFile(filePath, bytes);
            console.log('图片已保存到:', filePath);
        } catch (error) {
            console.error('保存图片失败:', error);
        }
    }
    
    const img = new Image();
    img.src = dataUrl;
    img.onload = () => {
        addImageToListNoHighlight(img, `拍摄${state.imageList.length + 1}`);
        expandSidebarIfCollapsed();
        console.log('已捕获摄像头画面到图像层');
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

function toggleMirror() {
    state.isMirrored = !state.isMirrored;
    console.log(state.isMirrored ? '已启用镜像' : '已取消镜像');
}

// 图像层功能
function importImage() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    
    input.onchange = (e) => {
        const file = e.target.files[0];
        if (!file) return;
        
        const reader = new FileReader();
        reader.onload = (event) => {
            const img = new Image();
            img.onload = () => {
                addImageToList(img, file.name || `图片${state.imageList.length + 1}`);
                console.log('图像已导入');
            };
            img.src = event.target.result;
        };
        reader.readAsDataURL(file);
    };
    
    input.click();
}

function addImageToList(img, name) {
    const maxSize = 150;
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
    
    const drawCanvas = document.createElement('canvas');
    drawCanvas.width = DRAW_CONFIG.canvasW * DRAW_CONFIG.dpr;
    drawCanvas.height = DRAW_CONFIG.canvasH * DRAW_CONFIG.dpr;
    
    const imgData = {
        full: img.src,
        thumbnail: thumbCanvas.toDataURL('image/jpeg', 0.7),
        name: name,
        width: img.width,
        height: img.height,
        drawData: null
    };
    
    state.imageList.push(imgData);
    state.currentImageIndex = state.imageList.length - 1;
    state.currentImage = img;
    
    if (!state.isCameraOpen) {
        drawImageToCenter(img);
    }
    
    updateSidebarContent();
}

function addImageToListNoHighlight(img, name) {
    const maxSize = 150;
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
    
    const imgData = {
        full: img.src,
        thumbnail: thumbCanvas.toDataURL('image/jpeg', 0.7),
        name: name,
        width: img.width,
        height: img.height,
        drawData: null
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

function removeCurrentImage() {
    state.currentImage = null;
    clearImageLayer();
    console.log('图像已移除');
}
