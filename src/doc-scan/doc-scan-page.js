/**
 * 文档扫描增强独立页面
 * 使用 EAST 文本检测自动裁剪文档
 */

const state = {
    currentImage: null,
    enhancedImage: null,
    cameraStream: null,
    isCameraOpen: false,
    isProcessing: false,
    scannedImages: [],
    selectedImageIndex: -1,
    isViewingImage: false
};

const dom = {};

document.addEventListener('DOMContentLoaded', init);

async function init() {
    initDOM();
    initEventListeners();
    await loadThemeColor();
    listenThemeChange();
    startCameraPreview();
}

function initDOM() {
    dom.btnCapture = document.getElementById('btnCapture');
    dom.btnBack = document.getElementById('btnBack');
    
    dom.previewVideo = document.getElementById('previewVideo');
    dom.previewCanvas = document.getElementById('previewCanvas');
    dom.processingOverlay = document.getElementById('processingOverlay');
    
    dom.sidebarContent = document.getElementById('sidebarContent');
    dom.imageCount = document.getElementById('imageCount');
}

function initEventListeners() {
    dom.btnCapture.addEventListener('click', handleCaptureButtonClick);
    dom.btnBack.addEventListener('click', closeWindow);
}

function handleCaptureButtonClick() {
    if (state.isViewingImage) {
        returnToCamera();
    } else {
        captureAndEnhance();
    }
}

function returnToCamera() {
    state.isViewingImage = false;
    state.selectedImageIndex = -1;
    
    dom.previewVideo.style.display = 'block';
    dom.previewCanvas.style.display = 'none';
    
    dom.btnCapture.textContent = '拍摄';
    dom.btnCapture.classList.add('primary');
    
    updateSidebarContent();
}

async function loadThemeColor() {
    try {
        if (window.__TAURI__) {
            const { invoke } = window.__TAURI__.core;
            const settings = await invoke('get_settings');
            const themeName = settings?.theme || 'simplify';
            
            const themeModule = await import(`../themes/${themeName}/theme.js`);
            const theme = themeModule.default;
            await theme.load?.();
            
            const bgColor = theme.getCanvasBgColor?.() || '#1a1a1a';
            document.documentElement.style.setProperty('--doc-scan-bg-color', bgColor);
            document.documentElement.style.setProperty('--doc-scan-preview-bg', bgColor);
        }
    } catch (error) {
        console.error('加载主题颜色失败:', error);
    }
}

function listenThemeChange() {
    if (window.__TAURI__) {
        const { listen } = window.__TAURI__.event;
        listen('settings-changed', async (event) => {
            const settings = event.payload;
            if (settings?.theme) {
                try {
                    const themeModule = await import(`../themes/${settings.theme}/theme.js`);
                    const theme = themeModule.default;
                    const bgColor = theme.getCanvasBgColor?.() || '#1a1a1a';
                    document.documentElement.style.setProperty('--doc-scan-bg-color', bgColor);
                    document.documentElement.style.setProperty('--doc-scan-preview-bg', bgColor);
                } catch (error) {
                    console.error('更新主题颜色失败:', error);
                }
            }
        }).catch(err => console.error('主题变化监听失败:', err));
    }
}

async function closeWindow() {
    if (state.cameraStream) {
        state.cameraStream.getTracks().forEach(track => track.stop());
    }
    
    if (window.__TAURI__) {
        const { getCurrentWindow } = window.__TAURI__.window;
        const win = getCurrentWindow();
        await win.close();
    }
}

async function startCameraPreview() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment' }
        });
        
        state.cameraStream = stream;
        state.isCameraOpen = true;
        
        dom.previewVideo.srcObject = stream;
    } catch (error) {
        console.error('打开摄像头失败:', error);
    }
}

async function captureAndEnhance() {
    if (state.isProcessing) return;
    if (!state.isCameraOpen) {
        await startCameraPreview();
        return;
    }
    
    state.isProcessing = true;
    dom.btnCapture.disabled = true;
    dom.processingOverlay.style.display = 'flex';
    
    try {
        console.log('开始捕获图像...');
        const video = dom.previewVideo;
        const canvas = dom.previewCanvas;
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0);
        
        state.currentImage = canvas.toDataURL('image/png');
        console.log('图像已捕获，大小:', state.currentImage.length);
        
        console.log('调用 scan_document...');
        const result = await invokeDocumentScan(state.currentImage);
        console.log('scan_document 结果:', result);
        
        state.enhancedImage = result.enhanced_image;
        
        await addToLocalList(state.enhancedImage);
        
        await saveToMainApp();
        
    } catch (error) {
        console.error('处理失败:', error);
        alert('处理失败: ' + error.message);
    } finally {
        state.isProcessing = false;
        dom.btnCapture.disabled = false;
        dom.processingOverlay.style.display = 'none';
    }
}

async function invokeDocumentScan(imageData) {
    if (!window.__TAURI__) {
        throw new Error('需要Tauri环境');
    }
    
    const { invoke } = window.__TAURI__.core;
    
    const request = {
        image_data: imageData
    };
    
    return await invoke('scan_document', { request });
}

async function addToLocalList(imageData) {
    const thumbnail = await generateThumbnail(imageData, 100);
    
    const imgData = {
        full: imageData,
        thumbnail: thumbnail,
        name: `扫描文档${Date.now()}`,
        timestamp: Date.now()
    };
    
    state.scannedImages.push(imgData);
    
    updateSidebarContent();
}

async function generateThumbnail(dataUrl, size) {
    return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => {
            const canvas = document.createElement('canvas');
            const aspectRatio = img.width / img.height;
            
            if (aspectRatio > 1) {
                canvas.width = size;
                canvas.height = size / aspectRatio;
            } else {
                canvas.height = size;
                canvas.width = size * aspectRatio;
            }
            
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            
            resolve(canvas.toDataURL('image/jpeg', 0.7));
        };
        img.src = dataUrl;
    });
}

function updateSidebarContent() {
    if (!dom.sidebarContent) return;
    
    dom.imageCount.textContent = state.scannedImages.length;
    
    if (state.scannedImages.length === 0) {
        dom.sidebarContent.innerHTML = '<div class="sidebar-empty">暂无图片</div>';
        return;
    }
    
    let html = '';
    state.scannedImages.forEach((imgData, index) => {
        const isActive = index === state.selectedImageIndex ? 'active' : '';
        html += `
            <div class="sidebar-image-item ${isActive}" data-index="${index}">
                <img src="${imgData.thumbnail}" class="sidebar-thumbnail" alt="扫描图片${index + 1}">
                <div class="sidebar-image-actions">
                    <button class="sidebar-btn-delete" title="删除">✕</button>
                </div>
            </div>
        `;
    });
    
    dom.sidebarContent.innerHTML = html;
    
    document.querySelectorAll('.sidebar-image-item').forEach(item => {
        const index = parseInt(item.dataset.index);
        
        item.querySelector('.sidebar-btn-delete')?.addEventListener('click', (e) => {
            e.stopPropagation();
            deleteImage(index);
        });
        
        item.addEventListener('click', () => selectImage(index));
    });
}

function selectImage(index) {
    if (index < 0 || index >= state.scannedImages.length) return;
    
    if (state.selectedImageIndex === index && state.isViewingImage) {
        returnToCamera();
        return;
    }
    
    state.selectedImageIndex = index;
    state.isViewingImage = true;
    const imgData = state.scannedImages[index];
    
    const canvas = dom.previewCanvas;
    const ctx = canvas.getContext('2d');
    const img = new Image();
    img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
    };
    img.src = imgData.full;
    
    dom.previewVideo.style.display = 'none';
    canvas.style.display = 'block';
    
    dom.btnCapture.textContent = '返回摄像头';
    dom.btnCapture.classList.remove('primary');
    
    updateSidebarContent();
}

function deleteImage(index) {
    if (index < 0 || index >= state.scannedImages.length) return;
    
    state.scannedImages.splice(index, 1);
    
    if (state.selectedImageIndex === index) {
        if (state.scannedImages.length > 0) {
            state.selectedImageIndex = Math.min(index, state.scannedImages.length - 1);
            selectImage(state.selectedImageIndex);
        } else {
            returnToCamera();
        }
    } else if (state.selectedImageIndex > index) {
        state.selectedImageIndex--;
        updateSidebarContent();
    } else {
        updateSidebarContent();
    }
}

async function saveToMainApp() {
    if (!state.enhancedImage) return;
    
    try {
        if (window.__TAURI__) {
            const { emit } = window.__TAURI__.event;
            
            const photoName = `扫描文档${Date.now()}`;
            
            await emit('doc-scan-save-image', {
                imageData: state.enhancedImage,
                name: photoName
            });
        }
    } catch (error) {
        console.error('保存到主应用失败:', error);
    }
}

console.log('文档扫描页面已加载');
