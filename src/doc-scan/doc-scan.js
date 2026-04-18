/**
 * 文档扫描模块（主页面嵌入面板）
 * 使用 EAST 文本检测自动裁剪文档
 */

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initDocScanEvents);
} else {
    initDocScanEvents();
}

function initDocScanEvents() {
}

function toggleDocScanPanel() {
    const panel = window.dom?.docScanPanel;
    if (!panel) return;
    
    if (panel.classList.contains('visible')) {
        hideDocScanPanel();
    } else {
        showDocScanPanel();
    }
}

function showDocScanPanel() {
    if (!window.dom?.docScanPanel) return;
    
    if (window.hidePenControlPanel) window.hidePenControlPanel();
    if (window.hideSettingsPanel) window.hideSettingsPanel();
    
    window.dom.docScanPanel.classList.add('visible');
}

function hideDocScanPanel() {
    if (!window.dom?.docScanPanel) return;
    window.dom.docScanPanel.classList.remove('visible');
}

async function applyDocumentScan() {
    const hasCamera = window.state?.isCameraOpen;
    const hasImage = window.state?.currentImage;
    const hasPdfPage = window.state?.currentFolderIndex >= 0 && window.state?.currentFolderPageIndex >= 0;
    const hasImageIndex = window.state?.currentImageIndex >= 0;
    
    console.log('文档扫描状态检查:', {
        hasCamera,
        hasImage: !!hasImage,
        hasPdfPage,
        hasImageIndex
    });
    
    if (!hasCamera && !hasImage && !hasPdfPage && !hasImageIndex) {
        alert(window.i18n?.t('docScan.noImage') || '请先打开图片、文档或摄像头');
        return;
    }
    
    try {
        showProcessingIndicator();
        
        const imageData = await getScanImageData();
        
        const result = await invokeDocumentScan(imageData);
        
        await applyScanResult(result);
        
        hideDocScanPanel();
        
        console.log('文档扫描完成，置信度:', result.confidence);
    } catch (error) {
        console.error('文档扫描失败:', error);
        alert(window.i18n?.t('docScan.failed') || '文档扫描失败: ' + error.message);
    } finally {
        hideProcessingIndicator();
    }
}

async function getScanImageData() {
    if (window.state?.isCameraOpen) {
        const video = document.getElementById('cameraVideo');
        if (!video) throw new Error('摄像头未找到');
        
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = video.videoWidth;
        tempCanvas.height = video.videoHeight;
        const ctx = tempCanvas.getContext('2d');
        ctx.drawImage(video, 0, 0);
        
        return tempCanvas.toDataURL('image/png');
    }
    
    if (window.state?.currentImage) {
        return window.state.currentImage.src;
    }
    
    const imageElement = document.getElementById('imageElement');
    if (imageElement && imageElement.src) {
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = parseInt(imageElement.style.width) || imageElement.naturalWidth;
        tempCanvas.height = parseInt(imageElement.style.height) || imageElement.naturalHeight;
        const ctx = tempCanvas.getContext('2d');
        ctx.drawImage(imageElement, 0, 0, tempCanvas.width, tempCanvas.height);
        return tempCanvas.toDataURL('image/png');
    }
    
    throw new Error('没有可用的图像');
}

async function invokeDocumentScan(imageData) {
    if (!window.__TAURI__) {
        throw new Error('文档扫描功能需要Tauri后端支持');
    }
    
    const { invoke } = window.__TAURI__.core;
    
    const request = {
        image_data: imageData
    };
    
    return await invoke('scan_document', { request });
}

async function applyScanResult(result) {
    if (!result.enhanced_image) {
        throw new Error('扫描结果无效');
    }
    
    const enhancedImg = new Image();
    
    await new Promise((resolve, reject) => {
        enhancedImg.onload = resolve;
        enhancedImg.onerror = () => reject(new Error('加载增强图像失败'));
        enhancedImg.src = result.enhanced_image;
    });
    
    if (window.state?.isCameraOpen) {
        if (window.addImageToListNoHighlight) {
            const photoName = window.i18n?.t('docScan.scannedDoc') || `扫描文档${window.state.imageList.length + 1}`;
            await window.addImageToListNoHighlight(enhancedImg, photoName);
        }
    } else if (window.state?.currentImageIndex >= 0) {
        window.state.currentImage = enhancedImg;
        
        if (window.updateImageDisplay) {
            await window.updateImageDisplay();
        }
        
        if (window.state.imageList && window.state.currentImageIndex < window.state.imageList.length) {
            window.state.imageList[window.state.currentImageIndex].full = result.enhanced_image;
            window.state.imageList[window.state.currentImageIndex].thumbnail = result.enhanced_image;
            
            if (window.updateSidebarContent) {
                window.updateSidebarContent();
            }
        }
        
        if (window.clearAllDrawings) {
            window.clearAllDrawings();
        }
    }
    
    if (result.text_bbox) {
        console.log('检测到文本区域:', result.text_bbox);
    }
}

function showProcessingIndicator() {
    const btn = document.getElementById('btnApplyScan');
    if (btn) {
        btn.disabled = true;
        btn.textContent = window.i18n?.t('docScan.processing') || '处理中...';
    }
}

function hideProcessingIndicator() {
    const btn = document.getElementById('btnApplyScan');
    if (btn) {
        btn.disabled = false;
        btn.textContent = window.i18n?.t('docScan.apply') || '应用';
    }
}

window.toggleDocScanPanel = toggleDocScanPanel;
window.showDocScanPanel = showDocScanPanel;
window.hideDocScanPanel = hideDocScanPanel;
window.applyDocumentScan = applyDocumentScan;

console.log('文档扫描模块已加载');
