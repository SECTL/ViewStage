export class CameraManager {
    constructor(deps) {
        this.d = deps;
        this._lastVideoStyleCache = {
            drawW: 0, drawH: 0, offsetX: 0, offsetY: 0,
            rotation: -1, isMirrored: null
        };
        this._deferred_camera_setup = false;
    }

    // === 摄像头开启/关闭 ===

    async open() {
        const { state } = this.d;
        if (state.isCameraOpen) return;

        this.d.saveCurrentSourceData();

        // 如果没有保存的分辨率，探测摄像头能力以获取最高分辨率
        let maxW = state.cameraWidth;
        let maxH = state.cameraHeight;

        // 确定目标设备 ID
        let targetDeviceId = state.defaultCameraId;
        if (!targetDeviceId) {
            try {
                let devices = await navigator.mediaDevices.enumerateDevices();
                let videoDevices = devices.filter(d => d.kind === 'videoinput');
                if (videoDevices.length > 0 && !videoDevices[0].label) {
                    const temp = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
                    temp.getTracks().forEach(t => t.stop());
                    devices = await navigator.mediaDevices.enumerateDevices();
                    videoDevices = devices.filter(d => d.kind === 'videoinput');
                }
                for (const d of videoDevices) {
                    if (d.label.toLowerCase().includes('seewo')) {
                        targetDeviceId = d.deviceId;
                        break;
                    }
                }
            } catch (_) {}
        }

        // 探测目标摄像头的最高分辨率
        if (!maxW && targetDeviceId) {
            try {
                const probe = await navigator.mediaDevices.getUserMedia({
                    video: { deviceId: { exact: targetDeviceId } },
                    audio: false
                });
                const track = probe.getVideoTracks()[0];
                const caps = track.getCapabilities();
                probe.getTracks().forEach(t => t.stop());
                if (caps.width?.max) maxW = caps.width.max;
                if (caps.height?.max) maxH = caps.height.max;
            } catch (_) {}
        }

        let constraints;
        if (targetDeviceId) {
            const vc = { deviceId: { exact: targetDeviceId } };
            if (maxW) vc.width = { ideal: maxW };
            if (maxH) vc.height = { ideal: maxH };
            constraints = { video: vc, audio: false };
        } else if (maxW && maxH) {
            constraints = {
                video: {
                    width: { ideal: maxW },
                    height: { ideal: maxH },
                    facingMode: state.useFrontCamera ? 'user' : 'environment'
                },
                audio: false
            };
        } else {
            constraints = {
                video: { facingMode: state.useFrontCamera ? 'user' : 'environment' },
                audio: false
            };
        }

        try {
            state.cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
        } catch (constraintError) {
            if (constraintError.name === 'OverconstrainedError') {
                console.warn('指定摄像头不可用，使用默认摄像头');
                state.cameraStream = await navigator.mediaDevices.getUserMedia({
                    video: true,
                    audio: false
                });
            } else {
                throw constraintError;
            }
        }

        if (targetDeviceId && !state.defaultCameraId) {
            state.defaultCameraId = targetDeviceId;
        }
        state.isCameraOpen = true;
        state.cameraAvailable = true;
        this.d.updateSettingsControlsState();
        await this.d.updateSource('cam');

        state.currentImageIndex = -1;
        state.currentFolderIndex = -1;
        state.currentFolderPageIndex = -1;

        this.hideNoCameraMessage();

        const videoTrack = state.cameraStream.getVideoTracks()[0];
        const settings = videoTrack.getSettings();
        const label = videoTrack.label.toLowerCase();
        state.isMirrored = label.includes('front') || label.includes('user') || label.includes('前置') || settings.facingMode === 'user';

        this._createVideo();
        this.d.updatePhotoButtonState();
        this.d.deleteSidebarSelection();
        if (window.__lightAvailable && this.d.dom.btnLight) {
            this.d.dom.btnLight.style.display = '';
            if (window.__savedLightState) {
                try {
                    window.__TAURI__.core.invoke('camera_light_on');
                    window.__lightOn = true;
                    this.d.dom.btnLight.classList.add('active');
                    const img = this.d.dom.btnLight.querySelector('img');
                    if (img) img.src = this.d.ThemeManager.theme_fetch_icon_path('light');
                } catch (e) {
                    console.error('展台灯自动开启失败:', e);
                }
            }
        }

        console.log('摄像头已打开:', videoTrack.label || '未知设备', '分辨率:', settings.width, 'x', settings.height);
    }

    async close() {
        const { state, dom } = this.d;
        if (state.cameraStream) {
            state.cameraStream.getTracks().forEach(track => track.stop());
            state.cameraStream = null;
        }
        state.isCameraOpen = false;
        state.isCameraReady = false;
        this.d.updateSettingsControlsState();
        if (dom.cameraVideo) {
            dom.cameraVideo.style.display = 'none';
            dom.cameraVideo.srcObject = null;
        }
        this.d.updatePhotoButtonState();
        this.d.saveCurrentSourceData();

        if (state.currentImage && state.currentImageIndex >= 0) {
            const imgData = state.imageList[state.currentImageIndex];
            if (imgData?.sourceId) await this.d.updateSource(imgData.sourceId);
            this.d.renderImageCentered(state.currentImage);
        } else if (state.currentImage && state.currentFolderIndex >= 0 && state.currentFolderPageIndex >= 0) {
            const folder = state.fileList[state.currentFolderIndex];
            const page = folder.pages[state.currentFolderPageIndex];
            if (page?.sourceId) await this.d.updateSource(page.sourceId);
            this.d.renderImageCentered(state.currentImage);
        } else {
            this.d.deleteImageLayer();
            this.d.deleteDrawCanvas();
            state.strokeHistory = [];
            this.d.historyDeleteAll();
            this.d.resetSourceId();
        }
        console.log('摄像头已关闭');
    }

    async toggle() {
        const { state } = this.d;
        if (state.isCameraOpen) {
            await this.close();
        } else {
            await this.open();
        }
    }

    async switchCamera() {
        const { state } = this.d;
        state.useFrontCamera = !state.useFrontCamera;
        if (state.isCameraOpen) {
            await this.close();
            await this.open();
        }
        console.log(state.useFrontCamera ? '已切换到前置摄像头' : '已切换到后置摄像头');
    }

    // === 延迟初始化 ===

    setupDeferred() {
        if (this._deferred_camera_setup) return;
        this._deferred_camera_setup = true;

        const retry = async () => {
            document.removeEventListener('click', retry);
            document.removeEventListener('touchstart', retry);
            this._deferred_camera_setup = false;
            try {
                if (!this.d.state.isCameraOpen) await this.open();
            } catch (error) {
                const err_name = error?.name || '';
                if (err_name === 'NotAllowedError' || err_name === 'PermissionDeniedError') {
                    this.showNoCameraMessage(window.i18n?.format_translate('camera.noPermission') || '无摄像头权限');
                } else if (err_name === 'NotFoundError' || err_name === 'DevicesNotFoundError') {
                    this.showNoCameraMessage(window.i18n?.format_translate('camera.notDetected') || '未检测到摄像头');
                } else {
                    console.warn('[deferred-camera] 摄像头初始化失败:', err_name, error?.message);
                }
            }
        };
        document.addEventListener('click', retry, { once: true });
        document.addEventListener('touchstart', retry, { once: true });
    }

    async initWithoutCamera(message) {
        const { state, dom } = this.d;
        try {
            state.isCameraOpen = false;
            state.isCameraReady = false;
            state.cameraAvailable = false;
            state.cameraStream = null;
            this.d.updateSettingsControlsState();
            if (dom.cameraVideo) {
                dom.cameraVideo.style.display = 'none';
                dom.cameraVideo.srcObject = null;
            }
            const bgColor = this._getThemeBgColor();
            this.d.updateCanvasBgColor(bgColor);
            await this.d.updateSource('cam');
            this.d.updateCanvasTransform();
            this.d.updateMoveBound();
            this.d.updateCanvasPosition();
            this.d.updatePhotoButtonState();
            this.showNoCameraMessage(message);
            console.log('无摄像头模式初始化完成');
        } catch (error) {
            console.error('无摄像头模式初始化失败:', error);
            this.d.updateCanvasBgColor(this._getThemeBgColor());
            this.showNoCameraMessage(message || '摄像头不可用');
        }
    }

    // === 视频渲染 ===

    _createVideo() {
        const { state, dom } = this.d;
        const video = dom.cameraVideo;
        if (!video) { console.error('找不到 video 元素素'); return; }
        video.srcObject = state.cameraStream;
        video.play();
        video.onloadedmetadata = () => {
            state.isCameraReady = true;
            console.log('摄像头视频就绪:', video.videoWidth, 'x', video.videoHeight);
            this.updateVideoStyle();
            video.style.display = 'block';
        };
    }

    updateVideoStyle() {
        const { state, dom, DRAW_CONFIG } = this.d;
        const video = dom.cameraVideo;
        if (!video) return;
        const videoW = video.videoWidth;
        const videoH = video.videoHeight;
        if (!videoW || !videoH) return;

        const rotation = state.cameraRotation;
        const videoRatio = videoW / videoH;
        const screenRatio = DRAW_CONFIG.screenW / DRAW_CONFIG.screenH;

        let drawW, drawH;
        if (videoRatio > screenRatio) {
            drawW = DRAW_CONFIG.screenW;
            drawH = DRAW_CONFIG.screenW / videoRatio;
        } else {
            drawH = DRAW_CONFIG.screenH;
            drawW = DRAW_CONFIG.screenH * videoRatio;
        }
        const offsetX = (DRAW_CONFIG.canvasW - drawW) / 2;
        const offsetY = (DRAW_CONFIG.canvasH - drawH) / 2;

        const cache = this._lastVideoStyleCache;
        if (cache.drawW === drawW && cache.drawH === drawH && cache.offsetX === offsetX &&
            cache.offsetY === offsetY && cache.rotation === rotation && cache.isMirrored === state.isMirrored) return;
        this._lastVideoStyleCache = { drawW, drawH, offsetX, offsetY, rotation, isMirrored: state.isMirrored };

        const transforms = [];
        if (rotation !== 0) transforms.push(`rotate(${rotation}deg)`);
        if (state.isMirrored) transforms.push('scaleX(-1)');

        video.style.width = `${drawW}px`;
        video.style.height = `${drawH}px`;
        video.style.left = `${offsetX}px`;
        video.style.top = `${offsetY}px`;
        video.style.transform = transforms.join(' ');
        video.style.transformOrigin = 'center center';
        video.style.display = 'block';
        this.applyFilters();
    }

    applyFilters() {
        const { state, dom } = this.d;
        const video = dom.cameraVideo;
        const img = dom.imageElement;
        if (!video && !img) return;

        if (img && state.currentImageIndex >= 0 && state.currentImageIndex < state.imageList.length) {
            const curData = state.imageList[state.currentImageIndex];
            if (curData?.captureFilter) {
                if (video) video.style.filter = curData.captureFilter;
                return;
            }
        }

        const b = state.camera_brightness ?? 10;
        const c = state.camera_contrast ?? 1.4;
        const g = state.camera_grayscale ?? 0;
        const filterStr = `brightness(${Math.max(0, 1 + b / 100)}) contrast(${Math.max(0, c)}) grayscale(${Math.max(0, Math.min(1, g))})`;
        if (video) video.style.filter = filterStr;
        if (img) img.style.filter = filterStr;
    }

    // === 拍照 ===

    async saveImage() {
        const { state } = this.d;
        const video = document.getElementById('cameraVideo');
        if (!video) { console.error('找不到视频元素'); return; }
        if (!state.isCameraReady) {
            this.d.showErrorDialog(
                window.i18n?.format_translate('camera.notReady') || '摄像头未就绪',
                window.i18n?.format_translate('camera.notReadyDesc') || '摄像头尚未就绪，请稍后再试'
            );
            return;
        }
        const videoW = video.videoWidth;
        const videoH = video.videoHeight;
        if (!videoW || !videoH) {
            this.d.showErrorDialog(
                window.i18n?.format_translate('camera.notReady') || '摄像头未就绪',
                window.i18n?.format_translate('camera.notReadyDesc') || '摄像头尚未就绪，请稍后再试'
            );
            return;
        }

        this.d.saveCurrentSourceData();

        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        const rotation = state.cameraRotation || 0;
        if (rotation % 180 === 0) {
            tempCanvas.width = videoW;
            tempCanvas.height = videoH;
        } else {
            tempCanvas.width = videoH;
            tempCanvas.height = videoW;
        }
        tempCtx.save();
        tempCtx.translate(tempCanvas.width / 2, tempCanvas.height / 2);
        if (rotation !== 0) tempCtx.rotate(rotation * Math.PI / 180);
        if (state.isMirrored) tempCtx.scale(-1, 1);
        tempCtx.drawImage(video, -videoW / 2, -videoH / 2);
        tempCtx.restore();

        const b = state.camera_brightness ?? 10;
        const c = state.camera_contrast ?? 1.4;
        const g = state.camera_grayscale ?? 0;
        const captureFilter = `brightness(${Math.max(0, 1 + b / 100)}) contrast(${Math.max(0, c)}) grayscale(${Math.max(0, Math.min(1, g))})`;
        const rawDataUrl = tempCanvas.toDataURL('image/png');

        if (window.__TAURI__) {
            try {
                const { invoke } = window.__TAURI__.core;
                const bakeCanvas = document.createElement('canvas');
                bakeCanvas.width = tempCanvas.width;
                bakeCanvas.height = tempCanvas.height;
                const bakeCtx = bakeCanvas.getContext('2d');
                bakeCtx.filter = captureFilter;
                bakeCtx.drawImage(tempCanvas, 0, 0);
                const bakeBlob = await new Promise((resolve, reject) => {
                    bakeCanvas.toBlob(b2 => { if (b2) resolve(b2); else reject(new Error('Failed to create blob')); }, 'image/png');
                });
                const bakeDataUrl = await camera_format_blob_to_data_url(bakeBlob);
                const result = await invoke('image_save_file', { imageData: bakeDataUrl, prefix: 'photo' });
                console.log('图片已保存:', result.path);
            } catch (error) {
                console.error('保存图片失败:', error);
            }
        }

        const img = new Image();
        img.src = rawDataUrl;
        img.onload = () => {
            const photoName = window.i18n?.format_translate('camera.photoName', { n: state.imageList.length + 1 }) || `拍摄${state.imageList.length + 1}`;
            this.d.saveImageToList(img, photoName, captureFilter);
            this.d.showSidebarIfHidden();
            console.log('已捕获摄像头画面并保存到图片列表');
        };
        img.onerror = () => console.error('加载拍摄的图片失败');
    }

    updateFrameRate(idealFps) {
        const { state } = this.d;
        if (!state.cameraStream) return;
        const track = state.cameraStream.getVideoTracks()[0];
        if (!track) return;
        const constraints = idealFps !== null
            ? { frameRate: { ideal: idealFps } }
            : { frameRate: { ideal: 30 } };
        track.applyConstraints(constraints).catch(e => console.warn('摄像头帧率约束设置失败:', e));
    }

    // === UI ===

    showNoCameraMessage(message) {
        const { dom, DRAW_CONFIG, ThemeManager } = this.d;
        if (!dom.canvasWrapper) return;

        let msgElement = document.getElementById('noCameraMessage');
        if (!msgElement) {
            msgElement = document.createElement('div');
            msgElement.id = 'noCameraMessage';
            dom.canvasWrapper.appendChild(msgElement);
        }

        let style = { textColor: '#ffffff', secondaryTextColor: 'rgba(255,255,255,0.8)', tertiaryTextColor: 'rgba(255,255,255,0.5)', textShadow: '0 1px 3px rgba(0,0,0,0.5)' };
        try {
            const themeStyle = ThemeManager.theme_fetch_no_camera_style();
            if (themeStyle) style = themeStyle;
        } catch (_) {}

        msgElement.style.cssText = `position:absolute;top:0;left:0;width:${DRAW_CONFIG.canvasW}px;height:${DRAW_CONFIG.canvasH}px;display:flex;flex-direction:column;align-items:center;justify-content:center;z-index:1;pointer-events:none;`;
        msgElement.dataset.message = message || '';
        msgElement.innerHTML = `
            <div style="font-size:4vw;color:${style.textColor};margin-bottom:3vh;text-shadow:${style.textShadow}">( $ _ $ )</div>
            <div style="font-size:1.8vw;color:${style.secondaryTextColor};margin-bottom:1.5vh;text-shadow:${style.textShadow}">${window.i18n?.format_translate('camera.deviceNotFound') || '找不到展台设备'}</div>
            <div style="font-size:1.2vw;color:${style.tertiaryTextColor};text-shadow:${style.textShadow}">${message}</div>
        `;
    }

    hideNoCameraMessage() {
        const msgElement = document.getElementById('noCameraMessage');
        if (msgElement) msgElement.style.display = 'none';
    }

    updatePhotoButtonState() {
        const { state, dom, ThemeManager } = this.d;
        const btnPhoto = dom.btnPhoto;
        if (!state.cameraAvailable) {
            if (btnPhoto) btnPhoto.style.display = 'none';
            return;
        }
        if (btnPhoto) btnPhoto.style.display = '';
        if (!btnPhoto) return;

        const showText = ThemeManager.theme_fetch_toolbar_text();
        let newState, html, title;
        const photoText = window.i18n?.format_translate('toolbar.photo') || '拍照';
        const switchToCameraText = window.i18n?.format_translate('camera.switchToCamera') || '切换到摄像头';

        if (state.isCameraOpen) {
            newState = 'camera';
            html = `${ThemeManager.theme_fetch_icon('camera', { alt: photoText })}${showText ? photoText : ''}`;
            title = window.i18n?.format_translate('camera.captureFrame') || '捕获摄像头画面';
        } else if ((state.currentImageIndex >= 0 && state.imageList.length > 0) ||
                   (state.currentFolderIndex >= 0 && state.currentFolderPageIndex >= 0)) {
            newState = 'switch';
            html = `${ThemeManager.theme_fetch_icon('camera-fill', { alt: switchToCameraText })}${showText ? switchToCameraText : ''}`;
            title = window.i18n?.format_translate('camera.switchToCamera') || '返回摄像头';
        } else {
            newState = 'save';
            html = `${ThemeManager.theme_fetch_icon('camera', { alt: photoText })}${showText ? photoText : ''}`;
            title = window.i18n?.format_translate('camera.saveScreenshot') || '保存画布截图';
        }

        if (this._lastPhotoButtonState === newState) return;
        this._lastPhotoButtonState = newState;
        btnPhoto.innerHTML = html;
        btnPhoto.title = title;
    }

    // === 工具 ===

    _getThemeBgColor() {
        try {
            const c = this.d.ThemeManager.theme_fetch_canvas_bg_color();
            if (c && typeof c === 'string' && c.match(/^#[0-9a-fA-F]{6}$/)) return c;
        } catch (_) {}
        return '#2a2a2a';
    }
}

export function camera_format_blob_to_data_url(blob) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
    });
}
