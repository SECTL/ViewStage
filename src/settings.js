document.addEventListener('DOMContentLoaded', async () => {
    const btnClose = document.getElementById('btnClose');
    const auroraBg = document.getElementById('auroraBg');
    
    async function loadAppVersion() {
        if (window.__TAURI__) {
            try {
                const { invoke } = window.__TAURI__.core;
                const version = await invoke('get_app_version');
                
                const versionNumber = document.getElementById('versionNumber');
                const currentVersion = document.getElementById('currentVersion');
                const latestVersion = document.getElementById('latestVersion');
                
                if (versionNumber) versionNumber.textContent = version;
                if (currentVersion) currentVersion.textContent = version;
                if (latestVersion) latestVersion.textContent = version;
            } catch (error) {
                console.error('获取版本号失败:', error);
            }
        }
        
        const copyrightYear = document.getElementById('copyrightYear');
        if (copyrightYear) {
            copyrightYear.textContent = new Date().getFullYear();
        }
    }
    
    async function loadSettings() {
        if (window.__TAURI__) {
            try {
                const { invoke } = window.__TAURI__.core;
                const settings = await invoke('get_settings');
                
                const selectSelected = document.getElementById('selectSelected');
                const languageOptions = document.querySelectorAll('#selectOptions .select-option');
                
                if (settings.language && selectSelected) {
                    languageOptions.forEach(option => {
                        if (option.dataset.value === settings.language) {
                            selectSelected.textContent = option.textContent;
                            option.classList.add('selected');
                        } else {
                            option.classList.remove('selected');
                        }
                    });
                }
                
                const resolutionSelected = document.getElementById('resolutionSelected');
                const resolutionOptionsContainer = document.getElementById('resolutionOptions');
                
                if (resolutionSelected && resolutionOptionsContainer) {
                    const availableResolutions = await invoke('get_available_resolutions');
                    
                    resolutionOptionsContainer.innerHTML = '';
                    
                    availableResolutions.forEach(res => {
                        const [width, height, label] = res;
                        const option = document.createElement('div');
                        option.className = 'select-option';
                        option.dataset.value = `${width}x${height}`;
                        option.dataset.width = width;
                        option.dataset.height = height;
                        option.textContent = label;
                        resolutionOptionsContainer.appendChild(option);
                    });
                    
                    if (settings.width && settings.height) {
                        const resolution = `${settings.width}x${settings.height}`;
                        const resolutionOptions = document.querySelectorAll('#resolutionOptions .select-option');
                        resolutionOptions.forEach(option => {
                            if (option.dataset.value === resolution) {
                                resolutionSelected.textContent = option.textContent;
                                option.classList.add('selected');
                            } else {
                                option.classList.remove('selected');
                            }
                        });
                    }
                }
                
                const cameraSelected = document.getElementById('cameraSelected');
                const cameraOptionsContainer = document.getElementById('cameraOptions');
                
                if (cameraSelected && cameraOptionsContainer) {
                    try {
                        const devices = await navigator.mediaDevices.enumerateDevices();
                        const videoDevices = devices.filter(device => device.kind === 'videoinput');
                        
                        cameraOptionsContainer.innerHTML = '';
                        
                        if (videoDevices.length === 0) {
                            cameraSelected.textContent = '未检测到摄像头';
                        } else {
                            videoDevices.forEach((device, index) => {
                                const option = document.createElement('div');
                                option.className = 'select-option';
                                option.dataset.value = device.deviceId;
                                
                                let label = device.label || `摄像头 ${index + 1}`;
                                if (label.includes('back') || label.includes('后置') || label.includes('rear')) {
                                    label = `后置: ${label}`;
                                } else if (label.includes('front') || label.includes('前置') || label.includes('user')) {
                                    label = `前置: ${label}`;
                                }
                                
                                option.textContent = label;
                                cameraOptionsContainer.appendChild(option);
                            });
                            
                            if (settings.defaultCamera) {
                                const cameraOptions = cameraOptionsContainer.querySelectorAll('.select-option');
                                let found = false;
                                cameraOptions.forEach(option => {
                                    if (option.dataset.value === settings.defaultCamera) {
                                        cameraSelected.textContent = option.textContent;
                                        option.classList.add('selected');
                                        found = true;
                                    } else {
                                        option.classList.remove('selected');
                                    }
                                });
                                
                                if (!found && cameraOptions.length > 0) {
                                    cameraSelected.textContent = cameraOptions[0].textContent;
                                    cameraOptions[0].classList.add('selected');
                                }
                            } else {
                                const cameraOptions = cameraOptionsContainer.querySelectorAll('.select-option');
                                if (cameraOptions.length > 0) {
                                    cameraSelected.textContent = cameraOptions[0].textContent;
                                    cameraOptions[0].classList.add('selected');
                                }
                            }
                        }
                    } catch (error) {
                        console.error('获取摄像头列表失败:', error);
                        cameraSelected.textContent = '获取失败';
                    }
                }
                
                // 摄像头分辨率设置
                const cameraResolutionSelected = document.getElementById('cameraResolutionSelected');
                const cameraResolutionOptionsContainer = document.getElementById('cameraResolutionOptions');
                
                if (cameraResolutionSelected && cameraResolutionOptionsContainer) {
                    let stream = null;
                    let track = null;
                    try {
                        stream = await navigator.mediaDevices.getUserMedia({ video: true });
                        track = stream.getVideoTracks()[0];
                        const capabilities = track.getCapabilities();
                        
                        const resolutions = [];
                        if (capabilities.width && capabilities.height) {
                            const widths = capabilities.width;
                            const heights = capabilities.height;
                            
                            const commonResolutions = [
                                { w: 640, h: 480, label: '640 x 480 (VGA)' },
                                { w: 800, h: 600, label: '800 x 600 (SVGA)' },
                                { w: 1280, h: 720, label: '1280 x 720 (720p)' },
                                { w: 1280, h: 960, label: '1280 x 960' },
                                { w: 1600, h: 1200, label: '1600 x 1200' },
                                { w: 1920, h: 1080, label: '1920 x 1080 (1080p)' },
                                { w: 2560, h: 1440, label: '2560 x 1440 (2K)' },
                                { w: 3840, h: 2160, label: '3840 x 2160 (4K)' }
                            ];
                            
                            commonResolutions.forEach(res => {
                                if (widths.max >= res.w && heights.max >= res.h) {
                                    resolutions.push(res);
                                }
                            });
                            
                            if (widths.max && heights.max) {
                                const maxResLabel = `${widths.max} x ${heights.max}`;
                                const exists = resolutions.some(r => r.w === widths.max && r.h === heights.max);
                                if (!exists) {
                                    resolutions.push({ w: widths.max, h: heights.max, label: `${maxResLabel} (最大)` });
                                }
                            }
                        }
                        
                        cameraResolutionOptionsContainer.innerHTML = '';
                        
                        if (resolutions.length === 0) {
                            cameraResolutionSelected.textContent = '无法获取';
                        } else {
                            resolutions.forEach(res => {
                                const option = document.createElement('div');
                                option.className = 'select-option';
                                option.dataset.width = res.w;
                                option.dataset.height = res.h;
                                option.dataset.value = `${res.w}x${res.h}`;
                                option.textContent = res.label;
                                cameraResolutionOptionsContainer.appendChild(option);
                            });
                            
                            // 加载保存的分辨率
                            const savedWidth = settings.cameraWidth || 1280;
                            const savedHeight = settings.cameraHeight || 720;
                            const savedRes = `${savedWidth}x${savedHeight}`;
                            
                            const resOptions = cameraResolutionOptionsContainer.querySelectorAll('.select-option');
                            let found = false;
                            resOptions.forEach(option => {
                                if (option.dataset.value === savedRes) {
                                    cameraResolutionSelected.textContent = option.textContent;
                                    option.classList.add('selected');
                                    found = true;
                                }
                            });
                            
                            if (!found && resOptions.length > 0) {
                                // 默认选择 720p 或第一个
                                const defaultOption = Array.from(resOptions).find(opt => 
                                    opt.dataset.value === '1280x720'
                                ) || resOptions[0];
                                cameraResolutionSelected.textContent = defaultOption.textContent;
                                defaultOption.classList.add('selected');
                            }
                        }
                    } catch (error) {
                        console.error('获取摄像头分辨率失败:', error);
                        cameraResolutionSelected.textContent = '获取失败';
                    } finally {
                        if (track) {
                            track.stop();
                        }
                        if (stream) {
                            stream.getTracks().forEach(t => t.stop());
                        }
                    }
                }
                
                // 帧率设置
                const moveFpsSelected = document.getElementById('moveFpsSelected');
                const moveFpsOptionsContainer = document.getElementById('moveFpsOptions');
                const drawFpsSelected = document.getElementById('drawFpsSelected');
                const drawFpsOptionsContainer = document.getElementById('drawFpsOptions');
                
                // 获取摄像头最大帧率
                let maxFps = 30;
                let fpsStream = null;
                let fpsTrack = null;
                try {
                    fpsStream = await navigator.mediaDevices.getUserMedia({ video: true });
                    fpsTrack = fpsStream.getVideoTracks()[0];
                    const capabilities = fpsTrack.getCapabilities();
                    if (capabilities.frameRate && capabilities.frameRate.max) {
                        maxFps = Math.min(capabilities.frameRate.max, 60);
                    }
                } catch (e) {
                    console.log('无法获取摄像头帧率能力，使用默认值');
                } finally {
                    if (fpsTrack) {
                        fpsTrack.stop();
                    }
                    if (fpsStream) {
                        fpsStream.getTracks().forEach(t => t.stop());
                    }
                }
                
                // 生成帧率选项
                const fpsOptions = [];
                const fpsValues = [5, 10, 15, 20, 24, 30, 60];
                fpsValues.forEach(fps => {
                    if (fps <= maxFps) {
                        fpsOptions.push(fps);
                    }
                });
                
                // 移动时帧率选项
                if (moveFpsSelected && moveFpsOptionsContainer) {
                    moveFpsOptionsContainer.innerHTML = '';
                    fpsOptions.forEach(fps => {
                        const option = document.createElement('div');
                        option.className = 'select-option';
                        option.dataset.value = fps;
                        option.textContent = `${fps} FPS`;
                        moveFpsOptionsContainer.appendChild(option);
                    });
                    
                    const savedMoveFps = settings.moveFps || 30;
                    const moveFpsOptionElements = moveFpsOptionsContainer.querySelectorAll('.select-option');
                    moveFpsOptionElements.forEach(option => {
                        if (parseInt(option.dataset.value) === savedMoveFps) {
                            moveFpsSelected.textContent = option.textContent;
                            option.classList.add('selected');
                        }
                    });
                }
                
                // 绘画时帧率选项
                if (drawFpsSelected && drawFpsOptionsContainer) {
                    drawFpsOptionsContainer.innerHTML = '';
                    fpsOptions.forEach(fps => {
                        const option = document.createElement('div');
                        option.className = 'select-option';
                        option.dataset.value = fps;
                        option.textContent = `${fps} FPS`;
                        drawFpsOptionsContainer.appendChild(option);
                    });
                    
                    const savedDrawFps = settings.drawFps || 10;
                    const drawFpsOptionElements = drawFpsOptionsContainer.querySelectorAll('.select-option');
                    drawFpsOptionElements.forEach(option => {
                        if (parseInt(option.dataset.value) === savedDrawFps) {
                            drawFpsSelected.textContent = option.textContent;
                            option.classList.add('selected');
                        }
                    });
                }
                
                // 文件关联设置
                const assocPdf = document.getElementById('assocPdf');
                if (assocPdf) {
                    // 检查 ViewStage 是否是 PDF 的默认打开程序
                    try {
                        const isDefault = await invoke('check_pdf_default_app');
                        
                        if (isDefault) {
                            // 如果是默认程序，根据配置设置勾选状态
                            if (settings.fileAssociations) {
                                assocPdf.checked = settings.fileAssociations.includes('pdf');
                            }
                        } else {
                            // 如果不是默认程序，取消勾选
                            assocPdf.checked = false;
                            
                            // 如果配置中有记录，更新配置
                            if (settings.fileAssociations && settings.fileAssociations.includes('pdf')) {
                                const newAssociations = settings.fileAssociations.filter(ext => ext !== 'pdf');
                                await saveSettings({ fileAssociations: newAssociations });
                            }
                        }
                    } catch (error) {
                        console.error('检查默认程序失败:', error);
                        assocPdf.checked = false;
                    }
                }
                
                return settings;
            } catch (error) {
                console.error('加载设置失败:', error);
                return {};
            }
        }
        return {};
    }
    
    async function saveSettings(settings) {
        if (window.__TAURI__) {
            try {
                const { invoke } = window.__TAURI__.core;
                await invoke('save_settings', { settings });
                return true;
            } catch (error) {
                console.error('保存设置失败:', error);
                return false;
            }
        }
        return false;
    }
    
    loadAppVersion();
    loadSettings().then(() => {
        setupResolutionOptions();
    });
    
    const languageSelect = document.getElementById('languageSelect');
    const selectSelected = document.getElementById('selectSelected');
    const languageOptions = document.querySelectorAll('#selectOptions .select-option');
    
    if (languageSelect && selectSelected) {
        selectSelected.addEventListener('click', () => {
            languageSelect.classList.toggle('open');
        });
        
        languageOptions.forEach(option => {
            option.addEventListener('click', async () => {
                const value = option.dataset.value;
                selectSelected.textContent = option.textContent;
                
                languageOptions.forEach(opt => opt.classList.remove('selected'));
                option.classList.add('selected');
                
                languageSelect.classList.remove('open');
                
                const saved = await saveSettings({ language: value });
                
                if (saved) {
                    const restartModal = document.getElementById('restartModal');
                    if (restartModal) {
                        restartModal.classList.add('active');
                    }
                } else {
                    alert('保存设置失败，请重试');
                }
            });
        });
        
        document.addEventListener('click', (e) => {
            if (!languageSelect.contains(e.target)) {
                languageSelect.classList.remove('open');
            }
        });
    }
    
    const resolutionSelect = document.getElementById('resolutionSelect');
    const resolutionSelected = document.getElementById('resolutionSelected');
    
    if (resolutionSelect && resolutionSelected) {
        resolutionSelected.addEventListener('click', () => {
            resolutionSelect.classList.toggle('open');
        });
        
        document.addEventListener('click', (e) => {
            if (!resolutionSelect.contains(e.target)) {
                resolutionSelect.classList.remove('open');
            }
        });
    }
    
    function setupResolutionOptions() {
        const resolutionOptions = document.querySelectorAll('#resolutionOptions .select-option');
        
        resolutionOptions.forEach(option => {
            option.addEventListener('click', async () => {
                const width = parseInt(option.dataset.width);
                const height = parseInt(option.dataset.height);
                resolutionSelected.textContent = option.textContent;
                
                resolutionOptions.forEach(opt => opt.classList.remove('selected'));
                option.classList.add('selected');
                
                resolutionSelect.classList.remove('open');
                
                const saved = await saveSettings({ width, height });
                
                if (saved) {
                    const restartModal = document.getElementById('restartModal');
                    if (restartModal) {
                        restartModal.classList.add('active');
                    }
                } else {
                    alert('保存设置失败，请重试');
                }
            });
        });
    }
    
    const cameraSelect = document.getElementById('cameraSelect');
    const cameraSelected = document.getElementById('cameraSelected');
    
    if (cameraSelect && cameraSelected) {
        cameraSelected.addEventListener('click', () => {
            cameraSelect.classList.toggle('open');
        });
        
        document.addEventListener('click', (e) => {
            if (!cameraSelect.contains(e.target)) {
                cameraSelect.classList.remove('open');
            }
        });
        
        cameraSelect.addEventListener('click', async (e) => {
            const option = e.target.closest('.select-option');
            if (!option) return;
            
            const value = option.dataset.value;
            cameraSelected.textContent = option.textContent;
            
            const cameraOptions = cameraSelect.querySelectorAll('.select-option');
            cameraOptions.forEach(opt => opt.classList.remove('selected'));
            option.classList.add('selected');
            
            cameraSelect.classList.remove('open');
            
            const saved = await saveSettings({ defaultCamera: value });
            
            if (!saved) {
                alert('保存设置失败，请重试');
            }
        });
    }
    
    // 摄像头分辨率选择
    const cameraResolutionSelect = document.getElementById('cameraResolutionSelect');
    const cameraResolutionSelected = document.getElementById('cameraResolutionSelected');
    
    if (cameraResolutionSelect && cameraResolutionSelected) {
        cameraResolutionSelected.addEventListener('click', () => {
            cameraResolutionSelect.classList.toggle('open');
        });
        
        document.addEventListener('click', (e) => {
            if (!cameraResolutionSelect.contains(e.target)) {
                cameraResolutionSelect.classList.remove('open');
            }
        });
        
        cameraResolutionSelect.addEventListener('click', async (e) => {
            const option = e.target.closest('.select-option');
            if (!option) return;
            
            const width = parseInt(option.dataset.width);
            const height = parseInt(option.dataset.height);
            cameraResolutionSelected.textContent = option.textContent;
            
            const resOptions = cameraResolutionSelect.querySelectorAll('.select-option');
            resOptions.forEach(opt => opt.classList.remove('selected'));
            option.classList.add('selected');
            
            cameraResolutionSelect.classList.remove('open');
            
            await saveSettings({ cameraWidth: width, cameraHeight: height });
        });
    }
    
    // 移动时帧率选择
    const moveFpsSelect = document.getElementById('moveFpsSelect');
    const moveFpsSelected = document.getElementById('moveFpsSelected');
    
    if (moveFpsSelect && moveFpsSelected) {
        moveFpsSelected.addEventListener('click', () => {
            moveFpsSelect.classList.toggle('open');
        });
        
        document.addEventListener('click', (e) => {
            if (!moveFpsSelect.contains(e.target)) {
                moveFpsSelect.classList.remove('open');
            }
        });
        
        moveFpsSelect.addEventListener('click', async (e) => {
            const option = e.target.closest('.select-option');
            if (!option) return;
            
            const value = parseInt(option.dataset.value);
            moveFpsSelected.textContent = option.textContent;
            
            const moveFpsOptions = moveFpsSelect.querySelectorAll('.select-option');
            moveFpsOptions.forEach(opt => opt.classList.remove('selected'));
            option.classList.add('selected');
            
            moveFpsSelect.classList.remove('open');
            
            await saveSettings({ moveFps: value });
        });
    }
    
    // 绘画时帧率选择
    const drawFpsSelect = document.getElementById('drawFpsSelect');
    const drawFpsSelected = document.getElementById('drawFpsSelected');
    
    if (drawFpsSelect && drawFpsSelected) {
        drawFpsSelected.addEventListener('click', () => {
            drawFpsSelect.classList.toggle('open');
        });
        
        document.addEventListener('click', (e) => {
            if (!drawFpsSelect.contains(e.target)) {
                drawFpsSelect.classList.remove('open');
            }
        });
        
        drawFpsSelect.addEventListener('click', async (e) => {
            const option = e.target.closest('.select-option');
            if (!option) return;
            
            const value = parseInt(option.dataset.value);
            drawFpsSelected.textContent = option.textContent;
            
            const drawFpsOptions = drawFpsSelect.querySelectorAll('.select-option');
            drawFpsOptions.forEach(opt => opt.classList.remove('selected'));
            option.classList.add('selected');
            
            drawFpsSelect.classList.remove('open');
            
            await saveSettings({ drawFps: value });
        });
    }
    
    // 默认打开方式复选框
    const assocPdf = document.getElementById('assocPdf');
    if (assocPdf) {
        assocPdf.addEventListener('change', async () => {
            const associations = [];
            if (assocPdf.checked) {
                associations.push('pdf');
            }
            await saveSettings({ fileAssociations: associations });
        });
    }
    
    const btnReset = document.getElementById('btnReset');
    const modalOverlay = document.getElementById('modalOverlay');
    const modalCancel = document.getElementById('modalCancel');
    const modalConfirm = document.getElementById('modalConfirm');
    
    if (btnReset && modalOverlay && window.__TAURI__) {
        btnReset.addEventListener('click', () => {
            modalOverlay.classList.add('active');
        });
        
        modalCancel.addEventListener('click', () => {
            modalOverlay.classList.remove('active');
        });
        
        modalOverlay.addEventListener('click', (e) => {
            if (e.target === modalOverlay) {
                modalOverlay.classList.remove('active');
            }
        });
        
        modalConfirm.addEventListener('click', async () => {
            try {
                const { invoke } = window.__TAURI__.core;
                await invoke('reset_settings');
            } catch (error) {
                console.error('重置失败:', error);
                alert('重置失败: ' + error);
                modalOverlay.classList.remove('active');
            }
        });
    }
    
    const restartModal = document.getElementById('restartModal');
    const restartLater = document.getElementById('restartLater');
    const restartNow = document.getElementById('restartNow');
    
    if (restartModal && window.__TAURI__) {
        restartLater.addEventListener('click', () => {
            restartModal.classList.remove('active');
        });
        
        restartModal.addEventListener('click', (e) => {
            if (e.target === restartModal) {
                restartModal.classList.remove('active');
            }
        });
        
        restartNow.addEventListener('click', async () => {
            try {
                const { invoke } = window.__TAURI__.core;
                await invoke('restart_app');
            } catch (error) {
                console.error('重启失败:', error);
                alert('重启失败: ' + error);
            }
        });
    }
    
    let blobs = [];
    let animationId = null;
    let lastTime = 0;
    const updateInterval = 50;
    
    function generateRandomColor() {
        const hue = Math.floor(Math.random() * 360);
        const saturation = 60 + Math.floor(Math.random() * 30);
        const lightness = 50 + Math.floor(Math.random() * 20);
        return `hsla(${hue}, ${saturation}%, ${lightness}%, 0.6)`;
    }
    
    function createBlobs() {
        if (!auroraBg) return;
        
        auroraBg.innerHTML = '';
        blobs = [];
        
        const blobCount = 5;
        const width = window.innerWidth;
        const height = window.innerHeight;
        
        for (let i = 0; i < blobCount; i++) {
            const blob = document.createElement('div');
            blob.className = 'aurora-blob';
            
            const size = 350 + Math.random() * 250;
            blob.style.width = size + 'px';
            blob.style.height = size + 'px';
            blob.style.background = generateRandomColor();
            
            const x = Math.random() * width;
            const y = Math.random() * height;
            blob.style.transform = `translate(${x}px, ${y}px)`;
            
            auroraBg.appendChild(blob);
            
            blobs.push({
                element: blob,
                x: x,
                y: y,
                vx: 0,
                vy: 0
            });
        }
    }
    
    function updateBlobs(currentTime) {
        const width = window.innerWidth;
        const height = window.innerHeight;
        
        if (currentTime - lastTime >= updateInterval) {
            lastTime = currentTime;
            
            blobs.forEach(blob => {
                const noise = 0.2;
                blob.vx += (Math.random() - 0.5) * noise;
                blob.vy += (Math.random() - 0.5) * noise;
                
                blob.vx *= 0.98;
                blob.vy *= 0.98;
                
                const maxSpeed = 1;
                const speed = Math.sqrt(blob.vx * blob.vx + blob.vy * blob.vy);
                if (speed > maxSpeed) {
                    blob.vx = (blob.vx / speed) * maxSpeed;
                    blob.vy = (blob.vy / speed) * maxSpeed;
                }
                
                blob.x += blob.vx;
                blob.y += blob.vy;
                
                const margin = 100;
                if (blob.x < -margin) blob.x = width + margin;
                if (blob.x > width + margin) blob.x = -margin;
                if (blob.y < -margin) blob.y = height + margin;
                if (blob.y > height + margin) blob.y = -margin;
                
                blob.element.style.transform = `translate(${blob.x}px, ${blob.y}px)`;
            });
        }
        
        animationId = requestAnimationFrame(updateBlobs);
    }
    
    function startAurora() {
        if (blobs.length === 0) {
            createBlobs();
        }
        if (!animationId) {
            updateBlobs();
        }
    }
    
    function stopAurora() {
        if (animationId) {
            cancelAnimationFrame(animationId);
            animationId = null;
        }
    }
    
    if (btnClose) {
        btnClose.addEventListener('click', async () => {
            if (window.__TAURI__) {
                try {
                    const { getCurrentWindow } = window.__TAURI__.window;
                    const appWindow = getCurrentWindow();
                    await appWindow.close();
                } catch (error) {
                    console.error('关闭窗口失败:', error);
                }
            }
        });
    }

    const sidebarBtns = document.querySelectorAll('.sidebar-btn');
    const pages = document.querySelectorAll('.page');
    
    function showPage(pageId) {
        pages.forEach(page => page.classList.remove('active'));
        const targetPage = document.getElementById(pageId);
        if (targetPage) {
            targetPage.classList.add('active');
        }
        
        if (auroraBg) {
            if (pageId === 'pageAbout' || pageId === 'pageUpdate') {
                startAurora();
                auroraBg.classList.add('active');
            } else {
                auroraBg.classList.remove('active');
                stopAurora();
            }
        }
    }

    sidebarBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            sidebarBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            const pageMap = {
                'btnApp': 'pageApp',
                'btnCanvas': 'pageCanvas',
                'btnSource': 'pageSource',
                'btnAbout': 'pageAbout'
            };
            
            const pageId = pageMap[btn.id];
            if (pageId) {
                showPage(pageId);
            }
        });
    });

    const btnCheckUpdate = document.getElementById('btnCheckUpdate');
    if (btnCheckUpdate) {
        btnCheckUpdate.addEventListener('click', () => {
            showPage('pageUpdate');
            checkForUpdate();
        });
    }

    const btnBackToAbout = document.getElementById('btnBackToAbout');
    if (btnBackToAbout) {
        btnBackToAbout.addEventListener('click', () => {
            showPage('pageAbout');
            sidebarBtns.forEach(b => b.classList.remove('active'));
            document.getElementById('btnAbout')?.classList.add('active');
        });
    }

    async function checkForUpdate() {
        const updateStatus = document.getElementById('updateStatus');
        const updateInfo = document.getElementById('updateInfo');
        const updateIcon = document.querySelector('.update-icon');
        const latestVersionEl = document.getElementById('latestVersion');
        
        if (updateIcon) {
            updateIcon.style.animation = 'spin 2s linear infinite';
        }
        
        if (updateStatus) {
            updateStatus.textContent = '正在检查更新...';
        }
        
        if (updateInfo) {
            updateInfo.style.display = 'none';
        }
        
        try {
            if (window.__TAURI__) {
                const { invoke } = window.__TAURI__.core;
                
                const release = await invoke('check_update');
                const currentVersion = await invoke('get_app_version');
                
                const latestVersion = release.tag_name.replace(/^v/, '');
                
                if (latestVersionEl) {
                    latestVersionEl.textContent = latestVersion;
                }
                
                if (updateIcon) {
                    updateIcon.style.animation = 'none';
                }
                
                if (currentVersion === latestVersion) {
                    if (updateStatus) {
                        updateStatus.textContent = '当前已是最新版本';
                    }
                } else {
                    if (updateStatus) {
                        updateStatus.innerHTML = `发现新版本 <a href="#" id="downloadLink" style="color: #3498db; cursor: pointer;">${latestVersion}</a>`;
                        
                        const downloadLink = document.getElementById('downloadLink');
                        if (downloadLink && release.html_url) {
                            downloadLink.addEventListener('click', (e) => {
                                e.preventDefault();
                                window.__TAURI__.opener.openUrl(release.html_url);
                            });
                        }
                    }
                }
                
                if (updateInfo) {
                    updateInfo.style.display = 'block';
                }
            } else {
                if (updateIcon) {
                    updateIcon.style.animation = 'none';
                }
                if (updateStatus) {
                    updateStatus.textContent = '请在应用中检查更新';
                }
            }
        } catch (error) {
            console.error('检查更新失败:', error);
            
            if (updateIcon) {
                updateIcon.style.animation = 'none';
            }
            
            if (updateStatus) {
                updateStatus.textContent = '检查更新失败，请稍后重试';
            }
        }
    }

    const linkGithub = document.getElementById('linkGithub');
    if (linkGithub && window.__TAURI__) {
        linkGithub.addEventListener('click', (e) => {
            e.preventDefault();
            window.__TAURI__.opener.openUrl('https://github.com/ospneam/ViewStage');
        });
    }

    const linkLicense = document.getElementById('linkLicense');
    if (linkLicense && window.__TAURI__) {
        linkLicense.addEventListener('click', (e) => {
            e.preventDefault();
            window.__TAURI__.opener.openUrl('https://github.com/ospneam/ViewStage?tab=Apache-2.0-1-ov-file');
        });
    }

    showPage('pageApp');
    document.getElementById('btnApp')?.classList.add('active');
});
