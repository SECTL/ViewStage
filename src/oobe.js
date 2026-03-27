const invoke = window.__TAURI__?.core?.invoke;

(async function initI18n() {
    if (window.i18n) {
        await window.i18n.init();
        updateOOBEPageTexts();
    }
})();

function updateOOBEPageTexts() {
    if (!window.i18n) return;
    
    document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.getAttribute('data-i18n');
        const translation = window.i18n.t(key);
        if (translation) {
            if (el.tagName === 'INPUT' && el.hasAttribute('placeholder')) {
                el.placeholder = translation;
            } else {
                el.textContent = translation;
            }
        }
    });
    
    document.title = window.i18n.t('oobe.welcome') || '欢迎使用 ViewStage';
}

let blobs = [];
let animationId = null;
let lastFrameTime = 0;
const frameInterval = 33;
let carouselInterval = null;
let currentSlide = 0;
let cachedSettings = {};
let importedSettings = null;

const defaultConfig = {
    width: 1920,
    height: 1080,
    language: "zh-CN",
    defaultCamera: "",
    cameraWidth: 1280,
    cameraHeight: 720,
    pdfScale: 1.5,
    canvasScale: 2,
    dprLimit: 2,
    highResOptimization: false,
    penColors: [
        {"r": 52, "g": 152, "b": 219},
        {"r": 46, "g": 204, "b": 113},
        {"r": 231, "g": 76, "b": 60},
        {"r": 243, "g": 156, "b": 18},
        {"r": 155, "g": 89, "b": 182},
        {"r": 26, "g": 188, "b": 156},
        {"r": 52, "g": 73, "b": 94},
        {"r": 233, "g": 30, "b": 99},
        {"r": 0, "g": 188, "b": 212},
        {"r": 139, "g": 195, "b": 74},
        {"r": 255, "g": 87, "b": 34},
        {"r": 103, "g": 58, "b": 183},
        {"r": 121, "g": 85, "b": 72},
        {"r": 0, "g": 0, "b": 0},
        {"r": 255, "g": 255, "b": 255}
    ],
    fileAssociations: false
};

function generateRandomColor() {
    const hue = Math.floor(Math.random() * 360);
    const saturation = 55 + Math.floor(Math.random() * 25);
    const lightness = 45 + Math.floor(Math.random() * 20);
    return `hsla(${hue}, ${saturation}%, ${lightness}%, 0.6)`;
}

function createBlobs() {
    const auroraBg = document.getElementById('auroraBg');
    if (!auroraBg) return;
    
    auroraBg.innerHTML = '';
    blobs = [];
    
    const blobCount = 5;
    const width = window.innerWidth;
    const height = window.innerHeight;
    
    for (let i = 0; i < blobCount; i++) {
        const blob = document.createElement('div');
        blob.className = 'aurora-blob';
        
        const size = 400 + Math.random() * 300;
        blob.style.width = size + 'px';
        blob.style.height = size + 'px';
        blob.style.background = generateRandomColor();
        
        auroraBg.appendChild(blob);
        
        const x = Math.random() * width;
        const y = Math.random() * height;
        const angle = Math.random() * Math.PI * 2;
        const speed = 0.5 + Math.random() * 1.5;
        
        blobs.push({
            element: blob,
            x: x,
            y: y,
            vx: Math.cos(angle) * speed,
            vy: Math.sin(angle) * speed,
            speed: speed
        });
    }
}

function updateBlobs(currentTime) {
    if (currentTime - lastFrameTime < frameInterval) {
        animationId = requestAnimationFrame(updateBlobs);
        return;
    }
    lastFrameTime = currentTime;
    
    const width = window.innerWidth;
    const height = window.innerHeight;
    
    blobs.forEach(blob => {
        blob.x += blob.vx;
        blob.y += blob.vy;
        
        if (blob.x < -200 || blob.x > width + 200) {
            blob.vx = -blob.vx;
            blob.x = Math.max(-200, Math.min(width + 200, blob.x));
        }
        if (blob.y < -200 || blob.y > height + 200) {
            blob.vy = -blob.vy;
            blob.y = Math.max(-200, Math.min(height + 200, blob.y));
        }
        
        blob.element.style.transform = `translate(${blob.x}px, ${blob.y}px)`;
    });
    
    animationId = requestAnimationFrame(updateBlobs);
}

function startAurora() {
    const auroraBg = document.getElementById('auroraBg');
    if (!auroraBg) return;
    
    if (blobs.length === 0) {
        createBlobs();
    }
    if (!animationId) {
        lastFrameTime = 0;
        updateBlobs(performance.now());
    }
    auroraBg.classList.add('active');
}

function setupCarousel() {
    const images = document.querySelectorAll('.carousel-image');
    const carouselPage = document.getElementById('carouselPage');
    
    function showSlide(index) {
        images.forEach((img, i) => {
            img.classList.toggle('active', i === index);
        });
        currentSlide = index;
    }
    
    function nextSlide() {
        const next = (currentSlide + 1) % images.length;
        showSlide(next);
    }
    
    carouselInterval = setInterval(nextSlide, 6000);
    
    carouselPage.addEventListener('click', () => {
        showPage1();
    });
}

function showPage1() {
    clearInterval(carouselInterval);
    
    const carouselPage = document.getElementById('carouselPage');
    const page1 = document.getElementById('page1');
    const closeBtn = document.getElementById('closeBtn');
    
    carouselPage.style.opacity = '0';
    
    setTimeout(() => {
        carouselPage.style.display = 'none';
        page1.style.display = 'flex';
        closeBtn.style.display = 'flex';
        
        setTimeout(() => {
            page1.classList.add('visible');
        }, 10);
        
        setupCustomSelects();
        setupPage1Buttons();
        setupCloseButton();
    }, 250);
}

function showPage2() {
    const page1 = document.getElementById('page1');
    const page2 = document.getElementById('page2');
    
    page1.classList.remove('visible');
    
    setTimeout(() => {
        page1.style.display = 'none';
        page2.style.display = 'flex';
        
        setTimeout(() => {
            page2.classList.add('visible');
        }, 10);
        
        setupPage2Buttons();
    }, 250);
}

function showPage1FromPage2() {
    const page1 = document.getElementById('page1');
    const page2 = document.getElementById('page2');
    
    page2.classList.remove('visible');
    
    setTimeout(() => {
        page2.style.display = 'none';
        page1.style.display = 'flex';
        
        setTimeout(() => {
            page1.classList.add('visible');
        }, 10);
    }, 250);
}

async function showPage3() {
    const page2 = document.getElementById('page2');
    const page3 = document.getElementById('page3');
    
    page2.classList.remove('visible');
    
    setTimeout(() => {
        page2.style.display = 'none';
        page3.style.display = 'flex';
        
        setTimeout(() => {
            page3.classList.add('visible');
        }, 10);
        
        initResolutionSelect();
        setupPage3Buttons();
    }, 250);
}

function showPage2FromPage3() {
    const page2 = document.getElementById('page2');
    const page3 = document.getElementById('page3');
    
    page3.classList.remove('visible');
    
    setTimeout(() => {
        page3.style.display = 'none';
        page2.style.display = 'flex';
        
        setTimeout(() => {
            page2.classList.add('visible');
        }, 10);
    }, 250);
}

async function initResolutionSelect() {
    const resolutions = await invoke('get_available_resolutions');
    const resolutionOptions = document.getElementById('resolutionOptions');
    const resolutionSelected = document.getElementById('resolutionSelected');

    resolutionOptions.innerHTML = '';
    resolutions.forEach((res, index) => {
        const option = document.createElement('div');
        option.className = 'select-option' + (index === 0 ? ' selected' : '');
        option.dataset.value = `${res[0]}x${res[1]}`;
        option.textContent = res[2];
        resolutionOptions.appendChild(option);
    });

    if (resolutions.length > 0) {
        resolutionSelected.textContent = resolutions[0][2];
    }
    
    setupCustomSelects();
}

function setupCustomSelects() {
    document.querySelectorAll('.custom-select:not([data-initialized])').forEach(select => {
        select.setAttribute('data-initialized', 'true');
        
        const selected = select.querySelector('.select-selected');
        const options = select.querySelector('.select-options');

        selected.addEventListener('click', (e) => {
            e.stopPropagation();
            document.querySelectorAll('.custom-select').forEach(s => {
                if (s !== select) s.classList.remove('open');
            });
            select.classList.toggle('open');
        });

        options.addEventListener('click', async (e) => {
            const option = e.target.closest('.select-option');
            if (option) {
                selected.textContent = option.textContent;
                options.querySelectorAll('.select-option').forEach(o => o.classList.remove('selected'));
                option.classList.add('selected');
                select.classList.remove('open');
                
                if (select.id === 'languageSelect' && window.i18n) {
                    const newLocale = option.dataset.value;
                    await window.i18n.setLocale(newLocale);
                    updateOOBEPageTexts();
                }
            }
        });
    });
}

let documentClickInitialized = false;

function initDocumentClickHandler() {
    if (documentClickInitialized) return;
    documentClickInitialized = true;
    
    document.addEventListener('click', () => {
        document.querySelectorAll('.custom-select').forEach(s => s.classList.remove('open'));
    });
}

function setupPage1Buttons() {
    document.getElementById('btnNext1').addEventListener('click', async () => {
        const languageSelect = document.getElementById('languageSelect');
        const language = languageSelect.querySelector('.select-option.selected').dataset.value;

        cachedSettings.language = language;
        showPage2();
    });
}

function setupCloseButton() {
    document.getElementById('closeBtn').addEventListener('click', async () => {
        await invoke('exit_app');
    });
}

function setupPage2Buttons() {
    document.getElementById('quickSetup').addEventListener('click', async () => {
        showPage3();
    });

    document.getElementById('importConfig').addEventListener('click', async () => {
        try {
            const { open } = window.__TAURI__.dialog;
            const { readTextFile } = window.__TAURI__.fs;
            
            const filePath = await open({
                filters: [{ name: 'JSON', extensions: ['json'] }]
            });
            
            if (filePath) {
                const jsonStr = await readTextFile(filePath);
                const settings = JSON.parse(jsonStr);
                
                if (validateConfig(settings)) {
                    await invoke('save_settings', { settings });
                    console.log('设置已导入:', filePath);
                    showPage5();
                } else {
                    console.error('配置文件格式不正确');
                }
            }
        } catch (error) {
            console.error('导入设置失败:', error);
        }
    });

    document.getElementById('btnBack2').addEventListener('click', () => {
        showPage1FromPage2();
    });
}

function validateConfig(config) {
    if (!config || typeof config !== 'object') return false;
    
    const requiredFields = ['width', 'height', 'language'];
    for (const field of requiredFields) {
        if (config[field] === undefined) {
            return false;
        }
    }
    
    return true;
}

function setupPage3Buttons() {
    document.getElementById('btnBack3').addEventListener('click', () => {
        showPage2FromPage3();
    });

    document.getElementById('btnNext3').addEventListener('click', async () => {
        const resolutionSelect = document.getElementById('resolutionSelect');
        const highResOptimizationToggle = document.getElementById('highResOptimizationToggle');
        
        const resolution = resolutionSelect.querySelector('.select-option.selected').dataset.value;
        const [width, height] = resolution.split('x').map(Number);
        
        cachedSettings.width = width;
        cachedSettings.height = height;
        cachedSettings.highResOptimization = highResOptimizationToggle ? highResOptimizationToggle.checked : false;
        
        showPage4();
    });
}

async function showPage4() {
    const page3 = document.getElementById('page3');
    const page4 = document.getElementById('page4');
    
    page3.classList.remove('visible');
    
    setTimeout(() => {
        page3.style.display = 'none';
        page4.style.display = 'flex';
        
        setTimeout(() => {
            page4.classList.add('visible');
        }, 10);
        
        initCameraSelect();
        setupPage4Buttons();
    }, 250);
}

function showPage3FromPage4() {
    const page3 = document.getElementById('page3');
    const page4 = document.getElementById('page4');
    
    page4.classList.remove('visible');
    
    setTimeout(() => {
        page4.style.display = 'none';
        page3.style.display = 'flex';
        
        setTimeout(() => {
            page3.classList.add('visible');
        }, 10);
    }, 250);
}

async function initCameraSelect() {
    const cameraOptions = document.getElementById('cameraOptions');
    const cameraSelected = document.getElementById('cameraSelected');
    const cameraResolutionSelected = document.getElementById('cameraResolutionSelected');

    let stream = null;
    let track = null;
    
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        track = stream.getVideoTracks()[0];
        
        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter(device => device.kind === 'videoinput');

        cameraOptions.innerHTML = '';
        
        if (videoDevices.length === 0) {
            cameraSelected.textContent = window.i18n?.t('settings.noCameraDetected') || '未检测到摄像头';
            cameraResolutionSelected.textContent = '-';
            disableCameraSettings();
            return;
        }

        videoDevices.forEach((device, index) => {
            const option = document.createElement('div');
            option.className = 'select-option' + (index === 0 ? ' selected' : '');
            option.dataset.value = device.deviceId;
            const cameraText = window.i18n?.t('camera.camera') || '摄像头';
            option.textContent = device.label || `${cameraText} ${index + 1}`;
            cameraOptions.appendChild(option);
        });

        const cameraText = window.i18n?.t('camera.camera') || '摄像头';
        cameraSelected.textContent = videoDevices[0].label || `${cameraText} 1`;
        
        await initCameraResolutionSelect(track);
        
        setupCustomSelects();
    } catch (error) {
        console.error('摄像头检测失败:', error.name);
        
        if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
            cameraSelected.textContent = window.i18n?.t('settings.noCameraPermission') || '无摄像头权限';
        } else if (error.name === 'NotFoundError') {
            cameraSelected.textContent = window.i18n?.t('settings.noCameraDetected') || '未检测到摄像头';
        } else {
            cameraSelected.textContent = window.i18n?.t('settings.getFailed') || '获取失败';
        }
        
        cameraResolutionSelected.textContent = '-';
        
        disableCameraSettings();
    } finally {
        if (track) {
            track.stop();
        }
        if (stream) {
            stream.getTracks().forEach(t => t.stop());
        }
    }
}

function disableCameraSettings() {
    const cameraSettingItems = [
        document.querySelector('#cameraSelect')?.closest('.setting-item'),
        document.querySelector('#cameraResolutionSelect')?.closest('.setting-item'),
    ];
    
    cameraSettingItems.forEach(item => {
        if (item) {
            item.classList.add('disabled');
        }
    });
}

async function initCameraResolutionSelect(track) {
    const cameraResolutionOptions = document.getElementById('cameraResolutionOptions');
    const cameraResolutionSelected = document.getElementById('cameraResolutionSelected');

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
                const maxText = window.i18n?.t('settings.maximum') || '最大';
                resolutions.push({ w: widths.max, h: heights.max, label: `${maxResLabel} (${maxText})` });
            }
        }
    }
    
    cameraResolutionOptions.innerHTML = '';
    
    if (resolutions.length === 0) {
        cameraResolutionSelected.textContent = window.i18n?.t('settings.cannotGet') || '无法获取';
    } else {
        const defaultOption = resolutions.find(r => r.w === 1280 && r.h === 720) || resolutions[0];
        const defaultIndex = resolutions.indexOf(defaultOption);
        
        resolutions.forEach((res, index) => {
            const option = document.createElement('div');
            option.className = 'select-option' + (index === defaultIndex ? ' selected' : '');
            option.dataset.width = res.w;
            option.dataset.height = res.h;
            option.dataset.value = `${res.w}x${res.h}`;
            option.textContent = res.label;
            cameraResolutionOptions.appendChild(option);
        });
        
        cameraResolutionSelected.textContent = defaultOption.label;
    }
}

function setupPage4Buttons() {
    document.getElementById('btnBack4').addEventListener('click', () => {
        showPage3FromPage4();
    });

    document.getElementById('btnNext4').addEventListener('click', async () => {
        const cameraSelect = document.getElementById('cameraSelect');
        const cameraResolutionSelect = document.getElementById('cameraResolutionSelect');
        const assocPdf = document.getElementById('assocPdf');
        
        const cameraOption = cameraSelect.querySelector('.select-option.selected');
        if (cameraOption) {
            cachedSettings.defaultCamera = cameraOption.dataset.value;
        }
        
        const resolutionOption = cameraResolutionSelect.querySelector('.select-option.selected');
        if (resolutionOption) {
            cachedSettings.cameraWidth = parseInt(resolutionOption.dataset.width);
            cachedSettings.cameraHeight = parseInt(resolutionOption.dataset.height);
        }
        
        cachedSettings.fileAssociations = assocPdf.checked;
        
        const finalSettings = mergeSettings(cachedSettings);
        
        await invoke('save_settings', { settings: finalSettings });
        showPage5();
    });
}

function showPage5() {
    const currentPage = document.querySelector('.oobe-container.visible');
    const page5 = document.getElementById('page5');
    
    if (currentPage) {
        currentPage.classList.remove('visible');
    }
    
    setTimeout(() => {
        if (currentPage) {
            currentPage.style.display = 'none';
        }
        page5.style.display = 'flex';
        
        setTimeout(() => {
            page5.classList.add('visible');
        }, 10);
        
        setupPage5Buttons();
    }, 250);
}

function setupPage5Buttons() {
    document.getElementById('btnRestart').addEventListener('click', async () => {
        await invoke('complete_oobe');
    });
}

function mergeSettings(cached) {
    const base = importedSettings ? { ...importedSettings } : { ...defaultConfig };
    
    return { ...base, ...cached };
}

startAurora();
setupCarousel();
initDocumentClickHandler();
