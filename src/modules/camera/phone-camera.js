/**
 * phone-camera.js — 手机摄像头推流独立模块
 * 从 camera.js 中解耦，仅在 phoneServerEnabled 时加载
 *
 * 依赖注入（由 main.js 创建实例时传入）：
 *   state                       — window.state（共享引用）
 *   dom                         — window.dom（共享引用）
 *   hideNoCameraMessage()       — 隐藏无摄像头提示
 *   updateSettingsControlsState() — 刷新设置面板控件状态
 *   updatePhotoButtonState()    — 刷新拍照按钮状态
 */

export class PhoneCameraManager {
    constructor(deps) {
        this.d = deps;
        // H.264 MSE 实例状态
        this._phone_h264_video = null;
        this._phone_h264_ms = null;
        this._phone_h264_sb = null;
        this._phone_h264_queue = [];
        this._phone_h264_appending = false;
        this._phone_h264_reader = null;
        this._phone_h264_abort = null;
        // MJPEG 实例状态
        this._phone_mjpeg_img = null;
    }

    // ==================== 公开 API ====================

    async startPhoneCamera() {
        const { state } = this.d;
        if (state.phoneCameraActive) return;

        try {
            const result = await window.__TAURI__.core.invoke('phone_camera_start');
            if (result) {
                state.phoneCameraActive = true;
                state.phoneCameraReady = true;
                state.isCameraOpen = true;
                state.isCameraReady = true;
                state.cameraAvailable = true;

                this.d.hideNoCameraMessage();
                this._startPhoneStream();

                this.d.updateSettingsControlsState();
                this.d.updatePhotoButtonState();

                console.log('[phone-camera] 手机摄像头已启动');
            }
        } catch (error) {
            console.error('[phone-camera] 启动手机摄像头失败:', error);
            throw error;
        }
    }

    async stopPhoneCamera() {
        const { state } = this.d;
        if (!state.phoneCameraActive) return;

        this._stopPhoneH264Stream();
        this._stopPhoneMjpegStream();

        try {
            await window.__TAURI__.core.invoke('phone_camera_stop');
        } catch (error) {
            console.warn('[phone-camera] 停止手机摄像头API调用失败:', error);
        }

        state.phoneCameraActive = false;
        state.phoneCameraReady = false;
        state.isCameraOpen = false;
        state.isCameraReady = false;

        this.d.updateSettingsControlsState();
        this.d.updatePhotoButtonState();

        console.log('[phone-camera] 手机摄像头已停止');
    }

    async togglePhoneCamera() {
        if (this.d.state.phoneCameraActive) {
            await this.stopPhoneCamera();
        } else {
            await this.startPhoneCamera();
        }
    }

    async checkPhoneCameraStatus() {
        try {
            return await window.__TAURI__.core.invoke('phone_camera_status');
        } catch (error) {
            console.warn('[phone-camera] 查询手机摄像头状态失败:', error);
            return { active: false, device_name: null };
        }
    }

    // ==================== 流选择 ====================

    _startPhoneStream() {
        if (this._canPlayH264()) {
            console.log('[phone-camera] 使用 H.264 MSE 模式');
            this._startPhoneH264Stream();
        } else {
            console.log('[phone-camera] 浏览器不支持 H.264 MSE，降级到 MJPEG');
            this._startPhoneMjpegStream();
        }
    }

    _canPlayH264() {
        if (!window.MediaSource) return false;
        return MediaSource.isTypeSupported('video/mp4; codecs="avc1.42E01E"');
    }

    // ==================== H.264 MSE 流 ====================

    _startPhoneH264Stream() {
        this._stopPhoneH264Stream();

        const { dom } = this.d;
        const wrapper = dom.canvasWrapper;
        if (!wrapper) return;

        const info = window.phone_server_info?.();
        if (!info) {
            console.warn('[phone-camera] 服务信息不可用，降级到 MJPEG');
            this._startPhoneMjpegStream();
            return;
        }

        const video = document.createElement('video');
        video.id = 'phoneCameraStream';
        video.className = 'camera-video';
        video.autoplay = true;
        video.playsInline = true;
        video.muted = true;

        const ms = new MediaSource();
        video.src = URL.createObjectURL(ms);

        wrapper.appendChild(video);
        this._phone_h264_video = video;
        this._phone_h264_ms = ms;
        this._phone_h264_sb = null;
        this._phone_h264_queue = [];
        this._phone_h264_appending = false;
        this._phone_h264_reader = null;
        this._phone_h264_abort = null;

        if (dom.cameraVideo) dom.cameraVideo.style.display = 'none';

        ms.addEventListener('sourceopen', () => {
            console.log('[phone-camera] MediaSource opened');
            this._phone_h264_fetch(info);
        });

        ms.addEventListener('sourceended', () => {
            console.log('[phone-camera] MediaSource ended');
        });

        video.addEventListener('error', (e) => {
            console.error('[phone-camera] H.264 video 元素错误:', e);
            this._reconnectPhoneH264(info);
        });

        video.play().catch(() => {});
    }

    async _phone_h264_fetch(info) {
        const url = `http://${info.ip}:${info.port}/camera/h264`;

        try {
            const controller = new AbortController();
            this._phone_h264_abort = controller;

            const response = await fetch(url, { signal: controller.signal });
            if (!response.ok) {
                console.warn(`[phone-camera] H.264 流 HTTP ${response.status}，降级到 MJPEG`);
                this._stopPhoneH264Stream();
                this._startPhoneMjpegStream();
                return;
            }

            const reader = response.body.getReader();
            this._phone_h264_reader = reader;

            let buffer = new Uint8Array(0);

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const newBuf = new Uint8Array(buffer.length + value.length);
                newBuf.set(buffer);
                newBuf.set(value, buffer.length);
                buffer = newBuf;

                while (buffer.length >= 4) {
                    const segLen = (buffer[0] << 24) | (buffer[1] << 16) | (buffer[2] << 8) | buffer[3];
                    if (segLen <= 0 || segLen > 10 * 1024 * 1024) {
                        buffer = buffer.slice(4);
                        continue;
                    }
                    if (buffer.length < 4 + segLen) break;

                    const segment = buffer.slice(4, 4 + segLen);
                    buffer = buffer.slice(4 + segLen);

                    this._phone_h264_append(segment);
                }
            }
        } catch (e) {
            if (e.name === 'AbortError') return;
            console.error('[phone-camera] H.264 流读取错误:', e);
            this._reconnectPhoneH264(info);
        }
    }

    _phone_h264_append(segmentData) {
        const sb = this._phone_h264_sb;
        if (!sb) {
            const ms = this._phone_h264_ms;
            if (!ms || ms.readyState !== 'open') return;

            const codecs = [
                'video/mp4; codecs="avc1.42E01E"',
                'video/mp4; codecs="hvc1.1.6.L93.B0"',
            ];

            let added = false;
            for (const codec of codecs) {
                if (MediaSource.isTypeSupported(codec)) {
                    try {
                        this._phone_h264_sb = ms.addSourceBuffer(codec);
                        this._phone_h264_sb.addEventListener('updateend', () => {
                            this._phone_h264_appending = false;
                            this._phone_h264_drain();
                        });
                        this._phone_h264_sb.mode = 'sequence';
                        added = true;
                        console.log(`[phone-camera] SourceBuffer 已创建: ${codec}`);
                        break;
                    } catch (e) {
                        console.warn(`[phone-camera] 创建 SourceBuffer 失败: ${codec}`, e);
                    }
                }
            }

            if (!added) {
                console.error('[phone-camera] 无可用 codec，降级到 MJPEG');
                this._stopPhoneH264Stream();
                this._startPhoneMjpegStream();
                return;
            }
        }

        this._phone_h264_queue.push(segmentData);
        this._phone_h264_drain();
    }

    _phone_h264_drain() {
        const sb = this._phone_h264_sb;
        if (!sb || sb.updating || this._phone_h264_appending) return;
        if (this._phone_h264_queue.length === 0) return;

        this._phone_h264_appending = true;
        const data = this._phone_h264_queue.shift();
        try {
            sb.appendBuffer(data);
        } catch (e) {
            console.error('[phone-camera] appendBuffer 失败:', e);
            this._phone_h264_appending = false;
            if (e.name === 'QuotaExceededError' && sb.buffered.length > 0) {
                try {
                    sb.remove(0, sb.buffered.end(0) - 0.5);
                } catch (_) {}
            }
        }
    }

    _reconnectPhoneH264(info) {
        if (!this.d.state.phoneCameraActive) return;
        console.log('[phone-camera] 3 秒后重连 H.264 流...');
        this._stopPhoneH264Stream();
        setTimeout(() => {
            if (this.d.state.phoneCameraActive) {
                this._startPhoneH264Stream();
            }
        }, 3000);
    }

    _stopPhoneH264Stream() {
        if (this._phone_h264_abort) {
            this._phone_h264_abort.abort();
            this._phone_h264_abort = null;
        }
        if (this._phone_h264_reader) {
            this._phone_h264_reader.cancel().catch(() => {});
            this._phone_h264_reader = null;
        }
        if (this._phone_h264_video) {
            this._phone_h264_video.src = '';
            this._phone_h264_video.remove();
            this._phone_h264_video = null;
        }
        if (this._phone_h264_ms) {
            try { URL.revokeObjectURL(this._phone_h264_ms); } catch (_) {}
            this._phone_h264_ms = null;
        }
        this._phone_h264_sb = null;
        this._phone_h264_queue = [];
        this._phone_h264_appending = false;
    }

    // ==================== MJPEG 流（兜底） ====================

    _startPhoneMjpegStream() {
        this._stopPhoneMjpegStream();

        const { dom } = this.d;
        const wrapper = dom.canvasWrapper;
        if (!wrapper) return;

        const info = window.phone_server_info?.();
        if (!info) {
            console.warn('[phone-camera] 服务信息不可用');
            return;
        }

        const img = document.createElement('img');
        img.id = 'phoneCameraStream';
        img.className = 'camera-video';
        img.src = `http://${info.ip}:${info.port}/camera/mjpeg`;

        img.onload = () => {
            console.log('[phone-camera] MJPEG 流已加载');
        };

        img.onerror = () => {
            console.warn('[phone-camera] MJPEG 流连接失败，3 秒后重连...');
            if (this.d.state.phoneCameraActive) {
                setTimeout(() => {
                    if (this.d.state.phoneCameraActive) {
                        this._startPhoneMjpegStream();
                    }
                }, 3000);
            }
        };

        wrapper.appendChild(img);
        this._phone_mjpeg_img = img;

        if (dom.cameraVideo) dom.cameraVideo.style.display = 'none';
    }

    _stopPhoneMjpegStream() {
        if (this._phone_mjpeg_img) {
            this._phone_mjpeg_img.src = '';
            this._phone_mjpeg_img.remove();
            this._phone_mjpeg_img = null;
        }
    }
}
