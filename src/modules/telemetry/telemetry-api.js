/**
 * HTTP 请求封装：reportOnline(), reportFieldValue()
 */

import {
    API_BASE,
    PLATFORM_ID,
    API_ENDPOINT_ONLINE,
    API_ENDPOINT_FIELD_VALUE
} from './telemetry-config.js';
import { getInstallUUID, getDeviceType } from './telemetry-identity.js';
import { getGeo } from './telemetry-geo.js';

const CAMERA_STORAGE_KEY = 'telemetry_camera_device';
const CPU_GPU_STORAGE_KEY = 'telemetry_cpu_gpu';

/**
 * 获取摄像头设备信息
 * 返回摄像头设备名称列表，若无摄像头则返回 null
 */
async function getCameraDevices() {
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const cameras = devices.filter(d => d.kind === 'videoinput');
        if (cameras.length === 0) return null;
        return cameras.map(c => c.label || `Camera ${cameras.indexOf(c) + 1}`).join(', ');
    } catch (e) {
        console.warn('[telemetry] camera devices fetch failed:', e);
        return null;
    }
}

/**
 * 获取 CPU 和 GPU 信息（通过 Tauri IPC）
 */
async function getCpuGpuInfo() {
    try {
        const invoke = window.__TAURI__?.core?.invoke;
        if (!invoke) return null;
        const info = await invoke('telemetry_fetch_cpu_gpu');
        return info || null;
    } catch (e) {
        console.warn('[telemetry] cpu/gpu fetch failed:', e);
        return null;
    }
}

/**
 * 检查摄像头是否需要上报（与上次上报的设备不同）
 */
function shouldReportCamera(cameraDevice) {
    if (!cameraDevice) return false;
    const lastReported = localStorage.getItem(CAMERA_STORAGE_KEY);
    return lastReported !== cameraDevice;
}

/**
 * 保存已上报的摄像头设备信息
 */
function saveReportedCamera(cameraDevice) {
    if (cameraDevice) {
        localStorage.setItem(CAMERA_STORAGE_KEY, cameraDevice);
    }
}

/**
 * 检查 CPU/GPU 是否需要上报（与上次上报的信息不同）
 */
function shouldReportCpuGpu(cpuGpuInfo) {
    if (!cpuGpuInfo) return false;
    const lastReported = localStorage.getItem(CPU_GPU_STORAGE_KEY);
    return lastReported !== cpuGpuInfo;
}

/**
 * 保存已上报的 CPU/GPU 信息
 */
function saveReportedCpuGpu(cpuGpuInfo) {
    if (cpuGpuInfo) {
        localStorage.setItem(CPU_GPU_STORAGE_KEY, cpuGpuInfo);
    }
}

/**
 * 上报自定义字段值（通过 Tauri IPC 绕过 CORS）
 * 返回 true 表示成功，false 表示失败
 */
export async function reportFieldValue(fieldKey, value, reportedBy = 'viewstage-client') {
    try {
        const body = {
            platform_id: PLATFORM_ID,
            field_key: fieldKey,
            value: value,
            source: 'api',
            reported_by: reportedBy
        };

        const invoke = window.__TAURI__?.core?.invoke;
        if (!invoke) {
            console.warn('[telemetry] Tauri IPC unavailable');
            return false;
        }

        const result = await invoke('telemetry_http_post', {
            url: `${API_BASE}${API_ENDPOINT_FIELD_VALUE}`,
            body: JSON.stringify(body)
        });

        console.log(`[telemetry] field value reported: ${fieldKey} = ${value}`);
        return true;
    } catch (e) {
        console.warn('[telemetry] field value report error:', e);
        return false;
    }
}

/**
 * 上报设备在线状态（通过 Tauri IPC 绕过 CORS）
 */
export async function reportOnline() {
    try {
        const installId = await getInstallUUID();
        const deviceType = getDeviceType();
        const geo = await getGeo();
        const cameraDevice = await getCameraDevices();
        const cpuGpuInfo = await getCpuGpuInfo();

        const body = {
            platform_id: PLATFORM_ID,
            device_uuid: installId,
            device_type: deviceType,
            ip_address: geo?.ip || '',
            country: geo?.country_name || '未知',
            province: geo?.region || '未知',
            city: geo?.city || '未知',
            district: geo?.district || geo?.city || '未知'
        };

        console.log('[telemetry] request body:', JSON.stringify(body, null, 2));

        const invoke = window.__TAURI__?.core?.invoke;
        if (!invoke) {
            console.warn('[telemetry] Tauri IPC unavailable');
            return;
        }

        const result = await invoke('telemetry_http_post', {
            url: `${API_BASE}${API_ENDPOINT_ONLINE}`,
            body: JSON.stringify(body)
        });

        const parsed = JSON.parse(result);
        console.log(`[telemetry] online reported, count: ${parsed.online_count}`);

        // 若有摄像头且与上次上报不同，则上报 camera_device 字段
        // 只有上传成功后才保存，失败时下次启动会继续尝试
        if (shouldReportCamera(cameraDevice)) {
            const success = await reportFieldValue('camera_device', cameraDevice);
            if (success) {
                saveReportedCamera(cameraDevice);
            }
        }

        // 若有 CPU/GPU 信息且与上次上报不同，则上报 cpu_and_gpu 字段
        if (shouldReportCpuGpu(cpuGpuInfo)) {
            const success = await reportFieldValue('cpu_and_gpu', cpuGpuInfo);
            if (success) {
                saveReportedCpuGpu(cpuGpuInfo);
            }
        }
    } catch (e) {
        console.warn('[telemetry] online report error:', e);
    }
}