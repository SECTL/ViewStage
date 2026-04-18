//! ViewStage - 图像处理 Rust 后端
//! 
//! 功能模块：
//! - 图像增强 (enhance_image): 对比度、亮度、饱和度调整
//! - 缩略图生成 (generate_thumbnail, generate_thumbnails_batch): 并行批量生成
//! - 图像旋转 (rotate_image): 90/180/270度旋转
//! - 图片保存 (save_image, save_images_batch): 保存到指定目录
//! - 笔画压缩 (compact_strokes): 将笔画渲染到图片
//! - 设置管理 (get_settings, save_settings): 应用配置持久化
//! - 摄像头管理 (get_camera_list, set_camera_state): 设备枚举与状态
//!
//! 性能优化：
//! - 使用 rayon 并行处理像素
//! - 使用 base64 编码传输数据
//! - 使用 image 库进行图像处理

use tauri::{Manager, Emitter};
use image::{DynamicImage, ImageBuffer, Rgba, RgbaImage, GrayImage, Luma, Rgb, GenericImageView};
use imageproc::filter::gaussian_blur_f32;
use base64::{Engine as _, engine::general_purpose};

mod image_processing;

use image_processing::{
    decode_base64_image, extract_base64,
    enhance_image, generate_thumbnail, generate_thumbnails_batch, rotate_image,
    apply_enhance_filter,
};

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;

#[cfg(target_os = "windows")]
use opencv::{
    core::{Mat, Vector, Size, Scalar, CV_32F, CV_8UC3, CV_8UC1, Point, Point2f, BorderTypes},
    dnn::{read_net_from_onnx, Net, blob_from_image},
    imgproc::{
        resize, cvt_color, COLOR_BGR2GRAY, COLOR_BGR2Lab, COLOR_Lab2BGR,
        threshold, find_contours, contour_area,
        hough_lines_p, get_perspective_transform, warp_perspective,
        THRESH_BINARY, THRESH_OTSU, INTER_LINEAR,
    },
    photo::fast_nl_means_denoising,
    prelude::*,
};

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

// ==================== 模型缓存 ====================

/// 全局 DexiNed 模型缓存
static DEXINED_CACHE: once_cell::sync::Lazy<Mutex<Option<Arc<Mutex<ort::session::Session>>>>> = 
    once_cell::sync::Lazy::new(|| Mutex::new(None));
static DEXINED_MODEL_PATH: once_cell::sync::Lazy<Mutex<Option<PathBuf>>> = 
    once_cell::sync::Lazy::new(|| Mutex::new(None));

/// 全局 UVDoc 模型缓存
static UVDOC_CACHE: once_cell::sync::Lazy<Mutex<Option<Arc<Mutex<ort::session::Session>>>>> = 
    once_cell::sync::Lazy::new(|| Mutex::new(None));
static UVDOC_MODEL_PATH: once_cell::sync::Lazy<Mutex<Option<PathBuf>>> = 
    once_cell::sync::Lazy::new(|| Mutex::new(None));

/// 全局 GCNet 模型缓存
static GCNET_CACHE: once_cell::sync::Lazy<Mutex<Option<Arc<Mutex<ort::session::Session>>>>> = 
    once_cell::sync::Lazy::new(|| Mutex::new(None));
static GCNET_MODEL_PATH: once_cell::sync::Lazy<Mutex<Option<PathBuf>>> = 
    once_cell::sync::Lazy::new(|| Mutex::new(None));

/// 全局 NAFDPM 模型缓存
static NAFDPM_CACHE: once_cell::sync::Lazy<Mutex<Option<Arc<Mutex<ort::session::Session>>>>> = 
    once_cell::sync::Lazy::new(|| Mutex::new(None));
static NAFDPM_MODEL_PATH: once_cell::sync::Lazy<Mutex<Option<PathBuf>>> = 
    once_cell::sync::Lazy::new(|| Mutex::new(None));

/// 全局 DBNet (ONNX Runtime) 模型缓存
static DBNET_ORT_CACHE: once_cell::sync::Lazy<Mutex<Option<Arc<Mutex<ort::session::Session>>>>> = 
    once_cell::sync::Lazy::new(|| Mutex::new(None));
static DBNET_ORT_MODEL_PATH: once_cell::sync::Lazy<Mutex<Option<PathBuf>>> = 
    once_cell::sync::Lazy::new(|| Mutex::new(None));

/// 获取或加载 DexiNed 模型
fn get_or_load_dexined_model(model_path: &std::path::Path) -> Result<Arc<Mutex<ort::session::Session>>, String> {
    {
        let cached_path = DEXINED_MODEL_PATH.lock().unwrap();
        if let Some(ref path) = *cached_path {
            if path == model_path {
                let cached_session = DEXINED_CACHE.lock().unwrap();
                if let Some(ref session) = *cached_session {
                    log::info!("使用缓存的 DexiNed 模型");
                    return Ok(session.clone());
                }
            }
        }
    }
    
    log::info!("加载 DexiNed 模型: {:?}", model_path);
    
    use ort::session::Session;
    
    let session = Session::builder()
        .map_err(|e| format!("创建 Session builder 失败: {}", e))?
        .with_execution_providers(&[ort::ep::DirectMLExecutionProvider::default().into()])
        .map_err(|e| format!("设置 DirectML 执行提供器失败: {}", e))?
        .commit_from_file(model_path)
        .map_err(|e| format!("加载 DexiNed ONNX 模型失败: {}", e))?;
    
    log::info!("DexiNed 模型加载成功（DirectML GPU 加速）");
    
    let session = Arc::new(Mutex::new(session));
    
    {
        let mut cached_session = DEXINED_CACHE.lock().unwrap();
        *cached_session = Some(session.clone());
    }
    {
        let mut cached_path = DEXINED_MODEL_PATH.lock().unwrap();
        *cached_path = Some(model_path.to_path_buf());
    }
    
    Ok(session)
}

/// 获取或加载 UVDoc 模型
fn get_or_load_uvdoc_model(model_path: &std::path::Path) -> Result<Arc<Mutex<ort::session::Session>>, String> {
    {
        let cached_path = UVDOC_MODEL_PATH.lock().unwrap();
        if let Some(ref path) = *cached_path {
            if path == model_path {
                let cached_session = UVDOC_CACHE.lock().unwrap();
                if let Some(ref session) = *cached_session {
                    log::info!("使用缓存的 UVDoc 模型");
                    return Ok(session.clone());
                }
            }
        }
    }
    
    log::info!("加载 UVDoc 模型: {:?}", model_path);
    
    use ort::session::Session;
    
    let session = Session::builder()
        .map_err(|e| format!("创建 Session builder 失败: {}", e))?
        .commit_from_file(model_path)
        .map_err(|e| format!("加载 UVDoc ONNX 模型失败: {}", e))?;
    
    log::info!("UVDoc 模型加载成功");
    
    let session = Arc::new(Mutex::new(session));
    
    {
        let mut cached_session = UVDOC_CACHE.lock().unwrap();
        *cached_session = Some(session.clone());
    }
    {
        let mut cached_path = UVDOC_MODEL_PATH.lock().unwrap();
        *cached_path = Some(model_path.to_path_buf());
    }
    
    Ok(session)
}

/// 获取或加载 GCNet 模型
fn get_or_load_gcnet_model(model_path: &std::path::Path) -> Result<Arc<Mutex<ort::session::Session>>, String> {
    {
        let cached_path = GCNET_MODEL_PATH.lock().unwrap();
        if let Some(ref path) = *cached_path {
            if path == model_path {
                let cached_session = GCNET_CACHE.lock().unwrap();
                if let Some(ref session) = *cached_session {
                    log::info!("使用缓存的 GCNet 模型");
                    return Ok(session.clone());
                }
            }
        }
    }
    
    log::info!("加载 GCNet 模型: {:?}", model_path);
    
    use ort::session::Session;
    
    let session = Session::builder()
        .map_err(|e| format!("创建 Session builder 失败: {}", e))?
        .commit_from_file(model_path)
        .map_err(|e| format!("加载 GCNet ONNX 模型失败: {}", e))?;
    
    log::info!("GCNet 模型加载成功");
    
    let session = Arc::new(Mutex::new(session));
    
    {
        let mut cached_session = GCNET_CACHE.lock().unwrap();
        *cached_session = Some(session.clone());
    }
    {
        let mut cached_path = GCNET_MODEL_PATH.lock().unwrap();
        *cached_path = Some(model_path.to_path_buf());
    }
    
    Ok(session)
}

/// 获取或加载 NAFDPM 模型
fn get_or_load_nafdpm_model(model_path: &std::path::Path) -> Result<Arc<Mutex<ort::session::Session>>, String> {
    {
        let cached_path = NAFDPM_MODEL_PATH.lock().unwrap();
        if let Some(ref path) = *cached_path {
            if path == model_path {
                let cached_session = NAFDPM_CACHE.lock().unwrap();
                if let Some(ref session) = *cached_session {
                    log::info!("使用缓存的 NAFDPM 模型");
                    return Ok(session.clone());
                }
            }
        }
    }
    
    log::info!("加载 NAFDPM 模型: {:?}", model_path);
    
    use ort::session::Session;
    
    let session = Session::builder()
        .map_err(|e| format!("创建 Session builder 失败: {}", e))?
        .commit_from_file(model_path)
        .map_err(|e| format!("加载 NAFDPM ONNX 模型失败: {}", e))?;
    
    log::info!("NAFDPM 模型加载成功");
    
    let session = Arc::new(Mutex::new(session));
    
    {
        let mut cached_session = NAFDPM_CACHE.lock().unwrap();
        *cached_session = Some(session.clone());
    }
    {
        let mut cached_path = NAFDPM_MODEL_PATH.lock().unwrap();
        *cached_path = Some(model_path.to_path_buf());
    }
    
    Ok(session)
}

/// 获取或加载 DBNet (ONNX Runtime) 模型
fn get_or_load_dbnet_ort_model(model_path: &std::path::Path) -> Result<Arc<Mutex<ort::session::Session>>, String> {
    {
        let cached_path = DBNET_ORT_MODEL_PATH.lock().unwrap();
        if let Some(ref path) = *cached_path {
            if path == model_path {
                let cached_session = DBNET_ORT_CACHE.lock().unwrap();
                if let Some(ref session) = *cached_session {
                    log::info!("使用缓存的 DBNet 模型");
                    return Ok(session.clone());
                }
            }
        }
    }
    
    log::info!("加载 DBNet 模型: {:?}", model_path);
    
    use ort::session::Session;
    
    let session = Session::builder()
        .map_err(|e| format!("创建 Session builder 失败: {}", e))?
        .commit_from_file(model_path)
        .map_err(|e| format!("加载 DBNet ONNX 模型失败: {}", e))?;
    
    log::info!("DBNet 模型加载成功");
    
    let session = Arc::new(Mutex::new(session));
    
    {
        let mut cached_session = DBNET_ORT_CACHE.lock().unwrap();
        *cached_session = Some(session.clone());
    }
    {
        let mut cached_path = DBNET_ORT_MODEL_PATH.lock().unwrap();
        *cached_path = Some(model_path.to_path_buf());
    }
    
    Ok(session)
}

/// 清空所有模型缓存
#[tauri::command]
fn clear_model_cache() -> Result<(), String> {
    log::info!("清空所有模型缓存");
    
    {
        let mut cache = DEXINED_CACHE.lock().unwrap();
        *cache = None;
    }
    {
        let mut path = DEXINED_MODEL_PATH.lock().unwrap();
        *path = None;
    }
    {
        let mut cache = UVDOC_CACHE.lock().unwrap();
        *cache = None;
    }
    {
        let mut path = UVDOC_MODEL_PATH.lock().unwrap();
        *path = None;
    }
    {
        let mut cache = GCNET_CACHE.lock().unwrap();
        *cache = None;
    }
    {
        let mut path = GCNET_MODEL_PATH.lock().unwrap();
        *path = None;
    }
    {
        let mut cache = NAFDPM_CACHE.lock().unwrap();
        *cache = None;
    }
    {
        let mut path = NAFDPM_MODEL_PATH.lock().unwrap();
        *path = None;
    }
    {
        let mut cache = DBNET_ORT_CACHE.lock().unwrap();
        *cache = None;
    }
    {
        let mut path = DBNET_ORT_MODEL_PATH.lock().unwrap();
        *path = None;
    }
    
    log::info!("模型缓存已清空");
    Ok(())
}

// ==================== 数据结构 ====================
// 用于前后端通信的结构体定义

/// 图片保存结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageSaveResult {
    pub path: String,                    // 保存路径
    pub success: bool,                   // 是否成功
    pub error: Option<String>,           // 错误信息
    pub enhanced_data: Option<String>,   // 增强后的图片数据 (base64)
}

/// 笔画点 (线段)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrokePoint {
    pub from_x: f32,
    pub from_y: f32,
    pub to_x: f32,
    pub to_y: f32,
}

/// 笔画 (绘制或擦除)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stroke {
    #[serde(rename = "type")]
    pub stroke_type: String,            // "draw" 或 "erase"
    pub points: Vec<StrokePoint>,       // 线段点集合
    pub color: Option<String>,          // 颜色 (#RRGGBB)
    pub line_width: Option<u32>,        // 线宽
    pub eraser_size: Option<u32>,       // 橡皮大小
}

/// 笔画压缩请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactStrokesRequest {
    pub base_image: Option<String>,     // 基础图片 (base64)
    pub strokes: Vec<Stroke>,           // 待压缩笔画
    pub canvas_width: u32,              // 画布宽度
    pub canvas_height: u32,             // 画布高度
}

// ==================== 系统目录 ====================
// 获取应用缓存目录、配置目录、ViewStage目录

/// 获取应用缓存目录
#[tauri::command]
fn get_cache_dir(app: tauri::AppHandle) -> Result<String, String> {
    let config_dir = app.path().app_config_dir()
        .map_err(|e| format!("Failed to get config dir: {}", e))?;
    
    let cache_dir = config_dir.join("cache");
    
    if !cache_dir.exists() {
        std::fs::create_dir_all(&cache_dir)
            .map_err(|e| format!("Failed to create cache dir: {}", e))?;
    }
    
    Ok(cache_dir.to_string_lossy().to_string())
}

/// 获取缓存大小
#[tauri::command]
fn get_cache_size(app: tauri::AppHandle) -> Result<u64, String> {
    let config_dir = app.path().app_config_dir()
        .map_err(|e| format!("Failed to get config dir: {}", e))?;
    
    let cache_dir = config_dir.join("cache");
    
    if !cache_dir.exists() {
        return Ok(0);
    }
    
    fn dir_size(path: &std::path::Path) -> u64 {
        let mut size = 0;
        if path.is_dir() {
            if let Ok(entries) = std::fs::read_dir(path) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        size += dir_size(&path);
                    } else {
                        size += entry.metadata().map(|m| m.len()).unwrap_or(0);
                    }
                }
            }
        }
        size
    }
    
    Ok(dir_size(&cache_dir))
}

/// 清除缓存
#[tauri::command]
fn clear_cache(app: tauri::AppHandle) -> Result<String, String> {
    let config_dir = app.path().app_config_dir()
        .map_err(|e| format!("Failed to get config dir: {}", e))?;
    
    let cache_dir = config_dir.join("cache");
    
    if !cache_dir.exists() {
        return Ok("缓存目录不存在".to_string());
    }
    
    fn remove_dir_contents(path: &std::path::Path) -> (u64, u32) {
        let mut size = 0u64;
        let mut count = 0u32;
        
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                let entry_path = entry.path();
                if entry_path.is_dir() {
                    let (s, c) = remove_dir_contents(&entry_path);
                    size += s;
                    count += c;
                    let _ = std::fs::remove_dir(&entry_path);
                } else {
                    size += entry.metadata().map(|m| m.len()).unwrap_or(0);
                    if std::fs::remove_file(&entry_path).is_ok() {
                        count += 1;
                    }
                }
            }
        }
        (size, count)
    }
    
    let (cleared_size, cleared_files) = remove_dir_contents(&cache_dir);
    
    log::info!("清除缓存: {} 字节, {} 个文件", cleared_size, cleared_files);
    
    Ok(format!("已清除 {} 个文件，共 {:.2} MB", cleared_files, cleared_size as f64 / 1024.0 / 1024.0))
}

/// 检查并执行自动清除缓存
#[tauri::command]
fn check_auto_clear_cache(app: tauri::AppHandle) -> Result<bool, String> {
    let config_dir = app.path().app_config_dir()
        .map_err(|e| format!("Failed to get config dir: {}", e))?;
    
    let config_file = config_dir.join("config.json");
    
    if !config_file.exists() {
        return Ok(false);
    }
    
    let config_content = std::fs::read_to_string(&config_file)
        .map_err(|e| format!("Failed to read config: {}", e))?;
    
    let config: serde_json::Value = serde_json::from_str(&config_content)
        .map_err(|e| format!("Failed to parse config: {}", e))?;
    
    let auto_clear_days = config.get("autoClearCacheDays")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    
    if auto_clear_days == 0 {
        log::info!("自动清除缓存已关闭");
        return Ok(false);
    }
    
    let last_clear_date = config.get("lastCacheClearDate")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    
    let today = chrono::Local::now().format("%Y-%m-%d").to_string();
    
    if last_clear_date == today {
        log::info!("今日已执行过自动清除缓存");
        return Ok(false);
    }
    
    if last_clear_date.is_empty() {
        let mut updated_config = config.clone();
        updated_config["lastCacheClearDate"] = serde_json::json!(today);
        let updated_content = serde_json::to_string_pretty(&updated_config)
            .map_err(|e| format!("Failed to serialize config: {}", e))?;
        std::fs::write(&config_file, updated_content)
            .map_err(|e| format!("Failed to write config: {}", e))?;
        log::info!("首次设置自动清除缓存日期");
        return Ok(false);
    }
    
    let last_date = chrono::NaiveDate::parse_from_str(last_clear_date, "%Y-%m-%d")
        .map_err(|e| format!("Failed to parse last clear date: {}", e))?;
    let today_date = chrono::Local::now().date_naive();
    
    let days_since_last_clear = (today_date - last_date).num_days();
    
    if days_since_last_clear >= auto_clear_days as i64 {
        log::info!("执行自动清除缓存，距上次清除 {} 天", days_since_last_clear);
        
        let cache_dir = config_dir.join("cache");
        
        if cache_dir.exists() {
            fn remove_dir_contents(path: &std::path::Path) {
                if let Ok(entries) = std::fs::read_dir(path) {
                    for entry in entries.flatten() {
                        let entry_path = entry.path();
                        if entry_path.is_dir() {
                            remove_dir_contents(&entry_path);
                            let _ = std::fs::remove_dir(&entry_path);
                        } else {
                            let _ = std::fs::remove_file(&entry_path);
                        }
                    }
                }
            }
            remove_dir_contents(&cache_dir);
        }
        
        let mut updated_config = config.clone();
        updated_config["lastCacheClearDate"] = serde_json::json!(today);
        let updated_content = serde_json::to_string_pretty(&updated_config)
            .map_err(|e| format!("Failed to serialize config: {}", e))?;
        std::fs::write(&config_file, updated_content)
            .map_err(|e| format!("Failed to write config: {}", e))?;
        
        log::info!("自动清除缓存完成");
        return Ok(true);
    }
    
    Ok(false)
}

/// 获取应用配置目录
#[tauri::command]
fn get_config_dir(app: tauri::AppHandle) -> Result<String, String> {
    let config_dir = app.path().app_config_dir()
        .map_err(|e| format!("Failed to get config dir: {}", e))?;
    
    if !config_dir.exists() {
        std::fs::create_dir_all(&config_dir)
            .map_err(|e| format!("Failed to create config dir: {}", e))?;
    }
    
    Ok(config_dir.to_string_lossy().to_string())
}

/// 获取图片保存目录 (~/Pictures/ViewStage)
#[tauri::command]
fn get_cds_dir() -> Result<String, String> {
    let pictures_dir = dirs::picture_dir()
        .ok_or("Failed to get pictures directory")?;
    
    let cds_dir = pictures_dir.join("ViewStage");
    
    if !cds_dir.exists() {
        std::fs::create_dir_all(&cds_dir)
            .map_err(|e| format!("Failed to create ViewStage dir: {}", e))?;
    }
    
    Ok(cds_dir.to_string_lossy().to_string())
}

/// 获取用户主题目录 (%APPDATA%/SECTL/ViewStage/themes)
#[tauri::command]
fn get_theme_dir(app: tauri::AppHandle) -> Result<String, String> {
    let config_dir = app.path().app_config_dir()
        .map_err(|e| format!("Failed to get config dir: {}", e))?;
    
    let theme_dir = config_dir.join("themes");
    
    if !theme_dir.exists() {
        std::fs::create_dir_all(&theme_dir)
            .map_err(|e| format!("Failed to create theme dir: {}", e))?;
    }
    
    Ok(theme_dir.to_string_lossy().to_string())
}

/// 获取模型文件目录 (%APPDATA%/SECTL/ViewStage/models)
fn get_models_dir(app: &tauri::AppHandle) -> Result<std::path::PathBuf, String> {
    let config_dir = app.path().app_config_dir()
        .map_err(|e| format!("获取配置目录失败: {}", e))?;
    
    let models_dir = config_dir.join("models");
    
    if !models_dir.exists() {
        std::fs::create_dir_all(&models_dir)
            .map_err(|e| format!("创建模型目录失败: {}", e))?;
    }
    
    Ok(models_dir)
}

// ==================== 图片保存 ====================

/// 生成保存路径
/// - 按日期创建子目录: YYYY-MM-DD
/// - 文件名格式: {prefix}_HH-MM-SS-SSS.{extension}
fn get_save_path(base_dir: &str, prefix: &str, extension: &str) -> Result<(PathBuf, String), String> {
    use std::time::{SystemTime, UNIX_EPOCH};
    
    let now = chrono::Local::now();
    let date_str = now.format("%Y-%m-%d").to_string();
    let time_str = now.format("%H-%M-%S").to_string();
    
    let date_dir = PathBuf::from(base_dir).join(&date_str);
    
    if !date_dir.exists() {
        std::fs::create_dir_all(&date_dir)
            .map_err(|e| format!("Failed to create date directory: {}", e))?;
    }
    
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| format!("Failed to get timestamp: {}", e))?
        .subsec_millis();
    
    let file_name = format!("{}_{}-{:03}.{}", prefix, time_str, timestamp, extension);
    let file_path = date_dir.join(&file_name);
    
    Ok((file_path, file_name))
}

fn sanitize_prefix(prefix: &str) -> String {
    let sanitized: String = prefix
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-')
        .collect();
    if sanitized.is_empty() { "photo".to_string() } else { sanitized }
}

#[tauri::command]
fn save_image(image_data: String, prefix: Option<String>) -> Result<ImageSaveResult, String> {
    let base_dir = get_cds_dir()?;
    let prefix_str = sanitize_prefix(&prefix.unwrap_or_else(|| "photo".to_string()));
    
    let decoded = extract_base64(&image_data)?;
    
    let extension = if image_data.contains("image/png") {
        "png"
    } else if image_data.contains("image/jpeg") || image_data.contains("image/jpg") {
        "jpg"
    } else {
        "png"
    };
    
    let (file_path, _file_name) = get_save_path(&base_dir, &prefix_str, extension)?;
    
    std::fs::write(&file_path, &decoded)
        .map_err(|e| format!("Failed to write image file: {}", e))?;
    
    Ok(ImageSaveResult {
        path: file_path.to_string_lossy().to_string(),
        success: true,
        error: None,
        enhanced_data: None,
    })
}

#[tauri::command]
fn save_image_with_enhance(image_data: String, prefix: Option<String>, contrast: f32, brightness: f32, saturation: f32, sharpen: f32) -> Result<ImageSaveResult, String> {
    let base_dir = get_cds_dir()?;
    let prefix_str = sanitize_prefix(&prefix.unwrap_or_else(|| "photo".to_string()));
    
    let img = decode_base64_image(&image_data)?;
    
    let enhanced = apply_enhance_filter(&img, contrast, brightness, saturation, sharpen);
    
    let mut buffer = Vec::new();
    enhanced
        .write_to(&mut std::io::Cursor::new(&mut buffer), image::ImageFormat::Png)
        .map_err(|e| format!("Failed to encode enhanced image: {}", e))?;
    
    let (file_path, _file_name) = get_save_path(&base_dir, &prefix_str, "png")?;
    
    std::fs::write(&file_path, &buffer)
        .map_err(|e| format!("Failed to write enhanced image file: {}", e))?;
    
    let enhanced_data = format!("data:image/png;base64,{}", general_purpose::STANDARD.encode(&buffer));
    
    Ok(ImageSaveResult {
        path: file_path.to_string_lossy().to_string(),
        success: true,
        error: None,
        enhanced_data: Some(enhanced_data),
    })
}

// ==================== 笔画压缩 ====================
// 将笔画渲染到图片，用于撤销功能

/// 解析颜色字符串为 RGBA
/// 支持格式: #RRGGBB 或 #RRGGBBAA
fn parse_color(color_str: &str) -> Result<Rgba<u8>, String> {
    if !color_str.starts_with('#') {
        return Err(format!("Invalid color format: must start with '#', got: {}", color_str));
    }
    
    match color_str.len() {
        7 => {
            let r = u8::from_str_radix(&color_str[1..3], 16)
                .map_err(|_| format!("Invalid red component in color: {}", color_str))?;
            let g = u8::from_str_radix(&color_str[3..5], 16)
                .map_err(|_| format!("Invalid green component in color: {}", color_str))?;
            let b = u8::from_str_radix(&color_str[5..7], 16)
                .map_err(|_| format!("Invalid blue component in color: {}", color_str))?;
            Ok(Rgba([r, g, b, 255]))
        }
        9 => {
            let r = u8::from_str_radix(&color_str[1..3], 16)
                .map_err(|_| format!("Invalid red component in color: {}", color_str))?;
            let g = u8::from_str_radix(&color_str[3..5], 16)
                .map_err(|_| format!("Invalid green component in color: {}", color_str))?;
            let b = u8::from_str_radix(&color_str[5..7], 16)
                .map_err(|_| format!("Invalid blue component in color: {}", color_str))?;
            let a = u8::from_str_radix(&color_str[7..9], 16)
                .map_err(|_| format!("Invalid alpha component in color: {}", color_str))?;
            Ok(Rgba([r, g, b, a]))
        }
        _ => Err(format!("Invalid color format: expected #RRGGBB or #RRGGBBAA, got: {}", color_str))
    }
}

const DEFAULT_COLOR: Rgba<u8> = Rgba([52, 152, 219, 255]);

fn draw_line_on_canvas(canvas: &mut RgbaImage, x1: i32, y1: i32, x2: i32, y2: i32, color: Rgba<u8>, width: u32) {
    let dx = (x2 - x1).abs();
    let dy = (y2 - y1).abs();
    let sx = if x1 < x2 { 1 } else { -1 };
    let sy = if y1 < y2 { 1 } else { -1 };
    let mut err = dx - dy;
    let mut x = x1;
    let mut y = y1;
    
    let half_width = (width / 2) as i32;
    
    loop {
        for wx in -half_width..=half_width {
            for wy in -half_width..=half_width {
                let px = x + wx;
                let py = y + wy;
                if px >= 0 && py >= 0 && (px as u32) < canvas.width() && (py as u32) < canvas.height() {
                    let dist = ((wx * wx + wy * wy) as f32).sqrt();
                    if dist <= half_width as f32 {
                        let pixel = canvas.get_pixel_mut(px as u32, py as u32);
                        if color[3] == 255 {
                            *pixel = color;
                        } else {
                            let alpha = color[3] as f32 / 255.0;
                            let inv_alpha = 1.0 - alpha;
                            pixel[0] = (color[0] as f32 * alpha + pixel[0] as f32 * inv_alpha) as u8;
                            pixel[1] = (color[1] as f32 * alpha + pixel[1] as f32 * inv_alpha) as u8;
                            pixel[2] = (color[2] as f32 * alpha + pixel[2] as f32 * inv_alpha) as u8;
                        }
                    }
                }
            }
        }
        
        if x == x2 && y == y2 {
            break;
        }
        
        let e2 = 2 * err;
        if e2 > -dy {
            err -= dy;
            x += sx;
        }
        if e2 < dx {
            err += dx;
            y += sy;
        }
    }
}

fn erase_line_on_canvas(canvas: &mut RgbaImage, x1: i32, y1: i32, x2: i32, y2: i32, width: u32) {
    let dx = (x2 - x1).abs();
    let dy = (y2 - y1).abs();
    let sx = if x1 < x2 { 1 } else { -1 };
    let sy = if y1 < y2 { 1 } else { -1 };
    let mut err = dx - dy;
    let mut x = x1;
    let mut y = y1;
    
    let half_width = (width / 2) as i32;
    
    loop {
        for wx in -half_width..=half_width {
            for wy in -half_width..=half_width {
                let px = x + wx;
                let py = y + wy;
                if px >= 0 && py >= 0 && (px as u32) < canvas.width() && (py as u32) < canvas.height() {
                    let dist = ((wx * wx + wy * wy) as f32).sqrt();
                    if dist <= half_width as f32 {
                        let pixel = canvas.get_pixel_mut(px as u32, py as u32);
                        pixel[3] = 0;
                    }
                }
            }
        }
        
        if x == x2 && y == y2 {
            break;
        }
        
        let e2 = 2 * err;
        if e2 > -dy {
            err -= dy;
            x += sx;
        }
        if e2 < dx {
            err += dx;
            y += sy;
        }
    }
}

#[tauri::command]
fn compact_strokes(request: CompactStrokesRequest) -> Result<String, String> {
    let mut canvas: RgbaImage = ImageBuffer::new(request.canvas_width, request.canvas_height);
    
    for pixel in canvas.pixels_mut() {
        *pixel = Rgba([0, 0, 0, 0]);
    }
    
    if let Some(base_image_data) = request.base_image {
        if let Ok(base_img) = decode_base64_image(&base_image_data) {
            let base_rgba = base_img.to_rgba8();
            for (x, y, pixel) in base_rgba.enumerate_pixels() {
                if x < canvas.width() && y < canvas.height() {
                    canvas.put_pixel(x, y, *pixel);
                }
            }
        }
    }
    
    for stroke in &request.strokes {
        let points = &stroke.points;
        
        if stroke.stroke_type == "clear" {
            for pixel in canvas.pixels_mut() {
                *pixel = Rgba([0, 0, 0, 0]);
            }
            continue;
        }
        
        if points.is_empty() {
            continue;
        }
        
        if stroke.stroke_type == "draw" {
            let color = parse_color(stroke.color.as_deref().unwrap_or("#3498db"))
                .unwrap_or(DEFAULT_COLOR);
            let line_width = stroke.line_width.unwrap_or(2);
            
            for point in points {
                draw_line_on_canvas(
                    &mut canvas,
                    point.from_x as i32,
                    point.from_y as i32,
                    point.to_x as i32,
                    point.to_y as i32,
                    color,
                    line_width,
                );
            }
        } else if stroke.stroke_type == "erase" {
            let eraser_size = stroke.eraser_size.unwrap_or(15);
            
            for point in points {
                erase_line_on_canvas(
                    &mut canvas,
                    point.from_x as i32,
                    point.from_y as i32,
                    point.to_x as i32,
                    point.to_y as i32,
                    eraser_size,
                );
            }
        }
    }
    
    let mut buffer = Vec::new();
    DynamicImage::ImageRgba8(canvas)
        .write_to(&mut std::io::Cursor::new(&mut buffer), image::ImageFormat::Png)
        .map_err(|e| format!("Failed to encode compacted image: {}", e))?;
    
    Ok(format!("data:image/png;base64,{}", general_purpose::STANDARD.encode(&buffer)))
}

// ==================== 全局状态 ====================
// 镜像、增强等全局状态，使用原子类型保证线程安全

use std::sync::atomic::{AtomicBool, Ordering};

static MIRROR_STATE: AtomicBool = AtomicBool::new(false);
static ENHANCE_STATE: AtomicBool = AtomicBool::new(false);
static OOBE_ACTIVE: AtomicBool = AtomicBool::new(false);

// ==================== 设置窗口 ====================
// 打开设置窗口、状态同步

#[tauri::command]
async fn open_settings_window(app: tauri::AppHandle) -> Result<(), String> {
    use tauri::WebviewWindowBuilder;
    
    if let Some(window) = app.get_webview_window("settings") {
        window.set_focus().map_err(|e| format!("Failed to focus settings window: {}", e))?;
        return Ok(());
    }
    
    let window = WebviewWindowBuilder::new(
        &app,
        "settings",
        tauri::WebviewUrl::App("settings.html".into())
    )
    .title("设置")
    .inner_size(600.0, 600.0)
    .resizable(false)
    .decorations(false)
    .always_on_top(true)
    .center()
    .build()
    .map_err(|e| format!("Failed to create settings window: {}", e))?;
    
    window.set_focus().map_err(|e| format!("Failed to focus new settings window: {}", e))?;
    
    Ok(())
}

#[tauri::command]
async fn rotate_main_image(app: tauri::AppHandle, direction: String) -> Result<(), String> {
    let _ = app.emit("rotate-image", direction.clone());
    Ok(())
}

#[tauri::command]
async fn set_mirror_state(enabled: bool, app: tauri::AppHandle) -> Result<(), String> {
    MIRROR_STATE.store(enabled, Ordering::SeqCst);
    let _ = app.emit("mirror-changed", enabled);
    Ok(())
}

#[tauri::command]
async fn get_mirror_state() -> Result<bool, String> {
    Ok(MIRROR_STATE.load(Ordering::SeqCst))
}

#[tauri::command]
async fn set_enhance_state(enabled: bool, app: tauri::AppHandle) -> Result<(), String> {
    ENHANCE_STATE.store(enabled, Ordering::SeqCst);
    let _ = app.emit("enhance-changed", enabled);
    Ok(())
}

#[tauri::command]
async fn get_enhance_state() -> Result<bool, String> {
    Ok(ENHANCE_STATE.load(Ordering::SeqCst))
}

#[tauri::command]
async fn switch_camera(app: tauri::AppHandle) -> Result<(), String> {
    let _ = app.emit("switch-camera", ());
    Ok(())
}

#[tauri::command]
fn get_app_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GitHubRelease {
    tag_name: String,
    name: Option<String>,
    html_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct UpdateCheckResult {
    has_update: bool,
    current_version: String,
    latest_version: String,
    release: Option<GitHubRelease>,
}

fn parse_version(version: &str) -> Option<(u32, u32, u32)> {
    let version = version.trim_start_matches('v');
    let parts: Vec<&str> = version.split('.').collect();
    
    if parts.len() >= 3 {
        let major = parts[0].parse::<u32>().ok()?;
        let minor = parts[1].parse::<u32>().ok()?;
        let patch = parts[2].parse::<u32>().ok()?;
        return Some((major, minor, patch));
    }
    None
}

fn is_newer_version(current: &str, latest: &str) -> bool {
    let current_ver = parse_version(current);
    let latest_ver = parse_version(latest);
    
    match (current_ver, latest_ver) {
        (Some(c), Some(l)) => l > c,
        _ => false,
    }
}

fn validate_github_url(url: &str) -> Result<(), String> {
    let parsed = url::Url::parse(url).map_err(|e| format!("Invalid URL: {}", e))?;
    
    let valid_domains = ["github.com", "www.github.com", "api.github.com"];
    let host = parsed.host_str().unwrap_or("");
    
    if !valid_domains.contains(&host) {
        return Err(format!("Invalid GitHub URL: unexpected domain {}", host));
    }
    
    Ok(())
}

#[tauri::command]
async fn check_update() -> Result<UpdateCheckResult, String> {
    let current_version = env!("CARGO_PKG_VERSION");
    
    let client = reqwest::Client::builder()
        .user_agent("ViewStage")
        .timeout(std::time::Duration::from_secs(10))
        .https_only(true)
        .build()
        .map_err(|e| e.to_string())?;
    
    let response = client
        .get("https://api.github.com/repos/ospneam/ViewStage/releases/latest")
        .send()
        .await
        .map_err(|e| format!("Network error: {}", e))?;
    
    if !response.status().is_success() {
        return Err(format!("GitHub API error: {}", response.status()));
    }
    
    let release: GitHubRelease = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse response: {}", e))?;
    
    if release.tag_name.is_empty() {
        return Err("Invalid release: empty tag name".to_string());
    }
    
    validate_github_url(&release.html_url)?;
    
    let latest_version = release.tag_name.trim_start_matches('v');
    let has_update = is_newer_version(current_version, latest_version);
    
    Ok(UpdateCheckResult {
        has_update,
        current_version: current_version.to_string(),
        latest_version: latest_version.to_string(),
        release: if has_update { Some(release) } else { None },
    })
}

fn get_default_config() -> serde_json::Value {
    serde_json::json!({
        "width": 1920,
        "height": 1080,
        "language": "zh-CN",
        "defaultCamera": "",
        "cameraWidth": 1280,
        "cameraHeight": 720,
        "moveFps": 30,
        "drawFps": 10,
        "pdfScale": 1.5,
        "defaultRotation": 0,
        "contrast": 1.4,
        "brightness": 10,
        "saturation": 1.2,
        "sharpen": 0,
        "canvasScale": 2,
        "dprLimit": 2,
        "highFrameRate": false,
        "smoothStrength": 0.5,
        "blurEffect": true,
        "penColors": [
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
        "fileAssociations": false,
        "wordAssociations": false,
        "autoClearCacheDays": 15,
        "lastCacheClearDate": ""
    })
}

fn merge_with_defaults(existing: &serde_json::Value, defaults: &serde_json::Value) -> serde_json::Value {
    let mut merged = defaults.clone();
    
    if let (Some(existing_obj), Some(merged_obj)) = (existing.as_object(), merged.as_object_mut()) {
        for (key, value) in existing_obj {
            merged_obj.insert(key.clone(), value.clone());
        }
    }
    
    merged
}

#[tauri::command]
async fn get_settings(app: tauri::AppHandle) -> Result<serde_json::Value, String> {
    let config_dir = app.path().app_config_dir().map_err(|e| e.to_string())?;
    let config_path = config_dir.join("config.json");
    
    let default_config = get_default_config();
    
    if !config_path.exists() {
        return Ok(default_config);
    }
    
    if let Ok(config_content) = std::fs::read_to_string(&config_path) {
        if let Ok(existing_config) = serde_json::from_str::<serde_json::Value>(&config_content) {
            let merged_config = merge_with_defaults(&existing_config, &default_config);
            
            let merged_str = serde_json::to_string_pretty(&merged_config).map_err(|e| e.to_string())?;
            std::fs::write(&config_path, merged_str).map_err(|e| e.to_string())?;
            
            return Ok(merged_config);
        }
    }
    
    Ok(default_config)
}

#[tauri::command]
async fn save_settings(app: tauri::AppHandle, settings: serde_json::Value) -> Result<(), String> {
    let config_dir = app.path().app_config_dir().map_err(|e| e.to_string())?;
    
    if !config_dir.exists() {
        std::fs::create_dir_all(&config_dir).map_err(|e| e.to_string())?;
    }
    
    let config_path = config_dir.join("config.json");
    let temp_path = config_path.with_extension("json.tmp");
    
    let existing_settings = if config_path.exists() {
        if let Ok(config_content) = std::fs::read_to_string(&config_path) {
            if let Ok(mut existing) = serde_json::from_str::<serde_json::Value>(&config_content) {
                if let Some(obj) = existing.as_object_mut() {
                    if let Some(new_obj) = settings.as_object() {
                        for (key, value) in new_obj {
                            obj.insert(key.clone(), value.clone());
                        }
                    }
                }
                existing
            } else {
                settings
            }
        } else {
            settings
        }
    } else {
        settings
    };
    
    let config_str = serde_json::to_string_pretty(&existing_settings).map_err(|e| e.to_string())?;
    
    std::fs::write(&temp_path, &config_str).map_err(|e| e.to_string())?;
    std::fs::rename(&temp_path, &config_path).map_err(|e| {
        let _ = std::fs::remove_file(&temp_path);
        format!("Failed to rename config file: {}", e)
    })?;
    
    Ok(())
}

#[tauri::command]
async fn open_doc_scan_window(app: tauri::AppHandle) -> Result<(), String> {
    use tauri::WebviewWindowBuilder;
    
    if let Some(window) = app.get_webview_window("doc-scan") {
        window.set_focus().map_err(|e| format!("Failed to focus doc-scan window: {}", e))?;
        return Ok(());
    }
    
    let window = WebviewWindowBuilder::new(
        &app,
        "doc-scan",
        tauri::WebviewUrl::App("doc-scan/index.html".into())
    )
    .title("文档扫描增强")
    .fullscreen(true)
    .resizable(true)
    .decorations(false)
    .build()
    .map_err(|e| format!("Failed to create doc-scan window: {}", e))?;
    
    window.set_focus().map_err(|e| format!("Failed to focus new doc-scan window: {}", e))?;
    
    Ok(())
}

#[cfg(target_os = "windows")]
#[tauri::command]
async fn check_pdf_default_app() -> Result<bool, String> {
    use winreg::RegKey;
    use winreg::enums::*;
    
    let hkcu = RegKey::predef(HKEY_CURRENT_USER);
    
    // 检查用户设置的默认程序
    if let Ok(prog_id_key) = hkcu.open_subkey("Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\FileExts\\.pdf\\UserChoice") {
        if let Ok(prog_id) = prog_id_key.get_value::<String, _>("ProgId") {
            // 检查是否是 ViewStage 的 ProgId
            if prog_id.contains("ViewStage") || prog_id.contains("viewstage") {
                return Ok(true);
            }
        }
    }
    
    // 检查系统默认程序
    let hkcr = RegKey::predef(HKEY_CLASSES_ROOT);
    if let Ok(pdf_key) = hkcr.open_subkey(".pdf") {
        if let Ok(default_prog) = pdf_key.get_value::<String, _>("") {
            if default_prog.contains("ViewStage") || default_prog.contains("viewstage") {
                return Ok(true);
            }
        }
    }
    
    Ok(false)
}

#[cfg(not(target_os = "windows"))]
#[tauri::command]
async fn check_pdf_default_app() -> Result<bool, String> {
    Ok(false)
}

fn restart_application(app: &tauri::AppHandle) {
    app.restart();
}

#[tauri::command]
async fn reset_settings(app: tauri::AppHandle) -> Result<(), String> {
    let config_dir = app.path().app_config_dir().map_err(|e| e.to_string())?;
    
    if config_dir.exists() {
        std::fs::remove_dir_all(&config_dir).map_err(|e| e.to_string())?;
        
        if config_dir.exists() {
            return Err("配置目录删除失败".to_string());
        }
    }
    
    restart_application(&app);
    
    Ok(())
}

#[tauri::command]
async fn restart_app(app: tauri::AppHandle) -> Result<(), String> {
    restart_application(&app);
    
    Ok(())
}

#[tauri::command]
async fn get_available_resolutions(app: tauri::AppHandle) -> Result<Vec<(u32, u32, String)>, String> {
    let primary_monitor = app.primary_monitor()
        .map_err(|e| e.to_string())?
        .ok_or("无法获取主显示器".to_string())?;
    
    let max_width = primary_monitor.size().width;
    let max_height = primary_monitor.size().height;
    
    let mut resolutions = Vec::new();
    
    let base_resolutions: Vec<(u32, u32)> = vec![
        (1920, 1080),
        (1600, 900),
        (1366, 768),
        (1280, 720),
        (1024, 576),
    ];
    
    for (base_width, base_height) in base_resolutions {
        if base_width <= max_width && base_height <= max_height {
            resolutions.push((base_width, base_height, format!("{} x {}", base_width, base_height)));
        }
    }
    
    resolutions.push((max_width, max_height, format!("{} x {} (最大)", max_width, max_height)));
    
    Ok(resolutions)
}

#[tauri::command]
async fn close_splashscreen(app: tauri::AppHandle) -> Result<(), String> {
    if let Some(splashscreen) = app.get_webview_window("splashscreen") {
        let _ = splashscreen.close();
    }
    if let Some(main_window) = app.get_webview_window("main") {
        let _ = main_window.show();
    }
    Ok(())
}

#[tauri::command]
async fn complete_oobe(app: tauri::AppHandle) -> Result<(), String> {
    OOBE_ACTIVE.store(false, Ordering::SeqCst);
    
    restart_application(&app);
    
    Ok(())
}

#[tauri::command]
fn is_oobe_active() -> bool {
    OOBE_ACTIVE.load(Ordering::SeqCst)
}

#[tauri::command]
fn exit_app() {
    std::process::exit(0);
}

// ==================== Office 文件转换 ====================

/// Office 软件类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OfficeSoftware {
    MicrosoftWord,
    WpsOffice,
    LibreOffice,
    None,
}

/// Office 检测结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OfficeDetectionResult {
    pub has_word: bool,
    pub has_wps: bool,
    pub has_libreoffice: bool,
    pub recommended: OfficeSoftware,
}

#[cfg(target_os = "windows")]
fn detect_office_windows() -> OfficeDetectionResult {
    use winreg::RegKey;
    use winreg::enums::*;
    
    let hkcu = RegKey::predef(HKEY_CURRENT_USER);
    let hklm = RegKey::predef(HKEY_LOCAL_MACHINE);
    
    let has_word = check_word_installed(&hkcu, &hklm);
    let has_wps = check_wps_installed(&hkcu, &hklm);
    let has_libreoffice = check_libreoffice_installed(&hkcu, &hklm);
    
    let recommended = if has_word {
        OfficeSoftware::MicrosoftWord
    } else if has_wps {
        OfficeSoftware::WpsOffice
    } else if has_libreoffice {
        OfficeSoftware::LibreOffice
    } else {
        OfficeSoftware::None
    };
    
    OfficeDetectionResult {
        has_word,
        has_wps,
        has_libreoffice,
        recommended,
    }
}

#[cfg(target_os = "windows")]
fn check_word_installed(hkcu: &winreg::RegKey, hklm: &winreg::RegKey) -> bool {
    let paths = [
        "SOFTWARE\\Microsoft\\Office\\Word",
        "SOFTWARE\\Microsoft\\Office\\16.0\\Word",
        "SOFTWARE\\Microsoft\\Office\\15.0\\Word",
        "SOFTWARE\\Microsoft\\Office\\14.0\\Word",
        "SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\App Paths\\WINWORD.EXE",
    ];
    
    for path in &paths {
        if hkcu.open_subkey(path).is_ok() || hklm.open_subkey(path).is_ok() {
            return true;
        }
    }
    false
}

#[cfg(target_os = "windows")]
fn check_wps_installed(hkcu: &winreg::RegKey, hklm: &winreg::RegKey) -> bool {
    let paths = [
        "SOFTWARE\\Kingsoft\\Office",
        "SOFTWARE\\WPS",
        "SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\App Paths\\wps.exe",
    ];
    
    for path in &paths {
        if hkcu.open_subkey(path).is_ok() || hklm.open_subkey(path).is_ok() {
            return true;
        }
    }
    false
}

#[cfg(target_os = "windows")]
fn check_libreoffice_installed(hkcu: &winreg::RegKey, hklm: &winreg::RegKey) -> bool {
    let paths = [
        "SOFTWARE\\LibreOffice",
        "SOFTWARE\\The Document Foundation\\LibreOffice",
    ];
    
    for path in &paths {
        if hkcu.open_subkey(path).is_ok() || hklm.open_subkey(path).is_ok() {
            return true;
        }
    }
    false
}

#[cfg(not(target_os = "windows"))]
fn detect_office_windows() -> OfficeDetectionResult {
    OfficeDetectionResult {
        has_word: false,
        has_wps: false,
        has_libreoffice: false,
        recommended: OfficeSoftware::None,
    }
}

#[tauri::command]
fn detect_office() -> OfficeDetectionResult {
    detect_office_windows()
}

#[cfg(target_os = "windows")]
#[tauri::command]
async fn convert_docx_to_pdf_from_bytes(file_data: Vec<u8>, file_name: String, app: tauri::AppHandle) -> Result<String, String> {
    use std::fs;
    use std::io::Write;
    
    println!("收到文件数据: {} 字节", file_data.len());
    println!("文件名: {}", file_name);
    
    if file_data.len() < 4 {
        return Err("文件数据太小，可能已损坏".to_string());
    }
    
    let header: Vec<String> = file_data.iter().take(16).map(|b| format!("{:02x}", b)).collect();
    println!("文件头: {}", header.join(" "));
    
    if file_data[0] == 0x50 && file_data[1] == 0x4B {
        println!("检测到 ZIP 格式 (docx)");
    } else if file_data[0] == 0xD0 && file_data[1] == 0xCF {
        println!("检测到 OLE 格式 (doc)");
    } else {
        println!("未知文件格式");
    }
    
    let detection = detect_office_windows();
    println!("推荐使用: {:?}", detection.recommended);
    
    let config_dir = app.path().app_config_dir().map_err(|e| e.to_string())?;
    let cache_dir = config_dir.join("cache");
    fs::create_dir_all(&cache_dir).map_err(|e| e.to_string())?;
    
    let folder_name = format!("document_{}", chrono::Local::now().format("%Y%m%d%H%M%S"));
    let doc_cache_dir = cache_dir.join(&folder_name);
    fs::create_dir_all(&doc_cache_dir).map_err(|e| e.to_string())?;
    
    let temp_name = format!("temp_{}.docx", chrono::Local::now().format("%Y%m%d%H%M%S"));
    let temp_docx_path = doc_cache_dir.join(&temp_name);
    
    {
        let mut file = fs::File::create(&temp_docx_path)
            .map_err(|e| format!("创建临时文件失败: {}", e))?;
        file.write_all(&file_data)
            .map_err(|e| format!("写入临时文件失败: {}", e))?;
        file.sync_all()
            .map_err(|e| format!("同步文件失败: {}", e))?;
    }
    
    let pdf_name = format!("{}.pdf", folder_name);
    let pdf_path = doc_cache_dir.join(&pdf_name);
    
    if pdf_path.exists() {
        fs::remove_file(&pdf_path).map_err(|e| e.to_string())?;
    }
    
    let docx_path_str = temp_docx_path.to_string_lossy().to_string();
    let pdf_path_str = pdf_path.to_string_lossy().to_string();
    
    println!("临时文件路径: {}", docx_path_str);
    println!("输出 PDF 路径: {}", pdf_path_str);
    
    let result = match detection.recommended {
        OfficeSoftware::MicrosoftWord => {
            let r = convert_with_word_com(&docx_path_str, &pdf_path_str);
            if r.is_err() && detection.has_wps {
                println!("Word 转换失败，尝试 WPS...");
                convert_with_wps_com(&docx_path_str, &pdf_path_str)
            } else if r.is_err() && detection.has_libreoffice {
                println!("Word 转换失败，尝试 LibreOffice...");
                convert_with_libreoffice(&docx_path_str, &pdf_path_str, &doc_cache_dir)
            } else {
                r
            }
        }
        OfficeSoftware::WpsOffice => {
            let r = convert_with_wps_com(&docx_path_str, &pdf_path_str);
            if r.is_err() && detection.has_word {
                println!("WPS 转换失败，尝试 Word...");
                convert_with_word_com(&docx_path_str, &pdf_path_str)
            } else if r.is_err() && detection.has_libreoffice {
                println!("WPS 转换失败，尝试 LibreOffice...");
                convert_with_libreoffice(&docx_path_str, &pdf_path_str, &doc_cache_dir)
            } else {
                r
            }
        }
        OfficeSoftware::LibreOffice => {
            convert_with_libreoffice(&docx_path_str, &pdf_path_str, &doc_cache_dir)
        }
        OfficeSoftware::None => {
            Err("未检测到可用的 Office 软件，请安装 Microsoft Word、WPS Office 或 LibreOffice".to_string())
        }
    };
    
    if let Err(e) = fs::remove_file(&temp_docx_path) {
        println!("清理临时文件失败: {}", e);
    }
    
    result?;
    
    for _ in 0..10 {
        if pdf_path.exists() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    
    if pdf_path.exists() {
        Ok(pdf_path_str)
    } else {
        Err("PDF 文件生成失败".to_string())
    }
}

#[cfg(target_os = "windows")]
fn convert_with_libreoffice(docx_path: &str, _pdf_path: &str, cache_dir: &std::path::Path) -> Result<(), String> {
    use std::process::Command;
    let output_dir = cache_dir.to_str()
        .ok_or("Invalid cache directory path")?
        .to_string();
    Command::new("soffice")
        .args(["--headless", "--convert-to", "pdf", "--outdir", &output_dir, docx_path])
        .output()
        .map(|_| ())
        .map_err(|e| format!("LibreOffice 转换失败: {}", e))
}

#[cfg(target_os = "windows")]
#[tauri::command]
async fn convert_docx_to_pdf(docx_path: String, app: tauri::AppHandle) -> Result<String, String> {
    use std::process::Command;
    use std::fs;
    
    let detection = detect_office_windows();
    
    let docx = std::path::Path::new(&docx_path);
    let docx_absolute = std::fs::canonicalize(docx)
        .map_err(|e| format!("无法获取文件绝对路径: {}", e))?;
    
    if !docx_absolute.exists() {
        return Err(format!("文件不存在: {}", docx_absolute.display()));
    }
    
    println!("转换文件: {}", docx_absolute.display());
    
    let config_dir = app.path().app_config_dir().map_err(|e| e.to_string())?;
    let cache_dir = config_dir.join("cache");
    fs::create_dir_all(&cache_dir).map_err(|e| e.to_string())?;
    
    let folder_name = format!("document_{}", chrono::Local::now().format("%Y%m%d%H%M%S"));
    let doc_cache_dir = cache_dir.join(&folder_name);
    fs::create_dir_all(&doc_cache_dir).map_err(|e| e.to_string())?;
    
    let pdf_name = format!("{}.pdf", folder_name);
    let pdf_path = doc_cache_dir.join(&pdf_name);
    
    if pdf_path.exists() {
        fs::remove_file(&pdf_path).map_err(|e| e.to_string())?;
    }
    
    let docx_path_str = docx_absolute.to_string_lossy().to_string();
    let pdf_path_str = pdf_path.to_string_lossy().to_string();
    
    match detection.recommended {
        OfficeSoftware::MicrosoftWord => {
            convert_with_word_com(&docx_path_str, &pdf_path_str)?;
        }
        OfficeSoftware::WpsOffice => {
            convert_with_wps_com(&docx_path_str, &pdf_path_str)?;
        }
        OfficeSoftware::LibreOffice => {
            let output_dir = doc_cache_dir.to_str()
                .ok_or("Invalid cache directory path")?
                .to_string();
            Command::new("soffice")
                .args(["--headless", "--convert-to", "pdf", "--outdir", &output_dir, &docx_path_str])
                .output()
                .map_err(|e| format!("LibreOffice 转换失败: {}", e))?;
        }
        OfficeSoftware::None => {
            return Err("未检测到可用的 Office 软件，请安装 Microsoft Word、WPS Office 或 LibreOffice".to_string());
        }
    }
    
    std::thread::sleep(std::time::Duration::from_millis(500));
    
    if pdf_path.exists() {
        Ok(pdf_path_str)
    } else {
        Err("PDF 文件生成失败".to_string())
    }
}

#[cfg(target_os = "windows")]
fn convert_with_word_com(docx_path: &str, pdf_path: &str) -> Result<(), String> {
    use std::process::Command;
    
    println!("Word COM 转换开始");
    println!("  输入文件: {}", docx_path);
    println!("  输出文件: {}", pdf_path);
    
    let ps_script = format!(r#"
        $ErrorActionPreference = 'Stop'
        
        $word = New-Object -ComObject Word.Application
        $word.Visible = $false
        $word.DisplayAlerts = 0
        $doc = $null
        try {{
            $doc = $word.Documents.Open('{input}', $false, $false, $false)
            if (-not $doc) {{
                throw "无法打开文档，文件可能已损坏或格式不支持"
            }}
            $doc.ExportAsFixedFormat('{output}', 17)
        }}
        finally {{
            if ($doc) {{ 
                try {{ $doc.Close($false) }} catch {{}}
                [System.Runtime.Interopservices.Marshal]::ReleaseComObject($doc) | Out-Null
            }}
            try {{ $word.Quit() }} catch {{}}
            [System.Runtime.Interopservices.Marshal]::ReleaseComObject($word) | Out-Null
            [GC]::Collect()
            [GC]::WaitForPendingFinalizers()
        }}
    "#, input = docx_path.replace("'", "''"), output = pdf_path.replace("'", "''"));
    
    let output = Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", &ps_script])
        .creation_flags(CREATE_NO_WINDOW)
        .output()
        .map_err(|e| format!("PowerShell 执行失败: {}", e))?;
    
    if output.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(format!("Word 转换失败: {}", stderr))
    }
}

#[cfg(target_os = "windows")]
fn convert_with_wps_com(docx_path: &str, pdf_path: &str) -> Result<(), String> {
    use std::process::Command;
    
    println!("WPS COM 转换开始");
    println!("  输入文件: {}", docx_path);
    println!("  输出文件: {}", pdf_path);
    
    let ps_script = format!(r#"
        $ErrorActionPreference = 'Stop'
        
        $wps = $null
        try {{
            $wps = New-Object -ComObject Kwps.Application
        }} catch {{
            $wps = New-Object -ComObject WPS.Application
        }}
        $wps.Visible = $false
        $wps.DisplayAlerts = 0
        $doc = $null
        try {{
            $doc = $wps.Documents.Open('{input}', $false, $false, $false)
            if (-not $doc) {{
                throw "无法打开文档，文件可能已损坏或格式不支持"
            }}
            $doc.ExportAsFixedFormat('{output}', 17)
        }}
        finally {{
            if ($doc) {{ 
                try {{ $doc.Close($false) }} catch {{}}
                [System.Runtime.Interopservices.Marshal]::ReleaseComObject($doc) | Out-Null
            }}
            try {{ $wps.Quit() }} catch {{}}
            [System.Runtime.Interopservices.Marshal]::ReleaseComObject($wps) | Out-Null
            [GC]::Collect()
            [GC]::WaitForPendingFinalizers()
        }}
    "#, input = docx_path.replace("'", "''"), output = pdf_path.replace("'", "''"));
    
    let output = Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", &ps_script])
        .creation_flags(CREATE_NO_WINDOW)
        .output()
        .map_err(|e| format!("PowerShell 执行失败: {}", e))?;
    
    if output.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(format!("WPS 转换失败: {}", stderr))
    }
}

#[cfg(not(target_os = "windows"))]
#[tauri::command]
async fn convert_docx_to_pdf(_docx_path: String, _app: tauri::AppHandle) -> Result<String, String> {
    Err("此功能仅支持 Windows 系统".to_string())
}

#[cfg(target_os = "windows")]
#[tauri::command]
async fn set_file_type_icons(app: tauri::AppHandle) -> Result<(), String> {
    use std::process::Command;
    
    let resource_dir = app.path().resource_dir()
        .map_err(|e| format!("获取资源目录失败: {}", e))?;
    
    let pdf_icon = resource_dir.join("icons").join("pdf.ico").to_string_lossy().to_string();
    let word_icon = resource_dir.join("icons").join("word.ico").to_string_lossy().to_string();
    
    let app_id = "SECTL.ViewStage";
    
    println!("PDF 图标路径: {}", pdf_icon);
    println!("Word 图标路径: {}", word_icon);
    
    let ps_script = format!(r#"
        $ErrorActionPreference = 'SilentlyContinue'
        
        # 设置 PDF 文件图标
        $pdfKey = 'HKCU:\Software\Classes\{app_id}.pdf'
        New-Item -Path $pdfKey -Force | Out-Null
        New-Item -Path "$pdfKey\DefaultIcon" -Force | Out-Null
        Set-ItemProperty -Path "$pdfKey\DefaultIcon" -Name '(Default)' -Value '{pdf_icon}'
        
        # 设置 DOCX 文件图标
        $docxKey = 'HKCU:\Software\Classes\{app_id}.docx'
        New-Item -Path $docxKey -Force | Out-Null
        New-Item -Path "$docxKey\DefaultIcon" -Force | Out-Null
        Set-ItemProperty -Path "$docxKey\DefaultIcon" -Name '(Default)' -Value '{word_icon}'
        
        # 设置 DOC 文件图标
        $docKey = 'HKCU:\Software\Classes\{app_id}.doc'
        New-Item -Path $docKey -Force | Out-Null
        New-Item -Path "$docKey\DefaultIcon" -Force | Out-Null
        Set-ItemProperty -Path "$docKey\DefaultIcon" -Name '(Default)' -Value '{word_icon}'
        
        # 刷新图标缓存
        $code = @'
        [DllImport("shell32.dll")]
        public static extern void SHChangeNotify(int wEventId, uint uFlags, IntPtr dwItem1, IntPtr dwItem2);
'@
        Add-Type -MemberDefinition $code -Name Shell -Namespace WinAPI
        [WinAPI.Shell]::SHChangeNotify(0x8000000, 0x1000, [IntPtr]::Zero, [IntPtr]::Zero)
        
        Write-Host "文件类型图标已设置"
    "#, app_id = app_id, pdf_icon = pdf_icon, word_icon = word_icon);
    
    let output = Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", &ps_script])
        .creation_flags(CREATE_NO_WINDOW)
        .output()
        .map_err(|e| format!("设置图标失败: {}", e))?;
    
    if output.status.success() {
        println!("文件类型图标设置成功");
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(format!("设置图标失败: {}", stderr))
    }
}

#[cfg(not(target_os = "windows"))]
#[tauri::command]
async fn set_file_type_icons() -> Result<(), String> {
    Err("此功能仅支持 Windows 系统".to_string())
}

// ==================== 文档扫描增强 ====================
// 边缘检测、透视变换、文档增强

/// 文档扫描请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentScanRequest {
    pub image_data: String,
    pub east_model_path: Option<String>,
    pub grayscale: Option<bool>,
}

/// 文档扫描结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentScanResult {
    pub enhanced_image: String,
    pub confidence: f32,
    pub text_bbox: Option<(i32, i32, i32, i32)>,
}

/// DexiNed 边缘检测请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DexiNedDetectionRequest {
    pub image_data: String,
    pub model_path: Option<String>,
}

/// DexiNed 边缘检测结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DexiNedDetectionResult {
    pub edge_image: String,
    pub success: bool,
    pub error: Option<String>,
}

/// 检查 DexiNed 模型是否存在
#[tauri::command]
fn check_dexined_model(app: tauri::AppHandle) -> Result<bool, String> {
    let models_dir = get_models_dir(&app)?;
    let model_path = models_dir.join("dexined_mbv4.onnx");
    Ok(model_path.exists())
}

/// 导入 DexiNed 模型文件
#[tauri::command]
fn import_dexined_model(app: tauri::AppHandle, source_path: String) -> Result<(), String> {
    let models_dir = get_models_dir(&app)?;
    
    let dest_path = models_dir.join("dexined_mbv4.onnx");
    
    std::fs::copy(&source_path, &dest_path)
        .map_err(|e| format!("复制模型文件失败: {}", e))?;
    
    log::info!("DexiNed 模型已导入: {:?}", dest_path);
    
    Ok(())
}

/// 删除 DexiNed 模型
#[tauri::command]
fn delete_dexined_model(app: tauri::AppHandle) -> Result<(), String> {
    let models_dir = get_models_dir(&app)?;
    let model_path = models_dir.join("dexined_mbv4.onnx");
    
    if model_path.exists() {
        std::fs::remove_file(&model_path)
            .map_err(|e| format!("删除模型文件失败: {}", e))?;
        log::info!("DexiNed 模型已删除");
    }
    
    Ok(())
}

/// DexiNed 边缘检测命令 (ONNX)
#[tauri::command]
fn detect_edges_dexined(app: tauri::AppHandle, request: DexiNedDetectionRequest) -> Result<DexiNedDetectionResult, String> {
    use ort::session::Session;
    use ort::value::Tensor;
    
    let img = decode_base64_image(&request.image_data)?;
    
    let model_path = match request.model_path {
        Some(path) => path,
        None => {
            let models_dir = get_models_dir(&app)?;
            models_dir.join("dexined_mbv4.onnx")
                .to_string_lossy().to_string()
        }
    };
    
    log::info!("加载 DexiNed ONNX 模型: {}", model_path);
    
    let mut session = Session::builder()
        .map_err(|e| format!("创建 Session builder 失败: {}", e))?
        .with_execution_providers(&[ort::ep::DirectMLExecutionProvider::default().into()])
        .map_err(|e| format!("设置 DirectML 执行提供器失败: {}", e))?
        .commit_from_file(&model_path)
        .map_err(|e| format!("加载 DexiNed ONNX 模型失败: {}", e))?;
    
    log::info!("DexiNed ONNX 模型加载成功（DirectML GPU 加速）");
    
    let (orig_width, orig_height) = (img.width(), img.height());
    
    let rgba = img.to_rgba8();
    
    let mut input_data = Vec::with_capacity((orig_width * orig_height * 3) as usize);
    for pixel in rgba.pixels() {
        input_data.push(pixel[0] as f32 / 255.0);
        input_data.push(pixel[1] as f32 / 255.0);
        input_data.push(pixel[2] as f32 / 255.0);
    }
    
    let input_shape = [1usize, 3usize, orig_height as usize, orig_width as usize];
    let input_tensor = Tensor::from_array((input_shape, input_data.into_boxed_slice()))
        .map_err(|e| format!("创建输入tensor失败: {}", e))?;
    
    log::info!("DexiNed 开始推理");
    
    let outputs = session.run(ort::inputs![input_tensor])
        .map_err(|e| format!("DexiNed 推理失败: {}", e))?;
    
    let (output_shape, output_data) = outputs[0].try_extract_tensor::<f32>()
        .map_err(|e| format!("获取输出失败: {}", e))?;
    
    let output_dims: Vec<usize> = output_shape.iter().map(|d| *d as usize).collect();
    log::info!("DexiNed 输出维度: {:?}", output_dims);
    
    let (out_height, out_width) = if output_dims.len() == 4 {
        (output_dims[2], output_dims[3])
    } else if output_dims.len() == 3 {
        (output_dims[1], output_dims[2])
    } else {
        return Err(format!("不支持的输出维度: {:?}", output_dims));
    };
    
    let mut edge_image: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::new(out_width as u32, out_height as u32);
    
    let min_val = output_data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = output_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = max_val - min_val;
    
    for y in 0..out_height {
        for x in 0..out_width {
            let idx = y * out_width + x;
            let val = output_data[idx];
            let normalized = if range > 0.0 { (val - min_val) / range } else { 0.0 };
            let pixel_val = (normalized * 255.0).clamp(0.0, 255.0) as u8;
            edge_image.put_pixel(x as u32, y as u32, Luma([pixel_val]));
        }
    }
    
    let final_edge = image::imageops::resize(
        &edge_image,
        orig_width,
        orig_height,
        image::imageops::FilterType::Triangle
    );
    
    let mut buffer = Vec::new();
    DynamicImage::ImageLuma8(final_edge)
        .write_to(&mut std::io::Cursor::new(&mut buffer), image::ImageFormat::Png)
        .map_err(|e| format!("编码边缘图像失败: {}", e))?;
    
    let result_image = format!("data:image/png;base64,{}", general_purpose::STANDARD.encode(&buffer));
    
    log::info!("DexiNed 边缘检测完成");
    
    Ok(DexiNedDetectionResult {
        edge_image: result_image,
        success: true,
        error: None,
    })
}

/// 文档边界检测结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentBoundaryResult {
    pub corners: Option<Vec<(f32, f32)>>,
    pub success: bool,
    pub error: Option<String>,
}

/// 使用 DexiNed 边缘检测 + OpenCV 检测文档边界
#[cfg(target_os = "windows")]
fn detect_document_boundary_opencv(edge_image: &ImageBuffer<Luma<u8>, Vec<u8>>) -> Option<Vec<Point2f>> {
    let (width, height) = edge_image.dimensions();
    let _img_area = width * height;
    
    let mut gray_mat = unsafe { Mat::new_rows_cols(height as i32, width as i32, CV_8UC1) }
        .ok()?;
    
    {
        let data = gray_mat.data_bytes_mut().ok()?;
        for y in 0..height {
            for x in 0..width {
                let pixel = edge_image.get_pixel(x, y);
                let idx = (y * width + x) as usize;
                data[idx] = pixel[0];
            }
        }
    }
    
    let mut blurred = Mat::default();
    opencv::imgproc::gaussian_blur(
        &gray_mat,
        &mut blurred,
        opencv::core::Size::new(5, 5),
        0.0,
        0.0,
        opencv::core::BORDER_DEFAULT
    ).ok()?;
    
    let mut binary = Mat::default();
    threshold(&blurred, &mut binary, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU).ok()?;
    
    let kernel = opencv::imgproc::get_structuring_element(
        opencv::imgproc::MORPH_RECT,
        opencv::core::Size::new(3, 3),
        opencv::core::Point::new(-1, -1)
    ).ok()?;
    
    let mut morphed = Mat::default();
    opencv::imgproc::morphology_ex(
        &binary,
        &mut morphed,
        opencv::imgproc::MORPH_CLOSE,
        &kernel,
        opencv::core::Point::new(-1, -1),
        2,
        opencv::core::BORDER_CONSTANT,
        Scalar::default()
    ).ok()?;
    
    let min_line_length = (width.min(height) as f32 * 0.1) as i32;
    let max_line_gap = (width.min(height) as f32 * 0.05) as i32;
    
    let mut lines = Vector::<opencv::core::Vec4i>::new();
    hough_lines_p(
        &morphed,
        &mut lines,
        1.0,
        std::f64::consts::PI / 180.0,
        50,
        min_line_length as f64,
        max_line_gap as f64
    ).ok()?;
    
    log::info!("检测到 {} 条直线", lines.len());
    
    if lines.len() < 4 {
        return None;
    }
    
    let mut all_lines: Vec<(f32, f32, f32, f32)> = Vec::new();
    for i in 0..lines.len() {
        let line = lines.get(i).ok()?;
        all_lines.push((line[0] as f32, line[1] as f32, line[2] as f32, line[3] as f32));
    }
    
    let mut horizontal_lines: Vec<(f32, f32, f32, f32)> = Vec::new();
    let mut vertical_lines: Vec<(f32, f32, f32, f32)> = Vec::new();
    
    for (x1, y1, x2, y2) in all_lines {
        let angle = (y2 - y1).atan2(x2 - x1).abs();
        if !(0.2..=std::f32::consts::PI - 0.2).contains(&angle) {
            horizontal_lines.push((x1, y1, x2, y2));
        } else if (angle - std::f32::consts::PI / 2.0).abs() < 0.3 {
            vertical_lines.push((x1, y1, x2, y2));
        }
    }
    
    log::info!("水平线: {}, 垂直线: {}", horizontal_lines.len(), vertical_lines.len());
    
    if horizontal_lines.len() < 2 || vertical_lines.len() < 2 {
        return None;
    }
    
    horizontal_lines = merge_nearby_lines(&horizontal_lines, true);
    vertical_lines = merge_nearby_lines(&vertical_lines, false);
    
    log::info!("合并后 - 水平线: {}, 垂直线: {}", horizontal_lines.len(), vertical_lines.len());
    
    if horizontal_lines.len() < 2 || vertical_lines.len() < 2 {
        return None;
    }
    
    horizontal_lines.sort_by(|a, b| {
        let y_a = (a.1 + a.3) / 2.0;
        let y_b = (b.1 + b.3) / 2.0;
        y_a.partial_cmp(&y_b).unwrap_or(std::cmp::Ordering::Equal)
    });
    
    vertical_lines.sort_by(|a, b| {
        let x_a = (a.0 + a.2) / 2.0;
        let x_b = (b.0 + b.2) / 2.0;
        x_a.partial_cmp(&x_b).unwrap_or(std::cmp::Ordering::Equal)
    });
    
    let top_line = horizontal_lines.first()?;
    let bottom_line = horizontal_lines.last()?;
    let left_line = vertical_lines.first()?;
    let right_line = vertical_lines.last()?;
    
    let top_left = line_intersection(top_line, left_line)?;
    let top_right = line_intersection(top_line, right_line)?;
    let bottom_left = line_intersection(bottom_line, left_line)?;
    let bottom_right = line_intersection(bottom_line, right_line)?;
    
    if !is_valid_corner(&top_left, width, height) ||
       !is_valid_corner(&top_right, width, height) ||
       !is_valid_corner(&bottom_left, width, height) ||
       !is_valid_corner(&bottom_right, width, height) {
        log::warn!("角点超出图像边界，放弃检测");
        return None;
    }
    
    let corners = vec![
        Point2f::new(top_left.0, top_left.1),
        Point2f::new(top_right.0, top_right.1),
        Point2f::new(bottom_right.0, bottom_right.1),
        Point2f::new(bottom_left.0, bottom_left.1),
    ];
    
    if !validate_corners(&corners, width, height) {
        log::warn!("角点验证失败，放弃检测");
        return None;
    }
    
    log::info!("检测到文档角点: {:?}", corners);
    
    Some(corners)
}

fn merge_nearby_lines(lines: &[(f32, f32, f32, f32)], is_horizontal: bool) -> Vec<(f32, f32, f32, f32)> {
    if lines.is_empty() {
        return Vec::new();
    }
    
    let mut sorted_lines = lines.to_vec();
    if is_horizontal {
        sorted_lines.sort_by(|a, b| {
            let y_a = (a.1 + a.3) / 2.0;
            let y_b = (b.1 + b.3) / 2.0;
            y_a.partial_cmp(&y_b).unwrap_or(std::cmp::Ordering::Equal)
        });
    } else {
        sorted_lines.sort_by(|a, b| {
            let x_a = (a.0 + a.2) / 2.0;
            let x_b = (b.0 + b.2) / 2.0;
            x_a.partial_cmp(&x_b).unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    
    let mut merged: Vec<(f32, f32, f32, f32)> = Vec::new();
    let threshold = 30.0f32;
    
    for line in sorted_lines {
        if let Some(last) = merged.last_mut() {
            let last_pos = if is_horizontal { (last.1 + last.3) / 2.0 } else { (last.0 + last.2) / 2.0 };
            let curr_pos = if is_horizontal { (line.1 + line.3) / 2.0 } else { (line.0 + line.2) / 2.0 };
            
            if (curr_pos - last_pos).abs() < threshold {
                last.0 = last.0.min(line.0);
                last.1 = last.1.min(line.1);
                last.2 = last.2.max(line.2);
                last.3 = last.3.max(line.3);
                continue;
            }
        }
        merged.push(line);
    }
    
    merged
}

fn is_valid_corner(point: &(f32, f32), width: u32, height: u32) -> bool {
    let margin = -50.0f32;
    point.0 >= margin && 
    point.0 <= width as f32 - margin &&
    point.1 >= margin && 
    point.1 <= height as f32 - margin
}

fn validate_corners(corners: &[Point2f], width: u32, height: u32) -> bool {
    if corners.len() != 4 {
        return false;
    }
    
    let min_area = (width as f32 * height as f32) * 0.1;
    
    let area = ((corners[0].x - corners[1].x) * (corners[0].y - corners[3].y)).abs();
    
    if area < min_area {
        log::warn!("检测到的区域面积过小: {} < {}", area, min_area);
        return false;
    }
    
    let cross = (corners[1].x - corners[0].x) * (corners[3].y - corners[0].y) -
                (corners[1].y - corners[0].y) * (corners[3].x - corners[0].x);
    if cross < 0.0 {
        log::warn!("角点顺序不正确（可能是顺时针）");
        return false;
    }
    
    true
}

/// 计算两条直线的交点
fn line_intersection(line1: &(f32, f32, f32, f32), line2: &(f32, f32, f32, f32)) -> Option<(f32, f32)> {
    let (x1, y1, x2, y2) = *line1;
    let (x3, y3, x4, y4) = *line2;
    
    let denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
    if denom.abs() < 1e-10 {
        return None;
    }
    
    let t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom;
    
    let px = x1 + t * (x2 - x1);
    let py = y1 + t * (y2 - y1);
    
    Some((px, py))
}

/// 对图像进行透视变换
#[cfg(target_os = "windows")]
fn perspective_transform(img: &DynamicImage, corners: &[Point2f]) -> Option<DynamicImage> {
    let (width, height) = img.dimensions();
    
    let top_width = ((corners[1].x - corners[0].x).powi(2) + (corners[1].y - corners[0].y).powi(2)).sqrt();
    let bottom_width = ((corners[2].x - corners[3].x).powi(2) + (corners[2].y - corners[3].y).powi(2)).sqrt();
    let left_height = ((corners[3].y - corners[0].y).powi(2) + (corners[3].x - corners[0].x).powi(2)).sqrt();
    let right_height = ((corners[2].y - corners[1].y).powi(2) + (corners[2].x - corners[1].x).powi(2)).sqrt();
    
    let max_width = top_width.max(bottom_width) as i32;
    let max_height = left_height.max(right_height) as i32;
    
    if max_width < 10 || max_height < 10 {
        return None;
    }
    
    let rgba = img.to_rgba8();
    let mut src_mat = unsafe { Mat::new_rows_cols(height as i32, width as i32, CV_8UC3) }.ok()?;
    
    {
        let data = src_mat.data_bytes_mut().ok()?;
        for y in 0..height {
            for x in 0..width {
                let pixel = rgba.get_pixel(x, y);
                let idx = (y * width + x) as usize * 3;
                data[idx] = pixel[2];
                data[idx + 1] = pixel[1];
                data[idx + 2] = pixel[0];
            }
        }
    }
    
    let mut src_points = Vector::<Point2f>::new();
    for corner in corners {
        src_points.push(*corner);
    }
    
    let mut dst_points = Vector::<Point2f>::new();
    dst_points.push(Point2f::new(0.0, 0.0));
    dst_points.push(Point2f::new(max_width as f32, 0.0));
    dst_points.push(Point2f::new(max_width as f32, max_height as f32));
    dst_points.push(Point2f::new(0.0, max_height as f32));
    
    let transform_matrix = get_perspective_transform(&src_points, &dst_points, 0).ok()?;
    
    let mut dst_mat = Mat::default();
    warp_perspective(
        &src_mat,
        &mut dst_mat,
        &transform_matrix,
        Size::new(max_width, max_height),
        INTER_LINEAR,
        BorderTypes::BORDER_CONSTANT as i32,
        Scalar::default()
    ).ok()?;
    
    let mut enhanced_mat = Mat::default();
    {
        let mut lab = Mat::default();
        cvt_color(&dst_mat, &mut lab, COLOR_BGR2Lab, 0).ok()?;
        
        let mut lab_planes = Vector::<Mat>::new();
        opencv::core::split(&lab, &mut lab_planes).ok()?;
        
        if lab_planes.len() >= 3 {
            let mut l_channel = lab_planes.get(0).ok()?;
            
            let mut clahe = opencv::imgproc::create_clahe(2.0, opencv::core::Size::new(8, 8)).ok()?;
            clahe.apply(&l_channel.clone(), &mut l_channel).ok()?;
            
            lab_planes.set(0, l_channel).ok()?;
            
            let mut merged_lab = Mat::default();
            opencv::core::merge(&lab_planes, &mut merged_lab).ok()?;
            
            cvt_color(&merged_lab, &mut enhanced_mat, COLOR_Lab2BGR, 0).ok()?;
        } else {
            enhanced_mat = dst_mat.clone();
        }
    }
    
    let result_height = enhanced_mat.rows() as u32;
    let result_width = enhanced_mat.cols() as u32;
    let result_data = enhanced_mat.data_bytes().ok()?;
    
    let mut result_img = ImageBuffer::new(result_width, result_height);
        for y in 0..result_height {
            for x in 0..result_width {
                let idx = (y * result_width + x) as usize * 3;
                let r = result_data.get(idx + 2).copied().unwrap_or(0);
                let g = result_data.get(idx + 1).copied().unwrap_or(0);
                let b = result_data.get(idx).copied().unwrap_or(0);
                result_img.put_pixel(x, y, Rgb([r, g, b]));
            }
        }
        
        Some(DynamicImage::ImageRgb8(result_img))
    }

#[cfg(not(target_os = "windows"))]
fn detect_document_boundary_opencv(_edge_image: &ImageBuffer<Luma<u8>, Vec<u8>>) -> Option<Vec<(f32, f32)>> {
    None
}

#[cfg(not(target_os = "windows"))]
fn perspective_transform(_img: &DynamicImage, _corners: &[(f32, f32)]) -> Option<DynamicImage> {
    None
}

/// 使用 DexiNed 检测文档边界
#[tauri::command]
fn detect_document_boundary_dexined(app: tauri::AppHandle, request: DexiNedDetectionRequest) -> Result<DocumentBoundaryResult, String> {
    use ort::session::Session;
    use ort::value::Tensor;
    
    let img = decode_base64_image(&request.image_data)?;
    
    let model_path = match request.model_path {
        Some(path) => path,
        None => {
            let models_dir = get_models_dir(&app)?;
            models_dir.join("dexined_mbv4.onnx")
                .to_string_lossy().to_string()
        }
    };
    
    if !std::path::Path::new(&model_path).exists() {
        return Ok(DocumentBoundaryResult {
            corners: None,
            success: false,
            error: Some("DexiNed 模型不存在".to_string()),
        });
    }
    
    let (orig_width, orig_height) = (img.width(), img.height());
    
    let mut session = Session::builder()
        .map_err(|e| format!("创建 Session builder 失败: {}", e))?
        .with_execution_providers(&[ort::ep::DirectMLExecutionProvider::default().into()])
        .map_err(|e| format!("设置 DirectML 执行提供器失败: {}", e))?
        .commit_from_file(&model_path)
        .map_err(|e| format!("加载 DexiNed ONNX 模型失败: {}", e))?;
    
    let rgba = img.to_rgba8();
    
    let mut input_data = Vec::with_capacity((orig_width * orig_height * 3) as usize);
    for pixel in rgba.pixels() {
        input_data.push(pixel[0] as f32 / 255.0);
        input_data.push(pixel[1] as f32 / 255.0);
        input_data.push(pixel[2] as f32 / 255.0);
    }
    
    let input_shape = [1usize, 3usize, orig_height as usize, orig_width as usize];
    let input_tensor = Tensor::from_array((input_shape, input_data.into_boxed_slice()))
        .map_err(|e| format!("创建输入tensor失败: {}", e))?;
    
    let outputs = session.run(ort::inputs![input_tensor])
        .map_err(|e| format!("DexiNed 推理失败: {}", e))?;
    
    let (output_shape, output_data) = outputs[0].try_extract_tensor::<f32>()
        .map_err(|e| format!("获取输出失败: {}", e))?;
    
    let output_dims: Vec<usize> = output_shape.iter().map(|d| *d as usize).collect();
    
    let (out_height, out_width) = if output_dims.len() == 4 {
        (output_dims[2], output_dims[3])
    } else if output_dims.len() == 3 {
        (output_dims[1], output_dims[2])
    } else {
        return Err(format!("不支持的输出维度: {:?}", output_dims));
    };
    
    let mut edge_image: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::new(out_width as u32, out_height as u32);
    
    let min_val = output_data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = output_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = max_val - min_val;
    
    for y in 0..out_height {
        for x in 0..out_width {
            let idx = y * out_width + x;
            let val = output_data[idx];
            let normalized = if range > 0.0 { (val - min_val) / range } else { 0.0 };
            let pixel_val = (normalized * 255.0).clamp(0.0, 255.0) as u8;
            edge_image.put_pixel(x as u32, y as u32, Luma([pixel_val]));
        }
    }
    
    let final_edge = image::imageops::resize(
        &edge_image,
        orig_width,
        orig_height,
        image::imageops::FilterType::Triangle
    );
    
    let corners = detect_document_boundary_opencv(&final_edge);
    
    match corners {
        Some(c) => {
            let corners_vec: Vec<(f32, f32)> = c.iter().map(|p| (p.x, p.y)).collect();
            Ok(DocumentBoundaryResult {
                corners: Some(corners_vec),
                success: true,
                error: None,
            })
        }
        None => Ok(DocumentBoundaryResult {
            corners: None,
            success: false,
            error: Some("未能检测到文档边界".to_string()),
        })
    }
}

/// DBNet 文本检测请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DBNetDetectionRequest {
    pub image_data: String,
    pub model_path: Option<String>,
    pub binary_threshold: f32,
}

/// DBNet 文本检测结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DBNetDetectionResult {
    pub bbox: Option<(i32, i32, i32, i32)>,
    pub success: bool,
    pub error: Option<String>,
}

/// 获取 DBNet 模型默认路径
#[tauri::command]
fn get_dbnet_model_path(app: tauri::AppHandle) -> Result<String, String> {
    let models_dir = get_models_dir(&app)?;
    let model_path = models_dir.join("text_detection_db_TD500_resnet18.onnx");
    Ok(model_path.to_string_lossy().to_string())
}

/// DBNet 文本检测命令
#[tauri::command]
fn detect_text_dbnet(app: tauri::AppHandle, request: DBNetDetectionRequest) -> Result<DBNetDetectionResult, String> {
    let img = decode_base64_image(&request.image_data)?;
    
    let model_path = match request.model_path {
        Some(path) => path,
        None => {
            let models_dir = get_models_dir(&app)?;
            models_dir.join("text_detection_db_TD500_resnet18.onnx")
                .to_string_lossy().to_string()
        }
    };
    
    match detect_text_regions_dbnet(&img, &model_path, request.binary_threshold) {
        Ok(bbox) => Ok(DBNetDetectionResult {
            bbox,
            success: true,
            error: None,
        }),
        Err(e) => Ok(DBNetDetectionResult {
            bbox: None,
            success: false,
            error: Some(e),
        }),
    }
}

/// 文档扫描 - 自动选择最佳模型（优先DexiNed边界检测，其次DBNet，最后EAST）
#[tauri::command]
async fn scan_document(app: tauri::AppHandle, request: DocumentScanRequest) -> Result<DocumentScanResult, String> {
    let img = decode_base64_image(&request.image_data)?;
    let grayscale = request.grayscale.unwrap_or(false);
    
    log::info!("开始文档扫描，图像尺寸: {}x{}, 黑白模式: {}", img.width(), img.height(), grayscale);
    
    let models_dir = get_models_dir(&app)?;
    log::info!("模型目录: {:?}", models_dir);
    
    let dexined_model_path = models_dir.join("dexined_mbv4.onnx");
    let dbnet_model_path = models_dir.join("text_detection_db_TD500_resnet18.onnx");
    
    let dexined_session = if dexined_model_path.exists() {
        match get_or_load_dexined_model(&dexined_model_path) {
            Ok(session) => Some(session),
            Err(e) => {
                log::warn!("加载 DexiNed 模型失败: {}", e);
                None
            }
        }
    } else {
        None
    };
    
    let result = tokio::task::spawn_blocking(move || {
        let mut result_img = img.clone();
        let mut text_bbox: Option<(i32, i32, i32, i32)> = None;
        let mut used_dexined = false;
        let output_grayscale = grayscale;
        
        if let Some(ref session) = dexined_session {
            log::info!("尝试使用 DexiNed 边界检测");
            
            match try_dexined_boundary_detection(&img, session) {
                Ok(Some(transformed_img)) => {
                    log::info!("DexiNed 边界检测成功，已进行透视变换");
                    result_img = transformed_img;
                    used_dexined = true;
                }
                Ok(None) => {
                    log::info!("DexiNed 未检测到文档边界，回退到文本检测");
                }
                Err(e) => {
                    log::warn!("DexiNed 边界检测失败: {}, 回退到文本检测", e);
                }
            }
        }
        
        if !used_dexined {
            text_bbox = if dbnet_model_path.exists() {
                log::info!("尝试使用 ONNX Runtime DBNet 模型: {:?}", dbnet_model_path);
                match detect_text_regions_dbnet_ort(&img, dbnet_model_path.to_string_lossy().to_string().as_str(), 0.1) {
                    Ok(bbox) => {
                        log::info!("ONNX Runtime DBNet 检测成功: {:?}", bbox);
                        bbox
                    }
                    Err(e) => {
                        log::error!("ONNX Runtime DBNet 检测失败: {}", e);
                        None
                    }
                }
            } else {
                log::error!("没有可用的文本检测模型");
                None
            };
            
            result_img = if let Some((x1, y1, x2, y2)) = text_bbox {
                log::info!("裁剪区域: ({}, {}) - ({}, {})", x1, y1, x2, y2);
                let (width, height) = (img.width() as i32, img.height() as i32);
                let x1 = x1.max(0).min(width - 1) as u32;
                let y1 = y1.max(0).min(height - 1) as u32;
                let x2 = x2.max(0).min(width) as u32;
                let y2 = y2.max(0).min(height) as u32;
                
                if x2 > x1 && y2 > y1 {
                    log::info!("执行裁剪: ({}, {}) - ({}, {})", x1, y1, x2, y2);
                    img.crop_imm(x1, y1, x2 - x1, y2 - y1)
                } else {
                    log::warn!("裁剪区域无效，返回原图");
                    img.clone()
                }
            } else {
                log::warn!("未检测到文本区域，返回原图");
                img.clone()
            };
        }

        let enhanced_img = match enhance_document_opencv(&result_img, output_grayscale) {
            Ok(img) => img,
            Err(e) => {
                log::error!("图像增强失败: {}", e);
                return Err(e);
            }
        };
        
        let mut buffer = Vec::new();
        if let Err(e) = enhanced_img.write_to(&mut std::io::Cursor::new(&mut buffer), image::ImageFormat::Png) {
            return Err(format!("Failed to encode image: {}", e));
        }
        
        let result_image = format!("data:image/png;base64,{}", general_purpose::STANDARD.encode(&buffer));
        
        Ok(DocumentScanResult {
            enhanced_image: result_image,
            confidence: if used_dexined || text_bbox.is_some() { 0.9 } else { 0.0 },
            text_bbox,
        })
    }).await.map_err(|e| format!("任务执行失败: {}", e))?;
    
    result
}

/// 尝试使用 DexiNed 进行边界检测和透视变换
fn try_dexined_boundary_detection(img: &DynamicImage, session: &Arc<Mutex<ort::session::Session>>) -> Result<Option<DynamicImage>, String> {
    use ort::value::Tensor;
    
    let (orig_width, orig_height) = (img.width(), img.height());
    
    let max_size = 640u32;
    let scale = if orig_width > max_size || orig_height > max_size {
        max_size as f32 / orig_width.max(orig_height) as f32
    } else {
        1.0
    };
    
    let input_width = (orig_width as f32 * scale) as u32;
    let input_height = (orig_height as f32 * scale) as u32;
    
    log::info!("DexiNed 输入尺寸: {}x{} (缩放: {:.2})", input_width, input_height, scale);
    
    let resized = img.resize_exact(input_width, input_height, image::imageops::FilterType::Triangle);
    let rgba = resized.to_rgba8();
    
    let mut input_data = Vec::with_capacity((input_width * input_height * 3) as usize);
    for pixel in rgba.pixels() {
        input_data.push(pixel[0] as f32 / 255.0);
        input_data.push(pixel[1] as f32 / 255.0);
        input_data.push(pixel[2] as f32 / 255.0);
    }
    
    let input_shape = [1usize, 3usize, input_height as usize, input_width as usize];
    let input_tensor = Tensor::from_array((input_shape, input_data.into_boxed_slice()))
        .map_err(|e| format!("创建输入tensor失败: {}", e))?;
    
    let mut session = session.lock().unwrap();
    let outputs = session.run(ort::inputs![input_tensor])
        .map_err(|e| format!("DexiNed 推理失败: {}", e))?;
    
    let (output_shape, output_data) = outputs[0].try_extract_tensor::<f32>()
        .map_err(|e| format!("获取输出失败: {}", e))?;
    
    let output_dims: Vec<usize> = output_shape.iter().map(|d| *d as usize).collect();
    log::info!("DexiNed 输出维度: {:?}", output_dims);
    
    let (out_height, out_width) = if output_dims.len() == 4 {
        (output_dims[2], output_dims[3])
    } else if output_dims.len() == 3 {
        (output_dims[1], output_dims[2])
    } else {
        return Err(format!("不支持的输出维度: {:?}", output_dims));
    };
    
    let mut edge_image: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::new(out_width as u32, out_height as u32);
    
    let min_val = output_data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_val = output_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = max_val - min_val;
    
    for y in 0..out_height {
        for x in 0..out_width {
            let idx = y * out_width + x;
            let val = output_data[idx];
            let normalized = if range > 0.0 { (val - min_val) / range } else { 0.0 };
            let pixel_val = (normalized * 255.0).clamp(0.0, 255.0) as u8;
            edge_image.put_pixel(x as u32, y as u32, Luma([pixel_val]));
        }
    }
    
    let final_edge = image::imageops::resize(
        &edge_image,
        orig_width,
        orig_height,
        image::imageops::FilterType::Triangle
    );
    
    let corners = detect_document_boundary_opencv(&final_edge);
    
    match corners {
        Some(c) => {
            let transformed = perspective_transform(img, &c);
            Ok(transformed)
        }
        None => Ok(None)
    }
}

// ==================== OpenCV 文档增强 ====================

#[cfg(target_os = "windows")]
fn enhance_document_opencv(img: &DynamicImage, grayscale: bool) -> Result<DynamicImage, String> {
    if !grayscale {
        return Ok(img.clone());
    }
    
    let (orig_width, orig_height) = (img.width(), img.height());
    
    let max_process_size = 1280u32;
    let scale = if orig_width > max_process_size || orig_height > max_process_size {
        max_process_size as f32 / orig_width.max(orig_height) as f32
    } else {
        1.0
    };
    
    let process_width = (orig_width as f32 * scale) as u32;
    let process_height = (orig_height as f32 * scale) as u32;
    
    log::info!("黑白文档增强处理尺寸: {}x{} (缩放: {:.2})", process_width, process_height, scale);
    
    let processed_img = if scale < 1.0 {
        img.resize_exact(process_width, process_height, image::imageops::FilterType::Triangle)
    } else {
        img.clone()
    };
    
    let rgba = processed_img.to_rgba8();
    let (width, height) = rgba.dimensions();
    
    let mut bgr_mat = unsafe { Mat::new_rows_cols(height as i32, width as i32, CV_8UC3) }
        .map_err(|e| format!("创建 Mat 失败: {}", e))?;
    {
        let data = bgr_mat.data_bytes_mut()
            .map_err(|e| format!("获取 Mat 数据失败: {}", e))?;
        for y in 0..height {
            for x in 0..width {
                let pixel = rgba.get_pixel(x, y);
                let idx = (y * width + x) as usize * 3;
                data[idx] = pixel[2];
                data[idx + 1] = pixel[1];
                data[idx + 2] = pixel[0];
            }
        }
    }
    
    let mut gray = Mat::default();
    cvt_color(&bgr_mat, &mut gray, COLOR_BGR2GRAY, 0)
        .map_err(|e| format!("灰度转换失败: {}", e))?;
    
    let mut denoised = Mat::default();
    fast_nl_means_denoising(&gray, &mut denoised, 2.0, 7, 21)
        .map_err(|e| format!("黑白降噪失败: {}", e))?;
    
    let mut clahe = opencv::imgproc::create_clahe(3.0, opencv::core::Size::new(8, 8))
        .map_err(|e| format!("创建黑白 CLAHE 失败: {}", e))?;
    
    let mut enhanced = Mat::default();
    clahe.apply(&denoised, &mut enhanced)
        .map_err(|e| format!("黑白 CLAHE 增强失败: {}", e))?;
    
    let mut binary = Mat::default();
    threshold(&enhanced, &mut binary, 0.0, 255.0, THRESH_BINARY | THRESH_OTSU)
        .map_err(|e| format!("二值化失败: {}", e))?;
    
    let mut result_gray = Mat::default();
    opencv::core::add_weighted(
        &enhanced,
        0.7,
        &binary,
        0.3,
        0.0,
        &mut result_gray,
        CV_8UC1
    ).map_err(|e| format!("混合失败: {}", e))?;
    
    let sharpen_kernel = Mat::from_slice_2d(&[
        [0.0f64, -1.0, 0.0],
        [-1.0, 5.0, -1.0],
        [0.0, -1.0, 0.0]
    ]).map_err(|e| format!("创建锐化核失败: {}", e))?;
    
    let mut sharpened_gray = Mat::default();
    opencv::imgproc::filter_2d(
        &result_gray,
        &mut sharpened_gray,
        -1,
        &sharpen_kernel,
        opencv::core::Point::new(-1, -1),
        0.0,
        opencv::core::BORDER_DEFAULT
    ).map_err(|e| format!("黑白锐化失败: {}", e))?;
    
    let result_data = sharpened_gray.data_bytes()
        .map_err(|e| format!("获取黑白结果数据失败: {}", e))?;
    
    let mut result_img = ImageBuffer::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let val = result_data.get(idx).copied().unwrap_or(0);
            result_img.put_pixel(x, y, Luma([val]));
        }
    }
    
    let final_img = if scale < 1.0 {
        DynamicImage::ImageLuma8(result_img).resize_exact(orig_width, orig_height, image::imageops::FilterType::Triangle)
    } else {
        DynamicImage::ImageLuma8(result_img)
    };
    
    Ok(final_img)
}

#[cfg(not(target_os = "windows"))]
fn enhance_document_opencv(img: &DynamicImage, grayscale: bool) -> Result<DynamicImage, String> {
    if grayscale {
        Ok(img.grayscale())
    } else {
        Ok(img.clone())
    }
}

// ==================== DBNet 文本检测 ====================
// 使用 OpenCV DNN 模块实现 DBNet 文本检测（ONNX格式）

#[cfg(target_os = "windows")]
static DBNET_NET: std::sync::OnceLock<std::sync::Mutex<Option<Net>>> = std::sync::OnceLock::new();

#[cfg(target_os = "windows")]
fn get_dbnet_net(model_path: &str) -> Result<std::sync::MutexGuard<'static, Option<Net>>, String> {
    let net_guard = DBNET_NET.get_or_init(|| {
        match read_net_from_onnx(model_path) {
            Ok(net) => {
                log::info!("DBNet ONNX 模型加载成功: {}", model_path);
                std::sync::Mutex::new(Some(net))
            }
            Err(e) => {
                log::error!("DBNet ONNX 模型加载失败: {}", e);
                std::sync::Mutex::new(None)
            }
        }
    });
    
    net_guard.lock().map_err(|e| format!("获取模型锁失败: {}", e))
}

#[cfg(target_os = "windows")]
fn detect_text_regions_dbnet(
    img: &DynamicImage, 
    model_path: &str, 
    binary_threshold: f32
) -> Result<Option<(i32, i32, i32, i32)>, String> {
    let mut net_guard = get_dbnet_net(model_path)?;
    let net = match net_guard.as_mut() {
        Some(n) => n,
        None => return Err("DBNet 模型未加载".to_string()),
    };
    
    let (orig_width, orig_height) = (img.width() as i32, img.height() as i32);
    
    let target_size = 640i32;
    let scale = target_size as f32 / orig_width.max(orig_height) as f32;
    let new_width = (orig_width as f32 * scale) as i32;
    let new_height = (orig_height as f32 * scale) as i32;
    
    let rw = orig_width as f32 / new_width as f32;
    let rh = orig_height as f32 / new_height as f32;
    
    let rgba = img.to_rgba8();
    let (width, height) = rgba.dimensions();
    
    let mut bgr_mat = unsafe { Mat::new_rows_cols(height as i32, width as i32, opencv::core::CV_8UC3) }
        .map_err(|e| format!("创建 Mat 失败: {}", e))?;
    {
        let data = bgr_mat.data_bytes_mut()
            .map_err(|e| format!("获取 Mat 数据失败: {}", e))?;
        for y in 0..height {
            for x in 0..width {
                let pixel = rgba.get_pixel(x, y);
                let idx = (y * width + x) as usize * 3;
                data[idx] = pixel[2];
                data[idx + 1] = pixel[1];
                data[idx + 2] = pixel[0];
            }
        }
    }
    
    let mut resized = Mat::default();
    resize(&bgr_mat, &mut resized, Size::new(new_width, new_height), 0.0, 0.0, opencv::imgproc::INTER_LINEAR)
        .map_err(|e| format!("调整大小失败: {}", e))?;
    
    let blob = blob_from_image(
        &resized,
        1.0,
        Size::new(new_width, new_height),
        Scalar::new(123.675, 116.28, 103.53, 0.0),
        true,
        false,
        CV_32F,
    ).map_err(|e| format!("创建 blob 失败: {}", e))?;
    
    log::info!("DBNet blob 尺寸: {}x{}", new_width, new_height);
    
    net.set_input(&blob, "", 1.0, Scalar::default())
        .map_err(|e| format!("设置输入失败: {}", e))?;
    
    let mut outputs = Vector::<Mat>::new();
    net.forward(&mut outputs, &Vector::<String>::new())
        .map_err(|e| format!("前向传播失败: {}", e))?;
    
    log::info!("DBNet 输出数量: {}", outputs.len());
    
    if outputs.is_empty() {
        return Err("DBNet 没有输出".to_string());
    }
    
    let probability_map = outputs.get(0).map_err(|e| format!("获取概率图失败: {}", e))?;
    
    let dims = probability_map.dims();
    log::info!("DBNet 输出维度: {:?}", dims);
    
    let mut binary_map = Mat::default();
    threshold(&probability_map, &mut binary_map, binary_threshold as f64, 255.0, opencv::imgproc::THRESH_BINARY)
        .map_err(|e| format!("二值化失败: {}", e))?;
    
    let mut contours = Vector::<Mat>::new();
    find_contours(
        &binary_map,
        &mut contours,
        opencv::imgproc::RETR_LIST,
        opencv::imgproc::CHAIN_APPROX_SIMPLE,
        Point::new(0, 0)
    ).map_err(|e| format!("轮廓检测失败: {}", e))?;
    
    let mut text_regions: Vec<(i32, i32, i32, i32)> = Vec::new();
    
    for i in 0..contours.len() {
        let contour = contours.get(i).map_err(|e| format!("获取轮廓失败: {}", e))?;
        let area = contour_area(&contour, false).map_err(|e| format!("计算面积失败: {}", e))?;
        
        if area < 100.0 {
            continue;
        }
        
        let bounding_rect = opencv::imgproc::bounding_rect(&contour)
            .map_err(|e| format!("计算边界矩形失败: {}", e))?;
        
        let x1 = (bounding_rect.x as f32 * rw) as i32;
        let y1 = (bounding_rect.y as f32 * rh) as i32;
        let x2 = ((bounding_rect.x + bounding_rect.width) as f32 * rw) as i32;
        let y2 = ((bounding_rect.y + bounding_rect.height) as f32 * rh) as i32;
        
        text_regions.push((x1, y1, x2, y2));
    }
    
    log::info!("DBNet 检测到 {} 个文本区域", text_regions.len());
    
    if text_regions.is_empty() {
        return Ok(None);
    }
    
    let min_x = text_regions.iter().map(|p| p.0).min().unwrap_or(0);
    let min_y = text_regions.iter().map(|p| p.1).min().unwrap_or(0);
    let max_x = text_regions.iter().map(|p| p.2).max().unwrap_or(orig_width);
    let max_y = text_regions.iter().map(|p| p.3).max().unwrap_or(orig_height);
    
    let margin = 20;
    let x1 = (min_x - margin).max(0);
    let y1 = (min_y - margin).max(0);
    let x2 = (max_x + margin).min(orig_width);
    let y2 = (max_y + margin).min(orig_height);
    
    let bbox = (x1, y1, x2, y2);
    
    log::info!("DBNet 最终边界框: {:?}", bbox);
    
    Ok(Some(bbox))
}

#[cfg(not(target_os = "windows"))]
fn detect_text_regions_dbnet(_img: &DynamicImage, _model_path: &str, _binary_threshold: f32) -> Result<Option<(i32, i32, i32, i32)>, String> {
    Err("DBNet 文本检测仅支持 Windows 系统".to_string())
}

// ==================== ONNX Runtime DBNet 文本检测 ====================
// 使用 ort (ONNX Runtime) 实现 DBNet 文本检测

fn detect_text_regions_dbnet_ort(
    img: &DynamicImage,
    model_path: &str,
    binary_threshold: f32
) -> Result<Option<(i32, i32, i32, i32)>, String> {
    use ort::value::Tensor;
    
    let session = get_or_load_dbnet_ort_model(std::path::Path::new(model_path))?;
    let mut session = session.lock().unwrap();
    
    let (orig_width, orig_height) = (img.width() as i32, img.height() as i32);
    
    let target_size = 800i32;
    let scale = target_size as f32 / orig_width.max(orig_height) as f32;
    let new_width = (orig_width as f32 * scale) as i32;
    let new_height = (orig_height as f32 * scale) as i32;
    
    let rw = orig_width as f32 / new_width as f32;
    let rh = orig_height as f32 / new_height as f32;
    
    let resized = img.resize_exact(new_width as u32, new_height as u32, image::imageops::FilterType::Triangle);
    
    let mut input_data = vec![0.0f32; (target_size * target_size * 3) as usize];
    
    let rgba = resized.to_rgba8();
    for (y, row) in rgba.rows().enumerate() {
        for (x, pixel) in row.enumerate() {
            let r = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;
            let b = pixel[2] as f32 / 255.0;
            
            let base_idx = (y * target_size as usize + x) * 3;
            input_data[base_idx] = (r - 0.485) / 0.229;
            input_data[base_idx + 1] = (g - 0.456) / 0.224;
            input_data[base_idx + 2] = (b - 0.406) / 0.225;
        }
    }
    
    let input_shape = [1usize, 3usize, target_size as usize, target_size as usize];
    let input_tensor = Tensor::from_array((input_shape, input_data.into_boxed_slice()))
        .map_err(|e| format!("创建输入tensor失败: {}", e))?;
    
    log::info!("ONNX Runtime 开始推理");
    
    let outputs = session.run(ort::inputs![input_tensor])
        .map_err(|e| format!("ONNX Runtime 推理失败: {}", e))?;
    
    let (output_shape, output_data) = outputs[0].try_extract_tensor::<f32>()
        .map_err(|e| format!("获取输出失败: {}", e))?;
    
    let output_dims: Vec<usize> = output_shape.iter().map(|d| *d as usize).collect();
    log::info!("ONNX Runtime 输出维度: {:?}", output_dims);
    
    let out_height = output_dims[2];
    let out_width = output_dims[3];
    
    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;
    let mut sum_val = 0.0f32;
    let total_pixels = out_height * out_width;
    
    for &v in output_data.iter().take(total_pixels) {
        if v < min_val { min_val = v; }
        if v > max_val { max_val = v; }
        sum_val += v;
    }
    log::info!("ONNX Runtime 输出值范围: min={}, max={}, avg={}", min_val, max_val, sum_val / total_pixels as f32);
    
    let scale_to_output = out_height as f32 / target_size as f32;
    let valid_width = (new_width as f32 * scale_to_output) as usize;
    let valid_height = (new_height as f32 * scale_to_output) as usize;
    
    let mut text_regions: Vec<(i32, i32, i32, i32)> = Vec::new();
    let threshold = binary_threshold;
    
    log::info!("使用阈值: {}, 有效区域: {}x{}", threshold, valid_width, valid_height);
    
    for y in 0..valid_height {
        for x in 0..valid_width {
            let idx = y * out_width + x;
            let prob = output_data[idx];
            if prob > threshold {
                let x1 = (x as f32 / scale_to_output * rw) as i32;
                let y1 = (y as f32 / scale_to_output * rh) as i32;
                let x2 = ((x + 1) as f32 / scale_to_output * rw) as i32;
                let y2 = ((y + 1) as f32 / scale_to_output * rh) as i32;
                
                text_regions.push((x1, y1, x2, y2));
            }
        }
    }
    
    log::info!("ONNX Runtime 检测到 {} 个文本区域", text_regions.len());
    
    if text_regions.is_empty() {
        return Ok(None);
    }
    
    let min_x = text_regions.iter().map(|p| p.0).min().unwrap_or(0);
    let min_y = text_regions.iter().map(|p| p.1).min().unwrap_or(0);
    let max_x = text_regions.iter().map(|p| p.2).max().unwrap_or(orig_width);
    let max_y = text_regions.iter().map(|p| p.3).max().unwrap_or(orig_height);
    
    let margin = 20;
    let x1 = (min_x - margin).max(0);
    let y1 = (min_y - margin).max(0);
    let x2 = (max_x + margin).min(orig_width);
    let y2 = (max_y + margin).min(orig_height);
    
    let bbox = (x1, y1, x2, y2);
    
    log::info!("ONNX Runtime 最终边界框: {:?}", bbox);
    
    Ok(Some(bbox))
}

// ==================== 模型资源管理 ====================
// 模型下载和管理功能

/// 模型信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub size_mb: f64,
    pub description: String,
    pub download_url: String,
    pub exists: bool,
}

/// 检查 DBNet 模型是否存在
#[tauri::command]
fn check_dbnet_model_exists(app: tauri::AppHandle) -> Result<bool, String> {
    let models_dir = get_models_dir(&app)?;
    let model_path = models_dir.join("text_detection_db_TD500_resnet18.onnx");
    Ok(model_path.exists())
}

/// 获取 DBNet 模型信息
#[tauri::command]
fn get_dbnet_model_info(app: tauri::AppHandle) -> Result<ModelInfo, String> {
    let models_dir = get_models_dir(&app)?;
    let model_path = models_dir.join("text_detection_db_TD500_resnet18.onnx");
    
    let exists = model_path.exists();
    let size_mb = if exists {
        let metadata = std::fs::metadata(&model_path)
            .map_err(|e| format!("获取模型文件信息失败: {}", e))?;
        metadata.len() as f64 / 1024.0 / 1024.0
    } else {
        50.0
    };
    
    Ok(ModelInfo {
        name: "DBNet ResNet-18".to_string(),
        size_mb,
        description: "文本检测模型，用于文档扫描功能。相比EAST模型，体积更小（54MB vs 92MB），速度更快，精度更高。支持中英文文本检测。".to_string(),
        download_url: "https://modelscope.cn/models/iic/cv_resnet18_ocr-detection-db-line-level_damo/resolve/master/db_resnet18_public_line_640x640.onnx".to_string(),
        exists,
    })
}

/// 下载 DBNet 模型（带进度）
#[tauri::command]
async fn download_dbnet_model(app: tauri::AppHandle, window: tauri::Window) -> Result<(), String> {
    let models_dir = get_models_dir(&app)?;
    let model_path = models_dir.join("text_detection_db_TD500_resnet18.onnx");
    
    if model_path.exists() {
        return Err("模型文件已存在".to_string());
    }
    
    let url = "https://modelscope.cn/models/iic/cv_resnet18_ocr-detection-db-line-level_damo/resolve/master/db_resnet18_public_line_640x640.onnx";
    
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .map_err(|e| format!("创建HTTP客户端失败: {}", e))?;
    
    let response = client
        .get(url)
        .header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        .send()
        .await
        .map_err(|e| format!("下载请求失败: {}", e))?;
    
    if !response.status().is_success() {
        return Err(format!("下载失败，HTTP状态码: {}", response.status()));
    }
    
    let total_size = response.content_length().unwrap_or(0);
    
    if total_size == 0 {
        return Err("无法获取文件大小，可能链接无效".to_string());
    }
    
    let mut downloaded = 0u64;
    let mut stream = response.bytes_stream();
    use futures::StreamExt;
    
    let mut file = std::fs::File::create(&model_path)
        .map_err(|e| format!("创建文件失败: {}", e))?;
    
    use std::io::Write;
    
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("读取数据失败: {}", e))?;
        file.write_all(&chunk).map_err(|e| format!("写入文件失败: {}", e))?;
        
        downloaded += chunk.len() as u64;
        
        if total_size > 0 {
            let progress = (downloaded as f64 / total_size as f64 * 100.0) as u32;
            let _ = window.emit("download-progress", progress);
        }
    }
    
    let _ = window.emit("download-complete", ());
    
    Ok(())
}

/// 删除 DBNet 模型
#[tauri::command]
fn delete_dbnet_model(app: tauri::AppHandle) -> Result<(), String> {
    let models_dir = get_models_dir(&app)?;
    let model_path = models_dir.join("text_detection_db_TD500_resnet18.onnx");
    
    if model_path.exists() {
        std::fs::remove_file(&model_path)
            .map_err(|e| format!("删除模型文件失败: {}", e))?;
    }
    
    Ok(())
}

// ==================== UVDoc 文档矫正模型管理 ====================

/// 检查 UVDoc 模型是否存在
#[tauri::command]
fn check_uvdoc_model_exists(app: tauri::AppHandle) -> Result<bool, String> {
    let models_dir = get_models_dir(&app)?;
    let model_path = models_dir.join("uvdoc.onnx");
    Ok(model_path.exists())
}

/// 获取 UVDoc 模型信息
#[tauri::command]
fn get_uvdoc_model_info(app: tauri::AppHandle) -> Result<ModelInfo, String> {
    let models_dir = get_models_dir(&app)?;
    let model_path = models_dir.join("uvdoc.onnx");
    
    let exists = model_path.exists();
    let size_mb = if exists {
        let metadata = std::fs::metadata(&model_path)
            .map_err(|e| format!("获取模型文件信息失败: {}", e))?;
        metadata.len() as f64 / 1024.0 / 1024.0
    } else {
        30.0
    };
    
    Ok(ModelInfo {
        name: "UVDoc".to_string(),
        size_mb,
        description: "文档扭曲矫正模型，用于矫正弯曲、折叠的文档图像。".to_string(),
        download_url: "https://modelscope.cn/models/PaddlePaddle/UVDoc/resolve/master/inference.pdiparams".to_string(),
        exists,
    })
}

/// 下载 UVDoc 模型（带进度）
#[tauri::command]
async fn download_uvdoc_model(app: tauri::AppHandle, window: tauri::Window) -> Result<(), String> {
    let models_dir = get_models_dir(&app)?;
    let model_path = models_dir.join("uvdoc.onnx");
    
    if model_path.exists() {
        return Err("模型文件已存在".to_string());
    }
    
    let url = "https://modelscope.cn/models/PaddlePaddle/UVDoc/resolve/master/inference.pdiparams";
    
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .map_err(|e| format!("创建HTTP客户端失败: {}", e))?;
    
    let response = client
        .get(url)
        .header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        .send()
        .await
        .map_err(|e| format!("下载请求失败: {}", e))?;
    
    if !response.status().is_success() {
        return Err(format!("下载失败，HTTP状态码: {}", response.status()));
    }
    
    let total_size = response.content_length().unwrap_or(0);
    
    if total_size == 0 {
        return Err("无法获取文件大小，可能链接无效".to_string());
    }
    
    let mut downloaded = 0u64;
    let mut stream = response.bytes_stream();
    use futures::StreamExt;
    
    let mut file = std::fs::File::create(&model_path)
        .map_err(|e| format!("创建文件失败: {}", e))?;
    
    use std::io::Write;
    
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("读取数据失败: {}", e))?;
        file.write_all(&chunk).map_err(|e| format!("写入文件失败: {}", e))?;
        
        downloaded += chunk.len() as u64;
        
        if total_size > 0 {
            let progress = (downloaded as f64 / total_size as f64 * 100.0) as u32;
            let _ = window.emit("uvdoc-download-progress", progress);
        }
    }
    
    let _ = window.emit("uvdoc-download-complete", ());
    
    Ok(())
}

/// 删除 UVDoc 模型
#[tauri::command]
fn delete_uvdoc_model(app: tauri::AppHandle) -> Result<(), String> {
    let models_dir = get_models_dir(&app)?;
    let model_path = models_dir.join("uvdoc.onnx");
    
    if model_path.exists() {
        std::fs::remove_file(&model_path)
            .map_err(|e| format!("删除模型文件失败: {}", e))?;
    }
    
    Ok(())
}

// ==================== 文档增强模型管理 ====================

/// 检查文档增强模型是否存在
#[tauri::command]
fn check_enhance_model_exists(app: tauri::AppHandle) -> Result<bool, String> {
    let models_dir = get_models_dir(&app)?;
    let gcnet_path = models_dir.join("gcnet.onnx");
    let nafdpm_path = models_dir.join("nafdpm.onnx");
    Ok(gcnet_path.exists() && nafdpm_path.exists())
}

/// 获取文档增强模型信息
#[tauri::command]
fn get_enhance_model_info(app: tauri::AppHandle) -> Result<ModelInfo, String> {
    let models_dir = get_models_dir(&app)?;
    let gcnet_path = models_dir.join("gcnet.onnx");
    let nafdpm_path = models_dir.join("nafdpm.onnx");
    
    let exists = gcnet_path.exists() && nafdpm_path.exists();
    let mut total_size = 0.0;
    
    if gcnet_path.exists() {
        if let Ok(metadata) = std::fs::metadata(&gcnet_path) {
            total_size += metadata.len() as f64 / 1024.0 / 1024.0;
        }
    }
    if nafdpm_path.exists() {
        if let Ok(metadata) = std::fs::metadata(&nafdpm_path) {
            total_size += metadata.len() as f64 / 1024.0 / 1024.0;
        }
    }
    
    Ok(ModelInfo {
        name: "文档增强模型包".to_string(),
        size_mb: if total_size > 0.0 { total_size } else { 50.0 },
        description: "包含去阴影(GCNet)和去模糊(NAFDPM)模型，用于文档图像增强。".to_string(),
        download_url: "".to_string(),
        exists,
    })
}

/// 下载文档增强模型（带进度）
#[tauri::command]
async fn download_enhance_model(app: tauri::AppHandle, window: tauri::Window) -> Result<(), String> {
    let models_dir = get_models_dir(&app)?;
    
    let models = vec![
        ("gcnet.onnx", "https://modelscope.cn/models/RapidAI/RapidUnDistort/resolve/master/models/gcnet.onnx"),
        ("nafdpm.onnx", "https://modelscope.cn/models/RapidAI/RapidUnDistort/resolve/master/models/nafdpm.onnx"),
    ];
    
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .map_err(|e| format!("创建HTTP客户端失败: {}", e))?;
    
    let total_models = models.len();
    let mut completed = 0;
    
    for (filename, url) in models {
        let model_path = models_dir.join(filename);
        
        if model_path.exists() {
            completed += 1;
            continue;
        }
        
        let response = client
            .get(url)
            .header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
            .send()
            .await
            .map_err(|e| format!("下载 {} 失败: {}", filename, e))?;
        
        if !response.status().is_success() {
            return Err(format!("下载 {} 失败，HTTP状态码: {}", filename, response.status()));
        }
        
        let total_size = response.content_length().unwrap_or(0);
        let mut downloaded = 0u64;
        let mut stream = response.bytes_stream();
        use futures::StreamExt;
        
        let mut file = std::fs::File::create(&model_path)
            .map_err(|e| format!("创建文件 {} 失败: {}", filename, e))?;
        
        use std::io::Write;
        
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| format!("读取数据失败: {}", e))?;
            file.write_all(&chunk).map_err(|e| format!("写入文件失败: {}", e))?;
            downloaded += chunk.len() as u64;
            
            if total_size > 0 {
                let file_progress = downloaded as f64 / total_size as f64;
                let overall_progress = ((completed as f64 + file_progress) / total_models as f64 * 100.0) as u32;
                let _ = window.emit("enhance-download-progress", overall_progress);
            }
        }
        
        completed += 1;
        let overall_progress = (completed as f64 / total_models as f64 * 100.0) as u32;
        let _ = window.emit("enhance-download-progress", overall_progress);
    }
    
    let _ = window.emit("enhance-download-complete", ());
    
    Ok(())
}

/// 删除文档增强模型
#[tauri::command]
fn delete_enhance_model(app: tauri::AppHandle) -> Result<(), String> {
    let models_dir = get_models_dir(&app)?;
    
    let models = vec!["gcnet.onnx", "nafdpm.onnx", "drnet.onnx", "unetcnn.onnx"];
    
    for model_name in models {
        let model_path = models_dir.join(model_name);
        if model_path.exists() {
            std::fs::remove_file(&model_path)
                .map_err(|e| format!("删除模型文件失败: {}", e))?;
        }
    }
    
    Ok(())
}

// ==================== UVDoc 文档扭曲矫正推理 ====================

/// 使用 UVDoc 模型矫正扭曲文档
fn unwarp_document_uvdoc(
    img: &DynamicImage,
    model_path: &str,
) -> Result<DynamicImage, String> {
    use ort::value::Tensor;
    
    let session = get_or_load_uvdoc_model(std::path::Path::new(model_path))?;
    let mut session = session.lock().unwrap();
    
    let (orig_width, orig_height) = (img.width(), img.height());
    
    let target_size = 288usize;
    let resized = img.resize_exact(target_size as u32, target_size as u32, image::imageops::FilterType::Triangle);
    
    let rgba = resized.to_rgba8();
    let mut input_data = vec![0.0f32; target_size * target_size * 3];
    
    for (y, row) in rgba.rows().enumerate() {
        for (x, pixel) in row.enumerate() {
            let r = pixel[0] as f32 / 255.0;
            let g = pixel[1] as f32 / 255.0;
            let b = pixel[2] as f32 / 255.0;
            
            let base_idx = (y * target_size + x) * 3;
            input_data[base_idx] = r;
            input_data[base_idx + 1] = g;
            input_data[base_idx + 2] = b;
        }
    }
    
    let input_shape = [1usize, 3usize, target_size, target_size];
    let input_tensor = Tensor::from_array((input_shape, input_data.into_boxed_slice()))
        .map_err(|e| format!("创建输入tensor失败: {}", e))?;
    
    log::info!("UVDoc 开始推理");
    
    let outputs = session.run(ort::inputs![input_tensor])
        .map_err(|e| format!("UVDoc 推理失败: {}", e))?;
    
    let (grid_shape, grid_data) = outputs[0].try_extract_tensor::<f32>()
        .map_err(|e| format!("获取输出失败: {}", e))?;
    
    let grid_dims: Vec<usize> = grid_shape.iter().map(|d| *d as usize).collect();
    log::info!("UVDoc 输出维度: {:?}", grid_dims);
    
    let grid_height = grid_dims[2];
    let grid_width = grid_dims[3];
    
    let mut result = image::RgbaImage::new(orig_width, orig_height);
    
    let scale_x = orig_width as f32 / (grid_width - 1) as f32;
    let scale_y = orig_height as f32 / (grid_height - 1) as f32;
    
    let orig_rgba = img.to_rgba8();
    
    for y in 0..orig_height {
        for x in 0..orig_width {
            let src_x_f = x as f32 / scale_x;
            let src_y_f = y as f32 / scale_y;
            
            let src_x = src_x_f.min((grid_width - 1) as f32).max(0.0) as usize;
            let src_y = src_y_f.min((grid_height - 1) as f32).max(0.0) as usize;
            
            let grid_idx = src_y * grid_width + src_x;
            
            let coord_x = grid_data[grid_idx * 2] * orig_width as f32;
            let coord_y = grid_data[grid_idx * 2 + 1] * orig_height as f32;
            
            let sample_x = coord_x.clamp(0.0, (orig_width - 1) as f32) as u32;
            let sample_y = coord_y.clamp(0.0, (orig_height - 1) as f32) as u32;
            
            let pixel = orig_rgba.get_pixel(sample_x, sample_y);
            result.put_pixel(x, y, *pixel);
        }
    }
    
    log::info!("UVDoc 矫正完成");
    
    Ok(DynamicImage::ImageRgba8(result))
}

/// 文档扭曲矫正命令
#[tauri::command]
fn unwarp_document(app: tauri::AppHandle, request: DocumentScanRequest) -> Result<DocumentScanResult, String> {
    let img = decode_base64_image(&request.image_data)?;
    
    log::info!("开始文档扭曲矫正，图像尺寸: {}x{}", img.width(), img.height());
    
    let models_dir = get_models_dir(&app)?;
    
    let uvdoc_model_path = models_dir.join("uvdoc.onnx");
    
    if !uvdoc_model_path.exists() {
        return Err("UVDoc 模型未安装，请在设置中下载".to_string());
    }
    
    let result_img = unwarp_document_uvdoc(&img, uvdoc_model_path.to_string_lossy().to_string().as_str())?;
    
    let mut buffer = Vec::new();
    result_img
        .write_to(&mut std::io::Cursor::new(&mut buffer), image::ImageFormat::Png)
        .map_err(|e| format!("Failed to encode image: {}", e))?;
    
    let result_image = format!("data:image/png;base64,{}", general_purpose::STANDARD.encode(&buffer));
    
    Ok(DocumentScanResult {
        enhanced_image: result_image,
        confidence: 1.0,
        text_bbox: None,
    })
}

// ==================== 文档增强推理 ====================

/// 使用 GCNet 模型去除阴影
fn remove_shadow_gcnet(
    img: &DynamicImage,
    model_path: &str,
) -> Result<DynamicImage, String> {
    use ort::value::Tensor;
    
    let session = get_or_load_gcnet_model(std::path::Path::new(model_path))?;
    let mut session = session.lock().unwrap();
    
    let (orig_width, orig_height) = (img.width(), img.height());
    
    let target_size = 640usize;
    let scale_x = orig_width as f32 / target_size as f32;
    let scale_y = orig_height as f32 / target_size as f32;
    
    let resized = img.resize_exact(target_size as u32, target_size as u32, image::imageops::FilterType::Triangle);
    
    let rgba = resized.to_rgba8();
    let mut input_data = vec![0.0f32; target_size * target_size * 3];
    
    for (y, row) in rgba.rows().enumerate() {
        for (x, pixel) in row.enumerate() {
            let base_idx = (y * target_size + x) * 3;
            input_data[base_idx] = pixel[0] as f32 / 255.0;
            input_data[base_idx + 1] = pixel[1] as f32 / 255.0;
            input_data[base_idx + 2] = pixel[2] as f32 / 255.0;
        }
    }
    
    let input_shape = [1usize, 3usize, target_size, target_size];
    let input_tensor = Tensor::from_array((input_shape, input_data.into_boxed_slice()))
        .map_err(|e| format!("创建输入tensor失败: {}", e))?;
    
    log::info!("GCNet 开始推理");
    
    let outputs = session.run(ort::inputs![input_tensor])
        .map_err(|e| format!("GCNet 推理失败: {}", e))?;
    
    let (output_shape, output_data) = outputs[0].try_extract_tensor::<f32>()
        .map_err(|e| format!("获取输出失败: {}", e))?;
    
    let out_dims: Vec<usize> = output_shape.iter().map(|d| *d as usize).collect();
    log::info!("GCNet 输出维度: {:?}", out_dims);
    
    let out_height = out_dims[2];
    let out_width = out_dims[3];
    
    let mut result = image::RgbaImage::new(orig_width, orig_height);
    let orig_rgba = img.to_rgba8();
    
    for y in 0..orig_height {
        for x in 0..orig_width {
            let src_x = (x as f32 / scale_x) as usize;
            let src_y = (y as f32 / scale_y) as usize;
            
            let src_x = src_x.min(out_width - 1);
            let src_y = src_y.min(out_height - 1);
            
            let idx = src_y * out_width + src_x;
            
            let r = (output_data[idx * 3] * 255.0).clamp(0.0, 255.0) as u8;
            let g = (output_data[idx * 3 + 1] * 255.0).clamp(0.0, 255.0) as u8;
            let b = (output_data[idx * 3 + 2] * 255.0).clamp(0.0, 255.0) as u8;
            
            let orig_pixel = orig_rgba.get_pixel(x, y);
            result.put_pixel(x, y, image::Rgba([r, g, b, orig_pixel[3]]));
        }
    }
    
    log::info!("GCNet 去阴影完成");
    
    Ok(DynamicImage::ImageRgba8(result))
}

/// 使用 NAFDPM 模型去除模糊
fn remove_blur_nafdpm(
    img: &DynamicImage,
    model_path: &str,
) -> Result<DynamicImage, String> {
    use ort::value::Tensor;
    
    let session = get_or_load_nafdpm_model(std::path::Path::new(model_path))?;
    let mut session = session.lock().unwrap();
    
    let (orig_width, orig_height) = (img.width(), img.height());
    
    let target_size = 256usize;
    let scale_x = orig_width as f32 / target_size as f32;
    let scale_y = orig_height as f32 / target_size as f32;
    
    let resized = img.resize_exact(target_size as u32, target_size as u32, image::imageops::FilterType::Triangle);
    
    let rgba = resized.to_rgba8();
    let mut input_data = vec![0.0f32; target_size * target_size * 3];
    
    for (y, row) in rgba.rows().enumerate() {
        for (x, pixel) in row.enumerate() {
            let base_idx = (y * target_size + x) * 3;
            input_data[base_idx] = pixel[0] as f32 / 255.0;
            input_data[base_idx + 1] = pixel[1] as f32 / 255.0;
            input_data[base_idx + 2] = pixel[2] as f32 / 255.0;
        }
    }
    
    let input_shape = [1usize, 3usize, target_size, target_size];
    let input_tensor = Tensor::from_array((input_shape, input_data.into_boxed_slice()))
        .map_err(|e| format!("创建输入tensor失败: {}", e))?;
    
    log::info!("NAFDPM 开始推理");
    
    let outputs = session.run(ort::inputs![input_tensor])
        .map_err(|e| format!("NAFDPM 推理失败: {}", e))?;
    
    let (output_shape, output_data) = outputs[0].try_extract_tensor::<f32>()
        .map_err(|e| format!("获取输出失败: {}", e))?;
    
    let out_dims: Vec<usize> = output_shape.iter().map(|d| *d as usize).collect();
    log::info!("NAFDPM 输出维度: {:?}", out_dims);
    
    let out_height = out_dims[2];
    let out_width = out_dims[3];
    
    let mut result = image::RgbaImage::new(orig_width, orig_height);
    let orig_rgba = img.to_rgba8();
    
    for y in 0..orig_height {
        for x in 0..orig_width {
            let src_x = (x as f32 / scale_x) as usize;
            let src_y = (y as f32 / scale_y) as usize;
            
            let src_x = src_x.min(out_width - 1);
            let src_y = src_y.min(out_height - 1);
            
            let idx = src_y * out_width + src_x;
            
            let r = (output_data[idx * 3] * 255.0).clamp(0.0, 255.0) as u8;
            let g = (output_data[idx * 3 + 1] * 255.0).clamp(0.0, 255.0) as u8;
            let b = (output_data[idx * 3 + 2] * 255.0).clamp(0.0, 255.0) as u8;
            
            let orig_pixel = orig_rgba.get_pixel(x, y);
            result.put_pixel(x, y, image::Rgba([r, g, b, orig_pixel[3]]));
        }
    }
    
    log::info!("NAFDPM 去模糊完成");
    
    Ok(DynamicImage::ImageRgba8(result))
}

/// 文档增强命令（去阴影+去模糊）
#[tauri::command]
fn enhance_document(app: tauri::AppHandle, request: DocumentScanRequest, enable_unshadow: bool, enable_unblur: bool) -> Result<DocumentScanResult, String> {
    let mut img = decode_base64_image(&request.image_data)?;
    
    log::info!("开始文档增强，图像尺寸: {}x{}", img.width(), img.height());
    
    let models_dir = get_models_dir(&app)?;
    
    if enable_unshadow {
        let gcnet_path = models_dir.join("gcnet.onnx");
        if gcnet_path.exists() {
            match remove_shadow_gcnet(&img, gcnet_path.to_string_lossy().to_string().as_str()) {
                Ok(enhanced) => {
                    img = enhanced;
                    log::info!("去阴影完成");
                }
                Err(e) => {
                    log::warn!("去阴影失败: {}", e);
                }
            }
        } else {
            log::warn!("GCNet 模型未安装，跳过去阴影");
        }
    }
    
    if enable_unblur {
        let nafdpm_path = models_dir.join("nafdpm.onnx");
        if nafdpm_path.exists() {
            match remove_blur_nafdpm(&img, nafdpm_path.to_string_lossy().to_string().as_str()) {
                Ok(enhanced) => {
                    img = enhanced;
                    log::info!("去模糊完成");
                }
                Err(e) => {
                    log::warn!("去模糊失败: {}", e);
                }
            }
        } else {
            log::warn!("NAFDPM 模型未安装，跳过去模糊");
        }
    }
    
    let mut buffer = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut buffer), image::ImageFormat::Png)
        .map_err(|e| format!("Failed to encode image: {}", e))?;
    
    let result_image = format!("data:image/png;base64,{}", general_purpose::STANDARD.encode(&buffer));
    
    Ok(DocumentScanResult {
        enhanced_image: result_image,
        confidence: 1.0,
        text_bbox: None,
    })
}

// ==================== 高级文档增强算法 ====================
// 使用 imageproc 库实现

/// 光照归一化 - 去除不均匀光照和阴影（保留彩色）
/// 使用 imageproc 的 gaussian_blur_f32
fn normalize_illumination(img: &DynamicImage, sigma: f32) -> DynamicImage {
    let rgba = img.to_rgba8();
    let (width, height) = rgba.dimensions();
    
    let mut r_channel: GrayImage = ImageBuffer::new(width, height);
    let mut g_channel: GrayImage = ImageBuffer::new(width, height);
    let mut b_channel: GrayImage = ImageBuffer::new(width, height);
    
    for y in 0..height {
        for x in 0..width {
            let pixel = rgba.get_pixel(x, y);
            r_channel.put_pixel(x, y, Luma([pixel[0]]));
            g_channel.put_pixel(x, y, Luma([pixel[1]]));
            b_channel.put_pixel(x, y, Luma([pixel[2]]));
        }
    }
    
    let r_blurred = gaussian_blur_f32(&r_channel, sigma);
    let g_blurred = gaussian_blur_f32(&g_channel, sigma);
    let b_blurred = gaussian_blur_f32(&b_channel, sigma);
    
    let mut result = ImageBuffer::new(width, height);
    
    let mean_bg = 128.0f32;
    
    for y in 0..height {
        for x in 0..width {
            let pixel = rgba.get_pixel(x, y);
            let r_orig = pixel[0] as f32;
            let g_orig = pixel[1] as f32;
            let b_orig = pixel[2] as f32;
            let a = pixel[3];
            
            let r_bg = r_blurred.get_pixel(x, y)[0] as f32;
            let g_bg = g_blurred.get_pixel(x, y)[0] as f32;
            let b_bg = b_blurred.get_pixel(x, y)[0] as f32;
            
            let r_bg = r_bg.max(16.0);
            let g_bg = g_bg.max(16.0);
            let b_bg = b_bg.max(16.0);
            
            let r = (r_orig * mean_bg / r_bg).clamp(0.0, 255.0) as u8;
            let g = (g_orig * mean_bg / g_bg).clamp(0.0, 255.0) as u8;
            let b = (b_orig * mean_bg / b_bg).clamp(0.0, 255.0) as u8;
            
            result.put_pixel(x, y, Rgba([r, g, b, a]));
        }
    }
    
    DynamicImage::ImageRgba8(result)
}

/// 自适应二值化 - 自己实现，支持偏移参数
/// 类似 OpenCV 的 ADAPTIVE_THRESH_MEAN_C
/// block_size: 块大小（必须是奇数）
/// c: 从局部均值中减去的偏移值
fn adaptive_binarize_custom(img: &DynamicImage, block_size: u32, c: i32) -> DynamicImage {
    let gray = img.to_luma8();
    let (width, height) = gray.dimensions();
    
    let block_size = block_size.clamp(3, 99) | 1;
    let half = block_size / 2;
    
    let integral_width = width as usize + 1;
    let integral_height = height as usize + 1;
    let mut integral = vec![0u64; integral_width * integral_height];
    
    for y in 0..height {
        let mut row_sum = 0u64;
        for x in 0..width {
            row_sum += gray.get_pixel(x, y)[0] as u64;
            let idx = ((y + 1) as usize) * integral_width + ((x + 1) as usize);
            integral[idx] = integral[idx - integral_width] + row_sum;
        }
    }
    
    let mut result = ImageBuffer::new(width, height);
    
    for y in 0..height {
        for x in 0..width {
            let x1 = (x as i32 - half as i32).max(0) as u32;
            let y1 = (y as i32 - half as i32).max(0) as u32;
            let x2 = (x + half).min(width - 1);
            let y2 = (y + half).min(height - 1);
            
            let count = ((x2 - x1 + 1) * (y2 - y1 + 1)) as u64;
            
            let idx1 = (y1 as usize) * integral_width + (x1 as usize);
            let idx2 = (y1 as usize) * integral_width + ((x2 + 1) as usize);
            let idx3 = ((y2 + 1) as usize) * integral_width + (x1 as usize);
            let idx4 = ((y2 + 1) as usize) * integral_width + ((x2 + 1) as usize);
            
            let sum = integral[idx4] - integral[idx2] - integral[idx3] + integral[idx1];
            let mean = sum as f64 / count as f64;
            
            let pixel_val = f64::from(gray.get_pixel(x, y)[0]);
            let threshold = mean - f64::from(c);
            
            let value = if pixel_val > threshold { 255 } else { 0 };
            result.put_pixel(x, y, Luma([value]));
        }
    }
    
    DynamicImage::ImageLuma8(result)
}

/// 形态学闭运算 - 填补断裂
fn morphological_close(img: &GrayImage, kernel_size: u32) -> GrayImage {
    let dilated = dilate_gray(img, kernel_size);
    erode_gray(&dilated, kernel_size)
}

fn dilate_gray(img: &GrayImage, kernel_size: u32) -> GrayImage {
    let (width, height) = img.dimensions();
    let half = (kernel_size / 2) as i32;
    
    ImageBuffer::from_fn(width, height, |x, y| {
        let mut max_val = 0u8;
        for dy in -half..=half {
            for dx in -half..=half {
                let px = (x as i32 + dx).max(0).min(width as i32 - 1) as u32;
                let py = (y as i32 + dy).max(0).min(height as i32 - 1) as u32;
                let val = img.get_pixel(px, py)[0];
                if val > max_val { max_val = val; }
            }
        }
        Luma([max_val])
    })
}

fn erode_gray(img: &GrayImage, kernel_size: u32) -> GrayImage {
    let (width, height) = img.dimensions();
    let half = (kernel_size / 2) as i32;
    
    ImageBuffer::from_fn(width, height, |x, y| {
        let mut min_val = 255u8;
        for dy in -half..=half {
            for dx in -half..=half {
                let px = (x as i32 + dx).max(0).min(width as i32 - 1) as u32;
                let py = (y as i32 + dy).max(0).min(height as i32 - 1) as u32;
                let val = img.get_pixel(px, py)[0];
                if val < min_val { min_val = val; }
            }
        }
        Luma([min_val])
    })
}

/// 文档增强主函数
fn enhance_document_advanced_internal(img: &DynamicImage, binarize: bool) -> DynamicImage {
    let normalized = normalize_illumination(img, 25.0);
    
    if binarize {
        let binary = adaptive_binarize_custom(&normalized, 51, 15);
        let gray = binary.to_luma8();
        let refined = morphological_close(&gray, 3);
        DynamicImage::ImageLuma8(refined)
    } else {
        normalized
    }
}

/// 文档增强选项
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DocumentEnhanceOptions {
    pub binarize: bool,
}

/// 高级文档增强命令
#[tauri::command]
fn enhance_document_advanced(image_data: String, options: DocumentEnhanceOptions) -> Result<String, String> {
    let img = decode_base64_image(&image_data)?;
    
    let enhanced = enhance_document_advanced_internal(&img, options.binarize);
    
    let mut buffer = Vec::new();
    enhanced
        .write_to(&mut std::io::Cursor::new(&mut buffer), image::ImageFormat::Png)
        .map_err(|e| format!("Failed to encode image: {}", e))?;
    
    let result = format!("data:image/png;base64,{}", general_purpose::STANDARD.encode(&buffer));
    
    Ok(result)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    use simplelog::{CombinedLogger, WriteLogger, LevelFilter, Config, TermLogger, TerminalMode, ColorChoice};
    use std::fs::File;
    
    let config_dir = dirs::config_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("SECTL")
        .join("ViewStage");
    let log_dir = config_dir.join("log");
    
    if let Err(e) = std::fs::create_dir_all(&log_dir) {
        eprintln!("无法创建日志目录: {}", e);
    }
    
    let log_file = log_dir.join(format!("viewstage_{}.log", chrono::Local::now().format("%Y%m%d")));
    
    if let Ok(file) = File::create(&log_file) {
        let _ = CombinedLogger::init(vec![
            WriteLogger::new(LevelFilter::Info, Config::default(), file),
            TermLogger::new(LevelFilter::Info, Config::default(), TerminalMode::Mixed, ColorChoice::Auto),
        ]);
        log::info!("日志系统初始化成功");
    }
    
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_single_instance::init(|app, args, _cwd| {
            println!("单实例回调: args={:?}", args);
            if args.len() > 1 {
                let file_path = args[1].clone();
                println!("从第二个实例接收文件: {}", file_path);
                let _ = app.emit("file-opened", file_path);
            }
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.set_focus();
                let _ = window.unminimize();
            }
        }))
        .setup(|app| {
            let window = app.get_webview_window("main")
                .expect("Failed to get main window");
            
            let _ = window.set_decorations(false);
            
            let config_dir = app.path().app_config_dir()
                .expect("Failed to get config directory");
            let config_path = config_dir.join("config.json");
            
            let is_first_run = !config_path.exists();
            
            if is_first_run {
                println!("首次运行，打开 OOBE 界面");
                
                OOBE_ACTIVE.store(true, Ordering::SeqCst);
                
                use tauri::WebviewWindowBuilder;
                
                let oobe_window = WebviewWindowBuilder::new(
                    app,
                    "oobe",
                    tauri::WebviewUrl::App("oobe.html".into())
                )
                .title("欢迎使用 ViewStage")
                .inner_size(500.0, 520.0)
                .resizable(false)
                .decorations(false)
                .center()
                .always_on_top(true)
                .build()
                .expect("Failed to create OOBE window");
                
                let _ = oobe_window.set_focus();
                
                if let Some(splashscreen) = app.get_webview_window("splashscreen") {
                    let _ = splashscreen.close();
                }
            } else {
                if let Ok(config_content) = std::fs::read_to_string(&config_path) {
                    if let Ok(config) = serde_json::from_str::<serde_json::Value>(&config_content) {
                        if let (Some(width), Some(height)) = (
                            config.get("width").and_then(serde_json::Value::as_u64),
                            config.get("height").and_then(serde_json::Value::as_u64)
                        ) {
                            let _ = window.set_size(tauri::Size::Physical(tauri::PhysicalSize {
                                width: width as u32,
                                height: height as u32,
                            }));
                        }
                        
                        let _ = window.set_fullscreen(true);
                    }
                }
                
                let args: Vec<String> = std::env::args().collect();
                println!("启动参数: {:?}", args);
                
                if args.len() > 1 {
                    let file_path = args[1].clone();
                    println!("检测到文件参数: {}", file_path);
                    
                    let app_handle = app.handle().clone();
                    std::thread::spawn(move || {
                        std::thread::sleep(std::time::Duration::from_millis(2000));
                        println!("发送文件打开事件: {}", file_path);
                        let _ = app_handle.emit("file-opened", file_path.clone());
                        println!("已发送文件打开事件: {}", file_path);
                    });
                }
                
                println!("应用已启动，等待文件打开事件...");
                
                let app_handle = app.handle().clone();
                std::thread::spawn(move || {
                    std::thread::sleep(std::time::Duration::from_millis(1000));
                    if let Some(splashscreen) = app_handle.get_webview_window("splashscreen") {
                        let _ = splashscreen.close();
                    }
                    if let Some(main_window) = app_handle.get_webview_window("main") {
                        let _ = main_window.show();
                        let _ = main_window.set_focus();
                    }
                });
            }
            
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_cache_dir, 
            get_cache_size,
            clear_cache,
            check_auto_clear_cache,
            get_config_dir, 
            get_cds_dir,
            get_theme_dir,
            enhance_image, 
            generate_thumbnail, 
            rotate_image,
            save_image,
            save_image_with_enhance,
            compact_strokes,
            generate_thumbnails_batch,
            open_settings_window,
            open_doc_scan_window,
            rotate_main_image,
            set_mirror_state,
            get_mirror_state,
            set_enhance_state,
            get_enhance_state,
            switch_camera,
            get_app_version,
            check_update,
            get_settings,
            save_settings,
            reset_settings,
            restart_app,
            get_available_resolutions,
            check_pdf_default_app,
            close_splashscreen,
            complete_oobe,
            is_oobe_active,
            exit_app,
            detect_office,
            convert_docx_to_pdf,
            convert_docx_to_pdf_from_bytes,
            set_file_type_icons,
            scan_document,
            enhance_document_advanced,
            detect_text_dbnet,
            get_dbnet_model_path,
            check_dbnet_model_exists,
            get_dbnet_model_info,
            download_dbnet_model,
            delete_dbnet_model,
            check_uvdoc_model_exists,
            get_uvdoc_model_info,
            download_uvdoc_model,
            delete_uvdoc_model,
            check_enhance_model_exists,
            get_enhance_model_info,
            download_enhance_model,
            delete_enhance_model,
            unwarp_document,
            enhance_document,
            check_dexined_model,
            detect_edges_dexined,
            import_dexined_model,
            delete_dexined_model,
            detect_document_boundary_dexined,
            clear_model_cache
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
