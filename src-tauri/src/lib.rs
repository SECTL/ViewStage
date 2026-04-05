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
use image::{DynamicImage, ImageBuffer, Rgba, GenericImageView, RgbaImage, GrayImage, Luma};
use imageproc::filter::gaussian_blur_f32;
use base64::{Engine as _, engine::general_purpose};
use rayon::prelude::*;

mod gpu;

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;

#[cfg(target_os = "windows")]
use opencv::{
    core::{Mat, Vector, Size, Scalar, CV_32F, CV_8UC3, Point, BORDER_CONSTANT},
    dnn::{read_net_from_tensorflow, Net, blob_from_image},
    imgproc::{
        resize, cvt_color, COLOR_BGR2GRAY,
        adaptive_threshold, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY,
        get_structuring_element, morphology_ex, MORPH_RECT, MORPH_CLOSE,
    },
    photo::fast_nl_means_denoising,
    prelude::*,
};

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

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

/// 缩略图请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThumbnailRequest {
    pub image_data: String,     // 原图数据
    pub name: Option<String>,   // 文件名
}

/// 缩略图生成结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThumbnailResult {
    pub thumbnail: Option<String>,  // 缩略图数据 (base64)，失败时为 None
    pub error: Option<String>,      // 错误信息
}

// ==================== 工具函数 ====================
// base64 解码、图像格式转换等辅助函数

const MAX_IMAGE_SIZE: usize = 50 * 1024 * 1024;

/// 解码 base64 图片
fn decode_base64_image(image_data: &str) -> Result<DynamicImage, String> {
    let base64_data = if image_data.starts_with("data:image") {
        image_data.split(',')
            .nth(1)
            .ok_or("Invalid base64 image data")?
            .to_string()
    } else {
        image_data.to_string()
    };
    
    if base64_data.len() > MAX_IMAGE_SIZE * 4 / 3 {
        return Err("Image data too large (max 50MB)".to_string());
    }
    
    let decoded = general_purpose::STANDARD
        .decode(&base64_data)
        .map_err(|e| format!("Failed to decode base64: {}", e))?;
    
    let img = image::load_from_memory(&decoded)
        .map_err(|e| format!("Failed to load image: {}", e))?;
    
    if img.width() == 0 || img.height() == 0 {
        return Err("Invalid image dimensions: width or height is zero".to_string());
    }
    
    Ok(img)
}

// ==================== 图像增强 ====================
// 对比度、亮度、饱和度调整，使用 rayon 并行处理

/// 图像增强命令 (对比度、亮度、饱和度、锐化调整)
#[tauri::command]
fn enhance_image(image_data: String, contrast: f32, brightness: f32, saturation: f32, sharpen: f32) -> Result<String, String> {
    let img = decode_base64_image(&image_data)?;
    
    let enhanced = apply_enhance_filter(&img, contrast, brightness, saturation, sharpen);
    
    let mut buffer = Vec::new();
    enhanced
        .write_to(&mut std::io::Cursor::new(&mut buffer), image::ImageFormat::Png)
        .map_err(|e| format!("Failed to encode image: {}", e))?;
    
    let result = format!("data:image/png;base64,{}", general_purpose::STANDARD.encode(&buffer));
    
    Ok(result)
}

/// 应用图像增强滤镜 (并行处理)
fn apply_enhance_filter(img: &DynamicImage, contrast: f32, brightness: f32, saturation: f32, sharpen: f32) -> DynamicImage {
    let (width, height) = (img.width(), img.height());
    
    let rgba_img = img.to_rgba8();
    
    // 第一步：对比度、亮度、饱和度调整
    let pixels: Vec<(u32, u32, Rgba<u8>)> = rgba_img
        .enumerate_pixels()
        .par_bridge()
        .map(|(x, y, pixel)| {
            let r = pixel[0] as f32;
            let g = pixel[1] as f32;
            let b = pixel[2] as f32;
            let a = pixel[3];
            
            let mut new_r = ((r - 128.0) * contrast) + 128.0 + brightness;
            let mut new_g = ((g - 128.0) * contrast) + 128.0 + brightness;
            let mut new_b = ((b - 128.0) * contrast) + 128.0 + brightness;
            
            let gray = 0.299 * new_r + 0.587 * new_g + 0.114 * new_b;
            new_r = gray + (new_r - gray) * saturation;
            new_g = gray + (new_g - gray) * saturation;
            new_b = gray + (new_b - gray) * saturation;
            
            new_r = new_r.clamp(0.0, 255.0);
            new_g = new_g.clamp(0.0, 255.0);
            new_b = new_b.clamp(0.0, 255.0);
            
            (x, y, Rgba([new_r as u8, new_g as u8, new_b as u8, a]))
        })
        .collect();
    
    let mut enhanced_img: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(width, height);
    for (x, y, pixel) in pixels {
        enhanced_img.put_pixel(x, y, pixel);
    }
    
    // 第二步：锐化处理 (USM 锐化) - 并行优化
    if sharpen > 0.0 && width > 2 && height > 2 {
        let original = enhanced_img.clone();
        let original_raw = original.as_raw();
        let sharpen_amount = sharpen / 100.0;
        
        let sharpened_pixels: Vec<(u32, u32, Rgba<u8>)> = (1..height - 1)
            .into_par_iter()
            .flat_map(|y| {
                (1..width - 1).into_par_iter().map(move |x| {
                    let idx = ((y * width + x) * 4) as usize;
                    
                    let r = original_raw[idx] as f32;
                    let g = original_raw[idx + 1] as f32;
                    let b = original_raw[idx + 2] as f32;
                    let a = original_raw[idx + 3];
                    
                    // 正确计算字节偏移
                    let prev_row_start = ((y - 1) * width * 4) as usize;
                    let curr_row_start = (y * width * 4) as usize;
                    let next_row_start = ((y + 1) * width * 4) as usize;
                    let x_bytes = (x * 4) as usize;
                    
                    // R 通道邻居 (偏移 0)
                    let neighbors_r: f32 = [
                        original_raw[prev_row_start + x_bytes - 4],
                        original_raw[prev_row_start + x_bytes],
                        original_raw[prev_row_start + x_bytes + 4],
                        original_raw[curr_row_start + x_bytes - 4],
                        original_raw[curr_row_start + x_bytes + 4],
                        original_raw[next_row_start + x_bytes - 4],
                        original_raw[next_row_start + x_bytes],
                        original_raw[next_row_start + x_bytes + 4],
                    ].iter().map(|&v| v as f32).sum();
                    
                    // G 通道邻居 (偏移 1)
                    let neighbors_g: f32 = [
                        original_raw[prev_row_start + x_bytes - 3],
                        original_raw[prev_row_start + x_bytes + 1],
                        original_raw[prev_row_start + x_bytes + 5],
                        original_raw[curr_row_start + x_bytes - 3],
                        original_raw[curr_row_start + x_bytes + 5],
                        original_raw[next_row_start + x_bytes - 3],
                        original_raw[next_row_start + x_bytes + 1],
                        original_raw[next_row_start + x_bytes + 5],
                    ].iter().map(|&v| v as f32).sum();
                    
                    // B 通道邻居 (偏移 2)
                    let neighbors_b: f32 = [
                        original_raw[prev_row_start + x_bytes - 2],
                        original_raw[prev_row_start + x_bytes + 2],
                        original_raw[prev_row_start + x_bytes + 6],
                        original_raw[curr_row_start + x_bytes - 2],
                        original_raw[curr_row_start + x_bytes + 6],
                        original_raw[next_row_start + x_bytes - 2],
                        original_raw[next_row_start + x_bytes + 2],
                        original_raw[next_row_start + x_bytes + 6],
                    ].iter().map(|&v| v as f32).sum();
                    
                    let laplacian_r = r * 9.0 - neighbors_r;
                    let laplacian_g = g * 9.0 - neighbors_g;
                    let laplacian_b = b * 9.0 - neighbors_b;
                    
                    let new_r = r + laplacian_r * sharpen_amount;
                    let new_g = g + laplacian_g * sharpen_amount;
                    let new_b = b + laplacian_b * sharpen_amount;
                    
                    (x, y, Rgba([
                        new_r.clamp(0.0, 255.0) as u8,
                        new_g.clamp(0.0, 255.0) as u8,
                        new_b.clamp(0.0, 255.0) as u8,
                        a
                    ]))
                })
            })
            .collect();
        
        for (x, y, pixel) in sharpened_pixels {
            enhanced_img.put_pixel(x, y, pixel);
        }
    }
    
    DynamicImage::ImageRgba8(enhanced_img)
}

// ==================== 缩略图生成 ====================
// 单张/批量生成缩略图，支持固定比例裁剪

/// 生成单张缩略图
/// @param image_data: 原图 base64
/// @param max_size: 最大边长
/// @param fixed_ratio: 是否固定 16:9 比例
#[tauri::command]
fn generate_thumbnail(image_data: String, max_size: u32, fixed_ratio: bool) -> Result<String, String> {
    let img = decode_base64_image(&image_data)?;
    
    if max_size == 0 {
        return Err("max_size must be greater than 0".to_string());
    }
    
    let (width, height) = (img.width(), img.height());
    
    let (thumb_w, thumb_h, scaled_w, scaled_h, offset_x, offset_y) = if fixed_ratio {
        let tw = max_size;
        let th = ((max_size as f32 * 9.0 / 16.0).max(1.0)) as u32;
        
        let img_ratio = width as f32 / height as f32;
        let canvas_ratio = 16.0 / 9.0;
        
        let (sw, sh) = if img_ratio > canvas_ratio {
            (tw, ((tw as f32 / img_ratio).max(1.0)) as u32)
        } else {
            (((th as f32 * img_ratio).max(1.0)) as u32, th)
        };
        
        let ox = (tw - sw) / 2;
        let oy = (th - sh) / 2;
        
        (tw, th, sw, sh, ox, oy)
    } else {
        let (tw, th) = if width > height {
            (max_size, ((height as f32 * max_size as f32 / width as f32).max(1.0)) as u32)
        } else {
            (((width as f32 * max_size as f32 / height as f32).max(1.0)) as u32, max_size)
        };
        
        (tw, th, tw, th, 0, 0)
    };
    
    let scaled_img = img.thumbnail(scaled_w, scaled_h);
    
    let mut canvas: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(thumb_w, thumb_h);
    
    for pixel in canvas.pixels_mut() {
        *pixel = Rgba([0, 0, 0, 255]);
    }
    
    for (x, y, pixel) in scaled_img.pixels() {
        let canvas_x = x + offset_x;
        let canvas_y = y + offset_y;
        if canvas_x < thumb_w && canvas_y < thumb_h {
            canvas.put_pixel(canvas_x, canvas_y, pixel);
        }
    }
    
    let mut buffer = Vec::new();
    DynamicImage::ImageRgba8(canvas)
        .write_to(&mut std::io::Cursor::new(&mut buffer), image::ImageFormat::Jpeg)
        .map_err(|e| format!("Failed to encode thumbnail: {}", e))?;
    
    let result = format!("data:image/jpeg;base64,{}", general_purpose::STANDARD.encode(&buffer));
    
    Ok(result)
}

// ==================== 图像旋转 ====================
// 90/180/270度旋转，用于摄像头和图片旋转

/// 旋转图像 (90度/270度)
/// @param image_data: 原图 base64
/// @param direction: "left" (270度) 或 "right" (90度)
#[tauri::command]
fn rotate_image(image_data: String, direction: String) -> Result<String, String> {
    let img = decode_base64_image(&image_data)?;
    
    let rotated = if direction == "left" {
        img.rotate270()
    } else {
        img.rotate90()
    };
    
    let mut buffer = Vec::new();
    rotated
        .write_to(&mut std::io::Cursor::new(&mut buffer), image::ImageFormat::Png)
        .map_err(|e| format!("Failed to encode rotated image: {}", e))?;
    
    let result = format!("data:image/png;base64,{}", general_purpose::STANDARD.encode(&buffer));
    
    Ok(result)
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

/// 获取用户主题目录 (%APPDATA%/com.viewstage.app/themes)
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

// ==================== 图片保存 ====================
// 保存图片到本地文件系统，支持批量保存和增强保存

/// 提取 base64 数据
fn extract_base64(image_data: &str) -> Result<Vec<u8>, String> {
    let base64_data = if image_data.starts_with("data:image") {
        image_data.split(',')
            .nth(1)
            .ok_or("Invalid base64 image data")?
    } else {
        image_data
    };
    
    general_purpose::STANDARD
        .decode(base64_data)
        .map_err(|e| format!("Failed to decode base64: {}", e))
}

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

// ==================== 批量缩略图 ====================
// 并行生成多张缩略图，使用 rayon 加速

#[tauri::command]
fn generate_thumbnails_batch(images: Vec<ThumbnailRequest>, max_size: u32, fixed_ratio: bool) -> Result<Vec<ThumbnailResult>, String> {
    if max_size == 0 {
        return Err("max_size must be greater than 0".to_string());
    }
    
    let results: Vec<ThumbnailResult> = images
        .par_iter()
        .map(|req| {
            match generate_thumbnail_internal(&req.image_data, max_size, fixed_ratio) {
                Ok(thumbnail) => ThumbnailResult {
                    thumbnail: Some(thumbnail),
                    error: None,
                },
                Err(e) => ThumbnailResult {
                    thumbnail: None,
                    error: Some(e),
                },
            }
        })
        .collect();
    
    Ok(results)
}

fn generate_thumbnail_internal(image_data: &str, max_size: u32, fixed_ratio: bool) -> Result<String, String> {
    let img = decode_base64_image(image_data)?;
    
    let (width, height) = (img.width(), img.height());
    
    let (thumb_w, thumb_h, scaled_w, scaled_h, offset_x, offset_y) = if fixed_ratio {
        let tw = max_size;
        let th = ((max_size as f32 * 9.0 / 16.0).max(1.0)) as u32;
        
        let img_ratio = width as f32 / height as f32;
        let canvas_ratio = 16.0 / 9.0;
        
        let (sw, sh) = if img_ratio > canvas_ratio {
            (tw, ((tw as f32 / img_ratio).max(1.0)) as u32)
        } else {
            (((th as f32 * img_ratio).max(1.0)) as u32, th)
        };
        
        let ox = (tw - sw) / 2;
        let oy = (th - sh) / 2;
        
        (tw, th, sw, sh, ox, oy)
    } else {
        let (tw, th) = if width > height {
            (max_size, ((height as f32 * max_size as f32 / width as f32).max(1.0)) as u32)
        } else {
            (((width as f32 * max_size as f32 / height as f32).max(1.0)) as u32, max_size)
        };
        
        (tw, th, tw, th, 0, 0)
    };
    
    let scaled_img = img.thumbnail(scaled_w, scaled_h);
    
    let mut canvas: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(thumb_w, thumb_h);
    
    for pixel in canvas.pixels_mut() {
        *pixel = Rgba([0, 0, 0, 255]);
    }
    
    for (x, y, pixel) in scaled_img.pixels() {
        let canvas_x = x + offset_x;
        let canvas_y = y + offset_y;
        if canvas_x < thumb_w && canvas_y < thumb_h {
            canvas.put_pixel(canvas_x, canvas_y, pixel);
        }
    }
    
    let mut buffer = Vec::new();
    DynamicImage::ImageRgba8(canvas)
        .write_to(&mut std::io::Cursor::new(&mut buffer), image::ImageFormat::Jpeg)
        .map_err(|e| format!("Failed to encode thumbnail: {}", e))?;
    
    Ok(format!("data:image/jpeg;base64,{}", general_purpose::STANDARD.encode(&buffer)))
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
    
    let temp_name = format!("temp_{}.docx", chrono::Local::now().format("%Y%m%d%H%M%S"));
    let temp_docx_path = cache_dir.join(&temp_name);
    
    {
        let mut file = fs::File::create(&temp_docx_path)
            .map_err(|e| format!("创建临时文件失败: {}", e))?;
        file.write_all(&file_data)
            .map_err(|e| format!("写入临时文件失败: {}", e))?;
        file.sync_all()
            .map_err(|e| format!("同步文件失败: {}", e))?;
    }
    
    let pdf_name = temp_name.replace(".docx", ".pdf");
    let pdf_path = cache_dir.join(&pdf_name);
    
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
                convert_with_libreoffice(&docx_path_str, &pdf_path_str, &cache_dir)
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
                convert_with_libreoffice(&docx_path_str, &pdf_path_str, &cache_dir)
            } else {
                r
            }
        }
        OfficeSoftware::LibreOffice => {
            convert_with_libreoffice(&docx_path_str, &pdf_path_str, &cache_dir)
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
fn convert_with_libreoffice(docx_path: &str, _pdf_path: &str, cache_dir: &std::path::PathBuf) -> Result<(), String> {
    use std::process::Command;
    let output_dir = cache_dir.to_str().unwrap().to_string();
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
    
    let pdf_name = docx_absolute.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("converted")
        .to_string() + ".pdf";
    let pdf_path = cache_dir.join(&pdf_name);
    
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
            let output_dir = cache_dir.to_str().unwrap().to_string();
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
    
    let app_id = "com.viewstage.app";
    
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
}

/// 文档扫描结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentScanResult {
    pub enhanced_image: String,
    pub confidence: f32,
    pub text_bbox: Option<(i32, i32, i32, i32)>,
}

/// EAST 文本检测请求
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EastDetectionRequest {
    pub image_data: String,
    pub model_path: Option<String>,
    pub min_confidence: f32,
}

/// EAST 文本检测结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EastDetectionResult {
    pub bbox: Option<(i32, i32, i32, i32)>,
    pub success: bool,
    pub error: Option<String>,
}

/// 获取 EAST 模型默认路径
#[tauri::command]
fn get_east_model_path(app: tauri::AppHandle) -> Result<String, String> {
    let resource_dir = app.path().resource_dir()
        .map_err(|e| format!("获取资源目录失败: {}", e))?;
    let model_path = resource_dir.join("weights").join("frozen_east_text_detection.pb");
    Ok(model_path.to_string_lossy().to_string())
}

/// EAST 文本检测命令
#[tauri::command]
fn detect_text_east(app: tauri::AppHandle, request: EastDetectionRequest) -> Result<EastDetectionResult, String> {
    let img = decode_base64_image(&request.image_data)?;
    
    let model_path = match request.model_path {
        Some(path) => path,
        None => {
            let resource_dir = app.path().resource_dir()
                .map_err(|e| format!("获取资源目录失败: {}", e))?;
            resource_dir.join("weights").join("frozen_east_text_detection.pb")
                .to_string_lossy().to_string()
        }
    };
    
    match detect_text_regions_east(&img, &model_path, request.min_confidence) {
        Ok(bbox) => Ok(EastDetectionResult {
            bbox,
            success: true,
            error: None,
        }),
        Err(e) => Ok(EastDetectionResult {
            bbox: None,
            success: false,
            error: Some(e),
        }),
    }
}

/// 文档扫描 - EAST 文本检测
#[tauri::command]
fn scan_document(app: tauri::AppHandle, request: DocumentScanRequest) -> Result<DocumentScanResult, String> {
    let mut img = decode_base64_image(&request.image_data)?;
    
    log::info!("开始文档扫描，图像尺寸: {}x{}", img.width(), img.height());
    
    let model_path = match request.east_model_path {
        Some(ref path) => path.clone(),
        None => {
            let resource_dir = app.path().resource_dir()
                .map_err(|e| format!("获取资源目录失败: {}", e))?;
            log::info!("资源目录: {:?}", resource_dir);
            resource_dir.join("weights").join("frozen_east_text_detection.pb")
                .to_string_lossy().to_string()
        }
    };
    
    log::info!("模型路径: {}", model_path);
    
    let text_bbox = match detect_text_regions_east(&img, &model_path, 0.5) {
        Ok(bbox) => {
            log::info!("检测结果: {:?}", bbox);
            bbox
        }
        Err(e) => {
            log::error!("EAST 检测失败: {}", e);
            None
        }
    };
    
    let result_img = if let Some((x1, y1, x2, y2)) = text_bbox {
        log::info!("裁剪区域: ({}, {}) - ({}, {})", x1, y1, x2, y2);
        let (width, height) = (img.width() as i32, img.height() as i32);
        let x1 = x1.max(0).min(width - 1) as u32;
        let y1 = y1.max(0).min(height - 1) as u32;
        let x2 = x2.max(0).min(width) as u32;
        let y2 = y2.max(0).min(height) as u32;
        
        if x2 > x1 && y2 > y1 {
            log::info!("执行裁剪: ({}, {}) - ({}, {})", x1, y1, x2, y2);
            img.crop(x1, y1, x2 - x1, y2 - y1)
        } else {
            log::warn!("裁剪区域无效，返回原图");
            img
        }
    } else {
        log::warn!("未检测到文本区域，返回原图");
        img
    };

    let enhanced_img = enhance_document_opencv(&result_img)?;
    
    let mut buffer = Vec::new();
    enhanced_img
        .write_to(&mut std::io::Cursor::new(&mut buffer), image::ImageFormat::Png)
        .map_err(|e| format!("Failed to encode image: {}", e))?;
    
    let result_image = format!("data:image/png;base64,{}", general_purpose::STANDARD.encode(&buffer));
    
    Ok(DocumentScanResult {
        enhanced_image: result_image,
        confidence: if text_bbox.is_some() { 0.9 } else { 0.0 },
        text_bbox,
    })
}

// ==================== OpenCV 文档增强 ====================

#[cfg(target_os = "windows")]
fn enhance_document_opencv(img: &DynamicImage) -> Result<DynamicImage, String> {
    let rgba = img.to_rgba8();
    let (width, height) = rgba.dimensions();
    
    let mut bgr_mat = unsafe { Mat::new_rows_cols(height as i32, width as i32, CV_8UC3) }
        .map_err(|e| format!("创建 Mat 失败: {}", e))?;
    {
        let data = bgr_mat.data_bytes_mut()
            .map_err(|e| format!("获取 Mat 数据失败: {}", e))?;
        for y in 0..height {
            for x in 0..width {
                let pixel = rgba.get_pixel(x, y);
                let idx = (y * width as u32 + x) as usize * 3;
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
    fast_nl_means_denoising(&gray, &mut denoised, 1.0, 7, 21)
        .map_err(|e| format!("降噪失败: {}", e))?;
    
    let result_data = denoised.data_bytes()
        .map_err(|e| format!("获取结果数据失败: {}", e))?;
    
    let mut result_img = ImageBuffer::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width as u32 + x) as usize;
            let val = result_data[idx];
            result_img.put_pixel(x, y, Luma([val]));
        }
    }
    
    Ok(DynamicImage::ImageLuma8(result_img))
}

#[cfg(not(target_os = "windows"))]
fn enhance_document_opencv(img: &DynamicImage) -> Result<DynamicImage, String> {
    Ok(img.clone())
}

// ==================== EAST 文本检测 ====================
// 使用 OpenCV DNN 模块实现 EAST 文本检测

#[cfg(target_os = "windows")]
static EAST_NET: std::sync::OnceLock<std::sync::Mutex<Option<Net>>> = std::sync::OnceLock::new();

#[cfg(target_os = "windows")]
fn get_east_net(model_path: &str) -> Result<std::sync::MutexGuard<'static, Option<Net>>, String> {
    let net_guard = EAST_NET.get_or_init(|| {
        match read_net_from_tensorflow(model_path, "") {
            Ok(net) => {
                log::info!("EAST 模型加载成功: {}", model_path);
                std::sync::Mutex::new(Some(net))
            }
            Err(e) => {
                log::error!("EAST 模型加载失败: {}", e);
                std::sync::Mutex::new(None)
            }
        }
    });
    
    net_guard.lock().map_err(|e| format!("获取模型锁失败: {}", e))
}

#[cfg(target_os = "windows")]
fn detect_text_regions_east(img: &DynamicImage, model_path: &str, min_confidence: f32) -> Result<Option<(i32, i32, i32, i32)>, String> {
    let mut net_guard = get_east_net(model_path)?;
    let net = match net_guard.as_mut() {
        Some(n) => n,
        None => return Err("EAST 模型未加载".to_string()),
    };
    
    let (orig_width, orig_height) = (img.width() as i32, img.height() as i32);
    
    let new_width = 320i32;
    let new_height = 320i32;
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
                let idx = (y * width as u32 + x) as usize * 3;
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
        Scalar::new(103.94, 116.78, 123.68, 0.0),
        true,
        false,
        CV_32F,
    ).map_err(|e| format!("创建 blob 失败: {}", e))?;
    
    net.set_input(&blob, "", 1.0, Scalar::default())
        .map_err(|e| format!("设置输入失败: {}", e))?;
    
    let mut output_names = Vector::<String>::new();
    output_names.push("feature_fusion/Conv_7/Sigmoid");
    output_names.push("feature_fusion/concat_3");
    
    let mut outputs = Vector::<Mat>::new();
    net.forward(&mut outputs, &output_names)
        .map_err(|e| format!("前向传播失败: {}", e))?;
    
    let scores_raw = outputs.get(0).map_err(|e| format!("获取 scores 失败: {}", e))?;
    let geometry_raw = outputs.get(1).map_err(|e| format!("获取 geometry 失败: {}", e))?;
    
    let scores_dims = scores_raw.mat_size();
    let geometry_dims = geometry_raw.mat_size();
    log::info!("scores dims: {:?}", scores_dims);
    log::info!("geometry dims: {:?}", geometry_dims);
    
    let num_rows = if scores_dims.len() >= 4 {
        scores_dims[2] as i32
    } else {
        scores_raw.rows()
    };
    let num_cols = if scores_dims.len() >= 4 {
        scores_dims[3] as i32
    } else {
        scores_raw.cols()
    };
    
    log::info!("numRows: {}, numCols: {}", num_rows, num_cols);
    
    let mut scores = Mat::default();
    scores_raw.reshape(1, num_rows * num_cols)
        .map_err(|e| format!("reshape scores 失败: {}", e))?
        .copy_to(&mut scores)
        .map_err(|e| format!("copy scores 失败: {}", e))?;
    
    let mut geometry = Mat::default();
    geometry_raw.reshape(1, num_rows * num_cols)
        .map_err(|e| format!("reshape geometry 失败: {}", e))?
        .copy_to(&mut geometry)
        .map_err(|e| format!("copy geometry 失败: {}", e))?;
    
    let mut rects: Vec<(f32, f32, f32, f32)> = Vec::new();
    let mut confidences: Vec<f32> = Vec::new();
    
    let scores_data = scores.data_bytes().map_err(|e| format!("获取 scores 数据失败: {}", e))?;
    let geometry_data = geometry.data_bytes().map_err(|e| format!("获取 geometry 数据失败: {}", e))?;
    
    for y in 0..num_rows {
        for x in 0..num_cols {
            let idx = (y * num_cols + x) as usize;
            let score = f32::from_le_bytes([
                scores_data[idx * 4],
                scores_data[idx * 4 + 1],
                scores_data[idx * 4 + 2],
                scores_data[idx * 4 + 3],
            ]);
            
            if score < min_confidence {
                continue;
            }
            
            let offset_x = x as f32 * 4.0;
            let offset_y = y as f32 * 4.0;
            
            let geo_base = idx * 20;
            let x0 = f32::from_le_bytes([
                geometry_data[geo_base],
                geometry_data[geo_base + 1],
                geometry_data[geo_base + 2],
                geometry_data[geo_base + 3],
            ]);
            let x1 = f32::from_le_bytes([
                geometry_data[geo_base + 4],
                geometry_data[geo_base + 5],
                geometry_data[geo_base + 6],
                geometry_data[geo_base + 7],
            ]);
            let x2 = f32::from_le_bytes([
                geometry_data[geo_base + 8],
                geometry_data[geo_base + 9],
                geometry_data[geo_base + 10],
                geometry_data[geo_base + 11],
            ]);
            let x3 = f32::from_le_bytes([
                geometry_data[geo_base + 12],
                geometry_data[geo_base + 13],
                geometry_data[geo_base + 14],
                geometry_data[geo_base + 15],
            ]);
            let angle = f32::from_le_bytes([
                geometry_data[geo_base + 16],
                geometry_data[geo_base + 17],
                geometry_data[geo_base + 18],
                geometry_data[geo_base + 19],
            ]);
            
            let cos = angle.cos();
            let sin = angle.sin();
            
            let h = x0 + x2;
            let w = x1 + x3;
            
            let end_x = offset_x + cos * x1 + sin * x2;
            let end_y = offset_y - sin * x1 + cos * x2;
            let start_x = end_x - w;
            let start_y = end_y - h;
            
            rects.push((start_x, start_y, end_x, end_y));
            confidences.push(score);
        }
    }
    
    log::info!("检测到 {} 个文本区域", rects.len());
    
    if rects.is_empty() {
        return Ok(None);
    }
    
    let boxes_indices = non_max_suppression(&rects, &confidences, 0.3);
    
    let mut points: Vec<(i32, i32, i32, i32)> = Vec::new();
    for idx in boxes_indices {
        let (sx, sy, ex, ey) = rects[idx];
        let start_x = (sx * rw) as i32;
        let start_y = (sy * rh) as i32;
        let end_x = (ex * rw) as i32;
        let end_y = (ey * rh) as i32;
        points.push((start_x, start_y, end_x, end_y));
    }
    
    if points.is_empty() {
        return Ok(None);
    }
    
    let min_x = points.iter().map(|p| p.0).min().unwrap_or(0);
    let min_y = points.iter().map(|p| p.1).min().unwrap_or(0);
    let max_x = points.iter().map(|p| p.2).max().unwrap_or(orig_width);
    let max_y = points.iter().map(|p| p.3).max().unwrap_or(orig_height);
    
    let margin = 20;
    let x1 = (min_x - margin).max(0);
    let y1 = (min_y - margin).max(0);
    let x2 = (max_x + margin).min(orig_width);
    let y2 = (max_y + margin).min(orig_height);
    
    let bbox = (x1, y1, x2, y2);
    
    log::info!("最终边界框: {:?}", bbox);
    
    Ok(Some(bbox))
}

#[cfg(target_os = "windows")]
fn non_max_suppression(rects: &[(f32, f32, f32, f32)], confidences: &[f32], overlap_threshold: f32) -> Vec<usize> {
    let n = rects.len();
    if n == 0 {
        return Vec::new();
    }
    
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| confidences[b].partial_cmp(&confidences[a]).unwrap());
    
    let mut result = Vec::new();
    let mut suppressed = vec![false; n];
    
    for i in 0..n {
        if suppressed[indices[i]] {
            continue;
        }
        
        result.push(indices[i]);
        
        for j in (i + 1)..n {
            if suppressed[indices[j]] {
                continue;
            }
            
            let idx_i = indices[i];
            let idx_j = indices[j];
            
            let iou = compute_iou(&rects[idx_i], &rects[idx_j]);
            if iou > overlap_threshold {
                suppressed[indices[j]] = true;
            }
        }
    }
    
    result
}

#[cfg(target_os = "windows")]
fn compute_iou(a: &(f32, f32, f32, f32), b: &(f32, f32, f32, f32)) -> f32 {
    let x1 = a.0.max(b.0);
    let y1 = a.1.max(b.1);
    let x2 = a.2.min(b.2);
    let y2 = a.3.min(b.3);
    
    let intersection = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    
    let area_a = (a.2 - a.0) * (a.3 - a.1);
    let area_b = (b.2 - b.0) * (b.3 - b.1);
    
    let union = area_a + area_b - intersection;
    
    if union > 0.0 { intersection / union } else { 0.0 }
}

#[cfg(not(target_os = "windows"))]
fn detect_text_regions_east(_img: &DynamicImage, _model_path: &str, _min_confidence: f32) -> Result<Option<(i32, i32, i32, i32)>, String> {
    Err("EAST 文本检测仅支持 Windows 系统".to_string())
}

// ==================== 高级文档增强算法 ====================
// 使用 imageproc 库实现

/// 光照归一化 - 去除不均匀光照和阴影（保留彩色）
/// 使用 imageproc 的 gaussian_blur_f32
fn normalize_illumination(img: &DynamicImage, sigma: f32) -> DynamicImage {
    let rgba = img.to_rgba8();
    let (width, height) = rgba.dimensions();
    
    let r_channel: GrayImage = ImageBuffer::from_fn(width, height, |x, y| {
        Luma([rgba.get_pixel(x, y)[0]])
    });
    let g_channel: GrayImage = ImageBuffer::from_fn(width, height, |x, y| {
        Luma([rgba.get_pixel(x, y)[1]])
    });
    let b_channel: GrayImage = ImageBuffer::from_fn(width, height, |x, y| {
        Luma([rgba.get_pixel(x, y)[2]])
    });
    
    let r_blurred = gaussian_blur_f32(&r_channel, sigma);
    let g_blurred = gaussian_blur_f32(&g_channel, sigma);
    let b_blurred = gaussian_blur_f32(&b_channel, sigma);
    
    let mut result = ImageBuffer::new(width, height);
    
    let mean_bg = 128.0f32;
    
    for y in 0..height {
        for x in 0..width {
            let r_orig = rgba.get_pixel(x, y)[0] as f32;
            let g_orig = rgba.get_pixel(x, y)[1] as f32;
            let b_orig = rgba.get_pixel(x, y)[2] as f32;
            let a = rgba.get_pixel(x, y)[3];
            
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
    
    let block_size = block_size.max(3).min(99) | 1;
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
            
            let pixel_val = gray.get_pixel(x, y)[0] as f64;
            let threshold = mean - c as f64;
            
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    use simplelog::*;
    use std::fs::File;
    
    let config_dir = dirs::config_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("com.viewstage.app");
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
            let window = app.get_webview_window("main").unwrap();
            
            // 初始化 GPU 上下文
            std::thread::spawn(|| {
                match pollster::block_on(gpu::GpuContext::init()) {
                    Ok(_) => log::info!("GPU 上下文初始化成功"),
                    Err(e) => log::warn!("GPU 上下文初始化失败: {}", e),
                }
            });
            
            let _ = window.set_decorations(false);
            
            let config_dir = app.path().app_config_dir().unwrap();
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
                            config.get("width").and_then(|v| v.as_u64()),
                            config.get("height").and_then(|v| v.as_u64())
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
            detect_text_east,
            get_east_model_path
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
