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
use image::{DynamicImage, ImageBuffer, Rgba, GenericImageView, RgbaImage};
use base64::{Engine as _, engine::general_purpose};
use rayon::prelude::*;

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;

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
                    
                    let prev_row = ((y - 1) * width) as usize;
                    let curr_row = (y * width) as usize;
                    let next_row = ((y + 1) * width) as usize;
                    let x_offset = (x * 4) as usize;
                    
                    let neighbors_r: f32 = [
                        original_raw[prev_row + x_offset - 4],
                        original_raw[prev_row + x_offset],
                        original_raw[prev_row + x_offset + 4],
                        original_raw[curr_row + x_offset - 4],
                        original_raw[curr_row + x_offset + 4],
                        original_raw[next_row + x_offset - 4],
                        original_raw[next_row + x_offset],
                        original_raw[next_row + x_offset + 4],
                    ].iter().map(|&v| v as f32).sum();
                    
                    let neighbors_g: f32 = [
                        original_raw[prev_row + x_offset - 3],
                        original_raw[prev_row + x_offset + 1],
                        original_raw[prev_row + x_offset + 5],
                        original_raw[curr_row + x_offset - 3],
                        original_raw[curr_row + x_offset + 5],
                        original_raw[next_row + x_offset - 3],
                        original_raw[next_row + x_offset + 1],
                        original_raw[next_row + x_offset + 5],
                    ].iter().map(|&v| v as f32).sum();
                    
                    let neighbors_b: f32 = [
                        original_raw[prev_row + x_offset - 2],
                        original_raw[prev_row + x_offset + 2],
                        original_raw[prev_row + x_offset + 6],
                        original_raw[curr_row + x_offset - 2],
                        original_raw[curr_row + x_offset + 6],
                        original_raw[next_row + x_offset - 2],
                        original_raw[next_row + x_offset + 2],
                        original_raw[next_row + x_offset + 6],
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
        let _ = WriteLogger::init(LevelFilter::Info, Config::default(), file);
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
            enhance_image, 
            generate_thumbnail, 
            rotate_image,
            save_image,
            save_image_with_enhance,
            compact_strokes,
            generate_thumbnails_batch,
            open_settings_window,
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
            set_file_type_icons
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
