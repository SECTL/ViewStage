//! ViewStage - 图像处理 Rust 后端
//! 
//! 功能模块：
//! - 图像增强 (enhance_image)
//! - 缩略图生成 (generate_thumbnail, generate_thumbnails_batch)
//! - 图像旋转 (rotate_image)
//! - 图片保存 (save_image, save_images_batch)
//! - 笔画压缩 (compact_strokes)

use tauri::{Manager, Emitter};
use image::{DynamicImage, ImageBuffer, Rgba, GenericImageView, RgbaImage};
use base64::{Engine as _, engine::general_purpose};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// ==================== 数据结构 ====================

/// 图片保存结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageSaveResult {
    pub path: String,                    // 保存路径
    pub success: bool,                   // 是否成功
    pub error: Option<String>,           // 错误信息
    pub enhanced_data: Option<String>,   // 增强后的图片数据 (base64)
}

/// 图片数据 (用于批量操作)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageData {
    pub data: String,           // base64 图片数据
    pub name: Option<String>,   // 文件名前缀
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

// ==================== 工具函数 ====================

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
    
    let decoded = general_purpose::STANDARD
        .decode(&base64_data)
        .map_err(|e| format!("Failed to decode base64: {}", e))?;
    
    image::load_from_memory(&decoded)
        .map_err(|e| format!("Failed to load image: {}", e))
}

// ==================== 图像增强 ====================

/// 图像增强命令 (对比度、亮度、饱和度调整)
#[tauri::command]
fn enhance_image(image_data: String) -> Result<String, String> {
    let img = decode_base64_image(&image_data)?;
    
    let enhanced = apply_enhance_filter(&img);
    
    let mut buffer = Vec::new();
    enhanced
        .write_to(&mut std::io::Cursor::new(&mut buffer), image::ImageFormat::Png)
        .map_err(|e| format!("Failed to encode image: {}", e))?;
    
    let result = format!("data:image/png;base64,{}", general_purpose::STANDARD.encode(&buffer));
    
    Ok(result)
}

/// 应用图像增强滤镜 (并行处理)
/// - 对比度: 1.4x
/// - 亮度: +10
/// - 饱和度: 1.2x
fn apply_enhance_filter(img: &DynamicImage) -> DynamicImage {
    let (width, height) = (img.width(), img.height());
    let contrast: f32 = 1.4;
    let brightness: f32 = 10.0;
    let saturation: f32 = 1.2;
    
    let rgba_img = img.to_rgba8();
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
    
    DynamicImage::ImageRgba8(enhanced_img)
}

// ==================== 缩略图生成 ====================

/// 生成单张缩略图
/// @param image_data: 原图 base64
/// @param max_size: 最大边长
/// @param fixed_ratio: 是否固定 16:9 比例
#[tauri::command]
fn generate_thumbnail(image_data: String, max_size: u32, fixed_ratio: bool) -> Result<String, String> {
    let img = decode_base64_image(&image_data)?;
    
    let (width, height) = (img.width(), img.height());
    
    let (thumb_w, thumb_h, scaled_w, scaled_h, offset_x, offset_y) = if fixed_ratio {
        let tw = max_size;
        let th = (max_size as f32 * 9.0 / 16.0) as u32;
        
        let mut canvas: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(tw, th);
        
        for pixel in canvas.pixels_mut() {
            *pixel = Rgba([0, 0, 0, 255]);
        }
        
        let img_ratio = width as f32 / height as f32;
        let canvas_ratio = 16.0 / 9.0;
        
        let (sw, sh) = if img_ratio > canvas_ratio {
            (tw, (tw as f32 / img_ratio) as u32)
        } else {
            ((th as f32 * img_ratio) as u32, th)
        };
        
        let ox = (tw - sw) / 2;
        let oy = (th - sh) / 2;
        
        (tw, th, sw, sh, ox, oy)
    } else {
        let (tw, th) = if width > height {
            (max_size, (height as f32 * max_size as f32 / width as f32) as u32)
        } else {
            ((width as f32 * max_size as f32 / height as f32) as u32, max_size)
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

/// 获取应用缓存目录
#[tauri::command]
fn get_cache_dir(app: tauri::AppHandle) -> Result<String, String> {
    let cache_dir = app.path().app_cache_dir()
        .map_err(|e| format!("Failed to get cache dir: {}", e))?;
    
    if !cache_dir.exists() {
        std::fs::create_dir_all(&cache_dir)
            .map_err(|e| format!("Failed to create cache dir: {}", e))?;
    }
    
    Ok(cache_dir.to_string_lossy().to_string())
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

#[tauri::command]
fn save_image(image_data: String, prefix: Option<String>) -> Result<ImageSaveResult, String> {
    let base_dir = get_cds_dir()?;
    let prefix_str = prefix.unwrap_or_else(|| "photo".to_string());
    
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
fn save_image_with_enhance(image_data: String, prefix: Option<String>) -> Result<ImageSaveResult, String> {
    let base_dir = get_cds_dir()?;
    let prefix_str = prefix.unwrap_or_else(|| "photo".to_string());
    
    let img = decode_base64_image(&image_data)?;
    
    let enhanced = apply_enhance_filter(&img);
    
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

#[tauri::command]
fn save_images_batch(images: Vec<ImageData>, prefix: Option<String>) -> Result<Vec<ImageSaveResult>, String> {
    let base_dir = get_cds_dir()?;
    let prefix_str = prefix.unwrap_or_else(|| "photo".to_string());
    
    let results: Vec<ImageSaveResult> = images
        .par_iter()
        .map(|img_data| {
            let decoded = match extract_base64(&img_data.data) {
                Ok(d) => d,
                Err(e) => {
                    return ImageSaveResult {
                        path: String::new(),
                        success: false,
                        error: Some(e),
                        enhanced_data: None,
                    };
                }
            };
            
            let extension = if img_data.data.contains("image/png") {
                "png"
            } else if img_data.data.contains("image/jpeg") || img_data.data.contains("image/jpg") {
                "jpg"
            } else {
                "png"
            };
            
            let name_prefix = img_data.name.clone().unwrap_or_else(|| prefix_str.clone());
            
            let (file_path, _file_name) = match get_save_path(&base_dir, &name_prefix, extension) {
                Ok(p) => p,
                Err(e) => {
                    return ImageSaveResult {
                        path: String::new(),
                        success: false,
                        error: Some(e),
                        enhanced_data: None,
                    };
                }
            };
            
            match std::fs::write(&file_path, &decoded) {
                Ok(_) => ImageSaveResult {
                    path: file_path.to_string_lossy().to_string(),
                    success: true,
                    error: None,
                    enhanced_data: None,
                },
                Err(e) => ImageSaveResult {
                    path: String::new(),
                    success: false,
                    error: Some(format!("Failed to write file: {}", e)),
                    enhanced_data: None,
                },
            }
        })
        .collect();
    
    Ok(results)
}

#[tauri::command]
fn save_images_batch_with_enhance(images: Vec<ImageData>, prefix: Option<String>) -> Result<Vec<ImageSaveResult>, String> {
    let base_dir = get_cds_dir()?;
    let prefix_str = prefix.unwrap_or_else(|| "photo".to_string());
    
    let results: Vec<ImageSaveResult> = images
        .par_iter()
        .map(|img_data| {
            let img = match decode_base64_image(&img_data.data) {
                Ok(i) => i,
                Err(e) => {
                    return ImageSaveResult {
                        path: String::new(),
                        success: false,
                        error: Some(e),
                        enhanced_data: None,
                    };
                }
            };
            
            let enhanced = apply_enhance_filter(&img);
            
            let mut buffer = Vec::new();
            if let Err(e) = enhanced.write_to(&mut std::io::Cursor::new(&mut buffer), image::ImageFormat::Png) {
                return ImageSaveResult {
                    path: String::new(),
                    success: false,
                    error: Some(format!("Failed to encode image: {}", e)),
                    enhanced_data: None,
                };
            }
            
            let name_prefix = img_data.name.clone().unwrap_or_else(|| prefix_str.clone());
            
            let (file_path, _file_name) = match get_save_path(&base_dir, &name_prefix, "png") {
                Ok(p) => p,
                Err(e) => {
                    return ImageSaveResult {
                        path: String::new(),
                        success: false,
                        error: Some(e),
                        enhanced_data: None,
                    };
                }
            };
            
            let enhanced_data = format!("data:image/png;base64,{}", general_purpose::STANDARD.encode(&buffer));
            
            match std::fs::write(&file_path, &buffer) {
                Ok(_) => ImageSaveResult {
                    path: file_path.to_string_lossy().to_string(),
                    success: true,
                    error: None,
                    enhanced_data: Some(enhanced_data),
                },
                Err(e) => ImageSaveResult {
                    path: String::new(),
                    success: false,
                    error: Some(format!("Failed to write file: {}", e)),
                    enhanced_data: None,
                },
            }
        })
        .collect();
    
    Ok(results)
}

fn parse_color(color_str: &str) -> Rgba<u8> {
    if color_str.starts_with('#') && color_str.len() == 7 {
        let r = u8::from_str_radix(&color_str[1..3], 16).unwrap_or(52);
        let g = u8::from_str_radix(&color_str[3..5], 16).unwrap_or(152);
        let b = u8::from_str_radix(&color_str[5..7], 16).unwrap_or(219);
        Rgba([r, g, b, 255])
    } else {
        Rgba([52, 152, 219, 255])
    }
}

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
            let color = parse_color(stroke.color.as_deref().unwrap_or("#3498db"));
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

#[tauri::command]
fn generate_thumbnails_batch(images: Vec<ThumbnailRequest>, max_size: u32, fixed_ratio: bool) -> Result<Vec<String>, String> {
    let results: Vec<String> = images
        .par_iter()
        .map(|req| {
            match generate_thumbnail_internal(&req.image_data, max_size, fixed_ratio) {
                Ok(thumbnail) => thumbnail,
                Err(e) => {
                    eprintln!("Failed to generate thumbnail: {}", e);
                    String::new()
                }
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
        let th = (max_size as f32 * 9.0 / 16.0) as u32;
        
        let img_ratio = width as f32 / height as f32;
        let canvas_ratio = 16.0 / 9.0;
        
        let (sw, sh) = if img_ratio > canvas_ratio {
            (tw, (tw as f32 / img_ratio) as u32)
        } else {
            ((th as f32 * img_ratio) as u32, th)
        };
        
        let ox = (tw - sw) / 2;
        let oy = (th - sh) / 2;
        
        (tw, th, sw, sh, ox, oy)
    } else {
        let (tw, th) = if width > height {
            (max_size, (height as f32 * max_size as f32 / width as f32) as u32)
        } else {
            ((width as f32 * max_size as f32 / height as f32) as u32, max_size)
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

use std::sync::atomic::{AtomicBool, Ordering};

static MIRROR_STATE: AtomicBool = AtomicBool::new(false);
static ENHANCE_STATE: AtomicBool = AtomicBool::new(false);

#[tauri::command]
async fn open_settings_window(app: tauri::AppHandle) -> Result<(), String> {
    use tauri::WebviewWindowBuilder;
    
    let existing = app.get_webview_window("settings");
    if existing.is_some() {
        if let Some(window) = existing {
            let _ = window.set_focus();
        }
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
    
    let _ = window.set_focus();
    
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

#[tauri::command]
async fn check_update() -> Result<GitHubRelease, String> {
    let client = reqwest::Client::builder()
        .user_agent("ViewStage")
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|e| e.to_string())?;
    
    let response = client
        .get("https://api.github.com/repos/ospneam/ViewStage/releases/latest")
        .send()
        .await
        .map_err(|e| e.to_string())?;
    
    if !response.status().is_success() {
        return Err(format!("请求失败: {}", response.status()));
    }
    
    let release: GitHubRelease = response
        .json()
        .await
        .map_err(|e| e.to_string())?;
    
    Ok(release)
}

#[tauri::command]
async fn get_settings(app: tauri::AppHandle) -> Result<serde_json::Value, String> {
    let config_dir = app.path().app_config_dir().map_err(|e| e.to_string())?;
    let config_path = config_dir.join("config.json");
    
    if config_path.exists() {
        if let Ok(config_content) = std::fs::read_to_string(&config_path) {
            if let Ok(config) = serde_json::from_str::<serde_json::Value>(&config_content) {
                return Ok(config);
            }
        }
    }
    
    if !config_dir.exists() {
        std::fs::create_dir_all(&config_dir).map_err(|e| e.to_string())?;
    }
    
    let default_config = serde_json::json!({
        "width": 1920,
        "height": 1080,
        "language": "zh-CN",
        "defaultCamera": "",
        "cameraWidth": 1280,
        "cameraHeight": 720,
        "moveFps": 30,
        "drawFps": 10,
        "fileAssociations": []
    });
    
    let config_str = serde_json::to_string_pretty(&default_config).map_err(|e| e.to_string())?;
    std::fs::write(&config_path, config_str).map_err(|e| e.to_string())?;
    
    Ok(default_config)
}

#[tauri::command]
async fn save_settings(app: tauri::AppHandle, settings: serde_json::Value) -> Result<(), String> {
    let config_dir = app.path().app_config_dir().map_err(|e| e.to_string())?;
    
    if !config_dir.exists() {
        std::fs::create_dir_all(&config_dir).map_err(|e| e.to_string())?;
    }
    
    let config_path = config_dir.join("config.json");
    
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
    std::fs::write(&config_path, &config_str).map_err(|e| e.to_string())?;
    
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
    let config_path = config_dir.join("config.json");
    
    if config_path.exists() {
        std::fs::remove_file(&config_path).map_err(|e| e.to_string())?;
        
        if config_path.exists() {
            return Err("配置文件删除失败".to_string());
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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_fs::init())
        .setup(|app| {
            let window = app.get_webview_window("main").unwrap();
            
            let _ = window.set_decorations(false);
            
            let config_dir = app.path().app_config_dir().unwrap();
            let config_path = config_dir.join("config.json");
            
            if config_path.exists() {
                if let Ok(config_content) = std::fs::read_to_string(config_path) {
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
            }
            
            let args: Vec<String> = std::env::args().collect();
            println!("启动参数: {:?}", args);
            
            if args.len() > 1 {
                let file_path = args[1].clone();
                println!("检测到文件参数: {}", file_path);
                
                let app_handle = app.handle().clone();
                std::thread::spawn(move || {
                    std::thread::sleep(std::time::Duration::from_millis(500));
                    let _ = app_handle.emit("file-opened", file_path.clone());
                    println!("已发送文件打开事件: {}", file_path);
                });
            }
            
            println!("应用已启动，等待文件打开事件...");
            
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_cache_dir, 
            get_config_dir, 
            get_cds_dir, 
            enhance_image, 
            generate_thumbnail, 
            rotate_image,
            save_image,
            save_image_with_enhance,
            save_images_batch,
            save_images_batch_with_enhance,
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
            check_pdf_default_app
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
