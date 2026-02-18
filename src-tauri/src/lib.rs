// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/

use tauri::{Manager, Emitter};
use image::{DynamicImage, ImageBuffer, Rgba, GenericImageView};
use base64::{Engine as _, engine::general_purpose};

#[tauri::command]
fn enhance_image(image_data: String) -> Result<String, String> {
    let base64_data = if image_data.starts_with("data:image") {
        image_data.split(',')
            .nth(1)
            .ok_or("Invalid base64 image data")?
            .to_string()
    } else {
        image_data
    };
    
    let decoded = general_purpose::STANDARD
        .decode(&base64_data)
        .map_err(|e| format!("Failed to decode base64: {}", e))?;
    
    let img = image::load_from_memory(&decoded)
        .map_err(|e| format!("Failed to load image: {}", e))?;
    
    let enhanced = apply_enhance_filter(&img);
    
    let mut buffer = Vec::new();
    enhanced
        .write_to(&mut std::io::Cursor::new(&mut buffer), image::ImageFormat::Png)
        .map_err(|e| format!("Failed to encode image: {}", e))?;
    
    let result = format!("data:image/png;base64,{}", general_purpose::STANDARD.encode(&buffer));
    
    Ok(result)
}

fn apply_enhance_filter(img: &DynamicImage) -> DynamicImage {
    let (width, height) = (img.width(), img.height());
    let mut enhanced_img: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(width, height);
    
    let contrast: f32 = 1.4;
    let brightness: f32 = 10.0;
    let saturation: f32 = 1.2;
    
    for (x, y, pixel) in enhanced_img.enumerate_pixels_mut() {
        let original_pixel = img.get_pixel(x, y);
        let r = original_pixel[0] as f32;
        let g = original_pixel[1] as f32;
        let b = original_pixel[2] as f32;
        let a = original_pixel[3];
        
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
        
        *pixel = Rgba([new_r as u8, new_g as u8, new_b as u8, a]);
    }
    
    DynamicImage::ImageRgba8(enhanced_img)
}

#[tauri::command]
fn generate_thumbnail(image_data: String, max_size: u32) -> Result<String, String> {
    let base64_data = if image_data.starts_with("data:image") {
        image_data.split(',')
            .nth(1)
            .ok_or("Invalid base64 image data")?
            .to_string()
    } else {
        image_data
    };
    
    let decoded = general_purpose::STANDARD
        .decode(&base64_data)
        .map_err(|e| format!("Failed to decode base64: {}", e))?;
    
    let img = image::load_from_memory(&decoded)
        .map_err(|e| format!("Failed to load image: {}", e))?;
    
    let (width, height) = (img.width(), img.height());
    
    let (thumb_w, thumb_h) = if width > height {
        (max_size, (height as f32 * max_size as f32 / width as f32) as u32)
    } else {
        ((width as f32 * max_size as f32 / height as f32) as u32, max_size)
    };
    
    let thumbnail = img.thumbnail(thumb_w, thumb_h);
    
    let mut buffer = Vec::new();
    thumbnail
        .write_to(&mut std::io::Cursor::new(&mut buffer), image::ImageFormat::Jpeg)
        .map_err(|e| format!("Failed to encode thumbnail: {}", e))?;
    
    let result = format!("data:image/jpeg;base64,{}", general_purpose::STANDARD.encode(&buffer));
    
    Ok(result)
}

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

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_fs::init())
        .setup(|app| {
            let window = app.get_webview_window("main").unwrap();
            
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
        .invoke_handler(tauri::generate_handler![greet, get_cache_dir, get_config_dir, get_cds_dir, enhance_image, generate_thumbnail])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
