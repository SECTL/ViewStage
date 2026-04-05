use image::{DynamicImage, ImageBuffer, Rgba, GenericImageView};
use rayon::prelude::*;
use base64::{Engine as _, engine::general_purpose};
use serde::{Deserialize, Serialize};

const MAX_IMAGE_SIZE: usize = 50 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThumbnailRequest {
    pub image_data: String,
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThumbnailResult {
    pub thumbnail: Option<String>,
    pub error: Option<String>,
}

pub fn decode_base64_image(image_data: &str) -> Result<DynamicImage, String> {
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

pub fn extract_base64(image_data: &str) -> Result<Vec<u8>, String> {
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

#[tauri::command]
pub fn enhance_image(image_data: String, contrast: f32, brightness: f32, saturation: f32, sharpen: f32) -> Result<String, String> {
    let img = decode_base64_image(&image_data)?;
    
    let enhanced = apply_enhance_filter(&img, contrast, brightness, saturation, sharpen);
    
    let mut buffer = Vec::new();
    enhanced
        .write_to(&mut std::io::Cursor::new(&mut buffer), image::ImageFormat::Png)
        .map_err(|e| format!("Failed to encode image: {}", e))?;
    
    let result = format!("data:image/png;base64,{}", general_purpose::STANDARD.encode(&buffer));
    
 Ok(result)
}



pub fn apply_enhance_filter(img: &DynamicImage, contrast: f32, brightness: f32, saturation: f32, sharpen: f32) -> DynamicImage {
    let (width, height) = (img.width(), img.height());
    
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
                    
                    let prev_row_start = ((y - 1) * width * 4) as usize;
                    let curr_row_start = (y * width * 4) as usize;
                    let next_row_start = ((y + 1) * width * 4) as usize;
                    let x_bytes = (x * 4) as usize;
                    
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

#[tauri::command]
pub fn generate_thumbnail(image_data: String, max_size: u32, fixed_ratio: bool) -> Result<String, String> {
    if max_size == 0 {
        return Err("max_size must be greater than 0".to_string());
    }
    
    let img = decode_base64_image(&image_data)?;
    generate_thumbnail_from_image(&img, max_size, fixed_ratio)
}

fn generate_thumbnail_from_image(img: &DynamicImage, max_size: u32, fixed_ratio: bool) -> Result<String, String> {
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

#[tauri::command]
pub fn generate_thumbnails_batch(images: Vec<ThumbnailRequest>, max_size: u32, fixed_ratio: bool) -> Result<Vec<ThumbnailResult>, String> {
    if max_size == 0 {
        return Err("max_size must be greater than 0".to_string());
    }
    
    let results: Vec<ThumbnailResult> = images
        .par_iter()
        .map(|req| {
            match decode_base64_image(&req.image_data) {
                Ok(img) => match generate_thumbnail_from_image(&img, max_size, fixed_ratio) {
                    Ok(thumbnail) => ThumbnailResult {
                        thumbnail: Some(thumbnail),
                        error: None,
                    },
                    Err(e) => ThumbnailResult {
                        thumbnail: None,
                        error: Some(e),
                    },
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

#[tauri::command]
pub fn rotate_image(image_data: String, direction: String) -> Result<String, String> {
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
