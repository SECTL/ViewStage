use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};
use base64::{Engine as _, engine::general_purpose};

/// 线段点结构
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StrokePoint {
    pub from_x: f32,
    pub from_y: f32,
    pub to_x: f32,
    pub to_y: f32,
}

/// 笔画结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stroke {
    pub r#type: String,
    pub points: Vec<StrokePoint>,
    pub color: Option<String>,
    pub line_width: Option<u32>,
    pub eraser_size: Option<u32>,
}

/// 点处理配置
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PointOptimizationConfig {
    pub epsilon: f32,
    pub min_distance: f32,
    pub quantization: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProcessPointsRequest {
    pub points: Vec<StrokePoint>,
    pub config: PointOptimizationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchProcessRequest {
    pub strokes: Vec<Stroke>,
    pub config: PointOptimizationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DrawCommand {
    pub r#type: String,
    pub from_x: f32,
    pub from_y: f32,
    pub to_x: f32,
    pub to_y: f32,
    pub color: String,
    pub line_width: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchDrawRequest {
    pub commands: Vec<DrawCommand>,
    pub canvas_width: u32,
    pub canvas_height: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageFilterRequest {
    pub image_data: String,
    pub brightness: f32,
    pub contrast: f32,
    pub saturation: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchImageFilterRequest {
    pub images: Vec<String>,
    pub brightness: f32,
    pub contrast: f32,
    pub saturation: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformRequest {
    pub points: Vec<StrokePoint>,
    pub matrix: [f32; 9],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollisionRequest {
    pub stroke: Stroke,
    pub rect: [f32; 4],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistanceFieldRequest {
    pub points: Vec<StrokePoint>,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CollectPointsRequest {
    pub points: Vec<StrokePoint>,
    pub config: PointOptimizationConfig,
    pub last_time: u64,
    pub last_x: f32,
    pub last_y: f32,
    pub current_time: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CollectPointsResponse {
    pub collected_points: Vec<StrokePoint>,
    pub last_time: u64,
    pub last_x: f32,
    pub last_y: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchProcessDrawCommandsRequest {
    pub commands: Vec<DrawCommand>,
    pub min_distance: f32,
    pub max_batch_size: u32,
}

// 颜色相关结构体
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RGBColor {
    pub r: f32, // 0-255
    pub g: f32, // 0-255
    pub b: f32, // 0-255
    pub a: f32, // 0-1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HSVColor {
    pub h: f32, // 0-360
    pub s: f32, // 0-1
    pub v: f32, // 0-1
    pub a: f32, // 0-1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorAdjustRequest {
    pub color: RGBColor,
    pub brightness: f32,
    pub contrast: f32,
    pub saturation: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorConvertRequest {
    pub color: RGBColor,
    pub target_format: String, // "hsv" or "rgb"
}

// 路径平滑相关结构体
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathSmoothRequest {
    pub points: Vec<StrokePoint>,
    pub smoothness: f32, // 0-1
    pub algorithm: String, // "bezier" or "moving_average"
}

// 扩展碰撞检测相关结构体
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Circle {
    pub x: f32,
    pub y: f32,
    pub radius: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Line {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexCollisionRequest {
    pub shape1_type: String, // "rect", "circle", "line", "stroke"
    pub shape1_data: serde_json::Value,
    pub shape2_type: String, // "rect", "circle", "line", "stroke"
    pub shape2_data: serde_json::Value,
}

#[wasm_bindgen]
pub fn distance(x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    let dx = x2 - x1;
    let dy = y2 - y1;
    (dx * dx + dy * dy).sqrt()
}

#[wasm_bindgen]
pub fn quantize_coord(coord: f32, step: f32) -> f32 {
    (coord / step).round() * step
}

#[wasm_bindgen]
pub fn perpendicular_distance(px: f32, py: f32, x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    let a = px - x1;
    let b = py - y1;
    let c = x2 - x1;
    let d = y2 - y1;
    
    let dot = a * c + b * d;
    let len_sq = c * c + d * d;
    let mut param = -1.0;
    
    if len_sq != 0.0 {
        param = dot / len_sq;
    }
    
    let (xx, yy) = if param < 0.0 {
        (x1, y1)
    } else if param > 1.0 {
        (x2, y2)
    } else {
        (x1 + param * c, y1 + param * d)
    };
    
    distance(px, py, xx, yy)
}

// 点简化函数 - 优化版
#[inline]
fn simplify_points_iterative(points: &[StrokePoint], epsilon: f32) -> Vec<StrokePoint> {
    let point_count = points.len();
    if point_count == 0 {
        return Vec::new();
    }
    
    if point_count <= 2 {
        return points.to_vec();
    }
    
    // 预分配内存，避免多次扩容
    let mut result = Vec::with_capacity(point_count);
    let mut stack = Vec::with_capacity(16); // 栈深度通常不会太大
    
    // 使用栈存储需要处理的区间 [start, end]
    stack.push((0, point_count - 1));
    
    while let Some((start, end)) = stack.pop() {
        if start >= end || start >= point_count || end >= point_count {
            if start < point_count {
                result.push(points[start].clone());
            }
            continue;
        }
        
        let mut max_dist = 0.0;
        let mut max_index = start;
        
        let start_point = &points[start];
        let end_point = &points[end];
        
        // 优化：只计算关键帧点，减少计算量
        let step = if end - start > 100 {
            (end - start) / 100
        } else {
            1
        };
        
        // 确保step至少为1
        let step = step.max(1);
        
        // 遍历关键帧点
        for i in ((start + 1)..end).step_by(step) {
            if i >= point_count {
                break;
            }
            let point = &points[i];
            let dist = perpendicular_distance(
                point.from_x,
                point.from_y,
                start_point.from_x,
                start_point.from_y,
                end_point.to_x,
                end_point.to_y
            );
            
            if dist > max_dist {
                max_dist = dist;
                max_index = i;
            }
        }
        
        // 对找到的最大距离点周围进行精细检查
        let fine_start = (max_index.saturating_sub(step)).max(start + 1);
        let fine_end = (max_index + step).min(end - 1);
        
        // 确保fine_start <= fine_end
        if fine_start <= fine_end {
            for i in fine_start..=fine_end {
                if i >= point_count {
                    break;
                }
                let point = &points[i];
                let dist = perpendicular_distance(
                    point.from_x,
                    point.from_y,
                    start_point.from_x,
                    start_point.from_y,
                    end_point.to_x,
                    end_point.to_y
                );
                
                if dist > max_dist {
                    max_dist = dist;
                    max_index = i;
                }
            }
        }
        
        if max_dist > epsilon {
            // 先压入右半部分，再压入左半部分，保证处理顺序
            stack.push((max_index, end));
            stack.push((start, max_index));
        } else {
            result.push(points[start].clone());
            result.push(points[end].clone());
        }
    }
    
    // 去重相邻重复点 - 优化版
    let mut unique_result = Vec::with_capacity(result.len());
    for point in result {
        if unique_result.is_empty() || {
            let last: &StrokePoint = unique_result.last().unwrap();
            !((point.from_x - last.from_x).abs() < 0.001 &&
              (point.from_y - last.from_y).abs() < 0.001 &&
              (point.to_x - last.to_x).abs() < 0.001 &&
              (point.to_y - last.to_y).abs() < 0.001)
        } {
            unique_result.push(point);
        }
    }
    
    unique_result
}

#[wasm_bindgen]
pub fn simplify_points(points_json: &str, epsilon: f32) -> String {
    let points: Vec<StrokePoint> = match serde_json::from_str(points_json) {
        Ok(p) => p,
        Err(e) => {
            return serde_json::to_string(&ErrorResponse {
                error: format!("Failed to parse points: {}", e)
            }).unwrap_or_default();
        }
    };
    
    let result = simplify_points_iterative(&points, epsilon);
    
    serde_json::to_string(&result).unwrap_or_default()
}

#[wasm_bindgen]
pub fn process_stroke_points(request_json: &str) -> String {
    let request: ProcessPointsRequest = match serde_json::from_str(request_json) {
        Ok(r) => r,
        Err(e) => {
            return serde_json::to_string(&ErrorResponse {
                error: format!("Failed to parse request: {}", e)
            }).unwrap_or_default();
        }
    };
    let mut processed_points = Vec::new();
    
    for point in request.points {
        let q_from_x = quantize_coord(point.from_x, request.config.quantization);
        let q_from_y = quantize_coord(point.from_y, request.config.quantization);
        let q_to_x = quantize_coord(point.to_x, request.config.quantization);
        let q_to_y = quantize_coord(point.to_y, request.config.quantization);
        
        if distance(q_from_x, q_from_y, q_to_x, q_to_y) >= request.config.min_distance {
            processed_points.push(StrokePoint {
                from_x: q_from_x,
                from_y: q_from_y,
                to_x: q_to_x,
                to_y: q_to_y,
            });
        }
    }
    
    let simplified = simplify_points_iterative(&processed_points, request.config.epsilon);
    
    serde_json::to_string(&simplified).unwrap_or_default()
}

#[wasm_bindgen]
pub fn batch_process_strokes(request_json: &str) -> String {
    let request: BatchProcessRequest = match serde_json::from_str(request_json) {
        Ok(r) => r,
        Err(e) => {
            return serde_json::to_string(&ErrorResponse {
                error: format!("Failed to parse request: {}", e)
            }).unwrap_or_default();
        }
    };
    
    let result = request.strokes.into_iter()
        .map(|mut stroke| {
            let mut processed_points = Vec::new();
            
            for point in stroke.points {
                let q_from_x = quantize_coord(point.from_x, request.config.quantization);
                let q_from_y = quantize_coord(point.from_y, request.config.quantization);
                let q_to_x = quantize_coord(point.to_x, request.config.quantization);
                let q_to_y = quantize_coord(point.to_y, request.config.quantization);
                
                if distance(q_from_x, q_from_y, q_to_x, q_to_y) >= request.config.min_distance {
                    processed_points.push(StrokePoint {
                        from_x: q_from_x,
                        from_y: q_from_y,
                        to_x: q_to_x,
                        to_y: q_to_y,
                    });
                }
            }
            
            let simplified = simplify_points_iterative(&processed_points, request.config.epsilon);
            
            stroke.points = simplified;
            stroke
        })
        .collect::<Vec<Stroke>>();
    
    serde_json::to_string(&result).unwrap_or_default()
}

#[wasm_bindgen]
pub fn optimize_draw_commands(request_json: &str) -> String {
    let request: BatchDrawRequest = match serde_json::from_str(request_json) {
        Ok(r) => r,
        Err(e) => {
            return serde_json::to_string(&ErrorResponse {
                error: format!("Failed to parse request: {}", e)
            }).unwrap_or_default();
        }
    };
    
    let original_count = request.commands.len();
    
    use std::collections::HashMap;
    let mut groups: HashMap<String, Vec<DrawCommand>> = HashMap::new();
    
    for command in request.commands {
        let key = format!("{}_{}_{}", command.r#type, command.color, command.line_width);
        groups.entry(key).or_default().push(command);
    }
    
    let mut optimized_commands = Vec::new();
    
    for (_, commands) in groups {
        optimized_commands.extend(commands);
    }
    
    let optimized_count = optimized_commands.len();
    let state_switches_reduced = original_count.saturating_sub(optimized_count);
    
    web_sys::console::log_1(&format!(
        "批量绘制命令优化: 从 {} 到 {} (减少 {} 次状态切换)",
        original_count, optimized_count, state_switches_reduced
    ).into());
    
    serde_json::to_string(&optimized_commands).unwrap_or_default()
}

#[wasm_bindgen]
pub fn apply_image_filter(request_json: &str) -> String {
    let request: ImageFilterRequest = match serde_json::from_str(request_json) {
        Ok(r) => r,
        Err(e) => {
            return serde_json::to_string(&ErrorResponse {
                error: format!("Failed to parse request: {}", e)
            }).unwrap_or_default();
        }
    };
    
    let img = match decode_base64_image(&request.image_data) {
        Ok(i) => i,
        Err(e) => {
            return serde_json::to_string(&ErrorResponse {
                error: e
            }).unwrap_or_default();
        }
    };
    
    let filtered = apply_filter(&img, request.brightness, request.contrast, request.saturation);
    
    let mut buffer = Vec::new();
    match filtered.write_to(&mut std::io::Cursor::new(&mut buffer), image::ImageFormat::Png) {
        Ok(_) => {}
        Err(e) => {
            return serde_json::to_string(&ErrorResponse {
                error: format!("Failed to encode image: {}", e)
            }).unwrap_or_default();
        }
    }
    
    let result = format!("data:image/png;base64,{}", general_purpose::STANDARD.encode(&buffer));
    
    result
}

#[wasm_bindgen]
pub fn batch_apply_image_filter(request_json: &str) -> String {
    let request: BatchImageFilterRequest = match serde_json::from_str(request_json) {
        Ok(r) => r,
        Err(e) => {
            return serde_json::to_string(&ErrorResponse {
                error: format!("Failed to parse request: {}", e)
            }).unwrap_or_default();
        }
    };
    
    let mut results = Vec::new();
    
    for image_data in request.images {
        let img = match decode_base64_image(&image_data) {
            Ok(i) => i,
            Err(e) => {
                results.push(format!("error: {}", e));
                continue;
            }
        };
        
        let filtered = apply_filter(&img, request.brightness, request.contrast, request.saturation);
        
        let mut buffer = Vec::new();
        match filtered.write_to(&mut std::io::Cursor::new(&mut buffer), image::ImageFormat::Png) {
            Ok(_) => {}
            Err(e) => {
                results.push(format!("error: {}", e));
                continue;
            }
        }
        
        let result = format!("data:image/png;base64,{}", general_purpose::STANDARD.encode(&buffer));
        results.push(result);
    }
    
    serde_json::to_string(&results).unwrap_or_default()
}

#[wasm_bindgen]
pub fn transform_points(request_json: &str) -> String {
    let request: TransformRequest = match serde_json::from_str(request_json) {
        Ok(r) => r,
        Err(e) => {
            return serde_json::to_string(&ErrorResponse {
                error: format!("Failed to parse request: {}", e)
            }).unwrap_or_default();
        }
    };
    
    let mut transformed_points = Vec::new();
    
    for point in request.points {
        let from_x = request.matrix[0] * point.from_x + request.matrix[1] * point.from_y + request.matrix[2];
        let from_y = request.matrix[3] * point.from_x + request.matrix[4] * point.from_y + request.matrix[5];
        
        let to_x = request.matrix[0] * point.to_x + request.matrix[1] * point.to_y + request.matrix[2];
        let to_y = request.matrix[3] * point.to_x + request.matrix[4] * point.to_y + request.matrix[5];
        
        transformed_points.push(StrokePoint {
            from_x,
            from_y,
            to_x,
            to_y,
        });
    }
    
    serde_json::to_string(&transformed_points).unwrap_or_default()
}

#[wasm_bindgen]
pub fn detect_collision(request_json: &str) -> bool {
    let request: CollisionRequest = match serde_json::from_str(request_json) {
        Ok(r) => r,
        Err(_) => return false,
    };
    
    let rect = request.rect;
    let rect_left = rect[0];
    let rect_top = rect[1];
    let rect_right = rect[0] + rect[2];
    let rect_bottom = rect[1] + rect[3];
    
    for point in &request.stroke.points {
        if line_rect_intersect(
            point.from_x, point.from_y,
            point.to_x, point.to_y,
            rect_left, rect_top, rect_right, rect_bottom
        ) {
            return true;
        }
    }
    
    false
}

#[wasm_bindgen]
pub fn calculate_distance_field(request_json: &str) -> String {
    let request: DistanceFieldRequest = match serde_json::from_str(request_json) {
        Ok(r) => r,
        Err(e) => {
            return serde_json::to_string(&ErrorResponse {
                error: format!("Failed to parse request: {}", e)
            }).unwrap_or_default();
        }
    };
    
    let width = request.width;
    let height = request.height;
    let mut distance_field = vec![f32::MAX; (width * height) as usize];
    
    for point in &request.points {
        calculate_segment_distance(
            point.from_x, point.from_y,
            point.to_x, point.to_y,
            &mut distance_field,
            width, height
        );
    }
    
    serde_json::to_string(&distance_field).unwrap_or_default()
}

#[wasm_bindgen]
pub fn collect_points(request_json: &str) -> String {
    let request: CollectPointsRequest = match serde_json::from_str(request_json) {
        Ok(r) => r,
        Err(e) => {
            return serde_json::to_string(&ErrorResponse {
                error: format!("Failed to parse request: {}", e)
            }).unwrap_or_default();
        }
    };
    
    let mut collected_points = Vec::new();
    let mut last_time = request.last_time;
    let mut last_x = request.last_x;
    let mut last_y = request.last_y;
    
    for point in request.points {
        let q_from_x = quantize_coord(point.from_x, request.config.quantization);
        let q_from_y = quantize_coord(point.from_y, request.config.quantization);
        let q_to_x = quantize_coord(point.to_x, request.config.quantization);
        let q_to_y = quantize_coord(point.to_y, request.config.quantization);
        
        if distance(q_from_x, q_from_y, q_to_x, q_to_y) < request.config.min_distance {
            continue;
        }
        
        let now = request.current_time;
        
        if now - last_time < 30 {
            continue;
        }
        
        last_time = now;
        last_x = q_to_x;
        last_y = q_to_y;
        
        collected_points.push(StrokePoint {
            from_x: q_from_x,
            from_y: q_from_y,
            to_x: q_to_x,
            to_y: q_to_y
        });
        
        if collected_points.len() > 1500 {
            collected_points = simplify_points_iterative(&collected_points, request.config.epsilon);
        }
    }
    
    let response = CollectPointsResponse {
        collected_points,
        last_time,
        last_x,
        last_y
    };
    
    serde_json::to_string(&response).unwrap_or_default()
}

#[wasm_bindgen]
pub fn batch_process_draw_commands(request_json: &str) -> String {
    let request: BatchProcessDrawCommandsRequest = match serde_json::from_str(request_json) {
        Ok(r) => r,
        Err(e) => {
            return serde_json::to_string(&ErrorResponse {
                error: format!("Failed to parse request: {}", e)
            }).unwrap_or_default();
        }
    };
    
    use std::collections::HashMap;
    let mut groups: HashMap<String, Vec<DrawCommand>> = HashMap::new();
    
    for command in request.commands {
        let dist = distance(command.from_x, command.from_y, command.to_x, command.to_y);
        if dist < request.min_distance {
            continue;
        }
        
        let key = format!("{}_{}_{}", command.r#type, command.color, command.line_width);
        groups.entry(key).or_default().push(command);
    }
    
    let mut optimized_commands = Vec::new();
    
    for (_, commands) in groups {
        optimized_commands.extend(commands);
    }
    
    serde_json::to_string(&optimized_commands).unwrap_or_default()
}

// 颜色处理函数 - 优化版
#[inline]
fn rgb_to_hsv(r: f32, g: f32, b: f32, a: f32) -> HSVColor {
    let r_norm = r * 0.0039215686; // 1/255
    let g_norm = g * 0.0039215686;
    let b_norm = b * 0.0039215686;
    
    let max = r_norm.max(g_norm).max(b_norm);
    let min = r_norm.min(g_norm).min(b_norm);
    let delta = max - min;
    
    let mut h = 0.0;
    let mut s = 0.0;
    let v = max;
    
    if delta > 0.0001 {
        if max == r_norm {
            h = 60.0 * (((g_norm - b_norm) / delta) % 6.0);
        } else if max == g_norm {
            h = 60.0 * ((b_norm - r_norm) / delta + 2.0);
        } else {
            h = 60.0 * ((r_norm - g_norm) / delta + 4.0);
        }
        
        s = delta / max;
    }
    
    HSVColor { h, s, v, a }
}

#[inline]
fn hsv_to_rgb(h: f32, s: f32, v: f32, a: f32) -> RGBColor {
    let c = v * s;
    let x = c * (1.0 - ((h * 0.016666668) % 2.0 - 1.0).abs()); // 1/60
    let m = v - c;
    
    let (r_prime, g_prime, b_prime) = if h < 60.0 {
        (c, x, 0.0)
    } else if h < 120.0 {
        (x, c, 0.0)
    } else if h < 180.0 {
        (0.0, c, x)
    } else if h < 240.0 {
        (0.0, x, c)
    } else if h < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    
    RGBColor {
        r: (r_prime + m) * 255.0,
        g: (g_prime + m) * 255.0,
        b: (b_prime + m) * 255.0,
        a,
    }
}

#[wasm_bindgen]
pub fn adjust_color(request_json: &str) -> String {
    let request: ColorAdjustRequest = match serde_json::from_str(request_json) {
        Ok(r) => r,
        Err(e) => {
            return serde_json::to_string(&ErrorResponse {
                error: format!("Failed to parse request: {}", e)
            }).unwrap_or_default();
        }
    };
    
    let hsv = rgb_to_hsv(request.color.r, request.color.g, request.color.b, request.color.a);
    
    let mut adjusted_hsv = hsv;
    adjusted_hsv.v *= 1.0 + request.brightness / 100.0;
    adjusted_hsv.v = adjusted_hsv.v.max(0.0).min(1.0);
    
    adjusted_hsv.s *= 1.0 + request.saturation / 100.0;
    adjusted_hsv.s = adjusted_hsv.s.max(0.0).min(1.0);
    
    if request.contrast > 0.0 {
        adjusted_hsv.v = (adjusted_hsv.v - 0.5) * (1.0 + request.contrast / 100.0) + 0.5;
        adjusted_hsv.v = adjusted_hsv.v.max(0.0).min(1.0);
    }
    
    let adjusted_rgb = hsv_to_rgb(adjusted_hsv.h, adjusted_hsv.s, adjusted_hsv.v, adjusted_hsv.a);
    
    serde_json::to_string(&adjusted_rgb).unwrap_or_default()
}

#[wasm_bindgen]
pub fn convert_color(request_json: &str) -> String {
    let request: ColorConvertRequest = match serde_json::from_str(request_json) {
        Ok(r) => r,
        Err(e) => {
            return serde_json::to_string(&ErrorResponse {
                error: format!("Failed to parse request: {}", e)
            }).unwrap_or_default();
        }
    };
    
    if request.target_format == "hsv" {
        let hsv = rgb_to_hsv(request.color.r, request.color.g, request.color.b, request.color.a);
        serde_json::to_string(&hsv).unwrap_or_default()
    } else {
        serde_json::to_string(&request.color).unwrap_or_default()
    }
}

#[wasm_bindgen]
pub fn smooth_path(request_json: &str) -> String {
    let request: PathSmoothRequest = match serde_json::from_str(request_json) {
        Ok(r) => r,
        Err(e) => {
            return serde_json::to_string(&ErrorResponse {
                error: format!("Failed to parse request: {}", e)
            }).unwrap_or_default();
        }
    };
    
    let point_count = request.points.len();
    if point_count < 2 {
        return serde_json::to_string(&request.points).unwrap_or_default();
    }
    
    let mut smoothed_points = Vec::with_capacity(point_count);
    
    if request.algorithm == "moving_average" {
        let window_size = (3.0 + request.smoothness * 7.0) as usize;
        let half_window = window_size / 2;
        
        let mut prefix_from_x = vec![0.0; point_count + 1];
        let mut prefix_from_y = vec![0.0; point_count + 1];
        let mut prefix_to_x = vec![0.0; point_count + 1];
        let mut prefix_to_y = vec![0.0; point_count + 1];
        
        for i in 0..point_count {
            prefix_from_x[i + 1] = prefix_from_x[i] + request.points[i].from_x;
            prefix_from_y[i + 1] = prefix_from_y[i] + request.points[i].from_y;
            prefix_to_x[i + 1] = prefix_to_x[i] + request.points[i].to_x;
            prefix_to_y[i + 1] = prefix_to_y[i] + request.points[i].to_y;
        }
        
        for i in 0..point_count {
            let start = i.saturating_sub(half_window);
            let end = (i + half_window).min(point_count - 1);
            let count = end - start + 1;
            
            let sum_from_x = prefix_from_x[end + 1] - prefix_from_x[start];
            let sum_from_y = prefix_from_y[end + 1] - prefix_from_y[start];
            let sum_to_x = prefix_to_x[end + 1] - prefix_to_x[start];
            let sum_to_y = prefix_to_y[end + 1] - prefix_to_y[start];
            
            smoothed_points.push(StrokePoint {
                from_x: sum_from_x / count as f32,
                from_y: sum_from_y / count as f32,
                to_x: sum_to_x / count as f32,
                to_y: sum_to_y / count as f32,
            });
        }
    } else {
        smoothed_points.push(request.points[0].clone());
        
        let smooth_factor = request.smoothness * 0.5;
        
        for i in 1..point_count - 1 {
            let prev = &request.points[i - 1];
            let curr = &request.points[i];
            let next = &request.points[i + 1];
            
            let control1_x = curr.from_x + (prev.to_x - curr.from_x) * smooth_factor;
            let control1_y = curr.from_y + (prev.to_y - curr.from_y) * smooth_factor;
            let control2_x = curr.to_x + (next.from_x - curr.to_x) * smooth_factor;
            let control2_y = curr.to_y + (next.from_y - curr.to_y) * smooth_factor;
            
            smoothed_points.push(StrokePoint {
                from_x: control1_x,
                from_y: control1_y,
                to_x: control2_x,
                to_y: control2_y,
            });
        }
        
        smoothed_points.push(request.points.last().unwrap().clone());
    }
    
    serde_json::to_string(&smoothed_points).unwrap_or_default()
}

#[wasm_bindgen]
pub fn complex_collision_detection(request_json: &str) -> bool {
    let request: ComplexCollisionRequest = match serde_json::from_str(request_json) {
        Ok(r) => r,
        Err(_) => return false,
    };
    
    match (request.shape1_type.as_str(), request.shape2_type.as_str()) {
        ("rect", "rect") => {
            // 矩形与矩形碰撞
            let rect1: serde_json::Value = request.shape1_data.clone();
            let rect2: serde_json::Value = request.shape2_data.clone();
            
            let rect1_x = rect1["x"].as_f64().unwrap_or(0.0) as f32;
            let rect1_y = rect1["y"].as_f64().unwrap_or(0.0) as f32;
            let rect1_w = rect1["width"].as_f64().unwrap_or(0.0) as f32;
            let rect1_h = rect1["height"].as_f64().unwrap_or(0.0) as f32;
            
            let rect2_x = rect2["x"].as_f64().unwrap_or(0.0) as f32;
            let rect2_y = rect2["y"].as_f64().unwrap_or(0.0) as f32;
            let rect2_w = rect2["width"].as_f64().unwrap_or(0.0) as f32;
            let rect2_h = rect2["height"].as_f64().unwrap_or(0.0) as f32;
            
            // 轴对齐矩形碰撞检测
            !(rect1_x + rect1_w < rect2_x || 
              rect2_x + rect2_w < rect1_x || 
              rect1_y + rect1_h < rect2_y || 
              rect2_y + rect2_h < rect1_y)
        }
        ("rect", "circle") | ("circle", "rect") => {
            // 矩形与圆形碰撞
            let (rect_data, circle_data) = if request.shape1_type == "rect" {
                (&request.shape1_data, &request.shape2_data)
            } else {
                (&request.shape2_data, &request.shape1_data)
            };
            
            let rect_x = rect_data["x"].as_f64().unwrap_or(0.0) as f32;
            let rect_y = rect_data["y"].as_f64().unwrap_or(0.0) as f32;
            let rect_w = rect_data["width"].as_f64().unwrap_or(0.0) as f32;
            let rect_h = rect_data["height"].as_f64().unwrap_or(0.0) as f32;
            
            let circle_x = circle_data["x"].as_f64().unwrap_or(0.0) as f32;
            let circle_y = circle_data["y"].as_f64().unwrap_or(0.0) as f32;
            let circle_radius = circle_data["radius"].as_f64().unwrap_or(0.0) as f32;
            
            // 计算圆心到矩形的最近点
            let closest_x = circle_x.max(rect_x).min(rect_x + rect_w);
            let closest_y = circle_y.max(rect_y).min(rect_y + rect_h);
            
            // 计算圆心到最近点的距离
            let dx = circle_x - closest_x;
            let dy = circle_y - closest_y;
            let distance_sq = dx * dx + dy * dy;
            
            distance_sq <= circle_radius * circle_radius
        }
        ("circle", "circle") => {
            // 圆形与圆形碰撞
            let circle1: serde_json::Value = request.shape1_data.clone();
            let circle2: serde_json::Value = request.shape2_data.clone();
            
            let circle1_x = circle1["x"].as_f64().unwrap_or(0.0) as f32;
            let circle1_y = circle1["y"].as_f64().unwrap_or(0.0) as f32;
            let circle1_radius = circle1["radius"].as_f64().unwrap_or(0.0) as f32;
            
            let circle2_x = circle2["x"].as_f64().unwrap_or(0.0) as f32;
            let circle2_y = circle2["y"].as_f64().unwrap_or(0.0) as f32;
            let circle2_radius = circle2["radius"].as_f64().unwrap_or(0.0) as f32;
            
            // 计算两圆心之间的距离平方
            let dx = circle1_x - circle2_x;
            let dy = circle1_y - circle2_y;
            let distance_sq = dx * dx + dy * dy;
            let radius_sum = circle1_radius + circle2_radius;
            
            distance_sq <= radius_sum * radius_sum
        }
        ("line", "line") => {
            // 线段与线段碰撞
            let line1: serde_json::Value = request.shape1_data.clone();
            let line2: serde_json::Value = request.shape2_data.clone();
            
            let line1_x1 = line1["x1"].as_f64().unwrap_or(0.0) as f32;
            let line1_y1 = line1["y1"].as_f64().unwrap_or(0.0) as f32;
            let line1_x2 = line1["x2"].as_f64().unwrap_or(0.0) as f32;
            let line1_y2 = line1["y2"].as_f64().unwrap_or(0.0) as f32;
            
            let line2_x1 = line2["x1"].as_f64().unwrap_or(0.0) as f32;
            let line2_y1 = line2["y1"].as_f64().unwrap_or(0.0) as f32;
            let line2_x2 = line2["x2"].as_f64().unwrap_or(0.0) as f32;
            let line2_y2 = line2["y2"].as_f64().unwrap_or(0.0) as f32;
            
            // 使用快速排斥实验和跨立实验检测线段相交
            line_segments_intersect(
                line1_x1, line1_y1, line1_x2, line1_y2,
                line2_x1, line2_y1, line2_x2, line2_y2
            )
        }
        _ => {
            // 其他情况，默认返回false
            false
        }
    }
}

// 线段相交检测函数
#[inline]
fn line_segments_intersect(
    x1: f32, y1: f32, x2: f32, y2: f32,
    x3: f32, y3: f32, x4: f32, y4: f32
) -> bool {
    // 快速排斥实验
    if x1.max(x2) < x3.min(x4) || x3.max(x4) < x1.min(x2) ||
       y1.max(y2) < y3.min(y4) || y3.max(y4) < y1.min(y2) {
        return false;
    }
    
    // 跨立实验
    let d1 = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1);
    let d2 = (x2 - x1) * (y4 - y1) - (y2 - y1) * (x4 - x1);
    let d3 = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3);
    let d4 = (x4 - x3) * (y2 - y3) - (y4 - y3) * (x2 - x3);
    
    // 线段相交的条件：d1*d2<=0 且 d3*d4<=0
    (d1 * d2 <= 0.0) && (d3 * d4 <= 0.0)
}

// 线段与矩形相交检测
fn line_rect_intersect(
    x1: f32, y1: f32,
    x2: f32, y2: f32,
    rect_left: f32, rect_top: f32,
    rect_right: f32, rect_bottom: f32
) -> bool {
    // 快速排斥实验
    if x1 < rect_left && x2 < rect_left {
        return false;
    }
    if x1 > rect_right && x2 > rect_right {
        return false;
    }
    if y1 < rect_top && y2 < rect_top {
        return false;
    }
    if y1 > rect_bottom && y2 > rect_bottom {
        return false;
    }
    
    // 跨立实验
    let d1 = (rect_left - x1) * (y2 - y1) - (rect_top - y1) * (x2 - x1);
    let d2 = (rect_right - x1) * (y2 - y1) - (rect_top - y1) * (x2 - x1);
    let d3 = (rect_left - x1) * (y2 - y1) - (rect_bottom - y1) * (x2 - x1);
    let d4 = (rect_right - x1) * (y2 - y1) - (rect_bottom - y1) * (x2 - x1);
    
    if (d1 * d2) > 0.0 && (d3 * d4) > 0.0 {
        return false;
    }
    
    true
}

// 计算线段到网格点的距离
fn calculate_segment_distance(
    x1: f32, y1: f32,
    x2: f32, y2: f32,
    distance_field: &mut Vec<f32>,
    width: u32, height: u32
) {
    let min_x = x1.min(x2).floor() as u32;
    let max_x = x1.max(x2).ceil() as u32;
    let min_y = y1.min(y2).floor() as u32;
    let max_y = y1.max(y2).ceil() as u32;
    
    for y in min_y..=max_y {
        if y >= height {
            continue;
        }
        
        for x in min_x..=max_x {
            if x >= width {
                continue;
            }
            
            let idx = (y * width + x) as usize;
            let dist = point_to_segment_distance(x as f32, y as f32, x1, y1, x2, y2);
            
            if dist < distance_field[idx] {
                distance_field[idx] = dist;
            }
        }
    }
}

// 点到线段的距离
fn point_to_segment_distance(
    px: f32, py: f32,
    x1: f32, y1: f32,
    x2: f32, y2: f32
) -> f32 {
    let dx = x2 - x1;
    let dy = y2 - y1;
    
    if dx == 0.0 && dy == 0.0 {
        return distance(px, py, x1, y1);
    }
    
    let t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy);
    let t_clamped = t.max(0.0).min(1.0);
    
    let closest_x = x1 + t_clamped * dx;
    let closest_y = y1 + t_clamped * dy;
    
    distance(px, py, closest_x, closest_y)
}

// 解码 base64 图像
fn decode_base64_image(image_data: &str) -> Result<image::DynamicImage, String> {
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

// 应用图像滤镜
fn apply_filter(img: &image::DynamicImage, brightness: f32, contrast: f32, saturation: f32) -> image::DynamicImage {
    let (width, height) = (img.width(), img.height());
    
    let rgba_img = img.to_rgba8();
    let pixels: Vec<(u32, u32, image::Rgba<u8>)> = rgba_img
        .enumerate_pixels()
        .map(|(x, y, pixel)| {
            let r = pixel[0] as f32;
            let g = pixel[1] as f32;
            let b = pixel[2] as f32;
            let a = pixel[3];
            
            // 应用亮度和对比度
            let mut new_r = ((r - 128.0) * contrast) + 128.0 + brightness;
            let mut new_g = ((g - 128.0) * contrast) + 128.0 + brightness;
            let mut new_b = ((b - 128.0) * contrast) + 128.0 + brightness;
            
            // 应用饱和度
            let gray = 0.299 * new_r + 0.587 * new_g + 0.114 * new_b;
            new_r = gray + (new_r - gray) * saturation;
            new_g = gray + (new_g - gray) * saturation;
            new_b = gray + (new_b - gray) * saturation;
            
            //  clamp values
            new_r = new_r.clamp(0.0, 255.0);
            new_g = new_g.clamp(0.0, 255.0);
            new_b = new_b.clamp(0.0, 255.0);
            
            (x, y, image::Rgba([new_r as u8, new_g as u8, new_b as u8, a]))
        })
        .collect();
    
    let mut filtered_img: image::ImageBuffer<image::Rgba<u8>, Vec<u8>> = image::ImageBuffer::new(width, height);
    for (x, y, pixel) in pixels {
        filtered_img.put_pixel(x, y, pixel);
    }
    
    image::DynamicImage::ImageRgba8(filtered_img)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ViewportCullRequest {
    pub strokes: Vec<Stroke>,
    pub viewport: Viewport,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Viewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StrokeWithBounds {
    pub stroke: Stroke,
    pub bounds: Bounds,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Bounds {
    pub min_x: f32,
    pub min_y: f32,
    pub max_x: f32,
    pub max_y: f32,
}

#[wasm_bindgen]
pub fn cull_strokes_by_viewport(request_json: &str) -> String {
    let request: ViewportCullRequest = match serde_json::from_str(request_json) {
        Ok(r) => r,
        Err(e) => {
            return serde_json::to_string(&ErrorResponse {
                error: format!("Failed to parse request: {}", e)
            }).unwrap_or_default();
        }
    };
    
    let vp = &request.viewport;
    let vp_left = vp.x;
    let vp_top = vp.y;
    let vp_right = vp.x + vp.width;
    let vp_bottom = vp.y + vp.height;
    
    let mut visible_strokes = Vec::new();
    
    for stroke in request.strokes {
        if stroke.points.is_empty() {
            continue;
        }
        
        let mut is_visible = false;
        
        for point in &stroke.points {
            if line_rect_intersect(
                point.from_x, point.from_y,
                point.to_x, point.to_y,
                vp_left, vp_top, vp_right, vp_bottom
            ) {
                is_visible = true;
                break;
            }
        }
        
        if is_visible {
            visible_strokes.push(stroke);
        }
    }
    
    serde_json::to_string(&visible_strokes).unwrap_or_default()
}

#[wasm_bindgen]
pub fn calculate_stroke_bounds(stroke_json: &str) -> String {
    let stroke: Stroke = match serde_json::from_str(stroke_json) {
        Ok(s) => s,
        Err(e) => {
            return serde_json::to_string(&ErrorResponse {
                error: format!("Failed to parse stroke: {}", e)
            }).unwrap_or_default();
        }
    };
    
    if stroke.points.is_empty() {
        return serde_json::to_string(&Bounds {
            min_x: 0.0,
            min_y: 0.0,
            max_x: 0.0,
            max_y: 0.0,
        }).unwrap_or_default();
    }
    
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;
    
    for point in &stroke.points {
        min_x = min_x.min(point.from_x).min(point.to_x);
        min_y = min_y.min(point.from_y).min(point.to_y);
        max_x = max_x.max(point.from_x).max(point.to_x);
        max_y = max_y.max(point.from_y).max(point.to_y);
    }
    
    serde_json::to_string(&Bounds {
        min_x,
        min_y,
        max_x,
        max_y,
    }).unwrap_or_default()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EraserCollisionRequest {
    pub strokes: Vec<Stroke>,
    pub eraser_stroke: Stroke,
    pub tolerance: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EraserCollisionResponse {
    pub hit_stroke_indices: Vec<usize>,
    pub hit_point_indices: Vec<Vec<usize>>,
}

#[wasm_bindgen]
pub fn detect_eraser_collision(request_json: &str) -> String {
    let request: EraserCollisionRequest = match serde_json::from_str(request_json) {
        Ok(r) => r,
        Err(e) => {
            return serde_json::to_string(&ErrorResponse {
                error: format!("Failed to parse request: {}", e)
            }).unwrap_or_default();
        }
    };
    
    let tolerance = request.tolerance;
    let eraser_points = &request.eraser_stroke.points;
    
    let mut hit_stroke_indices = Vec::new();
    let mut hit_point_indices = Vec::new();
    
    for (stroke_idx, stroke) in request.strokes.iter().enumerate() {
        if stroke.r#type == "erase" {
            continue;
        }
        
        let mut stroke_hit_points = Vec::new();
        
        for (point_idx, stroke_point) in stroke.points.iter().enumerate() {
            for eraser_point in eraser_points {
                let dist_start = distance(
                    stroke_point.from_x, stroke_point.from_y,
                    eraser_point.from_x, eraser_point.from_y
                );
                let dist_end = distance(
                    stroke_point.to_x, stroke_point.to_y,
                    eraser_point.to_x, eraser_point.to_y
                );
                
                if dist_start <= tolerance || dist_end <= tolerance {
                    stroke_hit_points.push(point_idx);
                    break;
                }
                
                let perp_dist = perpendicular_distance(
                    eraser_point.from_x, eraser_point.from_y,
                    stroke_point.from_x, stroke_point.from_y,
                    stroke_point.to_x, stroke_point.to_y
                );
                
                if perp_dist <= tolerance {
                    stroke_hit_points.push(point_idx);
                    break;
                }
            }
        }
        
        if !stroke_hit_points.is_empty() {
            hit_stroke_indices.push(stroke_idx);
            hit_point_indices.push(stroke_hit_points);
        }
    }
    
    serde_json::to_string(&EraserCollisionResponse {
        hit_stroke_indices,
        hit_point_indices,
    }).unwrap_or_default()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BatchStrokeProcessRequest {
    pub strokes: Vec<Stroke>,
    pub config: PointOptimizationConfig,
    pub viewport: Option<Viewport>,
}

#[wasm_bindgen]
pub fn batch_process_strokes_optimized(request_json: &str) -> String {
    let request: BatchStrokeProcessRequest = match serde_json::from_str(request_json) {
        Ok(r) => r,
        Err(e) => {
            return serde_json::to_string(&ErrorResponse {
                error: format!("Failed to parse request: {}", e)
            }).unwrap_or_default();
        }
    };
    
    let mut processed_strokes = Vec::new();
    
    let vp_bounds = request.viewport.as_ref().map(|vp| {
        (vp.x, vp.y, vp.x + vp.width, vp.y + vp.height)
    });
    
    for mut stroke in request.strokes {
        if stroke.points.is_empty() {
            continue;
        }
        
        if let Some((vp_left, vp_top, vp_right, vp_bottom)) = vp_bounds {
            let mut is_visible = false;
            for point in &stroke.points {
                if line_rect_intersect(
                    point.from_x, point.from_y,
                    point.to_x, point.to_y,
                    vp_left, vp_top, vp_right, vp_bottom
                ) {
                    is_visible = true;
                    break;
                }
            }
            if !is_visible {
                continue;
            }
        }
        
        let mut processed_points = Vec::new();
        
        for point in stroke.points.iter() {
            let q_from_x = quantize_coord(point.from_x, request.config.quantization);
            let q_from_y = quantize_coord(point.from_y, request.config.quantization);
            let q_to_x = quantize_coord(point.to_x, request.config.quantization);
            let q_to_y = quantize_coord(point.to_y, request.config.quantization);
            
            if distance(q_from_x, q_from_y, q_to_x, q_to_y) >= request.config.min_distance {
                processed_points.push(StrokePoint {
                    from_x: q_from_x,
                    from_y: q_from_y,
                    to_x: q_to_x,
                    to_y: q_to_y,
                });
            }
        }
        
        let simplified = simplify_points_iterative(&processed_points, request.config.epsilon);
        
        stroke.points = simplified;
        processed_strokes.push(stroke);
    }
    
    serde_json::to_string(&processed_strokes).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_distance() {
        assert!((distance(0.0, 0.0, 3.0, 4.0) - 5.0).abs() < 0.0001);
    }
    
    #[test]
    fn test_quantize_coord() {
        assert_eq!(quantize_coord(1.3, 0.25), 1.25);
        assert_eq!(quantize_coord(1.4, 0.25), 1.5);
    }
    
    #[test]
    fn test_simplify_points() {
        let points = vec![
            StrokePoint { from_x: 0.0, from_y: 0.0, to_x: 1.0, to_y: 0.0 },
            StrokePoint { from_x: 1.0, from_y: 0.0, to_x: 2.0, to_y: 0.0 },
            StrokePoint { from_x: 2.0, from_y: 0.0, to_x: 3.0, to_y: 0.0 },
        ];
        
        let points_json = serde_json::to_string(&points).unwrap();
        let result_json = simplify_points(&points_json, 0.1);
        let simplified: Vec<StrokePoint> = serde_json::from_str(&result_json).unwrap();
        assert_eq!(simplified.len(), 2);
    }
}