// 多帧降噪着色器
// 通过对多帧图像进行像素级平均来降低噪声

struct Params {
    frame_count: u32,
    width: u32,
    height: u32,
    _padding: u32,
}

@group(0) @binding(0) var<uniform> params: Params;

// 所有输入帧合并到一个缓冲区
@group(0) @binding(1) var<storage, read> input_frames: array<u32>;

// 输出
@group(0) @binding(2) var<storage, read_write> output: array<u32>;

// 从 u32 中提取 RGBA 分量
fn unpack_pixel(packed: u32) -> vec4<f32> {
    return vec4<f32>(
        f32((packed >> 0u) & 0xFFu),
        f32((packed >> 8u) & 0xFFu),
        f32((packed >> 16u) & 0xFFu),
        f32((packed >> 24u) & 0xFFu)
    );
}

// 将 RGBA 分量打包为 u32
fn pack_pixel(rgba: vec4<f32>) -> u32 {
    let r = u32(clamp(rgba.r, 0.0, 255.0));
    let g = u32(clamp(rgba.g, 0.0, 255.0));
    let b = u32(clamp(rgba.b, 0.0, 255.0));
    let a = u32(clamp(rgba.a, 0.0, 255.0));
    return (a << 24u) | (b << 16u) | (g << 8u) | r;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= params.width || y >= params.height) {
        return;
    }
    
    let pixel_idx = y * params.width + x;
    let frame_size = params.width * params.height;
    
    var sum = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    
    // 累加所有帧的像素值
    for (var f = 0u; f < params.frame_count; f++) {
        let frame_pixel_idx = f * frame_size + pixel_idx;
        sum += unpack_pixel(input_frames[frame_pixel_idx]);
    }
    
    // 计算平均值
    let count = f32(params.frame_count);
    let avg = sum / count;
    
    // 输出结果
    output[pixel_idx] = pack_pixel(avg);
}
