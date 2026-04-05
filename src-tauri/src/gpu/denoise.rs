use wgpu::*;
use wgpu::util::{DeviceExt, BufferInitDescriptor};
use image::{DynamicImage, ImageBuffer, Rgba, GenericImageView};
use bytemuck::{Pod, Zeroable};

use super::context::GpuContext;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DenoiseParams {
    frame_count: u32,
    width: u32,
    height: u32,
    _padding: u32,
}

#[allow(dead_code)]
pub fn gpu_multi_frame_denoise(frames: &[DynamicImage]) -> Result<DynamicImage, String> {
    if frames.is_empty() {
        return Err("No frames provided".to_string());
    }
    
    let context = GpuContext::get()
        .ok_or("GPU context not initialized")?;
    
    let first_frame = &frames[0];
    let (width, height) = first_frame.dimensions();
    
    // 如果只有一帧，直接返回
    if frames.len() == 1 {
        return Ok(first_frame.clone());
    }
    
    // 限制最多 8 帧
    let frame_count = frames.len().min(8) as u32;
    let frames_to_use = &frames[..frame_count as usize];
    
    // 创建着色器模块
    let shader = context.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("Multi-frame Denoise Shader"),
        source: ShaderSource::Wgsl(include_str!("../shaders/denoise.wgsl").into()),
    });
    
    // 创建计算管线
    let pipeline = context.device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("Multi-frame Denoise Pipeline"),
        layout: None,
        module: &shader,
        entry_point: "main",
    });
    
    // 准备帧数据 - 将所有帧合并到一个 u32 数组
    let pixel_count = (width * height) as usize;
    let mut all_frames_data: Vec<u32> = Vec::with_capacity(pixel_count * frames_to_use.len());
    
    for frame in frames_to_use {
        let rgba = frame.to_rgba8();
        // 将 u8 数据重新解释为 u32 数组
        let u32_data: Vec<u32> = rgba
            .chunks_exact(4)
            .map(|chunk| {
                (chunk[3] as u32) << 24 |
                (chunk[2] as u32) << 16 |
                (chunk[1] as u32) << 8 |
                (chunk[0] as u32)
            })
            .collect();
        all_frames_data.extend(u32_data);
    }
    
    // 创建输入缓冲区（所有帧）- 手动转换为字节数组避免字节序问题
    let input_bytes: Vec<u8> = all_frames_data.iter().flat_map(|&v| {
        v.to_le_bytes().to_vec()
    }).collect();
    
    let input_buffer = context.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Input Frames Buffer"),
        contents: &input_bytes,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
    });
    
    // 创建输出缓冲区
    let output_buffer = context.device.create_buffer(&BufferDescriptor {
        label: Some("Output Buffer"),
        size: (pixel_count * 4) as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    
    // 创建参数缓冲区
    let params = DenoiseParams {
        frame_count,
        width,
        height,
        _padding: 0,
    };
    let params_buffer = context.device.create_buffer_init(&BufferInitDescriptor {
        label: Some("Params Buffer"),
        contents: bytemuck::bytes_of(&params),
        usage: BufferUsages::UNIFORM,
    });
    
    // 创建绑定组布局
    let bind_group_layout = pipeline.get_bind_group_layout(0);
    
    // 创建绑定组 - 只需要 3 个绑定
    let entries = [
        BindGroupEntry {
            binding: 0,
            resource: params_buffer.as_entire_binding(),
        },
        BindGroupEntry {
            binding: 1,
            resource: input_buffer.as_entire_binding(),
        },
        BindGroupEntry {
            binding: 2,
            resource: output_buffer.as_entire_binding(),
        },
    ];
    
    let bind_group = context.device.create_bind_group(&BindGroupDescriptor {
        label: Some("Denoise Bind Group"),
        layout: &bind_group_layout,
        entries: &entries,
    });
    
    // 创建命令编码器
    let mut encoder = context.device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("Denoise Encoder"),
    });
    
    // 计算工作组大小
    let workgroup_size_x = width.div_ceil(16);
    let workgroup_size_y = height.div_ceil(16);
    
    // 执行计算
    {
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("Denoise Compute Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(workgroup_size_x, workgroup_size_y, 1);
    }
    
    // 读取结果
    let frame_size = (pixel_count * 4) as u64;
    let staging_buffer = context.device.create_buffer(&BufferDescriptor {
        label: Some("Staging Buffer"),
        size: frame_size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, frame_size);
    
    context.queue.submit(std::iter::once(encoder.finish()));
    
    // 映射并读取结果
    let buffer_slice = staging_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer_slice.map_async(MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    context.device.poll(Maintain::Wait);
    
    rx.recv()
        .map_err(|e| format!("Failed to receive mapping result: {:?}", e))?
        .map_err(|e| format!("Failed to map buffer: {:?}", e))?;
    
    let data = buffer_slice.get_mapped_range().to_vec();
    
    // 将字节数组转换为 u32 数组（使用小端序）
    let u32_data: Vec<u32> = data
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    
    let rgba_data: Vec<u8> = u32_data
        .iter()
        .flat_map(|&packed| {
            vec![
                (packed & 0xFF) as u8,
                ((packed >> 8) & 0xFF) as u8,
                ((packed >> 16) & 0xFF) as u8,
                ((packed >> 24) & 0xFF) as u8,
            ]
        })
        .collect();
    
    // 创建输出图像
    let output_image: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::from_raw(width, height, rgba_data)
        .ok_or("Failed to create output image buffer")?;
    
    Ok(DynamicImage::ImageRgba8(output_image))
}
