use wgpu::*;
use std::sync::OnceLock;

static GPU_CONTEXT: OnceLock<GpuContext> = OnceLock::new();

#[allow(dead_code)]
pub struct GpuContext {
    pub device: Device,
    pub queue: Queue,
}

impl GpuContext {
    #[allow(dead_code)]
    pub fn get() -> Option<&'static GpuContext> {
        GPU_CONTEXT.get()
    }
    
    pub async fn init() -> Result<&'static GpuContext, String> {
        if GPU_CONTEXT.get().is_some() {
            return Ok(GPU_CONTEXT.get().unwrap());
        }
        
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });
        
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or("Failed to find GPU adapter")?;
        
        let (device, queue) = adapter
            .request_device(&DeviceDescriptor {
                label: Some("ViewStage GPU Device"),
                required_features: Features::empty(),
                required_limits: Limits::default(),
            }, None)
            .await
            .map_err(|e| format!("Failed to create GPU device: {:?}", e))?;
        
        let context = GpuContext { device, queue };
        
        GPU_CONTEXT.set(context)
            .map_err(|_| "Failed to set GPU context".to_string())?;
        
        Ok(GPU_CONTEXT.get().unwrap())
    }
}
