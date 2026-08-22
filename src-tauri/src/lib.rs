// lib.rs — ViewStage Rust 后端
// Tauri IPC 命令注册入口，集成了图像处理、设置管理、文件转换、更新检测等核心模块

use tauri::{Manager, Emitter};

use base64::{Engine as _, engine::general_purpose};
use zip::ZipArchive;
use std::io::{Read, Write};

mod image_processing;
#[cfg(target_os = "windows")]
mod camera_light;
mod connect;

use image_processing::{
    image_fetch_base64_data,
    image_update_rotation,
};

#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

#[cfg(target_os = "windows")]
const CREATE_NO_WINDOW: u32 = 0x08000000;





use serde::{Deserialize, Serialize};
use std::path::PathBuf;



// ==================== 数据结构 ====================

/// Tauri IPC 返回的图片保存结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageSaveResult {
    pub path: String,
    pub success: bool,
    pub error: Option<String>,
    pub enhanced_data: Option<String>,
}

// ==================== 系统目录 ====================

/// 集中管理应用所有存储路径
struct AppPaths {
    config_dir: std::path::PathBuf,
    cache_dir: std::path::PathBuf,
    log_dir: std::path::PathBuf,
    themes_dir: std::path::PathBuf,
    updates_dir: std::path::PathBuf,
    config_path: std::path::PathBuf,
    device_path: std::path::PathBuf,
    identity_path: std::path::PathBuf,
}

impl AppPaths {
    /// 构造所有路径，按需创建目录
    fn new(app: &tauri::AppHandle) -> Result<Self, String> {
        let config_dir = app.path().app_config_dir()
            .map_err(|e| format!("Failed to get config dir: {}", e))?;
        let cache_dir = app.path().app_cache_dir()
            .map_err(|e| format!("Failed to get cache dir: {}", e))?;
        let data_dir = app.path().app_data_dir()
            .map_err(|e| format!("Failed to get data dir: {}", e))?;

        Ok(Self {
            log_dir: config_dir.join("log"),
            themes_dir: config_dir.join("themes"),
            updates_dir: data_dir.join("updates"),
            config_path: config_dir.join("config.json"),
            device_path: config_dir.join("device.json"),
            identity_path: config_dir.join("identity.json"),
            config_dir,
            cache_dir,
        })
    }
}

/// Tauri IPC 命令：获取应用缓存目录，不存在则创建
#[tauri::command]
fn dir_fetch_cache(app: tauri::AppHandle) -> Result<String, String> {
    let paths = AppPaths::new(&app)?;
    
    if !paths.cache_dir.exists() {
        std::fs::create_dir_all(&paths.cache_dir)
            .map_err(|e| format!("Failed to create cache dir: {}", e))?;
    }
    
    Ok(paths.cache_dir.to_string_lossy().to_string())
}

/// Tauri IPC 命令：获取缓存目录总字节数
#[tauri::command]
fn cache_fetch_size(app: tauri::AppHandle) -> Result<u64, String> {
    let paths = AppPaths::new(&app)?;
    
    if !paths.cache_dir.exists() {
        return Ok(0);
    }
    
    fn directory_calc_size(path: &std::path::Path) -> u64 {
        let mut size = 0;
        if path.is_dir() {
            if let Ok(entries) = std::fs::read_dir(path) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        size += directory_calc_size(&path);
                    } else {
                        size += entry.metadata().map(|m| m.len()).unwrap_or(0);
                    }
                }
            }
        }
        size
    }
    
    Ok(directory_calc_size(&paths.cache_dir))
}

/// Tauri IPC 命令：清空缓存目录所有文件
#[tauri::command]
fn cache_delete_all(app: tauri::AppHandle) -> Result<String, String> {
    let paths = AppPaths::new(&app)?;
    
    if !paths.cache_dir.exists() {
        return Ok("缓存目录不存在".to_string());
    }
    
    fn directory_delete_contents(path: &std::path::Path) -> (u64, u32) {
        let mut size = 0u64;
        let mut count = 0u32;
        
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                let entry_path = entry.path();
                if entry_path.is_dir() {
                    let (s, c) = directory_delete_contents(&entry_path);
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
    
    let (cleared_size, cleared_files) = directory_delete_contents(&paths.cache_dir);
    
    log::info!("清除缓存: {} 字节, {} 个文件", cleared_size, cleared_files);
    
    Ok(format!("已清除 {} 个文件，共 {:.2} MB", cleared_files, cleared_size as f64 / 1024.0 / 1024.0))
}

/// Tauri IPC 命令：仅删除文档阅读器批注缓存
#[tauri::command]
fn cache_delete_doc_annotations(app: tauri::AppHandle) -> Result<String, String> {
    let paths = AppPaths::new(&app)?;

    let mut deleted = 0u32;
    
    // 删除 config_dir/doc_state 中的文件
    let doc_state_dir = paths.config_dir.join("doc_state");
    if doc_state_dir.exists() {
        if let Ok(entries) = std::fs::read_dir(&doc_state_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if !path.is_file() {
                    continue;
                }
                let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                    continue;
                };
                if name.starts_with("doc_annotations_") && name.ends_with(".json") {
                    if std::fs::remove_file(&path).is_ok() {
                        deleted += 1;
                    }
                }
            }
        }
    }
    
    // 兼容旧版：删除缓存目录中的文件
    if paths.cache_dir.exists() {
        if let Ok(entries) = std::fs::read_dir(&paths.cache_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if !path.is_file() {
                    continue;
                }
                let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                    continue;
                };
                if name.starts_with("doc_annotations_") && name.ends_with(".json") {
                    if std::fs::remove_file(&path).is_ok() {
                        deleted += 1;
                    }
                }
            }
        }
    }

    log::info!("清除文档阅读器批注缓存: {} 个文件", deleted);
    Ok(format!("已清除 {} 个文档批注缓存文件", deleted))
}

/// Tauri IPC 命令：检查是否达到自动清理缓存的间隔，若达到则执行清理
#[tauri::command]
fn cache_validate_auto_clear(app: tauri::AppHandle) -> Result<bool, String> {
    let paths = AppPaths::new(&app)?;
    let config_file = &paths.config_path;
    
    if !config_file.exists() {
        return Ok(false);
    }
    
    let config_content = match std::fs::read_to_string(&config_file) {
        Ok(c) => c,
        Err(e) => {
            log::warn!("cache_validate_auto_clear 读取配置文件失败: {}，跳过自动清除", e);
            return Ok(false);
        }
    };
    
    let config: serde_json::Value = match serde_json::from_str(&config_content) {
        Ok(v) => v,
        Err(e) => {
            log::warn!("cache_validate_auto_clear 解析配置文件失败: {}，跳过自动清除", e);
            return Ok(false);
        }
    };
    
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
        let temp_path = config_file.with_extension("json.tmp");
        write_atomic(&temp_path, &config_file, &updated_config)?;
        log::info!("首次设置自动清除缓存日期");
        return Ok(false);
    }
    
    let last_date = chrono::NaiveDate::parse_from_str(last_clear_date, "%Y-%m-%d")
        .map_err(|e| format!("Failed to parse last clear date: {}", e))?;
    let today_date = chrono::Local::now().date_naive();
    
    let days_since_last_clear = (today_date - last_date).num_days();
    
    if days_since_last_clear >= auto_clear_days as i64 {
        log::info!("执行自动清除缓存，距上次清除 {} 天", days_since_last_clear);
        
        let cache_dir = &paths.cache_dir;
        
        if cache_dir.exists() {
            // 清理缓存目录
            fn directory_delete_contents(path: &std::path::Path) {
                if let Ok(entries) = std::fs::read_dir(path) {
                    for entry in entries.flatten() {
                        let entry_path = entry.path();
                        if entry_path.is_dir() {
                            directory_delete_contents(&entry_path);
                            let _ = std::fs::remove_dir(&entry_path);
                        } else {
                            let _ = std::fs::remove_file(&entry_path);
                        }
                    }
                }
            }
            directory_delete_contents(&cache_dir);
        }
        
        // 清理 config_dir/doc_state 中超过 15 天未打开的文档状态
        let doc_state_dir = paths.config_dir.join("doc_state");
        if doc_state_dir.exists() {
            let today_date = chrono::Local::now().date_naive();
            if let Ok(entries) = std::fs::read_dir(&doc_state_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if !path.is_file() {
                        continue;
                    }
                    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                        continue;
                    };
                    if !name.starts_with("doc_annotations_") || !name.ends_with(".json") {
                        continue;
                    }
                    // 读取文件检查 last_open_date
                    if let Ok(content) = std::fs::read_to_string(&path) {
                        if let Ok(data) = serde_json::from_str::<serde_json::Value>(&content) {
                            let last_open = data.get("last_open_date")
                                .and_then(|v| v.as_str())
                                .unwrap_or("");
                            if !last_open.is_empty() {
                                if let Ok(open_date) = chrono::NaiveDate::parse_from_str(last_open, "%Y-%m-%d") {
                                    let days = (today_date - open_date).num_days();
                                    if days >= 15 {
                                        let _ = std::fs::remove_file(&path);
                                        log::info!("清理超过 {} 天未打开的文档状态: {}", days, name);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        let mut updated_config = config.clone();
        updated_config["lastCacheClearDate"] = serde_json::json!(today);
        
        // 清理超过 15 天未打开的文档状态记录
        if let Some(last_doc) = config.get("lastOpenDoc") {
            if last_doc.is_object() && !last_doc.is_null() {
                let last_open_date = last_doc.get("last_open_date")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                
                if !last_open_date.is_empty() {
                    if let Ok(doc_date) = chrono::NaiveDate::parse_from_str(last_open_date, "%Y-%m-%d") {
                        let days_since_open = (today_date - doc_date).num_days();
                        if days_since_open >= 15 {
                            log::info!("清理超过 {} 天未打开的文档状态记录", days_since_open);
                            updated_config["lastOpenDoc"] = serde_json::Value::Null;
                        }
                    }
                }
            }
        }
        
        let temp_path = config_file.with_extension("json.tmp");
        write_atomic(&temp_path, &config_file, &updated_config)?;
        
        log::info!("自动清除缓存完成");
        return Ok(true);
    }
    
    Ok(false)
}

/// Tauri IPC 命令：获取应用配置目录，不存在则创建
#[tauri::command]
fn dir_fetch_config(app: tauri::AppHandle) -> Result<String, String> {
    let paths = AppPaths::new(&app)?;
    
    if !paths.config_dir.exists() {
        std::fs::create_dir_all(&paths.config_dir)
            .map_err(|e| format!("Failed to create config dir: {}", e))?;
    }
    
    let config_dir_str = paths.config_dir.to_string_lossy().to_string();
    if config_dir_str.is_empty() {
        return Err("配置目录路径为空".to_string());
    }
    Ok(config_dir_str)
}

/// Tauri IPC 命令：获取日志目录
#[tauri::command]
fn dir_fetch_log(app: tauri::AppHandle) -> Result<String, String> {
    let paths = AppPaths::new(&app)?;
    
    if !paths.log_dir.exists() {
        std::fs::create_dir_all(&paths.log_dir)
            .map_err(|e| format!("Failed to create log dir: {}", e))?;
    }
    
    Ok(paths.log_dir.to_string_lossy().to_string())
}

/// Tauri IPC 命令：获取图片保存目录 ~/Pictures/ViewStage
#[tauri::command]
fn dir_fetch_pictures_viewstage() -> Result<String, String> {
    let pictures_dir = dirs::picture_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    
    let cds_dir = pictures_dir.join("ViewStage");
    
    if !cds_dir.exists() {
        std::fs::create_dir_all(&cds_dir)
            .map_err(|e| format!("Failed to create ViewStage dir: {}", e))?;
    }
    
    Ok(cds_dir.to_string_lossy().to_string())
}

/// Tauri IPC 命令：获取用户主题目录，不存在则创建
#[tauri::command]
fn dir_fetch_theme(app: tauri::AppHandle) -> Result<String, String> {
    let paths = AppPaths::new(&app)?;
    
    if !paths.themes_dir.exists() {
        std::fs::create_dir_all(&paths.themes_dir)
            .map_err(|e| format!("Failed to create theme dir: {}", e))?;
    }
    
    Ok(paths.themes_dir.to_string_lossy().to_string())
}

#[derive(Serialize)]
struct ThemeInfo {
    name: String,
    display_name: String,
    canvas_bg: String,
    text_color: String,
}

/// Tauri IPC 命令：获取用户主题目录下所有已安装的主题信息
#[tauri::command]
fn theme_list_user(app: tauri::AppHandle) -> Result<Vec<ThemeInfo>, String> {
    let paths = AppPaths::new(&app)?;
    let theme_dir = &paths.themes_dir;

    if !theme_dir.exists() {
        return Ok(Vec::new());
    }

    let mut themes = Vec::new();
    let entries = std::fs::read_dir(&theme_dir)
        .map_err(|e| format!("Failed to read theme dir: {}", e))?;

    for entry in entries {
        let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let dir_name = path.file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();

        // 优先从 config.json 读取身份信息，回退到 theme.json
        let identity_paths = [path.join("config.json"), path.join("theme.json")];
        let mut found = false;

        for identity_path in &identity_paths {
            if identity_path.exists() {
                let content = match std::fs::read_to_string(identity_path) {
                    Ok(c) => c,
                    Err(_) => continue,
                };
                let json: serde_json::Value = match serde_json::from_str(&content) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                let pkg = json["packageName"].as_str().filter(|s| !s.is_empty());
                let disp = json["displayName"].as_str().filter(|s| !s.is_empty());
                let theme_name = pkg.unwrap_or(&dir_name);

                let theme_json_path = path.join("theme.json");
                let (canvas_bg, text_color) = if theme_json_path.exists() {
                    if let Ok(tc) = std::fs::read_to_string(&theme_json_path) {
                        if let Ok(tj) = serde_json::from_str::<serde_json::Value>(&tc) {
                            let bg = tj["canvasBgColor"].as_str().unwrap_or("#1a1a1a").to_string();
                            let txt = tj["noCameraMessage"]["textColor"].as_str().unwrap_or("#ffffff").to_string();
                            (bg, txt)
                        } else {
                            ("#1a1a1a".to_string(), "#ffffff".to_string())
                        }
                    } else {
                        ("#1a1a1a".to_string(), "#ffffff".to_string())
                    }
                } else {
                    ("#1a1a1a".to_string(), "#ffffff".to_string())
                };

                themes.push(ThemeInfo {
                    name: theme_name.to_string(),
                    display_name: disp.unwrap_or(theme_name).to_string(),
                    canvas_bg,
                    text_color,
                });
                found = true;
                break;
            }
        }

        if !found {
            let (canvas_bg, text_color) = if path.join("theme.json").exists() {
                if let Ok(tc) = std::fs::read_to_string(path.join("theme.json")) {
                    if let Ok(tj) = serde_json::from_str::<serde_json::Value>(&tc) {
                        let bg = tj["canvasBgColor"].as_str().unwrap_or("#1a1a1a").to_string();
                        let txt = tj["noCameraMessage"]["textColor"].as_str().unwrap_or("#ffffff").to_string();
                        (bg, txt)
                    } else {
                        ("#1a1a1a".to_string(), "#ffffff".to_string())
                    }
                } else {
                    ("#1a1a1a".to_string(), "#ffffff".to_string())
                }
            } else {
                ("#1a1a1a".to_string(), "#ffffff".to_string())
            };
            themes.push(ThemeInfo {
                name: dir_name.clone(),
                display_name: dir_name,
                canvas_bg,
                text_color,
            });
        }
    }

    themes.sort_by(|a, b| a.display_name.to_lowercase().cmp(&b.display_name.to_lowercase()));
    Ok(themes)
}

/// Tauri IPC 命令：删除用户安装的主题
///
/// # 参数
/// * `app` — Tauri 应用句柄
/// * `name` — 主题名称（packageName）
///
/// # 异常
/// * 主题名为空
/// * 路径遍历检测失败
/// * 主题不存在或不是用户主题
/// * 删除目录失败
#[tauri::command]
fn theme_delete(app: tauri::AppHandle, name: String) -> Result<(), String> {
    if name.is_empty() {
        return Err("Theme name cannot be empty".to_string());
    }

    let paths = AppPaths::new(&app)?;
    let theme_base = &paths.themes_dir;

    // 规范化路径防止路径遍历
    let theme_base_canonical = std::fs::canonicalize(&theme_base)
        .map_err(|_| "Themes directory not found".to_string())?;
    let theme_dir = theme_base.join(&name);
    let theme_dir_canonical = std::fs::canonicalize(&theme_dir)
        .map_err(|_| format!("Theme '{}' not found", name))?;

    if !theme_dir_canonical.starts_with(&theme_base_canonical) {
        return Err("Invalid theme name".to_string());
    }

    // 确保不是内置主题（内置主题不在 themes/ 目录下）
    if !theme_dir_canonical.join("theme.json").exists() && !theme_dir_canonical.join("config.json").exists() {
        return Err(format!("'{}' is not a valid user theme", name));
    }

    std::fs::remove_dir_all(&theme_dir_canonical)
        .map_err(|e| format!("Failed to delete theme '{}': {}", name, e))?;

    log::info!("Theme '{}' deleted", name);
    Ok(())
}

/// 在 ZIP 中按文件名模糊匹配条目索引（忽略路径前缀差异）
fn zip_find_entry(archive: &mut ZipArchive<std::fs::File>, target: &str) -> Option<usize> {
    for i in 0..archive.len() {
        if let Ok(entry) = archive.by_index(i) {
            let name = entry.name().replace('\\', "/");
            if name.ends_with(target) && (name == target || name.ends_with(&format!("/{}", target))) {
                return Some(i);
            }
        }
    }
    None
}

/// 从 ZIP 中读取指定文件名的文本内容
fn zip_read_text(archive: &mut ZipArchive<std::fs::File>, target: &str) -> Result<String, String> {
    let idx = zip_find_entry(archive, target)
        .ok_or_else(|| format!("Missing {} in .vst file", target))?;
    let mut entry = archive.by_index(idx)
        .map_err(|e| format!("Failed to read {}: {}", target, e))?;
    let mut content = String::new();
    entry.read_to_string(&mut content)
        .map_err(|e| format!("Failed to read {}: {}", target, e))?;
    Ok(content)
}

/// Tauri IPC 命令：从 .vst 文件导入主题
///
/// .vst 是重命名的 ZIP 压缩包，包含 theme.json / config.json / theme.css 等文件
///
/// # 参数
/// * `app` — Tauri 应用句柄
/// * `file_path` — .vst 文件的本地路径
/// * `force` — 是否允许覆盖已存在的同名主题
///
/// # 返回值
/// * `Ok(ThemeInfo)` — 导入成功的主题信息
///
/// # 异常
/// * 文件打开或 ZIP 解析失败
/// * 缺少必需文件（theme.json / config.json / theme.css）
/// * config.json 校验失败（缺少字段或 packageName 格式非法）
/// * theme.json 字段校验失败
/// * 主题已存在且 force 为 false
/// * 解压写入磁盘失败
#[tauri::command]
fn theme_import_vst(app: tauri::AppHandle, file_path: String, force: Option<bool>) -> Result<ThemeInfo, String> {
    let paths = AppPaths::new(&app)?;
    let theme_base = &paths.themes_dir;

    if !theme_base.exists() {
        std::fs::create_dir_all(&theme_base)
            .map_err(|e| format!("Failed to create theme dir: {}", e))?;
    }

    let file = std::fs::File::open(&file_path)
        .map_err(|e| format!("Failed to open file: {}", e))?;
    let mut archive = ZipArchive::new(file)
        .map_err(|e| format!("Invalid .vst file: {}", e))?;

    // 检测 ZIP 中是否包含公共根目录前缀（用于解压时剥离）
    let common_prefix = {
        let mut names = Vec::new();
        for i in 0..archive.len() {
            if let Ok(entry) = archive.by_index(i) {
                if !entry.is_dir() {
                    names.push(entry.name().replace('\\', "/").to_string());
                }
            }
        }

        if names.is_empty() {
            return Err("Empty .vst file".to_string());
        }

        let first = names[0].clone();
        let prefix = first.find('/').map(|i| &first[..=i]).unwrap_or("");
        if !prefix.is_empty() && names.iter().all(|n| n.starts_with(prefix)) {
            prefix.to_string()
        } else {
            String::new()
        }
    };

    if zip_find_entry(&mut archive, "theme.json").is_none() {
        return Err("Missing theme.json in .vst file (visual config)".to_string());
    }
    if zip_find_entry(&mut archive, "config.json").is_none() {
        return Err("Missing config.json in .vst file (identity)".to_string());
    }
    if zip_find_entry(&mut archive, "theme.css").is_none() {
        return Err("Missing theme.css in .vst file".to_string());
    }

    let config_json_content = zip_read_text(&mut archive, "config.json")?;
    let config_json: serde_json::Value = serde_json::from_str(&config_json_content)
        .map_err(|e| format!("Invalid config.json: {}", e))?;

    let _theme_name = config_json["name"]
        .as_str()
        .ok_or_else(|| "config.json: 'name' is required (string)".to_string())?;

    let package_name = config_json["packageName"]
        .as_str()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| "config.json: 'packageName' is required (non-empty string)".to_string())?;

    if !package_name.chars().all(|c| c.is_ascii_lowercase() || c == '.' || c == '_')
        || package_name.starts_with('.')
        || package_name.ends_with('.')
        || !package_name.contains('.')
    {
        return Err("config.json: 'packageName' must be a reverse-domain name, e.g. com.example.mytheme".to_string());
    }

    let display_name = config_json["displayName"]
        .as_str()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| "config.json: 'displayName' is required (non-empty string)".to_string())?;

    let theme_json_content = zip_read_text(&mut archive, "theme.json")?;
    let theme_json: serde_json::Value = serde_json::from_str(&theme_json_content)
        .map_err(|e| format!("Invalid theme.json: {}", e))?;

    if theme_json["showToolbarText"].as_bool().is_none() {
        return Err("theme.json: 'showToolbarText' is required (bool)".to_string());
    }

    if theme_json["showAuroraEffect"].as_bool().is_none() {
        return Err("theme.json: 'showAuroraEffect' is required (bool)".to_string());
    }

    {
        let bg = theme_json["canvasBgColor"].as_str().filter(|s| !s.is_empty());
        if bg.is_none() {
            return Err("theme.json: 'canvasBgColor' is required (non-empty string)".to_string());
        }
    }

    {
        let no_cam = theme_json.get("noCameraMessage")
            .and_then(|v| v.as_object())
            .ok_or_else(|| "theme.json: 'noCameraMessage' is required (object)".to_string())?;

        for key in &["textColor", "secondaryTextColor", "tertiaryTextColor", "textShadow"] {
            if !no_cam.contains_key(*key) {
                return Err(format!("theme.json: 'noCameraMessage.{}' is required", key));
            }
        }
    }

    // 校验 icons 字段并验证 SVG 文件存在
    let icons = theme_json.get("icons")
        .and_then(|v| v.as_object())
        .ok_or_else(|| "theme.json: 'icons' is required (object)".to_string())?;

    let required_icons = [
        "menu", "minimize", "move", "pen", "eraser", "undo", "clear",
        "camera", "camera-fill", "settings", "image", "file", "folder",
        "close", "collapse", "addFile", "word", "pdf", "scan",
        "app-settings", "doc-scan", "canvas", "source", "theme-icon", "about"
    ];

    for key in &required_icons {
        if !icons.contains_key(*key) {
            return Err(format!("theme.json: 'icons.{}' is required", key));
        }
    }

    // 不强制，仅警告：引用的图标 SVG 在 ZIP 中不存在
    for (_key, val) in icons.iter() {
        if let Some(icon_name) = val.as_str() {
            let svg_path = format!("icons/{}.svg", icon_name);
            if zip_find_entry(&mut archive, &svg_path).is_none() {
                log::warn!("Icon file 'icons/{}.svg' referenced in theme.json but not found in .vst", icon_name);
            }
        }
    }

    let target_dir = theme_base.join(package_name);
    if target_dir.exists() {
        if force.unwrap_or(false) {
            std::fs::remove_dir_all(&target_dir)
                .map_err(|e| format!("Failed to remove existing theme '{}': {}", package_name, e))?;
        } else {
            return Err(format!("Theme '{}' already exists", package_name));
        }
    }

    let prefix_len = common_prefix.len();
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)
            .map_err(|e| format!("Failed to read zip entry {}: {}", i, e))?;

        if entry.is_dir() {
            continue;
        }

        let entry_name = entry.name().replace('\\', "/");
        let relative = if prefix_len > 0 && entry_name.starts_with(&common_prefix) {
            entry_name[prefix_len..].to_string()
        } else {
            entry_name.clone()
        };

        let target_path = target_dir.join(&relative);

        if let Some(parent) = target_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create directory {:?}: {}", parent, e))?;
        }

        let mut buffer = Vec::new();
        entry.read_to_end(&mut buffer)
            .map_err(|e| format!("Failed to read entry '{}': {}", entry_name, e))?;

        let mut out_file = std::fs::File::create(&target_path)
            .map_err(|e| format!("Failed to create file {:?}: {}", target_path, e))?;
        out_file.write_all(&buffer)
            .map_err(|e| format!("Failed to write file {:?}: {}", target_path, e))?;
    }

    log::info!("Theme imported successfully: packageName='{}', displayName='{}'", package_name, display_name);

    let canvas_bg = theme_json["canvasBgColor"].as_str().unwrap_or("#1a1a1a").to_string();
    let text_color = theme_json["noCameraMessage"]["textColor"].as_str().unwrap_or("#ffffff").to_string();

    Ok(ThemeInfo {
        name: package_name.to_string(),
        display_name: display_name.to_string(),
        canvas_bg,
        text_color,
    })
}

/// Tauri IPC 命令：获取用户主题的预览图片（Base64 编码）
#[tauri::command]
fn theme_get_preview(app: tauri::AppHandle, name: String) -> Result<Option<String>, String> {
    let paths = AppPaths::new(&app)?;
    let preview_path = paths.themes_dir.join(&name).join("preview.png");

    if !preview_path.exists() {
        return Ok(None);
    }

    let bytes = std::fs::read(&preview_path)
        .map_err(|e| format!("Failed to read preview: {}", e))?;
    let b64 = general_purpose::STANDARD.encode(&bytes);
    Ok(Some(format!("data:image/png;base64,{}", b64)))
}

// ==================== 图片保存 ====================

/// 按日期生成保存路径，格式：YYYY-MM-DD/{prefix}_HH-MM-SS-SSS.{extension}
fn path_calc_save(base_dir: &str, prefix: &str, extension: &str) -> Result<(PathBuf, String), String> {
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

/// 过滤前缀字符串，只保留字母数字下划线和中划线，为空则回退 "photo"
fn string_format_prefix(prefix: &str) -> String {
    let sanitized: String = prefix
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-')
        .collect();
    if sanitized.is_empty() { "photo".to_string() } else { sanitized }
}

/// Tauri IPC 命令：将 base64 编码的图片保存到 ~/Pictures/ViewStage
///
/// # 参数
/// * `image_data` — 含 data:image 前缀的 base64 图片数据
/// * `prefix` — 文件名前缀，为空则使用 "photo"
///
/// # 返回值
/// * `Ok(ImageSaveResult)` — 包含保存路径及成功状态的保存结果
///
/// # 异常
/// * base64 解码失败
/// * 目录创建失败
/// * 文件写入失败
#[tauri::command]
fn image_save_file(image_data: String, prefix: Option<String>) -> Result<ImageSaveResult, String> {
    let base_dir = dir_fetch_pictures_viewstage()?;
    let prefix_str = string_format_prefix(&prefix.unwrap_or_else(|| "photo".to_string()));

    let decoded = image_fetch_base64_data(&image_data)?;

    let extension = if image_data.contains("image/png") {
        "png"
    } else if image_data.contains("image/jpeg") || image_data.contains("image/jpg") {
        "jpg"
    } else {
        "png"
    };

    let (file_path, _file_name) = path_calc_save(&base_dir, &prefix_str, extension)?;
    
    std::fs::write(&file_path, &decoded)
        .map_err(|e| format!("Failed to write image file: {}", e))?;
    
    Ok(ImageSaveResult {
        path: file_path.to_string_lossy().to_string(),
        success: true,
        error: None,
        enhanced_data: None,
    })
}

// ==================== WebView2 摄像头/麦克风权限 ====================

/// 为 WebView2 注册 PermissionRequested 事件处理，自动允许摄像头和麦克风权限
#[cfg(target_os = "windows")]
fn register_webview_permission_handler(webview: &tauri::Webview) {
    use webview2_com::{Microsoft::Web::WebView2::Win32::*, PermissionRequestedEventHandler};
    let _ = webview.with_webview(|pwv| unsafe {
        let controller = pwv.controller();
        let core_webview = match controller.CoreWebView2() {
            Ok(w) => w,
            Err(_) => return,
        };
        let mut token: i64 = 0;
        let _ = core_webview.add_PermissionRequested(
            &PermissionRequestedEventHandler::create(Box::new(|_webview, args| {
                if let Some(args) = args {
                    let mut kind = COREWEBVIEW2_PERMISSION_KIND::default();
                    let _ = args.PermissionKind(&mut kind);
                    match kind {
                        COREWEBVIEW2_PERMISSION_KIND_CAMERA
                        | COREWEBVIEW2_PERMISSION_KIND_MICROPHONE => {
                            let _ = args.SetState(COREWEBVIEW2_PERMISSION_STATE_ALLOW);
                        }
                        _ => {}
                    }
                }
                Ok(())
            })),
            &mut token,
        );
    });
}

#[cfg(not(target_os = "windows"))]
fn register_webview_permission_handler(_webview: &tauri::Webview) {}

// ==================== 全局状态 ====================

use std::sync::atomic::{AtomicBool, Ordering};

static MIRROR_STATE: AtomicBool = AtomicBool::new(false);
static OOBE_ACTIVE: AtomicBool = AtomicBool::new(false);
static MAIN_SCRIPT_LOADED: AtomicBool = AtomicBool::new(false);
static DOWNLOAD_CANCELLED: AtomicBool = AtomicBool::new(false);

// ==================== 设置窗口 ====================

/// Tauri IPC 命令：打开或聚焦设置窗口（600×600，无边框，置顶）
#[tauri::command]
async fn window_show_settings(app: tauri::AppHandle) -> Result<(), String> {
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
    .inner_size(960.0, 540.0)
    .resizable(false)
    .decorations(false)
    .always_on_top(true)
    .center()
    .build()
    .map_err(|e| format!("Failed to create settings window: {}", e))?;

    register_webview_permission_handler(window.as_ref());
    
    window.set_focus().map_err(|e| format!("Failed to focus new settings window: {}", e))?;

    Ok(())
}

/// Tauri IPC 命令：更新镜像状态并通知前端
#[tauri::command]
async fn mirror_update_state(enabled: bool, app: tauri::AppHandle) -> Result<(), String> {
    MIRROR_STATE.store(enabled, Ordering::SeqCst);
    let _ = app.emit("mirror-changed", enabled);
    Ok(())
}

/// Tauri IPC 命令：获取当前镜像状态
#[tauri::command]
async fn mirror_fetch_state() -> Result<bool, String> {
    Ok(MIRROR_STATE.load(Ordering::SeqCst))
}

/// Tauri IPC 命令：获取应用版本号（编译时注入）
#[tauri::command]
fn app_fetch_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Tauri IPC 命令：获取当前操作系统平台标识
#[tauri::command]
fn app_fetch_platform() -> String {
    #[cfg(target_os = "windows")]
    { "windows".to_string() }
    #[cfg(target_os = "linux")]
    { "linux".to_string() }
    #[cfg(target_os = "macos")]
    { "macos".to_string() }
    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
    { "unknown".to_string() }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GitHubAsset {
    name: String,
    browser_download_url: String,
    size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GitHubRelease {
    tag_name: String,
    name: Option<String>,
    html_url: String,
    body: Option<String>,
    assets: Vec<GitHubAsset>,
}

/// SECTL latest-tag API 响应
#[derive(Debug, Deserialize)]
struct SectlLatestTagResponse {
    #[allow(dead_code)]
    success: bool,
    #[allow(dead_code)]
    project: Option<SectlProject>,
    latest: Option<SectlLatestRelease>,
    tag: Option<String>,
    #[allow(dead_code)]
    error: Option<String>,
    error_description: Option<String>,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct SectlProject {
    id: String,
    name: String,
    slug: String,
    repo: String,
    cached_latest_version: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SectlLatestRelease {
    #[allow(dead_code)]
    tag: String,
    name: Option<String>,
    #[allow(dead_code)]
    source: Option<String>,
    html_url: Option<String>,
    #[allow(dead_code)]
    published_at: Option<String>,
    #[allow(dead_code)]
    prerelease: Option<bool>,
    #[allow(dead_code)]
    draft: Option<bool>,
}

/// 版本检测结果
#[derive(Debug, Clone, Serialize, Deserialize)]
struct UpdateCheckResult {
    has_update: bool,
    current_version: String,
    latest_version: String,
    release: Option<GitHubRelease>,
    current_release: Option<GitHubRelease>,
}

/// 解析语义化版本字符串为三元组，忽略前导 'v'
fn version_calc_parse(version: &str) -> Option<(u32, u32, u32)> {
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

/// 比较两个版本号，判断 latest 是否比 current 更新
fn version_validate_newer(current: &str, latest: &str) -> bool {
    let current_ver = version_calc_parse(current);
    let latest_ver = version_calc_parse(latest);
    
    match (current_ver, latest_ver) {
        (Some(c), Some(l)) => l > c,
        _ => false,
    }
}

/// 校验 URL 是否为合法的 GitHub 域名，支持 gh-proxy.com / ghproxy.sectl.cn 等镜像前缀
fn url_validate_github(url: &str) -> Result<(), String> {
    let known_mirror_prefixes = [
        "https://gh-proxy.com/",
        "https://ghproxy.sectl.cn/",
    ];

    for prefix in &known_mirror_prefixes {
        if url.starts_with(prefix) {
            let original_url = url.strip_prefix(prefix).unwrap_or(url);
            let parsed = url::Url::parse(original_url).map_err(|e| format!("Invalid URL: {}", e))?;
            let host = parsed.host_str().unwrap_or("");
            let valid_domains = ["github.com", "www.github.com", "api.github.com"];
            if !valid_domains.contains(&host) {
                return Err(format!("Invalid GitHub URL: unexpected domain {}", host));
            }
            return Ok(());
        }
    }

    let parsed = url::Url::parse(url).map_err(|e| format!("Invalid URL: {}", e))?;
    
    let valid_domains = ["github.com", "www.github.com", "api.github.com"];
    let host = parsed.host_str().unwrap_or("");
    
    if !valid_domains.contains(&host) {
        return Err(format!("Invalid GitHub URL: unexpected domain {}", host));
    }
    
    Ok(())
}

/// Tauri IPC 命令：检查是否有新版本
///
/// 通过 SECTL API 获取最新 Tag 并与当前编译版本比较，
/// 有更新时再从 GitHub 拉取具体 Release 数据（已确认版本，非匿名探测）
#[tauri::command]
async fn update_fetch_check() -> Result<UpdateCheckResult, String> {
    let current_version = env!("CARGO_PKG_VERSION");
    
    let client = reqwest::Client::builder()
        .user_agent("ViewStage")
        .timeout(std::time::Duration::from_secs(10))
        .https_only(true)
        .build()
        .map_err(|e| e.to_string())?;
    
    // 1. 通过 SECTL 获取最新 Tag
    let response = client
        .get("https://appwrite.sectl.cn/api/software/latest-tag?projectSlug=ViewStage")
        .send()
        .await
        .map_err(|e| format!("Network error: {}", e))?;
    
    if !response.status().is_success() {
        return Err(format!("SECTL API error: {}", response.status()));
    }
    
    let sectl: SectlLatestTagResponse = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse SECTL response: {}", e))?;
    
    if !sectl.success {
        let desc = sectl.error_description.unwrap_or_default();
        return Err(format!("SECTL API error: {}", desc));
    }
    
    let latest_tag = sectl.tag.ok_or("Missing tag in SECTL response")?;
    if latest_tag.is_empty() {
        return Err("Invalid SECTL response: empty tag".to_string());
    }
    
    let latest_version = latest_tag.trim_start_matches('v');
    let has_update = version_validate_newer(current_version, latest_version);
    
    // 2. 有更新时获取该 Release 的详细数据（GitHub，已知版本）
    let release = if has_update {
        let github_url = format!(
            "https://api.github.com/repos/ospneam/ViewStage/releases/tags/{}",
            latest_tag
        );
        match client.get(&github_url).send().await {
            Ok(resp) if resp.status().is_success() => resp.json::<GitHubRelease>().await.ok(),
            _ => {
                // 降级：用 SECTL 数据构建最小 release
                let sectl_latest = sectl.latest.as_ref();
                Some(GitHubRelease {
                    tag_name: latest_tag.clone(),
                    name: sectl_latest.and_then(|l| l.name.clone()),
                    html_url: sectl_latest
                        .and_then(|l| l.html_url.clone())
                        .unwrap_or_default(),
                    body: None,
                    assets: vec![],
                })
            }
        }
    } else {
        None
    };
    
    // 3. 尝试获取当前版本的 Release 信息（GitHub，已知版本）
    let current_tag = format!("v{}", current_version);
    let current_release = match client
        .get(&format!(
            "https://api.github.com/repos/ospneam/ViewStage/releases/tags/{}",
            current_tag
        ))
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => resp.json::<GitHubRelease>().await.ok(),
        _ => None,
    };
    
    Ok(UpdateCheckResult {
        has_update,
        current_version: current_version.to_string(),
        latest_version: latest_version.to_string(),
        release,
        current_release,
    })
}

/// 遥测 HTTP POST 请求（绕过 CORS）
#[tauri::command]
async fn telemetry_http_post(url: String, body: String) -> Result<String, String> {
    let client = reqwest::Client::builder()
        .user_agent("ViewStage")
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|e| e.to_string())?;

    let response = client
        .post(&url)
        .header("Content-Type", "application/json")
        .body(body)
        .send()
        .await
        .map_err(|e| format!("Network error: {}", e))?;

    let status = response.status();
    let text = response
        .text()
        .await
        .map_err(|e| format!("Failed to read response: {}", e))?;

    if !status.is_success() {
        log::warn!("Telemetry POST failed: status={}, body={}", status, text);
        return Err(format!("HTTP error: {} - {}", status, text));
    }

    Ok(text)
}

/// 遥测 HTTP GET 请求（绕过 CORS）
#[tauri::command]
async fn telemetry_http_get(url: String) -> Result<String, String> {
    let client = reqwest::Client::builder()
        .user_agent("ViewStage")
        .timeout(std::time::Duration::from_secs(5))
        .build()
        .map_err(|e| e.to_string())?;

    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("Network error: {}", e))?;

    if !response.status().is_success() {
        return Err(format!("HTTP error: {}", response.status()));
    }

    let text = response
        .text()
        .await
        .map_err(|e| format!("Failed to read response: {}", e))?;

    Ok(text)
}

/// 获取 CPU 和 GPU 信息
#[tauri::command]
async fn telemetry_fetch_cpu_gpu() -> Result<String, String> {
    let mut result = String::new();

    // 获取 CPU 信息
    let sys = sysinfo::System::new_all();
    let cpus = sys.cpus();
    if !cpus.is_empty() {
        let cpu_brand = cpus[0].brand().trim();
        if !cpu_brand.is_empty() {
            result.push_str(cpu_brand);
        }
    }

    // 获取 GPU 信息（Windows）
    #[cfg(windows)]
    {
        use std::process::Command;
        let output = Command::new("wmic")
            .args(["path", "win32_VideoController", "get", "name"])
            .output();

        if let Ok(output) = output {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let lines: Vec<&str> = stdout.lines().collect();
            // 跳过第一行 "Name"，获取所有 GPU
            let gpus: Vec<&str> = lines
                .iter()
                .skip(1)
                .map(|l| l.trim())
                .filter(|l| !l.is_empty())
                .collect();

            if !gpus.is_empty() {
                if !result.is_empty() {
                    result.push_str(" | ");
                }
                result.push_str(&gpus.join(", "));
            }
        }
    }

    if result.is_empty() {
        result = "Unknown".to_string();
    }

    Ok(result)
}

/// 备份损坏的配置文件，文件名带时间戳
fn config_backup_corrupted(config_path: &std::path::Path) {
    let parent = config_path.parent().unwrap_or_else(|| std::path::Path::new("."));
    let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
    let backup_name = format!("config.json.corrupted_{}", timestamp);
    let backup_path = parent.join(&backup_name);
    if let Err(e) = std::fs::copy(config_path, &backup_path) {
        log::warn!("备份损坏的配置文件失败: {}", e);
    } else {
        log::info!("损坏的配置文件已备份到: {:?}", backup_path);
    }
}

/// 生成默认配置（各字段均设初始值）
fn config_fetch_default() -> serde_json::Value {
    serde_json::json!({
        "language": "zh-CN",
        "defaultCamera": "",
        "cameraWidth": 1280,
        "cameraHeight": 720,
        "frameRateMode": "adaptive",
        "defaultRotation": 0,
        "dprLimit": 2,
        "dynamicDprEnabled": true,
        "dprMin": 1,
        "dprMax": 4,
        "dprStep": 0.5,
        "penSizePresets": [2, 5, 10, 15, 21],
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
        "autoClearCacheDays": 15,
        "lastCacheClearDate": "",
        "theme": "com.viewstage.theme.simplify",
        "eraserSpeedEnabled": false,
        "eraserSizePresets": [5, 15, 25, 38, 50],
        "penEffectMode": "limited",
        "developerMode": false,
        "penMinWidthRatio": 0.2,
        "lastOpenDoc": null,
        "restoreLastDoc": true,
        "memclean": {
            "mask": 231
        },
        "blurEnabled": false,
        "pinchZoomV2": true,
        "pinchZoomV2Migrated": false
    })
}

/// JSON 值的类型名称（用于类型校验）
fn json_type_name(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

/// 一次性迁移：捏合缩放 V2 从实验性开发者选项转为默认开启。
/// 必须在读取路径（fetch）与写入路径（save 应用传入设置之前）都执行，
/// 否则"首会话内显式关闭"会被下次启动的迁移覆盖
fn config_migrate_pinch_zoom_v2(config: &mut serde_json::Value) {
    if config.get("pinchZoomV2Migrated").and_then(|v| v.as_bool()) == Some(false) {
        if let Some(obj) = config.as_object_mut() {
            obj.insert("pinchZoomV2".to_string(), serde_json::Value::Bool(true));
            obj.insert("pinchZoomV2Migrated".to_string(), serde_json::Value::Bool(true));
        }
    }
}

/// 校验并合并配置：类型不匹配的字段跳过现有值，保留默认值，并将字段名加入 recovered
fn config_validate_and_merge(
    existing: &serde_json::Value,
    defaults: &serde_json::Value,
    recovered: &mut Vec<String>,
) -> serde_json::Value {
    if let (Some(existing_obj), Some(defaults_obj)) = (existing.as_object(), defaults.as_object()) {
        let mut merged = serde_json::Map::new();
        
        for (key, value) in defaults_obj {
            merged.insert(key.clone(), value.clone());
        }
        
        for (key, value) in existing_obj {
            if let Some(default_val) = defaults_obj.get(key) {
                if json_type_name(value) == json_type_name(default_val) {
                    if !config_sanitize_value(key, value) {
                        log::warn!(
                            "配置项 '{}' 数值异常 ({}), 已恢复默认值",
                            key, value
                        );
                        recovered.push(key.clone());
                    } else {
                        merged.insert(key.clone(), value.clone());
                    }
                } else {
                    log::warn!(
                        "配置项 '{}' 类型异常 (期望 {}, 实际 {})，已恢复默认值",
                        key, json_type_name(default_val), json_type_name(value)
                    );
                    recovered.push(key.clone());
                }
            } else {
                merged.insert(key.clone(), value.clone());
            }
        }
        
        return serde_json::Value::Object(merged);
    }
    
    defaults.clone()
}

/// 对特定配置字段做数值范围检测。返回 true 表示数值在合理范围内，false 表示异常。
fn config_sanitize_value(key: &str, value: &serde_json::Value) -> bool {
    match key {
        "cameraWidth" | "cameraHeight" => {
            value.as_f64().is_some_and(|n| n >= 16.0 && n <= 7680.0)
        }
        "dprLimit" | "dprMin" | "dprMax" => {
            value.as_f64().is_some_and(|n| n >= 0.5 && n <= 16.0)
        }
        "dprStep" => {
            value.as_f64().is_some_and(|n| n >= 0.1 && n <= 4.0)
        }
        "overlayDpr" => {
            value.as_f64().is_some_and(|n| n >= 0.25 && n <= 8.0)
        }
        "defaultRotation" => {
            value.as_f64().is_some_and(|n| {
                let n = n as i64;
                n >= 0 && n <= 360
            })
        }
        "autoClearCacheDays" => {
            value.as_f64().is_some_and(|n| n >= 0.0 && n <= 3650.0)
        }
        "penMinWidthRatio" => {
            value.as_f64().is_some_and(|n| n >= 0.0 && n <= 1.0)
        }
        "penWidth" | "eraserSize" => {
            value.as_f64().is_some_and(|n| n >= 0.5 && n <= 500.0)
        }
        "maxScaleImage" => {
            value.as_f64().is_some_and(|n| n >= 1.0 && n <= 20.0)
        }
        "gestureFrameDelta" => {
            value.as_f64().is_some_and(|n| n >= 1.0 && n <= 500.0)
        }
        "penTailDuration" => {
            value.as_f64().is_some_and(|n| n >= 0.0 && n <= 2000.0)
        }
        _ => true,
    }
}

/// settings_fetch_all 命令的返回结构

/// settings_fetch_all 命令的返回结构
#[derive(Serialize)]
struct SettingsResult {
    settings: serde_json::Value,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    recovered: Vec<String>,
}

/// Tauri IPC 命令：读取配置文件，校验并合并后返回完整配置。
///
/// 配置文件不存在时返回默认配置；读取/解析失败时备份损坏文件并返回默认配置；
/// 字段类型异常时自动恢复为默认值并记录到 recovered 列表。
#[tauri::command]
async fn settings_fetch_all(app: tauri::AppHandle) -> Result<SettingsResult, String> {
    let paths = AppPaths::new(&app)?;
    let config_path = &paths.config_path;
    
    // 确保配置目录存在（首次启动或异常删除后防御）
    if !paths.config_dir.exists() {
        if let Err(e) = std::fs::create_dir_all(&paths.config_dir) {
            log::warn!("创建配置目录失败: {}", e);
        }
    }
    
    let default_config = config_fetch_default();
    
    if !config_path.exists() {
        log::info!("配置文件不存在，使用默认配置");
        return Ok(SettingsResult { settings: default_config, recovered: Vec::new() });
    }
    
    let config_content = match std::fs::read_to_string(&config_path) {
        Ok(c) => c,
        Err(e) => {
            log::warn!("读取配置文件失败: {}，使用默认配置", e);
            config_backup_corrupted(&config_path);
            return Ok(SettingsResult { settings: default_config, recovered: Vec::new() });
        }
    };
    
    let existing_config = match serde_json::from_str::<serde_json::Value>(&config_content) {
        Ok(v) => v,
        Err(e) => {
            log::warn!("解析配置文件失败: {}，使用默认配置", e);
            config_backup_corrupted(&config_path);
            return Ok(SettingsResult { settings: default_config, recovered: Vec::new() });
        }
    };
    
    let mut recovered: Vec<String> = Vec::new();
    let mut merged_config = config_validate_and_merge(&existing_config, &default_config, &mut recovered);

    // 一次性迁移：旧配置中已固化的 pinchZoomV2:false 强制翻转为 true，
    // 用户之后仍可在 Canvas 设置页手动切回经典算法
    config_migrate_pinch_zoom_v2(&mut merged_config);
    
    if merged_config != existing_config {
        let merged_str = serde_json::to_string_pretty(&merged_config)
            .map_err(|e| format!("序列化配置失败: {}", e))?;
        std::fs::write(&config_path, merged_str)
            .map_err(|e| format!("保存配置失败: {}", e))?;
    }
    
    if !recovered.is_empty() {
        let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
        let backup_name = format!("config.json.before_recovery_{}", timestamp);
        if let Some(parent_dir) = config_path.parent() {
            let backup_path = parent_dir.join(&backup_name);
            let _ = std::fs::write(&backup_path, &config_content);
            log::info!("恢复前的配置已备份到: {:?}", backup_path);
        } else {
            log::warn!("配置路径无父目录，跳过备份: {:?}", config_path);
        }
    }
    
    Ok(SettingsResult { settings: merged_config, recovered })
}

/// 将传入的 settings 合并到默认配置中（无类型校验，用于文件损坏的紧急恢复）
fn config_apply_settings_to_defaults(defaults: &serde_json::Value, settings: &serde_json::Value) -> serde_json::Value {
    let mut merged = defaults.clone();
    if let Some(obj) = merged.as_object_mut() {
        if let Some(new_obj) = settings.as_object() {
            for (key, value) in new_obj {
                obj.insert(key.clone(), value.clone());
            }
        }
    }
    merged
}

/// Tauri IPC 命令：增量保存配置（用原子写入避免文件损坏）
///
/// 现有配置与传入设置按 key 合并，先写临时文件再 rename 实现原子替换。
/// 写入前校验传入值类型，类型不匹配的字段将被跳过。
/// 配置文件损坏时备份并回退默认配置。
#[tauri::command]
async fn settings_save_all(app: tauri::AppHandle, settings: serde_json::Value) -> Result<(), String> {
    let paths = AppPaths::new(&app)?;
    
    if !paths.config_dir.exists() {
        std::fs::create_dir_all(&paths.config_dir).map_err(|e| e.to_string())?;
    }
    
    let config_path = &paths.config_path;
    let temp_path = config_path.with_extension("json.tmp");
    
    let default_config = config_fetch_default();
    
    let existing_settings: serde_json::Value = match std::fs::read_to_string(&config_path) {
        Ok(content) => {
            match serde_json::from_str::<serde_json::Value>(&content) {
                Ok(mut existing) => {
                    // 先对存量配置执行一次性迁移，再叠加传入设置：
                    // 保证"首会话内显式关闭 V2"不会被迁移覆盖
                    config_migrate_pinch_zoom_v2(&mut existing);
                    if let Some(obj) = existing.as_object_mut() {
                        if let Some(new_obj) = settings.as_object() {
                            for (key, value) in new_obj {
                                if let Some(default_val) = default_config.get(key) {
                                    if json_type_name(value) == json_type_name(default_val) {
                                        obj.insert(key.clone(), value.clone());
                                    } else {
                                        log::warn!(
                                            "保存配置时跳过字段 '{}'：类型不匹配 (期望 {}, 实际 {})",
                                            key, json_type_name(default_val), json_type_name(value)
                                        );
                                    }
                                } else {
                                    obj.insert(key.clone(), value.clone());
                                }
                            }
                        }
                    }
                    existing
                }
                Err(e) => {
                    log::warn!("保存时解析配置文件失败: {}，使用默认配置", e);
                    config_backup_corrupted(&config_path);
                    return write_atomic(&temp_path, &config_path, &config_apply_settings_to_defaults(&default_config, &settings));
                }
            }
        }
        Err(e) => {
            if config_path.exists() {
                log::warn!("保存时读取配置文件失败: {}，使用默认配置", e);
                config_backup_corrupted(&config_path);
            }
            return write_atomic(&temp_path, &config_path, &config_apply_settings_to_defaults(&default_config, &settings));
        }
    };
    
    write_atomic(&temp_path, &config_path, &existing_settings)
}

/// 原子写入 JSON 到文件（临时文件 + rename）
fn write_atomic(temp_path: &std::path::Path, config_path: &std::path::Path, value: &serde_json::Value) -> Result<(), String> {
    let config_str = serde_json::to_string_pretty(value).map_err(|e| e.to_string())?;
    std::fs::write(&temp_path, &config_str).map_err(|e| e.to_string())?;
    std::fs::rename(&temp_path, &config_path).map_err(|e| {
        let _ = std::fs::remove_file(&temp_path);
        format!("Failed to rename config file: {}", e)
    })?;
    Ok(())
}

/// Tauri IPC 命令（Windows）：检测 ViewStage 是否已设为 PDF 默认打开程序
///
/// 分别检查 HKCU UserChoice 和 HKCR 注册表路径
#[cfg(target_os = "windows")]
#[tauri::command]
async fn filetype_validate_pdf_default() -> Result<bool, String> {
    use winreg::RegKey;
    use winreg::enums::*;
    
    let hkcu = RegKey::predef(HKEY_CURRENT_USER);
    
    if let Ok(prog_id_key) = hkcu.open_subkey("Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\FileExts\\.pdf\\UserChoice") {
        if let Ok(prog_id) = prog_id_key.get_value::<String, _>("ProgId") {
            if prog_id.contains("ViewStage") || prog_id.contains("viewstage") {
                return Ok(true);
            }
        }
    }
    
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

/// Tauri IPC 命令（非 Windows）：PDF 默认程序检测始终返回 false
#[cfg(not(target_os = "windows"))]
#[tauri::command]
async fn filetype_validate_pdf_default() -> Result<bool, String> {
    Ok(false)
}

/// Tauri IPC 命令（Windows）：检测 Word 文档（.docx/.doc）是否关联到 ViewStage
#[cfg(target_os = "windows")]
#[tauri::command]
async fn filetype_validate_word_default() -> Result<bool, String> {
    use winreg::RegKey;
    use winreg::enums::*;

    fn check_ext(hkcu: &RegKey, ext: &str) -> Result<bool, String> {
        let path = format!("Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\FileExts\\{}\\UserChoice", ext);
        if let Ok(prog_id_key) = hkcu.open_subkey(&path) {
            if let Ok(prog_id) = prog_id_key.get_value::<String, _>("ProgId") {
                if prog_id.contains("ViewStage") || prog_id.contains("viewstage") {
                    return Ok(true);
                }
            }
        }
        let hkcr = RegKey::predef(HKEY_CLASSES_ROOT);
        if let Ok(ext_key) = hkcr.open_subkey(ext) {
            if let Ok(default_prog) = ext_key.get_value::<String, _>("") {
                if default_prog.contains("ViewStage") || default_prog.contains("viewstage") {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    let hkcu = RegKey::predef(HKEY_CURRENT_USER);
    if check_ext(&hkcu, ".docx")? || check_ext(&hkcu, ".doc")? {
        return Ok(true);
    }
    Ok(false)
}

/// Tauri IPC 命令（非 Windows）：Word 文档检测始终返回 false
#[cfg(not(target_os = "windows"))]
#[tauri::command]
async fn filetype_validate_word_default() -> Result<bool, String> {
    Ok(false)
}

/// 重启当前应用
fn app_restart(app: &tauri::AppHandle) {
    app.restart();
}

/// Tauri IPC 命令：删除整个配置目录后重启应用
#[tauri::command]
async fn settings_delete_all(app: tauri::AppHandle) -> Result<(), String> {
    let paths = AppPaths::new(&app)?;
    
    if paths.config_dir.exists() {
        std::fs::remove_dir_all(&paths.config_dir).map_err(|e| e.to_string())?;
        
        if paths.config_dir.exists() {
            return Err("配置目录删除失败".to_string());
        }
    }
    
    app_restart(&app);
    
    Ok(())
}

/// Tauri IPC 命令：重启应用进程
#[tauri::command]
async fn app_restart_process(app: tauri::AppHandle) -> Result<(), String> {
    app_restart(&app);
    
    Ok(())
}

/// Tauri IPC 命令：取消正在进行的更新下载
#[tauri::command]
async fn update_download_cancel() -> Result<(), String> {
    DOWNLOAD_CANCELLED.store(true, Ordering::SeqCst);
    log::info!("已发送下载取消信号");
    Ok(())
}

/// 通过 SECTL 分发接口下载文件
///
/// 调用 /api/software/download 获取 302 跳转，然后下载目标文件。
async fn download_from_sectl(
    app: &tauri::AppHandle,
    file_path: &std::path::Path,
    tag: &str,
    file_name: &str,
) -> Result<(), String> {
    let client = reqwest::Client::builder()
        .user_agent("ViewStage")
        .timeout(std::time::Duration::from_secs(300))
        .https_only(true)
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|e| format!("Failed to build client: {}", e))?;

    let params = [
        ("projectSlug", "ViewStage"),
        ("tag", tag),
        ("fileName", file_name),
        ("source", "server"),
    ];
    let sectl_url = url::Url::parse_with_params(
        "https://appwrite.sectl.cn/api/software/download",
        &params,
    )
    .map_err(|e| format!("Failed to build SECTL URL: {}", e))?
    .to_string();
    log::info!("SECTL 下载请求: {}", sectl_url);

    let response = client
        .get(&sectl_url)
        .send()
        .await
        .map_err(|e| format!("SECTL request failed: {}", e))?;

    let status = response.status();

    if !status.is_success() && status != reqwest::StatusCode::FOUND
        && status != reqwest::StatusCode::MOVED_PERMANENTLY
    {
        let body = response.text().await.unwrap_or_default();
        return Err(format!("SECTL download failed (HTTP {}): {}", status, body));
    }

    let dl_url = if status == reqwest::StatusCode::FOUND
        || status == reqwest::StatusCode::MOVED_PERMANENTLY
    {
        let location = response
            .headers()
            .get(reqwest::header::LOCATION)
            .and_then(|v| v.to_str().ok())
            .ok_or_else(|| "SECTL 302 missing Location header".to_string())?
            .to_string();
        log::info!("SECTL 重定向到: {}", location);
        location
    } else {
        log::info!("SECTL 直接返回文件内容 (HTTP {})", status);
        sectl_url.clone()
    };

    let dl_response = if dl_url == sectl_url {
        response
    } else {
        let download_client = reqwest::Client::builder()
            .user_agent("ViewStage")
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .map_err(|e| format!("Failed to build download client: {}", e))?;
        download_client
            .get(&dl_url)
            .send()
            .await
            .map_err(|e| format!("SECTL redirect download failed: {}", e))?
    };

    if !dl_response.status().is_success() {
        return Err(format!(
            "SECTL download HTTP {}",
            dl_response.status()
        ));
    }

    let total_size = dl_response.content_length().unwrap_or(0);
    let mut file = std::fs::File::create(file_path)
        .map_err(|e| format!("Failed to create file: {}", e))?;

    let mut downloaded: u64 = 0;
    let mut stream = dl_response.bytes_stream();
    use futures::stream::StreamExt;
    let mut last_reported: u32 = 0;

    while let Some(chunk) = stream.next().await {
        if DOWNLOAD_CANCELLED.load(Ordering::SeqCst) {
            let _ = std::fs::remove_file(file_path);
            return Err("Download cancelled".to_string());
        }
        let data = chunk.map_err(|e| format!("Read chunk error: {}", e))?;
        file.write_all(&data)
            .map_err(|e| format!("Write file error: {}", e))?;
        downloaded += data.len() as u64;
        if total_size > 0 {
            let pct = (downloaded as f64 / total_size as f64 * 100.0) as u32;
            if pct != last_reported {
                last_reported = pct;
                let _ = app.emit("update-download-progress", pct);
            }
        }
    }

    if total_size == 0 || last_reported < 100 {
        let _ = app.emit("update-download-progress", 100);
    }

    file.flush().map_err(|e| format!("Flush file error: {}", e))?;
    log::info!("SECTL 下载完成: {:?}", file_path);
    Ok(())
}

/// Tauri IPC 命令：下载更新文件
///
/// 优先使用 SECTL 分发接口下载，失败时回退到 GitHub 镜像加速。
#[tauri::command]
async fn update_download_file(
    app: tauri::AppHandle,
    url: String,
    file_name: String,
    mirror_url: Option<String>,
    version_tag: Option<String>,
) -> Result<String, String> {
    DOWNLOAD_CANCELLED.store(false, Ordering::SeqCst);
    log::info!("开始下载更新，文件: {}, 镜像: {:?}", file_name, mirror_url);

    let paths = AppPaths::new(&app)?;
    let updates_dir = &paths.updates_dir;
    std::fs::create_dir_all(updates_dir)
        .map_err(|e| format!("Failed to create updates dir: {}", e))?;

    let file_path = updates_dir.join(&file_name);
    log::info!("保存路径: {:?}", file_path);

    // 1. 尝试 SECTL 分发下载
    if let Some(tag) = &version_tag {
        match download_from_sectl(&app, &file_path, tag, &file_name).await {
            Ok(()) => return Ok(file_path.to_string_lossy().to_string()),
            Err(e) => {
                log::warn!("SECTL 下载失败，回退到 GitHub: {}", e);
            }
        }
    }

    // 2. 回退：GitHub 镜像加速
    log::info!("使用 GitHub 镜像回退下载: {}", file_name);
    url_validate_github(&url)?;

    let use_mirror = mirror_url.as_ref().map_or(false, |m| !m.is_empty());
    let fallback_urls: Vec<String> = if use_mirror {
        let mirror = mirror_url.as_ref().unwrap();
        let proxy_url = format!("{}/{}", mirror.trim_end_matches('/'), &url);
        log::info!("镜像 URL: {}", proxy_url);
        vec![proxy_url, url]
    } else {
        log::info!("使用原始地址下载: {}", url);
        vec![url]
    };

    let client = reqwest::Client::builder()
        .user_agent("ViewStage")
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .map_err(|e| {
            log::error!("创建 HTTP 客户端失败: {}", e);
            e.to_string()
        })?;

    // 逐个尝试 URL，直到成功或全部失败
    let mut last_error = String::new();
    let mut succeeded = false;

    for (attempt_idx, download_url) in fallback_urls.iter().enumerate() {
        if DOWNLOAD_CANCELLED.load(Ordering::SeqCst) {
            return Err("Download cancelled".to_string());
        }

        if attempt_idx > 0 {
            log::info!("镜像下载失败，尝试回退到原始地址: {}", download_url);
            app.emit("update-download-progress", 0).unwrap_or(());
        }

        log::info!("正在发起下载请求...");
        let response = match client.get(download_url).send().await {
            Ok(r) if r.status().is_success() => r,
            Ok(r) => {
                let status = r.status();
                let msg = format!("Download error: {} (HTTP {})", download_url, status);
                log::error!("{}", msg);
                last_error = msg;
                continue;
            }
            Err(e) => {
                let msg = format!("Network error for {}: {}", download_url, e);
                log::error!("{}", msg);
                last_error = msg;
                continue;
            }
        };

        let total_size = response.content_length().unwrap_or(0);
        log::info!("文件大小: {} bytes ({:.2} MB)", total_size, total_size as f64 / 1024.0 / 1024.0);

        let mut file = std::fs::File::create(&file_path)
            .map_err(|e| {
                log::error!("创建文件失败: {}", e);
                format!("Failed to create file: {}", e)
            })?;

        let mut downloaded: u64 = 0;
        let mut stream = response.bytes_stream();
        use futures::stream::StreamExt;

        let mut last_reported_progress: u32 = 0;

        log::info!("开始接收数据...");
        let mut stream_error = None;
        while let Some(chunk) = stream.next().await {
            if DOWNLOAD_CANCELLED.load(Ordering::SeqCst) {
                let _ = std::fs::remove_file(&file_path);
                log::info!("下载已被用户取消");
                return Err("Download cancelled".to_string());
            }

            match chunk {
                Ok(data) => {
                    if let Err(e) = file.write_all(&data) {
                        log::error!("写入文件失败: {}", e);
                        stream_error = Some(format!("Failed to write file: {}", e));
                        break;
                    }
                    downloaded += data.len() as u64;

                    if total_size > 0 {
                        let progress = (downloaded as f64 / total_size as f64) * 100.0;
                        let current_progress = progress as u32;
                        if current_progress != last_reported_progress {
                            last_reported_progress = current_progress;
                            log::debug!("下载进度: {}%", current_progress);
                            app.emit("update-download-progress", current_progress).unwrap_or(());
                        }
                    }
                }
                Err(e) => {
                    log::error!("读取数据块失败: {}", e);
                    stream_error = Some(format!("Failed to read chunk: {}", e));
                    break;
                }
            }
        }

        if let Some(err) = stream_error {
            last_error = err;
            let _ = std::fs::remove_file(&file_path);
            continue;
        }

        // 确保最终到达 100%
        if total_size == 0 || last_reported_progress < 100 {
            app.emit("update-download-progress", 100).unwrap_or(());
        }

        file.flush().map_err(|e| {
            log::error!("刷新文件失败: {}", e);
            format!("Failed to flush file: {}", e)
        })?;

        log::info!("下载完成，已保存到: {:?}", file_path);
        succeeded = true;
        break;
    }

    if succeeded {
        Ok(file_path.to_string_lossy().to_string())
    } else {
        Err(last_error)
    }
}

/// Tauri IPC 命令：启动已下载的更新安装包并退出应用
///
/// 启动安装程序后自动退出当前应用，由安装程序接管后续流程
#[tauri::command]
async fn update_install_release(app: tauri::AppHandle, file_path: String) -> Result<(), String> {
    let path = std::path::Path::new(&file_path);
    if !path.exists() {
        log::error!("安装文件不存在: {}", file_path);
        return Err(format!("安装文件不存在: {}", file_path));
    }

    log::info!("启动安装程序: {:?}", path);

    #[cfg(target_os = "windows")]
    {
        let exe_path = path.to_string_lossy().to_string();
        std::process::Command::new("cmd")
            .arg("/c")
            .arg("start")
            .arg("")
            .arg(&exe_path)
            .spawn()
            .map_err(|e| {
                log::error!("启动安装程序失败: {}", e);
                format!("启动安装程序失败: {}", e)
            })?;
    }

    #[cfg(not(target_os = "windows"))]
    {
        std::process::Command::new("open")
            .arg(path)
            .spawn()
            .map_err(|e| {
                log::error!("启动安装程序失败: {}", e);
                format!("启动安装程序失败: {}", e)
            })?;
    }

    // 延迟退出以确保 IPC 响应返回前端
    let app_clone = app.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        app_clone.exit(0);
    });

    Ok(())
}

/// Tauri IPC 命令：隐藏启动画面，显示并聚焦主窗口
#[tauri::command]
async fn window_hide_splashscreen(app: tauri::AppHandle) -> Result<(), String> {
    if let Some(splashscreen) = app.get_webview_window("splashscreen") {
        let _ = splashscreen.close();
    }
    if let Some(main_window) = app.get_webview_window("main") {
        let _ = main_window.show();
        let _ = main_window.set_focus();
    }
    Ok(())
}

/// Tauri IPC 命令：完成 OOBE 引导后重启应用
#[tauri::command]
async fn oobe_submit_complete(app: tauri::AppHandle) -> Result<(), String> {
    OOBE_ACTIVE.store(false, Ordering::SeqCst);
    
    app_restart(&app);
    
    Ok(())
}

/// Tauri IPC 命令：检测 OOBE 是否处于激活状态
#[tauri::command]
fn oobe_check_active() -> bool {
    OOBE_ACTIVE.load(Ordering::SeqCst)
}

/// Tauri IPC 命令：标记前端主脚本已加载完成
#[tauri::command]
fn main_signal_loaded() {
    MAIN_SCRIPT_LOADED.store(true, Ordering::SeqCst);
}

/// Tauri IPC 命令：查询前端主脚本是否已加载完成
#[tauri::command]
fn main_check_loaded() -> bool {
    MAIN_SCRIPT_LOADED.load(Ordering::SeqCst)
}

/// Tauri IPC 命令：退出应用进程
#[tauri::command]
fn app_submit_exit() {
    std::process::exit(0);
}

// ==================== 设备信息检测 ====================

/// 聚合的设备信息，包含 Windows 版本、CPU、GPU、内存、磁盘、触屏等
#[derive(Debug, Clone, Serialize)]
pub struct DeviceInfo {
    pub windows_version: String,
    pub windows_build: u32,
    pub windows_display_version: String,
    pub cpu_name: String,
    pub cpu_cores: usize,
    pub cpu_arch: String,
    pub gpu_name: String,
    pub gpu_driver_version: String,
    pub gpu_driver_date: String,
    pub gpu_dedicated_memory_mb: u64,
    pub total_ram_mb: u64,
    pub system_type: String,
    pub disk_total_gb: u64,
    pub disk_type: String,
    pub has_touchscreen: bool,
}

/// Tauri IPC 命令：检测设备信息并写入 device.json
#[tauri::command]
async fn device_detect_all(app: tauri::AppHandle) -> Result<DeviceInfo, String> {
    let device_info = device_collect_info();
    let paths = AppPaths::new(&app)?;

    if !paths.config_dir.exists() {
        std::fs::create_dir_all(&paths.config_dir).map_err(|e| e.to_string())?;
    }

    let json = serde_json::to_string_pretty(&device_info).map_err(|e| e.to_string())?;
    std::fs::write(&paths.device_path, &json).map_err(|e| format!("保存设备信息失败: {}", e))?;

    log::info!("设备信息已保存到: {:?}", paths.device_path);

    Ok(device_info)
}

/// Tauri IPC 命令：读取或生成本机设备 UUID，持久化到 identity.json
/// 确保设备重置（清空 localStorage / 重装）后 UUID 不变
#[tauri::command]
fn get_device_uuid(app: tauri::AppHandle) -> Result<String, String> {
    let paths = AppPaths::new(&app)?;

    if paths.identity_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&paths.identity_path) {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(uuid) = val.get("device_uuid").and_then(|v| v.as_str()) {
                    if !uuid.is_empty() {
                        return Ok(uuid.to_string());
                    }
                }
            }
        }
    }

    let uuid = uuid::Uuid::new_v4().to_string();
    let data = serde_json::json!({ "device_uuid": uuid });

    if let Some(parent) = paths.identity_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(&paths.identity_path, serde_json::to_string_pretty(&data).map_err(|e| e.to_string())?)
        .map_err(|e| format!("保存设备身份失败: {}", e))?;

    log::info!("设备 UUID 已生成/读取: {}", uuid);
    Ok(uuid)
}

/// 聚合所有子检测函数的设备信息
fn device_collect_info() -> DeviceInfo {
    let (win_ver, win_build, win_display) = device_detect_windows_version();
    let (cpu_name, cpu_cores, cpu_arch) = device_detect_cpu();
    let (gpu_name, gpu_driver, gpu_driver_date, gpu_mem) = device_detect_gpu();
    let (total_ram_mb, system_type) = device_detect_system();
    let (disk_total_gb, disk_type) = device_detect_disk();
    let has_touchscreen = device_detect_touchscreen();

    DeviceInfo {
        windows_version: win_ver,
        windows_build: win_build,
        windows_display_version: win_display,
        cpu_name,
        cpu_cores,
        cpu_arch,
        gpu_name,
        gpu_driver_version: gpu_driver,
        gpu_driver_date: gpu_driver_date,
        gpu_dedicated_memory_mb: gpu_mem,
        total_ram_mb,
        system_type,
        disk_total_gb,
        disk_type,
        has_touchscreen,
    }
}

/// 检测操作系统版本信息，跨平台返回 (名称, 构建号, 显示版本)
fn device_detect_windows_version() -> (String, u32, String) {
    #[cfg(target_os = "windows")]
    {
        use winreg::RegKey;
        use winreg::enums::*;

        if let Ok(hklm) = RegKey::predef(HKEY_LOCAL_MACHINE).open_subkey(r"SOFTWARE\Microsoft\Windows NT\CurrentVersion") {
            let product_name: String = hklm.get_value("ProductName").unwrap_or_else(|_| "Windows".to_string());
            let current_build: String = hklm.get_value("CurrentBuild").unwrap_or_else(|_| "0".to_string());
            let display_version: String = hklm.get_value("DisplayVersion").unwrap_or_default();
            let release_id: String = hklm.get_value("ReleaseId").unwrap_or_default();
            let _edition_id: String = hklm.get_value("EditionID").unwrap_or_default();

            let build_number: u32 = current_build.parse().unwrap_or(0);
            let version_str = if !display_version.is_empty() {
                format!("{} {} (Build {})", product_name.trim(), display_version, current_build)
            } else if !release_id.is_empty() {
                format!("{} {} (Build {})", product_name.trim(), release_id, current_build)
            } else {
                format!("{} (Build {})", product_name.trim(), current_build)
            };

            return (version_str, build_number, display_version);
        }
    }

    #[cfg(target_os = "linux")]
    {
        let name = std::fs::read_to_string("/etc/os-release")
            .ok()
            .and_then(|content| {
                for line in content.lines() {
                    if line.starts_with("PRETTY_NAME=") {
                        let val = line.trim_start_matches("PRETTY_NAME=");
                        let trimmed = val.trim_matches('"').trim().to_string();
                        return Some(trimmed);
                    }
                }
                None
            })
            .unwrap_or_else(|| "Linux".to_string());

        let kernel = std::fs::read_to_string("/proc/version")
            .ok()
            .and_then(|content| {
                content.split_whitespace().nth(2).map(|s| s.to_string())
            })
            .unwrap_or_default();

        let build: u32 = kernel.split('.').next().and_then(|s| s.parse().ok()).unwrap_or(0);
        return (name, build, kernel);
    }

    #[cfg(not(target_os = "linux"))]
    {
        ("Unknown".to_string(), 0, String::new())
    }
}

/// 检测 CPU 型号、逻辑核心数、架构
fn device_detect_cpu() -> (String, usize, String) {
    let cpu_name: String;

    #[cfg(target_os = "windows")]
    {
        use winreg::RegKey;
        use winreg::enums::*;

        if let Ok(hklm) = RegKey::predef(HKEY_LOCAL_MACHINE).open_subkey(r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") {
            cpu_name = hklm.get_value("ProcessorNameString").unwrap_or_else(|_| "Unknown".to_string());
        } else {
            cpu_name = "Unknown".to_string();
        }
    }

    #[cfg(target_os = "linux")]
    {
        cpu_name = std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|content| {
                for line in content.lines() {
                    if line.starts_with("model name") {
                        return line.split(':').nth(1).map(|s| s.trim().to_string());
                    }
                }
                None
            })
            .unwrap_or_else(|| "Unknown".to_string());
    }

    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        cpu_name = "Unknown".to_string();
    }

    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let arch = if cfg!(target_arch = "x86_64") { "x64".to_string() }
               else if cfg!(target_arch = "x86") { "x86".to_string() }
               else if cfg!(target_arch = "aarch64") { "ARM64".to_string() }
               else { "Unknown".to_string() };

    (cpu_name.trim().to_string(), cores, arch)
}

/// 检测 GPU 名称、驱动版本、驱动日期、显存大小（MB）
fn device_detect_gpu() -> (String, String, String, u64) {
    #[cfg(target_os = "windows")]
    {
        let output = std::process::Command::new("powershell")
            .args([
                "-NoProfile", "-NonInteractive", "-Command",
                "Get-CimInstance -ClassName Win32_VideoController | Select-Object -First 1 Name, DriverVersion, DriverDate, AdapterRAM | ConvertTo-Json -Compress"
            ])
            .creation_flags(CREATE_NO_WINDOW)
            .output();

        if let Ok(output) = output {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&stdout) {
                    let name = json.get("Name").and_then(|v| v.as_str()).unwrap_or("Unknown").to_string();
                    let driver = json.get("DriverVersion").and_then(|v| v.as_str()).unwrap_or_default().to_string();
                    let driver_date = json.get("DriverDate").and_then(|v| v.as_str()).unwrap_or_default().to_string();
                    let ram = json.get("AdapterRAM").and_then(|v| v.as_u64()).unwrap_or(0);
                    return (name, driver, driver_date, ram / (1024 * 1024));
                }
            }
        }
    }

    #[cfg(target_os = "linux")]
    {
        if let Ok(output) = std::process::Command::new("lspci")
            .args(["-mm"])
            .output()
        {
            let stdout = String::from_utf8_lossy(&output.stdout);
            for line in stdout.lines() {
                if line.contains("VGA") || line.contains("3D") || line.contains("Display") {
                    let parts: Vec<&str> = line.split('"').collect();
                    if parts.len() >= 3 {
                        let name = parts[1].trim().to_string();
                        if !name.is_empty() {
                            // Try to get VRAM from sysfs
                            let vram = std::fs::read_to_string("/sys/class/drm/card0/device/mem_info_vram_total")
                                .ok()
                                .and_then(|s| s.trim().parse::<u64>().ok())
                                .map(|b| b / (1024 * 1024))
                                .unwrap_or(0);
                            return (name, String::new(), String::new(), vram);
                        }
                    }
                }
            }
        }

        // Fallback: read from /sys/class/drm
        if let Ok(entries) = std::fs::read_dir("/sys/class/drm") {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with("card") && !name.contains('-') {
                    let device_path = entry.path().join("device");
                    let gpu_name = std::fs::read_to_string(device_path.join("uevent"))
                        .ok()
                        .and_then(|c| {
                            for l in c.lines() {
                                if l.starts_with("DRIVER=") {
                                    return l.split('=').nth(1).map(|s| s.to_string());
                                }
                            }
                            None
                        })
                        .unwrap_or_else(|| "Unknown".to_string());
                    return (gpu_name, String::new(), String::new(), 0);
                }
            }
        }
    }

    ("Unknown".to_string(), String::new(), String::new(), 0)
}

/// 检测总物理内存（MB）和系统类型（Desktop/Laptop/Tablet 等）
fn device_detect_system() -> (u64, String) {
    #[cfg(target_os = "windows")]
    {
        let output = std::process::Command::new("powershell")
            .args([
                "-NoProfile", "-NonInteractive", "-Command",
                "Get-CimInstance -ClassName Win32_ComputerSystem | Select-Object TotalPhysicalMemory, PCSystemType | ConvertTo-Json -Compress"
            ])
            .creation_flags(CREATE_NO_WINDOW)
            .output();

        if let Ok(output) = output {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout);
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&stdout) {
                    let ram = json.get("TotalPhysicalMemory").and_then(|v| v.as_u64()).unwrap_or(0);
                    let sys_type = json.get("PCSystemType").and_then(|v| v.as_u64()).unwrap_or(0);
                    let type_str = match sys_type {
                        1 => "Desktop".to_string(),
                        2 => "Laptop".to_string(),
                        3 => "Workstation".to_string(),
                        4 => "Enterprise Server".to_string(),
                        5 => "Tablet".to_string(),
                        _ => "Unknown".to_string(),
                    };
                    return (ram / (1024 * 1024), type_str);
                }
            }
        }
    }

    #[cfg(target_os = "linux")]
    {
        // Read total RAM from /proc/meminfo
        let total_ram_mb = std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|content| {
                for line in content.lines() {
                    if line.starts_with("MemTotal:") {
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        if parts.len() >= 2 {
                            return parts[1].parse::<u64>().ok().map(|kb| kb / 1024);
                        }
                    }
                }
                None
            })
            .unwrap_or(0);

        // Detect system type from DMI chassis type
        let system_type = std::fs::read_to_string("/sys/class/dmi/id/chassis_type")
            .ok()
            .and_then(|content| {
                match content.trim() {
                    "3" | "4" | "5" | "6" | "7" | "15" | "16" => Some("Desktop"),
                    "8" | "9" | "10" | "11" | "12" => Some("Laptop"),
                    "14" => Some("Notebook"),
                    "17" | "19" | "29" | "30" => Some("Tablet"),
                    "21" | "22" | "23" => Some("Server"),
                    _ => None,
                }
            })
            .map(|s| s.to_string())
            .unwrap_or_else(|| "Unknown".to_string());

        return (total_ram_mb, system_type);
    }

    #[cfg(not(target_os = "linux"))]
    {
        (0, "Unknown".to_string())
    }
}

/// 检测系统盘总容量（GB）和类型（SSD/HDD）
fn device_detect_disk() -> (u64, String) {
    #[cfg(target_os = "windows")]
    {
        let disk_size = {
            let output = std::process::Command::new("powershell")
                .args([
                    "-NoProfile", "-NonInteractive", "-Command",
                    "Get-CimInstance -ClassName Win32_LogicalDisk -Filter \"DriveType=3\" | Select-Object -First 1 Size | ConvertTo-Json -Compress"
                ])
                .creation_flags(CREATE_NO_WINDOW)
                .output();

            match output {
                Ok(o) if o.status.success() => {
                    let stdout = String::from_utf8_lossy(&o.stdout);
                    serde_json::from_str::<serde_json::Value>(&stdout)
                        .ok()
                        .and_then(|v| v.get("Size").and_then(|s| s.as_u64()))
                        .unwrap_or(0)
                }
                _ => 0,
            }
        };

        let disk_type = if disk_size > 0 {
            let output = std::process::Command::new("powershell")
                .args([
                    "-NoProfile", "-NonInteractive", "-Command",
                    "Get-CimInstance -ClassName Win32_DiskDrive | Select-Object -First 1 @{N='RPM';E={if ($_.RotationsPerMinute -eq $null -or $_.RotationsPerMinute -eq 0) {'SSD'} else {'HDD'}}} | ConvertTo-Json -Compress"
                ])
                .creation_flags(CREATE_NO_WINDOW)
                .output();

            match output {
                Ok(o) if o.status.success() => {
                    let stdout = String::from_utf8_lossy(&o.stdout);
                    match serde_json::from_str::<serde_json::Value>(&stdout) {
                        Ok(ref v) => v.get("RPM")
                            .and_then(|m| m.as_str())
                            .unwrap_or("Unknown")
                            .to_string(),
                        Err(_) => "Unknown".to_string(),
                    }
                }
                _ => "Unknown".to_string(),
            }
        } else {
            "Unknown".to_string()
        };

        return (disk_size / (1024 * 1024 * 1024), disk_type);
    }

    #[cfg(target_os = "linux")]
    {
        // Get total disk size for root filesystem using df
        let disk_size_gb = std::process::Command::new("df")
            .args(["-B1", "--output=size", "/"])
            .output()
            .ok()
            .and_then(|output| {
                if output.status.success() {
                    let stdout = String::from_utf8_lossy(&output.stdout);
                    stdout.lines().nth(1)
                        .and_then(|line| line.trim().parse::<u64>().ok())
                        .map(|bytes| bytes / (1024 * 1024 * 1024))
                } else {
                    None
                }
            })
            .unwrap_or(0);

        // Detect disk type (SSD/HDD) from rotational flag
        let disk_type = std::fs::read_dir("/sys/block")
            .ok()
            .and_then(|entries| {
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if name.starts_with("sd") || name.starts_with("nvme") || name.starts_with("vd") || name.starts_with("mmcblk") {
                        let rotational_path = entry.path().join("queue").join("rotational");
                        if let Ok(content) = std::fs::read_to_string(&rotational_path) {
                            let val = content.trim();
                            return Some(if val == "0" { "SSD".to_string() } else { "HDD".to_string() });
                        }
                    }
                }
                None
            })
            .unwrap_or_else(|| "Unknown".to_string());

        return (disk_size_gb, disk_type);
    }

    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    { (0, "Unknown".to_string()) }
}

/// 检测设备是否支持触摸屏
fn device_detect_touchscreen() -> bool {
    #[cfg(target_os = "windows")]
    {
        let output = std::process::Command::new("powershell")
            .args([
                "-NoProfile", "-NonInteractive", "-Command",
                "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SystemInformation]::IsTouchEnabled"
            ])
            .creation_flags(CREATE_NO_WINDOW)
            .output();

        if let Ok(output) = output {
            if output.status.success() {
                let stdout = String::from_utf8_lossy(&output.stdout).trim().to_lowercase();
                return stdout == "true" || stdout == "True";
            }
        }
    }

    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = std::fs::read_to_string("/proc/bus/input/devices") {
            let low = content.to_lowercase();
            if low.contains("touchscreen") || low.contains("touch screen") {
                return true;
            }
        }
        // Also check /dev/input for event devices with touchscreen in name
        if let Ok(entries) = std::fs::read_dir("/dev/input") {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_lowercase();
                if name.contains("touch") {
                    return true;
                }
            }
        }
        // Check through sysfs
        if let Ok(entries) = std::fs::read_dir("/sys/bus/input/devices") {
            for entry in entries.flatten() {
                let path = entry.path().join("capabilities");
                let abs_path = path.join("abs");
                if abs_path.exists() {
                    if let Ok(entries2) = std::fs::read_dir(entry.path()) {
                        for e2 in entries2.flatten() {
                            let name = e2.file_name().to_string_lossy().to_lowercase();
                            if name.contains("touch") {
                                return true;
                            }
                        }
                    }
                }
            }
        }
    }

    false
}

// ==================== Office 文件转换 ====================

/// 可用 Office 软件类型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OfficeSoftware {
    MicrosoftWord,
    WpsOffice,
    LibreOffice,
    None,
}

/// 检测到的 Office 安装情况与推荐软件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OfficeDetectionResult {
    pub has_word: bool,
    pub has_wps: bool,
    pub has_libreoffice: bool,
    pub recommended: OfficeSoftware,
}

/// Windows 平台：通过注册表检测 Office 安装情况
#[cfg(target_os = "windows")]
fn office_detect_windows() -> OfficeDetectionResult {
    use winreg::RegKey;
    use winreg::enums::*;
    
    let hkcu = RegKey::predef(HKEY_CURRENT_USER);
    let hklm = RegKey::predef(HKEY_LOCAL_MACHINE);
    
    let has_word = office_check_word(&hkcu, &hklm);
    let has_wps = office_check_wps(&hkcu, &hklm);
    let has_libreoffice = office_check_libreoffice(&hkcu, &hklm);
    
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

/// Windows 平台：检测 Microsoft Word 是否安装（多版本注册表路径）
#[cfg(target_os = "windows")]
fn office_check_word(hkcu: &winreg::RegKey, hklm: &winreg::RegKey) -> bool {
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

/// Windows 平台：检测 WPS Office 是否安装（注册表和路径双重检测）
#[cfg(target_os = "windows")]
fn office_check_wps(hkcu: &winreg::RegKey, hklm: &winreg::RegKey) -> bool {
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

/// Windows 平台：检测 LibreOffice 是否安装
#[cfg(target_os = "windows")]
fn office_check_libreoffice(hkcu: &winreg::RegKey, hklm: &winreg::RegKey) -> bool {
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

/// Linux 平台：检查命令是否可用
#[cfg(target_os = "linux")]
fn office_check_command_exists(name: &str) -> bool {
    std::process::Command::new("which")
        .arg(name)
        .output()
        .ok()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Linux 平台：通过 which 命令检测 Office 安装情况
#[cfg(target_os = "linux")]
fn office_detect_linux() -> OfficeDetectionResult {
    let has_libreoffice = office_check_command_exists("soffice") || office_check_command_exists("libreoffice");
    let has_wps = office_check_command_exists("wps") || office_check_command_exists("wpp");
    let has_word = office_check_command_exists("winword") || office_check_command_exists("WINWORD.EXE");

    let recommended = if has_libreoffice {
        OfficeSoftware::LibreOffice
    } else if has_wps {
        OfficeSoftware::WpsOffice
    } else if has_word {
        OfficeSoftware::MicrosoftWord
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

#[tauri::command]
fn office_detect_all() -> OfficeDetectionResult {
    #[cfg(target_os = "windows")]
    {
        office_detect_windows()
    }
    #[cfg(target_os = "linux")]
    {
        office_detect_linux()
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    {
        OfficeDetectionResult {
            has_word: false,
            has_wps: false,
            has_libreoffice: false,
            recommended: OfficeSoftware::None,
        }
    }
}

/// 文档加载运行时检测结果
#[derive(Debug, Serialize)]
pub struct OfficeRuntimeResult {
    pub word: bool,
    pub wps: bool,
    pub libreoffice: bool,
}

/// 检测文档加载时各 Office 软件是否可用（COM/CLI 运行时检测）
#[tauri::command]
fn office_check_runtime() -> OfficeRuntimeResult {
    #[cfg(target_os = "windows")]
    {
        OfficeRuntimeResult {
            word: office_com_check("Word.Application"),
            wps: office_com_check_wps(),
            libreoffice: office_cli_check("soffice", "--version"),
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        OfficeRuntimeResult {
            word: false,
            wps: false,
            libreoffice: office_cli_check("soffice", "--version"),
        }
    }
}

/// Windows：通过 PowerShell 创建 COM 对象来检测 Office COM 接口是否可用
#[cfg(target_os = "windows")]
fn office_com_check(prog_id: &str) -> bool {
    use std::process::Command;
    let script = format!(
        "$ErrorActionPreference = 'Stop'; $o = New-Object -ComObject {p}; $o.Quit(); [System.Runtime.Interopservices.Marshal]::ReleaseComObject($o) | Out-Null",
        p = prog_id
    );
    Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", &script])
        .creation_flags(CREATE_NO_WINDOW)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Windows：检测 WPS Office COM 接口（优先 Kwps.Application，回退 WPS.Application）
#[cfg(target_os = "windows")]
fn office_com_check_wps() -> bool {
    use std::process::Command;
    let script = r#"
        $ErrorActionPreference = 'Stop'
        $o = $null
        try { $o = New-Object -ComObject Kwps.Application } catch { try { $o = New-Object -ComObject WPS.Application } catch {} }
        if ($o) { try { $o.Quit() } catch {}; [System.Runtime.Interopservices.Marshal]::ReleaseComObject($o) | Out-Null; exit 0 } else { exit 1 }
    "#;
    Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", script])
        .creation_flags(CREATE_NO_WINDOW)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// 检测命令行工具是否可用
fn office_cli_check(cmd: &str, arg: &str) -> bool {
    use std::process::Command;
    Command::new(cmd)
        .arg(arg)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// 通过 LibreOffice 命令行将 docx 转换为 PDF（soffice --headless --convert-to pdf）
fn office_convert_libreoffice(docx_path: &str, _pdf_path: &str, cache_dir: &std::path::Path) -> Result<(), String> {
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

/// Tauri IPC 命令：接收 docx 文件字节数据，转换为 PDF 后返回缓存路径
///
/// 自动检测可用 Office 软件并按优先级尝试，使用临时缓存目录减少重复转换
#[tauri::command]
async fn office_convert_docx_to_pdf_bytes(file_data: Vec<u8>, file_name: String, app: tauri::AppHandle) -> Result<String, String> {
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

    let detection = office_detect_all();
    println!("推荐使用: {:?}", detection.recommended);

    let paths = AppPaths::new(&app)?;
    fs::create_dir_all(&paths.cache_dir).map_err(|e| e.to_string())?;

    let folder_name = format!("document_{}", chrono::Local::now().format("%Y%m%d%H%M%S"));
    let doc_cache_dir = paths.cache_dir.join(&folder_name);
    fs::create_dir_all(&doc_cache_dir).map_err(|e| e.to_string())?;

    let ext = std::path::Path::new(&file_name)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("docx");
    let temp_name = format!("temp_{}.{}", chrono::Local::now().format("%Y%m%d%H%M%S"), ext);
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
            #[cfg(target_os = "windows")]
            {
                let r = office_convert_word(&docx_path_str, &pdf_path_str);
                if r.is_err() && detection.has_wps {
                    println!("Word 转换失败，尝试 WPS...");
                    office_convert_wps(&docx_path_str, &pdf_path_str)
                } else if r.is_err() && detection.has_libreoffice {
                    println!("Word 转换失败，尝试 LibreOffice...");
                    office_convert_libreoffice(&docx_path_str, &pdf_path_str, &doc_cache_dir)
                } else {
                    r
                }
            }
            #[cfg(not(target_os = "windows"))]
            {
                Err("Microsoft Word 不支持当前操作系统".to_string())
            }
        }
        OfficeSoftware::WpsOffice => {
            #[cfg(target_os = "windows")]
            {
                let r = office_convert_wps(&docx_path_str, &pdf_path_str);
                if r.is_err() && detection.has_word {
                    println!("WPS 转换失败，尝试 Word...");
                    office_convert_word(&docx_path_str, &pdf_path_str)
                } else if r.is_err() && detection.has_libreoffice {
                    println!("WPS 转换失败，尝试 LibreOffice...");
                    office_convert_libreoffice(&docx_path_str, &pdf_path_str, &doc_cache_dir)
                } else {
                    r
                }
            }
            #[cfg(not(target_os = "windows"))]
            {
                Err("WPS Office 不支持当前操作系统".to_string())
            }
        }
        OfficeSoftware::LibreOffice => {
            office_convert_libreoffice(&docx_path_str, &pdf_path_str, &doc_cache_dir)
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

/// Tauri IPC 命令：将本地 docx 文件路径转换为 PDF
///
/// 自动检测可用 Office 软件，返回缓存目录中的 PDF 路径
#[tauri::command]
async fn office_convert_docx_to_pdf(docx_path: String, app: tauri::AppHandle) -> Result<String, String> {
    use std::fs;

    let detection = office_detect_all();

    let docx = std::path::Path::new(&docx_path);
    let docx_absolute = std::fs::canonicalize(docx)
        .map_err(|e| format!("无法获取文件绝对路径: {}", e))?;

    if !docx_absolute.exists() {
        return Err(format!("文件不存在: {}", docx_absolute.display()));
    }

    println!("转换文件: {}", docx_absolute.display());

    let paths = AppPaths::new(&app)?;
    fs::create_dir_all(&paths.cache_dir).map_err(|e| e.to_string())?;

    let folder_name = format!("document_{}", chrono::Local::now().format("%Y%m%d%H%M%S"));
    let doc_cache_dir = paths.cache_dir.join(&folder_name);
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
            #[cfg(target_os = "windows")]
            {
                office_convert_word(&docx_path_str, &pdf_path_str)?;
            }
            #[cfg(not(target_os = "windows"))]
            {
                return Err("Microsoft Word 不支持当前操作系统".to_string());
            }
        }
        OfficeSoftware::WpsOffice => {
            #[cfg(target_os = "windows")]
            {
                office_convert_wps(&docx_path_str, &pdf_path_str)?;
            }
            #[cfg(not(target_os = "windows"))]
            {
                return Err("WPS Office 不支持当前操作系统".to_string());
            }
        }
        OfficeSoftware::LibreOffice => {
            let output_dir = doc_cache_dir.to_str()
                .ok_or("Invalid cache directory path")?
                .to_string();
            std::process::Command::new("soffice")
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

/// Windows 平台：通过 PowerShell COM 调用 Microsoft Word 将 docx 转为 PDF
#[cfg(target_os = "windows")]
fn office_convert_word(docx_path: &str, pdf_path: &str) -> Result<(), String> {
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

/// Windows 平台：通过 PowerShell COM 调用 WPS Office 将 docx 转为 PDF
///
/// 尝试 Kwps.Application 和 WPS.Application 两个 COM 接口
#[cfg(target_os = "windows")]
fn office_convert_wps(docx_path: &str, pdf_path: &str) -> Result<(), String> {
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

/// Tauri IPC 命令：设置文件类型关联（PDF / DOC / DOCX）
///
/// 平台差异：Windows 通过注册表创建 ProgID，Linux 通过 XDG 规范
#[tauri::command]
async fn filetype_set_icons(app: tauri::AppHandle) -> Result<(), String> {
    #[cfg(target_os = "windows")]
    {
        return filetype_set_icons_windows(app).await;
    }
    #[cfg(target_os = "linux")]
    {
        return filetype_set_icons_linux(&app);
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux")))]
    Err("此功能仅支持 Windows 和 Linux 系统".to_string())
}

/// Linux 平台：通过 XDG 规范注册 ViewStage 为 PDF/DOCX/DOC 默认程序
#[cfg(target_os = "linux")]
fn filetype_set_icons_linux(app: &tauri::AppHandle) -> Result<(), String> {
    use std::process::Command;

    let resource_dir = app.path().resource_dir()
        .map_err(|e| format!("获取资源目录失败: {}", e))?;

    let data_home = std::env::var("XDG_DATA_HOME")
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".to_string());
            format!("{}/.local/share", home)
        });

    let applications_dir = std::path::Path::new(&data_home).join("applications");
    let mime_packages_dir = std::path::Path::new(&data_home).join("mime").join("packages");
    let icons_dir = std::path::Path::new(&data_home).join("icons").join("hicolor").join("scalable").join("apps");

    std::fs::create_dir_all(&applications_dir).map_err(|e| e.to_string())?;
    std::fs::create_dir_all(&mime_packages_dir).map_err(|e| e.to_string())?;
    std::fs::create_dir_all(&icons_dir).map_err(|e| e.to_string())?;

    let exe_path = std::env::current_exe().map_err(|e| e.to_string())?;

    // Copy icon files if available
    for (icon_name, ext) in &[("viewstage", "png"), ("viewstage", "svg")] {
        let src = resource_dir.join("icons").join(format!("{}.{}", icon_name, ext));
        if src.exists() {
            let dst = icons_dir.join(format!("{}.{}", icon_name, ext));
            let _ = std::fs::copy(&src, &dst);
        }
    }

    // Create .desktop file
    let desktop_entry = format!(
        "[Desktop Entry]\n\
         Type=Application\n\
         Name=ViewStage\n\
         Exec={} %f\n\
         MimeType=application/pdf;application/vnd.openxmlformats-officedocument.wordprocessingml.document;application/msword;\n\
         Icon=viewstage\n\
         Categories=Office;Viewer;\n\
         NoDisplay=true\n",
        exe_path.display()
    );
    std::fs::write(applications_dir.join("viewstage.desktop"), &desktop_entry)
        .map_err(|e| format!("写入 .desktop 文件失败: {}", e))?;

    // Create MIME XML
    let mime_xml = r#"<?xml version="1.0"?>
<mime-info xmlns="http://www.freedesktop.org/standards/shared-mime-info">
  <mime-type type="application/pdf">
    <comment>PDF Document</comment>
    <glob pattern="*.pdf"/>
  </mime-type>
  <mime-type type="application/vnd.openxmlformats-officedocument.wordprocessingml.document">
    <comment>Word Document</comment>
    <glob pattern="*.docx"/>
  </mime-type>
  <mime-type type="application/msword">
    <comment>Word 97-2003 Document</comment>
    <glob pattern="*.doc"/>
  </mime-type>
</mime-info>"#;
    std::fs::write(mime_packages_dir.join("viewstage-mime.xml"), mime_xml)
        .map_err(|e| format!("写入 MIME XML 文件失败: {}", e))?;

    // Set as default for PDF using xdg-mime
    let _ = Command::new("xdg-mime")
        .args(["default", "viewstage.desktop", "application/pdf"])
        .output();
    let _ = Command::new("xdg-mime")
        .args(["default", "viewstage.desktop", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"])
        .output();
    let _ = Command::new("xdg-mime")
        .args(["default", "viewstage.desktop", "application/msword"])
        .output();

    // Update desktop and MIME databases
    let _ = Command::new("update-desktop-database")
        .arg(&applications_dir)
        .output();
    let _ = Command::new("update-mime-database")
        .arg(std::path::Path::new(&data_home).join("mime"))
        .output();

    log::info!("Linux 文件关联设置完成");
    Ok(())
}

/// Windows 平台：通过注册表创建 ProgID 和 UserChoice 设置文件关联
///
/// 为 .pdf / .docx / .doc 分别创建 ProgID，注册关联并设置默认程序，最后刷新图标缓存
#[cfg(target_os = "windows")]
async fn filetype_set_icons_windows(app: tauri::AppHandle) -> Result<(), String> {
    use std::process::Command;
    use winreg::RegKey;
    use winreg::enums::*;
    
    let resource_dir = app.path().resource_dir()
        .map_err(|e| format!("获取资源目录失败: {}", e))?;
    
    let pdf_icon = resource_dir.join("icons").join("pdf.ico").display().to_string();
    let word_icon = resource_dir.join("icons").join("word.ico").display().to_string();
    
    let exe_path = std::env::current_exe()
        .map_err(|e| format!("获取可执行文件路径失败: {}", e))?;
    let exe_path_str = exe_path.display().to_string();
    
    let app_id = "SECTL.ViewStage";
    
    log::info!("开始设置文件关联");
    log::info!("可执行文件: {}", exe_path_str);
    log::info!("PDF 图标: {}", pdf_icon);
    log::info!("Word 图标: {}", word_icon);
    
    let hkcu = RegKey::predef(HKEY_CURRENT_USER);
    let classes_key = hkcu.create_subkey("Software\\Classes")
        .map_err(|e| format!("创建 Classes 键失败: {}", e))?.0;
    
    /// 在 HKCU\Software\Classes 下创建 ProgID，包含 DefaultIcon 和 shell/open/command
    fn filetype_create_progid(
        classes_key: &RegKey,
        prog_id: &str,
        icon_path: &str,
        exe_path: &str,
        friendly_name: &str,
    ) -> Result<(), String> {
        let (prog_key, _) = classes_key
            .create_subkey(prog_id)
            .map_err(|e| format!("创建 {} 键失败: {}", prog_id, e))?;
        
        prog_key
            .set_value("", &friendly_name)
            .map_err(|e| format!("设置 {} 友好名称失败: {}", prog_id, e))?;
        
        let (icon_key, _) = prog_key
            .create_subkey("DefaultIcon")
            .map_err(|e| format!("创建 {}\\DefaultIcon 键失败: {}", prog_id, e))?;
        icon_key
            .set_value("", &icon_path)
            .map_err(|e| format!("设置 {} 图标失败: {}", prog_id, e))?;
        
        let (command_key, _) = prog_key
            .create_subkey("shell\\open\\command")
            .map_err(|e| format!("创建 {}\\shell\\open\\command 键失败: {}", prog_id, e))?;
        let command = format!("\"{}\" \"%1\"", exe_path);
        command_key
            .set_value("", &command)
            .map_err(|e| format!("设置 {} 命令失败: {}", prog_id, e))?;
        
        log::info!("ProgID {} 设置完成", prog_id);
        Ok(())
    }
    
    filetype_create_progid(&classes_key, &format!("{}.pdf", app_id), &pdf_icon, &exe_path_str, "ViewStage PDF Document")?;
    filetype_create_progid(&classes_key, &format!("{}.docx", app_id), &word_icon, &exe_path_str, "ViewStage Word Document")?;
    filetype_create_progid(&classes_key, &format!("{}.doc", app_id), &word_icon, &exe_path_str, "ViewStage Word 97-2003 Document")?;
    
    /// 在扩展名下注册关联，设置默认值使图标生效
    fn filetype_create_association(classes_key: &RegKey, ext: &str, prog_id: &str) -> Result<(), String> {
        let (ext_key, _) = classes_key
            .create_subkey(ext)
            .map_err(|e| format!("创建 {} 键失败: {}", ext, e))?;
        
        ext_key
            .set_value("", &prog_id)
            .map_err(|e| format!("设置 {} 默认值失败: {}", ext, e))?;
        
        let (openwith_key, _) = ext_key
            .create_subkey("OpenWithProgids")
            .map_err(|e| format!("创建 {}\\OpenWithProgids 键失败: {}", ext, e))?;
        
        openwith_key
            .set_value(prog_id, &"")
            .map_err(|e| format!("关联 {} 到 {} 失败: {}", ext, prog_id, e))?;
        
        log::info!("文件扩展名 {} 已关联到 {}", ext, prog_id);
        Ok(())
    }
    
    filetype_create_association(&classes_key, ".pdf", &format!("{}.pdf", app_id))?;
    filetype_create_association(&classes_key, ".docx", &format!("{}.docx", app_id))?;
    filetype_create_association(&classes_key, ".doc", &format!("{}.doc", app_id))?;
    
    /// 通过 UserChoice 设置扩展名的默认打开程序（可能需要管理员权限）
    fn filetype_update_default(hkcu: &RegKey, ext: &str, prog_id: &str) -> Result<(), String> {
        let user_choice_path = format!(
            "Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\FileExts\\{}\\UserChoice",
            ext
        );
        
        let result = hkcu.create_subkey(&user_choice_path);
        
        match result {
            Ok((user_choice_key, _)) => {
                match user_choice_key.set_value("ProgId", &prog_id) {
                    Ok(_) => {
                        log::info!("成功设置 {} 为 {} 的默认程序", prog_id, ext);
                        Ok(())
                    }
                    Err(e) => {
                        log::warn!("设置 UserChoice 失败（可能需要管理员权限）: {}", e);
                        Err(format!("设置默认程序失败，请手动在系统设置中设置: {}", e))
                    }
                }
            }
            Err(e) => {
                log::warn!("创建 UserChoice 键失败: {}", e);
                Err(format!("无法设置默认程序，请手动在系统设置中设置: {}", e))
            }
        }
    }
    
    let mut errors = Vec::new();
    
    if let Err(e) = filetype_update_default(&hkcu, ".pdf", &format!("{}.pdf", app_id)) {
        errors.push(e);
    }
    
    if let Err(e) = filetype_update_default(&hkcu, ".docx", &format!("{}.docx", app_id)) {
        errors.push(e);
    }
    
    if let Err(e) = filetype_update_default(&hkcu, ".doc", &format!("{}.doc", app_id)) {
        errors.push(e);
    }
    
    let ps_script = r#"
        $code = @'
        [DllImport("shell32.dll")]
        public static extern void SHChangeNotify(int wEventId, uint uFlags, IntPtr dwItem1, IntPtr dwItem2);
'@
        Add-Type -MemberDefinition $code -Name Shell -Namespace WinAPI
        [WinAPI.Shell]::SHChangeNotify(0x8000000, 0x1000, [IntPtr]::Zero, [IntPtr]::Zero)
        Write-Host "图标缓存已刷新"
    "#;
    
    let output = Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", ps_script])
        .creation_flags(CREATE_NO_WINDOW)
        .output()
        .map_err(|e| format!("刷新图标缓存失败: {}", e))?;
    
    if !output.status.success() {
        log::warn!("刷新图标缓存失败");
    }
    
    if errors.is_empty() {
        log::info!("文件关联设置完成，已设置为默认程序");
        Ok(())
    } else {
        let error_msg = errors.join("\n");
        log::warn!("部分设置失败:\n{}", error_msg);
        Err(error_msg)
    }
}

/// 卸载时清理文件关联：删除注册的 ProgID、OpenWithProgids、UserChoice
#[cfg(target_os = "windows")]
pub fn filetype_delete_icons_windows_sync() -> Result<(), String> {
    use winreg::RegKey;
    use winreg::enums::*;

    let hkcu = RegKey::predef(HKEY_CURRENT_USER);
    let classes_path = "Software\\Classes";
    let classes_key = hkcu.create_subkey(classes_path)
        .map_err(|e| format!("打开 Classes 失败: {}", e))?.0;

    let app_id = "SECTL.ViewStage";
    let exts = [".pdf", ".docx", ".doc"];
    let prog_ids: Vec<String> = exts.iter().map(|ext| format!("{}{}", app_id, ext)).collect();

    // 删除 UserChoice (FileExts\.{ext}\UserChoice)
    for ext in &exts {
        let user_choice_path = format!(
            "Software\\Microsoft\\Windows\\CurrentVersion\\Explorer\\FileExts\\{}\\UserChoice",
            ext
        );
        match hkcu.delete_subkey_all(&user_choice_path) {
            Ok(_) => log::info!("卸载清理: 已删除 {}\\UserChoice", ext),
            Err(ref e) if e.kind() == std::io::ErrorKind::NotFound => {},
            Err(e) => log::warn!("卸载清理: 删除 {}\\UserChoice 失败: {}", ext, e),
        }
    }

    // 删除 ProgID 和 OpenWithProgids 条目
    for (ext, prog_id) in exts.iter().zip(prog_ids.iter()) {
        // 从 OpenWithProgids 中移除引用
        let openwith_path = format!("{}{}\\OpenWithProgids", classes_path, ext);
        match classes_key.open_subkey_with_flags(&openwith_path, KEY_SET_VALUE) {
            Ok(key) => {
                let _ = key.delete_value(prog_id);
                log::info!("卸载清理: 已从 {}\\OpenWithProgids 移除 {}", ext, prog_id);
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::NotFound => {},
            Err(e) => log::warn!("卸载清理: 无法访问 {}\\OpenWithProgids: {}", ext, e),
        }

        // 删除 ProgID 本身
        match hkcu.delete_subkey_all(&format!("{}\\{}", classes_path, prog_id)) {
            Ok(_) => log::info!("卸载清理: 已删除 ProgID {}", prog_id),
            Err(ref e) if e.kind() == std::io::ErrorKind::NotFound => {},
            Err(e) => log::warn!("卸载清理: 删除 ProgID {} 失败: {}", prog_id, e),
        }
    }

    // 刷新图标缓存
    let _ = std::process::Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command", r#"
            Add-Type -MemberDefinition '[DllImport("shell32.dll")] public static extern void SHChangeNotify(int,int,System.IntPtr,System.IntPtr);' -Name Shell -Namespace WinAPI
            [WinAPI.Shell]::SHChangeNotify(0x8000000,0x1000,[IntPtr]::Zero,[IntPtr]::Zero)
        "#])
        .creation_flags(CREATE_NO_WINDOW)
        .output();

    log::info!("卸载清理: 文件关联清理完成");
    Ok(())
}

/// 卸载清理：移除文件关联、计划任务、注册表残留
///
/// 如果检测到 ViewStage-upgrading 标记文件（正在升级中）则跳过，
/// 标记超过 5 分钟视为升级中断残留，仍执行清理。
#[cfg(target_os = "windows")]
pub fn uninstall_cleanup_perform() -> i32 {
    use std::process::Command;

    // ── 升级标记检测 ──────────────────────────────────────
    let marker = std::env::temp_dir().join("ViewStage-upgrading");
    if marker.exists() {
        let fresh = std::fs::metadata(&marker)
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.elapsed().ok())
            .is_some_and(|d| d.as_secs() < 300);
        if fresh {
            log::info!("卸载清理: 检测到升级标记（5分钟内），跳过");
            let _ = std::fs::remove_file(&marker);
            return 0;
        }
        log::info!("卸载清理: 升级标记已过期（>5分钟），视为中断残留，继续清理");
        let _ = std::fs::remove_file(&marker);
    }

    // ── 计划任务清理 ──────────────────────────────────────
    for name in &["ViewStage_MemClean", "memreduct-viewstageTask"] {
        log::info!("卸载清理: 尝试删除计划任务 {}", name);
        let output = Command::new("schtasks")
            .args(["/delete", "/f", "/tn", name])
            .creation_flags(CREATE_NO_WINDOW)
            .output();
        match output {
            Ok(o) if o.status.success() => log::info!("卸载清理: 计划任务 {} 已删除", name),
            Ok(o) => {
                let stderr = String::from_utf8_lossy(&o.stderr);
                log::info!("卸载清理: 计划任务 {} 不存在或已删除 ({})", name, stderr.trim());
            }
            Err(e) => log::warn!("卸载清理: 无法执行 schtasks 删除 {}: {}", name, e),
        }
    }

    // ── 文件关联清理 ──────────────────────────────────────
    log::info!("卸载清理: 开始清理文件关联注册表");
    if let Err(e) = filetype_delete_icons_windows_sync() {
        log::warn!("卸载清理: 文件关联清理失败: {}", e);
    } else {
        log::info!("卸载清理: 文件关联清理完成");
    }

    // ── 应用数据目录清理 ──────────────────────────────────
    if let Some(appdata) = dirs_data_dir() {
        let config_path = appdata.join("config.json");
        if config_path.exists() {
            log::info!("卸载清理: 删除配置文件 {}", config_path.display());
            let _ = std::fs::remove_file(&config_path);
        }
        let log_dir = appdata.join("log");
        if log_dir.exists() {
            log::info!("卸载清理: 删除日志目录 {}", log_dir.display());
            let _ = std::fs::remove_dir_all(&log_dir);
        }
        let models_dir = appdata.join("models");
        if models_dir.exists() {
            log::info!("卸载清理: 删除模型目录 {}", models_dir.display());
            let _ = std::fs::remove_dir_all(&models_dir);
        }
    }

    log::info!("卸载清理: 完成");
    0
}

/// 获取应用数据目录（%APPDATA%/SECTL/ViewStage）
#[cfg(target_os = "windows")]
fn dirs_data_dir() -> Option<std::path::PathBuf> {
    std::env::var("APPDATA").ok().map(|a| {
        let p = std::path::PathBuf::from(a).join("SECTL").join("ViewStage");
        log::info!("应用数据目录: {}", p.display());
        p
    })
}



// ==================== 内存清理子进程（memreduct.exe） ====================

/// 查找 memreduct.exe 路径
#[cfg(target_os = "windows")]
fn find_memreduct_exe() -> Result<std::path::PathBuf, String> {
    let names = ["memreduct-viewstage.exe"];

    // 优先查找同级目录（生产环境）
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(parent) = exe_path.parent() {
            for name in &names {
                let candidate = parent.join(name);
                if candidate.exists() {
                    log::info!("memreduct: 找到子进程: {}", candidate.display());
                    return Ok(candidate);
                }
            }
        }
    }

    // 开发环境：查找项目 memreduct 构建输出
    let dev_paths = [
        "memreduct/bin/64/memreduct-viewstage.exe",
        "memreduct/bin/x64/Release/memreduct-viewstage.exe",
        "memreduct/bin/Release/memreduct-viewstage.exe",
        "memreduct/bin/x64/Debug/memreduct-viewstage.exe",
        "memreduct/bin/Debug/memreduct-viewstage.exe",
    ];
    // 尝试 CWD 下的路径
    if let Ok(cwd) = std::env::current_dir() {
        for rel in &dev_paths {
            let candidate = cwd.join(rel);
            if candidate.exists() {
                log::info!("memreduct: 找到子进程: {}", candidate.display());
                return Ok(candidate);
            }
            // CWD 可能是 src-tauri，尝试父目录
            if let Some(parent) = cwd.parent() {
                let candidate2 = parent.join(rel);
                if candidate2.exists() {
                    log::info!("memreduct: 找到子进程: {}", candidate2.display());
                    return Ok(candidate2);
                }
            }
        }
    }
    // 也尝试直接相对路径（备用）
    for rel in &dev_paths {
        let candidate = std::path::PathBuf::from(rel);
        if candidate.exists() {
            log::info!("memreduct: 找到子进程: {}", candidate.canonicalize().unwrap_or(candidate.clone()).display());
            return Ok(candidate);
        }
    }

    log::error!("memreduct: 找不到子进程");
    Err("找不到 memreduct-viewstage.exe，请确保子进程已构建".to_string())
}

/// 运行 memreduct 子进程并等待完成
#[cfg(target_os = "windows")]
fn run_memreduct(args: &[&str]) -> Result<(), String> {
    use std::process::{Command, Stdio};

    let exe = find_memreduct_exe()?;

    log::info!("memreduct: 执行 {:?} {:?}", exe.file_name().unwrap_or_default(), args);

    let status = Command::new(&exe)
        .args(args)
        .creation_flags(CREATE_NO_WINDOW)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| {
            log::error!("memreduct: 启动失败: {}", e);
            format!("启动 memreduct 失败: {}", e)
        })?
        .wait()
        .map_err(|e| {
            log::error!("memreduct: 等待失败: {}", e);
            format!("等待 memreduct 失败: {}", e)
        })?;

    if status.success() {
        log::info!("memreduct: 执行成功 {:?}", args);
        Ok(())
    } else {
        let code = status.code().unwrap_or(-1);
        log::error!("memreduct: 退出码 {} {:?}", code, args);
        Err(format!("memreduct 退出码 {}", code))
    }
}

/// IPC 命令：立刻清理内存
/// mask 可选，0 或 None 使用默认值 / 配置文件中的 ReductMask2
#[cfg(target_os = "windows")]
#[tauri::command]
async fn memreduct_clean_now(mask: Option<u32>) -> Result<(), String> {
    match mask {
        Some(0) | None => run_memreduct(&["--clean"]),
        Some(m) => run_memreduct(&[&format!("--clean:{}", m)]),
    }
}

/// IPC 命令：设置跳过 UAC（创建计划任务）
#[cfg(target_os = "windows")]
#[tauri::command]
async fn memreduct_setup() -> Result<(), String> {
    run_memreduct(&["--skipuac"])
}

/// IPC 命令：卸载跳过 UAC 计划任务
#[cfg(target_os = "windows")]
#[tauri::command]
async fn memreduct_uninstall() -> Result<(), String> {
    run_memreduct(&["--uninstall"])
}

/// IPC 命令：检查计划任务是否已安装
#[cfg(target_os = "windows")]
#[tauri::command]
async fn memreduct_check_skipuac() -> bool {
    match run_memreduct_raw(&["--check-skipuac"]) {
        Ok(exit_code) => exit_code == 0,
        Err(_) => false,
    }
}

/// IPC 命令：获取当前内存使用率（0-100）
#[cfg(target_os = "windows")]
#[tauri::command]
fn memreduct_get_usage() -> u32 {
    use windows_sys::Win32::System::SystemInformation::GlobalMemoryStatusEx;
    use windows_sys::Win32::System::SystemInformation::MEMORYSTATUSEX;
    unsafe {
        let mut mem = std::mem::zeroed::<MEMORYSTATUSEX>();
        mem.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        if GlobalMemoryStatusEx(&mut mem) != 0 {
            return mem.dwMemoryLoad;
        }
    }
    0
}

#[cfg(not(target_os = "windows"))]
#[tauri::command]
fn memreduct_get_usage() -> u32 { 0 }

#[cfg(not(target_os = "windows"))]
#[tauri::command]
async fn memreduct_clean_now(_mask: Option<u32>) -> Result<(), String> { Err("仅支持 Windows".into()) }

#[cfg(not(target_os = "windows"))]
#[tauri::command]
async fn memreduct_setup() -> Result<(), String> { Err("仅支持 Windows".into()) }

#[cfg(not(target_os = "windows"))]
#[tauri::command]
async fn memreduct_uninstall() -> Result<(), String> { Err("仅支持 Windows".into()) }

#[cfg(not(target_os = "windows"))]
#[tauri::command]
async fn memreduct_check_skipuac() -> bool { false }

/// 运行 memreduct 子进程并获取退出码（不映射为 Result）
#[cfg(target_os = "windows")]
fn run_memreduct_raw(args: &[&str]) -> Result<i32, String> {
    use std::process::{Command, Stdio};

    let exe = find_memreduct_exe()?;

    log::info!("memreduct: 执行 {:?} {:?}", exe.file_name().unwrap_or_default(), args);

    let status = Command::new(&exe)
        .args(args)
        .creation_flags(CREATE_NO_WINDOW)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| {
            log::error!("memreduct: 启动失败: {}", e);
            format!("启动 memreduct 失败: {}", e)
        })?
        .wait()
        .map_err(|e| {
            log::error!("memreduct: 等待失败: {}", e);
            format!("等待 memreduct 失败: {}", e)
        })?;

    let code = status.code().unwrap_or(-1);
    log::info!("memreduct: 退出码 {} {:?}", code, args);

    Ok(code)
}

/// 非 Windows 平台空实现
#[cfg(not(target_os = "windows"))]
fn run_memreduct(_args: &[&str]) -> Result<(), String> {
    Err("内存清理仅在 Windows 上可用".to_string())
}

#[cfg(not(target_os = "windows"))]
fn run_memreduct_raw(_args: &[&str]) -> Result<i32, String> {
    Err("内存清理仅在 Windows 上可用".to_string())
}

// ==================== 展台灯控制命令 ====================

#[cfg(target_os = "windows")]
#[tauri::command]
fn camera_light_on(app: tauri::AppHandle) -> Result<(), String> {
    camera_light::camera_light_set_on(&app)?;
    Ok(())
}

#[cfg(not(target_os = "windows"))]
#[tauri::command]
fn camera_light_on(_app: tauri::AppHandle) -> Result<(), String> {
    Err("展台灯控制仅支持 Windows".to_string())
}

#[cfg(target_os = "windows")]
#[tauri::command]
fn camera_light_off(app: tauri::AppHandle) -> Result<(), String> {
    camera_light::camera_light_set_off(&app)?;
    Ok(())
}

#[cfg(not(target_os = "windows"))]
#[tauri::command]
fn camera_light_off(_app: tauri::AppHandle) -> Result<(), String> {
    Err("展台灯控制仅支持 Windows".to_string())
}

#[cfg(target_os = "windows")]
#[tauri::command]
fn camera_light_get_state() -> Result<(bool, u8), String> {
    camera_light::camera_light_get_state()
}

#[cfg(not(target_os = "windows"))]
#[tauri::command]
fn camera_light_get_state() -> Result<(bool, u8), String> {
    Err("展台灯控制仅支持 Windows".to_string())
}

#[cfg(target_os = "windows")]
#[tauri::command]
fn camera_light_detect() -> bool {
    camera_light::camera_light_detect()
}

#[cfg(not(target_os = "windows"))]
#[tauri::command]
fn camera_light_detect() -> bool {
    false
}

#[tauri::command]
async fn camera_light_detect_and_save(app: tauri::AppHandle) -> bool {
    #[cfg(target_os = "windows")]
    {
        let config_dir = app.path().app_config_dir().unwrap_or_default();
        let device_path = config_dir.join("device.json");

        if device_path.exists() {
            if let Ok(content) = std::fs::read_to_string(&device_path) {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) {
                    if val.get("seewoDetected").and_then(|v| v.as_bool()) == Some(true) {
                        return true;
                    }
                }
            }
            // seewoDetected 非 true（设备曾未检测到），重新检测以应对热插拔
        }

        let detected = camera_light::camera_light_detect();
        let data = serde_json::json!({ "seewoDetected": detected });
        if let Some(parent) = device_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(&device_path, serde_json::to_string_pretty(&data).unwrap_or_default());
        detected
    }
    #[cfg(not(target_os = "windows"))]
    {
        false
    }
}

/// 应用入口函数
///
/// 初始化日志、注册 Tauri 插件和 IPC 命令，配置 OOBE/主窗口启动流程。
/// 首次运行打开 OOBE 引导窗口，非首次运行读取配置设置窗口尺寸并全屏显示。
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn app_init_run() {
    use simplelog::{CombinedLogger, WriteLogger, LevelFilter, Config, TermLogger, TerminalMode, ColorChoice};
    use std::fs::File;
    
    let config_dir = dirs::config_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("SECTL.ViewStage");
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
            let window = match app.get_webview_window("main") {
                Some(w) => w,
                None => {
                    log::error!("启动失败: 无法获取主窗口");
                    return Err("无法获取主窗口".into());
                }
            };
            
            let _ = window.set_decorations(false);
            register_webview_permission_handler(window.as_ref());
            
            let config_dir = match app.path().app_config_dir() {
                Ok(d) => d,
                Err(e) => {
                    log::error!("启动失败: 无法获取配置目录: {}", e);
                    return Err(format!("无法获取配置目录: {}", e).into());
                }
            };
            let config_path = config_dir.join("config.json");
            
            let is_first_run = !config_path.exists();
            
            if is_first_run {
                println!("首次运行，打开 OOBE 界面");
                
                OOBE_ACTIVE.store(true, Ordering::SeqCst);
                
                use tauri::WebviewWindowBuilder;
                
                let oobe_window = match WebviewWindowBuilder::new(
                    app,
                    "oobe",
                    tauri::WebviewUrl::App("oobe.html".into())
                )
                .title("欢迎使用 ViewStage")
                .inner_size(960.0, 540.0)
                .resizable(false)
                .decorations(false)
                .center()
                .build() {
                    Ok(w) => w,
                    Err(e) => {
                        log::error!("启动失败: 无法创建 OOBE 窗口: {}", e);
                        return Err(format!("无法创建 OOBE 窗口: {}", e).into());
                    }
                };
                
                let _ = oobe_window.set_focus();
                register_webview_permission_handler(oobe_window.as_ref());
                
                if let Some(splashscreen) = app.get_webview_window("splashscreen") {
                    let _ = splashscreen.close();
                }
            } else {
                let _ = window.set_fullscreen(true);
                
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
                
            }

            // 手机互联：根据配置决定是否启动服务（需重启生效）
            let phone_server_enabled = std::fs::read_to_string(&config_path)
                .ok()
                .and_then(|c| serde_json::from_str::<serde_json::Value>(&c).ok())
                .and_then(|v| v.get("phoneServerEnabled").and_then(|v| v.as_bool()))
                .unwrap_or(false);

            if phone_server_enabled {
                println!("[setup] phoneServerEnabled=true，启动手机互联服务");
                if let Err(e) = connect::init_server(app.handle()) {
                    log::error!("[setup] 手机互联服务启动失败: {}", e);
                }
            } else {
                println!("[setup] phoneServerEnabled=false，跳过手机互联服务");
            }

            Ok(())
        })
        // 注册所有 Tauri IPC 命令
        .invoke_handler(tauri::generate_handler![
            dir_fetch_cache, 
            cache_fetch_size,
            cache_delete_all,
            cache_delete_doc_annotations,
            cache_validate_auto_clear,
            dir_fetch_config, 
            dir_fetch_log,
            dir_fetch_pictures_viewstage,
            dir_fetch_theme,
            theme_list_user,
            theme_delete,
            theme_import_vst,
            theme_get_preview,
            image_update_rotation,
            image_save_file,
            window_show_settings,
            mirror_update_state,
            mirror_fetch_state,
            app_fetch_version,
            app_fetch_platform,
            update_fetch_check,
            update_download_file,
            update_download_cancel,
            update_install_release,
            settings_fetch_all,
            settings_save_all,
            settings_delete_all,
            app_restart_process,
            filetype_validate_pdf_default,
            filetype_validate_word_default,
            window_hide_splashscreen,
            oobe_submit_complete,
            oobe_check_active,
            main_signal_loaded,
            main_check_loaded,
            app_submit_exit,
            office_detect_all,
            office_check_runtime,
            office_convert_docx_to_pdf,
            office_convert_docx_to_pdf_bytes,
            filetype_set_icons,
            device_detect_all,
            get_device_uuid,
            memreduct_clean_now,
            memreduct_setup,
            memreduct_uninstall,
            memreduct_check_skipuac,
            memreduct_get_usage,
            telemetry_http_post,
            telemetry_http_get,
            telemetry_fetch_cpu_gpu,
            camera_light_on,
            camera_light_off,
            camera_light_get_state,
            camera_light_detect,
            camera_light_detect_and_save,
            connect::phone_server_status,
            connect::phone_server_start,
            connect::phone_server_stop,
            connect::phone_camera_status,
            connect::phone_camera_start,
            connect::phone_camera_stop
        ])
        .run(tauri::generate_context!())
        .unwrap_or_else(|e| {
            log::error!("应用运行失败: {}", e);
            eprintln!("应用运行失败: {}", e);
        });
}
