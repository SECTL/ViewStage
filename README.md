<div align="center">
   <img src="https://github.com/ospneam/ViewStage/blob/main/src-tauri/icons/Square1024x1024Logo.png" width=15%>
   <h1>ViewStage</h1>
   <p>基于 <strong>Tauri v2</strong> 构建的桌面实物展台与演示批注应用，适用于教学、会议、产品展示等多种场景。</p>
   <p>无需 Node.js 构建前端 — 原生 ES Module 直接加载，零 bundler 依赖。</p>
</div>

<p align="center">
    <img src="https://img.shields.io/github/v/release/SECTL/ViewStage" alt="版本">
    <img src="https://img.shields.io/badge/Tauri-2-ffc131.svg" alt="Tauri v2">
    <img src="https://img.shields.io/github/license/SECTL/ViewStage" alt="许可证">
</p>

>由于开发者高三了，所以开发计划暂停，后续仅进行功能修复，与2027年6月后继续
>截至目前，项目正在全力推进转向纯rust+iced开发，旨在提供更好的性能，此项目将主要进行功能修复

## 功能概览

| 类别 | 功能 |
|------|------|
| 📷 **摄像头** | 实时画面采集，多设备/分辨率切换，旋转/镜像/黑白/亮度对比度调节，降噪，最小化自动关开 |
| 🖼 **图片** | 导入本地图片，缩略图侧边栏，旋转/居中/删除 |
| 📄 **文档** | PDF.js 渲染 PDF；PowerShell COM 自动转换 Word（Office/WPS/LibreOffice）为 PDF；系统文件关联 |
| ✏️ **批注** | 移动/批注/橡皮擦三模式，压感笔锋，Catmull-Rom 平滑，自定义颜色粗细，笔画分割擦除 |
| ↩️ **撤销** | Command 模式撤销重做，上限 50 步，超限自动压缩快照 |
| 🎨 **主题** | 深色/浅色双内置主题，支持 .vst 自定义导入，实时切换 |
| 🌐 **国际化** | 简体中文、繁体中文、英文、日语、韩语、法语、德语、西班牙语、俄语（9 种语言） |
| ⚙️ **设置** | 画布/画笔/信号源/文件关联/缓存/日志管理，设置导入导出 JSON |
| 🔄 **更新** | 检查 GitHub Release，多镜像下载，进度条，自动安装 |
| 📸 **截图** | 画布合并导出 PNG，摄像头帧捕获，源切换自动保存批注快照 |
| 🧩 **源管理** | 摄像头/图片/文档三源统一管理，缩放状态与批注自动保存恢复 |
| 🖥 **渲染** | 双图层 + 平铺渲染 + 动态 DPR + 四叉树索引 + 自适应帧率 |
| 📱 **手机互联** | LAN 局域网连接，UDP 自动发现，远程控制（翻页/批注/截图等 18 种操作），文件上传，手机摄像头推流（JPEG + H.264） |
| 💡 **展台灯** | Seewo 文档摄像头灯光控制，USB HID 协议，支持 12 款型号，自动检测连接状态 |
| 🖤 **黑板** | 全屏黑板模式，多页绘图，独立绘图引擎 |
| ✋ **手掌擦除** | 多触点手掌识别，自动切换橡皮擦模式 |
| 🚀 **其他** | OOBE 首次引导、Splashscreen 启动屏、无框全屏窗口 |

## 技术栈

| 层级 | 技术 |
|------|------|
| **前端** | Vanilla HTML5 + CSS3 + JavaScript (ES Module) — 无 bundler、无 Node.js 构建 |
| **后端** | Rust |
| **桌面框架** | Tauri v2 |
| **PDF 渲染** | PDF.js |
| **Word 转换** | PowerShell COM 互操作 (Office/WPS/LibreOffice) |
| **日志** | simplelog |


> 项目无需 `npm` / `package.json`，前端直接以 ES Module 方式加载。

## 运行条件

### 系统要求

- **操作系统**：Windows 10 或更高版本、Linux（x86_64）
- **运行时**：WebView2（[下载地址](https://developer.microsoft.com/en-us/microsoft-edge/webview2/#download-section)）

### 硬件要求

- **摄像头**：用于展台功能（可选）
- **内存**：建议 4GB 以上
- **存储**：约 100MB（含运行时）

### 可选依赖

- **Microsoft Office** / **WPS Office** / **LibreOffice**：用于 Word 文档转换

## 开发环境要求

- **Rust**（稳定版，[安装](https://rustup.rs/)）
- **Tauri CLI**：`cargo install tauri-cli --locked`
- **Cargo**（随 Rust 一起安装）

## 构建与运行

```bash
# 开发模式（热重载）
cargo tauri dev

# 生产构建
cargo tauri build
```

CI 触发器：推送 `v*` 标签或手动触发。

## 许可证

本项目采用开源许可证，详见 [LICENSE](LICENSE) 文件。使用本仓库代码构建的应用也必须开源。

## 致谢

### 核心框架

- [Tauri](https://tauri.app/) — 构建更安全、更轻量的桌面应用
- [Tokio](https://tokio.rs/) — Rust 异步运行时

### 前端库

- [PDF.js](https://mozilla.github.io/pdf.js/) — Mozilla 的 PDF 渲染库

### Rust 库

- [image](https://github.com/image-rs/image) — 图像编解码与处理
- [imageproc](https://github.com/image-rs/imageproc) — 图像处理算法
- [serde](https://serde.rs/) — 序列化框架
- [rayon](https://github.com/rayon-rs/rayon) — 数据并行计算
- [chrono](https://github.com/chronotope/chrono) — 日期时间库
- [reqwest](https://github.com/seanmonstar/reqwest) — HTTP 客户端
- [winreg](https://github.com/gentoo90/winreg-rs) — Windows 注册表操作（文件关联）
- [simplelog](https://github.com/dermesser/simplelog) — 日志记录
- [windows-sys](https://github.com/microsoft/windows-rs) — Windows API 绑定（内存清理 FFI）
- [axum](https://github.com/tokio-rs/axum) — HTTP 服务器框架（手机互联）
- [hidapi](https://github.com/ruabmbua/hidapi-rs) — USB HID 设备通信（展台灯控制）
- [tower-http](https://github.com/tower-rs/tower-http) — HTTP 中间件（CORS）
- [local-ip-address](https://github.com/EstebanBorai/local-ip-address) — 本机 IP 地址检测
- [sysinfo](https://github.com/GuillaumeGomez/sysinfo) — 系统信息获取

### Tauri 插件

- [tauri-plugin-opener](https://github.com/tauri-apps/plugins-workspace) — 文件打开
- [tauri-plugin-fs](https://github.com/tauri-apps/plugins-workspace) — 文件系统
- [tauri-plugin-dialog](https://github.com/tauri-apps/plugins-workspace) — 对话框
- [tauri-plugin-single-instance](https://github.com/tauri-apps/plugins-workspace) — 单实例

感谢所有开源社区的贡献！

## 更新日志

完整更新日志请查看 [GitHub Releases](https://github.com/SECTL/ViewStage/releases)。
