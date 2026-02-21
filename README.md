<div align="center">
   <img src="https://github.com/ospneam/ViewStage/blob/main/src-tauri/icons/Square1024x1024Logo.png" width=15%>
   <h1>ViewStage</h1>
   <p>一个基于 Tauri 构建的摄像头及PDF展台应用，采用原生 HTML、CSS 和 JavaScript 开发，提供简洁高效的课堂及其他用途的全屏展台。</p>
</div>

>[!IMPORTANT]
>本项目目前还在开发中，若需要长时间允许在教学环境中，请先进行测试
>若出现问题，欢迎到issues中提交，我看到后会火速进行修复

## 技术栈

- **前端**：Vanilla HTML 5 + CSS 3 + JavaScript（无框架依赖）+ Rust wasm
- **后端**：Rust + Tauri
- **构建工具**：Cargo

> [!IMPORTANT]
>这个应用部分使用了Tare编写与进行性能优化、检测
>>若您介意或排斥，请无视次项目，感谢(❁´◡`❁)
>本项目前尚未开发完，若出现性能问题请反馈，若能够优化，我一定会尽快进行优化

## 功能特点

- 🚀 **轻量高性能**：基于 Tauri 框架，应用体积小、启动快、内存占用低
- 🎨 **原生 UI 体验**：使用系统原生 WebView，提供流畅的桌面应用交互
- 📦 **开箱即用**：无复杂前端框架依赖，代码结构清晰易维护

## 运行条件
>本项目目前着重适配了16：9的屏幕，对其他比例的屏幕并没有进行检测，若您使用的屏幕比例非16：9，可能会导致显示异常
>>如果你的屏幕为非16：9的屏幕，且允许此应用出现了异常，你可以尝试将分辨率改成16：9，或在issues中提出问题
>>在提出问题中请包含你的屏幕比例、屏幕分辨率、应用版本、操作系统版本等信息，以便我能够更好地帮助你

- 1：我们强烈推荐你在Windows10以上的系统使用本应用，若你使用的是Windows7或Windows8，可能会导致显示异常
>如果你为非Windows10以上的系统，若出现问题，请不要提交到issues中，谢谢
>>如果你的班班通CPU型号太老了，请尝试使用其他应用，本应用可能在您的设备上无法流畅运行（ps：我已经很努力优化了）
>>若您的设备为2018年以后的设备，建议使用最新版Windows10以上的系统（推荐使用Windows10LTSC版本）

- 2:在系统中必须安装有WebView2运行时，若未安装，请前往[Microsoft Edge WebView2 运行时](https://developer.microsoft.com/en-us/microsoft-edge/webview2/#download-section)下载并安装最新版

>若您的设备无法升级Windows10以上的系统，您依旧可以尝试安装本应用，并安装WebView2运行时，但出现未知问题时，我无法提供技术支持


## 快速开始
 - 前往release下载最新安装包
 - 下一步、下一步~~~~ 直到安装完成
 - 尽情使用吧
>项目目前只支持Windows，因Macos与Linux在教学中应用不是很多，故暂时先不做适配
>若后续MacOS或Linux上有需求，我会考虑进行适配

## 开发环境要求
- Node.js（推荐 20.x）
- Rust（稳定版）
- Tauri CLI

>这是本人第一个开发的项目，若有不足之处可以到issues中反馈提交，如果看到了我会进行回复
>如果你也有意向参与开发，欢迎到项目中提交PR，只要是对此项目有益的代码，我都会进行处理合并
>若你有意参与开发，但是在某处不知道如何操作或实现，可以联系邮箱：[rewreqw1@outlook.com]

## 项目进展
项目还在施工，很多功能可能还没有写完，很多图标都是随便搞的，后面还会换
 - ✅批注
 - ✅触控
 - ✅PDF打开
 - ✅允许PDF以ViewStage为默认打开程序
 - ✅在拍摄时实现文档增强（ps：实时实现增强太卡了）
 - ✅允许从外部导入图片
 - ✅图标完善（采用icons.bootcss.com中的图标，在此表示感谢(*^_^*)）
 - ❎设置
 - ❎笔颜色
 - ❓更多功能(欢迎到issues提交(*^_^*))

## 许可证
本项目采用开源许可证，详见 [LICENSE](https://github.com/ospneam/ViewStage/blob/main/LICENSE) 文件。


