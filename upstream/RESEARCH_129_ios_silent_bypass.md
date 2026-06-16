# RESEARCH — `@activetheory/ios-silent-bypass` · xiaodi #129

## iOS 静音开关绕过的技术原理

iOS 的静音拨片（Silent Switch / Action Button）会拦截系统音频输出，但 **`HTMLAudioElement` 与 `AudioContext` 遵循不同的音频路由策略**：通过 `<audio>` 标签播放的媒体音频在用户主动触发后可绕过静音开关，而纯 WebAudio 合成声音则会被压制。该库正是利用这一系统级差异实现绕过：先用 `<audio>` 标签"激活"设备音频会话，再让 AudioContext 搭便车输出。

## Web Audio API AudioContext 解锁策略

库在构造函数中立即向 `window` 注册 9 种用户交互事件（`click`、`touchend`、`keydown` 等），全部使用 `{ capture: true, passive: true }` 捕获阶段监听，确保在事件冒泡前抢先响应。一旦捕获到任意手势，即在**用户手势调用栈内**同步触发音频播放请求，满足 iOS WebKit 对"用户激活（User Activation）"的要求，从而解锁被挂起的 AudioContext。

## 空音频播放 Trick 的实现细节

`#createAudioData()` 用 `ArrayBuffer`（10 字节）手工构造一段 **48 kHz、单声道、时长约 7 帧的最小合法 WAV** 文件，其 PCM 采样值均为 `0x80`（静默中值），编码为 `data:audio/wav;base64,…` 内联 URI，体积极小、无网络请求。`<audio>` 元素额外设置 `x-webkit-airplay="deny"` 阻止 AirPlay 路由干扰，`loop=true` 保持音频会话持续活跃。`play()` 返回 Promise：resolved 则将内部状态切为 `allowed`，rejected 则清理元素并重置，等待下次手势重试。

## 在 activetheory.net 音效播放中的角色

activetheory.net 是以沉浸式音效著称的创意技术工作室，其网站体验大量依赖 Web Audio API 实时合成音景。移动端用户因静音拨片导致音效哑火是典型痛点。该库作为**前置解锁层**：在任何用户交互后静默播放无声 WAV，将 iOS 音频会话切换至媒体模式，随后网站的 AudioContext 节点图即可正常输出，无需用户手动关闭静音开关，保障了沉浸体验的一致性。