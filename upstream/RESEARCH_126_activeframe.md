이제 충분한 정보가 확보되었습니다. RESEARCH 문서를 작성합니다.# RESEARCH — ActiveFrame 动画帧管理器
**xiaodi #126 · Active Theory `activetheory/activeframe`**

---

## 核心定位：并非 rAF 包装器，而是 Web Codecs 拉取式解码器

ActiveFrame 由两层组成：

**`af.js`（Node.js 构建管线）**
CLI 工具，接受任意视频，经 `ffmpeg-static`（libx264 / libx265）重编码，再由 `mp4box` 解封装，将裸 NAL 编码单元顺序打包成单一 `.af` 二进制文件。文件尾部 4 字节（LE uint32）记录 JSON manifest 的起始偏移；manifest 包含 codec、fps、totalFrames、每帧 `{o, l, t, ty, i}`（偏移/长度/时间戳/类型/序号）及 base64 codec description。Meridian 样本命令：`node af.js meridian.mp4 out.af 1920 h265 5 23`，GOP=5、CRF=23。

**`ActiveFrame.js`（浏览器运行时）**
使用 **Web Codecs API `VideoDecoder`**，无 `<video>` 元素，无第三方 JS 解封装。核心方法 `setFrame(index)` 实现 GOP 感知随机访问：若目标帧为 delta 帧，向前遍历找最近 keyframe，串行 decode key→delta 链；若为顺序 +1 帧则直接送 delta chunk。输出通过 `process(frame)` 回调暴露给调用方，`frame.close()` 后立即释放。

**与原生 rAF 的区别**
库内部**不含任何 `requestAnimationFrame`**，也无暂停/恢复、优先级队列或帧率节流机制。调度完全由调用方驱动（demo 中为 `scroll` 事件），属于**拉取模型（pull）**，而非 rAF 的推送模型（push）。帧率上限由解码器吞吐量与调用频率共同决定；GOP 大小（默认 5）控制随机跳帧的最大解码代价。

**在 astro-svgfigure epoch loop 中的用法**
在 epoch ticker 的 rAF 回调内，将归一化进度映射为帧索引：

```js
// epoch ∈ [0, 1]，在你自己的 rAF / ticker 中调用
activeFrame.setFrame(Math.round(epoch * (activeFrame.manifest.totalFrames - 1)));
```

`setFrame` 是幂等的（重复同一帧提前返回），多个实例共享 `cacheActiveFrameList` Map，可实现横/竖版视频帧同步。

---

> **结论**：ActiveFrame 是以 Web Codecs 为引擎的**帧精确视频纹理层**，不替代 rAF，而是作为其内部 payload，适合 scroll-scrub、WebGL 序列帧、epoch 驱动的逐帧动画等场景。