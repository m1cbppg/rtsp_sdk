# RTSP + YOLO 实测验证报告

验证日期：2026-07-26

## 结论

这套架构已经实际跑通以下完整链路：

```text
1080p/25 FPS H.264 RTSP 输入
→ PyAV 低缓冲解码
→ Ultralytics YOLO 实时推理并画框
→ FFmpeg libx264 低延迟编码
→ MediaMTX
→ 新的 1080p/25 FPS H.264 RTSP 输出
```

原验证版本已目视确认人物框、类别名称和置信度；当前版本根据需求已将
框上文字改为中文并移除置信度。程序不传`--classes`时会显示模型识别出的
全部类别；在测试图片上实际得到
`4 person + 1 bus`。传 `--classes 0` 时才只保留 COCO 模型中的人物。

## 测试条件

- 主机：Apple M1 Pro，8 核 CPU、14 核 GPU、16 GB 统一内存
- Python 3.12.10
- PyTorch 2.13.0
- Ultralytics 8.4.106
- PyAV 16.1.0
- FFmpeg 7.1.1，`libx264`
- MediaMTX 1.19.2 官方独立程序
- 输入与输出：RTSP over TCP，1920×1080，25 FPS，H.264
- 推理尺寸：640
- 测试模型：官方 YOLO26n、YOLO26s

端到端延迟通过在输入流中加入逐帧移动的标记，同时以低缓存方式读取输入
和输出并比较标记位置测得。测试在同一台机器的回环网络上进行，因此不包含
真实摄像头编码、局域网抖动和播放器默认缓存。

## 实测结果

| 模型与设备 | 输出 FPS | 检测更新 FPS | 平均推理 | 端到端中位数 | 端到端 P95 |
|---|---:|---:|---:|---:|---:|
| YOLO26n / Apple MPS | 25 | 25 | 约 20 ms | 116.7 ms | 125.0 ms |
| YOLO26s / Apple MPS | 25 | 25 | 约 23–24 ms | 116.7 ms | 125.0 ms |
| YOLO26n / CPU-only | 25 | 约 22–24 | 约 42–46 ms | 125.0 ms | 166.7 ms |

YOLO26s 的独立 CPU 推理约 84.8 ms/帧，即约 11.8 FPS，因此没有把它列为
CPU-only 的 25 FPS 方案。程序在推理跟不上时仍按 25 FPS 发布最近完成的
识别画面，并覆盖过时的输入帧，所以延迟不会不断累积。

输出流已经用 `ffprobe` 确认为：

```text
codec_name=h264
width=1920
height=1080
pix_fmt=yuv420p
r_frame_rate=25/1
```

另外完成了以下检查：

- 输入流中断并恢复后，拉流线程能自动重连；
- 输入分辨率从 720p 变为 1080p 后，发布器自动按新分辨率重启；
- FFmpeg 推流管道使用非阻塞写入，停机不会长期卡死；
- Docker Compose 配置可成功解析；
- 39 个单元测试全部通过，源码编译检查通过。

## 最低硬件判断

这里必须区分“能运行”和“稳定低延迟 25 FPS”。

### 已实测通过的下限

单路 1080p/25 FPS、`imgsz=640`、YOLO nano/small：

- Apple M1 Pro（14 核 GPU）；
- 16 GB 内存；
- 使用 `--device mps`。

这套配置对 YOLO26n 和 YOLO26s 都能保持逐帧 25 FPS，并有一定推理余量。

### CPU-only

- 8 GB 内存是安装和运行下限，16 GB 更稳妥；
- 现代 6–8 核 CPU；
- 仅建议 nano 级模型，或者把 `--imgsz` 降到 512/416；
- 适合约 10–20 检测 FPS，不应承诺 1080p 每帧 25 FPS。

本机的 8 核 M1 Pro CPU 跑 nano 已只有约 22–24 检测 FPS，说明把更弱的
四核 CPU 作为“低延迟逐帧 25 FPS”最低配置并不可靠。

### 建议采购配置

若目标明确是单路 1080p/25 FPS、nano/small 模型、长期稳定运行：

- 现代 6 核或更高 CPU；
- 16 GB 系统内存；
- NVIDIA Turing 或更新架构、至少 6 GB 显存的 CUDA GPU；
- 有条件时把 `.pt` 导出为 TensorRT，并使用硬件 H.264 编码。

如果是低功耗边缘设备，可评估 Jetson Orin Nano Super 8 GB；但应使用
TensorRT 和硬件编码后再验收，不能把本报告中的 macOS/PyTorch 数字直接
套用到 Jetson。NVIDIA 官方规格为 67 INT8 TOPS、8 GB LPDDR5。

如果是多路摄像头、YOLO medium/large、实例分割或姿态模型，应从 8–16 GB
显存起步。T4 具备 16 GB GDDR6；Ultralytics 官方 YOLO26 TensorRT/T4
数据中，n/s/m/l/x 的纯模型延迟分别约为 1.7/2.5/4.7/6.2/11.8 ms。
这些是纯模型基准，不包含拉流、解码、画框、编码与网络延迟。

官方参考：

- [Ultralytics YOLO26 性能表](https://docs.ultralytics.com/models/yolo26)
- [Ultralytics 模型格式与基准说明](https://docs.ultralytics.com/modes/benchmark)
- [PyTorch MPS 后端](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [Jetson Orin Nano Super 规格](https://www.nvidia.com/en-in/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/)
- [NVIDIA T4 规格](https://www.nvidia.com/en-gb/data-center/tesla-t4/)

## 尚不能替代现场验收的部分

当前结果证明了软件架构和实现可以工作，但不能在没有用户实际输入的情况下
保证任意 `xx.pt` 和任意监控流都得到相同数字，原因包括：

- 模型 n/s/m/l/x 规模、任务类型和训练质量不同；
- 摄像头可能使用 H.265、B 帧、超长 GOP，或自身缓存较大；
- 网络丢包、Wi-Fi、跨网传输会增加延迟；
- VLC 等播放器的默认缓存可能额外增加数百毫秒到数秒；
- 本程序当前输出不复制音频。

拿到实际模型和 RTSP 后，建议以这些条件验收单路 25 FPS 目标：

- 平均推理小于 30 ms，P95 小于 40 ms；
- `inference FPS` 和 `publish FPS` 均不低于源 FPS 的 95%；
- 低缓存播放器端到端 P95 小于 250 ms；
- 连续运行至少 30 分钟，无推流重启，网络短断后能恢复；
- 在实际白天、夜间、遮挡和远距离画面上检查漏检与误检。
