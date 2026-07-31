# RTX 3060 Ti 服务器部署

适用设备：NVIDIA GeForce RTX 3060 Ti 8 GB，Linux，NVIDIA Driver
595.84。`nvidia-smi` 显示的 `CUDA Version: 13.2` 是驱动能够支持的最高 CUDA
版本，并不表示必须安装 CUDA 13.2 版 PyTorch。

## 1. 安装

项目要求 Python 3.10–3.12、FFmpeg（含 `libx264`）和 Docker Compose。

```bash
cd /部署目录/rtsp
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

595.84 驱动可以向后运行 PyTorch 官方 CUDA 12.8 构建。无需为了运行本项目
额外安装完整 CUDA Toolkit；PyTorch wheel 会携带所需的 CUDA 运行时库。

## 2. 部署前自检

```bash
source .venv/bin/activate
python -m rtsp_annotator.runtime_check --device cuda:0
```

必须看到：

```text
Device: cuda:0 (NVIDIA GeForce RTX 3060 Ti)
设备计算自检: 通过
libx264: 可用
结论: 基础运行环境通过
```

如果显示 `PyTorch CUDA runtime: None` 或“检测不到可用 CUDA”，通常是误装了
CPU 版 PyTorch。重新执行上面的 `cu128` 安装命令。

使用实际模型测预热后的平均/P95 推理耗时：

```bash
python -m rtsp_annotator.runtime_check \
  --device cuda:0 \
  --model './models/best.pt' \
  --imgsz 640 \
  --half \
  --iterations 50
```

这个数字不含拉流、画框、编码和播放器缓存，但可以直接判断模型是否有足够
算力余量：25 FPS 每帧预算是 40 ms，建议模型平均耗时小于 30 ms。

## 3. 启动

先启动项目附带的 RTSP Server：

```bash
docker compose up -d
docker compose ps
```

再启动单路识别。`--classes 0` 仅适用于人物类别 ID 确实为 0 的模型：

```bash
source .venv/bin/activate
python -m rtsp_annotator \
  --input 'rtsp://用户名:密码@摄像头IP:554/原流路径' \
  --model './models/best.pt' \
  --output 'rtsp://127.0.0.1:8554/detected' \
  --device cuda:0 \
  --half \
  --imgsz 640 \
  --classes 0
```

低延迟播放：

```bash
ffplay \
  -rtsp_transport tcp \
  -fflags nobuffer \
  -flags low_delay \
  -framedrop \
  'rtsp://服务器IP:8554/detected'
```

## 4. 这台设备的预期效果

以下是单路 1920×1080、25 FPS、`imgsz=640`、检测模型、FP16 的工程估算，
不是尚未在该服务器实测的承诺值：

| 模型档位 | 预期检测更新率 | 预期平均推理 | 建议 |
|---|---:|---:|---|
| nano | 25 FPS | 约 4–10 ms | 最低延迟，余量最大 |
| small | 25 FPS | 约 7–15 ms | 单路人物检测首选 |
| medium | 25 FPS 或接近 | 约 12–25 ms | 精度优先，需现场压测 |
| large/xlarge | 不预先承诺逐帧 25 FPS | 约 20–50+ ms | 先实测，再决定 |

GPU 推理只是总延迟的一部分。使用低缓存 `ffplay`、摄像头与服务器同一局域网
且摄像头 GOP 合理时，预计：

- 程序内部帧龄约 30–90 ms；
- 服务器到低缓存播放器的新增端到端延迟约 100–250 ms；
- 包含摄像头自身编码后，实际“现场动作到屏幕”通常约 150–500 ms。

VLC 默认缓存、跨网、Wi-Fi、H.265/B 帧或摄像头长 GOP 可能把延迟增加到
0.5–2 秒以上。项目采用“最新帧覆盖”，模型偶尔跟不上时会丢弃旧帧，因此
不会越跑越延迟；但检测框更新率会随推理 FPS 降低。

独立CLI输出H.264默认使用CPU `libx264 ultrafast`；HTTP API的CUDA模板默认
使用RTX 3060 Ti的`h264_nvenc/p4`。B460M-HDV只是主板型号，无法
判断 CPU 性能；单路 1080p/25 FPS 通常可行，但必须用 `lscpu` 确认具体 CPU，
并以运行日志中的 `publish FPS` 验收。建议标准：

```text
capture FPS   ≥ 源流 FPS 的 95%
inference FPS ≥ 源流 FPS 的 95%
publish FPS   ≥ 源流 FPS 的 95%
平均推理      < 30 ms
内部帧龄      < 100 ms
```

观察资源：

```bash
watch -n 1 nvidia-smi
```

启动后首轮推理会有模型加载和 CUDA 预热，不计入稳定运行延迟。用日志连续观察
至少 10 分钟，再以实际 RTSP、模型和播放器完成最终验收。

## 5. 兼容性依据

- [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)
- [NVIDIA CUDA 驱动兼容说明](https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html)
- [Ultralytics YOLO26 性能表](https://docs.ultralytics.com/models/yolo26)
