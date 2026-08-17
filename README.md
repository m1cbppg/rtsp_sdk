# RTSP + YOLO 实时识别转推

项目当前完成度、服务器部署状态和各业务场景的完整调用示例见
[`PROJECT_STATUS_AND_SCENARIOS.md`](PROJECT_STATUS_AND_SCENARIOS.md)。

输入一个 RTSP 监控流和一个本地 Ultralytics YOLO 模型（`xx.pt`），程序会持续读取最新画面、运行识别、画框，并把结果发布成新的 RTSP 流。

网络受限的NVIDIA Ubuntu主机可以在联网机器预先制作`linux/amd64`离线CUDA
镜像包，模型和全部Python依赖都会包含在包内。操作见
[OFFLINE_CUDA.md](OFFLINE_CUDA.md)。

通过机房公网IP和端口映射提供带认证的外网RTSP播放，见
[PUBLIC_RTSP.md](PUBLIC_RTSP.md)。

通过HTTP创建、查询和停止识别流，并获得独立的处理后RTSP地址，见
[HTTP_API.md](HTTP_API.md)。

按`stream_id`实时查看播放、卡顿、断流和马赛克风险日志，见
[STREAM_LOGGING.md](STREAM_LOGGING.md)。

四路及以上 NVIDIA 部署建议使用新 DeepStream/TensorRT 后端。它使用
NVDEC、TensorRT、NvDCF、GPU OSD 和 NVENC 的零拷贝管线，每两路固定共用
一个模型实例，完整打包和服务器验证步骤见
[DEEPSTREAM_DEPLOY.md](DEEPSTREAM_DEPLOY.md)。

DeepStream后端支持与白天逻辑隔离的夜间推理配置。默认关闭，启用参数和
验收方法见[NIGHT_VISION.md](NIGHT_VISION.md)。

区域停留、垃圾变化、疑似乱丢垃圾、实时画面提示、事件截图和Webhook见
[EVENT_DETECTION.md](EVENT_DETECTION.md)。该功能使用低帧率可丢帧旁路，
不把垃圾分析耗时串入主推流链路。

固定机位燃气瓶逐个识别和稳定计数见
[GAS_CYLINDER.md](GAS_CYLINDER.md)。该功能使用YOLOE低频丢帧旁路，输入和
输出仍为RTSP，不阻塞主视频转发。

远距离、雨雾监控视角下的高召回船舶框选见
[VESSEL_DETECTION.md](VESSEL_DETECTION.md)。它复用现有预训练模型，不要求
训练新模型，并为不同摄像头提供独立水域ROI、透视分区和固定误报排除区。

数据链路：

```text
监控 RTSP
    │
    ▼
PyAV/FFmpeg 拉流 ───► YOLO 推理与叠框 ──► FFmpeg H.264 低延迟编码
   最新帧覆盖              最新帧覆盖                  │
                                                    ▼
                                           MediaMTX RTSP Server
                                                    │
                                                    ▼
                                  rtsp://服务器:8554/detected
```

“最新帧覆盖”是实时性的关键：如果模型处理速度低于摄像头 FPS，程序会跳过已经过时的帧，而不是排队处理。这样延迟不会随运行时间不断增加。

## 1. 环境

- Python 3.10–3.12（推荐 3.12；不要使用当前目录已有的 Python 3.14 虚拟环境）
- FFmpeg，且包含 H.264 编码器 `libx264`
- 一个本地 YOLO `.pt` 模型
- 一个可接收发布流的 RTSP Server。项目附带 MediaMTX Compose；已有 RTSP Server 时可以不用它

在本目录执行：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

项目直接依赖 PyAV 和 Ultralytics；Ultralytics 会带入与其兼容的 PyTorch、
OpenCV 和 NumPy。NVIDIA 服务器应先通过
[PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)安装与驱动
匹配的 CUDA 版 PyTorch，再安装本项目依赖。

RTX 3060 Ti / Driver 595.84 的完整安装、自检命令和延迟预期见
[SERVER_CUDA.md](SERVER_CUDA.md)。

确认 Apple MPS：

```bash
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

确认 NVIDIA CUDA：

```bash
python -m rtsp_annotator.runtime_check --device cuda:0
```

确认 FFmpeg：

```bash
ffmpeg -hide_banner -encoders | grep libx264
```

## 2. 启动 RTSP Server

先启动 Docker Desktop，再执行：

```bash
docker compose up -d
docker compose logs -f mediamtx
```

MediaMTX 的 RTSP 端口为 `8554`。官方镜像使用主版本标签 `bluenviron/mediamtx:1`，容器内只启用 TCP RTSP，避免 Docker 端口映射影响 RTP/UDP。

如果已经有可接收 RTSP 发布的服务器，直接把它的地址用于 `--output`，无需启动此容器。

## 3. 启动识别转推

假设：

- 原流：`rtsp://admin:password@192.168.1.20:554/Streaming/Channels/101`
- 模型：`./models/best.pt`
- 输出流：`rtsp://127.0.0.1:8554/detected`

运行：

```bash
source .venv/bin/activate
python -m rtsp_annotator \
  --input 'rtsp://admin:password@192.168.1.20:554/Streaming/Channels/101' \
  --model './models/best.pt' \
  --output 'rtsp://127.0.0.1:8554/detected' \
  --classes 0 \
  --device auto
```

`--classes 0` 的前提是模型中的人物类别 ID 为 `0`。COCO 模型通常如此，自训练模型必须以训练数据集的类别顺序为准。不传 `--classes` 就显示模型识别到的所有类别。

先从监控首帧中用鼠标绘制识别区域：

```bash
python -m rtsp_annotator.roi_selector \
  --input 'rtsp://admin:password@192.168.1.20:554/Streaming/Channels/101'
```

鼠标左键添加顶点，右键撤销，`R` 清空，`Enter` 保存，`Esc` 取消。工具会
输出一段 `--roi '...'` 参数。将它加入识别命令：

```bash
python -m rtsp_annotator \
  --input 'rtsp://admin:password@192.168.1.20:554/Streaming/Channels/101' \
  --model './models/best.pt' \
  --output 'rtsp://127.0.0.1:8554/detected' \
  --device mps \
  --roi '0.10,0.15;0.90,0.15;0.85,0.90;0.15,0.90'
```

ROI 坐标是相对画面宽高的比例，范围为 `0` 到 `1`：左上角为
`0,0`，右下角为 `1,1`。至少提供三个点，程序会在输出画面中画出黄色边界，
并只保留检测框中心点位于多边形内部或边界上的目标。不传 `--roi` 时仍识别
全画面。ROI 当前用于结果过滤，不会减少 YOLO 对整张画面的推理计算量。

也可以通过环境变量提供三个必需值，避免把摄像头密码留在 shell 历史里：

```bash
export RTSP_INPUT_URL='rtsp://admin:password@192.168.1.20:554/Streaming/Channels/101'
export RTSP_OUTPUT_URL='rtsp://127.0.0.1:8554/detected'
export YOLO_MODEL_PATH='./models/best.pt'
python -m rtsp_annotator --classes 0
```

程序日志会隐藏 RTSP URL 中的用户名和密码。

## 4. 播放新流

本机低缓存预览：

```bash
ffplay \
  -rtsp_transport tcp \
  -fflags nobuffer \
  -flags low_delay \
  -framedrop \
  'rtsp://127.0.0.1:8554/detected'
```

也可以用 VLC 打开：

```text
rtsp://127.0.0.1:8554/detected
```

局域网其他设备要把 `127.0.0.1` 换成运行 MediaMTX 的服务器 IP，并放通 TCP `8554`。

## 5. 常用参数

```text
--conf 0.25             置信度阈值
--iou 0.45              NMS IoU 阈值
--imgsz 640             推理尺寸；降低可提速
--classes 0             只识别人（类别 ID 取决于模型）
--roi 'x,y;x,y;x,y'     多边形识别区域，坐标范围 0～1
--roi-line-width 3      输出画面中的区域边界宽度
--device auto           自动选择 CUDA > MPS > CPU
--device mps            Apple Silicon GPU
--device cuda           NVIDIA 第一张 GPU
--device cuda:0         NVIDIA 指定 GPU 编号
--device 0              cuda:0 的简写
--device cpu            强制 CPU
--half                  CUDA 使用 FP16；不要用于 MPS/CPU
--output-fps 15          固定输出 FPS；默认跟随源流
--bitrate 2500k         H.264 码率
--encoder libx264       FFmpeg 编码器
--input-transport tcp   拉取摄像头时使用 TCP
--output-transport tcp  向 RTSP Server 发布时使用 TCP
--line-width 3          框线宽度
--no-labels             不显示类别名
--label-map FILE        自训练模型的中文标签映射 JSON
--font FILE             手动指定中文字体；一般无需设置
```

识别框只显示中文类别名称，不显示置信度。官方 COCO 80 类已经内置中文名。
自训练模型如果使用英文类别名，可复制
`config/labels.zh.example.json` 后填写映射，并传入
`--label-map config/labels.zh.json`。映射键支持原类别名或类别 ID；未配置的
英文类别会显示为“类别N”，避免把英文画到视频中。

查看完整说明：

```bash
python -m rtsp_annotator --help
```

### 硬件编码

独立CLI默认使用已经验证的`libx264/ultrafast`。HTTP API的CUDA模板使用专门
适配过低延迟参数的`h264_nvenc/p4`，不是简单替换编码器名称；如果运行环境
缺少NVENC支持，可在JSON配置中回退到`libx264/ultrafast`。

`--device mps/cuda` 只控制 YOLO 推理设备，默认不会改变 H.264 编码器。同一个
Ultralytics `.pt` 模型可以在 Mac MPS 测试后复制到 NVIDIA 服务器，服务器
使用 `--device cuda:0 --half` 启动。程序会在启动日志中打印最终选择的设备
和 GPU 名称；显式指定不可用的 MPS/CUDA 时会直接报错，不会静默退回 CPU。

## 6. 延迟与性能

端到端延迟通常来自：

1. 摄像头自身编码和 GOP；
2. RTSP 网络传输与播放器缓存；
3. 解码；
4. YOLO 推理；
5. H.264 再编码。

推荐按这个顺序调优：

1. 使用较小的 YOLO 模型（如自训练的 n/s 规格）和 GPU；
2. 把 `--imgsz` 从 `640` 降到 `512` 或 `416`，确认精度仍可接受；
3. 摄像头端把 GOP/关键帧间隔设为约 1 秒；
4. 使用上面的低缓存 `ffplay` 参数；
5. 监控日志中的 `capture FPS`、`inference FPS` 和平均推理耗时。

当前 Python 后端已将视频发布与检测线程解耦：当 `inference FPS` 低于源流
FPS 时，视频仍持续发布最新原始帧，并在短时间内跟踪/外推最近一次检测框，
不会再靠重复整张旧画面伪装成 25 FPS。`unique_publish_fps` 是判断画面是否
真正流畅的指标，`duplicate_publish_fps` 应接近 0。

HTTP API模式下，同一个`.pt`按每两路创建一个模型实例；第3路自动创建第2个
实例。参数和输入尺寸兼容的同组两路最新帧会在默认最多2ms的微等待窗口内动态
batch；实例只有一路时不会等待凑batch。每一路的真实等待、推理和画框总耗时
及`model_instance_id`可通过`GET /v1/streams/{stream_id}`中的`metrics`查看。

上述 PyTorch 动态 batch 后端继续保留，适合 Mac/MPS 和兼容回退。RTX 3060 Ti
四路正式部署使用 DeepStream 后端；它不会走 PyAV → NumPy → Python 画框 →
FFmpeg 的 CPU 往返路径。

DeepStream船舶旁路还支持默认关闭的`fishing_risk`纯规则分析：只用固定监控中的
船舶轨迹、禁渔区域和时间表生成疑似捕捞人工复核事件，不需要训练新模型。它可
通过API运行中关闭并退回纯船舶框模式，详见[FISHING_RISK.md](FISHING_RISK.md)。

本项目已经在本地完成 1080p/25 FPS 的真实 RTSP 输入、YOLO 推理、叠框、H.264 编码和 RTSP 转推测试。测试环境、端到端延迟、模型档位与硬件结论见 [VALIDATION.md](VALIDATION.md)。

## 7. 故障定位

### 一直提示“输出发布中断”

确认 MediaMTX 正在监听：

```bash
docker compose ps
docker compose logs --tail=100 mediamtx
```

确认端口：

```bash
nc -vz 127.0.0.1 8554
```

### 拉不到摄像头

先绕过本程序测试原流：

```bash
ffplay -rtsp_transport tcp '原始 RTSP 地址'
```

如果摄像头只支持 UDP，使用 `--input-transport udp`。Docker 内的 MediaMTX 默认只启用了 TCP，所以保持 `--output-transport tcp` 即可。

### 识别很慢

日志里的平均推理时间决定理论上限。例如平均 `100 ms` 约等于最多 `10 inference FPS`。优先确认 `--device` 确实选择了 GPU，再减小模型或 `--imgsz`。

### 没有人框

- 去掉 `--classes 0` 看模型能否识别其他类别；
- 确认自训练模型中 person 的类别 ID；
- 适当降低 `--conf`；
- 用同一模型对截图离线运行 YOLO，先排除模型本身的问题。

## 8. 当前边界

- 输出只有带识别结果的视频，不复制原流音频；
- 独立CLI仍是一进程一路流；HTTP API使用进程内多路会话和共享模型调度器；
- MediaMTX 默认未配置账号权限，适合内网验证。公网部署前必须配置认证、防火墙和 TLS；
- 模型速度不足时只能降低实际画面更新率，任何软件都无法在算力不足时保持原 FPS 且对每帧完成同等推理。

## 9. 测试

单元测试不需要摄像头、模型依赖或运行中的 MediaMTX：

```bash
python3.12 -m unittest discover -s tests -v
python3.12 -m compileall -q rtsp_annotator tests
```
