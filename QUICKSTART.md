# RTSP 实时识别快速运行

## 1. 准备环境

需要：

- Python 3.12
- FFmpeg
- Docker Desktop
- YOLO 模型，例如 `best.pt`

进入项目：

```bash
cd /Users/mlcbppg/Desktop/backend/python_script/rtsp
```

创建虚拟环境并安装依赖：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

当前 Mac 验证 MPS：

```bash
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

确认 FFmpeg 可用：

```bash
ffmpeg -version
```

## 2. 启动 RTSP Server

先启动 Docker Desktop，然后执行：

```bash
docker compose up -d
```

查看服务状态：

```bash
docker compose ps
```

## 3. 放置模型

```bash
mkdir -p models
```

将模型放到：

```text
/Users/mlcbppg/Desktop/backend/python_script/rtsp/models/best.pt
```

## 4. 启动实时识别

识别模型支持的全部类别：

```bash
source .venv/bin/activate

python -m rtsp_annotator \
  --input 'rtsp://用户名:密码@摄像头IP:554/原流路径' \
  --model './models/best.pt' \
  --output 'rtsp://127.0.0.1:8554/detected' \
  --device mps \
  --imgsz 640
```

只识别人：

```bash
python -m rtsp_annotator \
  --input 'rtsp://用户名:密码@摄像头IP:554/原流路径' \
  --model './models/best.pt' \
  --output 'rtsp://127.0.0.1:8554/detected' \
  --device mps \
  --imgsz 640 \
  --classes 0
```

注意：`--classes 0` 只有在模型的人物类别 ID 为 `0` 时才表示人物。不传
`--classes` 就会显示模型识别出的全部类别。

框上只显示中文类别名，不显示置信度。COCO模型已内置全部中文名。自训练
模型使用英文类别名时，复制并修改`config/labels.zh.example.json`，启动
命令增加：

```bash
--label-map config/labels.zh.json
```

先从监控画面中用鼠标绘制识别区域：

```bash
python -m rtsp_annotator.roi_selector \
  --input 'rtsp://用户名:密码@摄像头IP:554/原流路径'
```

操作方式：

```text
鼠标左键       添加区域顶点
鼠标右键       撤销上一个点
R              清空重画
Enter          保存并输出 --roi 参数
Esc            取消
```

选区工具会输出类似：

```text
--roi '0.1,0.15;0.9,0.15;0.85,0.9;0.15,0.9'
```

把它加入识别命令：

```bash
python -m rtsp_annotator \
  --input 'rtsp://用户名:密码@摄像头IP:554/原流路径' \
  --model './models/best.pt' \
  --output 'rtsp://127.0.0.1:8554/detected' \
  --device mps \
  --classes 0 \
  --roi '0.10,0.15;0.90,0.15;0.85,0.90;0.15,0.90'
```

也可以不使用选区工具，直接填写坐标。ROI 使用 `0～1` 的相对坐标：

```text
0,0 ---------------- 1,0
 |                    |
 |       画面         |
 |                    |
0,1 ---------------- 1,1
```

程序会画出黄色多边形边界，只保留检测框中心点在区域内的目标。不传
`--roi` 就识别全画面。

设备参数：

```text
--device auto      自动选择：CUDA > MPS > CPU
--device mps       Apple Silicon GPU，当前 M1 Pro 推荐
--device cuda      NVIDIA 第一张 GPU
--device cuda:0    NVIDIA 指定 GPU 编号
--device 0         cuda:0 的简写
--device cpu       只使用 CPU
```

MPS 不要添加 `--half`。

### 部署到 NVIDIA GPU

在服务器创建虚拟环境后，先根据服务器驱动，从
[PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)复制对应
CUDA 版本的安装命令；再安装本项目依赖：

```bash
python -m pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

确认 CUDA 可用：

```bash
python -m rtsp_annotator.runtime_check --device cuda:0
```

CUDA 运行：

```bash
python -m rtsp_annotator \
  --input 'rtsp://用户名:密码@摄像头IP:554/原流路径' \
  --model './models/best.pt' \
  --output 'rtsp://127.0.0.1:8554/detected' \
  --device cuda:0 \
  --half \
  --imgsz 640
```

同一个 `best.pt` 可以在 MPS 和 CUDA 之间迁移，不需要重新训练。`--device`
控制的是YOLO推理设备。独立CLI默认仍由`libx264`在CPU编码；HTTP API的CUDA
模板默认使用`h264_nvenc`，macOS模板使用`libx264`。
RTX 3060 Ti / Driver 595.84 的部署步骤与延迟预期见
[SERVER_CUDA.md](SERVER_CUDA.md)。

## 5. 播放识别后的新流

低延迟播放：

```bash
ffplay \
  -rtsp_transport tcp \
  -fflags nobuffer \
  -flags low_delay \
  -framedrop \
  'rtsp://127.0.0.1:8554/detected'
```

也可以在 VLC 中打开：

```text
rtsp://127.0.0.1:8554/detected
```

其他电脑播放时，把 `127.0.0.1` 换成运行本程序的电脑 IP。

## 6. 停止

识别程序按：

```text
Ctrl+C
```

停止 RTSP Server：

```bash
docker compose down
```

## 常见问题

没有检测框：

- 暂时去掉 `--classes 0`；
- 检查模型类别 ID；
- 确认模型确实能识别当前画面。

画面卡顿：

- 确认使用了 `--device mps`；
- 将 `--imgsz 640` 改为 `--imgsz 512`；
- 查看日志中的 `inference FPS` 是否低于原流 FPS。

其他设备无法播放：

- 输出地址和播放地址使用服务器局域网 IP；
- 防火墙放通 TCP 8554；
- 确认 `docker compose ps` 显示 MediaMTX 正在运行。
