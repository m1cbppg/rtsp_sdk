# CUDA离线镜像部署

这个方案把以下内容一次性打入离线包：

- Ubuntu 24.04 / Python 3.12运行环境；
- CUDA 12.8版PyTorch、Ultralytics、PyAV和FFmpeg；
- `rtsp_annotator`代码；
- 构建时`models/`目录下的全部模型；
- MediaMTX RTSP Server镜像。

目标Ubuntu不会执行`pip install`、不会下载模型，也不会拉取Docker镜像。

## 一、在可以联网的机器制作离线包

构建前，把需要使用的模型放进项目的`models/`目录。不要把RTSP密码、
`.env`或其他密钥放进镜像。

在macOS项目目录执行：

```bash
cd /Users/mlcbppg/Desktop/backend/python_script/rtsp
chmod +x scripts/build_cuda_offline_bundle.sh
./scripts/build_cuda_offline_bundle.sh
```

如果此前已经成功构建过同一套CUDA/PyTorch依赖，这次只修改了项目代码或
`models/`，可以复用旧镜像的依赖层，避免重新下载数GB的PyTorch：

```bash
REUSE_EXISTING_IMAGE=1 \
  ./scripts/build_cuda_offline_bundle.sh \
  dist/rtsp-yolo-cuda-shared-amd64.tar.gz
```

只有依赖版本、CUDA版本或Ubuntu系统包发生变化时才执行全量构建。增量构建仍
会生成包含完整镜像的离线包，Ubuntu部署端不需要旧镜像。

输出文件：

```text
dist/rtsp-yolo-cuda-amd64.tar.gz
```

Mac为Apple Silicon ARM架构，而RTX 3060 Ti Ubuntu主机为x86_64。脚本明确使用
`--platform linux/amd64`构建，因此得到的镜像可以在Ubuntu目标机原生运行。
首次构建会下载Ubuntu、PyTorch和依赖，镜像及离线包可能达到数GB。

构建默认使用阿里云Ubuntu镜像：先通过HTTP安装CA证书，再切换HTTPS下载其余
依赖，避免部分代理的HTTP长连接在下载中途断开。需要切换镜像时可在Mac执行：

```bash
UBUNTU_MIRROR='http://其他Ubuntu镜像/ubuntu/' \
UBUNTU_MIRROR_SECURE='https://其他Ubuntu镜像/ubuntu/' \
  ./scripts/build_cuda_offline_bundle.sh
```

APT下载会在同一构建层内最多续传6轮，并使用BuildKit缓存保留已经完成的包；
构建中断后直接重新运行脚本，不要执行`docker builder prune`。

如果模型有更新，需要重新构建并传输离线包；也可以把模型单独复制到Ubuntu，
然后通过只读volume覆盖容器中的`/app/models`。

## 二、Ubuntu宿主机的一次性准备

宿主机必须有：

1. 可正常运行的NVIDIA驱动；
2. Docker Engine和Docker Compose插件；
3. NVIDIA Container Toolkit。

代码和CUDA Toolkit不需要安装到宿主机。先确认：

```bash
nvidia-smi
docker version
docker compose version
```

安装NVIDIA Container Toolkit后执行：

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Toolkit是宿主机组件，不能封装进应用镜像。如果Ubuntu完全不能联网，需要在
另一台相同Ubuntu版本、相同amd64架构的机器下载对应`.deb`包后离线安装。

## 三、导入离线包

把`rtsp-yolo-cuda-amd64.tar.gz`复制到Ubuntu，例如`/opt/rtsp-offline/`：

```bash
sudo mkdir -p /opt/rtsp-offline
sudo chown "$USER":"$USER" /opt/rtsp-offline
cd /opt/rtsp-offline
tar -xzf rtsp-yolo-cuda-amd64.tar.gz
docker load --input images.tar
```

确认两个镜像都存在且架构正确：

```bash
docker image inspect \
  rtsp-yolo-annotator:cuda-amd64 \
  --format '{{.RepoTags}} {{.Architecture}}'
docker image inspect \
  bluenviron/mediamtx:1 \
  --format '{{.RepoTags}} {{.Architecture}}'
```

## 四、验证容器能调用GPU和模型

```bash
docker run --rm \
  --gpus all \
  --entrypoint python \
  rtsp-yolo-annotator:cuda-amd64 \
  -m rtsp_annotator.runtime_check \
  --device cuda:0 \
  --model /app/models/yolo26s.pt \
  --imgsz 640 \
  --half \
  --iterations 50
```

必须看到：

```text
Device: cuda:0 (NVIDIA GeForce RTX 3060 Ti)
设备计算自检: 通过
YOLO 模型基准: 平均 ... ms
结论: 基础运行环境通过
```

## 五、启动RTSP识别

在当前终端设置输入流。使用shell环境变量可以避免把密码写进Compose文件：

```bash
export RTSP_INPUT_URL='rtsp://用户名:密码@摄像头IP:554/原流路径'
export YOLO_MODEL_PATH='/app/models/yolo26s.pt'
export YOLO_IMGSZ='640'
export YOLO_CLASSES='0'
```

如果要识别模型中的全部类别：

```bash
export YOLO_CLASSES=''
```

启动：

```bash
docker compose -f docker-compose.cuda.yml up -d
docker compose -f docker-compose.cuda.yml ps
docker compose -f docker-compose.cuda.yml logs -f annotator
```

输出流：

```text
rtsp://Ubuntu服务器IP:8554/detected
```

低延迟播放：

```bash
ffplay \
  -rtsp_transport tcp \
  -fflags nobuffer \
  -flags low_delay \
  -framedrop \
  'rtsp://Ubuntu服务器IP:8554/detected'
```

停止：

```bash
docker compose -f docker-compose.cuda.yml down
```

Compose中两个镜像都配置了`pull_policy: never`，运行过程不会尝试联网拉镜像。

如果需要通过机房公网IP和独立端口直接播放，不要使用上面的匿名配置。改用
带发布/读取账号隔离的`docker-compose.cuda.public.yml`，并按照
[PUBLIC_RTSP.md](PUBLIC_RTSP.md)配置机房TCP端口映射和外网播放器。

如果要由业务系统通过HTTP动态提交原始RTSP并获取独立的识别后RTSP地址，使用
`docker-compose.cuda.api.yml`，具体请求和部署方法见
[HTTP_API.md](HTTP_API.md)。

## 六、重要限制

- 模型和Python依赖可以完全离线封装；
- NVIDIA驱动、Docker Engine、NVIDIA Container Toolkit必须安装在Ubuntu宿主机；
- GPU驱动不能封装进镜像，因为容器运行时使用宿主机内核和驱动；
- HTTP API的CUDA模板默认使用`h264_nvenc/p4`；同一`.pt`按每两路创建一个模型
  实例，并对同实例中兼容的最新帧执行低等待动态batch；
- 独立CLI/旧Compose仍可使用`libx264 ultrafast`，便于兼容和故障回退；
- 不要在镜像或Compose文件中保存摄像头账号密码。
