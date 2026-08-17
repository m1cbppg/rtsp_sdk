# DeepStream 低延迟版部署

该版本用于 Ubuntu + NVIDIA GPU。稳定运行时，每两路流共用一个
DeepStream/TensorRT 模型实例；解码、推理、跟踪、叠框和 H.264 编码都在
GPU 管线内完成，不再把 1080p 帧来回拷贝到 Python/CPU。

## 1. Mac 打离线包

项目首次需要补齐 ONNX 导出依赖：

```bash
cd /Users/mlcbppg/Desktop/backend/python_script/rtsp
.venv/bin/pip install onnx onnxslim onnxscript
.venv/bin/pip install 'git+https://github.com/ultralytics/CLIP.git'
./scripts/build_deepstream_offline_bundle.sh
```

若刚改的只是 Python 代码，且本机已有
`rtsp-yolo-annotator:deepstream8-amd64`，可复用大镜像层：

```bash
REUSE_EXISTING_IMAGE=1 ./scripts/build_deepstream_offline_bundle.sh
```

默认输出：

```text
dist/rtsp-yolo-deepstream8-amd64.zip
```

DeepStream 镜像未压缩约 23 GB，本次 ZIP 约 13 GB。打包前建议 Mac 至少留
15 GB 可用空间。Ubuntu 使用下面的流式 `docker load` 时建议至少留 40 GB；
如果先完整解压 `images.tar`，则建议至少留 65 GB。新打包器会把
`docker save` 直接流式压入 ZIP，不再额外生成一份 23 GB 的 Mac 中间文件。
若同名 ZIP 已存在，脚本会停止，避免静默覆盖旧包。

## 2. 上传并替换服务器版本

Mac：

```bash
scp -P 21002 \
  dist/rtsp-yolo-deepstream8-lpr-amd64.zip \
  dist/rtsp-yolo-deepstream8-lpr-amd64.zip.sha256 \
  sf01@14.21.88.97:~/
```

Ubuntu：

```bash
mkdir -p ~/rtsp-deepstream
cd ~/rtsp-deepstream

# 先验证上传没有损坏
cd ~
sha256sum -c rtsp-yolo-deepstream8-lpr-amd64.zip.sha256
cd ~/rtsp-deepstream

# 已部署过时先备份配置并停服务；不会删除engines目录
cp config/api.json ~/api.json.before-lpr 2>/dev/null || true
docker compose -f docker-compose.deepstream.api.yml down 2>/dev/null || true

# 只解出Compose、配置和文档，不落盘23 GB的images.tar
python3 - ~/rtsp-yolo-deepstream8-lpr-amd64.zip . <<'PY'
import sys, zipfile
archive, destination = sys.argv[1:3]
with zipfile.ZipFile(archive) as bundle:
    for item in bundle.infolist():
        if item.filename != "images.tar":
            bundle.extract(item, destination)
PY

# 把ZIP中的images.tar直接送给docker load
set -o pipefail
python3 - ~/rtsp-yolo-deepstream8-lpr-amd64.zip <<'PY' | docker load
import shutil, sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as bundle:
    with bundle.open("images.tar") as source:
        shutil.copyfileobj(source, sys.stdout.buffer, 8 * 1024 * 1024)
PY

# 只有首次部署才从示例创建；升级不要覆盖原有密码和公网地址
test -f config/api.json || cp config/api.deepstream.example.json config/api.json
nano config/api.json
mkdir -p engines
docker compose -f docker-compose.deepstream.api.yml config -q

# 保证已退出的engine-builder重新运行并生成LPD/LPR引擎
docker compose -f docker-compose.deepstream.api.yml up -d --force-recreate
```

`config/api.json` 至少修改：

- `api.key`
- RTSP 发布/读取密码
- `rtsp.public_base_url`，当前映射应为
  `rtsp://14.21.88.97:38554`

不要改这些性能契约：

- `inference.backend`: `deepstream`
- `deepstream.streams_per_group`: `2`
- `deepstream.model_input_size`: `640`
- `deepstream.encoder_iframe_interval`: `25`
- `deepstream.tracker_max_shadow_tracking_age`: `15`
- API 请求中的 `imgsz`: `640`

首次启动会在 3060 Ti 上构建 TensorRT FP16 引擎。查看进度：

```bash
docker compose -f docker-compose.deepstream.api.yml logs -f engine-builder
```

生成的 `.engine` 保存在 `./engines`。后续换代码或重启会直接复用；更换
GPU 型号、DeepStream/TensorRT 版本或模型后，应删除对应 `.engine`，再让
它重建。

带中国车牌识别的镜像还会构建`lpdnet_ch_*`和`lprnet_ch_*`两个引擎。
`engine-builder`成功退出后API才启动，因此首个车牌请求不会在播放过程中临时
编译模型。车牌功能的请求方式和指标见`LICENSE_PLATE.md`。

开启垃圾事件分析时还会预构建
`yolo_world_garbage_640_b2_gpu0_fp16.engine`。垃圾 ONNX 和固定词表已经打入
离线镜像，Ubuntu 服务器无需访问互联网。请求与验收见`EVENT_DETECTION.md`。

固定机位燃气瓶识别使用镜像内的`YOLOE-26L-seg`和摄像头视觉提示Profile，
运行在独立丢帧旁路，不生成DeepStream主推理引擎。完整请求、36瓶录像基线和
码流质量验收见`GAS_CYLINDER.md`。

## 3. 验证

```bash
curl http://127.0.0.1:8080/health

curl -H 'X-API-Key: 你的API_KEY' \
  http://127.0.0.1:8080/v1/models
```

创建一路：

```bash
curl -X POST http://127.0.0.1:8080/v1/streams \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: 你的API_KEY' \
  -d '{
    "input_url": "rtsp://摄像头账号:密码@摄像头地址/路径",
    "model": "yolo26s.pt",
    "classes": [0],
    "conf": 0.25,
    "imgsz": 640,
    "bitrate": "2500k"
  }'
```

同时启用中国车牌识别时，在请求中增加：

```json
"license_plate": {"enabled": true}
```

启用独立夜间识别时，在请求中增加：

```json
"night_vision": {
  "enabled": true,
  "confidence": 0.18,
  "input_gain": 1.18,
  "plate_detector_confidence": 0.20
}
```

夜间参数只作用于模型输入和检测阈值，最终转发画面保持原图。白天请求不传
该字段，完整说明见`NIGHT_VISION.md`。

如需识别区域：

```json
"roi": [[0.1,0.1],[0.9,0.1],[0.9,0.9],[0.1,0.9]]
```

API 返回的公网 RTSP 地址继续使用：

```text
rtsp://viewer:读取密码@14.21.88.97:38554/detected/流ID
```

低缓存播放：

```bash
ffplay -rtsp_transport tcp -fflags nobuffer -flags low_delay \
  -framedrop -probesize 32 -analyzeduration 0 '返回的rtsp_url'
```

## 4. 四路是否达标

四路都创建完成并稳定 30 秒后：

```bash
curl -H 'X-API-Key: 你的API_KEY' \
  http://127.0.0.1:8080/v1/streams

docker compose -f docker-compose.deepstream.api.yml logs \
  --since=1m api | grep '状态:'

nvidia-smi dmon -s pucvmet
```

25 FPS 摄像头的最低验收线：

- 每路 `unique_publish_fps` 和 `publish_fps` 持续不低于 20；
- `duplicate_publish_fps` 必须为 0；
- `pipeline_healthy` 为 `true`；
- 四路形成两个 `model_instance_id`，每个
  `model_instance_clients` 为 2；
- 连续观察 10 分钟没有推流退出、显存持续增长或画面冻结。

目标值是每路接近源流的 25 FPS。公网端到端延迟还会受到摄像头 GOP、
专线、端口映射和播放器缓存影响，因此不能只用服务器内部 FPS 代替真实
延迟测试。使用低缓存 ffplay 时，合理目标是约 0.3–0.8 秒，最终以服务器
和公网播放器实测为准。

## 5. 增删流注意

同一模型、同一 `imgsz` 的第 1/2 路进入同一组，第 3/4 路进入下一组。
当前实现为保证一个进程内真正共享 TensorRT 上下文，在组内加入第 2 路或
删除其中一路时，会重启该组，组内另一条流会短暂中断几秒；稳定运行后不
会周期性重启。

删除：

```bash
curl -X DELETE \
  -H 'X-API-Key: 你的API_KEY' \
  http://127.0.0.1:8080/v1/streams/流ID
```

删除最后一路后，worker、推流和含 RTSP 密码的临时配置会一起清理；
TensorRT 引擎缓存保留。
