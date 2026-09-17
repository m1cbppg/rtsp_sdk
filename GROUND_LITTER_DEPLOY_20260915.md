# 零散垃圾场景（`ground_litter`）1021 试点部署单（2026-09-15）

状态：**本地代码、测试与 Dockerfile 干跑已完成；未上传、未在服务器构建、未部署。**
本文只是可执行的部署单，每一步都需要用户单独授权后才能执行。

## 1. 交付物与校验值

| 文件 | 说明 | SHA-256 |
| --- | --- | --- |
| `dist/ground-litter-20260915/ground-litter-code.tar.gz` | 4 个改动模块 + 3 个新/依赖模块 + 增量 Dockerfile，72 KB | `7de44cc3bbf8dfab7eb7508dc13d2502f474778f1d14c7b7be444729fb17155f` |
| `models/litter/turhancan_yolov8m_seg_trash.pt` | 唯一一份已复核权重，54,786,410 字节 | `a2f8de0c7f714e2ab8b70c62490e2a41fd4a6681ca4a8dd442797c809a140278` |

权重摘要与 `models/litter/manifest.json` 记录一致，上传前应再核对一次。

## 1.1 为什么镜像必须在服务器上构建

本地 Mac 上的 `rtsp-yolo-annotator:deepstream8-amd64` 创建于 2026-08-02，实测**只有
numpy 1.26.4 / cupy / pillow / fastapi，没有 cv2、torch、ultralytics**；而线上运行的是
2026-09-10 的 `…-demo-continuous-20260910`（含 PTZ v12、demo-continuous、HLS 修复）。
用本地旧基础镜像构建再上传有两个后果：把生产代码回退，以及 `docker save` 需要传约
23 GB（增量层只有约 55 MB，但整镜像打包会带上全部基础层）。

因此流程固定为：**本地干跑验证 Dockerfile → 上传 72 KB 代码包 + 55 MB 权重 →
在服务器用真实基础镜像构建**。

本地干跑已验证：12 个构建步骤全部通过（COPY 路径、`py_compile`、权重存在性检查）；
新增的依赖断言在旧基础镜像上按预期失败并打印具体缺失项：

```text
基础镜像缺少零散垃圾场景所需依赖: cv2: ModuleNotFoundError: No module named 'cv2';
torch: ModuleNotFoundError: No module named 'torch';
ultralytics: ModuleNotFoundError: No module named 'ultralytics'
```

构建命令必须显式指定平台与真实基础镜像：

```bash
docker build --platform linux/amd64 \
  --build-arg BASE_IMAGE=<预检得到的实际运行镜像> \
  -f Dockerfile.deepstream.ground-litter-update \
  -t rtsp-yolo-annotator:deepstream8-amd64-ground-litter-20260915 .
```

本机是 arm64、镜像是 amd64：漏掉 `--platform linux/amd64` 时 buildx 会去 registry 找
arm64 变体并报 `pull access denied`。

如果断言只在 `cv2` 上报缺失（torch/ultralytics 存在），说明该基础镜像的 ultralytics 是
免依赖安装的；此时在断言前加一行
`RUN python3 -m pip install --no-cache-dir --break-system-packages "opencv-python-headless>=4.6,<5"`
后重建即可，不要改动其它层。

## 2. 服务器只读预检（需要授权 1）

```bash
ssh -p 21002 sf01@14.21.88.97
cd /home/sf01/rtsp-deepstream
docker compose -f docker-compose.deepstream.api.yml ps
docker compose -f docker-compose.deepstream.api.yml config | head -40
docker inspect --format '{{.Config.Image}} {{.Image}}' rtsp-yolo-deepstream-api   # 以实际容器名为准
docker images | grep rtsp-yolo-annotator
df -h /var/lib/docker; nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
curl -s localhost:8080/health
```

还要在**运行中的容器内**确认基线内容（决定增量镜像是否足够）：

```bash
docker exec <api容器> ls -l /app/models/litter/ 2>&1
docker exec <api容器> python3 -c "import rtsp_annotator.ground_litter_geometry as m; print(m.__file__)"
docker exec <api容器> python3 -c "import cv2, torch, ultralytics, numpy; print(cv2.__version__, torch.__version__, ultralytics.__version__, numpy.__version__)"
```

预期：前两条报"文件不存在"（生产镜像早于这些文件），第三条必须**全部成功**——
这是零散垃圾旁路的运行前提，也正是增量 Dockerfile 里那条断言要检查的内容。
预检结论必须按实际输出记录，不能沿用本段预期。

同时记录当前活动流列表：API 容器一旦重建，进程内的所有流任务都会停止，需业务方重建。

## 3. 上传（需要授权 2）

```bash
scp -P 21002 dist/ground-litter-20260915/ground-litter-code.tar.gz sf01@14.21.88.97:/home/sf01/rtsp-deepstream/
scp -P 21002 models/litter/turhancan_yolov8m_seg_trash.pt sf01@14.21.88.97:/home/sf01/rtsp-deepstream/models/litter/
```

上传后在服务器核对 SHA-256 与第 1 节一致，再解压到部署目录（不要覆盖已有同名文件；
解压前先列出 tar 内容确认路径）。

## 4. 构建与切换（需要授权 3 + 维护窗口）

```bash
cd /home/sf01/rtsp-deepstream
tar -xzf ground-litter-code.tar.gz
docker build -f Dockerfile.deepstream.ground-litter-update \
  -t rtsp-yolo-annotator:deepstream8-amd64-ground-litter-20260915 .
docker tag <当前运行镜像> rtsp-yolo-annotator:deepstream8-before-ground-litter-20260915   # 先留回滚标签
# 修改 compose 中 api 服务的 image（或加 override）后：
docker compose -f docker-compose.deepstream.api.yml -f <override> config -q
docker compose ... up -d api
```

不需要重建任何 TensorRT 引擎，也不要删除 `engines/` 下任何文件：本次改动只新增
一个 ultralytics 旁路，主模型引擎完全不变。

## 5. 1021 试点流与验收（需要授权 4）

请求体见 `config/ground_litter_1021_stream_request.example.json`，把 `input_url` 换成
真实地址后由调用方执行（不要把账号密码提交进仓库）。

验收清单：

1. `GET /v1/streams/{id}` 返回 `ground_litter.enabled=true`、`state=running`、
   `tile_count=5`、`analyzed_frames` 持续增长；
2. 输出 RTSP 能解码，画面出现黄色地面区域轮廓；有垃圾时出现橙色框；
3. 主线健康不退化：`publish_fps`≈源帧率、`duplicate_publish_fps=0`、
   `pipeline_healthy=true`，与部署前对比；
4. 容器日志无 `Traceback`/`SIGSEGV`，旁路进程异常不能影响主链；
5. 公网播放器观察 10–30 分钟，记录延迟、停顿和误报样例。

## 6. 回滚

```bash
docker compose ... stop api
docker tag rtsp-yolo-annotator:deepstream8-before-ground-litter-20260915 <原镜像标签>
docker compose ... up -d api
```

删除试点流（`DELETE /v1/streams/{id}`）也会一并释放该旁路进程；API 回滚后需要业务方
重新创建其它流。

## 7. 明确不包含

- 不做区域多边形的现场校准（用本目录的 `zones_render.json`/`zone_geometry_check.json`
  与渲染图核对后再上）；
- 不产生 `item_id`、不判断"已清理"、不发通知；
- 不做五路并行吞吐验收（当前只做 1021 单路）。

## 8. 部署执行记录（2026-09-15 实际执行，已完成）

| 项 | 实际值 |
| --- | --- |
| 基础镜像（预检得到） | `rtsp-yolo-annotator:deepstream8-amd64-demo-continuous-20260910`，ID `8f4bf64b95f4…`，创建 2026-09-10 |
| 基础镜像依赖 | cv2 4.11.0、numpy 1.26.4、torch 2.11.0+cu128、ultralytics 8.4.107（断言通过） |
| 构建上下文 | `/home/sf01/rtsp-deepstream/releases/ground-litter-20260915/`（`models/` 属 root，未写入） |
| 代码包 SHA-256 | `17d3b4c67d62e1b42673ee76f777e195e3b1a8038011efe8bde49fcb4529199b` |
| 新镜像 | `rtsp-yolo-annotator:deepstream8-amd64-ground-litter-20260915`，manifest `49c0149c4222…` |
| 回滚点 | `rtsp-yolo-annotator:deepstream8-before-ground-litter-20260915` → `8f4bf64b95f4…`（原 `…-demo-continuous-20260910` 标签同样保留） |
| Compose | 在原三条 `-f` 之后新增 `-f docker-compose.ground-litter.override.yml` |
| 容器 | `rtsp-yolo-api` 重建为新镜像；`rtsp-mediamtx`、`camera-control`、`rtsp-web-gateway` **启动时间与重启次数均未变** |
| 试点流 | `stream_id=498de92c72974807b0b9fdaa2eec8555`（1021），`ground_litter.enabled=true`、`region_count=5`、`tile_count=2` |
| 主链验收 | `publish_fps=25.01`、`unique_publish_fps=25.01`、`duplicate_publish_fps=0`、`pipeline_healthy=true`、坏帧 0 |
| 旁路验收 | `state=running`、分析约 1 FPS、`last_inference_ms≈33ms`、旁路异常 0；GPU 1228 MiB / 8% |
| 输出流 | 容器内 OpenCV 连续解码 120 帧、1920×1080 成功；抓图 3 张 |
| 抓图与颜色校验 | `output/ground_litter_1021_pilot/`；区域轮廓颜色命中约 33%，垃圾框橙色像素 368/434/8960 |

执行中发现并修复一个真实缺陷：`UltralyticsGroundLitterDetector.actor_boxes` 返回了 6 列预测行
（含置信度/类别），而 `box_overlap_fraction` 只解包 4 个坐标，导致启用 `actor_model` 时
**每一帧都在旁路进程内抛异常**（主链不受影响，符合隔离设计）。已改为只返回 `row[:4]`，
并补充回归测试（`tests/test_ground_litter_detection.py` 断言返回值为 4 元组且能进入重叠过滤）。
第一次部署（镜像 manifest `b9a96bf4217c`）即为此缺陷版本，已用修复版重构建覆盖标签。

同时确认的现场事实：生产 `mux_width/height = 1920×1080`，**1021 摄像头原生是 2560×1440**，
旁路拿到的是 mux 缩放后的 1080p。因此试点把区域像素门槛按 0.75 缩放为 `6px/36px²`
（与原生 2560 下的 8px/64px² 物理等价），否则用户已确认的那件 8×8px 垃圾会退化成 6×6px 被过滤。
分块数也从原生下的 5 块降到 2 块（单帧推理 33ms）。

## 9. 后续增量与回滚（2026-09-15 同日）

- 增量一：顶层 `display_detections`（默认 `true`，关闭后不画人/车框）。
- 增量二：分区最大尺寸上限 `maximum_short_side_px`/`maximum_box_area_px` —— **当日按用户要求
  已回滚**，代码、测试、文档与示例均恢复到只有增量一的状态；含该功能的镜像保留标签
  `rtsp-yolo-annotator:deepstream8-ground-litter-with-maxsize-20260915`。
- 本部署单第 2–6 节的流程未变；切换镜像只需改
  `docker-compose.ground-litter.override.yml` 中的 image 后
  `up -d --no-deps api`。回滚到本次部署前的生产版本仍用
  `rtsp-yolo-annotator:deepstream8-before-ground-litter-20260915`。
