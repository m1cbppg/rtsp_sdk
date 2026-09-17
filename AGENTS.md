# RTSP 项目 AI 交接与工作约定

本文件适用于本仓库全部目录。它的目标不是替代专题文档，而是让第一次进入本仓库的
AI 能先恢复项目上下文、知道哪些结论已经验证、哪些仍需现场验证，以及什么时候必须
向用户申请服务器权限。

最近基线核对：2026-08-20，当时本地分支为 `main`，HEAD 为 `351d568`，与
`origin/main` 一致且工作区干净。2026-08-21 又完成了尚未提交的PTZ船舶近景复核实现；
这里的提交号只表示旧基线。开始新任务时仍必须重新执行
`git status --short --branch`，不能把本段当作永远有效的状态。

## 1. 开始工作的第一原则

1. 从用户的业务目标和可验证证据出发，不要默认用户已经把实现方案定义完整。
2. 写代码前必须先给出计划。复杂改动要先拆解数据链路、故障隔离、兼容性、测试、
   部署和回滚，再开始编辑。
3. 先阅读本文件，再按任务阅读对应专题文档和代码。专题文档比本文件中的摘要更详细；
   代码和实时运行证据比可能过期的文档更权威。
4. 区分三类结论：本地代码可证明、历史文档记录、当前服务器实时验证。不得把后两类
   混为一谈。
5. 未经用户明确授权，不连接服务器、不上传文件、不改线上配置、不重启容器、不创建或
   删除线上流、不修改防火墙/端口映射，也不执行任何需要 `sudo` 的命令。

## 2. 项目是什么

这是一个“RTSP 输入、实时视频分析、叠加中文结果、再发布 RTSP”的服务。HTTP API
负责动态创建、查询、更新部分规则和停止任务；视频不经 HTTP 返回，播放器使用 API
响应中的 `rtsp_url`。

核心目标：

- 持续读取 RTSP 最新画面，避免模型慢时排队造成延迟无限增长；
- 在输出视频上显示中文检测框、业务提示和事件状态；
- 通过 MediaMTX 发布每路独立的处理后 RTSP；
- 在 NVIDIA 生产环境中让解码、主推理、跟踪、OSD、编码尽量留在 GPU 管线；
- 将高耗时或低频业务分析放入容量为 1 的可丢帧旁路，不能拖住主推流；
- 保存逐流健康日志、事件 JSON、截图，并支持异步 Webhook 和人工复核。

项目不是：

- 摄像头管理平台、播放器前端或视频存储系统；
- 能从单帧直接给出法律结论的执法系统；
- 自动保证任意摄像头准确率的通用模型。新机位必须用现场录像校准和验收；
- 音视频无损中继。当前输出不复制原流音频，叠框必然需要重新编码。

## 3. 两条运行链路

### 3.1 Python / PyTorch 兼容链路

适用于 macOS/MPS、本地开发、基础能力验证和兼容回退：

```text
RTSP -> PyAV/FFmpeg 拉流 -> 最新帧槽 -> Ultralytics YOLO
     -> 检测结果缓存/跟踪外推 -> FFmpeg H.264 -> MediaMTX -> RTSP
```

- 独立 CLI 是一进程一路流。
- HTTP API 可使用共享模型调度器；兼容的每两路可共享一个模型实例并动态 batch。
- 默认软件编码为 `libx264/ultrafast`；CUDA API 模板可使用适配过的 NVENC 参数。
- macOS 可以验证基础 YOLO RTSP 链路，但不能等价验证 DeepStream 专属功能。

### 3.2 DeepStream / TensorRT 生产链路

适用于 Ubuntu + NVIDIA GPU 的正式多路部署：

```text
输入 RTSP
  -> NVDEC
  -> YOLO TensorRT -> NvDCF 跟踪 -> 可选车牌链路
  -> 事件状态机与低帧率业务旁路
  -> GPU OSD -> NVENC -> MediaMTX -> 输出 RTSP
```

- 每两路固定共享一个 DeepStream/TensorRT 组，第 3/4 路进入第二组。
- 组内增加第 2 路或删除其中一路会重启该组，另一条流可能短暂中断数秒。
- 默认关键契约：`streams_per_group=2`、模型输入 640、GOP/关键帧间隔 25、
  `tracker_max_shadow_tracking_age=15`。更改前必须说明性能或兼容原因并回归测试。
- 主发布队列不能为了旁路推理改成丢帧队列。高耗时旁路只能丢弃旧分析帧，不能
  积压或丢弃待编码主视频帧。
- TensorRT `.engine` 只对相应 GPU、DeepStream/TensorRT 版本和模型有效。上述任一
  发生变化时，需在获得服务器授权后删除对应引擎并重建；不要无差别清空目录。

## 4. 已实现能力和真实边界

| 能力 | 后端 | 当前实现与边界 |
| --- | --- | --- |
| 普通 YOLO 人/车/COCO 类别 | Python、DeepStream | `classes` 只控制普通框显示；中文标签，不显示置信度 |
| 顶层普通 ROI | Python、DeepStream | 过滤显示结果；不自动产生事件；Python 链路不会因此减少整图推理量 |
| 中国车牌 | DeepStream | LPDNet + 异步 LPRNet + 时序投票；车牌建议至少约 80 像素宽 |
| 夜间配置 | DeepStream | 只增强模型输入并使用独立阈值，不提亮最终画面；日夜任务不共组 |
| 区域停留 | DeepStream | 人/车轨迹进入事件 ROI 达到时间后生成 `zone_dwell` |
| 垃圾变化/乱丢线索 | DeepStream | YOLO-World 或垃圾堆模型 + 背景变化 + 轨迹状态机；结论必须人工复核 |
| 燃气瓶固定机位计数 | DeepStream 旁路 | YOLOE 独立进程、多帧共识；历史样例稳定计数 36，不代表所有机位 |
| 船舶高召回框选 | DeepStream 旁路 | COCO boat、1280 输入、分区推理、ROI/排除区；不识别船名或证照 |
| PTZ船舶近景复核/跟踪 | DeepStream旁路 + camera_control | 可选运动+局部外观高召回小目标提议、船框反馈自适应变焦（默认宽/高约33%，保留固定回退）、近景boat确认、原生抓图；默认截图后强制回HOME，可显式开启持续跟踪、中心纠偏和双向变焦，目标丢失/超时后回HOME；支持流级紧急中断并回HOME；SQLite跨ID去重；默认关闭，现场仍需校准方向/倍率/预置位 |
| 疑似非法捕捞线索 | DeepStream 规则层 | 依赖船舶轨迹、区域、时间表、停留/折返；不能自动认定违法 |
| 逐流日志和健康诊断 | API 两类管理器外层 | JSONL + REST + SSE；可区分 degraded/stalled/mosaic_risk 等 |
| 事件证据与 Webhook | DeepStream 事件功能 | JSON、截图持久化；Webhook 异步，失败不应影响视频 |

重要业务边界：垃圾和捕捞相关结果是“疑似事件/人工复核线索”，不能直接用于处罚
或不可逆动作。公开模型或合成测试通过，不代表目标现场生产准确率已经通过。

## 5. 当前部署记忆（先验证，后引用）

`PROJECT_STATUS_AND_SCENARIOS.md` 在 2026-08-01 记录过以下正式部署：

- SSH 目标记录：`sf01@14.21.88.97`，端口 `21002`；
- 服务器部署目录：`/home/sf01/rtsp-deepstream`；
- 公网 API：`http://14.21.88.97:38080`；
- 公网输出 RTSP 基址：`rtsp://14.21.88.97:38554`；
- 硬件/运行栈：Ubuntu、RTX 3060 Ti、DeepStream 8、TensorRT FP16；
- 文档当时记录运行镜像为
  `rtsp-yolo-annotator:deepstream8-events-final-amd64`。

但仓库当前 `docker-compose.deepstream.api.yml` 使用的镜像名是
`rtsp-yolo-annotator:deepstream8-amd64`。这可能是历史部署标签与当前打包标签不同，
不是可以擅自“修正”的错误。任何部署前都必须以服务器上的 `docker compose config`、
`docker compose ps` 和 `docker image inspect` 为准，并向用户报告差异。

历史验证记录：真实 1920x1080、25 FPS 流在普通识别、车牌、区域停留和垃圾旁路
同时开启时曾保持 pipeline/pre-encode/publish 约 25 FPS、duplicate 0、
`pipeline_healthy=true`；公网连续解码 30 秒曾得到 751 帧。此结论是历史证据，
不是当前服务器健康状态。需要声称“现在正常”时必须实时查询。

尚未完成的核心现场工作：目标摄像头下的丢垃圾/捡垃圾/路过/保洁/车辆停靠、
昼夜雨雾、遮挡与拖影等准确率验收。船舶、燃气瓶等新机位也必须分别校准 ROI、
Profile 和阈值。

2026-08-21 已在上述服务器完成PTZ船舶复核增量部署：当前
`rtsp-yolo-annotator:deepstream8-amd64`镜像ID前缀为`2898bbce0189`，切换前镜像保留为
`deepstream8-before-ptz-20260821`（ID前缀`62d604c72d61`）。独立
`camera-control:dahua-sdk-20260821`容器加入
`rtsp-yolo-deepstream-api_default`网络，仅绑定服务器回环地址`127.0.0.1:18080`；
API与camera-control共享未回显的内部环境密钥。部署时无活动流，MediaMTX未重启，
公网`/health`与跨容器Mock定位/对焦/原生抓图/回HOME已通过。camera-control当前仍使用
`mock-001`，大华SDK原生库只完成无设备初始化；真实摄像机方向、倍率、HOME和抓图必须
到内网现场验证，不能把本次Mock部署记成真机验收。

2026-08-21 已继续部署PTZ P1修复：当前生产标签
`rtsp-yolo-annotator:deepstream8-amd64`镜像ID前缀为`6dfee6d58055`，直接上一版保留为
`deepstream8-before-p1fix-20260821`（ID前缀`2898bbce0189`）。本次只重建API容器，
MediaMTX与camera-control未重启；公网健康、P1默认参数、SQLite增量迁移、固定视角冲突
422校验和跨容器camera-control健康均通过。部署时仍无活动流，不能视为真实摄像机验收。

2026-08-24 已部署船框反馈自适应变焦：当前生产标签
`rtsp-yolo-annotator:deepstream8-amd64`镜像ID前缀为`68b8c7ef5781`，切换前实际线上镜像
保留为`deepstream8-before-adaptive-ptz-20260824`（ID前缀`865b65a633b4`）。只重建API，
MediaMTX未重启。另发现线上camera-control缺少已实现的SDK状态和一键诊断路由，已用
保留原生SDK层的增量镜像补齐；当前`camera-control:dahua-sdk-20260821`镜像ID前缀为
`363f4f3b8b43`，旧镜像保留为`before-diagnostics-20260824`（ID前缀`b34bb0b8b630`）。
线上用本次船舶样例完整验证：识别-only约30 FPS、duplicate 0、稳定检出1到3艘；
自适应Mock联动生成`boat_confirmed`证据并成功回HOME；输出RTSP另行解码60帧成功；
SDK在容器内初始化`loaded=true`。测试流、Mock证据和临时视频均已清理。这里仍只证明
服务器软件、Mock控制和SDK加载，真实摄像机登录、方向、倍率、抓图及HOME仍需现场验收。

2026-08-24 又部署了水面运动+局部外观的小目标高召回和PTZ候选碎片合并：当前生产
`rtsp-yolo-annotator:deepstream8-amd64`镜像ID前缀为`6b1b6ae04dcc`；直接上一版高召回
镜像保留为`deepstream8-before-dedup-fix-20260824`（ID前缀`ece449c49073`），原自适应
变焦基线仍保留为`deepstream8-before-high-recall-20260824`对应的`68b8c7ef5781`镜像。
只重建API，MediaMTX和camera-control未重启。`8月12日.mp4`前60秒线上回放中，
detection-only稳定候选达到8，主链约30 FPS、duplicate 0；Mock PTZ联动首轮暴露了
复核后目标速度外推导致的重复建任务，修复为“完成后固定触发坐标、冷却期不外推也不被
碎片拖移”。修复版跨完整视频周期得到20个复核位置，任意两次触发距离均大于4%，
19个`boat_confirmed`、1个`candidate_not_confirmed`、全部回HOME；证据JPEG及SHA-256
一致，输出RTSP经NVDEC解码60帧成功。195项本地测试和服务器完整DeepStream构图通过。
本次52条Mock任务、46张假证据、临时流/视频/构建目录均已清理，线上最终无活动流。
这些结果仍不是大华真机或目标现场准确率验收。

2026-09-01 修复了 API `/internal/mediamtx/auth` 对 HLS/WebRTC 读取鉴权过严的问题：原逻辑只放行
`action=read` 且路径以无前导斜杠的 `detected/` 开头，而 MediaMTX 对 HLS（浏览器端）上报
`/detected/<id>`（带前导斜杠）且读取动作可能为 `play`，导致 HLS 浏览器播放一直 401。
现统一 `path.lstrip("/")` 并接受 `action in ("read","play")`，发布权限不变；另新增一个
`rtsp-web-gateway` 容器（`0.0.0.0:8088`，同源反代 HLS，机房 NAT `38088->8088`）对外提供
浏览器页面(`/`)与 HLS(`/detected/*`)。本次以服务器现有基础镜像做 `COPY api.py` 覆盖层重建，
当前生产标签仍为 `rtsp-yolo-annotator:deepstream8-amd64`，镜像ID前缀改为`d6d273e2`，切换前
镜像保留为`deepstream8-before-hlsfix-20260901`，配置备份为`config/api.json.bak-hlsfix-20260901091915`；
只为WEB视图目的改动，MediaMTX、engine-builder、camera-control 未重启，224项单测通过。
同日进一步确认：MediaMTX 的 HLS **只接受 HTTP Basic Auth，忽略 `?user=&pass=` 查询参数**，因此
网关改为在反代 `/detected/*` 时从 `/srv/secrets.json`（宿主 `secrets-viewer.json`，mode 600）注入
Basic 头，浏览器端不再依赖查询参数即播放；网关容器已重建以挂载该 secret。
同页还支持多路 RTSP（每行一个 `名称|rtsp://...`）并展示 SQLite 复核图片：网关新增
`/v1/vessel-verifications[...]` 到 `api:8080` 的反代并注入 `X-API-Key`（`api_key` 已加入
`secrets-viewer.json`），页面同源读取 `/v1/vessel-verifications` 列表及
`/v1/vessel-verifications/{job_id}/images/{image_id}` 图片，无需浏览器持有密钥。
同日为大屏页做长播健壮性优化：hls.js 已自托管到 `web/hls.min.js`（网关 `/hls.min.js` 同源提供，避开外网）。
HLS 配置加 `backBufferLength:30/maxMaxBufferLength:60` 限制后退缓冲（防长时间播放内存无限增长）；重连改为
指数退避（5s→10s→20s→30s封顶）而非每 8 秒高频重建；重建配置前先 `stopCard` 销毁旧 Hls/worker 防泄漏；
加 `beforeunload` 清理与页面隐藏时暂停视频。上墙页面按 2x2 铺满左侧、单路点视频可全屏。

2026-09-03 已部署船舶持续跟踪与紧急回位：生产标签
`rtsp-yolo-annotator:deepstream8-amd64`镜像ID前缀为`cd7d5856ce9d`，切换前镜像保留为
`deepstream8-before-continuous-tracking-20260903`（ID前缀`d6d273e263b9`）。创建流时
`ptz_verification.continuous_tracking`默认`false`以保持“截图后回HOME”，显式传`true`才持续
跟踪；默认自适应目标宽高均为`0.33`。新增
`POST /v1/streams/{stream_id}/ptz/return-home`，会中断活动PTZ、回HOME并锁存为
`manual_hold`，更新或重建流后才恢复自动PTZ。完全虚拟且禁用真机镜像的DeepStream闭环中，
实测跟踪10.4秒、自动纠偏1次，最后目标框宽约23.8%、高约32.3%、输出约10 FPS；跟踪中
紧急回位后倍率1.0、预置位1，20秒内即使继续检出船也未新增定位。234项本地测试和114项
服务器候选镜像关键测试通过。部署时生产无活动流，只重建API；MediaMTX和camera-control
未重启。以上不是大华真机或目标现场准确率验收，真机仍需维护窗口验证倍率映射、延迟和
跟踪稳定性。
注：API 容器重启会停掉进程内所有流任务，需调用方重建；网关容器本身不影响流。

2026-09-10 用户授权只读 SSH 核查：实际运行 API 为
`rtsp-yolo-annotator:deepstream8-amd64-ptz-v12`，镜像 ID 前缀 `1add8299293c`，
创建于 2026-09-04，容器于 2026-09-07 启动；本地 v12 包内 4 个 Python 文件均与运行
容器逐字节一致。Compose 同时使用 `docker-compose.deepstream.api.yml` 与
`docker-compose.ptz-v12.override.yml`；默认 `deepstream8-amd64` 标签不是运行镜像。
camera-control 实际镜像 ID 前缀 `966377e24325`。API `/health`、控制服务 `/healthz`
正常，检查时流列表为空。逐文件比较确认船舶主链有 RTSP 8 个、camera-control 2 个核心
代码文件未更新，另 6 个演示工具脚本未交付到部署目录。本地已存在 `demo_continuous`
代码，旧进展文档中“尚无 profile”等早期段落不能继续作为当前实现状态。23 项相关单测
通过，但重跑模拟仍有压力/加速失败，完整集成与真机验收未完成。本次未上传、部署、
重启或控制摄像头。准确计数、排除项和证据见 `DEPLOYMENT_AUDIT_20260910.md`。

2026-09-10 后续更新及续验已完成：最终保留 API 镜像
`rtsp-yolo-annotator:deepstream8-amd64-demo-continuous-20260910`（ID 前缀
`8f4bf64b95f4`）及 `camera-control:dahua-sdk-command-cancel-20260910`（`c891e8d5379f`），
续验没有重新部署、重启或回滚。API 的 Compose 在 v12 override 上再加
`docker-compose.demo-continuous.override.yml`；camera-control 增加
`docker-compose.camera-control-command-cancel.override.yml`。旧镜像仍保留为
`before-demo-continuous-20260910`（`1add8299293c`）和
`before-command-cancel-20260910`（`966377e24325`）。仅更新 RTSP 8 个、camera-control 2 个
运行模块；6 个演示辅助脚本在部署目录和 API 容器内均不存在。
临时标准流此前没有指标的原因已定位：`boat-test.mp4` 实为 H.265，旧测试发布器使用
`h264parse`，管线虽存活却没有视频缓冲，最新流持续收到输入 404。改用 `h265parse`
并先确认输入 NVDEC 解码 62 帧后，仅做一次 PTZ 关闭的标准流复测
（`1b6ca7c7a17e4bc5bbe7464d636b460c`）：观察 120.6 秒，预热后 10 次采样发布与唯一帧率
29.985–30.022、重复帧率 0、健康为 true，船舶旁路 running、检出 2–5，输出 NVDEC
实际解码 3,406 帧。测试流删除后超过两分钟复核，两个新容器 RestartCount=0、健康正常，
启动日志未检出 ERROR/Traceback/SIGSEGV；活动流为 0，无发布器、worker、runtime 或测试
临时文件残留，测试输入/输出路径均已 404。MediaMTX 与网关仍为 9 月 6 日原容器且未重启。
原始视频、回滚镜像和逐流诊断日志均保留。相关 163 项与 camera-control 30 项单测通过是
部署交接证据，RTSP 全量仍因共享推理线程卡住而中断，不能称全量通过。压力模拟仍失败，
本次仅证明单路标准链路；demo_continuous 真机、长时公网及其他业务专项尚未验收。
完整镜像 ID、时间、证据和边界以 `DEPLOYMENT_AUDIT_20260910.md` 顶部最终结论为准。

## 6. 仓库导航

入口和配置：

- `rtsp_annotator/__main__.py`、`cli.py`：单路 CLI；
- `rtsp_annotator/api.py`：FastAPI 请求模型、鉴权和路由；
- `rtsp_annotator/api_settings.py`：严格配置模型和管理器设置转换；
- `config/api.*.example.json`：macOS、CUDA、DeepStream 配置样例；
- `docker-compose.deepstream.api.yml`：生产 DeepStream API、engine-builder、MediaMTX；
- `pyproject.toml`：Python 版本、依赖、命令入口。

主链和管理：

- `pipeline.py`：Python 拉流、检测、叠框、FFmpeg 发布和统计；
- `shared_inference.py`、`shared_stream_manager.py`：PyTorch 共享推理和多流管理；
- `stream_manager.py`：基础流生命周期、模型校验和 RTSP 鉴权地址；
- `deepstream_manager.py`：DeepStream 分组、worker 生命周期和动态配置；
- `deepstream_worker.py`：GStreamer/DeepStream 管线、OSD、指标和业务旁路集成；
- `deepstream_engine_builder.py`：TensorRT 主模型及辅助模型引擎构建；
- `stream_observability.py`：逐流日志、健康判定和 SSE。

业务模块：

- `events.py`、`event_engine.py`、`background_change.py`、`event_evidence.py`、
  `event_delivery.py`：事件规则、状态、证据和 Webhook；
- `license_plate.py`：中国车牌规范化和投票稳定；
- `gas_cylinder.py`、`gas_cylinder_process.py`：燃气瓶固定机位旁路；
- `vessel_detection.py`、`vessel_detection_process.py`、`vessel_calibration.py`：
  船舶识别、进程隔离和新机位标定；
- `ground_litter_detection.py`、`ground_litter_process.py`：`/v1/streams` 的
  `ground_litter` 场景（原生像素分块、旁路进程、显示层投票/保持）；
- `ground_litter*.py`（其余）：离线影子试点、校准、验收与批量回放工具，
  不接入生产API；
- `ptz_verification.py`：PTZ复核状态机、camera_control客户端、SQLite目标记忆和证据；
- `fishing_risk.py`：基于船舶轨迹的风险评分规则；
- `labels.py`：中文标签和字体。

持久化和临时状态：

- `data/events`：事件 JSON 与截图，删除流不会自动删除历史事件；
- `data/events/vessel-verifications`：PTZ复核SQLite/WAL与近景原图；
- `data/stream-logs`：逐流 JSONL 及轮转文件；
- `engines`：服务器 TensorRT 缓存；
- `/app/runtime`：容器 tmpfs 内的临时 worker 配置，最后一路删除后应清理；
- API 流任务保存在进程内存中，API 容器重启后调用方需要重新创建任务。

## 7. 文档路由

开始任务时只读相关文档，但必须把选中的文档读完整：

- 总览、场景请求、部署记录：`README.md`、`PROJECT_STATUS_AND_SCENARIOS.md`；
- 本地快速运行：`QUICKSTART.md`、`VALIDATION.md`；
- DeepStream 打包与部署：`DEEPSTREAM_DEPLOY.md`；
- API、认证与生命周期：`HTTP_API.md`；
- 事件/垃圾：`EVENT_DETECTION.md`；
- 零散垃圾当前进度与 VLM 方案：`GROUND_LITTER_VLM_PLAN.md`（2026-09-17；VLM 尚未实施，区分 API 显示轨迹和独立影子库存）；
- 百炼 VLM 离线小样本实测：`output/ground_litter_vlm_probe_20260917/REPORT.md`（2026-09-17；用户授权13次Qwen调用，4个历史模型误报均拒绝，但5个垃圾参考正例仅1个通过、4个不确定；不支持直接强制VLM放行，未接入生产）；复现入口 `scripts/probe_ground_litter_vlm.py`。
- 1021 原始流只读实测：2026-09-17 确认为 HEVC 2560×1440、25 FPS；当前垃圾旁路仍在 1920×1080 mux 后。后续优先改为 mux 前原图旁路，并让 1021/1022 各负责近端半段、交界重叠区固定唯一主摄像头；完整证据边界和像素换算见 `GROUND_LITTER_VLM_PLAN.md`，不得保存或回显现场 RTSP 凭据。
- 百炼极小目标多帧补测：`output/ground_litter_vlm_temporal_20260917/REPORT.md`（同一件用户确认的8像素垃圾用三个时刻复核，仍为uncertain；两轮共14次/8050 tokens后停止调用）。两摄像头各负责近端半段方向合理，但必须让另一台真正获得更多目标像素；只缩ROI/放大无效，同物体跨机同步对照尚未完成。
- 车牌：`LICENSE_PLATE.md`；夜间：`NIGHT_VISION.md`；
- 燃气瓶：`GAS_CYLINDER.md`；船舶：`VESSEL_DETECTION.md`；
- 小目标PTZ近景复核：`PTZ_VESSEL_VERIFICATION.md`；
- 捕捞风险：`FISHING_RISK.md`；
- 日志诊断：`STREAM_LOGGING.md`；公网 RTSP：`PUBLIC_RTSP.md`；
- 离线 CUDA 与服务器基础环境：`OFFLINE_CUDA.md`、`SERVER_CUDA.md`；
- 第三方模型许可：`THIRD_PARTY_MODEL_NOTICES.md`。

若代码、样例配置和文档不一致，先确定运行代码的事实，再同时修正文档；不要只改一个
请求样例留下新的漂移。

## 8. 本地开发和验证

Python 要求 3.10 到 3.12，推荐 3.12。不要误用工作区上层的 Python 3.14 环境。
本仓库自己的 `.venv` 在编写本文件时存在，但新环境仍应验证解释器版本。

基础命令：

```bash
cd /Users/mlcbppg/Desktop/backend/python_script/rtsp
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m compileall -q rtsp_annotator tests
```

测试约定：

1. 先运行与改动直接相关的测试，再运行完整 `unittest`。
2. 单元测试设计上不要求真实摄像头、模型依赖或 MediaMTX；不得以没有服务器为由跳过。
3. 修改 DeepStream 管线、打包或配置契约时，还要运行对应的
   `tests/test_deepstream_*`、`tests/test_package_deepstream_bundle.py` 和
   `scripts/validate_deepstream_runtime_contract.py`（按脚本参数要求执行）。
4. 性能、码流、NVENC、TensorRT、车牌和旁路故障隔离不能只靠单元测试证明，必须在
   获得权限后用服务器与真实 RTSP 做专项验收。
5. 不要为了让测试通过而弱化“旁路不得阻塞主流”“duplicate 必须为 0”等核心契约。

生产 25 FPS 源流的最低健康指标通常是：`publish_fps` 和
`unique_publish_fps >= 20`、`duplicate_publish_fps=0`、
`pipeline_healthy=true`。正式验收目标是接近源帧率，并至少连续观察 10 到 30 分钟；
还要从公网播放器验证延迟、停顿、马赛克和解码错误，不能只看服务器内部 FPS。

## 9. API 任务生命周期要点

- `GET /health` 不使用业务 API Key；其他公开管理接口使用 `X-API-Key`。
- `POST /v1/streams` 创建任务并返回 `stream_id`、状态和带读取认证的 `rtsp_url`。
- `GET /v1/streams/{id}` 查询状态与指标；`DELETE` 才是真正停止，关闭播放器不停止任务。
- 修改模型、类别、普通 ROI、事件 ROI、夜间配置或多数阈值，一般应删除后重建。
- `PATCH /v1/streams/{id}/fishing-risk` 是当前明确支持的运行中规则切换，但会重载
  所在 DeepStream 组，可能短暂抖动。
- `classes` 只控制普通 YOLO 结果展示，不会自动关闭车牌、事件内部车辆追踪或旁路模型。
- 顶层 `roi` 与 `event_detection.rois` 用途不同，不能混用。
- MediaMTX 通过内部 HTTP 回调向 API 校验发布/读取权限。

## 10. 服务器权限：什么时候必须停下来问用户

仓库中出现服务器地址、用户名、端口和部署命令，只是上下文，不代表永久授权。以下操作
前必须向用户申请一次具体授权：

- 首次 SSH/远程命令，即使只是查看状态；
- `scp`、`rsync`、上传离线包或将本地内容发送到服务器；
- `docker load/build/pull/tag/rmi`、`docker compose up/down/restart`；
- 修改 `/home/sf01/rtsp-deepstream` 下任何文件、真实 `config/api.json` 或事件数据；
- 停止/删除流、清理 `.engine`、镜像、日志、事件、截图或旧压缩包；
- 防火墙、端口映射、TLS/VPN/ACL、系统服务、NVIDIA runtime 和任何 `sudo` 操作；
- 可能造成组内其他流短暂中断的在线更新。

申请时不要只说“给我服务器权限”。至少告诉用户：

```text
目的：要验证或部署什么
目标：服务器、目录、服务或具体 stream_id
权限：只读 / 上传 / 修改配置 / 重启服务 / sudo
准备执行：关键命令或操作摘要
影响：预计中断范围和时长，是否影响同组其他流
保护：备份、校验与回滚方案
需要用户提供或确认：SSH 授权、API Key、摄像头测试流、维护窗口等
```

推荐措辞示例：

> 本地测试已通过。下一步需要只读 SSH 到 `sf01@14.21.88.97:21002`，检查
> `/home/sf01/rtsp-deepstream` 的 Compose 状态、实际镜像标签和最近日志，不会修改
> 文件或重启服务。请确认是否授权本次只读检查；如果需要登录凭据，请通过现有安全方式
> 完成认证，不要把密码写入仓库。

部署授权应再明确上传文件、备份配置、加载镜像、重建容器、预计中断和回滚镜像。用户只
授权“查看”时不能顺手修复；只授权“上传”时不能顺手重启。若执行工具另外弹出沙箱或
网络权限审批，仍需按工具流程申请，不能绕过。

## 11. 标准部署与回滚思路

不要机械复制旧文档中的包名。当前打包脚本默认生成：

```text
dist/rtsp-yolo-deepstream8-amd64.zip
```

标准流程：

1. 本地完整测试、编译检查和运行契约验证；
2. 确认必需模型/Profile 资产齐全，构建 `linux/amd64` 离线包；
3. 生成并核对 SHA-256；记录包名、Git 提交、镜像标签和变更摘要；
4. 向用户申请上传及维护窗口授权；
5. 服务器只读预检：磁盘、GPU、Docker、当前容器、当前配置、当前镜像和活动流；
6. 备份真实配置和当前 Compose/镜像信息，绝不以 example 覆盖现有密码；
7. 校验上传文件后再流式 `docker load`，运行 `docker compose ... config -q`；
8. 经用户授权重建服务；等待 engine-builder 成功，再验证 API、模型、流和公网播放；
9. 失败时用备份配置和原镜像/Compose 回滚，复核现有流是否需要业务方重建。

离线包很大：DeepStream 镜像未压缩约 23 GB、ZIP 历史约 13 GB。Mac 构建前建议至少
15 GB 空余；服务器流式加载建议至少 40 GB。不得静默覆盖同名包。只有纯 Python/模型
更新且本机已有正确基础镜像时才考虑 `REUSE_EXISTING_IMAGE=1`；先确认增量 Dockerfile
确实包含本次改动所需层。

## 12. 凭据与敏感数据

高优先级事实：`config/api.json` 当前被 Git 跟踪，并且 2026-08-20 的非回显检查表明
API Key、RTSP 发布密码和读取密码都不是 example 占位值。把它视为真实敏感配置：

- 不要在终端输出、回复、日志、diff、测试快照或文档中展示其值；
- 不要把摄像头 RTSP 用户名/密码、API Key 或 RTSP 密码写入新代码、Compose、镜像、
  命令历史和示例；
- 修改文件前先与用户确认凭据管理及轮换方案。简单加入 `.gitignore` 不能清除 Git
  历史；是否移出跟踪、重写历史和轮换线上凭据都属于需用户批准的安全变更；
- 日志代码必须继续对 RTSP URL 凭据脱敏；新增日志字段也要经过敏感信息审查；
- 不把真实事件截图、现场视频或摄像头地址上传到外部服务，除非用户明确授权用途和范围。

如果任务必须使用凭据，优先使用已配置的 SSH agent、受控环境变量或用户完成的安全认证；
只请求所需最小权限，不要求用户把秘密直接粘贴进 `AGENTS.md` 或提交到 Git。

## 13. 代码与 Git 工作约定

- 搜索优先使用 `rg` / `rg --files`；编辑使用最小、可审查的补丁。
- 开始前检查 `git status`；用户的未提交改动不得覆盖、还原或顺手格式化。
- 不使用 `git reset --hard`、递归删除、强推或改写历史，除非用户明确要求并确认风险。
- 不提交大模型、引擎、离线包、视频、事件数据或密钥。当前 `.gitignore` 已忽略
  `.venv/`、`*.pt`、`dist/` 等，但新增产物仍要检查。
- 新增配置字段时同步更新：Pydantic 请求模型、内部 Options/Spec、管理器序列化、
  DeepStream worker 配置、响应、示例 JSON、专题文档和测试。
- 新增旁路能力必须证明：容量有界、取最新帧、异常隔离、资源释放、主链指标不退化。
- 不宣称“修复卡顿/马赛克/准确率”而只提供单元测试；给出对应真实码流证据。
- 完成后报告改动文件、测试结果、未执行的服务器验证和仍存在的风险。

## 14. 常见误区

1. 降低模型阈值不是所有漏检的答案；先检查 ROI、输入尺寸、目标像素和现场域差异。
2. 降低输出 FPS 或复制旧帧不能伪装流畅。`unique_publish_fps` 才是关键，
   `duplicate_publish_fps` 应为 0。
3. 输入流网络、摄像头 GOP 和播放器缓存也影响端到端延迟，不能全归因于推理耗时。
4. `mosaic_risk` 是坏帧/不连续缓冲的观测风险，不等同于人工已看到马赛克。
5. ROI 坐标是归一化值；同机位改分辨率通常可复用，但移动摄像头、变焦、云台预置位
   或改变红外/彩色模式后必须重新标定。
6. 船舶分区推理每增加一个区域近似增加一份旁路计算；不能无视 GPU 余量无限加区域。
7. 只把旁路参数写成 2K/4K 不能恢复 mux 阶段已经丢失的细节。
8. 事件、捕捞风险和燃气瓶计数都有业务/现场假设，不能把历史样例数字当通用精度。

## 15. 如何维护这份“工作区记忆”

2026-09-09 荷兴广场零散垃圾为独立新场景，用户明确要求拿来即用、无需训练的公开模型。
本地两个 PS 录像完成 YOLO-World、街景垃圾堆及通用 YOLO26s 共 3 款权重抽样对照，
目前无已验证可用的推荐模型；原“YOLO-World 最佳”的表述已撤回。详见
`LITTER_MODEL_EVALUATION.md`。分块仅测试 2×2、每块宽高62.5%，不能推广为所有分块均无效。
没有严格真值召回率，不能将有框帧数当识别率；不接垃圾堆/事件链路，不连接生产服务器。
同日新增下载 `BaraaLazkani/trash-detection-yolov8` 的五类 YOLOv8m 权重，收窄到左侧
商户人行道并排除垃圾桶/店铺固定区后，夜间31帧、白天29帧均无有效零散垃圾框；整幅
裁剪出现大面积 Metal/Plastic 错框。该权重为 Custom Academic License，禁止商业使用，
不能作为生产推荐。详见 `output/litter_eval_trash_yolov8m.json` 和评估报告。

2026-09-09 同日继续本地筛查：累计11款不同权重完成推理，其中10款有昼夜抽样、Bower
仅白天代表帧初筛；包括COCO对照和原垃圾堆模型。另下载Jhandry/TrashDetection但因
RetinaNet依赖未完成加载，不计为已测或0检出。本轮Turhancan、CatSat、Esapzoi和原生
LitterCam YOLOv9-C均完成同一组58帧、4块640原生像素裁剪复测，取消旧固定排除区；
旧排除区覆盖部分地面真目标，旧PT脚本实际是整图1280后过滤，不能称ROI裁剪推理。
助手目视选取5个地面物24次出现，阈值0.15、位置IoU≥0.3点检分别命中20/9/1/0次；
这是筛查后选出的重复目标局部点检，未经人类确认，不是整体准确率/召回率。
当前优先实验候选为Turhancan YOLOv8m-seg+分块；原CatSat最佳表述撤回。它仍误报人脸、
电动车/固定物，无已验证稳定拿来即用的推荐成品。HF的MIT/Apache为发布者声明，不能
替代底层代码/数据许可审计。来源校验见`models/litter/manifest.json`，点检和误报图见
`output/litter_review/`，完整方法/边界见`LITTER_MODEL_EVALUATION.md`。新建的离线评测
脚本没有接入API、事件或生产旁路；没有训练、上传录像、连接服务器或部署。当前用户
Python3.12环境下284项单测通过；项目.venv全量尝试因共享推理线程等待/超时中断，未修复
该环境问题。脚本真实推理、点检和编译检查通过。

2026-09-09 用户授权清理本次模型评测产物，仅保留当前优先的Turhancan权重。已清理8款
淘汰权重、批量截图、抽帧缓存、临时YOLOv9代码/依赖和解压视频副本；用户原始ZIP已验证
完整性并保留，原有项目模型、其它任务产物和共享Python环境未作清理。评测JSON、少量
关键对照图、4个评测/点检脚本和来源manifest保留。历史模型表不代表当前全部权重在盘；
复跑需重新解压/抽帧，淘汰模型需重新下载。空间统计见`output/litter_review/cleanup_summary.json`。

2026-09-07 后续用户允许突然加速/出画风险时小幅保护性拉远，正常航行不应频繁触发。
已新增默认关闭的 `tracking_edge_guard_enabled`，仅替换现有持续跟踪阶段的变焦策略，
不等于完整 demo profile。详见 `TRACKING_EDGE_GUARD.md`：正常航行固定模拟 24/24
无出画无缩小，但突发加速与压力仍有失败；真实设备、完整 L1/L2 均未通过验收。

2026-09-07 本地连续跟踪方案基础修复见 `DEMO_IMPLEMENTATION_PROGRESS.md`：
旁路逐目标实测/保留属性、独立新鲜度、异步视角代际拒收、近景大框过滤保留已实现；
经单独本地写权限批准，关联 camera_control 加入 STOP 队列取消和短 SDK 动作代际检查。
尚未实现 demo profile、完整连续会话和延迟闭环，L1 未完成、L2 未验证；不得将本次
基础修复视作演示版已交付。没有连接服务器或真实摄像头，也没有构建或部署镜像。

2026-09-07 本地继续实施已补充 demo 控制故障锁存、动作完成后的边缘保护确认、旧视角
代际拒收、同船连续性重捕获、失锁持位会话、忙时跳过证据旁路、PTZ 增量 Dockerfile
模块复制和结构化脱敏。完整 RTSP 单测 280 项通过，延迟闭环与边缘保护模拟仍分别有
压力失败（详见 `DEMO_IMPLEMENTATION_PROGRESS.md`）；这仍不是 L1/L2 完成或真机效果
验收。

每次完成会改变项目事实的工作，都应同时更新本文件中相关摘要或明确链接：

- 生产部署地址、硬件、镜像或端口变化；
- 新增/删除 API、模型、旁路或持久化数据；
- 性能验收、现场准确率验收或已知失败模式出现新证据；
- 权限流程、维护窗口、回滚方法或凭据管理发生变化；
- 旧文档已过期或权威文档入口改变。

更新时写明日期和证据来源。不要把临时猜测、一次未复现现象、真实密码、私人 SSH
材料或大段聊天记录写进本文件。若用户提到“我们以前聊过”但仓库没有可验证记录，
应明确指出缺失的记忆并向用户确认，然后把确认后的稳定结论写入这里。


2026-09-10 荷兴广场实施设计更新：商户绑定在摄像头的zones区域层，一台摄像头可多商户；
通知暂不实施。同垃圾需要持续识别但只创建一个item_id，禁止按5分钟周期或单纯漏检超时
重新上报。已新增SQLite身份原型ground_litter_inventory.py；清理须局部可见空地证据，
断流/遮挡/重启不能视为清理；证据生成尚未实现。用户授权下再次读取五路各一张原图，均为
2560×1440，画面时间2026-09-10 15:31左右；未保存临时RTSP地址，未SSH/部署/创建线上流。
已画出五路独立地面区域草案，商户仍待绑定；配置schema2仅供本地校准，不是API请求。
原生分块有效ROI覆盖由42块优化至21块（6/4/4/3/4），执行真实本地推理；仍见座椅、井盖
等误报，不能当精度验收。旧22/13/7/14时序确认数及“5参考物均出现”不作选型依据，原因见
LITTER_MODEL_EVALUATION.md勘误。完整方案、场景图、当前实现/待实现清单及验收计划见
GROUND_LITTER_IMPLEMENTATION.md。

本次当前用户Python3.12完整unittest为303项通过（19.460秒），其中垃圾规则/库存身份19项；
日志output/litter_camera_calibration/unittest.log。独立本地测试不等同于五路实时GPU验收。

2026-09-11 零散垃圾独立影子批量闭环已补齐：`ground_litter_batch.py` 和
`scripts/evaluate_ground_litter_batch.py` 在有限本地回放上复用原生分块、ROI/排除区、人车
遮挡、时序确认和SQLite身份，生成JSONL、候选帧、证据图、复核页和下载标签；离线评估不
执行实时2秒新鲜度过滤，生产runner仍执行。1021日间/夜间各120帧、0.5 FPS重测结果和
边界见`GROUND_LITTER_IMPLEMENTATION_EXECUTION_20260911.md`，产物仅保留
`output/litter_batch_1021_20260911_rerun/`（约22MB）；错误区域的1030调试产物已清理。
日间唯一持续记录目视为固定设施误报，夜间候选较多但无持续记录；无已知真值，不能计算
准确率。当前Python3.12完整unittest为332项通过，compileall通过。未接入生产、未上传、
未重启容器、未发送通知。OpenAPI另有`/p-api/v1/monitor/play/replay`但缺少每路
`ip/channel`映射，`ctseelink`返回流尚未证明历史时间一致，不能当回放验收。

2026-09-12 用户继续授权完成 1021 路独立 GPU 对照：RTX 3060 Ti 临时容器内以同一昼夜
录像分别运行 ORB/SIFT。白天两者均 142 抽样/140 有效观测、138 候选帧、226 候选总数、
1 条持续记录；夜间 ORB 为 152/150、102/142、0 条记录，SIFT 为 152/152、102/142、
0 条记录。SIFT 夜间视角拒绝 0 次（ORB 2 次），但推理 P50/P95 约 0.977/1.311 秒，
ORB 约 0.229/0.307 或 0.271/0.403 秒；SIFT 只改善校验，不改善垃圾误报。白天持续记录
确认是固定红色清洁桶/设施，夜间候选主要是电动车/固定设施/反光物，录像没有垃圾摆放
真值，不能计算准确率或召回率。1021 直播 180 秒完整日志为 145 有效观测、7 候选帧/7
候选框、0 条记录；主要过滤原因是人车重叠176、超出ROI399、局部遮挡35，0条记录不等于
没有垃圾。固定红色设施实验排除区使白天候选帧138→67、记录1→0，夜间候选不变；仅为
消融证据，未写入正式 profile。服务器四个生产容器 ID/启动时间/重启次数前后不变，试验
容器已退出。证据摘要与复核图见 `GROUND_LITTER_IMPLEMENTATION_EXECUTION_20260911.md`
和 `output/litter_validation_20260912/`；完整单测342项通过、compileall通过。当前仍未
完成历史回放时间一致性、五路并行吞吐和现场摆放/清理真值验收。

2026-09-12 本地实施阶段 A-C 准备已完成：新增 `ground_litter_acceptance.py`、
`scripts/prepare_ground_litter_acceptance.py`、`scripts/evaluate_ground_litter_acceptance.py`，
生成 `output/ground_litter_acceptance_20260912/`。准备包冻结 49 个代码/配置/模型/参考图
指纹，包含五路昼夜采集清单、现场摆放记录模板和 10 份空白人工真值模板；验收器绑定录像
SHA-256、机位核验、时间锚点、日志快照和 item_id 复核，缺任一条件即输出“无法计算”，
不会从候选框推断真值。当前 Python3.12 完整 unittest 为 358 项通过，compileall 通过；
没有新增服务器连接、上传、生产配置修改或通知。下一步需历史回放可验证接口或现场摆放
资源，再按 `GROUND_LITTER_IMPLEMENTATION_NEXT_STEPS_20260912.md` 执行。

同日回放采样器修正：`sample_stream` 现在会按回放接口返回的 `offsetSeconds` 丢弃流起始
前置帧，并在 `manifest.json`/`recording.json` 中记录偏移和 `content_time_verified=false`。
这只减少起点偏移风险，不能证明 OSD/画面时间与请求时刻一致；仍需人工时间锚点验收。

2026-09-13 用户要求查看四小时 1021 直播影子结果，已只读核查和下载完整轮转日志：
北京时间 02:15—06:15 运行结束，退出码124来自 timeout，run_stopped/summary完整；
6,129有效观测、4,364拒绝、7,971候选框次数、0 item_id/0 cleared，最长候选窗口23.928秒，
未达到45秒。实际拍到05:30左右至06:00后的保洁与垃圾消失，已可用这些现成样本回归，
不再以缺少现场摆放为唯一进展条件。此前红色稳定候选是扫把头误报；白色物在日志04:52后
不再检出但OSD05:44仍在，05:49已消失，证明持续漏检，不能用最后检出反推清扫时间。
挂载源码仍为整图人车1280输入，“已上传640优化”应撤回；本轮独立新SQLite未承接旧ID，
也未配置clean_reference_image，因此等待不能自动完成清理闭环。另见影子输入灰块坏帧
及OSD/采集时钟差，结果年龄仅取帧后局部指标。生产四容器ID/启动时间/重启数未变，健康正常；
当前试点已停，未新启任务。详见 `output/litter_4h_review_20260913/REPORT.md` 与同目录
`server_audit.json`、`log_analysis.json`、`before_cleaning_after.jpg`。本次未修改运行代码。

2026-09-13 根据四小时实测制定下一轮计划，当前执行入口为
`GROUND_LITTER_EXECUTION_PLAN_20260913.md`：先冻结运行版本/已有真实清扫样本，定位输入
时延及白色物漏检/扫把误报，修复身份和局部清理证据，再做1021有界连续验证，达标后扩五路。
无需用户训练、逐帧标注或盯保洁；助手目视标签须标初步复核，稀疏图片不能当连续时序验收，
清扫后参考不能倒灌过去以声称在线成功。已有服务器隔离试验授权继续适用，生产容器不得受影响。
本次仅新增计划及文档路由，未修改识别代码、连接服务器或启动新监控；计划目标不是实测成绩。

2026-09-13 用户授权开始实施后完成首批：`output/litter_next_20260913/INDEX.md`是当前日志入口，
`REPORT.md`为结果。34张同图做CPU/GPU各1280/640对照，共136次静态分析；四小时实际runtime/profile
散列已复现。重要勘误：棚边白色物在18张非明显损坏可见样本均有原始候选，后9张被outside_roi
过滤（框中心进入棚顶排除区）；地面/棚顶归属不确定，不能继续称明确区内模型漏检或垃圾召回率。
消失区间OSD修正为05:42:05仍有、05:44:44已无。扫把误报仍在；GPU640另增4个候选，含桶/电动车
误报，保留1280默认。更小地面碎屑的定点放大/低阈值探测仍不可靠，P2未通过，未启动长期直播/五路。
已增加有界逐层诊断、分段耗时、拒绝样本计时、运行文件/模型指纹；本地另修短暂未知暂停但不累计
证据时间、显式跨run库存路径、重启置不可判断及源状态不推进帧时钟。最终380项单测、compileall和`git diff --check`通过；新增固定物过滤及取流诊断结果见`output/litter_next_20260913/filter_r2/`。1021直播PyAV基线60秒仅发布47帧且端到端新鲜度未验证；补充OpenCV回退探针60秒发布50帧、10分钟发布489帧，发布间隔P50/P95/最大约1.05/1.96/2.51秒，但镜像缺少PyAV、无PTS，仍不能启动长期监控。
两轮GPU仅部署诊断快照，随后本地状态修复未部署/未完成真实清理验收；版本分别见gpu_release_manifest
与local_final_manifest。授权隔离目录releases/diagnostics-20260913-r1及active/diagnostics-20260913-r1；
两次限速离线容器均正常退出自动删除。已有一路生产流68次采样唯一帧率24.202–25.849、duplicate0、
健康true；四生产容器ID/启动/重启数前后一致。只证明约3.5分钟离线共存，不是五路实时通过。

2026-09-13 继续取流诊断：最新入口仍为`output/litter_next_20260913/INDEX.md`，本轮详见
`source_r3/REPORT.md`。10分钟OpenCV源探针解码14980帧、邮箱发布489帧，约0.815 FPS；
这不是有效模型分析频率。OpenCV不提供PTS/坏帧标记，旧计数0不代表零坏帧；早期PyAV抽样帧
PTS相隔5.7秒不能推断逐帧时间戳跳变。后续30分钟源探针已主动停止，无完整摘要，未通过验收。
生产流`bd1b5e85ab6442eb82631e1680092c0f`返回的0.0466 FPS及健康false实际最后更新于北京时间
12:16:57（UTC04:16:57），早于该次试验；不能作为当前帧率，也不能据此归因于试验。
生产容器ID/启动时间/重启数未变，未重建生产流。健康无法验证时暂停直播试验。
新增有界原始解码时间线、5秒进度落盘、PyAV强制检查、代码指纹与SIGTERM安全退出；修复信号
函数操作multiprocessing.Event锁导致退出卡住，本地PS复测约0.185秒退出并保留摘要。新增
`scripts/guard_ground_litter_source_probe.py`对生产指标过期/异常实施启动前和运行中保护，
只停止自己的唯一命名容器；服务器实际预检已返回`production_metrics_stale_or_unknown`，
在建容器前拒绝运行。全量387项单测、compileall及diff检查通过。本地PS使用PyAV18.1.0，
不能替代项目约定的16.1.0服务器兼容性验收。隔离PyAV16.1.0源码/wheel包已上传校验至
`releases/source-r3-20260913`，依赖不装入生产容器；候选镜像构建/加载结果以本轮
`offline_build.json`和报告为准，不能把guard预检的`image_id=null`当成镜像通过。
P1完整输入/模型链、P2小碎屑/语义误报及P3真实清理仍未通过，通知及长期监控继续关闭。
2026-09-14核查确认生产worker持续收到RTSP EOF并重置源，指标文件陈旧，API的running不能证明画面健康。输入设备不在用户五路设备内；“旧演示流”用途未经证实。撤回先前“地址已过期约25小时、根因确定”的结论：TimeStamp可能为签发/签名时间，没有接口到期语义不能判过期；EOF原因仍未确定。`inspect_rtsp_url_expiry`已修正为仅报告时间提示，不拒绝URL。未重启生产容器、未改配置；历史诊断见`output/litter_next_20260913/source_r3/REPORT.md`。

2026-09-14用户复核确认1021夜间样本中的白色小物体为独立零散垃圾。结合此前用户确认的两个候选物体，本轮已知独立垃圾至少3件：2件进入候选，1件因约8像素尺寸规则被过滤；有限样本最终记录召回2/3（66.7%），不能外推全天或其他机位。第二个约8×6像素亮点仍未确认。证据见`output/litter_source_20260914/deduplicated_labels.json`、`REPORT.md`。
同日最终完成独立镜像`ground-litter-source-probe:pyav16-20260913`（ID前缀`630891dd1d05`）
的离线构建及PyAV16.1.0加载；服务器昼夜PS各20秒均解码485帧、邮箱发布19帧、PTS/坏帧标记
异常0，均正常退出。无网络/GPU、0.5 CPU/512MiB，仅证明离线兼容。修复移除容器权限后结果
目录不可写的问题：外部启动器现在使用宿主UID/GID，最终版本`releases/source-r3-20260913-r1`
与本地`source_r3/release-r1/`一致，不覆写原版本。19:29:23北京时间最终门禁仍因生产指标停在
12:16:57而拒绝创建直播试验容器，P1未通过。最终387项测试日志为`source_r3/full_tests_final.log`；
镜像与离线实测见`offline_build.json`、`offline_validation.json`，生产容器未修改或重启。

2026-09-15 零散垃圾识别已接入 `/v1/streams` 作为独立场景（本地实现，未部署）：新增
`ground_litter_detection.py`（选项/结果缓存/显示层/原生分块检测器）与
`ground_litter_process.py`（lossy 旁路进程与客户端），并在 `api.py`、`stream_manager.py`、
`deepstream_manager.py`、`deepstream_worker.py` 中接线。请求块与 `license_plate`/
`vessel_detection` 同级，一次调用返回的 `rtsp_url` 即"地面区域轮廓+疑似垃圾框"输出流，
不产生 `item_id`/清理状态/通知，且不能与 `ptz_verification`（云台运动会破坏固定地面区域）
同时启用。worker 侧新增 `analytics_tee` 分支：`queue(max-size-buffers=1, leaky=2)` →
`nvvideoconvert` → `capsfilter(video/x-raw(memory:NVMM), format=RGB)`（**刻意不写宽高**，
保持 mux 原生分辨率，分块推理才有意义）→ `appsink(drop=True)`；人车遮挡默认复用主链已跟踪
目标，可选 `actor_model` 独立推理；显示层要求 `hit_window` 内命中 `minimum_hits` 次并在最后
一次命中后保持 `hold_seconds` 秒，避免 1 FPS 分析在输出流上闪烁。新增测试 72 项，全量
467 项通过（`python3.12 -m unittest discover -s tests`），compileall 通过。新增增量镜像
`Dockerfile.deepstream.ground-litter-update`（复制 4 改动模块+2 新模块+
`ground_litter_geometry.py`+唯一一份已复核权重，不重建 TensorRT 引擎），并在
`scripts/validate_deepstream_runtime_contract.py` 加入该分支的真实 ServiceMaker 构图检查
（只能在带 DeepStream 的镜像/服务器运行）。**未构建、未上传、未部署、未创建线上流**；
`config/ground_litter_1021_demo.json` 是 9/15 生成、仍未被 git 跟踪的单路样板区域配置，
区域与像素门槛必须按本地校准流程逐路核对后才能用于现场。本轮没有服务器连接。

2026-09-15 零散垃圾场景本地干跑与构建策略（同日晚些补充）：用户本机 Docker 已可用，但仍
**必须在服务器上构建镜像**，理由已实测：本地 `rtsp-yolo-annotator:deepstream8-amd64`
创建于 2026-08-02，容器内实测只有 numpy 1.26.4/cupy/pillow/fastapi，**没有 cv2、torch、
ultralytics**，而线上运行的是 2026-09-10 的 `…-demo-continuous-20260910`；用本地旧基础镜像
构建既会回退生产代码，`docker save` 又要传约 23 GB（新增层仅约 55 MB）。`Dockerfile.deepstream.ground-litter-update`
现已内置基础镜像依赖断言（import cv2/numpy/torch/ultralytics + `build_ground_litter_tiles` 构图检查），
在旧基础镜像上按预期在构建期失败并打印缺失项——即"构建期失败而不是运行期才发现"。另注意
本机 arm64、生产镜像 amd64：构建必须带 `--platform linux/amd64`，否则 buildx 会去 registry
找 arm64 变体并报 `pull access denied`。1021 区域核对产物在
`output/ground_litter_1021_zones_20260915/`（含昼夜区域图、实测帧已知物体对照图与
几何量化 JSON）：三个已知垃圾位置都在区内且都过 8px/64px² 门槛，昼夜参考图位移 1.09px、
一套多边形成立，1021 该区域只需 5 个 640 分块；但棚顶排除区吃掉 merchant_04 的 17%、
merchant_05 的 8.3%，且四小时那件白色垃圾正贴该边界（1–2px 临界，会闪进闪出），需用户目视确认
后决定是否内收边界。上传包与部署单见 `dist/ground-litter-20260915/`、`GROUND_LITTER_DEPLOY_20260915.md`。
本轮仍未上传、未在服务器构建、未部署、未创建线上流。

2026-09-15 零散垃圾场景已在服务器完成 1021 单路试点部署（首次真实上线）：
新镜像 `rtsp-yolo-annotator:deepstream8-amd64-ground-litter-20260915`（manifest 前缀 `49c0149c4222`），
基础镜像用预检得到的线上 `…-demo-continuous-20260910`（`8f4bf64b95f4`），回滚标签
`deepstream8-before-ground-litter-20260915` 指向同一旧镜像；Compose 在原三条 `-f` 后新增
`docker-compose.ground-litter.override.yml`；只 `up -d --no-deps api`，MediaMTX、camera-control、
web-gateway 的启动时间与重启次数均未变。构建上下文放
`releases/ground-litter-20260915/`（`models/` 属 root，未写入；权重与代码包 SHA-256 已核对）。
试点流 `498de92c72974807b0b9fdaa2eec8555`：主链 `publish=25.01 FPS`、`duplicate=0`、
`pipeline_healthy=true`、坏帧 0；垃圾旁路 `state=running`、约 1 FPS、单帧 33ms、异常 0、
`tile_count=2`；输出流容器内解码 120 帧 1920×1080 成功，抓图与颜色校验见
`output/ground_litter_1021_pilot/`。重启 API 清掉了原先那个源已 EOF 的僵尸流 `bd1b5e85`
（业务方如需继续需重建）。**现场关键事实**：生产 mux 是 1920×1080 而摄像头原生 2560×1440，
旁路只能拿到 mux 缩放后的 1080p，因此试点门槛按 0.75 缩放为 6px/36px²（与原生 8px/64px² 物理等价，
否则已确认的那件 8×8px 垃圾会退化成 6×6px 被过滤）；若要让旁路真正用原生像素，需要单独评估把
`mux_width/height` 提到 2560×1440 的全局影响。执行中发现并修复了一个真实缺陷：
`actor_boxes` 返回 6 列预测行而重叠过滤只解包 4 个坐标，导致启用 `actor_model` 时旁路每帧抛异常
（主链不受影响）；已改为 `row[:4]` 并补回归测试，第一次部署镜像即该缺陷版本、已用修复版覆盖。

2026-09-15 应现场反馈新增 `display_detections`（顶层布尔，默认`true`）：只控制输出画面是否
绘制普通检测框（人/车），关闭后仅保留业务叠加（例如地面区域轮廓与垃圾框）。经只读核对
运行容器的 `nvinfer.txt` 确认 `num-detected-classes=80` 且无类别过滤，因此**`classes` 与
`display_detections` 都只是显示过滤**：元数据里始终有全部类别，跟踪、事件状态机、
`interval_detections` 统计和 `ground_litter` 的人车遮挡判定都不受影响（`_hide_object`
只把 border_width 置 0、清空文字）。据此更正此前"`classes:[0]`会让车辆不在元数据里"的
错误说法：`ground_litter.actor_model` 并非必需，主链元数据已含人车框，保留它的唯一差别是
遮挡判定阈值（主链 `conf=0.35` 起，独立 actor 模型默认 0.20）。实现落在
`api.py`、`stream_manager.py`（StreamSpec）、`deepstream_manager.py`（worker payload）、
`deepstream_worker.py`（StreamPolicy + 绘制处 gate，不改 `_object_allowed` 以免影响统计与
PTZ 船框收集）；新增/更新测试后全量 473 项通过，镜像已重新构建并重建 API 容器
（MediaMTX/camera-control/web-gateway 未动）。**现场验证被阻断**：用户提供的 1021 输入地址
在 11:54 已返回 `401 Unauthorized`（token 过期），无法做"无人员框"的实拍确认；已删除该
取不到流的空流，避免再产生刷重连的僵尸流。`display_detections` 的实际观感仍需给一条新鲜
有效的摄像头地址后复核。同日另一个待办：用户截图里左下角店铺门口存在 `Paper 0.293`、
149×135px 的固定物误报；实测已确认真垃圾置信度在 mux 1080p 下只有 0.10–0.16（原生 2560
下 24/31 帧 ≥0.20），**不能靠提高 confidence 修**；建议对 merchant_04/05 加
`exclude_zones: [[[0.070,0.700],[0.160,0.700],[0.160,0.850],[0.070,0.850]]]`
（紧贴版：误报消失、地面 −11.2%、三件已确认垃圾均不受影响），渲染图在
`output/ground_litter_1021_fp_fix/`；用户尚未选定该方案。

2026-09-15 零散垃圾新增分区级"最大尺寸上限"（`maximum_short_side_px`/`maximum_box_area_px`，0=不限制）
并已部署。起因：用户实测流 `dc431458…`（320分块）稳定显示 3–4 个框，经用户目视确认**全是误报**
——左下三个是**停放电动车**、另一个是**蔬菜摊的菜**。逐条排查结论：
(1) **人车遮挡过滤对此无解**：整图人车模型(1280)在左下区域 0 个框，降到 conf 0.05 或提到
imgsz 1920 也只有 0.06/0.10 的 truck 噪声；离线试点里的"局部640补检"同样检不出（局部只检出
画面中部 y234–533 的 motorcycle，与 y>769 的候选零重叠）。所以这是 COCO 模型对俯视停放电动车的
盲区，不是阈值问题。
(2) **真垃圾与这类误报在尺寸上可分**（1080p）：误报面积 3040/7205/14036/28482px²，三件已确认
真垃圾 66/462/1666px²。因此把 `maximum_box_area_px` 设为 **2500** 可在**不挖任何地面区域**的前提下
清掉全部 4 个误报（实测该帧 4→0，拒绝原因记为 `too_large`）；6 张实拍输出画面的每帧候选从
1.83 降到 0.17。代价：**大于 ~50×50px 的垃圾（大袋/成堆）不再上报**，需要报大件时应调到 3000，
但 3000 距蔬菜误报 3040 太近，风险高。
(3) **同时把 `confidence` 从 0.20 降到 0.12** 才让已确认的小垃圾真正显示：9/14 夜间样本在 320 分块下，
`conf 0.20` → 5 帧 0 候选；`0.14` → 只过 1/5 帧、显示层 2/3 投票后仍不显示；`0.12` → 每帧 1 个，
显示层 **4/5 帧显示**那件 8×8 已确认垃圾。0.14 这个"看起来合理"的值实际是失败点。
实现落在 `ground_litter_detection.py`（Zone 两个字段+校验+资格过滤+`too_large` 原因）与
`api.py`（请求模型两字段）；新增 7 项测试后全量 **480 项通过**，镜像已重建并重建 API 容器
（MediaMTX/camera-control/web-gateway 未动）。注意：本轮部署再次重启 API，用户的活动流
`dc431458…` 已随进程消失，需用新鲜有效的摄像头地址重建（旧 token 又会过期）。
另修正两个我自己的测量错误以免后人重犯：渲染脚本 `_zone_from_profile` 起初**没有透传**新增的
最大尺寸字段，导致第一次"上限生效"的测量无效；以及把归一化中心当作像素坐标比较，导致
"已确认垃圾是否被显示"的第一次统计全为 0。

2026-09-15 按用户要求**回滚了"最大尺寸上限"这一步**（同日撤销）：`GroundLitterZone` 的
`maximum_short_side_px`/`maximum_box_area_px`、`api.py` 请求模型对应字段、资格过滤里的上限判断与
`too_large` 原因、以及随它引入的有界 `rejected_candidates` 诊断列表全部移除；渲染脚本
`_zone_from_profile` 的透传也一并撤回（否则会向已删除字段传参）。测试从 480 回到 **473 项通过**，
compileall 通过。**保留** `ground_litter` 与 `display_detections` 两项。示例与交付 JSON 里的
`maximum_box_area_px` 已清除；线上用真实请求验证旧字段现在返回 `extra_forbidden`(422)。
镜像重新构建并重建 API 容器（当前 `…-ground-litter-20260915` 镜像ID前缀 `efd099d`→新构建），
MediaMTX/camera-control/web-gateway 启动时间与重启次数未变；含尺寸上限的上一版已另存标签
`rtsp-yolo-annotator:deepstream8-ground-litter-with-maxsize-20260915` 作为**回滚点**，需要时可以
用同一个 override 文件切回去。**回滚带来的现场后果**：电动车×3 + 蔬菜这 4 个大框误报（Plastic/Paper
0.20–0.73）会重新出现，尤其在 `confidence: 0.12` 下；如果用户不想看到它们，剩下的手段只有
`exclude_zones`（左下条带 `[[0.070,0.700],[0.160,0.700],[0.160,0.850],[0.070,0.850]]`，地面 −11.2%）
或把区域左边界收到 x≥0.16（地面 −26%，三件已确认真垃圾仍在内）。API 重启后活动流为 0，
用户表示会自行用旧版接口重建。
