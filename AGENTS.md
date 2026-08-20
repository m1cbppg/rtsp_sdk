# RTSP 项目 AI 交接与工作约定

本文件适用于本仓库全部目录。它的目标不是替代专题文档，而是让第一次进入本仓库的
AI 能先恢复项目上下文、知道哪些结论已经验证、哪些仍需现场验证，以及什么时候必须
向用户申请服务器权限。

最后一次本地核对：2026-08-20。编写时本地分支为 `main`，HEAD 为 `351d568`，与
`origin/main` 一致且工作区干净。这里的提交号只表示本文件编写时的基线；开始新任务
时仍必须重新执行 `git status --short --branch`，不能把本段当作永远有效的状态。

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
- `fishing_risk.py`：基于船舶轨迹的风险评分规则；
- `labels.py`：中文标签和字体。

持久化和临时状态：

- `data/events`：事件 JSON 与截图，删除流不会自动删除历史事件；
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
- 车牌：`LICENSE_PLATE.md`；夜间：`NIGHT_VISION.md`；
- 燃气瓶：`GAS_CYLINDER.md`；船舶：`VESSEL_DETECTION.md`；
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

每次完成会改变项目事实的工作，都应同时更新本文件中相关摘要或明确链接：

- 生产部署地址、硬件、镜像或端口变化；
- 新增/删除 API、模型、旁路或持久化数据；
- 性能验收、现场准确率验收或已知失败模式出现新证据；
- 权限流程、维护窗口、回滚方法或凭据管理发生变化；
- 旧文档已过期或权威文档入口改变。

更新时写明日期和证据来源。不要把临时猜测、一次未复现现象、真实密码、私人 SSH
材料或大段聊天记录写进本文件。若用户提到“我们以前聊过”但仓库没有可验证记录，
应明确指出缺失的记忆并向用户确认，然后把确认后的稳定结论写入这里。
