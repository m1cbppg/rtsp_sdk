# 船舶识别部署核查（2026-09-10）

## 最终部署验收：保留两个新镜像，不回滚

2026-09-10 后续经用户授权继续 SSH 检查和标准模式复测。最终保留已切换的新镜像，
本次续验没有重新构建、重新部署或重启任何服务，也没有执行回滚或真实摄像头控制。
下文“部署前只读核查”记录是同日更早的历史基线，不代表当前运行版本。

| 服务 | 最终运行镜像 | 完整镜像 ID（SHA-256） |
| --- | --- | --- |
| RTSP API | `rtsp-yolo-annotator:deepstream8-amd64-demo-continuous-20260910` | `8f4bf64b95f47ea551b66b3e6ec5f734c5769c71029386e0d8decc163693db88` |
| camera-control | `camera-control:dahua-sdk-command-cancel-20260910` | `c891e8d5379fa569586ca30a55a1b9f8251bf5f903c8ccdafc44c5cde45ab0f3` |

API 于北京时间 10:28:19 启动，camera-control 于 10:27:50 启动；最终复核均为
`RestartCount=0`。API 使用原主 Compose、`docker-compose.ptz-v12.override.yml` 加
`docker-compose.demo-continuous.override.yml`；camera-control 使用
`docker-compose.server.yml` 加 `docker-compose.camera-control-command-cancel.override.yml`。
运行容器的 RTSP 8 个及 camera-control 2 个更新模块逐文件 SHA-256 与当前本地代码一致。
本次增量范围仍为 RTSP 8 个模块、camera-control 的 `api.py`/`commands.py` 两个模块。

### 临时流没有指标的根因

最新失败流为 `4de7e21a8b884548a347bbb1af3ee4f1`，北京时间 10:42:56 创建、10:44:13 停止。
它之前还有 `15b0a3674a134d88bd859653f5e7ec17`（10:40:39–10:41:56）和已知的
`08d2c87e795144e68d17e17f8ebac443`。最新流的 worker 在约 0.6 秒内完成 TensorRT
引擎反序列化和 NvDCF 初始化，随后整个窗口都收到输入源 `404 Not Found`，约每 5 秒
重连一次。逐流日志没有指标，删除后 `exit_code=0`；并非脚本漏读已有指标或初始化太慢。

用 GStreamer Discoverer 只读检查原始 `boat-test.mp4`，实际视频为 **H.265、2702×1520、
30 FPS、174.522 秒**，带 AAC 音轨。之前本地保留的测试脚本固定使用 `h264parse`，
视频轨无法连接；短时发布器诊断显示管线虽为 PLAYING，视频缓冲计数仍为 0。
因此仅检测发布器进程存活会误判输入已就绪，MediaMTX 实际没有可读视频路径。
前一次普通路径鉴权 401 是另一项已知测试错误，不能与公网其他 401 混为一谈。
精确时间段的 API worker 日志提供了上述 404 证据；MediaMTX 的 `docker logs` 查询
没有返回该窗口的日志，未将其解释为“所有访问均正常”。

### 唯一一次修正后的标准流复测

只修正测试发布器的视频解析器为 `h265parse`，保留按时间戳限速和音轨消耗。
诊断与测试代码通过 SSH 标准输入临时执行，没有安装到部署目录或镜像中。
先独立 NVDEC 解码输入 62 帧，确认源可用后才创建业务流：

- stream_id：`1b6ca7c7a17e4bc5bbe7464d636b460c`。
- 北京时间 11:01:14 创建，11:03:15 删除；观察 120.6 秒。
- 主模型 `yolo26s.pt`、输入 640；船舶旁路输入 1280、分析目标 5 FPS。
- 明确 `ptz_verification.enabled=false`、`tracking_profile=standard`、边缘保护关闭；
  PTZ 运行指标一直为 `disabled`，没有真实摄像头动作。
- 启动预热时曾出现 `stalled` 和低 FPS；20–110 秒的 10 次稳定采样中，
  `publish_fps=unique_publish_fps=29.985–30.022`、`duplicate_publish_fps=0`，
  `pipeline_healthy=true`。worker 日志同时记录 pipeline/pre-encode/publish 约 30 FPS。
- 船舶旁路为 `running`，上述采样检出数 2–5、结果版本持续递增；不是准确率或船只真值计数。
- H.264 输出 RTSP 经独立 NVDEC 实际解码 **3,406 帧**；不是仅 depay/parse 到 fakesink，
  也不是仅检查播放器进程存活。测试脚本退出码 0。

### 清理、稳定观察及保留项

11:06:17（北京时间）及后续状态复核距临时流停止超过两分钟：API `/health`、
API 容器到 camera-control 的 `/healthz` 均为 `ok`，活动流数量 **0**，两个新容器
重启计数均为 0。查询本次启动以来的两个服务日志，未检出 ERROR、Traceback、SIGSEGV
或 Segmentation fault；失败试验的 404、引擎设备兼容提示等 WARNING 仍保留在日志中。

MediaMTX 和 web-gateway 容器 ID、启动时间与初检一致，均仍为 9 月 6 日 14:38:27
（北京时间）启动，`RestartCount=0`，续验未重启它们。MediaMTX 实际镜像 ID 为
`482fe138c9a16df19222e5449ab27af73d866fd1965217fe493358402000b257`，网关为
`d6d273e263b9066e138361c2ec9b1c964be75e5b357d5e6ec0111da2cd598c51`。

临时流已 DELETE，`/app/runtime` 无残留文件，测试发布器和 worker 已退出。
临时输入路径 `detected/deploy-audit-input-20260910` 及本次输出路径均通过带认证的
RTSP DESCRIBE 确认为 404。容器临时视频副本（112,329,981 字节）已删除；旧测试文件
亦不存在。宿主原始 `/home/sf01/boat-test.mp4` 保留，大小仍为 112,329,981 字节。
6 个演示采集/报告/模拟/导出脚本在宿主部署目录和 API 容器内均不存在，未上线。
逐流诊断 JSONL 保留在 `data/stream-logs/<stream_id>.jsonl`，用于追溯。

以下回滚镜像仍保留，未删除：

- `rtsp-yolo-annotator:before-demo-continuous-20260910` →
  `1add8299293c131abee033761f48ecfe96a3509e5b241b808c118ffb120dbf9a`。
- `camera-control:before-command-cancel-20260910` →
  `966377e2432590e529f5157bf50a8d8ed09143cf6b4a8d622c92f81a526dd522`。

### 验收边界与已有测试

本次证明新镜像的单路标准船舶识别、旁路、编码与容器网络内 RTSP 解码可用，支持保留部署。
没有重跑 10–30 分钟公网长播、多路组内增删、车牌/垃圾等全部业务，也未验证真机 PTZ。
`demo_continuous`、快速跟踪、方向/倍率/HOME、船号清晰度和现场准确率仍待内网验收。

部署交接记录的 camera-control 全量测试为 30 passed，RTSP 船舶/PTZ 回归为 163 passed，
此前专项为 23 passed；本次续验未重复这些单测。RTSP 全量 unittest 卡在未修改的
`tests/test_shared_inference.py::test_single_stream_does_not_wait_for_micro_batch_window`，
共享推理线程未退出，经 Ctrl-C 中断，**不能记为全量通过**。
模拟结果仍为普通过程 6/6、高延迟/加速 2/15、边缘保护普通 24/24、突然加速 6/12、
压力加速 0/12；本次标准模式通过不改变这些失败与未验收边界。

## 部署前只读核查（同日早期历史记录）

结论：相对实际运行容器，船舶 DeepStream/PTZ 链路有 **10 个核心代码文件未更新**，
其中 RTSP 服务 8 个（6 个修改、2 个新增），camera-control 2 个修改。
另有 **6 个演示采集/评估/导出脚本**在服务器部署目录和 API 容器中均不存在。
核心代码增量按逐行比较为新增 1,254 行、删除 23 行；行数不代表功能已验收。

本次通过用户授权的 SSH 只读核查，没有上传文件、构建镜像、重启服务、修改配置或控制摄像头。
比对基准为容器内实际文件的 SHA-256，而非 Git 未提交列表、默认镜像标签或历史文档。

## 部署前实际版本（历史）

| 项目 | 实测结果 |
| --- | --- |
| 服务器 | `sf01@14.21.88.97:21002` |
| 部署目录 | `/home/sf01/rtsp-deepstream` |
| API 容器 | `rtsp-yolo-api` |
| API 运行镜像 | `rtsp-yolo-annotator:deepstream8-amd64-ptz-v12` |
| API 镜像 ID | `1add8299293c131abee033761f48ecfe96a3509e5b241b808c118ffb120dbf9a` |
| API 镜像创建时间 | 2026-09-04 16:19:30 +08:00 |
| API 容器启动时间 | 2026-09-07 08:33:02 +08:00 |
| Compose 文件 | `docker-compose.deepstream.api.yml` + `docker-compose.ptz-v12.override.yml` |
| camera-control 镜像 | `camera-control:dahua-sdk-20260821` |
| camera-control 镜像 ID | `966377e2432590e529f5157bf50a8d8ed09143cf6b4a8d622c92f81a526dd522` |
| API `/health` | `status=ok` |
| camera-control `/healthz` | `status=ok`，从 API 容器访问 |
| 当前流任务 | `GET /v1/streams` 返回空列表 |

camera-control 镜像标签日期不等于最后更新时间。
API 运行代码没有宿主源码目录挂载，上传源码本身不会更新运行中的服务。
默认标签 `rtsp-yolo-annotator:deepstream8-amd64` 当前指向 `c25cf4053e94…`，
不是正在运行的 API 镜像。后续构建与回滚须保留实际 v12 镜像和 Compose override。

本地 `dist/ptz-persistent-tracking-recovery-20260904-v12.tar.gz` 的 SHA-256 为
`88fbf798be72ac4dcc148370a70bd3d60ef1b33a7c07514dd6e0b83bd25f5039`。
包内 4 个 Python 文件（api、deepstream_manager、deepstream_worker、ptz_verification）
均与运行容器逐字节一致，确认 v12 已部署。

线上 OpenAPI 已包含 `continuous_tracking`、`tracking_recovery_enabled` 和
`vessel_number_recognition_enabled`，尚无 `tracking_profile`、`tracking_edge_guard_enabled`。
接口存在仅证明代码契约存在，不证明设备跟踪效果或船号识别准确率。

## 10 个核心代码文件

| 本地相对路径 | 线上差异 | 作用 |
| --- | --- | --- |
| `rtsp/rtsp_annotator/api.py` | 修改 | 新演示模式、边缘保护的请求参数 |
| `rtsp/rtsp_annotator/deepstream_manager.py` | 修改 | 返回有效策略、保护参数 |
| `rtsp/rtsp_annotator/deepstream_worker.py` | 修改 | 主路观测来源、更新时间、视角代际 |
| `rtsp/rtsp_annotator/ptz_verification.py` | 修改 | 统一演示会话、异步取证、旧框过滤、失锁持位、保护集成 |
| `rtsp/rtsp_annotator/vessel_detection.py` | 修改 | 区分实测与保留显示框，拒收过期/乱序结果 |
| `rtsp/rtsp_annotator/vessel_detection_process.py` | 修改 | 拒收旧视角结果，保留近景大框过滤配置 |
| `rtsp/rtsp_annotator/continuous_tracking.py` | 缺失 | 最新控制意图调度、跟随中逐步放大、模拟器 |
| `rtsp/rtsp_annotator/tracking_edge_guard.py` | 缺失 | 有依据且受限的出画风险保护性拉远 |
| `camera_control/camera_control/api.py` | 修改 | 控制动作接入代际检查与 STOP 取消 |
| `camera_control/camera_control/commands.py` | 修改 | STOP 作废排队任务，取消与短 SDK 动作互斥 |

RTSP 路径根目录为 `/Users/mlcbppg/Desktop/backend/python_script`；
camera_control 路径根目录为 `/Users/mlcbppg/Desktop/backend`。

## 6 个辅助脚本

以下文件位于本地 `rtsp/scripts/`；不计入上面的 10 个运行代码文件：

- `check_demo_environment.py`：内网环境检查。
- `collect_demo_session.py`：被动采集任务信息。
- `build_demo_report.py`：离线日志报告。
- `export_demo_bundle.py`：脱敏导出。
- `evaluate_demo_continuous.py`：发现到近景的确定性模拟。
- `evaluate_tracking_edge_guard.py`：边缘保护策略模拟。

相关 Dockerfile、测试和说明文档也需随候选版本整理。辅助工具不要求全部放入生产镜像。
近期 `video_tools/` 中的离线视频处理和船号展示改动没有接入实时服务，不计为生产漏部署。
垃圾模型筛查脚本同样不在本次船舶链路范围内。

## 其他差异及计数口径

全量 Python 模块比较结果：RTSP 本地 33 个模块中 23 个一致、7 个修改、3 个线上缺失；
camera-control 13 个模块中 10 个一致、3 个修改。
共 13 个模块差异中，以下 3 个不计入本次 10 个核心待更新文件：

- `rtsp_annotator/pipeline.py`：Python 兼容链路的坏帧、帧间断等统计补充；当前运行的是 DeepStream。
- `rtsp_annotator/ptz_test_service.py`：隔离测试服务；生产 API 镜像缺少此模块不等于生产漏部署。
- `camera_control/controllers/mock.py`：Mock 对焦策略差异，不是大华真机驱动更新。

大华驱动 `controllers/dahua.py`、`models.py`、`leases.py` 与线上一致，
不能依据它们在本地 Git 中尚未提交就判定未部署。

## 本次复核与验收缺口

本次执行 `.venv/bin/python -B -m unittest tests.test_continuous_tracking
tests.test_tracking_edge_guard tests.test_tracking_observation_contract -q`，
23 项测试全部通过，耗时 29.038 秒；这是针对性复核，没有重新执行两个项目的全量回归。

本次实际重跑两个已有模拟器，结果为：

| 模拟 | 通过情况 |
| --- | --- |
| 完整过程：预定义低视频延迟普通场景 | 6/6 |
| 完整过程：高延迟/加速压力场景 | 2/15 |
| 局部边缘保护：普通航行 | 24/24 |
| 局部边缘保护：突然加速 | 6/12 |
| 局部边缘保护：压力加速 | 0/12 |

边缘保护脚本以退出码 1 结束，失败未忽略。完整过程脚本退出码 0 只检查普通场景，
不能据此说全部场景通过。上述均为检测桩/物理或策略模拟，没有执行真实 HTTP→PTZ→画面叠框闭环。
本次没有进行真实 RTSP 播放、真机动作或现场准确率验收；无活动流时健康接口正常也不能证明这些效果。

本地已有 `demo_continuous` 实现；进展文档后段仍保留“尚无 demo profile”等早期记录，
本次已以当前代码校正该结论。但构建与完整集成验收尚未完成，不能把代码上线等同于演示达标。

## 早期后续建议（已由顶部部署与续验记录更新）

当前不需要用户处理 SSH 登录或提供密码。先完成候选版本的 Linux/DeepStream 集成与失败场景复核，
明确实验模式的适用边界，再协调两个服务的更新。部署时需要使用实际 v12 基础镜像、保留真实配置与
Compose override、保存回滚镜像，只重建必要服务并复核流任务。
若当时已有活动流，API 重启会终止进程内任务，需要调用方重建；目前空列表只是本次检查时的状态。
真机效果仍需现场录像、控制日志及机型方向/倍率/HOME 的验证。

机器比对清单及模拟原始结果位于 `output/deployment_audit_20260910/`，不含真实配置或凭据。
