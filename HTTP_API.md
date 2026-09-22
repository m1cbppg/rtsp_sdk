# RTSP实时识别HTTP API

## 1. 工作方式

HTTP请求只负责创建、查询和停止识别任务，视频仍通过RTSP播放：

```text
POST /v1/streams
  → 启动独立拉流/画框/发布会话
  → 同一个.pt按每两路创建一个YOLO实例
  → 参数兼容的最新帧在极短窗口内动态batch推理
  → 生成detected/{stream_id}
  → 返回带只读账号的公网RTSP URL
```

API容器必须只运行一个Uvicorn worker。任务保存在该进程内存中；API容器重启
会停止全部任务，调用方需要重新创建。

CUDA模板允许最多8路，但这只是API保护上限，不代表硬件承诺。必须按照实际
模型、分辨率和FPS逐路压测，再把生产上限设置为稳定压测值的70%～80%。

## 2. 网络端口

如果API和RTSP都需要被外部系统直接访问，应让机房配置两条TCP映射，例如：

```text
14.21.88.97:38080/TCP → 192.168.1.3:8080/TCP   HTTP API
14.21.88.97:38554/TCP → 192.168.1.3:8554/TCP   RTSP播放
```

也可以只映射RTSP端口，HTTP API继续通过SSH隧道调用。

## 3. 配置

复制CUDA配置模板：

```bash
mkdir -p config
cp config/api.cuda.example.json config/api.json
nano config/api.json
```

配置结构：

```json
{
  "api": {
    "key": "替换为随机API密钥",
    "host": "0.0.0.0",
    "port": 8080
  },
  "models": {
    "root": "/app/models"
  },
  "rtsp": {
    "internal_base_url": "rtsp://mediamtx:8554",
    "public_base_url": "rtsp://14.21.88.97:38554",
    "publish_user": "publisher",
    "publish_password": "替换为随机发布密码",
    "read_user": "viewer",
    "read_password": "替换为随机观看密码"
  },
  "inference": {
    "device": "cuda:0",
    "half": true,
    "shared_model": true,
    "max_batch_size": 2,
    "batch_wait_ms": 2,
    "streams_per_model_instance": 2,
    "max_streams": 8,
    "startup_grace_seconds": 3
  },
  "output": {
    "encoder": "h264_nvenc",
    "preset": "p4"
  },
  "labels": {
    "map_file": null,
    "font_file": null
  }
}
```

生成三个不同的随机值：

```bash
openssl rand -hex 32
openssl rand -hex 16
openssl rand -hex 16
```

然后：

```bash
chmod 600 config/api.json
```

`public_base_url`必须填写外部播放者真正能访问的公网IP和机房映射端口，
不要写容器名、Ubuntu内网IP或`127.0.0.1`。

API Key、RTSP账号、设备和并发数都只从这个JSON文件读取。Compose不再读取
这些环境变量。MediaMTX通过容器内部HTTP回调向API验证发布和读取权限。

`streams_per_model_instance=2`表示每个模型实例固定最多绑定两路；第3路自动
创建第2个实例，第5路创建第3个。删除流后，完全空闲的实例会自动卸载。
`max_batch_size=2`是每个实例的动态batch上限；`batch_wait_ms`是为了等待同实例
另一路就绪帧所允许的最大微等待。实例只有一路客户端时不会等待。默认2ms以
低延迟为优先，不应为了提高batch命中率直接设成几十毫秒。

`shared_model=true`启用一对二模型池和动态batch。如果现场A/B测试发现
`average_inference_ms`相对旧版本明显增长，可以暂时设为`false`并重启API，
恢复一流一Python进程；NVENC配置仍然有效。这是性能回退开关，不改变模型、
输入尺寸或识别阈值。

CUDA模板使用`h264_nvenc`和低延迟`p4`预设，把H.264编码从CPU迁移到RTX 3060 Ti
的独立编码单元。macOS模板继续使用`libx264/ultrafast`。

识别框默认使用中文类别名且不显示置信度。COCO 80类不需要额外配置。
自训练模型可复制`config/labels.zh.example.json`为
`config/labels.zh.json`，然后把`labels.map_file`设为
`/app/config/labels.zh.json`（Docker）或本机绝对路径。Ubuntu镜像已安装
Noto CJK字体；`font_file`通常保持`null`即可。

macOS本机测试使用另一份模板：

```bash
cp config/api.macos.example.json config/api.json
nano config/api.json
chmod 600 config/api.json
python -m rtsp_annotator.api --config config/api.json
```

macOS模板已经设置`device=mps`和`half=false`；如果本机MPS不可用，将
`device`改为`cpu`。

## 4. 启动

```bash
docker compose \
  -f docker-compose.cuda.api.yml \
  up -d
```

查看：

```bash
docker compose \
  -f docker-compose.cuda.api.yml \
  ps

docker compose \
  -f docker-compose.cuda.api.yml \
  logs -f api
```

Swagger接口页面：

```text
http://UbuntuIP:8080/docs
```

## 5. 创建识别流

只识别人：

```bash
curl --fail-with-body \
  -X POST \
  'http://14.21.88.97:38080/v1/streams' \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: API密钥' \
  -d '{
    "input_url": "rtsp://camera-user:camera-pass@192.168.10.20:554/live",
    "model": "yolo26s.pt",
    "classes": [0],
    "conf": 0.25,
    "iou": 0.45,
    "imgsz": 640,
    "bitrate": "2500k"
  }'
```

返回示例：

```json
{
  "stream_id": "9d87d1b9e72148f0bb4dd61bc44ba20d",
  "status": "starting",
  "rtsp_url": "rtsp://viewer:读取密码@14.21.88.97:38554/detected/9d87d1b9e72148f0bb4dd61bc44ba20d",
  "model": "yolo26s.pt",
  "classes": [0],
  "created_at": "2026-07-28T11:00:00+00:00",
  "exit_code": null,
  "metrics": null
}
```

`classes`为`null`或省略时识别模型支持的全部类别。

`starting`表示会话刚创建；约3秒后仍未退出会显示`running`。`running`表示识别
会话存活，不代表摄像头一定已成功出首帧：输入RTSP暂时不可达时，底层会持续
重连。最终以返回的RTSP URL能被`ffprobe`/播放器读取，以及容器日志没有持续
报错为准。

带识别区域的请求：

```json
{
  "input_url": "rtsp://camera-user:camera-pass@192.168.10.20:554/live",
  "model": "yolo26s.pt",
  "classes": [0],
  "roi": [
    [0.1, 0.15],
    [0.9, 0.15],
    [0.85, 0.9],
    [0.15, 0.9]
  ]
}
```

夜间识别使用独立配置，不修改白天`conf`和输出画面：

```json
{
  "input_url": "rtsp://camera-user:camera-pass@192.168.10.20:554/live",
  "model": "yolo26s.pt",
  "classes": [0],
  "conf": 0.25,
  "bitrate": "2000k",
  "night_vision": {
    "enabled": true,
    "confidence": 0.18,
    "input_gain": 1.18,
    "plate_detector_confidence": 0.20
  }
}
```

未传`night_vision`时继续使用原有白天逻辑。白天流和夜间流不会共用同一个
DeepStream实例；切换时删除旧任务后重新创建即可，不需要重启Docker。参数和
验收方法见`NIGHT_VISION.md`。

同一路同时进行YOLO和中国车牌识别（仅DeepStream后端）：

```json
{
  "input_url": "rtsp://camera-user:camera-pass@192.168.10.20:554/live",
  "model": "yolo26s.pt",
  "classes": [0],
  "license_plate": {
    "enabled": true
  }
}
```

`classes:[0]`只表示最终画面仅显示YOLO的“人员”框，不会阻止内部车辆候选进入
车牌支路。完整参数、性能设计与验收方法见`LICENSE_PLATE.md`。

区域停留、垃圾变化和疑似乱丢垃圾通过`event_detection`启用。它可以与普通
YOLO、夜间模式和车牌识别同时配置，完整请求、事件接口、Webhook和现场验收
方法见`EVENT_DETECTION.md`。

固定机位燃气瓶识别通过`gas_cylinder`启用，仅支持DeepStream后端。它不会
逐帧运行YOLOE，完整参数、摄像头Profile和RTSP质量要求见`GAS_CYLINDER.md`。

地面零散垃圾识别通过`ground_litter`启用，仅支持DeepStream后端。一次调用即可在
同一条输出流上得到"地面识别区域轮廓 + 疑似垃圾框"，不需要另建脚本或第二路流。
只想要区域和垃圾框、不要人/车框时，加`"display_detections": false`：

固定机位 Clean Reference V3.2 使用
`"mode":"clean_reference_v32"`和已审核的`profile_id`。该模式只绘制达到确认时长、
当前仍有异常支撑的事件；原始变化组件和待确认事件不会进入 OSD。当前 Camera 01
验收请求、状态字段和保守 ROI 见`GROUND_LITTER_V32_PRODUCTION.md`及
`config/ground_litter_v32_stream_request.example.json`。未传`mode`时保持下面的原有
YOLO 显示层行为。

```bash
curl --location 'http://14.21.88.97:38080/v1/streams' \
  --header 'X-API-Key: API密钥' \
  --header 'Content-Type: application/json' \
  --data-raw '{
      "input_url": "rtsp://admin:密码@摄像头:554/Streaming/channels/101",
      "model": "yolo26s.pt",
      "conf": 0.35,
      "bitrate": "2000k",
      "display_detections": false,
      "ground_litter": {
          "enabled": true,
          "analysis_fps": 1.0,
          "confidence": 0.20,
          "tile_size_px": 640,
          "tile_overlap": 0.2,
          "minimum_hits": 2,
          "hit_window": 3,
          "hold_seconds": 3,
          "maximum_boxes": 8,
          "label": "疑似垃圾",
          "display_zones": true,
          "zones": [
              {
                  "region_id": "merchant_01",
                  "name": "门店01门前人行道",
                  "polygon": [[0.32,0.15],[0.367,0.15],[0.35,0.30],[0.27,0.30]],
                  "exclude_zones": [],
                  "minimum_short_side_px": 6,
                  "minimum_box_area_px": 36
              }
          ],
          "overlay_exclude_zones": [
              [[0.0,0.035],[0.4,0.035],[0.4,0.105],[0.0,0.105]]
          ]
      }
  }'
```

`display_detections`语义：**只控制输出画面是否绘制普通检测框**（人/车等，绿色框与中文
标签）。它不影响跟踪、事件状态机、逐流`interval_detections`统计，也不影响
`ground_litter`的人车遮挡判定——元数据里始终有全部80类，`classes`和
`display_detections`都只是显示过滤。默认`true`保持原有行为。

垃圾结果与参数语义：

- `zones`是归一化地面多边形，`enabled=true`时至少一个；`exclude_zones`用于排除
  该区域内的固定物，`overlay_exclude_zones`是整幅画面的固定排除区（水印、棚顶等）。
- `minimum_short_side_px`/`minimum_box_area_px`是**原生像素**门槛，按
  `DEEPSTREAM_MUX_WIDTH x DEEPSTREAM_MUX_HEIGHT`计算。要保持摄像头原生像素
  （推荐），必须把mux尺寸设成摄像头分辨率；否则门槛值会按缩小的画面解释。
- `zones[].confidence`/`zones[].night_confidence`允许按区域设置阈值：远端小目标可降低，
  近端固定设施密集区域可提高。模型先按所有区域最低阈值保留候选，再按所属区域过滤。
- `context_class_ids`用于动态遮挡物（默认包含雨伞、长椅、椅子、花盆、餐桌的COCO类别）。
  这些框只让被遮挡候选进入暂缓状态，不会把区域永久屏蔽；遮挡物移动后区域仍可检测新垃圾。
- 模型支持`tile_size_px`选择分块尺寸（常用`640`或`320`）：分块是**原生像素裁剪**，
  `inference_imgsz:null`（默认）时推理输入与分块同尺寸、不做整图缩放。实测 1021 该区域在 1920×1080 下
  `640`为2块、`320`为7块；320分块对已确认的小垃圾置信度更高（约0.14 对 0.11），
  但服务器 GPU 上单帧更慢（约63–79ms 对 33ms），1 FPS 下都可接受。
- `inference_imgsz`可独立设置160–1920，例如`tile_size_px:320,inference_imgsz:640`
  表示裁剪320再按640推理；省略/null保持旧模式。放大不会恢复mux之前丢失的细节。
- `local_actor_max_crops`默认0（关闭），范围0–8；需`actor_model`。对通过ROI的候选做
  最多N个局部人车复核，合并全图/主链人车框后重新过滤，不能保证检出所有俯视停放车辆。
- `box_smoothing_alpha`默认1（不平滑），可设0.5平滑显示框；关联仍用原始框。
- 模型在原生像素裁剪上推理；
  分块数由区域形状决定，受`maximum_tiles`上限保护；超限时该流创建失败。
- `analysis_fps`默认`1.0`，是旁路分析频率而不是显示帧率。显示层要求
  首次在`hit_window`次分析里至少`minimum_hits`次命中后确认。确认后允许模型短时漏检，
  最后一次命中超过`hold_seconds`即隐藏且须重新确认。重复/倒序时间戳不累计命中。
- 人车框覆盖候选达到阈值时立即隐藏并重置确认；仅在旁边、不重叠不应隐藏。
- `display_confidence`显示最近一次命中的置信度（短时保持期间是最后一次命中值），
  不再使用轨迹历史最大值。旁路停止更新时仍受OSD新鲜度限制。
- 查询流的`ground_litter.options`返回生效参数，`effective_imgsz`返回实际选择的输入尺寸；
  `last_inference_ms`包含整图人车、分块垃圾及局部复核耗时。
- `confidence`/`night_confidence`：后者只在流的`night_vision.enabled`为`true`
  时生效；两者都是模型原始阈值，不是最终判定。
- 框上默认只写`label`（默认"疑似垃圾"）。模型类别（Glass/Metal/Paper/
  Plastic/Waste）是材质不是业务结论，需要时必须显式打开`display_class`。
- 结果只是"疑似垃圾"的人工复核线索，不产生`item_id`、不写清理状态、不发通知。
- `ground_litter`不能与`ptz_verification`同时启用（云台运动会让固定地面区域失效）。
- 完整参数、离线校准方法、当前精度边界见`GROUND_LITTER_IMPLEMENTATION.md`。
- 请求体是严格校验的（`extra="forbid"`），不要加自定义注释字段；1021 单路起步范例见
  `config/ground_litter_1021_stream_request.example.json`（其中`input_url`是占位地址，
  真实的摄像头账号密码不要写进仓库）。该范例在2560x1440下生成5个原生640分块，
  实际分块数随区域形状变化，受`maximum_tiles`保护。

远距离船舶框选通过`vessel_detection`启用，仅支持DeepStream后端。它使用
1280推理尺寸、低阈值候选和时序确认，并可给每个机位配置水域ROI、透视分区与
固定误报排除区；完整请求和验收方法见`VESSEL_DETECTION.md`。

只依靠监控画面生成疑似非法捕捞线索时，可在启用`vessel_detection`的基础上
增加`fishing_risk`。它不识别渔网，也不输出法律结论，而是根据禁渔时空、
小范围长时间停留和多次折返生成可解释的人工复核事件。该参数默认关闭；完整
请求、运行中开关和评分语义见`FISHING_RISK.md`。

运行中关闭风险分析并保留船舶框：

```bash
curl -X PATCH \
  -H 'X-API-Key: API密钥' \
  -H 'Content-Type: application/json' \
  -d '{"enabled":false}' \
  'http://14.21.88.97:38080/v1/streams/STREAM_ID/fishing-risk'
```

该接口保持`stream_id`和输出RTSP地址，但会重载所在DeepStream组，期间可能有
短暂流抖动。关闭后不再保存长期船舶轨迹、不显示风险分，也不产生新的疑似捕捞
事件，现有`vessel_detection`船框继续工作。

全景小目标的云台近景确认通过`ptz_verification`显式启用，默认关闭；关闭时就是
原来的纯船舶框模式。它会串行放大、用真实boat类别复核、原生抓图、回HOME并在
SQLite中记住已看过的位置，避免目标ID变化后反复放大。同一摄像机不能同时绑定
两路活动流；运行时还会持有camera_control短时独占租约，防止多worker或人工请求穿插
控制。原生JPEG必须通过独立船舶复检、目标尺度和清晰度校验后才会保存。HOME命令完成后
还必须收到稳定的新全景帧，否则进入恢复锁定而不是继续接目标。PTZ流不能同时启用车牌、事件/垃圾或燃气瓶等固定视角功能。完整请求和现场接入见
`PTZ_VESSEL_VERIFICATION.md`。

开关参数就是`ptz_verification.enabled`。传`false`时不创建摄像头控制客户端，流状态
返回`integration_mode:detection_only`和`state:disabled`；传`true`时才调用
`camera_control`，流状态返回`integration_mode:camera_control`。建议严格按“只测识别
→只测控制→再启用联动”三阶段验收，摄像头控制的独立诊断接口见
`camera_control/docs/API.md`。

需要逐任务排查控制链时，可在创建流的`ptz_verification`中传
`"trace_logging_enabled":true`。每个PTZ复核`job_id`对应一份
`event_root/vessel-verifications/task-traces/<job_id>.jsonl`，记录控制动作、耗时、错误和
最终HOME状态；默认关闭，且不会写入密钥、租约令牌、摄像头凭据或图片内容。

联动默认使用`zoom_strategy:adaptive`：每次放大后依据重捕获船框的宽高和实际增长
重新选择下一步，达到目标尺寸、无有效增长、达到最大轮数/累计增量或丢失目标时立即
停止；最终只有真实`boat`类别可以截图。需要现场保守回退时可传
`zoom_strategy:fixed`和`zoom_steps`。完整参数见`PTZ_VESSEL_VERIFICATION.md`。

需要近景确认后持续跟随同一艘船时，创建流传
`ptz_verification.continuous_tracking:true`；默认`false`仍是截图后回HOME。持续模式以
`adaptive_target_width_ratio:0.33`和`adaptive_target_height_ratio:0.33`作为船框宽或高
约占对应画面边长三分之一的目标，并用中心死区、
缩放滞回和控制间隔避免云台抖动。开启`tracking_recovery_enabled`后，短暂
丢框会先在最后目标位置逐档缩小视野重捕获；船持续丢失、流停止、达到非零的
最长跟踪时间或控制异常后
都会回HOME。流详情中的`ptz_verification.state`会返回`tracking`、`reacquiring`或
`returning_home`，metrics包含`tracking_duration_seconds`、`tracking_corrections`、
`tracking_target_width_ratio`、`tracking_target_height_ratio`和
`tracking_last_end_reason`。
如果需要在达到目标尺寸后再放大一档，传
`tracking_initial_extra_zoom_step:1`；默认`0`保持兼容。演示场景建议同时传
`tracking_recovery_enabled:true`、`tracking_lost_timeout_seconds:30`和
`tracking_max_duration_seconds:0`，表示短暂丢检先扩大视野重捕获，并且不因时长到期
中断。同时传`lost_retry_seconds:15`可避免真正丢失回HOME后长时间冷却。

PTZ触发同时接收主检测器绿色船框和高分辨率旁路船框，重叠框会自动合并。
metrics中的`ptz_primary_candidate_count`、`ptz_sidecar_candidate_count`、
`ptz_trigger_status`、`ptz_trigger_observations`和
`ptz_trigger_cooldown_remaining_seconds`可用于判断当前是尚未累积足够观测，
还是正处于冷却。
`primary_target_minimum_observations`默认为`1`，表示绿色主检测船框首次出现就申请
PTZ任务；`minimum_target_observations`仍只要求青色高分辨率旁路累积观测。

需要人工立即终止当前复核/跟踪并回到预置位时调用：

```text
POST /v1/streams/{stream_id}/ptz/return-home
```

接口返回HTTP 202及`request_id`。worker通过独立控制监控线程停止当前动作，并使用原PTZ
租约执行HOME，不重启流或DeepStream组。可轮询流详情：状态从`returning_home`变为锁存的
`manual_hold`，且metrics中的`ptz_last_return_home_request_id`等于返回的`request_id`、
`ptz_manual_hold=true`时，表示worker已接收请求并禁止新的自动PTZ动作。更新或重建流后才
恢复自动PTZ。流不存在返回404，未启用PTZ或worker未运行返回409。

删除PTZ流时，`DELETE /v1/streams/{stream_id}`会在worker退出前停止在途控制并再次下发
HOME。删除采用两阶段握手：先由仍持有租约的worker执行紧急HOME，并等待metrics上报匹配
的`request_id`和`manual_hold`确认；确认后才发送SIGTERM停止worker。进程退出路径会再做
一次最终HOME作为兜底。即使删除时没有活动跟踪，也不会依赖租约TTL自然过期。删除完成
后该流释放控制租约，不再发送任何摄像机命令。

`vessel_detection.display_proposals`默认`false`：未分类疑似目标不画到输出RTSP，但仍
可供PTZ后台复核。可用`proposal_roi`和运动/形状阈值压制浪纹，并用
`ptz_verification.proposal_minimum_interval_seconds`及
`proposal_maximum_verifications_per_hour`限制疑似目标控制摄像机的频率。

复核结果接口：

```text
GET /v1/vessel-verifications
GET /v1/vessel-verifications/live
GET /v1/vessel-verifications/{job_id}
GET /v1/vessel-verifications/{job_id}/images/{image_id}
```

列表只返回已经完成的复核任务，支持`stream_id`、`result`、`after_sequence`和
`limit`；`sequence`是任务完成事件序号，SSE断线后可用它继续且不会错过最终图片。
`creation_sequence`保留任务创建顺序。所有接口都要求`X-API-Key`。

垃圾分析启用后默认在视频中持续显示语义垃圾框：普通检测为绿色、候选事件为
橙色、确认告警为红色。可通过`garbage.display_detections`关闭普通绿框，但保留
事件判断和告警框。

`garbage.detection_mode`取值为`items`或`pile`。`items`保持原有零散垃圾类别；
`pile`使用独立街景模型，并把相邻垃圾组成框合并为中文`垃圾堆`框。只展示固定
机位垃圾堆时建议同时设置`analysis_fps:1`、`minimum_confidence:0.15`、
`background_change_enabled:false`和`display_hold_seconds:3`，避免全屏背景运动
误画事件框，同时降低旁路对主视频的GPU竞争。

## 6. 查询和停止

查询模型：

```bash
curl \
  -H 'X-API-Key: API密钥' \
  'http://14.21.88.97:38080/v1/models'
```

查询单个任务：

```bash
curl \
  -H 'X-API-Key: API密钥' \
  'http://14.21.88.97:38080/v1/streams/STREAM_ID'
```

运行约10秒后，响应中的`metrics`会包含该路最近统计周期的指标：

```json
{
  "capture_fps": 25.0,
  "inference_fps": 24.1,
  "publish_fps": 25.0,
  "average_inference_ms": 27.3,
  "average_frame_age_ms": 62.0,
  "interval_inference_skipped": 9,
  "total_inference_skipped": 83,
  "total_detections": 1260,
  "capture_reconnects": 0,
  "publisher_restarts": 0,
  "shared_model_clients": 2,
  "model_instance_id": 1,
  "model_instance_clients": 2,
  "average_batch_size": 1.8,
  "average_batch_inference_ms": 31.4,
  "last_batch_size": 2,
  "max_batch_observed": 2
}
```

`average_inference_ms`按每路请求从提交共享调度器、等待、完成模型推理到画框完成
的真实耗时统计，不会用批次耗时除以batch大小来美化数字。优化前后应使用同一
输入、模型和参数比较；建议平均值增幅不超过10%且内部帧龄仍低于100ms。
`average_batch_size`接近1表示很少合批；大于1才说明共享调度器确实在利用
batch。`average_batch_inference_ms`是整个batch的模型调用耗时，不是每路延迟。
四路正常分配时，四个任务的`model_instance_id`应为两组，例如`1,1,2,2`，
且每组的`model_instance_clients`不超过2。

列出任务：

```bash
curl \
  -H 'X-API-Key: API密钥' \
  'http://14.21.88.97:38080/v1/streams'
```

停止：

```bash
curl \
  -X DELETE \
  -H 'X-API-Key: API密钥' \
  'http://14.21.88.97:38080/v1/streams/STREAM_ID'
```

## 7. 播放

直接播放创建接口返回的`rtsp_url`：

```bash
ffplay \
  -rtsp_transport tcp \
  -fflags nobuffer \
  -flags low_delay \
  -framedrop \
  '接口返回的rtsp_url'
```

## 8. 安全说明

- 所有管理接口除`/health`外都必须提供`X-API-Key`；
- 只允许选择镜像`/app/models`内已有的`.pt`文件；
- MediaMTX发布账号只能发布`detected/*`，读取账号只能读取`detected/*`；
- 请求中的原RTSP可能包含摄像头密码，不应通过明文公网HTTP传输；
- 公网生产环境应使用HTTPS反向代理，或通过机房ACL只允许固定调用方IP访问
  API端口；
- 普通RTSP本身也不加密，敏感场景使用RTSPS、VPN或专线ACL。
# 持续跟踪边缘保护补充（2026-09-07）

`ptz_verification.tracking_edge_guard_enabled` 默认 `false`。启用时要求 `enabled=true`
和 `continuous_tracking=true`：在持续跟踪阶段按出画风险而非尺寸偏大决定是否减一档，
关闭丢框后的盲目缩小，稳定后才恢复渐进放大。参数、延迟假设、详情响应与尚未通过的
模拟场景见 [TRACKING_EDGE_GUARD.md](TRACKING_EDGE_GUARD.md)。此开关不替代完整
`demo_continuous` 会话；前置抓图、初始放大和原停止/HOME 生命周期尚未改变。
