# 小目标 PTZ 船舶复核

## 目标与结论边界

固定全景里只有几像素到几十像素的目标，不适合直接做可靠船舶分类。该功能先用高召回候选找“值得看一眼的位置”，串行控制云台放大，再用现有 COCO `boat` 模型确认近景是否为船，最后抓取摄像机原生 JPEG 并回到 HOME 预置位。

当前结果只有：

- `boat_confirmed`：近景模型确认是船，已保存原始近景图片；
- `candidate_not_confirmed`：近景未确认，回位后处理下一个候选；
- `evidence_not_confirmed`：控制过程识别到船，但原生JPEG二次验真或清晰度不达标，未保存证据；
- `insufficient_resolution`：达到安全变焦上限后目标仍不够大或没有有效增长；
- `target_ambiguous`：近景中出现多个空间分离且得分接近的目标组，拒绝切换目标；
- `target_lost`：控制、重捕获、对焦或抓图失败，稍后可重试；
- `recovery_failed`：无法确认已回 HOME，停止该摄像机的后续自动控制，等待人工处理。

它不把“看不到船号”直接判成非法船。船号 OCR、船证/AIS 名录核验属于下一阶段，只有近景字符像素足够时才应启用。

## 工作流与安全约束

每个摄像机只运行一个串行协调器：

1. 全景船模型框或时序小目标提议进入候选队列；
2. 候选先连续观察多帧，SQLite 按位置、速度、时间和冷却期做持久化去重；
3. 默认按重捕获船框大小自适应选择下一次`locate`变焦增量；每步重新居中，中间级允许
   疑似目标继续引导，最终截图前必须出现真实`boat`；
4. 等待期间会跳过空帧，只有真实 `boat` 类能通过最终近景确认；
5. 自动对焦、原生抓图；JPEG会再次解码并独立运行船模型，只有中央船框尺寸和船体区域
   清晰度都达标才原子保存，失败会重抓且绝不把坏图写成证据；
6. `finally` 中回 HOME；首次失败会先 STOP 再重试，只有HOME命令完成且连续收到指定数量
   的新全景帧才恢复监控，延迟到达的近景帧会按视角代际丢弃；
7. 回位失败进入 `recovery_required`，不再发新的定位命令。

协调器启动后会为`camera_id`持有并定期续期短时独占租约，结束时释放。这样即使API进程
内的活动流检查失效、存在多个worker或人工命令并发，也不能把另一条PTZ命令插入
“定位→变焦→抓图→HOME”的事务中。紧急STOP仍可随时执行。

删除流或重启同组 worker 时，管理器会按组内 PTZ 协调器的最大离位时间预留动态退出窗口。正在复核的任务会完成当前受控步骤或响应中断，随后回 HOME 并写入合法终态；API 只有在该收尾完成后才返回停止结果，避免 SQLite 遗留 `running` 任务。

云台离开全景期间，疑似非法捕捞轨迹分析会暂停并清空空间历史，但船舶旁路继续取帧用于近景确认。这样不会把镜头运动当成船舶折返或徘徊。

船舶旁路在PTZ复核期间自动切换到近景配置：取消仅对HOME视角成立的水线ROI、固定
障碍排除区和水域裁剪，改用全画面船舶检测及中央外观候选。回HOME并收到稳定新帧后
自动恢复原配置。视角切换会清空坐标相关轨迹和运动背景，但保持结果版本递增，避免
复核线程误读旧帧或永久等待。

输出画面也使用同一PTZ状态切换：离开HOME后隐藏旧水面ROI和排除区，顶层普通检测
临时不按HOME ROI过滤；回HOME且全景帧稳定后再恢复原多边形。不能把HOME多边形简单
固定在屏幕上，也不能在没有云台姿态标定和水面空间模型时猜测它随视角的几何变换。

PTZ 流是专用的可变视角流，不能同时启用 `license_plate`、`event_detection`（含垃圾分析）或 `gas_cylinder`；这些固定视角功能应使用其他摄像头或独立的固定机位流。

## 创建流

先在 API 容器环境中设置和 `camera_control` 相同的内部密钥：

```bash
export CAMERA_CONTROL_API_KEY='请替换为随机长密钥'
```

请求示例：

```json
{
  "input_url": "rtsp://user:password@camera/live",
  "vessel_detection": {
    "enabled": true,
    "analysis_fps": 5,
    "confidence": 0.10,
    "imgsz": 1280,
    "class_ids": [8],
    "roi": [[0.0, 0.58], [0.42, 0.51], [0.58, 0.46], [1.0, 0.46], [1.0, 0.75], [0.0, 0.75]],
    "exclude_rois": [[[0.96, 0.70], [1.0, 0.70], [1.0, 0.75], [0.96, 0.75]]],
    "minimum_hits": 4,
    "display_proposals": false,
    "small_target_proposals": true,
    "proposal_roi": [[0.0, 0.58], [1.0, 0.46], [1.0, 0.75], [0.0, 0.75]],
    "proposal_threshold": 60,
    "proposal_appearance_enabled": true,
    "proposal_appearance_threshold": 18,
    "proposal_appearance_blur_pixels": 31,
    "proposal_border_margin": 0.01,
    "proposal_minimum_area_pixels": 20,
    "proposal_maximum_area_pixels": 1000,
    "proposal_minimum_width_pixels": 4,
    "proposal_minimum_height_pixels": 3,
    "proposal_minimum_fill_ratio": 0.30,
    "proposal_minimum_motion_ratio": 0.10,
    "proposal_maximum_candidates": 4
  },
  "ptz_verification": {
    "enabled": true,
    "camera_id": "river-ptz-01",
    "camera_control_url": "http://camera-control:8080",
    "trace_logging_enabled": false,
    "display_operation_log": false,
    "vessel_number_recognition_enabled": false,
    "vessel_number_fallback": "",
    "zoom_strategy": "adaptive",
    "adaptive_target_width_ratio": 0.33,
    "adaptive_target_height_ratio": 0.33,
    "adaptive_min_step": 1,
    "adaptive_max_step": 6,
    "adaptive_max_rounds": 3,
    "adaptive_max_total_zoom_delta": 12,
    "adaptive_min_scale_growth_ratio": 1.12,
    "continuous_tracking": false,
    "tracking_center_deadband": 0.10,
    "tracking_command_interval_seconds": 0.5,
    "tracking_settle_seconds": 0.25,
    "tracking_recovery_enabled": false,
    "tracking_recovery_interval_seconds": 2.0,
    "tracking_recovery_zoom_out_step": 1,
    "tracking_recovery_max_attempts": 3,
    "tracking_lost_timeout_seconds": 4,
    "tracking_max_duration_seconds": 300,
    "tracking_zoom_hysteresis_ratio": 0.20,
    "tracking_zoom_step": 1,
    "tracking_initial_extra_zoom_step": 0,
    "confirmed_target_fallback_zoom_rounds": 1,
    "confirmed_target_fallback_zoom_step": 3,
    "reacquire_strict_center_radius": 0.22,
    "reacquire_center_radius": 0.45,
    "reacquire_cluster_radius": 0.18,
    "settle_seconds": 0.5,
    "home_frame_delay_seconds": 1.5,
    "home_stable_frames": 2,
    "evidence_validation_required": true,
    "evidence_capture_attempts": 2,
    "evidence_minimum_sharpness": 12.0,
    "evidence_target_scale_ratio": 0.70,
    "minimum_target_observations": 3,
    "primary_target_minimum_observations": 1,
    "proposal_merge_radius": 0.04,
    "proposal_minimum_interval_seconds": 30,
    "proposal_maximum_verifications_per_hour": 12,
    "reacquire_timeout_seconds": 4,
    "maximum_off_home_seconds": 25,
    "confirmed_cooldown_seconds": 1200,
    "negative_cooldown_seconds": 3600,
    "lost_retry_seconds": 180
  }
}
```

`roi` 必须按每个机位单独标定。上例按本次`8月12日.mp4`中人工框出的六个值得
放大的水面小目标调整：向上覆盖远处水线，同时在下方截断高频近景水波。其他视角不应
直接复制。`small_target_proposals`默认关闭；PTZ高召回模式应显式开启。候选同时来自
运行背景运动差分和局部明暗外观，近似静止的暗色船体或航行灯在背景适应后仍可保留。
这些结果只表示“值得放大看”，不会在证据或捕捞规则中冒充已确认的船。
`display_proposals:false`仅隐藏转发画面上的疑似框和计数，不关闭后台候选。正常
`boat`框仍显示。`proposal_roi`可只保留中远水面，避免近景浪花持续占用候选队列。

`proposal_appearance_threshold`越低召回越高，水波和岸灯也越多；建议从18开始，配合
连续4次命中、准确ROI和固定结构排除区。`proposal_border_margin`用于过滤解码边缘闪烁。
`proposal_merge_radius`只在PTZ入库前合并几乎同位置的未分类碎片；本机位默认4%，
人工框出的相邻远点间隔约8%，不会被合并。真实`boat`框永远不会因这个参数互相合并。
全景阶段可以多看，近景最终截图前仍必须出现真实`boat`；未确认目标写入负结果冷却，
回HOME后继续下一个，不会立即重复查看。
另外，未分类候选默认至少间隔30秒才允许触发一次，每小时最多12次；这两个预算只
约束疑似目标，不会阻止已经由模型识别为`boat`的船舶复核。

完成一次复核后，SQLite会把冷却记忆固定在当次实际触发坐标并清零旧速度。冷却期内不再
对该位置做速度外推，也不允许检测ID变化、波纹或倒影碎片把这条记忆拖到其他位置；落在
`proposal_merge_radius`内的新候选会命中同一条已复核记忆并被抑制。这样回HOME后不会因
检测ID重建立即再次放大同一位置，同时约8%间隔的相邻人工目标仍可分别进入队列。

`ptz_verification.enabled`就是船舶识别是否接入摄像头控制的独立参数。关闭时 worker
不会创建`CameraControlClient`，即使`camera_control`服务未启动，船舶识别旁路也可以
正常框船。查询流状态时会返回`integration_mode:detection_only`和`state:disabled`。

### 每个控制任务的 trace 日志

`trace_logging_enabled`默认`false`。设为`true`后，每次候选成功创建的PTZ复核任务都以
既有`job_id`同时作为`task_id`和`trace_id`，在
`event_root/vessel-verifications/task-traces/<job_id>.jsonl`生成一份独立日志文件。
同一任务中的定位/变焦、自动对焦、原生抓图、持续跟踪纠偏、STOP恢复和回HOME都记录
开始、成功或失败状态及耗时；首尾另有`task.started`和`task.finished`，可还原完整控制链。

日志只记录归一化坐标、相对变焦步长、动作原因、抓图大小/MIME类型和错误，不记录
`camera_control` API key、租约令牌、摄像头账号密码或JPEG内容。日志写入采用尽力而为：
磁盘不可写或序列化失败时会停止该task的trace，但不会阻止定位、STOP或回HOME等安全动作。
关闭开关时不会创建`task-traces`目录或task日志文件。该开关只覆盖自动PTZ复核任务；
worker启动时的初始HOME和没有活动复核task时的人工HOME不伪装成候选触发任务。

只识别船、不控制摄像机时，省略 `ptz_verification` 或传：

```json
{"ptz_verification":{"enabled":false}}
```

如果全景已有船模型框，亦可保持 `small_target_proposals:false`，仅对已检测船做近景复核。

### 持续跟踪模式

`continuous_tracking`默认`false`，保持原有“近景确认、截图、回HOME”行为。传`true`后，
只有真实`boat`完成近景确认才进入持续跟踪；系统仍会先尝试保存首张证据图，但抓图或
证据复检失败不会阻止已经确认的船继续被跟踪，复核任务会保留
`evidence_not_confirmed`结果。疑似运动或外观候选不会直接进入跟踪。跟踪期间继续使用
近景船舶检测结果驱动云台：

- 船框中心偏离画面中心超过`tracking_center_deadband`时才纠偏，避免细小检测抖动导致
  云台持续振荡；
- 目标尺度以`adaptive_target_width_ratio`和`adaptive_target_height_ratio`为基准，
  `tracking_zoom_hysteresis_ratio:0.20`表示低于目标尺度80%时放大、高于120%时缩小，
  中间区间保持当前倍率；
- `tracking_initial_extra_zoom_step`只在持续跟踪模式生效，会在初始闭环放大
  完成后、首次抓图和跟踪前再补一次变倍并重新确认目标。`0`保持原行为，
  现场需要“再放大一档”时传`1`；
- 相邻控制命令至少间隔`tracking_command_interval_seconds`，默认最多约每秒两次；
- 每次持续追踪指令只等待`tracking_settle_seconds`再读取新目标位置，不再复用初次放大
  的`settle_seconds`；这允许初次变焦等待画面稳定，同时以更短周期纠偏移动船舶；
- 开启`tracking_recovery_enabled`后，短暂丢框会在船最后出现位置按
  `tracking_recovery_interval_seconds`逐档缩小画面，最多尝试
  `tracking_recovery_max_attempts`次，重新找到船后继续跟踪；
- 连续`tracking_lost_timeout_seconds`仍没有安全重捕获到目标才回HOME。
  `tracking_max_duration_seconds:0`表示不设跟踪时长上限；非零值到期后回HOME；
- 流停止、控制失败或任务关闭同样会进入HOME恢复；即使控制请求在摄像机已经移动后
  超时，也会按“已离开HOME”处理；
- 多个空间分离目标得分接近时不强行接管其中一艘，超时后安全回HOME。

例如开启演示模式：

```json
{
  "ptz_verification": {
    "enabled": true,
    "camera_id": "river-ptz-01",
    "continuous_tracking": true,
    "adaptive_target_width_ratio": 0.33,
    "adaptive_target_height_ratio": 0.33,
    "tracking_initial_extra_zoom_step": 1,
    "tracking_center_deadband": 0.05,
    "tracking_command_interval_seconds": 0.2,
    "tracking_settle_seconds": 0.1,
    "tracking_recovery_enabled": true,
    "tracking_recovery_interval_seconds": 2.0,
    "tracking_recovery_zoom_out_step": 1,
    "tracking_recovery_max_attempts": 3,
    "tracking_lost_timeout_seconds": 30,
    "tracking_max_duration_seconds": 0,
    "vessel_number_recognition_enabled": true,
    "vessel_number_fallback": "10032",
    "lost_retry_seconds": 15
  }
}
```

这里的`0.33`表示船框宽度或高度任一达到画面对应边长约三分之一即停止继续放大，
不是船框面积占整幅画面的三分之一。船体通常扁长，使用“宽或高任一达到”可以避免为了
强求高度三分之一而把船头、船尾裁出画面。
`tracking_initial_extra_zoom_step:1`会在此基础上再补一档。演示环境下建议开启
扩大视野重捕获，将`tracking_lost_timeout_seconds`设为30秒、
`tracking_max_duration_seconds`设为0，并将`lost_retry_seconds`设为15秒；否则一次
短暂丢失会按生产默认值冷却180秒，表现为后续仍识别到船却长时间不再转动。
即使首张证据图已经确认船舶，只要持续跟踪的结束原因是`target_lost`，也会
使用`lost_retry_seconds`而不是`confirmed_cooldown_seconds`，使同一艘移动船能够快速
重新进入联动。
持续跟踪状态通过流详情中的`ptz_verification.state`返回，可见`tracking`、
`reacquiring`和`returning_home`；底层metrics还包含跟踪时长、纠偏次数、当前船框宽高比例
及最近结束原因。当前参数仍属于创建流参数；要切换模式，应先停止原流并确认回HOME，
再重新创建。

PTZ候选现在同时来自两条路径：主DeepStream检测器画出的绿色船框，以及高分辨率
船舶旁路画出的青色船框。两条路径在进入PTZ前按空间重叠去重，因此只要画面出现
满足主流置信度和类别条件的绿色船框，也会直接参与PTZ连续观测，不再要求青色旁路
必须同时命中。流metrics中的`ptz_primary_candidate_count`和
`ptz_sidecar_candidate_count`分别表示两条路径当前候选数；`ptz_trigger_status`会显示
`waiting_for_candidates`、`observing`、`cooldown`、`suppressed`或`triggered`，并配合
`ptz_trigger_observations`、`ptz_trigger_required_observations`及
`ptz_trigger_cooldown_remaining_seconds`解释为什么当前尚未转动。
`primary_target_minimum_observations`单独控制绿色主检测框的触发次数，默认`1`：
第一个通过类别、置信度和ROI过滤的绿色船框就可以申请PTZ任务。
`minimum_target_observations`仍用于青色高分辨率旁路候选，避免放宽该路径的误触发。

### 随时中断并紧急回HOME

任何时候都可以调用下面的流级接口，请求中断当前PTZ复核或持续跟踪并回到HOME：

```bash
curl -X POST \
  -H 'X-API-Key: API密钥' \
  'http://API地址/v1/streams/STREAM_ID/ptz/return-home'
```

接口返回HTTP 202和唯一`request_id`，表示安全命令已经写入对应DeepStream worker的控制
队列。worker内独立的控制监控线程会立即设置中断信号，并通过当前持有租约的
`CameraControlClient`先发stop、再由协调器执行HOME，避免绕过租约与正在执行的命令抢占
摄像头。该动作是锁存式安全停机：流详情状态会先变为`returning_home`，HOME成功且收到
稳定全景帧后变为`manual_hold`，不再接受新的自动PTZ任务，防止刚回原位又被下一次检测
立即拉走。需要恢复自动PTZ时，应更新或重建该流。metrics中的
`ptz_last_return_home_request_id`可与接口返回值核对，`ptz_manual_hold=true`表示锁存已生效。
流不存在返回404，未启用PTZ或worker未运行返回409。

该接口不会停止识别流，也不会重启DeepStream组；它只锁停当前流的后续自动PTZ动作。
如果希望完全停止本路任务，仍应调用`DELETE /v1/streams/{stream_id}`；任务停止路径同样
会先通过控制队列要求仍持有租约的worker中断任何在途PTZ动作并回HOME。API会等待worker
上报与本次请求匹配的`request_id`、`manual_hold`和HOME完成状态，再停止进程；SIGTERM
退出路径还会重新获取租约、停止残余运动并再次强制HOME作为兜底。这样不会先杀掉唯一
持有租约的worker再等待租约TTL自然过期。删除成功后算法不再持有租约或下发控制命令，
摄像机保持在HOME，除非另一个客户端主动控制它。

### 自适应变焦

`zoom_strategy`默认是`adaptive`。它不会把`zoom_delta`误当成固定光学倍数，而是在每次
控制后使用重新检测到的船框作为反馈：

- 船框宽达到画面宽度的33%，或高达到画面高度的33%时，提前停止变焦；
- 船越小，首次步长越大，但单次限制在`adaptive_min_step`到`adaptive_max_step`；
- 放大后重新计算下一步，目标接近所需尺寸时自动减小步长；
- 已确认船舶在首次转动后短暂丢检时，会保持当前中心做至多
  `confirmed_target_fallback_zoom_rounds`轮保底变焦，每轮步长由
  `confirmed_target_fallback_zoom_step`控制；随后重新搜索近景，避免远处小船尚未放大到
  可识别尺寸就立即回HOME；
- 保底变焦只适用于全景中已经由模型确认为`boat`的目标，不适用于水纹等疑似目标；
- 保底步骤仍受`adaptive_max_total_zoom_delta`和`maximum_off_home_seconds`限制，连续丢检
  后会停止并回HOME，避免方向标定有误时持续盲目放大；
- 同一类别船框的实际增长会持续修正当前摄像机的变焦增益估计；
- 已达到尺寸但偏离中心时只重新居中，`zoom_delta`为0；
- 只有真实`boat`、尺寸达标且位于中央安全区时才截图；
- 连续两轮没有有效增长会写入`insufficient_resolution`并回HOME，不保存低质量证据；
- PTZ动作后先在`reacquire_strict_center_radius`中央锁定区搜索，超时后才有限扩大到
  `reacquire_center_radius`；全画面推理不等于允许全画面任意目标接管控制；
- 多个相邻小船按`reacquire_cluster_radius`聚成同一控制目标簇，云台对准船群中心，
  但变焦尺寸使用成员单船宽高的中位数，避免船群大框掩盖单船分辨率不足；
- 船群放大后逐渐分开时继续跟随预期中心附近成员，不会因相邻多船直接回HOME；
- 只有空间上分离的多个目标组得分过于接近时才视为竞争，不强行切换目标，也不执行
  保底盲目变焦；
- 最多执行`adaptive_max_rounds`轮，并限制累计`zoom_delta`；
- 框面积尺度增长低于`adaptive_min_scale_growth_ratio`时停止无效放大；
- 无增益目标只有已被真实`boat`类别确认时才可截图，运动提议会被判为未确认；
- 目标已经足够大时不移动云台，直接自动对焦和抓取原生图片。

目标宽高比例是归一化画面比例。例如1920×1080画面中宽度0.33约为634像素。四台
摄像机可以分别配置，不要求型号、初始焦距或安装距离相同。

`evidence_minimum_sharpness`使用船框裁剪区域的拉普拉斯方差，只是最低防呆阈值；现场应
从清晰/模糊原生抓图样本标定。`evidence_target_scale_ratio`要求证据图里的船至少保留
控制结束时目标尺度的一定比例，避免SDK抓图通道与识别流不同步。默认最多抓两次，全部
失败时结果为`evidence_not_confirmed`，数据库不保存JPEG。

现场需要固定步骤回退时仍支持：

```json
{
  "zoom_strategy": "fixed",
  "zoom_steps": [4, 4]
}
```

## 推荐的三阶段验收

1. **只测船舶识别**：创建流时启用`vessel_detection`，保持
   `ptz_verification.enabled=false`。验收船框召回率、误报和输出RTSP，不需要启动
   `camera_control`。
2. **只测摄像头控制**：不创建PTZ联动流，直接调用`camera_control`的
   `GET /v1/providers/dahua/sdk-status`和
   `POST /v1/cameras/{camera_id}/tests/diagnostic`。先保持`allow_motion=false`验证登录和
   原生截图，再由现场人员显式传`allow_motion=true`验证定位、变焦、自动对焦、截图、
   HOME和紧急恢复。
3. **再测两者联动**：前两阶段分别通过后，重新创建流并传
   `ptz_verification.enabled=true`、`camera_id`和自适应目标尺寸。此时流状态
   返回`integration_mode:camera_control`。

`ptz_verification`是创建流参数，不建议在摄像机正离开HOME时热切换。需要改变联动模式
时，应先停止原流并确认摄像机已回HOME，再用目标参数重新创建流。

## 追踪画面提示

启用`ptz_verification`后，当前追踪目标会直接显示在输出视频上。右上角操作日志由
`display_operation_log`单独控制，默认`false`；演示时设为`true`即可开启：

```json
{
  "ptz_verification": {
    "enabled": true,
    "display_operation_log": true
  }
}
```

- 当前被锁定、复核或持续追踪的船使用**粗红框**，标签显示`追踪中｜船舶`；
- 其他普通船舶仍使用原有绿框，避免演示时无法判断摄像头正在跟随哪一艘船；
- `display_operation_log=true`时，右上角显示一行当前PTZ状态，以及最近4条摄像头操作；
- 操作日志覆盖定位、追踪纠偏、变焦、自动对焦、证据抓图、停止云台、返回HOME和失败；
- 日志区采用紧凑字号和半透明底色，最多5行，不会随运行时间持续扩大。

当目标丢失、追踪任务结束或返回HOME后，红色目标框会被清除。日志会保留最近4条操作，
用于现场确认是否实际下发过云台命令。

放大阶段会使用框包含比例消除主检测器整船框与船舶旁路局部框的重复显示，并且同一时刻
只允许一个框使用`追踪中`样式。启用`vessel_number_recognition_enabled`后，系统在近景
抓图中优先搜索当前船框内的蓝色船牌并读取数字；若运行镜像没有OCR组件或结果不可靠，
则使用`vessel_number_fallback`。演示可传`"10032"`，正式环境应留空并安装、标定OCR
组件，避免把演示编号误当作真实识别结果。

## 证据接口

以下接口都使用 RTSP API 的 `X-API-Key`：

```bash
curl -H 'X-API-Key: API密钥' \
  'http://API地址/v1/vessel-verifications?stream_id=STREAM_ID&result=boat_confirmed'
```

增量轮询使用上次最大 `sequence`。该序号在任务完成并且图片已经保存后生成，不会暴露尚未完成的任务：

```bash
curl -H 'X-API-Key: API密钥' \
  'http://API地址/v1/vessel-verifications?after_sequence=123'
```

也可以连接 SSE：

```bash
curl -N -H 'X-API-Key: API密钥' \
  'http://API地址/v1/vessel-verifications/live?after_sequence=123'
```

列表中的 `images[].download_url` 可实时拉取 JPEG；`creation_sequence` 仅用于审计任务创建顺序。数据库和图片位于该流 `event_root/vessel-verifications`，SQLite 使用 WAL，图片写入采用临时文件后原子替换。

## 无真机验证与现场接入

离线候选回放：

```bash
.venv/bin/python scripts/replay_ptz_candidates.py \
  --input '/path/to/camera.mp4' \
  --roi '0,0.58;0.42,0.51;0.58,0.46;1,0.46;1,0.75;0,0.75' \
  --exclude-roi '0.96,0.70;1,0.70;1,0.75;0.96,0.75' \
  --threshold 60 \
  --appearance-threshold 18 \
  --appearance-blur 31 \
  --minimum-area 20 \
  --maximum-area 1000 \
  --minimum-hits 4
```

这只验证候选负载和稳定轨迹，不代表候选都是真船。单帧候选数也不等于实际PTZ次数：
协调器串行工作，SQLite会跨检测ID合并并对已确认、未确认和丢失目标分别冷却。真实设备
首次接入必须先执行 `camera_control/scripts/camera_doctor.py`：先登录和原生抓图，再由
现场人员明确允许 `--control` 校准 `invert_x`、`invert_y`、HOME预置位和变倍响应。

在没有设备时已能验证 HTTP 合约、SDK 抓图回调关联与超时、状态机回位、故障注入、SQLite 去重、证据接口和真实录像候选负载；无法离线证明的只有具体机型的坐标方向、实际光学倍率、预置位和 SDK 二进制/固件兼容性。
# 近景跟踪保护补充（2026-09-07）

用户允许在突然加速或即将出画时小幅拉远，但正常航行应保持连续近景跟随。
新增默认关闭的 `tracking_edge_guard_enabled`，实现范围、启用参数和明确失败边界见
[TRACKING_EDGE_GUARD.md](TRACKING_EDGE_GUARD.md)。当前仅修改现有跟踪阶段，
不是完整连续演示方案，也不保证读清船号或任意加速下不丢失。没有进行真机部署。
