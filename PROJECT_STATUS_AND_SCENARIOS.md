# RTSP 实时识别项目现状与分场景使用手册

更新日期：2026-08-01

> **2026-09-21 零散垃圾路线更新**：现有垃圾服务链路、语义模型和事件状态机仍在，但“七天回放生成 Profile，再由背景差分独立召回小垃圾”的路线已在预注册实验中判定 **NO-GO**（oracle 命中率 0.1276）。v3～v5 Profile Bank 不属于生产可用资产，v6 已暂停；原自动 Profile 切换方案不再实施。下一步改为保留 `turhancan_yolov8m_seg_trash.pt` 检测常规垃圾，并训练一个 ROI 分块的一类现场小垃圾检测器。详见 `docs/decisions/2026-09-21-ground-litter-profile-prior-no-go.md` 与 `docs/plans/2026-09-21-ground-litter-small-detector-roadmap.md`。下文 2026-08-01 的完成状态是历史快照，不能单独作为当前小垃圾准确率结论。

## 1. 当前完成程度

项目已经从单路 YOLO 画框程序，发展为可通过 HTTP 动态创建任务、通过 RTSP
播放结果、支持事件检测和人工复核的 DeepStream 实时视频分析服务。

当前服务器已经正式部署，不是仅在本机测试：

- 部署目录：`/home/sf01/rtsp-deepstream`
- API：`http://14.21.88.97:38080`
- 输出 RTSP：`rtsp://14.21.88.97:38554/detected/{stream_id}`
- 当前镜像：`rtsp-yolo-annotator:deepstream8-events-final-amd64`
- 运行平台：Ubuntu、RTX 3060 Ti、DeepStream 8、TensorRT FP16

### 1.1 已实现能力

| 能力 | 当前状态 | 说明 |
| --- | --- | --- |
| RTSP 拉流、识别、中文画框、重新推流 | 已完成并部署 | 支持 HTTP 创建和停止 |
| 人、车及 COCO 类别识别 | 已完成 | `classes` 控制普通框显示类别 |
| 低延迟硬件处理链路 | 已完成 | NVDEC、TensorRT、NvDCF、GPU OSD、NVENC |
| 中国车牌检测、跟踪、识别 | 已完成 | 与普通 YOLO 同一路运行 |
| 独立夜间识别配置 | 已完成 | 不改变白天参数和输出画面亮度 |
| 普通识别区域 | 已完成 | 区域外普通框不显示 |
| 人员/车辆区域停留告警 | 已完成并完成链路验证 | 默认停留 20 秒触发 |
| 垃圾新增、移除、移动检测 | 已完成 | YOLO-World + 背景变化联合判断 |
| 画面持续显示垃圾框 | 已完成 | 普通绿色、候选橙色、确认告警红色 |
| 疑似乱丢垃圾关联 | 已完成 | 关联人员/车辆轨迹、离开和垃圾持续出现 |
| 实时事件叠字 | 已完成 | 观察中为橙色，确认事件为红色 |
| 事件查询、截图、确认、驳回 | 已完成 | 历史事件保存在 `data/events` |
| Webhook | 已完成 | 后台异步发送，不阻塞视频链路 |

### 1.2 已验证结果

使用真实 1920×1080、25 FPS 监控流，同时开启普通识别、车牌、区域停留和垃圾
旁路后，服务器长期指标为：

```text
pipeline       25 FPS
pre-encode     25 FPS
publish        25 FPS
duplicate       0 FPS
pipeline_healthy = true
```

公网连续解码 30 秒得到 751 帧，没有发现重复帧、H.264 解码错误或持续马赛克。
垃圾分析默认约 3 FPS，通过容量为 1 的旁路队列运行；垃圾分析过慢时只丢弃待
分析帧，不积压 25 FPS 主视频。

区域停留已完成服务器端真实链路测试，约 20 秒正常产生 `zone_dwell` 事件并在
画面显示。垃圾状态机已有自动测试覆盖新增、清理、移动、人物关联、远距离
误关联、检测闪烁和光照变化。

### 1.3 尚需现场验收的部分

“疑似乱丢垃圾”属于复杂业务判断，不等同于已经达到任意摄像头下的生产准确率。
公开垃圾素材在测试环境中无法下载，因此目前完成的是算法链路、状态机和性能
验证，尚未使用目标现场的真实丢垃圾、捡垃圾视频完成端到端准确率验收。

上线前仍需每个固定机位提供少量白天、夜间、丢垃圾、捡垃圾、普通路过、保洁
清理、车辆停靠的录像，用于确定 ROI 和阈值。摄像头视角、目标像素尺寸、遮挡、
夜间拖影会直接决定最终识别率。

## 2. 处理架构

```text
输入 RTSP
   │
   ├─ NVDEC 硬解码
   │
   ├─ YOLO TensorRT ─ NvDCF 跟踪 ─ 车牌检测/异步识别
   │                                  │
   │                                  └─ 稳定车牌文本
   │
   ├─ 事件状态机：区域停留、人物/车辆与垃圾变化关联
   │
   ├─ 低帧率垃圾旁路：YOLO-World + 背景变化
   │
   └─ GPU OSD 叠加 ─ NVENC 编码 ─ MediaMTX ─ 新 RTSP
```

HTTP API 只管理任务。视频数据不经过 HTTP 返回，播放器使用创建接口返回的
`rtsp_url`。

## 3. 公共调用约定

以下示例统一使用：

```text
API地址：http://14.21.88.97:38080
请求头：X-API-Key: 你的API密钥
```

创建成功后保存响应里的：

```json
{
  "stream_id": "任务ID",
  "status": "starting",
  "rtsp_url": "识别后的RTSP播放地址"
}
```

约 3 秒后查询任务，确认状态和性能指标：

```bash
curl -H 'X-API-Key: 你的API密钥' \
  'http://14.21.88.97:38080/v1/streams/STREAM_ID'
```

播放器建议强制使用 RTSP over TCP：

```bash
ffplay -rtsp_transport tcp -fflags nobuffer -flags low_delay -framedrop \
  '接口返回的rtsp_url'
```

## 4. 场景一：只识别人

适用于普通人员监控，只显示人员框，不做行为告警。

```bash
curl -X POST 'http://14.21.88.97:38080/v1/streams' \
  -H 'X-API-Key: 你的API密钥' \
  -H 'Content-Type: application/json' \
  -d '{
    "input_url": "rtsp://摄像头账号:密码@摄像头地址/路径",
    "model": "yolo26s.pt",
    "classes": [0],
    "conf": 0.25,
    "iou": 0.45,
    "imgsz": 640,
    "bitrate": "2500k"
  }'
```

`classes:[0]` 是 COCO 的人员类别。画面显示中文“人员”，不显示置信度。

## 5. 场景二：识别人和车辆

```json
{
  "input_url": "rtsp://摄像头地址/路径",
  "model": "yolo26s.pt",
  "classes": [0, 2, 3, 5, 7],
  "conf": 0.25,
  "bitrate": "2500k"
}
```

常用 COCO 类别：

| ID | 类别 | ID | 类别 |
| --- | --- | --- | --- |
| 0 | 人员 | 1 | 自行车 |
| 2 | 汽车 | 3 | 摩托车 |
| 5 | 公交车 | 7 | 卡车 |
| 24 | 背包 | 26 | 手提包 |
| 39 | 瓶子 | 56 | 椅子 |
| 63 | 笔记本电脑 | 67 | 手机 |

`classes` 省略或传 `null` 表示显示模型支持的全部类别。增加 `classes` 数量不会
让 YOLO 重复运行，也不会改变模型本身的推理次数；主要增加可见框和 OSD 数量，
并可能明显增加误检观感。生产环境只配置业务真正需要的类别。

## 6. 场景三：只在指定区域显示普通识别框

顶层 `roi` 用于普通 YOLO 框过滤。目标框中心不在多边形内时不显示。

```json
{
  "input_url": "rtsp://摄像头地址/路径",
  "model": "yolo26s.pt",
  "classes": [0],
  "roi": [
    [0.10, 0.15],
    [0.90, 0.15],
    [0.85, 0.90],
    [0.15, 0.90]
  ]
}
```

坐标是归一化坐标，左上角为 `[0,0]`，右下角为 `[1,1]`。例如 1920×1080
画面中的像素点 `(960,540)` 对应 `[0.5,0.5]`。

注意：顶层 `roi` 不产生停留或垃圾事件。需要告警时必须配置
`event_detection.rois`。

## 7. 场景四：中国车牌识别跟随

适用于车辆经过时，在视频上持续跟随显示识别后的中国车牌。

```bash
curl -X POST 'http://14.21.88.97:38080/v1/streams' \
  -H 'X-API-Key: 你的API密钥' \
  -H 'Content-Type: application/json' \
  -d '{
    "input_url": "rtsp://摄像头账号:密码@摄像头地址/路径",
    "model": "yolo26s.pt",
    "classes": [0],
    "conf": 0.25,
    "bitrate": "2500k",
    "license_plate": {
      "enabled": true,
      "detector_interval": 0,
      "recognition_reinfer_interval": 15,
      "minimum_confirmations": 2,
      "minimum_plate_confidence": 0.5,
      "vehicle_classes": [2, 3, 5, 7]
    }
  }'
```

即使 `classes` 只有 `[0]`，内部车牌支路仍会使用车辆类别，不会被关闭。
车牌宽度建议至少约 80 像素。车牌过小、运动模糊、严重倾斜或夜间反光会降低
识别率，但不会让主视频停下来等待 OCR。

## 8. 场景五：夜间人员识别

夜间模式只增强模型看到的输入，不提亮或改变转发画面。

```json
{
  "input_url": "rtsp://摄像头地址/路径",
  "model": "yolo26s.pt",
  "classes": [0],
  "conf": 0.25,
  "bitrate": "2500k",
  "night_vision": {
    "enabled": true,
    "confidence": 0.18,
    "input_gain": 1.18,
    "plate_detector_confidence": 0.20
  }
}
```

建议先用默认值。漏检时先把 `confidence` 调到 `0.16`；画面确实很暗时再把
`input_gain` 调到 `1.25`。增益过大会增加灯光附近误检，不能恢复拖影和过曝
已经丢失的细节。

白天和夜间配置不能在线修改。日夜切换需要删除旧任务后按另一套参数重新创建，
但不需要重启 Docker。

## 9. 场景六：夜间人员和车牌同时识别

```json
{
  "input_url": "rtsp://摄像头地址/路径",
  "model": "yolo26s.pt",
  "classes": [0],
  "conf": 0.25,
  "bitrate": "2500k",
  "license_plate": {
    "enabled": true,
    "detector_interval": 0,
    "recognition_reinfer_interval": 15,
    "minimum_confirmations": 2,
    "minimum_plate_confidence": 0.5,
    "vehicle_classes": [2, 3, 5, 7]
  },
  "night_vision": {
    "enabled": true,
    "confidence": 0.18,
    "input_gain": 1.18,
    "plate_detector_confidence": 0.20
  }
}
```

## 10. 场景七：人员或车辆在区域停留 20 秒告警

该场景不启用垃圾分析，只做区域停留，性能开销最小。

```bash
curl -X POST 'http://14.21.88.97:38080/v1/streams' \
  -H 'X-API-Key: 你的API密钥' \
  -H 'Content-Type: application/json' \
  -d '{
    "input_url": "rtsp://摄像头账号:密码@摄像头地址/路径",
    "model": "yolo26s.pt",
    "classes": [0, 2, 3, 5, 7],
    "conf": 0.25,
    "bitrate": "2500k",
    "event_detection": {
      "enabled": true,
      "person_classes": [0],
      "vehicle_classes": [2, 3, 5, 7],
      "rois": [
        {
          "id": "shop_entrance",
          "polygon": [[0.10,0.30],[0.90,0.30],[0.90,0.95],[0.10,0.95]],
          "dwell_enabled": true,
          "garbage_enabled": false,
          "rules": {
            "person_dwell_seconds": 20,
            "vehicle_dwell_seconds": 20
          }
        }
      ],
      "garbage": {
        "enabled": false
      }
    }
  }'
```

进入区域后画面显示橙色停留计时；达到阈值后变为红色，并生成
`zone_dwell` 事件。同一条轨迹在同一次停留期间只告警一次。目标框的底边中心点
进入多边形才算进入区域，因此 ROI 应沿地面活动范围绘制，不要沿人物头部绘制。

## 11. 场景八：监控垃圾堆新增、清理和移动

该场景关注区域内垃圾状态，不要求人员停留达到 20 秒。

```json
{
  "input_url": "rtsp://摄像头地址/路径",
  "model": "yolo26s.pt",
  "classes": [0, 2, 3, 5, 7],
  "conf": 0.25,
  "bitrate": "2500k",
  "event_detection": {
    "enabled": true,
    "person_classes": [0],
    "vehicle_classes": [2, 3, 5, 7],
    "rois": [
      {
        "id": "garbage_area",
        "polygon": [[0.15,0.35],[0.90,0.35],[0.90,0.95],[0.15,0.95]],
        "dwell_enabled": false,
        "garbage_enabled": true,
        "rules": {
          "actor_leave_grace_seconds": 3,
          "actor_association_seconds": 60,
          "garbage_persistence_seconds": 15,
          "minimum_change_area": 0.002
        }
      }
    ],
    "garbage": {
      "enabled": true,
      "analysis_fps": 3,
      "minimum_confidence": 0.35,
      "display_detections": true,
      "display_hold_seconds": 1.5,
      "maximum_display_boxes": 20,
      "prompts": [
        "plastic bottle",
        "garbage bag",
        "plastic bag",
        "cardboard box",
        "paper waste",
        "can",
        "trash pile",
        "waste"
      ]
    }
  }
}
```

可能产生：

| 事件类型 | 含义 |
| --- | --- |
| `unattended_garbage` | 区域出现持续垃圾，但没有可靠关联到人员/车辆 |
| `garbage_removed` | 原有垃圾减少，通常表示被清理 |
| `garbage_moved` | 垃圾位置发生变化 |
| `garbage_interaction_uncertain` | 画面变化存在，但语义证据不足，需要人工确认 |

`prompts` 对应镜像中已经导出的固定模型标签。可以选择其子集，但不能在服务器
运行时增加模型没有导出的新词。建议生产初期保持默认词表。

普通识别结果使用绿色中文框，例如“垃圾：垃圾堆”“垃圾：塑料瓶”。任务启动
前已经存在的垃圾也可以显示。候选变化时，同位置框改为橙色；确认告警后改为
红色。垃圾分析仍是低帧率旁路，主画面会逐帧绘制缓存框而不会等待垃圾模型。

## 12. 场景九：疑似乱丢垃圾

这是“人物/车辆轨迹 + 离开 + 垃圾新增 + 持续存在”的联合规则。不是简单判断
人物和垃圾框重叠 20 秒。

```json
{
  "input_url": "rtsp://摄像头地址/路径",
  "model": "yolo26s.pt",
  "classes": [0, 2, 3, 5, 7],
  "conf": 0.25,
  "bitrate": "2500k",
  "event_detection": {
    "enabled": true,
    "person_classes": [0],
    "vehicle_classes": [2, 3, 5, 7],
    "rois": [
      {
        "id": "shop_front_ground",
        "polygon": [[0.12,0.42],[0.92,0.42],[0.92,0.96],[0.12,0.96]],
        "dwell_enabled": false,
        "garbage_enabled": true,
        "rules": {
          "actor_leave_grace_seconds": 3,
          "actor_association_seconds": 60,
          "garbage_persistence_seconds": 15,
          "minimum_change_area": 0.002
        }
      }
    ],
    "garbage": {
      "enabled": true,
      "analysis_fps": 3,
      "minimum_confidence": 0.35
    },
    "webhook": {
      "url": "https://你的业务系统.example.com/rtsp-events",
      "timeout_seconds": 3
    }
  }
}
```

推荐过程：

1. 人员或车辆进入垃圾 ROI，系统保存轨迹和活动范围；
2. 人员离开后等待 `actor_leave_grace_seconds`；
3. 检测垃圾语义面积和背景发生变化；
4. 变化持续 `garbage_persistence_seconds`；
5. 根据空间和时间关联生成 `suspected_littering`；
6. 实时画面显示“疑似乱丢垃圾”，同时保存事件和截图；
7. 业务系统或人员进行确认、驳回。

如果垃圾被拿走，系统应倾向生成 `garbage_removed`，而不是乱丢垃圾。仅有像素
变化但垃圾模型不能确认时，只产生待确认事件。

## 13. 场景十：全部功能同时开启

适用于单路综合验收。正式运行时不要直接把 ROI 设置成全画面。

```json
{
  "input_url": "rtsp://摄像头地址/路径",
  "model": "yolo26s.pt",
  "classes": [0, 2, 3, 5, 7],
  "conf": 0.25,
  "iou": 0.45,
  "imgsz": 640,
  "bitrate": "4096k",
  "license_plate": {
    "enabled": true,
    "detector_interval": 0,
    "recognition_reinfer_interval": 15,
    "minimum_confirmations": 2,
    "minimum_plate_confidence": 0.5,
    "vehicle_classes": [2, 3, 5, 7]
  },
  "night_vision": {
    "enabled": false
  },
  "event_detection": {
    "enabled": true,
    "person_classes": [0],
    "vehicle_classes": [2, 3, 5, 7],
    "rois": [
      {
        "id": "shop_front",
        "polygon": [[0.10,0.35],[0.92,0.35],[0.92,0.96],[0.10,0.96]],
        "dwell_enabled": true,
        "garbage_enabled": true,
        "rules": {
          "person_dwell_seconds": 20,
          "vehicle_dwell_seconds": 20,
          "actor_leave_grace_seconds": 3,
          "actor_association_seconds": 60,
          "garbage_persistence_seconds": 15,
          "minimum_change_area": 0.002
        }
      }
    ],
    "garbage": {
      "enabled": true,
      "analysis_fps": 3,
      "minimum_confidence": 0.35,
      "display_detections": true,
      "display_hold_seconds": 1.5,
      "maximum_display_boxes": 20
    }
  }
}
```

如果当前是夜间，把 `night_vision.enabled` 改为 `true`，并增加夜间三个参数即可。

## 14. 多个区域

一个任务最多可配置 16 个事件 ROI。每个区域可以有独立规则：

```json
{
  "event_detection": {
    "enabled": true,
    "rois": [
      {
        "id": "door_dwell",
        "polygon": [[0.1,0.3],[0.5,0.3],[0.5,0.9],[0.1,0.9]],
        "dwell_enabled": true,
        "garbage_enabled": false,
        "rules": {
          "person_dwell_seconds": 20,
          "vehicle_dwell_seconds": 30
        }
      },
      {
        "id": "roadside_garbage",
        "polygon": [[0.5,0.4],[0.95,0.4],[0.95,0.95],[0.5,0.95]],
        "dwell_enabled": false,
        "garbage_enabled": true,
        "rules": {
          "garbage_persistence_seconds": 15,
          "minimum_change_area": 0.002
        }
      }
    ],
    "garbage": {
      "enabled": true
    }
  }
}
```

不要让多个垃圾 ROI 大面积重叠，否则同一变化可能在多个区域分别产生事件。

## 15. 事件查询和人工复核

查询某一路最近事件：

```bash
curl -H 'X-API-Key: 你的API密钥' \
  'http://14.21.88.97:38080/v1/streams/STREAM_ID/events?limit=100'
```

查询全部流事件：

```bash
curl -H 'X-API-Key: 你的API密钥' \
  'http://14.21.88.97:38080/v1/events?limit=100'
```

查询事件详情和下载截图：

```bash
curl -H 'X-API-Key: 你的API密钥' \
  'http://14.21.88.97:38080/v1/events/EVENT_ID'

curl -H 'X-API-Key: 你的API密钥' \
  -o event.jpg \
  'http://14.21.88.97:38080/v1/events/EVENT_ID/snapshot'
```

人工确认或驳回：

```bash
curl -X POST -H 'X-API-Key: 你的API密钥' \
  'http://14.21.88.97:38080/v1/events/EVENT_ID/confirm'

curl -X POST -H 'X-API-Key: 你的API密钥' \
  'http://14.21.88.97:38080/v1/events/EVENT_ID/reject'
```

事件初始状态为 `pending`。确认后为 `confirmed`，驳回后为 `rejected`。
删除流不会删除历史事件和截图。

## 16. Webhook 对接

在 `event_detection.webhook.url` 填写业务系统接收地址。系统创建事件后会异步
POST 事件 JSON。接收方应快速返回 HTTP 2xx，把短信、电话或工单等慢操作放到
自己的后台队列中。

Webhook 失败不会阻塞 RTSP 视频，但业务系统仍应定期通过 `/v1/events` 补偿
拉取，避免只依赖一次通知。

## 17. 查询、监控和停止任务

列出所有任务：

```bash
curl -H 'X-API-Key: 你的API密钥' \
  'http://14.21.88.97:38080/v1/streams'
```

重点监控 `GET /v1/streams/{stream_id}` 返回的：

- `metrics.publish_fps`：输出 RTSP 实际发布帧率；
- `metrics.pre_encode_fps`：进入编码器前的帧率；
- `metrics.duplicate_publish_fps`：应长期为 0；
- `metrics.pipeline_healthy`：应为 `true`；
- `metrics.garbage_analysis_fps`：垃圾旁路分析帧率，默认约 3；
- `metrics.total_plate_detections`：车牌检测累计数；
- `metrics.total_plate_reads`：有效车牌文字累计数。

停止任务：

```bash
curl -X DELETE -H 'X-API-Key: 你的API密钥' \
  'http://14.21.88.97:38080/v1/streams/STREAM_ID'
```

播放器关闭不代表服务器任务停止。业务系统应在以下任一条件发生时调用 DELETE：

- 用户明确关闭识别任务；
- 摄像头或业务对象被删除；
- 切换白天/夜间配置；
- 修改模型、类别、ROI 或告警阈值；
- 输入流长期不可用且业务决定不再重连。

## 18. 生产验收标准

每个机位至少连续运行 30 分钟，建议按以下标准验收：

1. `publish_fps` 和 `pre_encode_fps` 接近源流帧率；25 FPS 源建议不低于 24；
2. `duplicate_publish_fps` 为 0，`pipeline_healthy=true`；
3. 播放器无周期性停顿、大片马赛克和持续 H.264 解码错误；
4. 人员和车辆框在移动时连续跟随，不出现明显旧框滞留；
5. 区域停留在阈值附近只触发一次；
6. 丢垃圾、捡垃圾、路过、保洁、阴影和车灯分别回放验证；
7. 车牌足够清晰时 `total_plate_reads` 增长且文字稳定；
8. Webhook 中断不会影响视频，恢复后业务系统能补偿查询事件。

垃圾事件建议先作为“疑似事件”进入人工复核，不建议在完成现场样本验收前直接
用于处罚或不可逆业务动作。

## 19. 配置原则

- 优先缩小和画准 ROI，再调置信度；
- 默认省略 `output_fps`，让输出跟随输入流帧率；只有明确需要降帧时才设置；
- 输出码率通常设为输入实际码率附近，而不是盲目设置成摄像头标称最高码率；
- 1080p 监控建议从 `2500k` 开始，复杂运动画面可试 `3500k～4096k`；
- 垃圾误报多时先提高持续时间或缩小 ROI；
- 垃圾漏报时逐步降低 `minimum_confidence`，每次只改一个参数；
- 不要把店门外整幅画面作为停留 ROI，否则长期停放车辆都会正常触发告警；
- 新机位先做单路完整验收，再逐步增加并发。

## 20. 平台限制

- 完整的车牌、夜间和事件功能要求 Ubuntu + NVIDIA GPU + DeepStream 后端；
- macOS/MPS 可测试基础 YOLO RTSP 链路，但不能等价验证 DeepStream 生产链路；
- 当前垃圾模型使用固定英文提示词及其对应中文业务含义；
- API 任务保存在进程内存中，API 容器重启后需要业务系统重新创建流；
- RTSP 和公网 HTTP 默认不加密，生产环境应使用专线 ACL、VPN、RTSPS 或 HTTPS
  反向代理保护摄像头密码和 API Key。
