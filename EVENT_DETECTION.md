# 区域停留、垃圾变化与疑似乱丢垃圾

该功能只支持 DeepStream 后端。它不会用单帧画面直接判断“某人在乱丢”，而是
组合四类证据：

1. 主 YOLO + NvDCF 持续跟踪人员和车辆；
2. ROI 状态机计算真实停留时间和人员离开时间；
3. 低帧率旁路可以选择两种独立模型：`items` 使用固定词表 YOLO-World 识别
   瓶、袋、纸箱和易拉罐，`pile` 使用街景模型识别垃圾组成并聚类为垃圾堆；
4. 160×90 背景变化检测补充发现语义模型漏掉的场景变化，并补偿全局亮度变化。

只有“人员/车辆与区域发生交互、离开后垃圾语义面积增加、变化持续达到阈值”
才生成 `suspected_littering`。只有背景变化但垃圾模型不能确认时生成
`garbage_interaction_uncertain`，不会直接判为乱丢。垃圾减少生成
`garbage_removed`，位置变化生成 `garbage_moved`，因此捡垃圾不会按乱丢处理。

## 创建请求

```bash
curl -X POST 'http://服务器:8080/v1/streams' \
  -H 'X-API-Key: API密钥' \
  -H 'Content-Type: application/json' \
  -d '{
    "input_url": "rtsp://账号:密码@摄像头/Streaming/Channels/101",
    "model": "yolo26s.pt",
    "classes": [0],
    "conf": 0.25,
    "imgsz": 640,
    "bitrate": "2500k",
    "event_detection": {
      "enabled": true,
      "person_classes": [0],
      "vehicle_classes": [2, 3, 5, 7],
      "rois": [
        {
          "id": "shop_entrance",
          "polygon": [[0.10,0.20],[0.90,0.20],[0.90,0.95],[0.10,0.95]],
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
        "detection_mode": "items",
        "background_change_enabled": true,
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
      },
      "webhook": {
        "url": null,
        "timeout_seconds": 3
      }
    }
  }'
```

主请求的 `classes:[0]` 只控制普通 YOLO 框的显示，不会关闭事件内部对车辆类别
2/3/5/7 的跟踪。`garbage.prompts` 必须来自镜像内固定标签文件；离线 ONNX 无法
在服务器运行时临时增加新提示词。

## 实时画面

- 黄色 ROI：事件分析区域；
- 绿色垃圾框：垃圾模型当前识别到的垃圾堆、垃圾袋、塑料瓶等普通结果；
- 橙色框和文字：停留计时、疑似新增/移动/清理，尚未达到持续阈值；
- 红色框和文字：已达到规则阈值的事件；
- `疑似乱丢垃圾` 或 `发现新增垃圾` 会在垃圾仍存在时保留；清理/移动提示显示
  10 秒后自动消失；
- 画面不显示置信度。

`display_detections=true`时，即使垃圾在任务启动前已经存在，也会作为普通垃圾
结果持续显示。垃圾旁路默认只分析3 FPS，主画面会缓存最近坐标并在每个输出帧
绘制；`display_hold_seconds`控制旁路暂时漏检或停顿时保留最近框的时间，默认
1.5秒。`maximum_display_boxes`限制每路最多绘制的普通垃圾框，避免极端误检时
OSD数量失控。发生候选事件时，同一区域普通绿框被橙框替代；事件确认后再变成
红框，因此不会叠加三种颜色。

垃圾检测分支使用容量为 1 的丢帧队列和 appsink。处理不过来时只丢垃圾分析帧，
不会积压主画面；主链仍是 NVDEC → 主 TensorRT → NvDCF → GPU OSD → NVENC →
RTSP。垃圾支路按单调时钟采样而不是假定摄像头固定为 25 FPS。默认分析 3 FPS，
因此事件确认延迟约为配置的持续时间再加 0～0.4 秒，而输出视频仍应跟随源流
自身帧率。

## 只识别垃圾堆

垃圾堆不是一个形状固定的物体。`pile`模式先用街景垃圾模型检测堆内多个组成
部分，再按空间距离聚类并合并为一个稳定的`垃圾堆`框，不显示瓶子、纸张等
零散类别名称。固定机位只需要展示当前垃圾堆时，建议使用：

```json
"garbage": {
  "enabled": true,
  "detection_mode": "pile",
  "analysis_fps": 1,
  "minimum_confidence": 0.15,
  "background_change_enabled": false,
  "display_detections": true,
  "display_hold_seconds": 3,
  "maximum_display_boxes": 5,
  "minimum_pile_detections": 2,
  "pile_merge_distance": 0.18,
  "pile_box_padding": 0.04
}
```

`background_change_enabled=false`只关闭像素变化证据，不关闭垃圾堆模型和绿框。
全画面 ROI 中有树叶、车流或大面积阴影时建议关闭，否则背景变化可能把树木或
路面运动画成橙色候选框。需要判断“新增、移动、清理”时再开启，并把 ROI 缩到
实际垃圾投放区域。`pile`模式推荐 1 FPS；主视频仍按源流约 25 FPS 输出，最近
垃圾堆坐标会在主链每一帧上绘制。

## 事件接口

```bash
# 查询某路事件
curl -H 'X-API-Key: API密钥' \
  'http://服务器:8080/v1/streams/STREAM_ID/events?limit=100'

# 查询单个事件
curl -H 'X-API-Key: API密钥' \
  'http://服务器:8080/v1/events/EVENT_ID'

# 获取垃圾事件截图
curl -H 'X-API-Key: API密钥' \
  -o event.jpg \
  'http://服务器:8080/v1/events/EVENT_ID/snapshot'

# 人工复核
curl -X POST -H 'X-API-Key: API密钥' \
  'http://服务器:8080/v1/events/EVENT_ID/confirm'
curl -X POST -H 'X-API-Key: API密钥' \
  'http://服务器:8080/v1/events/EVENT_ID/reject'
```

事件类型包括 `zone_dwell`、`suspected_littering`、`unattended_garbage`、
`garbage_removed`、`garbage_moved` 和 `garbage_interaction_uncertain`。事件 JSON
与截图保存在 `./data/events`，删除流不会删除历史事件。Webhook 在有界后台
队列发送；超时或接收方卡住不会阻塞视频处理。

## 能力边界和现场调参

这是一套“疑似事件筛查”系统，不是法律意义上的行为定性。公开预训练模型对
常见瓶、袋、箱和成堆垃圾可直接使用，但小物体、遮挡、夜间红外、强反光和
远距离目标仍可能漏检。背景变化能发现画面变了，却无法仅凭像素理解人的意图，
所以系统保留 `pending / confirmed / rejected` 人工复核闭环。

上线前应为每个固定机位标注 ROI，并用该机位白天、夜间、下雨、保洁清理和
正常路人片段做回放验收。建议先保持默认阈值；误报多时先缩小 ROI 或增大
`garbage_persistence_seconds`，漏报多时再逐步降低 `minimum_confidence`，不要
直接把所有阈值降到最低。

YOLO-World 权重与 Ultralytics 代码适用其上游许可证。用于商业闭源交付前应由
使用方确认相应 AGPL/企业许可义务。
