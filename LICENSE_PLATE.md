# 中国车牌识别

该功能只支持 Ubuntu + NVIDIA GPU 的 DeepStream 后端。macOS/MPS 仍可测试
原有 YOLO 流，但不能运行本车牌链路。

## 创建

同一路流同时运行 `yolo26s.pt` 和中国车牌识别：

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
    "bitrate": "2500k",
    "license_plate": {
      "enabled": true
    }
  }'
```

`classes`只控制最终画面显示哪些 YOLO 类别。即使它是`[0]`，车牌支路仍会在
YOLO 的车、摩托车、公交车和卡车结果上工作。

默认配置：

```json
{
  "enabled": true,
  "detector_interval": 0,
  "recognition_reinfer_interval": 15,
  "minimum_confirmations": 2,
  "minimum_plate_confidence": 0.5,
  "vehicle_classes": [2, 3, 5, 7]
}
```

建议先保持默认值。`detector_interval=0`表示每帧检测车牌，确保移动中的车牌框
连续；调大该值可以降低GPU负载，但车牌框会只在检测帧出现。
降低`recognition_reinfer_interval`会更频繁地重复OCR。

## 为什么不会拖住主流

管线为：

```text
RTSP硬解码 → YOLO26s → 车辆NvDCF → 中国LPDNet
           → 轻量车牌ID跟踪 → 中国LPRNet异步分类 → 中文OSD → NVENC → RTSP
```

- LPDNet只处理YOLO框出的车辆，不扫整张1080p画面；
- LPDNet默认每帧检测，保证车牌框随车辆连续移动；
- 轻量几何跟踪只处理车牌元数据，为每块车牌分配稳定ID，不复制视频帧；
- LPRNet启用`classifier-async-mode`，视频帧不会等待字符识别完成；
- 同一track-id默认每15帧才重识别一次；
- 车牌字符串经过中国车牌格式校验和时序稳定后再显示；
- YOLO显示类别与内部车辆候选分开，所以`classes:[0]`不会关闭车牌识别。

这能从架构上避免OCR成为逐帧同步瓶颈。是否达到每路稳定25 FPS仍必须用目标
摄像头和目标并发数在服务器验收；网络、摄像头GOP和播放器缓存不由模型控制。

## 指标和验收

查询流任务时，`metrics`新增：

- `license_plate_enabled`
- `interval_plate_detections`
- `interval_plate_reads`
- `total_plate_detections`
- `total_plate_reads`

25 FPS源流建议验收：

- `publish_fps`和`pre_encode_fps`长期保持24～25；
- `pipeline_healthy=true`；
- 连续10分钟无周期性停顿、花屏、推流重启；
- 有车辆经过时`total_plate_detections`增长；
- 车牌足够清晰时`total_plate_reads`增长并显示`车牌：粤B12345`一类文字。

车牌在画面中的有效宽度建议至少80像素，尽量正对镜头并减少运动模糊、强反光
和夜间过曝。过小或严重倾斜的车牌即使不影响视频流畅度，也会降低识别率。

## 模型

镜像离线携带 NVIDIA TAO 的中国 LPDNet 与 LPRNet ONNX 模型、中文字符字典和
DS8兼容解析器。服务器不需要访问Docker Hub或NGC。首次启动时
`engine-builder`会为当前GPU构建两个额外的FP16 TensorRT引擎；换GPU、
DeepStream/TensorRT版本或模型后必须删除旧`.engine`重新构建。
