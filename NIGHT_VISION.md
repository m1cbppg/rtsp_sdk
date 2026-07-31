# DeepStream夜间识别

夜间识别是独立、显式启用的DeepStream配置。默认关闭，因此原有请求、白天
置信度、输出画面和模型分组方式保持不变。

## 创建夜间流

在`POST /v1/streams`请求中增加：

```json
{
  "input_url": "rtsp://摄像头地址/路径",
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

参数含义：

- `enabled`：启用独立夜间推理逻辑。
- `confidence`：夜间YOLO最终显示阈值，不覆盖请求中的白天`conf`。
- `input_gain`：模型输入线性增益，范围`1.0～1.5`。只在TensorRT输入归一化
  阶段使用，默认`1.18`。
- `plate_detector_confidence`：启用车牌功能时的夜间车牌检测阈值。

响应和`GET /v1/streams/{id}`会包含：

```json
{
  "night_vision": {
    "enabled": true,
    "profile": "night",
    "confidence": 0.18,
    "input_gain": 1.18
  }
}
```

`metrics.vision_profile`应为`night`，
`metrics.night_vision_enabled`应为`true`。

## 与白天逻辑的隔离

- 未传`night_vision`时等同于`enabled:false`，继续使用原来的`conf`和
  `1/255`输入归一化。
- 白天流和夜间流不会进入同一个DeepStream模型实例。
- 夜间增强只改变模型看到的输入，不修改送入OSD和NVENC的原始画面。
- 夜间没有增加第二次YOLO推理，也没有增加一次解码或编码。
- 同时保留白天流和夜间流会占用两个模型实例；正常使用应停止旧任务后创建
  新任务。

夜间模式只支持Ubuntu、NVIDIA GPU和DeepStream后端。macOS/MPS和Python
后端会返回`422`，避免静默忽略夜间参数。

## 建议调参顺序

先使用默认值连续测试至少10分钟，再按以下顺序每次只修改一个值：

1. 漏检较多：把`confidence`从`0.18`降到`0.16`。
2. 画面非常暗：把`input_gain`从`1.18`升到`1.25`。
3. 误检增加：提高`confidence`，不要继续提高`input_gain`。
4. 夜间车牌漏检：把`plate_detector_confidence`从`0.20`降到`0.16`。

不建议`input_gain`直接设置为`1.5`。线性增益不能恢复已经被拖影、过曝或
压缩丢失的细节，过高还可能使灯光附近特征失真。

## 切换日夜

当前任务的配置在创建后不可变。切换时先删除旧流，再使用白天或夜间参数重新
调用`POST /v1/streams`。API调用方可按照业务时间表执行切换，不需要修改
`api.json`或重启Docker。

切换后重点观察：

- `publish_fps`是否维持源流帧率；
- `pipeline_healthy`是否为`true`；
- `average_inference_ms`和`p95_inference_ms`是否明显增长；
- 同一段夜间录像的漏检数和误检数是否改善。
