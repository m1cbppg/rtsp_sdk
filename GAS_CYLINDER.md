# 固定机位燃气瓶实时识别

该功能只支持DeepStream后端。输入和输出都是RTSP，燃气瓶模型运行在容量为1
的丢帧旁路，并在独立进程中运行PyTorch/YOLOE；主视频仍按原始帧率执行NVDEC、
主模型、OSD、NVENC和RTSP发布。YOLOE处理不过来或CUDA调用异常时只丢弃旧分析
帧，不会持有主DeepStream进程的Python GIL，也不会丢弃或积压待编码的视频帧。

## 创建任务

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
    "bitrate": "4000k",
    "gas_cylinder": {
      "enabled": true,
      "profile_id": "camera_01_ir",
      "analysis_fps": 1,
      "sample_count": 11,
      "sample_interval_seconds": 3,
      "minimum_confirmations": 4,
      "scene_stable_seconds": 3,
      "change_confirm_seconds": 3,
      "forced_refresh_seconds": 300,
      "alarm_threshold": 18,
      "display_ids": false,
      "include_partial": true
    }
  }'
```

返回的`rtsp_url`播放方式不变。流状态会增加：

```json
{
  "gas_cylinder": {
    "enabled": true,
    "profile_id": "camera_01_ir",
    "alarm_threshold": 18,
    "alarm_active": true,
    "state": "stable",
    "count": 36,
    "result_version": 1,
    "last_updated_at_unix": 1785840000.0
  }
}
```

状态含义：

- `sampling`：启动或定期采集11张样本；
- `stable`：显示已确认的绿色框和当前数量；
- `dirty`：场景发生持续变化，旧结果以橙色标记并等待画面稳定；
- `error`：YOLOE旁路异常，主RTSP继续播放并保留上次结果。

当识别数量大于`alarm_threshold`（默认18）时，左上角统计背景和全部燃气瓶
识别框变为红色，并显示“超量告警”；数量等于阈值时不触发告警。

## 运行方式

当前机位使用`YOLOE-26L-seg`和三组视觉提示：普通瓶、顶部暗色瓶和画面底部
残缺瓶。模型只加载一次，三组提示只缓存视觉特征。启动时每3秒采集
一张图片，共11张；同一空间目标至少出现4次才进入稳定结果。该窗口覆盖红外
摄像头的自动曝光周期。得到首次稳定结果后，YOLOE独立进程会主动退出并释放CUDA
上下文；主流继续缓存和绘制稳定框。当前固定仓库机位不自动周期复核，瓶子发生实质
变化时应显式重建任务，以优先保证长期转发始终流畅。

在用户提供的录像上，11张样本的稳定结果为36个可见燃气瓶。这个数字只代表
可见目标，完全遮挡或位于画面外的瓶子无法从视频中统计。

## 视频质量约束

- 燃气瓶分支必须保持`max-size-buffers=1`、`leaky=2`；
- 主发布队列必须保持非丢帧，不能为燃气瓶推理改成`leaky`；
- OSD每个DisplayMeta最多批量绘制4个燃气瓶框，默认不显示逐框文字；
- H.264仍由NVENC编码，继续插入AUD、SPS和PPS，GOP默认25且不使用B帧；
- 720p建议从`4000k`开始验收，1080p建议从`6000k`开始，最终按源画面复杂度
  调整。叠框需要重新编码，因此无法做到码流逐字节无损，但不应出现持续马赛克、
  花屏、节奏性丢帧或延迟不断增长。

## 离线资源

构建镜像前必须存在：

```text
models/gas/yoloe-26l-seg.pt
models/gas/profiles/camera_01_ir.json
models/gas/profiles/camera_01_ir.jpg
```

新摄像头不需要训练，但固定机位、焦距、红外/彩色模式变化较大时，需要制作新的
参考图和Profile。摄像头被移动或分辨率改变后，必须重新校准Profile。

## 服务器验收

至少连续运行10分钟，同时检查：

```bash
curl -H 'X-API-Key: API密钥' http://127.0.0.1:8080/v1/streams
nvidia-smi dmon -s pucvmet
ffprobe -rtsp_transport tcp -v error -show_streams '输出RTSP地址'
```

最低验收线：`publish_fps`和`unique_publish_fps`持续不低于20，
`duplicate_publish_fps=0`，`pipeline_healthy=true`；启动识别时播放端不能出现明显
停顿或马赛克。还应使用一段增减燃气瓶后的录像验证重建任务后能够得到新的稳定
结果，不能只测试当前固定数量。
