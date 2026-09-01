# 船舶识别旁路

## 目标与边界

第一阶段只回答“画面中哪里有船”，尽量提高召回率并在输出RTSP中画中文`船`
框。默认复用项目已有`yolo26s.pt`的COCO `boat`类别（零基类别ID `8`），不训练
新模型。船名、船号、证照和“正规/非法”判断不在本阶段范围内；后续高清机位可
把船框作为OCR输入，再与AIS、船证、禁渔区和作业时段做规则核验。

对于全景中小到现有船模型无法分类的目标，可以显式开启
`small_target_proposals`生成水面运动和局部明暗外观候选，再配合PTZ近景确认。局部
外观分支会保留背景已经适应的近似静止暗色船体或航行灯。候选类别不会
被当作已确认船，也不会进入疑似捕捞行为评分。完整闭环、去重和证据接口见
[PTZ_VESSEL_VERIFICATION.md](PTZ_VESSEL_VERIFICATION.md)。

这套旁路只在DeepStream后端启用。主链继续以原FPS执行NVDEC、TensorRT、OSD、
NVENC和RTSP发布；旁路按默认5 FPS取最新帧，即使船舶模型变慢或退出，也不会让
主视频排队或断流。

## 推荐请求

下面的水域形状和右下排除区是按本次`8月12日.mp4`画面的起始建议，上线前仍应
用现场画面复核。其他摄像头必须重画`roi`，避免把岸上塔架、桥墩和反光纳入
候选；`exclude_rois`用于排除水域内仍会反复误报的固定结构。

```json
{
  "input_url": "rtsp://camera-user:camera-pass@192.168.10.20:554/live",
  "model": "yolo26s.pt",
  "conf": 0.25,
  "imgsz": 640,
  "bitrate": "2500k",
  "vessel_detection": {
    "enabled": true,
    "model": null,
    "analysis_fps": 5,
    "confidence": 0.10,
    "iou": 0.45,
    "imgsz": 1280,
    "class_ids": [8],
    "input_width": 1920,
    "input_height": 1080,
    "inference_regions": [
      [0.0, 0.0, 1.0, 1.0],
      [0.34, 0.44, 1.0, 0.92]
    ],
    "roi": [
      [0.0, 0.61],
      [0.42, 0.54],
      [0.58, 0.49],
      [1.0, 0.49],
      [1.0, 1.0],
      [0.0, 1.0]
    ],
    "exclude_rois": [
      [[0.96, 0.70], [1.0, 0.70], [1.0, 1.0], [0.96, 1.0]]
    ],
    "minimum_hits": 2,
    "hold_seconds": 1.0,
    "duplicate_containment_threshold": 0.80,
    "large_box_area_threshold": 0.20,
    "large_box_minimum_confidence": 0.25,
    "maximum_box_area": 1.0,
    "display_ids": false,
    "display_roi": true,
    "display_proposals": false,
    "small_target_proposals": false
  }
}
```

`inference_regions`是真正的推理裁剪，不只是结果过滤。上例同时分析全画面和一块
放大的中远水域，重叠结果会统一映射回原画面并做NMS。每增加一个区域，旁路推理
量也近似增加一份；先以一个全画面区域压测，再根据远船漏检情况增加第二个区域。

`roi`和`exclude_rois`按检测框中心点过滤。所有坐标都是相对画面宽高的0到1比例，
所以更换同视角的分辨率后无需重画；更换机位必须单独标定。

顶层`roi`约束主链640模型，`vessel_detection.roi`约束高分辨率船舶旁路；如果两边都
可能画船框，必须同时配置，否则固定障碍物可能仍从另一条分支显示。`proposal_roi`
只限制未分类小目标，可进一步收窄到中远水面，不会缩小正常`boat`检测范围。

全图和放大分区可能给同一艘船生成大小不同的嵌套框。
`duplicate_containment_threshold`只在不同推理分区之间生效：小框有80%以上被另一
框包含时保留置信度较高的一个，不会把同一分区内紧邻的多艘小船误合并。

`large_box_area_threshold`表示框面积占整幅画面的比例；超过该比例且置信度低于
`large_box_minimum_confidence`时会被过滤，用于压制堤岸、厂房等大面积低分误报。
`maximum_box_area`是硬上限，默认1.0等于关闭。近景可能出现占画面很大的真船，
所以修改这两个面积参数前必须用该机位的近景船视频回归，不能只看无船画面。

## 新机位自动标定（无需训练）

容器内置MIT许可的OpenMMLab UPerNet ConvNeXt Tiny ADE20K预训练语义分割模型。
它只在部署新机位时离线抽帧，找出多帧稳定的`water/sea/river/lake`证据，再沿
水域上边界生成保守的下方包络，不进入实时推流链路。保守包络会容纳遮住水面的
密集船只和栈桥，避免直接使用语义掩码时漏掉码头内的真船。运行：

```bash
docker compose -f docker-compose.deepstream.api.yml run --rm api \
  python3 -m rtsp_annotator.vessel_calibration \
  --input 'rtsp://user:password@camera/live' \
  --output /app/runtime/camera-01-vessel.json \
  --preview /app/runtime/camera-01-vessel.jpg \
  --vessel-model /app/models/yolo26s.pt \
  --sample-count 12 \
  --sample-interval 2
```

输出JSON中的`vessel_detection.roi`和`inference_regions`可复制到创建流请求。预览图
中绿色区域是建议水域，红框只表示检测位置在采样期间稳定，并不代表它是误报。
工具不会生成`exclude_rois`：长期靠泊的真船同样可能稳定不动，必须由人确认红框
内是桥墩、浮标或固定建筑后，才可从`stable_detection_review_candidates`手工复制
对应`polygon`。若分割结果不完整，宁可用
`yolo-roi-select`人工重画，也不要上线一个会漏掉航道的ROI。

## 为什么不是把主模型阈值直接降到0.10

主模型固定640输入并逐帧服务主链。在本次2702×1520、雨雾和远船素材中，640会
把远船压成很少的像素；直接降阈值主要增加塔架、桥墩等误报。船舶旁路解决的是
三个不同问题：1280输入保留远船细节，透视分区进一步放大重点水域，连续2次命中
才显示并短暂保留1秒，从而允许原始候选阈值降到0.10。

当主链也显示COCO船框时，OSD会对IoU较高的主链框和旁路框去重；旁路仍负责补出
主链640输入漏掉的船。

## 参数调优顺序

1. 先画准确的`roi`，把天空、岸上建筑和近景道路排除。
2. 将稳定重复误报的桥墩、塔架加入`exclude_rois`，不要先提高阈值。
3. 远船仍漏检时增加一个中远水域`inference_regions`，然后观察GPU余量。
4. 仍需提高召回时把`confidence`从0.10逐步降到0.08；如果误报增多，把
   `minimum_hits`从2调到3。
5. 船框断续时先把`hold_seconds`调到1.5，不要无上限增加`analysis_fps`。

如果目标小到船模型完全没有候选，而且摄像机可控，再开启
`small_target_proposals`。本次清远实景录像的离线初值为运动阈值60、局部外观阈值
18、局部背景模糊31像素、面积20到1000像素、宽至少4像素、高至少3像素、连续4次
命中；应先用回放脚本测每帧候选数。外观阈值越低越容易包含水波、岸灯和固定结构，
因此必须同时校准ROI、排除区和PTZ负结果冷却，而不能把候选直接认定为船。
转发画面默认`display_proposals:false`，只显示已识别的船；疑似候选仍可在后台引导
PTZ。现场浪纹较多时，优先配置`proposal_roi`、`proposal_minimum_motion_ratio`、
`proposal_minimum_fill_ratio`和`proposal_maximum_candidates`，不要仅隐藏OSD。
这些ROI只约束HOME全景候选；进入PTZ近景复核后会自动改用近景检测配置，避免全景
水线在转向和放大后错误过滤已经居中的船。

`input_width/input_height`不能高于DeepStream的mux尺寸。未来接入2K/4K摄像头并
希望保留真实细节时，必须同步提高API配置中的
`deepstream.mux_width/mux_height`；只
把旁路参数写成4K会被API拒绝，因为已在mux阶段丢失的像素无法通过放大恢复。

## 验收

不要用单帧准确率判断。每个典型机位至少准备白天、逆光、雨雾、夜间各20分钟，
以一次“船舶完整经过可见水域”为一个事件统计：

- 事件召回率：实际经过的船中，至少被稳定框出一次的比例；第一阶段建议目标
  `>=90%`，恶劣雨雾和极小远船单独报告，不混入白天平均值。
- 持续覆盖率：船可见期间有框的时长占比；建议目标`>=70%`。
- 固定误报：无船30分钟内持续超过2秒的假船框次数；建议每机位`<=1`。
- 主流健康度：启用前后`publish_fps`不下降超过5%，且不能新增持续卡顿或断流。

查询`GET /v1/streams/{stream_id}`时，`metrics`会包含
`vessel_detection_state/count/result_version/last_inference_ms`。如果状态为
`error`，主RTSP仍应正常播放；这正是独立旁路的故障隔离要求。

## 后续船号与合规判定

高清机位满足船体字符高度约20像素以上后，可在已确认船框内做超分辨率、多帧
OCR和结果投票。OCR只能得到可见编号；“非法渔船”不能由外观模型可靠决定，最终
应综合AIS/MMSI、船证名录、OCR船号、地理围栏、禁渔期、航迹和作业行为，输出
`合规 / 疑似违规 / 信息不足`及证据，而不是让视觉模型直接下法律结论。
