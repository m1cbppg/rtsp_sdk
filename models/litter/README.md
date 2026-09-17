# 零散垃圾预训练权重筛查（2026-09-09）

现场：荷兴广场本地昼夜 PS 录像。需求：不训练，直接使用公开预训练权重。

当前优先实验候选为 **Turhancan YOLOv8m-seg + 640 原生像素 ROI 分块**。在相同的
5 个目视参考地面物、24 次可见画面点检中，它比 CatSat 定位更连续；仍有明显的
人物/电动车/设施误报。尚未选出稳定拿来即用的成品，没有总体准确率或已确认的“最佳模型”。
原 YOLO-World / CatSat 最佳表述均撤回。详见 [评测报告](../../LITTER_MODEL_EVALUATION.md)。

按用户要求于2026-09-09清理评测产物，**当前仅保留 `turhancan_yolov8m_seg_trash.pt` 权重**。
下表其余8款为历史评测记录，权重已删除；来源和校验值保留在 manifest 中，必要时可重新下载。
筛查 JSON 和少量关键对照图片仍保留，批量截图、临时推理依赖和解压视频副本已清理。

| 本地文件 | 发布者与公开来源 | 发布者声明许可 | 本地状态 |
| --- | --- | --- | --- |
| `turhancan_yolov8m_seg_trash.pt` | [Turhancan](https://huggingface.co/turhancan97/yolov8-segment-trash-detection) | MIT | 昼夜 58 帧分块；优先实验候选，但误报明显 |
| `catsat_litter_yolo11s_seg.pt` | [CatSat](https://huggingface.co/CatSat/yolov11-litter-materials) | MIT | 昼夜 58 帧分块、整图对照；小物点检低于 Turhancan |
| `esapzoi_litter_yolov8.pt` | [Esapzoi](https://huggingface.co/esapzoi/litter-detection-yolov8) | MIT | 昼夜 58 帧分块，场景大框误报 |
| `littercam_yolov9c.pt` | [LitterCam](https://huggingface.co/aryanshh/littercam-yolov9c) | Apache-2.0 | 原生 YOLOv9 双头，昼夜 58 帧分块，参考地面物无有效定位 |
| `alope_trash_yolo11n.pt` | [Alope](https://huggingface.co/Alope/trash-detection-yolo11n) | AGPL-3.0 | 旧整图/排除区配置，昼夜 60 帧无保留框 |
| `oguri_trash_yolo11l.pt` | [Oguri](https://huggingface.co/Oguri02/trash-detection-yolo11l) | AGPL-3.0 | 旧配置昼夜 60 帧；paper 常为墙面等误报 |
| `bower_yolov8m_object_material.pt` | [Bower](https://huggingface.co/BowerApp/bowie-yolov8-multihead-trash-detection) | MIT | 仅历史白天代表帧初筛，无检测；未完成昼夜系统评测 |
| `trash_detection_best.pt` | [Baraa Lazkani](https://github.com/BaraaLazkani/trash-detection-yolov8) | Custom Academic License | 旧配置昼夜 60 帧无保留框；商业用途需书面许可 |
| `jhandry_trash.pt` | [Jhandry](https://huggingface.co/Jhandry/TrashDetection) | OpenRAIL（具体条款未核清） | 已下载，缺少 RetinaNet 配套代码；加载失败，不算完成测试 |

许可为模型卡/仓库的发布者声明，未完成底层数据和代码的完整授权审计。不能因 HF 标记 MIT
或 Apache 就认定整个商业部署无附带义务；Ultralytics、原生 YOLOv9、数据各有条款。
AGPL 允许商业使用但有相应义务；Baraa 的非商业限制则明确排除无授权商用。

[manifest.json](manifest.json) 记录本地文件大小、SHA-256、来源与可查版本。
权重均由根目录 `.gitignore` 的 `*.pt` 排除，不提交、不随生产镜像自动打包。
此目录中的下载文件不代表任何模型已经接入 API 或通过生产验收。

保留候选的固定下载信息：

- 仓库：`turhancan97/yolov8-segment-trash-detection`
- 公开文件：`yolov8m-seg.pt`
- 公开版本：`89f1b197852760ee110213e655400a7187da03d2`
- 本地 SHA-256：`a2f8de0c7f714e2ab8b70c62490e2a41fd4a6681ca4a8dd442797c809a140278`

加载：`YOLO('models/litter/turhancan_yolov8m_seg_trash.pt')`。
材料类别到“遗弃在地面上的垃圾”的业务判断仍需单独处理。
