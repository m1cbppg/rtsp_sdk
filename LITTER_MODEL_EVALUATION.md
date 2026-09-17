# 荷兴广场零散垃圾模型评估

更新：2026-09-09。范围：本地昼夜录像、预训练权重、无需训练的独立新场景。

## 当前选型判断

**优先保留 Turhancan YOLOv8m-seg + 原分辨率 ROI 分块作为实验候选；尚无已经验证能稳定拿来即用的成品。**

它在同一批可见地面物点检中，定位比 CatSat YOLO11s-seg 更连续。但存在人脸、电动车、
商店物品等明显误报，不能说综合准确率最好，也不能承诺达到某个现场识别率。
此前“CatSat 最接近/最佳”和更早“YOLO-World 最佳”的表述均撤回；候选次序以本次同输入复测为依据。

累计 **11 款不同权重完成过推理**：其中 10 款做过昼夜录像抽样，Bower 仅有白天代表帧初筛。
这 11 款包括 1 款普通 COCO 对照和原垃圾堆模型；不能说全是专门训练的零散垃圾模型。
另下载 Jhandry/TrashDetection，但缺少 RetinaNet 配套代码，加载失败，**不计为已测模型，也不记为 0 检出**。

当前证据说明这个视角能检出一部分纸片、塑料袋；尚未证明烟头、菜叶、透明薄膜等所有零散垃圾
都能可靠检出。模型自报类别或数据集成绩不等于本机位表现。

## 视频和评测方法

用户提供 `/Users/mlcbppg/Desktop/荷兴广场视频/新建文件夹.zip`：

- 夜间：`20260908000011-20260908000515.ps`，约 303.92 秒；
- 白天：`20260908115917-20260908120421.ps`，可解码时长约 283.92 秒。

均为 2560×1440、25 FPS、HEVC。视频只在本地处理，没有上传或连接生产服务器。
PS 解码存在音频头和 HEVC 参考帧警告，不能把缺失/损坏帧计为漏检。

三轮方法应区分，不能直接用有框数量比较模型：

1. 初轮 ONNX：近整图 ROI、640 输入；YOLO-World 阈值 0.25、垃圾堆 0.15、5 秒抽帧。
   后加 2×2 重叠分块、10 秒抽帧；每块宽高为原图 62.5%，约放大 1.6 倍。
2. 旧 PT：整图 1280 输入、10 秒抽帧；预测后按左侧人行道 ROI 和两块固定排除区过滤。
   **该脚本不是先裁剪 ROI 再推理。旧排除区也覆盖了一部分路面和地面小物，0 框只代表这套配置。**
3. 本轮统一复测：FFmpeg 按解码帧序号每 250 帧缓存一张，去掉昼夜各第 1 张（白天首张损坏），
   余下同一组白天 28、夜间 30，共 58 张 JPEG。使用下述暂定 ROI，不设固定排除区。
   在 ROI 包围范围内取四个重叠的 640×640 原生像素裁剪，各输入 640，再合并框、过滤中心点、
   用类别无关 NMS（IoU 0.5）去重。阈值 0.15。CatSat 另有同帧整图 1280 对照。

暂定 ROI 为 `[[0.02,0.08],[0.37,0.08],[0.46,0.96],[0.02,0.96]]`，由助手根据画面选取，
尚未由用户确认。它包含墙面、设施和路沿，不能当作最终业务检测区域。若以后缩小/排除，必须
同步重算被排除的真实垃圾，不能靠屏蔽目标“提高准确率”。

下表均为**抽帧数/有保留框帧数**，不是召回率、精确率、独立垃圾数量或连续视频稳定性。
完整逐帧结果保存在 `output/litter_eval_*.json` 和 `output/litter_screen_*/report.json`。

2026-09-09 按用户要求清理产物：只保留 Turhancan 权重、轻量逐帧 JSON、点检证据及脚本。
淘汰权重、批量带框截图、抽帧缓存、YOLOv9临时代码/依赖和临时解压视频已删除。
原始 ZIP 保留且已校验完整性；复跑推理须重新解压和抽帧。清理统计见
[cleanup_summary.json](output/litter_review/cleanup_summary.json)。下表是清理前完成的历史测试，
不是本地仍持有全部权重的清单。

## 各候选结果

| 权重 | 方法 | 夜间 | 白天 | 当前观察 |
| --- | --- | ---: | ---: | --- |
| 项目 YOLO-World 固定词表 | 初轮整图 | 61/13 | 57/17 | `can` 常落在绿色垃圾桶等固定物；未证明适合零散垃圾 |
| 同一 YOLO-World（不另算模型） | 初轮分块 | 31/18 | 29/28 | 框增加主要来自固定物；不是可确认的召回改善 |
| Street garbage pile | 初轮整图 | 61/45 | 57/27 | 遮阳伞/雨棚等误报；目标定义本身是垃圾堆 |
| YOLO26s COCO | 初轮分块 | 31/31 | 29/29 | 291/601 个普通人车等框，两段抽样没有 bottle；无纸片/袋子类 |
| Baraa Lazkani YOLOv8m 五类 | 旧 PT | 31/0 | 29/0 | 旧 ROI 过滤后无框；局部检查有大面积 Metal/Plastic 错框 |
| Turhancan YOLOv8m-seg | 旧 PT | 31/31 | 29/28 | 110/98 个框，已见人物/设施误报；不能据此判定没有真实召回 |
| 同一 Turhancan（不另算模型） | 本轮分块 | 30/30 | 28/28 | 424/186 个框；可定位多种地面物，同时存在明显误报 |
| Bower 多头模型 | 仅代表帧 | 未测 | 1/0 | 历史初筛记录：1280 输入、阈值 0.15 与 0.01 均无框；未做昼夜系统评测 |
| Alope YOLO11n | 旧 PT | 31/0 | 29/0 | 公开说明为四类垃圾，当前配置无保留框 |
| Oguri YOLO11l | 旧 PT | 31/31 | 29/28 | 31/28 个 paper 框；查看样本为墙面等大块误报 |
| CatSat YOLO11s-seg/TACO | 本轮分块 | 30/30 | 28/28 | 262/130 个框；能检出部分袋子、小物，也误报人物、雨棚和墙面 |
| 同一 CatSat（不另算模型） | 同帧整图 1280 | 30/30 | 28/28 | 204/170 个框；小地面物点检表现比其分块差，近处红袋另有改善 |
| Esapzoi YOLOv8 Litter | 本轮分块 | 30/30 | 28/24 | 119/40 个框；查看样本多为覆盖大面积场景的 Litter 框 |
| LitterCam YOLOv9-C | 本轮分块 | 30/30 | 28/28 | 227/147 个 general_litter 框，查看样本主要覆盖整个裁剪或大块场景，点检目标均未定位 |

LitterCam 用官方 WongKinYiu/yolov9 原生双头推理的第二分支完成测试；模型卡中的当前
Ultralytics 调用不能直接加载该格式。权重 SHA-256 与下载仓库 LFS 值一致。
下载 checkpoint 记录 `epoch=8`，模型卡称训练 200 epochs；早期最优 checkpoint 可以来自
更长训练，所以这本身不能证明说明错误，但本轮也没有复现其所称训练过程。

## 具体地面物点检

助手目视选出 5 个地面物，在昼夜每约 50 秒的代表画面里保留清晰可见的 24 次出现。
白天白色小物只选仍存在的 2 次；夜间红袋和透明状地面物在 `night_017` 遮挡，排除该次。
像素宽高约为：白色小物 17×18、青白色小物 27×19、夜间白色物 61×48、红袋 98×79、
透明状物 126×43。材质命名只是外观描述，不是现场实物确认。

**这些参考是在筛查后挑选的、同一目标重复出现、并非穷尽标注，尚无人类审核或独立留出集。**
只测位置，不要求材质类别正确；不能用下面的比例声称“识别率 83%”或评价总体误报率。

阈值 0.15、参考框与预测框 IoU ≥0.3 的局部匹配：

| 参考物 | 出现次数 | Turhancan 分块 | CatSat 分块 | CatSat 整图 | Esapzoi 分块 | LitterCam 分块 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 白天地面白色小物 | 2 | 1 | 0 | 0 | 0 | 0 |
| 白天地面青白色小物 | 6 | 5 | 2 | 0 | 0 | 0 |
| 夜间路沿白色物 | 6 | 6 | 4 | 0 | 0 | 0 |
| 夜间近处红色袋子 | 5 | 5 | 2 | 5 | 1 | 0 |
| 夜间近处透明状物 | 5 | 3 | 1 | 1 | 0 | 0 |
| 合计 | 24 | 20 | 9 | 6 | 1 | 0 |

将匹配要求收严到 IoU ≥0.5，同一顺序分别为 **16、7、5、1、0 /24**，候选顺序未变。
将置信度调为 0.25、保持 IoU ≥0.3，Turhancan 变为 17/24、CatSat 8/24；调为 0.4 则为
8/24、5/24。提高阈值会丢掉不少小物，不能直接当作稳定性的解决办法。

可复核文件：

- [同一物体的原图/两模型结果](output/litter_review/object_comparison.jpg)
- [跨时间参考物局部图](output/litter_review/reference_sheet.jpg)
- [明确误报示例](output/litter_review/false_positive_examples.jpg)
- [参考坐标和局限](output/litter_review/reference_objects.json)
- [阈值与 IoU 点检结果](output/litter_review/spotcheck_results.json)
- [Turhancan 白天上下文](output/litter_review/turhancan_day_context.jpg)、[夜间上下文](output/litter_review/turhancan_night_context.jpg)
- [CatSat 白天上下文](output/litter_review/catsat_day_context.jpg)、[夜间上下文](output/litter_review/catsat_night_context.jpg)
- [LitterCam 夜间上下文](output/litter_review/littercam_night_context.jpg)

局部比较图只显示中心位于局部窗口内的预测框；大框及窗口外背景请结合上下文图核对，
不能从局部图统计误报率。原图局部仅最近邻放大显示，没有生成或修复目标细节。

## 为什么还不能说可靠

在真实画面中，Turhancan 将人脸识别为 Paper（约 0.31）、停放电动车识别为 Plastic
（约 0.51）；CatSat 把雨棚识别为 Paper（约 0.23）、墙面识别为 Plastic（约 0.43）。
人物携带的袋子也可能确实属于 Plastic，但不属于地面遗弃垃圾。材料分类正确不等于业务判断正确。

固定设施误报可能持续出现，多帧投票不能自动消除；本次每 10 秒抽帧也没有检验连续跟踪或报警稳定性。
更细 ROI、人物/设施过滤、跨帧稳定逻辑可以在不训练的情况下开发，但**尚未验证组合后的效果**。
不能把这些待做工作描述为已经拿来即用，也不能认定其它未测权重都无效。

## 2026-09-10 实施设计与时序试验勘误

实施方案以 [GROUND_LITTER_IMPLEMENTATION.md](GROUND_LITTER_IMPLEMENTATION.md) 为准。
本次按五路当前2560×1440画面分别画了区域草案，做了本地真实候选推理；没有接入生产API。
草案及当前候选见 `output/litter_camera_calibration/roi_bounds_review/`。

原时序脚本的22/13/7/14个确认只是稀疏抽样上的历史状态机输出。每10秒抽一帧不能证明
1–2FPS连续视频的稳定性；旧关联半径0.15可跨越横向384像素，能错误合并相邻对象。
此前“5个参考物均出现”的补充判定只是宽松中心距离匹配，还混合了昼夜参考坐标，不能当作
时序目标检出证据，现撤回。49与22来自迭代期间不同的丢失/关联参数，不能作为严格消融实验。
相关JSON保留用于审计，不作为生产参数选型或召回改善证据。

Turhancan继续作为优先实验候选的依据仍是上面的同帧局部定位点检，不是“框数少”或
“时序确认数少”。目前没有证据证明其综合精确率最佳，更不能将20/24称为现场83%准确率。

用户现在要求同一垃圾不重复记录，旧“每5分钟再确认/漏检60秒后重置”不适用。已移除
tracker周期重确认，新增 `ground_litter_inventory.py` SQLite原型：稳定ID、区域初始归属、
存量持久化；仅显式局部可见空地证据能关闭旧垃圾，遮挡、断流和重启不算清理。完整窗口
确认、空地证据生成及生产接入仍待开发，详见实施方案的状态表和验收步骤。

## 权重来源与许可

优先候选：[turhancan97/yolov8-segment-trash-detection](https://huggingface.co/turhancan97/yolov8-segment-trash-detection)，
本地 `models/litter/turhancan_yolov8m_seg_trash.pt`。直接读取的类别为 Glass、Metal、Paper、Plastic、Waste。
该模型用检测框定位物体，同时支持实例分割；本轮评的是框，未用分割掩码二次过滤。
模型卡声明 MIT，列出了 TrashNet/TACO/COCO 标签，但没有完整训练划分和可复现评测说明。

CatSat 的模型卡声明 MIT、TACO 五类材质分割、960 输入、80 epochs；自报 box mAP50=0.291、
box mAP50-95=0.220。这些是作者的验证数据结果，不是本录像准确率。

Bower、Esapzoi 的模型卡声明 MIT；Alope/Oguri 声明 AGPL-3.0；LitterCam 声明 Apache-2.0；
Jhandry 仅标注 OpenRAIL，具体版本/适用条款未核清。Baraa 为 Custom Academic License，
仅非商业使用，商业需书面许可，因此不能据公开下载就用于商用交付。

**以上为发布者声明，不等于完整商用许可审计。** Ultralytics、原生 YOLOv9 推理代码及
底层数据有各自条款；MIT/Apache 模型卡不能替代这些条款，AGPL 也不等于禁止商业使用。
来源、版本与 SHA-256 见 [模型目录说明](models/litter/README.md)。

## 复跑方法与验证范围

`scripts/evaluate_litter_frames.py` 只读本地缓存图、离线推理、输出 JSON/JPEG；不调用项目事件链路。
环境：Python 3.12、Ultralytics 8.4.144、Torch 2.14.0，CPU；不代表生产 GPU 吞吐量。

```bash
# 每 250 个解码帧缓存一张，分别替换输入为昼夜文件；首次运行使用空缓存目录。
mkdir -p /tmp/litter_eval/frames
ffmpeg -hide_banner -loglevel error -i /path/to/day.ps -an \
  -vf 'select=not(mod(n\,250))' -fps_mode vfr -q:v 2 /tmp/litter_eval/frames/day_%03d.jpg

python3 scripts/evaluate_litter_frames.py \
  --model models/litter/turhancan_yolov8m_seg_trash.pt \
  --frames /tmp/litter_eval/frames --output output/litter_screen_turhancan_rerun --mode tiles

python3 scripts/review_litter_candidates.py \
  --references output/litter_review/reference_objects.json \
  --reports output/litter_screen_turhancan/report.json output/litter_screen_catsat/report.json \
  --output output/litter_review/spotcheck_rerun.json
```

原生 LitterCam 的历史复跑参数为 `--yolov9-source /path/to/WongKinYiu/yolov9`；本轮临时
IPython 依赖曾安装于 `/tmp/litter_eval/deps`，现已清理。复跑该淘汰候选须重新下载权重、
代码并安装依赖。适配器直接加载下载的
legacy checkpoint，并选取官方 `detect_dual.py` 相同的第二个检测分支，不依赖临时修改的 detect.py。

新脚本编译和上述真实样本推理、点检已完成。没有新增生产 API/旁路、训练、部署、在线流变更或
服务器性能验收；离线筛查结论不代表项目已接入新的稳定垃圾识别能力。

回归：`python3 -m unittest discover -s tests` 在当前用户 Python 3.12 环境下通过284项，
20.161秒。先前项目 `.venv` 全量尝试出现共享推理线程等待/超时，已中断，不能称该环境
测试通过；本轮没有修改共享推理实现。清理后日志保存在
`output/litter_review/unittest_system.log` 及 `output/litter_review/unittest.log`。
