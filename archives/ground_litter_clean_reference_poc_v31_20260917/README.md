# Ground Litter：Clean Reference + Temporal PoC V3.1 归档

归档时间：2026-09-17（Asia/Shanghai）

源代码基线：`1a29604253907f436e0a63076ea5755cbbf088d3`

## 阶段结论

当前摄像头、红框地面 ROI、正常白天地面可见条件下，Clean Reference 路线已经通过离线 PoC：

- 人工审核的 38 条 V3 Track 包含 16 条垃圾 Track 和 22 条非垃圾 Track。
- 固定地面坐标合并后为 16 个事件：5 个垃圾位置、11 个非垃圾事件。
- 在线状态机共创建 17 个事件，只有 6 个通过 5 秒有效可见确认。
- 人工审核映射结果：TP=5、FP=1、FN=0。
- 事件级 Precision=83.3%、Recall=100%、F1=90.9%。
- 11 个非垃圾事件中 10 个被 context、actor 或时序暂缓，暂缓率 90.9%。
- T01 72 秒首次出现、81 秒确认，只形成一个主事件；遮挡时间不累计为可见证据。
- T01 的 8 个上下文保护样本全部保留。
- clean holdout 共 20 帧，创建事件 0、确认事件 0。
- 唯一确认误报是 reviewed event `negative-v31-event-013` / track 186，留给后续 VLM 或固定设施规则处理。

这些数字只代表本归档素材，不能直接外推为长期线上或跨摄像头准确率。

## 已验证能力

1. Clean Reference Diff 可以独立发现 YOLO 漏掉的 T01 级别小垃圾。
2. 同一固定位置的断裂 Track 可以合并为一个事件，Persistence 使用累计可见时间。
3. 人员、车辆、购物车和大范围结构变化可以进入 `OCCLUDED`，不会当作垃圾，也不会更新 Clean Reference。
4. 短时噪声在 15 秒 pending TTL 后淘汰。
5. 候选累计 5 秒有效可见证据后确认；遮挡和环境不可用时间不累计。
6. Clean Reference 在所有实验中只读，没有被长期垃圾或遮挡覆盖。
7. 3～4px 的极小变化不属于本阶段目标；T01、纸巾、瓶罐、袋装垃圾等可见尺寸是目标范围。

## 尚未验证

- 没有完整的“投放—停留—清走—地面重新可见”真实视频，`CLEARED` 的连续 5 秒关闭条件目前只有单元测试。
- 雨后湿地、积水反光、严重过曝、夜间补光切换尚未形成系统数据集。
- 当前 20/30px 距离和 200px 外部变化面积是当前 1440p 视角的绝对像素参数，跨摄像头前需要透视尺度归一化。
- 生产 `ground_litter`、`GroundLitterDisplayTracker` 和 API 尚未接入这套 PoC。
- VLM 尚未接入；现有 Precision 83.3% 是 VLM 之前的结果。

## 状态语义

- `VISIBLE_ANOMALY`：当前地面可见且异常候选存在。
- `ANOMALY_PENDING`：残差仍在，但当前采样没有形成候选或证据未达到确认门槛。
- `OCCLUDED`：人员、车辆、摊位或大结构使地面暂时不可判断。
- `ENVIRONMENT_CHANGE`：当前环境使有效地面信息不足，暂停证据累计。
- `CLEAN_PENDING`：当前位置重新接近 Clean Reference，等待连续干净观测。
- `CLEARED`：连续 5 秒地面可见且恢复干净，关闭事件。
- `EXPIRED_PENDING`：未确认短时异常超过 TTL 后淘汰。

`GLOBAL_LIGHT_CHANGE` 是明显光度补偿正在工作的诊断状态，本身不会暂停事件；只有 `ENVIRONMENT_CHANGE` 会暂停。

## 目录

```text
code/scripts/       V1、V2、V3、V3.1 和在线评估脚本
code/tests/         本阶段相关测试
code/rtsp_annotator/实验时的生产包源码快照（本阶段未修改）
code/review_ui/     人工审核页面源码与构建产物
evidence/v1/        第一轮 Clean Reference PoC 摘要
evidence/v2/        Adaptive/Temporal V2 摘要
evidence/v3/        ROI、光照保护、人工审核及 V3 摘要
evidence/v31_event/ 固定坐标事件、context、actor 代表帧实验
evidence/v31_online/逐秒在线回放、人工审核映射和最终报告
METADATA.json       环境、输入文件和模型信息
MANIFEST.sha256     归档文件校验值
```

建议优先阅读：

1. `evidence/v31_online/REPORT.md`
2. `evidence/v31_online/human_evaluation.json`
3. `evidence/v31_event/FINAL_REPORT.md`
4. `evidence/v3/human_review/SUMMARY.md`

## 运行环境

- Python 3.12.0
- OpenCV 4.13.0
- NumPy 2.5.2
- Ultralytics 8.4.118
- actor model：`models/yolo26s.pt`
- litter model：`models/litter/turhancan_yolov8m_seg_trash.pt`

模型权重和原始视频未复制进归档。

## 复现命令

以下命令在仓库 `rtsp` 根目录执行；输入视频目录和输出目录可按实际位置调整。

```bash
.venv/bin/python scripts/run_ground_litter_event_memory_v31.py \
  --input-dir /Users/mlcbppg/Desktop/9月17日 \
  --v1-output output/ground_litter_clean_temporal_poc_20260917 \
  --v3-output output/ground_litter_prior_region_v3_20260917 \
  --review output/ground_litter_prior_region_v3_20260917/human_review/ground_litter_negative_v3-reviews.json \
  --output output/ground_litter_event_memory_v31_20260917

.venv/bin/python scripts/probe_ground_litter_event_actor_v31.py \
  --input-dir /Users/mlcbppg/Desktop/9月17日 \
  --v3-output output/ground_litter_prior_region_v3_20260917 \
  --v31-output output/ground_litter_event_memory_v31_20260917 \
  --litter-model models/litter/turhancan_yolov8m_seg_trash.pt \
  --actor-model models/yolo26s.pt \
  --device mps

.venv/bin/python scripts/run_ground_litter_online_replay_v31.py \
  --input-dir /Users/mlcbppg/Desktop/9月17日 \
  --v1-output output/ground_litter_clean_temporal_poc_20260917 \
  --v3-output output/ground_litter_prior_region_v3_20260917 \
  --output output/ground_litter_online_replay_v31_20260917 \
  --litter-model models/litter/turhancan_yolov8m_seg_trash.pt \
  --actor-model models/yolo26s.pt \
  --device mps

.venv/bin/python scripts/evaluate_ground_litter_online_replay_v31.py \
  --online-root output/ground_litter_online_replay_v31_20260917 \
  --reviewed-root output/ground_litter_event_memory_v31_20260917

.venv/bin/python -m pytest -q \
  tests/test_ground_litter_event_memory_v31.py \
  tests/test_ground_litter_online_replay_v31.py
```

归档时相关测试结果为 `12 passed`。

## 下一阶段入口

1. 补一段“投放—停留—清走”的可控素材，真实验证 `CLEARED`。
2. 把绝对像素阈值改为基于地面透视分区和候选尺度的归一化参数。
3. 冻结本摄像头参数，用 2～3 个相似视角只配置 ROI、Clean Profile 和尺度图做迁移验证。
4. 迁移通过后，将 anomaly candidate 接入现有 GroundLitter Track/Display 流程。
5. 最后让 VLM 只复核确认事件，优先验证 track 186，输入 reference crop、current crop、bbox、context 和 persistence。
