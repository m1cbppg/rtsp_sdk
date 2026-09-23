# Ground Litter Step 0B：评估协议冻结（V1.0）

日期：2026-09-23  
状态：**FROZEN — Detector Feasibility 评估协议**  
前置条件：Step 0A PASS  

当前评估资产：
- Development：65 个 PS / 5.53 GiB
- Sealed：64 个 PS / 5.64 GiB
- 合计：129 个 PS / 11.17 GiB
- SHA256：129 / 129 通过

本协议只服务于 Detector Feasibility。当前不评估报警、Webhook、复杂事件状态、清走判定、重启去重或生产五路调度。

---

## 1. 冻结原则

Step 0B 的目的，是在查看正式模型结果之前固定：
- 什么是真垃圾
- 什么不要求识别
- 什么算一个独立 episode
- truth 如何建立
- training tile 什么时候可以训练
- prediction 如何与 truth 匹配
- TP / FP / FN 如何计算
- recall / stability / FP 如何统计
- Development 可以做什么
- Sealed 什么时候才能看

除本文明确允许在 Development 冻结的参数外，模型结果出现后不得为了让结果更好而修改评估口径。

---

## 2. Truth 语义定义

### 2.1 REQUIRED_LITTER

定义：

> 位于业务 ROI 内，在 source-resolution 正常查看条件下可以较稳定确认，属于独立地面遗留物，并具有实际清理意义的目标。

典型包括：
- 铺开的纸巾 / 纸张
- 揉团纸巾 / 纸团
- 塑料袋
- 瓶子
- 易拉罐
- 饮料杯
- 包装袋 / 包装纸
- 纸盒
- 其他明显需要清理的地面遗留物

不要求：
- 是新出现的
- 有人放下
- 持续一定时间
- actor 已离开

这些属于后续事件层，不参与 Detector Feasibility 真值定义。

### 2.2 IGNORE_SMALL

定义：

> 可以确认是真实垃圾，但目标过小、正常业务无需系统处理，或在正常 source-resolution 查看条件下不要求系统稳定识别。

规则：
- IGNORE_SMALL 必须能确认“它确实是垃圾”
- 如果连是不是垃圾都不能判断，应标 UNCERTAIN
- 不使用固定像素阈值定义 IGNORE_SMALL
- short-side pixel size 只记录为诊断字段，不决定业务真值

评估：
- prediction 命中 IGNORE_SMALL：不计 TP
- 不计 FP
- 记为 IGNORED_DETECTION

### 2.3 NON_LITTER

定义：

> 可以明确判断不是 REQUIRED_LITTER / IGNORE_SMALL 的非垃圾目标或背景结构。

例如：
- 井盖
- 地砖纹理
- 普通固定设施
- 阴影
- 反光
- 桌椅
- 车辆边缘
- 与业务无关的固定结构

Blind Truth 不要求把整个 ROI 中所有 NON_LITTER 物体逐个标框。

### 2.4 UNCERTAIN

定义：

> 审核者无法稳定确认目标是不是垃圾，或者无法稳定判断其是否属于 Required Litter。

例如：
- 极小模糊白点，不知道是纸屑还是纹理
- 严重遮挡
- 解码异常导致无法判断
- 材质 / 边界完全不可辨

评估：
- 不计 TP
- 不计 FP
- 不计 FN
- 单独记录 UNRESOLVED

---

## 3. Episode 定义

### 3.1 基本定义

> Episode = 同一 camera、同一 scene_version 下，同一个物理垃圾目标持续存在的一段真实世界过程。

Episode identity 跨越：
- 视频帧
- PS 文件
- review cards
- training tiles
- detector outputs

不能因为换了 PS、换了审核批次或模型重新出框就创建新 episode。

### 3.2 Episode 结束条件

只有以下情况可以结束一个 episode：
1. 明确被清走
2. 明确离开 ROI
3. scene_version 变化
4. 出现长时间观测缺口，无法继续确认物理身份连续性

短暂遮挡、短暂漏帧、模型没有检测到均不能结束 truth episode。

### 3.3 同位置再次出现

“垃圾 A -> 明确清走 -> 同位置垃圾 B”必须是两个 episode。

位置相同不等于物理身份相同。

### 3.4 移动中的同一垃圾

如果连续画面能确认是同一个物体，例如塑料袋被风吹动：
- bbox 可以变化
- 仍属于同一个 episode

### 3.5 多目标

同一画面出现多个 Required Litter 时：
- 每个真实物体分别建立 truth_target_id
- 分别建立 episode_id

一个“窗口有垃圾”不能代替逐目标 truth。

---

## 4. 跨 Split Episode 隔离

硬规则：

> 同一个物理垃圾 episode 的全部帧、PS 文件中的相关时间段，以及由它产生的 review card、crop、training tile 和其他衍生产物，只能属于一个 split。

严禁：
“同一张纸：前一小时 -> Development，后一小时 -> Sealed”。

即使来自不同 PS 文件也属于泄漏。

### 4.1 无法确定是否同一 episode

如果无法确认两个时间段中的垃圾是否为同一个物理物体：

> 保守按同一个 episode_group 处理，不允许跨 split。

宁可减少数据，也不冒泄漏风险。

### 4.2 Step 0A 已先封存的处理

Step 0A 的 Development / Sealed 是时间窗口初始分配。

Blind Truth 完成后必须做一次 cross-split episode audit。

如果发现同一 episode 跨越 Development 和 Sealed：
- 整个 episode 归 Sealed
- Development 中与该 episode 重叠的 truth target 和计分帧退出 Development scoring
- 对应 manifest 记录 reassignment 原因
- 不需要删除原始 PS，只调整评估 manifest

原则：

> Sealed 优先，避免把 Sealed 的目标提前暴露到 Development。

---

## 5. Blind Truth 标注协议

### 5.1 Blind 原则

Truth 建立阶段禁止显示：
- Turhancan prediction
- YOLO prediction
- RF-DETR prediction
- YOLOE prediction
- confidence
- proposal source
- 模型预测标签

模型 prediction 只能在 truth freeze 后参与 matching。

### 5.2 审核单位

审核单位为：

> 一个预先确定的时间片 + 无模型框 ROI 帧 / contact sheet。

允许使用多帧帮助人工判断目标连续性，但不得使用模型结果引导寻找目标。

### 5.3 多目标必须分别记录

发现多个垃圾时，每个目标单独建立 truth_target_id 和 episode_id。

不能只记录 window_has_litter=true。

### 5.4 Truth target 最小字段

每个 truth target 至少记录：
- truth_target_id
- episode_id
- camera_id
- scene_version
- truth_class：REQUIRED_LITTER / IGNORE_SMALL / UNCERTAIN
- visible_intervals
- location_type：BBOX / POINT / COARSE_REGION / UNLOCALIZED
- location
- appearance_tag
- target_short_side_px（有可靠 bbox 时）
- review_status
- review_version

appearance_tag 只用于诊断，不作为 detector 类别。

### 5.5 定位方式

人工不画 bbox。

优先顺序：
1. 已有独立可信框且人工确认正确 -> BBOX
2. 人工单击目标中心 -> POINT
3. point / proposal helper 生成框，人工确认 -> BBOX
4. 只能确认大致区域 -> COARSE_REGION
5. 能确认 Required Litter 存在但无法可靠定位 -> UNLOCALIZED

UNLOCALIZED truth 可以参与 episode 存在性统计，但不能进入自动 bbox-level matching。

### 5.6 Visible Interval

visible interval 只能由人工真值定义。

以下时间不进入 visible-frame denominator：
- 严重遮挡
- 坏帧
- 强模糊
- 目标不可判断
- ROI 画面失效

短暂不可见不等于 episode 结束。

---

## 6. Annotation-Complete Training Tile

### 6.1 正式定义

> Annotation-complete training tile = 保持部署 source-scale 分布，且其中所有清晰可见的 REQUIRED_LITTER 均具有训练可用定位标注，不存在会被错误当成背景的未解决真实垃圾。

只有 annotation_complete=true 的 tile 才允许进入 detection training manifest。

### 6.2 REQUIRED_LITTER

tile 中所有 REQUIRED_LITTER 必须全部有训练可用 bbox。

只标出一个候选目标，但 tile 其他位置还有漏标垃圾：
- annotation_complete=false
- 禁止训练

### 6.3 IGNORE_SMALL

V1 不依赖训练框架未验证的 ignore-region 行为。

如果 IGNORE_SMALL 明显存在且可能造成错误背景监督，优先：
1. 重新选 source-native tile
2. 排除该区域且不改变主目标尺度
3. 无法安全处理则整 tile 暂不训练

### 6.4 UNCERTAIN / positive_unlocalized

只要仍然位于 tile 中：
- annotation_complete=false
- 不进入 detector training

除非后续重新确认 / 定位成功。

### 6.5 Negative Tile

一个 candidate 被审核为 NON_LITTER：

> 不代表整张 tile 是 clean negative。

Negative training tile 必须人工确认：
> 整个 tile 中不存在任何 REQUIRED_LITTER。

### 6.6 Source-Scale

禁止把小 negative crop 大幅放大后冒充部署 tile。

例如禁止：
“64×64 井盖 -> resize 到 640×640 -> detector negative”。

应从 source frame 中生成真实 640×640 source tile。

### 6.7 完整性审核

每个候选 training tile 显示：
- 完整 source-scale tile
- 当前所有已知 Required Litter 框

人工只判断：
- A. 已框出全部 Required Litter
- B. 还有遗漏 Required Litter
- C. 无法确认

B：
- 用户点击遗漏目标中心
- 机器补 proposal
- 用户选择
- 重新确认完整性

C：
- tile 不训练

---

## 7. 固定评估抽帧协议

必须使用固定采样，避免长 episode 因帧多而主导指标。

### 7.1 Episode Stability Sampling

对每个 REQUIRED_LITTER episode：
1. 只在人工确认的 visible_intervals 内取帧
2. 先按 5 秒时间网格生成 eligible timestamps
3. 如果 eligible frames <= 5：全部使用
4. 如果 > 5：按整个 visible duration 均匀选取 5 帧
5. 多个 visible intervals 时按累计 visible duration 均匀覆盖
6. 坏帧 / 不可判断帧跳过，并选择最近的下一个 eligible frame
7. 若最终没有任何 eligible frame，该 episode 标为 evaluation_unresolved，不进入稳定性 denominator

因此每个 episode 对稳定性指标最多贡献 5 帧。

### 7.2 Global ROI Sampling：用于 FP

对完整 Development / Sealed 窗口：
- 每 30 秒固定抽取一张 source-resolution ROI evaluation frame
- 时间点以冻结窗口起始时间为基准
- 与 detector 结果无关
- 不因为“这一帧有垃圾 / 没垃圾”改变采样
- decode failure 明确记录并从 denominator 排除

这套 frame manifest 同时用于所有模型 / 配置的 FP 比较。

---

## 8. Prediction 输出层级

Feasibility 阶段保存三个层级：
1. RAW：模型原始候选经过模型自身必要 NMS
2. GEOMETRY_FILTERED：ROI / overlay deterministic filter 后
3. WORKING_THRESHOLD：选定工作阈值后

当前不启用 actor hard filter。

评估必须能分别报告 RAW / FILTERED / WORKING，避免后处理损失被错误归因于 detector。

---

## 9. Prediction-GT Matching

所有模型使用同一 matching protocol。

先对 REQUIRED_LITTER 做一对一匹配，再处理 IGNORE_SMALL / UNCERTAIN，最后剩余 prediction 才计 FP。

### 9.1 BBOX Truth：普通目标

若 truth 有 verified bbox：

> IoU(pred, gt) >= 0.30

即可成为 eligible match。

### 9.2 BBOX Truth：小目标补充规则

若 gt_short_side <= 20 source pixels，允许以下任一条件成为 eligible match：

A. IoU >= 0.20

或 B.
- prediction center 位于 gt bbox 向四周扩张 50% 后的区域内
- 且 0.25 <= pred_area / gt_area <= 4.0

目的：
- 避免小目标 2～3px 坐标偏移导致 IoU 极不稳定
- 同时防止覆盖大面积 ROI 的巨框“蹭命中”

20px 仅用于 matching 数值稳定性，不改变 REQUIRED_LITTER / IGNORE_SMALL 的业务定义。

### 9.3 一对一匹配

同一帧：
- 一个 prediction 最多匹配一个 REQUIRED_LITTER truth
- 一个 truth 最多匹配一个 prediction

在 eligible edges 中，优先最大 IoU；IoU 相同时优先更高 prediction score。

如果多个 prediction 都围绕同一个 GT：
- 最佳一个计 TP
- 其他未匹配 prediction 继续进入 ignore / unresolved / FP 判定
- 不能全部算 TP

### 9.4 POINT / COARSE_REGION Truth

POINT / COARSE_REGION 不进入自动 bbox precision 统计。

对于可能匹配的 prediction：
- 生成待裁决对
- 审核界面隐藏模型身份和 score
- 人工判断“该 prediction 是否确实对应这个 truth target”

裁决结果：
- COARSE_TP
- COARSE_MISS
- UNRESOLVED

COARSE_TP 可以用于 episode recall / visible-frame hit rate，但必须与 bbox-TP 数量分别报告。

### 9.5 UNLOCALIZED Truth

UNLOCALIZED truth：
- 不进行自动 prediction matching
- 如果该 episode 对最终结论重要，必须补 point / coarse location 或人工逐目标裁决
- 在补定位前，只计 truth existence，不用于自动 TP / FN 统计

不能因为“窗口中模型出了任意垃圾框”就认定 UNLOCALIZED truth 被命中。

---

## 10. TP / FP / FN / IGNORE / UNRESOLVED

对每个固定评估帧，按以下顺序判定。

### 10.1 TP

prediction 一对一匹配 REQUIRED_LITTER：
- 计 TP
- BBOX 匹配和 COARSE 匹配分别记来源

### 10.2 FN

一个可评估 REQUIRED_LITTER truth 在该帧没有匹配 prediction：
- 计 FN

UNLOCALIZED 且尚不可匹配的 truth 不强行记 FN，记 evaluation_unresolved。

### 10.3 IGNORED_DETECTION

未匹配 prediction 对应已确认 IGNORE_SMALL：
- 不计 TP
- 不计 FP
- 记 IGNORED_DETECTION

### 10.4 UNRESOLVED_PREDICTION

未匹配 prediction 对应 UNCERTAIN 区域 / 目标：
- 不计 TP
- 不计 FP
- 单列数量和比例

### 10.5 FP

完成 Required / Ignore / Uncertain 处理后仍未匹配的 prediction：
- 计 FP

包括：
- 井盖
- 地面纹理
- 固定设施
- 阴影
- 反光
- 车辆边缘
- 重复框

---

## 11. Low-Threshold Proposal Recall

目标：

> 判断模型是否具备“看见垃圾”的视觉能力，而不是评价最终工作点。

固定 proposal mode：
- model confidence floor = 0.01（模型 / API 支持时）
- tile merge / NMS IoU = 0.50
- max candidates = top 100 / ROI frame

若某模型无法导出 0.01 score candidates：
- 使用该实现允许的最低 score
- 必须记录实际 floor
- 不直接比较 score 数值本身

指标：

proposal_episode_recall =
至少在一个固定 episode evaluation frame 中有正确 match 的 REQUIRED_LITTER episodes
/
可评估 REQUIRED_LITTER episodes

同时报告：
- proposal_recall@100
- proposal FP / 100 ROI frames

低阈值指标只用于判断视觉信号，不作为最终工作阈值。

---

## 12. Episode Recall

所有 episode 指标只使用第 7.1 节固定 episode evaluation frames。

### 12.1 Episode Hit

一个 REQUIRED_LITTER episode，在至少一个固定 evaluation frame 上有 TP / COARSE_TP：
- episode_hit=true

否则：
- episode_hit=false

### 12.2 Episode Recall

episode_recall =
hit REQUIRED_LITTER episodes
/
evaluable REQUIRED_LITTER episodes

必须同时报告：
- hit_count
- episode_count
- 百分比

不能只报百分比。

---

## 13. Visible-Frame Hit Rate

每个 episode：

visible_frame_hit_rate =
有 TP / COARSE_TP 的固定 evaluation frames
/
该 episode 可评估固定 frames

总体使用宏平均：

macro_visible_frame_hit_rate =
mean(visible_frame_hit_rate per episode)

每个 episode 权重相同。

这是判断：

> 模型是真的稳定看到垃圾，还是只偶尔碰巧命中一次。

---

## 14. FP / 100 ROI Frames

使用第 7.2 节的固定 30 秒 Global ROI Sampling。

FP_per_100_ROI_frames =
total FP predictions
/
evaluable global ROI frames
* 100

同时必须报告：

FP_positive_frame_rate =
含 >=1 FP 的 ROI frames
/
evaluable ROI frames
* 100%

这样既能看到“总共出了多少错误框”，也能看到“多少帧会出现误识别”。

IGNORE_SMALL 和 UNCERTAIN 对应 prediction 不计 FP。

---

## 15. Development Working Threshold 选择规则

Step 0B 不凭空指定生产阈值，但现在冻结选择方法。

正式 threshold search 只允许在 Development。

### 15.1 Threshold Grid

每个 detector 使用固定 grid：
- 0.05
- 0.10
- 0.15
- 0.20
- 0.25
- 0.30
- 0.40
- 0.50

如果模型 score 输出定义不同导致该 grid 明显无效，可以在正式 Development 比较前追加模型专属 grid，但：
- 必须先登记
- 对该模型整轮固定
- 不能看 Sealed 后再改

### 15.2 单配置 working threshold

对一个模型 / 配置：
1. 找到 Development 上最高 episode hit count：Hmax
2. 候选阈值必须满足 hit_count >= Hmax - 1
3. 在候选中选择 macro_visible_frame_hit_rate 最高者
4. 如果 macro hit rate 相差 <= 5 个百分点，选 FP/100 ROI frames 更低者
5. 再平手，选择更高 threshold

目的：
- 不为减少少量 FP 大幅牺牲召回
- 又避免永远使用最低阈值

---

## 16. Development 模型 / Fusion 选择规则

每个配置先按第 15 节得到自己的 working threshold。

比较：
- Turhancan only
- YOLO26s fine-tuned
- RF-DETR-S fine-tuned
- 必要的 Fusion

排序规则：
1. episode hit count 更高
2. 若相差 <= 1 个 episode，比较 macro_visible_frame_hit_rate
3. 若 macro hit rate 相差 <= 5 个百分点，比较 FP/100 ROI frames
4. 若仍近似持平，优先单模型、较低推理成本、较简单部署

不预设 Fusion 必须获胜。

Fusion 使用各 constituent 在 Development 已冻结的 working threshold，union 后做 class-agnostic merge / NMS。Fusion 自身规则必须在 Sealed 前写入 evaluation lock。

---

## 17. Development 数据使用边界

Development 可以用于：
- threshold
- tile overlap
- matching 参数验证
- YOLO26 / RF-DETR / Turhancan / Fusion 选择
- 错误类型分析
- 是否需要第二轮训练

### 17.1 Development FN / FP 不直接回灌

优先：

> 根据 Development 暴露的错误类型，从 Training Pool / 未使用录像找相似但独立样本。

### 17.2 如果必须直接转入 Training

某个 Development episode 一旦转入 Training：
- 该 episode
- 相邻帧
- 同源重叠 tiles
- 相关衍生产物

全部退出 Development scoring。

### 17.3 Development Core

在第一轮正式模型比较前冻结 Development Core。

Development Core：
- 永不进入 Training
- 第一版 / 第二版都必须在相同 Core 上比较
- 用于证明第二轮改善不是因为把验证样本背下来

---

## 18. Sealed Test 使用边界

Sealed 可以先完成人工 Blind Truth，但在 evaluation lock 完成前：
- 禁止模型 inference
- 禁止查看模型在 Sealed 上的任何指标
- 禁止使用 Sealed 样本训练
- 禁止根据 Sealed 修改阈值 / overlap / matching / fusion

第一次 Sealed inference 前必须生成并冻结 evaluation_lock，至少包含：
- protocol_version
- code_commit
- model_weight_hashes
- training_manifest_hash
- development_manifest_hash
- sealed_manifest_hash
- tile_size
- tile_overlap
- imgsz
- per_model_low_threshold_floor
- per_model_working_threshold
- nms / merge config
- matching config version
- episode_sampling_manifest_hash
- global_roi_sampling_manifest_hash
- selected configuration
- fusion config（如有）
- stable_pass_thresholds（如已定义）

Sealed 跑完后：

> 本轮 detector 版本不得再根据 Sealed 结果修改后重新声称这是同一个封存测试。

后续修改属于新实验版本，需要新的独立 Sealed 数据，或明确标记为 post-test iteration。

---

## 19. 样本量与结论等级

自然 REQUIRED_LITTER episode 数：

### < 20

结论最多：
- EXPLORATORY

不能声称稳定识别成立。

### 20～49

可以：
- DIRECTIONAL

用于判断路线是否有明显正向信号。

### >= 50

并且：
- 覆盖 >=3 路 camera
- 覆盖多种垃圾外观
- 无明显单一 episode cluster 主导

才允许形成较可信 overall detector feasibility 判断。

所有 recall 必须报告：
- numerator
- denominator
- Wilson 95% interval（适用时）

---

## 20. Go / Continue / No-Go 结论规则

### 20.1 PIPELINE_PASS

小样本 overfit / sanity check 通过：
- bbox 坐标正确
- class mapping 正确
- source-scale 处理正确
- fine-tuned model 能明显拟合 Gold 训练样本

未通过时：

> 只能修训练链路，不能讨论模型 / 摄像头路线失败。

### 20.2 SIGNAL_GO

在样本量允许做方向判断时：

> best Development natural episode recall >= 70%

并且没有证据表明结果来自明显数据泄漏。

这表示：

> detector 路线有真实视觉信号，值得继续。

不等价于“稳定识别已经通过”。

### 20.3 STABLE_RECOGNITION_PASS

Step 0B 当前不凭空设定 macro-visible-hit 和 FP/100 的业务门槛。

若要在 Sealed 后使用“稳定识别通过”这一表述，必须在第一次 Sealed inference 前，在 Development 上预先冻结：
- stable_episode_recall_min
- stable_macro_visible_hit_min
- stable_fp_per_100_max

其中：
- stable_episode_recall_min 不得低于 85%
- 目标建议 90%

如果在 Development 阶段没有足够依据冻结合理的 stability / FP 门槛：

> Sealed 最多支持 SIGNAL_GO / EVIDENCE_INSUFFICIENT / NO_GO，不得声称 STABLE_RECOGNITION_PASS。

### 20.4 EVIDENCE_INSUFFICIENT

典型情况：
- 自然正 episode 太少
- 可评估 visible frames 太少
- 某关键 camera / appearance 完全没有覆盖

例如 3 / 4 = 75%，只能报告证据不足，不能判路线通过。

### 20.5 NO_GO / REFRAME

前提：
- PIPELINE_PASS 已通过
- truth / split 无明显污染
- source-native 输入正确
- 至少一个现场 fine-tuned 主 detector 已正确训练
- 必要时已验证异构 challenger

若：

> 超过 50% 的独立自然 REQUIRED_LITTER episodes 在 low-threshold proposal mode 下都完全没有合理 proposal，

则停止：
- classifier
- temporal
- event state
- 更多 confidence 微调
- 第三个 / 第四个 detector 堆叠

转而检查：
- 目标有效像素
- 焦距
- 码率 / 压缩
- 模糊
- 摄像机视角
- 是否需要更高有效分辨率

---

## 21. Step 0B 完成定义

Step 0B 在以下内容冻结后视为 PASS：
- Truth taxonomy
- Episode identity
- cross-split isolation
- Blind Truth
- annotation-complete
- fixed evaluation sampling
- prediction-GT matching
- TP / FP / FN / Ignore / Unresolved
- proposal recall
- episode recall
- visible-frame hit rate
- FP / 100 ROI frames
- Development threshold / model selection procedure
- Development -> Training 边界
- Sealed evaluation lock
- conclusion levels

允许后续在 Development 阶段填写、但必须在 Sealed 前冻结的只有：
- per-model working threshold
- 如确有需要的 model-specific threshold grid
- Fusion 最终配置
- stable_macro_visible_hit_min
- stable_fp_per_100_max
- 最终 selected configuration

其余规则进入实验后不得根据结果随意修改。

---

## 22. 当前下一步

Step 0A 已 PASS。

Step 0B 协议冻结后，实验执行进入：

Step 1：
- 历史 Silver -> Gold V1
- 小批 annotation-complete training tiles

随后：
- Step 3 sanity check
- 正式 Detector A/B

当前不增加报警、事件持久化或生产调度工作。
