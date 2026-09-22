# 五路监控零散地面垃圾识别最终技术方案（V1 冻结候选）

日期：2026-09-22  
状态：**方案评审完成；进入离线数据重建与 A/B 验证前冻结**  
范围：五路现有监控 + 后续 ROI 变化、摄像头轻微移动和新增同类监控的复用  
生产边界：本方案第一阶段只做离线数据、训练、回放评估，不修改生产流

---

## 1. 决策摘要

本项目不再把目标定义为“极小垃圾碎片检测”，而定义为：

> 在白天、固定或准固定监控视角下，对用户指定的动态 ROI 内，识别肉眼可明确判断、具有清理意义的地面垃圾，包括铺开的纸巾、揉成团的纸巾、塑料袋、瓶罐、包装袋、纸盒及其他现场明显垃圾。极小碎片允许忽略。

最终推荐架构：

~~~text
2560×1440 原始帧
        │
        ▼
运行时动态 ROI / scene_version
        │
        ▼
原生像素 ROI crop / overlapping tiles
        │
        ├──────────────► Turhancan 语义垃圾通道
        │
        └──────────────► 新 Shared Ground-Litter Detector
                           YOLO26s / RF-DETR-S A/B
                              │
两路候选 ─────────────────────┘
        │
        ▼
候选坐标回原图 + 跨模型去重
        │
        ▼
ROI / overlay 确定性过滤
        │
        ▼
actor / occlusion guard
        │
        ▼
事件级多帧证据聚合
        │
        ▼
confirmed litter event
~~~

数据侧独立形成闭环：

~~~text
7 天 PS + 历史审核资产 + 少量可控摆放样本
        │
        ▼
多源机器 proposal
Turhancan / YOLOE / 新模型 / 随机盲抽
        │
        ▼
事件去重 + 自动框
        │
        ▼
人工审核页（只判断，不画框）
        │
        ▼
Verified Dataset
        │
        ▼
训练 Shared Detector
        │
        ▼
独立 Blind Replay
        │
        ▼
FN + 高价值 FP
        └──────────────► 下一轮审核
~~~

核心原则：

1. **Turhancan 保留**，作为已有垃圾语义召回通道和数据挖掘器；新模型不是它的下游过滤器，而是独立并行召回器。
2. **Profile / Clean Reference / prior 不再承担垃圾召回职责**。历史 NO-GO 已证明其对本项目小垃圾信号上限不足。
3. **ROI 是运行时几何配置，不是模型知识**。模型不得依赖 camera_id、绝对坐标或固定背景位置。
4. **第一版不做摄像头专属模型、LoRA/adapter、Super Resolution 或二阶段分类器**。
5. **先解决 detector 的召回能力，再解决误报**。如果 detector 连候选都不给出，classifier 和时间规则都救不了。
6. **生产指标按事件计算**，而不是按单帧框计算。
7. **人力是最稀缺资源**。机器负责 proposal、框、去重、挑高价值样本；人工只做审核判断。

---

## 2. 已知约束与业务目标

### 2.1 数据与视频

当前已确认：

- 五路监控，参考分辨率 2560×1440；
- 每路可通过接口获取滚动约 7 天录像；
- 回放为 PS 文件，单文件约 5 分钟，但不是严格固定长度；
- 实测五路 4 小时共 241 个 PS、约 20.69 GB，即平均约 4.98 分钟/文件；
- 七天容量约 0.8 TB 量级，因此必须始终采用“下载一个 → 解码/推理 → 保存小型派生产物 → 删除一个”的有界流程；
- 生产 ROI 会变化，摄像头可能人工移动，也可能新增同类监控；
- 暂不要求夜间识别。

历史人工审核资产：

- 2,460 张卡片；
- 189 LITTER；
- 2,083 NON_LITTER；
- 153 UNCERTAIN；
- 35 BOX_WRONG；
- 2,272 张明确正负样本已有冻结清单；
- 最终 ROI 内目前只有 44 张垃圾、370 张非垃圾；
- 历史卡片存在候选选择偏差，且无法保证标签 100% 正确。

因此历史 2,460 张数据定义为 **Silver Data**，不能直接等同于 Ground Truth。

### 2.2 人工能力

人工可以：

- 在本地审核页面快速查看图片；
- 判断垃圾 / 非垃圾 / 不确定等标签；
- 持续增加审核量；
- 对多个机器候选框做 A/B/C 选择。

人工不要求：

- 手动画 bbox；
- 手工描 segmentation mask；
- 逐帧观看全部七天录像。

### 2.3 业务验收目标

首版限定白天。

最终生产目标：

- 肉眼明确、具有清理意义的垃圾事件召回：**目标 ≥ 90%，生产最低门槛建议 ≥ 85%**；
- 确认误报：**每路每天 ≤ 5 次**；
- 理想误报目标：每路每天 ≤ 2～3 次；
- 极小、肉眼需明显放大才能确认且无实际清理意义的碎屑不进入主验收；
- 夜间暂不纳入主验收。

误报必须按实际“已发出的 confirmed false alert”计数。若同一假目标因系统重复重新报警，每次实际报警都算一次误报，不能只按物理对象去重来美化指标。

同时报告：

- FP / camera-day；
- FP / camera-hour；
- event recall numerator / denominator；
- 按摄像头、scene_version、目标原生尺寸分层结果。

---

## 3. 为什么这条路线具有现实可行性

### 3.1 原生像素 ROI / tile 能直接解决主要信号损失

全帧从 2560 宽缩到 640，线性缩放约为 0.25。

例如：

- 原图 12 px 目标 → 约 3 px；
- 原图 20 px 目标 → 约 5 px；
- 原图 30 px 目标 → 约 7.5 px。

这会把原本可见的纸巾、包装物继续压缩到几乎不可识别。

当前五路 ROI 面积仅约 0.85%～22.03%。如果在原始分辨率上先裁 ROI 或切 640×640 native tiles，再按 640 输入 detector，目标像素尺寸基本被保留。

仓库当前 ground_litter_detection.py 已明确实现和描述“native-pixel tiling”：从原始 2560×1440 帧切 640 tile，而不是先缩整帧。这说明该工程基础已经存在，不需要从零实现。

SAHI 的公开 sliced inference 方案也采用相同思想：大图切成有重叠的小块、独立推理、再把结果映射回整图并合并；其文档明确把大图小目标和 surveillance 作为适用问题。

### 3.2 现场已经存在可学习的视觉信号

历史实际审核结果显示：

- 01030 试点中，semantic tile 60 个候选找到 14 个真实垃圾；
- semantic full 30 个只找到 1 个；
- random grid 50 个只找到 1 个；
- 正式批次中 tiled semantic 同样贡献了主要正例。

这至少证明：

1. 真实监控画面中的正常垃圾不是“完全不可见”；
2. Turhancan 已能召回部分真实垃圾；
3. native tiling 对当前场景实际有效；
4. 下一步用现场数据 fine-tune detector 有合理依据。

新方案不是从零赌博，而是在一个已有正向视觉信号上提升召回和泛化。

### 3.3 误报存在现实可学习数据

已有 2,083 个现场 NON_LITTER，覆盖：

- 地砖 / 纹理；
- 店铺固定结构；
- 招牌与画面叠字；
- 桌椅；
- 井盖；
- 车辆 / 电动车边缘；
- 反光；
- 行人附近候选。

这些正是公共垃圾数据通常没有的现场困难负例。它们不能原样全部塞进训练集，但非常适合经过聚类、复核后做 hard-negative learning。

### 3.4 人工审核闭环可持续

当前仓库已有本地审核页、数据指纹、键盘快速分类、前后帧和候选 crop 等基础能力。因此新的人工流程主要是扩展标签和采样逻辑，不需要重新建设完整标注平台。

---

## 4. 现有失败路线的处理边界

### 4.1 Clean Reference / Profile Bank

结论：**不进入新检测关键路径。**

历史实验中 Profile prior 在 oracle 层的目标命中率已经很低，说明问题不是阈值、Selector 或状态机调参。

允许保留的用途：

- 历史研究证据；
- 场景变化诊断；
- 可选的“出现/消失”辅助证据；
- 数据挖掘 proposal。

禁止：

- 将 prior-only 再作为生产垃圾独立召回通道；
- 用更长时间确认、更多 Profile 或更低阈值继续掩盖 detector 能力不足。

### 4.2 冻结 ConvNeXt 分类头

结论：**不作为第一阶段主路线。**

历史事件回放中：

- 白纸可以保住；
- 仍出现大量固定结构、纹理、叠字等确认事件；
- 延迟也不理想。

二阶段 classifier 只有在“detector proposal recall 已足够高，但误报仍是主矛盾”时才允许重新评估，并且届时应端到端微调，而不是继续仅用冻结特征线性头。

### 4.3 简单 3-hit 时间确认

结论：**不能用于判断真假。**

固定设施也能稳定连续命中。时间层只用于：

- 抑制单帧噪声；
- 聚合相同事件；
- 增强出现 / 持续 / 清走证据；
- 避免重复报警。

是否是垃圾，核心仍由视觉 detector 决定。

---

## 5. Runtime V4 架构

建议新路线以 V4 语义建立，不在 hybrid_v33 的 prior 双通道语义上继续叠加。

当前 ground_litter_v33.py 明确写的是：

- semantic = Turhancan；
- prior = Clean Reference；
- 两路 event fusion。

但 prior 已 NO-GO。因此新实现应：

- 保留已有 Turhancan 检测能力；
- 复用可泛化的 event memory / lifecycle 思想；
- 移除 prior 作为 recall source；
- 增加 new_detector source；
- 将 event aggregator 改为 source-agnostic。

建议新的逻辑来源：

- source=turhancan；
- source=shared_detector；
- 后续如果新增模型，可继续注册新的 detector source，而不改变 event state machine。

### 5.1 动态 ROI tile 生成

不把 tile 坐标写死。

算法：

1. 读取当前 scene_version 的 polygon ROI；
2. 计算 polygon bounding rectangle；
3. 在原生像素空间生成最多 640×640 source tiles；
4. 默认 overlap 20%；
5. 过滤 ROI 有效面积过低的 tile；
6. ROI 小于单 tile 时，以 ROI 为中心做 padded source tile，不强行把整幅 2560×1440 resize；
7. detector 输入默认 640；
8. 推理结果映射回原始坐标；
9. 再以 polygon center / overlap 做最终 ROI 过滤；
10. 跨 tile 使用 class-agnostic NMS / merge。

初始只固定 640 + 20% overlap，避免第一轮同时调 tile size、imgsz、overlap 三个变量。

如 blind set 显示 tile 边缘仍有明显漏检，再单独测试 25%～30% overlap。

### 5.2 ROI / 摄像头变化

引入：

- camera_id；
- scene_version；
- roi_polygon；
- overlay_exclude_zones。

摄像头移动或几何结构明显改变时：

1. scene_version + 1；
2. 重新画 ROI；
3. 重新确认 overlay exclusion；
4. 清空旧 scene 的 event / track / temporal memory；
5. shared detector 权重不变；
6. 先 shadow 验证新 scene 的 FP 和 recall；
7. 新场景 hard negatives / missed positives 进入共享数据集。

不得把 camera_id、绝对 x/y 坐标作为 detector 输入特征。

### 5.3 Overlay 与确定性规则

可由几何明确解决的误报不要让模型承担：

- 时间戳；
- 画面叠字；
- 永久 UI 区域；
- 明确不属于业务区域的静态遮罩。

这些使用 overlay_exclude_zones。

ROI / overlay 改变后必须随 scene_version 重校验。

固定真实物体（桌椅、井盖、设施）不能大量依赖“绝对坐标黑名单”解决，应主要作为视觉 hard negatives，以支持摄像头移动和新增监控。

### 5.4 Actor / occlusion guard

actor 只作为上下文证据：

- person / vehicle 高重叠时可暂缓确认；
- actor 离开后垃圾仍存在，增加事件可信度；
- 不应因为目标曾靠近人或车辆就永久判定非垃圾。

这样能支持“人放下垃圾后离开”的业务过程。

---

## 6. 三个模型的明确角色

### 6.1 Turhancan：保留

models/litter/turhancan_yolov8m_seg_trash.pt 保留两种职责。

职责 A：生产召回通道。

它与新 detector 并行：

~~~text
frame / ROI tiles
    ├── Turhancan
    └── New Detector
             ↓
      candidate union
~~~

绝对不能设计成：

~~~text
Turhancan
    ↓
New Detector
~~~

否则 Turhancan 漏掉的纸巾，新模型永远无法补召回。

职责 B：数据挖掘器。

在 7 天 PS 上低阈值 tiled inference，继续高效挖自然垃圾候选。

未来是否退役 Turhancan，只能通过相同独立测试集上的消融决定：

- Turhancan only；
- New detector only；
- Fusion。

如果 Turhancan 不再提供有效独立增量召回，且明显增加 FP / GPU 成本，再退役。

### 6.2 YOLO26s：第一主力训练候选

第一轮训练统一类别：

- 0: ground_litter

不拆 tissue / bottle / bag / can 等细类。

选择原因：

- 项目已有 Ultralytics 生态和 yolo26s.pt；
- 官方支持 detection train / val / export；
- YOLO26 训练策略包含 STAL（Small-Target-Aware Label Assignment），用于维持小目标正样本覆盖；
- 640 输入与当前 native tile 基础设施直接匹配；
- 生产部署路线清晰。

当前官方还提供 YOLO26 P2 small-object architecture YAML，但没有对应 scale-specific P2 预训练权重。第一轮不引入 P2，避免在数据尚少时额外增加训练变量。只有标准 YOLO26s 已表现出明确正向信号但小目标仍成为主要瓶颈时，再把 P2 作为第二阶段实验。

### 6.3 RF-DETR-S：异构 Challenger

RF-DETR-S 只承担 A/B challenger，不一开始接生产。

意义：

- 与 YOLO 架构不同；
- 官方支持 custom dataset fine-tune；
- 支持导出；
- 可以帮助判断失败究竟来自数据/成像，还是某一 detector 架构。

第一轮只比较：

- YOLO26s；
- RF-DETR-S。

暂不同时加入 D-FINE、RT-DETR、Mask R-CNN 等，避免扩大实验空间。

### 6.4 YOLOE：只做数据挖掘

YOLOE 支持 text / visual / prompt-free open-vocabulary detection / segmentation。

第一阶段建议只用于 7 天录像挖掘：

- paper；
- tissue；
- plastic bag；
- bottle；
- can；
- cup；
- wrapper；
- cardboard；
- trash；
- litter；
- garbage。

如果它找到 Turhancan 没找到的真实垃圾，经人工审核后进入 shared detector 训练集。

YOLOE 暂不作为最终生产 detector，避免把开放词汇模型的不稳定类别语义直接带入报警链路。

---

## 7. 新数据体系：Gold / Silver / Ignore

### 7.1 Gold

只有人工重新确认、且用途明确的数据进入 Gold。

包括：

- LITTER_BOX_OK；
- NON_LITTER；
- 经机器重新 proposal 后确认的正确框。

Gold 是 YOLO26s / RF-DETR 的主训练资产。

### 7.2 Silver

历史 2,460 张卡全部先视为 Silver。

用途：

- 正事件聚类；
- 困难负例聚类；
- 主动学习排序；
- 找代表样本；
- 后期低权重辅助。

不得直接声称为 Ground Truth。

### 7.3 Ignore

以下不进入主 detector 监督：

- UNCERTAIN；
- 未修复 BOX_WRONG；
- IGNORE_SMALL；
- positive_unlocalized。

positive_unlocalized 仍然很有价值：

- 用来测候选器漏检；
- 后续新 detector / YOLOE / SAM 类工具可重新提框；
- 可作为 blind recall 的人工证据。

---

## 8. 人工审核设计

人工永远不画 bbox。

### 8.1 Candidate Review 模式

每张卡展示：

- 原始 ROI / context；
- 当前候选框；
- 候选局部放大；
- before / current / after；
- 1:1 native-pixel 局部视图；
- 不显示模型名称、置信度、选择原因。

固定五类：

1. **垃圾-框可用（LITTER_BOX_OK）**
   - 是应该识别的垃圾；
   - 红框大体框住目标即可，不要求像素级贴边。

2. **垃圾-框不准（LITTER_BOX_BAD）**
   - 明确有目标垃圾；
   - 但当前框漏掉大半目标、主要框到背景或把无关大区域包入。

3. **非垃圾（NON_LITTER）**
   - 候选目标不是垃圾。
   - 训练 hard-negative 时只使用围绕候选的紧凑负 patch，避免把整帧其他未标垃圾当成背景。

4. **太小忽略（IGNORE_SMALL）**
   - 确实像垃圾；
   - 但在 native 1:1 视图下无法稳定判断、或属于业务明确允许漏掉的极小碎屑。
   - 不标成 NON_LITTER，避免教错模型。

5. **不确定（UNCERTAIN）**
   - 人也不能稳定判断。
   - 不进入主训练。

### 8.2 BOX_BAD 自动修框

如果人工选择 LITTER_BOX_BAD：

1. 系统从多个 proposal source 生成最多 3 个候选框；
2. 页面显示 A / B / C / 都不对；
3. 用户只选一个；
4. 若都不对，标为 positive_unlocalized；
5. positive_unlocalized 不进入 detector localization 训练。

SAM 类工具只作为后续可选框 refinement，不是 V1 必需依赖。

### 8.3 Blind ROI Audit 模式

必须与 candidate review 分开。

Blind 模式：

- 不显示任何模型框；
- 展示完整 ROI 的无标记画面 / contact sheet；
- 人只判断：
  - 有明显应识别垃圾；
  - 无明显垃圾；
  - 不确定。

一旦标“有垃圾”，再进入机器框 A/B/C 选择；没有合适框时保留 positive_unlocalized。

Blind 数据不允许被训练阶段读取。

### 8.4 为什么必须有 Blind 模式

Candidate Review 只能告诉我们：

> 模型找到的候选里有多少是真的。

它无法告诉我们：

> 模型一共漏掉多少垃圾。

Blind ROI Audit 才能暴露“所有模型共同漏掉”的垃圾，从而形成真实端到端 recall 证据。

---

## 9. 历史数据如何重新利用

### 9.1 189 LITTER + 35 BOX_WRONG

先自动聚成“独立垃圾事件”。

聚类依据：

- camera / scene_version；
- 时间邻近；
- bbox 空间邻近；
- crop / context visual embedding。

同一垃圾持续多帧，只算一个 event。

每个 event 自动挑：

- 首次可见代表帧；
- 最清晰 / 最大目标帧；
- 最后可见代表帧；
- before / after。

人工只重新审核 event card，而不是逐帧审核。

目标首先不是得到多少图片，而是得到：

> verified independent positive events

### 9.2 2,083 NON_LITTER

不得把所有负样本直接喂给模型。

先按：

- camera；
- scene_version；
- 位置；
- 外观 embedding；
- 时间跨度；

做聚类。

优先人工复核：

- 当前 detector 高置信误报；
- 多个日期重复出现的困难模式；
- 不同模型共同误报；
- 新颖外观簇。

同一个井盖重复 100 次，不应等价于 100 个独立 hard negatives。

---

## 10. 7 天 PS 数据工厂

### 10.1 处理原则

严格沿用：

1. 按 1～4 小时窗口查询文件列表；
2. 下载一个 PS；
3. 校验 file_id / size / SHA-256；
4. 解码需要的帧；
5. 运行候选挖掘；
6. 只保存小型 JPEG/WebP、JSON/JSONL、必要特征；
7. 删除 PS；
8. 更新 checkpoint；
9. 处理下一个。

禁止预先下载七天全部录像。

### 10.2 第一轮只处理白天

夜间暂不纳入训练主域和主验收。

记录：

- camera_id；
- scene_version；
- 时间；
- brightness / daylight 标识；
- file_id；
- frame timestamp。

### 10.3 候选源

训练候选池至少包含：

1. Turhancan tiled low-threshold；
2. YOLOE open-vocabulary mining；
3. 新 detector（第一版模型产生后）；
4. 少量短时变化 proposal，仅用于数据挖掘；
5. 与 detector 完全无关的随机 / 分层 ROI 抽样。

短时变化只能是 proposal source，不能重新成为生产垃圾 detector。

### 10.4 人工预算原则

每个审核 batch 优先包含：

- 高置信 potential FP；
- Turhancan 与新 detector 分歧；
- 两模型都认为可能为垃圾的样本；
- decision boundary / low-margin；
- 视觉上与已有数据距离较远的新簇；
- 固定比例 candidate-independent random blind samples。

目标不是“多审核”，而是最大化每次点击带来的模型信息增益。

---

## 11. 少量人工摆放垃圾的用途

即使只能获得约 10～20 个摆放事件，仍建议做，但定位必须正确：

> 高质量现场域锚点，不是最终数据集，也不是测试集。

优先覆盖：

- 铺开纸巾；
- 揉团纸巾；
- 白 / 透明塑料袋；
- 彩色包装袋；
- 矿泉水瓶；
- 易拉罐；
- 饮料杯；
- 纸盒 / 包装物。

刻意覆盖：

- 不同摄像头；
- ROI 不同位置；
- 近 / 中 / 远；
- 深色 / 浅色 / 纹理复杂地面；
- 不同朝向。

每个事件选 2～4 个差异较大的训练帧即可，禁止把连续几十帧当成几十个独立事件。

摆放数据只进入训练 / 诊断，不进入 Blind Test。

---

## 12. 公共数据和合成数据

### 12.1 公共垃圾数据

TACO 等公开垃圾数据用于增加垃圾外观多样性，不用于现场测试。

如果 verified 现场自然正事件低于约 50 个，可采用：

1. 公共垃圾数据把细分类统一映射为 ground_litter；
2. 先做轻量 warm-up；
3. 最终阶段只用现场 Gold 数据 fine-tune / 校准。

如果现场 Gold 已经有约 50+ 个多样独立事件，第一版优先先跑 field-only baseline，保持实验解释性，再决定是否加入公共 warm-up。

### 12.2 Copy-Paste / 合成

只作为补覆盖手段。

允许：

- 已确认垃圾 cutout；
- 公共垃圾 cutout；
- 按现场地面透视、亮度、模糊和压缩合成。

不允许：

- 合成数据进入 Blind Test；
- 合成数据数量压倒现场真实数据；
- 只靠合成结果宣称生产召回。

Diffusion 生成暂不作为 V1 必需步骤。

---

## 13. Detector 训练设计

### 13.1 数据单位

一类目标：

- ground_litter

Positive：

- LITTER_BOX_OK；
- 经 A/B/C 修框后的 LITTER。

Negative：

- 经人工验证的紧凑 NON_LITTER patch；
- 随机 clean ROI tile。

Exclude：

- UNCERTAIN；
- IGNORE_SMALL；
- unresolved BOX_BAD。

### 13.2 防泄漏

切分单位必须是：

- independent event；
- PS file；
- date / time window。

严禁把同一事件的相邻帧随机拆到 train / val / test。

同一人工摆放事件、同一 cutout、同一背景帧只能属于一个 split。

### 13.3 第一轮固定训练变量

YOLO26s 与 RF-DETR-S 使用同一 Gold 数据。

基础配置：

- 一类 ground_litter；
- 训练 crop/tile 尽量与线上 640 native tile 分布一致；
- 首轮不同时试多个 imgsz；
- 轻量 brightness / color / blur / compression / translation / scale；
- 控制强 mosaic / 强 perspective，避免再次破坏小目标；
- early stopping；
- 保存 manifest、权重 hash、代码 commit、训练配置。

训练增强的目的之一是防止模型偷学固定摄像头背景和固定坐标。

### 13.4 两轮上限

第一轮：

- YOLO26s；
- RF-DETR-S。

独立验证后，只允许一次定向修正：

- 加入 FN；
- 加入高价值 FP；
- 补缺失垃圾外观；
- 再训练一次。

第二轮仍没有形成明确正向信号，则暂停模型调参，转向检查成像质量、目标像素和数据质量。

---

## 14. Event Fusion V4

### 14.1 单帧 candidate merge

同一 tick：

- Turhancan；
- shared detector。

候选统一映射回原图后，以 IoU / center distance / size ratio 去重。

每个候选保留：

- source_set；
- per-source confidence；
- bbox；
- timestamp；
- scene_version。

### 14.2 Event state

建议最简状态：

- PENDING；
- CONFIRMED；
- OCCLUDED；
- ABSENT_PENDING；
- CLEARED / EXPIRED。

不要延续 prior-specific 状态作为新架构核心。

### 14.3 时间证据的正确用途

时间证据可以增加：

- 多次 detector 支持；
- 两个 detector 共同支持；
- actor 离开后仍存在；
- 新出现后持续；
- 最终消失 / 清走。

但不要求：

- 必须先观察到“新出现”。

原因：系统启动时垃圾可能已经存在。

同样不允许：

- “连续 3 次命中 = 一定是垃圾”。

固定设施也可以无限连续命中。

### 14.4 告警去重

同一持续垃圾事件只发一次告警。

如果事件真正清走后再次出现，可生成新事件。

生产 FP 指标按实际发出的错误告警计算。

---

## 15. 多路 GPU 调度

五路 3060 Ti 环境下，不按视频帧率运行双 detector。

继续使用项目已有思想：

- low-frequency analysis；
- latest-wins；
- queue capacity 1；
- stale frame 丢弃；
- 多路 tile batch；
- 推流主链路与分析旁路解耦。

初始推荐：

- detector scan interval：约 4～6 秒 / camera；
- 同一批次将当前待分析摄像头的 ROI tiles 合并 batch；
- 先离线测 YOLO26s / RF-DETR-S / Turhancan 的真实 P50/P95；
- 生产 cadence 由“5 路总 GPU 占用 + 事件可接受确认延迟”共同决定。

垃圾是持续性目标，不需要每帧检测。宁可 4～6 秒看一次最新帧，也不要排队处理过时帧。

上线前必须报告：

- model P50 / P95；
- tile 数分布；
- batch size；
- GPU memory；
- analysis tick P50 / P95；
- frame age；
- dropped ticks；
- 主 RTSP publish FPS。

---

## 16. 评估协议

评估拆成两个集合，不再试图用一个 dataset 同时回答所有问题。

### 16.1 Positive Challenge Set

目的：

- 快速比较真实垃圾敏感度；
- 分析哪些垃圾外观被漏掉。

来源：

- 独立日期自然垃圾；
- 不进入训练的 verified events。

报告：

- event recall；
- 每个 camera；
- 原生目标尺寸；
- Turhancan-only / New-only / Both；
- 主要 FN contact sheet。

它不是无偏 prevalence 测试。

### 16.2 Blind Time-Window Set

目的：

- 测端到端漏检；
- 测真实 FP；
- 避免 candidate selection bias。

冻结独立白天时间窗口。

人审不看模型框。

为了节省人工，可按固定时间片生成无框 ROI contact sheet；一旦发现明显垃圾再展开该时间片。

模型训练、调 threshold、挑 winner 时禁止读取 Blind Test 标签。

### 16.3 样本量解释

如果独立自然正事件很少：

- 直接报告 5/5、17/20 等 numerator / denominator；
- 报告 Wilson 95% 区间；
- 不把小样本命中率写成“生产 100% recall”。

建议：

- <20 个自然正事件：只做探索结论；
- 20～49：可以做方向判断，但不做强生产声明；
- ≥50 个、覆盖至少 3 路摄像头和多种垃圾外观后，才开始形成较可信的 overall recall 判断；
- 单摄像头正事件 <5 时，不单独宣称该摄像头 recall，只报告 FP 和观察结果。

### 16.4 Go / No-Go

#### 方向成立

满足：

- 新 detector 或 Fusion 对自然 Required Litter 的 event recall 明显高于 Turhancan；
- 低阈值 proposal recall 足够高，主要问题已从“完全不出候选”转变为“候选里有误报”；
- hard-negative 一次修正能明显降低 FP，而 recall 没有同步崩掉。

#### 生产候选

建议同时达到：

- overall natural-event recall ≥ 85%；
- 目标 ≥ 90%；
- 每路白天 confirmed FP ≤ 5 / camera-day；
- 主要垃圾外观不存在明显接近 0 的类别性盲区；
- 主 RTSP 性能无明显退化。

#### Stop / Reframe

若已经具备：

- verified 真实正例；
- native ROI/tile；
- YOLO26s；
- RF-DETR-S；

但两个 detector 在独立自然明显垃圾上仍有 >50% 事件连低阈值 proposal 都完全不给出，则停止 classifier / 阈值 / 状态机调参，优先检查：

- 实际目标像素；
- 视频码率和压缩；
- 焦距；
- 运动模糊；
- 摄像机视角；
- 是否需要 PTZ / 更高有效分辨率 / 人工复核。

---

## 17. 3～7 天离线判定计划

### Day 1：Gold V1 重建

目标：

- 历史正例做 event clustering；
- 189 LITTER + 35 BOX_WRONG 生成 event cards；
- 历史 NON_LITTER 做位置 + 外观聚类；
- 修改审核标签为 5 类；
- 人工只审核事件代表和高价值 hard negatives。

输出：

- verified_positive_events；
- verified_hard_negatives；
- unresolved / ignore；
- 数据 manifest + fingerprint。

### Day 2：Blind Set + 新自然数据

目标：

- 五路选择独立白天时间窗口；
- 建 Blind ROI Audit；
- 流式处理部分七天 PS；
- Turhancan + YOLOE 挖新自然候选；
- 如可行，补约 10～20 个人工摆放事件，只进训练。

输出：

- frozen blind windows；
- 新 Gold 候选；
- natural positive challenge。

### Day 3：Detector A/B

使用同一 Gold：

- YOLO26s；
- RF-DETR-S。

不做第三个 detector。

输出：

- 两套冻结权重；
- 相同 validation / challenge 结果；
- candidate-level recall / FP 诊断。

### Day 4：三路消融

统一跑：

- Turhancan only；
- YOLO26s or RF-DETR winner only；
- Turhancan + winner Fusion。

重点看：

- independent event recall；
- unique recall contribution；
- FP event source；
- 速度和 GPU。

### Day 5：唯一一次主动修正

自动生成：

- winner FN；
- 高置信 FP；
- Turhancan-only positive；
- New-only positive；
- 双模型 conflict。

用户审核一轮。

只针对主要错误补数据，训练第二版。

### Day 6～7：长回放压力测试

至少在各路独立白天录像上跑长时间 replay。

目标：

- confirmed FP / camera-day；
- event recall；
- GPU；
- frame age；
- event duplicate；
- 是否出现新的高频 FP 簇。

达到方向门槛后才进入生产 shadow，不直接报警。

---

## 18. 第一阶段明确不做什么

暂不做：

- 夜间模型；
- Super Resolution；
- 五路独立 detector；
- camera LoRA / adapter；
- 复杂 VLM 在线复核；
- 二阶段 classifier；
- segmentation 精标；
- Profile Bank 恢复；
- 无限阈值调优；
- 大规模 diffusion 合成；
- 从零训练 detector。

这些不是永远禁止，而是目前证据不足以证明其收益大于复杂度。

---

## 19. 实施模块建议

为了避免污染现有 V3.3，建议新增 V4 离线模块，验证通过后再接 production。

建议模块边界：

- ground_litter_roi_tiles.py  
  动态 ROI native tile 生成、坐标映射、merge。

- ground_litter_detector_v4.py  
  Turhancan / shared detector 统一 detector adapter。

- ground_litter_event_v4.py  
  source-agnostic event memory，不依赖 prior/profile。

- build_ground_litter_verified_dataset.py  
  历史 Silver 聚类、PS mining、Gold manifest。

- build_ground_litter_blind_audit.py  
  candidate-independent Blind ROI Audit。

- train_ground_litter_detector.py  
  同一数据接口支持 YOLO26s / RF-DETR-S。

- evaluate_ground_litter_v4.py  
  event recall、FP/camera-day、source ablation、size/camera/scene_version 分层。

现有：

- PS 流式下载；
- native tiling；
- ROI geometry；
- actor detection；
- latest-wins；
- review UI；
- event evidence；
- DeepStream 主旁路隔离；

尽量复用，不重新造基础设施。

---

## 20. 仍需冻结但不阻塞 Day 1 的业务参数

当前只剩少量业务参数需要后续用真实结果校准：

1. “需要清理”的最短持续时间。  
   V1 不把它写死成垃圾定义；建议 event 层初始只用于减少瞬态误报，后续根据现场业务确认是否把 20～30 秒以内快速消失目标视为无需报警。

2. 白天运行窗口。  
   暂按现场正常可见时段执行，不把夜间混进 V1。

3. 报警延迟容忍。  
   当前工程上建议 4～6 秒扫描一次，预计 confirmed alert 延迟可在数秒到十几秒量级，后续以实际业务需求调整。

这些参数不会改变 detector 和数据路线。

---

## 21. 最终判断

当前证据支持继续投入一次严格受控的 detector 路线验证。

可行性的关键链条是：

1. Turhancan tiled proposal 已在真实现场找到过多批真实垃圾；
2. 全帧缩放会严重损失小目标像素，而 native ROI/tile 能直接保留信号；
3. 现有大量现场 NON_LITTER 提供了难得的 hard-negative 资产；
4. 人工可以持续做低成本审核，足以建立 active-learning 闭环；
5. 预训练 detector 可以在少量高质量现场样本上做 domain fine-tune，不需要从零训练；
6. 模型与 camera / ROI 解耦，因此可以支持 ROI 变化、摄像头移动和新增监控；
7. 项目已经具备 PS 有界处理、ROI、native tiling、审核页、事件状态机和旁路调度基础设施，主要工作是数据重建和 detector 替换/扩展，不是重写系统。

真正尚未被证明的核心问题只有一个：

> 在保留原始局部像素后，一个共享 detector 能否从几十到上百个高质量独立真实垃圾事件中学到足够稳定的现场 ground_litter 概念，并在 Turhancan 补充下达到 ≥85% 的自然事件召回，同时把 confirmed FP 压到 ≤5 / camera-day。

这个问题应该通过上述 3～7 天严格离线实验回答，而不是继续新增算法分支。

---

## 22. 参考资料

项目内部：

- docs/GROUND_LITTER_DATA_ASSET_INVENTORY_20260922.md
- docs/decisions/2026-09-21-ground-litter-profile-prior-no-go.md
- docs/plans/2026-09-21-ground-litter-small-detector-roadmap.md
- docs/plans/2026-09-22-ground-litter-convnext-event-replay.md
- rtsp_annotator/ground_litter_detection.py
- rtsp_annotator/ground_litter_v33.py
- rtsp_annotator/ground_litter_audit_dataset.py
- tools/ground_litter_review_ui/

外部官方资料（检索日期 2026-09-22）：

- Ultralytics YOLO26: https://docs.ultralytics.com/models/yolo26
- Ultralytics YOLOE: https://docs.ultralytics.com/models/yoloe
- SAHI Sliced Inference: https://github.com/obss/sahi/blob/main/docs/guides/sliced-inference.md
- RF-DETR: https://github.com/roboflow/rf-detr/blob/develop/docs/index.md
