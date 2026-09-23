# 地面垃圾 Detector Feasibility 技术方案（当前执行版）

日期：2026-09-23  
状态：**当前执行方案**  
目标：先证明“白天、当前五路监控、动态 ROI、原始高分辨率条件下，正常可见垃圾可以被稳定检测”，暂不做报警、Webhook、复杂事件生命周期和生产 V4 接入。对新增摄像头、轻微移动后的 scene 泛化只做后续迁移诊断，不作为第一轮 feasibility 的硬成功条件。

---

## 1. 当前阶段只回答一个问题

> 对铺开的纸巾、揉团纸巾、塑料袋、瓶罐、包装袋、纸盒等肉眼明显、具有清理意义的地面垃圾，在当前五路监控中，能否通过共享 detector 稳定地给出正确候选框，并把主要误报压到可控范围。新增摄像头与 scene 变化只作为后续迁移诊断。

当前**不要求**：

- 极小垃圾碎片；
- 夜间；
- 告警；
- 清走判断；
- Webhook；
- 重启告警去重；
- 复杂 actor 生命周期；
- 五路最终生产性能优化。

只有 detector 路线成立，才进入生产 V4。

---

## 2. 业务边界

### 2.1 Required Litter

纳入主检测目标：

- 铺开的纸巾 / 纸张；
- 揉成团的纸巾 / 纸团；
- 塑料袋；
- 瓶子 / 易拉罐 / 饮料杯；
- 包装袋 / 包装纸；
- 纸盒；
- 其他肉眼可明确判断、具有清理意义的地面遗留物。

统一训练为一个类别：

~~~text
ground_litter
~~~

不拆 tissue / bottle / bag / can 等细类。

### 2.2 Ignore

以下不进入主召回验收：

- 很小的纸屑 / 碎片；
- 肉眼正常查看难以稳定判断、需要明显放大才能确认的目标；
- 无实际清理意义的极小物体。

审核统一标记为：

~~~text
IGNORE_SMALL
~~~

不能标成 NON_LITTER，否则会教错模型。

### 2.3 环境范围

第一阶段：

- 只做白天 / 正常可见时段；
- ROI 可变化；
- 摄像头允许轻微移动；
- 架构目标仍是后续新增同类摄像头复用同一个 shared detector，但第一轮只证明当前五路；新增 camera / scene 的迁移能力后续单独验证。

模型不得依赖：

- camera_id；
- 固定绝对坐标；
- 某个固定背景位置。

---

## 3. 当前算法主链路

第一阶段只保留最短链路：

~~~text
PS / 原始视频帧（优先 source 2560×1440）
                │
                ▼
        当前运行时 ROI polygon
                │
                ▼
      source-native ROI crop / tile
                │
        ┌───────┴────────┐
        ▼                ▼
   Turhancan        Shared Detector
                    YOLO26s / RF-DETR-S
        │                │
        └───────┬────────┘
                ▼
      坐标还原 + 跨模型去重
                ▼
        ROI / overlay 硬过滤
                ▼
       detector 输出与评估
~~~

当前阶段**默认不启用 actor hard filter**，避免模型已经检测到垃圾却被旧规则提前丢弃，从而无法判断 detector 本身是否有效。

时间聚合只用于：

- 把连续多帧同一个垃圾合成一个 episode；
- 避免把一个垃圾出现 20 次算 20 个成功样本。

不用于判断“是不是垃圾”。

---

## 4. 为什么必须使用 source-native ROI / tile

当前摄像头参考分辨率为 2560×1440。

若整帧直接缩到 640：

~~~text
20 px 目标 -> 约 5 px
30 px 目标 -> 约 7.5 px
~~~

这会严重损失正常垃圾的视觉信息。

因此第一阶段统一使用：

~~~text
source frame
   ↓
原分辨率 ROI
   ↓
640×640 source tile
   ↓
detector imgsz=640
~~~

默认：

- tile source size：640×640；
- overlap：20%；
- 小 ROI：使用 padded source tile；
- 大 ROI：overlapping tiles；
- detector 结果再映射回 source frame。

**重要工程事实：**

当前生产 DeepStream ground-litter 旁路位于 nvstreammux 之后，默认实际输入是 1920×1080 mux frame，不是原始 2560×1440。

因此：

- 当前 feasibility 实验优先使用 PS 解码出的 source frame；
- 这一步证明的是“视觉 detector 上限”；
- detector 通过后，生产 V4 再解决 mux 前 source-frame 低频旁路；
- 不允许用离线 source-frame 结果直接宣称现有生产入口已达标。

---

## 5. 三类模型的角色

### 5.1 Turhancan：必须保留

模型：

~~~text
models/litter/turhancan_yolov8m_seg_trash.pt
~~~

作用一：已有 baseline。

它已经在真实现场 tiled inference 中找到过多批真实垃圾，因此不是废模型。

作用二：并行召回。

~~~text
source tile
  ├── Turhancan
  └── New Detector
~~~

两路必须并行，不能设计为：

~~~text
Turhancan -> New Detector
~~~

否则 Turhancan 漏掉的纸巾，新模型永远没有机会补召回。

作用三：数据挖掘。

七天录像中继续用低阈值 Turhancan 找自然垃圾候选。

未来是否删除 Turhancan，只看消融：

- Turhancan only；
- New Detector only；
- Fusion。

如果 Turhancan 对 Fusion 几乎没有独立增量召回，同时明显增加 FP / GPU，再退役。

### 5.2 YOLO26s：第一主力候选

统一一类：

~~~text
ground_litter
~~~

作为第一主力原因：

- 项目已有 Ultralytics 生态；
- 仓库已有 yolo26s.pt；
- 训练、导出、TensorRT 路线简单；
- 适合第一轮快速判断现场 fine-tune 是否有效。

### 5.3 RF-DETR-S：异构 Challenger

第一轮只增加一个异构模型：

~~~text
RF-DETR-S
~~~

目的不是堆模型，而是帮助判断：

- YOLO 不行，是 YOLO 架构问题；
- 还是数据 / 摄像机 /像素本身的问题。

第一轮不再同时加入更多 detector。

### 5.4 YOLOE：只做数据挖掘

不进第一版最终生产判断。

用于从七天录像提候选，例如：

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

它找到的新垃圾经过人工审核后进入训练池。

---

## 6. 数据体系

历史 2,460 张卡片全部先视为：

~~~text
Silver Data
~~~

不是 Ground Truth。

现有资产：

- 189 LITTER；
- 2,083 NON_LITTER；
- 153 UNCERTAIN；
- 35 BOX_WRONG。

新的数据层分为：

### 6.1 Gold

人工重新确认后才能进入。

正例：

- LITTER_BOX_OK；
- LITTER_BOX_BAD 后重新自动 proposal 并确认正确框。

负例：

- 经人工确认的 hard negative；
- 经审核确认整块 tile 没有 Required Litter 的 clean tile。

### 6.2 Silver

历史标签未重新确认的数据。

用途：

- 聚类；
- 找代表样本；
- 主动学习；
- 数据挖掘。

不直接作为最终真值。

### 6.3 Ignore

- UNCERTAIN；
- IGNORE_SMALL；
- unresolved BOX_BAD；
- positive_unlocalized。

其中 positive_unlocalized 仍可用于 episode recall 判断，只是不参与 detector localization training。

---

## 7. 人工审核方案

用户不画框。

审核页面固定五个结论：

1. **垃圾-框可用**
2. **垃圾-框不准**
3. **非垃圾**
4. **太小忽略**
5. **不确定**

页面展示：

- before；
- current；
- after；
- 当前候选框；
- context；
- crop；
- 1:1 source-pixel 局部视图。

页面不显示：

- 模型来源；
- confidence；
- 模型认为的标签。

避免影响人工判断。

### 7.1 BOX_BAD

如果是垃圾但机器框明显错误：

~~~text
A / B / C / 都不对
~~~

由机器重新给最多 3 个候选框。

用户只选择，不画框。

如果都不对：

~~~text
positive_unlocalized
~~~

暂不进入 detector training。

### 7.2 Training Tile 完整性确认

候选审核通过不等于 training tile 已完整标注。每个准备进入 detector training 的 640×640 source tile，还需要一次低成本完整性确认：

页面显示**完整 source-scale tile + 当前所有已知框**，人工只回答：

~~~text
A. 已框出全部 Required Litter
B. 还有漏掉的 Required Litter
C. 无法确认
~~~

若选择 B：

- 用户只需单击遗漏目标中心；
- 机器根据 point prompt / 多 proposal source 自动生成候选框；
- 用户选择正确框；
- 若仍无法定位，则该 tile 标为待修复，不进入 detector training。

若选择 C：

- tile 不进入 detector training。

这样不要求人工画框，但能真正满足 §9 的 annotation-complete 训练约束。

---

## 8. 三个身份必须分开

必须明确区分：

### Review Card

给人工看的审核单位。

~~~text
review_card_id
~~~

### Episode

现实中的一个独立垃圾事件。

~~~text
episode_id
~~~

例如一张纸在地上放 20 分钟，是一个 episode，而不是几十张正例。

### Training Tile

真正喂给 detector 的 640×640 source tile。

~~~text
training_tile_id
~~~

三者不能互相替代。

尤其现有按位置去重逻辑只适合减少审核卡，不足以定义独立事件。

如果同一位置：

~~~text
上午垃圾 A -> 清走 -> 下午垃圾 B
~~~

必须是两个 episode。

---

## 9. Detector 训练 tile 必须“标注完整”

这是新数据方案最重要的规则之一。

假设 training tile 中：

~~~text
左边：已标纸巾
右边：还有一个未标塑料袋
~~~

直接训练会把右边塑料袋当背景。

因此每个 detector training tile 必须满足：

- 所有 Required Litter 都有框；
- 不允许可见 UNCERTAIN / positive_unlocalized 静默留在图里；
- IGNORE_SMALL 如果训练框架不能可靠 ignore，则该区域裁掉或整 tile 暂不训练；
- NON_LITTER 不能直接推导成“整张 context 都是负图”。

hard negative 也必须保持 source-scale 分布。

不要：

~~~text
64×64 井盖
↓
放大成 640×640
↓
当 detector 负图
~~~

应该：

~~~text
原始 source frame
↓
以误报点为中心裁 640×640
↓
确认整 tile 没有 Required Litter
↓
作为 negative tile
~~~

---

## 10. 历史数据怎么重建

### 10.1 189 LITTER + 35 BOX_WRONG

第一步不是逐张重审。

先自动按：

- camera；
- 时间；
- bbox 位置；
- visual embedding；
- before / after；

聚成 episode 候选。

每个候选 episode 给人工一个代表卡。

目标得到：

~~~text
verified independent positive events
~~~

而不是追求图片数量。

### 10.2 2,083 NON_LITTER

先聚类：

- 同一井盖；
- 同一地砖；
- 同一固定设施；
- 同类反光；
- 同类车辆边缘；
- 其他重复 FP 模式。

优先重审：

- 高置信 FP；
- 不同 detector 共同误报；
- 跨日期重复出现；
- 新颖视觉簇。

第一轮不需要重新审核全部 2,083 张。

建议约：

~~~text
300～500 个高价值 hard negatives
~~~

其余按主动学习结果再追加。

---

## 11. 新数据怎么来

### 11.1 七天 PS

采用已有 managed recording cache：

~~~text
download
↓
verify SHA-256
↓
lease
↓
decode
↓
mine candidates
↓
artifact commit
↓
release
↓
delete temporary PS
~~~

训练挖掘候选来源：

- Turhancan；
- YOLOE；
- 后续新 detector；
- 少量 temporal proposal；
- detector-independent random ROI sampling。

其中 random sampling 必须保留，用来发现所有 detector 都漏掉的垃圾。

### 11.2 少量人工摆放

即使只有 10～20 个事件仍值得做。

定位：

> 高质量现场域锚点，不是主数据集。

尽量覆盖：

- 铺开纸巾；
- 团纸；
- 塑料袋；
- 瓶；
- 罐；
- 杯；
- 包装袋；
- 纸盒；
- 不同摄像头；
- 不同位置；
- 近 / 中 / 远；
- 不同地面颜色。

每个事件选 2～4 个代表帧。

只进入训练，不进入最终测试。

### 11.3 公共垃圾数据

TACO 等只作为补充。

如果自然 Gold 正事件明显不足，可：

~~~text
public warm-up
↓
field Gold fine-tune
~~~

不能用公共数据验证现场效果。

---

## 12. 数据集严格三分

### Training

用于：

- 模型训练；
- hard-negative learning；
- 数据增强。

### Development Validation

用于：

- YOLO26 vs RF-DETR 选择；
- confidence；
- tile overlap；
- Turhancan fusion；
- 唯一一次 FP/FN 错误分析；
- 其他所有调参。

Development 暴露的问题可以指导补数据，但遵守以下顺序：

1. **优先**根据错误类型，从 Training Pool / 未使用录像中寻找相似新样本加入训练；
2. 如果必须直接把某个 Development FN/FP episode 加入训练，则该 episode、其相邻帧、同源重叠 tile 全部从 Development 计分集合移除；
3. 用于第一版 vs 第二版前后比较的固定 Development Core 不得被加入训练。

禁止“把验证集 FN 加进训练，再继续用同一个 FN 证明第二版提升”。

### Sealed Test

真正封存。

在：

- 模型；
- threshold；
- overlap；
- fusion；
- 数据修正；

全部冻结后才第一次运行。

看完 Sealed Test 后，本轮不再修改模型重新测试。

---

## 13. Development / Sealed 评估资产必须可重放

远端录像只有约 7 天滚动。

因此正式 Development / Sealed 窗口不能只保存文件 ID。

第一版直接封存：

~~~text
原始 PS
+ SHA-256
+ scene_version
+ ROI
+ 实际时间信息
+ 抽样协议
~~~

初始先封存五路**每路总计约 2 小时**：

~~~text
Development：约 1 小时 / camera
Sealed Test：约 1 小时 / camera
总计：10 camera-hours
≈ 10 GB 量级
~~~

这是**初始覆盖预算，不是样本充分性的保证**。如果自然 Required Litter episode 太少，只能报告“证据不足”，不能因为例如 3/4 > 70% 就判定路线成立。

若需要扩大样本：

- 在未查看模型结果前，按预先规则追加新的时间窗口；
- Development 与 Sealed 分别扩展，保持互斥；
- Sealed 不允许因为“某个模型在这段表现好/差”而挑窗口。

由于远端只保留约 7 天，窗口选择和 PS 封存必须**优先于耗时的 Silver 重建**执行，可以并行进行。

---

## 14. 第一轮训练

### 14.0 训练链路自检：先做小样本拟合

在正式 A/B 前，先从 Gold Training 中选一小组已完整确认的 source tiles（包含正例和 hard negatives），做 overfit/sanity check。

目的不是证明泛化，而是验证训练管线真的正确：

- bbox 坐标映射没有错位；
- class id / names 映射正确；
- source tile -> resize -> label 变换一致；
- 正例能被模型明显学到；
- hard negative 不会全部被错误预测为垃圾；
- loss 正常下降；
- 导出的预测框能回到正确位置。

如果连这一小组训练样本都无法明显拟合，先修数据/预处理/训练代码，禁止进入模型优劣和摄像机成像结论。

另外：

- `yolo26s.pt` 若只是通用预训练权重，它对纸巾/垃圾的漏检**不能**作为“现场垃圾不可识别”的证据；
- pretrained baseline 只用于初始化和诊断已有重叠类别；
- 真正判断 detector 路线，必须看现场 Gold fine-tune 后的模型。

### 14.1 正式第一轮训练

统一数据、统一输入。

只训练：

~~~text
YOLO26s
RF-DETR-S
~~~

共同设置：

- one-class ground_litter；
- source-native 640 tiles；
- imgsz 先固定 640；
- overlap 先固定 20%；
- 轻量 brightness / blur / compression / translation / scale；
- 避免强 mosaic / 强 perspective 把小目标再次破坏；
- event/file/date 级切分；
- 禁止相邻帧随机进入 train 和 validation 两边。

第一轮后只允许**一次**数据修正。

---

## 15. 当前阶段怎么评估

第一阶段先看 detector，而不是报警。评估同时区分“能不能找到”和“能不能稳定找到”。

### 15.1 低阈值 Proposal Recall：回答“有没有视觉信号”

在预注册的低 confidence 下：

> Required Litter 是否至少产生过一个正确 proposal。

这是探索阶段最先看的指标，因为如果垃圾连 proposal 都进不来：

- classifier 没用；
- temporal rule 没用；
- event layer 没用。

但“一个 episode 至少命中一次”**不能单独证明稳定识别**。

### 15.2 Episode Recall：回答“独立垃圾事件有没有被发现”

按 independent episode 计算：

~~~text
真实 Required Litter episodes：30
至少命中一次：26
episode recall = 26 / 30
~~~

同一垃圾持续几十帧仍只算一个 episode。

这个指标保留作为核心探索指标，但必须与 §15.3 一起看。

### 15.3 Episode 内稳定命中率：回答“是不是偶尔碰巧看到”

对每个 truth episode，使用固定采样协议从人工确认的 clear-visible interval 中抽取评估帧，例如：

- 每个 episode 最多均匀抽 5～10 帧；
- 长 episode 不因持续更久而获得更高权重；
- 明显遮挡、坏帧、目标不可判断帧不进入 denominator；
- 采样规则在看模型输出前固定。

每个 episode 计算：

~~~text
visible_frame_hit_rate
= 正确命中的采样帧数 / 清晰可见采样帧数
~~~

最终按 episode 做宏平均：

~~~text
macro_visible_frame_hit_rate
= mean(each_episode_hit_rate)
~~~

避免一个持续 20 分钟的大事件因为采样帧多而主导结果。

这不引入时间确认逻辑，只是在同一个真实垃圾上检查 detector 是否稳定。

### 15.4 正确框匹配规则

不能“模型出了任何大框就算命中”。

第一阶段使用预注册的空间匹配规则。优先使用 verified bbox 时：

- 正确类别为 ground_litter；
- detection 与 truth 的 IoU >= 0.3，**或**
- 对极小目标允许 center-in-truth + box size ratio 合理的 small-object matching；
- 禁止覆盖大半 ROI 的超大框通过 center 命中“蹭”成功；
- 同帧多个 truth litter 与 detections 做一对一匹配，一个 detection 不能抵扣多个独立目标。

若 truth 只有人工 point / coarse location，则单独标记为 coarse-match，不与 bbox-level 指标混在一起。

具体阈值只允许在 Development 上冻结，Sealed 不再修改。

### 15.5 工作阈值下的 Recall + FP Rate

低阈值 proposal recall 只看 detector 上限。

同时必须选定一个实际 working threshold，并报告：

- working-threshold episode recall；
- macro visible-frame hit rate；
- FP / 100 张固定采样 ROI 帧；
- FP 类型分布。

FP/100 ROI frames 使用相同固定抽样协议，才能公平比较 Turhancan、YOLO26、RF-DETR 和 Fusion。

不能只说“某模型 FP 总数更少”，因为不同模型可能实际处理帧数不同。

### 15.6 过滤前后都保留

当前 feasibility 先不启用 actor hard filter，但仍要分别保存：

1. raw detector outputs；
2. ROI / overlay deterministic filter 后 outputs；
3. working-threshold outputs。

这样可以定位召回损失来自模型还是后处理。

### 15.7 分层结果

同时按：

- camera；
- scene_version；
- target short side；
- 垃圾外观类型；
- Turhancan-only；
- New-only；
- Both；

分析。

“至少命中一次”“稳定帧命中率”“working-threshold FP”三类证据一起看，才允许使用“稳定检测”这个表述。

---

## 16. 当前阶段成功标准

### 16.1 路线成立：不绑定“双模型融合”预设

任一候选配置都可以成为成功方案：

- Turhancan only；
- YOLO26s fine-tuned；
- RF-DETR-S fine-tuned；
- Turhancan + New Detector Fusion。

**不要求 Fusion 必须优于单模型，也不要求新模型必须提供独立增量。**

如果某个单模型已经达到目标，Fusion 没有收益，就直接使用单模型。

同样，第二轮训练不是强制步骤：

- 第一轮已达到预注册目标，可以直接冻结；
- 第一轮未达标但错误模式可通过补数据合理修正，才做第二轮；
- 第二轮只是最多一次的机会，不是成功条件。

### 16.2 数值目标

探索阶段的最低继续门槛可以保留：

~~~text
natural Required Litter episode recall >= 70%
~~~

但它**不能单独证明“稳定识别”**。

路线进入后续 Production V4 前，至少同时要求：

- episode recall 达到预注册目标；
- macro visible-frame hit rate 达到可接受水平；
- working-threshold FP / 100 ROI frames 可控；
- 不存在某一主要垃圾外观几乎完全识别不到的明显盲区。

生产候选 recall：

~~~text
最低 >= 85%
目标 >= 90%
~~~

早期样本不足时必须报告：

~~~text
命中数 / 总数
+ 样本量
+ Wilson 95% 区间（适用时）
~~~

例如只有 4 个自然 episode，3/4 即使是 75%，也只能标记为**证据不足**，不能判定路线成立。

建议至少：

- <20 个独立自然 Required Litter episodes：只做探索；
- 20～49：可做方向判断；
- >=50 且覆盖至少 3 路 camera、多种垃圾外观：才开始形成较可信的 overall recall 判断。

### 16.3 Stop Rule 前置条件：先证明训练链路没有坏

进入 No-Go 前必须先通过 §14.0 小样本拟合检查。

只有在确认：

- 标签坐标正确；
- 类别映射正确；
- source-scale 预处理正确；
- fine-tuned model 能拟合训练样本；
- Development 数据无明显标注污染；

之后，才允许把多模型共同失败主要归因于成像 / domain 难度。

### 16.4 Stop Rule

如果上述训练链路自检已通过，并完成：

- source-native ROI/tile；
- verified 正例；
- 至少一个正确完成现场 fine-tune 的主 detector；
- 必要时异构 challenger；
- Turhancan baseline；

但对肉眼明显垃圾：

> 超过 50% 的独立自然 episode 在预注册低阈值下仍完全没有任何合理 proposal。

则暂停：

- classifier；
- temporal；
- event state；
- confidence 微调；
- 更多模型堆叠。

优先检查：

- 实际目标像素；
- 摄像机焦距；
- 视频码率 / 压缩；
- 模糊；
- 视角；
- 是否需要更高有效分辨率。

---

## 17. 推荐执行顺序

### Step 0A：封存评估录像 — PASS

已完成：

- Development：65 个 PS / 5.53 GiB；
- Sealed：64 个 PS / 5.64 GiB；
- 合计：129 个 PS / 11.17 GiB；
- SHA256：129 / 129 通过。

后续若自然正 episode 不足，只能按预注册规则追加新窗口并报告“证据不足”，不能降低样本要求硬判成功。

### Step 0B：冻结数据与评估协议 — FROZEN

完整冻结协议单独维护于：

`docs/plans/2026-09-23-ground-litter-step0b-evaluation-protocol.md`

该协议已经冻结：

- REQUIRED_LITTER / IGNORE_SMALL / NON_LITTER / UNCERTAIN；
- episode identity 与跨 split 隔离；
- Blind Truth；
- annotation-complete training tile；
- 固定 episode / global ROI 抽帧；
- prediction-GT matching；
- TP / FP / FN / IGNORED / UNRESOLVED；
- low-threshold proposal recall；
- episode recall；
- macro visible-frame hit rate；
- FP / 100 ROI frames；
- Development threshold / model selection procedure；
- Development -> Training 边界；
- Sealed evaluation_lock；
- EXPLORATORY / DIRECTIONAL / SIGNAL_GO / STABLE_RECOGNITION_PASS / NO_GO 结论等级。

协议层 Step 0B 已完成；审核页、manifest、evaluator 的代码实现由实验执行阶段按该冻结协议落地，不再重新讨论口径。

### Step 1：历史 Silver 重建

- 189 LITTER + 35 BOX_WRONG -> episode candidates；
- 人工重新审核所有独立正事件代表；
- 2,083 NON_LITTER 聚类；
- 人工审核约 300～500 个高价值 hard negatives；
- 生成 training tiles 后再做一次完整性确认。

输出 Gold V1。

### Step 2：Blind Truth 建立

对 Development / Sealed：

1. 先看无模型框 ROI / contact sheet；
2. 人工独立记录 Required Litter 目标、visible interval 和粗位置；
3. 多目标分别建立 truth identity；
4. 再通过 point-click + machine proposal 补足可用定位；
5. 模型输出最后才参与匹配。

`positive_unlocalized` 可以参与人工逐目标 episode recall，但不能仅凭“窗口有垃圾 + 模型出了某个框”判定命中。

### Step 3：source-native baseline + 训练链路自检

先跑：

- Turhancan baseline；
- YOLO26s pretrained 仅作初始化/重叠类别诊断；
- YOLOE mining。

随后立即执行 §14.0 小样本 overfit：

- 验 bbox；
- 验 class mapping；
- 验 source scale；
- 验训练确实学得动。

pretrained YOLO26 未检出纸巾不能作为路线 No-Go 证据。

### Step 4：正式训练 A/B

同一 Gold V1：

- YOLO26s fine-tuned；
- RF-DETR-S fine-tuned。

只看 Development。

### Step 5：Development 选择“最佳配置”

比较：

- Turhancan only；
- YOLO26s；
- RF-DETR-S；
- 必要的 Fusion。

按 §15 的完整指标选最佳配置，不预设 Fusion 一定获胜。

如果第一轮已经达到预注册目标，可以跳过 Step 6。

### Step 6：最多一次数据修正

优先：

> 根据 Development 暴露出的错误类型，从 Training Pool / 未使用录像补**相似但独立**的新样本。

只有找不到独立样本且确有必要时，才允许把 Development 某个 FN/FP episode 转入 Training；一旦转入：

- 该 episode；
- 相邻帧；
- 重叠 tiles；

全部退出 Development 计分。

固定 Development Core 始终保持不变，用于第一轮 vs 第二轮比较。

完成第二轮后冻结：

- 模型；
- threshold；
- overlap；
- single/fusion 配置；
- matching / evaluation protocol。

### Step 7：Sealed Test

第一次运行封存测试。

输出：

- low-threshold proposal recall；
- episode recall；
- macro visible-frame hit rate；
- working-threshold recall；
- FP / 100 ROI frames；
- bbox/coarse match breakdown；
- camera / size / appearance 分层结果。

Sealed 结果不能用于本轮继续调参。

### Step 8：迁移诊断（可选，不阻塞第一轮）

如果有成本很低的条件，可留一个未参与训练的 camera 或新的 scene_version 做迁移诊断。

只回答：

> 当前 shared detector 对轻微视角变化 / 新同类摄像头有没有明显退化？

该结果不纳入第一轮五路 detector feasibility 的硬 Go/No-Go。

---

## 18. Detector 成立以后才做什么

只有 Step 7 证明 detector 路线成立，才进入 Production V4。

后续工程包括：

1. mux 前 source-frame 低频旁路；
2. source / mux PTS 和坐标映射；
3. actor metadata 对齐；
4. detector raw candidate + gate evidence；
5. 简化事件状态；
6. SQLite item/event identity；
7. alert ledger；
8. 重启去重；
9. 五路 GPU benchmark；
10. 必要时共享推理调度器；
11. shadow；
12. 最终才谈每路每天 <=5 次误报报警目标。

这些不属于当前 detector feasibility 的阻塞项。

---

## 19. 当前明确不做

当前不做：

- 夜间；
- Super Resolution；
- Profile / Clean Reference 召回；
- Frozen ConvNeXt classifier；
- 五路独立模型；
- camera-specific adapter；
- LoRA；
- VLM 在线复核；
- 复杂事件状态机；
- 告警 / Webhook；
- 五路跨 camera batch scheduler；
- 大规模合成；
- 无限调参；
- 第三个、第四个 detector。

---

## 20. 最终判断

当前最重要的技术判断不是：

> 怎么报警？

而是：

> **保留 source 像素后，shared detector 能不能稳定看对正常垃圾。**

已有证据支持继续验证：

1. Turhancan tiled inference 已找到真实垃圾；
2. tile 明显优于 full-frame；
3. 现场已有大量真实 hard negatives；
4. 可以持续人工审核；
5. 可以从 7 天录像继续挖自然事件；
6. shared detector + runtime ROI 的架构为未来 ROI / 摄像头变化提供了复用基础，但跨 camera / scene 泛化仍需后续迁移实验验证；
7. 用户允许忽略极小碎片，问题难度明显降低。

因此当前方案冻结为：

> **优先封存评估数据 -> 数据重建 -> source-native detector sanity/A-B -> Development 选择最佳单模型或融合配置 -> 必要时最多一次独立数据修正 -> Sealed Test。**

在这条链路证明有效之前，不继续扩展生产事件和告警系统。
