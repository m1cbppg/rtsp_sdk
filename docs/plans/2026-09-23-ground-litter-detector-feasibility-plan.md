# 地面垃圾 Detector Feasibility 技术方案（当前执行版）

日期：2026-09-23  
状态：**当前执行方案**  
目标：先证明“白天、动态 ROI、原始高分辨率条件下，正常可见垃圾可以被稳定检测”，暂不做报警、Webhook、复杂事件生命周期和生产 V4 接入。

---

## 1. 当前阶段只回答一个问题

> 对铺开的纸巾、揉团纸巾、塑料袋、瓶罐、包装袋、纸盒等肉眼明显、具有清理意义的地面垃圾，在现有和新增同类监控中，能否通过共享 detector 稳定地给出正确候选框，并把主要误报压到可控范围。

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
- 后续新增同类摄像头要求复用同一个 shared detector。

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
- 唯一一次 FP/FN 修正；
- 其他所有调参。

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

## 13. Sealed Test 必须可重放

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

五路各约 2 小时：

~~~text
10 camera-hours
≈ 10 GB 量级
~~~

这个存储成本完全值得。

---

## 14. 第一轮训练

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

第一阶段先看 detector，而不是报警。

### 15.1 Raw Proposal Recall

低 confidence 下：

> Required Litter 是否至少被正确 proposal 一次。

这是最重要的指标。

如果垃圾连 proposal 都进不来：

- classifier 没用；
- temporal rule 没用；
- event layer 没用。

### 15.2 Event Recall

按 independent episode 计算。

例如：

~~~text
真实垃圾事件：30
模型成功找到：26

event recall = 26 / 30
~~~

同一事件几十帧都命中仍只算 1 个成功。

### 15.3 False Positive

当前阶段先报告：

- FP candidate 数；
- FP episode 数；
- FP 类型分布。

例如：

- 井盖；
- 地砖；
- 招牌；
- 桌椅；
- 反光；
- 车辆边缘。

第一阶段不用急着强行达到每路每天 5 次报警，因为当前还没做报警。

但目标是看到：

> FP 是否集中在可学习的少数 hard-negative 模式，以及第二轮训练后是否显著下降。

### 15.4 分层结果

同时按：

- camera；
- scene_version；
- target short side；
- 垃圾外观类型；
- Turhancan-only；
- New-only；
- Both；

分析。

---

## 16. 当前阶段成功标准

### 16.1 路线成立

认为 detector 路线成立，需要看到：

1. source-native tile 下，新 detector 对 Required Litter 有明显真实召回；
2. New Detector 相比 Turhancan 有独立增量召回；
3. Fusion 明显高于任一单模型；
4. 主要误报集中在可以通过 hard negatives 学习的模式；
5. 第二轮 hard-negative / FN 修正后，FP 明显下降且 recall 没明显崩掉。

### 16.2 数值目标

方向阶段：

~~~text
自然 Required Litter event recall >= 70%
~~~

即可证明路线值得继续。

生产候选前再要求：

~~~text
>= 85%
目标 >= 90%
~~~

由于早期正事件可能不多，必须同时报告：

~~~text
命中数 / 总数
~~~

例如：

~~~text
17 / 20
~~~

不能只有百分比。

### 16.3 Stop Rule

如果已经完成：

- source-native ROI/tile；
- verified 正例；
- YOLO26s；
- RF-DETR-S；
- Turhancan；

但对肉眼明显垃圾：

> 超过 50% 的独立 episode 三个 detector 在低阈值下都完全没有 proposal。

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

### Step 0：冻结数据协议

先实现：

- Required Litter / IGNORE_SMALL 定义；
- 五类人工审核；
- review_card / episode / training_tile 三种 ID；
- Training / Development / Sealed split；
- annotation-complete tile 规则。

### Step 1：历史 Silver 重建

- 189 LITTER + 35 BOX_WRONG -> episode candidates；
- 人工重新审核所有独立正事件代表；
- 2,083 NON_LITTER 聚类；
- 人工审核约 300～500 个高价值 hard negatives。

输出 Gold V1。

### Step 2：冻结 Development / Sealed 录像

- 五路选择白天独立时间窗口；
- 保存可重放 PS；
- 保存 SHA-256 / scene_version / ROI；
- 人工做 Blind ROI Audit。

### Step 3：source-native baseline

同一批数据分别跑：

- Turhancan；
- YOLO26s pretrained；
- YOLOE mining。

首先确认原始高分辨率输入本身的 proposal 能力。

### Step 4：训练 A/B

同一 Gold V1：

- YOLO26s；
- RF-DETR-S。

只看 Development。

### Step 5：消融

Development 上比较：

- Turhancan only；
- YOLO/RF-DETR winner only；
- Fusion。

### Step 6：唯一一次数据修正

把：

- FN；
- 高置信 FP；
- Turhancan-only positive；
- New-only positive；
- conflict；

重新给人工审核。

补训练数据，再训练一次。

然后冻结：

- 模型；
- threshold；
- overlap；
- fusion。

### Step 7：Sealed Test

第一次运行封存测试。

输出：

- raw proposal recall；
- event recall；
- FP；
- 模型互补；
- 分层 error analysis。

Sealed 结果不能用于本轮继续调参。

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
6. ROI / 摄像头变化可以通过 shared detector + runtime ROI 支持；
7. 用户允许忽略极小碎片，问题难度明显降低。

因此当前方案冻结为：

> **数据重建 -> source-native detector A/B -> Turhancan Fusion -> 一次 hard-negative/FN 修正 -> Sealed Test。**

在这条链路证明有效之前，不继续扩展生产事件和告警系统。
