# Ground Litter V3.3 Hybrid 架构与实施规格

> **SUPERSEDED（2026-09-18）**：本文件把垃圾模型设计成 prior 的硬语义闸门，
> 与最新产品目标不一致。DS4.1 不得据此实施。唯一有效规格为
> `docs/plans/2026-09-18-ground-litter-v33-dual-recall-architecture.md`。

状态：待 DS4.1 实施  
日期：2026-09-18  
基线：当前工作区中的 V3.2 production hardening。**该硬化版已于 2026-09-18 15:13 CST
切换进生产**，不是"尚未切换"。  
交接背景：`HANDOFF_GROUND_LITTER_V32_HARDENING_20260918.md`（其第 1 节与 8.A–8.C 已过期，
顶部有 SUPERSEDED 横幅），实际生产状态以
`GROUND_LITTER_V32_HARDENING_RESULT_20260918.md` 为准。

> 基线校正（2026-09-18 17:xx，实测）：生产 API 运行
> `rtsp-yolo-annotator:deepstream8-ground-litter-v32-hardening-20260918`
> （`sha256:7aa71d92df2e0061df9898a658a3f51430736418dda224adff2423626d5f49ab`），
> 容器 15:13:36 CST 启动、`RestartCount=0`。回滚标签
> `…-before-ground-litter-v32-hardening-20260918` → `sha256:841ca526317f…`（上一版 V3.2）。

## 1. 架构结论

V3.3 必须同时使用两类证据：

1. `turhancan_yolov8m_seg_trash.pt` 负责回答“这个目标是否具有垃圾语义”。
2. Clean Reference 负责回答“干净地面上是否出现了新的、持续的目标”，并负责遮挡、清走、同位置再次出现等生命周期判断。

两者不能简单取并集。简单 OR 会把垃圾模型误报和场景差分误报叠加，
导致总体误报率上升。V3.3 使用分级证据融合：

```text
                    ┌─ 全 ROI 低频垃圾模型扫描 ───────────────┐
实时帧 ─────────────┤                                        ├─→ 语义证据
                    └─ Clean Reference 小目标候选 ─→ 放大裁剪 ─┘

Clean Reference 候选 ─────────────────────────────────────────→ 变化证据

语义证据 + 变化证据 + actor/context 遮挡证据
                    ↓
              V3.3 统一事件状态机
                    ↓
        确认 / 遮挡 / 清走 / 同位置新事件 / OSD
```

默认产品规则：**没有垃圾模型语义证据的先验候选，不得显示为“疑似垃圾”。**
它可以进入内部诊断或人工审核队列，名称必须是“地面异物候选”，不能与
语义确认垃圾使用相同标签。

## 2. 设计动机与能力边界

### 2.1 当前 V3.2 的问题

纯 Clean Reference 能发现垃圾模型漏掉的小变化，但它本质是异常检测，
不能区分垃圾、阴影、商户设施、车辆变化和地面反光。现场已经证明：清晨
参考图用于下午场景时会累计大量错误事件。

### 2.2 垃圾模型的作用

垃圾模型提供语义约束和材质类别。它有两种使用方式：

- 低频扫描完整 ROI，直接发现尺寸较大或特征明显的垃圾。
- 对 Clean Reference 提议的小区域进行扩边、放大和批量推理，使原本在
  全图中像素过少的目标获得更高有效分辨率。

### 2.3 无法绕过的事实

如果同一个垃圾模型在放大裁剪后仍然无法识别某类小目标，那么系统不能
诚实地把这个目标称为“模型确认垃圾”。此时只有三种合理选择：

1. 仅作为“地面异物候选”进入人工审核；
2. 训练小目标二分类器或重新训练/微调垃圾模型；
3. 接受先验独立报警，但产品标签和置信等级必须与垃圾识别分开。

因此，实施前必须完成第 5 节的模型可行性闸门，不能直接假设裁剪放大一定
能解决模型漏检。

## 3. 模式和兼容性

新增模式：

```json
"mode": "hybrid_v33"
```

兼容原则：

- `mode=yolo` 保持原行为，不修改现有接口语义。
- `mode=clean_reference_v32` 保持硬化后的纯先验实验行为。
- `mode=hybrid_v33` 才启用语义与先验融合。
- 不要把 V3.3 行为静默塞进 `clean_reference_v32`，否则历史参数和测试
  无法解释。
- V3.3 继续使用 V3.2 的不可变 Profile、启动抑制、环境变化 abstain 和
  生命周期基础，但使用独立事件数据结构，避免破坏 V3.2 回归结果。

## 4. 证据模型

### 4.1 PriorProposal

由 Clean Reference 产生，建议新增冻结数据类：

```python
PriorProposal(
    box_xyxy: tuple[float, float, float, float],
    anomaly_score: float,
    region_id: str,
    support_pixels: int,
)
```

含义：该位置相对干净参考图出现了局部持续变化。它不是垃圾分类结果。

### 4.2 SemanticObservation

由 `turhancan_yolov8m_seg_trash.pt` 产生：

```python
SemanticObservation(
    box_xyxy: tuple[float, float, float, float],
    confidence: float,
    class_name: str,
    source: Literal["full_roi", "prior_crop"],
    matched_prior_index: int | None,
)
```

所有坐标在进入融合层前必须映射回 Profile 分辨率空间。

### 4.3 FusedObservation

```python
FusedObservation(
    box_xyxy: tuple[float, float, float, float],
    region_id: str,
    prior_score: float | None,
    semantic_confidence: float | None,
    semantic_class: str | None,
    evidence_kind: Literal[
        "semantic_and_prior",
        "semantic_only",
        "prior_only",
    ],
)
```

融合框选择规则：

- 有语义框时优先使用语义分割/检测框作为显示框。
- 语义框明显小于先验支持区域时，事件 anchor 使用二者加权中位框，显示框
  仍使用语义框。
- `prior_only` 仅保留为内部事件，不进入垃圾 OSD。

### 4.4 匹配规则

语义框与先验框满足任一条件即可匹配：

- IoU ≥ 0.10；
- 语义框中心位于先验扩展框内；
- 中心距离 ≤ `max(24px × profile_scale, 0.5 × 较大框短边)`，且面积比
  不超过 8。

同一语义框只能匹配一个先验候选，使用最大 IoU、最小中心距离作为排序键。

## 5. Phase 0：实际模型可行性闸门

DS4.1 在写融合状态机前先完成这个实验。实验失败时必须停止并报告，不能
用更多时序规则掩盖模型本身不具备的小目标语义能力。

### 5.1 数据

**语料实际盘点（2026-09-18 核对，勿再写"5 个位置"）**

盘上可用的已审核真实垃圾语料只有 **3 件已知物体**，其中仅 2 件有候选观测：

| 物体 | 原生框（2560×1440） | 帧数 | 来源 |
|---|---|---|---|
| `item-001` | `[837, 597, 925, 668]`（88×71px） | 17 | `output/litter_source_20260914/deduplicated_labels.json` |
| `item-002` | `[565, 627, 599, 651]`（34×24px） | 5 | 同上 |
| `item-003` | `[803, 565, 818, 573]`（**15×8px**） | 13 张裁剪 | 同上 `user_confirmed_additions`，被 `too_small` 全数过滤、`final_retained_frames: 0` |

原始帧：`output/litter_source_20260914/review/live000..live019.jpg`（2560×1440）
与 `local_10m_network/snapshots/frame-000..019.jpg`。

同目录 `decision.json` 明确记录：`unique_reviewed_litter_count: 2`、
`known_independent_litter_count: 3`、`limited_sample_item_recall: 0.667`、
`"no independent truth labels"`、`"decision": "do_not_deploy"`，且 `next_action` 是
"limit_to_review_queue_or_collect_verified_same-view_negative/positive samples
before any classifier trial"。

负例来源：

- `output/ground_litter_v32_production_hardening_20260918/`（下午生产帧与采样）；
- `output/litter_4h_review_20260913/`（4 小时影子运行，含已知固定物/扫把误报位置）；
- `output/ground_litter_lifecycle_fixture_v32_20260918/` 的 clean 段。

注意：该 fixture 的 `MANIFEST.json` 标注 `"semi_synthetic": true`，由
`/Users/mlcbppg/Desktop/9月17日/小垃圾正样本.mp4` 拼接，**不能计入独立真实投放样本**。

每个样本必须保存：原图、目标框、先验框、人工标签和来源时间，不得只保存
模型输出。

### 5.2 网格实验

对每个 PriorProposal 生成上下文裁剪，至少比较：

- 扩边倍数：2.0、3.0、4.0；
- 输入尺寸：640、960、1280；
- confidence：0.05、0.10、0.15、0.20、0.25；
- 单裁剪和批量裁剪结果一致性。

同时运行完整 ROI 分块扫描作为基线。

### 5.3 闸门

**前置警告（2026-09-18 实测发现，先读再定闸门）**

`output/litter_source_20260914/` 的 `inference_part*/results.jsonl` 里带有逐帧
`targets` 权威标注，与 `deduplicated_labels.json` 的 `true_litter` 直接冲突：

| dedup 条目 | dedup 标注 | 实际落在 | targets 文件的真实标注 |
|---|---|---|---|
| `item-001` | `true_litter` | `curb_bags` | `bag_like_objects_check_roi_membership`，`inside_roi: false` |
| `item-002` | `true_litter` | `broom` | **`non_litter_tool_provisional`**（即扫把硬负例） |
| `item-003` | 用户确认垃圾 | `walkway_small_white` | `..._probable_litter_unverified`（**唯一正例**） |

即：dedup 文件里那 2 个 `true_litter` 条目，在更权威的 `targets` 记录里分别对应
"ROI 未确认的疑似袋状物"和"非垃圾工具（扫把）"。**把 `item-002`（扫把）当作正例，
等于奖励了 V3.3 本来要压制的那个误报。**

因此**当前语料下可用的无歧义真实垃圾正例只有 1 个**（`walkway_small_white`，
18×19px）。在 N=1 上谈论"recall ≥ 80%"没有统计意义，且会掩盖"模型只会认这一件"的
风险。

**修正后的闸门**

A. 语料前置条件（必须先满足，否则 Phase 0 只能出"不可判定"报告）：

- 完成 `targets` 与 `dedup` 的标注仲裁，产出 `item-00x ↔ target_id` 的确认映射；
- 明确把扫把、桶等 `non_litter_*` 标为**硬负例**，不得出现在正例集；
- 在同机位补采至少 8–10 件**独立**真实垃圾（不同物体、不同时段），
  以及等量的同机位硬负例（停放电动车、蔬菜摊、扫把、桶、固定设施、地面反光/水渍）。

B. 满足 A 之后，进入实现的闸门：

- 逐物体报告命中/漏检，**不允许只给平均数**；任何一件漏检都必须单独解释；
- 在**至少 8 件独立正例**上，裁剪命中率 ≥ 80%；
- 硬负例：在选定配置下，**每件硬负例的裁剪命中帧率不得超过同配置正例的最低命中率**——
  即模型必须能把真垃圾与扫把/蔬菜/电动车分开，而不是一起点亮；
- 时序 2/4 命中后，硬负例的确认事件为 0；
- 选定配置在 RTX 3060 Ti 上的批量 crop P95 推理 ≤ 800 ms；
- 完整 ROI 低频扫描 P95 ≤ 1500 ms。

C. 若 A 未完成，或 B 的正例数不足：

输出实验报告并明确写"不可判定"，然后选择以下路线之一：

- 先补采同机位已验证正/负样本（与 `decision.json` 的 `next_action` 一致）；
- 增加专用小目标垃圾/异物分类器；
- 微调现有模型；
- 保留 prior-only 人审，不把它显示为垃圾。

**不得**因为正例不足就放宽阈值、延长确认时间或直接把先验候选当垃圾显示。

### 5.4 Phase 0 实测结果（2026-09-18 已执行）

DS4.1 **不需要**从零重跑 Phase 0——本节闸门已被实测执行一次，结论是**未通过**：

- 报告：`output/ground_litter_v33_phase0_20260918/VERDICT.md`（含逐目标数据）
- 脚本：`scripts/run_ground_litter_v33_phase0.py`
- 结构化结果：同目录 `phase0_corpus.json` / `phase0_cropgrid.json` /
  `phase0_summary.json` / `phase0_verdict.json` / `REPORT.md`

要点：

1. **放大裁剪假设成立**：18×19px 真小垃圾在全 ROI 扫描下 11/20 帧命中、
   conf 0.21–0.34；按 §6.2 规则裁剪放大后 **16/20 帧、中位 conf 0.50**。
2. **但硬负例失败**：`bucket`（61×63px 非垃圾容器）在 imgsz640 下命中率
   0.90–1.00、中位 conf 0.46–0.61，**高于真垃圾**。唯一能把桶压到 0 的配置
   （exp2/imgsz1280/conf0.25）同时把正例压到 0.75（低于 80%）。
   **没有任何配置同时通过召回与硬负例闸门。**
3. **语料先决问题**：无歧义正例只有 1 件；`item-001`/`item-002` 分别落在
   `curb_bags`（ROI 未确认）和 `broom`（非垃圾扫把）内。
4. 延迟闸门未测（需 RTX 3060 Ti）。

**因此：在补采语料并重跑之前，不要开始 §9 的融合主链开发。** 下一步按 §5.3 C
执行，并先完成 §5.3 A 的标注仲裁与样本补采。

## 6. 运行时流水线

### 6.1 每个分析 tick

默认 `analysis_fps=0.5`，即每 2 秒一个 tick。

顺序必须是：

1. 取该 pad 最新帧；过期帧直接丢弃。
2. Profile 对齐和 protected normalization。
3. 若处于启动抑制期：返回 `warming_up`，不运行垃圾模型，不累计事件。
4. 若环境不是 `NORMAL`：返回 `abstaining`，清空 V3.3 事件内存，不运行
   垃圾模型。
5. 连续正常样本不足阈值：返回 `abstaining`，不累计事件。
6. 生成 PriorProposal。
7. 按优先级选取最多 N 个先验裁剪，批量运行垃圾模型。
8. 到达完整 ROI 扫描周期时，再运行一次完整 ROI 分块扫描。
9. 融合、NMS、actor/context 过滤。
10. 更新 V3.3 事件状态机。
11. 只把具有语义证据并达到确认条件的事件投影到 OSD。

### 6.2 Prior-guided crops

默认配置：

- 每 tick 最多 4 个裁剪；
- 先按未确认事件、anomaly score、持续时间排序；
- 裁剪中心为 prior box 中心；
- 边长为 `max(160px, prior_long_side × 4)`；
- 最大源图裁剪边长 480px；
- 正方形裁剪，越界补边；
- 模型输入 640，作为一个 batch 推理；
- 记录 crop 到全图的仿射映射并将检测框映射回去。

禁止逐 crop 单独调用模型，否则 GPU kernel 启动和预处理开销会线性增加。

### 6.3 Full ROI scan

完整 ROI 扫描用于发现先验没有提议但模型能够识别的目标。

默认：

- 间隔 6 秒；
- 仅覆盖 16 点 ROI，不覆盖道路、墙面和商户排除区；
- `tile_size_px=640`；
- `inference_imgsz=640`；
- `tile_overlap=0.20`；
- `maximum_tiles=8`；
- 所有 tile 批量或小批量推理；
- 运行时间超过预算时，优先延长 full-scan 周期，不降低主链 FPS。

### 6.4 Actor/context

- `actor_model` 默认必须是 null。
- 复用主 DeepStream 链上的 person/vehicle metadata。
- PriorProposal 和 SemanticObservation 都要执行 actor overlap 过滤。
- actor/context 只表示当前位置当前不可判断，不能写入 Clean Reference，
  也不能永久否定垃圾事件。
- 进入 `OCCLUDED` 后隐藏框并暂停确认/清除计时。

### 6.5 队列和背压

主视频链不能等待 V3.3。

- 保持独立进程和 leaky GStreamer side branch。
- 输入必须是 latest-wins。**已核实**：`ground_litter_process.py` 的 `submit()`
  在 `queue.Full` 时确实丢掉新帧并保留旧帧，与它自己的注释
  （"A queued old frame is less useful than the next fresh one"）相反。
- **严重度校正（实测）**：输入队列 `maxsize = max(pads*2, 2)`，单路即 **2**。
  所以积压是**有界的（约 2 个周期）**，不会无限增长——不要把它描述成"逐渐卡死"。
  真实后果是：当子进程慢于周期时，新帧被持续拒绝、子进程处理的是最多 2 个周期前
  的旧帧，且分析结果对应的是陈旧画面。
- 推荐为每个 pad 保存一个最新帧 mailbox，通知队列只传 pad/version；如果
  暂不重构 mailbox，子进程必须检查 monotonic timestamp，丢弃年龄超过
  `max(2 × analysis_period, 4s)` 的帧。
- 输出队列继续 latest-wins（`_put_latest()` 实现正确：满时丢旧插新）。
- 记录 `input_frame_age_ms`、`dropped_analysis_frames` 和
  `semantic_queue_depth`。

### 6.6 环境闸门硬悬崖（原稿缺失，V3.3 必须处理）

原稿 §3 说 V3.3"继续使用 V3.2 的…环境变化 abstain"，§6.1 第 3–4 步在非 `NORMAL`
时直接 abstain 且不跑语义模型。**这会原样继承一个已实测的硬悬崖**：

- 判据在 `protected_normalize()` 尾部：`saturated_fraction > 0.35` → `ENVIRONMENT_CHANGE`；
  `max|gain-1| > 0.10` 或 `max|bias| > 18` 或 **`local_extent > 16`** → `GLOBAL_LIGHT_CHANGE`；
  否则 `NORMAL`。
- 2026-09-18 以真实 H.265 帧实测：几何对齐 799 内点 / 0.81px、平均亮度仅差 3.7%、
  gains 1.022–1.039、biases −4.05..−5.40、saturated 0.0074 —— 全部大幅通过；
  **唯一超标项是 `local_extent = 17.15` vs 阈值 16，只超 1.15**，结果整路 abstain、
  零候选零事件。
- 这些阈值是 `ground_litter_v32.py` 的**模块常量**，`api.py` 与 detection options
  里**都没有对应字段，无法经 API 配置**。

由此产生两个必须解决的问题：

1. **V3.3 在光线失配时段依然全盲**，而且因为第 4 步不跑模型，垃圾语义证据根本不会
   产生——再好的融合也无从谈起。需要在 V3.3 中定义降级路径：环境非 `NORMAL` 时
   允许**只做语义扫描**（不累计生命周期、不显示框），或明确把"profile 时段匹配"
   写成运行前置条件并在 API 上拒绝创建。
2. **Phase 0 通过不等于现场可用**。Phase 0 在离线帧上验证模型能力，不经过环境闸门；
   必须额外补一个"闸门时段覆盖"验证：至少在两个不同时段各取一帧，确认
   `local_extent` 落在阈值内，否则该 profile 不能用于该时段。

**不要**通过放宽 `local_extent` 来"解决"这个问题——那会同时放过真实场景变化，
属检测语义变更，必须有多小时回归证据。

## 7. V3.3 事件状态机

建议新建 `V33EventMemory`，不要继续在 `V32EventMemory` 中堆条件。

### 7.1 事件字段

至少包括：

- `event_id`
- `anchor_box`
- `display_box`
- `first_seen`
- `last_prior_at`
- `last_semantic_at`
- `prior_hits`
- `semantic_hits`
- `semantic_class`
- `maximum_semantic_confidence`
- `evidence_kind`
- `confirmed_at`
- `closed_at`
- `closed_reason`
- `state`
- 有界 `state_history`

### 7.2 确认规则

默认建议：

| 证据类型 | 确认条件 | 是否显示垃圾框 |
|---|---:|---|
| semantic + prior | 4 个样本窗口内至少 2 次语义命中，且 prior 持续 | 是 |
| semantic only | 4 个样本窗口内至少 3 次语义命中 | 是 |
| prior crop semantic | 按 semantic + prior | 是 |
| prior only | 持续 20 秒后进入 `REVIEW_PENDING` | 默认否 |

所有确认都必须发生在：

- 启动抑制结束后；
- 环境连续稳定；
- actor/context 未遮挡；
- 候选中心位于 ROI 且不在 exclusion 中。

### 7.3 显示规则

只有以下状态允许进入 OSD：

- `SEMANTIC_VISIBLE`
- `SEMANTIC_ANOMALY_PENDING`，且事件已经确认并仍有 prior/semantic 支持

以下状态必须隐藏：

- `WARMING_UP`
- `ENVIRONMENT_CHANGE`
- `STABILITY_PENDING`
- `PRIOR_ONLY`
- `REVIEW_PENDING`
- `OCCLUDED`
- `GROUND_UNAVAILABLE`
- `CLEAN_PENDING`
- `CLEARED`
- `EXPIRED_PENDING`
- `MERGED`

### 7.4 清走和再次投放

- 使用 Clean Reference 的有效干净观察作为清走证据；垃圾模型未检出本身
  不能证明目标已经清走。
- 默认连续有效干净时间 6 秒，即 0.5 FPS 下至少 3 个样本。
- 清走确认后关闭 event ID。
- 同位置再次出现必须生成新 ID，不复用历史 ID。
- 环境非正常时不累计 clean 时间。

## 8. API 配置

新增或复用字段如下。字段名是实施契约，若 DS4.1 需要调整，必须同步 API、
序列化、OpenAPI、示例请求和测试。

```json
{
  "ground_litter": {
    "enabled": true,
    "mode": "hybrid_v33",
    "profile_id": "camera_01_v32_afternoon_1080p",
    "model": "turhancan_yolov8m_seg_trash.pt",
    "actor_model": null,
    "analysis_fps": 0.5,

    "confidence": 0.15,
    "night_confidence": 0.12,
    "nms_iou": 0.5,

    "semantic_full_scan_interval_seconds": 6.0,
    "semantic_max_prior_crops": 4,
    "semantic_crop_expansion": 4.0,
    "semantic_crop_source_max_px": 480,
    "semantic_crop_imgsz": 640,
    "semantic_min_hits": 2,
    "semantic_hit_window": 4,
    "display_prior_only": false,
    "prior_only_review_seconds": 20.0,

    "tile_size_px": 640,
    "inference_imgsz": 640,
    "tile_overlap": 0.2,
    "maximum_tiles": 8,

    "startup_suppress_seconds": 15.0,
    "normal_stability_samples": 3,
    "clear_confirm_seconds": 6.0,
    "pending_expire_seconds": 20.0,
    "maximum_boxes": 4
  }
}
```

验证范围建议：

- `semantic_full_scan_interval_seconds`: 2–60
- `semantic_max_prior_crops`: 1–8
- `semantic_crop_expansion`: 1.5–6.0
- `semantic_crop_source_max_px`: 160–960
- `semantic_crop_imgsz`: 320–1280
- `semantic_min_hits`: 1–10
- `semantic_hit_window`: `semantic_min_hits`–20
- `prior_only_review_seconds`: 5–300

新建示例：

`config/ground_litter_v33_hybrid_stream_request.example.json`

不要覆盖 V3.2 示例，以便回归和回滚。

## 9. 模块边界和逐文件实施顺序

### 9.1 新模块

建议新增：

`rtsp_annotator/ground_litter_v33.py`

包含：

- `PriorProposal`
- `SemanticObservation`
- `FusedObservation`
- `V33Event`
- `V33EventMemory`
- `HybridGroundLitterProcessor`
- 纯函数匹配、融合和坐标映射

该模块不能直接依赖 Ultralytics 对象。模型推理由 process/detector 层注入，
使状态机能够用假语义结果做确定性单元测试。

### 9.2 修改 `ground_litter_v32.py`

- 抽出可复用的单帧先验分析结果，但保持 V3.2 对外行为不变。
- 建议新增 `CleanReferenceFrameAnalysis`：包含 normalized frame、valid、
  support、prior proposals、environment state 和坐标缩放信息。
- `CleanReferenceV32Processor.update()` 可以调用该分析函数，V3.3 也调用同一
  函数，避免两套 normalization 漂移。

### 9.3 修改 `ground_litter_detection.py`

- `GROUND_LITTER_MODES` 增加 `hybrid_v33`。
- 增加第 8 节配置字段、验证和 payload 往返。
- 给 `UltralyticsGroundLitterDetector` 增加批量 prior-crop 推理方法。
- 增加 full ROI scan 调度所需的 tile batch API。
- 不修改 legacy yolo candidate 的默认结果。

### 9.4 修改 `ground_litter_process.py`

- `yolo`：保持原处理器。
- `clean_reference_v32`：保持硬化 V3.2；没有 actor model 时不构造
  Ultralytics detector。
- `hybrid_v33`：必须构造垃圾 detector，但 actor model 仍为 null。
- 建立每 pad full-scan 调度时间。
- 每 tick 批量运行 prior crops；到期再运行 full scan。
- 修复输入 latest-wins/过期帧问题。
- 分开记录 prior、crop semantic、full semantic 和 fusion 耗时。

### 9.5 修改 `deepstream_manager.py`

- `hybrid_v33` 必须验证 Profile 和垃圾模型都存在。
- V3.2 只验证 Profile；yolo 只验证模型；错误信息必须指出缺少的是哪一类
  资产。
- 继续限制路径越界并校验 Profile SHA。

### 9.6 修改 `deepstream_worker.py`

- 主链仍只负责取样、actor metadata 和 OSD。
- 增加第 10 节 metrics。
- OSD 标签来源：语义 class 可选显示；默认仍为“疑似垃圾”。
- prior-only 不得进入垃圾 OSD。
- side snapshot 过期时清空框，不保持陈旧垃圾框。

### 9.7 修改 Dockerfile 和文档

- Dockerfile 复制 `ground_litter_v33.py` 并执行 `py_compile`。
- 候选镜像构建检查必须真实加载垃圾模型或至少验证权重可由 Ultralytics
  解析，不能只检查文件大小。
- 更新 `HTTP_API.md`、生产验收文档和 Postman 示例。

## 10. 可观测性

在 `ground_litter` 状态及顶层 metrics 中增加：

- `prior_candidates`
- `semantic_candidates`
- `semantic_crop_candidates`
- `semantic_full_scan_candidates`
- `fused_candidates`
- `prior_only_events`
- `semantic_confirmed_events`
- `semantic_model_runs`
- `semantic_crop_batches`
- `semantic_full_scans`
- `last_prior_ms`
- `last_semantic_crop_ms`
- `last_semantic_full_scan_ms`
- `last_total_analysis_ms`
- `input_frame_age_ms`
- `dropped_analysis_frames`
- `environment_state`
- `evidence_mode: hybrid_v33`

`count` 仍表示实际发送到 OSD 的框数，不能包含 prior-only。

状态消息示例：

```text
hybrid_v33 profile=camera_01_v32_afternoon_1080p env=NORMAL
prior=3 semantic_crop=1 semantic_full=0 fused=1 displayed=0
```

禁止把输入 URL、输出 URL、API key 或完整模型路径写入 metrics/log。

## 11. 性能预算

目标服务器：RTX 3060 Ti 8 GiB。

**先验阶段的实测基线（2026-09-18，生产服务器 `/proc` 侧进程 CPU 计时真值）**

| 环境状态 | 实测 `last_inference_ms` |
|---|---|
| `NORMAL` 稳定场景 | **1076.7 CPU-ms/帧**（32.30 CPU-秒 ÷ 30 帧），观测区间 750–1080 ms |
| `GLOBAL_LIGHT_CHANGE` | 1800–3150 ms |

`protected_normalize()` 的掩膜/膨胀/连通域是数据相关的，所以该值随环境状态变化。

**因此原稿"prior < 300 ms"不成立**：先验单独就 ~1.08 s，比原预算高约 3.6 倍。
按原稿的 300+800+1500 相加会严重低估——实际 full-scan tick ≈ 1080+800+1500 ≈ 3380 ms，
**必然超过 2000 ms 的 0.5 FPS 周期**。

硬指标：

- 主链 `capture_fps`、`pre_encode_fps`、`publish_fps` ≥ 20 FPS，目标约 25。
- `duplicate_publish_fps = 0`。
- side process 不得让 GStreamer pipeline 进入 unhealthy。
- 0.5 FPS 下，`last_total_analysis_ms` P95 < 2000 ms。**注意 full-scan tick 会超**，
  必须靠 §6.3 的"延长 full-scan 周期"或 §6.2 的裁剪数量来摊平，且必须实测确认。
- 分阶段目标（基于实测基线，而非假设）：
  - prior ≤ 1100 ms（即当前实测水平；若要做到 < 500 ms 需先做算法优化，
    见 §11.1）；
  - crop batch ≤ 800 ms；
  - full scan ≤ 1500 ms；
  - **三者不得在同一 tick 内无条件叠加**。
- `max_sample_gap` 交互：`analysis_fps=0.5` → `sample_period=2.0`、
  `max_sample_gap=3.0s`。任何超过 3.0 s 的 tick 都会让 `observe_clean()`
  重置 `clear_observed_seconds`，使"清走确认"无法完成（V3.2 上已实测到
  `cleared_events` 长期为 0）。V3.3 必须改为按**真实经过时间**累计清走证据，
  而不是按 `sample_period` 累加。
- side process 排队帧年龄 < 4 秒；超过即丢弃。
- V3.3 额外 GPU 显存应保持在 1.5 GiB 内，具体值必须记录。

### 11.1 若要把 prior 压到 500 ms 以内

已定位的四个热区（单帧 1920×1080 口径）：

| 阶段 | 实测 | 可做的优化 |
|---|---|---|
| `_robust_global_color` | ~1264 ms | 仅在采样像素上算残差，避免整幅 float32 预测/残差重建 |
| `residual_maps`（3× sigma=12 全幅） | ~571 ms | 半分辨率模糊后上采样 |
| `default_field`（sigma=24，ksize≈145） | ~417 ms | 降采样后模糊，或只在 protected 包围盒内计算 |
| `propose_v32` | ~773 ms | 连通域与生长按需裁剪 |

这些都会改变检测数值，属于**检测语义变更**，必须配回归证据。V3.3 若需要该预算，
应作为独立工作项，不要顺手混进融合开发。

**测量方法警告**：不要在 API 容器内跑 OpenCV 基准来估这些耗时——宿主 CPU 被
`idle_inject` 节流，两个 16 线程 OpenCV 进程互相争用会把结果放大 2–3 倍
（基准 2056–2900 ms vs 真实 1077 ms）。测活进程的 `/proc/<pid>/stat` utime+stime。

降级顺序：

1. 延长 full ROI scan 周期；
2. 减少 prior crops 数量；
3. 将 crop imgsz 从 960/1280 降至 640；
4. 将 analysis_fps 从 0.5 降至 0.25；
5. 绝不能降低主视频链帧率来保 side analysis。

## 12. 测试计划

### 12.1 单元测试

至少覆盖：

- API `hybrid_v33` 字段验证和 payload 往返；
- 缺 Profile、缺模型、路径越界；
- prior crop 坐标映射回全图；
- crop batch 结果顺序和空结果；
- semantic + prior 2/4 确认；
- semantic-only 3/4 确认；
- prior-only 永不进入垃圾 OSD；
- startup 不运行语义模型、不累计事件；
- `GLOBAL_LIGHT_CHANGE` 清空事件并 abstain；
- 稳定 3 帧后恢复；
- actor/context 遮挡暂停计时；
- 清走后关闭；
- 同位置再次投放获得新 ID；
- 合并不会把 prior-only 升级成 semantic-confirmed；
- maximum boxes 和 closed-event memory 有界；
- latest-wins 或过期帧丢弃；
- side error 清空旧框，主链不退出。

### 12.2 离线集成测试

- 使用现有 V3.2 生命周期 fixture，注入确定性的 fake semantic results，
  验证完整状态链。
- 使用真实 `turhancan` 权重跑审核正负样本，报告每个目标，不只给汇总。
- 使用当前下午生产采样，启动和稳定段 displayed count 必须为 0。
- 检查模型实际被调用：`semantic_model_runs > 0`，不能只通过 mock。

### 12.3 回归测试

- `tests/` 全量通过；不要从仓库根收集 `archives/`。
- V3.2 既有 fixture 结果不变。
- legacy yolo ground-litter 路径结果不变。
- API 旧请求仍可解析。

## 13. 验收闸门

### 离线必须满足

- Phase 0 语料前置条件（§5.3 A）完成，标注冲突（`item-001↔curb_bags`、
  `item-002↔broom`）已仲裁并有书面结论。
- Phase 0 模型可行性闸门通过（§5.3 B）。
- 在**补采后的**独立正例集上逐物体报告命中/漏检；漏掉的每一件单独说明。
  若正例集仍不足 8 件，只能输出"不可判定"报告，**不得声称召回已验收**。
- 硬负例（扫把/桶/电动车/蔬菜/固定设施/反光）在选定配置下不得与正例同等点亮。
- 审核清洁/环境变化段 displayed false events = 0。
- prior-only 永远不以“疑似垃圾”输出。
- 生命周期 fixture：确认、遮挡、清走、再次投放全部通过。
- 全量测试、compileall、`git diff --check` 通过。

> 原稿此处写"5 个已审核垃圾位置至少 4 个获得语义证据"。按 §5.1 实测，无歧义正例
> 只有 1 件，该条**在当前语料下无法满足**，会把完成定义锁死，故改写如上。

### 生产 canary 必须满足

1. 启动 15 秒：零 active/confirmed/displayed 垃圾事件。
2. 稳定恢复后连续观察至少 30 分钟普通现场。
3. 普通现场 displayed false boxes = 0。
4. `semantic_model_runs` 持续增长，证明垃圾模型真实参与。
5. 主链 FPS ≥ 20，目标约 25；duplicate FPS = 0。
6. side P95 低于 2 秒周期，无持续积压。
7. 输出流连续解码至少 10 分钟，无静止帧段和断流。
8. 如能布置可回收测试物：语义确认、遮挡隐藏、清走关闭、再次投放新 ID
   全部通过。

无法布置真实物体时，只能批准“清洁现场低误报 canary”，不能宣称现场垃圾
召回已经验收。

## 14. 部署策略

**生产状态校正（2026-09-18 实测，原稿此处已过期）**：生产**已经**运行 V3.2 硬化版
`rtsp-yolo-annotator:deepstream8-ground-litter-v32-hardening-20260918`
（`sha256:7aa71d92…`，15:13:36 CST 启动，`RestartCount=0`），不是"旧 V3.2 / 未切换"。
回滚标签 `…-before-ground-litter-v32-hardening-20260918` → `sha256:841ca526317f…`。
V3.3 应以当前工作区 hardening 代码为基线开发，不要从 archive 或线上旧镜像反向覆盖。

建议镜像标签：

`rtsp-yolo-annotator:deepstream8-ground-litter-v33-hybrid-20260918`

部署顺序：

1. 完成离线闸门和独立候选镜像检查。
2. 记录当前 API 精确镜像标签与活动流（当时为 `481fb4213b5e…`，`analysis_fps=0.5`，
   下午 profile）。签名 URL **只在内存或 mode-600 临时文件中**保存，且注意：
   签名地址**不会在流删除后存活**，删流后必须用 `ctseelink` 重新获取。
3. 新增独立 V3.3 Compose override，不覆盖 V3.2 override。
4. 校验完整 Compose 链（当前是六文件，见
   `GROUND_LITTER_V32_HARDENING_RESULT_20260918.md` §2）。
5. 只重建 API（会终止进程内所有流任务，需重建）。
6. 使用 V3.3 示例请求恢复 canary 流。
7. 按第 13 节观察；失败时移除 V3.3 override，回到 V3.2 硬化镜像并恢复流。

生产切换、活动流恢复、回滚和敏感信息处理细节继续遵循：

- `GROUND_LITTER_V32_HARDENING_RESULT_20260918.md`（权威实测状态）
- `HANDOFF_GROUND_LITTER_V32_HARDENING_20260918.md` 第 9 节（安全/凭据规则仍有效；
  第 1 节与 8.A–8.C 已过期）

## 15. DS4.1 交付物清单

DS4.1 完成时必须交付：

- Phase 0 小目标模型可行性报告及结构化 JSON；
- `rtsp_annotator/ground_litter_v33.py`；
- detector/process/manager/worker/API 集成；
- V3.3 Postman 请求示例；
- 单元和集成测试；
- 实际权重离线评估报告；
- 性能报告，含主链 FPS、side 各阶段 P50/P95、GPU 显存；
- ROI/语义/prior/fusion 调试可视化；
- 候选镜像、manifest、SHA-256 和回滚步骤；
- 生产 canary 记录。

## 16. 禁止事项

- 不得把 prior 和 semantic 框简单取并集。
- 不得让 prior-only 框默认显示为垃圾。
- 不得在环境非正常或启动阶段累计垃圾事件。
- 不得重新启用第二套 person/vehicle actor 模型作为默认配置。
- 不得逐裁剪串行调用垃圾模型。
- 不得让 side process 背压主视频链。
- 不得通过提高确认时间掩盖模型不可用或 Profile 不匹配。
- 不得把签名输入 URL、输出凭证或 API key 写入代码、文档、日志或测试夹具。
- 不得清理当前 dirty workspace、archives、现有模型、引擎或审核输出。

## 17. 完成定义

V3.3 只有在以下事实同时成立时才算完成：

1. 垃圾模型真实运行并为所有显示的垃圾框提供语义证据。
2. Clean Reference 真实参与小目标提议、环境判断和清走判定。
3. prior-only 目标不会冒充语义确认垃圾。
4. 启动、光照变化和遮挡期间不闪框。
5. 清洁生产画面低误报 canary 通过。
6. 主输出持续流畅，side analysis 不积压。
7. 真实垃圾召回结果有逐样本证据，而不是仅凭架构推断。
