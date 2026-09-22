# Ground Litter V3.3 双通道召回架构与实施规格

> **部分被取代（2026-09-21）**：semantic 独立召回、事件合并、遮挡/清走和 `latest-wins` 调度仍可复用；Clean Reference 作为独立小垃圾召回通道已被 oracle NO-GO 实验否定，不得继续作为生产路线。正式决策见[背景 Profile 先验路线 NO-GO](../decisions/2026-09-21-ground-litter-profile-prior-no-go.md)，下一路线见[现场小垃圾检测模型](2026-09-21-ground-litter-small-detector-roadmap.md)。

状态：**历史实现契约；prior 召回部分已停止**

日期：2026-09-18

实施目标：同时保留垃圾模型对常规垃圾的检测能力，并用 Clean Reference 补充模型漏掉的
小型零散垃圾；两条通道分别抑制误报，在事件层关联、去重和管理生命周期。

本文件取代：

- `2026-09-18-ground-litter-v33-hybrid-architecture.md` 中“所有显示框必须具有语义证据”
  的架构；
- `2026-09-18-v33-phase0-architecture-decision.md` 中“prior-only 只能进入人工审核”
  的决策。

上述文件保留为实验与决策历史，不得继续作为实现依据。

## 1. 用户目标与完成标准

V3.3 必须实现三类有效结果：

1. `turhancan_yolov8m_seg_trash.pt` 原本能识别的瓶子、袋装垃圾、塑料袋等，
   即使 Clean Reference 没有提议，也能独立确认并显示；
2. 垃圾模型漏掉的纸片、小包装、零散碎屑等，只要 Clean Reference 提议通过自身
   环境、尺寸和时序过滤，也能独立确认并显示；
3. 两路同时发现同一目标时只显示一个框，使用更快的确认路径，并由统一状态机负责
   遮挡、清走和同位置再次出现。

系统允许遗留少量误报。本版本需要减少明显的单帧、极小连通域、光照、反光、人物和
车辆误报，并提供来源指标，供后续针对剩余误报继续优化。本版本不以“误报必须为零”
为设计前提。

### 1.1 不可违反的产品规则

- 模型通道和先验通道是两个独立召回通道；任一路都可以独立确认事件。
- `prior_only` 不需要垃圾模型语义确认，不得因模型漏检而被否决。
- `semantic_only` 不需要 Clean Reference 变化确认，不得因 Profile 不匹配而停止。
- 融合是两个已经过滤的事件流的并集，不是两个单帧原始框的直接并集。
- 内部必须保留 `semantic_only`、`prior_only`、`semantic_and_prior` 来源。
- 对外默认均显示 `疑似垃圾`；API 和 metrics 必须能区分来源。
- 置信度不同源不可直接比较；默认 OSD 不显示跨源混合置信度。

## 2. 本期范围与明确排除项

### 2.1 本期实施

- 新增 `mode=hybrid_v33`；
- 垃圾模型完整 ROI 分块检测；
- Clean Reference 小目标提议；
- prior-guided crop 垃圾模型辅助推理；
- 两通道各自的候选过滤和事件确认；
- 事件级关联、去重、遮挡、清走和再次出现；
- latest-wins、过期帧丢弃、性能和分通道指标；
- 完整单元、集成、真实模型 smoke 和离线回放验证。

### 2.2 本期不处理

“单个先验 Profile 素材不足，无法覆盖全天环境”是独立项目，不在本方案中实现：

- 不实现七天回放采样；
- 不实现多环境 Profile Factory；
- 不实现环境聚类和 Reference Bank；
- 不通过放宽 `local_extent` 常量假装覆盖全天；
- 不改变现有 Profile 文件内容。

本期假设当前 Profile 只在已验证的环境时段提供 prior 能力。Profile 不匹配时：

- prior 通道进入 `PRIOR_ABSTAINING`，不产生或清除 prior 事件；
- semantic 通道继续运行和独立输出；
- 整个 V3.3 不能因为 prior 环境异常而进入全局 `abstaining`。

## 3. 总体架构

```text
原生帧 + 主链 actor/context metadata
                │
      ┌─────────┴─────────┐
      │                   │
      ▼                   ▼
Semantic 通道          Prior 通道
turhancan ROI tiles     Clean Reference
      │                   │
ROI/尺寸/置信度          环境/变化/尺寸
actor/时序过滤           actor/稳定性/时序过滤
      │                   │
semantic events         prior events
      └─────────┬─────────┘
                ▼
       V3.3 统一事件关联层
  semantic_only / prior_only / both
                │
       遮挡、清走、再次出现、去重
                │
                ▼
              OSD/API
```

垃圾模型还可以对 prior crop 做批量推理。该结果有两个作用：

- 与 prior 同时命中时提升为 `semantic_and_prior`，缩短确认时间；
- 记录类别和置信度，帮助审核与后续误报优化。

模型在 prior crop 上没有命中时，prior 候选仍按 prior-only 规则继续处理。

## 4. 证据与事件数据结构

建议在新文件 `rtsp_annotator/ground_litter_v33.py` 中定义冻结的数据结构。

### 4.1 SemanticObservation

```python
@dataclass(frozen=True, slots=True)
class SemanticObservation:
    box_xyxy: tuple[float, float, float, float]
    confidence: float
    class_name: str
    region_id: str
    source: Literal["full_roi", "prior_crop"]
    observed_at: float
```

含义：垃圾模型在地面 ROI 内通过单帧候选过滤后的观察。它还不是已确认事件。

### 4.2 PriorObservation

```python
@dataclass(frozen=True, slots=True)
class PriorObservation:
    box_xyxy: tuple[float, float, float, float]
    anomaly_score: float
    region_id: str
    support_pixels: int
    observed_at: float
```

含义：相对 Clean Reference 新出现的局部稳定变化。它不声明材质类别，但可以独立成为
“疑似垃圾”。

### 4.3 HybridObservation

```python
@dataclass(frozen=True, slots=True)
class HybridObservation:
    anchor_box: tuple[float, float, float, float]
    display_box: tuple[float, float, float, float]
    region_id: str
    evidence_kind: Literal[
        "semantic_only",
        "prior_only",
        "semantic_and_prior",
    ]
    semantic: SemanticObservation | None
    prior: PriorObservation | None
    observed_at: float
```

### 4.4 V33Event

每个事件至少保存：

- `event_id`；
- `first_seen_at`、`last_seen_at`；
- `last_semantic_at`、`last_prior_at`；
- `anchor_box`、`display_box`；
- `semantic_hits`、`prior_hits` 和各自的有界时间戳；
- `semantic_class`、`maximum_semantic_confidence`；
- `evidence_kind`，允许随证据变化升级；
- `current_support`，记录当前由 semantic、prior 或两者支撑；
- `confirmed_at`、`closed_at`、`closed_reason`；
- `state` 和有界 `state_history`；
- 按真实时间计算的 clean/unmatched evidence；
- `region_id`。

同一事件可以从 `prior_only` 升级为 `semantic_and_prior`。`evidence_kind` 记录事件历史上
获得过的最高证据等级，不因一次漏检降级；`current_support` 单独决定当前显示和清走逻辑。
不得因为某一帧缺少另一路证据而关闭已确认事件。

## 5. 单帧候选过滤

### 5.1 Semantic 通道

复用 `UltralyticsGroundLitterDetector.candidates()` 的原生像素分块能力，保留：

- zone polygon 和 exclude zone；
- zone 级日间/夜间 confidence；
- 最小短边、最小框面积；
- cross-tile NMS；
- actor/context overlap；
- `maximum_tiles` 上限。

必须增加或确认：

- 框中心必须位于有效地面 mask；
- 同一模型框只归属一个最具体的 region；
- 同一分析时间中同一位置的多 tile 框只能算一次命中；
- 单次 full ROI scan 的多个类别框不能累加成多次事件命中；
- 模型类别只作为记录，不使用类别名改变确认规则。

Semantic 通道允许识别启动时已经存在、因此未形成 prior 变化的垃圾。

### 5.2 Prior 通道

复用 V3.2 的对齐、`protected_normalize()`、`propose_v32()`、ROI 分配和 actor/context
过滤。prior-only 候选必须同时满足：

- 环境状态为 `NORMAL`；
- 已通过 `normal_stability_samples`；
- 中心位于有效 zone 且不在 exclusion；
- 短边和框面积达到该 zone 的最小像素阈值；
- 没有超过 V3.2 已有最大支持面积和最大边长限制；
- support/seed 不是孤立的单像素或压缩噪点；
- 没有与 actor/context 超过重叠阈值；
- 位置、面积和形状在确认窗口内保持可关联。

极小候选过滤必须发生在创建事件之前，不能先创建事件再依靠 OSD 隐藏。

### 5.3 prior-guided crop

每个 prior tick 最多选择 4 个未确认或高 anomaly-score 候选：

- 先按未确认、持续时间、anomaly score 排序；
- 正方形裁剪；
- 边长 `max(160px, prior_long_side × 4)`；
- 源图最大边长 480px；
- 模型输入默认 640；
- 所有 crop 一次 batch 推理；
- 结果映射回 Profile 坐标并与原 prior 关联。

prior crop 没有语义命中不能删除 prior observation。它只是辅助证据。

prior crop 中与发起该 crop 的 prior 无空间关联的额外检测，只记录到诊断数据，不得创建
semantic-only 事件。semantic-only 独立召回只能来自完整 ROI 扫描，避免裁剪采样偏差制造
新的模型事件。

## 6. Semantic/Prior 单帧关联

时间差不超过 `max(semantic_scan_interval_seconds, 2 / analysis_fps)` 的 semantic 与 prior
才允许尝试空间关联。满足任一条件即可关联：

- IoU ≥ 0.10；
- semantic 中心位于 prior 向外扩展 25% 的框内；
- 中心距离不超过 `max(24px × profile_scale, 0.5 × 较大框短边)`，且面积比不超过 8。

采用一对一贪心匹配，排序键依次为：

1. IoU 高；
2. 中心距离小；
3. semantic confidence 高。

关联后：

- `display_box` 优先使用 semantic box；
- `anchor_box` 使用两路框的稳健中位或 prior box；
- 未匹配 semantic 形成 semantic-only observation；
- 未匹配 prior 形成 prior-only observation。

禁止把没有空间关系的两个候选仅因时间相近而合并。

### 6.1 跨时间事件关联与合并

每个 HybridObservation 先与现有 active event 做一对一空间匹配，继续使用有界的中心距离、
尺寸比例和 IoU 规则。若 semantic 和 prior 曾先后创建两个独立 pending event，后来证明
两者属于同一位置：

- 以较早 event ID 为主事件；
- 合并两路时间戳、框历史和最高分数；
- 一个时间戳仍只计一次对应通道命中；
- 次事件以 `merged_into:<event_id>` 关闭；
- OSD 始终只投影主事件。

已经确认且中心明显不同的两个相邻垃圾不得仅因框边缘相交而合并。事件合并需要同时满足
中心距离和尺寸比例约束，并至少在两个不同分析时间保持可关联。

## 7. 独立确认与统一事件状态机

所有命中按“不同分析时间戳”计数。一个 tick 中多个 tile/crop 命中只能算一次。

### 7.1 默认确认规则

| 事件证据 | 默认确认条件 | 设计目的 |
|---|---|---|
| semantic-only | 最近3次 semantic scan 至少2次命中，且跨度≥4秒 | 保留常规垃圾召回并过滤单帧误报 |
| prior-only | 最近6次 prior tick 至少4次命中，且可见跨度≥6秒 | 允许独立补充小垃圾，同时比模型通道更保守 |
| semantic+prior | 最近4个分析 tick 中，两路在至少2个不同时间形成关联，跨度≥2秒 | 双证据快速确认 |

这些是实现默认值，不是不可修改常量。配置必须有边界校验，测试使用确定值，生产参数
必须通过离线回放报告确认。

### 7.2 状态

建议状态：

```text
PENDING
SEMANTIC_VISIBLE
PRIOR_VISIBLE
FUSED_VISIBLE
OCCLUDED
CLEAN_PENDING
ABSENT_PENDING
CLEARED
EXPIRED_PENDING
MERGED
```

状态要求：

- semantic-only 达标进入 `SEMANTIC_VISIBLE`；
- prior-only 达标进入 `PRIOR_VISIBLE`；
- 两路达标或已确认事件获得另一路证据进入 `FUSED_VISIBLE`；
- 遮挡时隐藏显示框并暂停确认与清走计时；
- 未确认事件超时后关闭；
- 已确认事件短暂漏检不能立刻关闭；
- 时间倒退或重复结果不得重复累计命中。

### 7.3 启动阶段

默认前 15 秒：

- 可以完成模型加载、Profile 对齐和一次推理预热；
- 不累计事件命中；
- 不输出垃圾框；
- warmup 结束时清空预热结果；
- semantic 从空事件内存开始；
- prior 还需满足环境连续稳定样本数。

这样既避免启动旧结果闪烁，也不会让 Profile 环境状态限制后续 semantic 通道。

## 8. 清走、消失和再次出现

两通道不能共用一个含糊的“未检测到即清走”规则。

### 8.1 prior-only 与 fused 事件

Clean Reference 在 anchor 区域提供有效干净观察时累计 clean evidence：

- 按相邻有效观察的真实时间差累计；
- 间隔超过最大允许 gap 时重新开始；
- 默认累计 6 秒后关闭为 `clean_confirmed`；
- 环境非 `NORMAL`、地面不可见或 actor/context 遮挡时暂停，不累计清走。

### 8.2 semantic-only 事件

semantic-only 可能是启动前已经存在的垃圾，不能要求一定有 prior 变化。关闭条件为：

- 连续 semantic scan 不再命中；
- anchor 所在地面可判断且无 actor/context 遮挡；
- 按真实时间累计 absence evidence；
- 至少经历2次实际 semantic full scan 缺失；
- 默认同时达到2次缺失和累计8秒后，关闭为 `semantic_absent_confirmed`。

如果同一位置同时能获得 Clean Reference 干净证据，使用二者中更可靠、较晚满足的条件，
避免一次模型漏检导致误清走。

### 8.3 同位置再次出现

事件关闭后，同位置重新满足任何一路确认条件时必须创建新 event ID。关闭事件不得复活，
只可作为回溯记录。

## 9. Profile 环境状态的局部影响

环境状态只控制 prior 通道：

| Profile 状态 | semantic | prior | 已有 semantic-only | 已有 prior/fused |
|---|---:|---:|---:|---:|
| `NORMAL` | 运行 | 运行 | 正常更新 | 正常更新 |
| `GLOBAL_LIGHT_CHANGE` | 运行 | 暂停 | 正常更新 | prior-only 隐藏；有当前 semantic 支撑的 fused 继续显示 |
| `ENVIRONMENT_CHANGE` | 运行 | 暂停 | 正常更新 | prior-only 隐藏；有当前 semantic 支撑的 fused 继续显示 |
| 对齐失败 | 运行 | 暂停 | 正常更新 | prior-only 隐藏；有当前 semantic 支撑的 fused 继续显示 |

禁止沿用 V3.2 的“环境非 NORMAL 就清空所有事件并让整路 abstain”行为。prior-only 事件
可以暂存到有界期限，但不能在无可靠参考时继续确认、显示或判定清走。fused 事件若仍有
当前 semantic 支撑，按 semantic 通道继续显示和更新；只有 prior 支撑时按 prior-only 暂停。

prior 连续暂停超过默认120秒时，以 `profile_unavailable_timeout` 关闭事件，但不得记录为
“已清走”。Profile 恢复后仍存在的目标重新满足确认条件时获得新 event ID，确保事件内存
有界且不会永久保留不可判断状态。

全天多环境 Profile 如何让 prior 更长时间保持 `NORMAL`，由独立方案处理。

## 10. 推理调度与性能

目标设备：RTX 3060 Ti 8 GiB。默认 `analysis_fps=0.5`，每2秒一个 side tick。

推荐调度：

- 每个 tick 都运行 prior 分析；
- 每4秒运行一次 semantic 完整 ROI 分块扫描；
- 非 full-scan tick 对最多4个 prior crop 进行一次 batch 推理；
- full-scan tick 直接将其结果与当次 prior 关联，不重复运行相同位置 crop；
- semantic 通道在 prior abstaining 时仍按4秒周期运行；
- 模型调用必须按 tiles 或 crops 批量化，不得逐块串行启动一次模型调用；
- 性能不足时先把 full-scan 周期从4秒调到6秒，再减少 crop 数量；
- 不得降低主视频链 FPS 来满足 side 分析。

### 10.1 latest-wins

当前 `GroundLitterProcessClient.submit()` 在 input queue 满时丢弃新帧、保留旧帧，必须修正。

首选实现：每个 pad 一个 latest-frame mailbox，通知队列只发送 pad/version。允许的过渡实现：

1. 队列满时移除该 pad 可移除的旧输入；
2. 插入新输入；
3. 子进程收到后检查 monotonic frame timestamp；
4. 年龄超过 `max(2 × analysis_period, 4s)` 的帧直接丢弃。

多 pad 时不能为了一个 pad 的新帧任意删除其他 pad 的唯一待处理帧，因此简单全局
`get_nowait()` 不是最终正确方案。

### 10.2 性能预算

- 主链 capture/pre-encode/publish FPS ≥20，目标约25；
- side 分析不得阻塞 GStreamer；
- 普通 tick P95 <2000ms；
- full ROI scan P95 <1500ms；
- prior crop batch P95 <800ms；
- 输入帧年龄 P95 <4000ms；
- V3.3 新增 GPU 显存目标 ≤1.5GiB；
- 队列和事件内存必须有界。

## 11. API 配置契约

新增模式：

```json
"mode": "hybrid_v33"
```

`hybrid_v33` 必须同时设置 `model` 和 `profile_id`。保留现有 zone、模型、Profile、
actor/context、显示和生命周期字段，并新增：

```json
{
  "semantic_scan_interval_seconds": 4.0,
  "semantic_confirm_hits": 2,
  "semantic_hit_window": 3,
  "semantic_confirm_span_seconds": 4.0,
  "semantic_clear_seconds": 8.0,
  "semantic_clear_min_misses": 2,
  "prior_confirm_hits": 4,
  "prior_hit_window": 6,
  "prior_confirm_span_seconds": 6.0,
  "prior_suspend_expire_seconds": 120.0,
  "fused_confirm_hits": 2,
  "fused_hit_window": 4,
  "fused_confirm_span_seconds": 2.0,
  "prior_crop_maximum": 4,
  "prior_crop_expand_ratio": 4.0,
  "prior_crop_maximum_source_px": 480,
  "prior_crop_imgsz": 640
}
```

字段要求：

- hits 必须在对应 window 范围内；
- 时间字段必须为有限正数并有合理上限；
- `prior_crop_maximum` 范围 0～8；
- crop 尺寸必须在模型支持范围内；
- `mode=yolo` 和 `mode=clean_reference_v32` 行为保持不变；
- 旧请求缺少新字段时使用默认值；
- OpenAPI、payload round-trip、worker serialization 和示例请求同步更新。

不新增“prior 必须有 semantic”或“semantic 必须有 prior”的配置开关，避免调用方把架构
重新配置成单向硬闸门。

## 12. 输出与可观测性

`GroundLitterDetection.source` 必须为：

```text
hybrid_v33_semantic
hybrid_v33_prior
hybrid_v33_fused
```

Snapshot/API/metrics 至少增加：

- `semantic_raw_candidates`；
- `semantic_retained_candidates`；
- `prior_raw_candidates`；
- `prior_retained_candidates`；
- `semantic_only_active/confirmed`；
- `prior_only_active/confirmed`；
- `fused_active/confirmed`；
- `cross_source_merges`；
- `prior_environment_state`；
- `semantic_model_runs_full`；
- `semantic_model_runs_crop`；
- `input_frame_age_ms`；
- `dropped_analysis_frames`；
- `last_prior_ms`、`last_full_scan_ms`、`last_crop_batch_ms`、`last_total_ms`。

OSD 默认：

- 统一标签 `疑似垃圾`；
- 默认不显示模型材质类别和置信度；
- debug 模式可以显示 `S`、`P`、`S+P` 来源；
- 不得把模型路径、输入 URL、凭证写入日志或 metrics。

## 13. 模块边界与实施顺序

### 13.1 `ground_litter_detection.py`

- `GROUND_LITTER_MODES` 增加 `hybrid_v33`；
- 增加配置字段和验证、序列化；
- 保持 `UltralyticsGroundLitterDetector` 的 yolo 行为兼容；
- 抽出可批量处理 tiles/crops 的推理入口，避免逐 tile 调用；
- 保留现有 `GroundLitterDisplayTracker` 给 legacy yolo 模式使用，不把 V3.3 逻辑塞进去。

### 13.2 `ground_litter_v32.py`

- 不修改 V3.2 对外行为；
- 抽出一个无 V32EventMemory 副作用的单帧 prior 分析函数，返回：
  `candidates/support/valid/environment/alignment`；
- V3.2 processor 继续调用该函数并保持原回归结果；
- 修复时间证据时避免改变已冻结 fixture 语义，必要时只在 V3.3 使用新实现。

### 13.3 新建 `ground_litter_v33.py`

实现：

- 数据结构；
- semantic/prior 单帧关联；
- `V33EventMemory`；
- 独立确认规则；
- source 升级与跨源合并；
- 清走/消失/遮挡；
- snapshot 投影；
- 有界历史和 metrics。

该模块不加载模型、不管理进程，保持纯逻辑可单元测试。

### 13.4 `ground_litter_process.py`

- hybrid 模式同时构造 detector 和 Profile analyzer；
- 实现第10节调度；
- 使用主链传入 actor/context metadata，默认 `actor_model=null`；
- 修复 latest-wins；
- side 异常时清空旧框但不能退出主视频链；
- semantic 和 prior 每次分析分别捕获错误并记录 branch state；
- semantic 运行时失败时 prior 仍可按 prior-only 规则继续，snapshot 标记 semantic degraded；
- prior 运行时失败时 semantic 仍可按 semantic-only 规则继续，snapshot 标记 prior degraded；
- 模型/Profile 初始化失败属于配置错误，hybrid pad 进入 error，但不能退出主视频链。

### 13.5 API、manager、worker 和镜像

- 更新 `api.py`、`deepstream_manager.py`、`deepstream_worker.py` 的字段传递；
- Profile 根目录和模型路径继续使用安全解析；
- Dockerfile 只打包必要代码、Profile 和现有模型；
- 不改生产 compose 镜像标签，候选镜像使用新 tag；
- 生成新的 Postman/JSON 请求示例。

## 14. 测试计划

### 14.1 配置与兼容性

- `hybrid_v33` 必须要求 model 和 profile；
- 新字段范围、hits/window 关系；
- payload round-trip；
- 缺 Profile、缺模型、路径越界；
- legacy yolo 和 clean_reference_v32 回归不变。

### 14.2 双通道核心行为

必须使用确定性 fake observations 覆盖：

1. semantic 连续命中、prior 永不命中，最终输出 semantic 框；
2. prior 连续命中、semantic 永不命中，最终输出 prior 框；
3. 两路命中同一目标，只生成一个 fused 事件；
4. 两路命中不同位置，生成两个独立事件；
5. prior crop 模型无结果，不影响 prior-only 确认；
6. full scan semantic 没有 prior，仍可确认；
7. 单帧 semantic 高分误报不确认；
8. 单帧 prior 和极小连通域不创建事件；
9. 同 tick 多 tile 命中只算一次；
10. evidence source 升级不会更换 event ID；
11. 最大框数量不会造成事件内存无界；
12. 先创建的 semantic/prior pending 事件后来相遇时只保留较早 event ID。

### 14.3 环境和遮挡

- prior 环境非 NORMAL 时暂停；
- 同一时刻 semantic 仍运行并可确认；
- 环境恢复稳定前 prior 不累计；
- actor/context 重叠不确认并隐藏已确认事件；
- 遮挡期间不累计清走或 absence；
- Profile 对齐失败不导致 semantic 事件消失；
- prior 暂停超时以 unavailable 关闭，不计作清走。

### 14.4 生命周期

- prior-only 用 clean reference 证据清走；
- semantic-only 用有效 absence 时间清走；
- fused 不因一次模型漏检清走；
- 不规则采样按真实 elapsed time；
- 大 gap 重新开始连续证据；
- 清走后同位置再次出现得到新 event ID；
- 时间戳倒退、重复和陈旧帧不重复累计。

### 14.5 进程和性能

- input latest-wins；
- 多 pad 不相互误删唯一待处理帧；
- 过期帧丢弃；
- output latest-wins；
- 子进程错误清空旧框、主链继续；
- 单一 branch 运行时异常时另一 branch 继续输出，degraded 状态可观测；
- 调度器按 full scan/crop 周期调用；
- 模型真实 batch 调用次数有断言；
- GPU/CPU smoke 不允许自动下载权重。

## 15. 离线验收

### 15.1 功能验收

必须用三个确定性场景证明用户目标：

| 场景 | semantic | prior | 预期 |
|---|---:|---:|---|
| 模型可识别的瓶子/袋装垃圾 | 有 | 可有可无 | 显示 semantic 或 fused |
| 模型漏掉的小纸片 | 无 | 持续有 | 显示 prior |
| 同一垃圾两路都发现 | 有 | 有 | 只显示一个 fused 框 |

必须保留逐时间戳事件 trace，不能只给最终截图。

### 15.2 误报验收

分别报告：

- semantic-only displayed false events/hour；
- prior-only displayed false events/hour；
- fused displayed false events/hour；
- 合并后的总误报事件/hour；
- 单帧闪框数量；
- 极小候选拒绝数量；
- actor/environment 拒绝数量。

首版工程目标：已审核清洁回放中无启动闪框、无单帧显示事件，总持续误报不超过
每摄像头每4小时1个。该数字是候选版门槛，最终业务误报预算由后续现场验收决定。

### 15.3 Phase 0 的新解释

现有 Phase 0 继续作为以下回归事实：

- turhancan 不得作为 prior-only 的硬否决器；
- semantic confidence 不得冒充跨目标可比较的“垃圾概率”；
- 扫把、桶、货架、车辆局部、反光必须进入 hard-negative 回放。

它不再阻塞本方案，因为本方案允许 prior-only 独立确认，也允许 semantic 独立保留原有
能力。现有唯一真小垃圾样本不能用于声称整体准确率，只能作为管线回归样本。

## 16. 候选构建与交付物

DS4.1 应提交：

- 实现代码和逐文件变更说明；
- 新增/更新测试及全量测试结果；
- 三个核心场景的事件 trace 和可视化；
- 真实 turhancan 权重 smoke 报告，证明 full scan 和 prior crop 都实际运行；
- 清洁回放分来源误报报告；
- 性能报告：主链 FPS、各阶段 P50/P95、帧年龄、丢帧和显存；
- 候选镜像、manifest、SHA-256；
- Postman 请求示例；
- 已知限制和回滚步骤。

本任务完成后不得直接切换生产。候选镜像和离线证据先交架构审核，再决定生产 canary。

## 17. 禁止事项

- 不得要求 prior-only 获得 semantic 才能显示；
- 不得要求 semantic-only 获得 prior 才能显示；
- 不得把两路单帧原始框直接绘制到 OSD；
- 不得用同一命中阈值、同一置信度或同一清走规则处理两条通道；
- 不得因 prior 环境异常停止 semantic 通道；
- 不得把 prior crop 没有模型命中解释成“不是垃圾”；
- 不得通过放宽环境硬阈值解决全天 Profile 覆盖；
- 不得把相邻帧当作独立真实垃圾样本夸大准确率；
- 不得重新启用第二套 actor 模型作为默认配置；
- 不得逐 crop/逐 tile 串行调用模型；
- 不得让 side 分析阻塞主视频链；
- 不得写入或输出输入 URL、API key、输出凭证；
- 不得部署生产或改生产镜像标签。

## 18. 完成定义

V3.3 候选实现只有同时满足以下事实才算完成：

1. 瓶子、袋装垃圾等 semantic-only 场景不依赖 prior 就能确认和显示；
2. 模型漏检的小纸片 prior-only 场景不依赖 semantic 就能确认和显示；
3. 两路同目标只生成一个框和一个 event ID；
4. 两路各自过滤单帧、极小、遮挡和环境噪声；
5. prior Profile 不匹配时 semantic 通道仍工作；
6. 启动阶段不闪框，side 积压不导致陈旧结果长期显示；
7. 清走、遮挡和同位置再次出现按真实时间正确工作；
8. API、metrics 和事件 trace 能区分三种 evidence source；
9. legacy 模式、全量测试、compileall 和 `git diff --check` 通过；
10. 离线误报、性能、候选镜像和回滚材料完整；
11. 全天多环境 Profile 问题没有被混入或用调阈值掩盖；
12. 未经架构复核没有部署生产。

## 19. 架构自审结论

本规格在交付前按五个维度完成自审：

| 维度 | 结论 | 证据 |
|---|---|---|
| 用户目标覆盖 | 通过 | §1、§7 和 §15 分别要求 semantic-only、prior-only、fused 三种场景 |
| 逻辑闭环 | 通过 | 两路都有候选、确认、显示、遮挡、关闭和再次出现规则 |
| 代码可落地 | 通过 | 复用现有 detector 和 V3.2 单帧分析，新增纯逻辑 V33 层；逐文件顺序见 §13 |
| 测试可证伪 | 通过 | §14 明确要求“模型无 prior”和“prior 无模型”都必须真实输出，失败即不通过 |
| 性能与主链隔离 | 有条件通过 | 设计满足独立进程和 latest-wins；tiles/crops batch 的实测 P95 是候选验收硬条件 |

自审中已修正三项初稿问题：

1. Profile 失配时，有当前 semantic 支撑的 fused 事件继续显示，避免间接阻断模型通道；
2. prior crop 中未匹配原 prior 的额外框不能创建 semantic-only 事件，避免裁剪偏差；
3. semantic-only 清走同时要求真实时间和最少缺失扫描次数，避免一次漏检误清走。

仍然存在但不应伪装成已解决的限制：

- 当前真实垃圾正例很少，本规格能保证行为结构符合需求，不能凭现有数据保证最终准确率；
- persistent semantic hard negatives 可能通过时序确认，后续需要基于来源指标继续治理；
- prior 在当前 Profile 不匹配的时段会暂停，全天覆盖由独立 Profile 项目解决；
- 误报预算为候选工程门槛，最终阈值需要现场回放与 canary 决定。

这些限制不会破坏本规格的核心目标：模型通道保留常规垃圾召回，先验通道独立补充模型
漏掉的小垃圾，两路分别过滤后在事件层合并。
