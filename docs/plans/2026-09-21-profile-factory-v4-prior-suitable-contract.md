# Profile Factory v4 跨模块契约修复计划（prior_suitable）

状态：**待架构复核**。阶段 1/2 在本地完成后再申请服务器复算；复核通过前不完整重建。

## 0. 问题陈述（已确认）

v4 最终只发布 `p0003`、`p0010`，两者 `prior_suitable=false`；而唯一两个
`prior_suitable=true` 的候选 `p0011`（g015）、`p0013`（g018）被剪枝删除。
运行时目前完全不消费 `prior_suitable`，所以字段只是报告文本。

v4 实测的删除代价（同一个 98 帧评分矩阵、逐次删除）：

| 轮次 | 删除 | 环境覆盖 delta | 暂停 delta |
| --- | --- | --- | --- |
| 1 | p0006 | **-0.03572** | 0 |
| 2 | p0007 | **-0.02381** | 0 |
| 3–9 | p0004/p0005/p0008/p0009/**p0011**/p0012/**p0013** | 0 | 0 |
| 10 | p0001 | -0.0119 | 0 |
| 11 | p0002 | 0 | 0 |

也就是说：`p0011`/`p0013` 对**环境匹配**确实零贡献，删掉它们符合当前判据；
但它们携带唯一的 prior 能力，而剪枝目标里没有这一项。这不是 C1 回归，
是剪枝目标函数缺了一个维度。

## 1. 契约（先冻结）

### 1.1 Profile 发布字段（`profile.json` + `bank.json` 索引）

| 字段 | 取值 | 语义 |
| --- | --- | --- |
| `match_eligible` | bool | 可参与环境匹配（当前所有已发布 Profile 都是 true；保留字段以便未来排除坏参考） |
| `prior_suitable` | bool | 可产生 prior-only 候选（校准充分且外观匹配） |
| `calibration_state` | enum | `independent_matched` / `reference_self_low_support` / `no_independent_material` |
| `degradation_reason` | str\|null | 降级原因 |
| `valid_fraction` | float | 已发布的 `valid_fraction_of_roi` |
| `support` | dict | 已发布的 days/distinct_files/low_support/selected_blocks |

`calibration_state` 由 `prior_suitable` 与来源唯一确定：

* `independent_matched` ⟺ `calibration_sufficient=true` ⟹ `prior_suitable=true`；
* `reference_self_low_support`：有校准素材但本组外观不匹配 / 只有一块；
* `no_independent_material`：校准日完全没有可用观测。

### 1.2 Bank 加载

* `load_bank(require_calibration=True)` 校验每个 Profile 的 `prior_suitable` 与
  `calibration_state` 存在且自洽（`prior_suitable=true` 必须对应
  `calibration_state=independent_matched`）；
* `load_bank(..., allow_legacy_profile_capabilities=True)` 才允许读取缺失字段的
  历史 Bank（v3 及更早）；缺失时**保守视为 `prior_suitable=false`**，
  并在 `ProfileRecord` 上标注 `capability_source="legacy_conservative"`；
* 缺字段且没开兼容开关 → 直接拒绝加载（`BankError`）。

### 1.3 选择结果（`SelectionDecision` 新增字段）

保持"一个 active Profile，同时决定它能否提供 prior"的简单模型（用户的优先方案）：

| 字段 | 语义 |
| --- | --- |
| `active_profile_id` | 当前最匹配环境的 Profile（等价现有 `selected_profile_id`） |
| `prior_profile_id` | 允许产生 prior 的 Profile；unsuitable 时为 `None` |
| `profile_match_available` | 是否有可用参考（distinguishes 无匹配 vs 匹配但 prior 不可靠） |
| `prior_available` | 本 tick 是否允许 prior-only 候选 |
| `prior_unavailable_reason` | `PROFILE_PRIOR_UNSUITABLE` / `NO_MATCHED_PROFILE` / `NOT_OBSERVABLE` / `null` |

**不引入第二个活跃参考。** 理由：先验残差是相对 active 参考的 valid/threshold
定义的，若 prior 用另一个 Profile，就会出现"参考 A 的阈值解释参考 B 的画面"，
大面积残差会直接变成误报；事件身份、清走证据与切换代际都绑在 active 参考的
generation 上，双活跃参考需要同时重做这三套账；当前预算模型（`top_k` +
`max_small_matches_per_tick`）也只按一个参考保留保底名额。因此
`prior_profile_id` 恒等于 `active_profile_id`（当且仅当它 suitable），
否则为 `None`。

### 1.4 三个模块如何共同消费

```text
建库（publish）        bank.json/profiles/*.json: calibration_state + prior_suitable
      │
      ▼
Bank loader           校验字段；legacy 缺字段 → 保守 false（需显式开关）
      │
      ▼
ProfileSelector       持有 prior_suitable 集合；
                      observe() 计算 prior_available / prior_profile_id /
                      prior_unavailable_reason；commit 切换时清空 prior 证据
      │
      ▼
prior adapter         prior_available=false → 不产生 prior-only 候选、
（evaluator / 生产     不累计 prior 命中或清走证据；semantic 通道完全不受影响
  hybrid 调度）        prior_available=true → 行为与现在一致
```

## 2. 剪枝目标（阶段 1 实现）

保留现有公平评分矩阵、同帧同时间轴、逐次删除、终选/资源裁剪复核。
每次尝试删除一个候选时，在**当前剩余集合**上同时计算：

| 指标 | 定义 |
| --- | --- |
| `match_coverage` | 现有 `effective_fraction`（active 参考合格且可判断的区间占比） |
| `match_pause_max` | 现有 `pause_max` |
| `prior_effective_coverage` | `active_profile_id` 属于 suitable 集合且结果新鲜的可判断区间占比 |
| `prior_pause_max` | prior 不可用区间的最长连续时长 |
| `semantic_only_intervals` | prior 不可用但仍有可判断观测的区间数与总时长 |

删除必须**同时**满足（当前规则 + prior 规则）：

1. `match_coverage` 下降 ≤ `min_coverage_delta`（0.005，不变）；
2. `match_pause_max` 增加 ≤ `max_pause_delta`（5.0，不变）；
3. `prior_effective_coverage` 下降 ≤ `min_prior_coverage_delta`
   （新参数，默认 0.0 表示"不允许 prior 覆盖下降"）；
4. `prior_pause_max` 增加 ≤ `max_prior_pause_delta`（新参数，默认 0.0）；
5. 删除后 suitable 集合非空；若当前 suitable 集合只剩 1 个，直接禁止删除它。

候选阶段如果**一个 suitable 都没有**：不做"假装通过"的发布，
`report["prior_bank"] = {"available": false, "reason": "NO_PRIOR_SUITABLE_CANDIDATE"}`
并在顶层给出 `semantic_only` 结论；Bank 仍可发布用于环境匹配。

资源裁剪（`_trim_to_resource_limit`）用同一套双目标规则，裁剪后再次复核两类覆盖。

### 新增报告

`report["n_selection"]["match_coverage"]`、`prior_effective_coverage`、
`prior_pause_max`、`semantic_only_intervals`、`prior_suitable_profiles_kept`、
每轮 `deletion_order[*].metrics_before/after`、最终集合完整复核
（`final_set_verification` 同时给两类覆盖）。

## 3. 运行时门禁（阶段 1 实现）

* `ProfileSelector.__init__(..., prior_suitable_profile_ids=...)`；
* `prior_available = prior_allowed AND observable AND active ∈ suitable`；
* `prior_unavailable_reason` 优先级：
  `NO_MATCHED_PROFILE`（没有 active）→ `PROFILE_PRIOR_UNSUITABLE`（有 active 但不 suitable）
  → `NOT_OBSERVABLE`；
* 提交切换时（`commit`）重置 prior 证据：沿用现有 generation 机制，
  并把 `prior_evidence_generation` 一并推进，保证 unsuitable → suitable 不继承旧证据；
* offline evaluator 的 `_run_tick` 直接消费新字段：
  `prior_allowed` 字段保留兼容，但候选输出以 `prior_available` 为准；
  `candidate_boxes_detail` 增加 `prior_eligible`，语义候选单独计数；
* 生产 hybrid 调度（`ground_litter_process.py`）里 `prior_available` 追加
  `state.prior_suitable` 条件；该字段来自 `CleanReferenceProfileV32`，
  缺省 `True` 以不改变现有 V3.2/V3.3 行为（Bank 路径显式写入）。

## 4. 测试（阶段 1）

`tests/test_ground_litter_profile_c1c4.py` 新增/加强：

1. 环境覆盖冗余但唯一 `prior_suitable=true` 的参考**不能**被删除；
2. 两个完全重复且都 suitable 的参考可删一个（prior 覆盖不变）；
3. match-only 参考对环境覆盖有贡献时可保留；
4. 所有候选都不 suitable → `prior_bank.available=false`，
   报告明确 `NO_PRIOR_SUITABLE_CANDIDATE`，不宣称 prior 可用；
5. 资源裁剪不删除全部 prior-capable；
6. 用 v4 的 13 候选**真实指标**做离线预演（见阶段 2），断言不会重演
   "只剩两个 unsuitable 且宣称 prior 可用"；
7. Selector：unsuitable active → `prior_available=false`、
   `prior_unavailable_reason=PROFILE_PRIOR_UNSUITABLE`、不产生 prior 候选；
8. semantic 候选在 unsuitable 时仍然保留；
9. unsuitable → suitable 切换后重新计 prior 证据；suitable → unsuitable 立即暂停；
10. 字段缺失的历史 Bank 默认拒绝加载，显式兼容开关下按 `prior_suitable=false` 读取。

## 5. 阶段 2：离线预演 + A/B/C 诊断

* 用 v4 已保存的 score matrix 指标（`v4_factory_report.json` 的
  `deletion_order`、逐轮 LOO、环境覆盖）离线重跑新的双目标剪枝，
  输出预计最终集合与两类覆盖；这一步不需要服务器、不需要重新下载。
* 同帧诊断在 **v4 候选集合**上做 A/B/C：
  - A：v3 全 13 参考（已有结果，直接复用保存的 JSON）；
  - B：v4 最终 `p0003/p0010`；
  - C：修正后包含可靠 prior Profile 的模拟集合（至少含 `p0011`/`p0013`）；
  每组记录 active、`prior_suitable`、目标位置 valid、支持像素、是否成框、
  `prior_available`、是否属于允许输出的 prior 候选，以及 semantic 路径不受影响。

## 6. 阶段 3（复核通过后）

只同步必要代码、核对 SHA；优先复用 `work_v4` 的可信缓存；若只改选择/发布逻辑，
优先只重跑最后阶段；确实无法复用时才完整重建一次，版本号 `v5`，不覆盖 v1–v4。

## 7. 耗时口径（交付要求）

分四类记账：真实 CPU/下载运行、与运行重叠的等待、作业完成后的轮询延迟、
失败重跑产生的额外时间。长任务一律后台执行 + 状态查询，不用同步 `sleep`。
