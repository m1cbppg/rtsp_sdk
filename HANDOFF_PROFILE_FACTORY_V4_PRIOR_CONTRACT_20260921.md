# Profile Factory v4 prior_suitable 跨模块契约修复（2026-09-21）

本轮只处理 v4 复核提出的最后一个契约问题：自动剪枝把唯一两个 `prior_suitable=true`
的候选删掉，最终 Bank 只剩两个 `prior_suitable=false` 的 Profile，却仍被当作
可用 prior 库。没有重开 C1–C3，也没有改参考选择算法或校准匹配阈值。

契约与计划见 `docs/plans/2026-09-21-profile-factory-v4-prior-suitable-contract.md`。

## 1. 根因（v4 实测数据）

v4 逐轮删除代价（98 帧同一评分矩阵）：

| 轮 | 删除 | 环境覆盖 delta | prior 能力 |
| --- | --- | --- | --- |
| 1 | p0006 | -0.03572 | 否 |
| 2 | p0007 | -0.02381 | 否 |
| 3–9 | p0004/p0005/p0008/p0009/**p0011**/p0012/**p0013** | 0.0 | **p0011/p0013 是** |
| 10 | p0001 | -0.0119 | 否 |
| 11 | p0002 | 0.0 | 否 |

`p0011`/`p0013` 对环境匹配确实零贡献——删它们**符合当时的判据**。问题是判据里
没有 prior 这一维：剪枝只优化环境覆盖/暂停，`prior_suitable` 只是报告字段，
运行时也完全不读它。于是"环境冗余"直接等价于"可以删"，把唯一的 prior 能力删空。

## 2. 契约（已实现）

每个 Profile：

| 字段 | 语义 |
| --- | --- |
| `match_eligible` | 可参与环境匹配 |
| `prior_suitable` | 可产生 prior-only 候选 |
| `calibration_state` | `independent_matched` / `reference_self_low_support` / `no_independent_material` |
| `prior_degradation_reason` | 降级原因 |
| `valid_fraction` / `support` | 已发布字段 |

选择结果（`SelectionDecision` 新增）：

* `active_profile_id`：当前最匹配环境的 Profile；
* `prior_profile_id`：允许产生 prior 的 Profile，可为 `None`；
* `profile_match_available`：是否有可用参考；
* `prior_available`：本 tick 是否允许 prior-only 候选；
* `prior_unavailable_reason`：`NO_MATCHED_PROFILE` / `PROFILE_PRIOR_UNSUITABLE` /
  `NOT_OBSERVABLE` / `None`；
* `prior_generation`：prior 能力代际（能力变化即推进）。

**采用单 active 模型**：`prior_profile_id` 恒等于 `active_profile_id`（当且仅当它
suitable），否则为 `None`。没有引入第二个活跃参考——先验残差是相对 active 参考的
valid/threshold 定义的，换成另一个 Profile 会制造大面积残差误报；事件身份、清走
证据与切换代际都绑在 active generation 上；预算模型也只给一个参考保留保底名额。

### 三个模块怎么用

```text
publish   profile.json: calibration_state + prior_suitable（由 classify_calibration_state 推出）
loader    load_bank 校验自洽；缺字段默认拒绝，allow_legacy_profile_capabilities=True 才按 false 读
Selector  prior_available = current_hold AND observable AND active ∈ prior_suitable
adapter   prior_available=false → 不产生 prior 候选、不累计 prior/清走证据；semantic 不受影响
```

## 3. 剪枝目标（已实现）

每次尝试删除一个候选时，在**当前剩余集合**上同时计算：

* `match_coverage`（现有 `effective_fraction`）与 `match_pause_max`；
* `prior_effective_coverage`：`active` 属于 suitable 集合且结果新鲜的可判断区间占比；
* `prior_pause_max`：prior 不可用区间的最长连续时长；
* `semantic_only_intervals`：prior 不可用但仍可判断的区间（含 `reason`）。

删除必须**同时**满足：环境覆盖下降 ≤ 0.005、环境暂停增加 ≤ 5.0 s、
**prior 覆盖不下降（默认 0.0）**、**prior 暂停不增加（默认 0.0）**、
且删除后 prior-capable 集合非空（只剩一个时禁止删它）。
资源裁剪用同一套规则并再次复核。

报告新增：`match_coverage`、`prior_effective_coverage`、`prior_pause_max`、
`semantic_only_intervals`、`prior_suitable_profiles_kept`、
每轮 `metrics_before/metrics_after`、`prior_bank_available`。
候选阶段一个 suitable 都没有时：`prior_bank.available=false`、
`reason=NO_PRIOR_SUITABLE_CANDIDATE`、`semantic_only=true`，
**不宣称 prior 可用**。

## 4. 阶段 2：v4 候选集合离线重新剪枝

用 `scripts/export_profile_bank_score_matrix.py` 在服务器隔离目录导出 v4 分区的
真实评分矩阵（6 个构建文件 → 60 帧 × 13 候选；只复用已有缓存，不做完整重建），
再用 `scripts/replay_pruning_precheck.py` 跑新判据：

| 项 | 结果 |
| --- | --- |
| 候选 | 13（`p0011`、`p0013` 为 prior-capable） |
| 删除轮数 | 11 |
| 预计最终集合 | **`p0012`、`p0013`** |
| prior-capable 保留 | **`p0013`** |
| `prior_bank_available` | **true** |
| prior 有效覆盖 | 0.60 |
| 环境覆盖 | 0.88889（全库 0.85185，删除后不降） |

对照 v4 的"只剩 `p0003`/`p0010`、prior 不可用"：新判据**不会**再产生那个错误结论。
注意 `p0011` 在轮 5 仍被删除（当时 prior 覆盖还是 0），轮 8 删 `p0010` 之后
prior 覆盖才变成 0.6——留下的 `p0013` 承担了 prior 能力。这符合"不要求保留所有
prior-capable，但不能删空"。

## 5. 阶段 2：A/B/C 同帧小目标诊断

同一帧（09-19 PS 第 40 帧）、同一注入位置、同一目标尺寸（8/16/24 px 原生）、
同一配置；每组都有同帧未注入基线；A 为 v3 全 13 个参考，B 为 v4 最终
`p0003`/`p0010`，C 为修正后候选集合（`p0011`/`p0012`/`p0013`，`p0011` 为
prior-capable）。

| 组 | 原生 8px | 在线 3px | 目标位置支持像素 | 判定 |
| --- | --- | --- | --- | --- |
| A（v3 全 13） | 命中，5 个参考有支持 | 命中（p0011） | 61 → 6 | 原生 SURVIVED / 在线 REFERENCE_NOT_SELECTED |
| B（v4 最终） | **0 命中** | **0 命中** | **0** | THRESHOLD_BELOW_NOISE_FLOOR |
| C（修正集合） | 命中（p0011） | 命中（p0011） | 52 → 6 | REFERENCE_NOT_SELECTED |

结论（只依据本表）：

1. **v4 Bank 确实丢了目标信号**：B 组在原生与在线画布、8/16/24 px 全部 0 命中、
   支持像素为 0，丢失环节是"残差未过阈值"——两个 Profile 的噪声都来自
   `reference_self`，阈值不具备跨环境余量。
2. 目标信号是**存在**的：A 组与 C 组在同一帧、同一位置都能在多个参考上得到
   支持像素（原生 37–61，在线 6+）。
3. A 组的 active 参考是 `p0004`（不 suitable），而实际承载目标是 `p0011`
   （suitable，原生支持 52、`enter_eligible=true`）。C 组里 `p0011` 同样承载目标、
   并且是 prior-capable，但预热后的 active 是 `p0012`。所以"参考选择/切换"仍然是
   一个真实的次级损失环节——**本轮不修改参考选择算法**，只记录证据。
4. semantic 路径不受影响：诊断里 `prior_available=false` 只暂停 prior 通道，
   候选框统计仍来自共享 adapter 的 semantic/残差支持。

## 6. 运行时门禁（已实现 + 测试）

* `load_bank` 校验能力字段：缺字段且未显式开兼容开关 → 拒绝加载；
  兼容模式下保守视为 `prior_suitable=false` 并标注 `capability_source`。
* `ProfileSelector(..., prior_suitable_profile_ids=...)`：
  * unsuitable active → `prior_available=false`、
    `reason=PROFILE_PRIOR_UNSUITABLE`、`prior_profile_id=None`；
  * 没有 active → `NO_MATCHED_PROFILE`（与"匹配但 prior 不可靠"区分）；
  * suitable → unsuitable 立即暂停，且 `prior_generation` 推进（不继承旧证据）；
  * 老调用不传能力集合时保持原行为（避免静默改变历史对照）。
* `prior_summary()`：prior 覆盖、prior 暂停、semantic-only 区间全部由**真实决策
  记录**推出，而不是事后推断。

## 7. 测试

`tests/test_ground_litter_profile_c1c4.py` 新增 10 项：

* `PriorCapabilityContractTests`：唯一适合 prior 的参考不可删；两个完全重复且都
  适合 prior 可删一个且 prior 覆盖不变；match-only 参考的删除必须同时满足两类
  阈值；全部 unsuitable → `NO_PRIOR_SUITABLE_CANDIDATE` + `semantic_only`；
  资源裁剪不删空 prior；报告同时暴露两类能力。
* `PriorRuntimeGateTests`：unsuitable active 不输出 prior；
  suitable 允许；`NO_MATCHED_PROFILE` 与 `PROFILE_PRIOR_UNSUITABLE` 可区分；
  能力切换推进 prior 代际；不传能力集合保持旧行为。
* `LegacyBankCapabilityTests`：历史 Bank 默认拒绝，显式开关下按 false 读取。

本地：**785 passed + 4 pre-existing torch 失败**；相关 153 项
（c1c4 37 + repairs 38 + factory 78）全绿。

## 8. 是否需要重建 / 实际重跑了哪些阶段

需要：`prior_suitable` 现在参与发布与剪枝，v4 的产物（`profile.json` 能力字段、
最终 Profile 集合）都不代表修复后的行为。

**只重跑一次完整工厂**，分区不变（构建 09-14~09-17 / 校准 09-18），
版本号 v5，不覆盖 v1–v4。中间产物无法安全复用（合成阶段释放了原始 PS，
且 v4 的 `work_v4` 缓存已被回收），因此走完整流水线；`work_v5` 复用 v2 的
清单缓存避免重复查询。

## 8.1 v5 重建实测（阶段 3）

`V5_EXIT=0`（2026-09-21T13:54:59+08:00），manifest
`0de0093edba79a3c799d74c381ee935e60fcddbd55812303c83c4f35a81ca6bf`，
产物 `out/bank/camera_01030/v5`，v1–v4 未覆盖。

| 项 | v4 | v5 |
| --- | --- | --- |
| 候选 → 终选 | 13 → 2 | 10 → 10 |
| 删除轮数 | 11 | **0** |
| `prior_suitable` 候选 | 2（p0011、p0013） | **0** |
| `prior_bank.available` | （无该字段） | **false**，`reason=NO_PRIOR_SUITABLE_CANDIDATE` |
| `semantic_only` | 未声明 | **true** |
| prior 有效覆盖 | — | 0.0，`prior_pause_max=3690.651` |
| semantic-only 区间 | — | 2 段：`NO_MATCHED_PROFILE` 86.8 s、`PROFILE_PRIOR_UNSUITABLE` 3603.8 s |
| 环境覆盖 | 0.78571 | 0.7619（pause 260.52） |

**这个结果是正确的行为，不是修复失败**：v5 的候选阶段确实一个 prior-capable 都
没有，契约要求"拒绝宣称 prior 可用"，它就输出了 `semantic_only=true`。
它同时证明门禁是真生效的——运行时只有 `PROFILE_PRIOR_UNSUITABLE` 一种暂停原因，
不会有任何 unsuitable Profile 产生 prior-only 候选。

### 为什么 v5 一个 suitable 都没有（新发现，需后续处理）

v4 与 v5 用**同一分区、同一代码、同一 `--per-day-hours/--max-files`**，但：

| 项 | v4 | v5 |
| --- | --- | --- |
| 采样样本 / 时间块 | 56 / 56 | 56 / 56 |
| 采样清单 sha256 | `0d448020…` | `7f84b437…` |
| 分组数（centers/groups） | 18 / 18 | **16 / 16** |
| `requested_radius` | 12.993 | **15.094** |
| 最大组块数 | 8 | **16** |
| 校准观测分配（assigned/unassigned） | 8 / 8 | **4 / 12** |

即两侧都采到 56 个时间块、每天块数完全一致（8/16/16/16），但落到分组里的
**块集合不同**，导致组半径、以及"独立校准观测能否匹配到该组"整体改变。
v5 只有 4 个校准观测匹配上、且每组最多 1 块，因此 `calibration_sufficient`
全为 false。

要点：

* 这不是本轮门禁代码的问题——把同一份 v4 评分矩阵喂给新判据，结果是
  `{p0012, p0013}` 且 `prior_bank_available=true`（第 4 节）；合成回归也全部
  通过。v5 的 0 是**素材/分组层面**的结果。
* 它说明"外观匹配的独立校准"是**素材受限**的：`clusters=6` 而组有 16 个，
  独立校准观测只能覆盖其中一部分外观簇。
* 它同时暴露一个此前没被记录的测量问题：**分组对采样块集合敏感**，两次同参数
  重建会得到不同的候选集合与不同的 prior 能力。在补素材或固定分组之前，
  "重建一次就能拿到 prior-capable Bank" 是不可保证的。

### 建议的下一步（不在本轮范围）

1. 先让采样/分组可复现（记录并比对样本清单、或固定抽帧偏移），
   否则任何"再重建一次看看"的实验都无法比较；
2. 针对没有被独立校准覆盖的外观簇定向补采（只补校准日素材，不需要七天）；
3. 补齐后再重建一次，用 `prior_bank.available` 与
   `prior_suitable_profiles_kept` 验收；若仍为 false，应如实发布 semantic-only
   Bank，而不是放宽外观匹配阈值。

## 9. 留给方案二

* 在线验收先前版本优先级、切换时延与 active prior 下的算力；
* 参考选择/切换的改进（本文档第 5 节第 3 点给出证据，本轮未改算法）；
* 把 `prior_suitable` 门禁接到生产 hybrid 调度的 bank 路径
  （`CleanReferenceProfileV32` 侧字段默认为 `True`，未改变现有 V3.2/V3.3 行为）；
* 事件 memory 完整生命周期。

## 10. 可复现命令

```bash
# 本地相关回归
.venv-profile/bin/python -m pytest tests/test_ground_litter_profile_c1c4.py \
    tests/test_ground_litter_profile_repairs.py \
    tests/test_ground_litter_profile_factory.py -q

# 离线重新剪枝预演（用导出的评分矩阵，不需要素材）
.venv-profile/bin/python scripts/replay_pruning_precheck.py \
  output/c1c4_v3fix_20260921/replay_score_matrix.json

# 服务器：导出 v4 候选集合的评分矩阵（只读，复用缓存）
cd ~/profile-factory
./venv/bin/python scripts/export_profile_bank_score_matrix.py

# 服务器：A/B/C 同帧诊断（每次一个组）
./venv/bin/python scripts/diagnose_ground_litter_small_target.py \
  --bank-root out/bank --bank-id camera_01030 --version v3 \
  --media diag_media --frame-index 40 --native-size-px 8,16,24 \
  --online-size 960x540 --allow-legacy-capabilities \
  --profiles p0003,p0010 --label B_v4_final2 --output out/c1c4_diag_abc/armB
```

## 11. 耗时口径

| 类别 | 实际 |
| --- | --- |
| 真实 CPU/下载 | 评分矩阵导出 464 s 帧收集 + 31 s 矩阵；A/B/C 三组各约 15–20 min；v5 完整重建约 29 min |
| 与运行重叠的等待 | A 与 C 并行执行，重叠部分约 12 min（未额外增加墙钟） |
| 作业完成后的轮询延迟 | 约 3 min（本轮用后台 job + 状态查询，没有长同步 sleep） |
| 失败重跑 | 约 25 min：评分矩阵导出第一次用 v4 Bank（只有 2 个 Profile）导出无效；`pkill -f <文件名>` 误杀自己的 ssh 会话导致两次上传/启动静默失败 |

**教训**：① `pkill -f <模式>` 的 ssh 命令行本身含有该模式，会杀掉自己；
② 后台任务必须用 `run_in_background` 的 job 通道，而不是 `setsid ... &`；
③ 先确认导出脚本的候选来源（v4 最终只有 2 个 Profile）再跑，能省一次 8 分钟下载。
