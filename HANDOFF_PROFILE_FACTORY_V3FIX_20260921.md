# Profile Factory v3 复核后定向修复与小目标诊断（2026-09-21）

本轮只处理 v3 复核列出的三项缺陷 + 一项小目标诊断，保留上一轮的公平评分矩阵、
冻结资产、双口径命中与同帧基线修复；未实施方案二生产接入，未重开 R1–R11。

## 0. 版本与哈希

| 文件 | 本地 SHA-256（前16） |
| --- | --- |
| `scripts/build_ground_litter_profile_bank.py` | `c5bcc9f1576de3d1` |
| `rtsp_annotator/ground_litter_profile_background.py` | `48752b2ce7e59089` |
| `rtsp_annotator/ground_litter_profile_analysis.py` | `5a07d5bc68208d2f` |
| `scripts/diagnose_ground_litter_small_target.py` | `55cc3ff66ee5963a` |
| `tests/test_ground_litter_profile_c1c4.py` | `09b69616e21583c0` |

服务器 `~/profile-factory` 与本地逐字节一致（逐文件 SHA-256 比对）。
相关测试：本地 **773 passed + 4 pre-existing torch 失败**；服务器
`test_ground_litter_profile_{c1c4,repairs,factory}.py` **141 passed**。

## 1. C1：剪枝错误提前退出

### 根因（已复现）

`prune_profiles_conservatively` 调 `project_selection_timeline(stop_below_fraction=...)`，
而回放开头本来就有"恢复窗口"：前几个 tick 的 `effective_fraction` 天然接近 0。
提前退出把每个删减对照都在启动期截断，于是

* 所有候选的 `effective_fraction_delta` 都≈0、`pause_max_delta` 都≈0；
* `early_stopped=True` 又让 `break` 立刻触发；
* 结果 `deletion_order=[]`、`removed=[]`，两个完全相同的参考一个都删不掉。

v3 真实报告正是这样：`candidates 13 → selected 13`、`deletion rounds 0`、
`pruning.leave_one_out = {}`。

### 修复

`scripts/build_ground_litter_profile_bank.py`

* `project_selection_timeline` 删除 `stop_below_fraction` 参数与早退分支，
  一律跑满整条时间轴；
* `prune_profiles_conservatively` 的每轮删减对照改为完整执行，
  删减记录里用 `frames_evaluated` 记录实际跑满的帧数；
* 保留同帧同时间轴、评分矩阵复用、逐个删除、终选/资源裁剪复核。
  **没有**引入任何替代性早退条件。

### 回归证据

`tests/test_ground_litter_profile_c1c4.py::C1FairComparisonTests`：

* `test_identical_reference_pair_deletes_at_most_one`：两个完全相同的参考，
  **恰好留下 1 个**；`removed` 非空且 reason 为 `DYNAMICALLY_REDUNDANT`；
  `deletion_order` 非空；保留者的完整回放覆盖与暂停与删除前**完全一致**
  （`effective_fraction` 0.89474、`pause_max` 相等）。
* `test_conservative_pruning_removes_one_at_a_time_and_reverifies`：
  p1≡p2 冗余 + p3 唯一外观 → 删的是 {p1,p2} 之一，p3 一定保留，
  `removed_ids + kept_ids == {p1,p2,p3}`，每轮 `chosen_reason` 为
  `min_loo_cost_below_threshold`。
* `test_final_and_trimmed_sets_are_reverified_on_shared_matrix` 不变。

### 真实素材（v4 重建，105 帧重放，13 候选）

| 指标 | v3（含早退 bug） | v4（修复后） |
| --- | --- | --- |
| 候选 → 终选 | 13 → 13 | 13 → **2** |
| 删除轮数 | 0 | **11**（`deletion_order` 全部留证） |
| 删除顺序 | — | p0006, p0007, p0004, p0005, p0008, p0009, p0011, p0012, p0013, p0001, p0002 |
| 全库指标 | 0.78571 / pause 130.256 | 0.78571 / pause 86.84 |
| 终选集合复核 | — | 0.85714 / pause 86.84 / switch 7（105 帧同帧复核） |
| 任意单候选 | 0.37778 / pause 824.958 | 0.26191 / pause 1910.467 |

结论直接：早退确实把**所有**删减对照都误判成"不可删"，修复后真实数据上发生了
11 轮删除，而且终选集合的复核指标**不差于**全库（覆盖 0.85714 > 0.78571、
暂停 86.84 < 130.256、切换 10 → 7）。这符合验收边界"不要求减少真实 Bank 的 N"，
但确实减少了：减少来自自动化判据，不是预设目标。**不要求真实 Bank 一定有可删
候选**——这一轮恰好有，因为 13 个候选里有大量互相冗余的参考。

## 2. C2：从"阶段结束释放"改为"逐文件消费后释放"

### 根因（已复现）

上一轮只做到"排除盲测日 + 失败分类 + 阶段结束统一释放"。合成/包络仍会在阶段
开始时对全部 `needed_file_ids` 逐个重拉：4 个文件、缓存只够 1 个时，
只有第一个成功，其余三个被记 `BUDGET_REJECTED`——那是**实现没释放**，
不是素材不可达。

### 修复

新增 `FileMaterializer`（`scripts/build_ground_litter_profile_bank.py`）：

* 唯一入口 `need(item)`：命中缓存/本地复用/远端下载，失败按
  `NO_SOURCE_TO_REPULL` / `BUDGET_REJECTED` / `URL_REFRESH_FAILED` /
  `NETWORK_FAILURE` / `MATERIAL_INVALID` 分类；
* `release(item)`：消费完立刻归还临时 PS；
* 背压**不是失败**：先 `wait_for_capacity`（内部会先尝试驱逐可重拉的
  EVICTABLE 条目），仍放不下才报告；只有"单文件本身就超过整个原始缓存配额"
  才立即失败（`single_file_exceeds_raw_cache_budget`），不靠等待超时表达同一事实。

接线：

* `stage_composite_and_noise`：每个文件 `need` → 用该文件的高清帧 →
  整组处理完 `release`；跳过组、异常路径也释放；
* `fit_frozen_envelopes`：逐文件 `need` → 打分 → `release`，并记录
  `calibration_material.per_file`（planned/consumed/state/frames）；
* `_collect_replay_frames`：逐文件 `need` → 抽帧 → 释放，删除原来"阶段结束
  统一释放"的路径；回放自己按需重取合成阶段释放掉的构建 PS；
* 删除已无用的 `_ensure_remote_entries` / `_release_materialized`。

### 回归证据

`tests/test_ground_litter_profile_c1c4.py::C2BoundedPipelineTests`：

* `test_materializer_consumes_and_releases_one_file_at_a_time`：
  配额只够约 1 个文件（`budget = 1.5×单文件真实字节`），4 个文件
  `need`+`release` 全部成功：`downloaded=4`、`failure_kinds={}`、`released=4`、
  `peak_raw_bytes ≤ budget`。
* `test_backpressure_drives_consumption_instead_of_failure`：
  配额只够约 2 个文件，4 个文件全部被消费，`released=4`，
  `peak_raw_bytes ≤ budget`。
* `test_single_file_above_whole_budget_is_reported_immediately`：
  单文件 > 整配额时 **< 5 秒**返回 `BUDGET_REJECTED`，不长时间等待。
* `test_remote_source_failure_is_classified_not_material_quality`：
  URL 刷新失败只记 `URL_REFRESH_FAILED`，不混成 `MATERIAL_INVALID`。
* `test_whole_factory_chain_consumes_every_needed_file_above_quota`：
  **真实受管缓存 + 假远端下载器 + 完整工厂编排**；8 个文件、配额
  只够约 1 个（`budget = 1.5×单文件真实字节`，`budget < 2×单文件`）；
  断言：
  - 合成阶段 `need == reused_cache + reused_local + downloaded`、
    `failure_kinds == {}`；
  - 回放阶段 `consumed_files == planned_files`、`released > 0`、
    `failure_kinds == {}`；
  - 校准阶段 `planned=1`、`consumed=1`；
  - 盲测日文件既不在 `needed_file_ids` 里、也从未出现在 `downloader.downloaded`；
  - 实际 `peak_raw_bytes ≤ budget`、`peak_work_bytes ≤ work_budget`、
    结束时 `final_raw_bytes == 0`、`space.budget.raw_bytes == 0`（不是只看记账）。
* `test_collect_replay_frames_consumes_each_file_once`：
  3 个文件各下载一次、各释放一次，`managed_bytes()==0`。

## 3. C3：校准必须适配参考，并共享实际补偿路径

### 根因

1. **外观不匹配的素材仍在学容差**：v3 报告 11/13 参考外观匹配失败，却仍用
   跨外观校准日素材估噪，只把 `degradation_reason` 写进报告。来源独立 ≠ 适合
   校准这个参考——跨环境残差会把门槛抬高。
2. **残差不一致**：`estimate_noise` 直接 `residual_maps(原始参考, 帧)`，而在线
   `score_profile` 先用共同拟合区估受限全局增益/偏移、把参考归一化到当前帧，
   再算残差。同一像素的亮度残差中位数一个是补偿前的 2.31、一个是补偿后的 0.50，
   学到的容差和运行时用的门槛不是同一个量。

### 修复

`rtsp_annotator/ground_litter_profile_background.py`

* 新增 `valid=` 参数（缺省全有效）；
* 显式复用运行时的补偿路径：`robust_color_compensation` →
  `apply_compensation` → `residual_maps`；拟合区取"有效 ∩ 各观测掩膜"的并集
  最大者、参考帧取拟合区最大的那一帧（同一地点几分钟光照稳定，逐帧重估只会把
  噪声混进系数）；诊断写入 `noise.diagnostics.shared_compensation` 与
  `residual_path`。

`scripts/build_ground_litter_profile_bank.py`

* 新增 `classify_calibration_state(...)`：把三个状态判成实际行为——
  `source_independent` / `appearance_matched` / `calibration_sufficient`；
  只有"匹配到本组且 ≥2 块独立观测"才允许 `source=calibration_day`；
* 外观不匹配时**不再**找跨组素材凑数，直接退回参考自身观测 + 基础阈值，
  `note=no_appearance_match_low_support`、`prior_suitable=false`；
* 每个 `profile.json.noise_calibration` 同时写
  `source_independent` / `appearance_matched` / `calibration_sufficient` /
  `independent_of_reference` / `prior_suitable` / `degradation_reason` /
  `matched_blocks`。

### 回归证据

* `test_offline_and_online_residuals_match_after_shared_compensation`：
  带全局光照偏移的夹具；未补偿的离线亮度残差中位数与补偿后差异显著，
  修复后**逐元素相等**（`np.testing.assert_allclose(atol=1e-3)`），
  同时校验在线 `score["compensation"]` 的 gains/biases 与离线一致。
* `test_unmatched_calibration_cannot_change_normal_noise_thresholds`：
  暖色夹具确实能把残差中心抬高 >3×；`classify_calibration_state` 在外观
  不匹配时返回 `source=reference_self`、`source_independent=True`、
  `appearance_matched=False`、`calibration_sufficient=False`、`prior_suitable=False`。
* 同一个 helper 的正向分支：匹配到 ≥2 块时才 `source=calibration_day` 且
  `prior_suitable=True`；只有 1 块匹配 → `single_matched_observation`（仍不够）。
* `test_low_support_profiles_keep_base_thresholds` 校验实际阈值退化为基础值。

低支持的**实际行为**：外观不匹配的参考用参考自身观测估噪 → 阈值来自参考自身
分布而非跨环境分布，且 `prior_suitable=false` 会在 `profile.json` 与实际
`source=reference_self` 上同时体现。

## 4. 小目标诊断：同帧对照与损失定位

新增 `scripts/diagnose_ground_litter_small_target.py`（只读，不改任何阈值）：

* 固定源帧、注入位置、目标尺寸、Bank 与配置；每个对照都有**同帧未注入**基线；
* A 用 `build_prior_context_at(bank, pid, 原生尺寸)` 在**原生 2560×1440** 上
  真正执行共享 adapter（不是注入后缩小冒充）；B 用在线 960×540；
* 同一物理位置按原生坐标映射到各画布；注入边长按同一 native size 缩放；
* C 用**真实共享 Selector**（预热后在注入帧上取决策：`effective_profile_id`、
  `prior_allowed`、`status`、`reason`）；D 是离线逐参考比较的最佳候选，仅作上界；
* 在目标位置比较注入/未注入差异（`matched_candidates_at_target`），
  并分开报告搜索候选、生效参考候选与允许输出的候选；
* 逐阶段判定：`FRAME_QUALITY` / `ROI_OR_VALID` / `RESIDUAL_BELOW_THRESHOLD` /
  `THRESHOLD_BELOW_NOISE_FLOOR` / `MORPHOLOGY_EMPTY` / `SIZE_FILTERED` /
  `REFERENCE_NOT_SELECTED` / `PRIOR_NOT_ALLOWED` / `SURVIVED`。

### 真实素材结果（v3 Bank，09-19 第 40 帧，13 个参考）

| 对照 | 原生尺寸 | 画布边长 | 未注入基线候选(13 参考合计) | 注入后 | 最佳参考命中 | 有效支持像素 | 判定 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A 原生 adapter | 8px | 8px | 1736 | 1720 | p0005 | 61 | SURVIVED |
| B 在线 960×540 | 8px | **3px** | 108 | 109 | p0011 | 6 | REFERENCE_NOT_SELECTED |
| A 原生 adapter | 16px | 16px | 1736 | 1726 | p0011 | 60 | REFERENCE_NOT_SELECTED |
| B 在线 960×540 | 16px | 6px | 108 | 113 | p0009 | 36 | SURVIVED |
| A 原生 adapter | 24px | 24px | 1736 | 1734 | p0009 | 37 | REFERENCE_NOT_SELECTED |
| B 在线 960×540 | 24px | 9px | 108 | 110 | p0011 | 72 | REFERENCE_NOT_SELECTED |

关键观察（真实调用链，非字段存在性）：

1. **目标在残差/阈值/形态学/尺寸过滤各步都活下来了**：8px 原生目标在原生画布上
   产生 61 个支持像素、在 5 个参考里都命中；缩到 3px 后仍有 6 个支持像素、
   1 个参考命中。因此主要损失**不是**"被噪声阈值或形态学吃掉"。
2. **运行时生效参考与命中参考不一致**：预热后的真实 Selector 稳定停在
   `p0004`，而命中目标的是 `p0005`/`p0009`/`p0011`。判定因此是
   `REFERENCE_NOT_SELECTED`——损失在**参考选择/切换**这一步。
3. **缩放仍然有贡献**：原生 8px 有 5 个参考能命中，缩到 3px 只剩 1 个，
   余量被压缩；但即使 24px（9px 在线）也仍然是"选错参考"。
4. 未注入基线本身有 1736（原生）/108（在线）个候选，说明这份夜间素材噪声很高，
   "命中"必须配成对基线看，不能单独当识别能力证据。

### 结论：下一步先修哪里

**优先修参考选择/切换，而不是重采七天，也不是先调阈值。** 具体需要的是：
让"目标位置有支持"能影响候选参考的挑战/切换判断（当前挑战只看全局提升比例），
或者在候选参考集合内先做一次目标位置支持度的比较。缩放损失是第二位的
（在线 3px 余量很小），噪声校准已在第 3 节修好但不会改变"选错参考"这个主因。
补素材只对"某个外观簇完全没有可匹配参考"有意义——本轮诊断里命中参考是存在的，
所以不是缺素材。

## 5. 是否需要重建 Bank

* C1 只影响**定稿阶段**（剪枝决策），不改变已发布的参考/阈值；
* C2 不改变产物，只改变 IO 与峰值；
* C3 **改变产物**：外观不匹配的参考现在用参考自身观测估噪，
  已发布的 v3 `noise.npz` 与 `profile.json` 不再代表修复后的行为。

因此需要一次重建来发布修复后的资产。**只重跑一次完整工厂即可**，不需要补采七天、
不需要改分区（构建 09-14~09-17 / 校准 09-18 / 独立 09-19~09-20 保持不变）。
v4 重建已完成：产物在 `out/bank/camera_01030/v4`，manifest
`779ddc7ceccf21900ef3e7eb6fc62e6276ab7a6a46d573edfd0219b8f3fa0aa0`，
发布 `p0003` / `p0010` 两个 Profile；v1/v2/v3 与历史报告均未覆盖。

### v4 实测（C1/C2/C3）

* **C1**：13 → 2，11 轮删除，见上表。
* **C2**：合成阶段 `need=0`（高清帧全部由采样期缓存提供，没有无目的重拉）；
  回放阶段 `planned=16, consumed=14, released=14, need=16`（2 个复用缓存），
  失败 2 个且分类明确（1 个 `URL_REFRESH_FAILED` 业务码 1500、
  1 个 `NETWORK_FAILURE` 下载字节数不符）；校准阶段 `planned=4, consumed=4`；
  峰值原始缓存 **79,621,876 B / 8 GiB**、峰值工作目录 85,979,029 B、
  结束 `final_raw_bytes=0`；计划终态 `consumed=14` +
  `failed:download_error:RecordingSourceError=2`。
  注意峰值约等于**一个**文件大小，说明同一时刻只驻留一个原始 PS。
* **C3**：13 个 Profile 里 **2 个**（g015、g018）拿到外观匹配且 ≥2 块独立校准，
  `source=calibration_day`、`calibration_sufficient=true`、`prior_suitable=true`；
  其余 **11 个**退回 `source=reference_self`、`no_appearance_match_low_support`、
  `prior_suitable=false`，退化原因是 `no_appearance_match` 或
  `single_matched_observation`。发布的两个 Profile（p0003、p0010）都属于后者：
  `source=reference_self`、`prior_suitable=false`、`low_support=true/false`，
  阈值来自参考自身分布与基础阈值。

  **保守行为的边界**：`prior_suitable` 目前是**报告字段**，运行时没有用它去做
  准入判断——真正生效的保守行为是"噪声来自参考自身观测 + 基础阈值 + 低支持
  标记"，这正是用户要求的两种做法之一。把"不适合 prior"变成运行时门禁需要改
  Selector 契约，已列入留给方案二的清单，本轮没有改 Selector 代码。

## 6. 留给方案二（不变）

事件 memory 完整生命周期、在线背景准备与原子提交的实时表现、全天覆盖与在线
恢复速度、以及"构建期 score-only 回放画布 vs 生产 availability-aware adapter"
的边界测试。参考选择/切换的修改会动到 Selector 契约，属于方案二范围；
本轮只给出诊断证据，没有改 Selector 代码。

## 7. 可复现命令

```bash
# 本地：相关回归
.venv-profile/bin/python -m pytest tests/test_ground_litter_profile_c1c4.py \
    tests/test_ground_litter_profile_repairs.py \
    tests/test_ground_litter_profile_factory.py -q

# 本地：CI 反例脚本（保持只读）
.venv-profile/bin/python output/profile_factory_c1c4_20260921/verify_c1c4.py

# 服务器：小目标诊断（只读，复用已落盘素材）
cd ~/profile-factory
./venv/bin/python scripts/diagnose_ground_litter_small_target.py \
  --bank-root out/bank --bank-id camera_01030 --version v3 \
  --media diag_media --frame-index 40 \
  --native-size-px 8,16,24 --online-size 960x540 \
  --output out/c1c4_diag/small_target

# 服务器：v4 重建（构建 09-14~09-17 / 校准 09-18）
./run_v4.sh v4 4 40
```
