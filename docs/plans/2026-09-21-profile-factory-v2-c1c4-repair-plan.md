# Profile Factory v2 复核后 C1–C4 定向修复计划

日期：2026-09-21
范围：只针对 `docs/plans/2026-09-20-profile-factory-v2-acceptance-review.md` 的 C1–C4。
保留 R1–R11 全部结论；**不**实现方案二生产接入；不重开 R1–R11 争议。
基线证据：`output/profile_factory_v2_review_20260920/reproduce_remaining.py`（只读复现器，
本轮不得修改）与 `review_evidence.json`。

## 0. 验收边界（沿用本轮用户口径）

必须保证：

1. 对照实验公平（同一批帧、同一时间轴、同一观测条件）；
2. 全链路资源有界（PS、抽帧、中间产物、内存都算）；
3. 噪声标定真正接入（不能用"参考自身观测"冒充独立校准）；
4. 小目标统计对应真实可用输出（不是任意参考命中、不是拿降采样图糊弄）。

不要求：零误报、100% 召回、重采完整七天、压缩 Profile 数量、全天覆盖/现场精度/在线性能。
允许：持续七天存在的物体进背景、少量残余误差。r2 ROI 保持不变。

## 1. 服务器/本地版本漂移（已实测确认）

| 文件 | 本地 SHA-256（前16） | 服务器 SHA-256（前16） | 状态 |
| --- | --- | --- | --- |
| `rtsp_annotator/ground_litter_profile_selector.py` | `7936664f95014b88` | `0694d519ea35b419` | **漂移**：服务器缺 `a27c2ab`（观测间隔随实际节拍自适应） |
| `rtsp_annotator/ground_litter_profile_background.py` | `b2f2d4bd19f1e4be` | `b2f2d4bd19f1e4be` | 一致 |
| `rtsp_annotator/ground_litter_profile_analysis.py` | `07a60dd900f41bc7` | `07a60dd900f41bc7` | 一致 |
| `rtsp_annotator/ground_litter_profile_sampling.py` | `e74c56995343a5db` | `e74c56995343a5db` | 一致 |
| `scripts/build_ground_litter_profile_bank.py` | `b49d662282163244` | `b49d662282163244` | 一致 |
| `scripts/evaluate_ground_litter_profile_bank.py` | `701a9ceeef0013bc` | `701a9ceeef0013bc` | 一致 |

结论：v2 服务器产物与本地代码一致，**唯一**差异是 selector 的节拍自适应补丁。
`replay_all_candidates` 内部显式设置 `max_observation_gap_seconds`，因此该漂移对
工厂动态定稿无影响，但 **不能**据此认为服务器与本地等价——本轮必须把补丁同步到
伺服器隔离目录 `~/profile-factory`，再记录最终源码 SHA。

## 2. C1：留一法对照不公平 + 一次性删除

### 2.1 根因

`scripts/build_ground_litter_profile_bank.py::replay_all_candidates`

- baseline 用 `run(all_ids)`（stride=1，全部帧），LOO 用 `run(subset, stride=loo_stride)`，
  默认 `loo_stride=4` → **两者看到的帧不同**，`effective_fraction_delta` 与
  `pause_max_delta` 里混入了采样密度差，不是候选删除的真实代价。
- 每个 subset 都重新 `score_profile` 全量评分，`CandidateMatch` 列表在 subset 间不可比，
  也无法复核"同一帧同一候选分数相同"。
- `select_profiles_dynamically` 用**一次性**删除：所有标记为冗余的候选被同时移除，
  LOO 代价只在"从全库删除"这一点上成立，删除集合内部会互相掩盖。
  两个完全相同的参考互为例外：各自 LOO 代价都为 0，于是被同时删除。

### 2.2 修改

1. **分数矩阵缓存**：新增 `_replay_score_matrix(...)`，对每个 `(frame, candidate)`
   只算一次 `score_profile` + `evaluate_match`，缓存 `CandidateMatch` 与
   `frame_sha256`、`tick_interval_seconds`、`replay_time`。
   baseline 与所有 LOO/终选复核子集都从同一矩阵取分 → 帧、时间轴、观测条件完全一致。
2. **候选级时间线复核**：`_timeline_from_score_matrix(matrix, subset, ...)` 只做
   Selector 状态机推进，纯内存、可比、可重复。
3. **保守逐次删除**：新增 `prune_profiles_conservatively(...)`：
   每一轮只删除**当前剩余集合**中 LOO 代价最小且低于阈值的那**一个**候选，
   删除后**重新**在剩余集合上做 LOO 复核，直到不可再删或触发保底。
   这样"两个相同参考"最多删掉一个（删掉一个后另一个的 LOO 代价必然上升）。
4. **终选集合与资源裁剪集合都要复核**：
   - `final_set_verification`：对删除后的终选集合重跑完整回放，与 baseline 逐 tick 对比
     （帧哈希、tick 数、覆盖、暂停、切换次数）；
   - `resource_trimmed_verification`：若终选集合超过 `max_profiles`，对裁剪后的集合
     再复核一次，并把两次指标一起写进报告。
5. **公平对照报告**：新产物 `n_selection_fair_comparison.json`，包含
   `method`、`frames_identical=true`、`loo_stride_effective=1`、每个候选的
   删除次序与删除时的复核指标（不是"删除前一次算出来的"旧值）、
   以及"全库 / 终选 / 资源裁剪 / 每个候选单独"的显式指标对照。
6. 保留 `loo_stride` 参数，但只作为**显式实验开关**记录在报告里，默认 1，
   不再让 baseline/LOO 使用不同值。

### 2.3 回归测试（新增到 `tests/test_ground_litter_profile_repairs.py`）

- `test_replay_uses_identical_frames_for_baseline_and_loo`：断言报告中两侧 frame 哈希列表相同。
- `test_identical_reference_pair_deletes_at_most_one`：造两个完全相同的候选，
  逐次删除后至少保留一个，且覆盖不为 0。
- `test_final_set_and_trimmed_set_are_reverified`：断言两类复核都出现在报告里。
- `test_score_matrix_reused`：同一 `(frame,candidate)` 的分数在 baseline 与 LOO 中相等。

### 2.4 需要重跑的阶段

粗采样（复用缓存）→ 分组 → 高清合成/噪声 → 包络拟合 → **动态定稿（本项）** → 发布。
v3 版本号重建资产，不覆盖 v1/v2。

## 3. C2：高清/回放/标定阶段仍然无界

### 3.1 根因

`_ensure_remote_entries` 在合成前对 `files`（= 全部选中文件，含校准日/盲测日）
逐个 `can_reserve` → 刷新 URL → 下载。缺点：

- 不区分"这一阶段真正需要哪些文件"；盲测日素材会被拉下来但从不参与构建/校准；
- 一次失败的预留只会写进 `failed`，阶段继续，且失败会被后续统计当成"素材失败"；
- 原始 PS、抽帧、合成中间产物三者各自有预算，但没有一个整链路的峰值口径。

### 3.2 修改

1. `_ensure_remote_entries(..., needed_file_ids=...)`：只对**该阶段声明需要**的文件
   做重拉；调用方传入构建块 + 校准块对应的 `file_id` 集合。盲测日文件不再进入高清阶段。
2. `materialize_entry`：单一素材准备单元（复用缓存/复用本地/下载/失败分类）。
   `begin_download` 要求条目已登记，统一在这里 `register`；这同时修复了
   "新版本重建复用清单缓存时画布预取以 `CacheError` 中止整段作业"。
3. `_collect_replay_frames(..., args=, local_paths=, report=)`：回放是独立消费阶段，
   自己按需（有界、顺序）重取缺失素材。历史缺陷是合成阶段释放构建 PS 后，
   `--resume` 又会回收残留，回放静默收集 0 帧、动态定稿失去时间轴。
4. 阶段化记账：`commit_stage_bytes` / `release_stage_bytes` / `log_stage_event` /
   `stage_report`，并在合成、包络拟合、回放收集结束后显式 `_release_materialized`。
5. 失败口径：`NO_SOURCE_TO_REPULL` / `BUDGET_REJECTED` / `URL_REFRESH_FAILED` /
   `NETWORK_FAILURE` / `MATERIAL_INVALID` 分别记录，**不计入**素材质量失败。
6. `report["resource_envelope"]`：原始 PS / 工作目录峰值、阶段字节、抽帧与回放帧
   内存口径（`frames × H × W × 3`）、预算收缩、素材失败清单、计划文件终态。
   超预算时缩小该阶段的抽帧数并记录，不得静默截断校准/回放范围。

### 3.3 测试

- `test_hd_repull_only_fetches_needed_files`：给一个含盲测日的文件清单，
  断言请求下载的 file_id 集合不含盲测日。
- `test_stage_budget_never_exceeds_quota`：总量 > 配额时断言 `requested_bytes` 单调有界、
  且失败被记录为预算类原因而非素材失败。
- `test_factory_chain_bounded_integration`：走完"清单→分区→分阶段拉取→合成→包络→
  回放→定稿→发布"的完整调用链（假 I/O），总输入 > 缓存配额，断言峰值不超过配额
  且每个阶段都有 `released` 记账。**必须覆盖多轮采样**，不能只测第一轮。

## 4. C3：噪声标定实际没接上

### 4.1 根因

`run_factory` 里 `calibration_blocks` 由 `samples` 的 `time_block` 与校准日文件比对得到，
但 `samples` 只来自 `build_files`（分区里 `build`），所以该集合**恒为空**。
`stage_composite_and_noise` 于是走 `low_support_used_all_blocks` 分支：
13/13 Profile 的噪声估计都用了参考自身的观测，`independent_of_reference=false`。

### 4.2 修改

1. **校准素材进入采样**：粗采样阶段对 `calibration_files` 也做一次外观采样
   （同样的 `BoundedPreviewSampler` 口径、同样的画布注册），得到 `calibration_samples`。
   校准样本**不参与**参考合成/分组，只用于噪声标定。
2. **外观匹配选校准素材**：新增 `select_calibration_observations(...)`，
   对每个候选组按描述子粗距离 + 外观分组把校准样本分配给最匹配的组；
   匹配不上的组标注 `no_appearance_match`。
3. **独立块判定**：`calibration_blocks` 由"校准日样本的 time_block"构造，
   并在合成阶段断言 `calibration_blocks ∩ build_blocks = ∅`；若交集非空则报错，
   不允许用构建块冒充独立校准。
4. **共享补偿/残差定义**：`estimate_noise` 与运行时 `foreground_support` 都必须走
   `residual_maps(reference, frame)` 的同一补偿路径与同一定义（`compensated_reference`），
   新增断言测试防止两边分叉。
5. **逐 Profile 报告**：`profile.json` 增加
   `noise_calibration = {source: "calibration_day"|"reference_self"|..., 
   independent_blocks: N, appearance_match_distance: d, degradation_reason: str|null}`。
   `independent=true` 只能由真实"来源不同 + 匹配成功 + 独立块≥2"推出，**禁止手写**。
6. 少数确实无独立素材的参考允许保持低支持（`min_support_blocks` 约束下用基础阈值），
   但必须写明降级原因；接线缺失不得导致全库一起降级。

### 4.3 测试

- `test_calibration_noise_sources_are_disjoint_from_synthesis`：工厂级断言
  `calibration_blocks ∩ build_blocks = ∅` 且 `independent_of_reference=true`（有素材时）。
- `test_noise_residual_uses_shared_compensation`：运行时 `foreground_support` 与
  `estimate_noise` 对同一 `(reference, frame)` 得到同一残差中心定义。
- `test_profile_json_reports_noise_source`：逐 Profile 报告字段存在且非手写常量。
- `test_low_support_profiles_are_conservative`：无独立素材时仍发布，但阈值退化为 base。

## 5. C4：评估口径把"任意参考命中"当成了可用命中

### 5.1 根因

`scripts/evaluate_ground_litter_profile_bank.py::_small_target_summary`

- 只统计"任一参考有候选"，没有区分命中是否来自**当时真正生效**的参考；
- 没有强制 `prior_allowed`（先验不允许时仍计入命中）；
- 成对基线不是同帧"注入 vs 非注入"同状态对照；
- 命中行统计不完整（有候选但被判定的行被丢掉）。

### 5.2 修改

1. 双口径：`potential_hit`（任一参考有候选）与 `effective_hit`
   （候选来自 `selector.selected_profile_id` 且 `prior_allowed=true`）。两者分别报告，
   不合并、不新增"通过线"。
2. 成对基线：同一源帧、同一参考、同一 selector 状态，跑一次注入一次不注入；
   禁止用上一帧冒充配对基线。
3. 空间匹配在真值位置：命中点必须落在注入位置邻域内，按原始比例换算，
   不能用整幅 ROI 命中糊弄。
4. 完整计数：所有基线行都进分母，包括"目标被命中但参考未生效"的行；
   `_small_target_summary` 输出 `rows_total / rows_scored / rows_dropped` 与丢弃原因。
5. 原生尺度样本：新增按 2560×1440 原图注入（不是先在降采样图上画亮块再放大），
   并走真实共享 adapter（`BankPriorContext` + `evaluate_bank_frame`）。
6. 事件记忆只实例化/通知，不在此处声明生命周期验收；生命周期交给方案二。

### 5.3 测试

- `test_hit_reporting_splits_potential_and_effective`：构造"有候选但非生效参考"的行，
  断言只进 potential 不进 effective。
- `test_effective_hits_require_prior_allowed`：`prior_allowed=false` 时 effective=0。
- `test_paired_baseline_uses_same_frame_and_state`：断言配对两侧 frame_sha256 相同。
- `test_native_scale_injection_goes_through_shared_adapter`：断言注入尺寸为原生画布
  且调用栈经过共享 adapter。
- `test_all_baseline_rows_are_counted`：断言分母包含被命中的行。

## 6. 复现与验收命令

```bash
# 反例基线（不得修改）
.venv-profile/bin/python output/profile_factory_v2_review_20260920/reproduce_remaining.py

# 目标回归
.venv-profile/bin/python -m pytest tests/test_ground_litter_profile_repairs.py -q
.venv-profile/bin/python -m pytest tests/test_ground_litter_profile_factory.py -q
.venv-profile/bin/python -m pytest tests -q

# 服务器隔离验证（短命令 + nohup + 轮询）
# 1) 同步 selector 补丁与本次改动到 ~/profile-factory
# 2) 复用 work_v2/inventory_5307975a77b599f5.json 与已缓存素材，构建 v3
# 3) 跑独立日评估与原生尺度小目标样本
```

## 7. 可复用资产

- `config/ground_litter_01030_geometry.json`（r2 ROI，保持不改）；
- 服务器 `~/profile-factory/work_v2/` 的清单缓存与已下载 PS；
- `output/profile_factory_v2_20260920/` 的 v2 产物（只读对照，不覆盖）；
- 本地测试夹具 `tests/profile_bank_fixtures.py`。

## 8. 明确留给方案二

- 事件记忆的**完整生命周期**验证（创建/更新/清理/重启恢复）；
- 在线背景准备、最新帧验证与原子提交的实时表现；
- 全天覆盖、在线恢复速度、现场准确率；
- 构建期 N 选择用的 score-only 回放画布与生产 availability-aware adapter 的边界测试。

## 9. 交付物

1. C1–C4 闭项表（代码位置 / 测试 / 实测证据）；
2. 最终源码 SHA + 本地/服务器测试结果；
3. 公平 N 选择对照 + 终选集合复核；
4. 全链路资源峰值与计划文件终态；
5. 逐 Profile 噪声来源与低支持状态；
6. 修正后的逐目标/成对基线报告 + 原生尺度样本；
7. 新资产与哈希（若重建）+ 可复现命令 + 精简交接文档；
8. 留给方案二的清单。
