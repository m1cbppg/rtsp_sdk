# Profile Factory v2 复核后 C1–C4 定向修复交接（2026-09-21）

本轮只处理 `docs/plans/2026-09-20-profile-factory-v2-acceptance-review.md` 的
C1–C4，保留 R1–R11 全部结论；未实现方案二生产接入，也未重开 R1–R11 争议。
计划见 `docs/plans/2026-09-21-profile-factory-v2-c1c4-repair-plan.md`。

## 0. 版本与哈希

| 项 | 值 |
| --- | --- |
| 本地提交 | `308a80a`（C1–C4 修复）+ 本轮后续补丁 |
| `scripts/build_ground_litter_profile_bank.py` | `e678b032deed0ac4…` |
| `scripts/evaluate_ground_litter_profile_bank.py` | `10214df734fb329b…` |
| `rtsp_annotator/ground_litter_profile_background.py` | `0154feb34237d955…` |
| `rtsp_annotator/ground_litter_recording_cache.py` | `d65af36b9e436aa2…` |
| `rtsp_annotator/ground_litter_profile_selector.py` | `7936664f95014b88…`（补 `a27c2ab` 节拍自适应） |
| 测试 | `tests/test_ground_litter_profile_c1c4.py`（20 项） |

**服务器/本地漂移已消除**：v2 复核时服务器 `ground_litter_profile_selector.py`
为 `0694d519ea35b419…`（缺 `a27c2ab`），本轮已同步为本地 `7936664f95014b88…`。
其余 5 个源文件本轮开始前即与本地一致（逐文件 SHA-256 比对见计划文档 §1）。

## 1. C1：对照不公平 + 一次性删除

### 根因

`replay_all_candidates` 的 baseline 用 `stride=1`、留一法用 `loo_stride=4`（默认），
两者看到的帧不同；`select_profiles_dynamically` 用**一次性**删除，所有 LOO 代价
都相对全库计算，删掉一个后另一个原本冗余的候选不再冗余时不会被发现。
`output/profile_factory_v2_review_20260920/review_evidence.json` 记录：
stride 4 时两个完全相同的参考互相删除后覆盖为 0。

### 修改位置

- `scripts/build_ground_litter_profile_bank.py`
  - `ReplayScoreMatrix` / `build_replay_score_matrix`：逐 `(frame, candidate)`
    只评一次分，基线、LOO、终选复核共用同一矩阵。
  - `project_selection_timeline`：在矩阵上推进 Selector（不重新评分）；
    子集按自己的真实观测时刻推导节拍，避免"抽样变稀"被误读成"删候选导致归零"。
  - `replay_all_candidates`：`loo_stride` 一律按 1 执行并在报告里标注
    `loo_stride_ignored`；`frames_identical_to_baseline` 恒真。
  - `prune_profiles_conservatively`：每轮只删**当前剩余集合** LOO 代价最小的一个
    冗余候选，删后在剩余集合上重新计算；`deletion_order` 逐步留证。
  - `_trim_to_resource_limit`：超过 `max_profiles` 时继续逐次删除并复核。
  - `_verify_selected_subsets`：终选集合与资源裁剪集合都重跑完整时间线，
    逐帧核对与基线一致；产物写入 `n_selection_fair_comparison.json`。
- 兼容：`select_profiles_dynamically` 保留原名，行为改为保守逐次删除；
  只拿到预计算 LOO 的旧调用走 `implementation_path=legacy_one_shot_from_precomputed_loo`。

### 测试与证据

`tests/test_ground_litter_profile_c1c4.py::C1FairComparisonTests`：
同帧一致性、评分缓存命中次数（3 候选 × 6 帧 = 18 次，不因 baseline+LOO 翻倍）、
相同参考最多删一个、逐次删除留证、终选/裁剪集合都在共享矩阵上复核。

## 2. C2：高清/回放/标定阶段无界

### 根因

`_ensure_remote_entries` 对分区后的**全部**文件（含盲测日）逐个重拉，盲测素材
下载后从不参与合成/校准；失败原因不分类；原始 PS、抽帧、中间产物没有整链路口径。
复核证据：真实 HD 重拉 23 个文件 / 1,666,452,323 字节。

### 修改位置

- `_ensure_remote_entries(..., needed_file_ids=...)`：只处理本阶段声明的文件；
  `skipped_not_needed` 与 `skipped_not_needed_ids` 逐条留证。
- `run_factory`：调用前算出 `needed_hd_files`（有可用构建样本的构建文件 + 校准文件），
  写入 `report["material_plan"]`；盲测日文件不进高清阶段。
- `materialize_entry`：单一素材准备单元（复用缓存/复用本地/下载/失败分类）。
  `begin_download` 要求先登记，统一在这里 `register`，修复"换工作目录后画布预取
  以 CacheError 中止"的问题。
- 失败分类：`NO_SOURCE_TO_REPULL` / `BUDGET_REJECTED` / `URL_REFRESH_FAILED` /
  `NETWORK_FAILURE` / `MATERIAL_INVALID`，不再把网络/预算受限记成素材失败。
- `_release_materialized`：合成结束、包络拟合结束、回放收集结束都显式归还临时 PS；
  `require_committed` 只保留真正已提交的阶段名。
- `_collect_replay_frames`：回放是独立消费阶段，会自己按需（有界、顺序）重取缺失
  素材；历史缺陷是合成阶段释放 + `--resume` 回收导致回放静默收集 0 帧。
- `ManagedRecordingCache.commit_stage_bytes/release_stage_bytes/log_stage_event/
  stage_report`：按阶段记账。
- `report["resource_envelope"]`：原始 PS / 工作目录峰值、阶段字节、抽帧与回放帧
  内存口径、预算收缩记录、素材失败清单、计划文件终态。

### 测试

`C2BoundedPipelineTests`：只拉声明文件、失败分类不混用、全链路（清单→分区→分阶段
拉取→合成→包络→回放→定稿→发布）在"总输入 > 缓存配额"下跑完且峰值不超配额；
断言覆盖多轮采样（`planned > prefetch_slots`）。

## 3. C3：噪声标定实际没接上

### 根因

`run_factory` 里 `calibration_blocks` 由 `samples`（只来自 `build_files`）与校准日
文件比对得出，恒为空；合成阶段于是全部走 `low_support_used_all_blocks`。
服务器 v2 报告：13/13 `independent_of_reference=false`。

### 修改位置

- `run_factory`：对校准日文件**再做一次采样**（同一 `BoundedPreviewSampler` 口径、
  同一冻结画布），得到 `calibration_observations`；构建块/校准块由样本自带的
  `day` 字段分流（`block_ids_by_day`），不再依赖调用方手工拼装。
- 校准采样会覆盖 `pipeline/sampling/hd_plan/registration/frozen_canvas`，
  这里备份构建侧记录并在事后恢复，另存 `calibration_sampling`。
- `ground_litter_profile_background.select_calibration_observations`：独立校准观测
  先按自身外观聚类，再按"候选组自身成员离散度"作标尺分配给最匹配的组；
  匹配距离、限值、近失记录都逐组留证。
- `stage_composite_and_noise`：只接受**分配给本组**的校准块；构建块与校准块求交
  非空直接报错拒绝发布；匹配失败时退化为"跨组但仍是校准日"，并写明
  `degradation_reason`；完全没有独立素材才退化为参考自身观测。
- 独立性由代码推导：`independent_of_reference = 来源是校准日 且 独立块 ≥ 2`，
  `appearance_matched` 单列；两者连同 `source`、`independent_blocks`、
  `appearance_match_distance`、`degradation_reason` 写入
  `report["composite"][*]["noise_calibration"]` 与每个 `profile.json`。

### 测试

`C3NoiseCalibrationTests`：校准观测按外观入组、不匹配时显式拒绝并留近失记录、
运行时评分暴露与离线估计同一套受限补偿、无素材时保持基础阈值。

## 4. C4：把"任意参考命中"当成可用命中

### 根因

`_inject_small_target_in_roi` 先缩放到评估画布再画亮块（2560→960 等于放大 2.67 倍）；
`_small_target_summary` 只统计"任一参考有候选"，不区分是否生效参考、不强制
`prior_allowed`；成对基线用上一帧；命中行统计不完整。

### 修改位置

- `inject_small_target_native_scale`：先在原生分辨率注入 `--small-target-size-native-px`
  大小的目标，再把**已注入整幅图**缩放到评估画布；填充值取该块原生均值 +90，
  不再是纯白块。旧口径保留为 `--small-target-canvas-injection`（对照用）。
- `_match_target`：`potential_hit`（任意参考）与 `effective_hit`（候选来自当时生效
  参考且 `prior_allowed=true`）分开；同时记录候选的 profile 列表、生效参考 id。
- 评估主循环：成对基线改为 `copy.deepcopy(selector)` 上跑**同一源帧未注入**版本，
  不再用 `ticks[-1]`。
- `_small_target_summary`：`rows_total/rows_scored/rows_dropped`、
  `potential_fraction`、`effective_fraction`、`potential_but_not_effective`、
  `hits_when_prior_not_allowed`、`native_scale_injections` 一并报告；
  不新增通过线。
- `report["small_target_method"]`：写明注入口径、成对基线口径、命中口径，以及
  `event_memory_lifecycle = instantiate_and_notify_only; full lifecycle validation is
  handed to 方案二`。

### 测试

`C4HitReportingTests`：潜力/可用分离、`prior_allowed` 门槛、汇总行完整、
原生注入在画布上确实更小、成对基线辅助代码同帧同状态。

## 5. 明确留给方案二

- 事件 memory 的完整生命周期验证（创建/更新/清理/重启恢复）；
- 在线背景准备、最新帧验证与原子提交的实时表现；
- 全天覆盖、在线恢复速度、现场准确率；
- 构建期 score-only 回放画布与生产 availability-aware adapter 的边界测试。

## 6. 本轮刻意不改的东西

- r2 ROI（`config/ground_litter_01030_geometry.json`）保持不变；
- 评审反例脚本 `output/profile_factory_v2_review_20260920/reproduce_remaining.py`
  保持只读：它的 stride 断言针对旧行为，修复后必然失败，这是"反例已闭合"的证据，
  不得为了让脚本通过而改它。新的验证放在
  `tests/test_ground_litter_profile_c1c4.py`。
