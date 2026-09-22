# Profile Factory v2 复核后 C1–C4 定向修复交接（2026-09-21）

> **历史记录**：后续 oracle Go/No-Go 实验已经停止 Profile prior 路线。本文不再构成继续修复、建库或生产接入的依据。见 `docs/decisions/2026-09-21-ground-litter-profile-prior-no-go.md`。

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
| 测试 | `tests/test_ground_litter_profile_c1c4.py`（21 项） |

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

## 5. 服务器隔离验证（v3 重建 + 独立日评估）

服务器：`sf01@14.21.88.97:21002` 隔离目录 `~/profile-factory`（未触碰生产部署目录、
未重启任何生产容器、未创建生产流）。分区与 v2 完全一致：构建 09-14~09-17、
校准 09-18、独立验证 09-19/09-20。产物写入新版本 `out/bank/camera_01030/v3`，
v1/v2 目录与历史报告均未覆盖。

v3 manifest：`9347f99e01f9f8303192db9751ad2a69064978b192a56c39692bc32bbe190ac6`。
`V3_EXIT=0`（2026-09-20T23:52:33+08:00）。原始抽取结果
`out/c1c4_v3_analysis.json`、报告 `out/bank/camera_01030/v3/reports/factory_report.json`。

### C1 实测（v2 → v3）

| 指标 | v2 | v3 |
| --- | --- | --- |
| 候选 → 终选 | 13 → 13 | 13 → 13 |
| baseline 有效覆盖 | 0.71111 | 0.71111 |
| 留一法 `loo_stride` | 4（与基线不同帧） | **1**，`stride_ignored=true` |
| 留一法 `effective_fraction_without` | 13/13 全为 **0.0** | 全部 > 0，见下 |
| 删除轮数 | 一次性删除（无逐轮复核） | 0（没有任何候选满足删除阈值） |
| 终选复核 | 无 | 105 帧同帧复核，覆盖差 0.0，暂停差 0.0 |
| 资源裁剪复核 | 无 | 13 ≤ 24，与终选一致，同样复核 |

反例直接闭合（fake I/O，`output/profile_factory_c1c4_20260921/verify_c1c4.py`）：
两个完全相同的参考，`loo_stride=4` 与 `1` 下覆盖都不再归零
（baseline 0.89474，单独删任一个仍 0.89474，两个都保留）；
评分缓存使 3 候选 × 6 帧只调用 18 次 `score_profile`，不因 baseline+LOO 翻倍。

子集对照（同一矩阵、105 帧）：全库 / 终选 / 资源裁剪后都是
`effective_fraction=0.71111, pause_max=130.256, switch_count=13`；
任意单候选只有 `0.37778, pause_max=824.958`，说明"保留全部"不是靠阈值放水。

### C2 实测（v2 → v3）

| 指标 | v2 | v3 |
| --- | --- | --- |
| 高清重拉 checked / considered | 25 / 25 | 19 / 25（6 个盲测日文件被跳过） |
| 重拉字节 | 1,666,452,323 | 1,406,880,200 |
| 原始缓存峰值 / 预算 | — / 8 GiB | 1,406,880,200 / 8,589,934,592 |
| 工作目录峰值 / 预算 | — / 60 GiB | 1,413,204,521 / 64,424,509,440 |
| 阶段字节 | 无 | `hd_materialize=2,523,806,486`、`composite=822,067,200` |
| 阶段释放 | 仅最后 `evict_to_budget` | `after_composite=15`、`after_envelope_fit=4`、`after_replay_collect=15` |
| 失败分类 | 未分类 | 2 个文件 `NETWORK_FAILURE`（下载字节数不符），0 个被当成素材质量问题 |

第一次 v3 运行还暴露并修掉两个真实缺陷（两者都不是评审列出的，但都会让"有界"
名不副实）：
1. 合成阶段释放构建 PS 后，`--resume` 又会回收残留，回放阶段静默收集到 **0 帧**；
   现在回放会自己按需重取，实测重新收集到 **105 帧**（`replay_materialize.ok=15`）。
2. `begin_download` 要求条目先登记；换工作目录后画布预取以 `CacheError` 中止整段作业，
   现在统一在 `materialize_entry` 里 `register`。
计划文件终态：`consumed=15`、`failed:download_error:RecordingSourceError=2`，
也就是说 2 个素材缺口被明确记成"来源下载不完整"，而不是被算作采样失败。

### C3 实测（v2 → v3）

| 指标 | v2 | v3 |
| --- | --- | --- |
| `noise_note` | 13/13 `low_support_used_all_blocks` | 13/13 `independent_calibration_*` |
| 噪声来源 | 全部是参考自身观测 | 全部 `calibration_day` |
| `independent_of_reference` | 13/13 false | **13/13 true** |
| 独立块数 | 0 | 3~16（`calibration_blocks=16`，与 `build_blocks=60` 无交集） |
| 外观匹配成功 | 无该机制 | 2/13（g016、g019） |
| 参考自身观测 | 13/13 | **0** |

校准日样本确实被采样：`calibration_files=4`、`calibration_observations=16`、
`overlap_blocks=[]`（构建块与校准块求交为空是硬校验，重叠直接拒绝发布）。

外观匹配没有全部成功的**诚实原因**：校准日（09-18）本身就是一个独立的外观簇，
它到多数候选组的最小距离（22~40）大于"候选组自身离散度"标尺（9~23）。
本轮已把标尺下限放宽到校准观测整体的公共离散尺度，仍然只有 2 个组匹配上；
其余 11 个组退化为 `independent_calibration_day_cross_group`
（**仍是独立校准日的观测，不是参考自身观测**），并在每个
`profile.json.noise_calibration` 里写明 `degradation_reason=no_appearance_match_for_group`、
`appearance_match_distance`、`appearance_match_limit` 与近失列表。
这属于素材条件，不是接线缺失；要真正逐组匹配需要给每个外观簇准备各自的独立日素材。

### C4 实测（同口径独立日评估，v2 与 v3 各跑一遍）

命令：`./run_c1c4_eval.sh`（`eval_download_and_evaluate.py` +
`--small-target-trials 12 --small-target-size-native-px 8`，**原生 2560×1440 注入后
整体缩放到 960×540**，走真实共享 adapter；`--analysis-fps 0.05`；
v2/v3 使用同一批素材、同一画布、同一随机种子）。

| 口径 | v2 09-19 | v3 09-19 | v2 09-20 | v3 09-20 |
| --- | --- | --- | --- | --- |
| 行数（total / dropped） | 10 / 0 | 10 / 0 | 10 / 0 | 10 / 0 |
| 潜力命中 potential | 3 (0.3) | 4 (0.4) | 2 (0.2) | 2 (0.2) |
| 可用命中 effective | 0 (0.0) | 0 (0.0) | 1 (0.1) | 1 (0.1) |
| 先验不允许时的命中 | 2 | 2 | 1 | 1 |
| 命中但非生效参考 | 3 | 4 | 1 | 1 |
| 成对基线带候选 | 10/10 | 10/10 | 10/10 | 10/10 |
| 原生尺度注入行 | 10/10 | 10/10 | 10/10 | 10/10 |
| 动态有效覆盖 | 0.63087 | 0.61745 | 0.81879 | 0.81879 |
| 静态潜在覆盖 | 0.89474 | 0.89474 | 0.94737 | 0.94737 |

结论按用户口径原样报告，不设通过线：

* v2 与 v3 在同一天上几乎一致（09-20 完全相同），说明**这份数字的变化来自复核
  口径本身，而不是 Bank 变化**。v2 旧报告写的是 09-19 8/10、09-20 7/12；
  换成"原生尺度注入 + 生效参考 + prior_allowed"后只剩 3~4/10 与 2/10。
  差的那些就是旧口径的虚高来源：目标在画布上被放大 2.67 倍，以及先验不允许 /
  非生效参考的命中被算成命中。
* 8 像素原生目标在 960×540 画布上只剩约 3 像素边长；
  **可用命中只有 09-20 的 1 例（0.1），09-19 为 0**。
* 成对基线 10/10 都带候选（51~151 个），说明独立日素材本身噪声很高，
  单看"命中"不能当作识别能力证据；报告同时给出这两个数就是为了避免这种误读。
* `event_memory_lifecycle=not_enabled`；事件 memory 本轮只做到实例化与切换通知，
  完整生命周期验证**交给方案二**。

### 测试结果（本地 / 服务器同一源码）

| 范围 | 本地（`.venv-profile`） | 服务器（`~/profile-factory/venv`） |
| --- | --- | --- |
| `test_ground_litter_profile_{c1c4,repairs,factory}.py` | **137 passed** | **137 passed**（41.70s） |
| 全量 `tests` | 769 passed + 4 failed（`test_shared_inference`，本机无 torch） | 720 passed + 17 failed |

服务器那 17 项的构成必须说清楚，不能含糊成"服务器也全绿"：

* 4 项 `test_shared_inference` 与本地同样失败，原因是该 venv 没有 torch；
* 3 项 `test_ground_litter_v32_production`、5 项 `test_ground_litter_v33_api`、
  5 项 `test_ground_litter_v33_bundle` 需要打包清单/Dockerfile/示例配置；
  `~/profile-factory` 只是"profile-factory 子集 + 少量依赖"的隔离目录，
  本来就不含这些生产打包文件，属于环境缺失，不是本轮改动回归。
  这些测试在本地（含完整仓库）全部通过。

**结论：与本轮 C1–C4 相关的 137 项在服务器上全绿，且服务器与本地 6 个源文件
SHA-256 逐字节一致。**

## 6. 明确留给方案二

- 事件 memory 的完整生命周期验证（创建/更新/清理/重启恢复）；
- 在线背景准备、最新帧验证与原子提交的实时表现；
- 全天覆盖、在线恢复速度、现场准确率；
- 构建期 score-only 回放画布与生产 availability-aware adapter 的边界测试。

## 7. 本轮刻意不改的东西

- r2 ROI（`config/ground_litter_01030_geometry.json`）保持不变；
- 评审反例脚本 `output/profile_factory_v2_review_20260920/reproduce_remaining.py`
  保持只读：它的 stride 断言针对旧行为，修复后必然失败，这是"反例已闭合"的证据，
  不得为了让脚本通过而改它。新的验证放在
  `tests/test_ground_litter_profile_c1c4.py`（21 项）与
  `output/profile_factory_c1c4_20260921/verify_c1c4.py`（可独立运行，
  结果见同目录 `verification_results.json`）。
