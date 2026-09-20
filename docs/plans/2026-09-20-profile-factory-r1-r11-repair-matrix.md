# 方案一实施评审 R1～R11 修复矩阵（2026-09-20）

依据：`docs/plans/2026-09-20-profile-factory-implementation-review.md`（本次验收不通过）。
基线反例：`output/profile_factory_review_20260920/reproduce_findings.py`，我在本机重跑，
8 条断言**全部成立**（见本文件末尾“基线确认”）。旧产物 `camera_01030/v1`、旧报告与
评审材料全部保留，新产物使用未被占用的新版本号。

## 决策记录（文档冲突/歧义）

| 冲突点 | 决策 | 理由 |
|---|---|---|
| 评审说“2560×1440 是 16:9，不能用录像 4:3 解释几何偏移” | **采纳评审**。r2 几何的偏移原因是：原 16 点来自 config 的示例机位、按 16:9 截图目视估计后直接套到 2560×1440 帧；不是宽高比算错 | 我上一轮 HANDOFF §6 写“录像 4:3”是**错误表述**，本文与新版 HANDOFF 更正；r2 多边形经真实帧复核后仍保留（评审 §ROI 决策要求保留业务覆盖范围） |
| 评审 R2 要求“按共同回放评估候选增减” vs 资源上限 | 采用**共同帧序列 + 有界动态剪枝**：所有候选面对同一 `(source_time, frame)`；候选数超过 `max_scored_candidates` 时，先按共同序列上的静态得分粗排，再对入选集合做**留一法动态对照** | 直接枚举全部子集不可行；留一法能证明“删掉某个候选会让动态指标变差”，这是评审要求的可检查证据 |
| 评审 R5 要求“背压等待消费” vs 单进程样本量 | 采用**有界预取窗口（默认 2 下载槽 + 1 处理槽）+ 等待重试**，处理完立即释放；不再 `break` 截断计划 | 与方案一 §4.6 的槽位设计一致；`break` 会让分区静默缩小 |
| 评审 R3 “不能凭现有汇总换算一个正确数值” | 汇总改为**逐 tick 有效区间积分**（区间 = min(下一 tick 源时间, 该 tick 结果有效期) − 本 tick 源时间），并把录像空隙/坏帧/不可判断分别记账 | 不再前向填充 |

## R1 训练/校准/验证隔离与冻结包络发布（P1）

* 根因：`evaluate_ground_litter_profile_bank.py:141` 把待评目录喂给 `_calibration_scores()`
  重新拟合包络；`build_...py:1233` 发布 `default_matcher_config()`，`matcher.json` 标记
  `uncalibrated_defaults`。调用链：evaluate.main → _calibration_scores → envelope_from_samples
  → ProfileSelector.observe。
* 拟改：`ground_litter_profile_bank.py`（manifest/envelope 契约 + loader 校验）、
  `build_...py`（只用校准集拟合，写入 `matcher.json.calibration` 与 `profiles[].envelope`）、
  `evaluate_...py`（只读冻结包络，禁止重估；新增 `--allow-refit` 仅用于回归对照且报告标注）。
* 回归：`test_frozen_bank_envelope_is_not_refit_on_blind_data`、
  `test_bank_loader_rejects_missing_envelope`。
* 需重生成：v2 Bank（含包络）、v2 报告。
* 证据：`matcher.json` 含 `calibration.source/inputs/hash`；盲测报告 `envelopes` 与 Bank 完全一致。

## R2 同一 tick 同一真实帧 + 动态选 N（P1）

* 根因：`build_...py:807-812` 每个候选取自己的 `evaluation_frames`；`_match_frames_to_candidates:911`
  顺序轮转；`_prune_candidates:933` 只看静态覆盖，`del selector` 空转。
* 拟改：`stage_preselect_and_finalize` 重写为「共同帧序列 → 每 tick 全候选评分 → 冻结包络 →
  同一 Selector 回放 → 留一法动态剪枝」；`_match_frames_to_candidates` 删除；回放时间改用
  真实源时间。
* 回归：`test_all_candidates_see_same_frame_and_source_time`、
  `test_dynamic_pruning_keeps_transition_reference`、`test_n_not_decided_by_static_ranking`。
* 证据：报告 `replay.scored_frames` 每 tick 的 `frame_sha` 唯一；`n_selection.decisions` 含
  每个候选的 `leave_one_out` 指标差。

## R3 覆盖按真实观测区间积分（P1）

* 根因：`ground_litter_profile_selector.py:609` 用“下一条决策时间”当区间，只按
  `join_gap_seconds` 截断。
* 拟改：`observer` 记录 `observation_interval`（本 tick 源时间 → min(下一 tick, 结果有效期)）与
  `availability_fraction`；`summarise(join_gap_seconds=...)` 改为逐区间积分，空隙/坏帧/
  不可判断分别计入 `non_observable_seconds`；新增 `record_gap()`。
* 回归：`test_coverage_does_not_forward_fill_unobserved_span`（200s 缺口 → 有效 ≈4s）、
  `test_long_gap_and_zero_availability_do_not_add_coverage`。
* 证据：报告给出 `effective_seconds/observed_seconds/non_observable_seconds` 三者可对账。

## R4 搜索预算与游标接线（P1）

* 根因：`evaluate_...py:438-442` 把 current+Top-K+reserved 拼接后截断；`plan_tick` 已推进游标，
  被截掉的 reserved 再也不会被检查；`current_hold_eligible=current_id is not None` 误把
  “存在当前参考”当成“当前仍可保持”。
* 拟改：`_run_tick` 改为「reserved 优先占位 + Top-K 补足 + current 保底」，并用真实
  `hold_eligible` 调 `plan_tick`；检查后调用新的 `selector.note_tested(...)` 反馈实际检查结果，
  避免游标空转。
* 回归：`test_run_tick_reaches_kth_plus_one_candidate`（真实 `_run_tick`，非仅 Selector 类）。
* 证据：评审反例场景下 p5 在有限 tick 内被检查并被选中。

## R5 有界“拉取→处理→提交→释放”流水线（P1）

* 根因：`build_...py:346` 先整批下载，`:361` 容量不足 `break`；`:490-519` 在 lease 上下文内
  调 release，必然被 `LEASED` 拒绝。
* 拟改：新增 `ground_litter_recording_cache.wait_for_capacity()`；`build_...py` 新增
  `stream_materialize_and_sample()`：有界预取窗口（默认 2+1）、等待重试、逐个文件
  `acquire → sample → commit → 退出 lease → release`；下载失败记终态但**不重新划分分区**。
* 回归：`test_bounded_pipeline_processes_all_planned_files_under_tight_budget`、
  `test_release_happens_after_lease_exit`。
* 证据：报告 `pipeline.final_states` 覆盖全部计划文件（无静默截断），`peak_raw_bytes` ≤ 预算。

## R6 原图尺度小目标注入与逐目标空间匹配（P1）

* 根因：`evaluate_...py:400-401` 在画面左上半区随机取点，与 ROI 无关（10/10 落在 ROI 外）；
  报告只统计“任意候选”，未与真值框匹配；`_notify_switch` 为空操作。
* 拟改：`_inject_small_target` 改为**在 ROI 内、按原图尺度**选点（先按 `bank.reference_size`
  注入再缩放，或按 scale 折算尺寸），保存真值框与像素数；新增
  `_match_detections_to_targets`（IoU/中心距）与成对无目标基线；报告区分
  “任意候选 / 目标命中 / 事件确认”；接入 `V33EventMemory` 做离线生命周期。
* 回归：`test_injected_target_lands_inside_roi_at_native_scale`、
  `test_target_hit_requires_spatial_match`、`test_paired_baseline_has_no_target_hits`。
* 证据：`evaluation.json.small_target.per_target[]` 给出真值框、命中框、IoU 与是否确认。

## R7 遮挡/运动排除、独立噪声校准、局部有效区域（P1）

* 根因：`build_...py:608/634/684/709` 合成 mask 直接取整个 ROI；噪声“留出块”最终回退到同一批；
  `valid` 发布为整个 ROI（16 张哈希相同）。
* 拟改：合成阶段由**每帧逐像素时间统计**（跨帧偏差 + 局部运动）导出 `observation_mask`，
  遮挡/运动像素不参与中位数与替换；噪声优先用**独立校准块**（`calibration_blocks` 不再丢弃），
  不足时明确写 `calibration_block_count=0`；发布的 `valid` 改为
  `ROI ∩ 合成可用性 ∩ 非持续偏差`；`low_support` 统一定义并影响准入（更严格 enter）。
* 回归：`test_composite_excludes_moving_occluders`、`test_valid_mask_reflects_local_availability`、
  `test_noise_uses_independent_calibration_blocks`、`test_low_support_profile_gets_stricter_entry`。

## R8 噪声统计保留逐帧逐块有效 mask（P1）

* 根因：`ground_litter_profile_background.py:574-578` 每条记录都对并集内所有像素写 `weights=1`，
  被遮挡像素照样计入支持。
* 拟改：权重按 `keep` 逐像素写入并与已有值取 max（支持同块重复聚合），支持计数与分位只用
  有效观测；全无效像素标 `low_support` 并用基础阈值。
* 回归：`test_noise_support_counts_only_valid_observations`（评审反例：左右各半 → 每像素支持=1）。

## R9 零可用/坏帧不得通过准入（P1）

* 根因：`ground_litter_profile_analysis.py:347-354` 算了 availability 却没进 outcome；
  `evaluate_...py:462` 直接 `verified=True`。
* 拟改：`evaluate_bank_frame` 在 `availability_fraction` 低于 `min_availability_fraction`
  或命中 `FRAME_CORRUPT/GEOMETRY_INVALID/NO_COMMON_VISIBILITY` 时，`enter/hold` 全部置否并给出
  `UNAVAILABLE`/`CORRUPT` 原因；`_run_tick` 用真实可用性构造 `CandidateMatch(verified=...)`，
  零可用性 tick 不进入清走/提交路径。
* 回归：`test_zero_availability_blocks_admission`、`test_bad_frame_does_not_accumulate_clean`。
* 证据：反例 `eligibility_ignores_unavailability` 由 true/true 变为 false/false。

## R10 统一 prior 补偿/残差与候选几何过滤（P1）

* 根因：`ground_litter_profile_analysis.py:79` 前景直接比较原 reference/current（未补偿）；
  `_support_candidates:367` 无尺寸/面积限制（1 像素噪点与 120×120 都成候选）。
* 拟改：把「受限全局补偿 → 残差 → 阈值 → 形态学 → 候选」抽成
  `ground_litter_profile_match.analyze_residual_support()`，评分与前景共用同一补偿结果；
  `_support_candidates` 恢复 V3.2 的尺寸/面积门槛（短边、面积、长边、面积上限，按画布比例），
  并保留小目标（≥ 最小面积即保留）。
* 回归：`test_foreground_uses_same_compensation_as_score`、`test_single_pixel_noise_is_not_a_candidate`、
  `test_large_blob_is_not_a_candidate`、`test_small_target_survives_geometry_filter`。

## R11 冻结共同画布（P2）

* 根因：`sampling.py:790` 每个文件用自己第一帧当配准基准；`build_...py:480` 建立的外层基准没有
  传进 sampler，也没传 overlay 排除区。
* 拟改：`BoundedPreviewSampler` 接受 `canvas_reference` 与 `overlay_exclude_zones`；构建阶段把
  训练集冻结画布写入 `work/canvas_reference.png` 并在报告中给出 SHA-256；每个文件记录配准
  诊断与 `applied_to_canvas` 标志；超限文件标 `registration_rejected`。
* 回归：`test_all_files_register_to_frozen_canvas`（跨文件平移 → 残差/ROI/目标坐标一致）。

## 新增/更新资产与报告

* 新 Bank：`camera_01030/v2`（或下一个未占用版本），含冻结包络、校准来源哈希、共同画布指纹、
  每 Profile 来源/支持范围/配准诊断。
* 新报告：`reports/factory_report.json`（含 pipeline 终态、共同回放逐 tick 记录、留一法对照）、
  `reports/coverage.json`（静态/动态/不可判断分开）、独立验证报告（新时间段）。
* 旧报告保留，并在旧 HANDOFF 上加“已被取代”横幅。

## 验收证据清单

1. R1～R11 各自：代码位置 + 回归测试名 + 实测证据路径。
2. 盲测数据变化不改变冻结参数：对同一 Bank 用两个不同盲测集跑评估，`matcher.json` 与
   `profiles[].envelope` 哈希不变。
3. 逐 tick 同帧同一时间；候选增删动态对照；长缺口/坏帧/零可用不产生覆盖；
   粗排末位可被检查；超配额全量有界处理；遮挡像素不入统计；原图尺度小目标空间匹配；
   亮度/噪点/大块/小目标过滤；跨文件共同画布；切换期事件身份。

## 基线确认（本次重跑）

`reproduce_findings.py` 8 条断言在修复前全部成立：

* `wrong_replay_frames`：6 tick 中 p1 恒记 50、p2 恒记 60（输入序列 10..60）→ R2 成立
* `pruning_ignores_selector`：传 `object()` 仍正常剪枝 → R2 成立
* `integration_search_starvation`：20 tick 从未检查 p5，`selected=null` → R4 成立
* `eligibility_ignores_unavailability`：availability=0 而 enter/hold=true → R9 成立
* `candidate_size_filters_missing`：1 像素噪点与 120×120 大块都成候选 → R10 成立
* `noise_counts_invalid_observations`：左右各半有效却都报支持 2 → R8 成立
* `gap_forward_fill`：200s 无观测仍算 202s 有效 → R3 成立
* `small_target_roi`：10/10 注入框与 ROI 交集为 0 → R6 成立

R1（盲测参与拟合）与 R5（批式下载+lease 内释放）与 R7（合成用整个 ROI、噪声回退同批）、
R11（每文件独立基准）由代码位置 + 服务器报告字段确认（`matcher.json.calibration=
uncalibrated_defaults`、`noise_note=low_support_used_all_blocks`、`valid_mask_hashes` 仅 1 种）。
