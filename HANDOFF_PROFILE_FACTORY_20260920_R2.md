# 方案一 R1～R11 关闭表与重新验收报告（2026-09-20）

对象：`docs/plans/2026-09-20-profile-factory-implementation-review.md` 的 11 项发现。
旧产物 `camera_01030/v1`、旧报告、评审反例全部保留；本报告使用新产物 `camera_01030/v2`。
旧 `HANDOFF_PROFILE_FACTORY_20260920.md` 顶部已加“已被取代”横幅，旧数字不改写。

基线确认：评审的 8 条反例我在修改前逐条重跑，**断言全部成立**（见
`docs/plans/2026-09-20-profile-factory-r1-r11-repair-matrix.md` 末尾“基线确认”）。

## 1. R1～R11 逐项关闭

| 项 | 根因（位置） | 修复（代码位置） | 回归测试 | 证据 |
|---|---|---|---|---|
| R1 盲测参与拟合 | `evaluate_...py` 把待评目录喂 `_calibration_scores()`；`build_...py` 发布 `default_matcher_config()` | `build_...py:fit_frozen_envelopes`（只用校准日）、`matcher["calibration"]`+`matcher["profiles"][*].envelope`；`bank.py:validate_calibration/load_bank(require_calibration=True)`；`evaluate_...py:main` 只读冻结包络，重拟合需显式 `--refit-envelope-on-input` | `test_r1_bank_loader_requires_frozen_envelopes`、`test_r1_evaluation_does_not_refit_envelope_on_blind_data`、`test_r1_refit_flag_is_explicit_and_flagged`、`test_r1_uncalibrated_bank_refused_without_override` | v2 `matcher.json.calibration.source=calibration_day`、13/13 包络；两次独立评估包络载荷**逐字节相同**、`envelope_source=frozen_in_bank` |
| R2 同一 tick 不同帧 / 静态前 16 定 N | `stage_preselect_and_finalize` 用各候选自己的 `evaluation_frames`；`_match_frames_to_candidates` 轮转；`_prune_candidates` 只看静态覆盖 | `_collect_replay_frames`（真实源时间+帧散列）、`replay_all_candidates`（全候选同帧、真实时间轴、节拍随源缩放）、`compare_selection_metrics`、`select_profiles_dynamically`（留一法动态对照） | `test_r2_all_candidates_see_same_frame_and_source_time`、`test_r2_dynamic_pruning_uses_leave_one_out_not_static`、`test_r2_n_not_capped_by_static_ranking` | v2 报告 `replay.frame_reuse`、`nominal_tick_seconds=43.4`；`selection_comparison`：13 库 0.7111，单候选 0.2222–0.5（**候选增删确实改变指标**） |
| R3 覆盖率前向填充 | `selector.summarise()` 用“下一条决策时间”当区间，仅按 300s 截断 | `ProfileSelector.observe(tick_interval_seconds=...)` + `summarise(join_gap_seconds=...)`：区间 = min(结果有效期, 下一真实观测)，录像空缺与非可判断分开记账；`mark_non_observable()` | `test_r3_no_forward_fill_over_long_gap`、`test_r3_explicit_gap_is_not_coverage` | 200s 无观测缺口：旧实现 202s 有效 → 现在有效 ≤8s、`off_air=300s`；独立验证报告 `observation_accounting` 三项可对账 |
| R4 搜索预算截断 | `_run_tick` 把 current+Top-K+reserved 拼接后截断；`current_hold_eligible=current_id is not None` | `_run_tick` 改为 reserved 优先、Top-K 补足、current 保底；`ProfileSelector.last_current_hold_eligible()` 用真实评分历史 | `test_r4_run_tick_reaches_kth_plus_one`、`test_r4_budget_is_still_bounded` | 评审反例场景下 p5 在有限 tick 内被检查并**被选中**（旧为 20 tick 从未检查、selected=null） |
| R5 无有界流水线 | `materialize_remote` 整批下载、容量不足 `break`；`release_after_preview` 在 lease 内调用必被拒 | `recording_cache.wait_for_capacity()`；`build_...py:stream_materialize_and_sample`（有界预取窗口、等待消费、退出 lease 后 release） | `test_r5_all_planned_files_consumed_under_tight_budget`、`test_r5_wait_for_capacity_reports_timeout` | v2 `pipeline`：planned 17 / consumed 15 / failed 2（09-13 过期素材）/ `truncated_plan=false` / `released_files=15`；测试中预算 1.4×单文件仍全部消费 |
| R6 小目标未验命中 | 注入在画面左上半区随机取点，10/10 落在 ROI 外；只统计“任意候选”；`_notify_switch` 空实现 | `_inject_small_target_in_roi`（ROI 内、按原图尺度、保存真值）、`_match_target`（IoU≥0.1 或中心命中）、成对无目标基线、`V33EventMemory` 接入 | `test_r6_injection_lands_inside_roi_at_native_scale`、`test_r6_target_match_requires_spatial_overlap`、`test_r6_paired_baseline_is_reported` | 评审反例 10/10 ROI 交集为 0 → 现在 12/12 注入框落在 ROI 内；独立验证逐目标命中 8/10（09-19）、7/12（09-20），并给出成对基线计数（2/10、5/12） |
| R7 遮挡与局部有效未落实 | 合成 mask 直接取整块 ROI；噪声“留出块”回退同批；发布 `valid` 为整个 ROI（16 张同哈希） | `background.observation_mask_from_frames`（逐帧运动/持续偏差）、`estimate_noise` 优先独立校准块、`build_...py` 发布 `valid = ROI ∩ 观测可用 ∩ 非持续偏差 ∩ 非低支持` | `test_r11_observation_mask_excludes_movers`、`test_r11_moving_objects_do_not_enter_background`、`test_noise_records_independent_calibration`、`test_composite_uses_observation_masks` | v2 13 张 `valid_mask.png` **13/13 哈希互不相同**（旧为 1 种）；`composite[*].observation` 给出可用比例与持续偏差像素 |
| R8 噪声丢掉逐帧 mask | `estimate_noise` 对并集内所有像素写 `weights=1.0` | 权重按逐观测 `keep` 取 max 写入；支持计数与分位只用有效观测；重复块不额外加权 | `test_r8_support_counts_only_valid_observations`、`test_r8_repeated_block_does_not_inflate_support` | 评审反例：左右各半有效 → 旧报两侧支持 2 → 现在两侧均为 **1** |
| R9 零可用仍可准入 | `evaluate_bank_frame` 算了 availability 却没进 outcome；调用方 `verified=True` | `_admission_block_reason`（坏帧/几何失效/无共用视野/可用率低于阈值→阻断）；`_run_tick` 用真实可用性构造 `verified` | `test_r9_bad_frame_blocks_admission`、`test_r9_geometry_invalid_blocks_admission`、`test_r9_zero_availability_from_roi_blocks_admission` | 评审反例 availability=0 且 enter/hold=true → 现在 `enter=False/hold=False`，原因 `BLOCKED_FRAME_CORRUPT`；独立验证报告含 `availability_checks` |
| R10 补偿/残差不一致 + 无候选几何过滤 | 前景直接比较原始 reference；`_support_candidates` 无尺寸限制 | `match.analyze_residual_support`（共享补偿后的残差）、`BankPriorContext.foreground_support(compensated_reference=...)`、`_support_candidates(pixel_scale=...)` 恢复 V3.2 几何门槛 | `test_r10_foreground_uses_same_compensation_as_score`、`test_r10_single_pixel_noise_and_large_blob_are_not_candidates`、`test_r10_small_target_survives_geometry_filter` | 评审反例：1 像素噪点与 120×120 大块都成候选 → 现在两类都被拒（`rejected.too_small/too_large`），8×8 小目标仍保留；adapter 的前景支持与“补偿参考”逐像素相同 |
| R11 每文件独立基准 | sampler 以各文件首帧为基准；外层冻结画布未传入、overlay 未传 | `BoundedPreviewSampler(canvas_reference=..., overlay_exclude_zones=...)`（缺画布直接报错）、`write_canvas_reference` + 报告指纹、逐帧配准诊断 | `test_r11_sampler_requires_frozen_canvas`、`test_r11_cross_file_shift_registers_to_same_canvas` | v2 `frozen_canvas.sha256=3e595d16…`，`registration.applied_to_canvas` 统计；跨文件整数像素平移测试要求配准后残差不劣于原始 |

### 交付表述更正（评审第六节）

1. 报告已分列 `files.planned/attempted/succeeded/failed`（v2：17/17/15/2；独立验证 5/5/5/0 与 4/4/4/0）。
2. **撤回**旧 HANDOFF §6 的“录像 4:3”表述：2560×1440 是 **16:9**；r2 几何偏差的来源是
   沿用示例机位的 16 点、按 16:9 截图目视估计后套到真实帧，不是宽高比算错。
3. 空间分别统计：v2 资产 **3.369 GB**（13×4 文件）、v2 工作目录峰值见 `space.peak_work_bytes`、
   原始缓存峰值见 `space.peak_raw_bytes`、进程内存峰值见下方 §3。
4. 查询 1/2/4 小时成功只说明这些窗口可用；约 6.5 天可获取范围是本次观测。
5. 本轮仍是稀疏试跑（每天 4 个小时槽）；先修算法与口径，再扩大采样。

## 2. 独立验证分区（v2）

| 角色 | 日期 | 是否参与拟合 |
|---|---|---|
| 构建（合成参考） | 09-14、09-15、09-16、09-17（09-13 单条跨日素材解析失败，未参与） | 是 |
| 校准（冻结包络） | **09-18** | 是（仅包络） |
| 独立验证 | **09-19、09-20** | **否**（v2 从未用于合成、包络、剪枝） |

说明：v1 曾用 09-20 做评估，因此 09-20 对**旧**库不再独立；对 v2 它未被任何拟合步骤
使用，报告 `split.declared_role=independent_validation` 记录实际消费日期。

## 3. 独立验证实测（冻结资产，只读包络）

| 指标 | 09-19（5 文件） | 09-20（4 文件） |
|---|---|---|
| 静态潜在覆盖 | 0.85526 | 0.93443 |
| **动态有效覆盖** | **0.63087** | **0.82353** |
| 有效秒 / 可观测秒 | 188.0 / 298.0 | 196.0 / 238.0 |
| 录像空缺秒（不计入分母） | 55,232.5 | 41,508.4 |
| 不可判断秒 | 0.0 | 0.0 |
| 暂停 P95 / 最长 | 9.1 / 30.0 | 15.5 / 18.0 |
| 切换次数 | 6 | 3 |
| 小目标逐目标命中 | 8/10（0.80），成对基线含候选 2 | 7/12（0.583），成对基线含候选 5 |
| 解码秒 / 墙钟秒 | 557.0 / 569.3 | 未单独记录（同流程） |
| 包络来源 | `frozen_in_bank` | `frozen_in_bank` |
| Bank manifest SHA-256 | `cfbdfeac46d237147b550f4aee204c48112675800c3c1fe9e2c795b525e865ee` | 同上 |

**同一 Bank 两次独立评估的包络载荷逐字节相同**，证明盲测数据没有改动冻结参数。

## 4. 动态选 N 的依据（v2）

* 候选：19 个外观组（其中 2 组因参考不足未进入候选），全部进入共同回放。
* 共同回放：105 个 tick，画布 960×540，计划节拍 43.4s（**随源时间轴缩放**，
  不再是“同一秒内 105 个 tick”）。
* 基线（13 候选）：动态有效覆盖 **0.71111**、有效 2778.8s、最长暂停 217.1s、切换 12。
* 留一法对照：每个候选删除后的覆盖/暂停差记入 `pruning.leave_one_out`；
  **删除必须同时满足**覆盖下降 ≤0.005 且最长暂停增加 ≤5s。本轮没有任何候选满足删除条件，
  因此保留 13 个（上限 24 未触及）。**N 由动态对照决定，不再由静态前 16 截断。**
* 候选增删对照（`selection_comparison`）：13 库 0.7111；单候选 0.2222 / 0.3889 / 0.4222 /
  0.5 —— 库规模变化会**实质改变**动态覆盖，说明选择过程有区分力。
* 诚实限制：本轮没有候选被剪掉，所以“剪枝最优性”只有机制证明（单元测试构造的冗余/过渡
  候选），没有真实素材上的删减证据。

## 5. 尚未解决 / 已知限制

* **小目标成对基线存在噪声**：09-20 有 5/12 的无目标基线也出现候选，说明部分“命中”
  可能来自环境残差。需要更多真值样本才能给准确率；本报告只声明机制与逐例结果。
* **独立验证仍是稀疏试跑**：每天 4–5 个小时槽、0.05 FPS，不能外推全天或现场准确率。
* **09-13 单条跨日素材解析失败**：远端 URL 已失效（`RecordingSourceError`），
  报告如实记为 `failed`，未静默改成成功；分区仍以 09-14 起算。
* **暂停与切换时延是离线虚拟时钟**：`wall_seconds` 是 I/O 成本，不是在线恢复速度；
  active prior + semantic 的整 tick 时延、显存与主流 FPS 仍属方案二 B6 服务器验收。
* **低支持候选**：v2 仍按 `low_support` 标注，但“更严格进入门槛/更小可用范围”的
  运行时消费属于方案二 B3/B4，本轮只在资产中保留该字段与诊断。
* **未做**：生产 API/manager/worker 接入、在线切换、跨机位迁移、真实摄像机小目标对照。

## 5.1 一轮自审发现并修掉的问题

重新验收过程中我又发现并修掉了 4 个会直接影响结论的问题（都已进回归）：

1. **回放时间轴退化**：所有回放帧都取文件开头，导致 105 个 tick 落在同一秒内，
   覆盖率分母只有 3.6s。改为按文件时长均匀抽帧，并用真实相邻间隔作为计划节拍
   （v2 最终 `nominal_tick_seconds=43.4`）。
2. **回放画布与参考尺寸不一致**：参考被缩放到 960×540 而帧仍是 2560×1440，
   `score_profile` 抛错被 catch 掉，所有 tick 变成 `NO_ELIGIBLE_PROFILE`。
   现在帧与参考统一缩放到回放画布。
3. **观测间隔上限不随节拍缩放**：离线回放与低分析频率下，4s 默认值每 tick 清空
   连续证据，恢复永远无法完成。`observe()` 现在按实际 tick 间隔自适应到至少 2×，
   并有专门回归。
4. **动态剪枝可能清空全部候选**：留一法显示每个候选都可删时会发布 0 个 Profile。
   现在保留贡献最大者并写 `ALL_CANDIDATES_REDUNDANT` 警告，绝不发空库。

前 3 项都会让“动态覆盖”看起来像算法结论，实际是接线缺陷；这也是评审 R2/R3 的核心。

## 6. 复现命令

```bash
# 本地确定性测试（R1～R11 回归 + 原工厂测试）
.venv-profile/bin/python -m pytest tests/test_ground_litter_profile_repairs.py -q   # 32 passed
.venv-profile/bin/python -m pytest tests/test_ground_litter_profile_factory.py -q   # 78 passed
.venv-profile/bin/python -m pytest tests -q                                        # 742 passed（4 项缺 torch）

# 服务器：构建 v2（构建 09-14~09-17 / 校准 09-18）
ssh -p 21002 sf01@14.21.88.97
cd ~/profile-factory && ./run_v2.sh v2 4 40

# 服务器：独立验证（09-19、09-20，冻结 Bank，只读包络）
./run_v2_eval.sh
```

产物：`/home/sf01/profile-factory/out/bank/camera_01030/v2/`（`bank.json`
SHA-256 `cfbdfeac…`）、`out/eval_v2indep/`、`out/eval_v2indep20/`；本地副本见
`output/profile_factory_v2_20260920/`。
