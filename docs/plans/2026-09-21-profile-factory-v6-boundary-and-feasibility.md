# 方案一最后一轮：回放边界 + 剪枝可行性 + 能力契约 + 诊断口径

> **已暂停且不得恢复（2026-09-21）**：本文对应的代码草稿没有完成回归或 v6 重建；随后 oracle Go/No-Go 实验已判定背景 Profile 不具备通用小垃圾召回能力。保留现场用于追溯，不执行本文余下步骤。见[正式 NO-GO 决策](../decisions/2026-09-21-ground-litter-profile-prior-no-go.md)。

状态：历史暂停计划。
目标：修复三个已确认缺陷，冻结可复现输入，最多执行一次 v6 重建。

## 0. 已确认的事实（来自 v4/v5 报告）

| 项 | v4 | v5 |
| --- | --- | --- |
| build_days | 09-14~09-17 | **09-13**~09-17 |
| 构建文件 | 16 | 17 |
| 查询范围 | `start=09-14 00:00:00` | 同 |
| 分组 centers | 18 | 16 |
| 候选 | 13 | 10 |
| suitable | 2 | 0 |
| 删除轮数 | 11 | 0 |

v5 唯一的多余文件是 `2026-09-13 23:58:12` 开始的跨日文件：查询起点是 09-14 00:00:00，
接口返回了它，代码用 `record_start` 推导日期，把 09-13 直接算成构建日。
所以 v4/v5 **输入不同**，"同参数重建不稳定"这个结论目前不成立，本轮先修边界并把输入冻结。

## 1. 回放请求边界与不可变输入清单

### 1.1 半开区间与 eligible 窗口

* 任务范围 `[start_time, end_time)`；文件与任务区间**有交集**才保留身份。
* 每个文件计算并保存（新增 `EligibleWindow`）：
  * `eligible_start_offset` / `eligible_end_offset`（文件内秒，`absolute = record_start + offset`）；
  * `effective_start_time` / `effective_end_time`（绝对时间字符串）；
  * `intersection_seconds`。
* 完全在任务范围外（含"结束时间早于 start"或"开始时间 ≥ end"）→ **剔除**，
  在报告里记录数量与原因 `before_range` / `after_range` / `zero_length`。
* 所有取帧路径（粗采样、高清按需解码、回放收集、校准包络拟合）都必须落在
  eligible 窗口内：
  * `BoundedPreviewSampler.sample_file(..., eligible_start_offset=, eligible_end_offset=)`
    只在该窗口内生成粗采样/加密偏移；
  * `_collect_replay_frames` 用窗口裁时长，并只对窗口内偏移取帧；
  * `fit_frozen_envelopes` / `_score_calibration_file` 同样传入窗口。
* 分区依据**实际采样帧的绝对时间**（`AppearanceSample.frame_time_seconds` +
  `record_start`），不再只看文件 `record_start`：跨日文件里属于 09-14 的帧进
  build（09-14），不属于任何构建/校准/盲测日的帧进 `outside`。
  文件级角色由它**实际被采到的帧**决定，报告里保留逐帧归属。

### 1.2 清单缓存键

`inventory_cache_path` 必须包含：deviceCode、start/end、endpoint（source identity）、
时区、schema/version。新增 `INVENTORY_CACHE_SCHEMA = 2`。
缓存文件里记录同样字段；读取时逐字段比对，不一致就**拒绝复用并重新查询**，
不允许把别的工作目录或别的时间范围的清单当本轮输入。

### 1.3 不可变 `input_manifest.json`

新增 `--input-manifest PATH`：

* 生成时机：首次规划（inventory → eligible → 抽样计划）后写出；
* 内容：deviceCode / camera_id / timezone / 请求范围 / 每个文件的稳定 fileId、
  record_start/end、fileSize、eligible 窗口、角色（build/calibration/blind/outside）、
  计划采样偏移与绝对时间、下载后补写的源文件 SHA-256、采样配置、算法版本、
  配置散列、manifest 自身 SHA-256；
* 使用该入口重跑时：
  * **不重新查询、不重新选文件、不重新选偏移**；
  * 只允许为相同 fileId 刷新临时下载 URL；
  * 文件不可获取 → 明确失败，**不替换**成别的文件；
  * 角色、采样点、预期分组输入必须稳定；
  * 与当前 `deviceCode/camera_id/时区/范围` 不一致时拒绝。

## 2. 剪枝：可行性筛选 + 未满足上界报告

现状两个错误：

1. 初始候选全 `prior_suitable=false` 时，删任意候选都 `prior_would_be_empty=true`，
   prior 约束永远阻断 → v5 的 10→10、0 次删除。
2. 每轮只看排序第一名；第一名被 prior 约束挡住就整体停止，不再看第二名。

正确算法：

* `initial_prior_capable = bool(initial_capability_ids)`；
* 每轮对**所有**候选算删除后的完整指标，逐方案判定 `feasible`：
  * match 覆盖下降 ≤ 阈值、match 暂停增量 ≤ 阈值；
  * 若 `initial_prior_capable`：prior 覆盖下降 ≤ 阈值、prior 暂停增量 ≤ 阈值、
    删除后 prior-capable 集合非空；
  * 若 `initial_prior_capable == False`：保持 `semantic_only=true`，
    **不**用"prior 集合非空"阻断环境冗余剪枝，仍报告 prior 覆盖 0 / prior 不可用；
* 在所有 `feasible=true` 中取代价最低者；只有**本轮没有任何 feasible 方案**才停止；
* `_trim_to_resource_limit` 用完全相同的规则；若在约束下无法满足 `max_profiles`，
  报告 `RESOURCE_LIMIT_UNSATISFIED`，不偷偷违规、不假装成功。

## 3. 能力契约补完

每个新 Profile 显式包含 `match_eligible` / `prior_suitable` / `calibration_state` /
`prior_degradation_reason`。Loader 默认校验四项存在且自洽：

1. `calibration_state == independent_matched` ⟺ `prior_suitable == true`；
2. `reference_self_low_support` / `no_independent_material` ⟹ `prior_suitable == false`；
3. `prior_suitable == true` ⟹ `match_eligible == true`；
4. `independent_matched` 的 degradation reason 必须为空；
5. 其他状态必须有明确 degradation reason；
6. legacy 模式显式开启：`prior_suitable=false`、`capability_source=legacy_conservative`、
   不产生 prior 候选。

Bank 提供 `match_eligible_ids` / `prior_suitable_ids`（后者为前者子集）。
Selector 的搜索候选只放 `match_eligible=true`，能力集合取 `prior_suitable_ids` 的交集。

## 4. 诊断口径

A/B/C 报告分开记录：raw seed pixels / raw support pixels / 成框后的 matched
candidate 与 support / 尺寸过滤计数 / 生效参考是否被选中 / `prior_available` 是否允许输出。
armB 在线 3px 有 9 个原始 support pixels 但未成框，文档必须如实写"9 raw support,
0 matched candidate"，不能写成"支持像素为 0"（原始 JSON 不改）。

A/B/C 改为一次加载、一次逐 Profile 计算：共享源帧、注入图与逐 Profile 残差结果，
不同集合只重放自己的 Selector，避免三份重复的 15–20 分钟诊断。

## 5. 可复现性两级验证

* 级别 A（不下载、不完整重建）：同一冻结 manifest 跑两次输入规划与分组，
  比对文件与角色、采样绝对时间与偏移、sample identity、sample manifest SHA、
  分组成员集合、representative、候选能力分类；Profile ID 可重编号，
  但按成员集合计算的 group signature 必须一致。
* 级别 B：级别 A 通过后才允许一次 v6 重建；先确认录像仍能按稳定 fileId 获取，
  过期就停止并报告，不换一批文件冒充同输入复验。

## 6. 执行顺序与效率

1. 本计划 + 失败反例；
2. 改代码（§1–§4）；
3. 相关测试一次；
4. 用保存的 JSON/矩阵做离线预演（分钟级）；
5. 同一 manifest 双跑确定性（级别 A）；
6. 级别 A 通过后服务器**一次** v6 重建；
7. 最后一次相关测试 + 一次全量测试。

禁止：重复完整重建"看看结果"、三次重复跑 A/B/C、`pkill -f`、长同步 sleep、
没有冻结 manifest 就启动重建、每改一点跑一次全量测试。
长任务用后台 job + 短轮询/日志增量；失败先查根因。
