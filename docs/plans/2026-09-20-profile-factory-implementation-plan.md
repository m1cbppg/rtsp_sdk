# 方案一实施计划（A0～A6 + 共享核心 B1/B2）

日期：2026-09-20。契约来源：`2026-09-20-profile-factory-v1.md` r3、`2026-09-20-profile-runtime-selector-v1.md` r3、
`2026-09-20-profile-bank-design-review.md`（r1 问题为历史发现，r2 已修订正文）。

本文件是实施前的计划提交，随后按此顺序执行。范围：方案一 A0～A6，另加方案一必需的共享
bank loader、matcher、Bank prior adapter、纯 Selector 与离线证据衔接。生产 API/manager/worker
接入、线上自动切换、生产部署留给方案二 B3/B4/B6，本次不做。

## 1. 模块与依赖顺序

| 顺序 | 新增/修改模块 | 内容 | 依赖 |
|---|---|---|---|
| 1 | `rtsp_annotator/ground_litter_profile_bank.py`（新） | A0 资产 schema、loader 校验、原子版本发布、SHA-256、有界上下文缓存、Bank root 解析 | 无 |
| 2 | `rtsp_annotator/ground_litter_profile_match.py`（新） | A0/B2 16×9 分块描述子、公共稳健尺度、受限颜色补偿、残差、S(p) 评分、进入/保持包络、几何/锚点判定 | 1 |
| 3 | `rtsp_annotator/ground_litter_recording_source.py`（新） | A1 回放 `file-urls` 列表解析（fileId/真实起止/字符串 fileSize）、临近刷新、稳定身份去重、有界下载（.part、边写边 SHA-256、Range 能力探测、断点续传仅在实测一致后） | 无 |
| 4 | `rtsp_annotator/ground_litter_recording_cache.py`（新） | A1 SQLite 作业/材料化状态机、1GiB raw 预算 + 总预算、下载/预取/处理槽、租约、阶段事务提交、提交后删除、崩溃恢复、源散列变化失效、按需重拉 | 3 |
| 5 | `rtsp_annotator/ground_litter_profile_sampling.py`（新） | A2 有界租约消费、粗采样 4 时刻、质量过滤、配准到共同画布、分块外观特征、时间块与集合（构建/校准/盲测）划分 | 1,2,4 |
| 6 | `rtsp_annotator/ground_litter_profile_background.py`（新） | A3 代表点分组（最远点 + medoid 重分配）、时间均衡高清选帧、遮挡/运动掩膜、时间中位数合成 + 真实观测块替换、有限噪声（中心/MAD/Q95/超 cap/正 epsilon/bias_flag 诊断） | 1,2,5 |
| 7 | `rtsp_annotator/ground_litter_profile_analysis.py`（新） | A4/B2 三类掩膜（fit_mask / foreground_support / availability_mask）与可枚举失效原因、Bank prior adapter、静态潜在覆盖 | 1,2 |
| 8 | `rtsp_annotator/ground_litter_profile_selector.py`（新） | B1 纯有状态 Selector：Top-K → 有界全库扩展、游标、冷却、恢复、驻留、迟到/跳过、预算不足原因 | 无（纯逻辑） |
| 9 | `scripts/build_ground_litter_profile_bank.py`（新） | A5 总入口：索引 → 抽样 → 分组 → 合成 → noise → 静态预选 → Selector 动态定稿/剪枝 → 冻结 Bank + 报告；`--resume` 可续跑 | 1–8 |
| 10 | `scripts/evaluate_ground_litter_profile_bank.py`（新） | A5 离线评估：静态/动态覆盖、暂停 P95/最长、切换/确认时延、小目标对照、误报样例、峰值空间与流量 | 1–8 |
| 11 | `scripts/inventory_ground_litter_ps.py`（扩展） | A1 索引清单支持 ctseelink 远程来源 | 3 |
| 12 | `tests/test_ground_litter_profile_bank.py` 等（新） | 见 §3 必测清单 | 全部 |

## 2. 关键设计决定（与文档条款对应）

1. **fileId 与 URL 分离**：清单只保存 `fileId/record_start/record_end/size`；签名 URL 只存在于
   下载上下文内存中，不写日志/报告/Bank。`redact_url` 复用现有常量脱敏。
2. **有效期**：`issued_monotonic + urlExpireSeconds` 保守计时；开始下载前剩余 <30s 即刷新；
   排队不跨有效期。刷新后 URL 变化不产生新录像条目。
3. **材料化状态与阶段状态分开**：`ABSENT/DOWNLOADING/READY/LEASED/EVICTABLE` 与
   `preview/hd_extract/calibration_replay/blind_replay` 两组字段；`preview` 完成不代表文件任务完成。
4. **删除条件**（全部满足才删）：受管缓存内 + 无租约 + 当前阶段产物已落盘并校验 +
   阶段已事务提交 + 有重拉路径。绝不调用远端删除，绝不删除用户本地源。
5. **空间预算**：`raw_cache_budget`（默认 1GiB，含 .part/预取/处理中/失败残留，按 fileSize 预留）
   与 `work_budget`（默认 20GiB，含高清样本/资产/报告）。预算是硬上限，超限停止派发并报背压。
6. **三类掩膜分开**：`fit_mask`（拟合/评分）、`foreground_support`（前景证据）、
   `availability_mask`（可判断性，原因可枚举）。目标尺度残差不直接产生 invalid；
   只有独立证据（actor 遮挡、坏帧、配准失败、大范围成像损失）才使 availability=false。
7. **噪声**：`raw_T = max(T_base, m + max(k*MAD, epsilon))`，`stored_T = min(raw_T, T_cap)`，
   `bias_flag = raw_T > T_cap`；保留残差中心/MAD/Q95/超 cap 比例/支持时间块数。
   恒定大残差但 MAD=0 会被残差中心捕获（F3 回归）。
8. **N 由完整 Selector 定稿**：静态贪心只产预选；动态回放同一份 Selector 计算有效覆盖、
   暂停、切换次数，删冗余前先确认过渡暂停不恶化，只在过渡起作用的参考保留（F4）。
9. **跨文件连续**：回放时 Selector/V33 memory/虚拟时钟跨五分钟文件保持；首尾重叠按源时间去重，
   真实缺口按缺帧处理；下载等待单独计时，不混入源时间轴。
10. **不碰生产**：本次不改 `api.py`/manager/worker 请求契约，不创建线上流，不重启服务。

## 3. 测试与验收

确定性测试（`tests/test_ground_litter_profile_*.py`）：

- 来源与缓存：120s 到期刷新、重复/重叠窗口按 fileId 去重、列表截断检测、fileSize 字符串、
  200 但错误正文、下载中断、Range 不支持、磁盘背压、租约保护、提交前后崩溃恢复、
  preview 完成后高清重拉、源散列变化失效、用户本地 PS 不被清理、URL 不落盘。
- 数据划分与时间：前五天/第六天/第七天不泄漏、跨文件状态连续、缺帧不填成正常、
  下载等待不混入源时间。
- 匹配与选择：第 K+1 个才可用、前几名高清失败后扩展搜索、粗匹配失败不饿死、
  冷却后补足名额、预算不足有明确原因、静态与动态覆盖差异、过渡参考不被错误剪枝。
- 掩膜与噪声：小纸片不被局部 invalid 或噪声门槛吞掉、恒定大残差 MAD=0 被诊断、
  fit/foreground/availability 分离。
- 旧模式回归：现有固定 Profile 模式（`CleanReferenceV32Processor` / `propose_v32`）
  行为不变（跑现有测试，不改语义）。

真实素材验证（需授权）：先 1 小时、再 24h；记录下载/解码耗时、实际流量、峰值磁盘、
URL 过期恢复与清理。真实 PS 验证与合成测试结论分开报告；未执行项标为“未验证”。

## 4. 阶段交付物

A0/A1 完成后即可单独运行索引与下载试点；A2/A3 产出候选与基础噪声；
A4/B1 产出 N 的选择过程与静态/动态覆盖；A5 产出可续跑构建与评估 CLI；
A6 产出冻结 Bank + manifest + 报告 + HANDOFF（含完成/未完成/已知问题/方案二接入接口）。
