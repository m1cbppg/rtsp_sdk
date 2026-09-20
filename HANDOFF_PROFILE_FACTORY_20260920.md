# HANDOFF：方案一 Profile 工厂 + 共享 matcher/Selector（2026-09-20）

本文交接方案一 A0～A6 与方案一必需的共享核心（bank loader、matcher、Bank prior
adapter、纯 B1 Selector、离线证据衔接）。生产 API/manager/worker 接入、线上自动切换、
生产部署属于方案二 B3/B4/B6，本次**未做**。

契约来源：`docs/plans/2026-09-20-profile-factory-v1.md`（r3）、
`docs/plans/2026-09-20-profile-runtime-selector-v1.md`（r3）、
`docs/plans/2026-09-20-profile-bank-design-review.md`。
实施计划：`docs/plans/2026-09-20-profile-factory-implementation-plan.md`。

## 1. 状态分类（严格区分，不含预计结果）

| 类别 | 结论 |
|---|---|
| 代码完成 | A0～A6 与 B1/B2 共享核心已实现，共 8 个新模块 + 4 个脚本 + 2 个测试文件 |
| 确定性测试通过 | 新增工厂测试 **78/78 通过**；项目全量 **710 passed + 40 subtests**，仅 4 项 `tests/test_shared_inference.py` 因本地验证环境**未安装 torch** 失败（与本次改动无关）。见 §9 的环境说明 |
| 真实 PS 验证 | **通过（受预算限制）**：服务器上完成 6 天素材建库，产出 16 个 Profile 的不可变 Bank，并在**独立日**（09-20，未参与构建）完成连续回放评估；详见 §5 与 §11 |
| 生产性能通过 | **未验证**。本次不创建生产流、不接入 API、不做服务器性能测试 |

## 2. 交付物

### 2.1 新增模块

| 文件 | 作用 |
|---|---|
| `rtsp_annotator/ground_litter_profile_bank.py` | A0：Bank 资产 schema、loader 校验（散列/尺寸/量纲/有效地面）、原子版本发布、`BoundedContextCache`（张数+字节双上限）、旧 V3.2 二值 tolerance → 连续 noise 的兼容转换 |
| `rtsp_annotator/ground_litter_profile_match.py` | A0/B2：16×9 分块描述子、公共稳健尺度、受限全局增益/偏移（保留原始需求）、`residual_maps` 同量纲残差、S(p) 评分、进入/保持包络、全库粗排序 |
| `rtsp_annotator/ground_litter_recording_source.py` | A1：`ctseelink/playback/file-urls` 列表解析、稳定身份去重、一小时/半小时截断检测、有效期保守刷新策略、有界 `.part` 下载（边写边 SHA-256）、Range 探测与实测续传、`file_looks_like_media` 内容探针 |
| `rtsp_annotator/ground_litter_recording_cache.py` | A1：SQLite 材料化状态机（ABSENT/DOWNLOADING/READY/LEASED/EVICTABLE）、四类阶段事务、租约、raw/work 双预算、满足全部条件才删除、崩溃恢复、源散列变化失效、按需重拉 |
| `rtsp_annotator/ground_litter_profile_sampling.py` | A2：有界租约消费、粗采样 4 时刻、质量过滤、配准到共同画布（含恒等回退）、时间块、构建/校准/盲测划分、高清需求清单 |
| `rtsp_annotator/ground_litter_profile_background.py` | A3：最远点 + medoid 重分配分组、时间均衡选帧、分块时间中位数合成 + 真实观测块替换、残差中心/MAD/Q95/超 cap/bias 诊断的有限噪声 |
| `rtsp_annotator/ground_litter_profile_analysis.py` | A4/B2：`fit_mask`/`foreground_support`/`availability_mask` 三类分离、可枚举失效原因、Bank prior adapter、静态潜在覆盖 |
| `rtsp_annotator/ground_litter_profile_selector.py` | B1：纯有状态 Selector（Top-K → 有界全库扩展游标、冷却、驻留、恢复跨度、提交/失败反馈、预算不足原因、动态覆盖/暂停统计） |

### 2.2 新增脚本

| 脚本 | 用途 |
|---|---|
| `scripts/build_ground_litter_profile_bank.py` | A5 总入口：索引 → 有界下载 → 粗采样 → 分组 → 高清重拉/合成 → 噪声 → 静态预选 → Selector 动态定稿/剪枝 → 冻结 Bank + 报告 |
| `scripts/evaluate_ground_litter_profile_bank.py` | A5/B5：冻结 Bank 的连续回放评估（静态/动态覆盖、暂停、切换、缺口、小目标对照、误报样例、性能与空间） |
| `scripts/inventory_ground_litter_ps.py` | A1：清单索引（远程/本地），**不**下载媒体、**不**写签名 URL |
| `scripts/pilot_ground_litter_playback_cache.py` | A1 真实试点：清单 → 刷新 → 有界下载 → Range/续传 → 解码 → 粗采样 → 清理与峰值空间 |

### 2.3 修改的既有模块

* `rtsp_annotator/ground_litter_v32.py`：抽出 `compute_compensation_masks`、
  `local_luminance_field`、`_classify_environment` 三个纯函数，`protected_normalize`
  改为调用它们。**行为逐字节不变**（有专门回归测试比较两者输出），旧固定 Profile 模式
  与 V3.2/V3.3 全部既有测试继续通过。
* `rtsp_annotator/ground_litter_v33.py`：新增公开方法
  `V33Event.reset_cross_reference_evidence()` / `invalidate_previous_support()` /
  `V33EventMemory.event_availability()` / `per_event_support()` / `notify_reference_switch()`，
  以及 `reference_generation` / `reference_profile_id` 字段。**不修改既有 tick 逻辑**。

### 2.4 测试

* `tests/test_ground_litter_profile_factory.py`（78 项）
* `tests/profile_bank_fixtures.py`（合成夹具；无网络、无真实素材）

## 3. 可直接运行的命令

```bash
cd /Users/mlcbppg/Desktop/backend/python_script/rtsp
source .venv/bin/activate

# 全量确定性测试
python -m pytest tests -q

# 只跑本次新增工厂/Selector/掩膜/回归测试
python -m pytest tests/test_ground_litter_profile_factory.py -q

# 编译检查
python -m compileall -q rtsp_annotator tests scripts

# 清单索引（本地目录，不下载媒体）
python scripts/inventory_ground_litter_ps.py --source local \
  --input /path/to/ps_dir --output output/profile_bank_index.json

# 清单索引（远程回放接口，需要网络；不写签名 URL）
python scripts/inventory_ground_litter_ps.py --source ctseelink-file-urls \
  --device-code 44180209031322001030 \
  --start "2026-09-15 00:00:00" --end "2026-09-15 01:00:00" \
  --output output/profile_bank_index.json --check-truncation

# 真实 PS 试点（下载 + Range/续传 + 解码 + 清理；报告只含脱敏字段）
python scripts/pilot_ground_litter_playback_cache.py \
  --output output/pilot/report.json --work-dir output/pilot/work \
  --max-downloads 1 --test-resume

# 本地 PS 建库（可续跑；--geometry 必须来自同一摄像头同一 view）
python scripts/build_ground_litter_profile_bank.py \
  --input /path/to/ps_dir --camera camera_01 --version v1 \
  --geometry config/ground_litter_<camera>_geometry.json \
  --output output/profile_bank --work-dir output/profile_bank_work \
  --raw-cache-budget-gib 1 --work-budget-gib 20 --resume

# 远程回放接口建库（临近下载时刷新 URL）
python scripts/build_ground_litter_profile_bank.py \
  --source ctseelink-file-urls --device-code 44180209031322001030 \
  --start "2026-09-15 00:00:00" --end "2026-09-15 01:00:00" \
  --camera camera_01 --version v1 --geometry geometry.json \
  --output output/profile_bank --work-dir output/profile_bank_work \
  --raw-cache-budget-gib 1 --work-budget-gib 2 --max-downloads 6 --resume

# 评估冻结 Bank（连续回放，跨文件不重置 Selector）
python scripts/evaluate_ground_litter_profile_bank.py \
  --bank-root output/profile_bank --bank-id camera_01 --version v1 \
  --input /path/to/ps_dir --output output/profile_bank_eval \
  --analysis-fps 0.5
```

`--geometry-from-config <现有流配置>` 可从已有 `config/ground_litter_*.json` 复用同一
摄像头的 ROI/排除区（本机可用的 1021 配置面向设备 `...01021`，与本次试点设备
`...01030` **不是同一路**，不得混用）。

## 4. 确定性测试覆盖（与文档必测清单对应）

| 要求 | 测试 |
|---|---|
| URL 过期刷新 | `RefreshPolicyTests::test_refresh_when_remaining_below_margin`、`test_refresh_window_lookup_by_file_id` |
| 下载中断 / 错误正文 / 重复 / 截断清单 | `DownloadTests`（3 项）、`UrlListParsingTests::test_http_200_with_error_body_is_rejected`、`test_deduplication_by_device_and_file_id`、`test_truncation_detection` |
| 磁盘背压 | `BudgetAndLeaseTests::test_backpressure_blocks_dispatch`、`test_backpressure_reported_when_raw_files_exceed_budget`、`test_work_budget_covers_hd_samples` |
| 租约保护 | `test_lease_protects_file_from_deletion` |
| 提交前后崩溃恢复 | `test_crash_before_commit_does_not_release_source`、`test_crash_recovery_is_idempotent`、`test_recovery_marks_missing_file_absent` |
| preview 后高清重拉 | `test_preview_commit_is_not_file_task_completion`、`test_commit_required_before_release` |
| 源版本变化 | `test_source_version_change_invalidates_artifacts`、`test_stage_key_includes_input_hash_and_versions` |
| 本地源不被清理 | `test_unmanaged_local_source_is_never_deleted` |
| 时间隔离 / 跨日 | `SamplingTests::test_partition_keeps_days_isolated`、`SelectorGapTests` |
| 第 K+1 个才可用 | `SelectorTests::test_kth_plus_one_candidate_is_eventually_reached`、`test_expanded_search_advances_cursor_across_ticks`、`test_expanded_search_confirms_k_plus_one_over_multiple_ticks` |
| 前几名高清验证失败后扩展 | `test_failed_full_size_verification_cools_candidate` |
| 小纸片不被吞 | `AnalysisMaskTests::test_small_paper_is_not_swallowed_by_local_invalid` |
| 恒定大残差 MAD=0 | `BackgroundTests::test_noise_flags_constant_large_residual_with_zero_mad` |
| 噪声上限保留小目标 | `test_noise_keeps_small_target_threshold_low` |
| 静态/动态覆盖差异 | `SelectorTests::test_static_and_dynamic_coverage_differ_on_alternating_appearance` |
| 过渡参考不被错误剪枝 | `test_transition_reference_can_reduce_pause` |
| 缺帧不填成正常 | `SelectorGapTests::test_off_air_gap_is_not_counted_as_pause` |
| 现有固定 Profile 模式未退化 | `V32RegressionTests`（3 项）+ 既有 `tests/test_ground_litter_v32_production.py`、`tests/test_ground_litter_v33*.py` |
| 端到端构建/评估/索引 | `CliEndToEndTests`（2 项） |

## 5. 真实输入试点结果（实测）

设备 `44180209031322001030`，窗口 `2026-09-15 00:00:00`–`01:00:00`。

| 项目 | 实测值 |
|---|---|
| 清单请求 | HTTP 200、业务 code=200、12 项；耗时约 1.95s |
| 清单范围 | 返回 `2026-09-14 23:59:44`–`2026-09-15 01:00:31`（跨出查询区间） |
| 声明总大小 | 749,501,533 bytes |
| 记录长度 | 全部 303–304s（12 项），未出现恰好 300s |
| 截断检测 | 一小时 12 项 vs 两个半小时并集 12 项，`suspected_truncation=false` |
| **等待 380s 后重复查询** | 12 项 fileId/大小/起止时间完全一致；**12 个 URL 全部更新**；有效期仍为 120s |
| 单文件下载 | 62,224,552 bytes（与声明一致），2.834–3.625s，约 **17–22 MB/s** |
| 内容探测 | 通过（HEVC/MPEG-PS） |
| Range | `bytes=0-0` 返回 **206 + Content-Range** → 支持 |
| 断点续传 | 保留前半段后续传成功，最终 SHA-256 与整文件一致（`a5cd657b…`） |
| 解码 | 2560×1440、HEVC、时长 303.958s |
| 粗采样 | 3 个顺序时间点全部命中（6.0/150.0/297.0s），PS 时间戳重定基后可用 |
| 峰值工作目录 | 62,269,608 bytes（单文件）；释放后回落到 45,071 bytes |
| 清理 | preview 提交后释放成功，无租约残留 |

**重要实测发现**：PS 容器的 PTS 基值不是 0（本例首帧约 10394s），且同一文件内 PTS
可重置。`SequentialFrameReader` 现在统一把时间戳重定基为文件内相对秒；绝对时间只来自
清单记录。忽略这一点会让所有采样点都命中失败。

**未验证**：
* 多文件「整链路建库」的完整结果（见下）。
* 远端录像保留期限与「跨天重拉」能力：只证明同一小时内刷新可用，**未**证明更早录像
  仍可重取。
* `sample_with_seek`：本机实测在真实 PS 上返回空结果（PyAV 16.1.0/macOS），
  因此 CLI 仍以顺序解码为默认；seek 性能未通过。
* 120s 边界：本次下载都在刷新后立即开始（余量充足），**没有**故意等到剩余 <30s 再试。

## 6. 已知问题与限制

1. **ROI 缺项**：本次试点设备（`...01030`）没有可信的七天日期范围与已审核 ROI。
   仓库里唯一可用的多边形 ROI 属于 `...01021`（`config/ground_litter_1021_demo.json`、
   `config/ground_litter_v32_stream_request.example.json`）。按要求**没有**把它套到
   `...01030` 上；未提供 `--geometry` 时 CLI 使用整幅画面并在报告写
   `geometry_warning`。**因此本次任何有效地面/覆盖数字都不是现场验收。**
2. **动态覆盖数字不可判读为现场结论**：工厂动态定稿使用显式虚拟时钟与零加载时延；
   报告明确标注。在线恢复速度必须由方案二 B6 在服务器上实测。
3. **噪声资产分辨率**：阈值图与参考同尺寸（Bank loader 契约）。为控制内存，早期草稿
   曾按 stride 降采样，现已统一为全分辨率；`stride` 参数仍保留但建库固定为 1。
4. **分组来源归属**：`_match_frames_to_candidates` 目前按顺序轮转把构建帧分配给候选，
   没有把「哪些文件属于哪个外观组」的真实归属回填。候选数 >1 时，静态预选与动态回放
   的帧分配是近似，不影响机制正确性，但会让多候选的覆盖统计偏保守。
5. **未接入 V33 事件 memory 的在线回放**：`evaluate_ground_litter_profile_bank.py` 回放
   Selector 与匹配，但没有实例化 `V33EventMemory` 去产出事件级「确认时延/清走」。
   方案二 B3 接入时需补这一段（v33 的公开方法已就绪）。
6. **`frame_count`**：真实 PS 的 PyAV 流 `frames=0`，节拍靠时长×实际解码统计推断，
   报告里也不把 frame_count 当权威。
7. **`_resume_test` 会重建 `.part`**：它把已就绪文件切一半做续传实测，属于试点工具的
   自检行为，只对第一个文件执行。

## 7. 方案二接入接口（下一步）

已冻结并可直接消费：

```python
from rtsp_annotator.ground_litter_profile_bank import load_bank, BoundedContextCache
from rtsp_annotator.ground_litter_profile_analysis import build_prior_context, evaluate_bank_frame
from rtsp_annotator.ground_litter_profile_match import (score_profile, evaluate_match,
    envelope_from_samples, rank_by_coarse_distance, extract_grid_descriptor)
from rtsp_annotator.ground_litter_profile_selector import ProfileSelector, CandidateMatch
```

接入顺序建议（对应方案二 §9）：

1. **B2**：用 `BoundedContextCache` 替换单参考上下文；用 `evaluate_bank_frame` 替换
   固定环境门槛。注意 `evaluate_bank_frame` **不**修改事件 memory，符合「最新帧验证」
   要求。旧的 `local_extent > 16` 一类硬悬崖必须由 Bank 包络取代（本次已提供原始
   `gains/biases/local_extent` 诊断，用于观测裁剪后的原始需求）。
2. **B3**：提交边界调用 `V33EventMemory.notify_reference_switch(...)`，它保留事件
   身份、清空 pending/fused/clean 窗口并递增 `reference_generation`；同 `profile_id`
   恢复也走同一路径。
3. **B4**：`ProfileSelector.plan_tick(ts, current_hold_eligible=...)` →
   `observe(...)` → 全尺寸验证 → `commit()/fail()`。`discard_stale` 用于拒绝过期结果。
4. **B5**：用 `scripts/evaluate_ground_litter_profile_bank.py` 的同一份 Selector 做
   第六天校准/第七天冻结回放，并把 V33 memory 接进 `_run_tick`。
5. **B6**：服务器实测暖/冷加载、active prior + semantic 的整 tick P95、主流 FPS 与
   输入帧年龄；离线报告里的 `performance` 段是 I/O 成本，不是在线时延。

API 字段（`profile_bank_id` 互斥 `profile_id`）、manager/worker 传递与 metrics
**本次未实现**，按方案二 §8 执行。

## 8. 建议的下一步（按优先级）

1. 取得 `...01030` 的**已审核 ROI**与明确七天日期范围，然后按 §3 的本地/远程命令
   重跑建库；这是把「代码完成」推进到「真实素材验收」的唯一前提。
2. 用同一份 Bank 跑第七天盲测（`--input` 指向第七天目录），报告改动幅度而不是绝对准确率。
3. 接入 V33 memory 到评估脚本，补齐事件确认时延与清走保护的真实数字。
4. 在多候选（≥3 组外观）真实素材上复核分组归属与动态剪枝。
5. 把 `--geometry` 输出整理成受控几何 JSON 模板，避免每次都手抄多边形。


## 9. 本地环境事故与恢复（必须知道）

本次在本地做真实 PS 试点时，`/tmp` 所在数据卷一度被占到 98%，导致：
1. `.venv/lib/python3.12/site-packages/numpy/version.py` 被截断为 0 字节（已用等价内容修复为
   numpy 2.5.2）；
2. `.venv` 里的 `cv2` 包在随后导入时卡死在内核 `read`（疑似被截断的 `.dylibs`），
   重新安装未能在 10 分钟内完成，因此 `.venv` 的 `cv2` 目前**缺失**。

已删除全部本地试点素材（`/tmp/ps_real_work*`、`/tmp/build_real_work`、`/tmp/ps_pilot` 等），
本地磁盘已释放到约 15GiB 可用。

**恢复 `.venv`（任选其一）**

```bash
cd /Users/mlcbppg/Desktop/backend/python_script/rtsp
# 方案 A：重建（推荐，uv.lock 已固定版本）
rm -rf .venv && python3.12 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt pytest
# 方案 B：只补 cv2
.venv/bin/python -m pip install --no-cache-dir --force-reinstall "opencv-python==5.0.0.93"
```

本次验证使用了一个**最小**虚拟环境 `.venv-profile/`（numpy 2.5.3、opencv-headless 4.13、
PyAV 16.1.0、pytest 9.1.1、fastapi、pillow；已加入 `.gitignore`），命令：

```bash
.venv-profile/bin/python -m pytest tests -q      # 710 passed, 40 subtests, 4 torch 相关失败
.venv-profile/bin/python -m pytest tests/test_ground_litter_profile_factory.py -q   # 78 passed
.venv-profile/bin/python -m pytest tests -q -k "ground_litter or v32 or v33"        # 416 passed
```

## 10. 服务器测试方案（下一步，未执行）

本地**不再**做真实 PS 下载或建库；真实素材验证放到服务器（空闲空间更大）。建议流程：

1. 只读预检：`df -h`、`nvidia-smi`、`docker ps`、确认无活动生产流，并记录当前
   容器启动时间与 RestartCount（用于证明本次不动生产）。
2. 上传代码与最小依赖（不要上传 `.venv*`、不要上传任何 PS 素材）：
   `rsync -a --exclude '.venv*' --exclude 'output' --exclude 'data' . sf01@<host>:<dir>/profile-factory/`
3. 在服务器建一个隔离虚拟环境，只装 `numpy/opencv-python-headless/av/fastapi/pytest`
   （**不装 torch/ultralytics**，本次不需要模型推理）。
4. 跑确定性测试：`pytest tests/test_ground_litter_profile_factory.py -q` 与
   `pytest tests -q -k "ground_litter or v32 or v33"`，把日志落盘。
5. 真实试点（服务器 `/data` 或 `/home` 分区，**不要**写 `/tmp`）：
   * `scripts/pilot_ground_litter_playback_cache.py --work-dir <大分区>/pilot_work`
     `--max-downloads 1 --refresh-wait-seconds 380 --test-resume`
     目的：核实网络、120s 刷新、Range 续传、HEVC 2.5K 解码、峰值空间与清理。
   * 同机跑生产流时，必须同时观测主流 `publish_fps/duplicate/pipeline_healthy`。
6. 小规模建库（例如 6 个文件）先验证链路；再视空间与保留期限决定是否扩大。
7. 立即用同样方式确认**远端录像可重拉期限**：隔 1 小时、1 天各重查一次同一
   `fileId` 是否仍可取 URL；不可重拉就必须在报告中列为材料缺口。
8. 把服务器实测数字回填本文 §5，并把「未验证」逐条改成实测或明确放弃。

**服务器上仍不得做**：接入生产 API/worker、重启生产容器、创建生产识别流、自动切换
Profile（属于方案二 B3/B4/B6，需单独授权）。


## 11. 服务器真实素材建库与盲测（2026-09-20 实测）

服务器：`sf01@14.21.88.97:21002`，`/home/sf01/profile-factory`（隔离目录，未动部署目录、
未重启任何容器；生产 API 全程 `Up`、无活动流）。空闲磁盘 319G，16 核，15G 内存，
RTX 3060 Ti 空闲。

### 11.1 素材审计（逐小时 156 次请求）

| 日期 | 文件数 | 声明大小 |
|---|---|---|
| 2026-09-14 ~ 09-19 | 每天 275–285 | 每天约 19.7–20.6 GB |
| 2026-09-20（截至 11:37） | 137 | 8.6 GB |
| 合计 | **1,826** | **约 129.3 GB** |

* 09-13 返回业务码 1500（无录像）→ **远端保留窗口约 6.5 天**，与用户"09-14 至今"一致。
* 单次查询 1h/2h/4h 均成功；**原设计猜测的"最大查询跨度"限制不成立**——早期 6h/24h 报错
  是因为查询窗口越过了"当前时刻"。逐小时扫描 156 个窗口全部成功。
* 记录长度 18–311s，常见 300–304s。

### 11.2 构建配置与产物

```text
输入：ctseelink file-urls，2026-09-14 00:00:00 → 2026-09-20 12:00:00
抽样：每天 6 个小时槽（覆盖凌晨/白天/夜晚），最多 60 个文件，seek 取帧
几何：config/ground_litter_01030_geometry.json（r2，14 点人行道多边形）
时间划分：构建 09-14~09-18（28 文件）、校准 09-19（6）、盲测 09-20（6）
产物：/home/sf01/profile-factory/out/bank/camera_01030/v1
```

| 指标 | 实测 |
|---|---|
| Bank 版本 | `camera_01030/v1`，`bank.json` SHA-256 `cf39406a922b0becc64e88eae80a115f1e454747b5aa35a4913c2bca5a4f6df7` |
| 参考尺寸 | 2560×1440（与 PS 原生一致） |
| 候选外观组 | 36 组；按静态覆盖粗排取前 **16** 做完整评分 |
| 冻结 Profile 数 | **N = 16** |
| 采样样本 | 112 个（112 个独立时间块，跨 5 个构建日） |
| 下载量 | 43 个文件、**11.43 GB**、519.7s（seek 模式期间另有高清重拉需求，实际为 0） |
| 峰值工作目录 | **2.86 GB**（raw 峰值 2.86 GB） |
| 临时 PS 清理 | 全部 EVICTABLE 条目已释放 |
| 动态定稿（构建集 84 帧） | 有效覆盖 **0.9647**，切换 1 次，最长暂停 3.0s |
| 噪声诊断 | 每组 `bias_flag` 连通区域均为 0（无持续大范围偏差） |

**N=16 的选择过程**：36 个候选先按静态潜在覆盖粗排，前 16 进入完整评分；对这 16 个用
同一份 `ProfileSelector` 在 960×540 回放画布上做 84 tick 时序回放，得到有效覆盖 0.9647；
剪枝阶段没有候选被删除（每个候选都带来非零静态覆盖）。**上限 24 与资源约束未触及。**

### 11.3 独立日盲测（2026-09-20，未参与构建）

从 09-20 的 02/06/10/14 时各取 1 个文件（该日 18/22 时尚未产生录像），在 **960×540
评估画布**、0.05 FPS 下做连续回放：

| 指标 | 值 |
|---|---|
| 覆盖的 tick | 46（4 个文件） |
| **静态潜在覆盖** | **1.000**（46/46） |
| **动态有效覆盖** | **0.9453** |
| 观测时间 | 1,461.6s（空隙 27,506s 按停机截断，不计入分母） |
| 暂停 P95 / 最长 | **39.98s / 39.98s**（两次切换过渡各一次） |
| 切换次数 | 2 |
| 缺口 | 1（`SOURCE_TIME_REWIND`，来自文件名时间替代 mtime） |
| 小目标对照 | 46 个 tick 均出现 prior 候选；合成小目标仅用于机制对照 |

**关键结论：静态 1.000 与动态 0.945 明确分离**，实测差值约 5.5 个百分点。这正是设计复核
F4 要求区分的东西——静态潜在覆盖可以到 100%，实际有状态 Selector 只有 94.5%，
且恢复过程产生约 40s 的暂停。

### 11.4 本轮在服务器上发现并修复的真实缺陷

1. **PS 的 PTS 基值不是 0**（实测首帧约 10,394s）：顺序解码时所有采样点全部落空。
   已把时间戳重定基为文件内相对秒，绝对时间只来自清单。
2. **seek 取帧系统性偏晚 2–4 秒**：只往目标之后解码，被关键帧间隔拖后。改为同时保留
   目标前/后两帧取更近者，误差降到 **0.017s**，约 **3s/文件**（顺序解码同一文件 180.9s）。
3. **`estimate_noise` 单组 326s**：2560×1440×24 帧逐像素 `nanquantile` 全排序。改为
   只在 ROI 有效像素上做「逐阈值计数 + 二分」分位，降到 **6.7s/组**。
4. **低分析节拍下动态覆盖恒为 0**：评估器把 `max_observation_gap_seconds` 固定为 4s，
   而 0.05 FPS 下两 tick 相隔 20s，每次观测都清空连续证据。修正为随节拍缩放后，
   动态覆盖从 0.0 → 0.9453。**这是"静态冒充动态"的反向陷阱，必须保留这条回归。**
5. **内存峰值 9.9GB**（15G 机器）：预选阶段 36 候选 × 84 帧在 2.5K 画布上全量评分。
   改为回放画布 960×540、只评前 16 个候选、合成后立即释放 HD 帧缓存。
6. **网络抖动让整段作业失败**：清单查询缺有限退避。已加 3 次指数退避 + 单窗口跳过不
   中断，并把清单缓存到 `work/inventory_*.json`，续跑不再重扫六天。

### 11.5 仍未验证 / 已知限制

* **小目标对照不是准确率**：合成叠加只证明「不被噪声/局部无效吞掉」的机制；没有现场
  摆放真值，不能据此给出准确率或召回率。
* **"第七天盲测"方法学偏弱**：远端只保留约 6.5 天，所谓第七天只能是 09-20 当天上午
  （同一天的另一时段），不是完整独立日；评估只用 4 个文件、0.05 FPS。
* **暂停与切换时延是离线虚拟时钟**：报告里 `wall_seconds` 是 I/O 成本，**不是在线恢复
  速度**；active prior + semantic 的整 tick 时延必须在方案二 B6 服务器实测。
* **评估缺口 1 次来自文件名时间**：下载器把 `record_start` 写进文件名，评估器优先用它；
  少数文件仍回退 mtime，导致时间回退告警。不影响覆盖结论，但应在接入真实来源时消除。
* **ROI 仍待用户最终确认**：r2 多边形已在真实 2560×1440 帧上目视贴合人行道（见
  `output/profile_01030_roi_20260920/roi_corrected_on_real_frame.jpg`），但左边界是否
  再内收以排除停车带，需要用户定夺。
* **低支持 Profile 偏多**：16 个里 12 个只由 2–8 个样本支撑（`low_support=true`），
  多数来自单日。文档允许"一天出现但跨多个片段"的参考上线，但这些参考的稳健性弱，
  应按更严格匹配/更小可用范围使用；若要做正式验收，应提高每天小时槽数量后重建。
* **未做**：生产 API/manager/worker 接入与在线切换（方案二 B3/B4/B6）、真实摄像机下的
  小目标对照、跨机位 ROI 迁移。

### 11.6 复现命令（服务器）

```bash
ssh -p 21002 sf01@14.21.88.97
cd ~/profile-factory && source venv/bin/activate

# 确定性测试
python -m pytest tests/test_ground_litter_profile_factory.py -q

# 六天建库（清单有缓存，可反复续跑）
./run_pipeline.sh v1 6 60 "2026-09-14 00:00:00" "2026-09-20 12:00:00"

# 独立日盲测评估
python eval_download_and_evaluate.py --tag blind --day 2026-09-20 \
  --hours 2,6,10,14 --max-files 12 --version v1
```
