# Ground Litter Detector Feasibility — Step 0A Runbook

日期：2026-09-23  
实验：`ground-litter-detector-feasibility-20260923-r1`  
基线提交：`a37b69b87992e2c6b0e2e9ab38cdf4d0b4bafd39`  
基线方案：`docs/plans/2026-09-23-ground-litter-detector-feasibility-plan.md`

## 范围

Step 0A 只负责在任何 detector 结果参与之前，固定并永久封存 Development / Sealed Test 原始 PS。

不跑 detector，不看模型框，不做阈值、overlap、fusion 或训练决策。

预注册配置：

`config/ground_litter_feasibility_step0a_20260923.json`

固定窗口（Asia/Shanghai）：

- Development：2026-09-22 16:00:00 ～ 17:00:00
- Sealed Test：2026-09-23 08:00:00 ～ 09:00:00
- 五路摄像头统一使用上述窗口
- 每个评估窗口前后各留 30 分钟 Training exclusion guard
- 窗口不能因为垃圾多少或模型表现更换

若后续 Blind Truth 发现同一物理垃圾 episode 跨 split，则整个冲突 episode 从计分集合隔离；本轮不能把 Sealed episode 转入 Training。

## Freeze：先冻结 exact fileId

在能访问录像接口的机器执行：

```bash
STATE_DIR=output/ground_litter_detector_feasibility_20260923/step0a

python scripts/seal_ground_litter_feasibility_step0a.py freeze \
  --config config/ground_litter_feasibility_step0a_20260923.json \
  --state-dir "$STATE_DIR"
```

Freeze 对每个 camera × split：

1. 查询预注册固定窗口；
2. 再查询两个半窗口检查列表截断；
3. 固定所有与窗口相交的完整 PS `file_id`；
4. 计算时间覆盖率和最大 gap；
5. 检查 `deviceCode + fileId` 是否跨 split；
6. 写 `FROZEN_MANIFEST.json` 与 `FREEZE_REPORT.json`。

当前 freeze gate：

- coverage fraction >= 0.99；
- max gap <= 10 秒；
- 无列表截断嫌疑；
- 无跨 split file identity 冲突；
- 5 camera × 2 split 全部查询成功。

网络或接口查询失败不会换窗口，只能重试同一预注册窗口。

## Materialize：按 frozen fileId 永久下载

原始 PS 必须放在 Git 仓库外的持久磁盘：

```bash
export GROUND_LITTER_ARCHIVE_ROOT=/path/to/persistent/ground-litter-feasibility

STATE_DIR=output/ground_litter_detector_feasibility_20260923/step0a

python scripts/seal_ground_litter_feasibility_step0a.py materialize \
  --config config/ground_litter_feasibility_step0a_20260923.json \
  --state-dir "$STATE_DIR" \
  --archive-root "$GROUND_LITTER_ARCHIVE_ROOT" \
  --split all
```

Materialize 只按 frozen manifest 中 exact `file_id` 下载，不允许重新挑窗口。

每个 PS 保存：

- 完整原始文件；
- declared / actual bytes；
- SHA-256；
- record start/end；
- camera / split / scene_version；
- ROI provenance；
- 能解码时保存 codec / width / height / duration；
- `signed_url_persisted=false`。

完成的 PS 改为只读；Sealed 目录生成 `SEALED_DO_NOT_TUNE.txt`。

## 必须保留的实验证据

`STATE_DIR` 中保留并提交：

- `FROZEN_MANIFEST.json`
- `FREEZE_REPORT.json`
- `MATERIALIZATION_REPORT.json`
- `SHA256SUMS.json`

原始 PS 不进入 Git。

只有 `MATERIALIZATION_REPORT.json` 的 `step0a_gate=PASS` 后，Step 0A 才算完成。这个 PASS 只表示“Development / Sealed 原始录像已可靠封存”；`overall_detector_go_no_go` 此时仍必须是 `NOT_EVALUATED`。

## Split 纪律

- Sealed Test 的录像、帧、tile、标签、错误分析和模型输出都不能用于本轮调参或补训练；
- Development 可用于后续模型、阈值、overlap、fusion 选择；
- Training 禁止使用 frozen source files 以及 guard 内相邻帧；
- Blind Truth 后还必须做 episode-level 隔离；
- `IGNORE_SMALL` 不计 TP，也不计 FP。

## 当前状态

仓库侧 Step 0A 预注册配置、封存工具和测试已准备。只有真实录像接口可访问的执行机完成 Freeze + Materialize 并产出 `step0a_gate=PASS`，才能把状态更新为“原始录像已封存”。
