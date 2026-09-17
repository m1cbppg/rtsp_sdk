# 荷兴广场零散垃圾验收准备包

本目录只用于本地数据采集和人工验收准备，未启用摄像头、未连接生产 API、未发送通知。

## 使用步骤

1. 打开 `index.html`，按优先级完成五路昼夜采集。若历史回放无法证明机位和时间，改用现场可控摆放。
2. 在 `placements.json` 记录每次摆放、遮挡、移除、清空和再次摆放；保存原始媒体并计算 SHA-256。
3. 将对应 `truth/*.json` 中的 `source`、`episodes`、`record_reviews` 和时间轴补齐。`null`、`false`、`uncertain` 表示未完成，不能当作负例。
4. 对完整日志快照运行：

   ```bash
   python3.12 scripts/evaluate_ground_litter_acceptance.py \
     --truth output/ground_litter_acceptance_20260912/truth/44180209031322001030-day.json \
     --logs /path/to/observations.jsonl \
     --media /path/to/original.mp4 \
     --output output/ground_litter_acceptance_20260912/report-1030-day
   ```

5. 报告返回“无法计算”时先补齐 blockers；不要用候选框数量替代真值。报告即使计算成功，也只代表该样本，不能自动批准生产或通知。

`baseline/manifest.json` 记录配置、模型、参考图和代码指纹。基线 profile 中全部摄像头保持 `enabled=false`，原文件不要覆盖。
