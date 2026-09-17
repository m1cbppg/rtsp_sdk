# Ground-litter pilot log

状态：仅日志分析，不是准确率报告。

创建记录：1，唯一 ID：1，重复创建：0
重复 ID 指同一 ID 被重复写入；同一实物分配不同 ID 仍需对照截图核查。
事件统计：{"run_started": 1, "rejected": 2, "observation": 118, "run_stopped": 1}
拒绝原因：{"view_alignment_unknown": 2}
源状态变化：{}

| 摄像头 | 观测 | 候选 | 创建ID | 推理P50/P95(s) | 结果年龄P95(s) |
|---|---:|---:|---:|---:|---:|
| 44180209031322001021 | 118 | 192 | 1 | 1.799/2.312 | None |

优化顺序：
- review every created item before enabling notifications
- inspect rejected/stale/view-change counters from summary.json
- compare candidate labels and ROI boundaries before changing confidence
- run a fixed day/night replay after each profile or threshold change
