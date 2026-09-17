# Ground-litter pilot log

状态：仅日志分析，不是准确率报告。

创建记录：2，唯一 ID：2，重复创建：0
重复 ID 指同一 ID 被重复写入；同一实物分配不同 ID 仍需对照截图核查。
事件统计：{"run_started": 1, "storage": 30, "source_state": 1, "sample": 30, "rejected": 1, "observation": 1609, "run_stopped": 1}
拒绝原因：{"analysis_result_stale": 1}
源状态变化：{"live": 1}
人工复核：false_positive=2

| 摄像头 | 观测 | 候选 | 创建ID | 推理P50/P95(s) | 结果年龄P95(s) |
|---|---:|---:|---:|---:|---:|
| 44180209031322001021 | 1609 | 2666 | 2 | 0.259/0.482 | 0.516 |

优化顺序：
- review every created item before enabling notifications
- inspect rejected/stale/view-change counters from summary.json
- compare candidate labels and ROI boundaries before changing confidence
- run a fixed day/night replay after each profile or threshold change
