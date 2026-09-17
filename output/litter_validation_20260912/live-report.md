# Ground-litter pilot log

状态：仅日志分析，不是准确率报告。

创建记录：0，唯一 ID：0，重复创建：0
重复 ID 指同一 ID 被重复写入；同一实物分配不同 ID 仍需对照截图核查。
事件统计：{"run_started": 1, "storage": 3, "source_state": 1, "sample": 3, "rejected": 1, "observation": 145, "run_stopped": 1}
拒绝原因：{"analysis_result_stale": 1}
源状态变化：{"live": 1}

| 摄像头 | 观测 | 候选 | 创建ID | 推理P50/P95(s) | 结果年龄P95(s) |
|---|---:|---:|---:|---:|---:|
| 44180209031322001021 | 145 | 7 | 0 | 0.404/0.459 | 0.49 |

未确认原因（按候选轨迹观测计次，同一物体可重复出现）：{"actor_occluded": 5, "confirm_duration": 15, "actor_clear_duration": 13, "recent_hits": 12, "hit_fraction": 6, "model_miss": 4}
模型候选过滤原因：{"actor_overlap": 176, "outside_roi": 399, "local_actor_overlap": 35, "too_small": 5}

优化顺序：
- review every created item before enabling notifications
- inspect rejected/stale/view-change counters from summary.json
- compare candidate labels and ROI boundaries before changing confidence
- run a fixed day/night replay after each profile or threshold change
