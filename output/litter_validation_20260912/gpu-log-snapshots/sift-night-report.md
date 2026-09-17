# Ground-litter pilot log

状态：仅日志分析，不是准确率报告。

创建记录：0，唯一 ID：0，重复创建：0
重复 ID 指同一 ID 被重复写入；同一实物分配不同 ID 仍需对照截图核查。
事件统计：{"run_started": 1, "observation": 152, "run_stopped": 1}
拒绝原因：{}
源状态变化：{}

| 摄像头 | 观测 | 候选 | 创建ID | 推理P50/P95(s) | 结果年龄P95(s) |
|---|---:|---:|---:|---:|---:|
| 44180209031322001021 | 152 | 142 | 0 | 0.957/1.311 | None |

未确认原因（按候选轨迹观测计次，同一物体可重复出现）：{"confirm_duration": 232, "actor_clear_duration": 196, "recent_hits": 203, "model_miss": 60, "hit_fraction": 55, "actor_occluded": 30}
模型候选过滤原因：{"outside_roi": 864, "actor_overlap": 72, "too_small": 24, "local_actor_overlap": 9}

优化顺序：
- review every created item before enabling notifications
- inspect rejected/stale/view-change counters from summary.json
- compare candidate labels and ROI boundaries before changing confidence
- run a fixed day/night replay after each profile or threshold change
