# Ground-litter pilot log

状态：仅日志分析，不是准确率报告。

创建记录：0，唯一 ID：0，重复创建：0
重复 ID 指同一 ID 被重复写入；同一实物分配不同 ID 仍需对照截图核查。
事件统计：{"run_started": 1, "storage": 238, "source_state": 11, "sample": 237, "rejected": 4364, "observation": 6129, "run_stopped": 1}
拒绝原因：{"analysis_result_stale": 2749, "view_alignment_unknown": 1360, "view_changed_recalibration_required": 254, "image_quality_unknown": 1}
源状态变化：{"live": 6, "source_unavailable": 5}

| 摄像头 | 观测 | 候选 | 创建ID | 推理P50/P95(s) | 结果年龄P95(s) |
|---|---:|---:|---:|---:|---:|
| 44180209031322001021 | 6129 | 7971 | 0 | 1.239/1.559 | 1.96 |

未确认原因（按候选轨迹观测计次，同一物体可重复出现）：{"confirm_duration": 8440, "actor_clear_duration": 6984, "recent_hits": 6650, "model_miss": 275, "hit_fraction": 477, "actor_occluded": 304}
模型候选过滤原因：{"outside_roi": 24481, "actor_overlap": 6539, "local_actor_overlap": 783, "too_small": 172}

优化顺序：
- review every created item before enabling notifications
- inspect rejected/stale/view-change counters from summary.json
- compare candidate labels and ROI boundaries before changing confidence
- run a fixed day/night replay after each profile or threshold change
