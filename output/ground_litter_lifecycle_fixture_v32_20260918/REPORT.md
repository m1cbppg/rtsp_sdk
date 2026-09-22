# Ground Litter V3.2 半合成生命周期验证

## 结论

**FAIL**

本素材由同一摄像头真实画面按时间重新排列，用于验证状态机生命周期，不用于估计真实准确率或召回率。

## 素材时间线

| 阶段 | 原视频时间 | 合成视频时间 | 预期画面语义 |
|---|---:|---:|---|
| clean_before_first_deposit | 60–71s | 0–11s | CLEAN |
| first_litter_with_real_occlusion | 72–92s | 11–31s | LITTER_VISIBLE_OR_OCCLUDED |
| clean_after_removal | 60–71s | 31–42s | CLEAN |
| second_litter_same_location | 80–100s | 42–62s | LITTER_VISIBLE |

源视频 SHA-256：`bb265cadb83bc6d36fa7aa0ba4bedd3c7bf8c03a173b9edc7e451ef20481896a`  
合成视频 SHA-256：`fc26aa82bc2cd40845fd5fe531d9e663037b9de1e552890dfce1ec69fd5657d1`

## 验收项

- PASS：first_event_confirmed
- PASS：first_event_observed_real_occlusion
- FAIL：first_event_cleared_during_clean_segment
- PASS：second_event_confirmed
- PASS：second_event_has_new_id
- FAIL：second_event_started_after_first_cleared

## 目标事件

- 第一次事件：ID `2`，首次出现 `11.0` 秒，确认 `21.0` 秒，关闭 `None` 秒，原因 `None`。
- 第二次事件：ID `6`，首次出现 `50.0` 秒，确认 `54.0` 秒，当前状态 `VISIBLE_ANOMALY`。

## 证据边界

- 投放物、遮挡人员和地面画面均来自真实摄像头。
- “清走”和“再次投放”通过重排真实片段构造，不是自然连续发生的现场录像。
- 本测试通过只表示 V3.2 完整视频链路能正确关闭旧事件并在同位置建立新事件。
