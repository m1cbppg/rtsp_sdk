# Ground Litter V3.2 半合成生命周期验证

## 结论

**PASS**

本素材由同一摄像头真实画面按时间重新排列，用于验证状态机生命周期，不用于估计真实准确率或召回率。

基线决策：`ACCEPTED_AS_CURRENT_OFFLINE_BASELINE`。V3.2 允许作为当前离线 PoC 和后续集成基线；尚未部署生产，仍需通过未来自然发生的现场事件补充验证。

## 素材时间线

| 阶段 | 来源 | 原视频时间 | 合成视频时间 | 预期画面语义 |
|---|---|---:|---:|---|
| clean_before_first_deposit | clean_reference | 36–46s | 0–10s | CLEAN |
| first_litter_with_real_occlusion | positive | 72–92s | 10–30s | LITTER_VISIBLE_OR_OCCLUDED |
| clean_after_removal | clean_reference | 36–51s | 30–45s | CLEAN |
| second_litter_same_location | positive | 80–100s | 45–65s | LITTER_VISIBLE |

Clean Reference SHA-256：`d91d9aa50d06f7b2025f66776ce31af9d919391d65cfb1c0a97336ca2e7693f8`  
正样本 SHA-256：`bb265cadb83bc6d36fa7aa0ba4bedd3c7bf8c03a173b9edc7e451ef20481896a`  
合成视频 SHA-256：`fa0df7b821627791e556742c5069caa6b0a6ec074c43842025d19cc301a32709`

## 验收项

- PASS：exactly_two_target_confirmed_primary_events
- PASS：first_event_confirmed
- PASS：first_event_observed_real_occlusion
- PASS：first_event_cleared_during_clean_segment
- PASS：second_event_confirmed
- PASS：second_event_has_new_id
- PASS：second_event_started_after_first_cleared

## 目标事件

- 第一次事件：ID `1`，首次出现 `10.0` 秒，确认 `20.0` 秒，关闭 `35.0` 秒，原因 `clean_confirmed`。
- 第二次事件：ID `6`，首次出现 `46.0` 秒，确认 `50.0` 秒，当前状态 `VISIBLE_ANOMALY`。

## 证据边界

- 投放物、遮挡人员和地面画面均来自真实摄像头。
- “清走”和“再次投放”通过重排真实片段构造，不是自然连续发生的现场录像。
- 本测试通过只表示 V3.2 完整视频链路能正确关闭旧事件并在同位置建立新事件。
