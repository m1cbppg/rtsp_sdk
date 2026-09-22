# Clean Reference + Temporal 视频 PoC

## 结论

- **本轮 PoC 不通过进入生产链路的门槛。** 数字上的时序漏斗通过，但 T01 补检与负样本候选负载均未达到目标。
- Gate 1（Raw → 5 秒轨迹减少至少 80%）：**PASS**，负样本减少 96.1%。
- Gate 2（T01 在 5 秒内稳定且 clean holdout 无 >10 秒误轨迹）：**FAIL**。主配置 T01 延迟为 61.0；面积 4–9 像素的诊断分支延迟为 13.0；holdout >10 秒轨迹为 0。
- 本轮主配置保持 `signature=3.0 / luminance=100 / min_area=10`。预探针发现 T01 的阈值核心通常只有 4–9 像素，因此另保留 `min_area=4` 诊断结果，但它不参与 PASS 判定。
- 负样本为 HEVC、约 2.13 Mbps；参考与正样本为 H.264、约 14.3 Mbps。负样本结论属于 **compression stress**，不能直接等同同编码条件下的误报率。
- 可选压缩对照把原 clean 视频转为 1440p/25 FPS/HEVC/约 2.1 Mbps；其 holdout 主配置为 0 个 raw、0 条 ≥10 秒轨迹。低码率本身没有复现负样本长轨迹，场景时段/局部外观变化才是主因。
- Reference 强负证据只抑制 13 / 698 个 YOLO raw 候选（1.9%）。Fusion 把 ≥5 秒轨迹从 YOLO 的 51 增加到 155，本轮没有获得净收益。
- 被抑制候选的证据卡多数是地砖线、固定斑点和推车结构，方向基本正确；问题是覆盖率不足，而不是这条负证据完全无效。
- 四段配准的中位重投影误差均低于 0.34px、P95 均低于 0.96px，没有 camera shift 证据；本轮失败主要来自环境外观失配与候选/轨迹表达。
- 参考视频内有持续停放的推车、车辆/电动车和圆凳；median 会保留这些长期遮挡。当前 valid mask 只依据 temporal MAD，**没有满足“长期遮挡不能进入 Clean Reference”这一要求**。
- 后验核心地面 ROI 诊断把负样本 ≥10 秒轨迹从 156 降到 20，说明边界/设施贡献很大；但 T01 主分支延迟仍为 61.0 秒，缩 ROI 不能修复小目标链路。

## 输入与方法

- Clean Reference：前 65%（0–35.8s）按 1 FPS 采样 36 帧做逐像素 median；后 35% 只做 holdout，未参与背景生成。
- Reference/Temporal：人行道 ROI，1 FPS，SIFT/RANSAC 轻量配准，逐帧通道增益/偏置和低频亮度归一化，局部 connected components，多目标时序关联。
- YOLO：正样本全时段 0.5 FPS；参数固定为 confidence 0.08、384px tile、768 inference、25% overlap，并使用现有 actor/context 过滤。
- Clean Reference 在完成构建后只读；所有当前帧、长期物体和负样本都没有写回参考。

## 六个问题

1. **时序是否显著减少瞬时误报？** 负样本 raw 候选 7102 个，5 秒轨迹 275 条，减少 96.1%。Gate 1 为 PASS。
2. **T01 是否稳定形成候选？** 主配置首次 raw=88.0，稳定 1/3/5 秒分别为 89.0 / 131.0 / 133.0。诊断小核心分支稳定 5 秒=85.0。
3. **YOLO 漏检时 Reference 是否补回？** 本样本无法验证：YOLO 在 72 秒已命中 T01，Reference 主分支反而明显更晚。YOLO 首次 T01 raw=72.0；Reference 首次 raw=88.0。
4. **生命周期能否分开瞬时变化和落地物？** 能过滤大量瞬时点，但不能区分持续固定结构残差与落地物。负样本 ≥1/3/5/10/30 秒轨迹分别为 812 / 481 / 275 / 156 / 39。
5. **正常视频是否存在长时误异常？** 最长五条轨迹为 `[{"track_id": 14, "first_seen": 0.0, "last_seen": 303.0, "hits": 303, "lifetime_seconds": 304.0, "continuity": 0.9967, "median_box": [1777.0, 563.0, 1783.0, 566.0]}, {"track_id": 110, "first_seen": 24.0, "last_seen": 303.0, "hits": 279, "lifetime_seconds": 280.0, "continuity": 0.9964, "median_box": [1580.0, 467.0, 1584.0, 471.0]}, {"track_id": 348, "first_seen": 112.0, "last_seen": 303.0, "hits": 190, "lifetime_seconds": 192.0, "continuity": 0.9896, "median_box": [1734.0, 440.0, 1738.0, 444.0]}, {"track_id": 20, "first_seen": 0.0, "last_seen": 179.0, "hits": 176, "lifetime_seconds": 180.0, "continuity": 0.9778, "median_box": [1127.0, 669.0, 1132.0, 672.0]}, {"track_id": 461, "first_seen": 147.0, "last_seen": 303.0, "hits": 157, "lifetime_seconds": 157.0, "continuity": 1.0, "median_box": [1748.0, 478.0, 1753.0, 481.0]}]`。证据以 `negative_reference/timeline.png`、`candidate_heatmap.png` 和 overlay 为准。
6. **下一步优先级？** 先改 Reference/Temporal，不应先接 VLM。需要用多时段干净参考或只读环境基线压制固定结构残差，增加跨视频 stable-noise map，并重做 4–9 像素核心的聚合与遮挡恢复；之后在独立正样本盲测。当前每 5 分钟仍有 156 条 ≥10 秒负样本轨迹，直接送 VLM 的负载和语义歧义都过高。

## 主要指标

| 数据段 | Raw | Raw/min | ≥1s | ≥3s | ≥5s | ≥10s | ≥30s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Clean holdout | 0 | 0.0 | 0 | 0 | 0 | 0 | 0 |
| Positive Reference | 3725 | 1425.4 | 418 | 229 | 124 | 69 | 34 |
| Negative Reference | 7102 | 1402.3 | 812 | 481 | 275 | 156 | 39 |
| Positive YOLO | 698 | 267.1 | 122 | 82 | 51 | 29 | 11 |
| Positive Fusion | 2395 | 916.5 | 405 | 264 | 155 | 94 | 41 |
| HEVC clean holdout control | 0 | 0.0 | 0 | 0 | 0 | 0 | 0 |

## 审核材料

- `clean_reference.jpg`、`reference_valid_mask.png`、`reference_noise_map.png`
- `holdout/`、`positive_reference/`、`negative_reference/` 的 raw JSONL、轨迹 JSON、时间线、热图和稳定框叠加图
- `positive_reference/t01_evidence.jpg`
- `positive_yolo_fusion/` 的 YOLO/Fusion 原始候选、被参考负证据抑制的候选、轨迹和时间线
- `compression_control_holdout/` 的同源低码率 HEVC 对照
- `positive_reference/core_roi_diagnostic/`、`negative_reference/core_roi_diagnostic/` 的边界排除诊断（不参与 Gate）

## 限制

- 只有一个摄像头、一个正目标和一个负视频，不能估计总体精确率或召回率。
- T01 位置由人工查看后冻结，因此是定向案例，不是盲测。
- 参考视频是清晨、正样本约 10 点、负样本约 8 点；结果同时检验了明显光照变化，但没有覆盖雨天。
- 本实验把大面积高残差区域作为 temporary unavailable；没有逐帧语义分割时，车辆、货架和细碎人体边缘仍会进入 raw 候选。负样本没有逐框人工真值，其中某些长时新增物可能是真实异物，不能全部称为语义误报。
