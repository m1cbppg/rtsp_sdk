# Clean Reference V2：正常尺寸优先实验

## 结论

- V2 忽略真正只有 3–4px、没有周边支持的变化；候选必须有 ≥4 个强 seed，并恢复为面积 ≥30px、短边 ≥6px 的物体区域。
- T01 首次 raw=72.0，稳定 1/3/5 秒=73.0 / 75.0 / 77.0，5 秒延迟=5.0。目标门槛：**PASS**。
- 负样本评估从 60 秒开始，raw=2362，≥5/10/30 秒轨迹=82 / 45 / 18。
- 正样本 Fusion ≥5 秒轨迹：YOLO=32，V2 Fusion=53。
- 同一 60 秒后区间，正样本 V1→V2：raw 2295→646，≥10 秒轨迹 44→8。
- 同一 60 秒后区间，负样本 V1→V2：raw 5732→2362，≥10 秒轨迹 123→45。
- T01 区域在目标出现前 64–66 秒也有环境残差轨迹；V2 解决了尺寸和召回问题，但语义精度仍需 actor/context 与 VLM。
- Adaptive mask 只使用已声明无目标的 0–60 秒，60 秒后冻结；Clean Reference 始终只读。这是开发集方法，生产中必须由人工 clean confirmation 和 actor/OCCLUDED mask 保护初始化。

## 固定参数

- 强 seed：signature≥3.0 且 luminance≥100.0
- 周边支持：signature≥1.2 且 luminance≥25.0
- seed≥4px，support area≥30px，support short side≥6px
- Adaptive 高频出现率≥40% 或 support 出现率≥80%

## VLM 接口

- 已生成 positive 40 条、negative 40 条 review manifest。
- 每条包含 clean reference crop、current crop、context crop、bbox、时间戳和 persistence。
- 本轮没有调用 VLM；manifest 只验证输入契约。

## 边界

- V2 参数在 T01 开发样本上确定，不能作为独立召回结论。
- Negative 视频没有逐框语义真值；长轨迹可能包含真实新增地面物、车辆边缘或设施变化。
- 下一轮必须使用新的 YOLO 漏检样本做盲测，并固定 V2 参数不再调节。
