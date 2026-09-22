# Ground Litter V3.3 服务器候选评审（2026-09-19）

## 结论

V3.3 候选代码、镜像和隔离服务器稳态测试通过；**尚未部署生产**。
测试使用同机位白天正样本帧编码成 1920×1080、25 FPS H.264 回放，隔离 API、
隔离 MediaMTX、独立端口和独立运行目录。生产容器、生产流、生产镜像标签和 compose
均未改变。

先验 profile 在该素材上判定 `GLOBAL_LIGHT_CHANGE`，因此先验分支正确 abstain。
本轮能验 semantic 主路、状态机持久性、主链吞吐、显存、输入积压和 RTSP 实际发布；
不能验 prior-only / fused 现场召回、prior crop 服务器 P95 或全天误报率。

## 本轮修复

1. `maximum_boxes` 按来源轮转，semantic 不再永久挤掉 prior-only 框。
2. actor/context 坐标统一到 profile 空间，`context_class_ids` 参与遮挡过滤。
3. 跨来源事件合并要求两个不同时间戳的桥接证据。
4. hybrid `result_version` 单调递增，语义拒绝计数进入快照。
5. side inference 保持 latest-wins，不积压旧帧。
6. API `metrics` 响应允许嵌套 `ground_litter_hybrid`；修复首批指标到达后
   `GET /v1/streams/{id}` 返回 500 的问题。
7. Clean Reference 全局颜色鲁棒拟合只在既有最多 25 万采样点上做三轮残差迭代，
   最终仍只生成一次全分辨率结果。相同服务器同帧缓存 tick 从约 2.83–3.21 秒降到
   约 1.01 秒，未改变阈值、profile、拟合次数或候选规则。
8. 服务器测量脚本用当前 `last_full_scan_ms > 0` 区分扫描 tick；不再误用累计扫描次数。

## 最终稳态结果

| 指标 | 实测 | 门槛 | 结论 |
|---|---:|---:|---|
| normal tick P95 | 1762.65 ms | < 2000 ms | 通过 |
| full ROI scan P95 | 81.95 ms | < 1500 ms | 通过 |
| input frame age P95 | 220.86 ms | < 4000 ms | 通过 |
| V3.3 新增显存 | 665 MiB | ≤ 1536 MiB | 通过 |
| publish FPS 最低值 | 24.590 | ≥ 20 | 通过 |
| unique publish FPS 最低值 | 24.590 | ≥ 20 | 通过 |
| duplicate publish FPS 最大值 | 0 | = 0 | 通过 |
| pipeline healthy | 60/60 true | 全部 true | 通过 |
| dropped analysis frames | 最大 0 | 不积压 | 通过 |
| crop batch P95 | 未触发 | < 800 ms | 不可判定 |

隔离 MediaMTX 已确认候选输出路径上线并接收 1 条 H.264 轨道，以上 publish 指标不是
只有编码支路的内部计数。

测试期间 `semantic_only_confirmed` 始终为 1，`prior_only_confirmed=0`、
`fused_confirmed=0`，符合 profile abstain 的测试边界。显示计数曾短时从 1 变 0 后恢复，
但事件确认状态没有丢失且当次语义候选仍存在；代码路径表明这是 actor/context 遮挡时
主动隐藏框，解除后恢复同一事件，不是事件重建或启动期误报闪烁。

## 源重连韧性

一次刻意跨越 180 秒回放 EOF 的测试中，输入重连让一个统计窗口的 publish FPS 降到
10.14，随后恢复到约 25 FPS，worker、API 和事件内存均未退出。这证明不会永久卡死，
但该轮因包含人工 EOF，不作为稳态性能通过证据。

## 可复现证据

- `final-baseline-measurement.json`：真实 RTSP sink 下的同输入基线。
- `final-hybrid-measurement.json`：最终 60 点、2 秒间隔的 V3.3 稳态测量。
- `source-restart-resilience-measurement.json`：跨回放 EOF 的恢复测量。
- 候选镜像：`rtsp-yolo-annotator:deepstream8-ground-litter-v33-dual-recall-20260918`
- 候选镜像 ID：`sha256:39243d33587cc848838d788f8d8cc21000ac9b8ddcaa1dee248fd2a59db10f1f`
- 最终构建包 SHA-256：`a250b25f87b5eb7abaa8d1c26e52cab14d6ffe6582d552511ba2d121b37c3961`
- 本地回归：636 passed，40 subtests passed。

## 仍需单独完成

1. 用近七天回放制作可覆盖时段/天气的 profile；这是独立任务，不在本轮通过范围。
2. profile 可用后补 prior-only、fused 和 crop batch 的现场服务器验收。
3. 使用连续真实监控回放做按小时误报率评估；当前静态正样本回放不能回答全天误报率。
4. 生产切换仍需单独部署步骤；本轮只保留候选镜像与发布包。
