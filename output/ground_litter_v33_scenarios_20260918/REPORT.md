# V3.3 双通道 — 三场景事件 trace 报告

| 场景 | 期望来源 | 实际来源 | 事件数 | evidence_kind | 关闭原因 | 判定 |
|---|---|---|---|---|---|---|
| semantic_only | `hybrid_v33_semantic` | `hybrid_v33_semantic` | 1 | `semantic_only` | `semantic_absent_confirmed` | 通过 |
| prior_only | `hybrid_v33_prior` | `hybrid_v33_prior` | 1 | `prior_only` | `clean_confirmed` | 通过 |
| fused | `hybrid_v33_fused` | `hybrid_v33_fused` | 1 | `semantic_and_prior` | `clean_confirmed` | 通过 |

显示时间戳（确认后才允许上屏）：
- semantic_only: [20.0, 22.0, 24.0, 26.0, 28.0]
- prior_only: [22.0, 24.0, 26.0, 28.0]
- fused: [20.0, 22.0, 24.0, 26.0, 28.0]

逐时间戳明细与状态机历史见 `trace.json`，可读视图见 `trace.html`。
