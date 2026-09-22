# V3.3 双通道 — 性能报告（本机 CPU，非目标 GPU）

> **重要边界**：本机为 macOS/CPU，规格 §10.2 的预算（普通 tick <2000ms、全扫描 <1500ms、crop 批 <800ms、显存 ≤1.5GiB）是针对 RTX 3060 Ti 的。本报告的数字**不能用于判定是否达标**，只能用于观察阶段占比与相对量级；达标判定必须在服务器 GPU 上重测。

## 真实权重 smoke（单帧，本机 CPU）

- 权重：54786410 字节，本地加载，无下载
- 模型加载：11.17s
- 全 ROI 扫描：3204.1ms，**1 次批量调用**，batch=6（tile 数 6）
- prior crop：593.8ms，**1 次批量调用**，batch=2
- 判定：passed=True

## 回放分阶段耗时（P50 / P95 / max，ms）

| 回放 | 阶段 | P50 | P95 | max |
|---|---|---|---|---|
| `clean_negative` | prior_ms | 547.2 | 592.4 | 646.3 |
| `clean_negative` | full_scan_ms | 1390.5 | 1841.5 | 2086.0 |
| `clean_negative` | total_ms | 1869.2 | 2319.5 | 2728.3 |
| `clean_negative` | wall_ms | 1869.8 | 2320.2 | 2730.5 |
| `clean_negative_strict` | prior_ms | 538.0 | 562.1 | 622.7 |
| `clean_negative_strict` | full_scan_ms | 1304.8 | 1597.7 | 1944.3 |
| `clean_negative_strict` | total_ms | 1805.2 | 1888.1 | 2526.4 |
| `clean_negative_strict` | wall_ms | 1806.2 | 1888.7 | 2527.0 |
| `positive_small_litter` | prior_ms | 563.7 | 587.1 | 616.3 |
| `positive_small_litter` | full_scan_ms | 1434.3 | 1850.1 | 2417.7 |
| `positive_small_litter` | total_ms | 1860.8 | 2306.6 | 2968.4 |
| `positive_small_litter` | wall_ms | 1861.2 | 2307.3 | 2969.0 |

## 调度与批量化

- `clean_negative`：模型调用 full=76、crop=0；crop 未关联拒绝=0；降级 tick=0/143
- `clean_negative_strict`：模型调用 full=76、crop=0；crop 未关联拒绝=0；降级 tick=0/143
- `positive_small_litter`：模型调用 full=40、crop=0；crop 未关联拒绝=0；降级 tick=0/71

## 主链影响

离线回放不构建 DeepStream 主管线，因此**本报告不含主链 FPS、duplicate FPS、输入帧年龄与显存**。这些必须在服务器候选镜像上实测后补充；在主链 FPS ≥20、duplicate=0 得到实测证据之前，不得声称性能达标。
