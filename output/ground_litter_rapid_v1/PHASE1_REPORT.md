# Ground Litter Rapid Eval + Active Learning v1 — 第一阶段完成报告

**scope**: 仅代表当前四路可用摄像头 01021/01022/01027/01030；01028 严重遮挡，本轮不做任何结论。

## Split

- Rapid-Train: **40 PS**
- Rapid-Eval Holdout: **12 PS**
- seed: `ground-litter-rapid-v1-20260929`

| camera | rapid_train | rapid_eval | forced_to_train (prior inference) |
| --- | ---: | ---: | ---: |
| 01021 | 10 | 3 | 1 |
| 01022 | 10 | 3 | 3 |
| 01027 | 10 | 3 | 1 |
| 01030 | 10 | 3 | 1 |

- prior-inference PS 全部强制进入 Rapid-Train: **True** (6 PS)
- split SHA256: `e7d2eeea31651d7b6921ae9e088c7f5e59cfeab01de71bb51e827df992d41afd`

## Frames

- fixed train: **200**
- fixed eval: **60**
- bonus train: **5**
- 解码方式: `sequential_frame_index`，source-native 2560x1440
- decode failures: **0**
- missing: **0**
- max |delta_ms|: **14.0**
- frame_index != nominal 的记录数: 0

## Baseline

- checkpoint: `/home/sf01/step2b-20260923/out/runs/full_finetune/weights/last.pt`
- SHA256: `4852392aeae9a68669a50752eb1f7466fbcc9a426524faba86fb20a3b6351a94` (与冻结值一致: True)
- 使用 `best.pt`: **否**（禁止）
- inference frames: **265**
- tiles: 1730，raw candidates: 104
- raw predictions @ conf 0.01: **77**
- threshold 网格预测数: 0.01: 77, 0.03: 53, 0.05: 44, 0.10: 29, 0.20: 16, 0.30: 11
- device: cpu（服务器 GPU 驱动当前不可用，CPU 推理）

> 本阶段**不报告** eval 好坏：Human Truth 尚未完成，任何准确率数字都会是伪造。

## Review UI

- URL: `http://127.0.0.1:18810/`（本地隧道 → 服务器 127.0.0.1:8810，独立 artifact）
- 需要人工审核: **265** 帧（固定 260 + bonus 5；Rapid-Train 200+5，Rapid-Eval 60）

| 键 | 作用 |
| --- | --- |
| `R / I / U` | 选择真值类别（必需垃圾 / 微小忽略 / 不确定） |
| `点击画面` | 在该类别下落下真值中心点（不画框） |
| `N` | 本帧无目标并完成真值 |
| `Enter` | 本帧真值确认完成 |
| `Y` | Stage B：当前 prediction 判定正确 |
| `X` | Stage B：当前 prediction 属 ignore / 不确定 |
| `F 或 N` | Stage B：当前 prediction 是误报 |
| `M` | Stage B：实际是垃圾但 Stage A 漏点，回到 Stage A 补点 |
| `1 / 2 / 3` | Stage C：选择候选框 A / B / C |
| `0` | Stage C：None → UNLOCALIZED_SKIP |
| `← / →` | 上一帧 / 下一帧 |
| `滚轮 / 拖拽` | 缩放 / 平移（8px 小目标请先放大或用右下 4x 放大镜） |

## Safety

- Sealed = 0
- official writes = 0（只读读取 2C artifact）
- 8801 automated access = 0（审核服务独立使用 8810）
- Step 2B `last.pt` 未被覆盖，仅只读引用

- 隔离 smoke test: **PASS** (22/22 checks；正式 artifact 前后摘要一致 = True)
- 真实 Chrome smoke test: **PASS** (24/24 checks)

## Next

**现在开始人工 Rapid Review；完成后回复“继续”。**

第二阶段（只有在你明确说“继续”之后才会执行）：baseline threshold selection → baseline Rapid-Eval metrics → Rapid-Train localization → 构建 V2 dataset → 30 epoch 微调 → V2 threshold selection → V2 Rapid-Eval → comparison。
