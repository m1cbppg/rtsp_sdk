# Ground Litter Rapid Eval + Active Learning v1

工程快速闭环，用来在**一天内**回答四个问题：

1. 当前 Step 2B `last.pt` 在真实 4 路监控场景到底有多差；
2. 主要漏什么、误什么；
3. 如何用最少人工操作补一轮高价值训练数据；
4. 微调 V2 后是否有明确改善。

**这不是正式 Step 0B Go/No-Go，也不是 Formal Step 2C。** 正式 Step 2C 当前暂停。

---

## 1. 适用范围（硬边界，不得外推）

| 项 | 值 |
| --- | --- |
| 摄像头 | `01021` `01022` `01027` `01030` —— **仅这四路** |
| 不适用 | `01028` 当前严重遮挡，本轮**不做任何结论** |
| Development PS | 4 camera × 13 PS = **52 PS** |
| 抽取帧 | 每 PS 固定 5 帧（30 / 90 / 150 / 210 / 270 s）= **260 帧** + bonus train 帧 |
| 画布 | source-native **2560 × 1440**（绝不 resize，绝不用 Proxy B 截图做训练/评测源） |
| baseline 模型 | Step 2B `last.pt`，SHA256 `4852392a…`，**禁止用 `best.pt` 替代** |

任何结果都必须限定为：**“仅代表当前四路可用摄像头”**。

---

## 2. 正式 Development 与 Rapid Development 的关系（必读）

一旦某个 Development PS 被用于 Rapid Training，它就**不再是未来“完全未调参正式
Development”**。因此本工作区把 52 PS 立刻切成：

```
Rapid-Train        40 PS / 200 固定帧 (+ bonus)
Rapid-Eval Holdout 12 PS /  60 固定帧
```

隔离规则：

* **Rapid-Eval Holdout 永远不得用于训练 V2**，不得进入训练数据、不得作为 crop、不得
  作为 hard negative、不得用于 threshold selection 或 early stopping。
* 代码里有硬保护：`assert_trainable()` 对任何非 `rapid_train` 的导出直接 `RapidError`；
  审核服务额外提供 `POST /api/train_export_probe`，对 eval 帧返回 **HTTP 409**。
* **Sealed 仍然完全 untouched**，本轮一次都没有访问。
* 正式最终确认如果后续需要，仍然应该依靠 **untouched Sealed**，或重新定义正式协议。
  本轮的 Rapid-Eval Holdout 只服务于“这一步有没有变好”的工程判断。

### split 冻结方式

* seed：`ground-litter-rapid-v1-20260929`
* 每 camera 13 PS → 10 train / 3 eval
* 之前 exploratory detector probe 已经用过的 PS **强制进入 Rapid-Train**（因为它们已经不是
  干净 holdout）。来源：`output/fourcam-exploratory-20260929/selection.json`，6 个 PS。
* 其余 PS 按 `sha256(seed|file_id)` 升序排序，前 3 个成为 Rapid-Eval Holdout。
* `split.json` 记录每个 PS 的 `camera_id / file_id / split / selection_hash /
  selection_reason / excluded_from_holdout_due_prior_inference`，以及
  `split_sha256`。**生成后冻结，不得为了结果好看重换 holdout。**

若需要重新生成，见 §6；重建后 `split_sha256` 必须与冻结值一致。

---

## 3. 两阶段盲审 UI

审核服务**独立**：独立 artifact、独立端口 `8810`、本地隧道 `18810`。

```
本地浏览器 → ssh -N -L 18810:127.0.0.1:8810 → 服务器 127.0.0.1:8810
```

**8801 是 USER-ONLY OFFICIAL REVIEW ENVIRONMENT。** Rapid Eval 绝不使用 8801，
任何 automated browser 也禁止指向 8801（UI 冒烟脚本内置拒绝逻辑）。

### Stage A — Human Truth Discovery（模型输出完全隐藏）

* 显示 source-native 帧 + ROI overlay + 已落点位。
* **不显示** box、confidence、candidate 数量、model hit/miss，也不显示 prediction 数量。
  服务端在 `truth_complete=false` 时根本不序列化任何模型输出（连计数都不发）。
* 交互：先选类别（R / I / U），再点击目标中心点。**不画 bbox。**
* 类别定义：
  * `REQUIRED_LITTER`：正常观看可以看清，有实际清理意义的独立垃圾；
  * `IGNORE_SMALL`：确认是垃圾，但太小、不值得要求稳定识别；
  * `UNCERTAIN`：无法稳定判断是否垃圾。
* 每帧必须人工确认“本帧真值确认完成”（Enter）；没有垃圾则按 N，等价于
  `truth_complete=true` 且 0 点。**只有主动完成后才允许进入 Stage B。**

### Stage B — Prediction Review

* `truth_complete` 之后才出现 `last.pt` 的框。
* 判定：`Y` 正确垃圾框 / `N`（或 `F`）误报 / `X` ignore 相关 / `M` 实际是垃圾但
  Stage A 漏点 → 自动回到 Stage A 补点后重新确认，避免把“prediction 发现的漏标垃圾”
  错误算成 FP。
* **默认不显示 confidence 数字**，避免被置信度锚定。

### Stage C — Localization（仅 Rapid-Train）

* 对每个已确认的 `REQUIRED` 点给出最多 3 个候选框：

  | 候选 | 来源 |
  | --- | --- |
  | A | `last.pt` 中被判 `Y` 且包含该点的 bbox |
  | B | Turhancan semantic model（按 SHA256 定位，**不联网下载**；本机已找到） |
  | C | 现有 classical point-seeded proposal（Step 1C 逻辑） |

* 用户只按 `1 / 2 / 3 / 0`（None → `UNLOCALIZED_SKIP`），**禁止拖框 / resize / 手绘**。
* 每个候选显示 source-native crop（4× 最近邻放大，带候选框与点十字）。
* Rapid-Eval 帧不做 localization（不需要，也不应产生训练信号）。

### 快捷键

| 键 | 作用 |
| --- | --- |
| `R` / `I` / `U` | 选择真值类别 |
| 点击画面 | 落点 |
| `Shift` + 点击 | 删除 16px 内最近的点 |
| `N` | Stage A：本帧无目标并完成 |
| `Enter` | Stage A：本帧真值确认完成 |
| `Y` / `X` / `F`(或 `N`) / `M` | Stage B：正确 / ignore / 误报 / 漏标回退 |
| `1` `2` `3` `0` | Stage C：候选 A/B/C / None |
| `←` `→` | 上一帧 / 下一帧 |
| 滚轮 / 拖拽 | 缩放 / 平移（8px 小目标请先放大；右下角有 4× source-native 放大镜） |

### Resume

所有人工状态 atomic save（tmp + `os.replace`）到 `review/`：

* `review_state.json`（每帧 `truth_complete` / stage / 时间戳）
* `truth_points.jsonl`
* `prediction_reviews.jsonl`
* `localization_reviews.jsonl`

页面关闭重开从**未完成 frame** 继续，不丢进度。

---

## 4. 目录结构

```
output/ground_litter_rapid_v1/            # 本地（Git 只提交代码/测试/README/小 JSON）
  README.md
  split.json                              # 冻结 split（含 rows + split_sha256）
  frame_manifest.jsonl                    # 265 行：260 固定 + 5 bonus
  frame_manifest.meta.json
  split_provenance.json
  inherited_truth/                        # 既有 official Discovery REQUIRED point 的 seed
  _official_readonly_snapshot/            # 只读快照（manifest / truth / review_state）
  PHASE1_REPORT.md                        # 第一阶段报告
  phase1_report.json
  review/                                 # 人工状态（正式审核时由服务端写）
  baseline/
    predictions.jsonl
    inference_manifest.json
  dataset_v2/ training_v2/ evaluation_v2/ comparison.md   # 第二阶段产物（本轮为空）
```

服务器隔离资产：

```
/home/sf01/ground-litter-rapid-v1/
  code/{rtsp_annotator,scripts,tests}     # 与本地逐字节一致（SHA256 核对）
  artifact/                               # = 上面 output/ground_litter_rapid_v1/ 的服务器侧
  seed/fourcam-selection.json             # prior-inference PS 来源
  smoke/artifact/                          # 一次性 smoke test，绝不写正式 artifact
  logs/
```

**绝不与 `/home/sf01/step2c1-blind-truth/artifact` 混用。**

---

## 5. Baseline 推理链（V1 与 V2 必须完全一致）

```
source-native 2560×1440
  → ROI
  → 640×640 tiles，stride = 512（含右/下完整覆盖）
  → YOLO
  → tile → source 坐标
  → 只保留 prediction center 在 ROI 内
  → class-wise NMS，IoU = 0.50
  → conf floor = 0.01
  → top 100 / frame
```

一次性保存所有 bbox / confidence / tile id / source 坐标，后续 threshold sweep **不重新
inference**。

固定 threshold 网格：`0.01 0.03 0.05 0.10 0.20 0.30`

### threshold 选择规则（只能用 Rapid-Train reviewed frames）

1. Required Hit Rate 最大；
2. 若多个 threshold hit rate 相差 ≤ 2 个百分点 → 选 FP/100 frames 更低；
3. 若仍相同 → 选更高 threshold。

得到 `baseline_selected_threshold` 写入 `baseline/threshold_selection.json` 并**冻结**，
之后才允许在 Rapid-Eval 60 帧上计算 baseline metrics。

---

## 6. 复现命令

本地（无 GPU、无服务器）生成 split：

```bash
python3.12 scripts/build_ground_litter_rapid_split.py \
  --manifest output/ground_litter_rapid_v1/_official_readonly_snapshot/development_manifest.json \
  --truth    output/ground_litter_rapid_v1/_official_readonly_snapshot/truth_objects.jsonl \
  --review-state output/ground_litter_rapid_v1/_official_readonly_snapshot/review_state.json \
  --exploratory-selection output/fourcam-exploratory-20260929/selection.json \
  --output-dir output/ground_litter_rapid_v1
```

服务器（在 `code/` 下，用 `/home/sf01/ground_litter_train/.venv/bin/python`）：

```bash
DEV=/home/sf01/ground-litter-feasibility/20260923-r1/archive/ground-litter-detector-feasibility-20260923-r1/development
A=/home/sf01/ground-litter-rapid-v1/artifact

# 1) 冻结 split（只读 official artifact）
python -B scripts/build_ground_litter_rapid_split.py \
  --manifest /home/sf01/step2c1-blind-truth/artifact/development_manifest.json \
  --truth    /home/sf01/step2c1-blind-truth/artifact/truth_objects.jsonl \
  --review-state /home/sf01/step2c1-blind-truth/artifact/review_state.json \
  --exploratory-selection /home/sf01/ground-litter-rapid-v1/seed/fourcam-selection.json \
  --output-dir $A

# 2) 抽帧（顺序解码 + 精确 frame index）
python -B scripts/extract_ground_litter_rapid_frames.py \
  --artifact $A --development-root $DEV --workers 6

# 3) baseline 推理
python -B scripts/run_ground_litter_rapid_baseline.py \
  --artifact $A --model /home/sf01/step2b-20260923/out/runs/full_finetune/weights/last.pt \
  --model-sha256 4852392aeae9a68669a50752eb1f7466fbcc9a426524faba86fb20a3b6351a94

# 4) 盲审 UI（独立 8810）
python -B scripts/serve_ground_litter_rapid_review.py --artifact $A serve --port 8810

# 5) 隔离 smoke test（一次性副本，端口 8811）
python -B scripts/verify_ground_litter_rapid_phase1.py \
  --artifact $A --smoke-dir /home/sf01/ground-litter-rapid-v1/smoke/artifact --port 8811
```

本地隧道：

```bash
ssh -N -L 18810:127.0.0.1:8810 sf01@14.21.88.97 -p 21002
# 浏览器打开 http://127.0.0.1:18810/
```

真实浏览器冒烟（本机，使用已安装的 Google Chrome；脚本拒绝 8801）：

```bash
# 完整交互冒烟：只能指向一次性 smoke 副本（会写入点位与判定）
python3.12 scripts/smoke_ground_litter_rapid_browser.py \
  --base-url http://127.0.0.1:18811 --frame-id <rapid_train frame> \
  --out output/ground_litter_rapid_v1/browser_smoke

# 只读冒烟：可以安全指向正式 artifact（只 GET，不写任何状态）
python3.12 scripts/smoke_ground_litter_rapid_browser.py \
  --base-url http://127.0.0.1:18810 --read-only \
  --out output/ground_litter_rapid_v1/browser_readonly
```

> 冒烟必须打在**一次性副本**上，绝不能把 smoke 写进正式 Rapid artifact。只读模式用于
> 验证正式 8810 服务本身（Stage A 盲态 + 放大镜 + 进度面板）。

本地单测（不需要 cv2 / torch）：

```bash
.venv/bin/python -m unittest tests.test_ground_litter_rapid -v
```

---

## 7. 抽帧为什么是顺序解码

这些 PS 是 HEVC，随机 seek **不可靠**。在 01021 的 Development PS
（`…1790063733000__20260922155533-20260922160037.ps`）上做了两组实测，用顺序解码的
帧哈希作为真值：

| 目标 | seek 后拿到 | 顺序解码真值 | 结论 |
| --- | --- | --- | --- |
| 30.000 s (idx 750) | idx **751** | idx 750 | 晚一帧 |
| 7.476 s (idx 187) | idx 187 | idx 187 | 命中 |
| 90 / 150 / 210 / 270 s | 报告 `POS_MSEC` 对应的 idx 正确 | **画面不是同一帧** | 索引对、内容错 |

也就是说 `round(POS_MSEC/1000 × fps)` 这个修正虽然能把 idx 算“对”（2250 → 2250），但拿到的
**图像内容与真正的 frame 2250 不同**：seek 之后 `POS_MSEC` 的误差可达 ±1 帧，不能用来定位
帧。seq 750 / 2250 / 3750 / 5250 / 6750 的哈希与 seek 结果逐一比对，只有 187 与 750 命中。

因此抽帧器对每个 PS **从 frame 0 顺序解码**，精确取
`frame_index = round(offset_seconds × 25fps)`：

* `delta_ms` 恒为 `0.0`（构造性成立）；
* bonus 帧直接沿用 official Discovery 的 `frame_index`（例如 187），与官方标记的是
  同一帧；
* 记录 `decoded_relative_seconds / frame_index / delta_ms / decode_mode / source_sha256 /
  roi / roi_geometry_version / image_sha256`。

---

## 8. 指标定义（只用 point truth，不要求完整 GT bbox）

* **Required Point Hit Rate** = 命中的 REQUIRED 点数 / REQUIRED 点数
* **命中判定**：某个 prediction 被人工判 `Y`，其 bbox 包含该 REQUIRED 点，且**一对一**
  匹配（一个 prediction 只能命中一个点）
* **Miss Count** = 未被任何 `Y` 框覆盖的 REQUIRED 点数
* **FP** = `truth_complete` 帧中被人工明确判 `N` 的 prediction 数
* **FP / 100 frames**
* **Positive-frame Hit Rate** = 至少命中 1 个 REQUIRED 的帧 / 含 REQUIRED 的帧
* **Camera Breakdown**、**prediction count / frame**
* `IGNORE_SMALL` / `UNCERTAIN` 被 prediction 覆盖 → **既不记 TP 也不记 FP**（判 `X`）
* **不把 mAP 当核心指标**

---

## 9. 明确不做的声明

* **frame-level FP ≠ production alert/day。** 当前没有 temporal event layer，Rapid FP/100
  frames **不能**直接换算成“每天误报多少次”。
* 不使用 `Production PASS` / `Formal GO` / `Sealed PASS` 这类措辞。Rapid 结论只允许：
  `RAPID_ITERATION_POSITIVE` / `DATA_OR_IMAGING_BOTTLENECK` / `RAPID_V2_REGRESSION`。
* 没有完整人工真值的帧，不计算任何准确率；已完成的帧才进入分母
  （`require_review_complete()` 硬保护）。
* Rapid-Eval 没有 bbox 真值时，**不编造 size bucket**。

---

## 10. 安全边界

绝对禁止，且代码内有对应保护：

| 禁止项 | 保护 |
| --- | --- |
| Sealed access | `assert_development_asset()` 拒绝含 `sealed` 的路径 |
| 写 official artifact | `assert_writable_root()` 拒绝 `/home/sf01/step2c1-blind-truth` 与 `/home/sf01/step2b-20260923` |
| Rapid-Eval 进入训练 | `assert_trainable()` + `POST /api/train_export_probe` → 409 |
| Stage A 泄露模型输出 | `frame_payload()` 在 `truth_complete=false` 时不返回任何 prediction 字段 |
| 覆盖 Step 2B `last.pt` | 只读引用，并在推理前核对 SHA256 |
| automated browser 指向 8801 | `smoke_ground_litter_rapid_browser.py` 直接拒绝 8801 与非本地主机 |
