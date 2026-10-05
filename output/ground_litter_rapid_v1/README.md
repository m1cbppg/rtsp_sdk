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
| `C` | 复制上一已完成帧的真值（**不会**完成本帧） |
| `N` | Stage A：本帧无目标并完成 |
| `Enter` | Stage A：本帧真值确认完成（也是复制后的确认键） |
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

## 3A. 重复帧快速确认（Copy Previous Truth）

固定抽帧里大量画面是同一机位、同一批静止垃圾的近重复帧。`C` 用来省掉重复点击，但
**不省掉人工确认**。

### 可用条件

当前帧 vs **上一个已完成的 review frame**（同一 review 队列顺序）：

* `camera_id` 必须相同；
* 且满足下面之一：
  * **同一 PS**（`same_ps`），或
  * **时间连续**（`temporally_continuous`）：两帧解码时间差 ≤ `CONTINUITY_MAX_GAP_SECONDS`
    = 400 s（一个 PS 长 304 s，所以跨 PS 边界的相邻帧约 64 s，仍算连续）。

不满足时页面显示原因（`different_camera` / `time_gap_too_large` / `no_record_start` /
`no_completed_previous_frame_in_lookback`），**不会**静默复制。

### 复制什么

* `REQUIRED_LITTER` / `IGNORE_SMALL` / `UNCERTAIN` points，**source-native 坐标原样复制**，
  不做任何变换或位移；
* 对 Rapid-Train：源点上**已完成的 localization selection 会作为候选框复制**（候选
  `copied_from_previous_frame`，排在 A）。它**只是候选**，Stage C 仍要你自己按键选，
  不会自动接受。

### 复制后的状态

```
copy_state = COPIED_PENDING_CONFIRM
truth_complete = false      ← 关键：不算完成
```

页面顶部横幅显示 **“COPIED FROM PREVIOUS · 待人工确认”**，并且 Stage B **仍然隐藏**
（`truth_complete=false`，服务端不返回任何模型输出）。

你必须核对画面：

* 垃圾消失 → 删掉该 point；
* 出现新垃圾 → 补 point；
* 垃圾明显移动 → 重新点位置（复制坐标不会自动跟随）。

然后按 `Enter`（或 `N`）才变成 `truth_complete=true`，`copy_state=CONFIRMED_COPY`。
复制后做过增删会在状态里记 `copy_edited=true`，作为训练样本的 provenance。

> **不做自动视觉相似度判断代替人工**：页面只会提示“可复制上一帧”，是否复制完全由你按 `C`
> 决定；复制后是否成立也完全由你按 `Enter` 决定。

### Rapid-Eval 分母不变

* **60 个 fixed eval frames 全部照旧进入评估分母**，一帧不少。
* 即使用了 `C`，**每一帧仍必须人工 `Enter` 确认**；不会因为近重复而自动跳过任何 eval frame。
* 复制只是一次写入起点，`truth_complete` 仍然必须由人置位。

### Rapid-Train 训练导出的近重复上限

review 阶段仍然保留**所有** fixed frame 的 truth（不改 `split.json`、不改
`frame_manifest.jsonl`、不改已有 truth、不改 eval 分母）。近重复去重只发生在
**training export**：

* 同一 camera、同一/邻近位置（`center_xy` 距离 ≤ `NEAR_DUP_POSITION_RADIUS_PX` = 96 px）、
  相邻时间（≤ `NEAR_DUP_TIME_WINDOW_S` = 600 s）的样本归为一个 cluster；
* 每个 cluster **最多导出 `NEAR_DUP_MAX_PER_CLUSTER` = 2 个代表**：
  * positive：保留**最早一次**与**尺度最大的一次**；
  * hard negative：保留**最早**与**最晚**各一个；
* 同一静止垃圾不会在大量近重复帧里反复进入训练集；
* hard negatives 同样去重；
* Rapid-Eval 样本一律拒绝导出（`RapidError` / HTTP 409）。

`POST /api/export_plan` 可以随时查看当前会导出什么（只读，不写任何状态）：
返回 `positives` / `hard_negatives` 的 cluster 明细、`kept` / `dropped`，以及
`rapid_eval_denominator`（证明 eval 分母未被改动）。

---

## 3B. Stage C 提交契约（定位 submit / advance）

Stage C 的候选卡点击与数字键 `1 / 2 / 3 / 0` **走完全相同的一条路径**
（`selectCandidate(choice)` → `POST /api/localize_select`）。

### 请求

```json
POST /api/localize_select
{"truth_id": "t-00039", "frame_id": "<当前 frame_id>", "choice": 1}
```

`frame_id` **必须带**。`truth_id` 在同一个 artifact 里历史上可能重复（早期版本用
`len(points)+1` 生成 id，删除后会重用），所以服务端一律按
**(frame_id, truth_id)** 定位 point：

* 只给 `truth_id` 且该 id 在多个 frame 上存在 → 直接报错（`ambiguous truth_id`），
  绝不猜测、绝不写到别的 frame；
* `truth_id` 生成现在是**单调递增、永不复用**的（高水位存在
  `review_state.json` 的 `truth_id_counter`）。

### 响应

```json
{
  "ok": true,
  "truth_id": "t-00039",
  "frame_id": "..._t090",
  "choice": 1,
  "status": "LOCALIZED",          // 或 UNLOCALIZED_SKIP
  "already_recorded": false,      // 重复提交同一选择时为 true，且不写第二次
  "localization_complete": false, // 本帧 Required 是否全部定位完
  "remaining": 4,                 // 本帧剩余待定位数
  "next_truth_id": "t-00040",     // 下一个待定位点；完成时为 null
  "selection": {...},
  "stage": "localization",
  "progress": {...}
}
```

* **幂等**：同一个 (frame_id, truth_id) 最多一行 decision。重复提交同一 choice →
  `already_recorded: true`，不新增行；提交不同 choice → **原地覆盖**，也不新增行。
* 成功提交后前端自动切到 `next_truth_id`，`remaining` 递减，`Localization` 计数 +1，
  已选点不再出现在 pending 列表。
* 最后一个点完成后 `localization_complete: true`、`next_truth_id: null`、stage → `done`，
  可直接下一帧。
* **失败必须可见**：`4xx`（例如候选不存在、ambiguity）会在页面上弹出红色提示，并且
  **不会**前进——不会出现“页面跳一下但什么都没发生”。

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
