# Ground Litter Historical Active Mining + Rapid Dataset v2

工程目标：用最近 7 天历史录像，按**单位人工分钟收益**最大化来挖掘
**独立垃圾实例**与**独立 hard-negative 模式**，产出 Rapid Dataset v2。

验收单位从 `frame` 改成：

1. **unique Required litter instance / review group**
2. **independent hard-negative cluster**

同一件固定垃圾连续出现几十分钟，训练侧默认只保留 **1 个 NORMAL 代表帧**，
确有明显外观变化时最多 2 帧，少数困难状态最多 3 帧。

> 旧的 265-frame Rapid Review **不再扩展、不 reset**。它的人工真值、定位、
> 预测、split 全部作为历史诊断资产保留在
> `output/ground_litter_rapid_v1/`，v2 的 split 与 eval 完全独立定义。

---

## 1. 硬边界

| 项 | 值 |
| --- | --- |
| 摄像头 | `01021` `01022` `01027` `01030`；`01028` 暂不参与 |
| 源分辨率 | source-native **2560x1440**，训练/推理绝不 resize |
| baseline | Step 2B `last.pt`，SHA256 `4852392a…`（只做 baseline / student candidate） |
| 候选生成器 | Turhancan `…a2f8de0c…`（只做 high-recall proposal，**不产生伪标签**） |
| 回放来源 | `/monitor/play/ctseelink/playback/file-urls`（只读清单 + 120s 临时签名 URL） |
| 不下载 | 禁止全量下载 7 天录像 |

---

## 2. 冻结时间 split

生成时间 2026-09-29，冻结于 `historical_window_manifest.jsonl`，之后**不得**因模型
结果更换 dev/final 日期。

| 日期 | split | 用途 |
| --- | --- | --- |
| 2026-09-23 … 09-27 | TRAIN | 挖掘与训练侧审核（D1–D5） |
| 2026-09-28 | DEV | 阈值选择与 unique recall |
| 2026-09-29（当天） | FINAL | 只用已经完整发生的白天时段（`final_day_usable_end`） |

白天定义 `06:00–18:00`，桶 `early 06–10 / mid 10–14 / late 14–18`（camera-local）。

- TRAIN：每 camera × 每 day × 每桶各 1 个 ≈5 分钟 window = **60**
- DEV：每 camera 4 个白天 window = **16**
- FINAL：每 camera 4 个白天 window = **16**
- 合计 **92 windows ≈ 460 分钟录像**
- 选择种子 `ground-litter-historical-v2-20260929`，同一 `(camera, bucket)` 优先取
  用量最少的整点，避免不同日期总落在同一个小时。
- `manifest_sha256 = 1e10448b9d16debcc294484be6f0cff11fdfe4de909ab99c75d345c6e401fc94`
  （只覆盖冻结字段；下载状态不会移动它）

window manifest 每行记录：`camera_id / date / start_time / end_time / day_split /
purpose / selection_bucket / selection_seed / source_file_id / download_status /
local_path / source_sha256 / reason_downloaded`。

Day-1 下载预算（§31）：TRAIN 每 camera×day 取 1 个 window = **20**，
并缓存 DEV/FINAL 各 16 个 window（只取帧，不做推理，保持 eval 独立）。

---

## 3. 粗扫与生产尺度推理

每个 5 分钟 window 只**顺序解码** `~30s / ~150s / ~270s` 三个 source-native 帧
（dense 窗口改为每 15 秒一帧）。这些 MPEG-PS HEVC 文件 random seek 不帧准
（实测晚一帧且 `POS_MSEC` 不可信），所以一律从第 0 帧顺序解码，
`round(offset * 25fps)` 由构造保证精确。

推理契约（与四路 exploratory probe 完全一致）：

- 640×640 source-native tiles，stride **512**，overlap **128**，含右/下边缘补齐；
- 只推理与 ROI 相交的 tile，预测框**中心**必须落在 ROI 内；
- 跨 tile per-class NMS IoU **0.50**；
- 两个模型都先宽松保存，`conf >= 0.01`，**绝不比较两个模型的 confidence**；
- 记录每个来源的框与分数，逐模型原始框进 `raw_candidates.jsonl`，跨模型按
  IoU≥0.30 或小目标中心距合并成 `candidate_observations.jsonl`。

审核准入：低于 `REVIEW_MIN_CONFIDENCE = 0.15` 的观测仍全量落盘，但不进入人工队列
（§6 把「Turhancan 极低分」列为低优先；实测一个 window 的中位数只有 0.03）。

---

## 4. 两层 identity

**episode_id**（§9.1）——连续/相对连续观测支持的一次垃圾存在过程：

- same camera；按真实 timestamp 关联，**PS 文件边界不会结束 episode**；
- `max_gap`：dense 15s 采样 90s，coarse 采样 360s；
- 空间半径 `r = clamp(0.6*sqrt(w*h), 8, 32)` source px（相对 episode 中位框）；
- 尺寸一致性：宽/高比例在 `1/3 ~ 3x`；
- 一对一匹配，取归一化中心距最小者；
- 小目标不依赖 IoU：中心距 + 最小像素容差 + episode 中位位置；
- 两个 episode 几乎等距时判为 **ambiguous**，不自动合并。

**review_group_id**（§9.3）——相隔几十分钟/几小时/跨日但位置、尺寸、外观接近的
episode，不自动延长，group 起来交人工判定：

- 半径 `2r`，尺寸比例同上，外观距离阈值 0.35（无 embedding，用 ROI 均值色）；
- UI 显示 before/after，用户选 `SAME` / `NEW` / `UNCERTAIN`。

模型中间漏检只表示 `unobserved`，**不能当 disappearance**；同位置明确出现 clean
interval 后再出现才算 new episode。

---

## 5. 代表帧

每个 group 默认 1 个 NORMAL 代表帧。选择**不是**最高 confidence、最大 bbox 或最好
识别的一帧，而是：接近 episode 时间中点、bbox 接近中位尺寸、清晰度正常、外观最常见、
没有被 tile 边界截断。

只有出现明显阴影/对比变化、姿态改变、轻度遮挡或换背景时，才允许加第 2 帧；
少数实例最多 3 帧。其余观测只在 `episodes.jsonl` 里留档，不给人工看。

---

## 6. Diversity 队列

candidate group 超过 150 时不全给用户。`queue.json` 分批：

| batch | 内容 |
| --- | --- |
| 1 | 首批 40 个 candidate review group（§19 的「先审前 30」区间） |
| 2 / 3 | 之后各 50 个，**据第一批实际产出重新排序**后再定 |
| 4 | 32 个 blind ROI frame（隐藏模型输出） |
| 5 | 备选池，不提前固定 |

排序是简单 greedy set cover（不用 embedding），bucket 维度：
`camera / location 4×3 grid / size(sqrt(w*h) <16,16–32,≥32) / time early-mid-late /
appearance 粗颜色 / background 亮暗+纹理 / source(turhancan-only|yolo-only|both) /
disagreement(tile 截断)`。优先选能补最多未覆盖 bucket 的 group，最后保留一小撮
seeded random group。

优先级 tier（§6）：

- `P1` Turhancan-only 且 conf ≥ 0.25 且位置/外观未覆盖 → student miss
- `P2` YOLO 在真实工作阈值附近（0.12–0.40）→ hard negative
- `P3` 双方都有但覆盖不足；`P3b` 双方都有且已覆盖；YOLO-only 不在工作带
- `LOW` Turhancan 极低分

重复背景折叠（§18）：同一 `(camera, location cell, size class, 粗颜色)` 的重复观测
只留 1 个 primary，其余标 `duplicate_background`；**运行成本仍按每次误报计**。

---

## 7. Blind mining（§17）

第一批 TRAIN 固定 32 个 blind ROI frame（每 camera 8）：16 个真正随机、16 个补
日期/时段/背景覆盖。blind unit 在用户给出 R/I/U/N 之前，服务端**不序列化任何模型
输出**（连候选 URL 都不发）；判定后才揭示。若发现是以前同一件垃圾，走 group 关联，
不重做整套标注。

---

## 8. Review UI

```bash
ssh -N -L 18812:127.0.0.1:8812 sf01@<host> -p <port>
# 浏览器打开 http://127.0.0.1:18812/
```

| 键 | 作用 |
| --- | --- |
| `R / I / U / N` | 必需垃圾 / 微小忽略 / 不确定 / 非垃圾背景 |
| `1 / 2 / 3 / 0` | bbox 阶段候选 A/B/C；`0` = `UNLOCALIZED_REQUIRED`（**不是 Ignore**） |
| `S / W / D` | suspected same-object：SAME / NEW / UNCERTAIN |
| `← / →` | 上/下一个 review group |
| 多帧 group | 可切 normal / hard state / before-after |

bbox 候选：A = Turhancan 语义框，B = Step 2B `last.pt` 框，C = classical point-seeded
proposal；近乎重复的框会被去掉，最多 3 个。

---

## 9. 运行手册（服务器侧）

代码与产物都在 `/home/sf01/ground-litter-historical-v2/`：

```
code/        本次上传的最小代码包（scripts + rtsp_annotator 子集 + ROI config）
artifact/    manifest / frames / 候选表 / review units / review 决策
logs/        长任务日志
run_coarse.sh
```

```bash
PY=/home/sf01/ground_litter_train/.venv/bin/python
ROOT=/home/sf01/ground-litter-historical-v2
ART=$ROOT/artifact
CFG=$ROOT/code/config

# 1. 冻结 window manifest（本地生成后上传；不重新查询就不会漂移）
#    scripts/build_ground_litter_historical_windows.py

# 2. 粗扫：Day-1 20 个 TRAIN window 下载 + 取帧 + 双模型推理
$PY -B code/scripts/run_ground_litter_historical_coarse.py \
    --artifact $ART --roi-config-dir $CFG --selection day1 --device cpu

# 3. 缓存 DEV/FINAL（只取帧，不推理，保持独立）
$PY -B code/scripts/run_ground_litter_historical_coarse.py \
    --artifact $ART --roi-config-dir $CFG --selection dev  --skip-inference
$PY -B code/scripts/run_ground_litter_historical_coarse.py \
    --artifact $ART --roi-config-dir $CFG --selection final --skip-inference

# 4. episode / review_group / diversity queue + blind frames
$PY -B code/scripts/build_ground_litter_historical_review.py --artifact $ART

# 5. 隔离 smoke（artifact 前后逐字节一致）
$PY -B code/scripts/smoke_ground_litter_historical_v2.py --artifact $ART

# 6. 审核服务（独立端口 8812，独立 artifact）
$PY -B code/scripts/serve_ground_litter_historical_review.py \
    --artifact $ART serve --bind 127.0.0.1 --port 8812

# 7. 第一阶段报告
$PY -B code/scripts/report_ground_litter_historical_phase1.py \
    --artifact $ART --review-url http://127.0.0.1:18812/
```

---

## 10. 安全与隔离

- v2 使用独立 artifact 与独立端口 **8812**（本地隧道 **18812**）；
- 官方 **8801** 审核环境从不访问；Sealed 从不访问；
- 官方 Step 2C artifact `/home/sf01/step2c1-blind-truth` 只读拒绝写入；
- 旧 Rapid v1 artifact 不 reset、不修改；
- DEV/FINAL 观测被硬过滤，永远不进入训练侧 review queue（§22 leakage）；
- 本阶段**不训练、不导出 dataset、不代替用户审核**。

---

## 11. 本阶段明确没做

- 没有人工真值，所以**没有**任何 accuracy / recall / FP 数字；
- 没有 dense 加密扫描（只有触发 §7 条件的窗口才做）；
- 没有 SAM/segmentation；先审 30 个 Required，若 >20% 出现 A/B/C 全不可用，
  才做 20 个失败样本的局部 spike（开发+接入 ≤1 小时，且要能解决 ≥50% 才接入）；
- 没有 hard-negative 60 簇的完整挖掘，只有候选队列；
- 没有 V2 训练、没有 threshold selection、没有 Final。
