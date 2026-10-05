# Ground Litter Historical Active Mining + Rapid Dataset v2 — 第一阶段报告

生成时间：2026-09-29T17:20:50
范围：仅 01021/01022/01027/01030 四路；01028 不参与。

## Historical Windows

- train downloaded: **20**（第一批目标 20）
- dev cached: **16**
- final cached: **16**
- windows done: **52**（manifest 中 download_status=downloaded）
- total bytes: **4.64 GB**
- source shorter than 270s（已按可用帧降级并记录）: **6**，例如 01021_2026-09-28_1545 缺 [270.0]
- 冻结 split：{"2026-09-23": "TRAIN", "2026-09-24": "TRAIN", "2026-09-25": "TRAIN", "2026-09-26": "TRAIN", "2026-09-27": "TRAIN", "2026-09-28": "DEV", "2026-09-29": "FINAL"}
- manifest SHA256: `1e10448b9d16debcc294484be6f0cff11fdfe4de909ab99c75d345c6e401fc94`

## Candidates

- Turhancan-only: **1887**
- YOLO-only: **6**
- both: **43**
- 观测总数: **1936**
- student FP candidates (P2, Step2B 工作阈值附近): **2**
- 重复背景观测（同一固定目标重复出现）: **206**

## Groups

- episodes: **130**
- review groups: **98**
- selected first batch: **40**（queue 总长 98，分批 {'1': 'first 40 candidate groups', '2': 'next 50', '3': 'next 50', '4': 'blind ROI frames (model hidden)', '5': 'reserve, re-ranked after batch 1'}）
- blind ROI units: **32**（每 camera 8）
- suspected same-object groups: **16**
- 多帧 group（有状态变化才 >1）: **16**

| camera | first batch |
| --- | ---: |
| 01021 | 14 |
| 01022 | 8 |
| 01027 | 1 |
| 01030 | 17 |

## Review UI

- URL: `http://127.0.0.1:18814/`

| 操作 | 说明 |
| --- | --- |
| `R / I / U / N` | 必需垃圾 / 微小忽略 / 不确定 / 非垃圾背景 |
| `1 / 2 / 3 / 0` | bbox 阶段：候选 A/B/C，`0` = UNLOCALIZED_REQUIRED |
| `S / W / D` | suspected same-object：SAME / NEW / UNCERTAIN |
| `← / →` | 上/下一个 review group |
| blind unit | 判定前隐藏全部模型输出，判定后才揭示候选 |

审核单位是 **review group**（一个独立垃圾实例 / 一个独立 hard-negative 簇），不是 frame。
同一固定垃圾连续出现的近重复帧默认只保留 1 个 NORMAL 代表帧，最多 3 帧。

## Safety

- Sealed = **0**
- official Step2C write = **0**
- 8801 automated access = **0**
- 旧 Rapid v1 artifact 未 reset、未修改；本阶段 split 与 eval 独立定义
- 未训练、未导出 dataset、未代替用户审核

## Efficiency

- downloaded video hours: **4.39 h**
- candidate groups / hour video: **22.3**
- blind units / hour video: **7.3**
- 抽取帧: **150**

---

不报告模型 accuracy：目前没有人工真值，任何准确率都会是伪造的。

STOP：请只审核第一批 30~50 个 review group，不要继续旧的 265-frame Rapid Review。
审核完成后回报，再据实算 Required yield / HN yield / 重复噪声比，然后决定下一批 50 的排序。
