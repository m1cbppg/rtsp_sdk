# v2 First Batch — 只读诊断（未修改 artifact / queue / split）

生成时间：2026-09-29T17:32:37
admission floor: `conf >= 0.15`（本轮诊断的实际门槛）

## Camera Funnel（TRAIN 20 windows）

| camera | raw Turhancan ≥0.01 | raw YOLO ≥0.01 | firing tiles | merged obs | obs ≥0.15 | episodes(0.15) | groups(0.15) | groups(no floor) | ≥0.15 | 0.05–0.15 | 0.01–0.05 | first batch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 01021 | 628 | 6 | 69 | 629 | 94 | 44 | 34 | 97 | 31 | 16 | 50 | 14 |
| 01022 | 754 | 25 | 76 | 758 | 107 | 34 | 24 | 108 | 18 | 24 | 66 | 8 |
| 01027 | 47 | 0 | 14 | 47 | 4 | 2 | 2 | 13 | 2 | 3 | 8 | 1 |
| 01030 | 501 | 18 | 85 | 502 | 110 | 50 | 38 | 158 | 35 | 40 | 83 | 17 |

`firing tiles` = 至少产出一个 raw box 的 ROI tile 数（从已有 raw_candidates 统计，未重新推理）。

## Why 01027 Has Only 1

- **A. 本来就没有候选（供给太小）** — yes：ROI 只占整帧 0.0085（像素 bbox [2276, 663, 2552, 935]，一条窄带）；5 个 day-1 window 里只有 14 个 ROI tile 真的产出过框，全四路 raw 只占 2.4%（47 个 Turhancan、0 个 YOLO）；零候选 window: ['01027_2026-09-27_1340']；候选不足 3 帧的 window: ['01027_2026-09-26_0916(2/3)']
- **B. 有候选但大部分 <0.15** — yes：no-floor groups: {'0.01-0.05': 8, '0.05-0.15': 3, '>=0.15': 2}；低于 0.15 的 group = 11
- **C. clustering 合并掉了** — minor：47 observations -> 13 episodes -> 13 groups (no floor)
- **D. diversity ranking 没选它** — partial：queue 里 01027 有 2 个 candidate unit，first batch 只进了 1；（queue 总长 98，first batch 40，所以这是排序位置问题而不是被丢弃）
- **E. 其它原因（下载/解码失败）** — no：01027 的 TRAIN day-1 window 全部 extraction_status=done；short PS 记录 4

## Conf Distribution

| camera | official groups (≥0.15 分组) | no-floor groups | ≥0.15 | 0.05–0.15 | 0.01–0.05 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 01021 | 34 | 97 | 31 | 16 | 50 |
| 01022 | 24 | 108 | 18 | 24 | 66 |
| 01027 | 2 | 13 | 2 | 3 | 8 |
| 01030 | 38 | 158 | 35 | 40 | 83 |

## Low Conf Diagnostic

共 17 个（每 camera 最多 5）：01021=5，01022=5，01027=2，01030=5

## Balanced First Batch Preview

共 38 个：01021=12，01022=12，01027=2，01030=12
与原 first batch 重合 33，新增 5，去掉 7（原 queue 未被修改）。

## Recommendation

1) 候选供给本身极不均匀：queue 中每 camera candidate unit = 01021=34，01022=24，01027=2，01030=38（合计 98）。first batch 原 40 的分布 14/8/1/17 基本按供给比例分配，不是 selector 把某一台藏起来。供给不足 8 的 camera：['01027']；供给足够但 first batch 低于 8 的 camera：无。
2) 因此 balanced preview（每 camera ≤12、供给不足就全给）会得到 38 个：01021=12，01022=12，01027=2，01030=12，与原 first batch 重合 33、新增 5、去掉 7。它牺牲的是 01021/01030 的高 gain 排序位，换来四路均衡。
3) admission floor：no-floor 聚类共 376 个 group，其中 <0.15 的有 291 个，official ≥0.15 是 98。关键在于这 291 个里 **248 个是重复背景**（同一机位/格位/尺寸/粗颜色的反复检出，被 duplicate filter 排除），只有 43 个进入低置信度池（0.05–0.15 仅 10 个，0.01–0.05 有 33 个）。也就是说把 0.15 降到 0.05 主要放进来的不是新的独立垃圾实例，而是同一批固定背景；01027 的缺口即使完全移除 floor 也补不满 —— 它的问题是供给，不是门槛。
4) 纯数据结论：是否改 balanced 更多是策略选择，并保留原 40 queue 作为对照；LOW_CONF_DIAGNOSTIC 只作为门槛校准样本，不要混进正式 first batch，也不要自动进训练。
5) 01027 的低置信度池只有 2 个（目标 5）；说明该机位在这 5 个 day-1 window 里本来就没有多少可用的独立目标，要提升它必须换 window/时段或独立标定 ROI，而不是放宽门槛。
