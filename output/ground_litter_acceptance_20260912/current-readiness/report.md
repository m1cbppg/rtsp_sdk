# 零散垃圾真值验收

机位：44180209031322001021；day；validation

状态：**无法计算**

生产接入：No-Go；通知：No-Go。本工具只计算样本指标，不自动批准上线。

待补齐：

- 尚未填写录像时长
- 缺少人工复核人或复核时间
- 未穷尽标注整个验收时段的地面垃圾
- 持续记录尚未全部人工复核
- 缺少原始录像 SHA-256
- 画面机位尚未经人工核对
- 至少需要两处画面时间/现场动作核验锚点
- 缺少日志到录像时间轴的人工核验映射
- 缺少日志快照指纹
- 未提供原始录像或录像指纹不匹配
- 日志快照与人工复核绑定的指纹不一致

```json
{
  "descriptive": {
    "observations": 145,
    "candidate_boxes": 7
  },
  "metrics": {
    "record_precision": null,
    "visible_episode_recall": null,
    "false_records_per_camera_hour": null,
    "duplicate_records": null,
    "merged_episodes": null,
    "erroneous_clears": null,
    "clear_recall": null,
    "confirmation_delay_p95_seconds": null
  }
}
```

- 候选框数量不是识别率；未计算逐框精度/召回率。
- 没有定义真负样本，整体准确率及 FP/(FP+TN) 无法计算。
- 置信区间仅描述样本不确定性；同场景相关样本不能代表生产总体。
