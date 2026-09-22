# Ground Litter V3.1：Actor Cascade 补充验证

- 结构上下文门后只对仍可见事件运行现有 actor/context 模型。
- 最终保留 6 个待 YOLO/VLM 判断事件，其中垃圾 5 个。
- 当前样本事件级候选 precision 为 83.3%，垃圾事件保留率为 100.0%。
- 非垃圾事件进入 `OCCLUDED` 的比例为 90.9%。
- 剩余非垃圾事件：negative-v31-event-013=186。

`OCCLUDED` 只暂停地面判断。Actor 结果不会写入 Clean Reference，也不会永久学习为非垃圾。
