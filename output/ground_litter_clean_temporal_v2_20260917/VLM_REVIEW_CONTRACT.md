# VLM review contract

VLM 只复核候选，不从整图主动寻找垃圾。每条 JSONL 输入包含：

- `clean_reference_crop`：冻结 Clean Reference 的局部图；
- `current_crop`：同位置当前局部图；
- `context_crop`：包含周边语义的当前图；
- `anomaly_bbox`、`timestamp`、`track`：位置和 persistence；
- `suggested_state`：`ANOMALY_PENDING` 或 `ANOMALY_CONFIRMED`。

期望输出：`label`、`confidence`、`reason`。`label` 只能是：
`NEW_GROUND_OBJECT`、`OCCLUDED`、`FIXED_FACILITY`、
`ENVIRONMENT_CHANGE`、`CLEAR_NON_LITTER`、`UNCERTAIN`。

只有 `NEW_GROUND_OBJECT` 且通过业务垃圾类别规则时进入
`LITTER_CONFIRMED`；`UNCERTAIN` 保留为疑似新增地面异物。
