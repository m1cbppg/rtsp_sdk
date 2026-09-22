# Ground Litter V3.3 Phase 0 架构决策

> **SUPERSEDED（2026-09-18）**：本文件将 prior-only 限制为人工审核，不能满足
> “先验独立补充模型漏检小垃圾”的产品目标。DS4.1 不得据此实施。唯一有效规格为
> `docs/plans/2026-09-18-ground-litter-v33-dual-recall-architecture.md`。

状态：已决策，可据此解除非生产工作的阻塞  
日期：2026-09-18  
依据：`2026-09-18-v33-phase0-blocker-brief.md`、
`output/ground_litter_v33_phase0_20260918/` 及七天回放可获得性

## 1. 决策结论

当前 `turhancan_yolov8m_seg_trash.pt` 不得作为“是否为现场垃圾”的硬语义闸门，
不得据其置信度直接产生自动垃圾告警。Phase 0 对这一部署形态的阻塞保持有效。

V3.3 改为两阶段推进：

1. `hybrid_v33_shadow`：Clean Reference 生成变化事件，turhancan 只提供弱语义特征；
   所有事件进入人工审核，产品标签统一为“地面异物候选”，不得进入垃圾 OSD 或自动告警。
2. `hybrid_v33_auto`：只有新的候选验证器通过独立留出集和生产 canary 后才能启用；
   届时才允许输出“疑似垃圾”。

短期选择相当于路线 C，目的是建立可用数据闭环；中期采用路线 A/B 的组合：先训练
场景候选验证器，再根据产品是否必须区分“垃圾”决定是否微调垃圾语义模型。当前不直接
选择 B，因为只有一个无歧义正例，数据不足以训练或证明有效。

## 2. Phase 0 证明了什么

已证明：

- 当前模型在已测机位和配置下不能用单一置信度阈值分开真垃圾与硬负例；
- semantic-only 全 ROI 自动输出不可用；
- 当前模型的类别 `Glass/Metal/Paper/Plastic/Waste` 更接近材质或废物外观，不能可靠
  判断一个桶、扫把或袋状物在现场是否属于“遗弃垃圾”；
- 调整命中次数或确认时间不能修复上游排序问题；
- 当前标注存在冲突，任何训练和最终验收前都必须先仲裁。

尚未证明：

- 所有先验与语义融合都无效。已有参考图中的固定误检可以被 prior 匹配过滤；
- 新的场景验证器或完成本域训练的模型也无效；
- 现有数据足以估计真实召回率。无歧义正例只有一件，置信区间没有实际意义。

因此不得把当前模型升级为生产裁决者，也不应删除融合基础设施和数据闭环工作。

## 3. 产品语义边界

系统内部必须区分三种结果：

| 结果 | 含义 | 是否显示“疑似垃圾” | 是否自动告警 |
|---|---|---:|---:|
| `GROUND_CHANGE_CANDIDATE` | 相对干净参考出现持续变化 | 否 | 否 |
| `REVIEWED_LITTER` | 人工确认是垃圾 | 可用于审核记录 | 由业务流程决定 |
| `AUTO_LITTER_CANDIDATE` | 已通过生产验收的验证器判定为垃圾 | 是 | 可配置 |

在 `hybrid_v33_shadow` 阶段，turhancan 的类别和置信度只作为审核排序字段，不改变
`GROUND_CHANGE_CANDIDATE` 的产品标签。

## 4. 解除阻塞后的工作范围

### 4.1 立即继续

- 修正 side input 为真正的 latest-wins，并记录输入帧年龄和丢帧数；
- 清走证据按真实 elapsed time 累积，并覆盖不规则采样间隔；
- 完成 `item-00x ↔ target_id` 标注仲裁；
- 建设七天回放采样、Profile Factory、多环境参考库和环境覆盖验证；
- 实现与具体模型解耦的 `CandidateEvidenceProvider` 接口；
- 实现 `hybrid_v33_shadow` 的候选事件导出、去重、审核和指标；
- turhancan prior-crop 批量推理可以实现，但输出只能进入弱证据和审核记录；
- 性能优化、Profile 对齐监控、未知环境采样和测试基础设施可以继续。

### 4.2 继续阻塞

- turhancan semantic-only 检测框进入 OSD；
- 以 turhancan 置信度确认“疑似垃圾”；
- `hybrid_v33_auto` 模式；
- 对外自动垃圾告警；
- 未经独立数据验证就调整阈值绕过 Phase 0。

## 5. 七天回放数据闭环

以事件为单位构建数据集，不能把相邻帧当成独立样本。

1. 第 1～5 天用于 Profile 发现、候选生成和训练数据准备。
2. 第 6 天用于阈值和模型选择。
3. 第 7 天保持盲测；最终用全七天构建 Profile 后，再做未来至少 48 小时影子验证。
4. 对七天回放运行 prior，按位置和时间合并为事件。
5. 对每个事件保存当前 crop、参考 crop、残差图、时间上下文、turhancan 原始输出、
   Profile regime 和 actor/context 信息。
6. 审核优先级为：高 turhancan 分数、持续时间长、跨 regime 出现、环境切换附近、
   先验与模型冲突。
7. 标注集合至少为：`litter`、`legitimate_tool_or_container`、`fixed_facility`、
   `actor_related`、`shadow_or_reflection`、`unknown`。
8. 数据切分必须按事件、日期和摄像头隔离，禁止同一事件的相邻帧跨训练/验证集。

七天回放预计能提供大量硬负例和环境样本，但不保证包含足够真实垃圾正例。
公开数据和合成数据可以用于预训练或管线测试，不能替代同机位真实正例验收。

## 6. 新验证器路线

新增统一接口，运行时不依赖某个具体模型：

```python
class CandidateEvidenceProvider(Protocol):
    def evaluate(self, batch: list[CandidateContext]) -> list[CandidateEvidence]: ...
```

`CandidateContext` 至少包含：

- 当前候选 crop；
- 对齐后的 reference crop；
- 归一化残差或变化 mask；
- 候选位置和透视尺度；
- 持续时间与出现方式；
- turhancan 类别、分数和可选中间特征；
- actor/context 重叠信息。

首个可训练验证器应回答“这是稳定的实体变化，还是光照/反射/压缩/遮挡”，用于
减少 prior 假变化。它可以支持“地面异物候选”，但不能单独承诺垃圾语义。

如果产品必须输出“疑似垃圾”，还需第二层本域分类器区分 `litter` 与
`legitimate_tool_or_container`。是否微调 turhancan、使用其 backbone 特征或训练独立
轻量分类器，由离线模型 bake-off 决定，不能预先指定胜者。

## 7. 自动模式重新开闸条件

`hybrid_v33_auto` 至少同时满足：

1. 标注冲突全部仲裁，有版本化真值清单；
2. 正负样本按独立事件统计，并按摄像头/日期隔离；
3. 至少先完成一个有 50 个独立真实垃圾事件的可行性集；生产验收应继续扩大，
   不得把同一事件的多帧计作多个正例；
4. 硬负例必须包含扫把、桶、袋状固定物、车辆局部、货架、阴影、反光和水渍；
5. 在独立留出集上报告逐事件 precision、recall、PR 曲线和置信区间；
6. 满足业务方书面确定的误报预算，不能只用当前任意阈值；
7. 七天负样本回放中没有持续自动垃圾事件；
8. 未来至少 48 小时生产 shadow canary 通过；
9. 性能满足主链 FPS、side P95、显存和帧年龄预算；
10. 自动模式仍保留不确定结果进入人工审核的出口。

50 个独立正例是重新进行模型可行性判断的最低样本量，不是生产充分性证明。

## 8. 环境闸门决策

不得简单把 `local_extent > 16` 改成更大的常数，也不把该阈值作为普通 API 参数交给
调用方。当前硬悬崖改由 Profile Factory 解决：

1. 每个环境 reference regime 在构建时统计 `local_extent`、gain、bias、饱和比例和
   对齐误差的经验分布；
2. Profile 中保存经过留出集验证的接受区间和来源统计；
3. 运行时使用 `MATCHED`、`UNCERTAIN`、`REJECTED` 三态和切换迟滞；
4. `UNCERTAIN/REJECTED` 时 prior 不累计确认或清走证据，turhancan 可继续做影子采样，
   但不得显示或告警；
5. 未见环境进入待审核样本池，生成新的不可变 Profile 版本后才能启用；
6. 视角改变与光照失配分开报告，视角改变必须重新标定。

在多环境 Profile 可用前，现有下午 Profile 只能声明覆盖已验证的相近时段，不能以放宽
阈值的方式宣称全天覆盖。

## 9. DS4.1 当前执行顺序

1. 完成 latest-wins 和 elapsed-time 清走修复及测试；
2. 完成标注仲裁清单与审核页；
3. 实现七天回放稀疏采样和可复现 manifest；
4. 实现 Profile Factory 的质量过滤、视角 epoch、环境聚类、参考合成和盲测；
5. 实现 `CandidateEvidenceProvider` 和 `hybrid_v33_shadow`；
6. 回放七天数据，生成去重事件和审核语料；
7. 基于已审核数据进行验证器 bake-off；
8. 满足第 7 节开闸条件后，另行评审 `hybrid_v33_auto`；
9. 未经评审不得部署或改变生产输出语义。

## 10. 对原 V3.3 规格的修订

本决策覆盖原规格中以下内容：

- turhancan 从硬语义闸门降为弱证据提供者；
- full ROI semantic-only 由自动候选来源改为影子诊断和样本发现；
- 原 §9 自动融合主链拆分为 shadow 数据闭环与 auto 生产路径；
- 原 Phase 0 未通过继续阻塞 auto，不再阻塞 Profile Factory、可靠性修复、接口和
  shadow 审核闭环；
- Profile 环境闸门由单一常量改为多环境经验分布和三态拒绝机制。
