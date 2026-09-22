# 垃圾候选分类器训练操作教程

日期：2026-09-22  
操作者：用户  
阶段：训练与留出评估均已完成；正式结论见
`2026-09-22-ground-litter-candidate-filter-holdout-review.md`。

## 1. 这次训练什么

这次训练两个很小的线性分类头，不重训 YOLO 主干：

- Turhancan 垃圾模型的 576 维冻结视觉特征；
- YOLO26s 的 512 维冻结通用视觉特征。

输入图片已经转换成特征并缓存，所以本次不需要重新读视频、下载 PS 或跑检测模型。
每个分类头只有一层线性权重，目的是快速回答“当前数据能否把垃圾候选
和现场硬负例分开”。如果这一层都没有可靠增益，直接微调整个检测器的风险会更高。

本次按用户要求使用 `cuda:0` 训练线性头。特征矩阵、标签、`Linear` 层、加权
`BCEWithLogitsLoss` 和 AdamW 优化步骤都在 GPU 上执行。由于矩阵很小，GPU 不一定
比 CPU 更快；使用 GPU 不会把这次实验变成 YOLO 主干微调。

训练集固定为 1152 个明确语义候选：147 正例、1005 负例。不确定、框错和随机
网格都不会进入训练。

## 2. 为什么有五个 head

五个摄像头各对应一个“排除该摄像头”的 head。例如评估 01022 时，使用训练过程
中从未看过 01022 标签的 head。五路训练集外预测合并后，才选择满足 90% 目标召回
的阈值。这避免用训练集自身的高分假装泛化能力。

产物同时保留五个 head。后续留出评估以“对应摄像头被排除的 head”为主口径；五头
平均只作为探索指标。

## 3. 从画面到分类结果的数据流

```text
监控帧
  ↓ Turhancan 低阈值检测
候选框 bbox
  ↓ 从原图截取候选区域
候选 crop 小图
  ↓ 冻结模型 embed(imgsz=320)
576维或512维特征向量
  ↓ 本次训练的线性分类头
一个实数分数 logit
  ↓ 与训练集外校准阈值比较
保留候选 / 过滤候选
```

Turhancan 仍然负责“在哪里可能有垃圾”。本次分类头负责“这个已提出的候选是否更像
现场垃圾”。如果 Turhancan 完全没有产生框，分类头没有输入，也无法找回该物体。
所以这一步主要解决候选误报和候选排序，不等同于训练一个新的完整垃圾检测器。

## 4. 专业名词解释

- **候选框（bounding box / bbox）**：模型在画面中圈出的矩形，格式为左上角和
  右下角坐标 `[x1,y1,x2,y2]`。
- **crop**：按候选框从原图截下来的小图。审核卡中的无框细节图就是 crop。
- **模型权重**：神经网络通过训练获得的大量数字参数。这里 Turhancan 和 YOLO26s
  的原权重保持不变，因此称为“冻结”。
- **特征或 embedding**：模型把一张 crop 压缩成的一串数字。576 维表示 576 个
  数字；它们不是人工定义的颜色、面积等字段，而是模型学习出的视觉描述。
- **线性分类头（linear head）**：一个很小的分类器，计算 `z = w·x + b`。
  `x` 是特征，`w` 是要学习的权重，`b` 是偏置，`z` 是分数。
- **logit**：线性层输出的原始实数。越大越倾向垃圾；它不是百分比概率。
- **L2 归一化**：把每个特征向量缩放到相同长度，避免只因数值整体较大而得高分。
- **标准化**：每一维减去训练折均值，再除以标准差，让各维尺度接近。均值和标准差
  只从训练摄像头计算，避免提前看到被排除摄像头。
- **标签**：人工答案。`LITTER=1` 是正例，`NON_LITTER=0` 是负例。
- **类别不平衡**：负例 1005、正例 147，负例远多于正例。若不处理，模型可能靠
  一直猜“非垃圾”得到表面上的高准确率。
- **BCEWithLogitsLoss**：二分类常用的损失函数。损失越小，模型输出和标签越一致。
  它直接接受 logit，数值上比先算概率更稳定。
- **pos_weight**：正例损失权重。本实验每个训练折使用 `负例数/正例数`，让漏掉
  一个垃圾付出更高代价。
- **优化器 AdamW**：根据损失的梯度更新 `w` 和 `b` 的算法。
- **学习率 0.015**：每一步更新幅度的控制量；过大可能震荡，过小学习太慢。
- **weight decay 0.05**：限制权重无限变大，降低记住训练样本细节的风险。
- **step**：一次完整的“算分→算损失→反向传播→更新权重”。这里每个 head 训练
  500 step，使用全部训练折样本，属于全批量训练。
- **过拟合**：训练数据表现很好，但新数据表现差。独立留出集用来暴露这种情况。
- **OOF（out-of-fold，训练集外预测）**：某条样本只能由没有用它所属摄像头训练的
  head 预测。它比直接报告训练集成绩更接近新场景表现。
- **阈值（threshold）**：分数高于阈值就保留。降低阈值通常提高召回，也增加误报。
- **召回率（recall）**：真实垃圾中被保留的比例，`TP/(TP+FN)`。目标为至少 90%。
- **精确率（precision）**：被保留候选中真实垃圾的比例，`TP/(TP+FP)`。
- **负例减少率**：原来非垃圾候选中被过滤掉的比例。它直接表示误报工作量下降多少。
- **留出集（holdout/test set）**：训练时完全不读取答案的一组考试数据。本次有
  200 张卡，其中 178 张有明确标签，可评分为 21 正、157 负。

## 5. 第一步：登录和只读检查

```bash
ssh -p 21002 sf01@14.21.88.97
cd ~/ground-litter-audit
test ! -e candidate_filter_training_v1 && echo "输出目录可用"
ls -lh classifier_manifest.json \
  classifier_experiment/turhancan/embeddings_turhancan.npz \
  classifier_experiment/yolo26s/embeddings_yolo26s.npz \
  active_day3/active_selection/pool_turhancan.npz \
  active_day3/active_selection/pool_yolo26s.npz \
  active_day3/active_selection/reviews_frozen.json
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader
```

`test` 没有输出且后面的“输出目录可用”出现，表示不会覆盖已有实验。`ls` 用于确认
六个输入存在。`nvidia-smi` 用于确认 GPU 0 与显存状态。本步骤不训练。

再检查训练容器中的 PyTorch 是否真正看见 GPU：

```bash
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -v "$PWD":/work \
  -w /work \
  rtsp-yolo-annotator:deepstream8-ground-litter-v32-hardening-20260918 \
  python3 scripts/train_ground_litter_candidate_filter.py probe --device cuda:0
```

必须看到 `cuda_available: true`、`resolved_type: cuda` 和显卡名称。探针只读取运行
环境，不加载训练数据，也不创建模型。

## 6. 六类输入如何使用

1. `classifier_manifest.json`：前两批 955 个明确语义候选的身份、摄像头、日期、
   原模型分数、框和二元标签，共 125 正例、830 负例。
2. `classifier_experiment/turhancan/embeddings_turhancan.npz`：上述 955 个 crop
   经冻结 Turhancan 模型 `embed(imgsz=320)` 得到的 576 维特征，包含逐行
   `sample_ids`，脚本严格按 ID 对齐，不依赖数组位置猜测。
3. `classifier_experiment/yolo26s/embeddings_yolo26s.npz`：相同 955 个 crop 的
   512 维通用视觉特征，身份契约相同。
4. `active_day3/active_selection/selection.json` 与 `reviews_frozen.json`：第三批
   300 张卡的选择元数据和人工答案。脚本只接受 240 个语义候选中的 22 个
   `LITTER` 和 175 个 `NON_LITTER`；36 个 `UNCERTAIN`、7 个 `BOX_WRONG`
   被排除，60 个 `random_grid` 即使标成非垃圾也不会混入候选分类器。
5. `active_day3/active_selection/pool_turhancan.npz`：第三日 2812 个语义候选的
   Turhancan 特征。脚本按 `proposal_id` 只取上述 197 个明确主动学习样本。
6. `active_day3/active_selection/pool_yolo26s.npz`：相同 2812 个候选的 YOLO26s
   特征，同样只按 ID 取 197 个。

拼接后恰好是 1152 行：147 正例、1005 负例。任一 ID 缺失、重复或计数变化，
脚本都会中止，不会发布模型。

## 7. 第二步：由你执行 GPU 训练

在服务器的 `~/ground-litter-audit` 目录执行：

```bash
docker run --rm --gpus all \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -v "$PWD":/work \
  -w /work \
  rtsp-yolo-annotator:deepstream8-ground-litter-v32-hardening-20260918 \
  python3 scripts/train_ground_litter_candidate_filter.py fit \
  --base-manifest classifier_manifest.json \
  --active-selection active_day3/active_selection/selection.json \
  --active-reviews active_day3/active_selection/reviews_frozen.json \
  --base-embedding turhancan=classifier_experiment/turhancan/embeddings_turhancan.npz \
  --base-embedding yolo26s=classifier_experiment/yolo26s/embeddings_yolo26s.npz \
  --pool-embedding turhancan=active_day3/active_selection/pool_turhancan.npz \
  --pool-embedding yolo26s=active_day3/active_selection/pool_yolo26s.npz \
  --target-recall 0.90 \
  --device cuda:0 \
  --output candidate_filter_training_v1
```

`--gpus all` 把服务器 GPU 暴露给一次性容器，`--device cuda:0` 强制训练张量位于
第一张 GPU。`--user "$(id -u):$(id -g)"` 让容器使用当前 `sf01` 用户的 UID/GID
写文件，避免产物变成仅 root 可读；`HOME=/tmp` 给非 root 容器一个可写的临时主目录。
容器加 `--rm`，训练结束后自动删除；挂载目录中的模型产物会保留。

脚本拒绝覆盖既有输出。中途失败时只留下自动清理的临时目录，不会发布半成品。

## 8. GPU 内部具体做什么

对 Turhancan 和 YOLO26s 两套特征分别执行：

1. 每个特征向量先做 L2 归一化；
2. 在当前训练折内计算逐维均值和标准差，再标准化，防止验证摄像头信息泄漏；
3. 轮流排除一个摄像头，用其余四路训练 `Linear(576,1)` 或 `Linear(512,1)`；
4. 正负比例约 1:6.8，使用 `BCEWithLogitsLoss(pos_weight=负例数/正例数)`，避免
   分类器只猜非垃圾；
5. 使用 AdamW，学习率 0.015、weight decay 0.05，全批量训练 500 步；
6. 被排除摄像头只用于产生训练集外预测，不参与该 head 的均值、标准差和权重；
7. 五路训练集外预测合并，在正例分数中选择达到 90% 召回的阈值；
8. 保存五组权重、bias、归一化参数、摄像头身份、阈值和 SHA-256。

两套模型共训练 10 个 head。脚本在 CUDA 训练前后同步计时，并把每个 head 的训练
秒数写入报告。候选分数计算和 JSON 输出在 CPU 完成。

## 9. 第三步：检查训练产物

```bash
find candidate_filter_training_v1 -maxdepth 1 -type f -printf '%f\t%s bytes\n' | sort
python3 - <<'PY'
import json
from pathlib import Path
p = Path("candidate_filter_training_v1/FIT_REPORT.json")
d = json.loads(p.read_text())
print("训练样本:", d["training"])
print("计算设备:", d["compute"])
print("是否读取留出答案:", d["holdout_labels_read"])
for name, result in d["models"].items():
    print(name, result["camera_oof_metrics"])
PY
```

正常时应有：

- `turhancan.npz`；
- `yolo26s.npz`；
- `FIT_REPORT.json`；
- `training.rows = 1152`、`positive = 147`、`negative = 1005`；
- `holdout_labels_read = false`。
- `compute.requested = cuda:0`、`compute.cuda_available = true`；
- `compute.gpu_name = NVIDIA GeForce RTX 3060 Ti`。

OOF 召回接近 0.90 是阈值设计结果，本身不代表通过。此阶段重点看负例减少率，但
最终判断必须等独立留出评估。

## 10. 运行后怎么交给我检查

执行完不要改阈值、不要重复训练，也不要创建第二个输出目录。回复“训练跑完了”，
并可粘贴终端最后的 JSON。我会只读核对产物哈希与训练报告，然后才把已冻结的
200 张留出答案上传服务器，并给你第二条 `evaluate` 命令。

这种顺序保证第一次模型选择没有看过考试答案。留出评估完成后，再一起解释召回、
负例减少、加权指标和是否值得进入新日期 shadow。
