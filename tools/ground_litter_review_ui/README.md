# 零散垃圾候选人工审核台

这是一个只在本机运行的离线审核页。它读取 V3 PoC 生成的候选清单，展示 Clean Reference、当前异常局部图和带定位框的上下文图。审核结果保存在浏览器本地存储中，可导出 JSON 或 CSV。

## 生成审核数据

在 `rtsp` 目录运行：

```bash
python tools/ground_litter_review_ui/build_review_data.py
```

也可以传入其他同格式清单：

```bash
python tools/ground_litter_review_ui/build_review_data.py --manifest /absolute/path/to/manifest.jsonl
```

## 启动

```bash
python tools/ground_litter_review_ui/serve.py
```

打开：

```text
http://127.0.0.1:8765/tools/ground_litter_review_ui/dist/index.html
```

## 快速审核

- 数字键 `1` 到 `8` 直接选择结论。
- 方向键或 `J` / `K` 切换候选。
- `Enter` 保存当前结论并前往下一条。
- 默认启用“分类后自动下一条”，适合连续审核。
- 定期导出 JSON；浏览器本地存储不应作为最终数据归档。

导出的结果包含审核集指纹。重新导入时会校验指纹，避免把结论写入错误的数据集。
