# 燃气瓶模型资源

运行时需要：

- `yoloe-26l-seg.pt`：YOLOE-26L 实例分割权重；
- `profiles/camera_01_ir.json`：当前固定机位的归一化视觉提示；
- `profiles/camera_01_ir.jpg`：与提示框对应的参考图。

模型权重受 `.gitignore` 管理，不应在缺少许可证确认时提交到代码仓库。离线
DeepStream 镜像构建前，打包脚本会检查这三个资源是否存在。新机位不需要训练，
但需要单独保存一张参考图并配置视觉提示框和排除区域。
