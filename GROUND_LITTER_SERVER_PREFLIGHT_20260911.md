# 零散垃圾影子试点：服务器只读预检

检查时间：2026-09-11 08:56（Asia/Shanghai）。用户本次授权范围为只读 SSH；本次未上传、安装、
构建镜像、修改配置、创建业务流、启动试点进程或重启服务。凭据只在现有 API 容器内存中用于
查询，不记录凭据或 RTSP 地址。

## 当前实测

| 项目 | 结果 |
| --- | --- |
| 目标 | `sf01@14.21.88.97:21002`，`/home/sf01/rtsp-deepstream` |
| API `/health` | HTTP 200，`status=ok` |
| API `/v1/streams` | HTTP 200，活动流数量 0 |
| API 镜像 | `rtsp-yolo-annotator:deepstream8-amd64-demo-continuous-20260910` |
| 镜像 ID | `sha256:8f4bf64b95f47ea551b66b3e6ec5f734c5769c71029386e0d8decc163693db88` |
| API 启动时间 / 重启次数 | 2026-09-10 02:28:19 UTC / 0 |
| MediaMTX 启动时间 / 重启次数 | 2026-09-06 06:38:27 UTC / 0 |
| 有效 Compose | base + `docker-compose.ptz-v12.override.yml` + `docker-compose.demo-continuous.override.yml`，静默校验通过 |
| GPU | RTX 3060 Ti，8,192 MiB；显存使用 94 MiB；采样 GPU 利用率 0% |
| 驱动 | 595.84 |
| 磁盘 | 根分区 468 GB，剩余 337 GB（`df -h`） |
| 容器 Python | 3.12.3 |
| Torch / Ultralytics | 2.11.0+cu128 / 8.4.107 |
| OpenCV / NumPy | 4.11.0.86 / 1.26.4 |
| PyAV | 不存在 |
| 模型 | `yolo26s.pt` 存在；Turhancan 垃圾权重不存在 |
| 新识别模块 | runner、runtime、source、inventory 均不存在 |

默认 `deepstream8-amd64` 标签指向另一张旧镜像（`c25cf4053e94...`）；不得将它误认作运行镜像。
依赖检查首次因远程命令引号错误失败，随后通过标准输入执行 Python 元数据查询，已取得上表结果。

## 结论与后续交付

服务器当前资源有余量，适合准备单路影子试点；这是空闲时的资源快照，不能证明五路 GPU 吞吐。
现有 API 容器不能直接运行本地 runner，尚缺 PyAV、新模块和权重；本次没有对现有容器安装软件。
本地验证使用的 Torch/Ultralytics 版本与服务器不同，候选环境还需完成实际模型加载和 CUDA 推理检查。

下一份部署交付应是独立的垃圾试点包：

1. 固定基础镜像与依赖版本，补齐 Linux PyAV、垃圾模块、权重和五路参考图/Profile；核对 SHA-256。
2. 完成日志落盘与轮转、拒绝分析原因、断流状态、证据保存、输出容量和进程停止检查。
3. 配置独立输出目录和独立容器，先只开启 1030，限时 30 分钟、通知关闭；不复用生产 Compose 更新命令。
4. 上传及启动前另行申请具体授权；不替换生产 API、不重启 MediaMTX。GPU 仍共享，试点需监测竞争，
   一旦已有任务性能退化就停止试点容器，保留日志和 SQLite。
5. 单路成功后再扩至五路，分别统计有效分析频率、帧龄、误报和抽样漏检；部署成功不等于准确率通过。

仅有本次只读授权不能执行上述上传、安装、镜像构建或容器启动。
