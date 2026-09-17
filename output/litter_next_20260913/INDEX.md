# 1021首批实施日志索引

更新：2026-09-14。最新续验入口为[本机采样与模型复测](../litter_source_20260914/REPORT.md)。原四小时任务、两轮GPU对照、10分钟源探针均已结束；30分钟源探针已中止，无完整摘要。此前服务器门禁因生产指标陈旧而拒绝启动，不能将该故障归因为URL已过期。通知关闭。

- [实施结果与下一步](REPORT.md)
- [最新源诊断、安全退出与门禁续验](source_r3/REPORT.md)
- [原图和逐层结果对照页](regression/index.html)
- `baseline_manifest.json`：修改前本地文件/权重、四小时服务器runtime/profile指纹。
- `baseline_source/`：修改前源码；`server_audited/`子目录保留四小时runtime/profile/参考图。
- `gpu_release_manifest.json`：GPU诊断包内容和散列；`local_final_manifest.json`：包含随后状态修复的本地最终版本，二者分开。
- `regression/sample_manifest.jsonl`：34张原图的路径、SHA-256和固定探测位置；`labels_draft.json`：另存的助手目视标签、OSD时间及边界不确定性。
- `regression/small_object_probe.json`：已知位置辅助的两个小碎屑局部探测，不是全场景评测。
- `runs/cpu_baseline_1280/`、`runs/cpu_actor640/`：CPU两轮`run.json`、`results.jsonl`、`summary.json`。
- `runs/diagnostics-20260913-r1/gpu_actor1280/`、`gpu_actor640/`：GPU两轮完整结果及实际运行模块散列。
- `runs/diagnostics-20260913-r1/coexistence.jsonl`：68次生产流/GPU共存采样；`guardian_summary.json`：退出码、停止原因及前后指标。
- `server_preflight.json`、`server_postflight.json`：生产容器及运行流前后只读核查。
- `../litter_source_20260914/full_tests.log`：9月14日最新393项全量单测；`source_r3/full_tests_final.log`实际为此前389项，`filter_r2/full_tests_final_latest.log`为380项，根目录`full_tests_final.log`为369项版本。
- `filter_r2/source_live_1021/summary.json`：1021直播60秒取流诊断；`source_day/summary.json`、`source_night/summary.json`：昼夜本地录像取流诊断；`filter_r2/results.jsonl`：固定物过滤回归。
- `filter_r2/source_live_1021/summary_timing.json`：补充发布间隔探针；使用OpenCV回退路径，不能视为PyAV/端到端新鲜度验收。
- `filter_r2/source_live_1021/summary_10m.json`：1021路10分钟隔离冒烟；489帧发布、间隔P95约1.96秒，OpenCV回退且无PTS。
- `source_r3/local_day/`、`local_stop_fixed/`：真实PS回放、逐帧解码时间线和SIGTERM退出结果；`local_stop/`保留修复前退出失败的进度。
- `source_r3/production_diagnostic.json`：生产指标最后更新于UTC04:16:57的只读证据；`guard_preflight.json`：新门禁在启动前拒绝运行。
- `../litter_source_20260914/decision.json`：9月14日独立二次复核与门槛敏感性实验的放行决定。
- `../litter_source_20260914/deduplicated_labels.json`：用户复核的20帧标签按空间聚类后的2个独立垃圾物体，避免重复帧重复计数。
- `source_r3/server_config_diagnostic.json`：此前worker配置非敏感摘要；其中时间戳仅为提示，不能证明到期。已知故障是连续RTSP EOF和指标陈旧，根因未确定。
- `source_r3/release/`、`release-r1/`、`dependency_manifest.json`、`local_final_manifest.json`：PyAV16.1.0镜像输入和最终外部启动器，源码与wheel散列。
- `source_r3/offline_build.json`：实际镜像`630891dd1d05…`构建与PyAV16.1.0加载结果；`offline_validation.json`、`server_day/`、`server_night/`：服务器昼夜PS离线完整验证，各485帧解码/19帧发布。
- `source_r3/guard_preflight_final.json`：最终启动器仍在建容器前被陈旧生产指标拦截；`server_final.json`：最后容器核对。

服务器本轮只涉及隔离目录：

```text
/home/sf01/ground-litter-pilot-20260911/releases/diagnostics-20260913-r1
/home/sf01/ground-litter-pilot-20260911/active/diagnostics-20260913-r1
/home/sf01/ground-litter-pilot-20260911/releases/source-r3-20260913
/home/sf01/ground-litter-pilot-20260911/releases/source-r3-20260913-r1
/home/sf01/ground-litter-pilot-20260911/active/source-r3-guard-preflight-20260913
/home/sf01/ground-litter-pilot-20260911/active/source-r3-guard-preflight-20260913-r1
/home/sf01/ground-litter-pilot-20260911/active/source-r3-offline-build-20260913
```

早期源探针产物另在服务器`/tmp/litter_source_smoke_1021_20260913_r2`（10分钟）和`/tmp/litter_source_30m_1021_20260913`（中止，仅容器日志）。旧诊断容器已删除，中止的30分钟容器`festive_archimedes`亦已停止删除。未创建新的持久化直播reader。后续查看先读本索引和结果报告，再按run.json校验版本；不能从目录存在推断任务仍在运行。
