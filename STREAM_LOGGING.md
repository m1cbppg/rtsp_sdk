# RTSP逐流实时日志

日志系统把任务状态、实时帧率、帧间断、拉流重连和解码坏帧统一关联到
`stream_id`。每一路流都有独立JSONL文件，也可以通过HTTP读取或SSE实时跟随。

## 快速查看指定流

直接在服务器跟随文件：

```bash
tail -F data/stream-logs/STREAM_ID.jsonl | jq -c .
```

通过API查看最近200条：

```bash
curl -sS \
  -H 'X-API-Key: 你的API_KEY' \
  'http://服务器:8080/v1/streams/STREAM_ID/logs?limit=200' | jq
```

通过SSE实时跟随；断线重连时可把最后看到的`sequence`传给
`after_sequence`：

```bash
curl -N \
  -H 'X-API-Key: 你的API_KEY' \
  'http://服务器:8080/v1/streams/STREAM_ID/logs/live?after_sequence=0'
```

历史接口还支持：

- `level=WARNING`：只看某一级别；
- `event=playback.health`：只看播放健康日志；
- `after_sequence=123`：只返回指定序号之后的日志；
- `limit=1..5000`：控制返回数量。

## 如何判断播放状态

重点看`event=playback.health`及`details.health`：

| health | 含义 | 主要证据 |
|---|---|---|
| `healthy` | 播放正常 | 拉流和发布FPS正常，未见坏帧 |
| `degraded` | 仍可播放但质量下降 | FPS低于健康阈值、拉流重连 |
| `stalled` | 已判定卡顿、断流或管线无更新 | FPS接近0、超过500ms帧间断、指标超时 |
| `mosaic_risk` | 存在花屏或马赛克风险 | 解码坏帧`CORRUPTED`或缓冲`DISCONT` |
| `failed` | 流处理进程失败 | 非零退出码 |
| `starting` | 正在等待首帧 | 启动宽限期内 |
| `unobservable` | 进程在运行但无逐流指标 | 当前后端尚未上报指标 |

`mosaic_risk`表示系统发现了可观测的底层证据，并不冒充人工肉眼确认：

- `interval_corrupt_frames > 0`是解码链路直接报告的坏帧，证据最强；
- `interval_discontinuities > 0`表示视频缓冲不连续，可能表现为短时卡顿、
  花屏或画面跳变；
- 单纯低FPS只记为`degraded`或`stalled`，不会误报为马赛克。

每条日志的`details.metrics`保留原始指标，便于进一步定位输入、推理、编码或
发布环节。RTSP URL中的用户名和密码在落盘前会自动替换为`***:***`。

## 配置和持久化

在API JSON配置中加入：

```json
{
  "observability": {
    "enabled": true,
    "log_root": "/app/data/stream-logs",
    "monitor_interval_seconds": 1,
    "metrics_stale_seconds": 12,
    "stall_fps": 1,
    "degraded_fps_ratio": 0.8,
    "max_file_mb": 32,
    "backup_count": 3
  }
}
```

每个文件达到`max_file_mb`后轮转，最多保留`backup_count`个旧文件。Docker
部署需要把`/app/data`挂载到宿主机；项目CUDA和DeepStream API Compose模板
已包含该持久化目录。

阈值建议：

- 25 FPS摄像头可保持`minimum_healthy_fps=20`；
- 低帧率摄像头应同时降低DeepStream的`minimum_healthy_fps`，否则会持续记录
  `degraded`；
- `metrics_stale_seconds`应至少大于两倍`stats_interval_seconds`，默认12秒对应
  5秒指标周期。
