# RTSP 接口与真实画面验证（2026-09-13）

## 接口验证

公开 Swagger 定义确认：

- `POST /ims-mainte-pc/p-api/v1/monitor/play/ctseelink/devices/rtsp`
  的 operationId 是 `getRtspUrlUsingPOST`；请求支持 `deviceCode`、`playback`、
  `playbackTime`，但返回值只有 `data.url` 和 `expireTime`，没有内容时间戳。
- 历史回放另有 `POST /ims-mainte-pc/p-api/v1/monitor/play/replay`，必须提供
  `ip`、`channel`、`startTime`，可选 `endTime`。

本次向 `getRtspUrlUsingPOST` 请求 `playback=1`、`playbackTime=2026-09-08 12:00:00`，
成功得到 RTSP 并读取首帧；首帧 OSD 显示为 **2026-09-13 01:44:41**，不是请求的
2026-09-08 12:00:00。因此该调用在当前环境实际返回直播内容，不能当作历史回放。

## 真实垃圾画面与模型结果

首帧人工可见多处道路/人行道散落物。使用当前 Turhancan 分块模型、夜间配置、人车遮挡
过滤推理，得到 2 个稳定候选：

- 约 `[450,1230,496,1296]`，连续 60 秒均检出，类别在 Paper/Plastic 间变化；
- 约 `[556,630,587,663]`，连续 60 秒均检出，类别为 Plastic。

13 帧序列中两框位置基本不变，说明模型确实能在这一路的真实含垃圾画面产生稳定候选。
由于采样间隔为 5 秒，超过当前 3 秒时序最大间隔，所以没有升级为确认记录；这不代表
模型漏检，也不代表物体已被确认是垃圾。

## 下一步

要评估历史时段，需改用 `/play/replay` 并提供每路 `ip/channel` 映射，或让平台确认
`getRtspUrlUsingPOST` 的 `playbackTime` 参数已生效。当前探针图和序列结果仅作为直播
候选证据，未接入生产、未发送通知。
