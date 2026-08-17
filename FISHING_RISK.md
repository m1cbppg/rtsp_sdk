# 仅凭监控生成疑似非法捕捞线索

## 定位和边界

`fishing_risk`是船舶检测后的可选规则层，不是新的视觉模型。它只输出“疑似
非法捕捞人工复核线索”，不会自动认定违法，也不尝试识别渔网、渔具或具体捕捞
动作。

它依赖`vessel_detection`提供稳定船框，然后按每个固定机位配置的水域和时间表
判断：

- 船舶是否在受限时空内持续出现；
- 是否在较小范围内长时间停留；
- 是否在短时间内多次反向航行或折返。

所有坐标都是相对画面宽高的归一化坐标。更换机位或云台预置位后必须重新标定
区域；只改变同一机位的分辨率无需重画。

## 创建时开启

下面的请求同时开启船舶框和风险分析。`schedules`为空表示所有时间都分析；正式
使用时应填写当地正式禁渔通告对应的带UTC偏移量时间范围。

```json
{
  "input_url": "rtsp://camera/live",
  "vessel_detection": {
    "enabled": true,
    "confidence": 0.10,
    "imgsz": 1280,
    "class_ids": [8],
    "roi": [[0,0.55],[0.45,0.45],[1,0.43],[1,1],[0,1]]
  },
  "fishing_risk": {
    "enabled": true,
    "timezone": "Asia/Shanghai",
    "zones": [
      {
        "id": "protected_water",
        "polygon": [[0,0.55],[0.45,0.45],[1,0.43],[1,1],[0,1]]
      }
    ],
    "schedules": [
      {
        "id": "closure_2026",
        "start_at": "2026-05-01T12:00:00+08:00",
        "end_at": "2026-09-16T12:00:00+08:00"
      }
    ],
    "rules": {
      "minimum_presence_seconds": 30,
      "loitering_seconds": 180,
      "loitering_radius_box_lengths": 4,
      "reversal_window_seconds": 120,
      "minimum_reversals": 2,
      "startup_grace_seconds": 60
    },
    "restricted_presence_score": 40,
    "loitering_score": 20,
    "direction_reversal_score": 20,
    "alert_score": 60,
    "cooldown_seconds": 300,
    "display_risk": true
  }
}
```

`fishing_risk.enabled=true`而`vessel_detection.enabled=false`会返回HTTP 422，避免
创建一个没有船舶轨迹来源的风险任务。

## 只识别船

创建时省略`fishing_risk`，或者明确关闭：

```json
{
  "vessel_detection": {"enabled": true},
  "fishing_risk": {"enabled": false}
}
```

关闭状态下不建立风险轨迹、不计算分数、不显示风险文字，也不会生成新的疑似
捕捞事件；船舶检测参数和船框保持原样。

已有任务可以运行中关闭，保持相同`stream_id`和`rtsp_url`：

```bash
curl -X PATCH \
  -H 'X-API-Key: API密钥' \
  -H 'Content-Type: application/json' \
  -d '{"enabled":false}' \
  'http://服务器:端口/v1/streams/STREAM_ID/fishing-risk'
```

重新开启时，PATCH请求体使用与创建请求相同的完整`fishing_risk`对象。切换会重载
任务所在的DeepStream组，因此输出流可能有短暂抖动；失败时管理器恢复原配置。

## 分数和事件

默认评分：

- 受限时空内持续出现超过30秒：40分；
- 小范围停留超过180秒：20分；
- 120秒内至少发生2次有效折返：20分；
- 总分达到60分：生成一次人工复核事件。

事件类型为`suspected_illegal_fishing`，通过现有接口查询：

```text
GET /v1/streams/{stream_id}/events
POST /v1/events/{event_id}/confirm
POST /v1/events/{event_id}/reject
GET /v1/events/{event_id}/snapshot
```

事件`metadata`包含`risk_score`、`reasons`、区域、停留秒数、折返次数、长期风险
轨迹ID和底层船舶ID，并固定包含：

```json
{
  "legal_conclusion": false,
  "review_required": true
}
```

同一艘船在同一区域的一次连续停留只告警一次。离开后重新进入仍受
`cooldown_seconds`限制，防止边界抖动反复报警。

## 误报控制

`startup_grace_seconds`用于吸收任务启动时画面里已经存在的停泊船。预热期内建立
的船舶轨迹先作为存量目标，只有移动超过
`preexisting_activation_box_lengths`个自身框尺度后才进入风险分析。

速度、停留半径和匹配距离都以船框尺度归一化，而不是使用固定像素，因此远处小船
不会仅因为像素移动较慢就天然判定为停留。底层船舶ID因短暂漏检发生变化时，风险
层还会按位置和船身尺度重新关联，并在`track_lost_seconds`内保留长期状态。

现场调参顺序：

1. 先保证船舶ROI和风险zone准确，排除码头正常停泊区和航道；
2. 再调整`minimum_presence_seconds`和`loitering_seconds`；
3. 误报仍多时提高`alert_score`，不要先把船舶检测阈值调高；
4. 每个机位至少用正常通航、停泊、折返和真实执法线索录像分别验收。

仅凭监控不能核验捕捞许可证、AIS状态、渔具是否合法或是否实际获得渔获物。事件
只能用于缩小人工巡查范围，最终结论必须结合当地规则和执法核查。
