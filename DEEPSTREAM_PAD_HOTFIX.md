# DeepStream 8 输出流畅性热修复

这个增量补丁包含：

- 修复 `nvstreamdemux` 与 `rtspclientsink` 的请求 pad 连接；
- 修复中文标注的 DeepStream 8 OSD 属性；
- H.264 每秒发送 IDR，并在 IDR 重发 SPS/PPS；
- 禁用 B 帧，以减少实时播放重排延迟；
- 编码器前采用短时无损队列，避免短暂阻塞破坏参考帧连续性；
- 编码后的 H.264 AU 按时间戳匀速送入 RTSP 发布器；
- `publish FPS` 改为编码后采样，另增加 `pre-encode FPS`。
- worker重组时的优雅退出等待由12秒缩短为默认0.5秒，避免
  `POST /v1/streams`和`DELETE /v1/streams/{id}`长时间阻塞。

## 在服务器应用

假设原部署目录是 `~/rtsp-deepstream`：

```bash
mkdir -p ~/rtsp-smooth-hotfix
python3 -m zipfile -e \
  ~/rtsp-deepstream8-pad-hotfix.zip \
  ~/rtsp-smooth-hotfix

chmod +x \
  ~/rtsp-smooth-hotfix/scripts/apply_deepstream_pad_hotfix.sh

~/rtsp-smooth-hotfix/scripts/apply_deepstream_pad_hotfix.sh \
  ~/rtsp-deepstream
```

构建过程完全离线，只复用服务器已有的大镜像并增加一个很小的代码层。
脚本会将旧镜像保留为：

```text
rtsp-yolo-annotator:deepstream8-amd64-before-padfix
```

不需要重启 Docker 服务。脚本只会强制重建 `api` 容器，不重启
`mediamtx`。但是 API 内存里的流记录和 worker 会停止，补丁完成后需要
重新调用 `POST /v1/streams`，获得新的 RTSP URL。

## 验证

```bash
cd ~/rtsp-deepstream
docker compose -f docker-compose.deepstream.api.yml ps
docker compose -f docker-compose.deepstream.api.yml logs \
  --tail=200 api
```

重新调用 `POST /v1/streams`。稳定 15 秒后，日志中的 `pipeline`、
`pre-encode`、`publish` 应基本相等。例如 25 FPS 源流三者都应接近
25 FPS；若 `pre-encode` 接近 25 而 `publish` 明显偏低，瓶颈已确定在
NVENC 或发布端。

## 回滚

```bash
docker tag \
  rtsp-yolo-annotator:deepstream8-amd64-before-padfix \
  rtsp-yolo-annotator:deepstream8-amd64

cd ~/rtsp-deepstream
docker compose -f docker-compose.deepstream.api.yml \
  up -d --no-deps --force-recreate api
```
