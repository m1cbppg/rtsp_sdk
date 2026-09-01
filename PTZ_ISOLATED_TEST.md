# 船舶识别与 PTZ 隔离联动测试

这套环境用于在直播画面没有船时，完整验证“发现小船 → 对准目标 → 多轮自适应放大 → 近景确认 → 截图 → 回 HOME”。它不会修改正式 API、MediaMTX、数据库或端口。

## 工作方式

含船回放视频由虚拟 PTZ 服务发布成 RTSP。正式船舶识别代码读取这路 RTSP，并把控制命令发给虚拟 PTZ 服务。每次 `locate` 后，虚拟服务会按目标坐标重新裁剪原视频，并按指令放大；识别服务随后看到的确实是新的近景，因此能测试真正的闭环，而不是伪造成功响应。

隔离资源如下：

- 测试 API：`18081`
- 测试 RTSP：`18554`
- 虚拟 camera_control：`19080`
- Docker 网络：`rtsp-ptz-isolated-test`
- 数据目录：`ptz-test-runtime/`

画面会固定显示 `ISOLATED PTZ TEST - NOT PRODUCTION EVIDENCE`，避免测试截图进入正式证据。

## 首次准备

在代码目录执行。`--public-host` 填现场电脑能够访问的服务器 IP：

```bash
cd ~/rtsp-deepstream

docker build \
  --build-arg BASE_IMAGE=rtsp-yolo-annotator:deepstream8-amd64 \
  -f Dockerfile.deepstream.ptz-update \
  -t rtsp-yolo-annotator:ptz-adaptive-fix-20260826 \
  .

python3 scripts/prepare_ptz_test_runtime.py \
  '/实际路径/8月12日.mp4' \
  --public-host 192.168.22.69 \
  --engine-dir /home/sf01/rtsp-deepstream/engines

docker compose \
  --env-file .env.ptz-test \
  -f docker-compose.ptz-test.yml \
  up -d --build

docker compose \
  --env-file .env.ptz-test \
  -f docker-compose.ptz-test.yml \
  ps
```

三个容器均为 `Up` 后注册测试流：

```bash
python3 scripts/register_ptz_isolated_stream.py
```

脚本会打印识别结果流和虚拟摄像机流。分别用 `ffplay -rtsp_transport tcp '脚本打印的地址'` 打开，建议两个窗口并排观察。

## 验收结果

```bash
python3 scripts/inspect_ptz_isolated_test.py
```

重点检查：

- `virtual_camera.rounds`：每一步的输入坐标、变焦步长和变焦前后状态；
- `virtual_zoom_changed=true`：实际发生过虚拟变焦；
- `verification_jobs`：最终结果、每轮目标尺寸以及结束原因；
- `capture_returned=true`：截图接口真实返回了 JPEG；
- `home_returned=true`：流程结束后回到全景 HOME。

截图位于 `virtual_camera.captures[*].result`，可这样下载：

```bash
source .env.ptz-test
curl -H "X-Camera-Control-Key: $PTZ_TEST_CONTROL_KEY" \
  'http://127.0.0.1:19080/v1/artifacts/图片ID' \
  -o ptz-test-capture.jpg
```

测试密钥保存在权限为 `0600` 的 `.env.ptz-test`，不要提交到 Git。

## 可选：把相同动作镜像到真实摄像头

先完成纯虚拟闭环，再启用真实镜像。回放视频必须来自同一台摄像头、同一 HOME 预置点；否则回放坐标与真实画面没有空间对应关系。

编辑 `.env.ptz-test`：

```text
PTZ_TEST_MIRROR_REAL_CAMERA=true
REAL_CAMERA_CONTROL_URL=http://camera-control:8080
REAL_CAMERA_ID=river-ptz-01
REAL_CAMERA_CONTROL_API_KEY=真实camera_control使用的KEY
```

让现有 `camera-control` 加入测试网络，再只重建虚拟控制容器：

```bash
docker network connect rtsp-ptz-isolated-test camera-control 2>/dev/null || true

docker compose \
  --env-file .env.ptz-test \
  -f docker-compose.ptz-test.yml \
  up -d --force-recreate ptz-test-camera-control
```

此时将测试识别结果、虚拟 PTZ 画面、真实摄像头直播三幅画面并排。每个 `locate/home/capture` 都会同时作用于虚拟画面和真实设备。真实截图位于 `captures[*].result.real_capture`，也通过 `19080/v1/artifacts/...` 下载。

这项测试能够验证控制方向、分步变焦、回位和真实抓图；只有回放 HOME 与当前真实 HOME 高度一致时，才能进一步验证落点精度。

## 停止

```bash
source .env.ptz-test
STREAM_ID="$(python3 -c 'import json; print(json.load(open("ptz-test-runtime/session.json"))["stream_id"])')"

curl -sS -X DELETE \
  -H "X-Api-Key: $PTZ_TEST_API_KEY" \
  "http://127.0.0.1:18081/v1/streams/$STREAM_ID"

docker compose \
  --env-file .env.ptz-test \
  -f docker-compose.ptz-test.yml \
  down
```

`down` 只删除隔离测试容器和网络，不删除测试报告、日志和截图，也不会修改正式容器。

## 常见问题

- 测试 API 起不来：检查 NVIDIA 驱动、`PTZ_TEST_ENGINE_DIR` 和基础镜像 tag。
- 虚拟 RTSP 没画面：检查 `ptz-test-camera-control` 日志，以及视频是否位于 `ptz-test-runtime/fixtures/boat.mp4`。
- 没触发 PTZ：先确认回放片段能被当前模型识别；需要测试疑似目标时，重新注册并加 `--small-target-proposals`。
- 真实摄像头不动：先单独验证 `camera-control`，再检查它是否加入 `rtsp-ptz-isolated-test` 网络。
