# 通过机房公网端口播放识别流

## 1. 网络关系

Ubuntu没有公网IP并不影响使用，但机房网关必须增加独立的TCP端口映射。
假设：

```text
机房公网IP              14.21.88.97
公网RTSP端口            28554
Ubuntu内网IP            192.168.10.50
Docker发布到宿主机端口  8554
```

需要机房配置：

```text
14.21.88.97:28554/TCP → 192.168.10.50:8554/TCP
```

`21002`已经映射给SSH，不能同时用于RTSP。公网端口`28554`只是示例，应以
机房实际分配为准。项目强制RTSP over TCP，因此只需要这一条TCP映射。

查看Ubuntu内网地址：

```bash
hostname -I
ip route
```

同时必须保证Ubuntu能够访问原摄像头RTSP地址。摄像头在同一机房内网时，
`RTSP_INPUT_URL`直接使用摄像头内网IP即可。

## 2. 创建部署配置

在解压离线包的目录创建`.env.public`：

```bash
umask 077
nano .env.public
```

内容示例：

```dotenv
RTSP_INPUT_URL='rtsp://摄像头用户:摄像头密码@192.168.10.20:554/原流路径'
YOLO_MODEL_PATH='/app/models/yolo26s.pt'
YOLO_IMGSZ='640'
YOLO_CLASSES='0'

RTSP_HOST_PORT='8554'
RTSP_PUBLISH_USER='publisher'
RTSP_PUBLISH_PASSWORD='替换为随机发布密码'
RTSP_READ_USER='viewer'
RTSP_READ_PASSWORD='替换为随机观看密码'
```

用两个不同的随机十六进制密码，避免URL特殊字符转义问题：

```bash
openssl rand -hex 16
openssl rand -hex 16
```

`.env.public`使用单引号时，Docker Compose会按字面值读取`$`等字符。限制
配置文件权限：

```bash
chmod 600 .env.public
```

## 3. 启动

```bash
docker compose \
  --env-file .env.public \
  -f docker-compose.cuda.public.yml \
  up -d
```

检查：

```bash
docker compose \
  --env-file .env.public \
  -f docker-compose.cuda.public.yml \
  ps

docker compose \
  --env-file .env.public \
  -f docker-compose.cuda.public.yml \
  logs --tail=100 annotator
```

配置中：

- `publisher`账号只能向`detected`路径发布；
- `viewer`账号只能读取`detected`路径；
- 匿名用户不能发布或读取；
- RTSP媒体传输固定为TCP；
- Docker不会联网拉取镜像。

## 4. 先在Ubuntu本机验收

读取密码以`.env.public`中的实际值为准：

```bash
ffplay \
  -rtsp_transport tcp \
  -fflags nobuffer \
  -flags low_delay \
  -framedrop \
  'rtsp://viewer:观看密码@127.0.0.1:8554/detected'
```

如果Ubuntu没有桌面，可以用容器内的`ffprobe`：

```bash
docker exec rtsp-yolo-annotator \
  ffprobe \
  -v error \
  -rtsp_transport tcp \
  -show_entries stream=codec_name,width,height,r_frame_rate \
  'rtsp://viewer:观看密码@mediamtx:8554/detected'
```

## 5. 从真正的外网验收

必须使用不在机房内网的设备测试，例如手机热点下的电脑：

```bash
ffplay \
  -rtsp_transport tcp \
  -fflags nobuffer \
  -flags low_delay \
  -framedrop \
  'rtsp://viewer:观看密码@14.21.88.97:28554/detected'
```

VLC：

```bash
vlc \
  --rtsp-tcp \
  --network-caching=100 \
  'rtsp://viewer:观看密码@14.21.88.97:28554/detected'
```

如果公网端口不通，先检查：

```bash
nc -vz 14.21.88.97 28554
```

端口不通且Ubuntu本机能播放，问题在机房NAT/ACL，而不是YOLO或Docker。

## 6. 安全边界

这份配置提供账号鉴权和权限隔离，但普通`rtsp://`本身没有TLS加密。公网部署
应优先让机房ACL只允许固定的观看端公网IP访问`28554/TCP`。如果观看端IP不
固定且监控内容敏感，应进一步使用RTSPS、VPN或SSH隧道。

不要把`.env.public`放入镜像、代码仓库或发送给无关人员。每个观看客户端会
占用一份输出带宽；默认`2500k`码率下，单客户端约需2.5 Mbps加协议开销。
