#!/usr/bin/env python3
"""RTSP viewer 网关：静态页 + HLS 同源反代 + 复核图片 API 反代。

- 静态：  GET /                      -> WEB_DIR 下的 rtsp_viewer.html
- HLS：   GET /detected/*            -> http://rtsp-mediamtx:8888/detected/*（注入 Basic 鉴权）
- 图片：  GET /v1/vessel-verifications[...]  -> http://api:8080/v1/vessel-verifications[...]（注入 X-API-Key）

观看凭据/API Key 从 SECRETS_FILE（默认 /srv/secrets.json，mode 600，不在 WEB_DIR 下）读取，
也可通过环境变量 VIEWER_USER/VIEWER_PASS/API_KEY 覆盖。启动（容器需加入网络）：
  docker run -d --name rtsp-web-gateway --restart unless-stopped \
    --network rtsp-yolo-deepstream-api_default \
    -v /home/sf01/rtsp-deepstream/web:/srv/web:ro \
    -v /home/sf01/rtsp-deepstream/secrets-viewer.json:/srv/secrets.json:ro \
    -p 8088:8088 --entrypoint python3 \
    rtsp-yolo-annotator:deepstream8-amd64 /srv/web/gateway.py
"""
import base64
import http.server
import json
import os
import socketserver
import sys
import urllib.error
import urllib.request

WEB_DIR = os.environ.get("WEB_DIR", "/srv/web")
MEDIAMTX_HLS = os.environ.get("MEDIAMTX_HLS", "http://rtsp-mediamtx:8888")
API_HOST = os.environ.get("API_HOST", "http://api:8080")
PORT = int(os.environ.get("GATEWAY_PORT", "8088"))
SECRETS_FILE = os.environ.get("SECRETS_FILE", "/srv/secrets.json")


def _load_secrets():
    """读取观看账号与 API Key（file > env）。file 不在 WEB_DIR 下，避免被静态服务暴露。"""
    secrets = {
        "user": os.environ.get("VIEWER_USER", ""),
        "password": os.environ.get("VIEWER_PASS", ""),
        "api_key": os.environ.get("API_KEY", ""),
    }
    if os.path.exists(SECRETS_FILE):
        try:
            with open(SECRETS_FILE, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            for key in ("user", "password", "api_key"):
                if data.get(key):
                    secrets[key] = data[key]
        except Exception:  # noqa: BLE001
            pass
    return secrets


def _basic_header(user, password):
    token = base64.b64encode(("%s:%s" % (user, password)).encode("utf-8")).decode("ascii")
    return "Basic " + token


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._serve_viewer()
        elif self.path.startswith("/detected/"):
            self._proxy_media()
        elif self.path.startswith("/v1/vessel-verifications"):
            self._proxy_api()
        else:
            super().do_GET()

    def _respond(self, status, data, ctype, extra_headers=None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if status not in (204, 304):
            self.wfile.write(data)

    def _serve_viewer(self):
        path = os.path.join(WEB_DIR, "rtsp_viewer.html")
        try:
            with open(path, "rb") as fp:
                data = fp.read()
            self._respond(200, data, "text/html; charset=utf-8")
        except OSError as exc:
            self._respond(404, ("viewer not found: %s" % exc).encode("utf-8"), "text/plain; charset=utf-8")

    def _proxy_media(self):
        target = MEDIAMTX_HLS + self.path  # 保留 path 与查询参数
        try:
            req = urllib.request.Request(target, headers={"User-Agent": "rtsp-viewer-gateway"})
            # MediaMTX 的 HLS 只认 HTTP Basic Auth，不认查询参数。用服务器端观看凭据注入
            # Basic 头（若客户端自带 Authorization 也转发），使浏览器 HLS 读取能通过鉴权。
            secrets = _load_secrets()
            if secrets["user"] and secrets["password"]:
                req.add_header("Authorization", _basic_header(secrets["user"], secrets["password"]))
            client_auth = self.headers.get("Authorization")
            if client_auth:
                req.add_header("Authorization", client_auth)
            try:
                up = urllib.request.urlopen(req, timeout=15)
            except urllib.error.HTTPError as herr:
                self._respond(herr.code, b"", "text/plain; charset=utf-8")
                return
            with up:
                self._respond(
                    up.status,
                    up.read(),
                    up.headers.get("Content-Type", "application/octet-stream"),
                )
        except Exception as exc:  # noqa: BLE001
            self._respond(502, ("gateway error: %s" % exc).encode("utf-8"), "text/plain; charset=utf-8")

    def _proxy_api(self):
        target = API_HOST + self.path
        try:
            req = urllib.request.Request(target, headers={"User-Agent": "rtsp-viewer-gateway"})
            api_key = _load_secrets()["api_key"]
            if api_key:
                req.add_header("X-API-Key", api_key)
            try:
                up = urllib.request.urlopen(req, timeout=20)
            except urllib.error.HTTPError as herr:
                self._respond(herr.code, b"", "text/plain; charset=utf-8")
                return
            with up:
                self._respond(
                    up.status,
                    up.read(),
                    up.headers.get("Content-Type", "application/octet-stream"),
                )
        except Exception as exc:  # noqa: BLE001
            self._respond(502, ("gateway error: %s" % exc).encode("utf-8"), "text/plain; charset=utf-8")

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


if __name__ == "__main__":
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("0.0.0.0", PORT), Handler) as httpd:
        httpd.daemon_threads = True
        sys.stderr.write("rtsp-web-gateway listening on %s\n" % PORT)
        httpd.serve_forever()
