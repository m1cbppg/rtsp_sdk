#!/usr/bin/env python3
"""Serve audit cards and persist one-click human decisions atomically."""
from __future__ import annotations
import argparse, json, os, tempfile
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ALLOWED={"LITTER","NON_LITTER","BOX_WRONG","UNCERTAIN"}

class Handler(SimpleHTTPRequestHandler):
    root: Path
    def translate_path(self,path):
        clean=urlparse(path).path.lstrip("/") or "index.html"
        target=(self.root/clean).resolve()
        if target != self.root and self.root not in target.parents:
            return str(self.root/"__forbidden__")
        return str(target)
    def _json(self,status,payload):
        data=json.dumps(payload,ensure_ascii=False).encode(); self.send_response(status); self.send_header("Content-Type","application/json; charset=utf-8"); self.send_header("Content-Length",str(len(data))); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        path=urlparse(self.path).path
        if path=="/api/reviews":
            target=self.root/"reviews.json"; return self._json(200,json.loads(target.read_text()) if target.exists() else {"reviews":{}})
        if path=="/api/data": return self._json(200,json.loads((self.root/"review-data.json").read_text()))
        return super().do_GET()
    def do_POST(self):
        if urlparse(self.path).path!="/api/reviews": return self._json(404,{"error":"not found"})
        try:
            size=int(self.headers.get("Content-Length","0")); item=json.loads(self.rfile.read(size)); rid=str(item["review_id"]); label=str(item["label"])
            dataset=json.loads((self.root/"review-data.json").read_text()); valid={row["review_id"] for row in dataset["items"]}
            if rid not in valid or label not in ALLOWED: raise ValueError("invalid review")
            target=self.root/"reviews.json"; payload=json.loads(target.read_text()) if target.exists() else {"dataset":dataset["dataset"],"fingerprint":dataset["fingerprint"],"reviews":{}}
            payload["reviews"][rid]=item
            fd,tmp=tempfile.mkstemp(dir=self.root,prefix="reviews-",suffix=".tmp")
            with os.fdopen(fd,"w") as stream: json.dump(payload,stream,ensure_ascii=False,indent=2); stream.flush(); os.fsync(stream.fileno())
            os.replace(tmp,target); return self._json(200,{"ok":True,"count":len(payload["reviews"])})
        except Exception as exc: return self._json(400,{"error":str(exc)[:120]})

def main():
    p=argparse.ArgumentParser(); p.add_argument("--dataset",type=Path,required=True); p.add_argument("--port",type=int,default=8767); a=p.parse_args(); Handler.root=a.dataset.resolve()
    server=ThreadingHTTPServer(("127.0.0.1",a.port),Handler); print(f"Local URL: http://127.0.0.1:{a.port}/",flush=True); server.serve_forever()
if __name__=="__main__": raise SystemExit(main())
