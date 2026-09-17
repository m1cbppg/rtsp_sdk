"""Prepare a local-only frozen baseline and unlabelled five-camera collection pack."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtsp_annotator.ground_litter_acceptance import make_truth_template, sha256_file


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare(profiles_path: Path, destination: Path, root: Path) -> dict:
    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
    cameras = profiles["cameras"]
    if len({c["device_code"] for c in cameras}) != len(cameras):
        raise ValueError("duplicate camera")
    if any(c.get("enabled") is not False for c in cameras):
        raise ValueError("acceptance baseline must have every camera disabled")
    if destination.exists():
        raise FileExistsError("use a new acceptance directory; existing evidence is preserved")
    files = {profiles_path.resolve(), (root / profiles["model"]["path"]).resolve(),
             (root / "models/yolo26s.pt").resolve()}
    for camera in cameras:
        files.add((root / camera["reference_image"]).resolve())
    for pattern in ("rtsp_annotator/ground_litter*.py", "rtsp_annotator/playback_source.py",
                    "scripts/*ground_litter*.py", "tests/test_ground_litter*.py",
                    "tests/test_playback_source.py", "models/litter/manifest.json"):
        files.update(p.resolve() for p in root.glob(pattern))
    manifest_files = []
    for path in sorted(files):
        relative = str(path.relative_to(root.resolve()))
        manifest_files.append({"path": relative, "sha256": sha256_file(path),
                               "bytes": path.stat().st_size})
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    # Only names/status, never diffs or real API config contents.
    status = subprocess.check_output(["git", "status", "--short"], cwd=root, text=True)
    manifest = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                "git_head": head, "worktree_dirty": bool(status.strip()),
                "purpose": "experimental_baseline_not_production_approval", "files": manifest_files,
                "notifications_enabled": False, "production_integration": False,
                "notes": ["文件散列绑定当前未提交代码；Git HEAD 不能独自重现本次基线。",
                          "只冻结实验资产；尚无现场效果、五路吞吐或生产共存验收。",
                          "本目录为验收准备材料，不是可运行的部署包。"]}
    destination.mkdir(parents=True, exist_ok=False)
    baseline = destination / "baseline"
    baseline.mkdir()
    # Byte-for-byte copy: no threshold/ROI/exclusion changes to the user's profile.
    (baseline / "profiles.json").write_bytes(profiles_path.read_bytes())
    write_json(baseline / "manifest.json", manifest)
    collection, links = [], []
    truth_dir = destination / "truth"
    truth_dir.mkdir()
    for camera in sorted(cameras, key=lambda c: c.get("trial_priority", 99)):
        for mode in ("day", "night"):
            code = camera["device_code"]
            name = f"{code}-{mode}.json"
            write_json(truth_dir / name, make_truth_template(camera, mode))
            collection.append({"camera_id": code, "view_id": camera["view_id"], "mode": mode,
                               "priority": camera.get("trial_priority"),
                               "regions": [z["region_id"] for z in camera["zones"]],
                               "route": "pending_playback_or_controlled_placement",
                               "media_path": None, "start_time": None, "end_time": None,
                               "camera_verified": False, "time_verified": False,
                               "operator": None, "truth_file": "truth/" + name})
            links.append(f'<tr><td>{html.escape(code)}</td><td>{mode}</td><td>待采集/复核</td>'
                         f'<td><a href="truth/{name}">真值模板</a></td></tr>')
    write_json(destination / "collection.json", {"schema_version": 1, "timezone": "Asia/Shanghai", "tasks": collection})
    write_json(destination / "placements.json", {"schema_version": 1, "status": "blank_template_not_truth",
        "columns": ["camera_id", "mode", "session_id", "episode_id", "category", "region_id",
                    "distance_band", "placement_time", "occlusion_start", "occlusion_end",
                    "removal_time", "clean_visible_start", "clean_visible_end", "operator", "evidence"],
        "records": []})
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>荷兴广场验收准备</title>
<style>body{font:16px system-ui;max-width:1050px;margin:40px auto;padding:0 20px;color:#192c39}
table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #ccc;text-align:left}
.notice{padding:20px;background:#fff4d6}a{color:#075ba1}</style>
<h1>荷兴广场 · 零散垃圾验收准备</h1>
<p class="notice">当前没有人工真值，准确率、召回率和误报率无法计算。所有机位保持未启用。优先采集 1030 昼夜样本。</p>
<p><a href="baseline/manifest.json">实验基线指纹</a> · <a href="baseline/profiles.json">配置快照</a> ·
<a href="collection.json">五路采集任务</a> · <a href="placements.json">现场摆放记录表</a></p>
<p>每段录像先核对机位、开头/中段/末尾画面时间，再穷尽标注真实物体、遮挡和清理区间。
模板中的 null/false/空数组表示未完成，不能用模型输出自动填写人工结论。</p>
<table><thead><tr><th>机位</th><th>时段</th><th>状态</th><th>标注</th></tr></thead><tbody>'''
    (destination / "index.html").write_text(page + "\n".join(links) + "</tbody></table></html>\n", encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, default=Path("config/ground_litter_profiles.example.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = prepare(args.profiles, args.output, Path(__file__).resolve().parents[1])
    print(json.dumps({"output": str(args.output), "fingerprinted_files": len(manifest["files"]),
                      "status": "待采集人工真值"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
