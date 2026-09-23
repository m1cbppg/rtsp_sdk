#!/usr/bin/env python3
"""Step 2A: YOLO26s loader sanity + tiny overfit.

Subcommands
-----------
preflight      data/weight preflight: counts, hashes, conflicts, environment
stage          freeze the deterministic tiny subset and stage the YOLO dataset view
loader-sanity  run the *real* Ultralytics dataset/dataloader over positive + negative
baseline       pretrained yolo26s inference on the frozen tiny subset (before training)
train          tiny overfit (requires an explicitly selected CUDA device by default)
post           post-train inference + tiny-overfit metrics + verdict
sheet          contact sheets (GT / baseline / post) for visual sanity
report         write SUMMARY.json + MANIFEST.json from the recorded pieces

The frozen datasets are never modified: staging copies or hardlinks bytes and re-verifies
SHA-256.  No Development / Sealed asset is ever opened.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rtsp_annotator.ground_litter_episode_candidates import git_commit_of  # noqa: E402
from rtsp_annotator.ground_litter_localization_review import (  # noqa: E402
    SealedAssetError,
    sha256_file,
)
from rtsp_annotator.ground_litter_step2a import (  # noqa: E402
    CONF_LEVELS,
    DATASET_LOADER_SANITY,
    GENERATOR_VERSION,
    IOU_NORMAL,
    IOU_SMALL,
    SCHEMA_VERSION,
    SEED,
    SMALL_AREA_RATIO_MAX,
    SMALL_AREA_RATIO_MIN,
    SMALL_EXPAND,
    SMALL_GT_SHORT_SIDE_PX,
    TINY_DATASET,
    TINY_POSITIVE_TARGET,
    Step2AError,
    build_manifest,
    build_summary,
    evaluate_predictions,
    load_pools,
    overfit_verdict,
    select_subsets,
    stage_dataset,
    subset_manifest,
    verify_preflight,
    verify_staging,
    write_reports,
)

POSITIVE_ROOT = ROOT / "output" / "ground_litter_positive_tile_completion_20260923"
NEGATIVE_ROOT = ROOT / "output" / "ground_litter_hard_negatives_20260923"
WEIGHT = ROOT / "models" / "yolo26s.pt"
DEFAULT_OUTPUT = ROOT / "output" / "ground_litter_step2a_20260923"

STEP1C2M_MANIFEST = "8457ad4"
STEP1D_EXECUTION = "4d3e05c8153a41319c3102f1491ba1204582c935"
STEP1D_EVIDENCE = "0983596"

#: Step 2A names.  Step 2B rebinds these two module globals to its own manifest and
#: dataset directory names and then reuses loader-sanity / baseline / predict unchanged,
#: so the two steps never share a filename and the Step 2A behaviour is the default.
MANIFEST_NAME = "tiny_overfit_manifest.json"
DATASET_DIR_NAME = TINY_DATASET


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--positive-root", type=Path, default=POSITIVE_ROOT)
    p.add_argument("--negative-root", type=Path, default=NEGATIVE_ROOT)
    p.add_argument("--weight", type=Path, default=WEIGHT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--repo-root", type=Path, default=ROOT)
    p.add_argument("--generated-at", default=None)
    p.add_argument("--device", default=None,
                   help="explicit device for train/post (cuda / mps / cpu)")
    p.add_argument("--allow-non-cuda", action="store_true",
                   help="explicitly allow a non-CUDA device for the tiny overfit")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("preflight", "stage", "loader-sanity", "baseline", "train", "post",
                 "sheet", "report"):
        sub.add_parser(name)
    return p


def _stamp(args: argparse.Namespace) -> str:
    return args.generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _dataset_dir(args: argparse.Namespace) -> Path:
    return args.output / DATASET_DIR_NAME


def _write_json(args: argparse.Namespace, name: str, payload) -> Path:
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return path


def _pools(args: argparse.Namespace) -> dict:
    return load_pools(args.positive_root, args.negative_root)


def _environment() -> dict:
    info: dict = {"platform": platform.platform(), "machine": platform.machine()}
    info["python"] = sys.version.split()[0]
    info["executable"] = sys.executable
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_cuda_version"] = getattr(torch.version, "cuda", None)
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device_count"] = int(torch.cuda.device_count())
        info["cuda_device_name"] = (torch.cuda.get_device_name(0)
                                    if torch.cuda.is_available() else None)
        info["cuda_vram_bytes"] = (torch.cuda.get_device_properties(0).total_memory
                                   if torch.cuda.is_available() else None)
        info["mps_built"] = bool(torch.backends.mps.is_built())
        info["mps_available"] = bool(torch.backends.mps.is_available())
    except Exception as exc:                                 # pragma: no cover
        info["torch_error"] = f"{type(exc).__name__}: {exc}"
    info["pythonpath"] = os.environ.get("PYTHONPATH") or ""
    try:
        import cv2

        info["opencv"] = cv2.__version__
        info["opencv_path"] = str(cv2.__file__)
    except Exception as exc:
        info["opencv_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import ultralytics

        info["ultralytics"] = ultralytics.__version__
        info["ultralytics_path"] = str(ultralytics.__file__)
    except Exception as exc:
        info["ultralytics_error"] = f"{type(exc).__name__}: {exc}"
    info["nvidia_smi_present"] = shutil.which("nvidia-smi") is not None
    if info["nvidia_smi_present"]:
        try:
            info["nvidia_smi"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader"], capture_output=True, text=True,
                timeout=20).stdout.strip()
        except Exception as exc:                             # pragma: no cover
            info["nvidia_smi_error"] = str(exc)
    blockers = []
    if not info.get("cuda_available"):
        blockers.append(
            "no CUDA GPU on this machine: torch.cuda.is_available() is False, "
            "nvidia-smi absent; the tiny overfit must not silently run on CPU")
    if info.get("ultralytics_error"):
        blockers.append(f"ultralytics cannot be imported here: {info['ultralytics_error']}")
    info["blockers"] = blockers
    return info


def _required_device(args: argparse.Namespace, environment: dict) -> str:
    """§3: never silently fall back to CPU for the formal tiny overfit."""
    if args.device is None:
        raise Step2AError(
            "no --device given. This machine reports cuda_available="
            f"{environment.get('cuda_available')}; pass --device cuda on a CUDA machine, "
            "or --device mps/cpu together with --allow-non-cuda to run the sanity tiny "
            "overfit on a non-target device on purpose")
    if args.device.startswith("cuda") and not environment.get("cuda_available"):
        raise Step2AError(f"--device {args.device} requested but no CUDA GPU is available")
    if not args.device.startswith("cuda") and not args.allow_non_cuda:
        raise Step2AError(
            f"--device {args.device} is not CUDA; pass --allow-non-cuda to confirm this "
            "is only a training-pipeline sanity run, not the target GPU validation")
    return args.device


def _tiny_manifest(args: argparse.Namespace) -> dict:
    """Always reuse the frozen subset when it is already recorded (§10).

    The frozen manifest is checked **before** the pools so that a machine holding only
    the uploaded tiny dataset (the CUDA training host) never needs the full frozen
    pools, and so that no command can re-select a different subset after a baseline
    or training result is known.
    """
    manifest_path = args.output / MANIFEST_NAME
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    pools = _pools(args)
    return subset_manifest(pools, select_subsets(pools))


def _manifest_provenance(args: argparse.Namespace) -> dict:
    path = args.output / MANIFEST_NAME
    return {
        "path": str(path),
        "reused_frozen_manifest": path.is_file(),
        "source": ("frozen_file" if path.is_file() else "reselect_from_pools"),
        "sha256": sha256_file(path) if path.is_file() else None,
    }


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #


def cmd_preflight(args: argparse.Namespace) -> int:
    pools = _pools(args)
    preflight = verify_preflight(pools, args.weight)
    environment = _environment()
    _write_json(args, "preflight.json", preflight)
    _write_json(args, "environment.json", environment)
    print(json.dumps({"preflight": preflight, "environment": environment},
                     ensure_ascii=False, indent=2))
    if not preflight["counts_match"]:
        raise Step2AError(f"dataset counts do not match: {preflight['counts']} != "
                          f"{preflight['expected']}")
    if preflight["problem_count"] or preflight["image_sha_conflict_count"] or \
            preflight["source_crop_conflict_count"] or not preflight["negative_label_all_empty"]:
        raise Step2AError(f"preflight problems: {preflight['problems']}")
    return 0


def cmd_stage(args: argparse.Namespace) -> int:
    pools = _pools(args)
    manifest = subset_manifest(pools, select_subsets(pools))
    dataset = _dataset_dir(args)
    if dataset.exists():
        shutil.rmtree(dataset)
    report = stage_dataset(manifest, dataset, mode="hardlink")
    integrity = verify_staging(report)
    _write_json(args, MANIFEST_NAME, manifest)
    _write_json(args, "staging_integrity.json", {"staging": report,
                                                "integrity": integrity})
    check = {
        "positive_images": len(manifest["positive"]),
        "negative_images": len(manifest["negative"]),
        "positive_boxes": sum(entry["box_count"] for entry in manifest["positive"]),
        "data_yaml": report["data_yaml"],
        "integrity": integrity,
        "coverage": manifest["coverage"],
    }
    print(json.dumps(check, ensure_ascii=False, indent=2))
    if not integrity["ok"]:
        raise Step2AError(f"staging changed bytes: {integrity}")
    return 0


def _build_loader(args: argparse.Namespace, manifest: dict):
    import yaml
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset, build_dataloader

    dataset = _dataset_dir(args)
    data = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))
    cfg = get_cfg()
    cfg.task = "detect"
    # loader sanity reads labels, not augmentations: keep the view deterministic
    cfg.augment = False
    cfg.mosaic = 0.0
    cfg.rect = False
    ds = build_yolo_dataset(cfg, str(dataset / "images" / "train"), batch=args.batch,
                            data=data, mode="train", rect=False, stride=32)
    loader = build_dataloader(ds, batch=min(args.batch, 4), workers=0, shuffle=False,
                              rank=-1)
    return ds, loader


def cmd_loader_sanity(args: argparse.Namespace) -> int:
    """§5/§12: the real Ultralytics dataset + dataloader, positive/negative/mixed."""
    import torch
    from torch.utils.data import DataLoader, Subset

    manifest = _tiny_manifest(args)
    ds, loader = _build_loader(args, manifest)
    kinds = {**{entry["tile_id"]: "positive" for entry in manifest["positive"]},
             **{entry["tile_id"]: "negative" for entry in manifest["negative"]}}
    per_sample = []
    for index in range(len(ds)):
        parsed = ds.labels[index]
        item = ds[index]
        tile_id = Path(item["im_file"]).stem
        raw_boxes = int(parsed["cls"].shape[0])
        per_sample.append({
            "tile_id": tile_id, "kind": kinds.get(tile_id, "unknown"),
            "raw_label_boxes": raw_boxes,
            "raw_label_bytes": Path(parsed["im_file"]).with_suffix("").name,
            "raw_cls_shape": list(parsed["cls"].shape),
            "raw_bboxes_shape": list(parsed["bboxes"].shape),
            "bbox_format": parsed.get("bbox_format"),
            "normalized": parsed.get("normalized"),
            "sample_img_shape": list(item["img"].shape),
            "sample_img_dtype": str(item["img"].dtype),
            "sample_boxes": int(item["cls"].shape[0]),
        })
        per_sample[-1]["label_bytes"] = (
            Path(str(Path(item["im_file"]).with_suffix(""))
                 .replace("images", "labels")).with_suffix(".txt").stat().st_size
            if Path(str(Path(item["im_file"]).with_suffix(""))
                    .replace("images", "labels")).with_suffix(".txt").is_file() else None)
    positive = [row for row in per_sample if row["kind"] == "positive"]
    negative = [row for row in per_sample if row["kind"] == "negative"]
    unknown = [row for row in per_sample if row["kind"] == "unknown"]

    batches = []
    for index in range(6):          # the whole tiny set in batches of 4
        batch = next(iter(loader))
        batches.append({
            "index": index, "image_shape": list(batch["img"].shape),
            "dtype": str(batch["img"].dtype), "batch_size": len(batch["im_file"]),
            "total_boxes": int(batch["cls"].shape[0]),
            "boxes_per_image": [int((batch["batch_idx"] == j).sum())
                                for j in range(len(batch["im_file"]))],
            "class_ids": sorted({int(value) for value in batch["cls"].flatten().tolist()}),
            "files": [Path(name).stem for name in batch["im_file"]],
        })

    def subset_batch(indices, batch_size):
        if not indices:
            return None
        subset = DataLoader(Subset(ds, indices), batch_size=batch_size, shuffle=False,
                            collate_fn=ds.collate_fn)
        batch = next(iter(subset))
        return {"batch_size": len(batch["im_file"]),
                "image_shape": list(batch["img"].shape),
                "total_boxes": int(batch["cls"].shape[0]),
                "boxes_per_image": [int((batch["batch_idx"] == j).sum())
                                    for j in range(len(batch["im_file"]))],
                "files": [Path(name).stem for name in batch["im_file"]]}

    positive_indices = [index for index, row in enumerate(per_sample)
                        if row["kind"] == "positive"]
    negative_indices = [index for index, row in enumerate(per_sample)
                        if row["kind"] == "negative"]
    report = {
        "schema_version": SCHEMA_VERSION,
        "stage": DATASET_LOADER_SANITY,
        "manifest_provenance": _manifest_provenance(args),
        "ultralytics_dataset_class": type(ds).__name__,
        "dataset_length": len(ds),
        "positive_samples": len(positive),
        "negative_samples": len(negative),
        "unknown_samples": len(unknown),
        "positive_with_zero_boxes": [row["tile_id"] for row in positive
                                     if row["raw_label_boxes"] == 0],
        "negative_with_boxes": [row["tile_id"] for row in negative
                                if row["raw_label_boxes"] != 0],
        "positive_min_boxes": min((row["raw_label_boxes"] for row in positive), default=0),
        "positive_box_total": sum(row["raw_label_boxes"] for row in positive),
        "negative_box_total": sum(row["raw_label_boxes"] for row in negative),
        "negative_label_bytes_zero": all(row["label_bytes"] == 0 for row in negative),
        "mixed_batches": batches,
        "positive_only_batch": subset_batch(positive_indices, 4),
        "negative_only_batch": subset_batch(negative_indices, 4),
        "per_sample": per_sample,
        "ok": (not unknown
               and len(positive) == len(manifest["positive"])
               and len(negative) == len(manifest["negative"])
               and all(row["raw_label_boxes"] >= 1 for row in positive)
               and all(row["raw_label_boxes"] == 0 for row in negative)
               and all(row["label_bytes"] == 0 for row in negative)
               and sum(row["raw_label_boxes"] for row in positive)
               == sum(entry["box_count"] for entry in manifest["positive"])
               and len(batches) >= 2
               and any(batch["total_boxes"] > 0 for batch in batches)),
    }
    _write_json(args, "loader_sanity.json", report)
    print(json.dumps({key: report[key] for key in (
        "dataset_length", "positive_samples", "negative_samples", "positive_min_boxes",
        "positive_box_total", "negative_box_total", "negative_label_bytes_zero", "ok")},
        ensure_ascii=False, indent=2))
    print("mixed batches:", json.dumps(
        [{key: batch[key] for key in ("image_shape", "batch_size", "total_boxes",
                                      "boxes_per_image", "class_ids")}
         for batch in batches]))
    if not report["ok"]:
        raise Step2AError("real Ultralytics loader sanity failed; see loader_sanity.json")
    return 0


def _predict(args: argparse.Namespace, weights: Path, tag: str, device: str) -> dict:
    from ultralytics import YOLO

    manifest = _tiny_manifest(args)
    model = YOLO(str(weights))
    tiles = [entry for entry in manifest["positive"]] + \
        [entry for entry in manifest["negative"]]
    images = [str(_dataset_dir(args) / "images" / "train" / f"{entry['tile_id']}.png")
              for entry in tiles]
    results = model.predict(images, imgsz=args.imgsz, conf=min(CONF_LEVELS), iou=0.7,
                            max_det=300, device=device, verbose=False)
    predictions: dict[str, list[dict]] = {}
    for entry, result in zip(tiles, results):
        rows = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes):
            for bbox, score, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(),
                                        boxes.cls.tolist()):
                rows.append({"bbox": [round(float(v), 3) for v in bbox],
                             "score": round(float(score), 6), "class_id": int(cls)})
        predictions[entry["tile_id"]] = rows
    metrics = evaluate_predictions(manifest, predictions)
    payload = {"tag": tag, "weights": str(weights), "device": device,
               "conf_floor": min(CONF_LEVELS), "iou_nms": 0.7, "imgsz": args.imgsz,
               "manifest_provenance": _manifest_provenance(args),
               "predictions": predictions, "metrics": metrics}
    _write_json(args, f"{tag}_metrics.json", payload)
    out_dir = args.output / f"{tag}_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return payload


def cmd_baseline(args: argparse.Namespace) -> int:
    payload = _predict(args, args.weight, "baseline", args.device or "cpu")
    print(json.dumps({conf: {key: payload["metrics"]["per_conf"][conf][key]
                             for key in ("positive_gt_proposal_recall",
                                         "positive_image_hit_rate",
                                         "negative_fp_images",
                                         "negative_total_predictions")}
                      for conf in payload["metrics"]["per_conf"]},
                     ensure_ascii=False, indent=2))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from ultralytics import YOLO

    environment = _environment()
    device = _required_device(args, environment)
    manifest = _tiny_manifest(args)
    runs = args.output / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weight))
    train_args = {
        "data": str(_dataset_dir(args) / "data.yaml"),
        "imgsz": args.imgsz, "epochs": args.epochs, "batch": args.batch,
        "device": device, "seed": SEED, "deterministic": True,
        "project": str(runs), "name": "tiny_overfit", "exist_ok": True,
        "pretrained": True, "val": True, "plots": False,
        # §15: deliberately minimal augmentation for a memory check
        "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0, "close_mosaic": 0,
        "degrees": 0.0, "translate": 0.0, "scale": 0.0, "shear": 0.0,
        "perspective": 0.0, "flipud": 0.0, "fliplr": 0.0,
        "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0, "erasing": 0.0,
        "optimizer": "auto", "verbose": True,
    }
    # ``YOLO.train`` is declared as ``(self, trainer=None, **kwargs)`` in every
    # Ultralytics release we checked (8.4.107 / 8.4.118 / 8.4.150), so the
    # trainer's own signature cannot tell us which hyperparameters are accepted.
    # The authoritative list of accepted arguments is the Ultralytics default cfg;
    # filtering on ``model.train``'s signature would silently drop every frozen
    # parameter and turn the tiny overfit into an unfrozen default run.
    supported: set[str] = set()
    try:
        from ultralytics.cfg import get_cfg

        supported |= set(vars(get_cfg()).keys())
    except Exception:                                        # pragma: no cover
        supported |= set(train_args)
    try:
        import inspect

        supported |= {name for name in inspect.signature(model.train).parameters
                      if name not in ("self", "trainer", "kwargs")}
    except (TypeError, ValueError):                          # pragma: no cover
        pass
    unsupported = sorted(key for key in train_args if key not in supported)
    applied = {key: value for key, value in train_args.items() if key in supported}
    missing_required = sorted(key for key in (
        "data", "imgsz", "epochs", "batch", "device", "seed", "deterministic",
        "mosaic", "mixup", "copy_paste", "optimizer", "project", "name")
        if key not in applied)
    if missing_required:
        raise Step2AError(f"this ultralytics build would silently drop frozen "
                          f"training parameters: {missing_required}")
    results = model.train(**applied)
    metrics_csv = Path(getattr(results, "save_dir", runs / "tiny_overfit")) / "results.csv"
    epochs_rows = []
    if metrics_csv.is_file():
        lines = metrics_csv.read_text(encoding="utf-8").strip().splitlines()
        header = [value.strip() for value in lines[0].split(",")] if lines else []
        for line in lines[1:]:
            values = [value.strip() for value in line.split(",")]
            epochs_rows.append(dict(zip(header, values)))
    losses = [float(row.get("train/box_loss") or "nan") for row in epochs_rows
              if row.get("train/box_loss")]
    nan = any(value != value or value in (float("inf"), float("-inf")) for value in losses)
    payload = {
        "device": device, "imgsz": args.imgsz, "epochs_requested": args.epochs,
        "epochs_run": len(epochs_rows), "batch": args.batch, "seed": SEED,
        "deterministic": True, "requested_args": train_args, "applied_args": applied,
        "unsupported_args_skipped": unsupported,
        "manifest_provenance": _manifest_provenance(args),
        "epoch_metrics": epochs_rows,
        "initial_box_loss": losses[0] if losses else None,
        "final_box_loss": losses[-1] if losses else None,
        "loss_decreased": bool(losses and losses[-1] < losses[0] * 0.8),
        "nan_or_inf": nan,
        "save_dir": str(getattr(results, "save_dir", runs / "tiny_overfit")),
        "note": ("OVERFIT_SANITY_ONLY: train and val are the same tiny subset"),
    }
    _write_json(args, "training.json", payload)
    print(json.dumps({key: payload[key] for key in (
        "epochs_run", "batch", "imgsz", "device", "initial_box_loss",
        "final_box_loss", "loss_decreased", "nan_or_inf")}, ensure_ascii=False, indent=2))
    if nan:
        raise Step2AError("NaN/Inf loss detected: OVERFIT_FAIL (§31)")
    return 0


def cmd_post(args: argparse.Namespace) -> int:
    training = json.loads((args.output / "training.json").read_text(encoding="utf-8"))
    best = Path(training["save_dir"]) / "weights" / "best.pt"
    if not best.is_file():
        raise Step2AError(f"trained checkpoint not found: {best}")
    payload = _predict(args, best, "post", training["device"])
    baseline_path = args.output / "baseline_metrics.json"
    baseline = (json.loads(baseline_path.read_text(encoding="utf-8"))
                if baseline_path.is_file() else None)
    verdict = overfit_verdict(payload["metrics"], loader_ok=True, training_ok=True,
                              nan_free=not training["nan_or_inf"],
                              loss_decreased=bool(training["loss_decreased"]))
    payload["verdict"] = verdict
    if baseline:
        payload["baseline_comparison"] = {
            conf: {"recall_before":
                   baseline["metrics"]["per_conf"][conf]["positive_gt_proposal_recall"],
                   "recall_after":
                   payload["metrics"]["per_conf"][conf]["positive_gt_proposal_recall"]}
            for conf in payload["metrics"]["per_conf"]}
    _write_json(args, "post_metrics.json", payload)
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    return 0


def cmd_sheet(args: argparse.Namespace) -> int:
    """Diagnostic contact sheets only; never a training input."""
    import numpy as np
    from rtsp_annotator.ground_litter_positive_tiles import png_bytes

    manifest = _tiny_manifest(args)
    dataset = _dataset_dir(args)

    def load(path: Path):
        from PIL import Image

        return np.asarray(Image.open(path).convert("RGB"))

    def draw(image, boxes, colour, thickness=3):
        canvas = image.copy()
        for box in boxes:
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(canvas.shape[1] - 1, x2), min(canvas.shape[0] - 1, y2)
            for offset in range(thickness):
                canvas[max(0, y1 + offset), x1:x2 + 1] = colour
                canvas[max(0, y2 - offset), x1:x2 + 1] = colour
                canvas[y1:y2 + 1, max(0, x1 + offset)] = colour
                canvas[y1:y2 + 1, max(0, x2 - offset)] = colour
        return canvas

    def sheet(canvases, out_path: Path, cols=4):
        rows = max(1, (len(canvases) + cols - 1) // cols)
        canvas = np.full((rows * 660, cols * 660, 3), 24, dtype="uint8")
        for index, tile in enumerate(canvases):
            row, col = divmod(index, cols)
            y, x = row * 660 + 10, col * 660 + 10
            canvas[y:y + 640, x:x + 640] = tile[:640, :640]
        out_path.write_bytes(png_bytes(np.ascontiguousarray(canvas)))

    baseline = None
    base_path = args.output / "baseline_predictions" / "predictions.json"
    if base_path.is_file():
        baseline = json.loads(base_path.read_text(encoding="utf-8"))
    post = None
    post_path = args.output / "post_predictions" / "predictions.json"
    if post_path.is_file():
        post = json.loads(post_path.read_text(encoding="utf-8"))

    gt_canvases = []
    for entry in manifest["positive"]:
        image = load(dataset / "images" / "train" / f"{entry['tile_id']}.png")
        canvases = [draw(image, entry["boxes"], (0, 255, 0))]
        label = entry["tile_id"]
        if baseline:
            canvases.append(draw(image, [row["bbox"] for row in
                                         baseline.get(label, [])
                                         if row["score"] >= 0.01], (60, 130, 255)))
        if post:
            canvases.append(draw(image, [row["bbox"] for row in post.get(label, [])
                                         if row["score"] >= 0.01], (224, 96, 58)))
        gt_canvases.extend(canvases)
    sheet(gt_canvases, args.output / "training_batch_contact_sheet.png", cols=4)

    negative_canvases = []
    for entry in manifest["negative"]:
        image = load(dataset / "images" / "train" / f"{entry['tile_id']}.png")
        canvases = [image]
        if baseline:
            canvases.append(draw(image, [row["bbox"] for row in
                                         baseline.get(entry["tile_id"], [])
                                         if row["score"] >= 0.01], (60, 130, 255)))
        if post:
            canvases.append(draw(image, [row["bbox"] for row in
                                         post.get(entry["tile_id"], [])
                                         if row["score"] >= 0.01], (224, 96, 58)))
        negative_canvases.extend(canvases)
    sheet(negative_canvases, args.output / "negative_contact_sheet.png", cols=3)
    print("sheets written to", args.output)
    return 0


def _provenance() -> dict:
    return {
        "step1c2m_evidence_commit": STEP1C2M_MANIFEST,
        "step1d_reported_execution_commit": STEP1D_EXECUTION,
        "step1d_evidence_commit": STEP1D_EVIDENCE,
        "note": ("Step 2A reads only the frozen training-side pools recorded here; "
                 "Development and Sealed assets are never opened"),
    }


def cmd_report(args: argparse.Namespace) -> int:
    pools = _pools(args)
    preflight = json.loads((args.output / "preflight.json").read_text(encoding="utf-8"))
    environment = json.loads((args.output / "environment.json").read_text(encoding="utf-8"))
    manifest = _tiny_manifest(args)
    staging = json.loads((args.output / "staging_integrity.json").read_text(encoding="utf-8"))
    def optional(name):
        path = args.output / name
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    loader = optional("loader_sanity.json")
    baseline = optional("baseline_metrics.json")
    post = optional("post_metrics.json")
    training = optional("training.json")
    verdict = (post or {}).get("verdict") or {"verdict": "NOT_EVALUATED",
                                              "reason": "tiny overfit not run"}
    # CUDA-host evidence: upload integrity, frozen-input verification, checkpoints and
    # the manifest the trainer actually reused.  Absent for a local-only run.
    server_run = {
        "upload_sha256sums": optional("upload_sha256sums.json"),
        "evidence_archive": optional("evidence_archive.json"),
        "frozen_input_verification": optional("frozen_input_verification.json"),
        "data_yaml_record": optional("data_yaml_record.json"),
        "checkpoint_hashes": optional("checkpoint_hashes.json"),
        "training_manifest_provenance": (training or {}).get("manifest_provenance"),
    }
    summary = build_summary(pools, preflight, environment, manifest, staging, loader,
                            baseline, post, training, verdict, server_run=server_run)
    provenance = _provenance()
    provenance["server_run"] = server_run
    manifest_json = build_manifest(
        pools, summary, code_commit=git_commit_of(args.repo_root),
        generated_at=_stamp(args),
        config={"positive_root": str(args.positive_root),
                "negative_root": str(args.negative_root),
                "weight": str(args.weight), "output": str(args.output),
                "device": args.device, "batch": args.batch, "epochs": args.epochs,
                "imgsz": args.imgsz, "seed": SEED},
        artifact_root=args.output, provenance=provenance)
    paths = write_reports(args.output, tiny_manifest=manifest, loader=loader,
                          summary=summary, manifest=manifest_json)
    print(json.dumps({"verdict": verdict, "artifacts": paths}, ensure_ascii=False,
                     indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    commands = {
        "preflight": cmd_preflight, "stage": cmd_stage,
        "loader-sanity": cmd_loader_sanity, "baseline": cmd_baseline,
        "train": cmd_train, "post": cmd_post, "sheet": cmd_sheet, "report": cmd_report,
    }
    try:
        return commands[args.command](args)
    except (SealedAssetError, Step2AError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
