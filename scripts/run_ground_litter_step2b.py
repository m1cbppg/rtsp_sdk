#!/usr/bin/env python3
"""Step 2B: full fine-tune on the frozen training pool + train-set sanity + freeze.

Subcommands
-----------
preflight      full-pool checks: 44 positive / 96 boxes + 63 negative / 0 boxes
stage          stage the whole frozen pool as the training set (bytes re-verified)
loader-sanity  run the real Ultralytics dataset/dataloader over all 107 tiles
baseline       pretrained yolo26s on the training set (before)
train          full fine-tune with the operator-frozen light-augmentation recipe
post           frozen-checkpoint inference on the training set + §16.3 sanity verdict
freeze         hash and read-only-freeze exactly one checkpoint for Step 2C
report         write SUMMARY.json + MANIFEST.json from the recorded pieces

The shared engine (environment probe, device gate, Ultralytics loader, predictor and
the conf-level evaluation) is imported from ``scripts/run_ground_litter_step2a.py``
with only ``MANIFEST_NAME`` / ``DATASET_DIR_NAME`` rebound, so the metric schema and the
frozen Step 0B matching are literally the same code as Step 2A.  Nothing in Step 2A is
modified or overwritten: Step 2B uses its own directory names and its own output root.

This step makes no generalisation claim: every metric is on the training tiles
(train == val).  Development evaluation is Step 2C and is not started here.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
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
    SEED,
    Step2AError,
    load_pools,
    stage_dataset,
    verify_preflight,
    verify_staging,
)
from rtsp_annotator.ground_litter_step2b import (  # noqa: E402
    DATA_YAML_NOTE,
    FULL_TRAIN_DATASET,
    FULL_TRAIN_MANIFEST,
    REQUIRED_ARGS,
    TRAIN_RECIPE,
    TRAIN_SET_RECALL_BAR,
    Step2BError,
    build_manifest,
    build_summary,
    checkpoint_freeze_record,
    full_finetune_verdict,
    full_pool_manifest,
    verify_full_pool,
    write_reports,
)
import rtsp_annotator.ground_litter_step2a as _STEP2A_MODULE  # noqa: E402

# The staging writer lives in the Step 2A module and stamps its own data.yaml note;
# rebind it so the Step 2B dataset declares FULL_TRAIN_SET_SANITY_ONLY instead.
_STEP2A_MODULE.DATA_YAML_NOTE = DATA_YAML_NOTE

POSITIVE_ROOT = ROOT / "output" / "ground_litter_positive_tile_completion_20260923"
NEGATIVE_ROOT = ROOT / "output" / "ground_litter_hard_negatives_20260923"
WEIGHT = ROOT / "models" / "yolo26s.pt"
DEFAULT_OUTPUT = ROOT / "output" / "ground_litter_step2b_20260923"

STEP1C2M_MANIFEST = "8457ad4"
STEP1D_EXECUTION = "4d3e05c8153a41319c3102f1491ba1204582c935"
STEP1D_EXCLUSION = "0983596"


def _load_step2a_cli():
    """Import the Step 2A CLI as a module and rebind only its two name knobs."""
    spec = importlib.util.spec_from_file_location(
        "run_ground_litter_step2a", ROOT / "scripts" / "run_ground_litter_step2a.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MANIFEST_NAME = FULL_TRAIN_MANIFEST
    module.DATASET_DIR_NAME = FULL_TRAIN_DATASET
    return module


STEP2A = _load_step2a_cli()
_write_json = STEP2A._write_json
_environment = STEP2A._environment
_required_device = STEP2A._required_device
_manifest_provenance = STEP2A._manifest_provenance
_predict = STEP2A._predict
cmd_loader_sanity = STEP2A.cmd_loader_sanity
cmd_baseline = STEP2A.cmd_baseline


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
                   help="explicitly allow a non-CUDA device for the fine-tune")
    p.add_argument("--batch", type=int, default=TRAIN_RECIPE["batch"])
    p.add_argument("--epochs", type=int, default=TRAIN_RECIPE["epochs"])
    p.add_argument("--imgsz", type=int, default=TRAIN_RECIPE["imgsz"])
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("preflight", "stage", "loader-sanity", "baseline", "train", "post",
                 "freeze", "report"):
        sub.add_parser(name)
    return p


def _stamp(args: argparse.Namespace) -> str:
    return args.generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _dataset_dir(args: argparse.Namespace) -> Path:
    return args.output / FULL_TRAIN_DATASET


def _manifest_path(args: argparse.Namespace) -> Path:
    return args.output / FULL_TRAIN_MANIFEST


def _pools(args: argparse.Namespace) -> dict:
    return load_pools(args.positive_root, args.negative_root)


def _train_manifest(args: argparse.Namespace) -> dict:
    """The full training set: always the frozen pool manifest once it is recorded."""
    path = _manifest_path(args)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return full_pool_manifest(_pools(args))


def _provenance() -> dict:
    return {
        "step1c2m_evidence_commit": STEP1C2M_MANIFEST,
        "step1d_reported_execution_commit": STEP1D_EXECUTION,
        "step1d_evidence_commit": STEP1D_EXCLUSION,
        "note": ("Step 2B reads only the frozen training-side pools recorded here and "
                 "copies bytes with SHA-256 re-verification; Development and Sealed assets "
                 "are never opened"),
    }


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #


def cmd_preflight(args: argparse.Namespace) -> int:
    pools = _pools(args)
    preflight = verify_preflight(pools, args.weight)
    manifest = full_pool_manifest(pools)
    full_pool = verify_full_pool(pools, manifest, preflight)
    environment = _environment()
    _write_json(args, "preflight.json", preflight)
    _write_json(args, "full_pool_check.json", full_pool)
    _write_json(args, "environment.json", environment)
    print(json.dumps({"counts": preflight["counts"],
                      "counts_match": preflight["counts_match"],
                      "full_pool_ok": full_pool["ok"],
                      "coverage": manifest["coverage"],
                      "problems": full_pool["problems"],
                      "blockers": environment.get("blockers")},
                     ensure_ascii=False, indent=2))
    if not preflight["counts_match"]:
        raise Step2BError(f"frozen pool counts do not match: {preflight['counts']} != "
                          f"{preflight['expected']}")
    if preflight["problem_count"] or preflight["image_sha_conflict_count"] or \
            preflight["source_crop_conflict_count"] or not preflight["negative_label_all_empty"]:
        raise Step2BError(f"preflight problems: {preflight['problems']}")
    if not full_pool["ok"]:
        raise Step2BError(f"full-pool check failed: {full_pool['problems']}")
    return 0


def cmd_stage(args: argparse.Namespace) -> int:
    pools = _pools(args)
    preflight = verify_preflight(pools, None)
    manifest = full_pool_manifest(pools)
    full_pool = verify_full_pool(pools, manifest, preflight)
    if not full_pool["ok"]:
        raise Step2BError(f"full-pool check failed: {full_pool['problems']}")
    report = stage_dataset(manifest, _dataset_dir(args), mode="hardlink")
    integrity = verify_staging(report)
    _write_json(args, FULL_TRAIN_MANIFEST, manifest)
    _write_json(args, "full_pool_check.json", full_pool)
    _write_json(args, "staging_integrity.json", {"staging": report,
                                                 "integrity": integrity})
    print(json.dumps({"positive_images": len(manifest["positive"]),
                      "negative_images": len(manifest["negative"]),
                      "positive_boxes": sum(entry["box_count"]
                                            for entry in manifest["positive"]),
                      "data_yaml": report["data_yaml"],
                      "integrity": integrity,
                      "coverage": manifest["coverage"]},
                     ensure_ascii=False, indent=2))
    if not integrity["ok"]:
        raise Step2BError(f"staging changed bytes: {integrity}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from ultralytics import YOLO

    environment = _environment()
    device = _required_device(args, environment)
    runs = args.output / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weight))
    train_args = {
        "data": str(_dataset_dir(args) / "data.yaml"),
        "device": device,
        "project": str(runs), "name": "full_finetune", "exist_ok": True,
        "verbose": True,
    }
    train_args.update(TRAIN_RECIPE)
    train_args["imgsz"] = args.imgsz
    train_args["epochs"] = args.epochs
    train_args["batch"] = args.batch
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
    missing_required = sorted(key for key in REQUIRED_ARGS if key not in applied)
    if missing_required:
        raise Step2BError(f"this ultralytics build would silently drop frozen training "
                          f"parameters: {missing_required}")
    results = model.train(**applied)
    save_dir = Path(getattr(results, "save_dir", runs / "full_finetune"))
    metrics_csv = save_dir / "results.csv"
    epochs_rows: list[dict[str, str]] = []
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
        "stage": "full_finetune", "device": device, "imgsz": args.imgsz,
        "epochs_requested": args.epochs, "epochs_run": len(epochs_rows),
        "batch": args.batch, "seed": SEED, "deterministic": True,
        "recipe": dict(TRAIN_RECIPE),
        "requested_args": train_args, "applied_args": applied,
        "unsupported_args_skipped": unsupported,
        "manifest_provenance": _manifest_provenance(args),
        "epoch_metrics": epochs_rows,
        "initial_box_loss": losses[0] if losses else None,
        "final_box_loss": losses[-1] if losses else None,
        "loss_decreased": bool(losses and losses[-1] < losses[0] * 0.8),
        "nan_or_inf": nan,
        "save_dir": str(save_dir),
        "note": "FULL_TRAIN_SET_SANITY_ONLY: train and val are the whole frozen pool",
    }
    _write_json(args, "training.json", payload)
    print(json.dumps({key: payload[key] for key in (
        "epochs_run", "batch", "imgsz", "device", "initial_box_loss",
        "final_box_loss", "loss_decreased", "nan_or_inf")}, ensure_ascii=False, indent=2))
    if nan:
        raise Step2BError("NaN/Inf loss detected: full fine-tune failed")
    return 0


def cmd_post(args: argparse.Namespace) -> int:
    training = json.loads((args.output / "training.json").read_text(encoding="utf-8"))
    best = Path(training["save_dir"]) / "weights" / "best.pt"
    if not best.is_file():
        raise Step2BError(f"trained checkpoint not found: {best}")
    loader_path = args.output / "loader_sanity.json"
    loader_ok = (bool(json.loads(loader_path.read_text(encoding="utf-8"))["ok"])
                 if loader_path.is_file() else True)
    full_pool_path = args.output / "full_pool_check.json"
    full_pool_ok = (bool(json.loads(full_pool_path.read_text(encoding="utf-8"))["ok"])
                    if full_pool_path.is_file() else True)
    payload = _predict(args, best, "post", training["device"])
    baseline_path = args.output / "baseline_metrics.json"
    baseline = (json.loads(baseline_path.read_text(encoding="utf-8"))
                if baseline_path.is_file() else None)
    verdict = full_finetune_verdict(training, payload, loader_ok=loader_ok,
                                    full_pool_ok=full_pool_ok)
    payload["verdict"] = verdict
    payload["stage"] = "full_finetune_post"
    if baseline:
        payload["baseline_comparison"] = {
            conf: {"recall_before":
                   baseline["metrics"]["per_conf"][conf]["positive_gt_proposal_recall"],
                   "recall_after":
                   payload["metrics"]["per_conf"][conf]["positive_gt_proposal_recall"],
                   "image_hit_before":
                   baseline["metrics"]["per_conf"][conf]["positive_image_hit_rate"],
                   "image_hit_after":
                   payload["metrics"]["per_conf"][conf]["positive_image_hit_rate"],
                   "negative_fp_images_before":
                   baseline["metrics"]["per_conf"][conf]["negative_fp_images"],
                   "negative_fp_images_after":
                   payload["metrics"]["per_conf"][conf]["negative_fp_images"]}
            for conf in payload["metrics"]["per_conf"]}
    _write_json(args, "post_metrics.json", payload)
    print(json.dumps(verdict, ensure_ascii=False, indent=2))
    return 0


def _chmod_read_only(weights_dir: Path) -> dict:
    """Freeze the checkpoint files against accidental overwrite."""
    frozen: dict[str, dict] = {}
    for path in sorted(weights_dir.glob("*.pt")):
        mode = path.stat().st_mode & 0o777
        try:
            os.chmod(path, mode & ~0o222)
            frozen[path.name] = {"mode_before": oct(mode),
                                 "mode_after": oct(path.stat().st_mode & 0o777)}
        except OSError as exc:                               # pragma: no cover
            frozen[path.name] = {"mode_before": oct(mode), "error": str(exc)}
    return frozen


def cmd_freeze(args: argparse.Namespace) -> int:
    training = json.loads((args.output / "training.json").read_text(encoding="utf-8"))
    run_dir = Path(training["save_dir"]) / "weights"
    checkpoints: dict[str, object] = {}
    for name in ("best.pt", "last.pt"):
        path = run_dir / name
        if not path.is_file():
            raise Step2BError(f"checkpoint missing: {path}")
        checkpoints[name] = {"path": str(path), "bytes": path.stat().st_size,
                             "sha256": sha256_file(path)}
    pretrained_sha = json.loads(
        (args.output / "preflight.json").read_text(encoding="utf-8"))["weight"]["sha256"]
    if checkpoints["best.pt"]["sha256"] == pretrained_sha:   # type: ignore[index]
        raise Step2BError("best.pt is byte-identical to the pretrained weight")
    checkpoints["best_equals_last"] = (checkpoints["best.pt"]["sha256"]  # type: ignore[index]
                                       == checkpoints["last.pt"]["sha256"])  # type: ignore[index]
    _write_json(args, "checkpoint_hashes.json", checkpoints)
    manifest_path = _manifest_path(args)
    record = checkpoint_freeze_record(
        training, checkpoints, primary="best.pt", frozen_at=_stamp(args),
        manifest_sha256=sha256_file(manifest_path) if manifest_path.is_file() else None)
    record["read_only"] = _chmod_read_only(run_dir)
    _write_json(args, "checkpoint_freeze.json", record)
    print(json.dumps({"primary": record["primary"],
                      "primary_sha256": record["primary_sha256"],
                      "primary_bytes": record["primary_bytes"],
                      "best_equals_last": checkpoints["best_equals_last"],
                      "read_only": record["read_only"],
                      "step2c_weights": record["step2c_handoff"]["weights"]},
                     ensure_ascii=False, indent=2))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    pools = _pools(args)
    preflight = json.loads((args.output / "preflight.json").read_text(encoding="utf-8"))
    environment = json.loads((args.output / "environment.json").read_text(encoding="utf-8"))
    manifest = _train_manifest(args)
    full_pool = json.loads((args.output / "full_pool_check.json").read_text(encoding="utf-8"))
    staging = json.loads((args.output / "staging_integrity.json").read_text(encoding="utf-8"))

    def optional(name):
        path = args.output / name
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    loader = optional("loader_sanity.json")
    baseline = optional("baseline_metrics.json")
    post = optional("post_metrics.json")
    training = optional("training.json")
    freeze = optional("checkpoint_freeze.json")
    if post is not None and training is not None and not post.get("verdict"):
        post["verdict"] = full_finetune_verdict(
            training, post, loader_ok=bool((loader or {}).get("ok")),
            full_pool_ok=bool(full_pool.get("ok")))
    verdict = ((post or {}).get("verdict")
               or {"verdict": "NOT_EVALUATED", "reason": "full fine-tune not run"})
    server_run = {
        "upload_sha256sums": optional("upload_sha256sums.json"),
        "evidence_archive": optional("evidence_archive.json"),
        "frozen_input_verification": optional("frozen_input_verification.json"),
        "data_yaml_record": optional("data_yaml_record.json"),
        "checkpoint_hashes": optional("checkpoint_hashes.json"),
        "training_manifest_provenance": (training or {}).get("manifest_provenance"),
    }
    manifest_path = _manifest_path(args)
    summary = build_summary(pools, preflight, environment, manifest, full_pool, staging,
                            loader, baseline, post, training, verdict, freeze,
                            manifest_sha256=(sha256_file(manifest_path)
                                             if manifest_path.is_file() else None),
                            server_run=server_run)
    provenance = _provenance()
    provenance["server_run"] = server_run
    manifest_json = build_manifest(
        pools, summary, code_commit=git_commit_of(args.repo_root),
        generated_at=_stamp(args),
        config={"positive_root": str(args.positive_root),
                "negative_root": str(args.negative_root),
                "weight": str(args.weight), "output": str(args.output),
                "device": args.device, "batch": args.batch, "epochs": args.epochs,
                "imgsz": args.imgsz, "seed": SEED,
                "train_set_recall_bar": TRAIN_SET_RECALL_BAR},
        artifact_root=args.output, provenance=provenance)
    paths = write_reports(args.output, manifest=manifest, loader=loader, summary=summary,
                          manifest_json=manifest_json)
    print(json.dumps({"verdict": verdict, "artifacts": paths}, ensure_ascii=False,
                     indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    commands = {
        "preflight": cmd_preflight, "stage": cmd_stage,
        "loader-sanity": cmd_loader_sanity, "baseline": cmd_baseline,
        "train": cmd_train, "post": cmd_post, "freeze": cmd_freeze,
        "report": cmd_report,
    }
    try:
        return commands[args.command](args)
    except (SealedAssetError, Step2AError, Step2BError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
