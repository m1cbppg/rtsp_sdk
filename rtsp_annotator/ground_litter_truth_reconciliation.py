"""Step 1C-1R: reconcile the truth of the 60 TRUTH_REVIEW_REQUIRED episodes.

Step 1C-1 froze localization.  It also surfaced 60 episodes whose *truth* a human could
not settle while looking at the frame — too small, not litter, incomplete, or several
objects with an unclear intended target.  This step lets a human classify exactly those
60 into one of five outcomes and writes an immutable **truth overlay**.

It is deliberately narrow:
  * it never re-reviews the 109 VERIFIED_BBOX or the 11 LOCALIZATION_UNRESOLVED,
  * it never touches bbox/localization — no box verdict, no candidate selection, no
    detector, no segmentation,
  * it never rewrites Gold; the overlay is a separate artifact,
  * it never derives a decision from the previous free-text reason (no NLP, §22).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .ground_litter_localization_review import (  # reuse, do not duplicate
    LocalizationError,
    ReviewError,
    SealedAssetError,
    assert_not_sealed,
    atomic_write_json,
    read_jsonl,
    sha256_file,
    write_jsonl,
)

SCHEMA_VERSION = "ground_litter_truth_reconciliation_v1"
GENERATOR_VERSION = "step1c1r-1.0.0"
REVIEW_SCHEMA_VERSION = "truth_reconciliation_review_v1"

#: The five and only five human outcomes (§3).
DECISIONS = (
    "KEEP_REQUIRED",
    "IGNORE_SMALL",
    "NON_LITTER",
    "UNCERTAIN",
    "IDENTITY_AMBIGUOUS",
)

#: KEEP_REQUIRED -> REQUIRED_LITTER and so on.  IDENTITY_AMBIGUOUS deliberately maps to
#: None: it must not be smuggled into UNCERTAIN or any other truth class (§7).
DECISION_TO_TRUTH_CLASS: dict[str, str | None] = {
    "KEEP_REQUIRED": "REQUIRED_LITTER",
    "IGNORE_SMALL": "IGNORE_SMALL",
    "NON_LITTER": "NON_LITTER",
    "UNCERTAIN": "UNCERTAIN",
    "IDENTITY_AMBIGUOUS": None,
}

#: Counting buckets for the summary; IDENTITY_AMBIGUOUS is its own bucket rather than a
#: fabricated truth class.
EFFECTIVE_BUCKETS = (
    "REQUIRED_LITTER",
    "IGNORE_SMALL",
    "NON_LITTER",
    "UNCERTAIN",
    "IDENTITY_AMBIGUOUS",
)

TRUTH_CLASSES = ("REQUIRED_LITTER", "IGNORE_SMALL", "NON_LITTER", "UNCERTAIN")

REQUIRED_SOURCE_STATUS = "RECOVERED_SOURCE_NATIVE"
REQUIRED_LOCALIZATION_STATUS = "VERIFIED_BBOX"

ORIGINS = ("historical_litter", "box_wrong", "manual_missing_target", "split_derived")


class ReconciliationError(RuntimeError):
    """Truth reconciliation could not proceed."""


# --------------------------------------------------------------------------- #
# input
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EpisodeContext:
    """Everything needed to decide truth eligibility, from the frozen upstream steps."""

    episode_id: str
    camera_id: str
    scene_version: str
    origin: str
    source_timestamp: str | None
    source_file_id: str | None
    source_width: int
    source_height: int
    verification_frame_path: str
    source_recovery_status: str
    original_truth_class: str
    localization_status: str
    localization_decision: str | None
    localization_verified_bbox: list[float] | None
    original_bbox: list[float] | None
    original_point: Mapping[str, Any] | None
    original_point_source: Mapping[str, Any] | None
    in_scope: bool
    prior_truth_review_reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "camera_id": self.camera_id,
            "scene_version": self.scene_version,
            "origin": self.origin,
            "source_timestamp": self.source_timestamp,
            "source_file_id": self.source_file_id,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "verification_frame_path": self.verification_frame_path,
            "source_recovery_status": self.source_recovery_status,
            "original_truth_class": self.original_truth_class,
            "localization_status": self.localization_status,
            "localization_decision": self.localization_decision,
            "localization_verified_bbox": self.localization_verified_bbox,
            "original_bbox": self.original_bbox,
            "original_point": dict(self.original_point or {}) or None,
            "original_point_source": dict(self.original_point_source or {}) or None,
            "in_scope": self.in_scope,
            "prior_truth_review_reason": self.prior_truth_review_reason,
        }


@dataclass(frozen=True)
class ReconciliationInput:
    gold_path: Path
    gold_sha256: str
    recovery_evidence_sha256: str
    localization_path: Path
    localization_sha256: str
    localization_summary_sha256: str
    localization_manifest_sha256: str
    step1c1_code_commit: str
    episodes: tuple[EpisodeContext, ...]

    @property
    def required_before(self) -> int:
        return len(self.episodes)

    @property
    def in_scope(self) -> tuple[EpisodeContext, ...]:
        return tuple(e for e in self.episodes if e.in_scope)

    @property
    def by_id(self) -> dict[str, EpisodeContext]:
        return {e.episode_id: e for e in self.episodes}

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for episode in self.episodes:
            out[episode.localization_status] = out.get(episode.localization_status, 0) + 1
        return out


def load_reconciliation_input(
    gold_path: Path | str,
    recovery_root: Path | str,
    localization_root: Path | str,
    *,
    gold_manifest: Path | str | None = None,
) -> ReconciliationInput:
    gold = Path(gold_path)
    recovery = Path(recovery_root)
    localization = Path(localization_root)
    assert_not_sealed(gold, recovery, localization, gold_manifest)
    for path in (gold, localization / "localizations.jsonl"):
        if not path.is_file():
            raise FileNotFoundError(f"required input missing: {path}")

    evidence_path = recovery / "episode_source_evidence.jsonl"
    evidence = {str(r.get("episode_id")): r for r in read_jsonl(evidence_path)}
    loc_rows = read_jsonl(localization / "localizations.jsonl")
    if not loc_rows:
        raise ReconciliationError("localization artifact has no rows")

    loc_manifest_path = localization / "MANIFEST.json"
    loc_summary_path = localization / "SUMMARY.json"
    loc_manifest = (json.loads(loc_manifest_path.read_text(encoding="utf-8"))
                    if loc_manifest_path.is_file() else {})

    episodes: list[EpisodeContext] = []
    seen: set[str] = set()
    for row in loc_rows:
        episode_id = str(row.get("episode_id") or "")
        if not episode_id:
            raise ReconciliationError("localization row without episode_id")
        if episode_id in seen:
            raise ReconciliationError(f"duplicate episode_id: {episode_id}")
        seen.add(episode_id)
        if str(row.get("truth_class") or "REQUIRED_LITTER") != "REQUIRED_LITTER":
            continue                      # only REQUIRED_LITTER is in this experiment
        ev = evidence.get(episode_id) or {}
        status = str(row.get("localization_status") or "NEEDS_RELOCALIZATION")
        episodes.append(EpisodeContext(
            episode_id=episode_id,
            camera_id=str(row.get("camera_id") or ""),
            scene_version=str(row.get("scene_version") or ""),
            origin=str(row.get("origin") or "historical_litter"),
            source_timestamp=row.get("source_timestamp"),
            source_file_id=row.get("source_file_id"),
            source_width=int(row.get("source_width") or 0),
            source_height=int(row.get("source_height") or 0),
            verification_frame_path=str(row.get("verification_frame_path") or ""),
            source_recovery_status=str(ev.get("source_recovery_status") or ""),
            original_truth_class="REQUIRED_LITTER",
            localization_status=status,
            localization_decision=row.get("localization_decision"),
            localization_verified_bbox=(list(row["verified_bbox"])
                                        if row.get("verified_bbox") else None),
            original_bbox=list(row["original_bbox"]) if row.get("original_bbox") else None,
            original_point=row.get("original_point"),
            original_point_source=row.get("original_point_source"),
            in_scope=status == "TRUTH_REVIEW_REQUIRED",
            prior_truth_review_reason=str(row.get("truth_review_reason") or "").strip(),
        ))

    episodes.sort(key=lambda e: (not e.in_scope, e.camera_id, e.episode_id))
    return ReconciliationInput(
        gold_path=gold,
        gold_sha256=sha256_file(gold),
        recovery_evidence_sha256=(sha256_file(evidence_path)
                                  if evidence_path.is_file() else ""),
        localization_path=localization / "localizations.jsonl",
        localization_sha256=sha256_file(localization / "localizations.jsonl"),
        localization_summary_sha256=(sha256_file(loc_summary_path)
                                     if loc_summary_path.is_file() else ""),
        localization_manifest_sha256=(sha256_file(loc_manifest_path)
                                      if loc_manifest_path.is_file() else ""),
        step1c1_code_commit=str(loc_manifest.get("code_commit") or ""),
        episodes=tuple(episodes),
    )


def verify_preflight(data: ReconciliationInput) -> dict[str, Any]:
    counts = data.counts()
    checks = {
        "required_before": data.required_before,
        "verified_bbox_before": counts.get("VERIFIED_BBOX", 0),
        "localization_unresolved": counts.get("LOCALIZATION_UNRESOLVED", 0),
        "truth_review_required": counts.get("TRUTH_REVIEW_REQUIRED", 0),
        "total": data.required_before,
        "sum_matches": (counts.get("VERIFIED_BBOX", 0)
                        + counts.get("LOCALIZATION_UNRESOLVED", 0)
                        + counts.get("TRUTH_REVIEW_REQUIRED", 0)) == data.required_before,
    }
    return checks


# --------------------------------------------------------------------------- #
# review state
# --------------------------------------------------------------------------- #


@dataclass
class ReviewState:
    path: Path
    schema_version: str = REVIEW_SCHEMA_VERSION
    input_localization_sha256: str = ""
    in_scope_count: int = 0
    decisions: dict[str, dict[str, Any]] = field(default_factory=dict)
    audit_trail: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path | str, *, data: ReconciliationInput | None = None
             ) -> "ReviewState":
        target = Path(path)
        if not target.is_file():
            state = cls(path=target)
            if data is not None:
                state.input_localization_sha256 = data.localization_sha256
                state.in_scope_count = len(data.in_scope)
            return state
        payload = json.loads(target.read_text(encoding="utf-8"))
        state = cls(
            path=target,
            schema_version=str(payload.get("review_schema_version") or REVIEW_SCHEMA_VERSION),
            input_localization_sha256=str(
                (payload.get("input") or {}).get("localization_sha256") or ""),
            in_scope_count=int((payload.get("input") or {}).get("in_scope_count") or 0),
            decisions=dict(payload.get("decisions") or {}),
            audit_trail=list(payload.get("audit_trail") or []),
        )
        if data is not None and state.input_localization_sha256 and \
                state.input_localization_sha256 != data.localization_sha256:
            raise ReviewError("review state belongs to a different localization artifact")
        return state

    def get(self, episode_id: str) -> dict[str, Any] | None:
        return self.decisions.get(episode_id)

    def save(self) -> None:
        atomic_write_json(self.path, {
            "review_schema_version": REVIEW_SCHEMA_VERSION,
            "loaded_schema_version": self.schema_version,
            "input": {"localization_sha256": self.input_localization_sha256,
                      "in_scope_count": self.in_scope_count},
            "decisions": self.decisions,
            "audit_trail": self.audit_trail,
        })

    def _record(self, episode_id: str, action: str, payload: Mapping[str, Any]) -> None:
        self.audit_trail.append({
            "at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "episode_id": episode_id, "action": action, "payload": dict(payload)})

    # -- mutations ---------------------------------------------------------- #

    def decide(self, episode: EpisodeContext, decision: str, *,
               reason: str = "") -> dict[str, Any]:
        if not episode.in_scope:
            raise ReviewError(
                f"{episode.episode_id} is not TRUTH_REVIEW_REQUIRED; it is frozen at "
                f"{episode.localization_status} and must not be re-reviewed")
        if decision not in DECISIONS:
            raise ReviewError(f"unknown reconciliation decision {decision!r}")
        if decision == "IDENTITY_AMBIGUOUS" and not reason.strip():
            raise ReviewError("IDENTITY_AMBIGUOUS requires a short reason")
        truth_class = DECISION_TO_TRUTH_CLASS[decision]
        now = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        previous = self.decisions.get(episode.episode_id) or {}
        row = {
            "episode_id": episode.episode_id,
            "truth_target_id": episode.episode_id,
            "camera_id": episode.camera_id,
            "scene_version": episode.scene_version,
            "origin": episode.origin,
            "source_timestamp": episode.source_timestamp,
            "original_truth_class": episode.original_truth_class,
            "reconciled_truth_class": truth_class,
            "reconciliation_decision": decision,
            "effective_truth_bucket": decision if decision == "IDENTITY_AMBIGUOUS"
            else str(truth_class),
            "training_excluded": decision != "KEEP_REQUIRED",
            "exclusion_reason": (None if decision == "KEEP_REQUIRED" else decision),
            # audit context (§16)
            "previous_truth_class": episode.original_truth_class,
            "previous_localization_status": episode.localization_status,
            "previous_truth_review_reason": episode.prior_truth_review_reason or None,
            "prior_reason": episode.prior_truth_review_reason or None,
            "review_reason_optional": reason.strip(),
            "localization_status": episode.localization_status,
            "source_recovery_status": episode.source_recovery_status,
            "verified_bbox": list(episode.localization_verified_bbox)
            if episode.localization_verified_bbox else None,
            "reviewed_at": now,
            "review_status": "human_reviewed",
            "review_schema_version": REVIEW_SCHEMA_VERSION,
            "revision": int(previous.get("revision") or 0) + 1,
        }
        self.decisions[episode.episode_id] = row
        self._record(episode.episode_id, f"reconcile:{decision}", {
            "reconciled_truth_class": truth_class, "reason": reason.strip(),
            "prior_reason": episode.prior_truth_review_reason or None})
        self.save()
        return row

    def reset(self, episode_id: str) -> None:
        self.decisions.pop(episode_id, None)
        self._record(episode_id, "reset", {})
        self.save()

    def progress(self, data: ReconciliationInput) -> dict[str, int]:
        scope = data.in_scope
        reviewed = sum(1 for e in scope if e.episode_id in self.decisions)
        return {"total": len(scope), "reviewed": reviewed,
                "pending": len(scope) - reviewed}


# --------------------------------------------------------------------------- #
# eligibility + overlay
# --------------------------------------------------------------------------- #


def eligibility(episode: EpisodeContext, decision: Mapping[str, Any] | None) -> dict[str, Any]:
    """The single rule that decides whether an episode may enter detector training."""
    in_scope = episode.in_scope
    if in_scope and decision is None:
        effective_truth = None
        bucket = "PENDING_RECONCILIATION"
        exclusion = "truth_reconciliation_pending"
    elif in_scope:
        effective_truth = decision.get("reconciled_truth_class")
        bucket = str(decision.get("effective_truth_bucket") or "IDENTITY_AMBIGUOUS")
        exclusion = decision.get("exclusion_reason")
    else:
        effective_truth = episode.original_truth_class
        bucket = episode.original_truth_class
        exclusion = None

    truth_ok = effective_truth == "REQUIRED_LITTER"
    source_ok = episode.source_recovery_status == REQUIRED_SOURCE_STATUS
    localization_ok = episode.localization_status == REQUIRED_LOCALIZATION_STATUS
    eligible = bool(truth_ok and source_ok and localization_ok)

    if eligible:
        reason = None
    elif not truth_ok:
        reason = (exclusion or f"effective_truth_class_{bucket}")
    elif not source_ok:
        reason = f"source_recovery_status_{episode.source_recovery_status or 'unknown'}"
    else:
        reason = f"localization_status_{episode.localization_status}"

    return {
        "effective_truth_class": effective_truth,
        "effective_truth_bucket": bucket,
        "truth_ok": truth_ok,
        "source_ok": source_ok,
        "localization_ok": localization_ok,
        "training_eligible": eligible,
        "training_exclusion_reason": reason,
    }


def build_overlay_rows(
    data: ReconciliationInput, state: ReviewState,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (truth overlay rows, training eligibility manifest rows)."""
    overlay: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for episode in data.episodes:
        decision = state.get(episode.episode_id) if episode.in_scope else None
        verdict = eligibility(episode, decision)
        overlay.append({
            "episode_id": episode.episode_id,
            "truth_target_id": episode.episode_id,
            "camera_id": episode.camera_id,
            "scene_version": episode.scene_version,
            "origin": episode.origin,
            "in_scope": episode.in_scope,
            "original_truth_class": episode.original_truth_class,
            "reconciled_truth_class": (decision or {}).get("reconciled_truth_class"),
            "effective_truth_class": verdict["effective_truth_class"],
            "effective_truth_bucket": verdict["effective_truth_bucket"],
            "reconciliation_decision": (decision or {}).get("reconciliation_decision"),
            "reason": (decision or {}).get("review_reason_optional") or None,
            "prior_reason": episode.prior_truth_review_reason or None,
            "training_excluded": bool(verdict["training_eligible"] is False
                                      and episode.in_scope
                                      and (decision or {}).get(
                                          "reconciliation_decision") != "KEEP_REQUIRED"),
            "exclusion_reason": (decision or {}).get("exclusion_reason"),
            "localization_status": episode.localization_status,
            "source_recovery_status": episode.source_recovery_status,
            "verified_bbox": episode.localization_verified_bbox,
            "reviewed_at": (decision or {}).get("reviewed_at"),
            "review_schema_version": REVIEW_SCHEMA_VERSION,
            "evaluation_semantics": _evaluation_semantics(verdict["effective_truth_bucket"]),
        })
        manifest.append({
            "episode_id": episode.episode_id,
            "camera_id": episode.camera_id,
            "origin": episode.origin,
            "effective_truth_class": verdict["effective_truth_class"],
            "effective_truth_bucket": verdict["effective_truth_bucket"],
            "source_recovery_status": episode.source_recovery_status,
            "localization_status": episode.localization_status,
            "verified_bbox": episode.localization_verified_bbox,
            "verification_frame_path": episode.verification_frame_path,
            "source_file_id": episode.source_file_id,
            "source_timestamp": episode.source_timestamp,
            "source_width": episode.source_width,
            "source_height": episode.source_height,
            "training_eligible": verdict["training_eligible"],
            "training_exclusion_reason": verdict["training_exclusion_reason"],
        })
    overlay.sort(key=lambda r: (r["camera_id"], r["episode_id"]))
    manifest.sort(key=lambda r: (r["camera_id"], r["episode_id"]))
    return overlay, manifest


def _evaluation_semantics(bucket: str) -> str:
    """What the bucket means for the later evaluation step (§14)."""
    return {
        "REQUIRED_LITTER": "required_recall_denominator_candidate",
        "IGNORE_SMALL": "excluded_from_required_recall_denominator",
        "NON_LITTER": "excluded_from_positive_denominator",
        "UNCERTAIN": "excluded_from_tp_fp_fn",
        "IDENTITY_AMBIGUOUS": "excluded_from_automated_bbox_scoring_pending_identity",
        "PENDING_RECONCILIATION": "excluded_pending_human_reconciliation",
    }.get(bucket, "unknown")


def build_summary(data: ReconciliationInput, state: ReviewState,
                  overlay: Sequence[Mapping[str, Any]],
                  manifest: Sequence[Mapping[str, Any]],
                  preflight: Mapping[str, Any]) -> dict[str, Any]:
    decisions = {d: 0 for d in DECISIONS}
    buckets = {b: 0 for b in EFFECTIVE_BUCKETS}
    pending_bucket = 0
    per_camera: dict[str, dict[str, int]] = {}
    per_origin: dict[str, dict[str, int]] = {}
    for row in overlay:
        bucket = str(row["effective_truth_bucket"])
        if bucket in buckets:
            buckets[bucket] += 1
        elif bucket == "PENDING_RECONCILIATION":
            # never silently counted as REQUIRED_LITTER while a human has not decided
            pending_bucket += 1
        decision = row.get("reconciliation_decision")
        if decision in decisions:
            decisions[decision] += 1
        cam = per_camera.setdefault(str(row["camera_id"]), {
            "original_required": 0, "keep_required": 0, "ignore_small": 0,
            "non_litter": 0, "uncertain": 0, "identity_ambiguous": 0,
            "training_eligible": 0})
        cam["original_required"] += 1
        if decision:
            cam[decision.lower()] = cam.get(decision.lower(), 0) + 1
        origin = per_origin.setdefault(str(row["origin"]), {d: 0 for d in DECISIONS})
        if decision in origin:
            origin[decision] += 1

    eligible_ids: list[str] = []
    excluded_truth: list[str] = []
    excluded_localization: list[str] = []
    for row in manifest:
        if row["training_eligible"]:
            eligible_ids.append(str(row["episode_id"]))
            cam = per_camera.setdefault(str(row["camera_id"]), {
                "original_required": 0, "keep_required": 0, "ignore_small": 0,
                "non_litter": 0, "uncertain": 0, "identity_ambiguous": 0,
                "training_eligible": 0})
            cam["training_eligible"] += 1
        else:
            reason = str(row.get("training_exclusion_reason") or "")
            if reason.startswith("localization_status_") or \
                    reason.startswith("source_recovery_status_"):
                excluded_localization.append(str(row["episode_id"]))
            else:
                excluded_truth.append(str(row["episode_id"]))

    scope = data.in_scope
    reviewed = sum(1 for e in scope if e.episode_id in state.decisions)
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "input": {
            "gold_artifact": str(data.gold_path),
            "gold_sha256": data.gold_sha256,
            "recovery_evidence_sha256": data.recovery_evidence_sha256,
            "localization_artifact": str(data.localization_path),
            "localization_sha256": data.localization_sha256,
            "localization_summary_sha256": data.localization_summary_sha256,
            "localization_manifest_sha256": data.localization_manifest_sha256,
            "step1c1_code_commit": data.step1c1_code_commit,
            "required_before_reconciliation": data.required_before,
            "verified_bbox_before": preflight.get("verified_bbox_before"),
            "localization_unresolved": preflight.get("localization_unresolved"),
            "truth_review_required": preflight.get("truth_review_required"),
        },
        "review": {
            "total_in_scope": len(scope),
            "reviewed": reviewed,
            "pending": len(scope) - reviewed,
            "skipped": 0,
        },
        "reconciliation": decisions,
        "effective_truth_totals": buckets,
        "pending_reconciliation_count": pending_bucket,
        "total_episodes": len(overlay),
        "training": {
            "training_eligible_episode_count": len(eligible_ids),
            "training_eligible_episode_ids": sorted(eligible_ids),
            "training_excluded_truth_reason": sorted(excluded_truth),
            "training_excluded_localization_reason": sorted(excluded_localization),
        },
        "per_camera": dict(sorted(per_camera.items())),
        "per_origin": {k: per_origin[k] for k in sorted(per_origin)},
        "consistency": {
            "reconciliation_sum": sum(decisions.values()),
            "in_scope_count": len(scope),
            "reconciliation_sum_equals_in_scope": sum(decisions.values()) == len(scope),
            "buckets_plus_pending_equals_total":
                sum(buckets.values()) + pending_bucket == len(overlay),
            "effective_truth_totals_sum": sum(buckets.values()),
            "verified_plus_unresolved_plus_truth_review":
                (preflight.get("verified_bbox_before", 0)
                 + preflight.get("localization_unresolved", 0)
                 + preflight.get("truth_review_required", 0)),
            "totals_180": (preflight.get("verified_bbox_before", 0)
                           + preflight.get("localization_unresolved", 0)
                           + preflight.get("truth_review_required", 0)) == 180,
        },
        "boundaries": {
            "gold_modified": False,
            "localization_artifact_modified": False,
            "recovery_evidence_modified": False,
            "bbox_generated_or_modified": False,
            "localization_re_reviewed": False,
            "proposal_run": False,
            "detector_run": False,
            "segmentation_run": False,
            "training_tiles_generated": False,
            "training_started": False,
            "sealed_accessed": False,
            "auto_classified_from_reason_text": False,
            "identity_ambiguous_folded_into_uncertain": False,
        },
    }


def build_manifest(data: ReconciliationInput, summary: Mapping[str, Any], *,
                   code_commit: str, generated_at: str, config: Mapping[str, Any],
                   artifact_root: Path, overlay_path: Path, manifest_path: Path,
                   review_state_path: Path, provenance: Mapping[str, Any]
                   ) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "generated_at": generated_at,
        "code_commit": code_commit,
        "gold_input_sha256": data.gold_sha256,
        "recovery_evidence_sha256": data.recovery_evidence_sha256,
        "localization_input_sha256": data.localization_sha256,
        "truth_reconciliation_sha256": sha256_file(overlay_path)
        if Path(overlay_path).is_file() else "",
        "training_episode_manifest_sha256": sha256_file(manifest_path)
        if Path(manifest_path).is_file() else "",
        "artifact_root": str(artifact_root),
        "config": dict(config),
        "outputs": {
            "truth_reconciliation": str(overlay_path),
            "training_episode_manifest": str(manifest_path),
            "review_state": str(review_state_path),
        },
        "upstream_provenance": dict(provenance),
        "counts": {
            "required_before_reconciliation": data.required_before,
            "in_scope": len(data.in_scope),
            "reviewed": summary["review"]["reviewed"],
            "training_eligible_episode_count":
                summary["training"]["training_eligible_episode_count"],
        },
        "boundaries": dict(summary["boundaries"]),
    }


def build_upstream_provenance(
    *, step1c0_execution_commit: str, step1c0_manifest_commit: str,
    step1c0_evidence_commit: str, step1c1_execution_commit: str,
    step1c1_manifest_commit: str, step1c1_evidence_commit: str,
    code_equivalence_verified: bool, note: str = "",
) -> dict[str, Any]:
    """Record both upstream commit pairs so no step has to infer them (§27 pattern)."""
    return {
        "step1c0_reported_execution_commit": step1c0_execution_commit,
        "step1c0_manifest_commit": step1c0_manifest_commit,
        "step1c0_evidence_commit": step1c0_evidence_commit,
        "step1c1_reported_execution_commit": step1c1_execution_commit,
        "step1c1_manifest_commit": step1c1_manifest_commit,
        "step1c1_evidence_commit": step1c1_evidence_commit,
        "step1c0_code_equivalence_verified": bool(code_equivalence_verified),
        "step1c1_code_equivalence_verified": bool(code_equivalence_verified),
        "mismatch_explained": (
            "Each step's MANIFEST records the git HEAD at the moment its evidence was "
            "written, which is later than the commit that introduced the code when a "
            "further evidence commit happened in between. The relevant module files are "
            "byte-identical between the reported execution commit and the recorded one "
            "(checked with git diff), so the recorded hash names the same code. Recorded "
            "here so no downstream step has to guess."),
        "note": note,
    }


def write_outputs(output_dir: Path, *, overlay: Sequence[Mapping[str, Any]],
                  manifest_rows: Sequence[Mapping[str, Any]],
                  summary: Mapping[str, Any], manifest: Mapping[str, Any]
                  ) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = output_dir / "truth_reconciliation.jsonl"
    write_jsonl(overlay_path, overlay)
    training_path = output_dir / "training_episode_manifest.jsonl"
    write_jsonl(training_path, manifest_rows)
    summary_path = output_dir / "SUMMARY.json"
    atomic_write_json(summary_path, summary)
    # The manifest must fingerprint the bytes that were actually written, so the two
    # artifact hashes are computed here rather than at manifest-build time.
    payload = dict(manifest)
    payload["truth_reconciliation_sha256"] = sha256_file(overlay_path)
    payload["training_episode_manifest_sha256"] = sha256_file(training_path)
    payload["summary_sha256"] = sha256_file(summary_path)
    manifest_path = output_dir / "MANIFEST.json"
    atomic_write_json(manifest_path, payload)
    return {"truth_reconciliation": str(overlay_path),
            "training_episode_manifest": str(training_path),
            "summary": str(summary_path), "manifest": str(manifest_path)}
