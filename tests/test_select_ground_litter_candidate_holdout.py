import copy

import pytest

from scripts.select_ground_litter_candidate_holdout import select_holdout


def make_pool(device: str) -> dict:
    rows = []
    for source in ("semantic_full", "semantic_tile"):
        for rank in range(1, 61):
            for repeat in range(2):
                proposal_id = f"{device[-5:]}-{source}-{rank:02d}-{repeat}"
                rows.append({
                    "proposal_id": proposal_id,
                    "device_code": device,
                    "timestamp": "2026-09-18 12:00:00",
                    "frame_id": "f00s00",
                    "bbox": [10, 10, 20, 20],
                    "source": source,
                    "score": 0.2,
                    "rank": rank,
                    "class_id": 3,
                    "class_name": "Plastic",
                    "raw_kept_count": 60,
                    "image": f"crops/{proposal_id}.jpg",
                })
    return {"device_code": device, "semantic": rows, "random": []}


def test_selection_is_deterministic_balanced_and_excludes_prior_ids():
    pools = [make_pool(f"4418020903132200{suffix}") for suffix in (1021, 1022)]
    excluded = {pools[0]["semantic"][0]["proposal_id"]}
    first = select_holdout(pools, excluded, per_stratum=2, seed="fixed")
    second = select_holdout(copy.deepcopy(pools), excluded, per_stratum=2, seed="fixed")
    assert first == second
    assert first["count"] == 2 * 4 * 2 * 2
    assert excluded.isdisjoint(row["proposal_id"] for row in first["selected"])
    assert len({row["holdout_stratum"] for row in first["selected"]}) == 16
    assert all(row["holdout_sample_count"] == 2 for row in first["selected"])


def test_selection_rejects_classifier_derived_fields():
    pool = make_pool("44180209031322001021")
    pool["semantic"][0]["turhancan_classifier_score"] = 1.0
    with pytest.raises(ValueError, match="classifier-derived fields"):
        select_holdout([pool], set(), per_stratum=1, seed="fixed")


def test_selection_fails_when_a_stratum_is_too_small():
    pool = make_pool("44180209031322001021")
    pool["semantic"] = [row for row in pool["semantic"] if row["source"] == "semantic_full"]
    with pytest.raises(ValueError, match="insufficient candidates"):
        select_holdout([pool], set(), per_stratum=1, seed="fixed")
