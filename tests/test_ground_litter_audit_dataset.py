from rtsp_annotator.ground_litter_audit_dataset import Proposal, box_iou, dedupe_persistent_proposals, merge_proposals, select_files_evenly


def test_even_selection_is_deterministic_and_spans_day():
    rows=[{"file_id":str(i),"record_start":f"2026-09-20 {i:02d}:00:00"} for i in range(24)]
    assert [r["file_id"] for r in select_files_evenly(rows,4)]==["3","9","14","20"]


def test_source_quota_preserves_random_coverage():
    rows=[Proposal("f",(i*20,0,i*20+10,10),"semantic_tile",100-i) for i in range(8)]
    rows += [Proposal("f",(200+i*20,0,210+i*20,10),"random_grid",i) for i in range(3)]
    out=merge_proposals(rows,{"semantic_tile":2,"random_grid":2},10)
    assert sum(x.source=="semantic_tile" for x in out)==8
    assert sum(x.source=="random_grid" for x in out)==2


def test_unused_capacity_fill_supports_generator_input():
    rows=(Proposal(f"s{i}",(20*i,0,20*i+10,10),"semantic_tile_low",i) for i in range(5))
    out=merge_proposals(rows,{"semantic_tile_low":1},4)
    assert len(out)==4


def test_total_budget_does_not_starve_last_source_when_quotas_fit():
    rows=[]
    for source,offset in (("semantic_tile",0),("temporal",1000),("random_grid",2000)):
        rows += [Proposal(f"f{i}",(offset,0,offset+10,10),source,10-i) for i in range(5)]
    out=merge_proposals(rows,{"semantic_tile":2,"temporal":2,"random_grid":2},6)
    assert {source:sum(x.source==source for x in out) for source in ("semantic_tile","temporal","random_grid")} == {"semantic_tile":2,"temporal":2,"random_grid":2}


def test_unused_source_budget_is_filled_without_dropping_random():
    rows=[Proposal(f"r{i}",(0,0,10,10),"random_grid",i) for i in range(2)]
    rows += [Proposal(f"s{i}",(20,0,30,10),"semantic_tile_low",i) for i in range(6)]
    out=merge_proposals(rows,{"random_grid":2,"semantic_tile_high":2,"semantic_tile_low":1},5)
    assert len(out)==5
    assert sum(x.source=="random_grid" for x in out)==2


def test_iou_known_values():
    assert box_iou((0,0,10,10),(0,0,10,10))==1
    assert box_iou((0,0,10,10),(20,20,30,30))==0


def test_persistent_dedupe_keeps_random_and_collapses_semantic_repeat():
    rows=[Proposal("f1",(10,10,30,30),"semantic_tile",.9),Proposal("f2",(11,10,31,30),"semantic_full",.8),Proposal("f1",(10,10,30,30),"random_grid",.1),Proposal("f2",(10,10,30,30),"random_grid",.2)]
    out=dedupe_persistent_proposals(rows)
    assert sum(x.source.startswith("semantic") for x in out)==1
    assert sum(x.source=="random_grid" for x in out)==2
