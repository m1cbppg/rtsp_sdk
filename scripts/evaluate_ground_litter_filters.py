"""Frozen appearance/quality regression; no temporal or semantic truth claims."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np

from rtsp_annotator.ground_litter_facility import ReviewedNonLitter
from rtsp_annotator.ground_litter_quality import DetailLossGuard


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    samples=list(map(json.loads,(args.root/'regression/sample_manifest.jsonl').read_text().splitlines()))
    labels={r['sample_id']:r for r in json.loads((args.root/'regression/labels_draft.json').read_text())}
    camera=json.loads((args.root/'regression/profiles_reuse.json').read_text())['cameras'][0]
    frames={r['sample_id']:cv2.imread(r['path']) for r in samples}
    baseline={r['sample_id']:r for r in map(json.loads,(args.root/'runs/diagnostics-20260913-r1/gpu_actor1280/results.jsonl').read_text().splitlines())}
    quality=DetailLossGuard(cv2.imread(camera['reference_image']))
    records=[]
    for sample in samples:
        sid=sample['sample_id'];cam=copy.deepcopy(camera)
        templates=[]
        # Reviewed object profiles become usable only after their source frame.
        # No future reference or calibration self-match is counted as success.
        for index,box,context,name in [(0,[557,633,587,663],[515,470,625,700],'broom-dark'),
                (9,[557,633,587,663],[515,470,625,700],'broom-bright'),
                (32,[759,514,806,565],[704,440,855,620],'bucket-dawn')]:
            source=samples[index]
            if sid<=source['sample_id']:continue
            templates.append({'id':name,'box':box,'context_box':context,
                'reference_image':source['path'],'reference_sha256':source['sha256'],
                'camera_id':camera['device_code'],'view_id':camera['view_id'],
                'reviewed_by':'assistant_visual_provisional','reviewed_at':'2026-09-13',
                'reason':'Recognizable tool/container, not discarded ground litter'})
        cam['night']['reviewed_non_litter']=templates
        checker=ReviewedNonLitter(cam,'night');frame=frames[sid]
        proposals=baseline[sid]['candidates'];started=time.perf_counter()
        kept,decisions=checker.filter(frame,proposals)
        seconds=time.perf_counter()-started
        q=quality.inspect(frame)
        records.append({'sample_id':sid,'quality_label':labels[sid]['quality_label'],
          'quality':q,'baseline_candidates':proposals,'kept_candidates':kept,'decisions':decisions,
          'calibration_sources':[x['id'] for x in templates],'filter_seconds':seconds})
    (args.output/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    summary={'samples':len(records),'corrupt_labels':sum(r['quality_label']=='corrupt' for r in records),
      'corrupt_labels_rejected':sum(r['quality_label']=='corrupt' and not r['quality']['usable'] for r in records),
      'other_samples_rejected':sum(r['quality_label']!='corrupt' and not r['quality']['usable'] for r in records),
      'suppressed':sum(d['suppressed'] for r in records for d in r['decisions']),
      'filter_p95_seconds':float(np.percentile([r['filter_seconds'] for r in records],95)),
      'accuracy':None,'scope':'Retrospective causal-order calibration experiment; no continuous-time validation'}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
