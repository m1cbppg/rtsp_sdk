import copy
import hashlib
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from rtsp_annotator.ground_litter_facility import ReviewedNonLitter,validate_facilities
from rtsp_annotator.ground_litter_quality import DetailLossGuard


class FacilityTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        rng=np.random.default_rng(5)
        self.frame=cv2.GaussianBlur(rng.integers(20,210,(640,640,3),dtype=np.uint8),(3,3),.7)
        p=Path(self.directory.name)/'reference.png';cv2.imwrite(str(p),self.frame)
        self.camera={'device_code':'camera','view_id':'fixed','reference_size':[640,640],
          'night':{'reviewed_non_litter':[{'id':'tool','camera_id':'camera','view_id':'fixed',
          'reference_image':str(p),'reference_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
          'reviewed_by':'reviewer','reviewed_at':'2026-09-13','reason':'tool with handle',
          'box':[200,220,240,260],'context_box':[160,140,290,320]}]}}

    def test_matching_tool_suppressed_but_adjacent_object_retained(self):
        tool={'box':[200,220,240,260]};adjacent={'box':[245,220,265,240]}
        kept,decisions=ReviewedNonLitter(self.camera,'night').filter(self.frame,[tool,adjacent])
        self.assertEqual(kept,[adjacent]);self.assertTrue(decisions[0]['suppressed'])

    def test_new_object_on_tool_cannot_be_suppressed_by_old_template(self):
        changed=self.frame.copy();changed[220:260,200:240]=[230,230,230]
        item={'box':[200,220,240,260]}
        kept,decisions=ReviewedNonLitter(self.camera,'night').filter(changed,[item])
        self.assertEqual(kept,[item]);self.assertFalse(decisions[0]['suppressed'])

    def test_context_change_abstains_even_if_head_looks_same(self):
        changed=self.frame.copy();changed[140:220,160:290]=[10,10,10]
        item={'box':[200,220,240,260]}
        kept,_=ReviewedNonLitter(self.camera,'night').filter(changed,[item])
        self.assertEqual(kept,[item])

    def test_large_candidate_is_not_hidden_by_smaller_template(self):
        item={'box':[160,140,290,320]}
        self.assertEqual(ReviewedNonLitter(self.camera,'night').filter(self.frame,[item])[0],[item])

    def test_disabled_by_default_and_templates_cannot_cross_views(self):
        cam=copy.deepcopy(self.camera);cam['night']={}
        self.assertEqual(ReviewedNonLitter(cam,'night').templates,[])
        cam=copy.deepcopy(self.camera);cam['view_id']='other'
        with self.assertRaisesRegex(ValueError,'camera/view'):ReviewedNonLitter(cam,'night')

    def test_reference_fingerprint_and_review_required(self):
        cam=copy.deepcopy(self.camera);cam['night']['reviewed_non_litter'][0]['reference_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'hash mismatch'):ReviewedNonLitter(cam,'night')
        for key in ['reviewed_by','reviewed_at','reason']:
            cam=copy.deepcopy(self.camera);cam['night']['reviewed_non_litter'][0].pop(key)
            with self.assertRaises(ValueError):validate_facilities(cam,'night')

    def test_context_and_capacity_bounded(self):
        cam=copy.deepcopy(self.camera);cam['night']['reviewed_non_litter']*=17
        with self.assertRaisesRegex(ValueError,'At most 16'):validate_facilities(cam,'night')
        cam=copy.deepcopy(self.camera);cam['night']['reviewed_non_litter'][0]['context_box']=[0,0,640,640]
        with self.assertRaises(ValueError):validate_facilities(cam,'night')


class QualityTests(unittest.TestCase):
    def textured(self):
        return np.random.default_rng(21).integers(20,220,(720,1280,3),dtype=np.uint8)

    def test_broad_grey_detail_loss_abstains(self):
        frame=self.textured();broken=frame.copy();broken[50:700,:1100]=128
        self.assertFalse(DetailLossGuard(frame).inspect(broken)['usable'])

    def test_normal_exposure_and_smaller_occlusion_are_not_global_failures(self):
        frame=self.textured();current=(frame*.7+15).astype(np.uint8)
        current[260:500,250:500]=80
        self.assertTrue(DetailLossGuard(frame).inspect(current)['usable'])

    def test_flat_wall_present_in_reference_is_not_new_detail_loss(self):
        frame=self.textured();frame[:,:1100]=128
        self.assertTrue(DetailLossGuard(frame).inspect(frame)['usable'])


if __name__=='__main__':unittest.main()
