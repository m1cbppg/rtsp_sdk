import unittest

import cv2
import numpy as np

from rtsp_annotator.ground_litter_alignment import ViewAlignment


def scene(seed=12):
    rng=np.random.default_rng(seed)
    gray=rng.integers(15,240,(720,960),dtype=np.uint8)
    gray=cv2.GaussianBlur(gray,(5,5),1)
    return cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)


class ViewAlignmentTests(unittest.TestCase):
    def test_illumination_and_partial_occlusion_keep_fixed_view(self):
        reference=scene()
        current=(reference*.65+25).astype(np.uint8)
        current[260:570,270:720]=80
        alignment=ViewAlignment(reference,{})
        alignment.check(current)
        self.assertGreater(alignment.diagnostics['inliers'],12)
        self.assertLess(alignment.diagnostics['displacement_native_px'],2)

    def test_translation_and_zoom_still_require_recalibration(self):
        reference=scene()
        for transform in (np.float32([[1,0,20],[0,1,0]]),
                          cv2.getRotationMatrix2D((480,360),0,1.04)):
            current=cv2.warpAffine(reference,transform,(960,720))
            with self.assertRaisesRegex(ValueError,'view_changed_recalibration_required'):
                ViewAlignment(reference,{}).check(current)

    def test_unrelated_view_and_blank_frame_fail_closed(self):
        alignment=ViewAlignment(scene(),{})
        with self.assertRaisesRegex(ValueError,'view_alignment_unknown'):
            alignment.check(scene(51))
        with self.assertRaisesRegex(ValueError,'image_quality_unknown'):
            alignment.check(np.zeros((720,960,3),np.uint8))

    def test_one_small_patch_cannot_certify_whole_frame(self):
        reference=np.full((720,960,3),100,np.uint8)
        reference[300:380,400:480]=scene()[:80,:80]
        with self.assertRaisesRegex(ValueError,'view_alignment_unknown'):
            ViewAlignment(reference,{}).check(reference)

    def test_burned_in_overlay_cannot_hide_camera_motion(self):
        reference=scene()
        current=cv2.warpAffine(reference,np.float32([[1,0,20],[0,1,0]]),(960,720))
        current[:130]=reference[:130]
        camera={'overlay_exclude_zones':[[[0,0],[1,0],[1,.2],[0,.2]]]}
        with self.assertRaisesRegex(ValueError,'view_changed_recalibration_required'):
            ViewAlignment(reference,camera).check(current)


if __name__=='__main__':
    unittest.main()
