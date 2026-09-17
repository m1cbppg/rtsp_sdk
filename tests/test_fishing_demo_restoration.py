"""Offline video restoration contracts; no cameras or external services."""
import importlib.util
from pathlib import Path
import unittest

import cv2
import numpy as np

spec=importlib.util.spec_from_file_location(
    "restore_fishing_demo",Path(__file__).parents[1]/"video_tools"/"restore_fishing_demo.py")
demo=importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


class RestorationTest(unittest.TestCase):
    def test_mask_targets_overlay_but_protects_real_colours(self):
        f=np.full((360,640,3),80,np.uint8)
        cv2.rectangle(f,(30,120),(600,320),(0,0,240),3)
        cv2.rectangle(f,(180,180),(235,230),(0,0,220),-1)
        cv2.rectangle(f,(330,200),(480,245),(220,110,20),-1)
        m=demo.osd_mask(f)
        self.assertGreater(cv2.countNonZero(m[115:126,50:580]),1000)
        self.assertEqual(cv2.countNonZero(m[180:231,180:236]),0)
        self.assertEqual(cv2.countNonZero(m[200:246,330:481]),0)

    def test_clean_source_is_unchanged(self):
        f=np.full((360,640,3),80,np.uint8)
        class Donors:
            def prepare(self,t):pass
            def get(self,t):raise AssertionError("No donor needed")
        out,mask,stats=demo.restore(f,1,Donors())
        np.testing.assert_array_equal(out,f)
        self.assertEqual(stats["masked"],0)

    def test_repair_cannot_modify_unmasked_pixels(self):
        f=np.full((360,640,3),80,np.uint8)
        cv2.rectangle(f,(30,120),(600,320),(0,0,240),3)
        class Donors:
            def prepare(self,t):pass
            def get(self,t):return None
        out,mask,stats=demo.restore(f,1,Donors())
        np.testing.assert_array_equal(out[mask==0],f[mask==0])
        self.assertGreater(stats["spatial"],0)
        self.assertLess(int(out[mask>0,2].max()),120)

    def test_keyframes_do_not_extrapolate_a_lost_plate(self):
        keys=[[32,100,200,180,60],[33,120,200,200,60]]
        self.assertIsNone(demo.at_keyframes(keys,31))
        self.assertIsNone(demo.at_keyframes(keys,34))
        self.assertEqual(demo.at_keyframes(keys,32.5),[110,200,190,60])


if __name__=="__main__":unittest.main()
