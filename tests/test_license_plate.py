from __future__ import annotations

import unittest

from rtsp_annotator.license_plate import (
    PlateConsensus,
    normalize_chinese_plate,
)


class ChineseLicensePlateTests(unittest.TestCase):
    def test_normalizes_standard_and_new_energy_plates(self) -> None:
        self.assertEqual(normalize_chinese_plate("粤·B12345"), "粤B12345")
        self.assertEqual(normalize_chinese_plate("京a12345d"), "京A12345D")

    def test_rejects_non_chinese_and_ambiguous_letters(self) -> None:
        self.assertIsNone(normalize_chinese_plate("ABC1234"))
        self.assertIsNone(normalize_chinese_plate("粤I12345"))
        self.assertIsNone(normalize_chinese_plate("识别中"))

    def test_consensus_stabilizes_by_track_and_expires(self) -> None:
        consensus = PlateConsensus(
            minimum_confirmations=2,
            expire_after_frames=10,
        )
        first = consensus.observe(
            pad_index=0,
            track_id=9,
            frame_number=1,
            value="粤B12345",
        )
        second = consensus.observe(
            pad_index=0,
            track_id=9,
            frame_number=2,
            value="粤B12345",
        )
        expired = consensus.get(
            pad_index=0,
            track_id=9,
            frame_number=20,
        )
        self.assertIsNone(first)
        self.assertEqual(second, "粤B12345")
        self.assertIsNone(expired)


if __name__ == "__main__":
    unittest.main()
