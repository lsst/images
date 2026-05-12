# This file is part of lsst-images.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# Use of this source code is governed by a 3-clause BSD-style
# license that can be found in the LICENSE file.

from __future__ import annotations

import os
import unittest

from lsst.images.cameras import AmplifierRawGeometry, Detector
from lsst.images.tests import DP2_VISIT_DETECTOR_DATA_ID, RoundtripFits, compare_detector_to_legacy

try:
    from lsst.afw.cameraGeom import Camera as LegacyCamera
    from lsst.afw.cameraGeom import DetectorType as LegacyDetectorType
    from lsst.afw.image import ExposureFitsReader
except ImportError:  # pragma: no cover
    HAVE_AFW = False
else:  # pragma: no cover
    HAVE_AFW = True

DATA_DIR = os.environ.get("TESTDATA_IMAGES_DIR", None)


@unittest.skipUnless(DATA_DIR is not None, "TESTDATA_IMAGES_DIR is not in the environment.")
@unittest.skipUnless(HAVE_AFW, "lsst.afw could not be imported.")
class CamerasTestCase(unittest.TestCase):
    """Tests for the 'cameras' module."""

    @classmethod
    def setUpClass(cls):
        cls.legacy_camera = LegacyCamera.readFits(os.path.join(DATA_DIR, "dp2", "legacy", "camera.fits"))
        cls.legacy_detector = ExposureFitsReader(
            os.path.join(DATA_DIR, "dp2", "legacy", "visit_image.fits")
        ).readDetector()

    def test_visit_image_detector_legacy_conversions(self) -> None:
        """Test converting a detector attached to a visit image from its
        legacy type and back, with serialization in between.
        """
        detector = Detector.from_legacy(
            self.legacy_detector,
            instrument=DP2_VISIT_DETECTOR_DATA_ID["instrument"],
            visit=DP2_VISIT_DETECTOR_DATA_ID["visit"],
        )
        compare_detector_to_legacy(self, detector, self.legacy_detector, is_raw_assembled=True)
        with RoundtripFits(self, detector) as roundtrip:
            pass
        compare_detector_to_legacy(self, roundtrip.result, self.legacy_detector, is_raw_assembled=True)
        compare_detector_to_legacy(self, detector, roundtrip.result.to_legacy(), is_raw_assembled=True)

    def test_camera_detector_legacy_conversions(self) -> None:
        """Test converting a detector extracted from a camera from its
        legacy type and back, with serialization in between.
        """
        # Test one detector of each type for a reasonable balance of
        # completeness and test speed.
        detector_types_seen: set[LegacyDetectorType] = set()
        for legacy_detector_1 in self.legacy_camera:
            if legacy_detector_1.getType() in detector_types_seen:
                continue
            detector_types_seen.add(legacy_detector_1.getType())
            with self.subTest(detector_id=legacy_detector_1.getId()):
                detector = Detector.from_legacy(
                    legacy_detector_1,
                    instrument=DP2_VISIT_DETECTOR_DATA_ID["instrument"],
                )
                compare_detector_to_legacy(self, detector, legacy_detector_1, is_raw_assembled=False)
                with RoundtripFits(self, detector) as roundtrip:
                    pass
                compare_detector_to_legacy(self, roundtrip.result, legacy_detector_1, is_raw_assembled=False)
                legacy_detector_2 = roundtrip.result.to_legacy()
                compare_detector_to_legacy(self, detector, legacy_detector_2, is_raw_assembled=False)

    def test_expanded_detector_roundtrip(self) -> None:
        """Test roudntripping a detector that holds both assembled and
        unassembled raw amplifier geometry.
        """
        legacy_camera_detector = self.legacy_camera[DP2_VISIT_DETECTOR_DATA_ID["detector"]]
        detector = Detector.from_legacy(
            legacy_camera_detector,
            instrument=DP2_VISIT_DETECTOR_DATA_ID["instrument"],
            visit=DP2_VISIT_DETECTOR_DATA_ID["visit"],
            is_raw_assembled=False,
        )
        for amplifier, legacy_assembled_amplifier in zip(
            detector.amplifiers, self.legacy_detector.getAmplifiers(), strict=True
        ):
            assert amplifier.unassembled_raw_geometry is not None
            assert amplifier.assembled_raw_geometry is None
            amplifier.assembled_raw_geometry = AmplifierRawGeometry.from_legacy_amplifier(
                legacy_assembled_amplifier
            )
        with RoundtripFits(self, detector) as roundtrip:
            pass
        compare_detector_to_legacy(self, roundtrip.result, legacy_camera_detector, is_raw_assembled=False)
        compare_detector_to_legacy(self, roundtrip.result, self.legacy_detector, is_raw_assembled=True)


if __name__ == "__main__":
    unittest.main()
