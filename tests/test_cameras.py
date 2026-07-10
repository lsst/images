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
from typing import Any

import pytest

from lsst.images import YX
from lsst.images.cameras import AmplifierRawGeometry, Detector, ReadoutCorner
from lsst.images.tests import DP2_VISIT_DETECTOR_DATA_ID, RoundtripFits, compare_detector_to_legacy

EXTERNAL_DATA_DIR = os.environ.get("TESTDATA_IMAGES_DIR", None)


@pytest.fixture(scope="session")
def legacy_camera_data() -> dict[str, Any]:
    """Return a dict with legacy_camera and legacy_detector loaded from the
    test data directory.
    """
    if EXTERNAL_DATA_DIR is None:
        pytest.skip("TESTDATA_IMAGES_DIR is not in the environment.")
    try:
        from lsst.afw.cameraGeom import Camera as LegacyCamera
        from lsst.afw.image import ExposureFitsReader
    except ImportError:
        pytest.skip("lsst.afw could not be imported.")
    legacy_camera = LegacyCamera.readFits(os.path.join(EXTERNAL_DATA_DIR, "dp2", "legacy", "camera.fits"))
    legacy_detector = ExposureFitsReader(
        os.path.join(EXTERNAL_DATA_DIR, "dp2", "legacy", "visit_image.fits")
    ).readDetector()
    return {"legacy_camera": legacy_camera, "legacy_detector": legacy_detector}


def test_visit_image_detector_legacy_conversions(legacy_camera_data: dict[str, Any]) -> None:
    """Test converting a visit-image detector from legacy and back with
    serialization in between.
    """
    legacy_detector = legacy_camera_data["legacy_detector"]
    detector = Detector.from_legacy(
        legacy_detector,
        instrument=DP2_VISIT_DETECTOR_DATA_ID["instrument"],
        visit=DP2_VISIT_DETECTOR_DATA_ID["visit"],
    )
    compare_detector_to_legacy(detector, legacy_detector, is_raw_assembled=True)
    with RoundtripFits(detector) as roundtrip:
        pass
    compare_detector_to_legacy(roundtrip.result, legacy_detector, is_raw_assembled=True)
    compare_detector_to_legacy(detector, roundtrip.result.to_legacy(), is_raw_assembled=True)


def test_camera_detector_legacy_conversions(legacy_camera_data: dict[str, Any]) -> None:
    """Test converting one detector of each type from a legacy Camera and
    back.
    """
    legacy_camera = legacy_camera_data["legacy_camera"]
    detector_types_seen = set()
    for legacy_detector_1 in legacy_camera:
        if legacy_detector_1.getType() in detector_types_seen:
            continue
        detector_types_seen.add(legacy_detector_1.getType())
        detector = Detector.from_legacy(
            legacy_detector_1,
            instrument=DP2_VISIT_DETECTOR_DATA_ID["instrument"],
        )
        compare_detector_to_legacy(detector, legacy_detector_1, is_raw_assembled=False)
        with RoundtripFits(detector) as roundtrip:
            pass
        compare_detector_to_legacy(roundtrip.result, legacy_detector_1, is_raw_assembled=False)
        legacy_detector_2 = roundtrip.result.to_legacy()
        compare_detector_to_legacy(detector, legacy_detector_2, is_raw_assembled=False)


def test_expanded_detector_roundtrip(legacy_camera_data: dict[str, Any]) -> None:
    """Test round-tripping a detector that holds both assembled and
    unassembled raw amplifier geometry.
    """
    legacy_camera = legacy_camera_data["legacy_camera"]
    legacy_detector = legacy_camera_data["legacy_detector"]
    legacy_camera_detector = legacy_camera[DP2_VISIT_DETECTOR_DATA_ID["detector"]]
    detector = Detector.from_legacy(
        legacy_camera_detector,
        instrument=DP2_VISIT_DETECTOR_DATA_ID["instrument"],
        visit=DP2_VISIT_DETECTOR_DATA_ID["visit"],
        is_raw_assembled=False,
    )
    for amplifier, legacy_assembled_amplifier in zip(
        detector.amplifiers, legacy_detector.getAmplifiers(), strict=True
    ):
        assert amplifier.unassembled_raw_geometry is not None
        assert amplifier.assembled_raw_geometry is None
        amplifier.assembled_raw_geometry = AmplifierRawGeometry.from_legacy_amplifier(
            legacy_assembled_amplifier
        )
    with RoundtripFits(detector) as roundtrip:
        pass
    compare_detector_to_legacy(roundtrip.result, legacy_camera_detector, is_raw_assembled=False)
    compare_detector_to_legacy(roundtrip.result, legacy_detector, is_raw_assembled=True)


def test_as_flips() -> None:
    """Tes that ReadoutCorner.as_flips returns the correct YX flip flags for
    each corner.
    """
    assert ReadoutCorner.LL.as_flips() == YX(y=False, x=False)
    assert ReadoutCorner.LR.as_flips() == YX(y=False, x=True)
    assert ReadoutCorner.UL.as_flips() == YX(y=True, x=False)
    assert ReadoutCorner.UR.as_flips() == YX(y=True, x=True)


def test_flips_roundtrip() -> None:
    """Test that ReadoutCorner.from_flips is the inverse of as_flips for all
    corners.
    """
    for corner in ReadoutCorner:
        flips = corner.as_flips()
        assert ReadoutCorner.from_flips(y=flips.y, x=flips.x) is corner


def test_apply_flips() -> None:
    """Test that ReadoutCorner.apply_flips with identity returns the same
    corner, and a few specific flips are correct.
    """
    for corner in ReadoutCorner:
        assert corner.apply_flips(y=False, x=False) is corner
    assert ReadoutCorner.LL.apply_flips(y=True, x=True) is ReadoutCorner.UR
    assert ReadoutCorner.LR.apply_flips(y=False, x=True) is ReadoutCorner.LL
    assert ReadoutCorner.UL.apply_flips(y=True, x=False) is ReadoutCorner.LL
    assert ReadoutCorner.UR.apply_flips(y=True, x=True) is ReadoutCorner.LL
