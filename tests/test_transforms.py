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

import functools
import os
import tempfile
import unittest
from typing import Any

import numpy as np
import pydantic

from lsst.images import (
    Box,
    CameraFrameSet,
    CameraFrameSetSerializationModel,
    DetectorFrame,
    Projection,
    ProjectionSerializationModel,
    Transform,
    TransformSerializationModel,
)
from lsst.images.fits import FitsInputArchive, FitsOutputArchive
from lsst.images.serialization import TableCellReferenceModel
from lsst.images.tests import (
    DP2_VISIT_DETECTOR_DATA_ID,
    check_transform,
    compare_projection_to_legacy_wcs,
    legacy_points_to_xy_array,
)

DATA_DIR = os.environ.get("TESTDATA_IMAGES_DIR", None)


class TransformTestCase(unittest.TestCase):
    """Tests for the Transform, Projection, and FrameSet classes."""

    def test_identity(self) -> None:
        """Test an identity transform."""
        try:
            import astshim  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("astshim could not be imported.") from None
        frame = DetectorFrame(**DP2_VISIT_DETECTOR_DATA_ID, bbox=Box.factory[:5, :4])
        xy = frame.bbox.meshgrid().map(np.ravel)
        identity = Transform.identity(frame)
        check_transform(self, identity, xy, xy, frame, frame)
        self.assertEqual(identity.decompose(), [])

    @unittest.skipUnless(DATA_DIR is not None, "TESTDATA_IMAGES_DIR is not in the environment.")
    def test_camera(self) -> None:
        """Test that we can:

        - make a CameraFrameSet from the AST representation returned by afw;
        - transform points and get the same result as afw;
        - round-trip the CameraFrameSet through FITS serialization and still
          do all of that;
        - also roundtrip a Transform that can be obtained from the
          CameraFrameSet, by referencing the mappings in the frame set.

        This test is skipped if legacy modules cannot be imported.
        """
        try:
            from lsst.afw.cameraGeom import Camera
        except ImportError:
            raise unittest.SkipTest("'lsst.afw.cameraGeom' could not be imported.") from None
        assert DATA_DIR is not None, "Guaranteed by decorator."
        filename = os.path.join(DATA_DIR, "dp2", "legacy", "camera.fits")
        legacy_camera = Camera.readFits(filename)
        frame_set = CameraFrameSet.from_legacy(legacy_camera)
        detector_id: int = DP2_VISIT_DETECTOR_DATA_ID["detector"]
        self.compare_to_legacy_camera(legacy_camera, frame_set)
        with tempfile.NamedTemporaryFile(suffix=".fits", delete_on_close=False, delete=True) as tmp:
            tmp.close()
            with FitsOutputArchive.open(tmp.name) as output_archive:
                frame_set_ptr = output_archive.serialize_frame_set(
                    "frames", frame_set, frame_set.serialize, key=id(frame_set)
                )
                pixels_to_fp = frame_set[frame_set.detector(detector_id), frame_set.focal_plane()]
                pixels_to_fp_ser = output_archive.serialize_direct(
                    "pixels_to_fp", functools.partial(pixels_to_fp.serialize, use_frame_sets=True)
                )
                output_archive.add_tree(
                    CameraFramesTestModel(frames=frame_set_ptr, pixels_to_fp=pixels_to_fp_ser)
                )
            self.assertEqual(len(pixels_to_fp_ser.frames), 2)
            self.assertEqual(len(pixels_to_fp_ser.bounds), 2)
            self.assertEqual(len(pixels_to_fp_ser.mappings), 1)
            # Instead of storing the AST mapping directly, we should have
            # stored a reference to the frame set:
            self.assertIsInstance(pixels_to_fp_ser.mappings[0], TableCellReferenceModel)
            with FitsInputArchive.open(tmp.name) as input_archive:
                tree = input_archive.get_tree(CameraFramesTestModel[TableCellReferenceModel])
                rt_frame_set = input_archive.deserialize_pointer(
                    tree.frames, CameraFrameSetSerializationModel, CameraFrameSet.deserialize
                )
                rt_pixels_to_fp = Transform.deserialize(tree.pixels_to_fp, input_archive)
        self.compare_to_legacy_camera(legacy_camera, rt_frame_set)
        self.assertEqual(rt_pixels_to_fp.in_frame, frame_set.detector(detector_id))
        self.assertEqual(rt_pixels_to_fp.out_frame, frame_set.focal_plane())
        self.assertEqual(
            rt_pixels_to_fp._ast_mapping.simplified().show(), pixels_to_fp._ast_mapping.simplified().show()
        )

    def compare_to_legacy_camera(self, legacy_camera: Any, frame_set: CameraFrameSet) -> None:
        """Test the transforms extracted from a CameraFrameSet against the
        legacy lsst.afw.cameraGeom implementations.
        """
        from lsst.afw.cameraGeom import FIELD_ANGLE, FOCAL_PLANE, PIXELS
        from lsst.geom import Point2D

        legacy_detector = legacy_camera[16]
        pixel_legacy_points = [Point2D(50.0, 60.0), Point2D(801.2, 322.8), Point2D(33.5, 22.1)]
        fp_legacy_points = [legacy_detector.transform(p, PIXELS, FOCAL_PLANE) for p in pixel_legacy_points]
        fa_legacy_points = [legacy_detector.transform(p, PIXELS, FIELD_ANGLE) for p in pixel_legacy_points]
        pixel_xy_array = legacy_points_to_xy_array(pixel_legacy_points)
        fp_xy_array = legacy_points_to_xy_array(fp_legacy_points)
        fa_xy_array = legacy_points_to_xy_array(fa_legacy_points)
        # Test transforms extracted directly from the frame set.
        pixel_to_fp = frame_set[frame_set.detector(16), frame_set.focal_plane()]
        check_transform(
            self, pixel_to_fp, pixel_xy_array, fp_xy_array, frame_set.detector(16), frame_set.focal_plane()
        )
        pixel_to_fa = frame_set[frame_set.detector(16), frame_set.field_angle()]
        check_transform(
            self, pixel_to_fa, pixel_xy_array, fa_xy_array, frame_set.detector(16), frame_set.field_angle()
        )
        fp_to_fa = frame_set[frame_set.focal_plane(), frame_set.field_angle()]
        check_transform(
            self, fp_to_fa, fp_xy_array, fa_xy_array, frame_set.focal_plane(), frame_set.field_angle()
        )
        # Test a composition.
        pixel_to_fa_indirect = pixel_to_fp.then(fp_to_fa)
        check_transform(
            self,
            pixel_to_fa_indirect,
            pixel_xy_array,
            fa_xy_array,
            frame_set.detector(16),
            frame_set.field_angle(),
        )
        pixel_to_fp_d, fp_to_fa_d = pixel_to_fa_indirect.decompose()
        check_transform(
            self, pixel_to_fp_d, pixel_xy_array, fp_xy_array, frame_set.detector(16), frame_set.focal_plane()
        )
        check_transform(
            self, fp_to_fa_d, fp_xy_array, fa_xy_array, frame_set.focal_plane(), frame_set.field_angle()
        )
        fa_to_fp_d, fp_to_pixel_d = pixel_to_fa_indirect.inverted().decompose()
        check_transform(
            self, fa_to_fp_d, fa_xy_array, fp_xy_array, frame_set.field_angle(), frame_set.focal_plane()
        )
        check_transform(
            self, fp_to_pixel_d, fp_xy_array, pixel_xy_array, frame_set.focal_plane(), frame_set.detector(16)
        )

    def test_detector_wcs(self) -> None:
        """Test the Transform/Projection representation of a detector WCS."""
        try:
            from lsst.afw.image import ExposureFitsReader
        except ImportError:
            raise unittest.SkipTest("'lsst.afw.image' could not be imported.") from None
        assert DATA_DIR is not None, "Guaranteed by decorator."
        filename = os.path.join(DATA_DIR, "dp2", "legacy", "visit_image.fits")
        reader = ExposureFitsReader(filename)
        legacy_wcs = reader.readWcs()
        wcs_bbox = Box.from_legacy(reader.readDetector().getBBox())
        subimage_bbox = Box.from_legacy(reader.readBBox())
        detector_frame = DetectorFrame(**DP2_VISIT_DETECTOR_DATA_ID, bbox=wcs_bbox)
        projection = Projection.from_legacy(legacy_wcs, detector_frame)
        compare_projection_to_legacy_wcs(self, projection, legacy_wcs, detector_frame, subimage_bbox)
        with tempfile.NamedTemporaryFile(suffix=".fits", delete_on_close=False, delete=True) as tmp:
            tmp.close()
            with FitsOutputArchive.open(tmp.name) as output_archive:
                tree = output_archive.serialize_direct("projection", projection.serialize)
                output_archive.add_tree(tree)
            with FitsInputArchive.open(tmp.name) as input_archive:
                tree = input_archive.get_tree(ProjectionSerializationModel)
                roundtripped = Projection.deserialize(tree, input_archive)
        compare_projection_to_legacy_wcs(self, roundtripped, legacy_wcs, detector_frame, subimage_bbox)


class CameraFramesTestModel[P: pydantic.BaseModel](pydantic.BaseModel):
    """A testing serialization model that holds both a CamaraFrameSet and
    a Transform extracted from it.
    """

    frames: P
    pixels_to_fp: TransformSerializationModel[P]


if __name__ == "__main__":
    unittest.main()
