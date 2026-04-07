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

import dataclasses
import functools
import os
import unittest
from typing import Any

import numpy as np
import pydantic

from lsst.images import (
    Box,
    CameraFrameSet,
    CameraFrameSetSerializationModel,
    DetectorFrame,
    FocalPlaneFrame,
    Projection,
    Transform,
    TransformSerializationModel,
)
from lsst.images.serialization import ArchiveTree, InputArchive, OutputArchive, TableCellReferenceModel
from lsst.images.tests import (
    DP2_VISIT_DETECTOR_DATA_ID,
    RoundtripFits,
    check_transform,
    compare_projection_to_legacy_wcs,
    legacy_points_to_xy_array,
)

DATA_DIR = os.environ.get("TESTDATA_IMAGES_DIR", None)


class TransformTestCase(unittest.TestCase):
    """Tests for the Transform, Projection, and FrameSet classes."""

    def test_identity(self) -> None:
        """Test an identity transform."""
        frame = DetectorFrame(**DP2_VISIT_DETECTOR_DATA_ID, bbox=Box.factory[:5, :4])
        xy = frame.bbox.meshgrid().map(np.ravel)
        identity = Transform.identity(frame)
        check_transform(self, identity, xy, xy, frame, frame)
        self.assertEqual(identity.decompose(), [])
        with RoundtripFits(self, identity) as roundtrip:
            pass
        check_transform(self, roundtrip.result, xy, xy, frame, frame)

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

        This test provides coverage for the archive system's pointer and
        frame-set reference machinery.
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
        test_holder = FrameSetTestHolder(
            frames=frame_set,
            pixels_to_fp=frame_set[frame_set.detector(detector_id), frame_set.focal_plane()],
        )
        with RoundtripFits(self, test_holder) as roundtrip:
            self.assertEqual(len(roundtrip.serialized.pixels_to_fp.frames), 2)
            self.assertEqual(len(roundtrip.serialized.pixels_to_fp.bounds), 2)
            self.assertEqual(len(roundtrip.serialized.pixels_to_fp.mappings), 1)
            # Instead of storing the AST mapping directly, we should have
            # stored a reference to the frame set:
            self.assertIsInstance(roundtrip.serialized.pixels_to_fp.mappings[0], TableCellReferenceModel)
        self.compare_to_legacy_camera(legacy_camera, roundtrip.result.frames)
        self.assertEqual(roundtrip.result.pixels_to_fp.in_frame, frame_set.detector(detector_id))
        self.assertEqual(roundtrip.result.pixels_to_fp.out_frame, frame_set.focal_plane())
        self.assertEqual(
            roundtrip.result.pixels_to_fp._ast_mapping.simplified().show(),
            test_holder.pixels_to_fp._ast_mapping.simplified().show(),
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

    @unittest.skipUnless(DATA_DIR is not None, "TESTDATA_IMAGES_DIR is not in the environment.")
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
        assert projection.fits_approximation is not None
        compare_projection_to_legacy_wcs(self, projection, legacy_wcs, detector_frame, subimage_bbox)
        # When we convert from a legacy SkyWcs, the internal AST Mapping needs
        # to really be an AST FrameSet in order to be able to convert back.
        self.assertIn("Begin FrameSet", projection.show())
        compare_projection_to_legacy_wcs(
            self, projection, projection.to_legacy(), detector_frame, subimage_bbox
        )
        self.assertIn("Begin FrameSet", projection.fits_approximation.show())
        compare_projection_to_legacy_wcs(
            self,
            projection.fits_approximation,
            projection.fits_approximation.to_legacy(),
            detector_frame,
            subimage_bbox,
            is_fits=True,
        )
        with RoundtripFits(self, projection, "Projection") as roundtrip:
            pass
        compare_projection_to_legacy_wcs(self, roundtrip.result, legacy_wcs, detector_frame, subimage_bbox)
        # The AST FrameSet-ness needs to propagate through serialization.
        self.assertIn("Begin FrameSet", roundtrip.result.show())
        compare_projection_to_legacy_wcs(
            self, projection, roundtrip.result.to_legacy(), detector_frame, subimage_bbox
        )
        with RoundtripFits(self, projection.fits_approximation, "Projection") as roundtrip:
            pass
        compare_projection_to_legacy_wcs(
            self,
            roundtrip.result,
            legacy_wcs.getFitsApproximation(),
            detector_frame,
            subimage_bbox,
            is_fits=True,
        )
        self.assertIn("Begin FrameSet", roundtrip.result.show())
        compare_projection_to_legacy_wcs(
            self,
            projection.fits_approximation,
            roundtrip.result.to_legacy(),
            detector_frame,
            subimage_bbox,
            is_fits=True,
        )


@dataclasses.dataclass
class FrameSetTestHolder:
    """A top-level object that holds a CameraFrameSet and a transform
    extracted from it, for testing archive pointers and frame set references.
    """

    frames: CameraFrameSet
    pixels_to_fp: Transform[DetectorFrame, FocalPlaneFrame]

    def serialize[P: pydantic.BaseModel](self, archive: OutputArchive[P]) -> FrameSetTestHolderModel[P]:
        frames_model = archive.serialize_frame_set(
            "frames", self.frames, self.frames.serialize, key=id(self.frames)
        )
        pixels_to_fp_model = archive.serialize_direct(
            "pixels_to_fp", functools.partial(self.pixels_to_fp.serialize, use_frame_sets=True)
        )
        return FrameSetTestHolderModel[P](frames=frames_model, pixels_to_fp=pixels_to_fp_model)

    @staticmethod
    def deserialize(model: FrameSetTestHolderModel[Any], archive: InputArchive[Any]) -> FrameSetTestHolder:
        assert not isinstance(model.frames, CameraFrameSetSerializationModel), "Archive pointer expected."
        frames = archive.deserialize_pointer(
            model.frames, CameraFrameSetSerializationModel, CameraFrameSet.deserialize
        )
        pixels_to_fp = Transform.deserialize(model.pixels_to_fp, archive)
        return FrameSetTestHolder(frames, pixels_to_fp)

    @staticmethod
    def _get_archive_tree_type[P: pydantic.BaseModel](
        pointer_type: type[P],
    ) -> type[FrameSetTestHolderModel[P]]:
        return FrameSetTestHolderModel[pointer_type]  # type: ignore


class FrameSetTestHolderModel[P: pydantic.BaseModel](ArchiveTree):
    """The serialization model for FrameSetTestHolder."""

    frames: CameraFrameSetSerializationModel | P
    pixels_to_fp: TransformSerializationModel[P]


if __name__ == "__main__":
    unittest.main()
