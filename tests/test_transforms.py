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
from typing import Any, ClassVar

import astropy.units as u
import numpy as np
import pydantic

from lsst.images import (
    ICRS,
    XY,
    Box,
    CameraFrameSet,
    CameraFrameSetSerializationModel,
    DetectorFrame,
    FocalPlaneFrame,
    GeneralFrame,
    SkyProjection,
    Transform,
    TransformSerializationModel,
)
from lsst.images._transforms import _ast as astshim
from lsst.images.fits import PointerModel
from lsst.images.serialization import ArchiveTree, InputArchive, JsonRef, OutputArchive
from lsst.images.tests import (
    DP2_VISIT_DETECTOR_DATA_ID,
    RoundtripFits,
    RoundtripJson,
    check_transform,
    compare_sky_projection_to_legacy_wcs,
    legacy_points_to_xy_array,
)

DATA_DIR = os.environ.get("TESTDATA_IMAGES_DIR", None)


class TransformTestCase(unittest.TestCase):
    """Tests for the Transform, SkyProjection, and FrameSet classes."""

    def test_identity(self) -> None:
        """Test an identity transform."""
        frame = DetectorFrame(**DP2_VISIT_DETECTOR_DATA_ID, bbox=Box.factory[:5, :4])
        xy = frame.bbox.meshgrid().map(np.ravel)
        identity = Transform.identity(frame)
        check_transform(self, identity, xy, xy, frame, frame)
        self.assertEqual(identity.decompose(), [])
        with RoundtripJson(self, identity) as roundtrip:
            pass
        check_transform(self, roundtrip.result, xy, xy, frame, frame)

    def test_transform_equality(self) -> None:
        """Test ``Transform.__eq__`` across all of its comparison branches."""
        pixel_frame = DetectorFrame(**DP2_VISIT_DETECTOR_DATA_ID, bbox=Box.factory[:5, :4])
        focal_plane = FocalPlaneFrame(instrument="LSSTCam", visit=1, unit=u.mm)
        # A distinct frame for the in-frame and out-frame branches.
        alt_frame = DetectorFrame(instrument="LSSTCam", visit=1, detector=12, bbox=Box.factory[:5, :4])
        in_bounds = Box.factory[:5, :4]
        out_bounds = Box.factory[:10, :8]

        def make(
            *,
            in_frame: Any = pixel_frame,
            out_frame: Any = focal_plane,
            ast_mapping: astshim.Mapping | None = None,
            in_bounds_: Box | None = in_bounds,
            out_bounds_: Box | None = out_bounds,
            components: Any = (),
        ) -> Transform[Any, Any]:
            return Transform(
                in_frame,
                out_frame,
                ast_mapping if ast_mapping is not None else astshim.UnitMap(2),
                in_bounds=in_bounds_,
                out_bounds=out_bounds_,
                components=components,
            )

        base = make()

        # Identity short-circuit: an object is always equal to itself.
        self.assertEqual(base, base)

        # Two independently constructed but equivalent transforms are equal,
        # and equality is symmetric.
        self.assertEqual(base, make())
        self.assertEqual(make(), base)

        # Comparison against a non-Transform yields NotImplemented, so Python
        # falls back to identity: the objects are unequal and ``!=`` is True.
        self.assertFalse(base == "not a transform")
        self.assertTrue(base != "not a transform")
        self.assertNotEqual(base, None)
        self.assertNotEqual(base, 42)

        # Each remaining branch differs from ``base`` in exactly one attribute.
        self.assertNotEqual(base, make(ast_mapping=astshim.ShiftMap([1.0, 2.0])))
        self.assertNotEqual(base, make(in_bounds_=out_bounds))
        self.assertNotEqual(base, make(out_bounds_=in_bounds))
        self.assertNotEqual(base, make(in_frame=alt_frame))
        self.assertNotEqual(base, make(out_frame=alt_frame))
        self.assertNotEqual(base, make(components=[Transform.identity(alt_frame)]))

    def test_sky_projection_equality(self) -> None:
        """Test ``SkyProjection.__eq__`` across all of its comparison
        branches.
        """
        pixel_frame = DetectorFrame(**DP2_VISIT_DETECTOR_DATA_ID, bbox=Box.factory[:5, :4])

        # Check the two failure modes.
        with self.assertRaises(ValueError):
            SkyProjection(Transform(ICRS, ICRS, astshim.UnitMap(2)))

        with self.assertRaises(ValueError):
            SkyProjection(Transform(pixel_frame, pixel_frame, astshim.UnitMap(2)))

        def make_pixel_to_sky(ast_mapping: astshim.Mapping | None = None) -> Transform[Any, Any]:
            mapping = ast_mapping if ast_mapping is not None else astshim.UnitMap(2)
            return Transform(pixel_frame, ICRS, mapping)

        base = SkyProjection(make_pixel_to_sky())

        # Identity short-circuit: an object is always equal to itself.
        self.assertEqual(base, base)

        # Two independently constructed but equivalent projections are equal.
        self.assertEqual(base, SkyProjection(make_pixel_to_sky()))

        # Comparison against a non-SkyProjection yields NotImplemented, so
        # Python falls back to identity.
        self.assertFalse(base == "not a projection")
        self.assertTrue(base != "not a projection")
        self.assertNotEqual(base, None)

        # Differ only in the pixel-to-sky transform.
        self.assertNotEqual(base, SkyProjection(make_pixel_to_sky(astshim.ShiftMap([1.0, 2.0]))))

        # The fits_approximation branch: absent on ``base`` but present here.
        with_approx = SkyProjection(
            make_pixel_to_sky(), fits_approximation=make_pixel_to_sky(astshim.ShiftMap([0.1, 0.2]))
        )
        self.assertNotEqual(base, with_approx)

        # Equal pixel-to-sky and equal fits_approximations are equal.
        with_approx_again = SkyProjection(
            make_pixel_to_sky(), fits_approximation=make_pixel_to_sky(astshim.ShiftMap([0.1, 0.2]))
        )
        self.assertEqual(with_approx, with_approx_again)

        # Same pixel-to-sky transform but a different fits_approximation.
        other_approx = SkyProjection(
            make_pixel_to_sky(), fits_approximation=make_pixel_to_sky(astshim.ShiftMap([0.3, 0.4]))
        )
        self.assertNotEqual(with_approx, other_approx)

    def test_affine_2x2(self) -> None:
        """Test an affine transform constructed from a 2x2 matrix."""
        in_frame = DetectorFrame(**DP2_VISIT_DETECTOR_DATA_ID, bbox=Box.factory[:5, :4])
        out_frame = GeneralFrame(unit=u.pix)
        transform_matrix = np.array([[2.0, 0.25], [-0.75, 0.8]])
        in_xy = in_frame.bbox.meshgrid().map(np.ravel)
        in_matrix = np.array([in_xy.x, in_xy.y])
        out_matrix = np.dot(transform_matrix, in_matrix)
        check_transform(
            self,
            Transform.affine(in_frame, out_frame, transform_matrix),
            in_xy,
            XY(x=out_matrix[0, :], y=out_matrix[1, :]),
            in_frame,
            out_frame,
            in_atol=1e-15 * u.pix,
            out_atol=1e-15 * u.pix,
        )

    def test_affine_3x3(self) -> None:
        """Test an affine transform constructed from a 3x3 matrix."""
        in_frame = DetectorFrame(**DP2_VISIT_DETECTOR_DATA_ID, bbox=Box.factory[:5, :4])
        out_frame = GeneralFrame(unit=u.pix)
        transform_matrix = np.array([[2.0, 0.25, -0.5], [-0.75, 0.8, 0.4], [0.0, 0.0, 1.0]])
        in_xy = in_frame.bbox.meshgrid().map(np.ravel)
        in_matrix = np.array([in_xy.x, in_xy.y, np.ones(in_xy.x.shape)])
        out_matrix = np.dot(transform_matrix, in_matrix)
        check_transform(
            self,
            Transform.affine(in_frame, out_frame, transform_matrix),
            in_xy,
            XY(x=out_matrix[0, :], y=out_matrix[1, :]),
            in_frame,
            out_frame,
            in_atol=1e-15 * u.pix,
            out_atol=1e-15 * u.pix,
        )

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
        with RoundtripFits(self, test_holder) as roundtrip1:
            self.assertEqual(len(roundtrip1.serialized.pixels_to_fp.frames), 2)
            self.assertEqual(len(roundtrip1.serialized.pixels_to_fp.bounds), 2)
            self.assertEqual(len(roundtrip1.serialized.pixels_to_fp.mappings), 1)
            # Instead of storing the AST mapping directly, we should have
            # stored a reference to the frame set:
            self.assertIsInstance(roundtrip1.serialized.pixels_to_fp.mappings[0], PointerModel)
        self.compare_to_legacy_camera(legacy_camera, roundtrip1.result.frames)
        self.assertEqual(roundtrip1.result.pixels_to_fp.in_frame, frame_set.detector(detector_id))
        self.assertEqual(roundtrip1.result.pixels_to_fp.out_frame, frame_set.focal_plane())
        self.assertEqual(
            roundtrip1.result.pixels_to_fp._ast_mapping.simplified().show(),
            test_holder.pixels_to_fp._ast_mapping.simplified().show(),
        )
        with RoundtripJson(self, test_holder) as roundtrip2:
            self.assertEqual(len(roundtrip2.serialized.pixels_to_fp.frames), 2)
            self.assertEqual(len(roundtrip2.serialized.pixels_to_fp.bounds), 2)
            self.assertEqual(len(roundtrip2.serialized.pixels_to_fp.mappings), 1)
            # Instead of storing the AST mapping directly, we should have
            # stored a reference to the frame set:
            self.assertIsInstance(roundtrip2.serialized.pixels_to_fp.mappings[0], JsonRef)
            raw_data = roundtrip2.inspect()
            self.assertEqual(len(raw_data["indirect"]), 1)
            self.assertEqual(raw_data["frames"], {"$ref": "#/indirect/0"})
        self.compare_to_legacy_camera(legacy_camera, roundtrip2.result.frames)
        self.assertEqual(roundtrip2.result.pixels_to_fp.in_frame, frame_set.detector(detector_id))
        self.assertEqual(roundtrip2.result.pixels_to_fp.out_frame, frame_set.focal_plane())
        self.assertEqual(
            roundtrip2.result.pixels_to_fp._ast_mapping.simplified().show(),
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
        """Test the Transform/SkyProjection representation of a detector
        WCS.
        """
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
        sky_projection = SkyProjection.from_legacy(legacy_wcs, detector_frame)
        assert sky_projection.fits_approximation is not None
        compare_sky_projection_to_legacy_wcs(self, sky_projection, legacy_wcs, detector_frame, subimage_bbox)
        # When we convert from a legacy SkyWcs, the internal AST Mapping needs
        # to really be an AST FrameSet in order to be able to convert back.
        self.assertIn("Begin FrameSet", sky_projection.show())
        compare_sky_projection_to_legacy_wcs(
            self, sky_projection, sky_projection.to_legacy(), detector_frame, subimage_bbox
        )
        self.assertIn("Begin FrameSet", sky_projection.fits_approximation.show())
        compare_sky_projection_to_legacy_wcs(
            self,
            sky_projection.fits_approximation,
            sky_projection.fits_approximation.to_legacy(),
            detector_frame,
            subimage_bbox,
            is_fits=True,
        )
        with RoundtripJson(self, sky_projection, "SkyProjection") as roundtrip:
            pass
        compare_sky_projection_to_legacy_wcs(
            self, roundtrip.result, legacy_wcs, detector_frame, subimage_bbox
        )
        # The AST FrameSet-ness needs to propagate through serialization.
        self.assertIn("Begin FrameSet", roundtrip.result.show())
        compare_sky_projection_to_legacy_wcs(
            self, sky_projection, roundtrip.result.to_legacy(), detector_frame, subimage_bbox
        )
        with RoundtripJson(self, sky_projection.fits_approximation, "SkyProjection") as roundtrip:
            pass
        compare_sky_projection_to_legacy_wcs(
            self,
            roundtrip.result,
            legacy_wcs.getFitsApproximation(),
            detector_frame,
            subimage_bbox,
            is_fits=True,
        )
        self.assertIn("Begin FrameSet", roundtrip.result.show())
        compare_sky_projection_to_legacy_wcs(
            self,
            sky_projection.fits_approximation,
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
    def _get_archive_tree_type[P: pydantic.BaseModel](
        pointer_type: type[P],
    ) -> type[FrameSetTestHolderModel[P]]:
        return FrameSetTestHolderModel[pointer_type]  # type: ignore


class FrameSetTestHolderModel[P: pydantic.BaseModel](ArchiveTree):
    """The serialization model for FrameSetTestHolder."""

    SCHEMA_NAME: ClassVar[str] = "_test_frame_set_holder"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
    PUBLIC_TYPE: ClassVar[type] = FrameSetTestHolder

    frames: CameraFrameSetSerializationModel | P
    pixels_to_fp: TransformSerializationModel[P]

    def deserialize(self, archive: InputArchive[Any]) -> FrameSetTestHolder:
        assert not isinstance(self.frames, CameraFrameSetSerializationModel), "Archive pointer expected."
        frames = archive.deserialize_pointer(
            self.frames, CameraFrameSetSerializationModel, CameraFrameSetSerializationModel.deserialize
        )
        pixels_to_fp = self.pixels_to_fp.deserialize(archive)
        return FrameSetTestHolder(frames, pixels_to_fp)


if __name__ == "__main__":
    unittest.main()
