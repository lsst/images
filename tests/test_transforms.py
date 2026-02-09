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

import astropy.units as u
import astropy.wcs.wcsapi
import numpy as np
import pydantic
from astropy.coordinates import SkyCoord

from lsst.images import (
    XY,
    Box,
    CameraFrameSet,
    CameraFrameSetSerializationModel,
    DetectorFrame,
    Projection,
    ProjectionSerializationModel,
    SerializableFrame,
    SkyFrame,
    Transform,
    TransformSerializationModel,
)
from lsst.images.fits import FitsInputArchive, FitsOutputArchive
from lsst.images.serialization import TableCellReferenceModel
from lsst.images.tests.data_ids import DP2_VISIT_DETECTOR_DATA_ID

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
        self.check_transform(identity, xy, xy, frame, frame)
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
        pixel_xy_array = self._legacy_points_to_xy_array(pixel_legacy_points)
        fp_xy_array = self._legacy_points_to_xy_array(fp_legacy_points)
        fa_xy_array = self._legacy_points_to_xy_array(fa_legacy_points)
        # Test transforms extracted directly from the frame set.
        pixel_to_fp = frame_set[frame_set.detector(16), frame_set.focal_plane()]
        self.check_transform(
            pixel_to_fp, pixel_xy_array, fp_xy_array, frame_set.detector(16), frame_set.focal_plane()
        )
        pixel_to_fa = frame_set[frame_set.detector(16), frame_set.field_angle()]
        self.check_transform(
            pixel_to_fa, pixel_xy_array, fa_xy_array, frame_set.detector(16), frame_set.field_angle()
        )
        fp_to_fa = frame_set[frame_set.focal_plane(), frame_set.field_angle()]
        self.check_transform(
            fp_to_fa, fp_xy_array, fa_xy_array, frame_set.focal_plane(), frame_set.field_angle()
        )
        # Test a composition.
        pixel_to_fa_indirect = pixel_to_fp.then(fp_to_fa)
        self.check_transform(
            pixel_to_fa_indirect, pixel_xy_array, fa_xy_array, frame_set.detector(16), frame_set.field_angle()
        )
        pixel_to_fp_d, fp_to_fa_d = pixel_to_fa_indirect.decompose()
        self.check_transform(
            pixel_to_fp_d, pixel_xy_array, fp_xy_array, frame_set.detector(16), frame_set.focal_plane()
        )
        self.check_transform(
            fp_to_fa_d, fp_xy_array, fa_xy_array, frame_set.focal_plane(), frame_set.field_angle()
        )
        fa_to_fp_d, fp_to_pixel_d = pixel_to_fa_indirect.inverted().decompose()
        self.check_transform(
            fa_to_fp_d, fa_xy_array, fp_xy_array, frame_set.field_angle(), frame_set.focal_plane()
        )
        self.check_transform(
            fp_to_pixel_d, fp_xy_array, pixel_xy_array, frame_set.focal_plane(), frame_set.detector(16)
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
        self.compare_to_legacy_wcs(legacy_wcs, projection, detector_frame, subimage_bbox)
        with tempfile.NamedTemporaryFile(suffix=".fits", delete_on_close=False, delete=True) as tmp:
            tmp.close()
            with FitsOutputArchive.open(tmp.name) as output_archive:
                tree = output_archive.serialize_direct("projection", projection.serialize)
                output_archive.add_tree(tree)
            with FitsInputArchive.open(tmp.name) as input_archive:
                tree = input_archive.get_tree(ProjectionSerializationModel)
                roundtripped = Projection.deserialize(tree, input_archive)
        self.compare_to_legacy_wcs(legacy_wcs, roundtripped, detector_frame, subimage_bbox)

    def compare_to_legacy_wcs[F: SerializableFrame](
        self,
        legacy_wcs: Any,
        projection: Projection[F],
        pixel_frame: F,
        subimage_bbox: Box,
        is_fits: bool = False,
    ) -> None:
        """Test a Projection object by comparing it to an equivalent
        `lsst.afw.geom.SkyWcs`.
        """
        # Pixel coordinates to test on over the subimage region of interest:
        pixel_xy = subimage_bbox.meshgrid(step=50).map(np.ravel)
        # Array indices of those pixel values (subtract off bbox starts):
        subimage_array_xy = XY(x=pixel_xy.x - subimage_bbox.x.start, y=pixel_xy.y - subimage_bbox.y.start)
        sky_coords = self._legacy_coords_to_astropy(
            legacy_wcs.pixelToSky(self._arrays_to_legacy_points(pixel_xy.x, pixel_xy.y))
        )
        # Test transforming with the Projection itself, which also tests its
        # nested Transform and an Astropy High-Level WCS view with no origin
        # change.
        self.check_projection(projection, pixel_xy, sky_coords, pixel_frame)
        # Also test the Astropy High-Level WCS view with an origin change to
        # array indices.
        self.check_astropy_wcs_interface(
            projection.as_astropy(subimage_bbox), subimage_array_xy, sky_coords, pixel_atol=1e-5
        )
        if is_fits:
            fits_wcs = projection.as_fits_wcs(subimage_bbox, allow_approximation=True)
            assert fits_wcs is not None
            self.check_astropy_wcs_interface(fits_wcs, subimage_array_xy, sky_coords, pixel_atol=1e-5)
            # Use that FITS approximation to check that we can make a
            # Projection from a FITS WCs, too.
            fits_projection = Projection.from_fits_wcs(fits_wcs, pixel_frame)
            self.check_projection(
                fits_projection,
                subimage_array_xy,
                sky_coords,
                pixel_frame,
                pixel_atol=1e-5 * u.pix,
            )
        else:
            self.assertIsNone(projection.as_fits_wcs(subimage_bbox, allow_approximation=False))
            # The legacy SkyWcs should instead have a FITS approximation
            # attached; run the same tests on that.
            assert projection.fits_approximation is not None
            self.compare_to_legacy_wcs(
                legacy_wcs.getFitsApproximation(),
                projection.fits_approximation,
                pixel_frame,
                subimage_bbox,
                is_fits=True,
            )

    def check_transform[I: SerializableFrame, O: SerializableFrame](
        self,
        transform: Transform[I, O],
        input_xy: XY[np.ndarray],
        output_xy: XY[np.ndarray],
        in_frame: SerializableFrame,
        out_frame: SerializableFrame,
        *,
        check_inverted: bool = True,
        in_atol: u.Quantity | None = None,
        out_atol: u.Quantity | None = None,
    ) -> None:
        """Check the frames and various apply_ overloads of Transform."""
        self.assertEqual(transform.in_frame, in_frame)
        self.assertEqual(transform.out_frame, out_frame)
        in_atol_v = in_atol.to_value(in_frame.unit) if in_atol is not None else None
        out_atol_v = out_atol.to_value(out_frame.unit) if out_atol is not None else None
        # Test array interfaces.
        test_output_xy = transform.apply_forward(x=input_xy.x, y=input_xy.y)
        self.assert_close(test_output_xy.x, output_xy.x, atol=out_atol_v)
        self.assert_close(test_output_xy.y, output_xy.y, atol=out_atol_v)
        test_input_xy = transform.apply_inverse(x=output_xy.x, y=output_xy.y)
        self.assert_close(test_input_xy.x, input_xy.x, atol=in_atol_v)
        self.assert_close(test_input_xy.y, input_xy.y, atol=in_atol_v)
        # Test scalar interfaces with numpy scalars.
        for input_x, input_y, output_x, output_y in zip(input_xy.x, input_xy.y, output_xy.x, output_xy.y):
            self.assert_close(transform.apply_forward(x=input_x, y=input_y).x, output_x, atol=out_atol_v)
            self.assert_close(transform.apply_forward(x=input_x, y=input_y).y, output_y, atol=out_atol_v)
            self.assert_close(transform.apply_inverse(x=output_x, y=output_y).x, input_x, atol=in_atol_v)
            self.assert_close(transform.apply_inverse(x=output_x, y=output_y).y, input_y, atol=in_atol_v)
        # Test quantity array interfaces.
        input_q_xy = XY(x=input_xy.x * transform.in_frame.unit, y=input_xy.y * transform.in_frame.unit)
        output_q_xy = XY(x=output_xy.x * transform.out_frame.unit, y=output_xy.y * transform.out_frame.unit)
        test_output_q_xy = transform.apply_forward_q(x=input_q_xy.x, y=input_q_xy.y)
        self.assert_close(test_output_q_xy.x, output_q_xy.x, atol=out_atol)
        self.assert_close(test_output_q_xy.y, output_q_xy.y, atol=out_atol)
        test_input_q_xy = transform.apply_inverse_q(x=output_q_xy.x, y=output_q_xy.y)
        self.assert_close(test_input_q_xy.x, input_q_xy.x, atol=in_atol)
        self.assert_close(test_input_q_xy.y, input_q_xy.y, atol=in_atol)
        # Test quantity scalar interfaces.
        for input_q_x, input_q_y, output_q_x, output_q_y in zip(
            input_q_xy.x, input_q_xy.y, output_q_xy.x, output_q_xy.y
        ):
            self.assert_close(
                transform.apply_forward_q(x=input_q_x, y=input_q_y).x, output_q_x, atol=out_atol
            )
            self.assert_close(
                transform.apply_forward_q(x=input_q_x, y=input_q_y).y, output_q_y, atol=out_atol
            )
            self.assert_close(
                transform.apply_inverse_q(x=output_q_x, y=output_q_y).x, input_q_x, atol=in_atol
            )
            self.assert_close(
                transform.apply_inverse_q(x=output_q_x, y=output_q_y).y, input_q_y, atol=in_atol
            )
        if check_inverted:
            # Test the inverse transform.
            self.check_transform(
                transform.inverted(),
                output_xy,
                input_xy,
                out_frame,
                in_frame,
                check_inverted=False,
                out_atol=in_atol,
                in_atol=out_atol,
            )

    def check_projection[P: SerializableFrame](
        self,
        projection: Projection[P],
        pixel_xy: XY[np.ndarray],
        sky_coords: SkyCoord,
        pixel_frame: SerializableFrame,
        *,
        pixel_atol: u.Quantity | None = None,
        sky_atol: u.Quantity | None = None,
    ) -> None:
        self.assertEqual(projection.pixel_frame, pixel_frame)
        self.assertEqual(projection.sky_frame, SkyFrame.ICRS)
        pixel_atol_v = pixel_atol.to_value(pixel_frame.unit) if pixel_atol is not None else None
        sky_atol_v = sky_atol.to_value(SkyFrame.ICRS.unit) if sky_atol is not None else None
        # Test array interfaces.
        test_pixel_xy = projection.sky_to_pixel(sky_coords)
        self.assert_close(test_pixel_xy.x, pixel_xy.x, atol=pixel_atol_v)
        self.assert_close(test_pixel_xy.y, pixel_xy.y, atol=pixel_atol_v)
        test_sky_astropy = projection.pixel_to_sky(x=pixel_xy.x, y=pixel_xy.y)
        self.assert_close(test_sky_astropy.ra, sky_coords.ra, atol=sky_atol_v)
        self.assert_close(test_sky_astropy.dec, sky_coords.dec, atol=sky_atol_v)
        # Test scalar interfaces.
        for pixel_x, pixel_y, sky_single in zip(pixel_xy.x, pixel_xy.y, sky_coords):
            self.assert_close(projection.sky_to_pixel(sky_single).x, pixel_x, atol=pixel_atol_v)
            self.assert_close(projection.sky_to_pixel(sky_single).y, pixel_y, atol=pixel_atol_v)
            self.assert_close(
                projection.pixel_to_sky(x=pixel_x, y=pixel_y).ra, sky_single.ra, atol=sky_atol_v
            )
            self.assert_close(
                projection.pixel_to_sky(x=pixel_x, y=pixel_y).dec, sky_single.dec, atol=sky_atol_v
            )
        # Test the underlying Transform object.
        sky_xy = XY(x=sky_coords.ra.to_value(u.rad), y=sky_coords.dec.to_value(u.rad))
        self.check_transform(
            projection.pixel_to_sky_transform,
            pixel_xy,
            sky_xy,
            pixel_frame,
            SkyFrame.ICRS,
            check_inverted=False,
            in_atol=pixel_atol,
            out_atol=sky_atol,
        )
        self.check_transform(
            projection.sky_to_pixel_transform,
            sky_xy,
            pixel_xy,
            SkyFrame.ICRS,
            pixel_frame,
            check_inverted=False,
            in_atol=sky_atol,
            out_atol=pixel_atol,
        )
        # Test the Astropy interface adapter.
        self.check_astropy_wcs_interface(
            projection.as_astropy(), pixel_xy, sky_coords, pixel_atol=pixel_atol_v, sky_atol=sky_atol
        )

    def check_astropy_wcs_interface(
        self,
        wcs: astropy.wcs.wcsapi.BaseHighLevelWCS,
        pixel_xy: XY[np.ndarray],
        sky_astropy: SkyCoord,
        bbox: Box | None = None,
        *,
        pixel_atol: float | None = None,
        sky_atol: u.Quantity | None = None,
    ) -> None:
        test_x, test_y = wcs.world_to_pixel(sky_astropy)
        self.assert_close(test_x, pixel_xy.x, atol=pixel_atol)
        self.assert_close(test_y, pixel_xy.y, atol=pixel_atol)
        test_sky_astropy = wcs.pixel_to_world(pixel_xy.x, pixel_xy.y)
        self.assert_close(test_sky_astropy.ra, sky_astropy.ra, atol=sky_atol)
        self.assert_close(test_sky_astropy.dec, sky_astropy.dec, atol=sky_atol)

    def assert_close(self, a: np.ndarray | u.Quantity, b: np.ndarray | u.Quantity, **kwargs: Any) -> None:
        self.assertTrue(u.allclose(a, b, **kwargs), msg=f"{a} != {b}")

    @staticmethod
    def _legacy_points_to_xy_array(legacy_points: list[Any]) -> XY[np.ndarray]:
        return XY(x=np.array([p.x for p in legacy_points]), y=np.array([p.y for p in legacy_points]))

    @staticmethod
    def _legacy_coords_to_astropy(legacy_coords: list[Any]) -> astropy.coordinates.SkyCoord:
        return SkyCoord(
            ra=np.array([p.getRa().asRadians() for p in legacy_coords]) * u.rad,
            dec=np.array([p.getDec().asRadians() for p in legacy_coords]) * u.rad,
        )

    @staticmethod
    def _arrays_to_legacy_points(x: np.ndarray, y: np.ndarray) -> list[Any]:
        from lsst.geom import Point2D

        return [Point2D(x=xv, y=yv) for xv, yv in zip(x, y)]


class CameraFramesTestModel[P: pydantic.BaseModel](pydantic.BaseModel):
    """A testing serialization model that holds both a CamaraFrameSet and
    a Transform extracted from it.
    """

    frames: P
    pixels_to_fp: TransformSerializationModel[P]


if __name__ == "__main__":
    unittest.main()
