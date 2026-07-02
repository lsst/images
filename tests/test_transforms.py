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
from typing import Any, ClassVar

import astropy.units as u
import numpy as np
import pydantic
import pytest

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

EXTERNAL_DATA_DIR = os.environ.get("TESTDATA_IMAGES_DIR", None)


@pytest.fixture(scope="session")
def legacy_camera() -> Any:
    """Return a legacy Camera loaded from camera.fits.

    Skips if TESTDATA_IMAGES_DIR is unset or lsst.afw.cameraGeom is
    unavailable.
    """
    if EXTERNAL_DATA_DIR is None:
        pytest.skip("TESTDATA_IMAGES_DIR is not in the environment.")
    try:
        from lsst.afw.cameraGeom import Camera
    except ImportError:
        pytest.skip("'lsst.afw.cameraGeom' could not be imported.")
    filename = os.path.join(EXTERNAL_DATA_DIR, "dp2", "legacy", "camera.fits")
    return Camera.readFits(filename)


@pytest.fixture(scope="session")
def legacy_detector_wcs() -> dict[str, Any]:
    """Return WCS-related objects read from visit_image.fits.

    Skips if TESTDATA_IMAGES_DIR is unset or lsst.afw.image is unavailable.
    """
    if EXTERNAL_DATA_DIR is None:
        pytest.skip("TESTDATA_IMAGES_DIR is not in the environment.")
    try:
        from lsst.afw.image import ExposureFitsReader
    except ImportError:
        pytest.skip("'lsst.afw.image' could not be imported.")
    filename = os.path.join(EXTERNAL_DATA_DIR, "dp2", "legacy", "visit_image.fits")
    reader = ExposureFitsReader(filename)
    return {
        "legacy_wcs": reader.readWcs(),
        "wcs_bbox": Box.from_legacy(reader.readDetector().getBBox()),
        "subimage_bbox": Box.from_legacy(reader.readBBox()),
    }


def test_identity() -> None:
    """Test an identity transform."""
    frame = DetectorFrame(**DP2_VISIT_DETECTOR_DATA_ID, bbox=Box.factory[:5, :4])
    xy = frame.bbox.meshgrid().map(np.ravel)
    identity = Transform.identity(frame)
    check_transform(identity, xy, xy, frame, frame)
    assert identity.decompose() == []
    with RoundtripJson(identity) as roundtrip:
        pass
    check_transform(roundtrip.result, xy, xy, frame, frame)


def test_transform_equality() -> None:
    """Test Transform.__eq__ across all of its comparison branches."""
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
    assert base == base

    # Two independently constructed but equivalent transforms are equal,
    # and equality is symmetric.
    assert base == make()
    assert make() == base

    # Comparison against a non-Transform yields NotImplemented, so Python
    # falls back to identity: the objects are unequal and != is True.
    assert not (base == "not a transform")
    assert base != "not a transform"
    assert base != None  # noqa: E711
    assert base != 42

    # Each remaining branch differs from base in exactly one attribute.
    assert base != make(ast_mapping=astshim.ShiftMap([1.0, 2.0]))
    assert base != make(in_bounds_=out_bounds)
    assert base != make(out_bounds_=in_bounds)
    assert base != make(in_frame=alt_frame)
    assert base != make(out_frame=alt_frame)
    assert base != make(components=[Transform.identity(alt_frame)])


def test_sky_projection_equality() -> None:
    """Test SkyProjection.__eq__ across all of its comparison branches."""
    pixel_frame = DetectorFrame(**DP2_VISIT_DETECTOR_DATA_ID, bbox=Box.factory[:5, :4])

    # Check the two failure modes.
    with pytest.raises(ValueError):
        SkyProjection(Transform(ICRS, ICRS, astshim.UnitMap(2)))

    with pytest.raises(ValueError):
        SkyProjection(Transform(pixel_frame, pixel_frame, astshim.UnitMap(2)))

    def make_pixel_to_sky(ast_mapping: astshim.Mapping | None = None) -> Transform[Any, Any]:
        mapping = ast_mapping if ast_mapping is not None else astshim.UnitMap(2)
        return Transform(pixel_frame, ICRS, mapping)

    base = SkyProjection(make_pixel_to_sky())

    # Identity short-circuit: an object is always equal to itself.
    assert base == base

    # Two independently constructed but equivalent projections are equal.
    assert base == SkyProjection(make_pixel_to_sky())

    # Comparison against a non-SkyProjection yields NotImplemented.
    assert not (base == "not a projection")
    assert base != "not a projection"
    assert base != None  # noqa: E711

    # Differ only in the pixel-to-sky transform.
    assert base != SkyProjection(make_pixel_to_sky(astshim.ShiftMap([1.0, 2.0])))

    # The fits_approximation branch: absent on base but present here.
    with_approx = SkyProjection(
        make_pixel_to_sky(), fits_approximation=make_pixel_to_sky(astshim.ShiftMap([0.1, 0.2]))
    )
    assert base != with_approx

    # Equal pixel-to-sky and equal fits_approximations are equal.
    with_approx_again = SkyProjection(
        make_pixel_to_sky(), fits_approximation=make_pixel_to_sky(astshim.ShiftMap([0.1, 0.2]))
    )
    assert with_approx == with_approx_again

    # Same pixel-to-sky transform but a different fits_approximation.
    other_approx = SkyProjection(
        make_pixel_to_sky(), fits_approximation=make_pixel_to_sky(astshim.ShiftMap([0.3, 0.4]))
    )
    assert with_approx != other_approx


def test_affine_2x2() -> None:
    """Test an affine transform constructed from a 2x2 matrix."""
    in_frame = DetectorFrame(**DP2_VISIT_DETECTOR_DATA_ID, bbox=Box.factory[:5, :4])
    out_frame = GeneralFrame(unit=u.pix)
    transform_matrix = np.array([[2.0, 0.25], [-0.75, 0.8]])
    in_xy = in_frame.bbox.meshgrid().map(np.ravel)
    in_matrix = np.array([in_xy.x, in_xy.y])
    out_matrix = np.dot(transform_matrix, in_matrix)
    check_transform(
        Transform.affine(in_frame, out_frame, transform_matrix),
        in_xy,
        XY(x=out_matrix[0, :], y=out_matrix[1, :]),
        in_frame,
        out_frame,
        in_atol=1e-15 * u.pix,
        out_atol=1e-15 * u.pix,
    )


def test_affine_3x3() -> None:
    """Test an affine transform constructed from a 3x3 matrix."""
    in_frame = DetectorFrame(**DP2_VISIT_DETECTOR_DATA_ID, bbox=Box.factory[:5, :4])
    out_frame = GeneralFrame(unit=u.pix)
    transform_matrix = np.array([[2.0, 0.25, -0.5], [-0.75, 0.8, 0.4], [0.0, 0.0, 1.0]])
    in_xy = in_frame.bbox.meshgrid().map(np.ravel)
    in_matrix = np.array([in_xy.x, in_xy.y, np.ones(in_xy.x.shape)])
    out_matrix = np.dot(transform_matrix, in_matrix)
    check_transform(
        Transform.affine(in_frame, out_frame, transform_matrix),
        in_xy,
        XY(x=out_matrix[0, :], y=out_matrix[1, :]),
        in_frame,
        out_frame,
        in_atol=1e-15 * u.pix,
        out_atol=1e-15 * u.pix,
    )


def compare_to_legacy_camera(legacy_camera: Any, frame_set: CameraFrameSet) -> None:
    """Assert that transforms extracted from a CameraFrameSet match the
    legacy afw implementations.
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
    check_transform(pixel_to_fp, pixel_xy_array, fp_xy_array, frame_set.detector(16), frame_set.focal_plane())
    pixel_to_fa = frame_set[frame_set.detector(16), frame_set.field_angle()]
    check_transform(pixel_to_fa, pixel_xy_array, fa_xy_array, frame_set.detector(16), frame_set.field_angle())
    fp_to_fa = frame_set[frame_set.focal_plane(), frame_set.field_angle()]
    check_transform(fp_to_fa, fp_xy_array, fa_xy_array, frame_set.focal_plane(), frame_set.field_angle())
    # Test a composition.
    pixel_to_fa_indirect = pixel_to_fp.then(fp_to_fa)
    check_transform(
        pixel_to_fa_indirect,
        pixel_xy_array,
        fa_xy_array,
        frame_set.detector(16),
        frame_set.field_angle(),
    )
    pixel_to_fp_d, fp_to_fa_d = pixel_to_fa_indirect.decompose()
    check_transform(
        pixel_to_fp_d, pixel_xy_array, fp_xy_array, frame_set.detector(16), frame_set.focal_plane()
    )
    check_transform(fp_to_fa_d, fp_xy_array, fa_xy_array, frame_set.focal_plane(), frame_set.field_angle())
    fa_to_fp_d, fp_to_pixel_d = pixel_to_fa_indirect.inverted().decompose()
    check_transform(fa_to_fp_d, fa_xy_array, fp_xy_array, frame_set.field_angle(), frame_set.focal_plane())
    check_transform(
        fp_to_pixel_d, fp_xy_array, pixel_xy_array, frame_set.focal_plane(), frame_set.detector(16)
    )


def test_camera(legacy_camera: Any) -> None:
    """Test CameraFrameSet construction, transforms, and FITS/JSON
    serialization round-trips.

    Also verifies the archive system's pointer and frame-set reference
    machinery.
    """
    legacy_camera = legacy_camera
    frame_set = CameraFrameSet.from_legacy(legacy_camera)
    detector_id: int = DP2_VISIT_DETECTOR_DATA_ID["detector"]
    compare_to_legacy_camera(legacy_camera, frame_set)
    test_holder = FrameSetTestHolder(
        frames=frame_set,
        pixels_to_fp=frame_set[frame_set.detector(detector_id), frame_set.focal_plane()],
    )
    with RoundtripFits(test_holder) as roundtrip1:
        assert len(roundtrip1.serialized.pixels_to_fp.frames) == 2
        assert len(roundtrip1.serialized.pixels_to_fp.bounds) == 2
        assert len(roundtrip1.serialized.pixels_to_fp.mappings) == 1
        # Instead of storing the AST mapping directly, we should have
        # stored a reference to the frame set:
        assert isinstance(roundtrip1.serialized.pixels_to_fp.mappings[0], PointerModel)
    compare_to_legacy_camera(legacy_camera, roundtrip1.result.frames)
    assert roundtrip1.result.pixels_to_fp.in_frame == frame_set.detector(detector_id)
    assert roundtrip1.result.pixels_to_fp.out_frame == frame_set.focal_plane()
    assert (
        roundtrip1.result.pixels_to_fp._ast_mapping.simplified().show()
        == test_holder.pixels_to_fp._ast_mapping.simplified().show()
    )
    with RoundtripJson(test_holder) as roundtrip2:
        assert len(roundtrip2.serialized.pixels_to_fp.frames) == 2
        assert len(roundtrip2.serialized.pixels_to_fp.bounds) == 2
        assert len(roundtrip2.serialized.pixels_to_fp.mappings) == 1
        # Instead of storing the AST mapping directly, we should have
        # stored a reference to the frame set:
        assert isinstance(roundtrip2.serialized.pixels_to_fp.mappings[0], JsonRef)
        raw_data = roundtrip2.inspect()
        assert len(raw_data["indirect"]) == 1
        assert raw_data["frames"] == {"$ref": "#/indirect/0"}
    compare_to_legacy_camera(legacy_camera, roundtrip2.result.frames)
    assert roundtrip2.result.pixels_to_fp.in_frame == frame_set.detector(detector_id)
    assert roundtrip2.result.pixels_to_fp.out_frame == frame_set.focal_plane()
    assert (
        roundtrip2.result.pixels_to_fp._ast_mapping.simplified().show()
        == test_holder.pixels_to_fp._ast_mapping.simplified().show()
    )


def test_detector_wcs(legacy_detector_wcs: dict[str, Any]) -> None:
    """Test the Transform/SkyProjection representation of a detector WCS."""
    legacy_wcs = legacy_detector_wcs["legacy_wcs"]
    wcs_bbox = legacy_detector_wcs["wcs_bbox"]
    subimage_bbox = legacy_detector_wcs["subimage_bbox"]
    detector_frame = DetectorFrame(**DP2_VISIT_DETECTOR_DATA_ID, bbox=wcs_bbox)
    sky_projection = SkyProjection.from_legacy(legacy_wcs, detector_frame)
    assert sky_projection.fits_approximation is not None
    compare_sky_projection_to_legacy_wcs(sky_projection, legacy_wcs, detector_frame, subimage_bbox)
    # When we convert from a legacy SkyWcs, the internal AST Mapping needs
    # to really be an AST FrameSet in order to be able to convert back.
    assert "Begin FrameSet" in sky_projection.show()
    compare_sky_projection_to_legacy_wcs(
        sky_projection, sky_projection.to_legacy(), detector_frame, subimage_bbox
    )
    assert "Begin FrameSet" in sky_projection.fits_approximation.show()
    compare_sky_projection_to_legacy_wcs(
        sky_projection.fits_approximation,
        sky_projection.fits_approximation.to_legacy(),
        detector_frame,
        subimage_bbox,
        is_fits=True,
    )
    with RoundtripJson(sky_projection, "SkyProjection") as roundtrip:
        pass
    compare_sky_projection_to_legacy_wcs(roundtrip.result, legacy_wcs, detector_frame, subimage_bbox)
    # The AST FrameSet-ness needs to propagate through serialization.
    assert "Begin FrameSet" in roundtrip.result.show()
    compare_sky_projection_to_legacy_wcs(
        sky_projection, roundtrip.result.to_legacy(), detector_frame, subimage_bbox
    )
    with RoundtripJson(sky_projection.fits_approximation, "SkyProjection") as roundtrip:
        pass
    compare_sky_projection_to_legacy_wcs(
        roundtrip.result,
        legacy_wcs.getFitsApproximation(),
        detector_frame,
        subimage_bbox,
        is_fits=True,
    )
    assert "Begin FrameSet" in roundtrip.result.show()
    compare_sky_projection_to_legacy_wcs(
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
