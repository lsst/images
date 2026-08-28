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
import astropy.wcs
import numpy as np
import pydantic
import pytest
from astropy.coordinates import Longitude

from lsst.images import (
    ICRS,
    XY,
    YX,
    Box,
    DetectorFrame,
    FocalPlaneFrame,
    GeneralFrame,
    SkyProjection,
    Transform,
    TransformSerializationModel,
)
from lsst.images._transforms import _ast as astshim
from lsst.images.cameras import CameraFrameSet, CameraFrameSetSerializationModel
from lsst.images.describe import FieldRole, Report
from lsst.images.fits import PointerModel
from lsst.images.serialization import ArchiveTree, InputArchive, JsonRef, OutputArchive
from lsst.images.tests import (
    DP2_VISIT_DETECTOR_DATA_ID,
    RoundtripFits,
    RoundtripJson,
    check_transform,
    compare_sky_projection_to_legacy_wcs,
    legacy_points_to_xy_array,
    make_random_sky_projection,
    reset_afw_mask_planes,  # noqa: F401
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


@pytest.fixture
def legacy_detector_wcs(reset_afw_mask_planes: None) -> dict[str, Any]:  # noqa: F811
    """Return WCS-related objects read from visit_image.fits.

    Skips if TESTDATA_IMAGES_DIR is unset or lsst.afw.image is unavailable.
    """
    # reset_afw_mask_planes will have already skipped if afw is not available.
    from lsst.afw.image import ExposureFitsReader

    if EXTERNAL_DATA_DIR is None:
        pytest.skip("TESTDATA_IMAGES_DIR is not in the environment.")
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
    with pytest.raises(ValueError, match="not a mapping from pixel coordinates"):
        SkyProjection(Transform(ICRS, ICRS, astshim.UnitMap(2)))

    with pytest.raises(ValueError, match="not a mapping to ICRS"):
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


def test_fits_wcs_projection_to_legacy() -> None:
    """Verify that a projection created by from_fits_wcs can be converted
    to a legacy SkyWcs.

    The AST pixel frame uses the domain PIXEL, while lsst.afw.geom.SkyWcs
    requires PIXELS, so to_legacy has to rename it.
    """
    pytest.importorskip("lsst.afw.geom")
    rng = np.random.default_rng(43)
    bbox = Box.factory[75:275, 25:225]
    pixel_frame = GeneralFrame(unit=u.pix)
    sky_projection = make_random_sky_projection(rng, pixel_frame, bbox)
    legacy_wcs = sky_projection.to_legacy()
    compare_sky_projection_to_legacy_wcs(sky_projection, legacy_wcs, pixel_frame, bbox, is_fits=True)
    # The conversion must not modify the projection in place: its own AST
    # mapping keeps the PIXEL domain, and the conversion is repeatable.
    frame_set = sky_projection.pixel_to_sky_transform._ast_mapping
    assert isinstance(frame_set, astshim.FrameSet)
    domains = {frame_set.getFrame(i, copy=False).domain for i in range(1, frame_set.nFrame + 1)}
    assert "PIXEL" in domains
    assert "PIXELS" not in domains
    compare_sky_projection_to_legacy_wcs(
        sky_projection, sky_projection.to_legacy(), pixel_frame, bbox, is_fits=True
    )


def test_as_fits_wcs_unrepresentable_returns_none_not_keyerror() -> None:
    """Test that a transform that is not exactly FITS representable returns
    None from ``as_fits_wcs`` rather than crashing on the partial header AST
    writes.

    A pixel->sky mapping that AST can only partially encode (here a polynomial
    that produces no valid primary celestial WCS) writes a nonzero number of
    FITS cards whose header lacks a primary ``CTYPE``; constructing
    ``astropy.wcs.WCS`` from it previously raised ``KeyError``.
    """
    pixel_frame = GeneralFrame(unit=u.pix)
    coeff_f = np.array(
        [
            [1.0, 1, 0, 0],  # out1 += 1.0
            [1.0, 2, 0, 0],  # out2 += 1.0
            [1.0, 1, 1, 0],  # out1 += x
            [1.0, 2, 0, 1],  # out2 += y
            [1e-6, 1, 2, 0],  # out1 += 1e-6 x^2
            [1e-6, 2, 0, 2],  # out2 += 1e-6 y^2
        ]
    )
    sky_projection = SkyProjection(Transform(pixel_frame, ICRS, astshim.PolyMap(coeff_f, 2, "")))
    bbox = Box.factory[0:32, 0:32]
    assert sky_projection.as_fits_wcs(bbox, allow_approximation=False) is None


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


def test_detector_wcs_full_bbox(legacy_detector_wcs: dict[str, Any]) -> None:
    """Test that a from_legacy sky projection matchesthe legacy WCS over the
    whole detector bbox, not just a subregion away from the origin.

    We use a smaller box in most other tests for performance reasons, but this
    one is more senstive to the precision floor close to the origin in
    particular.
    """
    legacy_wcs = legacy_detector_wcs["legacy_wcs"]
    wcs_bbox = legacy_detector_wcs["wcs_bbox"]
    detector_frame = DetectorFrame(**DP2_VISIT_DETECTOR_DATA_ID, bbox=wcs_bbox)
    sky_projection = SkyProjection.from_legacy(legacy_wcs, detector_frame)
    compare_sky_projection_to_legacy_wcs(sky_projection, legacy_wcs, detector_frame, wcs_bbox)


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


@dataclasses.dataclass
class _BroadcastTestData:
    """Shared inputs for broadcasting/scalar tests on Transform and
    SkyProjection.
    """

    in_frame: DetectorFrame
    out_frame: GeneralFrame
    matrix: np.ndarray
    scalar_x: float
    scalar_y: float
    xv: list[int]
    yv: list[int]
    sky_proj: SkyProjection[DetectorFrame]


@pytest.fixture
def broadcast_test_data() -> _BroadcastTestData:
    """Return shared inputs for broadcasting/scalar tests."""
    in_frame = DetectorFrame(instrument="Inst", visit=1, detector=1, bbox=Box.factory[0:20, 0:20])
    return _BroadcastTestData(
        in_frame=in_frame,
        out_frame=GeneralFrame(unit=u.pix),
        matrix=np.array([[2.0, 0.5], [-0.3, 1.5]]),
        scalar_x=3.0,
        scalar_y=7.0,
        xv=[1, 2, 3],
        yv=[4, 5, 6],
        sky_proj=make_random_sky_projection(np.random.default_rng(42), in_frame, in_frame.bbox),
    )


def test_apply_forward_scalar(broadcast_test_data: _BroadcastTestData) -> None:
    """Verify that apply_forward and apply_inverse accept scalar x/y and return
    scalar floats, and that the _q variants accept scalar Quantity inputs.
    """
    t = broadcast_test_data.sky_proj.pixel_to_sky_transform
    # apply_forward with Python float scalars should return XY of floats.
    result_fwd = t.apply_forward(x=broadcast_test_data.scalar_x, y=broadcast_test_data.scalar_y)
    assert type(result_fwd.x) is float
    assert type(result_fwd.y) is float
    # Values must match the corresponding single-element array call.
    ref_fwd = t.apply_forward(
        x=np.array([broadcast_test_data.scalar_x]), y=np.array([broadcast_test_data.scalar_y])
    )
    assert result_fwd.x == ref_fwd.x[0]
    assert result_fwd.y == ref_fwd.y[0]
    # apply_inverse round-trips back to the original scalars.
    result_inv = t.apply_inverse(x=result_fwd.x, y=result_fwd.y)
    assert type(result_inv.x) is float
    assert type(result_inv.y) is float
    np.testing.assert_allclose(result_inv.x, broadcast_test_data.scalar_x, atol=1e-12)
    np.testing.assert_allclose(result_inv.y, broadcast_test_data.scalar_y, atol=1e-12)
    # apply_forward_q / apply_inverse_q with scalar Quantity inputs.
    x_q = broadcast_test_data.scalar_x * t.in_frame.unit
    y_q = broadcast_test_data.scalar_y * t.in_frame.unit
    result_fwd_q = t.apply_forward_q(x=x_q, y=y_q)
    assert result_fwd_q.x.shape == ()
    assert result_fwd_q.y.shape == ()
    np.testing.assert_allclose(result_fwd_q.x.to_value(t.out_frame.unit), result_fwd.x, atol=1e-12)
    result_inv_q = t.apply_inverse_q(x=result_fwd_q.x, y=result_fwd_q.y)
    assert result_inv_q.x.shape == ()
    np.testing.assert_allclose(
        result_inv_q.x.to_value(t.in_frame.unit), broadcast_test_data.scalar_x, atol=1e-12
    )


def test_apply_array_like_and_integer_input(broadcast_test_data: _BroadcastTestData) -> None:
    """Verify that apply_forward accepts Python lists and integer-dtype
    arrays, returning float64 ndarray results consistent with float64
    array input.
    """
    t = broadcast_test_data.sky_proj.pixel_to_sky_transform
    # Python list input should return an ndarray.
    result_list = t.apply_forward(x=broadcast_test_data.xv, y=broadcast_test_data.yv)
    assert isinstance(result_list.x, np.ndarray)
    assert isinstance(result_list.y, np.ndarray)
    ref = t.apply_forward(x=np.array(broadcast_test_data.xv), y=np.array(broadcast_test_data.yv))
    np.testing.assert_array_equal(result_list.x, ref.x)
    np.testing.assert_array_equal(result_list.y, ref.y)
    # Integer dtype arrays should not raise and should return float64.
    xi = np.array(broadcast_test_data.xv, dtype=np.int32)
    yi = np.array(broadcast_test_data.yv, dtype=np.int32)
    result_int = t.apply_forward(x=xi, y=yi)
    assert result_int.x.dtype == np.float64
    assert result_int.y.dtype == np.float64
    np.testing.assert_array_equal(result_int.x, ref.x)
    np.testing.assert_array_equal(result_int.y, ref.y)


def test_apply_broadcast(broadcast_test_data: _BroadcastTestData) -> None:
    """Verify that apply_forward and apply_inverse broadcast x and y like
    a NumPy ufunc, in both 1-D and 2-D cases.
    """
    t = broadcast_test_data.sky_proj.pixel_to_sky_transform
    xv = np.array(broadcast_test_data.xv)
    yv = np.array(broadcast_test_data.yv + [7])
    # 1-D broadcast: array x, scalar y.
    result_1d = t.apply_forward(x=xv, y=broadcast_test_data.scalar_y)
    assert isinstance(result_1d.x, np.ndarray)
    assert result_1d.x.shape == xv.shape
    ref_1d = t.apply_forward(x=xv, y=np.full_like(xv, broadcast_test_data.scalar_y))
    np.testing.assert_array_equal(result_1d.x, ref_1d.x)
    np.testing.assert_array_equal(result_1d.y, ref_1d.y)
    # 2-D broadcast: column x (M,1) × row y (1,N) -> (M,N).
    x2d = xv[:, np.newaxis]  # shape (3, 1)
    y2d = yv[np.newaxis, :]  # shape (1, 4)
    result_2d = t.apply_forward(x=x2d, y=y2d)
    assert result_2d.x.shape == (3, 4)
    assert result_2d.y.shape == (3, 4)
    # Values must match the fully expanded meshgrid call.
    xmesh, ymesh = np.meshgrid(xv, yv, indexing="ij")
    ref_2d = t.apply_forward(x=xmesh, y=ymesh)
    np.testing.assert_array_equal(result_2d.x, ref_2d.x)
    np.testing.assert_array_equal(result_2d.y, ref_2d.y)
    # apply_inverse also broadcasts.
    result_inv_2d = t.apply_inverse(x=result_2d.x, y=result_2d.y)
    assert result_inv_2d.x.shape == (3, 4)
    np.testing.assert_allclose(result_inv_2d.x, xmesh, atol=1e-12)
    np.testing.assert_allclose(result_inv_2d.y, ymesh, atol=1e-12)


def test_sky_projection_broadcast(broadcast_test_data: _BroadcastTestData) -> None:
    """Verify that SkyProjection.pixel_to_sky, sky_to_pixel, and the
    Astropy view broadcast x and y like a NumPy ufunc.
    """
    p = broadcast_test_data
    xv = np.array(p.xv)
    yv = np.array(p.yv + [7])
    # 1-D broadcast: array x, scalar y.
    sc_1d = p.sky_proj.pixel_to_sky(x=xv, y=p.scalar_y)
    assert sc_1d.shape == xv.shape
    ref_1d = p.sky_proj.pixel_to_sky(x=xv, y=np.full_like(xv, p.scalar_y))
    np.testing.assert_allclose(sc_1d.ra.rad, ref_1d.ra.rad, atol=1e-12)
    np.testing.assert_allclose(sc_1d.dec.rad, ref_1d.dec.rad, atol=1e-12)
    # 2-D broadcast: column x (M,1) × row y (1,N) -> (M,N).
    x2d = xv[:, np.newaxis]  # shape (3, 1)
    y2d = yv[np.newaxis, :]  # shape (1, 4)
    sc_2d = p.sky_proj.pixel_to_sky(x=x2d, y=y2d)
    assert sc_2d.shape == (3, 4)
    xmesh, ymesh = np.meshgrid(xv, yv, indexing="ij")
    ref_2d = p.sky_proj.pixel_to_sky(x=xmesh, y=ymesh)
    np.testing.assert_allclose(sc_2d.ra.rad, ref_2d.ra.rad, atol=1e-12)
    np.testing.assert_allclose(sc_2d.dec.rad, ref_2d.dec.rad, atol=1e-12)
    # sky_to_pixel round-trips back to the original grid.
    pix_2d = p.sky_proj.sky_to_pixel(sc_2d)
    assert pix_2d.x.shape == (3, 4)
    np.testing.assert_allclose(pix_2d.x, xmesh, atol=1e-9)
    np.testing.assert_allclose(pix_2d.y, ymesh, atol=1e-9)
    # SkyProjectionAstropyView.pixel_to_world_values also broadcasts.
    view = p.sky_proj.as_astropy()
    world_2d = view.pixel_to_world_values(x2d, y2d)
    assert world_2d[0].shape == (3, 4)
    assert world_2d[1].shape == (3, 4)
    np.testing.assert_allclose(world_2d[0], ref_2d.ra.rad, atol=1e-12)
    np.testing.assert_allclose(world_2d[1], ref_2d.dec.rad, atol=1e-12)
    # SkyProjectionAstropyView.world_to_pixel_values also broadcasts.
    ra_2d = ref_2d.ra.rad[:, np.newaxis, :]  # (3, 1, 4) — over-broadcast to check
    dec_2d = ref_2d.dec.rad[np.newaxis, :, :]  # (1, 3, 4)
    pix_world = view.world_to_pixel_values(ra_2d, dec_2d)
    assert pix_world[0].shape == (3, 3, 4)


def test_apply_xy_yx(broadcast_test_data: _BroadcastTestData) -> None:
    """Verify that apply_forward, apply_inverse, and the _q variants accept
    XY and YX positional arguments, producing results identical to the
    equivalent x=/y= keyword calls.
    """
    p = broadcast_test_data
    t = p.sky_proj.pixel_to_sky_transform
    sx, sy = p.scalar_x, p.scalar_y
    xv, yv = np.array(p.xv, dtype=float), np.array(p.yv, dtype=float)

    # --- apply_forward: scalar ---
    ref_fwd = t.apply_forward(x=sx, y=sy)
    assert t.apply_forward(XY(sx, sy)) == ref_fwd
    assert t.apply_forward(YX(sy, sx)) == ref_fwd

    # --- apply_forward: array ---
    ref_fwd_arr = t.apply_forward(x=xv, y=yv)
    np.testing.assert_array_equal(t.apply_forward(XY(xv, yv)).x, ref_fwd_arr.x)
    np.testing.assert_array_equal(t.apply_forward(YX(yv, xv)).x, ref_fwd_arr.x)

    # --- apply_inverse: scalar ---
    ref_inv = t.apply_inverse(x=ref_fwd.x, y=ref_fwd.y)
    assert t.apply_inverse(XY(ref_fwd.x, ref_fwd.y)) == ref_inv
    assert t.apply_inverse(YX(ref_fwd.y, ref_fwd.x)) == ref_inv

    # --- apply_forward_q: scalar ---
    x_q = sx * t.in_frame.unit
    y_q = sy * t.in_frame.unit
    ref_fwd_q = t.apply_forward_q(x=x_q, y=y_q)
    result_q = t.apply_forward_q(XY(x_q, y_q))
    np.testing.assert_allclose(result_q.x.value, ref_fwd_q.x.value, atol=1e-12)
    result_q_yx = t.apply_forward_q(YX(y_q, x_q))
    np.testing.assert_allclose(result_q_yx.x.value, ref_fwd_q.x.value, atol=1e-12)

    # --- apply_inverse_q: scalar ---
    ref_inv_q = t.apply_inverse_q(x=ref_fwd_q.x, y=ref_fwd_q.y)
    result_inv_q = t.apply_inverse_q(XY(ref_fwd_q.x, ref_fwd_q.y))
    np.testing.assert_allclose(result_inv_q.x.value, ref_inv_q.x.value, atol=1e-12)

    # --- TypeError on bad combinations ---
    with pytest.raises(TypeError):
        t.apply_forward(XY(sx, sy), x=sx)
    with pytest.raises(TypeError):
        t.apply_forward(YX(sy, sx), y=sy)
    with pytest.raises(TypeError):
        t.apply_forward()


def test_pixel_to_sky_xy_yx(broadcast_test_data: _BroadcastTestData) -> None:
    """Verify that SkyProjection.pixel_to_sky accepts XY and YX positional
    arguments, producing results identical to the x=/y= keyword form.
    """
    p = broadcast_test_data
    sx, sy = p.scalar_x, p.scalar_y
    xv, yv = np.array(p.xv, dtype=float), np.array(p.yv, dtype=float)

    # Scalar XY and YX.
    ref_scalar = p.sky_proj.pixel_to_sky(x=sx, y=sy)
    result_xy = p.sky_proj.pixel_to_sky(XY(sx, sy))
    result_yx = p.sky_proj.pixel_to_sky(YX(sy, sx))
    np.testing.assert_allclose(result_xy.ra.rad, ref_scalar.ra.rad, atol=1e-12)
    np.testing.assert_allclose(result_yx.ra.rad, ref_scalar.ra.rad, atol=1e-12)

    # Array XY and YX.
    ref_array = p.sky_proj.pixel_to_sky(x=xv, y=yv)
    result_xy_arr = p.sky_proj.pixel_to_sky(XY(xv, yv))
    result_yx_arr = p.sky_proj.pixel_to_sky(YX(yv, xv))
    np.testing.assert_allclose(result_xy_arr.ra.rad, ref_array.ra.rad, atol=1e-12)
    np.testing.assert_allclose(result_yx_arr.ra.rad, ref_array.ra.rad, atol=1e-12)

    # TypeError on bad combinations.
    with pytest.raises(TypeError):
        p.sky_proj.pixel_to_sky(XY(sx, sy), x=sx)
    with pytest.raises(TypeError):
        p.sky_proj.pixel_to_sky(YX(sy, sx), y=sy)
    with pytest.raises(TypeError):
        p.sky_proj.pixel_to_sky()


def _rotated_tan(rot_deg: float, *, crval2: float = 30.0, scale_y: float = 0.2) -> SkyProjection:
    """Return a rotated TAN projection with given pixel scales."""
    cx = (0.2 * u.arcsec).to_value(u.deg)
    cy = (scale_y * u.arcsec).to_value(u.deg)
    t = np.deg2rad(rot_deg)
    header = {
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 50,
        "CRPIX2": 100,
        "CRVAL1": 45.0,
        "CRVAL2": crval2,
        "CD1_1": -cx * np.cos(t),
        "CD1_2": cy * np.sin(t),
        "CD2_1": -cx * np.sin(t),
        "CD2_2": -cy * np.cos(t),
    }
    return SkyProjection.from_fits_wcs(astropy.wcs.WCS(header), GeneralFrame(unit=u.pix))


def test_sky_projection_nominal_pixel_scale() -> None:
    """_nominal_pixel_scale reports per sky axis [longitude, latitude].

    Faithful KPG1_PXSCL port: the scale attaches to the sky axis, so a 90 deg
    rotation swaps the returned [RA, Dec] scales.  Great-circle distances keep
    it correct near the poles.
    """
    bbox = Box.factory[0:200, 0:100]

    # Unrotated anisotropic WCS: RA scale 0.2, Dec scale 0.3.
    np.testing.assert_allclose(
        _rotated_tan(0.0, scale_y=0.3)._nominal_pixel_scale(bbox), [0.2, 0.3], rtol=1e-3
    )
    # Rotated 90 deg: the sky-axis scales swap.
    np.testing.assert_allclose(
        _rotated_tan(90.0, scale_y=0.3)._nominal_pixel_scale(bbox), [0.3, 0.2], rtol=1e-3
    )
    # Reference pixel ~2 arcsec from the north pole: great-circle scale holds.
    np.testing.assert_allclose(
        _rotated_tan(30.0, crval2=89.9995)._nominal_pixel_scale(bbox), [0.2, 0.2], rtol=1e-3
    )


def test_sky_projection_describe() -> None:
    """SkyProjection._describe reports pixels, pixel scale, and a corners
    table.
    """
    rng = np.random.default_rng(43)
    bbox = Box.factory[0:200, 0:100]
    pixel_frame = GeneralFrame(unit=u.pix)
    sky_projection = make_random_sky_projection(rng, pixel_frame, bbox)

    # Without a bbox this projection still has pixel_bounds, so the report is
    # characterized over those rather than falling back to the pixel origin.
    assert sky_projection.pixel_bounds is not None
    report = sky_projection.describe()
    assert isinstance(report, Report)
    assert report.type_name == "SkyProjection"
    assert report.title == "ICRS coordinates"
    # The title carries the sky frame, so it is not repeated as a field.
    assert not any(f.label == "domain" for f in report.fields)
    # The uninformative pixel_to_sky field is gone, so repr has no ARG fields.
    assert not any(f.role is FieldRole.ARG for f in report.fields)
    assert not any(f.label == "origin pixel" for f in report.fields)
    assert any(f.label == "center pixel" for f in report.fields)
    assert any(t.title == "Corners" for t in report.tables)
    # Exactly one nominal-pixel-scale field is present (single or dual form).
    scale_fields = [f for f in report.fields if f.label.startswith("Nominal pixel scale")]
    assert len(scale_fields) == 1

    # With a bbox: a Corners table and a center-pixel field for the box
    # center, and still no origin-pixel fallback.
    report = sky_projection.describe(bbox=bbox)
    corners = next(t for t in report.tables if t.title == "Corners")
    assert corners.columns == ["x", "y", "RA", "Dec", "RA (°)", "Dec (°)"]
    assert len(corners.rows) == 4
    # The first two columns carry the pixel coordinates of each corner.  These
    # are the box corners expanded by half a pixel, so that they bound the full
    # area the image covers rather than the centers of the outermost pixels.
    assert [(row[0], row[1]) for row in corners.rows] == [
        ("-0.5", "-0.5"),
        ("-0.5", "199.5"),
        ("99.5", "199.5"),
        ("99.5", "-0.5"),
    ]
    assert not any(f.label == "origin pixel" for f in report.fields)
    center = next(f for f in report.fields if f.label == "center pixel")
    assert center.role is FieldRole.DERIVED
    assert center.value.startswith("(x=")

    # FITS-WCS availability is reported (projection is FITS-representable).
    fits_field = next(f for f in report.fields if f.label == "fits_wcs")
    assert fits_field.value == "available"

    # A projection cannot be rebuilt from a string, so repr is descriptive
    # rather than an eval-ish call.  It does not depend on a bbox and does not
    # evaluate the mapping.
    assert repr(sky_projection) == "<SkyProjection: GeneralFrame → ICRS>"


def test_sky_projection_describe_pixel_scale_forms() -> None:
    """The pixel-scale field collapses to one value when the axes agree."""
    bbox = Box.factory[0:200, 0:100]

    # Isotropic scale: a single "Nominal pixel scale" field to 0.01 arcsec.
    report = _rotated_tan(0.0).describe(bbox=bbox)
    scale = next(f for f in report.fields if f.label == "Nominal pixel scale")
    assert scale.value == "0.20 arcsec"
    assert not any(f.label == "Nominal pixel scales" for f in report.fields)

    # Anisotropic scale: axes are named with the AST SkyFrame sky-axis labels.
    report = _rotated_tan(0.0, scale_y=0.3).describe(bbox=bbox)
    scales = next(f for f in report.fields if f.label == "Nominal pixel scales")
    assert scales.value == "Right ascension 0.20 arcsec; Declination 0.30 arcsec"
    assert not any(f.label == "Nominal pixel scale" for f in report.fields)


def test_sky_projection_describe_fits_wcs_availability() -> None:
    """fits_wcs is probed against a box, never blindly reported available."""
    # No supplied bbox and no pixel bounds: representability cannot be checked.
    projection = _rotated_tan(0.0)
    assert projection.pixel_bounds is None
    report = projection.describe()
    fits_field = next(f for f in report.fields if f.label == "fits_wcs")
    assert fits_field.value == "unknown"

    # A projection with pixel bounds probes those bounds when no bbox is given.
    bounded = SkyProjection.from_fits_wcs(
        _rotated_tan(0.0).pixel_to_sky_transform.as_fits_wcs(Box.factory[0:32, 0:32]),
        GeneralFrame(unit=u.pix),
        pixel_bounds=Box.factory[0:32, 0:32],
    )
    report = bounded.describe()
    fits_field = next(f for f in report.fields if f.label == "fits_wcs")
    assert fits_field.value in ("available", "none")


def test_sky_projection_astropy_view_repr_str() -> None:
    """The Astropy view has informative str and repr, not the default object
    form.
    """
    bbox = Box.factory[0:200, 0:100]
    sky_projection = _rotated_tan(0.0)

    bounded = sky_projection.as_astropy(bbox)
    assert str(bounded) == "SkyProjectionAstropyView([y=0:200, x=0:100] → ICRS)"
    bounded_repr = repr(bounded)
    assert bounded_repr.startswith("SkyProjectionAstropyView\n")
    assert "ICRS (ra, dec)" in bounded_repr
    assert str(bbox.shape) in bounded_repr
    # The reported pixel (0, 0) sky position matches the view's own transform.
    ra, dec = bounded.pixel_to_world_values(0.0, 0.0)
    ra_hms = Longitude(float(ra) * u.rad).to_string(unit=u.hour, sep="hms", pad=True, precision=1)
    assert ra_hms in bounded_repr

    unbounded = sky_projection.as_astropy()
    assert str(unbounded) == "SkyProjectionAstropyView(unbounded → ICRS)"
    # An unbounded view omits the array-shape line but keeps the reference.
    assert "array shape" not in repr(unbounded)
    assert "pixel (0, 0)" in repr(unbounded)


def test_sky_projection_describe_origin_pixel_is_a_last_resort() -> None:
    """The origin pixel appears only when no bounding box can be had.

    Pixel (0, 0) is not a reference point of a projection, so it is reported
    only when neither the caller nor the projection itself supplies a box to
    characterize over.
    """
    rng = np.random.default_rng(43)
    bbox = Box.factory[0:200, 0:100]
    bounded = make_random_sky_projection(rng, GeneralFrame(unit=u.pix), bbox)
    assert bounded.pixel_bounds is not None
    # A box from either source displaces the origin fallback.
    for report in (bounded.describe(), bounded.describe(bbox=bbox)):
        assert not any(f.label == "origin pixel" for f in report.fields)

    # With neither, the origin is all that is left, and the corners table and
    # FITS-WCS probe drop out with it.
    unbounded = _rotated_tan(0.0)
    assert unbounded.pixel_bounds is None
    report = unbounded.describe()
    assert not any(f.label == "center pixel" for f in report.fields)
    assert not any(t.title == "Corners" for t in report.tables)
    assert next(f for f in report.fields if f.label == "fits_wcs").value == "unknown"
    origin = next(f for f in report.fields if f.label == "origin pixel")
    assert origin.role is FieldRole.DERIVED
    assert origin.value.startswith("(x=0, y=0) →")


def test_sky_projection_describe_origin_pixel_matches_transform() -> None:
    """The origin-pixel field reports pixel (0, 0)'s actual sky position."""
    sky_projection = _rotated_tan(0.0)
    assert sky_projection.pixel_bounds is None

    report = sky_projection.describe()
    ref = next(f for f in report.fields if f.label == "origin pixel")
    sky00 = sky_projection.pixel_to_sky(x=0, y=0)
    # Sexagesimal (hms/dms) and labeled decimal degrees both appear.
    assert sky00.ra.to_string(unit=u.hour, sep="hms", pad=True, precision=1) in ref.value
    assert sky00.dec.to_string(sep="dms", pad=True, alwayssign=True, precision=0) in ref.value
    assert f"RA {sky00.ra.deg:.6f}°" in ref.value
    assert f"Dec {sky00.dec.deg:+.6f}°" in ref.value


def test_frame_describe_preserves_pydantic_repr() -> None:
    """Frames expose describe() while retaining pydantic's repr."""
    frame = GeneralFrame(unit=u.pix)
    report = frame.describe()
    assert report.type_name == "GeneralFrame"
    assert {f.label for f in report.fields} >= {"unit"}
    # The mixin must not shadow pydantic's repr.
    assert repr(frame).startswith("GeneralFrame(")


def test_transform_describe() -> None:
    """Transform._describe reports its frames and bounds."""
    pixel_frame = DetectorFrame(**DP2_VISIT_DETECTOR_DATA_ID, bbox=Box.factory[:5, :4])
    transform = Transform(pixel_frame, ICRS, astshim.UnitMap(2))
    report = transform.describe()
    assert isinstance(report, Report)
    assert report.type_name == "Transform"
    labels = {f.label for f in report.fields}
    assert {"in_frame", "out_frame"} <= labels
    assert any(f.label == "mapping" for f in report.fields)


def test_sky_projection_describe_extent() -> None:
    """The extent gives the box's angular size and its orientation.

    The size is the great-circle distance across the area the box covers, and
    the orientation is the position angle of the pixel y axis.
    """
    projection = _rotated_tan(40.0)
    # 275 pixels at 0.2 arcsec/pixel spans 55 arcsec.
    report = projection.describe(bbox=Box.factory[0:275, 0:275])
    extent = next(f for f in report.fields if f.label == "Image extent")
    assert extent.role is FieldRole.DERIVED
    assert extent.value == "55 x 55 arcsec @ 140 deg E of N"

    # The unit follows the scale of the box.
    def size(pixels: int) -> str:
        report = projection.describe(bbox=Box.factory[0:pixels, 0:pixels])
        return next(f.value for f in report.fields if f.label == "Image extent")

    assert "arcsec" in size(275)
    assert "arcmin" in size(4000)
    assert "deg" in size(60000)

    # A long, thin box has no unit that suits both sides, so each gets its
    # own; a square one names the shared unit once.
    def extent_of(x_pixels: int, y_pixels: int) -> str:
        report = projection.describe(bbox=Box.factory[0:y_pixels, 0:x_pixels])
        return next(f.value for f in report.fields if f.label == "Image extent")

    assert extent_of(10, 1000).startswith("2 arcsec x 3.33 arcmin ")
    assert extent_of(1000, 10).startswith("3.33 arcmin x 2 arcsec ")
    assert extent_of(275, 275).startswith("55 x 55 arcsec ")

    # No box, so no extent to report.
    assert projection.pixel_bounds is None
    assert not any(f.label == "Image extent" for f in projection.describe().fields)


def test_sky_projection_describe_extent_position_angle() -> None:
    """The reported angle is the position angle of the pixel y axis.

    ``_rotated_tan`` builds a CD matrix whose y axis lands at ``180 - rot``
    degrees East of North, which pins both the axis and the direction of
    increasing angle.
    """
    bbox = Box.factory[0:200, 0:100]
    for rot in (0.0, 30.0, 90.0, 270.0):
        report = _rotated_tan(rot).describe(bbox=bbox)
        value = next(f.value for f in report.fields if f.label == "Image extent")
        assert value.endswith(f"@ {(180 - rot) % 360:g} deg E of N"), (rot, value)


def test_sky_projection_describe_extent_is_measured_at_the_box() -> None:
    """The orientation is sampled at the box, not at the pixel origin.

    Meridians converge near the pole, so the position angle at a pixel far
    outside the box says nothing about the box: here the origin differs from
    the box by nearly 200 degrees.
    """
    projection = _rotated_tan(40.0, crval2=89.9995)
    bbox = Box.factory[0:200, 0:100]

    def position_angle_at(x: float, y: float) -> float:
        here = projection.pixel_to_sky(x=x, y=y)
        up = projection.pixel_to_sky(x=x, y=y + 1.0)
        return here.position_angle(up).to_value(u.deg) % 360.0

    at_origin = position_angle_at(0.0, 0.0)
    at_center = position_angle_at(bbox.x.center, bbox.y.center)
    assert abs(at_origin - at_center) > 100.0

    value = next(f.value for f in projection.describe(bbox=bbox).fields if f.label == "Image extent")
    assert value.endswith(f"@ {round(at_center) % 360} deg E of N")
