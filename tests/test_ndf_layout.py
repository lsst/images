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

"""Layout sanity tests for NdfOutputArchive.

Opens files written by NdfOutputArchive with raw h5py and verifies
the on-disk layout matches the HDS-on-HDF5 / NDF spec.

Notes on mask routing
---------------------
NDF serialization stores ``Mask`` arrays as a 3-D ``uint8`` DATA primitive
whose HDS axes are ``(x, y, mask-byte)``.  The HDF5 dataset shape is reversed
from that, following hds-v5 convention.  It also writes a 2-D ``QUALITY``
view: single-byte masks are copied directly, while wider masks collapse to
0/1 values.
"""

from __future__ import annotations

import numpy as np
import pytest

from lsst.images import Box, Image, MaskedImage, MaskPlane, MaskSchema
from lsst.images.tests import RoundtripNdf

try:
    import h5py

    from lsst.images.ndf import _hds

    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False


skip_no_h5py = pytest.mark.skipif(not HAVE_H5PY, reason="h5py is not installed")


def _cls(node: h5py.Group) -> str:
    """Return the HDS type (CLASS attribute) of an h5py group as a
    Python str.
    """
    val = node.attrs.get(_hds.ATTR_CLASS)
    if val is None:
        # Legacy fallback used by older HDS variants.
        val = node.attrs.get("HDSTYPE")
    if isinstance(val, bytes):
        return val.decode("ascii")
    return str(val)


def _hds_type(dataset: h5py.Dataset) -> str:
    """Return the HDS primitive type string inferred from a dataset's
    numpy dtype.
    """
    dataset_type = dataset.id.get_type()
    if dataset_type.get_class() == h5py.h5t.BITFIELD:
        return "_LOGICAL"
    return _hds.hds_type_for_dtype(dataset.dtype)


def _hds_shape(dataset: h5py.Dataset) -> tuple[int, ...]:
    """Return the dataset shape in HDS/Fortran axis order."""
    return tuple(reversed(dataset.shape))


@skip_no_h5py
def test_image_layout() -> None:
    """Verify the on-disk layout produced by ``ndf.write()`` for a plain
    Image.
    """
    image = Image(
        np.arange(20, dtype=np.float32).reshape(4, 5),
        bbox=Box.factory[10:14, 20:25],
    )
    with RoundtripNdf(image) as roundtrip:
        f = roundtrip.inspect()
        # Root group carries CLASS="NDF".
        assert _cls(f["/"]) == "NDF"

        # DATA_ARRAY is an ARRAY structure.
        assert "DATA_ARRAY" in f
        assert _cls(f["/DATA_ARRAY"]) == "ARRAY"

        # DATA is a 2-D _REAL primitive whose shape matches the image.
        assert "DATA" in f["/DATA_ARRAY"]
        ds = f["/DATA_ARRAY/DATA"]
        assert _hds_type(ds) == "_REAL"
        assert ds.ndim == 2
        assert ds.shape == image.array.shape

        # ORIGIN stores bbox lower bounds as int64 in (x_min, y_min) order.
        assert "ORIGIN" in f["/DATA_ARRAY"]
        origin = f["/DATA_ARRAY/ORIGIN"][()]
        assert origin.dtype == np.int64
        assert int(origin[0]) == 20  # x_min from Box.factory[10:14, 20:25]
        assert int(origin[1]) == 10  # y_min

        # /MORE is the standard NDF extension container (EXT) and
        # /MORE/LSST carries the type "LSST" matching its name.
        assert "MORE" in f
        assert _cls(f["/MORE"]) == "EXT"
        assert "LSST" in f["/MORE"]
        assert _cls(f["/MORE/LSST"]) == "LSST"

        # Main JSON serialisation tree is present.
        assert "JSON" in f["/MORE/LSST"]


@skip_no_h5py
def test_masked_image_compatible_mask_layout() -> None:
    """Verify the on-disk layout for a MaskedImage whose mask fits in a
    single byte.

    Even though the mask schema has only 2 planes (which would fit in a single
    NDF QUALITY byte), MaskedImage writes the native 3-D uint8 backing array
    in ``/MORE/LSST/MASK`` and a direct 2-D copy in ``/QUALITY``.
    """
    planes = [MaskPlane("BAD", "Bad pixel"), MaskPlane("SAT", "Saturated")]
    schema = MaskSchema(planes)  # default dtype=uint8, mask_size=1
    image = Image(
        np.arange(20, dtype=np.float32).reshape(4, 5),
        bbox=Box.factory[10:14, 20:25],
    )
    # Pass an explicit float64 Image as variance so we can verify _DOUBLE
    # on disk (the default variance is float32, matching the image dtype).
    variance = Image(np.ones((4, 5), dtype=np.float64), bbox=image.bbox)
    masked = MaskedImage(image, mask_schema=schema, variance=variance)
    masked.mask.set("BAD", image.array % 2 == 0)
    masked.mask.set("SAT", image.array > 10)

    with RoundtripNdf(masked) as roundtrip:
        f = roundtrip.inspect()
        assert "QUALITY" in f
        assert _cls(f["/QUALITY"]) == "QUALITY"
        assert _cls(f["/QUALITY/QUALITY"]) == "ARRAY"
        quality_ds = f["/QUALITY/QUALITY/DATA"]
        assert _hds_type(quality_ds) == "_UBYTE"
        assert quality_ds.shape == image.array.shape
        assert _hds_shape(quality_ds) == (image.array.shape[1], image.array.shape[0])
        np.testing.assert_array_equal(quality_ds[()], masked.mask.array[:, :, 0])
        quality_origin = f["/QUALITY/QUALITY/ORIGIN"]
        assert _hds_type(quality_origin) == "_INTEGER"
        assert list(quality_origin[()]) == [20, 10]
        bad_pixel = f["/QUALITY/QUALITY/BAD_PIXEL"]
        assert _hds_type(bad_pixel) == "_LOGICAL"
        assert not bad_pixel[()]
        assert f["/QUALITY/BADBITS"][()] == 255

        # /MORE/LSST/MASK is a sub-NDF (CLASS="NDF") with a
        # canonical DATA_ARRAY structure containing DATA + ORIGIN.
        assert "MORE" in f
        assert "LSST" in f["/MORE"]
        assert "MASK" in f["/MORE/LSST"]
        assert _cls(f["/MORE/LSST/MASK"]) == "NDF"
        assert _cls(f["/MORE/LSST/MASK/DATA_ARRAY"]) == "ARRAY"
        mask_ds = f["/MORE/LSST/MASK/DATA_ARRAY/DATA"]
        assert _hds_type(mask_ds) == "_UBYTE"
        assert mask_ds.ndim == 3
        assert mask_ds.shape == (1, 4, 5)
        assert _hds_shape(mask_ds) == (5, 4, 1)
        origin = f["/MORE/LSST/MASK/DATA_ARRAY/ORIGIN"]
        assert origin.dtype == np.int64
        # The mask shares the parent image's bbox; the trailing mask
        # byte axis keeps a zero origin.
        assert list(origin[()]) == [20, 10, 0]
        bad_pixel = f["/MORE/LSST/MASK/DATA_ARRAY/BAD_PIXEL"]
        assert _hds_type(bad_pixel) == "_LOGICAL"
        assert not bad_pixel[()]

        # VARIANCE is an ARRAY structure whose DATA is _DOUBLE (float64).
        assert "VARIANCE" in f
        assert _cls(f["/VARIANCE"]) == "ARRAY"
        assert "DATA" in f["/VARIANCE"]
        assert _hds_type(f["/VARIANCE/DATA"]) == "_DOUBLE"


@skip_no_h5py
def test_masked_image_incompatible_mask_layout() -> None:
    """Verify the on-disk layout for a MaskedImage with more than 8
    mask planes.

    A 12-plane uint8 mask has ``mask_size=2`` (two bytes per pixel), and the
    on-disk HDS axes are ``(x, y, mask-byte)``.
    """
    planes = [MaskPlane(f"P{i}", f"Plane {i}") for i in range(12)]
    schema = MaskSchema(planes)  # default uint8; mask_size = ceil(12/8) = 2
    image = Image(
        np.arange(20, dtype=np.float32).reshape(4, 5),
        bbox=Box.factory[10:14, 20:25],
    )
    masked = MaskedImage(image, mask_schema=schema)
    masked.mask.set("P0", image.array % 2 == 0)
    masked.mask.set("P11", image.array > 10)
    expected_quality = np.any(masked.mask.array != 0, axis=2).astype(np.uint8)

    with RoundtripNdf(masked) as roundtrip:
        f = roundtrip.inspect()
        assert "QUALITY" in f
        assert _cls(f["/QUALITY/QUALITY"]) == "ARRAY"
        quality_ds = f["/QUALITY/QUALITY/DATA"]
        assert _hds_type(quality_ds) == "_UBYTE"
        assert quality_ds.shape == image.array.shape
        np.testing.assert_array_equal(quality_ds[()], expected_quality)
        assert f["/QUALITY/BADBITS"][()] == 255

        # /MORE/LSST/MASK is a sub-NDF.
        assert "MORE" in f
        assert "LSST" in f["/MORE"]
        assert "MASK" in f["/MORE/LSST"]
        assert _cls(f["/MORE/LSST/MASK"]) == "NDF"
        assert _cls(f["/MORE/LSST/MASK/DATA_ARRAY"]) == "ARRAY"

        ds = f["/MORE/LSST/MASK/DATA_ARRAY/DATA"]
        assert _hds_type(ds) == "_UBYTE"
        assert ds.ndim == 3
        rows, cols = image.array.shape
        assert ds.shape == (2, rows, cols)
        assert _hds_shape(ds) == (cols, rows, 2)
        bad_pixel = f["/MORE/LSST/MASK/DATA_ARRAY/BAD_PIXEL"]
        assert _hds_type(bad_pixel) == "_LOGICAL"
        assert not bad_pixel[()]


@skip_no_h5py
def test_masked_image_many_plane_mask_layout() -> None:
    """Verify the on-disk layout for a MaskedImage with more than 31 planes."""
    planes = [MaskPlane(f"P{i}", f"Plane {i}") for i in range(40)]
    schema = MaskSchema(planes)
    image = Image(
        np.arange(20, dtype=np.float32).reshape(4, 5),
        bbox=Box.factory[10:14, 20:25],
    )
    masked = MaskedImage(image, mask_schema=schema)
    masked.mask.set("P0", image.array % 2 == 0)
    masked.mask.set("P17", image.array > 10)
    masked.mask.set("P39", image.array == 19)
    expected_quality = np.any(masked.mask.array != 0, axis=2).astype(np.uint8)

    with RoundtripNdf(masked) as roundtrip:
        f = roundtrip.inspect()
        assert "QUALITY" in f
        assert _cls(f["/QUALITY/QUALITY"]) == "ARRAY"
        quality_ds = f["/QUALITY/QUALITY/DATA"]
        assert _hds_type(quality_ds) == "_UBYTE"
        assert quality_ds.shape == image.array.shape
        np.testing.assert_array_equal(quality_ds[()], expected_quality)
        assert f["/QUALITY/BADBITS"][()] == 255
        ds = f["/MORE/LSST/MASK/DATA_ARRAY/DATA"]
        assert _hds_type(ds) == "_UBYTE"
        assert ds.ndim == 3
        rows, cols = image.array.shape
        assert ds.shape == (5, rows, cols)
        assert _hds_shape(ds) == (cols, rows, 5)
