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

import tempfile
import unittest

import h5py
import numpy as np

from lsst.images import Box, Image, MaskedImage, MaskPlane, MaskSchema
from lsst.images.ndf import _hds, write


def _cls(node: h5py.Group) -> str:
    """Return the HDS type (CLASS attribute) of an h5py group as a Python
    str.
    """
    val = node.attrs.get(_hds.ATTR_CLASS)
    if val is None:
        # Legacy fallback used by older HDS variants.
        val = node.attrs.get("HDSTYPE")
    if isinstance(val, bytes):
        return val.decode("ascii")
    return str(val)


def _hds_type(dataset: h5py.Dataset) -> str:
    """Return the HDS primitive type string inferred from a dataset's numpy
    dtype or low-level HDF5 type class.
    """
    dataset_type = dataset.id.get_type()
    if dataset_type.get_class() == h5py.h5t.BITFIELD:
        return "_LOGICAL"
    return _hds.hds_type_for_dtype(dataset.dtype)


def _hds_shape(dataset: h5py.Dataset) -> tuple[int, ...]:
    """Return the dataset shape in HDS/Fortran axis order."""
    return tuple(reversed(dataset.shape))


class NdfImageLayoutTestCase(unittest.TestCase):
    """Verify the on-disk layout produced by ``ndf.write()`` for a plain
    ``Image``.
    """

    def test_image_layout(self) -> None:
        """Write an Image and verify root CLASS, DATA_ARRAY, ORIGIN, and LSST
        ext.
        """
        image = Image(
            np.arange(20, dtype=np.float32).reshape(4, 5),
            bbox=Box.factory[10:14, 20:25],
        )
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete_on_close=False) as tmp:
            tmp.close()
            write(image, tmp.name)
            with h5py.File(tmp.name, "r") as f:
                # Root group carries CLASS="NDF".
                self.assertEqual(_cls(f["/"]), "NDF")

                # DATA_ARRAY is an ARRAY structure.
                self.assertIn("DATA_ARRAY", f)
                self.assertEqual(_cls(f["/DATA_ARRAY"]), "ARRAY")

                # DATA is a 2-D _REAL primitive whose shape matches the image.
                self.assertIn("DATA", f["/DATA_ARRAY"])
                ds = f["/DATA_ARRAY/DATA"]
                self.assertEqual(_hds_type(ds), "_REAL")
                self.assertEqual(ds.ndim, 2)
                self.assertEqual(ds.shape, image.array.shape)

                # ORIGIN stores bbox lower bounds as int64 in (x_min, y_min)
                # order.
                self.assertIn("ORIGIN", f["/DATA_ARRAY"])
                origin = f["/DATA_ARRAY/ORIGIN"][()]
                self.assertEqual(origin.dtype, np.int64)
                self.assertEqual(int(origin[0]), 20)  # x_min from Box.factory[10:14, 20:25]
                self.assertEqual(int(origin[1]), 10)  # y_min

                # /MORE/LSST is a general-purpose extension (EXT) group.
                self.assertIn("MORE", f)
                self.assertIn("LSST", f["/MORE"])
                self.assertEqual(_cls(f["/MORE/LSST"]), "EXT")

                # Main JSON serialisation tree is present.
                self.assertIn("JSON", f["/MORE/LSST"])


class NdfCompatibleMaskLayoutTestCase(unittest.TestCase):
    """Layout test for a MaskedImage whose mask fits in a single uint8 byte.

    Even though the mask schema has only 2 planes (which would fit in a single
    NDF QUALITY byte), MaskedImage writes the native 3-D uint8 backing array
    in ``/MORE/LSST/MASK`` and a direct 2-D copy in ``/QUALITY``.
    """

    def test_masked_image_compatible_mask_layout(self) -> None:
        """Write a MaskedImage with a ≤8-plane mask; verify QUALITY,
        LSST/MASK, and VARIANCE.
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

        with tempfile.NamedTemporaryFile(suffix=".sdf", delete_on_close=False) as tmp:
            tmp.close()
            write(masked, tmp.name)
            with h5py.File(tmp.name, "r") as f:
                self.assertIn("QUALITY", f)
                self.assertEqual(_cls(f["/QUALITY"]), "QUALITY")
                self.assertEqual(_cls(f["/QUALITY/QUALITY"]), "ARRAY")
                quality_ds = f["/QUALITY/QUALITY/DATA"]
                self.assertEqual(_hds_type(quality_ds), "_UBYTE")
                self.assertEqual(quality_ds.shape, image.array.shape)
                self.assertEqual(_hds_shape(quality_ds), (image.array.shape[1], image.array.shape[0]))
                np.testing.assert_array_equal(quality_ds[()], masked.mask.array[:, :, 0])
                quality_origin = f["/QUALITY/QUALITY/ORIGIN"]
                self.assertEqual(_hds_type(quality_origin), "_INTEGER")
                self.assertEqual(list(quality_origin[()]), [20, 10])
                bad_pixel = f["/QUALITY/QUALITY/BAD_PIXEL"]
                self.assertEqual(_hds_type(bad_pixel), "_LOGICAL")
                self.assertFalse(bad_pixel[()])
                self.assertEqual(f["/QUALITY/BADBITS"][()], 1)

                # /MORE/LSST/MASK is a sub-NDF (CLASS="NDF") with a
                # canonical DATA_ARRAY structure containing DATA + ORIGIN.
                self.assertIn("MORE", f)
                self.assertIn("LSST", f["/MORE"])
                self.assertIn("MASK", f["/MORE/LSST"])
                self.assertEqual(_cls(f["/MORE/LSST/MASK"]), "NDF")
                self.assertEqual(_cls(f["/MORE/LSST/MASK/DATA_ARRAY"]), "ARRAY")
                mask_ds = f["/MORE/LSST/MASK/DATA_ARRAY/DATA"]
                self.assertEqual(_hds_type(mask_ds), "_UBYTE")
                self.assertEqual(mask_ds.ndim, 3)
                self.assertEqual(mask_ds.shape, (1, 4, 5))
                self.assertEqual(_hds_shape(mask_ds), (5, 4, 1))
                origin = f["/MORE/LSST/MASK/DATA_ARRAY/ORIGIN"]
                self.assertEqual(origin.dtype, np.int64)
                # The mask shares the parent image's bbox; the trailing mask
                # byte axis keeps a zero origin.
                self.assertEqual(list(origin[()]), [20, 10, 0])

                # VARIANCE is an ARRAY structure whose DATA is _DOUBLE
                # (float64).
                self.assertIn("VARIANCE", f)
                self.assertEqual(_cls(f["/VARIANCE"]), "ARRAY")
                self.assertIn("DATA", f["/VARIANCE"])
                self.assertEqual(_hds_type(f["/VARIANCE/DATA"]), "_DOUBLE")


class NdfIncompatibleMaskLayoutTestCase(unittest.TestCase):
    """Layout test for a MaskedImage with more than 8 mask planes.

    A 12-plane uint8 mask has ``mask_size=2`` (two bytes per pixel), and the
    on-disk HDS axes are ``(x, y, mask-byte)``.
    """

    def test_masked_image_incompatible_mask_layout(self) -> None:
        """Write a MaskedImage with 12 planes; verify LSST/MASK and absent
        QUALITY.
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

        with tempfile.NamedTemporaryFile(suffix=".sdf", delete_on_close=False) as tmp:
            tmp.close()
            write(masked, tmp.name)
            with h5py.File(tmp.name, "r") as f:
                self.assertIn("QUALITY", f)
                self.assertEqual(_cls(f["/QUALITY/QUALITY"]), "ARRAY")
                quality_ds = f["/QUALITY/QUALITY/DATA"]
                self.assertEqual(_hds_type(quality_ds), "_UBYTE")
                self.assertEqual(quality_ds.shape, image.array.shape)
                np.testing.assert_array_equal(quality_ds[()], expected_quality)
                self.assertEqual(f["/QUALITY/BADBITS"][()], 1)

                # /MORE/LSST/MASK is a sub-NDF.
                self.assertIn("MORE", f)
                self.assertIn("LSST", f["/MORE"])
                self.assertIn("MASK", f["/MORE/LSST"])
                self.assertEqual(_cls(f["/MORE/LSST/MASK"]), "NDF")
                self.assertEqual(_cls(f["/MORE/LSST/MASK/DATA_ARRAY"]), "ARRAY")

                ds = f["/MORE/LSST/MASK/DATA_ARRAY/DATA"]
                self.assertEqual(_hds_type(ds), "_UBYTE")
                self.assertEqual(ds.ndim, 3)
                rows, cols = image.array.shape
                self.assertEqual(ds.shape, (2, rows, cols))
                self.assertEqual(_hds_shape(ds), (cols, rows, 2))

    def test_masked_image_many_plane_mask_layout(self) -> None:
        """Write a MaskedImage with more than 31 planes as one native mask."""
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

        with tempfile.NamedTemporaryFile(suffix=".sdf", delete_on_close=False) as tmp:
            tmp.close()
            write(masked, tmp.name)
            with h5py.File(tmp.name, "r") as f:
                self.assertIn("QUALITY", f)
                self.assertEqual(_cls(f["/QUALITY/QUALITY"]), "ARRAY")
                quality_ds = f["/QUALITY/QUALITY/DATA"]
                self.assertEqual(_hds_type(quality_ds), "_UBYTE")
                self.assertEqual(quality_ds.shape, image.array.shape)
                np.testing.assert_array_equal(quality_ds[()], expected_quality)
                self.assertEqual(f["/QUALITY/BADBITS"][()], 1)
                ds = f["/MORE/LSST/MASK/DATA_ARRAY/DATA"]
                self.assertEqual(_hds_type(ds), "_UBYTE")
                self.assertEqual(ds.ndim, 3)
                rows, cols = image.array.shape
                self.assertEqual(ds.shape, (5, rows, cols))
                self.assertEqual(_hds_shape(ds), (cols, rows, 5))


if __name__ == "__main__":
    unittest.main()
