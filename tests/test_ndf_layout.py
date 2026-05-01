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
``Mask.serialize`` always calls ``schema.split(np.int32)`` before writing,
which converts mask planes to ``int32`` regardless of the original schema
dtype.  Consequently all masks written through ``ndf.write()`` land in
``/MORE/LSST/MASK/DATA`` as a 2-D ``_INTEGER`` array rather than in the
NDF ``QUALITY`` component.  The QUALITY component can only be populated by
calling ``NdfOutputArchive.add_array`` directly with a 2-D ``uint8`` array
(see ``test_ndf_output_archive.py``).
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
    dtype.
    """
    return _hds.hds_type_for_dtype(dataset.dtype)


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
    NDF QUALITY byte), ``Mask.serialize`` uses ``schema.split(np.int32)`` which
    converts the mask to int32.  The resulting 2-D int32 array is not
    compatible with the NDF QUALITY component (which requires uint8), so the
    mask is stored in ``/MORE/LSST/MASK/DATA`` instead.
    """

    def test_masked_image_compatible_mask_layout(self) -> None:
        """Write a MaskedImage with a ≤8-plane mask; verify LSST/MASK and
        VARIANCE.
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

        with tempfile.NamedTemporaryFile(suffix=".sdf", delete_on_close=False) as tmp:
            tmp.close()
            write(masked, tmp.name)
            with h5py.File(tmp.name, "r") as f:
                # Mask is serialised as int32 via schema.split(np.int32).
                # QUALITY is therefore absent even for a single-byte-wide
                # mask.
                self.assertNotIn(
                    "QUALITY",
                    f,
                    msg="QUALITY must not be written for masks serialised via MaskedImage.write()",
                )

                # The mask lands in /MORE/LSST/MASK as a STRUCT with a 2-D
                # _INTEGER (int32) DATA primitive.
                self.assertIn("MORE", f)
                self.assertIn("LSST", f["/MORE"])
                self.assertIn("MASK", f["/MORE/LSST"])
                self.assertEqual(_cls(f["/MORE/LSST/MASK"]), "STRUCT")
                self.assertIn("DATA", f["/MORE/LSST/MASK"])
                mask_ds = f["/MORE/LSST/MASK/DATA"]
                self.assertEqual(_hds_type(mask_ds), "_INTEGER")
                self.assertEqual(mask_ds.ndim, 2)
                self.assertEqual(mask_ds.shape, (4, 5))

                # VARIANCE is an ARRAY structure whose DATA is _DOUBLE
                # (float64).
                self.assertIn("VARIANCE", f)
                self.assertEqual(_cls(f["/VARIANCE"]), "ARRAY")
                self.assertIn("DATA", f["/VARIANCE"])
                self.assertEqual(_hds_type(f["/VARIANCE/DATA"]), "_DOUBLE")


class NdfIncompatibleMaskLayoutTestCase(unittest.TestCase):
    """Layout test for a MaskedImage with more than 8 mask planes.

    A 12-plane uint8 mask has ``mask_size=2`` (two bytes per pixel).  After
    ``schema.split(np.int32)`` the 12 planes still fit in a single int32
    element, so the on-disk layout remains a 2-D _INTEGER array in
    ``/MORE/LSST/MASK/DATA``.  The NDF QUALITY component is absent.
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

        with tempfile.NamedTemporaryFile(suffix=".sdf", delete_on_close=False) as tmp:
            tmp.close()
            write(masked, tmp.name)
            with h5py.File(tmp.name, "r") as f:
                # A 12-plane mask cannot fit in the NDF QUALITY component
                # (uint8 only holds 8 planes); QUALITY must be absent.
                self.assertNotIn("QUALITY", f, msg="12-plane mask must not produce /QUALITY")

                # The mask is stored in /MORE/LSST/MASK as a STRUCT.
                self.assertIn("MORE", f)
                self.assertIn("LSST", f["/MORE"])
                self.assertIn("MASK", f["/MORE/LSST"])
                self.assertEqual(_cls(f["/MORE/LSST/MASK"]), "STRUCT")

                # DATA is a 2-D _INTEGER (int32) array.  schema.split(np.int32)
                # folds the 12 planes into a single int32 element per pixel.
                self.assertIn("DATA", f["/MORE/LSST/MASK"])
                ds = f["/MORE/LSST/MASK/DATA"]
                self.assertEqual(_hds_type(ds), "_INTEGER")
                self.assertEqual(ds.ndim, 2)
                rows, cols = image.array.shape
                self.assertEqual(ds.shape, (rows, cols))


if __name__ == "__main__":
    unittest.main()
