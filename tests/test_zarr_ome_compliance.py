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
import tempfile
import unittest

import numpy as np

from lsst.images import Box, Image, MaskedImage, MaskPlane, MaskSchema

try:
    import ngff_zarr
    import zarr
    from jsonschema import ValidationError

    from lsst.images.zarr import write
    from lsst.images.zarr._common import OME_VERSION

    # ``ngff_zarr.validate`` imports jsonschema lazily; importing it here
    # makes the skip condition reflect the ``ngff-zarr[validate]`` extra.
    HAVE_NGFF_VALIDATE = True
except ImportError:
    HAVE_NGFF_VALIDATE = False


@unittest.skipUnless(HAVE_NGFF_VALIDATE, "ngff-zarr[validate] is not installed")
class NgffComplianceTestCase(unittest.TestCase):
    """Archives written by the zarr backend validate against the NGFF schema.

    Uses the in-process validator from ``ngff-zarr[validate]``, which bundles
    the official OME-NGFF JSON schemas and checks a metadata dict with
    ``jsonschema``.
    """

    def _validate(self, target: str) -> None:
        """Validate a written archive's root OME metadata against NGFF.

        Reads the raw ``zarr.json`` attributes from the root group (where the
        backend writes the ``ome`` namespace) and checks them against the
        bundled image schema for the version the backend emits.
        """
        store = zarr.storage.LocalStore(target, read_only=True)
        root = zarr.open_group(store=store, mode="r")
        attrs = dict(root.attrs)
        try:
            ngff_zarr.validate(attrs, version=OME_VERSION, model="image")
        except ValidationError as exc:
            self.fail(f"NGFF validation failed for {target}: {exc.message}")

    def test_image_validates(self) -> None:
        image = Image(
            np.arange(20, dtype=np.float32).reshape(4, 5),
            bbox=Box.factory[10:14, 20:25],
        )
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "out.zarr")
            write(image, target)
            self._validate(target)

    def test_masked_image_validates(self) -> None:
        schema = MaskSchema([MaskPlane("BAD", "Bad pixel.")])
        image = Image(
            np.arange(20, dtype=np.float32).reshape(4, 5),
            bbox=Box.factory[10:14, 20:25],
        )
        masked = MaskedImage(image, mask_schema=schema)

        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "masked.zarr")
            write(masked, target)
            self._validate(target)


if __name__ == "__main__":
    unittest.main()
