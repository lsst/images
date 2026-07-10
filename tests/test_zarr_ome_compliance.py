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

from pathlib import Path

import numpy as np
import pytest

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

skip_no_ngff_validate = pytest.mark.skipif(
    not HAVE_NGFF_VALIDATE, reason="ngff-zarr[validate] is not installed"
)


def validate_ngff(target: str) -> None:
    """Validate a written archive's root OME metadata against NGFF.

    Uses the in-process validator from ``ngff-zarr[validate]``, which bundles
    the official OME-NGFF JSON schemas and checks a metadata dict with
    ``jsonschema``.

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
        pytest.fail(f"NGFF validation failed for {target}: {exc.message}")


@skip_no_ngff_validate
def test_image_validates(tmp_path: Path) -> None:
    """An Image archive validates against the NGFF schema."""
    image = Image(
        np.arange(20, dtype=np.float32).reshape(4, 5),
        bbox=Box.factory[10:14, 20:25],
    )
    target = str(tmp_path / "out.zarr")
    write(image, target)
    validate_ngff(target)


@skip_no_ngff_validate
def test_masked_image_validates(tmp_path: Path) -> None:
    """A MaskedImage archive validates against the NGFF schema."""
    schema = MaskSchema([MaskPlane("BAD", "Bad pixel.")])
    image = Image(
        np.arange(20, dtype=np.float32).reshape(4, 5),
        bbox=Box.factory[10:14, 20:25],
    )
    masked = MaskedImage(image, mask_schema=schema)

    target = str(tmp_path / "masked.zarr")
    write(masked, target)
    validate_ngff(target)
