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

__all__ = ()

import os

import click
import numpy as np

from lsst.afw.fits import (
    CompressionAlgorithm,
    CompressionOptions,
    DitherAlgorithm,
    QuantizationOptions,
    ScalingAlgorithm,
)
from lsst.daf.butler import Butler, DatasetRef
from lsst.geom import Box2I, Extent2I, Point2I

from .data_ids import DP2_VISIT_DETECTOR_DATA_ID


def extract_visit_image(
    butler: Butler,
    output_path: str,
    dataset_ref: DatasetRef,
    shuffle: bool,
    wcs_dataset_ref: DatasetRef | None = None,
) -> None:
    """Load a subimage of a processed visit image from the ci_hsc output
    repository and save it to testdata_images.
    """
    visit_image = butler.get(dataset_ref, parameters={"bbox": Box2I(Point2I(5, 4), Extent2I(256, 250))})
    if shuffle:
        indices = np.arange(visit_image.image.array.size, dtype=int)
        rng = np.random.default_rng()
        rng.shuffle(indices)
        visit_image.image.array[:, :] = visit_image.image.array.flat[indices].reshape(250, 256)
        visit_image.mask.array[:, :] = visit_image.mask.array.flat[indices].reshape(250, 256)
        visit_image.variance.array[:, :] = visit_image.variance.array.flat[indices].reshape(250, 256)
    if wcs_dataset_ref is not None:
        visit_summary = butler.get(wcs_dataset_ref)
        visit_summary_row = visit_summary.find(dataset_ref.dataId["detector"])
        visit_image.setWcs(visit_summary_row.getWcs())
    float_compression = CompressionOptions(
        algorithm=CompressionAlgorithm.RICE_1,
        tile_height=50,
        tile_width=64,
        quantization=QuantizationOptions(
            dither=DitherAlgorithm.SUBTRACTIVE_DITHER_2,
            scaling=ScalingAlgorithm.STDEV_MASKED,
            level=16,
            seed=747,
        ),
    )
    mask_compression = CompressionOptions(
        algorithm=CompressionAlgorithm.GZIP_2,
        tile_height=50,
        tile_width=64,
        quantization=None,
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    visit_image.writeFits(
        output_path,
        imageOptions=float_compression,
        maskOptions=mask_compression,
        varianceOptions=float_compression,
    )


def extract_camera(butler: Butler, output_path: str, dataset_ref: DatasetRef) -> None:
    camera = butler.get(dataset_ref)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    camera.writeFits(output_path)


def find_dataset_or_raise(
    butler: Butler, dataset_type: str, *, collections: str | None = None, **kwargs
) -> DatasetRef:
    ref = butler.find_dataset(dataset_type, collections=collections, **kwargs)
    if ref is None:
        raise LookupError(f"Could not find dataset {dataset_type} with data ID {kwargs}.")
    return ref


@click.group("extract_test_data")
def extract_test_data() -> None:
    pass


@extract_test_data.command("dp2")
@click.option("-b", "--butler-repo", help="Path to the butler repository.")
@click.option("-d", "--testdata-dir", help="Path to the testdata_images directory.")
@click.option(
    "-c",
    "--collection",
    default="LSSTCam/runs/DRP/20250515-20251214/v30_0_0_rc2/DM-53697",
    help="Collection to use for most data products.",
)
@click.option(
    "--wcs-collection",
    default="LSSTCam/runs/DRP/v30_0_0/DM-53877",
    help="Collection to search for visit_summary datasets used to update the WCS.",
)
def extract_dp2(
    butler_repo: str | None, testdata_dir: str | None, collection: str, wcs_collection: str
) -> None:
    """Extract test data from a butler repository."""
    if butler_repo is None:
        butler_repo = "dp2_prep"
    if testdata_dir is None:
        testdata_dir = os.environ["TESTDATA_IMAGES_DIR"]
    butler = Butler.from_config(butler_repo, collections=[collection])
    extract_visit_image(
        butler,
        os.path.join(
            testdata_dir,
            "dp2",
            "legacy",
            "visit_image.fits",
        ),
        find_dataset_or_raise(butler, "visit_image", **DP2_VISIT_DETECTOR_DATA_ID),
        shuffle=True,
        wcs_dataset_ref=find_dataset_or_raise(
            butler, "visit_summary", **DP2_VISIT_DETECTOR_DATA_ID, collections=wcs_collection
        ),
    )
    extract_visit_image(
        butler,
        os.path.join(
            testdata_dir,
            "dp2",
            "legacy",
            "preliminary_visit_image.fits",
        ),
        find_dataset_or_raise(butler, "preliminary_visit_image", **DP2_VISIT_DETECTOR_DATA_ID),
        shuffle=True,
    )
    extract_camera(
        butler,
        os.path.join(
            testdata_dir,
            "dp2",
            "legacy",
            "camera.fits",
        ),
        find_dataset_or_raise(butler, "camera", instrument="LSSTCam"),
    )


if __name__ == "__main__":
    extract_test_data()
