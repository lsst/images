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

from lsst.afw.fits import (
    CompressionAlgorithm,
    CompressionOptions,
    DitherAlgorithm,
    QuantizationOptions,
    ScalingAlgorithm,
)
from lsst.daf.butler import Butler
from lsst.geom import Box2I, Extent2I, Point2I

VISIT_DETECTOR_DATA_ID = {"instrument": "HSC", "visit": 903334, "detector": 16}


def extract_visit_image(butler: Butler, testdata_dir: str) -> None:
    """Load a subimage of a PVI from the ci_hsc output repository and save it
    to testdata_images.
    """
    visit_image = butler.get(
        "pvi", VISIT_DETECTOR_DATA_ID, parameters={"bbox": Box2I(Point2I(5, 4), Extent2I(256, 250))}
    )
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
    visit_image.writeFits(
        os.path.join(testdata_dir, "extracted", "visit_image.fits"),
        imageOptions=float_compression,
        maskOptions=mask_compression,
        varianceOptions=float_compression,
    )


@click.command("extract_hsc")
@click.option("-b", "--butler-repo", help="Path to the ci_hsc (or equivalent) butler repository.")
@click.option("-d", "--testdata-dir", help="Path to the testdata_images directory.")
@click.option(
    "-c", "--collection", default="HSC/runs/ci_hsc", help="Collection name in the ci_hsc butler repository."
)
def main(butler_repo: str | None, testdata_dir: str | None, collection: str) -> None:
    """Extract test data from the ci_hsc output repository and save it to
    testdata_images.
    """
    if butler_repo is None:
        butler_repo = os.path.join(os.environ["CI_HSC_GEN3_DIR"], "DATA")
    if testdata_dir is None:
        testdata_dir = os.environ["TESTDATA_IMAGES_DIR"]
    butler = Butler.from_config(butler_repo, collections=[collection])
    extract_visit_image(butler, testdata_dir)


if __name__ == "__main__":
    main()
