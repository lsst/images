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

__all__ = ("main",)

import click

from ..tests.extract_legacy_test_data import extract_test_data
from ..tests.verify_rewrite import verify_rewrite
from ._convert import convert
from ._diagram import diagram
from ._fuzz import fuzz_masked_image
from ._inspect import inspect
from ._minify import minify
from ._reformat import reformat
from ._schemas import schemas


@click.group(name="lsst-images-admin", context_settings={"help_option_names": ["-h", "--help"]})
def main() -> None:
    """Administrative tools for lsst.images files."""


main.add_command(convert)
main.add_command(diagram)
main.add_command(inspect)
main.add_command(minify)
main.add_command(reformat)
main.add_command(schemas)
main.add_command(extract_test_data, name="extract-test-data")
main.add_command(verify_rewrite, name="verify-rewrite")
main.add_command(fuzz_masked_image, name="fuzz-masked-image")
