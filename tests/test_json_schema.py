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

import warnings

import pytest

from lsst.images import VisitImageSerializationModel
from lsst.images.cells import CellCoaddSerializationModel


def _check_json_schema(model: type, mode: str) -> None:
    """Assert that ``model.model_json_schema(mode=mode)`` succeeds without
    warnings and returns a well-formed object schema dict.
    """
    with warnings.catch_warnings():
        # Any warning emitted during schema generation (e.g.
        # PydanticJsonSchemaWarning for non-serializable defaults) is
        # treated as a test failure.
        warnings.simplefilter("error")
        schema = model.model_json_schema(mode=mode)
    assert isinstance(schema, dict)
    assert schema.get("type") == "object"
    assert "properties" in schema


@pytest.mark.parametrize("mode", ["validation", "serialization"])
def test_visit_image(mode: str) -> None:
    """Test that pydantic JSON Schema generation succeeds without warnings
    for VisitImageSerializationModel.
    """
    _check_json_schema(VisitImageSerializationModel, mode)


@pytest.mark.parametrize("mode", ["validation", "serialization"])
def test_cell_coadd(mode: str) -> None:
    """Test that pydantic JSON Schema generation succeeds without warnings
    for CellCoaddSerializationModel.
    """
    _check_json_schema(CellCoaddSerializationModel, mode)
