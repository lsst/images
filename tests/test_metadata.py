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

from collections.abc import Mapping

import pytest

from lsst.images.serialization import EmptyExternalMetadata


def test_empty_external_metadata() -> None:
    """Test that EmptyExternalMetadata behaves as an empty read-only
    mapping.
    """
    external = EmptyExternalMetadata()
    assert isinstance(external, Mapping)
    assert len(external) == 0
    assert list(external) == []
    assert "EXPTIME" not in external
    assert external.get("EXPTIME") is None
    with pytest.raises(KeyError):
        external["EXPTIME"]
    with pytest.raises(KeyError):
        external.get_all("EXPTIME")
    assert repr(external) == "EmptyExternalMetadata({})"
