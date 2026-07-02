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

from lsst.images.utils import round_half_away_from_zero, round_half_up


def test_round_half_away_from_zero() -> None:
    """Test round_half_away_from_zero against a range of values."""
    assert round_half_away_from_zero(-2.0) == -2
    assert round_half_away_from_zero(-1.7) == -2
    assert round_half_away_from_zero(-1.5) == -2
    assert round_half_away_from_zero(-1.2) == -1
    assert round_half_away_from_zero(-1.0) == -1
    assert round_half_away_from_zero(-0.7) == -1
    assert round_half_away_from_zero(-0.5) == -1
    assert round_half_away_from_zero(-0.2) == 0
    assert round_half_away_from_zero(0.0) == 0
    assert round_half_away_from_zero(0.2) == 0
    assert round_half_away_from_zero(0.5) == 1
    assert round_half_away_from_zero(0.7) == 1
    assert round_half_away_from_zero(1.0) == 1
    assert round_half_away_from_zero(1.2) == 1
    assert round_half_away_from_zero(1.5) == 2
    assert round_half_away_from_zero(1.7) == 2
    assert round_half_away_from_zero(2.0) == 2


def test_round_up() -> None:
    """Test round_half_up against a range of values."""
    assert round_half_up(-2.0) == -2
    assert round_half_up(-1.7) == -2
    assert round_half_up(-1.5) == -1
    assert round_half_up(-1.2) == -1
    assert round_half_up(-1.0) == -1
    assert round_half_up(-0.7) == -1
    assert round_half_up(-0.5) == 0
    assert round_half_up(-0.2) == 0
    assert round_half_up(0.0) == 0
    assert round_half_up(0.2) == 0
    assert round_half_up(0.5) == 1
    assert round_half_up(0.7) == 1
    assert round_half_up(1.0) == 1
    assert round_half_up(1.2) == 1
    assert round_half_up(1.5) == 2
    assert round_half_up(1.7) == 2
    assert round_half_up(2.0) == 2
