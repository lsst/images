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

import astropy.io.fits

from lsst.images.fits import header_from_legacy, header_to_legacy


def test_header_round_trip_through_legacy() -> None:
    """Test that an Astropy header converted to legacy metadata and back
    keeps all of its cards, including duplicated keywords and HISTORY and
    COMMENT cards.
    """
    header = astropy.io.fits.Header()
    header.append(("PLATFORM", "lsstcam"), end=True)
    # SubtractBackgroundTask writes one BGMEAN card per background fit.
    header.append(("BGMEAN", 1.5), end=True)
    header.append(("BGMEAN", 2.5), end=True)
    header.add_history("made by lsst_pipe")
    header.add_history("processed by cp_pipe")
    # Text containing "=" makes astropy treat this as a valueless card.
    header.add_comment("derived zeropoint = 27.3")
    header.append(astropy.io.fits.Card("", ""), end=True)  # blank card
    header["EXPTIME"] = 30.0

    legacy = header_to_legacy(header)
    # Duplicated keys keep all of their values in the legacy PropertyList.
    assert legacy.getArray("BGMEAN") == [1.5, 2.5]
    assert legacy.getArray("HISTORY") == ["made by lsst_pipe", "processed by cp_pipe"]

    round_tripped = header_from_legacy(legacy)
    # Blank cards are dropped; everything else keeps its keyword and value,
    # in order.
    expected = [(card.keyword, card.value) for card in header.cards if card.keyword]
    assert [(card.keyword, card.value) for card in round_tripped.cards] == expected
    # HISTORY and COMMENT cards must render without "= value" syntax.
    assert [str(card).rstrip() for card in round_tripped.cards if card.keyword == "HISTORY"] == [
        "HISTORY made by lsst_pipe",
        "HISTORY processed by cp_pipe",
    ]
    assert [str(card).rstrip() for card in round_tripped.cards if card.keyword == "COMMENT"] == [
        "COMMENT derived zeropoint = 27.3"
    ]


def test_legacy_metadata_round_trip_through_header() -> None:
    """Test that legacy metadata converted to an Astropy header and back
    keeps all of its values, including duplicated keys and HISTORY and
    COMMENT entries.
    """
    from lsst.daf.base import PropertyList

    metadata = PropertyList()
    metadata["PLATFORM"] = "lsstcam"
    metadata.add("BGMEAN", 1.5)
    metadata.add("BGMEAN", 2.5)
    metadata.add("HISTORY", "made by lsst_pipe")
    metadata.add("HISTORY", "processed by cp_pipe")
    metadata.add("COMMENT", "derived zeropoint = 27.3")
    metadata.add("BGMEAN", 3.5)

    header = header_from_legacy(metadata)
    # Each value becomes its own card, in the order the values were added.
    assert [(card.keyword, card.value) for card in header.cards] == [
        ("PLATFORM", "lsstcam"),
        ("BGMEAN", 1.5),
        ("BGMEAN", 2.5),
        ("BGMEAN", 3.5),
        ("HISTORY", "made by lsst_pipe"),
        ("HISTORY", "processed by cp_pipe"),
        ("COMMENT", "derived zeropoint = 27.3"),
    ]
    assert [str(card).rstrip() for card in header.cards if card.keyword == "HISTORY"] == [
        "HISTORY made by lsst_pipe",
        "HISTORY processed by cp_pipe",
    ]

    round_tripped = header_to_legacy(header)
    assert round_tripped.getArray("BGMEAN") == [1.5, 2.5, 3.5]
    assert round_tripped.getArray("HISTORY") == ["made by lsst_pipe", "processed by cp_pipe"]
    assert round_tripped.getArray("COMMENT") == ["derived zeropoint = 27.3"]
    assert round_tripped["PLATFORM"] == "lsstcam"


def _card80(image: str) -> str:
    """Pad a FITS card image to exactly 80 characters."""
    return image.ljust(80)[:80]


def test_header_to_legacy_skips_unrepresentable_cards() -> None:
    """Test that cards with values legacy metadata cannot represent are
    skipped instead of raising: unparsable values (real raws have headers
    with unterminated quoted strings), `Undefined` values (e.g. an empty
    ``SEEING =``), and comment-only value-less cards.
    """
    header = astropy.io.fits.Header.fromstring(
        "".join(
            [
                _card80("PLATFORM= 'lsstcam'"),
                _card80("SEEING  ="),
                _card80("FOO     = / comment only"),
                _card80("PROGRAM = 'HITS: real-time detection of &"),
                _card80("CONTINUE  'stellar explosions&'"),
                _card80("BGMEAN  = 1.5"),
                _card80("BGMEAN  = 2.5"),
            ]
        )
    )
    # The unparsable and undefined cards must at least be present.
    assert "PROGRAM" in header
    assert isinstance(header.cards["SEEING"].value, astropy.io.fits.card.Undefined)

    legacy = header_to_legacy(header)
    assert legacy["PLATFORM"] == "lsstcam"
    assert legacy.getArray("BGMEAN") == [1.5, 2.5]
    assert not legacy.exists("SEEING")
    assert not legacy.exists("FOO")
    assert not legacy.exists("PROGRAM")

    round_tripped = header_from_legacy(legacy)
    assert [(card.keyword, card.value) for card in round_tripped.cards] == [
        ("PLATFORM", "lsstcam"),
        ("BGMEAN", 1.5),
        ("BGMEAN", 2.5),
    ]
