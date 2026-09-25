# External Metadata Access Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `GeneralizedImage.metadata` a `ChainMap`-based view that reads native (JSON-serialized) metadata first and falls back to a case-insensitive, read-only view of external metadata such as the primary FITS header.

**Architecture:** A format-neutral `ExternalMetadata` abstract `Mapping` lives in `lsst.images.serialization`, and the `OpaqueArchiveMetadata` protocol gains `external_metadata()` returning one.
`FitsOpaqueMetadata` implements it with `FitsExternalMetadata`, which wraps the primary `astropy.io.fits.Header` by reference.
`lsst.images` gains `NativeMetadata` (a `MutableMapping` over the existing `_metadata` dict that refuses new keys shadowing external ones) and `MetadataView` (a `collections.ChainMap` of the two), which `GeneralizedImage.metadata` builds on every access.

**Tech Stack:** Python 3.12+, `collections.ChainMap`, `astropy.io.fits`, pytest, ruff, mypy, LSST EUPS stack (`lsst_distrib`), optional `lsst.afw` and `h5py`.

**Spec:** `docs/superpowers/specs/2026-09-25-external-metadata-design.md`

## Global Constraints

- All commands that import `lsst.*` run through the EUPS wrapper: `~/.claude/skills/lsst-eups/scripts/lsst-run -l . -- <command>`, from the repository root. This plan writes it as `lsst-run`.
- All changes must be ruff clean (`lsst-run -l . -- ruff check python tests` and `lsst-run -l . -- ruff format --check python tests`) and mypy clean (`lsst-run -l . -- mypy python/lsst/images tests/test_metadata.py`).
- Line length 110; docstring and comment lines at most 79 characters; numpy docstring convention.
- Prefer top-level imports; function-scoped imports only for optional dependencies (`lsst.afw`).
- Comments and docstrings describe the code as it is, never the history of the change.
- Prose documents (Markdown, reStructuredText): one sentence per line, American English.
- Commit messages follow the repository style (an imperative sentence, no `feat:` prefix) and end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- Never push.
- `COMMENT`, `HISTORY`, and blank-keyword cards are never visible through `external`.
- Only the primary header is exposed.
- For a repeated keyword, which value `[]` returns is unspecified; `get_all` returns every value in source order.
- `ExternalMetadataValue = bool | int | float | complex | str | None`.
- Tests build their own data; no new files in `testdata_images`.

## Review Focus

1. Native keys that are not valid FITS keywords (`"roundtrip_test_1"`, `"a=b"`, `"é"`, non-`str` keys) must make external lookups return "absent", never raise anything but `KeyError`. Pinned in Task 2 (`test_fits_external_metadata_odd_keys`) and Task 4 (`test_native_metadata_accepts_non_fits_keys`).
2. HIERARCH keywords that astropy stores in lower case (`HIERARCH lower key`) must be reported in upper case by iteration and still be found by `[]`, `in`, and `get_all`. Pinned in Task 2 (`test_fits_external_metadata_lookup`, `test_fits_external_metadata_iteration`).
3. Inherited `ChainMap` operations (`|`, reversed `|`, `copy.copy`, `new_child`, `parents`, `pop` with a default) must either work sensibly or raise `TypeError`, never `AttributeError` from the `copy()` override. Pinned in Task 4 (`test_metadata_view_chainmap_operations`).
4. Internal code touching the `id` key must not fall through to, or be blocked by, an external `ID` card. Pinned in Task 6 (`test_external_metadata_legacy_round_trip` adds an `ID` card).
5. Assigning one image's `metadata` view to another image must share the underlying native dict, not nest a view. Pinned in Task 5 (`test_metadata_setter`).

---

### Task 1: `ExternalMetadata` base class

**Files:**
- Create: `python/lsst/images/serialization/_external_metadata.py`
- Modify: `python/lsst/images/serialization/__init__.py` (add a star import)
- Create: `tests/test_metadata.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `lsst.images.serialization.ExternalMetadataValue` (type alias `bool | int | float | complex | str | None`).
  - `lsst.images.serialization.ExternalMetadata(Mapping[str, ExternalMetadataValue])`, abstract, with `get_all(self, key: str) -> tuple[ExternalMetadataValue, ...]` and a `__repr__` of the form `ClassName({...})`.
  - `lsst.images.serialization.EmptyExternalMetadata()`, a concrete `ExternalMetadata` with no keys.

- [ ] **Step 0: Record the lint/type baseline**

Run: `lsst-run -l . -- mypy python/lsst/images` and `lsst-run -l . -- ruff check python tests`
Expected: both clean. If not, note the pre-existing failures so they are not attributed to this work.

- [ ] **Step 1: Write the failing test**

Create `tests/test_metadata.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `lsst-run -l . -- pytest tests/test_metadata.py -v`
Expected: FAIL with `ImportError: cannot import name 'EmptyExternalMetadata'`.

- [ ] **Step 3: Write the implementation**

Create `python/lsst/images/serialization/_external_metadata.py`:

```python
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

__all__ = ("EmptyExternalMetadata", "ExternalMetadata", "ExternalMetadataValue")

from abc import abstractmethod
from collections.abc import Iterator, Mapping

type ExternalMetadataValue = bool | int | float | complex | str | None


class ExternalMetadata(Mapping[str, ExternalMetadataValue]):
    """Read-only access to metadata that was carried in from an external
    source, such as the primary header of a FITS file, and is not part of
    the data model.

    Notes
    -----
    Lookups and membership tests are case-insensitive.  Iteration yields
    each key once, in the form the underlying source reports it.  When a key
    is repeated in the source, which of its values ``[]`` returns is
    unspecified; use `get_all` to obtain every value.
    """

    @abstractmethod
    def get_all(self, key: str) -> tuple[ExternalMetadataValue, ...]:
        """Return every value for a key, in source order.

        Parameters
        ----------
        key
            Key to look up (case-insensitive).

        Returns
        -------
        values : `tuple`
            All values for ``key``.

        Raises
        ------
        KeyError
            Raised if ``key`` is not present.
        """
        raise NotImplementedError()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({dict(self)!r})"


class EmptyExternalMetadata(ExternalMetadata):
    """An `ExternalMetadata` with no keys."""

    def __getitem__(self, key: str) -> ExternalMetadataValue:
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0

    def get_all(self, key: str) -> tuple[ExternalMetadataValue, ...]:
        # Docstring inherited.
        raise KeyError(key)
```

In `python/lsst/images/serialization/__init__.py`, add the import in alphabetical position, after `from ._dtypes import *`:

```python
from ._external_metadata import *
```

- [ ] **Step 4: Run test to verify it passes**

Run: `lsst-run -l . -- pytest tests/test_metadata.py -v`
Expected: PASS.

- [ ] **Step 5: Lint and type-check**

Run: `lsst-run -l . -- ruff check python tests && lsst-run -l . -- ruff format --check python tests && lsst-run -l . -- mypy python/lsst/images tests/test_metadata.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add python/lsst/images/serialization/_external_metadata.py python/lsst/images/serialization/__init__.py tests/test_metadata.py
git commit -m "Add a read-only ExternalMetadata mapping

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: FITS implementation and the protocol hook

**Files:**
- Modify: `python/lsst/images/serialization/_common.py` (`OpaqueArchiveMetadata` protocol, around line 484; imports)
- Modify: `python/lsst/images/fits/_common.py` (`__all__`, imports, new class before `FitsOpaqueMetadata` at line 335, new method on `FitsOpaqueMetadata`)
- Test: `tests/test_metadata.py`

**Interfaces:**
- Consumes: `ExternalMetadata`, `ExternalMetadataValue`, `EmptyExternalMetadata` from Task 1.
- Produces:
  - `OpaqueArchiveMetadata.external_metadata(self) -> ExternalMetadata` (protocol member).
  - `lsst.images.fits.FitsExternalMetadata(header: astropy.io.fits.Header | None)`.
  - `FitsOpaqueMetadata.external_metadata(self) -> FitsExternalMetadata`, wrapping `self.headers.get(ExtensionKey())`.
  - Test helper `_make_header() -> astropy.io.fits.Header` in `tests/test_metadata.py`, used by later tasks. Its external keys, in order, are `EXPTIME` (30.0), `BGMEAN` (1.5, 2.5), `LSST ISR UNITS` ("adu"), `LOWER KEY` (3), `NOVAL` (None), `CPLX` (1+2j); it also holds `COMMENT`, `HISTORY`, and a blank card.

- [ ] **Step 1: Write the failing tests**

Add to the imports of `tests/test_metadata.py`:

```python
import warnings

import astropy.io.fits

from lsst.images.fits import ExtensionKey, FitsExternalMetadata, FitsOpaqueMetadata
```

Append to `tests/test_metadata.py`:

```python
EXTERNAL_KEYS = ["EXPTIME", "BGMEAN", "LSST ISR UNITS", "LOWER KEY", "NOVAL", "CPLX"]


def _make_header() -> astropy.io.fits.Header:
    """Return a primary header exercising repeated, HIERARCH, commentary,
    valueless, and complex cards.
    """
    header = astropy.io.fits.Header()
    header.append(("EXPTIME", 30.0), end=True)
    header.append(("BGMEAN", 1.5), end=True)
    header.append(("BGMEAN", 2.5), end=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", astropy.io.fits.verify.VerifyWarning)
        header.append(("HIERARCH LSST ISR UNITS", "adu"), end=True)
        # astropy preserves the case of HIERARCH keywords.
        header.append(("HIERARCH lower key", 3), end=True)
    header.append(("COMMENT", "a comment"), end=True)
    header.append(("HISTORY", "a history"), end=True)
    header.append(("", "blank"), end=True)
    header.append(astropy.io.fits.Card.fromstring("NOVAL   =".ljust(80)), end=True)
    header.append(("CPLX", 1 + 2j), end=True)
    return header


def test_fits_external_metadata_lookup() -> None:
    """Test case-insensitive lookup, repeated keywords, HIERARCH keywords,
    and value conversion.
    """
    external = FitsExternalMetadata(_make_header())
    assert external["EXPTIME"] == 30.0
    assert external["exptime"] == 30.0
    assert external["lsst isr units"] == "adu"
    assert external["HIERARCH LSST ISR UNITS"] == "adu"
    assert external["LOWER KEY"] == 3
    assert external["lower key"] == 3
    assert external["NOVAL"] is None
    assert external["CPLX"] == 1 + 2j
    assert external["BGMEAN"] in (1.5, 2.5)
    assert external.get_all("bgmean") == (1.5, 2.5)
    assert external.get_all("exptime") == (30.0,)
    assert external.get_all("NOVAL") == (None,)
    assert external.get_all("Lower Key") == (3,)
    assert "Exptime" in external
    assert "lower key" in external


def test_fits_external_metadata_hides_commentary_cards() -> None:
    """Test that COMMENT, HISTORY, and blank cards are not visible."""
    external = FitsExternalMetadata(_make_header())
    for key in ("COMMENT", "comment", "HISTORY", ""):
        assert key not in external
        with pytest.raises(KeyError):
            external[key]
        with pytest.raises(KeyError):
            external.get_all(key)


def test_fits_external_metadata_odd_keys() -> None:
    """Test that keys that cannot be FITS keywords are simply absent."""
    external = FitsExternalMetadata(_make_header())
    for key in ("roundtrip_test_1", "a=b", "é", "x" * 100):
        assert key not in external
        with pytest.raises(KeyError):
            external[key]
    assert 1 not in external


def test_fits_external_metadata_iteration() -> None:
    """Test that iteration yields each upper-case keyword once."""
    external = FitsExternalMetadata(_make_header())
    assert list(external) == EXTERNAL_KEYS
    assert len(external) == len(EXTERNAL_KEYS)
    assert repr(external).startswith("FitsExternalMetadata({'EXPTIME': 30.0")


def test_fits_external_metadata_empty() -> None:
    """Test that a missing header behaves as an empty one."""
    external = FitsExternalMetadata(None)
    assert len(external) == 0
    assert "EXPTIME" not in external


def test_fits_opaque_metadata_external_metadata() -> None:
    """Test that only the primary header is exposed."""
    opaque_metadata = FitsOpaqueMetadata()
    assert len(opaque_metadata.external_metadata()) == 0
    opaque_metadata.add_header(_make_header(), name="", ver=1)
    extension_header = astropy.io.fits.Header()
    extension_header["EXTRA"] = 1
    opaque_metadata.add_header(extension_header, name="IMAGE", ver=1)
    external = opaque_metadata.external_metadata()
    assert list(external) == EXTERNAL_KEYS
    assert "EXTRA" not in external
    assert opaque_metadata.headers[ExtensionKey("IMAGE")]["EXTRA"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `lsst-run -l . -- pytest tests/test_metadata.py -v`
Expected: FAIL with `ImportError: cannot import name 'FitsExternalMetadata'`.

- [ ] **Step 3: Add the protocol member**

In `python/lsst/images/serialization/_common.py`, add a top-level import next to the other package-relative imports (after `from ._migrations import ...`):

```python
from ._external_metadata import ExternalMetadata
```

Add this method to `OpaqueArchiveMetadata`, after `subset`:

```python
    def external_metadata(self) -> ExternalMetadata:
        """Return a read-only view of the metadata carried by this object
        that is not part of the data model.
        """
        ...
```

- [ ] **Step 4: Implement `FitsExternalMetadata`**

In `python/lsst/images/fits/_common.py`:

Add `"FitsExternalMetadata"` to `__all__`, keeping it sorted (between `"FitsDitherAlgorithm"` and `"FitsOpaqueMetadata"`).

Replace the serialization import line with:

```python
from ..serialization import (
    ArchiveReadError,
    ExternalMetadata,
    ExternalMetadataValue,
    OpaqueArchiveMetadata,
    TableColumnModel,
)
```

Add, immediately before the `@final` decorator of `FitsOpaqueMetadata`:

```python
_HIDDEN_KEYWORDS = frozenset({"", "COMMENT", "HISTORY"})
"""Keywords of FITS commentary cards, which `FitsExternalMetadata` hides."""


def _from_card_value(value: Any) -> ExternalMetadataValue:
    """Convert a FITS card value to an `ExternalMetadataValue`, mapping the
    value of a card with no value to `None`.
    """
    if isinstance(value, astropy.io.fits.card.Undefined):
        return None
    return value


@final
class FitsExternalMetadata(ExternalMetadata):
    """Read-only access to the cards of a FITS header by keyword.

    Parameters
    ----------
    header
        Header to wrap.  It is held by reference and must not be modified
        while this object is in use.  `None` is equivalent to an empty
        header.

    Notes
    -----
    ``COMMENT``, ``HISTORY``, and blank-keyword cards are not visible.
    Keywords are reported in upper case, with HIERARCH keywords lacking the
    ``HIERARCH`` prefix (e.g. ``LSST ISR UNITS``).  Cards with no value are
    reported as `None`.
    """

    def __init__(self, header: astropy.io.fits.Header | None) -> None:
        self._header = header if header is not None else astropy.io.fits.Header()

    def _find(self, key: object) -> str:
        """Return the normalized keyword for ``key``, raising `KeyError` if
        the header has no visible card with that keyword.
        """
        if isinstance(key, str):
            keyword = astropy.io.fits.Card.normalize_keyword(key)
            if keyword not in _HIDDEN_KEYWORDS and keyword in self._header:
                return keyword
        raise KeyError(key)

    def __getitem__(self, key: str) -> ExternalMetadataValue:
        return _from_card_value(self._header[self._find(key)])

    def __iter__(self) -> Iterator[str]:
        keywords = (keyword.upper() for keyword in self._header.keys())
        return iter(dict.fromkeys(keyword for keyword in keywords if keyword not in _HIDDEN_KEYWORDS))

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def get_all(self, key: str) -> tuple[ExternalMetadataValue, ...]:
        # Docstring inherited.
        keyword = self._find(key)
        return tuple(
            _from_card_value(card.value) for card in self._header.cards if card.keyword.upper() == keyword
        )
```

Add this method to `FitsOpaqueMetadata`, after `subset`:

```python
    def external_metadata(self) -> FitsExternalMetadata:
        # Docstring inherited.
        return FitsExternalMetadata(self.headers.get(ExtensionKey()))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `lsst-run -l . -- pytest tests/test_metadata.py -v`
Expected: PASS.

- [ ] **Step 6: Lint, type-check, and run the FITS-related suites**

Run: `lsst-run -l . -- ruff check python tests && lsst-run -l . -- ruff format --check python tests && lsst-run -l . -- mypy python/lsst/images tests/test_metadata.py`
Expected: clean.
Run: `lsst-run -l . -- pytest tests/test_fits_output_archive.py tests/test_ndf_input_archive.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add python/lsst/images/serialization/_common.py python/lsst/images/fits/_common.py tests/test_metadata.py
git commit -m "Expose the primary FITS header as external metadata

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Restore native metadata from legacy files and strip its cards from the opaque header

The `LSST IMAGES KEY n` / `LSST IMAGES VALUE n` cards hold native metadata in legacy files.
`extract_legacy_primary_header` returns them as a dict, but copies the header before removing them, so they also stay in the opaque header.
`Image.read_legacy`, `Mask.read_legacy`, and `MaskedImage._read_legacy_hdus` ignore the returned dict; only the `VisitImage` readers use it.
Fixing only the stripping would make those three readers lose the native keys entirely, so both changes land together.

`MaskedImage.to_legacy` cannot write these cards itself (it returns an afw `MaskedImageF`, which has no metadata), but `MaskedImageF.writeFits` accepts primary-header metadata, which the test uses to build such a file.

**Files:**
- Modify: `python/lsst/images/fits/_common.py` (`FitsOpaqueMetadata.extract_legacy_primary_header`, around line 419)
- Modify: `python/lsst/images/_image.py` (`Image.read_legacy`, around lines 506-523)
- Modify: `python/lsst/images/_mask.py` (`Mask.read_legacy`, around lines 1154-1162)
- Modify: `python/lsst/images/_masked_image.py` (`MaskedImage._read_legacy_hdus`, around lines 504-549)
- Test: `tests/test_metadata.py`

**Interfaces:**
- Consumes: `FitsOpaqueMetadata`, `ExtensionKey`; `MaskedImage._fill_legacy_metadata(legacy_metadata: PropertyList) -> None`.
- Produces:
  - `extract_legacy_primary_header` keeps its signature `(self, header: astropy.io.fits.Header) -> dict[str, Any]` and its destructive removal of the cards from `header`, and additionally leaves no `LSST IMAGES` cards in `self.headers[ExtensionKey()]`.
  - `Image.read_legacy`, `Mask.read_legacy`, and `MaskedImage.read_legacy` (including component reads) set the result's native metadata (`_metadata`) from those cards. When `VisitImage` passes its own `opaque_metadata` into `MaskedImage._read_legacy_hdus`, it has already extracted the dict, and that path is unchanged.

- [ ] **Step 1: Write the failing tests**

Add to the imports of `tests/test_metadata.py`:

```python
from pathlib import Path

import numpy as np

from lsst.images import Image, Mask, MaskedImage, MaskPlane, MaskSchema
from lsst.images.tests import reset_afw_mask_planes  # noqa: F401
```

(merge the `lsst.images` names with any already imported, sorted.)

Append to `tests/test_metadata.py`:

```python
def test_extract_legacy_primary_header_strips_native_cards() -> None:
    """Test that the cards holding native metadata in a legacy file are
    returned as native metadata and not kept in the opaque header.
    """
    header = astropy.io.fits.Header()
    header.append(("PLATFORM", "lsstcam"), end=True)
    header.append(("HIERARCH LSST IMAGES KEY 1", "native_key"), end=True)
    header.append(("HIERARCH LSST IMAGES VALUE 1", 7), end=True)
    header.append(("HIERARCH LSST IMAGES KEY 2", "MixedCase"), end=True)
    header.append(("HIERARCH LSST IMAGES VALUE 2", "yes"), end=True)
    opaque_metadata = FitsOpaqueMetadata()
    assert opaque_metadata.extract_legacy_primary_header(header) == {"native_key": 7, "MixedCase": "yes"}
    stored = opaque_metadata.headers[ExtensionKey()]
    assert stored["PLATFORM"] == "lsstcam"
    assert not [keyword for keyword in stored if keyword.startswith("LSST IMAGES")]
    assert list(opaque_metadata.external_metadata()) == ["PLATFORM"]


def test_legacy_readers_restore_native_metadata(
    tmp_path: Path,
    reset_afw_mask_planes: None,  # noqa: F811
) -> None:
    """Test that every legacy reader restores native metadata from the
    ``LSST IMAGES`` cards and does not keep those cards as opaque metadata.
    """
    from lsst.daf.base import PropertyList

    masked_image = MaskedImage(
        Image(1.0, shape=(4, 5), dtype=np.float32),
        mask_schema=MaskSchema([MaskPlane("BAD", "Pixel is bad.")]),
        metadata={"native_key": 7, "MixedCase": "yes"},
    )
    legacy_metadata = PropertyList()
    masked_image._fill_legacy_metadata(legacy_metadata)
    path = tmp_path / "legacy_masked_image.fits"
    masked_image.to_legacy().writeFits(str(path), metadata=legacy_metadata)
    with astropy.io.fits.open(path) as hdu_list:
        assert hdu_list[0].header["LSST IMAGES KEY 1"] == "native_key"
    results = {
        "MaskedImage": MaskedImage.read_legacy(path),
        "MaskedImage image component": MaskedImage.read_legacy(path, component="image"),
        "MaskedImage mask component": MaskedImage.read_legacy(path, component="mask"),
        "Image": Image.read_legacy(path),
        "Mask": Mask.read_legacy(path, ext=2),
    }
    for label, result in results.items():
        assert result._metadata == {"native_key": 7, "MixedCase": "yes"}, label
        opaque_header = result._opaque_metadata.headers[ExtensionKey()]
        assert not [keyword for keyword in opaque_header if keyword.startswith("LSST IMAGES")], label
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `lsst-run -l . -- pytest tests/test_metadata.py -v -k "legacy"`
Expected: `test_extract_legacy_primary_header_strips_native_cards` FAILS on the `assert not [keyword ...]` line; `test_legacy_readers_restore_native_metadata` FAILS on `assert result._metadata == {...}` with `{}` for `MaskedImage` (or skips if afw is unavailable, in which case the implementer must use an environment with afw).

- [ ] **Step 3: Fix the extraction order**

In `extract_legacy_primary_header`, move the extraction loop so it runs before the header is copied. The method body becomes:

```python
        metadata: dict[str, Any] = {}
        for n in itertools.count():
            if (key := header.pop(f"LSST IMAGES KEY {n + 1}", ...)) is ...:
                break
            value = header.pop(f"LSST IMAGES VALUE {n + 1}")
            metadata[key] = value
        primary_header = header.copy(strip=True)
        # No idea what these spare TAN-SIP headers are doing in the afw
        # FITS files, but we'll strip them here:
        primary_header.remove("A_ORDER", ignore_missing=True)
        primary_header.remove("B_ORDER", ignore_missing=True)
        primary_header.remove("DATE", ignore_missing=True)
        strip_legacy_exposure_cards(primary_header)
        strip_butler_cards(primary_header)
        self.headers[ExtensionKey()] = primary_header
        return metadata
```

- [ ] **Step 4: Restore native metadata in the three legacy readers**

In `Image.read_legacy` (`python/lsst/images/_image.py`), replace

```python
            opaque_metadata.extract_legacy_primary_header(hdu_list[0].header)
```

with

```python
            native_metadata = opaque_metadata.extract_legacy_primary_header(hdu_list[0].header)
```

and after `result._opaque_metadata = opaque_metadata` add

```python
            result._metadata = native_metadata
```

Make the same two changes in `Mask.read_legacy` (`python/lsst/images/_mask.py`).

In `MaskedImage._read_legacy_hdus` (`python/lsst/images/_masked_image.py`), replace

```python
        if opaque_metadata is None:
            opaque_metadata = fits.FitsOpaqueMetadata()
            opaque_metadata.extract_legacy_primary_header(hdu_list[0].header)
```

with

```python
        # A caller that passes opaque_metadata has already extracted the
        # native metadata from the primary header itself.
        native_metadata: dict[str, Any] | None = None
        if opaque_metadata is None:
            opaque_metadata = fits.FitsOpaqueMetadata()
            native_metadata = opaque_metadata.extract_legacy_primary_header(hdu_list[0].header)
```

and replace the end of the method

```python
        result._opaque_metadata = opaque_metadata
        return result
```

with

```python
        result._opaque_metadata = opaque_metadata
        if native_metadata is not None:
            result._metadata = native_metadata
        return result
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `lsst-run -l . -- pytest tests/test_metadata.py tests/test_image.py tests/test_mask.py tests/test_masked_image.py tests/test_visit_image.py -q`
Expected: PASS (tests needing `TESTDATA_IMAGES_DIR` may skip).

- [ ] **Step 6: Lint and commit**

Run: `lsst-run -l . -- ruff check python tests && lsst-run -l . -- ruff format --check python tests && lsst-run -l . -- mypy python/lsst/images tests/test_metadata.py`
Expected: clean.

```bash
git add python/lsst/images/fits/_common.py python/lsst/images/_image.py python/lsst/images/_mask.py python/lsst/images/_masked_image.py tests/test_metadata.py
git commit -m "Restore native metadata from legacy files in every legacy reader

The LSST IMAGES cards were returned to the VisitImage readers only and
were also left behind in the opaque primary header.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `NativeMetadata` and `MetadataView`

**Files:**
- Create: `python/lsst/images/_metadata.py`
- Modify: `python/lsst/images/__init__.py` (add `from ._metadata import *` after `from ._masked_image import *`)
- Test: `tests/test_metadata.py`

**Interfaces:**
- Consumes: `ExternalMetadata`, `ExternalMetadataValue`, `MetadataValue` from `lsst.images.serialization`; `FitsExternalMetadata` and `_make_header` (tests only).
- Produces:
  - `lsst.images.NativeMetadata(data: dict[str, MetadataValue], external: ExternalMetadata)`, a `MutableMapping[str, MetadataValue]`. `__setitem__` raises `KeyError` when `key not in data and key in external`. The wrapped dict is available to package code as `NativeMetadata._data`.
  - `lsst.images.MetadataView(native: NativeMetadata, external: ExternalMetadata)`, a `ChainMap[str, MetadataValue | ExternalMetadataValue]` with properties `native -> NativeMetadata` and `external -> ExternalMetadata`, `get_all(key) -> tuple[...]`, `copy() -> dict[str, MetadataValue]` (also `__copy__`), `__or__`/`__ror__` returning plain dicts, and `new_child`/`parents` raising `TypeError`.

- [ ] **Step 1: Write the failing tests**

Add to the imports of `tests/test_metadata.py`:

```python
import copy

from lsst.images import MetadataView, NativeMetadata
```

Append to `tests/test_metadata.py`:

```python
def _make_view(data: dict) -> MetadataView:
    external = FitsExternalMetadata(_make_header())
    return MetadataView(NativeMetadata(data, external), external)


def test_native_metadata_shadowing() -> None:
    """Test that new native keys may not shadow external keys, while keys
    already present may be updated.
    """
    external = FitsExternalMetadata(_make_header())
    data: dict = {"exptime": 1.0}
    native = NativeMetadata(data, external)
    native["exptime"] = 2.0
    assert data == {"exptime": 2.0}
    for key in ("EXPTIME", "Bgmean", "lsst isr units", "LOWER KEY", "noval"):
        with pytest.raises(KeyError):
            native[key] = 3.0
    with pytest.raises(KeyError):
        native.update({"cplx": 1})
    with pytest.raises(KeyError):
        native.setdefault("NOVAL", 1)
    native["fresh"] = 1
    del native["fresh"]
    assert data == {"exptime": 2.0}
    assert dict(native) == {"exptime": 2.0}
    assert len(native) == 1


def test_native_metadata_accepts_non_fits_keys() -> None:
    """Test that keys that cannot be FITS keywords are accepted."""
    external = FitsExternalMetadata(_make_header())
    data: dict = {}
    native = NativeMetadata(data, external)
    for key in ("roundtrip_test_1", "a=b", "é", "MixedCaseKey"):
        native[key] = 1
    assert set(data) == {"roundtrip_test_1", "a=b", "é", "MixedCaseKey"}


def test_metadata_view_lookup() -> None:
    """Test that native keys are found first with exact case, then external
    keys case-insensitively.
    """
    view = _make_view({"native_key": 7, "exptime": 1.0})
    assert view["native_key"] == 7
    assert view["exptime"] == 1.0
    assert view["EXPTIME"] == 30.0
    assert view["ExpTime"] == 30.0
    assert view["BGMEAN"] in (1.5, 2.5)
    assert view["noval"] is None
    assert "lsst isr units" in view
    assert view.get("missing") is None
    assert view.get_all("BGMEAN") == (1.5, 2.5)
    assert view.get_all("exptime") == (1.0,)
    assert view.get_all("EXPTIME") == (30.0,)
    for key in ("COMMENT", "HISTORY", ""):
        assert key not in view
    with pytest.raises(KeyError):
        view["missing"]
    with pytest.raises(KeyError):
        view.get_all("missing")


def test_metadata_view_iteration() -> None:
    """Test that iteration is the exact-case union of both sources and
    agrees with lookup.
    """
    view = _make_view({"native_key": 7, "exptime": 1.0})
    keys = list(view)
    assert sorted(keys) == sorted(["native_key", "exptime", *EXTERNAL_KEYS])
    assert len(view) == len(keys) == len(set(keys))
    as_dict = dict(view)
    assert as_dict["exptime"] == 1.0
    assert as_dict["EXPTIME"] == 30.0
    for key in keys:
        assert as_dict[key] == view[key]
    assert view == as_dict


def test_metadata_view_writes() -> None:
    """Test that writes and deletes go to native and respect shadowing."""
    data: dict = {"native_key": 7}
    view = _make_view(data)
    view["new"] = 1
    assert data == {"native_key": 7, "new": 1}
    with pytest.raises(KeyError):
        view["cplx"] = 1
    with pytest.raises(KeyError):
        del view["EXPTIME"]
    with pytest.raises(KeyError):
        view.pop("EXPTIME")
    assert view.pop("missing", None) is None
    assert view.pop("new") == 1
    view |= {"another": 2}
    assert data == {"native_key": 7, "another": 2}
    view.clear()
    assert data == {}
    assert view["EXPTIME"] == 30.0


def test_metadata_view_chainmap_operations() -> None:
    """Test the ChainMap operations that the view overrides."""
    data: dict = {"native_key": 7}
    view = _make_view(data)
    copied = view.copy()
    assert type(copied) is dict
    assert copied == data
    assert copied is not data
    shallow = copy.copy(view)
    assert type(shallow) is dict
    assert shallow == data
    merged = view | {"z": 1}
    assert type(merged) is dict
    assert merged == {**dict(view), "z": 1}
    reverse_merged = {"z": 1} | view
    assert type(reverse_merged) is dict
    assert reverse_merged == {"z": 1, **dict(view)}
    with pytest.raises(TypeError):
        view.new_child()
    with pytest.raises(TypeError):
        view.parents
    assert view.native is not None
    assert list(view.external) == EXTERNAL_KEYS
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `lsst-run -l . -- pytest tests/test_metadata.py -v`
Expected: FAIL with `ImportError: cannot import name 'MetadataView'`.

- [ ] **Step 3: Write the implementation**

Create `python/lsst/images/_metadata.py`:

```python
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

__all__ = ("MetadataView", "NativeMetadata")

from collections import ChainMap
from collections.abc import Iterator, Mapping, MutableMapping
from typing import Any, NoReturn, cast

from .serialization import ExternalMetadata, ExternalMetadataValue, MetadataValue


class NativeMetadata(MutableMapping[str, MetadataValue]):
    """The flexible metadata that is part of an image's data model.

    Parameters
    ----------
    data
        Dictionary holding the metadata.  It is held by reference, so changes
        are visible to every object sharing it.
    external
        External metadata that new keys may not shadow.

    Notes
    -----
    Keys are case-sensitive.  Adding a key that case-insensitively matches a
    key of ``external`` raises `KeyError`; a key that is already present may
    always be updated.
    """

    def __init__(self, data: dict[str, MetadataValue], external: ExternalMetadata) -> None:
        self._data = data
        self._external = external

    def __getitem__(self, key: str) -> MetadataValue:
        return self._data[key]

    def __setitem__(self, key: str, value: MetadataValue) -> None:
        if key not in self._data and key in self._external:
            raise KeyError(
                f"Metadata key {key!r} would shadow a key in the external metadata "
                "(external keys are case-insensitive)."
            )
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"NativeMetadata({self._data!r})"


class MetadataView(ChainMap[str, MetadataValue | ExternalMetadataValue]):
    """Combined view of an image's native and external metadata.

    Parameters
    ----------
    native
        Metadata that is part of the image's data model.
    external
        Read-only metadata carried in from an external source.

    Notes
    -----
    Lookups check ``native`` first, matching case exactly, and then
    ``external``, ignoring case.  Writes and deletes go to ``native`` only,
    and adding a new key that case-insensitively matches an ``external`` key
    raises `KeyError`.  Iteration yields the exact-case union of the keys of
    both.

    When a key is repeated in the external source, which of its values
    ``[]`` returns is unspecified; use `get_all` to obtain every value.
    """

    def __init__(self, native: NativeMetadata, external: ExternalMetadata) -> None:
        # ChainMap only ever writes to its first mapping, so the read-only
        # external mapping is never mutated through it.
        super().__init__(
            cast(MutableMapping[str, MetadataValue | ExternalMetadataValue], native),
            cast(MutableMapping[str, MetadataValue | ExternalMetadataValue], external),
        )
        self._native = native
        self._external = external

    @property
    def native(self) -> NativeMetadata:
        """Metadata that is part of the image's data model and is saved with
        it (`NativeMetadata`).
        """
        return self._native

    @property
    def external(self) -> ExternalMetadata:
        """Read-only metadata carried in from an external source, such as
        the primary header of a FITS file (`.serialization.ExternalMetadata`).
        """
        return self._external

    def get_all(self, key: str) -> tuple[MetadataValue | ExternalMetadataValue, ...]:
        """Return every value for a key.

        Parameters
        ----------
        key
            Key to look up.

        Returns
        -------
        values : `tuple`
            A single-element tuple holding the native value if ``key`` is a
            native key, and otherwise every external value for ``key``, in
            source order.

        Raises
        ------
        KeyError
            Raised if ``key`` is not present.
        """
        if key in self._native:
            return (self._native[key],)
        return self._external.get_all(key)

    def copy(self) -> dict[str, MetadataValue]:  # type: ignore[override]
        """Return a plain `dict` copy of the native metadata.

        Returns
        -------
        metadata : `dict`
            A copy of the native metadata.
        """
        return dict(self._native)

    __copy__ = copy

    def __or__(self, other: Mapping[str, Any]) -> dict[str, Any]:  # type: ignore[override]
        return {**self, **other}

    def __ror__(self, other: Mapping[str, Any]) -> dict[str, Any]:  # type: ignore[override]
        return {**other, **self}

    def new_child(self, m: Any = None, **kwargs: Any) -> NoReturn:  # type: ignore[override]
        """Not supported.

        Parameters
        ----------
        m
            Ignored.
        **kwargs
            Ignored.

        Raises
        ------
        TypeError
            Always raised.
        """
        raise TypeError("MetadataView does not support new_child.")

    @property
    def parents(self) -> NoReturn:
        """Not supported; always raises `TypeError`."""
        raise TypeError("MetadataView does not support parents.")
```

If mypy reports any of the `# type: ignore[override]` comments as unused (the repository sets `warn_unused_ignores`), remove that comment; if it reports an override error on a method without one, add it.

In `python/lsst/images/__init__.py`, add after `from ._masked_image import *`:

```python
from ._metadata import *
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `lsst-run -l . -- pytest tests/test_metadata.py -v`
Expected: PASS.

- [ ] **Step 5: Lint and type-check**

Run: `lsst-run -l . -- ruff check python tests && lsst-run -l . -- ruff format --check python tests && lsst-run -l . -- mypy python/lsst/images tests/test_metadata.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add python/lsst/images/_metadata.py python/lsst/images/__init__.py tests/test_metadata.py
git commit -m "Add a ChainMap view over native and external metadata

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Wire the view into `GeneralizedImage.metadata`

**Files:**
- Modify: `python/lsst/images/_generalized_image.py` (imports; `metadata` property and setter at lines 170-185)
- Modify: `python/lsst/images/_image.py:364`, `python/lsst/images/_mask.py:993`, `python/lsst/images/_masked_image.py:293` and `:581`, `python/lsst/images/_color_image.py:214`, `python/lsst/images/cells/_coadd.py:460`, `python/lsst/images/_visit_image.py:744`, `:774`, `:969`, `python/lsst/images/tests/_minify_for_fixtures.py:231`
- Modify: `python/lsst/images/tests/_checks.py:304-309`, `:354`, `:387`
- Modify: `doc/user-guide/for-afw-users.rst` (Exposure component list, around line 150)
- Create: `doc/changes/DM-54770.api.md`, `doc/changes/DM-54770.feature.md`
- Test: `tests/test_metadata.py`

**Interfaces:**
- Consumes: `MetadataView`, `NativeMetadata` (Task 4); `EmptyExternalMetadata` (Task 1); `OpaqueArchiveMetadata.external_metadata()` (Task 2); `_make_header`, `EXTERNAL_KEYS` (tests).
- Produces: `GeneralizedImage.metadata -> MetadataView`; setter accepting `Mapping[str, MetadataValue] | MetadataView`. Internal code uses `self._metadata` (the plain dict) wherever it needs native metadata.

- [ ] **Step 1: Write the failing tests**

Add `Box` to the existing `from lsst.images import ...` line of `tests/test_metadata.py` (sorted); `numpy` and `Image` are already imported by Task 3.

Append to `tests/test_metadata.py`:

```python
def _make_image(metadata: dict | None = None) -> Image:
    """Return a small image whose opaque metadata holds `_make_header`."""
    image = Image(0.0, shape=(4, 5), dtype=np.float32, metadata=metadata)
    opaque_metadata = FitsOpaqueMetadata()
    opaque_metadata.add_header(_make_header(), name="", ver=1)
    image._opaque_metadata = opaque_metadata
    return image


def test_image_metadata_view() -> None:
    """Test the metadata view on an image with external metadata."""
    image = _make_image({"native_key": 7})
    assert isinstance(image.metadata, MetadataView)
    assert image.metadata["native_key"] == 7
    assert image.metadata["exptime"] == 30.0
    assert image.metadata.native == {"native_key": 7}
    assert list(image.metadata.external) == EXTERNAL_KEYS
    image.metadata["other"] = "x"
    assert image.metadata.native == {"native_key": 7, "other": "x"}
    with pytest.raises(KeyError):
        image.metadata["ExpTime"] = 1.0
    with pytest.raises(KeyError):
        image.metadata.native["ExpTime"] = 1.0
    with pytest.raises(KeyError):
        image.metadata.update({"bgmean": 1.0})


def test_image_metadata_without_opaque_metadata() -> None:
    """Test that an in-memory image has empty external metadata."""
    image = Image(0.0, shape=(4, 5), dtype=np.float32, metadata={"a": 1})
    assert len(image.metadata.external) == 0
    assert image.metadata == {"a": 1}
    image.metadata["EXPTIME"] = 2.0
    assert image.metadata.native == {"a": 1, "EXPTIME": 2.0}


def test_image_metadata_view_reflects_later_opaque_metadata() -> None:
    """Test that the view sees opaque metadata attached after
    construction.
    """
    image = Image(0.0, shape=(4, 5), dtype=np.float32, metadata={"exptime": 1.0})
    opaque_metadata = FitsOpaqueMetadata()
    opaque_metadata.add_header(_make_header(), name="", ver=1)
    image._opaque_metadata = opaque_metadata
    assert image.metadata["exptime"] == 1.0
    assert image.metadata["EXPTIME"] == 30.0
    image.metadata["exptime"] = 2.0
    assert image.metadata.native == {"exptime": 2.0}


def test_subimage_metadata() -> None:
    """Test that subimages share native metadata and see the same external
    metadata.
    """
    image = _make_image({"native_key": 7})
    subimage = image[Box.factory[0:2, 0:3]]
    subimage.metadata["new"] = 1
    assert image.metadata["new"] == 1
    assert subimage.metadata["EXPTIME"] == 30.0
    copied = image.copy()
    copied.metadata["copied_only"] = 1
    assert "copied_only" not in image.metadata
    assert copied.metadata["EXPTIME"] == 30.0


def test_metadata_setter() -> None:
    """Test that the setter replaces native metadata without a shadowing
    check and shares the assigned dict.
    """
    image = _make_image()
    image.metadata = {"exptime": 1.0}
    assert image.metadata["exptime"] == 1.0
    assert image.metadata["EXPTIME"] == 30.0
    image.metadata["exptime"] = 2.0
    shared: dict = {"s": 1}
    image.metadata = shared
    shared["t"] = 2
    assert image.metadata["t"] == 2
    other = Image(0.0, shape=(4, 5), dtype=np.float32)
    other.metadata = image.metadata
    other.metadata["u"] = 3
    assert image.metadata["u"] == 3
    assert other.metadata.native == {"s": 1, "t": 2, "u": 3}
    image.metadata = image.metadata.copy()
    image.metadata["v"] = 4
    assert "v" not in other.metadata
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `lsst-run -l . -- pytest tests/test_metadata.py -v -k "image or setter"`
Expected: FAIL (`assert isinstance(image.metadata, MetadataView)` fails because `metadata` still returns a `dict`).

- [ ] **Step 3: Replace the property**

In `python/lsst/images/_generalized_image.py`:

Add `from collections.abc import Mapping` to the imports, add `from ._metadata import MetadataView, NativeMetadata` after `from ._geom import ...`, and add `EmptyExternalMetadata` to the `from .serialization import (...)` block (sorted).

Replace the `metadata` property and setter with:

```python
    @property
    def metadata(self) -> MetadataView:
        """Flexible metadata associated with the image
        (`~lsst.images.MetadataView`).

        Notes
        -----
        The view combines two sources:

        - ``metadata.native``: metadata that is part of the image's data
          model and is saved with it.  Keys are case-sensitive.
        - ``metadata.external``: read-only metadata carried in from the file
          the image was read from, such as the primary FITS header.  Keys are
          case-insensitive, and ``COMMENT`` and ``HISTORY`` cards are not
          included.

        Lookups check native metadata first, matching case exactly, and then
        external metadata.  Writes go to native metadata; adding a new key
        that case-insensitively matches an external key raises `KeyError`.
        When an external key is repeated, which of its values is returned is
        unspecified; use `~lsst.images.MetadataView.get_all` to obtain every
        value.

        Native metadata is shared with subimages and other views.  It can be
        disconnected by reassigning to a copy explicitly:

            image.metadata = image.metadata.copy()
        """
        external = (
            self._opaque_metadata.external_metadata()
            if self._opaque_metadata is not None
            else EmptyExternalMetadata()
        )
        return MetadataView(NativeMetadata(self._metadata, external), external)

    @metadata.setter
    def metadata(self, value: Mapping[str, MetadataValue] | MetadataView) -> None:
        if isinstance(value, MetadataView):
            self._metadata = value.native._data
        elif isinstance(value, dict):
            self._metadata = value
        else:
            self._metadata = dict(value)
```

Check for an import cycle: `_metadata.py` imports only from `.serialization`, which `_generalized_image.py` already imports, so none is introduced.

- [ ] **Step 4: Switch internal callers to the plain dict**

Make each of these one-line replacements:

- `python/lsst/images/_image.py:364`: `metadata=self.metadata,` → `metadata=self._metadata,`
- `python/lsst/images/_mask.py:993`: `metadata=self.metadata,` → `metadata=self._metadata,`
- `python/lsst/images/_masked_image.py:293`: `metadata=self.metadata,` → `metadata=self._metadata,` (this is a `model_construct` call, which does no validation, so passing the view would serialize incorrectly)
- `python/lsst/images/_masked_image.py:581`: `for n, (k, v) in enumerate(self.metadata.items()):` → `for n, (k, v) in enumerate(self._metadata.items()):` (external cards are already written by the loop above it)
- `python/lsst/images/_color_image.py:214`: `metadata=self.metadata` → `metadata=self._metadata`
- `python/lsst/images/cells/_coadd.py:460`: `metadata=self.metadata,` → `metadata=self._metadata,`
- `python/lsst/images/_visit_image.py:744`: `result.metadata["id"] = legacy.info.getId()` → `result._metadata["id"] = legacy.info.getId()`
- `python/lsst/images/_visit_image.py:774`: `result_info.setId(self.metadata.get("id"))` → `result_info.setId(self._metadata.get("id"))`
- `python/lsst/images/_visit_image.py:969`: `result.metadata["id"] = reader.readExposureId()` → `result._metadata["id"] = reader.readExposureId()`
- `python/lsst/images/tests/_minify_for_fixtures.py:231`: `metadata=subset.metadata,` → `metadata=subset._metadata,`

Then confirm no other internal reader remains:

Run: `rg -n "\.metadata\b" -tpy python/lsst/images | rg -v "legacy_metadata|table_model|tree\.metadata|getMetadata|importlib|_roundtrip.py|_checks.py|_generalized_image.py"`
Expected: no output.

- [ ] **Step 5: Update the shared test helpers**

In `python/lsst/images/tests/_checks.py`, inside `assert_images_equal` (lines 302-309), replace:

```python
        if expect_view == "array":
            assert a.metadata == b.metadata
        else:
            assert (a.metadata is b.metadata) == expect_view
    if not expect_view:
        assert_values_equal(a.array, b.array, atol=atol, rtol=rtol)
        assert a.metadata == b.metadata
```

with:

```python
        if expect_view == "array":
            assert a.metadata.native == b.metadata.native
        else:
            assert (a._metadata is b._metadata) == expect_view
    if not expect_view:
        assert_values_equal(a.array, b.array, atol=atol, rtol=rtol)
        assert a.metadata.native == b.metadata.native
```

In `assert_masks_equal` (line 354) and `assert_masked_images_equal` (line 387), replace `assert a.metadata == b.metadata` with `assert a.metadata.native == b.metadata.native`.

- [ ] **Step 6: Run the new tests and the whole suite**

Run: `lsst-run -l . -- pytest tests/test_metadata.py -v`
Expected: PASS.
Run: `lsst-run -l . -- pytest -n 8 tests`
Expected: PASS (tests needing afw, h5py, butler, or `TESTDATA_IMAGES_DIR` may skip). If a test fails because it compares `image.metadata` against a dict while the image has external metadata, change that comparison to `image.metadata.native`.

- [ ] **Step 7: Document the change**

In `doc/user-guide/for-afw-users.rst`, add to the `VisitImage` component list, after the `photoCalib` line:

```rst
- ``metadata`` (`lsst.daf.base.PropertyList`) -> `VisitImage.metadata` (`MetadataView`); header cards read from a legacy file are available read-only through `MetadataView.external`
```

Create `doc/changes/DM-54770.api.md`:

```markdown
`GeneralizedImage.metadata` now returns a `MetadataView` instead of a plain `dict`.
The view behaves as a mutable mapping over the image's own metadata, with read-only external metadata such as the primary FITS header layered underneath.
Code that needs the plain dictionary should use ``metadata.native`` or ``metadata.copy()``.
Adding a new key that case-insensitively matches an external key now raises `KeyError`.
```

Create `doc/changes/DM-54770.feature.md`:

```markdown
Made the header cards of the file an image was read from available by name through `GeneralizedImage.metadata`.
Lookups check the image's own metadata first and then fall back to a case-insensitive lookup in `MetadataView.external`, which is read-only and omits ``COMMENT`` and ``HISTORY`` cards.
Use `MetadataView.get_all` to obtain every value of a repeated keyword.
```

- [ ] **Step 8: Lint and type-check**

Run: `lsst-run -l . -- ruff check python tests && lsst-run -l . -- ruff format --check python tests && lsst-run -l . -- mypy python/lsst/images tests/test_metadata.py`
Expected: clean.

- [ ] **Step 9: Commit**

```bash
git add python/lsst/images tests/test_metadata.py doc/user-guide/for-afw-users.rst doc/changes/DM-54770.api.md doc/changes/DM-54770.feature.md
git commit -m "Return a native/external metadata view from GeneralizedImage.metadata

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Round-trip tests

**Files:**
- Test: `tests/test_metadata.py` (native FITS and NDF round trips)
- Test: `tests/test_visit_image.py` (legacy round trip, next to `test_repeated_metadata_keys_legacy_round_trip` at line 454)

**Interfaces:**
- Consumes: everything above; `RoundtripFits`, `RoundtripNdf` from `lsst.images.tests`; the `visit_image_components` and `reset_afw_mask_planes` fixtures in `tests/test_visit_image.py`.
- Produces: tests only.

- [ ] **Step 1: Write the native and NDF round-trip tests**

Add to the imports of `tests/test_metadata.py`:

```python
from lsst.images.tests import RoundtripFits, RoundtripNdf
```

(merge into the existing `from lsst.images.tests import ...` line, sorted; `MaskedImage`, `MaskPlane`, and `MaskSchema` are already imported by Task 3.)

Add after the imports:

```python
try:
    import h5py  # noqa: F401

    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False
```

Append to `tests/test_metadata.py`:

```python
def _make_masked_image() -> MaskedImage:
    """Return a small masked image with native metadata and opaque metadata
    holding `_make_header`.
    """
    masked_image = MaskedImage(
        Image(1.0, shape=(4, 5), dtype=np.float32),
        mask_schema=MaskSchema([MaskPlane("BAD", "Pixel is bad.")]),
        metadata={"native_key": 7, "MixedCase": "yes"},
    )
    opaque_metadata = FitsOpaqueMetadata()
    opaque_metadata.add_header(_make_header(), name="", ver=1)
    masked_image._opaque_metadata = opaque_metadata
    return masked_image


def _check_round_trip(result: MaskedImage) -> None:
    assert result.metadata.native == {"native_key": 7, "MixedCase": "yes"}
    assert list(result.metadata.external) == EXTERNAL_KEYS
    assert result.metadata["exptime"] == 30.0
    assert result.metadata.get_all("BGMEAN") == (1.5, 2.5)
    assert result.metadata["NOVAL"] is None
    assert result.metadata["CPLX"] == 1 + 2j
    assert result.metadata["lower key"] == 3
    assert "COMMENT" not in result.metadata


def test_external_metadata_fits_round_trip() -> None:
    """Test that native and external metadata survive a FITS round trip."""
    with RoundtripFits(_make_masked_image()) as roundtrip:
        pass
    _check_round_trip(roundtrip.result)


@pytest.mark.skipif(not HAVE_H5PY, reason="h5py is not installed")
def test_external_metadata_ndf_round_trip() -> None:
    """Test that native and external metadata survive an NDF round trip."""
    with RoundtripNdf(_make_masked_image()) as roundtrip:
        pass
    _check_round_trip(roundtrip.result)
```

- [ ] **Step 2: Write the legacy round-trip test**

In `tests/test_visit_image.py`, insert after `test_repeated_metadata_keys_legacy_round_trip`:

```python
def test_external_metadata_legacy_round_trip(
    visit_image_components: dict[str, Any],
    reset_afw_mask_planes: None,  # noqa: F811
    tmp_path: Path,
) -> None:
    """Verify that native metadata is written to a legacy file as
    ``LSST IMAGES`` cards and is not visible as external metadata when the
    file is read back.
    """
    from lsst.afw.detection import GaussianPsf

    opaque_metadata = FitsOpaqueMetadata()
    header = astropy.io.fits.Header()
    header.append(("PLATFORM", "lsstcam"), end=True)
    header.append(("BGMEAN", 1.5), end=True)
    header.append(("BGMEAN", 2.5), end=True)
    # An external ID card must neither block nor replace the native "id".
    header.append(("ID", 99), end=True)
    opaque_metadata.extract_legacy_primary_header(header)
    visit_image = VisitImage(
        visit_image_components["image"],
        variance=visit_image_components["variance"],
        psf=PointSpreadFunction.from_legacy(GaussianPsf(33, 33, 2.5), bounds=Box.factory[0:1024, 0:1024]),
        mask_schema=visit_image_components["mask_schema"],
        sky_projection=visit_image_components["sky_projection"],
        detector=visit_image_components["detector"],
        obs_info=visit_image_components["obs_info"],
        band="r",
        metadata={"native_key": 7, "MixedCase": "yes"},
    )
    visit_image._opaque_metadata = opaque_metadata
    path = tmp_path / "legacy.fits"
    visit_image.to_legacy().writeFits(str(path))

    with astropy.io.fits.open(path) as hdu_list:
        primary = hdu_list[0].header
        native_cards = {}
        n = 1
        while f"LSST IMAGES KEY {n}" in primary:
            native_cards[primary[f"LSST IMAGES KEY {n}"]] = primary[f"LSST IMAGES VALUE {n}"]
            n += 1
        assert native_cards == {"native_key": 7, "MixedCase": "yes"}
        assert [card.value for card in primary.cards if card.keyword == "BGMEAN"] == [1.5, 2.5]

    result = VisitImage.read_legacy(
        str(path),
        instrument=visit_image_components["obs_info"].instrument,
        visit=visit_image_components["sky_projection"].pixel_frame.visit,
    )
    assert result.metadata.native["native_key"] == 7
    assert result.metadata.native["MixedCase"] == "yes"
    assert "id" in result.metadata.native
    assert not [key for key in result.metadata.external if key.startswith("LSST IMAGES")]
    assert result.metadata.external.get_all("BGMEAN") == (1.5, 2.5)
    assert result.metadata["platform"] == "lsstcam"
    assert result.metadata.external["ID"] == 99

    # Writing the result back out must not duplicate the external cards.
    legacy_metadata = result.to_legacy().getMetadata()
    assert legacy_metadata.getArray("BGMEAN") == [1.5, 2.5]
```

Check the imports at the top of `tests/test_visit_image.py`: `Path`, `astropy.io.fits`, `Box`, `VisitImage`, `FitsOpaqueMetadata`, and `PointSpreadFunction` are already imported; add nothing else.

- [ ] **Step 3: Run the tests**

Run: `lsst-run -l . -- pytest tests/test_metadata.py tests/test_visit_image.py -v -k "round_trip"`
Expected: PASS for all, including `test_external_metadata_legacy_round_trip` when afw is available (it skips otherwise).
To confirm the legacy test guards the Task 3 fix, temporarily move the extraction loop in `extract_legacy_primary_header` back after `header.copy(strip=True)`, rerun `test_external_metadata_legacy_round_trip`, and expect FAIL on the `LSST IMAGES` external assertion; then restore the fix (`git checkout python/lsst/images/fits/_common.py`) and rerun to PASS.

- [ ] **Step 4: Full suite, lint, type-check**

Run: `lsst-run -l . -- pytest -n 8 tests`
Expected: PASS.
Run: `lsst-run -l . -- ruff check python tests && lsst-run -l . -- ruff format --check python tests && lsst-run -l . -- mypy python/lsst/images tests/test_metadata.py`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add tests/test_metadata.py tests/test_visit_image.py
git commit -m "Test external metadata through native, NDF, and legacy round trips

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```
