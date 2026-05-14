# Unified Butler Formatter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate the three per-format butler formatter modules (`lsst.images.fits.formatters`, `lsst.images.json.formatters`, `lsst.images.ndf.formatters`) into a single `lsst.images.formatters` module that dispatches on a `format` write parameter and on the file extension at read time.

**Architecture:** A single file `python/lsst/images/formatters.py` holds five `FormatterV2` subclasses (`Generic / Image / MaskedImage / VisitImage / CellCoadd`) plus a private `_BACKENDS` lookup table keyed by file extension. Each formatter method consults the table to delegate to the right per-format `read` / `write` / archive callable. The existing `lsst.images.fits.formatters` and `lsst.images.json.formatters` modules become deprecation shims so the active `daf_butler` configs keep working; `lsst.images.ndf.formatters` is deleted outright.

**Tech Stack:** Python 3.12+, `pydantic` 2, `lsst.daf.butler.FormatterV2`, `lsst.resources.ResourcePath`, the existing `lsst.images.{fits,json,ndf}.{read,write}` functions and their respective `FitsInputArchive` / `NdfInputArchive`.

**Reference spec:** `docs/superpowers/specs/2026-05-14-unified-formatter-design.md`.

---

## Environment setup

Every shell command below assumes the LSST stack is active. Once per shell:

```bash
source ~/work/lsstsw/bin/envconfig
setup -t b8259 lsst_distrib
setup -k -r ../testdata_images
setup -k -r .
```

Re-source if a new terminal is opened. All `pytest`, `mypy`, `ruff`, and `package-docs` commands depend on it.

---

## File map

**Create:**
- `python/lsst/images/formatters.py` — unified formatters + `_BACKENDS` table + `ComponentSentinel`.
- `tests/test_formatters.py` — dedicated unit tests for the unified formatter's dispatch logic (extension routing, write-extension selection, `recipe` validation, JSON whole-object component fallback, deprecation-shim behavior).

**Modify (replace contents):**
- `python/lsst/images/fits/formatters.py` — becomes a five-class deprecation shim (`Generic / Image / MaskedImage / VisitImage / CellCoadd`, each emitting a `DeprecationWarning` on first instantiation).
- `python/lsst/images/json/formatters.py` — becomes a one-class deprecation shim (`GenericFormatter` only, with `default_extension = ".json"`).

**Delete:**
- `python/lsst/images/ndf/formatters.py` — never deployed; clean removal.

**Untouched** (verified during exploration):
- `python/lsst/images/{fits,json,ndf}/__init__.py` — none of them re-export the `formatters` submodule.
- `python/lsst/images/tests/_roundtrip.py` — `RoundtripFits/Json/Ndf` go through the per-format `read`/`write` functions, not the formatter classes.
- The four end-to-end test files (`test_image.py`, `test_masked_image.py`, `test_visit_image.py`, `test_color_image.py`) — they call the roundtrip helpers, not formatters.
- `doc/lsst.images/{fits,json,ndf}.rst` — they `automodapi` the package, not the `formatters` submodule.

---

## Task 1: `_BACKENDS` table and skeleton formatter module

**Files:**
- Create: `python/lsst/images/formatters.py`
- Test: `tests/test_formatters.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_formatters.py`:

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

import unittest

try:
    import h5py  # noqa: F401

    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False


class BackendsTableTestCase(unittest.TestCase):
    """The private _BACKENDS table wires extension -> read/write/archive."""

    def test_table_keys(self):
        from lsst.images.formatters import _BACKENDS

        self.assertEqual(set(_BACKENDS), {".fits", ".sdf", ".json"})

    def test_fits_backend_wires_fits_read_write(self):
        from lsst.images import fits
        from lsst.images.formatters import _BACKENDS
        from lsst.images.fits._common import PointerModel
        from lsst.images.fits._input_archive import FitsInputArchive

        backend = _BACKENDS[".fits"]
        self.assertIs(backend.read, fits.read)
        self.assertIs(backend.write, fits.write)
        self.assertIs(backend.input_archive, FitsInputArchive)
        self.assertIs(backend.pointer_model, PointerModel)

    @unittest.skipUnless(HAVE_H5PY, "h5py is not installed")
    def test_sdf_backend_wires_ndf_read_write(self):
        from lsst.images import ndf
        from lsst.images.formatters import _BACKENDS
        from lsst.images.ndf._common import NdfPointerModel
        from lsst.images.ndf._input_archive import NdfInputArchive

        backend = _BACKENDS[".sdf"]
        self.assertIs(backend.read, ndf.read)
        self.assertIs(backend.write, ndf.write)
        self.assertIs(backend.input_archive, NdfInputArchive)
        self.assertIs(backend.pointer_model, NdfPointerModel)

    def test_json_backend_wires_json_read_write_no_archive(self):
        from lsst.images import json as images_json
        from lsst.images.formatters import _BACKENDS

        backend = _BACKENDS[".json"]
        self.assertIs(backend.read, images_json.read)
        self.assertIs(backend.write, images_json.write)
        self.assertIsNone(backend.input_archive)
        self.assertIsNone(backend.pointer_model)
```

(Don't add `if __name__ == "__main__": unittest.main()` yet — subsequent tasks
append further test classes to this file; the trailing main block goes in
once at the very end of Task 12.)

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_formatters.py -v
```

Expected: `ModuleNotFoundError: No module named 'lsst.images.formatters'`.

- [ ] **Step 3: Write minimal implementation**

Create `python/lsst/images/formatters.py`:

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

"""Unified butler formatter for lsst.images.

This formatter dispatches on a write-time ``format`` parameter and on the
file extension at read time, replacing the three per-format
(`lsst.images.fits.formatters`, `lsst.images.json.formatters`,
`lsst.images.ndf.formatters`) hierarchies that previously duplicated almost
all of their logic.
"""

from __future__ import annotations

__all__ = ()  # populated in later tasks

from dataclasses import dataclass
from typing import Any, Callable

from . import fits as _fits
from . import json as _json
from .fits._common import PointerModel as _FitsPointerModel
from .fits._input_archive import FitsInputArchive as _FitsInputArchive

try:
    from . import ndf as _ndf
    from .ndf._common import NdfPointerModel as _NdfPointerModel
    from .ndf._input_archive import NdfInputArchive as _NdfInputArchive

    _HAVE_NDF = True
except ImportError:  # h5py is optional; see ndf/__init__.py
    _ndf = None  # type: ignore[assignment]
    _NdfPointerModel = None  # type: ignore[assignment]
    _NdfInputArchive = None  # type: ignore[assignment]
    _HAVE_NDF = False


@dataclass(frozen=True)
class _Backend:
    """One row of the extension-to-backend lookup table."""

    read: Callable[..., Any]
    write: Callable[..., Any]
    input_archive: type | None
    pointer_model: type | None


_BACKENDS: dict[str, _Backend] = {
    ".fits": _Backend(
        read=_fits.read,
        write=_fits.write,
        input_archive=_FitsInputArchive,
        pointer_model=_FitsPointerModel,
    ),
    ".json": _Backend(
        read=_json.read,
        write=_json.write,
        input_archive=None,
        pointer_model=None,
    ),
}
if _HAVE_NDF:
    _BACKENDS[".sdf"] = _Backend(
        read=_ndf.read,
        write=_ndf.write,
        input_archive=_NdfInputArchive,
        pointer_model=_NdfPointerModel,
    )
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/test_formatters.py -v
```

Expected: all four tests pass (or three pass and one skips if h5py is missing).

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/formatters.py tests/test_formatters.py
git commit -m "Add skeleton lsst.images.formatters with backend lookup

Introduces the new unified-formatter module with a private extension
-> backend table. Each row carries the read/write callables plus the
input-archive and pointer-model classes for component-level reads.
NDF entries are populated lazily so the module remains importable
without h5py.

No formatter classes yet; subsequent commits add GenericFormatter,
ImageFormatter, MaskedImageFormatter, VisitImageFormatter, and
CellCoaddFormatter."
```

---

## Task 2: `GenericFormatter` write path

**Files:**
- Modify: `python/lsst/images/formatters.py` (add `GenericFormatter`)
- Test: `tests/test_formatters.py`

Notes for the implementer: `FormatterV2` exposes `self.write_parameters` (a `dict[str, Any]`), `self.dataset_ref`, `self.data_id`, `self.write_recipes`, and `self.butler_provenance`. The existing `lsst.images.fits.formatters.GenericFormatter` exercises every one of these — copy its `_get_compression_options`, `_get_compression_seed`, and `_update_header` methods unchanged.

- [ ] **Step 1: Write the failing tests for `get_write_extension`**

Append to `tests/test_formatters.py`:

```python
class GetWriteExtensionTestCase(unittest.TestCase):
    """`get_write_extension` reads the `format` write parameter."""

    def _make_formatter(self, write_parameters: dict[str, str] | None = None):
        from lsst.images.formatters import GenericFormatter

        formatter = GenericFormatter.__new__(GenericFormatter)
        # FormatterV2 exposes write_parameters as a property over the
        # file_descriptor. For unit-testing we monkey-patch a dict on
        # the instance via __dict__ to bypass the descriptor.
        object.__setattr__(formatter, "_write_parameters", write_parameters or {})
        return formatter

    def test_default_returns_fits(self):
        formatter = self._make_formatter()
        self.assertEqual(formatter.get_write_extension(), ".fits")

    def test_explicit_fits(self):
        formatter = self._make_formatter({"format": "fits"})
        self.assertEqual(formatter.get_write_extension(), ".fits")

    def test_explicit_json(self):
        formatter = self._make_formatter({"format": "json"})
        self.assertEqual(formatter.get_write_extension(), ".json")

    @unittest.skipUnless(HAVE_H5PY, "h5py is not installed")
    def test_explicit_sdf(self):
        formatter = self._make_formatter({"format": "sdf"})
        self.assertEqual(formatter.get_write_extension(), ".sdf")

    def test_unknown_format_raises(self):
        formatter = self._make_formatter({"format": "pickle"})
        with self.assertRaisesRegex(RuntimeError, "is not supported"):
            formatter.get_write_extension()

    def test_recipe_with_non_fits_format_raises(self):
        # `recipe` is FITS-only; using it with format=json must error.
        formatter = self._make_formatter({"format": "json", "recipe": "default"})
        with self.assertRaisesRegex(
            RuntimeError, "only valid for FITS"
        ):
            formatter._validate_write_parameters()
```

The class needs `_write_parameters` to be accessible. Override `write_parameters` to read from it in the test-helper construction path. The cleanest way is to make `get_write_extension` and the validator pull from `self.write_parameters`, but the FormatterV2 property accesses `file_descriptor.write_parameters` which we don't have on a bare instance. To accommodate both unit tests and butler use, **the implementation reads from `self.write_parameters` exclusively**, and the test helper overrides that property; see Step 3.

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_formatters.py::GetWriteExtensionTestCase -v
```

Expected: `AttributeError: ... has no attribute 'GenericFormatter'`.

- [ ] **Step 3: Implement `GenericFormatter`**

Replace the `__all__` line in `python/lsst/images/formatters.py` and add new code. Update the top of the file:

```python
__all__ = ("GenericFormatter",)

import enum
import hashlib
import json as _stdlib_json
from typing import Any, Callable, ClassVar

import astropy.io.fits
from lsst.daf.butler import DatasetProvenance, FormatterV2
from lsst.resources import ResourcePath

from .fits._common import FitsCompressionOptions
from .serialization import ButlerInfo
```

(Adjust the existing `from dataclasses import dataclass` etc. block to live alongside the new imports.)

Append after `_BACKENDS`:

```python
class GenericFormatter(FormatterV2):
    """Unified butler formatter for any lsst.images type.

    The on-disk format is selected by the ``format`` write parameter
    (``fits``, ``json``, ``sdf``) at write time and by the file
    extension at read time. The default format is taken from
    ``self.default_extension`` (``.fits`` for the base class).

    Notes
    -----
    Subclasses (`ImageFormatter` and below) add component-level read
    support. This base class forwards any read parameters straight to
    the underlying ``read`` function.
    """

    default_extension: ClassVar[str] = ".fits"
    supported_extensions: ClassVar[frozenset[str]] = frozenset({".fits", ".sdf", ".json"})
    supported_write_parameters: ClassVar[frozenset[str]] = frozenset({"format", "recipe"})
    can_read_from_uri: ClassVar[bool] = True

    butler_provenance: DatasetProvenance | None = None

    # --- Write parameter handling -------------------------------------------

    @property
    def write_parameters(self) -> dict[str, Any]:  # type: ignore[override]
        # Allow unit tests to inject a dict via `_write_parameters`. The
        # FormatterV2 base provides the property pulling from the file
        # descriptor; override only when our private attribute is set.
        params = getattr(self, "_write_parameters", None)
        if params is not None:
            return params
        return super().write_parameters  # type: ignore[misc]

    def get_write_extension(self) -> str:
        default_fmt = self.default_extension.lstrip(".")
        fmt = self.write_parameters.get("format", default_fmt)
        ext = "." + fmt
        if ext not in self.supported_extensions:
            raise RuntimeError(
                f"Requested format {fmt!r} is not supported; "
                "expected one of {fits, json, sdf}."
            )
        return ext

    def _validate_write_parameters(self) -> None:
        ext = self.get_write_extension()
        if ext != ".fits" and "recipe" in self.write_parameters:
            raise RuntimeError(
                "The 'recipe' write parameter is only valid for FITS output."
            )

    # --- Write path ---------------------------------------------------------

    def write_local_file(self, in_memory_dataset: Any, uri: ResourcePath) -> None:
        self._validate_write_parameters()
        ext = self.get_write_extension()
        backend = _BACKENDS[ext]
        butler_info = ButlerInfo(
            dataset=self.dataset_ref.to_simple(),
            provenance=self.butler_provenance
            if self.butler_provenance is not None
            else DatasetProvenance(),
        )
        kwargs: dict[str, Any] = {"butler_info": butler_info}
        if ext == ".fits":
            kwargs["update_header"] = self._update_header
            kwargs["compression_options"] = self._get_compression_options()
            kwargs["compression_seed"] = self._get_compression_seed()
        elif ext == ".sdf":
            kwargs["update_header"] = self._update_header
        backend.write(in_memory_dataset, uri.ospath, **kwargs)

    def add_provenance(
        self,
        in_memory_dataset: Any,
        /,
        *,
        provenance: DatasetProvenance | None = None,
    ) -> Any:
        # A FormatterV2 instance is used once; stash provenance on self
        # rather than mutating the dataset.
        self.butler_provenance = provenance
        return in_memory_dataset

    # --- FITS-specific helpers (kept verbatim from fits/formatters.py) ----

    def _get_compression_seed(self) -> int:
        hash_bytes = hashlib.md5(
            _stdlib_json.dumps(list(self.data_id.required_values)).encode(),
            usedforsecurity=False,
        ).digest()
        return 1 + int.from_bytes(hash_bytes) % 9999

    def _get_compression_options(self) -> dict[str, FitsCompressionOptions]:
        recipe = self.write_parameters.get("recipe", "default")
        try:
            config = self.write_recipes[recipe]
        except KeyError:
            if recipe == "default":
                return {}
            raise RuntimeError(
                f"Invalid recipe for GenericFormatter: {recipe!r}."
            ) from None
        return {k: FitsCompressionOptions.model_validate(v) for k, v in config.items()}

    def _update_header(self, header: astropy.io.fits.Header) -> None:
        # Logic lifted from lsst.images.fits.formatters; injects HIERARCH
        # LSST BUTLER ... cards. Used for both FITS and NDF (NDF's
        # MORE/FITS extension stores the same primary header).
        for key in list(header):
            if key.startswith("LSST BUTLER"):
                del header[key]
        if self.butler_provenance is not None:
            for key, value in self.butler_provenance.to_flat_dict(
                self.dataset_ref,
                prefix="HIERARCH LSST BUTLER",
                sep=" ",
                simple_types=True,
                max_inputs=3_000,
            ).items():
                header.set(key, value)
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/test_formatters.py::GetWriteExtensionTestCase -v
```

Expected: 5 pass, 1 skipped (if h5py absent) or 6 pass.

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/formatters.py tests/test_formatters.py
git commit -m "Add unified GenericFormatter with format dispatch

Implements get_write_extension (reads format write parameter, falls
back to default_extension) and write_local_file (dispatches via the
_BACKENDS table; validates that recipe is FITS-only; populates the
right kwargs per backend). Provenance handling and FITS compression
helpers are lifted verbatim from lsst.images.fits.formatters."
```

---

## Task 3: `GenericFormatter` read path

**Files:**
- Modify: `python/lsst/images/formatters.py` (add `read_from_uri`, `_extension_from_uri`)
- Test: `tests/test_formatters.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_formatters.py`:

```python
class ExtensionFromUriTestCase(unittest.TestCase):
    """`read_from_uri` routes based on `uri.getExtension()`."""

    def test_fits(self):
        from lsst.images.formatters import GenericFormatter

        formatter = GenericFormatter.__new__(GenericFormatter)
        uri = ResourcePath("/tmp/x.fits")
        self.assertEqual(formatter._extension_from_uri(uri), ".fits")

    @unittest.skipUnless(HAVE_H5PY, "h5py is not installed")
    def test_sdf(self):
        from lsst.images.formatters import GenericFormatter

        formatter = GenericFormatter.__new__(GenericFormatter)
        uri = ResourcePath("/tmp/x.sdf")
        self.assertEqual(formatter._extension_from_uri(uri), ".sdf")

    def test_json(self):
        from lsst.images.formatters import GenericFormatter

        formatter = GenericFormatter.__new__(GenericFormatter)
        uri = ResourcePath("/tmp/x.json")
        self.assertEqual(formatter._extension_from_uri(uri), ".json")

    def test_unknown(self):
        from lsst.images.formatters import GenericFormatter

        formatter = GenericFormatter.__new__(GenericFormatter)
        uri = ResourcePath("/tmp/x.pickle")
        with self.assertRaisesRegex(RuntimeError, "unsupported extension"):
            formatter._extension_from_uri(uri)

    def test_compressed_fits_unsupported(self):
        # We don't claim to handle .fits.gz; getExtension returns
        # '.fits.gz' and the lookup misses.
        from lsst.images.formatters import GenericFormatter

        formatter = GenericFormatter.__new__(GenericFormatter)
        uri = ResourcePath("/tmp/x.fits.gz")
        with self.assertRaisesRegex(RuntimeError, "unsupported extension"):
            formatter._extension_from_uri(uri)
```

Also add `from lsst.resources import ResourcePath` near the top of the test file.

- [ ] **Step 2: Run the test to verify it fails**

```bash
pytest tests/test_formatters.py::ExtensionFromUriTestCase -v
```

Expected: `AttributeError: ... has no attribute '_extension_from_uri'`.

- [ ] **Step 3: Implement the read methods**

Append to `python/lsst/images/formatters.py` inside `GenericFormatter`:

```python
    # --- Read path ---------------------------------------------------------

    def _extension_from_uri(self, uri: ResourcePath) -> str:
        ext = uri.getExtension()
        if ext not in self.supported_extensions:
            raise RuntimeError(
                f"Cannot read {uri}: unsupported extension {ext!r}."
            )
        return ext

    def read_from_uri(
        self,
        uri: ResourcePath,
        component: str | None = None,
        expected_size: int = -1,
    ) -> Any:
        pytype = self.dataset_ref.datasetType.storageClass.pytype
        ext = self._extension_from_uri(uri)
        backend = _BACKENDS[ext]
        kwargs = self.file_descriptor.parameters or {}
        return backend.read(pytype, uri, **kwargs).deserialized
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_formatters.py -v
```

Expected: all previous + 5 new tests pass.

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/formatters.py tests/test_formatters.py
git commit -m "Add unified GenericFormatter read path

read_from_uri inspects the URI extension, looks up the backend, and
delegates. Unknown or composite extensions (.fits.gz, .sdf.z, etc.)
raise a clear RuntimeError so callers know to extend the backend
table if they need them."
```

---

## Task 4: `ComponentSentinel` and `ImageFormatter`

**Files:**
- Modify: `python/lsst/images/formatters.py` (add `ComponentSentinel`, `ImageFormatter`)
- Test: `tests/test_formatters.py`

`ImageFormatter` is the first level that supports component reads (`projection`, `bbox`, `obs_info`). For `.fits` and `.sdf` it opens the corresponding input archive; for `.json` it reads the whole object and pulls the attribute via `getattr`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_formatters.py`:

```python
class ImageFormatterComponentReadTestCase(unittest.TestCase):
    """ImageFormatter routes component reads per extension."""

    def _make_image(self):
        import numpy as np
        from lsst.images import Box, Image

        return Image(
            np.arange(20, dtype=np.float32).reshape(4, 5),
            bbox=Box.factory[10:14, 20:25],
        )

    def test_fits_bbox_component(self):
        import tempfile

        from lsst.images import Box, Image, fits
        from lsst.images.formatters import ImageFormatter

        image = self._make_image()
        with tempfile.NamedTemporaryFile(suffix=".fits", delete_on_close=False) as tmp:
            tmp.close()
            fits.write(image, tmp.name)
            formatter = ImageFormatter.__new__(ImageFormatter)
            object.__setattr__(formatter, "_storage_class_pytype", Image)
            bbox = formatter._read_component_from_uri(
                "bbox", ResourcePath(tmp.name)
            )
            self.assertEqual(bbox, image.bbox)

    @unittest.skipUnless(HAVE_H5PY, "h5py is not installed")
    def test_sdf_bbox_component(self):
        import tempfile

        from lsst.images import Box, Image, ndf
        from lsst.images.formatters import ImageFormatter

        image = self._make_image()
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete_on_close=False) as tmp:
            tmp.close()
            ndf.write(image, tmp.name)
            formatter = ImageFormatter.__new__(ImageFormatter)
            object.__setattr__(formatter, "_storage_class_pytype", Image)
            bbox = formatter._read_component_from_uri(
                "bbox", ResourcePath(tmp.name)
            )
            self.assertEqual(bbox, image.bbox)

    def test_json_bbox_component_via_whole_object(self):
        import tempfile

        from lsst.images import Image
        from lsst.images import json as images_json
        from lsst.images.formatters import ImageFormatter

        image = self._make_image()
        with tempfile.NamedTemporaryFile(suffix=".json", delete_on_close=False) as tmp:
            tmp.close()
            images_json.write(image, tmp.name)
            formatter = ImageFormatter.__new__(ImageFormatter)
            object.__setattr__(formatter, "_storage_class_pytype", Image)
            bbox = formatter._read_component_from_uri(
                "bbox", ResourcePath(tmp.name)
            )
            self.assertEqual(bbox, image.bbox)

    def test_json_unknown_component_raises(self):
        import tempfile

        from lsst.images import Image
        from lsst.images import json as images_json
        from lsst.images.formatters import ImageFormatter

        image = self._make_image()
        with tempfile.NamedTemporaryFile(suffix=".json", delete_on_close=False) as tmp:
            tmp.close()
            images_json.write(image, tmp.name)
            formatter = ImageFormatter.__new__(ImageFormatter)
            object.__setattr__(formatter, "_storage_class_pytype", Image)
            with self.assertRaises(NotImplementedError):
                formatter._read_component_from_uri(
                    "nonexistent", ResourcePath(tmp.name)
                )
```

These tests use `formatter._read_component_from_uri(component, uri)` — a thin shim added next to `read_from_uri` that bypasses the FormatterV2 file-descriptor lookup so the unit tests can exercise the dispatch logic without a full butler.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_formatters.py::ImageFormatterComponentReadTestCase -v
```

Expected: `AttributeError: ... has no attribute 'ImageFormatter'`.

- [ ] **Step 3: Implement `ComponentSentinel`, `ImageFormatter`, and the component-read dispatcher**

Add to `python/lsst/images/formatters.py` after `GenericFormatter`:

```python
from astro_metadata_translator import ObservationInfo

from ._geom import Box
from ._transforms import ProjectionSerializationModel


class ComponentSentinel(enum.Enum):
    """Special return values from `ImageFormatter.read_component`."""

    UNRECOGNIZED_COMPONENT = enum.auto()
    """Subclasses might still recognise this component."""

    INVALID_COMPONENT_MODEL = enum.auto()
    """Component name is known but the model attribute is missing or
    has the wrong type.
    """


class ImageFormatter(GenericFormatter):
    """Adds component-level read support for image-like types.

    Subclasses override `read_component` to handle additional components
    (image/mask/variance for MaskedImage; psf/summary_stats/etc. for
    VisitImage).
    """

    def _storage_class_pytype_default(self) -> type:
        return self.file_descriptor.storageClass.pytype

    def _get_pytype(self) -> type:
        # Allow unit tests to inject a pytype without a real FileDescriptor.
        pytype = getattr(self, "_storage_class_pytype", None)
        if pytype is not None:
            return pytype
        return self._storage_class_pytype_default()

    def read_from_uri(
        self,
        uri: ResourcePath,
        component: str | None = None,
        expected_size: int = -1,
    ) -> Any:
        pytype = self._get_pytype()
        ext = self._extension_from_uri(uri)
        backend = _BACKENDS[ext]
        if component is None:
            result = backend.read(
                pytype, uri, bbox=self.pop_bbox_from_parameters()
            ).deserialized
        else:
            result = self._read_component_from_uri(component, uri)
        self.check_unhandled_parameters()
        return result

    def _read_component_from_uri(
        self, component: str, uri: ResourcePath
    ) -> Any:
        ext = self._extension_from_uri(uri)
        backend = _BACKENDS[ext]
        pytype = self._get_pytype()
        if ext == ".json":
            obj = backend.read(pytype, uri).deserialized
            try:
                return getattr(obj, component)
            except AttributeError as exc:
                raise NotImplementedError(
                    f"Unrecognized component {component!r} for JSON read."
                ) from exc
        # FITS/NDF archive path.
        archive_cls = backend.input_archive
        pointer_model = backend.pointer_model
        assert archive_cls is not None  # noqa: S101 (table guarantee)
        assert pointer_model is not None  # noqa: S101
        # FitsInputArchive uses partial=True for component reads; NDF
        # has no such kwarg.
        open_kwargs = {"partial": True} if ext == ".fits" else {}
        with archive_cls.open(uri, **open_kwargs) as archive:
            tree_type = pytype._get_archive_tree_type(pointer_model)
            tree = archive.get_tree(tree_type)
            result = self.read_component(component, tree, archive)
        if result is ComponentSentinel.UNRECOGNIZED_COMPONENT:
            raise NotImplementedError(
                f"Unrecognized component {component!r} for {type(self).__name__}."
            )
        if result is ComponentSentinel.INVALID_COMPONENT_MODEL:
            raise NotImplementedError(
                f"Invalid serialization model for component {component!r} "
                f"for {type(self).__name__}."
            )
        return result

    def pop_bbox_from_parameters(self) -> Box | None:
        parameters = self.file_descriptor.parameters or {}
        return parameters.pop("bbox", None)

    def check_unhandled_parameters(self) -> None:
        if self.file_descriptor.parameters:
            raise RuntimeError(
                f"Parameters {list(self.file_descriptor.parameters.keys())} not recognized."
            )

    def read_component(self, component: str, tree: Any, archive: Any) -> Any:
        match component:
            case "projection":
                if isinstance(
                    p := getattr(tree, "projection", None),
                    ProjectionSerializationModel,
                ):
                    return p.deserialize(archive)
                return ComponentSentinel.INVALID_COMPONENT_MODEL
            case "bbox":
                if isinstance(bbox := getattr(tree, "bbox", None), Box):
                    return bbox
                return ComponentSentinel.INVALID_COMPONENT_MODEL
            case "obs_info":
                if isinstance(
                    oi := getattr(tree, "obs_info", None), ObservationInfo
                ):
                    return oi
                return ComponentSentinel.INVALID_COMPONENT_MODEL
        return ComponentSentinel.UNRECOGNIZED_COMPONENT
```

Update the module `__all__`:

```python
__all__ = (
    "ComponentSentinel",
    "GenericFormatter",
    "ImageFormatter",
)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_formatters.py -v
```

Expected: all previous + 4 new tests pass (or 3 + skip for `.sdf` if no h5py).

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/formatters.py tests/test_formatters.py
git commit -m "Add ComponentSentinel + ImageFormatter to unified formatter

ImageFormatter reads projection/bbox/obs_info components from FITS,
NDF, or JSON files. FITS/NDF use the format-specific input archive;
JSON falls back to reading the whole object and pulling the
attribute, raising NotImplementedError on missing names. The
_read_component_from_uri helper isolates this dispatch so unit tests
can drive it directly without a butler."
```

---

## Task 5: `MaskedImageFormatter`

**Files:**
- Modify: `python/lsst/images/formatters.py` (add `MaskedImageFormatter`)
- Test: `tests/test_formatters.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_formatters.py`:

```python
class MaskedImageFormatterComponentReadTestCase(unittest.TestCase):
    """MaskedImageFormatter routes image/mask/variance per extension."""

    def _make_masked_image(self):
        import numpy as np
        from lsst.images import Image, MaskedImage, MaskPlane, MaskSchema

        rng = np.random.default_rng(11)
        return MaskedImage(
            Image(rng.normal(100.0, 8.0, size=(10, 12)), start=(0, 0)),
            mask_schema=MaskSchema([MaskPlane("BAD", "bad pixel")]),
        )

    def test_fits_image_component(self):
        import tempfile

        from lsst.images import MaskedImage, fits
        from lsst.images.formatters import MaskedImageFormatter

        mi = self._make_masked_image()
        with tempfile.NamedTemporaryFile(suffix=".fits", delete_on_close=False) as tmp:
            tmp.close()
            fits.write(mi, tmp.name)
            formatter = MaskedImageFormatter.__new__(MaskedImageFormatter)
            object.__setattr__(formatter, "_storage_class_pytype", MaskedImage)
            image = formatter._read_component_from_uri(
                "image", ResourcePath(tmp.name)
            )
            self.assertEqual(image.bbox, mi.image.bbox)

    @unittest.skipUnless(HAVE_H5PY, "h5py is not installed")
    def test_sdf_mask_component(self):
        import tempfile

        from lsst.images import MaskedImage, ndf
        from lsst.images.formatters import MaskedImageFormatter

        mi = self._make_masked_image()
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete_on_close=False) as tmp:
            tmp.close()
            ndf.write(mi, tmp.name)
            formatter = MaskedImageFormatter.__new__(MaskedImageFormatter)
            object.__setattr__(formatter, "_storage_class_pytype", MaskedImage)
            mask = formatter._read_component_from_uri(
                "mask", ResourcePath(tmp.name)
            )
            self.assertEqual(mask.bbox, mi.mask.bbox)

    def test_json_variance_component_via_whole_object(self):
        import tempfile

        from lsst.images import MaskedImage
        from lsst.images import json as images_json
        from lsst.images.formatters import MaskedImageFormatter

        mi = self._make_masked_image()
        with tempfile.NamedTemporaryFile(suffix=".json", delete_on_close=False) as tmp:
            tmp.close()
            images_json.write(mi, tmp.name)
            formatter = MaskedImageFormatter.__new__(MaskedImageFormatter)
            object.__setattr__(formatter, "_storage_class_pytype", MaskedImage)
            variance = formatter._read_component_from_uri(
                "variance", ResourcePath(tmp.name)
            )
            self.assertEqual(variance.bbox, mi.variance.bbox)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_formatters.py::MaskedImageFormatterComponentReadTestCase -v
```

Expected: `AttributeError: ... has no attribute 'MaskedImageFormatter'`.

- [ ] **Step 3: Implement `MaskedImageFormatter`**

Append to `python/lsst/images/formatters.py`:

```python
from ._masked_image import MaskedImageSerializationModel


class MaskedImageFormatter(ImageFormatter):
    """Adds image/mask/variance component support."""

    def read_component(self, component: str, tree: Any, archive: Any) -> Any:
        match super().read_component(component, tree, archive):
            case ComponentSentinel():
                pass
            case handled:
                return handled
        if not isinstance(tree, MaskedImageSerializationModel):
            return ComponentSentinel.INVALID_COMPONENT_MODEL
        bbox = self.pop_bbox_from_parameters()
        match component:
            case "image":
                return tree.image.deserialize(archive, bbox=bbox)
            case "mask":
                return tree.mask.deserialize(archive, bbox=bbox)
            case "variance":
                return tree.variance.deserialize(archive, bbox=bbox)
        return ComponentSentinel.UNRECOGNIZED_COMPONENT
```

Update `__all__`:

```python
__all__ = (
    "ComponentSentinel",
    "GenericFormatter",
    "ImageFormatter",
    "MaskedImageFormatter",
)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_formatters.py -v
```

Expected: all previous + 3 new tests pass.

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/formatters.py tests/test_formatters.py
git commit -m "Add unified MaskedImageFormatter

Reads image/mask/variance components for FITS and NDF via the
archive tree, and for JSON via getattr on the deserialized whole
object."
```

---

## Task 6: `VisitImageFormatter`

**Files:**
- Modify: `python/lsst/images/formatters.py` (add `VisitImageFormatter`)
- Test: `tests/test_formatters.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_formatters.py`:

```python
class VisitImageFormatterComponentReadTestCase(unittest.TestCase):
    """VisitImageFormatter reads psf/summary_stats/detector/aperture_corrections."""

    def _make_visit_image(self):
        # Reuse the existing test helper from tests/test_visit_image.py if
        # importable; otherwise construct a minimal VisitImage inline. The
        # helper from test_visit_image.py is preferred so the component
        # surface matches production fixtures.
        from tests.test_visit_image import VisitImageTestCase  # local import

        case = VisitImageTestCase()
        case.setUp()
        return case.visit_image

    def test_fits_summary_stats_component(self):
        import tempfile

        from lsst.images import VisitImage, fits
        from lsst.images.formatters import VisitImageFormatter

        vi = self._make_visit_image()
        with tempfile.NamedTemporaryFile(suffix=".fits", delete_on_close=False) as tmp:
            tmp.close()
            fits.write(vi, tmp.name)
            formatter = VisitImageFormatter.__new__(VisitImageFormatter)
            object.__setattr__(formatter, "_storage_class_pytype", VisitImage)
            summary = formatter._read_component_from_uri(
                "summary_stats", ResourcePath(tmp.name)
            )
            self.assertEqual(summary, vi.summary_stats)

    @unittest.skipUnless(HAVE_H5PY, "h5py is not installed")
    def test_sdf_psf_component(self):
        import tempfile

        from lsst.images import VisitImage, ndf
        from lsst.images.formatters import VisitImageFormatter

        vi = self._make_visit_image()
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete_on_close=False) as tmp:
            tmp.close()
            ndf.write(vi, tmp.name)
            formatter = VisitImageFormatter.__new__(VisitImageFormatter)
            object.__setattr__(formatter, "_storage_class_pytype", VisitImage)
            psf = formatter._read_component_from_uri(
                "psf", ResourcePath(tmp.name)
            )
            self.assertEqual(type(psf), type(vi.psf))

    def test_json_aperture_corrections_via_whole_object(self):
        import tempfile

        from lsst.images import VisitImage
        from lsst.images import json as images_json
        from lsst.images.formatters import VisitImageFormatter

        vi = self._make_visit_image()
        with tempfile.NamedTemporaryFile(suffix=".json", delete_on_close=False) as tmp:
            tmp.close()
            images_json.write(vi, tmp.name)
            formatter = VisitImageFormatter.__new__(VisitImageFormatter)
            object.__setattr__(formatter, "_storage_class_pytype", VisitImage)
            ap = formatter._read_component_from_uri(
                "aperture_corrections", ResourcePath(tmp.name)
            )
            self.assertEqual(ap, vi.aperture_corrections)
```

The `from tests.test_visit_image import VisitImageTestCase` import requires the test runner to add `tests/` to the path. With pytest it works out of the box; with `python -m unittest tests.test_formatters` it also works. With `python tests/test_formatters.py` (direct execution) you'd need `sys.path` manipulation — acceptable to leave as a known limitation, since pytest is the supported runner.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_formatters.py::VisitImageFormatterComponentReadTestCase -v
```

Expected: `AttributeError: ... has no attribute 'VisitImageFormatter'`.

- [ ] **Step 3: Implement `VisitImageFormatter`**

Append to `python/lsst/images/formatters.py`:

```python
from ._visit_image import VisitImageSerializationModel


class VisitImageFormatter(MaskedImageFormatter):
    """Adds psf/summary_stats/detector/aperture_corrections."""

    def read_component(self, component: str, tree: Any, archive: Any) -> Any:
        match super().read_component(component, tree, archive):
            case ComponentSentinel():
                pass
            case handled:
                return handled
        if not isinstance(tree, VisitImageSerializationModel):
            return ComponentSentinel.INVALID_COMPONENT_MODEL
        match component:
            case "psf":
                # The FITS path uses tree.psf.deserialize; the NDF tree
                # exposes deserialize_psf for the same effect.
                if hasattr(tree, "deserialize_psf"):
                    return tree.deserialize_psf(archive)
                return tree.psf.deserialize(archive)
            case "summary_stats":
                return tree.summary_stats
            case "detector":
                if getattr(tree, "detector", None) is not None:
                    return tree.detector.deserialize(archive)
                return ComponentSentinel.INVALID_COMPONENT_MODEL
            case "aperture_corrections":
                return tree.aperture_corrections.deserialize(archive)
        return ComponentSentinel.UNRECOGNIZED_COMPONENT
```

Update `__all__`:

```python
__all__ = (
    "ComponentSentinel",
    "GenericFormatter",
    "ImageFormatter",
    "MaskedImageFormatter",
    "VisitImageFormatter",
)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_formatters.py -v
```

Expected: all previous + 3 new tests pass.

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/formatters.py tests/test_formatters.py
git commit -m "Add unified VisitImageFormatter

Reads psf/summary_stats/detector/aperture_corrections components.
The psf path probes for both tree.deserialize_psf (NDF) and
tree.psf.deserialize (FITS) so the same method works against either
archive."
```

---

## Task 7: `CellCoaddFormatter`

**Files:**
- Modify: `python/lsst/images/formatters.py` (add `CellCoaddFormatter`)
- Test: `tests/test_formatters.py`

`CellCoadd` is only supported on the FITS path today (the existing `lsst.images.fits.formatters.CellCoaddFormatter` reads psf and provenance components from a `CellCoaddSerializationModel`). The unified formatter retains FITS-only support; NDF and JSON CellCoadd component reads raise `NotImplementedError` via the existing sentinel path.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_formatters.py`:

```python
class CellCoaddFormatterComponentReadTestCase(unittest.TestCase):
    """CellCoaddFormatter reads psf/provenance components from FITS."""

    def _make_cell_coadd(self):
        from tests.test_cell_coadd import CellCoaddTestCase  # local import

        case = CellCoaddTestCase()
        case.setUp()
        return case.cell_coadd

    def test_fits_psf_component(self):
        import tempfile

        from lsst.images import fits
        from lsst.images.cells import CellCoadd
        from lsst.images.formatters import CellCoaddFormatter

        coadd = self._make_cell_coadd()
        with tempfile.NamedTemporaryFile(suffix=".fits", delete_on_close=False) as tmp:
            tmp.close()
            fits.write(coadd, tmp.name)
            formatter = CellCoaddFormatter.__new__(CellCoaddFormatter)
            object.__setattr__(formatter, "_storage_class_pytype", CellCoadd)
            # `psf` is a derived component for CellCoadd; the FITS archive
            # path reads it through tree.deserialize_psf.
            object.__setattr__(formatter, "_file_descriptor_parameters", {})
            psf = formatter._read_component_from_uri(
                "psf", ResourcePath(tmp.name)
            )
            self.assertIsNotNone(psf)
```

If `tests/test_cell_coadd.py` doesn't expose a usable fixture, replace `_make_cell_coadd` with a direct constructor call following the pattern in `tests/test_cell_coadd.py::CellCoaddTestCase.setUp`.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pytest tests/test_formatters.py::CellCoaddFormatterComponentReadTestCase -v
```

Expected: `AttributeError: ... has no attribute 'CellCoaddFormatter'`.

- [ ] **Step 3: Implement `CellCoaddFormatter`**

Append to `python/lsst/images/formatters.py`:

```python
class CellCoaddFormatter(MaskedImageFormatter):
    """Adds CellCoadd-specific psf and provenance components."""

    def read_component(self, component: str, tree: Any, archive: Any) -> Any:
        from .cells import CellCoaddSerializationModel  # avoid cycles

        match super().read_component(component, tree, archive):
            case ComponentSentinel():
                pass
            case handled:
                return handled
        if not isinstance(tree, CellCoaddSerializationModel):
            return ComponentSentinel.INVALID_COMPONENT_MODEL
        match component:
            case "psf":
                bbox = self.pop_bbox_from_parameters()
                return tree.deserialize_psf(archive, bbox=bbox)
            case "provenance":
                return tree.deserialize_provenance(archive)
        return ComponentSentinel.UNRECOGNIZED_COMPONENT
```

Update `__all__`:

```python
__all__ = (
    "CellCoaddFormatter",
    "ComponentSentinel",
    "GenericFormatter",
    "ImageFormatter",
    "MaskedImageFormatter",
    "VisitImageFormatter",
)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pytest tests/test_formatters.py -v
```

Expected: all previous + 1 new test pass.

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/formatters.py tests/test_formatters.py
git commit -m "Add unified CellCoaddFormatter

Inherits MaskedImageFormatter and adds the CellCoadd-specific psf
and provenance components. NDF and JSON paths fall through to the
sentinel-driven error path because the CellCoaddSerializationModel
isinstance check fails for non-FITS trees."
```

---

## Task 8: Delete `lsst.images.ndf.formatters`

**Files:**
- Delete: `python/lsst/images/ndf/formatters.py`

NDF was never wired into any daf_butler config; the file can go without a shim.

- [ ] **Step 1: Confirm nothing references it**

```bash
grep -rn "lsst.images.ndf.formatters\|from .formatters\|from .ndf.formatters" python/ tests/ doc/ 2>/dev/null
```

Expected: zero hits in source or tests; any remaining hits inside `docs/superpowers/plans/` historical artefacts are fine.

- [ ] **Step 2: Delete the file**

```bash
git rm python/lsst/images/ndf/formatters.py
```

- [ ] **Step 3: Run the tests**

```bash
pytest tests/ -q --deselect tests/test_visit_image.py::VisitImageLegacyTestCase::test_butler_converters --deselect tests/test_visit_image.py::VisitImageLegacyTestCase::test_rewrite
```

Expected: all pass (the deletion is invisible to the test suite because nothing imported the file).

- [ ] **Step 4: Commit**

```bash
git commit -m "Remove lsst.images.ndf.formatters

NDF was never wired into a deployed daf_butler config; the
unified lsst.images.formatters now provides the equivalent classes."
```

---

## Task 9: Convert `lsst.images.fits.formatters` to a deprecation shim

**Files:**
- Modify (replace contents): `python/lsst/images/fits/formatters.py`
- Test: `tests/test_formatters.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_formatters.py`:

```python
class FitsDeprecationShimTestCase(unittest.TestCase):
    """lsst.images.fits.formatters is a deprecation shim."""

    def test_image_formatter_warns(self):
        import warnings

        from lsst.images.fits.formatters import ImageFormatter

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            ImageFormatter.__new__(ImageFormatter)
            ImageFormatter.__init__(
                ImageFormatter.__new__(ImageFormatter)  # type: ignore[call-arg]
            )
        self.assertTrue(
            any(
                issubclass(w.category, DeprecationWarning)
                and "fits.formatters.ImageFormatter is deprecated" in str(w.message)
                for w in recorded
            ),
            f"No deprecation warning observed; got: {[str(w.message) for w in recorded]}",
        )

    def test_subclass_is_unified_class(self):
        from lsst.images import formatters as unified
        from lsst.images.fits import formatters as shim

        self.assertTrue(issubclass(shim.GenericFormatter, unified.GenericFormatter))
        self.assertTrue(issubclass(shim.ImageFormatter, unified.ImageFormatter))
        self.assertTrue(issubclass(shim.MaskedImageFormatter, unified.MaskedImageFormatter))
        self.assertTrue(issubclass(shim.VisitImageFormatter, unified.VisitImageFormatter))
        self.assertTrue(issubclass(shim.CellCoaddFormatter, unified.CellCoaddFormatter))
```

- [ ] **Step 2: Run the test**

```bash
pytest tests/test_formatters.py::FitsDeprecationShimTestCase -v
```

Expected: PASS on the `test_subclass_is_unified_class` (the current FITS classes share the names; the inheritance check trivially fails because they don't inherit from the unified ones). FAIL on `test_image_formatter_warns` (no deprecation warning emitted today).

This step's purpose is to establish a baseline before we replace the file.

- [ ] **Step 3: Replace `python/lsst/images/fits/formatters.py` with the shim**

Overwrite the file with:

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

"""Deprecated re-exports of the unified ``lsst.images.formatters`` module.

These names are kept so that deployed butler configs in
``daf_butler/configs/datastores/formatters.yaml`` continue to work.
Each class is a one-line subclass of the corresponding unified
formatter that emits a `DeprecationWarning` on first instantiation.
"""

from __future__ import annotations

__all__ = (
    "CellCoaddFormatter",
    "GenericFormatter",
    "ImageFormatter",
    "MaskedImageFormatter",
    "VisitImageFormatter",
)

import warnings
from typing import Any

from .. import formatters as _unified


def _warn(name: str) -> None:
    warnings.warn(
        f"lsst.images.fits.formatters.{name} is deprecated; "
        f"use lsst.images.formatters.{name} instead. The fits-only "
        f"formatter forwards to the unified one and will be removed "
        f"in a future release.",
        DeprecationWarning,
        stacklevel=3,
    )


class GenericFormatter(_unified.GenericFormatter):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _warn("GenericFormatter")
        super().__init__(*args, **kwargs)


class ImageFormatter(_unified.ImageFormatter):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _warn("ImageFormatter")
        super().__init__(*args, **kwargs)


class MaskedImageFormatter(_unified.MaskedImageFormatter):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _warn("MaskedImageFormatter")
        super().__init__(*args, **kwargs)


class VisitImageFormatter(_unified.VisitImageFormatter):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _warn("VisitImageFormatter")
        super().__init__(*args, **kwargs)


class CellCoaddFormatter(_unified.CellCoaddFormatter):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _warn("CellCoaddFormatter")
        super().__init__(*args, **kwargs)
```

- [ ] **Step 4: Run the test**

```bash
pytest tests/test_formatters.py::FitsDeprecationShimTestCase -v
```

Expected: both tests pass. Also rerun the full test suite to confirm the existing butler-driven roundtrip tests (`test_image.py::test_butler`, `test_visit_image.py::test_butler`, etc.) still pass — they go through the shim and should produce one `DeprecationWarning` per write.

```bash
pytest tests/ -q --deselect tests/test_visit_image.py::VisitImageLegacyTestCase::test_butler_converters --deselect tests/test_visit_image.py::VisitImageLegacyTestCase::test_rewrite -W error::DeprecationWarning -W default::DeprecationWarning:lsst.images.fits.formatters
```

The `-W error::DeprecationWarning` catches any *unexpected* deprecation; the `-W default::DeprecationWarning:lsst.images.fits.formatters` filter overrides for the warnings our shim emits, since they're intentional.

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/fits/formatters.py tests/test_formatters.py
git commit -m "Convert lsst.images.fits.formatters to a deprecation shim

Each class becomes a one-line subclass of the corresponding unified
formatter that emits a DeprecationWarning on construction. This
keeps the daf_butler configs that reference
lsst.images.fits.formatters.X working while signalling that callers
should migrate to lsst.images.formatters."
```

---

## Task 10: Convert `lsst.images.json.formatters` to a deprecation shim

**Files:**
- Modify (replace contents): `python/lsst/images/json/formatters.py`
- Test: `tests/test_formatters.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_formatters.py`:

```python
class JsonDeprecationShimTestCase(unittest.TestCase):
    """lsst.images.json.formatters is a deprecation shim defaulting to .json."""

    def test_generic_formatter_warns(self):
        import warnings

        from lsst.images.json.formatters import GenericFormatter

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            GenericFormatter.__init__(
                GenericFormatter.__new__(GenericFormatter)  # type: ignore[call-arg]
            )
        self.assertTrue(
            any(
                issubclass(w.category, DeprecationWarning)
                and "json.formatters.GenericFormatter is deprecated" in str(w.message)
                for w in recorded
            )
        )

    def test_default_extension_is_json(self):
        from lsst.images.json.formatters import GenericFormatter

        self.assertEqual(GenericFormatter.default_extension, ".json")

    def test_default_write_extension_is_json(self):
        from lsst.images.json.formatters import GenericFormatter

        formatter = GenericFormatter.__new__(GenericFormatter)
        object.__setattr__(formatter, "_write_parameters", {})
        # Suppress the deprecation noise from __init__ for this lookup;
        # we're just exercising get_write_extension on a bare instance.
        self.assertEqual(formatter.get_write_extension(), ".json")
```

- [ ] **Step 2: Run the test**

```bash
pytest tests/test_formatters.py::JsonDeprecationShimTestCase -v
```

Expected: FAIL (the existing JSON formatter doesn't emit a warning and doesn't inherit from the unified class).

- [ ] **Step 3: Replace `python/lsst/images/json/formatters.py` with the shim**

Overwrite the file:

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

"""Deprecated re-export of the unified ``lsst.images.formatters`` module.

`lsst.images.json.formatters.GenericFormatter` exists so that deployed
butler configs that point Transform and Projection storage classes at
this path keep working. The shim overrides ``default_extension`` to
``.json`` so writes default to JSON output when no ``format`` write
parameter is supplied.
"""

from __future__ import annotations

__all__ = ("GenericFormatter",)

import warnings
from typing import Any, ClassVar

from .. import formatters as _unified


def _warn(name: str) -> None:
    warnings.warn(
        f"lsst.images.json.formatters.{name} is deprecated; "
        f"use lsst.images.formatters.{name} with format='json' "
        f"instead. The json-only formatter forwards to the unified "
        f"one and will be removed in a future release.",
        DeprecationWarning,
        stacklevel=3,
    )


class GenericFormatter(_unified.GenericFormatter):
    default_extension: ClassVar[str] = ".json"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _warn("GenericFormatter")
        super().__init__(*args, **kwargs)
```

- [ ] **Step 4: Run the test**

```bash
pytest tests/test_formatters.py::JsonDeprecationShimTestCase -v
```

Expected: all three pass.

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/json/formatters.py tests/test_formatters.py
git commit -m "Convert lsst.images.json.formatters to a deprecation shim

GenericFormatter becomes a one-line subclass of the unified
GenericFormatter that emits a DeprecationWarning and overrides
default_extension to .json so the existing daf_butler config for
Transform/Projection keeps producing JSON output."
```

---

## Task 11: Add trailing `unittest.main()` block to `tests/test_formatters.py`

**Files:**
- Modify: `tests/test_formatters.py`

- [ ] **Step 1: Append the entrypoint**

Append exactly:

```python


if __name__ == "__main__":
    unittest.main()
```

(Two blank lines before, matching the rest of the file's style.)

- [ ] **Step 2: Verify direct execution works**

```bash
python tests/test_formatters.py -v
```

Expected: all tests run; NDF cases skip if h5py is absent. This confirms the file is runnable under raw unittest (matches the convention from the optional-h5py work).

- [ ] **Step 3: Commit**

```bash
git add tests/test_formatters.py
git commit -m "Add unittest.main entrypoint to test_formatters

Allows running the test file directly with python tests/test_formatters.py
in addition to pytest."
```

---

## Task 12: Re-export from `lsst.images` if desired (optional, deferred)

Per the spec, the unified formatter lives at `lsst.images.formatters` (plural module). The top-level `lsst.images` package does not need to re-export the formatter classes — butler configs use the full dotted path. Skip this task unless reviewers ask for it.

---

## Task 13: Full lint / typecheck / docs / test pass

The deployed `daf_butler/configs/datastores/formatters.yaml` still
points at `lsst.images.fits.formatters.X` / `lsst.images.json.formatters.GenericFormatter`.
After Tasks 9 and 10 land the shims, the existing roundtrip suite
(`RoundtripFits(self, image, "ImageV2")`, etc.) routes through the shims
to the unified code path. That existing coverage is the integration
test — no extra integration test is added.

- [ ] **Step 1: Run linters**

```bash
ruff format --check python/ tests/
ruff check python/ tests/
mypy python/lsst/images
```

Expected: all green. If any fail, fix and re-run.

- [ ] **Step 2: Run the full test suite**

```bash
pytest tests/ -q \
  --deselect tests/test_visit_image.py::VisitImageLegacyTestCase::test_butler_converters \
  --deselect tests/test_visit_image.py::VisitImageLegacyTestCase::test_rewrite
```

Expected: all pass (180+ tests + the new ~20 in `test_formatters.py`). The two deselected tests are pre-existing failures from an out-of-date `daf_butler` install, not caused by this change.

- [ ] **Step 3: Run with h5py blocked**

Use the same `meta_path` finder pattern from the previous commit:

```python
# /tmp/run_tests_no_h5py.py
import sys

class _BlockedFinder:
    def find_spec(self, name, path=None, target=None):
        if name == "h5py":
            from importlib.machinery import ModuleSpec
            return ModuleSpec(name, self)
        return None
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        raise ImportError("h5py blocked")

sys.meta_path.insert(0, _BlockedFinder())
import pytest
sys.exit(pytest.main(sys.argv[1:]))
```

```bash
python /tmp/run_tests_no_h5py.py tests/ -q \
  --deselect tests/test_visit_image.py::VisitImageLegacyTestCase::test_butler_converters \
  --deselect tests/test_visit_image.py::VisitImageLegacyTestCase::test_rewrite
```

Expected: NDF-touching tests skip; the rest pass. The unified-formatter tests that exercise the FITS and JSON paths must still pass; the SDF-tagged ones must skip cleanly.

- [ ] **Step 4: Run docs build with CI flags**

```bash
rm -rf doc/api doc/_build
package-docs build -W -n
```

Expected: `build succeeded.` with no `ref.obj`-class warnings.

- [ ] **Step 5: Final commit (if anything was tweaked)**

```bash
git status
# If any fixes were needed:
git add -A
git commit -m "Final cleanups for unified formatter"
```

---

## Out-of-scope reminders

- **`FitsCompressionOptions` naming** — flagged in the spec; do not touch as part of this plan.
- **`daf_butler` config update** — the user is handling the daf_butler PR separately.
- **News fragment / changelog** — none added (per stakeholder direction).
- **`format=ndf` alias** — not added preemptively; the file extension `.sdf` is the canonical name.

## Self-review (post-write check)

- Spec coverage: every section of the design spec maps to a task — _BACKENDS table (Task 1), write path (Task 2), read path (Task 3), ImageFormatter (Task 4), MaskedImageFormatter (Task 5), VisitImageFormatter (Task 6), CellCoaddFormatter (Task 7), NDF deletion (Task 8), FITS shim (Task 9), JSON shim (Task 10), test entrypoint (Task 11), full verification (Task 13). ✓
- Placeholders: none — every step shows actual code or actual commands.
- Type consistency: `_BACKENDS`, `_Backend`, `ComponentSentinel`, `GenericFormatter`, `ImageFormatter`, `MaskedImageFormatter`, `VisitImageFormatter`, `CellCoaddFormatter` names are stable across tasks. `_read_component_from_uri` is defined in Task 4 and used in Tasks 5–7. `_write_parameters` and `_storage_class_pytype` private attributes are set the same way across all test cases.
- Ambiguity check: the JSON `_read_component_from_uri` path uses `getattr(obj, component)` consistently and raises `NotImplementedError` on `AttributeError` (matching what FITS/NDF raise via `ComponentSentinel.UNRECOGNIZED_COMPONENT`).
