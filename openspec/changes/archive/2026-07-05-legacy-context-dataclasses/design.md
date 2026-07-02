## Context

`tests/test_visit_image.py` has two parametrized session-scoped fixtures — `legacy_dataset_context` and `legacy_nJy_dataset_context` — that return `dict[str, Any]`. A third fixture, `legacy_visit_image`, depends on `legacy_dataset_context` solely to pre-read a `VisitImage` and hand it to five tests that also receive `legacy_dataset_context` directly. All field access throughout the file uses string subscript syntax with no type safety.

`tests/test_masked_image.py` already demonstrates the preferred pattern: a `@dataclasses.dataclass` with an optional-dependency type alias guarded by `try/except ImportError`.

## Goals / Non-Goals

**Goals:**
- Replace `dict[str, Any]` return types with a typed `_LegacyDatasetContext` dataclass.
- Absorb the `legacy_visit_image` fixture into the dataclass, eliminating it.
- Switch all consumers from `ctx["key"]` subscript to `ctx.key` attribute access.
- Remove the `legacy_visit_image` parameter from the five tests that currently receive both fixtures.
- Keep `Any` out of the picture for callers (the `try/except` guard is an annotation-only escape hatch).

**Non-Goals:**
- Changing any production code.
- Converting `visit_image_components` (kept as `dict[str, Any]` per explicit decision).
- Converting `_legacy_dataset_params`, which returns `list[pytest.param]` and is unaffected.
- Altering test logic or assertions in any way.

## Decisions

**Single dataclass for both fixtures**

Both `legacy_dataset_context` and `legacy_nJy_dataset_context` share the same six base fields; the nJy variant adds `visit_image`. Rather than two separate classes or inheritance, use one class with `visit_image` always present — both fixtures populate it. This avoids `visit_image: VisitImage | None` leaking into consumers while keeping the class hierarchy flat.

Alternatives considered:
- *Two independent dataclasses*: Duplicates six fields; rejected.
- *Inheritance (`_LegacyNJyDatasetContext(_LegacyDatasetContext)`)*: Adds complexity for a single extra field; rejected.
- *`visit_image: VisitImage | None = None`*: Forces `None`-checks everywhere; rejected.

**`legacy_exposure` type alias via `try/except ImportError`**

`lsst.afw.image.Exposure` is an optional stack dependency. Follow `test_masked_image.py`'s pattern:

```python
try:
    from lsst.afw.image import Exposure as LegacyExposure
except ImportError:
    type LegacyExposure = Any
```

The fixture already `pytest.skip()`s when the import fails, so the `Any` fallback is annotation-only and never used at runtime in a context where the field is actually `None`.

**`read_cls` typed as `type[VisitImage]`**

`DifferenceImage` is a subclass of `VisitImage`, so `type[VisitImage]` is accurate and sufficient. No union needed.

**Delete `legacy_visit_image` fixture**

It is a thin wrapper that calls `ctx["read_cls"].read_legacy(...)` — exactly what the updated `legacy_dataset_context` body now does to populate `ctx.visit_image`. Keeping both would be redundant and confusing.

## Risks / Trade-offs

[Parametrize input stays as dict] `_legacy_dataset_params` passes plain dicts into `pytest.param(...)`, which `request.param` returns as a dict. The fixture body must continue to do `dict(request.param)` and then construct the dataclass from those keys. This is a minor layer of indirection but keeps the parametrize call readable. → No mitigation needed; pattern is standard pytest.

[Optional import alias scope] The `type LegacyExposure = Any` soft-alias uses Python 3.12 `type` statement syntax. Since the package requires Python ≥ 3.12 this is fine, and matches the style already present in `test_masked_image.py`.
