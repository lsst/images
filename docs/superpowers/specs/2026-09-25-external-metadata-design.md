# External metadata access through `GeneralizedImage.metadata` (DM-54770)

## Goal

Give users read-only, by-keyword access to metadata that was carried in from the source file but is not part of the data model, such as the primary FITS header of a raw or legacy image.
Today that information lives in the private `_opaque_metadata` attribute and is unreachable through public API.

The new access must not commit the data model to FITS.
The opaque metadata happens to be backed by `astropy.io.fits.Header` because the input files were FITS, but another input format (for example HDF5) would supply a different opaque metadata implementation with the same public interface on top.

## Non-goals

- No access to card comments or units.
- No writing, updating, or deleting of external values.
- No access to `COMMENT`, `HISTORY`, or blank cards.
  If history tracking is needed later it should get its own place in the data model.
- No access to non-primary HDU headers.
  Their remaining content (WCS, mask plane definitions, and similar) is already represented in the model.

## User-facing behavior

`GeneralizedImage.metadata` returns a `MetadataView`, a `collections.ChainMap` subclass that layers two sources:

- `image.metadata.native`: the existing flexible metadata dict that is serialized in the JSON tree.
  Keys are case-sensitive.
- `image.metadata.external`: a read-only view of the external metadata.
  Lookups are case-insensitive.

Reading `image.metadata[key]` returns the native value if `key` is present in native (exact case); otherwise it performs a case-insensitive lookup in external; otherwise it raises `KeyError`.

Writes and deletes through `image.metadata` go to native.
Deleting a key that exists only in external raises `KeyError`.

A write to native (through either `image.metadata` or `image.metadata.native`) is refused with `KeyError` if the key is not already present in native and it case-insensitively matches a key in external.
Updating a key that is already present in native is always allowed, even if it shadows an external key.
This matters because external metadata is attached after construction, so a native key read from an existing file may already shadow an external one; reads remain well-defined because native wins.

The constructor's `metadata` argument and the `metadata` property setter replace the native dict wholesale and do not check for shadowing.
This keeps the documented idiom `image.metadata = image.metadata.copy()` working for images whose native metadata already shadows an external key.

When a keyword is repeated in the external source, which value `image.metadata[key]` returns is an implementation detail and is documented as unspecified.
Callers that need every value use `get_all(key)`, which returns a tuple in source order.

## Components

### `ExternalMetadataValue` (in `lsst.images.serialization`)

`type ExternalMetadataValue = bool | int | float | complex | str | None`

This is a superset of `MetadataValue` because FITS permits complex values and keywords without a value.

### `ExternalMetadata` (abstract class in `lsst.images.serialization`)

A read-only `Mapping[str, ExternalMetadataValue]` with one extra method:

- `get_all(key: str) -> tuple[ExternalMetadataValue, ...]`: all values for `key` in source order; raises `KeyError` if absent.

Contract for implementations:

- `__getitem__`, `__contains__`, `get`, and `get_all` are case-insensitive.
- Iteration yields each key once, in the form the underlying source reports it.
- `len` counts distinct keys.
- For a repeated key, `__getitem__` returns one of its values; which one is unspecified.

The module also provides an empty implementation, used when an image has no opaque metadata.

### `OpaqueArchiveMetadata.external_metadata()`

The `OpaqueArchiveMetadata` protocol gains:

```python
def external_metadata(self) -> ExternalMetadata: ...
```

This is the format-neutral hook through which `GeneralizedImage` reaches external metadata.
`FitsOpaqueMetadata` is the only implementation of the protocol today.

### `FitsExternalMetadata` (in `lsst.images.fits`)

Returned by `FitsOpaqueMetadata.external_metadata()`.
It wraps the primary header (`headers[ExtensionKey()]`) by reference; the opaque headers are already documented as immutable and this class exposes no mutation.
If there is no primary header it behaves as empty.

- Lookups of `COMMENT`, `HISTORY`, and `""` raise `KeyError`, and those keywords are never yielded by iteration or counted by `len`.
- Values that astropy reports as `astropy.io.fits.card.Undefined` are returned as `None`.
- Keys are yielded as astropy reports them: upper case, and HIERARCH keywords without the `HIERARCH ` prefix (for example `LSST ISR UNITS`).
- Case-insensitive lookup relies on astropy's own keyword normalization.

### `NativeMetadata` (in `lsst.images`)

A `MutableMapping[str, MetadataValue]` wrapping the image's `_metadata` dict and the image's current `ExternalMetadata`.
`__setitem__` enforces the shadowing rule described above; all other operations delegate to the dict.
`update`, `setdefault`, and similar inherited methods go through `__setitem__` and therefore through the check.

### `MetadataView` (in `lsst.images`)

A subclass of `collections.ChainMap` whose `maps` are `[NativeMetadata, ExternalMetadata]`.
`ChainMap` already provides the required semantics:
lookup tries each map's own `__getitem__` in order, so native is exact-case and external is case-insensitive;
`in` and `get` follow the same rule;
writes and deletes go to the first map, so the shadowing check applies and deleting an external-only key raises `KeyError`;
iteration and `len` use the exact-case union of the keys of both maps.

The union is exact-case so that iteration agrees with lookup.
A native key `exptime` does not hide `image.metadata["EXPTIME"]`, which still reads the external value, so both `exptime` and `EXPTIME` are yielded.
Such pairs arise only from pre-existing data, because new clashing native keys are refused.

The subclass adds:

- `native` and `external` properties returning `maps[0]` and `maps[1]`.
- `get_all(key)`, returning `(native[key],)` if `key` is in native, otherwise `external.get_all(key)`.
- A `copy()` override returning a plain `dict` copy of native, so `image.metadata.copy()` behaves as it does today.

Equality is standard `Mapping` equality over the merged view.
The other `ChainMap` API (`maps`, `new_child`, `parents`) is inherited but not documented as part of the interface.

### `GeneralizedImage.metadata`

The property constructs a new `MetadataView` (with its `NativeMetadata`) on every access.
The view holds references to `self._metadata` and to the external metadata obtained from `self._opaque_metadata` (or the empty implementation if that is `None`); no data is copied.
Building a new wrapper on each access means the view always reflects the current `_opaque_metadata`, which readers assign after construction (`serialization/_reader.py`).
A consequence is that `image.metadata is image.metadata` is `False`.

The setter accepts any `Mapping[str, MetadataValue]`; given a `MetadataView` it uses that view's native contents.

## Internal changes

- `FitsOpaqueMetadata.extract_legacy_primary_header` currently copies the header before popping the `LSST IMAGES KEY n` and `LSST IMAGES VALUE n` cards, so those cards remain in the stored opaque header.
  They must be removed from the stored copy so they do not appear in `external`.
- `Image.read_legacy`, `Mask.read_legacy`, and `MaskedImage._read_legacy_hdus` ignore the dict of native metadata that `extract_legacy_primary_header` returns; only the `VisitImage` readers use it.
  They must set the result's native metadata from it (including for component reads), because once the cards are stripped from the opaque header these readers would otherwise lose the native keys entirely.
  When `VisitImage` passes its own opaque metadata into `MaskedImage._read_legacy_hdus`, it has already extracted the dict, and that path is unchanged.
  `MaskedImage.to_legacy` cannot write these cards itself, because it returns an afw `MaskedImageF`, which has no metadata; a legacy `Exposure` file, or a `MaskedImageF` written with `writeFits(..., metadata=...)`, can carry them.
- Internal code that needs the plain native dict uses `self._metadata` instead of `self.metadata`:
  constructor and `model_construct` passes such as `metadata=self.metadata` in `_image.py`, `_mask.py`, `_masked_image.py`, `_color_image.py`, `cells/_coadd.py`, and `tests/_minify_for_fixtures.py`, the loop in `MaskedImage._fill_legacy_metadata`, and the reads and writes of the `id` key in `_visit_image.py`.
  The `id` accesses matter because, through the view, a missing native `id` would fall through to an external `ID` card, and writing `id` would be refused if such a card existed.
- `MaskedImage._fill_legacy_metadata` must write only native keys as `LSST IMAGES KEY/VALUE` cards; the external cards are already written separately from the opaque primary header.
- `_transfer_metadata` continues to use `self._metadata`, so native metadata sharing between an image and its subimages is unchanged.
- `lsst.images.tests._checks` compares `.native` for value equality and the underlying dicts for its view-sharing (`is`) check, so that an image round-tripped through a format that drops opaque metadata is not reported as different.

## Documentation

- Update the `GeneralizedImage.metadata` docstring to describe the native/external layering, case sensitivity, shadowing rule, and unspecified choice for repeated keys.
- Add the new public classes to the API reference.
- Add `doc/changes/DM-54770.api.md` for the change in what `metadata` returns and `doc/changes/DM-54770.feature.md` for external metadata access.

## Testing

Tests build their own data; no new files are added to `testdata_images`.

1. View unit tests using a hand-built `astropy.io.fits.Header` wrapped in `FitsOpaqueMetadata`:
   native precedence over external;
   case-insensitive external lookup;
   a repeated keyword returns one of its values from `[]` and all of them in order from `get_all`;
   `COMMENT`, `HISTORY`, and blank cards are absent from lookup, `in`, iteration, and `len`;
   `Undefined` values become `None`;
   HIERARCH keys;
   iteration yields each exact-case key once, includes an external key whose case differs from a pre-existing native key, and agrees with lookup for every yielded key;
   deleting an external-only key raises;
   empty external when there is no opaque metadata.
2. Shadowing tests:
   a new native key that clashes with an external key in any case raises `KeyError` via both `image.metadata[...]` and `image.metadata.native[...]`;
   `update()` applies the same check;
   updating an existing shadowing native key succeeds;
   the constructor and setter accept shadowing keys.
3. Native round trip: construct a `MaskedImage`, attach a `FitsOpaqueMetadata` whose primary header contains example cards (including a repeated keyword, a HIERARCH keyword, `COMMENT`/`HISTORY`, and a valueless keyword), set some native keys, write it in native FITS form, read it back, and check `native` and `external`.
   Repeat for NDF.
4. Legacy round trip, skipped when `lsst.afw` is unavailable: build a synthetic `VisitImage` (as the existing `test_repeated_metadata_keys_legacy_round_trip` does), attach example cards and native keys, convert it with `to_legacy` and save it with `writeFits`, then
   (a) open the file with astropy and assert the `LSST IMAGES KEY/VALUE` cards are present in the primary header, and
   (b) read it back with `VisitImage.read_legacy` and assert those cards are absent from `external` while the native values are restored in `native`.
   `VisitImage` is used because `MaskedImage.to_legacy` returns an afw `MaskedImage`, which has no metadata, and `MaskedImage.read_legacy` does not restore native keys.
   Also check that external cards are not duplicated in the legacy output.
5. Legacy readers, skipped when `lsst.afw` is unavailable: write a `MaskedImage` with `to_legacy().writeFits(..., metadata=...)`, passing a `PropertyList` filled by `_fill_legacy_metadata`, then read it with `MaskedImage.read_legacy` (full and component reads), `Image.read_legacy`, and `Mask.read_legacy`, and assert each restores the native keys and keeps no `LSST IMAGES` cards in its opaque header.
6. The existing test suite passes, and all changes are ruff and mypy clean.
