.. py:currentmodule:: lsst.images

.. _lsst.images-schema-versioning-future:

#############################
Future schema-versioning work
#############################

This page records design discussions for schema-versioning features that are **not implemented** in the current release.
It exists so that the reasoning is not lost and so a future implementer can pick the work up without re-deriving it.
Nothing described here is in the code today; see :ref:`lsst.images-schema-versioning` for what *is* implemented.

.. _lsst.images-schema-versioning-deferred-fail:

Deferred-fail sub-model substitution
====================================

The v1 behavior is hard-fail: any version mismatch in any sub-model rejects the entire read.
The design below would instead let an incompatible *sub-model* fail at its point of use while the rest of the tree still reads.
**It is not implemented.**

Motivation
----------

When reading a `~lsst.images.MaskedImage` whose ``psf`` sub-model claims a ``min_read_version`` newer than the running release, hard-fail rejects the whole file even though the image plane is fine.
The PSF code already defers a *different* failure to point-of-use: ``deserialize()`` raises `~lsst.images.serialization.ArchiveReadError` when an optional dependency (e.g. ``piff``) is missing rather than at read time.
Schema incompatibility could behave the same way — substitute a placeholder that validates trivially but raises when actually used.

The ``_ReadFailed`` substitute
------------------------------

A single generic substitute class is enough, because the only behavior to override is ``deserialize`` / ``deserialize_component``; no per-subclass substitute is needed:

.. code-block:: python

   class _ReadFailed(ArchiveTree):
       """Substitute for an ArchiveTree subclass whose written
       schema_version/min_read_version is incompatible.  Validates
       trivially; raises on use.
       """
       on_disk_data: dict[str, Any]
       reason: str
       original_class: str  # SCHEMA_NAME of the model we replaced

       def deserialize(self, archive, **kwargs):
           raise ArchiveReadError(self.reason)

Where substitution happens
--------------------------

The v1 compatibility check runs in a ``mode="after"`` validator for performance: pydantic-core has already parsed the input dict into a concrete instance by the time it runs, so it cannot swap in a different model class.
Substitution therefore needs a ``mode="before"`` validator that runs *only when deferred-fail is enabled*:

.. code-block:: python

   @pydantic.model_validator(mode="before")
   @classmethod
   def _maybe_substitute_read_failed(cls, data, info):
       if not isinstance(data, dict):
           return data
       if not _deferred_failures_enabled(info.context):
           return data  # Fast path: the after-validator handles the check.
       on_disk_version = data.get("schema_version", "1.0.0")
       on_disk_min_read = data.get("min_read_version", 1)
       try:
           _check_compat(cls.SCHEMA_NAME, on_disk_version, on_disk_min_read, cls.SCHEMA_VERSION)
       except ArchiveReadError as exc:
           return _ReadFailed.placeholder_dict(cls, on_disk_data=data, reason=str(exc))
       return data

The before-validator is a no-op when deferred-fail is off (the common case), so the after-validator's fast path is preserved.
When it is on, the before-validator runs the compatibility check itself and either substitutes a ``_ReadFailed`` or returns the dict unchanged.
``info.context`` is Pydantic's per-validation context dict, set by the input-archive layer when the caller opts in.

Caller-facing API
-----------------

.. code-block:: python

   def read(..., defer_schema_failures: bool = False) -> ArchiveTree: ...

The flag would flow through the input archive into ``info.context``.
The default stays ``False`` even after this lands.

Known limitation: unknown union variants
----------------------------------------

Pydantic discriminator validation runs *before* per-subclass model validators, so an *unknown* discriminated-union variant tag is detected before the substitution path runs.
Two cases:

#. *Known variant, version mismatch* — the variant resolves, the validator runs, substitution applies.
   Works.
#. *Unknown variant* — Pydantic raises before the data is seen; substitution would require intercepting at the union level.

Case 2 is out of scope even when deferred-fail lands; it is documented so a future implementer is not surprised.

Testing retroactively
----------------------

Deferred-fail can be tested without producing real incompatible files: hand-craft fixtures whose ``min_read_version`` or ``schema_version`` is set to an incompatible value, read them with ``defer_schema_failures=True``, and assert the resulting tree carries ``_ReadFailed`` instances in the right places.

.. _lsst.images-schema-versioning-migration:

Schema migration (morphing v1 into v2)
======================================

The asymmetric design already lets new code read an *old* file whenever the current Pydantic model can validate the older shape directly; that covers additive changes, where defaulting the new fields on input is enough.
A migration is what is needed when it is *not* enough: a backward-incompatible v2 (a renamed or retyped field, a split or merged field, a restructured sub-tree) that the v2 model cannot validate against a raw v1 tree.
The goal is to keep ``MIN_READ_VERSION = 2`` (so v1 code refuses v2 files it would otherwise mis-read) while still letting v2 code read v1 files by *morphing* the v1 tree into the v2 shape before validation.

A migration is a per-schema function that rewrites an on-disk tree from one major to the next.
Registering them one major at a time (1→2, 2→3, …) means only adjacent-major transforms are ever written, and the reader chains them to cross a larger gap.

.. code-block:: python

   # One entry per (schema_name, from_major); each bumps a single major.
   _MIGRATIONS: dict[tuple[str, int], Callable[[dict], dict]] = {}

   def migration(schema_name: str, from_major: int):
       def register(func):
           _MIGRATIONS[(schema_name, from_major)] = func
           return func
       return register

   @migration("visit_image", 1)
   def _visit_image_1_to_2(data: dict) -> dict:
       # v2 renamed photo_calib -> photometric_scaling; morph the v1 tree.
       data["photometric_scaling"] = data.pop("photo_calib", None)
       return data

Migration runs in a ``mode="before"`` validator, ahead of the per-instance compatibility check, chaining registered steps until the tree reaches the in-code major:

.. code-block:: python

   @pydantic.model_validator(mode="before")
   @classmethod
   def _migrate(cls, data):
       if not isinstance(data, dict):
           return data
       on_disk_major = _parse_major(data.get("schema_version", "1.0.0"))
       in_code_major = _parse_major(cls.SCHEMA_VERSION)
       while on_disk_major < in_code_major:
           try:
               step = _MIGRATIONS[(cls.SCHEMA_NAME, on_disk_major)]
           except KeyError:
               raise ArchiveReadError(
                   f"{cls.SCHEMA_NAME}: no migration from major "
                   f"{on_disk_major} to {on_disk_major + 1}."
               )
           data = step(data)
           on_disk_major += 1
           data["schema_version"] = f"{on_disk_major}.0.0"
       return data

After it runs the tree is in the current shape, so the existing ``mode="after"`` validator's compatibility check and normalization proceed unchanged; the instance ends up stamped with the in-code version, and re-serializing writes a v2 file.

This is the exact complement of ``min_read_version``: ``min_read_version`` gates the *old reader vs new file* direction, while a migration handles the *new reader vs old file* direction.
A coherent breaking change therefore ships three things together — ``SCHEMA_VERSION = "2.0.0"``, ``MIN_READ_VERSION = 2``, and a registered ``(schema_name, 1)`` migration — after which v1 code rejects v2 files and v2 code transparently reads both.

Migrations compose down the tree: each ``ArchiveTree`` subclass migrates its own dict, and because the before-validator runs per sub-model, a nested v1 sub-tree is morphed by its own migration as the parent is validated.
If a sub-model has no migration registered across a gap, the read raises `~lsst.images.serialization.ArchiveReadError`; pairing migration with :ref:`lsst.images-schema-versioning-deferred-fail` would let one un-migratable sub-model fail at point-of-use rather than rejecting the whole tree.
The committed ``tests/data/schema_v1/`` fixtures make this testable without time travel: a migration test reads each retained old-version fixture under current code and asserts it morphs to the current shape and round-trips.

Other deferred items
====================

- **Per-instance ``min_read_version`` for union variants.**
  For a new discriminated-union variant we would want only the files that actually contain the new variant to carry a higher ``min_read_version``, so old readers reject just those files.
  The field is already a normal Pydantic field, so the mechanism exists; what is missing is a convention for setting it per-instance at write time.

- **Schema-snapshot test.**
  Diff each tree's ``model_json_schema()`` against a committed snapshot and fail on change, to catch a shape change that was not accompanied by a ``SCHEMA_VERSION`` bump.
  Currently this is only a review-time discipline.

- **Schema hosting.**
  The ``schema_url`` values follow the ``https://images.lsst.io/schemas/<name>-<version>`` pattern but need not resolve to a fetchable document yet.

- **Forensic version stamping.**
  Record upstream package versions (e.g. ``astro_metadata_translator.__version__``) and/or ``lsst.images.__version__`` into the tree ``metadata`` (or the FITS primary header) at write time, useful if embedded external models drift more often than expected.

- **Fold ``ObservationSummaryStats.version``.**
  The ad-hoc ``version: int`` field on ``ObservationSummaryStats`` predates this scheme; folding it in is a mechanical follow-up.
