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

Other deferred items
====================

- **Per-instance ``min_read_version`` for union variants.**
  For a new discriminated-union variant we would want only the files that actually contain the new variant to carry a higher ``min_read_version``, so old readers reject just those files.
  The field is already a normal Pydantic field, so the mechanism exists; what is missing is a convention for setting it per-instance at write time.

- **Backend-owned inline array threshold.**
  ``SplineField.serialize`` inlines a small data array rather than writing it through ``add_array``, because reading a handful of numbers out of the JSON tree is much cheaper than opening a separate binary extension for them.
  It is the only model that makes this decision.
  The general fix is to push the policy down into each backend's ``add_array`` so every model benefits; every array-holding field already accepts the ``ArrayReferenceModel | InlineArrayModel`` union, so no frozen document would change, and only ``ArrayReferenceQuantityModel.value`` would need widening.

- **A size cap when inlining arrays in JSON.**
  ``JsonOutputArchive.add_array`` inlines unconditionally, which is why a 100 MB FITS ``CellCoadd`` converts to roughly a 600 MB JSON file.

- **FITS-backed fixtures.**
  Every fixture is JSON today, so no fixture exercises the ``ArrayReferenceModel`` branch of those unions.

- **Schema-snapshot test.**
  Diff each tree's ``model_json_schema()`` against a committed snapshot and fail on change, to catch a shape change that was not accompanied by a ``SCHEMA_VERSION`` bump.
  Currently this is only a review-time discipline.

- **Schema hosting.**
  The ``schema_url`` values follow the ``https://images.lsst.io/schemas/<name>-<version>`` pattern but need not resolve to a fetchable document yet.

- **Forensic version stamping.**
  Record upstream package versions (e.g. ``astro_metadata_translator.__version__``) and/or ``lsst.images.__version__`` into the tree ``metadata`` (or the FITS primary header) at write time, useful if embedded external models drift more often than expected.

- **Fold ``ObservationSummaryStats.version``.**
  The ad-hoc ``version: int`` field on ``ObservationSummaryStats`` predates this scheme; folding it in is a mechanical follow-up.
