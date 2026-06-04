.. py:currentmodule:: lsst.images

.. _lsst.images-schema-versioning:

#################
Schema versioning
#################

Every serialized ``lsst.images`` data product records enough version information to answer one question on read: *can this release safely interpret these bytes?*
This page describes the scheme, what it writes, and how to evolve a schema over time.

Two version axes
================

Versioning happens on two independent axes.

**Data-model version.**
   Each concrete `~lsst.images.serialization.ArchiveTree` subclass declares a ``major.minor.patch`` schema version and a minimum reader major.
   Both are written into every serialized JSON tree and travel with it regardless of the container that holds it.
   The version bumps when the Pydantic shape of *that* subclass changes.

**Container (file-format) version.**
   The FITS and NDF backends each carry a single integer that bumps when the backend *layout* changes (HDU placement, ``NdfDocument`` shape), not when any data model changes.
   JSON has no container distinct from the root tree, so its data-model version covers everything.

The axes evolve separately: changing the FITS HDU layout bumps the FITS container version without touching any data model, and bumping a model from ``1.0.0`` to ``1.1.0`` touches no container version.

The data-model version fields
=============================

`~lsst.images.serialization.ArchiveTree` declares three class-level constants on every concrete subclass:

``SCHEMA_NAME``
   The lowercase, hyphen-free schema name (e.g. ``image``, ``cell_coadd``, ``piff_psf``).
   Used to build the schema URL and the JSON Schema title.

``SCHEMA_VERSION``
   The full ``major.minor.patch`` version of the data-model shape.

``MIN_READ_VERSION``
   The smallest reader major that can safely interpret a tree written by this code.

Two of these are reflected into per-instance fields that round-trip through JSON, ``schema_version`` and ``min_read_version``, plus a computed ``schema_url``:

.. code-block:: json

   {
     "schema_url": "https://images.lsst.io/schemas/image-1.0.0",
     "schema_version": "1.0.0",
     "min_read_version": 1,
     "data": { "...": "..." }
   }

Declaring the fields on the base class places them ahead of every subclass's fields, so a human reading a JSON tree sees them first.

URL scheme
==========

``schema_url`` follows::

   https://images.lsst.io/schemas/<schema-name>-<major>.<minor>.<patch>

It is informational and need not resolve to a hosted document.
The same URL appears in the FITS ``DATAMODL`` keyword and the NDF ``.MORE.LSST.DATA_MODEL`` component so the data model is visible to tooling without parsing the JSON tree.

Why two fields per model
========================

The symmetric rule "different major rejects in either direction" is wrong by default for the most common breaking changes (adding a required field, adding a discriminated-union variant).
Those want **old code to reject new files** but **new code to keep reading old files**.

Splitting the version into ``(schema_version, min_read_version)`` lets the writer say "here is the shape; here is the smallest reader major that can safely interpret it."
Bumping ``schema_version`` from ``1.0.0`` to ``2.0.0`` does not force ``min_read_version`` to move: old readers may be unable to read a ``2.0.0`` file, while new readers can still read a ``1.0.0`` file by defaulting the new field on input.

``min_read_version`` is an integer because only the major component drives compatibility.

The compatibility rule
======================

On read, ``min_read_version`` is the *sole* gate:

.. code-block:: text

   reject when  on_disk.min_read_version > this_release.major

The check deliberately ignores the on-disk ``schema_version`` major.
A redundant ``on_disk_major > in_code_major`` test would re-impose the symmetric rule and defeat the asymmetric escape (e.g., a ``2.0.0`` file deliberately written with ``min_read_version = 1`` so major-1 code can still read it).

The "new code reading an old file" direction is not gated here at all: if the current Pydantic model validates the older tree, the read succeeds; otherwise Pydantic raises its own validation error.
Making new code handle an older incompatible shape means adding backfill logic in the model validator (or, in the future, a migration).

Container versions are integer-only and gated the same way: a newer on-disk container version than the running release is rejected; older ones are accepted.

After a successful read the instance's version fields are normalized to the in-code constants, so re-serializing immediately re-stamps the tree at the current version.

Absence is the v1 default
=========================

Files written before versioning landed carry none of the stamps.
The reader treats their absence as the v1 defaults — ``schema_version = "1.0.0"``, ``min_read_version = 1``, container version ``1`` — so legacy files continue to read.
Once re-written by versioned code, the stamps appear.

Evolving a schema
=================

When the Pydantic shape of a subclass changes, bump its ``SCHEMA_VERSION``:

- **Backward-compatible addition** (a new optional field): bump the minor (``1.0.0`` → ``1.1.0``); leave ``MIN_READ_VERSION`` at 1.
- **Backward-incompatible change** (a new required field, a renamed or retyped field, a new discriminated-union variant): bump the major (``1.0.0`` → ``2.0.0``).
  Whether to also bump ``MIN_READ_VERSION`` is a *separate* decision driven only by "does the new shape mislead an old reader?":

  - If old readers can safely ignore the change, or new readers carry backfill logic for old files, leave ``MIN_READ_VERSION`` at 1.
  - If silently dropping the new data is dangerous, bump ``MIN_READ_VERSION`` so old code refuses the file.

Bump the container version (independently of any data model) only when the backend layout itself changes.

A patch bump (``1.0.0`` → ``1.0.1``) is for changes that do not affect file-format interpretation, such as documentation fixes.

A unit test enforces that every concrete subclass declares all three constants, that every ``SCHEMA_NAME`` is unique, and that ``MIN_READ_VERSION`` does not exceed the schema major.
It does *not* enforce that a shape change was accompanied by a version bump — that remains a review-time discipline.

Schema discovery and entry points
=================================

Concrete `~lsst.images.serialization.ArchiveTree` subclasses register themselves when their defining module is imported.
Schemas whose model classes are imported unconditionally by ``lsst.images`` need no additional discovery metadata: for example, ``VisitImageSerializationModel`` is already imported by the core package and is registered before generic reads need it.

Models in subpackages or external packages may not be imported before `lsst.images.serialization.read` inspects a file's ``schema_url``.
Those packages should expose a schema-specific entry point in the ``lsst.images.schemas`` group, with the entry point name matching ``SCHEMA_NAME`` and the value pointing at the serialization model class:

.. code-block:: toml

   [project.entry-points."lsst.images.schemas"]
   extended_psf_image = "lsst.pipe.tasks.extended_psf.extended_psf_image:ExtendedPsfImageSerializationModel"
   extended_psf_candidates = "lsst.pipe.tasks.extended_psf.extended_psf_candidates:ExtendedPsfCandidatesSerializationModel"

When `~lsst.images.serialization.class_for_schema` cannot find a schema in the in-memory registry, it loads only entry points with the requested schema name.
Loading the entry point imports the model's module, which triggers the normal subclass registration hook.
The entry point does not need to call `~lsst.images.serialization.register_schema_class` directly.

The entry point is keyed by schema name only, not by ``SCHEMA_VERSION``.
Version compatibility remains the responsibility of the selected model's ``schema_version`` / ``min_read_version`` validation.

``lsst.images`` also maintains a small built-in lazy-provider table for schemas it owns but does not import unconditionally, such as the ``lsst.images.cells`` models.
This mirrors the package's own ``lsst.images.schemas`` entry points while keeping source-tree development via ``PYTHONPATH=python`` working before the package is installed.

Embedded external models
========================

Some subclasses embed Pydantic models from outside this package (e.g. ``astro_metadata_translator.ObservationInfo``).
These do not get their own stamp; their effective version is tied to the containing tree's ``SCHEMA_VERSION``.
If an upstream model changes shape in a way that breaks older files, the containing tree must bump its ``SCHEMA_VERSION`` (and possibly ``MIN_READ_VERSION``) to express that.
The on-read failure mode for an unanticipated upstream change is a Pydantic validation error rather than a clean compatibility error; callers should treat both as "this release cannot read this file."

Future work
===========

Several extensions to this scheme have been designed but not implemented, including deferred-fail sub-model substitution (failing an incompatible sub-model at its point of use rather than rejecting the whole tree), a migration framework, and a schema-snapshot regression test.
See :ref:`lsst.images-schema-versioning-future`.

.. toctree::
   :maxdepth: 1

   schema-versioning-future.rst
