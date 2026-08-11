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
   The FITS, NDF, and zarr backends each carry a single integer that bumps when the backend *layout* changes (HDU placement, ``NdfDocument`` shape, zarr group and attribute layout), not when any data model changes.
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

   {SCHEMA_URL_BASE}/<schema-name>-<major>.<minor>.<patch>

where ``SCHEMA_URL_BASE`` is a class-level constant that defaults to ``https://images.lsst.io/schemas`` for the schemas this package owns.
External packages providing their own schemas override it so their URLs are minted under a documentation site they control; readers accept any ``http(s)`` URL whose final directory is ``schemas``, without restricting the host.
The same URL appears in the FITS ``DATAMODL`` keyword, the NDF ``.MORE.LSST.DATA_MODEL`` component, and the zarr root group's ``lsst.data_model`` attribute so the data model is visible to tooling without parsing the JSON tree.
The URL resolves to a generated documentation page for that schema version, with the raw JSON Schema document published alongside it at ``{schema_url}.json``; see :ref:`lsst.images-frozen-schemas`.
Readers never fetch it: version compatibility is decided entirely by the ``schema_version`` / ``min_read_version`` stamps.

Why two fields per model
========================

The symmetric rule "different major rejects in either direction" is wrong by default for the most common breaking changes (adding a required field, adding a discriminated-union variant).
Those want **old code to reject new files** but **new code to keep reading old files**.

Splitting the version into ``(schema_version, min_read_version)`` lets the writer say "here is the shape; here is the smallest reader major that can safely interpret it."
Bumping ``schema_version`` from ``1.0.0`` to ``2.0.0`` does not force ``min_read_version`` to move: old readers may be unable to read a ``2.0.0`` file, while new readers can still read a ``1.0.0`` file by defaulting the new field on input.

``min_read_version`` is an integer because only the major component drives compatibility.

The compatibility rule
======================

On read, ``min_read_version`` is the gate for the *old reader vs new file* direction:

.. code-block:: text

   reject when  on_disk.min_read_version > this_release.major

The check deliberately ignores the on-disk ``schema_version`` major.
A redundant ``on_disk_major > in_code_major`` test would re-impose a symmetric rule and reject the files this asymmetry exists to allow: a ``2.0.0`` file deliberately written with ``min_read_version = 1`` so major-1 code can still read it.

The "new code reading an old file" direction has no gate of this kind for a schema that has never registered a migration: if the current Pydantic model validates the older tree, the read succeeds; otherwise Pydantic raises its own validation error.
A schema that *has* registered at least one migration (see :ref:`lsst.images-schema-versioning-migration` below) adds a second gate on this same direction, ahead of Pydantic's own validation: an on-disk major with no registered step to the next major raises a clean `~lsst.images.serialization.ArchiveReadError` naming the missing step, regardless of what the tree actually contains.
This second gate is scoped to the schema being read — whether some *other*, unrelated schema has registered a migration is irrelevant to it.
Making new code handle an older incompatible shape means adding backfill logic in the model validator, or registering a migration for a schema that needs one.

Container versions are integer-only and gated the same way: a newer on-disk container version than the running release is rejected; older ones are accepted.

After a successful read the instance's version fields are normalized to the in-code constants, so re-serializing immediately re-stamps the tree at the current version.

Version stamps are required on disk
===================================

Every archive tree must carry an explicit ``schema_version``.
Archive backends mark their Pydantic validation with an internal read context, and `~lsst.images.serialization.ArchiveTree` rejects any tree in that context that omits the stamp rather than guessing which shape it contains.
Fresh in-memory construction remains free to omit the instance fields: serializer code constructs the current model directly, and its defaults are populated from the class constants before writing.

The container version is required for the same reason: a writer always stamps the layout it wrote, so a missing stamp is a damaged file rather than an old one, and guessing would guess at the layout.
Where each backend applies that differs only in how a file is recognized as one this package wrote.
Every FITS file the reader can open is ours already — the container cards it needs carry no defaults — so ``FMTVER`` is required outright, and a file without it is reported as such instead of failing later on a missing card.
An NDF, by contrast, can be a Starlink product with no LSST extension at all: those have no layout of ours to version and are read by `~lsst.images.ndf.read_starlink`, so ``FORMAT_VERSION`` is required only of a file carrying an LSST JSON tree.
A zarr store is recognized by its root group's ``lsst`` attribute namespace, which the writer always stamps with ``lsst.version``, so that attribute is required of any store carrying the namespace while a store without it is reported as not being one of ours.

Evolving a schema
=================

When the Pydantic shape of a subclass changes, bump its ``SCHEMA_VERSION``:

- **Backward-compatible addition** (a new optional field): bump the minor (``1.0.0`` → ``1.1.0``); leave ``MIN_READ_VERSION`` at 1.
- **Backward-incompatible change** (a new required field, a renamed or retyped field, a new discriminated-union variant): bump the major (``1.0.0`` → ``2.0.0``).
  Whether to also bump ``MIN_READ_VERSION`` is a *separate* decision driven only by "does the new shape mislead an old reader?":

  - If old readers can safely ignore the change, or new readers carry backfill logic for old files (in the model validator directly, or through a registered migration; see :ref:`lsst.images-schema-versioning-migration`), leave ``MIN_READ_VERSION`` at 1.
  - If silently dropping the new data is dangerous, bump ``MIN_READ_VERSION`` so old code refuses the file.

Bump the container version (independently of any data model) only when the backend layout itself changes.

A patch bump (``1.0.0`` → ``1.0.1``) is for changes that do not affect file-format interpretation, such as documentation fixes.

A unit test enforces that every concrete subclass declares all three constants, that every ``SCHEMA_NAME`` is unique, and that ``MIN_READ_VERSION`` does not exceed the schema major.
It does *not* enforce that a shape change was accompanied by a version bump — that remains a review-time discipline.

.. _lsst.images-schema-versioning-migration:

Schema migration (morphing v1 into v2)
=======================================

The asymmetric design above lets new code read an *old* file whenever the current Pydantic model can validate the older shape directly; that covers additive changes, where defaulting the new fields on input is enough.
A migration is what is needed when it is *not* enough: a backward-incompatible v2 (a renamed or retyped field, a split or merged field, a restructured sub-tree) that the v2 model cannot validate against a raw v1 tree.
The goal is to keep ``MIN_READ_VERSION = 2`` (so v1 code refuses v2 files it would otherwise mis-read) while still letting v2 code read v1 files by *morphing* the v1 tree into the v2 shape before validation.

A migration is a per-schema function that rewrites an on-disk tree from one major to the next, registered one adjacent major pair at a time (1→2, 2→3, …) with `~lsst.images.serialization.migration`:

.. code-block:: python

   type Migration = Callable[[dict[str, Any]], dict[str, Any]]

   _MIGRATIONS: dict[tuple[str, int], Migration] = {}
   _MIGRATABLE_NAMES: set[str] = set()


   def migration(schema_name: str, from_major: int) -> Callable[[Migration], Migration]:
       def register(func: Migration) -> Migration:
           key = (schema_name, from_major)
           if (existing := _MIGRATIONS.get(key)) is not None and existing is not func:
               raise RuntimeError(
                   f"A migration for {schema_name!r} major {from_major} is already "
                   f"registered to {existing.__qualname__}; refusing to replace it "
                   f"with {func.__qualname__}."
               )
           _MIGRATIONS[key] = func
           _MIGRATABLE_NAMES.add(schema_name)
           return func

       return register


   @migration("visit_image", 1)
   def _visit_image_1_to_2(data: dict[str, Any]) -> dict[str, Any]:
       # v2 renamed photo_calib -> photometric_scaling; morph the v1 tree.
       data["photometric_scaling"] = data.pop("photo_calib", None)
       return data

Registering only adjacent-major steps means the reader chains them to cross a larger gap, one step at a time; ``_MIGRATABLE_NAMES`` exists so a schema with no migration at all — every ``lsst.images`` schema today — can be recognized with a single set-membership test rather than a dictionary probe per major.

Migration itself runs in a ``mode="before"`` validator declared on `~lsst.images.serialization.ArchiveTree`, ahead of the per-instance compatibility check, chaining registered steps until the tree reaches the in-code major:

.. code-block:: python

   @pydantic.model_validator(mode="before")
   @classmethod
   def _migrate_from_older_major(
       cls, data: Any, info: pydantic.ValidationInfo
   ) -> Any:
       if not isinstance(data, dict):
           return data
       if not hasattr(cls, "SCHEMA_NAME"):
           return data
       name = cls.SCHEMA_NAME
       if "schema_version" not in data:
           if info.context is _ARCHIVE_READ_CONTEXT:
               raise _MissingSchemaVersionError(...)
           return data  # Fresh in-memory construction.
       if not _MIGRATABLE_NAMES:
           return data  # Fast path: nothing has a migration.
       if name not in _MIGRATABLE_NAMES:
           return data  # This schema itself has no registered migration.
       on_disk_major = _parse_on_disk_major(data["schema_version"])
       in_code_major = _parse_major(cls.SCHEMA_VERSION)
       if on_disk_major < in_code_major:
           data = deepcopy(data)  # Isolate failed union candidates.
       while on_disk_major < in_code_major:
           try:
               step = _MIGRATIONS[(name, on_disk_major)]
           except KeyError:
               raise _MigrationGapError(
                   f"{name}: no migration from major {on_disk_major} to {on_disk_major + 1}."
               ) from None
           data = step(data)
           on_disk_major += 1
           data["schema_version"] = f"{on_disk_major}.0.0"
       return data

The gate at ``name not in _MIGRATABLE_NAMES`` checks only whether *this* schema has registered a migration, independent of its major and independent of whether some other, unrelated schema has registered one — a schema with no migration of its own always falls through to ordinary Pydantic validation, whatever major it is at.
``_MigrationGapError`` is a narrow exception deriving from both `~lsst.images.serialization.ArchiveReadError` and `ValueError`: Pydantic only treats a `ValueError` (or `TypeError` or `AssertionError`) raised by a validator as "this candidate failed, try the next one" in a discriminated union, so a plain `RuntimeError`-only ``ArchiveReadError`` would abort a union read (e.g. ``VisitImageSerializationModel.psf``) entirely instead of falling through to a variant that does match.
Deriving from both fixes that, at a cost: Pydantic wraps *any* validator-raised `ValueError` into its own `pydantic.ValidationError` (itself a `ValueError`) at the boundary of whichever model is being validated, whether or not a union is involved, so a caller reading a schema with a genuine migration gap at the top level sees a `pydantic.ValidationError` rather than a bare ``ArchiveReadError`` — its message still names the missing step, and this is the same tradeoff already accepted for other schema-incompatibility failures (see :ref:`lsst.images-schema-fixtures`, where both exception types are treated as an equally valid "this shape is rejected").
Every failure this validator can raise carries the same dual nature for the same reason: a missing stamp, an unparsable one (``_parse_on_disk_major``, distinct from ``_parse_major`` precisely so that a malformed in-code ``SCHEMA_VERSION`` stays a plain `RuntimeError` a union cannot swallow), and a migration gap.
This matters most for an unparsable stamp, because the parse happens before any field validation and so runs for every migratable variant a union tries, including ones the payload does not belong to.

That dual nature is specific to this ``mode="before"`` validator and must not be extended to the ``mode="after"`` compatibility check, which stays a `RuntimeError`-only ``ArchiveReadError`` and so aborts a union outright.
The difference is which candidates each one can fire on.
A before-validator runs ahead of field validation, so it judges variants the payload may have nothing to do with, and a failure there means "not this one" rather than "not readable".
The compatibility check runs only once a candidate has validated structurally, so the tree really is that variant; letting a union fall through from a ``min_read_version`` the reader cannot satisfy would hand the tree to whichever *other* variant happened to accept it, producing an object built by the very model the file declared must not read it — the silent misread that bumping ``MIN_READ_VERSION`` exists to prevent.

After it runs the tree is in the current shape, so the existing ``mode="after"`` validator's compatibility check and normalization proceed unchanged; the instance ends up stamped with the in-code version, and re-serializing writes a v2 file.

The compatibility check deliberately runs in the ``mode="after"`` validator for performance: pydantic-core has already parsed the input by then, so the check itself is cheap.
The before-validator above preserves that fast path rather than undoing it, returning immediately unless the schema actually has a registered migration, so a migration-free schema pays only the ``_MIGRATABLE_NAMES`` truthiness test and, once inside this validator, one more set-membership test — orders of magnitude below what reading the node itself costs.

This is the exact complement of ``min_read_version``: ``min_read_version`` gates the *old reader vs new file* direction, while a migration handles the *new reader vs old file* direction.
A coherent breaking change therefore ships three things together — ``SCHEMA_VERSION = "2.0.0"``, ``MIN_READ_VERSION = 2``, and a registered ``(schema_name, 1)`` migration — after which v1 code rejects v2 files and v2 code transparently reads both.

Migrations compose down the tree: each ``ArchiveTree`` subclass migrates its own dict, and because the before-validator runs per sub-model, a nested v1 sub-tree is morphed by its own migration as the parent is validated.
If a sub-model has no migration registered across a gap, the read raises `~lsst.images.serialization.ArchiveReadError`; pairing migration with :ref:`lsst.images-schema-versioning-deferred-fail` would let one un-migratable sub-model fail at point-of-use rather than rejecting the whole tree.

How a migration gap surfaces
----------------------------

The gap error is raised from a validator, so Pydantic wraps it.
A standalone read of a tree whose major is below the oldest registered step therefore surfaces a `pydantic.ValidationError` rather than a bare `~lsst.images.serialization.ArchiveReadError`, with the ``no migration from major X to Y`` message preserved inside it.
Callers should treat the two alike, as :ref:`lsst.images-schema-versioning-embedded-external-models` already advises for an unanticipated upstream change.

The wrapping is deliberate rather than incidental.
The gap error derives from both `~lsst.images.serialization.ArchiveReadError` and `ValueError`, because Pydantic only treats a `ValueError` as a failed union member.
Without that, raising inside a plain union — such as the ``psf`` field of ``visit_image``, which tries its variants left to right — would abort the whole union instead of moving to the next variant, so a file whose PSF matched a *later* variant would become unreadable as soon as an *earlier* variant's schema gained a major.
A validator cannot tell whether it is the read's true target or merely a union candidate being tried, so the same exception has to serve both cases.

Migration validation context
----------------------------

Migration validators receive the same archive-read context used to enforce the required stamp.
Direct in-memory construction has no such context and therefore does not spuriously enter the migration chain.

The chain is proven by purpose-built schemas defined under ``tests/`` that are never frozen, published, or part of shipped data, with fixtures alongside the package's own under the same layout described in :ref:`lsst.images-schema-fixtures`.

.. _lsst.images-frozen-schemas:

Frozen schema files
===================

The JSON Schema for every serialization model is committed to the ``schemas/`` directory at the top of the repository as ``{name}/{name}-{version}.json``.
These files are the published source of truth for the canonical schema URLs: the documentation build generates one page per file at ``https://images.lsst.io/schemas/{name}-{version}``, with a field table, a composition diagram, and the raw JSON published alongside at ``https://images.lsst.io/schemas/{name}-{version}.json``.
The versionless URL ``https://images.lsst.io/schemas/{name}`` resolves to a per-schema index listing every published version, which is also how the version pages are grouped in the site navigation.

A schema still under development carries a PEP 440 development-release suffix on its ``SCHEMA_VERSION``, e.g. ``1.0.0.dev0``.
A development schema is never frozen, published, or documented, and writing a file that contains one emits a ``DevelopmentSchemaWarning``; it may be changed in place freely.
Dropping the suffix (``1.0.0.dev0`` becomes ``1.0.0``) finalizes the schema: the next ``lsst-images-admin schemas write`` from the repository root writes its first frozen file, the documentation build publishes it, and it becomes immutable.
A finalized frozen file is never overwritten; if you change a finalized model the ``test_committed_frozen_schemas_are_current`` test fails, and you must bump ``SCHEMA_VERSION`` (the new version develops at ``X.Y.(Z+1).dev0``) rather than overwrite it, so the superseded file and its published URL are kept forever.

Composition and version cascades
--------------------------------

A frozen schema inlines the sub-schemas of every embedded `~lsst.images.serialization.ArchiveTree` model, pinning the exact sub-schema versions its writer emits.
Which *older* sub-schema versions a reader accepts is not recorded in the schema document, because that is a property of the reader code and the per-node ``min_read_version`` gate described above.
Consequently a version bump in an embedded schema (e.g. ``sky_projection``) changes the frozen document of every schema that embeds it (e.g. ``visit_image``), and after the first data release those containing schemas must take a minor bump of their own even though their fields did not change.
``schemas write`` identifies exactly which containing schemas are affected.

.. _lsst.images-schema-fixtures:

Test fixtures
=============

Every retained fixture under ``tests/data/schemas`` is the instance-level twin of the frozen schema document of the same version under ``schemas``.
The layout mirrors it: ``{name}/{name}-{version}[-{variant}].json``, where ``{version}`` is ``X.Y.Z`` for a finalized schema or ``X.Y.Z.dev`` for one still in development.
A filename carries only the release part of the version, so a development schema has exactly one fixture path whatever its ``devN`` counter, and the counter lives only in the ``schema_version`` stamp inside the file.

Every fixture is read through its live model on every test run, so a model change that alters what a file looks like fails a test rather than passing silently.

Lifecycle
---------

Editing a development model means running ``lsst-images-admin fixtures refresh``, which rewrites the ``.dev`` fixture in canonical form; the fixture update lands in the same commit as the model change because ``fixtures check --schema-dir schemas`` fails while it is stale.
Finalizing a schema means dropping the ``.devN`` suffix, running ``schemas write`` to write the frozen document for the first time, and running ``fixtures freeze``, which writes the final-version fixture and deletes the ``.dev`` one.
Both directions are checked: a frozen document with no same-version fixture and a fixture at a version that was never frozen are each reported, so an incomplete freeze cannot be committed unnoticed.
Finalizing is also the point at which to review what the fixture set actually exercises, because a frozen fixture cannot be widened afterwards without a ``SCHEMA_VERSION`` bump; `Coverage`_ is how to see what it reaches and `Variants`_ is how to widen it.

Retiring a fixture means moving it into the schema's ``retired`` subdirectory, where the contract inverts: it must then raise `~lsst.images.serialization.ArchiveReadError`.
``MIN_READ_VERSION`` does not retire a fixture on its own — it gates old readers refusing new files, and says nothing about new code refusing old files.
A fixture is retired when the current model genuinely can no longer validate that shape and no migration covers it, and retirement is how a read contract ends: it asserts the new behavior rather than merely stopping to test the old one.
Because retirement keeps a *superseded* version's fixture, a retired file must be at a version older than the live one; a retired fixture at the live version is reported, since nothing else would catch a file misplaced there — self-consistent stamps and being rejected are what a retired fixture is supposed to look like.

Why ``refresh`` stays strict
-----------------------------

``refresh_schema_fixtures`` refuses to rewrite a fixture at a finalized version, raising `~lsst.images.tests.SchemaFixtureError` rather than silently updating it; the remedy it names is bumping ``SCHEMA_VERSION``, not editing the frozen fixture in place.
It stays strict even when a finalized fixture is genuinely stale, because fixture drift is the *only* signal available when a writer switches which union branch it emits: that kind of change leaves the JSON Schema document byte-identical, so relaxing ``refresh`` to accept "the schema document didn't change" would remove the one thing that would ever catch it.
Correcting such a fixture is therefore a deliberate, reviewed, by-hand act.
Two were corrected that way once (``cell_psf-1.0.0`` and ``cell_aperture_correction_map-1.0.0``, which still spelled ``cell_shape`` as ``[4, 4]`` after the writer began emitting ``{"y": 4, "x": 4}``), against `~lsst.images.serialization.check_frozen_schemas` confirming no finalized model had changed.

What the checks prove
----------------------

Reading a fixture proves only that it was accepted.
Storing it in canonical form — ``model_dump_json(indent=2)`` plus a trailing newline — and asserting byte-identity proves the read was lossless, because a dropped, mis-defaulted, or renamed field surfaces as a diff.
That check cannot police the payload itself: it compares a fixture against what the model makes of that same fixture, so any edit that still round-trips cleanly is self-consistent by construction, and editing a plain data value inside the fixture does not fail it.
For a version older than the current one that check cannot apply at all, since one model class serves every version and re-serializing always emits the current shape; instead the older fixture's read is compared against the current-version fixture on every path the older file actually expresses, with later-born fields ignored.
When a migration renames or restructures a path, that version pair must be listed in the test's explicit expected-divergence mapping with a reason, and the migration's dedicated test must assert the transformed value directly.
This makes every place where the generic projection oracle cannot apply a reviewed declaration rather than a silent omission.
What pins the actual payload data is, instead, the ``as_shipped``/``canonical`` pairwise check described below, the cross-version projection check just described, and git history; none of those three is satisfied by a value that merely happens to round-trip.

Variants
--------

A variant is a further fixture of the same schema at the same version, marked by a ``-{variant}`` suffix on the filename; variant names match ``[a-z0-9_]+``.
Variants exist because one exemplar cannot express every branch of a composite model's optional and union fields, and because a fixture derived from a real file should sit beside a synthetic one rather than replace it.
``visit_image`` shows both: its synthetic base fixture takes the ``gaussian_psf`` branch of the ``psf`` union, while its ``dp1`` and ``dp2`` variants come from real files and take the ``piff_psf`` branch.

Every check applies to each variant independently, and the machinery tracks variants separately end to end: ``refresh`` seeds a missing fixture per variant from the newest existing version of that same variant, ``freeze`` carries every ordinary variant over to the finalized version, and the cross-version projection oracle pairs an older fixture with the current one of the same variant.

Adding a variant means writing the file and putting it in canonical form.
Write it at ``{name}/{name}-{version}-{variant}.json``, generated by serializing a model rather than hand-authored: the round-trip check cannot police payload values, as `What the checks prove`_ describes, so a hand-edited value is held by nothing but review.
``lsst.images.tests._minify_for_fixtures`` does this for a fixture derived from a real on-disk file, reading a real archive and writing back a small representative subset.
Then canonicalize it: ``fixtures refresh`` does that while the version is in development, but at a finalized version it refuses to write and reports the file as needing a version bump, so a variant added after the freeze has to be written canonically from the start.

The fixture sweep in ``tests/test_serialization_io.py`` also keys its expected-type mapping by ``(name, variant)`` and has to name the new pair, but that is not a step anyone needs to remember: the mapping is asserted to match the fixture tree exactly, so the suite fails until the pair is listed and the failure names what to add.

Coverage
--------

``lsst-images-admin fixtures coverage`` reports what the fixture set exercises, which is the question to ask when finalizing a version.
Coverage is credited to the schema that owns each stamped subtree rather than to the file it was found in: every embedded tree carries its own version stamp, so a container's fixture exercises each sub-model it embeds, and a schema reached from several containers is credited by all of them.
That aggregation is the point — a sub-model's coverage is the union over the whole tree, so the report can never claim one container fails to cover a schema that another one exercises.
The schemas where a variant buys the most are therefore the ones no container embeds, since those rest on their own fixtures alone.

The report has two parts.
*Absent* paths are the ones a schema declares that no fixture expresses, collapsed to the root of each absent subtree because an unset field takes its whole declared subtree with it.
*Positions* are the places a schema can hold another stamped schema, listing the candidates each one admits and the ones fixtures actually put there.
A position that misses a candidate is marked ``gap``, and a gap is information rather than a failure: ``visit_image`` admits several field types at ``photometric_scaling`` and several PSF models at ``psf``, and nothing obliges a fixture to reach all of them.
Knowing which are unreached is what tells you whether a fixture set covers what you assumed it did, so the command always exits zero.

Absences come in three kinds worth telling apart when reading the output: a field left unset, a branch of a union that no fixture takes, and a path only reachable through the out-of-line array form, which a JSON fixture cannot express at all because it inlines its arrays.
The report measures reach, not judgement — a path counted as expressed only proves some fixture put a value there, and what pins payload data is described in `What the checks prove`_.

Reserved variants
-----------------

Two variant names are reserved, and they come as a pair.
``as_shipped`` marks a fixture whose bytes are preserved exactly as a real shipped file produced them; it is never canonicalized or rewritten.
``canonical`` is its generated twin, and comparing the two pins how a shipped file is normalized on read.
``fixtures refresh`` may regenerate that twin only while its version is in development; at a finalized version a mismatch is a compatibility failure that requires a schema-version bump, not an artifact the command may overwrite.

``cell_coadd`` 1.0.0 is the only schema that is both finalized and shipped, and it shipped before validated schema management existed, so shipped files spell an XY pair as ``[4, 4]`` where the code now writes ``{"y": 4, "x": 4}``.
Both spellings must keep reading, and the ``as_shipped`` fixture is what records that requirement.

When fixture validation was introduced, fourteen of the twenty-three committed fixtures had drifted from what their models produce.
Thirteen of those were rewritten in canonical form; the fourteenth was ``cell_coadd``, whose drift is exactly the reserved XY spelling above, and it was deliberately left byte-identical rather than rewritten, with its canonical form emitted separately as the ``canonical`` twin instead.

External packages
------------------

An external package providing its own schemas can reuse this machinery: keep a fixture tree in the same layout, guard it with `lsst.images.tests.check_schema_fixtures` in its own test, and manage it with ``lsst-images-admin fixtures check --schema-dir <their schemas> --package <their.package>`` and the corresponding ``fixtures refresh|freeze --package <their.package>`` commands.

Schema discovery and entry points
=================================

Concrete `~lsst.images.serialization.ArchiveTree` subclasses register themselves when their defining module is imported.
Schemas whose model classes are imported unconditionally by ``lsst.images`` need no additional discovery metadata: for example, ``VisitImageSerializationModel`` is already imported by the core package and is registered before generic reads need it.

Models in subpackages or external packages may not be imported before `lsst.images.serialization.read_archive` inspects a file's ``schema_url``.
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

External packages that provide schemas should also:

- Override the ``SCHEMA_URL_BASE`` class variable (once, on a shared intermediate base class) so their schema URLs are minted under a documentation site they control; ``images.lsst.io`` only hosts the schemas this package owns.
- Freeze their schemas into their own repository with ``lsst-images-admin schemas write --package <their.package>``, and guard them with an equivalent of this package's ``test_committed_frozen_schemas_are_current`` test using `lsst.images.serialization.check_frozen_schemas`.
- Generate documentation pages for their site with `lsst.images.schema_docs.generate_schema_docs`, which links any embedded ``lsst.images`` sub-schemas back to their canonical ``images.lsst.io`` URLs.

``lsst.images`` also maintains a small built-in lazy-provider table for schemas it owns but does not import unconditionally, such as the ``lsst.images.cells`` models.
This mirrors the package's own ``lsst.images.schemas`` entry points while keeping source-tree development via ``PYTHONPATH=python`` working before the package is installed.

.. _lsst.images-schema-versioning-embedded-external-models:

Embedded external models
========================

Some subclasses embed Pydantic models from outside this package (e.g. ``astro_metadata_translator.ObservationInfo``).
These do not get their own stamp; their effective version is tied to the containing tree's ``SCHEMA_VERSION``.
If an upstream model changes shape in a way that breaks older files, the containing tree must bump its ``SCHEMA_VERSION`` (and possibly ``MIN_READ_VERSION``) to express that.
The on-read failure mode for an unanticipated upstream change is a Pydantic validation error rather than a clean compatibility error; callers should treat both as "this release cannot read this file."

Future work
===========

Further extensions to this scheme have been designed but not implemented, most notably deferred-fail sub-model substitution: failing an incompatible sub-model at its point of use rather than rejecting the whole tree.
See :ref:`lsst.images-schema-versioning-future` for this and other deferred items.

.. toctree::
   :maxdepth: 1

   schema-versioning-future.rst
