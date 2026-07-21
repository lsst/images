.. py:currentmodule:: lsst.images

.. _lsst.images:

###########
lsst.images
###########

Modern image types and serialization for Rubin Observatory data products.

The images library is the home of the Python data structures used to represent Rubin image data products (`cells.CellCoadd`, `VisitImage`, `DifferenceImage`, `ColorImage`), which carry extensive characterization and provenance information (e.g. :ref:`point-spread function models <reference-psf-models>` and :ref:`background models <reference-fields>`) as well as the actual `image <Image>` and (usually) `mask <Mask>` and variance arrays.

The library is heavily inspired by the `lsst.afw` package that has been used in previous Rubin Data Previews (and is still used almost exclusively in the Rubin pipelines), but it has been written from scratch with a focus on more intuitive interfaces, better interoperability with the rest of the Python ecosystem (especially `Astropy <https://www.astropy.org/>`__), and generally applying the lessons learned from more than a decade of working with the original `lsst.afw` image types.
While some optional functionality still depends on `lsst.afw` and other LSST Science Pipelines packages being importable, `lsst.images` is designed as a standalone package and can be installed on its own:

.. code:: sh

   pip install lsst-images

In addition to in-memory Python types, the images library contains a `serialization` framework for storing them via a combination of `FITS <fits>` and `JSON <json>` (with experimental support for `HDF5/NDF <ndf>`).
`Pydantic <https://pydantic.dev/docs/validation/latest/get-started/>`__ is used to represent serializable objects as a tree of simple types (mappable directly to JSON or YAML), while providing hooks for writing image-like arrays and tables in file-format-specific ways.
This approach should be extensible to any file format that can handle JSON-like data and arrays.
While we do not yet have an `ASDF <https://www.asdf-format.org/en/latest/>`__ serialization backend, we have adopted the ASDF schemas for describing times, units, tables, and array references, and hope to explore an ASDF implementation in the future.

See `DMTN-339 <https://dmtn-339.lsst.io>`__ for a more complete description of the rationale for this overhaul of LSST's image types and file formats.

Stability
---------

`lsst.images` is a new library that is still under heavy development, and we are not yet ready to declare the full Python API to be stable.
But we will not change any Python objects that correspond to LSST DP2 data products that have been released (e.g. `~cells.CellCoadd` for Early DP2) in a backwards-incompatible way without a major-release deprecation period
The types for unreleased data products and the serialization system are still subject to change without notice.

It is also likely that we will spin off a separate framework/building-blocks package in the future, leaving `lsst.images` as the home of just high-level the Rubin image product types.
This would be done with temporary forwarding aliases and at least a full major-release deprecation period as well.

User Guides
-----------

Conceptual and how-to guides for working with the `lsst.images` Python library.

.. toctree::
   :maxdepth: 2

   user-guide/index.rst

API Reference
-------------

Detailed information about Python interaces, organized by subpackage.

.. toctree::
   :maxdepth: 2

   reference/index.rst

Schemas
-------

Documentation for how concrete `lsst.images` types are stored.
All concrete `serialization` implementations use the same schemas, but map them to different on-disk representations.

.. toctree::
   :maxdepth: 1

   schema-versioning.rst
   schemas/index.rst

Changes
-------

.. toctree::
   :maxdepth: 2

   CHANGES.rst

For Developers
--------------

.. toctree::
   :maxdepth: 1

   admin.rst
