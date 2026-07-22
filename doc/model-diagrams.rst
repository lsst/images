.. py:currentmodule:: lsst.images

.. _lsst.images-model-diagrams:

###############
Model diagrams
###############

The `lsst-images-admin diagram <admin-cli.html#lsst-images-admin-diagram>`__ subcommand renders the composition layout of an ``lsst.images`` serialization model: which model nests which sub-models, where unions branch, and which fields are lists or mappings.
It is useful for documentation, for presentations, and for understanding an unfamiliar data product.

The diagram describes the Python/Pydantic composition of a model, not the on-disk byte layout of any particular container.

Quick start
===========

Pass a schema name to diagram the abstract model.
Three output formats are available via ``--format`` (default ``mermaid``): ``mermaid`` and ``dot`` are graph descriptions for `Mermaid <https://mermaid.js.org>`_ and `Graphviz <https://graphviz.org>`_, and ``tree`` is an indented hierarchy in the style of the ``tree`` command.

.. code-block:: bash

   lsst-images-admin diagram visit-image                       # Mermaid (default)
   lsst-images-admin diagram cell-coadd --format dot | dot -Tsvg -o coadd.svg
   lsst-images-admin diagram image --format tree

Output is plain text written to standard output (or to ``--output PATH``); the subcommand never invokes a renderer itself.
List the schema names that can be diagrammed with ``--list``.

By default only model composition is shown; scalar fields (including schema bookkeeping such as ``schema_version``) are omitted, and an all-scalar model collapses to a bare leaf.
Pass ``--attributes`` to list the scalar fields as well.

Nodes are labelled with the public class name people interact with (``Image``, ``SkyProjection``, ...) rather than the underlying serialization model; the serialization helpers that have no public class keep their own names.
Pass ``--serialization-names`` to label everything with the raw serialization-model class names instead.

The ``tree`` format is the most convenient for a terminal or a documentation code block:

.. code-block:: text

   Image
   ├── butler_info?: ButlerInfo
   ├── data (one of):
   │   ├── ArrayReferenceQuantityModel
   │   ├── ArrayReferenceModel
   │   ├── InlineArrayModel
   │   └── InlineArrayQuantityModel
   └── sky_projection?: SkyProjection
       └── pixel_to_sky: Transform

Field-name markers record cardinality: ``?`` marks an optional field, ``*`` a list, and ``+`` a mapping.
A union field is shown as ``(one of)`` with one branch per member type, and ``…(other)`` marks a union that also admits non-model types (for example a raw ``Any``).
A field that points back to a model already on the current branch is marked ``(↻)`` rather than expanded again, so recursive models terminate.

Controlling detail
==================

Serialization-plumbing helpers that carry array or table payloads (``ArrayReferenceModel``, ``InlineArrayModel``, ``TableModel`` and similar) are collapsed to leaves by default, because their internals are uninteresting for a layout view.
There are many options to control whether fields or types are hidden or shown and whether a type should be collapsed.
The `full set of options are described on the CLI page <admin.html#lsst-images-admin-diagram>`__.

``--collapse`` and ``--expand`` match by name (the unparameterized class name), matching either the public or the serialization name, so ``--collapse Image`` and ``--collapse ImageSerializationModel`` are equivalent; ``--hide-field`` matches the field name.

Diagramming a concrete file
===========================

With ``--from-file PATH`` the diagram is built from a serialized file rather than from a model's type annotations.
Only the on-disk reference tree is read (pointers, not pixel data), so this is cheap even for large images.

Because it follows the actual stored values, a file diagram collapses each union to the member that is really present.
A ``VisitImage`` whose abstract diagram shows the PSF as ``Piff | PSFEx | Gaussian | …`` resolves, for a file with a Gaussian PSF, to:

.. code-block:: bash

   lsst-images-admin diagram --from-file visit_image.json --format tree

.. code-block:: text

   VisitImage
   ├── image: Image
   │   └── data: InlineArrayQuantityModel
   ├── psf: GaussianPointSpreadFunction
   ├── obs_info: ObservationInfo
   ├── summary_stats: ObservationSummaryStats
   └── ...

A file diagram reports only what the file holds: a model whose list or dict field is empty shows no edge for it, even if the model could in principle contain sub-models there.

``MODEL`` and ``--from-file`` are mutually exclusive; provide exactly one.

Here is an example diagram made from a Data Preview 2 visit image from the command:

.. code-block:: bash

   lsst-images-admin diagram --from-file dp2-visit.fits --hide-field data --collapse ButlerInfo --collapse Transform --hide-type TableModel --format mermaid --collapse Amplifier --collapse DetectorAttributes

.. mermaid::

   classDiagram
  class VisitImage["VisitImage"] {
  }
  class Image["Image"] {
  }
  class Mask["Mask"] {
  }
  class SkyProjection["SkyProjection"] {
  }
  class Transform["Transform"] {
  }
  class PiffWrapper["PiffWrapper"] {
  }
  class PiffObjectModel["PiffObjectModel"] {
  }
  class GalSimPixelScaleModel["GalSimPixelScaleModel"] {
  }
  class PiffTableModel["PiffTableModel"] {
  }
  class MinimalStar["MinimalStar"] {
  }
  class ObservationInfo["ObservationInfo"] {
  }
  class ObservationSummaryStats["ObservationSummaryStats"] {
  }
  class Detector["Detector"] {
  }
  class DetectorAttributes["DetectorAttributes"] {
  }
  class Amplifier["Amplifier"] {
  }
  class CameraFrameSet["CameraFrameSet"] {
  }
  class ApertureCorrectionMapSerializationModel["ApertureCorrectionMapSerializationModel"] {
  }
  class ChebyshevField["ChebyshevField"] {
  }
  class RegionSerializationModel["RegionSerializationModel"] {
  }
  class BackgroundMap["BackgroundMap"] {
  }
  VisitImage --> Image : image
  VisitImage --> Mask : mask
  VisitImage --> Image : variance
  VisitImage --> SkyProjection : sky_projection
  VisitImage --> PiffWrapper : psf
  VisitImage --> ObservationInfo : obs_info
  VisitImage --> ObservationSummaryStats : summary_stats
  VisitImage --> Detector : detector
  VisitImage --> ApertureCorrectionMapSerializationModel : aperture_corrections
  VisitImage --> RegionSerializationModel : bounds
  VisitImage --> BackgroundMap : backgrounds
  SkyProjection --> Transform : pixel_to_sky
  SkyProjection --> Transform : fits_approximation
  PiffWrapper --> PiffObjectModel : piff
  PiffWrapper --> MinimalStar : stars*
  PiffObjectModel --> GalSimPixelScaleModel : wcs+
  PiffObjectModel --> PiffTableModel : tables+
  PiffObjectModel --> PiffObjectModel : objects+
  Detector --> DetectorAttributes : attributes
  Detector --> Amplifier : amplifiers*
  Detector --> CameraFrameSet : frames
  ApertureCorrectionMapSerializationModel --> ChebyshevField : fields+

Programmatic use
================

The subcommand is a thin wrapper around `lsst.images.diagram`, which the documentation build and other tooling can call directly.
`~lsst.images.diagram.build_graph` walks a model class, `~lsst.images.diagram.graph_from_file` walks a file, and `~lsst.images.diagram.render` serializes the resulting graph to one of the three formats.
`~lsst.images.diagram.make_policy` builds the policy; the library defaults to the full, literal view (serialization names, scalar fields included), so pass options such as ``make_policy(public_names=True, include_attributes=False)`` for the curated view the subcommand shows by default.

.. code-block:: python

   from lsst.images import VisitImageSerializationModel
   from lsst.images.serialization._asdf_utils import ArrayReferenceModel
   from lsst.images.diagram import build_graph, render

   graph = build_graph(VisitImageSerializationModel[ArrayReferenceModel])
   print(render(graph, "mermaid"))
