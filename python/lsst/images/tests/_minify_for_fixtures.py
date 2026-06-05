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

"""Minify a real on-disk archive into a small JSON test fixture.

Reads a FITS or NDF file via the appropriate input archive, takes a
small subset of the in-memory object, and writes JSON via
``JsonOutputArchive``. Used to populate ``tests/data/schema_v1/legacy/``
with derived-from-real test data that exercises the full read path
including the absence-of-stamp legacy default.

Per top-level type the subset rule is:

  VisitImage   Crop the image/mask/variance planes to a small (~16x16)
               corner, keeping the real single-instance structures (PSF
               such as Piff, detector frames) that synthetic fixtures
               cannot reproduce -- the whole point of deriving a fixture
               from real data.  Homogeneous repeated collections (detector
               amplifiers, aperture-correction entries) are trimmed to a
               representative few, since one entry exercises the schema as
               well as sixteen.  The projection's pixel->sky mapping is
               replaced by its linear (affine) approximation over the kept
               box: a real TAN-SIP WCS serializes as a ~100 KB AST
               polynomial dump, but over a 16x16 box it is linear to far
               below a pixel, so the affine form is schema-identical and
               orders of magnitude smaller.  A Piff PSF's field
               interpolation is truncated to a low order (the order-4
               solution table is ~225 KB; order 0, the field-averaged PSF,
               is schema-identical and ~13x smaller).

  CellCoadd    Crop to a small block of cells (preferring a block that
               includes a missing cell so the sparse-grid path is
               exercised) and then *morph* that block onto a tiny cell
               grid: each cell's planes are decimated from the native
               cell size down to a few pixels and re-stitched, and the
               PSF kernels are cropped to a small odd window.  The grid
               topology (number of cells, the missing-cell set, band,
               mask schema and provenance shape) is preserved; the pixel
               values and WCS are *not* physically meaningful.  This is
               the "morph cells in place" fallback: it sidesteps the
               outer-ring problem (inputs/PSFs that overlap kept cells)
               by rebuilding a self-consistent miniature coadd rather
               than trying to carve an accurate subset out of the real
               one.  An accurate per-cell subset would inline several
               150x150 planes per cell and produce multi-megabyte JSON,
               which defeats the purpose of a fixture.

Run interactively (CellCoadd works with just this package installed;
VisitImage needs a full Rubin environment so the real PSF can be read)::

    python -c "
    from lsst.images.tests._minify_for_fixtures import minify
    minify('cell_example.fits', 'tests/data/schema_v1/legacy/cell_coadd.json')
    minify('dp1.fits', 'tests/data/schema_v1/legacy/visit_image_dp1.json')
    minify('dp2.fits', 'tests/data/schema_v1/legacy/visit_image_dp2.json')
    "

The helper is invoked manually by developers when they have a real
on-disk file to derive from; it is not exercised by CI.
"""

from __future__ import annotations

__all__ = ("minify",)

import json
import os
from collections.abc import Callable
from typing import Any

import numpy as np

from .. import DifferenceImage, VisitImage
from .. import json as images_json
from .._cell_grid import CellGrid, CellGridBounds, CellIJ, PatchDefinition
from .._geom import YX, Box
from .._image import Image
from .._mask import Mask
from .._transforms import Projection, TractFrame, Transform
from .._transforms._ast import PolyMap
from ..cells import CellCoadd
from ..cells._provenance import CoaddProvenance
from ..cells._psf import CellPointSpreadFunction
from ..psfs import PiffWrapper
from ..serialization import backend_for_path, read
from ._creation import make_random_projection

# Default morph parameters for CellCoadd.  ``CELL_SIZE`` should divide the
# native cell size evenly; ``KERNEL_SIZE`` must be odd.  ``MAX_INPUTS`` caps
# the provenance ``inputs`` table (a real coadd has hundreds of visits); the
# full provenance schema is already exercised by the ``coadd_provenance``
# fixture, so here we keep just enough rows to be representative.
_CELL_SIZE = 6
_KERNEL_SIZE = 5
_MAX_INPUTS = 6

# Default trim parameters for VisitImage.  Amplifiers and aperture-correction
# entries are homogeneous collections, so a couple of each cover the schema
# just as well as the full set (a real detector has 16 amplifiers and dozens
# of aperture corrections).
_MAX_AMPLIFIERS = 2
_MAX_APERTURE_CORRECTIONS = 2

# Field-interpolation order to truncate a Piff PSF to (the solution table of a
# real order-4 PixelGrid PSF dominates the fixture at ~225 KB).  Order 0 is the
# field-averaged PSF; set to `None` to leave the PSF untouched.
_PSF_INTERP_ORDER = 0

# Maximum permitted deviation (radians) when approximating a projection's
# pixel->sky mapping with an affine one.  Over a fixture's tiny box the real
# mapping is linear well below this, so the fit always succeeds.
_PROJECTION_LINEAR_APPROX_TOL = 1e-8


def minify(in_path: str, out_path: str, *, schema_name: str | None = None) -> None:
    """Read a real archive at ``in_path``, take a small subset, and write JSON.

    Parameters
    ----------
    in_path
        Path to a FITS (``.fits`` / ``.fits.gz``) or NDF (``.sdf`` / ``.ndf``)
        file to read.
    out_path
        Path to the JSON fixture to write. The parent directory is
        created if it does not exist.
    schema_name
        Top-level schema name (e.g. ``"visit_image"`` or ``"cell_coadd"``).
        If `None`, it is auto-detected from the file.

    Raises
    ------
    ValueError
        If the file extension is not recognised.
    NotImplementedError
        If the top-level type is not one this helper knows how to subset.
    """
    backend = backend_for_path(in_path)
    if schema_name is None:
        schema_name = backend.input_archive.get_basic_info(in_path).schema_name

    cls, subsetter = _dispatch(schema_name)

    obj: Any = read(in_path, cls)
    subset = subsetter(obj)

    tree = images_json.write(subset)
    dumped = tree.model_dump(mode="json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as stream:
        stream.write(json.dumps(dumped, indent=2, sort_keys=False) + "\n")


def _dispatch(schema_name: str) -> tuple[type, Callable[[Any], Any]]:
    """Return the ``(class, subsetter)`` pair for a top-level schema name."""
    registry: dict[str, tuple[type, Callable[[Any], Any]]] = {
        "visit_image": (VisitImage, _subset_visit_image),
        "difference_image": (DifferenceImage, _subset_visit_image),
        "cell_coadd": (CellCoadd, _subset_cell_coadd),
    }
    try:
        return registry[schema_name]
    except KeyError:
        raise NotImplementedError(
            f"No minify rule for schema {schema_name!r}; supported: {sorted(registry)}."
        ) from None


# -- VisitImage ------------------------------------------------------------


def _subset_visit_image(
    visit_like_image: VisitImage | DifferenceImage,
    *,
    size: int = 16,
    max_amplifiers: int = _MAX_AMPLIFIERS,
    max_aperture_corrections: int = _MAX_APERTURE_CORRECTIONS,
    linearize_projection: bool = True,
    projection_tol: float = _PROJECTION_LINEAR_APPROX_TOL,
    psf_interp_order: int | None = _PSF_INTERP_ORDER,
) -> VisitImage | DifferenceImage:
    """Crop a VisitImage's pixel planes to a small corner and trim its
    homogeneous collections.

    The detector frames are a single structure carried through unchanged by
    ``__getitem__``.  The detector's amplifiers and the aperture-correction map
    are repeated, schema-identical entries, so they are trimmed to a
    representative few.  The projection's pixel->sky mapping is replaced by its
    affine approximation over the kept box (see ``_linear_approx_projection``)
    unless ``linearize_projection`` is false.  A Piff PSF's field interpolation
    is truncated to ``psf_interp_order`` (see ``_simplify_piff_psf``) unless
    that is `None`.
    """
    bbox = visit_like_image.bbox
    y0 = bbox.y.start
    x0 = bbox.x.start
    y1 = min(y0 + size, bbox.y.stop)
    x1 = min(x0 + size, bbox.x.stop)
    subset = visit_like_image[Box.factory[y0:y1, x0:x1]]

    # ``subset`` is a fresh throwaway object whose detector amplifier list,
    # aperture-correction map and PSF are live, mutable components.  Trim them
    # in place through the public accessors rather than reaching for private
    # attributes.
    del subset.detector.amplifiers[max_amplifiers:]
    aperture_corrections = subset.aperture_corrections
    for key in list(aperture_corrections)[max_aperture_corrections:]:
        del aperture_corrections[key]
    if psf_interp_order is not None and isinstance(subset.psf, PiffWrapper):
        _simplify_piff_psf(subset.psf, order=psf_interp_order)

    if not linearize_projection or subset.projection is None:
        return subset

    # The pixel planes carry the projection immutably (there is no public
    # setter for it), so install the affine approximation by rebuilding the
    # VisitImage from its public components with re-viewed planes.  Only the
    # image plane's projection is actually serialized, but keeping all three
    # consistent avoids surprises.
    linear = _linear_approx_projection(subset.projection, subset.image.bbox, tol=projection_tol)
    return type(visit_like_image)(
        subset.image.view(projection=linear),
        mask=subset.mask.view(projection=linear),
        variance=subset.variance.view(projection=linear),
        projection=linear,
        psf=subset.psf,
        obs_info=subset.obs_info,
        bounds=subset.bounds,
        summary_stats=subset.summary_stats,
        detector=subset.detector,
        photometric_scaling=subset.photometric_scaling,
        aperture_corrections=subset.aperture_corrections,
        backgrounds=subset.backgrounds,
        band=subset.band,
        metadata=subset.metadata,
    )


def _linear_approx_projection(projection: Projection, bbox: Box, *, tol: float) -> Projection:
    """Return a copy of ``projection`` whose pixel->sky mapping is replaced by
    its best linear (affine) approximation over ``bbox``.

    Real WCS mappings (e.g. TAN-SIP) serialize as large AST polynomial dumps.
    Over the small box of a fixture they are linear to far below a pixel, so
    an affine approximation is schema-identical but orders of magnitude
    smaller.  The result carries no FITS approximation (the affine is itself
    trivially FITS-representable).

    This is written as a self-contained ``projection -> projection`` transform
    so it can be promoted to a public ``Projection.linear_approx(bbox, tol)``
    method later with essentially no change.  It assumes a 2-D pixel->sky
    mapping.

    Parameters
    ----------
    projection
        The projection to approximate.
    bbox
        Box (in pixel coordinates) over which the approximation must hold.
    tol
        Maximum permitted deviation from linearity, as a Cartesian
        displacement in the output (sky, radians) coordinates.  AST raises
        ``RuntimeError`` if no fit within ``tol`` exists.
    """
    transform = projection.pixel_to_sky_transform
    mapping = transform._ast_mapping
    lbnd = [bbox.x.start, bbox.y.start]
    ubnd = [bbox.x.stop, bbox.y.stop]
    # linearApprox yields [offsets; Jacobian] as a (1 + n_out, n_in) array on
    # both AST backends (astshim returns the flat buffer in the same order, so
    # the reshape recovers the same layout the starlink-pyast bridge returns).
    fit = np.asarray(mapping.linearApprox(lbnd, ubnd, tol), dtype=float).reshape(3, 2)
    offset = fit[0]  # (lon0, lat0), radians
    jacobian = fit[1:]  # jacobian[i, j] = d(out_i) / d(in_j), in = (x, y)
    jacobian_inv = np.linalg.inv(jacobian)
    forward = _affine_polymap_coeffs(jacobian, offset)
    inverse = _affine_polymap_coeffs(jacobian_inv, -jacobian_inv @ offset)
    affine = Transform(
        transform.in_frame,
        transform.out_frame,
        PolyMap(forward, inverse),
        in_bounds=projection.pixel_bounds,
    )
    return affine.as_projection()


def _affine_polymap_coeffs(matrix: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Build AST ``PolyMap`` coefficients for ``out = matrix @ in + offset``.

    Each row is ``[coefficient, output_axis (1-based), power_of_in_1, ...]``;
    one constant row plus one row per input per output axis.  Returned as a
    float array, which is the form both AST backends require.
    """
    n = len(offset)
    coeffs: list[list[float]] = []
    for i in range(n):
        coeffs.append([float(offset[i]), i + 1, *([0] * n)])
        for j in range(n):
            powers = [1 if k == j else 0 for k in range(n)]
            coeffs.append([float(matrix[i][j]), i + 1, *powers])
    return np.array(coeffs, dtype=float)


def _simplify_piff_psf(psf: PiffWrapper, *, order: int) -> None:
    """Truncate a Piff PSF's field interpolation to ``order``, in place.

    A real Piff PSF interpolates a per-pixel model across the focal plane with
    a high-order 2-D polynomial; that solution table dominates the serialized
    size (a 25x25 PixelGrid x order-4 polynomial is ~225 KB).  Truncating to
    ``order`` keeps only the lowest-order field terms -- order 0 is the
    field-averaged PSF -- which is schema-identical but far smaller, and needs
    no stars or refit (the fitted ``stars`` are already dropped on serialize).

    Only ``BasisPolynomial``-interpolated PSFs are handled; anything else (a
    higher-order model already at/under ``order``, a non-polynomial interp) is
    left untouched.

    ``piff`` is imported lazily because it is an optional dependency; this is
    only ever reached when the PSF being simplified is itself a Piff PSF.
    """
    interp = getattr(psf.piff_psf, "interp", None)
    if interp is None or type(interp).__name__ != "BasisPolynomial" or interp.q is None:
        return
    if order >= max(interp._orders):
        return

    from piff import BasisPolynomial

    # ``q`` has one column per active basis term; the terms are the True cells
    # of ``_mask`` in row-major (i, j) order (see BasisPolynomial.basis).  Make
    # the same ordering for a lower-order interp and copy the shared columns.
    def _terms(orders: tuple[int, ...], mask: np.ndarray) -> list[tuple[int, ...]]:
        grids = np.meshgrid(*[np.arange(o + 1) for o in orders], indexing="ij")
        return list(zip(*(grid[mask].tolist() for grid in grids)))

    old_terms = _terms(interp._orders, interp._mask)
    truncated = BasisPolynomial(order, keys=list(interp._keys))
    new_terms = _terms(truncated._orders, truncated._mask)
    column_of = {term: index for index, term in enumerate(old_terms)}
    truncated.q = np.ascontiguousarray(interp.q[:, [column_of[term] for term in new_terms]])
    psf.piff_psf.interp = truncated


# -- CellCoadd -------------------------------------------------------------


def _subset_cell_coadd(
    cell_coadd: CellCoadd,
    *,
    cell_size: int = _CELL_SIZE,
    kernel_size: int = _KERNEL_SIZE,
    max_inputs: int = _MAX_INPUTS,
) -> CellCoadd:
    """Crop a CellCoadd to a small block of cells and morph it onto a tiny
    grid (see the module docstring for the rationale).
    """
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd, got {kernel_size}.")

    # 1. Pick a block of (up to) 2x2 cells, preferring one that contains a
    #    missing cell so the sparse-grid path is exercised.  Falls back to the
    #    first available block when the coadd is fully dense.
    block = cell_coadd[_choose_block_bbox(cell_coadd)]

    grid = block.grid
    cs = grid.cell_shape
    start = block.bounds.grid_start
    stop = block.bounds.grid_stop
    n_i = stop.i - start.i
    n_j = stop.j - start.j

    # 2. Build a tiny full-patch grid with the same cell *count* as the
    #    original patch but ``cell_size`` pixels per cell, anchored at (0, 0).
    full_shape = grid.grid_shape
    new_grid = CellGrid(
        bbox=Box.factory[0 : full_shape.i * cell_size, 0 : full_shape.j * cell_size],
        cell_shape=YX(y=cell_size, x=cell_size),
    )
    new_block_bbox = _scale_box_to_grid(block.bbox, grid, cell_size)
    new_bounds = CellGridBounds(grid=new_grid, bbox=new_block_bbox, missing=block.bounds.missing)

    # 3. Decimate each plane.  Because the block's planes tile the kept cells
    #    contiguously, a uniform stride that maps one native cell onto
    #    ``cell_size`` samples is equivalent to per-cell decimation.
    step_y = max(1, cs.y // cell_size)
    step_x = max(1, cs.x // cell_size)
    ny = n_i * cell_size
    nx = n_j * cell_size

    def shrink2d(array: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(array[::step_y, ::step_x][:ny, :nx])

    def shrink3d(array: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(array[::step_y, ::step_x, :][:ny, :nx, :])

    # 4. Synthetic-but-valid projection over the tiny tract frame.
    rng = np.random.default_rng(0)
    tract_frame = TractFrame(skymap=cell_coadd.skymap, tract=cell_coadd.tract, bbox=new_grid.bbox)
    projection = make_random_projection(rng, tract_frame, new_block_bbox)

    unit = cell_coadd.unit
    image = Image(shrink2d(block.image.array), bbox=new_block_bbox, unit=unit, projection=projection)
    mask = Mask(shrink3d(block.mask.array), schema=block.mask.schema, bbox=new_block_bbox)
    variance = Image(shrink2d(block.variance.array), bbox=new_block_bbox, unit=unit**2)
    mask_fractions = {
        name: Image(shrink2d(plane.array), bbox=new_block_bbox)
        for name, plane in block.mask_fractions.items()
    }
    noise_realizations = [
        Image(shrink2d(plane.array), bbox=new_block_bbox) for plane in block.noise_realizations
    ]

    # 5. Crop the PSF kernels to a small odd window about their centre,
    #    keeping the (n_i, n_j) per-cell structure and NaN-for-missing cells.
    psf_array = block.psf._array
    ky, kx = psf_array.shape[2:]
    half = kernel_size // 2
    cy, cx = ky // 2, kx // 2
    psf_array = np.ascontiguousarray(psf_array[:, :, cy - half : cy + half + 1, cx - half : cx + half + 1])
    psf = CellPointSpreadFunction(psf_array, bounds=new_bounds)

    # 6. Patch geometry scaled onto the tiny grid; provenance and backgrounds
    #    are reused as-is (provenance is cell-indexed and already subset).
    patch = PatchDefinition(
        id=block.patch.id,
        index=block.patch.index,
        inner_bbox=_scale_box_to_grid(block.patch.inner_bbox, grid, cell_size),
        cells=new_grid,
    )

    provenance = block._provenance
    if provenance is not None:
        provenance = _trim_provenance(provenance, max_inputs=max_inputs)

    return CellCoadd(
        image,
        mask=mask,
        variance=variance,
        mask_fractions=mask_fractions,
        noise_realizations=noise_realizations,
        projection=projection,
        band=block.band,
        psf=psf,
        patch=patch,
        provenance=provenance,
        backgrounds=block._backgrounds,
    )


def _trim_provenance(provenance: CoaddProvenance, *, max_inputs: int) -> CoaddProvenance:
    """Cap the provenance ``inputs`` table to ``max_inputs`` rows and drop any
    contributions that reference the removed inputs.

    The two-table structure, polygon arrays and string dictionary-compression
    paths are all preserved; only the number of contributing visits shrinks.
    """
    inputs = provenance.inputs
    if len(inputs) <= max_inputs:
        return provenance
    kept_inputs = inputs[:max_inputs]
    keys = {(str(row["instrument"]), int(row["visit"]), int(row["detector"])) for row in kept_inputs}
    contributions = provenance.contributions
    mask = np.array(
        [
            (str(instrument), int(visit), int(detector)) in keys
            for instrument, visit, detector in zip(
                contributions["instrument"], contributions["visit"], contributions["detector"]
            )
        ],
        dtype=bool,
    )
    return CoaddProvenance(inputs=kept_inputs, contributions=contributions[mask])


def _choose_block_bbox(cell_coadd: CellCoadd) -> Box:
    """Return the pixel bbox of a (up to) 2x2 block of cells to keep.

    Prefers a block containing a missing cell; otherwise the block anchored at
    the start of the populated region.  Never raises if there is no missing
    cell.
    """
    bounds = cell_coadd.bounds
    grid = bounds.grid
    start = bounds.grid_start
    stop = bounds.grid_stop
    span_i = min(2, stop.i - start.i)
    span_j = min(2, stop.j - start.j)

    target = next(iter(sorted(bounds.missing)), None)
    if target is not None:
        # Anchor the block so it includes the missing cell, clamped to the
        # populated index range.
        i0 = min(max(target.i, start.i), stop.i - span_i)
        j0 = min(max(target.j, start.j), stop.j - span_j)
    else:
        i0 = start.i
        j0 = start.j

    lo = grid.bbox_of(CellIJ(i=i0, j=j0))
    hi = grid.bbox_of(CellIJ(i=i0 + span_i - 1, j=j0 + span_j - 1))
    return Box.factory[lo.y.start : hi.y.stop, lo.x.start : hi.x.stop]


def _scale_box_to_grid(box: Box, grid: CellGrid, cell_size: int) -> Box:
    """Map a grid-aligned box onto a grid with ``cell_size`` pixels per cell,
    anchored at the origin.
    """
    cs = grid.cell_shape
    s = grid.bbox.start
    iy0 = (box.y.start - s.y) // cs.y
    iy1 = (box.y.stop - s.y) // cs.y
    ix0 = (box.x.start - s.x) // cs.x
    ix1 = (box.x.stop - s.x) // cs.x
    return Box.factory[iy0 * cell_size : iy1 * cell_size, ix0 * cell_size : ix1 * cell_size]
