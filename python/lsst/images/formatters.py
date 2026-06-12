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

__all__ = ("GenericFormatter",)

import copy
import hashlib
import json as _stdlib_json  # disambiguates from .json subpackage
import threading
import uuid
from collections.abc import Mapping
from typing import Any, ClassVar, NamedTuple

import astropy.io.fits

from lsst.daf.butler import DatasetProvenance, FormatterV2
from lsst.resources import ResourcePath, ResourcePathExpression
from lsst.utils.iteration import ensure_iterable

from . import fits as _fits
from . import serialization as ser
from .serialization import ButlerInfo, write


class _TreeCache(NamedTuple):
    """Single-slot cache pairing a dataset ID with its validated
    serialization tree.
    """

    id_: uuid.UUID | None = None
    tree: ser.ArchiveTree | None = None


_DETACHED_ARCHIVE = ser.DetachedArchive()


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
    can_read_from_local_file: ClassVar[bool] = True

    butler_provenance: DatasetProvenance | None = None

    # Most recently read serialization tree, kept so that repeated component
    # reads of the same dataset do not reopen the file.
    _tree_cache_lock: ClassVar[threading.Lock] = threading.Lock()
    _tree_cache: ClassVar[_TreeCache] = _TreeCache()

    # --- Write parameter handling -------------------------------------------

    def get_write_extension(self) -> str:
        default_fmt = self.default_extension.lstrip(".")
        fmt = self.write_parameters.get("format", default_fmt)
        ext = "." + fmt
        if ext not in self.supported_extensions:
            raise RuntimeError(
                f"Requested format {fmt!r} is not supported; expected one of {{fits, json, sdf}}."
            )
        return ext

    def _validate_write_parameters(self) -> None:
        ext = self.get_write_extension()
        if ext != ".fits" and "recipe" in self.write_parameters:
            raise RuntimeError("The 'recipe' write parameter is only valid for FITS output.")

    @classmethod
    def validate_write_recipes(cls, recipes: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        if not recipes:
            return recipes
        for name, recipe in recipes.items():
            try:
                _fits.FitsCompressionOptions.model_validate(recipe)
            except Exception as err:
                err.add_note(name)
                raise
        return recipes

    # --- Write path ---------------------------------------------------------

    def write_local_file(self, in_memory_dataset: Any, uri: ResourcePath) -> None:
        self._validate_write_parameters()
        ext = self.get_write_extension()
        butler_info = ButlerInfo(
            dataset=self.dataset_ref.to_simple(),
            provenance=self.butler_provenance if self.butler_provenance is not None else DatasetProvenance(),
        )
        kwargs: dict[str, Any] = {"butler_info": butler_info}
        if ext == ".fits":
            kwargs["update_header"] = self._update_header
            kwargs["compression_options"] = self._get_compression_options()
            kwargs["compression_seed"] = self._get_compression_seed()
        # The generic write() dispatches to the FITS / JSON / NDF backend by
        # the file extension, which get_write_extension has already set on uri.
        write(in_memory_dataset, uri.ospath, **kwargs)

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
        # Set the seed based on data ID (all logic here duplicated from
        # obs_base). We can't just use 'hash', since like 'set' that's not
        # deterministic. And we can't rely on a DimensionPacker because those
        # are only defined for certain combinations of dimensions. Doing an MD5
        # of the JSON feels like overkill but I don't really see anything much
        # simpler.
        hash_bytes = hashlib.md5(
            _stdlib_json.dumps(list(self.data_id.required_values)).encode(),
            usedforsecurity=False,
        ).digest()
        # And it *really* feels like overkill when we squash that into the [1,
        # 10000] range allowed by FITS.
        return 1 + int.from_bytes(hash_bytes) % 9999

    def _get_compression_options(self) -> dict[str, _fits.FitsCompressionOptions]:
        recipe = self.write_parameters.get("recipe", "default")
        try:
            config = self.write_recipes[recipe]
        except KeyError:
            if recipe == "default":
                # If there's no default recipe just use the software defaults.
                return {}
            raise RuntimeError(f"Invalid recipe for GenericFormatter: {recipe!r}.") from None
        return {k: _fits.FitsCompressionOptions.model_validate(v) for k, v in config.items()}

    def _update_header(self, header: astropy.io.fits.Header) -> None:
        # Logic here largely lifted from lsst.obs.base.utils, which we
        # can't use directly for dependency and maybe mapping-type
        # (PropertyList vs. astropy) reasons. We assume we can always add
        # long cards (astropy will CONTINUE them) but not comments
        # (astropy will truncate and warn on long cards).
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

    # --- Component tree cache -----------------------------------------------

    def _component_from_cache(self, component: str) -> tuple[bool, Any]:
        """Try to deserialize a component from the most recently read tree.

        Parameters
        ----------
        component
            Name of the component to read.

        Returns
        -------
        hit : `bool`
            Whether the component could be served from the cache.
        value
            The deserialized component; `None` on a cache miss.

        Raises
        ------
        lsst.images.serialization.InvalidComponentError
            Raised if the cached tree does not recognize ``component``.
        """
        with self._tree_cache_lock:
            cache = type(self)._tree_cache
        if cache.tree is None or cache.id_ != self.dataset_ref.id:
            return False, None
        try:
            value = cache.tree.deserialize_component(component, _DETACHED_ARCHIVE)
        except ser.ArchiveAccessRequiredError:
            # The component points at data stored outside the JSON tree, so
            # the file has to be opened and read.
            return False, None
        return True, self._detach_component(cache.tree, component, value)

    def _cache_tree(self, tree: ser.ArchiveTree) -> None:
        """Remember a validated tree so that later component reads of the
        same dataset can be served without reopening the file.
        """
        with self._tree_cache_lock:
            type(self)._tree_cache = _TreeCache(id_=self.dataset_ref.id, tree=tree)

    @staticmethod
    def _detach_component(tree: ser.ArchiveTree, component: str, value: Any) -> Any:
        """Copy a component value if it is owned by the given tree.

        `~lsst.images.serialization.ArchiveTree.deserialize_component`
        returns plain (non-tree) models by reference from the tree, so
        without a copy repeated reads of a mutable component would alias
        each other through the cache.
        """
        if value is not None and value is getattr(tree, component, None):
            return copy.deepcopy(value)
        return value

    # --- Read path ---------------------------------------------------------

    def read_from_uri(
        self,
        uri: ResourcePath,
        component: str | None = None,
        expected_size: int = -1,
    ) -> Any:
        # For full read, always use local file read since the entire file has
        # to be read anyhow and we should allow it to be cached. Cutouts
        # can use remote reads since that is generally less to be downloaded
        # than the full file.
        if not component and not self.file_descriptor.parameters:
            return NotImplemented

        # Now call the generalized reader.
        return self._read_from_resource_path(uri, component)

    def _read_from_resource_path(self, uri: ResourcePathExpression, component: str | None = None) -> Any:
        # General purpose reader that can be called with both local and remote
        # file. The URI and local file readers are distinct to allow decisions
        # to be made regarding caching.
        kwargs = dict(self.file_descriptor.parameters or {})
        pytype: type[Any] = self.dataset_ref.datasetType.storageClass.pytype
        all_components = self.dataset_ref.datasetType.storageClass.allComponents()

        # The components parameter is special in that it's not modifying
        # a particular component in any way. It requires the "component"
        # component to be specified so that the butler knows to expect a dict
        # to be returned. Use a set to ensure we do not get asked for the same
        # component multiple times.
        components = set()
        want_component_dict = False
        if component == "components":
            # Read the list of components from the parameter of the same name.
            want_component_dict = True

            # Try to distinguish no parameter specified (which we take to
            # imply all components) vs explicit empty list (which we take
            # to be an error).
            requested_components = kwargs.pop("components", None)

            if requested_components is None:
                # No explicit request so fill with all components.

                # If there are no kwargs then this is effectively a full
                # read of the entire file. We should trigger a cache write
                # if this is a remote file.
                if not kwargs and isinstance(uri, ResourcePath) and not uri.isLocal:
                    return NotImplemented

                # Make sure that we drop the "components" component.
                # Do not return masked_image since that will likely already
                # be included in separate masked image components.
                # Hard-coding this knowledge is not great so we have to be
                # aware of similar issues in the future.
                components = {c for c in all_components if c not in {"components", "masked_image"}}
            elif not requested_components:
                raise RuntimeError("Requesting multiple components but received empty request.")
            else:
                # Force to a list in case someone has tried doing
                # "components=x".
                components = set(ensure_iterable(requested_components))

                if "components" in components:
                    raise RuntimeError(
                        "The 'components' component should not be specified in the 'components' parameter. "
                        "To request all components, do not specify any value for the 'components' parameter."
                    )

        elif component:
            # Simplify logic below so we only deal with a list.
            components = {component}

        if "components" in kwargs:
            raise RuntimeError(
                "Multiple component requests can only be specified if you use the 'components' component."
            )

        # If this is not a component read but a full read with parameters,
        # do that now before we focus on the component logic.
        if component is None:
            with ser.open(uri, cls=pytype, partial=bool(kwargs)) as reader:
                tree = reader.get_tree()
                self._cache_tree(tree)
                # Cutout read.
                return reader.read(**kwargs)

        # Associate remaining parameters with the component that understands
        # it. This should allow you to ask for image cutout along with a PSF.
        used_parameter_keys = set()
        component_kwargs: dict[str, Any] = {}
        for comp in components:
            if comp not in all_components:
                raise RuntimeError(
                    f"Requested data for component {comp} but that component is not understood "
                    f"by storage class {self.dataset_ref.datasetType.storageClass.name}."
                )
            if comp not in component_kwargs:
                # Make sure we have an entry even if no params.
                component_kwargs[comp] = {}
            for param, value in kwargs.items():
                if param in all_components[comp].parameters:
                    component_kwargs[comp][param] = value
                    used_parameter_keys.add(param)

        if kwargs and (unused := (set(kwargs) - used_parameter_keys)):
            raise RuntimeError(
                f"Specified parameters ({unused}) that are not known to any of the "
                f"requested components ({components})."
            )

        # See if we can get some of the components from the cache.
        components_to_return = {}
        for comp, params in component_kwargs.items():
            if not params:
                hit, value = self._component_from_cache(comp)
                if hit:
                    components_to_return[comp] = value

        if len(components_to_return) != len(components):
            # Some components were not available in the cache. At this point
            # we have to open the file to read the remaining components.
            with ser.open(uri, cls=pytype, partial=True) as reader:
                tree = reader.get_tree()
                self._cache_tree(tree)

                for comp, params in component_kwargs.items():
                    if comp not in components_to_return:
                        components_to_return[comp] = self._detach_component(
                            tree, comp, reader.get_component(comp, **params)
                        )

        if want_component_dict:
            return components_to_return
        return components_to_return.popitem()[1]

    def read_from_local_file(self, path: str, component: str | None = None, expected_size: int = -1) -> Any:
        # Docstring inherited.
        # Call the generalized reader that does not care whether this is
        # a local or remote file. The distinction exists here to ensure we
        # can trigger a cache load.
        return self._read_from_resource_path(path, component)
