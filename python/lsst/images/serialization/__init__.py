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

"""Base classes and utilities for the serialization framework.

This includes the `read_archive` and `write_archive` functions, which provide
the highest-level interfaces for reading arbitrary objects from and writing
them to storage, respectively (for a concrete `.GeneralizedImage` subclass,
prefer its `~.GeneralizedImage.read` and `~.GeneralizedImage.write` methods).

`InputArchive` and `OutputArchive` are the abstract interfaces for
implementing serialization for a new file format.

The base classes make some concessions to both FITS and ASDF in order to make
the (potential) representations in those formats conform to their respective
expectations.

For ASDF, this is simple: we use ASDF schemas whenever possible to represent
primitive types, from units and times to multidimensional arrays. While the
archive interfaces use Pydantic, which maps to JSON, not YAML, the expectation
is that by encoding YAML tag information in the JSON Schema (which Pydantic
allows us to customize), it should be straightforward for an ASDF archive
implementation to have Pydantic dump to a Python `dict` (etc) tree, and then
convert that to tagged YAML by walking the tree along with its schema.

For FITS, the challenge is primarily to populate standard FITS header cards
when writing, despite the fact that FITS headers are generally too limiting to
be our preferred way of round-tripping any information.  To do this, the
archive interfaces accept ``update_header`` and ``strip_header`` callback
arguments that are only called by FITS implementations.
"""

from ._asdf_utils import *
from ._backends import *
from ._common import *
from ._dtypes import *
from ._frozen_schemas import *
from ._input_archive import *
from ._io import *
from ._output_archive import *
from ._reader import *
from ._tables import *
