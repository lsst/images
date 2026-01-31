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

"""Abstract interfaces and helper classes for `OutputArchive` and
`InputArchive`, which abstract over different file formats.

These archive interfaces are designed with two specific implementations in
mind:

- FITS augmented with a JSON block in a special BINTABLE HDU (see the `fits`
  module for details), inspired by the now-defunct ASDF-in-FITS concept.

- ASDF (just hypothetical for now).

The base classes make some concessions to both FITS and ASDF in order to make
the representations in those formats conform to their respective expectations.

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
archive interfaces accept `update_header` and `strip_header` callback arguments
that are only called by FITS implementations.

An implementation that writes HDF5 while embedding JSON should also be possible
with these interfaces, but is not something we've designed around. A more
natural HDF5 implementation might be possible by translating the JSON tree into
a binary HDF5 hierarchy as well, but this would be considerably more effort at
best.
"""

from ._common import *
from ._output_archive import *
from ._input_archive import *
