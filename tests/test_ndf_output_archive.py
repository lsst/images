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

from __future__ import annotations

import tempfile
import unittest

import h5py
import pydantic

from lsst.images.ndf._output_archive import NdfOutputArchive


class TinyTree(pydantic.BaseModel):
    name: str


class NdfOutputArchiveBasicsTestCase(unittest.TestCase):
    def test_serialize_direct_calls_serializer_with_nested_archive(self):
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                arch = NdfOutputArchive(f)
                tree = arch.serialize_direct(
                    "top", lambda nested: TinyTree(name="hello")
                )
                self.assertEqual(tree.name, "hello")

    def test_constructor_marks_root_as_ndf(self):
        """The constructor should set CLASS=NDF on the root group so that
        Starlink tools recognise the file as an NDF.
        """
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                NdfOutputArchive(f)
            with h5py.File(tmp.name, "r") as f:
                self.assertEqual(f["/"].attrs["CLASS"], "NDF")
