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

__all__ = ("Transform", "TransformCompositionError")

from typing import TYPE_CHECKING, Any

import astropy.units as u
import numpy as np

from .._geom import XY, Domain
from ._frames import Frame, SkyFrame

if TYPE_CHECKING:
    from ._projection import Projection


class TransformCompositionError(RuntimeError):
    """Exception raised when two transforms cannot be composed."""


class Transform[I: Frame, O: Frame]:
    """A transform that maps two coordinate frames.

    Notes
    -----
    When applied to celestial coordinate systems, ``x=ra`` and ``y=dec``.
    """

    def __init__(
        self,
        in_frame: I,
        out_frame: O,
        ast_mapping: Any,
        forward_domain: Domain | None = None,
        inverse_domain: Domain | None = None,
    ):
        self._in_frame = in_frame
        self._out_frame = out_frame
        self._ast_mapping = ast_mapping
        self._forward_domain = forward_domain
        self._inverse_domain = inverse_domain

    @staticmethod
    def identity(frame: I) -> Transform[I, I]:
        import astshim

        return Transform(frame, frame, astshim.UnitMap(2))

    @property
    def in_frame(self) -> I:
        return self._in_frame

    @property
    def out_frame(self) -> O:
        return self._out_frame

    @property
    def forward_domain(self) -> Domain | None:
        return self._forward_domain

    @property
    def inverse_domain(self) -> Domain | None:
        return self._inverse_domain

    def apply_forward[T: np.ndarray | float](self, *, x: T, y: T) -> XY[T]:
        return _ast_apply(self._ast_mapping.applyForward, x=x, y=y)

    def apply_inverse[T: np.ndarray | float](self, *, x: T, y: T) -> XY[T]:
        return _ast_apply(self._ast_mapping.applyInverse, x=x, y=y)

    def apply_forward_q(self, *, x: u.Quantity, y: u.Quantity) -> XY[u.Quantity]:
        xy = self.apply_forward(x=x.to_value(self._in_frame.unit), y=y.to_value(self._in_frame.unit))
        return XY(xy.x * self._out_frame.unit, xy.y * self._out_frame.unit)

    def apply_inverse_q(self, *, x: u.Quantity, y: u.Quantity) -> XY[u.Quantity]:
        xy = self.apply_inverse(x=x.to_value(self._out_frame.unit), y=y.to_value(self._out_frame.unit))
        return XY(xy.x * self._in_frame.unit, xy.y * self._in_frame.unit)

    def inverted(self) -> Transform[O, I]:
        return Transform(
            self._out_frame,
            self._in_frame,
            self._ast_mapping.inverted(),
            forward_domain=self.inverse_domain,
            inverse_domain=self.forward_domain,
        )

    def then[F: Frame](self, next: Transform[O, F]) -> Transform[I, F]:
        # TODO: include the intermediate domains somehow.
        if self._out_frame != next._in_frame:
            raise TransformCompositionError(
                "Cannot compose transforms that do not share a common intermediate frame: "
                f"{self._out_frame} != {next._in_frame}."
            )
        return Transform(
            self._in_frame,
            next._out_frame,
            self._ast_mapping.then(next._ast_mapping),
            forward_domain=self.forward_domain,
            inverse_domain=next.inverse_domain,
        )

    def as_projection(self: Transform[I, SkyFrame]) -> Projection[I]:
        from ._projection import Projection

        return Projection(self)


def _ast_apply[T: np.ndarray | float](method: Any, *, x: T, y: T) -> XY[T]:
    # TODO: add domain argument and check inputs
    xy_in = np.hstack([x, y])
    xy_out = method(xy_in)
    return XY(xy_out[:, 0], xy_out[:, 1])
