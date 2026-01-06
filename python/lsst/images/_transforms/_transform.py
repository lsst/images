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

__all__ = (
    "Transform",
    "TransformCompositionError",
    "TransformModel",
)

import textwrap
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any

import astropy.units as u
import numpy as np
import pydantic

from .._geom import XY, Domain, SerializableDomain
from ._frames import Frame, SkyFrame

if TYPE_CHECKING:
    from ._frame_set import FrameSet
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
        components: Iterable[Transform[Any, Any]] = (),
    ):
        self._in_frame = in_frame
        self._out_frame = out_frame
        self._ast_mapping = ast_mapping
        self._forward_domain = forward_domain
        self._inverse_domain = inverse_domain
        self._components = list(components)

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

    def decompose(self) -> list[Transform[Any, Any]]:
        if not self._components:
            if self.in_frame == self._out_frame:
                return []
            else:
                return [self]
        else:
            return list(self._components)

    def inverted(self) -> Transform[O, I]:
        return Transform(
            self._out_frame,
            self._in_frame,
            self._ast_mapping.inverted(),
            forward_domain=self.inverse_domain,
            inverse_domain=self.forward_domain,
            components=[t.inverted() for t in reversed(self._components)],
        )

    def then[F: Frame](self, next: Transform[O, F], remember_components: bool = True) -> Transform[I, F]:
        if self._out_frame != next._in_frame:
            raise TransformCompositionError(
                "Cannot compose transforms that do not share a common intermediate frame: "
                f"{self._out_frame} != {next._in_frame}."
            )
        components = self.decompose() + next.decompose() if remember_components else ()
        return Transform(
            self._in_frame,
            next._out_frame,
            self._ast_mapping.then(next._ast_mapping),
            forward_domain=self.forward_domain,
            inverse_domain=next.inverse_domain,
            components=components,
        )

    def as_projection(self: Transform[I, SkyFrame]) -> Projection[I]:
        from ._projection import Projection

        return Projection(self)

    def serialize[P: pydantic.BaseModel](
        self,
        frame_sets: Sequence[tuple[FrameSet, P]] = (),
    ) -> TransformModel[P]:
        model = TransformModel[P]()
        for link in self.decompose():
            model.frames.append(link.in_frame)
            model.domains.append(link.forward_domain.serialize() if link.forward_domain is not None else None)
            for frame_set, pointer in frame_sets:
                if link.in_frame in frame_set and link.out_frame in frame_set:
                    model.mappings.append(pointer)
                    break
            else:
                model.mappings.append(MappingModel(root=link._ast_mapping.show()))
        model.frames.append(self.out_frame)
        model.domains.append(self.inverse_domain.serialize() if self.inverse_domain is not None else None)
        return model

    @staticmethod
    def deserialize[P: pydantic.BaseModel](
        model: TransformModel[P], get_frame_set: Callable[[P], FrameSet]
    ) -> Transform[Any, Any]:
        import astshim

        transform = Transform.identity(model.frames[0])
        for n, mapping in enumerate(model.mappings):
            match mapping:
                case MappingModel(root=serialized_mapping):
                    ast_mapping = astshim.Mapping.fromString(serialized_mapping)
                    forward_domain = model.domains[n]
                    inverse_domain = model.domains[n + 1]
                    transform = transform.then(
                        Transform(
                            model.frames[n],
                            model.frames[n + 1],
                            ast_mapping,
                            Domain.deserialize(forward_domain) if forward_domain is not None else None,
                            Domain.deserialize(inverse_domain) if inverse_domain is not None else None,
                        )
                    )
                case pointer:
                    frame_set = get_frame_set(pointer)
                    transform = transform.then(frame_set[model.frames[n], model.frames[n + 1]])
        return transform


def _ast_apply[T: np.ndarray | float](method: Any, *, x: T, y: T) -> XY[T]:
    # TODO: add domain argument and check inputs
    xy_in = np.hstack([x, y])
    xy_out = method(xy_in)
    return XY(xy_out[:, 0], xy_out[:, 1])


class MappingModel(pydantic.RootModel[str]):
    root: str


class TransformModel[P: pydantic.BaseModel](pydantic.BaseModel):
    """Serialization model for coordinate transforms."""

    frames: list[Frame] = pydantic.Field(
        default_factory=list,
        description=textwrap.dedent(
            """
            List of frames that this transform passes through.

            All transforms include at least two frames (the endpoints).  Others
            intermediate frames may be included to facilitate data-sharing
            between transforms.
            """
        ),
    )

    domains: list[SerializableDomain | None] = pydantic.Field(
        default_factory=list,
        description=textwrap.dedent(
            """
            List of the domains of the ``frames`` for this transform.

            This always has the same number of elements as ``frames``.
            """
        ),
    )

    mappings: list[P | MappingModel] = pydantic.Field(
        default_factory=list,
        description=textwrap.dedent(
            """
            The actual mappings between frames, or archive pointers to
            serialized FrameSet objects from which they can be obtained.

            This always has one fewer element than ``frames``.
            """
        ),
    )

    @pydantic.model_validator(mode="after")
    def _validate_lens(self) -> TransformModel[P]:
        if len(self.frames) != len(self.domains):
            raise ValueError("Inconsistent lengths for 'frames' and 'domains'.")
        if len(self.frames) != len(self.mappings) + 1:
            raise ValueError("Inconsistent lengths for 'frames' and 'mappings'.")
        return self
