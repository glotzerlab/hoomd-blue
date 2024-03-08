# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

r"""MPCD geometries.

A geometry defines solid boundaries that cannot be penetrated. These
geometries are used for various operations in the MPCD algorithm including:

* :class:`~hoomd.mpcd.stream.BounceBack` streaming for the solvent
* Bounce back integration for MD particles
* Virtual particle filling

Each geometry may put constraints on the size of the simulation and where
particles are allowed. These constraints will be documented by each object.

"""

from hoomd.data.parameterdicts import ParameterDict
from hoomd.mpcd import _mpcd
from hoomd.operation import _HOOMDBaseObject


class Geometry(_HOOMDBaseObject):
    r"""Geometry.

    Args:
        no_slip (bool): If True, surfaces have no-slip boundary condition.
            Otherwise, they have the slip boundary condition.

    Attributes:
        no_slip (bool): If True, plates have no-slip boundary condition.
            Otherwise, they have the slip boundary condition.

            `V` will have no effect if `no_slip` is False because the slip
            surface cannot generate shear stress.

    """

    def __init__(self, no_slip):
        super().__init__()

        param_dict = ParameterDict(no_slip=bool(no_slip),)
        self._param_dict.update(param_dict)


class ParallelPlates(Geometry):
    r"""Parallel-plate channel.

    Args:
        H (float): Channel half-width.
        V (float): Wall speed.
        no_slip (bool): If True, surfaces have no-slip boundary condition.
            Otherwise, they have the slip boundary condition.

    `ParallelPlates` confines the MPCD particles between two infinite parallel
    plates centered around the origin. The plates are placed at :math:`y=-H`
    and :math:`y=+H`, so the total channel width is :math:`2H`. The plates may
    be put into motion, moving with speeds :math:`-V` and :math:`+V` in the *x*
    direction, respectively. If combined with a no-slip boundary condition,
    this motion can be used to generate simple shear flow.

    Attributes:
        H (float): Channel half-width.

        V (float): Wall speed.

    """

    def __init__(self, H, V=0.0, no_slip=True):
        super().__init__(no_slip)
        param_dict = ParameterDict(
            H=float(H),
            V=float(V),
        )
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        self._cpp_obj = _mpcd.ParallelPlates(self.H, self.V, self.no_slip)
        super()._attach_hook()


class PlanarPore(Geometry):
    r"""Pore with parallel plate opening.

    Args:
        H (float): Pore half-width.
        L (float): Pore half-length.
        no_slip (bool): If True, surfaces have no-slip boundary condition.
            Otherwise, they have the slip boundary condition.

    `PlanarPore` is a finite-length version of `ParallelPlates`. The
    geometry is similar, except that the plates extend from :math:`x=-L` to
    :math:`x=+L` (total length *2L*). Additional solid walls
    with normals in *x* prevent penetration into the regions above / below the
    plates. The plates are infinite in *z*. Outside the pore, the simulation box
    has full periodic boundaries; it is not confined by any walls. This model
    hence mimics a narrow pore in, e.g., a membrane.

    Attributes:
        H (float): Pore half-width.

        L (float): Pore half-length.

    """

    def __init__(self, H, L, no_slip=True):
        super().__init__(no_slip)

        param_dict = ParameterDict(
            H=float(H),
            L=float(L),
        )
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        self._cpp_obj = _mpcd.PlanarPore(self.H, self.L, self.no_slip)
        super()._attach_hook()
