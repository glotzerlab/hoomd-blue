# Copyright (c) 2009-2025 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Box deformers for MD simulations.

Box deformers apply time-dependent transformations to the simulation box.
They are intended to be attached to a `hoomd.md.Integrator`.
"""

from hoomd.md import _md
from hoomd.data.parameterdicts import ParameterDict
from hoomd.data.typeconverter import OnlyTypes, positive_real
from hoomd.operation import _HOOMDBaseObject


class BoxDeformer(_HOOMDBaseObject):
    """Base class for box deformers.

    Subclasses implement specific deformation protocols. Instances are attached
    to a :class:`hoomd.md.Integrator`.
    """

    pass


class LeesEdwardsBoxDeformer(BoxDeformer):
    r"""Lees-Edwards shear box deformer.

    Applies Lees-Edwards boundary conditions to impose a constant shear
    rate :math:`\dot{\gamma}` in the xy-plane:

    .. math::

        xy(t) = xy(0) + \dot{\gamma} \, t

    Args:
        shear_rate (float): shear rate :math:`[\mathrm{time}^{-1}]`.
        max_tilt (float, optional): maximum xy tilt factor before
            box flipping occurs (value must be positive; default value is 0.5).

    Example::

        deformer = hoomd.md.deformer.LeesEdwardsBoxDeformer(
            shear_rate=0.01,
            max_tilt=0.5,
        )
        integrator = hoomd.md.Integrator(dt=0.001, deformer=deformer)
        simulation.operations.integrator = integrator
    """

    def __init__(self, shear_rate, max_tilt=0.5):
        super().__init__()

        param_dict = ParameterDict(
            shear_rate=float(shear_rate),
            max_tilt=OnlyTypes(float, preprocess=positive_real),
        )
        param_dict["max_tilt"] = max_tilt
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        """Create the underlying C++ Lees-Edwards deformer object."""
        self._cpp_obj = _md.LeesEdwardsBoxDeformer(
            self._simulation.state._cpp_sys_def,
            self.shear_rate,
            self.max_tilt,
        )

        super()._attach_hook()


__all__ = [
    "BoxDeformer",
    "LeesEdwardsBoxDeformer",
]
