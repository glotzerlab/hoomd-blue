# Copyright (c) 2009-2025 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

r"""Box deformation operations for MD simulations.

Box deformers apply time-dependent deformations to the simulation box.

These operations are typically attached to a `hoomd.md.Integrator` and are
applied every time step.

In general, a box deformation modifies the simulation box matrix
:math:`\mathbf{H}` as a function of time:

.. math::

    \mathbf{H}(t)

which may include changes in box lengths and/or tilt factors.

"""

from hoomd.md import _md
from hoomd.data.parameterdicts import ParameterDict
from hoomd.data.typeconverter import OnlyTypes, positive_real
from hoomd.operation import _HOOMDBaseObject
import inspect


class BoxDeformer(_HOOMDBaseObject):
    r"""Base class for box deformers.

    `BoxDeformer` defines the interface for time-dependent box deformations
    applied during a simulation run.

    Subclasses implement specific deformation protocols (e.g., shear,
    elongation, etc.). Instances are attached to a
    `hoomd.md.Integrator`.

    Warning:
        This class should not be instantiated directly by users. Use a
        subclass instead.

    {inherited}

    ----------

    **Members defined in** `BoxDeformer`:

    Subclasses define their specific deformation protocols and parameters.
    """

    _doc_inherited = """
    ----------

    **Members inherited from**
    `BoxDeformer <hoomd.md.deformer.BoxDeformer>`:
    """

    pass


class LeesEdwardsBoxDeformer(BoxDeformer):
    r"""Lees-Edwards shear box deformer.

    Args:
        shear_rate (float): Shear rate :math:`[\mathrm{time}^{-1}]`.

        max_tilt (float, optional): Maximum allowed value of the box tilt
            factor ``xy`` before a box flip is performed. It must be positive.
            Defaults to 0.5.

    `LeesEdwardsBoxDeformer` applies Lees-Edwards boundary conditions to impose
    a homogeneous shear flow in the simulation box by manipulating the box tilt
    factor ``xy``.

    The box tilt deforms as

    .. math::

        xy(t) = xy(0) + \dot{\gamma} \, t

    where :math:`\dot{\gamma}` is the imposed shear rate.

    This produces a linear velocity profile

    .. math::

        v_x(y) = \dot{\gamma} y

    corresponding to a simple shear flow in the *x*-direction, with gradient in
    the *y*-direction.

    The ``max_tilt`` parameter controls when the simulation box is flipped
    to avoid excessive skew. When :math:`|xy| > \mathrm{max\_tilt}`,
    the box is flipped (remapped by a lattice vector) to bring the
    tilt back into the range :math:`[-\mathrm{max\_tilt}, +\mathrm{max\_tilt}]`.

    Box flipping and particle remap is mathematically equivalent to the original
    sheared system and does not affect the dynamics or measured properties, but it
    improves numerical stability and avoids highly distorted boxes.

    Recommended usage::

        - Values of `max_tilt` around 0.5 are commonly used and provide
        a good balance between minimizing remapping frequency and avoiding
        extreme box distortions.

        - Smaller values lead to more frequent flips, while larger values may
        result in highly skewed boxes that can impact computational performance.

    {inherited}

    ----------

    **Members defined in** `LeesEdwardsBoxDeformer`:

    Attributes:
        shear_rate (float):
            Imposed shear rate :math:`[\mathrm{time}^{-1}]`.

        max_tilt (float):
            Maximum allowed tilt before remapping.

    Example::

        deformer = hoomd.md.deformer.LeesEdwardsBoxDeformer(
            shear_rate=0.01,
            max_tilt=0.5,
        )
        integrator = hoomd.md.Integrator(dt=0.005, deformer=deformer)
        simulation.operations.integrator = integrator
    """

    __doc__ = inspect.cleandoc(__doc__).replace(
        "{inherited}", inspect.cleandoc(BoxDeformer._doc_inherited)
    )

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
