# Copyright (c) 2009-2025 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Box deformers for MD simulations.

Box deformers apply time-dependent transformations to the simulation box.
These transformations enable deformation protocols such as shear, strain,
and controlled box shape evolution during molecular dynamics simulations.

Classes in this module are intended to be assigned to
:attr:`hoomd.md.Integrator.box_deformer`.
"""

from hoomd.md import _md
from hoomd.data.parameterdicts import ParameterDict
from hoomd.data.typeconverter import OnlyTypes, positive_real


class BoxDeformer:
    """Base class for all box deformers.

    A :class:`BoxDeformer` defines how the simulation box evolves over time.
    Subclasses implement specific deformation protocols.

    Instances must be attached to a :class:`hoomd.Simulation` before they
    can modify the simulation box.
    """

    def __init__(self):
        """Initialize the box deformer.

        Newly created instances are detached and do not affect any simulation
        until :meth:`_attach` is called internally by HOOMD.
        """
        self._attached = False
        self._simulation = None
        self._param_dict = ParameterDict()
        self._cpp_obj_private = None

    def _attach(self, simulation):
        """Attach the box deformer to a simulation.

        Args:
            simulation (:class:`hoomd.Simulation`):
                The simulation instance to attach to.
        """
        self._simulation = simulation
        self._attached = True

    def _detach(self):
        """Detach the box deformer from the simulation.

        After detachment, the object no longer modifies the simulation box.
        """
        self._attached = False
        self._simulation = None
        self._cpp_obj_private = None

    @property
    def _cpp_obj(self):
        """Access the underlying box deformer object.

        Returns:
            The box deformer instance.

        Raises:
            RuntimeError:
                If the object is not yet attached to a simulation.
        """
        if self._cpp_obj_private is None:
            raise RuntimeError(
                "box deformer object not created yet. Attach to a simulation first."
            )
        return self._cpp_obj_private


class LeesEdwardsBoxDeformer(BoxDeformer):
    r"""Lees-Edwards shear box deformer.

    Applies Lees-Edwards boundary conditions to impose a constant shear rate
    :math:`\dot{\gamma}` in the xy-plane. This method is commonly used to model
    steady shear flow in nonequilibrium molecular dynamics simulations.

    The box tilt evolves as:

    .. math::

        xy(t) = xy(0) + \dot{\gamma} \, t

    See Also:
        :class:`hoomd.md.Integrator`
        :attr:`hoomd.md.Integrator.box_deformer`

    Args:
        shear_rate (float):
            Shear rate :math:`[\mathrm{time}^{-1}]`.
        max_xy_tilt (float, optional):
            Maximum allowed xy tilt factor before box flipping occurs.
            Must be positive. Defaults to ``0.5``.

    Example::

        deformer = hoomd.md.box_deformer.LeesEdwardsBoxDeformer(
            shear_rate=0.01,
            max_xy_tilt=0.5,
        )

        integrator = hoomd.md.Integrator(dt=0.001)
        integrator.box_deformer = deformer
        simulation.operations.integrator = integrator
    """

    def __init__(self, shear_rate, max_xy_tilt=0.5):
        super().__init__()

        param_dict = ParameterDict(
            shear_rate=float(shear_rate),
            max_xy_tilt=OnlyTypes(float, preprocess=positive_real),
        )
        param_dict["max_xy_tilt"] = max_xy_tilt
        self._param_dict.update(param_dict)

    def _attach(self, simulation):
        """Attach the deformer and construct it.

        Called automatically when assigned to
        :attr:`hoomd.md.Integrator.box_deformer`.
        """
        super()._attach(simulation)
        sysdef = simulation.state._cpp_sys_def
        self._cpp_obj_private = _md.LeesEdwardsBoxDeformer(
            sysdef, self.shear_rate, self.max_xy_tilt
        )

    @property
    def shear_rate(self):
        r"""float: Shear rate :math:`\dot{\gamma}`.

        Controls the rate of shear applied to the simulation box.
        """
        return self._param_dict["shear_rate"]

    @shear_rate.setter
    def shear_rate(self, value):
        """Set the shear rate."""
        self._param_dict["shear_rate"] = float(value)
        if self._attached:
            self._cpp_obj.setShearRate(float(value))

    @property
    def max_xy_tilt(self):
        r"""float: Maximum allowed xy tilt before box flipping occurs.
        The tilt is constrained such that:

        .. math::

            -\mathrm{max\_xy\_tilt} \le xy(t) \le \mathrm{max\_xy\_tilt}
        """
        return self._param_dict["max_xy_tilt"]

    @max_xy_tilt.setter
    def max_xy_tilt(self, value):
        """Set the maximum allowed xy tilt."""
        self._param_dict["max_xy_tilt"] = float(value)
        if self._attached:
            self._cpp_obj.setMaxXYTilt(float(value))


__all__ = [
    "BoxDeformer",
    "LeesEdwardsBoxDeformer",
]
