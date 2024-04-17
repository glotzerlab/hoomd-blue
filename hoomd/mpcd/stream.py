# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

r"""MPCD streaming methods.

An MPCD streaming method is required to update the particle positions over time.
It is meant to be used in conjunction with an :class:`.mpcd.Integrator` and
:class:`~hoomd.mpcd.collide.CollisionMethod`. Particle positions are propagated
ballistically according to Newton's equations using a velocity-Verlet scheme for
a time :math:`\Delta t`:

.. math::

    \mathbf{v}(t + \Delta t/2) &= \mathbf{v}(t) + (\mathbf{f}/m)(\Delta t / 2)

    \mathbf{r}(t+\Delta t) &= \mathbf{r}(t) + \mathbf{v}(t+\Delta t/2) \Delta t

    \mathbf{v}(t + \Delta t) &= \mathbf{v}(t + \Delta t/2) +
    (\mathbf{f}/m)(\Delta t / 2)

where **r** and **v** are the particle position and velocity, respectively, and
**f** is the external force acting on the particles of mass *m*. For a list of
forces that can be applied, see :mod:`.mpcd.force`.

.. invisible-code-block: python

    simulation = hoomd.util.make_example_simulation(mpcd_types=["A"])
    simulation.operations.integrator = hoomd.mpcd.Integrator(dt=0.1)

"""

import hoomd
from hoomd.data.parameterdicts import ParameterDict
from hoomd.data.typeconverter import OnlyTypes
from hoomd.mpcd import _mpcd
from hoomd.mpcd.force import BodyForce
from hoomd.mpcd.geometry import Geometry
from hoomd.operation import Operation


class StreamingMethod(Operation):
    """Base streaming method.

    Args:
        period (int): Number of integration steps covered by streaming step.
        mpcd_particle_force (BodyForce): Force on MPCD particles.

    Attributes:
        period (int): Number of integration steps covered by streaming step
            (*read only*).

            The MPCD particles will be streamed every time the
            :attr:`~hoomd.Simulation.timestep` is a multiple of `period`. The
            streaming time is hence equal to `period` steps of the
            :class:`~hoomd.mpcd.Integrator`. Typically `period` should be equal
            to the :attr:`~hoomd.mpcd.collide.CollisionMethod.period` for the
            corresponding collision method. A smaller fraction of this may be
            used if an external force is applied, and more faithful numerical
            integration is needed.

        mpcd_particle_force (BodyForce): Body force on MPCD particles.

            The `mpcd_particle_force` cannot be changed after the
            `StreamingMethod` is constructed, but its attributes can be
            modified.

    """

    def __init__(self, period, mpcd_particle_force=None):
        super().__init__()

        param_dict = ParameterDict(
            period=int(period),
            mpcd_particle_force=OnlyTypes(BodyForce, allow_none=True),
        )
        param_dict["mpcd_particle_force"] = mpcd_particle_force
        self._param_dict.update(param_dict)


class Bulk(StreamingMethod):
    """Bulk fluid.

    Args:
        period (int): Number of integration steps covered by streaming step.
        mpcd_particle_force (BodyForce): Body force on MPCD particles.

    `Bulk` streams the MPCD particles in a fully periodic geometry (2D or 3D).
    This geometry is appropriate for modeling bulk fluids, i.e., those that
    are not confined by any surfaces.

    .. rubric:: Examples:

    Bulk streaming.

    .. code-block:: python

        stream = hoomd.mpcd.stream.Bulk(period=1)
        simulation.operations.integrator.streaming_method = stream

    Bulk streaming with applied force.

    .. code-block:: python

        stream = hoomd.mpcd.stream.Bulk(
            period=1,
            mpcd_particle_force=hoomd.mpcd.force.ConstantForce((1, 0, 0)))
        simulation.operations.integrator.streaming_method = stream

    """

    def _attach_hook(self):
        sim = self._simulation

        # attach and use body force if present
        if self.mpcd_particle_force is not None:
            self.mpcd_particle_force._attach(sim)
            mpcd_particle_force = self.mpcd_particle_force._cpp_obj
        else:
            mpcd_particle_force = None

        # try to find force in map, otherwise use default
        force_type = type(self.mpcd_particle_force)
        try:
            class_info = self._cpp_class_map[force_type]
        except KeyError:
            if self.mpcd_particle_force is not None:
                force_name = force_type.__name__
            else:
                force_name = "NoForce"
            class_info = (
                _mpcd,
                "BulkStreamingMethod" + force_name,
            )
        class_info = list(class_info)
        if isinstance(sim.device, hoomd.device.GPU):
            class_info[1] += "GPU"
        class_ = getattr(*class_info, None)
        assert class_ is not None, ("C++ streaming method could not be "
                                    "determined")

        self._cpp_obj = class_(
            sim.state._cpp_sys_def,
            sim.timestep,
            self.period,
            0,
            mpcd_particle_force,
        )

        super()._attach_hook()

    def _detach_hook(self):
        if self.mpcd_particle_force is not None:
            self.mpcd_particle_force._detach()
        super()._detach_hook()

    _cpp_class_map = {}

    @classmethod
    def _register_cpp_class(cls, force, module, class_name):
        cls._cpp_class_map[force] = (module, class_name)


class BounceBack(StreamingMethod):
    """Streaming with bounce-back rule for surfaces.

    Args:
        period (int): Number of integration steps covered by streaming step.
        geometry (hoomd.mpcd.geometry.Geometry): Surface to bounce back from.
        mpcd_particle_force (BodyForce): Body force on MPCD particles.

    One of the main strengths of the MPCD algorithm is that it can be coupled to
    complex boundaries, defined by a `geometry`. This `StreamingMethod` reflects
    the MPCD particles from boundary surfaces using specular reflections
    (bounce-back) rules consistent with either "slip" or "no-slip" hydrodynamic
    boundary conditions. The external force is only applied to the particles at
    the beginning and the end of this process.

    Although a streaming geometry is enforced on the MPCD particles,
    there are a few important caveats:

    1. Embedded particles are not coupled to the boundary walls. They must be
       confined by an appropriate method, e.g., an external potential, an
       explicit particle wall, or a bounce-back method
       (`hoomd.mpcd.methods.BounceBack`).
    2. The `geometry` exists inside a fully periodic simulation box.
       Hence, the box must be padded large enough that the MPCD cells do not
       interact through the periodic boundary. Usually, this means adding at
       least one extra layer of cells in the confined dimensions. Your periodic
       simulation box will be validated by the `geometry`.
    3. It is an error for MPCD particles to lie "outside" the `geometry`.
       You must initialize your system carefully to ensure all particles are
       "inside" the geometry.

    .. rubric:: Examples:

    Shear flow between moving parallel plates.

    .. code-block:: python

        stream = hoomd.mpcd.stream.BounceBack(
            period=1,
            geometry=hoomd.mpcd.geometry.ParallelPlates(
                separation=6.0, speed=1.0, no_slip=True))
        simulation.operations.integrator.streaming_method = stream

    Pressure driven flow between parallel plates.

    .. code-block:: python

        stream = hoomd.mpcd.stream.BounceBack(
            period=1,
            geometry=hoomd.mpcd.geometry.ParallelPlates(
                separation=6.0, no_slip=True),
            mpcd_particle_force=hoomd.mpcd.force.ConstantForce((1, 0, 0)))
        simulation.operations.integrator.streaming_method = stream

    Attributes:
        geometry (hoomd.mpcd.geometry.Geometry): Surface to bounce back from
            (*read only*).

    """

    _cpp_class_map = {}

    def __init__(self, period, geometry, mpcd_particle_force=None):
        super().__init__(period, mpcd_particle_force)

        param_dict = ParameterDict(geometry=Geometry)
        param_dict["geometry"] = geometry
        self._param_dict.update(param_dict)

    def check_mpcd_particles(self):
        """Check if MPCD particles are inside `geometry`.

        This method can only be called after this object is attached to a
        simulation.

        Returns:
            True if all MPCD particles are inside `geometry`.

        .. rubric:: Examples:

        .. code-block:: python

            assert stream.check_mpcd_particles()

        """
        return self._cpp_obj.check_mpcd_particles()

    def _attach_hook(self):
        sim = self._simulation

        self.geometry._attach(sim)

        # attach and use body force if present
        if self.mpcd_particle_force is not None:
            self.mpcd_particle_force._attach(sim)
            mpcd_particle_force = self.mpcd_particle_force._cpp_obj
        else:
            mpcd_particle_force = None

        # try to find force in map, otherwise use default
        geom_type = type(self.geometry)
        force_type = type(self.mpcd_particle_force)
        try:
            class_info = self._cpp_class_map[geom_type, force_type]
        except KeyError:
            if self.mpcd_particle_force is not None:
                force_name = force_type.__name__
            else:
                force_name = "NoForce"

            class_info = (
                _mpcd,
                "BounceBackStreamingMethod" + geom_type.__name__ + force_name,
            )
        class_info = list(class_info)
        if isinstance(sim.device, hoomd.device.GPU):
            class_info[1] += "GPU"
        class_ = getattr(*class_info, None)
        assert class_ is not None, "Streaming method for geometry not found"

        self._cpp_obj = class_(
            sim.state._cpp_sys_def,
            sim.timestep,
            self.period,
            0,
            self.geometry._cpp_obj,
            mpcd_particle_force,
        )

        super()._attach_hook()

    def _detach_hook(self):
        self.geometry._detach()
        if self.mpcd_particle_force is not None:
            self.mpcd_particle_force._detach()
        super()._detach_hook()

    @classmethod
    def _register_cpp_class(cls, geometry, force, module, cpp_class_name):
        # we will allow None for the force, but we need its class type not value
        if force is None:
            force = type(None)
        cls._cpp_cpp_class_map[geometry, force] = (module, cpp_class_name)
