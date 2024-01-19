# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

r""" MPCD streaming methods.

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

"""

import hoomd
from hoomd.data.parameterdicts import ParameterDict
from hoomd.mpcd import _mpcd
from hoomd.mpcd.geometry import Geometry
from hoomd.operation import Operation


# TODO: add force
class StreamingMethod(Operation):
    """Base streaming method.

    Args:
        period (int): Number of integration steps covered by streaming step.

    Attributes:
        period (int): Number of integration steps covered by streaming step.

            The MPCD particles will be streamed every time the
            :attr:`~hoomd.Simulation.timestep` is a multiple of `period`. The
            streaming time is hence equal to `period` steps of the
            :class:`~hoomd.mpcd.Integrator`. Typically `period` should be equal
            to the :attr:`~hoomd.mpcd.collide.CollisionMethod.period` for the
            corresponding collision method. A smaller fraction of this may be
            used if an external force is applied, and more faithful numerical
            integration is needed.

    """

    def __init__(self, period):
        super().__init__()

        param_dict = ParameterDict(period=int(period),)
        self._param_dict.update(param_dict)


class Bulk(StreamingMethod):
    """Bulk fluid.

    Args:
        period (int): Number of integration steps covered by streaming step.

    `Bulk` streams the MPCD particles in a fully periodic geometry (2D or 3D).
    This geometry is appropriate for modeling bulk fluids, i.e., those that
    are not confined by any surfaces.

    """

    def _attach_hook(self):
        sim = self._simulation
        if isinstance(sim.device, hoomd.device.GPU):
            class_ = _mpcd.BulkStreamingMethodNoForceGPU
        else:
            class_ = _mpcd.BulkStreamingMethodNoForce

        self._cpp_obj = class_(sim.state._cpp_sys_def, sim.timestep,
                               self.period, 0, None)

        super()._attach_hook()


class BounceBack(StreamingMethod):
    """Streaming with bounce-back rule for surfaces.

    Args:
        period (int): Number of integration steps covered by streaming step.
        geometry (hoomd.mpcd.geometry.Geometry): Surface to bounce back from.

    One of the main strengths of the MPCD algorithm is that it can be coupled
    to complex boundaries, defined by a `geometry`. This `StreamingMethod` reflects
    the MPCD solvent particles from boundary surfaces using specular reflections
    (bounce-back) rules consistent with either "slip" or "no-slip" hydrodynamic
    boundary conditions. (The external force is only applied to the particles at the
    beginning and the end of this process.)

    Although a streaming geometry is enforced on the MPCD solvent particles, there
    are a few important caveats:

    1. Embedded particles are not coupled to the boundary walls. They must be
       confined by an appropriate method, e.g., an external potential, an
       explicit particle wall, or a bounce-back method (`hoomd.mpcd.methods.BounceBack`).
    2. The `geometry` exists inside a fully periodic simulation box.
       Hence, the box must be padded large enough that the MPCD cells do not
       interact through the periodic boundary. Usually, this means adding at
       least one extra layer of cells in the confined dimensions. Your periodic
       simulation box will be validated by the `geometry`.
    3. It is an error for MPCD particles to lie "outside" the `geometry`.
       You must initialize your system carefully to ensure all particles are
       "inside" the geometry. An error will be raised otherwise.

    Attributes:
        geometry (hoomd.mpcd.geometry.Geometry): Surface to bounce back from.

    """

    def __init__(self, period, geometry):
        super().__init__(period)

        param_dict = ParameterDict(geometry=Geometry)
        param_dict["geometry"] = geometry
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        sim = self._simulation

        self.geometry._attach(sim)

        # try to find class in map, otherwise default to internal MPCD module
        geom_type = type(self.geometry)
        try:
            class_info = self._class_map[geom_type]
        except KeyError:
            class_info = (
                _mpcd,
                "BounceBackStreamingMethod" + geom_type.__name__ + "NoForce",
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
            None,
        )

        super()._attach_hook()

    def _detach_hook(self):
        self.geometry._detach()
        super()._detach_hook()

    _class_map = {}

    @classmethod
    def _register_geometry(cls, geometry, module, class_name):
        cls._class_map[geometry] = (module, class_name)
