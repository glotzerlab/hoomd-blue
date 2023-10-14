# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

r""" MPCD streaming methods.

An MPCD streaming method is required to update the particle positions over time.
It is meant to be used in conjunction with an `.mpcd.Integrator`
and `.mpcd.collide.CollisionMethod`. Particle positions are
propagated ballistically according to Newton's equations using a velocity-Verlet
scheme for a time :math:`\Delta t`:

.. math::

    \mathbf{v}(t + \Delta t/2) &= \mathbf{v}(t) + (\mathbf{f}/m)(\Delta t / 2)

    \mathbf{r}(t+\Delta t) &= \mathbf{r}(t) + \mathbf{v}(t+\Delta t/2) \Delta t

    \mathbf{v}(t + \Delta t) &= \mathbf{v}(t + \Delta t/2) + (\mathbf{f}/m)(\Delta t / 2)

where **r** and **v** are the particle position and velocity, respectively, and **f**
is the external force acting on the particles of mass *m*. For a list of forces
that can be applied, see :py:mod:`.mpcd.force`.

Since one of the main strengths of the MPCD algorithm is that it can be coupled to
complex boundaries, the streaming geometry can be configured. MPCD solvent particles
will be reflected from boundary surfaces using specular reflections (bounce-back)
rules consistent with either "slip" or "no-slip" hydrodynamic boundary conditions.
(The external force is only applied to the particles at the beginning and the end
of this process.) To help fully enforce the boundary conditions, "virtual" MPCD
particles can be inserted near the boundary walls.

Although a streaming geometry is enforced on the MPCD solvent particles, there are
a few important caveats:

    1. Embedded particles are not coupled to the boundary walls. They must be confined
       by an appropriate method, e.g., an external potential, an explicit particle wall,
       or a bounce-back method.
    2. The confined geometry exists inside a fully periodic simulation box. Hence, the
       box must be padded large enough that the MPCD cells do not interact through the
       periodic boundary. Usually, this means adding at least one extra layer of cells
       in the confined dimensions. Your periodic simulation box will be validated by
       the confined geometry.
    3. It is an error for MPCD particles to lie "outside" the confined geometry. You
       must initialize your system carefully to ensure all particles are "inside" the
       geometry. An error will be raised otherwise.

"""

import hoomd
from hoomd.data.parameterdicts import ParameterDict
from hoomd.mpcd import _mpcd
from hoomd.operation import AutotunedObject


# TODO: add force and filler
class StreamingMethod(AutotunedObject):
    """Base streaming method.

    Args:
        every (int): Number of integration steps covered by streaming step.

    Attributes:
        every (int): Number of integration steps covered by streaming step.

            The MPCD particles with be streamed every time the `~hoomd.Simulation`
            timestep is a multiple of `every`. The streaming time is hence equal
            to `every` steps of the `~hoomd.mpcd.Integrator`. Typically `every`
            should be equal to `~hoomd.mpcd.CollisionMethod.every` for the
            corresponding collision method. A smaller fraction of this may be
            used if an external force is applied, and more faithful numerical
            integration is needed.

    """

    def __init__(self, every):
        super().__init__()

        param_dict = ParameterDict(every=int(every),)
        self._param_dict.update(param_dict)


class Bulk(StreamingMethod):
    """Bulk fluid.

    Args:
        every (int): Number of integration steps covered by streaming step.

    `Bulk` streams the MPCD particles in a fully periodic geometry (2D or 3D).
    This geometry is appropriate for modeling bulk fluids.

    """

    def _attach_hook(self):
        sim = self._simulation
        if isinstance(sim.device, hoomd.device.GPU):
            class_ = _mpcd.ConfinedStreamingMethodGPUBulk
        else:
            class_ = _mpcd.ConfinedStreamingMethodBulk

        self._cpp_obj = class_(
            sim.state._cpp_sys_def,
            sim.timestep,
            self.every,
            0,
            _mpcd.BulkGeometry(),
        )

        super()._attach_hook()


class ParallelPlates(StreamingMethod):
    r"""Parallel-plate channel.

    Args:
        every (int): Number of integration steps covered by streaming step.
        H (float): Channel half-width.
        V (float): Wall speed.
        no_slip (bool): If True, plates have no-slip boundary condition.
            Otherwise, they have the slip boundary condition.

    `Slit` streams the MPCD particles between two infinite parallel
    plates. The slit is centered around the origin, and the walls are placed
    at :math:`z=-H` and :math:`z=+H`, so the total channel width is :math:`2H`.
    The walls may be put into motion, moving with speeds :math:`-V` and
    :math:`+V` in the *x* direction, respectively. If combined with a
    no-slip boundary condition, this motion can be used to generate simple
    shear flow.

    Attributes:
        H (float): Channel half-width.

        V (float): Wall speed.

        no_slip (bool): If True, plates have no-slip boundary condition.
            Otherwise, they have the slip boundary condition.

            `V` will have no effect if `no_slip` is False because the slip
            surface cannot generate shear stress.

    """

    def __init__(self, every, H, V=0.0, no_slip=True):
        super().__init__(every)

        param_dict = ParameterDict(
            H=float(H),
            V=float(V),
            no_slip=bool(no_slip),
        )
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        sim = self._simulation
        if isinstance(sim.device, hoomd.device.GPU):
            class_ = _mpcd.ConfinedStreamingMethodGPUSlit
        else:
            class_ = _mpcd.ConfinedStreamingMethodSlit

        bc = _mpcd.boundary.no_slip if self.no_slip else _mpcd.boundary.slip
        slit = _mpcd.SlitGeometry(self.H, self.V, bc)
        self._cpp_obj = class_(
            sim.state._cpp_sys_def,
            sim.timestep,
            self.every,
            0,
            slit,
        )

        super()._attach_hook()


class PlanarPore(StreamingMethod):
    r"""Parallel plate pore.

    Args:
        every (int): Number of integration steps covered by streaming step.
        H (float): Channel half-width.
        L (float): Pore half-length.
        no_slip (bool): If True, plates have no-slip boundary condition.
            Otherwise, they have the slip boundary condition.

    `PlanarPore` is a finite-length pore version of `ParallelPlates`. The
    geometry is similar, except that the plates extend from :math:`x=-L` to
    :math:`x=+L` (total length *2L*). Additional solid walls
    with normals in *x* prevent penetration into the regions above / below the
    plates. The plates are infinite in *y*. Outside the pore, the simulation box
    has full periodic boundaries; it is not confined by any walls. This model
    hence mimics a narrow pore in, e.g., a membrane.

    .. image:: mpcd_slit_pore.png

    Attributes:
        H (float): Channel half-width.

        L (float): Pore half-length.

        no_slip (bool): If True, plates have no-slip boundary condition.
            Otherwise, they have the slip boundary condition.

    """

    def __init__(self, trigger, H, L, no_slip=True):
        super().__init__(trigger)

        param_dict = ParameterDict(
            H=float(H),
            L=float(L),
            no_slip=bool(no_slip),
        )
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        sim = self._simulation
        if isinstance(sim.device, hoomd.device.GPU):
            class_ = _mpcd.ConfinedStreamingMethodGPUSlitPore
        else:
            class_ = _mpcd.ConfinedStreamingMethodSlitPore

        bc = _mpcd.boundary.no_slip if self.no_slip else _mpcd.boundary.slip
        slit = _mpcd.SlitPoreGeometry(self.H, self.L, bc)
        self._cpp_obj = class_(
            sim.state._cpp_sys_def,
            sim.timestep,
            self.every,
            0,
            slit,
        )

        super()._attach_hook()
