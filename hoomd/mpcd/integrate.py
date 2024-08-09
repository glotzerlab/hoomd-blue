# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Implement MPCD Integrator."""

import hoomd
from hoomd.data.parameterdicts import ParameterDict
from hoomd.data import syncedlist
from hoomd.data.typeconverter import OnlyTypes
from hoomd.md.integrate import Integrator as _MDIntegrator, _set_synced_list
from hoomd.mpcd import _mpcd
from hoomd.mpcd.collide import CellList, CollisionMethod
from hoomd.mpcd.fill import VirtualParticleFiller
from hoomd.mpcd.stream import StreamingMethod
from hoomd.mpcd.tune import ParticleSorter


@hoomd.logging.modify_namespace(("mpcd", "Integrator"))
class Integrator(_MDIntegrator):
    r"""MPCD integrator.

    Args:
        dt (float): Integrator time step size :math:`[\mathrm{time}]`.

        methods (Sequence[hoomd.md.methods.Method]): Sequence of integration
          methods. The default value of ``None`` initializes an empty list.

        forces (Sequence[hoomd.md.force.Force]): Sequence of forces applied to
          the particles in the system. The default value of ``None`` initializes
          an empty list.

        integrate_rotational_dof (bool): When True, integrate rotational degrees
          of freedom.

        constraints (Sequence[hoomd.md.constrain.Constraint]): Sequence of
          constraint forces applied to the particles in the system.
          The default value of ``None`` initializes an empty list. Rigid body
          objects (i.e. `hoomd.md.constrain.Rigid`) are not allowed in the
          list.

        rigid (hoomd.md.constrain.Rigid): An object defining the rigid bodies in
          the simulation.

        half_step_hook (hoomd.md.HalfStepHook): Enables the user to perform
            arbitrary computations during the half-step of the integration.

        streaming_method (hoomd.mpcd.stream.StreamingMethod): Streaming method
            for the MPCD particles.

        collision_method (hoomd.mpcd.collide.CollisionMethod): Collision method
            for the MPCD particles and any embedded particles.

        virtual_particle_fillers
            (Sequence[hoomd.mpcd.fill.VirtualParticleFiller]): MPCD
            virtual-particle filler(s).

        mpcd_particle_sorter (hoomd.mpcd.tune.ParticleSorter): Tuner that sorts
            the MPCD particles.

    The MPCD `Integrator` enables the MPCD algorithm concurrently with standard
    MD methods.

    In MPCD simulations, :attr:`~hoomd.md.Integrator.dt` defines the amount of
    time that the system is advanced forward every time step. MPCD streaming and
    collision steps can be defined to occur in multiples of
    :attr:`~hoomd.md.Integrator.dt`. In these cases, any MD particle data will
    be updated every :attr:`~hoomd.md.Integrator.dt`, while the MPCD particle
    data is updated asynchronously for performance. For example, if MPCD
    streaming happens every 5 steps, then the particle data will be updated as
    follows::

                0     1     2     3     4     5
        MD:     |---->|---->|---->|---->|---->|
        MPCD:   |---------------------------->|

    If the MPCD particle data is accessed via the snapshot interface at time
    step 3, it will actually contain the MPCD particle data for time step 5.
    The MD particles can be read at any time step because their positions
    are updated every step.

    .. rubric:: Examples:

    .. invisible-code-block: python

        simulation = hoomd.util.make_example_simulation(mpcd_types=["A"])

    Integrator for only MPCD particles.

    .. code-block:: python

        stream = hoomd.mpcd.stream.Bulk(period=1)
        collide = hoomd.mpcd.collide.StochasticRotationDynamics(
            period=1,
            angle=130)
        integrator = hoomd.mpcd.Integrator(
            dt=0.1,
            streaming_method=stream,
            collision_method=collide,
            mpcd_particle_sorter=hoomd.mpcd.tune.ParticleSorter(trigger=20))
        simulation.operations.integrator = integrator

    MPCD integrator with solutes.

    .. code-block:: python

        dt_md = 0.005
        md_steps_per_collision = 20 # collision time = 0.1

        stream = hoomd.mpcd.stream.Bulk(period=md_steps_per_collision)
        collide = hoomd.mpcd.collide.StochasticRotationDynamics(
            period=md_steps_per_collision,
            angle=130,
            embedded_particles=hoomd.filter.All())
        solute_method = hoomd.md.methods.ConstantVolume(
            filter=collide.embedded_particles)

        integrator = hoomd.mpcd.Integrator(
            dt=dt_md,
            methods=[solute_method],
            streaming_method=stream,
            collision_method=collide,
            mpcd_particle_sorter=hoomd.mpcd.tune.ParticleSorter(
                trigger=20*md_steps_per_collision)
            )
        simulation.operations.integrator = integrator

    MPCD integrator with virtual particle filler.

    .. code-block:: python

        plates = hoomd.mpcd.geometry.ParallelPlates(separation=6.0)
        stream = hoomd.mpcd.stream.BounceBack(period=1, geometry=plates)
        collide = hoomd.mpcd.collide.StochasticRotationDynamics(
            period=1,
            angle=130,
            kT=1.0)
        filler = hoomd.mpcd.fill.GeometryFiller(
            type="A",
            density=5.0,
            kT=1.0,
            geometry=plates)

        integrator = hoomd.mpcd.Integrator(
            dt=0.1,
            streaming_method=stream,
            collision_method=collide,
            virtual_particle_fillers=[filler],
            mpcd_particle_sorter=hoomd.mpcd.tune.ParticleSorter(trigger=20))
        simulation.operations.integrator = integrator

    Attributes:
        collision_method (hoomd.mpcd.collide.CollisionMethod): Collision method
            for the MPCD particles and any embedded particles.

        mpcd_particle_sorter (hoomd.mpcd.tune.ParticleSorter): Tuner that sorts
            the MPCD particles (recommended).

        streaming_method (hoomd.mpcd.stream.StreamingMethod): Streaming method
            for the MPCD particles.

    """

    def __init__(
        self,
        dt,
        integrate_rotational_dof=False,
        forces=None,
        constraints=None,
        methods=None,
        rigid=None,
        half_step_hook=None,
        streaming_method=None,
        collision_method=None,
        virtual_particle_fillers=None,
        mpcd_particle_sorter=None,
    ):
        super().__init__(
            dt,
            integrate_rotational_dof,
            forces,
            constraints,
            methods,
            rigid,
            half_step_hook,
        )

        self._cell_list = CellList()

        virtual_particle_fillers = ([] if virtual_particle_fillers is None else
                                    virtual_particle_fillers)
        self._virtual_particle_fillers = syncedlist.SyncedList(
            VirtualParticleFiller,
            syncedlist._PartialGetAttr("_cpp_obj"),
            iterable=virtual_particle_fillers,
        )

        param_dict = ParameterDict(
            streaming_method=OnlyTypes(StreamingMethod, allow_none=True),
            collision_method=OnlyTypes(CollisionMethod, allow_none=True),
            mpcd_particle_sorter=OnlyTypes(ParticleSorter, allow_none=True),
        )
        param_dict.update(
            dict(
                streaming_method=streaming_method,
                collision_method=collision_method,
                mpcd_particle_sorter=mpcd_particle_sorter,
            ))
        self._param_dict.update(param_dict)

    @property
    def cell_list(self):
        """hoomd.mpcd.collide.CellList: Collision cell list.

        A `CellList` is automatically created with each `Integrator` using the
        default settings.

        """
        return self._cell_list

    @property
    def virtual_particle_fillers(self):
        """Sequence[hoomd.mpcd.fill.VirtualParticleFiller]: MPCD \
        virtual-particle fillers."""
        return self._virtual_particle_fillers

    @virtual_particle_fillers.setter
    def virtual_particle_fillers(self, value):
        _set_synced_list(self._virtual_particle_fillers, value)

    def _attach_hook(self):
        self._cell_list._attach(self._simulation)
        if self.streaming_method is not None:
            self.streaming_method._attach(self._simulation)
        if self.collision_method is not None:
            self.collision_method._attach(self._simulation)
        if self.mpcd_particle_sorter is not None:
            self.mpcd_particle_sorter._attach(self._simulation)

        self._cpp_obj = _mpcd.Integrator(self._simulation.state._cpp_sys_def,
                                         self.dt)
        self._virtual_particle_fillers._sync(self._simulation,
                                             self._cpp_obj.fillers)
        self._cpp_obj.cell_list = self._cell_list._cpp_obj

        super(_MDIntegrator, self)._attach_hook()

    def _detach_hook(self):
        self._cell_list._detach()
        self._virtual_particle_fillers._unsync()
        if self.streaming_method is not None:
            self.streaming_method._detach()
        if self.collision_method is not None:
            self.collision_method._detach()
        if self.mpcd_particle_sorter is not None:
            self.mpcd_particle_sorter._detach()

        super()._detach_hook()

    def _setattr_param(self, attr, value):
        if attr in ("streaming_method", "collision_method",
                    "mpcd_particle_sorter"):
            cur_value = getattr(self, attr)
            if value is cur_value:
                return

            if value is not None and value._attached:
                raise ValueError("Cannot attach to multiple integrators.")

            # if already attached, change out and set parameter
            if self._attached:
                if cur_value is not None:
                    cur_value._detach()
                if value is not None:
                    value._attach(self._simulation)
            self._param_dict[attr] = value
        else:
            super()._setattr_param(attr, value)
