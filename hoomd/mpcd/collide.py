# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""MPCD collision methods.

An MPCD collision method is required to update the particle velocities over
time. Particles are binned into cells based on their positions, and all
particles in a cell undergo a stochastic collision that updates their velocities
while conserving linear momentum. Collision rules can optionally be extended to
also conserve angular momentum The stochastic collisions lead to a build up of
hydrodynamic interactions, and the choice of collision rule determines the
transport coefficients.

.. invisible-code-block: python

    simulation = hoomd.util.make_example_simulation(mpcd_types=["A"])
    simulation.operations.integrator = hoomd.mpcd.Integrator(dt=0.1)

"""

import hoomd
from hoomd.data.parameterdicts import ParameterDict
from hoomd.data.typeconverter import OnlyTypes, variant_preprocessing
from hoomd.mpcd import _mpcd
from hoomd.operation import Compute, Operation


class CellList(Compute):
    """Collision cell list.

    Args:
        shift (bool): When True, randomly shift underlying collision cells.

    The MPCD `CellList` bins particles into cubic cells of edge length 1.0.
    The simulation box must be orthorhombic, and its edges must be an integer.

    When the total mean-free path of the MPCD particles is small, the cells
    should be randomly shifted in order to ensure Galilean invariance of the
    algorithm. This random shift is drawn from a uniform distribution, and it
    can shift the grid by up to half a cell in each direction. The performance
    penalty from grid shifting is small, so it is recommended to enable it in
    all simulations.

    .. rubric:: Example:

    Access default cell list from integrator.

    .. code-block:: python

        cell_list = simulation.operations.integrator.cell_list

    Attributes:
        shift (bool): When True, randomly shift underlying collision cells.

            .. rubric:: Example:

            .. code-block:: python

                cell_list.shift = True

    """

    def __init__(self, shift=True):
        super().__init__()

        param_dict = ParameterDict(shift=bool(shift),)
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        sim = self._simulation
        sim._warn_if_seed_unset()

        if isinstance(sim.device, hoomd.device.GPU):
            cpp_class = _mpcd.CellListGPU
        else:
            cpp_class = _mpcd.CellList

        self._cpp_obj = cpp_class(sim.state._cpp_sys_def, 1.0, self.shift)

        super()._attach_hook()


class CollisionMethod(Operation):
    """Base collision method.

    Args:
        period (int): Number of integration steps between collisions.
        embedded_particles (hoomd.filter.filter_like): HOOMD particles to
            include in collision.

    Attributes:
        embedded_particles (hoomd.filter.filter_like): HOOMD particles to
            include in collision (*read only*).

            These particles are included in per-cell quantities and have their
            velocities updated along with the MPCD particles.

            You will need to create an appropriate method to integrate the
            positions of these particles. The recommended integrator is
            :class:`~hoomd.md.methods.ConstantVolume` with no thermostat (NVE).
            It is generally **not** a good idea to use a thermostat because the
            MPCD particles themselves already act as a heat bath for the
            embedded particles.

            Warning:
                Do not embed particles that are part of a rigid body. Momentum
                will not be correctly transferred to the body. Support for this
                is planned in future.

        period (int): Number of integration steps between collisions
            (*read only*).

            A collision is executed each time the
            :attr:`~hoomd.Simulation.timestep` is a multiple of `period`. It
            must be a multiple of `period` for the
            :class:`~hoomd.mpcd.stream.StreamingMethod` if one is attached to
            the :class:`~hoomd.mpcd.Integrator`.

    """

    def __init__(self, period, embedded_particles=None):
        super().__init__()

        param_dict = ParameterDict(
            period=int(period),
            embedded_particles=OnlyTypes(hoomd.filter.ParticleFilter,
                                         allow_none=True),
        )
        param_dict["embedded_particles"] = embedded_particles
        self._param_dict.update(param_dict)


class AndersenThermostat(CollisionMethod):
    r"""Andersen thermostat collision method.

    Args:
        period (int): Number of integration steps between collisions.
        kT (hoomd.variant.variant_like): Temperature of the thermostat
            :math:`[\mathrm{energy}]`.
        embedded_particles (hoomd.filter.ParticleFilter): HOOMD particles to
            include in collision.


    This class implements the Andersen thermostat collision rule for MPCD, as
    described by `Allahyarov and Gompper
    <https://doi.org/10.1103/PhysRevE.66.036702>`_. Every
    :attr:`~CollisionMethod.period` steps, the particles are binned into cells.
    New particle velocities are then randomly drawn from a Gaussian distribution
    relative to the center-of-mass velocity for the cell. The random velocities
    are given zero mean so that the cell linear momentum is conserved. This
    collision rule naturally imparts the constant-temperature ensemble
    consistent with `kT`.

    .. rubric:: Examples:

    Collision for only MPCD particles.

    .. code-block:: python

        andersen_thermostat = hoomd.mpcd.collide.AndersenThermostat(
            period=1,
            kT=1.0)
        simulation.operations.integrator.collision_method = andersen_thermostat

    Collision including embedded particles.

    .. code-block:: python

        andersen_thermostat = hoomd.mpcd.collide.AndersenThermostat(
            period=20,
            kT=1.0,
            embedded_particles=hoomd.filter.All())
        simulation.operations.integrator.collision_method = andersen_thermostat

    Attributes:
        kT (hoomd.variant.variant_like): Temperature of the thermostat
            :math:`[\mathrm{energy}]`.

            This temperature determines the distribution used to generate the
            random numbers.

            .. rubric:: Examples:

            Constant temperature.

            .. code-block:: python

                andersen_thermostat.kT = 1.0

            Variable temperature.

            .. code-block:: python

                andersen_thermostat.kT = hoomd.variant.Ramp(1.0, 2.0, 0, 100)

    """

    def __init__(self, period, kT, embedded_particles=None):
        super().__init__(period, embedded_particles)

        param_dict = ParameterDict(kT=hoomd.variant.Variant)
        param_dict["kT"] = kT
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        sim = self._simulation
        sim._warn_if_seed_unset()

        if isinstance(sim.device, hoomd.device.GPU):
            cpp_class = _mpcd.ATCollisionMethodGPU
        else:
            cpp_class = _mpcd.ATCollisionMethod

        self._cpp_obj = cpp_class(sim.state._cpp_sys_def, sim.timestep,
                                  self.period, 0, self.kT)

        if self.embedded_particles is not None:
            self._cpp_obj.setEmbeddedGroup(
                sim.state._get_group(self.embedded_particles))

        super()._attach_hook()


class StochasticRotationDynamics(CollisionMethod):
    r"""Stochastic rotation dynamics collision method.

    Args:
        period (int): Number of integration steps between collisions.
        angle (float): Rotation angle (in degrees).
        kT (hoomd.variant.variant_like): Temperature for the collision
            thermostat :math:`[\mathrm{energy}]`. If None, no thermostat is
            used.
        embedded_particles (hoomd.filter.ParticleFilter): HOOMD particles to
            include in collision.

    This class implements the stochastic rotation dynamics (SRD) collision
    rule for MPCD proposed by `Malevanets and Kapral
    <https://doi.org/10.1063/1.478857>`_. Every :attr:`~CollisionMethod.period`
    steps, the particles are binned into cells. The particle velocities are then
    rotated by `angle` around an axis randomly drawn from the unit sphere. The
    rotation is done relative to the average velocity, so this rotation rule
    conserves linear momentum and kinetic energy within each cell.

    The SRD method naturally imparts the NVE ensemble to the system comprising
    the MPCD particles and the `embedded_particles`. Accordingly, the system
    must be properly initialized to the correct temperature. A thermostat can be
    applied in conjunction with the SRD method through the `kT` parameter. SRD
    employs a `Maxwell-Boltzmann thermostat
    <https://doi.org/10.1016/j.jcp.2009.09.024>`_ on the cell level, which
    generates a constant-temperature ensemble. The temperature is defined
    relative to the cell-average velocity, and so can be used to dissipate heat
    in nonequilibrium simulations. Under this thermostat, the SRD algorithm
    still conserves linear momentum, but kinetic energy is no longer conserved.

    .. rubric:: Examples:

    Collision of MPCD particles.

    .. code-block:: python

        srd = hoomd.mpcd.collide.StochasticRotationDynamics(period=1, angle=130)
        simulation.operations.integrator.collision_method = srd

    Collision of MPCD particles with thermostat.

    .. code-block:: python

        srd = hoomd.mpcd.collide.StochasticRotationDynamics(
            period=1,
            angle=130,
            kT=1.0)
        simulation.operations.integrator.collision_method = srd

    Collision including embedded particles.

    .. code-block:: python

        srd = hoomd.mpcd.collide.StochasticRotationDynamics(
            period=20,
            angle=130,
            kT=1.0,
            embedded_particles=hoomd.filter.All())
        simulation.operations.integrator.collision_method = srd

    Attributes:
        angle (float): Rotation angle (in degrees)

            .. rubric:: Example:

            .. code-block:: python

                srd.angle = 130

        kT (hoomd.variant.variant_like): Temperature for the collision
            thermostat :math:`[\mathrm{energy}]`.

            .. rubric:: Examples:

            Constant temperature.

            .. code-block:: python

                srd.kT = 1.0

            Variable temperature.

            .. code-block:: python

                srd.kT = hoomd.variant.Ramp(1.0, 2.0, 0, 100)

    """

    def __init__(self, period, angle, kT=None, embedded_particles=None):
        super().__init__(period, embedded_particles)

        param_dict = ParameterDict(
            angle=float(angle),
            kT=OnlyTypes(hoomd.variant.Variant,
                         allow_none=True,
                         preprocess=variant_preprocessing),
        )
        param_dict["kT"] = kT
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        sim = self._simulation
        sim._warn_if_seed_unset()

        if isinstance(sim.device, hoomd.device.GPU):
            cpp_class = _mpcd.SRDCollisionMethodGPU
        else:
            cpp_class = _mpcd.SRDCollisionMethod

        self._cpp_obj = cpp_class(sim.state._cpp_sys_def, sim.timestep,
                                  self.period, 0, self.angle)

        if self.embedded_particles is not None:
            self._cpp_obj.setEmbeddedGroup(
                sim.state._get_group(self.embedded_particles))

        super()._attach_hook()
