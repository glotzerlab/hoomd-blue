# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

"""Apply forces to particles."""

from abc import abstractmethod

import hoomd
from hoomd.md import _md
from hoomd.operation import _HOOMDBaseObject
from hoomd.logging import log
from hoomd.data.typeparam import TypeParameter
from hoomd.data.typeconverter import OnlyTypes
from hoomd.data.parameterdicts import ParameterDict, TypeParameterDict
from hoomd.filter import ParticleFilter
from hoomd.md.manifold import Manifold
import numpy


class _force:  # noqa - This will be removed eventually. Needed to build docs.
    pass


class Force(_HOOMDBaseObject):
    """Defines a force in HOOMD-blue.

    Pair, angle, bond, and other forces are subclasses of this class.

    Note:
        :py:class:`Force` is the base class for all loggable forces.
        Users should not instantiate this class directly.

    Initializes some loggable quantities.
    """

    @log(requires_run=True)
    def energy(self):
        """float: Total contribution to the potential energy of the system \
        :math:`[\\mathrm{energy}]`."""
        self._cpp_obj.compute(self._simulation.timestep)
        return self._cpp_obj.calcEnergySum()

    @log(category="particle", requires_run=True)
    def energies(self):
        """(*N_particles*, ) `numpy.ndarray` of ``float``: Energy \
        contribution from each particle :math:`[\\mathrm{energy}]`.

        Attention:
            In MPI parallel execution, the array is available on rank 0 only.
            `energies` is `None` on ranks >= 1.
        """
        self._cpp_obj.compute(self._simulation.timestep)
        return self._cpp_obj.getEnergies()

    @log(requires_run=True)
    def additional_energy(self):
        """float: Additional energy term not included in `energies` \
        :math:`[\\mathrm{energy}]`."""
        self._cpp_obj.compute(self._simulation.timestep)
        return self._cpp_obj.getExternalEnergy()

    @log(category="particle", requires_run=True)
    def forces(self):
        """(*N_particles*, 3) `numpy.ndarray` of ``float``: The \
        force applied to each particle :math:`[\\mathrm{force}]`.

        Attention:
            In MPI parallel execution, the array is available on rank 0 only.
            `forces` is `None` on ranks >= 1.
        """
        self._cpp_obj.compute(self._simulation.timestep)
        return self._cpp_obj.getForces()

    @log(category="particle", requires_run=True)
    def torques(self):
        """(*N_particles*, 3) `numpy.ndarray` of ``float``: The torque applied \
        to each particle :math:`[\\mathrm{force} \\cdot \\mathrm{length}]`.

        Attention:
            In MPI parallel execution, the array is available on rank 0 only.
            `torques` is `None` on ranks >= 1.
        """
        self._cpp_obj.compute(self._simulation.timestep)
        return self._cpp_obj.getTorques()

    @log(category="particle", requires_run=True)
    def virials(self):
        """(*N_particles*, 6) `numpy.ndarray` of ``float``: Virial tensor \
        contribution from each particle :math:`[\\mathrm{energy}]`.

        The 6 elements form the upper-triangular virial tensor in the order:
        xx, xy, xz, yy, yz, zz.

        Attention:
            To improve performance `Force` objects only compute virials when
            needed. When not computed, `virials` is `None`. Virials are computed
            on every step when using a `md.methods.NPT` or `md.methods.NPH`
            integrator, on steps where a writer is triggered (such as
            `write.GSD` which may log pressure or virials), or when
            `Simulation.always_compute_pressure` is `True`.

        Attention:
            In MPI parallel execution, the array is available on rank 0 only.
            `virials` is `None` on ranks >= 1.
        """
        self._cpp_obj.compute(self._simulation.timestep)
        return self._cpp_obj.getVirials()

    @log(category="sequence", requires_run=True)
    def additional_virial(self):
        """(1, 6) `numpy.ndarray` of ``float``: Additional virial tensor \
        term not included in `virials` :math:`[\\mathrm{energy}]`."""
        self._cpp_obj.compute(self._simulation.timestep)
        virial = []
        for i in range(6):
            virial.append(self._cpp_obj.getExternalVirial(i))
        return numpy.array(virial, dtype=numpy.float64)


class Custom(Force):
    """Custom forces implemented in python.

    Derive a custom force class from `Custom`, and override the `set_forces`
    method to  compute forces on particles. Use have direct, zero-copy access to
    the C++ managed buffers via either the `cpu_local_force_arrays` or
    `gpu_local_force_arrays` property. Choose the property that corresponds to
    the device you wish to alter the data on. In addition to zero-copy access to
    force buffers, custom forces have access to the local snapshot API via the
    ``_state.cpu_local_snapshot`` or the ``_state.gpu_local_snapshot`` property.

    See Also:
      See the documentation in `hoomd.State` for more information on the local
      snapshot API.

    Examples::

        class MyCustomForce(hoomd.force.Custom):
            def __init__(self):
                super().__init__()

            def set_forces(self, timestep):
                with self.cpu_local_force_arrays as arrays:
                    arrays.force[:] = -5
                    arrays.torque[:] = 3
                    arrays.potential_energy[:] = 27
                    arrays.virial[:] = np.arange(6)[None, :]

    In addition, since data is MPI rank-local, there may be ghost particle data
    associated with each rank. To access this read-only ghost data, access the
    property name with either the prefix ``ghost_`` of the suffix
    ``_with_ghost``.

    Examples::

        class MyCustomForce(hoomd.force.Custom):
            def __init__(self):
                super().__init__()

            def set_forces(self, timestep):
                with self.cpu_local_force_arrays as arrays:
                    # access only the ghost particle forces
                    ghost_force_data = arrays.ghost_force

                    # access torque data on this rank and ghost torque data
                    torque_data = arrays.torque_with_ghost

    Note:
        When accessing the local force arrays, always use a context manager.

    Note:
        The shape of the exposed arrays cannot change while in the context
        manager.

    Note:
        All force data buffers are MPI rank local, so in simulations with MPI,
        only the data for a single rank is available.

    Note:
        Access to the force buffers is constant (O(1)) time.
    """

    def __init__(self):
        self._in_context_manager = False
        self._state = None  # to be set on attaching

    def _attach(self):
        self._state = self._simulation.state
        self._cpp_obj = _md.CustomForceCompute(self._state._cpp_sys_def,
                                               self.set_forces)
        super()._attach()

    @property
    def cpu_local_force_arrays(self):
        """hoomd.md.data.ForceLocalAccess: Expose force arrays on the CPU.

        Provides direct access to the force, potential energy, torque, and
        virial data of the particles in the system on the cpu through a context
        manager. All data is MPI rank-local.

        The `hoomd.md.data.ForceLocalAccess` object returned by this property
        has four arrays through which one can modify the force data:

        Examples::

            with self.cpu_local_force_arrays as arrays:
                arrays.force[:] = ...
                arrays.potential_energy[:] = ...
                arrays.torque[:] = ...
                arrays.virial[:] = ...
        """
        if self._in_context_manager:
            raise RuntimeError("Cannot enter cpu_local_force_arrays context "
                               "manager inside another local_force_arrays "
                               "context manager")
        return hoomd.md.data.ForceLocalAccess(self)

    @property
    def gpu_local_force_arrays(self):
        """hoomd.md.data.ForceLocalAccessGPU: Expose force arrays on the GPU.

        Provides direct access to the force, potential energy, torque, and
        virial data of the particles in the system on the gpu through a context
        manager. All data is MPI rank-local.

        The `hoomd.md.data.ForceLocalAccessGPU` object returned by this property
        has four arrays through which one can modify the force data:

        Examples::

            with self.gpu_local_force_arrays as arrays:
                arrays.force[:] = ...
                arrays.potential_energy[:] = ...
                arrays.torque[:] = ...
                arrays.virial[:] = ...

        Note:
            GPU local force data is not available if the chosen device for the
            simulation is `hoomd.device.CPU`.
        """
        if not isinstance(self._simulation.device, hoomd.device.GPU):
            raise RuntimeError(
                "Cannot access gpu_local_force_arrays without a GPU device")
        if self._in_context_manager:
            raise RuntimeError(
                "Cannot enter gpu_local_force_arrays context manager inside "
                "another local_force_arrays context manager")
        return hoomd.md.data.ForceLocalAccessGPU(self)

    @abstractmethod
    def set_forces(self, timestep):
        """Set the forces in the simulation loop.

        Args:
            timestep (int): The current timestep in the simulation.
        """
        pass


class Active(Force):
    r"""Active force.

    Args:
        filter (:py:mod:`hoomd.filter`): Subset of particles on which to apply
            active forces.

    :py:class:`Active` specifies that an active force should be added to
    particles selected by the filter.  particles.  Obeys :math:`\delta {\bf r}_i
    = \delta t v_0 \hat{p}_i`, where :math:`v_0` is the active velocity. In 2D
    :math:`\hat{p}_i = (\cos \theta_i, \sin \theta_i)` is the active force
    vector for particle :math:`i`.  The active force and the active torque
    vectors in the particle frame stay constant during the simulation. Hence,
    the active forces in the system frame are composed of the forces in particle
    frame and the current orientation of the particle.

    Note:
        To introduce rotational diffusion to the particle orientations, use
        `create_diffusion_updater`.

        .. seealso::

            `hoomd.md.update.ActiveRotationalDiffusion`

    Examples::

        all = hoomd.filter.All()
        active = hoomd.md.force.Active(
            filter=hoomd.filter.All()
            )
        active.active_force['A','B'] = (1,0,0)
        active.active_torque['A','B'] = (0,0,0)
        rotational_diffusion_updater = active.create_diffusion_updater(
            trigger=10)
        sim.operations += rotational_diffusion_updater

    Attributes:
        filter (:py:mod:`hoomd.filter`): Subset of particles on which to apply
            active forces.

    .. py:attribute:: active_force

        Active force vector in the local reference frame of the particle
        :math:`[\mathrm{force}]`.  It is defined per particle type and stays
        constant during the simulation.

        Type: `TypeParameter` [``particle_type``, `tuple` [`float`, `float`,
        `float`]]

    .. py:attribute:: active_torque

        Active torque vector in the local reference frame of the particle
        :math:`[\mathrm{force} \cdot \mathrm{length}]`. It is defined per
        particle type and stays constant during the simulation.

        Type: `TypeParameter` [``particle_type``, `tuple` [`float`, `float`,
        `float`]]
    """

    def __init__(self, filter):
        # store metadata
        param_dict = ParameterDict(filter=ParticleFilter)
        param_dict["filter"] = filter
        # set defaults
        self._param_dict.update(param_dict)

        active_force = TypeParameter(
            "active_force",
            type_kind="particle_types",
            param_dict=TypeParameterDict((1.0, 0.0, 0.0), len_keys=1),
        )
        active_torque = TypeParameter(
            "active_torque",
            type_kind="particle_types",
            param_dict=TypeParameterDict((0.0, 0.0, 0.0), len_keys=1),
        )

        self._extend_typeparam([active_force, active_torque])

    def _add(self, simulation):
        """Add the operation to a simulation.

        Active forces use RNGs. Warn the user if they did not set the seed.
        """
        if isinstance(simulation, hoomd.Simulation):
            simulation._warn_if_seed_unset()

        super()._add(simulation)

    def _attach(self):

        # initialize the reflected c++ class
        sim = self._simulation

        if isinstance(sim.device, hoomd.device.CPU):
            my_class = _md.ActiveForceCompute
        else:
            my_class = _md.ActiveForceComputeGPU

        self._cpp_obj = my_class(sim.state._cpp_sys_def,
                                 sim.state._get_group(self.filter))

        # Attach param_dict and typeparam_dict
        super()._attach()

    def create_diffusion_updater(self, trigger, rotational_diffusion):
        """Create a rotational diffusion updater for this active force.

        Args:
            trigger (hoomd.trigger.Trigger): Select the timesteps to update
                rotational diffusion.
            rotational_diffusion (hoomd.variant.Variant or float): The
                rotational diffusion as a function of time or a constant.

        Returns:
            hoomd.md.update.ActiveRotationalDiffusion:
                The rotational diffusion updater.
        """
        return hoomd.md.update.ActiveRotationalDiffusion(
            trigger, self, rotational_diffusion)


class ActiveOnManifold(Active):
    r"""Active force on a manifold.

    Args:
        filter (`hoomd.filter.ParticleFilter`): Subset of particles on which to
            apply active forces.
        manifold_constraint (`hoomd.md.manifold.Manifold`): Manifold constraint.

    :py:class:`ActiveOnManifold` specifies that a constrained active force
    should be added to particles selected by the filter similar to
    :py:class:`Active`. The active force vector :math:`\hat{p}_i` is restricted
    to the local tangent plane of the manifold constraint at point :math:`{\bf
    r}_i`. For more information see :py:class:`Active`.

    Hint:
        Use `ActiveOnManifold` with a `md.methods.rattle` integration method
        with the same manifold constraint.

    Examples::

        all = filter.All()
        sphere = hoomd.md.manifold.Sphere(r=10)
        active = hoomd.md.force.ActiveOnManifold(
            filter=hoomd.filter.All(), rotation_diff=0.01,
            manifold_constraint = sphere
            )
        active.active_force['A','B'] = (1,0,0)
        active.active_torque['A','B'] = (0,0,0)

    Attributes:
        filter (`hoomd.filter.ParticleFilter`): Subset of particles on which to
            apply active forces.
        manifold_constraint (`hoomd.md.manifold.Manifold`): Manifold constraint.

    .. py:attribute:: active_force

        Active force vector in the local reference frame of the particle
        :math:`[\mathrm{force}]`.  It is defined per particle type and stays
        constant during the simulation.

        Type: `TypeParameter` [``particle_type``, `tuple` [`float`, `float`,
        `float`]]

    .. py:attribute:: active_torque

        Active torque vector in local reference frame of the particle
        :math:`[\mathrm{force} \cdot \mathrm{length}]`. It is defined per
        particle type and stays constant during the simulation.

        Type: `TypeParameter` [``particle_type``, `tuple` [`float`, `float`,
        `float`]]
    """

    def __init__(self, filter, manifold_constraint):
        # store metadata
        super().__init__(filter)
        param_dict = ParameterDict(
            manifold_constraint=OnlyTypes(Manifold, allow_none=False))
        param_dict["manifold_constraint"] = manifold_constraint
        self._param_dict.update(param_dict)

    def _getattr_param(self, attr):
        if self._attached:
            if attr == "manifold_constraint":
                return self._param_dict["manifold_constraint"]
            parameter = getattr(self._cpp_obj, attr)
            return parameter
        else:
            return self._param_dict[attr]

    def _setattr_param(self, attr, value):
        if attr == "manifold_constraint":
            raise AttributeError(
                "Cannot set manifold_constraint after construction.")
        super()._setattr_param(attr, value)

    def _attach(self):

        # initialize the reflected c++ class
        sim = self._simulation

        if not self.manifold_constraint._attached:
            self.manifold_constraint._attach()

        base_class_str = 'ActiveForceConstraintCompute'
        base_class_str += self.manifold_constraint.__class__.__name__
        if isinstance(sim.device, hoomd.device.GPU):
            base_class_str += "GPU"
        self._cpp_obj = getattr(
            _md, base_class_str)(sim.state._cpp_sys_def,
                                 sim.state._get_group(self.filter),
                                 self.manifold_constraint._cpp_obj)

        # Attach param_dict and typeparam_dict
        super()._attach()
