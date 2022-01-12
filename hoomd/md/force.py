# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Apply forces to particles."""

import hoomd
from hoomd import _hoomd
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


class constant(Force):  # noqa - this will be renamed when it is ported to v3
    R"""Constant force.

    Args:
        fvec (tuple): force vector :math:`[force]`
        tvec (tuple): torque vector :math:`[force \cdot length]`
        fx (float): x component of force, retained for backwards compatibility
          :math:`[\mathrm{force}]`
        fy (float): y component of force, retained for backwards compatibility
          :math:`[\mathrm{force}]`
        fz (float): z component of force, retained for backwards compatibility
          :math:`[\mathrm{force}]`
        group (``hoomd.group``): Group for which the force will be set.
        callback (`callable`): A python callback invoked every time the forces
            are computed

    :py:class:`constant` specifies that a constant force should be added to
    every particle in the simulation or optionally to all particles in a group.

    Note:
        Forces are kept constant during the simulation. If a callback should
        re-compute particle forces every time step, it needs to overwrite the
        old forces of **all** particles with new values.

    Note:
        Per-particle forces take precedence over a particle group, which takes
        precedence over constant forces for all particles.

    Examples::

        force.constant(fx=1.0, fy=0.5, fz=0.25)
        const = force.constant(fvec=(0.4,1.0,0.5))
        const = force.constant(fvec=(0.4,1.0,0.5),group=fluid)
        const = force.constant(fvec=(0.4,1.0,0.5), tvec=(0,0,1) ,group=fluid)

        def updateForces(timestep):
            global const
            const.setForce(tag=1, fvec=(1.0*timestep,2.0*timestep,3.0*timestep))
        const = force.constant(callback=updateForces)
    """

    def __init__(
        self,
        fx=None,
        fy=None,
        fz=None,
        fvec=None,
        tvec=None,
        group=None,
        callback=None,
    ):

        if (fx is not None) and (fy is not None) and (fz is not None):
            self.fvec = (fx, fy, fz)
        elif fvec is not None:
            self.fvec = fvec
        else:
            self.fvec = (0, 0, 0)

        if tvec is not None:
            self.tvec = tvec
        else:
            self.tvec = (0, 0, 0)

        if (self.fvec == (0, 0, 0)) and (self.tvec == (0, 0, 0)
                                         and callback is None):
            hoomd.context.current.device.cpp_msg.warning(
                "The constant force specified has no non-zero components\n")

        # initialize the base class
        Force.__init__(self)

        # create the c++ mirror class
        if group is not None:
            self.cppForce = _hoomd.ConstForceCompute(
                hoomd.context.current.system_definition,
                group.cpp_group,
                self.fvec[0],
                self.fvec[1],
                self.fvec[2],
                self.tvec[0],
                self.tvec[1],
                self.tvec[2],
            )
        else:
            self.cppForce = _hoomd.ConstForceCompute(
                hoomd.context.current.system_definition,
                self.fvec[0],
                self.fvec[1],
                self.fvec[2],
                self.tvec[0],
                self.tvec[1],
                self.tvec[2],
            )

        if callback is not None:
            self.cppForce.setCallback(callback)

        hoomd.context.current.system.addCompute(self.cppForce, self.force_name)

    R""" Change the value of the constant force.

    Args:
        fx (float) New x-component of the force :math:`[\mathrm{force}]`
        fy (float) New y-component of the force :math:`[\mathrm{force}]`
        fz (float) New z-component of the force :math:`[\mathrm{force}]`
        fvec (tuple) New force vector
        tvec (tuple) New torque vector
        group Group for which the force will be set
        tag (int) Particle tag for which the force will be set
            .. versionadded:: 2.3

     Using setForce() requires that you saved the created constant force in a
     variable. i.e.

     Examples:
        const = force.constant(fx=0.4, fy=1.0, fz=0.5)

        const.setForce(fx=0.2, fy=0.1, fz=-0.5)
        const.setForce(fx=0.2, fy=0.1, fz=-0.5, group=fluid)
        const.setForce(fvec=(0.2,0.1,-0.5), tvec=(0,0,1), group=fluid)
    """

    def setForce(  # noqa - this will be documented when it is ported to v3
        self,
        fx=None,
        fy=None,
        fz=None,
        fvec=None,
        tvec=None,
        group=None,
        tag=None,
    ):

        if (fx is not None) and (fy is not None) and (fx is not None):
            self.fvec = (fx, fy, fz)
        elif fvec is not None:
            self.fvec = fvec
        else:
            self.fvec = (0, 0, 0)

        if tvec is not None:
            self.tvec = tvec
        else:
            self.tvec = (0, 0, 0)

        if (fvec == (0, 0, 0)) and (tvec == (0, 0, 0)):
            hoomd.context.current.device.cpp_msg.warning(
                "You are setting the constant force to have no non-zero "
                "components\n")

        self.check_initialization()
        if group is not None:
            self.cppForce.setGroupForce(
                group.cpp_group,
                self.fvec[0],
                self.fvec[1],
                self.fvec[2],
                self.tvec[0],
                self.tvec[1],
                self.tvec[2],
            )
        elif tag is not None:
            self.cppForce.setParticleForce(
                tag,
                self.fvec[0],
                self.fvec[1],
                self.fvec[2],
                self.tvec[0],
                self.tvec[1],
                self.tvec[2],
            )
        else:
            self.cppForce.setForce(
                self.fvec[0],
                self.fvec[1],
                self.fvec[2],
                self.tvec[0],
                self.tvec[1],
                self.tvec[2],
            )

    R""" Set a python callback to be called before the force is evaluated

    Args:
        callback (`callable`) The callback function

     Examples:
        const = force.constant(fx=0.4, fy=1.0, fz=0.5)

        def updateForces(timestep):
            global const
            const.setForce(tag=1, fvec=(1.0*timestep,2.0*timestep,3.0*timestep))

        const.set_callback(updateForces)
        run(100)

        # Reset the callback
        const.set_callback(None)
    """

    def set_callback(self, callback=None):  # noqa - will be ported to v3
        self.cppForce.setCallback(callback)

    # there are no coeffs to update in the constant force compute
    def update_coeffs(self):  # noqa - will be ported to v3
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
