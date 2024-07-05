# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Compute properties of molecular dynamics simulations.

The MD compute classes compute instantaneous properties of the simulation state
and provide results as loggable quantities for use with `hoomd.logging.Logger`
or by direct access via the Python API.
"""

from hoomd.md import _md
from hoomd.operation import Compute
from hoomd.data.parameterdicts import ParameterDict
from hoomd.logging import log
import hoomd


class ThermodynamicQuantities(Compute):
    """Compute thermodynamic properties of a subset of the system.

    Args:
        filter (`hoomd.filter`): Particle filter to compute thermodynamic
            properties for.

    `ThermodynamicQuantities` acts on a subset of particles in the system and
    calculates thermodynamic properties of those particles. Add a
    `ThermodynamicQuantities` instance to a logger to save these quantities to a
    file, see `hoomd.logging.Logger` for more details.

    Note:
        For compatibility with `hoomd.md.constrain.Rigid`,
        `ThermodynamicQuantities` performs all sums
        :math:`\\sum_{i \\in \\mathrm{filter}}` over free particles and
        rigid body centers - ignoring constituent particles to avoid double
        counting.

    Examples::

        f = filter.Type('A')
        compute.ThermodynamicQuantities(filter=f)
    """

    def __init__(self, filter):
        super().__init__()
        self._filter = filter

    def _attach_hook(self):
        if isinstance(self._simulation.device, hoomd.device.CPU):
            thermo_cls = _md.ComputeThermo
        else:
            thermo_cls = _md.ComputeThermoGPU
        group = self._simulation.state._get_group(self._filter)
        self._cpp_obj = thermo_cls(self._simulation.state._cpp_sys_def, group)

    @log(requires_run=True)
    def kinetic_temperature(self):
        """Instantaneous thermal energy :math:`kT_k` of the subset \
        :math:`[\\mathrm{energy}]`.

        .. math::

            kT_k = 2 \\cdot \\frac{K}{N_{\\mathrm{dof}}}
        """
        self._cpp_obj.compute(self._simulation.timestep)
        return self._cpp_obj.kinetic_temperature

    @log(requires_run=True)
    def pressure(self):
        """Instantaneous pressure :math:`P` of the subset \
        :math:`[\\mathrm{pressure}]`.

        .. math::

            P = \\frac{ 2 \\cdot K_{\\mathrm{translational}}
                + W_\\mathrm{isotropic} }{D \\cdot V},

        where :math:`D` is the dimensionality of the system, :math:`V` is the
        volume of the simulation box (or area in 2D), and
        :math:`W_\\mathrm{isotropic}` is the isotropic virial:

        .. math::

            W_\\mathrm{isotropic} = & \\left(
            W_{\\mathrm{net},\\mathrm{additional}}^{xx}
            + W_{\\mathrm{net},\\mathrm{additional}}^{yy}
            + W_{\\mathrm{net},\\mathrm{additional}}^{zz} \\right) \\\\
            + & \\sum_{i \\in \\mathrm{filter}}
            \\left( W_\\mathrm{{net},i}^{xx}
            + W_\\mathrm{{net},i}^{yy}
            + W_\\mathrm{{net},i}^{zz}
            \\right)

        where the net virial terms are computed by `hoomd.md.Integrator`
        over all of the forces in `hoomd.md.Integrator.forces` and
        :math:`W^{zz}=0` in 2D simulations.
        """
        self._cpp_obj.compute(self._simulation.timestep)
        return self._cpp_obj.pressure

    @log(category='sequence', requires_run=True)
    def pressure_tensor(self):
        """Instantaneous pressure tensor of the subset \
        :math:`[\\mathrm{pressure}]`.

        The six components of the pressure tensor are given in the order:
        (:math:`P^{xx}`, :math:`P^{xy}`, :math:`P^{xz}`, :math:`P^{yy}`,
        :math:`P^{yz}`, :math:`P^{zz}`):

        .. math::

            P^{kl} = \\frac{1}{V} \\left(
            W_{\\mathrm{net},\\mathrm{additional}}^{kl}
            + \\sum_{i \\in \\mathrm{filter}} m_i
            \\cdot v_i^k \\cdot v_i^l +
            W_{\\mathrm{net},i}^{kl}
            \\right),

        where the net virial terms are computed by `hoomd.md.Integrator` over
        all of the forces in `hoomd.md.Integrator.forces`, :math:`v_i^k` is the
        k-th component of the velocity of particle :math:`i` and :math:`V` is
        the total simulation box volume (or area in 2D).
        """
        self._cpp_obj.compute(self._simulation.timestep)
        return self._cpp_obj.pressure_tensor

    @log(requires_run=True)
    def kinetic_energy(self):
        """Total kinetic energy :math:`K` of the subset \
        :math:`[\\mathrm{energy}]`.

        .. math::

            K = K_\\mathrm{rotational} + K_\\mathrm{translational}
        """
        self._cpp_obj.compute(self._simulation.timestep)
        return self._cpp_obj.kinetic_energy

    @log(requires_run=True)
    def translational_kinetic_energy(self):
        """Translational kinetic energy :math:`K_{\\mathrm{translational}}` \
        of the subset :math:`[\\mathrm{energy}]`.

        .. math::

            K_\\mathrm{translational} = \\frac{1}{2}
            \\sum_{i \\in \\mathrm{filter}} m_i v_i^2
        """
        self._cpp_obj.compute(self._simulation.timestep)
        return self._cpp_obj.translational_kinetic_energy

    @log(requires_run=True)
    def rotational_kinetic_energy(self):
        """Rotational kinetic energy :math:`K_\\mathrm{rotational}` of  \
        the subset :math:`[\\mathrm{energy}]`.

        .. math::

            K_\\mathrm{rotational,d} =
            \\frac{1}{2}
            \\sum_{i \\in \\mathrm{filter}}
            \\begin{cases}
            \\frac{L_{d,i}^2}{I_{d,i}} & I^d_i > 0 \\\\
            0 & I^d_i = 0
            \\end{cases}

            K_\\mathrm{rotational} = K_\\mathrm{rotational,x} +
            K_\\mathrm{rotational,y} + K_\\mathrm{rotational,z}

        :math:`I` is the moment of inertia and :math:`L` is the angular
        momentum in the (diagonal) reference frame of the particle.
        """
        self._cpp_obj.compute(self._simulation.timestep)
        return self._cpp_obj.rotational_kinetic_energy

    @log(requires_run=True)
    def potential_energy(self):
        """Potential energy :math:`U` that the subset contributes to the \
        system :math:`[\\mathrm{energy}]`.

        .. math::

            U = U_{\\mathrm{net},\\mathrm{additional}}
                + \\sum_{i \\in \\mathrm{filter}} U_{\\mathrm{net},i},

        where the net energy terms are computed by `hoomd.md.Integrator` over
        all of the forces in `hoomd.md.Integrator.forces`.
        """
        self._cpp_obj.compute(self._simulation.timestep)
        return self._cpp_obj.potential_energy

    @log(requires_run=True)
    def degrees_of_freedom(self):
        r"""Number of degrees of freedom in the subset :math:`N_{\mathrm{dof}}`.

        .. math::

            N_\mathrm{dof} = N_\mathrm{dof, translational}
                             + N_\mathrm{dof, rotational}

        See Also:
            `hoomd.State.update_group_dof` describes when
            :math:`N_{\mathrm{dof}}` is updated.
        """
        return self._cpp_obj.degrees_of_freedom

    @log(requires_run=True)
    def translational_degrees_of_freedom(self):
        """Number of translational degrees of freedom in the subset \
        :math:`N_{\\mathrm{dof, translational}}`.

        When using a single integration method on all particles that is momentum
        conserving, the center of mass motion is conserved and the number of
        translational degrees of freedom is:

        .. math::

            N_\\mathrm{dof, translational} = DN
            - D\\frac{N}{N_\\mathrm{particles}}
            - N_\\mathrm{constraints}(\\mathrm{filter})

        where :math:`D` is the dimensionality of the system and
        :math:`N_\\mathrm{constraints}(\\mathrm{filter})` is the number of
        degrees of freedom removed by constraints (`hoomd.md.constrain`) in the
        subset. The fraction :math:`\\frac{N}{N_\\mathrm{particles}}`
        distributes the momentum conservation constraint evenly when
        `ThermodynamicQuantities` is applied to multiple subsets.

        Note:
            When using rigid bodies (`hoomd.md.constrain.Rigid`), :math:`N`
            is the number of rigid body centers plus free particles selected
            by the filter and :math:`N_\\mathrm{particles}` is total number
            of rigid body centers plus free particles in the whole system.

            When `hoomd.md.Integrator.rigid` is not set, :math:`N` is the
            total number of particles selected by the filter and
            :math:`N_\\mathrm{particles}` is the total number of particles in
            the system, regardless of their ``body`` value.

        When using multiple integration methods, a single integration method
        on fewer than all particles, or a single integration method that is
        not momentum conserving, `hoomd.md.Integrator` assumes that linear
        momentum is not conserved and counts the center of mass motion in the
        degrees of freedom:

        .. math::

            N_{\\mathrm{dof, translational}} = DN
            - N_\\mathrm{constraints}(\\mathrm{filter})
        """
        return self._cpp_obj.translational_degrees_of_freedom

    @log(requires_run=True)
    def rotational_degrees_of_freedom(self):
        """Number of rotational degrees of freedom in the subset \
        :math:`N_\\mathrm{dof, rotational}`.

        Integration methods (`hoomd.md.methods`) determine the number of degrees
        of freedom they give to each particle. Each integration method operates
        on a subset of the system :math:`\\mathrm{filter}_m` that may be
        distinct from the subset from the subset given to
        `ThermodynamicQuantities`.

        When `hoomd.md.Integrator.integrate_rotational_dof` is ``False``,
        :math:`N_\\mathrm{dof, rotational} = 0`. When it is ``True``, the
        given degrees of freedom depend on the dimensionality of the system.

        In 2D:

        .. math::

            N_\\mathrm{dof, rotational} =
            \\sum_{m \\in \\mathrm{methods}} \\;
            \\sum_{i \\in \\mathrm{filter} \\cup \\mathrm{filter}_m}
            \\left[ I_{z,i} > 0 \\right]

        where :math:`I` is the particles's moment of inertia.

        In 3D:

        .. math::

            N_\\mathrm{dof, rotational} =
            \\sum_{m \\in \\mathrm{methods}} \\;
            \\sum_{i \\in \\mathrm{filter} \\cup \\mathrm{filter}_m}
            \\left[ I_{x,i} > 0 \\right]
            + \\left[ I_{y,i} > 0 \\right]
            + \\left[ I_{z,i} > 0 \\right]

        See Also:
            `hoomd.State.update_group_dof` describes when
            :math:`N_{\\mathrm{dof, rotational}}` is updated.
        """
        return self._cpp_obj.rotational_degrees_of_freedom

    @log(requires_run=True)
    def num_particles(self):
        """Number of particles :math:`N` in the subset."""
        return self._cpp_obj.num_particles

    @log(requires_run=True)
    def volume(self):
        """Volume :math:`V` of the simulation box (area in 2D) \
        :math:`[\\mathrm{length}^{D}]`."""
        return self._cpp_obj.volume


class HarmonicAveragedThermodynamicQuantities(Compute):
    """Compute harmonic averaged thermodynamic properties of particles.

    Args:
        filter (hoomd.filter.filter_like): Particle filter to compute
            thermodynamic properties for.
        kT (float): Temperature of the system :math:`[\\mathrm{energy}]`.
        harmonic_pressure (float): Harmonic contribution to the pressure
            :math:`[\\mathrm{pressure}]`. If omitted, the HMA pressure can
            still be computed, but will be similar in precision to
            the conventional pressure.

    `HarmonicAveragedThermodynamicQuantities` acts on a given subset of
    particles and calculates harmonically mapped average (HMA) properties of
    those particles when requested. HMA computes properties more precisely (with
    less variance) for atomic crystals in NVT simulations.  The presence of
    diffusion (vacancy hopping, etc.) will prevent HMA from providing
    improvement.  HMA tracks displacements from the lattice positions, which are
    saved either during first call to `Simulation.run` or when the compute is
    first added to the simulation, whichever occurs last.

    Note:
        `HarmonicAveragedThermodynamicQuantities` is an implementation of the
        methods section of Sabry G. Moustafa, Andrew J. Schultz, and David A.
        Kofke. (2015).  "Very fast averaging of thermal properties of crystals
        by molecular simulation". Phys. Rev. E 92, 043303
        doi:10.1103/PhysRevE.92.043303

    Examples::

        hma = hoomd.compute.HarmonicAveragedThermodynamicQuantities(
            filter=hoomd.filter.Type('A'), kT=1.0)


    Attributes:
        filter (hoomd.filter.filter_like): Subset of particles compute
            thermodynamic properties for.

        kT (float): Temperature of the system
            :math:`[\\mathrm{energy}]`.

        harmonic_pressure (float): Harmonic contribution to the pressure
            :math:`[\\mathrm{pressure}]`.
    """

    def __init__(self, filter, kT, harmonic_pressure=0):

        # store metadata
        param_dict = ParameterDict(kT=float(kT),
                                   harmonic_pressure=float(harmonic_pressure))
        # set defaults
        self._param_dict.update(param_dict)

        self._filter = filter
        # initialize base class
        super().__init__()

    def _attach_hook(self):
        if isinstance(self._simulation.device, hoomd.device.CPU):
            thermoHMA_cls = _md.ComputeThermoHMA
        else:
            thermoHMA_cls = _md.ComputeThermoHMAGPU
        group = self._simulation.state._get_group(self._filter)
        self._cpp_obj = thermoHMA_cls(self._simulation.state._cpp_sys_def,
                                      group, self.kT, self.harmonic_pressure)

    @log(requires_run=True)
    def potential_energy(self):
        """Average potential energy :math:`[\\mathrm{energy}]`."""
        self._cpp_obj.compute(self._simulation.timestep)
        return self._cpp_obj.potential_energy

    @log(requires_run=True)
    def pressure(self):
        """Average pressure :math:`[\\mathrm{pressure}]`."""
        self._cpp_obj.compute(self._simulation.timestep)
        return self._cpp_obj.pressure
