# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: joaander / All Developers are free to add commands for new features

""" Compute system properties """

from hoomd import _hoomd;
from hoomd.md import _md
from hoomd.operation import _Compute
from hoomd.logging import log
import hoomd;
import sys;

class _Thermo(_Compute):

    def __init__(self, filter):
        self._filter = filter


class ThermodynamicQuantities(_Thermo):
    """ Compute thermodynamic properties of a group of particles.

    Args:
        filter (``hoomd.filter``): Particle filter to compute thermodynamic properties for.

    :py:class:`ThermodynamicQuantities` acts on a given group of particles and calculates thermodynamic properties of
    those particles when requested. All specified :py:class:`ThermodynamicQuantities` objects can be added to a logger
    for logging during a simulation, see :py:class:`hoomd.logging.Logger` for more details.

    Examples::

        f = filter.Type('A')
        compute.ThermodynamicQuantities(filter=f)
    """

    def __init__(self, filter):
        super().__init__(filter)

    def _attach(self):
        if isinstance(self._simulation.device, hoomd.device.CPU):
            thermo_cls = _md.ComputeThermo
        else:
            thermo_cls = _md.ComputeThermoGPU
        group = self._simulation.state.get_group(self._filter)
        self._cpp_obj = thermo_cls(self._simulation.state._cpp_sys_def, group, "")
        super()._attach()

    @log
    def kinetic_temperature(self):
        """
        :math:`kT_k`, instantaneous thermal energy of the group (in energy units).

        Calculated as:

          .. math::

            kT_k = 2 \\cdot \\frac{K}{N_{\\mathrm{dof}}}
        """
        if self._attached:
            self._cpp_obj.compute(self._simulation.timestep)
            return self._cpp_obj.kinetic_temperature
        else:
            return None

    @log
    def pressure(self):
        """
        :math:`P`, instantaneous pressure of the group (in pressure units).

        Calculated as:

        .. math::

            P = \\frac{ 2 \\cdot K_{\\mathrm{trans}} + W }{D \\cdot V},

        where :math:`D` is the dimensionality of the system, :math:`V` is the total volume of the simulation box (or area in 2D), and :math:`W` is calculated as:

        .. math::

            W = \\frac{1}{2} \\sum_{i \\in \\mathrm{filter}} \\sum_{j} \\vec{F}_{ij} \\cdot \\vec{r_{ij}} + \\sum_{k} \\vec{F}_{k} \\cdot \\vec{r_{k}},

        where :math:`i` and :math:`j` are particle tags, :math:`\\vec{F}_{ij}` are pairwise forces between particles and :math:`\\vec{F}_k` are forces due to explicit constraints, implicit rigid body constraints, external walls, and fields.
        """
        if self._attached:
            self._cpp_obj.compute(self._simulation.timestep)
            return self._cpp_obj.pressure
        else:
            return None

    @log(flag='sequence')
    def pressure_tensor(self):
        """
        (:math:`P_{xx}`, :math:`P_{xy}`, :math:`P_{xz}`, :math:`P_{yy}`, :math:`P_{yz}`, :math:`P_{zz}`).

        Instantaneous pressure tensor of the group (in pressure units), calculated as:

          .. math::

              P_{ij} = \\left[  \\sum_{k \\in \\mathrm{filter}} m_k v_{k,i} v_{k,j} +
                               \\sum_{k \\in \\mathrm{filter}} \\sum_{l} \\frac{1}{2} \\left(\\vec{r}_{kl,i} \\vec{F}_{kl,j} + \\vec{r}_{kl,j} \\vec{F}_{kl, i} \\right) \\right]/V

        where :math:`V` is the total simulation box volume (or area in 2D).
        """
        if self._attached:
            self._cpp_obj.compute(self._simulation.timestep)
            return self._cpp_obj.pressure_tensor
        else:
            return None

    @log
    def kinetic_energy(self):
        """
        :math:`K`, total kinetic energy of all particles in the group (in energy units).

        .. math::

            K = K_{\\mathrm{rot}} + K_{\\mathrm{trans}}

        """
        if self._attached:
            self._cpp_obj.compute(self._simulation.timestep)
            return self._cpp_obj.kinetic_energy
        else:
            return None

    @log
    def translational_kinetic_energy(self):
        """
        :math:`K_{\\mathrm{trans}}`, translational kinetic energy of all particles in the group (in energy units).

        .. math::

            K_{\\mathrm{trans}} = \\frac{1}{2}\\sum_{i \\in \\mathrm{filter}} m_i|\\vec{v}_i|^2

        """
        if self._attached:
            self._cpp_obj.compute(self._simulation.timestep)
            return self._cpp_obj.translational_kinetic_energy
        else:
            return None

    @log
    def rotational_kinetic_energy(self):
        """
        :math:`K_{\\mathrm{rot}}`, rotational kinetic energy of all particles in the group (in energy units).

        Calculated as:

        .. math::

            K_{\\mathrm{rot}} = \\frac{1}{2} \\sum_{i \\in \\mathrm{filter}} \\frac{L_{x,i}^2}{I_{x,i}} + \\frac{L_{y,i}^2}{I_{y,i}} + \\frac{L_{z,i}^2}{I_{z,i}},

        where :math:`I` is the moment of inertia and :math:`L` is the angular momentum in the (diagonal) reference frame of the particle.

        """
        if self._attached:
            self._cpp_obj.compute(self._simulation.timestep)
            return self._cpp_obj.rotational_kinetic_energy
        else:
            return None

    @log
    def potential_energy(self):
        """
        :math:`U`, potential energy that the group contributes to the entire system (in energy units).

        The potential energy is calculated as a sum of per-particle energy contributions:

        .. math::

            U = \\sum_{i \\in \\mathrm{filter}} U_i,

        where :math:`U_i` is defined as:

        .. math::

            U_i = U_{\\mathrm{pair}, i} + U_{\\mathrm{bond}, i} + U_{\\mathrm{angle}, i} + U_{\\mathrm{dihedral}, i} + U_{\\mathrm{improper}, i} + U_{\\mathrm{external}, i} + U_{\\mathrm{other}, i}

        and each term on the RHS is calculated as:

        .. math::

            U_{\\mathrm{pair}, i} &= \\frac{1}{2} \\sum_j V_{\\mathrm{pair}, ij}

            U_{\\mathrm{bond}, i} &= \\frac{1}{2} \\sum_{(j, k) \\in \\mathrm{bonds}} V_{\\mathrm{bond}, jk}

            U_{\\mathrm{angle}, i} &= \\frac{1}{3} \\sum_{(j, k, l) \\in \\mathrm{angles}} V_{\\mathrm{angle}, jkl}

            U_{\\mathrm{dihedral}, i} &= \\frac{1}{4} \\sum_{(j, k, l, m) \\in \\mathrm{dihedrals}} V_{\\mathrm{dihedral}, jklm}

            U_{\\mathrm{improper}, i} &= \\frac{1}{4} \\sum_{(j, k, l, m) \\in \\mathrm{impropers}} V_{\\mathrm{improper}, jklm}

        In each summation above, the indices go over all particles and we only use terms where one of the summation indices (:math:`j`, :math:`k`, :math:`l`, or :math:`m`)
        is equal to :math:`i`. External and other potentials are summed similar to the other terms using per-particle contributions.
        """
        if self._attached:
            self._cpp_obj.compute(self._simulation.timestep)
            return self._cpp_obj.potential_energy
        else:
            return None

    @log
    def degrees_of_freedom(self):
        """
        :math:`N_{\\mathrm{dof}}`, number of degrees of freedom given to the group by its integration method.

        Calculated as:

        .. math::

            N_{\\mathrm{dof}} = N_{\\mathrm{dof, trans}} + N_{\\mathrm{dof, rot}}

        """
        if self._attached:
            return self._cpp_obj.degrees_of_freedom
        else:
            return None

    @log
    def translational_degrees_of_freedom(self):
        """
        :math:`N_{\\mathrm{dof, trans}}`, number of translational degrees of freedom given to the group by its integration method.

        When using a single integration method that is momentum conserving and operates on all particles, :math:`N_{\\mathrm{dof, trans}} = DN - D - N_{constraints}`, where :math:`D` is the dimensionality of the system.

        Note:
            The removal of :math:`D` degrees of freedom accounts for the fixed center of mass in using periodic boundary conditions. When the *filter* in :py:class:`ThermodynamicQuantities` selects a subset of all particles, the removed degrees of freedom are spread proportionately.
        """
        if self._attached:
            return self._cpp_obj.translational_degrees_of_freedom
        else:
            return None

    @log
    def rotational_degrees_of_freedom(self):
        """ :math:`N_{\\mathrm{dof, rot}}`, number of rotational degrees of freedom given to the group by its integration method. """
        if self._attached:
            return self._cpp_obj.rotational_degrees_of_freedom
        else:
            return None

    @log
    def num_particles(self):
        """ :math:`N`, number of particles in the group. """
        if self._attached:
            return self._cpp_obj.num_particles
        else:
            return None


class thermoHMA(_Compute):
    R""" Compute HMA thermodynamic properties of a group of particles.

    Args:
        group (``hoomd.group``): Group to compute thermodynamic properties for.
        temperature (float): Temperature
        harmonicPressure (float): Harmonic contribution to the pressure.  If ommitted, the HMA pressure can still be
            computed, but will be similar in precision to the conventional pressure.

    :py:class:`thermoHMA` acts on a given group of particles and calculates HMA (harmonically mapped
    averaging) properties of those particles when requested.  HMA computes properties more precisely (with less
    variance) for atomic crystals in NVT simulations.  The presence of diffusion (vanacy hopping, etc.) will prevent
    HMA from providing improvement.  HMA tracks displacements from the lattice positions, which are saved when the
    :py:class:`thermoHMA` is instantiated.

    The specified properties are available for logging via the ``hoomd.analyze.log`` command. Each one provides
    a set of quantities for logging, suffixed with *_groupname*, so that values for different groups are differentiated
    in the log file. The default :py:class:`thermoHMA` specified on the group of all particles has no suffix
    placed on its quantity names.

    The quantities provided are (where **groupname** is replaced with the name of the group):

    * **potential_energyHMA_groupname** - :math:`U` HMA potential energy that the group contributes to the entire
      system (in energy units)
    * **pressureHMA_groupname** - :math:`P` HMA pressure that the group contributes to the entire
      system (in pressure units)

    See Also:
        ``hoomd.analyze.log``.

    Examples::

        g = group.all()
        compute.thermoHMA(group=g, temperature=1.0)
    """

    def __init__(self, group, temperature, harmonicPressure=0):

        # initialize base class
        _compute.__init__(self);

        suffix = '';
        if group.name != 'all':
            suffix = '_' + group.name;

        # create the c++ mirror class
        if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
            self.cpp_compute = _hoomd.ComputeThermoHMA(hoomd.context.current.system_definition, group.cpp_group, temperature, harmonicPressure, suffix);
        else:
            self.cpp_compute = _hoomd.ComputeThermoHMAGPU(hoomd.context.current.system_definition, group.cpp_group, temperature, harmonicPressure, suffix);

        hoomd.context.current.system.addCompute(self.cpp_compute, self.compute_name);

        # save the group for later referencing
        self.group = group;

    def disable(self):
        R""" Disables the thermoHMA.

        Examples::

            my_thermo.disable()

        Executing the disable command will remove the thermoHMA compute from the system. Any ```hoomd.run``` command
        executed after disabling a thermoHMA compute will not be able to log computed values with ``hoomd.analyze.log``.

        A disabled thermoHMA compute can be re-enabled with :py:meth:`enable()`.
        """

        _compute.disable(self)

    def enable(self):
        R""" Enables the thermoHMA compute.

        Examples::

            my_thermo.enable()

        See ``disable``.
        """
        _compute.enable(self)
