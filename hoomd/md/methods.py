# coding: utf-8

# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: joaander / All Developers are free to add commands for new features


from hoomd import _hoomd
from hoomd.md import _md
import hoomd
from hoomd.operation import _Operation, NotAttachedError
from hoomd.parameterdicts import ParameterDict, TypeParameterDict
from hoomd.filter import _ParticleFilter
from hoomd.typeparam import TypeParameter
from hoomd.typeconverter import OnlyType, OnlyIf, to_type_converter
from hoomd.variant import Variant
from hoomd.typeconverter import OnlyFrom
import copy
from collections.abc import Sequence

def none_or(type_):
    def None_or_type(value):
        if None or isinstance(value, type_):
            return value
        else:
            try:
                return type_(value)
            except Exception:
                raise ValueError("Value {} of type {} could not be made type "
                                 "{}.".format(value, type(value), type_))


class _Method(_Operation):
    pass


class NVT(_Method):
    R""" NVT Integration via the Nosé-Hoover thermostat.

    Args:
        filter (:py:mod:`hoomd.filter`): Subset of particles on which to apply this
            method.
        kT (:py:mod:`hoomd.variant` or :py:obj:`float`): Temperature set point
            for the Nosé-Hoover thermostat. (in energy units).
        tau (float): Coupling constant for the Nosé-Hoover thermostat. (in time
            units).

    :py:class:`NVT` performs constant volume, constant temperature simulations
    using the Nosé-Hoover thermostat, using the MTK equations described in Refs.
    `G. J. Martyna, D. J. Tobias, M. L. Klein  1994
    <http://dx.doi.org/10.1063/1.467468>`_ and `J. Cao, G. J. Martyna 1996
    <http://dx.doi.org/10.1063/1.470959>`_.

    :py:class:`NVT` is an integration method. It must be used in connection with
    ``mode_standard``.

    :py:class:`NVT` uses the proper number of degrees of freedom to compute the
    temperature of the system in both 2 and 3 dimensional systems, as long as
    the number of dimensions is set before the integrate.NVT command is
    specified.

    :math:`\tau` is related to the Nosé mass :math:`Q` by

    .. math::

        \tau = \sqrt{\frac{Q}{g k_B T_0}}

    where :math:`g` is the number of degrees of freedom, and :math:`k_B T_0` is
    the set point (*kT* above).

    *kT* can be a variant type, allowing for temperature ramps in simulation
    runs.

    A :py:class:`hoomd.compute.thermo` is automatically specified and associated
    with *group*.

    Examples::

        all = filter.All()
        integrate.NVT(filter=all, kT=1.0, tau=0.5)
        integrator = integrate.NVT(filter=all, tau=1.0, kT=0.65)
        typeA = filter.Type('A')
        integrator = integrate.NVT(filter=typeA, tau=1.0, kT=hoomd.variant.linear_interp([(0, 4.0), (1e6, 1.0)]))
    """

    def __init__(self, filter, kT, tau):

        # store metadata
        param_dict = ParameterDict(
            filter=_ParticleFilter,
            kT=Variant,
            tau=float(tau),
        )
        param_dict.update(dict(kT=kT, filter=filter))
        # set defaults
        self._param_dict.update(param_dict)

    def attach(self, simulation):

        # initialize the reflected cpp class
        if not simulation.device.cpp_exec_conf.isCUDAEnabled():
            my_class = _md.TwoStepNVTMTK
            thermo_cls = _hoomd.ComputeThermo
        else:
            my_class = _md.TwoStepNVTMTKGPU
            thermo_cls = _hoomd.ComputeThermoGPU

        group = simulation.state.get_group(self.filter)
        cpp_sys_def = simulation.state._cpp_sys_def
        thermo = thermo_cls(cpp_sys_def, group, "")
        self._cpp_obj = my_class(cpp_sys_def,
                                 group,
                                 thermo,
                                 self.tau,
                                 self.kT,
                                 "")
        super().attach(simulation)

class NPT(_Method):
    R""" NPT Integration via MTK barostat-thermostat.

    Args:
        filter (:py:mod:`hoomd.filter._ParticleFilter`): Subset of particles on which to apply this method.
        kT (:py:mod:`hoomd.variant` or :py:obj:`float`): Temperature set point for the thermostat, not needed if *nph=True* (in energy units).
        tau (float): Coupling constant for the thermostat, not needed if *nph=True* (in time units).
        S (:py:class:`list` of :py:mod:`hoomd.variant` or :py:obj:`float`): Stress components set point for the barostat (in pressure units). 
        In Voigt notation: [Sxx, Syy, Szz, Syz, Sxz, Sxy]. In case of isotropic pressure P ( [ P, P, P, 0, 0, 0]), use S = P.
        tauS (float): Coupling constant for the barostat (in time units).
        couple (str): Couplings of diagonal elements of the stress tensor, can be "none", "xy", "xz","yz", or "all" (default).
        box_dof(list): Box degrees of freedom with six boolean elements corresponding to x, y, z, xy, xz, yz, each. (default: [True,True,True,False,False,False]) 
                       If turned on to True, rescale corresponding lengths or tilt factors and components of particle coordinates and velocities
        rescale_all (bool): if True, rescale all particles, not only those in the group
        gamma: (:py:obj:`float`): Dimensionless damping factor for the box degrees of freedom (default: 0)

    :py:class:`NPT` performs constant pressure, constant temperature simulations, allowing for a fully deformable
    simulation box.

    The integration method is based on the rigorous Martyna-Tobias-Klein equations of motion for NPT.
    For optimal stability, the update equations leave the phase-space measure invariant and are manifestly
    time-reversible.

    By default, :py:class:`NPT` performs integration in a cubic box under hydrostatic pressure by simultaneously
    rescaling the lengths *Lx*, *Ly* and *Lz* of the simulation box.

    :py:class:`NPT` can also perform more advanced integration modes. The integration mode
    is specified by a set of couplings and by specifying the box degrees of freedom that are put under
    barostat control.

    Couplings define which diagonal elements of the pressure tensor :math:`P_{\alpha,\beta}`
    should be averaged over, so that the corresponding box lengths are rescaled by the same amount.

    Valid couplings are:

    - none (all box lengths are updated independently)
    - xy (*Lx* and *Ly* are coupled)
    - xz (*Lx* and *Lz* are coupled)
    - yz (*Ly* and *Lz* are coupled)
    - all (*Lx* and *Ly* (and *Lz* if 3D) are coupled)

    The default coupling is **all**, i.e. the ratios between all box lengths stay constant.

    Degrees of freedom of the box specify which lengths and tilt factors of the box should be updated,
    and how particle coordinates and velocities should be rescaled.

    Valid form for elements of box_dof(box degrees of freedom) is :

    The `box_dof` tuple controls the way the box is rescaled and updated. The first three elements ``box_dof[:2]``
    controls whether the x, y, and z box lengths are rescaled and updated, respectively. The last three entries 
    control the rescaling or the tilt factors xy, xz, and yz. All options also appropriately rescale particle coordinates and velocities.

    By default, the x, y, and z degrees of freedom are updated. [True,True,True,False,False,False]

    Note:
        If any of the diagonal x, y, z degrees of freedom is not being integrated, pressure tensor components
        along that direction are not considered for the remaining degrees of freedom.

    For example:

    - Specifying all couplings and x, y, and z degrees of freedom amounts to cubic symmetry (default)
    - Specifying xy couplings and x, y, and z degrees of freedom amounts to tetragonal symmetry.
    - Specifying no couplings and all degrees of freedom amounts to a fully deformable triclinic unit cell

    :py:class:`NPT` Can also apply a constant stress to the simulation box. To do so, specify the symmetric
    stress tensor *S* instead of an isotropic pressure *P*.

    Note:
        :py:class:`NPT` assumes that isotropic pressures are positive. Conventions for the stress tensor sometimes
        assume negative values on the diagonal. You need to set these values negative manually in HOOMD.

    :py:class:`NPT` is an integration method.

    :py:class:`NPT` uses the proper number of degrees of freedom to compute the temperature and pressure of the system in
    both 2 and 3 dimensional systems, as long as the number of dimensions is set before the :py:class:`NPT` command
    is specified.

    For the MTK equations of motion, see:

    * `G. J. Martyna, D. J. Tobias, M. L. Klein  1994 <http://dx.doi.org/10.1063/1.467468>`_
    * `M. E. Tuckerman et. al. 2006 <http://dx.doi.org/10.1088/0305-4470/39/19/S18>`_
    * `T. Yu et. al. 2010 <http://dx.doi.org/10.1016/j.chemphys.2010.02.014>`_
    * Glaser et. al (2013), to be published

    Both *kT* and *P* can be variant types, allowing for temperature/pressure ramps in simulation runs.

    :math:`\tau` is related to the Nosé mass :math:`Q` by

    .. math::

        \tau = \sqrt{\frac{Q}{g k_B T_0}}

    where :math:`g` is the number of degrees of freedom, and :math:`k_B T_0` is the set point (*kT* above).

    A :py:class:`hoomd.compute.thermo` is automatically specified and associated with *group*.

    Examples::
        integrator = integrate.NPT(filter=filter.All(), tau=1.0, kT=0.65, tauS = 1.2, S=2.0)
        # orthorhombic symmetry
        integrator = integrate.NPT(filter=filter.All(), tau=1.0, kT=0.65, tauS = 1.2, S=2.0, couple="none")
        # tetragonal symmetry
        integrator = integrate.NPT(filter=filter.All(), tau=1.0, kT=0.65, tauS = 1.2, S=2.0, couple="xy")
        # triclinic symmetry
        integrator = integrate.NPT(filter=filter.All(), tau=1.0, kT=0.65, tauS = 1.2, S=2.0, couple="none", rescale_all=True)
    """
    def __init__(self, filter, kT, tau, S, tauS, couple="all", box_dof=[True,True,True,False,False,False], rescale_all=False, gamma=0.0):


        # store metadata
        param_dict = ParameterDict(
            filter=_ParticleFilter,
            kT=Variant,
            tau=float(tau),
            S=OnlyIf(to_type_converter((Variant,)*6), preprocess=self.__preprocess_stress),
            tauS=float(tauS),
            couple=str(couple),
            box_dof=(bool,)*6,
            rescale_all=bool(rescale_all),
            gamma=float(gamma)
            )
        param_dict.update(dict(filter=filter, kT=kT, S=S, 
                                 couple=couple, box_dof=box_dof))

        # set defaults
        self._param_dict.update(param_dict)


    def attach(self, simulation):
        # initialize the reflected c++ class
        if not simulation.device.cpp_exec_conf.isCUDAEnabled():
            cpp_cls = _md.TwoStepNPTMTK
            thermo_cls = _hoomd.ComputeThermo
        else:
            cpp_cls = _md.TwoStepNPTMTKGPU
            thermo_cls = _hoomd.ComputeThermoGPU

        cpp_sys_def = simulation.state._cpp_sys_def
        thermo_group = simulation.state.get_group(self.filter)

        thermo_half_step = thermo_cls(cpp_sys_def, 
                            thermo_group,
                            "")

        thermo_full_step = thermo_cls(cpp_sys_def, 
                              thermo_group, 
                              "")

        self._cpp_obj = cpp_cls(cpp_sys_def,
                                 thermo_group,
                                 thermo_half_step,
                                 thermo_full_step,
                                 self.tau,
                                 self.tauS,
                                 self.kT,
                                 self.S,
                                 self.couple,
                                 self.box_dof,
                                 False)

        # Attach param_dict and typeparam_dict
        super().attach(simulation)

    def __preprocess_stress(self,value):
        if isinstance(value, Sequence):
            if len(value) != 6:
                raise ValueError(
                    "Expected a single hoomd.variant.Variant / float or six.")
            return tuple(value)
        else:
            return (value,value,value,0,0,0)

class nph(NPT):
    R""" NPH Integration via MTK barostat-thermostat..

    Args:
        params: keyword arguments passed to :py:class:`NPT`.
        gamma: (:py:obj:`float`, units of energy): Damping factor for the box degrees of freedom

    :py:class:`nph` performs constant pressure (NPH) simulations using a Martyna-Tobias-Klein barostat, an
    explicitly reversible and measure-preserving integration scheme. It allows for fully deformable simulation
    cells and uses the same underlying integrator as :py:class:`NPT` (with *nph=True*).

    The available options are identical to those of :py:class:`NPT`, except that *kT* cannot be specified.
    For further information, refer to the documentation of :py:class:`NPT`.

    Note:
         A time scale *tauP* for the relaxation of the barostat is required. This is defined as the
         relaxation time the barostat would have at an average temperature *T_0 = 1*, and it
         is related to the internally used (Andersen) Barostat mass :math:`W` via
         :math:`W=d N T_0 \tau_P^2`, where :math:`d` is the dimensionality and :math:`N` the number
         of particles.

    :py:class:`nph` is an integration method and must be used with ``mode_standard``.

    Examples::

        # Triclinic unit cell
        nph=integrate.nph(group=all, P=2.0, tauP=1.0, couple="none", all=True)
        # Cubic unit cell
        nph = integrate.nph(group=all, P=2.0, tauP=1.0)
        # Relax the box
        nph = integrate.nph(group=all, P=0, tauP=1.0, gamma=0.1)
    """
    def __init__(self, **params):

        # initialize base class
        npt.__init__(self, nph=True, kT=1.0, **params)

    def randomize_velocities(self, kT, seed):
        R""" Assign random velocities and angular momenta to particles in the
        group, sampling from the Maxwell-Boltzmann distribution. This method
        considers the dimensionality of the system and particle anisotropy, and
        removes drift (the center of mass velocity).

        .. versionadded:: 2.3

        Starting in version 2.5, `randomize_velocities` also chooses random values
        for the internal integrator variables.

        Args:
            kT (float): Temperature (in energy units)
            seed (int): Random number seed

        Note:
            Randomization is applied at the start of the next call to ```hoomd.run```.

        Example::

            integrator = md.integrate.nph(group=group.all(), P=2.0, tauP=1.0)
            integrator.randomize_velocities(kT=1.0, seed=42)
            run(100)

        """
        self.cpp_method.setRandomizeVelocitiesParams(kT, seed)


class nve(_Method):
    R""" NVE Integration via Velocity-Verlet

    Args:
        group (``hoomd.group``): Group of particles on which to apply this method.
        limit (bool): (optional) Enforce that no particle moves more than a distance of \a limit in a single time step
        zero_force (bool): When set to true, particles in the \a group are integrated forward in time with constant
          velocity and any net force on them is ignored.


    :py:class:`nve` performs constant volume, constant energy simulations using the standard
    Velocity-Verlet method. For poor initial conditions that include overlapping atoms, a
    limit can be specified to the movement a particle is allowed to make in one time step.
    After a few thousand time steps with the limit set, the system should be in a safe state
    to continue with unconstrained integration.

    Another use-case for :py:class:`nve` is to fix the velocity of a certain group of particles. This can be achieved by
    setting the velocity of those particles in the initial condition and setting the *zero_force* option to True
    for that group. A True value for *zero_force* causes integrate.nve to ignore any net force on each particle and
    integrate them forward in time with a constant velocity.

    Note:
        With an active limit, Newton's third law is effectively **not** obeyed and the system
        can gain linear momentum. Activate the :py:class:`hoomd.md.update.zero_momentum` updater during the limited nve
        run to prevent this.

    :py:class:`nve` is an integration method. It must be used with ``mode_standard``.

    A :py:class:`hoomd.compute.thermo` is automatically specified and associated with *group*.

    Examples::

        all = group.all()
        integrate.nve(group=all)
        integrator = integrate.nve(group=all)
        typeA = group.type('A')
        integrate.nve(group=typeA, limit=0.01)
        integrate.nve(group=typeA, zero_force=True)

    """
    def __init__(self, group, limit=None, zero_force=False):

        # initialize base class
        _Method.__init__(self)

        # create the compute thermo
        hoomd.compute._get_unique_thermo(group=group)

        # initialize the reflected c++ class
        if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
            self.cpp_method = _md.TwoStepNVE(hoomd.context.current.system_definition, group.cpp_group, False)
        else:
            self.cpp_method = _md.TwoStepNVEGPU(hoomd.context.current.system_definition, group.cpp_group)

        # set the limit
        if limit is not None:
            self.cpp_method.setLimit(limit)

        self.cpp_method.setZeroForce(zero_force)

        self.cpp_method.validateGroup()

        # store metadata
        self.group = group
        self.limit = limit
        self.metadata_fields = ['group', 'limit']

    def set_params(self, limit=None, zero_force=None):
        R""" Changes parameters of an existing integrator.

        Args:
            limit (bool): (if set) New limit value to set. Removes the limit if limit is False
            zero_force (bool): (if set) New value for the zero force option

        Examples::

            integrator.set_params(limit=0.01)
            integrator.set_params(limit=False)
        """
        self.check_initialization()

        # change the parameters
        if limit is not None:
            if limit == False:
                self.cpp_method.removeLimit()
            else:
                self.cpp_method.setLimit(limit)
            self.limit = limit

        if zero_force is not None:
            self.cpp_method.setZeroForce(zero_force)

    def randomize_velocities(self, kT, seed):
        R""" Assign random velocities and angular momenta to particles in the
        group, sampling from the Maxwell-Boltzmann distribution. This method
        considers the dimensionality of the system and particle anisotropy, and
        removes drift (the center of mass velocity).

        .. versionadded:: 2.3

        Args:
            kT (float): Temperature (in energy units)
            seed (int): Random number seed

        Note:
            Randomization is applied at the start of the next call to ```hoomd.run```.

        Example::

            integrator = md.integrate.nve(group=group.all())
            integrator.randomize_velocities(kT=1.0, seed=42)
            run(100)

        """
        self.cpp_method.setRandomizeVelocitiesParams(kT, seed)


class Langevin(_Method):
    R""" Langevin dynamics.

    Args:
        filter (:py:mod:`hoomd.filter._ParticleFilter`): Group of particles to
            apply this method to.
        kT (:py:mod:`hoomd.variant` or :py:obj:`float`): Temperature of the
            simulation (in energy units).
        seed (int): Random seed to use for generating
            :math:`\vec{F}_\mathrm{R}`.
        lambda (float): (optional) When set, use :math:\lambda d:math: for the
            drag coefficient.
        tally_reservoir_energy (bool): (optional) If true, the energy exchange
            between the thermal reservoir and the particles is tracked. Total
            energy conservation can then be monitored by adding
            ``langevin_reservoir_energy_groupname`` to the logged quantities.

    .. rubric:: Translational degrees of freedom

    :py:class:`Langevin` integrates particles forward in time according to the
    Langevin equations of motion:

    .. math::

        m \frac{d\vec{v}}{dt} = \vec{F}_\mathrm{C} - \gamma \cdot \vec{v} +
        \vec{F}_\mathrm{R}

        \langle \vec{F}_\mathrm{R} \rangle = 0

        \langle |\vec{F}_\mathrm{R}|^2 \rangle = 2 d kT \gamma / \delta t

    where :math:`\vec{F}_\mathrm{C}` is the force on the particle from all
    potentials and constraint forces, :math:`\gamma` is the drag coefficient,
    :math:`\vec{v}` is the particle's velocity, :math:`\vec{F}_\mathrm{R}` is a
    uniform random force, and :math:`d` is the dimensionality of the system (2
    or 3).  The magnitude of the random force is chosen via the
    fluctuation-dissipation theorem to be consistent with the specified drag and
    temperature, :math:`T`.  When :math:`kT=0`, the random force
    :math:`\vec{F}_\mathrm{R}=0`.

    :py:class:`Langevin` generates random numbers by hashing together the
    particle tag, user seed, and current time step index. See `C. L. Phillips
    et. al. 2011 <http://dx.doi.org/10.1016/j.jcp.2011.05.021>`_ for more
    information.

    .. attention::

        Change the seed if you reset the simulation time step to 0.
        If you keep the same seed, the simulation will continue with the same
        sequence of random numbers used previously and may cause unphysical
        correlations.

        For MPI runs: all ranks other than 0 ignore the seed input and use the
        value of rank 0.

    Langevin dynamics includes the acceleration term in the Langevin equation
    and is useful for gently thermalizing systems using a small gamma. This
    assumption is valid when underdamped: :math:`\frac{m}{\gamma} \gg \delta t`.
    Use :py:class:`brownian` if your system is not underdamped.

    :py:class:`Langevin` uses the same integrator as :py:class:`nve` with the
    additional force term :math:`- \gamma \cdot \vec{v} + \vec{F}_\mathrm{R}`.
    The random force :math:`\vec{F}_\mathrm{R}` is drawn from a uniform random
    number distribution.

    You can specify :math:`\gamma` in two ways:

    1. Use ``set_gamma()`` to specify it directly, with independent
       values for each particle type in the system.
    2. Specify :math:`\lambda` which scales the particle diameter to
       :math:`\gamma = \lambda d_i`. The units of
       :math:`\lambda` are mass / distance / time.

    :py:class:`Langevin` must be used with ``mode_standard``.

    *kT* can be a variant type, allowing for temperature ramps in simulation
    runs.

    A :py:class:`hoomd.compute.thermo` is automatically created and associated
    with *group*.

    Warning:
        When restarting a simulation, the energy of the reservoir will be reset
        to zero.

    Examples::

        all = group.all()
        integrator = integrate.langevin(group=all, kT=1.0, seed=5)
        integrator = integrate.langevin(group=all, kT=1.0, dscale=1.5, tally=True)
        typeA = group.type('A')
        integrator = integrate.langevin(group=typeA, kT=hoomd.variant.linear_interp([(0, 4.0), (1e6, 1.0)]), seed=10)

    """

    def __init__(self, filter, kT, seed, alpha=None,
                 tally_reservoir_energy=False):

        # store metadata
        param_dict = ParameterDict(
            filter=_ParticleFilter,
            kT=Variant,
            seed=int(seed),
            alpha=OnlyType(float, allow_none=True),
            tally_reservoir_energy=bool(tally_reservoir_energy),
        )
        param_dict.update(dict(kT=kT, alpha=alpha, filter=filter))
        # set defaults
        self._param_dict.update(param_dict)

        gamma = TypeParameter('gamma', type_kind='particle_types',
                              param_dict=TypeParameterDict(1., len_keys=1)
                              )

        gamma_r = TypeParameter('gamma_r', type_kind='particle_types',
                                param_dict=TypeParameterDict((1., 1., 1.),
                                                             len_keys=1)
                                )

        self._extend_typeparam([gamma,gamma_r])

    def attach(self, simulation):

        # initialize the reflected c++ class
        if not simulation.device.cpp_exec_conf.isCUDAEnabled():
            my_class = _md.TwoStepLangevin
        else:
            my_class = _md.TwoStepLangevinGPU

        self._cpp_obj = my_class(simulation.state._cpp_sys_def,
                                 simulation.state.get_group(self.filter),
                                 self.kT, self.seed)

        # Attach param_dict and typeparam_dict
        super().attach(simulation)


class brownian(_Method):
    R""" Brownian dynamics.

    Args:
        group (``hoomd.group``): Group of particles to apply this method to.
        kT (:py:mod:`hoomd.variant` or :py:obj:`float`): Temperature of the simulation (in energy units).
        seed (int): Random seed to use for generating :math:`\vec{F}_\mathrm{R}`.
        dscale (bool): Control :math:`\lambda` options. If 0 or False, use :math:`\gamma` values set per type. If non-zero, :math:`\gamma = \lambda d_i`.
        noiseless_t (bool): If set true, there will be no translational noise (random force)
        noiseless_r (bool): If set true, there will be no rotational noise (random torque)

    :py:class:`brownian` integrates particles forward in time according to the overdamped Langevin equations of motion,
    sometimes called Brownian dynamics, or the diffusive limit.

    .. math::

        \frac{d\vec{x}}{dt} = \frac{\vec{F}_\mathrm{C} + \vec{F}_\mathrm{R}}{\gamma}

        \langle \vec{F}_\mathrm{R} \rangle = 0

        \langle |\vec{F}_\mathrm{R}|^2 \rangle = 2 d k T \gamma / \delta t

        \langle \vec{v}(t) \rangle = 0

        \langle |\vec{v}(t)|^2 \rangle = d k T / m


    where :math:`\vec{F}_\mathrm{C}` is the force on the particle from all potentials and constraint forces,
    :math:`\gamma` is the drag coefficient, :math:`\vec{F}_\mathrm{R}`
    is a uniform random force, :math:`\vec{v}` is the particle's velocity, and :math:`d` is the dimensionality
    of the system. The magnitude of the random force is chosen via the fluctuation-dissipation theorem
    to be consistent with the specified drag and temperature, :math:`T`.
    When :math:`kT=0`, the random force :math:`\vec{F}_\mathrm{R}=0`.

    :py:class:`brownian` generates random numbers by hashing together the particle tag, user seed, and current
    time step index. See `C. L. Phillips et. al. 2011 <http://dx.doi.org/10.1016/j.jcp.2011.05.021>`_ for more
    information.

    .. attention::
        Change the seed if you reset the simulation time step to 0. If you keep the same seed, the simulation
        will continue with the same sequence of random numbers used previously and may cause unphysical correlations.

        For MPI runs: all ranks other than 0 ignore the seed input and use the value of rank 0.

    :py:class:`brownian` uses the integrator from `I. Snook, The Langevin and Generalised Langevin Approach to the Dynamics of
    Atomic, Polymeric and Colloidal Systems, 2007, section 6.2.5 <http://dx.doi.org/10.1016/B978-0-444-52129-3.50028-6>`_,
    with the exception that :math:`\vec{F}_\mathrm{R}` is drawn from a uniform random number distribution.

    In Brownian dynamics, particle velocities are completely decoupled from positions. At each time step,
    :py:class:`brownian` draws a new velocity distribution consistent with the current set temperature so that
    :py:class:`hoomd.compute.thermo` will report appropriate temperatures and pressures if logged or needed by other
    commands.

    Brownian dynamics neglects the acceleration term in the Langevin equation. This assumption is valid when
    overdamped: :math:`\frac{m}{\gamma} \ll \delta t`. Use :py:class:`Langevin` if your system is not overdamped.

    You can specify :math:`\gamma` in two ways:

    1. Use :py:class:`set_gamma()` to specify it directly, with independent values for each particle type in the system.
    2. Specify :math:`\lambda` which scales the particle diameter to :math:`\gamma = \lambda d_i`. The units of
       :math:`\lambda` are mass / distance / time.

    :py:class:`brownian` must be used with integrate.mode_standard.

    *kT* can be a variant type, allowing for temperature ramps in simulation runs.

    A :py:class:`hoomd.compute.thermo` is automatically created and associated with *group*.

    Examples::

        all = group.all()
        integrator = integrate.brownian(group=all, kT=1.0, seed=5)
        integrator = integrate.brownian(group=all, kT=1.0, dscale=1.5)
        typeA = group.type('A')
        integrator = integrate.brownian(group=typeA, kT=hoomd.variant.linear_interp([(0, 4.0), (1e6, 1.0)]), seed=10)

    """
    def __init__(self, group, kT, seed, dscale=False, noiseless_t=False, noiseless_r=False):

        # initialize base class
        _Method.__init__(self)

        # setup the variant inputs
        kT = hoomd.variant._setup_variant_input(kT)

        # create the compute thermo
        hoomd.compute._get_unique_thermo(group=group)

        if dscale is False or dscale == 0:
            use_lambda = False
        else:
            use_lambda = True

        # initialize the reflected c++ class
        if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
            my_class = _md.TwoStepBD
        else:
            my_class = _md.TwoStepBDGPU

        self.cpp_method = my_class(hoomd.context.current.system_definition,
                                   group.cpp_group,
                                   kT.cpp_variant,
                                   seed,
                                   use_lambda,
                                   float(dscale),
                                   noiseless_t,
                                   noiseless_r)

        self.cpp_method.validateGroup()

        # store metadata
        self.group = group
        self.kT = kT
        self.seed = seed
        self.dscale = dscale
        self.noiseless_t = noiseless_t
        self.noiseless_r = noiseless_r
        self.metadata_fields = ['group', 'kT', 'seed', 'dscale','noiseless_t','noiseless_r']

    def set_params(self, kT=None):
        R""" Change langevin integrator parameters.

        Args:
            kT (:py:mod:`hoomd.variant` or :py:obj:`float`): New temperature (if set) (in energy units).

        Examples::

            integrator.set_params(kT=2.0)

        """
        self.check_initialization()

        # change the parameters
        if kT is not None:
            # setup the variant inputs
            kT = hoomd.variant._setup_variant_input(kT)
            self.cpp_method.setT(kT.cpp_variant)
            self.kT = kT

    def set_gamma(self, a, gamma):
        R""" Set gamma for a particle type.

        Args:
            a (str): Particle type name
            gamma (float): :math:`\gamma` for particle type a (in units of force/velocity)

        :py:meth:`set_gamma()` sets the coefficient :math:`\gamma` for a single particle type, identified
        by name. The default is 1.0 if not specified for a type.

        It is not an error to specify gammas for particle types that do not exist in the simulation.
        This can be useful in defining a single simulation script for many different types of particles
        even when some simulations only include a subset.

        Examples::

            bd.set_gamma('A', gamma=2.0)

        """
        self.check_initialization()
        a = str(a)

        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes()
        type_list = []
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getParticleData().getNameByType(i))

        # change the parameters
        for i in range(0,ntypes):
            if a == type_list[i]:
                self.cpp_method.setGamma(i,gamma)

    def set_gamma_r(self, a, gamma_r):
        R""" Set gamma_r for a particle type.

        Args:
            a (str):  Particle type name
            gamma_r (float or tuple): :math:`\gamma_r` for particle type a (in units of force/velocity), optionally for all body frame directions

        :py:meth:`set_gamma_r()` sets the coefficient :math:`\gamma_r` for a single particle type, identified
        by name. The default is 1.0 if not specified for a type. It must be positive or zero, if set
        zero, it will have no rotational damping or random torque, but still with updates from normal net torque.

        Examples::

            bd.set_gamma_r('A', gamma_r=2.0)
            bd.set_gamma_r('A', gamma_r=(1,2,3))

        """

        self.check_initialization()

        if not isinstance(gamma_r,tuple):
            gamma_r = (gamma_r, gamma_r, gamma_r)

        if (gamma_r[0] < 0 or gamma_r[1] < 0 or gamma_r[2] < 0):
            raise ValueError("The gamma_r must be positive or zero (represent no rotational damping or random torque, but with updates)")

        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes()
        type_list = []
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getParticleData().getNameByType(i))

        # change the parameters
        for i in range(0,ntypes):
            if a == type_list[i]:
                self.cpp_method.setGamma_r(i,_hoomd.make_scalar3(*gamma_r))


class berendsen(_Method):
    R""" Applies the Berendsen thermostat.

    Args:
        group (``hoomd.group``): Group to which the Berendsen thermostat will be applied.
        kT (:py:mod:`hoomd.variant` or :py:obj:`float`): Temperature of thermostat. (in energy units).
        tau (float): Time constant of thermostat. (in time units)

    :py:class:`berendsen` rescales the velocities of all particles on each time step. The rescaling is performed so that
    the difference in the current temperature from the set point decays exponentially:
    `Berendsen et. al. 1984 <http://dx.doi.org/10.1063/1.448118>`_.

    .. math::

        \frac{dT_\mathrm{cur}}{dt} = \frac{T - T_\mathrm{cur}}{\tau}

    .. attention::
        :py:class:`berendsen` does not function with MPI parallel simulations.

    .. attention::
        :py:class:`berendsen` does not integrate rotational degrees of freedom.
    """
    def __init__(self, group, kT, tau):

        # Error out in MPI simulations
        if (_hoomd.is_MPI_available()):
            if hoomd.context.current.system_definition.getParticleData().getDomainDecomposition():
                hoomd.context.current.device.cpp_msg.error("integrate.berendsen is not supported in multi-processor simulations.\n\n")
                raise RuntimeError("Error setting up integration method.")

        # initialize base class
        _Method.__init__(self)

        # setup the variant inputs
        kT = hoomd.variant._setup_variant_input(kT)

        # create the compute thermo
        thermo = hoomd.compute._get_unique_thermo(group = group)

        # initialize the reflected c++ class
        if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
            self.cpp_method = _md.TwoStepBerendsen(hoomd.context.current.system_definition,
                                                     group.cpp_group,
                                                     thermo.cpp_compute,
                                                     tau,
                                                     kT.cpp_variant)
        else:
            self.cpp_method = _md.TwoStepBerendsenGPU(hoomd.context.current.system_definition,
                                                        group.cpp_group,
                                                        thermo.cpp_compute,
                                                        tau,
                                                        kT.cpp_variant)

        # store metadata
        self.kT = kT
        self.tau = tau
        self.metadata_fields = ['kT','tau']

    def randomize_velocities(self, seed):
        R""" Assign random velocities and angular momenta to particles in the
        group, sampling from the Maxwell-Boltzmann distribution. This method
        considers the dimensionality of the system and particle anisotropy, and
        removes drift (the center of mass velocity).

        .. versionadded:: 2.3

        Args:
            seed (int): Random number seed

        Note:
            Randomization is applied at the start of the next call to ```hoomd.run```.

        Example::

            integrator = md.integrate.berendsen(group=group.all(), kT=1.0, tau=0.5)
            integrator.randomize_velocities(seed=42)
            run(100)

        """
        timestep = hoomd.get_step()
        kT = self.kT.cpp_variant.getValue(timestep)
        self.cpp_method.setRandomizeVelocitiesParams(kT, seed)
