# coding: utf-8

# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: joaander / All Developers are free to add commands for new features


from hoomd import _hoomd
from hoomd.md import _md
import hoomd
from hoomd.operation import _HOOMDBaseObject
from hoomd.parameterdicts import ParameterDict, TypeParameterDict
from hoomd.filter import _ParticleFilter
from hoomd.typeparam import TypeParameter
from hoomd.typeconverter import OnlyType, OnlyIf, to_type_converter
from hoomd.variant import Variant
from hoomd.typeconverter import OnlyFrom
import copy
from collections.abc import Sequence

class _Method(_HOOMDBaseObject):
    pass


class NVT(_Method):
    R""" NVT Integration via the Nosé-Hoover thermostat.

    Args:
        filter (`hoomd.filter._ParticleFilter`): Subset of particles on which to 
            apply this method.

        kT (`hoomd.variant.Variant` or `float`): Temperature set point
            for the Nosé-Hoover thermostat. (in energy units).

        tau (`float`): Coupling constant for the Nosé-Hoover thermostat. 
            (in time units).

    `NVT` performs constant volume, constant temperature simulations
    using the Nosé-Hoover thermostat, using the MTK equations described in Refs.
    `G. J. Martyna, D. J. Tobias, M. L. Klein  1994
    <http://dx.doi.org/10.1063/1.467468>`_ and `J. Cao, G. J. Martyna 1996
    <http://dx.doi.org/10.1063/1.470959>`_.

    :math:`\tau` is related to the Nosé mass :math:`Q` by

    .. math::

        \tau = \sqrt{\frac{Q}{g k_B T_0}}

    where :math:`g` is the number of degrees of freedom, and :math:`k_B T_0` is
    the set point (*kT* above).

    .. rubric:: Rotational degrees of freedom

    `NVT` integrates rotational degrees of freedom. #TODO

    Examples::

        nvt=hoomd.md.methods.NVT(filter=hoomd.filter.All(), kT=1.0, tau=0.5)
        integrator = hoomd.md.Integrator(dt=0.005, methods=[nvt], forces=[lj])


    Attributes:
        filter (hoomd.filter._ParticleFilter): Subset of particles on which to 
        apply this method.

        kT (hoomd.variant.Variant): Temperature set point
            for the Nosé-Hoover thermostat. (in energy units).

        tau (float): Coupling constant for the Nosé-Hoover thermostat. (in time
            units).
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

    def _attach(self):

        # initialize the reflected cpp class
        if isinstance(self._simulation.device, hoomd.device.CPU):
            my_class = _md.TwoStepNVTMTK
            thermo_cls = _hoomd.ComputeThermo
        else:
            my_class = _md.TwoStepNVTMTKGPU
            thermo_cls = _hoomd.ComputeThermoGPU

        group = self._simulation.state.get_group(self.filter)
        cpp_sys_def = self._simulation.state._cpp_sys_def
        thermo = thermo_cls(cpp_sys_def, group, "")
        self._cpp_obj = my_class(cpp_sys_def,
                                 group,
                                 thermo,
                                 self.tau,
                                 self.kT,
                                 "")
        super()._attach()

class NPT(_Method):
    R""" NPT Integration via MTK barostat-thermostat.

    Args:
        filter (`hoomd.filter._ParticleFilter`): Subset of particles on which to
            apply this method.

        kT (`hoomd.variant.Variant` or `float`): Temperature set point for the 
            thermostat. (in energy units).

        tau (`float`): Coupling constant for the thermostat (in time units).

        S (`list` of `hoomd.variant.Variant` or `float`): Stress components set 
            point for the barostat (in pressure units).
            In Voigt notation: [Sxx, Syy, Szz, Syz, Sxz, Sxy]. 
            In case of isotropic pressure P ( [ p, p, p, 0, 0, 0]), use S = p.

        tauS (`float`): Coupling constant for the barostat (in time units).

        couple (`str`): Couplings of diagonal elements of the stress tensor, 
            can be "none", "xy", "xz","yz", or "all", default to "all".

        box_dof(`list`): Box degrees of freedom with six boolean elements 
            corresponding to x, y, z, xy, xz, yz, each. Default to 
            [True,True,True,False,False,False]). If turned on to True, 
            rescale corresponding lengths or tilt factors and components of 
            particle coordinates and velocities.

        rescale_all (`bool`): if True, rescale all particles, not only those in 
            the group, Default to False.

        gamma: (`float`): Dimensionless damping factor for the box degrees of 
            freedom, Default to 0.

    `NPT` performs constant pressure, constant temperature simulations, allowing 
    for a fully deformable simulation box.

    The integration method is based on the rigorous Martyna-Tobias-Klein 
    equations of motion for NPT. For optimal stability, the update equations 
    leave the phase-space measure invariant and are manifestly time-reversible.

    By default, `NPT` performs integration in a cubic box under hydrostatic 
    pressure by simultaneously rescaling the lengths *Lx*, *Ly* and *Lz* of the 
    simulation box.

    `NPT` can also perform more advanced integration modes. The integration mode
    is specified by a set of couplings and by specifying the box degrees of 
    freedom that are put under barostat control.

    Couplings define which diagonal elements of the pressure tensor 
    :math:`P_{\alpha,\beta}` should be averaged over, so that the corresponding 
    box lengths are rescaled by the same amount.

    Valid couplings are:

    - none (all box lengths are updated independently)
    - xy (*Lx* and *Ly* are coupled)
    - xz (*Lx* and *Lz* are coupled)
    - yz (*Ly* and *Lz* are coupled)
    - all (*Lx* and *Ly* (and *Lz* if 3D) are coupled)

    The default coupling is **all**, i.e. the ratios between all box lengths 
    stay constant.

    Degrees of freedom of the box specify which lengths and tilt factors of the 
    box should be updated, and how particle coordinates and velocities should be
    rescaled.

    Valid form for elements of box_dof(box degrees of freedom) is :

    The ``box_dof`` tuple controls the way the box is rescaled and updated. 
    The first three elements ``box_dof[:3]`` controls whether the x, y, and z 
    box lengths are rescaled and updated, respectively. The last three entries
    ``box_dof[3:]`` control the rescaling or the tilt factors xy, xz, and yz. 
    All options also appropriately rescale particle coordinates and velocities.

    By default, the x, y, and z degrees of freedom are updated. 
    [True,True,True,False,False,False]

    Note:
        If any of the diagonal x, y, z degrees of freedom is not being 
        integrated, pressure tensor components along that direction are not 
        considered for the remaining degrees of freedom.

    For example:

    - Specifying all couplings and x, y, and z degrees of freedom amounts to 
      cubic symmetry (default)
    - Specifying xy couplings and x, y, and z degrees of freedom amounts to 
      tetragonal symmetry.
    - Specifying no couplings and all degrees of freedom amounts to a fully 
      deformable triclinic unit cell

    `NPT` Can also apply a constant stress to the simulation box. To do so, 
    specify the symmetric stress tensor *S* .

    Note:
        `NPT` assumes that isotropic pressures are positive. Conventions for 
        the stress tensor sometimes assume negative values on the diagonal. 
        You need to set these values negative manually in HOOMD.

    `NPT` uses the proper number of degrees of freedom to compute the 
    temperature and pressure of the system in both 2 and 3 dimensional systems, 
    as long as the number of dimensions is set before the `NPT` command is 
    specified.

    For the MTK equations of motion, see:

    * `G. J. Martyna, D. J. Tobias, M. L. Klein  1994 <http://dx.doi.org/10.1063/1.467468>`_
    * `M. E. Tuckerman et. al. 2006 <http://dx.doi.org/10.1088/0305-4470/39/19/S18>`_
    * `T. Yu et. al. 2010 <http://dx.doi.org/10.1016/j.chemphys.2010.02.014>`_
    * Glaser et. al (2013), to be published

    Both *kT* and *P* can be variant types, allowing for temperature/pressure 
    ramps in simulation runs.

    :math:`\tau` is related to the Nosé mass :math:`Q` by

    .. math::

        \tau = \sqrt{\frac{Q}{g k_B T_0}}

    where :math:`g` is the number of degrees of freedom, and :math:`k_B T_0` is 
    the set point (*kT* above).


    Examples::

        npt = hoomd.md.methods.NPT(filter=hoomd.filter.All(), tau=1.0, kT=0.65, 
        tauS = 1.2, S=2.0)
        # orthorhombic symmetry
        npt = hoomd.md.methods.NPT(filter=hoomd.filter.All(), tau=1.0, kT=0.65, 
        tauS = 1.2, S=2.0, couple="none")
        # tetragonal symmetry
        npt = hoomd.md.methods.NPT(filter=hoomd.filter.All(), tau=1.0, kT=0.65, 
        tauS = 1.2, S=2.0, couple="xy")
        # triclinic symmetry
        npt = hoomd.md.methods.NPT(filter=hoomd.filter.All(), tau=1.0, kT=0.65, 
        tauS = 1.2, S=2.0, couple="none", rescale_all=True)
        integrator = hoomd.md.Integrator(dt=0.005, methods=[npt], forces=[lj])

    Attributes:
        filter (hoomd.filter._ParticleFilter): Subset of particles on which to
            apply this method.

        kT (hoomd.variant.Variant or float): Temperature set point for the 
            thermostat. (in energy units).

        tau (float): Coupling constant for the thermostat (in time units).

        S (list of `hoomd.variant.Variant` or float): Stress components set 
            point for the barostat (in pressure units).
            In Voigt notation: [Sxx, Syy, Szz, Syz, Sxz, Sxy].

        tauS (float): Coupling constant for the barostat (in time units).

        couple (str): Couplings of diagonal elements of the stress tensor, 
            can be "none", "xy", "xz","yz", or "all".

        box_dof(list): Box degrees of freedom with six boolean elements 
            corresponding to x, y, z, xy, xz, yz, each. 

        rescale_all (bool): if True, rescale all particles, not only those in 
            the group.

        gamma: (float): Dimensionless damping factor for the box degrees of 
            freedom.

    """
    def __init__(self, filter, kT, tau, S, tauS, couple, box_dof=[True,True,True,False,False,False], rescale_all=False, gamma=0.0):


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


    def _attach(self):
        # initialize the reflected c++ class
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cpp_cls = _md.TwoStepNPTMTK
            thermo_cls = _hoomd.ComputeThermo
        else:
            cpp_cls = _md.TwoStepNPTMTKGPU
            thermo_cls = _hoomd.ComputeThermoGPU

        cpp_sys_def = self._simulation.state._cpp_sys_def
        thermo_group = self._simulation.state.get_group(self.filter)

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
        super()._attach()

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


class NVE(_Method):
    R""" NVE Integration via Velocity-Verlet

    Args:
        filter (`hoomd.filter._ParticleFilter`): Subset of particles on which to
         apply this method.

        limit (None or `float`): Enforce that no particle moves more than a 
            distance of a limit in a single time step. Defaults to None

    `NVE` performs constant volume, constant energy simulations using 
    the standard Velocity-Verlet method. For poor initial conditions that 
    include overlapping atoms, a limit can be specified to the movement a 
    particle is allowed to make in one time step. After a few thousand time 
    steps with the limit set, the system should be in a safe state to continue 
    with unconstrained integration.

    Note:
        With an active limit, Newton's third law is effectively **not** obeyed 
        and the system can gain linear momentum. Activate the 
        :py:class:`hoomd.md.update.zero_momentum` updater during the limited NVE
        run to prevent this.


    Examples::

        nve = hoomd.md.methods.NVE(filter=hoomd.filter.All())
        integrator = hoomd.md.Integrator(dt=0.005, methods=[nve], forces=[lj])


    Attributes:
        filter (hoomd.filter._ParticleFilter): Subset of particles on which to 
            apply this method.

        limit (None or float): Enforce that no particle moves more than a 
            distance of a limit in a single time step. Defaults to None

    """

    def __init__(self, filter, limit=None):

        # store metadata
        param_dict = ParameterDict(
            filter=_ParticleFilter,
            limit=OnlyType(float, allow_none=True),
            zero_force=OnlyType(bool, allow_none=False),
        )
        param_dict.update(dict(filter=filter, limit=limit, zero_force=False))

        # set defaults
        self._param_dict.update(param_dict)

    def _attach(self):

        # initialize the reflected c++ class
        if isinstance(self._simulation.device, hoomd.device.CPU):
            self._cpp_obj = _md.TwoStepNVE(self._simulation.state._cpp_sys_def,
                                        self._simulation.state.get_group(self.filter), False)
        else:
            self._cpp_obj = _md.TwoStepNVEGPU(self._simulation.state._cpp_sys_def,
                                 self._simulation.state.get_group(self.filter))

        # Attach param_dict and typeparam_dict
        super()._attach()

class Langevin(_Method):
    R""" Langevin dynamics.

    Args:
        filter (`hoomd.filter._ParticleFilter`): Subset of particles to
            apply this method to.

        kT (`hoomd.variant.Variant` or `float`): Temperature of the
            simulation (in energy units).

        seed (`int`): Random seed to use for generating
            :math:`\vec{F}_\mathrm{R}`.

        alpha (`float`): When set, use :math:`\alpha d_i` for the drag 
            coefficient where :math:`d_i` is particle diameter. 
            Defaults to None.

        tally_reservoir_energy (`bool`): If true, the energy exchange
            between the thermal reservoir and the particles is tracked. Total
            energy conservation can then be monitored by adding
            ``langevin_reservoir_energy_groupname`` to the logged quantities. 
            Defaults to False.

    .. rubric:: Translational degrees of freedom

    `Langevin` integrates particles forward in time according to the
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

    `Langevin` generates random numbers by hashing together the
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
    Use `Brownian` if your system is not underdamped.

    `Langevin` uses the same integrator as `NVE` with the additional force term
    :math:`- \gamma \cdot \vec{v} + \vec{F}_\mathrm{R}`. The random force 
    :math:`\vec{F}_\mathrm{R}` is drawn from a uniform random number 
    distribution.

    You can specify :math:`\gamma` in two ways:

    1. Specify :math:`\alpha` which scales the particle diameter to 
       :math:`\gamma = \alpha d_i`. The units of :math:`\alpha` are 
       mass / distance / time.
    2. After the method object is created, specify the 
       attribute ``gamma`` and ``gamma_r`` for rotational damping or random 
       torque to assign them directly, with independent values for each 
       particle type in the system.


    Warning:
        When restarting a simulation, the energy of the reservoir will be reset
        to zero.


    Examples::

        langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=0.2, 
        seed=1, alpha=1.0)
        integrator = hoomd.md.Integrator(dt=0.001, methods=[langevin], 
        forces=[lj])


    Examples of using ``gamma`` or ``gamma_r`` on drag coefficient::

        langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=0.2,
        seed=1)
        langevin.gamma.default = 2.0
        langevin.gamma_r.default = [1.0,2.0,3.0]


    Attributes:
        filter (hoomd.filter._ParticleFilter): Subset of particles to
            apply this method to.

        kT (hoomd.variant.Variant): Temperature of the
            simulation (in energy units).

        seed (int): Random seed to use for generating
            :math:`\vec{F}_\mathrm{R}`.

        alpha (float): When set, use :math:`\alpha d_i` for the drag 
            coefficient where :math:`d_i` is particle diameter. 
            Defaults to None.

        gamma (TypeParameter[``particle type``, float]): The drag coefficient 
            can be directly set instead of the ratio of particle diameter 
            (:math:`\gamma = \alpha d_i`). The type of ``gamma`` parameter is 
            either positive float or zero.

        gamma_r (TypeParameter[``particle type``, [float,float,float]]): The 
            rotational drag coefficient can be set. The type of ``gamma_r``
            parameter is a tuple of three float. The type of each element of 
            tuple is either positive float or zero.

        tally_reservoir_energy (bool): If true, the energy exchange
            between the thermal reservoir and the particles is tracked. Total
            energy conservation can then be monitored by adding 
            ``langevin_reservoir_energy_groupname`` to the logged quantities. 
            Defaults to False.

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

    def _attach(self):

        # initialize the reflected c++ class
        if isinstance(self._simulation.device, hoomd.device.CPU):
            my_class = _md.TwoStepLangevin
        else:
            my_class = _md.TwoStepLangevinGPU

        self._cpp_obj = my_class(self._simulation.state._cpp_sys_def,
                                 self._simulation.state.get_group(self.filter),
                                 self.kT, self.seed)

        # Attach param_dict and typeparam_dict
        super()._attach()


class Brownian(_Method):
    R""" Brownian dynamics.

    Args:
        filter (`hoomd.filter._ParticleFilter`): Subset of particles to
            apply this method to.

        kT (`hoomd.variant.Variant` or `float`): Temperature of the
            simulation (in energy units).

        seed (`int`): Random seed to use for generating
            :math:`\vec{F}_\mathrm{R}`.

        alpha (`float`): When set, use :math:`\alpha d_i` for the
            drag coefficient where :math:`d_i` is particle diameter. 
            Defaults to None.

    `Brownian` integrates particles forward in time according to the overdamped 
    Langevin equations of motion, sometimes called Brownian dynamics, or the 
    diffusive limit.

    .. math::

        \frac{d\vec{x}}{dt} = \frac{\vec{F}_\mathrm{C} + 
        \vec{F}_\mathrm{R}}{\gamma}

        \langle \vec{F}_\mathrm{R} \rangle = 0

        \langle |\vec{F}_\mathrm{R}|^2 \rangle = 2 d k T \gamma / \delta t

        \langle \vec{v}(t) \rangle = 0

        \langle |\vec{v}(t)|^2 \rangle = d k T / m


    where :math:`\vec{F}_\mathrm{C}` is the force on the particle from all 
    potentials and constraint forces, :math:`\gamma` is the drag coefficient, 
    :math:`\vec{F}_\mathrm{R}` is a uniform random force, :math:`\vec{v}` is the
    particle's velocity, and :math:`d` is the dimensionality of the system. 
    The magnitude of the random force is chosen via the fluctuation-dissipation 
    theorem to be consistent with the specified drag and temperature, :math:`T`.
    When :math:`kT=0`, the random force :math:`\vec{F}_\mathrm{R}=0`.

    `Brownian` generates random numbers by hashing together the particle tag, 
    user seed, and current time step index. See 
    `C. L. Phillips et. al. 2011 <http://dx.doi.org/10.1016/j.jcp.2011.05.021>`_
    for more information.

    .. attention::
        Change the seed if you reset the simulation time step to 0. If you keep 
        the same seed, the simulation will continue with the same sequence of 
        random numbers used previously and may cause unphysical correlations.

        For MPI runs: all ranks other than 0 ignore the seed input and use the 
        value of rank 0.

    `Brownian` uses the integrator from `I. Snook, The Langevin and Generalised 
    Langevin Approach to the Dynamics of Atomic, Polymeric and Colloidal Systems
    , 2007, section 6.2.5 <http://dx.doi.org/10.1016/B978-0-444-52129-3.50028-6>`_, 
    with the exception that :math:`\vec{F}_\mathrm{R}` is drawn from a 
    uniform random number distribution.

    In Brownian dynamics, particle velocities are completely decoupled from 
    positions. At each time step, `Brownian` draws a new velocity 
    distribution consistent with the current set temperature so that 
    `hoomd.compute.thermo` will report appropriate temperatures and 
    pressures if logged or needed by other commands.

    Brownian dynamics neglects the acceleration term in the Langevin equation. 
    This assumption is valid when overdamped: 
    :math:`\frac{m}{\gamma} \ll \delta t`. Use `Langevin` if your 
    system is not overdamped.

    You can specify :math:`\gamma` in two ways:

    1. Specify :math:`\alpha` which scales the particle diameter to 
       :math:`\gamma = \alpha d_i`. The units of :math:`\alpha` are mass / 
       distance / time.
    2. After the method object is created, specify the attribute ``gamma`` 
       and ``gamma_r`` for rotational damping or random torque to assign them 
       directly, with independent values for each particle type in the 
       system.


    Examples::

        brownian = hoomd.md.methods.Brownian(filter=hoomd.filter.All(), kT=0.2, 
        seed=1, alpha=1.0)
        integrator = hoomd.md.Integrator(dt=0.001, methods=[brownian], 
        forces=[lj])


    Examples of using ``gamma`` pr ``gamma_r`` on drag coefficient::

        brownian = hoomd.md.methods.Brownian(filter=hoomd.filter.All(), kT=0.2,
        seed=1)
        brownian.gamma.default = 2.0
        brownian.gamma_r.default = [1.0, 2.0, 3.0]


    Attributes:
        filter (hoomd.filter._ParticleFilter): Subset of particles to
            apply this method to.

        kT (hoomd.variant.Variant): Temperature of the
            simulation (in energy units).

        seed (int): Random seed to use for generating
            :math:`\vec{F}_\mathrm{R}`.

        alpha (float): When set, use :math:`\alpha d_i` for the drag 
            coefficient where :math:`d_i` is particle diameter. 
            Defaults to None.

        gamma (TypeParameter[``particle type``, `float`]): The drag coefficient 
            can be directly set instead of the ratio of particle diameter 
            (:math:`\gamma = \alpha d_i`). The type of ``gamma`` parameter is 
            either positive float or zero.

        gamma_r (TypeParameter[``particle type``, [`float`, `float`, `float`] ]): 
            The rotational drag coefficient can be set. The type of ``gamma_r``
            parameter is a tuple of three float. The type of each element of 
            tuple is either positive float or zero.

    """

    def __init__(self, filter, kT, seed, alpha=None):

        # store metadata
        param_dict = ParameterDict(
            filter=_ParticleFilter,
            kT=Variant,
            seed=int(seed),
            alpha=OnlyType(float, allow_none=True),
            )
        param_dict.update(dict(kT=kT, alpha=alpha, filter=filter))

        #set defaults
        self._param_dict.update(param_dict)

        gamma = TypeParameter('gamma', type_kind='particle_types',
                              param_dict=TypeParameterDict(1., len_keys=1)
                              )

        gamma_r = TypeParameter('gamma_r', type_kind='particle_types',
                                param_dict=TypeParameterDict((1., 1., 1.), len_keys=1)
                                )
        self._extend_typeparam([gamma,gamma_r])


    def _attach(self):

        # initialize the reflected c++ class
        sim = self._simulation
        if isinstance(sim.device, hoomd.device.CPU):
            self._cpp_obj = _md.TwoStepBD(sim.state._cpp_sys_def,
                                          sim.state.get_group(self.filter),
                                          self.kT, self.seed)
        else:
            self._cpp_obj = _md.TwoStepBDGPU(sim.state._cpp_sys_def,
                                             sim.state.get_group(self.filter),
                                             self.kT, self.seed)

        # Attach param_dict and typeparam_dict
        super()._attach()


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
