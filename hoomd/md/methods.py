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
from hoomd.typeconverter import OnlyType
from hoomd.variant import Variant
import copy


class _Method(_HOOMDBaseObject):
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
        nvt=hoomd.md.methods.NVT(filter=all, kT=1.0, tau=0.5)
        integrator = hoomd.md.Integrator(dt=0.005, methods=[nvt], forces=[lj])
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


class npt(_Method):
    R""" NPT Integration via MTK barostat-thermostat.

    Args:
        group (``hoomd.group``): Group of particles on which to apply this method.
        kT (:py:mod:`hoomd.variant` or :py:obj:`float`): Temperature set point for the thermostat, not needed if *nph=True* (in energy units).
        tau (float): Coupling constant for the thermostat, not needed if *nph=True* (in time units).
        S (:py:class:`list` of :py:mod:`hoomd.variant` or :py:obj:`float`): Stress components set point for the barostat (in pressure units). In Voigt notation: [Sxx, Syy, Szz, Syz, Sxz, Sxy]
        P (:py:mod:`hoomd.variant` or :py:obj:`float`): Isotropic pressure set point for the barostat (in pressure units). Overrides *S* if set.
        tauP (float): Coupling constant for the barostat (in time units).
        couple (str): Couplings of diagonal elements of the stress tensor, can be "none", "xy", "xz","yz", or "xyz" (default).
        x (bool): if True, rescale *Lx* and x component of particle coordinates and velocities
        y (bool): if True, rescale *Ly* and y component of particle coordinates and velocities
        z (bool): if True, rescale *Lz* and z component of particle coordinates and velocities
        xy (bool): if True, rescale xy tilt factor and x and y components of particle coordinates and velocities
        xz (bool): if True, rescale xz tilt factor and x and z components of particle coordinates and velocities
        yz (bool): if True, rescale yz tilt factor and y and z components of particle coordinates and velocities
        all (bool): if True, rescale all lengths and tilt factors and components of particle coordinates and velocities
        nph (bool): if True, integrate without a thermostat, i.e. in the NPH ensemble
        rescale_all (bool): if True, rescale all particles, not only those in the group
        gamma: (:py:obj:`float`): Dimensionless damping factor for the box degrees of freedom (default: 0)

    :py:class:`npt` performs constant pressure, constant temperature simulations, allowing for a fully deformable
    simulation box.

    The integration method is based on the rigorous Martyna-Tobias-Klein equations of motion for NPT.
    For optimal stability, the update equations leave the phase-space measure invariant and are manifestly
    time-reversible.

    By default, :py:class:`npt` performs integration in a cubic box under hydrostatic pressure by simultaneously
    rescaling the lengths *Lx*, *Ly* and *Lz* of the simulation box.

    :py:class:`npt` can also perform more advanced integration modes. The integration mode
    is specified by a set of couplings and by specifying the box degrees of freedom that are put under
    barostat control.

    Couplings define which diagonal elements of the pressure tensor :math:`P_{\alpha,\beta}`
    should be averaged over, so that the corresponding box lengths are rescaled by the same amount.

    Valid couplings are:

    - none (all box lengths are updated independently)
    - xy (*Lx* and *Ly* are coupled)
    - xz (*Lx* and *Lz* are coupled)
    - yz (*Ly* and *Lz* are coupled)
    - xyz (*Lx* and *Ly* and *Lz* are coupled)

    The default coupling is **xyz**, i.e. the ratios between all box lengths stay constant.

    Degrees of freedom of the box specify which lengths and tilt factors of the box should be updated,
    and how particle coordinates and velocities should be rescaled.

    Valid keywords for degrees of freedom are:

    - x (the box length Lx is updated)
    - y (the box length Ly is updated)
    - z (the box length Lz is updated)
    - xy (the tilt factor xy is updated)
    - xz (the tilt factor xz is updated)
    - yz (the tilt factor yz is updated)
    - all (all elements are updated, equivalent to x, y, z, xy, xz, and yz together)

    Any of the six keywords can be combined together. By default, the x, y, and z degrees of freedom
    are updated.

    Note:
        If any of the diagonal x, y, z degrees of freedom is not being integrated, pressure tensor components
        along that direction are not considered for the remaining degrees of freedom.

    For example:

    - Specifying xyz couplings and x, y, and z degrees of freedom amounts to cubic symmetry (default)
    - Specifying xy couplings and x, y, and z degrees of freedom amounts to tetragonal symmetry.
    - Specifying no couplings and all degrees of freedom amounts to a fully deformable triclinic unit cell

    :py:class:`npt` Can also apply a constant stress to the simulation box. To do so, specify the symmetric
    stress tensor *S* instead of an isotropic pressure *P*.

    Note:
        :py:class:`npt` assumes that isotropic pressures are positive. Conventions for the stress tensor sometimes
        assume negative values on the diagonal. You need to set these values negative manually in HOOMD.

    :py:class:`npt` is an integration method. It must be used with ``mode_standard``.

    :py:class:`npt` uses the proper number of degrees of freedom to compute the temperature and pressure of the system in
    both 2 and 3 dimensional systems, as long as the number of dimensions is set before the :py:class:`npt` command
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

        integrate.npt(group=all, kT=1.0, tau=0.5, tauP=1.0, P=2.0)
        integrator = integrate.npt(group=all, tau=1.0, kT=0.65, tauP = 1.2, P=2.0)
        # orthorhombic symmetry
        integrator = integrate.npt(group=all, tau=1.0, kT=0.65, tauP = 1.2, P=2.0, couple="none")
        # tetragonal symmetry
        integrator = integrate.npt(group=all, tau=1.0, kT=0.65, tauP = 1.2, P=2.0, couple="xy")
        # triclinic symmetry
        integrator = integrate.npt(group=all, tau=1.0, kT=0.65, tauP = 1.2, P=2.0, couple="none", rescale_all=True)
    """
    def __init__(self, group, kT=None, tau=None, S=None, P=None, tauP=None, couple="xyz", x=True, y=True, z=True, xy=False, xz=False, yz=False, all=False, nph=False, rescale_all=None, gamma=None):

        # check the input
        if (kT is None or tau is None):
            if nph is False:
                hoomd.context.current.device.cpp_msg.error("integrate.npt: Need temperature T and thermostat time scale tau.\n")
                raise RuntimeError("Error setting up NPT integration.")
            else:
                # use dummy values
                kT=1.0
                tau=1.0

        if (tauP is None):
                hoomd.context.current.device.cpp_msg.error("integrate.npt: Need barostat time scale tauP.\n")
                raise RuntimeError("Error setting up NPT integration.")

        # initialize base class
        _Method.__init__(self)

        # setup the variant inputs
        kT = hoomd.variant._setup_variant_input(kT)

        # If P is set
        if P is not None:
            self.S = [P,P,P,0,0,0]
        else:
            # S is a stress, should be [xx, yy, zz, yz, xz, xy]
            if S is not None and len(S)==6:
                self.S = S
            else:
                raise RuntimeError("Unrecognized stress tensor form")

        S = [hoomd.variant._setup_variant_input(self.S[i]) for i in range(6)]

        Svar = [S[i].cpp_variant for i in range(6)]

        # create the compute thermo for half time steps
        if group is hoomd.context.current.group_all:
            group_copy = copy.copy(group)
            group_copy.name = "__npt_all"
            thermo_group = hoomd.compute.thermo(group_copy)
            thermo_group.cpp_compute.setLoggingEnabled(False)
        else:
            thermo_group = hoomd.compute._get_unique_thermo(group=group)

        # create the compute thermo for full time step
        thermo_group_t = hoomd.compute._get_unique_thermo(group=group)

        # need to know if we are running 2D simulations
        twod = (hoomd.context.current.system_definition.getNDimensions() == 2)
        if twod:
            hoomd.context.current.device.cpp_msg.notice(2, "When running in 2D, z couplings and degrees of freedom are silently ignored.\n")

        # initialize the reflected c++ class
        if twod:
            # silently ignore any couplings that involve z
            if couple == "none":
                cpp_couple = _md.TwoStepNPTMTK.couplingMode.couple_none
            elif couple == "xy":
                cpp_couple = _md.TwoStepNPTMTK.couplingMode.couple_xy
            elif couple == "xz":
                cpp_couple = _md.TwoStepNPTMTK.couplingMode.couple_none
            elif couple == "yz":
                cpp_couple = _md.TwoStepNPTMTK.couplingMode.couple_none
            elif couple == "xyz":
                cpp_couple = _md.TwoStepNPTMTK.couplingMode.couple_xy
            else:
                hoomd.context.current.device.cpp_msg.error("Invalid coupling mode\n")
                raise RuntimeError("Error setting up NPT integration.")
        else:
            if couple == "none":
                cpp_couple = _md.TwoStepNPTMTK.couplingMode.couple_none
            elif couple == "xy":
                cpp_couple = _md.TwoStepNPTMTK.couplingMode.couple_xy
            elif couple == "xz":
                cpp_couple = _md.TwoStepNPTMTK.couplingMode.couple_xz
            elif couple == "yz":
                cpp_couple = _md.TwoStepNPTMTK.couplingMode.couple_yz
            elif couple == "xyz":
                cpp_couple = _md.TwoStepNPTMTK.couplingMode.couple_xyz
            else:
                hoomd.context.current.device.cpp_msg.error("Invalid coupling mode\n")
                raise RuntimeError("Error setting up NPT integration.")

        # set degrees of freedom flags
        # silently ignore z related degrees of freedom when running in 2d
        flags = 0
        if x or all:
            flags |= int(_md.TwoStepNPTMTK.baroFlags.baro_x)
        if y or all:
            flags |= int(_md.TwoStepNPTMTK.baroFlags.baro_y)
        if (z or all) and not twod:
            flags |= int(_md.TwoStepNPTMTK.baroFlags.baro_z)
        if xy or all:
            flags |= int(_md.TwoStepNPTMTK.baroFlags.baro_xy)
        if (xz or all) and not twod:
            flags |= int(_md.TwoStepNPTMTK.baroFlags.baro_xz)
        if (yz or all) and not twod:
            flags |= int(_md.TwoStepNPTMTK.baroFlags.baro_yz)

        if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
            self.cpp_method = _md.TwoStepNPTMTK(hoomd.context.current.system_definition,
                                                group.cpp_group,
                                                thermo_group.cpp_compute,
                                                thermo_group_t.cpp_compute,
                                                tau,
                                                tauP,
                                                kT.cpp_variant,
                                                Svar,
                                                cpp_couple,
                                                flags,
                                                nph)
        else:
            self.cpp_method = _md.TwoStepNPTMTKGPU(hoomd.context.current.system_definition,
                                                   group.cpp_group,
                                                   thermo_group.cpp_compute,
                                                   thermo_group_t.cpp_compute,
                                                   tau,
                                                   tauP,
                                                   kT.cpp_variant,
                                                   Svar,
                                                   cpp_couple,
                                                   flags,
                                                   nph)

        if rescale_all is not None:
            self.cpp_method.setRescaleAll(rescale_all)

        if gamma is not None:
            self.cpp_method.setGamma(gamma)

        self.cpp_method.validateGroup()

        # store metadata
        self.group  = group
        self.kT = kT
        self.tau = tau
        self.tauP = tauP
        self.couple = couple
        self.rescale_all = rescale_all
        self.all = all
        self.x = x
        self.y = y
        self.z = z
        self.xy = xy
        self.xz = xz
        self.yz = yz
        self.nph = nph

    def set_params(self, kT=None, tau=None, S=None, P=None, tauP=None, rescale_all=None, gamma=None):
        R""" Changes parameters of an existing integrator.

        Args:
            kT (:py:mod:`hoomd.variant` or :py:obj:`float`): New temperature (if set) (in energy units)
            tau (float): New coupling constant (if set) (in time units)
            S (:py:class:`list` of :py:mod:`hoomd.variant` or :py:obj:`float`): New stress components set point (if set) for the barostat (in pressure units). In Voigt notation: [Sxx, Syy, Szz, Syz, Sxz, Sxy]
            P (:py:mod:`hoomd.variant` or :py:obj:`float`): New isotropic pressure set point (if set) for the barostat (in pressure units). Overrides *S* if set.
            tauP (float): New barostat coupling constant (if set) (in time units)
            rescale_all (bool): When True, rescale all particles, not only those in the group

        Examples::

            integrator.set_params(tau=0.6)
            integrator.set_params(dt=3e-3, kT=2.0, P=1.0)

        """
        self.check_initialization()

        # change the parameters
        if kT is not None:
            # setup the variant inputs
            kT = hoomd.variant._setup_variant_input(kT)
            self.kT = kT
            self.cpp_method.setT(kT.cpp_variant)
        if tau is not None:
            self.cpp_method.setTau(tau)
            self.tau = tau

        # If P is set
        if P is not None:
            self.S = [P,P,P,0,0,0]
        elif S is not None:
            # S is a stress, should be [xx, yy, zz, yz, xz, xy]
            if (len(S)==6):
                self.S = S
            else:
                raise RuntimeError("Unrecognized stress tensor form")

        if P is not None or S is not None:
            S = [hoomd.variant._setup_variant_input(self.S[i]) for i in range(6)]

            Svar = [S[i].cpp_variant for i in range(6)]
            self.cpp_method.setS(Svar)

        if tauP is not None:
            self.cpp_method.setTauP(tauP)
            self.tauP = tauP
        if rescale_all is not None:
            self.cpp_method.setRescaleAll(rescale_all)
            self.rescale_all = rescale_all
        if gamma is not None:
            self.cpp_method.setGamma(gamma)

    ## \internal
    # \brief Return information about this integration method
    #
    def get_metadata(self):
        # Metadata output involves transforming some variables into human-readable
        # form, so we override get_metadata()
        data = _Method.get_metadata(self)
        data['group'] = self.group.name
        if not self.nph:
            data['kT'] = self.kT
            data['tau'] = self.tau
        data['S'] = self.S
        data['tauP'] = self.tauP

        lengths = ''
        if self.x or self.all:
            lengths += 'x '
        if self.y or self.all:
            lengths += 'y '
        if self.z or self.all:
            lengths += 'z '
        if self.xy or self.all:
            lengths += 'xy '
        if self.xz or self.all:
            lengths += 'xz '
        if self.yz or self.all:
            lengths += 'yz '
        data['lengths'] = lengths.rstrip()
        if self.rescale_all is not None:
            data['rescale_all'] = self.rescale_all

        return data

    def randomize_velocities(self, seed):
        R""" Assign random velocities and angular momenta to particles in the
        group, sampling from the Maxwell-Boltzmann distribution. This method
        considers the dimensionality of the system and particle anisotropy, and
        removes drift (the center of mass velocity).

        .. versionadded:: 2.3

        Starting in version 2.5, `randomize_velocities` also chooses random values
        for the internal integrator variables.

        Args:
            seed (int): Random number seed

        Note:
            Randomization is applied at the start of the next call to ```hoomd.run```.

        Example::

            integrator = md.integrate.npt(group=group.all(), kT=1.0, tau=0.5, tauP=1.0, P=2.0)
            integrator.randomize_velocities(seed=42)
            run(100)

        """
        timestep = hoomd.get_step()
        kT = self.kT.cpp_variant.getValue(timestep)
        self.cpp_method.setRandomizeVelocitiesParams(kT, seed)


class nph(npt):
    R""" NPH Integration via MTK barostat-thermostat..

    Args:
        params: keyword arguments passed to :py:class:`npt`.
        gamma: (:py:obj:`float`, units of energy): Damping factor for the box degrees of freedom

    :py:class:`nph` performs constant pressure (NPH) simulations using a Martyna-Tobias-Klein barostat, an
    explicitly reversible and measure-preserving integration scheme. It allows for fully deformable simulation
    cells and uses the same underlying integrator as :py:class:`npt` (with *nph=True*).

    The available options are identical to those of :py:class:`npt`, except that *kT* cannot be specified.
    For further information, refer to the documentation of :py:class:`npt`.

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
        filter (:py:mod:`hoomd.filter`): Subset of particles on which to apply this
            method.
        limit (bool): (optional) Enforce that no particle moves more than a distance of \a limit in a single time step
        zero_force (bool): When set to true, particles in the \a group are integrated forward in time with constant
          velocity and any net force on them is ignored.


    :py:class:`NVE` performs constant volume, constant energy simulations using the standard
    Velocity-Verlet method. For poor initial conditions that include overlapping atoms, a
    limit can be specified to the movement a particle is allowed to make in one time step.
    After a few thousand time steps with the limit set, the system should be in a safe state
    to continue with unconstrained integration.

    Another use-case for :py:class:`NVE` is to fix the velocity of a certain group of particles. This can be achieved by
    setting the velocity of those particles in the initial condition and setting the *zero_force* option to True
    for that group. A True value for *zero_force* causes integrate.NVE to ignore any net force on each particle and
    integrate them forward in time with a constant velocity.

    Note:
        With an active limit, Newton's third law is effectively **not** obeyed and the system
        can gain linear momentum. Activate the :py:class:`hoomd.md.update.zero_momentum` updater during the limited NVE
        run to prevent this.

    :py:class:`NVE` is an integration method. It must be used with ``mode_standard``.

    A :py:class:`hoomd.compute.thermo` is automatically specified and associated with *group*.

    Examples::

        all = hoomd.filter.All()
        nve = hoomd.md.methods.NVE(filter=all)
        nve = hoomd.md.methods.NVE(filter=all, limit=0.01)
        nve = hoomd.md.methods.NVE(filter=all, zero_force=True)
        integrator = hoomd.md.Integrator(dt=0.005, methods=[nve], forces=[lj])

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
        filter (:py:mod:`hoomd.filter._ParticleFilter`): Group of particles to
            apply this method to.
        kT (:py:mod:`hoomd.variant` or :py:obj:`float`): Temperature of the
            simulation (in energy units).
        seed (int): Random seed to use for generating
            :math:`\vec{F}_\mathrm{R}`.
        alpha (float): When set, use :math:\alpha d:math: for the
            drag coefficient. Defaults to None
        tally_reservoir_energy (bool): If true, the energy exchange
            between the thermal reservoir and the particles is tracked. Total
            energy conservation can then be monitored by adding
            ``langevin_reservoir_energy_groupname`` to the logged quantities. Defaults to False.

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
    Use `Brownian` if your system is not underdamped.

    :py:class:`Langevin` uses the same integrator as :py:class:`NVE` with the
    additional force term :math:`- \gamma \cdot \vec{v} + \vec{F}_\mathrm{R}`.
    The random force :math:`\vec{F}_\mathrm{R}` is drawn from a uniform random
    number distribution.

    You can specify :math:`\gamma` in two ways:

    1. Use ``set_gamma()`` to specify it directly, with independent
       values for each particle type in the system.
    2. Specify :math:`\alpha` which scales the particle diameter to
       :math:`\gamma = \alpha d_i`. The units of
       :math:`\alpha` are mass / distance / time.

    *kT* can be a variant type, allowing for temperature ramps in simulation
    runs.

    A :py:class:`hoomd.compute.thermo` is automatically created and associated
    with *group*.

    Warning:
        When restarting a simulation, the energy of the reservoir will be reset
        to zero.

    Examples::

        all=hoomd.filter.All()
        langevin = hoomd.md.methods.Langevin(filter=all, kT=0.2, seed=1, alpha=1.0)
        integrator = hoomd.md.Integrator(dt=0.001, methods=[langevin], forces=[lj])

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
        filter (:py:mod:`hoomd.filter._ParticleFilter`): Group of particles to
            apply this method to.
        kT (:py:mod:`hoomd.variant` or :py:obj:`float`): Temperature of the
            simulation (in energy units).
        seed (int): Random seed to use for generating
            :math:`\vec{F}_\mathrm{R}`.
        alpha (float): (optional) When set, use :math:\alpha d:math: for the
            drag coefficient.

    :py:class:`Brownian` integrates particles forward in time according to the overdamped Langevin equations of motion,
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

    :py:class:`Brownian` generates random numbers by hashing together the particle tag, user seed, and current
    time step index. See `C. L. Phillips et. al. 2011 <http://dx.doi.org/10.1016/j.jcp.2011.05.021>`_ for more
    information.

    .. attention::
        Change the seed if you reset the simulation time step to 0. If you keep the same seed, the simulation
        will continue with the same sequence of random numbers used previously and may cause unphysical correlations.

        For MPI runs: all ranks other than 0 ignore the seed input and use the value of rank 0.

    :py:class:`Brownian` uses the integrator from `I. Snook, The Langevin and Generalised Langevin Approach to the Dynamics of
    Atomic, Polymeric and Colloidal Systems, 2007, section 6.2.5 <http://dx.doi.org/10.1016/B978-0-444-52129-3.50028-6>`_,
    with the exception that :math:`\vec{F}_\mathrm{R}` is drawn from a uniform random number distribution.

    In Brownian dynamics, particle velocities are completely decoupled from positions. At each time step,
    :py:class:`Brownian` draws a new velocity distribution consistent with the current set temperature so that
    :py:class:`hoomd.compute.thermo` will report appropriate temperatures and pressures if logged or needed by other
    commands.

    Brownian dynamics neglects the acceleration term in the Langevin equation. This assumption is valid when
    overdamped: :math:`\frac{m}{\gamma} \ll \delta t`. Use :py:class:`Langevin` if your system is not overdamped.

    You can specify :math:`\gamma` in two ways:

    1. Use ``set_gamma`` to specify it directly, with independent values for each particle type in the system.
    2. Specify :math:`\alpha` which scales the particle diameter to :math:`\gamma = \alpha d_i`. The units of
       :math:`\alpha` are mass / distance / time.

    *kT* can be a variant type, allowing for temperature ramps in simulation runs.

    A :py:class:`hoomd.compute.thermo` is automatically created and associated with *group*.

    Examples::

        all=hoomd.filter.All()
        brownian = hoomd.md.methods.Brownian(filter=all, kT=0.2, seed=1, alpha=1.0)
        integrator = hoomd.md.Integrator(dt=0.001, methods=[brownian], forces=[lj])

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
