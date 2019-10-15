# coding: utf-8

# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: joaander / All Developers are free to add commands for new features

R""" Integration methods.

To integrate the system forward in time, an integration mode must be set. Only one integration mode can be active at
a time, and the last ``integrate.mode_*`` command before the :py:func:`hoomd.run()` command is the one that will take effect. It is
possible to set one mode, run for a certain number of steps and then switch to another mode before the next run
command.

The most commonly used mode is :py:class`mode_standard`. It specifies a standard mode where, at each time
step, all of the specified forces are evaluated and used in moving the system forward to the next step.
:py:class`mode_standard` doesn't integrate any particles by itself, one or more compatible integration methods must
be specified before the staring a :py:func:`hoomd.run()`. Like commands that specify forces, integration methods are
**persistent** and remain set until they are disabled.

To clarify, the following series of commands will run for 1000 time steps in the NVT ensemble and then switch to
NVE for another 1000 steps::

    all = group.all()
    integrate.mode_standard(dt=0.005)
    nvt = integrate.nvt(group=all, kT=1.2, tau=0.5)
    run(1000)
    nvt.disable()
    integrate.nve(group=all)
    run(1000)

You can change integrator parameters between runs::

    integrator = integrate.nvt(group=all, kT=1.2, tau=0.5)
    run(100)
    integrator.set_params(kT=1.0)
    run(100)

This code snippet runs the first 100 time steps with kT=1.2 and the next 100 with kT=1.0.
"""

from hoomd import _hoomd;
from hoomd.md import _md;
import hoomd;
from hoomd.integrate import _integrator, _integration_method
import copy;
import sys;

class mode_standard(_integrator):
    R""" Enables a variety of standard integration methods.

    Args:
        dt (float): Each time step of the simulation :py:func:`hoomd.run()` will advance the real time of the system forward by *dt* (in time units).
        aniso (bool): Whether to integrate rotational degrees of freedom (bool), default None (autodetect).

    :py:class:`mode_standard` performs a standard time step integration technique to move the system forward. At each time
    step, all of the specified forces are evaluated and used in moving the system forward to the next step.

    By itself, :py:class:`mode_standard` does nothing. You must specify one or more integration methods to apply to the
    system. Each integration method can be applied to only a specific group of particles enabling advanced simulation
    techniques.

    The following commands can be used to specify the integration methods used by integrate.mode_standard.

    - :py:class:`brownian`
    - :py:class:`langevin`
    - :py:class:`nve`
    - :py:class:`nvt`
    - :py:class:`npt`
    - :py:class:`nph`

    There can only be one integration mode active at a time. If there are more than one ``integrate.mode_*`` commands in
    a hoomd script, only the most recent before a given :py:func:`hoomd.run()` will take effect.

    Examples::

        integrate.mode_standard(dt=0.005)
        integrator_mode = integrate.mode_standard(dt=0.001)

    Some integration methods (notable :py:class:`nvt`, :py:class:`npt` and :py:class:`nph` maintain state between
    different :py:func:`hoomd.run()` commands, to allow for restartable simulations. After adding or removing particles, however,
    a new :py:func:`hoomd.run()` will continue from the old state and the integrator variables will re-equilibrate.
    To ensure equilibration from a unique reference state (such as all integrator variables set to zero),
    the method :py:method:reset_methods() can be use to re-initialize the variables.
    """
    def __init__(self, dt, aniso=None):
        hoomd.util.print_status_line();

        # initialize base class
        _integrator.__init__(self);

        # Store metadata
        self.dt = dt
        self.aniso = aniso
        self.metadata_fields = ['dt', 'aniso']

        # initialize the reflected c++ class
        self.cpp_integrator = _md.IntegratorTwoStep(hoomd.context.current.system_definition, dt);
        self.supports_methods = True;

        hoomd.context.current.system.setIntegrator(self.cpp_integrator);

        hoomd.util.quiet_status();
        if aniso is not None:
            self.set_params(aniso=aniso)
        hoomd.util.unquiet_status();

    ## \internal
    #  \brief Cached set of anisotropic mode enums for ease of access
    _aniso_modes = {
        None: _md.IntegratorAnisotropicMode.Automatic,
        True: _md.IntegratorAnisotropicMode.Anisotropic,
        False: _md.IntegratorAnisotropicMode.Isotropic}

    def set_params(self, dt=None, aniso=None):
        R""" Changes parameters of an existing integration mode.

        Args:
            dt (float): New time step delta (if set) (in time units).
            aniso (bool): Anisotropic integration mode (bool), default None (autodetect).

        Examples::

            integrator_mode.set_params(dt=0.007)
            integrator_mode.set_params(dt=0.005, aniso=False)

        """
        hoomd.util.print_status_line();
        self.check_initialization();

        # change the parameters
        if dt is not None:
            self.dt = dt
            self.cpp_integrator.setDeltaT(dt);

        if aniso is not None:
            if aniso in self._aniso_modes:
                anisoMode = self._aniso_modes[aniso]
            else:
                hoomd.context.msg.error("integrate.mode_standard: unknown anisotropic mode {}.\n".format(aniso));
                raise RuntimeError("Error setting anisotropic integration mode.");
            self.aniso = aniso
            self.cpp_integrator.setAnisotropicMode(anisoMode)

    def reset_methods(self):
        R""" (Re-)initialize the integrator variables in all integration methods

        .. versionadded:: 2.2

        Examples::

            run(100)
            # .. modify the system state, e.g. add particles ..
            integrator_mode.reset_methods()
            run(100)

        """
        hoomd.util.print_status_line();
        self.check_initialization();
        self.cpp_integrator.initializeIntegrationMethods();


class nvt(_integration_method):
    R""" NVT Integration via the Nosé-Hoover thermostat.

    Args:
        group (:py:mod:`hoomd.group`): Group of particles on which to apply this method.
        kT (:py:mod:`hoomd.variant` or :py:obj:`float`): Temperature set point for the Nosé-Hoover thermostat. (in energy units).
        tau (float): Coupling constant for the Nosé-Hoover thermostat. (in time units).

    :py:class:`nvt` performs constant volume, constant temperature simulations using the Nosé-Hoover thermostat,
    using the MTK equations described in Refs. `G. J. Martyna, D. J. Tobias, M. L. Klein  1994 <http://dx.doi.org/10.1063/1.467468>`_ and
    `J. Cao, G. J. Martyna 1996 <http://dx.doi.org/10.1063/1.470959>`_.

    :py:class:`nvt` is an integration method. It must be used in connection with :py:class:`mode_standard`.

    :py:class:`nvt` uses the proper number of degrees of freedom to compute the temperature of the system in both
    2 and 3 dimensional systems, as long as the number of dimensions is set before the integrate.nvt command
    is specified.

    :math:`\tau` is related to the Nosé mass :math:`Q` by

    .. math::

        \tau = \sqrt{\frac{Q}{g k_B T_0}}

    where :math:`g` is the number of degrees of freedom, and :math:`k_B T_0` is the set point (*kT* above).

    *kT* can be a variant type, allowing for temperature ramps in simulation runs.

    A :py:class:`hoomd.compute.thermo` is automatically specified and associated with *group*.

    Examples::

        all = group.all()
        integrate.nvt(group=all, kT=1.0, tau=0.5)
        integrator = integrate.nvt(group=all, tau=1.0, kT=0.65)
        typeA = group.type('A')
        integrator = integrate.nvt(group=typeA, tau=1.0, kT=hoomd.variant.linear_interp([(0, 4.0), (1e6, 1.0)]))
    """
    def __init__(self, group, kT, tau):
        hoomd.util.print_status_line();

        # initialize base class
        _integration_method.__init__(self);

        # setup the variant inputs
        kT = hoomd.variant._setup_variant_input(kT);

        # create the compute thermo
        # the NVT integrator uses the ComputeThermo in such a way that ComputeThermo stores half-time step
        # values. By assigning a separate ComputeThermo to the integrator, we are still able to log full time step values
        if group is hoomd.context.current.group_all:
            group_copy = copy.copy(group);
            group_copy.name = "__nvt_all";
            hoomd.util.quiet_status();
            thermo = hoomd.compute.thermo(group_copy);
            thermo.cpp_compute.setLoggingEnabled(False);
            hoomd.util.unquiet_status();
        else:
            thermo = hoomd.compute._get_unique_thermo(group=group);

        # store metadata
        self.group = group
        self.kT = kT
        self.tau = tau
        self.metadata_fields = ['group', 'kT', 'tau']

        # setup suffix
        suffix = '_' + group.name;

        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_method = _md.TwoStepNVTMTK(hoomd.context.current.system_definition, group.cpp_group, thermo.cpp_compute, tau, kT.cpp_variant, suffix);
        else:
            self.cpp_method = _md.TwoStepNVTMTKGPU(hoomd.context.current.system_definition, group.cpp_group, thermo.cpp_compute, tau, kT.cpp_variant, suffix);

        self.cpp_method.validateGroup()

    def set_params(self, kT=None, tau=None):
        R""" Changes parameters of an existing integrator.

        Args:
            kT (float): New temperature (if set) (in energy units)
            tau (float): New coupling constant (if set) (in time units)

        Examples::

            integrator.set_params(tau=0.6)
            integrator.set_params(tau=0.7, kT=2.0)

        """
        hoomd.util.print_status_line();
        self.check_initialization();

        # change the parameters
        if kT is not None:
            # setup the variant inputs
            kT = hoomd.variant._setup_variant_input(kT);
            self.cpp_method.setT(kT.cpp_variant);
            self.kT = kT

        if tau is not None:
            self.cpp_method.setTau(tau);
            self.tau = tau

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
            Randomization is applied at the start of the next call to :py:func:`hoomd.run`.

        Example::

            integrator = md.integrate.nvt(group=group.all(), kT=1.0, tau=0.5)
            integrator.randomize_velocities(seed=42)
            run(100)

        """
        timestep = hoomd.get_step()
        kT = self.kT.cpp_variant.getValue(timestep)
        self.cpp_method.setRandomizeVelocitiesParams(kT, seed)

class npt(_integration_method):
    R""" NPT Integration via MTK barostat-thermostat.

    Args:
        group (:py:mod:`hoomd.group`): Group of particles on which to apply this method.
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

    :py:class:`npt` is an integration method. It must be used with :py:class:`mode_standard`.

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
        hoomd.util.print_status_line();

        # check the input
        if (kT is None or tau is None):
            if nph is False:
                hoomd.context.msg.error("integrate.npt: Need temperature T and thermostat time scale tau.\n");
                raise RuntimeError("Error setting up NPT integration.");
            else:
                # use dummy values
                kT=1.0
                tau=1.0

        if (tauP is None):
                hoomd.context.msg.error("integrate.npt: Need barostat time scale tauP.\n");
                raise RuntimeError("Error setting up NPT integration.");

        # initialize base class
        _integration_method.__init__(self);

        # setup the variant inputs
        kT = hoomd.variant._setup_variant_input(kT);

        # If P is set
        if P is not None:
            self.S = [P,P,P,0,0,0]
        else:
            # S is a stress, should be [xx, yy, zz, yz, xz, xy]
            if S is not None and len(S)==6:
                self.S = S
            else:
                raise RuntimeError("Unrecognized stress tensor form");

        S = [hoomd.variant._setup_variant_input(self.S[i]) for i in range(6)]

        Svar = [S[i].cpp_variant for i in range(6)]

        # create the compute thermo for half time steps
        if group is hoomd.context.current.group_all:
            group_copy = copy.copy(group);
            group_copy.name = "__npt_all";
            hoomd.util.quiet_status();
            thermo_group = hoomd.compute.thermo(group_copy);
            thermo_group.cpp_compute.setLoggingEnabled(False);
            hoomd.util.unquiet_status();
        else:
            thermo_group = hoomd.compute._get_unique_thermo(group=group);

        # create the compute thermo for full time step
        thermo_group_t = hoomd.compute._get_unique_thermo(group=group);

        # need to know if we are running 2D simulations
        twod = (hoomd.context.current.system_definition.getNDimensions() == 2);
        if twod:
            hoomd.context.msg.notice(2, "When running in 2D, z couplings and degrees of freedom are silently ignored.\n");

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
                hoomd.context.msg.error("Invalid coupling mode\n");
                raise RuntimeError("Error setting up NPT integration.");
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
                hoomd.context.msg.error("Invalid coupling mode\n");
                raise RuntimeError("Error setting up NPT integration.");

        # set degrees of freedom flags
        # silently ignore z related degrees of freedom when running in 2d
        flags = 0;
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

        if not hoomd.context.exec_conf.isCUDAEnabled():
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
                                                nph);
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
                                                   nph);

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
        hoomd.util.print_status_line();
        self.check_initialization();

        # change the parameters
        if kT is not None:
            # setup the variant inputs
            kT = hoomd.variant._setup_variant_input(kT);
            self.kT = kT
            self.cpp_method.setT(kT.cpp_variant);
        if tau is not None:
            self.cpp_method.setTau(tau);
            self.tau = tau

        # If P is set
        if P is not None:
            self.S = [P,P,P,0,0,0]
        elif S is not None:
            # S is a stress, should be [xx, yy, zz, yz, xz, xy]
            if (len(S)==6):
                self.S = S
            else:
                raise RuntimeError("Unrecognized stress tensor form");

        if P is not None or S is not None:
            S = [hoomd.variant._setup_variant_input(self.S[i]) for i in range(6)]

            Svar = [S[i].cpp_variant for i in range(6)]
            self.cpp_method.setS(Svar)

        if tauP is not None:
            self.cpp_method.setTauP(tauP);
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
        data = _integration_method.get_metadata(self)
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
            Randomization is applied at the start of the next call to :py:func:`hoomd.run`.

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

    :py:class:`nph` is an integration method and must be used with :py:class:`mode_standard`.

    Examples::

        # Triclinic unit cell
        nph=integrate.nph(group=all, P=2.0, tauP=1.0, couple="none", all=True)
        # Cubic unit cell
        nph = integrate.nph(group=all, P=2.0, tauP=1.0)
        # Relax the box
        nph = integrate.nph(group=all, P=0, tauP=1.0, gamma=0.1)
    """
    def __init__(self, **params):
        hoomd.util.print_status_line();

        # initialize base class
        hoomd.util.quiet_status();
        npt.__init__(self, nph=True, kT=1.0, **params);
        hoomd.util.unquiet_status();

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
            Randomization is applied at the start of the next call to :py:func:`hoomd.run`.

        Example::

            integrator = md.integrate.nph(group=group.all(), P=2.0, tauP=1.0)
            integrator.randomize_velocities(kT=1.0, seed=42)
            run(100)

        """
        self.cpp_method.setRandomizeVelocitiesParams(kT, seed)

class nve(_integration_method):
    R""" NVE Integration via Velocity-Verlet

    Args:
        group (:py:mod:`hoomd.group`): Group of particles on which to apply this method.
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

    :py:class:`nve` is an integration method. It must be used with :py:class:`mode_standard`.

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
        hoomd.util.print_status_line();

        # initialize base class
        _integration_method.__init__(self);

        # create the compute thermo
        hoomd.compute._get_unique_thermo(group=group);

        # initialize the reflected c++ class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_method = _md.TwoStepNVE(hoomd.context.current.system_definition, group.cpp_group, False);
        else:
            self.cpp_method = _md.TwoStepNVEGPU(hoomd.context.current.system_definition, group.cpp_group);

        # set the limit
        if limit is not None:
            self.cpp_method.setLimit(limit);

        self.cpp_method.setZeroForce(zero_force);

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
        hoomd.util.print_status_line();
        self.check_initialization();

        # change the parameters
        if limit is not None:
            if limit == False:
                self.cpp_method.removeLimit();
            else:
                self.cpp_method.setLimit(limit);
            self.limit = limit

        if zero_force is not None:
            self.cpp_method.setZeroForce(zero_force);

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
            Randomization is applied at the start of the next call to :py:func:`hoomd.run`.

        Example::

            integrator = md.integrate.nve(group=group.all())
            integrator.randomize_velocities(kT=1.0, seed=42)
            run(100)

        """
        self.cpp_method.setRandomizeVelocitiesParams(kT, seed)

class langevin(_integration_method):
    R""" Langevin dynamics.

    Args:
        group (:py:mod:`hoomd.group`): Group of particles to apply this method to.
        kT (:py:mod:`hoomd.variant` or :py:obj:`float`): Temperature of the simulation (in energy units).
        seed (int): Random seed to use for generating :math:`\vec{F}_\mathrm{R}`.
        dscale (bool): Control :math:`\lambda` options. If 0 or False, use :math:`\gamma` values set per type. If non-zero, :math:`\gamma = \lambda d_i`.
        tally (bool): (optional) If true, the energy exchange between the thermal reservoir and the particles is
                            tracked. Total energy conservation can then be monitored by adding
                            ``langevin_reservoir_energy_groupname`` to the logged quantities.
        noiseless_t (bool): If set true, there will be no translational noise (random force)
        noiseless_r (bool): If set true, there will be no rotational noise (random torque)

    .. rubric:: Translational degrees of freedom

    :py:class:`langevin` integrates particles forward in time according to the Langevin equations of motion:

    .. math::

        m \frac{d\vec{v}}{dt} = \vec{F}_\mathrm{C} - \gamma \cdot \vec{v} + \vec{F}_\mathrm{R}

        \langle \vec{F}_\mathrm{R} \rangle = 0

        \langle |\vec{F}_\mathrm{R}|^2 \rangle = 2 d kT \gamma / \delta t

    where :math:`\vec{F}_\mathrm{C}` is the force on the particle from all potentials and constraint forces,
    :math:`\gamma` is the drag coefficient, :math:`\vec{v}` is the particle's velocity, :math:`\vec{F}_\mathrm{R}`
    is a uniform random force, and :math:`d` is the dimensionality of the system (2 or 3).  The magnitude of
    the random force is chosen via the fluctuation-dissipation theorem
    to be consistent with the specified drag and temperature, :math:`T`.
    When :math:`kT=0`, the random force :math:`\vec{F}_\mathrm{R}=0`.

    :py:class:`langevin` generates random numbers by hashing together the particle tag, user seed, and current
    time step index. See `C. L. Phillips et. al. 2011 <http://dx.doi.org/10.1016/j.jcp.2011.05.021>`_ for more
    information.

    .. attention::
        Change the seed if you reset the simulation time step to 0. If you keep the same seed, the simulation
        will continue with the same sequence of random numbers used previously and may cause unphysical correlations.

        For MPI runs: all ranks other than 0 ignore the seed input and use the value of rank 0.

    Langevin dynamics includes the acceleration term in the Langevin equation and is useful for gently thermalizing
    systems using a small gamma. This assumption is valid when underdamped: :math:`\frac{m}{\gamma} \gg \delta t`.
    Use :py:class:`brownian` if your system is not underdamped.

    :py:class:`langevin` uses the same integrator as :py:class:`nve` with the additional force term
    :math:`- \gamma \cdot \vec{v} + \vec{F}_\mathrm{R}`. The random force :math:`\vec{F}_\mathrm{R}` is drawn
    from a uniform random number distribution.

    You can specify :math:`\gamma` in two ways:

    1. Use :py:class:`set_gamma()` to specify it directly, with independent values for each particle type in the system.
    2. Specify :math:`\lambda` which scales the particle diameter to :math:`\gamma = \lambda d_i`. The units of
       :math:`\lambda` are mass / distance / time.

    :py:class:`langevin` must be used with :py:class:`mode_standard`.

    *kT* can be a variant type, allowing for temperature ramps in simulation runs.

    A :py:class:`hoomd.compute.thermo` is automatically created and associated with *group*.

    Warning:
        When restarting a simulation, the energy of the reservoir will be reset to zero.

    Examples::

        all = group.all();
        integrator = integrate.langevin(group=all, kT=1.0, seed=5)
        integrator = integrate.langevin(group=all, kT=1.0, dscale=1.5, tally=True)
        typeA = group.type('A');
        integrator = integrate.langevin(group=typeA, kT=hoomd.variant.linear_interp([(0, 4.0), (1e6, 1.0)]), seed=10)

    """
    def __init__(self, group, kT, seed, dscale=False, tally=False, noiseless_t=False, noiseless_r=False):
        hoomd.util.print_status_line();

        # initialize base class
        _integration_method.__init__(self);

        # setup the variant inputs
        kT = hoomd.variant._setup_variant_input(kT);

        # create the compute thermo
        hoomd.compute._get_unique_thermo(group=group);

        # setup suffix
        suffix = '_' + group.name;

        if dscale is False or dscale == 0:
            use_lambda = False;
        else:
            use_lambda = True;

        # initialize the reflected c++ class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            my_class = _md.TwoStepLangevin;
        else:
            my_class = _md.TwoStepLangevinGPU;

        self.cpp_method = my_class(hoomd.context.current.system_definition,
                                   group.cpp_group,
                                   kT.cpp_variant,
                                   seed,
                                   use_lambda,
                                   float(dscale),
                                   noiseless_t,
                                   noiseless_r,
                                   suffix);

        self.cpp_method.setTally(tally);

        self.cpp_method.validateGroup()

        # store metadata
        self.group = group
        self.kT = kT
        self.seed = seed
        self.dscale = dscale
        self.noiseless_t = noiseless_t
        self.noiseless_r = noiseless_r
        self.metadata_fields = ['group', 'kT', 'seed', 'dscale','noiseless_t','noiseless_r']

    def set_params(self, kT=None, tally=None):
        R""" Change langevin integrator parameters.

        Args:
            kT (:py:mod:`hoomd.variant` or :py:obj:`float`): New temperature (if set) (in energy units).
            tally (bool): (optional) If true, the energy exchange between the thermal reservoir and the particles is
                                tracked. Total energy conservation can then be monitored by adding
                                ``langevin_reservoir_energy_groupname`` to the logged quantities.

        Examples::

            integrator.set_params(kT=2.0)
            integrator.set_params(tally=False)

        """
        hoomd.util.print_status_line();
        self.check_initialization();

        # change the parameters
        if kT is not None:
            # setup the variant inputs
            kT = hoomd.variant._setup_variant_input(kT);
            self.cpp_method.setT(kT.cpp_variant);
            self.kT = kT

        if tally is not None:
            self.cpp_method.setTally(tally);

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
        hoomd.util.print_status_line();
        self.check_initialization();
        a = str(a);

        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getParticleData().getNameByType(i));

        # change the parameters
        for i in range(0,ntypes):
            if a == type_list[i]:
                self.cpp_method.setGamma(i,gamma);

    def set_gamma_r(self, a, gamma_r):
        R""" Set gamma_r for a particle type.

        Args:
            a (str):  Particle type name
            gamma_r (float or tuple): :math:`\gamma_r` for particle type a (in units of force/velocity), optionally for all body frame directions

        :py:meth:`set_gamma_r()` sets the coefficient :math:`\gamma_r` for a single particle type, identified
        by name. The default is 1.0 if not specified for a type. It must be positive or zero, if set
        zero, it will have no rotational damping or random torque, but still with updates from normal net torque.

        Examples::

            langevin.set_gamma_r('A', gamma_r=2.0)
            langevin.set_gamma_r('A', gamma_r=(1.0,2.0,3.0))

        """

        hoomd.util.print_status_line();
        self.check_initialization();

        if not isinstance(gamma_r,tuple):
            gamma_r = (gamma_r, gamma_r, gamma_r)

        if (gamma_r[0] < 0 or gamma_r[1] < 0 or gamma_r[2] < 0):
            raise ValueError("The gamma_r must be positive or zero (represent no rotational damping or random torque, but with updates)")

        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getParticleData().getNameByType(i));

        # change the parameters
        for i in range(0,ntypes):
            if a == type_list[i]:
                self.cpp_method.setGamma_r(i,_hoomd.make_scalar3(*gamma_r));

class brownian(_integration_method):
    R""" Brownian dynamics.

    Args:
        group (:py:mod:`hoomd.group`): Group of particles to apply this method to.
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
    overdamped: :math:`\frac{m}{\gamma} \ll \delta t`. Use :py:class:`langevin` if your system is not overdamped.

    You can specify :math:`\gamma` in two ways:

    1. Use :py:class:`set_gamma()` to specify it directly, with independent values for each particle type in the system.
    2. Specify :math:`\lambda` which scales the particle diameter to :math:`\gamma = \lambda d_i`. The units of
       :math:`\lambda` are mass / distance / time.

    :py:class:`brownian` must be used with integrate.mode_standard.

    *kT* can be a variant type, allowing for temperature ramps in simulation runs.

    A :py:class:`hoomd.compute.thermo` is automatically created and associated with *group*.

    Examples::

        all = group.all();
        integrator = integrate.brownian(group=all, kT=1.0, seed=5)
        integrator = integrate.brownian(group=all, kT=1.0, dscale=1.5)
        typeA = group.type('A');
        integrator = integrate.brownian(group=typeA, kT=hoomd.variant.linear_interp([(0, 4.0), (1e6, 1.0)]), seed=10)

    """
    def __init__(self, group, kT, seed, dscale=False, noiseless_t=False, noiseless_r=False):
        hoomd.util.print_status_line();

        # initialize base class
        _integration_method.__init__(self);

        # setup the variant inputs
        kT = hoomd.variant._setup_variant_input(kT);

        # create the compute thermo
        hoomd.compute._get_unique_thermo(group=group);

        if dscale is False or dscale == 0:
            use_lambda = False;
        else:
            use_lambda = True;

        # initialize the reflected c++ class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            my_class = _md.TwoStepBD;
        else:
            my_class = _md.TwoStepBDGPU;

        self.cpp_method = my_class(hoomd.context.current.system_definition,
                                   group.cpp_group,
                                   kT.cpp_variant,
                                   seed,
                                   use_lambda,
                                   float(dscale),
                                   noiseless_t,
                                   noiseless_r);

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
        hoomd.util.print_status_line();
        self.check_initialization();

        # change the parameters
        if kT is not None:
            # setup the variant inputs
            kT = hoomd.variant._setup_variant_input(kT);
            self.cpp_method.setT(kT.cpp_variant);
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
        hoomd.util.print_status_line();
        self.check_initialization();
        a = str(a);

        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getParticleData().getNameByType(i));

        # change the parameters
        for i in range(0,ntypes):
            if a == type_list[i]:
                self.cpp_method.setGamma(i,gamma);

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

        hoomd.util.print_status_line();
        self.check_initialization();

        if not isinstance(gamma_r,tuple):
            gamma_r = (gamma_r, gamma_r, gamma_r)

        if (gamma_r[0] < 0 or gamma_r[1] < 0 or gamma_r[2] < 0):
            raise ValueError("The gamma_r must be positive or zero (represent no rotational damping or random torque, but with updates)")

        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getParticleData().getNameByType(i));

        # change the parameters
        for i in range(0,ntypes):
            if a == type_list[i]:
                self.cpp_method.setGamma_r(i,_hoomd.make_scalar3(*gamma_r));

class mode_minimize_fire(_integrator):
    R""" Energy Minimizer (FIRE).

    Args:
        group (:py:mod:`hoomd.group`): Particle group to apply minimization to.
          Deprecated in version 2.2:
          :py:class:`hoomd.md.integrate.mode_minimize_fire()` now accepts integration methods, such as :py:class:`hoomd.md.integrate.nve()`
          and :py:class:`hoomd.md.integrate.nph()`. The functions operate on user-defined groups. If **group** is defined here,
          automatically :py:class:`hoomd.md.integrate.nve()` will be used for integration
        dt (float): This is the maximum step size the minimizer is permitted to use.  Consider the stability of the system when setting. (in time units)
        Nmin (int): Number of steps energy change is negative before allowing :math:`\alpha` and :math:`\delta t` to adapt.
        finc (float): Factor to increase :math:`\delta t` by
        fdec (float): Factor to decrease :math:`\delta t` by
        alpha_start (float): Initial (and maximum) :math:`\alpha`
        falpha (float): Factor to decrease :math:`\alpha t` by
        ftol (float): force convergence criteria (in units of force over mass)
        wtol (float): angular momentum convergence criteria (in units of angular momentum)
        Etol (float): energy convergence criteria (in energy units)
        min_steps (int): A minimum number of attempts before convergence criteria are considered
        aniso (bool): Whether to integrate rotational degrees of freedom (bool), default None (autodetect).
          Added in version 2.2

    .. versionadded:: 2.1
    .. versionchanged:: 2.2

    :py:class:`mode_minimize_fire` uses the Fast Inertial Relaxation Engine (FIRE) algorithm to minimize the energy
    for a group of particles while keeping all other particles fixed.  This method is published in
    `Bitzek, et. al., PRL, 2006 <http://dx.doi.org/10.1103/PhysRevLett.97.170201>`_.

    At each time step, :math:`\delta t`, the algorithm uses the NVE Integrator to generate a x, v, and F, and then adjusts
    v according to

    .. math::

        \vec{v} = (1-\alpha)\vec{v} + \alpha \hat{F}|\vec{v}|

    where :math:`\alpha` and :math:`\delta t` are dynamically adaptive quantities.  While a current search has been
    lowering the energy of system for more than
    :math:`N_{min}` steps, :math:`\alpha`  is decreased by :math:`\alpha \rightarrow \alpha f_{alpha}` and
    :math:`\delta t` is increased by :math:`\delta t \rightarrow max(\delta t \cdot f_{inc}, \delta t_{max})`.
    If the energy of the system increases (or stays the same), the velocity of the particles is set to 0,
    :math:`\alpha \rightarrow \alpha_{start}` and
    :math:`\delta t \rightarrow \delta t \cdot f_{dec}`.  Convergence is determined by both the force per particle and
    the change in energy per particle dropping below *ftol* and *Etol*, respectively or

    .. math::

        \frac{\sum |F|}{N*\sqrt{N_{dof}}} <ftol \;\; and \;\; \Delta \frac{\sum |E|}{N} < Etol

    where N is the number of particles the minimization is acting over (i.e. the group size)
    Either of the two criterion can be effectively turned off by setting the tolerance to a large number.

    If the minimization is acted over a subset of all the particles in the system, the "other" particles will be kept
    frozen but will still interact with the particles being moved.

    Examples::

        fire=integrate.mode_minimize_fire(dt=0.05, ftol=1e-2, Etol=1e-7)
        nve=integrate.nve(group=group.all())
        while not(fire.has_converged()):
           run(100)

    Examples::

        fire=integrate.mode_minimize_fire(dt=0.05, ftol=1e-2, Etol=1e-7)
        nph=integrate.nph(group=group.all(),P=0.0,gamma=.5)
        while not(fire.has_converged()):
           run(100)

    Note:
        The algorithm requires a base integrator to update the particle position and velocities.
        Usually this will be either NVE (to minimize energy) or NPH (to minimize energy and relax the box).
        The quantity minimized is in any case the energy (not the enthalpy or any other quantity).

    Note:
        As a default setting, the algorithm will start with a :math:`\delta t = \frac{1}{10} \delta t_{max}` and
        attempts at least 10 search steps.  In practice, it was found that this prevents the simulation from making too
        aggressive a first step, but also from quitting before having found a good search direction. The minimum number of
        attempts can be set by the user.

    """
    def __init__(self, dt, Nmin=5, finc=1.1, fdec=0.5, alpha_start=0.1, falpha=0.99, ftol = 1e-1, wtol=1e-1, Etol= 1e-5, min_steps=10, group=None, aniso=None):
        hoomd.util.print_status_line();

        # initialize base class
        _integrator.__init__(self);

        # initialize the reflected c++ class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_integrator = _md.FIREEnergyMinimizer(hoomd.context.current.system_definition, dt);
        else:
            self.cpp_integrator = _md.FIREEnergyMinimizerGPU(hoomd.context.current.system_definition, dt);

        self.supports_methods = True;

        hoomd.context.current.system.setIntegrator(self.cpp_integrator);

        if group is not None:
            hoomd.context.msg.warning("group is deprecated. Creating default integrate.nve().\n")
            integrate_nve = nve(group=group)

        self.aniso = aniso

        hoomd.util.quiet_status();
        if aniso is not None:
            self.set_params(aniso=aniso)
        hoomd.util.unquiet_status();

        # change the set parameters if not None
        self.dt = dt
        self.metadata_fields = ['dt','aniso']

        self.cpp_integrator.setNmin(Nmin);
        self.Nmin = Nmin
        self.metadata_fields.append('Nmin')

        self.cpp_integrator.setFinc(finc);
        self.finc = finc
        self.metadata_fields.append('finc')

        self.cpp_integrator.setFdec(fdec);
        self.fdec = fdec
        self.metadata_fields.append('fdec')

        self.cpp_integrator.setAlphaStart(alpha_start);
        self.alpha_start = alpha_start
        self.metadata_fields.append('alpha_start')

        self.cpp_integrator.setFalpha(falpha);
        self.falpha = falpha
        self.metadata_fields.append(falpha)

        self.cpp_integrator.setFtol(ftol);
        self.ftol = ftol
        self.metadata_fields.append(ftol)

        self.cpp_integrator.setWtol(wtol);
        self.wtol = wtol
        self.metadata_fields.append(wtol)

        self.cpp_integrator.setEtol(Etol);
        self.Etol = Etol
        self.metadata_fields.append(Etol)

        self.cpp_integrator.setMinSteps(min_steps);
        self.min_steps = min_steps
        self.metadata_fields.append(min_steps)

    ## \internal
    #  \brief Cached set of anisotropic mode enums for ease of access
    _aniso_modes = {
        None: _md.IntegratorAnisotropicMode.Automatic,
        True: _md.IntegratorAnisotropicMode.Anisotropic,
        False: _md.IntegratorAnisotropicMode.Isotropic}

    def get_energy(self):
        R""" Returns the energy after the last iteration of the minimizer
        """
        hoomd.util.print_status_line()
        self.check_initialization();
        return self.cpp_integrator.getEnergy()

    def set_params(self, aniso=None):
        R""" Changes parameters of an existing integration mode.

        Args:
            aniso (bool): Anisotropic integration mode (bool), default None (autodetect).

        Examples::

            integrator_mode.set_params(aniso=False)

        """
        hoomd.util.print_status_line();
        self.check_initialization();

        if aniso is not None:
            if aniso in self._aniso_modes:
                anisoMode = self._aniso_modes[aniso]
            else:
                hoomd.context.msg.error("integrate.mode_standard: unknown anisotropic mode {}.\n".format(aniso));
                raise RuntimeError("Error setting anisotropic integration mode.");
            self.aniso = aniso
            self.cpp_integrator.setAnisotropicMode(anisoMode)

    def has_converged(self):
        R""" Test if the energy minimizer has converged.

        Returns:
            True when the minimizer has converged. Otherwise, return False.
        """
        self.check_initialization();
        return self.cpp_integrator.hasConverged()

    def reset(self):
        R""" Reset the minimizer to its initial state.
        """
        self.check_initialization();
        return self.cpp_integrator.reset()

class berendsen(_integration_method):
    R""" Applies the Berendsen thermostat.

    Args:
        group (:py:mod:`hoomd.group`): Group to which the Berendsen thermostat will be applied.
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
        hoomd.util.print_status_line();

        # Error out in MPI simulations
        if (_hoomd.is_MPI_available()):
            if hoomd.context.current.system_definition.getParticleData().getDomainDecomposition():
                hoomd.context.msg.error("integrate.berendsen is not supported in multi-processor simulations.\n\n")
                raise RuntimeError("Error setting up integration method.")

        # initialize base class
        _integration_method.__init__(self);

        # setup the variant inputs
        kT = hoomd.variant._setup_variant_input(kT);

        # create the compute thermo
        thermo = hoomd.compute._get_unique_thermo(group = group);

        # initialize the reflected c++ class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_method = _md.TwoStepBerendsen(hoomd.context.current.system_definition,
                                                     group.cpp_group,
                                                     thermo.cpp_compute,
                                                     tau,
                                                     kT.cpp_variant);
        else:
            self.cpp_method = _md.TwoStepBerendsenGPU(hoomd.context.current.system_definition,
                                                        group.cpp_group,
                                                        thermo.cpp_compute,
                                                        tau,
                                                        kT.cpp_variant);

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
            Randomization is applied at the start of the next call to :py:func:`hoomd.run`.

        Example::

            integrator = md.integrate.berendsen(group=group.all(), kT=1.0, tau=0.5)
            integrator.randomize_velocities(seed=42)
            run(100)

        """
        timestep = hoomd.get_step()
        kT = self.kT.cpp_variant.getValue(timestep)
        self.cpp_method.setRandomizeVelocitiesParams(kT, seed)
