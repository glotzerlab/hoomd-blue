# coding: utf-8

# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: joaander / All Developers are free to add commands for new features


from hoomd.md import _md
import hoomd
from hoomd.operation import _HOOMDBaseObject
from hoomd.data.parameterdicts import ParameterDict, TypeParameterDict
from hoomd.filter import ParticleFilter
from hoomd.data.typeparam import TypeParameter
from hoomd.data.typeconverter import OnlyType, OnlyIf, to_type_converter
from hoomd.variant import Variant
from collections.abc import Sequence


class _Method(_HOOMDBaseObject):
    """Base class integration method.

    Provides common methods for all subclasses.

    Note:
        Users should use the subclasses and not instantiate `_Method` directly.
    """


class NVT(_Method):
    r"""NVT Integration via the Nosé-Hoover thermostat.

    Args:
        filter (`hoomd.filter.ParticleFilter`): Subset of particles on which to
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

        \tau = \sqrt{\frac{Q}{g k T_0}}

    where :math:`g` is the number of degrees of freedom, and :math:`k T_0` is
    the set point (*kT* above).

    The `NVT` equations of motion include a translational thermostat (with
    momentum :math:`\xi` and position :math:`\eta`) and a rotational thermostat
    (with momentum :math:`\xi_{\mathrm{rot}}` and position
    :math:`\eta_\mathrm{rot}`). Access these quantities using
    `translational_thermostat_dof` and `rotational_thermostat_dof`.

    Note:
        Coupling constant `tau` in Nosé-Hoover thermostat should be set within
        reasonable range to avoid abrupt fluctuation in temperature in case of
        small `tau` , also to avoid long time to equilibrate in case of large
        `tau`. Recommended value for most of systems is ``100 * dt``, where
        ``dt`` is the length of the time step.

    .. todo:: Rotational degrees of freedom

        `NVT` integrates rotational degrees of freedom.

    Examples::

        nvt=hoomd.md.methods.NVT(filter=hoomd.filter.All(), kT=1.0, tau=0.5)
        integrator = hoomd.md.Integrator(dt=0.005, methods=[nvt], forces=[lj])


    Attributes:
        filter (hoomd.filter.ParticleFilter): Subset of particles on which to
            apply this method.

        kT (hoomd.variant.Variant): Temperature set point
            for the Nosé-Hoover thermostat. (in energy units).

        tau (float): Coupling constant for the Nosé-Hoover thermostat. (in time
            units).

        translational_thermostat_dof (tuple[float, float]): Additional degrees
            of freedom for the translational thermostat (:math:`\xi`,
            :math:`\eta`)

        rotational_thermostat_dof (tuple[float, float]): Additional degrees
            of freedom for the rotational thermostat (:math:`\xi_\mathrm{rot}`,
            :math:`\eta_\mathrm{rot}`)
    """

    def __init__(self, filter, kT, tau):

        # store metadata
        param_dict = ParameterDict(
            filter=ParticleFilter,
            kT=Variant,
            tau=float(tau),
            translational_thermostat_dof=(float, float),
            rotational_thermostat_dof=(float, float)
        )
        param_dict.update(
            dict(kT=kT,
                 filter=filter,
                 translational_thermostat_dof=(0, 0),
                 rotational_thermostat_dof=(0, 0)))
        # set defaults
        self._param_dict.update(param_dict)

    def _attach(self):

        # initialize the reflected cpp class
        if isinstance(self._simulation.device, hoomd.device.CPU):
            my_class = _md.TwoStepNVTMTK
            thermo_cls = _md.ComputeThermo
        else:
            my_class = _md.TwoStepNVTMTKGPU
            thermo_cls = _md.ComputeThermoGPU

        group = self._simulation.state._get_group(self.filter)
        cpp_sys_def = self._simulation.state._cpp_sys_def
        thermo = thermo_cls(cpp_sys_def, group, "")
        self._cpp_obj = my_class(cpp_sys_def,
                                 group,
                                 thermo,
                                 self.tau,
                                 self.kT,
                                 "")
        super()._attach()

    def thermalize_thermostat_dof(self, seed):
        r"""Set the thermostat momenta to random values.

        Args:
            seed (int): Random number seed

        `thermalize_extra_dof` sets a random value for the momentum :math:`\xi`.
        When `Integrator.aniso` is `True`, it also sets a random value for the
        rotational thermostat momentum :math:`\xi_{\mathrm{rot}}`. Call
        `thermalize_extra_dof` to set a new random state for the thermostat.

        .. important::
            You must call `Simulation.run` before `thermalize_extra_dof`.
            Call ``run(steps=0)`` to prepare a newly created `Simulation`.

        .. seealso:: `State.thermalize_particle_momenta`

        Note:
            The seed for the pseudorandom number stream includes the
            simulation timestep and the provided *seed*.
        """
        if not self._attached:
            raise RuntimeError(
                "Call Simulation.run(0) before thermalize_thermostat_dof")

        self._cpp_obj.thermalizeThermostatDOF(seed, self._simulation.timestep)


class NPT(_Method):
    R""" NPT Integration via MTK barostat-thermostat.

    Args:
        filter (`hoomd.filter.ParticleFilter`): Subset of particles on which to
            apply this method.

        kT (`hoomd.variant.Variant` or `float`): Temperature set point for the
            thermostat. (in energy units).

        tau (`float`): Coupling constant for the thermostat (in time units).

        S (`list` [ `hoomd.variant.Variant` ] or `float`): Stress components set
            point for the barostat (in pressure units).  In Voigt notation:
            :math:`[S_{xx}, S_{yy}, S_{zz}, S_{yz}, S_{xz}, S_{xy}]`.  In case
            of isotropic pressure P (:math:`[p, p, p, 0, 0, 0]`), use ``S = p``.

        tauS (`float`): Coupling constant for the barostat (in time units).

        couple (`str`): Couplings of diagonal elements of the stress tensor,
            can be "none", "xy", "xz","yz", or "all", default to "all".

        box_dof(`list` [ `bool` ]): Box degrees of freedom with six boolean
            elements corresponding to x, y, z, xy, xz, yz, each. Default to
            [True,True,True,False,False,False]). If turned on to True,
            rescale corresponding lengths or tilt factors and components of
            particle coordinates and velocities.

        rescale_all (`bool`): if True, rescale all particles, not only those in
            the group, Default to False.

        gamma (`float`): Dimensionless damping factor for the box degrees of
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
    rescaled. The ``box_dof`` tuple controls the way the box is rescaled and
    updated. The first three elements ``box_dof[:3]`` controls whether the x, y,
    and z box lengths are rescaled and updated, respectively. The last three
    entries ``box_dof[3:]`` control the rescaling or the tilt factors xy, xz,
    and yz. All options also appropriately rescale particle coordinates and
    velocities.

    By default, the x, y, and z degrees of freedom are updated.
    ``[True,True,True,False,False,False]``

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


    For the MTK equations of motion, see:

    * `G. J. Martyna, D. J. Tobias, M. L. Klein  1994 <http://dx.doi.org/10.1063/1.467468>`_
    * `M. E. Tuckerman et. al. 2006 <http://dx.doi.org/10.1088/0305-4470/39/19/S18>`_
    * `T. Yu et. al. 2010 <http://dx.doi.org/10.1016/j.chemphys.2010.02.014>`_
    *  Glaser et. al (2013), unpublished


    :math:`\tau` is related to the Nosé mass :math:`Q` by

    .. math::

        \tau = \sqrt{\frac{Q}{g k T_0}}

    where :math:`g` is the number of degrees of freedom, and :math:`k T_0` is
    the set point (*kT* above).

    The `NPT` equations of motion include a translational thermostat (with
    momentum :math:`\xi` and position :math:`\eta`), a rotational thermostat
    (with momentum :math:`\xi_{\mathrm{rot}}` and position
    :math:`\eta_\mathrm{rot}`), and a barostat tensor :math:`\nu_{\mathrm{ij}}`.
    Access these quantities using `translational_thermostat_dof`,
    `rotational_thermostat_dof`, and `barostat_dof`.

    Note:
        Coupling constant for barostat `tauS` should be set within appropriate
        range for pressure and volume to fluctuate in reasonable rate and
        equilibrate. Too small `tauS` can cause abrupt fluctuation, whereas too
        large `tauS` would take long time to equilibrate. In most of systems,
        recommended value for `tauS` is ``1000 * dt``, where ``dt`` is the
        length of the time step.

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
        filter (hoomd.filter.ParticleFilter): Subset of particles on which to
            apply this method.

        kT (hoomd.variant.Variant): Temperature set point for the
            thermostat. (in energy units).

        tau (float): Coupling constant for the thermostat (in time units).

        S (List[hoomd.variant.Variant]): Stress components set
            point for the barostat (in pressure units).
            In Voigt notation,
            :math:`[S_{xx}, S_{yy}, S_{zz}, S_{yz}, S_{xz}, S_{xy}]`. Stress can
            be reset after method object is created. For example, An isoropic
            pressure can be set by ``npt.S = 4.``

        tauS (float): Coupling constant for the barostat (in time units).

        couple (str): Couplings of diagonal elements of the stress tensor,
            can be "none", "xy", "xz","yz", or "all".

        box_dof(List[bool]): Box degrees of freedom with six boolean elements
            corresponding to x, y, z, xy, xz, yz, each.

        rescale_all (bool): if True, rescale all particles, not only those in
            the group.

        gamma (float): Dimensionless damping factor for the box degrees of
            freedom.

        translational_thermostat_dof (tuple[float, float]): Additional degrees
            of freedom for the translational thermostat (:math:`\xi`,
            :math:`\eta`)

        rotational_thermostat_dof (tuple[float, float]): Additional degrees
            of freedom for the rotational thermostat (:math:`\xi_\mathrm{rot}`,
            :math:`\eta_\mathrm{rot}`)

        barostat_dof (tuple[float, float, float, float, float, float]):
            Additional degrees of freedom for the barostat (:math:`\nu_{xx}`,
            :math:`\nu_{xy}`, :math:`\nu_{xz}`, :math:`\nu_{yy}`,
            :math:`\nu_{yz}`, :math:`\nu_{zz}`)
    """
    def __init__(self, filter, kT, tau, S, tauS, couple, box_dof=[True,True,True,False,False,False], rescale_all=False, gamma=0.0):


        # store metadata
        param_dict = ParameterDict(
            filter=ParticleFilter,
            kT=Variant,
            tau=float(tau),
            S=OnlyIf(to_type_converter((Variant,)*6), preprocess=self.__preprocess_stress),
            tauS=float(tauS),
            couple=str(couple),
            box_dof=(bool,)*6,
            rescale_all=bool(rescale_all),
            gamma=float(gamma),
            translational_thermostat_dof=(float, float),
            rotational_thermostat_dof=(float, float),
            barostat_dof=(float, float, float, float, float, float)
            )
        param_dict.update(
            dict(filter=filter,
                 kT=kT,
                 S=S,
                 couple=couple,
                 box_dof=box_dof,
                 translational_thermostat_dof=(0, 0),
                 rotational_thermostat_dof=(0, 0),
                 barostat_dof=(0, 0, 0, 0, 0, 0)))

        # set defaults
        self._param_dict.update(param_dict)


    def _attach(self):
        # initialize the reflected c++ class
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cpp_cls = _md.TwoStepNPTMTK
            thermo_cls = _md.ComputeThermo
        else:
            cpp_cls = _md.TwoStepNPTMTKGPU
            thermo_cls = _md.ComputeThermoGPU

        cpp_sys_def = self._simulation.state._cpp_sys_def
        thermo_group = self._simulation.state._get_group(self.filter)

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

    def thermalize_thermostat_and_barostat_dof(self, seed):
        r"""Set the thermostat and barostat momenta to random values.

        Args:
            seed (int): Random number seed

        `thermalize_thermostat_and_barostat_dof` sets a random value for the
        momentum :math:`\xi` and the barostat :math:`\nu_{\mathrm{ij}}`. When
        `Integrator.aniso` is `True`, it also sets a random value for the
        rotational thermostat momentum :math:`\xi_{\mathrm{rot}}`. Call
        `thermalize_thermostat_and_barostat_dof` to set a new random state for
        the thermostat and barostat.

        .. important::
            You must call `Simulation.run` before
            `thermalize_thermostat_and_barostat_dof`. Call ``run(steps=0)`` to
            prepare a newly created `Simulation`.

        .. seealso:: `State.thermalize_particle_momenta`

        Note:
            The seed for the pseudorandom number stream includes the
            simulation timestep and the provided *seed*.
        """
        if not self._attached:
            raise RuntimeError(
                "Call Simulation.run(0) before"
                "thermalize_thermostat_and_barostat_dof")

        self._cpp_obj.thermalizeThermostatAndBarostatDOF(
            seed, self._simulation.timestep)


class NPH(_Method):
    r"""NPH Integration via MTK barostat-thermostat.

    Args:
        filter (`hoomd.filter.ParticleFilter`): Subset of particles on which to
            apply this method.

        S (`list` [ `hoomd.variant.Variant` ] or `float`): Stress components set
            point for the barostat (in pressure units).  In Voigt notation:
            :math:`[S_{xx}, S_{yy}, S_{zz}, S_{yz}, S_{xz}, S_{xy}]`.  In case
            of isotropic pressure P (:math:`[p, p, p, 0, 0, 0]`), use ``S = p``.

        tauS (`float`): Coupling constant for the barostat (in time units).

        couple (`str`): Couplings of diagonal elements of the stress tensor,
            can be "none", "xy", "xz","yz", or "all", default to "all".

        box_dof(`list` [ `bool` ]): Box degrees of freedom with six boolean
            elements corresponding to x, y, z, xy, xz, yz, each. Default to
            [True,True,True,False,False,False]). If turned on to True,
            rescale corresponding lengths or tilt factors and components of
            particle coordinates and velocities.

        rescale_all (`bool`): if True, rescale all particles, not only those in
            the group, Default to False.

        gamma (`float`): Dimensionless damping factor for the box degrees of
            freedom, Default to 0.

    :py:class:`NPH` performs constant pressure, constant enthalpy (NPH) simulations using a Martyna-Tobias-Klein barostat, an
    explicitly reversible and measure-preserving integration scheme. It allows for fully deformable simulation
    cells and uses the same underlying integrator as :py:class:`NPT` (with *nph=True*).

    Note:
        Coupling constant for barostat `tauS` should be set within appropriate
        range for pressure and volume to fluctuate in reasonable rate and
        equilibrate. Too small `tauS` can cause abrupt fluctuation, whereas too
        large `tauS` would take long time to equilibrate. In most of systems,
        recommended value for `tauS` is ``1000 * dt``, where ``dt`` is the
        length of the time step.

    Examples::

        nph = hoomd.md.methods.NPH(filter=hoomd.filter.All(), tauS = 1.2, S=2.0)
        # orthorhombic symmetry
        nph = hoomd.md.methods.NPH(filter=hoomd.filter.All(), tauS = 1.2, S=2.0, couple="none")
        # tetragonal symmetry
        nph = hoomd.md.methods.NPH(filter=hoomd.filter.All(), tauS = 1.2, S=2.0, couple="xy")
        # triclinic symmetry
        nph = hoomd.md.methods.NPH(filter=hoomd.filter.All(), tauS = 1.2, S=2.0, couple="none", rescale_all=True)
        integrator = hoomd.md.Integrator(dt=0.005, methods=[nph], forces=[lj])

    Attributes:
        filter (hoomd.filter.ParticleFilter): Subset of particles on which to
            apply this method.

        S (List[hoomd.variant.Variant]): Stress components set
            point for the barostat (in pressure units).
            In Voigt notation,
            :math:`[S_{xx}, S_{yy}, S_{zz}, S_{yz}, S_{xz}, S_{xy}]`. Stress can
            be reset after method object is created. For example, An isoropic
            pressure can be set by ``npt.S = 4.``

        tauS (float): Coupling constant for the barostat (in time units).

        couple (str): Couplings of diagonal elements of the stress tensor,
            can be "none", "xy", "xz","yz", or "all".

        box_dof(List[bool]): Box degrees of freedom with six boolean elements
            corresponding to x, y, z, xy, xz, yz, each.

        rescale_all (bool): if True, rescale all particles, not only those in
            the group.

        gamma (float): Dimensionless damping factor for the box degrees of
            freedom.

        translational_thermostat_dof (tuple[float, float]): Additional degrees
            of freedom for the translational thermostat (:math:`\xi`,
            :math:`\eta`)

        rotational_thermostat_dof (tuple[float, float]): Additional degrees
            of freedom for the rotational thermostat (:math:`\xi_\mathrm{rot}`,
            :math:`\eta_\mathrm{rot}`)

        barostat_dof (tuple[float, float, float, float, float, float]):
            Additional degrees of freedom for the barostat (:math:`\nu_{xx}`,
            :math:`\nu_{xy}`, :math:`\nu_{xz}`, :math:`\nu_{yy}`,
            :math:`\nu_{yz}`, :math:`\nu_{zz}`)
    """
    def __init__(self, filter, S, tauS, couple, box_dof=[True,True,True,False,False,False], rescale_all=False, gamma=0.0):
        # store metadata
        param_dict = ParameterDict(
            filter=ParticleFilter,
            kT=Variant,
            S=OnlyIf(to_type_converter((Variant,)*6), preprocess=self.__preprocess_stress),
            tauS=float(tauS),
            couple=str(couple),
            box_dof=(bool,)*6,
            rescale_all=bool(rescale_all),
            gamma=float(gamma),
            translational_thermostat_dof=(float, float),
            rotational_thermostat_dof=(float, float),
            barostat_dof=(float, float, float, float, float, float))

        param_dict.update(
            dict(filter=filter,
                 kT=hoomd.variant.Constant(1.0),
                 S=S,
                 couple=couple,
                 box_dof=box_dof,
                 translational_thermostat_dof=(0, 0),
                 rotational_thermostat_dof=(0, 0),
                 barostat_dof=(0, 0, 0, 0, 0, 0)))

        # set defaults
        self._param_dict.update(param_dict)

    def _attach(self):
        # initialize the reflected c++ class
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cpp_cls = _md.TwoStepNPTMTK
            thermo_cls = _md.ComputeThermo
        else:
            cpp_cls = _md.TwoStepNPTMTKGPU
            thermo_cls = _md.ComputeThermoGPU

        cpp_sys_def = self._simulation.state._cpp_sys_def
        thermo_group = self._simulation.state._get_group(self.filter)

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
                                1.0,
                                self.tauS,
                                self.kT,
                                self.S,
                                self.couple,
                                self.box_dof,
                                True)

        # Attach param_dict and typeparam_dict
        super()._attach()

    def __preprocess_stress(self, value):
        if isinstance(value, Sequence):
            if len(value) != 6:
                raise ValueError(
                    "Expected a single hoomd.variant.Variant / float or six.")
            return tuple(value)
        else:
            return (value, value, value, 0, 0, 0)

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

        constant_s = [hoomd.variant.Constant(1.0),
                      hoomd.variant.Constant(2.0),
                      hoomd.variant.Constant(3.0),
                      hoomd.variant.Constant(0.125),
                      hoomd.variant.Constant(.25),
                      hoomd.variant.Constant(.5)]
        nph = hoomd.md.methods.NPH(filter=hoomd.filter.All(),
                                   tau=2.0,
                                   S=constant_s,
                                   tauS=2.0,
                                   couple='xyz')
            run(100)

        """
        self._cpp_obj.setRandomizeVelocitiesParams(kT, seed)


class NVE(_Method):
    R""" NVE Integration via Velocity-Verlet

    Args:
        filter (`hoomd.filter.ParticleFilter`): Subset of particles on which to
         apply this method.

        limit (None or `float`): Enforce that no particle moves more than a
            distance of a limit in a single time step. Defaults to None

    `NVE` performs constant volume, constant energy simulations using
    the standard Velocity-Verlet method. For poor initial conditions that
    include overlapping atoms, a limit can be specified to the movement a
    particle is allowed to make in one time step. After a few thousand time
    steps with the limit set, the system should be in a safe state to continue
    with unconstrained integration.

    .. todo::
        Update when zero momentum updater is added.

    Examples::

        nve = hoomd.md.methods.NVE(filter=hoomd.filter.All())
        integrator = hoomd.md.Integrator(dt=0.005, methods=[nve], forces=[lj])


    Attributes:
        filter (hoomd.filter.ParticleFilter): Subset of particles on which to
            apply this method.

        limit (None or float): Enforce that no particle moves more than a
            distance of a limit in a single time step. Defaults to None

    """

    def __init__(self, filter, limit=None):

        # store metadata
        param_dict = ParameterDict(
            filter=ParticleFilter,
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
                                        self._simulation.state._get_group(self.filter), False)
        else:
            self._cpp_obj = _md.TwoStepNVEGPU(self._simulation.state._cpp_sys_def,
                                 self._simulation.state._get_group(self.filter))

        # Attach param_dict and typeparam_dict
        super()._attach()


class Langevin(_Method):
    R""" Langevin dynamics.

    Args:
        filter (`hoomd.filter.ParticleFilter`): Subset of particles to
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
        filter (hoomd.filter.ParticleFilter): Subset of particles to
            apply this method to.

        kT (hoomd.variant.Variant): Temperature of the
            simulation (in energy units).

        seed (int): Random seed to use for generating
            :math:`\vec{F}_\mathrm{R}`.

        alpha (float): When set, use :math:`\alpha d_i` for the drag
            coefficient where :math:`d_i` is particle diameter.
            Defaults to None.

        gamma (TypeParameter[ ``particle type``, `float` ]): The drag
            coefficient can be directly set instead of the ratio of particle
            diameter (:math:`\gamma = \alpha d_i`). The type of ``gamma``
            parameter is either positive float or zero.

        gamma_r (TypeParameter[ ``particle type``, [ `float`, `float` , `float` ]]):
            The rotational drag coefficient can be set. The type of ``gamma_r``
            parameter is a tuple of three float. The type of each element of
            tuple is either positive float or zero.

    """

    def __init__(self, filter, kT, seed, alpha=None,
                 tally_reservoir_energy=False):

        # store metadata
        param_dict = ParameterDict(
            filter=ParticleFilter,
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
                                 self._simulation.state._get_group(self.filter),
                                 self.kT, self.seed)

        # Attach param_dict and typeparam_dict
        super()._attach()


class Brownian(_Method):
    R""" Brownian dynamics.

    Args:
        filter (`hoomd.filter.ParticleFilter`): Subset of particles to
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
        filter (hoomd.filter.ParticleFilter): Subset of particles to
            apply this method to.

        kT (hoomd.variant.Variant): Temperature of the
            simulation (in energy units).

        seed (int): Random seed to use for generating
            :math:`\vec{F}_\mathrm{R}`.

        alpha (float): When set, use :math:`\alpha d_i` for the drag
            coefficient where :math:`d_i` is particle diameter.
            Defaults to None.

        gamma (TypeParameter[ ``particle type``, `float` ]): The drag
            coefficient can be directly set instead of the ratio of particle
            diameter (:math:`\gamma = \alpha d_i`). The type of ``gamma``
            parameter is either positive float or zero.

        gamma_r (TypeParameter[ ``particle type``, [ `float`, `float`, `float` ] ]):
            The rotational drag coefficient can be set. The type of ``gamma_r``
            parameter is a tuple of three float. The type of each element of
            tuple is either positive float or zero.
    """

    def __init__(self, filter, kT, seed, alpha=None):

        # store metadata
        param_dict = ParameterDict(
            filter=ParticleFilter,
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
                                          sim.state._get_group(self.filter),
                                          self.kT, self.seed)
        else:
            self._cpp_obj = _md.TwoStepBDGPU(sim.state._cpp_sys_def,
                                             sim.state._get_group(self.filter),
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
        if (hoomd.version.mpi_enabled):
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
