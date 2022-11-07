# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""MD integration methods."""

from hoomd.md import _md
import hoomd
from hoomd.operation import AutotunedObject
from hoomd.data.parameterdicts import ParameterDict, TypeParameterDict
from hoomd.data.typeparam import TypeParameter
from hoomd.data.typeconverter import OnlyTypes, OnlyIf, to_type_converter
from hoomd.filter import ParticleFilter
from hoomd.variant import Variant
from collections.abc import Sequence


class Method(AutotunedObject):
    """Base class integration method.

    Provides common methods for all subclasses.

    Note:
        Users should use the subclasses and not instantiate `Method` directly.
    """

    def _attach_hook(self):
        self._simulation.state.update_group_dof()

    def _detach_hook(self):
        self._simulation.state.update_group_dof()


class NVT(Method):
    r"""Constant volume, constant temperature dynamics.

    Args:
        filter (hoomd.filter.filter_like): Subset of particles on which to apply
            this method.

        kT (hoomd.variant.variant_like): Temperature set point
            for the Nosé-Hoover thermostat :math:`[\mathrm{energy}]`.

        tau (float): Coupling constant for the Nosé-Hoover thermostat
            :math:`[\mathrm{time}]`.

    `NVT` integrates integrates translational and rotational degrees of freedom
    in the canonical ensemble using the Nosé-Hoover thermostat. The thermostat
    is introduced as additional degrees of freedom in the Hamiltonian that
    couple with the velocities and angular momenta of the particles.

    The translational thermostat has a momentum :math:`\xi` and position
    :math:`\eta`. The rotational thermostat has momentum
    :math:`\xi_{\mathrm{rot}}` and position :math:`\eta_\mathrm{rot}`. Access
    these quantities using `translational_thermostat_dof` and
    `rotational_thermostat_dof`.

    `NVT` numerically integrates the equations of motion using the symplectic
    Martyna-Tobias-Klein formalism described refs. `G. J. Martyna, D. J.
    Tobias, M. L. Klein 1994 <http://dx.doi.org/10.1063/1.467468>`_ and `J.
    Cao, G. J. Martyna 1996 <http://dx.doi.org/10.1063/1.470959>`_.

    Note:
        The coupling constant `tau` should be set within a
        reasonable range to avoid abrupt fluctuations in the kinetic temperature
        and to avoid long time to equilibration. The recommended value for most
        systems is :math:`\tau = 100 \delta t`.

    Important:
        Ensure that your initial condition includes non-zero particle velocities
        and angular momenta (when appropriate). The coupling between the
        thermostat and the velocities and angular momenta occurs via
        multiplication, so `NVT` cannot convert a zero velocity into a non-zero
        one except through particle collisions.

    Examples::

        nvt=hoomd.md.methods.NVT(filter=hoomd.filter.All(), kT=1.0, tau=0.5)
        integrator = hoomd.md.Integrator(dt=0.005, methods=[nvt], forces=[lj])

    Attributes:
        filter (hoomd.filter.filter_like): Subset of particles on which to apply
            this method.

        kT (hoomd.variant.Variant): Temperature set point
            for the Nosé-Hoover thermostat :math:`[\mathrm{energy}]`.

        tau (float): Coupling constant for the Nosé-Hoover thermostat
            :math:`[\mathrm{time}]`.

        translational_thermostat_dof (tuple[float, float]): Additional degrees
            of freedom for the translational thermostat (:math:`\xi`,
            :math:`\eta`)

        rotational_thermostat_dof (tuple[float, float]): Additional degrees
            of freedom for the rotational thermostat (:math:`\xi_\mathrm{rot}`,
            :math:`\eta_\mathrm{rot}`)
    """

    def __init__(self, filter, kT, tau):

        # store metadata
        param_dict = ParameterDict(filter=ParticleFilter,
                                   kT=Variant,
                                   tau=float(tau),
                                   translational_thermostat_dof=(float, float),
                                   rotational_thermostat_dof=(float, float))
        param_dict.update(
            dict(kT=kT,
                 filter=filter,
                 translational_thermostat_dof=(0, 0),
                 rotational_thermostat_dof=(0, 0)))
        # set defaults
        self._param_dict.update(param_dict)

    def _attach_hook(self):

        # initialize the reflected cpp class
        if isinstance(self._simulation.device, hoomd.device.CPU):
            my_class = _md.TwoStepNVTMTK
            thermo_cls = _md.ComputeThermo
        else:
            my_class = _md.TwoStepNVTMTKGPU
            thermo_cls = _md.ComputeThermoGPU

        group = self._simulation.state._get_group(self.filter)
        cpp_sys_def = self._simulation.state._cpp_sys_def
        thermo = thermo_cls(cpp_sys_def, group)
        self._cpp_obj = my_class(cpp_sys_def, group, thermo, self.tau, self.kT)
        super()._attach_hook()

    def thermalize_thermostat_dof(self):
        r"""Set the thermostat momenta to random values.

        `thermalize_thermostat_dof` sets a random value for the momentum
        :math:`\xi`. When `Integrator.integrate_rotational_dof` is `True`, it
        also sets a random value for the rotational thermostat momentum
        :math:`\xi_{\mathrm{rot}}`. Call `thermalize_thermostat_dof` to set a
        new random state for the thermostat.

        .. important::
            You must call `Simulation.run` before `thermalize_thermostat_dof`.
            Call ``run(steps=0)`` to prepare a newly created `hoomd.Simulation`.

        .. seealso:: `State.thermalize_particle_momenta`
        """
        if not self._attached:
            raise RuntimeError(
                "Call Simulation.run(0) before thermalize_thermostat_dof")

        self._simulation._warn_if_seed_unset()
        self._cpp_obj.thermalizeThermostatDOF(self._simulation.timestep)

    @hoomd.logging.log(requires_run=True)
    def thermostat_energy(self):
        """Energy the thermostat contributes to the Hamiltonian \
        :math:`[\\mathrm{energy}]`."""
        return self._cpp_obj.getThermostatEnergy(self._simulation.timestep)


class NPH(Method):
    """Constant pressure, constant enthalpy dynamics.

    Args:
        filter (hoomd.filter.filter_like): Subset of particles on which to apply
            this method.

        S (tuple[hoomd.variant.variant_like, ...] or \
                hoomd.variant.variant_like): Stress components set point for
            the barostat.

            In Voigt notation:
            :math:`[S_{xx}, S_{yy}, S_{zz}, S_{yz}, S_{xz}, S_{xy}]`
            :math:`[\\mathrm{pressure}]`. In case of isotropic
            pressure P (:math:`[p, p, p, 0, 0, 0]`), use ``S = p``.

        tauS (float): Coupling constant for the barostat
           :math:`[\\mathrm{time}]`.

        couple (str): Couplings of diagonal elements of the stress tensor,
            can be "none", "xy", "xz","yz", or "all", default to "all".

        box_dof(`tuple` [ `bool` ]): Box degrees of freedom with six boolean
            elements corresponding to x, y, z, xy, xz, yz, each. Default to
            [True,True,True,False,False,False]). If turned on to True,
            rescale corresponding lengths or tilt factors and components of
            particle coordinates and velocities.

        rescale_all (bool): if True, rescale all particles, not only those in
            the group, Default to False.

        gamma (float): Dimensionless damping factor for the box degrees of
            freedom, Default to 0.

    `NPH` integrates translational and rotational degrees of freedom forward in
    time in the Isoenthalpic-isobaric ensemble. The barostat is introduced as
    additional degrees of freedom in the Hamiltonian that couple with the box
    parameters.

    The barostat tensor is :math:`\\nu_{\\mathrm{ij}}`. Access these quantities
    `barostat_dof`.

    See Also:
        Except for the thermostat, `NPH` shares parameters with `NPT`. See
        `NPT` for descriptions of the coupling and other barostat parameters.

    Examples::

        dt = 0.005
        tauS = 1000 * dt
        nph = hoomd.md.methods.NPH(filter=hoomd.filter.All(), tauS=tauS, S=2.0)
        # orthorhombic symmetry
        nph = hoomd.md.methods.NPH(filter=hoomd.filter.All(), tauS=tauS, S=2.0,
                                   couple="none")
        # tetragonal symmetry
        nph = hoomd.md.methods.NPH(filter=hoomd.filter.All(), tauS=tauS, S=2.0,
                                   couple="xy")
        # triclinic symmetry
        nph = hoomd.md.methods.NPH(filter=hoomd.filter.All(), tauS=tauS, S=2.0,
                                   couple="none", rescale_all=True)
        integrator = hoomd.md.Integrator(dt=dt, methods=[nph], forces=[lj])

    Attributes:
        filter (hoomd.filter.filter_like): Subset of particles on which to apply
            this method.

        S (tuple[hoomd.variant.Variant, ...]): Stress components set
            point for the barostat totalling 6 components.
            In Voigt notation,
            :math:`[S_{xx}, S_{yy}, S_{zz}, S_{yz}, S_{xz}, S_{xy}]`
            :math:`[\\mathrm{pressure}]`. Stress can be reset after
            method object is created. For example, an isotopic
            pressure can be set by ``nph.S = 4.``

        tauS (float): Coupling constant for the barostat
            :math:`[\\mathrm{time}]`.

        couple (str): Couplings of diagonal elements of the stress tensor,
            can be "none", "xy", "xz","yz", or "all".

        box_dof(tuple[bool, bool, bool, bool, bool, bool]): Box degrees of
            freedom with six boolean elements corresponding to x, y, z, xy, xz,
            yz, each.

        rescale_all (bool): if True, rescale all particles, not only those in
            the group.

        gamma (float): Dimensionless damping factor for the box degrees of
            freedom.

        barostat_dof (tuple[float, float, float, float, float, float]):
            Additional degrees of freedom for the barostat (:math:`\\nu_{xx}`,
            :math:`\\nu_{xy}`, :math:`\\nu_{xz}`, :math:`\\nu_{yy}`,
            :math:`\\nu_{yz}`, :math:`\\nu_{zz}`)
    """

    def __init__(self,
                 filter,
                 S,
                 tauS,
                 couple,
                 box_dof=(True, True, True, False, False, False),
                 rescale_all=False,
                 gamma=0.0):
        # store metadata
        param_dict = ParameterDict(filter=ParticleFilter,
                                   kT=Variant,
                                   S=OnlyIf(to_type_converter((Variant,) * 6),
                                            preprocess=self._preprocess_stress),
                                   tauS=float,
                                   couple=str,
                                   box_dof=(bool,) * 6,
                                   rescale_all=bool,
                                   gamma=float,
                                   barostat_dof=(float,) * 6)

        param_dict.update(
            dict(filter=filter,
                 kT=hoomd.variant.Constant(1.0),
                 S=S,
                 tauS=float(tauS),
                 couple=str(couple),
                 box_dof=tuple(box_dof),
                 rescale_all=bool(rescale_all),
                 gamma=float(gamma),
                 barostat_dof=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)))

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

        thermo_half_step = thermo_cls(cpp_sys_def, thermo_group)

        thermo_full_step = thermo_cls(cpp_sys_def, thermo_group)

        self._cpp_obj = cpp_cls(cpp_sys_def, thermo_group, thermo_half_step,
                                thermo_full_step, self.tauS, self.kT,
                                self.S, self.couple, self.box_dof, True, self.gamma)

        # Attach param_dict and typeparam_dict
        super()._attach()

    @staticmethod
    def _preprocess_stress(value):
        if isinstance(value, Sequence):
            if len(value) != 6:
                raise ValueError(
                    "Expected a single hoomd.variant.variant_like or six.")
            return tuple(value)
        else:
            return (value, value, value, 0, 0, 0)

    def thermalize_thermostat_and_barostat_dof(self):
        r"""Set the barostat momentum to random values.

        `thermalize_barostat_dof` sets a random value for the
        barostat :math:`\nu_{\mathrm{ij}}`. Call
        `thermalize_barostat_dof` to set a new random state for
        the barostat.

        .. important::
            You must call `Simulation.run` before
            `thermalize_barostat_dof`. Call ``run(steps=0)`` to
            prepare a newly created `hoomd.Simulation`.

        .. seealso:: `State.thermalize_particle_momenta`
        """
        if not self._attached:
            raise RuntimeError("Call Simulation.run(0) before"
                               "thermalize_thermostat_and_barostat_dof")

        self._simulation._warn_if_seed_unset()
        self._cpp_obj.thermalizeThermostatAndBarostatDOF(self._simulation.timestep)

    @hoomd.logging.log(requires_run=True)
    def barostat_energy(self):
        """Energy the barostat contributes to the Hamiltonian \
        :math:`[\\mathrm{energy}]`."""
        return self._cpp_obj.getBarostatEnergy(self._simulation.timestep)



class NPT(Method):
    """Constant pressure, constant temperature dynamics.

    Args:
        filter (hoomd.filter.filter_like): Subset of particles on which to apply
            this method.

        kT (hoomd.variant.variant_like): Temperature set point for the
            thermostat :math:`[\\mathrm{energy}]`.

        tau (float): Coupling constant for the thermostat
            :math:`[\\mathrm{time}]`.

        S (tuple[hoomd.variant.variant_like, ...] or \
                hoomd.variant.variant_like): Stress components set point for the
            barostat.

            In Voigt notation:
            :math:`[S_{xx}, S_{yy}, S_{zz}, S_{yz}, S_{xz}, S_{xy}]`
            :math:`[\\mathrm{pressure}]`. In case of isotropic
            pressure P (:math:`[p, p, p, 0, 0, 0]`), use ``S = p``.

        tauS (float): Coupling constant for the barostat
           :math:`[\\mathrm{time}]`.

        couple (str): Couplings of diagonal elements of the stress tensor,
            can be "none", "xy", "xz","yz", or "xyz".

        box_dof(`list` [ `bool` ]): Box degrees of freedom with six boolean
            elements corresponding to x, y, z, xy, xz, yz, each. Default to
            [True,True,True,False,False,False]). If turned on to True,
            rescale corresponding lengths or tilt factors and components of
            particle coordinates and velocities.

        rescale_all (bool): if True, rescale all particles, not only those
            in the group, Default to False.

        gamma (float): Dimensionless damping factor for the box degrees of
            freedom, Default to 0.

    `NPT` integrates integrates translational and rotational degrees of freedom
    in the Isothermal-isobaric ensemble.  The thermostat and barostat are
    introduced as additional degrees of freedom in the Hamiltonian that couple
    with the particle velocities and angular momenta and the box parameters.

    The translational thermostat has a momentum :math:`\\xi` and position
    :math:`\\eta`. The rotational thermostat has momentum
    :math:`\\xi_{\\mathrm{rot}}` and position :math:`\\eta_\\mathrm{rot}`. The
    barostat tensor is :math:`\\nu_{\\mathrm{ij}}`. Access these quantities
    using `translational_thermostat_dof`, `rotational_thermostat_dof`, and
    `barostat_dof`.

    By default, `NPT` performs integration in a cubic box under hydrostatic
    pressure by simultaneously rescaling the lengths *Lx*, *Ly* and *Lz* of the
    simulation box. Set the integration mode to change this default.

    The integration mode is defined by a set of couplings and by specifying
    the box degrees of freedom that are put under barostat control. Couplings
    define which diagonal elements of the pressure tensor
    :math:`P_{\\alpha,\\beta}` should be averaged over, so that the
    corresponding box lengths are rescaled by the same amount.

    Valid couplings are:

    - ``'none'`` (all box lengths are updated independently)
    - ``'xy`'`` (*Lx* and *Ly* are coupled)
    - ``'xz`'`` (*Lx* and *Lz* are coupled)
    - ``'yz`'`` (*Ly* and *Lz* are coupled)
    - ``'xyz`'`` (*Lx*, *Ly*, and *Lz* are coupled)

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

    `NPT` numerically integrates the equations of motion using the symplectic
    Martyna-Tobias-Klein equations of motion for NPT. For optimal stability, the
    update equations leave the phase-space measure invariant and are manifestly
    time-reversible.

    See Also:
        * `G. J. Martyna, D. J. Tobias, M. L. Klein  1994
          <http://dx.doi.org/10.1063/1.467468>`__
        * `M. E. Tuckerman et. al. 2006
          <http://dx.doi.org/10.1088/0305-4470/39/19/S18>`__
        * `T. Yu et. al. 2010
          <http://dx.doi.org/10.1016/j.chemphys.2010.02.014>`_

    Note:
        The coupling constant `tau` should be set within a
        reasonable range to avoid abrupt fluctuations in the kinetic temperature
        and to avoid long time to equilibration. The recommended value for most
        systems is :math:`\\tau = 100 \\delta t`.

    Note:
        The barostat coupling constant `tauS` should be set within a reasonable
        range to avoid abrupt fluctuations in the box volume and to avoid long
        time to equilibration. The recommend value for most systems is
        :math:`\\tau_S = 1000 \\delta t`.

    Important:
        Ensure that your initial condition includes non-zero particle velocities
        and angular momenta (when appropriate). The coupling between the
        thermostat and the velocities and angular momenta occurs via
        multiplication, so `NPT` cannot convert a zero velocity into a non-zero
        one except through particle collisions.

    Examples::

        npt = hoomd.md.methods.NPT(filter=hoomd.filter.All(), tau=1.0, kT=0.65,
        tauS = 1.2, S=2.0, couple="xyz")
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
        filter (hoomd.filter.filter_like): Subset of particles on which to apply
            this method.

        kT (hoomd.variant.Variant): Temperature set point for the
            thermostat :math:`[\\mathrm{energy}]`.

        tau (float): Coupling constant for the thermostat
            :math:`[\\mathrm{time}]`.

        S (tuple[hoomd.variant.Variant,...]): Stress components set point for
            the barostat.
            In Voigt notation,
            :math:`[S_{xx}, S_{yy}, S_{zz}, S_{yz}, S_{xz}, S_{xy}]`
            :math:`[\\mathrm{pressure}]`. Stress can be reset after the method
            object is created. For example, an isotropic pressure can be set by
            ``npt.S = 4.``

        tauS (float): Coupling constant for the barostat
            :math:`[\\mathrm{time}]`.

        couple (str): Couplings of diagonal elements of the stress tensor,
            can be "none", "xy", "xz","yz", or "xyz".

        box_dof(list[bool]): Box degrees of freedom with six boolean elements
            corresponding to x, y, z, xy, xz, yz, each.

        rescale_all (bool): if True, rescale all particles, not only those in
            the group.

        gamma (float): Dimensionless damping factor for the box degrees of
            freedom.

        translational_thermostat_dof (tuple[float, float]): Additional degrees
            of freedom for the translational thermostat (:math:`\\xi`,
            :math:`\\eta`)

        rotational_thermostat_dof (tuple[float, float]): Additional degrees
            of freedom for the rotational thermostat
            (:math:`\\xi_\\mathrm{rot}`, :math:`\\eta_\\mathrm{rot}`)

        barostat_dof (tuple[float, float, float, float, float, float]):
            Additional degrees of freedom for the barostat (:math:`\\nu_{xx}`,
            :math:`\\nu_{xy}`, :math:`\\nu_{xz}`, :math:`\\nu_{yy}`,
            :math:`\\nu_{yz}`, :math:`\\nu_{zz}`)
    """

    def __init__(self,
                 filter,
                 kT,
                 tau,
                 S,
                 tauS,
                 couple,
                 box_dof=[True, True, True, False, False, False],
                 rescale_all=False,
                 gamma=0.0):

        # store metadata
        param_dict = ParameterDict(filter=ParticleFilter,
                                   kT=Variant,
                                   tau=float(tau),
                                   S=OnlyIf(to_type_converter((Variant,) * 6),
                                            preprocess=self._preprocess_stress),
                                   tauS=float(tauS),
                                   couple=str(couple),
                                   box_dof=[
                                       bool,
                                   ] * 6,
                                   rescale_all=bool(rescale_all),
                                   gamma=float(gamma),
                                   translational_thermostat_dof=(float, float),
                                   rotational_thermostat_dof=(float, float),
                                   barostat_dof=(float, float, float, float,
                                                 float, float))
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

        thermo_half_step = thermo_cls(cpp_sys_def, thermo_group)

        thermo_full_step = thermo_cls(cpp_sys_def, thermo_group)

        self._cpp_obj = cpp_cls(cpp_sys_def, thermo_group, thermo_half_step,
                                thermo_full_step, self.tau, self.tauS, self.kT,
                                self.S, self.couple, self.box_dof, False)

        # Attach param_dict and typeparam_dict
        super()._attach()

    def _preprocess_stress(self, value):
        if isinstance(value, Sequence):
            if len(value) != 6:
                raise ValueError(
                    "Expected a single hoomd.variant.variant_like or six.")
            return tuple(value)
        else:
            return (value, value, value, 0, 0, 0)

    def thermalize_thermostat_and_barostat_dof(self):
        r"""Set the thermostat and barostat momenta to random values.

        `thermalize_thermostat_and_barostat_dof` sets a random value for the
        momentum :math:`\xi` and the barostat :math:`\nu_{\mathrm{ij}}`. When
        `Integrator.integrate_rotational_dof` is `True`, it also sets a random
        value for the rotational thermostat momentum :math:`\xi_{\mathrm{rot}}`.
        Call `thermalize_thermostat_and_barostat_dof` to set a new random state
        for the thermostat and barostat.

        .. important::
            You must call `Simulation.run` before
            `thermalize_thermostat_and_barostat_dof`. Call ``run(steps=0)`` to
            prepare a newly created `hoomd.Simulation`.

        .. seealso:: `State.thermalize_particle_momenta`
        """
        if not self._attached:
            raise RuntimeError("Call Simulation.run(0) before"
                               "thermalize_thermostat_and_barostat_dof")

        self._simulation._warn_if_seed_unset()
        self._cpp_obj.thermalizeThermostatAndBarostatDOF(
            self._simulation.timestep)

    @hoomd.logging.log(requires_run=True)
    def thermostat_energy(self):
        """Energy the thermostat contributes to the Hamiltonian \
        :math:`[\\mathrm{energy}]`."""
        return self._cpp_obj.getThermostatEnergy(self._simulation.timestep)

    @hoomd.logging.log(requires_run=True)
    def barostat_energy(self):
        """Energy the barostat contributes to the Hamiltonian \
        :math:`[\\mathrm{energy}]`."""
        return self._cpp_obj.getBarostatEnergy(self._simulation.timestep)



class NPTLangevinPiston(Method):
    """Constant pressure, constant temperature dynamics.

    Args:
        filter (hoomd.filter.filter_like): Subset of particles on which to apply
            this method.

        kT (hoomd.variant.variant_like): Temperature set point for the
            thermostat :math:`[\\mathrm{energy}]`.

        tau (float): Coupling constant for the thermostat
            :math:`[\\mathrm{time}]`.

        S (tuple[hoomd.variant.variant_like, ...] or \
                hoomd.variant.variant_like): Stress components set point for the
            barostat.

            In Voigt notation:
            :math:`[S_{xx}, S_{yy}, S_{zz}, S_{yz}, S_{xz}, S_{xy}]`
            :math:`[\\mathrm{pressure}]`. In case of isotropic
            pressure P (:math:`[p, p, p, 0, 0, 0]`), use ``S = p``.

        tauS (float): Coupling constant for the barostat
           :math:`[\\mathrm{time}]`.

        couple (str): Couplings of diagonal elements of the stress tensor,
            can be "none", "xy", "xz","yz", or "xyz".

        box_dof(`list` [ `bool` ]): Box degrees of freedom with six boolean
            elements corresponding to x, y, z, xy, xz, yz, each. Default to
            [True,True,True,False,False,False]). If turned on to True,
            rescale corresponding lengths or tilt factors and components of
            particle coordinates and velocities.

        rescale_all (bool): if True, rescale all particles, not only those
            in the group, Default to False.

        gamma (float): Dimensionless damping factor for the box degrees of
            freedom, Default to 0.

    `NPT` integrates integrates translational and rotational degrees of freedom
    in the Isothermal-isobaric ensemble.  The thermostat and barostat are
    introduced as additional degrees of freedom in the Hamiltonian that couple
    with the particle velocities and angular momenta and the box parameters.

    The translational thermostat has a momentum :math:`\\xi` and position
    :math:`\\eta`. The rotational thermostat has momentum
    :math:`\\xi_{\\mathrm{rot}}` and position :math:`\\eta_\\mathrm{rot}`. The
    barostat tensor is :math:`\\nu_{\\mathrm{ij}}`. Access these quantities
    using `translational_thermostat_dof`, `rotational_thermostat_dof`, and
    `barostat_dof`.

    By default, `NPT` performs integration in a cubic box under hydrostatic
    pressure by simultaneously rescaling the lengths *Lx*, *Ly* and *Lz* of the
    simulation box. Set the integration mode to change this default.

    The integration mode is defined by a set of couplings and by specifying
    the box degrees of freedom that are put under barostat control. Couplings
    define which diagonal elements of the pressure tensor
    :math:`P_{\\alpha,\\beta}` should be averaged over, so that the
    corresponding box lengths are rescaled by the same amount.

    Valid couplings are:

    - ``'none'`` (all box lengths are updated independently)
    - ``'xy`'`` (*Lx* and *Ly* are coupled)
    - ``'xz`'`` (*Lx* and *Lz* are coupled)
    - ``'yz`'`` (*Ly* and *Lz* are coupled)
    - ``'xyz`'`` (*Lx*, *Ly*, and *Lz* are coupled)

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

    `NPT` numerically integrates the equations of motion using the symplectic
    Martyna-Tobias-Klein equations of motion for NPT. For optimal stability, the
    update equations leave the phase-space measure invariant and are manifestly
    time-reversible.

    See Also:
        * `G. J. Martyna, D. J. Tobias, M. L. Klein  1994
          <http://dx.doi.org/10.1063/1.467468>`__
        * `M. E. Tuckerman et. al. 2006
          <http://dx.doi.org/10.1088/0305-4470/39/19/S18>`__
        * `T. Yu et. al. 2010
          <http://dx.doi.org/10.1016/j.chemphys.2010.02.014>`_

    Note:
        The coupling constant `tau` should be set within a
        reasonable range to avoid abrupt fluctuations in the kinetic temperature
        and to avoid long time to equilibration. The recommended value for most
        systems is :math:`\\tau = 100 \\delta t`.

    Note:
        The barostat coupling constant `tauS` should be set within a reasonable
        range to avoid abrupt fluctuations in the box volume and to avoid long
        time to equilibration. The recommend value for most systems is
        :math:`\\tau_S = 1000 \\delta t`.

    Important:
        Ensure that your initial condition includes non-zero particle velocities
        and angular momenta (when appropriate). The coupling between the
        thermostat and the velocities and angular momenta occurs via
        multiplication, so `NPT` cannot convert a zero velocity into a non-zero
        one except through particle collisions.

    Examples::

        npt = hoomd.md.methods.NPT(filter=hoomd.filter.All(), tau=1.0, kT=0.65,
        tauS = 1.2, S=2.0, couple="xyz")
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
        filter (hoomd.filter.filter_like): Subset of particles on which to apply
            this method.

        kT (hoomd.variant.Variant): Temperature set point for the
            thermostat :math:`[\\mathrm{energy}]`.

        tau (float): Coupling constant for the thermostat
            :math:`[\\mathrm{time}]`.

        S (tuple[hoomd.variant.Variant,...]): Stress components set point for
            the barostat.
            In Voigt notation,
            :math:`[S_{xx}, S_{yy}, S_{zz}, S_{yz}, S_{xz}, S_{xy}]`
            :math:`[\\mathrm{pressure}]`. Stress can be reset after the method
            object is created. For example, an isotropic pressure can be set by
            ``npt.S = 4.``

        tauS (float): Coupling constant for the barostat
            :math:`[\\mathrm{time}]`.

        couple (str): Couplings of diagonal elements of the stress tensor,
            can be "none", "xy", "xz","yz", or "xyz".

        box_dof(list[bool]): Box degrees of freedom with six boolean elements
            corresponding to x, y, z, xy, xz, yz, each.

        rescale_all (bool): if True, rescale all particles, not only those in
            the group.

        gamma (float): Dimensionless damping factor for the box degrees of
            freedom.

        translational_thermostat_dof (tuple[float, float]): Additional degrees
            of freedom for the translational thermostat (:math:`\\xi`,
            :math:`\\eta`)

        rotational_thermostat_dof (tuple[float, float]): Additional degrees
            of freedom for the rotational thermostat
            (:math:`\\xi_\\mathrm{rot}`, :math:`\\eta_\\mathrm{rot}`)

        barostat_dof (tuple[float, float, float, float, float, float]):
            Additional degrees of freedom for the barostat (:math:`\\nu_{xx}`,
            :math:`\\nu_{xy}`, :math:`\\nu_{xz}`, :math:`\\nu_{yy}`,
            :math:`\\nu_{yz}`, :math:`\\nu_{zz}`)
    """

    def __init__(self,
                 filter,
                 kT,
                 tau,
                 S,
                 tauS,
                 couple,
                 box_dof=[True, True, True, False, False, False],
                 rescale_all=False,
                 gamma=0.0):

        # store metadata
        param_dict = ParameterDict(filter=ParticleFilter,
                                   kT=Variant,
                                   tau=float(tau),
                                   S=OnlyIf(to_type_converter((Variant,) * 6),
                                            preprocess=self._preprocess_stress),
                                   tauS=float(tauS),
                                   couple=str(couple),
                                   box_dof=[
                                               bool,
                                           ] * 6,
                                   rescale_all=bool(rescale_all),
                                   gamma=float(gamma),
                                   translational_thermostat_dof=(float, float),
                                   rotational_thermostat_dof=(float, float),
                                   barostat_dof=(float, float, float, float,
                                                 float, float))
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

    def _attach_hook(self):
        # initialize the reflected c++ class
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cpp_cls = _md.TwoStepNPTLangevinPiston
            thermo_cls = _md.ComputeThermo
        else:
            cpp_cls = _md.TwoStepNPTLangevinPistonGPU
            thermo_cls = _md.ComputeThermoGPU

        cpp_sys_def = self._simulation.state._cpp_sys_def
        thermo_group = self._simulation.state._get_group(self.filter)

        thermo_half_step = thermo_cls(cpp_sys_def, thermo_group)

        thermo_full_step = thermo_cls(cpp_sys_def, thermo_group)

        self._cpp_obj = cpp_cls(cpp_sys_def, thermo_group, thermo_half_step,
                                thermo_full_step, self.tau, self.tauS, self.kT,
                                self.S, self.couple, self.box_dof, False)

        # Attach param_dict and typeparam_dict
        super()._attach_hook()

    def _preprocess_stress(self, value):
        if isinstance(value, Sequence):
            if len(value) != 6:
                raise ValueError(
                    "Expected a single hoomd.variant.variant_like or six.")
            return tuple(value)
        else:
            return (value, value, value, 0, 0, 0)

    def thermalize_thermostat_and_barostat_dof(self):
        r"""Set the thermostat and barostat momenta to random values.

        `thermalize_thermostat_and_barostat_dof` sets a random value for the
        momentum :math:`\xi` and the barostat :math:`\nu_{\mathrm{ij}}`. When
        `Integrator.integrate_rotational_dof` is `True`, it also sets a random
        value for the rotational thermostat momentum :math:`\xi_{\mathrm{rot}}`.
        Call `thermalize_thermostat_and_barostat_dof` to set a new random state
        for the thermostat and barostat.

        .. important::
            You must call `Simulation.run` before
            `thermalize_thermostat_and_barostat_dof`. Call ``run(steps=0)`` to
            prepare a newly created `hoomd.Simulation`.

        .. seealso:: `State.thermalize_particle_momenta`
        """
        if not self._attached:
            raise RuntimeError("Call Simulation.run(0) before"
                               "thermalize_thermostat_and_barostat_dof")

        self._simulation._warn_if_seed_unset()
        self._cpp_obj.thermalizeThermostatAndBarostatDOF(
            self._simulation.timestep)

    @hoomd.logging.log(requires_run=True)
    def thermostat_energy(self):
        """Energy the thermostat contributes to the Hamiltonian \
        :math:`[\\mathrm{energy}]`."""
        return self._cpp_obj.getThermostatEnergy(self._simulation.timestep)

    @hoomd.logging.log(requires_run=True)
    def barostat_energy(self):
        """Energy the barostat contributes to the Hamiltonian \
        :math:`[\\mathrm{energy}]`."""
        return self._cpp_obj.getBarostatEnergy(self._simulation.timestep)

class NPTBussiLangevinPiston(Method):
    """Constant pressure, constant temperature dynamics.

    Args:
        filter (hoomd.filter.filter_like): Subset of particles on which to apply
            this method.

        kT (hoomd.variant.variant_like): Temperature set point for the
            thermostat :math:`[\\mathrm{energy}]`.

        tau (float): Coupling constant for the thermostat
            :math:`[\\mathrm{time}]`.

        S (tuple[hoomd.variant.variant_like, ...] or \
                hoomd.variant.variant_like): Stress components set point for the
            barostat.

            In Voigt notation:
            :math:`[S_{xx}, S_{yy}, S_{zz}, S_{yz}, S_{xz}, S_{xy}]`
            :math:`[\\mathrm{pressure}]`. In case of isotropic
            pressure P (:math:`[p, p, p, 0, 0, 0]`), use ``S = p``.

        tauS (float): Coupling constant for the barostat
           :math:`[\\mathrm{time}]`.

        couple (str): Couplings of diagonal elements of the stress tensor,
            can be "none", "xy", "xz","yz", or "xyz".

        box_dof(`list` [ `bool` ]): Box degrees of freedom with six boolean
            elements corresponding to x, y, z, xy, xz, yz, each. Default to
            [True,True,True,False,False,False]). If turned on to True,
            rescale corresponding lengths or tilt factors and components of
            particle coordinates and velocities.

        rescale_all (bool): if True, rescale all particles, not only those
            in the group, Default to False.

        gamma (float): Dimensionless damping factor for the box degrees of
            freedom, Default to 0.

    `NPT` integrates integrates translational and rotational degrees of freedom
    in the Isothermal-isobaric ensemble.  The thermostat and barostat are
    introduced as additional degrees of freedom in the Hamiltonian that couple
    with the particle velocities and angular momenta and the box parameters.

    The translational thermostat has a momentum :math:`\\xi` and position
    :math:`\\eta`. The rotational thermostat has momentum
    :math:`\\xi_{\\mathrm{rot}}` and position :math:`\\eta_\\mathrm{rot}`. The
    barostat tensor is :math:`\\nu_{\\mathrm{ij}}`. Access these quantities
    using `translational_thermostat_dof`, `rotational_thermostat_dof`, and
    `barostat_dof`.

    By default, `NPT` performs integration in a cubic box under hydrostatic
    pressure by simultaneously rescaling the lengths *Lx*, *Ly* and *Lz* of the
    simulation box. Set the integration mode to change this default.

    The integration mode is defined by a set of couplings and by specifying
    the box degrees of freedom that are put under barostat control. Couplings
    define which diagonal elements of the pressure tensor
    :math:`P_{\\alpha,\\beta}` should be averaged over, so that the
    corresponding box lengths are rescaled by the same amount.

    Valid couplings are:

    - ``'none'`` (all box lengths are updated independently)
    - ``'xy`'`` (*Lx* and *Ly* are coupled)
    - ``'xz`'`` (*Lx* and *Lz* are coupled)
    - ``'yz`'`` (*Ly* and *Lz* are coupled)
    - ``'xyz`'`` (*Lx*, *Ly*, and *Lz* are coupled)

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

    `NPT` numerically integrates the equations of motion using the symplectic
    Martyna-Tobias-Klein equations of motion for NPT. For optimal stability, the
    update equations leave the phase-space measure invariant and are manifestly
    time-reversible.

    See Also:
        * `G. J. Martyna, D. J. Tobias, M. L. Klein  1994
          <http://dx.doi.org/10.1063/1.467468>`__
        * `M. E. Tuckerman et. al. 2006
          <http://dx.doi.org/10.1088/0305-4470/39/19/S18>`__
        * `T. Yu et. al. 2010
          <http://dx.doi.org/10.1016/j.chemphys.2010.02.014>`_

    Note:
        The coupling constant `tau` should be set within a
        reasonable range to avoid abrupt fluctuations in the kinetic temperature
        and to avoid long time to equilibration. The recommended value for most
        systems is :math:`\\tau = 100 \\delta t`.

    Note:
        The barostat coupling constant `tauS` should be set within a reasonable
        range to avoid abrupt fluctuations in the box volume and to avoid long
        time to equilibration. The recommend value for most systems is
        :math:`\\tau_S = 1000 \\delta t`.

    Important:
        Ensure that your initial condition includes non-zero particle velocities
        and angular momenta (when appropriate). The coupling between the
        thermostat and the velocities and angular momenta occurs via
        multiplication, so `NPT` cannot convert a zero velocity into a non-zero
        one except through particle collisions.

    Examples::

        npt = hoomd.md.methods.NPT(filter=hoomd.filter.All(), tau=1.0, kT=0.65,
        tauS = 1.2, S=2.0, couple="xyz")
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
        filter (hoomd.filter.filter_like): Subset of particles on which to apply
            this method.

        kT (hoomd.variant.Variant): Temperature set point for the
            thermostat :math:`[\\mathrm{energy}]`.

        tau (float): Coupling constant for the thermostat
            :math:`[\\mathrm{time}]`.

        S (tuple[hoomd.variant.Variant,...]): Stress components set point for
            the barostat.
            In Voigt notation,
            :math:`[S_{xx}, S_{yy}, S_{zz}, S_{yz}, S_{xz}, S_{xy}]`
            :math:`[\\mathrm{pressure}]`. Stress can be reset after the method
            object is created. For example, an isotropic pressure can be set by
            ``npt.S = 4.``

        tauS (float): Coupling constant for the barostat
            :math:`[\\mathrm{time}]`.

        couple (str): Couplings of diagonal elements of the stress tensor,
            can be "none", "xy", "xz","yz", or "xyz".

        box_dof(list[bool]): Box degrees of freedom with six boolean elements
            corresponding to x, y, z, xy, xz, yz, each.

        rescale_all (bool): if True, rescale all particles, not only those in
            the group.

        gamma (float): Dimensionless damping factor for the box degrees of
            freedom.

        translational_thermostat_dof (tuple[float, float]): Additional degrees
            of freedom for the translational thermostat (:math:`\\xi`,
            :math:`\\eta`)

        rotational_thermostat_dof (tuple[float, float]): Additional degrees
            of freedom for the rotational thermostat
            (:math:`\\xi_\\mathrm{rot}`, :math:`\\eta_\\mathrm{rot}`)

        barostat_dof (tuple[float, float, float, float, float, float]):
            Additional degrees of freedom for the barostat (:math:`\\nu_{xx}`,
            :math:`\\nu_{xy}`, :math:`\\nu_{xz}`, :math:`\\nu_{yy}`,
            :math:`\\nu_{yz}`, :math:`\\nu_{zz}`)
    """

    def __init__(self,
                 filter,
                 kT,
                 S,
                 tauS,
                 couple,
                 box_dof=[True, True, True, False, False, False],
                 rescale_all=False):

        # store metadata
        param_dict = ParameterDict(filter=ParticleFilter,
                                   kT=Variant,
                                   S=OnlyIf(to_type_converter((Variant,) * 6),
                                            preprocess=self._preprocess_stress),
                                   tauS=float(tauS),
                                   couple=str(couple),
                                   box_dof=[
                                               bool,
                                           ] * 6,
                                   rescale_all=bool(rescale_all),
                                   barostat_dof=(float, float, float, float,
                                                 float, float))
        param_dict.update(
            dict(filter=filter,
                 kT=kT,
                 S=S,
                 couple=couple,
                 box_dof=box_dof,
                 barostat_dof=(0, 0, 0, 0, 0, 0)))

        # set defaults
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        # initialize the reflected c++ class
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cpp_cls = _md.TwoStepNPTBussiLangevinPiston
            thermo_cls = _md.ComputeThermo
        else:
            cpp_cls = _md.TwoStepNPTBussiLangevinPistonGPU
            thermo_cls = _md.ComputeThermoGPU

        cpp_sys_def = self._simulation.state._cpp_sys_def
        thermo_group = self._simulation.state._get_group(self.filter)

        thermo_half_step = thermo_cls(cpp_sys_def, thermo_group)

        thermo_full_step = thermo_cls(cpp_sys_def, thermo_group)

        self._cpp_obj = cpp_cls(cpp_sys_def, thermo_group, thermo_half_step,
                                thermo_full_step, self.tauS, self.kT,
                                self.S, self.couple, self.box_dof, False)

        # Attach param_dict and typeparam_dict
        super()._attach_hook()

    def _preprocess_stress(self, value):
        if isinstance(value, Sequence):
            if len(value) != 6:
                raise ValueError(
                    "Expected a single hoomd.variant.variant_like or six.")
            return tuple(value)
        else:
            return (value, value, value, 0, 0, 0)

    def thermalize_barostat_dof(self):
        r"""Set the thermostat and barostat momenta to random values.

        `thermalize_thermostat_and_barostat_dof` sets a random value for the
        momentum :math:`\xi` and the barostat :math:`\nu_{\mathrm{ij}}`. When
        `Integrator.integrate_rotational_dof` is `True`, it also sets a random
        value for the rotational thermostat momentum :math:`\xi_{\mathrm{rot}}`.
        Call `thermalize_thermostat_and_barostat_dof` to set a new random state
        for the thermostat and barostat.

        .. important::
            You must call `Simulation.run` before
            `thermalize_thermostat_and_barostat_dof`. Call ``run(steps=0)`` to
            prepare a newly created `hoomd.Simulation`.

        .. seealso:: `State.thermalize_particle_momenta`
        """
        if not self._attached:
            raise RuntimeError("Call Simulation.run(0) before"
                               "thermalize_thermostat_and_barostat_dof")

        self._simulation._warn_if_seed_unset()
        self._cpp_obj.thermalizeBarostatDOF(self._simulation.timestep)

    @hoomd.logging.log(requires_run=True)
    def barostat_energy(self):
        """Energy the barostat contributes to the Hamiltonian \
        :math:`[\\mathrm{energy}]`."""
        return self._cpp_obj.getBarostatEnergy(self._simulation.timestep)

class NVE(Method):
    r"""Constant volume, constant energy dynamics.

    Args:
        filter (hoomd.filter.filter_like): Subset of particles on which to
            apply this method.

    `NVE` integrates integrates translational and rotational degrees of freedom
    in the microcanonical ensemble. The equations of motion are derived from the
    hamiltonian:

    .. math::

        H = U + K_\mathrm{translational} + K_\mathrm{rotational}

    `NVE` numerically integrates the translational degrees of freedom
    using Velocity-Verlet and the rotational degrees of freedom with a scheme
    based on `Kamberaj 2005`_.

    Examples::

        nve = hoomd.md.methods.NVE(filter=hoomd.filter.All())
        integrator = hoomd.md.Integrator(dt=0.005, methods=[nve], forces=[lj])

    .. _Kamberaj 2005: http://dx.doi.org/10.1063/1.1906216

    Attributes:
        filter (hoomd.filter.filter_like): Subset of particles on which to
            apply this method.
    """

    def __init__(self, filter):

        # store metadata
        param_dict = ParameterDict(filter=ParticleFilter,)
        param_dict["filter"] = filter

        # set defaults
        self._param_dict.update(param_dict)

    def _attach_hook(self):

        sim = self._simulation
        # initialize the reflected c++ class
        if isinstance(sim.device, hoomd.device.CPU):
            self._cpp_obj = _md.TwoStepNVE(sim.state._cpp_sys_def,
                                           sim.state._get_group(self.filter))
        else:
            self._cpp_obj = _md.TwoStepNVEGPU(sim.state._cpp_sys_def,
                                              sim.state._get_group(self.filter))

        # Attach param_dict and typeparam_dict
        super()._attach_hook()


class DisplacementCapped(NVE):
    r"""Newtonian dynamics with a cap on the maximum displacement per time step.

    The method employs a maximum displacement allowed each time step. This
    method can be helpful to relax a system with too much overlaps without
    "blowing up" the system.

    Warning:
        This method does not conserve energy or momentum.

    Args:
        filter (hoomd.filter.filter_like): Subset of particles on which to
            apply this method.
        maximum_displacement (hoomd.variant.variant_like): The maximum
            displacement allowed for a particular timestep
            :math:`[\mathrm{length}]`.

    `DisplacementCapped` integrates integrates translational and rotational
    degrees of freedom using modified microcanoncial dynamics. See `NVE` for the
    basis of the algorithm.

    Examples::

        relaxer = hoomd.md.methods.DisplacementCapped(
            filter=hoomd.filter.All(), maximum_displacement=1e-3)
        integrator = hoomd.md.Integrator(
            dt=0.005, methods=[relaxer], forces=[lj])

    Attributes:
        filter (hoomd.filter.filter_like): Subset of particles on which to
            apply this method.
        maximum_displacement (hoomd.variant.variant_like): The maximum
            displacement allowed for a particular timestep
            :math:`[\mathrm{length}]`.
    """

    def __init__(self, filter,
                 maximum_displacement: hoomd.variant.variant_like):

        # store metadata
        super().__init__(filter)
        param_dict = ParameterDict(maximum_displacement=hoomd.variant.Variant)
        param_dict["maximum_displacement"] = maximum_displacement

        # set defaults
        self._param_dict.update(param_dict)

class NVTStochastic(Method):
    def __init__(self, filter, kT):

        # store metadata
        param_dict = ParameterDict(filter=ParticleFilter,
                                   kT=Variant)
        param_dict.update(
            dict(kT=kT,
                 filter=filter))
        # set defaults
        self._param_dict.update(param_dict)

    def _attach(self):

        # initialize the reflected cpp class
        if isinstance(self._simulation.device, hoomd.device.CPU):
            my_class = _md.TwoStepNVTStochastic
            thermo_cls = _md.ComputeThermo
        else:
            my_class = _md.TwoStepNVTStochasticGPU
            thermo_cls = _md.ComputeThermoGPU

        group = self._simulation.state._get_group(self.filter)
        cpp_sys_def = self._simulation.state._cpp_sys_def
        thermo = thermo_cls(cpp_sys_def, group)
        self._cpp_obj = my_class(cpp_sys_def, group, thermo, self.kT)
        super()._attach()

class Langevin(Method):
    r"""Langevin dynamics.

    Args:
        filter (hoomd.filter.filter_like): Subset of particles to
            apply this method to.

        kT (hoomd.variant.variant_like): Temperature of the simulation
            :math:`[\mathrm{energy}]`.

        alpha (float): When set, use :math:`\alpha d_i` for the drag
            coefficient where :math:`d_i` is particle diameter
            :math:`[\mathrm{mass} \cdot
            \mathrm{length}^{-1} \cdot \mathrm{time}^{-1}]`.
            Defaults to None.

        tally_reservoir_energy (bool): If true, the energy exchange
            between the thermal reservoir and the particles is tracked. Total
            energy conservation can then be monitored by adding
            ``langevin_reservoir_energy_groupname`` to the logged quantities.
            Defaults to False :math:`[\mathrm{energy}]`.

    `Langevin` integrates particles forward in time according to the
    Langevin equations of motion.

    The translational degrees of freedom follow:

    .. math::

        m \frac{d\vec{v}}{dt} &= \vec{F}_\mathrm{C} - \gamma \cdot \vec{v} +
        \vec{F}_\mathrm{R}

        \langle \vec{F}_\mathrm{R} \rangle &= 0

        \langle |\vec{F}_\mathrm{R}|^2 \rangle &= 2 d kT \gamma / \delta t

    where :math:`\vec{F}_\mathrm{C}` is the force on the particle from all
    potentials and constraint forces, :math:`\gamma` is the drag coefficient,
    :math:`\vec{v}` is the particle's velocity, :math:`\vec{F}_\mathrm{R}` is a
    uniform random force, and :math:`d` is the dimensionality of the system (2
    or 3).  The magnitude of the random force is chosen via the
    fluctuation-dissipation theorem to be consistent with the specified drag and
    temperature, :math:`T`.

    About axes where :math:`I^i > 0`, the rotational degrees of freedom follow:

    .. math::

        I \frac{d\vec{L}}{dt} &= \vec{\tau}_\mathrm{C} - \gamma_r \cdot \vec{L}
        + \vec{\tau}_\mathrm{R}

        \langle \vec{\tau}_\mathrm{R} \rangle &= 0,

        \langle \tau_\mathrm{R}^i \cdot \tau_\mathrm{R}^i \rangle &=
        2 k T \gamma_r^i / \delta t,

    where :math:`\vec{\tau}_\mathrm{C} = \vec{\tau}_\mathrm{net}`,
    :math:`\gamma_r^i` is the i-th component of the rotational drag coefficient
    (`gamma_r`), :math:`\tau_\mathrm{R}^i` is a component of the uniform random
    the torque, :math:`\vec{L}` is the particle's angular momentum and :math:`I`
    is the the particle's moment of inertia. The magnitude of the random torque
    is chosen via the fluctuation-dissipation theorem to be consistent with the
    specified drag and temperature, :math:`T`.

    `Langevin` numerically integrates the translational degrees of freedom
    using Velocity-Verlet and the rotational degrees of freedom with a scheme
    based on `Kamberaj 2005`_.

    Langevin dynamics includes the acceleration term in the Langevin equation.
    This assumption is valid when underdamped: :math:`\frac{m}{\gamma} \gg
    \delta t`. Use `Brownian` if your system is not underdamped.

    You can set :math:`\gamma` in two ways:

    1. Specify :math:`\alpha` which scales the particle diameter to
       :math:`\gamma = \alpha d_i`.
    2. After the method object is created, specify the attribute `gamma`
       and `gamma_r` for rotational damping or random torque to assign them
       directly, with independent values for each particle type in the
       system.

    Examples::

        langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=0.2,
        alpha=1.0)
        integrator = hoomd.md.Integrator(dt=0.001, methods=[langevin],
        forces=[lj])

    Examples of using `gamma` and `gamma_r`::

        langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=0.2)
        langevin.gamma.default = 2.0
        langevin.gamma_r.default = [1.0,2.0,3.0]

    Warning:
        When restarting a simulation, the energy of the reservoir will be reset
        to zero.

    .. _Kamberaj 2005: http://dx.doi.org/10.1063/1.1906216

    Attributes:
        filter (hoomd.filter.filter_like): Subset of particles to
            apply this method to.

        kT (hoomd.variant.Variant): Temperature of the
            simulation :math:`[\mathrm{energy}]`.

        alpha (float): When set, use :math:`\alpha d_i` for the drag
            coefficient where :math:`d_i` is particle diameter
            :math:`[\mathrm{mass} \cdot \mathrm{length}^{-1}
            \cdot \mathrm{time}^{-1}]`. Defaults to None.

        gamma (TypeParameter[ ``particle type``, `float` ]): The drag
            coefficient can be directly set instead of the ratio of particle
            diameter (:math:`\gamma = \alpha d_i`). The type of ``gamma``
            parameter is either positive float or zero
            :math:`[\mathrm{mass} \cdot \mathrm{time}^{-1}]`.

        gamma_r (TypeParameter[``particle type``,[`float`, `float` , `float`]]):
            The rotational drag coefficient can be set. The type of ``gamma_r``
            parameter is a tuple of three float. The type of each element of
            tuple is either positive float or zero
            :math:`[\mathrm{mass} \cdot \mathrm{time}^{-1}]`.

    """

    def __init__(self, filter, kT, alpha=None, tally_reservoir_energy=False):

        # store metadata
        param_dict = ParameterDict(
            filter=ParticleFilter,
            kT=Variant,
            alpha=OnlyTypes(float, allow_none=True),
            tally_reservoir_energy=bool(tally_reservoir_energy),
        )
        param_dict.update(dict(kT=kT, alpha=alpha, filter=filter))
        # set defaults
        self._param_dict.update(param_dict)

        gamma = TypeParameter('gamma',
                              type_kind='particle_types',
                              param_dict=TypeParameterDict(1., len_keys=1))

        gamma_r = TypeParameter('gamma_r',
                                type_kind='particle_types',
                                param_dict=TypeParameterDict((1., 1., 1.),
                                                             len_keys=1))

        self._extend_typeparam([gamma, gamma_r])

    def _attach_hook(self):
        """Langevin uses RNGs. Warn the user if they did not set the seed."""
        self._simulation._warn_if_seed_unset()
        sim = self._simulation
        if isinstance(sim.device, hoomd.device.CPU):
            my_class = _md.TwoStepLangevin
        else:
            my_class = _md.TwoStepLangevinGPU

        self._cpp_obj = my_class(sim.state._cpp_sys_def,
                                 sim.state._get_group(self.filter), self.kT)

        # Attach param_dict and typeparam_dict
        super()._attach_hook()


class Brownian(Method):
    r"""Brownian dynamics.

    Args:
        filter (hoomd.filter.filter_like): Subset of particles to
            apply this method to.

        kT (hoomd.variant.variant_like): Temperature of the simulation
            :math:`[\mathrm{energy}]`.

        alpha (float): When set, use :math:`\alpha d_i` for the
            drag coefficient where :math:`d_i` is particle diameter
            :math:`[\mathrm{mass} \cdot \mathrm{length}^{-1}
            \cdot \mathrm{time}^{-1}]`.
            Defaults to ``None``

    `Brownian` integrates particles forward in time according to the overdamped
    Langevin equations of motion, sometimes called Brownian dynamics or the
    diffusive limit. It integrates both the translational and rotational
    degrees of freedom.

    The translational degrees of freedom follow:

    .. math::

        \frac{d\vec{r}}{dt} &= \frac{\vec{F}_\mathrm{C} +
        \vec{F}_\mathrm{R}}{\gamma},

        \langle \vec{F}_\mathrm{R} \rangle &= 0,

        \langle |\vec{F}_\mathrm{R}|^2 \rangle &= 2 d k T \gamma / \delta t,

        \langle \vec{v}(t) \rangle &= 0,

        \langle |\vec{v}(t)|^2 \rangle &= d k T / m,

    where :math:`\vec{F}_\mathrm{C} = \vec{F}_\mathrm{net}` is the net force on
    the particle from all forces (`hoomd.md.Integrator.forces`) and constraints
    (`hoomd.md.Integrator.constraints`), :math:`\gamma` is the translational
    drag coefficient (`gamma`), :math:`\vec{F}_\mathrm{R}` is a uniform random
    force, :math:`\vec{v}` is the particle's velocity, and :math:`d` is the
    dimensionality of the system. The magnitude of the random force is chosen
    via the fluctuation-dissipation theorem to be consistent with the specified
    drag and temperature, :math:`T`.

    About axes where :math:`I^i > 0`, the rotational degrees of freedom follow:

    .. math::

        \frac{d\mathbf{q}}{dt} &= \frac{\vec{\tau}_\mathrm{C} +
        \vec{\tau}_\mathrm{R}}{\gamma_r},

        \langle \vec{\tau}_\mathrm{R} \rangle &= 0,

        \langle \tau_\mathrm{R}^i \cdot \tau_\mathrm{R}^i \rangle &=
        2 k T \gamma_r^i / \delta t,

        \langle \vec{L}(t) \rangle &= 0,

        \langle L^i(t) \cdot L^i(t) \rangle &= k T \cdot I^i,

    where :math:`\vec{\tau}_\mathrm{C} = \vec{\tau}_\mathrm{net}`,
    :math:`\gamma_r^i` is the i-th component of the rotational drag coefficient
    (`gamma_r`), :math:`\tau_\mathrm{R}^i` is a component of the uniform random
    the torque, :math:`L^i` is the i-th component of the particle's angular
    momentum and :math:`I^i` is the i-th component of the particle's
    moment of inertia. The magnitude of the random torque is chosen
    via the fluctuation-dissipation theorem to be consistent with the specified
    drag and temperature, :math:`T`.

    `Brownian` uses the numerical integration method from `I. Snook 2007`_, The
    Langevin and Generalised Langevin Approach to the Dynamics of Atomic,
    Polymeric and Colloidal Systems, section 6.2.5, with the exception that
    :math:`\vec{F}_\mathrm{R}` is drawn from a uniform random number
    distribution.

    .. _I. Snook 2007: http://dx.doi.org/10.1016/B978-0-444-52129-3.50028-6

    In Brownian dynamics, particle velocities and angular momenta are completely
    decoupled from positions. At each time step, `Brownian` draws a new velocity
    distribution consistent with the current set temperature so that
    `hoomd.md.compute.ThermodynamicQuantities` will report appropriate
    temperatures and pressures when logged or used by other methods.

    Brownian dynamics neglects the acceleration term in the Langevin equation.
    This assumption is valid when overdamped:
    :math:`\frac{m}{\gamma} \ll \delta t`. Use `Langevin` if your
    system is not overdamped.

    You can set :math:`\gamma` in two ways:

    1. Specify :math:`\alpha` which scales the particle diameter to
       :math:`\gamma = \alpha d_i`.
    2. After the method object is created, specify the attribute `gamma`
       and `gamma_r` for rotational damping or random torque to assign them
       directly, with independent values for each particle type in the
       system.

    Examples::

        brownian = hoomd.md.methods.Brownian(filter=hoomd.filter.All(), kT=0.2,
        alpha=1.0)
        integrator = hoomd.md.Integrator(dt=0.001, methods=[brownian],
        forces=[lj])

    Examples of using `gamma` and `gamma_r`::

        brownian = hoomd.md.methods.Brownian(filter=hoomd.filter.All(), kT=0.2)
        brownian.gamma.default = 2.0
        brownian.gamma_r.default = [1.0, 2.0, 3.0]

    Attributes:
        filter (hoomd.filter.filter_like): Subset of particles to
            apply this method to.

        kT (hoomd.variant.Variant): Temperature of the
            simulation :math:`[\mathrm{energy}]`.

        alpha (float): When set, use :math:`\alpha d_i` for the drag
            coefficient where :math:`d_i` is particle diameter
            :math:`[\mathrm{mass} \cdot \mathrm{length}^{-1}
            \cdot \mathrm{time}^{-1}]`.

        gamma (TypeParameter[ ``particle type``, `float` ]): The drag
            coefficient can be directly set instead of the ratio of particle
            diameter (:math:`\gamma = \alpha d_i`). The type of ``gamma``
            parameter is either positive float or zero
            :math:`[\mathrm{mass} \cdot \mathrm{time}^{-1}]`.

        gamma_r (TypeParameter[``particle type``, [`float`, `float`, `float`]]):
            The rotational drag coefficient can be set. The type of ``gamma_r``
            parameter is a tuple of three float. The type of each element of
            tuple is either positive float or zero
            :math:`[\mathrm{force} \cdot \mathrm{length} \cdot
            \mathrm{radian}^{-1} \cdot \mathrm{time}^{-1}]`.
    """

    def __init__(self, filter, kT, alpha=None):

        # store metadata
        param_dict = ParameterDict(
            filter=ParticleFilter,
            kT=Variant,
            alpha=OnlyTypes(float, allow_none=True),
        )
        param_dict.update(dict(kT=kT, alpha=alpha, filter=filter))

        # set defaults
        self._param_dict.update(param_dict)

        gamma = TypeParameter('gamma',
                              type_kind='particle_types',
                              param_dict=TypeParameterDict(1., len_keys=1))

        gamma_r = TypeParameter('gamma_r',
                                type_kind='particle_types',
                                param_dict=TypeParameterDict((1., 1., 1.),
                                                             len_keys=1))
        self._extend_typeparam([gamma, gamma_r])

    def _attach_hook(self):
        """Brownian uses RNGs. Warn the user if they did not set the seed."""
        self._simulation._warn_if_seed_unset()
        sim = self._simulation
        if isinstance(sim.device, hoomd.device.CPU):
            self._cpp_obj = _md.TwoStepBD(sim.state._cpp_sys_def,
                                          sim.state._get_group(self.filter),
                                          self.kT, False, False)
        else:
            self._cpp_obj = _md.TwoStepBDGPU(sim.state._cpp_sys_def,
                                             sim.state._get_group(self.filter),
                                             self.kT, False, False)

        # Attach param_dict and typeparam_dict
        super()._attach_hook()


class Berendsen(Method):
    r"""Applies the Berendsen thermostat.

    Args:
        filter (hoomd.filter.filter_like): Subset of particles to apply this
            method to.

        kT (hoomd.variant.variant_like): Temperature of the simulation.
            :math:`[energy]`

        tau (float): Time constant of thermostat. :math:`[time]`

    `Berendsen` rescales the velocities of all particles on each time step. The
    rescaling is performed so that the difference in the current temperature
    from the set point decays exponentially:
    `Berendsen et. al. 1984 <http://dx.doi.org/10.1063/1.448118>`_.

    .. math::

        \frac{dT_\mathrm{cur}}{dt} = \frac{T - T_\mathrm{cur}}{\tau}

    .. attention::
        `Berendsen` does not function with MPI parallel simulations.

    .. attention::
        `Berendsen` does not integrate rotational degrees of freedom.

    Examples::

        berendsen = hoomd.md.methods.Berendsen(
            filter=hoomd.filter.All(), kT=0.2, tau=10.0)
        integrator = hoomd.md.Integrator(
            dt=0.001, methods=[berendsen], forces=[lj])


    Attributes:
        filter (hoomd.filter.filter_like): Subset of particles to apply this
            method to.

        kT (hoomd.variant.Variant): Temperature of the
            simulation. :math:`[energy]`

        tau (float): Time constant of thermostat. :math:`[time]`
    """

    def __init__(self, filter, kT, tau):
        # store metadata
        param_dict = ParameterDict(filter=ParticleFilter,
                                   kT=Variant,
                                   tau=float(tau))
        param_dict.update(dict(filter=filter, kT=kT))

        # set defaults
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        sim = self._simulation
        # Error out in MPI simulations
        if hoomd.version.mpi_enabled:
            if sim.device._comm.num_ranks > 1:
                raise RuntimeError(
                    "hoomd.md.methods.Berendsen is not supported in "
                    "multi-processor simulations.")

        group = sim.state._get_group(self.filter)
        if isinstance(sim.device, hoomd.device.CPU):
            cpp_method = _md.TwoStepBerendsen
            thermo_cls = _md.ComputeThermo
        else:
            cpp_method = _md.TwoStepBerendsenGPU
            thermo_cls = _md.ComputeThermoGPU
        self._cpp_obj = cpp_method(sim.state._cpp_sys_def, group,
                                   thermo_cls(sim.state._cpp_sys_def, group),
                                   self.tau, self.kT)
        super()._attach_hook()


class OverdampedViscous(Method):
    r"""Overdamped viscous dynamics.

    Args:
        filter (hoomd.filter.filter_like): Subset of particles to apply this
            method to.

        alpha (float): When set, use :math:`\alpha d_i` for the
            drag coefficient where :math:`d_i` is particle diameter
            :math:`[\mathrm{mass} \cdot \mathrm{length}^{-1}
            \cdot \mathrm{time}^{-1}]`.
            Defaults to ``None``

    `OverdampedViscous` integrates particles forward in time following
    Newtonian dynamics in the overdamped limit where there is no inertial term.
    (in the limit that the mass :math:`m` and moment of inertia :math:`I` go to
    0):

    .. math::

        \frac{d\vec{r}}{dt} &= \vec{v}

        \vec{v(t)} &= \frac{\vec{F}_\mathrm{C}}{\gamma}

        \frac{d\mathbf{q}}{dt} &= \vec{\tau}

        \tau^i &= \frac{\tau_\mathrm{C}^i}{\gamma_r^i}

    where :math:`\vec{F}_\mathrm{C} = \vec{F}_\mathrm{net}` is the net force on
    the particle from all forces (`hoomd.md.Integrator.forces`) and constraints
    (`hoomd.md.Integrator.constraints`), :math:`\gamma` is the translational
    drag coefficient (`gamma`) :math:`\vec{v}` is the particle's velocity,
    :math:`d` is the dimensionality of the system, :math:`\tau_\mathrm{C}^i` is
    the i-th component of the net torque from all forces and constraints, and
    :math:`\gamma_r^i` is the i-th component of the rotational drag coefficient
    (`gamma_r`).

    You can set :math:`\gamma` in two ways:

    1. Specify :math:`\alpha` which scales the particle diameter to
       :math:`\gamma = \alpha d_i`.
    2. After the method object is created, specify the attribute `gamma`
       and `gamma_r` for rotational damping or random torque to assign them
       directly, with independent values for each particle type in the
       system.

    Tip:
        `OverdampedViscous` can be used to simulate systems of athermal active
        matter, such as athermal Active Brownian Particles.

    Note:
        Even though `OverdampedViscous` models systems in the limit that
        :math:`m` and moment of inertia :math:`I` go to 0, you must still set
        non-zero moments of inertia to enable the integration of rotational
        degrees of freedom.

    Examples::

        odv = hoomd.md.methods.OverdampedViscous(filter=hoomd.filter.All())
        odv.gamma.default = 2.0
        odv.gamma_r.default = [1.0, 2.0, 3.0]

    Attributes:
        filter (hoomd.filter.filter_like): Subset of particles to apply this
            method to.

        alpha (float): When set, use :math:`\alpha d_i` for the drag
            coefficient where :math:`d_i` is particle diameter
            :math:`[\mathrm{mass} \cdot \mathrm{length}^{-1}
            \cdot \mathrm{time}^{-1}]`.

        gamma (TypeParameter[ ``particle type``, `float` ]): The drag
            coefficient can be directly set instead of the ratio of particle
            diameter (:math:`\gamma = \alpha d_i`). The type of ``gamma``
            parameter is either positive float or zero
            :math:`[\mathrm{mass} \cdot \mathrm{time}^{-1}]`.

        gamma_r (TypeParameter[``particle type``, [`float`, `float`, `float`]]):
            The rotational drag coefficient can be set. The type of ``gamma_r``
            parameter is a tuple of three float. The type of each element of
            tuple is either positive float or zero
            :math:`[\mathrm{force} \cdot \mathrm{length} \cdot
            \mathrm{radian}^{-1} \cdot \mathrm{time}^{-1}]`.
    """

    def __init__(self, filter, alpha=None):

        # store metadata
        param_dict = ParameterDict(
            filter=ParticleFilter,
            alpha=OnlyTypes(float, allow_none=True),
        )
        param_dict.update(dict(alpha=alpha, filter=filter))

        # set defaults
        self._param_dict.update(param_dict)

        gamma = TypeParameter('gamma',
                              type_kind='particle_types',
                              param_dict=TypeParameterDict(1., len_keys=1))

        gamma_r = TypeParameter('gamma_r',
                                type_kind='particle_types',
                                param_dict=TypeParameterDict((1., 1., 1.),
                                                             len_keys=1))
        self._extend_typeparam([gamma, gamma_r])

    def _attach_hook(self):
        """Class uses RNGs. Warn the user if they did not set the seed."""
        self._simulation._warn_if_seed_unset()
        sim = self._simulation
        if isinstance(sim.device, hoomd.device.CPU):
            self._cpp_obj = _md.TwoStepBD(sim.state._cpp_sys_def,
                                          sim.state._get_group(self.filter),
                                          hoomd.variant.Constant(0.0), True,
                                          True)
        else:
            self._cpp_obj = _md.TwoStepBDGPU(sim.state._cpp_sys_def,
                                             sim.state._get_group(self.filter),
                                             hoomd.variant.Constant(1.0), True,
                                             True)

        # Attach param_dict and typeparam_dict
        super()._attach_hook()
