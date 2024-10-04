# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""MD integration methods.

.. code-block:: python

    simulation = hoomd.util.make_example_simulation()
    simulation.operations.integrator = hoomd.md.Integrator(dt=0.001)
    logger = hoomd.logging.Logger()

    # Rename pytest's tmp_path fixture for clarity in the documentation.
    path = tmp_path
"""

from hoomd.md import _md
import hoomd
from hoomd.operation import AutotunedObject
from hoomd.data.parameterdicts import ParameterDict, TypeParameterDict
from hoomd.data.typeparam import TypeParameter
from hoomd.data.typeconverter import OnlyTypes, OnlyIf, to_type_converter
from hoomd.filter import ParticleFilter
from hoomd.variant import Variant
from collections.abc import Sequence
from .thermostats import Thermostat


class Method(AutotunedObject):
    """Base class integration method.

    Provides common methods for all subclasses.

    Note:
        Users should use the subclasses and not instantiate `Method` directly.
    """

    def _attach_hook(self):
        self._simulation.state.update_group_dof()

    def _detach_hook(self):
        if (sim := self._simulation) is not None:
            sim.state.update_group_dof()


class Thermostatted(Method):
    r"""Base class for thermostatted integrators.

    Provides a common interface for all methods using thermostats

    Note:
        Users should use the subclasses and not instantiate `Thermostatted`
        directly.
    """
    _remove_for_pickling = AutotunedObject._remove_for_pickling + ("_thermo",)
    _skip_for_equality = AutotunedObject._skip_for_equality | {
        "_thermo",
    }

    def _setattr_param(self, attr, value):
        if attr == "thermostat":
            self._thermostat_setter(value)
            return
        super()._setattr_param(attr, value)

    def _thermostat_setter(self, new_thermostat):
        if new_thermostat is self.thermostat:
            return

        if new_thermostat is None:
            if self._attached:
                self._cpp_obj.setThermostat(None)
            self._param_dict._dict["thermostat"] = None
            return

        if new_thermostat._attached:
            raise RuntimeError("Trying to set a thermostat that is "
                               "already attached")
        if self._attached:
            new_thermostat._set_thermo(self.filter, self._thermo)
            new_thermostat._attach(self._simulation)
            self._cpp_obj.setThermostat(new_thermostat._cpp_obj)
        self._param_dict._dict["thermostat"] = new_thermostat


class ConstantVolume(Thermostatted):
    r"""Constant volume dynamics.

    Args:
        filter (hoomd.filter.filter_like): Subset of particles on which to
            apply this method.

        thermostat (hoomd.md.methods.thermostats.Thermostat): Thermostat to
            control temperature. Setting this to ``None`` samples a constant
            energy (NVE, microcanonical) dynamics. Defaults to ``None``.

    `ConstantVolume` numerically integrates the translational degrees of freedom
    using Velocity-Verlet and the rotational degrees of freedom with a scheme
    based on `Kamberaj 2005`_.

    When set, the `thermostat` rescales the particle velocities to model a
    canonical (NVT) ensemble. Use no thermostat (``thermostat = None``) to
    perform constant energy integration.

    See Also:
        `hoomd.md.methods.thermostats`.

    .. rubric:: Examples:

    NVE integration:

    .. code-block:: python

        nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
        simulation.operations.integrator.methods = [nve]

    NVT integration:

    .. code-block:: python

        nvt = hoomd.md.methods.ConstantVolume(
            filter=hoomd.filter.All(),
            thermostat=hoomd.md.methods.thermostats.Bussi(kT=1.5))
        simulation.operations.integrator.methods = [nvt]

    Attributes:
        filter (hoomd.filter.filter_like): Subset of particles on which to apply
            this method.

        thermostat (hoomd.md.methods.thermostats.Thermostat): Temperature
            control for the integrator.

            .. rubric:: Examples:

            .. code-block:: python

                nvt.thermostat.kT = 1.0

            .. code-block:: python

                nvt.thermostat = hoomd.md.methods.thermostats.Bussi(kT=0.5)

    .. _Kamberaj 2005: https://dx.doi.org/10.1063/1.1906216
    """

    def __init__(self, filter, thermostat=None):
        super().__init__()
        # store metadata
        param_dict = ParameterDict(filter=ParticleFilter,
                                   thermostat=OnlyTypes(Thermostat,
                                                        allow_none=True))
        param_dict.update(dict(filter=filter, thermostat=thermostat))
        # set defaults
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        # initialize the reflected cpp class
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cls = _md.TwoStepConstantVolume
            thermo_cls = _md.ComputeThermo
        else:
            cls = _md.TwoStepConstantVolumeGPU
            thermo_cls = _md.ComputeThermoGPU

        group = self._simulation.state._get_group(self.filter)
        cpp_sys_def = self._simulation.state._cpp_sys_def
        self._thermo = thermo_cls(cpp_sys_def, group)

        if self.thermostat is None:
            self._cpp_obj = cls(cpp_sys_def, group, None)
        else:
            self.thermostat._set_thermo(self.filter, self._thermo)
            self.thermostat._attach(self._simulation)
            self._cpp_obj = cls(cpp_sys_def, group, self.thermostat._cpp_obj)
        super()._attach_hook()


class ConstantPressure(Thermostatted):
    r"""Constant pressure dynamics.

    Args:
        filter (hoomd.filter.filter_like): Subset of particles on which to apply
            this method.

        thermostat (hoomd.md.methods.thermostats.Thermostat): Thermostat to
            control temperature. Setting this to ``None`` samples a constant
            enthalpy (NPH) integration.

        S (tuple[variant.variant_like, ...] or variant.variant_like):
            Stress component set points for the barostat.

            In Voigt notation:
            :math:`[S_{xx}, S_{yy}, S_{zz}, S_{yz}, S_{xz}, S_{xy}]`
            :math:`[\mathrm{pressure}]`. In case of isotropic
            pressure P use ``S = p`` to imply (:math:`[p, p, p, 0, 0, 0]`).

        tauS (float): Coupling constant for the barostat
           :math:`[\mathrm{time}]`.

        couple (str): Couplings of diagonal elements of the stress tensor.
            One of "none", "xy", "xz","yz", or "xyz".

        box_dof(`list` [ `bool` ]): Box degrees of freedom with six boolean
            elements in the order x, y, z, xy, xz, yz. Defaults to
            [True,True,True,False,False,False]). When True, rescale
            corresponding lengths or tilt factors and components of particle
            coordinates and velocities.

        rescale_all (bool): When True, rescale all particles, not only those
            selected by the filter. Defaults to False.

        gamma (float): Friction constant for the box degrees of freedom.
            Defaults to 0 :math:`[\mathrm{time}^{-1}]`.

    `ConstantPressure` integrates translational and rotational degrees of
    freedom of the system held at constant pressure with a barostat. The
    barostat introduces additional degrees of freedom in the Hamiltonian that
    couple with box parameters. Use a thermostat to model an isothermal-isobaric
    (NPT) ensemble. Use no thermostat (``thermostat = None``) to model a
    isoenthalpic-isobaric (NPH) ensemble.

    See Also:
        `hoomd.md.methods.thermostats`.

    The barostat tensor is :math:`\nu_{\mathrm{ij}}`. Access these quantities
    using `barostat_dof`.

    By default, `ConstantPressure` performs integration in a cubic box under
    hydrostatic pressure by simultaneously rescaling the lengths *Lx*, *Ly* and
    *Lz* of the simulation box by the same factors. Set the couplings and/or
    box degrees of freedom to change this default.

    Couplings define which diagonal elements of the pressure tensor
    :math:`P_{\alpha,\beta}` should be averaged over, so that the corresponding
    box lengths are rescaled by the same amount.

    Valid couplings are:

    - ``'none'`` (all box lengths are updated independently)
    - ``'xy'`` (*Lx* and *Ly* are coupled)
    - ``'xz'`` (*Lx* and *Lz* are coupled)
    - ``'yz'`` (*Ly* and *Lz* are coupled)
    - ``'xyz'`` (*Lx*, *Ly*, and *Lz* are coupled)

    The degrees of freedom of the box set which lengths and tilt factors of the
    box should be updated, and how particle coordinates and velocities should be
    rescaled. The ``box_dof`` tuple controls the way the box is rescaled and
    updated. The first three elements ``box_dof[:3]`` controls whether the *x*,
    *y*, and *z* box lengths are rescaled and updated, respectively. The last
    three entries ``box_dof[3:]`` control the rescaling or the tilt factors
    *xy*, *xz*, and *yz*. All options also appropriately rescale particle
    coordinates and velocities.

    By default, the *x*, *y*, and *z* degrees of freedom are updated.
    ``[True,True,True,False,False,False]``

    Note:
        When any of the diagonal *x*, *y*, *z* degrees of freedom is not being
        integrated, pressure tensor components along that direction are not
        considered for the remaining degrees of freedom.

    `ConstantPressure` numerically integrates the equations of motion using the
    symplectic Martyna-Tobias-Klein integrator with a Langevin piston. The
    equation of motion of box dimensions is given by:

    .. math::

        \frac{d^2 L}{dt^2} &= V W^{-1} (S - S_{ext})
            - \gamma \frac{dL}{dt} + R(t)

        \langle R \rangle &= 0

        \langle |R|^2 \rangle &= 2 \gamma kT \delta t W^{-1}

    Where :math:`\gamma` is the friction on the barostat piston, which damps
    unphysical volume oscillations at the cost of non-deterministic integration,
    and :math:`R` is a random force, chosen appropriately for the coupled
    degrees of freedom.

    See Also:
        * `G. J. Martyna, D. J. Tobias, M. L. Klein  1994
          <https://dx.doi.org/10.1063/1.467468>`__
        * `S. E. Feller, Y. Zhang, R. W. Pastor, B. R. Brooks 1995
          <https://doi.org/10.1063/1.470648>`_
        * `M. E. Tuckerman et. al. 2006
          <https://dx.doi.org/10.1088/0305-4470/39/19/S18>`__
        * `T. Yu et. al. 2010
          <https://dx.doi.org/10.1016/j.chemphys.2010.02.014>`_

    Note:
        The barostat coupling constant `tauS` should be set within a reasonable
        range to avoid abrupt fluctuations in the box volume and to avoid long
        time to equilibration. The recommended value for most systems is
        :math:`\tau_S = 1000 \delta t`.

    Note:
        If :math:`\gamma` is used, its value should be chosen so that the system
        is near critical damping. A good initial guess is
        :math:`\gamma \approx 2 \tau_S^{-1}`. A value too high will result in
        long relaxation times.

    Note:
        Set `gamma` = 0 to obtain the same MTK equations of motion used in
        HOOMD-blue releases prior to 4.0.0.

    .. rubric:: Examples:

    NPH integrator with cubic symmetry:

    .. code-block:: python

        nph = hoomd.md.methods.ConstantPressure(filter=hoomd.filter.All(),
                                                tauS=1.0,
                                                S=2.0,
                                                couple="xyz")
        simulation.operations.integrator.methods = [nph]

    NPT integrator with cubic symmetry:

    .. code-block:: python

        npt = hoomd.md.methods.ConstantPressure(
            filter=hoomd.filter.All(),
            tauS=1.0,
            S=2.0,
            couple="xyz",
            thermostat=hoomd.md.methods.thermostats.Bussi(kT=1.5))
        simulation.operations.integrator.methods = [npt]

    NPT integrator with tetragonal symmetry:

    .. code-block:: python

        npt = hoomd.md.methods.ConstantPressure(
            filter=hoomd.filter.All(),
            tauS = 1.0,
            S=2.0,
            couple="xy",
            thermostat=hoomd.md.methods.thermostats.Bussi(kT=1.5))
        simulation.operations.integrator.methods = [npt]

    NPT integrator with orthorhombic symmetry:

    .. code-block:: python

        npt = hoomd.md.methods.ConstantPressure(
            filter=hoomd.filter.All(),
            tauS = 1.0,
            S=2.0,
            couple="none",
            thermostat=hoomd.md.methods.thermostats.Bussi(kT=1.5))
        simulation.operations.integrator.methods = [npt]


    NPT integrator with triclinic symmetry:

    .. code-block:: python

        npt = hoomd.md.methods.ConstantPressure(
            filter=hoomd.filter.All(),
            tauS = 1.0,
            S=2.0,
            couple="none",
            box_dof=[True, True, True, True, True, True],
            thermostat=hoomd.md.methods.thermostats.Bussi(kT=1.5))
        simulation.operations.integrator.methods = [npt]


    Attributes:
        filter (hoomd.filter.filter_like): Subset of particles on which to apply
            this method.

        thermostat (hoomd.md.methods.thermostats.Thermostat): Temperature
            control for the integrator.

        S (tuple[hoomd.variant.Variant,...]): Stress components set point for
            the barostat.
            In Voigt notation,
            :math:`[S_{xx}, S_{yy}, S_{zz}, S_{yz}, S_{xz}, S_{xy}]`
            :math:`[\mathrm{pressure}]`.

            .. rubric:: Examples:

            .. code-block:: python

                npt.S = 4.0

            .. code-block:: python

                npt.S = hoomd.variant.Ramp(A=1.0,
                                           B=2.0,
                                           t_start=0,
                                           t_ramp=1_000_000)

        tauS (float): Coupling constant for the barostat
            :math:`[\mathrm{time}]`.

            .. rubric:: Example:

            .. code-block:: python

                npt.tauS = 2.0

        couple (str): Couplings of diagonal elements of the stress tensor,
            can be 'none', 'xy', 'xz', 'yz', or 'xyz'.

            .. rubric:: Example:

            .. code-block:: python

                npt.couple = 'none'

        box_dof(list[bool]): Box degrees of freedom with six boolean elements in
            the order [x, y, z, xy, xz, yz].

            .. rubric:: Example:

            .. code-block:: python

                npt.box_dof = [False, False, True, False, False, False]

        rescale_all (bool): When True, rescale all particles, not only those
            selected by the filter.

            .. rubric:: Example:

            .. code-block:: python

                npt.rescale_all = True

        gamma (float): Friction constant for the box degrees of freedom
            :math:`[\mathrm{time^{-1}}]`.

        barostat_dof (tuple[float, float, float, float, float, float]):
            Additional degrees of freedom for the barostat (:math:`\nu_{xx}`,
            :math:`\nu_{xy}`, :math:`\nu_{xz}`, :math:`\nu_{yy}`,
            :math:`\nu_{yz}`, :math:`\nu_{zz}`)

            Save and restore the barostat degrees of freedom when continuing
            simulations:

            .. rubric:: Examples:

            Save before exiting:

            .. code-block:: python

                numpy.save(file=path / 'barostat_dof.npy',
                           arr=npt.barostat_dof)

            Load when continuing:

            .. code-block:: python

                npt = hoomd.md.methods.ConstantPressure(
                    filter=hoomd.filter.All(),
                    tauS=1.0,
                    S=2.0,
                    couple="xyz",
                    thermostat=hoomd.md.methods.thermostats.Bussi(kT=1.5))
                simulation.operations.integrator.methods = [npt]

                npt.barostat_dof = numpy.load(file=path / 'barostat_dof.npy')
    """

    def __init__(self,
                 filter,
                 S,
                 tauS,
                 couple,
                 thermostat=None,
                 box_dof=[True, True, True, False, False, False],
                 rescale_all=False,
                 gamma=0.0):
        super().__init__()
        # store metadata
        param_dict = ParameterDict(filter=ParticleFilter,
                                   thermostat=OnlyTypes(Thermostat,
                                                        allow_none=True),
                                   S=OnlyIf(to_type_converter((Variant,) * 6),
                                            preprocess=self._preprocess_stress),
                                   tauS=float(tauS),
                                   couple=str(couple),
                                   box_dof=[
                                       bool,
                                   ] * 6,
                                   rescale_all=bool(rescale_all),
                                   gamma=float(gamma),
                                   barostat_dof=(float, float, float, float,
                                                 float, float))
        param_dict.update(
            dict(filter=filter,
                 thermostat=thermostat,
                 S=S,
                 couple=couple,
                 box_dof=box_dof,
                 barostat_dof=(0, 0, 0, 0, 0, 0)))

        # set defaults
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        # initialize the reflected c++ class
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cpp_cls = _md.TwoStepConstantPressure
            thermo_cls = _md.ComputeThermo
        else:
            cpp_cls = _md.TwoStepConstantPressureGPU
            thermo_cls = _md.ComputeThermoGPU

        cpp_sys_def = self._simulation.state._cpp_sys_def
        thermo_group = self._simulation.state._get_group(self.filter)

        # Only save the half step thermo
        self._thermo = thermo_cls(cpp_sys_def, thermo_group)
        thermo_full_step = thermo_cls(cpp_sys_def, thermo_group)

        if self.thermostat is None:
            self._cpp_obj = cpp_cls(cpp_sys_def, thermo_group, thermo_full_step,
                                    self.tauS, self.S, self.couple,
                                    self.box_dof, None, self.gamma)
        else:
            self.thermostat._set_thermo(self.filter, self._thermo)
            self.thermostat._attach(self._simulation)

            self._cpp_obj = cpp_cls(cpp_sys_def, thermo_group, thermo_full_step,
                                    self.tauS, self.S, self.couple,
                                    self.box_dof, self.thermostat._cpp_obj,
                                    self.gamma)

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

        `thermalize_barostat_dof` sets random values for the the barostat
        momentum :math:`\nu_{\mathrm{ij}}`.

        .. important::
            You must call `Simulation.run` before `thermalize_barostat_dof`.

            .. code-block:: python

                simulation.run(0)
                npt.thermalize_barostat_dof()

        .. seealso::

            `State.thermalize_particle_momenta`

            `hoomd.md.methods.thermostats.MTTK.thermalize_dof`
        """
        if not self._attached:
            raise RuntimeError("Call Simulation.run(0) before"
                               "thermalize_barostat_dof")

        self._simulation._warn_if_seed_unset()
        self._cpp_obj.thermalizeBarostatDOF(self._simulation.timestep)

    @hoomd.logging.log(requires_run=True)
    def barostat_energy(self):
        """Energy the barostat contributes to the Hamiltonian \
        :math:`[\\mathrm{energy}]`.

        .. rubric:: Example:

        .. code-block:: python

            logger.add(obj=npt, quantities=['barostat_energy'])
        """
        return self._cpp_obj.getBarostatEnergy(self._simulation.timestep)


class DisplacementCapped(ConstantVolume):
    r"""Newtonian dynamics with a cap on the maximum displacement per time step.

    The method limits particle motion to a maximum displacement allowed each
    time step which may be helpful to relax a high energy initial condition.

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

    .. rubric:: Example:

    .. code-block:: python

        displacement_capped = hoomd.md.methods.DisplacementCapped(
            filter=hoomd.filter.All(),
            maximum_displacement=1e-3)
        simulation.operations.integrator.methods = [displacement_capped]

    Attributes:
        filter (hoomd.filter.filter_like): Subset of particles on which to
            apply this method.

        maximum_displacement (hoomd.variant.variant_like): The maximum
            displacement allowed for a particular timestep
            :math:`[\mathrm{length}]`.

            .. code-block:: python

                displacement_capped.maximum_displacement = 1e-5
    """

    def __init__(self, filter,
                 maximum_displacement: hoomd.variant.variant_like):

        # store metadata
        super().__init__(filter)
        param_dict = ParameterDict(maximum_displacement=hoomd.variant.Variant)
        param_dict["maximum_displacement"] = maximum_displacement

        # set defaults
        self._param_dict.update(param_dict)


class Langevin(Method):
    r"""Langevin dynamics.

    Args:
        filter (hoomd.filter.filter_like): Subset of particles to
            apply this method to.

        kT (hoomd.variant.variant_like): Temperature of the simulation
            :math:`[\mathrm{energy}]`.

        tally_reservoir_energy (bool): When True, track the energy exchange
            between the thermal reservoir and the particles.
            Defaults to False :math:`[\mathrm{energy}]`.

        default_gamma (float): Default drag coefficient for all particle types
            :math:`[\mathrm{mass} \cdot \mathrm{time}^{-1}]`.

        default_gamma_r ([`float`, `float`, `float`]): Default rotational drag
            coefficient tensor for all particles :math:`[\mathrm{mass} \cdot
            \mathrm{length}^{2} \cdot \mathrm{time}^{-1}]`.

    `Langevin` integrates particles forward in time according to the
    Langevin equations of motion, modelling a canonical ensemble (NVT).

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
    temperature, :math:`kT`.

    About axes where :math:`I^i > 0`, the rotational degrees of freedom follow:

    .. math::

        I \frac{d\vec{\omega}}{dt} &= \vec{\tau}_\mathrm{C} - \gamma_r \cdot
        \vec{\omega} + \vec{\tau}_\mathrm{R}

        \langle \vec{\tau}_\mathrm{R} \rangle &= 0,

        \langle \tau_\mathrm{R}^i \cdot \tau_\mathrm{R}^i \rangle &=
        2 k T \gamma_r^i / \delta t,

    where :math:`\vec{\tau}_\mathrm{C} = \vec{\tau}_\mathrm{net}`,
    :math:`\gamma_r^i` is the i-th component of the rotational drag coefficient
    (`gamma_r`), :math:`\tau_\mathrm{R}^i` is a component of the uniform random
    torque, :math:`\vec{\omega}` is the particle's angular velocity and
    :math:`I` is the the particle's moment of inertia. The magnitude of the
    random torque is chosen via the fluctuation-dissipation theorem to be
    consistent with the specified drag and temperature, :math:`kT`.

    `Langevin` numerically integrates the translational degrees of freedom
    using Velocity-Verlet and the rotational degrees of freedom with a scheme
    based on `Kamberaj 2005`_.

    The attributes `gamma` and `gamma_r` set the translational and rotational
    damping coefficients, respectivley, by particle type.

    .. rubric:: Example:

    .. code-block:: python

        langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=1.5)
        simulation.operations.integrator.methods = [langevin]

    .. _Kamberaj 2005: https://dx.doi.org/10.1063/1.1906216

    Attributes:
        filter (hoomd.filter.filter_like): Subset of particles to
            apply this method to.

        kT (hoomd.variant.Variant): Temperature of the
            simulation :math:`[\mathrm{energy}]`.

            .. rubric:: Examples:

            .. code-block:: python

                langevin.kT = 1.0

            .. code-block:: python

                langevin.kT = hoomd.variant.Ramp(A=2.0,
                                                 B=1.0,
                                                 t_start=0,
                                                 t_ramp=1_000_000)

        tally_reservoir_energy (bool): When True, track the energy exchange
            between the thermal reservoir and the particles.

            .. rubric:: Example:

            .. code-block:: python

                langevin.tally_reservoir_energy = True

        gamma (TypeParameter[ ``particle type``, `float` ]): The drag
            coefficient for each particle type
            :math:`[\mathrm{mass} \cdot \mathrm{time}^{-1}]`.

            .. rubric:: Example:

            .. code-block:: python

                langevin.gamma['A'] = 0.5

        gamma_r (TypeParameter[``particle type``,[`float`, `float` , `float`]]):
            The rotational drag coefficient tensor for each particle type
            :math:`[\mathrm{mass} \cdot \mathrm{length}^{2} \cdot
            \mathrm{time}^{-1}]`.

            .. rubric:: Example:

            .. code-block:: python

                langevin.gamma_r['A'] = [1.0, 2.0, 3.0]
    """

    def __init__(
            self,
            filter,
            kT,
            tally_reservoir_energy=False,
            default_gamma=1.0,
            default_gamma_r=(1.0, 1.0, 1.0),
    ):

        # store metadata
        param_dict = ParameterDict(
            filter=ParticleFilter,
            kT=Variant,
            tally_reservoir_energy=bool(tally_reservoir_energy),
        )
        param_dict.update(dict(kT=kT, filter=filter))
        # set defaults
        self._param_dict.update(param_dict)

        gamma = TypeParameter('gamma',
                              type_kind='particle_types',
                              param_dict=TypeParameterDict(float, len_keys=1))
        gamma.default = default_gamma

        gamma_r = TypeParameter('gamma_r',
                                type_kind='particle_types',
                                param_dict=TypeParameterDict(
                                    (float, float, float), len_keys=1))

        gamma_r.default = default_gamma_r

        self._extend_typeparam([gamma, gamma_r])

    def _attach_hook(self):
        """Langevin uses RNGs. Warn the user if they did not set the seed."""
        self._simulation._warn_if_seed_unset()
        sim = self._simulation
        if isinstance(sim.device, hoomd.device.CPU):
            cls = _md.TwoStepLangevin
        else:
            cls = _md.TwoStepLangevinGPU

        self._cpp_obj = cls(sim.state._cpp_sys_def,
                            sim.state._get_group(self.filter), self.kT)

        # Attach param_dict and typeparam_dict
        super()._attach_hook()

    @hoomd.logging.log(requires_run=True)
    def reservoir_energy(self):
        """Energy absorbed by the reservoir :math:`[\\mathrm{energy}]`.

        Set `tally_reservoir_energy` to `True` to track the reservoir energy.

        .. rubric:: Example:

        .. code-block:: python

            langevin.tally_reservoir_energy = True
            logger.add(obj=langevin, quantities=['reservoir_energy'])

        Warning:
            When continuing a simulation, the energy of the reservoir will be
            reset to zero.
        """
        return self._cpp_obj.reservoir_energy


class Brownian(Method):
    r"""Brownian dynamics.

    Args:
        filter (hoomd.filter.filter_like): Subset of particles to
            apply this method to.

        kT (hoomd.variant.variant_like): Temperature of the simulation
            :math:`[\mathrm{energy}]`.

        default_gamma (float): Default drag coefficient for all particle types
            :math:`[\mathrm{mass} \cdot \mathrm{time}^{-1}]`.

        default_gamma_r ([`float`, `float`, `float`]): Default rotational drag
            coefficient tensor for all particles :math:`[\mathrm{time}^{-1}]`.

    `Brownian` integrates particles forward in time according to the overdamped
    Langevin equations of motion, sometimes called Brownian dynamics or the
    diffusive limit.

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
    drag and temperature, :math:`kT`.

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
    drag and temperature, :math:`kT`.

    `Brownian` uses the numerical integration method from `I. Snook 2007`_, The
    Langevin and Generalised Langevin Approach to the Dynamics of Atomic,
    Polymeric and Colloidal Systems, section 6.2.5, with the exception that
    :math:`\vec{F}_\mathrm{R}` is drawn from a uniform random number
    distribution.

    .. _I. Snook 2007: https://dx.doi.org/10.1016/B978-0-444-52129-3.50028-6

    Warning:

        This numerical method has errors in :math:`O(\delta t)`, which is much
        larger than the errors of most other integration methods which are in
        :math:`O(\delta t^2)`. As a consequence, expect to use much smaller
        values of :math:`\delta t` with `Brownian` compared to e.g. `Langevin`
        or `ConstantVolume`.

    In Brownian dynamics, particle velocities and angular momenta are completely
    decoupled from positions. At each time step, `Brownian` draws a new velocity
    distribution consistent with the current set temperature so that
    `hoomd.md.compute.ThermodynamicQuantities` will report appropriate
    temperatures and pressures when logged or used by other methods.

    The attributes `gamma` and `gamma_r` set the translational and rotational
    damping coefficients, respectivley, by particle type.

    .. rubric:: Example:

    .. code-block:: python

        brownian = hoomd.md.methods.Brownian(filter=hoomd.filter.All(), kT=1.5)
        simulation.operations.integrator.methods = [brownian]

    Attributes:
        filter (hoomd.filter.filter_like): Subset of particles to apply this
            method to.

        kT (hoomd.variant.Variant): Temperature of the simulation
            :math:`[\mathrm{energy}]`.

            .. rubric:: Examples:

            .. code-block:: python

                brownian.kT = 1.0

            .. code-block:: python

                brownian.kT = hoomd.variant.Ramp(A=2.0,
                                                 B=1.0,
                                                 t_start=0,
                                                 t_ramp=1_000_000)

        gamma (TypeParameter[ ``particle type``, `float` ]): The drag
            coefficient for each particle type
            :math:`[\mathrm{mass} \cdot \mathrm{time}^{-1}]`.

            .. rubric:: Example:

            .. code-block:: python

                brownian.gamma['A'] = 0.5

        gamma_r (TypeParameter[``particle type``,[`float`, `float` , `float`]]):
            The rotational drag coefficient tensor for each particle type
            :math:`[\mathrm{time}^{-1}]`.

            .. rubric:: Example:

            .. code-block:: python

                brownian.gamma_r['A'] = [1.0, 2.0, 3.0]
    """

    def __init__(
            self,
            filter,
            kT,
            default_gamma=1.0,
            default_gamma_r=(1.0, 1.0, 1.0),
    ):

        # store metadata
        param_dict = ParameterDict(
            filter=ParticleFilter,
            kT=Variant,
        )
        param_dict.update(dict(kT=kT, filter=filter))

        # set defaults
        self._param_dict.update(param_dict)

        gamma = TypeParameter('gamma',
                              type_kind='particle_types',
                              param_dict=TypeParameterDict(float, len_keys=1))
        gamma.default = default_gamma

        gamma_r = TypeParameter('gamma_r',
                                type_kind='particle_types',
                                param_dict=TypeParameterDict(
                                    (float, float, float), len_keys=1))

        gamma_r.default = default_gamma_r
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


class OverdampedViscous(Method):
    r"""Overdamped viscous dynamics.

    Args:
        filter (hoomd.filter.filter_like): Subset of particles to apply this
            method to.

        default_gamma (float): Default drag coefficient for all particle types
            :math:`[\mathrm{mass} \cdot \mathrm{time}^{-1}]`.

        default_gamma_r ([`float`, `float`, `float`]): Default rotational drag
            coefficient tensor for all particles :math:`[\mathrm{time}^{-1}]`.

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
    drag coefficient (`gamma`), :math:`\vec{v}` is the particle's velocity,
    :math:`\tau_\mathrm{C}^i` is the i-th component of the net torque from all
    forces and constraints, and :math:`\gamma_r^i` is the i-th component of the
    rotational drag coefficient (`gamma_r`).

    The attributes `gamma` and `gamma_r` set the translational and rotational
    damping coefficients, respectivley, by particle type.

    Warning:

        This numerical method has errors in :math:`O(\delta t)`, which is much
        larger than the errors of most other integration methods which are in
        :math:`O(\delta t^2)`. As a consequence, expect to use much smaller
        values of :math:`\delta t` with `Brownian` compared to e.g. `Langevin`
        or `ConstantVolume`.

    Tip:
        `OverdampedViscous` can be used to simulate systems of athermal active
        matter.

    Note:
        `OverdampedViscous` models systems in the limit that :math:`m` and
        moment of inertia :math:`I` go to 0. However, you must still set
        non-zero moments of inertia to enable the integration of rotational
        degrees of freedom.

    .. rubric:: Example:

    .. code-block:: python

        overdamped_viscous = hoomd.md.methods.OverdampedViscous(
            filter=hoomd.filter.All())
        simulation.operations.integrator.methods = [overdamped_viscous]

    Attributes:
        filter (hoomd.filter.filter_like): Subset of particles to apply this
            method to.

        gamma (TypeParameter[ ``particle type``, `float` ]): The drag
            coefficient for each particle type
            :math:`[\mathrm{mass} \cdot \mathrm{time}^{-1}]`.

            .. rubric:: Example:

            .. code-block:: python

                overdamped_viscous.gamma['A'] = 0.5


        gamma_r (TypeParameter[``particle type``,[`float`, `float` , `float`]]):
            The rotational drag coefficient tensor for each particle type
            :math:`[\mathrm{time}^{-1}]`.

            .. rubric:: Example:

            .. code-block:: python

                overdamped_viscous.gamma_r['A'] = [1.0, 2.0, 3.0]
    """

    def __init__(
            self,
            filter,
            default_gamma=1.0,
            default_gamma_r=(1.0, 1.0, 1.0),
    ):

        # store metadata
        param_dict = ParameterDict(filter=ParticleFilter,)
        param_dict.update(dict(filter=filter))

        # set defaults
        self._param_dict.update(param_dict)

        gamma = TypeParameter('gamma',
                              type_kind='particle_types',
                              param_dict=TypeParameterDict(float, len_keys=1))
        gamma.default = default_gamma

        gamma_r = TypeParameter('gamma_r',
                                type_kind='particle_types',
                                param_dict=TypeParameterDict(
                                    (float, float, float), len_keys=1))

        gamma_r.default = default_gamma_r
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
