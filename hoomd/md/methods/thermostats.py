# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Provide classes for thermostatting simulations.

The thermostat classes are for use with `hoomd.md.methods.ConstantVolume` and
`hoomd.md.methods.ConstantPressure`.

.. invisible-code-block: python

    simulation = hoomd.util.make_example_simulation()
    constant_volume = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    simulation.operations.integrator = hoomd.md.Integrator(
        dt=0.001,
        methods=[constant_volume])

Important:
    Ensure that your initial condition includes non-zero particle velocities and
    angular momenta (when appropriate). The coupling between the thermostat and
    the velocities / angular momenta occurs via multiplication, so the
    thermostat cannot convert a zero velocity into a non-zero one except
    through particle collisions.

    .. rubric:: Example

    .. code-block:: python

        simulation.state.thermalize_particle_momenta(
            filter=hoomd.filter.All(),
            kT=1.5)

.. invisible-code-block: python

    # Rename pytest's tmp_path fixture for clarity in the documentation.
    path = tmp_path

    logger = hoomd.logging.Logger()
"""

from hoomd.md import _md
import hoomd
from hoomd.operation import _HOOMDBaseObject
from hoomd.data.parameterdicts import ParameterDict
from hoomd.variant import Variant


class Thermostat(_HOOMDBaseObject):
    """Base thermostat object class.

    Note:
        Users should use the subclasses and not instantiate `Thermostat`
        directly.
    """
    _remove_for_pickling = _HOOMDBaseObject._remove_for_pickling + ("_thermo",
                                                                    "_filter")
    _skip_for_equality = _HOOMDBaseObject._skip_for_equality | {
        "_thermo", "_filter"
    }

    def __init__(self, kT):
        param_dict = ParameterDict(kT=Variant)
        param_dict["kT"] = kT
        self._param_dict.update(param_dict)

    def _set_thermo(self, filter, thermo):
        self._filter = filter
        self._thermo = thermo


class MTTK(Thermostat):
    r"""The Nosé-Hoover thermostat.

    Controls the system temperature using velocity rescaling with the
    Nosé-Hoover thermostat.

    Args:
        kT (hoomd.variant.variant_like): Temperature set point for the
            thermostat :math:`[\mathrm{energy}]`.

        tau (float): Coupling constant for the thermostat
            :math:`[\mathrm{time}]`

    The translational thermostat has a momentum :math:`\xi` and position
    :math:`\eta`. The rotational thermostat has momentum
    :math:`\xi_{\mathrm{rot}}` and position :math:`\eta_\mathrm{rot}`. Access
    these quantities using `translational_dof` and `rotational_dof`.

    Note:
        The coupling constant `tau` should be set within a
        reasonable range to avoid abrupt fluctuations in the kinetic temperature
        and to avoid long time to equilibration. The recommended value for most
        systems is :math:`\tau = 100 \delta t`.

    See Also:
        `G. J. Martyna, D. J. Tobias, M. L. Klein 1994
        <https://dx.doi.org/10.1063/1.467468>`_ and `J. Cao, G. J. Martyna 1996
        <https://dx.doi.org/10.1063/1.470959>`_.

    .. rubric:: Examples:

    .. code-block:: python

        mttk = hoomd.md.methods.thermostats.MTTK(kT=1.5,
            tau=simulation.operations.integrator.dt*100)
        simulation.operations.integrator.methods[0].thermostat = mttk

    Attributes:
        kT (hoomd.variant.variant_like): Temperature set point for the
            thermostat :math:`[\mathrm{energy}]`.

            .. rubric:: Examples:

            .. code-block:: python

                mttk.kT = 1.0

            .. code-block:: python

                mttk.kT = hoomd.variant.Ramp(A=1.0,
                                             B=2.0,
                                             t_start=0,
                                             t_ramp=1_000_000)

        tau (float): Coupling constant for the thermostat
            :math:`[\mathrm{time}]`

            .. rubric:: Example:

            .. code-block:: python

                mttk.tau = 0.2

        translational_dof (tuple[float, float]): Additional degrees
            of freedom for the translational thermostat (:math:`\xi`,
            :math:`\eta`)

            Save and restore the thermostat degrees of freedom when continuing
            simulations:

            .. rubric:: Examples:

            Save before exiting:

            .. code-block:: python

                numpy.save(file=path / 'translational_dof.npy',
                           arr=mttk.translational_dof)

            Load when continuing:

            .. code-block:: python

                mttk = hoomd.md.methods.thermostats.MTTK(kT=1.5,
                    tau=simulation.operations.integrator.dt*100)
                simulation.operations.integrator.methods[0].thermostat = mttk

                mttk.translational_dof = numpy.load(
                    file=path / 'translational_dof.npy')


        rotational_dof (tuple[float, float]): Additional degrees
            of freedom for the rotational thermostat (:math:`\xi_\mathrm{rot}`,
            :math:`\eta_\mathrm{rot}`)

            Save and restore the thermostat degrees of freedom when continuing
            simulations:

            .. rubric:: Examples:

            Save before exiting:

            .. code-block:: python

                numpy.save(file=path / 'rotational_dof.npy',
                           arr=mttk.rotational_dof)

            Load when continuing:

            .. code-block:: python

                mttk = hoomd.md.methods.thermostats.MTTK(kT=1.5,
                    tau=simulation.operations.integrator.dt*100)
                simulation.operations.integrator.methods[0].thermostat = mttk

                mttk.rotational_dof = numpy.load(
                    file=path / 'rotational_dof.npy')
    """

    def __init__(self, kT, tau):
        super().__init__(kT)
        param_dict = ParameterDict(tau=float(tau),
                                   translational_dof=(float, float),
                                   rotational_dof=(float, float))
        param_dict.update(dict(translational_dof=(0, 0), rotational_dof=(0, 0)))
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        group = self._simulation.state._get_group(self._filter)
        self._cpp_obj = _md.MTTKThermostat(self.kT, group, self._thermo,
                                           self._simulation.state._cpp_sys_def,
                                           self.tau)

    @hoomd.logging.log(requires_run=True)
    def energy(self):
        """Energy the thermostat contributes to the Hamiltonian \
        :math:`[\\mathrm{energy}]`.

        .. rubric:: Example:

        .. code-block:: python

            logger.add(obj=mttk, quantities=['energy'])
        """
        return self._cpp_obj.getThermostatEnergy(self._simulation.timestep)

    def thermalize_dof(self):
        r"""Set the thermostat momenta to random values.

        `thermalize_dof` sets a random value for the momentum
        :math:`\xi`. When `Integrator.integrate_rotational_dof` is `True`, it
        also sets a random value for the rotational thermostat momentum
        :math:`\xi_{\mathrm{rot}}`. Call `thermalize_dof` to set a
        new random state for the thermostat.

        .. rubric:: Example

        .. code-block:: python

            mttk.thermalize_dof()

        .. important::
            You must call `Simulation.run` before `thermalize_dof`.

        .. seealso:: `State.thermalize_particle_momenta`
        """
        if not self._attached:
            raise RuntimeError(
                "Call Simulation.run(0) before attempting to thermalize the "
                "MTTK thermostat.")
        self._simulation._warn_if_seed_unset()
        self._cpp_obj.thermalizeThermostat(self._simulation.timestep)


class Bussi(Thermostat):
    r"""The Bussi-Donadio-Parrinello thermostat.

    Args:
        kT (hoomd.variant.variant_like): Temperature set point for the
            thermostat :math:`[\mathrm{energy}]`.

        tau (float): Thermostat time constant :math:`[\mathrm{time}]`.
            Defaults to 0.

    `Bussi` controls the system temperature by separately rescaling the velocity
    and angular momenta by the factor :math:`\alpha` sampled from the canonical
    distribution.

    When `tau` is 0, the stochastic evolution of system is instantly thermalized
    and :math:`\alpha` is given by:

    .. math::
        \alpha = \sqrt{\frac{g_N kT}{K}}

    where :math:`K` is the instantaneous kinetic energy of the corresponding
    translational or rotational degrees of freedom, :math:`N` is the number of
    degrees of freedom, and :math:`g_N` is a random value sampled from the
    distribution :math:`\mathrm{Gamma}(N, 1)`:

    .. math::
        f_N(g) = \frac{1}{\Gamma(N)} g^{N-1} e^{-g}.

    When `tau` is non-zero, the kinetic energies decay to equilibrium with the
    given characteristic time constant and :math:`\alpha` is given by:

    .. math::
        \alpha = \sqrt{e^{\delta t / \tau}
                 + (1 - e^{\delta t / \tau}) \frac{(2 g_{N-1} + n^2) kT}{2 K}
                 + 2 n \sqrt{e^{\delta t / \tau} (1-e^{\delta t / \tau})
                    \frac{kT}{2 K}}}

    where :math:`\delta t` is the step size and :math:`n` is a random value
    sampled from the normal distribution :math:`\mathcal{N}(0, 1)`.

    See Also:
        `Bussi et. al. 2007 <https://doi.org/10.1063/1.2408420>`_.

    .. rubric:: Example:

    .. code-block:: python

        bussi = hoomd.md.methods.thermostats.Bussi(kT=1.5,
            tau=simulation.operations.integrator.dt*20)
        simulation.operations.integrator.methods[0].thermostat = bussi

    Attributes:
        kT (hoomd.variant.variant_like): Temperature set point
            for the thermostat :math:`[\mathrm{energy}]`.

            .. rubric:: Examples:

            .. code-block:: python

                bussi.kT = 1.0

            .. code-block:: python

                bussi.kT = hoomd.variant.Ramp(A=1.0,
                                              B=2.0,
                                              t_start=0,
                                              t_ramp=1_000_000)

        tau (float): Thermostat time constant :math:`[\mathrm{time}].`

            .. rubric:: Example:

            .. code-block:: python

                bussi.tau = 0.0
    """

    def __init__(self, kT, tau=0.0):
        super().__init__(kT)
        param_dict = ParameterDict(tau=float)
        param_dict["tau"] = tau
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        group = self._simulation.state._get_group(self._filter)
        self._cpp_obj = _md.BussiThermostat(self.kT, group, self._thermo,
                                            self._simulation.state._cpp_sys_def,
                                            self.tau)
        self._simulation._warn_if_seed_unset()


class Berendsen(Thermostat):
    r"""The Berendsen thermostat.

    Args:
        kT (hoomd.variant.variant_like): Temperature of the simulation.
            :math:`[\mathrm{energy}]`

        tau (float): Thermostat time constant. :math:`[\mathrm{time}]`

    `Berendsen` rescales the velocities of all particles on each time step. The
    rescaling is performed so that the difference in the current temperature
    from the set point decays exponentially:

    .. math::

        \frac{dT_\mathrm{cur}}{dt} = \frac{T - T_\mathrm{cur}}{\tau}

    .. attention::
        `Berendsen` does **NOT** sample the correct distribution of kinetic
        energies.

    See Also:
        `Berendsen et. al. 1984 <https://dx.doi.org/10.1063/1.448118>`_.

    .. rubric:: Example:

    .. code-block:: python

        berendsen = hoomd.md.methods.thermostats.Berendsen(kT=1.5,
            tau=simulation.operations.integrator.dt * 10_000)
        simulation.operations.integrator.methods[0].thermostat = berendsen

    Attributes:
        kT (hoomd.variant.variant_like): Temperature of the simulation.
            :math:`[energy]`

            .. rubric:: Examples:

            .. code-block:: python

                berendsen.kT = 1.0

            .. code-block:: python

                berendsen.kT = hoomd.variant.Ramp(A=1.0,
                                                  B=2.0,
                                                  t_start=0,
                                                  t_ramp=1_000_000)

        tau (float): Time constant of thermostat. :math:`[time]`
    """

    def __init__(self, kT, tau):
        super().__init__(kT)
        param_dict = ParameterDict(tau=float)
        param_dict["tau"] = tau
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        group = self._simulation.state._get_group(self._filter)
        self._cpp_obj = _md.BerendsenThermostat(
            self.kT, group, self._thermo, self._simulation.state._cpp_sys_def,
            self.tau)
