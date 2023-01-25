# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Provide classes for thermostatting simulations.

Classes are for use with `hoomd.md.methods.ConstantVolume` and
`hoomd.md.methods.ConstantPressure`.

Important:
    Ensure that your initial condition includes non-zero particle velocities and
    angular momenta (when appropriate). The coupling between the thermostat and
    the velocities / angular momenta occurs via multiplication, so the
    thermostat cannot convert a zero velocity into a non-zero one except
    through particle collisions.
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
    _remove_for_pickling = _HOOMDBaseObject._remove_for_pickling + ("_thermo",)
    _skip_for_equality = _HOOMDBaseObject._skip_for_equality | {
        "_thermo",
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

    Produces temperature control through a Nosé-Hoover thermostat.

    Args:
        kT (hoomd.variant.variant_like): Temperature set point for the
            thermostat :math:`[\mathrm{energy}]`.

        tau (float): Coupling constant for the thermostat
            :math:`[\mathrm{time}]`

    The translational thermostat has a momentum :math:`\xi` and position
    :math:`\eta`. The rotational thermostat has momentum
    :math:`\xi_{\mathrm{rot}}` and position :math:`\eta_\mathrm{rot}`. Access
    these quantities using `translational_thermostat_dof` and
    `rotational_thermostat_dof`.

    See Also:
        `G. J. Martyna, D. J. Tobias, M. L. Klein 1994
        <http://dx.doi.org/10.1063/1.467468>`_ and `J. Cao, G. J. Martyna 1996
        <http://dx.doi.org/10.1063/1.470959>`_.

    Attributes:
        kT (hoomd.variant.variant_like): Temperature set point for the
            thermostat :math:`[\mathrm{energy}]`.

        tau (float): Coupling constant for the thermostat
            :math:`[\mathrm{time}]`

        translational_dof (tuple[float, float]): Additional degrees
            of freedom for the translational thermostat (:math:`\xi`,
            :math:`\eta`)

        rotational_dof (tuple[float, float]): Additional degrees
            of freedom for the rotational thermostat (:math:`\xi_\mathrm{rot}`,
            :math:`\eta_\mathrm{rot}`)
    """

    def __init__(self, kT, tau):
        super().__init__(kT)
        param_dict = ParameterDict(tau=float(tau),
                                   translational_dof=(float, float),
                                   rotational_dof=(float, float))
        param_dict.update(
            dict(translational_dof=(0, 0),
                 rotational_dof=(0, 0)))
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        group = self._simulation.state._get_group(self._filter)
        self._cpp_obj = _md.MTTKThermostat(self.kT, group, self._thermo,
                                           self._simulation.state._cpp_sys_def,
                                           self.tau)

    @hoomd.logging.log(requires_run=True)
    def thermostat_energy(self):
        """Energy the thermostat contributes to the Hamiltonian \
        :math:`[\\mathrm{energy}]`."""
        return self._cpp_obj.getThermostatEnergy(self._simulation.timestep)

    def thermalize_dof(self):
        r"""Set the thermostat momenta to random values.

        `thermalize_dof` sets a random value for the momentum
        :math:`\xi`. When `Integrator.integrate_rotational_dof` is `True`, it
        also sets a random value for the rotational thermostat momentum
        :math:`\xi_{\mathrm{rot}}`. Call `thermalize_dof` to set a
        new random state for the thermostat.

        .. important::
            You must call `Simulation.run` before `thermalize_dof`.
            Call ``run(steps=0)`` to prepare a newly created `hoomd.Simulation`.

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

    Provides temperature control by rescaling the velocity by a factor taken
    from the canonical velocity distribution. On each timestep, velocities are
    rescaled by a factor :math:`\alpha=\sqrt{K_t / K}`, where :math:`K` is the
    current kinetic energy, and :math:`K_t` is chosen randomly from the
    distribution

    .. math::
        P(K_t) \propto K_t^{N_f/2 - 1} \exp(-K_t / kT)

    where :math:`N_f` is the number of degrees of freedom thermalized.

    See Also:
        `Bussi et. al. 2007 <https://doi.org/10.1063/1.2408420>`_.


    Attributes:
        kT (hoomd.variant.variant_like): Temperature set point
            for the thermostat :math:`[\mathrm{energy}]`.
    """

    def __init__(self, kT):
        super().__init__(kT)

    def _attach_hook(self):
        group = self._simulation.state._get_group(self._filter)
        self._cpp_obj = _md.BussiThermostat(self.kT, group, self._thermo,
                                            self._simulation.state._cpp_sys_def)
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
        `Berendsen` does not yield a proper canonical ensemble

    See Also:
        `Berendsen et. al. 1984 <http://dx.doi.org/10.1063/1.448118>`_.

    Attributes:
        kT (hoomd.variant.variant_like): Temperature of the simulation.
            :math:`[energy]`

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
