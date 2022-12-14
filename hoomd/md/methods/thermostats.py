# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

from hoomd.md import _md
import hoomd
from hoomd.operation import _HOOMDBaseObject
from hoomd.data.parameterdicts import ParameterDict
from hoomd.variant import Variant


class Thermostat(_HOOMDBaseObject):
    """ Base thermostat object class

    Provides common methods for thermostat objects

    Note:
        Users should use the subclasses and not instantiate `Thermostat`
        directly.
    """

    def __init__(self, kT):
        param_dict = ParameterDict(kT=Variant)
        param_dict["kT"] = kT
        self._param_dict.update(param_dict)
        self._group = None
        self._computeThermo = None

    def _setGroupThermo(self, group, thermo):
        self._group = group
        self._computeThermo = thermo


class ConstantEnergy(Thermostat):
    def __init__(self):
        super().__init__(1.0)
        self._param_dict.pop('kT', None)

    def _attach_hook(self):
        self._cpp_obj = _md.Thermostat(hoomd.variant.Constant(1.0), None, None, None)


class MTTK(Thermostat):
    r"""Nosé-Hoover thermostat

    Produces themperature control through a Nosé-Hoover thermostat

    Args:
        kT (hoomd.variant.variant_like): Temperature set point
            for the thermostat :math:`[\mathrm{energy}]`.

        tau (float): Coupling constant for the thermostat
            :math:`[\mathrm{time}]`

    The translational thermostat has a momentum :math:`\xi` and position
    :math:`\eta`. The rotational thermostat has momentum
    :math:`\xi_{\mathrm{rot}}` and position :math:`\eta_\mathrm{rot}`. Access
    these quantities using `translational_thermostat_dof` and
    `rotational_thermostat_dof`.

    Important:
        Ensure that your initial condition includes non-zero particle velocities
        and angular momenta (when appropriate). The coupling between the
        thermostat and the velocities and angular momenta occurs via
        multiplication, so `MTTK` cannot convert a zero velocity into a non-zero
        one except through particle collisions.

    Attributes:
        kT (hoomd.variant.variant_like): Temperature set point
            for the thermostat :math:`[\mathrm{energy}]`.

        tau (float): Coupling constant for the thermostat
            :math:`[\mathrm{time}]`

        translational_thermostat_dof (tuple[float, float]): Additional degrees
            of freedom for the translational thermostat (:math:`\xi`,
            :math:`\eta`)

        rotational_thermostat_dof (tuple[float, float]): Additional degrees
            of freedom for the rotational thermostat (:math:`\xi_\mathrm{rot}`,
            :math:`\eta_\mathrm{rot}`)

    """

    def __init__(self, kT, tau):
        super().__init__(kT)
        param_dict = ParameterDict(tau=float(tau),
                                   translational_thermostat_dof=(float, float),
                                   rotational_thermostat_dof=(float, float))
        param_dict.update(
            dict(translational_thermostat_dof=(0, 0),
                 rotational_thermostat_dof=(0, 0))
        )
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        self._cpp_obj = _md.MTTKThermostat(self.kT, self._group,
                                           self._computeThermo, self.tau)

    @hoomd.logging.log(requires_run=True)
    def thermostat_energy(self):
        """Energy the thermostat contributes to the Hamiltonian \
        :math:`[\\mathrm{energy}]`."""
        return self._cpp_obj.getThermostatEnergy(self._simulation.timestep)

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
        if not self.is_attached:
            raise StandardError("Call run(0) before attempting to thermalize the MTTK thermostat")
        self._simulation._warn_if_seed_unset()
        self._cpp_obj.thermalizeThermostat(self._simulation.timestep)


class Bussi(Thermostat):
    r"""Bussi-Donadio-Parrinello thermostat

    Args:
        kT (hoomd.variant.variant_like): Temperature set point
            for the thermostat :math:`[\mathrm{energy}]`.

    Provides temperature control by rescaling the velocity by a factor taken
    from the canonical velocity distribution. On each timestep, velocities are
    rescaled by a factor :math:`\alpha=\sqrt{K_t / K}`, where :math:`K` is the
    current kinetic energy, and :math:`K_t` is taken from

    .. math::
        P(K_t) \propto K_t^{N_f/2 - 1} \exp(-K_t / kT)

    where :math:`N_f` is the number of degrees of freedom thermalized

    Attributes:
        kT (hoomd.variant.variant_like): Temperature set point
            for the thermostat :math:`[\mathrm{energy}]`.

    """

    def __init__(self, kT):
        super().__init__(kT)

    def _attach_hook(self):
        self._cpp_obj = _md.BussiThermostat(self.kT, self._group,
                                            self._computeThermo,
                                            self._simulation.state._cpp_sys_def)
        self._simulation._warn_if_seed_unset()


class Berendsen(Thermostat):
    r"""Berendsen thermostat

    Args:
        kT (hoomd.variant.variant_like): Temperature of the
            simulation. :math:`[\mathrm{energy}]`

        tau (float): Time constant of thermostat. :math:`[\mathrm{time}]`

    `Berendsen` rescales the velocities of all particles on each time step. The
    rescaling is performed so that the difference in the current temperature
    from the set point decays exponentially:
    `Berendsen et. al. 1984 <http://dx.doi.org/10.1063/1.448118>`_.

    .. math::

        \frac{dT_\mathrm{cur}}{dt} = \frac{T - T_\mathrm{cur}}{\tau}

    .. attention::
        `Berendsen` does not yield a proper canonical ensemble

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
        self._cpp_obj = _md.BerendsenThermostat(self.kT, self._group, self._computeThermo, self.tau, self._simulation.state._cpp_sys_def)
