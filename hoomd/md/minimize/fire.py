# coding: utf-8

# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

# Maintainer: joaander / All Developers are free to add commands for new
# features

"""Energy minimizer."""

import hoomd

from hoomd.data.parameterdicts import ParameterDict
from hoomd.data.typeconverter import OnlyFrom, OnlyTypes
from hoomd.md import _md
from hoomd.md.integrate import _DynamicIntegrator, _preprocess_aniso


def _raise_if_nonpositive(value):
    if value <= 0:
        raise ValueError("This value cannot be negative")
    return value


class FIRE(_DynamicIntegrator):
    """Energy Minimizer (FIRE).

    Args:
        dt (float):
            This is the maximum step size the minimizer is permitted to use.
            Consider the stability of the system when setting.
            math:`[\\mathrm{time}]`
        min_steps_adapt (int):
            Number of steps energy change is negative before allowing
            :math:`\\alpha` and :math:`\\delta t` to adapt.
        finc_dt (float):
            Factor to increase :math:`\\delta t` by
        fdec_dt (float):
            Factor to decrease :math:`\\delta t` by
        alpha_start (float):
            Initial (and maximum) :math:`\\alpha`
        fdec_alpha (float):
            Factor to decrease :math:`\\alpha t` by
        force_tol (float):
            force convergence criteria
            math:`[\\mathrm{force} / \\mathrm{mass}]`
        angmom_tol (float):
            angular momentum convergence criteria
            math:`[\\mathrm{energy} * \\mathrm{time}]`
        energy_tol (float):
            energy convergence criteria math:`[\\mathrm{energy}]`
        min_steps_conv (int):
            A minimum number of attempts before convergence criteria are
            considered
        aniso (bool):
            Whether to integrate rotational degrees of freedom (bool), default
            None (autodetect).

    `minimize.FIRE` uses the Fast Inertial Relaxation Engine (FIRE) algorithm
    to minimize the energy for a group of particles while keeping all other
    particles fixed.  This method is published in `Bitzek, et. al., PRL, 2006
    <http://dx.doi.org/10.1103/PhysRevLett.97.170201>`_.

    At each time step, :math:`\\delta t`, the algorithm uses the NVE Integrator
    to generate a x, v, and F, and then adjusts v according to

    .. math::

        \\vec{v} = (1-\\alpha)\\vec{v} + \\alpha \\hat{F}|\\vec{v}|

    where :math:`\\alpha` and :math:`\\delta t` are dynamically adaptive
    quantities.  While a current search has been lowering the energy of system
    for more than :math:`N_{min}` steps, :math:`\\alpha` is decreased by
    :math:`\\alpha \\rightarrow \\alpha fdec_{\\alpha}` and :math:`\\delta t`
    is increased by :math:`\\delta t \\rightarrow max(\\delta t \\cdot
    finc_{dt}, \\ \\delta t_{max})`. If the energy of the system increases (or
    stays the same), the velocity of the particles is set to 0, :math:`\\alpha
    \\rightarrow \\ \\alpha_{start}` and :math:`\\delta t \\rightarrow \\delta t
    \\cdot fdec_{\\alpha}`. Convergence is determined by both the force per
    particle and the change in energy per particle dropping below *angmom_tol*
    and *energy_tol*, respectively or

    .. math::

        \\frac{\\sum |F|}{N*\\sqrt{N_{dof}}} < angmom_{tol} \\;\\; and \\;\\
        \\Delta \\frac{\\sum|E|}{N} < energy_{tol}

    where N is the number of particles the minimization is acting over (i.e.
    the group size) Either of the two criterion can be effectively turned off
    by setting the tolerance to a large number.

    If the minimization is acted over a subset of all the particles in the
    system, the "other" particles will be kept frozen but will still interact
    with the particles being moved.

    Examples::

        fire = md.minimize.FIRE(dt=0.05)
        fire.force_tol = 1e-2
        fire.energy_tol = 1e-7
        fire.methods.append(methods.NVE(filter.All()))
        sim.operations.integrator = fire
        while not(fire.has_converged()):
           sim.run(100)

    Examples::

        fire = md.minimize.FIRE(dt=0.05)
        fire.methods.append(methods.NPH(filter.All(), P=0.0, gamma=.5))
        sim.operations.integrator = fire
        while not(fire.has_converged()):
           sim.run(100)

    Note:
        The algorithm requires a base integrator to update the particle
        position and velocities. Usually this will be either NVE (to minimize
        energy) or NPH (to minimize energy and relax the box). The quantity
        minimized is in any case the energy (not the enthalpy or any other
        quantity).

    Note:
        As a default setting, the algorithm will start with a :math:`\\delta t
        = \\frac{1}{10} \\delta t_{max}` and attempts at least 10 search steps.
        In practice, it was found that this prevents the simulation from making
        too aggressive a first step, but also from quitting before having found
        a good search direction. The minimum number of attempts can be set by
        the user.

    """
    _cpp_class_name = "FIREEnergyMinimizer"

    def __init__(self,
                 dt,
                 aniso='auto',
                 forces=None,
                 constraints=None,
                 methods=None,
                 rigid=None):

        super().__init__(forces, constraints, methods, rigid)

        self._param_dict.update(
            ParameterDict(
                dt=float(dt),
                aniso=OnlyFrom(['true', 'false', 'auto'],
                               preprocess=_preprocess_aniso),
                min_steps_adapt=OnlyTypes(int,
                                          preprocess=_raise_if_nonpositive),
                finc_dt=float,
                fdec_dt=float,
                alpha_start=float,
                fdec_alpha=float,
                force_tol=float,
                angmom_tol=float,
                energy_tol=float,
                min_steps_conv=OnlyTypes(int,
                                         preprocess=_raise_if_nonpositive),
                _defaults={
                    "aniso": "auto",
                    "min_steps_adapt": 5,
                    "finc_dt": 1.1,
                    "fdec_dt": 0.5,
                    "alpha_start": 0.1,
                    "fdec_alpha": 0.99,
                    "force_tol": 0.1,
                    "angmom_tol": 0.1,
                    "energy_tol": 1e-5,
                    "min_steps_conv": 10
                }))

    def _attach(self):
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cls = getattr(_md, self._cpp_class_name)
        else:
            cls = getattr(_md, self._cpp_class_name + "GPU")
        self._cpp_obj = cls(self._simulation.state._cpp_sys_def, self.dt)
        super()._attach()

    def get_energy(self):
        """Returns the energy after the last iteration of the minimizer."""
        return self._cpp_obj.getEnergy()

    def has_converged(self):
        """Test if the energy minimizer has converged.

        Returns:
            True when the minimizer has converged. Otherwise, return False.
        """
        return self._cpp_obj.hasConverged()

    def reset(self):
        """Reset the minimizer to its initial state."""
        return self._cpp_obj.reset()
