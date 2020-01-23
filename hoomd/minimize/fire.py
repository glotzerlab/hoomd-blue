# coding: utf-8

# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

# Maintainer: joaander / All Developers are free to add commands for new
# features


from hoomd.md import _md
import hoomd
from hoomd.integrate import _integrator


class FIRE(_integrator):
    R""" Energy Minimizer (FIRE).

    Args:
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

        \frac{\sum |F|}{N*\sqrt{N_{dof}}} <ftol \;\; and \;\ \Delta \frac{\sum |E|}{N} < Etol

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
    def __init__(self, dt, Nmin=5, finc=1.1, fdec=0.5, alpha_start=0.1, falpha=0.99, ftol = 1e-1, wtol=1e-1, Etol= 1e-5, min_steps=10, aniso=None):

        # initialize base class
        _integrator.__init__(self)

        # initialize the reflected c++ class
        if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
            self.cpp_integrator = _md.FIREEnergyMinimizer(hoomd.context.current.system_definition, dt)
        else:
            self.cpp_integrator = _md.FIREEnergyMinimizerGPU(hoomd.context.current.system_definition, dt)

        self.supports_methods = True

        hoomd.context.current.system.setIntegrator(self.cpp_integrator)

        self.aniso = aniso

        if aniso is not None:
            self.set_params(aniso=aniso)

        # change the set parameters if not None
        self.dt = dt
        self.metadata_fields = ['dt','aniso']

        self.cpp_integrator.setNmin(Nmin)
        self.Nmin = Nmin
        self.metadata_fields.append('Nmin')

        self.cpp_integrator.setFinc(finc)
        self.finc = finc
        self.metadata_fields.append('finc')

        self.cpp_integrator.setFdec(fdec)
        self.fdec = fdec
        self.metadata_fields.append('fdec')

        self.cpp_integrator.setAlphaStart(alpha_start)
        self.alpha_start = alpha_start
        self.metadata_fields.append('alpha_start')

        self.cpp_integrator.setFalpha(falpha)
        self.falpha = falpha
        self.metadata_fields.append(falpha)

        self.cpp_integrator.setFtol(ftol)
        self.ftol = ftol
        self.metadata_fields.append(ftol)

        self.cpp_integrator.setWtol(wtol)
        self.wtol = wtol
        self.metadata_fields.append(wtol)

        self.cpp_integrator.setEtol(Etol)
        self.Etol = Etol
        self.metadata_fields.append(Etol)

        self.cpp_integrator.setMinSteps(min_steps)
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
        self.check_initialization()
        return self.cpp_integrator.getEnergy()

    def set_params(self, aniso=None):
        R""" Changes parameters of an existing integration mode.

        Args:
            aniso (bool): Anisotropic integration mode (bool), default None (autodetect).

        Examples::

            integrator_mode.set_params(aniso=False)

        """
        self.check_initialization()

        if aniso is not None:
            if aniso in self._aniso_modes:
                anisoMode = self._aniso_modes[aniso]
            else:
                hoomd.context.current.device.cpp_msg.error("integrate.mode_standard: unknown anisotropic mode {}.\n".format(aniso))
                raise RuntimeError("Error setting anisotropic integration mode.")
            self.aniso = aniso
            self.cpp_integrator.setAnisotropicMode(anisoMode)

    def has_converged(self):
        R""" Test if the energy minimizer has converged.

        Returns:
            True when the minimizer has converged. Otherwise, return False.
        """
        self.check_initialization()
        return self.cpp_integrator.hasConverged()

    def reset(self):
        R""" Reset the minimizer to its initial state.
        """
        self.check_initialization()
        return self.cpp_integrator.reset()
