# coding: utf-8

# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

# Maintainer: joaander / All Developers are free to add commands for new
# features


from hoomd.md import _md
import hoomd
from hoomd.integrate import _integrator


class Integrator(_integrator):
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

        # initialize base class
        _integrator.__init__(self)

        # Store metadata
        self.dt = dt
        self.aniso = aniso
        self.metadata_fields = ['dt', 'aniso']

        # initialize the reflected c++ class
        self.cpp_integrator = _md.IntegratorTwoStep(hoomd.context.current.system_definition, dt)
        self.supports_methods = True

        hoomd.context.current.system.setIntegrator(self.cpp_integrator)

        if aniso is not None:
            self.set_params(aniso=aniso)

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
        self.check_initialization()

        # change the parameters
        if dt is not None:
            self.dt = dt
            self.cpp_integrator.setDeltaT(dt)

        if aniso is not None:
            if aniso in self._aniso_modes:
                anisoMode = self._aniso_modes[aniso]
            else:
                hoomd.context.current.device.cpp_msg.error("integrate.mode_standard: unknown anisotropic mode {}.\n".format(aniso))
                raise RuntimeError("Error setting anisotropic integration mode.")
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
        self.check_initialization()
        self.cpp_integrator.initializeIntegrationMethods()
