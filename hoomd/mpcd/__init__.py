# Copyright (c) 2009-2017 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: mphoward

R""" Multiparticle collision dynamics

MPCD is a mesoscale, particle-based simulation for hydrodynamics.

.. rubric:: Algorithm and applications

.. rubric:: Stability

:py:mod:`hoomd.mpcd` is **unstable**. (It is currently under development.)
**Maintainer:** Michael P. Howard, Princeton University.
"""

# these imports are necessary in order to link derived types between modules
import hoomd
from hoomd import _hoomd
from hoomd.md import _md

from hoomd.mpcd import collide
from hoomd.mpcd import data
from hoomd.mpcd import init
from hoomd.mpcd import stream
from hoomd.mpcd import update

class integrator(hoomd.integrate._integrator):
    """ MPCD integrator

    Args:
        dt (float): Each time step of the simulation :py:func:`hoomd.run()` will
                    advance the real time of the system forward by *dt* (in time units).
        aniso (bool): Whether to integrate rotational degrees of freedom (bool),
                      default None (autodetect).

    The MPCD integrator enables the MPCD algorithm concurrently with standard
    MD :py:mod:`~hoomd.md.integrate` methods. An integrator must be created
    in order for :py:mod:`~hoomd.mpcd.stream` and :py:mod:`~hoomd.mpcd.collide`
    methods to take effect. Embedded MD particles require the creation of an
    appropriate integration method. Typically, this will just be
    :py:class:`~hoomd.md.integrate.nve`.

    In MPCD simulations, *dt* defines the amount of time that the system is advanced
    forward every time step. MPCD streaming and collision steps can be defined to
    occur in multiples of *dt*. In these cases, any MD particle data will be updated
    every *dt*, while the MPCD particle data is updated asynchronously for performance.
    For example, if MPCD streaming happens every 5 steps, then the particle data will be
    updated as follows::

                0     1     2     3     4     5
        MD:     |---->|---->|---->|---->|---->|
        MPCD:   |---------------------------->|

    If the MPCD particle data is accessed via the snapshot interface at time
    step 3, it will actually contain the MPCD particle data for time step 5.
    The MD particles can be read at any time step because their positions
    are updated every step.

    Examples::

        mpcd.integrate.integrator(dt=0.1)
        mpcd.integrate.integrator(dt=0.01, aniso=True)

    """
    def __init__(self, dt, aniso=None):
        # check system is initialized
        if hoomd.context.current.mpcd is None:
            hoomd.context.msg.error('mpcd.integrate: an MPCD system must be initialized before the integrator\n')
            raise RuntimeError('MPCD system not initialized')

        hoomd.integrate._integrator.__init__(self)

        self.supports_methods = True
        self.dt = dt
        self.aniso = aniso
        self.metadata_fields = ['dt','aniso']

        # configure C++ integrator
        self.cpp_integrator = _mpcd.Integrator(hoomd.context.current.mpcd.data, self.dt)
        if hoomd.context.current.mpcd.comm is not None:
            self.cpp_integrator.setMPCDCommunicator(hoomd.context.current.mpcd.comm)
        hoomd.context.current.system.setIntegrator(self.cpp_integrator)

        if self.aniso is not None:
            hoomd.util.quiet_status()
            self.set_params(aniso=aniso)
            hoomd.util.unquiet_status()

    _aniso_modes = {
        None: _md.IntegratorAnisotropicMode.Automatic,
        True: _md.IntegratorAnisotropicMode.Anisotropic,
        False: _md.IntegratorAnisotropicMode.Isotropic}

    def set_params(self, dt=None, aniso=None):
        """ Changes parameters of an existing integration mode.

        Args:
            dt (float): New time step delta (if set) (in time units).
            aniso (bool): Anisotropic integration mode (bool), default None (autodetect).

        Examples::

            integrator.set_params(dt=0.007)
            integrator.set_params(dt=0.005, aniso=False)

        """
        hoomd.util.print_status_line()
        self.check_initialization()

        # change the parameters
        if dt is not None:
            self.dt = dt
            self.cpp_integrator.setDeltaT(dt)

        if aniso is not None:
            if aniso in self._aniso_modes:
                anisoMode = self._aniso_modes[aniso]
            else:
                hoomd.context.msg.error("mpcd.integrate: unknown anisotropic mode {}.\n".format(aniso))
                raise RuntimeError("Error setting anisotropic integration mode.")
            self.aniso = aniso
            self.cpp_integrator.setAnisotropicMode(anisoMode)

    def update_methods(self):
        self.check_initialization()

        # update the integration methods that are set
        self.cpp_integrator.removeAllIntegrationMethods()
        for m in hoomd.context.current.integration_methods:
            self.cpp_integrator.addIntegrationMethod(m.cpp_method)

        # ensure that the streaming and collision methods are up to date
        stream = hoomd.context.current.mpcd._stream
        if stream is not None:
            self.cpp_integrator.setStreamingMethod(stream._cpp)
        else:
            hoomd.context.msg.warning("Running mpcd without a streaming method!\n")
            self.cpp_integrator.removeStreamingMethod()

        collide = hoomd.context.current.mpcd._collide
        if collide is not None:
            if stream is not None and collide.period % stream.period != 0:
                hoomd.context.msg.error('mpcd.integrate: collision period must be multiple of integration period\n')
                raise ValueError('Collision period must be multiple of integration period')

            self.cpp_integrator.setCollisionMethod(collide._cpp)
        else:
            hoomd.context.msg.warning("Running mpcd without a collision method!\n")
            self.cpp_integrator.removeCollisionMethod()
