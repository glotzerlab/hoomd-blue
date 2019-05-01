# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: mphoward

R""" MPCD streaming methods

An MPCD streaming method is required to update the particle positions over time.
It is meant to be used in conjunction with an :py:class:`~hoomd.mpcd.integrator`
and collision method (see :py:mod:`~hoomd.mpcd.collide`). Particle positions are
propagated ballistically according to Newton's equations (without any acceleration)
for a time :math:`\Delta t`:

.. math::

    \mathbf{r}(t+\Delta t) = \mathbf{r}(t) + \mathbf{v}(t) \Delta t

where **r** and **v** are the particle position and velocity, respectively.

"""

import hoomd
from hoomd.md import _md

from . import _mpcd

class _streaming_method(hoomd.meta._metadata):
    """ Base streaming method

    Args:
        period (int): Number of integration steps between streaming step

    This class is not intended to be initialized directly by the user. Instead,
    initialize a specific streaming method directly. It is included in the documentation
    to supply signatures for common methods.

    """
    def __init__(self, period):
        # check for hoomd initialization
        if not hoomd.init.is_initialized():
            hoomd.context.msg.error("mpcd.stream: system must be initialized before streaming method\n")
            raise RuntimeError('System not initialized')

        # check for mpcd initialization
        if hoomd.context.current.mpcd is None:
            hoomd.context.msg.error('mpcd.stream: an MPCD system must be initialized before the streaming method\n')
            raise RuntimeError('MPCD system not initialized')

        # check for multiple collision rule initializations
        if hoomd.context.current.mpcd._stream is not None:
            hoomd.context.msg.error('mpcd.stream: only one streaming method can be created.\n')
            raise RuntimeError('Multiple initialization of streaming method')

        hoomd.meta._metadata.__init__(self)
        self.metadata_fields = ['period','enabled']

        self.period = period
        self.enabled = True
        self._cpp = None

        # attach the streaming method to the system
        hoomd.util.quiet_status()
        self.enable()
        hoomd.util.unquiet_status()

    def enable(self):
        """ Enable the streaming method

        Examples::

            method.enable()

        Enabling the streaming method adds it to the current MPCD system definition.
        Only one streaming method can be attached to the system at any time.
        If another method is already set, :py:meth:`disable()` must be called
        first before switching. Streaming will occur when the timestep is the next
        multiple of *period*.

        """
        hoomd.util.print_status_line()

        self.enabled = True
        hoomd.context.current.mpcd._stream = self

    def disable(self):
        """ Disable the streaming method

        Examples::

            method.disable()

        Disabling the streaming method removes it from the current MPCD system definition.
        Only one streaming method can be attached to the system at any time, so
        use this method to remove the current streaming method before adding another.

        """
        hoomd.util.print_status_line()

        self.enabled = False
        hoomd.context.current.mpcd._stream = None

    def set_period(self, period):
        """ Set the streaming period.

        Args:
            period (int): New streaming period.

        The MPCD streaming period can only be changed to a new value on a
        simulation timestep that is a multiple of both the previous *period*
        and the new *period*. An error will be raised if it is not.

        Examples::

            # The initial period is 5.
            # The period can be updated to 2 on step 10.
            hoomd.run_upto(10)
            method.set_period(period=2)

            # The period can be updated to 4 on step 12.
            hoomd.run_upto(12)
            hoomd.set_period(period=4)

        """
        hoomd.util.print_status_line()

        cur_tstep = hoomd.context.current.system.getCurrentTimeStep()
        if cur_tstep % self.period != 0 or cur_tstep % period != 0:
            hoomd.context.msg.error('mpcd.stream: streaming period can only be changed on multiple of current and new period.\n')
            raise RuntimeError('Streaming period can only be changed on multiple of current and new period')

        self._cpp.setPeriod(cur_tstep, period)
        self.period = period

class bulk(_streaming_method):
    """ Streaming method for bulk geometry.

    Args:
        period (int): Number of integration steps between collisions.

    :py:class:`bulk` performs the streaming step for MPCD particles in a fully
    periodic geometry (2D or 3D). This geometry is appropriate for modeling
    bulk fluids. The streaming time :math:`\Delta t` is equal to *period* steps
    of the :py:class:`~hoomd.mpcd.integrator`. For a pure MPCD fluid,
    typically *period* should be 1. When particles are embedded in the MPCD fluid
    through the collision step, *period* should be equal to the MPCD collision
    *period* for best performance. The MPCD particle positions will be updated
    every time the simulation timestep is a multiple of *period*. This is
    equivalent to setting a *phase* of 0 using the terminology of other
    periodic :py:mod:`~hoomd.update` methods.

    Example for pure MPCD fluid::

        mpcd.integrator(dt=0.1)
        mpcd.collide.srd(seed=42, period=1, angle=130.)
        mpcd.stream.bulk(period=1)

    Example for embedded particles::

        mpcd.integrator(dt=0.01)
        mpcd.collide.srd(seed=42, period=10, angle=130., group=hoomd.group.all())
        mpcd.stream.bulk(period=10)

    """
    def __init__(self, period):
        hoomd.util.print_status_line()

        _streaming_method.__init__(self, period)

        # create the base streaming class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            stream_class = _mpcd.StreamingMethod
        else:
            stream_class = _mpcd.StreamingMethodGPU
        self._cpp = stream_class(hoomd.context.current.mpcd.data,
                                 hoomd.context.current.system.getCurrentTimeStep(),
                                 self.period,
                                 0)
