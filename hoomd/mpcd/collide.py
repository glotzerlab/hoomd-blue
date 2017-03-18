# Copyright (c) 2009-2017 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: mphoward

R""" MPCD collision methods

MPCD collision rules.

"""

import hoomd
from hoomd.md import _md

from . import _mpcd

class _collision_method(hoomd.meta._metadata):
    def __init__(self, seed, period):
        # check for hoomd initialization
        if not hoomd.init.is_initialized():
            hoomd.context.msg.error("mpcd.collide: system must be initialized before collision method\n")
            raise RuntimeError('System not initialized')

        # check for mpcd initialization
        if hoomd.context.current.mpcd is None:
            hoomd.context.msg.error('mpcd.collide: an MPCD system must be initialized before the collision rule\n')
            raise RuntimeError('MPCD system not initialized')

        # check for multiple collision rule initializations
        if hoomd.context.current.mpcd._collide is not None:
            hoomd.context.msg.error('mpcd.collide: only one collision method can be set, use disable() first.\n')
            raise RuntimeError('Multiple initialization of collision method')

        hoomd.meta._metadata.__init__(self)
        self.metadata_fields = ['period','seed','enabled']

        self.period = period
        self.seed = seed
        self.enabled = True
        self._cpp = None

        hoomd.util.quiet_status()
        self.enable()
        hoomd.util.unquiet_status()

    def enable(self):
        """ Enable the collision method

        Examples::
            method.enable()

        Enabling the collision method adds it to the current MPCD system definition.
        Only one collision method can be attached to the system at any time.
        If another method is already set, :py:meth:`disable()` must be called
        first before switching.
        """
        hoomd.util.print_status_line()

        self.enabled = True
        hoomd.context.current.mpcd._collide = self

    def disable(self):
        """ Disable the collision method

        Examples::
            method.disable()

        Disabling the collision method removes it from the current MPCD system definition.
        Only one collision method can be attached to the system at any time, so
        use this method to remove the current collision method before adding another.
        """
        hoomd.util.print_status_line()

        self.enabled = False
        hoomd.context.current.mpcd._collide = None

class srd(_collision_method):
    def __init__(self, seed, period, angle):
        hoomd.util.print_status_line()

        _collision_method.__init__(self, seed, period)
        self.metadata_fields += ['angle']

        if not hoomd.context.exec_conf.isCUDAEnabled():
            collide_class = _mpcd.SRDCollisionMethod
        else:
            collide_class = _mpcd.SRDCollisionMethodGPU
        self._cpp = collide_class(hoomd.context.current.mpcd.data,
                                  hoomd.context.current.system.getCurrentTimeStep(),
                                  self.period,
                                  -1,
                                  self.seed,
                                  hoomd.context.current.mpcd._thermo)

        hoomd.util.quiet_status()
        self.set_params(angle=angle)
        hoomd.util.unquiet_status()

    def set_params(self, angle=None):
        hoomd.util.print_status_line()

        if angle is not None:
            self.angle = angle
            self._cpp.setRotationAngle(angle)
