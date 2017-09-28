# Copyright (c) 2009-2017 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: mphoward

R""" MPCD streaming methods

MPCD streaming methods.

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
        self.metadata_fields = ['period']

        self.period = period
        self._cpp = None

        # attach the streaming method to the system
        hoomd.context.current.mpcd._stream = self

class bulk(_streaming_method):
    """ Streaming method for bulk geometry.

    Args:
        period (int): Number of integration steps between collisions

    Examples::

        stream.bulk(period=10)

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
