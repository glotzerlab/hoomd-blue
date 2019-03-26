# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: mphoward

R""" MPCD external force fields.

"""

import hoomd
from hoomd import _hoomd

from . import _mpcd

class _force(hoomd.meta._metadata):
    """ Base external force field.

    This base class does some basic initialization tests, and then constructs the
    polymorphic external field base class in C++. This base class is essentially a
    factory that can initialize other derived classes. New classes need to be exported
    in C++ with the appropriate template parameters, and then can be constructed at
    the python level by a deriving type. Use :py:class:`constant` as an example.

    """
    def __init__(self):
        # check for hoomd initialization
        if not hoomd.init.is_initialized():
            hoomd.context.msg.error("mpcd.force: system must be initialized before the external force.\n")
            raise RuntimeError('System not initialized')

        # check for mpcd initialization
        if hoomd.context.current.mpcd is None:
            hoomd.context.msg.error('mpcd.force: an MPCD system must be initialized before the external force.\n')
            raise RuntimeError('MPCD system not initialized')

        hoomd.meta._metadata.__init__(self)
        self._cpp = _mpcd.ExternalField(hoomd.context.exec_conf)
        self.metadata_fields = []

class constant(_force):
    """ Constant force.

    Args:
        field (tuple): 3d vector specifying the force per particle.

    The same constant-force is applied to all particles, independently of time
    and their positions. This force is useful for simulating pressure-driven
    flow in conjunction with a confined geometry (e.g., :py:class:`~.stream.slit`)
    having no-slip boundary conditions.

    Examples::

        # tuple
        force.constant((1.,0.,0.))

        # list
        force.constant([1.,2.,3.])

        # NumPy array
        g = np.array([0.,0.,-1.])
        force.constant(g)

    """
    def __init__(self, field):
        hoomd.util.print_status_line()

        try:
            if len(field) != 3:
                hoomd.context.msg.error('mpcd.force.constant: field must be a 3-component vector.\n')
                raise ValueError('External field must be a 3-component vector')
        except TypeError:
            hoomd.context.msg.error('mpcd.force.constant: field must be a 3-component vector.\n')
            raise ValueError('External field must be a 3-component vector')

        # initialize python level
        _force.__init__(self)
        self.metadata_fields += ['field']
        self._field = field

        # initialize c++
        self._cpp.ConstantForce(_hoomd.make_scalar3(self.field[0], self.field[1], self.field[2]))

    @property
    def field(self):
        return self._field
