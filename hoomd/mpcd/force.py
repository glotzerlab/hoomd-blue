# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

r""" MPCD external force fields.

An external field specifies the force to be applied per MPCD particle in
the equations of motion (see :py:mod:`.mpcd.stream`). The external force should
be compatible with the chosen streaming geometry. Global momentum conservation is
typically broken by adding an external force field; care should be chosen that
the force field does not cause the system to net accelerate (i.e., it must maintain
*average* momentum conservation). Additionally, a thermostat will likely be
required to maintain temperature control in the driven system (see
:py:mod:`.mpcd.collide`).

.. note::

    The external force **must** be attached to a streaming method
    (see :py:mod:`.mpcd.stream`) using ``set_force`` to take effect.
    On its own, the force object will not affect the system.

"""

import hoomd
from hoomd import _hoomd

from . import _mpcd


class _force():
    r""" Base external force field.

    This base class does some basic initialization tests, and then constructs the
    polymorphic external field base class in C++. This base class is essentially a
    factory that can initialize other derived classes. New classes need to be exported
    in C++ with the appropriate template parameters, and then can be constructed at
    the python level by a deriving type. Use :py:class:`constant` as an example.

    """

    def __init__(self):
        # check for hoomd initialization
        if not hoomd.init.is_initialized():
            raise RuntimeError(
                'mpcd.force: system must be initialized before the external force.\n'
            )

        # check for mpcd initialization
        if hoomd.context.current.mpcd is None:
            hoomd.context.current.device.cpp_msg.error(
                'mpcd.force: an MPCD system must be initialized before the external force.\n'
            )
            raise RuntimeError('MPCD system not initialized')

        self._cpp = _mpcd.ExternalField(
            hoomd.context.current.device.cpp_exec_conf)


class block(_force):
    r""" Block force.

    Args:
        F (float): Magnitude of the force in *x* per particle.
        H (float or None): Half-width between centers of block regions.
        w (float or None): Half-width of blocks.

    Imposes a constant force in *x* as a function of position in *z*:

    .. math::
        :nowrap:

        \begin{equation}
        \mathbf{F} = \begin{cases}
        +F \mathbf{e}_x & |r_z - H| < w \\
        -F \mathbf{e}_x & |r_z + H| < w \\
           \mathbf{0}   & \mathrm{otherwise}
        \end{cases}
        \end{equation}

    The force is applied in blocks defined by *H* and *w* so that the force in *x*
    is :math:`+F` in the upper block, :math:`-F` in the lower block, and zero otherwise.
    The blocks must lie fully within the simulation box or an error will be raised.
    The blocks also should not overlap (the force will be zero in any overlapping
    regions), and a warning will be issued if the blocks overlap.

    This force field can be used to implement the double-parabola method for measuring
    viscosity by setting :math:`H = L_z/4` and :math:`w = L_z/4`, where :math:`L_z` is
    the size of the simulation box in *z*. If *H* or *w* is None, it will default to this
    value based on the current simulation box.

    Examples::

        # fully specified blocks
        force.block(F=1.0, H=5.0, w=5.0)

        # default blocks to full box
        force.block(F=0.5)

    .. note::

        The external force **must** be attached to a streaming method
        (see :py:mod:`.mpcd.stream`) using ``set_force`` to take effect.
        On its own, the force object will not affect the system.

    .. versionadded:: 2.6

    """

    def __init__(self, F, H=None, w=None):

        # current box size
        Lz = hoomd.context.current.system_definition.getParticleData(
        ).getGlobalBox().getL().z

        # setup default blocks if needed
        if H is None:
            H = Lz / 4
        if w is None:
            w = Lz / 4

        # validate block positions
        if H <= 0 or H > Lz / 2:
            hoomd.context.current.device.cpp_msg.error(
                'mpcd.force.block: H = {} should be nonzero and inside box.\n'
                .format(H))
            raise ValueError('Invalid block spacing')
        if w <= 0 or w > (Lz / 2 - H):
            hoomd.context.current.device.cpp_msg.error(
                'mpcd.force.block: w = {} should be nonzero and keep block in box (H = {}).\n'
                .format(w, H))
            raise ValueError('Invalid block width')
        if w > H:
            hoomd.context.current.device.cpp_msg.warning(
                'mpcd.force.block: blocks overlap with H = {} < w = {}.\n'
                .format(H, w))

        # initialize python level
        _force.__init__(self)
        self._F = F
        self._H = H
        self._w = w

        # initialize c++
        self._cpp.BlockForce(self.F, self.H, self.w)

    @property
    def F(self):
        return self._F

    @property
    def H(self):
        return self._H

    @property
    def w(self):
        return self._w


class constant(_force):
    r""" Constant force.

    Args:
        F (tuple): 3d vector specifying the force per particle.

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

    .. note::

        The external force **must** be attached to a streaming method
        (see :py:mod:`.mpcd.stream`) using ``set_force`` to take effect.
        On its own, the force object will not affect the system.

    .. versionadded:: 2.6

    """

    def __init__(self, F):

        try:
            if len(F) != 3:
                hoomd.context.current.device.cpp_msg.error(
                    'mpcd.force.constant: field must be a 3-component vector.\n'
                )
                raise ValueError('External field must be a 3-component vector')
        except TypeError:
            hoomd.context.current.device.cpp_msg.error(
                'mpcd.force.constant: field must be a 3-component vector.\n')
            raise ValueError('External field must be a 3-component vector')

        # initialize python level
        _force.__init__(self)
        self._F = F

        # initialize c++
        self._cpp.ConstantForce(
            _hoomd.make_scalar3(self.F[0], self.F[1], self.F[2]))

    @property
    def F(self):
        return self._F


class sine(_force):
    r""" Sine force.

    Args:
        F (float): Magnitude of the force in *x* per particle.
        k (float): Wavenumber for the force.

    Applies a force in *x* that is sinusoidally varying in *z*.

    .. math::

        \mathbf{F}(\mathbf{r}) = F \sin (k r_z) \mathbf{e}_x

    Typically, the wavenumber should be something that is commensurate
    with the simulation box. For example, :math:`k = 2\pi/L_z` will generate
    one period of the sine in :py:class:`~.stream.bulk` geometry.

    Examples::

        # one period
        k0 = 2.*np.pi/box.Lz
        force.sine(F=1.0, k=k0)

        # two periods
        force.sine(F=0.5, k=2*k0)

    The user will need to determine what value of *k* makes sense for their
    problem, as it is too difficult to validate all values of *k* for all
    streaming geometries.

    .. note::

        The external force **must** be attached to a streaming method
        (see :py:mod:`.mpcd.stream`) using ``set_force`` to take effect.
        On its own, the force object will not affect the system.

    .. versionadded:: 2.6

    """

    def __init__(self, F, k):

        # initialize python level
        _force.__init__(self)
        self._F = F
        self._k = k

        # initialize c++
        self._cpp.SineForce(self.F, self.k)

    @property
    def F(self):
        return self._F

    @property
    def k(self):
        return self._k
