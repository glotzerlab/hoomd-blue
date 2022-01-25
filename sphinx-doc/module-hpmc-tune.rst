.. Copyright (c) 2009-2022 The Regents of the University of Michigan.
.. Part of HOOMD-blue, released under the BSD 3-Clause License.

hpmc.tune
---------------

.. rubric:: Overview

.. py:currentmodule:: hoomd.hpmc.tune

.. autosummary::
    :nosignatures:

    MoveSize

.. rubric:: Details

.. automodule:: hoomd.hpmc.tune
    :synopsis: Tuners for HPMC.
    :members:

    .. autoclass:: MoveSize(trigger, moves, target, solver, types=None, max_move_size=None)
        :members: secant_solver, scale_solver

        .. method:: tuned()
            :property:

            Whether or not the moves sizes have converged to the desired acceptance rate.

            :type: bool
