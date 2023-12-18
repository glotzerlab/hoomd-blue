.. Copyright (c) 2009-2023 The Regents of the University of Michigan.
.. Part of HOOMD-blue, released under the BSD 3-Clause License.

hoomd.variant.box
-----------------

.. rubric:: Overview

.. py:currentmodule:: hoomd.variant.box

.. autosummary::
    :nosignatures:

    Box
    Constant
    Ramp
    LinearInverseVolume

.. rubric:: Details

.. automodule:: hoomd.variant.box
    :synopsis: Box variants.
    :no-members:

    .. autoclass:: Constant(box)
        :show-inheritance:
    .. autoclass:: Ramp(initial_box, final_box, t_start, t_ramp)
        :show-inheritance:
    .. autoclass:: LinearInverseVolume(initial_box, final_volume, t_start, t_ramp)
        :show-inheritance:
