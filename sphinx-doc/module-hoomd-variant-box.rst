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
    InverseVolumeRamp

.. rubric:: Details

.. automodule:: hoomd.variant.box
    :synopsis: Box variants.
    :no-members:

    .. autoclass:: Box()
    .. autoclass:: Constant(box)
        :show-inheritance:
    .. autoclass:: Ramp(initial_box, final_box, variant)
        :show-inheritance:
    .. autoclass:: InverseVolumeRamp(initial_box, final_volume, t_start, t_ramp)
        :show-inheritance:
