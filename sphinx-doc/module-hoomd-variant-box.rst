.. Copyright (c) 2009-2024 The Regents of the University of Michigan.
.. Part of HOOMD-blue, released under the BSD 3-Clause License.

hoomd.variant.box
-----------------

.. rubric:: Overview

.. py:currentmodule:: hoomd.variant.box

.. autosummary::
    :nosignatures:

    BoxVariant
    Constant
    Interpolate
    InverseVolumeRamp

.. rubric:: Details

.. automodule:: hoomd.variant.box
    :synopsis: Box variants.
    :no-members:

    .. autoclass:: BoxVariant()
    .. autoclass:: Constant(box)
        :show-inheritance:
    .. autoclass:: Interpolate(initial_box, final_box, variant)
        :show-inheritance:
    .. autoclass:: InverseVolumeRamp(initial_box, final_volume, t_start, t_ramp)
        :show-inheritance:
