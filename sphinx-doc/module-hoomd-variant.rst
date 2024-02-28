.. Copyright (c) 2009-2024 The Regents of the University of Michigan.
.. Part of HOOMD-blue, released under the BSD 3-Clause License.

hoomd.variant
-------------

.. rubric:: Overview

.. py:currentmodule:: hoomd.variant

.. autosummary::
    :nosignatures:

    Constant
    Cycle
    Power
    Ramp
    Variant
    variant_like

.. rubric:: Details

.. automodule:: hoomd.variant
    :synopsis: Values that vary as a function of time step.
    :no-members:

    .. autoclass:: Constant(value)
        :members: __eq__
        :show-inheritance:
    .. autoclass:: Cycle(A, B, t_start, t_A, t_AB, t_B, t_BA)
        :members: __eq__
        :show-inheritance:
    .. autoclass:: Power(A, B, power, t_start, t_ramp)
        :members: __eq__
        :show-inheritance:
    .. autoclass:: Ramp(A, B, t_start, t_ramp)
        :members: __eq__
        :show-inheritance:
    .. autoclass:: Variant()
        :members: min, max, __getstate__, __setstate__
    .. autodata:: variant_like


.. rubric:: Modules

.. toctree::
   :maxdepth: 1

   module-hoomd-variant-box
