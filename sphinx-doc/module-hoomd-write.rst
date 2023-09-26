.. Copyright (c) 2009-2023 The Regents of the University of Michigan.
.. Part of HOOMD-blue, released under the BSD 3-Clause License.

hoomd.write
------------

.. rubric:: Overview

.. py:currentmodule:: hoomd.write

.. autosummary::
    :nosignatures:

    Burst
    DCD
    CustomWriter
    GSD
    HDF5Log
    Table

.. rubric:: Details

.. automodule:: hoomd.write
    :synopsis: Write data out.

    .. autoclass:: Burst
        :show-inheritance:
        :members:

    .. autoclass:: CustomWriter
        :show-inheritance:
        :members:

    .. autoclass:: DCD
        :show-inheritance:
        :members:

    .. autoclass:: GSD
        :show-inheritance:
        :members:

    .. autoclass:: HDF5Log(trigger, filename, logger, mode="a")
        :show-inheritance:
        :members:

    .. autoclass:: Table(trigger, logger, output=stdout, header_sep='.', delimiter=' ', pretty=True, max_precision=10, max_header_len=None)
        :show-inheritance:
        :members:
