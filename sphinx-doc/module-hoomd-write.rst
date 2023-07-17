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
    HDF5Logger
    Table

.. rubric:: Details

.. automodule:: hoomd.write
    :synopsis: Write data out.
    :members: Burst, DCD, CustomWriter, GSD
    :show-inheritance:

    .. autoclass:: HDF5Logger(trigger, filename, logger, mode="a")
        :members:

    .. autoclass:: Table(trigger, logger, output=stdout, header_sep='.', delimiter=' ', pretty=True, max_precision=10, max_header_len=None)
        :members:
