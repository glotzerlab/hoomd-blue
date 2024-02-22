.. Copyright (c) 2009-2024 The Regents of the University of Michigan.
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

    .. autoclass:: Burst(trigger, filename, filter=hoomd.filter.All(), mode='ab', dynamic=None, logger=None, max_burst_size=-1, write_at_start=False)
        :show-inheritance:
        :members:

    .. autoclass:: CustomWriter
        :show-inheritance:
        :members:

    .. autoclass:: DCD(trigger, filename, filter=hoomd.filter.All(), overwrite=False, unwrap_full=False, unwrap_rigid=False, angle_z=False)
        :show-inheritance:
        :members:

    .. autoclass:: GSD(trigger, filename, filter=hoomd.filter.All(), mode='ab', truncate=False, dynamic=None, logger=None)
        :show-inheritance:
        :members:

    .. autoclass:: HDF5Log(trigger, filename, logger, mode="a")
        :show-inheritance:
        :members:

    .. autoclass:: Table(trigger, logger, output=stdout, header_sep='.', delimiter=' ', pretty=True, max_precision=10, max_header_len=None)
        :show-inheritance:
        :members:
