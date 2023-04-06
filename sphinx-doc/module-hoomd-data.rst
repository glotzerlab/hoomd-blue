.. Copyright (c) 2009-2023 The Regents of the University of Michigan.
.. Part of HOOMD-blue, released under the BSD 3-Clause License.

hoomd.data
------------------

.. rubric:: Overview

.. py:currentmodule:: hoomd.data

.. autosummary::
    :nosignatures:

    LocalSnapshot
    LocalSnapshotGPU
    AngleLocalAccessBase
    BondLocalAccessBase
    ConstraintLocalAccessBase
    DihedralLocalAccessBase
    ImproperLocalAccessBase
    PairLocalAccessBase
    ParticleLocalAccessBase
    hoomd.data.typeparam.TypeParameter

.. rubric:: Details

.. automodule:: hoomd.data
    :synopsis: Provide access in Python to data buffers on CPU or GPU.
    :members: AngleLocalAccessBase,
              BondLocalAccessBase,
              ConstraintLocalAccessBase,
              DihedralLocalAccessBase,
              ImproperLocalAccessBase,
              PairLocalAccessBase,
              ParticleLocalAccessBase

    .. autoclass:: LocalSnapshot
        :inherited-members:

    .. autoclass:: LocalSnapshotGPU
        :inherited-members:

    .. autoclass:: hoomd.data.typeparam.TypeParameter
        :members: default

.. rubric:: Modules

.. toctree::
   :maxdepth: 3

   module-hoomd-array
