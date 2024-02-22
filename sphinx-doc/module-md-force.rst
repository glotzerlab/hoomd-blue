.. Copyright (c) 2009-2024 The Regents of the University of Michigan.
.. Part of HOOMD-blue, released under the BSD 3-Clause License.

md.force
--------------

.. rubric:: Overview

.. py:currentmodule:: hoomd.md.force

.. autosummary::
    :nosignatures:

    Force
    Active
    ActiveOnManifold
    Constant
    Custom

.. rubric:: Details

.. automodule:: hoomd.md.force
    :synopsis: Apply forces to particles.

    .. autoclass:: Force
        :members:
        :show-inheritance:

    .. autoclass:: Active
        :show-inheritance:
        :no-inherited-members:
        :members: create_diffusion_updater

    .. autoclass:: ActiveOnManifold
        :show-inheritance:
        :members: create_diffusion_updater

    .. autoclass:: Constant
        :members:

    .. autoclass:: Custom
        :members:
