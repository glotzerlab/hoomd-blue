.. Copyright (c) 2009-2024 The Regents of the University of Michigan.
.. Part of HOOMD-blue, released under the BSD 3-Clause License.

hoomd.filter
------------

.. rubric:: Overview

.. py:currentmodule:: hoomd.filter

.. autosummary::
    :nosignatures:

    ParticleFilter
    All
    CustomFilter
    Intersection
    Null
    Rigid
    SetDifference
    Tags
    Type
    Union
    filter_like

.. rubric:: Details

.. automodule:: hoomd.filter
    :synopsis: Particle selection filters.
    :no-members:

    .. autoclass:: ParticleFilter()
        :special-members: __call__, __hash__, __eq__, __str__
    .. autoclass:: All()
    .. autoclass:: CustomFilter()
        :special-members: __call__
    .. autoclass:: Intersection(f, g)
    .. autoclass:: Null()
    .. autoclass:: Rigid(flags=("center",))
    .. autoclass:: SetDifference(f, g)
    .. autoclass:: Tags(tags)
        :members: tags
    .. autoclass:: Type(types)
        :members: types
    .. autoclass:: Union(f, g)
    .. autodata:: filter_like
