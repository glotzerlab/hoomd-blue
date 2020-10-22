hoomd.filter
------------

.. rubric:: Overview

.. py:currentmodule:: hoomd.filter

.. autosummary::
    :nosignatures:

    ParticleFilter
    All
    Intersection
    SetDifference
    Tags
    Type
    Union

.. rubric:: Details

.. automodule:: hoomd.filter
    :synopsis: Particle selection filters.

    .. autoclass:: ParticleFilter()
        :special-members: __call__, __hash__, __eq__, __str__
    .. autoclass:: All()
    .. autoclass:: Intersection(f, g)
    .. autoclass:: SetDifference(f, g)
    .. autoclass:: Tags(tags)
        :members: tags
    .. autoclass:: Type(types)
        :members: types
    .. autoclass:: Union(f, g)
