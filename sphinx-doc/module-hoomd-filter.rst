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
    SetDifference
    Tags
    Type
    Union

.. rubric:: Details

.. automodule:: hoomd.filter
    :synopsis: Particle selection filters.
    :no-members:

    .. autoclass:: ParticleFilter()
        :special-members: __call__, __hash__, __eq__, __str__
    .. autoclass:: All()
    .. autoclass:: CustomFilter()
        :special-members: __call__, __hash__, __eq__
    .. autoclass:: Intersection(f, g)
    .. autoclass:: Null()
    .. autoclass:: SetDifference(f, g)
    .. autoclass:: Tags(tags)
        :members: tags
    .. autoclass:: Type(types)
        :members: types
    .. autoclass:: Union(f, g)
