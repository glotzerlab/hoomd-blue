.. Copyright (c) 2009-2024 The Regents of the University of Michigan.
.. Part of HOOMD-blue, released under the BSD 3-Clause License.

How to tune move sizes in multicomponent HPMC systems
=====================================================

Set ``ignore_statistics`` in the HPMC integrator shape parameter dictionary to ``True`` for every
type except for one and run the tuner.
Repeat until every particle type has been the one with ``ignore_statistics`` set to ``False``.
For example:

.. literalinclude:: tune-mc-move-sizes-binary-system.py
    :language: python
