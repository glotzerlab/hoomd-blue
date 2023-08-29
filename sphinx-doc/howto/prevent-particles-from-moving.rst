.. Copyright (c) 2009-2023 The Regents of the University of Michigan.
.. Part of HOOMD-blue, released under the BSD 3-Clause License.

How to prevent particles from moving
====================================

MD simulations
--------------

To prevent a subset of particles from moving in MD simulations, omit the stationary particles from the
filter (or filters) that you provide to your integration method (or methods). For example:

.. literalinclude:: prevent-particles-from-moving-md.py
    :language: python

HPMC simulations
----------------

To prevent a subset of particles from moving in HPMC simulations:

1. Use different types for stationary particles and mobile particles.
2. Set the move sizes of the stationary type to 0.
3. Set the shape of the stationary type accordingly.

For example:

.. literalinclude:: prevent-particles-from-moving-hpmc.py
    :language: python
