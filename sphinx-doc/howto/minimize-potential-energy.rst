.. Copyright (c) 2009-2024 The Regents of the University of Michigan.
.. Part of HOOMD-blue, released under the BSD 3-Clause License.

How to minimize the potential energy of a system
================================================

To minimize the potential energy of a system:

1. Use `hoomd.md.minimize.FIRE` as the integrator.
2. Apply forces to the particles.
3. Run simulation steps until the minimization converges.

For example:

.. literalinclude:: minimize-potential-energy.py
    :language: python
