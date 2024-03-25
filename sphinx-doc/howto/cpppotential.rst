.. Copyright (c) 2009-2024 The Regents of the University of Michigan.
.. Part of HOOMD-blue, released under the BSD 3-Clause License.

How to apply arbitrary pair potentials in HPMC
==============================================

To apply arbitrary pair potentials between particles in HPMC simulations:

1. Write the C++ code that evaluates the potential.
2. Instantiate a `hoomd.hpmc.pair.user.CPPPotential` or `hoomd.hpmc.pair.user.CPPPotentialUnion`
   with the code.
3. Set the `pair_potential <hoomd.hpmc.integrate.HPMCIntegrator.pair_potential>` property of the
   HPMC integrator.

This code demonstrates the hard sphere square well potential.

.. literalinclude:: cpppotential.py
    :language: python

Use HPMC with pair potentials to model interactions with discontinuous steps
in the potential. Use molecular dynamics for models with continuous potentials.
