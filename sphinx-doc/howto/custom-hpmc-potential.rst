.. Copyright (c) 2009-2024 The Regents of the University of Michigan.
.. Part of HOOMD-blue, released under the BSD 3-Clause License.

How to apply arbitrary pair potentials in HPMC
==============================================

To apply arbitrary potentials between in HPMC simulations, you need to implement a C++
:doc:`component <../components>` that evaluates the energy. Fork the `hpmc-energy-template`_
repository and modify it to compute the desired pair and/or external potentials.

If you previously used ``CPPPotential`` potentials, you can copy and paste the C++ code into the template.
If you used ``param_array``, you will need to make some modifications to accept parameters from
Python.

.. _hpmc-energy-template: https://github.com/glotzerlab/hpmc-energy-template
