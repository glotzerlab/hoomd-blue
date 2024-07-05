.. Copyright (c) 2009-2024 The Regents of the University of Michigan.
.. Part of HOOMD-blue, released under the BSD 3-Clause License.

How to compute the free energy of solids
========================================

Follow these steps to perform `Frenkel-Ladd`_ (*Einstein crystal*) or `Vega-Noya`_ (*Einstein
molecule*) approach to compute the free energy of solid phases.

MD simulations
--------------

1. Initialize your system in the ideal crystal structure.
2. Add stationary reference particles at the equilibrium sites.
3. Set the pair force ``r_cut`` to 0 between the reference particle type and all particle types.
4. Add bonds (`hoomd.md.bond.Harmonic`) between the stationary reference particles and the mobile
   particles.
5. Integrate the equations of motion of the mobile particles.

   .. seealso::

       `prevent-particles-from-moving`.

6. Adjust the strength of bonds as needed during the simulation.

   .. seealso::

       `continuously-vary-potential-parameters`.

7. When using the *Einstein crystal* approach, apply `hoomd.update.RemoveDrift` to adjust the center
   of mass of the system.

HPMC simulations
----------------

1. Initialize your system in the ideal crystal structure.
2. Apply the harmonic external potential (`hoomd.hpmc.external.field.Harmonic`), using variants
   to adjust the spring constants as needed during the simulation.
3. Apply trial moves to the mobile particles.

   .. seealso::

       `prevent-particles-from-moving`.

4. When using the *Einstein crystal* approach, apply `hoomd.update.RemoveDrift` to adjust the center
   of mass of the system.

Log and analyze energies
------------------------

In both MD and HPMC simulations:

1. Log relevant energies (e.g. with `hoomd.write.HDF5Log`).
2. Post process the logged data to compute the free energy following the procedures in
   `Frenkel-Ladd`_ or `Vega-Noya`_ as appropriate.

.. _Frenkel-Ladd: https://doi.org/10.1063/1.448024
.. _Vega-Noya: https://doi.org/10.1063/1.2790426
