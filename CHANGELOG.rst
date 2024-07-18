.. Copyright (c) 2009-2024 The Regents of the University of Michigan.
.. Part of HOOMD-blue, released under the BSD 3-Clause License.

Change Log
==========

5.x
---

5.0.0 (not yet released)
^^^^^^^^^^^^^^^^^^^^^^^^

Removed:

* ``_InternalCustomUpdater.update``.
  (`#1699 <https://github.com/glotzerlab/hoomd-blue/pull/1699>`__).
* ``_InternalCustomTuner.tune``.
  (`#1699 <https://github.com/glotzerlab/hoomd-blue/pull/1699>`__).
* ``_InternalCustomWriter.write``.
  (`#1699 <https://github.com/glotzerlab/hoomd-blue/pull/1699>`__).
* ``HDF5Log.write``.
  (`#1699 <https://github.com/glotzerlab/hoomd-blue/pull/1699>`__).
* ``hoomd.util.GPUNotAvailableError``
  (`#1694 <https://github.com/glotzerlab/hoomd-blue/pull/1694>`__).

4.x
---

4.8.1 (2024-07-18)
^^^^^^^^^^^^^^^^^^

Fixed:

* Prevent illegal instruction when accessing 0 length snapshot arrays
  (`#1846 <https://github.com/glotzerlab/hoomd-blue/pull/1846>`__)
* Fix MPCD compiler warning.
  (`#1845 <https://github.com/glotzerlab/hoomd-blue/pull/1845>`__)

4.8.0 (2024-07-11)
^^^^^^^^^^^^^^^^^^

*Added*

* ``hoomd.mpcd`` reimplements the MPCD method for simulating hydrodynamic interactions.
  See the migrating page for an overview and individual class and method documentation for more
  information (`#1784 <https://github.com/glotzerlab/hoomd-blue/pull/1784>`__).
* MPCD tutorial.
* Support numpy 2.0
  (`#1797 <https://github.com/glotzerlab/hoomd-blue/pull/1797>`__)
* ``hoomd.hpmc.external.External`` provides an abstract interface to external potentials
  (`#1811 <https://github.com/glotzerlab/hoomd-blue/pull/1811>`__).
* ``hoomd.hpmc.external.Linear`` computes the potential as a linear function of the distance from a
  point to a plane (`#1811 <https://github.com/glotzerlab/hoomd-blue/pull/1811>`__).
* ``HPMCIntegrator.external_potentials`` sets the list of external potentials applied to the system
  (`#1811 <https://github.com/glotzerlab/hoomd-blue/pull/1811>`__).
* ``hpmc.pair.ExpandedGaussian`` computes the expanded Gaussian pair potential in HPMC
  (`#1817 <https://github.com/glotzerlab/hoomd-blue/pull/1817>`__).

*Changed*

* Miscellaneous documentation improvements
  (`#1786 <https://github.com/glotzerlab/hoomd-blue/pull/1786>`__,
  `#1800 <https://github.com/glotzerlab/hoomd-blue/pull/1800>`__,
  `#1820 <https://github.com/glotzerlab/hoomd-blue/pull/1820>`__).
* Provide an error message for invalid Ellipsoid shape parameters
  (`#1785 <https://github.com/glotzerlab/hoomd-blue/pull/1785>`__).
* Provide the full CUDA error message when scanning devices
  (`#1803 <https://github.com/glotzerlab/hoomd-blue/pull/1803>`__).
* Test with gcc14, clang17, and clang18. No longer test with clang10, clang11, or clang12.
  (`#1798 <https://github.com/glotzerlab/hoomd-blue/pull/1798>`__,
  `#1816 <https://github.com/glotzerlab/hoomd-blue/pull/1816>`__).
* Ensure that Gaussian-type pair potentials have positive sigma values
  (`#1810 <https://github.com/glotzerlab/hoomd-blue/pull/1810>`__).
* Demonstrate ``Step`` and ``AngularStep`` in the tutorial "Modelling Patchy Particles".
* Fixed typographical errors in all tutorials.

*Fixed*

* Issue the proper error message when ``ALJ.shape`` is not set for all particle types
  (`#1808 <https://github.com/glotzerlab/hoomd-blue/pull/1808>`__).
* Correctly apply Brownian torque when elements of the inertia tensor are 0
  (`#1825 <https://github.com/glotzerlab/hoomd-blue/pull/1825>`__).


*Deprecated*

* ``HPMCIntegrator.external_potential`` - use ``HPMCIntegrator.external_potentials``
  (`#1811 <https://github.com/glotzerlab/hoomd-blue/pull/1811>`__).
* ``hoomd.hpmc.external.user.CPPExternalPotential`` - use ``hoomd.hpmc.external.Linear`` or write a
  custom component in C++ (`#1811 <https://github.com/glotzerlab/hoomd-blue/pull/1811>`__).

*Removed*

* Support for Python 3.8
  (`#1797 <https://github.com/glotzerlab/hoomd-blue/pull/1797>`__).

4.7.0 (2024-05-16)
^^^^^^^^^^^^^^^^^^

*Fixed*

* ``md.methods.rattle.Brownian`` executes without causing a segmentation fault on the CPU with domain
  decomposition (`#1748 <https://github.com/glotzerlab/hoomd-blue/pull/1748>`__).
* Compile ``BoxDim.h`` without warnings
  (`#1756 <https://github.com/glotzerlab/hoomd-blue/pull/1756>`__).
* Do not compute dipole-dipole interactions that are not necessary
  (`#1758 <https://github.com/glotzerlab/hoomd-blue/pull/1758>`__).
* Correctly define the units of gamma in ``md.methods.Langevin``
  (`#1771 <https://github.com/glotzerlab/hoomd-blue/pull/1771>`__).
* Fix compile errors with external components that use the Expanded Mie potential
  (`#1781 <https://github.com/glotzerlab/hoomd-blue/pull/1781>`__).
* Allow HPMC pair potentials to be subclassed in external components
  (`#1780 <https://github.com/glotzerlab/hoomd-blue/pull/1780>`__).

*Added*

* "How to tune move sizes in multicomponent HPMC systems" documentation page
  (`#1750 <https://github.com/glotzerlab/hoomd-blue/pull/1750>`__).
* ``hoomd.box.from_basis_vectors`` - construct a box from arbitrary basis vectors
  (`#1769 <https://github.com/glotzerlab/hoomd-blue/pull/1769>`__).

*Changed*

* Make readthedocs builds more reproducible
  (`#1758 <https://github.com/glotzerlab/hoomd-blue/pull/1758>`__).


4.6.0 (2024-03-19)
^^^^^^^^^^^^^^^^^^

*Fixed*

* ``create_state_from_gsd`` reads bond/angle/dihedral/improper/pair types when there are no
  corresponding groups (`#1729 <https://github.com/glotzerlab/hoomd-blue/pull/1729>`__).

*Added*

* ``hoomd.variant.box.BoxVariant`` - Describe boxes that change as a function of timestep
  (`#1685 <https://github.com/glotzerlab/hoomd-blue/pull/1685>`__).
* ``hoomd.variant.box.Constant`` - A constant box
  (`#1685 <https://github.com/glotzerlab/hoomd-blue/pull/1685>`__).
* ``hoomd.variant.box.Interpolate`` - Linearly interpolate between two boxes
  (`#1685 <https://github.com/glotzerlab/hoomd-blue/pull/1685>`__).
* ``hoomd.variant.box.InverseVolumeRamp`` - Linearly ramp the inverse volume of the system
  (`#1685 <https://github.com/glotzerlab/hoomd-blue/pull/1685>`__).
* ``hoomd.hpmc.update.QuickCompress`` now accepts a ``hoomd.variant.box.BoxVariant`` object for
  `target_box` (`#1736 <https://github.com/glotzerlab/hoomd-blue/pull/1736>`__).
* ``box`` argument to ``hoomd.update.BoxResize`` that accepts a ``hoomd.variant.box.BoxVariant``
  (`#1740 <https://github.com/glotzerlab/hoomd-blue/pull/1740>`__).
* ``hoomd.hpmc.pair.Union`` computes pair potentials between unions of points. Replaces
  ``CPPPotentialUnion`` (`#1725 <https://github.com/glotzerlab/hoomd-blue/pull/1725>`__).
* ``hoomd.hpmc.pair.Step`` - A step function potential
  (`#1732 <https://github.com/glotzerlab/hoomd-blue/pull/1732>`__).
* ``hoomd.hpmc.pair.AngularStep`` - Angular patches on particles with step function interactions
  (e.g. Kern-Frenkel) (`#1728 <https://github.com/glotzerlab/hoomd-blue/pull/1728>`__).

*Changed*

* Use ``FindPython`` on modern CMake installations. You may need to adjust build scripts
  in cases where the new behavior does not exactly match the old (i.e. use
  ``-DPython_EXECUTABLE`` in place of ``-DPYTHON_EXECUTABLE``)
  (`#1730 <https://github.com/glotzerlab/hoomd-blue/pull/1730>`__).
* External components must switch from ``pybind11_add_module`` to ``hoomd_add_module``
  (`#1730 <https://github.com/glotzerlab/hoomd-blue/pull/1730>`__).

*Deprecated*

* ``box1``, ``box2``, and ``variant`` arguments to ``hoomd.update.BoxResize``
  (`#1740 <https://github.com/glotzerlab/hoomd-blue/pull/1740>`__).

4.5.0 (2024-02-13)
^^^^^^^^^^^^^^^^^^

*Fixed*

* ``hoomd.hpmc.update.Shape`` properly restores shape alchemy parameters on rejected trial moves
  (`#1696 <https://github.com/glotzerlab/hoomd-blue/pull/1696>`__).
* ``hoomd.hpmc.update.Shape`` now functions with ``hoomd.device.GPU``
  (`#1696 <https://github.com/glotzerlab/hoomd-blue/pull/1696>`__).
* ``hoomd.hpmc.update.MuVT`` applies external potentials
  (`#1711 <https://github.com/glotzerlab/hoomd-blue/pull/1711>`__).
* ``hoomd.hpmc.update.QuickCompress`` can now reshape boxes with tilt factors <= 0
  (`#1709 <https://github.com/glotzerlab/hoomd-blue/pull/1709>`__).

*Added*

* Improve component build documentation and link to the ``hoomd-component-template`` repository
  (`#1668 <https://github.com/glotzerlab/hoomd-blue/pull/1668>`__).
* ``hoomd.md.improper.Periodic`` - CHARMM-like periodic improper potential
  (`#1662 <https://github.com/glotzerlab/hoomd-blue/pull/1662>`__).
* ``allow_unsafe_resize`` flag to ``hoomd.hpmc.update.QuickCompress``
  (`#1678 <https://github.com/glotzerlab/hoomd-blue/pull/1678>`__).
* ``hoomd.error.GPUNotAvailableError``
  (`#1694 <https://github.com/glotzerlab/hoomd-blue/pull/1694>`__).
* HPMC compile time pair potential framework (CPU only). Allows potential energy in HPMC simulations
  without ``CPPPotential``.

  * ``hoomd.hpmc.pair.LennardJones`` - Evaluate Lennard Jones energy between particles
    (`#1676 <https://github.com/glotzerlab/hoomd-blue/pull/1676>`__).
  * ``HPMCIntegrator.pair_potentials`` - Set a list of pair potentials to evaluate
    (`#1676 <https://github.com/glotzerlab/hoomd-blue/pull/1676>`__).
  * ``HPMCIntegrator.pair_energy`` (loggable) - Total pair energy from all pair potentials.
    (`#1676 <https://github.com/glotzerlab/hoomd-blue/pull/1676>`__).

*Deprecated*

* ``_InternalCustomUpdater.update``.
  (`#1692 <https://github.com/glotzerlab/hoomd-blue/pull/1692>`__).
* ``_InternalCustomTuner.tune``.
  (`#1692 <https://github.com/glotzerlab/hoomd-blue/pull/1692>`__).
* ``_InternalCustomWriter.write``.
  (`#1692 <https://github.com/glotzerlab/hoomd-blue/pull/1692>`__).
* ``HDF5Log.write``.
  (`#1692 <https://github.com/glotzerlab/hoomd-blue/pull/1692>`__).
* ``hoomd.util.GPUNotAvailableError``
  (`#1694 <https://github.com/glotzerlab/hoomd-blue/pull/1694>`__).
* ``hoomd.hpmc.pair.user.CPPPotentialBase``
  (`#1676 <https://github.com/glotzerlab/hoomd-blue/pull/1676>`__).
* ``hoomd.hpmc.pair.user.CPPPotential`` - Use a built-in potential or compile your code in a component
  (`#1676 <https://github.com/glotzerlab/hoomd-blue/pull/1676>`__).
* ``hoomd.hpmc.pair.user.CPPPotentialUnion`` - Use a built-in potential or compile your code in a component
  (`#1676 <https://github.com/glotzerlab/hoomd-blue/pull/1676>`__).
* ``HPMCIntegrator.pair_potential`` - Use compiled potentials with ``pair_potentials``
  (`#1676 <https://github.com/glotzerlab/hoomd-blue/pull/1676>`__).
* Single-process multi-gpu code path
  (`#1706 <https://github.com/glotzerlab/hoomd-blue/pull/1706>`__).

*Changed*

* Refactored the C++ API for ``PatchEnergy`` potentials
  (`#1676 <https://github.com/glotzerlab/hoomd-blue/pull/1676>`__).
* Removed unused ``Operation._children`` methods
  (`#1713 <https://github.com/glotzerlab/hoomd-blue/pull/1713>`__).

4.4.1 (2023-12-18)
^^^^^^^^^^^^^^^^^^

*Fixed*

* Correct ``net_virial`` values in local snapshots
  (`#1672 <https://github.com/glotzerlab/hoomd-blue/pull/1672>`__).
* Improve HPMC performance on the CPU when using a pair potential
  (`#1679 <https://github.com/glotzerlab/hoomd-blue/pull/1679>`__).
* Improve HPMC performance with 3D hard shapes
  (`#1679 <https://github.com/glotzerlab/hoomd-blue/pull/1679>`__).
* Improve HPMC performance on the CPU
  (`#1687 <https://github.com/glotzerlab/hoomd-blue/pull/1687>`__).

*Changed*

* Provide support via GitHub discussions
  (`#1671 <https://github.com/glotzerlab/hoomd-blue/pull/1671>`__).

4.4.0 (2023-12-04)
^^^^^^^^^^^^^^^^^^

*Added*

* ``hoomd.md.external.field.Magnetic`` computes forces and torques on particles from an external
  magnetic field (`#1637 <https://github.com/glotzerlab/hoomd-blue/pull/1637>`__).
* Tutorial on placing barriers
  (`hoomd-examples/#111 <https://github.com/glotzerlab/hoomd-examples/pull/111>`__).

*Fixed*

* Use ``mpirun`` specific local ranks to select GPUs before checking ``SLURM_LOCALID``
  (`#1647 <https://github.com/glotzerlab/hoomd-blue/pull/1647>`__).
* Fix typographical errors in ``RevCross`` documentation
  (`#1642 <https://github.com/glotzerlab/hoomd-blue/pull/1642>`__).
* Use standards compliant ``thrust::get``
  (`#1660 <https://github.com/glotzerlab/hoomd-blue/pull/1660>`__).

*Changed*

* Removed unused code
  (`#1646 <https://github.com/glotzerlab/hoomd-blue/pull/1646>`__).
* No longer issue a warning when ``hoomd.md.Integrator`` is used without an integration method
  (`#1659 <https://github.com/glotzerlab/hoomd-blue/pull/1659>`__).
* Increase performance of ``Force.forces``, ``Force.torques``, ``Force.energies``, and
  ``Force.virials`` (`#1654 <https://github.com/glotzerlab/hoomd-blue/pull/1654>`__).

*Deprecated*

* ``num_cpu_threads > 1``. Use ``num_cpu_threads = 1``
  (`#1656 <https://github.com/glotzerlab/hoomd-blue/pull/1656>`__).
* ``HPMCIntegrator.depletant_fugacity > 0``
  (`#1657 <https://github.com/glotzerlab/hoomd-blue/pull/1657>`__).

4.3.0 (2023-10-24)
^^^^^^^^^^^^^^^^^^

*Fixed*

* ``md.alchemy.methods.NVT`` now evolves the elements provided in ``alchemical_dof``
  (`#1633 <https://github.com/glotzerlab/hoomd-blue/pull/1633>`__).
* More consistent notice messages regarding MPI ranks used in GPU selection
  (`#1635 <https://github.com/glotzerlab/hoomd-blue/pull/1635>`__).
* ``hoomd.hpmc.compute.SDF`` computes correct pressures with patchy potentials.
  (`#1636 <https://github.com/glotzerlab/hoomd-blue/pull/1636>`__).

*Added*

* Support GCC 13
  (`#1634 <https://github.com/glotzerlab/hoomd-blue/pull/1634>`__).
* Support Python 3.12
  (`#1634 <https://github.com/glotzerlab/hoomd-blue/pull/1634>`__).
* ``tau`` parameter to ``hoomd.md.methods.thermostats.Bussi``
  (`#1619 <https://github.com/glotzerlab/hoomd-blue/pull/1619>`__).

*Changed*

* Revise class documentation.
  (`#1628 <https://github.com/glotzerlab/hoomd-blue/pull/1628>`__).
* Add more code snippets to the class documentation
  (`#1628 <https://github.com/glotzerlab/hoomd-blue/pull/1628>`__).

4.2.1 (2023-10-02)
^^^^^^^^^^^^^^^^^^

*Fixed*

* ``hoomd.write.Table`` correctly displays floating point values that are exactly 0.0
  (`#1625 <https://github.com/glotzerlab/hoomd-blue/issues/1625>`__).
* ``hoomd.write.HDF5Log`` defaults to ``"f8"`` formatting except when the value is an `int`,
  or a `numpy.number` (`#1620 <https://github.com/glotzerlab/hoomd-blue/issues/1620>`__).
* Attempt to workaround ``PMI_Init returned 1`` error on OLCF Frontier
  (`#1629 <https://github.com/glotzerlab/hoomd-blue/pull/1629>`__).
* Apple clang 15 compiles HOOMD-blue without errors
  (`#1626 <https://github.com/glotzerlab/hoomd-blue/pull/1626>`__).

4.2.0 (2023-09-20)
^^^^^^^^^^^^^^^^^^

*Fixed*

* Make ``HDF5Log`` example more visible
  (`#1602 <https://github.com/glotzerlab/hoomd-blue/pull/1602>`__).
* Access valid GPU memory in ``hoomd.hpmc.update.Clusters``
  (`#1607 <https://github.com/glotzerlab/hoomd-blue/pull/1607>`__).
* Test suite passes on the ROCm GPU platform
  (`#1607 <https://github.com/glotzerlab/hoomd-blue/pull/1607>`__).
* Provide an error message when using ``md.external.field.Periodic`` in 2D
  (`#1603 <https://github.com/glotzerlab/hoomd-blue/pull/1603>`__).
* ``hoomd.write.GSD`` reports "File exists" in the exception description when using the ``'xb'``
  mode and the file exists (`#1609 <https://github.com/glotzerlab/hoomd-blue/pull/1609>`__).
* Write small numbers correctly in ``hoomd.write.Table``
  (`#1617 <https://github.com/glotzerlab/hoomd-blue/pull/1617>`__).
* Make examples in ``hoomd.md.methods.NVE`` and ``hoomd.md.methods.DisplacementCapped`` more visible
  (`#1601 <https://github.com/glotzerlab/hoomd-blue/pull/1601>`__).

*Added*

* Documentation page: "How to apply arbitrary forces in MD"
  (`#1610 <https://github.com/glotzerlab/hoomd-blue/pull/1610>`__).
* Documentation page: "How to prevent particles from moving"
  (`#1611 <https://github.com/glotzerlab/hoomd-blue/pull/1611>`__).
* Documentation page: "How to minimize the potential energy of a system"
  (`#1614 <https://github.com/glotzerlab/hoomd-blue/pull/1614>`__).
* Documentation page: "How to continuously vary potential parameters"
  (`#1612 <https://github.com/glotzerlab/hoomd-blue/pull/1612>`__).
* Documentation page: "How to determine the most efficient device"
  (`#1616 <https://github.com/glotzerlab/hoomd-blue/pull/1616>`__).
* Documentation page: "How to choose the neighbor list buffer distance"
  (`#1615 <https://github.com/glotzerlab/hoomd-blue/pull/1615>`__).
* Documentation page: "How to compute the free energy of solids"
  (`#1613 <https://github.com/glotzerlab/hoomd-blue/pull/1613>`__).
* MPCD particle data is now available included in ``Snapshot``
  (`#1580 <https://github.com/glotzerlab/hoomd-blue/pull/1580>`__).
* Add variable parameters to ``hpmc.external.user.CPPExternalPotential``
  (`#1608 <https://github.com/glotzerlab/hoomd-blue/pull/1608>`__).

*Changed*

* Removed the unused ``ExternalFieldComposite.h`` and all the related ``ExternalFieldComposite*``
  (`#1604 <https://github.com/glotzerlab/hoomd-blue/pull/1604>`__).

4.1.0 (2023-08-07)
^^^^^^^^^^^^^^^^^^

*Fixed*

* Improved documentation
  (`#1585 <https://github.com/glotzerlab/hoomd-blue/pull/1585>`__).
* Update mesh documentation
  (`#1587 <https://github.com/glotzerlab/hoomd-blue/pull/1587>`__).
* Follow detailed balance in ``hoomd.hpmc.update.Shape``
  (`#1595 <https://github.com/glotzerlab/hoomd-blue/pull/1595>`__).
* ``pre-commit`` environment installs correctly on macos-arm64
  (`#1597 <https://github.com/glotzerlab/hoomd-blue/pull/1597>`__).
* Install all HPMC headers for use by plugins
  (`#1573 <https://github.com/glotzerlab/hoomd-blue/pull/1573>`__).
* Bond potentials can now be implemented via external plugins
  (`#1591 <https://github.com/glotzerlab/hoomd-blue/issues/1591>`__).

*Added*

* Tested example code snippets in select modules
  (`#1574 <https://github.com/glotzerlab/hoomd-blue/pull/1574>`__)
  (`#1586 <https://github.com/glotzerlab/hoomd-blue/pull/1586>`__).
* ``hoomd.util.make_example_simulation`` - create an example Simulation object
  (`#1574 <https://github.com/glotzerlab/hoomd-blue/pull/1574>`__)
  (`#1586 <https://github.com/glotzerlab/hoomd-blue/pull/1586>`__).
* ``hoomd.write.Burst`` now has a ``__len__`` method
  (`#1575 <https://github.com/glotzerlab/hoomd-blue/pull/1575>`__).
* Support clang 15 and 16 on Linux
  (`#1593 <https://github.com/glotzerlab/hoomd-blue/pull/1593>`__).
* ``hoomd.write.HDF5Logger`` - write log quantities to HDF5 files
  (`#1588 <https://github.com/glotzerlab/hoomd-blue/pull/1588>`__).
* ``default_gamma`` and ``default_gamma_r`` arguments to ``hoomd.md.methods.rattle.Brownian``
  ``hoomd.md.methods.rattle.Langevin``, and ``hoomd.md.methods.rattle.OverdampedViscous``
  (`#1589 <https://github.com/glotzerlab/hoomd-blue/issues/1589>`__).

4.0.1 (2023-06-27)
^^^^^^^^^^^^^^^^^^

*Fixed*

* Prevent ``ValueError: signal only works in main thread of the main interpreter`` when importing
  hoomd in a non-main thread
  (`#1576 <https://github.com/glotzerlab/hoomd-blue/pull/1576>`__).
* The recommended conda install commands find the documented version
  (`#1578 <https://github.com/glotzerlab/hoomd-blue/pull/1578>`__).
* CMake completes without error when ``HOOMD_GPU_PLATFORM=HIP``
  (`#1579 <https://github.com/glotzerlab/hoomd-blue/pull/1579>`__).
* Tests pass with GSD 3.0.0 installed
  (`#1577 <https://github.com/glotzerlab/hoomd-blue/pull/1577>`__).
* Provide full CUDA error message when possible
  (`#1581 <https://github.com/glotzerlab/hoomd-blue/pull/1581>`__).
* Notice level 4 gives additional GPU initialization details
  (`#1581 <https://github.com/glotzerlab/hoomd-blue/pull/1581>`__).
* Show particle out of bounds error messages in exception description
  (`#1581 <https://github.com/glotzerlab/hoomd-blue/pull/1581>`__).

*Changed*

* Package source in ``hoomd-x.y.z.tar.gz`` (previously ``hoomd-vx.y.z.tar.gz``)
  (`#1572 <https://github.com/glotzerlab/hoomd-blue/pull/1572>`__).

4.0.0 (2023-06-06)
^^^^^^^^^^^^^^^^^^

*Fixed*

* Fix error with ``MPI_Allreduce`` on OLCF Frontier
  (`#1547 <https://github.com/glotzerlab/hoomd-blue/pull/1547>`__).
* Correct equations in virial pressure documentation
  (`#1548 <https://github.com/glotzerlab/hoomd-blue/pull/1548>`__).
* Work around cases where Python's garbage collector fails to collect ``Operation`` objects
  (`#1457 <https://github.com/glotzerlab/hoomd-blue/issues/1457>`__).
* Incorrect behavior with ``hpmc.external.user.CPPExternalPotential`` in MPI domain decomposition
  simulations (`#1562 <https://github.com/glotzerlab/hoomd-blue/issues/1562>`__).

*Added*

* ``hoomd.md.ConstantVolume`` integration method
  (`#1419 <https://github.com/glotzerlab/hoomd-blue/issues/1419>`__).
* ``hoomd.md.ConstantPressure`` integration method, implementing the Langevin piston barostat
  (`#1419 <https://github.com/glotzerlab/hoomd-blue/issues/1419>`__).
* Thermostats in ``hoomd.md.methods.thermostats`` that work with ``ConstantVolume`` and
  ``ConstantPressure``, including the new Bussi-Donadio-Parrinello thermostat
  (`#1419 <https://github.com/glotzerlab/hoomd-blue/issues/1419>`__).
* ``hoomd.md.external.wall.Gaussian``
  (`#1499 <https://github.com/glotzerlab/hoomd-blue/pull/1499>`__).
* ``hoomd.write.GSD.maximum_write_buffer_size`` - Set the maximum size of the GSD write buffer
  (`#1541 <https://github.com/glotzerlab/hoomd-blue/pull/1541>`__).
* ``hoomd.write.GSD.flush`` - flush the write buffer of an open GSD file
  (`#1541 <https://github.com/glotzerlab/hoomd-blue/pull/1541>`__).
* On importing ``hoomd``, install a ``SIGTERM`` handler that calls ``sys.exit(1)``
  (`#1541 <https://github.com/glotzerlab/hoomd-blue/pull/1541>`__).
* More descriptive error messages when calling ``Simulation.run``
  (`#1552 <https://github.com/glotzerlab/hoomd-blue/pull/1552>`__).
* `hoomd.Snapshot.from_gsd_frame` - convert a `gsd.hoomd.Frame` object to `hoomd.Snapshot`
  (`#1559 <https://github.com/glotzerlab/hoomd-blue/pull/1559>`__).
* `hoomd.device.NoticeFile` - a file-like object that writes to `hoomd.device.Device.notice`
  (`#1449 <https://github.com/glotzerlab/hoomd-blue/issues/1449>`__).
* `hoomd.write.Burst` - selective high-frequency frame writing to GSD files
  (`#1543 <https://github.com/glotzerlab/hoomd-blue/pull/1543>`__).
* Support LLVM 16
  (`#1568 <https://github.com/glotzerlab/hoomd-blue/pull/1568>`__).
* More detailed status message for found CUDA libraries
  (`#1566 <https://github.com/glotzerlab/hoomd-blue/pull/1566>`__).

*Changed*

* ``hoomd.md.constrain.Rigid`` no longer takes ``diameters`` or ``charges`` as keys in the ``body``
  parameters. ``create_bodies`` method now takes an optional ``charges`` argument to set charges
  (`#1350 <https://github.com/glotzerlab/hoomd-blue/issues/1350>`__).
* Control the precision with the CMake options ``HOOMD_LONGREAL_SIZE`` (default: 64) and
  ``HOOMD_SHORTREAL_SIZE`` (default: 32)
  (`#355 <https://github.com/glotzerlab/hoomd-blue/issues/355>`__).
* [developers] ``ShortReal`` and ``LongReal`` types enable mixed precision implementations
  (`#355 <https://github.com/glotzerlab/hoomd-blue/issues/355>`__).
* ``hoomd.md.constrain.Rigid`` now updates constituent particle types each step
  (`#1440 <https://github.com/glotzerlab/hoomd-blue/pull/1440>`__).
* Moved ``hoomd.mesh.Mesh.triangles`` to ``hoomd.mesh.Mesh.triangulation``
  (`#1464 <https://github.com/glotzerlab/hoomd-blue/pull/1464>`__).
* ``hoomd.write.GSD`` does not write ``particles/diameter`` by default
  (`#1266 <https://github.com/glotzerlab/hoomd-blue/issues/1266>`__).
* Updated tutorials to use HOOMD-blue v4 API, work with up to date releases of freud, gsd, and
  signac. Also make general improvements to the tutorials.
* Document changes needed to migrate from v3 to v4 in the migration guide.
* More descriptive error messages when calling ``Simulation.run``
  (`#1552 <https://github.com/glotzerlab/hoomd-blue/pull/1552>`__).
* Increase performance of ``hoomd.write.GSD``
  (`#1538 <https://github.com/glotzerlab/hoomd-blue/pull/1538>`__).
* Increase performance of ``hoomd.State.get_snapshot`` in serial
  (`#1538 <https://github.com/glotzerlab/hoomd-blue/pull/1538>`__).
* `hoomd.write.GSD.dynamic` now allows fine grained control over individual particle fields
  (`#1538 <https://github.com/glotzerlab/hoomd-blue/pull/1538>`__).
* No longer test with GCC 7-8, Python 3.6-3.7, or Clang 6-9)
  (`#1544 <https://github.com/glotzerlab/hoomd-blue/pull/1544>`__).
* Improved error messages with NVRTC compiled code
  (`#1567 <https://github.com/glotzerlab/hoomd-blue/pull/1567>`__).

*Deprecated*

* ``Scalar``, ``Scalar2``, ``Scalar3``, and ``Scalar4`` data types. Use ``LongReal[N]`` instead in
  new code
  (`#355 <https://github.com/glotzerlab/hoomd-blue/issues/355>`__).
* ``hoomd.Snapshot.from_gsd_snapshot`` - use `hoomd.Snapshot.from_gsd_frame`
  (`#1559 <https://github.com/glotzerlab/hoomd-blue/pull/1559>`__).

*Removed*

* ``fix_cudart_rpath`` CMake macro
  (`#1383 <https://github.com/glotzerlab/hoomd-blue/issues/1383>`__).
* ``ENABLE_MPI_CUDA`` CMake option
  (`#1401 <https://github.com/glotzerlab/hoomd-blue/issues/1401>`__).
* ``Berendsen``, ``NPH``, ``NPT``, ``NVE``, ``NVT`` MD integration methods
  (`#1419 <https://github.com/glotzerlab/hoomd-blue/issues/1419>`__).
* ``hoomd.write.GSD.log``
  (`#1480 <https://github.com/glotzerlab/hoomd-blue/issues/1480>`__).
* CMake option and compiler definition ``SINGLE_PRECISION``
  (`#355 <https://github.com/glotzerlab/hoomd-blue/issues/355>`__).
* ``charges`` key in ``hoomd.md.constrain.Rigid.body``
  (`#1496 <https://github.com/glotzerlab/hoomd-blue/issues/1496>`__).
* ``diameter`` key in ``hoomd.md.constrain.Rigid.body``.
  (`#1496 <https://github.com/glotzerlab/hoomd-blue/issues/1496>`__).
* ``hoomd.md.dihedral.Harmonic``.
  (`#1496 <https://github.com/glotzerlab/hoomd-blue/issues/1496>`__).
* ``hoomd.device.GPU.memory_traceback parameter``.
  (`#1496 <https://github.com/glotzerlab/hoomd-blue/issues/1496>`__).
* ``hoomd.md.pair.aniso.Dipole.mode`` parameter.
  (`#1496 <https://github.com/glotzerlab/hoomd-blue/issues/1496>`__).
* ``hoomd.md.pair.aniso.ALJ.mode`` parameter
  (`#1496 <https://github.com/glotzerlab/hoomd-blue/issues/1496>`__).
* ``hoomd.md.pair.Gauss``
  (`#1499 <https://github.com/glotzerlab/hoomd-blue/issues/1499>`__).
* ``hoomd.md.external.wall.Gauss``
  (`#1499 <https://github.com/glotzerlab/hoomd-blue/issues/1499>`__).
* ``msg_file`` property and argument in ``hoomd.device.Device``.
  (`#1499 <https://github.com/glotzerlab/hoomd-blue/issues/1499>`__).
* The ``sdf`` attribute of ``hoomd.hpmc.compute.SDF`` - use ``sdf_compression``
  (`#1523 <https://github.com/glotzerlab/hoomd-blue/pull/1523>`__).
* ``alpha`` parameter and attribute in ``Langevin``, ``BD``, and ``OverdampedViscous`` integration
  methods (`#1266 <https://github.com/glotzerlab/hoomd-blue/issues/1266>`__).
* ``needsDiameter`` and ``setDiameter`` API in C++ potential evaluators
  (`#1266 <https://github.com/glotzerlab/hoomd-blue/issues/1266>`__).

v3.x
----

v3.11.0 (2023-04-14)
^^^^^^^^^^^^^^^^^^^^

Added:

* ``hoomd.md.Integrator.validate_groups`` verifies that MD integration methods are applied to
  distinct subsets of the system and that those subsets consist of integrable particles
  (automatically called when attached)
  (`#1466 <https://github.com/glotzerlab/hoomd-blue/issues/1466>`__).

Changed:

* ``hoomd.hpmc.compute.SDF`` computes pressures for systems of concave and non-monotonic patch
  interactions (`#1391 <https://github.com/glotzerlab/hoomd-blue/pull/1391>`__).
* Reorganize documentation contents to fit in the sidebar, including landing pages for tutorials and
  how-to guides (`#1526 <https://github.com/glotzerlab/hoomd-blue/pull/1526>`_).

Fixed:

* Improved readability of images in the documentation
  (`#1521 <https://github.com/glotzerlab/hoomd-blue/issues/1521>`__).
* ``hoomd.write.Table`` now raises a meaningful error when given incorrect logger categories
  (`#1510 <https://github.com/glotzerlab/hoomd-blue/issues/1510>`__).
* Correctly document the 1/2 scaling factor in the pairwise virial computation
  (`#1525 <https://github.com/glotzerlab/hoomd-blue/pull/1525>`_).
* ``thermalize_particle_momenta`` now sets 0 velocity and angular momentum for rigid constituent
  particles  (`#1472 <https://github.com/glotzerlab/hoomd-blue/issues/1472>`__).
* Reduce likelihood of data corruption when writing GSD files
  (`#1531 <https://github.com/glotzerlab/hoomd-blue/pull/1531>`__).
* Clarify migration process for ``hoomd.md.pair.ExpandedLJ``
  (`#1501 <https://github.com/glotzerlab/hoomd-blue/pull/1501>`__).

Deprecated:

* The ``sdf`` attribute of ``hoomd.hpmc.compute.SDF`` - use ``sdf_compression``
  (`#1391 <https://github.com/glotzerlab/hoomd-blue/pull/1391>`__).

v3.10.0 (2023-03-14)
^^^^^^^^^^^^^^^^^^^^

Added:

* The ``message_filename`` property and argument to ``Device``, ``CPU``, and ``GPU`` to replace
  ``msg_file`` (`#1497 <https://github.com/glotzerlab/hoomd-blue/pull/1497>`_).
* ``hoomd.md.pair.Gaussian`` to replace ``hoomd.md.pair.Gauss``
  (`#1497 <https://github.com/glotzerlab/hoomd-blue/pull/1497>`_).
* ``hoomd.md.pair.ExpandedGaussian`` - the expanded Gaussian pair force
  (`#1493 <https://github.com/glotzerlab/hoomd-blue/pull/1493>`_).
* Guide: How to apply arbitrary pair potentials in HPMC
  (`#1505 <https://github.com/glotzerlab/hoomd-blue/issues/1505>`_).

Changed:

* Use ``furo`` style for HTML documentation
  (`#1498 <https://github.com/glotzerlab/hoomd-blue/pull/1498>`_).

Fixed:

* The ``hoomd.md.pair`` potentials ``ExpandedLJ``, ``ExpandedMie``, ``LJGauss``, and ``TWF`` now
  shift ``V(r_cut)`` to 0 properly when ``mode == 'shift'``
  (`#1504 <https://github.com/glotzerlab/hoomd-blue/issues/1504>`_).
* Corrected errors in the pair potential documentation
  (`#1504 <https://github.com/glotzerlab/hoomd-blue/issues/1504>`_).
* Note that the ``'body'`` exclusion should be used with ``hoomd.md.constrain.Rigid``
  (`#1465 <https://github.com/glotzerlab/hoomd-blue/issues/1465>`_).
* Correctly identify the ``'xyz'`` mode in ``hoomd.md.methods.NPH``
  (`#1509 <https://github.com/glotzerlab/hoomd-blue/pull/1509>`_).

Deprecated:

* The ``msg_file`` property and argument to ``Device``, ``CPU``, and ``GPU``.
* ``hoomd.md.pair.Gauss``.

v3.9.0 (2023-02-15)
^^^^^^^^^^^^^^^^^^^

Added:

* GPU code path for ``hoomd.update.BoxResize``
  (`#1462 <https://github.com/glotzerlab/hoomd-blue/pull/1462>`_).
* ``logger`` keyword argument and property to ``hoomd.write.GSD``
  (`#1481 <https://github.com/glotzerlab/hoomd-blue/pull/1481>`_).


Changed:

* Issue `FutureWarning` warnings when using deprecated APIs
  (`#1485 <https://github.com/glotzerlab/hoomd-blue/pull/1485>`_).
* Reformat the list of deprecated features.
  (`#1490 <https://github.com/glotzerlab/hoomd-blue/pull/1490>`_).
* In simulations with rigid bodies, remove D degrees of freedom when the system is momentum
  conserving
  (`#1467 <https://github.com/glotzerlab/hoomd-blue/issues/1467>`_).

Fixed:

* Compile without errors using ``hipcc`` and ROCM 5.1.0
  (`#1478 <https://github.com/glotzerlab/hoomd-blue/pull/1478>`_).
* Document that ``hoomd.md.force.Force`` can be added to ``Operations.computes``
  (`#1489 <https://github.com/glotzerlab/hoomd-blue/pull/1489>`_).
* ``hoomd.md.constrain.Rigid.create_bodies`` completes without segmentation faults when particle
  body tags are not -1
  (`#1476 <https://github.com/glotzerlab/hoomd-blue/issues/1476>`_).
* ``hoomd.hpmc.compute.FreeVolume`` computes the free area correctly in 2D simulations
  (`#1473 <https://github.com/glotzerlab/hoomd-blue/issues/1473>`_).

Deprecated:

* Deprecate ``write.GSD`` ``log`` keyword argument and property in favor of ``logger``
  (`#1481 <https://github.com/glotzerlab/hoomd-blue/pull/1481>`_).

v3.8.1 (2023-01-27)
^^^^^^^^^^^^^^^^^^^

Fixed:

* `#1468 <https://github.com/glotzerlab/hoomd-blue/issues/1468>`_: Conserve linear momentum in
  simulations using ``hoomd.md.constrain.Rigid`` on more than 1 MPI rank.

v3.8.0 (2023-01-12)
^^^^^^^^^^^^^^^^^^^

*Added*

* Support Python 3.11.
* Support CUDA 11.8.
* Support CUDA 12.0.0 final.

*Fixed*

* Improve numerical stability of orientation quaternions when using
  ``hoomd.md.update.ActiveRotationalDiffusion``
* Reduced memory usage and fix spurious failures in ``test_nlist.py``.
* Avoid triggering ``TypeError("expected x and y to have same length")`` in
  ``hoomd.hpmc.compute.SDF.betaP``.

*Deprecated*

* The following integration methods are deprecated. Starting in v4.0.0, the same functionalities
  will be available via ``hoomd.md.methods.ConstantVolume``/ ``hoomd.md.methods.ConstantPressure``
  with an appropriately chosen ``thermostat`` argument.

  * ``hoomd.md.methods.NVE``
  * ``hoomd.md.methods.NVT``
  * ``hoomd.md.methods.Berendsen``
  * ``hoomd.md.methods.NPH``
  * ``hoomd.md.methods.NPT``

*Removed*

* Support for CUDA 10.

v3.7.0 (2022-11-29)
^^^^^^^^^^^^^^^^^^^

*Added*

* ``Neighborlist.r_cut`` sets the base cutoff radius for neighbor search - for use when the neighbor
  list is used for analysis or custom Python code.
* ``Neighborlist.cpu_local_nlist_arrays`` provides zero-copy access to the computed neighbor list.
* ``Neighborlist.gpu_local_nlist_arrays`` provides zero-copy access to the computed neighbor list.
* ``Neighborlist.local_pair_list`` provides the rank local pair list by index.
* ``Neighborlist.pair_list`` provides the global pair list by tag on rank 0.
* ``hoomd.md.dihedral.Periodic`` - a new name for the previous ``Harmonic`` potential.
* ``default_gamma`` and ``default_gamma_r`` arguments to the ``hoomd.md.methods``: ``Brownian``,
  ``Langevin``, and ``OverdampedViscous``.
* ``reservoir_energy`` loggable in ``hoomd.md.methods.Langevin``.
* ``hoomd.md.force.Constant`` applies constant forces and torques to particles.

*Changed*

* [plugin developers] Refactored the ``LocalDataAccess`` C++ classes to add flexibility.

*Fixed*

* ``hoomd.hpmc.nec`` integrators compute non-infinite virial pressures for 2D simulations.
* Raise an exception when attempting to get the shape specification of shapes with 0 elements.
* Box conversion error message now names ``hoomd.Box``.

*Deprecated*

* ``hoomd.md.dihedral.Harmonic`` - use the functionally equivalent ``hoomd.md.dihedral.Periodic``.
* ``charges`` key in ``hoomd.md.constrain.Rigid.body``.
* ``diameters`` key in ``hoomd.md.constrain.Rigid.body``.

v3.6.0 (2022-10-25)
^^^^^^^^^^^^^^^^^^^

*Changed*

* In ``hoomd.md.pair.aniso.ALJ``, ``shape.rounding_radii`` now defaults to (0.0, 0.0, 0.0).
* Revise ``hoomd.md.pair.aniso.ALJ`` documentation.
* ``hoomd.md.force.Force`` instances can now be added to the ``Operations`` list allowing users to
  compute force, torque, energy, and virials of forces that are not included in the dynamics of
  the system.
* [developers]: Removed internal methods ``_remove`` and ``_add`` from the data model.

*Fixed*

* Increase the performance of ``md.pair.Table`` on the CPU.
* Improve accuracy of ``hoomd.hpmc.update.BoxMC`` when used with patch potentials.
* Provide an accurate warning message when creating the state with many bond/angle/... types.
* Add missing documentation for ``hoomd.md.methods.Berendsen``.
* CVE-2007-4559

v3.5.0 (2022-09-14)
^^^^^^^^^^^^^^^^^^^

*Added*

* Example plugin that demonstrates how to add a MD pair potential.
* Support a large number of particle and bond types (subject to available GPU memory and user
  patience) for the ``Cell`` neighbor list, MD pair potentials, MD bond potentials, Brownian, and
  Langevin integration methods.

*Changed*

* Raise an error when initializing with duplicate types.
* ``hpmc.compute.SDF`` now computes pressures of systems with patch interactions.
* Raise descriptive error messages when the shared memory request exceeds that available on the GPU.

*Fixed*

* Include all ``Neighborlist`` attributes in the documentation.
* Memory allocation errors in C++ now result in ``MemoryError`` exceptions in Python.
* Add missing ``Autotuned.h`` header file.
* External components build correctly when ``ENABLE_MPI=on`` or ``ENABLE_GPU=on``.
* Type parameter validation when items contain ``numpy.ndarray`` elements.
* Compile with CUDA 12.0.

*Deprecated*

* ``Device.memory_traceback`` attribute. This attribute has no effect.


v3.4.0 (2022-08-15)
^^^^^^^^^^^^^^^^^^^

*Added*

* The new HOOMD-blue logo is now available in the documentation.
* ``hoomd.md.methods.DisplacementCapped`` class for relaxing configurations with overlaps.
* ``hoomd.md.methods.rattle.DisplacementCapped`` class for relaxing configurations with overlaps.
* ``hoomd.device.Device.notice`` - print user-defined messages to the configured message output
  stream.
* Tutorial: Modelling Rigid Bodies.
* ``AutotunedObject`` class that provides an interface to read and write tuned kernel parameters,
  query whether tuning is complete, and start tuning again at the object level.
* ``is_tuning_complete`` method to ``Operations``. Check whether kernel parameter tuning is complete
  for all operations.
* ``tune_kernel_parameters`` methods to ``Operations`` and many other classes. Start tuning kernel
  parameters in all operations.
* ``hoomd.md.HalfStepHook`` - extensible hook class called between step 1 and 2 of MD integration.
* ``hoomd.md.Integrator.half_step_hook`` - property to get/set the half step hook.


*Fixed*

* Active forces on manifolds now attach to the ``Simulation`` correctly.
* ``hoomd.update.FilterUpdater`` now accepts ``hoomd.filter.CustomFilter`` subclasses.
* Correct error message is given when a sequence like parameter is not given to a type parameter.
* Fix non-axis-aligned Cylinder walls in MD.
* ``hoomd.md.constrain.Constraint`` now has ``hoomd.md.force.Force`` as a base class.
* Provide a warning instead of an error when passing an out of range seed to the ``Simulation``
  constructor.
* Compile with current versions of HIP and ROCm.
* Compilation errors with CUDA >=11.8.

v3.3.0 (2022-07-08)
^^^^^^^^^^^^^^^^^^^

*Added*

* A decorator that modifies the namespace of operation and custom action classes
  ``hoomd.logging.modify_namespace``.
* Tuner for the neighbor list buffer size ``hoomd.md.tune.NeighborListBuffer``.
* Solver infrastructure for optimization problems.
* ``Simulation.initial_timestep``: the timestep on which the last call to ``run`` started.
* ``variant_like``, ``trigger_like``, and ``filter_like`` typing objects for documentation.

*Changed*

* Removed ``"__main__"`` from some user custom action logging namespaces.

*Fixed*

* Improve documentation.
* Non-default loggables can now be explicitly specified with ``Logger.add``.
* Iteration of ``Logger`` instances.
* The logging category of ``hoomd.md.Integrate.linear_momentum``

v3.2.0 (2022-05-18)
^^^^^^^^^^^^^^^^^^^

*Added*

* ``hoomd.md.nlist.Neighborlist.num_builds`` property - The number of neighbor list builds since the
  last call to ``Simulation.run``.
* ``hoomd.md.nlist.Cell.dimensions`` property - The dimensions of the cell list.
* ``hoomd.md.nlist.Cell.allocated_particles_per_cell`` property -  The number of particle slots
  allocated per cell.
* ``hoomd.mesh.Mesh`` - Triangular mesh data structure.
* ``hoomd.md.mesh.bond`` - Bond potentials on mesh edges.
* Support gcc 12.
* Support clang 14.
* Set ``ENABLE_LLVM=on`` in conda binary builds.

*Fixed*

* Clarify documentation.
* ``Box.dimension`` reports the correct  value when reading in 2D boxes from GSD files generated in
  HOOMD v2.
* Improve performance of run time compiled HPMC potentials on the CPU.
* Pressing Ctrl-C or interrupting the kernel in Jupyter stops the run at the end of the current
  timestep.

v3.1.0 (2022-04-27)
^^^^^^^^^^^^^^^^^^^

*Added*

* Support LLVM 13 when ``ENABLE_LLVM=on``.
* ``hoomd.md.pair.LJGauss`` - Lennard-Jones-Gaussian pair potential.
* ``hoomd.md.alchemy.methods.NVT`` - Alchemical molecular dynamics integration method.
* ``hoomd.md.alchemy.pair.LJGauss`` - Lennard-Jones-Gaussian pair potential with alchemical degrees
  of freedom.
* ``hoomd.hpmc.update.Shape`` - Alchemical hard particle Monte Carlo through shape change moves.
* ``hoomd.hpmc.shape_move.Elastic`` - Shape move with elastic potential energy penalty.
* ``hoomd.hpmc.shape_move.ShapeSpace`` - Moves in a user defined shape space.
* ``hoomd.hpmc.shape_move.Vertex`` - Translate shape vertices.

*Changed*

* HPMC fugacity is now a per-type quantity.
* Improved documentation.
* [developers] Reduced the time needed for incremental builds.
* [developers] Reduced memory needed to compile HOOMD.

*Fixed*

* ALJ unit test passes in Debug builds.
* Add quotes to conda-forge gpu package installation example.
* ``hoomd.md.force.Custom`` zeroes forces, torques, energies, and virials before calling
  ``set_forces``.
* Point tarball download link to https://github.com/glotzerlab/hoomd-blue/releases.

*Deprecated*

* ``hoomd.md.pair.aniso.ALJ.mode`` - parameter has no effect.
* ``hoomd.md.pair.aniso.Dipole.mode`` - parameter has no effect.

v3.0.1 (2022-04-08)
^^^^^^^^^^^^^^^^^^^

*Fixed*

* Display status of ``trunk-patch`` branch in the GitHub actions badge.
* Add ``EvaluatorPairTable.h`` to installation directory.
* Add ``hoomd.filter.Rigid`` to the documentation.
* Prevent ``TypeError: 'bool' object is not iterable`` errors when comparing ``Tag`` filters with
  different lengths arrays.
* ``Simulation.tps`` and ``Simulation.walltime`` update every step of the run.

v3.0.0 (2022-03-22)
^^^^^^^^^^^^^^^^^^^

*Overview*

HOOMD-blue v3.0.0 is the first production release with the new API that has been developed and
implemented over more than 2 years. Those still using v2.x will need to make changes to their
scripts to use v3. See the `migrating` page for an overview and individual class and method
documentation for more information. To summarize, the new API is object oriented, allows HOOMD-blue
to work effectively as a Python package, and provides more hooks for Python code to directly
interface with the simulation.

*New features in v3 since v2.9.7:*

* Zero-copy data access through numpy and cupy.
* Triggers determine what timesteps operations execute on.
* User-defined operations, triggers, particle filters, variants, and forces.
* Logging subsystem supports array quantities.
* Implicit depletants for 2D shapes in HPMC.
* Harmonically mapped averaging for MD thermodynamic quantities of crystals.
* TWF and OPP pair potentials.
* Tether bond potential.
* Manifold constraints for MD integration methods (using RATTLE) and active forces.
* Document code architecture in ``ARCHITECTURE.md``.
* Overdamped viscous MD integration method.
* User-defined pair potentials work with HPMC on the GPU.
* Long range tail correction for Lennard-Jones potential.
* Anisotropic Lennard-Jones-like pair potential for polyhedra and ellipsoids.
* Newtownian event chain Monte Carlo for spheres and convex polyhedra.

See the full change log below for all v3 beta releases.

Changes from v3.0.0-beta.14:

*Added*

* ``hoomd.hpmc.tune.BoxMCMoveSize`` - Tune ``BoxMC`` move sizes to meet target acceptance ratios.
* ``hoomd.hpmc.nec.integrate.Sphere`` - Newtonian event chain Monte Carlo for hard spheres.
* ``hoomd.hpmc.nec.integrate.ConvexPolyhedron`` - Newtonian event chain Monte Carlo for hard convex
  polyhedra.
* ``hoomd.hpmc.nec.tune.ChainTime`` - Tune chain times in newtonian event chain Monte Carlo method.

*Changed*

* Improve documentation.
* [breaking] Renamed the ``hoomd.md.bond.Table`` energy parameter from ``V`` to ``U``.
* [breaking] Renamed the ``hoomd.md.pair.Table`` energy parameter from ``V`` to ``U``.
* [breaking] Renamed the ``hoomd.md.angle.Table`` energy parameter from ``V`` to ``U``.
* [breaking] Renamed the ``hoomd.md.dihedral.Table`` energy parameter from ``V`` to ``U``.
* [breaking] Renamed ``hoomd.md.nlist.Nlist`` to ``hoomd.md.nlist.NeighborList``.
* [developer] ``Updater`` and ``Analyzer`` in C++ have a ``m_trigger`` member now.
* [developer] ``_TriggeredOperation`` has been moved to ``TriggeredOperation`` and custom trigger
  setting and getting logic removed.

*Fixed*

* ``FIRE.converged`` may be queried before calling ``Simulation.run``.
* Bug where using ``__iadd__`` to certain attributes would fail with an exception.
* Bug where ``hoomd.md.pair.LJ.additional_energy`` is ``NaN`` when ``tail_correction`` is enabled
  and some pairs have ``r_cut=0``.
* Compile error with CUDA 11.7.
* Compile errors on native ubuntu 20.04 systems.
* Compile errors with ``ENABLE_GPU=on`` and ``clang`` as a host compiler.

*Removed*

* [developers] Removed ``IntegratorData`` class. It is replaced by structs that are defined in the
  integrator classes.
* ``get_ordered_vertices`` from ``hoomd.md.pair.aniso.ALJ``.
* Removed optional coxeter dependency.
* The ``limit`` parameter from ``hoomd.md.methods.NVE``.
* The ``limit`` parameter from ``hoomd.md.methods.rattle.NVE``.
* The ``diameter_shift`` parameter from ``hoomd.md.nlist.NeighborList``.
* The ``max_diameter`` parameter from ``hoomd.md.nlist.NeighborList``.

v3.0.0-beta.14 (2022-02-18)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

*Added*

* ``hoomd.hpmc.external.field.Harmonic`` - harmonic potential of particles to specific sites in
  the simulation box and orientations.
* Support ``cereal`` 1.3.1
* Guide on how to model molecular systems.
* ``version.floating_point_precision`` - Floating point width in bits for the particle
  properties and local calculations.
* ``hoomd.md.pair.LJ.tail_correction`` - Option to enable the isotropic integrated long range tail
  correction.
* ``hoomd.md.Integrator.linear_momentum`` - Compute the total system linear momentum. Loggable.
* ``hoomd.md.bond.Table`` - Tabulated bond potential.
* ``hoomd.md.angle.Table`` - Tabulated angle potential.
* ``hoomd.md.dihedral.Table`` - Tabulated dihedral potential.
* ``hoomd.md.improper.Harmonic`` - Compute the harmonic improper potential and forces.
* Tutorial on Organizing and executing simulations.
* C++ and build system overview in ``ARCHITECTURE.md``.
* ``hoomd.hpmc.external.wall`` - Overlap checks between particles and wall surfaces.
* ``hoomd.md.pair.ansio.ALJ`` - an anisotropic Lennard-Jones-like pair potential for polyhedra and
  ellipsoids.
* New optional dependency: ``coxeter``, needed for some ``ALJ`` methods.

*Changed*

* Support variant translational and rotational spring constants in
  ``hoomd.hpmc.external.field.Harmonic``.
* [breaking] Renamed ``hoomd.md.angle.Cosinesq`` to ``hoomd.md.angle.CosineSquared``.
* [breaking] ``hoomd.Box`` no longer has a ``matrix`` property use ``to_matrix`` and
  ``from_matrix``.

*Fixed*

* Compilation errors on FreeBSD.
* ``TypeError`` when instantiating special pair forces.
* Inconsistent state when using the ``walls`` setter of a ``hoomd.md.external.wall.WallPotential``.

*Removed*

* [breaking] Removed ``hoomd.md.pair.SLJ`` potential and wall. Use ``hoomd.md.pair.ExpandedLJ``.
* [breaking] ``hoomd.Box.lattice_vectors`` property no longer exists.

v3.0.0-beta.13 (2022-01-18)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

*Added*

* ``md.pair.ExpandedLJ`` - A Lennard-Jones potential where ``r`` is replaced with ``r-delta``.
* Support nested modification of operation parameters.
* ``wall`` - Define wall surfaces in the simulation box.
* ``md.external.wall`` - Pair interactions between particles and wall surfaces.
* ``Communicator.walltime`` - the wall clock time since creating the ``Communicator``.
* ``md.force.Custom`` - user defined forces in Python.

*Changed*

* Call ``update_group_dof`` implicitly in ``set_snapshot``, when changing integrators or integration
  methods, and on steps where ``FilterUpdater`` acts on the system.
* [breaking] ``update_group_dof`` defers counting the degrees of freedom until the next timestep or
  the next call to ``Simulation.run``.
* [breaking] Renamed ``md.bond.FENE`` to ``md.bond.FENEWCA``.
* ``md.bond.FENEWCA`` takes a user provided ``delta`` parameter and ignores the particle diameters.
* [breaking] ``md.pair.DLVO`` takes user provided ``a1`` and ``a2`` parameters and ignores the
  particle diameters.
* Removed invalid linker options when using gcc on Apple systems.
* Removed the ``r_on`` attribute and ``default_r_on`` constructor argument from pair potentials that
  do not use it.
* Building from source requires a C++17 compatible compiler.

*Fixed*

* Compile error with ``Apple clang clang-1300.0.29.30``.
* Incorrect OPLS dihedral forces when compiled with ``Apple clang clang-1300.0.29.30``.

*Deprecated*

* ``md.pair.SLJ`` - Replaced with ``md.pair.ExpandedLJ``.

*Removed*

* Leftover ``state`` logging category.

v3.0.0-beta.12 (2021-12-14)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

*Added*

* Support simulations with arbitrarily large or small scales (within the limits of the floating
  point representation).

*Changed*

* Report full error details in the exception message.
* Improved documentation.
* [breaking]: ``buffer`` is now a required argument when constructing a neighbor list.
* [breaking]: ``force_tol``, ``angmom_tol``, and ``energy_tol`` are now required arguments to
  ``md.minimize.FIRE``

*Fixed*

* Allow neighbor lists to store more than ``2**32-1`` total neighbors.
* Return expected parameter values instead of ``NaN`` when potential parameters are set to 0.

v3.0.0-beta.11 (2021-11-18)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

*Added*

- Support Python 3.10.
- Support clang 13.

*Changed*

- [developers] Place all all HOOMD C++ classes in the ``hoomd`` and nested namespaces.
- [developers] Use official pre-commit clang-format repository.

v3.0.0-beta.10 (2021-10-25)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

*Added*

- ``md.minimize.FIRE`` - MD integrator that minimizes the system's potential energy.
- Include example AKMA and MD unit conversion factors in the documentation.
- ``BUILD_LLVM`` CMake option  (defaults off) to enable features that require LLVM.
- ``hpmc.pair.user.CPPPotential`` - user-defined pair potentials between particles in HPMC.
- ``hpmc.pair.user.CPPPotentialUnion`` - user-defined site-site pair potentials between shapes
  in HPMC.
- ``hpmc.external.user.CPPExternalPotential`` - user-defined external potentials in HPMC.
- Support user-defined pair potentials in HPMC on the GPU.

*Changed*

- Improved documentation.
- Improved error messages when setting operation parameters.
- Noted some dependencies of dependencies for building documentation.
- [developers] Removed ``m_comm`` from most classes. Use ``m_sysdef->isDomainDecomposed()`` instead.
- Add support for LLVM 12
- ``ENABLE_LLVM=on`` requires the clang development libraries.
- [breaking] Renamed the Integrator attribute ``aniso`` to ``integrate_rotational_dof`` and removed
  the ``'auto'`` option. Users must now explicitly choose ``integrate_rotational_dof=True`` to
  integrate the rotational degrees of freedom in the system.

*Fixed*

- Calling ``Operations.__len__`` no longer raises a ``RecursionError``.
- RATTLE integration methods execute on the GPU.
- Include ``EvaluatorPairDLVO.h`` in the installation for plugins.
- Bug in setting zero sized ``ManagedArrays``.
- Kernel launch errors when one process uses different GPU devices.
- Race condition that lead to incorrect simulations with ``md.pair.Table``.
- Bug where some particle filers would have 0 rotational degrees of freedom.

*Removed*

- The ``BUILD_JIT`` CMake option.
- Support for LLVM <= 9.

v3.0.0-beta.9 (2021-09-08)
^^^^^^^^^^^^^^^^^^^^^^^^^^

*Added*

- ``Communicator.num_partitions`` - the number of partitions in the communicator.
- ``domain_decomposition`` argument to ``State`` factory methods - set the parameters of the MPI
  domain decomposition
- ``State.domain_decomposition`` - number of domains in the x, y, and z directions in the domain
  decomposition.
- ``State.domain_decomposition_split_fractions`` - the fractional positions of the split planes in
  the domain decomposition.
- ``hoomd.update.FilterUpdater`` - an updater that evaluates the particles associated with a
  `hoomd.filter.ParticleFilter` instance.
- ``hoomd.update.RemoveDrift`` - Remove the average drift from a system restrained on a lattice.
- Developer documentation for HOOMD-blue's Python object data model in ``ARCHITECTURE.md``.
- Autocomplete support for interactive notebooks.
- ``hoomd.md.methods.OverdampedViscous`` - Overdamped integrator with a drag force but no random
  force .
- ``MutabilityError`` exception when setting read-only operation parameters.

*Changed*

- Improved documentation.
- [breaking] Moved ``manifold_constrant`` to separate integration method classes in
  ``hoomd.md.methods.rattle``.
- [breaking] Moved ``trigger`` to first argument position in `hoomd.update.BoxResize`,
  `hoomd.write.DCD`, and `hoomd.write.GSD`.
- [breaking] ``hoomd.data.LocalSnapshot`` particle data API now matches ``Snapshot``. Changes to
  angular momentum, moment of intertia, and rigid body id attributes.
- ``hoomd.write.CustomWriter`` now exposes action through the ``writer`` attribute.
- [breaking] Active force rotational diffusion is managed by
  ``hoomd.md.update.ActiveRotationalDiffusion``.

*Fixed*

- ``TypeParameter`` can set multiple parameters after calling ``hoomd.Simulation.run``.
- ``tune.LoadBalancer`` can be used in a simulation.
- ``hoomd.md.pair.Pair`` ``r_cut`` type parameter can be set to 0.
- MD integration methods can be removed from the integrator's method list.
- Neighborlist exclusions update when the number of bonds change.
- Errors related to equality checks between HOOMD operations.
- The integrator can be removed from a simulation after running.
- ``hoomd.md.constrain.Rigid.create_bodies`` method correctly assigns the body attribute.
- Setting rigid attribute of a MD integrator to ``None`` is allowed.

*Deprecated*

*Removed*

- ``Snapshot.exists`` - use ``snapshot.communicator.rank == 0``
- ``State.snapshot`` - use ``get_snapshot`` and ``set_snapshot``
-   The ``State.box`` property setter - use ``State.set_box``

v3.0.0-beta.8 (2021-08-03)
^^^^^^^^^^^^^^^^^^^^^^^^^^

*Added*

- Consistent documentation of parameter dimensions and units reference documentation.
- ``md.update.ReversePerturbationFlow`` - implementation of ``mueller_plathe_flow`` from v2.
- ``md.pair.ExpandedMie`` - Mie potential where ``r`` is replaced with ``r - delta``.
- ``md.pair.Table`` - Pair potential evaluated using the given tabulated values.
- ``md.constrain.Distance`` - fix distances between pairs of particles.
- ``hpmc.compute.SDF`` - compute the pressure of convex hard particle systems.
- ``Snapshot.wrap()`` - wrap snapshot particles back into the box.
- Support gcc11.
- ``md.bond.Tether`` - A bond with minimum and maximum lengths.
- ``State.get_snapshot`` and ``State.set_snapshot`` - methods to access the global snapshot.
- ``State.set_box`` set a new simulation box without modifying particle properties.
- ``md.long_range.pppm.make_pppm_coulomb_forces`` - Long range electrostatics evaluated by PPPM.
- ``md.long_range.pppm.Coulomb`` - The reciprocal part of PPPM electrostatics.
- ``md.force.ActiveOnManifold`` - Active forces constrained to manifolds.

*Changed*

- Improved documentation.
- [breaking] Constructor arguments that set a default value per type or pair of types now have
  default in their name (e.g. ``r_cut`` to ``default_r_cut`` for pair potentials and ``a`` to
  ``default_a`` for HPMC integrators).
- [developer] Support git worktree checkouts.
- [breaking] Rename ``nrank`` to ``ranks_per_partition`` in ``Communicator``.
- rowan is now an optional dependency when running unit tests.
- ``Snapshot`` and ``Box`` methods that make in-place modifications return the object.

*Fixed*

- Bug where ``ThermdynamicQuantities.volume`` returned 0 in 2D simulations.
- Update neighbor list exclusions after the number of particles changes.
- Test failures with the CMake option ``BUILD_MD=off``.
- ``write.Table`` can now display MD pressures.

*Deprecated*

- ``State.snapshot`` - use ``get_snapshot`` and ``set_snapshot``.
- The ability to set boxes with the property ``State.box`` - use ``set_box``.

*Removed*

- [breaking] ``Simulation.write_debug_data``.
- [breaking] ``shared_msg_file`` option to ``Device``. ``msg_file`` now has the same behavior as
  ``shared_msg_file``.
- [developers] C++ and Python implementations of ``constraint_ellipsoid``, from ``hoomd.md.update``
  and ``sphere`` and ``oneD`` from ``hoomd.md.constrain``.
- [developers] Doxygen configuration files.


v3.0.0-beta.7 (2021-06-16)
^^^^^^^^^^^^^^^^^^^^^^^^^^

*Added*

- ``md.constrain.Rigid`` - Rigid body constraints.
- ``dem_built``, ``hpmc_built``, ``md_built``, and ``mpcd_built`` to ``hoomd.version`` - flags that
  indicate when optional submodules have been built.
- ``GPU.compute_capability`` property.
- [developers] pre-commit enforced style guidelines for the codebase.
- [developers] Validation tests for MD Lennard-Jones simulations.
- [developers] Unit tests for bond, angle, and dihedral potentials.

*Changed*

- Improved documentation on compiling HOOMD.
- Operations raise a ``DataAccessError`` when accessing properties that are not available because
  ``Simulation.run`` has not been called.
- ``TypeConversionError`` is now in the ``hoomd.error`` package.
- ``from_gsd_snapshot`` only accesses the GSD snapshot on MPI rank 0.

*Fixed*

- Some broken references in the documentation.
- Missing documentation for ``md.pair.TWF``.
- Inconsistent documentation in ``md.pair``.
- Correctly identify GPUs by ID in ``GPU.devices``.
- Don't initialize contexts on extra GPUs on MPI ranks.
- Support 2D inputs in ``from_gsd_snapshot``.

*Deprecated*

- ``Snapshot.exists`` - use ``Snapshot.communicator.rank == 0`` instead.

*Removed*

- [developers] C++ implementations of ``rescale_temp`` and ``enforce2d``.
- [developers] Unused methods of ``Integrator``.

v3.0.0-beta.6 (2021-05-17)
^^^^^^^^^^^^^^^^^^^^^^^^^^

*Added*

- ``md.pair.LJ0804`` - 8,4 Lennard-Jones pair potential.
- ``md.nlist.Stencil`` - Stencil algorithm to generate neighbor lists.
- ``md.nlist.Tree`` - BVH algorithm to generate neighbor lists.
- ``hoomd.md.Force``, ``hoomd.md.Operation``, and ``hoomd.md.Operations`` objects are now picklable.
- Manifold constraints using RATTLE with ``md.methods.NVE``, ``md.methods.Langevin`` and
  ``md.methods.Brownian``
  - Supporting sphere, ellipsoid, plane, cylinder, gyroid, diamond, and primitive manifolds.
- ``md.compute.HarmonicAveragedThermodynamicQuantities`` - More accurate thermodynamic quantities
  for crystals

*Changed*

- Raise an exception when initializing systems with invalid particle type ids.

*Fixed*

- Setting the operations attribute in ``Simulation`` objects in specific circumstances.
- Misc documentation updates.
- ``'sim' is not defined`` error when using ``md.dihedral`` potentials.

*Removed*

- C++ implemtation of v2 logging infrastructure.

v3.0.0-beta.5 (2021-03-23)
^^^^^^^^^^^^^^^^^^^^^^^^^^

*Added*

- ``filter`` parameter to ``update.BoxResize`` - A ``ParticleFilter`` that identifies the particles
  to scale with the box.
- ``Simulation.seed`` - one place to set random number seeds for all operations.
- ``net_force``, ``net_torque``, and ``net_energy`` per-particle arrays in local snapshots.
- Support ``hpmc.update.Clusters`` on the GPU.
- ``hpmc.update.MuVT`` - Gibbs ensemble simulations with HPMC.
- ``md.update.ZeroMomentum`` - Remove linear momentum from the system.
- ``hpmc.compute.FreeVolume`` - Compute free volume available to test particles.
- Custom action tutorials.

*Changed*

- [breaking]  Removed the parameter ``scale_particles`` in ``update.BoxResize``
- [internal] Modified signature of ``data.typeconverter.OnlyTypes``
- Remove use of deprecated numpy APIs.
- Added more details to the migration guide.
- Support timestep values in the range [0,2**64-1].
- [breaking] Removed *seed* argument from ``State.thermalize_particle_momenta``
- [breaking] Removed *seed* argument from ``md.methods.NVT.thermalize_thermostat_dof``
- [breaking] Removed *seed* argument from ``md.methods.NPT.thermalize_thermostat_and_barostat_dof``
- [breaking] Removed *seed* argument from ``md.methods.NPH.thermalize_barostat_dof``
- [breaking] Removed *seed* argument from ``md.methods.Langevin``
- [breaking] Removed *seed* argument from ``md.methods.Brownian``
- [breaking] Removed *seed* argument from ``md.force.Active``
- [breaking] Removed *seed* argument from ``md.pair.DPD``
- [breaking] Removed *seed* argument from ``md.pair.DPDLJ``
- [breaking] Removed *seed* argument from all HPMC integrators.
- [breaking] Removed *seed* argument from ``hpmc.update.Clusters``
- [breaking] Removed *seed* argument from ``hpmc.update.BoxMC``
- [breaking] Removed *seed* argument from ``hpmc.update.QuickCompress``
- Use latest version of getar library.
- Improve documentation.
- Improve performance of ``md.pair.Mie``.
- [breaking] ``hpmc.update.Clusters`` re-implemented with a rejection free, but not ergodic,
  algorithm for anisotropic particles. The new algorithm does not run in parallel over MPI ranks.
- [breaking] HPMC depletion algorithm rewritten.
- [breaking, temporary] HPMC depletant fugacity is now set for type pairs. This change will be
  reverted in a future release.
- Tutorials require fresnel 0.13.
- Support TBB 2021.

*Fixed*

- Install ``ParticleFilter`` header files for external plugins.
- ``md.force.Active`` keeps floating point values set for ``active_force`` and ``active_torque``.
- ``create_state_from_snapshot`` accepts ``gsd.hoomd.Snapshot`` objects without error.
- HOOMD compiles on Apple silicon macOS systems.
- Memory leak in PPPM force compute.
- Segmentation fault that occurred when dumping GSD shapes for spheropolygons and spheropolyhedra
  with 0 vertices.
- Incorrect MD neighbor lists in MPI simulations with more than 1 rank.
- ``md.bond.FENE`` accepts parameters.

*Removed*

- Testing with CUDA 9, GCC 4.8, GCC 5.x, GCC 6.x, clang 5

v3.0.0-beta.4 (2021-02-16)
^^^^^^^^^^^^^^^^^^^^^^^^^^

*Added*

- ``hoomd.write.DCD`` - DCD trajectory writer.
- ``hoomd.md.many_body`` - RevCross, SquareDensity, and Tersoff triplet
  potentials.
- ``hoomd.md.methods.Berendsen`` - Berendsen integration method.
- ``hoomd.md.methods.NPH`` - Constant pressure constant enthalpy integration
  method.
- ``hoomd.md.pair.TWF`` - Potential for modeling globular proteins by Pieter
  Rein ten Wolde and Daan Frenkel.
- Custom particle filters in Python via ``hoomd.filter.CustomFilter``.

*Changed*

- Documentation improvements.

*Fixed*

- Correctly determine the maximum ``r_cut`` in simulations with more than one
  pair potential and more than one type.

v3.0.0-beta.3 (2021-01-11)
^^^^^^^^^^^^^^^^^^^^^^^^^^

*Added*

- ``hoomd.variant.Variant`` objects are picklable.
- ``hoomd.filter.ParticleFilter`` objects are picklable.
- ``hoomd.trigger.Trigger`` objects are picklable.
- ``hoomd.Snapshot.from_gsd_snapshot`` - Convert GSD snapshots to HOOMD.
- ``hoomd.md.pair.aniso.GayBerne`` - Uniaxial ellipsoid pair potential.
- ``hoomd.md.pair.aniso.Dipole`` - Dipole pair potential.
- ``hoomd.md.pair.OPP`` - Oscillating pair potential.

*Changed*

- Improved compilation docs.
- Box equality checking now returns ``NotImplemented`` for non-``hoomd.Box``
  objects.
- ``Simulation.create_state_from_snapshot`` now accepts ``gsd.hoomd.Snapshot``
  objects.
- Attempting to run in a local snapshot context manager will now raise a
  ``RuntimeError``.
- Attempting to set the state to a new snapshot in a local snapshot context
  manager will now raise a ``RuntimeError``.

*Fixed*

- ``hoomd.variant.Power`` objects now have a ``t_ramp`` attribute as documented.
- Enable memory buffers larger than 2-4 GiB.
- Correctly write large image flags to GSD files.
- Support more than 26 default type names.
- Correctly represent fractional degrees of freedom.
- Compute the minimum image in double precision.

v3.0.0-beta.2 (2020-12-15)
^^^^^^^^^^^^^^^^^^^^^^^^^^

*Added*

- Support pybind11 2.6.0
- Exclusive creation file mode for ``write.GSD``.
- ``hpmc.update.BoxMC``.
- ``walltime`` and ``final_timestep`` loggable properties in ``Simulation``.
- ``Null`` particle filter.
- Logging tutorial.

*Changed*

- [breaking] Replace ``write.GSD`` argument ``overwrite`` with ``mode``.
- [breaking] Rename ``flags`` to ``categories`` in ``Logger``
- ``hoomd.snapshot.ConfigurationData.dimensions`` is not settable and is
  determined by the snapshot box. If ``box.Lz == 0``, the dimensions are 2
  otherwise 3.
- Building from source requires a C++14 compatible compiler.
- Improved documentation.
- ``hpmc.integrate.FacetedEllipsoid``'s shape specification now has a default
  origin of (0, 0, 0).
- Document loggable quantities in property docstrings.
- Skip GPU tests when no GPU is present.
- ``write.Table`` writes integers with integer formatting.

*Fixed*

- ``Simulation.run`` now ends with a ``KeyboardInterrupt`` exception when
  Jupyter interrupts the kernel.
- Logging the state of specific objects with nested attributes.
- Broken relative RPATHs.
- Add missing documentation for ``version.version``
- Error when removing specific operations from a simulation's operations
  attribute.
- Find CUDA libraries on additional Linux distributions.
- ``hpmc.update.Clusters`` now works with all HPMC integrators.
- ``Simulation.timestep`` reports the correct value when analyzers are called.
- ``Logger`` names quantities with the documented namespace name.

v3.0.0-beta.1 (2020-10-15)
^^^^^^^^^^^^^^^^^^^^^^^^^^

*Overview*

v3 has a completely new Python API. See the tutorials, migration guide and new
API documentation learn about it. The API documentation serves as the complete
list of all features currently implemented in v3.0.0-beta.1. Not all features in
v2 have been ported in v3.0.0-beta.1. Future beta releases will add additional
functionality.

*Added*

- Zero-copy data access through numpy (CPU) and cupy (GPU).
- User-defined operations in Python.
- User-defined triggers determine what time steps operations execute on.
- New logging subsystem supports array quantities and binary log files.
- Implicit depletants are now supported by any **hpmc** integrator through
  ``mc.set_fugacity('type', fugacity)``.
- Enable implicit depletants for two-dimensional shapes in **hpmc**.
- ``jit.patch.user()`` and ``jit.patch.user_union()`` now support GPUs via
  NVRTC.
- Add harmonically mapped averaging.
- Add Visual Studio Code workspace

*Changed*

- The ``run`` method has minimal overhead
- All loggable quantities are directly accessible as object properties.
- Operation parameters are always synchronized.
- Operations can be instantiated without a device or MPI communicator.
- Writers write output for ``step+1`` at the bottom of the ``run`` loop.
- HOOMD writes minimal output to stdout/stderr by default.
- *CMake* >=3.9, *cereal*, *eigen*, and *pybind11* are required to compile
  HOOMD.
- Plugins must be updated to build against v3.
- By default, HOOMD installs to the ``site-packages`` directory associated with
  the ``python`` executable given, which may be inside a virtual environment.
- Refactored CMake code.
- ``git submodule update`` no longer runs when during CMake configuration.
- Use ``random123`` library for implicit depletants in **hpmc**.
- HOOMD requires a GPU that supports concurrent managed memory access (Pascal
  or newer).

*Bug fixes*

- Improved accuracy of DLVO potential on the GPU.
- Improved performance of HPMC simulations on the CPU in non-cubic boxes.

*Removed*

- HOOMD-blue no longer parses command line options.
- Type swap moves in ``hpmc.update.muvt()`` are no longer supported
  (``transfer_ratio`` option to ``muvt.set_params()``)
- The option ``implicit=True`` to ``hpmc.integrate.*`` is no longer available
  (use ``set_fugacity``).
- ``static`` parameter in ``dump.gsd``
- ``util.quiet_status`` and ``util.unquiet_status``.
- ``deprecated.analyze.msd``.
- ``deprecated.dump.xml``.
- ``deprecated.dump.pos``.
- ``deprecated.init.read_xml``.
- ``deprecated.init.create_random``.
- ``deprecated.init.create_random_polymers``.
- **hpmc** ``ignore_overlaps`` parameter.
- **hpmc** ``sphere_union::max_members`` parameter.
- **hpmc** ``convex_polyhedron_union``.
- **hpmc** ``setup_pos_writer`` method.
- **hpmc** ``depletant_mode='circumsphere'``.
- **hpmc** ``max_verts`` parameter.
- **hpmc** ``depletant_mode`` parameter.
- **hpmc** ``ntrial`` parameter.
- **hpmc** ``implicit`` boolean parameter.
- ``group`` parameter to ``md.integrate.mode_minimize_fire``
- ``cgcmm.angle.cgcmm``
- ``cgcmm.pair.cgcmm``
- ``COPY_HEADERS`` *CMake* option.
- Many other python modules have been removed or re-implemented with new names.
  See the migration guide and new API documentation for a complete list.
- Support for NVIDIA GPUS with compute capability < 6.0.

v2.x
----

v2.9.7 (2021-08-03)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

* Support CUDA 11.5. A bug in CUDA 11.4 may result in the error
  ``__global__ function call is not configured`` when running HOOMD.

v2.9.6 (2021-03-16)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

* Support TBB 2021.

v2.9.5 (2021-03-15)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

* Support macos-arm64.
* Support TBB 2021.
* Fix memory leak in PPPM.

v2.9.4 (2021-02-05)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

* Support thrust 1.10
* Support LLVM11
* Fix Python syntax warnings
* Fix compile errors with gcc 10

v2.9.3 (2020-08-05)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

* Fix a compile error with CUDA 11

v2.9.2 (2020-06-26)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

* Fix a bug where repeatedly using objects with ``period=None`` would use
  significant amounts of memory.
* Support CUDA 11.
* Reccomend citing the 2020 Computational Materials Science paper
  10.1016/j.commatsci.2019.109363.

v2.9.1 (2020-05-28)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

* Fixed a minor bug where the variable period timestep would be off by one when
  the timestep got sufficiently large.
* Updated collections API to hide ``DeprecationWarning``.
* Fix scaling of cutoff in Gay-Berne potential to scale the current maximum
  distance based on the orientations of the particles, ensuring ellipsoidal
  energy isocontours.
* Misc documentation fixes.


v2.9.0 (2020-02-03)
^^^^^^^^^^^^^^^^^^^

*New features*

* General

  * Read and write GSD 2.0 files.

    * HOOMD >=2.9 can read and write GSD files created by HOOMD <= 2.8 or GSD
      1.x. HOOMD <= 2.8 cannot read GSD files created by HOOMD >=2.9 or GSD >=
      2.0.
    * OVITO >=3.0.0-dev652 reads GSD 2.0 files.
    * A future release of the ``gsd-vmd`` plugin will read GSD 2.0 files.

* HPMC

  * User-settable parameters in ``jit.patch``.
  * 2D system support in muVT updater.
  * Fix bug in HPMC where overlaps were not checked after adding new particle
    types.

* MD

  * The performance of ``nlist.tree`` has been drastically improved for a
    variety of systems.

v2.8.2 (2019-12-20)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

* Fix randomization of barostat and thermostat velocities with
  ``randomize_velocities()`` for non-unit temperatures.
* Improve MPCD documentation.
* Fix uninitialized memory in some locations which could have led to
  unreproducible results with HPMC in MPI, in particular with
  ``ALWAYS_USE_MANAGED_MEMORY=ON``.
* Fix calculation of cell widths in HPMC (GPU) and ``nlist.cell()`` with MPI.
* Fix potential memory-management issue in MPI for migrating MPCD particles and
  cell energy.
* Fix bug where exclusions were sometimes ignored when ``charge.pppm()`` is
  the only potential using the neighbor list.
* Fix bug where exclusions were not accounted for properly in the
  ``pppm_energy`` log quantity.
* Fix a bug where MD simulations with MPI start off without a ghost layer,
  leading to crashes or dangerous builds shortly after ``run()``.
* ``hpmc.update.remove_drift`` now communicates particle positions after
  updating them.

v2.8.1 (2019-11-26)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

* Fix a rare divide-by-zero in the ``collide.srd`` thermostat.
* Improve performance of first frame written by ``dump.gsd``.
* Support Python 3.8.
* Fix an error triggering migration of embedded particles for MPCD with MPI +
  GPU configurations.

v2.8.0 (2019-10-30)
^^^^^^^^^^^^^^^^^^^

*New Features*

- MD:

  - ``hoomd.md.dihedral.harmonic`` now accepts phase offsets, ``phi_0``, for CHARMM-style periodic dihedrals.
  - Enable per-type shape information for anisotropic pair potentials that complements the existing pair parameters struct.

- HPMC:

  - Enable the use of an array with adjustable parameters within the user defined pair potential.
  - Add muVT updater for 2D systems.


*Bug fixes*

- Fix missing header in external plugin builds.
- Enable ``couple='none'`` option to ``md.integrate.npt()`` when randomly initializing velocities.
- Documentation improvements.
- Skip gsd shape unit test when required modules are not compiled.
- Fix default particle properties when new particles are added to the system (e.g., via the muVT updater).
- Fix ``charge.pppm()`` execution on multiple GPUs.
- Enable ``with SimulationContext() as c``.
- Fix a bug for ``mpcd.collide.at`` with embedded particles, which may have given incorrect results or simulation crashes.

v2.7.0 (2019-10-01)
^^^^^^^^^^^^^^^^^^^

*New features*

- General:

  - Allow components to use ``Logger`` at the C++ level.
  - Drop support for python 2.7.
  - User-defined log quantities in ``dump.gsd``.
  - Add ``hoomd.dump.gsd.dump_shape`` to save particle shape information in GSD files.

- HPMC:

  - Add ``get_type_shapes`` to ``ellipsoid``.

- MPCD:

  - ``mpcd.stream.slit_pore`` allows for simulations through parallel-plate (lamellar) pores.
  - ``mpcd.integrate`` supports integration of MD (solute) particles with bounce-back rules in MPCD streaming geometries.

*Bug fixes*

- ``hoomd.hdf5.log.query`` works with matrix quantities.
- ``test_group_rigid.py`` is run out of the ``md`` module.
- Fix a bug in ``md.integrate.langevin()`` and ``md.integrate.bd()`` where on the GPU the value of ``gamma`` would be ignored.
- Fix documentation about interoperability between ``md.mode_minimize_fire()`` and MPI.
- Clarify ``dump.gsd`` documentation.
- Improve documentation of ``lattice_field`` and ``frenkel_ladd_energy`` classes.
- Clarify singularity image download documentation.
- Correctly document the functional form of the Buckingham pair potential.
- Correct typos in HPMC example snippets.
- Support compilation in WSL.

v2.6.0 (2019-05-28)
^^^^^^^^^^^^^^^^^^^

*New features*

- General:

  - Enable ``HPMC`` plugins.
  - Fix plug-in builds when ``ENABLE_TBB`` or ``ALWAYS_USE_MANAGED_MEMORY`` CMake parameters are set.
  - Remove support for compute 3.0 GPUs.
  - Report detailed CUDA errors on initialization.
  - Document upcoming feature removals and API changes.

- MD:

  - Exclude neighbors that belong to the same floppy molecule.
  - Add fourier potential.

- HPMC:

  - New shape class: ``hpmc.integrate.faceted_ellipsoid_union()``.
  - Store the *orientable* shape state.

- MPCD:

  - ``mpcd.stream.slit`` allows for simulations in parallel-plate channels. Users can implement other geometries as a plugin.
  - MPCD supports virtual particle filling in bounded geometries through the ``set_filler`` method of ``mpcd.stream`` classes.
  - ``mpcd.stream`` includes an external ``mpcd.force`` acting on the MPCD particles. A block force, a constant force, and a sine force are implemented.

*Bug fixes*

- Fix compile errors with LLVM 8 and ``-DBUILD_JIT=on``.
- Allow simulations with 0 bonds to specify bond potentials.
- Fix a problem where HOOMD could not be imported in ``mpi4py`` jobs.
- Validate snapshot input in ``restore_snapshot``.
- Fix a bug where rigid body energy and pressure deviated on the first time step after ``run()``.
- Fix a bug which could lead to invalid MPI simulations with ``nlist.cell()`` and ``nlist.stencil()``.

*C++ API changes*

- Refactor handling of ``MPI_Comm`` inside library
- Use ``random123`` for random number generation
- CMake version 2.8.10.1 is now a minimum requirement for compiling from source

v2.5.2 (2019-04-30)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

- Support LLVM 9 in ``jit``
- Fix error when importing ``jit`` before ``hpmc``
- HPMC integrators raise errors when ``restore_state=True`` and state information is missing
- Send messages to replaced ``sys.stdout`` and ``sys.stderr`` streams
- Add ``hpmc.update.clusters`` to documentation index
- Fix a bug in the MPCD Gaussian random number generator that could lead to NaN values
- Fix issue where an initially cubic box can become non-cubic with ``integrate.npt()`` and ``randomize_velocities()``
- Fix illegal memory access in NeighborListGPU with ``-DALWAYS_USE_MANAGED_MEMORY=ON`` on single GPUs
- Improve ``pair.table`` performance with multi-GPU execution
- Improve ``charge.pppm`` performance with multi-GPU execution
- Improve rigid body performance with multi-GPU execution
- Display correct cell list statistics with the ``-DALWAYS_USE_MANAGED_MEMORY=ON`` compile option
- Fix a sporadic data corruption / bus error issue when data structures are dynamically resized in simulations that use unified memory (multi-GPU, or with -DALWAYS_USE_MANAGED_MEMORY=ON compile time option)
- Improve ``integrate.nve`` and ``integrate.npt`` performance with multi-GPU execution
- Improve some angular degrees of freedom integrators with multi-GPU execution
- Improve rigid body pressure calculation performance with multi-GPU execution

v2.5.1 (2019-03-14)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

- fix out-of-range memory access in ``hpmc.integrate.convex_polyheron``
- Remove support for clang3.8 and 4.0
- Documentation improvements
- Fix a segfault when using ``SLURM_LOCALID``

v2.5.0 (2019-02-05)
^^^^^^^^^^^^^^^^^^^

*New features*

-  General:

   -  Fix BondedGroupData and CommunicatorGPU compile errors in certain
      build configurations

-  MD:

   -  Generalize ``md.integrate.brownian`` and ``md.integrate.langevin``
      to support anisotropic friction coefficients for rotational
      Brownian motion.
   -  Improve NVLINK performance with rigid bodies
   -  ``randomize_velocities`` now chooses random values for the
      internal integrator thermostat and barostat variables.
   -  ``get_net_force`` returns the net force on a group of particles
      due to a specific force compute

-  HPMC:

   -  Fix a bug where external fields were ignored with the HPMC
      implicit integrator unless a patch potential was also in use.

-  JIT:

   -  Add ``jit.external.user`` to specify user-defined external fields
      in HPMC.
   -  Use ``-DHOOMD_LLVMJIT_BUILD`` now instead of ``-DHOOMD_NOPYTHON``

v2.4.2 (2018-12-20)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Miscellaneous documentation updates
-  Fix compile error with ``with -DALWAYS_USE_MANAGED_MEMORY=ON``
-  Fix MuellerPlatheFlow, cast input parameter to int to avoid C++
   constructor type mismatch
-  Improve startup time with multi-GPU simulations
-  Correctly assign GPUs to MPI processes on Summit when launching with
   more than one GPU per resource set
-  Optimize multi-GPU performance with NVLINK
-  Do not use mapped memory with MPI/GPU anymore
-  Fix some cases where a multi-GPU simulation fails with an alignment
   error
-  Eliminate remaining instance of unsafe ``__shfl``
-  Hide CMake warnings regarding missing CPU math libraries
-  Hide CMake warning regarding missing MPI<->CUDA interoperability
-  Refactor memory management to fix linker errors with some compilers

*C++ API Changes*

-  May break some plug-ins which rely on ``GPUArray`` data type being
   returned from ``ParticleData`` and other classes (replace by
   ``GlobalArray``)

v2.4.1 (2018-11-27)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Install ``WarpTools.cuh`` for use by plugins
-  Fix potential violation of detailed balance with anisotropic
   particles with ``hpmc.update.clusters`` in periodic boundary
   conditions
-  Support llvm 7.0

v2.4.0 (2018-11-07)
^^^^^^^^^^^^^^^^^^^

*New features*

-  General:

   -  Misc documentation updates
   -  Accept ``mpi4py`` communicators in ``context.initialize``.
   -  CUDA 10 support and testing
   -  Sphinx 1.8 support
   -  Flush message output so that ``python -u`` is no longer required
      to obtain output on some batch job systems
   -  Support multi-GPU execution on dense nodes using CUDA managed
      memory. Execute with ``--gpu=0,1,..,n-1`` command line option to
      run on the first n GPUs (Pascal and above).

      -  Node-local acceleration is implemented for a subset of kernels.
         Performance improvements may vary.
      -  Improvements are only expected with NVLINK hardware. Use MPI
         when NVLINK is not available.
      -  Combine the ``--gpu=..`` command line option with mpirun to
         execute on many dense nodes

   -  Bundle ``libgetar`` v0.7.0 and remove ``sqlite3`` dependency
   -  When building with ENABLE_CUDA=on, CUDA 8.0 is now a minimum
      requirement

-  MD:

   -  *no changes*.

-  HPMC:

   -  Add ``convex_spheropolyhedron_union`` shape class.
   -  Correctly count acceptance rate when maximum particle move is is
      zero in ``hpmc.integrate.*``.
   -  Correctly count acceptance rate when maximum box move size is zero
      in ``hpmc.update.boxmc``.
   -  Fix a bug that may have led to overlaps between polygon soups with
      ``hpmc.integrate.polyhedron``.
   -  Improve performance in sphere trees used in
      ``hpmc.integrate.sphere_union``.
   -  Add ``test_overlap`` method to python API

-  API:

   -  Allow external callers of HOOMD to set the MPI communicator
   -  Removed all custom warp reduction and scan operations. These are
      now performed by CUB.
   -  Separate compilation of pair potentials into multiple files.
   -  Removed compute 2.0 workaround implementations. Compute 3.0 is now
      a hard minimum requirement to run HOOMD.
   -  Support and enable compilation for sm70 with CUDA 9 and newer.

-  Deprecated:

   -  HPMC: The implicit depletant mode ``circumsphere`` with
      ``ntrial > 0`` does not support compute 7.0 (Volta) and newer GPUs
      and is now disabled by default. To enable this functionality,
      configure HOOMD with option the ``-DENABLE_HPMC_REINSERT=ON``,
      which will not function properly on compute 7.0 (Volta) and newer
      GPUs.
   -  HPMC: ``convex_polyhedron_union`` is replaced by
      ``convex_spheropolyhedron_union`` (when sweep_radii are 0 for all
      particles)

v2.3.5 (2018-10-07)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Document ``--single-mpi`` command line option.
-  HPMC: Fix a bug where ``hpmc.field.lattice_field`` did not resize 2D
   systems properly in combination with ``update.box_resize``.

v2.3.4 (2018-07-30)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  ``init.read_gsd`` no longer applies the *time_step* override when
   reading the *restart* file
-  HPMC: Add ``hpmc_patch_energy`` and ``hpmc_patch_rcut`` loggable
   quantities to the documentation

v2.3.3 (2018-07-03)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Fix ``libquickhull.so`` not found regression on Mac OS X

v2.3.2 (2018-06-29)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Fix a bug where gsd_snapshot would segfault when called without an
   execution context.
-  Compile warning free with gcc8.
-  Fix compile error when TBB include files are in non-system directory.
-  Fix ``libquickhull.so`` not found error on additional platforms.
-  HOOMD-blue is now available on **conda-forge** and the **docker
   hub**.
-  MPCD: Default value for ``kT`` parameter is removed for
   ``mpcd.collide.at``. Scripts that are correctly running are not
   affected by this change.
-  MPCD: ``mpcd`` notifies the user of the appropriate citation.
-  MD: Correct force calculation between dipoles and point charge in
   ``pair.dipole``

*Deprecated*

-  The **anaconda** channel **glotzer** will no longer be updated. Use
   **conda-forge** to upgrade to v2.3.2 and newer versions.

v2.3.1 (2018-05-25)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Fix doxygen documentation syntax errors
-  Fix libquickhull.so not found error on some platforms
-  HPMC: Fix bug that allowed particles to pas through walls
-  HPMC: Check spheropolyhedra with 0 vertices against walls correctly
-  HPMC: Fix plane wall/spheropolyhedra overlap test
-  HPMC: Restore detailed balance in implicit depletant integrator
-  HPMC: Correctly choose between volume and lnV moves in
   ``hpmc.update.boxmc``
-  HPMC: Fix name of log quantity ``hpmc_clusters_pivot_acceptance``
-  MD: Fix image list for tree neighbor lists in 2d

v2.3.0 (2018-04-25)
^^^^^^^^^^^^^^^^^^^

*New features*

-  General:

   -  Store ``BUILD_*`` CMake variables in the hoomd cmake cache for use
      in external plugins.
   -  ``init.read_gsd`` and ``data.gsd_snapshot`` now accept negative
      frame indices to index from the end of the trajectory.
   -  Faster reinitialization from snapshots when done frequently.
   -  New command line option ``--single-mpi`` allows non-mpi builds of
      hoomd to launch within mpirun (i.e. for use with mpi4py managed
      pools of jobs)
   -  For users of the University of Michigan Flux system: A ``--mode``
      option is no longer required to run hoomd.

-  MD:

   -  Improve performance with ``md.constrain.rigid`` in multi-GPU
      simulations.
   -  New command ``integrator.randomize_velocities()`` sets a particle
      group’s linear and angular velocities to random values consistent
      with a given kinetic temperature.
   -  ``md.force.constant()`` now supports setting the force per
      particle and inside a callback

-  HPMC:

   -  Enabled simulations involving spherical walls and convex
      spheropolyhedral particle shapes.
   -  Support patchy energetic interactions between particles (CPU only)
   -  New command ``hpmc.update.clusters()`` supports geometric cluster
      moves with anisotropic particles and/or depletants and/or patch
      potentials. Supported move types: pivot and line reflection
      (geometric), and AB type swap.

-  JIT:

   -  Add new experimental ``jit`` module that uses LLVM to compile and
      execute user provided C++ code at runtime. (CPU only)
   -  Add ``jit.patch.user``: Compute arbitrary patch energy between
      particles in HPMC (CPU only)
   -  Add ``jit.patch.user_union``: Compute arbitrary patch energy
      between rigid unions of points in HPMC (CPU only)
   -  Patch energies operate with implicit depletant and normal HPMC
      integration modes.
   -  ``jit.patch.user_union`` operates efficiently with additive
      contributions to the cutoff.

-  MPCD:

   -  The ``mpcd`` component adds support for simulating hydrodynamics
      using the multiparticle collision dynamics method.

*Beta feature*

-  Node local parallelism (optional, build with ``ENABLE_TBB=on``):

   -  The Intel TBB library is required to enable this feature.
   -  The command line option ``--nthreads`` limits the number of
      threads HOOMD will use. The default is all CPU cores in the
      system.
   -  Only the following methods in HOOMD will take advantage of
      multiple threads:

      -  ``hpmc.update.clusters()``
      -  HPMC integrators with implicit depletants enabled
      -  ``jit.patch.user_union``

Node local parallelism is still under development. It is not enabled in
builds by default and only a few methods utilize multiple threads. In
future versions, additional methods in HOOMD may support multiple
threads.

To ensure future workflow compatibility as future versions enable
threading in more components, explicitly set –nthreads=1.

*Bug fixes*

-  Fixed a problem with periodic boundary conditions and implicit
   depletants when ``depletant_mode=circumsphere``
-  Fixed a rare segmentation fault with ``hpmc.integrate.*_union()`` and
   ``hpmc.integrate.polyhedron``
-  ``md.force.active`` and ``md.force.dipole`` now record metadata
   properly.
-  Fixed a bug where HPMC restore state did not set ignore flags
   properly.
-  ``hpmc_boxmc_ln_volume_acceptance`` is now available for logging.

*Other changes*

-  Eigen is now provided as a submodule. Plugins that use Eigen headers
   need to update include paths.
-  HOOMD now builds with pybind 2.2. Minor changes to source and cmake
   scripts in plugins may be necessary. See the updated example plugin.
-  HOOMD now builds without compiler warnings on modern compilers (gcc6,
   gcc7, clang5, clang6).
-  HOOMD now uses pybind11 for numpy arrays instead of ``num_util``.
-  HOOMD versions v2.3.x will be the last available on the anaconda
   channel ``glotzer``.

v2.2.5 (2018-04-20)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Pin cuda compatible version in conda package to resolve ``libcu*.so``
   not found errors in conda installations.

v2.2.4 (2018-03-05)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Fix a rare error in ``md.nlist.tree`` when particles are very close
   to each other.
-  Fix deadlock when ```init.read_getar``` is given different file names
   on different ranks.
-  Sample from the correct uniform distribution of depletants in a
   sphere cap with ``depletant_mode='overlap_regions'`` on the CPU
-  Fix a bug where ternary (or higher order) mixtures of small and large
   particles were not correctly handled with
   ``depletant_mode='overlap_regions'`` on the CPU
-  Improve acceptance rate in depletant simulations with
   ``depletant_mode='overlap_regions'``

v2.2.3 (2018-01-25)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Write default values to gsd frames when non-default values are
   present in frame 0.
-  ``md.wall.force_shifted_lj`` now works.
-  Fix a bug in HPMC where ``run()`` would not start after
   ``restore_state`` unless shape parameters were also set from python.
-  Fix a bug in HPMC Box MC updater where moves were attempted with zero
   weight.
-  ``dump.gsd()`` now writes ``hpmc`` shape state correctly when there
   are multiple particle types.
-  ``hpmc.integrate.polyhedron()`` now produces correct results on the
   GPU.
-  Fix binary compatibility across python minor versions.

v2.2.2 (2017-12-04)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  ``md.dihedral.table.set_from_file`` now works.
-  Fix a critical bug where forces in MPI simulations with rigid bodies
   or anisotropic particles were incorrectly calculated
-  Ensure that ghost particles are updated after load balancing.
-  ``meta.dump_metadata`` no longer reports an error when used with
   ``md.constrain.rigid``
-  Miscellaneous documentation fixes
-  ``dump.gsd`` can now write GSD files with 0 particles in a frame
-  Explicitly report MPI synchronization delays due to load imbalance
   with ``profile=True``
-  Correctly compute net torque of rigid bodies with anisotropic
   constituent particles in MPI execution on multiple ranks
-  Fix ``PotentialPairDPDThermoGPU.h`` for use in external plugins
-  Use correct ghost region with ``constrain.rigid`` in MPI execution on
   multiple ranks
-  ``hpmc.update.muvt()`` now works with
   ``depletant_mode='overlap_regions'``
-  Fix the sampling of configurations with in ``hpmc.update.muvt`` with
   depletants
-  Fix simulation crash after modifying a snapshot and re-initializing
   from it
-  The pressure in simulations with rigid bodies
   (``md.constrain.rigid()``) and MPI on multiple ranks is now computed
   correctly

v2.2.1 (2017-10-04)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Add special pair headers to install target
-  Fix a bug where ``hpmc.integrate.convex_polyhedron``,
   ``hpmc.integrate.convex_spheropolyhedron``,
   ``hpmc.integrate.polyedron``, ``hpmc.integrate.faceted_sphere``,
   ``hpmc.integrate.sphere_union`` and
   ``hpmc.integrate.convex_polyhedron_union`` produced spurious overlaps
   on the GPU

v2.2.0 (2017-09-08)
^^^^^^^^^^^^^^^^^^^

*New features*

-  General:

   -  Add ``hoomd.hdf5.log`` to log quantities in hdf5 format. Matrix
      quantities can be logged.
   -  ``dump.gsd`` can now save internal state to gsd files. Call
      ``dump_state(object)`` to save the state for a particular object.
      The following objects are supported:

      -  HPMC integrators save shape and trial move size state.

   -  Add *dynamic* argument to ``hoomd.dump.gsd`` to specify which
      quantity categories should be written every frame.
   -  HOOMD now inter-operates with other python libraries that set the
      active CUDA device.
   -  Add generic capability for bidirectional ghost communication,
      enabling multi body potentials in MPI simulation.

-  MD:

   -  Added support for a 3 body potential that is harmonic in the local
      density.
   -  ``force.constant`` and ``force.active`` can now apply torques.
   -  ``quiet`` option to ``nlist.tune`` to quiet the output of the
      embedded ``run()`` commands.
   -  Add special pairs as exclusions from neighbor lists.
   -  Add cosine squared angle potential ``md.angle.cosinesq``.
   -  Add ``md.pair.DLVO()`` for evaluation of colloidal dispersion and
      electrostatic forces.
   -  Add Lennard-Jones 12-8 pair potential.
   -  Add Buckingham (exp-6) pair potential.
   -  Add Coulomb 1-4 special_pair potential.
   -  Check that composite body dimensions are consistent with minimum
      image convention and generate an error if they are not.
   -  ``md.integrate.mode.minimize_fire()`` now supports anisotropic
      particles (i.e. composite bodies)
   -  ``md.integrate.mode.minimize_fire()`` now supports flexible
      specification of integration methods
   -  ``md.integrate.npt()/md.integrate.nph()`` now accept a friction
      parameter (gamma) for damping out box fluctuations during
      minimization runs
   -  Add new command ``integrate.mode_standard.reset_methods()`` to
      clear NVT and NPT integrator variables

-  HPMC:

   -  ``hpmc.integrate.sphere_union()`` takes new capacity parameter to
      optimize performance for different shape sizes
   -  ``hpmc.integrate.polyhedron()`` takes new capacity parameter to
      optimize performance for different shape sizes
   -  ``hpmc.integrate.convex_polyhedron`` and
      ``convex_spheropolyhedron`` now support arbitrary numbers of
      vertices, subject only to memory limitations (``max_verts`` is now
      ignored).
   -  HPMC integrators restore state from a gsd file read by
      ``init.read_gsd`` when the option ``restore_state`` is ``True``.
   -  Deterministic HPMC integration on the GPU (optional):
      ``mc.set_params(deterministic=True)``.
   -  New ``hpmc.update.boxmc.ln_volume()`` move allows logarithmic
      volume moves for fast equilibration.
   -  New shape: ``hpmc.integrate.convex_polyhedron_union`` performs
      simulations of unions of convex polyhedra.
   -  ``hpmc.field.callback()`` now enables MC energy evaluation in a
      python function
   -  The option ``depletant_mode='overlap_regions'`` for
      ``hpmc.integrate.*`` allows the selection of a new depletion
      algorithm that restores the diffusivity of dilute colloids in
      dense depletant baths

*Deprecated*

-  HPMC: ``hpmc.integrate.sphere_union()`` no longer needs the
   ``max_members`` parameter.
-  HPMC: ``hpmc.integrate.convex_polyhedron`` and
   ``convex_spheropolyhedron`` no longer needs the ``max_verts``
   parameter.
-  The *static* argument to ``hoomd.dump.gsd`` should no longer be used.
   Use *dynamic* instead.

*Bug fixes*

-  HPMC:

   -  ``hpmc.integrate.sphere_union()`` and
      ``hpmc.integrate.polyhedron()`` missed overlaps.
   -  Fix alignment error when running implicit depletants on GPU with
      ntrial > 0.
   -  HPMC integrators now behave correctly when the user provides
      different RNG seeds on different ranks.
   -  Fix a bug where overlapping configurations were produced with
      ``hpmc.integrate.faceted_sphere()``

-  MD:

   -  ``charge.pppm()`` with ``order=7`` now gives correct results
   -  The PPPM energy for particles excluded as part of rigid bodies now
      correctly takes into account the periodic boundary conditions

-  EAM:

   -  ``metal.pair.eam`` now produces correct results.

*Other changes*

-  Optimized performance of HPMC sphere union overlap check and
   polyhedron shape
-  Improved performance of rigid bodies in MPI simulations
-  Support triclinic boxes with rigid bodies
-  Raise an error when an updater is given a period of 0
-  Revised compilation instructions
-  Misc documentation improvements
-  Fully document ``constrain.rigid``
-  ``-march=native`` is no longer set by default (this is now a
   suggestion in the documentation)
-  Compiler flags now default to CMake defaults
-  ``ENABLE_CUDA`` and ``ENABLE_MPI`` CMake options default OFF. User
   must explicitly choose to enable optional dependencies.
-  HOOMD now builds on powerpc+CUDA platforms (tested on summitdev)
-  Improve performance of GPU PPPM force calculation
-  Use sphere tree to further improve performance of
   ``hpmc.integrate.sphere_union()``

v2.1.9 (2017-08-22)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Fix a bug where the log quantity ``momentum`` was incorrectly
   reported in MPI simulations.
-  Raise an error when the user provides inconsistent ``charge`` or
   ``diameter`` lists to ``md.constrain.rigid``.
-  Fix a bug where ``pair.compute_energy()`` did not report correct
   results in MPI parallel simulations.
-  Fix a bug where make rigid bodies with anisotropic constituent
   particles did not work on the GPU.
-  Fix hoomd compilation after the rebase in the cub repository.
-  ``deprecated.dump.xml()`` now writes correct results when particles
   have been added or deleted from the simulation.
-  Fix a critical bug where ``charge.pppm()`` calculated invalid forces
   on the GPU

v2.1.8 (2017-07-19)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  ```init.read_getar``` now correctly restores static quantities when
   given a particular frame.
-  Fix bug where many short calls to ``run()`` caused incorrect results
   when using ``md.integrate.langevin``.
-  Fix a bug in the Saru pseudo-random number generator that caused some
   double-precision values to be drawn outside the valid range [0,1) by
   a small amount. Both floats and doubles are now drawn on [0,1).
-  Fix a bug where coefficients for multi-character unicode type names
   failed to process in Python 2.

*Other changes*

-  The Saru generator has been moved into ``hoomd/Saru.h``, and plugins
   depending on Saru or SaruGPU will need to update their includes. The
   ``SaruGPU`` class has been removed. Use ``hoomd::detail::Saru``
   instead for both CPU and GPU plugins.

v2.1.7 (2017-05-11)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Fix PPM exclusion handling on the CPU
-  Handle ``r_cut`` for special pairs correctly
-  Fix tauP reference in NPH documentation
-  Fixed ``constrain.rigid`` on compute 5.x.
-  Fixed random seg faults when using sqlite getar archives with LZ4
   compression
-  Fixed XZ coupling with ``hoomd.md.integrate.npt`` integration
-  Fixed aspect ratio with non-cubic boxes in
   ``hoomd.hpmc.update.boxmc``

v2.1.6 (2017-04-12)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Document ``hpmc.util.tune_npt``
-  Fix dump.getar.writeJSON usage with MPI execution
-  Fix a bug where integrate.langevin and integrate.brownian correlated
   RNGs between ranks in multiple CPU execution
-  Bump CUB to version 1.6.4 for improved performance on Pascal
   architectures. CUB is now embedded using a git submodule. Users
   upgrading existing git repositories should reinitialize their git
   submodules with ``git submodule update --init``
-  CMake no longer complains when it finds a partial MKL installation.

v2.1.5 (2017-03-09)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Fixed a compile error on Mac

v2.1.4 (2017-03-09)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Fixed a bug re-enabling disabled integration methods
-  Fixed a bug where adding particle types to the system failed for
   anisotropic pair potentials
-  scipy is no longer required to execute DEM component unit tests
-  Issue a warning when a subsequent call to context.initialize is given
   different arguments
-  DPD now uses the seed from rank 0 to avoid incorrect simulations when
   users provide different seeds on different ranks
-  Miscellaneous documentation updates
-  Defer initialization message until context.initialize
-  Fixed a problem where a momentary dip in TPS would cause walltime
   limited jobs to exit prematurely
-  HPMC and DEM components now correctly print citation notices

v2.1.3 (2017-02-07)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Fixed a bug where the WalltimeLimitReached was ignored

v2.1.2 (2017-01-11)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  (HPMC) Implicit depletants with spheres and faceted spheres now
   produces correct ensembles
-  (HPMC) Implicit depletants with ntrial > 0 now produces correct
   ensembles
-  (HPMC) NPT ensemble in HPMC (``hpmc.update.boxmc``) now produces
   correct ensembles
-  Fix a bug where multiple nvt/npt integrators caused warnings from
   analyze.log.
-  update.balance() is properly ignored when only one rank is available
-  Add missing headers to plugin install build
-  Fix a bug where charge.pppm calculated an incorrect pressure

-  Other changes \*

-  Drop support for compute 2.0 GPU devices
-  Support cusolver with CUDA 8.0

v2.1.1 (2016-10-23)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Fix ``force.active`` memory allocation bug
-  Quiet Python.h warnigns when building (python 2.7)
-  Allow multi-character particle types in HPMC (python 2.7)
-  Enable ``dump.getar.writeJSON`` in MPI
-  Allow the flow to change directions in
   ``md.update.mueller_plathe_flow``
-  Fix critical bug in MPI communication when using HPMC integrators

v2.1.0 (2016-10-04)
^^^^^^^^^^^^^^^^^^^

*New features*

-  enable/disable overlap checks between pairs of constituent particles
   for ``hpmc.integrate.sphere_union()``
-  Support for non-additive mixtures in HPMC, overlap checks can now be
   enabled/disabled per type-pair
-  Add ``md.constrain.oned`` to constrain particles to move in one
   dimension
-  ``hpmc.integrate.sphere_union()`` now takes max_members as an
   optional argument, allowing to use GPU memory more efficiently
-  Add ``md.special_pair.lj()`` to support scaled 1-4 (or other)
   exclusions in all-atom force fields
-  ``md.update.mueller_plathe_flow()``: Method to create shear flows in
   MD simulations
-  ``use_charge`` option for ``md.pair.reaction_field``
-  ``md.charge.pppm()`` takes a Debye screening length as an optional
   parameter
-  ``md.charge.pppm()`` now computes the rigid body correction to the
   PPPM energy

*Deprecated*

-  HPMC: the ``ignore_overlaps`` flag is replaced by
   ``hpmc.integrate.interaction_matrix``

*Other changes*

-  Optimized MPI simulations of mixed systems with rigid and non-rigid
   bodies
-  Removed dependency on all boost libraries. Boost is no longer needed
   to build hoomd
-  Intel compiler builds are no longer supported due to c++11 bugs
-  Shorter compile time for HPMC GPU kernels
-  Include symlinked external components in the build process
-  Add template for external components
-  Optimized dense depletant simulations with HPMC on CPU

*Bug fixes*

-  fix invalid mesh energy in non-neutral systems with
   ``md.charge.pppm()``
-  Fix invalid forces in simulations with many bond types (on GPU)
-  fix rare cases where analyze.log() would report a wrong pressure
-  fix possible illegal memory access when using
   ``md.constrain.rigid()`` in GPU MPI simulations
-  fix a bug where the potential energy is misreported on the first step
   with ``md.constrain.rigid()``
-  Fix a bug where the potential energy is misreported in MPI
   simulations with ``md.constrain.rigid()``
-  Fix a bug where the potential energy is misreported on the first step
   with ``md.constrain.rigid()``
-  ``md.charge.pppm()`` computed invalid forces
-  Fix a bug where PPPM interactions on CPU where not computed correctly
-  Match logged quantitites between MPI and non-MPI runs on first time
   step
-  Fix ``md.pair.dpd`` and ``md.pair.dpdlj`` ``set_params``
-  Fix diameter handling in DEM shifted WCA potential
-  Correctly handle particle type names in lattice.unitcell
-  Validate ``md.group.tag_list`` is consistent across MPI ranks

v2.0.3 (2016-08-30)
^^^^^^^^^^^^^^^^^^^

-  hpmc.util.tune now works with particle types as documented
-  Fix pressure computation with pair.dpd() on the GPU
-  Fix a bug where dump.dcd corrupted files on job restart
-  Fix a bug where HPMC walls did not work correctly with MPI
-  Fix a bug where stdout/stderr did not appear in MPI execution
-  HOOMD will now report an human readable error when users forget
   context.initialize()
-  Fix syntax errors in frenkel ladd field

v2.0.2 (2016-08-09)
^^^^^^^^^^^^^^^^^^^

-  Support CUDA Toolkit 8.0
-  group.rigid()/nonrigid() did not work in MPI simulations
-  Fix builds with ENABLE_DOXYGEN=on
-  Always add -std=c++11 to the compiler command line arguments
-  Fix rare infinite loops when using hpmc.integrate.faceted_sphere
-  Fix hpmc.util.tune to work with more than one tunable
-  Fix a bug where dump.gsd() would write invalid data in simulations
   with changing number of particles
-  replicate() sometimes did not work when restarting a simulation

v2.0.1 (2016-07-15)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Fix acceptance criterion in mu-V-T simulations with implicit
   depletants (HPMC).
-  References to disabled analyzers, computes, updaters, etc. are
   properly freed from the simulation context.
-  Fix a bug where ``init.read_gsd`` ignored the ``restart`` argument.
-  Report an error when HPMC kernels run out of memory.
-  Fix ghost layer when using rigid constraints in MPI runs.
-  Clarify definition of the dihedral angle.

v2.0.0 (2016-06-22)
^^^^^^^^^^^^^^^^^^^

HOOMD-blue v2.0 is released under a clean BSD 3-clause license.

*New packages*

-  ``dem`` - simulate faceted shapes with dynamics
-  ``hpmc`` - hard particle Monte Carlo of a variety of shape classes.

*Bug fixes*

-  Angles, dihedrals, and impropers no longer initialize with one
   default type.
-  Fixed a bug where integrate.brownian gave the same x,y, and z
   velocity components.
-  Data proxies verify input types and vector lengths.
-  dump.dcd no longer generates excessive metadata traffic on lustre
   file systems

*New features*

-  Distance constraints ``constrain.distance`` - constrain pairs of
   particles to a fixed separation distance
-  Rigid body constraints ``constrain.rigid`` - rigid bodies now have
   central particles, and support MPI and replication
-  Multi-GPU electrostatics ``charge.pppm`` - the long range
   electrostatic forces are now supported in MPI runs
-  ``context.initialize()`` can now be called multiple times - useful in
   jupyter notebooks
-  Manage multiple simulations in a single job script with
   ``SimulationContext`` as a python context manager.
-  ``util.quiet_status() / util.unquiet_status()`` allow users to
   control if line status messages are output.
-  Support executing hoomd in Jupyter (ipython) notebooks. Notice,
   warning, and error messages now show up in the notebook output
   blocks.
-  ``analyze.log`` can now register python callback functions as sources
   for logged quantities.
-  The GSD file format (http://gsd.readthedocs.io) is fully implemented
   in hoomd

   -  ``dump.gsd`` writes GSD trajectories and restart files (use
      ``truncate=true`` for restarts).
   -  ``init.read_gsd`` reads GSD file and initializes the system, and
      can start the simulation from any frame in the GSD file.
   -  ``data.gsd_snapshot`` reads a GSD file into a snapshot which can
      be modified before system initialization with
      ``init.read_snapshot``.
   -  The GSD file format is capable of storing all particle and
      topology data fields in hoomd, either static at frame 0, or
      varying over the course of the trajectory. The number of
      particles, types, bonds, etc. can also vary over the trajectory.

-  ``force.active`` applies an active force (optionally with rotational
   diffusion) to a group of particles
-  ``update.constrain_ellipsoid`` constrains particles to an ellipsoid
-  ``integrate.langevin`` and ``integrate.brownian`` now apply
   rotational noise and damping to anisotropic particles
-  Support dynamically updating groups. ``group.force_update()`` forces
   the group to rebuild according to the original selection criteria.
   For example, this can be used to periodically update a cuboid group
   to include particles only in the specified region.
-  ``pair.reaction_field`` implements a pair force for a screened
   electrostatic interaction of a charge pair in a dielectric medium.
-  ``force.get_energy`` allows querying the potential energy of a
   particle group for a specific force
-  ``init.create_lattice`` initializes particles on a lattice.

   -  ``lattice.unitcell`` provides a generic unit cell definition for
      ``create_lattice``
   -  Convenience functions for common lattices: sq, hex, sc, bcc, fcc.

-  Dump and initialize commands for the GTAR file format
   (http://libgetar.readthedocs.io).

   -  GTAR can store trajectory data in zip, tar, sqlite, or bare
      directories
   -  The current version stores system properties, later versions will
      be able to capture log, metadata, and other output to reduce the
      number of files that a job script produces.

-  ``integrate.npt`` can now apply a constant stress tensor to the
   simulation box.
-  Faceted shapes can now be simulated through the ``dem`` component.

*Changes that require job script modifications*

-  ``context.initialize()`` is now required before any other hoomd
   script command.
-  ``init.reset()`` no longer exists. Use ``context.initialize()`` or
   activate a ``SimulationContext``.
-  Any scripts that relied on undocumented members of the ``globals``
   module will fail. These variables have been moved to the ``context``
   module and members of the currently active ``SimulationContext``.
-  bonds, angles, dihedrals, and impropers no longer use the
   ``set_coeff`` syntax. Use ``bond_coeff.set``, ``angle_coeff.set``,
   ``dihedral_coeff.set``, and ``improper_coeff.set`` instead.
-  ``hoomd_script`` no longer exists, python commands are now spread
   across ``hoomd``, ``hoomd.md``, and other sub packages.
-  ``integrate.\*_rigid()`` no longer exists. Use a standard integrator
   on ``group.rigid_center()``, and define rigid bodies using
   ``constrain.rigid()``
-  All neighbor lists must be explicitly created using ``nlist.\*``, and
   each pair potential must be attached explicitly to a neighbor list. A
   default global neighbor list is no longer created.
-  Moved cgcmm into its own package.
-  Moved eam into the metal package.
-  Integrators now take ``kT`` arguments for temperature instead of
   ``T`` to avoid confusion on the units of temperature.
-  phase defaults to 0 for updaters and analyzers so that restartable
   jobs are more easily enabled by default.
-  ``dump.xml`` (deprecated) requires a particle group, and can dump
   subsets of particles.

*Other changes*

-  CMake minimum version is now 2.8
-  Convert particle type names to ``str`` to allow unicode type name
   input
-  ``__version__`` is now available in the top level package
-  ``boost::iostreams`` is no longer a build dependency
-  ``boost::filesystem`` is no longer a build dependency
-  New concepts page explaining the different styles of neighbor lists
-  Default neighbor list buffer radius is more clearly shown to be
   r_buff = 0.4
-  Memory usage of ``nlist.stencil`` is significantly reduced
-  A C++11 compliant compiler is now required to build HOOMD-blue

*Removed*

-  Removed ``integrate.bdnvt``: use ``integrate.langevin``
-  Removed ``mtk=False`` option from ``integrate.nvt`` - The MTK NVT
   integrator is now the only implementation.
-  Removed ``integrate.\*_rigid()``: rigid body functionality is now
   contained in the standard integration methods
-  Removed the global neighbor list, and thin wrappers to the neighbor
   list in ``nlist``.
-  Removed PDB and MOL2 dump writers.
-  Removed init.create_empty

*Deprecated*

-  Deprecated analyze.msd.
-  Deprecated dump.xml.
-  Deprecated dump.pos.
-  Deprecated init.read_xml.
-  Deprecated init.create_random.
-  Deprecated init.create_random_polymers.

v1.x
----

v1.3.3 (2016-03-06)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Fix problem incluing ``hoomd.h`` in plugins
-  Fix random memory errors when using walls

v1.3.2 (2016-02-08)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Fix wrong access to system.box
-  Fix kinetic energy logging in MPI
-  Fix particle out of box error if particles are initialized on the
   boundary in MPI
-  Add integrate.brownian to the documentation index
-  Fix misc doc typos
-  Fix runtime errors with boost 1.60.0
-  Fix corrupt metadata dumps in MPI runs

v1.3.1 (2016-1-14)
^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Fix invalid MPI communicator error with Intel MPI
-  Fix python 3.5.1 seg fault

v1.3.0 (2015-12-8)
^^^^^^^^^^^^^^^^^^

*New features*

-  Automatically load balanced domain decomposition simulations.
-  Anisotropic particle integrators.
-  Gay-Berne pair potential.
-  Dipole pair potential.
-  Brownian dynamics ``integrate.brownian``
-  Langevin dynamics ``integrate.langevin`` (formerly ``bdnvt``)
-  ``nlist.stencil`` to compute neighbor lists using stencilled cell
   lists.
-  Add single value scale, ``min_image``, and ``make_fraction`` to
   ``data.boxdim``
-  ``analyze.log`` can optionally not write a file and now supports
   querying current quantity values.
-  Rewritten wall potentials.

   -  Walls are now sums of planar, cylindrical, and spherical
      half-spaces.
   -  Walls are defined and can be modified in job scripts.
   -  Walls execute on the GPU.
   -  Walls support per type interaction parameters.
   -  Implemented for: lj, gauss, slj, yukawa, morse, force_shifted_lj,
      and mie potentials.

-  External electric field potential: ``external.e_field``

*Bug fixes*

-  Fixed a bug where NVT integration hung when there were 0 particles in
   some domains.
-  Check SLURM environment variables for local MPI rank identification
-  Fixed a typo in the box math documentation
-  Fixed a bug where exceptions weren’t properly passed up to the user
   script
-  Fixed a bug in the velocity initialization example
-  Fixed an openmpi fork() warning on some systems
-  Fixed segfaults in PPPM
-  Fixed a bug where compute.thermo failed after reinitializing a system
-  Support list and dict-like objects in init.create_random_polymers.
-  Fall back to global rank to assign GPUs if local rank is not
   available

*Deprecated commands*

-  ``integrate.bdnvt`` is deprecated. Use ``integrate.langevin``
   instead.
-  ``dump.bin`` and ``init.bin`` are now removed. Use XML files for
   restartable jobs.

*Changes that may break existing scripts*

-  ``boxdim.wrap`` now returns the position and image in a tuple, where
   it used to return just the position.
-  ``wall.lj`` has a new API
-  ``dump.bin`` and ``init.bin`` have been removed.

v1.2.1 (2015-10-22)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Fix a crash when adding or removing particles and reinitializing
-  Fix a bug where simulations hung on sm 5.x GPUs with CUDA 7.5
-  Fix compile error with long tests enabled
-  Issue a warning instead of an error for memory allocations greater
   than 4 GiB.
-  Fix invalid RPATH when building inside ``zsh``.
-  Fix incorrect simulations with ``integrate.npt_rigid``
-  Label mie potential correctly in user documentation

v1.2.0 (2015-09-30)
^^^^^^^^^^^^^^^^^^^

*New features*

-  Performance improvements for systems with large particle size
   disparity
-  Bounding volume hierarchy (tree) neighbor list computation
-  Neighbor lists have separate ``r_cut`` values for each pair of types
-  addInfo callback for dump.pos allows user specified information in
   pos files

*Bug fixes*

-  Fix ``test_pair_set_energy`` unit test, which failed on numpy < 1.9.0
-  Analyze.log now accepts unicode strings.
-  Fixed a bug where calling ``restore_snapshot()`` during a run zeroed
   potential parameters.
-  Fix segfault on exit with python 3.4
-  Add ``cite.save()`` to documentation
-  Fix a problem were bond forces are computed incorrectly in some MPI
   configurations
-  Fix bug in pair.zbl
-  Add pair.zbl to the documentation
-  Use ``HOOMD_PYTHON_LIBRARY`` to avoid problems with modified CMake
   builds that preset ``PYTHON_LIBRARY``

v1.1.1 (2015-07-21)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  ``dump.xml(restart=True)`` now works with MPI execution
-  Added missing documentation for ``meta.dump_metadata``
-  Build all unit tests by default
-  Run all script unit tests through ``mpirun -n 1``

v1.1.0 (2015-07-14)
^^^^^^^^^^^^^^^^^^^

*New features*

-  Allow builds with ninja.
-  Allow K=0 FENE bonds.
-  Allow number of particles types to change after initialization.

   .. code::

       system.particles.types.add('newtype')

-  Allow number of particles to change after initialization.

   .. code::

       system.particles.add(‘A’)
       del system.particles[0]

-  OPLS dihedral
-  Add ``phase`` keyword to analyzers and dumps to make restartable jobs easier.
-  ``HOOMD_WALLTIME_STOP`` environment variable to stop simulation runs before they hit a wall clock limit.
-  ``init.read_xml()`` Now accepts an initialization and restart file.
-  ``dump.xml()`` can now write restart files.
-   Added documentation concepts page on writing restartable jobs.
-   New citation management infrastructure. ``cite.save()`` writes ``.bib`` files with a list of references to
    features actively used in the current job script.
-   Snapshots expose data as numpy arrays for high performance access to particle properties.
-  ``data.make_snapshot()`` makes a new empty snapshot.
-  ``analyze.callback()`` allows multiple python callbacks to operate at different periods.
-  ``comm.barrier()``and`` comm.barrier_all()``allow users to insert barriers into their scripts.
-   Mie pair potential.
-  ``meta.dump_metadata()`` writes job metadata information out to a json file.
-  ``context.initialize()`` initializes the execution context.
-  Restart option for ``dump.xml()``

*Bug fixes*

-  Fix slow performance when initializing ``pair.slj()``\ in MPI runs.
-  Properly update particle image when setting position from python.
-  PYTHON_SITEDIR hoomd shell launcher now calls the python interpreter
   used at build time.
-  Fix compile error on older gcc versions.
-  Fix a bug where rigid bodies had 0 velocity when restarting jobs.
-  Enable ``-march=native`` builds in OS X clang builds.
-  Fix ``group.rigid()`` and ``group.nonrigid()``.
-  Fix image access from the python data access proxies.
-  Gracefully exit when launching MPI jobs with mixed execution
   configurations.

*Changes that may require updated job scripts*

-  ``context.initialize()`` **must** be called before any ``comm``
   method that queries the MPI rank. Call it as early as possible in
   your job script (right after importing ``hoomd_script``) to avoid
   problems.

*Deprecated*

-  ``init.create_empty()`` is deprecated and will be removed in a future
   version. Use ``data.make_snapshot()`` and ``init.read_snapshot()``
   instead.
-  Job scripts that do not call ``context.initialize()`` will result in
   a warning message. A future version of HOOMD will require that you
   call ``context.initialize()``.

*Removed*

-  Several ``option`` commands for controlling the execution
   configuration. Replaced with ``context.initialize``.

v1.0.5 (2015-05-19)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Fix segfault when changing integrators
-  Fix system.box to indicate the correct number of dimensions
-  Fix syntax error in comm.get_rank with –nrank
-  Enable CUDA enabled builds with the intel compiler
-  Use CMake builtin FindCUDA on recent versions of CMake
-  GCC_ARCH env var sets the -march command line option to gcc at
   configure time
-  Auto-assign GPU-ids on non-compute exclusive systems even with
   –mode=gpu
-  Support python 3.5 alpha
-  Fix a bug where particle types were doubled with boost 1.58.0
-  Fix a bug where angle_z=true dcd output was inaccurate near 0 angles
-  Properly handle lj.wall potentials with epsilon=0.0 and particles on
   top of the walls

v1.0.4 (2015-04-07)
^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Fix invalid virials computed in rigid body simulations when
   multi-particle bodies crossed box boundaries
-  Fix invalid forces/torques for rigid body simulations caused by race
   conditions
-  Fix compile errors on Mac OS X 10.10
-  Fix invalid pair force computations caused by race conditions
-  Fix invalid neighbour list computations caused by race conditions on
   Fermi generation GPUs

*Other*

-  Extremely long running unit tests are now off by default. Enable with
   -DHOOMD_SKIP_LONG_TESTS=OFF
-  Add additional tests to detect race conditions and memory errors in
   kernels

v1.0.3 (2015-03-18)
^^^^^^^^^^^^^^^^^^^

**Bug fixes**

-  Enable builds with intel MPI
-  Silence warnings coming from boost and python headers

v1.0.2 (2015-01-21)
^^^^^^^^^^^^^^^^^^^

**Bug fixes**

-  Fixed a bug where ``linear_interp`` would not take a floating point
   value for *zero*
-  Provide more useful error messages when cuda drivers are not present
-  Assume device count is 0 when ``cudaGetDeviceCount()`` returns an
   error
-  Link to python statically when ``ENABLE_STATIC=on``
-  Misc documentation updates

v1.0.1 (2014-09-09)
^^^^^^^^^^^^^^^^^^^

**Bug fixes**

1.  Fixed bug where error messages were truncated and HOOMD exited with
    a segmentation fault instead (e.g. on Blue Waters)
2.  Fixed bug where plug-ins did not load on Blue Waters
3.  Fixed compile error with gcc4.4 and cuda5.0
4.  Fixed syntax error in ``read_snapshot()``
5.  Fixed a bug where ``init.read_xml throwing`` an error (or any other
    command outside of ``run()``) would hang in MPI runs
6.  Search the install path for hoomd_script - enable the hoomd
    executable to be outside of the install tree (useful with cray
    aprun)
7.  Fixed CMake 3.0 warnings
8.  Removed dependancy on tr1/random
9.  Fixed a bug where ``analyze.msd`` ignored images in the r0_file
10. Fixed typos in ``pair.gauss`` documentation
11. Fixed compile errors on Ubuntu 12.10
12. Fix failure of ``integrate.nvt`` to reach target temperature in
    analyze.log. The fix is a new symplectic MTK integrate.nvt
    integrator. Simulation results in hoomd v1.0.0 are correct, just the
    temperature and velocity outputs are off slightly.
13. Remove MPI from Mac OS X dmg build.
14. Enable ``import hoomd_script as ...``

*Other changes*

1. Added default compile flag -march=native
2. Support CUDA 6.5
3. Binary builds for CentOS/RHEL 6, Fedora 20, Ubuntu 14.04 LTS, and
   Ubuntu 12.04 LTS.

Version 1.0.0 (2014-05-25)
^^^^^^^^^^^^^^^^^^^^^^^^^^

*New features*

-  Support for python 3
-  New NPT integrator capable of flexible coupling schemes
-  Triclinic unit cell support
-  MPI domain decomposition
-  Snapshot save/restore
-  Autotune block sizes at run time
-  Improve performance in small simulation boxes
-  Improve performance with smaller numbers of particles per GPU
-  Full double precision computations on the GPU (compile time option
   must be enabled, binary builds provided on the download page are
   single precision)
-  Tabulated bond potential ``bond.table``
-  Tabulated angle potential ``angle.table``
-  Tabulated dihedral potental ``dihedral.table``
-  ``update.box_resize`` now accepts ``period=None`` to trigger an
   immediate update of the box without creating a periodic updater
-  ``update.box_resize`` now replaces *None* arguments with the current
   box parameters
-  ``init.create_random`` and ``init.create_random_polymers`` can now
   create random configurations in triclinc and 2D boxes
-  ``init.create_empty`` can now create triclinic boxes
-  particle, bond, angle, dihedral, and impropers types can now be named
   in ``init.create_empty``
-  ``system.replicate`` command replicates the simulation box

*Bug fixes*

-  Fixed a bug where init.create_random_polymers failed when lx,ly,lz
   were not equal.
-  Fixed a bug in init.create_random_polymers and init.create_random
   where the separation radius was not accounted for correctly
-  Fixed a bug in bond.\* where random crashes would occur when more
   than one bond type was defined
-  Fixed a bug where dump.dcd did not write the period to the file

*Changes that may require updated job scripts*

-  ``integrate.nph``: A time scale ``tau_p`` for the relaxation of the
   barostat is now required instead of the barostat mass *W* of the
   previous release. The time scale is the relaxation time the barostat
   would have at an average temperature ``T_0 = 1``, and it is related
   to the internally used (Andersen) Barostat mass *W* via
   ``W = d N T_0 tau_p^2``, where *d* is the dimensionsality and *N* the
   number of particles.
-  ``sorter`` and ``nlist`` are now modules, not variables in the
   ``__main__`` namespace.
-  Data proxies function correctly in MPI simulations, but are extremely
   slow. If you use ``init.create_empty``, consider separating the
   generation step out to a single rank short execution that writes an
   XML file for the main run.
-  ``update.box_resize(Lx=...)`` no longer makes cubic box updates,
   instead it will keep the current **Ly** and **Lz**. Use the ``L=...``
   shorthand for cubic box updates.
-  All ``init.*`` commands now take ``data.boxdim`` objects, instead of
   ``hoomd.boxdim`` (or *3-tuples*). We strongly encourage the use of
   explicit argument names for ``data.boxdim()``. In particular, if
   ``hoomd.boxdim(123)`` was previously used to create a cubic box, it
   is now required to use ``data.boxdim(L=123)`` (CORRECT) instead of
   ``data.boxdim(123)`` (INCORRECT), otherwise a box with unit
   dimensions along the y and z axes will be created.
-  ``system.dimensions`` can no longer be set after initialization.
   System dimensions are now set during initialization via the
   ``data.boxdim`` interface. The dimensionality of the system can now
   be queried through ``system.box``.
-  ``system.box`` no longer accepts 3-tuples. It takes ``data.boxdim``
   objects.
-  ``system.dimensions`` no longer exists. Query the dimensionality of
   the system from ``system.box``. Set the dimensionality of the system
   by passing an appropriate ``data.boxdim`` to an ``init`` method.
-  ``init.create_empty`` no longer accepts ``n_*_types``. Instead, it
   now takes a list of strings to name the types.

*Deprecated*

-  Support for G80, G200 GPUs.
-  ``dump.bin`` and ``read.bin``. These will be removed in v1.1 and
   replaced with a new binary format.

*Removed*

-  OpenMP mult-core execution (replaced with MPI domain decomposition)
-  ``tune.find_optimal_block_size`` (replaced by Autotuner)

v0.x
----

Version 0.11.3 (2013-05-10)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

*Bug fixes*

-  Fixed a bug where charge.pppm could not be used after init.reset()
-  Data proxies can now set body angular momentum before the first run()
-  Fixed a bug where PPPM forces were incorrect on the GPU

Version 0.11.2 (2012-12-19)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

*New features*

-  Block sizes tuned for K20

*Bug fixes*

-  Warn user that PPPM ignores rigid body exclusions
-  Document that proxy iterators need to be deleted before init.reset()
-  Fixed a bug where body angular momentum could not be set
-  Fixed a bug where analyze.log would report nan for the pressure
   tensor in nve and nvt simulations

Version 0.11.1 (2012-11-2)
^^^^^^^^^^^^^^^^^^^^^^^^^^

*New features*

-  Support for CUDA 5.0
-  Binary builds for Fedora 16 and OpenSUSE 12.1
-  Automatically specify /usr/bin/gcc to nvcc when the configured gcc is
   not supported

*Bug fixes*

-  Fixed a compile error with gcc 4.7
-  Fixed a bug where PPPM forces were incorrect with neighborlist
   exclusions
-  Fixed an issue where boost 1.50 and newer were not detected properly
   when BOOST_ROOT is set
-  Fixed a bug where accessing force data in python prevented
   init.reset() from working
-  Fixed a bug that prevented pair.external from logging energy
-  Fixed a unit test that failed randomly

Version 0.11.0 (2012-07-27)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

*New features*

1.  Support for Kepler GPUs (GTX 680)
2.  NPH integration (*integrate.nph*)
3.  Compute full pressure tensor
4.  Example plugin for new bond potentials
5.  New syntax for bond coefficients: *bond.bond_coeff.set(‘type’,
    params)*
6.  New external potential: *external.periodic* applies a periodic
    potential along one direction (uses include inducing lamellar phases
    in copolymer systems)
7.  Significant performance increases when running *analyze.log*,
    *analyze.msd*, *update.box_resize*, *update.rescale_temp*, or
    *update.zero_momentum* with a small period
8.  Command line options may now be overwritten by scripts, ex:
    *options.set_gpu(2)*
9.  Added *–user* command line option to allow user defined options to
    be passed into job scripts, ex: *–user=“-N=5 -phi=0.56”*
10. Added *table.set_from_file* method to enable reading table based
    pair potentials from a file
11. Added *–notice-level* command line option to control how much extra
    information is printed during a run. Set to 0 to disable, or any
    value up to 10. At 10, verbose debugging information is printed.
12. Added *–msg-file* command line option which redirects the message
    output to a file
13. New pair potential *pair.force_shifted_lj* : Implements
    http://dx.doi.org/10.1063/1.3558787

*Bug fixes*

1. Fixed a bug where FENE bonds were sometimes computed incorrectly
2. Fixed a bug where pressure was computed incorrectly when using
   pair.dpd or pair.dpdlj
3. Fixed a bug where using OpenMP and CUDA at the same time caused
   invalid memory accesses
4. Fixed a bug where RPM packages did not work on systems where the CUDA
   toolkit was not installed
5. Fixed a bug where rigid body velocities were not set from python
6. Disabled OpenMP builds on Mac OS X. HOOMD-blue w/ openmp enabled
   crashes due to bugs in Apple’s OpenMP implementation.
7. Fixed a bug that allowed users to provide invalid rigid body data and
   cause a seg fault.
8. Fixed a bug where using PPPM resulted in error messages on program
   exit.

*API changes*

1.  Bond potentials rewritten with template evaluators
2.  External potentials use template evaluators
3.  Complete rewrite of ParticleData - may break existing plugins
4.  Bond/Angle/Dihedral data structures rewritten

    -  The GPU specific data structures are now generated on the GPU

5.  DPDThermo and DPDLJThermo are now processed by the same template
    class
6.  Headers that cannot be included by nvcc now throw an error when they
    are
7.  CUDA 4.0 is the new minimum requirement
8.  Rewrote BoxDim to internally handle minimum image conventions
9.  HOOMD now only compiles ptx code for the newest architecture, this
    halves the executable file size
10. New Messenger class for global control of messages printed to the
    screen / directed to a file.

*Testing changes*

1. Automated test suite now performs tests on OpenMPI + CUDA builds
2. Valgrind tests added back into automated test suite
3. Added CPU test in bd_ridid_updater_tests
4. ctest -S scripts can now set parallel makes (with cmake > 2.8.2)

Version 0.10.1 (2012-02-10)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Add missing entries to credits page
2. Add ``dist_check`` option to neighbor list. Can be used to force
   neighbor list builds at a specified frequency (useful in profiling
   runs with nvvp).
3. Fix typos in ubuntu compile documentation
4. Add missing header files to hoomd.h
5. Add torque to the python particle data access API
6. Support boost::filesystem API v3
7. Expose name of executing gpu, n_cpu, hoomd version, git sha1, cuda
   version, and compiler version to python
8. Fix a bug where multiple ``nvt_rigid`` or ``npt_rigid`` integrators
   didn’t work correctly
9. Fix missing pages in developer documentation

Version 0.10.0 (2011-12-14)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

*New features*

1.  Added *pair.dpdlj* which uses the DPD thermostat and the
    Lennard-Jones potential. In previous versions, this could be
    accomplished by using two pair commands but at the cost of reduced
    performance.
2.  Additional example scripts are now present in the documentation. The
    example scripts are cross-linked to the commands that are used in
    them.
3.  Most dump commands now accept the form:
    *dump.ext(filename=“filename.ext”)* which immediately writes out
    filename.ext.
4.  Added *vis* parameter to dump.xml which enables output options
    commonly used in files written for the purposes of visulization.
    dump.xml also now accepts parameters on the instantiation line.
    Combined with the previous feature, *dump.xml(filename=“file.xml”,
    vis=True)* is now a convenient short hand for what was previously

    .. code::

       xml = dump.xml()
       xml.set_params(position = True, mass = True, diameter = True,
                             type = True, bond = True, angle = True,
                             dihedral = True, improper = True, charge = True)
       xml.write(filename="file.xml")

5.  Specify rigid bodies in XML input files
6.  Simulations that contain rigid body constraints applied to groups of
    particles in BDNVT, NVE, NVT, and NPT ensembles.

    -  *integrate.bdnvt_rigid*
    -  *integrate.nve_rigid*
    -  *integrate.nvt_rigid*
    -  *integrate.npt_rigid*

7.  Energy minimization of rigid bodies
    (*integrate.mode_minimize_rigid_fire*)
8.  Existing commands are now rigid-body aware

    -  update.rescale_temp
    -  update.box_resize
    -  update.enforce2d
    -  update.zero_momentum

9.  NVT integration using the Berendsen thermostat
    (*integrate.berendsen*)
10. Bonds, angles, dihedrals, and impropers can now be created and
    deleted with the python data access API.
11. Attribution clauses added to the HOOMD-blue license.

*Changes that may break existing job scripts*

1. The *wrap* option to *dump.dcd* has been changed to *unwrap_full* and
   its meaning inverted. *dump.dcd* now offers two options for
   unwrapping particles, *unwrap_full* fully unwraps particles into
   their box image and *unwrap_rigid* unwraps particles in rigid bodies
   so that bodies are not broken up across a box boundary.

*Bug/fixes small enhancements*

1.  Fixed a bug where launching hoomd on mac os X 10.5 always resulted
    in a bus error.
2.  Fixed a bug where DCD output restricted to a group saved incorrect
    data.
3.  force.constant may now be applied to a group of particles, not just
    all particles
4.  Added C++ plugin example that demonstrates how to add a pair
    potential in a plugin
5.  Fixed a bug where box.resize would always transfer particle data
    even in a flat portion of the variant
6.  OpenMP builds re-enabled on Mac OS X
7.  Initial state of integrate.nvt and integrate.npt changed to decrease
    oscillations at startup.
8.  Fixed a bug where the polymer generator would fail to initialize
    very long polymers
9.  Fixed a bug where images were passed to python as unsigned ints.
10. Fixed a bug where dump.pdb wrote coordinates in the wrong order.
11. Fixed a rare problem where a file written by dump.xml would not be
    read by init.read_xml due to round-off errors.
12. Increased the number of significant digits written out to dump.xml
    to make them more useful for ad-hoc restart files.
13. Potential energy and pressure computations that slow performance are
    now only performed on those steps where the values are actually
    needed.
14. Fixed a typo in the example C++ plugin
15. Mac build instructions updated to work with the latest version of
    macports
16. Fixed a bug where set_period on any dump was ineffective.
17. print_status_line now handles multiple lines
18. Fixed a bug where using bdnvt tally with per type gammas resulted in
    a race condition.
19. Fix an issue where ENABLE_CUDA=off builds gave nonsense errors when
    –mode=gpu was requested.
20. Fixed a bug where dumpl.xml could produce files that init.xml would
    not read
21. Fixed a typo in the example plugin
22. Fix example that uses hoomd as a library so that it compiles.
23. Update maintainer lines
24. Added message to nlist exclusions that notifies if diameter or body
    exclusions are set.
25. HOOMD-blue is now hosted in a git repository
26. Added bibtex bibliography to the user documentation
27. Converted user documentation examples to use doxygen auto
    cross-referencing ``\example`` commands
28. Fix a bug where particle data is not released in dump.binary
29. ENABLE_OPENMP can now be set in the ctest builds
30. Tuned block sizes for CUDA 4.0
31. Removed unsupported GPUS from CUDA_ARCH_LIST

Version 0.9.2 (2011-04-04)
^^^^^^^^^^^^^^^^^^^^^^^^^^

*Note:* only major changes are listed here.

*New features*

1. *New exclusion option:* Particles can now be excluded from the
   neighbor list based on diameter consistent with pair.slj.
2. *New pair coeff syntax:* Coefficients for multiple type pairs can be
   specified conveniently on a single line.

   .. code::

      coeff.set(['A', 'B', 'C', 'D'], ['A', 'B', 'C', 'D'], epsilon=1.0)

3. *New documentation:* HOOMD-blue’s system of units is now fully
   documented, and every coefficient in the documentation is labeled
   with the appropriate unit.
4. *Performance improvements:* Performance has been significantly
   boosted for simulations of medium sized systems (5,000-20,000
   particles). Smaller performance boosts were made to larger runs.
5. *CUDA 3.2 support:* HOOMD-blue is now fully tested and performance
   tuned for use with CUDA 3.2.
6. *CUDA 4.0 support:* HOOMD-blue compiles with CUDA 4.0 and passes
   initial tests.
7. *New command:* tune.r_buff performs detailed auto-tuning of the
   r_buff neighborlist parameter.
8. *New installation method:* RPM, DEB, and app bundle packages are now
   built for easier installation
9. *New command:* charge.pppm computes the full long range electrostatic
   interaction using the PPPM method

*Bug/fixes small enhancements*

1.  Fixed a bug where the python library was linked statically.
2.  Added the PYTHON_SITEDIR setting to allow hoomd builds to install
    into the native python site directory.
3.  FIRE energy minimization convergence criteria changed to require
    both energy *and* force to converge
4.  Clarified that groups are static in the documentation
5.  Updated doc comments for compatibility with Doxygen#7.3
6.  system.particles.types now lists the particle types in the
    simulation
7.  Creating a group of a non-existant type is no longer an error
8.  Mention XML file format for walls in wall.lj documentation
9.  Analyzers now profile themselves
10. Use ``\n`` for newlines in dump.xml - improves
    performance when writing many XML files on a NFS file system
11. Fixed a bug where the neighbor list build could take an
    exceptionally long time (several seconds) to complete the first
    build.
12. Fixed a bug where certain logged quantities always reported as 0 on
    the first step of the simulation.
13. system.box can now be used to read and set the simulation box size
    from python
14. Numerous internal API updates
15. Fixed a bug the resulted in incorrect behavior when using
    integrate.npt on the GPU.
16. Removed hoomd launcher shell script. In non-sitedir installs,
    ${HOOMD_ROOT}/bin/hoomd is now the executable itself
17. Creating unions of groups of non-existent types no longer produces a
    seg fault
18. hoomd now builds on all cuda architectures. Modify CUDA_ARCH_LIST in
    cmake to add or remove architectures from the build
19. hoomd now builds with boost#46.0
20. Updated hoomd icons to maize/blue color scheme
21. hoomd xml file format bumped to#3, adds support for charge.
22. FENE and harmonic bonds now handle 0 interaction parameters and 0
    length bonds more gracefully
23. The packaged plugin template now actually builds and installs into a
    recent build of hoomd

Version 0.9.1 (2010-10-08)
^^^^^^^^^^^^^^^^^^^^^^^^^^

*Note:* only major changes are listed here.

*New features*

1. *New constraint*: constrain.sphere constrains a group of particles to
   the surface of a sphere
2. *New pair potential/thermostat*: pair.dpd implements the standard DPD
   conservative, random, and dissipative forces
3. *New pair potential*: pair.dpd_conservative applies just the
   conservative DPD potential
4. *New pair potential*: pair.eam implements the Embedded Atom Method
   (EAM) and supports both *alloy* and *FS* type computations.
5. *Faster performance*: Cell list and neighbor list code has been
   rewritten for performance.

   -  In our benchmarks, *performance increases* ranged from *10-50%*
      over HOOMD-blue 0.9.0. Simulations with shorter cutoffs tend to
      attain a higher performance boost than those with longer cutoffs.
   -  We recommended that you *re-tune r_buff* values for optimal
      performance with 0.9.1.
   -  Due to the nature of the changes, *identical runs* may produce
      *different trajectories*.

6. *Removed limitation*: The limit on the number of neighbor list
   exclusions per particle has been removed. Any number of exclusions
   can now be added per particle. Expect reduced performance when adding
   excessive numbers of exclusions.

*Bug/fixes small enhancements*

1.  Pressure computation is now correct when constraints are applied.
2.  Removed missing files from hoomd.h
3.  pair.yukawa is no longer referred to by “gaussian” in the
    documentation
4.  Fermi GPUs are now prioritized over per-Fermi GPUs in systems where
    both are present
5.  HOOMD now compiles against CUDA 3.1
6.  Momentum conservation significantly improved on compute#x hardware
7.  hoomd plugins can now be installed into user specified directories
8.  Setting r_buff=0 no longer triggers exclusion list updates on every
    step
9.  CUDA 2.2 and older are no longer supported
10. Workaround for compiler bug in 3.1 that produces extremely high
    register usage
11. Disabled OpenMP compile checks on Mac OS X
12. Support for compute 2.1 devices (such as the GTX 460)

Version 0.9.0 (2010-05-18)
^^^^^^^^^^^^^^^^^^^^^^^^^^

*Note:* only major changes are listed here.

*New features*

1.  *New pair potential*: Shifted LJ potential for particles of varying
    diameters (pair.slj)
2.  *New pair potential*: Tabulated pair potential (pair.table)
3.  *New pair potential*: Yukawa potential (pair.yukawa)
4.  *Update to pair potentials*: Most pair potentials can now accept
    different values of r_cut for different type pairs. The r_cut
    specified in the initial pair.**\* command is now treated as the
    default r_cut, so no changes to scripts are necessary.
5.  *Update to pair potentials*: Default pair coeff values are now
    supported. The parameter alpha for lj now defaults to#0, so there is
    no longer a need to specify it for a majority of simulations.
6.  *Update to pair potentials*: The maximum r_cut needed for the
    neighbor list is now determined at the start of each run(). In
    simulations where r_cut may decrease over time, increased
    performance will result.
7.  *Update to pair potentials*: Pair potentials are now specified via
    template evaluator classes. Adding a new pair potential to hoomd now
    only requires a small amount of additional code.
8.  *Plugin API* : Advanced users/developers can now write, install, and
    use plugins for hoomd without needing to modify core hoomd source
    code
9.  *Particle data access*: User-level hoomd scripts can now directly
    access the particle data. For example, one can change all particles
    in the top half of the box to be type B:

    .. code::

       top = group.cuboid(name="top", zmin=0)
       for p in top:
           p.type = 'B'

    . *All* particle data including position, velocity, type, ‘’et
    cetera’’, can be read and written in this manner. Computed forces
    and energies can also be accessed in a similar way.
10. *New script command*: init.create_empty() can be used in conjunction
    with the particle data access above to completely initialize a
    system within the hoomd script.
11. *New script command*: dump.bin() writes full binary restart files
    with the entire system state, including the internal state of
    integrators.

    -  File output can be gzip compressed (if zlib is available) to save
       space
    -  Output can alternate between two different output files for safe
       crash recovery

12. *New script command*: init.read_bin() reads restart files written by
    dump.bin()
13. *New option*: run() now accepts a quiet option. When True, it
    eliminates the status information printouts that go to stdout.
14. *New example script*: Example 6 demonstrates the use of the particle
    data access routines to initialize a system. It also demonstrates
    how to initialize velocities from a gaussian distribution
15. *New example script*: Example 7 plots the pair.lj potential energy
    and force as evaluated by hoomd. It can trivially be modified to
    plot any potential in hoomd.
16. *New feature*: Two dimensional simulations can now be run in hoomd:
    #259
17. *New pair potential*: Morse potential for particles of varying
    diameters (pair.morse)
18. *New command*: run_upto will run a simulation up to a given time
    step number (handy for breaking long simulations up into many
    independent jobs)
19. *New feature*: HOOMD on the CPU is now accelerated with OpenMP.
20. *New feature*: integrate.mode_minimize_fire performs energy
    minimization using the FIRE algorithm
21. *New feature*: analyze.msd can now accept an xml file specifying the
    initial particle positions (for restarting jobs)
22. *Improved feature*: analyze.imd now supports all IMD commands that
    VMD sends (pause, kill, change trate, etc.)
23. *New feature*: Pair potentials can now be given names, allowing
    multiple potentials of the same type to be logged separately.
    Additionally, potentials that are disabled and not applied to the
    system dynamics can be optionally logged.
24. *Performance improvements*: Simulation performance has been
    increased across the board, but especially when running systems with
    very low particle number densities.
25. *New hardware support*: 0.9.0 and newer support Fermi GPUs
26. *Deprecated hardware support*: 0.9.x might continue run on compute#1
    GPUs but that hardware is no longer officially supported
27. *New script command*: group.tag_list() takes a python list of
    particle tags and creates a group
28. *New script command*: compute.thermo() computes thermodynamic
    properties of a group of particles for logging
29. *New feature*: dump.dcd can now optionally write out only those
    particles that belong to a specified group

*Changes that will break jobs scripts written for 0.8.x*

1. Integration routines have changed significantly to enable new use
   cases. Where scripts previously had commands like:

   .. code::

      integrate.nve(dt=0.005)

   they now need

   .. code::

      all = group.all()
      integrate.mode_standard(dt=0.005)
      integrate.nve(group=all)

   . Integrating only specific groups of particles enables simulations
   to fix certain particles in place or integrate different parts of the
   system at different temperatures, among many other possibilities.
2. sorter.set_params no longer takes the ‘’bin_width’’ argument. It is
   replaced by a new ‘’grid’’ argument, see the documentation for
   details.
3. conserved_quantity is no longer a quantity available for logging.
   Instead log the nvt reservoir energy and compute the total conserved
   quantity in post processing.

*Bug/fixes small enhancements*

1.  Fixed a bug where boost#38 is not found on some machines
2.  dump.xml now has an option to write particle accelerations
3.  Fixed a bug where periods like 1e6 were not accepted by updaters
4.  Fixed a bug where bond.fene forces were calculated incorrectly
    between particles of differing diameters
5.  Fixed a bug where bond.fene energies were computed incorrectly when
    running on the GPU
6.  Fixed a bug where comments in hoomd xml files were not ignored as
    they aught to be: #331
7.  It is now possible to prevent bond exclusions from ever being added
    to the neighbor list: #338
8.  init.create_random_polymers can now generate extremely dense systems
    and will warn the user about large memory usage
9.  variant.linear_interp now accepts a user-defined zero (handy for
    breaking long simulations up into many independent jobs)
10. Improved installation and compilation documentation
11. Integration methods now silently ignore when they are given an empty
    group
12. Fixed a bug where disabling all forces resulted in some forces still
    being applied
13. Integrators now behave in a reasonable way when given empty groups
14. Analyzers now accept a floating point period
15. run() now aborts immediately if limit_hours=0 is specified.
16. Pair potentials that diverge at r=0 will no longer result in invalid
    simulations when the leading coefficients are set to zero.
17. integrate.bdnvt can now tally the energy transferred into/out of the
    “reservoir”, allowing energy conservation to be monitored during bd
    simulation runs.
18. Most potentials now prevent NaN results when computed for
    overlapping particles
19. Stopping a simulation from a callback or time limit no longer
    produces invalid simulations when continued
20. run() commands limited with limit_hours can now be set to only stop
    on given timestep multiples
21. Worked around a compiler bug where pair.morse would crash on Fermi
    GPUs
22. ULF stability improvements for G200 GPUs.

Version 0.8.2 (2009-09-10)
^^^^^^^^^^^^^^^^^^^^^^^^^^

*Note:* only major changes are listed here.

*New features*

1.  Quantities that vary over time can now be specified easily in
    scripts with the variant.linear_interp command.
2.  Box resizing updater (update.box_resize) command that uses the time
    varying quantity command to grow or shrink the simulation box.
3.  Individual run() commands can be limited by wall-clock time
4.  Angle forces can now be specified
5.  Dihedral forces can now be specified
6.  Improper forces can now be specified
7.  1-3 and 1-4 exclusions from the cutoff pair force can now be chosen
8.  New command line option: –minimize-cpu-usage cuts the CPU usage of
    HOOMD down to 10% of one CPU core while only decreasing overall
    performance by 10%
9.  Major changes have been made in the way HOOMD chooses the device on
    which to run (all require CUDA 2.2 or newer)

    -  there are now checks that an appropriate NVIDIA drivers is
       installed
    -  running without any command line options will now correctly
       revert to running on the CPU if no capable GPUs are installed
    -  when no gpu is explicitly specified, the default choice is now
       prioritized to choose the fastest GPU and one that is not
       attached to a display first
    -  new command line option: –ignore-display-gpu will prevent HOOMD
       from executing on any GPU attached to a display
    -  HOOMD now prints out a short description of the GPU(s) it is
       running on
    -  on linux, devices can be set to compute-exclusive mode and HOOMD
       will then automatically choose the first free GPU (see the
       documentation for details)

10. nlist.reset_exclusions command to control the particles that are
    excluded from the neighbor list

*Bug/fixes small enhancements*

1.  Default block size change to improve stability on compute#3 devices
2.  ULF workaround on GTX 280 now works with CUDA 2.2
3.  Standalone benchmark executables have been removed and replaced by
    in script benchmarking commands
4.  Block size tuning runs can now be performed automatically using the
    python API and results can be saved on the local machine
5.  Fixed a bug where GTX 280 bug workarounds were not properly applied
    in CUDA 2.2
6.  The time step read in from the XML file can now be optionally
    overwritten with a user-chosen one
7.  Added support for CUDA 2.2
8.  Fixed a bug where the WCA forces included in bond.fene had an
    improper cutoff
9.  Added support for a python callback to be executed periodically
    during a run()
10. Removed demos from the hoomd downloads. These will be offered
    separately on the webpage now to keep the required download size
    small.
11. documentation improvements
12. Significantly increased performance of dual-GPU runs when build with
    CUDA 2.2 or newer
13. Numerous stability and performance improvements
14. Temperatures are now calculated based on 3N-3 degrees of freedom.
    See #283 for a more flexible system that is coming in the future.
15. Emulation mode builds now work on systems without an NVIDIA card
    (CUDA 2.2 or newer)
16. HOOMD now compiles with CUDA 2.3
17. Fixed a bug where uninitialized memory was written to dcd files
18. Fixed a bug that prevented the neighbor list on the CPU from working
    properly with non-cubic boxes
19. There is now a compile time hack to allow for more than 4 exclusions
    per particle
20. Documentation added to aid users in migrating from LAMMPS
21. hoomd_script now has an internal version number useful for third
    party scripts interfacing with it
22. VMD#8.7 is now found by the live demo scripts
23. live demos now run in vista 64-bit
24. init.create_random_polymers can now create polymers with more than
    one type of bond

Version 0.8.1 (2009-03-24)
^^^^^^^^^^^^^^^^^^^^^^^^^^

*Note:* only major changes are listed here.

*New features*

1.  Significant performance enhancements
2.  New build option for compiling on UMich CAC clusters:
    ENABLE_CAC_GPU_ID compiles HOOMD to read in the *$CAC_GPU_ID*
    environment variable and use it to determine which GPUs to execute
    on. No –gpu command line required in job scripts any more.
3.  Particles can now be assigned a *non-unit mass*
4.  *init.reset()* command added to allow for the creation of a looped
    series of simulations all in python
5.  *dump.pdb()* command for writing PDB files
6.  pair.lj now comes with an option to *shift* the potential energy to
    0 at the cutoff
7.  pair.lj now comes with an opiton to *smoothly switch* both the
    *potential* and *force* to 0 at the cutoff with the XPLOR smoothing
    function
8.  *Gaussian pair potential* computation added (pair.gauss)
9.  update and analyze commands can now be given a function to determine
    a non-linear rate to run at
10. analyze.log, and dump.dcd can now append to existing files

*Changes that will break scripts from 0.8.0*

1. *dump.mol2()* has been changed to be more consistent with other dump
   commands. In order to get the same result as the previous behavior,
   replace

   .. code::

       dump.mol2(filename="file.mol2")

   with

   .. code::

       mol2 = dump.mol2()
       mol2.write(filename="file.mol2")

2. Grouping commands have been moved to their own package for
   organizational purposes. *group_all()* must now be called as
   *group.all()* and similarly for tags and type.

*Bug/fixes small enhancements*

1.  Documentation updates
2.  DCD file writing no longer crashes HOOMD in windows
3.  !FindBoost.cmake is patched upstream. Use CMake 2.6.3 if you need
    BOOST_ROOT to work correctly
4.  Validation tests now run with –gpu_error_checking
5.  ULF bug workarounds are now enabled only on hardware where they are
    needed. This boosts performance on C1060 and newer GPUs.
6.  !FindPythonLibs now always finds the shared python libraries, if
    they exist
7.  “make package” now works fine on mac os x
8.  Fixed erroneously reported dangerous neighbor list builds when using
    –mode=cpu
9.  Small tweaks to the XML file format.
10. Numerous performance enhancements
11. Workaround for ULF on compute#1 devices in place
12. dump.xml can now be given the option “all=true” to write all fields
13. total momentum can now be logged by analyze.log
14. HOOMD now compiles with boost#38 (and hopefully future versions)
15. Updaters can now be given floating point periods such as 1e5
16. Additional warnings are now printed when HOOMD is about to allocate
    a large amount of memory due to the specification of an extremely
    large box size
17. run() now shows up in the documentation index
18. Default sorter period is now 100 on CPUs to improve performance on
    chips with small caches

Version 0.8.0 (2008-12-22)
^^^^^^^^^^^^^^^^^^^^^^^^^^

*Note:* only major changes are listed here.

*New features*

1. Addition of FENE bond potential
2. Addition of update.zero_momentum command to zero a system’s linear
   momentum
3. Brownian dynamics integration implemented
4. Multi-GPU simulations
5. Particle image flags are now tracked. analyze.msd command added to
   calculate the mean squared displacement.

*Changes that will break scripts from 0.7.x*

1. analyze.log quantity names have changed

*Bug/fixes small enhancements*

1.  Performance of the neighbor list has been increased significantly on
    the GPU (overall performance improvements are approximately 10%)
2.  Profile option added to the run() command
3.  Warnings are now correctly printed when negative coefficients are
    given to bond forces
4.  Simulations no longer fail on G200 cards
5.  Mac OS X binaries will be provided for download: new documentation
    for installing on Mac OS x has been written
6.  Two new demos showcasing large systems
7.  Particles leaving the simulation box due to bad initial conditions
    now generate an error
8.  win64 installers will no longer attempt to install on win32 and
    vice-versa
9.  neighborlist check_period now defaults to 1
10. The elapsed time counter in run() now continues counting time over
    multiple runs.
11. init.create_random_polymers now throws an error if the bond length
    is too small given the specified separation radii
12. Fixed a bug where a floating point value for the count field in
    init.create_random_polymers produced an error
13. Additional error checking to test if particles go NaN
14. Much improved status line printing for identifying hoomd_script
    commands
15. Numerous documentation updates
16. The VS redistributable package no longer needs to be installed to
    run HOOMD on windows (these files are distributed with HOOMD)
17. Now using new features in doxygen#5.7 to build pdf user
    documentation for download.
18. Performance enhancements of the Lennard-Jones pair force
    computation, thanks to David Tarjan
19. A header prefix can be added to log files to make them more gnuplot
    friendly
20. Log quantities completely revamped. Common quantities (i.e. kinetic
    energy, potential energy can now be logged in any simulation)
21. Particle groups can now be created. Currently only analyze.msd makes
    use of them.
22. The CUDA toolkit no longer needs to be installed to run a packaged
    HOOMD binary in windows.
23. User documentation can now be downloaded as a pdf.
24. Analyzers and updaters now count time 0 as being the time they were
    created, instead of time step 0.
25. Added job test scripts to aid in validating HOOMD
26. HOOMD will now build with default settings on a linux/unix-like OS
    where the boost static libraries are not installed, but the dynamic
    ones are.

Version 0.7.1 (2008-09-12)
^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Fixed bug where extremely large box dimensions resulted in an
   argument error - ticket:118
2. Fixed bug where simulations ran incorrectly with extremely small box
   dimensions - ticket:138

Version 0.7.0 (2008-08-12)
^^^^^^^^^^^^^^^^^^^^^^^^^^

*Note:* only major changes are listed here.

1.  Stability and performance improvements.
2.  Cleaned up the hoomd_xml file format.
3.  Improved detection of errors in hoomd_xml files significantly.
4.  Users no longer need to manually specify HOOMD_ROOT, unless their
    installation is non-standard
5.  Particle charge can now be read in from a hoomd_xml file
6.  Consistency changes in the hoomd_xml file format: HOOMD 0.6.0 XML
    files are not compatible. No more compatibility breaking changes are
    planned after 0.7.0
7.  Enabled parallel builds in MSVC for faster compilation times on
    multicore systems
8.  Numerous small bug fixes
9.  New force compute for implementing walls
10. Documentation updates
11. Support for CUDA 2.0
12. Bug fixed allowing simulations with no integrator
13. Support for boost#35.0
14. Cleaned up GPU code interface
15. NVT integrator now uses tau (period) instead of Q (the mass of the
    extra degree of freedom).
16. Added option to NVE integration to limit the distance a particle
    moves in a single time step
17. Added code to dump system snapshots in the DCD file format
18. Particle types can be named by strings
19. A snapshot of the initial configuration can now be written in the
    .mol2 file format
20. The default build settings now enable most of the optional features
21. Separated the user and developer documentation
22. Mixed polymer systems can now be generated inside HOOMD
23. Support for CMake 2.6.0
24. Wrote the user documentation
25. GPU selection from the command line
26. Implementation of the job scripting system
27. GPU can now handle neighbor lists that overflow
28. Energies are now calculated
29. Added a logger for logging energies during a simulation run
30. Code now actually compiles on Mac OS X
31. Benchmark and demo scripts now use the new scripting system
32. Consistent error message format that is more visible.
33. Multiple types of bonds each with the own coefficients are now
    supported
34. Added python scripts to convert from HOOMD’s XML file format to
    LAMMPS input and dump files
35. Fixed a bug where empty xml nodes in input files resulted in an
    error message
36. Fixed a bug where HOOMD seg faulted when a particle left the
    simulation , vis=True)\* is now a convenient short hand for what was
    previously box now works fine on mac os x
37. Fixed erroneously reported dangerous neighbor list builds when using
    –mode=cpu
38. Small tweaks to the XML file format.
39. Numerous performance enhancements
40. Workaround for ULF on compute#1 devices in place
41. dump.xml can now be given the option
