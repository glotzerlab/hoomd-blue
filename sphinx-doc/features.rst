.. Copyright (c) 2009-2024 The Regents of the University of Michigan.
.. Part of HOOMD-blue, released under the BSD 3-Clause License.

Features
========

Hard particle Monte Carlo
-------------------------

HOOMD-blue can perform simulations of hard particles using the Monte Carlo method (`hpmc`). Hard
particles are defined by their shape, and the `HPMC integrator <hpmc.integrate>` supports
polygons, spheropolygons, polyhedra, spheropolyhedra, ellipsoids, faceted ellipsoids, spheres,
indented spheres, and unions of shapes. HPMC can make both constant volume and constant pressure
box moves (`hpmc.update.BoxMC`), perform cluster moves (`hpmc.update.Clusters`)
and can compute the pressure during constant volume simulations (`hpmc.compute.SDF`).

HPMC can also apply external and pair potentials to the particles. Use
`hpmc.external.field.Harmonic` to restrain particles to a lattice (e.g. for Frenkel-Ladd
calculations) or `hpmc.external.user.CPPExternalPotential` to implement arbitrary external fields
(e.g. gravity). Use `hpmc.pair.user` to define arbitrary pairwise interactions between particles.
At runtime, `hoomd.version.hpmc_built` indicates whether the build supports HPMC simulations.

.. seealso::

    Tutorial: :doc:`tutorial/00-Introducing-HOOMD-blue/00-index`

Molecular dynamics
------------------

HOOMD-blue can perform molecular dynamics simulations (`md`) with constant volume, constant
pressure, Langevin, Brownian, overdamped viscous integration methods (`md.methods`), and energy
minimization (`md.minimize`). The constant volume and constant pressure methods may be applied with
or without a thermostat (`md.methods.thermostats`). Unless otherwise stated in the documentation,
all integration methods integrate both translational and rotational degrees of freedom. Some
integration methods support manifold constraints (`md.methods.rattle`). HOOMD-blue provides a number
of cutoff potentials including pair potentials (`md.pair`), pair potentials that depend on particle
orientation (`md.pair.aniso`), and many body potentials (`md.many_body`). HOOMD-blue also provides
bond potentials and distance constraints commonly used in atomistic/coarse-grained force fields
(`md.angle`, `md.bond`, `md.constrain.Distance`, `md.dihedral`, `md.improper`, `md.special_pair`)
and can model rigid bodies (`md.constrain.Rigid`). External fields `md.external.field` apply
potentials based only on the particle's position and orientation, including walls
(`md.external.wall`) to confine particles in a specific region of space. `md.long_range` provides
long ranged interactions, including the PPPM method for electrostatics. HOOMD-blue enables active
matter simulations with `md.force.Active` and `md.update.ActiveRotationalDiffusion`. At runtime,
`hoomd.version.md_built` indicates whether the build supports MD simulations.

.. seealso::

    Tutorial: :doc:`tutorial/01-Introducing-Molecular-Dynamics/00-index`

Python package
--------------

HOOMD-blue is a Python package and is designed to interoperate with other packages in the scientific
Python ecosystem and to be extendable in user scripts. To enable interoperability, all operations
provide access to useful computed quantities as properties in native Python types or numpy arrays
where appropriate. Additionally, `State <hoomd.State>` and `md.force.Force` provide direct access to
particle properties and forces using Python array protocols. Users can customize their simulation or
extend HOOMD-blue with functionality implemented in Python code by subclassing `trigger.Trigger`,
`variant.Variant`, `hoomd.update.CustomUpdater`, `hoomd.write.CustomWriter`,
`hoomd.tune.CustomTuner`, or by using the HOOMD-blue API in combination with other Python packages
to implement methods that couple simulation, analysis, and multiple simulations (such as umbrella
sampling).

.. seealso::

    Tutorial: :doc:`tutorial/04-Custom-Actions-In-Python/00-index`

CPU and GPU devices
-------------------

HOOMD-blue can execute simulations on CPUs or GPUs. Typical simulations run more efficiently on
GPUs for system sizes larger than a few thousand particles, although this strongly depends on the
details of the simulation. The provided binaries support NVIDIA GPUs. Build from source to enable
preliminary support for AMD GPUs. CPU support is always enabled. GPU support must be enabled at
compile time with the ``ENABLE_GPU`` CMake option (see :doc:`building`). Select the device to use at
run time with the `device <hoomd.device>` module. Unless otherwise stated in the documentation,
**all** operations and methods support GPU execution. At runtime, `hoomd.version.gpu_enabled` indicates
whether the build supports GPU devices.

Autotuned kernel parameters
---------------------------

HOOMD-blue automatically tunes kernel parameters to improve performance when executing on a GPU
device. During the first 1,000 - 20,000 timesteps of the simulation run, HOOMD-blue will change
kernel parameters each time it calls a kernel. Kernels compute the same output regardless of the
parameter (within floating point precision), but the parameters have a large impact on performance.

Check to see whether tuning is complete with the `is_tuning_complete
<hoomd.Operations.is_tuning_complete>` attribute of your simulation's `Operations
<hoomd.Operations>`. For example, use this to run timed benchmarks after the performance stabilizes.

The optimal parameters can depend on the number of particles in the simulation and the density, and
may vary weakly with other system properties. To maintain peak performance, call
`tune_kernel_parameters <hoomd.Operations.tune_kernel_parameters>` to tune the parameters again after
making a change to your system.

`AutotunedObject` provides a settable dictionary parameter with the current kernel parameters in
`kernel_parameters <hoomd.operation.AutotunedObject.kernel_parameters>`. Use this to inspect the
autotuner's behavior or override with specific values (e.g. values saved from a previous execution).

MPI
---

HOOMD-blue can use the message passing interface (MPI) to execute simulations in less time using
more than one CPU core or GPU. Unless otherwise stated in the documentation, **all** operations and
methods support MPI parallel execution. MPI support is optional, requires a compatible MPI library,
and must be enabled at compile time with the ``ENABLE_MPI`` CMake option (see :doc:`building`).
At runtime, `hoomd.version.mpi_enabled` indicates whether the build supports MPI.

.. seealso::

    Tutorial: :doc:`tutorial/03-Parallel-Simulations-With-MPI/00-index`

Threading
---------

Some operations in HOOMD-blue can use multiple CPU threads in a single process. Control this with
the `device.Device.num_cpu_threads` property. In this release, threading support in HOOMD-blue is
very limited and only applies to implicit depletants in `hpmc.integrate.HPMCIntegrator`, and
`hpmc.pair.user.CPPPotentialUnion`. Threading must must be enabled at compile time with the
``ENABLE_TBB`` CMake option (see :doc:`building`). At runtime, `hoomd.version.tbb_enabled` indicates
whether the build supports threaded execution.

Mixed precision
---------------

HOOMD-blue performs computations with mixed floating point precision. There is a **high precision**
type and a **reduced precision** type. All particle properties are stored in the high precision
type, and most operations also perform all computations with high precision. Operations that do not
mention "Mixed precision" in their documentation perform all calculations in high precision. Some
operations use reduced precision when possible to improve performance, as detailed in the
documentation for each operation.

The precision is set at compile time with the ``HOOMD_LONGREAL_SIZE`` and
``HOOMD_SHORTREAL_SIZE`` CMake options (see :doc:`building`). By default, the high precision
width is 64 bits and the reduced precision width is 32 bits. At runtime,
`hoomd.version.floating_point_precision` indicates the width of the floating point types.

Plugins
-------

Plugin code that provides additional functionality to HOOMD-blue may be implemented in pure Python
or as a package with C++ compiled libraries.

.. seealso::

    :doc:`components`
