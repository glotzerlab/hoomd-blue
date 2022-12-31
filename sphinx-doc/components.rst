.. Copyright (c) 2009-2022 The Regents of the University of Michigan.
.. Part of HOOMD-blue, released under the BSD 3-Clause License.

Components
==========

Extend **HOOMD-blue** with a **component** implemented in C++ for performance-critical tasks, such
as pair potential evaluation. A component provides a number of related functionalities. For example,
the :py:mod:`hoomd.hpmc` component enables hard particle Monte Carlo methods with **HOOMD-blue**.

Compile and install components **built-in** or as **external** components. The **HOOMD-blue** build
process compiles all core and built-in components together, requiring one only one set of configure,
make, and install commands. External components compile and link against a **HOOMD-blue**
installation from a separate build directory with their own set of configure, make, and install
commands. You may compile a component either way. When the end user is compiling **HOOMD-blue** and
components from source, built-in components compile and install everything at once which minimizes
chances for errors (e.g. building **HOOMD-blue** against python 3.6, but the component against
python 3.7). External components provide more flexibility for packaging purposes.

The **HOOMD-Blue** source provides example component templates in the ``example_plugins``
subdirectory. ``updater_plugin`` demonstrates how to add a new ``update`` command with both CPU and GPU
implementations, and ``pair_plugin`` shows how to add a new MD pair potential.

Built-in components
-------------------

You can fork **HOOMD-blue** and add your component directly, or you can create a separate source
repository for your component. Create a symbolic link to the component in the ``hoomd`` source
directory to compile it as a built-in component::

  $ ln -s <path-to-component>/<component> hoomd-blue/hoomd/<component>

.. note::

    Built-in components may be used directly from the build directory or installed.

External components
-------------------

To compile an external component, you must first install **HOOMD-blue**. Then, configure your component
with CMake and install it into the hoomd python library. Point ``CMAKE_PREFIX_PATH`` at your virtual
environment (if needed) so that cmake can find **HOOMD-blue**::

  $ cmake -B build/<component> -S <path-to-component>
  $ cmake --build build/<component>
  $ cmake --install build/<component>

The component build environment, including the compiler, CUDA, MPI, python, and other libraries,
must match exactly with those used to build **HOOMD-blue**.

.. note::

    External components must be installed before use.
