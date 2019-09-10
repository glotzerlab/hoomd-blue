Developer Topics
================

Components
----------

Extend HOOMD-blue with a **component** implemented in C++ for performance-critical tasks, such as pair potential
evaluation. A component provides a number of related functionalities. For example, the :py:mod:`hoomd.hpmc` component
enables hard particle Monte Carlo methods with HOOMD-blue.

Components can be compiled and installed as **built-in** components or as **external** components. The build process
compiles HOOMD-blue and all built-in components together, requiring one ``cmake`` and ``make``. External components
compile and link against a HOOMD-blue installation with a separate invocation of ``cmake`` and ``make install``. A
properly configured component may be compiled either way. When the end user is compiling HOOMD-blue and components from
source, built-in components compile and install everything at once which minimizes chances for errors (e.g. building
HOOMD-blue against python 3.6, but the component against python 3.7). External components provide more flexibility for
packaging purposes.

The HOOMD-Blue source provides an example component template in the ``example_plugin`` subdirectory. ``example_plugin``
demonstrates how to add a new ``update`` command with both CPU and GPU implementations. Use this as a template when
developing your component.

Built-in components
^^^^^^^^^^^^^^^^^^^

You can fork HOOMD-blue and add your component directly, or you can create a separate source repository for your
component. Create a symbolic link to the component in the ``hoomd`` source directory to compile it as a built-in
component::

  ▶  cd /path/to/hoomd-blue/hoomd
  ▶  ln -s /path/to/your_component/your_component your_component
  ▶  cd ../build
  ▶  make

Built-in components may be used directly from the ``build`` directory or installed.

External components
^^^^^^^^^^^^^^^^^^^

To compile an external component, you must first install HOOMD-blue. Then, configure your component with ``cmake`` and
``make install`` it into the hoomd python library. Point ``CMAKE_PREFIX_PATH`` at your virtual environment (if needed)
so that cmake can find HOOMD::

  ▶  cd /path/to/your_component
  ▶  mkdir build && cd build
  ▶  CMAKE_PREFIX_PATH=/path/to/virtual/environment cmake ..
  ▶  make install

The component build environment, including the compiler, CUDA, MPI, python, and other libraries, must match exactly with
those used to build HOOMD-blue.

External components must be installed before use.
