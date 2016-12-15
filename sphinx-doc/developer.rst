Developer Topics
================

Plugins and Components
----------------------

HOOMD-Blue can be tuned for particular use cases in several ways. Many smaller workloads and higher-level use
cases can be handled through python-level :py:func:`hoomd.run` callbacks. For heavier-duty work such as pair
potential evaluation, HOOMD-Blue can be extended through ``components``. Components provide sets of
functionality with a common overarching goal; for example, the :py:mod:`hoomd.hpmc` component provides code
for hard particle Monte Carlo methods within HOOMD-Blue.

Components can be compiled and installed as **builtin** components or as **external** components. Builtin
components are built and installed alongside the rest of HOOMD-Blue, while external components are compiled
after HOOMD-Blue has already been compiled and installed at its destination. They have the same capabilities,
but builtin components are simpler to build while external components are more flexible for packaging
purposes.

The HOOMD-Blue source provides an example component template in the `example_plugin` subdirectory which
supports installation either as a builtin component or as an external component, depending on how it is
configured.

To set up the example component as a builtin component, simply create a symbolic link to the **internal**
`example_plugin` directory (`example_plugin/example_plugin`) inside the `hoomd` subdirectory::

  $ cd /path/to/hoomd-blue/hoomd
  $ ln -s ../example_plugin/example_plugin example_plugin
  $ cd ../build && make install

Note that this has already been done for the case of the example component.

Alternatively, one can use the example component as an external component. This relies on the
`FindHOOMD.cmake` cmake script to set up cmake in a way that closely mirrors the cmake environment that HOOMD
Blue was originally compiled with. The process is very similar to the process of installing HOOMD Blue
itself. For ease of configuration, it is best to make sure that the `hoomd` module that is automatically
imported by python is the one you wish to configure the component against and install to::

  $ cd /path/to/component
  $ mkdir build && cd build
  $ # This python command should print the location you wish to install into
  $ python -c 'import hoomd;print(hoomd.__file__)'
  $ # Add any extra cmake arguments you need here (like -DPYTHON_EXECUTABLE)
  $ cmake ..
  $ make install
