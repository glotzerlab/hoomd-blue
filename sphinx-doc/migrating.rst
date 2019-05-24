Migrating to HOOMD v3
=====================

HOOMD v3 introduces a number of breaking changes for both users and developers in order to provide a cleaner
python interface, enable new functionalities, and move away from unsupported tools. This guide highlights
those changes.

Compiling
---------

* CMake 3.8 or newer is required to build HOOMD.
* ``UPDATE_SUBMODULES`` no longer exists. Users and developers should use ``clone --recursive``,
  ``git submodule update`` and ``git submodule sync`` as appropriate.
* HOOMD now uses native CUDA support in CMake. Use ``CMAKE_CUDA_COMPILER`` to specify a specific ``nvcc``. Plugins
  will require updates to ``CMakeLists.txt`` to compile ``.cu`` files.
* External plugins will require additional updates to work with v3 (full list of changes pending).
* ``COPY_HEADERS`` no longer exists. Users must install HOOMD for use with external plugins.
* ``CMAKE_INSTALL_PREFIX`` is set to the python ``site-packages`` directory (if not explicitly set by the user)
