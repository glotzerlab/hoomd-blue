.. Copyright (c) 2009-2024 The Regents of the University of Michigan.
.. Part of HOOMD-blue, released under the BSD 3-Clause License.

When using external periodic fields in 2D systems, a division-by-zero error occurs because reciprocal vectors are defined only for 3D systems. To avoid this issue, the maintainers previously disabled this functionality for true 2D systems. These changes introduce a proper method for computing reciprocal vectors in 2D, resolving the error and enabling the use of external periodic fields in 2D systems.

.. include:: ../CHANGELOG.rst
