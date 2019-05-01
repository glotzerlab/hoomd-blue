# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

""" Deprecated functionalities

Commands in the :py:mod:`hoomd.deprecated` package are leftovers from previous versions of HOOMD-blue that are
kept temporarily for users whose workflow depends on them. Deprecated features may be removed in a future version.

.. rubric:: Stability

:py:mod:`hoomd.deprecated` is **deprecated**. When upgrad from version 2.x to 2.y (y > x),
functions and classes in the package may be removed. Continued support for features in this
package is not provided. These legacy functions will remain as long as they require minimal
code modifications to maintain. **Maintainer:** *not maintained*.
"""

from hoomd.deprecated import analyze
from hoomd.deprecated import dump
from hoomd.deprecated import init
