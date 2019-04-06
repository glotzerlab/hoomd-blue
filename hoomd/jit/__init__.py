# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

""" JIT

The JIT module provides *experimental* support to to JIT (just in time) compile C++ code and call it during the
simulation. Compiled C++ code will execute at full performance unlike interpreted python code.

.. rubric:: Stability

:py:mod:`hoomd.jit` is **unstable**. When upgrading from version 2.x to 2.y (y > x),
existing job scripts may need to be updated. **Maintainer:** Joshua A. Anderson, University of Michigan

.. versionadded:: 2.3
"""

from hoomd.hpmc import _hpmc

from hoomd.jit import patch
from hoomd.jit import external
