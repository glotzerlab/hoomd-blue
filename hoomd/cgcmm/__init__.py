# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

""" CGCMM

Coarse grained CGCMM potential.

.. rubric:: Stability

:py:mod:`hoomd.cgcmm` is **unstable**. When upgrading from version 2.x to 2.y (y > x),
existing job scripts may need to be updated. **Maintainer needed!** This package is not maintained.
"""

from hoomd.cgcmm import angle
from hoomd.cgcmm import pair

# Log that we imported cgcmm
from hoomd import meta
meta.MODULES.append('cgcmm')
