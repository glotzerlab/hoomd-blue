# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

""" CGCMM

Coarse grained CGCMM potential.

.. rubric:: Stability

:py:mod:`hoomd.cgcmm` is **unstable**. When upgrading from version 2.x to 2.y (y > x),
existing job scripts may need to be updated. **Maintainer needed!** This package is not maintained.

.. deprecated:: 2.6
   The cgcmm component has not been maintained in many years and will be removed.

"""

from hoomd.cgcmm import angle
from hoomd.cgcmm import pair
