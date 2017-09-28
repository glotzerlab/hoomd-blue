# Copyright (c) 2009-2017 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: mphoward

R""" Multiparticle collision dynamics

MPCD is a mesoscale, particle-based simulation for hydrodynamics.

.. rubric:: Algorithm and applications

.. rubric:: Stability

:py:mod:`hoomd.mpcd` is **unstable**. (It is currently under development.)
**Maintainer:** Michael P. Howard, Princeton University.
"""

# these imports are necessary in order to link derived types between modules
from hoomd import _hoomd
from hoomd.md import _md

from hoomd.mpcd import collide
from hoomd.mpcd import data
from hoomd.mpcd import init
from hoomd.mpcd import integrate
from hoomd.mpcd import stream
from hoomd.mpcd import update

# pull the integrator into the main module namespace for convenience
# (we want to type mpcd.integrator not mpcd.integrate.integrator)
from hoomd.mpcd.integrate import integrator
