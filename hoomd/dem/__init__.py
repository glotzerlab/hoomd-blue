# Copyright (c) 2009-2016 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

R"""Simulate rounded, faceted shapes in molecular dynamics.

The DEM component provides forces which apply short-range, purely
repulsive interactions between contact points of two shapes. The
resulting interaction is consistent with expanding the given polygon
or polyhedron by a disk or sphere of a particular rounding radius.
"""

# this file exists to mark this directory as a python module

# need to import all submodules defined in this directory

from hoomd.dem import pair
from hoomd.dem import params
from hoomd.dem import utils
