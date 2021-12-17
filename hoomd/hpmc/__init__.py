# -*- coding: iso-8859-1 -*-
# this file exists to mark this directory as a python module

"""Hard particle Monte Carlo.

HPMC performs hard particle Monte Carlo simulations of a variety of classes of
shapes.
"""

# need to import all submodules defined in this directory
from hoomd.hpmc import integrate
from hoomd.hpmc import update
from hoomd.hpmc import compute
from hoomd.hpmc import tune
from hoomd.hpmc import pair
from hoomd.hpmc import external
