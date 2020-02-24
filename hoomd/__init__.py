# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

""" HOOMD-blue python API

:py:mod:`hoomd` provides a high level user interface for executing
simulations using HOOMD::

    import hoomd
    from hoomd import md
    hoomd.context.initialize()

    # create a 10x10x10 square lattice of particles with name A
    hoomd.init.create_lattice(unitcell=hoomd.lattice.sc(a=2.0, type_name='A'), n=10)
    # specify Lennard-Jones interactions between particle pairs
    nl = md.nlist.cell()
    lj = md.pair.lj(r_cut=3.0, nlist=nl)
    lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
    # integrate at constant temperature
    all = hoomd.group.all();
    md.integrate.mode_standard(dt=0.005)
    hoomd.md.integrate.langevin(group=all, kT=1.2, seed=4)
    # run 10,000 time steps
    hoomd.run(10e3)

.. rubric:: Stability

:py:mod:`hoomd` is **stable**. When upgrading from version 2.x to 2.y (y > x),
existing job scripts that follow *documented* interfaces for functions and classes
will not require any modifications. **Maintainer:** Joshua A. Anderson

.. attention::

    This stability guarantee only applies to modules in the :py:mod:`hoomd` package.
    Subpackages (:py:mod:`hoomd.hpmc`, :py:mod:`hoomd.md`, etc...) may or may not
    have a stable API. The documentation for each subpackage specifies the level of
    API stability it provides.
"""

# Maintainer: joaander
import sys;
import ctypes;
import os;

# need to import HOOMD with RTLD_GLOBAL in python sitedir builds
if not ('NOT_HOOMD_PYTHON_SITEDIR' in os.environ):
    flags = sys.getdlopenflags();
    sys.setdlopenflags(flags | ctypes.RTLD_GLOBAL);

from hoomd import _hoomd;

if not ('NOT_HOOMD_PYTHON_SITEDIR' in os.environ):
    sys.setdlopenflags(flags);

from hoomd import meta
from hoomd import context
from hoomd import cite
from hoomd import analyze
from hoomd import benchmark
from hoomd import comm
from hoomd import compute
from hoomd import data
from hoomd import dump
from hoomd import group
from hoomd import init
from hoomd import integrate
from hoomd import option
from hoomd import update
from hoomd import util
from hoomd import variant
from hoomd import lattice
from hoomd import device
try:
    from hoomd import md
except ImportError:
    pass
try:
    from hoomd import hpmc
except ImportError:
    pass
try:
    from hoomd import dem
except ImportError:
    pass
# TODO: enable this import after updating MPCD to the new API
# try:
#     from hoomd import mpcd
# except ImportError:
#     pass

from hoomd.simulation import Simulation
from hoomd.operations import Operations
from hoomd.snapshot import Snapshot
from hoomd.logger import Logger

from hoomd._hoomd import WalltimeLimitReached;

_default_excepthook = sys.excepthook;

## \internal
# \brief Override pythons except hook to abort MPI runs
def _hoomd_sys_excepthook(type, value, traceback):
    _default_excepthook(type, value, traceback);
    sys.stderr.flush();
    if context.current.device is not None:
        _hoomd.abort_mpi(context.current.device.cpp_exec_conf);

sys.excepthook = _hoomd_sys_excepthook

__version__ = "{0}.{1}.{2}".format(*_hoomd.__version__)
