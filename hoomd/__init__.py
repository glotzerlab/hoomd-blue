# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

""" HOOMD-blue python API

:py:mod:`hoomd` provides a high level user interface for defining and executing
simulations using HOOMD.

.. rubric:: API stability

:py:mod:`hoomd` is **stable**. When upgrading from version 3.x to 3.y (y > x),
existing job scripts that follow *documented* interfaces for functions and
classes will not require any modifications.

**Maintainer:** Joshua A. Anderson

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
from hoomd.state import State
from hoomd.operations import Operations
from hoomd.snapshot import Snapshot
from hoomd.logger import Logger
from hoomd.box import Box
from hoomd import tuner
from hoomd.custom_action import CustomAction
from hoomd.custom_operation import _CustomOperation

from hoomd._hoomd import WalltimeLimitReached

_default_excepthook = sys.excepthook

def _hoomd_sys_excepthook(type, value, traceback):
    """Override Python's excepthook to abort MPI runs."""
    _default_excepthook(type, value, traceback)
    sys.stderr.flush()
    _hoomd.abort_mpi(comm._current_communicator.cpp_mpi_conf, 1)

sys.excepthook = _hoomd_sys_excepthook

__version__ = "2.9.0"
