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
from hoomd import cite
from hoomd import analyze
from hoomd import benchmark
from hoomd import comm
from hoomd import compute
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
from hoomd import trigger
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

from hoomd.box import Box
from hoomd.simulation import Simulation
from hoomd.state import State
from hoomd.operations import Operations
from hoomd.snapshot import Snapshot
from hoomd import tuner
from hoomd import output
from hoomd import logging
from hoomd import custom
from hoomd._hoomd import WalltimeLimitReached

_default_excepthook = sys.excepthook

def _hoomd_sys_excepthook(type, value, traceback):
    """Override Python's excepthook to abort MPI runs."""
    _default_excepthook(type, value, traceback)
    sys.stderr.flush()
    _hoomd.abort_mpi(comm._current_communicator.cpp_mpi_conf, 1)

sys.excepthook = _hoomd_sys_excepthook

__version__ = "2.9.0"

# TODO: Decide how these properties should be exposed.
## \internal
# \brief Gather context about HOOMD
class build_info():
    ## \internal
    # \brief Constructs the context object
    def __init__(self):
        hoomd.meta._metadata.__init__(self)
        self.metadata_fields = [
            'hoomd_version', 'hoomd_git_sha1', 'hoomd_git_refspec',
            'hoomd_compile_flags', 'cuda_version', 'compiler_version',
            ]

    # \brief Return the hoomd version.
    @property
    def hoomd_version(self):
        return _hoomd.__version__

    # \brief Return the hoomd git hash
    @property
    def hoomd_git_sha1(self):
        return _hoomd.__git_sha1__

    # \brief Return the hoomd git refspec
    @property
    def hoomd_git_refspec(self):
        return _hoomd.__git_refspec__

    # \brief Return the hoomd compile flags
    @property
    def hoomd_compile_flags(self):
        return _hoomd.hoomd_compile_flags();

    # \brief Return the cuda version
    @property
    def cuda_version(self):
        return _hoomd.__cuda_version__

    # \brief Return the compiler version
    @property
    def compiler_version(self):
        return _hoomd.__compiler_version__

## Global bibliography
_bib = None

# TODO: orphaned from context.py. Only used in option.py
## Global options
_options = None