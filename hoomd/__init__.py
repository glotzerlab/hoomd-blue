# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""HOOMD-blue python package.

`hoomd` is the top level HOOMD-blue Python package. It consists of the common
code shared among all types of HOOMD-blue simulations. The core data structures
used to construct a simulation include:

* `Simulation`

  * `hoomd.device.Device`
  * `hoomd.State`
  * `hoomd.Operations`

    * `hoomd.operations.Integrator>`
    * `hoomd.operations.Compute`
    * `hoomd.operations.Tuner`
    * `hoomd.operations.Updater`
    * `hoomd.operations.Writer`

See the table of contents or the modules section for a full list of classes,
methods, and variables in the API.

`hoomd` also contains subpackages that implement specific types of simulations:

* `hoomd.hpmc` - Hard particle Monte Carlo.
* `hoomd.md` - Molecular dynamics.

See Also:
    Tutorial: :doc:`tutorial/00-Introducing-HOOMD-blue/00-index`
"""
import sys
import pathlib
import os

if ((pathlib.Path(__file__).parent / 'CMakeLists.txt').exists()
        and 'SPHINX' not in os.environ):
    print("It appears that hoomd is being imported from the source directory:")
    print(pathlib.Path(__file__).parent)
    print()
    print("""Compile the package and import from the build directory or install
the package and import from the Python environment.

To run pytest, either:
(1) compile then execute `python3 -m pytest <build-directory>/hoomd` or
(2) compile and install. Then, ensuring your current working directory is
outside the hoomd source directory, execute `python3 -m pytest --pyargs hoomd`.
""",
          file=sys.stderr)

from hoomd import version
from hoomd import trigger
from hoomd import variant
from hoomd.box import Box
from hoomd import data
from hoomd import filter
from hoomd import device
from hoomd import error
from hoomd import update
from hoomd import communicator
from hoomd import util
from hoomd import write
from hoomd import wall
from hoomd import _hoomd
if version.md_built:
    from hoomd import md
if version.hpmc_built:
    from hoomd import hpmc
if version.dem_built and version.md_built:
    from hoomd import dem
# if version.metal_built:
#     from hoomd import metal
# if version.mpcd_built:
#     from hoomd import mpcd

from hoomd.simulation import Simulation
from hoomd.state import State
from hoomd.operations import Operations
from hoomd.snapshot import Snapshot
from hoomd import tune
from hoomd import logging
from hoomd import custom

_default_excepthook = sys.excepthook


def _hoomd_sys_excepthook(type, value, traceback):
    """Override Python's excepthook to abort MPI runs."""
    _default_excepthook(type, value, traceback)
    sys.stderr.flush()
    _hoomd.abort_mpi(communicator._current_communicator.cpp_mpi_conf, 1)


sys.excepthook = _hoomd_sys_excepthook
