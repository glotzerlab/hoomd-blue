# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Writers.

Writers write the state of the simulation, logger quantities, or calculated
results to output files or streams:

* `GSD` and `DCD` save the simulation trajectory to a file.
* Combine `GSD` with a `hoomd.logging.Logger` to save system properties or
  per-particle calculated results.
* Use `Table` to display the status of the simulation periodically to standard
  out.
* Implement custom output formats with `CustomWriter`.

Writers do not modify the system state.

Tip:
    `OVITO <https://www.ovito.org/>`_ has native support for GSD files,
    including logged per-particle array quantities and particle shapes.

See Also:
    Tutorial: :doc:`tutorial/00-Introducing-HOOMD-blue/00-index`

    Tutorial: :doc:`tutorial/02-Logging/00-index`
"""

from hoomd.write.custom_writer import CustomWriter
from hoomd.write.gsd import GSD
from hoomd.write.gsd_burst import Burst
from hoomd.write.dcd import DCD
from hoomd.write.table import Table
try:
    from hoomd.write.hdf5 import HDF5Logger
except ImportError:
    pass
