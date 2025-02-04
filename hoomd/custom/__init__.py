# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Custom operations.

`CustomOperation` provides a mechanism for users to insert Python code
via an `Action` subclass that executes during the simulation's run loop.
Use this to prototype new simulation methods in Python, analyze the system state
while the simulation progresses, or write output to custom file formats.

See Also:
    `hoomd.tune.CustomTuner`

    `hoomd.update.CustomUpdater`

    `hoomd.write.CustomWriter`

    `hoomd.md.force.Custom`
"""

from hoomd.custom.custom_action import Action, _InternalAction
from hoomd.custom.custom_operation import (CustomOperation,
                                           _InternalCustomOperation)
