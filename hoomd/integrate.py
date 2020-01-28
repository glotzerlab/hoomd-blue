# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

# Maintainer: joaander / All Developers are free to add commands for new
# features

from hoomd.meta import _Operation
from hoomd.syncedlist import SyncedList
from hoomd.md.methods import Method
from hoomd.md.force import Force
from hoomd.md.constrain import ConstraintForce


class _BaseIntegrator(_Operation):
    def attach(self, simulation):
        simulation._cpp_sys.setIntegrator(self._cpp_obj)
        super().attach(simulation)


class _DynamicIntegrator(_BaseIntegrator):
    def __init__(self, forces, constraint_forces, methods):
        self.forces = SyncedList(lambda x: isinstance(Force),
                                 to_synced_list=lambda x: x._cpp_obj,
                                 iterable=forces)

        self.constraints = SyncedList(lambda x: isinstance(ConstraintForce),
                                      to_synced_list=lambda x: x._cpp_obj,
                                      iterable=forces)

        self.methods = SyncedList(lambda x: isinstance(Method),
                                  to_synced_list=lambda x: x._cpp_obj,
                                  iterable=methods)

    def attach(self, simulation):
        self.forces.attach(simulation, self._cpp_obj.forces)
        self.constraints.attach(simulation, self._cpp_obj.constraints)
        self.methods.attach(simulation, self._cpp_obj.methods)
        super().attach(simulation)
