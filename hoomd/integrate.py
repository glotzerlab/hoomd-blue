# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

"""Implement BaseIntegrator."""

from hoomd.operation import Operation
from hoomd.data import syncedlist
from hoomd.wall import (_Base_Wall,Sphere)




class BaseIntegrator(Operation):
    """Defines the base for all HOOMD-blue integrators.

    An integrator in HOOMD-blue is the primary operation that drives a
    simulation state forward. In `hoomd.hpmc`, integrators perform particle
    based Monte Carlo moves. In `hoomd.md`, the `hoomd.md.Integrator` class
    organizes the forces, equations of motion, and other factors of the given
    simulation.
    """
    
    def __init__(self):
        self._walls = syncedlist.SyncedList(
            _Base_Wall,
            to_synced_list=self.individual_type_list_magic_function)
        self._sphere_walls = syncedlist.SyncedList(
            Sphere,
            syncedlist._PartialGetAttr('_cpp_obj'))
        self._wall_index={Sphere:[]}
        
    def individual_type_list_magic_function(self,wall):
        if isinstance(wall, Sphere):
            assert not self._sphere_walls.contains(wall)
            i = self._walls.index(wall)
            self.sphere_walls.append(wall)
            self._wall_index
        self._attach_wall(wall)
        # do all the indexing and syncing

        
    
    def _attach(self):
        self._simulation._cpp_sys.setIntegrator(self._cpp_obj)
        super()._attach()
        
        self._attach_walls()

        # The integrator has changed, update the number of DOF in all groups
        self._simulation.state.update_group_dof()
