# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

"""Access simulation state data directly."""

from abc import abstractmethod
import hoomd


class ForceLocalAccessBase(hoomd.data.local_access._LocalAccess):
    """Class for accessing HOOMD-Blue custom force data.

    Attributes:
        force ((N_particles, 3) `hoomd.data.array` of ``float``):
            Local force data. :math:`[\\mathrm{force}]`
        potential_energy ((N_particles,) `hoomd.data.array` of ``float``):
            Local potential energy data. :math:`[\\mathrm{energy}]`
        torque ((N_particles, 3) `hoomd.data.array` of ``float``):
            Local torque data. :math:`[\\mathrm{force} \\cdot \\mathrm{length}]`
        virial ((N_particles, 6) `hoomd.data.array` of ``float``):
            Local virial data. :math:`[\\mathrm{energy}]`
        rtag ((N_particles_global) `hoomd.data.array` of ``int``):
            The reverse tag of a particle. This means that the value
            ``force_data.rtag[0]`` represents the current index accessing data
            for the particle with tag 0.
    """
    __slots__ = ('_entered', '_accessed_fields', '_cpp_obj', '_force_obj')

    @property
    @abstractmethod
    def _cpp_cls(self):
        pass

    _fields = {
        'force': 'getForce',
        'potential_energy': 'getPotentialEnergy',
        'torque': 'getTorque',
        'virial': 'getVirial'
    }

    def __init__(self, force_obj):
        super().__init__()
        self._force_obj = force_obj
        self._cpp_obj = self._cpp_cls(force_obj._cpp_obj)

    def __enter__(self):
        """Called upon entering a context manager."""
        self._force_obj._in_context_manager = True
        self._enter()
        return self

    def __exit__(self, type, value, traceback):
        """Called upon exiting a context manager."""
        self._force_obj._in_context_manager = False
        self._exit()
