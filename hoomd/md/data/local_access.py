# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Access simulation state data directly."""

from abc import abstractmethod
import hoomd


class _ForceLocalAccessBase(hoomd.data.local_access._LocalAccess):
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
        self._force_obj._in_context_manager = True
        self._enter()
        return self

    def __exit__(self, type, value, traceback):
        self._force_obj._in_context_manager = False
        self._exit()


class _NeighborListLocalAccessBase(hoomd.data.local_access._LocalAccess):
    __slots__ = ('_entered', '_accessed_fields', '_cpp_obj', '_nlist_obj')

    # This should work without having to comment out the @abstractmethod
    # decorator, but it just doesn't... Once commented out, it is easily able to
    # find the _cpp_cls attribute.
    @property
    # @abstractmethod  # This has to be commented out for the code to work
    def _cpp_class(self):
        pass

    _fields = {
        'head_list': 'getHeadList',
        'n_neigh': 'getNNeigh',
        'nlist': 'getNList'
    }

    def __init__(self, nlist_obj):
        super().__init__()
        self._nlist_obj = nlist_obj
        self._cpp_obj = self._cpp_cls(nlist_obj._cpp_obj)

    def __enter__(self):
        self._nlist_obj._in_context_manager = True
        self._enter()
        return self

    def __exit__(self, type, value, traceback):
        self._nlist_obj._in_context_manager = False
        self._exit()
