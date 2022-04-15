# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

from collections.abc import Mapping

import hoomd.data
from hoomd.operation import _HOOMDBaseObject


class AlchemicalPairDOF(Mapping):
    """A read-only mapping of alchemical particles accessed by type."""

    def __init__(self, name, pair_instance, dof_cls):
        """Create an `AlchemicalPairDOF` object.

        Warning:
            Should not be instantiated by users.
        """
        self._name = name
        self._dof_cls = dof_cls
        self._pair_instance = pair_instance
        self._indexer = hoomd.data.parameterdicts._SmartTypeIndexer(2)
        self._data = {}

    def __getitem__(self, key):
        """Get the alchemical particle for the given type pair."""
        items = {}
        for k in self._indexer(key):
            if k not in self._data:
                self._data[k] = self._dof_cls(self._pair_instance, self._name,
                                              k)
            items[k] = self._data[k]
        if len(items) == 1:
            return items.popitem()[1]
        return items

    def __iter__(self):
        """Iterate over keys."""
        yield from self._data

    def __contains__(self, key):
        """Return whether the key is in the mapping."""
        keys = list(self._indexer(key))
        if len(keys) == 1:
            return keys[0] in self._data
        return [k in self._data for k in keys]

    def __len__(self):
        """Get the length of the mapping."""
        return len(self._data)

    def _attach(self, types):
        self._indexer.valid_types = types
        for key in self:
            if not self._indexer.are_valid_types(key):
                raise RuntimeError(
                    f"Alchemical DOF ({self._name}) for non-existent type pair "
                    f"{key} was accessed.")

    def _detach(self):
        self._indexer.valid_types = None


class _AlchemicalPairForce(_HOOMDBaseObject):
    _alchemical_dofs = []
    _dof_cls = None

    def __init__(self):
        self._set_alchemical_parameters()

    def _set_alchemical_parameters(self):
        self._alchemical_params = {}
        for dof in self._alchemical_dofs:
            self._alchemical_params[dof] = AlchemicalPairDOF(
                name=dof, pair_instance=self, dof_cls=self._dof_cls)

    def _setattr_hook(self, attr, value):
        if attr in self._alchemical_dofs:
            raise RuntimeError(f"{attr} is not settable.")
        super()._setattr_hook(attr, value)

    def _getattr_hook(self, attr):
        if attr in self._alchemical_dofs:
            return self._alchemical_params[attr]
        return super()._getattr_hook(attr)
