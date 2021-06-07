# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

"""Implement TypeParameter."""

from hoomd.data.parameterdicts import AttachedTypeParameterDict


class TypeParameter:
    """Store parameters per type."""
    __slots__ = ('name', 'type_kind', 'param_dict')

    def __init__(self, name, type_kind, param_dict):
        self.name = name
        self.type_kind = type_kind
        self.param_dict = param_dict

    def __getattr__(self, attr):
        """Access parameter attributes."""
        if attr in self.__slots__:
            return super().__getattr__(attr)
        try:
            return getattr(self.param_dict, attr)
        except AttributeError:
            raise AttributeError("'{}' object has no attribute "
                                 "'{}'".format(type(self), attr))

    def __getitem__(self, key):
        """Access parameters by key."""
        return self.param_dict[key]

    def __setitem__(self, key, value):
        """Set parameters by key."""
        self.param_dict[key] = value

    def __eq__(self, other):
        """Test for equality."""
        return self.name == other.name and \
            self.type_kind == other.type_kind and \
            self.param_dict == other.param_dict

    @property
    def default(self):
        """The default value of the parameter."""
        return self.param_dict.default

    @default.setter
    def default(self, value):
        self.param_dict.default = value

    def _attach(self, cpp_obj, sim):
        self.param_dict = AttachedTypeParameterDict(cpp_obj, self.name,
                                                    self.type_kind,
                                                    self.param_dict, sim)
        return self

    def _detach(self):
        self.param_dict = self.param_dict.to_detached()
        return self

    def to_dict(self):
        """Convert to a Python `dict`."""
        return self.param_dict.to_dict()

    def keys(self):
        """Get the keys in the dictionaty."""
        yield from self.param_dict.keys()

    def __getstate__(self):
        """Prepare data for pickling."""
        state = {
            'name': self.name,
            'type_kind': self.type_kind,
            'param_dict': self.param_dict
        }
        if isinstance(self.param_dict, AttachedTypeParameterDict):
            state['param_dict'] = self.param_dict.to_detached()
        return state

    def __setstate__(self, state):
        """Load pickled data."""
        for attr, value in state.items():
            setattr(self, attr, value)
