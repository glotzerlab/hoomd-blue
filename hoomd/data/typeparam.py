# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

"""Implement TypeParameter."""

from collections.abc import MutableMapping
from hoomd.data.parameterdicts import AttachedTypeParameterDict


class TypeParameter(MutableMapping):
    """Store parameters per type.

    `TypeParameter` instance allow for the setting of per-type parameters with
    smart default and value validation/processing abilities.

    .. rubric:: Indexing

    For getting and setting values, multiple index formats are supported. The
    base case is either a string representing the appropriate type or a tuple of
    such strings if multiple types are required per key. Extending this an
    iterator of the final types will retreive or set all keys in the iterator.
    Likewise, for each item in the final tuple (if the expected key type is a
    tuple of multiple string types), an iterator can be used instead of a string
    which will result in all permutations of such iterators in the tuple. Both
    advanced intexing methods can be combined.

    Note:
        Ordering in tuples does not matter.

    Below are some example indexing values for single and multiple key indexing.

    .. code-block:: python

        # "A", "B", "C"
        ["A", "B", "C"]
        # ("A", "B")
        ("A", "B")
        # ("A", "B") and ("B", "C")
        [("A", "B"), ("B", "C")]
        # ("A", "B"), ("A", "C"), and ("A", "D")
        ("A", ["B", "C", "D"])


    .. rubric:: Defaults and setting values

    `TypeParameter` instances have default values that can be accessed via
    ``default`` which will be used for all types not defined. In addition, when
    the type parameter expects a `dict`-like object, the default will be updated
    with the set value. This means that values that have defaults do not need to
    be explicitly specified.

    An example of "smart"-setting using the MD LJ potential,

    .. code-block:: python

        lj = hoomd.md.pair.LJ(nlist=hoomd.md.nlist.Cell())
        # params is a TypeParameter object.
        # We set epsilon to have a default but sigma is still required
        lj.params.default = {"epsilon": 4}
        print(lj.params.default)
        # {"epsilon": 4.0, "sigma": hoomd.data.typeconverter.RequiredArg}
        # We do no need to specify epsilon to use new default value when
        # setting
        lj.params[("A", "B")] = {"sigma": 1.0}
        print(lj.params[("A", "B")])
        # {"epsilon": 4.0, "sigma": 1.0}

    Note:
        Setting values for fictitious but valid types will not trigger an error
        before calling `hoomd.Simulation.run`, but attempts to access that data
        or set values for fictitious types afterwards will result in
        ``KeyError`` exceptions.

    Warning:
        For nested data structures, editing the internal stuctures such as a
        list inside a dict will not be reflected after calling
        `hoomd.Simulation.run`. Doing so even before a call to ``run`` is
        currently considered to be an anti-pattern. This restriction is planned
        to be lifted in the future. Examples where this nested structure appears
        are the union shape intergrators in HPMC and the table pair potential in
        MD.
    """
    __slots__ = ("name", "type_kind", "param_dict")

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

    def __delitem__(self, key):
        """__delitem__ is not available for `TypeParameter` objects."""
        raise NotImplementedError("__delitem__ is not defined for this type.")

    def get(self, key, value):
        """Get values for keys with undefined keys returning default.

        Args:
            key:
                Valid keys specifications (depends on the expected key length).
            default (``any``, optional):
                The value to default to if a key is not found in the mapping.
                If not set, the value defaults to the mapping's default.

        Returns:
            values:
                Returns a dict of the values for the keys asked for if multiple
                keys were specified; otherwise, returns the value for the single
                key.
        """
        return self.param_dict.get(key, value)

    def setdefault(self, key, default):
        """Set the value for the keys if not already specified.

        Args:
            key: Valid keys specifications (depends on the expected key
                length).
            default (``any``): The value to default to if a key is not found in
                the mapping.  Must be compatible with the typing specification
                specified on construction.
        """
        self.param_dict.setdefault(key, default)

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
        self.param_dict = AttachedTypeParameterDict(
            cpp_obj, self.name, getattr(sim.state, self.type_kind),
            self.param_dict)
        return self

    def _detach(self):
        self.param_dict = self.param_dict.to_detached()
        return self

    def to_dict(self):
        """Convert to a Python `dict`."""
        return self.param_dict.to_dict()

    def __iter__(self):
        """Get the keys in the dictionaty."""
        yield from self.param_dict.keys()

    def __len__(self):
        """Return mapping length."""
        return len(self.param_dict)

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
