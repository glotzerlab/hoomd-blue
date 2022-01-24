# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Implement TypeParameter."""

from collections.abc import MutableMapping


# This class serves as a shim class between TypeParameterDict and the user. This
# class is documented for users and methods are documented for the API
# documentation. For documentation on the mechanics and internal structure go to
# the documentation for hoomd.data.parameterdicts.TypeParameterDict.
class TypeParameter(MutableMapping):
    """Implement a type based mutable mapping.

    *Implements the* `collections.abc.MutableMapping` *interface
    (* ``__delitem__`` *is disallowed).*

    `TypeParameter` instances extend the base Python mapping interface with
    smart defaults, value/key validation/processing, and advanced indexing. The
    class's intended purpose is to store data per type or per unique
    combinations of type (such as type pairs for `hoomd.md.pair` potentials) of
    a prescribed length.

    .. rubric:: Indexing

    For getting and setting values, multiple index formats are supported. The
    base case is either a string representing the appropriate type or a tuple of
    such strings if multiple types are required per key. This is the exact same
    indexing behavior expect from a Python `dict`, and all functions (barring
    those that delete keys) should function as expected for a Python
    `collections.defaultdict`.

    Two ways to extend this base indexing are supported. First is using an
    iterator of the final key types. This will perform the method for all
    specified types in the iterator.  Likewise, for each item in the final tuple
    (if the expected key type is a tuple of multiple string types), an iterator
    can be used instead of a string which will result in all permutations of
    such iterators in the tuple. Both advanced indexing methods can be combined.

    Note:
        All methods support advanced indexing as well, and behave as one might
        expect. Methods that set values will do so for all keys specified, and
        methods that return values will return values for all keys (within a
        `dict` instance).

    Note:
        Ordering in tuples does not matter. Values in tuples are sorted before
        being stored or queried.

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
        # We do not need to specify epsilon to use new default value when
        # setting
        lj.params[("A", "B")] = {"sigma": 1.0}
        print(lj.params[("A", "B")])
        # {"epsilon": 4.0, "sigma": 1.0}

    Note:
        Before calling `hoomd.Simulation.run` for the `TypeParameter`
        instance's associated simulation, keys are not checked that their types
        exist in the `hoomd.State` object. After calling ``run``, however, all
        such data for non-existent types is removed, and querying or attempting
        to set those keys will result in a ``KeyError``.

    Warning:
        Values after calling `hoomd.Simulation.run` are returned **by copy** not
        reference. Beforehand, values are returned by reference.  For nested
        data structures, this means that directly editing the internal stuctures
        such as a list inside a dict will not be reflected after calling
        `hoomd.Simulation.run`. Examples of nested structure are the union shape
        intergrators in HPMC and the table pair potential in MD.  The
        recommended way to handle mutation for nested structures in general is a
        read-modify-write approach shown below in a code example. Future
        versions of HOOMD-blue version 3 may lift this restriction and allow for
        direct modification of nested structures.

        .. code-block:: python

            union_shape = hoomd.hpmc.integrate.SphereUnion()
            union_shape.shape["union"] = {
                "shapes": [{"diameter": 1.0}, {"diameter": 1.5}],
                "positions": [(0.0, 0.0, 0.0), (-1.0, 0.0, 0.0)]
                }
            # read
            shape_spec = union_shape.shape["union"]
            # modify
            shape_spec["shapes"][1] = {"diameter": 2.0}
            # write
            union_shape.shape["union"] = shape_spec
    """
    __slots__ = ("name", "type_kind", "param_dict")

    def __init__(self, name, type_kind, param_dict):
        self.name = name
        self.type_kind = type_kind
        self.param_dict = param_dict

    def __getattr__(self, attr):
        """Access parameter attributes."""
        if attr in self.__slots__:
            return object.__getattr__(self, attr)
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

    def get(self, key, default):
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
        return self.param_dict.get(key, default)

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

    def _attach(self, cpp_obj, state):
        self.param_dict._attach(cpp_obj, self.name,
                                getattr(state, self.type_kind))
        return self

    def _detach(self):
        self.param_dict._detach()
        return self

    def to_base(self):
        """Convert to a Python `dict`."""
        return self.param_dict.to_base()

    def to_dict(self):
        """Alias for `to_base`."""

    def __iter__(self):
        """Get the keys in the dictionaty."""
        yield from self.param_dict.keys()

    def __len__(self):
        """Return mapping length."""
        return len(self.param_dict)

    def __getstate__(self):
        """Prepare data for pickling."""
        return {k: getattr(self, k) for k in self.__slots__}

    def __setstate__(self, state):
        """Appropriately reset state."""
        for attr, value in state.items():
            setattr(self, attr, value)
