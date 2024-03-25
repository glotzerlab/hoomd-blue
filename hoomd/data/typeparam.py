# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Implement TypeParameter.

.. invisible-code-block: python

    # This should not be necessary, but without it, the first
    # "skip: next if" fails to skip the code block.
    pass
"""

from collections.abc import MutableMapping


# This class serves as a shim class between TypeParameterDict and the user. This
# class is documented for users and methods are documented for the API
# documentation. For documentation on the mechanics and internal structure go to
# the documentation for hoomd.data.parameterdicts.TypeParameterDict.
class TypeParameter(MutableMapping):
    """Store parameters by type or type pair.

    *Implements the* `collections.abc.MutableMapping` *interface* (excluding
    ``__delitem__``).

    Many operations in HOOMD-blue utilize parameters that depend on the type or
    pairs of types. For example, the Langevin drag coefficient
    (`hoomd.md.methods.Langevin.gamma`) are set by **particle type** and
    Lennard-Jones pair potential parameters (`hoomd.md.pair.LJ.params`) are set
    by **pairs** of particle types.

    `TypeParameter` holds the values of these type (or type pair) dependent
    parameters. It also provides convenience methods for setting defaults
    and multiple parameters on one line.

    Important:
        Parameters for all types (or unordered pairs of types) in the simulation
        state must be defined prior to calling `Simulation.run()`.

    Note:
        `TypeParameter` removes types (or type pairs) not present in the
        simulation state *after* its operation is added to the simulation and
        the simulation has been run for 0 or more steps.

    The examples below use `hoomd.md.methods.Langevin` and `hoomd.md.pair.LJ`
    to demonstrate.

    .. skip: next if(not hoomd.version.md_built)

    .. code-block:: python

        lj = hoomd.md.pair.LJ(nlist=hoomd.md.nlist.Cell(buffer=0.4))
        langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=1.0)
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

    def __setitem__(self, key, value):
        """Set parameters for a given type (or type pair).

        .. rubric:: Examples:

        Index types by name:

        .. skip: next if(not hoomd.version.md_built)

        .. code-block:: python

            langevin.gamma['A'] = 2.0

        Set parameters for multiple types:

        .. skip: next if(not hoomd.version.md_built)

        .. code-block:: python

            langevin.gamma[['B', 'C']] = 3.0

        Set type pair parameters with a tuple of names:

        .. skip: next if(not hoomd.version.md_built)

        .. code-block:: python

            lj.params[('A', 'A')] = dict(epsilon=1.5, sigma=2.0)

        Set parameters for multiple pairs (e.g. ('A', 'B') and ('A', 'C')):

        .. skip: next if(not hoomd.version.md_built)

        .. code-block:: python

            lj.params[('A', ['B', 'C'])] = dict(epsilon=0, sigma=0)

        Set parameters for multiple pairs (e.g. ('B', 'B'), ('B', 'C'), ('C',
        'B'), and ('C', 'C')):

        .. skip: next if(not hoomd.version.md_built)

        .. code-block:: python

            lj.params[(['B', 'C'], ['B', 'C'])] = dict(epsilon=1, sigma=1)

        Note:
            Setting the value for *(a,b)* automatically sets the symmetric
            *(b,a)* parameter to the same value.
        """
        self.param_dict[key] = value

    def __getitem__(self, key):
        """Access parameters by key or keys.

        .. rubric:: Examples:

        .. skip: next if(not hoomd.version.md_built)

        .. code-block:: python

            gamma_A = langevin.gamma['A']

        .. skip: next if(not hoomd.version.md_built)

        .. code-block:: python

            lj_epsilon_AB = lj.params[('A', 'B')]['epsilon']

        .. rubric:: Multiple keys

        When ``key`` denotes multiple pairs (see `__setitem__`), `__getitem__`
        returns multiple items in a dictionary:

        .. skip: next if(not hoomd.version.md_built)

        .. code-block:: python

            gammas = langevin.gamma[['A', 'B']]

        is equivalent to:

        .. skip: next if(not hoomd.version.md_built)

        .. code-block:: python

            gammas = {key: langevin.gamma[key] for key in ['A', 'B']}
        """
        return self.param_dict[key]

    def __delitem__(self, key):
        """__delitem__ is not available for `TypeParameter` objects."""
        raise NotImplementedError("__delitem__ is not defined for this type.")

    def get(self, key, default):
        """Get the value of the key with undefined keys returning default.

        Args:
            key:
                Valid keys specifications (depends on the expected key length).
            default:
                The value to default to if a key is not found in the mapping.

        Returns:
            Returns the parameter value for the key when set. Otherwise, returns
            the provided default.

        .. rubric:: Example:

        .. skip: next if(not hoomd.version.md_built)

        .. code-block:: python

            gamma_D = langevin.gamma.get('D', default=5.0)
        """
        return self.param_dict.get(key, default)

    def setdefault(self, key, default):
        """Set the value for the keys if not already specified.

        Args:
            key: Valid keys specifications (depends on the expected key
                length).
            default: The value to set when the key is not found in
                the mapping.

        .. rubric:: Example

        .. skip: next if(not hoomd.version.md_built)

        .. code-block:: python

            langevin.gamma.setdefault('D', default=5.0)
        """
        self.param_dict.setdefault(key, default)

    def __eq__(self, other):
        """Test for equality.

        .. skip: next if(not hoomd.version.md_built)

        .. code-block:: python

            langevin.gamma == lj.params
        """
        return self.name == other.name and \
            self.type_kind == other.type_kind and \
            self.param_dict == other.param_dict

    @property
    def default(self):
        """The default value of the parameter.

        `TypeParameter` uses the default value for any type (or type pair) in
        the simulation state that is not explicitly set by `__setitem__`
        or `setdefault`.

        .. rubric:: Examples:

        Set a default value:

        .. skip: next if(not hoomd.version.md_built)

        .. code-block:: python

            langevin.gamma.default = 2.0

        When the parameter is a dictionary, set defaults for zero or more
        keys in that dictionary:

        .. skip: next if(not hoomd.version.md_built)

        .. code-block:: python

            lj.params.default = dict(epsilon=0)
            lj.params.default = dict(epsilon=1, sigma=1)
        """
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
        """Convert to a Python `dict`.

        .. rubric:: Example:

        .. skip: next if(not hoomd.version.md_built)

        .. code-block:: python

            plain_dict = lj.params.to_base()
        """
        return self.param_dict.to_base()

    def __iter__(self):
        """Iterate over the keys in the mapping.

        .. rubric:: Example:

        .. skip: next if(not hoomd.version.md_built)

        .. code-block:: python

            for type_pair in lj.params:
                pass
        """
        yield from self.param_dict.keys()

    def __len__(self):
        """Get the number of type parameters in the mapping.

        .. rubric:: Example:

        .. skip: next if(not hoomd.version.md_built)

        .. code-block:: python

            n_type_pairs = len(lj.params)
        """
        return len(self.param_dict)

    def __getstate__(self):
        """Prepare data for pickling."""
        return {k: getattr(self, k) for k in self.__slots__}

    def __setstate__(self, state):
        """Appropriately reset state."""
        for attr, value in state.items():
            setattr(self, attr, value)
