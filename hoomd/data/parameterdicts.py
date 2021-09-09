# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

"""Implement parameter dictionaries."""

from abc import abstractmethod
from collections.abc import Mapping, MutableMapping
from itertools import product, combinations_with_replacement
from copy import copy
import numpy as np
from hoomd.util import _to_camel_case, _is_iterable
from hoomd.data.typeconverter import (to_type_converter, RequiredArg,
                                      TypeConverterMapping, OnlyIf, Either)
from hoomd.data.smart_default import (_to_base_defaults, _to_default,
                                      _SmartDefault, _NoDefault)
from hoomd.error import TypeConversionError, IncompleteSpecificationError


def _has_str_elems(obj):
    """Returns True if all elements of iterable are str."""
    return all([isinstance(elem, str) for elem in obj])


def _is_good_iterable(obj):
    """Returns True if object is iterable with respect to types."""
    return _is_iterable(obj) and _has_str_elems(obj)


def _proper_type_return(val):
    """Expects and requires a dictionary with type keys."""
    if len(val) == 0:
        return None
    elif len(val) == 1:
        return list(val.values())[0]
    else:
        return val


def _raise_if_required_arg(value, current_context=()):
    """Ensure that there are no RequiredArgs in value."""

    def _raise_error_with_context(context):
        """Produce a useful error message on missing parameters."""
        context_str = " "
        for c in context:
            if isinstance(c, str):
                context_str = context_str + f"in key {c} "
            elif isinstance(c, int):
                context_str = context_str + f"in index {c} "
        # error is lower cased as this is meant to be caught.
        raise IncompleteSpecificationError(f"value{context_str}is required")

    if value is RequiredArg:
        _raise_error_with_context(current_context)

    if isinstance(value, Mapping):
        for key, item in value.items():
            _raise_if_required_arg(item, current_context + (key,))
    # _is_good_iterable is required over isinstance(value, Sequence) because a
    # str of 1 character is still a sequence and results in infinite recursion.
    elif _is_good_iterable(value):
        for index, item in enumerate(value):
            _raise_if_required_arg(item, current_context + (index,))


class _ValidatedDefaultDict(MutableMapping):
    """Provide support for validating values and multi-type tuple keys.

    The class provides the interface for using `hoomd.data.typeconverter` value
    validation and processing as well as an infrastructure for setting and
    getting multiple mapping values at once.

    In addition, default values for all non-existent keys can be set (similar to
    default dict) using the `hoomd.data.smart_default` logic. This lets partial
    defaults be set.

    The constuctor expects either a single positional arugment defining the type
    validation for keys, or keyword arguments defining the dictionary of
    parameters for each key.

    The keyword argument ``_defaults`` is special and is used to specify default
    values at object construction.

    All keys into this mapping are expected to be str instance is _len_keys is
    one, otherwise a tuple of str instances. For tuples, the tuple is sorted
    first before accessing or setting any data. This is to prevent needing to
    store data for both ``("a", "b")`` and ``("b", "a")`` while preventing the
    user from needing to consider tuple item order.

    Note:
        This class is not directly instantiable due to abstract methods that
        must be written for subclasses: `__len__`, `_single_setitem`,
        `_single_getitem`, and `__iter__` (yield keys).
    """

    def _set_validation_and_defaults(self, *args, **kwargs):
        defaults = kwargs.pop('_defaults', _NoDefault)
        if len(kwargs) != 0 and len(args) != 0:
            raise ValueError("Positional argument(s) and keyword argument(s) "
                             "cannot both be specified.")

        if len(kwargs) == 0 and len(args) == 0:
            raise ValueError("Either a positional or keyword "
                             "argument must be specified.")
        if len(args) > 1:
            raise ValueError("Only one positional argument allowed.")

        if len(kwargs) > 0:
            type_spec = kwargs
        else:
            type_spec = args[0]
        self._type_converter = to_type_converter(type_spec)
        self._default = _to_default(type_spec, defaults)

    @abstractmethod
    def _single_getitem(self, key):
        pass

    def __getitem__(self, keys):
        """Access parameter by key."""
        return self.get(keys, self.default)

    @abstractmethod
    def _single_setitem(self, key, item):
        pass

    def __setitem__(self, keys, item):
        """Set parameter by key."""
        keys = self._yield_keys(keys)
        try:
            validated_value = self._validate_values(item)
        except (TypeConversionError, IncompleteSpecificationError) as err:
            raise err.__class__(f"For types {list(keys)} {str(err)}.") from err
        for key in keys:
            self._single_setitem(key, validated_value)

    def __delitem__(self, key):
        raise NotImplementedError("__delitem__ is not defined for this type.")

    def get(self, keys, default=None):
        """Get values for keys with undefined keys returning default.

        Args:
            keys:
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
        # We implement get manually since __getitem__ will always return a value
        # for a properly formatted key. This explicit method uses the provided
        # default with the benefit that __getitem__ can be defined in terms of
        # get.
        value = {}
        for key in self._yield_keys(keys):
            try:
                value[key] = self._single_getitem(key)
            except KeyError:
                value[key] = default
        return _proper_type_return(value)

    def setdefault(self, keys, default):
        """Set the value for the keys if not already specified.

        Args:
            keys: Valid keys specifications (depends on the expected key
                length).
            default (``any``): The value to default to if a key is not found in
                the mapping.  Must be compatible with the typing specification
                specified on construction.
        """
        self.__setitem__(filter(self.__contains__, self._yield_keys(keys)),
                         default)

    def _validate_values(self, value):
        validated_value = self._type_converter(value)
        # We can check the validated_value is a dict here since if it passed the
        # type validation it is of a form we expect.
        if isinstance(validated_value, dict):
            if isinstance(self._type_converter, TypeConverterMapping):
                expected_keys = set(self._type_converter.keys())
            elif isinstance(self._type_converter.converter, OnlyIf):
                expected_keys = set(self._type_converter.converter.cond.keys())
            elif isinstance(self._type_converter.converter, Either):
                mapping = next(
                    filter(lambda x: isinstance(x, TypeConverterMapping),
                           self._type_converter.converter.specs))
                expected_keys = set(mapping.keys())
            else:
                raise ValueError
            bad_keys = set(validated_value.keys()) - expected_keys
            if len(bad_keys) != 0:
                raise ValueError("Keys must be a subset of available keys. "
                                 "Bad keys are {}".format(bad_keys))
        # update validated_value with the default (specifically to add dict keys
        # that have defaults and were not manually specified).
        if isinstance(self._default, _SmartDefault):
            return self._default(validated_value)
        return validated_value

    def _validate_and_split_key(self, key):
        """Validate key given regardless of key length."""
        if self._len_keys == 1:
            return self._validate_and_split_len_one(key)
        else:
            return self._validate_and_split_len(key)

    def _validate_and_split_len_one(self, key):
        """Validate single type keys.

        Accepted input is a string, and arbitrarily nested iterators that
        culminate in str types.
        """
        if isinstance(key, str):
            return [key]
        elif _is_iterable(key):
            keys = []
            for k in key:
                keys.extend(self._validate_and_split_len_one(k))
            return keys
        else:
            raise KeyError("The key {} is not valid.".format(key))

    def _validate_and_split_len(self, key):
        """Validate all key lengths greater than one, N.

        Valid input is an arbitrarily deep series of iterables that culminate
        in N length tuples, this includes an iterable depth of zero.  The N
        length tuples can contain for each member either a type string or an
        iterable of type strings.
        """
        if isinstance(key, tuple) and len(key) == self._len_keys:
            if any([
                    not _is_good_iterable(v) and not isinstance(v, str)
                    for v in key
            ]):
                raise KeyError("The key {} is not valid.".format(key))
            # convert str to single item list for proper enumeration using
            # product
            key_types_list = [[v] if isinstance(v, str) else v for v in key]
            return list(product(*key_types_list))
        elif _is_iterable(key):
            keys = []
            for k in key:
                keys.extend(self._validate_and_split_len(k))
            return keys
        else:
            raise KeyError("The key {} is not valid.".format(key))

    def _yield_keys(self, key):
        """Returns the generated keys in proper sorted order.

        The order is necessary so ('A', 'B') is equivalent to ('B', A').
        """
        if self._len_keys > 1:
            keys = self._validate_and_split_key(key)
            for key in keys:
                yield tuple(sorted(list(key)))
        else:
            yield from self._validate_and_split_key(key)

    def __eq__(self, other):
        if not isinstance(other, _ValidatedDefaultDict):
            return NotImplemented
        if self.default != other.default:
            return False
        if set(self.keys()) != set(other.keys()):
            return False
        if isinstance(self.default, dict):
            return all(
                np.all(self[type_][key] == other[type_][key])
                for type_ in self
                for key in self[type_])
        return all(np.all(self[type_] == other[type_]) for type_ in self)

    @property
    def default(self):
        if isinstance(self._default, _SmartDefault):
            return self._default.to_base()
        else:
            return copy(self._default)

    @default.setter
    def default(self, new_default):
        new_default = self._type_converter(new_default)
        if isinstance(self._default, _SmartDefault):
            new_default = self._default(new_default)
        if isinstance(new_default, dict):
            keys = set(self._default.keys())
            provided_keys = set(new_default.keys())
            if keys.intersection(provided_keys) != provided_keys:
                raise ValueError("New default must a subset of current keys.")
        self._default = _to_default(new_default)


class TypeParameterDict(_ValidatedDefaultDict):
    """Type parameter dictionary."""

    def __init__(self, *args, len_keys, **kwargs):

        # Validate proper key constraint
        if len_keys < 1 or len_keys != int(len_keys):
            raise ValueError("len_keys must be a positive integer.")
        self._len_keys = len_keys
        self._set_validation_and_defaults(*args, **kwargs)
        self._dict = {}

    def _single_getitem(self, key):
        """Access parameter by key."""
        try:
            return self._dict[key]
        except KeyError:
            return self.default

    def _single_setitem(self, key, item):
        """Set parameter by key."""
        self._dict[key] = item

    def __iter__(self):
        """Get the keys in the mapping."""
        if self._len_keys == 1:
            yield from self._dict.keys()
        else:
            for key in self._dict.keys():
                yield tuple(sorted(list(key)))

    def __len__(self):
        """Return mapping length."""
        return len(self._dict)

    def to_dict(self):
        """Convert to a `dict`."""
        return self._dict


class AttachedTypeParameterDict(_ValidatedDefaultDict):
    """Parameter dictionary synchronized with a C++ class.

    This class serves as the "attached" version of the `TypeParameterDict`. The
    class performs the same indexing and mutation options as
    `TypeParameterDict`, but only allows querying for keys that match the actual
    types of the simulation it is attached to.

    The interface expects the passed in C++ object to have a getter and setter
    that follow the camel case style version of ``param_name``. Likewise
    type_kind must be a str of a valid attribute to query types from the state.

    Args:
        cpp_obj:
            A pybind11 wrapped C++ object to set and get the type parameters
            from.
        param_name (str):
            A snake case parameter name (handled automatically by
            ``TypeParameter``) that when changed to camel case prefixed by get
            or set is the str name for the pybind11 exported getter and setter.
        type_kind (str):
            The str name of the attribute to query the parent simulation's state
            for existent types.
        type_param_dict (TypeParameterDict):
            The `TypeParameterDict` to convert to the "attached" version.
        sim (hoomd.Simulation):
            The simulation to attach to.

    Note:
        This class should not be directly instantiated even by developers, but
        the `hoomd.data.type_param.TypeParameter` class should be used to
        automatically handle this in conjunction with
        `hoomd.operation._BaseHOOMDObject` subclasses.
    """

    def __init__(self, cpp_obj, param_name, types, type_param_dict):
        # store info to communicate with c++
        self._cpp_obj = cpp_obj
        self._setter = "set" + _to_camel_case(param_name)
        self._getter = "get" + _to_camel_case(param_name)
        self._len_keys = type_param_dict._len_keys
        self._type_keys = self._compute_type_keys(types)
        # Get default data
        self._default = type_param_dict._default
        self._type_converter = type_param_dict._type_converter
        # add all types to c++
        for key in self:
            parameter = type_param_dict._single_getitem(key)
            try:
                _raise_if_required_arg(parameter)
            except IncompleteSpecificationError as err:
                raise IncompleteSpecificationError(f"for key {key} {str(err)}")
            self._single_setitem(key, parameter)

    def to_detached(self):
        """Convert to a detached parameter dict."""
        if isinstance(self.default, dict):
            type_param_dict = TypeParameterDict(**self.default,
                                                len_keys=self._len_keys)
        else:
            type_param_dict = TypeParameterDict(self.default,
                                                len_keys=self._len_keys)
        type_param_dict._type_converter = self._type_converter
        for key in self:
            type_param_dict[key] = self[key]
        return type_param_dict

    def _single_getitem(self, key):
        """Access parameter by key."""
        return getattr(self._cpp_obj, self._getter)(key)

    def _single_setitem(self, key, item):
        """Set parameter by key."""
        getattr(self._cpp_obj, self._setter)(key, item)

    def _yield_keys(self, key):
        """Includes key check for existing simulation keys.

        Overwritting this means that __getitem__ and __setitem__ plus any
        methods that rely on them will error properly even if we don't check for
        the key's existence there.
        """
        for key in super()._yield_keys(key):
            if key not in self._type_keys:
                raise KeyError("Type {} does not exist in the "
                               "system.".format(key))
            else:
                yield key

    def _validate_values(self, val):
        val = super()._validate_values(val)
        _raise_if_required_arg(val)
        return val

    def _compute_type_keys(self, types):
        """Compute valid type keys from given types.

        We store types as a set since set iteration for ~50 items is marginally
        slower to iterate over than a list, but multiple times faster to check
        for contained values.
        """
        if self._len_keys == 1:
            return set(types)
        else:
            return {
                tuple(sorted(key))
                for key in combinations_with_replacement(types, self._len_keys)
            }

    def __iter__(self):
        """Iterate through mapping keys."""
        yield from self._type_keys

    def __len__(self):
        """Return mapping length."""
        return len(self._type_keys)

    def to_dict(self):
        """Convert to a `dict`."""
        rtn_dict = {}
        for key in self:
            rtn_dict[key] = getattr(self._cpp_obj, self._getter)(key)
        return rtn_dict


class ParameterDict(MutableMapping):
    """Parameter dictionary."""

    def __init__(self, _defaults=_NoDefault, **kwargs):
        self._type_converter = to_type_converter(kwargs)
        self._dict = {**_to_base_defaults(kwargs, _defaults)}

    def __setitem__(self, key, value):
        """Set parameter by key."""
        if key not in self._type_converter.keys():
            self._dict[key] = value
            self._type_converter[key] = to_type_converter(value)
        else:
            self._dict[key] = self._type_converter[key](value)

    def __getitem__(self, key):
        """Access parameter by key."""
        return self._dict[key]

    def __delitem__(self, key):
        """Remove parameter by key."""
        del self._dict[key]
        del self._type_converter[key]

    def __iter__(self):
        """Iterate over keys."""
        yield from self._dict

    def __len__(self):
        """int: The number of keys."""
        return len(self._dict)

    def __eq__(self, other):
        """Equality between ParameterDict objects."""
        if not isinstance(other, ParameterDict):
            return NotImplemented
        return (set(self.keys()) == set(other.keys()) and np.all(
            [np.all(self[key] == other[key]) for key in self.keys()]))

    def update(self, other):
        """Add keys and values to the dictionary."""
        if isinstance(other, ParameterDict):
            for key, value in other.items():
                self._type_converter[key] = other._type_converter[key]
                self._dict[key] = value
        else:
            for key, value in other.items():
                self[key] = value
