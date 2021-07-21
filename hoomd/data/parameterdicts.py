# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

"""Implement parameter dictionaries."""

from collections.abc import MutableMapping
from itertools import product, combinations_with_replacement
from copy import copy
import numpy as np
from hoomd.util import _to_camel_case, _is_iterable
from hoomd.data.typeconverter import (to_type_converter, RequiredArg,
                                      TypeConverterMapping, OnlyIf, Either)
from hoomd.data.smart_default import (_to_base_defaults, _to_default,
                                      _SmartDefault, _NoDefault)
from hoomd.error import TypeConversionError


def has_str_elems(obj):
    """Returns True if all elements of iterable are str."""
    return all([isinstance(elem, str) for elem in obj])


def is_good_iterable(obj):
    """Returns True if object is iterable with respect to types."""
    return _is_iterable(obj) and has_str_elems(obj)


def proper_type_return(val):
    """Expects and requires a dictionary with type keys."""
    if len(val) == 0:
        return None
    elif len(val) == 1:
        return list(val.values())[0]
    else:
        return val


class _ValidatedDefaultDict:

    def __init__(self, *args, **kwargs):
        _defaults = kwargs.pop('_defaults', _NoDefault)
        if len(kwargs) != 0 and len(args) != 0:
            raise ValueError("An unnamed argument and keyword arguments "
                             "cannot both be specified.")

        if len(kwargs) == 0 and len(args) == 0:
            raise ValueError("Either an unnamed argument or keyword "
                             "arguments must be specified.")
        if len(args) > 1:
            raise ValueError("Only one unnamed argument allowed.")
        if len(kwargs) > 0:
            default_arg = kwargs
        else:
            default_arg = args[0]
        self._type_converter = to_type_converter(default_arg)
        self._default = _to_default(default_arg, _defaults)

    def _validate_values(self, val):
        val = self._type_converter(val)
        if isinstance(val, dict):
            if isinstance(self._type_converter, TypeConverterMapping):
                dft_keys = set(self._type_converter.keys())
            elif isinstance(self._type_converter.converter, OnlyIf):
                dft_keys = set(self._type_converter.converter.cond.keys())
            elif isinstance(self._type_converter.converter, Either):
                mapping = next(
                    filter(lambda x: isinstance(x, TypeConverterMapping),
                           self._type_converter.converter.specs))
                dft_keys = set(mapping.keys())
            else:
                raise ValueError
            bad_keys = set(val.keys()) - dft_keys
            if len(bad_keys) != 0:
                raise ValueError("Keys must be a subset of available keys. "
                                 "Bad keys are {}".format(bad_keys))
        if isinstance(self._default, _SmartDefault):
            val = self._default(val)
        return val

    # Add function to validate dictionary keys' value types as well
    # Could follow current model on the args based type checking

    def _validate_and_split_key(self, key):
        """Validate key given regardless of key length."""
        if self._len_keys == 1:
            return self._validate_and_split_len_one(key)
        else:
            return self._validate_and_split_len(key)

    def _validate_and_split_len_one(self, key):
        """Validate single type keys.

        Accepted input is a type string, and arbitrarily nested interators that
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
        in N length tuples, this includes an iterable depth of zero. The N
        length tuples can contain for each member either a type string or an
        iterable of type strings.
        """
        if isinstance(key, tuple) and len(key) == self._len_keys:
            fst, snd = key
            if any([
                    not is_good_iterable(v) and not isinstance(v, str)
                    for v in key
            ]):
                raise KeyError("The key {} is not valid.".format(key))
            key = list(key)
            for ind in range(len(key)):
                if isinstance(key[ind], str):
                    key[ind] = [key[ind]]
            return list(product(*key))
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
        return (self.default == other.default
                and set(self.keys()) == set(other.keys())
                and np.all(self[key] == other[key] for key in self.keys()))

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
        super().__init__(*args, **kwargs)
        self._dict = dict()

    def __getitem__(self, key):
        """Access parameter by key."""
        vals = dict()
        for key in self._yield_keys(key):
            try:
                vals[key] = self._dict[key]
            except KeyError:
                vals[key] = self.default
        return proper_type_return(vals)

    def __setitem__(self, key, val):
        """Set parameter by key."""
        keys = self._yield_keys(key)
        try:
            val = self._validate_values(val)
        except TypeConversionError as err:
            raise TypeConversionError("For types {}, error {}.".format(
                list(keys), str(err)))
        for key in keys:
            self._dict[key] = val

    def keys(self):
        """Get the keys in the dictionary."""
        if self._len_keys == 1:
            yield from self._dict.keys()
        else:
            for key in self._dict.keys():
                yield tuple(sorted(list(key)))

    def to_dict(self):
        """Convert to a `dict`."""
        return self._dict


class AttachedTypeParameterDict(_ValidatedDefaultDict):
    """Parameter dictionary synchronized with a C++ class."""

    def __init__(self, cpp_obj, param_name, type_kind, type_param_dict, sim):
        # store info to communicate with c++
        self._cpp_obj = cpp_obj
        self._param_name = param_name
        self._sim = sim
        self._type_kind = type_kind
        self._len_keys = type_param_dict._len_keys
        # Get default data
        self._default = type_param_dict._default
        self._type_converter = type_param_dict._type_converter
        # add all types to c++
        for key in self.keys():
            self[key] = type_param_dict[key]

    def to_detached(self):
        """Convert to a detached parameter dict."""
        if isinstance(self.default, dict):
            type_param_dict = TypeParameterDict(**self.default,
                                                len_keys=self._len_keys)
        else:
            type_param_dict = TypeParameterDict(self.default,
                                                len_keys=self._len_keys)
        type_param_dict._type_converter = self._type_converter
        for key in self.keys():
            type_param_dict[key] = self[key]
        return type_param_dict

    def __getitem__(self, key):
        """Access parameter by key."""
        vals = dict()
        for key in self._yield_keys(key):
            vals[key] = getattr(self._cpp_obj, self._getter)(key)
        return proper_type_return(vals)

    def __setitem__(self, key, val):
        """Set parameter by key."""
        keys = self._yield_keys(key)
        try:
            val = self._validate_values(val)
        except TypeConversionError as err:
            raise TypeConversionError("For types {}, error {}.".format(
                list(keys), str(err)))

        for key in keys:
            getattr(self._cpp_obj, self._setter)(key, val)

    def _yield_keys(self, key):
        """Includes key check for existing simulation keys."""
        curr_keys = self.keys()
        for key in super()._yield_keys(key):
            if key not in curr_keys:
                raise KeyError("Type {} does not exist in the "
                               "system.".format(key))
            else:
                yield key

    def _validate_values(self, val):
        val = super()._validate_values(val)
        if isinstance(val, dict):
            not_set_keys = []
            for k, v in val.items():
                if v is RequiredArg:
                    not_set_keys.append(k)
            if not_set_keys != []:
                raise ValueError("{} were not set.".format(not_set_keys))
        return val

    @property
    def _setter(self):
        return 'set' + _to_camel_case(self._param_name)

    @property
    def _getter(self):
        return 'get' + _to_camel_case(self._param_name)

    def keys(self):
        """Iterate through the keys."""
        single_keys = getattr(self._sim.state, self._type_kind)
        if self._len_keys == 1:
            yield from single_keys
        else:
            for key in combinations_with_replacement(single_keys,
                                                     self._len_keys):
                yield tuple(sorted(list(key)))

    def to_dict(self):
        """Convert to a `dict`."""
        rtn_dict = dict()
        for key in self.keys():
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
            np.all(self[key] == other[key]) for key in self.keys()))

    def update(self, other):
        """Add keys and values to the dictionary."""
        if isinstance(other, ParameterDict):
            for key, value in other.items():
                self._type_converter[key] = other._type_converter[key]
                self._dict[key] = value
        else:
            for key, value in other.items():
                self[key] = value
