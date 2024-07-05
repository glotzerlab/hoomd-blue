# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Implement parameter dictionaries."""

from abc import abstractmethod
from collections.abc import Mapping, MutableMapping
from copy import copy
from itertools import product, combinations_with_replacement

import numpy as np

from hoomd.data.collections import (_HOOMDSyncedCollection, _to_hoomd_data,
                                    _to_base)
from hoomd.error import MutabilityError, TypeConversionError
from hoomd.util import _to_camel_case, _is_iterable
from hoomd.data.typeconverter import (to_type_converter, RequiredArg,
                                      TypeConverterMapping, OnlyIf, Either)
from hoomd.data.smart_default import (_to_base_defaults, _to_default,
                                      _SmartDefault, _NoDefault)
from hoomd.error import IncompleteSpecificationError


def _has_str_elems(obj):
    """Returns True if all elements of iterable are str."""
    return all([isinstance(elem, str) for elem in obj])


def _is_key_iterable(obj):
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
                context_str = context_str + f"for key {c} "
            elif isinstance(c, int):
                context_str = context_str + f"in index {c} "
        # error is lower cased as this is meant to be caught.
        raise IncompleteSpecificationError(f"value{context_str}is required")

    if value is RequiredArg:
        _raise_error_with_context(current_context)

    if isinstance(value, Mapping):
        for key, item in value.items():
            _raise_if_required_arg(item, current_context + (key,))
    # _is_iterable is required over isinstance(value, Sequence) because a
    # str of 1 character is still a sequence and results in infinite recursion.
    elif _is_iterable(value):
        # Special case where a sequence type spec was given no value.
        if len(value) == 1 and value[0] is RequiredArg:
            _raise_error_with_context(current_context)
        for index, item in enumerate(value):
            _raise_if_required_arg(item, current_context + (index,))


class _SmartTypeIndexer:

    def __init__(self, len_key, valid_types=None):
        self.len_key = len_key
        self._valid_types = valid_types

    def __call__(self, key):
        """Returns the generated keys filtered by ``valid_keys`` if set."""
        if self.valid_types is None:
            yield from self.raw_yield(key)
        else:
            for k in self.raw_yield(key):
                if not self.are_valid_types(k):
                    raise KeyError(
                        f"Key {k} from key {key} is not of valid types.")
                yield k

    def raw_yield(self, key):
        """Yield valid keys without filtering by valid keys.

        The keys are internally ordered. The order is necessary so ('A', 'B') is
        equivalent to ('B', A').
        """
        if self.len_key > 1:
            keys = self.validate_and_split_index(key)
            for key in keys:
                yield tuple(sorted(list(key)))
        else:
            yield from self.validate_and_split_index(key)

    def validate_and_split_index(self, key):
        """Validate key given regardless of key length."""
        if self.len_key == 1:
            return self.validate_and_split_len_one(key)
        else:
            return self.validate_and_split_len(key)

    def validate_and_split_len_one(self, key):
        """Validate single type keys.

        Accepted input is a string, and arbitrarily nested iterators that
        culminate in str types.
        """
        if isinstance(key, str):
            return [key]
        elif _is_iterable(key):
            keys = []
            for k in key:
                keys.extend(self.validate_and_split_len_one(k))
            return keys
        else:
            raise KeyError("The key {} is not valid.".format(key))

    def validate_and_split_len(self, key):
        """Validate all key lengths greater than one, N.

        Valid input is an arbitrarily deep series of iterables that culminate
        in N length tuples, this includes an iterable depth of zero.  The N
        length tuples can contain for each member either a type string or an
        iterable of type strings.
        """
        if isinstance(key, tuple) and len(key) == self.len_key:
            if any([
                    not _is_key_iterable(v) and not isinstance(v, str)
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
                keys.extend(self.validate_and_split_len(k))
            return keys
        else:
            raise KeyError("The key {} is not valid.".format(key))

    @property
    def valid_types(self):
        return self._valid_types

    @valid_types.setter
    def valid_types(self, types):
        self._valid_types = types if types is None else set(types)

    def are_valid_types(self, key):
        if self._valid_types is None:
            return True
        if self.len_key == 1:
            return key in self._valid_types
        # Multi-type key
        return all(type_ in self._valid_types for type_ in key)

    def yield_all_keys(self):
        if self._valid_types is None:
            yield from ()
        elif self.len_key == 1:
            yield from self._valid_types
        elif isinstance(self._valid_types, set):
            yield from (tuple(sorted(key))
                        for key in combinations_with_replacement(
                            self._valid_types, self.len_key))


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

    All keys into this mapping are expected to be str instance if the passed
    len_key is one, otherwise a tuple of str instances. For tuples, the tuple
    is sorted first before accessing or setting any data. This is to prevent
    needing to store data for both ``("a", "b")`` and ``("b", "a")`` while
    preventing the user from needing to consider tuple item order.

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
        self.setdefault(keys, self.default)
        return self.get(keys)

    @abstractmethod
    def _single_setitem(self, key, item):
        pass

    def __setitem__(self, keys, item):
        """Set parameter by key."""
        keys = self._indexer(keys)
        try:
            validated_value = self._validate_values(item)
        except ValueError as err:
            raise TypeConversionError(
                f"For types {list(keys)}: {str(err)}.") from err
        for key in keys:
            self._single_setitem(key, validated_value)

    def __delitem__(self, key):
        raise NotImplementedError("__delitem__ is not defined for this type.")

    def __contains__(self, key):
        try:
            keys = list(self._indexer.raw_yield(key))
        except KeyError:
            return False
        if self._attached:
            if len(keys) == 1:
                return self._indexer.are_valid_types(keys[0])
            return [self._indexer.are_valid_types(k) for k in keys]
        if len(keys) == 1:
            return keys[0] in self._dict
        return [key in self._dict for key in keys]

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
        # We shouldn't error here regardless of key (assuming it is well formed,
        # and get doesn't error on non-existent key. _SmartTypeIndexer.raw_yield
        # does not raise an exception on non-existent keys.
        for key in self._indexer.raw_yield(keys):
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
        set_keys = [key for key in self._indexer(keys) if key not in self]
        if len(set_keys) > 0:
            self.__setitem__(set_keys, default)

    def _validate_values(self, value):
        validated_value = self._type_converter(value)
        # We can check the validated_value is a dict here since if it passed the
        # type validation it is of a form we expect.
        if isinstance(validated_value, dict):
            if isinstance(self._type_converter, TypeConverterMapping):
                expected_keys = set(self._type_converter.keys())
            elif isinstance(self._type_converter, OnlyIf):
                expected_keys = set(self._type_converter.cond.keys())
            elif isinstance(self._type_converter, Either):
                mapping = next(
                    filter(lambda x: isinstance(x, TypeConverterMapping),
                           self._type_converter.specs))
                expected_keys = set(mapping.keys())
            else:
                # the code shouldn't reach here so raise an error.
                raise ValueError("Couid not identify specification.")
            bad_keys = set(validated_value.keys()) - expected_keys
            if len(bad_keys) != 0:
                raise KeyError("Keys must be a subset of available keys. "
                               "Bad keys are {}".format(bad_keys))
        # update validated_value with the default (specifically to add dict keys
        # that have defaults and were not manually specified).
        if isinstance(self._default, _SmartDefault):
            return self._default(validated_value)
        return validated_value

    def __eq__(self, other):
        if not isinstance(other, _ValidatedDefaultDict):
            return NotImplemented
        if self.default != other.default:
            return False
        if set(self.keys()) != set(other.keys()):
            return False
        for type_, value in self.items():
            if value != other[type_]:
                return False
        return True

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
                raise KeyError("New default must a subset of current keys.")
        self._default = _to_default(new_default)


class TypeParameterDict(_ValidatedDefaultDict):
    """Type parameter dictionary.

    This class has an attached and detached mode. When attached the mapping is
    synchronized with a C++ class.

    The class performs the same indexing and mutation options as
    `_ValidatedDefaultDict` allows. However, when attached it only allows
    querying for keys that match the actual types of the simulation it is
    attached to.

    The interface expects the passed in C++ object to have a getter and setter
    that follow the camel case style version of ``param_name`` as passed to
    `_attach`.

    Warning:
        This class should not be directly instantiated even by developers, but
        the `hoomd.data.type_param.TypeParameter` class should be used to
        automatically handle this in conjunction with
        `hoomd.operation._BaseHOOMDObject` subclasses.

    Attributes:
        _dict (dict): The underlying data when unattached.
        _cpp_obj: Either ``None`` when not attached or a pybind11 C++ wrapped
            class to interface setting and getting type parameters with.
        _getter (str or NoneType): ``None`` when instantiated and set when first
            attached. This records the getter name for the `cpp_obj` associated
            with the last `_attach` call.
        _setter (str or NoneType): ``None`` when instantiated and set when first
            attached. This records the setter name for the `cpp_obj` associated
            with the last `_attach` call.
        _indexer (_SmartTypeIndexer): A helper class to allow for complex
            indexing patterns such as setting multiple keys at once.
    """

    def __init__(self, *args, len_keys, **kwargs):

        # Validate proper key constraint
        if len_keys < 1 or len_keys != int(len_keys):
            raise ValueError("len_keys must be a positive integer.")
        self._indexer = _SmartTypeIndexer(len_keys)
        self._set_validation_and_defaults(*args, **kwargs)
        self._dict = {}
        self._cpp_obj = None

    @property
    def _attached(self):
        return self._cpp_obj is not None

    def _single_getitem(self, key):
        """Access parameter by key.

        __getitem__ expects an exception to indicate the key is not there.
        """
        if not self._attached:
            return self._dict[key]
        # We always attempt to keep the _dict up to date with the C++ values,
        # and isolate existing components otherwise.
        validated_cpp_value = self._validate_values(
            getattr(self._cpp_obj, self._getter)(key))
        if isinstance(self._dict[key], _HOOMDSyncedCollection):
            if self._dict[key]._update(validated_cpp_value):
                return self._dict[key]
            else:
                self._dict[key]._isolate()
        self._dict[key] = _to_hoomd_data(root=self,
                                         schema=self._type_converter,
                                         data=validated_cpp_value,
                                         parent=None,
                                         identity=key)
        return self._dict[key]

    def _single_setitem(self, key, item):
        """Set parameter by key.

        Assumes value to be validated already.
        """
        if isinstance(self._dict.get(key), _HOOMDSyncedCollection):
            self._dict[key]._isolate()
        self._dict[key] = _to_hoomd_data(root=self,
                                         schema=self._type_converter,
                                         data=item,
                                         parent=None,
                                         identity=key)
        if not self._attached:
            return
        # We don't need to set the _dict yet since we will query C++ when
        # retreiving the key the next time.
        getattr(self._cpp_obj, self._setter)(key, item)

    def __iter__(self):
        """Get the keys in the mapping."""
        if self._attached:
            yield from self._indexer.yield_all_keys()
            return
        # keys are already sorted so no need to sort again
        yield from self._dict.keys()

    def __len__(self):
        """Return mapping length."""
        if self._attached:
            return len(list(self._indexer.yield_all_keys()))
        return len(self._dict)

    def to_base(self):
        """Convert to a `dict`."""
        if not self._attached:
            return {k: _to_base(v) for k, v in self._dict.items()}
        return {key: self[key] for key in self}

    def _validate_values(self, val):
        val = super()._validate_values(val)
        if self._attached:
            _raise_if_required_arg(val)
        return val

    def _attach(self, cpp_obj, param_name, types):
        """Attach type parameter to a C++ object with per type data.

        Args:
            cpp_obj: A pybind11 wrapped C++ object to set and get the type
                parameters from.
            param_name (str): A snake case parameter name (handled automatically
                by ``TypeParameter``) that when changed to camel case prefixed
                by get or set is the str name for the pybind11 exported getter
                and setter.
            types (list[str]): The str names of the available types for the type
                parameter.
        """
        # store info to communicate with c++
        self._setter = "set" + _to_camel_case(param_name)
        self._getter = "get" + _to_camel_case(param_name)
        self._indexer.valid_types = types
        # add all types to c++
        parameters = {
            key: _to_base(self._dict.get(key, self.default))
            for key in self._indexer.yield_all_keys()
        }
        self._cpp_obj = cpp_obj
        for key in self:
            try:
                _raise_if_required_arg(parameters[key])
            except IncompleteSpecificationError as err:
                self._cpp_obj = None
                raise IncompleteSpecificationError(f"for key {key} {str(err)}")
            self._single_setitem(key, parameters[key])

    def _detach(self):
        """Convert to a detached parameter dict."""
        for key in self:
            self._dict[key] = self._single_getitem(key)
        self._cpp_obj = None
        self._indexer.valid_types = None

    def _read(self, obj):
        if not self._attached:
            return
        key = obj._identity
        new_value = getattr(self._cpp_obj, self._getter)(key)
        obj._parent._update(new_value)

    def _write(self, obj):
        if not self._attached:
            return
        # the _dict attribute is the store for the Python copy of the data.
        parent = obj._parent
        with parent._suspend_read:
            self._single_setitem(obj._identity, _to_base(obj._parent))

    def __repr__(self):
        """A string representation of the TypeParameterDict.

        As no single command could give the same object with ``eval``, this just
        returns a convenient form for debugging.
        """
        return f"TypeParameterDict{self.to_base()}"

    def __getstate__(self):
        """Get object state for deepcopying and pickling."""
        if self._attached:
            dict_ = {k: self[k] for k in self}
        else:
            dict_ = self._dict
        return {
            "_indexer": self._indexer,
            "_default": self._default,
            "_type_converter": self._type_converter,
            "_dict": dict_,
            "_cpp_obj": None
        }


class ParameterDict(MutableMapping):
    """Per-key validated mapping for syncing per-instance attributes to C++.

    This class uses the `hoomd.data.collections._HOOMDSyncedCollection`
    subclasses for supporting nested editing of data structures while
    maintaining synced status in C++. This class is used by
    `hoomd.operation._HOOMDGetSetAttrBase` for per-instance level variables.
    Instances must be _attached_ for syncing with a C++ object. The class
    expects that the pybind11 C++ object passed have properties corresponding to
    this instances keys.

    Attributes:
        _cpp_obj (pybind11 object): None when not attatched otherwise a pybind11
            created C++ object in Python.
        _dict (dict[str, any]): Python local mapping where data is stored. It is
            the only source of data when not attached.
        _type_converter (TypeConverterMapping): The validator for each key.
        _getters (dict[str, callable[ParameterDict]):
            A dictionary of instance keys with special getters. This is used to
            provide special logic for retreiving values from the `ParameterDict`
            when attached. The instance is passed to the callable.
        _setters (dict[str, callable[ParameterDict, any]):
            A dictionary of instance keys with special setters. This is used to
            provide special logic for setting values in C++ from the
            `ParameterDict` when attached. The instance and new value is passed
            to the callable.
    """

    def __init__(self, _defaults=_NoDefault, **kwargs):
        self._cpp_obj = None
        self._dict = {}
        self._getters = {}
        self._setters = {}
        # This if statement is necessary to avoid a RecursionError from Python's
        # collections.abc module. The reason for this error is not clear but
        # results from an isinstance check of an empty dictionary.
        if len(kwargs) != 0:
            self._type_converter = to_type_converter(kwargs)
            for key, value in _to_base_defaults(kwargs, _defaults).items():
                self._dict[key] = self._to_hoomd_data(key, value)
        else:
            self._type_converter = {}

    def _set_special_getset(self, attr, getter=None, setter=None):
        """Set special setter and/or getter for a given key.

        See documentation on `_getters` and `_setters`.
        """
        if getter is not None:
            self._getters[attr] = getter
        if setter is not None:
            self._setters[attr] = setter

    def _cpp_setting(self, key, value):
        """Handles setting a new value to C++.

        Assumes value is completely updated and validated.
        """
        setter = self._setters.get(key)
        if setter is not None:
            setter(self, key, value)
            return
        if hasattr(value, "_cpp_obj"):
            setattr(self._cpp_obj, key, value._cpp_obj)
            return
        if isinstance(value, _HOOMDSyncedCollection):
            with value._suspend_read_and_write:
                setattr(self._cpp_obj, key, _to_base(value))
            return
        setattr(self._cpp_obj, key, _to_base(value))

    def __setitem__(self, key, value):
        """Set parameter by key."""
        if key not in self._type_converter.keys():
            if self._attached:
                raise KeyError("Keys cannot be added after Simulation.run().")
            self._type_converter[key] = to_type_converter(value)
        validated_value = self._type_converter[key](value)
        if self._attached:
            try:
                self._cpp_setting(key, validated_value)
            except (AttributeError):
                raise MutabilityError(key)
        if key in self._dict and isinstance(self._dict[key],
                                            _HOOMDSyncedCollection):
            self._dict[key]._isolate()
        self._dict[key] = self._to_hoomd_data(key, validated_value)

    def __getitem__(self, key):
        """Access parameter by key."""
        if not self._attached:
            return self._dict[key]
        # The existence of obj._cpp_obj indicates that the object is split
        # between C++ and Python and the object is responsible for its own
        # syncing. Also, no synced data structure has such an attribute so we
        # just return the object.
        python_value = self._dict[key]
        if hasattr(python_value, "_cpp_obj"):
            return python_value
        try:
            # While trigger remains not a member of  the the C++ classes, we
            # have to allow for attributes that are not gettable.
            getter = self._getters.get(key)
            if getter is None:
                new_value = getattr(self._cpp_obj, key)
            else:
                new_value = getter(self, key)
        except AttributeError:
            return python_value

        if isinstance(python_value, _HOOMDSyncedCollection):
            if python_value._update(new_value):
                return python_value
            else:
                python_value._isolate()
        self._dict[key] = self._to_hoomd_data(key, new_value)
        return self._dict[key]

    def __delitem__(self, key):
        """Remove parameter by key."""
        if self._attached:
            raise RuntimeError(
                "Item deletion is not supported after calling Simulation.run()")
        del self._type_converter[key]
        self._dict.pop(key, None)

    def __iter__(self):
        """Iterate over keys."""
        for key in self._type_converter:
            if key in self._dict:
                yield key

    def __contains__(self, key):
        """Does the mapping contain the given key."""
        return key in self._type_converter and key in self._dict

    def __len__(self):
        """int: The number of keys."""
        return len(self._dict)

    def __eq__(self, other):
        """Determine equality between ParameterDict objects."""
        if not isinstance(other, ParameterDict):
            return NotImplemented
        return (set(self.keys()) == set(other.keys())
                and np.all([np.all(self[key] == other[key]) for key in self]))

    def update(self, other):
        """Add keys and values to the dictionary."""
        # We do not allow new keys or type specification after attaching so we
        # need to check if we are attached here.
        if not self._attached and isinstance(other, ParameterDict):
            self._type_converter.update(other._type_converter)
            self._getters.update(other._getters)
            self._setters.update(other._setters)
            for key, item in other._dict.items():
                if isinstance(item, _HOOMDSyncedCollection):
                    item._change_root(self)
                self._dict[key] = item
        else:
            super().update(other)

    def _attach(self, cpp_obj):
        self._cpp_obj = cpp_obj
        for key in self:
            try:
                self._cpp_setting(key, self._dict[key])
            except AttributeError:
                pass

    @property
    def _attached(self):
        return self._cpp_obj is not None

    def _detach(self):
        # Simply accessing the keys will update them.
        for key in self:
            self[key]
        self._cpp_obj = None

    def _to_hoomd_data(self, key, value):
        return _to_hoomd_data(root=self,
                              schema=self._type_converter[key],
                              parent=None,
                              identity=key,
                              data=value)

    def _write(self, obj):
        if self._attached:
            self._cpp_setting(obj._identity, obj._parent)

    def _read(self, obj):
        if self._attached:
            obj._parent._update(getattr(self._cpp_obj, obj._identity))

    def to_base(self):
        """Return a plain dictionary with equivalent data.

        This recursively convert internal data to base HOOMD types.
        """
        # Using _dict prevents unnecessary _read calls for
        # _HOOMDSyncedCollection objects.
        return {key: _to_base(value) for key, value in self._dict.items()}

    def __getstate__(self):
        """Get object state for deepcopying and pickling."""
        return {
            "_dict": self.to_base(),
            "_type_converter": self._type_converter,
            "_cpp_obj": None,
            "_setters": self._setters,
            "_getters": self._getters
        }

    def __repr__(self):
        """Return a string representation useful for debugging."""
        return f"ParameterDict{{{self.to_base()}}}"
