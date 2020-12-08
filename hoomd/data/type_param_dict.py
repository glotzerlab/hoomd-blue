from itertools import product, combinations_with_replacement
import copy
from collections.abc import MutableMapping
from hoomd.util import to_camel_case, is_iterable
from hoomd.data.typeconverter import (
    to_type_converter, TypeConversionError, RequiredArg)
from hoomd.data.smart_default import toDefault, SmartDefault, NoDefault
from hoomd.data.data_structures import (
    _to_synced_data_structure, _SyncedDataStructure)


def has_str_elems(obj):
    '''Returns True if all elements of iterable are str.'''
    return all([isinstance(elem, str) for elem in obj])


def is_good_iterable(obj):
    '''Returns True if object is iterable with respect to types.'''
    return is_iterable(obj) and has_str_elems(obj)


def proper_type_return(val):
    '''Expects and requires a dictionary with type keys.'''
    if len(val) == 0:
        return None
    elif len(val) == 1:
        return list(val.values())[0]
    else:
        return val


class _ValidatedDefaultDict:
    """Mappings with key/value validation and more extensive defaults.

    Used for HOOMD's type parameter concept which provides for type dependent
    attributes to be easily synced between C++/Python and validated in Python.

    Implements hard coded key validation which expects string keys when
    ``_len_keys == 1`` and tuples of strings when ``_len_keys > 1``. However,
    specifying multiple keys in a single call of `__setitem__` or `__getitem__`
    is allowed.  Anywhere a string is expected, a list or list-like object
    containing strings is also accepted. For multiple string keys this will
    result in the product of the list with the other string values. For
    instance, ``(['A', 'B'], 'A')`` will expand to ``[('A', 'A'), ('A', 'B')]``.
    Before storing any data the tuples produced are sorted lexically since the
    order of the tuple in not important, and this prevents having to check if
    permutations of a tuple exist in the mapping. This is why the example above
    uses `('A', 'B')` rather than `('B', 'A')` as might appear intuitive.
    In addition, a list of tuples of the appropriate length can also be passed.

    Note:
        Instances of `tuple` are handled differently. Tuple values are
        considered to be the innermost layer. As an example, ``(('A', 'B'),
        ('A', 'A'))` could never be a valid key specification, while ``[('A',
        'B'), ('A', 'A')]`` for ``_len_keys == 2`` is perfectly valid.

    Value validation allows the definition of general schemas for each type's
    value. These schemas are defined in `hoomd.data.typeconverter`.

    Default specification is through `hoomd.data.smart_default`. In general,
    defaults for `_ValidatedDefaultDict` allow for partially specified defaults
    allowing users to only specify the necessary parts of a schema or to use
    `self.default` to fully specify a default in lieu of setting each type
    individually.
    """

    def __init__(self, *args, len_keys, **kwargs):
        self._len_keys = len_keys
        if len_keys < 1 or len_keys != int(len_keys):
            raise ValueError("len_keys must be a positive integer.")
        self._len_keys = len_keys
        _defaults = kwargs.pop('_defaults', NoDefault)
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
        self._default = toDefault(default_arg, _defaults)

    def _validate_values(self, val):
        val = self._type_converter(val)
        if isinstance(val, dict):
            dft_keys = set(self.default.keys())
            bad_keys = set(val.keys()) - dft_keys
            if len(bad_keys) != 0:
                raise ValueError("Keys must be a subset of available keys. "
                                 "Bad keys are {}".format(bad_keys))
        if isinstance(self._default, SmartDefault):
            val = self._default(val)
        return val

    def _validate_and_split_key(self, key):
        '''Validate key given regardless of key length.'''
        if self._len_keys == 1:
            return self._validate_and_split_len_one(key)
        else:
            return self._validate_and_split_len(key)

    def _validate_and_split_len_one(self, key):
        '''Validate single type keys.

        Accepted input is a type string, and arbitrarily nested interators that
        culminate in str types.
        '''
        if isinstance(key, str):
            return [key]
        elif is_iterable(key):
            keys = []
            for k in key:
                keys.extend(self._validate_and_split_len_one(k))
            return keys
        else:
            raise KeyError("The key {} is not valid.".format(key))

    def _validate_and_split_len(self, key):
        '''Validate all key lengths greater than one, N.

        Valid input is an arbitrarily deep series of iterables that culminate
        in N length tuples, this includes an iterable depth of zero. The N
        length tuples can contain for each member either a type string or an
        iterable of type strings.
        '''
        if isinstance(key, tuple) and len(key) == self._len_keys:
            fst, snd = key
            if any([not is_good_iterable(v) and not isinstance(v, str)
                    for v in key]):
                raise KeyError("The key {} is not valid.".format(key))
            key = list(key)
            for ind in range(len(key)):
                if isinstance(key[ind], str):
                    key[ind] = [key[ind]]
            return list(product(*key))
        elif is_iterable(key):
            keys = []
            for k in key:
                keys.extend(self._validate_and_split_len(k))
            return keys
        else:
            raise KeyError("The key {} is not valid.".format(key))

    def _yield_keys(self, key):
        '''Returns the generated keys in proper sorted order.

        The order is necessary so ('A', 'B') is equivalent to ('B', A').
        '''
        keys = self._validate_and_split_key(key)
        if self._len_keys > 1:
            # We always set ('a', 'b') instead of ('b', 'a') to reduce storage
            # requirements and prevent errors.
            for key in keys:
                yield tuple(sorted(list(key)))
        else:
            yield from keys

    def __eq__(self, other):
        if self.default != other.default:
            return False
        keys = set(self.keys())
        if keys.union(other.keys()) != keys or \
                keys.difference(other.keys()) != set():
            return False
        for key in self.keys():
            if not self[key] == other[key]:
                return False
        return True

    @property
    def default(self):
        if isinstance(self._default, SmartDefault):
            return self._default.to_base()
        else:
            return copy.copy(self._default)

    @default.setter
    def default(self, new_default):
        new_default = self._type_converter(new_default)
        if isinstance(self._default, SmartDefault):
            new_default = self._default(new_default)
        if isinstance(new_default, dict):
            keys = set(self._default.keys())
            provided_keys = set(new_default.keys())
            if keys.intersection(provided_keys) != provided_keys:
                raise ValueError("New default must a subset of current keys.")
        self._default = toDefault(new_default)

    @staticmethod
    def convert_entry(entry, deepcopy=False):
        if isinstance(entry, _SyncedDataStructure):
            return entry.to_base()
        if deepcopy:
            return deepcopy(entry)
        return entry


class TypeParameterDict(_ValidatedDefaultDict, MutableMapping):
    """Extension of _ValidatedDefaultDict with MutableMapping interface.

    This class is used when HOOMD objects are not attached (i.e. no C++ object
    exists).  Validation only ensures that keys are of the right structure and
    that the values are of the right schema.  It is not possible to enforce that
    the key's type exists or (whatever the second thing is saying).

    Class works with `hoomd.data.data_structures`.
    """

    def __init__(self, *args, len_keys, **kwargs):
        super().__init__(*args, len_keys=len_keys, **kwargs)
        self._data = dict()

    def __getitem__(self, key):
        vals = dict()
        for key in self._yield_keys(key):
            if key in self._data:
                vals[key] = self._data[key]
            # if the key has not be used yet, we still have the default
            # information to retrieve, this also sets the key to the default
            # value explicit since not doing so would potentially cause
            # inconsistent results if users change the value given to them. This
            # means before returning, we must store the data into the _data
            # dict.
            else:
                data_struct = _to_synced_data_structure(
                    self.default, self._type_converter, self, key)
                self._data[key] = data_struct
                vals[key] = data_struct
        return proper_type_return(vals)

    def __setitem__(self, key, val):
        keys = self._yield_keys(key)
        try:
            val = self._validate_values(val)
        except TypeConversionError as err:
            raise TypeConversionError(
                "For types {}, error {}.".format(list(keys), str(err)))
        for key in keys:
            # We need to remove reference to self in all synced data structures
            # that are being removed from this object.
            if key in self and isinstance(self[key], _SyncedDataStructure):
                self[key]._parent = None
            # Likewise we need to convert to synced data structures for new
            # values
            self._data[key] = _to_synced_data_structure(
                val, self._type_converter, self, key)

    def __delitem__(self, key):
        for key in self._yield_keys(key):
            del self._data[key]

    def __iter__(self):
        if self._len_keys == 1:
            yield from self._data.keys()
        else:
            # This is to provide a consistent means of iterating over type keys
            # that are more than one type. We do this to prevent the need for
            # storing ('A', 'B') and ('B', 'A') separately.
            for key in self._data.keys():
                yield tuple(sorted(list(key)))

    def __len__(self):
        return len(self._data)

    def to_base(self):
        return {key: self.convert_entry(value) for key, value in self.items()}

    def _handle_update(self, obj, label=None):
        pass


class AttachedTypeParameterDict(_ValidatedDefaultDict, MutableMapping):
    """Type parameters when object is attached to C++.

    Handles syncing to C++ from Python when a key changes. Also validates that
    keys are of existent types. Incomplete schemas are not allowed as well.

    This class requires knowledge of the C++ object to sync to the name of the
    parameter it represents, the kind of type it accepts (e.g. particle or bond
    type), and the simulation to query for types when setting keys.

    Class works with `hoomd.data.data_structures`.
    """

    def __init__(self, cpp_obj, param_name,
                 type_kind, type_param_dict, sim):
        # store info to communicate with c++
        self._cpp_obj = cpp_obj
        self._param_name = param_name
        self._sim = sim
        self._type_kind = type_kind
        self._len_keys = type_param_dict._len_keys
        # Get default data
        self._default = type_param_dict._default
        self._type_converter = type_param_dict._type_converter

        # Change parent of data classes
        for value in type_param_dict.values():
            if isinstance(value, _SyncedDataStructure):
                value._parent = self
        self._data = type_param_dict._data
        # add all types to c++
        for key in self:
            self[key] = type_param_dict[key]

    def to_detached(self):
        if isinstance(self.default, dict):
            type_param_dict = TypeParameterDict(**self.default,
                                                len_keys=self._len_keys)
        else:
            type_param_dict = TypeParameterDict(self.default,
                                                len_keys=self._len_keys)
        type_param_dict._type_converter = self._type_converter
        for key in self:
            type_param_dict[key] = self[key]
            if isinstance(self[key], _SyncedDataStructure):
                type_param_dict[key]._parent = type_param_dict
        return type_param_dict

    def __getitem__(self, key):
        vals = dict()
        for key in self._yield_keys(key):
            cpp_val = getattr(self._cpp_obj, self._getter)(key)
            if (key in self._data and
                    isinstance(self._data[key], _SyncedDataStructure)):
                self._data[key]._parent = None
            data_struct = _to_synced_data_structure(
                cpp_val, self._type_converter, self, key)
            self._data[key] = data_struct
            vals[key] = data_struct
        return proper_type_return(vals)

    def __setitem__(self, key, val):
        keys = self._yield_keys(key)
        try:
            val = self._validate_values(val)
        except TypeConversionError as err:
            raise TypeConversionError(
                "For types {}, error {}.".format(list(keys), str(err)))
        for key in keys:
            getattr(self._cpp_obj, self._setter)(key, val)
            data_struct = _to_synced_data_structure(
                val, self._type_converter, self, key)
            if isinstance(data_struct, _SyncedDataStructure):
                if (key in self._data and
                        isinstance(self._data[key], _SyncedDataStructure)):
                    self._data[key]._parent = None
                self._data[key] = data_struct

    def __iter__(self):
        single_keys = getattr(self._sim.state, self._type_kind)
        if self._len_keys == 1:
            yield from single_keys
        else:
            for key in combinations_with_replacement(single_keys,
                                                     self._len_keys):
                yield tuple(sorted(list(key)))

    def __delitem__(self, key):
        raise RuntimeError(
            "Cannot delete keys, available are defined by types in the state.")

    def __len__(self):
        return len(list(iter(self)))

    def _yield_keys(self, key):
        '''Includes key check for existing simulation keys.'''
        curr_keys = set(iter(self))
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
        return 'set' + to_camel_case(self._param_name)

    @property
    def _getter(self):
        return 'get' + to_camel_case(self._param_name)

    def to_base(self):
        return {key: getattr(self._cpp_obj, self._getter)(key) for key in self}

    def _handle_update(self, obj, label):
        """Handle updates of child data structures."""
        getattr(self._cpp_obj, self._setter)(label, obj.to_base())
