from itertools import product, combinations_with_replacement
from copy import copy, deepcopy
from collections.abc import MutableMapping
from hoomd.util import to_camel_case, is_iterable
from hoomd.typeconverter import (
    to_type_converter, TypeConversionError, RequiredArg)
from hoomd.smart_default import toDefault, SmartDefault, NoDefault
from hoomd._data_structures import (
    _to_hoomd_data_structure, _HOOMDDataStructures, _HOOMDDict)


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

    def __init__(self, *args, **kwargs):
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

    # Add function to validate dictionary keys' value types as well
    # Could follow current model on the args based type checking

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
            return copy(self._default)

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


class TypeParameterDict(_ValidatedDefaultDict, MutableMapping):

    def __init__(self, *args, len_keys, **kwargs):

        # Validate proper key constraint
        if len_keys < 1 or len_keys != int(len_keys):
            raise ValueError("len_keys must be a positive integer.")
        self._len_keys = len_keys
        super().__init__(*args, **kwargs)
        self._dict = dict()

    def __getitem__(self, key):
        vals = dict()
        for key in self._yield_keys(key):
            try:
                vals[key] = self._dict[key]
            except KeyError:
                data_struct = _to_hoomd_data_structure(
                    self.default, self._type_converter, self, key)
                self._dict[key] = data_struct
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
            if key in self and isinstance(self[key], _HOOMDDataStructures):
                self[key]._parent = None
            self._dict[key] = _to_hoomd_data_structure(
                val, self._type_converter, self, key)

    def __delitem__(self, key):
        for key in self._yield_keys(key):
            del self._dict[key]

    def __iter__(self):
        if self._len_keys == 1:
            yield from self._dict.keys()
        else:
            for key in self._dict.keys():
                yield tuple(sorted(list(key)))

    def __len__(self):
        return len(self._dict)

    def to_base(self):
        rtn_dict = {}
        for key, value in self.items():
            if isinstance(value, _HOOMDDataStructures):
                rtn_dict[key] = value.to_base()
            else:
                try:
                    new_value = deepcopy(value)
                except Exception:
                    new_value = value
                rtn_dict[key] = new_value
        return rtn_dict

    def _handle_update(self, obj, label=None):
        pass


class AttachedTypeParameterDict(_ValidatedDefaultDict, MutableMapping):

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
            if isinstance(value, _HOOMDDataStructures):
                value._parent = self
        self._dict = type_param_dict._dict
        # add all types to c++
        for key in self:
            self[key] = type_param_dict[key]

    def to_dettached(self):
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
        vals = dict()
        for key in self._yield_keys(key):
            cpp_val = getattr(self._cpp_obj, self._getter)(key)
            if (key in self._dict
                    and isinstance(self._dict[key], _HOOMDDataStructures)):
                self._dict[key]._parent = None
            data_struct = _to_hoomd_data_structure(
                cpp_val, self._type_converter, self, key)
            self._dict[key] = data_struct
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
            data_struct = _to_hoomd_data_structure(
                val, self._type_converter, self, key)
            if isinstance(data_struct, _HOOMDDataStructures):
                if (key in self._dict
                        and isinstance(self._dict[key], _HOOMDDataStructures)):
                    self._dict[key]._parent = None
                self._dict[key] = data_struct

    def __iter__(self):
        single_keys = getattr(self._sim.state, self._type_kind)
        if self._len_keys == 1:
            yield from single_keys
        else:
            for key in combinations_with_replacement(single_keys,
                                                     self._len_keys):
                yield tuple(sorted(list(key)))

    def __delitem__(self, key):
        raise RuntimeError("Cannot delete keys from dict.")

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
        rtn_dict = {}
        for key in self:
            rtn_dict[key] = getattr(self._cpp_obj, self._getter)(key)
        return rtn_dict

    def _handle_update(self, obj, label=None):
        getattr(self._cpp_obj, self._setter)(label, obj.to_base())
