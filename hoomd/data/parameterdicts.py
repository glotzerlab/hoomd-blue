from itertools import product, combinations_with_replacement
from copy import copy

from hoomd.util import to_camel_case, is_iterable
from hoomd.data.typeconverter import (
    to_type_converter, TypeConversionError, RequiredArg)
from hoomd.data.smart_default import (
    to_base_defaults, toDefault, SmartDefault, NoDefault)


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
        if self._len_keys > 1:
            keys = self._validate_and_split_key(key)
            for key in keys:
                yield tuple(sorted(list(key)))
        else:
            yield from self._validate_and_split_key(key)

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


class TypeParameterDict(_ValidatedDefaultDict):

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
                vals[key] = self.default
        return proper_type_return(vals)

    def __setitem__(self, key, val):
        keys = self._yield_keys(key)
        try:
            val = self._validate_values(val)
        except TypeConversionError as err:
            raise TypeConversionError(
                "For types {}, error {}.".format(list(keys), str(err)))
        for key in keys:
            self._dict[key] = val

    def keys(self):
        if self._len_keys == 1:
            yield from self._dict.keys()
        else:
            for key in self._dict.keys():
                yield tuple(sorted(list(key)))

    def to_dict(self):
        return self._dict


class AttachedTypeParameterDict(_ValidatedDefaultDict):

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
        # add all types to c++
        for key in self.keys():
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
            vals[key] = getattr(self._cpp_obj, self._getter)(key)
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

    def _yield_keys(self, key):
        '''Includes key check for existing simulation keys.'''
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
        return 'set' + to_camel_case(self._param_name)

    @property
    def _getter(self):
        return 'get' + to_camel_case(self._param_name)

    def keys(self):
        single_keys = getattr(self._sim.state, self._type_kind)
        if self._len_keys == 1:
            yield from single_keys
        else:
            for key in combinations_with_replacement(single_keys,
                                                     self._len_keys):
                yield tuple(sorted(list(key)))

    def to_dict(self):
        rtn_dict = dict()
        for key in self.keys():
            rtn_dict[key] = getattr(self._cpp_obj, self._getter)(key)
        return rtn_dict


class ParameterDict(dict):
    def __init__(self, _defaults=NoDefault, **kwargs):
        self._type_converter = to_type_converter(kwargs)
        super().__init__(**to_base_defaults(kwargs, _defaults))

    def __setitem__(self, key, value):
        if key not in self._type_converter.keys():
            super().__setitem__(key, value)
            self._type_converter[key] = to_type_converter(value)
        else:
            super().__setitem__(key, self._type_converter[key](value))

    def __deepcopy__(self, memo):
        new_dict = ParameterDict()
        for key, value in self.items():
            try:
                new_dict[key] = deepcopy(value)
            except TypeError:
                new_dict[key] = value
            try:
                new_dict._type_converter[key] = deepcopy(
                    self._type_converter[key])
            except TypeError:
                new_dict._type_converter[key] = self._type_converter[key]
        return new_dict

    def update(self, dict_):
        if isinstance(dict_, ParameterDict):
            for key, value in dict_.items():
                self.setitem_with_validation_function(key,
                                                      value,
                                                      dict_._type_converter[key]
                                                      )
        else:
            for key, value in dict_.items():
                self[key] = value

    def setitem_with_validation_function(self, key, value, converter):
        self._type_converter[key] = converter
        super().__setitem__(key, value)
