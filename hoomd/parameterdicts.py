from itertools import product, combinations_with_replacement
from copy import deepcopy
from numpy import array, ndarray

# Psudonym for None that states an argument is required to be supplied by the
# user
RequiredArg = None


def is_iterable(obj):
    '''Returns True if object is iterable and not a str or dict.'''
    return hasattr(obj, '__iter__') and not bad_iterable_type(obj)


def bad_iterable_type(obj):
    '''Returns True if str or dict.'''
    return isinstance(obj, str) or isinstance(obj, dict)


def has_str_elems(obj):
    '''Returns True if all elements of iterable are str.'''
    return all([isinstance(elem, str) for elem in obj])


def is_good_iterable(obj):
    '''Returns True if object is iterable with respect to types.'''
    return is_iterable(obj) and has_str_elems(obj)


def to_camel_case(string):
    return string.replace('_', ' ').title().replace(' ', '')


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
        if len(kwargs) != 0 and len(args) != 0:
            raise ValueError("An unnamed argument and keyword arguments "
                             "cannot both be specified.")
        if len(kwargs) == 0 and len(args) == 0:
            raise ValueError("Either an unnamed argument or keyword "
                             "arguments must be specified.")
        if len(args) > 1:
            raise ValueError("Only one unnamed argument allowed.")
        if len(kwargs) > 0:
            self._default = kwargs
            self._dft_constructor = dict
        else:
            dft = args[0]
            self._default = dft
            if isinstance(dft, ndarray):
                self._dft_constructor = array
            else:
                self._dft_constructor = type(dft)

    def _cast_to_dft(self, val):
        '''Attempts to convert val to expected value type.'''
        try:
            val = self._dft_constructor(val)
        except TypeError:
            raise TypeError("Provided type {} cannot be cast to initialized "
                            "required type {}".format(type(val),
                                                      type(self._default))
                            )
        else:
            return val

    def _validate_values(self, val):
        val = self._cast_to_dft(val)
        if type(self._default) == dict:
            dft_copy = self.default
            dft_keys = set(dft_copy)
            if len(dft_keys.intersection(val.keys())) != len(val.keys()):
                raise ValueError("Keys must be a subset of available keys ")
            dft_copy.update(val)
            val = dft_copy
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

    @property
    def default(self):
        return deepcopy(self._default)

    @default.setter
    def default(self, new_default):
        new_default = self._cast_to_dft(new_default)
        if isinstance(new_default, dict):
            keys = set(self._default.keys())
            provided_keys = set(new_default.keys())
            if keys.intersection(provided_keys) != provided_keys:
                raise ValueError("New default must a subset of current keys.")
            self._default.update(new_default)
        else:
            self._default = new_default


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
        val = self._validate_values(val)
        for key in self._yield_keys(key):
            self._dict[key] = val

    def keys(self):
        if self._len_keys == 1:
            yield from self._dict.keys()
        else:
            for key in self._dict.keys():
                yield tuple(sorted(list(key)))


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
        self._default = type_param_dict.default
        self._dft_constructor = type_param_dict._dft_constructor
        # add all types to c++
        for key in self.keys():
            try:
                self[key] = type_param_dict[key]
            except ValueError as verr:
                raise ValueError("Type {} ".format(key) + verr.args[0])

    def to_dettached(self):
        if isinstance(self.default, dict):
            type_param_dict = TypeParameterDict(**self.default,
                                                len_keys=self._len_keys)
        else:
            type_param_dict = TypeParameterDict(self.default,
                                                len_keys=self._len_keys)
        for key in self.keys():
            type_param_dict[key] = self[key]
        return type_param_dict

    def __getitem__(self, key):
        vals = dict()
        for key in self._yield_keys(key):
            vals[key] = getattr(self._cpp_obj, self._getter)(key)
        return proper_type_return(vals)

    def __setitem__(self, key, val):
        val = self._validate_values(val)
        for key in self._yield_keys(key):
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
                if v is None:
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
