from collections import defaultdict
from itertools import product, combinations_with_replacement
from functools import partial
from copy import deepcopy

# Psudonym for None that states an argument is required to be supplied by the
# user
RequiredArg = None


# Checks if a value is iterable and not a string
def is_iterable(obj):
    return hasattr(obj, '__iter__') and not isinstance(obj, str)


def has_str_elems(obj):
    return all([isinstance(elem, str) for elem in obj])


def is_bad_iterable(obj):
    return is_iterable(obj) and not has_str_elems(obj)


def partial_dict(**kwargs):
    return partial(dict, **kwargs)


def const(val):
    return lambda: val


def to_camel_case(string):
    return string.replace('_', ' ').title().replace(' ', '')


class _ValidateDict:

    def _validate_and_split_key(self, key):
        if self._len_keys == 1:
            return self._validate_and_split_len_one(key)
        elif self._len_keys == 2:
            return self._validate_and_split_len_two(key)
        else:
            return None

    def _validate_and_split_len_one(self, key):
        if isinstance(key, str):
            return [key]
        elif is_iterable(key) and has_str_elems(key):
            return list(key)
        else:
            raise KeyError("The key {} is not valid.".format(key))

    def _validate_and_split_len_two(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            fst, snd = key
            if is_bad_iterable(fst) or is_bad_iterable(snd):
                raise KeyError("The key {} is not valid.".format(key))
            if isinstance(fst, str):
                fst = [fst]
            if isinstance(snd, str):
                snd = [snd]
            return product(fst, snd)
        elif is_iterable(key):
            keys = []
            for k in key:
                if isinstance(k, tuple) and len(k) == 2 and has_str_elems(k):
                    keys.append(k)
                else:
                    raise KeyError("The key {} is not valid.".format(k))
            return keys
        else:
            raise KeyError("The key {} is not valid.".format(key))


class TypeParameterDict(_ValidateDict):

    def __init__(self, *args, len_keys, **kwargs):

        # Validate proper key constraint
        if len_keys > 2:
            raise ValueError("TypeParameterDict does not support keys larger "
                             "than 2 types.")
        self._len_keys = len_keys

        # Create default dictionary
        if len(kwargs) != 0 and len(args) != 0:
            raise ValueError("An unnamed argument and keyword arguments "
                             "cannot both be specified.")
        if len(kwargs) == 0 and len(args) == 0:
            raise ValueError("Either an unnamed argument or keyword "
                             "arguments must be specified.")
        if len(args) > 1:
            raise ValueError("Only one unnamed argument allowed.")
        if len(kwargs) > 0:
            self._dict = defaultdict(partial_dict(**kwargs))
            self._is_kw = True
        else:
            self._dict = defaultdict(const(args[0]))
            self._is_kw = False

    def __getitem__(self, key):
        keys = self._validate_and_split_key(key)
        vals = dict()
        for key in keys:
            if self._len_keys > 1:
                key = tuple(sorted(key))
            vals[key] = self._dict[key]
        return vals

    def __setitem__(self, key, val):
        keys = self._validate_and_split_key(key)
        val = self._validate_values(val)
        for key in keys:
            if self._len_keys > 1:
                key = tuple(sorted(key))
            self._setkey(key, val)

    def _setkey(self, key, val):
        if self._is_kw:
            if key in self._dict:
                self._dict[key] = self._dict.default_factory()
                self._dict[key].update(val)
            else:
                self._dict[key].update(val)
        else:
            self._dict[key] = val

    def _validate_values(self, val):
        curr_dft = self._dict.default_factory()
        if self._is_kw:
            if not isinstance(val, dict):
                raise ValueError("Cannot set type to non-dictionary value.")
            dft_keys = set(curr_dft.keys())
            if len(dft_keys.intersection(val.keys())) != len(val.keys()):
                raise ValueError("Keys must be a subset of available keys ")
        return val

    @property
    def default(self):
        return self._dict.default_factory()

    @default.setter
    def default(self, val):
        curr_dft = self.default
        if not isinstance(val, type(curr_dft)):
            raise ValueError("New default expected type {} "
                             "but received type {}".format(type(curr_dft),
                                                           type(val)))
        if isinstance(val, dict):
            if curr_dft.keys() != val.keys():
                raise ValueError("New default must contain the same keys.")
            self._dict.default_factory = partial_dict(**val)
        else:
            self._dict.default_factory = const(val)


class AttachedTypeParameterDict(_ValidateDict):

    def __init__(self, cpp_obj, param_name,
                 type_kind, type_param_dict, sim):
        # add all types to c++
        self._cpp_obj = cpp_obj
        self._param_name = param_name
        self._sim = sim
        self._default = type_param_dict.default
        self._len_keys = type_param_dict._len_keys
        self._type_kind = type_kind
        for key, value in type_param_dict._dict.items():
            self[key] = value

    def to_dettached(self):
        pass

    def __getitem__(self, key):
        keys = self._validate_and_split_key(key)
        curr_keys = self.keys()
        vals = {}
        for key in keys:
            if self._len_keys > 1:
                key = tuple(sorted(key))
            if key not in curr_keys:
                raise KeyError("Type {} does not exist in the "
                               "system.".format(key))
            vals[key] = getattr(self._cpp_obj, self._getter)(key)
        return vals

    def __setitem__(self, key, val):
        keys = self._validate_and_split_key(key)
        curr_keys = self.keys()
        for key in keys:
            if key not in curr_keys:
                raise KeyError("Type {} does not exist in the "
                               "system.".format(key))
            getattr(self._cpp_obj, self._setter)(key, val)

    @property
    def _setter(self):
        return 'set' + to_camel_case(self._param_name)

    @property
    def _getter(self):
        return 'get' + to_camel_case(self._param_name)

    def keys(self):
        single_keys = getattr(self._sim.state, self._type_kind)
        if self._len_keys == 2:
            return combinations_with_replacement(single_keys, 2)
        else:
            return single_keys

    @property
    def default(self):
        return self._default

    @default.setter
    def default(self, val):
        curr_dft = self.default
        if not isinstance(val, type(curr_dft)):
            raise ValueError("New default expected type {} "
                             "but received type {}".format(type(curr_dft),
                                                           type(val)))
        if isinstance(val, dict):
            if curr_dft.keys() != val.keys():
                raise ValueError("New default must contain the same keys.")
            self._default = deepcopy(val)
        else:
            self._default = deepcopy(val)
