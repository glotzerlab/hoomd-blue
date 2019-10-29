from collections import defaultdict

RequiredArg = None

class TypeParameterDict:

    def __init__(self, len_keys=1, **kwargs):
        self._dict = defaultdict(kwargs)
        self._len_keys = len_keys

    def __getitem__(self, key):
        keys = self._validate_and_split_key(key)
        vals = dict()
        for key in keys:
            vals[key] = self._dict[key]
        return vals

    def __setitem__(self, key, val):
        keys = self._validate_and_split_key(key)
        val = self.validate_values(val)
        for key in keys:
            self._dict[key] = val

    def _validate_and_split_key(self, key):
        pass

    def _validate_values(self, val):
        pass


class AttachedTypeParameterDict:

    def __init__(self, types, type_param_dict, cpp_obj, sim):
        # add all types to c++
        pass

    def to_dettached(self):
        pass

    def __getitem__(self, key):
        pass

    def __setitem__(self, key, val):
        pass
