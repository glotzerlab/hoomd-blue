from hoomd.parameterdicts import AttachedTypeParameterDict
from copy import deepcopy


class TypeParameter:
    def __init__(self, name, type_kind, param_dict):
        self.name = name
        self.type_kind = type_kind
        self.param_dict = param_dict

    def __getattr__(self, attr):
        try:
            return getattr(self.param_dict, attr)
        except AttributeError:
            raise AttributeError("'{}' object has no attribute "
                                 "'{}'".format(type(self), attr))

    def __getitem__(self, key):
        return self.param_dict[key]

    def __setitem__(self, key, value):
        self.param_dict[key] = value

    def __eq__(self, other):
        return self.name == other.name and \
            self.type_kind == other.type_kind and \
            self.param_dict == other.param_dict

    @property
    def default(self):
        return self.param_dict.default

    @default.setter
    def default(self, value):
        self.param_dict.default = value

    def attach(self, cpp_obj, sim):
        self.param_dict = AttachedTypeParameterDict(cpp_obj,
                                                    self.name,
                                                    self.type_kind,
                                                    self.param_dict,
                                                    sim)
        return self

    def detach(self):
        self.param_dict = self.param_dict.to_dettached()
        return self

    def to_dict(self):
        return self.param_dict.to_dict()

    def keys(self):
        yield from self.param_dict.keys()

    @property
    def state(self):
        state = self.to_dict()
        if self.param_dict._len_keys > 1:
            state = {str(key): value for key, value in state.items()}
        state['__default'] = self.default
        return state

    def __deepcopy__(self, memo):
        return TypeParameter(self.name, self.type_kind,
                             deepcopy(self.param_dict))
