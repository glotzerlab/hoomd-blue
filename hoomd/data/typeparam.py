from hoomd.data.parameterdicts import AttachedTypeParameterDict


class TypeParameter:
    __slots__ = ('name', 'type_kind', 'param_dict')

    def __init__(self, name, type_kind, param_dict):
        self.name = name
        self.type_kind = type_kind
        self.param_dict = param_dict

    def __getattr__(self, attr):
        if attr in self.__slots__:
            return super().__getattr__(attr)
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

    def _attach(self, cpp_obj, sim):
        self.param_dict = AttachedTypeParameterDict(cpp_obj,
                                                    self.name,
                                                    self.type_kind,
                                                    self.param_dict,
                                                    sim)
        return self

    def _detach(self):
        self.param_dict = self.param_dict.to_dettached()
        return self

    def to_dict(self):
        return self.param_dict.to_dict()

    def keys(self):
        yield from self.param_dict.keys()

    def __getstate__(self):
        state = {'name': self.name,
                 'type_kind': self.type_kind,
                 'param_dict': self.param_dict
                 }
        if isinstance(self.param_dict, AttachedTypeParameterDict):
            state['param_dict'] = self.param_dict.to_dettached()
        return state

    def __setstate__(self, state):
        for attr, value in state.items():
            setattr(self, attr, value)
