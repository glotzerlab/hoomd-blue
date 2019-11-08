from hoomd.parameterdicts import AttachedTypeParameterDict


class TypeParameter:
    def __init__(self, name, type_kind, param_dict):
        self.name = name
        self.type_kind = type_kind
        self.param_dict = param_dict

    def __getitem__(self, key):
        return self.param_dict[key]

    def __setitem__(self, key, value):
        self.param_dict[key] = value

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
