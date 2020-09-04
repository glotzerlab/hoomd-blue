from copy import deepcopy
from collections.abc import MutableMapping
from hoomd.typeconverter import to_type_converter
from hoomd.smart_default import to_base_defaults, NoDefault


class ParameterDict(MutableMapping):
    def __init__(self, _defaults=NoDefault, **kwargs):
        self._type_converter = to_type_converter(kwargs)
        self._dict = to_base_defaults(kwargs, _defaults)

    def __setitem__(self, key, value):
        if key not in self._type_converter.keys():
            self._dict[key] = value
            self._type_converter[key] = to_type_converter(value)
        else:
            self._dict[key] = self._type_converter[key](value)

    def __getitem__(self, key):
        return self._dict[key]

    def __delitem__(self, key):
        del self._dict[key]
        del self._type_converter.converter[key]

    def __iter__(self):
        yield from self._dict

    def __len__(self):
        return len(self._dict)

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
            super().update(dict_)

    def setitem_with_validation_function(self, key, value, converter):
        self._type_converter[key] = converter
        self._dict[key] = value
