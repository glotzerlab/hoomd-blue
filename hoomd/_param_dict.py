from copy import deepcopy
from collections.abc import MutableMapping
from hoomd.typeconverter import to_type_converter
from hoomd.smart_default import to_base_defaults, NoDefault
from hoomd._data_structures import (
    _to_hoomd_data_structure, _HOOMDDataStructures)


class ParameterDict(MutableMapping):
    def __init__(self, _defaults=NoDefault, **kwargs):
        self._dict = dict()
        type_def = to_type_converter(kwargs)
        default_val = to_base_defaults(kwargs, _defaults)
        self._type_converter = type_def
        for key in default_val:
            self._dict[key] = _to_hoomd_data_structure(
                default_val[key], type_def[key], self, key)

    def __setitem__(self, key, value):
        if key not in self._type_converter.keys():
            type_def = to_type_converter(value)
            self._type_converter[key] = type_def
            self._dict[key] = _to_hoomd_data_structure(
                value, type_def, self, key)
        else:
            type_def = self._type_converter[key]
            self._dict[key] = _to_hoomd_data_structure(
                type_def(value), type_def, self, key)

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

    def to_base(self):
        rtn_dict = {}
        for key, value in self.items():
            if isinstance(value, _HOOMDDataStructures):
                rtn_dict[key] = value.to_base()
            else:
                try:
                    use_value = deepcopy(value)
                except Exception:
                    use_value = value
                rtn_dict[key] = use_value
        return rtn_dict
