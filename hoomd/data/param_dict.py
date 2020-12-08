from copy import deepcopy
from collections.abc import MutableMapping
from hoomd.data.typeconverter import to_type_converter
from hoomd.data.smart_default import to_base_defaults, NoDefault
from hoomd.data.data_structures import (
    _to_synced_data_structure, _SyncedDataStructure)


class ParameterDict(MutableMapping):
    """Mutable mapping that holds validation information for its keys.

    The class is designed to be used with `hoomd.operation._BaseHOOMDObject`
    classes to support their attributes with an easy to use type/validation
    specification. This class also integrates with the
    `hoomd.data.data_structures` to allow `ParameterDict` objects to be notified
    when one of their keys is updated if a key is not set through `__setitem__`.

    Users should not encounter this class and should always be used through
    the `hoomd.operation._BaseHOOMDObject` attribute interface.

    Args:
        _defaults:
            An optional mapping of defaults values for mapping keys. Defaults to
            `hoomd.data.smart_default.NoDefault` which is required since `None`
            can be a valid default value for the
            `hoomd.data.smart_default.to_base_defaults` function.
        \*\*kwargs:
            Any number of keyword arguments. Each key supports the full type
            specification allowed by `hoomd.data.typeconverter`.
    """
    def __init__(self, _defaults=NoDefault, **kwargs):
        self._dict = dict()
        self._parent = None
        type_def = to_type_converter(kwargs)
        default_val = to_base_defaults(kwargs, _defaults)
        self._type_converter = type_def
        for key in default_val:
            self._dict[key] = _to_synced_data_structure(
                default_val[key], type_def[key], self, key)

    def __setitem__(self, key, value):
        """Set key: value pair in mapping.

        We choose to allow for keys that are not specified by the schema defined
        at instantiation. For such keys we grab the value's type and use that
        for future validation. All other keys are validated according to the
        schema.
        """
        if key not in self:
            type_def = to_type_converter(value)
            self._type_converter[key] = type_def
            self._dict[key] = _to_synced_data_structure(
                value, type_def, self, key)
        else:
            if isinstance(self._dict[key], _SyncedDataStructure):
                self._dict[key]._parent = None
            type_def = self._type_converter[key]
            self._dict[key] = _to_synced_data_structure(
                type_def(value), type_def, self, key)

    def __getitem__(self, key):
        return self._dict[key]

    def __delitem__(self, key):
        """Deleting the key also removes any validation information for the key.
        """
        item = self._dict[key]
        # disconnect child data structure from parent
        if isinstance(item, _SyncedDataStructure):
            item._parent = None
        del self._dict[key]
        del self._type_converter.converter[key]

    def __iter__(self):
        yield from self._dict

    def __len__(self):
        return len(self._dict)

    def __deepcopy__(self, memo):
        """Return a deepcopy if possible else return a shallow copy.

        While this breaks assumptions of deepcopy the behavior as it is
        currently implemented is to work for most cases. As C++ objects through
        pybind11 are not deepcopiable by default these objects will cause the
        deepcopy to fail. For now this ignores those failures just using the
        object itself for the copy.
        """
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
        """Update the mapping with another mapping.

        Also updates validation information when other mapping is a
        `ParameterDict`.

        Args:
            dict_: `dict`
                A mapping to update the current mapping instance with.
        """
        if isinstance(dict_, ParameterDict):
            for key, value in dict_.items():
                self.setitem_with_validation_function(key,
                                                      value,
                                                      dict_._type_converter[key]
                                                      )
        else:
            super().update(dict_)

    def setitem_with_validation_function(self, key, value, converter):
        if key in self and isinstance(self[key], _SyncedDataStructure):
            self[key]._parent = None
        self._type_converter[key] = converter
        self._dict[key] = value

    def to_base(self):
        """Return a plain Python `dict`."""
        rtn_dict = {}
        for key, value in self.items():
            if isinstance(value, _SyncedDataStructure):
                rtn_dict[key] = value.to_base()
            else:
                try:
                    use_value = deepcopy(value)
                except Exception:
                    use_value = value
                rtn_dict[key] = use_value
        return rtn_dict

    def _handle_update(self, obj, label):
        """Handle updates of child data structure."""
        if self._parent:
            setattr(self._parent, label, obj.to_base())
