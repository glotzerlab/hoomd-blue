import copy
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
            An optional mapping of default values for mapping keys. Defaults to
            `hoomd.data.smart_default.NoDefault` which is required since `None`
            can be a valid default value for the
            `hoomd.data.smart_default.to_base_defaults` function.
        \*\*kwargs:
            Any number of keyword arguments. Each key supports the full type
            specification allowed by `hoomd.data.typeconverter`.
    """

    def __init__(self, _defaults=NoDefault, **kwargs):
        self._data = dict()
        self._parent = None
        type_def = to_type_converter(kwargs)
        default_val = to_base_defaults(kwargs, _defaults)
        self._type_converter = type_def
        for key in default_val:
            self._data[key] = _to_synced_data_structure(
                default_val[key], type_def[key], self, key)

    def __setitem__(self, key, value):
        """Set key: value pair in mapping.

        We choose to allow for keys that are not specified by the schema defined
        at instantiation. For such keys, we grab the value's type and use that
        for future validation. All other keys are validated according to the
        schema.
        """
        if key not in self:
            type_def = to_type_converter(value)
            self._type_converter[key] = type_def
            self._data[key] = _to_synced_data_structure(
                value, type_def, self, key)
        else:
            if isinstance(self._data[key], _SyncedDataStructure):
                self._data[key]._parent = None
            type_def = self._type_converter[key]
            self._data[key] = _to_synced_data_structure(
                type_def(value), type_def, self, key)

    def __getitem__(self, key):
        return self._data[key]

    def __delitem__(self, key):
        """Deleting the key also removes any validation information for the key.
        """
        item = self._data[key]
        # disconnect child data structure from parent
        if isinstance(item, _SyncedDataStructure):
            item._parent = None
        del self._data[key]
        del self._type_converter.converter[key]

    def __iter__(self):
        yield from self._data

    def __len__(self):
        return len(self._data)

    def update(self, other):
        """Update the mapping with another mapping.

        Also updates validation information when other mapping is a
        `ParameterDict`.

        Args:
            other: `dict`
                A mapping to update the current mapping instance with.
        """
        if isinstance(other, ParameterDict):
            for key, value in other.items():
                self.setitem_with_validation_function(key,
                                                      value,
                                                      other._type_converter[key]
                                                      )
        else:
            super().update(other)

    def setitem_with_validation_function(self, key, value, converter):
        if key in self and isinstance(self[key], _SyncedDataStructure):
            self[key]._parent = None
        self._type_converter[key] = converter
        self._data[key] = value

    def to_base(self, deepcopy=False):
        """Return a Python `dict`."""
        def convert_value(value):
            if isinstance(value, _SyncedDataStructure):
                return value.to_base()
            if deepcopy:
                return copy.deepcopy(value)
            return value

        return {key: convert_value(value) for key, value in self.items()}

    def _handle_update(self, obj, label):
        """Handle updates of child data structure."""
        if self._parent:
            setattr(self._parent, label, obj.to_base())
