"""Base data classes for HOOMD-blue objects that provide typing support."""
from abc import ABCMeta, abstractmethod
from collections.abc import (
    Mapping, MutableMapping, Sequence, MutableSequence, Set, MutableSet)
from contextlib import contextmanager
from copy import deepcopy
from itertools import cycle

from hoomd.data.typeconverter import (
    TypeConversionError, TypeConverterValue, OnlyIf, Either,
    TypeConverterMapping, TypeConverterSequence)


class _SyncedDataStructure(metaclass=ABCMeta):
    """Base class for HOOMD's container classes.

    These child classes exist to enable parameter validation and mantain
    consistency with C++ after attaching.
    """
    @abstractmethod
    def to_base(self):
        """Cast the object into the corresponding native Python container type."""
        pass

    @contextmanager
    def _buffer(self):
        """Allows buffering during modifications.

        This prevents unnecessary communication between C++ and Python.
        """
        self._buffered = True
        yield None
        self._buffered = False

    def _update(self):
        """Signal that the object has been updated."""
        if not self._buffered and self._parent is not None:
            self._parent._handle_update(self, self._label)


def _get_inner_typeconverter(type_def, desired_type):
    """Handles more complicated data structure validataion specifications.

    When using nested containers, where an entry could be a list or None, for
    instance the validation specification must the use something like
    typeconverter's OnlyIf or Either classes. We detect the internal validation
    present in these objects and if they are of the correct type use them in the
    data structure.

    An example of this would be a validation specification of ``Either([None,
    to_type_converter([float]))`` which specifies that we expect either `None`
    or a list of floats. Here we cannot use the `Either` instance for the type
    spec since what we want is the inner `float` requirement for the list.

    Args:
        type_def (`hoomd.data.typeconverter.TypeConverter`): The type converter
            which is desired to be introspected to find the desired type.
        desired_type (`hoomd.data.typeconverter.TypeConverter`): A child class
            of `hoomd.data.typeconverter.TypeConverter` that signals what type
            converter types is needed.
    """
    # If the type_def is the desired type, return it immediately.
    if isinstance(type_def, desired_type):
        return type_def
    # More complicated cases, deals with nested type definitions where the
    # nested containers have other requirements such as multiple valid container
    # representations or entry is allowed to be None. In these cases, we check
    # the TypeConverterValue instance's converter for the desired container type
    # converter type.
    return_type_def = None
    if isinstance(type_def, TypeConverterValue):
        if isinstance(type_def.converter, OnlyIf):
            if isinstance(type_def.converter.cond, desired_type):
                return type_def.converter.cond
        elif isinstance(type_def.converter, Either):
            matches = [spec for spec in type_def.converter.specs if isinstance(
                spec, desired_type)]
            if len(matches) > 1:
                raise ValueError(
                    "Parameter defined with multiple valid definitions of "
                    "the same type.")
            elif len(matches) == 1:
                return_type_def = matches[0]
    # When no valid type converter can be found error out.
    if return_type_def is None:
        raise ValueError("Unable to find type definition for attribute.")
    return return_type_def


def _to_synced_data_structure(data, type_def, parent=None, label=None):
    """Convert raw data to a synced _SyncedDataStructure object.

    This returns ``data`` if the data is not a supported data structure
    (sequence, mapping, or set).

    Args:
        data: Raw data to be potentially converted to a _SyncedDataStructure
            object.
        type_def (`hoomd.data.typeconverter.TypeConverter`): The type converter
            which is desired to be introspected to find the desired type.
        parent: The parent container if any for the data.
        label: Label to indicate to the parent where the child belongs. For
            instance, if the parent is a mapping the label would be the key for
            this value.
    """
    if isinstance(data, Mapping):
        type_validation = _get_inner_typeconverter(
            type_def, TypeConverterMapping)
        return HOOMDDict(type_validation, parent, data, label)
    elif isinstance(data, Sequence) and not isinstance(data, (str, tuple)):
        type_validation = _get_inner_typeconverter(
            type_def, TypeConverterSequence)
        return HOOMDList(type_validation, parent, data, label)
    elif isinstance(data, Set):
        return HOOMDSet(type_def, parent, data, label)
    return data


class HOOMDList(MutableSequence, _SyncedDataStructure):
    """List with type validation.

    Use `to_base` to get a plain `list`.

    Uses `collections.abc.MutableSequence` as a parent class.
    See the Python docs on `list` objects for more information.

    Warning:
        Users should not need to instantiate this class.
    """

    def __init__(self, type_definition, parent, initial_value=None,
                 label=None):
        self._type_definition = type_definition
        self._parent = parent
        self._label = label
        self._buffered = False
        self._data = []
        if initial_value is not None:
            for type_def, val in zip(cycle(type_definition), initial_value):
                self._data.append(
                    _to_synced_data_structure(val, type_def, self))

    def __getitem__(self, index):  # noqa: D105
        return self._data[index]

    def __setitem__(self, index, value):  # noqa: D105
        if isinstance(index, slice):
            with self._buffer():
                for i, v in zip(range(len(self))[index], value):
                    self[i] = v
        else:
            type_def = self._get_type_def(index)
            try:
                validated_value = type_def(value)
            except TypeConversionError as err:
                raise ValueError(
                    f"Error in setting item {index} in list.") from err
            else:
                # If we override a list item that is itself a
                # _SyncedDataStructure we disconnect the old structure to
                # prevent updates from it to force sync the list.
                if isinstance(self._data[i], _SyncedDataStructure):
                    self._data[i]._parent = None

                self._data[i] = _to_synced_data_structure(
                    validated_value, type_def, self)
        self._update()

    def _get_type_def(self, index):
        entry_index = index % len(self._type_definition)
        return self._type_definition[entry_index]

    def __delitem__(self, index):  # noqa: D105
        val = self._data[index]
        # Disconnect item from list
        if isinstance(val, _SyncedDataStructure):
            val._parent = None
        del self._data[index]
        # This is required for list type definitions which have index dependent
        # validation.
        if len(self._type_definition) > 1:
            try:
                _ = self._type_definition(self._data)
            except TypeConversionError as err:
                raise RuntimeError("Deleting items from list has caused an "
                                   "invalid state.") from err
        self._update()

    def __len__(self):  # noqa: D105
        return len(self._data)

    def insert(self, index, value):  # noqa: D102
        if index >= len(self):
            index = len(self)

        type_def = self._get_type_def(index)
        try:
            validated_value = type_def(value)
        except TypeConversionError as err:
            raise TypeConversionError(
                f"Error inserting {value} into list at position {index}."
                ) from err
        else:
            self._data.insert(index,
                              _to_synced_data_structure(validated_value,
                                                        type_def, self))
            if index != len(self) and len(self._type_definition) > 1:
                try:
                    _ = self._type_definition(self._data)
                except TypeConversionError as err:
                    raise TypeConversionError(
                        "List insertion invalidated list.") from err
        self._update()

    def extend(self, other):  # noqa: D102
        with self._buffer():
            super().extend(other)
        self._update()

    def clear(self):  # noqa: D102
        """Empty the list of all items."""
        with self._buffer():
            super().clear()
        self._update()

    def to_base(self):
        """Cast the object to a `list`.

        Recursively calls `to_base` for nested data structures.
        """
        return_data = []
        for entry in self:
            if isinstance(entry, _SyncedDataStructure):
                return_data.append(entry.to_base())
            else:
                try:
                    use_entry = deepcopy(entry)
                except Exception:
                    use_entry = entry
                return_data.append(use_entry)
        return return_data

    def __str__(self):  # noqa: D105
        return str(self._data)

    def __repr__(self):  # noqa: D105
        return repr(self._data)


class HOOMDDict(MutableMapping, _SyncedDataStructure):
    """Mapping with type validation.

    Allows dotted access to key values as well as long as they conform to
    Python's attribute name requirements.

    Uses `collections.abc.MutableMapping` as a parent class. See Python
    documentation on `dict` for more information.

    Use `to_base` to get a plain `dict`.

    Warning:
        Should not be instantiated by users.
    """
    _data = {}

    def __init__(self, type_def, parent, initial_value=None, label=None):
        self._type_definition = type_def
        self._parent = parent
        self._buffered = False
        self._label = label
        self._data = {}
        if initial_value is not None:
            for key, val in initial_value.items():
                self._data[key] = _to_synced_data_structure(
                    val, type_def[key], self, key)

    def __getitem__(self, key):  # noqa: D105
        return self._data[key]

    def __getattr__(self, attr):
        """Support dotted access to keys that are valid Python identifiers."""
        try:
            return self[attr]
        except KeyError:
            raise AttributeError("{} object has no attribute {}.".format(
                self, attr))

    def __setattr__(self, attr, value):  # noqa: D105
        if attr in self._data:
            self[attr] = value
        else:
            super().__setattr__(attr, value)

    def __setitem__(self, key, item):  # noqa: D105
        if key not in self._type_definition:
            raise KeyError(
                "Cannot set value for non-existent key {}.".format(key))
        type_def = self._type_definition[key]
        try:
            validated_value = type_def(item)
        except TypeConversionError as err:
            raise TypeConversionError(
                "Error setting key {}.".format(key)) from err
        else:
            # Disconnect child from parent to prevent child signaling update
            if isinstance(self._data[key], _SyncedDataStructure):
                self._data[key]._parent = None
            self._data[key] = _to_synced_data_structure(
                validated_value, type_def, self, key)
        self._update()

    def __delitem__(self, key):  # noqa: D105
        raise RuntimeError("This class does not support deleting keys.")

    def __iter__(self):  # noqa: D105
        yield from self._data

    def __len__(self):  # noqa: D105
        return len(self._data)

    def update(self, other):  # noqa: D102
        with self._buffer():
            super().update(other)
        self._update()

    def to_base(self):
        """Cast the object to a `dict`.

        Recursively calls `to_base` for nested data structures.
        """
        return_data = {}
        for key, entry in self.items():
            if isinstance(entry, _SyncedDataStructure):
                return_data[key] = entry.to_base()
            else:
                try:
                    use_entry = deepcopy(entry)
                except Exception:
                    use_entry = entry
                return_data[key] = use_entry
        return return_data

    def __str__(self):  # noqa: D105
        return str(self._data)

    def __repr__(self):  # noqa: D105
        return repr(self._data)


class HOOMDSet(MutableSet, _SyncedDataStructure):
    """Set with type validation.

    Use `to_base` to get a plain `set`.

    Uses `collections.abc.MutableSet` as a parent class. See Python
    documentation on `set` for more information.

    Warning:
        Should not be instantiated by users.
    """

    def __init__(self, type_def, parent, initial_value=None, label=None):
        self._type_definition = type_def
        self._parent = parent
        self._buffered = False
        self._label = label
        self._data = set()
        if initial_value is not None:
            for val in initial_value:
                self._data.add(_to_synced_data_structure(val, type_def, self))

    def __contains__(self, item):  # noqa: D105
        return item in self._data

    def __iter__(self):  # noqa: D105
        yield from self._data

    def __len__(self):  # noqa: D105
        return len(self._data)

    def add(self, item):  # noqa: D102
        if item not in self:
            try:
                validated_value = self._type_definition(item)
            except TypeConversionError as err:
                raise TypeConversionError(
                    "Error adding item {} to set.".format(item)) from err
            else:
                self._data.add(_to_synced_data_structure(
                    validated_value, self._type_definition, self))
        self._update()

    def discard(self, item):  # noqa: D102
        """Remove an item from the set if it is contained in the set."""
        if isinstance(item, _SyncedDataStructure):
            # Disconnect child from parent
            if item in self:
                item._parent = None
        self._data.discard(item)

    def to_base(self):
        """Cast the object into a `set`.

        Recursively calls `to_base` for nested data structures.
        """
        return_set = set()
        for item in self:
            if isinstance(item, _SyncedDataStructure):
                return_set.add(item.to_base())
            else:
                try:
                    use_item = deepcopy(item)
                except Exception:
                    use_item = item
                return_set.add(use_item)
        return return_set

    def __ior__(self, other):  # noqa: D105
        with self._buffer():
            super().__ior__(other)
        self._update()
        return self

    def __iand__(self, other):  # noqa: D105
        with self._buffer():
            super().__iand__(other)
        self._update()
        return self

    def __ixor__(self, other):  # noqa: D105
        with self._buffer():
            super().__ixor__(other)
        self._update()
        return self

    def __isub__(self, other):  # noqa: D105
        with self._buffer():
            super().__isub__(other)
        self._update()
        return self

    def __str__(self):  # noqa: D105
        return str(self._data)

    def __repr__(self):  # noqa: D105
        return repr(self._data)
