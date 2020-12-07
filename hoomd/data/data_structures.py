from abc import ABCMeta, abstractmethod
from collections.abc import MutableMapping, MutableSequence, MutableSet
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
        """Cast the object into the corresponding native Python container type.
        """
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
    """Handles more complicated data structure specification.

    When using nested containers where an entry could be a list or None for
    instance requires the use of something like typeconverter's OnlyIf or
    Either. We autdetect the internal type specification for these object and if
    they are of the correct type use these.

    Args:
        type_def (`hoomd.data.typeconverter.TypeConverter`):
            Type converter which is introspected to find the desired type.
        desired_type (`hoomd.data.typeconverter.TypeConverter`):
            A child class of `hoomd.data.typeconverter.TypeConverter` that
            signals what type converter types is needed.
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
                return_type_def = type_def.converter.cond  # Can this be None? If not, return here.
        elif isinstance(type_def.converter, Either):
            matches = [spec for spec in type_def.converter.specs if isinstance(spec, desired_type)]
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
    """Convert raw data to a synced _HOOMDDataStructures object.

    This returns ``data`` if the data is not a supported data structure (sequence, mapping, or set).

    Args:
        data:
            Raw data to be potentially converted to a _HOOMDDataStructures
            object.
        parent:
            The parent container if any for the data.
        label:
            Label to indicate to the parent where the child belongs. For
            instance, if the parent is a mapping the label would be the key for
            this value.
    """
    if isinstance(data, MutableMapping):
        typing = _get_inner_typeconverter(type_def, TypeConverterMapping)
        return HOOMDDict(typing, parent, data, label)
    elif isinstance(data, MutableSequence):
        typing = _get_inner_typeconverter(type_def, TypeConverterSequence)
        return HOOMDList(typing, parent, data, label)
    elif isinstance(data, MutableSet):
        return HOOMDSet(typing, parent, data, label)
    else:
        return data


class HOOMDList(MutableSequence, _HOOMDDataStructures):
    """List with type validation.

    Use `to_base` to get a plain `list`.

    Warning:
        Users should not need to instantiate this class.
    """
    def __init__(self, type_definition, parent, initial_value=None,
                 label=None):
        self._type_definition = type_definition
        self._parent = parent
        self._list = []
        self._label = label
        self._buffered = False
        if initial_value is not None:
            for type_def, val in zip(cycle(type_definition), initial_value):
                self._list.append(_to_hoomd_data_structure(val, type_def, self))

    def __getitem__(self, index):
        """Get the value(s) associated with entry index of the list.

        Supports slicing.
        """
        return self._list[index]

    def __setitem__(self, index, value):
        """Set an item(s) at index to the given value.

        Support assignment with slices.
        """
        if isinstance(index, slice):
            with self._buffer():
                for i, v in zip(self._slice_to_iterable(index), value):
                    self[i] = v
        else:
            type_def = self._get_type_def(index)
            try:
                validated_value = type_def(value)
            except TypeConversionError as err:
                raise TypeConversionError(
                    f"Error in setting item {index} in list.") from err
            else:
                # If we override a list item that is itself a
                # _HOOMDDataStructures we disconnect the old structure to
                # prevent updates from it to force sync the list.
                if isinstance(self._list[i], _HOOMDDataStructures):
                    self._list[i]._parent = None

                self._list[i] = _to_hoomd_data_structure(
                    validated_value, type_def, self)
        self._update()

    def _slice_to_iterable(self, slice_):
        return range(len(self))[slice_]

    def _get_type_def(self, index):
        entry_index = index % len(self._type_definition)
        return self._type_definition[entry_index]

    def __delitem__(self, index):
        """Delele the index-th item from the list.

        Supports slice deletion.
        """
        val = self._list[index]
        # Disconnect item from list
        if isinstance(val, _HOOMDDataStructures):
            val._parent = None
        del self._list[index]
        # This is required for list type definitions which have index dependent
        # validation.
        if len(self._type_definition) > 1:
            try:
                _ = self._type_definition(self._list)
            except TypeConversionError as err:
                raise RuntimeError("Deleting items from list has caused an "
                                   "invalid state.") from err
        self._update()

    def __len__(self):
        """Get the length of the list."""
        return len(self._list)

    def insert(self, index, value):
        """Insert a value at the index-th index of the list.

        Args:
            index : int
                The index to place the value into
            value
                The value to insert into the list.
        """
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
            self._list.insert(index,
                              _to_hoomd_data_structure(validated_value,
                                                       type_def, self))
            if index != len(self) and len(self._type_definition) > 1:
                try:
                    _ = self._type_definition(self._list)
                except TypeConversionError as err:
                    raise TypeConversionError(
                        "List insertion invalidated list.") from err
        self._update()

    def extend(self, other):
        """Add a list to the end of this list.

        Args:
            other : list
                A list or sequence object to add to the end of this list.
        """
        with self._buffer():
            super().extend(other)
        self._update()

    def clear(self):
        """Empty the list of all items."""
        with self._buffer():
            super().clear()
        self._update()

    def to_base(self):
        """Cast the object to a `list`.

        Recursively calls `to_base` for nested data structures.
        """
        return_list = []
        for entry in self:
            if isinstance(entry, _HOOMDDataStructures):
                return_list.append(entry.to_base())
            else:
                try:
                    use_entry = deepcopy(entry)
                except Exception:
                    use_entry = entry
                return_list.append(use_entry)
        return return_list

    def __str__(self):
        return str(self._list)

    def __repr__(self):
        return repr(self._list)


class HOOMDDict(MutableMapping, _HOOMDDataStructures):
    """Mapping with type validation.

    Allows dotted access to key values as well as long as they conform to
    Python's attribute name requirements.

    Use `to_base` to get a plain `dict`.

    Warning:
        Should not be instantiated by users.
    """
    _dict = {}

    def __init__(self, type_def, parent, initial_value=None, label=None):
        self._dict = {}
        self._type_definition = type_def
        self._parent = parent
        self._buffered = False
        self._label = label
        if initial_value is not None:
            for key, val in initial_value.items():
                self._dict[key] = _to_hoomd_data_structure(
                    val, type_def[key], self, key)

    def __getitem__(self, key):
        """Get the item associated with the given key."""
        return self._dict[key]

    def __getattr__(self, attr):
        """Support dotted access to keys that are valid Python identifiers."""
        try:
            return self[attr]
        except KeyError:
            raise AttributeError("{} object has not attribute {}.".format(
                self, attr))

    def __setattr__(self, attr, value):
        """Set"""
        if attr in self._dict:
            self[attr] = value
        else:
            super().__setattr__(attr, value)

    def __setitem__(self, key, item):
        """Set the value associated with key to item in the mapping."""
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
            if isinstance(self._dict[key], _HOOMDDataStructures):
                self._dict[key]._parent = None
            self._dict[key] = _to_hoomd_data_structure(
                validated_value, type_def, self, key)
        self._update()

    def __delitem__(self, key):
        """Delete a key from the mapping."""
        raise RuntimeError("mapping does not support deleting keys.")

    def __iter__(self):
        """Iterate through the mapping keys."""
        yield from self._dict.keys()

    def __len__(self):
        """Get the number of key, value pairs in the mapping."""
        return len(self._dict)

    def update(self, other):
        """Update the mapping with another mapping.

        The other mapping's pairs take priority over this mapping's.

        Args:
            other : dict
                A mapping to update key, value pairs in the calling mapping.
        """
        with self._buffer():
            super().update(other)
        self._update()

    def to_base(self):
        """Cast the object to a `dict`.

        Recursively calls `to_base` for nested data structures.
        """
        return_dict = {}
        for key, entry in self.items():
            if isinstance(entry, _HOOMDDataStructures):
                return_dict[key] = entry.to_base()
            else:
                try:
                    use_entry = deepcopy(entry)
                except Exception:
                    use_entry = entry
                return_dict[key] = use_entry
        return return_dict

    def __str__(self):
        return str(self._dict)

    def __repr__(self):
        return repr(self._dict)


class HOOMDSet(MutableSet, _HOOMDDataStructures):
    """Set with type validation.

    Use `to_base` to get a plain `set`.

    Warning:
        Should not be instantiated by users.
    """
    def __init__(self, type_def, parent, initial_value=None, label=None):
        self._type_definition = type_def
        self._parent = parent
        self._buffered = False
        self._label = label
        self._set = set()
        if initial_value is not None:
            for val in initial_value:
                self._set.add(_to_hoomd_data_structure(val, type_def, self))

    def __contains__(self, item):
        """Whether an item is found in the set."""
        return item in self._set

    def __iter__(self):
        """Iterate over items in the set."""
        yield from self._set

    def __len__(self):
        """Give the number of items in the set."""
        return len(self._set)

    def add(self, item):
        """Add an item to the set if it is not already contained in the set."""
        if item not in self:
            try:
                validated_value = self._type_definition(item)
            except TypeConversionError as err:
                raise TypeConversionError(
                    "Error adding item {} to set.".format(item)) from err
            else:
                self._set.add(_to_hoomd_data_structure(
                    validated_value, self._type_definition, self))
        self._update()

    def discard(self, item):
        """Remove an item from the set if it is contained in the set."""
        if isinstance(item, _HOOMDDataStructures):
            # Disconnect child from parent
            if item in self:
                item._parent = None
        self._set.discard(item)

    def to_base(self):
        """Cast the object into a `set`.

        Recursively calls `to_base` for nested data structures.
        """
        return_set = set()
        for item in self:
            if isinstance(item, _HOOMDDataStructures):
                return_set.add(item.to_base())
            else:
                try:
                    use_item = deepcopy(item)
                except Exception:
                    use_item = item
                return_set.add(use_item)
        return return_set

    def __ior__(self, other):
        """Perform an inplace union with the other set."""
        with self._buffer():
            super().__ior__(other)
        self._update()
        return self

    def __iand__(self, other):
        """Perform an inplace intersection with the other set."""
        with self._buffer():
            super().__iand__(other)
        self._update()
        return self

    def __ixor__(self, other):
        """The inplace union of the set differences of the two sets."""
        with self._buffer():
            super().__ixor__(other)
        self._update()
        return self

    def __isub__(self, other):
        """The inplace set difference of the two sets."""
        with self._buffer():
            super().__isub__(other)
        self._update()
        return self

    def __str__(self):
        return str(self._set)

    def __repr__(self):
        return repr(self._set)
