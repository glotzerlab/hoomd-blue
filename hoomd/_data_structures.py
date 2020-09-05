from abc import ABCMeta, abstractmethod
from collections.abc import MutableMapping, MutableSequence, MutableSet
from contextlib import contextmanager
from copy import deepcopy
from itertools import cycle

from hoomd.typeconverter import (
    TypeConversionError, TypeConverterValue, OnlyIf, Either,
    TypeConverterMapping, TypeConverterSequence, TypeConverter)


class _HOOMDDataStructures(metaclass=ABCMeta):
    @abstractmethod
    def to_base(self):
        pass

    @contextmanager
    def _buffer(self):
        self._buffered = True
        yield None
        self._buffered = False

    def _update(self):
        if self._buffered:
            return
        else:
            self._parent._handle_update(self, self._label)


def _get_inner_typeconverter(type_def, desired_type):
    rtn_type_def = None
    if isinstance(type_def, TypeConverterValue):
        if isinstance(type_def.converter, OnlyIf):
            if isinstance(type_def.converter.cond, desired_type):
                rtn_type_def = type_def.converter.cond
        elif isinstance(type_def.converter, Either):
            for spec in type_def.converter.specs:
                matches = []
                if isinstance(spec, desired_type):
                    matches.append(spec)
                if len(matches) > 1:
                    raise ValueError(
                        "Parameter defined with multiple valid definitions of "
                        "the same type.")
                elif len(matches) == 1:
                    rtn_type_def = matches[0]
    elif isinstance(type_def, desired_type):
        rtn_type_def = type_def
    if rtn_type_def is None:
        raise RuntimeError("Unable to find type definition for attribute.")
    return rtn_type_def


def _to_hoomd_data_structure(data, type_def,
                             parent=None, label=None, callback=None):
    if isinstance(data, MutableMapping):
        typing = _get_inner_typeconverter(type_def, TypeConverterMapping)
        return _HOOMDDict(typing, parent, data, label)
    elif isinstance(data, MutableSequence):
        typing = _get_inner_typeconverter(type_def, TypeConverterSequence)
        return _HOOMDList(typing, parent, data, label)
    elif isinstance(data, MutableSet):
        return _HOOMDSet(typing, parent, data, label, TypeConverterValue)
    else:
        return data


class _HOOMDList(MutableSequence, _HOOMDDataStructures):
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
        return self._list[index]

    def __setitem__(self, index, value):
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
                    "Error in setting item {} in list.".format(index)) from err
            else:
                self._list[i] = _to_hoomd_data_structure(
                    validated_value, type_def, self)
        self._update()

    def _slice_to_iterable(self, slice_):
        start, end, step = slice_.start, slice_.end, slice_.step
        if start is None:
            if step is not None and step < 0:
                iterable_start = len(self) - 1
            else:
                iterable_start = 0
        if end is None:
            if step is not None and step < 0:
                iterable_end = -1
            else:
                iterable_end = len(self)
        step = 1 if step is None else step
        return range(iterable_start, iterable_end, step)

    def _get_type_def(self, index):
        entry_index = index % len(self._type_definition)
        return self._type_definition[entry_index]

    def __delitem__(self, index):
        del self._list[index]
        if len(self._type_definition) > 1:
            try:
                _ = self._type_definition(self._list)
            except TypeConversionError as err:
                raise RuntimeError("Deleting items from list has caused an "
                                   "invalid state.") from err
        self._update()

    def __len__(self):
        return len(self._list)

    def insert(self, index, value):
        if index >= len(self):
            index = len(self)

        type_def = self._get_type_def(index)
        try:
            validated_value = type_def(value)
        except TypeConversionError as err:
            raise TypeConversionError(
                "Error inserting {} into list.".format(value)) from err
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
        with self._buffer():
            super().extend(other)
        self._update()

    def clear(self):
        with self._buffer():
            super().clear()
        self._update()

    def to_base(self):
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


class _HOOMDDict(MutableMapping, _HOOMDDataStructures):
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

    def __getitem__(self, item):
        return self._dict[item]

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            raise AttributeError("{} object has not attribute {}.".format(
                self, attr))

    def __setattr__(self, attr, value):
        if attr in self._dict:
            self[attr] = value
        else:
            super().__setattr__(attr, value)

    def __setitem__(self, key, item):
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
            self._dict[key] = _to_hoomd_data_structure(
                validated_value, type_def, self, key)
        self._update()

    def __delitem__(self, key):
        raise RuntimeError("mapping does not support deleting keys.")

    def __iter__(self):
        yield from self._dict.keys()

    def __len__(self):
        return len(self._dict)

    def update(self, other):
        with self._buffer():
            super().update(other)
        self._update()

    def to_base(self):
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


class _HOOMDSet(MutableSet, _HOOMDDataStructures):
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
        return item in self._set

    def __iter__(self):
        yield from self._set

    def __len__(self):
        return len(self._set)

    def add(self, item):
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
        self._set.discard(item)

    def to_base(self):
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
        with self._buffer():
            super().__ior__(other)
        self._update()
        return self

    def __iand__(self, other):
        with self._buffer():
            super().__iand__(other)
        self._update()
        return self

    def __ixor__(self, other):
        with self._buffer():
            super().__ixor__(other)
        self._update()
        return self

    def __isub__(self, other):
        with self._buffer():
            super().__isub__(other)
        self._update()
        return self

    def __str__(self):
        return str(self._set)

    def __repr__(self):
        return repr(self._set)
