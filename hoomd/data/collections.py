# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Implement various data structures for syncing C++ and Python data."""

from abc import abstractmethod
from collections import abc
import warnings

import numpy as np

import hoomd
import hoomd.data.typeconverter as _typeconverter


class _ReadAndWrite:
    """Context manager that reads on enter and writes on exit."""

    def __init__(self, collection):
        self._collection = collection

    def __enter__(self):
        self._collection._read()

    def __exit__(self, exec_type, exec_value, exec_traceback):
        self._collection._write()


class _Buffer:
    """Context manager that suspends read and/or write call when active."""

    def __init__(self, collection, read=False, write=False):
        self._collection = collection
        self.read = read
        self.write = write

    def __enter__(self):
        if self.read:
            self._collection._buffer_read = True
            self._collection._children.buffer_read = True
        if self.write:
            self._collection._buffer_write = True
            self._collection._children.buffer_write = True

    def __exit__(self, exec_type, exec_value, exec_traceback):
        if self.read:
            self._collection._buffer_read = False
            self._collection._children.buffer_read = False
        if self.write:
            self._collection._buffer_write = False
            self._collection._children.buffer_write = False


def _find_structural_validator(schema, type_):
    if isinstance(schema, type_):
        return schema
    elif isinstance(schema, _typeconverter.OnlyIf):
        if isinstance(schema.cond, type_):
            return schema.cond
    elif isinstance(schema, _typeconverter.Either):
        for spec in schema.specs:
            if isinstance(spec, type_):
                return spec
    raise RuntimeError("Appropriate validator could not be found.")


class _ChildRegistry(abc.MutableSet):
    """Keeps a record of and isolates children data structures.

    Also, helps handling of buffering for `_HOOMDSyncedCollection` subclasses.
    """

    def __init__(self):
        self._registry = {}
        self._buffer_read = False
        self._buffer_write = False
        self._isolated = False

    def add(self, a):
        if self._isolated:
            return
        if isinstance(a, _HOOMDSyncedCollection):
            self._registry.setdefault(id(a), a)
            a._buffer_read = self._buffer_read
            a._buffer_write = self._buffer_write

    def discard(self, a):
        if self._isolated:
            return
        item = self._registry.pop(id(a), None)
        if item is not None:
            item._isolate()
            item._buffer_read = False
            item._buffer_write = False

    def __contains__(self, obj):
        return id(obj) in self._registry

    def __len__(self):
        return len(self._registry)

    def __iter__(self):
        yield from self._registry.values()

    def add_with_passthrough(self, iterator):
        for item in iterator:
            self.add(item)
            yield item

    @property
    def buffer_read(self):
        return self._buffer_read

    @buffer_read.setter
    def buffer_read(self, value):
        self._buffer_read = value
        for item in self:
            item._buffer_read = value

    @property
    def buffer_write(self):
        return self._buffer_write

    @buffer_write.setter
    def buffer_write(self, value):
        self._buffer_write = value
        for item in self:
            item._buffer_write = value


class _HOOMDSyncedCollection(abc.Collection):
    """Base class for data classes used to support a read-modify-write model.

    Implements features and requirements general to all collections.

    Attributes:
        _root: Either a `ParameterDict` or `TypeParameterDict`. Enables reading
            and writing to C++.
        _schema: Type converter/validator for the element of the data class.
        _parent (_HOOMDSyncedCollection): The top level _HOOMDSyncedCollection
            subclass instance for the
        given root key.
        _identity (str): A string of other identity given by the object's root.
        _isolated (bool): Whether the object is still connected to a root.
        _buffer_read (bool): Whether the object is actively buffering reads.
        _buffer_write (bool): Whether the object is actively buffering writes.
        _children (_ChildRegistry): A record of children data objects for use in
            isolating and buffering nested data.
    """

    def __init__(self, root, schema, parent=None, identity=None):
        self._root = getattr(root, "_root", root)
        if parent is None:
            parent = self
        self._parent = parent
        self._schema = schema
        self._children = _ChildRegistry()
        if identity is None:
            self._identity = getattr(parent, "_identity", None)
        else:
            self._identity = identity
        self._isolated = False
        self._buffer_read = False
        self._buffer_write = False

    def __contains__(self, obj):
        self._read()
        if isinstance(obj, np.ndarray):
            return any(self._numpy_equality(obj, item) for item in self._data)
        for item in self._data:
            if isinstance(item, np.ndarray):
                if self._numpy_equality(item, obj):
                    return True
                continue
            if obj == item:
                return True
        return False

    def __iter__(self):
        self._read()
        yield from self._data

    def __len__(self):
        self._read()
        return len(self._data)

    def __eq__(self, other):
        if isinstance(other, _HOOMDSyncedCollection):
            return self.to_base() == other.to_base()
        return self.to_base() == other

    def _numpy_equality(self, a, b):
        """Whether to consider a and b equal for purposes of __contains__.

        Args:
            a (numpy.ndarray): Any array.
            b (any): Any value

        Returns:
            bool: Whether a and b are equal.
        """
        if not isinstance(b, np.ndarray):
            return False
        return a is b or np.array_equal(a, b, equal_nan=True)

    def to_base(self):
        """Return a base data object (e.g. list, dict, or tuple).

        Acts recursively.
        """
        return _to_base(self)

    @property
    def _read_and_write(self):
        """Context manager for read-modify-write."""
        return _ReadAndWrite(self)

    @property
    def _suspend_read(self):
        """Context manager for buffering reads."""
        return _Buffer(self, True)

    @property
    def _suspend_write(self):
        """Context manager for buffering writes."""
        return _Buffer(self, False, True)

    @property
    def _suspend_read_and_write(self):
        """Context manager for buffering reads and writes."""
        return _Buffer(self, True, True)

    def _read(self):
        """Update data if possible else isolate.

        Relies on root ``_read`` method. Generally updates parent.
        """
        if self._buffer_read:
            return
        if self._isolated:
            warnings.warn(hoomd.error.IsolationWarning())
            return
        self._root._read(self)

    def _write(self):
        """Write data to C++.

        Relies on root ``_write`` method. Generally sets parent.
        """
        if self._buffer_write:
            return
        if self._isolated:
            warnings.warn(hoomd.error.IsolationWarning())
            return
        self._root._write(self)

    @abstractmethod
    def _update(self, new_value):
        """Given a ``new_value`` update internal data isolating if failing.

        Returns:
            succeeded (bool): Whether the instance managed to update data.
        """
        pass

    def _isolate(self):
        """Remove link to root, parent, and children."""
        self._children.clear()
        self._children._isolated = True
        self._parent = None
        self._root = None
        self._identity = None
        self._isolated = True

    def _to_hoomd_data(self, schema, data):
        validated_value = _to_hoomd_data(root=self,
                                         schema=schema,
                                         parent=self._parent,
                                         identity=self._identity,
                                         data=data)

        if isinstance(validated_value, _HOOMDSyncedCollection):
            if self._isolated:
                validated_value._isolate()
        return validated_value

    def _change_root(self, new_root):
        """Used in updating ParameterDict."""
        self._root = new_root
        for child in self._children:
            child._change_root(new_root)

    def _validate(self, schema, data):
        """Validate and convert to _HOOMDSyncedCollection if applicable."""
        return self._to_hoomd_data(schema, schema(data))

    def __repr__(self):
        return f"{type(self).__name__}{self.to_base()}"


class _PopIndicator:
    pass


class _HOOMDDict(_HOOMDSyncedCollection, abc.MutableMapping):
    """Mimic the behavior of a dict."""

    def __init__(self, root, schema, parent=None, identity=None, data=None):
        super().__init__(root, schema, parent, identity)
        self._data = {}
        if data is None:
            return
        with self._suspend_read_and_write:
            for key in data:
                self._data[key] = self._to_hoomd_data(schema[key], data[key])
                self._children.add(self._data[key])

    def __setitem__(self, key, value):
        if key not in self._schema:
            raise KeyError(f"key '{key}' does not exist.")
        validated_value = self._validate(self._schema[key], value)
        with self._read_and_write:
            old_value = self._data.get(key)
            self._children.discard(old_value)
            self._children.add(validated_value)
            self._data[key] = validated_value

    def __getitem__(self, key):
        self._read()
        return self._data[key]

    def __delitem__(self, key):
        with self._read_and_write:
            self._children.discard(self._data[key])
            del self._data[key]

    def _update(self, new_value):
        if not isinstance(new_value, abc.Mapping):
            self._isolate()
            warnings.warn(hoomd.error.IsolationWarning())
            return False
        new_data = {}
        with self._suspend_read_and_write:
            for key, value in new_value.items():
                old_value = self._data.get(key)
                if isinstance(old_value, _HOOMDSyncedCollection):
                    if old_value._update(value):
                        new_data[key] = old_value
                        continue
                    else:
                        self._children.discard(old_value)
                validated_value = self._validate(self._schema[key], value)
                self._children.add(validated_value)
                new_data[key] = validated_value
        self._data = new_data
        return True

    def update(self, other, **kwargs):
        with self._read_and_write, self._suspend_read_and_write:
            for key, value in other.items():
                self[key] = value
            for key, value in kwargs.items():
                self[key] = value

    def clear(self):
        self._children.clear()
        self._data = {}
        self._write()

    def values(self):
        for key in self:
            yield self._data[key]

    def items(self):
        for key in self:
            yield key, self._data[key]

    def pop(self, key, default=_PopIndicator):
        with self._read_and_write:
            if key in self._data:
                value = self._data[key]
                del self._data[key]
                self._children.discard(value)
                return value
        if default is not _PopIndicator:
            return default
        raise KeyError(f"key {key} does not exist.")

    def popitem(self):
        with self._read_and_write, self._suspend_read_and_write:
            key = list(self)[-1]
            return key, self.pop(key)

    def setdefault(self, key, default):
        self._read()
        if key in self._data:
            return
        with self._suspend_read_and_write:
            self[key] = default
        self._write()


class _HOOMDList(_HOOMDSyncedCollection, abc.MutableSequence):
    """Mimic the behavior of a list."""

    def __init__(self, root, schema, parent=None, identity=None, data=None):
        super().__init__(root, schema, parent, identity)
        self._data = []
        if data is None:
            return
        with self._suspend_read_and_write:
            for item in data:
                self._data.append(self._to_hoomd_data(schema, item))
                self._children.add(self._data[-1])

    def __setitem__(self, index, value):
        validated_value = self._validate(value)
        with self._read_and_write:
            old_value = self._data[index]
            self._children.discard(old_value)
            self._children.add(validated_value)
            self._data[index] = validated_value

    def __getitem__(self, index):
        self._read()
        return self._data[index]

    def __delitem__(self, index):
        with self._read_and_write:
            self._children.discard(self._data[index])
            del self._data[index]

    def __reversed__(self):
        self._read()
        return reversed(self._data)

    def __iadd__(self, other):
        with self._read_and_write, self._suspend_read_and_write:
            for item in other:
                validated_item = self._validate(item)
                self._children.add(validated_item)
                self._data.append(validated_item)
        return self

    def __le__(self, other):
        if isinstance(other, _HOOMDSyncedCollection):
            return self.to_base() <= other.to_base()
        return self.to_base() <= other

    def __lt__(self, other):
        if isinstance(other, _HOOMDSyncedCollection):
            return self.to_base() < other.to_base()
        return self.to_base() < other

    def __ge__(self, other):
        if isinstance(other, _HOOMDSyncedCollection):
            return self.to_base() >= other.to_base()
        return self.to_base() >= other

    def __gt__(self, other):
        if isinstance(other, _HOOMDSyncedCollection):
            return self.to_base() > other.to_base()
        return self.to_base() > other

    def __radd__(self, other):
        try:
            return other + self.to_base()
        except AttributeError:
            raise TypeError(f"Cannot add list to object of type {type(other)}.")

    def __add__(self, other):
        if isinstance(other, _HOOMDSyncedCollection):
            return self.to_base() + other.to_base()
        return self.to_base() + other

    def insert(self, index, value):
        value = self._validate(value)
        self._children.add(value)
        with self._read_and_write:
            self._data.insert(index, value)

    def extend(self, other_seq):
        with self._read_and_write, self._suspend_read_and_write:
            for value in other_seq:
                value = self._validate(value)
                self._children.add(value)
                self._data.append(value)

    def clear(self):
        self._children.clear()
        self._data = []
        self._write()

    def remove(self, obj):
        with self._read_and_write:
            with self._suspend_read_and_write:
                i = self._data.index(obj)
                self._children.discard(self._data[i])
                self._data.pop(i)

    def reverse(self):
        with self._read_and_write:
            self._data.reverse()

    def index(self, obj):
        self._read()
        return self._data.index(obj)

    def count(self, obj):
        self._read()
        return self._data.count(obj)

    def _validate(self, value):
        return super()._validate(self._schema, value)

    def _update(self, new_value):
        if not isinstance(new_value, abc.Sequence):
            self._isolate()
            warnings.warn(hoomd.error.IsolationWarning())
            return False
        new_data = []
        with self._suspend_read_and_write:
            for index, value in enumerate(new_value):
                if index < len(self):
                    old_value = self[index]
                    if isinstance(old_value, _HOOMDSyncedCollection):
                        if old_value._update(value):
                            new_data.append(old_value)
                            continue
                        else:
                            self._children.discard(old_value)
                validated_value = self._validate(value)
                self._children.add(validated_value)
                new_data.append(validated_value)
        self._data = new_data
        return True


class _HOOMDTuple(_HOOMDSyncedCollection, abc.Sequence):
    """Mimic the behavior of a tuple."""

    def __init__(self, root, schema, parent=None, identity=None, data=()):
        super().__init__(root, schema, parent, identity)
        self._data = []
        if data is None:
            return
        with self._suspend_read_and_write:
            for converter, item in zip(schema, data):
                self._data.append(self._to_hoomd_data(converter, item))
                self._children.add(self._data[-1])

    def __getitem__(self, index):
        self._read()
        return self._data[index]

    def index(self, obj):
        self._read()
        return self._data.index(obj)

    def count(self, obj):
        self._read()
        return self._data.count(obj)

    def _update(self, new_value):
        if (not isinstance(new_value, abc.Sequence)
                or len(new_value) != len(self._data)):
            self._isolate()
            warnings.warn(hoomd.error.IsolationWarning())
            return False
        new_data = []
        with self._suspend_read_and_write:
            for index, value in enumerate(new_value):
                old_value = self._data[index]
                if isinstance(old_value, _HOOMDSyncedCollection):
                    if old_value._update(value):
                        new_data.append(old_value)
                        continue
                    else:
                        self._children.discard(old_value)
                validated_value = self._validate(self._schema[index], value)
                self._children.add(validated_value)
                new_data.append(validated_value)
        self._data = tuple(new_data)
        return True


def _to_hoomd_data(root, schema, parent=None, identity=None, data=None):
    _exclude_classes = (hoomd.logging.Logger,)
    if isinstance(data, _exclude_classes):
        return data
    # Even though a ndarray is a MutableSequence we need to ensure that it
    # remains a ndarray and not a list when the validation is for an array. In
    # addition, this would error if we allowed the MutableSequence conditional
    # to execute.
    if (isinstance(data, np.ndarray)
            and isinstance(schema, _typeconverter.NDArrayValidator)):
        return data
    if isinstance(data, abc.MutableMapping):
        spec = _find_structural_validator(schema,
                                          _typeconverter.TypeConverterMapping)
        return _HOOMDDict(root, spec, parent, identity, data)
    if isinstance(data, abc.MutableSequence):
        spec = _find_structural_validator(schema,
                                          _typeconverter.TypeConverterSequence)
        return _HOOMDList(root, spec.converter, parent, identity, data)
    if not isinstance(data, str) and isinstance(data, abc.Sequence):
        spec = _find_structural_validator(
            schema, _typeconverter.TypeConverterFixedLengthSequence)
        return _HOOMDTuple(root, spec, parent, identity, data)
    return data


def _to_base(collection):
    if isinstance(collection, _HOOMDSyncedCollection):
        if not collection._isolated:
            collection._read()
        # Suspending reading and writing will also prevent isolation warnings.
        with collection._suspend_read_and_write:
            if isinstance(collection, _HOOMDDict):
                return {
                    key: _to_base(value)
                    for key, value in collection._data.items()
                }
            if isinstance(collection, _HOOMDList):
                return [_to_base(value) for value in collection._data]
            if isinstance(collection, _HOOMDTuple):
                return tuple(_to_base(value) for value in collection._data)
    return collection
