"""Implement various data structures for syncing C++ and Python data."""

from abc import abstractmethod
from collections import abc

import hoomd.data.typeconverter as _typeconverter


class _ReadAndWrite:

    def __init__(self, collection):
        self._collection = collection

    def __enter__(self):
        self._collection._read()

    def __exit__(self, exec_type, exec_value, exec_traceback):
        self._collection._write()


class _Buffer:

    def __init__(self, collection, read=False, write=False):
        self._collection = collection
        self.read = read
        self.write = write

    def __enter__(self):
        if self.read:
            self._collection._buffer_read = True
            for child in self._collection._children:
                child._buffer_read = True
        if self.write:
            self._collection._buffer_write = True
            for child in self._collection._children:
                child._buffer_write = True

    def __exit__(self, exec_type, exec_value, exec_traceback):
        if self.read:
            self._collection._buffer_read = False
            for child in self._collection._children:
                child._buffer_read = False
        if self.write:
            self._collection._buffer_write = False
            for child in self._collection._children:
                child._buffer_write = False


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

    def __init__(self):
        self._registry = {}

    def add(self, a):
        if isinstance(a, _HOOMDSyncedCollection):
            self._registry.setdefault(id(a), a)

    def discard(self, a):
        item = self._registry.pop(id(a), None)
        if item is not None:
            item._isolate()

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


class _HOOMDSyncedCollection(abc.Collection):

    def __init__(self, root, schema, parent=None, identity=None):
        self._root = getattr(root, "_root", root)
        self._parent = parent
        self._schema = schema
        self._children = _ChildRegistry()
        if identity is None:
            self._identity = getattr(root, "_identity", None)
        else:
            self._identity = identity
        self._isolated = False

    def __contains__(self, obj):
        self._read()
        return obj in self._data

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

    def _sync(self, obj):
        self._root._sync(self)

    def to_base(self):
        return _to_base(self)

    @property
    def _read_and_write(self):
        return _ReadAndWrite(self)

    @property
    def _suspend_read(self):
        return _Buffer(self, True)

    @property
    def _suspend_write(self):
        return _Buffer(self, False, True)

    @property
    def _suspend_read_and_write(self):
        return _Buffer(self, True, True)

    def _read(self):
        if self._isolated:
            raise ValueError(
                "The collection is no longer connected to its original data "
                "source. The likely cause is the attribute being set to a "
                "different type. To use this data call obj.to_base().")
        if self._buffer_read:
            return
        self._root._read(self)

    def _write(self):
        if self._isolated:
            raise ValueError(
                "The collection is no longer connected to its original data "
                "source. The likely cause is the attribute being set to a "
                "different type.")
        if self._buffer_write:
            return
        self._root._write(self)

    @abstractmethod
    def _update(self):
        pass

    def _isolate(self):
        self._children.clear()
        self._isolated = True

    def __repr__(self):
        return f"{type(self).__name__}{self.to_base()}"


class _PopIndicator:
    pass


class _HOOMDDict(_HOOMDSyncedCollection, abc.MutableMapping):

    def __init__(self, root, schema, parent=None, identity=None, data=None):
        if parent is None:
            parent = self
        super().__init__(root, schema, parent, identity)
        if data is None:
            self._data = {}
        else:
            with self._suspend_read_and_write:
                self._data = {
                    key: _to_hoomd_data(self, schema[key], data[key], parent)
                    for key in data
                }
                for value in self._data.values():
                    self._children.add(value)

    def __setitem__(self, key, value):
        if key not in self._schema:
            raise KeyError(f"key '{key}' does not exist.")
        validated_value = self._validate(key, value)
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

    def _validate(self, key, value):
        return _to_hoomd_data(root=self,
                              schema=self._schema[key],
                              data=self._schema[key](value),
                              parent=self._parent)

    def _update(self, new_value):
        if not isinstance(new_value, abc.Mapping):
            raise ValueError(
                "Mapping is no longer valid. The attribute has been set to a "
                "different type elsewhere in the program.")
            new_data = {}
            for key, value in new_value.items():
                if key in self:
                    old_value = self._data.get(key)
                    if isinstance(old_value, _HOOMDSyncedCollection):
                        try:
                            old_value._update(value)
                        except ValueError:
                            self._children.discard(old_value)
                        else:
                            new_data[key] = old_value
                            continue
                new_value = _to_hoomd_data(self, self._schema[key], value,
                                           self._parent)
                self._children.add(new_value)
                new_data[key] = new_value
            self._data = new_data

    def update(self, other, **kwargs):
        with self._read_and_write:
            with self._suspend_read_and_write:
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
        with self._read_and_write:
            with self._suspend_read_and_write:
                key = list(self)[-1]
                return key, self.pop(key)

    def setdefault(self, key, default):
        with self._read_and_write:
            with self._suspend_read_and_write:
                if key in self._data:
                    return
                self[key] = default


class _HOOMDList(_HOOMDSyncedCollection, abc.MutableSequence):

    def __init__(self, root, schema, parent=None, identity=None, data=None):
        if parent is None:
            parent = self
        super().__init__(root, schema, parent, identity)
        if data is None:
            self._data = []
        else:
            with self._suspend_read_and_write:
                self._data = [
                    _to_hoomd_data(self, schema, item, parent) for item in data
                ]
                for item in self._data:
                    self._children.add(item)

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

    def insert(self, index, value):
        value = self._validate(value)
        self._children.add(value)
        with self._read_and_write:
            self._data.insert(index, value)

    def extend(self, other_seq):
        with self._read_and_write:
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
        return _to_hoomd_data(root=self,
                              schema=self._schema,
                              data=self._schema(value),
                              parent=self._parent)

    def _update(self, new_value):
        if not isinstance(new_value, abc.Sequence):
            raise ValueError(
                "Sequence is no longer valid. The attribute has been set to a "
                "different type elsewhere in the program.")
            new_data = []
            for index, value in enumerate(new_value):
                if index < len(self):
                    old_value = self[index]
                    if isinstance(old_value, _HOOMDSyncedCollection):
                        try:
                            old_value._update(value)
                        except ValueError:
                            self._children.discard(old_value)
                        else:
                            new_data.append(old_value)
                            continue
                new_value = _to_hoomd_data(self, self._schema, value,
                                           self._parent)
                self._children.add(new_value)
                new_data.append(new_value)
            self._data = new_data


class _HOOMDTuple(_HOOMDSyncedCollection, abc.Sequence):

    def __init__(self, root, schema, parent=None, identity=None, data=()):
        if parent is None:
            parent = self
        super().__init__(root, schema, parent, identity)
        with self._suspend_read_and_write:
            self._data = [
                _to_hoomd_data(self, converter, item, parent, identity)
                for converter, item in zip(schema, data)
            ]
            for item in self._data:
                self._children.add(item)

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
        if not isinstance(new_value, abc.Sequence):
            raise ValueError(
                "Sequence is no longer valid. The attribute has been set to a "
                "different type elsewhere in the program.")
        if len(new_value) != len(self._data):
            self._children.clear()
            raise ValueError("Immutable sequence cannot change length.")
        new_data = []
        for index, value in enumerate(new_value):
            old_value = self._data[index]
            if isinstance(old_value, _HOOMDSyncedCollection):
                try:
                    old_value._update(value)
                except ValueError:
                    self._children.discard(old_value)
                else:
                    new_data.append(old_value)
                    continue
            new_value = _to_hoomd_data(self, self._schema[index], value,
                                       self._parent)
            self._children.add(new_value)
            new_data.append(new_value)
        self._data = tuple(new_data)


def _to_hoomd_data(root, schema, data, parent=None, identity=None):
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
