import inspect
from copy import copy


class _PartialIsInstance:
    def __init__(self, classes):
        self.classes = classes

    def __call__(self, instance):
        return isinstance(instance, self.classes)


class _PartialGetAttr:
    def __init__(self, attr):
        self.attr = attr

    def __call__(self, obj):
        return getattr(obj, self.attr)


def identity(obj):
    return obj


class SyncedList:
    """Provides syncing and validation for a python and cpp list.

    Used to ensure once synced that standard list operations effect both
    python and cpp. Also guarentees that list remain in same order when using
    the public API.

    Args:
        validation (function or class): A function that takes one argument
            and returns a boolean based on whether the value is appropriate for
            the list. Can raise ValueError for helpful diagnosis of the problem
            with validation. Alternatively validation can be a class which
            indicates the expected type of items of the list.
        to_synced_list (function, optional): A function that takes one
            argument (a valid SyncedList python value) and does necessary
            conversion before adding to the cpp list. Defaults to a function
            that grabs the ``_cpp_obj`` attribute from the value.
        iterable (iterable, optional): An iterable whose members are valid
            members of the SyncedList instance. Defaults to None which causes
            SyncedList to start with an empty list.
    """

    def __init__(self, validation,
                 to_synced_list=None,
                 iterable=None):
        if to_synced_list is None:
            to_synced_list = identity

        if inspect.isclass(validation):
            self._validate = _PartialIsInstance(validation)
        else:
            self._validate = validation

        self._to_synced_list_conversion = to_synced_list
        self._simulation = None
        self._list = []
        if iterable is not None:
            for it in iterable:
                self.append(it)

    def __contains__(self, value):
        """Returns boolean based on if value is already in _list.

        Based on memory location (python's is).
        """
        for item in self._list:
            if item is value:
                return True
        return False

    def __len__(self):
        return len(self._list)

    def __iter__(self):
        """Iterate through python list."""
        yield from self._list

    def __setitem__(self, index, value):
        """Change self[index] to value.

        Detaches removed value and syncs cpp_list if necessary.
        """
        if len(self) <= index or -len(self) > index:
            raise IndexError("Cannot assign index {} to list of length {}."
                             "".format(index, len(self)))
        else:
            value = self._validate_or_error(value)
            # If synced need to change cpp_list and detach operation before
            # changing python list
            if self._synced:
                self._synced_list[index] = \
                    self._to_synced_list_conversion(value)
                self._list[index]._detach()
            self._list[index]._remove()
            self._list[index] = value

    def __getitem__(self, index):
        """Grabs the python list item."""
        index = self._handle_slices(index)
        if hasattr(index, '__iter__'):
            return [self._list[i] for i in index]
        else:
            return self._list[index]

    def __delitem__(self, index):
        """Deletes an item from list. Handles detaching if necessary."""
        index = self._handle_slices(index)
        if hasattr(index, '__iter__'):
            for pos, i in enumerate(index):
                fixed_index = i - pos if i > 0 else i
                del self[fixed_index]
            return
        if len(self) <= index or -len(self) > index:
            raise IndexError("Cannot delete index {} to list of length {}."
                             "".format(index, len(self)))
        else:
            # Since delitem may not del the underlying object, we need to
            # manually call detach here.
            if self._synced:
                del self._synced_list[index]
                self._list[index]._detach()
            self._list[index]._remove()
            del self._list[index]

    @property
    def _synced(self):
        """Has a cpp_list object means that we are currently syncing."""
        return hasattr(self, "_synced_list")

    def _handle_slices(self, index):
        length = len(self)
        if isinstance(index, slice):
            start = index.start if index.start is not None else 0
            start = start if start >= 0 else length - start
            stop = index.stop if index.stop is not None else len(self)
            stop = stop if stop >= 0 else length - stop
            step = index.step if index.step is not None else 1
            return list(range(start, stop, step))
        else:
            return index

    def synced_iter(self):
        """Iterate over values in the list. Does nothing when not synced.
        """
        if self._synced:
            yield from self._synced_list

    def _value_add_and_attach(self, value):
        """Attaches value if unattached while raising error if already in list.
        """
        if value._added:
            raise RuntimeError("Object cannot be added to two lists.")
        else:
            value._add(self._simulation)
        if self._synced:
            value._attach()
        return value

    def _validate_or_error(self, value):
        """
        Complete error checking and processing of value prior to adding to list.
        """
        try:
            if self._validate(value):
                return self._value_add_and_attach(value)
            else:
                raise ValueError("Value {} could not be validated."
                                 "".format(value))
        except ValueError as verr:
            raise ValueError("Validation failed: {}".format(verr.args[0]))

    def _sync(self, simulation, synced_list):
        """Attach all list items and update for automatic attachment."""
        self._simulation = simulation
        self._synced_list = synced_list
        for item in self:
            item._add(simulation)
            item._attach()
            self._synced_list.append(self._to_synced_list_conversion(item))

    def _unsync(self):
        """Detach all items, clear _synced_list, and remove cpp references."""
        if self._synced:
            for index in range(len(self)):
                del self._synced_list[0]
            for item in self:
                item._detach()
            del self._simulation
            del self._synced_list

    def append(self, value):
        """Append value to list, handling list syncing."""
        value = self._validate_or_error(value)
        if self._synced:
            self._synced_list.append(self._to_synced_list_conversion(value))
        self._list.append(value)

    def insert(self, pos, value):
        """Insert value to list at pos, handling list syncing."""
        value = self._validate_or_error(value)
        if self._synced:
            self._synced_list.insert(pos,
                                     self._to_synced_list_conversion(value)
                                     )
        self._list.insert(pos, value)

    def extend(self, value_list):
        """Add all elements of value_list to list handling list syncing."""
        for value in value_list:
            self.append(value)

    def clear(self):
        """Remove all items from list."""
        for index in range(len(self)):
            del self[0]

    def remove(self, value):
        """Remove all instances of value in list. Uses identity checking."""
        removal_list = []
        for index in range(len(self)):
            if self[index] is value:
                removal_list.append(index - len(removal_list))
        if len(removal_list) == 0:
            raise ValueError(f"{value} is not in list.")
        for index in removal_list:
            del self[index]

    def __getstate__(self):
        state = copy(self.__dict__)
        state['_simulation'] = None
        state.pop('_synced_list', None)
        return state
