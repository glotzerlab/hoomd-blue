# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Synced list utility classes."""

from collections.abc import MutableSequence
from copy import copy
import weakref

from hoomd.data.typeconverter import _BaseConverter, TypeConverter


class _PartialGetAttr:
    """Allows partial function application of isinstance over attributes.

    This is a solution to avoid lambdas to enable pickling. We cannot use
    functools.partial since we need to partially apply the second argument.
    """

    def __init__(self, attr):
        self.attr = attr

    def __call__(self, obj):
        return getattr(obj, self.attr)


def identity(obj):
    """Returns obj."""
    return obj


class SyncedList(MutableSequence):
    """Provides syncing and validation for a Python and C++ list.

    Warning:
        This class should not be instantiated by users, and this documentation
        is mostly for developers of HOOMD-blue. The class is documentated to
        highlight the object's API which is that of a `MutableSequence`.

    This class ensures that standard list methods affect both
    Python and C++.

    Args:
        validation (callable or class): A callable that takes one argument
            and returns a boolean based on whether the value is appropriate for
            the list. Can raise ValueError for helpful diagnosis of the problem
            with validation. Alternatively validation can be a class which
            indicates the expected type of items of the list.
        to_synced_list (callable, optional): A callable that takes one
            argument (a Python SyncedList) and does necessary
            conversion before adding to the C++ list. Defaults to simply passing
            the object to C++.
        iterable (iterable, optional): An iterable whose members are valid
            members of the SyncedList instance. Defaults to None which causes
            SyncedList to start with an empty list.
        callable_class (bool, optional): If a class is passed as validation and
            this is ``True`` (defaults to ``False``), then the class will be
            treated as a callable and not used for type checking.
        attach_members (bool, optional): Whether list members can be attached
            (defaults to ``True``). If ``True`` then the `SyncedList` object
            handles adding, attaching, detaching, and removing. If not, these
            steps are skipped regardless of synced status.
    """

    # Also guarantees that lists remain in same order when using the public API.

    def __init__(self,
                 validation,
                 to_synced_list=None,
                 iterable=None,
                 callable_class=False,
                 attach_members=True):
        self._attach_members = attach_members
        self._simulation_ = None
        if to_synced_list is None:
            to_synced_list = identity

        if not isinstance(validation, TypeConverter):
            self._validate = _BaseConverter.to_base_converter(validation)
        else:
            self._validate = validation

        self._to_synced_list_conversion = to_synced_list
        self._synced = False
        self._list = []
        if iterable is not None:
            for it in iterable:
                self.append(it)

    def __len__(self):
        """int: Length of the list."""
        return len(self._list)

    def __setitem__(self, index, value):
        """Set self[index] to value.

        Detaches removed value and syncs cpp_list if necessary.
        """
        # Convert negative to positive indices and validate index
        index = self._handle_int(index)
        value = self._validate_or_error(value)
        self._register_item(value)
        # If synced need to change cpp_list and detach operation before
        # changing python list
        if self._synced:
            self._synced_list[index] = \
                self._to_synced_list_conversion(value)
        self._unregister_item(self._list[index])
        self._list[index] = value

    def __getitem__(self, index):
        """Grabs the python list item."""
        index = self._handle_index(index)
        # since _handle_index always returns a range or int we can safely use an
        # isinstance check here.
        if isinstance(index, range):
            return [self._list[i] for i in index]
        return self._list[index]

    def __delitem__(self, index):
        """Deletes an item from list. Handles detaching if necessary."""
        index = self._handle_index(index)
        # since _handle_index always returns a range or int we can safely use an
        # isinstance check here.
        if isinstance(index, range):
            # We must iterate from highest value to lowest to ensure we don't
            # accidentally try to delete an index that doesn't exist any more.
            for i in sorted(index, reverse=True):
                del self[i]
            return
        # Since delitem may not del the underlying object, we need to
        # manually call detach here.
        if self._synced:
            del self._synced_list[index]
        old_value = self._list.pop(index)
        self._unregister_item(old_value)

    def insert(self, index, value):
        """Insert value to list at index, handling list syncing."""
        value = self._validate_or_error(value)
        self._register_item(value)
        # Wrap index like normal but allow for inserting a new element to the
        # beginning or end of the list for out of bounds index values.
        if index <= -len(self):
            index = 0
        elif index >= len(self):
            index = len(self)
        else:
            index = self._handle_int(index)
        if self._synced:
            self._synced_list.insert(index,
                                     self._to_synced_list_conversion(value))
        self._list.insert(index, value)

    def _handle_int(self, integer):
        """Converts negative indices to positive and validates index."""
        if integer < 0:
            if -integer > len(self):
                raise IndexError(
                    f"Negative index {integer} is too small for list of length "
                    f"{len(self)}")
            return integer % max(1, len(self))
        if integer >= len(self):
            raise IndexError(
                f"Index {integer} is outside bounds of a length {len(self)}"
                f"list.")
        return integer

    def _handle_index(self, index):
        if isinstance(index, slice):
            return self._handle_slice(index)
        return self._handle_int(index)

    def _handle_slice(self, index):
        return range(len(self))[index]

    def _synced_iter(self):
        """Iterate over values in the list. Does nothing when not synced."""
        if self._synced:
            yield from self._synced_list

    def _register_item(self, value):
        """Attaches and/or adds value to simulation if unattached.

        Raises an error if value is already in this or another list.
        """
        if not self._attach_members:
            return
        if self._synced:
            value._attach(self._simulation)
            return
        else:
            if value._attached:
                raise RuntimeError(
                    f"Cannot place {value} into two simulations.")

    def _unregister_item(self, value):
        """Detaches and/or removes value to simulation if attached.

        Args:
            value (``any``): The member of the synced list to dissociate from
                its current simulation.
        """
        if not self._attach_members:
            return
        if self._synced:
            value._detach()

    def _validate_or_error(self, value):
        """Complete error checking and processing of value."""
        try:
            processed_value = self._validate(value)
        except ValueError as verr:
            raise ValueError(f"Validation failed: {verr.args[0]}") from verr
        return processed_value

    @property
    def _simulation(self):
        sim = self._simulation_
        if sim is not None:
            sim = sim()
            if sim is not None:
                return sim
            else:
                self._unsync()

    @_simulation.setter
    def _simulation(self, simulation):
        if simulation is not None:
            simulation = weakref.ref(simulation)
        self._simulation_ = simulation

    def _sync(self, simulation, synced_list):
        """Attach all list items and update for automatic attachment."""
        self._simulation = simulation
        self._synced_list = synced_list
        self._synced = True
        # We use a try except block here to maintain valid state (_synced in
        # this case) even when facing an error.
        try:
            for item in self:
                self._register_item(item)
                self._synced_list.append(self._to_synced_list_conversion(item))
        except Exception as err:
            self._synced = False
            raise err

    def _unsync(self):
        """Detach all items, clear _synced_list, and remove cpp references."""
        if not self._synced:
            return
        # while not strictly necessary we check self._attach_members here to
        # avoid looping unless necessary (_unregister_item checks
        # self._attach_members as well) making the check a slight performance
        # bump for non-attaching members.
        self._simulation = None
        if self._attach_members:
            for item in self:
                self._unregister_item(item)
        del self._synced_list
        self._synced = False

    def __getstate__(self):
        """Get state for pickling."""
        state = copy(self.__dict__)
        state['_simulation_'] = None
        state.pop('_synced_list', None)
        return state

    def __eq__(self, other):
        """Test for equality."""
        return (len(self) == len(other)
                and all(a == b for a, b in zip(self, other)))
