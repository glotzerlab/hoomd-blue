# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Provides the Python analog and wrapper around the C++ class ArrayView.

See hoomd/ArrayView.h for more information.
"""

from collections.abc import MutableSequence
import functools


def _array_view_wrapper(method):
    """Provides the forwarding of methods to the C++ array_view."""

    @functools.wraps(method)
    def wrapped_method(self, *args, **kwargs):
        return getattr(self._get_array_view(), method.__name__)(*args, **kwargs)

    return wrapped_method


class _ArrayViewWrapper(MutableSequence):
    """A wrapper around the C++ array_view.

    Provides safe access to an array_view by letting go of the handle before
    exiting any of its methods. All that is required is a callable that returns
    a pybind11 array_view. Also, handles slices correctly.

    In general, this should be used with `hoomd.data.syncedlist.SyncedList`
    which will treat this object as the C++ list.
    """

    def __init__(self, get_array_view):
        self._get_array_view = get_array_view

    @_array_view_wrapper
    def __len__(self):
        pass

    def _handle_index(self, index, raise_if_negative=True):
        if isinstance(index, slice):
            return range(len(self))[index]
        if index < 0:
            new_index = len(self) + index
            if raise_if_negative and new_index < 0:
                raise IndexError(f"No index {new_index} exists.")
            return new_index
        return index

    def insert(self, index, value):
        array = self._get_array_view()
        index = self._handle_index(index, False)
        if index < 0:
            index = 0
        array.insert(index, value)

    def __delitem__(self, index):
        array = self._get_array_view()
        index = self._handle_index(index)
        if isinstance(index, int):
            del array[index]
            return
        for i in sorted(index, reverse=True):
            del array[i]

    def __getitem__(self, index):
        array = self._get_array_view()
        index = self._handle_index(index)
        if isinstance(index, int):
            return array[index]
        return [array[i] for i in index]

    def __setitem__(self, index, value):
        array = self._get_array_view()
        index = self._handle_index(index)
        array[index] = value

    @_array_view_wrapper
    def append(self, value):
        pass

    @_array_view_wrapper
    def extend(self, value):
        pass

    @_array_view_wrapper
    def clear(self):
        pass

    def pop(self, index=None):
        array = self._get_array_view()
        if index is None:
            return array.pop()
        index = self._handle_index(index)
        return array.pop(index)

    def __iter__(self):
        for i in range(len(self)):
            # yield releases the control flow, so we cannot hold on to a
            # reference of the array_view here.
            yield self[i]
