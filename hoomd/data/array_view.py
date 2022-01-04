# Copyright (c) 2009-2022 The Regents of the University of Michigan.
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

    @_array_view_wrapper
    def insert(self, index, value):
        pass

    def __delitem__(self, index):
        array = self._get_array_view()
        if not isinstance(index, slice):
            del array[index]
            return
        for i in sorted([j for j in range(len(array))[index]], reverse=True):
            del array[i]

    def __getitem__(self, index):
        array = self._get_array_view()
        if not isinstance(index, slice):
            return array[index]
        return [array[index] for index in range(len(array))[index]]

    @_array_view_wrapper
    def __setitem__(self, index, value):
        pass

    @_array_view_wrapper
    def append(self, value):
        pass

    @_array_view_wrapper
    def extend(self, value):
        pass

    @_array_view_wrapper
    def clear(self):
        pass

    @_array_view_wrapper
    def pop(self, index=None):
        pass

    def __iter__(self):
        for i in range(len(self)):
            # yield releases the control flow, so we cannot hold on to a
            # reference of the array_view here.
            yield self[i]
