# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: joaander

R""" Utilities.
"""

import sys
from copy import deepcopy
import traceback
import os.path
import linecache
import re
import hoomd
from hoomd import _hoomd

## \internal
# \brief Compatibility definition of a basestring for python 2/3
try:
    _basestring = basestring
except NameError:
    _basestring = str

## \internal
# \brief Checks if a variable is an instance of a string and always returns a list.
# \param s Variable to turn into a list
# \returns A list
def listify(s):
    if isinstance(s, _basestring):
        return [s]
    else:
        return list(s)

## \internal
# \brief Internal flag tracking if status lines should be quieted
_status_quiet_count = 0

def cuda_profile_start():
    """ Start CUDA profiling.

    When using nvvp to profile CUDA kernels in hoomd jobs, you usually don't care about all the initialization and
    startup. cuda_profile_start() allows you to not even record that. To use, uncheck the box "start profiling on
    application start" in your nvvp session configuration. Then, call cuda_profile_start() in your hoomd script when
    you want nvvp to start collecting information.

    Example::

        from hoomd import *
        init.read_xml("init.xml")
        # setup....
        run(30000);  # warm up and auto-tune kernel block sizes
        option.set_autotuner_params(enable=False);  # prevent block sizes from further autotuning
        cuda_profile_start()
        run(100)

    """
    # check if initialization has occurred
    if not hoomd.init.is_initialized():
        raise RuntimeError("Cannot start profiling before initialization\n")

    if hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
        hoomd.context.current.device.cpp_exec_conf.cudaProfileStart()

def cuda_profile_stop():
    """ Stop CUDA profiling.

        See Also:
            :py:func:`cuda_profile_start()`.
    """
    # check if initialization has occurred
    if not hoomd.init.is_initialized():
        hoomd.context.current.device.cpp_msg.error("Cannot stop profiling before initialization\n")
        raise RuntimeError('Error stopping profile')

    if hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
        hoomd.context.current.device.cpp_exec_conf.cudaProfileStop()


def to_camel_case(string):
    return string.replace('_', ' ').title().replace(' ', '')


def is_iterable(obj):
    '''Returns True if object is iterable and not a str or dict.'''
    return hasattr(obj, '__iter__') and not bad_iterable_type(obj)


def bad_iterable_type(obj):
    '''Returns True if str or dict.'''
    return isinstance(obj, str) or isinstance(obj, dict)


def dict_map(dict_, func):
    new_dict = dict()
    for key, value in dict_.items():
        if isinstance(value, dict):
            new_dict[key] = dict_map(value, func)
        else:
            new_dict[key] = func(value)
    return new_dict


def dict_fold(dict_, func, init_value, use_keys=False):
    final_value = init_value
    for key, value in dict_.items():
        if isinstance(value, dict):
            final_value = dict_fold(value, func, final_value)
        else:
            if use_keys:
                final_value = func(key, final_value)
            else:
                final_value = func(value, final_value)
    return final_value


def dict_flatten(dict_):
    return _dict_flatten(dict_, None)


def _dict_flatten(value, key):
    if key is None:
        new_dict = dict()
        for key, inner in value.items():
            new_dict.update(_dict_flatten(inner, (key,)))
        return new_dict
    elif not isinstance(value, dict):
        return {key: value}
    else:
        new_dict = dict()
        for k, val in value.items():
            new_dict.update(_dict_flatten(val, key + (k,)))
        return new_dict


class NamespaceDict:
    def __init__(self, dict_=None):
        self._dict = dict() if dict_ is None else dict_

    def __len__(self):
        return dict_fold(self._dict, lambda x, incr: incr + 1, 0)

    def key_exists(self, namespace):
        try:
            namespace = self.validate_namespace(namespace)
        except ValueError:
            return False
        current_dict = self._dict
        # traverse through dictionary hierarchy
        for name in namespace:
            try:
                if name in current_dict.keys():
                    current_dict = current_dict[name]
                    continue
                else:
                    return False
            except (TypeError, AttributeError):
                return False
        return True

    def keys(self):
        raise NotImplementedError

    def _pop_namespace(self, namespace):
        return (namespace[-1], namespace[:-1])

    def _setitem(self, namespace, value):
        # Grab parent dictionary creating sub dictionaries as necessary
        parent_dict = self._dict
        base_name, parent_namespace = self._pop_namespace(namespace)
        for name in parent_namespace:
            # If key does not exist create key with empty dictionary
            try:
                parent_dict = parent_dict[name]
            except KeyError:
                parent_dict[name] = dict()
                parent_dict = parent_dict[name]
        # Attempt to set the value
        parent_dict[base_name] = value

    def __setitem__(self, namespace, value):
        try:
            namespace = self.validate_namespace(namespace)
        except ValueError:
            raise KeyError("Expected a tuple or string key.")
        self._setitem(namespace, value)

    def __getitem__(self, namespace):
        return self._unsafe_getitem(namespace)

    def _unsafe_getitem(self, namespace):
        ret_val = self._dict
        if isinstance(namespace, str):
            namespace = (namespace,)
        try:
            for name in namespace:
                ret_val = ret_val[name]
        except (TypeError, KeyError):
            raise KeyError("Namespace {} not in dictionary.".format(namespace))
        return ret_val

    def __delitem__(self, namespace):
        '''Does not check that key exists.'''
        if isinstance(namespace, str):
            namespace = (namespace,)
        parent_dict = self._unsafe_getitem(namespace[:-1])
        del parent_dict[namespace[-1]]

    def __contains__(self, namespace):
        return self.key_exists(namespace)

    def validate_namespace(self, namespace):
        if isinstance(namespace, str):
            namespace = (namespace,)
        if not isinstance(namespace, tuple):
            raise ValueError("Expected a string or tuple namespace.")
        return namespace


class SafeNamespaceDict(NamespaceDict):
    def __setitem__(self, namespace, value):
        if namespace in self:
            raise KeyError("Namespace {} is being used. Remove before "
                           "replacing.".format(namespace))
        else:
            super().__setitem__(namespace, value)

    def __getitem__(self, namespace):
        return deepcopy(super().__getitem__(namespace))


# Functions for parsing stringified tuples
def _escaped_character(string, end):
    esc_char = string[end + 1]
    return _escaped_character.dict.get(esc_char, esc_char)


_escaped_character.dict = {'n': '\n', 't': '\t', 'r': '\r', 'b': '\b',
                           'f': '\f', 'v': '\v', '0': '\0', "'": "'", '"': '"'}


def str_to_tuple_parse(string):
    type_list = []
    next_type = ''
    final_location = len(string) - 2
    quote = string[1]
    beg = 2
    end = 2
    # find all types until the last
    while end < final_location:
        # if the type name is complete
        if string[end] == quote:
            type_list.append(next_type + string[beg:end])
            next_type = ''
            # Move to next type beginning
            end += 3
            quote = string[end]
            end += 1
            beg = end
        # Convert escaped character
        elif string[end] == '\\':
            next_type += string[beg:end] + _escaped_character(string, end)
            end += 2
            beg = end
        # Otherwise move forward one character
        else:
            end += 1
    # Add the last type
    type_list.append(next_type + string[beg:end])
    return tuple(type_list)


def str_to_tuple_keys(dict_):
    return {str_to_tuple_parse(key): value
            for key, value in dict_.items()}


def is_constructor(obj):
    type_ = type(obj)
    if type_ == type or type in type_.__mro__:
        return True
    else:
        return False
