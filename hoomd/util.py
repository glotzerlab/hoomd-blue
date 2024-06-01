# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Utilities."""

import hoomd
import io
from collections.abc import Iterable, Mapping, MutableMapping
from hoomd.error import GPUNotAvailableError  # noqa: F401


def _to_camel_case(string):
    """Switch from snake to camelcase strict.

    Note:
        This currently capitalizes the first word which is not correct
        camelcase.
    """
    return string.replace('_', ' ').title().replace(' ', '')


def _is_iterable(obj):
    """Returns True if object is iterable and not a str or dict."""
    return isinstance(obj, Iterable) and not _bad_iterable_type(obj)


def _bad_iterable_type(obj):
    """Returns True if str, dict, or IO (file) type."""
    return isinstance(obj, (str, dict, io.IOBase))


def _dict_map(dict_, func):
    r"""Perform a recursive map on a nested mapping.

    Args:
        dict\_ (dict): The nested mapping to perform the map on.
        func (``callable``): A callable taking in one value use to map over
            dictionary values.

    Returns:
        dict : A `dict` that has the same keys as the passed in ``dict_`` but
            with the values modified by ``func``.

    Note:
        This can be useful for handling dictionaries returned by
        `hoomd.logging.Logger.log`.
    """
    new_dict = dict()
    for key, value in dict_.items():
        if isinstance(value, Mapping):
            new_dict[key] = _dict_map(value, func)
        else:
            new_dict[key] = func(value)
    return new_dict


def _dict_fold(dict_, func, init_value, use_keys=False):
    r"""Perform a recursive fold on a nested mapping's values or keys.

    A fold is for a unnested mapping looks as follows.

    .. code-block:: python

        mapping = {'a': 0, 'b': 1, 'c': 2}
        accumulated_value = 0
        func = lambda x, y: x + y
        for value in mapping.values():
            accumulated_value = func(accumulated_value, value)

    Args:
        dict\_ (dict): The nested mapping to perform the map on.
        func (``callable``): A callable taking in one value use to fold over
            dictionary values or keys if ``use_keys`` is set.
        init_value: An initial value to use for the fold.
        use_keys (`bool`, optional): If true use keys instead of values for the
            fold. Defaults to ``False``.

    Returns:
        The final value of the fold.
    """
    final_value = init_value
    for key, value in dict_.items():
        if isinstance(value, dict):
            final_value = _dict_fold(value, func, final_value)
        else:
            if use_keys:
                final_value = func(key, final_value)
            else:
                final_value = func(value, final_value)
    return final_value


def _dict_flatten(dict_):
    r"""Flattens a nested mapping into a flat mapping.

    Args:
        dict\_ (dict): The nested mapping to flatten.

    Returns:
        dict: The flattened mapping as a `dict`.

    Note:
        This can be useful for handling dictionaries returned by
        `hoomd.logging.Logger.log`.
    """
    return _dict_flatten_implementation(dict_, None)


def _dict_flatten_implementation(value, key):
    if key is None:
        new_dict = dict()
        for key, inner in value.items():
            new_dict.update(_dict_flatten_implementation(inner, (key,)))
        return new_dict
    elif not isinstance(value, dict):
        return {key: value}
    else:
        new_dict = dict()
        for k, val in value.items():
            new_dict.update(_dict_flatten_implementation(val, key + (k,)))
        return new_dict


def _dict_filter(dict_, filter_):
    r"""Perform a recursive filter on a nested mapping.

    Args:
        dict\_ (dict): The nested mapping to perform the filter on.
        func (``callable``): A callable taking in one value use to filter over
            mapping values.

    Returns:
        dict : A `dict` that has the same keys as the passed in ``dict_`` with
        key value pairs with values that the filter returned ``False`` for
        removed.

    Note:
        This can be useful for handling dictionaries returned by
        `hoomd.logging.Logger.log`.

    """
    new_dict = dict()
    for key in dict_:
        if not isinstance(dict_[key], Mapping):
            if filter_(dict_[key]):
                new_dict[key] = dict_[key]
        else:
            sub_dict = _dict_filter(dict_[key], filter_)
            if sub_dict:
                new_dict[key] = sub_dict
    return new_dict


def _keys_helper(dict_, key=()):
    for k in dict_:
        if isinstance(dict_[k], dict):
            yield from _keys_helper(dict_[k], key + (k,))
            continue
        yield key + (k,)


class _NamespaceDict(MutableMapping):
    """A nested dictionary which can be nested indexed by tuples."""

    def __init__(self, dict_=None):
        self._dict = {} if dict_ is None else dict_

    def __len__(self):
        return _dict_fold(self._dict, lambda x, incr: incr + 1, 0)

    def __iter__(self):
        yield from _keys_helper(self._dict)

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
        """Does not check that key exists."""
        if isinstance(namespace, str):
            namespace = (namespace,)
        parent_dict = self._unsafe_getitem(namespace[:-1])
        del parent_dict[namespace[-1]]

    def __contains__(self, namespace):
        try:
            namespace = self.validate_namespace(namespace)
        except ValueError:
            return False
        current_dict = self._dict
        # traverse through dictionary hierarchy
        for name in namespace:
            try:
                if name in current_dict:
                    current_dict = current_dict[name]
                    continue
                else:
                    return False
            except (TypeError, AttributeError):
                return False
        return True

    def validate_namespace(self, namespace):
        if isinstance(namespace, str):
            namespace = (namespace,)
        if not isinstance(namespace, tuple):
            raise ValueError("Expected a string or tuple namespace.")
        return namespace


class _SafeNamespaceDict(_NamespaceDict):
    """A _NamespaceDict where keys cannot be overwritten."""

    def __setitem__(self, namespace, value):
        if namespace in self:
            raise KeyError("Namespace {} is being used. Remove before "
                           "replacing.".format(namespace))
        else:
            super().__setitem__(namespace, value)


def make_example_simulation(device=None,
                            dimensions=3,
                            particle_types=['A'],
                            mpcd_types=None):
    """Make an example Simulation object.

    The simulation state contains two particles at positions (-1, 0, 0) and
    (1, 0, 0).

    Args:
        device (hoomd.device.Device): The device to use. Create a
            `hoomd.device.CPU` when `None`.

        dimensions (int): Number of dimensions (2 or 3).

        particle_types (list[str]): Particle type names.

        mpcd_types (list[str]): If not `None`, also create two MPCD particles,
            and include these type names in the snapshot.

    Returns:
        hoomd.Simulation: The simulation object.

    Note:
        `make_example_simulation` is intended for use in the documentation and
        other minimal working examples. Use `hoomd.Simulation` directly in other
        cases.

    .. rubric:: Example:

    .. code-block:: python

        simulation = hoomd.util.make_example_simulation()

    """
    if device is None:
        device = hoomd.device.CPU()

    snapshot = hoomd.Snapshot()
    if snapshot.communicator.rank == 0:
        snapshot.particles.N = 2
        snapshot.particles.position[:] = [(-1, 0, 0), (1, 0, 0)]
        snapshot.particles.types = particle_types
        Lz = 10
        if dimensions == 2:
            Lz = 0
        snapshot.configuration.box = [10, 10, Lz, 0, 0, 0]

        if mpcd_types is not None:
            if not hoomd.version.mpcd_built:
                raise RuntimeError("MPCD component not built")
            snapshot.mpcd.N = 2
            snapshot.mpcd.position[:] = [(-1, 0, 0), (1, 0, 0)]
            snapshot.mpcd.types = mpcd_types

    simulation = hoomd.Simulation(device=device)
    simulation.create_state_from_snapshot(snapshot)

    # Ensure that documentation examples test attached objects.
    simulation.run(0)

    return simulation
