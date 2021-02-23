# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: joaander / All Developers are free to add commands for new features

"""Base classes for all HOOMD-blue operations."""

# Operation is a parent class of almost all other HOOMD objects.
# Triggered objects should inherit from _TriggeredOperation.


from copy import copy, deepcopy

from hoomd.util import is_iterable, dict_map, dict_filter
from hoomd.trigger import Trigger
from hoomd.variant import Variant, Constant
from hoomd.filter import ParticleFilter
from hoomd.logging import Loggable, log
from hoomd.data.typeconverter import RequiredArg
from hoomd.util import NamespaceDict
from hoomd._hoomd import GSDStateReader
from hoomd.data.parameterdicts import ParameterDict


def _convert_values_to_log_form(value):
    """Function for making state loggable quantity conform to spec.

    Since the state dictionary is composed of properties for a given class
    instance that does not have flags associated with it, we need to add the
    flags when querying for the state. This does makes state logger type flag
    generation dynamic meaning that we must be careful that we won't wrongly
    detect different flags for the same attribute. In general this shouldn't
    be a concern, though.
    """
    if value is RequiredArg:
        return RequiredArg
    elif isinstance(value, Variant):
        if isinstance(value, Constant):
            return (value.value, 'scalar')
        else:
            return (value, 'object')
    elif isinstance(value, Trigger) or isinstance(value, ParticleFilter):
        return (value, 'object')
    elif isinstance(value, Operation):
        return (value, 'object')
    elif isinstance(value, str):
        return (value, 'string')
    elif (is_iterable(value)
            and len(value) != 0
            and all([isinstance(v, str) for v in value])):
        return (value, 'strings')
    elif not is_iterable(value):
        return (value, 'scalar')
    else:
        return (value, 'sequence')


def _handle_gsd_arrays(arr):
    if arr.size == 1:
        return arr[0]
    if arr.ndim == 1:
        if arr.size < 3:
            return tuple(arr.flatten())
    else:
        return arr


class _ReturnCopies:
    def __init__(self, obj):
        self.obj = obj

    def __call__(self):
        return deepcopy(self.obj)


class _HOOMDGetSetAttrBase:
    """Provides the use of `ParameterDicts` and `TypeParameterDicts` as attrs.

    Provides many hooks for varying behavior.

    Attributes:
        _reserved_default_attrs (dict[str, Callable([], T)]): Attributes that
            have defaults and should be set using `object.__setattr__`. Has
            `_param_dict` and `_typeparam_dict` keys by default.
        _override_setattr (set[str]): Attributes that should not use the
            provided `__setattr__`. `super().__setattr__` is called for them.
            Likely, this wil no longer be necessary when triggers are added to
            C++ Updaters and Analyzers.
        _param_dict (ParameterDict): The `ParameterDict` for the class/instance.
        _typeparam_dict (dict[str, TypeParameter]): A dict of all the
            `TypeParameter`s for the class/instance.
    """
    _reserved_default_attrs = dict(_param_dict=ParameterDict,
                                   _typeparam_dict=dict)
    _override_setattr = set()

    def __getattr__(self, attr):
        if attr in self._reserved_default_attrs.keys():
            value = self._reserved_default_attrs[attr]()
            object.__setattr__(self, attr, value)
            return value
        elif attr in self._param_dict.keys():
            return self._getattr_param(attr)
        elif attr in self._typeparam_dict.keys():
            return self._getattr_typeparam(attr)
        else:
            raise AttributeError("Object {} has no attribute {}".format(
                type(self), attr))

    def _getattr_param(self, attr):
        """Hook for getting an attribute from `_param_dict`."""
        return self._param_dict[attr]

    def _getattr_typeparam(self, attr):
        """Hook for getting an attribute from `_typeparam_dict`."""
        return self._typeparam_dict[attr]

    def __setattr__(self, attr, value):
        if attr in self._override_setattr:
            super().__setattr__(attr, value)
        elif attr in self._param_dict.keys():
            self._setattr_param(attr, value)
        elif attr in self._typeparam_dict.keys():
            self._setattr_typeparam(attr, value)
        else:
            self._setattr_hook(attr, value)

    def _setattr_hook(self, attr, value):
        """Used when attr is not found in `_param_dict` or `_typeparam_dict`."""
        super().__setattr__(attr, value)

    def _setattr_param(self, attr, value):
        """Hook for setting an attribute in `_param_dict`."""
        old_value = self._param_dict[attr]
        self._param_dict[attr] = value
        new_value = self._param_dict[attr]
        if self._attached:
            try:
                setattr(self._cpp_obj, attr, new_value)
            except (AttributeError):
                self._param_dict[attr] = old_value
                raise AttributeError("{} cannot be set after cpp"
                                     " initialization".format(attr))

    def _setattr_typeparam(self, attr, value):
        """Hook for setting an attribute in `_typeparam_dict`."""
        try:
            for k, v in value.items():
                self._typeparam_dict[attr][k] = v
        except TypeError:
            raise ValueError("To set {}, you must use a dictionary "
                             "with types as keys.".format(attr))


class _DependencyRelation:
    """Defines a dependency relationship between Python objects.

    For the class to work all dependencies must occur between objects of
    subclasses of this class. This is not an abstract base class since many
    object that use this class may not deal directly with dependencies.

    Note:
        We could be more specific in the inheritance of this class to only use
        it when the class needs to deal with a dependency.
    """
    def __init__(self):
        self._dependents = []
        self._dependencies = []

    def _add_dependent(self, obj):
        """Adds a dependent to the object's dependent list."""
        if obj not in self._dependencies:
            self._dependents.append(obj)
            obj._dependencies.append(self)

    def _notify_disconnect(self, *args, **kwargs):
        """Notify that an object is being removed from all relationships.

        Notifies dependent object that it is being removed, and removes itself
        from its dependencies' list of dependents. Uses ``args`` and
        ``kwargs`` to allow flexibility in what information is given to
        dependents from dependencies.

        Note:
            This implementation does require that all dependents take in the
            same information, or at least that the passed ``args`` and
            ``kwargs`` can be used for all dependents'
            ``_handle_removed_dependency`` method.
        """
        for dependent in self._dependents:
            dependent.handle_detached_dependency(self, *args, **kwargs)
        self._dependents = []
        for dependency in self._dependencies:
            dependency._remove_dependent(self)
        self._dependencies = []

    def _handle_removed_dependency(self, obj, *args, **kwargs):
        """Handles having a dependency removed.

        Must be implemented by objects that have dependencies. Uses ``args`` and
        ``kwargs`` to allow flexibility in what information is given to
        dependents from dependencies.
        """
        pass

    def _remove_dependent(self, obj):
        """Removes a dependent from the list of dependencies."""
        self._dependencies.remove(obj)


class _HOOMDBaseObject(_HOOMDGetSetAttrBase, _DependencyRelation,
                       metaclass=Loggable):
    """Handles attaching/detaching to a simulation.

    ``_StatefulAttrBase`` handles getting and setting attributes as well as
    providing an API for getting object state and creating new objects from that
    state information. We overwrite ``_getattr_param`` and ``_setattr_param``
    hooks to handle internal C++ objects. For a similar reason, we overwrite the
    ``state`` property.

    ``_DependencyRelation`` handles dealing with dependency relationships
    between objects.

    The class's metaclass `hoomd.logging.Loggable` handles the logging
    infrastructure for HOOMD-blue objects.

    This class's main features are handling attaching and detaching from
    simulations and adding and removing from containing object such as methods
    for MD integrators and updaters for the operations list. Attaching is the
    idea of creating a C++ object that is tied to a given simulation while
    detaching is removing an object from its simulation.
    """
    _reserved_default_attrs = {**_HOOMDGetSetAttrBase._reserved_default_attrs,
                               '_cpp_obj': _ReturnCopies(None),
                               '_dependents': _ReturnCopies([]),
                               '_dependencies': _ReturnCopies([])}

    _skip_for_equality = set(['_cpp_obj', '_dependent_list'])
    _remove_for_pickling = ('_simulation', '_cpp_obj')

    def _getattr_param(self, attr):
        if self._attached:
            return getattr(self._cpp_obj, attr)
        else:
            return self._param_dict[attr]

    def _setattr_param(self, attr, value):
        self._param_dict[attr] = value
        if self._attached:
            new_value = self._param_dict[attr]
            try:
                setattr(self._cpp_obj, attr, new_value)
            except (AttributeError):
                raise AttributeError("{} cannot be set after cpp"
                                     " initialization".format(attr))

    def __eq__(self, other):
        other_keys = set(other.__dict__.keys())
        for key in self.__dict__.keys():
            if key in self._skip_for_equality:
                continue
            else:
                if key not in other_keys \
                        or self.__dict__[key] != other.__dict__[key]:
                    return False
        return True

    def _detach(self):
        if self._attached:
            self._unapply_typeparam_dict()
            self._update_param_dict()
            self._cpp_obj.notifyDetach()

            self._cpp_obj = None
            self._notify_disconnect(self._simulation)
            return self

    def _attach(self):
        self._apply_param_dict()
        self._apply_typeparam_dict(self._cpp_obj, self._simulation)

        # pass the system communicator to the object
        if self._simulation._system_communicator is not None:
            self._cpp_obj.setCommunicator(self._simulation._system_communicator)

    @property
    def _attached(self):
        return self._cpp_obj is not None

    def _add(self, simulation):
        self._simulation = simulation

    def _remove(self):
        del self._simulation

    @property
    def _added(self):
        return hasattr(self, '_simulation')

    def _apply_param_dict(self):
        for attr, value in self._param_dict.items():
            try:
                setattr(self, attr, value)
            except AttributeError:
                pass

    def _apply_typeparam_dict(self, cpp_obj, simulation):
        for typeparam in self._typeparam_dict.values():
            try:
                typeparam._attach(cpp_obj, simulation)
            except ValueError as verr:
                raise ValueError("In TypeParameter {}:"
                                 " ".format(typeparam.name) + verr.args[0])

    def _update_param_dict(self):
        for key in self._param_dict.keys():
            self._param_dict[key] = getattr(self, key)

    def _unapply_typeparam_dict(self):
        for typeparam in self._typeparam_dict.values():
            typeparam._detach()

    def _add_typeparam(self, typeparam):
        self._typeparam_dict[typeparam.name] = typeparam

    def _extend_typeparam(self, typeparams):
        for typeparam in typeparams:
            self._add_typeparam(typeparam)

    @property
    def _children(self):
        """A set of child objects.

        These objects do not appear directly in any of the operations lists but
        are owned in lists or members of those operations.
        """
        return []

    def __getstate__(self):
        state = copy(self.__dict__)
        for attr in self._remove_for_pickling:
            state.pop(attr, None)
        return state


class Operation(_HOOMDBaseObject):
    """Represents operations that are added to an `hoomd.Operations` object.

    Operations in the HOOMD-blue data scheme are objects that *operate* on a
    `hoomd.Simulation` object. They broadly consist of 5 subclasses: `Updater`,
    `Writer`, `Compute`, `Tuner`, and `hoomd.integrate.BaseIntegrator`. All
    HOOMD-blue operations inherit from one of these five base classes. To find
    the purpose of each class see its documentation.
    """
    pass


class _TriggeredOperation(Operation):
    _cpp_list_name = None

    _override_setattr = {'trigger'}

    def __init__(self, trigger):
        trigger_dict = ParameterDict(trigger=Trigger)
        trigger_dict['trigger'] = trigger
        self._param_dict.update(trigger_dict)

    @property
    def trigger(self):
        return self._param_dict['trigger']

    @trigger.setter
    def trigger(self, new_trigger):
        # Overwrite python trigger
        old_trigger = self.trigger
        self._param_dict['trigger'] = new_trigger
        new_trigger = self.trigger
        if self._attached:
            sys = self._simulation._cpp_sys
            triggered_ops = getattr(sys, self._cpp_list_name)
            for index in range(len(triggered_ops)):
                op, trigger = triggered_ops[index]
                # If tuple is the operation and trigger according to memory
                # location (python's is), replace with new trigger
                if op is self._cpp_obj and trigger is old_trigger:
                    triggered_ops[index] = (op, new_trigger)

    def _attach(self):
        super()._attach()

    def _update_param_dict(self):
        if self._attached:
            for key in self._param_dict:
                if key == 'trigger':
                    continue
                self._param_dict[key] = getattr(self._cpp_obj, key)


class Updater(_TriggeredOperation):
    """Base class for all HOOMD updaters.

    An updater is an operation which modifies a simulation's state.

    Note:
        This class should not be instantiated by users. The class can be used
        for `isinstance` or `issubclass` checks.
    """
    _cpp_list_name = 'updaters'


class Writer(_TriggeredOperation):
    """Base class for all HOOMD analyzers.

    An analyzer is an operation which writes out a simulation's state.

    Note:
        This class should not be instantiated by users. The class can be used
        for `isinstance` or `issubclass` checks.
    """
    _cpp_list_name = 'analyzers'


class Compute(Operation):
    """Base class for all HOOMD computes.

    A compute is an operation which computes some property for another operation
    or use by a user.

    Note:
        This class should not be instantiated by users. The class can be used
        for `isinstance` or `issubclass` checks.
    """
    pass


class Tuner(Operation):
    """Base class for all HOOMD tuners.

    A tuner is an operation which tunes the parameters of another operation for
    performance or other reasons. A tuner does not modify the current microstate
    of the simulation. That is a tuner does not change quantities like
    temperature, particle position, or the number of bonds in a simulation.

    Note:
        This class should not be instantiated by users. The class can be used
        for `isinstance` or `issubclass` checks.
    """
    pass
