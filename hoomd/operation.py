# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Operation class types.

Operations act on the state of the system at defined points during the
simulation's run loop. Add operation objects to the `Simulation.operations`
collection.

See Also:
    `hoomd.Operations`

    `hoomd.Simulation`
"""

# Operation is a parent class of almost all other HOOMD objects.

from copy import copy
import itertools

import hoomd
from hoomd.logging import Loggable
from hoomd.data.parameterdicts import ParameterDict


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
            TypeParameters for the class/instance.
        _skip_for_equality (set[str]): The attribute names to not use for
            equality checks used during tests. This will not effect attributes
            that exist due to ``__getattr__`` such as those from ``_param_dict``
            or ``_typeparam_dict``.
    """
    _reserved_default_attrs = dict(_param_dict=ParameterDict,
                                   _typeparam_dict=dict)
    _override_setattr = set()

    _skip_for_equality = set()

    def __getattr__(self, attr):
        if attr in self._reserved_default_attrs:
            default = self._reserved_default_attrs[attr]
            value = default() if callable(default) else default
            object.__setattr__(self, attr, value)
            return value
        elif attr in self._param_dict:
            return self._getattr_param(attr)
        elif attr in self._typeparam_dict:
            return self._getattr_typeparam(attr)
        else:
            return self._getattr_hook(attr)

    def _getattr_hook(self, attr):
        raise AttributeError(f"Object {type(self)} has no attribute {attr}")

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
        self._param_dict[attr] = value

    def _setattr_typeparam(self, attr, value):
        """Hook for setting an attribute in `_typeparam_dict`."""
        try:
            for k, v in value.items():
                self._typeparam_dict[attr][k] = v
        except TypeError:
            raise ValueError("To set {}, you must use a dictionary "
                             "with types as keys.".format(attr))

    def __dir__(self):
        """Expose all attributes for dynamic querying in notebooks and IDEs."""
        return super().__dir__() + [
            k for k in itertools.chain(self._param_dict, self._typeparam_dict)
        ]


class _DependencyRelation:
    """Defines a dependency relationship between Python objects.

    For the class to work all dependencies must occur between objects of
    subclasses of this class. This is not an abstract base class since many
    object that use this class may not deal directly with dependencies.

    Note:
        This only handles one way dependencies. Circular dependencies are out of
        scope for now.

    Note:
        This class expects that the ``_dependents`` and ``_dependencies`` are
        available.

    Note:
        We could be more specific in the inheritance of this class to only use
        it when the class needs to deal with a dependency.
    """

    def _add_dependent(self, obj):
        """Adds a dependent to the object's dependent list."""
        if obj not in self._dependents:
            self._dependents.append(obj)
            obj._dependencies.append(self)

    def _add_dependency(self, obj):
        """Adds a dependency to the object's dependency list."""
        if obj not in self._dependencies:
            obj._dependents.append(self)
            self._dependencies.append(obj)

    def _notify_disconnect(self):
        """Notify that an object is being removed from all relationships.

        Notifies dependent object that it is being removed, and removes itself
        from its dependencies' list of dependents. By default the method passes
        itself to all dependents' ``_handle_removed_dependency`` methods.

        Note:
            If more information is needed to pass to _handle_removed_dependency,
            then overwrite this method.
        """
        for dependent in self._dependents:
            dependent._handle_removed_dependency(self)
        self._dependents = []
        for dependency in self._dependencies:
            dependency._remove_dependent(self)
        self._dependencies = []

    def _handle_removed_dependency(self, obj):
        """Handles having a dependency removed.

        Default behavior does nothing. Overwrite to enable handling detaching of
        dependencies.
        """
        pass

    def _remove_dependent(self, obj):
        """Removes a dependent from the list of dependencies."""
        try:
            self._dependents.remove(obj)
        except ValueError:
            pass


class _HOOMDBaseObject(_HOOMDGetSetAttrBase,
                       _DependencyRelation,
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
    _reserved_default_attrs = {
        **_HOOMDGetSetAttrBase._reserved_default_attrs, '_cpp_obj': None,
        '_simulation': None,
        '_dependents': list,
        '_dependencies': list
    }

    _skip_for_equality = {
        '_cpp_obj', '_dependents', '_dependencies', '_simulation'
    }
    _remove_for_pickling = ('_simulation', '_cpp_obj')

    def _detach(self):
        if self._attached:
            self._unapply_typeparam_dict()
            self._param_dict._detach()
            if hasattr(self._cpp_obj, "notifyDetach"):
                self._cpp_obj.notifyDetach()
            # In case the C++ object is necessary for proper disconnect
            # notification we call _notify_disconnect here as well.
            self._notify_disconnect()
            self._cpp_obj = None
            return self

    def _attach(self):
        self._apply_param_dict()
        self._apply_typeparam_dict(self._cpp_obj, self._simulation)

    @property
    def _attached(self):
        return self._cpp_obj is not None

    def _add(self, simulation):
        self._simulation = simulation

    def _remove(self):
        # Since objects can be added without being attached, we need to call
        # _notify_disconnect on both _remove and _detach. The method should be
        # do nothing after being called onces so being called twice is not a
        # concern. I should note that if
        # `hoomd.operations.Operations._unschedule` is called this is
        # invalidated, but as that is not public facing this should be fine.
        self._notify_disconnect()
        self._simulation = None

    @property
    def _added(self):
        return self._simulation is not None

    def _apply_param_dict(self):
        self._param_dict._attach(self._cpp_obj)

    def _apply_typeparam_dict(self, cpp_obj, simulation):
        for typeparam in self._typeparam_dict.values():
            try:
                typeparam._attach(cpp_obj, simulation.state)
            except ValueError as err:
                raise err.__class__(
                    f"For {type(self)} in TypeParameter {typeparam.name} "
                    f"{str(err)}")

    def _unapply_typeparam_dict(self):
        for typeparam in self._typeparam_dict.values():
            typeparam._detach()

    def _add_typeparam(self, typeparam):
        self._append_typeparam(typeparam)

    def _append_typeparam(self, typeparam):
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
    """Represents an operation.

    Operations in the HOOMD-blue data scheme are objects that *operate* on a
    `hoomd.Simulation` object. They broadly consist of 5 subclasses: `Updater`,
    `Writer`, `Compute`, `Tuner`, and `Integrator`. All HOOMD-blue operations
    inherit from one of these five base classes. To find the purpose of each
    class see its documentation.

    Warning:
        This class should not be instantiated by users. The class can be used
        for `isinstance` or `issubclass` checks.

    Note:
        Developers or those contributing to HOOMD-blue, see our architecture
        `file`_ for information on HOOMD-blue's architecture decisions regarding
        operations.

    .. _file: https://github.com/glotzerlab/hoomd-blue/blob/trunk-minor/ \
        ARCHITECTURE.md
    """


class TriggeredOperation(Operation):
    """Operations that include a trigger to determine when to run.

    Warning:
        This class should not be instantiated by users. The class can be used
        for `isinstance` or `issubclass` checks.
    """

    def __init__(self, trigger):
        trigger_param = ParameterDict(trigger=hoomd.trigger.Trigger)
        self._param_dict.update(trigger_param)
        self.trigger = trigger


class Updater(TriggeredOperation):
    """Change the simulation's state.

    An updater is an operation which modifies a simulation's state.

    Warning:
        This class should not be instantiated by users. The class can be used
        for `isinstance` or `issubclass` checks.
    """
    _cpp_list_name = 'updaters'


class Writer(TriggeredOperation):
    """Write output that depends on the simulation's state.

    A writer is an operation which writes out a simulation's state.

    Warning:
        This class should not be instantiated by users. The class can be used
        for `isinstance` or `issubclass` checks.
    """
    _cpp_list_name = 'analyzers'


class Compute(Operation):
    """Compute properties of the simulation's state.

    A compute is an operation which computes some property for another operation
    or use by a user.

    Warning:
        This class should not be instantiated by users. The class can be used
        for `isinstance` or `issubclass` checks.
    """
    pass


class Tuner(TriggeredOperation):
    """Adjust the parameters of other operations to improve performance.

    A tuner is an operation which tunes the parameters of another operation for
    performance or other reasons. A tuner does not modify the current microstate
    of the simulation. That is a tuner does not change quantities like
    temperature, particle position, or the number of bonds in a simulation.

    Warning:
        This class should not be instantiated by users. The class can be used
        for `isinstance` or `issubclass` checks.
    """
    pass


class Integrator(Operation):
    """Advance the simulation state forward one time step.

    An integrator is the operation which evolves a simulation's state in time.
    In `hoomd.hpmc`, integrators perform particle based Monte Carlo moves. In
    `hoomd.md`, the `hoomd.md.Integrator` class organizes the forces, equations
    of motion, and other factors of the given simulation.

    Warning:
        This class should not be instantiated by users. The class can be used
        for `isinstance` or `issubclass` checks.
    """

    def _attach(self):
        self._simulation._cpp_sys.setIntegrator(self._cpp_obj)
        super()._attach()

        # The integrator has changed, update the number of DOF in all groups
        self._simulation.state.update_group_dof()
