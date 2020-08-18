# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: joaander / All Developers are free to add commands for new features

R""" Classes for (all/most) HOOMD objects.

_Operation is inherented by almost all other HOOMD objects.
_TriggeredOperation is _Operation for objects that are triggered.
"""

from hoomd.util import is_iterable, dict_map, str_to_tuple_keys
from hoomd.trigger import Trigger
from hoomd.variant import Variant, Constant
from hoomd.filter import _ParticleFilter
from hoomd.logging import Loggable, log
from hoomd.typeconverter import RequiredArg
from hoomd.util import NamespaceDict
from hoomd._hoomd import GSDStateReader
from hoomd.parameterdicts import ParameterDict

from copy import deepcopy


class NotAttachedError(RuntimeError):
    """ Raised when something that requires attachment happens before attaching """
    pass


def convert_values_to_log_form(value):
    if value is RequiredArg:
        return (None, 'scalar')
    elif isinstance(value, Variant):
        if isinstance(value, Constant):
            return (value.value, 'scalar')
        else:
            return (value, 'object')
    elif isinstance(value, Trigger) or isinstance(value, _ParticleFilter):
        return (value, 'object')
    elif isinstance(value, _Operation):
        return (value, 'object')
    elif isinstance(value, str):
        return (value, 'string')
    elif is_iterable(value) and all([isinstance(v, str) for v in value]):
        return (value, 'strings')
    elif not is_iterable(value):
        return (value, 'scalar')
    else:
        return (value, 'multi')


def handle_gsd_arrays(arr):
    if arr.size == 1:
        return arr[0]
    if arr.ndim == 1:
        if arr.size < 3:
            return tuple(arr.flatten())
    else:
        return arr


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
                self, attr))

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
        if self.is_attached:
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


class _Operation(_HOOMDGetSetAttrBase, metaclass=Loggable):
    _reserved_default_attrs = {**_HOOMDGetSetAttrBase._reserved_default_attrs,
                               '_cpp_obj': lambda: None,
                               '_dependent_list': lambda: []}

    _skip_for_equality = set(['_cpp_obj', '_dependent_list'])

    def _getattr_param(self, attr):
        if self.is_attached:
            return getattr(self._cpp_obj, attr)
        else:
            return self._param_dict[attr]

    def _setattr_param(self, attr, value):
        self._param_dict[attr] = value
        if self.is_attached:
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

    def __del__(self):
        if self.is_attached and hasattr(self, '_simulation'):
            self.notify_detach(self._simulation)

    def detach(self):
        self._unapply_typeparam_dict()
        self._update_param_dict()
        self._cpp_obj = None
        if hasattr(self, '_simulation'):
            self.notify_detach(self._simulation)
            del self._simulation
        return self

    def add_dependent(self, obj):
        self._dependent_list.append(obj)

    def notify_detach(self, sim):
        new_objs = []
        for dependent in self._dependent_list:
            new_objs.extend(dependent.handle_detached_dependency(sim, self))
        return new_objs

    def handle_detached_dependency(self, sim, obj):
        self.detach()
        new_objs = self.attach(sim)
        return new_objs if new_objs is not None else []

    def attach(self, simulation):
        self._apply_param_dict()
        self._apply_typeparam_dict(self._cpp_obj, simulation)

        # pass the system communicator to the object
        if simulation._system_communicator is not None:
            self._cpp_obj.setCommunicator(simulation._system_communicator)

    @property
    def is_attached(self):
        return self._cpp_obj is not None

    def _apply_param_dict(self):
        for attr, value in self._param_dict.items():
            try:
                setattr(self, attr, value)
            except AttributeError:
                pass

    def _apply_typeparam_dict(self, cpp_obj, simulation):
        for typeparam in self._typeparam_dict.values():
            try:
                typeparam.attach(cpp_obj, simulation)
            except ValueError as verr:
                raise ValueError("In TypeParameter {}:"
                                 " ".format(typeparam.name) + verr.args[0])

    def _update_param_dict(self):
        for key in self._param_dict.keys():
            self._param_dict[key] = getattr(self, key)

    def _unapply_typeparam_dict(self):
        for typeparam in self._typeparam_dict.values():
            typeparam.detach()

    def _add_typeparam(self, typeparam):
        self._typeparam_dict[typeparam.name] = typeparam

    def _extend_typeparam(self, typeparams):
        for typeparam in typeparams:
            self._add_typeparam(typeparam)

    def _typeparam_states(self):
        state = {name: tp.state for name, tp in self._typeparam_dict.items()}
        return deepcopy(state)

    @log(flag='state')
    def state(self):
        self._update_param_dict()
        state = self._typeparam_states()
        state['params'] = dict(self._param_dict)
        return dict_map(state, convert_values_to_log_form)

    @classmethod
    def from_state(cls, state, final_namespace=None, **kwargs):
        state_dict, unused_args = cls._get_state_dict(state,
                                                      final_namespace,
                                                      **kwargs)
        return cls._from_state_with_state_dict(state_dict, **unused_args)

    @classmethod
    def _get_state_dict(cls, data, final_namespace, **kwargs):

        # resolve the namespace
        namespace = list(cls._export_dict.values())[0].namespace
        if final_namespace is not None:
            namespace = namespace[:-1] + (final_namespace,)
        namespace = namespace + ('state',)
        # Filenames
        if isinstance(data, str):
            if data.endswith('gsd'):
                state, kwargs = cls._state_from_gsd(data, namespace, **kwargs)

        # Dictionaries and like objects
        elif isinstance(data, dict):
            try:
                # try to grab the namespace
                state = deepcopy(NamespaceDict(data)[namespace])
            except KeyError:
                # if namespace can't be found assume that dictionary is the
                # state dictionary (This assumes that values are of the form
                # (value, flag)
                try:
                    state = dict_map(data, lambda x: x[0])
                except TypeError:
                    state = deepcopy(data)

        elif isinstance(data, NamespaceDict):
            state = deepcopy(data[namespace])

        # Data is of an unusable type
        else:
            raise ValueError("Object {} cannot be used to get state."
                             "".format(data))

        return (state, kwargs)

    @classmethod
    def _state_from_gsd(cls, filename, namespace, **kwargs):
        if 'frame' not in kwargs.keys():
            frame = -1
        else:
            frame = kwargs['frame']
            del kwargs['frame']
        # Grab state keys from gsd
        reader = GSDStateReader(filename, frame)
        namespace_str = 'log/' + '/'.join(namespace)
        state_chunks = reader.getAvailableChunks(namespace_str)
        state_dict = NamespaceDict()
        chunk_slice = slice(len(namespace_str) + 1, None)
        # Build up state dict
        for state_chunk in state_chunks:
            state_dict_key = tuple(state_chunk[chunk_slice].split('/'))
            state_dict[state_dict_key] = \
                handle_gsd_arrays(reader.readChunk(state_chunk))
        return (state_dict._dict, kwargs)

    @classmethod
    def _from_state_with_state_dict(cls, state, **kwargs):

        # Initialize object using params from state and passed arguments
        params = state['params']
        params.update(kwargs)
        obj = cls(**params)
        del state['params']

        # Add typeparameter information
        for name, tp_dict in state.items():
            if '__default' in tp_dict.keys():
                obj._typeparam_dict[name].default = tp_dict['__default']
                del tp_dict['__default']
            # Parse the stringified tuple back into tuple
            if obj._typeparam_dict[name]._len_keys > 1:
                tp_dict = str_to_tuple_keys(tp_dict)
            setattr(obj, name, tp_dict)
        return obj


class _TriggeredOperation(_Operation):
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
        if self.is_attached:
            sys = self._simulation._cpp_sys
            triggered_ops = getattr(sys, self._cpp_list_name)
            for index in range(len(triggered_ops)):
                op, trigger = triggered_ops[index]
                # If tuple is the operation and trigger according to memory
                # location (python's is), replace with new trigger
                if op is self._cpp_obj and trigger is old_trigger:
                    triggered_ops[index] = (op, new_trigger)

    def attach(self, simulation):
        self._simulation = simulation
        super().attach(simulation)


class _Updater(_TriggeredOperation):
    _cpp_list_name = 'updaters'


class _Analyzer(_TriggeredOperation):
    _cpp_list_name = 'analyzers'


class _Tuner(_Operation):
    pass
