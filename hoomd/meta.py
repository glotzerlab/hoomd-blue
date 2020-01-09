# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: joaander / All Developers are free to add commands for new features

R""" Write out simulation and environment context metadata.

Metadata is stored in form of key-value pairs in a JSON file and used
to summarize the per-run simulation parameters so that they can be easily
taken up by other scripts and stored in a database.

Example::

    metadata = meta.dump_metadata()
    meta.dump_metadata(filename = "metadata.json", user = {'debug': True}, indent=2)

"""

import hoomd
from hoomd.util import is_iterable, dict_map, str_to_tuple_keys
from hoomd.triggers import PeriodicTrigger, Trigger
from hoomd.logger import Loggable
from hoomd.util import NamespaceDict
import json
import time
import datetime
import copy

from collections import OrderedDict
from collections import Mapping
from copy import deepcopy


def note_type(value):
    if not is_iterable(value):
        return (value, 'scalar')
    elif isinstance(value, str):
        return (value, 'string')
    else:
        return (value, 'multi')


class _Operation(metaclass=Loggable):
    _reserved_attrs_with_dft = {'_cpp_obj': lambda: None,
                                '_param_dict': dict,
                                '_typeparam_dict': dict,
                                '_dependent_list': lambda: []}

    def __getattr__(self, attr):
        if attr in self._reserved_attrs_with_dft.keys():
            setattr(self, attr, self._reserved_attrs_with_dft[attr]())
            return self.__dict__[attr]
        elif attr in self._param_dict.keys():
            return self._getattr_param(attr)
        elif attr in self._typeparam_dict.keys():
            return self._getattr_typeparam(attr)
        else:
            raise AttributeError("Object {} has no attribute {}"
                                 "".format(self, attr))

    def _getattr_param(self, attr):
        if self.is_attached:
            return getattr(self._cpp_obj, attr)
        else:
            return self._param_dict[attr]

    def _getattr_typeparam(self, attr):
        return self._typeparam_dict[attr]

    def __setattr__(self, attr, value):
        if attr in self._reserved_attrs_with_dft.keys():
            super().__setattr__(attr, value)
        elif attr in self._param_dict.keys():
            self._setattr_param(attr, value)
        elif attr in self._typeparam_dict.keys():
            self._setattr_typeparam(attr, value)
        else:
            super().__setattr__(attr, value)

    def _setattr_param(self, attr, value):
        if self.is_attached:
            try:
                setattr(self._cpp_obj, attr, value)
            except (AttributeError):
                raise AttributeError("{} cannot be set after cpp"
                                     " initialization".format(attr))
        self._param_dict[attr] = value

    def _setattr_typeparam(self, attr, value):
        try:
            for k, v in value.items():
                self._typeparam_dict[attr][k] = v
        except TypeError:
            raise ValueError("To set {}, you must use a dictionary "
                             "with types as keys.".format(attr))

    def detach(self):
        self._unapply_typeparam_dict()
        self._update_param_dict()
        self._cpp_obj = None
        if hasattr(self, '_simulation'):
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

    def attach(self, sim):
        raise NotImplementedError

    @property
    def is_attached(self):
        return self._cpp_obj is not None

    def _apply_param_dict(self):
        for attr, value in self._param_dict.items():
            try:
                setattr(self, attr, value)
            except AttributeError:
                pass

    def _apply_typeparam_dict(self, cpp_obj, sim):
        for typeparam in self._typeparam_dict.values():
            try:
                typeparam.attach(cpp_obj, sim)
            except ValueError as verr:
                raise ValueError("TypeParameter {}:"
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

    @Loggable.log(flag='dict')
    def state(self):
        self._update_param_dict()
        state = self._typeparam_states()
        state['params'] = deepcopy(self._param_dict)
        return dict_map(state, note_type)

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
            obj._typeparam_dict[name] = tp_dict
        return obj


class _TriggeredOperation(_Operation):
    _cpp_list_name = None

    def __init__(self, trigger):
        if isinstance(trigger, int):
            trigger = PeriodicTrigger(period=trigger, phase=0)
        self._trigger = trigger

    @property
    def trigger(self):
        return self._trigger

    @trigger.setter
    def trigger(self, new_trigger):
        if type(new_trigger) == int:
            new_trigger = PeriodicTrigger(period=new_trigger, phase=0)
        elif not isinstance(new_trigger, Trigger):
            raise ValueError("Trigger of type {} must be a subclass of "
                             "hoomd.triggers.Trigger".format(type(new_trigger))
                             )
        self._trigger = new_trigger
        if self.is_attached:
            sys = self._simulation._cpp_sys
            triggered_ops = getattr(sys, self._cpp_list_name)
            for index in range(len(triggered_ops)):
                if triggered_ops[index][0] == self._cpp_obj:
                    new_tuple = (self._cpp_obj, new_trigger)
                    triggered_ops[index] = new_tuple

    def attach(self, simulation):
        self._simulation = simulation
        sys = simulation._cpp_sys
        getattr(sys, self._cpp_list_name).append((self._cpp_obj, self.trigger))


class _Updater(_TriggeredOperation):
    _cpp_list_name = 'updaters'


class _Analyzer(_TriggeredOperation):
    _cpp_list_name = 'analyzers'


# \brief A Mixin to facilitate storage of simulation metadata
class _metadata(object):
    def __init__(self):
        # No metadata provided per default
        self.metadata_fields = []

    # \internal
    # \brief Return the metadata
    def get_metadata(self):
        data = OrderedDict()

        for m in self.metadata_fields:
            data[m] = getattr(self, m)

        return data

class _metadata_from_dict:
    def __init__(self, d):
        self.d = d;

    def get_metadata(self):
        data = OrderedDict()

        for m in self.d.keys():
            data[m] = self.d[m];

        return data

def dump_metadata(filename=None,user=None,indent=4):
    R""" Writes simulation metadata into a file.

    Args:
        filename (str): The name of the file to write JSON metadata to (optional)
        user (dict): Additional metadata.
        indent (int): The json indentation size

    Returns:
        metadata as a dictionary

    When called, this function will query all registered forces, updaters etc.
    and ask them to provide metadata. E.g. a pair potential will return
    information about parameters, the Logger will output the filename it is
    logging to, etc.

    Custom metadata can be provided as a dictionary to *user*.

    The output is aggregated into a dictionary and written to a
    JSON file, together with a timestamp. The file is overwritten if
    it exists.
    """

    if not hoomd.init.is_initialized():
        raise RuntimeError("Need to initialize system first.")

    metadata = dict()

    if user is not None:
        if not isinstance(user, Mapping):
            hoomd.context.current.device.cpp_msg.warning("Extra meta data needs to be a mapping type. Ignoring.\n")
        else:
            metadata['user'] = _metadata_from_dict(user);

    # Generate time stamp
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    metadata['timestamp'] = st
    metadata['device'] = hoomd.context.current.device
    metadata['hoomd'] = hoomd.context.HOOMDContext()

    global_objs = [hoomd.data.system_data(hoomd.context.current.system_definition)];
    global_objs += [hoomd.context.current.integrator];

    for o in global_objs:
        if o is not None:
            name = o.__module__+'.'+o.__class__.__name__;
            if len(name) > 13 and name[:13] == 'hoomd.':
                name = name[13:];
            metadata[name] = o

    global_objs = copy.copy(hoomd.context.current.forces);
    global_objs += hoomd.context.current.constraint_forces;
    global_objs += hoomd.context.current.integration_methods;
    global_objs += hoomd.context.current.forces
    global_objs += hoomd.context.current.analyzers;
    global_objs += hoomd.context.current.updaters;

    for o in global_objs:
        if o is not None:
            name = o.__module__+'.'+o.__class__.__name__;
            if len(name) > 13 and name[:13] == 'hoomd.':
                name = name[13:];
            metadata.setdefault(name, set())
            assert isinstance(metadata[name], set)
            metadata[name].add(o)

    # handler for unknown objects
    def default_handler(obj):
        if isinstance(obj, set) and len(obj) > 0:
            return list(filter(None, (default_handler(o) for o in obj)))
        try:
            return obj.get_metadata()
        except (AttributeError, NotImplementedError):
            return None

    # dump to JSON
    meta_str = json.dumps(
        metadata, default=default_handler,indent=indent, sort_keys=True)

    # only write files on rank 0
    if filename is not None and hoomd.context.current.device.comm.rank == 0:
        with open(filename, 'w') as file:
            file.write(meta_str)
    return json.loads(meta_str)
