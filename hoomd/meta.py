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

import hoomd;
import json, collections;
import time
import datetime
import copy

from collections import OrderedDict
from collections import Mapping

## \internal
class _Operation:

    _cpp_obj = None
    _param_dict = dict()
    _typeparam_dict = dict()

    def __getattr__(self, attr):
        if attr in self._param_dict.keys():
            return self._getattr_param(attr)
        elif attr in self._typeparam_dict.keys():
            return self._getattr_typeparam(attr)
        else:
            raise AttributeError("Object {} has no attribute {}"
                                 "".format(self, attr))

    def _getattr_param(self, attr):
        if self._cpp_obj is not None:
            return getattr(self._cpp_obj, attr)
        else:
            return self._param_dict[attr]

    def _getattr_typeparam(self, attr):
        return self._typeparam_dict[attr]


    def __setattr__(self, attr, value):
        if attr in self._param_dict.keys():
            self._setattr_param(attr, value)
        elif attr in self._typeparam_dict.keys():
            self._setattr_typeparam(attr, value)
        else:
            self.__dict__[attr] = value

    def _setattr_param(self, attr, value):
        if self._cpp_obj is not None:
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
        self._cpp_obj = None

    def attach(self):
        raise NotImplementedError

    @property
    def is_attached(self):
        return hasattr(self, '_cpp_obj')

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

    def _unapply_typeparam_dict(self):
        for typeparam in self._typeparam_dict.values():
            typeparam.detach()

    def _add_typeparam(self, typeparam):
        self._typeparam_dict[typeparam.name] = typeparam


# \brief A Mixin to facilitate storage of simulation metadata
class _metadata(object):
    def __init__(self):
        # No metadata provided per default
        self.metadata_fields = []

    ## \internal
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
        if not isinstance(user, collections.Mapping):
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
