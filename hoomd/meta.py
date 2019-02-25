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
import warnings

from collections import OrderedDict
from collections import Mapping

## \internal
# \brief A Mixin to facilitate storage of simulation metadata
class _metadata(object):
    R"""Track the metadata of all subclasses.

    The goal of metadata tracking is to make it possible to completely
    reproduce the exact simulation protocol in a script. In order to do so,
    every class that needs to be tracked (system data, neighbor lists,
    integrators, pair potentials, etc) should all inherit from _metadata. The
    class tracks information in three ways:

    #. All constructor arguments are automatically tracked.
    #. Any method decorated with the @track decorator will be recorded whenever called.
    #. Every class may define a class level variable `metadata_fields` that indicates class variables (maybe, haven't decided yet).

    The constructor tracking is accomplished by overriding the __new__ method
    to automatically store all constructor arguments. Similarly, all methods
    decorated with the @track decorator will automatically log their inputs.
    """

    def __new__(cls, *args, **kwargs):
        obj = super(_metadata, cls).__new__(cls)
        obj.args = args
        obj.kwargs = kwargs
        return obj

    def __init__(self):
        # No metadata provided per default
        self.metadata_fields = []

    ## \internal
    # \brief Return the metadata
    def get_metadata(self):
        varnames = self.__init__.__code__.co_varnames[1:]  # Skip `self`

        # Fill in positional arguments first, then update all kwargs. No need
        # for extensive error checking since the function signature must have
        # been valid to construct the object in the first place.
        metadata = OrderedDict()
        for varname, arg in zip(varnames, self.args):
            metadata[varname] = arg
        metadata.update(self.kwargs)
        return metadata

class _metadata_from_dict(object):
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
    hoomd.util.print_status_line();

    if not hoomd.init.is_initialized():
        hoomd.context.msg.error("Need to initialize system first.\n")
        raise RuntimeError("Error writing out metadata.")

    metadata = dict()

    if user is not None:
        if not isinstance(user, collections.Mapping):
            hoomd.context.msg.warning("Extra meta data needs to be a mapping type. Ignoring.\n")
        else:
            metadata['user'] = _metadata_from_dict(user);

    # Generate time stamp
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    metadata['timestamp'] = st
    metadata['context'] = hoomd.context.ExecutionContext()
    metadata['hoomd'] = hoomd.context.HOOMDContext()

    # global_objs = [hoomd.data.system_data(hoomd.context.current.system_definition)];
    # global_objs += [hoomd.context.current.integrator];
    global_objs = [hoomd.context.current.integrator];

    for o in global_objs:
        if o is not None:
            name = o.__module__+'.'+o.__class__.__name__;
            if len(name) > 13 and name[:13] == 'hoomd.':
                name = name[13:];
            metadata[name] = o

    global_objs = copy.copy(hoomd.context.current.integration_methods)
    global_objs += hoomd.context.current.forces
    global_objs += hoomd.context.current.analyzers
    global_objs += hoomd.context.current.updaters
    global_objs += hoomd.context.current.constraint_forces

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
        print("Handling object ", obj)
        if isinstance(obj, set) and len(obj) > 0:
            return list(filter(None, (default_handler(o) for o in obj)))
        try:
            return obj.get_metadata()
        except (AttributeError, NotImplementedError):
            hoomd.context.msg.error(
                "The {} class does not provide metadata dumping\n".format(obj))
            raise

    # dump to JSON
    meta_str = json.dumps(
        metadata, default=default_handler,indent=indent, sort_keys=True)

    # only write files on rank 0
    if filename is not None and hoomd.comm.get_rank() == 0:
        with open(filename, 'w') as file:
            file.write(meta_str)
    return json.loads(meta_str)

def load_metadata(system, metadata=None, filename=None):
    R"""Initialize system information from metadata.

    This function must be called after the system is initialized, but before
    any other HOOMD functions are called. This function will update the system
    data with any bonds, constraints, etc that are encoded in the metadata.
    Additionally, it will instantiate any pair potentials, forces, etc that
    need to be created. All of the created objects will be returned to the
    user, who can modify them or create new objects as necessary.

    Args:
        system (:py:class:`hoomd.data.system_data`): The initial system.
        metadata (dict): The metadata to initialize with. Defaults to None, but
                         must be provided unless a filename is given.
        filename (str): A file containing metadata. Is ignored if a metadata
                        dictionary is provided.

    Returns:
        dict: A mapping from class to the instance created by this function.
    """
    # For now, only use filename if no metadata is given.
    if filename is not None:
        if metadata is None:
            with open(filename) as f:
                metadata = json.load(f)
        else:
            warnings.warn(
                "Both filename and data specified. Ignoring provided file.")
    elif metadata is None:
        raise RuntimeError(
            "You must provide either a dictionary with metadata or a file to "
            "read from.")

    # Ignored keys are those we don't need to do anything with
    ignored_keys = ['timestamp', 'context', 'hoomd']
    objects = {}
    for key, vals in metadata.items():
        if key in ignored_keys:
            continue

        # Top level is always hoomd, but may be removed
        parts = key.split('.')
        if parts[0] == 'hoomd':
            parts.pop(0)
        obj = hoomd
        while parts:
            obj = getattr(obj, parts.pop(0))

        if obj.__name__ == "system_data":
            # System data is the special case that needs the actual system
            # object to be passed through as well.
            instance = obj.from_metadata(vals, system)
        else:
            instance = obj.from_metadata(vals)

        name = obj.__module__ + '.' + obj.__class__.__name__
        objects[name] = instance
    return objects
