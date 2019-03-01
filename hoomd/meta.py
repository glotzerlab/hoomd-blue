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
import json
import collections
import time
import datetime
import copy
import warnings
import re
import traceback

def track(func):
    """Decorator for any method whose calls must be tracked in order to
    completely reproduce a simulation. Assumes input is a class method.
    """
    ###TODO: MAKE SURE THAT FORCE CLASSES UPDATE COEFFS BEFORE GETTING METADATA
    if not hasattr(func, call_history):
        func.call_history = []
    def tracked_func(self, *args, **kwargs):
        func.call_history.append({
            'args': args,
            'kwargs': kwargs
            })
        return func(self, *args, **kwargs)
    return tracked_func

# Registry of all metadata tracking classes created.
INSTANCES = []

def should_track():
    R"""We only want to track objects created by the user, so we need to
    check where the constructor call is coming from. To do that, get the
    third to last element of the call stack (-1 will be the
    traceback.extract_stack call, -2 will be this call), then check what file
    that's coming from.

    Returns:
        bool: Whether or not the last constructed object should be tracked.
    """
    stack = traceback.extract_stack().format()
    last_file = re.findall('File "(.*?)"', stack[-3])[0]
    return hoomd.__file__.replace('__init__.py', '') not in last_file

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
        obj.metadata_fields = []

        if should_track():
            INSTANCES.append(obj)
        return obj

    ## \internal
    # \brief Return the metadata
    def get_metadata(self):
        varnames = self.__init__.__code__.co_varnames[1:]  # Skip `self`

        # Fill in positional arguments first, then update all kwargs. No need
        # for extensive error checking since the function signature must have
        # been valid to construct the object in the first place.
        argdata = {}
        for varname, arg in zip(varnames, self.args):
            argdata[varname] = arg
        argdata.update(self.kwargs)

        # Add additional fields only if requested.
        tracked_fields = {}
        if self.metadata_fields:
            for field in self.metadata_fields:
                tracked_fields[field] = getattr(self, field)
        metadata = {'arguments': argdata, 'tracked_fields': tracked_fields}
        return metadata

# TODO: Maybe include a subclass that calls set_params on a set of special parameters. More generally, that allows any set* functions to be tracked... now I'm going back to what I was doing before.

def dump_metadata(filename=None, user=None, indent=4, execution_info=False, hoomd_info=False):
    R""" Writes simulation metadata into a file.

    Args:
        filename (str): The name of the file to write JSON metadata to (optional)
        user (dict): Additional metadata.
        indent (int): The json indentation size.
        execution_info (bool): If True, include information on execution environment.
        hoomd_info (bool): If True, include information on the HOOMD executable.

    Returns:
        dict: The metadata

    When called, this function will query all registered forces, updaters etc.
    and ask them to provide metadata. E.g. a pair potential will return
    information about parameters, the Logger will output the filename it is
    logging to, etc.

    Custom metadata can be provided as a dictionary to *user*.

    The output is aggregated into a dictionary and written to a
    JSON file, together with a timestamp. The file is overwritten if
    it exists.
    """
    hoomd.util.print_status_line()

    if not hoomd.init.is_initialized():
        hoomd.context.msg.error("Need to initialize system first.\n")
        raise RuntimeError("Error writing out metadata.")

    metadata = dict()

    to_name = lambda obj: obj.__module__ + '.' + obj.__class__.__name__

    # First put all classes into a set to avoid saving duplicates.
    for o in INSTANCES:
        if o is not None:
            name = to_name(o)
            metadata.setdefault(name, set())
            assert isinstance(metadata[name], set)
            metadata[name].add(o)

    def to_metadata(metadata):
        """Convert object to metadata. At all but the top level, we return a
        mapping of object name->metadata."""
        meta = {}
        for k, v in metadata.items():
            if hasattr(v, 'get_metadata'):
                m = v.get_metadata()
                argdata = to_metadata(m['arguments'])
                tracked_fields = to_metadata(m['tracked_fields'])
                obj_data = {}
                if argdata:
                    obj_data['arguments'] = argdata
                if tracked_fields:
                    obj_data['tracked_fields'] = tracked_fields
                meta[k] = {to_name(v): obj_data}
            else:
                meta[k] = v
        return meta

    # Loop over all class names and their instances
    for cls, instances in metadata.items():
        instances_metadata = []
        for obj in instances:
            # This logic is duplicated here because inside to_metadata we
            # append a dictionary with the object as the name, but at the top
            # level that object already exists. We should probably change that
            # structure by modifying the existing construction, but for now I'd
            # like to work within the existing framework.
            obj_metadata = obj.get_metadata()
            argdata = to_metadata(obj_metadata['arguments'])
            tracked_fields = to_metadata(obj_metadata['tracked_fields'])
            obj_data = {}
            if argdata:
                obj_data['arguments'] = argdata
            if tracked_fields:
                obj_data['tracked_fields'] = tracked_fields
            instances_metadata.append(obj_data)

        metadata[cls] = instances_metadata


    # Add additional configuration info, including user provided quantities.
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    metadata['timestamp'] = st
    if execution_info:
        metadata['execution_info'] = hoomd.context.ExecutionContext().get_metadata()
    if hoomd_info:
        metadata['hoomd_info'] = hoomd.context.HOOMDContext().get_metadata()

    if user is not None:
        if not isinstance(user, collections.Mapping):
            hoomd.context.msg.warning("Extra meta data needs to be a mapping type. Ignoring.\n")
        else:
            # Make sure the object actually is a dict, not some other mapping object.
            metadata['user'] = dict(user)

    # Only write files on rank 0
    if filename is not None and hoomd.comm.get_rank() == 0:
        with open(filename, 'w') as file:
            meta_str = json.dumps(
                metadata, indent=indent, sort_keys=True)
            file.write(meta_str)
    return metadata

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
