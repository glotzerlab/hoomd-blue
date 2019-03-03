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
import datetime
import copy
import warnings
import re
import traceback
import inspect

# Registry of all metadata tracking classes created.
INSTANCES = []

# Registry of all modules used in the script.
MODULES = set()

# Top-level keys defining the metadata schema.
META_KEY_TIMESTAMP = 'timestamp'
META_KEY_EXECINFO = 'execution_info'
META_KEY_HOOMDINFO = 'hoomd_info'
META_KEY_OBJECTS = 'objects'
META_KEY_MODULES = 'modules'

# Second-level keys for list of objects.
META_KEY_ARGS = 'arguments'
META_KEY_TRACKED = 'tracked_fields'

to_name = lambda obj: obj.__module__ + '.' + obj.__class__.__name__

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
    #. Every class may, in its constructor, append to the attribute
       `metadata_fields` to indicate class attributes that should be saved to
       metadata.

    While this supports a wide array of behaviors, the following are currently NOT supported:

    #. The creation of multiple SimulationContext objects will not be detected,
       and context switching will not occur.
    #. Calls to `hoomd.run` are not tracked; `run` (or `run_upto`) must be
       called after metadata is loaded.
    #. Objects that support modification after construction. Such objects will
       be reconstructed based on the original constructor, and if any
       `metadata_fields` are specified, the logged values will be those defined
       at the time `get_metadata` is called on the object. Note that this
       logging still supports classes that require the setting of parameters
       after construction (*e.g.* md.pair.* or like md.rigid.constrain), but it
       will not support maintaining state information that involves changing
       these parameters, *e.g.* between multiple `hoomd.run` calls.
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
        metadata = {}
        try:
            # This can be removed when we remove Python2 support.
            signature = inspect.signature(self.__init__)
        except AttributeError:
            warnings.warn("Restartable metadata dumping is only available in Python 3.")
        else:
            bound_args = signature.bind(*self.args, **self.kwargs)
            argdata = {}
            for varname, arg in bound_args.arguments.items():
                kind = signature.parameters[varname].kind
                if kind == inspect.Parameter.VAR_POSITIONAL:
                    varname = "*" + varname
                elif kind == inspect.Parameter.VAR_KEYWORD:
                    varname = "**" + varname
                argdata[varname] = arg
            metadata[META_KEY_ARGS] = argdata

        # Add additional fields only if requested.
        if self.metadata_fields:
            tracked_fields = {}
            for field in self.metadata_fields:
                # Most objects in HOOMD can be disabled, so we treat this
                # special key explicitly here and only save it if the object
                # was disabled.
                if field == 'enabled' and getattr(self, field):
                    continue

                tracked_fields[field] = getattr(self, field)
            metadata[META_KEY_TRACKED] = tracked_fields
        return metadata

    @classmethod
    def from_metadata(cls, params, args=[]):
        """This function creates an instance of cls according to a set of
        metadata parameters. Accepts optional positional arguments to support
        logging of *args."""
        obj = cls(*args, **params[META_KEY_ARGS])

        if META_KEY_TRACKED in params:
            if not params[META_KEY_TRACKED].get('enabled', True):
                obj.disable()
            for field, value in params[META_KEY_TRACKED].items():
                # Need to differentiate between attributes that are themselves
                # HOOMD classes and others.
                if isinstance(value, dict) and isinstance(next(iter(value.keys())), str) and next(iter(value.keys())).startswith('hoomd.'):
                    k, v = next(iter(value.items()))
                    parts = k.split('.')[1:]  # Ignore `hoomd`
                    nested_obj = hoomd
                    while parts:
                        nested_obj = getattr(nested_obj, parts.pop(0))
                    setattr(obj, field, nested_obj(**v[META_KEY_ARGS]))

                else:
                    setattr(obj, field, value)
        return obj

def dump_metadata(filename=None, user=None, indent=4, fields=['timestamp', 'modules', 'objects']):
    R""" Writes simulation metadata into a file.

    Args:
        filename (str): The name of the file to write JSON metadata to (optional)
        user (dict): Additional metadata.
        indent (int): The json indentation size.
        fields (list):
            A list of information to include. Valid keys are 'timestamp',
            'objects', 'modules', 'execution_info', 'hoomd_info'. Defaults to
            'timestamp', 'objects', and 'modules'.

    Returns:
        dict: The metadata

    When called, this function will query all registered forces, updaters etc.
    and ask them to provide metadata. For example, a pair potential will return
    information about parameters, a logger will output the filename it is
    logging to, etc. The output is aggregated into a dictionary and returned.
    If the filename parameters is provided, the metadata is also written out to
    the file. The file is overwritten if it exists. Custom metadata can be
    provided as a dictionary to *user*.

    The metadata dump is designed to also support restartability using the
    `load_metadata` command, which accepts a metadata dictionary or a filename,
    and uses the metadata to build up a simulation consisting of the objects
    specified in the 'objects' key of the dumped metadata. While this
    methodology supports a wide array of simulations, the following are
    currently NOT supported:

    #. The creation of multiple SimulationContext objects will not be detected,
       and context switching will not occur.
    #. Calls to `hoomd.run` are not tracked; `run` (or `run_upto`) must be
       called after metadata is loaded.
    #. Objects that support modification after construction. Such objects will
       be reconstructed based on the original constructor, and if any
       `metadata_fields` are specified, the logged values will be those defined
       at the time `get_metadata` is called on the object. Note that this
       logging still supports classes that require the setting of parameters
       after construction (*e.g.* md.pair.* or like md.rigid.constrain), but it
       will not support maintaining state information that involves changing
       these parameters, *e.g.* between multiple `hoomd.run` calls.
    """
    hoomd.util.print_status_line()

    if not hoomd.init.is_initialized():
        hoomd.context.msg.error("Need to initialize system first.\n")
        raise RuntimeError("Error writing out metadata.")

    def to_metadata(obj):
        """Convert object to metadata. At all but the top level, we return a
        mapping of object name->metadata."""
        metadata = obj.get_metadata()

        def convert(element):
            """Contains the recursive call to `to_metadata` for when metadata
            dict values are actually other HOOMD objects. When this is the
            case, this function checks to make sure that the object hasn't
            already been created; if it has, it generates a reference to the
            old object. If the dict values are simply values, they are returned
            as is."""
            if hasattr(element, 'get_metadata'):
                # Refer to previously appended objects if already added
                if element in INSTANCES:
                    return "Object #{}".format(INSTANCES.index(element))
                else:
                    return to_metadata(element)
            else:
                return element

        for mapping in metadata.values():
            # The possible top_keys are META_KEY_ARGS and META_KEY_TRACKED. We'll
            # modify the values in place.
            for k, v in mapping.items():
                # Check for *args/**kwargs
                if k[:2] == '**':
                    mapping[k] = {k2: convert(v2) for k2, v2 in v.items()}
                elif k[0] == '*':
                    mapping[k] = [convert(e) for e in v]
                else:
                    mapping[k] = convert(v)
        return {to_name(obj): metadata}

    # Add all instances to the metadata list of objects in series.
    metadata = {META_KEY_OBJECTS: [to_metadata(o) for o in INSTANCES]}

    # Add all modules that were loaded.
    metadata = {META_KEY_MODULES: MODULES}

    # Add additional configuration info, including user provided quantities.
    st = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if META_KEY_TIMESTAMP in fields:
        metadata[META_KEY_TIMESTAMP] = st
    if META_KEY_EXECINFO in fields:
        metadata[META_KEY_EXECINFO] = hoomd.context.ExecutionContext().get_metadata()
    if META_KEY_HOOMDINFO in fields:
        metadata[META_KEY_HOOMDINFO] = hoomd.context.HOOMDContext().get_metadata()
    if META_KEY_HOOMDINFO in fields:
        metadata[META_KEY_HOOMDINFO] = hoomd.context.HOOMDContext().get_metadata()

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
    user, who can modify them or create new objects as necessary. See the
    documentation of the `dump_metadata` function for information on the
    limitations of this approach for running simulations.

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

    # Explicitly import all subpackages so getattr works
    packages = ['hoomd.hpmc', 'hoomd.md', 'hoomd.mpcd', 'hoomd.dem',
                'hoomd.cgcmm', 'hoomd.jit', 'hoomd.metal']
    for package in packages:
        try:
            exec('import {}'.format(package))
        except (ImportError, ModuleNotFoundError):
            # Not all packages need to be installed
            pass

    # All object data is stored in the `objects` key
    objects = []
    hoomd.util.quiet_status()

    for entry in metadata[META_KEY_OBJECTS]:
        # There will always be just one element in the dict
        key, vals = next(iter(entry.items()))
        parts = key.split('.')[1:]  # Ignore `hoomd`
        obj = hoomd
        while parts:
            obj = getattr(obj, parts.pop(0))

        # Unpack *args/**kwargs before proceeding. We can't modify the
        # dictionary during iteration, so we have to store values to change.
        # The args default to a list that can be overridden; similarly, kwargs
        # default to a dict.
        args = []
        updated_params = {}
        to_remove = []
        for key, value in vals[META_KEY_ARGS].items():
            if key.startswith('**'):
                to_remove.append(key)
                for k, v in value.items():
                    updated_params[k] = v
            elif key.startswith('*'):
                to_remove.append(key)
                args = value

        vals[META_KEY_ARGS].update(updated_params)
        for key in to_remove:
            vals[META_KEY_ARGS].pop(key)

        # Replace the Object entries with the actual objects
        for key, value in vals[META_KEY_ARGS].items():
            if isinstance(value, str) and 'Object #' in value:
                index = int(re.findall('Object #(\d*)', value)[0])
                vals[META_KEY_ARGS][key] = objects[index]

        for i, arg in enumerate(args):
            if isinstance(arg, str) and 'Object #' in arg:
                index = int(re.findall('Object #(\d*)', arg)[0])
                args[i] = objects[index]

        if META_KEY_TRACKED in vals:
            for key, value in vals[META_KEY_TRACKED].items():
                if isinstance(value, str) and 'Object #' in value:
                    index = int(re.findall('Object #(\d*)', value)[0])
                    vals[META_KEY_TRACKED][key] = objects[index]

        instance = obj.from_metadata(vals, args)
        objects.append(instance)
    hoomd.util.unquiet_status()

    return objects
