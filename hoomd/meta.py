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
import re
import traceback
import inspect

# Registry of all modules used in the script.
MODULES = []

# Top-level keys defining the metadata schema.
META_KEY_TIMESTAMP = 'timestamp'
META_KEY_EXECINFO = 'execution_info'
META_KEY_HOOMDINFO = 'hoomd_info'
META_KEY_OBJECTS = 'objects'
META_KEY_MODULES = 'modules'
META_KEY_USER = 'user'

# Second-level keys for list of objects.
META_KEY_ARGS = 'arguments'
META_KEY_TRACKED = 'tracked_fields'

# Simple lamba to generate a fully qualified class name.
to_name = lambda obj: obj.__module__ + '.' + obj.__class__.__name__

def cls_from_name(name):
    """Gets the class object named by a fully qualified name, e.g. calling
    with 'hoomd.dump.gsd' results in the gsd class (which can then be used
    to create objects, etc.)."""
    parts = name.split('.')[1:]  # Ignore `hoomd`
    cls = hoomd
    while parts:
        cls = getattr(cls, parts.pop(0))
    return cls

def should_track():
    R"""Determine whether or not the last object created should be tracked for metadata.

    This function is designed to be called within the initializer (__init__)
    for a class. Since only objects created by the user should be tracked (not
    objects HOOMD creates internally), we need to check where the constructor
    call is coming from. The check is implemented by inspecting the third to
    last element of the call stack (-1 will be the traceback.extract_stack
    call, -2 will be this function's call, so -3 is the __init__ call), then
    check what file that's coming from.

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
            hoomd.context.current.meta_objects.append(obj)
        return obj

    ## \internal
    # \brief Return the metadata
    # Metadata is kept as brief as possible, so all fields are omitted if they
    # are empty or if their value is implied (e.g. enabled=True for all classes
    # is the default).
    def get_metadata(self):
        metadata = {}
        try:
            # This can be removed when we remove Python2 support.
            signature = inspect.signature(self.__init__)
        except AttributeError:
            hoomd.context.msg.warning("Object-level metadata dumping is only available in Python 3.")
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
            if argdata:
                metadata[META_KEY_ARGS] = argdata

        # The metadata_fields defaults to being empty, so this will do nothing
        # in the default case.
        tracked_fields = {}
        for field in self.metadata_fields:
            # Most objects in HOOMD can be disabled, but they are enabled by
            # default, so we only save this key when they are disabled.
            if field == 'enabled' and getattr(self, field):
                continue
            tracked_fields[field] = getattr(self, field)
        if tracked_fields:
            metadata[META_KEY_TRACKED] = tracked_fields
        return metadata

    ## \internal
    # \brief Create object from metadata.
    @classmethod
    def from_metadata(cls, params, args=[]):
        obj = cls(*args, **params.get(META_KEY_ARGS, {}))

        if META_KEY_TRACKED in params:
            # Disable if needed before doing anything else.
            if not params[META_KEY_TRACKED].get('enabled', True):
                obj.disable()
            for field, value in params[META_KEY_TRACKED].items():
                # Need to differentiate between attributes that are themselves
                # HOOMD classes and others.
                if isinstance(value, dict) and isinstance(next(iter(value.keys())), str) and next(iter(value.keys())).startswith('hoomd.'):
                    classname, class_params = next(iter(value.items()))
                    setattr(obj, field, cls_from_name(classname)(**class_params[META_KEY_ARGS]))
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
            'timestamp', 'objects', and 'modules'. Note that dumping out
            objects is only supported in Python 3.

    Returns:
        dict: The metadata.

    When called, this function will query all registered forces, updaters etc.
    and ask them to provide metadata. For example, a pair potential will return
    information about parameters, a logger will output the filename it is
    logging to, etc. The output is aggregated into a dictionary and returned.
    If the filename parameters is provided, the metadata is also written out to
    the file. The file is overwritten if it exists. Custom metadata can be
    provided as a dictionary to *user*.

    The metadata dump is designed to also support restartability using the
    :func:`~.load_metadata` command, which accepts a metadata dictionary or a
    filename, and uses the metadata to build up a simulation consisting of the
    objects specified in the 'objects' key of the dumped metadata. While this
    methodology supports a wide array of simulations, the following are
    currently NOT supported:

    * If multiple SimulationContext objects are created, metadata will be
      stored to each context. However, metadata for each context must be stored
      and reloaded separately. No information about the sequence of context
      creation is maintained.
    * Calls to :func:`hoomd.run` are not tracked; :func:`hoomd.run` (or
      :func:`hoomd.run_upto`) must be called after metadata is loaded.
    * State changes to objects are only supported to the extent necessary for
      simulation runs to succeed. This means that any object that requires
      *e.g.* calls to `set_params` should log enough information to reproduce
      the exact state of the object when :func:`~.dump_metadata` is called;
      however, any intermediate changes in the state of the object, for
      instance the changing of parameter values in between calls to
      :func:`hoomd.run`, will not be tracked or logged.
    * No callables will be serialized. For instance, any functions used to log
      custom quantities in :class:`hoomd.analyze.log` will not be included in
      the metadata dump.
    """
    hoomd.util.print_status_line()

    if not hoomd.init.is_initialized():
        hoomd.context.msg.error("Need to initialize system first.\n")
        raise RuntimeError("Error writing out metadata.")

    def to_metadata(obj):
        """Converts the object to its metadata representation. This involves
        potentially making recursive calls to generate metadata representations
        for nested objects. The recursion is handled using the helper internal
        `convert` function."""
        metadata = obj.get_metadata()

        def convert(obj):
            """Contains the recursive call to `to_metadata` for when metadata
            dict values are actually other HOOMD objects. When this is the
            case, this function checks to make sure that the object hasn't
            already been created; if it has, it generates a reference to the
            old object. If the dict values are simply values, they are returned
            as is."""
            if hasattr(obj, 'get_metadata'):
                # Refer to previously appended objects if already added
                if obj in hoomd.context.current.meta_objects:
                    return "Object #{}".format(hoomd.context.current.meta_objects.index(obj))
                else:
                    return to_metadata(obj)
            else:
                return obj

        # At the top level of the metadata, the values correspond to keys
        # META_KEY_ARGS and META_KEY_TRACKED. We loop over each set, and modify
        # them in place.
        for mapping in metadata.values():
            for arg_or_field, value in mapping.items():
                # Check for *args/**kwargs
                if arg_or_field[:2] == '**':
                    mapping[arg_or_field] = {k: convert(v) for (k, v) in value.items()}
                elif arg_or_field[0] == '*':
                    mapping[arg_or_field] = [convert(e) for e in value]
                else:
                    mapping[arg_or_field] = convert(value)
        return {to_name(obj): metadata}

    metadata = {}

    # Add additional configuration info, including user provided quantities.
    if META_KEY_MODULES in fields:
        metadata[META_KEY_MODULES] = MODULES
    if META_KEY_OBJECTS in fields:
        metadata[META_KEY_OBJECTS] = [to_metadata(o) for o in hoomd.context.current.meta_objects]
    if META_KEY_TIMESTAMP in fields:
        metadata[META_KEY_TIMESTAMP] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if META_KEY_EXECINFO in fields:
        metadata[META_KEY_EXECINFO] = hoomd.context.ExecutionContext().get_metadata()
    if META_KEY_HOOMDINFO in fields:
        metadata[META_KEY_HOOMDINFO] = hoomd.context.HOOMDContext().get_metadata()

    if user is not None:
        if not isinstance(user, collections.Mapping):
            hoomd.context.msg.warning("Extra meta data needs to be a mapping type. Ignoring.\n")
        else:
            metadata[META_KEY_USER] = user

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
            hoomd.context.msg.warning(
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

    def replace_object(potential_obj):
        """Check if the argument is a placeholder for a previously
        created object. If so, replace it with the actual object."""
        if isinstance(potential_obj, str) and 'Object #' in potential_obj:
            index = int(re.findall('Object #(\d*)', potential_obj)[0])
            return objects[index]
        else:
            return potential_obj

    if META_KEY_OBJECTS not in metadata:
        hoomd.context.msg.error("Provided metadata object does not contain "
                                 "object information to reload.")

    hoomd.util.quiet_status()
    for entry in metadata[META_KEY_OBJECTS]:
        # There will always be just one element in each dictionary. The key is
        # the class of the object, and the values are the argument and tracked
        # field information.
        classname, params = next(iter(entry.items()))

        # Any positional arguments to pass to the class constructor.
        args = []

        if META_KEY_ARGS in params:
            # Unpack *args/**kwargs before proceeding. We can't modify the
            # dictionary during iteration, so we have to store values to change.
            updated_params = {}
            to_remove = []
            for argname, argvalue in params[META_KEY_ARGS].items():
                if argname.startswith('**'):
                    to_remove.append(argname)
                    for k, v in argvalue.items():
                        updated_params[k] = v
                elif argname.startswith('*'):
                    to_remove.append(argname)
                    args = argvalue

            params[META_KEY_ARGS].update(updated_params)
            for arg in to_remove:
                params[META_KEY_ARGS].pop(arg)

            # Replace the Object entries with the actual objects
            for argname, argvalue in params[META_KEY_ARGS].items():
                params[META_KEY_ARGS][argname] = replace_object(argvalue)

            for i, arg in enumerate(args):
                args[i] = replace_object(arg)

        if META_KEY_TRACKED in params:
            for field, value in params[META_KEY_TRACKED].items():
                params[META_KEY_TRACKED][field] = replace_object(value)

        cls = cls_from_name(classname)
        objects.append(cls.from_metadata(params, args))
    hoomd.util.unquiet_status()

    return objects
