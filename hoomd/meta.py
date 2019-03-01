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
import inspect

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
            metadata['arguments'] = argdata

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
            metadata['tracked_fields'] = tracked_fields
        return metadata

    @classmethod
    def from_metadata(cls, params):
        """This function creates an instance of cls according to a set of metadata parameters."""
        # for key, value in params['arguments']:
            # if key.startswith('**'):
                # value
                # and not
        obj = cls(**params['arguments'])
        if 'tracked_fields' in params:
            if not getattr(params['tracked_fields'], 'enabled', True):
                obj.disable()
            for field, value in params['tracked_fields'].items():
                setattr(obj, field, value)
        return obj


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

    def to_metadata(obj):
        """Convert object to metadata. At all but the top level, we return a
        mapping of object name->metadata."""
        metadata = obj.get_metadata()

        def convert(element):
            """Converts object to metadata representation if needed"""
            if hasattr(element, 'get_metadata'):
                # Refer to previously appended objects if already added
                if element in INSTANCES:
                    return "Object #{}".format(INSTANCES.index(element))
                else:
                    return to_metadata(element)
            else:
                return element

        for mapping in metadata.values():
            # The possible top_keys are 'arguments' and 'tracked_fields'. We'll
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

    # Loop over all class names and their instances
    metadata = {'objects': [to_metadata(o) for o in INSTANCES]}

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

def load_metadata(system, metadata=None, filename=None, output_script=None):
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
        generate_script (str): If provided, instead of creating the objects
                               based on the metadata, the function will
                               generate a script equivalent to what
                               load_metadata would accomplish.

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
    for entry in metadata['objects']:
        # There will always be just one element in the dict
        key, vals = next(iter(entry.items()))
        parts = key.split('.')[1:]  # Ignore `hoomd`
        obj = hoomd
        while parts:
            obj = getattr(obj, parts.pop(0))

        # Replace the Object entries with the actual objects
        for key, value in vals['arguments'].items():
            if isinstance(value, str) and 'Object #' in value:
                index = int(re.findall('Object #(\d*)', value)[0])
                vals['arguments'][key] = objects[index]

        instance = obj.from_metadata(vals)
        objects.append(instance)
    hoomd.util.unquiet_status()

    return objects
