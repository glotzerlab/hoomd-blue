# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Write GSD files storing simulation trajectories and logging data.

.. invisible-code-block: python

    simulation = hoomd.util.make_example_simulation()
    gsd_filename = tmp_path / 'trajectory.gsd'
"""

from collections.abc import Mapping, Collection
from hoomd.trigger import Periodic
from hoomd import _hoomd
from hoomd.util import _dict_flatten
from hoomd.data.typeconverter import OnlyFrom, RequiredArg
from hoomd.filter import ParticleFilter, All
from hoomd.data.parameterdicts import ParameterDict
from hoomd.logging import Logger, LoggerCategories
from hoomd.operation import Writer
import numpy as np
import json
import atexit
import weakref

# Track open gsd writers to flush at exit.
_open_gsd_writers = []


def _flush_open_gsd_writers():
    """Flush all open gsd writers at exit."""
    for weak_writer in _open_gsd_writers:
        writer = weak_writer()
        if writer is not None:
            writer.flush()


atexit.register(_flush_open_gsd_writers)


def _array_to_strings(value):
    if isinstance(value, np.ndarray):
        string_list = []
        for string in value:
            string_list.append(
                string.view(
                    dtype='|S{}'.format(value.shape[1])).decode('UTF-8'))
        return string_list
    else:
        return value


def _finalize_gsd(weak_writer, cpp_obj):
    """Finalize a GSD writer."""
    _open_gsd_writers.remove(weak_writer)
    cpp_obj.flush()


class GSD(Writer):
    r"""Write simulation trajectories in the GSD format.

    Args:
        trigger (hoomd.trigger.trigger_like): Select the timesteps to write.
        filename (any type that converts to `str`): File name to write.
        filter (hoomd.filter.filter_like): Select the particles to write.
            Defaults to `hoomd.filter.All`.
        mode (str): The file open mode. Defaults to ``'ab'``.
        truncate (bool): When `True`, truncate the file and write a new frame 0
            each time this operation triggers. Defaults to `False`.
        dynamic (list[str]): Field names and/or field categores to save in
            all frames. Defaults to ``['property']``.
        logger (hoomd.logging.Logger): Provide log quantities to write. Defaults
            to `None`.

    `GSD` writes the simulation trajectory to the specified file in the GSD
    format. `GSD` can store all particle, bond, angle, dihedral, improper,
    pair, and constraint data fields in every frame of the trajectory.  `GSD`
    can write trajectories where the number of particles, number of particle
    types, particle types, diameter, mass, charge, or other quantities change
    over time. `GSD` can also store arbitrary, scalar, string, and array
    quantities provided by a `hoomd.logging.Logger` instance in `logger`.

    Valid file open modes:

    +------------------+---------------------------------------------+
    | mode             | description                                 |
    +==================+=============================================+
    | ``'wb'``         | Open the file for writing. Create the file  |
    |                  | if needed, or overwrite an existing file.   |
    +------------------+---------------------------------------------+
    | ``'xb'``         | Create a GSD file exclusively.              |
    |                  | Raise an exception when the file exists.    |
    +------------------+---------------------------------------------+
    | ``'ab'``         | Create the file if needed, or append to an  |
    |                  | existing file.                              |
    +------------------+---------------------------------------------+

    To reduce file size, `GSD` does not write properties that are set to
    defaults. When masses, orientations, angular momenta, etc... are left
    default for all particles, these fields will not take up any space in the
    file, except on frame 1+ when the field is also non-default in frame 0.
    `GSD` writes all non-default fields to frame 0 in the file.

    To further reduce file sizes, `GSD` allows the user to select which specific
    fields will be considered for writing to frame 1+ in the `dynamic` list.
    When reading a GSD file, the data in frame 0 is read when a quantity is
    missing in frame *i*, so any fields not in `dynamic` are fixed for the
    entire trajectory.

    The `dynamic` list can contain one or more of the following strings:

    * ``'property'``

      * ``'configuration/box'``
      * ``'particles/N'``
      * ``'particles/position'``
      * ``'particles/orientation'``

    * ``'momentum'``

      * ``'particles/velocity'``
      * ``'particles/angmom'``
      * ``'particles/image'``

    * ``'attribute'``

      * ``'particles/types'``
      * ``'particles/typeid'``
      * ``'particles/mass'``
      * ``'particles/charge'``
      * ``'particles/diameter'``
      * ``'particles/body'``
      * ``'particles/moment_inertia'``

    * ``'topology'``

      * bonds/*
      * angles/*
      * dihedrals/*
      * impropers/*
      * constraints/*
      * pairs/*

    When you set a category string (``'property'``, ``'momentum'``,
    ``'attribute'``), `GSD` makes all the category member's fields dynamic.

    Warning:
        `GSD` buffers writes in memory. Abnormal exits (e.g. ``kill``,
        ``scancel``, reaching walltime limits) may cause loss of data. Ensure
        that your scripts exit cleanly and call `flush()` as needed to write
        buffered frames to the file.

    See Also:
        See the `GSD documentation <https://gsd.readthedocs.io/>`__, `GSD HOOMD
        Schema <https://gsd.readthedocs.io/en/stable/schema-hoomd.html>`__, and
        `GSD GitHub project <https://github.com/glotzerlab/gsd>`__ for more
        information on GSD files.

    Note:
        When you use ``filter`` to select a subset of the whole system, `GSD`
        writes only the selected particles in ascending tag order and does
        **not** write out **topology**.

    Tip:
        All logged data fields must be present in the first frame in the gsd
        file to provide the default value. To achieve this, set the `logger`
        attribute before the operation is triggered to write the first frame
        in the file.

        Some (or all) fields may be omitted on later frames. You can set
        `logger` to `None` or remove specific quantities from the logger, but do
        not add additional quantities after the first frame.

    .. rubric:: Example:

    .. code-block:: python

        gsd = hoomd.write.GSD(trigger=hoomd.trigger.Periodic(1_000_000),
                              filename=gsd_filename)
        simulation.operations.writers.append(gsd)

    Attributes:
        filename (str): File name to write (*read-only*).

            .. rubric:: Example:

            .. code-block:: python

                filename = gsd.filename

        filter (hoomd.filter.filter_like): Select the particles to write
            (*read-only*).

            .. rubric:: Example:

            .. code-block:: python

                filter_ = gsd.filter

        mode (str): The file open mode (*read-only*).

            .. rubric:: Example:

            .. code-block:: python

                mode = gsd.mode

        truncate (bool): When `True`, truncate the file and write a new frame 0
            each time this operation triggers (*read-only*).

            .. rubric:: Example:

            .. code-block:: python

                truncate = gsd.truncate

        dynamic (list[str]): Field names and/or field categores to save in
            all frames.

            .. rubric:: Examples:

            .. code-block:: python

                gsd.dynamic = ['property']

            .. code-block:: python

                gsd.dynamic = ['property', 'momentum']

            .. code-block:: python

                gsd.dynamic = ['property',
                               'particles/image',
                               'particles/typeid']

        write_diameter (bool): When `False`, do not write
            ``particles/diameter``. Set to `True` to write non-default particle
            diameters.

            .. rubric:: Example:

            .. code-block:: python

                gsd.write_diameter = True

        maximum_write_buffer_size (int): Size (in bytes) to buffer in memory
           before writing to the file.

            .. rubric:: Example:

            .. code-block:: python

                gsd.maximum_write_buffer_size = 128 * 1024**2
    """

    def __init__(self,
                 trigger,
                 filename,
                 filter=All(),
                 mode='ab',
                 truncate=False,
                 dynamic=None,
                 logger=None):

        super().__init__(trigger)

        dynamic_validation = OnlyFrom([
            'attribute',
            'property',
            'momentum',
            'topology',
            'configuration/box',
            'particles/N',
            'particles/position',
            'particles/orientation',
            'particles/velocity',
            'particles/angmom',
            'particles/image',
            'particles/types',
            'particles/typeid',
            'particles/mass',
            'particles/charge',
            'particles/diameter',
            'particles/body',
            'particles/moment_inertia',
        ],
                                      preprocess=_array_to_strings)

        dynamic = ['property'] if dynamic is None else dynamic
        self._param_dict.update(
            ParameterDict(filename=str(filename),
                          filter=ParticleFilter,
                          mode=str(mode),
                          truncate=bool(truncate),
                          dynamic=[dynamic_validation],
                          write_diameter=False,
                          maximum_write_buffer_size=64 * 1024 * 1024,
                          _defaults=dict(filter=filter, dynamic=dynamic)))

        self._logger = None if logger is None else _GSDLogWriter(logger)

    def _attach_hook(self):
        self._cpp_obj = _hoomd.GSDDumpWriter(
            self._simulation.state._cpp_sys_def, self.trigger, self.filename,
            self._simulation.state._get_group(self.filter), self.mode,
            self.truncate)

        self._cpp_obj.log_writer = self.logger

        # Maintain a list of open gsd writers
        weak_writer = weakref.ref(self)
        _open_gsd_writers.append(weak_writer)
        self._finalizer = weakref.finalize(self, _finalize_gsd, weak_writer,
                                           self._cpp_obj),

    @staticmethod
    def write(state, filename, filter=All(), mode='wb', logger=None):
        """Write the given simulation state out to a GSD file.

        Args:
            state (State): Simulation state.
            filename (str): File name to write.
            filter (hoomd.filter.filter_like): Select the particles to write.
            mode (str): The file open mode. Defaults to ``'wb'``.
            logger (hoomd.logging.Logger): Provide log quantities to write.

        The valid file modes for `write` are ``'wb'`` and ``'xb'``.
        """
        if mode != 'wb' and mode != 'xb':
            raise ValueError(f"Invalid GSD.write file mode: {mode}")

        writer = _hoomd.GSDDumpWriter(state._cpp_sys_def, Periodic(1),
                                      str(filename), state._get_group(filter),
                                      mode, False)

        if logger is not None:
            writer.log_writer = _GSDLogWriter(logger)
        writer.analyze(state._simulation.timestep)
        writer.flush()

    @property
    def logger(self):
        """hoomd.logging.Logger: Provide log quantities to write.

        May be `None`.

        """
        return self._logger

    @logger.setter
    def logger(self, logger):
        if logger is not None and isinstance(logger, Logger):
            logger = _GSDLogWriter(logger)
        else:
            raise ValueError("GSD.logger can only be set with a Logger.")
        if self._attached:
            self._cpp_obj.log_writer = logger
        self._logger = logger
        return self.logger

    def flush(self):
        """Flush the write buffer to the file.

        Example::

            gsd_writer.flush()

        Flush all write buffers::

            for writer in simulation.operations.writers:
                if hasattr(writer, 'flush'):
                    writer.flush()
        """
        if not self._attached:
            raise RuntimeError("The GSD file is unavailable until the"
                               "simulation runs for 0 or more steps.")

        self._cpp_obj.flush()


def _iterable_is_incomplete(iterable):
    """Checks that any nested attribute has no instances of RequiredArg.

    Given the arbitrary nesting of container types in the data model, we need to
    ensure that no RequiredArg values exist at any depth in a state loggable
    key. Otherwise, the GSD backend will fail in its conversion to NumPy arrays.
    """
    if (not isinstance(iterable, Collection) or isinstance(iterable, str)
            or len(iterable) == 0):
        return False
    incomplete = False

    if isinstance(iterable, Mapping):
        iter_ = iterable.keys()
    else:
        iter_ = iterable
    for v in iter_:
        if isinstance(v, Collection):
            incomplete |= _iterable_is_incomplete(v)
        else:
            incomplete |= v is RequiredArg
    return incomplete


class _GSDLogWriter:
    """Helper class to store `hoomd.logging.Logger` log data to GSD file.

    Class Attributes:
        _per_categories (`hoomd.logging.LoggerCategories`): category that
            contains all per-{particle,bond,...} quantities.
        _convert_categories (`hoomd.logging.LoggerCategories`): categories that
            contains all types that must be converted for storage in a GSD file.
        _skip_categories (`hoomd.logging.LoggerCategories`): categories that
            should be skipped by and not stored.
        _special_keys (`list` of `str`): list of loggable quantity names that
            need to be treated specially. In general, this is only for
            `type_shapes`.
        _global_prepend (`str`): a str that gets prepending into the namespace
            of each logged quantity.
    """
    _per_categories = LoggerCategories.any([
        'angle', 'bond', 'constraint', 'dihedral', 'improper', 'pair',
        'particle'
    ])
    _convert_categories = LoggerCategories.any(['string', 'strings'])
    _skip_categories = LoggerCategories['object']
    _special_keys = ['type_shapes']
    _global_prepend = 'log'

    def __init__(self, logger):
        self.logger = logger

    def log(self):
        """Get the flattened dictionary for consumption by GSD object."""
        log = dict()
        for key, value in _dict_flatten(self.logger.log()).items():
            if 'state' in key and _iterable_is_incomplete(value[0]):
                pass
            log_value, type_category = value
            type_category = LoggerCategories[type_category]
            # This has to be checked first since type_shapes has a category
            # LoggerCategories.object.
            if key[-1] in self._special_keys:
                self._log_special(log, key[-1], log_value)
            # Now we can skip any categories we don't process, in this case
            # LoggerCategories.object.
            if type_category not in self._skip_categories:
                if log_value is None:
                    continue
                else:
                    # This places logged quantities that are
                    # per-{particle,bond,...} into the correct GSD namespace
                    # log/particles/{remaining namespace}. This preserves OVITO
                    # intergration.
                    if type_category in self._per_categories:
                        log['/'.join((self._global_prepend, type_category.name
                                      + 's') + key)] = log_value
                    elif type_category in self._convert_categories:
                        self._log_convert_value(
                            log, '/'.join((self._global_prepend,) + key),
                            type_category, log_value)
                    else:
                        log['/'.join((self._global_prepend,) + key)] = \
                            log_value
            else:
                pass
        return log

    def _log_special(self, dict_, key, value):
        """Handles special keys such as type_shapes.

        When adding a key to this make sure this is the only option. In general,
        special cases like this should be avoided if possible.
        """
        if key == 'type_shapes':
            shape_list = [
                bytes(json.dumps(type_shape) + '\0', 'UTF-8')
                for type_shape in value
            ]
            max_len = np.max([len(shape) for shape in shape_list])
            num_shapes = len(shape_list)
            str_array = np.array(shape_list)
            dict_['particles/type_shapes'] = \
                str_array.view(dtype=np.int8).reshape(num_shapes, max_len)

    def _log_convert_value(self, dict_, key, category, value):
        """Convert loggable types that cannot be directly stored by GSD."""
        if category == LoggerCategories.string:
            value = bytes(value, 'UTF-8')
            value = np.array([value], dtype=np.dtype((bytes, len(value) + 1)))
            value = value.view(dtype=np.int8)
        elif category == LoggerCategories.strings:
            value = [bytes(v + '\0', 'UTF-8') for v in value]
            max_len = np.max([len(string) for string in value])
            num_strings = len(value)
            value = np.array(value)
            value = value.view(dtype=np.int8).reshape(num_strings, max_len)
        dict_[key] = value
