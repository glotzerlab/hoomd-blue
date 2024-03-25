# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Provides an HDF5 logging backend for a fast binary logging format.

.. invisible-code-block: python

    import hoomd

    try:
        import h5py
        h5py_not_available = False
    except ImportError:
        h5py_not_available = True

    if not h5py_not_available:
        simulation = hoomd.util.make_example_simulation()
        hdf5_filename = tmp_path / "simulation_log.h5"
        logger = hoomd.logging.Logger(
            hoomd.write.HDF5Log.accepted_categories)
        hdf5_writer = hoomd.write.HDF5Log(
            10_000, hdf5_filename, logger)

.. skip: start if(h5py_not_available)
"""

import copy
import functools
from pathlib import PurePath

import numpy as np

import hoomd.custom as custom
import hoomd.logging as logging
import hoomd.data.typeconverter as typeconverter
import hoomd.util as util

from hoomd.write.custom_writer import _InternalCustomWriter
from hoomd.data.parameterdicts import ParameterDict
import warnings

try:
    import h5py
except ImportError:
    h5py = None


class _SkipIfNone:

    def __init__(self, attr):
        self._attr = attr

    def __call__(self, method):

        @functools.wraps(method)
        def func(s, *args, **kwargs):
            if getattr(s, self._attr, None) is None:
                return
            return method(s, *args, **kwargs)

        return func


_skip_fh = _SkipIfNone("_fh")


class _HDF5LogInternal(custom._InternalAction):
    """A HDF5 HOOMD logging backend."""

    _skip_for_equality = custom._InternalAction._skip_for_equality | {
        "_fh", "_attached_"
    }

    flags = (
        custom.Action.Flags.ROTATIONAL_KINETIC_ENERGY,
        custom.Action.Flags.PRESSURE_TENSOR,
        custom.Action.Flags.EXTERNAL_FIELD_VIRIAL,
    )

    _reject_categories = logging.LoggerCategories.any((
        logging.LoggerCategories.object,
        logging.LoggerCategories.strings,
        logging.LoggerCategories.string,
    ))

    accepted_categories = ~_reject_categories

    _SCALAR_CHUNK = 512
    _MULTIFRAME_ARRAY_CHUNK_MAXIMUM = 4096

    def __init__(self, filename, logger, mode="a"):
        if h5py is None:
            raise ImportError(f"{type(self)} requires the h5py pacakge.")
        param_dict = ParameterDict(filename=typeconverter.OnlyTypes(
            (str, PurePath)),
                                   logger=logging.Logger,
                                   mode=str)
        if (rejects := self._reject_categories
                & logger.categories) != logging.LoggerCategories["NONE"]:
            reject_str = logging.LoggerCategories._get_string_list(rejects)
            raise ValueError(f"Cannot have {reject_str} in logger categories.")
        param_dict.update({
            "filename": filename,
            "logger": logger,
            "mode": mode
        })
        self._param_dict = param_dict
        self._fh = None
        self._attached_ = False

    def _initialize(self, communicator):
        if communicator is None or communicator.rank == 0:
            self._fh = h5py.File(self.filename, mode=self.mode)
        else:
            self._fh = None
        self._validate_scheme()
        self._frame = self._find_frame()

    def __del__(self):
        """Closes file upon destruction."""
        if getattr(self, "_fh", None) is not None:
            self._fh.close()

    def _setattr_param(self, attr, value):
        """Makes self._param_dict attributes read only."""
        raise ValueError("Attribute {} is read-only.".format(attr))

    def attach(self, simulation):
        self._initialize(simulation.device.communicator)
        self._attached_ = True

    def _attached(self):
        return self._attached_

    def detach(self):
        self._attached_ = False
        if self._fh is not None:
            self._fh.close()

    def act(self, timestep):
        """Write a new frame of logger data to the HDF5 file."""
        log_dict = util._dict_flatten(self.logger.log())
        if self._fh is None:
            return
        if self._frame == 0:
            self._initialize_datasets(log_dict)
        for key, (value, category) in log_dict.items():
            if logging.LoggerCategories[category] in self._reject_categories:
                continue
            if value is None:
                continue
            str_key = "/".join(("hoomd-data",) + key)
            if str_key not in self._fh:
                raise RuntimeError(
                    "The logged quantities cannot change within a file.")
            dataset = self._fh[str_key]
            dataset.resize(self._frame + 1, axis=0)
            dataset[self._frame, ...] = value
        self._frame += 1
        self._fh["hoomd-data"].attrs["frames"] = self._frame

    @_skip_fh
    def flush(self):
        """Write out all data currently buffered in memory.

        .. rubric:: Examples:

        Flush one writer:

        .. code-block:: python

            hdf5_writer.flush()

        Flush all write buffers:

        .. code-block:: python

            for writer in simulation.operations.writers:
                if hasattr(writer, 'flush'):
                    writer.flush()
        """
        self._fh.flush()

    @_skip_fh
    def _create_dataset(self, key: str, shape, dtype, chunk_size):
        self._fh.create_dataset(
            key,
            shape,
            dtype=dtype,
            chunks=chunk_size,
            maxshape=(None,) + shape[1:],
        )

    @_skip_fh
    def _initialize_datasets(self, log_dict):
        """Create datasets setting shape, dtype, and chunk size.

        Chunk size does not seem to matter much to the write performance of the
        file surprisingly. However, it has a drastic effect on reads. The
        current settings balance file size (smaller scalar chunks) with reading
        speed.

        Tests were done on 1,000 frame files for writes and reads and tested for
        writing and reading speed.
        """
        for key, (value, category) in log_dict.items():
            chunk_size = None
            if category == "scalar":
                data_shape = (1,)
                if isinstance(value, (np.number, np.bool_)):
                    dtype = value.dtype
                elif isinstance(value, int):
                    dtype = np.dtype(bool) if isinstance(value, bool) else "i8"
                else:
                    dtype = "f8"
                chunk_size = (self._SCALAR_CHUNK,)
            else:
                if not isinstance(value, np.ndarray):
                    value = np.asarray(value)
                data_shape = (1,) + value.shape
                dtype = value.dtype
                chunk_size = (max(
                    self._MULTIFRAME_ARRAY_CHUNK_MAXIMUM // value.nbytes,
                    1),) + data_shape[1:]
            self._create_dataset("/".join(("hoomd-data",) + key), data_shape,
                                 dtype, chunk_size)

    @_skip_fh
    def _find_frame(self):
        if "hoomd-data" in self._fh:
            return self._fh["hoomd-data"].attrs["frames"]
        return 0

    @_skip_fh
    def _validate_scheme(self):
        if "hoomd-data" in self._fh:
            if "hoomd-schema" not in self._fh["hoomd-data"].attrs:
                raise RuntimeError("Validation of existing HDF5 file failed.")
        else:
            group = self._fh.create_group("hoomd-data")
            group.attrs["hoomd-schema"] = [0, 1]
            group.attrs["frames"] = 0

    def __getstate__(self):
        state = copy.copy(self.__dict__)
        del state["_fh"]
        state["_attached_"] = False
        return state

    def __setstate__(self, state):
        self.__dict__ = state


class HDF5Log(_InternalCustomWriter):
    """Write loggable simulation data to HDF5 files.

    This class stores resizable scalar and array data in the HDF5 file format.

    Note:
        This class requires that ``h5py`` be installed.

    Important:
        The HDF5 file can be used for other data storage; however, the
        "hoomd-data" key is reserved for use by this class. An exception will be
        thrown if this requirement is not met.

    Warning:
        This class cannot handle string, strings, or object loggables.

    Args:
        trigger (hoomd.trigger.trigger_like): The trigger to determine when to
            write to the HDF5 file.
        filename (str): The filename of the HDF5 file to write to.
        logger (hoomd.logging.Logger): The logger instance to use for querying
            log data.
        mode (`str`, optional): The mode to open the file in. Available values
            are "w", "x" and "w-", "a", and "r+". Defaults to "a". See the
            h5py_ documentation for more details).

    .. _h5py:
        https://docs.h5py.org/en/stable/high/file.html#opening-creating-files

    .. rubric:: Example:

    .. code-block:: python

        logger = hoomd.logging.Logger(
            hoomd.write.HDF5Log.accepted_categories)
        hdf5_log = hoomd.write.HDF5Log(
            trigger=hoomd.trigger.Periodic(10_000),
            filename=hdf5_filename,
            logger=logger)
        simulation.operations.writers.append(hdf5_log)

    Attributes:
        accepted_categories (hoomd.logging.LoggerCategories): The enum value
            for all accepted categories for `HDF5Log` instances which is all
            categories other than "string", "strings", and "object" (see
            `hoomd.logging.LoggerCategories`).

            .. rubric:: Example:

            .. code-block:: python

                accepted_categories = hoomd.write.HDF5Log.accepted_categories

        filename (str): The filename of the HDF5 file written to (*read only*).

            .. rubric:: Example:

            .. code-block:: python

                filename = hdf5_log.filename

        logger (hoomd.logging.Logger): The logger instance used for querying
            log data (*read only*).

            .. rubric:: Example:

            .. code-block:: python

                logger = hdf5_log.logger

        mode (str): The mode the file was opened in (*read only*).

            .. rubric:: Example:

            .. code-block:: python

                mode = hdf5_log.mode
    """
    _internal_class = _HDF5LogInternal
    _wrap_methods = ("flush",)

    def write(self, timestep=None):
        """Write out data to the HDF5 file.

        Writes out a frame at the current timestep from the composed logger.

        Warning:
            This may not be able to write out quantities which require the
            pressure tensor, rotational kinetic energy, or external field
            virial.

        .. rubric:: Example:

        .. code-block:: python

            hdf5_writer.write()

        .. deprecated:: 4.5.0

            Use `Simulation` to call the operation.
        """
        warnings.warn(
            "`HDF5Log.writer` is deprecated,"
            "use `Simulation` to call the operation.", FutureWarning)
        self._action.act(timestep)


__all__ = ["HDF5Log"]
