# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Provides an HDF5 logging backend for a fast binary logging format."""

import h5py
import numpy as np

import hoomd.custom as custom
import hoomd.logging as logging
import hoomd.util as util

from hoomd.write.custom_writer import _InternalCustomWriter
from hoomd.data.parameterdicts import ParameterDict


class _HDF5LoggerInternal(custom._InternalAction):
    """A HDF5 HOOMD logging backend."""

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

    def __init__(self, filename, logger, mode="a"):
        param_dict = ParameterDict(filename=str,
                                   logger=logging.Logger,
                                   mode=str)
        param_dict.update({
            "filename": filename,
            "logger": logger,
            "mode": mode
        })
        self._param_dict = param_dict
        self._fh = h5py.File(self.filename, mode=mode)
        self._validate_scheme()
        self._frame = self._find_frame()

    def __del__(self):
        """Closes file upon destruction."""
        self._fh.close()

    def _setattr_param(self, attr, value):
        """Makes self._param_dict attributes read only."""
        raise ValueError("Attribute {} is read-only.".format(attr))

    def act(self, timestep):
        """Write a new frame of logger data to the HDF5 file."""
        log_dict = util._dict_flatten(self.logger.log())
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

    def flush(self):
        """Write out all data currently buffered in memory.

        Without calling this, data may be stored in the h5py.File object without
        being written to the on disk file yet.
        """
        self._fh.flush()

    def _create_database(self, key: str, shape, dtype, chunk_size):
        self._fh.create_dataset(
            key,
            shape,
            dtype=dtype,
            chunks=chunk_size,
            maxshape=(None,) + shape[1:],
        )

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
                dtype = "f8" if isinstance(value, float) else "i8"
                chunk_size = (500,)
            else:
                if not isinstance(value, np.ndarray):
                    value = np.asarray(value)
                data_shape = (1,) + value.shape
                dtype = value.dtype
                chunk_size = [dim if dim <= 10 else dim for dim in data_shape]
                chunk_size[0] = 1
                chunk_size = tuple(chunk_size)
            self._create_database("/".join(("hoomd-data",) + key), data_shape,
                                  dtype, chunk_size)

    def _find_frame(self):
        if "hoomd-data" in self._fh:
            return self._fh["hoomd-data"].attrs["frames"]
        return 0

    def _validate_scheme(self):
        if "hoomd-data" in self._fh:
            if "hoomd-schema" not in self._fh["hoomd-data"].attrs:
                raise RuntimeError("Validation of existing HDF5 file failed.")
        else:
            group = self._fh.create_group("hoomd-data")
            group.attrs["hoomd-schema"] = [0, 1]
            group.attrs["frames"] = 0


class HDF5Logger(_InternalCustomWriter):
    """An HDF5 logger backend.

    This class handles scalar and array data storing them in HDF5 resizable
    datasets.

    Note:
        This class requires that ``h5py`` be installed.

    Important:
        The HDF5 can be used for other data storage; however, the "hoomd-data"
        key is reserved for use by this class. An exception will be thrown if
        this requirement is not met.

    Warning:
        This class cannot handle string loggables.

    Args:
        trigger (hoomd.trigger.trigger_like): The trigger to determine when to
            run the HDF5 backend.
        filename (str): The filename of the HDF5 file to write to.
        logger (hoomd.logging.Logger): The logger instance to use for querying
            log data.
        mode (`str`, optional): The mode to open the file in. Defaults to "a".

    Attributes:
        trigger (hoomd.trigger.trigger_like): The trigger to determine when to
            run the HDF5 backend.
        filename (str): The filename of the HDF5 file written to.
        logger (hoomd.logging.Logger): The logger instance used for querying
            log data.
        mode (str): The mode the file was opened in.

    .. rubric:: Example

    .. invisible-code-block: python
        import hoomd

        simulation = hoomd.util.make_example_simulation()
        logger = hoomd.loging.Logger()

    .. code-block:: python

        h5_writer = hoomd.write.HDF5Logger(10_000, "simulation-log.h5", logger)
        simulation.operations += h5_writer
    """
    _internal_class = _HDF5LoggerInternal

    def write(self):
        """Write out data to the HDF5 file.

        Writes out a frame from the composed logger.
        """
        self._action.act()


__all__ = ["HDF5Logger"]
