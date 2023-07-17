# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Provides an HDF5 logging backend for a fast binary logging format."""

import h5py
import numpy as np

import hoomd.custom as custom
import hoomd.logging as logging
import hoomd.util as util


class HDF5Logger(custom.Action):
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

    def __init__(self, fn, logger, mode="a"):
        self.fn = fn
        self.logger = logger
        self.mode = mode
        self._fh = h5py.File(self.fn, mode=mode)
        self._validate_scheme()
        self._frame = self._find_frame()

    def __del__(self):
        """Closes file upon destruction."""
        self._fh.close()

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
            str_key = "/".join(("data",) + key)
            if str_key not in self._fh:
                raise RuntimeError(
                    "The logged quantities cannot change within a file.")
            dataset = self._fh[str_key]
            dataset.resize(self._frame + 1, axis=0)
            dataset[self._frame, ...] = value
        self._frame += 1
        self._fh["data"].attrs["frames"] = self._frame

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
            self._create_database("/".join(("data",) + key), data_shape, dtype,
                                  chunk_size)

    def _find_frame(self):
        if "data" in self._fh:
            return self._fh["data"].attrs["frames"]
        return 0

    def _validate_scheme(self):
        if "data" in self._fh:
            if "hoomd-schema" not in self._fh["data"].attrs:
                raise RuntimeError("Validation of existing HDF5 file failed.")
        else:
            group = self._fh.create_group("data")
            group.attrs["hoomd-schema"] = [0, 1]
            group.attrs["frames"] = 0


__all__ = ["HDF5Logger"]
