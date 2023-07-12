# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Provides an HDF5 logging backend for a fast binary logging format."""

import h5py

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
        self.extend_by = 1
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
        self._fh["data"].attrs["frames"] = self._frame
        if self._frame % self.extend_by == 0:
            new_size = self._frame + self.extend_by
        else:
            new_size = 0
        for key, (value, category) in log_dict.items():
            if logging.LoggerCategories[category] in self._reject_categories:
                continue
            if value is None:
                continue
            str_key = "/".join(("data",) + key)
            if str_key not in self._fh:
                # TODO: We should be able to extend this to just fill the
                # previous frames with 0.0 and warn.
                raise RuntimeError(
                    "The logged quantities cannot change within a file.")
            dataset = self._fh[str_key]
            if new_size:
                dataset.resize(new_size, axis=0)
            dataset[self._frame, ...] = value
        self._frame += 1

    def _create_database(self, key: str, shape, dtype):
        self._fh.create_dataset(
            key,
            (self.extend_by,) + shape,
            dtype=dtype,
            maxshape=(None,) + shape,
        )

    def _initialize_datasets(self, log_dict):
        for key, (value, category) in log_dict.items():
            if category == "scalar":
                data_shape = (1,)
                dtype = "f8" if isinstance(value, float) else "i8"
            else:
                data_shape = value.shape
                dtype = value.dtype
            self._create_database("/".join(("data",) + key), data_shape, dtype)

    def _find_frame(self):
        if "data" in self._fh:
            return self._fh["data"].attrs["frames"] + 1
        return 0

    def _validate_scheme(self):
        if "data" in self._fh:
            if "hoomd-schema" not in self._fh["data"].attrs:
                raise RuntimeError("Validation of existing HDF5 file failed.")
        else:
            self._fh["data"].attrs["hoomd-schema"] = [0, 1]


__all__ = ["HDF5Logger"]
