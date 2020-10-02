# Copyright (c) 2009-2020 The Regents of the University of Michigan This file is
# part of the HOOMD-blue project, released under the BSD 3-Clause License.

""" Write GSD files storing simulation trajectories and logging data."""

from hoomd import _hoomd
from hoomd.util import dict_flatten, array_to_strings
from hoomd.data.typeconverter import OnlyFrom
from hoomd.filter import _ParticleFilter, All
from hoomd.data.parameterdicts import ParameterDict
from hoomd.logging import Logger, TypeFlags
from hoomd.operation import _Analyzer
import numpy as np
import json


class GSD(_Analyzer):
    R""" Write simulation trajectories in the GSD format.

    Args:
        filename (str): File name to write.
        trigger (hoomd.trigger.Trigger): Select the timesteps to write.
        filter_ (hoomd.filter._ParticleFilter): Select the particles
            to write, defaults to `hoomd.filter.All`.
        overwrite (bool): When ``True``, overwite the file. When
            ``False`` append frames to ``filename`` if it exists and create the
            file if it does not, defaults to ``False``.
        truncate (bool): When ``True``, truncate the file and write a
            new frame 0 each time this operation triggers, defaults to
            ``False``.
        dynamic (list[str]): Quantity categories to save in every
            frame, defaults to property.
        log (hoomd.logging.Logger): A ``Logger`` object for GSD
            logging, defaults to ``None``.

    .. note::

        All parameters are also available as instance attributes. Only
        *trigger* and *log* may be modified after construction.

    `GSD` writes a simulation snapshot to the specified file each time it
    triggers. `GSD` can store all particle, bond, angle, dihedral, improper,
    pair, and constraint data fields in every frame of the trajectory.  `GSD`
    can write trajectories where the number of particles, number of particle
    types, particle types, diameter, mass, charge, or other quantities change
    over time. `GSD` can also store operation-specific state information
    necessary for restarting simulations and user-defined log quantities.

    To reduce file size, `GSD` does not write properties that are set to
    defaults. When masses, orientations, angular momenta, etc... are left
    default for all particles, these fields will not take up any space in the
    file. Additionally, `GSD` only writes *dynamic* quantities to all frames. It
    writes non-dynamic quantities only the first frame. When reading a GSD file,
    the data in frame 0 is read when a quantity is missing in frame *i*,
    supplying data that is static over the entire trajectory.  Set the *dynamic*
    parameter to specify dynamic attributes by category.  **property** is always
    dynamic:

    * **property**

        * particles/position
        * particles/orientation (*only written when values are not the
          default: [1,0,0,0]*)

    * **momentum**

        * particles/velocity
        * particles/angmom (*only written when values are not the
          default: [0,0,0,0]*)
        * particles/image (*only written when values are not the
          default: [0,0,0]*)

    * **attribute**

        * particles/N
        * particles/types
        * particles/typeid
        * particles/mass
        * particles/charge
        * particles/diameter
        * particles/body
        * particles/moment_inertia

    * **topology**

        * bonds/
        * angles/
        * dihedrals/
        * impropers/
        * constraints/
        * pairs/


    .. seealso::

        See the `GSD documentation <http://gsd.readthedocs.io/>`_ and `GitHub
        project <https://github.com/glotzerlab/gsd>`_ for more information on
        GSD files.

    .. note::

        When you use *filter_* to select a subset of the whole system,
        :py:class:`GSD` will write out all of the selected particles in
        ascending tag order and will **not** write out **topology**.

    .. note::

        All logged data chunks must be present in the first frame in the gsd
        file to provide the default value. Some (or all) chunks may be omitted
        on later frames.

    .. note::

        In MPI parallel simulations, the callback will be called on all ranks.
        :py:class:`GSD` will write the data returned by the root rank. Return
        values from all other ranks are ignored (and may be None).

    .. rubric:: Examples:

    TODO: link to example notebooks
    """

    def __init__(self,
                 filename,
                 trigger,
                 filter=All(),
                 overwrite=False,
                 truncate=False,
                 dynamic=None,
                 log=None):

        super().__init__(trigger)

        dynamic_validation = OnlyFrom(
            ['attribute', 'property', 'momentum', 'topology'],
            preprocess=array_to_strings)

        dynamic = ['property'] if dynamic is None else dynamic
        self._param_dict.update(
            ParameterDict(filename=str(filename),
                          filter=_ParticleFilter,
                          overwrite=bool(overwrite), truncate=bool(truncate),
                          dynamic=[dynamic_validation],
                          _defaults=dict(filter=filter, dynamic=dynamic)
                          )
            )

        self._log = None if log is None else _GSDLogWriter(log)

    def _attach(self):
        # validate dynamic property
        categories = ['attribute', 'property', 'momentum', 'topology']
        dynamic_quantities = ['property']

        if self.dynamic is not None:
            for v in self.dynamic:
                if v not in categories:
                    raise RuntimeError("GSD: dynamic quantity " + v +
                                       " is not valid")

            dynamic_quantities = ['property'] + self.dynamic

        self._cpp_obj = _hoomd.GSDDumpWriter(
            self._simulation.state._cpp_sys_def,
            self.filename,
            self._simulation.state.get_group(self.filter),
            self.overwrite,
            self.truncate)

        self._cpp_obj.setWriteAttribute('attribute' in dynamic_quantities)
        self._cpp_obj.setWriteProperty('property' in dynamic_quantities)
        self._cpp_obj.setWriteMomentum('momentum' in dynamic_quantities)
        self._cpp_obj.setWriteTopology('topology' in dynamic_quantities)
        self._cpp_obj.log_writer = self.log
        super()._attach()

    @staticmethod
    def write(state, filename, filter=All(), log=None):
        """Write the given simulation state out to a GSD file.

        Args:
            state (State): Simulation state.
            filename (str): File name to write.
            filter_ (``hoomd._ParticleFilter``): Select the particles to write.
            log (``hoomd.logger.Logger``): A ``Logger`` object for GSD logging.

        Note:
            The file is always overwritten.
        """
        writer = _hoomd.GSDDumpWriter(state._cpp_sys_def,
                                      filename,
                                      state.get_group(filter),
                                      True,
                                      False)

        if log is not None:
            writer.log_writer = GSDLogWriter(log)
        writer.analyze(state._simulation.timestep)

    @property
    def log(self):
        return self._log

    @log.setter
    def log(self, log):
        if isinstance(log, Logger):
            log = _GSDLogWriter(log)
        else:
            raise ValueError("GSD.log can only be set with a Logger.")
        if self._attached:
            self._cpp_obj.log_writer = log
        self._log = log


class _GSDLogWriter:
    """Helper class to store `hoomd.logging.Logger` log data to GSD file.

    Class Attributes:
        _per_flags (`hoomd.logging.TypeFlags`): flag that contains all
            per-{particle,bond,...} quantities.
        _convert_flags (`hoomd.logging.TypeFlags`): flag that contains all types
            that must be converted for storage in a GSD file.
        _skip_flags (`hoomd.logging.TypeFlags`): flags that should be skipped by
            and not stored.
        _special_keys (`list` of `str`): list of loggable quantity names that
            need to be treated specially. In general, this is only for
            `type_shapes`.
        _global_prepend (`str`): a str that gets prepending into the namespace
            of each logged quantity.
    """
    _per_flags = TypeFlags.any(['angle', 'bond', 'constraint', 'dihedral',
                                'improper', 'pair', 'particle'])
    _convert_flags = TypeFlags.any(['string', 'strings'])
    _skip_flags = TypeFlags['object']
    _special_keys = ['type_shapes']
    _global_prepend = 'log'

    def __init__(self, logger):
        self.logger = logger

    def log(self):
        """Get the flattened dictionary for consumption by GSD object."""
        log = dict()
        for key, value in dict_flatten(self.logger.log()).items():
            log_value, type_flag = value
            type_flag = TypeFlags[type_flag]
            # This has to be checked first since type_shapes has a flag
            # TypeFlags.object.
            if key[-1] in self._special_keys:
                self._log_special(log, key[-1], log_value)
            # Now we can skip any flags we don't process, in this case
            # TypeFlags.object.
            if type_flag not in self._skip_flags:
                if log_value is None:
                    continue
                else:
                    # This places logged quantities that are
                    # per-{particle,bond,...} into the correct GSD namespace
                    # log/particles/{remaining namespace}. This preserves OVITO
                    # intergration.
                    if type_flag in self._per_flags:
                        log['/'.join((self._global_prepend,
                                      type_flag.name + 's') + key)] = log_value
                    elif type_flag in self._convert_flags:
                        self._log_convert_value(
                            log, '/'.join((self._global_prepend,) + key),
                            type_flag, log_value)
                    else:
                        log['/'.join((self._global_prepend,) + key)] = \
                            log_value
            else:
                pass
        return log

    def _write_frame(self, _gsd):
        _gsd.writeLogQuantities(self.log())

    def _log_special(self, dict_, key, value):
        """Handles special keys such as type_shapes.

        When adding a key to this make sure this is the only option. In general,
        special cases like this should be avoided if possible.
        """
        if key == 'type_shapes':
            shape_list = [bytes(json.dumps(type_shape) + '\0', 'UTF-8')
                          for type_shape in value]
            max_len = np.max([len(shape) for shape in shape_list])
            num_shapes = len(shape_list)
            str_array = np.array(shape_list)
            dict_['particles/type_shapes'] = \
                str_array.view(dtype=np.int8).reshape(num_shapes, max_len)

    def _log_convert_value(self, dict_, key, flag, value):
        """Convert loggable types that cannot be directly stored by GSD."""
        if flag == TypeFlags.string:
            value = bytes(value, 'UTF-8')
            value = np.array([value], dtype=np.dtype((bytes, len(value) + 1)))
            value = value.view(dtype=np.int8)
        if flag == TypeFlags.strings:
            value = [bytes(v + '\0', 'UTF-8') for v in value]
            max_len = np.max([len(string) for string in value])
            num_strings = len(value)
            value = np.array(value)
            value = value.view(dtype=np.int8).reshape(num_strings, max_len)
        dict_[key] = value
