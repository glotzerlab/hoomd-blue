# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Implement DCD."""

from hoomd import _hoomd
from hoomd.filter import ParticleFilter, All
from hoomd.data.parameterdicts import ParameterDict
from hoomd.operation import Writer


class DCD(Writer):
    """Writes simulation trajectories in the DCD format.

    Args:
        trigger (hoomd.trigger.Periodic): Select the timesteps to write.
        filename (str): File name to write.
        filter (hoomd.filter.ParticleFilter): Select the particles to write.
            Defaults to `hoomd.filter.All`.
        overwrite (bool): When False, (the default) an existing DCD file will be
            appended to. When True, an existing DCD file *filename* will be
            overwritten.
        unwrap_full (bool): When False, (the default) particle coordinates are
            always written inside the simulation box.  When True, particles will
            be unwrapped into their current box image before writing to the DCD
            file.
        unwrap_rigid (bool): When False, (the default) individual particles are
            written inside the simulation box which breaks up rigid bodies near
            box boundaries. When True, particles belonging to the same rigid
            body will be unwrapped so that the body is continuous. The center of
            mass of the body remains in the simulation box, but some particles
            may be written just outside it. *unwrap_rigid* is ignored when
            *unwrap_full* is True.
        angle_z (bool): When True, the particle orientation angle is written to
            the z component (only useful for 2D simulations)

    `DCD` writes the simulation trajectory to the specified file in the DCD file
    format. DCD stores only particle positions and the box parameters, in
    length units, and is limited to simulations where the number of particles
    is fixed.

    Examples::

        writer = hoomd.write.DCD("trajectory.dcd", hoomd.trigger.Periodic(1000))
        dcd = hoomd.write.DCD(filename="data/dump.dcd",
                              trigger=hoomd.trigger.Periodic(100, 10))

    Warning:
        When you use `DCD` to append to an existing DCD file:

        * The period must be the same or the time data in the file will not be
          consistent.
        * `DCD` will not write out data at time steps that already are
          present in the DCD file.

    Attributes:
        filename (str): File name to write.
        trigger (hoomd.trigger.Periodic): Select the timesteps to write.
        filter (hoomd.filter.ParticleFilter): Select the particles to write.
        overwrite (bool): When False, an existing DCD file will be appended to.
            When True, an existing DCD file *filename* will be overwritten.
        unwrap_full (bool): When False, particle coordinates are always written
            inside the simulation box.  When True, particles will be unwrapped
            into their current box image before writing to the DCD file.
        unwrap_rigid (bool): When False, individual particles are written inside
            the simulation box which breaks up rigid bodies near box boundaries.
            When True, particles belonging to the same rigid body will be
            unwrapped so that the body is continuous. The center of mass of the
            body remains in the simulation box, but some particles may be
            written just outside it. *unwrap_rigid* is ignored when
            *unwrap_full* is True.
        angle_z (bool): When True, the particle orientation angle is written to
            the z component
    """

    def __init__(self,
                 trigger,
                 filename,
                 filter=All(),
                 overwrite=False,
                 unwrap_full=False,
                 unwrap_rigid=False,
                 angle_z=False):

        # initialize base class
        super().__init__(trigger)
        self._param_dict.update(
            ParameterDict(filename=str(filename),
                          filter=ParticleFilter,
                          overwrite=bool(overwrite),
                          unwrap_full=bool(unwrap_full),
                          unwrap_rigid=bool(unwrap_rigid),
                          angle_z=bool(angle_z)))
        self.filter = filter

    def _attach(self):
        group = self._simulation.state._get_group(self.filter)
        self._cpp_obj = _hoomd.DCDDumpWriter(
            self._simulation.state._cpp_sys_def, self.filename,
            int(self.trigger.period), group, self.overwrite)
        super()._attach()
