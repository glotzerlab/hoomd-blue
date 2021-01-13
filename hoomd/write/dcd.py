import hoomd
from hoomd import _hoomd
from hoomd.filter import ParticleFilter, All
from hoomd.data.parameterdicts import ParameterDict
from hoomd.operation import Writer



class DCD(Writer):
    R""" Writes simulation snapshots in the DCD format

    Args:
        filename (str): File name to write.
        trigger (hoomd.trigger.Trigger): Select the timesteps to write.
        filter (hoomd.filter.ParticleFilter): Select the particles to write.
            Defaults to `hoomd.filter.All`.
        overwrite (bool): When False, (the default) an existing DCD file will be appended to. When True, an existing DCD
                          file *filename* will be overwritten.
        unwrap_full (bool): When False, (the default) particle coordinates are always written inside the simulation box.
                            When True, particles will be unwrapped into their current box image before writing to the dcd file.
        unwrap_rigid (bool): When False, (the default) individual particles are written inside the simulation box which
               breaks up rigid bodies near box boundaries. When True, particles belonging to the same rigid body will be
               unwrapped so that the body is continuous. The center of mass of the body remains in the simulation box, but
               some particles may be written just outside it. *unwrap_rigid* is ignored when *unwrap_full* is True.
        angle_z (bool): When True, the particle orientation angle is written to the z component (only useful for 2D simulations)

    Every *period* time steps a new simulation snapshot is written to the
    specified file in the DCD file format. DCD only stores particle positions,
    in distance units.

    Due to constraints of the DCD file format, once you stop writing to
    a file via ``disable``, you cannot continue writing to the same file,
    nor can you change the period of the dump at any time. Either of these tasks
    can be performed by creating a new dump file with the needed settings.

    Examples::

        dump.dcd(filename="trajectory.dcd", trigger=hoomd.trigger.Periodic(1000))
        dcd = dump.dcd(filename="data/dump.dcd", trigger=hoomd.trigger.Periodic(100, 10))

    Warning:
        When you use dump.dcd to append to an existing dcd file:

        * The period must be the same or the time data in the file will not be consistent.
        * dump.dcd will not write out data at time steps that already are present in the dcd file to maintain a
          consistent timeline
    """
    def __init__(self,
                 filename,
                 trigger,
                 filter=All(),
                 overwrite=False,
                 unwrap_full=False,
                 unwrap_rigid=False,
                 angle_z=False):

        # initialize base class
        super().__init__(trigger)
        self.filter = filter
        self.unwrap_full = unwrap_full
        self.unwrap_rigid = unwrap_rigid
        self.angle_z = angle_z
        self._param_dict.update(
            ParameterDict(filename=str(filename),
                          period=trigger.period,
                          overwrite=bool(overwrite)))

    def _attach(self):
        self._cpp_obj = _hoomd.DCDDumpWriter(self._simulation.state._cpp_sys_def,
                                             self.filename,
                                             int(self.period),
                                             self._simulation.state._get_group(self.filter),
                                             self.overwrite)

        self._cpp_obj.setUnwrapFull(self.unwrap_full)
        self._cpp_obj.setUnwrapRigid(self.unwrap_rigid)
        self._cpp_obj.setAngleZ(self.angle_z)

    def enable(self):
        """ The DCD dump writer cannot be re-enabled """
        if self.enabled is False:
            raise RuntimeError('DCD output cannot be re-enabled after it has been disabled');

    def set_period(self, period):
        raise RuntimeError('The period of a DCD writer cannot be changed')
