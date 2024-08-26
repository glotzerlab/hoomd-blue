# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Write GSD last :math:`N` frames at user direction.

.. invisible-code-block: python

    simulation = hoomd.util.make_example_simulation()
    burst_filename = tmp_path / 'trajectory.gsd'
"""

from hoomd import _hoomd
from hoomd.filter import All
from hoomd.data.parameterdicts import ParameterDict
from hoomd.write.gsd import GSD


class Burst(GSD):
    r"""Write the last :math:`N` stored frames in the GSD format.

    When triggered, `Burst` adds a frame (up the last :math:`N` frames) in a
    buffer. Call `dump` to write the frames to the file. When the the next frame
    would result in :math:`N + 1` frames being stored, the oldest frame is
    removed and the new frame is added.

    Args:
        trigger (hoomd.trigger.trigger_like): Select the timesteps to store
            in the buffer.
        filename (str): File name to write.
        filter (hoomd.filter.filter_like): Select the particles to write.
            Defaults to `hoomd.filter.All`.
        mode (str): The file open mode. Defaults to ``'ab'``.
        dynamic (list[str]): Field names and/or field categores to save in
            all frames. Defaults to ``['property']``.
        logger (hoomd.logging.Logger): Provide log quantities to write. Defaults
            to `None`.
        max_burst_size (int): The maximum number of frames to store before
            between writes. -1 represents no limit. Defaults to -1.
        write_at_start (bool): When ``True`` **and** the file does not exist or
            has 0 frames: write one frame with the current state of the system
            when `hoomd.Simulation.run` is called. Defaults to ``False``.
        clear_whole_buffer_after_dump (bool): When ``True`` the buffer is
            emptied after calling `dump` each time. When ``False``, `dump` removes
            frames from the buffer unil the ``end`` index. Defaults to ``True``.

    Warning:
        `Burst` errors when attempting to create a file or writing to one with
        zero frames when ``write_at_start`` is ``False``.

    Note:
        When analyzing files created by `Burst`, generally the first frame is
        not associated with the call to `Burst.dump`.

    .. rubric:: Example:

    .. code-block:: python

        burst = hoomd.write.Burst(trigger=hoomd.trigger.Periodic(1_000),
                                filename=burst_filename,
                                max_burst_size=100,
                                write_at_start=True)
        simulation.operations.writers.append(burst)

    See Also:
        The base class `hoomd.write.GSD`

    Attributes:
        max_burst_size (int): The maximum number of frames to store before
            between writes. -1 represents no limit.

            .. rubric:: Example:

            .. code-block:: python

                burst.max_burst_size = 200

        write_at_start (bool): When ``True`` **and** the file does not exist or
            has 0 frames: write one frame with the current state of the system
            when `hoomd.Simulation.run` is called (*read only*).

            .. rubric:: Example:

            .. code-block:: python

                write_at_start = burst.write_at_start

        clear_whole_buffer_after_dump (bool): When ``True`` the buffer is
            emptied after calling `dump` each time. When ``False``, `dump` removes
            frames from the buffer unil the ``end`` index.

            .. rubric:: Example:

            .. code-block:: python

                    burst.clear_buffer_after_dump = False
    """

    def __init__(self,
                 trigger,
                 filename,
                 filter=All(),
                 mode='ab',
                 dynamic=None,
                 logger=None,
                 max_burst_size=-1,
                 write_at_start=False,
                 clear_whole_buffer_after_dump=True):
        super().__init__(trigger=trigger,
                         filename=filename,
                         filter=filter,
                         mode=mode,
                         dynamic=dynamic,
                         logger=logger)
        self._param_dict.pop("truncate")
        self._param_dict.update(
            ParameterDict(max_burst_size=int, write_at_start=bool))
        self._param_dict.update({
            "max_burst_size": max_burst_size,
            "write_at_start": write_at_start,
            "clear_whole_buffer_after_dump": clear_whole_buffer_after_dump
        })

    def _attach_hook(self):
        sim = self._simulation
        self._cpp_obj = _hoomd.GSDDequeWriter(
            sim.state._cpp_sys_def, self.trigger, self.filename,
            sim.state._get_group(self.filter), self.logger, self.max_burst_size,
            self.mode, self.write_at_start, self.clear_whole_buffer_after_dump,
            sim.timestep)

    def dump(self, start=0, end=-1):
        """Write stored frames in range to the file and empties the buffer.

        This method alllows for custom writing of frames at user specified
        conditions.

        Args:
            start (int): The first frame to write. Defaults to 0.
            end (int): The last frame to write.
                Defaults to -1 (last frame).

        .. rubric:: Example:

        .. code-block:: python

            burst.dump()
        """
        if self._attached:
            self._cpp_obj.dump(start, end)

    def __len__(self):
        """Get the current length of the internal frame buffer.

        .. rubric:: Example:

        .. code-block:: python

            buffered_frames = len(burst)
        """
        if self._attached:
            return len(self._cpp_obj)
        return 0
