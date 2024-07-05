# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Define the ParticleSorter class."""

from hoomd.data.parameterdicts import ParameterDict
from hoomd.data.typeconverter import OnlyTypes
from hoomd.operation import Tuner
from hoomd import _hoomd
import hoomd
from math import log2, ceil


class ParticleSorter(Tuner):
    """Order particles in memory to improve performance.

    Args:
        trigger (hoomd.trigger.trigger_like): Select the timesteps on which to
            sort. Defaults to a ``hoomd.trigger.Periodic(200)`` trigger.

        grid (int): Resolution of the grid to use when sorting. The default
            value of `None` sets ``grid=4096`` in 2D simulations and
            ``grid=256`` in 3D simulations.

    `ParticleSorter` improves simulation performance by sorting the particles in
    memory along a space-filling curve. This takes particles that are close in
    space and places them close in memory, leading to a higher rate of
    cache hits when computing pair potentials.

    Note:
        New `hoomd.Operations` instances include a `ParticleSorter`
        constructed with default parameters.

    Attributes:
        trigger (hoomd.trigger.Trigger): Select the timesteps on which to sort.

        grid (int): Set the resolution of the space-filling curve.
            `grid` rounds up to the nearest power of 2 when set. Larger values
            of `grid` provide more accurate space-filling curves, but consume
            more memory (``grid**D * 4`` bytes, where *D* is the dimensionality
            of the system).
    """

    def __init__(self, trigger=200, grid=None):
        super().__init__(trigger)
        sorter_params = ParameterDict(
            grid=OnlyTypes(int,
                           postprocess=ParticleSorter._to_power_of_two,
                           preprocess=ParticleSorter._natural_number,
                           allow_none=True))
        self._param_dict.update(sorter_params)
        self.grid = grid

    @staticmethod
    def _to_power_of_two(value):
        return int(2.**ceil(log2(value)))

    @staticmethod
    def _natural_number(value):
        try:
            if value < 1:
                raise ValueError("Expected positive integer.")
            else:
                return value
        except TypeError:
            raise ValueError("Expected positive integer.")

    def _attach_hook(self):
        if isinstance(self._simulation.device, hoomd.device.GPU):
            cpp_cls = getattr(_hoomd, 'SFCPackTunerGPU')
        else:
            cpp_cls = getattr(_hoomd, 'SFCPackTuner')
        self._cpp_obj = cpp_cls(self._simulation.state._cpp_sys_def,
                                self.trigger)
