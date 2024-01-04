# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

r"""MPCD tuning operations.

These operations will affect the performance of MPCD simulations but not
their correctness.

"""

import hoomd
from hoomd.mpcd import _mpcd
from hoomd.operation import Tuner


class ParticleSorter(Tuner):
    r"""MPCD particle sorter.

    Args:
        trigger (hoomd.trigger.trigger_like): Number of integration steps
            between sorting.

    This tuner sorts the MPCD particles into cell order. To perform the sort,
    the cell list is first computed with the current particle order. Particles
    are then reordered in memory as they are stored in the cell list, which can
    significantly improve performance of all subsequent cell-based steps of the
    MPCD algorithm due to improved cache coherency.

    The optimal frequency for sorting depends on the number of particles, so the
    `trigger` itself should be tuned to give the maximum performance. The
    `trigger` should be a multiple of `hoomd.mpcd.collide.CollisionMethod.period`
    to avoid unnecessary cell list builds. Typically, using a small multiple
    (tens) of the collision period works best.

    For best performance, the `ParticleSorter` should **not** be added to
    `hoomd.Operations.tuners`. Instead, set it in `hoomd.mpcd.Integrator.solvent_sorter`.

    Essentially all MPCD systems benefit from sorting, so it is recommended
    to use one for all simulations!

    Attributes:
        trigger (hoomd.trigger.Trigger): Number of integration steps
            between sorting.

    """

    def __init__(self, trigger):
        super().__init__(trigger)

    def _attach_hook(self):
        if isinstance(self._simulation.device, hoomd.device.GPU):
            class_ = _mpcd.SorterGPU
        else:
            class_ = _mpcd.Sorter
        self._cpp_obj = class_(self._simulation.state._cpp_sys_def,
                               self.trigger)
