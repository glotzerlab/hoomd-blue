# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

r"""MPCD tuning operations.

These operations will affect the performance of MPCD simulations but not
their correctness.

.. invisible-code-block: python

    simulation = hoomd.util.make_example_simulation(mpcd_types=["A"])
    simulation.operations.integrator = hoomd.mpcd.Integrator(dt=0.1)

"""

import hoomd
from hoomd.mpcd import _mpcd
from hoomd.operation import TriggeredOperation


class ParticleSorter(TriggeredOperation):
    r"""MPCD particle sorter.

    Args:
        trigger (hoomd.trigger.trigger_like): Select the time steps on which to
            sort.

    This tuner sorts the MPCD particles into cell order. To perform the sort,
    the cell list is first computed with the current particle order. Particles
    are then reordered in memory as they are stored in the cell list, which can
    significantly improve performance of all subsequent cell-based steps of the
    MPCD algorithm due to improved cache coherency.

    The optimal frequency for sorting depends on the number of particles, so the
    `trigger` itself should be tuned to give the maximum performance. The
    trigger's period should be a multiple of
    `hoomd.mpcd.collide.CollisionMethod.period` to avoid unnecessary cell list
    builds. Typically, using a small multiple (tens) of the collision period
    works best.

    To achieve the best performance, the `ParticleSorter` is not added to
    `hoomd.Operations.tuners`. Instead, set it in
    `hoomd.mpcd.Integrator.mpcd_particle_sorter`.

    Essentially all MPCD systems benefit from sorting, so it is recommended
    to use one for all simulations!

    .. rubric:: Example:

    .. code-block:: python

        sorter = hoomd.mpcd.tune.ParticleSorter(trigger=20)
        simulation.operations.integrator.mpcd_particle_sorter = sorter

    Attributes:
        trigger (hoomd.trigger.Trigger): Number of integration steps
            between sorting.

            .. rubric:: Example:

            .. code-block:: python

                sorter.trigger = 20

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
