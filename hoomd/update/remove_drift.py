# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Implement RemoveDrift.

.. invisible-code-block: python

    simulation = hoomd.util.make_example_simulation()
"""

import hoomd
from hoomd.operation import Updater
from hoomd.data.typeconverter import NDArrayValidator
from hoomd.data.parameterdicts import ParameterDict
from hoomd import _hoomd
import numpy as np


class RemoveDrift(Updater):
    r"""Remove the average drift from a restrained system.

    Args:
        reference_positions ((*N_particles*, 3) `numpy.ndarray` of `float`):
            the reference positions :math:`[\mathrm{length}]`.
        trigger (hoomd.trigger.trigger_like): Select the timesteps to remove
            drift.

    `RemoveDrift` computes the mean drift :math:`\vec{D}` from the
    given `reference_positions` (:math:`\vec{r}_{ref, i}`):

    .. math::

        \vec{D} = \frac{1}{\mathrm{N_{particles}}}
            \sum_{i=0}^\mathrm{N_{particles-1}}
            \mathrm{minimum\_image}(\vec{r}_i - \vec{r}_{ref,i})

    `RemoveDrift` then shifts all particles in the system by
    :math:`-\vec{D}`:

    .. math::

        \vec{r}_i \leftarrow \mathrm{minimum\_image}(\vec{r}_i - \vec{D})

    Tip:
        Use `RemoveDrift` with `hoomd.hpmc.external.field.Harmonic` to
        improve the accuracy of Frenkel-Ladd calculations.

    .. rubric:: Example:

    .. code-block:: python

        remove_drift = hoomd.update.RemoveDrift(
            reference_positions=[(0,0,0), (1,0,0)])
        simulation.operations.updaters.append(remove_drift)

    Attributes:
        reference_positions ((*N_particles*, 3) `numpy.ndarray` of `float`):
            the reference positions :math:`[\mathrm{length}]`.

            .. rubric:: Example:

            .. code-block:: python

                remove_drift.reference_positions = [(0,0,0), (1,0,0)]
    """

    def __init__(self, reference_positions, trigger=1):
        super().__init__(trigger)
        self._param_dict.update(
            ParameterDict({
                "reference_positions": NDArrayValidator(np.float64, (None, 3))
            }))
        self.reference_positions = reference_positions

    def _attach_hook(self):
        if isinstance(self._simulation.device, hoomd.device.GPU):
            self._simulation.device._cpp_msg.warning(
                "Falling back on CPU. No GPU implementation available.\n")

        self._cpp_obj = _hoomd.UpdaterRemoveDrift(
            self._simulation.state._cpp_sys_def, self.trigger,
            self.reference_positions)
