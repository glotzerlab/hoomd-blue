# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

"""Implement RemoveDrift."""

from hoomd.operation import Updater
from hoomd.data.parameterdicts import ParameterDict
from hoomd import _hoomd


class RemoveDrift(Updater):
    r"""Remove the average drift from a system restrained on a lattice.

    Args:
        reference_positions ((*N_particles*, 3) `numpy.ndarray` of ``numpy.float64``): the
            reference positions of the lattice :math:`[\\mathrm{length}]`.
        trigger (`hoomd.trigger.Trigger`): Select the timesteps to remove drift.

    During the time steps specified by *trigger*, the average drift :math:`\\Delta\\vec{r}`
    from the *reference_positions* (:math:`\\vec{r}_{ref}`) is substracted from the
    current particle positions. The drift computed by this updater is given by:

    .. math::

        \\Delta\\vec{r} = \\frac{1}{\\mathrm{N_{particles}} \\sum_i \\mathrm{min\_image}(\\vec{r}_i - \\vec{r}_{ref,i})
    """

    def __init__(self, reference_positions, trigger=1):
        super().__init__(trigger)
        self._param_dict.update({
            "reference_positions":
                hoomd.data.typeconverter.NDArrayValidator(
                    np.float64, (None, 3))
        })
        self.reference_positions = reference_positions

    def _add(self, simulation):
        """Add the operation to a simulation."""
        super()._add(simulation)

    def _attach(self):
        integrator = self._simulation.operations.integrator
        if isinstance(self._simulation.device, hoomd.device.GPU):
            self._simulation.device._cpp_msg.warning(
                "Falling back on CPU. No GPU implementation available.\n")

        self._cpp_obj = _hoomd.UpdaterRemoveDrift(
            self._simulation.state._cpp_sys_def, self.reference_positions)
        super()._attach()
