# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

"""Compute properties of hard particle configurations."""

from __future__ import print_function

from hoomd import _hoomd
from hoomd.operation import Compute
from hoomd.hpmc import _hpmc
from hoomd.hpmc import integrate
from hoomd.data.parameterdicts import ParameterDict
from hoomd.logging import log
import hoomd


class FreeVolume(Compute):
    r"""Compute the free volume available to a test particle.

    Args:
        test_particle_type (str): Test particle type.
        num_samples (int): Number of samples to evaluate.

    `FreeVolume` computes the free volume in the simulation state available to a
    given test particle using Monte Carlo integration. It must be used in
    combination with an HPMC integrator, which defines the particle shape
    parameters.

    `FreeVolume` generates `num_samples` uniform random test particle placements
    (position and orientation) inside the box and counts the number of times
    these test placements overlap with the particles in the simulation state.
    It then computes the free volume with:

    .. math::
        V_\mathrm{free} = \left( \frac{n_\mathrm{samples} - n_\mathrm{overlaps}}
                               {n_\mathrm{samples}} \right) V_\mathrm{box}

    where :math:`V_\mathrm{free}` is the estimated free volume `free_volume`,
    :math:`n_\mathrm{samples}` is the number of samples `num_samples`,
    :math:`n_\mathrm{overlaps}` is the number of overlapping test placements,
    and :math:`V_\mathrm{box}` is the volume of the simulation box.

    Note:

        The test particle type must exist in the simulation state and its shape
        parameters must be set in the simulation's HPMC integrator. Particles
        with this type may or may not be present in the simulation state.

    Note:

        `FreeVolume` respects the ``interaction_matrix`` set in the HPMC
        integrator.

    Examples::

        fv = hoomd.hpmc.compute.FreeVolume(test_particle_type='B',
                                           num_samples=1000)


    Attributes:
        test_particle_type (str): Test particle type.

        num_samples (int): Number of samples to evaluate.

    """

    def __init__(self, test_particle_type, num_samples):
        # store metadata
        param_dict = ParameterDict(test_particle_type=str, num_samples=int)
        param_dict.update(
            dict(test_particle_type=test_particle_type,
                 num_samples=num_samples))
        self._param_dict.update(param_dict)

    def _attach(self):
        integrator = self._simulation.operations.integrator
        if not isinstance(integrator, integrate.HPMCIntegrator):
            raise RuntimeError("The integrator must be an HPMC integrator.")

        # Extract 'Shape' from '<hoomd.hpmc.integrate.Shape object>'
        integrator_name = integrator.__class__.__name__
        try:
            if isinstance(self._simulation.device, hoomd.device.CPU):
                cpp_cls = getattr(_hpmc, 'ComputeFreeVolume' + integrator_name)
            else:
                cpp_cls = getattr(_hpmc,
                                  'ComputeFreeVolume' + integrator_name + 'GPU')
        except AttributeError:
            raise RuntimeError("Unsupported integrator.")

        cl = _hoomd.CellList(self._simulation.state._cpp_sys_def)
        self._cpp_obj = cpp_cls(self._simulation.state._cpp_sys_def,
                                integrator._cpp_obj, cl)

        super()._attach()

    @log
    def free_volume(self):
        """Free volume available to the test particle."""
        if self._attached:
            self._cpp_obj.compute(self._simulation.timestep)
            return self._cpp_obj.free_volume
        else:
            return None
