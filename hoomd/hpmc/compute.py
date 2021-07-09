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
import numpy


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

    @log(requires_run=True)
    def free_volume(self):
        """Free volume available to the test particle \
        :math:`[\\mathrm{length}^{2}]` in 2D and \
        :math:`[\\mathrm{length}^{3}]` in 3D."""
        self._cpp_obj.compute(self._simulation.timestep)
        return self._cpp_obj.free_volume


class SDF(Compute):
    r"""Compute the scale distribution function.

    Args:
        xmax (float): Maximum *x* value at the right hand side of the rightmost
          bin (distance units).
        dx (float): Bin width (distance units).

    `SDF` computes a distribution function of scale parameters
    :math:`x`. For each particle, it finds the smallest scale factor :math:`1+x`
    that would cause the particle to touch one of its neighbors and records that
    in the histogram :math:`s(x)`. The histogram is discrete and :math:`s(x_i) =
    s[i]` where :math:`x_i = i \cdot dx + dx/2`.

    In an NVT simulation, the extrapolation of :math:`s(x)` to :math:`x = 0`,
    :math:`s(0+)` is related to the pressure.

    .. math::
        \frac{P}{kT} = \rho \left(1 + \frac{s(0+)}{2d} \right)

    where :math:`d` is the dimensionality of the system and :math:`\rho` is the
    number density.

    Suggested parameters for most systems (assuming particle diameters are ~1):

      * *xmax* = 0.02
      * *dx* = 1e-4

    In systems near densest packings, ``dx=1e-5`` may be needed along with
    either a smaller xmax or a smaller region to fit. A good rule of is to fit
    a region where ``numpy.sum(s[0:n]*dx)`` ~ 0.5.

    `betaP` performs the curve fit and computes :math:`\frac{P}{kT}` using a
    polynomial curve fit of degree 5.

    Warning:
        `SDF` does not compute correct pressures for simulations with
        concave particles or enthalpic interactions.

    Note:

        `SDF` runs on the CPU even in GPU simulations.

    Examples::

        sdf = hoomd.hpmc.compute.SDF(xmax=0.02, dx=1e-4)


    Attributes:
        xmax (float): Maximum *x* value at the right hand side of the rightmost
          bin.

        dx (float): Bin width.

    """

    def __init__(self, xmax, dx):
        # store metadata
        param_dict = ParameterDict(xmax=float(xmax), dx=float(dx))
        self._param_dict.update(param_dict)

    def _attach(self):
        integrator = self._simulation.operations.integrator
        if not isinstance(integrator, integrate.HPMCIntegrator):
            raise RuntimeError("The integrator must be an HPMC integrator.")

        # Extract 'Shape' from '<hoomd.hpmc.integrate.Shape object>'
        integrator_name = integrator.__class__.__name__

        cpp_cls = getattr(_hpmc, 'ComputeSDF' + integrator_name)

        self._cpp_obj = cpp_cls(self._simulation.state._cpp_sys_def,
                                integrator._cpp_obj, self.xmax, self.dx)

        super()._attach()

    @log(requires_run=True)
    def sdf(self):
        """Scale distribution function."""
        self._cpp_obj.compute(self._simulation.timestep)
        return self._cpp_obj.sdf

    @log(requires_run=True)
    def betaP(self):
        r"""Pressure in NVT simulations.

        Use a polynomial curve fit to estimate :math:`\frac{s(0+)}{2d}` and
        compute the pressure via:
        .. math::
            \frac{P}{kT} = \rho \left(1 + \frac{s(0+)}{2d} \right)
        """
        # get the values to fit
        n_fit = int(numpy.ceil(self.xmax / self.dx))
        sdf_fit = numpy.asarray(self.sdf[0:n_fit]) / sum(self.sdf[0:n_fit])
        # construct the x coordinates
        x_fit = numpy.arange(0, self.xmax, self.dx)
        x_fit += self.dx / 2
        # perform the fit and extrapolation
        p = numpy.polyfit(x_fit, sdf_fit, 5)

        box = self._simulation.state.box
        N = self._simulation.state.N_particles
        rho = N / box.volume
        return rho * (1 + numpy.polyval(p, 0.0) / (2 * box.dimensions))
