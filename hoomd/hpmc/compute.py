# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

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

    .. rubric:: Mixed precision

    `FreeVolume` uses reduced precision floating point arithmetic when checking
    for particle overlaps in the local particle reference frame.

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
            bin :math:`[\mathrm{length}]`.
        dx (float): Bin width :math:`[\mathrm{length}]`.

    `SDF` computes the proability distribution of particles overlapping
    :math:`s(x)` as a function of separation.

    .. rubric:: Implementation

    For each pair of particles :math:`i` and :math:`j` it scales the particle
    separation vector by :math:`1-x`, and finds the smallest value leading to
    particle shape overlap:

    .. math::

        x_{ij} = \min \{ x \in \mathbb{R}_{> 0} : \mathrm{overlap}\left(
            (1-x)(\vec{r_j} - \vec{r_i}),
            S_i(\mathbf{q}_i),
            S_j(\mathbf{q}_j) \ne \emptyset \right) \}

    where :math:`\mathrm{overlap}` is the shape overlap function defined in
    `hpmc.integrate.HPMCIntegrator` and :math:`S_i` is the shape of particle
    :math:`i`.

    `SDF` adds a single count to the histogram for each particle *i* at the
    minimum value:

    .. math::

        x_i = \min \{ x_{ij} \}

    .. rubric:: Pressure

    The extrapolation of :math:`s(x)` to :math:`x = 0`, :math:`s(0+)` is related
    to the pressure

    .. math::
        \beta P = \rho \left(1 + \frac{s(0+)}{2d} \right)

    where :math:`d` is the dimensionality of the system, :math:`\rho` is the
    number density, and :math:`\beta = \frac{1}{kT}`. This measurement of the
    pressure is inherently noisy due to the nature of the sampling. Average
    `betaP` over many timesteps to obtain accurate results.

    Assuming particle diameters are ~1, these paramater values typically
    achieve good results:

      * ``xmax = 0.02``
      * ``dx = 1e-4``

    In systems near densest packings, ``dx=1e-5`` may be needed along with
    smaller ``xmax``. Check that :math:`\sum_k s(x_k) \cdot dx \approx 0.5`.

    Warning:
        `SDF` does not compute correct pressures for simulations with
        concave particles or enthalpic interactions.

    Note:
        `SDF` runs on the CPU even in GPU simulations.

    .. rubric:: Mixed precision

    `SDF` uses reduced precision floating point arithmetic when checking
    for particle overlaps in the local particle reference frame.

    .. rubric:: Box images

    `SDF` does not apply the minimum image convention. It supports small boxes
    where particles can potentially with particles outside the primary box
    image.

    Attributes:
        xmax (float): Maximum *x* value at the right hand side of the rightmost
            bin :math:`[\mathrm{length}]`.

        dx (float): Bin width :math:`[\mathrm{length}]`.
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

    @log(category='sequence', requires_run=True)
    def sdf(self):
        """(*N_bins*,) `numpy.ndarray` of `float`): :math:`s[k]` - The scale \
        distribution function :math:`[\\mathrm{probability\\ density}]`.

        The :math:`x` value corresponding to bin :math:`k` is:
        :math:`x = k \\cdot dx + dx/2`.

        Attention:
            In MPI parallel execution, the array is available on rank 0 only.
            `sdf` is `None` on ranks >= 1.
        """
        self._cpp_obj.compute(self._simulation.timestep)
        return self._cpp_obj.sdf

    @log(requires_run=True)
    def betaP(self):  # noqa: N802 - allow function name
        """float: Beta times pressure in NVT simulations \
        :math:`\\left[ \\mathrm{length}^{-d} \\right]`.

        Use a polynomial curve fit of degree 5 to estimate
        :math:`s(0+)` and compute the pressure via:

        .. math::
            \\beta P = \\rho \\left(1 + \\frac{s(0+)}{2d} \\right)

        where :math:`d` is the dimensionality of the system, :math:`\\rho` is
        the number density, and :math:`\\beta = \\frac{1}{kT}`.

        Attention:
            In MPI parallel execution, `betaP` is available on rank 0 only.
            `betaP` is `None` on ranks >= 1.
        """
        if not numpy.isnan(self.sdf).all():
            # get the values to fit
            n_fit = int(numpy.ceil(self.xmax / self.dx))
            sdf_fit = self.sdf[0:n_fit]
            # construct the x coordinates
            x_fit = numpy.arange(0, self.xmax, self.dx)
            x_fit += self.dx / 2
            # perform the fit and extrapolation
            p = numpy.polyfit(x_fit, sdf_fit, 5)

            box = self._simulation.state.box
            N = self._simulation.state.N_particles
            rho = N / box.volume
            return rho * (1 + numpy.polyval(p, 0.0) / (2 * box.dimensions))
        else:
            return None
