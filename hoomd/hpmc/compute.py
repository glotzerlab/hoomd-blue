# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Compute properties of hard particle configurations.

The HPMC compute classes analyze the system configuration and provide results
as loggable quantities for use with `hoomd.logging.Logger` or by direct access
via the Python API. `FreeVolume` computes the free volume available to small
particles, such as depletants, and `SDF` computes the pressure in system of
convex particles with a fixed box size.
"""

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
    given test particle shape using Monte Carlo integration. Use it in
    combination with `hoomd.hpmc.integrate.HPMCIntegrator`, which defines the
    particle shape parameters. Particles of ``test_particle_type`` may or may
    not be present in the simulation state.

    `FreeVolume` generates `num_samples` (:math:`n_\mathrm{samples}`) trial
    particle configurations with positions :math:`\vec{r}^t_j` uniformly
    distributed in the simulation box, and orientations :math:`\mathbf{q}^t_j`
    uniformly distributed among rotations matching the box dimensionality.
    `FreeVolume` counts the number of successful samples that do not overlap
    particles in the simulation state:

    .. math::

        n_\mathrm{success} = \sum_{j=1}^{n_\mathrm{samples}}
            \prod_{i=0}^{N_\mathrm{particles}-1}
            \prod_{\vec{A} \in B_\mathrm{images}}
            \left[
            \mathrm{overlap}\left(
            S_i(\mathbf{q}_i),
            S_t(\mathbf{q}^t_j, \vec{r}^t_j - (\vec{r}_i + \vec{A}))
            \right) = \emptyset
            \right]

    where :math:`\mathrm{overlap}` is the shape overlap function defined in
    `hoomd.hpmc.integrate`, :math:`S_i` is the shape of particle :math:`i`,
    :math:`S_t` is the shape of the test particle, :math:`\vec{A} = h\vec{a}_1 +
    k\vec{a}_2 + l\vec{a}_3` is a vector that translates by periodic box images,
    the set of box images includes all image vectors necessary to find overlaps
    between particles in the primary image with particles in periodic images,
    and the square brackets denote the Iverson bracket.

    The free volume :math:`V_\mathrm{free}` is given by:

    .. math::
        V_\mathrm{free} = \frac{n_\mathrm{success}}
                               {n_\mathrm{samples}} V_\mathrm{box}

    where :math:`V_\mathrm{box}` is the volume of the simulation box (or area in
    2D).

    Note:

        `FreeVolume` respects the HPMC integrator's ``interaction_matrix``.

    .. rubric:: Mixed precision

    `FreeVolume` uses reduced precision floating point arithmetic when checking
    for particle overlaps in the local particle reference frame.

    .. rubric:: Box images

    On CPU devices, `FreeVolume` does not apply the minimum image convention. It
    supports small boxes where particles may overlap with non-primary images of
    other particles, including self overlap. On GPU devices, `FreeVolume`
    applies the minimum image convention.

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

    def _attach_hook(self):
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

    @log(requires_run=True)
    def free_volume(self):
        """Free volume available to the test particle \
        :math:`[\\mathrm{length}^{2}]` in 2D and \
        :math:`[\\mathrm{length}^{3}]` in 3D."""
        self._cpp_obj.compute(self._simulation.timestep)
        return self._cpp_obj.free_volume


class SDF(Compute):
    r"""Compute the scale distribution function via volume perturbations.

    Args:
        xmax (float): Maximum *x* value at the right hand side of the rightmost
            bin :math:`[\mathrm{length}]`.
        dx (float): Bin width :math:`[\mathrm{length}]`.

    `SDF` computes the probability distributions :math:`s_{\mathrm{comp}}(x)`
    and :math:`s_{\mathrm{exp}}(x)` of particles overlapping as a function of
    separation for compressive and expansive perturbations, respectively. It
    estimates :math:`s_{\mathrm{comp}}(x)` and :math:`s_{\mathrm{exp}}(x)`
    numerically by computing histograms with :math:`\lfloor x_\mathrm{max}/
    \delta x \rfloor` bins of width `dx` (:math:`\delta x`).

    See Also:
         `Anderson 2016 <https://dx.doi.org/10.1016/j.cpc.2016.02.024>`_
         describes the theory relating `SDF` to the system pressure and its
         implementation in HOOMD-blue. `Eppenga and Frenkel 1984
         <https://doi.org/10.1080/00268978400101951>`_ present a derivation
         relating the scale distribution function to the system pressure for
         hard, convex particles. `Allen 2006
         <https://doi.org/10.1063/1.2202352>`_ describes the theory for
         calculating the pressure in systems with discontinuous potential energy
         functions. The expansive perturbations are based on theory described in
         `de Miguel and Jackson <https://doi.org/10.1063/1.2363381>`_.

    .. rubric:: Implementation

    `SDF` constructs two histograms, one for compressive volume perturbations
    and one for expansive volume perturbations. These histograms represent
    :math:`s_{\mathrm{comp}}(x)` and :math:`s_{\mathrm{exp}}(x)`. The
    following discussion applies to compressive volume perturbations and the
    computation of :math:`s_{\mathrm{comp}}(x)`; the computation of
    :math:`s_{\mathrm{exp}}(x)` proceeds similarly as noted throughout the
    description.

    For each pair of particles :math:`i` and :math:`j` `SDF` scales the particle
    separation vector by the factor :math:`(1 \pm x)` (:math:`+` for expansive
    perturbations, :math:`-` for compressive perturbations) and finds the
    smallest positive value of :math:`x` leading to either an overlap of the
    particle shapes (a "hard overlap") or a discontinuous change in the pair
    energy :math:`U_{\mathrm{pair},ij}` (a "soft overlap"). For compressive
    perturbations:

    .. math::

        x_{ij}(\vec{A}) = \min \{ & x \in \mathbb{R}_{> 0} : \\
           & \mathrm{overlap}\left(
                S_i(\mathbf{q}_i),
                S_j(\mathbf{q}_j, (1-x)(\vec{r}_j - (\vec{r}_i + \vec{A})))
            \right) \ne \emptyset
            \\
            &\lor \\
            & U_{\mathrm{pair},ij}((1-x)(\vec{r}_j - (\vec{r}_i + \vec{A})),
                                 \mathbf{q}_i,
                                 \mathbf{q}_j)
                \ne
            U_{\mathrm{pair},ij}(\vec{r}_j - (\vec{r}_i + \vec{A}),
                                 \mathbf{q}_i,
                                 \mathbf{q}_j) \\
            \} &

    where :math:`\mathrm{overlap}` is the shape overlap function defined in
    `hoomd.hpmc.integrate`, :math:`S_i` is the shape of particle :math:`i`, and
    :math:`\vec{A} = h\vec{a}_1 + k\vec{a}_2 + l\vec{a}_3` is a vector that
    translates by periodic box images. For expansive perturbations,

    .. math::

        x_{ij}(\vec{A}) = \max \{ & x \in \mathbb{R}_{< 0} : \\
           & \mathrm{overlap}\left(
                S_i(\mathbf{q}_i),
                S_j(\mathbf{q}_j, (1+x)(\vec{r}_j - (\vec{r}_i + \vec{A})))
            \right) \ne \emptyset
            \\
            &\lor \\
            & U_{\mathrm{pair},ij}((1+x)(\vec{r}_j - (\vec{r}_i + \vec{A})),
                                 \mathbf{q}_i,
                                 \mathbf{q}_j)
                \ne
            U_{\mathrm{pair},ij}(\vec{r}_j - (\vec{r}_i + \vec{A}),
                                 \mathbf{q}_i,
                                 \mathbf{q}_j) \\
            \} &


    For particle :math:`i`, `SDF` finds the the minimum (maximum for expansive
    perturbations) value :math:`x_i`. For compressive perturbations:

    .. math::

        x_i = \min \{ x_{ij} : \vec{A} \in B_\mathrm{images},
                     j \in [0,N_\mathrm{particles}) \}

    where the set of box images includes all image vectors necessary to find
    overlaps between particles in the primary image with particles in periodic
    images. For expansive perturbations:

    .. math::

        x_i = \max \{ x_{ij} : \vec{A} \in B_\mathrm{images},
                     j \in [0,N_\mathrm{particles}) \}

    `SDF` adds a single count to each histogram for each particle :math:`i`,
    weighted by a factor that is a function of the change in energy upon
    overlap. For compressive perturbations:

    .. math::

        s_{\mathrm{comp}}(x + \delta x/2) = \frac{1}{N_\mathrm{particles}
            \cdot \delta x} \sum_{i=0}^{N_\mathrm{particles}-1}
            [x \le x_i < x + \delta x] \cdot (1 - \exp(-\beta \Delta U_{i}))

    where :math:`\Delta U_{i}` is the change in energy associated with the first
    overlap involving particle :math:`i` (:math:`\infty` for hard overlaps), the
    square brackets denote the Iverson bracket, and :math:`s_{\mathrm{comp}}(x +
    \delta x/2)` is evaluated for :math:`\{ x \in \mathbb{R}, 0 \le x <
    x_\mathrm{max}, x = k \cdot \delta x, k \in \mathbb{Z}^* \}` for compressive
    perturbations. For expansive perturbations,

    .. math::

        s_{\mathrm{exp}}(x - \delta x/2) = \frac{1}{N_\mathrm{particles}
            \cdot \delta x}
            \sum_{i=0}^{N_\mathrm{particles}-1}
            [x - \delta x \le x_i < x] \cdot (1 - \exp(-\beta \Delta U_{i}))

    where :math:`s_{\mathrm{exp}}(x - \delta x/2)` is evaluated for :math:`\{ x
    \in \mathbb{R}, -|x_\mathrm{max}| < x \le 0, x = (k - \lfloor
    x_\mathrm{max} / \delta x \rfloor + 1) \cdot \delta x, k \in
    \mathbb{Z}^* \}`.

    .. rubric:: Pressure

    The pressure :math:`P` is related to the one-sided limits
    :math:`s_{\mathrm{comp}}(0+)` and :math:`s_{\mathrm{exp}}(0-)`, computed by
    fitting and extrapolating :math:`s_{\mathrm{comp}}` and
    :math:`s_{\mathrm{exp}}` to :math:`x = 0`.

    .. math::
        \beta P = \rho \left(
            1 + \frac{s_{\mathrm{comp}}(0+) - s_{\mathrm{exp}}(0-)}{2d} \right)

    where :math:`d` is the dimensionality of the system, :math:`\rho` is the
    number density, and :math:`\beta = \frac{1}{kT}`. This measurement of the
    pressure is inherently noisy due to the nature of the sampling. Average
    `betaP` over many timesteps to obtain accurate results.

    Assuming particle diameters are ~1, these parameter values typically
    achieve good results:

    * ``xmax = 0.02``
    * ``dx = 1e-4``

    In systems near densest packings, ``dx=1e-5`` may be needed along with
    smaller ``xmax``. Check that :math:`\sum_k s_\mathrm{comp}(x_k) \cdot dx
    \approx 0.5`.

    Important:
        `SDF` samples pair configurations at discrete separations. Therefore,
        the computed pressure is correct only for potentials with constant
        values and step discontinuities.

    Note:
        `SDF` always runs on the CPU.

    .. rubric:: Mixed precision

    `SDF` uses reduced precision floating point arithmetic when checking
    for particle overlaps in the local particle reference frame.

    .. rubric:: Box images

    `SDF` does not apply the minimum image convention. It supports small boxes
    where particles may overlap with non-primary images of other particles,
    including self overlap.

    Attributes:
        xmax (float): Maximum *x* value at the right hand side of the rightmost
            bin :math:`[\mathrm{length}]`.

        dx (float): Bin width :math:`[\mathrm{length}]`.
    """

    def __init__(self, xmax, dx):
        # store metadata
        param_dict = ParameterDict(
            xmax=float(xmax),
            dx=float(dx),
        )
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        integrator = self._simulation.operations.integrator
        if not isinstance(integrator, integrate.HPMCIntegrator):
            raise RuntimeError("The integrator must be an HPMC integrator.")

        # Extract 'Shape' from '<hoomd.hpmc.integrate.Shape object>'
        integrator_name = integrator.__class__.__name__

        cpp_cls = getattr(_hpmc, 'ComputeSDF' + integrator_name)

        self._cpp_obj = cpp_cls(
            self._simulation.state._cpp_sys_def,
            integrator._cpp_obj,
            self.xmax,
            self.dx,
        )

    @log(category='sequence', requires_run=True)
    def sdf_compression(self):
        """(*N_bins*,) `numpy.ndarray` of `float`): :math:`s_\\mathrm{comp}[k]`\
        - The scale distribution function for compression moves \
        :math:`[\\mathrm{probability\\ density}]`.

        See Also:
            `x_compression` defines the bin center locations.

        Attention:
            In MPI parallel execution, the array is available on rank 0 only.
            `sdf_compression` is `None` on ranks >= 1.
        """
        self._cpp_obj.compute(self._simulation.timestep)
        return self._cpp_obj.sdf_compression

    @log(category='sequence', requires_run=True)
    def sdf_expansion(self):
        """(*N_bins*,) `numpy.ndarray` of `float`): :math:`s_\\mathrm{exp}[k]` \
        - The scale distribution function for the expansion moves \
        :math:`[\\mathrm{probability\\ density}]`.

        See Also:
            `x_expansion`  defines the bin center locations..

        Attention:
            In MPI parallel execution, the array is available on rank 0 only.
            `sdf_expansion` is `None` on ranks >= 1.
        """
        self._cpp_obj.compute(self._simulation.timestep)
        cpp_expansion = self._cpp_obj.sdf_expansion

        if cpp_expansion is not None:
            return numpy.array(cpp_expansion[::-1])
        else:
            return None

    @log(category='sequence', requires_run=True)
    def x_compression(self):
        """(*N_bins*,) `numpy.ndarray` of `float`): The x \
        values at the center of each bin corresponding to the scale \
        distribution function for the compressive perturbations \
        :math:`[\\mathrm{length}]`."""
        # Ensure that num_bins is up to date.
        self._cpp_obj.compute(self._simulation.timestep)

        x = numpy.arange(0, self._cpp_obj.num_bins, 1) * self.dx + self.dx / 2
        return x

    @log(category='sequence', requires_run=True)
    def x_expansion(self):
        """(*N_bins*,) `numpy.ndarray` of `float`): The x \
        values at the center of each bin corresponding to the scale \
        distribution function for the expansion moves \
        :math:`[\\mathrm{length}]`."""
        # Ensure that num_bins is up to date.
        self._cpp_obj.compute(self._simulation.timestep)

        x = numpy.arange(-self._cpp_obj.num_bins, 0, 1) * self.dx + self.dx / 2
        return x

    @log(requires_run=True)
    def betaP(self):  # noqa: N802 - allow function name
        """float: Beta times pressure in NVT simulations \
        :math:`\\left[ \\mathrm{length}^{-d} \\right]`.

        Uses a polynomial curve fit of degree 5 to estimate
        :math:`s_\\mathrm{comp}(0+)` (and :math:`s_\\mathrm{exp}(0-)` if
        required) and computes the pressure via:

        .. math::
            \\beta P = \\rho \\left(1 + \\frac{s_\\mathrm{comp}(0+)}{2d} +
            \\frac{s_\\mathrm{exp}(0-)}{2d} \
            \\right)

        where :math:`d` is the dimensionality of the system, :math:`\\rho` is
        the number density, and :math:`\\beta = \\frac{1}{kT}`.

        Attention:
            In MPI parallel execution, `betaP` is available on rank 0 only.
            `betaP` is `None` on ranks >= 1.
        """
        sdf_fit_compression = self.sdf_compression
        x_fit_compression = self.x_compression

        sdf_fit_expansion = self.sdf_expansion
        x_fit_expansion = self.x_expansion

        if self.sdf_compression is not None and self.sdf_expansion is not None:
            compression_contribution = 0
            expansion_contribution = 0
            box = self._simulation.state.box
            N = self._simulation.state.N_particles
            rho = N / box.volume

            # compressive contribution
            # perform the fit and extrapolation
            p = numpy.polyfit(x_fit_compression, sdf_fit_compression, 5)
            p0_compression = numpy.polyval(p, 0.0)
            compression_contribution = rho * p0_compression / (2
                                                               * box.dimensions)

            # expansive contribution
            # perform the fit and extrapolation
            p = numpy.polyfit(x_fit_expansion, sdf_fit_expansion, 5)
            p0_expansion = numpy.polyval(p, 0.0)
            expansion_contribution = -rho * p0_expansion / (2 * box.dimensions)

            return rho + compression_contribution + expansion_contribution
        else:
            return None
