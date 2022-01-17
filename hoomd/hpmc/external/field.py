# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Apply external fields to HPMC simulations."""

from hoomd.data.parameterdicts import ParameterDict
from hoomd.data.typeconverter import NDArrayValidator
from hoomd.hpmc import _hpmc
from hoomd.hpmc import integrate
from hoomd.logging import log
from hoomd.operation import Compute
from hoomd.operation import _HOOMDBaseObject
import hoomd
import numpy as np


# \internal
# \brief Base class for external fields
#
# An external in hoomd reflects an ExternalField in c++. It is responsible for
# all high-level management that happens behind the scenes for hoomd writers. 1)
# The instance of the c++ external itself is tracked optionally passed to the
# hpmc integrator. While external fields are Compute types and are added to the
# System they will not be enforced unless they are added to the integrator.
# Only one external field can be held by the integrator so if multiple fields
# are required use the external_field_composite class to manage them.
class _External(Compute):
    # \internal
    # \brief Initialize an empty external.
    #
    # \post nothing is done here yet.
    def __init__(self):
        # _compute.__init__(self)
        self.cpp_compute = None
        # nothing else to do.


class ExternalField(_HOOMDBaseObject):
    """Base class external field.

    Provides common methods for all external field subclasses.

    Note:
        Users should use the subclasses and not instantiate `ExternalField`
        directly.

    """


class Harmonic(ExternalField):
    r"""Restrain particle positions and orientations with harmonic springs.

    Args:
        reference_positions ((*N_particles*, 3) `numpy.ndarray` of
            `float`): the reference positions, to which particles are restrained
            :math:`[\mathrm{length}]`.
        reference_orientations ((*N_particles*, 4) `numpy.ndarray` of
            `float`): the reference orientations, to which particles are
            restrained :math:`[\mathrm{dimensionless}]`.
        k_translational (`float`): translational spring constant
            :math:`[\mathrm{energy} \cdot \mathrm{length}^{-2}]`.
        k_rotational (`float`): rotational spring constant
            :math:`[\mathrm{energy}]`.
        symmetries ((*N_sym*, 4) `numpy.ndarray` of
            `float`): the orientations that are equivalent through symmetry,
            i.e., the rotation quaternions that leave the particles unchanged.
            At a minimum, the identity quaternion (``[1, 0, 0, 0]``) must be
            included here :math:`[\mathrm{dimensionless}]`.

    :py:class:`Harmonic` specifies that harmonic springs are used to
    restrain the position and orientation of every particle:

    .. math::

        V_{i}(r_i)  = \frac{1}{2} k_{trans} \cdot (r_i-r_{0,i})^2 \\
        V_{i}(q_i)  = \frac{1}{2} k_{rot} \cdot \min_j \left[ (q_i-q_{0,i} \cdot
            q_{\mathrm{symmetry},j})^2 \right]

    where :math:`k_{trans}` and :math:`k_{rot}` correspond to the arguments
    ``k_translational`` and ``k_rotational``, respectively, :math:`r_i` and
    :math:`q_i` are the position and orientation of particle :math:`i`, the
    :math:`0` subscripts denote the given reference quantities, and
    :math:`q_{\mathrm{symmetry}}` is the given set of symmetric orientations
    from the ``symmetries`` argument.

    .. py:attribute:: reference_positions

        The reference positions, to which particles are restrained
        :math:`[\mathrm{length}]`.

         **Type:** np.ndarray, shape=(*N_particles*, 3), dtype=`float`

    .. py:attribute:: reference_orientations

        The reference orientations, to which particles are restrained
        :math:`[\mathrm{dimensionless}]`.

         **Type:** np.ndarray, shape=(*N_particles*, 4), dtype=`float`

    Attributes:
        k_translational (`float`): The translational spring constant
            :math:`[\mathrm{energy} \cdot \mathrm{length}^{-2}]`.
        k_rotational (`float`): The rotational spring constant
            :math:`[\mathrm{energy}]`.
        symmetries (np.ndarray, shape=(*N_particles*, 4), dtype=`float`)
            The orientations that are equivalent through symmetry,
            i.e., the rotation quaternions that leave the particles unchanged
            :math:`[\mathrm{dimensionless}]`.

    """

    def __init__(self, reference_positions, reference_orientations,
                 k_translational, k_rotational, symmetries):
        param_dict = ParameterDict(
            reference_positions=NDArrayValidator(dtype=np.double,
                                                 shape=(None, 3)),
            reference_orientations=NDArrayValidator(dtype=np.double,
                                                    shape=(None, 4)),
            k_translational=float,
            k_rotational=float,
            symmetries=NDArrayValidator(dtype=np.double, shape=(None, 4)),
        )
        param_dict['k_translational'] = k_translational
        param_dict['k_rotational'] = k_rotational
        param_dict['reference_positions'] = reference_positions
        param_dict['reference_orientations'] = reference_orientations
        param_dict['symmetries'] = symmetries
        self._param_dict.update(param_dict)

    def _attach(self):
        cls = None

        integrator = self._simulation.operations.integrator
        if not isinstance(integrator, integrate.HPMCIntegrator):
            raise RuntimeError("The integrator must be a HPMC integrator.")

        if not integrator._attached:
            raise RuntimeError("Integrator is not attached yet.")

        # get device and system definition from simulation
        device = self._simulation.device
        cpp_sys_def = self._simulation.state._cpp_sys_def

        if not isinstance(device, hoomd.device.GPU):
            if isinstance(integrator, integrate.Sphere):
                cls = _hpmc.ExternalFieldHarmonicSphere
            elif isinstance(integrator, integrate.ConvexPolygon):
                cls = _hpmc.ExternalFieldHarmonicConvexPolygon
            elif isinstance(integrator, integrate.SimplePolygon):
                cls = _hpmc.ExternalFieldHarmonicSimplePolygon
            elif isinstance(integrator, integrate.ConvexPolyhedron):
                cls = _hpmc.ExternalFieldHarmonicConvexPolyhedron
            elif isinstance(integrator, integrate.ConvexSpheropolyhedron):
                cls = _hpmc.ExternalFieldHarmonicSpheropolyhedron
            elif isinstance(integrator, integrate.Ellipsoid):
                cls = _hpmc.ExternalFieldHarmonicEllipsoid
            elif isinstance(integrator, integrate.ConvexSpheropolygon):
                cls = _hpmc.ExternalFieldHarmonicSpheropolygon
            elif isinstance(integrator, integrate.FacetedEllipsoid):
                cls = _hpmc.ExternalFieldHarmonicFacetedEllipsoid
            elif isinstance(integrator, integrate.Polyhedron):
                cls = _hpmc.ExternalFieldHarmonicPolyhedron
            elif isinstance(integrator, integrate.Sphinx):
                cls = _hpmc.ExternalFieldHarmonicSphinx
            elif isinstance(integrator, integrate.SphereUnion):
                cls = _hpmc.ExternalFieldHarmonicSphereUnion
            elif isinstance(integrator, integrate.FacetedEllipsoidUnion):
                cls = _hpmc.ExternalFieldHarmonicFacetedEllipsoidUnion
            elif isinstance(integrator, integrate.ConvexSpheropolyhedronUnion):
                cls = _hpmc.ExternalFieldHarmonicConvexPolyhedronUnion
            else:
                msg = 'Error initializing hoomd.hpmc.field.Harmonic: '
                msg += 'unsupported integrator'
                raise RuntimeError(msg)
        else:
            msg = 'Error initializing hoomd.hpmc.field.Harmonic: '
            msg += 'GPU not supported.'
            raise RuntimeError(msg)

        self._cpp_obj = cls(
            cpp_sys_def,
            self.reference_positions,
            self.k_translational,
            self.reference_orientations,
            self.k_rotational,
            self.symmetries,
        )
        super()._attach()

    @log(requires_run=True)
    def energy(self):
        """tuple(float): The energy of the harmonic field \
                :math:`[\\mathrm{energy}]`."""
        timestep = self._simulation.timestep
        return sum(self._cpp_obj.getEnergies(timestep))

    @log(requires_run=True)
    def energy_translational(self):
        """float: The energy associated with positional fluctuations \
            :math:`[\\mathrm{energy}]`."""
        timestep = self._simulation.timestep
        return self._cpp_obj.getEnergies(timestep)[0]

    @log(requires_run=True)
    def energy_rotational(self):
        """float: The energy associated with rotational fluctuations \
            :math:`[\\mathrm{energy}]`.

        This function will always return 0 for a non-orientable shape.

        """
        timestep = self._simulation.timestep
        return self._cpp_obj.getEnergies(timestep)[1]
