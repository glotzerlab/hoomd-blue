# Copyright (c) 2009-2025 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Harmonic potential that restrains particles to a lattice."""

from hoomd.data.parameterdicts import ParameterDict
from hoomd.data.typeconverter import NDArrayValidator
from hoomd.hpmc import _hpmc
import hoomd
import numpy as np
import inspect

from .external import External


@hoomd.logging.modify_namespace(("hpmc", "external", "Harmonic"))
class Harmonic(External):
    r"""Restrain particle positions and orientations with harmonic springs.

    Args:
        reference_positions ((*N_particles*, 3) `numpy.ndarray` of `float`):
            the reference positions to which particles are restrained
            :math:`[\mathrm{length}]`.
        reference_orientations ((*N_particles*, 4) `numpy.ndarray` of `float`):
            the reference orientations to which particles are restrained
            :math:`[\mathrm{dimensionless}]`.
        k_translational (hoomd.variant.variant_like): translational spring
            constant :math:`[\mathrm{energy} \cdot \mathrm{length}^{-2}]`.
        k_rotational (hoomd.variant.variant_like): rotational spring constant
            :math:`[\mathrm{energy}]`.
        symmetries ((*N_symmetries*, 4) `numpy.ndarray` of `float`): the
            orientations that are equivalent through symmetry, i.e., the
            rotation quaternions that leave the particles unchanged. At a
            minimum, the identity quaternion (``[1, 0, 0, 0]``) must be included
            here :math:`[\mathrm{dimensionless}]`.

    `Harmonic` computes harmonic spring energies between the particle
    positions/orientations and given reference positions/orientations:

    .. math::

        \begin{split}
        U_{\mathrm{external},i} & = U_{\mathrm{translational},i} +
        U_{\mathrm{rotational},i} \\
        U_{\mathrm{translational},i} & = \frac{1}{2}
            k_{translational} \cdot (\vec{r}_i-\vec{r}_{0,i})^2 \\
        U_{\mathrm{rotational},i} & = \frac{1}{2}
            k_{rotational} \cdot \min_j \left[
            (\mathbf{q}_i-\mathbf{q}_{0,i} \cdot
             \mathbf{q}_{\mathrm{symmetry},j})^2 \right]
        \end{split}


    where :math:`k_{translational}` and :math:`k_{rotational}` correspond to the
    parameters ``k_translational`` and ``k_rotational``, respectively,
    :math:`\vec{r}_i` and :math:`\mathbf{q}_i` are the position and orientation
    of particle :math:`i`, the :math:`0` subscripts denote the given reference
    quantities, and :math:`\mathbf{q}_{\mathrm{symmetry}}` is the given set of
    symmetric orientations from the ``symmetries`` parameter.

    Note:
        `Harmonic` does not support execution on GPUs.

    {inherited}

    ----------

    **Members defined in** `Harmonic`:

    Attributes:
        k_translational (hoomd.variant.Variant): The translational spring
            constant :math:`[\mathrm{energy} \cdot \mathrm{length}^{-2}]`.
        k_rotational (hoomd.variant.Variant): The rotational spring constant
            :math:`[\mathrm{energy}]`.
        reference_positions ((*N_particles*, 3) `numpy.ndarray` of `float`):
            The reference positions to which particles are restrained
            :math:`[\mathrm{length}]`.
        reference_orientations ((*N_particles*, 4) `numpy.ndarray` of `float`):
            The reference orientations to which particles are restrained
            :math:`[\mathrm{dimensionless}]`.
        symmetries ((*N_symmetries*, 4) `numpy.ndarray` of `float`):
            The orientations that are equivalent through symmetry,
            i.e., the rotation quaternions that leave the particles unchanged
            :math:`[\mathrm{dimensionless}]`.
    """

    __doc__ = inspect.cleandoc(__doc__).replace(
        "{inherited}", inspect.cleandoc(External._doc_inherited)
    )

    def __init__(
        self,
        reference_positions,
        reference_orientations,
        k_translational,
        k_rotational,
        symmetries,
    ):
        param_dict = ParameterDict(
            reference_positions=NDArrayValidator(dtype=np.double, shape=(None, 3)),
            reference_orientations=NDArrayValidator(dtype=np.double, shape=(None, 4)),
            k_translational=hoomd.variant.Variant,
            k_rotational=hoomd.variant.Variant,
            symmetries=NDArrayValidator(dtype=np.double, shape=(None, 4)),
        )
        param_dict["k_translational"] = k_translational
        param_dict["k_rotational"] = k_rotational
        param_dict["reference_positions"] = reference_positions
        param_dict["reference_orientations"] = reference_orientations
        param_dict["symmetries"] = symmetries
        self._param_dict.update(param_dict)

    def _make_cpp_obj(self):
        cpp_sys_def = self._simulation.state._cpp_sys_def
        return _hpmc.PotentialExternalHarmonic(
            cpp_sys_def,
            self.reference_positions,
            self.k_translational,
            self.reference_orientations,
            self.k_rotational,
            self.symmetries,
        )
