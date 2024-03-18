# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Angular-step pair potential.

.. invisible-code-block: python

    simulation = hoomd.util.make_example_simulation()
    sphere = hoomd.hpmc.integrate.Sphere()
    sphere.shape['A'] = dict(diameter=0.0)
    simulation.operations.integrator = sphere

    square_well = hoomd.hpmc.pair.Step()
    square_well.params[('A', 'A')] = dict(epsilon=[-1], r=[2.0])
"""

import hoomd
from hoomd.data.parameterdicts import TypeParameterDict
from hoomd.data.typeparam import TypeParameter
from hoomd.data.typeconverter import OnlyIf, to_type_converter

from .pair import Pair


@hoomd.logging.modify_namespace(('hpmc', 'pair', 'AngularStep'))
class AngularStep(Pair):
    r"""Angular-step pair potential (HPMC).

    Args:
        isotropic_potential (hoomd.hpmc.pair.Pair): the isotropic part of the
            interaction between the particles.

    `AngularStep` computes the the given isotropic potential multiplied by a
    step function that is dependent on the relative orientation between any
    two patches:

    .. math::
        U(\vec{r}_{ij}, \mathbf{q}_i, \mathbf{q}_j) =
        U_\mathrm{isotropic}(\vec{r}_{ij}) \cdot
        \max \left(1,
        \sum_{m=1}^{N_{\mathrm{patches},i}} \sum_{m=1}^{N_{\mathrm{patches},j}}
        f(\mathbf{q}_i \vec{d}_{n,i} \mathbf{q}_i^*,
          \mathbf{q}_j \vec{d}_{m,j} \mathbf{q}_j^*,
          \delta_{n,i},
          \delta_{m,j}) \right)

    where :math:`U_\mathrm{isotropic}` is the isotropic potential.
    For a given particle :math:`i`, :math:`N_{\mathrm{patches},i}` is the
    number of patches, :math:`\vec{d}_{n,i}` is the n-th ``director``, and
    :math:`\delta_{n,i}` is the n-th ``delta``.
    :math:`f_{ij}(\vec{a}, \vec{b}, \delta_a, \delta_b)` is an orientational
    masking function given by:

    .. math::
        f(\vec{a}, \vec{b}, \delta_a, \delta_b) =
        \begin{cases}
        1 & \hat{a} \cdot \hat{r}_{ij} \ge \cos \delta_{a} \land
        \hat{b} \cdot \hat{r}_{ji} \ge \cos \delta_{b} \\
        0 & \text{otherwise} \\
        \end{cases}

    One example of this form of potential is the Kern-Frenkel model that is
    composed of a square well potential and an orientational masking function.

    .. rubric:: Example

    .. code-block:: python

        angular_step = hoomd.hpmc.pair.AngularStep(
                       isotropic_potential=square_well)
        angular_step.mask['A'] = dict(directors=[(1.0, 0, 0)], deltas=[0.1])
        simulation.operations.integrator.pair_potentials = [angular_step]

    Set the patch directors :math:`\vec{d}_m` and delta :math:`\delta_m`  values
    for each particle type. Patch directors are the directional unit vectors
    that represent the patch locations on a particle, and deltas are the half
    opening angles of the patch in radian.

    .. py:attribute:: mask

        The mask definition.

        The mask describes the distribution of patches on the particle's
        surface and the masking function determines the interaction scale
        factor as a function of two interacting particle's masks.

        The dictionary has the following keys:

        - ``directors`` (`list` [`tuple` [`float`, `float`, `float`]]): List of
          directional vectors of the patches on a particle.
        - ``deltas`` (`list` [`float`]): List of delta values (the half opening
          angle of the patch in radian) of the patches.
    """
    _cpp_class_name = "PairPotentialAngularStep"

    def __init__(self, isotropic_potential):
        mask = TypeParameter(
            'mask', 'particle_types',
            TypeParameterDict(OnlyIf(to_type_converter(
                dict(directors=[(float,) * 3], deltas=[float])),
                                     allow_none=True),
                              len_keys=1))
        self._add_typeparam(mask)

        if not isinstance(isotropic_potential, hoomd.hpmc.pair.Pair):
            raise TypeError(
                "isotropic_potential must be subclass of hoomd.hpmc.pair.Pair")
        self._isotropic_potential = isotropic_potential

    @property
    def isotropic_potential(self):
        """Get the isotropic part of the interactions between patchy particles.

        This property returns the isotropic component of pairwise interaction
        potentials for patchy particle systems.

        .. rubric:: Example

        .. code-block:: python

            angular_step.isotropic_potential
        """
        return self._isotropic_potential

    def _make_cpp_obj(self):
        cpp_sys_def = self._simulation.state._cpp_sys_def
        cls = getattr(hoomd.hpmc._hpmc, self._cpp_class_name)
        return cls(cpp_sys_def, self.isotropic_potential._cpp_obj)

    def _attach_hook(self):
        self.isotropic_potential._attach(self._simulation)
        super()._attach_hook()
        self.isotropic_potential._cpp_obj.setParent(self._cpp_obj)

    def _detach_hook(self):
        if self.isotropic_potential is not None:
            self.isotropic_potential._detach()

        super()._detach_hook()
