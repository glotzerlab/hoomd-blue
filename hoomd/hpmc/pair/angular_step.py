# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Angular-step pair potential.

.. invisible-code-block: python

    simulation = hoomd.util.make_example_simulation()
    sphere = hoomd.hpmc.integrate.Sphere()
    sphere.shape['A'] = dict(diameter=0.0)
    simulation.operations.integrator = sphere

    lennard_jones =  hoomd.hpmc.pair.LennardJones()
    lennard_jones.params[('A', 'A')] = dict(epsilon=1, sigma=1, r_cut=4.0)
"""

import hoomd
from hoomd.data.parameterdicts import TypeParameterDict
from hoomd.data.typeparam import TypeParameter
from hoomd.data.typeconverter import OnlyIf, to_type_converter

from .pair import Pair


@hoomd.logging.modify_namespace(('hpmc', 'pair', 'AngularStep'))
class AngularStep(Pair):
    """Angular-step pair potential (HPMC).

    Args:
        isotropic_potential (hoomd.hpmc.pair.Pair): the pair potential that act
            as the isotropic part of the interaction between patchy particles.

    `AngularStep` computes the angular step potential, which is a composite
    potential consist of an isotropic potential and a step function that is
    dependent on the relative orientation between the patches. One example of
    this form of potential is the Kern-Frenkel model that is composed of
    a square well potential and an orientational masking function.

    .. math::
        U(\vec{r}_{ij}, \mathbf{\Omega}_i, \mathbf{\Omega}_j)) =
        \sum_{m=1}^{N_{\mathrm{patches},i}} \sum_{m=1}^{N_{\mathrm{patches},j}}
        U_\mathrm{isotropic}(\vec{r}_{ij}) \cdot 
        f(\mathbf{\Omega}_i, \mathbf{\Omega}_j)

    where :math:`N_{\mathrm{patches},i}` is the number of patches on the
    :math:`i` particle and :math:`U_\mathrm{isotropic}` is the isotropic 
    potential. :math:`f(\mathbf{\Omega}_i, \mathbf{\Omega}_j)` is an 
    orientational masking function given by: 

        f(\mathbf{\Omega}_i, \mathbf{\Omega}_j) = \left\{ \begin{array}{ll} 
        1 \quad \hat{e_{i}} \cdot \hat{r_{ij}} > cos\delta_{i} \quad and \quad 
        \hat{e_{i}} \cdot \hat{r_{ji}} > cos\delta_{j} \\ 
        0 \quad \text{otherwise} \end{array} \right.

    where :math:`\hat{e_{i}}` and :math:`\hat{e_{j}}` are the unit vectors 
    pointing from the particle center to the patches. :math:`\hat{r_{ij}}` 
    is the unit vector pointing from particle :math:`i` to particle :math:`j`.
    :math:`cos\delta_{i}` and :math:`cos\delta_{j}` are the half opening angles
    of the patches on particles :math:`i` and :math:`j` accordingly. 
 
    
    .. rubric:: Example

    .. code-block:: python

        angular_step = hoomd.hpmc.pair.AngularStep(
                       isotropic_potential=lennard_jones)
        angular_step.patch['A'] = dict(directors=[(1.0, 0, 0)],
                        deltas=[0.1])
        simulation.operations.integrator.pair_potentials = [angular_step]

    Set the patch directors and delta values for each particle
    type. Patch directors are the directional unit vectors that represent
    the patch locations on a particle, and deltas are the half opening angles of
    the patch in radian.

    .. py:attribute:: patch

        The patch definition.

        Define the patch director and delta of each patch on a particle type.

        The dictionary has the following keys:
        - ``directors`` (`list` [`tuple` [`float`, `float`, `float`]]): List of
          directional vectors of the patches on a particle.
        - ``deltas`` (`list` [`float`]): List of delta values (the half opening
          angle of the patch in radian) of the patches.

    """
    _cpp_class_name = "PairPotentialAngularStep"

    def __init__(self, isotropic_potential):
        patch = TypeParameter(
            'patch',
            'particle_types',
            TypeParameterDict(
                OnlyIf(to_type_converter(
                    dict(directors=[(float,) * 3], deltas=[float])),
                       allow_none=True),
            len_keys=1)
        )
        self._add_typeparam(patch)

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
