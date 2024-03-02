# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Angular-step pair potential.

.. invisible-code-block: python

    simulation = hoomd.util.make_example_simulation()
    sphere = hoomd.hpmc.integrate.Sphere()
    sphere.shape['A'] = dict(diameter=0.0)
    sphere.shape['B'] = dict(diameter=0.0)
    simulation.operations.integrator = sphere

    lennard_jones =  hoomd.hpmc.pair.LennardJones()
    lennard_jones.params[('A', 'A')] = dict(epsilon=1, sigma=1, r_cut=4.0)
    lennard_jones.params[('A', 'B')] = dict(epsilon=2, sigma=1, r_cut=4.0)
    lennard_jones.params[('B', 'B')] = dict(epsilon=3, sigma=1, r_cut=4.0)
"""


import hoomd
from hoomd.data.parameterdicts import ParameterDict, TypeParameterDict
from hoomd.data.typeparam import TypeParameter
from hoomd.data.typeconverter import OnlyIf, OnlyTypes, to_type_converter

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

    Note:
        directors and delta values are dependent on the particle type. One 
        particle type has only one unique set of the patch directors and 
        delta values. Once defined, all particles with the same type have 
        the same patch directors and delta values. 
        
    .. rubric:: Example

    .. code-block:: python

        angular_step = hoomd.hpmc.pair.AngularStep(
                       isotropic_potential=lennard_jones)
        angular_step.patch['A'] = dict(directors=[(1.0, 0, 0)], 
                        deltas=[0.1])
        angular_step.patch['B'] = dict(directors=[(1.0, 0, 0), (0, 1.0, 0)], 
                        deltas=[0.1, 0.2])
        simulation.operations.integrator.pair_potentials = [angular_step]

    Set user-defined group of patch directors and delta values for each particle
    type. Patch directors are the directional unit vectors that represent 
    the patch locations on a particle, and deltas are the half opening angles of the
    patch in radian. 
        
    .. py:attribute:: patch

        The patch definition.

        Define the patch director and delta of each patch on a particle type. 
        Set a particle type name for each particle type which will have a 
        unique combination of patch directors and delta values. 

        The dictionary has the following keys:
        - ``directors`` (`list` [`tuple` [`float`, `float`, `float`]]): List of
          directional vectors of the patches on a particle.
        - ``deltas`` (`list` [`float`]): List of delta values (the half opening 
        angle of the patch in radian) of the patches. 

    """
    _cpp_class_name = "PairPotentialAngularStep"

    def __init__(self, isotropic_potential):
        particle = TypeParameter(
            'particle', 'particle_types',
            TypeParameterDict(OnlyIf(to_type_converter(
                dict(directors=[(float,) * 3],
                     deltas=[float])),
                     allow_none=True)),
                     len_keys=1,)
        self._add_typeparam(particle)

        if not isinstance(isotropic_potential, hoomd.hpmc.pair.Pair):
            raise TypeError(
                "isotropic_potential must subclass hoomd.hpmc.pair.Pair")
        self._isotropic_potential = isotropic_potential

    @property
    def isotropic_potential(self):
        """hpmc.pair.Pair: isotropic part of the interactions between 
        patchy particles.
        .. rubric:: Example
        .. code-block:: python
            angular_step.isotropic_potential
        """
        return self._isotropic_potential

    def _attach_hook(self):
        # attach the constituent potential
        self.isotropic_potential._attach(self._simulation)

        cpp_sys_def = self._simulation.state._cpp_sys_def
        cls = getattr(hoomd.hpmc._hpmc, self._cpp_class_name)
        self._cpp_obj = cls(cpp_sys_def, self.isotropic_potential._cpp_obj)

        self.isotropic_potential._cpp_obj.setParent(self._cpp_obj)

        super()._attach_hook()

    def _detach_hook(self):
        if self.isotropic_potential is not None:
            self.isotropic_potential._detach()

        super()._detach_hook()

