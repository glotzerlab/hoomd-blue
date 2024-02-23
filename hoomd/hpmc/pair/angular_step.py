# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Angular-step pair potential.

.. invisible-code-block: python

    simulation = hoomd.util.make_example_simulation()
    sphere = hoomd.hpmc.integrate.Sphere()
    sphere.shape['A'] = dict(diameter=0.0)
    simulation.operations.integrator = sphere

    pair =  hoomd.hpmc.pair.AngularStep()
    pair.patch[('m')] = dict(delta=0.1)
    pair.patch[('n')] = dict(delta=0.2)

    logger = hoomd.logging.Logger()
"""


import hoomd
from .pair import Pair

@hoomd.logging.modify_namespace(('hpmc', 'pair', 'AngularStep'))
class AngularStep(Pair):
    """Angular-step pair potential (HPMC).
    
    Args:
        delta(float): half opening angle of the patch in radian

    `AngularStep` computes the angular step potential, which is a composite 
    potential consist of an isotropic potential and a step function that is 
    dependent on the relative orientation between patches. One example of this 
    form of potential is the Kern-Frenkel model that is composed of a square 
    well potential and a step function. 

    .. rubric:: Example

    .. code-block:: python

        angular_step =  hoomd.hpmc.pair.AngularStep()
        angular_step.patch[('m')] = dict(delta=0.1)
        angular_step.patch[('n')] = dict(delta=0.2)
        simulation.operations.integrator.pair_potentials = [angular_step]

    .. py:attribute:: delta

        The half opening angle of the patch in radian.

        Type: `float`
    """
    
    _cpp_class_name = "PairPotentialAngularStep"

    def __init__(self, delta=0.1):
        self.delta = delta

