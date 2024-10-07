# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Pair Potentials for Monte Carlo.

Define :math:`U_{\\mathrm{pair},ij}` for use with `HPMCIntegrator
<hoomd.hpmc.integrate.HPMCIntegrator>`, which will sum all the energy from all
`Pair` potential instances in the
`pair_potentials <hpmc.integrate.HPMCIntegrator.pair_potentials>` list.

.. rubric:: Example:

.. invisible-code-block: python

    simulation = hoomd.util.make_example_simulation()
    sphere = hoomd.hpmc.integrate.Sphere()
    sphere.shape['A'] = dict(diameter=0.0)
    simulation.operations.integrator = sphere

    pair =  hoomd.hpmc.pair.LennardJones()
    pair.params[('A', 'A')] = dict(epsilon=1, sigma=1, r_cut=2.5)

    logger = hoomd.logging.Logger()

.. code-block:: python

    simulation.operations.integrator.pair_potentials = [pair]
"""

from .pair import Pair
from .lennard_jones import LennardJones
from .expanded_gaussian import ExpandedGaussian
from .lj_gauss import LJGauss
from .opp import OPP
from .union import Union
from .angular_step import AngularStep
from .step import Step
