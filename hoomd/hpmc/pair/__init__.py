# Copyright (c) 2009-2023 The Regents of the University of Michigan.
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

from . import user

import hoomd


class Pair(hoomd.operation._HOOMDBaseObject):
    """Pair potential base class (HPMC).

    Pair potentials define energetic interactions between pairs of particles in
    `hoomd.hpmc.integrate.HPMCIntegrator`.  Particles within a cutoff distance
    interact with an energy that is a function the type and orientation of the
    particles and the vector pointing from the *i* particle to the *j* particle
    center.

    Note:
        The base class `Pair` implements common attributes (`energy`, for
        example) and may be used in for `isinstance` or `issubclass` checks.
        `Pair` should not be instantiated directly by users.
    """

    def _attach_hook(self):
        integrator = self._simulation.operations.integrator
        if not isinstance(integrator, hoomd.hpmc.integrate.HPMCIntegrator):
            raise RuntimeError("The integrator must be a HPMC integrator.")

        if not integrator._attached:
            raise RuntimeError("Integrator is not attached yet.")

        device = self._simulation.device

        if isinstance(device, hoomd.device.GPU):
            raise RuntimeError("Not implemented on the GPU")

        cpp_sys_def = self._simulation.state._cpp_sys_def
        cls = getattr(hoomd.hpmc._hpmc, self._cpp_class_name)
        self._cpp_obj = cls(cpp_sys_def)

        super()._attach_hook()

    @hoomd.logging.log(requires_run=True)
    def energy(self):
        """float: Potential energy contributed by this potential \
        :math:`[\\mathrm{energy}]`.

        Typically:

        .. math::

            U = \\sum_{i=0}^\\mathrm{N_particles-1}
            \\sum_{j=i+1}^\\mathrm{N_particles-1}
            U_{\\mathrm{pair},ij}

        See `hoomd.hpmc.integrate` for the full expression which includes
        the evaluation over multiple images when the simulation box is small.

        .. rubric:: Example

        .. code-block:: python

            logger.add(obj=pair, quantities=['energy'])
        """
        integrator = self._simulation.operations.integrator
        timestep = self._simulation.timestep
        return integrator._cpp_obj.computePairEnergy(timestep, self._cpp_obj)


class LennardJones(Pair):
    """Lennard-Jones pair potential (HPMC).

    Args:
        default_r_cut (float): Default cutoff radius :math:`[\\mathrm{length}]`.
        default_r_on (float): Default XPLOR on radius
          :math:`[\\mathrm{length}]`.
        default_mode (str): Default energy
          shifting/smoothing mode.

    `LennardJones` computes the Lennard-Jones pair potential between every pair
    of particles in the simulation state. The functional form of the potential,
    including its behavior under shifting modes, is identical to that in
    the MD pair potential `hoomd.md.pair.LJ`.

    See Also:
        `hoomd.md.pair.LJ`

        `hoomd.md.pair`

    .. rubric:: Example

    .. code-block:: python

        lennard_jones =  hoomd.hpmc.pair.LennardJones()
        lennard_jones.params[('A', 'A')] = dict(epsilon=1, sigma=1, r_cut=2.5)
        simulation.operations.integrator.pair_potentials = [lennard_jones]

    .. py:attribute:: params

        The potential parameters. The dictionary has the following keys:

        * ``epsilon`` (`float`, **required**) -
          Energy well depth :math:`\\varepsilon` :math:`[\\mathrm{energy}]`.
        * ``sigma`` (`float`, **required**) -
          Particle size :math:`\\sigma` :math:`[\\mathrm{length}]`.
        * ``r_cut`` (`float`): Cutoff radius :math:`[\\mathrm{length}]`.
          Defaults to the value given in ``default_r_cut`` on construction.
        * ``r_on`` (`float`): XPLOR on radius :math:`[\\mathrm{length}]`.
          Defaults to the value given in ``default_r_on`` on construction.
        * ``mode`` (`str`): The energy shifting/smoothing mode: Possible values:
          ``"none"``, ``"shift"``, ``"xplor"``. Defaults to the value given
          in ``default_mode`` on construction.

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `dict`]
    """
    _cpp_class_name = "PairPotentialLennardJones"

    def __init__(self,
                 default_r_cut=None,
                 default_r_on=0.0,
                 default_mode='none'):
        if default_r_cut is None:
            default_r_cut = float
        else:
            default_r_cut = float(default_r_cut)

        params = hoomd.data.typeparam.TypeParameter(
            'params', 'particle_types',
            hoomd.data.parameterdicts.TypeParameterDict(
                epsilon=float,
                sigma=float,
                r_cut=default_r_cut,
                r_on=float(default_r_on),
                mode=str(default_mode),
                len_keys=2))
        self._add_typeparam(params)
