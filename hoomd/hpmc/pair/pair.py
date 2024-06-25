# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Base Pair class.

.. invisible-code-block: python

    simulation = hoomd.util.make_example_simulation()
    sphere = hoomd.hpmc.integrate.Sphere()
    sphere.shape['A'] = dict(diameter=0.0)
    simulation.operations.integrator = sphere

    pair =  hoomd.hpmc.pair.LennardJones()
    pair.params[('A', 'A')] = dict(epsilon=1, sigma=1, r_cut=2.5)

    logger = hoomd.logging.Logger()
"""

import hoomd
from hoomd.hpmc import _hpmc


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

    _ext_module = _hpmc

    def _make_cpp_obj(self):
        cpp_sys_def = self._simulation.state._cpp_sys_def
        cls = getattr(self._ext_module, self._cpp_class_name)
        return cls(cpp_sys_def)

    def _attach_hook(self):
        integrator = self._simulation.operations.integrator
        if not isinstance(integrator, hoomd.hpmc.integrate.HPMCIntegrator):
            raise RuntimeError("The integrator must be a HPMC integrator.")

        if not integrator._attached:
            raise RuntimeError("Integrator is not attached yet.")

        device = self._simulation.device

        if isinstance(device, hoomd.device.GPU):
            raise RuntimeError("Not implemented on the GPU")

        self._cpp_obj = self._make_cpp_obj()

        super()._attach_hook()

    def _detach_hook(self):
        self._cpp_obj.setParent(None)
        super()._detach_hook()

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
