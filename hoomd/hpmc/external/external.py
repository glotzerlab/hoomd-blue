# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Base External class.

.. invisible-code-block: python

    simulation = hoomd.util.make_example_simulation()
    sphere = hoomd.hpmc.integrate.Sphere()
    sphere.shape['A'] = dict(diameter=0.0)
    simulation.operations.integrator = sphere

    external = hoomd.hpmc.external.Linear()
    external.alpha['A'] = 1

    logger = hoomd.logging.Logger()
"""

import hoomd
from hoomd.hpmc import _hpmc


class External(hoomd.operation._HOOMDBaseObject):
    """External potential base class (HPMC).

    External potentials define energetic interaction between particles and
    external fields in `hoomd.hpmc.integrate.HPMCIntegrator`.

    Note:
        The base class `External` implements common attributes (`energy`, for
        example) and may be used in for `isinstance` or `issubclass` checks.
        `External` should not be instantiated directly by users.
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

    @hoomd.logging.log(requires_run=True)
    def energy(self):
        """float: Potential energy contributed by this potential \
        :math:`[\\mathrm{energy}]`.

        Typically:

        .. math::

            U = \\sum_{i=0}^\\mathrm{N_particles-1}
            U_{\\mathrm{external},i}

        See `hoomd.hpmc.integrate` for the full expression which includes
        the evaluation over multiple images when the simulation box is small.

        .. rubric:: Example

        .. code-block:: python

            logger.add(obj=external, quantities=['energy'])
        """
        timestep = self._simulation.timestep
        return self._cpp_obj.totalEnergy(False)
