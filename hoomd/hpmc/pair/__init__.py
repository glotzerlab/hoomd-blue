# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Pair Potentials for Monte Carlo.

Define :math:`U_{\\mathrm{pair},ij}` for use with
`hoomd.hpmc.integrate.HPMCIntegrator`. Assign a pair potential instance to
`hpmc.integrate.HPMCIntegrator.pair_potential` to activate the potential.
"""

from . import user

import hoomd


class Pair(hoomd.operation._HOOMDBaseObject):
    """Pair potential for HPMC."""

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
        """float: Total interaction energy of the system in the current state.

        .. math::

            U = \\sum_{i=0}^\\mathrm{N_particles-1}
            \\sum_{j=0}^\\mathrm{N_particles-1}
            U_{\\mathrm{pair},ij}

        Returns `None` when the patch object and integrator are not
        attached.
        """
        integrator = self._simulation.operations.integrator
        timestep = self._simulation.timestep
        return integrator._cpp_obj.computePatchEnergy(timestep)


class LJ(Pair):
    """Lennard-Jones pair potential"""
    _cpp_class_name = "PatchEnergyLJ"

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
