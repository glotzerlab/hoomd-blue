# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Implement SphereResize."""

import hoomd
from hoomd.operation import Updater
from hoomd.sphere import sphere
from hoomd.data.parameterdicts import ParameterDict
from hoomd.variant import Variant, Constant
from hoomd import _hoomd
from hoomd.filter import ParticleFilter, All
from hoomd.trigger import Periodic


class SphereResize(Updater):
    """Resizes the sphere between an initial and final sphere.

    `SphereResize` resizes the sphere between gradually from the initial sphere to the
    final sphere. The simulation sphere follows the linear interpolation between the
    initial and final spheres where the minimum of the variant gives `sphere1` and
    the maximum gives `sphere2`:

    .. math::

        \\begin{align*}
        R' &= \\lambda R_{2} + (1 - \\lambda) R_{1} \\\\
        \\end{align*}

    Where `sphere1` is :math:`(R_{1})`,
    `sphere2` is :math:`(R_{2})`,
    :math:`\\lambda = \\frac{f(t) - \\min f}{\\max f - \\min f}`, :math:`t`
    is the timestep, and :math:`f(t)` is given by `variant`.

    Important:
        The passed `Variant` must be bounded on the interval :math:`t \\in
        [0,\\infty)` or the behavior of the updater is undefined.

    Note:
        When using rigid bodies, ensure that the `SphereResize` updater is last in
        the operations updater list. Immediately after the `SphereResize` updater
        triggers, rigid bodies (`hoomd.md.constrain.Rigid`) will be temporarily
        deformed. `hoomd.md.Integrator` will run after the last updater and
        resets the constituent particle positions before computing forces.

    Args:
        trigger (hoomd.trigger.trigger_like): The trigger to activate this
            updater.
        sphere1 (hoomd.sphere): The sphere associated with the minimum of the
            passed variant.
        sphere2 (hoomd.sphere): The sphere associated with the maximum of the
            passed variant.
        variant (hoomd.variant.variant_like): A variant used to interpolate
            between the two spherees.
        filter (hoomd.filter.filter_like): The subset of particle positions
            to update.

    Attributes:
        sphere1 (hoomd.sphere): The sphere associated with the minimum of the
            passed variant.
        sphere2 (hoomd.sphere): The sphere associated with the maximum of the
            passed variant.
        variant (hoomd.variant.Variant): A variant used to interpolate between
            the two spherees.
        trigger (hoomd.trigger.Trigger): The trigger to activate this updater.
        filter (hoomd.filter.filter_like): The subset of particles to
            update.
    """

    def __init__(self, trigger, sphere1, sphere2, variant, filter=All()):
        params = ParameterDict(sphere1=sphere,
                               sphere2=sphere,
                               variant=Variant,
                               filter=ParticleFilter)
        params['sphere1'] = sphere1
        params['sphere2'] = sphere2
        params['variant'] = variant
        params['trigger'] = trigger
        params['filter'] = filter
        self._param_dict.update(params)
        super().__init__(trigger)

    def _attach_hook(self):
        group = self._simulation.state._get_group(self.filter)
        if isinstance(self._simulation.device, hoomd.device.CPU):
            self._cpp_obj = _hoomd.SphereResizeUpdater(
                self._simulation.state._cpp_sys_def, self.trigger,
                self.sphere1._cpp_obj, self.sphere2._cpp_obj, self.variant, group)
        else:
            self._cpp_obj = _hoomd.SphereResizeUpdaterGPU(
                self._simulation.state._cpp_sys_def, self.trigger,
                self.sphere1._cpp_obj, self.sphere2._cpp_obj, self.variant, group)

    def get_sphere(self, timestep):
        """Get the sphere for a given timestep.

        Args:
            timestep (int): The timestep to use for determining the resized
                sphere.

        Returns:
            sphere: The sphere used at the given timestep.
            `None` before the first call to `Simulation.run`.
        """
        if self._attached:
            timestep = int(timestep)
            if timestep < 0:
                raise ValueError("Timestep must be a non-negative integer.")
            return sphere._from_cpp(self._cpp_obj.get_current_sphere(timestep))
        else:
            return None

    @staticmethod
    def update(state, sphere, filter=All()):
        """Immediately scale the particle in the system state to the given sphere.

        Args:
            state (State): System state to scale.
            sphere (hoomd.sphere.sphere_like): New sphere.
            filter (hoomd.filter.filter_like): The subset of particles to
                update.
        """
        group = state._get_group(filter)

        if isinstance(state._simulation.device, hoomd.device.CPU):
            updater = _hoomd.SphereResizeUpdater(state._cpp_sys_def, Periodic(1),
                                              state.sphere._cpp_obj, sphere._cpp_obj,
                                              Constant(1), group)
        else:
            updater = _hoomd.SphereResizeUpdaterGPU(state._cpp_sys_def,
                                                 Periodic(1),
                                                 state.sphere._cpp_obj,
                                                 sphere._cpp_obj, Constant(1),
                                                 group)
        updater.update(state._simulation.timestep)
