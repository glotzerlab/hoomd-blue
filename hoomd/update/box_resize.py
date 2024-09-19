# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Implement BoxResize.

.. invisible-code-block: python

    simulation = hoomd.util.make_example_simulation()

    inverse_volume_ramp = hoomd.variant.box.InverseVolumeRamp(
        initial_box=hoomd.Box.cube(6),
        final_volume=100,
        t_start=1_000,
        t_ramp=21_000)
"""

import hoomd
from hoomd.operation import Updater
from hoomd.box import Box
from hoomd.data.parameterdicts import ParameterDict
from hoomd import _hoomd
from hoomd.filter import ParticleFilter, All
from hoomd.trigger import Periodic


class BoxResize(Updater):
    """Vary the simulation box size as a function of time.

    Args:
        trigger (hoomd.trigger.trigger_like): The trigger to activate this
            updater.
        filter (hoomd.filter.filter_like): The subset of particle positions
            to update (defaults to `hoomd.filter.All`).
        box (hoomd.variant.box.BoxVariant): Box as a function of time.

    `BoxResize` resizes the simulation box as a function of time. For each
    particle :math:`i` matched by `filter`, `BoxResize` scales the particle to
    fit in the new box:

    .. math::

        \\vec{r}_i \\leftarrow s_x \\vec{a}_1' + s_y \\vec{a}_2' +
                               s_z \\vec{a}_3' -
                    \\frac{\\vec{a}_1' + \\vec{a}_2' + \\vec{a}_3'}{2}

    where :math:`\\vec{a}_k'` are the new box vectors determined by `box`
    evaluated at the current time step and the scale factors are determined
    by the current particle position :math:`\\vec{r}_i` and the previous box
    vectors :math:`\\vec{a}_k`:

    .. math::

        \\vec{r}_i = s_x \\vec{a}_1 + s_y \\vec{a}_2 + s_z \\vec{a}_3 -
                    \\frac{\\vec{a}_1 + \\vec{a}_2 + \\vec{a}_3}{2}

    After scaling particles that match the filter, `BoxResize` wraps all
    particles :math:`j` back into the new box:

    .. math::

        \\vec{r_j} \\leftarrow \\mathrm{minimum\\_image}_{\\vec{a}_k}'
                               (\\vec{r}_j)

    Warning:
        Rescaling particles in HPMC simulations with hard particles may
        introduce overlaps.

    Note:
        When using rigid bodies, ensure that the `BoxResize` updater is last in
        the operations updater list. Immediately after the `BoxResize` updater
        triggers, rigid bodies (`hoomd.md.constrain.Rigid`) will be temporarily
        deformed. `hoomd.md.Integrator` will run after the last updater and
        resets the constituent particle positions before computing forces.

    .. rubric:: Example:

    .. code-block:: python

        box_resize = hoomd.update.BoxResize(trigger=hoomd.trigger.Periodic(10),
                                            box=inverse_volume_ramp)
        simulation.operations.updaters.append(box_resize)

    Attributes:
        box (hoomd.variant.box.BoxVariant): The box as a function of time.

            .. rubric:: Example:

            .. code-block:: python

                box_resize.box = inverse_volume_ramp

        filter (hoomd.filter.filter_like): The subset of particles to
            update.

            .. rubric:: Example:

            .. code-block:: python

                filter_ = box_resize.filter
    """

    def __init__(
            self,
            trigger,
            box,
            filter=All(),
    ):
        params = ParameterDict(box=hoomd.variant.box.BoxVariant,
                               filter=ParticleFilter)

        params.update({'box': box, 'filter': filter})
        self._param_dict.update(params)
        super().__init__(trigger)

    def _attach_hook(self):
        group = self._simulation.state._get_group(self.filter)
        if isinstance(self._simulation.device, hoomd.device.CPU):
            self._cpp_obj = _hoomd.BoxResizeUpdater(
                self._simulation.state._cpp_sys_def, self.trigger, self.box,
                group)
        else:
            self._cpp_obj = _hoomd.BoxResizeUpdaterGPU(
                self._simulation.state._cpp_sys_def, self.trigger, self.box,
                group)

    def get_box(self, timestep):
        """Get the box for a given timestep.

        Args:
            timestep (int): The timestep to use for determining the resized
                box.

        Returns:
            Box: The box used at the given timestep.
            `None` before the first call to `Simulation.run`.

        .. rubric:: Example:

        .. code-block:: python

            box = box_resize.get_box(1_000_000)
        """
        if self._attached:
            timestep = int(timestep)
            if timestep < 0:
                raise ValueError("Timestep must be a non-negative integer.")
            return Box._from_cpp(self._cpp_obj.get_current_box(timestep))
        else:
            return None

    @staticmethod
    def update(state, box, filter=All()):
        """Immediately scale the particle in the system state to the given box.

        Args:
            state (State): System state to scale.
            box (hoomd.box.box_like): New box.
            filter (hoomd.filter.filter_like): The subset of particles to
                update (defaults to `hoomd.filter.All`).

        .. rubric:: Example:

        .. code-block:: python

            hoomd.update.BoxResize.update(state=simulation.state,
                                          box=box)
        """
        group = state._get_group(filter)

        box_variant = hoomd.variant.box.Constant(box)

        if isinstance(state._simulation.device, hoomd.device.CPU):
            updater = _hoomd.BoxResizeUpdater(state._cpp_sys_def, Periodic(1),
                                              box_variant, group)
        else:
            updater = _hoomd.BoxResizeUpdaterGPU(state._cpp_sys_def,
                                                 Periodic(1), box_variant,
                                                 group)
        updater.update(state._simulation.timestep)
