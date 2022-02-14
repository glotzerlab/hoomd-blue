# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Implement BoxResize."""

from hoomd.operation import Updater
from hoomd.box import Box
from hoomd.data.parameterdicts import ParameterDict
from hoomd.variant import Variant, Constant
from hoomd import _hoomd
from hoomd.filter import ParticleFilter, All


class BoxResize(Updater):
    """Resizes the box between an initial and final box.

    `BoxResize` resizes the box between gradually from the initial box to the
    final box. The simulation box follows the linear interpolation between the
    initial and final boxes where the minimum of the variant gives `box1` and
    the maximum gives `box2`:

    .. math::

        \\begin{align*}
        L_{x}' &= \\lambda L_{2x} + (1 - \\lambda) L_{1x} \\\\
        L_{y}' &= \\lambda L_{2y} + (1 - \\lambda) L_{1y} \\\\
        L_{z}' &= \\lambda L_{2z} + (1 - \\lambda) L_{1z} \\\\
        xy' &= \\lambda xy_{2} + (1 - \\lambda) xy_{2} \\\\
        xz' &= \\lambda xz_{2} + (1 - \\lambda) xz_{2} \\\\
        yz' &= \\lambda yz_{2} + (1 - \\lambda) yz_{2} \\\\
        \\end{align*}

    Where `box1` is :math:`(L_{1x}, L_{1y}, L_{1z}, xy_1, xz_1, yz_1)`,
    `box2` is :math:`(L_{2x}, L_{2y}, L_{2z}, xy_2, xz_2, yz_2)`,
    :math:`\\lambda = \\frac{f(t) - \\min f}{\\max f - \\min f}`, :math:`t`
    is the timestep, and :math:`f(t)` is given by `variant`.

    For each particle :math:`i` matched by `filter`, `BoxResize` scales the
    particle to fit in the new box:

    .. math::

        \\vec{r}_i \\leftarrow s_x \\vec{a}_1' + s_y \\vec{a}_2' +
                               s_z \\vec{a}_3' -
                    \\frac{\\vec{a}_1' + \\vec{a}_2' + \\vec{a}_3'}{2}

    where :math:`\\vec{a}_k'` are the new box vectors determined by
    :math:`(L_x', L_y', L_z', xy', xz', yz')` and the scale factors are
    determined by the current particle position :math:`\\vec{r}_i` and the old
    box vectors :math:`\\vec{a}_k`:

    .. math::

        \\vec{r}_i = s_x \\vec{a}_1 + s_y \\vec{a}_2 + s_z \\vec{a}_3 -
                    \\frac{\\vec{a}_1 + \\vec{a}_2 + \\vec{a}_3}{2}

    After scaling particles that match the filter, `BoxResize` wraps all
    particles :math:`j` back into the new box:

    .. math::

        \\vec{r_j} \\leftarrow \\mathrm{minimum\\_image}_{\\vec{a}_k}'
                               (\\vec{r}_j)

    Note:
        The passed `Variant` must be bounded (i.e. it must have a true minimum
        and maximum) or the behavior of the updater is undefined.

    Note:
        Rescaling particles fails in HPMC simulations with more than one MPI
        rank.

    Note:
        When using rigid bodies, ensure that the `BoxResize` updater is last in
        the operations updater list. Immediately after the `BoxResize` updater
        triggers, rigid bodies (`hoomd.md.constrain.rigid`) will be temporarily
        deformed. `hoomd.md.Integrator` will run after the last updater and
        resets the constituent particle positions before computing forces.

    Args:
        trigger (hoomd.trigger.Trigger): The trigger to activate this updater.
        box1 (hoomd.Box): The box associated with the minimum of the
            passed variant.
        box2 (hoomd.Box): The box associated with the maximum of the
            passed variant.
        variant (hoomd.variant.Variant): A variant used to interpolate between
            the two boxes.
        filter (hoomd.filter.ParticleFilter): The subset of particle positions
            to update.

    Attributes:
        box1 (hoomd.Box): The box associated with the minimum of the
            passed variant.
        box2 (hoomd.Box): The box associated with the maximum of the
            passed variant.
        variant (hoomd.variant.Variant): A variant used to interpolate between
            the two boxes.
        trigger (hoomd.trigger.Trigger): The trigger to activate this updater.
        filter (hoomd.filter.ParticleFilter): The subset of particles to
            update.
    """

    def __init__(self, trigger, box1, box2, variant, filter=All()):
        params = ParameterDict(box1=Box,
                               box2=Box,
                               variant=Variant,
                               filter=ParticleFilter)
        params['box1'] = box1
        params['box2'] = box2
        params['variant'] = variant
        params['trigger'] = trigger
        params['filter'] = filter
        self._param_dict.update(params)
        super().__init__(trigger)

    def _attach(self):
        group = self._simulation.state._get_group(self.filter)
        self._cpp_obj = _hoomd.BoxResizeUpdater(
            self._simulation.state._cpp_sys_def, self.box1._cpp_obj,
            self.box2._cpp_obj, self.variant, group)
        super()._attach()

    def get_box(self, timestep):
        """Get the box for a given timestep.

        Args:
            timestep (int): The timestep to use for determining the resized
                box.

        Returns:
            Box: The box used at the given timestep.
            `None` before the first call to `Simulation.run`.
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
            box (Box): New box.
            filter (hoomd.filter.ParticleFilter): The subset of particles to
                update.
        """
        group = state._get_group(filter)
        updater = _hoomd.BoxResizeUpdater(state._cpp_sys_def,
                                          state.box._cpp_obj, box._cpp_obj,
                                          Constant(1), group)
        updater.update(state._simulation.timestep)
