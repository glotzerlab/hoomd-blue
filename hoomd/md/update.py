# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""MD updaters."""

from hoomd.md import _md
import hoomd
from hoomd.error import SimulationDefinitionError
from hoomd.operation import Updater
from hoomd.data.parameterdicts import ParameterDict
from hoomd.data.typeconverter import OnlyTypes
from hoomd.logging import log


class ZeroMomentum(Updater):
    r"""Zeroes system momentum.

    Args:
        trigger (hoomd.trigger.trigger_like): Select the timesteps to zero
            momentum.

    `ZeroMomentum` computes the center of mass linear momentum of the system:

    .. math::

        \vec{p} = \frac{1}{N_\mathrm{free,central}}
                  \sum_{i \in \mathrm{free,central}}
                  m_i \vec{v}_i

    and removes it:

    .. math::

        \vec{v}_i' = \vec{v}_i - \frac{\vec{p}}{m_i}

    where the index :math:`i` includes only free and central particles (and
    excludes consitutent particles of rigid bodies).

    Note:
        `ZeroMomentum` executes on the CPU even when using a GPU device.

    Examples::

        zero_momentum = hoomd.md.update.ZeroMomentum(
            hoomd.trigger.Periodic(100))
    """

    def __init__(self, trigger):
        # initialize base class
        super().__init__(trigger)

    def _attach_hook(self):
        # create the c++ mirror class
        self._cpp_obj = _md.ZeroMomentumUpdater(
            self._simulation.state._cpp_sys_def, self.trigger)


class ReversePerturbationFlow(Updater):
    """Reverse Perturbation method to establish shear flow.

     "Florian Mueller-Plathe. Reversing the perturbation in nonequilibrium
     molecular dynamics: An easy way to calculate the shear viscosity of fluids.
     Phys. Rev. E, 59:4894-4898, May 1999."

    The simulation box is divided in a number of slabs.  Two distinct slabs of
    those are chosen. The "max" slab searches for the maximum velocity component
    in flow direction while the "min" slab searches for the minimum velocity
    component. Afterward, both velocity components are swapped.

    This introduces a momentum flow, which drives the flow. The strength of this
    flow is set through the `flow_target` argument, which defines a target value
    for the time-integrated momentum flux. The searching and swapping is
    repeated until the target is reached. Depending on the target sign, the
    "max" and "min" slab might be swapped.

    Args:
        filter (hoomd.filter.filter_like): Subset of particles on which to
            apply this updater.

        flow_target (hoomd.variant.variant_like): Target value of the
            time-integrated momentum flux.
            :math:`[\\delta t \\cdot \\mathrm{mass} \\cdot \\mathrm{length}
            \\cdot \\mathrm{time}^{-1}]` - where :math:`\\delta t` is the
            integrator step size.

        slab_direction (str): Direction perpendicular to the slabs. Can be "x",
            "y", or "z"

        flow_direction (str): Direction of the flow. Can be "x",
            "y", or "z"

        n_slabs (int): Number of slabs used to divide the simulation box along
            the shear gradient. Using too few slabs will lead to a larger volume
            being disturbed by the momentum exchange, while using too many slabs
            may mean that there are not enough particles to exchange the target
            momentum.

        max_slab (int): Id < n_slabs where the max velocity component is search
            for. If set < 0 the value is set to its default n_slabs/2.

        min_slab (int): Id < n_slabs where the min velocity component is search
            for. If set < 0 the value is set to its default 0.

    Attention:
        * This updater uses ``hoomd.trigger.Periodic(1)`` as a trigger, meaning
          it is applied every timestep.
        * This updater works currently only with orthorhombic boxes.


    Note:
        The attributes of this updater are immutable once the updater is
        attached to a simulation.

    Examples::

        # const integrated flow with 0.1 slope for max 1e8 timesteps
        ramp = hoomd.variant.Ramp(0.0, 0.1e8, 0, int(1e8))
        # velocity gradient in z direction and shear flow in x direction.
        mpf = hoomd.md.update.ReversePerturbationFlow(filter=hoomd.filter.All(),
                                                      flow_target=ramp,
                                                      slab_direction="Z",
                                                      flow_direction="X",
                                                      n_slabs=20)

    Attributes:
        filter (hoomd.filter.filter_like): Subset of particles on which to
            apply this updater.

        flow_target (hoomd.variant.Variant): Target value of the
            time-integrated momentum flux.

        slab_direction (str): Direction perpendicular to the
            slabs.

        flow_direction (str): Direction of the flow.

        n_slabs (int): Number of slabs.

        max_slab (int): Id < n_slabs where the max velocity component is
            searched for.

        min_slab (int): Id < n_slabs where the min velocity component is
            searched for.
    """

    def __init__(self,
                 filter,
                 flow_target,
                 slab_direction,
                 flow_direction,
                 n_slabs,
                 max_slab=-1,
                 min_slab=-1):

        params = ParameterDict(
            filter=hoomd.filter.ParticleFilter,
            flow_target=hoomd.variant.Variant,
            slab_direction=OnlyTypes(str,
                                     strict=True,
                                     postprocess=self._to_lowercase),
            flow_direction=OnlyTypes(str,
                                     strict=True,
                                     postprocess=self._to_lowercase),
            n_slabs=OnlyTypes(int, preprocess=self._preprocess_n_slabs),
            max_slab=OnlyTypes(int, preprocess=self._preprocess_max_slab),
            min_slab=OnlyTypes(int, preprocess=self._preprocess_min_slab),
            flow_epsilon=float(1e-2))
        params.update(
            dict(filter=filter,
                 flow_target=flow_target,
                 slab_direction=slab_direction,
                 flow_direction=flow_direction,
                 n_slabs=n_slabs))
        self._param_dict.update(params)
        self._param_dict.update(dict(max_slab=max_slab))
        self._param_dict.update(dict(min_slab=min_slab))

        # This updater has to be applied every timestep
        super().__init__(hoomd.trigger.Periodic(1))

    def _to_lowercase(self, letter):
        return letter.lower()

    def _preprocess_n_slabs(self, n_slabs):
        if n_slabs < 0:
            raise ValueError(f"The number of slabs is negative, \
                              n_slabs = {n_slabs}")
        return n_slabs

    def _preprocess_max_slab(self, max_slab):
        if max_slab < 0:
            max_slab = self.n_slabs / 2
        if max_slab <= -1 or max_slab > self.n_slabs:
            raise ValueError(f"Invalid max_slab of {max_slab}")
        return max_slab

    def _preprocess_min_slab(self, min_slab):
        if min_slab < 0:
            min_slab = 0
        if min_slab <= -1 or min_slab > self.n_slabs:
            raise ValueError(f"Invalid min_slab of {min_slab}")
        if min_slab == self.max_slab:
            raise ValueError(f"Min and max slab are equal. \
                              min_slab = max_slab = {min_slab}")
        return min_slab

    def _attach_hook(self):
        group = self._simulation.state._get_group(self.filter)
        sys_def = self._simulation.state._cpp_sys_def
        if isinstance(self._simulation.device, hoomd.device.CPU):
            self._cpp_obj = _md.MuellerPlatheFlow(
                sys_def, self.trigger, group, self.flow_target,
                self.slab_direction, self.flow_direction, self.n_slabs,
                self.min_slab, self.max_slab, self.flow_epsilon)
        else:
            self._cpp_obj = _md.MuellerPlatheFlowGPU(
                sys_def, self.trigger, group, self.flow_target,
                self.slab_direction, self.flow_direction, self.n_slabs,
                self.min_slab, self.max_slab, self.flow_epsilon)

    @log(category="scalar", requires_run=True)
    def summed_exchanged_momentum(self):
        """Returns the summed up exchanged velocity of the full simulation."""
        return self._cpp_obj.summed_exchanged_momentum


class ActiveRotationalDiffusion(Updater):
    r"""Updater to introduce rotational diffusion with an active force.

    Args:
        trigger (hoomd.trigger.trigger_like): Select the timesteps to update
            rotational diffusion.
        active_force (hoomd.md.force.Active): The active force associated with
            the updater can be any subclass of the class
            `hoomd.md.force.Active`.
        rotational_diffusion (hoomd.variant.variant_like): The rotational
            diffusion as a function of time.

    `ActiveRotationalDiffusion` works directly with `hoomd.md.force.Active` or
    `hoomd.md.force.ActiveOnManifold` to apply rotational diffusion to the
    particle quaternions :math:`\mathbf{q}_i` in simulations with active forces.
    The persistence length of an active particle's path is :math:`v_0 / D_r`.

    In 2D, the diffusion follows :math:`\delta \theta / \delta t = \Gamma
    \sqrt{2 D_r / \delta t}`, where :math:`D_r` is the rotational diffusion
    constant and the :math:`\Gamma` unit-variance random variable.

    In 3D, :math:`\hat{p}_i` is a unit vector in 3D space, and the diffusion
    follows :math:`\delta \hat{p}_i / \delta t = \Gamma \sqrt{2 D_r / \delta t}
    (\hat{p}_i (\cos \theta - 1) + \hat{p}_r \sin \theta)`, where
    :math:`\hat{p}_r` is an uncorrelated random unit vector.

    When used with `hoomd.md.force.ActiveOnManifold`, rotational diffusion is
    performed in the tangent plane of the manifold.

    Tip:
        Use `hoomd.md.force.Active.create_diffusion_updater` to construct
        a `ActiveRotationalDiffusion` instance.

    Attributes:
        trigger (hoomd.trigger.Trigger): Select the timesteps to update
            rotational diffusion.
        active_force (hoomd.md.force.Active): The active force associated with
            the updater. This is not settable after construction.
        rotational_diffusion (hoomd.variant.Variant): The rotational diffusion
            as a function of time.
    """

    def __init__(self, trigger, active_force, rotational_diffusion):
        super().__init__(trigger)
        param_dict = ParameterDict(rotational_diffusion=hoomd.variant.Variant,
                                   active_force=hoomd.md.force.Active)
        param_dict["rotational_diffusion"] = rotational_diffusion
        param_dict["active_force"] = active_force
        self._add_dependency(active_force)
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        # Since integrators are attached first, if the active force is not
        # attached then the active force is not a part of the simulation, and we
        # should error.
        if not self.active_force._attached:
            raise SimulationDefinitionError(
                "Active force for ActiveRotationalDiffusion object does not "
                "belong to the simulation integrator.")
        if self.active_force._simulation is not self._simulation:
            raise SimulationDefinitionError(
                "Active force for ActiveRotationalDiffusion object belongs to "
                "another simulation.")
        self._cpp_obj = _md.ActiveRotationalDiffusionUpdater(
            self._simulation.state._cpp_sys_def, self.trigger,
            self.rotational_diffusion, self.active_force._cpp_obj)

    def _handle_removed_dependency(self, active_force):
        sim = self._simulation
        if sim is not None:
            sim._operations.updaters.remove(self)
        super()._handle_removed_dependency(active_force)

    def _setattr_param(self, attr, value):
        if attr == "active_force":
            raise ValueError("active_force is not settable after construction.")
        super()._setattr_param(attr, value)
