# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

# Maintainer: joaander / All Developers are free to add commands for new
# features

"""Update particle properties.

When an updater is specified, it acts on the particle system each time step it
is triggered to change its state.
"""

from hoomd import _hoomd
from hoomd.md import _md
import hoomd
from hoomd.operation import Updater
from hoomd.data.parameterdicts import ParameterDict


class ZeroMomentum(Updater):
    """Zeroes system momentum.

    Args:
        trigger (hoomd.trigger.Trigger): Select the timesteps to zero momentum.

    During the time steps specified by *trigger*, particle velocities are
    modified such that the total linear momentum of the system is set to zero.

    Examples::

        zeroer = hoomd.md.update.ZeroMomentum(hoomd.trigger.Periodic(100))

    """

    def __init__(self, trigger):
        # initialize base class
        super().__init__(trigger)

    def _attach(self):
        # create the c++ mirror class
        self._cpp_obj = _md.ZeroMomentumUpdater(
            self._simulation.state._cpp_sys_def)
        super()._attach()


class constraint_ellipsoid:  # noqa: N801 (this will be removed)
    """Constrain particles to the surface of a ellipsoid.

    Args:
        group (``hoomd.group``): Group for which the update will be set
        P (tuple): (x,y,z) tuple indicating the position of the center of the
            ellipsoid (in distance units).
        rx (float): radius of an ellipsoid in the X direction (in distance
            units).
        ry (float): radius of an ellipsoid in the Y direction (in distance
            units).
        rz (float): radius of an ellipsoid in the Z direction (in distance
            units).
        r (float): radius of a sphere (in distance units), such that r=rx=ry=rz.

    `constraint_ellipsoid` specifies that all particles are constrained to the
    surface of an ellipsoid. Each time step particles are projected onto the
    surface of the ellipsoid. Method from:
    http://www.geometrictools.com/Documentation/\
            DistancePointEllipseEllipsoid.pdf

    .. attention::
        For the algorithm to work, we must have :math:`rx >= rz,~ry >= rz,~rz >
        0`.

    Note:
        This method does not properly conserve virial coefficients.

    Note:
        random thermal forces from the integrator are applied in 3D not 2D,
        therefore they aren't fully accurate.  Suggested use is therefore only
        for T=0.

    Examples::

        update.constraint_ellipsoid(P=(-1,5,0), r=9)
        update.constraint_ellipsoid(rx=7, ry=5, rz=3)

    """

    def __init__(self, group, r=None, rx=None, ry=None, rz=None, P=(0, 0, 0)):
        period = 1

        # Error out in MPI simulations
        if (hoomd.version.mpi_enabled):
            if hoomd.context.current.system_definition.getParticleData(
            ).getDomainDecomposition():
                hoomd.context.current.device.cpp_msg.error(
                    "constrain.ellipsoid is not supported in multi-processor "
                    "simulations.\n\n")
                raise RuntimeError("Error initializing updater.")

        # Error out if no radii are set
        if (r is None and rx is None and ry is None and rz is None):
            hoomd.context.current.device.cpp_msg.error(
                "no radii were defined in update.constraint_ellipsoid.\n\n")
            raise RuntimeError("Error initializing updater.")

        # initialize the base class
        # _updater.__init__(self)

        # Set parameters
        P = _hoomd.make_scalar3(P[0], P[1], P[2])
        if (r is not None):
            rx = ry = rz = r

        # create the c++ mirror class
        if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
            self.cpp_updater = _md.ConstraintEllipsoid(
                hoomd.context.current.system_definition, group.cpp_group, P, rx,
                ry, rz)
        else:
            self.cpp_updater = _md.ConstraintEllipsoidGPU(
                hoomd.context.current.system_definition, group.cpp_group, P, rx,
                ry, rz)

        self.setupUpdater(period)

        # store metadata
        self.group = group
        self.P = P
        self.rx = rx
        self.ry = ry
        self.rz = rz
        self.metadata_fields = ['group', 'P', 'rx', 'ry', 'rz']


class ReversePerturbationFlow(Updater):
    """Reverse Perturbation (Müller-Plathe) method to establish shear flow.

     "Florian Mueller-Plathe. Reversing the perturbation in nonequilibrium
     molecular dynamics: An easy way to calculate the shear viscosity of fluids.
     Phys. Rev. E, 59:4894-4898, May 1999."

    The simulation box is divided in a number of slabs.  Two distinct slabs of
    those are chosen. The "max" slab searched for the max.  velocity component
    in flow direction, the "min" is searched for the min.  velocity component.
    Afterward, both velocity components are swapped.

    This introduces a momentum flow, which drives the flow. The strength of this
    flow, can be controlled by the flow_target variant, which defines the
    integrated target momentum flow. The searching and swapping is repeated
    until the target is reached. Depending on the target sign, the "max" and
    "min" slap might be swapped.

    Args:
        filter (`hoomd.filter.ParticleFilter`): Subset of particles on which to
            apply this updater.

        flow_target (`hoomd.variant.Variant`): Integrated target flow.
            :math:`[\\delta t \\cdot \\mathrm{mass} \\cdot \\mathrm{length}
            \\cdot \\mathrm{time}^{-1}]` - where :math:`\\delta t` is the
            integrator step size.

        slab_direction (str): Direction perpendicular to the slabs. Can be "X",
            "Y", or "Z"

        flow_direction (str): Direction of the flow. Can be "X",
            "Y", or "Z"

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
        * This updater uses `hoomd.trigger.Periodic(1)` as a trigger, meaning it
          is applied every timestep.
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

        self._flow_target = hoomd.variant._setup_variant_input(flow_target)

    Attributes:
        filter (hoomd.filter.ParticleFilter): Subset of particles on which to
            apply this updater.

        flow_target (hoomd.variant.Variant): Integrated target flow in the
            natural units of the simulation.

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

        assert (n_slabs > 0), "Invalid negative number of slabs."
        if min_slab < 0:
            min_slab = 0
        if max_slab < 0:
            max_slab = n_slabs / 2
        if max_slab <= -1 or max_slab > n_slabs:
            raise ValueError("Invalid max_slab in [0," + str(n_slabs) + ").")
        if min_slab <= -1 or min_slab > n_slabs:
            raise ValueError("Invalid min_slab in [0," + str(n_slabs) + ").")
        if min_slab == max_slab:
            raise ValueError("Invalid min/max slabs. Both have the same value.")

        params = ParameterDict(filter=hoomd.filter.ParticleFilter,
                               flow_target=hoomd.variant.Variant,
                               slab_direction=str,
                               flow_direction=str,
                               n_slabs=int(n_slabs),
                               max_slab=int(max_slab),
                               min_slab=int(min_slab),
                               flow_epsilon=float(1e-2))
        params.update(
            dict(filter=filter,
                 flow_target=flow_target,
                 slab_direction=slab_direction,
                 flow_direction=flow_direction))
        self._param_dict.update(params)

        # This updater has to be applied every timestep
        super().__init__(hoomd.trigger.Periodic(1))

    def _attach(self):
        group = self._simulation.state._get_group(self.filter)
        sys_def = self._simulation.state._cpp_sys_def
        if isinstance(self._simulation.device, hoomd.device.CPU):
            self._cpp_obj = _md.MuellerPlatheFlow(
                sys_def, group, self.flow_target, self.slab_direction,
                self.flow_direction, self.n_slabs, self.min_slab, self.max_slab,
                self.flow_epsilon)
        else:
            self._cpp_obj = _md.MuellerPlatheFlowGPU(
                sys_def, group, self.flow_target, self.slab_direction,
                self.flow_direction, self.n_slabs, self.min_slab, self.max_slab,
                self.flow_epsilon)
        super()._attach()

    @property
    def summed_exchanged_momentum(self):
        R"""Returned the summed up exchanged velocity of the full simulation."""
        if self._attached:
            return self._cpp_obj.summed_exchanged_momentum
        else:
            return None
