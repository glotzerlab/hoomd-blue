# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

r"""MPCD integration methods.

Extra integration methods for solutes (MD particles) embedded in an MPCD
fluid. These methods are not restricted to MPCD simulations: they can be used
as methods of `hoomd.md.Integrator`. For example, `BounceBack` might be used to
run DPD simulations with surfaces.

.. invisible-code-block: python

    simulation = hoomd.util.make_example_simulation(mpcd_types=["A"])
    simulation.operations.integrator = hoomd.mpcd.Integrator(dt=0.1)

"""

import hoomd
from hoomd.data.parameterdicts import ParameterDict
from hoomd.filter import ParticleFilter
from hoomd.md.methods import Method
from hoomd.mpcd import _mpcd
from hoomd.mpcd.geometry import Geometry


class BounceBack(Method):
    r"""Velocity Verlet integration method with bounce-back from surfaces.

    Args:
        filter (hoomd.filter.filter_like): Subset of particles on which to
            apply this method.
        geometry (hoomd.mpcd.geometry.Geometry): Surface to bounce back from.

    A bounce-back method for integrating solutes (MD particles) embedded in an
    MPCD fluid. The integration scheme is velocity Verlet with bounce-back
    performed at the solid boundaries defined by a geometry, as in
    `hoomd.mpcd.stream.BounceBack`. This gives a simple approximation of the
    interactions required to keep a solute bounded in a geometry, and more
    complex interactions can be specified, for example, by writing custom
    external fields.

    Similar caveats apply to these methods as for
    `hoomd.mpcd.stream.BounceBack`. In particular:

    1. The simulation box is periodic, but the `geometry` may impose
       non-periodic boundary conditions. You must ensure that the box is
       sufficiently large to enclose the `geometry` and that all particles lie
       inside it
    2. You must also ensure that particles do not self-interact through the
       periodic boundaries. This is usually achieved for simple pair potentials
       by padding the box size by the largest cutoff radius. Failure to do so
       may result in unphysical interactions.
    3. Bounce-back rules do not always enforce no-slip conditions at surfaces
       properly. It may still be necessary to add additional "ghost" MD
       particles in the surface to achieve the right boundary conditions and
       reduce density fluctuations.

    Warning:

        This method does not support anisotropic integration because torques are
        not computed for collisions with the boundary. Rigid bodies will also
        not be treated correctly because the integrator is not aware of the
        extent of the particles. The surface reflections are treated as point
        particles. These conditions are too complicated to validate easily, so
        it is the user's responsibility to choose the `filter` correctly.

    .. rubric:: Example:

    .. code-block:: python

        plates = hoomd.mpcd.geometry.ParallelPlates(separation=6.0)
        nve = hoomd.mpcd.methods.BounceBack(
            filter=hoomd.filter.All(), geometry=plates)
        simulation.operations.integrator.methods.append(nve)

    Attributes:
        filter (hoomd.filter.filter_like): Subset of particles on which to apply
            this method (*read only*).

        geometry (hoomd.mpcd.geometry.Geometry): Surface to bounce back from
            (*read only*).

    """

    _cpp_class_map = {}

    def __init__(self, filter, geometry):
        super().__init__()

        param_dict = ParameterDict(filter=ParticleFilter, geometry=Geometry)
        param_dict.update(dict(filter=filter, geometry=geometry))
        self._param_dict.update(param_dict)

    def check_particles(self):
        """Check if particles are inside `geometry`.

        This method can only be called after this object is attached to a
        simulation.

        Returns:
            True if all particles are inside `geometry`.

        .. rubric:: Example:

        .. code-block:: python

            assert nve.check_particles()

        """
        return self._cpp_obj.check_particles()

    def _attach_hook(self):
        sim = self._simulation

        self.geometry._attach(sim)

        # try to find class in map, otherwise default to internal MPCD module
        geom_type = type(self.geometry)
        try:
            class_info = self._cpp_class_map[geom_type]
        except KeyError:
            class_info = (_mpcd, "BounceBackNVE" + geom_type.__name__)
        class_info = list(class_info)
        if isinstance(sim.device, hoomd.device.GPU):
            class_info[1] += "GPU"
        class_ = getattr(*class_info, None)
        assert class_ is not None, "Bounce back method for geometry not found"

        group = sim.state._get_group(self.filter)
        self._cpp_obj = class_(sim.state._cpp_sys_def, group,
                               self.geometry._cpp_obj)
        super()._attach_hook()

    def _detach_hook(self):
        self.geometry._detach()
        super()._detach_hook()

    @classmethod
    def _register_cpp_class(cls, geometry, module, cpp_class_name):
        cls._cpp_class_map[geometry] = (module, cpp_class_name)
