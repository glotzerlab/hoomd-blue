# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

r""" MPCD integration methods

Defines extra integration methods useful for solutes (MD particles) embedded in
an MPCD solvent. However, these methods are not restricted to MPCD
simulations: they can be used as methods of `hoomd.md.Integrator`. For example,
`BounceBack` might be used to run DPD simulations with surfaces.

"""

import hoomd
from hoomd.data.parameterdicts import ParameterDict
from hoomd.filter import ParticleFilter
from hoomd.md.methods import Method
from hoomd.mpcd import _mpcd
from hoomd.mpcd.geometry import Geometry


class BounceBack(Method):
    r"""Velocity Verlet integration method with bounce-back rule for surfaces.

    Args:
        filter (hoomd.filter.filter_like): Subset of particles on which to
            apply this method.
        geometry (hoomd.mpcd.geometry.Geometry): Surface to bounce back from.

    A bounce-back method for integrating solutes (MD particles) embedded in
    an MPCD solvent. The integration scheme is velocity Verlet with bounce-back
    performed at the solid boundaries defined by a geometry, as in
    `hoomd.mpcd.stream.BounceBack`. This gives a simple approximation of the
    interactions required to keep a solute bounded in a geometry, and more complex
    interactions can be specified, for example, by writing custom external fields.

    Similar caveats apply to these methods as for `hoomd.mpcd.stream.BounceBack`.
    In particular:

    1. The simulation box is periodic, but the `geometry` may impose non-periodic
       boundary conditions. You must ensure that the box is sufficiently large to
       enclose the `geometry` and that all particles lie inside it, or an error will
       be raised at runtime.
    2. You must also ensure that particles do not self-interact through the periodic
       boundaries. This is usually achieved for simple pair potentials by padding
       the box size by the largest cutoff radius. Failure to do so may result in
       unphysical interactions.
    3. Bounce-back rules do not always enforce no-slip conditions at surfaces
       properly. It may still be necessary to add additional "ghost" MD particles in
       the surface to achieve the right boundary conditions and reduce density
       fluctuations.

    Warning:

        This method does not support anisotropic integration because
        torques are not computed for collisions with the boundary.
        Rigid bodies will also not be treated correctly because the
        integrator is not aware of the extent of the particles. The surface
        reflections are treated as point particles. These conditions are too
        complicated to validate easily, so it is the user's responsibility to
        choose the `filter` correctly.

    Attributes:
        filter (hoomd.filter.filter_like): Subset of particles on which to apply
            this method.

        geometry (hoomd.mpcd.geometry.Geometry): Surface to bounce back from.

    """

    def __init__(self, filter, geometry):
        super().__init__()

        param_dict = ParameterDict(filter=ParticleFilter, geometry=Geometry)
        param_dict.update(dict(filter=filter, geometry=geometry))
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        sim = self._simulation

        self.geometry._attach(sim)

        # try to find class in map, otherwise default to internal MPCD module
        geom_type = type(self.geometry)
        try:
            class_info = self._class_map[geom_type]
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

    _class_map = {}

    @classmethod
    def _register_geometry(cls, geometry, module, class_name):
        cls._class_map[geometry] = (module, class_name)
