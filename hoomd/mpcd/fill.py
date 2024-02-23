# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

r""" MPCD virtual-particle fillers.

Virtual particles are MPCD solvent particles that are added to ensure MPCD
collision cells that are sliced by solid boundaries do not become "underfilled".
From the perspective of the MPCD algorithm, the number density of particles in
these sliced cells is lower than the average density, and so the solvent
properties may differ. In practice, this means that the boundary conditions do
not appear to be properly enforced.
"""

import hoomd
from hoomd.data.parameterdicts import ParameterDict
from hoomd.mpcd import _mpcd
from hoomd.mpcd.geometry import Geometry, ParallelPlates
from hoomd.operation import Operation


class VirtualParticleFiller(Operation):
    """Base virtual-particle filler.

    Args:
        type (str): Type of particles to fill.
        density (float): Particle number density.
        kT (hoomd.variant.variant_like): Temperature of particles.

    Virtual particles will be added with the specified `type` and `density`.
    Their velocities will be drawn from a Maxwell--Boltzmann distribution
    consistent with `kT`.

    Attributes:
        type (str): Type of particles to fill.

        density (float): Particle number density.

        kT (hoomd.variant.variant_like): Temperature of particles.

    """

    def __init__(self, type, density, kT):
        super().__init__()

        param_dict = ParameterDict(
            type=str(type),
            density=float(density),
            kT=hoomd.variant.Variant,
        )
        param_dict["kT"] = kT
        self._param_dict.update(param_dict)


class GeometryFiller(VirtualParticleFiller):
    """Virtual-particle filler for known geometry.

    Args:
        type (str): Type of particles to fill.
        density (float): Particle number density.
        kT (hoomd.variant.variant_like): Temperature of particles.
        geometry (hoomd.mpcd.geometry.Geometry): Surface to fill around.

    Virtual particles are inserted in cells whose volume is sliced by the
    specified `geometry`. The algorithm for doing the filling depends on the
    specific `geometry`.

    Attributes:
        geometry (hoomd.mpcd.geometry.Geometry): Surface to fill around.

    """

    _cpp_class_map = {}

    def __init__(self, type, density, kT, geometry):
        super().__init__(type, density, kT)

        param_dict = ParameterDict(geometry=Geometry,)
        param_dict["geometry"] = geometry
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        sim = self._simulation

        self.geometry._attach(sim)

        # try to find class in map, otherwise default to internal MPCD module
        geom_type = type(self.geometry)
        try:
            class_info = self._cpp_class_map[geom_type]
        except KeyError:
            class_info = (_mpcd, geom_type.__name__ + "GeometryFiller")
        class_info = list(class_info)
        if isinstance(sim.device, hoomd.device.GPU):
            class_info[1] += "GPU"
        class_ = getattr(*class_info, None)
        assert class_ is not None, "Virtual particle filler for geometry not found"

        self._cpp_obj = class_(
            sim.state._cpp_sys_def,
            self.type,
            self.density,
            self.kT,
            self.geometry._cpp_obj,
        )

        super()._attach_hook()

    def _detach_hook(self):
        self.geometry._detach()
        super()._detach_hook()

    @classmethod
    def _register_cpp_class(cls, geometry, module, cpp_class_name):
        cls._cpp_class_map[geometry] = (module, cpp_class_name)


GeometryFiller._register_cpp_class(ParallelPlates, _mpcd,
                                   "ParallelPlateGeometryFiller")
