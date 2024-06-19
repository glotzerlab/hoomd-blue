# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

r"""MPCD virtual-particle fillers.

Virtual particles are MPCD particles that are added to ensure MPCD
collision cells that are sliced by solid boundaries do not become "underfilled".
From the perspective of the MPCD algorithm, the number density of particles in
these sliced cells is lower than the average density, and so the transport
properties may differ. In practice, this usually means that the boundary
conditions do not appear to be properly enforced.

.. invisible-code-block: python

    simulation = hoomd.util.make_example_simulation(mpcd_types=["A"])
    simulation.operations.integrator = hoomd.mpcd.Integrator(dt=0.1)

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

    .. invisible-code-block: python

        filler = hoomd.mpcd.fill.VirtualParticleFiller(
            type="A",
            density=5.0,
            kT=1.0)
        simulation.operations.integrator.virtual_particle_fillers = [filler]

    Attributes:
        type (str): Type of particles to fill.

            .. rubric:: Example:

            .. code-block:: python

                filler.type = "A"

        density (float): Particle number density.

            .. rubric:: Example:

            .. code-block:: python

                filler.density = 5.0

        kT (hoomd.variant.variant_like): Temperature of particles.

            .. rubric:: Examples:

            Constant temperature.

            .. code-block:: python

                filler.kT = 1.0

            Variable temperature.

            .. code-block:: python

                filler.kT = hoomd.variant.Ramp(1.0, 2.0, 0, 100)

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
    """Virtual-particle filler for a bounce-back geometry.

    Args:
        type (str): Type of particles to fill.
        density (float): Particle number density.
        kT (hoomd.variant.variant_like): Temperature of particles.
        geometry (hoomd.mpcd.geometry.Geometry): Surface to fill around.

    Virtual particles are inserted in cells whose volume is sliced by the
    specified `geometry`. The algorithm for doing the filling depends on the
    specific `geometry`.

    .. rubric:: Example:

    Filler for parallel plate geometry.

    .. code-block:: python

        plates = hoomd.mpcd.geometry.ParallelPlates(separation=6.0)
        filler = hoomd.mpcd.fill.GeometryFiller(
            type="A",
            density=5.0,
            kT=1.0,
            geometry=plates)
        simulation.operations.integrator.virtual_particle_fillers = [filler]

    Attributes:
        geometry (hoomd.mpcd.geometry.Geometry): Surface to fill around
            (*read only*).

    """

    _cpp_class_map = {}

    def __init__(self, type, density, kT, geometry):
        super().__init__(type, density, kT)

        param_dict = ParameterDict(geometry=Geometry,)
        param_dict["geometry"] = geometry
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        sim = self._simulation
        sim._warn_if_seed_unset()

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
        assert class_ is not None, ("Virtual particle filler for geometry "
                                    "not found")

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
