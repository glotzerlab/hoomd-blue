# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Mesh potential base class."""

from hoomd.md import _md
from hoomd.mesh import Mesh
from hoomd.md.force import Force
from hoomd.data.typeconverter import OnlyTypes
import hoomd
import warnings
import copy

validate_mesh = OnlyTypes(Mesh)


class MeshPotential(Force):
    """Constructs the bond potential applied to a mesh.

    :py:class:`MeshPotential` is the base class for all bond potentials applied
    to meshes.

    Warning:
        This class should not be instantiated by users. The class can be used
        for `isinstance` or `issubclass` checks.
    """

    def __init__(self, mesh):
        self._mesh = validate_mesh(mesh)

    def _add(self, simulation):
        # if mesh was associated with multiple pair forces and is still
        # attached, we need to deepcopy existing mesh.
        mesh = self._mesh
        if (not self._attached and mesh._attached
                and mesh._simulation != simulation):
            warnings.warn(
                f"{self} object is creating a new equivalent mesh structure."
                f" This is happending since the force is moving to a new "
                f"simulation. To supress the warning explicitly set new mesh.",
                RuntimeWarning)
            self._mesh = copy.deepcopy(mesh)
        # We need to check if the force is added since if it is not then this is
        # being called by a SyncedList object and a disagreement between the
        # simulation and mesh._simulation is an error. If the force is added
        # then the mesh is compatible. We cannot just check the mesh's _added
        # property because _add is also called when the SyncedList is synced.
        elif (not self._added and mesh._added
              and mesh._simulation != simulation):
            raise RuntimeError(
                f"Mesh associated with {self} is associated with "
                f"another simulation.")
        super()._add(simulation)
        # this ideopotent given the above check.
        self._mesh._add(simulation)
        # This is ideopotent, but we need to ensure that if we change
        # mesh when not attached we handle correctly.
        self._add_dependency(self._mesh)

    def _attach(self):
        """Create the c++ mirror class."""
        if self._simulation != self._mesh._simulation:
            raise RuntimeError("{} object's mesh is used in a "
                               "different simulation.".format(type(self)))

        if not self.mesh._attached:
            self.mesh._attach()

        if isinstance(self._simulation.device, hoomd.device.CPU):
            cpp_cls = getattr(_md, self._cpp_class_name)
        else:
            cpp_cls = getattr(_md, self._cpp_class_name + "GPU")

        self._cpp_obj = cpp_cls(self._simulation.state._cpp_sys_def,
                                self._mesh._cpp_obj)

        super()._attach()

    def _apply_typeparam_dict(self, cpp_obj, simulation):
        for typeparam in self._typeparam_dict.values():
            try:
                typeparam._attach(cpp_obj, self.mesh)
            except ValueError as err:
                raise err.__class__(
                    f"For {type(self)} in TypeParameter {typeparam.name} "
                    f"{str(err)}")

    @property
    def mesh(self):
        """Mesh data structure used to compute the bond potential."""
        return self._mesh

    @mesh.setter
    def mesh(self, value):
        if self._attached:
            raise RuntimeError(
                "mesh cannot be set after calling Simulation.run().")
        mesh = validate_mesh(value)
        if self._added:
            if mesh._added and self._simulation != mesh._simulation:
                raise RuntimeError("Mesh and forces must belong to the same "
                                   "simulation or SyncedList.")
            self._mesh._add(self._simulation)
        self._mesh = mesh
