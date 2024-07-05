# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Triangulated mesh data structure.

The mesh data structure combines particles into a connected triangulated
network. The particles act as vertices of the triangulation and are
linked with their neighbors in both pairs via mesh bonds and triplets via
mesh triangles.

.. rubric:: Mesh triangles and mesh bonds

``Mesh.triangulation`` is a dictionary with a list of triangles that
constitutes the triangulation. Each triangle is defined by a triplet of
particle tags. For a given triangulation HOOMD-blue also constructs a
list of mesh bonds automatically. Each mesh bond is defined by a pair
of particle tags. The corresponding vertex particles share a common edge
in the triangulation.

.. rubric:: Mesh potentials

In MD simulations different bond potentials can be attached which connect
the vertex particles with a bond potential. The mesh data structure is
designed so that other potentials (like bending potentials or global
conservation potentials) can be implemented later.

See Also:
  See the documentation in `hoomd.md.mesh` for more information on how
  to apply potentials to the mesh object and in `hoomd.md.nlist` on
  adding mesh bond exceptions to the neighbor list.
"""

import hoomd
from hoomd import _hoomd
from hoomd.operation import _HOOMDBaseObject
from hoomd.data.parameterdicts import ParameterDict
from hoomd.data.typeconverter import OnlyIf, to_type_converter, NDArrayValidator
from hoomd.logging import log
import numpy as np


class Mesh(_HOOMDBaseObject):
    """Data structure combining multiple particles into a mesh.

    The mesh is defined by an array of triangles that make up a
    triangulated surface of particles. Each triangle consists of
    three particle tags and is assigned to a defined triangle
    type.

    .. rubric:: Example:

    .. code-block:: python

        mesh_obj = hoomd.mesh.Mesh()
        mesh_obj.types = ["mesh"]
        mesh_obj.triangulation = dict(type_ids = [0,0,0,0],
              triangles = [[0,1,2],[0,2,3],[0,1,3],[1,2,3]])

    .. py:attribute:: types

        Names of the triangle types.

        Type: `list` [`str`]

    .. py:attribute:: triangulation

        The mesh triangulation. The dictionary has the following keys:

        * ``type_ids`` ((*N*) `numpy.ndarray` of ``uint32``): List of
           triangle type ids.
        * ``triangles`` ((*N*, 3) `numpy.ndarray` of ``uint32``): List
          of triplets of particle tags which encodes the triangulation
          of the mesh structure.

        Type: `dict`
    """

    def __init__(self):

        param_dict = ParameterDict(
            types=[str],
            triangulation=OnlyIf(to_type_converter({
                "type_ids": NDArrayValidator(np.uint),
                "triangles": NDArrayValidator(np.uint, shape=(None, 3))
            }),
                                 postprocess=self._ensure_same_size))

        param_dict["types"] = ["mesh"]
        param_dict["triangulation"] = dict(type_ids=np.zeros(0, dtype=int),
                                           triangles=np.zeros((0, 3),
                                                              dtype=int))

        self._param_dict.update(param_dict)

    def _attach_hook(self):

        self._cpp_obj = _hoomd.MeshDefinition(
            self._simulation.state._cpp_sys_def, len(self._param_dict["types"]))

        self._cpp_obj.setTypes(list(self._param_dict['types']))

        if hoomd.version.mpi_enabled:
            pdata = self._simulation.state._cpp_sys_def.getParticleData()
            decomposition = pdata.getDomainDecomposition()
            if decomposition is not None:
                self._simulation._system_communicator.addMeshDefinition(
                    self._cpp_obj)

    def _ensure_same_size(self, triangulation):
        if triangulation is None:
            return None
        if len(triangulation["triangles"]) != len(triangulation["type_ids"]):
            raise ValueError(
                "Number of type_ids do not match number of triangles.")
        return triangulation

    @log(category='sequence', requires_run=True)
    def type_ids(self):
        """((*N*) `numpy.ndarray` of ``uint32``): Triangle type ids."""
        return self.triangulation["type_ids"]

    @log(category='sequence', requires_run=True)
    def triangles(self):
        """((*N*, 3) `numpy.ndarray` of ``uint32``): Mesh triangulation.

        A list of triplets of particle tags which encodes the
        triangulation of the mesh structure.
        """
        return self.triangulation["triangles"]

    @log(category='sequence', requires_run=True)
    def bonds(self):
        """((*N*, 2) `numpy.ndarray` of ``uint32``): Mesh bonds.

        A list of tuples of particle ids which encodes the
        bonds within the mesh structure.
        """
        return self._cpp_obj.getBondData().group

    @property
    def size(self):
        """(int): Number of triangles in the mesh."""
        if self._attached:
            return self._cpp_obj.getSize()
        if self.triangulation is None:
            return 0
        return len(self.triangulation["triangles"])
