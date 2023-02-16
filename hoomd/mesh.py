# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Triangulated mesh data structure.

The mesh data structure combines particles into a connected triangulated
network. The particles act as vertices of the triangulation and are
linked with their neighbors in both pairs via mesh bonds and triplets via
mesh triangles.

.. rubric:: Mesh triangles and mesh bonds

``Mesh.triangles`` is a list of triangle data that constitutes the
triangulation. Each triangle is defined by a triplet of particle tags.
For a given triangulation HOOMD-blue also constructs a list of mesh bonds
automatically. Each mesh bond is defined by a pair of particle tags. The
corresponding vertex particles share a common edge in the triangulation.


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
    three particle tags. The mesh object consists of only one
    mesh triangle type with the default type name "mesh".

    Examples::

        mesh_obj = mesh.Mesh()
        mesh_obj.types = ["mesh"]
        mesh_obj.triangulation = dict(type_ids = [0,0,0,0],
              triangles = [[0,1,2],[0,2,3],[0,1,3],[1,2,3]])

    """

    def __init__(self):

        param_dict = ParameterDict(
            types=[str],
            size=int,
            triangulation=OnlyIf(to_type_converter({
                "type_ids": NDArrayValidator(np.uint),
                "triangles": NDArrayValidator(np.uint, shape=[None, 3])
            }),
                                 allow_none=True))

        param_dict["types"] = ["mesh"]
        param_dict["size"] = 0
        param_dict["triangulation"] = None

        self._param_dict.update(param_dict)

    def _attach_hook(self):

        self._cpp_obj = _hoomd.MeshDefinition(
            self._simulation.state._cpp_sys_def, len(self._param_dict["types"]))

        self._cpp_obj.setTypes(list(self._param_dict['types']))
        if self._param_dict._dict["triangulation"] is not None:
            self._set_triangulation(self._param_dict._dict["triangulation"])

        if hoomd.version.mpi_enabled:
            pdata = self._simulation.state._cpp_sys_def.getParticleData()
            decomposition = pdata.getDomainDecomposition()
            if decomposition is not None:
                # create the c++ Communicator
                self._simulation._system_communicator.addMeshDefinition(
                    self._cpp_obj)
                self._cpp_obj.setCommunicator(
                    self._simulation._system_communicator)

    def _ensure_same_size(self, triangulation):
        if len(triangulation["triangles"]) != len(triangulation["type_ids"]):
            raise ValueError(
                "Number of type_ids do not match number of triangles.")

    def _setattr_param(self, attr, value):
        if attr == "triangulation":
            self._ensure_same_size(value)
            self._set_triangulation(value)
            return
        super()._setattr_param(attr, value)

    def _getattr_param(self, attr):
        if attr == "triangulation":
            return self._get_triangulation()
        return super()._getattr_param(attr)

    def _get_triangulation(self):
        if self._attached:
            return dict(type_ids=self.type_ids, triangles=self.triangles)
        return self._param_dict._dict["triangulation"]

    def _set_triangulation(self, value):
        if self._attached:
            self._cpp_obj.setTriangleData(value["triangles"], value["type_ids"])
        else:
            self._param_dict._dict["triangulation"] = value
            self.size = len(value["triangles"])

    @log(category='sequence', requires_run=True)
    def type_ids(self):
        """((*N*) `numpy.ndarray` of ``uint32``): Triangle type ids."""
        return self._cpp_obj.getTriangleData().typeid

    @log(category='sequence', requires_run=True)
    def triangles(self):
        """((*N*, 3) `numpy.ndarray` of ``uint32``): Mesh triangulation.

        A list of triplets of particle tags which encodes the
        triangulation of the mesh structure.
        """
        return self._cpp_obj.getTriangleData().group

    @log(category='sequence', requires_run=True)
    def bonds(self):
        """((*N*, 2) `numpy.ndarray` of ``uint32``): Mesh bonds.

        A list of tuples of particle ids which encodes the
        bonds within the mesh structure.
        """
        return self._cpp_obj.getBondData().group
