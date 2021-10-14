# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

"""Implement Mesh."""

from hoomd import _hoomd
from hoomd.operation import _HOOMDBaseObject
from hoomd.logging import log


class Mesh(_HOOMDBaseObject):
    """Data structure combining multiple particles into a mesh.

    Examples::

        mesh = mesh.Mesh()

    """

    def __init__(self, triangles, simulation):

        self._simulation = simulation
        self._triangles = triangles
        self._size = len(triangles)
        self._types = None
        self._typeid = None

        self._cpp_obj = _hoomd.MeshDefinition(
            self._simulation.state._cpp_sys_def.getParticleData())

        self.size = self._size
        self.types = ["A"]
        self.triangles = self._triangles

    @property
    def size(self):
        """(int): Number of triangles in the mesh."""
        if self._attached:
            return self._cpp_obj.triangles.N
        else:
            return self._size

    @size.setter
    def size(self, newN):
        if self._attached:
            self._update_triangles()
            self._cpp_obj.triangles.N = newN
        else:
            self._size = newN

    @property
    def types(self):
        """(list[str]): Names of the triangle types."""
        if self._attached:
            self._update_triangles()
            return self._cpp_obj.triangles.types
        else:
            return self._types

    @types.setter
    def types(self, newtypes):
        if self._attached:
            self._cpp_obj.triangles.types = newtypes
        else:
            self._types = newtypes

    @property
    def typeid(self):
        """((*N*,) `numpy.ndarray` of ``uint32``): Triangle type id."""
        if self._attached:
            self._update_triangles()
            return self._typeid
        else:
            return self._cpp_obj.triangles.typeid

    @typeid.setter
    def typeid(self, tid):
        if self._attached:
            self._cpp_obj.triangles.typeid[:] = tid
            if len(self._triangles) == self.size:
                self._update_mesh()
        else:
            self._typeid = tid

    @log(category='sequence')
    def triangles(self):
        """((*N*, 3) `numpy.ndarray` of ``uint32``): Mesh triangulation.

        A list of triplets of particle ids which encodes the
        triangulation of the mesh structure.
        """
        if self._attached:
            self._update_triangles()
            return self._cpp_obj.triangles.group
        else:
            return self._triangles

    @triangles.setter
    def triangles(self, triag):

        self._triangles = triag
        if self._attached:
            self._cpp_obj.triangles.group[:] = triag
            self._update_mesh()

    @log
    def energy(self):
        """(Scalar): Surface energy of the mesh."""
        return self._cpp_obj.mesh_energy

    def _update_mesh(self):
        self._cpp_obj.updateMeshData()

    def _update_triangles(self):
        self._cpp_obj.updateTriangleData()
