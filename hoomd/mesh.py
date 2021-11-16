# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

"""Implement Mesh."""

from hoomd import _hoomd
from hoomd.operation import _HOOMDBaseObject
from hoomd.logging import log
import numpy as np
import warnings


class Mesh(_HOOMDBaseObject):
    """Data structure combining multiple particles into a mesh.

    The mesh is defined by an array of triangles tht make up a
    triangulated surface.

    Examples::

        mesh = mesh.Mesh()
        mesh.size = 4
        mesh.triangles = [[0,1,2],[0,2,3],[0,1,3],[1,2,3]]

    """

    def __init__(self):

        self._triangles = np.empty([0, 3], dtype=int)
        self._size = 0

    def _attach(self):

        self._cpp_obj = _hoomd.MeshDefinition(
            self._simulation.state._cpp_sys_def)

        self.types = ["mesh"]
        if self._size != 0:
            self.size = self._size
            self.triangles = self._triangles

        super()._attach()

    def _remove_dependent(self, obj):
        super()._remove_dependent(obj)
        if len(self._dependents) == 0:
            if self._attached:
                self._detach()
                self._remove()
                return
            if self._added:
                self._remove()

    @property
    def size(self):
        """(int): Number of triangles in the mesh."""
        if self._attached:
            self._update_triangles()
            return self._cpp_obj.triangles.N
        return self._size

    @size.setter
    def size(self, newN):
        if self._attached:
            self._cpp_obj.triangles.N = newN
        self._size = newN

    @property
    def types(self):
        """(list[str]): Names of the triangle types."""
        return ["mesh"]

    @types.setter
    def types(self, newtypes):
        if self._attached:
            self._cpp_obj.triangles.types = ["mesh"]
        else:
            warnings.warn("Mesh only allows for one type!")

    @log(category='sequence')
    def triangles(self):
        """((*N*, 3) `numpy.ndarray` of ``uint32``): Mesh triangulation.

        A list of triplets of particle ids which encodes the
        triangulation of the mesh structure.
        """
        if self._attached:
            self._update_triangles()
            return self._cpp_obj.triangles.group
        return self._triangles

    @triangles.setter
    def triangles(self, triag):
        if self._attached:
            self._cpp_obj.triangles.group[:] = triag
            self._update_mesh()
        self._triangles = triag

    @log(category='sequence', requires_run=True)
    def bonds(self):
        """((*N*, 2) `numpy.ndarray` of ``uint32``): Mesh bonds.

        A list of tuples of particle ids which encodes the
        bonds within the mesh structure.
        """
        if self._attached:
            self._update_triangles()
            return self._cpp_obj.getBondData().group

    @log(requires_run=True)
    def energy(self):
        """(float): Surface energy of the mesh."""
        return self._cpp_obj.getEnergy()

    def _update_mesh(self):
        self._cpp_obj.updateMeshData()

    def _update_triangles(self):
        self._cpp_obj.updateTriangleData()
