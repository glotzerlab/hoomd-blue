# Copyright (c) 2009-2022 The Regents of the University of Michigan.
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
from hoomd.data.typeconverter import OnlyIf, to_type_converter
from collections.abc import Sequence
from hoomd.logging import log
import numpy as np


class Mesh(_HOOMDBaseObject):
    """Data structure combining multiple particles into a mesh.

    The mesh is defined by an array of triangles that make up a
    triangulated surface of particles. Each triangle consists of
    three particle tags. The mesh object consists of only one
    mesh triangle type with the default type name "mesh".

    Examples::

        mesh = mesh.Mesh()
        mesh.triangles = [[0,1,2],[0,2,3],[0,1,3],[1,2,3]]

    """

    def __init__(self):

        param_dict = ParameterDict(size=int,
                                   types=OnlyIf(
                                       to_type_converter([str]),
                                       preprocess=self._preprocess_type))

        param_dict["types"] = ["mesh"]
        param_dict["size"] = 0
        self._triangles = np.empty([0, 3], dtype=int)

        self._param_dict.update(param_dict)

    def _attach_hook(self):

        self._cpp_obj = _hoomd.MeshDefinition(
            self._simulation.state._cpp_sys_def)

        self.triangles = self._triangles

        if hoomd.version.mpi_enabled:
            pdata = self._simulation.state._cpp_sys_def.getParticleData()
            decomposition = pdata.getDomainDecomposition()
            if decomposition is not None:
                # create the c++ Communicator
                self._simulation._system_communicator.addMeshDefinition(
                    self._cpp_obj)
                self._cpp_obj.setCommunicator(
                    self._simulation._system_communicator)

    def _remove_dependent(self, obj):
        super()._remove_dependent(obj)
        if len(self._dependents) == 0:
            if self._attached:
                self._detach()
                self._remove()
                return
            if self._added:
                self._remove()

    @log(category='sequence')
    def triangles(self):
        """((*N*, 3) `numpy.ndarray` of ``uint32``): Mesh triangulation.

        A list of triplets of particle tags which encodes the
        triangulation of the mesh structure.
        """
        if self._attached:
            return self._cpp_obj.getTriangleData().group
        return self._triangles

    @triangles.setter
    def triangles(self, triag):
        if self._attached:
            self._cpp_obj.setTypes(list(self._param_dict['types']))

            self._cpp_obj.setTriangleData(triag)
        else:
            self.size = len(triag)
        self._triangles = triag

    @log(category='sequence', requires_run=True)
    def bonds(self):
        """((*N*, 2) `numpy.ndarray` of ``uint32``): Mesh bonds.

        A list of tuples of particle ids which encodes the
        bonds within the mesh structure.
        """
        return self._cpp_obj.getBondData().group

    def _preprocess_type(self, typename):
        if not isinstance(typename, str) and isinstance(typename, Sequence):
            if len(typename) != 1:
                raise ValueError("Only one meshtype is allowed.")
        return typename
