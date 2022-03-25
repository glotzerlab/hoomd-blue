# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Implement Mesh."""

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

    The mesh is defined by an array of triangles tht make up a
    triangulated surface.

    Args:
        name ([`str`]): name of the mesh that also acts as a
        type name. Only one type per mesh can be defined!

    Examples::

        mesh = mesh.Mesh()
        mesh.triangles = [[0,1,2],[0,2,3],[0,1,3],[1,2,3]]

    """

    def __init__(self, name=["mesh"]):

        param_dict = ParameterDict(size=int,
                                   types=OnlyIf(
                                       to_type_converter([
                                           str,
                                       ] * 1),
                                       preprocess=self._preprocess_type))

        param_dict["types"] = name
        param_dict["size"] = 0
        self._triangles = np.empty([0, 3], dtype=int)

        self._param_dict.update(param_dict)

    def _attach(self):

        self._cpp_obj = _hoomd.MeshDefinition(
            self._simulation.state._cpp_sys_def)

        if hoomd.version.mpi_enabled:
            pdata = self._simulation.state._cpp_sys_def.getParticleData()
            decomposition = pdata.getDomainDecomposition()
            if decomposition is not None:
                # create the c++ Communicator
                self._simulation._system_communicator.addMeshDefinition(
                    self._cpp_obj)
                self._cpp_obj.setCommunicator(
                    self._simulation._system_communicator)

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

    @log(category='sequence')
    def triangles(self):
        """((*N*, 3) `numpy.ndarray` of ``uint32``): Mesh triangulation.

        A list of triplets of particle ids which encodes the
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
        if self._attached:
            return self._cpp_obj.getBondData().group

    @log(requires_run=True)
    def energy(self):
        """(float): Surface energy of the mesh."""
        return self._cpp_obj.getEnergy()

    def _preprocess_type(self, typename):
        if isinstance(typename, Sequence):
            if len(typename) != 1:
                raise ValueError("Only one meshtype is allowed.")
            return typename
