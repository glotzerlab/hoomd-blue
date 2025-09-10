# Copyright (c) 2009-2025 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""The mesh data structure combines particles into a connected tetrahededralized
network. The particles act as vertices of the tetrahedra.

.. rubric:: Mesh tetrahedra

``Mesh.tetrahedralization`` is a dictionary with a list of tetrahedra that
constitutes the tetrahedralization. Each tetrahedron is defined by a quartet of
particle tags. 
"""


import hoomd
from hoomd import _hoomd
from hoomd.operation import _HOOMDBaseObject
from hoomd.data.parameterdicts import ParameterDict
from hoomd.data.typeconverter import OnlyIf, to_type_converter, NDArrayValidator
from hoomd.logging import log
import numpy as np

class Mesh3D(_HOOMDBaseObject):
    """Data structure combining multiple particles into a 3D mesh.

    The mesh is defined by an array of tetrahedra that make up a
    tetrahedralized volume of particles. Each tetrahedron consists of
    four particle tags and is assigned to a defined tetrahedron
    type.

    .. rubric:: Example:

    .. code-block:: python

        mesh_obj = hoomd.mesh.Mesh()
        mesh_obj.types = ["mesh"]
        mesh_obj.tetrahedralization = dict(
            type_ids=[0, 0, 0, 0],
            tetrahedra=[
                [0, 1, 2, 3],
                [1, 2, 3, 4],
                [1, 2, 4, 5],
                [1, 2, 0, 6],
            ],
        )

    .. py:attribute:: types

        Names of the tetrahedron types.

        Type: `list` [`str`]

    .. py:attribute:: tetrahedralization

        The 3D mesh tetrahedralization. The dictionary has the following keys:

        * ``type_ids`` ((*N*) `numpy.ndarray` of ``uint32``): List of
           triangle type ids.
        * ``tetrahedra`` ((*N*, 4) `numpy.ndarray` of ``uint32``): List
          of quartets of particle tags which encodes the tetrahedralization
          of the 3D mesh structure.

        Type: `dict`
    """
    def __init__(self, positions, tetrahedra, types, type_id, reference_tags):
        self._positions = positions
        self._tetrahedra = tetrahedra
        self._types = types
        self._type_id = type_id
        self._reference_tags = reference_tags


    def _attach_hook(self):
        snapshot = _hoomd.TetrahedronDataSnapshot()
        snapshot.types = self._types
        snapshot.N = len(self._tetrahedra)
        snapshot.group[:] = self._tetrahedra
        snapshot.group[:] = self._type_id

        self._cpp_obj = _hoomd.TetrahedronData(
            self._simulation.state._cpp_sys_def, snapshot
        )

        ## TO DO: mpi parallelization
        '''
        if hoomd.version.mpi_enabled:
            pdata = self._simulation.state._cpp_sys_def.getParticleData()
            decomposition = pdata.getDomainDecomposition()
            if decomposition is not None:
                self._simulation._system_communicator.addMeshDefinition(self._cpp_obj)
        '''

    # def _ensure_same_size():
    #     if tetrahedralization is None:
    #         return None
    #     if len(tetrahedralization["tetrahedra"]) != len(tetrahedralization["type_ids"]):
    #         raise ValueError("Number of type_ids do not match number of tetrahedra.")
    #     return tetrahedralization

    # @log(category="sequence", requires_run=True)
    # def type_ids(self):
    #     """((*N*) `numpy.ndarray` of ``uint32``): Tetrahedron type ids."""
    #     return self.tetrahedralization["type_ids"]
    
    # @log(category="sequence", requires_run=True)
    # def tetrahedra(self):
    #     """((*N*, 4) `numpy.ndarray` of ``uint32``): Mesh tetrahedralization.

    #     A list of quartets of particle tags which encodes the
    #     tetrahedralization of the 3D mesh structure.
    #     """
    #     return self.tetrahedralization["tetrahedra"]

    # @property
    # def size(self):
    #     """(int): Number of tetrahedra in the 3D mesh."""
    #     if self._attached:
    #         return self._cpp_obj.getSize()
    #     if self.tetrahedralization is None:
    #         return 0
    #     return len(self.tetrahedralization["tetrahedra"])

__all__= ["Mesh3D"]
