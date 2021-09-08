// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: joaander

/*! \file MeshData.h
    \brief Defines the MeshData class
 */

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "BondedGroupData.h"
#include "ParticleData.h"

#include <memory>
#include <pybind11/pybind11.h>

#ifndef __MESH_DATA_H__
#define __MESH_DATA_H__

//! Forward declaration of SnapshotSystemData
template<class Real> struct SnapshotSystemData;

class PYBIND11_EXPORT MeshData
    {
    public:
    //! Constructs a MeshData with a simply initialized ParticleData
    MeshData(std::shared_ptr<ParticleData> pdata,
                     unsigned int n_triangle_types = 0);

    //! Construct from a snapshot
    template<class Real>
    MeshData(std::shared_ptr<ParticleData> pdata,
		 std::shared_ptr<SnapshotSystemData<Real>> snapshot);

    //! Return a snapshot of the current system data
    template<class Real> void takeSnapshot(std::shared_ptr<SnapshotSystemData<Real>> snap);

    //! Re-initialize the system from a snapshot
    template<class Real>
    void initializeFromSnapshot(std::shared_ptr<SnapshotSystemData<Real>> snapshot);

    private:
    std::shared_ptr<MeshBondData> m_meshbond_data;             //!< Bond data for the mesh
    std::shared_ptr<MeshTriangleData> m_meshtriangle_data;             //!< Triangle data for the mesh
    std::shared_ptr<TriangleData> triangle_data;             //!< Triangle data for the mesh
    };

//! Exports MeshData to python
void export_MeshData(pybind11::module& m);

#endif
