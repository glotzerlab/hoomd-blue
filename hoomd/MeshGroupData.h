// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: jglaser

/*! \file BondedGroupData.h
    \brief Declares BondedGroupData
 */

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#ifndef __MESH_GROUP_DATA_H__
#define __MESH_GROUP_DATA_H__

#include "BondedGroupData.h"
#include "ExecutionConfiguration.h"
#include "GPUVector.h"
#include "HOOMDMPI.h"
#include "HOOMDMath.h"
#include "Index1D.h"
#include "ParticleData.h"
#include "Profiler.h"

#ifdef ENABLE_HIP
#include "BondedGroupData.cuh"
#include "CachedAllocator.h"
#endif

#include <hoomd/extern/nano-signal-slot/nano_signal_slot.hpp>
#include <memory>
#include <type_traits>
#ifndef __HIPCC__
#include <pybind11/pybind11.h>
#endif

#include <map>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <vector>

/*! BondedGroupData is a generic storage class for small particle groups of fixed
 *  size N=2,3,4..., such as bonds, angles or dihedrals, which form part of a molecule.
 *
 *  \tpp group_size Size of groups
 *  \tpp name Name of element, i.e. bond, angle, dihedral, ..
 */
template<unsigned int group_size, typename Group, const char* name, typename snap, bool bond>
class MeshGroupData: public BondedGroupData<group_size,Group,name,true>
    {
    public:
    //! Group size
    //
    //! Group data element type
    typedef union group_storage<group_size> members_t;

    //! Constructor for empty BondedGroupData
    MeshGroupData(std::shared_ptr<ParticleData> pdata, unsigned int n_group_types);

    //! Constructor to initialize from a snapshot
    MeshGroupData(std::shared_ptr<ParticleData> pdata, const TriangleData::Snapshot& snapshot);

    virtual ~MeshGroupData();

    //! Initialize from a snapshot
    //using BondedGroupData<group_size,Group,name,true>::initializeFromSnapshot;
    void initializeFromSnapshot(const TriangleData::Snapshot& snapshot);

    //! Take a snapshot
    //using BondedGroupData<group_size,Group,name,true>::takeSnapshot;
    std::map<unsigned int, unsigned int> takeSnapshot(snap& snapshot) const;

    /*
     * add/remove groups globally
     */

    //! Add a single bonded group on all processors
    /*! \param g Definition of group to add
     */
    unsigned int addBondedGroup(Group g);

    private:

    virtual void rebuildGPUTable();

#ifdef ENABLE_HIP
    //! Helper function to rebuild lookup by index table on the GPU
    virtual void rebuildGPUTableGPU();
#endif

    };

//! Exports BondData to python
template<class T, class Group>
void export_MeshGroupData(pybind11::module& m,
                            std::string name,
                            std::string snapshot_name,
                            bool export_struct = true);

/*!
 * Typedefs for template instantiations
 */

//! Definition of MeshBondData
typedef MeshGroupData<4, MeshBond, name_meshbond_data, BondData::Snapshot, true> MeshBondData;


//! Definition of MeshTriangleData
typedef MeshGroupData<6, MeshTriangle, name_meshtriangle_data, TriangleData::Snapshot, false> MeshTriangleData;

#endif
