// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file MeshGroupData.h
    \brief Declares MeshGroupData
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

namespace hoomd
    {
/*! MeshGroupData is a generic storage class for mesh data as
 *   meshbonds and meshtriangles.
 *
 *  \tpp group_size Size of groups
 *  \tpp name Name of element, i.e. meshbond, meshtriangle.
 */
template<unsigned int group_size, typename Group, const char* name, typename snap>
class MeshGroupData : public BondedGroupData<group_size, Group, name, true>
    {
    public:
    static const unsigned int size = group_size;

    //! Constructor for empty MeshGroupData
    MeshGroupData(std::shared_ptr<ParticleData> pdata, unsigned int n_group_types);

    //! Constructor to initialize from a snapshot
    MeshGroupData(std::shared_ptr<ParticleData> pdata, const TriangleData::Snapshot& snapshot);

    virtual ~MeshGroupData();

    virtual void initializeFromSnapshot(
        const typename BondedGroupData<group_size, Group, name, true>::Snapshot& snapshot)
        {
        throw std::runtime_error("Not implemented");
        }

    void initializeFromTriangleSnapshot(const TriangleData::Snapshot& snapshot);

    //! Take a snapshot
    std::map<unsigned int, unsigned int> takeSnapshot(snap& snapshot) const;

    //! Add a single bonded mesh group on all processors
    /*! \param g Definition of group to add
     */
    unsigned int addBondedGroup(Group g);
    };

namespace detail
    {
//! Exports MeshBondData to python
template<class T, class Group>
void export_MeshGroupData(pybind11::module& m,
                          std::string name,
                          std::string snapshot_name,
                          bool export_struct = true);
    } // end namespace detail

/*!
 * Typedefs for template instantiations
 */

//! Definition of MeshBondData
typedef MeshGroupData<4, MeshBond, name_meshbond_data, BondData::Snapshot> MeshBondData;

    } // end namespace hoomd
#endif
