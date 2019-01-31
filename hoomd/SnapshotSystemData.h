// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

/*! \file SnapshotSystemData.h
    \brief Defines the SnapshotSystemData class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __SNAPSHOT_SYSTEM_DATA_H__
#define __SNAPSHOT_SYSTEM_DATA_H__

#include "BoxDim.h"
#include "ParticleData.h"
#include "BondedGroupData.h"
#include "IntegratorData.h"

#ifndef NVCC
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#endif

/*! \ingroup data_structs
*/

//! Structure for initializing system data
/*! A snapshot is used for multiple purposes:
 * 1. for initializing the system
 * 2. during the simulation, e.g. to dump the system state or to analyze it
 *
 * Snapshots are temporary data-structures, they are only used for passing around data.
 *
 * A SnapshotSystemData is just a super-structure that holds snapshots of other data, such
 * as particles, bonds, etc. It is used by the SystemDefinition class to initially
 * set up these data structures, and can also be obtained from an object of that class to
 * analyze the current system state.
 *
 * \ingroup data_structs
 */
template <class Real>
struct SnapshotSystemData {
    unsigned int dimensions;               //!< The dimensionality of the system
    BoxDim global_box;                     //!< The dimensions of the simulation box
    SnapshotParticleData<Real> particle_data;    //!< The particle data
    std::map<unsigned int, unsigned int> map; //!< Lookup particle index by tag
    BondData::Snapshot bond_data;          //!< The bond data
    AngleData::Snapshot angle_data;         //!< The angle data
    DihedralData::Snapshot dihedral_data;    //!< The dihedral data
    ImproperData::Snapshot improper_data;    //!< The improper data
    ConstraintData::Snapshot constraint_data;//!< The constraint data
    PairData::Snapshot pair_data;            //!< The pair data
    std::vector<IntegratorVariables> integrator_data;  //!< The integrator data

    bool has_particle_data;                //!< True if snapshot contains particle data
    bool has_bond_data;                    //!< True if snapshot contains bond data
    bool has_angle_data;                   //!< True if snapshot contains angle data
    bool has_dihedral_data;                //!< True if snapshot contains dihedral data
    bool has_improper_data;                //!< True if snapshot contains improper data
    bool has_constraint_data;              //!< True if snapshot contains constraint data
    bool has_pair_data;                    //!< True if snapshot contains pair data
    bool has_integrator_data;              //!< True if snapshot contains integrator data

    //! Constructor
    SnapshotSystemData()
        {
        dimensions = 3;

        //! By default, all fields are used for initialization (even if they are empty)
        has_particle_data = true;
        has_bond_data = true;
        has_angle_data = true;
        has_dihedral_data = true;
        has_improper_data = true;
        has_constraint_data = true;
        has_pair_data = true;
        has_integrator_data = true;
        }

    // Replicate the system along three spatial dimensions
    /*! \param nx Number of times to replicate the system along the x direction
     *  \param ny Number of times to replicate the system along the y direction
     *  \param nz Number of times to replicate the system along the z direction
     */
    void replicate(unsigned int nx, unsigned int ny, unsigned int nz);

    // Broadcast information from rank 0 to all ranks
    /*! \param exec_conf The execution configuration
        Broadcasts the box and other metadata. Large particle data arrays are left on rank 0.
    */
    void broadcast_box(std::shared_ptr<ExecutionConfiguration> exec_conf);

    // Broadcast snapshot from root to all ranks
    /*! \param exec_conf The execution configuration
    */
    void broadcast(unsigned int root, std::shared_ptr<ExecutionConfiguration> exec_conf);

    // Broadcast snapshot from root to all partitions
    /*! \param exec_conf The execution configuration
    */
    void broadcast_all(unsigned int root, std::shared_ptr<ExecutionConfiguration> exec_conf);
    };

//! Export SnapshotParticleData to python

void export_SnapshotSystemData(pybind11::module& m);

#endif
