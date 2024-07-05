// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file SnapshotSystemData.h
    \brief Defines the SnapshotSystemData class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#ifndef __SNAPSHOT_SYSTEM_DATA_H__
#define __SNAPSHOT_SYSTEM_DATA_H__

#include "BondedGroupData.h"
#include "BoxDim.h"
#include "ParticleData.h"
#ifdef BUILD_MPCD
#include "hoomd/mpcd/ParticleDataSnapshot.h"
#endif

#ifndef __HIPCC__
#include <pybind11/pybind11.h>
#endif

/*! \ingroup data_structs
 */

namespace hoomd
    {
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
template<class Real> struct SnapshotSystemData
    {
    unsigned int dimensions;                  //!< The dimensionality of the system
    std::shared_ptr<BoxDim> global_box;       //!< The dimensions of the simulation box
    SnapshotParticleData<Real> particle_data; //!< The particle data
    BondData::Snapshot bond_data;             //!< The bond data
    AngleData::Snapshot angle_data;           //!< The angle data
    DihedralData::Snapshot dihedral_data;     //!< The dihedral data
    ImproperData::Snapshot improper_data;     //!< The improper data
    ConstraintData::Snapshot constraint_data; //!< The constraint data
    PairData::Snapshot pair_data;             //!< The pair data
#ifdef BUILD_MPCD
    mpcd::ParticleDataSnapshot mpcd_data; //!< The MPCD particle data
#endif

    //! Constructor
    SnapshotSystemData() : dimensions(3), global_box(std::make_shared<BoxDim>()) { }

    // Replicate the system along three spatial dimensions
    /*! \param nx Number of times to replicate the system along the x direction
     *  \param ny Number of times to replicate the system along the y direction
     *  \param nz Number of times to replicate the system along the z direction
     */
    void replicate(unsigned int nx, unsigned int ny, unsigned int nz);

    //! Move the snapshot's particle positions back into the box. Update particle images based on
    //! the number of wrapped images.
    void wrap();

    // Broadcast information from rank 0 to all ranks
    /*! \param mpi_conf The MPI configuration
        Broadcasts the box and other metadata. Large particle data arrays are left on rank 0.
    */
    void broadcast_box(std::shared_ptr<MPIConfiguration> mpi_conf);

    // Broadcast snapshot from root to all ranks
    /*! \param exec_conf The execution configuration
     */
    void broadcast(unsigned int root, std::shared_ptr<ExecutionConfiguration> exec_conf);

    // Broadcast snapshot from root to all partitions
    /*! \param exec_conf The execution configuration
     */
    void broadcast_all(unsigned int root, std::shared_ptr<ExecutionConfiguration> exec_conf);
    };

namespace detail
    {
//! Export SnapshotParticleData to python
void export_SnapshotSystemData(pybind11::module& m);

    } // end namespace detail

    } // end namespace hoomd
#endif
