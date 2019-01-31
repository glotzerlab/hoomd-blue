// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

#ifndef MPCD_PARTICLE_DATA_SNAPSHOT_H_
#define MPCD_PARTICLE_DATA_SNAPSHOT_H_

/*!
 * \file mpcd/ParticleDataSnapshot.h
 * \brief Declaration of mpcd::ParticleDataSnapshot
 */

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "hoomd/BoxDim.h"
#include "hoomd/VectorMath.h"
// pybind11
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"

#include <string>
#include <vector>

namespace mpcd
{

//! Container for all MPCD particle data
/*!
 * A mpcd::ParticleDataSnapshot is useful for manipulation / analysis of the
 * current state of the system, and more commonly, for initializing mpcd::ParticleData.
 * The snapshot data is stored in a global tag order that is tracked throughout
 * the simulation so that the global ordering can be restored.
 *
 * The mpcd::ParticleDataSnapshot is initialized to arrays with default values:
 * - position: (0,0,0)
 * - velocity: (0,0,0)
 * - type: 0
 * - type_mapping: None
 *
 * These arrays can be manipulated either directly in C++ (all members are public)
 * or at the python level (all members are exposed to python as numpy arrays).
 * The mpcd::ParticleData is initialized using mpcd::ParticleData::initializeFromSnapshot.
 *
 * \warning On the C++ level, it is assumed that the vectors holding the particle data
 * will not be resized when they are accessed. The snapshot should be resized
 * using **only** the public resize method instead.
 *
 * \ingroup data_structs
 */
class PYBIND11_EXPORT ParticleDataSnapshot
    {
    public:
        //! Default constructor
        ParticleDataSnapshot();

        //! Constructor
        ParticleDataSnapshot(unsigned int N);

        //! Destructor
        ~ParticleDataSnapshot() {};

        //! Resize the snapshot
        void resize(unsigned int N);

        //! Validate snapshot data
        bool validate() const;

        //! Replicate the snapshot data
        void replicate(unsigned int nx,
                       unsigned int ny,
                       unsigned int nz,
                       const BoxDim& old_box,
                       const BoxDim& new_box);

        unsigned int size;                      //!< Number of particles
        std::vector< vec3<Scalar> > position;   //!< MPCD particle positions
        std::vector< vec3<Scalar> > velocity;   //!< MPCD particle velocities
        std::vector<unsigned int> type;         //!< MPCD particle type IDs
        Scalar mass;                            //!< MPCD particle mass
        std::vector<std::string> type_mapping;  //!< Type name mapping
    };

namespace detail
{

//! Export mpcd::ParticleDataSnapshot to python
void export_ParticleDataSnapshot(pybind11::module& m);
} // end namespace detail

} // end namespace mpcd

#endif // MPCD_PARTICLE_DATA_SNAPSHOT_H_
