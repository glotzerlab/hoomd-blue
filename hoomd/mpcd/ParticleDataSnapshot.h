// Copyright (c) 2009-2017 The Regents of the University of Michigan
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
        ~ParticleDataSnapshot() { std::cout << "Destructing snapshot" << std::endl; };

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

//! Adapter to access MPCD particle data arrays as numpy arrays
/*!
 * mpcd::ParticleDataSnapshot needs to be visible itself for use in external plugins. This adapter class provides
 * The hidden API needed to access the data as numpy arrays.
 *
 * This class stores a python dict, m_holder, so that it can tie the lifetime of the numpy arrays to the lifetime
 * of this instance.
 *
 * \ingroup data_structs
 */
class ParticleDataSnapshotAdapter
    {
    public:
        ParticleDataSnapshotAdapter(ParticleDataSnapshot& pdata) : m_pdata(pdata) { }

        void resize(unsigned int N)
            {
            m_pdata.resize(N);
            }

        bool validate() const
            {
            return m_pdata.validate();
            }

        void replicate(unsigned int nx,
                       unsigned int ny,
                       unsigned int nz,
                       const BoxDim& old_box,
                       const BoxDim& new_box)
            {
            m_pdata.replicate(nx, ny, nz, old_box, new_box);
            }

        Scalar getMass()
            {
            return m_pdata.mass;
            }

        void setMass(Scalar mass)
            {
            m_pdata.mass = mass;
            }

        unsigned int getSize()
            {
            return m_pdata.size;
            }

        //! Get snapshot positions as a numpy array
        pybind11::object getPosition();

        //! Get snapshot velocities as a numpy array
        pybind11::object getVelocity();

        //! Get snapshot types as a numpy array
        pybind11::object getType();

        //! Get snapshot type names as a python list
        pybind11::list getTypeNames();

        //! Set snapshot type names from a python list
        void setTypeNames(pybind11::list types);

    private:
        ParticleDataSnapshot& m_pdata;
        pybind11::dict m_holder;
    };

//! Export mpcd::ParticleDataSnapshot to python
void export_ParticleDataSnapshotAdapter(pybind11::module& m);
} // end namespace detail

} // end namespace mpcd

#endif // MPCD_PARTICLE_DATA_SNAPSHOT_H_
