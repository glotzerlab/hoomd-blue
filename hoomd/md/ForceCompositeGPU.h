// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: jglaser

#include "ForceComposite.h"
#include "NeighborList.h"
#include "hoomd/Autotuner.h"

/*! \file ForceCompositeGPU.h
    \brief Implementation of a rigid body force compute, GPU version
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __ForceCompositeGPU_H__
#define __ForceCompositeGPU_H__

namespace hoomd
    {
namespace md
    {
class PYBIND11_EXPORT ForceCompositeGPU : public ForceComposite
    {
    public:
    //! Constructs the compute
    ForceCompositeGPU(std::shared_ptr<SystemDefinition> sysdef);

    //! Destructor
    virtual ~ForceCompositeGPU();

    //! Update the constituent particles of a composite particle
    /*  Using the position, velocity and orientation of the central particle
     * \param remote If true, consider remote bodies, otherwise bodies
     *        with a local central particle
     */
    virtual void updateCompositeParticles(uint64_t timestep);

    //! Set autotuner parameters
    /*! \param enable Enable/disable autotuning
        \param period period (approximate) in time steps when returning occurs

        Derived classes should override this to set the parameters of their autotuners.
    */
    virtual void setAutotunerParams(bool enable, unsigned int period)
        {
        ForceComposite::setAutotunerParams(enable, period);

        m_tuner_force->setPeriod(period);
        m_tuner_force->setEnabled(enable);

        m_tuner_virial->setPeriod(period);
        m_tuner_virial->setEnabled(enable);

        m_tuner_update->setPeriod(period);
        m_tuner_update->setEnabled(enable);
        }

    protected:
    //! Compute the forces and torques on the central particle
    virtual void computeForces(uint64_t timestep);

    //! Helper kernel to sort rigid bodies by their center particles
    virtual void findRigidCenters();

    //! Helper function to check if particles have been sorted and rebuild indices if necessary
    virtual void checkParticlesSorted()
        {
        if (m_rebuild_molecules)
            // identify center particles for use in GPU kernel
            findRigidCenters();

        // Must be called second since the method sets m_rebuild_molecules
        // to false if it is true.
        MolecularForceCompute::checkParticlesSorted();
        }

    std::unique_ptr<Autotuner> m_tuner_force; //!< Autotuner for block size and threads per particle
    std::unique_ptr<Autotuner>
        m_tuner_virial; //!< Autotuner for block size and threads per particle
    std::unique_ptr<Autotuner> m_tuner_update; //!< Autotuner for block size of update kernel

    GlobalArray<uint2> m_flag; //!< Flag to read out error condition

    GPUPartition m_gpu_partition; //!< Partition of the rigid bodies
    GlobalVector<unsigned int>
        m_rigid_center; //!< Contains particle indices of all central particles
    GlobalVector<unsigned int> m_lookup_center; //!< Lookup particle index -> central particle index
    };

namespace detail
    {
//! Exports the ForceCompositeGPU to python
void export_ForceCompositeGPU(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
