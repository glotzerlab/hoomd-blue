// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd::Integrator.h
 * \brief Declares mpcd::Integrator, which performs two-step integration on
 *        multiple groups with MPCD particles.
 */

#ifndef MPCD_INTEGRATOR_H_
#define MPCD_INTEGRATOR_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "CollisionMethod.h"
#include "StreamingMethod.h"
#include "SystemData.h"
#ifdef ENABLE_MPI
#include "Communicator.h"
#endif // ENABLE_MPI

#include "hoomd/md/IntegratorTwoStep.h"
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"

namespace mpcd
{

class Integrator : public ::IntegratorTwoStep
    {
    public:
        //! Constructor
        Integrator(std::shared_ptr<mpcd::SystemData> sysdata, Scalar deltaT);

        //! Destructor
        virtual ~Integrator();

        //! Sets the profiler for the compute to use
        virtual void setProfiler(std::shared_ptr<Profiler> prof);

        //! Take one timestep forward
        virtual void update(unsigned int timestep);

        //! Change the timestep
        virtual void setDeltaT(Scalar deltaT);

        //! Prepare for the run
        virtual void prepRun(unsigned int timestep);

#ifdef ENABLE_MPI
        //! Set the MPCD communicator to use
        virtual void setMPCDCommunicator(std::shared_ptr<mpcd::Communicator> comm);
#endif

        //! Set autotuner parameters
        virtual void setAutotunerParams(bool enable, unsigned int period);

        //! Set the embedded particle group
        void setEmbeddedGroup(std::shared_ptr<ParticleGroup> group)
            {
            m_embed_group = group;
            }

        //! Remove the embedded particle group
        void removeEmbeddedGroup()
            {
            m_embed_group = std::make_shared<ParticleGroup>();
            }

    protected:
        std::shared_ptr<mpcd::SystemData> m_mpcd_sys;   //!< MPCD system
        std::shared_ptr<ParticleGroup> m_embed_group;   //!< Embedded particle group
        std::shared_ptr<mpcd::CollisionMethod> m_collide;   //!< MPCD collision rule
        std::shared_ptr<mpcd::StreamingMethod> m_stream;    //!< MPCD streaming rule

        #ifdef ENABLE_MPI
        std::shared_ptr<mpcd::Communicator> m_mpcd_comm;    //!< MPCD communicator
        #endif // ENABLE_MPI
    };

namespace detail
{
//! Exports the mpcd::Integrator to python
void export_Integrator(pybind11::module& m);
} // end namespace detail
} // end namespace mpcd

#endif // MPCD_INTEGRATOR_H_
