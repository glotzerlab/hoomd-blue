// Copyright (c) 2009-2019 The Regents of the University of Michigan
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

class PYBIND11_EXPORT Integrator : public ::IntegratorTwoStep
    {
    public:
        //! Constructor
        Integrator(std::shared_ptr<mpcd::SystemData> sysdata, Scalar deltaT);

        //! Destructor
        virtual ~Integrator();

        //! Sets the profiler for the integrator to use
        virtual void setProfiler(std::shared_ptr<Profiler> prof);

        //! Take one timestep forward
        virtual void update(unsigned int timestep);

        //! Change the timestep
        virtual void setDeltaT(Scalar deltaT);

        //! Prepare for the run
        virtual void prepRun(unsigned int timestep);

#ifdef ENABLE_MPI
        //! Set the MPCD communicator to use
        virtual void setMPCDCommunicator(std::shared_ptr<mpcd::Communicator> comm)
            {
            m_mpcd_comm = comm;
            }
#endif

        //! Set autotuner parameters
        virtual void setAutotunerParams(bool enable, unsigned int period);

        //! Get current collision method
        std::shared_ptr<mpcd::CollisionMethod> getCollisionMethod() const
            {
            return m_collide;
            }

        //! Set collision method
        /*!
         * \param collide Collision method to use
         */
        void setCollisionMethod(std::shared_ptr<mpcd::CollisionMethod> collide)
            {
            m_collide = collide;
            }

        //! Remove the collision method
        /*!
         * \post The collision method is set to a null shared pointer.
         */
        void removeCollisionMethod()
            {
            m_collide.reset();
            }

        //! Get current streaming method
        std::shared_ptr<mpcd::StreamingMethod> getStreamingMethod() const
            {
            return m_stream;
            }

        //! Set the streaming method
        /*!
         * \param stream Streaming method to use
         */
        void setStreamingMethod(std::shared_ptr<mpcd::StreamingMethod> stream)
            {
            m_stream = stream;
            m_stream->setDeltaT(m_deltaT);
            }

        //! Remove the streaming method
        /*!
         * \post The streaming method is set to a null shared pointer.
         */
        void removeStreamingMethod()
            {
            m_stream.reset();
            }

    protected:
        std::shared_ptr<mpcd::SystemData> m_mpcd_sys;   //!< MPCD system
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
