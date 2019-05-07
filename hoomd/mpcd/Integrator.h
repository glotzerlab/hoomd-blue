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
#include "Sorter.h"
#include "VirtualParticleFiller.h"
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
            // if the current communicator is set, first disable the migrate signal request
            if (m_mpcd_comm)
                m_mpcd_comm->getMigrateRequestSignal().disconnect<mpcd::Integrator, &mpcd::Integrator::checkCollide>(this);

            // then set the new communicator with the migrate signal request
            m_mpcd_comm = comm;
            m_mpcd_comm->getMigrateRequestSignal().connect<mpcd::Integrator, &mpcd::Integrator::checkCollide>(this);
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

        //! Set the sorting method
        /*!
         * \param sorter Sorting method to use
         */
        void setSorter(std::shared_ptr<mpcd::Sorter> sorter)
            {
            m_sorter = sorter;
            }

        //! Get the current sorting method
        std::shared_ptr<mpcd::Sorter> getSorter() const
            {
            return m_sorter;
            }

        //! Remove the current sorting method
        /*!
         * \post The sorting method is set to a null shared pointer.
         */
        void removeSorter()
            {
            m_sorter.reset();
            }

        //! Add a virtual particle filling method
        void addFiller(std::shared_ptr<mpcd::VirtualParticleFiller> filler);

        //! Remove all virtual particle fillers
        void removeAllFillers()
            {
            m_fillers.clear();
            }

    protected:
        std::shared_ptr<mpcd::SystemData> m_mpcd_sys;   //!< MPCD system
        std::shared_ptr<mpcd::CollisionMethod> m_collide;   //!< MPCD collision rule
        std::shared_ptr<mpcd::StreamingMethod> m_stream;    //!< MPCD streaming rule
        std::shared_ptr<mpcd::Sorter> m_sorter;         //!< MPCD sorter

        #ifdef ENABLE_MPI
        std::shared_ptr<mpcd::Communicator> m_mpcd_comm;    //!< MPCD communicator
        #endif // ENABLE_MPI

        std::vector<std::shared_ptr<mpcd::VirtualParticleFiller>> m_fillers; //!< MPCD virtual particle fillers
    private:
        //! Check if a collision will occur at the current timestep
        bool checkCollide(unsigned int timestep)
            {
            return (m_collide && m_collide->peekCollide(timestep));
            }
    };

namespace detail
{
//! Exports the mpcd::Integrator to python
void export_Integrator(pybind11::module& m);
} // end namespace detail
} // end namespace mpcd

#endif // MPCD_INTEGRATOR_H_
