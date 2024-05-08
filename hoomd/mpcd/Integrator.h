// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd::Integrator.h
 * \brief Declares mpcd::Integrator, which performs two-step integration on
 *        multiple groups with MPCD particles.
 */

#ifndef MPCD_INTEGRATOR_H_
#define MPCD_INTEGRATOR_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "CollisionMethod.h"
#include "Sorter.h"
#include "StreamingMethod.h"
#include "VirtualParticleFiller.h"
#ifdef ENABLE_MPI
#include "Communicator.h"
#endif // ENABLE_MPI

#include "hoomd/SystemDefinition.h"
#include "hoomd/md/IntegratorTwoStep.h"
#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace mpcd
    {
class PYBIND11_EXPORT Integrator : public hoomd::md::IntegratorTwoStep
    {
    public:
    //! Constructor
    Integrator(std::shared_ptr<SystemDefinition> sysdef, Scalar deltaT);

    //! Destructor
    virtual ~Integrator();

    //! Take one timestep forward
    virtual void update(uint64_t timestep);

    //! Change the timestep
    virtual void setDeltaT(Scalar deltaT);

    //! Prepare for the run
    virtual void prepRun(uint64_t timestep);

    //! Get the MPCD cell list shared by all methods
    std::shared_ptr<mpcd::CellList> getCellList() const
        {
        return m_cl;
        }

    //! Set the MPCD cell list shared by all methods
    void setCellList(std::shared_ptr<mpcd::CellList> cl)
        {
        m_cl = cl;
        syncCellList();
        }

#ifdef ENABLE_MPI
    //! Set the MPCD communicator to use
    void setMPCDCommunicator(std::shared_ptr<mpcd::Communicator> comm)
        {
        // if the current communicator is set, first disable the migrate signal request
        if (m_mpcd_comm)
            m_mpcd_comm->getMigrateRequestSignal()
                .disconnect<mpcd::Integrator, &mpcd::Integrator::checkCollide>(this);

        // then set the new communicator with the migrate signal request
        m_mpcd_comm = comm;
        m_mpcd_comm->getMigrateRequestSignal()
            .connect<mpcd::Integrator, &mpcd::Integrator::checkCollide>(this);
        }
#endif

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
        if (m_stream)
            {
            m_stream->setDeltaT(m_deltaT);
            }
        }

    //! Get the current sorting method
    std::shared_ptr<mpcd::Sorter> getSorter() const
        {
        return m_sorter;
        }

    //! Set the sorting method
    /*!
     * \param sorter Sorting method to use
     */
    void setSorter(std::shared_ptr<mpcd::Sorter> sorter)
        {
        m_sorter = sorter;
        }

    //! Get the virtual particle fillers
    std::vector<std::shared_ptr<mpcd::VirtualParticleFiller>>& getFillers()
        {
        return m_fillers;
        }

    protected:
    std::shared_ptr<mpcd::CellList> m_cl;             //!< MPCD cell list
    std::shared_ptr<mpcd::CollisionMethod> m_collide; //!< MPCD collision rule
    std::shared_ptr<mpcd::StreamingMethod> m_stream;  //!< MPCD streaming rule
    std::shared_ptr<mpcd::Sorter> m_sorter;           //!< MPCD sorter
#ifdef ENABLE_MPI
    std::shared_ptr<mpcd::Communicator> m_mpcd_comm; //!< MPCD communicator
#endif
    std::vector<std::shared_ptr<mpcd::VirtualParticleFiller>>
        m_fillers; //!< MPCD virtual particle fillers

    private:
    //! Check if a collision will occur at the current timestep
    bool checkCollide(uint64_t timestep)
        {
        return (m_collide && m_collide->peekCollide(timestep));
        }

    //! Synchronize cell list to integrator dependencies
    void syncCellList();
    };

namespace detail
    {
//! Exports the mpcd::Integrator to python
void export_Integrator(pybind11::module& m);
    } // end namespace detail
    } // end namespace mpcd
    } // end namespace hoomd
#endif // MPCD_INTEGRATOR_H_
