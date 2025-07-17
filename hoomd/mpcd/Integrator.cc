// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/Integrator.cc
 * \brief Defines the mpcd::Integrator
 */

#include "Integrator.h"

#ifdef ENABLE_MPI
#include "Communicator.h"
#ifdef ENABLE_HIP
#include "CommunicatorGPU.h"
#endif
#endif

#include <pybind11/stl_bind.h>
PYBIND11_MAKE_OPAQUE(std::vector<std::shared_ptr<hoomd::mpcd::VirtualParticleFiller>>);

namespace hoomd
    {
/*!
 * \param sysdef System definition
 * \param deltaT Fundamental integration timestep
 */
mpcd::Integrator::Integrator(std::shared_ptr<SystemDefinition> sysdef, Scalar deltaT)
    : md::IntegratorTwoStep(sysdef, deltaT)
    {
    m_exec_conf->msg->notice(5) << "Constructing MPCD Integrator" << std::endl;

#ifdef ENABLE_MPI
    // automatically create MPCD communicator in MPI simulations
    if (m_pdata->getDomainDecomposition())
        {
        std::shared_ptr<mpcd::Communicator> mpcd_comm;
#ifdef ENABLE_HIP
        if (m_exec_conf->isCUDAEnabled())
            {
            mpcd_comm = std::make_shared<mpcd::CommunicatorGPU>(sysdef);
            }
        else
#endif // ENABLE_HIP
            {
            mpcd_comm = std::make_shared<mpcd::Communicator>(sysdef);
            }
        setMPCDCommunicator(mpcd_comm);
        }
#endif // ENABLE_MPI
    }

mpcd::Integrator::~Integrator()
    {
    m_exec_conf->msg->notice(5) << "Destroying MPCD Integrator" << std::endl;
#ifdef ENABLE_MPI
    if (m_mpcd_comm)
        m_mpcd_comm->getMigrateRequestSignal()
            .disconnect<mpcd::Integrator, &mpcd::Integrator::checkCollide>(this);
#endif // ENABLE_MPI
    }

/*!
 * \param timestep Current time step of the simulation
 * \post All integration methods previously added with addIntegrationMethod() are applied in order
 * to move the system state variables forward to \a timestep+1. \post Internally, all forces added
 * via Integrator::addForceCompute are evaluated at \a timestep+1
 */
void mpcd::Integrator::update(uint64_t timestep)
    {
    // remove leftover virtual particles, communicate MPCD particles, and refill
    if (checkCollide(timestep))
        {
        m_sysdef->getMPCDParticleData()->removeVirtualParticles();
        }

#ifdef ENABLE_MPI
    if (m_mpcd_comm)
        m_mpcd_comm->communicate(timestep);
#endif // ENABLE_MPI

    // fill in any virtual particles
    if (checkCollide(timestep))
        {
        for (auto& filler : m_fillers)
            {
            filler->fill(timestep);
            }
        }

    // optionally sort for performance
    if (m_sorter && (*m_sorter->getTrigger())(timestep))
        m_sorter->update(timestep);

    // perform the core MPCD steps of collision and streaming
    if (m_collide)
        m_collide->collide(timestep);
    if (m_stream)
        m_stream->stream(timestep);

    // execute MD steps
    md::IntegratorTwoStep::update(timestep);
    }

/*!
 * \param deltaT new deltaT to set
 * \post \a deltaT is also set on all contained integration methods
 */
void mpcd::Integrator::setDeltaT(Scalar deltaT)
    {
    md::IntegratorTwoStep::setDeltaT(deltaT);
    if (m_stream)
        m_stream->setDeltaT(deltaT);
    }

/*!
 * Compute accelerations if needed for the first step.
 * If acceleration is available in the restart file, then just call computeNetForce so that
 * net_force and net_virial are available for the logger. This solves ticket #393
 */
void mpcd::Integrator::prepRun(uint64_t timestep)
    {
    md::IntegratorTwoStep::prepRun(timestep);

    // synchronize cell list in mpcd methods
    syncCellList();
    if (m_cl)
        {
        m_cl->drawGridShift(timestep);
        }

#ifdef ENABLE_MPI
    // force a communication step if present
    if (m_mpcd_comm)
        {
        m_mpcd_comm->communicate(timestep);
        }
#endif // ENABLE_MPI
    }

void mpcd::Integrator::startAutotuning()
    {
    md::IntegratorTwoStep::startAutotuning();

    m_sysdef->getMPCDParticleData()->startAutotuning();

    if (m_collide)
        {
        m_collide->startAutotuning();
        }
    if (m_stream)
        {
        m_stream->startAutotuning();
        }
    if (m_sorter)
        {
        m_sorter->startAutotuning();
        }
#ifdef ENABLE_MPI
    if (m_mpcd_comm)
        {
        m_mpcd_comm->startAutotuning();
        }
#endif
    for (auto& filler : m_fillers)
        {
        filler->startAutotuning();
        }
    }

bool mpcd::Integrator::isAutotuningComplete()
    {
    bool result = md::IntegratorTwoStep::isAutotuningComplete();

    result = result && m_sysdef->getMPCDParticleData()->isAutotuningComplete();

    if (m_collide)
        {
        result = result && m_collide->isAutotuningComplete();
        }
    if (m_stream)
        {
        result = result && m_stream->isAutotuningComplete();
        }
    if (m_sorter)
        {
        result = result && m_sorter->isAutotuningComplete();
        }
#ifdef ENABLE_MPI
    if (m_mpcd_comm)
        {
        result = result && m_mpcd_comm->isAutotuningComplete();
        }
#endif
    for (auto& filler : m_fillers)
        {
        result = result && filler->isAutotuningComplete();
        }
    return result;
    }

void mpcd::Integrator::syncCellList()
    {
    if (m_collide)
        {
        m_collide->setCellList(m_cl);
        }
    if (m_stream)
        {
        m_stream->setCellList(m_cl);
        }
    if (m_sorter)
        {
        m_sorter->setCellList(m_cl);
        }
#ifdef ENABLE_MPI
    if (m_mpcd_comm)
        {
        m_mpcd_comm->setCellList(m_cl);
        }
#endif
    for (auto& filler : m_fillers)
        {
        filler->setCellList(m_cl);
        }
    }

namespace mpcd
    {
namespace detail
    {
/*!
 * \param m Python module to export to
 */
void export_Integrator(pybind11::module& m)
    {
    pybind11::bind_vector<std::vector<std::shared_ptr<mpcd::VirtualParticleFiller>>>(
        m,
        "VirtualParticleFillerList");

    pybind11::class_<mpcd::Integrator,
                     hoomd::md::IntegratorTwoStep,
                     std::shared_ptr<mpcd::Integrator>>(m, "Integrator")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, Scalar>())
        .def_property("cell_list", &mpcd::Integrator::getCellList, &mpcd::Integrator::setCellList)
        .def_property("collision_method",
                      &mpcd::Integrator::getCollisionMethod,
                      &mpcd::Integrator::setCollisionMethod)
        .def_property("streaming_method",
                      &mpcd::Integrator::getStreamingMethod,
                      &mpcd::Integrator::setStreamingMethod)
        .def_property("mpcd_particle_sorter",
                      &mpcd::Integrator::getSorter,
                      &mpcd::Integrator::setSorter)
        .def_property_readonly("fillers", &mpcd::Integrator::getFillers);
    }
    } // namespace detail
    } // namespace mpcd
    } // end namespace hoomd
