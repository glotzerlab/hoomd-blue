// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/Integrator.cc
 * \brief Defines the mpcd::Integrator
 */

#include "Integrator.h"

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#endif

namespace hoomd
    {
/*!
 * \param sysdata MPCD system data
 * \param deltaT Fundamental integration timestep
 */
mpcd::Integrator::Integrator(std::shared_ptr<mpcd::SystemData> sysdata, Scalar deltaT)
    : IntegratorTwoStep(sysdata->getSystemDefinition(), deltaT), m_mpcd_sys(sysdata)
    {
    assert(m_mpcd_sys);
    m_exec_conf->msg->notice(5) << "Constructing MPCD Integrator" << std::endl;
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
    IntegratorTwoStep::update(timestep);
    // issue a warning if no integration methods are set
    if (!m_gave_warning && m_methods.size() == 0 && !m_stream)
        {
        m_exec_conf->msg->warning()
            << "mpcd.integrate: No integration methods are set." << std::endl;
        m_gave_warning = true;
        }

    // remove any leftover virtual particles
    if (checkCollide(timestep))
        {
        m_mpcd_sys->getParticleData()->removeVirtualParticles();
        m_collide->drawGridShift(timestep);
        }

#ifdef ENABLE_MPI
    if (m_mpcd_comm)
        m_mpcd_comm->communicate(timestep);
#endif // ENABLE_MPI

    // fill in any virtual particles
    if (checkCollide(timestep) && !m_fillers.empty())
        {
        for (auto filler = m_fillers.begin(); filler != m_fillers.end(); ++filler)
            {
            (*filler)->fill(timestep);
            }
        }

    // optionally sort
    if (m_sorter)
        m_sorter->update(timestep);

    // call the MPCD collision rule before the first MD step so that any embedded velocities are
    // updated first
    if (m_collide)
        m_collide->collide(timestep);

    // perform the first MD integration step
    for (auto method = m_methods.begin(); method != m_methods.end(); ++method)
        (*method)->integrateStepOne(timestep);

// MD communication / rigid body updates
#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        m_comm->communicate(timestep + 1);
        }
    else
#endif // ENABLE_MPI
        {
        updateRigidBodies(timestep + 1);
        }

    // execute the MPCD streaming step now that MD particles are communicated onto their final
    // domains
    if (m_stream)
        {
        m_stream->stream(timestep);
        }

    // compute the net force on the MD particles
#ifdef ENABLE_HIP
    if (m_exec_conf->isCUDAEnabled())
        computeNetForceGPU(timestep + 1);
    else
#endif
        computeNetForce(timestep + 1);

    // perform the second step of the MD integration
    for (auto method = m_methods.begin(); method != m_methods.end(); ++method)
        (*method)->integrateStepTwo(timestep);
    }

/*!
 * \param deltaT new deltaT to set
 * \post \a deltaT is also set on all contained integration methods
 */
void mpcd::Integrator::setDeltaT(Scalar deltaT)
    {
    IntegratorTwoStep::setDeltaT(deltaT);
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
    IntegratorTwoStep::prepRun(timestep);

    // synchronize timestep in mpcd methods
    if (m_collide)
        {
        m_collide->drawGridShift(timestep);
        }

#ifdef ENABLE_MPI
    // force a communication step if present
    if (m_mpcd_comm)
        {
        m_mpcd_comm->communicate(timestep);
        }
#endif // ENABLE_MPI
    }

/// Start autotuning kernel launch parameters
void mpcd::Integrator::startAutotuning()
    {
    IntegratorTwoStep::startAutotuning();
    m_mpcd_sys->startAutotuning();
    if (m_collide)
        m_collide->startAutotuning();
    if (m_stream)
        m_stream->startAutotuning();
    if (m_sorter)
        m_sorter->startAutotuning();
    }

/*!
 * \param filler Virtual particle filler to add to the integrator
 *
 * The \a filler is attached to the integrator exactly once. An error is raised if this filler has
 * already been added.
 */
void mpcd::Integrator::addFiller(std::shared_ptr<mpcd::VirtualParticleFiller> filler)
    {
    auto it = std::find(m_fillers.begin(), m_fillers.end(), filler);
    if (it != m_fillers.end())
        {
        m_exec_conf->msg->error()
            << "Trying to add same MPCD virtual particle filler twice! Please report this bug."
            << std::endl;
        throw std::runtime_error("Duplicate attachment of MPCD virtual particle filler");
        }

    m_fillers.push_back(filler);
    }

/*!
 * \param m Python module to export to
 */
void mpcd::detail::export_Integrator(pybind11::module& m)
    {
    pybind11::class_<mpcd::Integrator,
                     hoomd::md::IntegratorTwoStep,
                     std::shared_ptr<mpcd::Integrator>>(m, "Integrator")
        .def(pybind11::init<std::shared_ptr<mpcd::SystemData>, Scalar>())
        .def("setCollisionMethod", &mpcd::Integrator::setCollisionMethod)
        .def("removeCollisionMethod", &mpcd::Integrator::removeCollisionMethod)
        .def("setStreamingMethod", &mpcd::Integrator::setStreamingMethod)
        .def("removeStreamingMethod", &mpcd::Integrator::removeStreamingMethod)
        .def("setSorter", &mpcd::Integrator::setSorter)
        .def("removeSorter", &mpcd::Integrator::removeSorter)
        .def("addFiller", &mpcd::Integrator::addFiller)
        .def("removeAllFillers", &mpcd::Integrator::removeAllFillers)
#ifdef ENABLE_MPI
        .def("setMPCDCommunicator", &mpcd::Integrator::setMPCDCommunicator)
#endif // ENABLE_MPI
        ;
    }
    } // end namespace hoomd
