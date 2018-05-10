// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/Integrator.cc
 * \brief Defines the mpcd::Integrator
 */

#include "Integrator.h"

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#endif

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
    }

/*!
 * \param prof The profiler to set
 * Sets the profiler both for this class and all of the contained integration methods
 */
void mpcd::Integrator::setProfiler(std::shared_ptr<Profiler> prof)
    {
    IntegratorTwoStep::setProfiler(prof);
    m_mpcd_sys->setProfiler(prof);
    if (m_collide)
        m_collide->setProfiler(prof);
    if (m_stream)
        m_stream->setProfiler(prof);
    #ifdef ENABLE_MPI
    if (m_mpcd_comm)
        m_mpcd_comm->setProfiler(prof);
    #endif // ENABLE_MPI
    }

/*!
 * \param timestep Current time step of the simulation
 * \post All integration methods previously added with addIntegrationMethod() are applied in order to move the system
 *       state variables forward to \a timestep+1.
 * \post Internally, all forces added via Integrator::addForceCompute are evaluated at \a timestep+1
 */
void mpcd::Integrator::update(unsigned int timestep)
    {
    // issue a warning if no integration methods are set
    if (!m_gave_warning && m_methods.size() == 0 && !m_stream)
        {
        m_exec_conf->msg->warning() << "mpcd.integrate: No integration methods are set." << std::endl;
        m_gave_warning = true;
        }

    // call the MPCD collision rule before the first MD step so that any embedded
    // velocities are updated first
    if (m_collide)
        {
        m_collide->collide(timestep);
        }

    // perform the first MD integration step
    if (m_prof) m_prof->push("Integrate");
    for (auto method = m_methods.begin(); method != m_methods.end(); ++method)
        (*method)->integrateStepOne(timestep);
    if (m_prof) m_prof->pop();

    // this handles the MD communication
    // TODO: should the MPCD communication step also be called (and typically go unused?)
    #ifdef ENABLE_MPI
    if (m_comm)
        {
        // perform all necessary communication steps. This ensures
        // a) that particles have migrated to the correct domains
        // b) that forces are calculated correctly, if ghost atom positions are updated every time step

        // also updates rigid bodies after ghost updating
        m_comm->communicate(timestep+1);
        }
    else
    #endif // ENABLE_MPI
        {
        updateRigidBodies(timestep+1);
        }

    // execute the MPCD streaming step now that MD particles are communicated onto their final domains
    if (m_stream)
        {
        m_stream->stream(timestep);
        }

    // compute the net force on the MD particles
#ifdef ENABLE_CUDA
    if (m_exec_conf->isCUDAEnabled())
        computeNetForceGPU(timestep+1);
    else
#endif
        computeNetForce(timestep+1);

    // perform the second step of the MD integration
    if (m_prof) m_prof->push("Integrate");
    for (auto method = m_methods.begin(); method != m_methods.end(); ++method)
        (*method)->integrateStepTwo(timestep);
    if (m_prof) m_prof->pop();

    // draw the MPCD grid shift at the next timestep in case analyzers are called in between
    // (note: this is usually a **bad** idea)
    if (m_collide)
        {
        m_collide->drawGridShift(timestep+1);
        }
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
 * If acceleration is available in the restart file, then just call computeNetForce so that net_force and net_virial
 * are available for the logger. This solves ticket #393
*/
void mpcd::Integrator::prepRun(unsigned int timestep)
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

/*! \param enable Enable/disable autotuning
    \param period period (approximate) in time steps when returning occurs
*/
void mpcd::Integrator::setAutotunerParams(bool enable, unsigned int period)
    {
    IntegratorTwoStep::setAutotunerParams(enable, period);
    m_mpcd_sys->setAutotunerParams(enable, period);
    if (m_collide)
        m_collide->setAutotunerParams(enable,period);
    if (m_stream)
        m_stream->setAutotunerParams(enable,period);
    }

/*!
 * \param m Python module to export to
 */
void mpcd::detail::export_Integrator(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<mpcd::Integrator, std::shared_ptr<mpcd::Integrator> >(m, "Integrator", py::base<::IntegratorTwoStep>())
        .def(py::init<std::shared_ptr<mpcd::SystemData>, Scalar>())
        .def("setCollisionMethod", &mpcd::Integrator::setCollisionMethod)
        .def("removeCollisionMethod", &mpcd::Integrator::removeCollisionMethod)
        .def("setStreamingMethod", &mpcd::Integrator::setStreamingMethod)
        .def("removeStreamingMethod", &mpcd::Integrator::removeStreamingMethod)
        #ifdef ENABLE_MPI
        .def("setMPCDCommunicator", &mpcd::Integrator::setMPCDCommunicator)
        #endif // ENABLE_MPI
        ;
    }
