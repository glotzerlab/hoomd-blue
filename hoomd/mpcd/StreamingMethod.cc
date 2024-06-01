// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/StreamingMethod.cc
 * \brief Definition of mpcd::StreamingMethod
 */

#include "StreamingMethod.h"

namespace hoomd
    {
/*!
 * \param sysdef System definition
 * \param cur_timestep Current system timestep
 * \param period Number of timesteps between collisions
 * \param phase Phase shift for periodic updates
 */
mpcd::StreamingMethod::StreamingMethod(std::shared_ptr<SystemDefinition> sysdef,
                                       unsigned int cur_timestep,
                                       unsigned int period,
                                       int phase)
    : m_sysdef(sysdef), m_pdata(m_sysdef->getParticleData()),
      m_mpcd_pdata(m_sysdef->getMPCDParticleData()), m_exec_conf(m_pdata->getExecConf()),
      m_mpcd_dt(0.0), m_period(period)
    {
    m_exec_conf->msg->notice(5) << "Constructing MPCD StreamingMethod" << std::endl;

    // setup next timestep for streaming
    m_next_timestep = cur_timestep;
    if (phase >= 0)
        {
        // determine next step that is in line with period + phase
        unsigned int multiple = cur_timestep / m_period + (cur_timestep % m_period != 0);
        m_next_timestep = multiple * m_period + phase;
        }
    }

mpcd::StreamingMethod::~StreamingMethod()
    {
    m_exec_conf->msg->notice(5) << "Destroying MPCD StreamingMethod" << std::endl;
    }

/*!
 * \param timestep Current timestep
 * \returns True when \a timestep is a \a m_period multiple of the the next timestep the streaming
 * should occur
 *
 * Using a multiple allows the streaming method to be disabled and then reenabled later if the \a
 * timestep has already exceeded the \a m_next_timestep.
 */
bool mpcd::StreamingMethod::peekStream(uint64_t timestep) const
    {
    if (timestep < m_next_timestep)
        return false;
    else
        return ((timestep - m_next_timestep) % m_period == 0);
    }

/*!
 * \param cur_timestep Current simulation timestep
 * \param period New period
 *
 * The streaming method period is updated to \a period only if streaming would occur at \a
 * cur_timestep for both the old period and the new period. It is the caller's responsibility to
 * ensure this condition is valid.
 */
void mpcd::StreamingMethod::setPeriod(unsigned int cur_timestep, unsigned int period)
    {
    if (!peekStream(cur_timestep))
        {
        m_exec_conf->msg->error()
            << "MPCD StreamingMethod period can only be changed on multiple of original period"
            << std::endl;
        throw std::runtime_error(
            "Streaming period can only be changed on multiple of original period");
        }

    // try to update the period
    const unsigned int old_period = m_period;
    m_period = period;

    // validate the new period, resetting to the old one before erroring out if it doesn't match.
    if (!peekStream(cur_timestep))
        {
        m_period = old_period;
        m_exec_conf->msg->error()
            << "MPCD StreamingMethod period can only be changed on multiple of new period"
            << std::endl;
        throw std::runtime_error("Streaming period can only be changed on multiple of new period");
        }
    }

/*!
 * \param timestep Current timestep
 * \returns True when \a timestep is equal to the next timestep the streaming should occur
 *
 * \post The next timestep is also advanced to the next timestep the collision should occur after \a
 * timestep. If this behavior is not desired, then use peekCollide() instead.
 */
bool mpcd::StreamingMethod::shouldStream(uint64_t timestep)
    {
    if (peekStream(timestep))
        {
        m_next_timestep = timestep + m_period;
        return true;
        }
    else
        {
        return false;
        }
    }

/*!
 * \param m Python module to export to
 */
void mpcd::detail::export_StreamingMethod(pybind11::module& m)
    {
    pybind11::class_<mpcd::StreamingMethod, std::shared_ptr<mpcd::StreamingMethod>>(
        m,
        "StreamingMethod")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, unsigned int, unsigned int, int>())
        .def_property_readonly("period", &mpcd::StreamingMethod::getPeriod);
    }

    } // end namespace hoomd
