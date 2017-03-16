// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/StreamingMethod.cc
 * \brief Definition of mpcd::StreamingMethod
 */

#include "StreamingMethod.h"

/*!
 * \param sysdata MPCD system data
 * \param cur_timestep Current system timestep
 * \param period Number of timesteps between collisions
 * \param phase Phase shift for periodic updates
 */
mpcd::StreamingMethod::StreamingMethod(std::shared_ptr<mpcd::SystemData> sysdata,
                                       unsigned int cur_timestep,
                                       unsigned int period,
                                       int phase)
    : m_mpcd_sys(sysdata),
      m_sysdef(m_mpcd_sys->getSystemDefinition()),
      m_pdata(m_sysdef->getParticleData()),
      m_mpcd_pdata(m_mpcd_sys->getParticleData()),
      m_exec_conf(m_pdata->getExecConf()),
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
 * \param timestep Current time to stream
 */
void mpcd::StreamingMethod::stream(unsigned int timestep)
    {
    if (!shouldStream(timestep)) return;

    if (m_prof) m_prof->push("stream");

    const BoxDim& box = m_pdata->getBox();

    ArrayHandle<Scalar4> h_pos(m_mpcd_pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(m_mpcd_pdata->getVelocities(), access_location::host, access_mode::read);

    for (unsigned int cur_p = 0; cur_p < m_mpcd_pdata->getN(); ++cur_p)
        {
        const Scalar4 postype = h_pos.data[cur_p];
        Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
        const unsigned int type = __scalar_as_int(postype.w);

        const Scalar4 vel_cell = h_vel.data[cur_p];
        const Scalar3 vel = make_scalar3(vel_cell.x, vel_cell.y, vel_cell.z);

        // propagate the particle to its new position ballistically
        pos += m_mpcd_dt * vel;

        // wrap and update the position
        int3 image = make_int3(0,0,0);
        box.wrap(pos, image);

        h_pos.data[cur_p] = make_scalar4(pos.x, pos.y, pos.z, __int_as_scalar(type));
        }
    if (m_prof) m_prof->pop();

    // particles have moved, so the cell cache is no longer valid
    m_mpcd_pdata->invalidateCellCache();
    }

/*!
 * \param timestep Current timestep
 * \returns True when \a timestep is equal to the next timestep the streaming should occur
 */
bool mpcd::StreamingMethod::peekStream(unsigned int timestep) const
    {
    return (timestep == m_next_timestep);
    }

/*!
 * \param timestep Current timestep
 * \returns True when \a timestep is equal to the next timestep the streaming should occur
 *
 * \post The next timestep is also advanced. If this behavior is not desired, then
 *       use peekStream() instead.
 */
bool mpcd::StreamingMethod::shouldStream(unsigned int timestep)
    {
    if (timestep == m_next_timestep)
        {
        m_next_timestep += m_period;
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
    namespace py = pybind11;
    py::class_<mpcd::StreamingMethod, std::shared_ptr<mpcd::StreamingMethod> >(m, "StreamingMethod")
        .def(py::init<std::shared_ptr<mpcd::SystemData>, unsigned int, unsigned int, int>());
    }
