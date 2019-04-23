// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/CollisionMethod.h
 * \brief Definition of mpcd::CollisionMethod
 */

#include "CollisionMethod.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/RNGIdentifiers.h"

/*!
 * \param sysdata MPCD system data
 * \param cur_timestep Current system timestep
 * \param period Number of timesteps between collisions
 * \param phase Phase shift for periodic updates
 * \param seed Seed to pseudo-random number generator
 */
mpcd::CollisionMethod::CollisionMethod(std::shared_ptr<mpcd::SystemData> sysdata,
                                       unsigned int cur_timestep,
                                       unsigned int period,
                                       int phase,
                                       unsigned int seed)
    : m_mpcd_sys(sysdata),
      m_sysdef(m_mpcd_sys->getSystemDefinition()),
      m_pdata(m_sysdef->getParticleData()),
      m_mpcd_pdata(m_mpcd_sys->getParticleData()),
      m_exec_conf(m_pdata->getExecConf()),
      m_cl(m_mpcd_sys->getCellList()),
      m_period(period), m_seed(seed), m_enable_grid_shift(true)
    {
    // setup next timestep for collision
    m_next_timestep = cur_timestep;
    if (phase >= 0)
        {
        // determine next step that is in line with period + phase
        unsigned int multiple = cur_timestep / m_period + (cur_timestep % m_period != 0);
        m_next_timestep = multiple * m_period + phase;
        }

    #ifdef ENABLE_MPI
    // synchronize seed from root across all ranks in MPI in case users has seeded from system time or entropy
    if (m_exec_conf->getNRanks() > 1)
        {
        bcast(m_seed, 0, m_exec_conf->getMPICommunicator());
        }
    #endif // ENABLE_MPI
    }

void mpcd::CollisionMethod::collide(unsigned int timestep)
    {
    if (!shouldCollide(timestep)) return;

    if (m_prof) m_prof->push("MPCD collide");
    // set random grid shift
    drawGridShift(timestep);
    if (m_prof) m_prof->pop();

    // update cell list
    m_cl->compute(timestep);

    rule(timestep);
    }

/*!
 * \param timestep Current timestep
 * \returns True when \a timestep is a \a m_period multiple of the the next timestep the collision should occur
 *
 * Using a multiple allows the collision method to be disabled and then reenabled later if the \a timestep has already
 * exceeded the \a m_next_timestep.
 */
bool mpcd::CollisionMethod::peekCollide(unsigned int timestep) const
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
 * The collision method period is updated to \a period only if collision would occur at \a cur_timestep.
 * It is the caller's responsibility to ensure this condition is valid.
 */
void mpcd::CollisionMethod::setPeriod(unsigned int cur_timestep, unsigned int period)
    {
    if (!peekCollide(cur_timestep))
        {
        m_exec_conf->msg->error() << "MPCD CollisionMethod period can only be changed on multiple of original period" << std::endl;
        throw std::runtime_error("Collision period can only be changed on multiple of original period");
        }

    // try to update the period
    const unsigned int old_period = m_period;
    m_period = period;

    // validate the new period, resetting to the old one before erroring out if it doesn't match
    if (!peekCollide(cur_timestep))
        {
        m_period = old_period;
        m_exec_conf->msg->error() << "MPCD CollisionMethod period can only be changed on multiple of new period" << std::endl;
        throw std::runtime_error("Collision period can only be changed on multiple of new period");
        }
    }

/*!
 * \param timestep Current timestep
 * \returns True when \a timestep is a \a m_period multiple of the the next timestep the collision should occur
 *
 * \post The next timestep is also advanced to the next timestep the collision should occur after \a timestep.
 *       If this behavior is not desired, then use peekCollide() instead.
 */
bool mpcd::CollisionMethod::shouldCollide(unsigned int timestep)
    {
    if (peekCollide(timestep))
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
 * \param timestep Timestep to set shifting for
 *
 * \post The MPCD cell list has its grid shift set for \a timestep.
 *
 * If grid shifting is enabled, three uniform random numbers are drawn using
 * the Mersenne twister generator. (In two dimensions, only two numbers are drawn.)
 *
 * If grid shifting is disabled, a zero vector is instead set.
 */
void mpcd::CollisionMethod::drawGridShift(unsigned int timestep)
    {
    // return zeros if shifting is off
    if (!m_enable_grid_shift)
        {
        m_cl->setGridShift(make_scalar3(0.0,0.0,0.0));
        }
    else
        {
        // PRNG using seed and timestep as seeds
        hoomd::RandomGenerator rng(hoomd::RNGIdentifier::CollisionMethod, m_seed, timestep / m_period);
        const Scalar max_shift = m_cl->getMaxGridShift();

        // draw shift variables from uniform distribution
        Scalar3 shift;
        hoomd::UniformDistribution<Scalar> uniform(-max_shift, max_shift);
        shift.x = uniform(rng);
        shift.y = uniform(rng);
        shift.z = (m_sysdef->getNDimensions() == 3) ? uniform(rng) : Scalar(0.0);

        m_cl->setGridShift(shift);
        }
    }

/*!
 * \param m Python module to export to
 */
void mpcd::detail::export_CollisionMethod(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<mpcd::CollisionMethod, std::shared_ptr<mpcd::CollisionMethod> >(m, "CollisionMethod")
        .def(py::init<std::shared_ptr<mpcd::SystemData>, unsigned int, unsigned int, int, unsigned int>())
        .def("enableGridShifting", &mpcd::CollisionMethod::enableGridShifting)
        .def("setEmbeddedGroup", &mpcd::CollisionMethod::setEmbeddedGroup)
        .def("setPeriod", &mpcd::CollisionMethod::setPeriod);
    }
