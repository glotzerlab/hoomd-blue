// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/CollisionMethod.h
 * \brief Definition of mpcd::CollisionMethod
 */

#include "CollisionMethod.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"

namespace hoomd
    {
/*!
 * \param sysdef System definition
 * \param cur_timestep Current system timestep
 * \param period Number of timesteps between collisions
 * \param phase Phase shift for periodic updates
 * \param seed Seed to pseudo-random number generator
 */
mpcd::CollisionMethod::CollisionMethod(std::shared_ptr<SystemDefinition> sysdef,
                                       uint64_t cur_timestep,
                                       uint64_t period,
                                       int phase)
    : m_sysdef(sysdef), m_pdata(m_sysdef->getParticleData()),
      m_mpcd_pdata(sysdef->getMPCDParticleData()), m_exec_conf(m_pdata->getExecConf()),
      m_period(period), m_enable_grid_shift(true)
    {
    // setup next timestep for collision
    m_next_timestep = cur_timestep;
    if (phase >= 0)
        {
        // determine next step that is in line with period + phase
        uint64_t multiple = cur_timestep / m_period + (cur_timestep % m_period != 0);
        m_next_timestep = multiple * m_period + phase;
        }
    }

void mpcd::CollisionMethod::collide(uint64_t timestep)
    {
    if (!shouldCollide(timestep))
        return;

    if (!m_cl)
        {
        throw std::runtime_error("Cell list has not been set");
        }

    // sync the embedded group
    m_cl->setEmbeddedGroup(m_embed_group);

    // set random grid shift
    drawGridShift(timestep);

    // update cell list
    m_cl->compute(timestep);

    rule(timestep);
    }

/*!
 * \param timestep Current timestep
 * \returns True when \a timestep is a \a m_period multiple of the the next timestep the collision
 * should occur
 *
 * Using a multiple allows the collision method to be disabled and then reenabled later if the \a
 * timestep has already exceeded the \a m_next_timestep.
 */
bool mpcd::CollisionMethod::peekCollide(uint64_t timestep) const
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
 * The collision method period is updated to \a period only if collision would occur at \a
 * cur_timestep. It is the caller's responsibility to ensure this condition is valid.
 */
void mpcd::CollisionMethod::setPeriod(unsigned int cur_timestep, unsigned int period)
    {
    if (!peekCollide(cur_timestep))
        {
        m_exec_conf->msg->error()
            << "MPCD CollisionMethod period can only be changed on multiple of original period"
            << std::endl;
        throw std::runtime_error(
            "Collision period can only be changed on multiple of original period");
        }

    // try to update the period
    const uint64_t old_period = m_period;
    m_period = period;

    // validate the new period, resetting to the old one before erroring out if it doesn't match
    if (!peekCollide(cur_timestep))
        {
        m_period = old_period;
        m_exec_conf->msg->error()
            << "MPCD CollisionMethod period can only be changed on multiple of new period"
            << std::endl;
        throw std::runtime_error("Collision period can only be changed on multiple of new period");
        }
    }

/*!
 * \param timestep Current timestep
 * \returns True when \a timestep is a \a m_period multiple of the the next timestep the collision
 * should occur
 *
 * \post The next timestep is also advanced to the next timestep the collision should occur after \a
 * timestep. If this behavior is not desired, then use peekCollide() instead.
 */
bool mpcd::CollisionMethod::shouldCollide(uint64_t timestep)
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
void mpcd::CollisionMethod::drawGridShift(uint64_t timestep)
    {
    uint16_t seed = m_sysdef->getSeed();

    // return zeros if shifting is off
    if (!m_enable_grid_shift)
        {
        m_cl->setGridShift(make_scalar3(0.0, 0.0, 0.0));
        }
    else
        {
        // PRNG using seed and timestep as seeds
        hoomd::RandomGenerator rng(
            hoomd::Seed(hoomd::RNGIdentifier::CollisionMethod, timestep, seed),
            hoomd::Counter(m_instance));
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
    pybind11::class_<mpcd::CollisionMethod, std::shared_ptr<mpcd::CollisionMethod>>(
        m,
        "CollisionMethod")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, uint64_t, uint64_t, int>())
        .def("enableGridShifting", &mpcd::CollisionMethod::enableGridShifting)
        .def("setEmbeddedGroup", &mpcd::CollisionMethod::setEmbeddedGroup)
        .def("setPeriod", &mpcd::CollisionMethod::setPeriod)
        .def_property("instance",
                      &mpcd::CollisionMethod::getInstance,
                      &mpcd::CollisionMethod::setInstance);
    }

    } // end namespace hoomd
