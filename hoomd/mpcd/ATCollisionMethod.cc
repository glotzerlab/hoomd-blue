// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/ATCollisionMethod.h
 * \brief Definition of mpcd::ATCollisionMethod
 */

#include "ATCollisionMethod.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"

namespace hoomd
    {
mpcd::ATCollisionMethod::ATCollisionMethod(std::shared_ptr<SystemDefinition> sysdef,
                                           uint64_t cur_timestep,
                                           uint64_t period,
                                           int phase,
                                           std::shared_ptr<Variant> T)
    : mpcd::CollisionMethod(sysdef, cur_timestep, period, phase), m_T(T)
    {
    m_exec_conf->msg->notice(5) << "Constructing MPCD AT collision method" << std::endl;
    }

mpcd::ATCollisionMethod::~ATCollisionMethod()
    {
    m_exec_conf->msg->notice(5) << "Destroying MPCD AT collision method" << std::endl;
    detachCallbacks();
    }

/*!
 * \param timestep Current timestep.
 */
void mpcd::ATCollisionMethod::rule(uint64_t timestep)
    {
    m_thermo->compute(timestep);

    // compute the cell average of the random velocities
    m_pdata->swapVelocities();
    m_mpcd_pdata->swapVelocities();
    m_rand_thermo->compute(timestep);
    m_pdata->swapVelocities();
    m_mpcd_pdata->swapVelocities();

    // apply random velocities
    applyVelocities();
    }

/*!
 * \param timestep Current timestep.
 */
void mpcd::ATCollisionMethod::drawVelocities(uint64_t timestep)
    {
    // mpcd particle data
    ArrayHandle<unsigned int> h_tag(m_mpcd_pdata->getTags(),
                                    access_location::host,
                                    access_mode::read);
    ArrayHandle<Scalar4> h_alt_vel(m_mpcd_pdata->getAltVelocities(),
                                   access_location::host,
                                   access_mode::overwrite);
    const unsigned int N_mpcd = m_mpcd_pdata->getN() + m_mpcd_pdata->getNVirtual();
    unsigned int N_tot = N_mpcd;

    // embedded particle data
    std::unique_ptr<ArrayHandle<unsigned int>> h_embed_idx;
    std::unique_ptr<ArrayHandle<Scalar4>> h_vel_embed;
    std::unique_ptr<ArrayHandle<Scalar4>> h_alt_vel_embed;
    std::unique_ptr<ArrayHandle<unsigned int>> h_tag_embed;
    if (m_embed_group)
        {
        h_embed_idx.reset(new ArrayHandle<unsigned int>(m_embed_group->getIndexArray(),
                                                        access_location::host,
                                                        access_mode::read));
        h_vel_embed.reset(new ArrayHandle<Scalar4>(m_pdata->getVelocities(),
                                                   access_location::host,
                                                   access_mode::read));
        h_alt_vel_embed.reset(new ArrayHandle<Scalar4>(m_pdata->getAltVelocities(),
                                                       access_location::host,
                                                       access_mode::overwrite));
        h_tag_embed.reset(new ArrayHandle<unsigned int>(m_pdata->getTags(),
                                                        access_location::host,
                                                        access_mode::read));
        N_tot += m_embed_group->getNumMembers();
        }

    uint16_t seed = m_sysdef->getSeed();

    // random velocities are drawn for each particle and stored into the "alternate" arrays
    const Scalar T = (*m_T)(timestep);
    for (unsigned int idx = 0; idx < N_tot; ++idx)
        {
        unsigned int pidx;
        unsigned int tag;
        Scalar mass;
        if (idx < N_mpcd)
            {
            pidx = idx;
            mass = m_mpcd_pdata->getMass();
            tag = h_tag.data[idx];
            }
        else
            {
            pidx = h_embed_idx->data[idx - N_mpcd];
            mass = h_vel_embed->data[pidx].w;
            tag = h_tag_embed->data[pidx];
            }

        // draw random velocities from normal distribution
        hoomd::RandomGenerator rng(
            hoomd::Seed(hoomd::RNGIdentifier::ATCollisionMethod, timestep, seed),
            hoomd::Counter(tag));
        hoomd::NormalDistribution<Scalar> gen(fast::sqrt(T / mass), 0.0);
        Scalar3 vel;
        gen(vel.x, vel.y, rng);
        vel.z = gen(rng);

        // save out velocities
        if (idx < N_mpcd)
            {
            h_alt_vel.data[pidx]
                = make_scalar4(vel.x, vel.y, vel.z, __int_as_scalar(mpcd::detail::NO_CELL));
            }
        else
            {
            h_alt_vel_embed->data[pidx] = make_scalar4(vel.x, vel.y, vel.z, mass);
            }
        }
    }

void mpcd::ATCollisionMethod::applyVelocities()
    {
    // mpcd particle data
    ArrayHandle<Scalar4> h_vel(m_mpcd_pdata->getVelocities(),
                               access_location::host,
                               access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel_alt(m_mpcd_pdata->getAltVelocities(),
                                   access_location::host,
                                   access_mode::read);
    const unsigned int N_mpcd = m_mpcd_pdata->getN() + m_mpcd_pdata->getNVirtual();
    unsigned int N_tot = N_mpcd;

    // embedded particle data
    std::unique_ptr<ArrayHandle<unsigned int>> h_embed_idx;
    std::unique_ptr<ArrayHandle<Scalar4>> h_vel_embed;
    std::unique_ptr<ArrayHandle<Scalar4>> h_vel_alt_embed;
    std::unique_ptr<ArrayHandle<unsigned int>> h_embed_cell_ids;
    if (m_embed_group)
        {
        h_embed_idx.reset(new ArrayHandle<unsigned int>(m_embed_group->getIndexArray(),
                                                        access_location::host,
                                                        access_mode::read));
        h_vel_embed.reset(new ArrayHandle<Scalar4>(m_pdata->getVelocities(),
                                                   access_location::host,
                                                   access_mode::readwrite));
        h_vel_alt_embed.reset(new ArrayHandle<Scalar4>(m_pdata->getAltVelocities(),
                                                       access_location::host,
                                                       access_mode::read));
        h_embed_cell_ids.reset(new ArrayHandle<unsigned int>(m_cl->getEmbeddedGroupCellIds(),
                                                             access_location::host,
                                                             access_mode::read));
        N_tot += m_embed_group->getNumMembers();
        }

    ArrayHandle<double4> h_cell_vel(m_thermo->getCellVelocities(),
                                    access_location::host,
                                    access_mode::read);
    ArrayHandle<double4> h_rand_vel(m_rand_thermo->getCellVelocities(),
                                    access_location::host,
                                    access_mode::read);

    for (unsigned int idx = 0; idx < N_tot; ++idx)
        {
        unsigned int cell, pidx;
        Scalar4 vel_rand;
        if (idx < N_mpcd)
            {
            pidx = idx;
            const Scalar4 vel_cell = h_vel.data[idx];
            cell = __scalar_as_int(vel_cell.w);
            vel_rand = h_vel_alt.data[idx];
            }
        else
            {
            pidx = h_embed_idx->data[idx - N_mpcd];
            cell = h_embed_cell_ids->data[idx - N_mpcd];
            vel_rand = h_vel_alt_embed->data[pidx];
            }

        // load cell data
        const double4 v_c = h_cell_vel.data[cell];
        const double4 vrand_c = h_rand_vel.data[cell];

        // compute new velocity using the cell + the random draw
        const Scalar3 vnew = make_scalar3(v_c.x - vrand_c.x + vel_rand.x,
                                          v_c.y - vrand_c.y + vel_rand.y,
                                          v_c.z - vrand_c.z + vel_rand.z);

        if (idx < N_mpcd)
            {
            h_vel.data[pidx] = make_scalar4(vnew.x, vnew.y, vnew.z, __int_as_scalar(cell));
            }
        else
            {
            h_vel_embed->data[pidx] = make_scalar4(vnew.x, vnew.y, vnew.z, vel_rand.w);
            }
        }
    }

void mpcd::ATCollisionMethod::setCellList(std::shared_ptr<mpcd::CellList> cl)
    {
    if (cl != m_cl)
        {
        CollisionMethod::setCellList(cl);

        detachCallbacks();
        if (m_cl)
            {
            m_thermo = std::make_shared<mpcd::CellThermoCompute>(m_sysdef, m_cl);
            m_rand_thermo = std::make_shared<mpcd::CellThermoCompute>(m_sysdef, m_cl);
            attachCallbacks();
            }
        else
            {
            m_thermo = std::shared_ptr<mpcd::CellThermoCompute>();
            m_rand_thermo = std::shared_ptr<mpcd::CellThermoCompute>();
            }
        }
    }

void mpcd::ATCollisionMethod::attachCallbacks()
    {
    assert(m_thermo);
    m_thermo->getCallbackSignal()
        .connect<mpcd::ATCollisionMethod, &mpcd::ATCollisionMethod::drawVelocities>(this);
    }

void mpcd::ATCollisionMethod::detachCallbacks()
    {
    if (m_thermo)
        {
        m_thermo->getCallbackSignal()
            .disconnect<mpcd::ATCollisionMethod, &mpcd::ATCollisionMethod::drawVelocities>(this);
        }
    }

namespace mpcd
    {
namespace detail
    {
/*!
 * \param m Python module to export to
 */
void export_ATCollisionMethod(pybind11::module& m)
    {
    pybind11::class_<mpcd::ATCollisionMethod,
                     mpcd::CollisionMethod,
                     std::shared_ptr<mpcd::ATCollisionMethod>>(m, "ATCollisionMethod")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            uint64_t,
                            uint64_t,
                            int,
                            std::shared_ptr<Variant>>())
        .def_property("kT",
                      &mpcd::ATCollisionMethod::getTemperature,
                      &mpcd::ATCollisionMethod::setTemperature);
    }
    } // namespace detail
    } // namespace mpcd
    } // end namespace hoomd
