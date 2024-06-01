// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/Sorter.cc
 * \brief Defines the mpcd::Sorter
 */

#include "Sorter.h"

namespace hoomd
    {
/*!
 * \param sysdef System definition
 */
mpcd::Sorter::Sorter(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<Trigger> trigger)
    : Tuner(sysdef, trigger), m_mpcd_pdata(m_sysdef->getMPCDParticleData()), m_order(m_exec_conf),
      m_rorder(m_exec_conf)
    {
    m_exec_conf->msg->notice(5) << "Constructing MPCD Sorter" << std::endl;
    }

mpcd::Sorter::~Sorter()
    {
    m_exec_conf->msg->notice(5) << "Destroying MPCD Sorter" << std::endl;
    }

/*!
 * \param timestep Current simulation timestep
 *
 * This method is just a driver for the computeOrder() and applyOrder() methods.
 */
void mpcd::Sorter::update(uint64_t timestep)
    {
    if (!m_cl)
        {
        throw std::runtime_error("Cell list has not been set");
        }

    // resize the sorted order vector to the current number of particles
    m_order.resize(m_mpcd_pdata->getN());
    m_rorder.resize(m_mpcd_pdata->getN());

    // generate and apply the sorted order
    computeOrder(timestep);
    applyOrder();

    // trigger the sort signal for ParticleData callbacks using the current sortings
    m_mpcd_pdata->notifySort(timestep, m_order, m_rorder);
    }

/*!
 * \param timestep Current timestep
 *
 * Loop through the computed cell list and generate a compacted list of the order
 * particles appear. This will put the particles into a cell-list order, which
 * should be more friendly for other MPCD cell-based operations.
 */
void mpcd::Sorter::computeOrder(uint64_t timestep)
    {
    // compute the cell list at current timestep, guarantees owned particles are on rank
    m_cl->compute(timestep);

    ArrayHandle<unsigned int> h_cell_list(m_cl->getCellList(),
                                          access_location::host,
                                          access_mode::read);
    ArrayHandle<unsigned int> h_cell_np(m_cl->getCellSizeArray(),
                                        access_location::host,
                                        access_mode::read);
    const Index2D& cli = m_cl->getCellListIndexer();

    // loop through the cell list to generate the sorting order for MPCD particles
    ArrayHandle<unsigned int> h_order(m_order, access_location::host, access_mode::overwrite);
    ArrayHandle<unsigned int> h_rorder(m_rorder, access_location::host, access_mode::overwrite);
    const unsigned int N_mpcd = m_mpcd_pdata->getN();
    unsigned int cur_p = 0;
    for (unsigned int idx = 0; idx < m_cl->getNCells(); ++idx)
        {
        const unsigned int np = h_cell_np.data[idx];
        for (unsigned int offset = 0; offset < np; ++offset)
            {
            const unsigned int pid = h_cell_list.data[cli(offset, idx)];
            // only count MPCD particles, and skip embedded particles
            if (pid < N_mpcd)
                {
                h_order.data[cur_p] = pid;
                h_rorder.data[pid] = cur_p;
                ++cur_p;
                }
            }
        }
    }

/*!
 * Loop through the ordered set of particles, and apply the sorted order. This is
 * intentionally broken out from computeOrder() so that other sorting rules could
 * be implemented without having to duplicate the application of the sort.
 *
 * The sorted order is applied by swapping out the alternate per-particle data
 * arrays. The communication flags are \b not sorted in MPI because by design,
 * the caller is responsible for clearing out any old flags before using them.
 */
void mpcd::Sorter::applyOrder() const
    {
        // apply the sorted order
        {
        ArrayHandle<unsigned int> h_order(m_order, access_location::host, access_mode::read);

        ArrayHandle<Scalar4> h_pos(m_mpcd_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::read);
        ArrayHandle<Scalar4> h_vel(m_mpcd_pdata->getVelocities(),
                                   access_location::host,
                                   access_mode::read);
        ArrayHandle<unsigned int> h_tag(m_mpcd_pdata->getTags(),
                                        access_location::host,
                                        access_mode::read);

        ArrayHandle<Scalar4> h_pos_alt(m_mpcd_pdata->getAltPositions(),
                                       access_location::host,
                                       access_mode::overwrite);
        ArrayHandle<Scalar4> h_vel_alt(m_mpcd_pdata->getAltVelocities(),
                                       access_location::host,
                                       access_mode::overwrite);
        ArrayHandle<unsigned int> h_tag_alt(m_mpcd_pdata->getAltTags(),
                                            access_location::host,
                                            access_mode::overwrite);

        for (unsigned int idx = 0; idx < m_mpcd_pdata->getN(); ++idx)
            {
            const unsigned int old_idx = h_order.data[idx];
            h_pos_alt.data[idx] = h_pos.data[old_idx];
            h_vel_alt.data[idx] = h_vel.data[old_idx];
            h_tag_alt.data[idx] = h_tag.data[old_idx];
            }

        // copy virtual particle data if it exists
        if (m_mpcd_pdata->getNVirtual() > 0)
            {
            const unsigned int N = m_mpcd_pdata->getN();
            const unsigned int Ntot = N + m_mpcd_pdata->getNVirtual();
            std::copy(h_pos.data + N, h_pos.data + Ntot, h_pos_alt.data + N);
            std::copy(h_vel.data + N, h_vel.data + Ntot, h_vel_alt.data + N);
            std::copy(h_tag.data + N, h_tag.data + Ntot, h_tag_alt.data + N);
            }
        }

    // swap out sorted data
    m_mpcd_pdata->swapPositions();
    m_mpcd_pdata->swapVelocities();
    m_mpcd_pdata->swapTags();
    }

/*!
 * \param m Python module to export to
 */
void mpcd::detail::export_Sorter(pybind11::module& m)
    {
    pybind11::class_<mpcd::Sorter, Tuner, std::shared_ptr<mpcd::Sorter>>(m, "Sorter")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<Trigger>>());
    }

    } // end namespace hoomd
