// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/Sorter.cc
 * \brief Defines the mpcd::Sorter
 */

#include "Sorter.h"

/*!
 * \param sysdata MPCD system data
 */
mpcd::Sorter::Sorter(std::shared_ptr<mpcd::SystemData> sysdata,
                     unsigned int cur_timestep,
                     unsigned int period)
    : m_mpcd_sys(sysdata),
      m_sysdef(m_mpcd_sys->getSystemDefinition()),
      m_pdata(m_sysdef->getParticleData()),
      m_exec_conf(m_pdata->getExecConf()),
      m_mpcd_pdata(m_mpcd_sys->getParticleData()),
      m_cl(m_mpcd_sys->getCellList()),
      m_order(m_exec_conf),
      m_rorder(m_exec_conf),
      m_period(period)
    {
    assert(m_mpcd_sys);
    m_exec_conf->msg->notice(5) << "Constructing MPCD Sorter" << std::endl;

    setPeriod(cur_timestep, period);
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
void mpcd::Sorter::update(unsigned int timestep)
    {
    if (!shouldSort(timestep)) return;

    if (m_prof) m_prof->push(m_exec_conf, "MPCD sort");

    // resize the sorted order vector to the current number of particles
    m_order.resize(m_mpcd_pdata->getN());
    m_rorder.resize(m_mpcd_pdata->getN());

    // generate and apply the sorted order
    computeOrder(timestep);
    applyOrder();

    // trigger the sort signal for ParticleData callbacks using the current sortings
    m_mpcd_pdata->notifySort(timestep, m_order, m_rorder);

    if (m_prof) m_prof->pop(m_exec_conf);
    }

/*!
 * \param timestep Current timestep
 *
 * Loop through the computed cell list and generate a compacted list of the order
 * particles appear. This will put the particles into a cell-list order, which
 * should be more friendly for other MPCD cell-based operations.
 */
void mpcd::Sorter::computeOrder(unsigned int timestep)
    {
    if (m_prof) m_prof->pop(m_exec_conf);
    // compute the cell list at current timestep, guarantees owned particles are on rank
    m_cl->compute(timestep);
    if (m_prof) m_prof->push(m_exec_conf,"MPCD sort");

    ArrayHandle<unsigned int> h_cell_list(m_cl->getCellList(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_cell_np(m_cl->getCellSizeArray(), access_location::host, access_mode::read);
    const Index2D& cli = m_cl->getCellListIndexer();

    // loop through the cell list to generate the sorting order for MPCD particles
    ArrayHandle<unsigned int> h_order(m_order, access_location::host, access_mode::overwrite);
    ArrayHandle<unsigned int> h_rorder(m_rorder, access_location::host, access_mode::overwrite);
    const unsigned int N_mpcd = m_mpcd_pdata->getN();
    unsigned int cur_p = 0;
    for (unsigned int idx=0; idx < m_cl->getNCells(); ++idx)
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

        ArrayHandle<Scalar4> h_pos(m_mpcd_pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_vel(m_mpcd_pdata->getVelocities(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_tag(m_mpcd_pdata->getTags(), access_location::host, access_mode::read);

        ArrayHandle<Scalar4> h_pos_alt(m_mpcd_pdata->getAltPositions(), access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar4> h_vel_alt(m_mpcd_pdata->getAltVelocities(), access_location::host, access_mode::overwrite);
        ArrayHandle<unsigned int> h_tag_alt(m_mpcd_pdata->getAltTags(), access_location::host, access_mode::overwrite);

        for (unsigned int idx=0; idx < m_mpcd_pdata->getN(); ++idx)
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

bool mpcd::Sorter::peekSort(unsigned int timestep) const
    {
    if (timestep < m_next_timestep)
        return false;
    else
        return ((timestep - m_next_timestep) % m_period == 0);
    }

bool mpcd::Sorter::shouldSort(unsigned int timestep)
    {
    if (peekSort(timestep))
        {
        m_next_timestep = timestep + m_period;
        return true;
        }
    else
        return false;
    }

/*!
 * \param m Python module to export to
 */
void mpcd::detail::export_Sorter(pybind11::module& m)
    {
    namespace py = pybind11;
    py::class_<mpcd::Sorter, std::shared_ptr<mpcd::Sorter> >(m, "Sorter")
        .def(py::init<std::shared_ptr<mpcd::SystemData>, unsigned int, unsigned int>())
        .def("setPeriod", &mpcd::Sorter::setPeriod)
        ;
    }
