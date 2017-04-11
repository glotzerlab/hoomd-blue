// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/CellListGPU.cc
 * \brief Definition of mpcd::CellListGPU
 */

#include "CellListGPU.h"
#include "CellListGPU.cuh"

mpcd::CellListGPU::CellListGPU(std::shared_ptr<SystemDefinition> sysdef,
                               std::shared_ptr<mpcd::ParticleData> mpcd_pdata)
        : mpcd::CellList(sysdef, mpcd_pdata)
    {
    m_tuner_cell.reset(new Autotuner(32, 1024, 32, 5, 100000, "mpcd_cell", m_exec_conf));
    }

mpcd::CellListGPU::~CellListGPU()
    {
    }

void mpcd::CellListGPU::buildCellList()
    {
    ArrayHandle<unsigned int> d_cell_list(m_cell_list, access_location::device, access_mode::overwrite);
    ArrayHandle<unsigned int> d_cell_np(m_cell_np, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_pos(m_mpcd_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_vel(m_mpcd_pdata->getVelocities(), access_location::device, access_mode::readwrite);

    const unsigned int N_mpcd = m_mpcd_pdata->getN();
    unsigned int N_tot = N_mpcd;

    // total effective number of cells in the global box, optionally padded by
    // extra cells in MPI simulations
    uint3 n_global_cells = m_global_cell_dim;
    #ifdef ENABLE_MPI
    if (isCommunicating(mpcd::detail::face::east)) n_global_cells.x += 2*m_num_extra;
    if (isCommunicating(mpcd::detail::face::north)) n_global_cells.y += 2*m_num_extra;
    if (isCommunicating(mpcd::detail::face::up)) n_global_cells.z += 2*m_num_extra;
    #endif // ENABLE_MPI

    if (m_embed_group)
        {
        ArrayHandle<unsigned int> d_embed_cell_ids(m_embed_cell_ids, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar4> d_pos_embed(m_pdata->getPositions(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_embed_member_idx(m_embed_group->getIndexArray(), access_location::device, access_mode::read);
        N_tot += m_embed_group->getNumMembers();

        m_tuner_cell->begin();
        mpcd::gpu::compute_cell_list(d_cell_np.data,
                                     d_cell_list.data,
                                     m_conditions.getDeviceFlags(),
                                     d_vel.data,
                                     d_embed_cell_ids.data,
                                     d_pos.data,
                                     d_pos_embed.data,
                                     d_embed_member_idx.data,
                                     m_pdata->getBox().getPeriodic(),
                                     m_origin_idx,
                                     m_grid_shift,
                                     m_pdata->getGlobalBox().getLo(),
                                     n_global_cells,
                                     m_cell_size,
                                     m_cell_np_max,
                                     m_cell_indexer,
                                     m_cell_list_indexer,
                                     N_mpcd,
                                     N_tot,
                                     m_tuner_cell->getParam());
        if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
        m_tuner_cell->end();
        }
    else
        {
        m_tuner_cell->begin();
        mpcd::gpu::compute_cell_list(d_cell_np.data,
                                     d_cell_list.data,
                                     m_conditions.getDeviceFlags(),
                                     d_vel.data,
                                     NULL,
                                     d_pos.data,
                                     NULL,
                                     NULL,
                                     m_pdata->getBox().getPeriodic(),
                                     m_origin_idx,
                                     m_grid_shift,
                                     m_pdata->getGlobalBox().getLo(),
                                     n_global_cells,
                                     m_cell_size,
                                     m_cell_np_max,
                                     m_cell_indexer,
                                     m_cell_list_indexer,
                                     N_mpcd,
                                     N_tot,
                                     m_tuner_cell->getParam());
        if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
        m_tuner_cell->end();
        }
    }

void mpcd::detail::export_CellListGPU(pybind11::module& m)
    {
    namespace py = pybind11;

    py::class_<mpcd::CellListGPU, std::shared_ptr<mpcd::CellListGPU> >(m, "CellListGPU", py::base<mpcd::CellList>())
        .def(py::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<mpcd::ParticleData> >())
    ;
    }
