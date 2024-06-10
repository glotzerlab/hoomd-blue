// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/CellListGPU.cc
 * \brief Definition of mpcd::CellListGPU
 */

#include "CellListGPU.h"
#include "CellListGPU.cuh"

namespace hoomd
    {
mpcd::CellListGPU::CellListGPU(std::shared_ptr<SystemDefinition> sysdef,
                               Scalar cell_size,
                               bool shift)
    : mpcd::CellList(sysdef, cell_size, shift)
    {
    m_tuner_cell.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                        m_exec_conf,
                                        "mpcd_cell"));
    m_tuner_sort.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                        m_exec_conf,
                                        "mpcd_cell_sort"));
    m_autotuners.insert(m_autotuners.end(), {m_tuner_cell, m_tuner_sort});

#ifdef ENABLE_MPI
    m_tuner_embed_migrate.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                                 m_exec_conf,
                                                 "mpcd_cell_embed_migrate"));
    m_autotuners.push_back(m_tuner_embed_migrate);

    GPUFlags<unsigned int> migrate_flag(m_exec_conf);
    m_migrate_flag.swap(migrate_flag);
#endif // ENABLE_MPI
    }

mpcd::CellListGPU::~CellListGPU() { }

void mpcd::CellListGPU::buildCellList()
    {
    ArrayHandle<unsigned int> d_cell_list(m_cell_list,
                                          access_location::device,
                                          access_mode::overwrite);
    ArrayHandle<unsigned int> d_cell_np(m_cell_np, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_pos(m_mpcd_pdata->getPositions(),
                               access_location::device,
                               access_mode::read);
    ArrayHandle<Scalar4> d_vel(m_mpcd_pdata->getVelocities(),
                               access_location::device,
                               access_mode::readwrite);

    const unsigned int N_mpcd = m_mpcd_pdata->getN() + m_mpcd_pdata->getNVirtual();
    unsigned int N_tot = N_mpcd;

    // total effective number of cells in the global box, optionally padded by
    // extra cells in MPI simulations
    uint3 n_global_cells = m_global_cell_dim;
#ifdef ENABLE_MPI
    if (isCommunicating(mpcd::detail::face::east))
        n_global_cells.x += 2 * m_num_extra;
    if (isCommunicating(mpcd::detail::face::north))
        n_global_cells.y += 2 * m_num_extra;
    if (isCommunicating(mpcd::detail::face::up))
        n_global_cells.z += 2 * m_num_extra;
#endif // ENABLE_MPI

    if (m_embed_group)
        {
        ArrayHandle<unsigned int> d_embed_cell_ids(m_embed_cell_ids,
                                                   access_location::device,
                                                   access_mode::overwrite);
        ArrayHandle<Scalar4> d_pos_embed(m_pdata->getPositions(),
                                         access_location::device,
                                         access_mode::read);
        ArrayHandle<unsigned int> d_embed_member_idx(m_embed_group->getIndexArray(),
                                                     access_location::device,
                                                     access_mode::read);
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
                                     m_pdata->getGlobalBox(),
                                     n_global_cells,
                                     m_global_cell_dim,
                                     m_cell_np_max,
                                     m_cell_indexer,
                                     m_cell_list_indexer,
                                     N_mpcd,
                                     N_tot,
                                     m_tuner_cell->getParam()[0]);
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
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
                                     m_pdata->getGlobalBox(),
                                     n_global_cells,
                                     m_global_cell_dim,
                                     m_cell_np_max,
                                     m_cell_indexer,
                                     m_cell_list_indexer,
                                     N_mpcd,
                                     N_tot,
                                     m_tuner_cell->getParam()[0]);
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_tuner_cell->end();
        }
    }

/*!
 * \param timestep Timestep that the sorting occurred
 * \param order Mapping of sorted particle indexes onto old particle indexes
 * \param rorder Mapping of old particle indexes onto sorted particle indexes
 */
void mpcd::CellListGPU::sort(uint64_t timestep,
                             const GPUArray<unsigned int>& order,
                             const GPUArray<unsigned int>& rorder)
    {
    // no need to do any sorting if we can still be called at the current timestep
    if (peekCompute(timestep))
        return;

    // force a recompute if mapping is invalid
    if (rorder.isNull())
        {
        m_force_compute = true;
        return;
        }

    // iterate through particles in cell list, and update their indexes using reverse mapping
    ArrayHandle<unsigned int> d_cell_list(m_cell_list,
                                          access_location::device,
                                          access_mode::readwrite);
    ArrayHandle<unsigned int> d_rorder(rorder, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_cell_np(m_cell_np, access_location::device, access_mode::read);

    m_tuner_sort->begin();
    mpcd::gpu::cell_apply_sort(d_cell_list.data,
                               d_rorder.data,
                               d_cell_np.data,
                               m_cell_list_indexer,
                               m_mpcd_pdata->getN(),
                               m_tuner_sort->getParam()[0]);
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_sort->end();
    }

#ifdef ENABLE_MPI
bool mpcd::CellListGPU::needsEmbedMigrate(uint64_t timestep)
    {
    // no migrate needed if no embedded particles
    if (!m_embed_group)
        return false;

    // ensure that the cell list has been sized first
    computeDimensions();

    // particle data
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_group(m_embed_group->getIndexArray(),
                                      access_location::device,
                                      access_mode::read);

    // check if any particles have left this rank on the gpu
    m_tuner_embed_migrate->begin();
    mpcd::gpu::cell_check_migrate_embed(m_migrate_flag.getDeviceFlags(),
                                        d_pos.data,
                                        d_group.data,
                                        m_cover_box,
                                        m_sysdef->getNDimensions(),
                                        m_embed_group->getNumMembers(),
                                        m_tuner_embed_migrate->getParam()[0]);
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_embed_migrate->end();

    // read flags from the gpu, and reduce across all ranks
    char migrate = static_cast<char>(m_migrate_flag.readFlags());
    MPI_Allreduce(MPI_IN_PLACE, &migrate, 1, MPI_CHAR, MPI_MAX, m_exec_conf->getMPICommunicator());

    return static_cast<bool>(migrate);
    }
#endif // ENABLE_MPI

void mpcd::detail::export_CellListGPU(pybind11::module& m)
    {
    pybind11::class_<mpcd::CellListGPU, mpcd::CellList, std::shared_ptr<mpcd::CellListGPU>>(
        m,
        "CellListGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, Scalar, bool>());
    }

    } // end namespace hoomd
