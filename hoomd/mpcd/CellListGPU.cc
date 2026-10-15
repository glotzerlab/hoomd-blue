// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/CellListGPU.cc
 * \brief Definition of mpcd::CellListGPU
 */

#include "CellListGPU.h"

namespace hoomd
    {
mpcd::CellListGPU::CellListGPU(std::shared_ptr<SystemDefinition> sysdef,
                               Scalar cell_size,
                               bool shift)
    : mpcd::CellList(sysdef, cell_size, shift), m_tmp_thermo(m_exec_conf), m_reduced(m_exec_conf)
    {
    m_tuner_cell.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                        m_exec_conf,
                                        "mpcd_cell"));
    m_tuner_property.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                            m_exec_conf,
                                            "mpcd_cell_property"));
    m_tuner_net.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                       m_exec_conf,
                                       "mpcd_cell_net_property"));
    m_autotuners.insert(m_autotuners.end(), {m_tuner_cell, m_tuner_property, m_tuner_net});

#ifdef ENABLE_MPI
    m_tuner_embed_migrate.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                                 m_exec_conf,
                                                 "mpcd_cell_embed_migrate"));
    m_tuner_filter_ghosts.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                                 m_exec_conf,
                                                 "mpcd_filter_ghosts"));
    m_tuner_send_num.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                            m_exec_conf,
                                            "mpcd_cell_send_num"));
    m_tuner_buffer.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                          m_exec_conf,
                                          "mpcd_cell_buffer"));
    m_tuner_ghost_cell.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                              m_exec_conf,
                                              "mpcd_cell_ghost_property"));
    m_tuner_ghost_update.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                                m_exec_conf,
                                                "mpcd_cell_ghost_update"));
    m_autotuners.insert(m_autotuners.end(),
                        {m_tuner_embed_migrate,
                         m_tuner_filter_ghosts,
                         m_tuner_send_num,
                         m_tuner_buffer,
                         m_tuner_ghost_cell,
                         m_tuner_ghost_update});

    GPUFlags<unsigned int> migrate_flag(m_exec_conf);
    m_migrate_flag.swap(migrate_flag);

    GPUFlags<unsigned int> num_ghost_scan(m_exec_conf);
    m_num_ghost_scan.swap(num_ghost_scan);
#endif // ENABLE_MPI
    }

mpcd::CellListGPU::CellListGPU(std::shared_ptr<SystemDefinition> sysdef,
                               const uint3& global_cell_dim,
                               bool shift)
    : mpcd::CellList(sysdef, global_cell_dim, shift), m_tmp_thermo(m_exec_conf),
      m_reduced(m_exec_conf)
    {
    m_tuner_cell.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                        m_exec_conf,
                                        "mpcd_cell"));
    m_tuner_property.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                            m_exec_conf,
                                            "mpcd_cell_property"));
    m_tuner_net.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                       m_exec_conf,
                                       "mpcd_cell_net_property"));
    m_autotuners.insert(m_autotuners.end(), {m_tuner_cell, m_tuner_property, m_tuner_net});

#ifdef ENABLE_MPI
    m_tuner_embed_migrate.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                                 m_exec_conf,
                                                 "mpcd_cell_embed_migrate"));
    m_tuner_filter_ghosts.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                                 m_exec_conf,
                                                 "mpcd_filter_ghosts"));
    m_tuner_send_num.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                            m_exec_conf,
                                            "mpcd_cell_send_num"));
    m_tuner_buffer.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                          m_exec_conf,
                                          "mpcd_cell_buffer"));
    m_tuner_ghost_cell.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                              m_exec_conf,
                                              "mpcd_cell_ghost_property"));
    m_tuner_ghost_update.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                                m_exec_conf,
                                                "mpcd_cell_ghost_update"));
    m_autotuners.insert(m_autotuners.end(),
                        {m_tuner_embed_migrate,
                         m_tuner_filter_ghosts,
                         m_tuner_send_num,
                         m_tuner_buffer,
                         m_tuner_ghost_cell,
                         m_tuner_ghost_update});

    GPUFlags<unsigned int> migrate_flag(m_exec_conf);
    m_migrate_flag.swap(migrate_flag);

    GPUFlags<unsigned int> num_ghost_scan(m_exec_conf);
    m_num_ghost_scan.swap(num_ghost_scan);
#endif // ENABLE_MPI
    }

mpcd::CellListGPU::~CellListGPU() { }

void mpcd::CellListGPU::buildCellList()
    {
    ArrayHandle<unsigned int> d_cell_np(m_cell_np, access_location::device, access_mode::overwrite);
    ArrayHandle<double4> d_cell_vel(m_cell_vel, access_location::device, access_mode::overwrite);
    ArrayHandle<double> d_cell_energy(m_cell_energy,
                                      access_location::device,
                                      access_mode::overwrite);
    ArrayHandle<Scalar4> d_pos(m_mpcd_pdata->getPositions(),
                               access_location::device,
                               access_mode::read);
    ArrayHandle<Scalar4> d_vel(m_mpcd_pdata->getVelocities(),
                               access_location::device,
                               access_mode::readwrite);

    const unsigned int N_mpcd = m_mpcd_pdata->getN() + m_mpcd_pdata->getNVirtual();
    unsigned int N_tot = N_mpcd;

    // get communcation variables
    uint3 rank_size = make_uint3(0, 0, 0);
#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        if (N_mpcd > m_ghost_idx.getNumElements())
            {
            GPUArray<unsigned int> m_ghost_dir_tmp(N_mpcd, m_exec_conf);
            m_ghost_dir.swap(m_ghost_dir_tmp);
            }
        ArrayHandle<unsigned int> d_ghost_dir(m_ghost_dir,
                                              access_location::device,
                                              access_mode::overwrite);

        // allocate the the number of ranks in each dimension
        Index3D di = m_decomposition->getDomainIndexer();
        rank_size = make_uint3(di.getW(), di.getH(), di.getD());

        if (m_embed_group)
            {
            ArrayHandle<unsigned int> d_embed_cell_ids(m_embed_cell_ids,
                                                       access_location::device,
                                                       access_mode::overwrite);
            ArrayHandle<Scalar4> d_pos_embed(m_pdata->getPositions(),
                                             access_location::device,
                                             access_mode::read);
            ArrayHandle<Scalar4> d_vel_embed(m_pdata->getVelocities(),
                                             access_location::device,
                                             access_mode::read);
            ArrayHandle<unsigned int> d_embed_member_idx(m_embed_group->getIndexArray(),
                                                         access_location::device,
                                                         access_mode::read);
            N_tot += m_embed_group->getNumMembers();

            m_tuner_cell->begin();
            mpcd::gpu::compute_cell_list(d_cell_np.data,
                                         d_cell_vel.data,
                                         d_cell_energy.data,
                                         m_conditions.getDeviceFlags(),
                                         d_vel.data,
                                         m_mpcd_pdata->getMass(),
                                         d_embed_cell_ids.data,
                                         d_pos.data,
                                         d_pos_embed.data,
                                         d_vel_embed.data,
                                         d_embed_member_idx.data,
                                         m_pdata->getGlobalBox().getPeriodic(),
                                         m_origin_idx,
                                         m_grid_shift,
                                         m_pdata->getGlobalBox(),
                                         m_global_cell_dim,
                                         m_cell_indexer,
                                         m_global_cell_indexer,
                                         d_ghost_dir.data,
                                         rank_size,
                                         true,
                                         N_mpcd,
                                         N_tot,
                                         m_flags[mpcd::detail::thermo_options::energy],
                                         m_tuner_cell->getParam()[0]);
            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            m_tuner_cell->end();
            }
        else
            {
            m_tuner_cell->begin();
            mpcd::gpu::compute_cell_list(d_cell_np.data,
                                         d_cell_vel.data,
                                         d_cell_energy.data,
                                         m_conditions.getDeviceFlags(),
                                         d_vel.data,
                                         m_mpcd_pdata->getMass(),
                                         NULL,
                                         d_pos.data,
                                         NULL,
                                         NULL,
                                         NULL,
                                         m_pdata->getGlobalBox().getPeriodic(),
                                         m_origin_idx,
                                         m_grid_shift,
                                         m_pdata->getGlobalBox(),
                                         m_global_cell_dim,
                                         m_cell_indexer,
                                         m_global_cell_indexer,
                                         d_ghost_dir.data,
                                         rank_size,
                                         true,
                                         N_mpcd,
                                         N_tot,
                                         m_flags[mpcd::detail::thermo_options::energy],
                                         m_tuner_cell->getParam()[0]);
            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            m_tuner_cell->end();
            }
        }
    else
#endif // ENABLE_MPI
        {
        if (m_embed_group)
            {
            ArrayHandle<unsigned int> d_embed_cell_ids(m_embed_cell_ids,
                                                       access_location::device,
                                                       access_mode::overwrite);
            ArrayHandle<Scalar4> d_pos_embed(m_pdata->getPositions(),
                                             access_location::device,
                                             access_mode::read);
            ArrayHandle<Scalar4> d_vel_embed(m_pdata->getVelocities(),
                                             access_location::device,
                                             access_mode::read);
            ArrayHandle<unsigned int> d_embed_member_idx(m_embed_group->getIndexArray(),
                                                         access_location::device,
                                                         access_mode::read);
            N_tot += m_embed_group->getNumMembers();

            m_tuner_cell->begin();
            mpcd::gpu::compute_cell_list(d_cell_np.data,
                                         d_cell_vel.data,
                                         d_cell_energy.data,
                                         m_conditions.getDeviceFlags(),
                                         d_vel.data,
                                         m_mpcd_pdata->getMass(),
                                         d_embed_cell_ids.data,
                                         d_pos.data,
                                         d_pos_embed.data,
                                         d_vel_embed.data,
                                         d_embed_member_idx.data,
                                         m_pdata->getGlobalBox().getPeriodic(),
                                         m_origin_idx,
                                         m_grid_shift,
                                         m_pdata->getGlobalBox(),
                                         m_global_cell_dim,
                                         m_cell_indexer,
                                         m_global_cell_indexer,
                                         NULL,
                                         rank_size,
                                         false,
                                         N_mpcd,
                                         N_tot,
                                         m_flags[mpcd::detail::thermo_options::energy],
                                         m_tuner_cell->getParam()[0]);
            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            m_tuner_cell->end();
            }
        else
            {
            m_tuner_cell->begin();
            mpcd::gpu::compute_cell_list(d_cell_np.data,
                                         d_cell_vel.data,
                                         d_cell_energy.data,
                                         m_conditions.getDeviceFlags(),
                                         d_vel.data,
                                         m_mpcd_pdata->getMass(),
                                         NULL,
                                         d_pos.data,
                                         NULL,
                                         NULL,
                                         NULL,
                                         m_pdata->getGlobalBox().getPeriodic(),
                                         m_origin_idx,
                                         m_grid_shift,
                                         m_pdata->getGlobalBox(),
                                         m_global_cell_dim,
                                         m_cell_indexer,
                                         m_global_cell_indexer,
                                         NULL,
                                         rank_size,
                                         false,
                                         N_mpcd,
                                         N_tot,
                                         m_flags[mpcd::detail::thermo_options::energy],
                                         m_tuner_cell->getParam()[0]);
            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            m_tuner_cell->end();
            }
        }
    }

void mpcd::CellListGPU::finishComputeProperties()
    {
    ArrayHandle<unsigned int> d_cell_np(m_cell_np, access_location::device, access_mode::read);
    ArrayHandle<double4> d_cell_vel(m_cell_vel, access_location::device, access_mode::readwrite);
    ArrayHandle<double> d_cell_energy(m_cell_energy, access_location::device, access_mode::read);
    ArrayHandle<double> d_cell_temp(m_cell_temp, access_location::device, access_mode::overwrite);

    m_tuner_property->begin();
    mpcd::gpu::finish_cell_properties(d_cell_np.data,
                                      d_cell_vel.data,
                                      d_cell_energy.data,
                                      d_cell_temp.data,
                                      getNCells(),
                                      m_sysdef->getNDimensions(),
                                      m_flags[mpcd::detail::thermo_options::energy],
                                      m_tuner_property->getParam()[0]);
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_property->end();
    }

void mpcd::CellListGPU::computeNetProperties()
    {
        {
        ArrayHandle<unsigned int> d_cell_np(m_cell_np, access_location::device, access_mode::read);
        ArrayHandle<double4> d_cell_vel(m_cell_vel, access_location::device, access_mode::read);
        ArrayHandle<double> d_cell_energy(m_cell_energy,
                                          access_location::device,
                                          access_mode::read);
        ArrayHandle<double> d_cell_temp(m_cell_temp, access_location::device, access_mode::read);

        m_tmp_thermo.resize(getNCells());

        ArrayHandle<mpcd::detail::cell_thermo_element> d_tmp_thermo(m_tmp_thermo,
                                                                    access_location::device,
                                                                    access_mode::overwrite);

        m_tuner_net->begin();
        mpcd::gpu::stage_net_cell_thermo(d_tmp_thermo.data,
                                         d_cell_np.data,
                                         d_cell_vel.data,
                                         d_cell_energy.data,
                                         d_cell_temp.data,
                                         getNCells(),
                                         m_flags[mpcd::detail::thermo_options::energy],
                                         m_tuner_net->getParam()[0]);
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_tuner_net->end();

        // use cub to reduce the properties on the gpu
        void* d_tmp = NULL;
        size_t tmp_bytes = 0;
        mpcd::gpu::reduce_net_cell_thermo(m_reduced.getDeviceFlags(),
                                          d_tmp,
                                          tmp_bytes,
                                          d_tmp_thermo.data,
                                          m_tmp_thermo.size());
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        ScopedAllocation<unsigned char> d_tmp_alloc(m_exec_conf->getCachedAllocator(),
                                                    (tmp_bytes > 0) ? tmp_bytes : 1);
        d_tmp = (void*)d_tmp_alloc();

        mpcd::gpu::reduce_net_cell_thermo(m_reduced.getDeviceFlags(),
                                          d_tmp,
                                          tmp_bytes,
                                          d_tmp_thermo.data,
                                          m_tmp_thermo.size());
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    // now copy the net properties back to host from the flags
    unsigned int n_temp_cells = 0;
        {
        const mpcd::detail::cell_thermo_element reduced = m_reduced.readFlags();

        ArrayHandle<double> h_net_properties(m_net_properties,
                                             access_location::host,
                                             access_mode::overwrite);
        h_net_properties.data[mpcd::detail::thermo_index::momentum_x] = reduced.momentum.x;
        h_net_properties.data[mpcd::detail::thermo_index::momentum_y] = reduced.momentum.y;
        h_net_properties.data[mpcd::detail::thermo_index::momentum_z] = reduced.momentum.z;

        h_net_properties.data[mpcd::detail::thermo_index::energy] = reduced.energy;
        h_net_properties.data[mpcd::detail::thermo_index::temperature] = reduced.temperature;

        n_temp_cells = reduced.flag;
        }

#ifdef ENABLE_MPI
        {
        ArrayHandle<double> h_net_properties(m_net_properties,
                                             access_location::host,
                                             access_mode::readwrite);
        MPI_Allreduce(MPI_IN_PLACE,
                      h_net_properties.data,
                      mpcd::detail::thermo_index::num_quantities,
                      MPI_DOUBLE,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());

        MPI_Allreduce(MPI_IN_PLACE,
                      &n_temp_cells,
                      1,
                      MPI_UNSIGNED,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());
        }
#endif // ENABLE_MPI

    if (n_temp_cells > 0)
        {
        ArrayHandle<double> h_net_properties(m_net_properties,
                                             access_location::host,
                                             access_mode::readwrite);
        h_net_properties.data[mpcd::detail::thermo_index::temperature] /= (double)n_temp_cells;
        }
    m_needs_net_reduce = false;
    }

#ifdef ENABLE_MPI
void mpcd::CellListGPU::fillGhostBuffers()
    {
    if (!m_decomposition)
        {
        return;
        }

        // sort the communication keys
        {
        const unsigned int N_mpcd = m_mpcd_pdata->getN() + m_mpcd_pdata->getNVirtual();
        if (N_mpcd > m_ghost_dir_sorted.getNumElements())
            {
            GPUArray<unsigned int> m_ghost_dir_sorted_tmp(N_mpcd, m_exec_conf);
            m_ghost_dir_sorted.swap(m_ghost_dir_sorted_tmp);

            GPUArray<unsigned int> m_ghost_idx_tmp(N_mpcd, m_exec_conf);
            m_ghost_idx.swap(m_ghost_idx_tmp);

            GPUArray<unsigned int> m_ghost_idx_sorted_tmp(N_mpcd, m_exec_conf);
            m_ghost_idx_sorted.swap(m_ghost_idx_sorted_tmp);
            }

            // filter the particles to sort down to just the ones we want to send
            // we will abuse the various ghost memory to not have to allocate extra
            {
            ArrayHandle<unsigned int> d_ghost_dir(m_ghost_dir,
                                                  access_location::device,
                                                  access_mode::read);
            ArrayHandle<unsigned int> d_ghost_dir_scan(m_ghost_dir_sorted,
                                                       access_location::device,
                                                       access_mode::overwrite);
            ArrayHandle<unsigned int> d_ghost_dir_filter(m_ghost_idx_sorted,
                                                         access_location::device,
                                                         access_mode::overwrite);
            ArrayHandle<unsigned int> d_ghost_idx_filter(m_ghost_idx,
                                                         access_location::device,
                                                         access_mode::overwrite);

            /* scan the directions to identify ghosts with prefix sum */
            void* d_tmp = NULL;
            size_t tmp_bytes = 0;
            mpcd::gpu::scan_for_ghosts(d_tmp,
                                       tmp_bytes,
                                       d_ghost_dir.data,
                                       d_ghost_dir_scan.data,
                                       N_mpcd);

            // make requested temporary allocation (1 char = 1B)
            size_t alloc_size = (tmp_bytes > 0) ? tmp_bytes : 4;
            ScopedAllocation<unsigned char> d_alloc(m_exec_conf->getCachedAllocator(), alloc_size);
            d_tmp = (void*)d_alloc();

            mpcd::gpu::scan_for_ghosts(d_tmp,
                                       tmp_bytes,
                                       d_ghost_dir.data,
                                       d_ghost_dir_scan.data,
                                       N_mpcd);

            /* filter down to just the ghosts that need sending */
            m_tuner_filter_ghosts->begin();
            mpcd::gpu::filter_ghosts(d_ghost_dir_filter.data,
                                     d_ghost_idx_filter.data,
                                     m_num_ghost_scan.getDeviceFlags(),
                                     d_ghost_dir.data,
                                     d_ghost_dir_scan.data,
                                     N_mpcd,
                                     m_tuner_filter_ghosts->getParam()[0]);
            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            m_tuner_filter_ghosts->end();

            // copy number of ghosts to send from device memory
            m_num_mpcd_ghosts_send = m_num_ghost_scan.readFlags();
            }

        // swap abused memory: m_ghost_idx_sorted has the filtered directions
        // m_ghost_idx has the filtered indexes already.
        // the contents of the *_sorted arrays don't matter because they will
        // get overwritten next
        m_ghost_dir.swap(m_ghost_idx_sorted);

        uchar2 swap;
            {
            ArrayHandle<unsigned int> d_ghost_dir(m_ghost_dir,
                                                  access_location::device,
                                                  access_mode::readwrite);
            ArrayHandle<unsigned int> d_ghost_dir_sorted(m_ghost_dir_sorted,
                                                         access_location::device,
                                                         access_mode::overwrite);

            ArrayHandle<unsigned int> d_ghost_idx(m_ghost_idx,
                                                  access_location::device,
                                                  access_mode::readwrite);
            ArrayHandle<unsigned int> d_ghost_idx_sorted(m_ghost_idx_sorted,
                                                         access_location::device,
                                                         access_mode::overwrite);

            void* d_tmp = NULL;
            size_t tmp_bytes = 0;
            mpcd::gpu::sort_ghosts_by_dir(d_tmp,
                                          tmp_bytes,
                                          d_ghost_dir.data,
                                          d_ghost_dir_sorted.data,
                                          d_ghost_idx.data,
                                          d_ghost_idx_sorted.data,
                                          m_num_mpcd_ghosts_send);

            // make requested temporary allocation (1 char = 1B)
            size_t alloc_size = (tmp_bytes > 0) ? tmp_bytes : 4;
            ScopedAllocation<unsigned char> d_alloc(m_exec_conf->getCachedAllocator(), alloc_size);
            d_tmp = (void*)d_alloc();

            // perform the sort
            swap = mpcd::gpu::sort_ghosts_by_dir(d_tmp,
                                                 tmp_bytes,
                                                 d_ghost_dir.data,
                                                 d_ghost_dir_sorted.data,
                                                 d_ghost_idx.data,
                                                 d_ghost_idx_sorted.data,
                                                 m_num_mpcd_ghosts_send);
            }
        if (swap.x)
            m_ghost_dir_sorted.swap(m_ghost_dir);
        if (swap.y)
            m_ghost_idx_sorted.swap(m_ghost_idx);
        }

        // determine the starting indexes and total number of ghost particles
        {
        ArrayHandle<unsigned int> d_mpcd_send_offsets(m_mpcd_send_offsets,
                                                      access_location::device,
                                                      access_mode::readwrite);
        ArrayHandle<unsigned int> d_ghost_dir_sorted(m_ghost_dir_sorted,
                                                     access_location::device,
                                                     access_mode::read);
        m_tuner_send_num->begin();
        mpcd::gpu::find_num_ghost_send(d_mpcd_send_offsets.data,
                                       d_ghost_dir_sorted.data,
                                       m_num_mpcd_ghosts_send,
                                       m_tuner_send_num->getParam()[0]);
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_tuner_send_num->end();
        }

        // fill send buffer
        {
        m_mpcd_vel_sendbuf.resize(m_num_mpcd_ghosts_send);

        ArrayHandle<Scalar4> d_mpcd_vel_sendbuf(m_mpcd_vel_sendbuf,
                                                access_location::device,
                                                access_mode::overwrite);
        ArrayHandle<Scalar4> d_vel(m_mpcd_pdata->getVelocities(),
                                   access_location::device,
                                   access_mode::read);
        ArrayHandle<unsigned int> d_ghost_idx_sorted(m_ghost_idx_sorted,
                                                     access_location::device,
                                                     access_mode::read);

        m_tuner_buffer->begin();
        mpcd::gpu::fill_buffer(d_mpcd_vel_sendbuf.data,
                               d_vel.data,
                               d_ghost_idx_sorted.data,
                               m_num_mpcd_ghosts_send,
                               m_tuner_buffer->getParam()[0]);
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_tuner_buffer->end();
        }
    }

void mpcd::CellListGPU::addGhostsToCells()
    {
    if (!m_decomposition)
        {
        return;
        }

    if (!m_num_mpcd_ghosts_recv)
        {
        return;
        }

    ArrayHandle<unsigned int> d_cell_np(m_cell_np, access_location::device, access_mode::readwrite);
    ArrayHandle<double4> d_cell_vel(m_cell_vel, access_location::device, access_mode::readwrite);
    ArrayHandle<double> d_cell_energy(m_cell_energy,
                                      access_location::device,
                                      access_mode::readwrite);
    ArrayHandle<Scalar4> d_mpcd_ghost_vel(m_mpcd_ghost_vel,
                                          access_location::device,
                                          access_mode::readwrite);

    m_tuner_ghost_cell->begin();
    mpcd::gpu::add_ghost_cell_properties(d_cell_np.data,
                                         d_cell_vel.data,
                                         d_cell_energy.data,
                                         m_conditions.getDeviceFlags(),
                                         d_mpcd_ghost_vel.data,
                                         m_mpcd_pdata->getMass(),
                                         m_origin_idx,
                                         m_global_cell_dim,
                                         m_cell_indexer,
                                         m_num_mpcd_ghosts_recv,
                                         m_flags[mpcd::detail::thermo_options::energy],
                                         m_tuner_ghost_cell->getParam()[0]);
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_ghost_cell->end();
    }

void mpcd::CellListGPU::updateLocalFromGhosts()
    {
    if (!m_decomposition)
        {
        return;
        }

    // fill update the local particles
    ArrayHandle<unsigned int> d_ghost_idx_sorted(m_ghost_idx_sorted,
                                                 access_location::device,
                                                 access_mode::read);
    ArrayHandle<Scalar4> d_mpcd_vel_sendbuf(m_mpcd_vel_sendbuf,
                                            access_location::device,
                                            access_mode::read);
    ArrayHandle<Scalar4> d_vel(m_mpcd_pdata->getVelocities(),
                               access_location::device,
                               access_mode::readwrite);

    m_tuner_ghost_update->begin();
    mpcd::gpu::update_local_from_ghosts(d_vel.data,
                                        d_mpcd_vel_sendbuf.data,
                                        d_ghost_idx_sorted.data,
                                        m_num_mpcd_ghosts_send,
                                        m_tuner_ghost_update->getParam()[0]);
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_ghost_update->end();
    }

#endif // ENABLE_MPI

namespace mpcd
    {
namespace detail
    {
void export_CellListGPU(pybind11::module& m)
    {
    pybind11::class_<mpcd::CellListGPU, mpcd::CellList, std::shared_ptr<mpcd::CellListGPU>>(
        m,
        "CellListGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, Scalar, bool>());
    }
    } // namespace detail
    } // namespace mpcd
    } // end namespace hoomd
