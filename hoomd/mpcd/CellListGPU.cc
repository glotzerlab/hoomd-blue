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
    m_autotuners.push_back(m_tuner_embed_migrate);

    GPUFlags<unsigned int> migrate_flag(m_exec_conf);
    m_migrate_flag.swap(migrate_flag);
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
    m_autotuners.push_back(m_tuner_embed_migrate);

    GPUFlags<unsigned int> migrate_flag(m_exec_conf);
    m_migrate_flag.swap(migrate_flag);
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
                                     m_pdata->getBox().getPeriodic(),
                                     m_origin_idx,
                                     m_grid_shift,
                                     m_pdata->getGlobalBox(),
                                     n_global_cells,
                                     m_global_cell_dim,
                                     m_cell_indexer,
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
                                     m_pdata->getBox().getPeriodic(),
                                     m_origin_idx,
                                     m_grid_shift,
                                     m_pdata->getGlobalBox(),
                                     n_global_cells,
                                     m_global_cell_dim,
                                     m_cell_indexer,
                                     N_mpcd,
                                     N_tot,
                                     m_flags[mpcd::detail::thermo_options::energy],
                                     m_tuner_cell->getParam()[0]);
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_tuner_cell->end();
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
