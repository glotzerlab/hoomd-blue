// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/CellThermoComputeGPU.cc
 * \brief Definition of mpcd::CellThermoComputeGPU
 */

#include "CellThermoComputeGPU.h"
#include "ReductionOperators.h"

/*!
 * \param sysdef System definition
 * \param cl MPCD cell list
 */
mpcd::CellThermoComputeGPU::CellThermoComputeGPU(std::shared_ptr<mpcd::SystemData> sysdata,
                                                 const std::string& suffix)
    : mpcd::CellThermoCompute(sysdata, suffix), m_tmp_thermo(m_exec_conf), m_reduced(m_exec_conf)
    {
    // construct a range of valid tuner parameters using the block size and number of threads per particle
    std::vector<unsigned int> valid_params;
    for (unsigned int block_size = 32; block_size <= 1024; block_size += 32)
        {
        for (auto s : Autotuner::getTppListPow2(this->m_exec_conf->dev_prop.warpSize))
            {
            valid_params.push_back(block_size * 10000 + s);
            }
        }

    m_begin_tuner.reset(new Autotuner(valid_params, 5, 100000, "mpcd_cell_thermo_begin", m_exec_conf));
    m_end_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "mpcd_cell_thermo_end", m_exec_conf));
    m_inner_tuner.reset(new Autotuner(valid_params, 5, 100000, "mpcd_cell_thermo_inner", m_exec_conf));
    m_stage_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "mpcd_cell_thermo_stage", m_exec_conf));
    }

mpcd::CellThermoComputeGPU::~CellThermoComputeGPU()
    {
    }

#ifdef ENABLE_MPI
void mpcd::CellThermoComputeGPU::beginOuterCellProperties()
    {
    ArrayHandle<double4> d_cell_vel(m_cell_vel, access_location::device, access_mode::overwrite);
    ArrayHandle<double3> d_cell_energy(m_cell_energy, access_location::device, access_mode::overwrite);

    ArrayHandle<unsigned int> d_cells(m_vel_comm->getCells(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_cell_np(m_cl->getCellSizeArray(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_cell_list(m_cl->getCellList(), access_location::device, access_mode::read);

    ArrayHandle<Scalar4> d_vel(m_mpcd_pdata->getVelocities(), access_location::device, access_mode::read);

    if (m_cl->getEmbeddedGroup())
        {
        // Embedded particle data
        ArrayHandle<Scalar4> d_embed_vel(m_pdata->getVelocities(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_embed_cell(m_cl->getEmbeddedGroup()->getIndexArray(), access_location::device, access_mode::read);

        mpcd::detail::thermo_args_t args(d_cell_vel.data,
                                         d_cell_energy.data,
                                         d_cell_np.data,
                                         d_cell_list.data,
                                         m_cl->getCellListIndexer(),
                                         d_vel.data,
                                         m_mpcd_pdata->getN() + m_mpcd_pdata->getNVirtual(),
                                         m_mpcd_pdata->getMass(),
                                         d_embed_vel.data,
                                         d_embed_cell.data,
                                         m_flags[mpcd::detail::thermo_options::energy]);

        m_begin_tuner->begin();
        const unsigned int param = m_begin_tuner->getParam();
        const unsigned int block_size = param / 10000;
        const unsigned int tpp = param % 10000;
        gpu::begin_cell_thermo(args,
                               d_cells.data,
                               m_vel_comm->getNCells(),
                               block_size,
                               tpp);
        if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
        m_begin_tuner->end();
        }
    else
        {
        mpcd::detail::thermo_args_t args(d_cell_vel.data,
                                         d_cell_energy.data,
                                         d_cell_np.data,
                                         d_cell_list.data,
                                         m_cl->getCellListIndexer(),
                                         d_vel.data,
                                         m_mpcd_pdata->getN() + m_mpcd_pdata->getNVirtual(),
                                         m_mpcd_pdata->getMass(),
                                         NULL,
                                         NULL,
                                         m_flags[mpcd::detail::thermo_options::energy]);

        m_begin_tuner->begin();
        const unsigned int param = m_begin_tuner->getParam();
        const unsigned int block_size = param / 10000;
        const unsigned int tpp = param % 10000;
        gpu::begin_cell_thermo(args,
                               d_cells.data,
                               m_vel_comm->getNCells(),
                               block_size,
                               tpp);
        if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
        m_begin_tuner->end();
        }
    }

void mpcd::CellThermoComputeGPU::finishOuterCellProperties()
    {
    ArrayHandle<double4> d_cell_vel(m_cell_vel, access_location::device, access_mode::readwrite);
    ArrayHandle<double3> d_cell_energy(m_cell_energy, access_location::device, access_mode::readwrite);
    ArrayHandle<unsigned int> d_cells(m_vel_comm->getCells(), access_location::device, access_mode::read);
    m_end_tuner->begin();
    gpu::end_cell_thermo(d_cell_vel.data,
                         d_cell_energy.data,
                         d_cells.data,
                         m_vel_comm->getNCells(),
                         m_sysdef->getNDimensions(),
                         m_flags[mpcd::detail::thermo_options::energy],
                         m_end_tuner->getParam());
    if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
    m_end_tuner->end();
    }
#endif // ENABLE_MPI

void mpcd::CellThermoComputeGPU::calcInnerCellProperties()
    {
    ArrayHandle<double4> d_cell_vel(m_cell_vel, access_location::device, access_mode::overwrite);
    ArrayHandle<double3> d_cell_energy(m_cell_energy, access_location::device, access_mode::overwrite);

    ArrayHandle<unsigned int> d_cell_np(m_cl->getCellSizeArray(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_cell_list(m_cl->getCellList(), access_location::device, access_mode::read);

    ArrayHandle<Scalar4> d_vel(m_mpcd_pdata->getVelocities(), access_location::device, access_mode::read);

    /*
     * Determine the inner cell indexer and offset. The inner indexer is the cube containing
     * all non-communicating cells, and its offset is the number of communicating cells on the
     * lo side of the box.
     */
    uint3 lo, hi;
    const Index3D& ci = m_cl->getCellIndexer();
    #ifdef ENABLE_MPI
    if (m_use_mpi)
        {
        auto num_comm_cells = m_cl->getNComm();
        lo = make_uint3(num_comm_cells[static_cast<unsigned int>(mpcd::detail::face::west)],
                        num_comm_cells[static_cast<unsigned int>(mpcd::detail::face::south)],
                        num_comm_cells[static_cast<unsigned int>(mpcd::detail::face::down)]);
        hi = make_uint3(ci.getW() - num_comm_cells[static_cast<unsigned int>(mpcd::detail::face::east)],
                        ci.getH() - num_comm_cells[static_cast<unsigned int>(mpcd::detail::face::north)],
                        ci.getD() - num_comm_cells[static_cast<unsigned int>(mpcd::detail::face::up)]);
        }
    else
    #endif // ENABLE_MPI
        {
        lo = make_uint3(0,0,0);
        hi = m_cl->getDim();
        }
    Index3D inner_ci(hi.x-lo.x,hi.y-lo.y,hi.z-lo.z);

    if (m_cl->getEmbeddedGroup())
        {
        // Embedded particle data
        ArrayHandle<Scalar4> d_embed_vel(m_pdata->getVelocities(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_embed_cell(m_cl->getEmbeddedGroup()->getIndexArray(), access_location::device, access_mode::read);

        mpcd::detail::thermo_args_t args(d_cell_vel.data,
                                         d_cell_energy.data,
                                         d_cell_np.data,
                                         d_cell_list.data,
                                         m_cl->getCellListIndexer(),
                                         d_vel.data,
                                         m_mpcd_pdata->getN() + m_mpcd_pdata->getNVirtual(),
                                         m_mpcd_pdata->getMass(),
                                         d_embed_vel.data,
                                         d_embed_cell.data,
                                         m_flags[mpcd::detail::thermo_options::energy]);

        m_inner_tuner->begin();
        const unsigned int param = m_inner_tuner->getParam();
        const unsigned int block_size = param / 10000;
        const unsigned int tpp = param % 10000;
        gpu::inner_cell_thermo(args,
                               ci,
                               inner_ci,
                               lo,
                               m_sysdef->getNDimensions(),
                               block_size,
                               tpp);
        if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
        m_inner_tuner->end();
        }
    else
        {
        mpcd::detail::thermo_args_t args(d_cell_vel.data,
                                         d_cell_energy.data,
                                         d_cell_np.data,
                                         d_cell_list.data,
                                         m_cl->getCellListIndexer(),
                                         d_vel.data,
                                         m_mpcd_pdata->getN() + m_mpcd_pdata->getNVirtual(),
                                         m_mpcd_pdata->getMass(),
                                         NULL,
                                         NULL,
                                         m_flags[mpcd::detail::thermo_options::energy]);

        m_inner_tuner->begin();
        const unsigned int param = m_inner_tuner->getParam();
        const unsigned int block_size = param / 10000;
        const unsigned int tpp = param % 10000;
        gpu::inner_cell_thermo(args,
                               ci,
                               inner_ci,
                               lo,
                               m_sysdef->getNDimensions(),
                               block_size,
                               tpp);
        if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
        m_inner_tuner->end();
        }
    }

void mpcd::CellThermoComputeGPU::computeNetProperties()
    {
    if (m_prof) m_prof->push(m_exec_conf, "MPCD thermo");
    // first reduce the properties on the rank
        {
        const Index3D& ci = m_cl->getCellIndexer();
        uint3 upper = make_uint3(ci.getW(), ci.getH(), ci.getD());
        #ifdef ENABLE_MPI
        // in MPI, remove duplicate cells along direction of communication
        if (m_use_mpi)
            {
            auto num_comm = m_cl->getNComm();
            upper.x -= num_comm[static_cast<unsigned int>(mpcd::detail::face::east)];
            upper.y -= num_comm[static_cast<unsigned int>(mpcd::detail::face::north)];
            upper.z -= num_comm[static_cast<unsigned int>(mpcd::detail::face::up)];
            }
        #endif // ENABLE_MPI

        // temporary cell indexer for mapping 1d kernel threads to 3d grid
        const Index3D tmp_ci(upper.x, upper.y, upper.z);
        m_tmp_thermo.resize(tmp_ci.getNumElements());

        ArrayHandle<mpcd::detail::cell_thermo_element> d_tmp_thermo(m_tmp_thermo, access_location::device, access_mode::overwrite);
        ArrayHandle<double4> d_cell_vel(m_cell_vel, access_location::device, access_mode::read);
        ArrayHandle<double3> d_cell_energy(m_cell_energy, access_location::device, access_mode::read);

        m_stage_tuner->begin();
        mpcd::gpu::stage_net_cell_thermo(d_tmp_thermo.data,
                                         d_cell_vel.data,
                                         d_cell_energy.data,
                                         tmp_ci,
                                         ci,
                                         m_flags[mpcd::detail::thermo_options::energy],
                                         m_stage_tuner->getParam());
        if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
        m_stage_tuner->end();


        // use cub to reduce the properties on the gpu
        void *d_tmp = NULL;
        size_t tmp_bytes = 0;
        mpcd::gpu::reduce_net_cell_thermo(m_reduced.getDeviceFlags(),
                                          d_tmp,
                                          tmp_bytes,
                                          d_tmp_thermo.data,
                                          m_tmp_thermo.size());
        if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();

        ScopedAllocation<unsigned char> d_tmp_alloc(m_exec_conf->getCachedAllocator(), (tmp_bytes > 0) ? tmp_bytes : 1);
        d_tmp = (void*)d_tmp_alloc();

        mpcd::gpu::reduce_net_cell_thermo(m_reduced.getDeviceFlags(),
                                          d_tmp,
                                          tmp_bytes,
                                          d_tmp_thermo.data,
                                          m_tmp_thermo.size());
        if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
        }

    // now copy the net properties back to host from the flags
    unsigned int n_temp_cells = 0;
        {
        const mpcd::detail::cell_thermo_element reduced = m_reduced.readFlags();

        ArrayHandle<double> h_net_properties(m_net_properties, access_location::host, access_mode::overwrite);
        h_net_properties.data[mpcd::detail::thermo_index::momentum_x] = reduced.momentum.x;
        h_net_properties.data[mpcd::detail::thermo_index::momentum_y] = reduced.momentum.y;
        h_net_properties.data[mpcd::detail::thermo_index::momentum_z] = reduced.momentum.z;

        h_net_properties.data[mpcd::detail::thermo_index::energy] = reduced.energy;
        h_net_properties.data[mpcd::detail::thermo_index::temperature] = reduced.temperature;

        n_temp_cells = reduced.flag;
        }

    #ifdef ENABLE_MPI
    if (m_use_mpi)
        {
        ArrayHandle<double> h_net_properties(m_net_properties, access_location::host, access_mode::readwrite);
        MPI_Allreduce(MPI_IN_PLACE,
                      h_net_properties.data,
                      mpcd::detail::thermo_index::num_quantities,
                      MPI_DOUBLE,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());

        MPI_Allreduce(MPI_IN_PLACE, &n_temp_cells, 1, MPI_UNSIGNED, MPI_SUM, m_exec_conf->getMPICommunicator());
        }
    #endif // ENABLE_MPI

    if (n_temp_cells > 0)
        {
        ArrayHandle<double> h_net_properties(m_net_properties, access_location::host, access_mode::readwrite);
        h_net_properties.data[mpcd::detail::thermo_index::temperature] /= (double)n_temp_cells;
        }

    m_needs_net_reduce = false;
    if (m_prof) m_prof->pop(m_exec_conf);
    }

void mpcd::detail::export_CellThermoComputeGPU(pybind11::module& m)
    {
    namespace py = pybind11;

    py::class_<mpcd::CellThermoComputeGPU, std::shared_ptr<mpcd::CellThermoComputeGPU> >
        (m, "CellThermoComputeGPU", py::base<mpcd::CellThermoCompute>())
        .def(py::init< std::shared_ptr<mpcd::SystemData> >())
        .def(py::init< std::shared_ptr<mpcd::SystemData>, const std::string& >());
    }
