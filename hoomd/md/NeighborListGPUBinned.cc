// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file NeighborListGPUBinned.cc
    \brief Defines NeighborListGPUBinned
*/

#include "NeighborListGPUBinned.h"
#include "NeighborListGPUBinned.cuh"

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#endif

namespace hoomd
    {
namespace md
    {
NeighborListGPUBinned::NeighborListGPUBinned(std::shared_ptr<SystemDefinition> sysdef,
                                             Scalar r_buff)
    : NeighborListGPU(sysdef, r_buff), m_cl(std::make_shared<CellListGPU>(sysdef)), m_param(0)
    {
    // with multiple GPUs, use indirect access via particle data arrays
    m_use_index = m_exec_conf->allConcurrentManagedAccess();

    // with multiple GPUs, request a cell list per device
    m_cl->setPerDevice(m_exec_conf->allConcurrentManagedAccess());

    m_cl->setComputeXYZF(!m_use_index);
    m_cl->setComputeIdx(m_use_index);

    m_cl->setRadius(1);
    m_cl->setComputeTDB(!m_use_index);
    m_cl->setFlagIndex();

    CHECK_CUDA_ERROR();

    // initialize autotuner
    // the full block size and threads_per_particle matrix is searched,
    // encoded as block_size*10000 + threads_per_particle
    std::vector<unsigned int> valid_params;

    const unsigned int warp_size = m_exec_conf->dev_prop.warpSize;
    for (unsigned int block_size = warp_size; block_size <= 1024; block_size += warp_size)
        {
        unsigned int s = 1;

        while (s <= warp_size)
            {
            valid_params.push_back(block_size * 10000 + s);
            s = s * 2;
            }
        }

    m_tuner.reset(new Autotuner(valid_params, 5, 100000, "nlist_binned", this->m_exec_conf));
    }

NeighborListGPUBinned::~NeighborListGPUBinned() { }

void NeighborListGPUBinned::buildNlist(uint64_t timestep)
    {
    if (m_storage_mode != full)
        {
        throw std::runtime_error("GPU neighbor lists require a full storage mode.");
        }

    // update the cell list size if needed
    if (m_update_cell_size)
        {
        Scalar rmax = getMaxRCut() + m_r_buff;
        if (m_diameter_shift)
            rmax += m_d_max - Scalar(1.0);

        m_cl->setNominalWidth(rmax);
        m_update_cell_size = false;
        }

    m_cl->compute(timestep);

    if (m_prof)
        m_prof->push(m_exec_conf, "compute");

    // acquire the particle data
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(),
                                   access_location::device,
                                   access_mode::read);
    ArrayHandle<unsigned int> d_body(m_pdata->getBodies(),
                                     access_location::device,
                                     access_mode::read);

    const BoxDim& box = m_pdata->getBox();

    // access the cell list data arrays
    ArrayHandle<unsigned int> d_cell_size(m_cl->getCellSizeArray(),
                                          access_location::device,
                                          access_mode::read);
    ArrayHandle<Scalar4> d_cell_xyzf(m_cl->getXYZFArray(),
                                     access_location::device,
                                     access_mode::read);
    ArrayHandle<unsigned int> d_cell_idx(m_cl->getIndexArray(),
                                         access_location::device,
                                         access_mode::read);
    ArrayHandle<Scalar4> d_cell_tdb(m_cl->getTDBArray(),
                                    access_location::device,
                                    access_mode::read);
    ArrayHandle<unsigned int> d_cell_adj(m_cl->getCellAdjArray(),
                                         access_location::device,
                                         access_mode::read);

    const ArrayHandle<unsigned int>& d_cell_size_per_device
        = m_cl->getPerDevice() ? ArrayHandle<unsigned int>(m_cl->getCellSizeArrayPerDevice(),
                                                           access_location::device,
                                                           access_mode::read)
                               : ArrayHandle<unsigned int>(GlobalArray<unsigned int>(),
                                                           access_location::device,
                                                           access_mode::read);
    const ArrayHandle<unsigned int>& d_cell_idx_per_device
        = m_cl->getPerDevice() ? ArrayHandle<unsigned int>(m_cl->getIndexArrayPerDevice(),
                                                           access_location::device,
                                                           access_mode::read)
                               : ArrayHandle<unsigned int>(GlobalArray<unsigned int>(),
                                                           access_location::device,
                                                           access_mode::read);

    ArrayHandle<size_t> d_head_list(m_head_list, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_Nmax(m_Nmax, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_conditions(m_conditions,
                                           access_location::device,
                                           access_mode::readwrite);
    ArrayHandle<unsigned int> d_nlist(m_nlist, access_location::device, access_mode::overwrite);
    ArrayHandle<unsigned int> d_n_neigh(m_n_neigh, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_last_pos(m_last_pos, access_location::device, access_mode::overwrite);

    ArrayHandle<Scalar> d_r_cut(m_r_cut, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_r_listsq(m_r_listsq, access_location::device, access_mode::read);

#ifdef __HIP_PLATFORM_NVCC__
    auto& gpu_map = m_exec_conf->getGPUIds();

    // prefetch some cell list arrays
    if (m_exec_conf->allConcurrentManagedAccess())
        {
        for (unsigned int idev = 0; idev < m_exec_conf->getNumActiveGPUs(); ++idev)
            {
            // prefetch cell adjacency
            cudaMemPrefetchAsync(d_cell_adj.data,
                                 m_cl->getCellAdjArray().getNumElements() * sizeof(unsigned int),
                                 gpu_map[idev]);
            }
        }

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
#endif

    m_exec_conf->beginMultiGPU();

    this->m_tuner->begin();
    unsigned int param = !m_param ? this->m_tuner->getParam() : m_param;
    unsigned int block_size = param / 10000;
    unsigned int threads_per_particle = param % 10000;

    kernel::gpu_compute_nlist_binned(
        d_nlist.data,
        d_n_neigh.data,
        d_last_pos.data,
        d_conditions.data,
        d_Nmax.data,
        d_head_list.data,
        d_pos.data,
        d_body.data,
        d_diameter.data,
        m_pdata->getN(),
        m_cl->getPerDevice() ? d_cell_size_per_device.data : d_cell_size.data,
        d_cell_xyzf.data,
        m_cl->getPerDevice() ? d_cell_idx_per_device.data : d_cell_idx.data,
        d_cell_tdb.data,
        d_cell_adj.data,
        m_cl->getCellIndexer(),
        m_cl->getCellListIndexer(),
        m_cl->getCellAdjIndexer(),
        box,
        d_r_cut.data,
        m_r_buff,
        m_pdata->getNTypes(),
        threads_per_particle,
        block_size,
        m_filter_body,
        m_diameter_shift,
        m_cl->getGhostWidth(),
        m_pdata->getGPUPartition(),
        m_use_index);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    this->m_tuner->end();

    m_exec_conf->endMultiGPU();

    if (m_prof)
        m_prof->pop(m_exec_conf);
    }

namespace detail
    {
void export_NeighborListGPUBinned(pybind11::module& m)
    {
    pybind11::class_<NeighborListGPUBinned,
                     NeighborListGPU,
                     std::shared_ptr<NeighborListGPUBinned>>(m, "NeighborListGPUBinned")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, Scalar>())
        .def("setTuningParam", &NeighborListGPUBinned::setTuningParam)
        .def_property("deterministic",
                      &NeighborListGPUBinned::getDeterministic,
                      &NeighborListGPUBinned::setDeterministic);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
