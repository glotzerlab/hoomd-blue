// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file CellListGPU.cc
    \brief Defines CellListGPU
*/

#include "CellListGPU.h"
#include "CellListGPU.cuh"

namespace py = pybind11;

using namespace std;

/*! \param sysdef system to compute the cell list of
*/
CellListGPU::CellListGPU(std::shared_ptr<SystemDefinition> sysdef)
    : CellList(sysdef), m_per_device(false)
    {
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating a CellListGPU with no GPU in the execution configuration" << endl;
        throw std::runtime_error("Error initializing CellListGPU");
        }

    m_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "cell_list", this->m_exec_conf));
    m_tuner_combine.reset(new Autotuner(32, 1024, 32, 5, 100000, "cell_list_combine", this->m_exec_conf));

    }

void CellListGPU::computeCellList()
    {
    if (m_prof)
        m_prof->push(m_exec_conf, "compute");

    // acquire the particle data
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_charge(m_pdata->getCharges(), access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_body(m_pdata->getBodies(), access_location::device, access_mode::read);

    BoxDim box = m_pdata->getBox();
    unsigned int ngpu = m_exec_conf->getNumActiveGPUs();


        {
        // access the cell list data arrays
        ArrayHandle<unsigned int> d_cell_size(m_cell_size, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar4> d_xyzf(m_xyzf, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar4> d_tdb(m_tdb, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar4> d_cell_orientation(m_orientation, access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> d_cell_idx(m_idx, access_location::device, access_mode::overwrite);

        // access the per-GPU cell list arrays (only needed with ngpu>1)
        ArrayHandle<unsigned int> d_cell_size_scratch(m_cell_size_scratch, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar4> d_xyzf_scratch(m_xyzf_scratch, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar4> d_tdb_scratch(m_tdb_scratch, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar4> d_cell_orientation_scratch(m_orientation_scratch, access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> d_cell_idx_scratch(m_idx_scratch, access_location::device, access_mode::overwrite);

        // error conditions
        ArrayHandle<uint3> d_conditions(m_conditions, access_location::device, access_mode::overwrite);

        // reset cell list contents
        cudaMemsetAsync(d_cell_size.data, 0, sizeof(unsigned int)*m_cell_indexer.getNumElements(),0);
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        if (ngpu > 1 || m_per_device)
            {
            // reset temporary arrays
            cudaMemsetAsync(d_cell_size_scratch.data, 0, sizeof(unsigned int)*m_cell_size_scratch.getNumElements(),0);
            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }

        m_exec_conf->beginMultiGPU();

        // autotune block sizes
        m_tuner->begin();

        // compute cell list, and write to temporary arrays with multi-GPU
        gpu_compute_cell_list((ngpu == 1 && !m_per_device) ? d_cell_size.data : d_cell_size_scratch.data,
                              (ngpu == 1 && !m_per_device) ? d_xyzf.data : d_xyzf_scratch.data,
                              (ngpu == 1 && !m_per_device) ? d_tdb.data : d_tdb_scratch.data,
                              (ngpu == 1 && !m_per_device) ? d_cell_orientation.data : d_cell_orientation_scratch.data,
                              (ngpu == 1 && !m_per_device) ? d_cell_idx.data : d_cell_idx_scratch.data,
                              d_conditions.data,
                              d_pos.data,
                              d_orientation.data,
                              d_charge.data,
                              d_diameter.data,
                              d_body.data,
                              m_pdata->getN(),
                              m_pdata->getNGhosts(),
                              m_Nmax,
                              m_flag_charge,
                              m_flag_type,
                              box,
                              m_cell_indexer,
                              m_cell_list_indexer,
                              getGhostWidth(),
                              m_tuner->getParam(),
                              m_pdata->getGPUPartition());
        if(m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_tuner->end();

        m_exec_conf->endMultiGPU();
        }

    if (m_sort_cell_list)
        {
        ArrayHandle<unsigned int> d_cell_size(m_cell_size, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar4> d_xyzf(m_xyzf, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar4> d_tdb(m_tdb, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar4> d_cell_orientation(m_orientation, access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> d_cell_idx(m_idx, access_location::device, access_mode::overwrite);

        // access the per-GPU cell list arrays (only needed with ngpu>1)
        ArrayHandle<unsigned int> d_cell_size_scratch(m_cell_size_scratch, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar4> d_xyzf_scratch(m_xyzf_scratch, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar4> d_tdb_scratch(m_tdb_scratch, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar4> d_cell_orientation_scratch(m_orientation_scratch, access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> d_cell_idx_scratch(m_idx_scratch, access_location::device, access_mode::overwrite);


        // sort partial cell lists
        for (unsigned int i = 0; i < m_exec_conf->getNumActiveGPUs(); ++i)
            {
            ScopedAllocation<uint2> d_sort_idx(m_exec_conf->getCachedAllocator(), m_cell_list_indexer.getNumElements());
            ScopedAllocation<unsigned int> d_sort_permutation(m_exec_conf->getCachedAllocator(), m_cell_list_indexer.getNumElements());
            ScopedAllocation<unsigned int> d_cell_idx_new(m_exec_conf->getCachedAllocator(), m_idx.getNumElements());
            ScopedAllocation<Scalar4> d_xyzf_new(m_exec_conf->getCachedAllocator(), m_xyzf.getNumElements());
            ScopedAllocation<Scalar4> d_cell_orientation_new(m_exec_conf->getCachedAllocator(), m_orientation.getNumElements());
            ScopedAllocation<Scalar4> d_tdb_new(m_exec_conf->getCachedAllocator(), m_tdb.getNumElements());

            gpu_sort_cell_list((ngpu == 1 && !m_per_device) ? d_cell_size.data : d_cell_size_scratch.data + i*m_cell_indexer.getNumElements(),
                               (ngpu == 1 && !m_per_device) ? d_xyzf.data : d_xyzf_scratch.data + i*m_cell_list_indexer.getNumElements(),
                               d_xyzf_new.data,
                               (ngpu == 1 && !m_per_device) ? d_tdb.data : d_tdb_scratch.data + i*m_cell_list_indexer.getNumElements(),
                               d_tdb_new.data,
                               (ngpu == 1 && !m_per_device) ? d_cell_orientation.data : d_cell_orientation_scratch.data + i*m_cell_list_indexer.getNumElements(),
                               d_cell_orientation_new.data,
                               (ngpu == 1 && !m_per_device) ? d_cell_idx.data : d_cell_idx_scratch.data + i*m_cell_list_indexer.getNumElements(),
                               d_cell_idx_new.data,
                               d_sort_idx.data,
                               d_sort_permutation.data,
                               m_cell_indexer,
                               m_cell_list_indexer);

            if(m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }
        }

    if (ngpu > 1 && !m_per_device)
        combineCellLists();

    if (m_prof)
        m_prof->pop(m_exec_conf);
    }

void CellListGPU::combineCellLists()
    {
    // access the cell list data arrays
    ArrayHandle<unsigned int> d_cell_size(m_cell_size, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_xyzf(m_xyzf, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_tdb(m_tdb, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_cell_orientation(m_orientation, access_location::device, access_mode::overwrite);
    ArrayHandle<unsigned int> d_cell_idx(m_idx, access_location::device, access_mode::overwrite);

    // access the per-GPU cell list arrays (only needed with per-device cell list)
    ArrayHandle<unsigned int> d_cell_size_scratch(m_cell_size_scratch, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_xyzf_scratch(m_xyzf_scratch, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_tdb_scratch(m_tdb_scratch, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_cell_orientation_scratch(m_orientation_scratch, access_location::device, access_mode::overwrite);
    ArrayHandle<unsigned int> d_cell_idx_scratch(m_idx_scratch, access_location::device, access_mode::overwrite);

    // error conditions
    ArrayHandle<uint3> d_conditions(m_conditions, access_location::device, access_mode::overwrite);

    // have to wait for all GPUs to sync up, to have cell sizes available
    m_exec_conf->beginMultiGPU();

    // autotune block sizes
    m_tuner_combine->begin();

    gpu_combine_cell_lists(d_cell_size_scratch.data,
                           d_cell_size.data,
                           d_cell_idx_scratch.data,
                           d_cell_idx.data,
                           d_xyzf_scratch.data,
                           d_xyzf.data,
                           d_tdb_scratch.data,
                           d_tdb.data,
                           d_cell_orientation_scratch.data,
                           d_cell_orientation.data,
                           m_cell_list_indexer,
                           m_exec_conf->getNumActiveGPUs(),
                           m_tuner_combine->getParam(),
                           m_Nmax,
                           d_conditions.data,
                           m_pdata->getGPUPartition());
    m_tuner_combine->end();

    m_exec_conf->endMultiGPU();
    }

void CellListGPU::initializeMemory()
    {
    // call base class method
    CellList::initializeMemory();

    // only need to keep separate cell lists with more than one GPU
    unsigned int ngpu = m_exec_conf->getNumActiveGPUs();

    if (ngpu == 1 && !m_per_device)
        return;

    m_exec_conf->msg->notice(10) << "CellListGPU initialize multiGPU memory" << endl;
    if (m_prof)
        m_prof->push("init");

    if (m_compute_adj_list && m_exec_conf->allConcurrentManagedAccess())
        {
        cudaMemAdvise(m_cell_adj.get(), m_cell_adj.getNumElements()*sizeof(unsigned int), cudaMemAdviseSetReadMostly, 0);
        CHECK_CUDA_ERROR();
        }

    // allocate memory
    GlobalArray<unsigned int> cell_size_scratch(m_cell_indexer.getNumElements()*ngpu, m_exec_conf);
    m_cell_size_scratch.swap(cell_size_scratch);
    TAG_ALLOCATION(m_cell_size_scratch);

    if (m_compute_xyzf)
        {
        GlobalArray<Scalar4> xyzf_scratch(m_cell_list_indexer.getNumElements()*ngpu, m_exec_conf);
        m_xyzf_scratch.swap(xyzf_scratch);
        TAG_ALLOCATION(m_xyzf_scratch);
        }
    else
        {
        GlobalArray<Scalar4> xyzf_scratch;
        m_xyzf_scratch.swap(xyzf_scratch);
        }

    if (m_compute_tdb)
        {
        GlobalArray<Scalar4> tdb_scratch(m_cell_list_indexer.getNumElements()*ngpu, m_exec_conf);
        m_tdb_scratch.swap(tdb_scratch);
        TAG_ALLOCATION(m_tdb_scratch);
        }
    else
        {
        // array is no longer needed, discard it
        GlobalArray<Scalar4> tdb_scratch;
        m_tdb_scratch.swap(tdb_scratch);
        }

    if (m_compute_orientation)
        {
        GlobalArray<Scalar4> orientation_scratch(m_cell_list_indexer.getNumElements()*ngpu, m_exec_conf);
        m_orientation_scratch.swap(orientation_scratch);
        TAG_ALLOCATION(m_orientation_scratch);
        }
    else
        {
        // array is no longer needed, discard it
        GlobalArray<Scalar4> orientation_scratch;
        m_orientation_scratch.swap(orientation_scratch);
        }

    if (m_compute_idx || m_sort_cell_list)
        {
        GlobalArray<unsigned int> idx_scratch(m_cell_list_indexer.getNumElements()*ngpu, m_exec_conf);
        m_idx_scratch.swap(idx_scratch);
        TAG_ALLOCATION(m_idx_scratch);
        }
    else
        {
        // array is no longer needed, discard it
        GlobalArray<unsigned int> idx_scratch;
        m_idx_scratch.swap(idx_scratch);
        }

    if (! m_exec_conf->allConcurrentManagedAccess())
        return;

    // map cell list arrays into memory of all active GPUs
    auto& gpu_map = m_exec_conf->getGPUIds();

    for (unsigned int idev = 0; idev < m_exec_conf->getNumActiveGPUs(); ++idev)
        {
        cudaMemAdvise(m_cell_size.get(), m_cell_size.getNumElements()*sizeof(unsigned int), cudaMemAdviseSetAccessedBy, gpu_map[idev]);
        }

    for (unsigned int idev = 0; idev < m_exec_conf->getNumActiveGPUs(); ++idev)
        {
        cudaMemAdvise(m_cell_size_scratch.get()+idev*m_cell_indexer.getNumElements(),
            m_cell_indexer.getNumElements()*sizeof(unsigned int), cudaMemAdviseSetPreferredLocation, gpu_map[idev]);

        if (! m_idx_scratch.isNull())
            cudaMemAdvise(m_idx_scratch.get()+idev*m_cell_list_indexer.getNumElements(),
                m_cell_list_indexer.getNumElements()*sizeof(unsigned int), cudaMemAdviseSetPreferredLocation, gpu_map[idev]);

        if (! m_xyzf_scratch.isNull())
            cudaMemAdvise(m_xyzf_scratch.get()+idev*m_cell_list_indexer.getNumElements(),
                m_cell_list_indexer.getNumElements()*sizeof(Scalar4), cudaMemAdviseSetPreferredLocation, gpu_map[idev]);

        if (! m_tdb_scratch.isNull())
            cudaMemAdvise(m_tdb_scratch.get()+idev*m_cell_list_indexer.getNumElements(),
            m_cell_list_indexer.getNumElements()*sizeof(Scalar4), cudaMemAdviseSetPreferredLocation, gpu_map[idev]);

        if (! m_orientation.isNull())
            cudaMemAdvise(m_orientation_scratch.get()+idev*m_cell_list_indexer.getNumElements(),
            m_cell_list_indexer.getNumElements()*sizeof(Scalar4), cudaMemAdviseSetPreferredLocation, gpu_map[idev]);

        // prefetch to preferred location
        cudaMemPrefetchAsync(m_cell_size_scratch.get()+idev*m_cell_indexer.getNumElements(),
            m_cell_indexer.getNumElements()*sizeof(unsigned int), gpu_map[idev]);

        if (! m_idx.isNull())
            cudaMemPrefetchAsync(m_idx_scratch.get()+idev*m_cell_list_indexer.getNumElements(),
                m_cell_list_indexer.getNumElements()*sizeof(unsigned int), gpu_map[idev]);

        if (! m_xyzf_scratch.isNull())
            cudaMemPrefetchAsync(m_xyzf_scratch.get()+idev*m_cell_list_indexer.getNumElements(),
                m_cell_list_indexer.getNumElements()*sizeof(Scalar4), gpu_map[idev]);

        if (! m_tdb_scratch.isNull())
            cudaMemPrefetchAsync(m_tdb_scratch.get()+idev*m_cell_list_indexer.getNumElements(),
            m_cell_list_indexer.getNumElements()*sizeof(Scalar4), gpu_map[idev]);

        if (! m_orientation_scratch.isNull())
            cudaMemPrefetchAsync(m_orientation_scratch.get()+idev*m_cell_list_indexer.getNumElements(),
            m_cell_list_indexer.getNumElements()*sizeof(Scalar4), gpu_map[idev]);
        }
    CHECK_CUDA_ERROR();

    if (m_prof)
        m_prof->pop();
    }

void export_CellListGPU(py::module& m)
    {
    py::class_<CellListGPU, std::shared_ptr<CellListGPU> >(m,"CellListGPU",py::base<CellList>())
    .def(py::init< std::shared_ptr<SystemDefinition> >())
        ;
    }
