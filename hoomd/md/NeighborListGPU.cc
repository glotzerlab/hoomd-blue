// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file NeighborListGPU.cc
    \brief Implementation of the NeighborListGPU class
*/

#include "NeighborListGPU.h"
#include "NeighborListGPU.cuh"

namespace py = pybind11;

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#endif

#include "hoomd/CachedAllocator.h"

#include <iostream>
using namespace std;

/*! \param num_iters Number of iterations to average for the benchmark
    \returns Milliseconds of execution time per calculation

    Calls filterNlist repeatedly to benchmark the neighbor list filter step.
*/
double NeighborListGPU::benchmarkFilter(unsigned int num_iters)
    {
    ClockSource t;
    // warm up run
    forceUpdate();
    compute(0);
    buildNlist(0);
    filterNlist();

#ifdef ENABLE_CUDA
    if(m_exec_conf->isCUDAEnabled())
        {
        cudaDeviceSynchronize();
        CHECK_CUDA_ERROR();
        }
#endif

    // benchmark
    uint64_t start_time = t.getTime();
    for (unsigned int i = 0; i < num_iters; i++)
        filterNlist();

#ifdef ENABLE_CUDA
    if(m_exec_conf->isCUDAEnabled())
        cudaDeviceSynchronize();
#endif
    uint64_t total_time_ns = t.getTime() - start_time;

    // convert the run time to milliseconds
    return double(total_time_ns) / 1e6 / double(num_iters);
    }

void NeighborListGPU::buildNlist(unsigned int timestep)
    {
    m_exec_conf->msg->error() << "nlist: O(N^2) neighbor lists are no longer supported." << endl;
    throw runtime_error("Error updating neighborlist bins");
    }

bool NeighborListGPU::distanceCheck(unsigned int timestep)
    {
    // prevent against unnecessary calls
    if (! shouldCheckDistance(timestep))
        {
        return false;
        }

    // scan through the particle data arrays and calculate distances
    if (m_prof) m_prof->push(m_exec_conf, "dist-check");

    // access data
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    BoxDim box = m_pdata->getBox();
    ArrayHandle<Scalar4> d_last_pos(m_last_pos, access_location::device, access_mode::read);

    // get current global nearest plane distance
    Scalar3 L_g = m_pdata->getGlobalBox().getNearestPlaneDistance();

    // Find direction of maximum box length contraction (smallest eigenvalue of deformation tensor)
    Scalar3 lambda = L_g / m_last_L;
    Scalar lambda_min = (lambda.x < lambda.y) ? lambda.x : lambda.y;
    lambda_min = (lambda_min < lambda.z) ? lambda_min : lambda.z;

    ArrayHandle<Scalar> d_rcut_max(m_rcut_max, access_location::device, access_mode::read);

        {
        ArrayHandle<unsigned int> d_flags(m_flags, access_location::device, access_mode::readwrite);

        m_exec_conf->beginMultiGPU();

        gpu_nlist_needs_update_check_new(d_flags.data,
                                         d_last_pos.data,
                                         d_pos.data,
                                         m_pdata->getN(),
                                         box,
                                         d_rcut_max.data,
                                         m_r_buff,
                                         m_pdata->getNTypes(),
                                         lambda_min,
                                         lambda,
                                         ++m_checkn,
                                         m_pdata->getGPUPartition());

        if(m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        m_exec_conf->endMultiGPU();
        }

    bool result;
        {
        // read back flags
        ArrayHandle<unsigned int> h_flags(m_flags, access_location::host, access_mode::read);
        result = m_checkn == *h_flags.data;
        }

    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        if (m_prof) m_prof->push(m_exec_conf,"MPI allreduce");
        // check if migrate criterion is fulfilled on any rank
        int local_result = result ? 1 : 0;
        int global_result = 0;
        MPI_Allreduce(&local_result,
            &global_result,
            1,
            MPI_INT,
            MPI_MAX,
            m_exec_conf->getMPICommunicator());
        result = (global_result > 0);
        if (m_prof) m_prof->pop();
        }
    #endif

    if (m_prof) m_prof->pop(m_exec_conf);

    return result;
    }

/*! Calls gpu_nlist_filter() to filter the neighbor list on the GPU
*/
void NeighborListGPU::filterNlist()
    {
    if (m_prof)
        m_prof->push(m_exec_conf, "filter");

    // access data

    ArrayHandle<unsigned int> d_n_ex_idx(m_n_ex_idx, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_ex_list_idx(m_ex_list_idx, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_n_neigh(m_n_neigh, access_location::device, access_mode::readwrite);
    ArrayHandle<unsigned int> d_nlist(m_nlist, access_location::device, access_mode::readwrite);
    ArrayHandle<unsigned int> d_head_list(m_head_list, access_location::device, access_mode::read);

    m_tuner_filter->begin();
    gpu_nlist_filter(d_n_neigh.data,
                     d_nlist.data,
                     d_head_list.data,
                     d_n_ex_idx.data,
                     d_ex_list_idx.data,
                     m_ex_list_indexer,
                     m_pdata->getN(),
                     m_tuner_filter->getParam());
    if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
    m_tuner_filter->end();

    if (m_prof)
        m_prof->pop(m_exec_conf);
    }


//! Update the exclusion list on the GPU
void NeighborListGPU::updateExListIdx()
    {
    assert(! m_need_reallocate_exlist);

    if (m_prof)
        m_prof->push(m_exec_conf,"update-ex");

    ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);

    ArrayHandle<unsigned int> d_n_ex_tag(m_n_ex_tag, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_ex_list_tag(m_ex_list_tag, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_n_ex_idx(m_n_ex_idx, access_location::device, access_mode::overwrite);
    ArrayHandle<unsigned int> d_ex_list_idx(m_ex_list_idx, access_location::device, access_mode::overwrite);

    gpu_update_exclusion_list(d_tag.data,
                              d_rtag.data,
                              d_n_ex_tag.data,
                              d_ex_list_tag.data,
                              m_ex_list_indexer_tag,
                              d_n_ex_idx.data,
                              d_ex_list_idx.data,
                              m_ex_list_indexer,
                              m_pdata->getN());
    if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();

    if (m_prof)
        m_prof->pop(m_exec_conf);
    }

//! Build the head list for neighbor list indexing on the GPU
void NeighborListGPU::buildHeadList()
    {
    // don't do anything if there are no particles owned by this rank
    if (!m_pdata->getN())
        return;

    if (m_prof) m_prof->push(m_exec_conf, "head-list");

        {
        ArrayHandle<unsigned int> h_req_size_nlist(m_req_size_nlist, access_location::host, access_mode::overwrite);
        // reset flags
        *h_req_size_nlist.data = 0;
        }

        {
        ArrayHandle<unsigned int> d_head_list(m_head_list, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_Nmax(m_Nmax, access_location::device, access_mode::read);

        ArrayHandle<unsigned int> d_req_size_nlist(m_req_size_nlist, access_location::device, access_mode::readwrite);

        m_tuner_head_list->begin();
        gpu_nlist_build_head_list(d_head_list.data,
                                  d_req_size_nlist.data,
                                  d_Nmax.data,
                                  d_pos.data,
                                  m_pdata->getN(),
                                  m_pdata->getNTypes(),
                                  m_tuner_head_list->getParam());
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_tuner_head_list->end();
        }

    unsigned int req_size_nlist;
        {
        ArrayHandle<unsigned int> h_req_size_nlist(m_req_size_nlist, access_location::host, access_mode::read);
        req_size_nlist = *h_req_size_nlist.data;
        }

    resizeNlist(req_size_nlist);

    // now that the head list is complete and the neighbor list has been allocated, update memory advice
    updateMemoryMapping();

    if (m_prof) m_prof->pop(m_exec_conf);
    }

void export_NeighborListGPU(py::module& m)
    {
    py::class_<NeighborListGPU, std::shared_ptr<NeighborListGPU> >(m, "NeighborListGPU", py::base<NeighborList>())
    .def(py::init< std::shared_ptr<SystemDefinition>, Scalar, Scalar >())
                     .def("benchmarkFilter", &NeighborListGPU::benchmarkFilter)
                     ;
    }
