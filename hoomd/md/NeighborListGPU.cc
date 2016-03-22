/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2016 The Regents of
the University of Michigan All rights reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: joaander

/*! \file NeighborListGPU.cc
    \brief Implementation of the NeighborListGPU class
*/

#include "NeighborListGPU.h"
#include "NeighborListGPU.cuh"

#include <boost/python.hpp>
using namespace boost::python;

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#endif

#include "CachedAllocator.h"

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
        cudaThreadSynchronize();
        CHECK_CUDA_ERROR();
        }
#endif

    // benchmark
    uint64_t start_time = t.getTime();
    for (unsigned int i = 0; i < num_iters; i++)
        filterNlist();

#ifdef ENABLE_CUDA
    if(m_exec_conf->isCUDAEnabled())
        cudaThreadSynchronize();
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

void NeighborListGPU::scheduleDistanceCheck(unsigned int timestep)
    {
    // prevent against unnecessary calls
    if (! shouldCheckDistance(timestep))
        {
        m_distcheck_scheduled = false;
        return;
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

    ArrayHandle<unsigned int> d_flags(m_flags, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> d_rcut_max(m_rcut_max, access_location::device, access_mode::read);
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
                                     ++m_checkn);

    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    m_distcheck_scheduled = true;
    m_last_schedule_tstep = timestep;

    // record synchronization point
    cudaEventRecord(m_event);

    if (m_prof) m_prof->pop(m_exec_conf);
    }

bool NeighborListGPU::distanceCheck(unsigned int timestep)
    {
    // check if we have scheduled a kernel for the current time step
    if (! m_distcheck_scheduled || m_last_schedule_tstep != timestep)
        scheduleDistanceCheck(timestep);

    m_distcheck_scheduled = false;

    ArrayHandleAsync<unsigned int> h_flags(m_flags, access_location::host, access_mode::read);

    // wait for kernel to complete
    cudaEventSynchronize(m_event);

    bool result = (*h_flags.data == m_checkn);

    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        if (m_prof) m_prof->push(m_exec_conf,"MPI allreduce");
        // check if migrate criterium is fulfilled on any rank
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


    return result;
    }

/*! Calls gpu_nlsit_filter() to filter the neighbor list on the GPU
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

    if (m_prof) m_prof->push(exec_conf, "head-list");

    ArrayHandle<unsigned int> d_head_list(m_head_list, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_Nmax(m_Nmax, access_location::device, access_mode::read);

    m_req_size_nlist.resetFlags(0);

    m_tuner_head_list->begin();
    gpu_nlist_build_head_list(d_head_list.data,
                              m_req_size_nlist.getDeviceFlags(),
                              d_Nmax.data,
                              d_pos.data,
                              m_pdata->getN(),
                              m_pdata->getNTypes(),
                              m_tuner_head_list->getParam());
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_head_list->end();

    unsigned int req_size_nlist = m_req_size_nlist.readFlags();
    resizeNlist(req_size_nlist);

    if (m_prof) m_prof->pop(exec_conf);
    }

void export_NeighborListGPU()
    {
    class_<NeighborListGPU, boost::shared_ptr<NeighborListGPU>, bases<NeighborList>, boost::noncopyable >
                     ("NeighborListGPU", init< boost::shared_ptr<SystemDefinition>, Scalar, Scalar >())
                     .def("benchmarkFilter", &NeighborListGPU::benchmarkFilter)
                     ;
    }
