/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

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
#include "Communicator.h"
#endif

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
    if (exec_conf->isCUDAEnabled())
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
    if (exec_conf->isCUDAEnabled())
        cudaThreadSynchronize();
#endif
    uint64_t total_time_ns = t.getTime() - start_time;
    
    // convert the run time to milliseconds
    return double(total_time_ns) / 1e6 / double(num_iters);
    }

void NeighborListGPU::buildNlist(unsigned int timestep)
    {
    if (m_storage_mode != full)
        {
        m_exec_conf->msg->error() << "Only full mode nlists can be generated on the GPU" << endl;
        throw runtime_error("Error computing neighbor list");
        }

    if (m_filter_body || m_filter_diameter)
        {
        m_exec_conf->msg->error() << "NeighborListGPU does not currently support body or diameter exclusions." << endl;
        m_exec_conf->msg->error() << "Please contact the developers and notify them that you need this functionality" << endl;
        
        throw runtime_error("Error computing neighbor list");
        }

    // check that the simulation box is big enough
    const BoxDim& box = m_pdata->getBox();

    Scalar3 L = box.getL();

    if (L.x <= (m_r_cut+m_r_buff+m_d_max-Scalar(1.0)) * 2.0 ||
        L.y <= (m_r_cut+m_r_buff+m_d_max-Scalar(1.0)) * 2.0 ||
        L.z <= (m_r_cut+m_r_buff+m_d_max-Scalar(1.0)) * 2.0)
        {
        m_exec_conf->msg->error() << "Simulation box is too small! Particles would be interacting with themselves."
             << endl;
        throw runtime_error("Error computing neighbor list");
        }

    if (m_prof)
        m_prof->push(exec_conf, "compute");

    // acquire the particle data
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    // access the nlist data arrays
    ArrayHandle<unsigned int> d_nlist(m_nlist, access_location::device, access_mode::overwrite);
    ArrayHandle<unsigned int> d_n_neigh(m_n_neigh, access_location::device, access_mode::overwrite);
    ArrayHandle<unsigned int> d_ghost_nlist(m_ghost_nlist, access_location::device, access_mode::overwrite);
    ArrayHandle<unsigned int> d_n_ghost_neigh(m_n_ghost_neigh, access_location::device, access_mode::overwrite);
    bool compute_ghost_nlist = false;
#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        compute_ghost_nlist = true;
#endif


    ArrayHandle<Scalar4> d_last_pos(m_last_pos, access_location::device, access_mode::overwrite);

    // start by creating a temporary copy of r_cut sqaured
    Scalar rmax = m_r_cut + m_r_buff;
    // add d_max - 1.0, if diameter filtering is not already taking care of it
    if (!m_filter_diameter)
        rmax += m_d_max - Scalar(1.0);
    Scalar rmaxsq = rmax*rmax;

    gpu_compute_nlist_nsq(d_nlist.data,
                          d_n_neigh.data,
                          d_ghost_nlist.data,
                          d_n_ghost_neigh.data,
                          d_last_pos.data,
                          m_conditions.getDeviceFlags(),
                          m_nlist_indexer,
                          d_pos.data,
                          m_pdata->getN(),
                          m_pdata->getNGhosts(),
                          box,
                          rmaxsq,
                          compute_ghost_nlist);

    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    if (m_prof)
        m_prof->pop(exec_conf);
    }

bool NeighborListGPU::distanceCheck()
    {
    // scan through the particle data arrays and calculate distances
    if (m_prof) m_prof->push(exec_conf, "dist-check");
    
    // access data
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    BoxDim box = m_pdata->getBox();
    ArrayHandle<Scalar4> d_last_pos(m_last_pos, access_location::device, access_mode::read);

    // get current global box lengths
    Scalar3 L_g = m_pdata->getGlobalBox().getL();

    // Cutoff distance for inclusion in neighbor list
    Scalar rmax = m_r_cut + m_r_buff;
    if (!m_filter_diameter)
        rmax += m_d_max - Scalar(1.0);

    // Find direction of maximum box length contraction (smallest eigenvalue of deformation tensor)
    Scalar3 lambda = L_g / m_last_L;
    Scalar lambda_min = (lambda.x < lambda.y) ? lambda.x : lambda.y;
    lambda_min = (lambda_min < lambda.z) ? lambda_min : lambda.z;

    // maximum displacement for each particle (after subtraction of homogeneous dilations)
    Scalar delta_max = (rmax*lambda_min - m_r_cut)/Scalar(2.0);
    Scalar maxshiftsq = delta_max > 0  ? delta_max*delta_max : 0;

    bool check_out_of_bounds = false;
#ifdef ENABLE_MPI
    check_out_of_bounds = m_pdata->getDomainDecomposition();
#endif

    gpu_nlist_needs_update_check_new(m_flags.getDeviceFlags(),
                                     d_last_pos.data,
                                     d_pos.data,
                                     m_pdata->getN(),
                                     box,
                                     maxshiftsq,
                                     lambda,
                                     m_checkn,
                                     check_out_of_bounds);
    
    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    bool result;
    uint2 flags = m_flags.readFlags();
    result = (flags.x == m_checkn);

    if (check_out_of_bounds && (flags.x == m_checkn+1))
        {
        ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
        unsigned int tag = h_tag.data[flags.y];
        m_exec_conf->msg->error() << "nlist: Particle " << tag << " has moved more than one box length"
                                  << std::endl << "between neighbor list builds."
                                  << std::endl << std::endl;

        throw std::runtime_error("Error checking particle displacements");
        }

    m_checkn++;

    if (m_prof) m_prof->pop(exec_conf);

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        // use MPI all_reduce to check if the neighbor list build criterium is fulfilled on any processor
        int local_result = result ? 1 : 0;
        int global_result = 0;
        MPI_Allreduce(&local_result, &global_result, 1, MPI_INT, MPI_MAX, m_exec_conf->getMPICommunicator());
        result = (global_result > 0);
        }
#endif
    return result;
    }

/*! Calls gpu_nlsit_filter() to filter the neighbor list on the GPU
*/
void NeighborListGPU::filterNlist()
    {
    if (m_prof)
        m_prof->push(exec_conf, "filter");
    
    // access data
    
    ArrayHandle<unsigned int> d_n_ex_idx(m_n_ex_idx, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_ex_list_idx(m_ex_list_idx, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_n_neigh(m_n_neigh, access_location::device, access_mode::readwrite);
    ArrayHandle<unsigned int> d_nlist(m_nlist, access_location::device, access_mode::readwrite);
   
    gpu_nlist_filter(d_n_neigh.data,
                     d_nlist.data,
                     m_nlist_indexer,
                     d_n_ex_idx.data,
                     d_ex_list_idx.data,
                     m_ex_list_indexer,
                     m_pdata->getN(),
                     m_block_size_filter);
  
#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        // filter ghost neighbors
        ArrayHandle<unsigned int> d_n_ghost_neigh(m_n_ghost_neigh, access_location::device, access_mode::readwrite);
        ArrayHandle<unsigned int> d_ghost_nlist(m_ghost_nlist, access_location::device, access_mode::readwrite);

        gpu_nlist_filter(d_n_ghost_neigh.data,
                     d_ghost_nlist.data,
                     m_nlist_indexer,
                     d_n_ex_idx.data,
                     d_ex_list_idx.data,
                     m_ex_list_indexer,
                     m_pdata->getN(),
                     m_block_size_filter);
        }
#endif

    if (m_prof)
        m_prof->pop(exec_conf);
    }


//! Update the exclusion list on the GPU
void NeighborListGPU::updateExListIdx()
    {
    if (m_prof)
        m_prof->push("update-ex");
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

    if (m_prof)
        m_prof->pop();
    }

void export_NeighborListGPU()
    {
    class_<NeighborListGPU, boost::shared_ptr<NeighborListGPU>, bases<NeighborList>, boost::noncopyable >
                     ("NeighborListGPU", init< boost::shared_ptr<SystemDefinition>, Scalar, Scalar >())
                     .def("setBlockSizeFilter", &NeighborListGPU::setBlockSizeFilter)
                     .def("benchmarkFilter", &NeighborListGPU::benchmarkFilter)
                     ;
    }

