/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
// Maintainer: joaander

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include <boost/bind.hpp>

#include "NeighborList.h"
#include "BondData.h"
#include "AngleData.h"
#include "DihedralData.h"

#include <sstream>
#include <fstream>

#include <iostream>
#include <stdexcept>

using namespace boost;
using namespace std;

#ifdef ENABLE_CUDA
#include "gpu_settings.h"
#endif

/*! \file NeighborList.cc
    \brief Defines the NeighborList class
*/

/*! \param sysdef System the neighborlist is to compute neighbors for
    \param r_cut Cuttoff radius under which particles are considered neighbors
    \param r_buff Buffere radius around \a r_cut in which neighbors will be included

    \post NeighborList is initialized and the list memory has been allocated,
        but the list will not be computed until compute is called.
    \post The storage mode defaults to half
*/
NeighborList::NeighborList(boost::shared_ptr<SystemDefinition> sysdef, Scalar r_cut, Scalar r_buff)
    : Compute(sysdef), m_r_cut(r_cut), m_r_buff(r_buff), m_storage_mode(half), m_exclude_same_body(false),
      m_updates(0), m_forced_updates(0), m_dangerous_updates(0), m_force_update(true)
    {
    // check for two sensless errors the user could make
    if (m_r_cut < 0.0)
        {
        cerr << endl << "***Error! Requested cuttoff radius for neighborlist less than zero" << endl << endl;
        throw runtime_error("Error initializing NeighborList");
        }
        
    if (m_r_buff < 0.0)
        {
        cerr << endl << "***Error! Requested cuttoff radius for neighborlist less than zero" << endl << endl;
        throw runtime_error("Error initializing NeighborList");
        }
        
    // allocate the list memory
    m_list.resize(m_pdata->getN());
    m_exclusions.resize(m_pdata->getN());
#if defined(LARGE_EXCLUSION_LIST)
    m_exclusions2.resize(m_pdata->getN());
    m_exclusions3.resize(m_pdata->getN());
    m_exclusions4.resize(m_pdata->getN());
#endif
    
    // allocate memory for storing the last particle positions
    m_last_x = new Scalar[m_pdata->getN()];
    m_last_y = new Scalar[m_pdata->getN()];
    m_last_z = new Scalar[m_pdata->getN()];
    
    assert(m_last_x);
    assert(m_last_y);
    assert(m_last_z);
    
    // zero data
    memset((void*)m_last_x, 0, sizeof(Scalar)*m_pdata->getN());
    memset((void*)m_last_y, 0, sizeof(Scalar)*m_pdata->getN());
    memset((void*)m_last_z, 0, sizeof(Scalar)*m_pdata->getN());
    
    m_last_updated_tstep = 0;
    m_every = 0;
    
#ifdef ENABLE_CUDA
    // setup the memory pointer list for all GPUs
    m_gpu_nlist.resize(exec_conf.gpu.size());
    
    // initialize the GPU and CPU mirror structures
    // there really should be a better way to determine the initial height, but we will just
    // choose a given value for now (choose it initially small to test the auto-expansion
    // code
    if (!exec_conf.gpu.empty())
        {
        allocateGPUData(32);
        m_data_location = cpugpu;
        hostToDeviceCopy();
        }
    else
        {
        m_data_location = cpu;
        }
#endif
        
    m_sort_connection = m_pdata->connectParticleSort(bind(&NeighborList::forceUpdate, this));
    }

NeighborList::~NeighborList()
    {
    delete[] m_last_x;
    delete[] m_last_y;
    delete[] m_last_z;
    
#ifdef ENABLE_CUDA
    if (!exec_conf.gpu.empty())
        freeGPUData();
#endif
        
    m_sort_connection.disconnect();
    }

#ifdef ENABLE_CUDA
void NeighborList::allocateGPUData(int height)
    {
    size_t pitch;
    const int N = m_pdata->getN();
    
    exec_conf.tagAll(__FILE__, __LINE__);
    for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
        {
        // allocate and zero device memory
        // alloate one extra row so that compute kernels can safely prefetch "past" the end
        // without causing memory errors
        exec_conf.gpu[cur_gpu]->call(bind(cudaMallocPitchHack, (void**)((void*)&m_gpu_nlist[cur_gpu].list), &pitch, N*sizeof(unsigned int), height + 1));
        // want pitch in elements, not bytes
        m_gpu_nlist[cur_gpu].pitch = (int)pitch / sizeof(int);
        m_gpu_nlist[cur_gpu].height = height;
        exec_conf.gpu[cur_gpu]->call(bind(cudaMemset, (void*)m_gpu_nlist[cur_gpu].list, 0, pitch * height));
        
        exec_conf.gpu[cur_gpu]->call(bind(cudaMallocHack, (void**)((void*)&m_gpu_nlist[cur_gpu].n_neigh), pitch));
        exec_conf.gpu[cur_gpu]->call(bind(cudaMemset, (void*)m_gpu_nlist[cur_gpu].n_neigh, 0, pitch));
        
        exec_conf.gpu[cur_gpu]->call(bind(cudaMallocHack, (void**)((void*)&m_gpu_nlist[cur_gpu].last_updated_pos), m_gpu_nlist[cur_gpu].pitch*sizeof(float4)));
        exec_conf.gpu[cur_gpu]->call(bind(cudaMemset, (void*)m_gpu_nlist[cur_gpu].last_updated_pos, 0, m_gpu_nlist[cur_gpu].pitch * sizeof(float4)));
        
        exec_conf.gpu[cur_gpu]->call(bind(cudaMallocHack, (void**)((void*)&m_gpu_nlist[cur_gpu].needs_update), sizeof(int)));
        exec_conf.gpu[cur_gpu]->call(bind(cudaMallocHack, (void**)((void*)&m_gpu_nlist[cur_gpu].overflow), sizeof(int)));
        
        exec_conf.gpu[cur_gpu]->call(bind(cudaMallocHack, (void**)((void*)&m_gpu_nlist[cur_gpu].exclusions), m_gpu_nlist[cur_gpu].pitch*sizeof(uint4)));
        exec_conf.gpu[cur_gpu]->call(bind(cudaMemset, (void*) m_gpu_nlist[cur_gpu].exclusions, 0xff, m_gpu_nlist[cur_gpu].pitch * sizeof(uint4)));
#if defined(LARGE_EXCLUSION_LIST)
        exec_conf.gpu[cur_gpu]->call(bind(cudaMallocHack, (void**)((void*)&m_gpu_nlist[cur_gpu].exclusions2), m_gpu_nlist[cur_gpu].pitch*sizeof(uint4)));
        exec_conf.gpu[cur_gpu]->call(bind(cudaMemset, (void*) m_gpu_nlist[cur_gpu].exclusions2, 0xff, m_gpu_nlist[cur_gpu].pitch * sizeof(uint4)));
        exec_conf.gpu[cur_gpu]->call(bind(cudaMallocHack, (void**)((void*)&m_gpu_nlist[cur_gpu].exclusions3), m_gpu_nlist[cur_gpu].pitch*sizeof(uint4)));
        exec_conf.gpu[cur_gpu]->call(bind(cudaMemset, (void*) m_gpu_nlist[cur_gpu].exclusions3, 0xff, m_gpu_nlist[cur_gpu].pitch * sizeof(uint4)));
        exec_conf.gpu[cur_gpu]->call(bind(cudaMallocHack, (void**)((void*)&m_gpu_nlist[cur_gpu].exclusions4), m_gpu_nlist[cur_gpu].pitch*sizeof(uint4)));
        exec_conf.gpu[cur_gpu]->call(bind(cudaMemset, (void*) m_gpu_nlist[cur_gpu].exclusions4, 0xff, m_gpu_nlist[cur_gpu].pitch * sizeof(uint4)));
#endif
        }
        
    // allocate and zero host memory
    exec_conf.gpu[0]->call(bind(cudaHostAllocHack, (void**)((void*)&m_host_nlist), pitch * height, cudaHostAllocPortable));
    memset((void*)m_host_nlist, 0, pitch*height);
    
    exec_conf.gpu[0]->call(bind(cudaHostAllocHack, (void**)((void*)&m_host_n_neigh), N * sizeof(unsigned int), cudaHostAllocPortable));
    memset((void*)m_host_n_neigh, 0, N * sizeof(unsigned int));
    
    exec_conf.gpu[0]->call(bind(cudaHostAllocHack, (void**)((void*)&m_host_exclusions), N * sizeof(uint4), cudaHostAllocPortable));
    memset((void*)m_host_exclusions, 0xff, N * sizeof(uint4));
#if defined(LARGE_EXCLUSION_LIST)
    exec_conf.gpu[0]->call(bind(cudaHostAllocHack, (void**)((void*)&m_host_exclusions2), N * sizeof(uint4), cudaHostAllocPortable));
    memset((void*)m_host_exclusions2, 0xff, N * sizeof(uint4));
    exec_conf.gpu[0]->call(bind(cudaHostAllocHack, (void**)((void*)&m_host_exclusions3), N * sizeof(uint4), cudaHostAllocPortable));
    memset((void*)m_host_exclusions3, 0xff, N * sizeof(uint4));
    exec_conf.gpu[0]->call(bind(cudaHostAllocHack, (void**)((void*)&m_host_exclusions4), N * sizeof(uint4), cudaHostAllocPortable));
    memset((void*)m_host_exclusions4, 0xff, N * sizeof(uint4));
#endif
    }

void NeighborList::freeGPUData()
    {
    exec_conf.gpu[0]->setTag(__FILE__, __LINE__);
    
    assert(m_host_nlist);
    assert(m_host_exclusions);
#if defined(LARGE_EXCLUSION_LIST)
    assert(m_host_exclusions2);
    assert(m_host_exclusions3);
    assert(m_host_exclusions4);
#endif
    assert(m_host_n_neigh);
    
    exec_conf.gpu[0]->call(bind(cudaFreeHost, m_host_nlist));
    m_host_nlist = NULL;
    exec_conf.gpu[0]->call(bind(cudaFreeHost, m_host_n_neigh));
    m_host_n_neigh = NULL;
    exec_conf.gpu[0]->call(bind(cudaFreeHost, m_host_exclusions));
    m_host_exclusions = NULL;
#if defined(LARGE_EXCLUSION_LIST)
    exec_conf.gpu[0]->call(bind(cudaFreeHost, m_host_exclusions2));
    m_host_exclusions2 = NULL;
    exec_conf.gpu[0]->call(bind(cudaFreeHost, m_host_exclusions3));
    m_host_exclusions3 = NULL;
    exec_conf.gpu[0]->call(bind(cudaFreeHost, m_host_exclusions4));
    m_host_exclusions4 = NULL;
#endif
    
    exec_conf.tagAll(__FILE__, __LINE__);
    for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
        {
        assert(m_gpu_nlist[cur_gpu].list);
        assert(m_gpu_nlist[cur_gpu].n_neigh);
        assert(m_gpu_nlist[cur_gpu].exclusions);
#if defined(LARGE_EXCLUSION_LIST)
        assert(m_gpu_nlist[cur_gpu].exclusions2);
        assert(m_gpu_nlist[cur_gpu].exclusions3);
        assert(m_gpu_nlist[cur_gpu].exclusions4);
#endif
        assert(m_gpu_nlist[cur_gpu].last_updated_pos);
        assert(m_gpu_nlist[cur_gpu].needs_update);
        
        exec_conf.gpu[cur_gpu]->call(bind(cudaFree, m_gpu_nlist[cur_gpu].n_neigh));
        m_gpu_nlist[cur_gpu].n_neigh = NULL;
        exec_conf.gpu[cur_gpu]->call(bind(cudaFree, m_gpu_nlist[cur_gpu].list));
        m_gpu_nlist[cur_gpu].list = NULL;
        exec_conf.gpu[cur_gpu]->call(bind(cudaFree, m_gpu_nlist[cur_gpu].exclusions));
        m_gpu_nlist[cur_gpu].exclusions = NULL;
#if defined(LARGE_EXCLUSION_LIST)
        exec_conf.gpu[cur_gpu]->call(bind(cudaFree, m_gpu_nlist[cur_gpu].exclusions2));
        m_gpu_nlist[cur_gpu].exclusions2 = NULL;
        exec_conf.gpu[cur_gpu]->call(bind(cudaFree, m_gpu_nlist[cur_gpu].exclusions3));
        m_gpu_nlist[cur_gpu].exclusions3 = NULL;
        exec_conf.gpu[cur_gpu]->call(bind(cudaFree, m_gpu_nlist[cur_gpu].exclusions4));
        m_gpu_nlist[cur_gpu].exclusions4 = NULL;
#endif
        exec_conf.gpu[cur_gpu]->call(bind(cudaFree, m_gpu_nlist[cur_gpu].last_updated_pos));
        m_gpu_nlist[cur_gpu].last_updated_pos = NULL;
        exec_conf.gpu[cur_gpu]->call(bind(cudaFree, m_gpu_nlist[cur_gpu].needs_update));
        m_gpu_nlist[cur_gpu].needs_update = NULL;
        exec_conf.gpu[cur_gpu]->call(bind(cudaFree, m_gpu_nlist[cur_gpu].overflow));
        m_gpu_nlist[cur_gpu].overflow = NULL;
        }
    }
#endif

/*! \param every Number of time steps to wait before beignning to check if particles have moved a sufficient distance
    to require a neighbor list upate.
*/
void NeighborList::setEvery(unsigned int every)
    {
    m_every = every;
    }

/*! Updates the neighborlist if it has not yet been updated this times step
    \param timestep Current time step of the simulation
*/
void NeighborList::compute(unsigned int timestep)
    {
    // skip if we shouldn't compute this step
    if (!shouldCompute(timestep) && !m_force_update)
        return;
        
    if (m_prof) m_prof->push("Neighbor");
    
#ifdef ENABLE_CUDA
    // update the exclusion data if this is a forced update
    if (m_force_update)
        updateExclusionData();
#endif
        
    // check if the list needs to be updated and update it
    if (needsUpdating(timestep))
        buildNlist();
        
    if (m_prof) m_prof->pop();
    }

/*! \param num_iters Number of iterations to average for the benchmark
    \returns Milliseconds of execution time per calculation

    Calls buildNlist repeatedly to benchmark the neighbor list.
*/
double NeighborList::benchmark(unsigned int num_iters)
    {
    if (m_prof) m_prof->push("Neighbor");
    
    ClockSource t;
    // warm up run
    buildNlist();;
    
#ifdef ENABLE_CUDA
    exec_conf.callAll(bind(cudaThreadSynchronize));
#endif
    
    // benchmark
    uint64_t start_time = t.getTime();
    for (unsigned int i = 0; i < num_iters; i++)
        buildNlist();
        
#ifdef ENABLE_CUDA
    exec_conf.callAll(bind(cudaThreadSynchronize));
#endif
    uint64_t total_time_ns = t.getTime() - start_time;
    
    if (m_prof) m_prof->pop();
    
    // convert the run time to milliseconds
    return double(total_time_ns) / 1e6 / double(num_iters);
    }

/*! \param r_cut New cuttoff radius to set
    \param r_buff New buffer radius to set
    \note Changing the cuttoff radius does NOT immeadiately update the neighborlist.
            The new cuttoff will take effect when compute is called for the next timestep.
*/
void NeighborList::setRCut(Scalar r_cut, Scalar r_buff)
    {
    m_r_cut = r_cut;
    m_r_buff = r_buff;
    
    // check for two sensless errors the user could make
    if (m_r_cut < 0.0)
        {
        cerr << endl << "***Error! Requested cuttoff radius for neighborlist less than zero" << endl << endl;
        throw runtime_error("Error changing NeighborList parameters");
        }
        
    if (m_r_buff < 0.0)
        {
        cerr << endl << "***Error! Requested cuttoff radius for neighborlist less than zero" << endl << endl;
        throw runtime_error("Error changing NeighborList parameters");
        }
        
    forceUpdate();
    }

/*! \return Reference to the neighbor list table
    If the neighbor list was last updated on the GPU, it is copied to the CPU first.
    This copy operation is only intended for debugging and status information purposes.
    It is not optimized in any way, and is quite slow.
*/
const std::vector< std::vector<unsigned int> >& NeighborList::getList()
    {
#ifdef ENABLE_CUDA
    
    // this is the complicated graphics card version, need to do some work
    // switch based on the current location of the data
    switch (m_data_location)
        {
        case cpu:
            // if the data is solely on the cpu, life is easy, return the data arrays
            // and stay in the same state
            return m_list;
            break;
        case cpugpu:
            // if the data is up to date on both the cpu and gpu, life is easy, return
            // the data arrays and stay in the same state
            return m_list;
            break;
        case gpu:
            // if the data resides on the gpu, it needs to be copied back to the cpu
            // this changes to the cpugpu state since the data is now fully up to date on
            // both
            deviceToHostCopy();
            m_data_location = cpugpu;
            return m_list;
            break;
        default:
            // anything other than the above is an undefined state!
            assert(false);
            return m_list;
            break;
        }
        
#else
        
    return m_list;
#endif
    }

/*! \returns an estimate of the number of neighbors per particle
    This mean-field estimate may be very bad dending on how clustered particles are.
    Derived classes can override this method to provide better estimates.

    \note Under NO circumstances should calling this method produce any
    appreciable amount of overhead. This is mainly a warning to
    derived classes.
*/
Scalar NeighborList::estimateNNeigh()
    {
    // calculate a number density of particles
    BoxDim box = m_pdata->getBox();
    Scalar vol = (box.xhi - box.xlo)*(box.yhi - box.ylo)*(box.zhi - box.zlo);
    Scalar n_dens = Scalar(m_pdata->getN()) / vol;
    
    // calculate the average number of neighbors by multiplying by the volume
    // within the cutoff
    Scalar r_max = m_r_cut + m_r_buff;
    Scalar vol_cut = Scalar(4.0/3.0 * M_PI) * r_max * r_max * r_max;
    return n_dens * vol_cut;
    }

#ifdef ENABLE_CUDA
/*! \returns Neighbor list data structure stored on the GPU.
    If the neighbor list was last updated on the CPU, calling this routine will result
    in a very time consuming copy to the device. It is meant only as a debugging/testing
    path and not for production simulations.
*/
vector<gpu_nlist_array>& NeighborList::getListGPU()
    {
    // this is the complicated graphics card version, need to do some work
    // switch based on the current location of the data
    switch (m_data_location)
        {
        case cpu:
            // if the data is on the cpu, we need to copy it over to the gpu
            hostToDeviceCopy();
            // now we are in the cpugpu state
            m_data_location = cpugpu;
            return m_gpu_nlist;
            break;
        case cpugpu:
            // if the data is up to date on both the cpu and gpu, life is easy
            // state remains the same
            return m_gpu_nlist;
            break;
        case gpu:
            // if the data resides on the gpu, life is easy, just make sure that
            // state remains the same
            return m_gpu_nlist;
            break;
        default:
            // anything other than the above is an undefined state!
            assert(false);
            return m_gpu_nlist;
            break;
        }
    }

/*! \post The entire neighbor list is copied from the CPU to the GPU.
    The copy is not optimized in any fashion and will be quite slow.
*/
void NeighborList::hostToDeviceCopy()
    {
    // commenting profiling: enable when benchmarking suspected slow portions of the code. This isn't needed all the time
    // if (m_prof) m_prof->push("NLIST C2G");
    
    const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
    assert(exec_conf.gpu.size() >= 1);
    
    // start by determining if we need to make the device list larger or not
    // find the maximum neighborlist height
    unsigned int max_h = 0;
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        if (m_list[i].size() > max_h)
            max_h = (unsigned int)m_list[i].size();
        }
        
    // if the largest nlist is bigger than the capacity of the device nlist,
    // make it 10% bigger (note that the capacity of the device list is height-1 since
    // the number of neighbors is stored in the first row)
    if (max_h > m_gpu_nlist[0].height)
        {
        freeGPUData();
        allocateGPUData((unsigned int)(float(max_h)*1.1));
        }
        
    // now we are good to copy the data over
    // start by zeroing the list
    memset(m_host_nlist, 0, sizeof(unsigned int) * m_gpu_nlist[0].pitch * m_gpu_nlist[0].height);
    memset(m_host_n_neigh, 0, sizeof(unsigned int) *m_pdata->getN());
    
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        // fill out the first row with the length of each list
        m_host_n_neigh[i] = (unsigned int)m_list[i].size();
        
        // now fill out the data
        for (unsigned int j = 0; j < m_list[i].size(); j++)
            m_host_nlist[j*m_gpu_nlist[0].pitch + i] = m_list[i][j];
        }
        
    exec_conf.tagAll(__FILE__, __LINE__);
    for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
        {
        // now that the host array is filled out, copy it to the card
        exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, m_gpu_nlist[cur_gpu].list, m_host_nlist,
                                          sizeof(unsigned int) * m_gpu_nlist[cur_gpu].height * m_gpu_nlist[cur_gpu].pitch,
                                          cudaMemcpyHostToDevice));
        exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, m_gpu_nlist[cur_gpu].n_neigh, m_host_n_neigh,
                                          sizeof(unsigned int) * m_pdata->getN(),
                                          cudaMemcpyHostToDevice));
        }
        
    // if (m_prof) m_prof->pop();
    }

/*! \post The entire neighbor list is copied from the GPU to the CPU.
    The copy is not optimized in any fashion and will be quite slow.
*/
void NeighborList::deviceToHostCopy()
    {
    // commenting profiling: enable when benchmarking suspected slow portions of the code. This isn't needed all the time
    // if (m_prof) m_prof->push("NLIST G2C");
    
    // clear out host version of the list
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        m_list[i].clear();
        
    exec_conf.tagAll(__FILE__, __LINE__);
    for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
        {
        // copy data back from the card
        exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, m_host_nlist, m_gpu_nlist[cur_gpu].list,
                                          sizeof(unsigned int) * m_gpu_nlist[cur_gpu].height * m_gpu_nlist[cur_gpu].pitch,
                                          cudaMemcpyDeviceToHost));
        exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, m_host_n_neigh, m_gpu_nlist[cur_gpu].n_neigh,
                                          sizeof(unsigned int) * m_pdata->getN(),
                                          cudaMemcpyDeviceToHost));
                                          
        // fill out host version of the list
        for (unsigned int i = m_pdata->getLocalBeg(cur_gpu); i < m_pdata->getLocalBeg(cur_gpu) + m_pdata->getLocalNum(cur_gpu); i++)
            {
            // now loop over all elements in the array
            unsigned int size = m_host_n_neigh[i];
            for (unsigned int j = 0; j < size; j++)
                m_list[i].push_back(m_host_nlist[j*m_gpu_nlist[0].pitch + i]);
            }
        }
        
    // if (m_prof) m_prof->pop();
    }

/*! \post The exclusion list is converted from tags to indicies and then copied up to the
    GPU.
*/
void NeighborList::updateExclusionData()
    {
    // cannot update unless we are running on a GPU
    if (exec_conf.gpu.empty())
        return;
        
    ParticleDataArraysConst arrays = m_pdata->acquireReadOnly();
    
    // setup each of the exclusions
    for (unsigned int tag_i = 0; tag_i < m_pdata->getN(); tag_i++)
        {
        unsigned int i = arrays.rtag[tag_i];
        if (m_exclusions[tag_i].e1 == EXCLUDE_EMPTY)
            m_host_exclusions[i].x = EXCLUDE_EMPTY;
        else
            m_host_exclusions[i].x = arrays.rtag[m_exclusions[tag_i].e1];
            
        if (m_exclusions[tag_i].e2 == EXCLUDE_EMPTY)
            m_host_exclusions[i].y = EXCLUDE_EMPTY;
        else
            m_host_exclusions[i].y = arrays.rtag[m_exclusions[tag_i].e2];
            
        if (m_exclusions[tag_i].e3 == EXCLUDE_EMPTY)
            m_host_exclusions[i].z = EXCLUDE_EMPTY;
        else
            m_host_exclusions[i].z = arrays.rtag[m_exclusions[tag_i].e3];
            
        if (m_exclusions[tag_i].e4 == EXCLUDE_EMPTY)
            m_host_exclusions[i].w = EXCLUDE_EMPTY;
        else
            m_host_exclusions[i].w = arrays.rtag[m_exclusions[tag_i].e4];
            
#if defined(LARGE_EXCLUSION_LIST)
        if (m_exclusions2[tag_i].e1 == EXCLUDE_EMPTY)
            m_host_exclusions2[i].x = EXCLUDE_EMPTY;
        else
            m_host_exclusions2[i].x = arrays.rtag[m_exclusions2[tag_i].e1];
            
        if (m_exclusions2[tag_i].e2 == EXCLUDE_EMPTY)
            m_host_exclusions2[i].y = EXCLUDE_EMPTY;
        else
            m_host_exclusions2[i].y = arrays.rtag[m_exclusions2[tag_i].e2];
            
        if (m_exclusions2[tag_i].e3 == EXCLUDE_EMPTY)
            m_host_exclusions2[i].z = EXCLUDE_EMPTY;
        else
            m_host_exclusions2[i].z = arrays.rtag[m_exclusions2[tag_i].e3];
            
        if (m_exclusions2[tag_i].e4 == EXCLUDE_EMPTY)
            m_host_exclusions2[i].w = EXCLUDE_EMPTY;
        else
            m_host_exclusions2[i].w = arrays.rtag[m_exclusions2[tag_i].e4];
            
        if (m_exclusions3[tag_i].e1 == EXCLUDE_EMPTY)
            m_host_exclusions3[i].x = EXCLUDE_EMPTY;
        else
            m_host_exclusions3[i].x = arrays.rtag[m_exclusions3[tag_i].e1];
            
        if (m_exclusions3[tag_i].e2 == EXCLUDE_EMPTY)
            m_host_exclusions3[i].y = EXCLUDE_EMPTY;
        else
            m_host_exclusions3[i].y = arrays.rtag[m_exclusions3[tag_i].e2];
            
        if (m_exclusions3[tag_i].e3 == EXCLUDE_EMPTY)
            m_host_exclusions3[i].z = EXCLUDE_EMPTY;
        else
            m_host_exclusions3[i].z = arrays.rtag[m_exclusions3[tag_i].e3];
            
        if (m_exclusions3[tag_i].e4 == EXCLUDE_EMPTY)
            m_host_exclusions3[i].w = EXCLUDE_EMPTY;
        else
            m_host_exclusions3[i].w = arrays.rtag[m_exclusions3[tag_i].e4];
            
        if (m_exclusions4[tag_i].e1 == EXCLUDE_EMPTY)
            m_host_exclusions4[i].x = EXCLUDE_EMPTY;
        else
            m_host_exclusions4[i].x = arrays.rtag[m_exclusions4[tag_i].e1];
            
        if (m_exclusions4[tag_i].e2 == EXCLUDE_EMPTY)
            m_host_exclusions4[i].y = EXCLUDE_EMPTY;
        else
            m_host_exclusions4[i].y = arrays.rtag[m_exclusions4[tag_i].e2];
            
        if (m_exclusions4[tag_i].e3 == EXCLUDE_EMPTY)
            m_host_exclusions4[i].z = EXCLUDE_EMPTY;
        else
            m_host_exclusions4[i].z = arrays.rtag[m_exclusions4[tag_i].e3];
            
        if (m_exclusions4[tag_i].e4 == EXCLUDE_EMPTY)
            m_host_exclusions4[i].w = EXCLUDE_EMPTY;
        else
            m_host_exclusions4[i].w = arrays.rtag[m_exclusions4[tag_i].e4];
#endif
        }
        
    exec_conf.tagAll(__FILE__, __LINE__);
    for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
        {
        exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, m_gpu_nlist[cur_gpu].exclusions, m_host_exclusions,
                                          sizeof(uint4) * m_pdata->getN(),
                                          cudaMemcpyHostToDevice));
#if defined(LARGE_EXCLUSION_LIST)
        exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, m_gpu_nlist[cur_gpu].exclusions2, m_host_exclusions2,
                                          sizeof(uint4) * m_pdata->getN(),
                                          cudaMemcpyHostToDevice));
        exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, m_gpu_nlist[cur_gpu].exclusions3, m_host_exclusions3,
                                          sizeof(uint4) * m_pdata->getN(),
                                          cudaMemcpyHostToDevice));
        exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, m_gpu_nlist[cur_gpu].exclusions4, m_host_exclusions4,
                                          sizeof(uint4) * m_pdata->getN(),
                                          cudaMemcpyHostToDevice));
#endif
        }
        
    m_pdata->release();
    }

#endif

/*! \param mode Storage mode to set
    - half only stores neighbors where i < j
    - full stores all neighbors

    The neighborlist is not immediately updated to reflect this change. It will take effect
    when compute is called for the next timestep.
*/
void NeighborList::setStorageMode(storageMode mode)
    {
    m_storage_mode = mode;
    forceUpdate();
    }

/*! \param tag1 TAG (not index) of the first particle in the pair
    \param tag2 TAG (not index) of the second particle in the pair
    \post The pair \a tag1, \a tag2 will not appear in the neighborlist
    \note This only takes effect on the next call to compute() that updates the list
    \note Only 4 particles can be excluded from a single particle's neighbor list,
    \note unless the code is compiled with the option LARGE_EXCLUSION_LIST, in which
    \note case the maximum is 16 exclusions. Duplicates are checked for and not added.
*/
void NeighborList::addExclusion(unsigned int tag1, unsigned int tag2)
    {
    if (tag1 >= m_pdata->getN() || tag2 >= m_pdata->getN())
        {
        cerr << endl << "***Error! Particle tag out of bounds when attempting to add neighborlist exclusion: " << tag1 << "," << tag2 << endl << endl;
        throw runtime_error("Error setting exclusion in NeighborList");
        }
    // don't add an exclusion twice and waste space in the memory restricted exclusion lists.
    if (isExcluded(tag1, tag2))
        return;
        
    // add tag2 to tag1's exculsion list
    if (m_exclusions[tag1].e1 == EXCLUDE_EMPTY)
        m_exclusions[tag1].e1 = tag2;
    else if (m_exclusions[tag1].e2 == EXCLUDE_EMPTY)
        m_exclusions[tag1].e2 = tag2;
    else if (m_exclusions[tag1].e3 == EXCLUDE_EMPTY)
        m_exclusions[tag1].e3 = tag2;
    else if (m_exclusions[tag1].e4 == EXCLUDE_EMPTY)
        m_exclusions[tag1].e4 = tag2;
#if defined(LARGE_EXCLUSION_LIST)
    else if (m_exclusions2[tag1].e1 == EXCLUDE_EMPTY)
        m_exclusions2[tag1].e1 = tag2;
    else if (m_exclusions2[tag1].e2 == EXCLUDE_EMPTY)
        m_exclusions2[tag1].e2 = tag2;
    else if (m_exclusions2[tag1].e3 == EXCLUDE_EMPTY)
        m_exclusions2[tag1].e3 = tag2;
    else if (m_exclusions2[tag1].e4 == EXCLUDE_EMPTY)
        m_exclusions2[tag1].e4 = tag2;
    else if (m_exclusions3[tag1].e1 == EXCLUDE_EMPTY)
        m_exclusions3[tag1].e1 = tag2;
    else if (m_exclusions3[tag1].e2 == EXCLUDE_EMPTY)
        m_exclusions3[tag1].e2 = tag2;
    else if (m_exclusions3[tag1].e3 == EXCLUDE_EMPTY)
        m_exclusions3[tag1].e3 = tag2;
    else if (m_exclusions3[tag1].e4 == EXCLUDE_EMPTY)
        m_exclusions3[tag1].e4 = tag2;
    else if (m_exclusions4[tag1].e1 == EXCLUDE_EMPTY)
        m_exclusions4[tag1].e1 = tag2;
    else if (m_exclusions4[tag1].e2 == EXCLUDE_EMPTY)
        m_exclusions4[tag1].e2 = tag2;
    else if (m_exclusions4[tag1].e3 == EXCLUDE_EMPTY)
        m_exclusions4[tag1].e3 = tag2;
    else if (m_exclusions4[tag1].e4 == EXCLUDE_EMPTY)
        m_exclusions4[tag1].e4 = tag2;
#endif
    else
        {
        // error: exclusion list full
        cerr << endl << "***Error! Exclusion list full for particle with tag: " << tag1 << endl << endl;
        throw runtime_error("Error setting exclusion in NeighborList");
        }
        
    // add tag1 to tag2's exclusion list
    if (m_exclusions[tag2].e1 == EXCLUDE_EMPTY)
        m_exclusions[tag2].e1 = tag1;
    else if (m_exclusions[tag2].e2 == EXCLUDE_EMPTY)
        m_exclusions[tag2].e2 = tag1;
    else if (m_exclusions[tag2].e3 == EXCLUDE_EMPTY)
        m_exclusions[tag2].e3 = tag1;
    else if (m_exclusions[tag2].e4 == EXCLUDE_EMPTY)
        m_exclusions[tag2].e4 = tag1;
#if defined(LARGE_EXCLUSION_LIST)
    else if (m_exclusions2[tag2].e1 == EXCLUDE_EMPTY)
        m_exclusions2[tag2].e1 = tag1;
    else if (m_exclusions2[tag2].e2 == EXCLUDE_EMPTY)
        m_exclusions2[tag2].e2 = tag1;
    else if (m_exclusions2[tag2].e3 == EXCLUDE_EMPTY)
        m_exclusions2[tag2].e3 = tag1;
    else if (m_exclusions2[tag2].e4 == EXCLUDE_EMPTY)
        m_exclusions2[tag2].e4 = tag1;
    else if (m_exclusions3[tag2].e1 == EXCLUDE_EMPTY)
        m_exclusions3[tag2].e1 = tag1;
    else if (m_exclusions3[tag2].e2 == EXCLUDE_EMPTY)
        m_exclusions3[tag2].e2 = tag1;
    else if (m_exclusions3[tag2].e3 == EXCLUDE_EMPTY)
        m_exclusions3[tag2].e3 = tag1;
    else if (m_exclusions3[tag2].e4 == EXCLUDE_EMPTY)
        m_exclusions3[tag2].e4 = tag1;
    else if (m_exclusions4[tag2].e1 == EXCLUDE_EMPTY)
        m_exclusions4[tag2].e1 = tag1;
    else if (m_exclusions4[tag2].e2 == EXCLUDE_EMPTY)
        m_exclusions4[tag2].e2 = tag1;
    else if (m_exclusions4[tag2].e3 == EXCLUDE_EMPTY)
        m_exclusions4[tag2].e3 = tag1;
    else if (m_exclusions4[tag2].e4 == EXCLUDE_EMPTY)
        m_exclusions4[tag2].e4 = tag1;
#endif
    else
        {
        // error: exclusion list full
        cerr << endl << "***Error! Exclusion list full for particle with tag: " << tag2 << endl << endl;
        throw runtime_error("Error setting exclusion in NeighborList");
        }
    forceUpdate();
    }

/*! \post No particles are excluded from the neighbor list
*/
void NeighborList::clearExclusions()
    {
    for (unsigned int i = 0; i < m_exclusions.size(); i++)
        {
        m_exclusions[i].e1 = EXCLUDE_EMPTY;
        m_exclusions[i].e2 = EXCLUDE_EMPTY;
        m_exclusions[i].e3 = EXCLUDE_EMPTY;
        m_exclusions[i].e4 = EXCLUDE_EMPTY;
        }
#if defined(LARGE_EXCLUSION_LIST)
    for (unsigned int i = 0; i < m_exclusions2.size(); i++)
        {
        m_exclusions2[i].e1 = EXCLUDE_EMPTY;
        m_exclusions2[i].e2 = EXCLUDE_EMPTY;
        m_exclusions2[i].e3 = EXCLUDE_EMPTY;
        m_exclusions2[i].e4 = EXCLUDE_EMPTY;
        }
    for (unsigned int i = 0; i < m_exclusions3.size(); i++)
        {
        m_exclusions3[i].e1 = EXCLUDE_EMPTY;
        m_exclusions3[i].e2 = EXCLUDE_EMPTY;
        m_exclusions3[i].e3 = EXCLUDE_EMPTY;
        m_exclusions3[i].e4 = EXCLUDE_EMPTY;
        }
    for (unsigned int i = 0; i < m_exclusions4.size(); i++)
        {
        m_exclusions4[i].e1 = EXCLUDE_EMPTY;
        m_exclusions4[i].e2 = EXCLUDE_EMPTY;
        m_exclusions4[i].e3 = EXCLUDE_EMPTY;
        m_exclusions4[i].e4 = EXCLUDE_EMPTY;
        }
#endif
    forceUpdate();
    }

/*! \post Gather some statistics about exclusions usage.
*/
void NeighborList::countExclusions()
    {
    unsigned int excluded_count[MAX_NUM_EXCLUDED+1];
    unsigned int num_excluded, max_num_excluded;
    
    max_num_excluded = 0;
    for (unsigned int c=0; c <= MAX_NUM_EXCLUDED; ++c)
        excluded_count[c] = 0;
        
#if !defined(LARGE_EXCLUSION_LIST)
    for (unsigned int i = 0; i < m_exclusions.size(); i++)
        {
        num_excluded = 0;
        if (m_exclusions[i].e1 != EXCLUDE_EMPTY) ++num_excluded;
        if (m_exclusions[i].e2 != EXCLUDE_EMPTY) ++num_excluded;
        if (m_exclusions[i].e3 != EXCLUDE_EMPTY) ++num_excluded;
        if (m_exclusions[i].e4 != EXCLUDE_EMPTY) ++num_excluded;
        if (num_excluded > max_num_excluded) max_num_excluded = num_excluded;
        excluded_count[num_excluded] += 1;
        }
#else
    for (unsigned int i = 0; i < m_exclusions.size(); i++)
        {
        num_excluded = 0;
        if (m_exclusions[i].e1 != EXCLUDE_EMPTY) ++num_excluded;
        if (m_exclusions[i].e2 != EXCLUDE_EMPTY) ++num_excluded;
        if (m_exclusions[i].e3 != EXCLUDE_EMPTY) ++num_excluded;
        if (m_exclusions[i].e4 != EXCLUDE_EMPTY) ++num_excluded;
        if (m_exclusions2[i].e1 != EXCLUDE_EMPTY) ++num_excluded;
        if (m_exclusions2[i].e2 != EXCLUDE_EMPTY) ++num_excluded;
        if (m_exclusions2[i].e3 != EXCLUDE_EMPTY) ++num_excluded;
        if (m_exclusions2[i].e4 != EXCLUDE_EMPTY) ++num_excluded;
        if (m_exclusions3[i].e1 != EXCLUDE_EMPTY) ++num_excluded;
        if (m_exclusions3[i].e2 != EXCLUDE_EMPTY) ++num_excluded;
        if (m_exclusions3[i].e3 != EXCLUDE_EMPTY) ++num_excluded;
        if (m_exclusions3[i].e4 != EXCLUDE_EMPTY) ++num_excluded;
        if (m_exclusions4[i].e1 != EXCLUDE_EMPTY) ++num_excluded;
        if (m_exclusions4[i].e2 != EXCLUDE_EMPTY) ++num_excluded;
        if (m_exclusions4[i].e3 != EXCLUDE_EMPTY) ++num_excluded;
        if (m_exclusions4[i].e4 != EXCLUDE_EMPTY) ++num_excluded;
        if (num_excluded > max_num_excluded) max_num_excluded = num_excluded;
        excluded_count[num_excluded] += 1;
        }
#endif
    cout << "-- Neighborlist exclusion statistics:" << endl;
    cout << "Max. number of exclusions: " << max_num_excluded << endl;
    for (unsigned int i=0; i <= max_num_excluded; ++i)
        cout << "Particles with " << i << " exclusions: " << excluded_count[i] << endl;
    }

/*! After calling addExclusionFromBonds() all bonds specified in the attached ParticleData will be
    added as exlusions. Any additional bonds added after this will not be automatically added as exclusions.
*/
void NeighborList::addExclusionsFromBonds()
    {
    boost::shared_ptr<BondData> bond_data = m_sysdef->getBondData();
    
    // for each bond
    for (unsigned int i = 0; i < bond_data->getNumBonds(); i++)
        {
        // add an exclusion
        Bond bond = bond_data->getBond(i);
        addExclusion(bond.a, bond.b);
        }
    }

/*! After calling addExclusionsFromAngles(), all angles specified in the attached ParticleData will be added to the
    exclusion list. Only the two end particles in the angle are excluded from interacting.
*/
void NeighborList::addExclusionsFromAngles()
    {
    boost::shared_ptr<AngleData> angle_data = m_sysdef->getAngleData();
    
    // for each bond
    for (unsigned int i = 0; i < angle_data->getNumAngles(); i++)
        {
        Angle angle = angle_data->getAngle(i);
        addExclusion(angle.a, angle.c);
        }
    }

/*! After calling addExclusionsFromAngles(), all dihedrals specified in the attached ParticleData will be added to the
    exclusion list. Only the two end particles in the dihedral are excluded from interacting.
*/
void NeighborList::addExclusionsFromDihedrals()
    {
    boost::shared_ptr<DihedralData> dihedral_data = m_sysdef->getDihedralData();
    
    // for each bond
    for (unsigned int i = 0; i < dihedral_data->getNumDihedrals(); i++)
        {
        Dihedral dihedral = dihedral_data->getDihedral(i);
        addExclusion(dihedral.a, dihedral.d);
        }
    }

/*! \param tag1 First particle tag in the pair
    \param tag2 Second particle tag in the pair
    \return true if the particles \a tag1 and \a tag2 have been excluded from the neighbor list
*/
bool NeighborList::isExcluded(unsigned int tag1, unsigned int tag2)
    {
    if (tag1 >= m_pdata->getN() || tag2 >= m_pdata->getN())
        {
        cerr << endl << "***Error! Particle tag out of bounds when attempting to add neighborlist exclusion: " << tag1 << "," << tag2 << endl << endl;
        throw runtime_error("Error setting exclusion in NeighborList");
        }
        
    if (m_exclusions[tag1].e1 == tag2)
        return true;
    if (m_exclusions[tag1].e2 == tag2)
        return true;
    if (m_exclusions[tag1].e3 == tag2)
        return true;
    if (m_exclusions[tag1].e4 == tag2)
        return true;
#if defined(LARGE_EXCLUSION_LIST)
    if (m_exclusions2[tag1].e1 == tag2)
        return true;
    if (m_exclusions2[tag1].e2 == tag2)
        return true;
    if (m_exclusions2[tag1].e3 == tag2)
        return true;
    if (m_exclusions2[tag1].e4 == tag2)
        return true;
    if (m_exclusions3[tag1].e1 == tag2)
        return true;
    if (m_exclusions3[tag1].e2 == tag2)
        return true;
    if (m_exclusions3[tag1].e3 == tag2)
        return true;
    if (m_exclusions3[tag1].e4 == tag2)
        return true;
    if (m_exclusions4[tag1].e1 == tag2)
        return true;
    if (m_exclusions4[tag1].e2 == tag2)
        return true;
    if (m_exclusions4[tag1].e3 == tag2)
        return true;
    if (m_exclusions4[tag1].e4 == tag2)
        return true;
#endif
    return false;
    }

/*! Add topologically derived exclusions for angles
 *
 * This excludes all non-bonded interactions between all pairs particles
 * that are bonded to the same atom.
 * To make the process quasi-linear scaling with system size we first
 * create a 1-d array the collects the number and index of bond partners.
 */
void NeighborList::addOneThreeExclusionsFromTopology()
    {
    boost::shared_ptr<BondData> bond_data = m_sysdef->getBondData();
    const unsigned int myNAtoms = m_pdata->getN();
    const unsigned int MAXNBONDS = 7+1; //! assumed maximum number of bonds per atom plus one entry for the number of bonds.
    const unsigned int nBonds = bond_data->getNumBonds();
    
    if (nBonds == 0)
        {
        cout << "***Warning! No bonds defined while trying to add topology derived 1-3 exclusions" << endl;
        return;
        }
        
    // build a per atom list with all bonding partners from the list of bonds.
    unsigned int *localBondList = new unsigned int[MAXNBONDS*myNAtoms];
    memset((void *)localBondList,0,sizeof(unsigned int)*MAXNBONDS*myNAtoms);
    
    for (unsigned int i = 0; i < nBonds; i++)
        {
        // loop over all bonds and make a 1D exlcusion map
        Bond bondi = bond_data->getBond(i);
        const unsigned int tagA = bondi.a;
        const unsigned int tagB = bondi.b;
        
        // next, incrememt the number of bonds, and update the tags
        const unsigned int nBondsA = ++localBondList[tagA*MAXNBONDS];
        const unsigned int nBondsB = ++localBondList[tagB*MAXNBONDS];
        
        if (nBondsA >= MAXNBONDS)
            {
            cerr << endl << "***Error! Too many bonds to process exclusions for particle with tag: " << tagA << endl
                 << "***Error! Maximum allowed is currently: " << MAXNBONDS-1 << endl;
            throw runtime_error("Error setting up toplogical exclusions in NeighborList");
            }
            
        if (nBondsB >= MAXNBONDS)
            {
            cerr << endl << "***Error! Too many bonds to process exclusions for particle with tag: " << tagB << endl
                 << "***Error! Maximum allowed is currently: " << MAXNBONDS-1 << endl;
            throw runtime_error("Error setting up toplogical exclusions in NeighborList");
            }
            
        localBondList[tagA*MAXNBONDS + nBondsA] = tagB;
        localBondList[tagB*MAXNBONDS + nBondsB] = tagA;
        }
        
    // now loop over the atoms and build exclusions if we have more than
    // one bonding partner, i.e. we are in the center of an angle.
    for (unsigned int i = 0; i < myNAtoms; i++)
        {
        // now, loop over all atoms, and find those in the middle of an angle
        const unsigned int iAtom = i*MAXNBONDS;
        const unsigned int nBonds = localBondList[iAtom];
        
        if (nBonds > 1) // need at least two bonds
            {
            for (unsigned int j = 1; j < nBonds; ++j)
                {
                for (unsigned int k = j+1; k <= nBonds; ++k)
                    addExclusion(localBondList[iAtom+j],localBondList[iAtom+k]);
                }
            }
        }
    // free temp memory
    delete[] localBondList;
    }

/*! Add topologically derived exclusions for dihedrals
 *
 * This excludes all non-bonded interactions between all pairs particles
 * that are connected to a common bond.
 *
 * To make the process quasi-linear scaling with system size we first
 * create a 1-d array the collects the number and index of bond partners.
 * and then loop over bonded partners.
 */
void NeighborList::addOneFourExclusionsFromTopology()
    {
    boost::shared_ptr<BondData> bond_data = m_sysdef->getBondData();
    const unsigned int myNAtoms = m_pdata->getN();
    const unsigned int MAXNBONDS = 7+1; //! assumed maximum number of bonds per atom plus one entry for the number of bonds.
    const unsigned int nBonds = bond_data->getNumBonds();
    
    if (nBonds == 0)
        {
        cout << "***Warning! No bonds defined while trying to add topology derived 1-4 exclusions" << endl;
        return;
        }
        
    // allocate and clear data.
    unsigned int *localBondList = new unsigned int[MAXNBONDS*myNAtoms];
    memset((void *)localBondList,0,sizeof(unsigned int)*MAXNBONDS*myNAtoms);
    
    for (unsigned int i = 0; i < nBonds; i++)
        {
        // loop over all bonds and make a 1D exlcusion map
        Bond bondi = bond_data->getBond(i);
        const unsigned int tagA = bondi.a;
        const unsigned int tagB = bondi.b;
        
        // next, incrememt the number of bonds, and update the tags
        const unsigned int nBondsA = ++localBondList[tagA*MAXNBONDS];
        const unsigned int nBondsB = ++localBondList[tagB*MAXNBONDS];
        
        if (nBondsA >= MAXNBONDS)
            {
            cerr << endl << "***Error! Too many bonds to process exclusions for particle with tag: " << tagA << endl
                 << "***Error! Maximum allowed is currently: " << MAXNBONDS-1 << endl;
            throw runtime_error("Error setting up toplogical exclusions in NeighborList");
            }
            
        if (nBondsB >= MAXNBONDS)
            {
            cerr << endl << "***Error! Too many bonds to process exclusions for particle with tag: " << tagB << endl
                 << "***Error! Maximum allowed is currently: " << MAXNBONDS-1 << endl;
            throw runtime_error("Error setting up toplogical exclusions in NeighborList");
            }
            
        localBondList[tagA*MAXNBONDS + nBondsA] = tagB;
        localBondList[tagB*MAXNBONDS + nBondsB] = tagA;
        }
        
    //  loop over all bonds
    for (unsigned int i = 0; i < nBonds; i++)
        {
        Bond bondi = bond_data->getBond(i);
        const unsigned int tagA = bondi.a;
        const unsigned int tagB = bondi.b;
        
        const unsigned int nBondsA = localBondList[tagA*MAXNBONDS];
        const unsigned int nBondsB = localBondList[tagB*MAXNBONDS];
        
        for (unsigned int j = 1; j <= nBondsA; j++)
            {
            const unsigned int tagJ = localBondList[tagA*MAXNBONDS+j];
            if (tagJ == tagB) // skip the bond in the middle of the dihedral
                continue;
                
            for (unsigned int k = 1; k <= nBondsB; k++)
                {
                const unsigned int tagK = localBondList[tagB*MAXNBONDS+k];
                if (tagK == tagA) // skip the bond in the middle of the dihedral
                    continue;
                    
                addExclusion(tagJ,tagK);
                }
            }
        }
    // free temp memory
    delete[] localBondList;
    }


/*! \returns true If any of the particles have been moved more than 1/2 of the buffer distance since the last call
        to this method that returned true.
    \returns false If none of the particles has been moved more than 1/2 of the buffer distance since the last call to this
        method that returned true.

    Note: this method relies on data set by setLastUpdatedPos(), which must be called to set the previous data used
    in the next call to distanceCheck();
*/
bool NeighborList::distanceCheck()
    {
    const ParticleDataArraysConst& arrays = m_pdata->acquireReadOnly();
    // sanity check
    assert(arrays.x != NULL && arrays.y != NULL && arrays.z != NULL);
    
    // profile
    if (m_prof) m_prof->push("Dist check");
    
    // temporary storage for the result
    bool result = false;
    
    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getBox();
    // sanity check
    assert(box.xhi > box.xlo && box.yhi > box.ylo && box.zhi > box.zlo);
    
    // precalculate box lenghts
    Scalar Lx = box.xhi - box.xlo;
    Scalar Ly = box.yhi - box.ylo;
    Scalar Lz = box.zhi - box.zlo;
    
    // actually scan the array looking for values over 1/2 the buffer distance
    Scalar maxsq = (m_r_buff/Scalar(2.0))*(m_r_buff/Scalar(2.0));
    for (unsigned int i = 0; i < arrays.nparticles; i++)
        {
        Scalar dx = arrays.x[i] - m_last_x[i];
        Scalar dy = arrays.y[i] - m_last_y[i];
        Scalar dz = arrays.z[i] - m_last_z[i];
        
        // if the vector crosses the box, pull it back
        if (dx >= box.xhi)
            dx -= Lx;
        else if (dx < box.xlo)
            dx += Lx;
            
        if (dy >= box.yhi)
            dy -= Ly;
        else if (dy < box.ylo)
            dy += Ly;
            
        if (dz >= box.zhi)
            dz -= Lz;
        else if (dz < box.zlo)
            dz += Lz;
            
        if (dx*dx + dy*dy + dz*dz >= maxsq)
            {
            result = true;
            break;
            }
        }
        
    // don't worry about computing flops here, this is fast
    if (m_prof) m_prof->pop();
    
    m_pdata->release();
    return result;
    }

/*! Copies the current positions of all particles over to m_last_x etc...
*/
void NeighborList::setLastUpdatedPos()
    {
    const ParticleDataArraysConst& arrays = m_pdata->acquireReadOnly();
    // sanity check
    assert(arrays.x != NULL && arrays.y != NULL && arrays.z != NULL);
    
    // profile
    if (m_prof) m_prof->push("Dist check");
    
    // if we are updating, update the last position arrays
    memcpy((void *)m_last_x, arrays.x, sizeof(Scalar)*arrays.nparticles);
    memcpy((void *)m_last_y, arrays.y, sizeof(Scalar)*arrays.nparticles);
    memcpy((void *)m_last_z, arrays.z, sizeof(Scalar)*arrays.nparticles);
    
    if (m_prof) m_prof->pop();
    m_pdata->release();
    }

/*! \returns true If the neighbor list needs to be updated
    \returns false If the neighbor list does not need to be updated
    \note This is designed to be called if (needsUpdating()) then update every step.
        It internally handles many state variables that rely on this assumption.
    \param timestep Current time step in the simulation
*/
bool NeighborList::needsUpdating(unsigned int timestep)
    {
    // perform an early check: specifiying m_r_buff = 0.0 will result in the neighbor list being
    // updated every single time.
    if (m_r_buff < 1e-6)
        return true;
    if (timestep < (m_last_updated_tstep + m_every) && !m_force_update)
        return false;
        
    // check if this is a dangerous time
    // we are dangerous if m_every is greater than 1 and this is the first check after the
    // last build
    bool dangerous = false;
    if (m_every > 1 && timestep == (m_last_updated_tstep + m_every))
        dangerous = true;
        
    // temporary storage for return result
    bool result = false;
    
    // if the update has been forced, the result defaults to true
    if (m_force_update)
        {
        result = true;
        m_force_update = false;
        m_forced_updates += 1;
        m_last_updated_tstep = timestep;
        
        // when an update is forced, there is no way to tell if the build
        // is dangerous or not: filter out the false positive errors
        dangerous = false;
        }
    else
        {
        // not a forced update, perform the distance check to determine
        // if the list needs to be updated
        result = distanceCheck();
        
        if (result)
            {
            m_last_updated_tstep = timestep;
            m_updates += 1;
            }
        }
        
    // need to update teh last position so it is ready for the next call to this
    // method
    if (result)
        setLastUpdatedPos();
        
    // warn the user if this is a dangerous build
    if (result && dangerous)
        {
        cout << "***Warning! Dangerous neighborlist build occured. Continuing this simulation may produce incorrect results and/or program crashes. Decrease the neighborlist check_period and rerun." << endl;
        m_dangerous_updates += 1;
        }
        
    return result;
    }

/*! Generic statistics that apply to any neighbor list, like the number of updates,
    average number of neighbors, etc... are printed to stdout. Derived classes should
    print any pertinient information they see fit to.
 */
void NeighborList::printStats()
    {
    cout << "-- Neighborlist stats:" << endl;
    cout << m_updates << " normal updates / " << m_forced_updates << " forced updates / " << m_dangerous_updates << " dangerous updates" << endl;
    
    // copy the list back from the device if we need to
#ifdef ENABLE_CUDA
    if (m_data_location == gpu)
        {
        deviceToHostCopy();
        m_data_location = cpugpu;
        }
#endif
        
    // build some simple statistics of the number of neighbors
    unsigned int n_neigh_min = m_pdata->getN();
    unsigned int n_neigh_max = 0;
    Scalar n_neigh_avg = 0.0;
    
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        unsigned int n_neigh = (unsigned int)m_list[i].size();
        if (n_neigh < n_neigh_min)
            n_neigh_min = n_neigh;
        if (n_neigh > n_neigh_max)
            n_neigh_max = n_neigh;
            
        n_neigh_avg += Scalar(n_neigh);
        }
        
    // divide to get the average
    n_neigh_avg /= Scalar(m_pdata->getN());
    
    cout << "n_neigh_min: " << n_neigh_min << " / n_neigh_max: " << n_neigh_max << " / n_neigh_avg: " << n_neigh_avg << endl;
    
    // lets dump out the nlist to a file
    /*ofstream f("nlist.m");
    f << "nlist_data2 = [";
    for (int j = 0; j < n_neigh_max; j++)
        {
        for (unsigned int i = 0; i < m_pdata->getN(); i++)
            {
            if (j < m_list[i].size())
                f << m_list[i][j] << " ";
            else
                f << "-1 ";
            }
        f << ";" << endl;
        }
    
    f << "];" << endl;*/
    }

void NeighborList::resetStats()
    {
    m_updates = m_forced_updates = m_dangerous_updates = 0;
    }

/*! Loops through the particles and finds all of the particles \c j who's distance is less than
    \c r_cut \c + \c r_buff from particle \c i, includes either i < j or all neighbors depending
    on the mode set by setStorageMode()
*/
void NeighborList::buildNlist()
    {
    // sanity check
    assert(m_pdata);
    
    // start up the profile
    if (m_prof) m_prof->push("Build list");
    
    // access the particle data
    const ParticleDataArraysConst& arrays = m_pdata->acquireReadOnly();
    // sanity check
    assert(arrays.x != NULL && arrays.y != NULL && arrays.z != NULL);
    
    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getBox();
    // sanity check
    assert(box.xhi > box.xlo && box.yhi > box.ylo && box.zhi > box.zlo);
    
    if ((box.xhi - box.xlo) <= (m_r_cut+m_r_buff) * 2.0 || (box.yhi - box.ylo) <= (m_r_cut+m_r_buff) * 2.0 || (box.zhi - box.zlo) <= (m_r_cut+m_r_buff) * 2.0)
        {
        cerr << endl << "***Error! Simulation box is too small! Particles would be interacting with themselves." << endl << endl;
        throw runtime_error("Error updating neighborlist bins");
        }
        
    // simple algorithm follows:
    
    // start by creating a temporary copy of r_cut sqaured
    Scalar rmaxsq = (m_r_cut + m_r_buff) * (m_r_cut + m_r_buff);
    
    // precalculate box lenghts
    Scalar Lx = box.xhi - box.xlo;
    Scalar Ly = box.yhi - box.ylo;
    Scalar Lz = box.zhi - box.zlo;
    Scalar Lx2 = Lx / Scalar(2.0);
    Scalar Ly2 = Ly / Scalar(2.0);
    Scalar Lz2 = Lz / Scalar(2.0);
    
    
    // start by clearing the entire list
    for (unsigned int i = 0; i < arrays.nparticles; i++)
        m_list[i].clear();
        
    // now we can loop over all particles in n^2 fashion and build the list
    unsigned int n_neigh = 0;
    for (unsigned int i = 0; i < arrays.nparticles; i++)
        {
        Scalar xi = arrays.x[i];
        Scalar yi = arrays.y[i];
        Scalar zi = arrays.z[i];
        const ExcludeList &excludes = m_exclusions[arrays.tag[i]];
#if defined(LARGE_EXCLUSION_LIST)
        const ExcludeList &excludes2 = m_exclusions2[arrays.tag[i]];
        const ExcludeList &excludes3 = m_exclusions3[arrays.tag[i]];
        const ExcludeList &excludes4 = m_exclusions4[arrays.tag[i]];
#endif
        unsigned int bodyi = 0;
        if (m_exclude_same_body)
            bodyi = arrays.body[i];
        
        // for each other particle with i < j
        for (unsigned int j = i + 1; j < arrays.nparticles; j++)
            {
            // early out if these are excluded pairs
            if (excludes.e1 == arrays.tag[j] || excludes.e2 == arrays.tag[j] ||
                    excludes.e3 == arrays.tag[j] || excludes.e4 == arrays.tag[j])
                continue;
#if defined(LARGE_EXCLUSION_LIST)
            if (excludes2.e1 == arrays.tag[j] || excludes2.e2 == arrays.tag[j] ||
                    excludes2.e3 == arrays.tag[j] || excludes2.e4 == arrays.tag[j] ||
                    excludes3.e1 == arrays.tag[j] || excludes3.e2 == arrays.tag[j] ||
                    excludes3.e3 == arrays.tag[j] || excludes3.e4 == arrays.tag[j] ||
                    excludes4.e1 == arrays.tag[j] || excludes4.e2 == arrays.tag[j] ||
                    excludes4.e3 == arrays.tag[j] || excludes4.e4 == arrays.tag[j])
                continue;
#endif
            // test for same rigid body exclusion
            unsigned int bodyj = 0;
            if (m_exclude_same_body && bodyi != NO_BODY)
                {
                bodyj = arrays.body[j];
                if (bodyi == bodyj)
                    continue;
                }
            
            // calculate dr
            Scalar dx = arrays.x[j] - xi;
            Scalar dy = arrays.y[j] - yi;
            Scalar dz = arrays.z[j] - zi;
            
            // if the vector crosses the box, pull it back
            if (dx >= Lx2)
                dx -= Lx;
            else if (dx < -Lx2)
                dx += Lx;
                
            if (dy >= Ly2)
                dy -= Ly;
            else if (dy < -Ly2)
                dy += Ly;
                
            if (dz >= Lz2)
                dz -= Lz;
            else if (dz < -Lz2)
                dz += Lz;
                
            // sanity check
            assert(dx >= box.xlo && dx <= box.xhi);
            assert(dy >= box.ylo && dy <= box.yhi);
            assert(dz >= box.zlo && dz <= box.zhi);
            
            // now compare rsq to rmaxsq and add to the list if it meets the criteria
            Scalar rsq = dx*dx + dy*dy + dz*dz;
            if (rsq < rmaxsq)
                {
                if (m_storage_mode == full)
                    {
                    n_neigh += 2;
                    m_list[i].push_back(j);
                    m_list[j].push_back(i);
                    }
                else
                    {
                    n_neigh += 1;
                    m_list[i].push_back(j);
                    }
                }
            }
        }
        
    m_pdata->release();
    
#ifdef ENABLE_CUDA
    // after computing, the device now resides on the CPU
    m_data_location = cpu;
#endif
    
    // upate the profile
    // the number of evaluations made is the number of pairs which is = N(N-1)/2
    // each evalutation transfers 3*sizeof(Scalar) bytes for the particle access
    // and performs approximately 15 FLOPs
    // there are an additional N * 3 * sizeof(Scalar) accesses for the xj lookup
    uint64_t N = arrays.nparticles;
    if (m_prof) m_prof->pop(15*N*(N-1)/2, 3*sizeof(Scalar)*N*(N-1)/2 + N*3*sizeof(Scalar) + uint64_t(n_neigh)*sizeof(unsigned int));
    }


//! helper function for accessing an element of the neighbor list vector: python __getitem__
/*! \param nlist Neighbor list to extract a column from
    \param i item to extract
*/
const vector<unsigned int> & getNlistList(std::vector< std::vector<unsigned int> >* nlist, unsigned int i)
    {
    return (*nlist)[i];
    }

//! helper function for accessing an elemeng of the neighb rlist: python __getitem__
/*! \param list List to extract an item from
    \param i item to extract
*/
unsigned int getNlistItem(std::vector<unsigned int>* list, unsigned int i)
    {
    return (*list)[i];
    }

void export_NeighborList()
    {
    class_< std::vector< std::vector<unsigned int> > >("std_vector_std_vector_uint")
    .def("__len__", &std::vector< std::vector<unsigned int> >::size)
    .def("__getitem__", &getNlistList, return_internal_reference<>())
    ;
    
    class_< std::vector<unsigned int> >("std_vector_uint")
    .def("__len__", &std::vector<unsigned int>::size)
    .def("__getitem__", &getNlistItem)
    .def("push_back", &std::vector<unsigned int>::push_back)
    ;
    
    scope in_nlist = class_<NeighborList, boost::shared_ptr<NeighborList>, bases<Compute>, boost::noncopyable >
                     ("NeighborList", init< boost::shared_ptr<SystemDefinition>, Scalar, Scalar >())
                     .def("setRCut", &NeighborList::setRCut)
                     .def("setEvery", &NeighborList::setEvery)
                     .def("getList", &NeighborList::getList, return_internal_reference<>())
                     .def("setStorageMode", &NeighborList::setStorageMode)
                     .def("addExclusion", &NeighborList::addExclusion)
                     .def("clearExclusions", &NeighborList::clearExclusions)
                     .def("countExclusions", &NeighborList::countExclusions)
                     .def("addExclusionsFromBonds", &NeighborList::addExclusionsFromBonds)
                     .def("addExclusionsFromAngles", &NeighborList::addExclusionsFromAngles)
                     .def("addExclusionsFromDihedrals", &NeighborList::addExclusionsFromDihedrals)
                     .def("addOneThreeExclusionsFromTopology", &NeighborList::addOneThreeExclusionsFromTopology)
                     .def("addOneFourExclusionsFromTopology", &NeighborList::addOneFourExclusionsFromTopology)
                     .def("forceUpdate", &NeighborList::forceUpdate)
                     .def("estimateNNeigh", &NeighborList::estimateNNeigh)
                     .def("setExcludeSameBody", &NeighborList::setExcludeSameBody)
                     .def("isExcludeSameBody", &NeighborList::isExcludeSameBody)
                     ;
                     
    enum_<NeighborList::storageMode>("storageMode")
    .value("half", NeighborList::half)
    .value("full", NeighborList::full)
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

