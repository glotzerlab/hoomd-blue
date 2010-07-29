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

/*! \file BinnedNeighborListGPU.cc
    \brief Defines the BinnedNeighborListGPU class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include "BinnedNeighborListGPU.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <math.h>

#include <boost/python.hpp>
using namespace boost::python;

#include <boost/bind.hpp>

using namespace boost;
using namespace std;

#ifdef ENABLE_CUDA
#include "gpu_settings.h"
#endif

/*! \param sysdef System the neighborlist is to compute neighbors for
    \param r_cut Cuttoff radius under which particles are considered neighbors
    \param r_buff Buffer distance to include around the cutoff
    
    \post The neighbor list is initialized and the list memory has been allocated,
    but the list will not be computed until compute is called.
    
    \post The storage mode defaults to full
    
    \sa NeighborList
*/
BinnedNeighborListGPU::BinnedNeighborListGPU(boost::shared_ptr<SystemDefinition> sysdef, Scalar r_cut, Scalar r_buff) 
    : NeighborList(sysdef, r_cut, r_buff), m_block_size(64)
    {
    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!exec_conf->isCUDAEnabled())
        {
        cerr << endl << "***Error! Creating a BinnedNeighborListGPU with no GPU in the execution configuration" << endl << endl;
        throw std::runtime_error("Error initializing BinnedNeighborListGPU");
        }

    m_storage_mode = full;
    // this is a bit of a trick, but initialize the last allocated Mx,My,Mz values to bogus settings so that
    // updatBins is sure to reallocate them
    // BUT, updateBins is going to free the arrays first, so allocate some dummy arrays
    allocateGPUBinData(1,1,1,1);
    
    // set a small default Nmax. This will expand as needed.
    m_Nmax = 1;
    m_curNmax = 0;
    m_avgNmax = Scalar(0.0);
    
    // bogus values for last value
    m_last_Mx = INT_MAX;
    m_last_My = INT_MAX;
    m_last_Mz = INT_MAX;
    m_Mx = 0;
    m_My = 0;
    m_Mz = 0;
    
    // allocate the array of bin ids
    GPUArray< unsigned int > bin_ids(m_pdata->getN(), exec_conf);
    m_bin_ids.swap(bin_ids);
    
    // allocate the thread mapping array
    GPUArray< unsigned int > thread_mapping(m_pdata->getN(), exec_conf);
    m_thread_mapping.swap(thread_mapping);
    }

BinnedNeighborListGPU::~BinnedNeighborListGPU()
    {
    freeGPUBinData();
    }

/*! \param Mx Number of bins in the x direction
    \param My Number of bins in the y direction
    \param Mz Number of bins in the z direction
    \param Nmax Maximum number of particles stored in any given bin

    \pre Bins are not allocated
    \post All memory needed for the bin data structure on the GPU is allocated along with the host mirror versions
*/
void BinnedNeighborListGPU::allocateGPUBinData(unsigned int Mx, unsigned int My, unsigned int Mz, unsigned int Nmax)
    {
    assert(exec_conf->isCUDAEnabled());
    
    // use mallocPitch to make sure that memory accesses are coalesced
    size_t pitch;
    
    // allocate and zero device memory
    if (Mx*My*Mz*Nmax >= 500000*128)
        cout << "***Warning! Allocating abnormally large cell list: " << Mx << " " << My << " " << Mz << " " << Nmax << endl;
        
    // setup the dimensions
    m_gpu_bin_data.Mx = Mx;
    m_gpu_bin_data.My = My;
    m_gpu_bin_data.Mz = Mz;

    cudaMallocPitch(&m_gpu_bin_data.idxlist, &pitch, Nmax*sizeof(unsigned int), Mx*My*Mz);
    // want pitch in elements, not bytes
    Nmax = (int)pitch / sizeof(unsigned int);
    cudaMemset(m_gpu_bin_data.idxlist, 0, pitch * Mx*My*Mz);
    
    cudaMalloc(&m_gpu_bin_data.bin_size, Mx*My*Mz*sizeof(unsigned int));
    
    cudaChannelFormatDesc idxlist_desc = cudaCreateChannelDesc< unsigned int >();
    cudaMallocArray(&m_gpu_bin_data.idxlist_array, &idxlist_desc, Nmax, Mx*My*Mz);
    
    // allocate the bin adjacent list array
    cudaChannelFormatDesc bin_adj_desc = cudaCreateChannelDesc< int >();
    cudaMallocArray(&m_gpu_bin_data.bin_adj_array, &bin_adj_desc, Mx*My*Mz, 27);
    CHECK_CUDA_ERROR();
    
    // allocate and zero host memory
    cudaHostAlloc(&m_host_idxlist, pitch * Mx*My*Mz, cudaHostAllocPortable);
    memset((void*)m_host_idxlist, 0, pitch*Mx*My*Mz);
    
    // allocate the host bin adj array
    int *bin_adj_host = new int[Mx*My*Mz*27];
    
    // initialize the coord and bin adj arrays
    for (int i = 0; i < (int)Mx; i++)
        {
        for (int j = 0; j < (int)My; j++)
            {
            for (int k = 0; k < (int)Mz; k++)
                {
                int bin = i*Mz*My + j*Mz + k;
                
                // loop over neighboring bins
                int cur_adj = 0;
                for (int neigh_i = i-1; neigh_i <= i+1; neigh_i++)
                    {
                    for (int neigh_j = j-1; neigh_j <= j+1; neigh_j++)
                        {
                        for (int neigh_k = k-1; neigh_k <= k+1; neigh_k++)
                            {
                            int a = neigh_i;
                            if (a < 0) a+= Mx;
                            if (a >= (int)Mx) a-= Mx;
                            
                            int b = neigh_j;
                            if (b < 0) b+= My;
                            if (b >= (int)My) b-= My;
                            
                            int c = neigh_k;
                            if (c < 0) c+= Mz;
                            if (c >= (int)Mz) c-= Mz;
                            
                            int neigh_bin = a*Mz*My + b*Mz + c;
                            bin_adj_host[bin + cur_adj*Mx*My*Mz] = neigh_bin;
                            cur_adj++;
                            }
                        }
                    }
                }
            }
        }
    // sort to improve memory access pattern
    unsigned int nbins = Mx*My*Mz;
    for (unsigned int cur_bin = 0; cur_bin < nbins; cur_bin++)
        {
        bool swapped = false;
        do
            {
            swapped = false;
            for (unsigned int i = 0; i < 27-1; i++)
                {
                if (bin_adj_host[nbins*i + cur_bin] > bin_adj_host[nbins*(i+1) + cur_bin])
                    {
                    unsigned int tmp = bin_adj_host[nbins*(i+1) + cur_bin];
                    bin_adj_host[nbins*(i+1) + cur_bin] = bin_adj_host[nbins*i + cur_bin];
                    bin_adj_host[nbins*i + cur_bin] = tmp;
                    swapped = true;
                    }
                }
            }
        while (swapped);
        }
        
    // copy it to the device. This only needs to be done once
    cudaMemcpyToArray(m_gpu_bin_data.bin_adj_array, 0, 0, bin_adj_host, 27*Mx*My*Mz*sizeof(int), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR();
        
    // don't need the temporary bin adj data any more
    delete[] bin_adj_host;
    
    size_t pitch_coord;
    cudaMallocPitch(&m_gpu_bin_data.coord_idxlist, &pitch_coord, Mx*My*Mz*sizeof(float4), Nmax);
    // want width in elements, not bytes
    m_gpu_bin_data.coord_idxlist_width = (int)pitch_coord / sizeof(float4);
    
    cudaMemset(m_gpu_bin_data.coord_idxlist, 0, pitch_coord * Nmax);
    
    cudaChannelFormatDesc coord_idxlist_desc = cudaCreateChannelDesc< float4 >();
    cudaMallocArray(&m_gpu_bin_data.coord_idxlist_array, &coord_idxlist_desc, m_gpu_bin_data.coord_idxlist_width, Nmax);
    CHECK_CUDA_ERROR();
        
    // assign allocated pitch
    m_gpu_bin_data.Nmax = Nmax;
    }

/*! \pre allocateGPUBinData() has been called
    \post Bin data is freed
*/
void BinnedNeighborListGPU::freeGPUBinData()
    {
    assert(exec_conf->isCUDAEnabled());
    
    // free the device memory
    cudaFree(m_gpu_bin_data.idxlist);
    cudaFreeArray(m_gpu_bin_data.idxlist_array);
    cudaFree(m_gpu_bin_data.coord_idxlist);
    cudaFree(m_gpu_bin_data.bin_size);
    cudaFreeArray(m_gpu_bin_data.coord_idxlist_array);
    cudaFreeArray(m_gpu_bin_data.bin_adj_array);
    CHECK_CUDA_ERROR();
    
    // set pointers to NULL so no one will think they are valid
    m_gpu_bin_data.idxlist = NULL;
        
    // free the hsot memory
    cudaFreeHost(m_host_idxlist);
    m_host_idxlist = NULL;
    }

/*! Makes all the calls needed to bring the neighbor list up to date on the GPU.
    This requires building the cell list, copying it to the GPU and then
    attempting to build the list repeatedly, increasing the allocated
    memory each time until the list does not overflow.
*/
void BinnedNeighborListGPU::buildNlist()
    {
    assert(exec_conf->isCUDAEnabled());
    
    // bin the particles
    updateBinsUnsorted();
    
    // copy those bins to the GPU
    if (m_prof) m_prof->push(exec_conf, "Bin copy");
    
    unsigned int nbytes = m_gpu_bin_data.Mx * m_gpu_bin_data.My * m_gpu_bin_data.Mz * m_gpu_bin_data.Nmax * sizeof(unsigned int);
    
    cudaMemcpyToArray(m_gpu_bin_data.idxlist_array, 0, 0, m_host_idxlist, nbytes, cudaMemcpyHostToDevice);
    cudaMemcpy(m_gpu_bin_data.bin_size, &m_bin_sizes, sizeof(unsigned int)*m_gpu_bin_data.Mx*m_gpu_bin_data.My*m_gpu_bin_data.Mz, cudaMemcpyHostToDevice);
    
    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    
    if (m_prof) m_prof->pop(exec_conf, 0, nbytes);
    // transpose the bins for a better memory access pattern on the GPU
    
    if (m_prof) m_prof->push(exec_conf, "Transpose");
    gpu_pdata_arrays& pdata = m_pdata->acquireReadOnlyGPU();
    gpu_nlist_idxlist2coord(&pdata, &m_gpu_bin_data, m_curNmax, 256);
    cudaMemcpyToArray(m_gpu_bin_data.coord_idxlist_array, 0, 0, m_gpu_bin_data.coord_idxlist, m_gpu_bin_data.coord_idxlist_width*m_curNmax*sizeof(float4), cudaMemcpyDeviceToDevice);

    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    
    m_pdata->release();
    
    if (m_prof) m_prof->pop(exec_conf, 0);
    
    // update the neighbor list using the bins. Need to check for overflows
    // and increase the size of the list as needed
    updateListFromBins();
    
    int overflow = 0;
    cudaMemcpy(&overflow, m_gpu_nlist.overflow, sizeof(int), cudaMemcpyDeviceToHost);
        
    while (overflow)
        {
        int new_height = (int)(Scalar(m_gpu_nlist.height) * 1.2);
        // cout << "Notice: Neighborlist overflowed on GPU, expanding to " << new_height << " neighbors per particle..." << endl;
        freeGPUData();
        allocateGPUData(new_height);
        updateExclusionData();
        
        updateListFromBins();
        cudaMemcpy(&overflow, m_gpu_nlist.overflow, sizeof(int), cudaMemcpyDeviceToHost);
        }
    
    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    
    m_data_location = gpu;
    }

/*! Particles are copied to the CPU and the binning step is done, putting the index of each particle
    into each bin. This is all done in the host memory copies of the bin arrays, they are not
    copied to the GPU here.

    \pre GPU Bin data has been allocated.
*/
void BinnedNeighborListGPU::updateBinsUnsorted()
    {
    assert(m_pdata);
    
    // start up the profile
    if (m_prof) m_prof->push(exec_conf, "Bin");
    
    // calculate the bin dimensions
    const BoxDim& box = m_pdata->getBox();
    assert(box.xhi > box.xlo && box.yhi > box.ylo && box.zhi > box.zlo);
    m_Mx = int((box.xhi - box.xlo) / (m_r_cut + m_r_buff));
    m_My = int((box.yhi - box.ylo) / (m_r_cut + m_r_buff));
    m_Mz = int((box.zhi - box.zlo) / (m_r_cut + m_r_buff));

    if (m_sysdef->getNDimensions() == 2) 
        m_Mz = 3;

    if (m_Mx < 3 || m_My < 3 || m_Mz < 3)
        {
        cerr << endl << "***Error! BinnedNeighborListGPU doesn't work on boxes where r_cut+r_buff is greater than 1/3 any box dimension" << endl << endl;
        throw runtime_error("Error updating neighborlist bins");
        }
        
    // TODO: this should really be determined as a minimum of the memcpy pitch and the 2D texture dimensions
    // decrease the number of bins if it exceeds 16384
    if (m_Mx * m_My *m_Mz > 16384)
        {
        float scale_factor = powf(16384.0f / float(m_Mx * m_My *m_Mz), 1.0f/3.0f);
        m_Mx = int(float(m_Mx)*scale_factor);
        m_My = int(float(m_My)*scale_factor);
        m_Mz = int(float(m_Mz)*scale_factor);
        }
        
    // if these dimensions are different than the previous dimensions, reallocate
    if (m_Mx != m_last_Mx || m_My != m_last_My || m_Mz != m_last_Mz)
        {
        freeGPUBinData();
        allocateGPUBinData(m_Mx, m_My, m_Mz, m_Nmax);
        // reassign m_Nmax since it may have been expanded by the allocation process
        m_Nmax = m_gpu_bin_data.Nmax;
        
        m_last_Mx = m_Mx;
        m_last_My = m_My;
        m_last_Mz = m_Mz;
        m_bin_sizes.resize(m_Mx*m_My*m_Mz);
        }
        
    // make even bin dimensions
    Scalar binx = (box.xhi - box.xlo) / Scalar(m_Mx);
    Scalar biny = (box.yhi - box.ylo) / Scalar(m_My);
    Scalar binz = (box.zhi - box.zlo) / Scalar(m_Mz);
    
    // precompute scale factors to eliminate division in inner loop
    Scalar scalex = Scalar(1.0) / binx;
    Scalar scaley = Scalar(1.0) / biny;
    Scalar scalez = Scalar(1.0) / binz;
    
    // use the GPU to determine which bin in which each particle resides
    // acquire the particle data
        {
        gpu_pdata_arrays& pdata = m_pdata->acquireReadOnlyGPU();
        ArrayHandle<unsigned int> d_bin_ids(m_bin_ids, access_location::device, access_mode::overwrite);
        
        // call the kernel to compute the bin ids
        gpu_compute_bin_ids(d_bin_ids.data,
                            pdata,
                            m_pdata->getBoxGPU(),
                            m_Mx,
                            m_My,
                            m_Mz,
                            scalex,
                            scaley,
                            scalez);
        
        if (exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        
        m_pdata->release();
        }
    
    // now prep to use those bin ids on the host to actually bin the particles
    // setup the memory arrays
    for (unsigned int i = 0; i < m_Mx*m_My*m_Mz; i++)
        m_bin_sizes[i] = 0;
        
    // clear the bins to 0xffffffff which means no particle in that bin
    memset((void*)m_host_idxlist, 0xff, sizeof(unsigned int)*m_Mx*m_My*m_Mz*m_Nmax);
    
    // reset the counter that keeps track of the current size of the largest bin
    m_curNmax = 0;
    
    // get the bin ids from the gpu
    // for each particle
    bool overflow = false;
    unsigned int overflow_value = 0;
    
    { // (scope h_bin_ids access so that it isn't already acquired in the recursive call)
    ArrayHandle<unsigned int> h_bin_ids(m_bin_ids, access_location::host, access_mode::read);
    
    for (unsigned int n = 0; n < m_pdata->getN(); n++)
        {
        unsigned int bin = h_bin_ids.data[n];
        if (bin == 0xffffffff)
            {
            cerr << endl << "***Error! Particle " << n << "'s coordinates are no longer finite" << endl << endl;
            throw runtime_error("Error updating neighbor list bins");
            }
        
        assert(bin < m_Mx*m_My*m_Mz);
        unsigned int size = m_bin_sizes[bin];
        
        // track the size of the largest bin
        if (size+1 > m_curNmax)
            m_curNmax = size+1;
            
        // make sure we don't overflow
        if (size < m_Nmax)
            {
            m_host_idxlist[size + bin*m_Nmax] = n;
            }
        else
            {
            overflow = true;
            if (size > overflow_value)
                overflow_value = size;
            }
        m_bin_sizes[bin]++;
        }
        
    m_avgNmax = Scalar(0.0);
    for (unsigned int i = 0; i < m_Mx * m_My * m_Mz; i++)
        {
        m_avgNmax += m_bin_sizes[i];
        }
    m_avgNmax /= Scalar(m_Mx * m_My * m_Mz);
    } // (end of h_bin_ids scope)

    // update profile
    if (m_prof) m_prof->pop(exec_conf);
    
    // we aren't done yet, if there was an overflow, update m_Nmax and recurse to make sure the list is fully up to date
    // since we are now certain that m_Nmax will hold all of the particles, the recursion should only happen once
    if (overflow)
        {
        // reallocate memory first, so there is room
        m_Nmax = overflow_value+1;
        freeGPUBinData();
        allocateGPUBinData(m_Mx, m_My, m_Mz, m_Nmax);
        // reassign m_Nmax since it may have been expanded by the allocation process
        m_Nmax = m_gpu_bin_data.Nmax;
        updateBinsUnsorted();
        }
    // note, we don't copy the binned values to the device yet, that is for the compute to do
    else
        {
        // assign particles to threads
        ArrayHandle<unsigned int> h_thread_mapping(m_thread_mapping, access_location::host, access_mode::overwrite);
        unsigned int current = 0;
       for (unsigned int bin = 0; bin < m_Mx * m_My * m_Mz; bin++)
           {
           unsigned int bin_size = m_bin_sizes[bin];
           for (unsigned int slot = 0; slot < bin_size; slot++)
               {
               h_thread_mapping.data[current] = m_host_idxlist[slot + bin*m_Nmax];
               current++;
               }
            }
        }
    }

/*! Sets up and executes the needed kernel calls to generate the actual neighbor list
    from the previously binned data.

    \pre updateBinsUnsorted() has been called.
    \pre The bin data has been copied to the GPU and transposed into the idxlist_coord array

    Calls gpu_compute_nlist_binned to do the dirty work.
*/
void BinnedNeighborListGPU::updateListFromBins()
    {
    assert(exec_conf->isCUDAEnabled());
    
    if (m_storage_mode != full)
        {
        cerr << endl << "***Error! Only full mode nlists can be generated on the GPU" << endl << endl;
        throw runtime_error("Error computing neighbor list");
        }
        
    // sanity check
    assert(m_pdata);
    
    // start up the profile
    if (m_prof) m_prof->push(exec_conf, "Build list");
    
    // access the particle data
    gpu_pdata_arrays& pdata = m_pdata->acquireReadOnlyGPU();
    gpu_boxsize box = m_pdata->getBoxGPU();
    
    Scalar r_max_sq = (m_r_cut + m_r_buff) * (m_r_cut + m_r_buff);
    ArrayHandle<unsigned int> d_bin_ids(m_bin_ids, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_thread_mapping(m_thread_mapping, access_location::device, access_mode::read);

    m_gpu_nlist.thread_mapping = d_thread_mapping.data;
    gpu_compute_nlist_binned(m_gpu_nlist, pdata, box, m_gpu_bin_data, d_bin_ids.data, r_max_sq, m_curNmax, m_block_size);
        
    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    
    m_pdata->release();
    
    int64_t flops = int64_t(m_pdata->getN() * (9 + 27 * m_avgNmax * (15 + 5 + 1)));
    int64_t mem_transfer = int64_t(m_pdata->getN() * (32 + 4 + 8 + 27 * (4 + m_avgNmax * 16) + estimateNNeigh() * 4));
    if (m_prof) m_prof->pop(exec_conf, flops, mem_transfer);
    }

/*! \returns true If any of the particles have been moved more than 1/2 of the buffer distance since the last call
        to this method that returned true.
    \returns false If none of the particles has been moved more than 1/2 of the buffer distance since the last call to this
        method that returned true.
*/
bool BinnedNeighborListGPU::distanceCheck()
    {
    assert(exec_conf->isCUDAEnabled());
    
    // scan through the particle data arrays and calculate distances
    if (m_prof) m_prof->push(exec_conf, "Dist check");
    
    gpu_pdata_arrays& pdata = m_pdata->acquireReadOnlyGPU();
    gpu_boxsize box = m_pdata->getBoxGPU();
    
    // create a temporary copy of r_max sqaured
    Scalar r_buffsq = (m_r_buff/Scalar(2.0)) * (m_r_buff/Scalar(2.0));
    
    int result = 0;
    gpu_nlist_needs_update_check(&pdata, &box, &m_gpu_nlist, r_buffsq, &result);

    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
        
    m_pdata->release();
    
    if (m_prof) m_prof->pop(exec_conf);
    
    return result;
    }

/*! Does nothing in BinnedNeighborListGPU. The last position values are set whenever the neighbor list is built.
*/
void BinnedNeighborListGPU::setLastUpdatedPos()
    {
    }

/*! This method just adds a few statistic numbers specific to this class. The base class method is called
    first to get the more general stats from there.
*/
void BinnedNeighborListGPU::printStats()
    {
    NeighborList::printStats();
    
    cout << "Nmax = " << m_Nmax << " / curNmax = " << m_curNmax << endl;
    int Nbins = m_gpu_bin_data.Mx * m_gpu_bin_data.My * m_gpu_bin_data.Mz;
    cout << "bins Nmax = " << m_gpu_bin_data.Nmax << " / Nbins = " << Nbins << endl;
    }

void export_BinnedNeighborListGPU()
    {
    class_<BinnedNeighborListGPU, boost::shared_ptr<BinnedNeighborListGPU>, bases<NeighborList>, boost::noncopyable >
    ("BinnedNeighborListGPU", init< boost::shared_ptr<SystemDefinition>, Scalar, Scalar >())
    .def("setBlockSize", &BinnedNeighborListGPU::setBlockSize)
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

