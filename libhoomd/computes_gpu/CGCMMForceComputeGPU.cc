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
// Maintainer: akohlmey

/*! \file CGCMMForceComputeGPU.cc
    \brief Defines the CGCMMForceComputeGPU class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include "CGCMMForceComputeGPU.h"
#include "cuda_runtime.h"

#include <stdexcept>

#include <boost/python.hpp>
using namespace boost::python;

#include <boost/bind.hpp>

using namespace boost;
using namespace std;

#ifdef ENABLE_CUDA
#include "gpu_settings.h"
#endif

/*! \param sysdef System to compute forces on
    \param nlist Neighborlist to use for computing the forces
    \param r_cut Cuttoff radius beyond which the force is 0
    
    \post memory is allocated and all parameters ljX are set to 0.0
    
    \note The CGCMMForceComputeGPU does not own the Neighborlist, the caller should
    delete the neighborlist when done.
*/
CGCMMForceComputeGPU::CGCMMForceComputeGPU(boost::shared_ptr<SystemDefinition> sysdef,
                                           boost::shared_ptr<NeighborList> nlist,
                                           Scalar r_cut)
    : CGCMMForceCompute(sysdef, nlist, r_cut), m_block_size(64)
    {
    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (exec_conf.gpu.size() == 0)
        {
        cerr << endl << "***Error! Creating a CGCMMForceComputeGPU with no GPU in the execution configuration" << endl << endl;
        throw std::runtime_error("Error initializing CGCMMForceComputeGPU");
        }
        
    if (m_ntypes > 44)
        {
        cerr << endl << "***Error! CGCMMForceComputeGPU cannot handle " << m_ntypes << " types" << endl << endl;
        throw runtime_error("Error initializing CGCMMForceComputeGPU");
        }
        
    // ulf workaround setup
#ifndef DISABLE_ULF_WORKAROUND
    // the ULF workaround is needed on GTX280 and older GPUS
    // it is not needed on C1060, S1070, GTX285, GTX295, and (hopefully) newer ones
    m_ulf_workaround = true;
    
    cudaDeviceProp deviceProp;
    int dev;
    exec_conf.gpu[0]->call(bind(cudaGetDevice, &dev));
    exec_conf.gpu[0]->call(bind(cudaGetDeviceProperties, &deviceProp, dev));
    
    if (deviceProp.major >= 1 && deviceProp.minor >= 2)
        m_ulf_workaround = false;
    if (string(deviceProp.name) == "GeForce GTX 280")
        m_ulf_workaround = true;
        
    if (m_ulf_workaround)
        cout << "Notice: ULF bug workaround enabled for CGCMMForceComputeGPU" << endl;
#else
    m_ulf_workaround = false;
#endif
        
    // allocate the coeff data on the GPU
    int nbytes = sizeof(float4)*m_pdata->getNTypes()*m_pdata->getNTypes();
    
    d_coeffs.resize(exec_conf.gpu.size());
    for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
        {
        exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
        exec_conf.gpu[cur_gpu]->call(bind(cudaMallocHack, (void **)((void *)&d_coeffs[cur_gpu]), nbytes));
        assert(d_coeffs[cur_gpu]);
        exec_conf.gpu[cur_gpu]->call(bind(cudaMemset, (void *)d_coeffs[cur_gpu], 0, nbytes));
        }
    // allocate the coeff data on the CPU
    h_coeffs = new float4[m_pdata->getNTypes()*m_pdata->getNTypes()];
    }


CGCMMForceComputeGPU::~CGCMMForceComputeGPU()
    {
    // free the coefficients on the GPU
    exec_conf.tagAll(__FILE__, __LINE__);
    for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
        {
        assert(d_coeffs[cur_gpu]);
        exec_conf.gpu[cur_gpu]->call(bind(cudaFree, (void *)d_coeffs[cur_gpu]));
        }
    delete[] h_coeffs;
    }

/*! \param block_size Size of the block to run on the device
    Performance of the code may be dependant on the block size run
    on the GPU. \a block_size should be set to be a multiple of 32.
*/
void CGCMMForceComputeGPU::setBlockSize(int block_size)
    {
    m_block_size = block_size;
    }

/*! \post The parameters \a lj12 through \a lj4 are set for the pairs \a typ1, \a typ2 and \a typ2, \a typ1.
    \note \a lj? are low level parameters used in the calculation. In order to specify
    these for a 12-4 and 9-6 lennard jones formula (with alpha), they should be set to the following.

        12-4
    - \a lj12 = 2.598076 * epsilon * pow(sigma,12.0)
    - \a lj9 = 0.0
    - \a lj6 = 0.0
    - \a lj4 = -alpha * 2.598076 * epsilon * pow(sigma,4.0)

        9-6
    - \a lj12 = 0.0
    - \a lj9 = 6.75 * epsilon * pow(sigma,9.0);
    - \a lj6 = -alpha * 6.75 * epsilon * pow(sigma,6.0)
    - \a lj4 = 0.0

       12-6
    - \a lj12 = 4.0 * epsilon * pow(sigma,12.0)
    - \a lj9 = 0.0
    - \a lj6 = -alpha * 4.0 * epsilon * pow(sigma,4.0)
    - \a lj4 = 0.0

    Setting the parameters for typ1,typ2 automatically sets the same parameters for typ2,typ1: there
    is no need to call this funciton for symmetric pairs. Any pairs that this function is not called
    for will have lj12 through lj4 set to 0.0.

    \param typ1 Specifies one type of the pair
    \param typ2 Specifies the second type of the pair
    \param lj12 1/r^12 term
    \param lj9  1/r^9 term
    \param lj6  1/r^6 term
    \param lj4  1/r^4 term
*/
void CGCMMForceComputeGPU::setParams(unsigned int typ1, unsigned int typ2, Scalar lj12, Scalar lj9, Scalar lj6, Scalar lj4)
    {
    assert(h_coeffs);
    if (typ1 >= m_ntypes || typ2 >= m_ntypes)
        {
        cerr << endl << "***Error! Trying to set CGCMM params for a non existant type! " << typ1 << "," << typ2 << endl << endl;
        throw runtime_error("CGCMMForceComputeGpu::setParams argument error");
        }
        
    // set coeffs in both symmetric positions in the matrix
    h_coeffs[typ1*m_pdata->getNTypes() + typ2] = make_float4(lj12, lj9, lj6, lj4);
    h_coeffs[typ2*m_pdata->getNTypes() + typ1] = make_float4(lj12, lj9, lj6, lj4);
    
    int nbytes = sizeof(float4)*m_pdata->getNTypes()*m_pdata->getNTypes();
    exec_conf.tagAll(__FILE__, __LINE__);
    for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
        exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, d_coeffs[cur_gpu], h_coeffs, nbytes, cudaMemcpyHostToDevice));
    }

/*! \post The CGCMM forces are computed for the given timestep on the GPU.
    The neighborlist's compute method is called to ensure that it is up to date
    before forces are computed.
    \param timestep Current time step of the simulation

    Calls gpu_compute_cgcmm_forces to do the dirty work.
*/
void CGCMMForceComputeGPU::computeForces(unsigned int timestep)
    {
    // start by updating the neighborlist
    m_nlist->compute(timestep);
    
    // start the profile
    if (m_prof) m_prof->push(exec_conf, "CGCMM pair");
    
    // The GPU implementation CANNOT handle a half neighborlist, error out now
    bool third_law = m_nlist->getStorageMode() == NeighborList::half;
    if (third_law)
        {
        cerr << endl << "***Error! CGCMMForceComputeGPU cannot handle a half neighborlist" << endl << endl;
        throw runtime_error("Error computing forces in CGCMMForceComputeGPU");
        }
        
    // access the neighbor list, which just selects the neighborlist into the device's memory, copying
    // it there if needed
    vector<gpu_nlist_array>& nlist = m_nlist->getListGPU();
    
    // access the particle data
    vector<gpu_pdata_arrays>& pdata = m_pdata->acquireReadOnlyGPU();
    gpu_boxsize box = m_pdata->getBoxGPU();
    
    // run the kernel on all GPUs in parallel
    exec_conf.tagAll(__FILE__, __LINE__);
    for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
        exec_conf.gpu[cur_gpu]->callAsync(bind(gpu_compute_cgcmm_forces, m_gpu_forces[cur_gpu].d_data, pdata[cur_gpu], box, nlist[cur_gpu], d_coeffs[cur_gpu], m_pdata->getNTypes(), m_r_cut * m_r_cut, m_block_size, m_ulf_workaround));
    exec_conf.syncAll();
    
    m_pdata->release();
    
    // the force data is now only up to date on the gpu
    m_data_location = gpu;
    
    Scalar avg_neigh = m_nlist->estimateNNeigh();
    int64_t n_calc = int64_t(avg_neigh * m_pdata->getN());
    int64_t mem_transfer = m_pdata->getN() * (4 + 16 + 20) + n_calc * (4 + 16);
    int64_t flops = n_calc * (3+12+5+2+3+11+3+8+7);
    if (m_prof) m_prof->pop(exec_conf, flops, mem_transfer);
    }

void export_CGCMMForceComputeGPU()
    {
    class_<CGCMMForceComputeGPU, boost::shared_ptr<CGCMMForceComputeGPU>, bases<CGCMMForceCompute>, boost::noncopyable >
    ("CGCMMForceComputeGPU", init< boost::shared_ptr<SystemDefinition>, boost::shared_ptr<NeighborList>, Scalar >())
    .def("setBlockSize", &CGCMMForceComputeGPU::setBlockSize)
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

