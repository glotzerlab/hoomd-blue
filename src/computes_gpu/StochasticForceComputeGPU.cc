/*
Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
Source Software License
Copyright (c) 2008 Ames Laboratory Iowa State University
All rights reserved.

Redistribution and use of HOOMD, in source and binary forms, with or
without modification, are permitted, provided that the following
conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names HOOMD's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id: StochasticForceComputeGPU.cc 1234 2008-09-11 16:29:13Z joaander $
// $URL: http://svn2.assembla.com/svn/hoomd/trunk/src/computes_gpu/StochasticForceComputeGPU.cc $
// Maintainer: phillicl

/*! \file StochasticForceComputeGPU.cc
    \brief Defines the StochasticForceComputeGPU class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include "StochasticForceComputeGPU.h"
#include "cuda_runtime.h"

#include <stdexcept>
#include <stdlib.h>
#include <math.h>

#include <boost/python.hpp>
using namespace boost::python;

#include <boost/bind.hpp>

using namespace boost;
using namespace std;

#ifdef ENABLE_CUDA
#include "gpu_settings.h"
#endif

/*! \param sysdef System to compute forces on
    \param Temp Temperature of the bath of random particles
    \param deltaT Length of the computation timestep
    \param seed Seed for initializing the RNG
*/
StochasticForceComputeGPU::StochasticForceComputeGPU(boost::shared_ptr<SystemDefinition> sysdef,
                                                     Scalar deltaT,
                                                     boost::shared_ptr<Variant> Temp,
                                                     unsigned int seed,
                                                     bool use_diam)
    : StochasticForceCompute(sysdef, deltaT, Temp, seed, use_diam)
    {
    // default block size is the highest performance in testing on different hardware
    // choose based on compute capability of the device
    cudaDeviceProp deviceProp;
    int dev;
    exec_conf.gpu[0]->call(bind(cudaGetDevice, &dev));
    exec_conf.gpu[0]->call(bind(cudaGetDeviceProperties, &deviceProp, dev));
    // catch Tesla C1060 first, as it requires a different tuning then the rest
    if (string(deviceProp.name) == string("Tesla C1060"))
        m_block_size = 128; // note: I don't know what the proper setting is, JA
    else if (deviceProp.major == 1 && deviceProp.minor == 0)
        m_block_size = 64;
    else if (deviceProp.major == 1 && deviceProp.minor == 1)
        m_block_size = 64;
    else if (deviceProp.major == 1 && deviceProp.minor < 4)
        m_block_size = 128;
    else
        {
        cout << "***Warning! Unknown compute " << deviceProp.major << "." << deviceProp.minor << " when tuning block size for StohasticForceComputeGPU" << endl;
        m_block_size = 64;
        }
        
    // allocate the gamma data on the GPU
    int nbytes = sizeof(float)*m_pdata->getNTypes();
    
    if (!m_use_diam)
        {
        // allocate the coeff data on the CPU
        h_gammas = new float[m_pdata->getNTypes()];
        //All gamma coefficients initialized to 1.0
        for (unsigned int j = 0; j < m_pdata->getNTypes(); j++) h_gammas[j] = 1.0;
        d_gammas.resize(exec_conf.gpu.size());
        
        exec_conf.tagAll(__FILE__, __LINE__);
        for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
            {
            exec_conf.gpu[cur_gpu]->call(bind(cudaMallocHack, (void **)((void *)&d_gammas[cur_gpu]), nbytes));
            assert(d_gammas[cur_gpu]);
            exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, (void **)(void *) d_gammas[cur_gpu], h_gammas, nbytes, cudaMemcpyHostToDevice));
            }
        }
    }


StochasticForceComputeGPU::~StochasticForceComputeGPU()
    {
    // deallocate our memory
    if (!m_use_diam)
        {
        exec_conf.tagAll(__FILE__, __LINE__);
        for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
            {
            assert(d_gammas[cur_gpu]);
            exec_conf.gpu[cur_gpu]->call(bind(cudaFree, (void *)d_gammas[cur_gpu]));
            }
        delete[] h_gammas;
        }
    }

/*! \param block_size Size of the block to run on the device
    Performance of the code may be dependant on the block size run
    on the GPU. \a block_size should be set to be a multiple of 32.
    \todo error check value
*/
void StochasticForceComputeGPU::setBlockSize(int block_size)
    {
    m_block_size = block_size;
    }

/*! \post The parameter \a gamma is set for \a typ,
    \note \a gamma is a low level parameters used in the calculation.

    \param typ Specifies the particle type
    \param gamma Parameter used to calcluate forces
*/
void StochasticForceComputeGPU::setParams(unsigned int typ, Scalar gamma)
    {
    if (!m_use_diam)
        {
        assert(h_gammas);
        if (typ >= m_ntypes)
            {
            cerr << endl << "***Error! Trying to set Stochastic Force param Gamma for a non existant type! " << typ << endl << endl;
            throw runtime_error("StochasticForceComputeGpu::setParams argument error");
            }
            
        // set gamma coeffs
        h_gammas[typ] = gamma;
        
        int nbytes = sizeof(float)*m_pdata->getNTypes();
        
        exec_conf.tagAll(__FILE__, __LINE__);
        for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
            exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, d_gammas[cur_gpu], h_gammas, nbytes, cudaMemcpyHostToDevice));
        }
    else cerr << endl << "***Error! Trying to set Stochastic Force param Gamma while using Diameter as Gamma!" << endl << endl;
    }

/*! \post The stochastic forces are computed for the given timestep on the GPU.
    \param timestep Current time step of the simulation

    Calls gpu_compute_stochastic_forces to do the dirty work.
*/
void StochasticForceComputeGPU::computeForces(unsigned int timestep)
    {
    // start the profile
    if (m_prof) m_prof->push(exec_conf, "Stochastic Baths");
    
    // access the particle data
    vector<gpu_pdata_arrays>& pdata = m_pdata->acquireReadOnlyGPU();
    
    exec_conf.tagAll(__FILE__, __LINE__);
    
    if (!m_use_diam)
        {
        // call the kernel on all GPUs in parallel
        for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
            exec_conf.gpu[cur_gpu]->callAsync(bind(gpu_compute_stochastic_forces, m_gpu_forces[cur_gpu].d_data, pdata[cur_gpu], m_dt, m_T->getValue(timestep), d_gammas[cur_gpu], m_seed, timestep, m_pdata->getNTypes(), m_block_size));
        }
    if (m_use_diam)
        {
        // call the kernel on all GPUs in parallel
        for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
            exec_conf.gpu[cur_gpu]->callAsync(bind(gpu_compute_stochastic_forces_diam, m_gpu_forces[cur_gpu].d_data, pdata[cur_gpu], m_dt, m_T->getValue(timestep), m_seed, timestep, m_block_size));
        }
        
    exec_conf.syncAll();
    
    m_pdata->release();
    
    // the force data is now only up to date on the gpu
    m_data_location = gpu;
    
//  int64_t mem_transfer = m_pdata->getN() * (4 + 16 + 16) + n_calc * (4 + 16);
//  int64_t flops = n_calc * (3+12+5+2+2+6+3+7);
    // if (m_prof) m_prof->pop(exec_conf, flops, mem_transfer);
    
    // I'm not sure why the above is commented out, but we cannot have a push (above) without a pop!
    if (m_prof) m_prof->pop();
    }

#ifdef WIN32
#pragma warning( pop )
#endif

