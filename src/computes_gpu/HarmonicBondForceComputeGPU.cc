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

// $Id$
// $URL$
// Maintainer: joaander

/*! \file HarmonicBondForceComputeGPU.cc
    \brief Defines the HarmonicBondForceComputeGPU class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

#include "HarmonicBondForceComputeGPU.h"

#include <boost/python.hpp>
using namespace boost::python;

#include <boost/bind.hpp>
using namespace boost;

using namespace std;

#ifdef ENABLE_CUDA
#include "gpu_settings.h"
#endif

/*! \param sysdef System to compute bond forces on
*/
HarmonicBondForceComputeGPU::HarmonicBondForceComputeGPU(boost::shared_ptr<SystemDefinition> sysdef)
        : HarmonicBondForceCompute(sysdef), m_block_size(64)
    {
    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (exec_conf.gpu.size() == 0)
        {
        cerr << endl << "***Error! Creating a BondForceComputeGPU with no GPU in the execution configuration" << endl << endl;
        throw std::runtime_error("Error initializing BondForceComputeGPU");
        }
        
    // allocate and zero device memory
    m_gpu_params.resize(exec_conf.gpu.size());
    exec_conf.tagAll(__FILE__, __LINE__);
    for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
        {
        exec_conf.gpu[cur_gpu]->call(bind(cudaMallocHack, (void**)((void*)&m_gpu_params[cur_gpu]), m_bond_data->getNBondTypes()*sizeof(float2)));
        exec_conf.gpu[cur_gpu]->call(bind(cudaMemset, (void*)m_gpu_params[cur_gpu], 0, m_bond_data->getNBondTypes()*sizeof(float2)));
        }
        
    m_host_params = new float2[m_bond_data->getNBondTypes()];
    memset(m_host_params, 0, m_bond_data->getNBondTypes()*sizeof(float2));
    }

HarmonicBondForceComputeGPU::~HarmonicBondForceComputeGPU()
    {
    // free memory on the GPU
    exec_conf.tagAll(__FILE__, __LINE__);
    for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
        {
        exec_conf.gpu[cur_gpu]->call(bind(cudaFree, (void*)m_gpu_params[cur_gpu]));
        m_gpu_params[cur_gpu] = NULL;
        }
        
    // free memory on the CPU
    delete[] m_host_params;
    m_host_params = NULL;
    }

/*! \param type Type of the bond to set parameters for
    \param K Stiffness parameter for the force computation
    \param r_0 Equilibrium length for the force computation

    Sets parameters for the potential of a particular bond type and updates the
    parameters on the GPU.
*/
void HarmonicBondForceComputeGPU::setParams(unsigned int type, Scalar K, Scalar r_0)
    {
    HarmonicBondForceCompute::setParams(type, K, r_0);
    
    // update the local copy of the memory
    m_host_params[type] = make_float2(K, r_0);
    
    // copy the parameters to the GPU
    exec_conf.tagAll(__FILE__, __LINE__);
    for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
        exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, m_gpu_params[cur_gpu], m_host_params, m_bond_data->getNBondTypes()*sizeof(float2), cudaMemcpyHostToDevice));
    }

/*! Internal method for computing the forces on the GPU.
    \post The force data on the GPU is written with the calculated forces

    \param timestep Current time step of the simulation

    Calls gpu_compute_harmonic_bond_forces to do the dirty work.
*/
void HarmonicBondForceComputeGPU::computeForces(unsigned int timestep)
    {
    // start the profile
    if (m_prof) m_prof->push(exec_conf, "Harmonic");
    
    vector<gpu_bondtable_array>& gpu_bondtable = m_bond_data->acquireGPU();
    
    // the bond table is up to date: we are good to go. Call the kernel
    vector<gpu_pdata_arrays>& pdata = m_pdata->acquireReadOnlyGPU();
    gpu_boxsize box = m_pdata->getBoxGPU();
    
    // run the kernel in parallel on all GPUs
    exec_conf.tagAll(__FILE__, __LINE__);
    for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
        exec_conf.gpu[cur_gpu]->callAsync(bind(gpu_compute_harmonic_bond_forces, m_gpu_forces[cur_gpu].d_data, pdata[cur_gpu], box, gpu_bondtable[cur_gpu], m_gpu_params[cur_gpu], m_bond_data->getNBondTypes(), m_block_size));
    exec_conf.syncAll();
    
    // the force data is now only up to date on the gpu
    m_data_location = gpu;
    
    m_pdata->release();
    
    int64_t mem_transfer = m_pdata->getN() * 4+16+20 + m_bond_data->getNumBonds() * 2 * (8+16+8);
    int64_t flops = m_bond_data->getNumBonds() * 2 * (3+12+16+3+7);
    if (m_prof) m_prof->pop(exec_conf, flops, mem_transfer);
    }

void export_HarmonicBondForceComputeGPU()
    {
    class_<HarmonicBondForceComputeGPU, boost::shared_ptr<HarmonicBondForceComputeGPU>, bases<HarmonicBondForceCompute>, boost::noncopyable >
    ("HarmonicBondForceComputeGPU", init< boost::shared_ptr<SystemDefinition> >())
    .def("setBlockSize", &HarmonicBondForceComputeGPU::setBlockSize)
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif
