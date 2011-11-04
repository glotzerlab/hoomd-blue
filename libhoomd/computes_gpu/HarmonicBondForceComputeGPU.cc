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

* All publications based on HOOMD-blue, including any reports or published
results obtained, in whole or in part, with HOOMD-blue, will acknowledge its use
according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website
at: http://codeblue.umich.edu/hoomd-blue/.

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

/*! \param sysdef System to compute bond forces on
    \param log_suffix Name given to this instance of the harmonic bond

*/
HarmonicBondForceComputeGPU::HarmonicBondForceComputeGPU(boost::shared_ptr<SystemDefinition> sysdef, const std::string& log_suffix )
    : HarmonicBondForceCompute(sysdef, log_suffix), m_block_size(64)
    {
    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!exec_conf->isCUDAEnabled())
        {
        cerr << endl << "***Error! Creating a BondForceComputeGPU with no GPU in the execution configuration" << endl << endl;
        throw std::runtime_error("Error initializing BondForceComputeGPU");
        }
        
    // allocate and zero device memory
    GPUArray<float2> params(m_bond_data->getNBondTypes(), exec_conf);
    m_params.swap(params);
    }

HarmonicBondForceComputeGPU::~HarmonicBondForceComputeGPU()
    {
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
   
    ArrayHandle<float2> h_params(m_params, access_location::host, access_mode::readwrite);
    h_params.data[type] = make_float2(K, r_0);
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
    
    gpu_bondtable_array& gpu_bondtable = m_bond_data->acquireGPU();
    
    // the bond table is up to date: we are good to go. Call the kernel
    gpu_pdata_arrays& pdata = m_pdata->acquireReadOnlyGPU();
    gpu_boxsize box = m_pdata->getBoxGPU();
      
    ArrayHandle<Scalar4> d_force(m_force,access_location::device,access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(m_virial,access_location::device,access_mode::overwrite);
    ArrayHandle<float2> d_params(m_params, access_location::device, access_mode::read);

    // run the kernel in parallel on all GPUs
    gpu_compute_harmonic_bond_forces(d_force.data,
                                     d_virial.data,
                                     pdata,
                                     box,
                                     gpu_bondtable,
                                     d_params.data,
                                     m_bond_data->getNBondTypes(),
                                     m_block_size);
    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
       
    m_pdata->release();
    
    int64_t mem_transfer = m_pdata->getN() * 4+16+20 + m_bond_data->getNumBonds() * 2 * (8+16+8);
    int64_t flops = m_bond_data->getNumBonds() * 2 * (3+12+16+3+7);
    if (m_prof) m_prof->pop(exec_conf, flops, mem_transfer);
    }

void export_HarmonicBondForceComputeGPU()
    {
    class_<HarmonicBondForceComputeGPU, boost::shared_ptr<HarmonicBondForceComputeGPU>, bases<HarmonicBondForceCompute>, boost::noncopyable >
    ("HarmonicBondForceComputeGPU", init< boost::shared_ptr<SystemDefinition>, const std::string& >())
    .def("setBlockSize", &HarmonicBondForceComputeGPU::setBlockSize)
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

