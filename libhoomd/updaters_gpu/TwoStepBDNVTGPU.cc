/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
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

// Maintainer: joaander

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;
#include <boost/bind.hpp>
using namespace boost;

#include "TwoStepBDNVTGPU.h"
#include "TwoStepNVEGPU.cuh"
#include "TwoStepBDNVTGPU.cuh"

/*! \file TwoStepBDNVTGPU.h
    \brief Contains code for the TwoStepBDNVTGPU class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param T Temperature set point as a function of time
    \param seed Random seed to use in generating random numbers
    \param gamma_diam Set gamma to the particle diameter of each particle if true, otherwise use a per-type
                      gamma via setGamma()
    \param suffix Suffix to attach to the end of log quantity names
*/
TwoStepBDNVTGPU::TwoStepBDNVTGPU(boost::shared_ptr<SystemDefinition> sysdef,
                                 boost::shared_ptr<ParticleGroup> group,
                                 boost::shared_ptr<Variant> T,
                                 unsigned int seed,
                                 bool gamma_diam,
                                 const std::string& suffix)
    : TwoStepBDNVT(sysdef, group, T, seed, gamma_diam, suffix)
    {
    // only one GPU is supported
    if (!exec_conf->isCUDAEnabled())
        {
        cerr << endl << "***Error! Creating a TwoStepNVEGPU what CUDA is disabled" << endl << endl;
        throw std::runtime_error("Error initializing TwoStepNVEGPU");
        }
        
    // allocate the sum arrays
    GPUArray<float> sum(1, exec_conf);
    m_sum.swap(sum);
    
    // initialize the partial sum array
    m_block_size = 256; 
    unsigned int group_size = m_group->getIndexArray().getNumElements();    
    m_num_blocks = group_size / m_block_size + 1;
    GPUArray<float> partial_sum1(m_num_blocks, exec_conf);
    m_partial_sum1.swap(partial_sum1);          
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per the velocity verlet
          method.
    
    This method is copied directoy from TwoStepNVEGPU::integrateStepOne() and reimplemented here to avoid multiple
    inheritance.
*/
void TwoStepBDNVTGPU::integrateStepOne(unsigned int timestep)
    {
    // profile this step
    if (m_prof)
        m_prof->push(exec_conf, "NVE step 1");
    
    // access all the needed data
    gpu_pdata_arrays& d_pdata = m_pdata->acquireReadWriteGPU();
    gpu_boxsize box = m_pdata->getBoxGPU();
    ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);
    unsigned int group_size = m_group->getIndexArray().getNumElements();
    
    // perform the update on the GPU
    gpu_nve_step_one(d_pdata,
                     d_index_array.data,
                     group_size,
                     box,
                     m_deltaT,
                     m_limit,
                     m_limit_val,
                     m_zero_force);

    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    
    m_pdata->release();
    
    // done profiling
    if (m_prof)
        m_prof->pop(exec_conf);
    }
        
/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1 on the GPU
*/
void TwoStepBDNVTGPU::integrateStepTwo(unsigned int timestep)
    {
    const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();
    
    // profile this step
    if (m_prof)
        m_prof->push(exec_conf, "NVE step 2");
    
    // get the dimensionality of the system
    const Scalar D = Scalar(m_sysdef->getNDimensions());
    
    gpu_pdata_arrays& d_pdata = m_pdata->acquireReadWriteGPU();
    ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_gamma(m_gamma, access_location::device, access_mode::read);
    ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);
 
        {
        ArrayHandle<float> d_partial_sumBD(m_partial_sum1, access_location::device, access_mode::overwrite);
        ArrayHandle<float> d_sumBD(m_sum, access_location::device, access_mode::overwrite);
        
        // perform the update on the GPU
        bdnvt_step_two_args args;
        args.d_gamma = d_gamma.data;
        args.n_types = m_gamma.getNumElements();
        args.gamma_diam = m_gamma_diam;
        args.T = m_T->getValue(timestep);
        args.timestep = timestep;
        args.seed = m_seed;
        args.d_sum_bdenergy = d_sumBD.data;
        args.d_partial_sum_bdenergy = d_partial_sumBD.data;
        args.block_size = m_block_size;
        args.num_blocks = m_num_blocks;
        args.tally = m_tally;
        
        unsigned int group_size = m_group->getIndexArray().getNumElements();
   
        gpu_bdnvt_step_two(d_pdata,
                           d_index_array.data,
                           group_size,
                           d_net_force.data,
                           args,
                           m_deltaT,
                           D,
                           m_limit,
                           m_limit_val);

        if (exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        
        }
    m_pdata->release();
 
    if (m_tally)
        {
        ArrayHandle<float> h_sumBD(m_sum, access_location::host, access_mode::read);   
        m_reservoir_energy -= h_sumBD.data[0]*m_deltaT;
        m_extra_energy_overdeltaT= 0.5*h_sumBD.data[0];
        }
    // done profiling
    if (m_prof)
        m_prof->pop(exec_conf);
    }

void export_TwoStepBDNVTGPU()
    {
    class_<TwoStepBDNVTGPU, boost::shared_ptr<TwoStepBDNVTGPU>, bases<TwoStepBDNVT>, boost::noncopyable>
        ("TwoStepBDNVTGPU", init< boost::shared_ptr<SystemDefinition>,
                         boost::shared_ptr<ParticleGroup>,
                         boost::shared_ptr<Variant>,
                         unsigned int,
                         bool,
                         const std::string&
                         >())
        ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

