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
// Maintainer: askeys

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;
#include <boost/bind.hpp>
using namespace boost;

#include "FIREEnergyMinimizerGPU.h"
#include "FIREEnergyMinimizerGPU.cuh"
#include "TwoStepNVEGPU.h"

/*! \file FIREEnergyMinimizerGPU.h
    \brief Contains code for the FIREEnergyMinimizerGPU class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param dt Default step size

    \post The method is constructed with the given particle data and a NULL profiler.
*/
FIREEnergyMinimizerGPU::FIREEnergyMinimizerGPU(boost::shared_ptr<SystemDefinition> sysdef, boost::shared_ptr<ParticleGroup> group, Scalar dt)
    :   FIREEnergyMinimizer(sysdef, group, dt, false)
    {
    const ParticleDataArraysConst& arrays = m_pdata->acquireReadOnly();
    m_nparticles = arrays.nparticles;
    m_pdata->release();

    // only one GPU is supported
    if (exec_conf.gpu.size() != 1)
        {
        cerr << endl << "***Error! Creating a FIREEnergyMinimizer with 0 or more than one GPUs" << endl << endl;
        throw std::runtime_error("Error initializing FIREEnergyMinimizer");
        }
    
    // allocate the sum arrays
    GPUArray<float> sum(1, m_pdata->getExecConf());
    m_sum.swap(sum);
    GPUArray<float> sum3(3, m_pdata->getExecConf());
    m_sum3.swap(sum3);
    
    // initialize the partial sum arrays
    m_block_size = 256; //128;
    m_num_blocks = m_nparticles / m_block_size + 1;
    GPUArray<float> partial_sum1(m_num_blocks, m_pdata->getExecConf());
    m_partial_sum1.swap(partial_sum1);
    GPUArray<float> partial_sum2(m_num_blocks, m_pdata->getExecConf());
    m_partial_sum2.swap(partial_sum2);
    GPUArray<float> partial_sum3(m_num_blocks, m_pdata->getExecConf());
    m_partial_sum3.swap(partial_sum3);
    
    reset();
    createIntegrator();
    }

void FIREEnergyMinimizerGPU::createIntegrator()
    {
//   boost::shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(m_sysdef, 0, m_pdata->getN()-1));
//    boost::shared_ptr<ParticleGroup> group_all(new ParticleGroup(m_sysdef, selector_all));
    boost::shared_ptr<TwoStepNVEGPU> integrator(new TwoStepNVEGPU(m_sysdef, m_group));
    addIntegrationMethod(integrator);
    setDeltaT(m_deltaT);
    }

void FIREEnergyMinimizerGPU::reset()
    {
    m_converged = false;
    m_n_since_negative = 0;
    m_alpha = m_alpha_start;
    m_was_reset = true;
    vector<gpu_pdata_arrays>& d_pdata = m_pdata->acquireReadWriteGPU();
    ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);
    unsigned int group_size = m_group->getIndexArray().getNumElements();    
    exec_conf.gpu[0]->call(bind(gpu_fire_zero_v, 
                                d_pdata[0],
                                d_index_array.data,
                                group_size));
    m_pdata->release();
    setDeltaT(m_deltaT_set);
    }

/*! \param timesteps is the iteration number
*/
void FIREEnergyMinimizerGPU::update(unsigned int timesteps)
    {
        
    if (m_converged)
        return;
        
    IntegratorTwoStep::update(timesteps);
        
    Scalar P(0.0);
    Scalar vnorm(0.0);
    Scalar fnorm(0.0);
    Scalar energy(0.0);

    // compute the total energy on the GPU
    // CPU version is Scalar energy = computePotentialEnergy(timesteps)/Scalar(group_size);
    
    if (m_prof)
        m_prof->push(exec_conf, "FIRE compute total energy");
    
    vector<gpu_pdata_arrays>& d_pdata = m_pdata->acquireReadWriteGPU();
    ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);
    unsigned int group_size = m_group->getIndexArray().getNumElements();
    ArrayHandle<Scalar4> d_net_force(m_pdata->getNetForce(), access_location::device, access_mode::read);
    ArrayHandle<float> d_partial_sumE(m_partial_sum1, access_location::device, access_mode::overwrite);
    ArrayHandle<float> d_sumE(m_sum, access_location::device, access_mode::overwrite);
    
    
    exec_conf.gpu[0]->call(bind(gpu_fire_compute_sum_pe, 
                                d_pdata[0], 
                                d_index_array.data,
                                group_size,
                                d_net_force.data, 
                                d_sumE.data, 
                                d_partial_sumE.data, 
                                m_block_size, 
                                m_num_blocks));
    
    
    ArrayHandle<float> h_sumE(m_sum, access_location::host, access_mode::read);
    energy = h_sumE.data[0]/Scalar(group_size);    
    
    if (m_prof)
        m_prof->pop(exec_conf);

    

    if (m_was_reset)
        {
        m_was_reset = false;
        m_old_energy = energy + Scalar(100000)*m_etol;
        }
    
    //sum P, vnorm, fnorm
    
    if (m_prof)
        m_prof->push(exec_conf, "FIRE P, vnorm, fnorm");
    
    
    ArrayHandle<float> d_partial_sum_P(m_partial_sum1, access_location::device, access_mode::overwrite);
    ArrayHandle<float> d_partial_sum_vsq(m_partial_sum2, access_location::device, access_mode::overwrite);
    ArrayHandle<float> d_partial_sum_fsq(m_partial_sum3, access_location::device, access_mode::overwrite);
    ArrayHandle<float> d_sum(m_sum3, access_location::device, access_mode::overwrite);
    
    exec_conf.gpu[0]->call(bind(gpu_fire_compute_sum_all, 
                                d_pdata[0], 
                                d_index_array.data,
                                group_size,
                                d_sum.data, 
                                d_partial_sum_P.data, 
                                d_partial_sum_vsq.data, 
                                d_partial_sum_fsq.data, 
                                m_block_size,
                                m_num_blocks));
    
    
    ArrayHandle<float> h_sum(m_sum3, access_location::host, access_mode::read);
    P = h_sum.data[0];
    vnorm = sqrt(h_sum.data[1]);
    fnorm = sqrt(h_sum.data[2]);
    
    if (m_prof)
        m_prof->pop(exec_conf);            
    
    
    if (fnorm/sqrt(m_sysdef->getNDimensions()*group_size) < m_ftol || fabs(energy-m_old_energy) < m_etol)
        {
        m_converged = true;
        return;
        }

    //update velocities
    
    if (m_prof)
        m_prof->push(exec_conf, "FIRE update velocities");

    Scalar invfnorm = 1.0/fnorm;        

    exec_conf.gpu[0]->call(bind(gpu_fire_update_v,
                                d_pdata[0], 
                                d_index_array.data,
                                group_size,
                                m_alpha, 
                                vnorm, 
                                invfnorm));

    if (m_prof)
        m_prof->pop(exec_conf);                
    
        
    if (P > Scalar(0.0))
        {
        m_n_since_negative++;
        if (m_n_since_negative > m_nmin)
            {
            IntegratorTwoStep::setDeltaT(std::min(m_deltaT*m_finc, m_deltaT_max));
            m_alpha *= m_falpha;
            }
        }
    else if (P <= Scalar(0.0))
        {
        IntegratorTwoStep::setDeltaT(m_deltaT*m_fdec);
        m_alpha = m_alpha_start;
        m_n_since_negative = 0;
        if (m_prof)
            m_prof->push(exec_conf, "FIRE zero velocities");

        exec_conf.gpu[0]->call(bind(gpu_fire_zero_v, 
                                    d_pdata[0],
                                    d_index_array.data,
                                    group_size));
        if (m_prof)
            m_prof->pop(exec_conf);        
        }
    m_old_energy = energy;
    
    m_pdata->release();  
    }


void export_FIREEnergyMinimizerGPU()
    {
    class_<FIREEnergyMinimizerGPU, boost::shared_ptr<FIREEnergyMinimizerGPU>,  bases<FIREEnergyMinimizer>, boost::noncopyable>
        ("FIREEnergyMinimizerGPU", init< boost::shared_ptr<SystemDefinition>, boost::shared_ptr<ParticleGroup>, Scalar >())
        ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

