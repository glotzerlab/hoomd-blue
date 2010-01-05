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
using namespace boost;

#include "TwoStepNPTGPU.h"
#include "TwoStepNPTGPU.cuh"
#include "TwoStepNVTGPU.cuh"

/*! \file TwoStepNPTGPU.h
    \brief Contains code for the TwoStepNPTGPU class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param tau NPT temperature period
    \param tauP NPT pressure period
    \param T Temperature set point
    \param P Pressure set point
*/
TwoStepNPTGPU::TwoStepNPTGPU(boost::shared_ptr<SystemDefinition> sysdef,
                             boost::shared_ptr<ParticleGroup> group,
                             Scalar tau,
                             Scalar tauP,
                             boost::shared_ptr<Variant> T,
                             boost::shared_ptr<Variant> P)
    : TwoStepNPT(sysdef, group, tau, tauP, T, P)
    {
    // only one GPU is supported
    if (exec_conf.gpu.size() != 1)
        {
        cerr << endl << "***Error! Creating a TwoStepNPTGPU with 0 or more than one GPUs" << endl << endl;
        throw std::runtime_error("Error initializing TwoStepNVEGPU");
        }
    
    // allocate the total sum variables
    GPUArray<float> sum2K(1, m_pdata->getExecConf());
    m_sum2K.swap(sum2K);
    GPUArray<float> sumW(1, m_pdata->getExecConf());
    m_sumW.swap(sumW);
    
    // initialize the partial sum2K array
    m_block_size = 128;
    m_group_num_blocks = m_group->getNumMembers() / m_block_size + 1;
    m_full_num_blocks = m_pdata->getN() / m_block_size + 1;
    GPUArray<float> partial_sum2K(m_full_num_blocks, m_pdata->getExecConf());
    m_partial_sum2K.swap(partial_sum2K);
    GPUArray<float> partial_sumW(m_full_num_blocks, m_pdata->getExecConf());
    m_partial_sumW.swap(partial_sumW);
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per the NPT method
*/
void TwoStepNPTGPU::integrateStepOne(unsigned int timestep)
    {
    // profile this step
    if (m_prof)
        m_prof->push(exec_conf, "NPT step 1");

    IntegratorVariables v = getIntegratorVariables();
    Scalar& xi = v.variable[0];
    Scalar& eta = v.variable[1];

    // access all the needed data
    vector<gpu_pdata_arrays>& d_pdata = m_pdata->acquireReadWriteGPU();
    gpu_boxsize box = m_pdata->getBoxGPU();
    ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);
    unsigned int group_size = m_group->getIndexArray().getNumElements();
    ArrayHandle< float > d_partial_sum2K(m_partial_sum2K, access_location::device, access_mode::overwrite);

    // advance thermostat(m_Xi) half a time step
    xi += Scalar(1.0/2.0)/(m_tau*m_tau)*(m_curr_group_T/m_T->getValue(timestep) - Scalar(1.0))*m_deltaT;
    
    // advance barostat (m_Eta) half time step
    Scalar N = Scalar(m_group->getNumMembers());
    eta += Scalar(1.0/2.0)/(m_tauP*m_tauP)*m_V/(N*m_T->getValue(timestep))
            *(m_curr_P - m_P->getValue(timestep))*m_deltaT; 

    // perform the particle update on the GPU
    exec_conf.tagAll(__FILE__, __LINE__);
    exec_conf.gpu[0]->call(bind(gpu_npt_step_one,
                                d_pdata[0],
                                d_index_array.data,
                                group_size,
                                m_block_size,
                                m_group_num_blocks,
                                m_partial_scale,
                                xi,
                                eta,
                                m_deltaT));
    
    // advance volume
    m_V *= exp(Scalar(3.0)*eta*m_deltaT);
    
    // get the scaling factor for the box (V_new/V_old)^(1/3)
    Scalar box_len_scale = exp(eta*m_deltaT);
    m_Lx *= box_len_scale;
    m_Ly *= box_len_scale;
    m_Lz *= box_len_scale;
    
    // two things are done here
    // 1. particles may have been moved slightly outside the box by the above steps, wrap them back into place
    // 2. all particles in the box are rescaled to fit in the new box 
    exec_conf.tagAll(__FILE__,__LINE__);
    exec_conf.gpu[0]->call(bind(gpu_npt_boxscale, d_pdata[0],
                                                  box,
                                                  m_block_size,
                                                  m_partial_scale,
                                                  eta,
                                                  m_deltaT));
    
    // done profiling
    if (m_prof)
        m_prof->pop(exec_conf);
    
    m_pdata->release();
    m_pdata->setBox(BoxDim(m_Lx, m_Ly, m_Lz));
    }
        
/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1 on the GPU
*/
void TwoStepNPTGPU::integrateStepTwo(unsigned int timestep)
    {
    const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();
    
    // compute temperature for the next half time step
    m_curr_group_T = computeGroupTemperature(timestep+1);
    // compute pressure for the next half time step
    m_curr_P = computePressure(timestep+1);
    
    // profile this step
    if (m_prof)
        m_prof->push(exec_conf, "NPT step 2");

    IntegratorVariables v = getIntegratorVariables();
    Scalar& xi = v.variable[0];
    Scalar& eta = v.variable[1];

    vector<gpu_pdata_arrays>& d_pdata = m_pdata->acquireReadWriteGPU();
    ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::read);
    ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);
    unsigned int group_size = m_group->getIndexArray().getNumElements();
    
    // perform the update on the GPU
    exec_conf.gpu[0]->call(bind(gpu_npt_step_two,
                                d_pdata[0],
                                d_index_array.data,
                                group_size,
                                d_net_force.data,
                                m_block_size,
                                m_group_num_blocks,
                                xi,
                                eta,
                                m_deltaT));

    // Update state variables
    Scalar N = Scalar(m_group->getNumMembers());
    eta += Scalar(1.0/2.0)/(m_tauP*m_tauP)*m_V/(N*m_T->getValue(timestep))
                            *(m_curr_P - m_P->getValue(timestep))*m_deltaT;
    xi += Scalar(1.0/2.0)/(m_tau*m_tau)*(m_curr_group_T/m_T->getValue(timestep) - Scalar(1.0))*m_deltaT;

    m_pdata->release();
    setIntegratorVariables(v);
    
    // done profiling
    if (m_prof)
        m_prof->pop(exec_conf);
    }

Scalar TwoStepNPTGPU::computePressure(unsigned int timestep)
    {
    // grab access to the particle data
    vector<gpu_pdata_arrays>& d_pdata = m_pdata->acquireReadOnlyGPU();
    
    // first, run kernels on the GPU to compute the sum2K and sumW values
        {
        ArrayHandle<float> d_partial_sum2K(m_partial_sum2K, access_location::device, access_mode::overwrite);
        ArrayHandle<float> d_partial_sumW(m_partial_sumW, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar> d_virial(m_pdata->getNetVirial(), access_location::device, access_mode::read);
        
        // do the partial sum
        exec_conf.gpu[0]->call(bind(gpu_npt_pressure2,
                                      d_partial_sum2K.data,
                                      d_partial_sumW.data,
                                      d_pdata[0],
                                      d_virial.data,
                                      m_block_size,
                                      m_full_num_blocks));
        
        ArrayHandle<float> d_sum2K(m_sum2K, access_location::device, access_mode::overwrite);
        ArrayHandle<float> d_sumW(m_sumW, access_location::device, access_mode::overwrite);
        // reduce the partial sums
        exec_conf.gpu[0]->call(bind(gpu_nvt_reduce_sum2K, d_sum2K.data, d_partial_sum2K.data, m_full_num_blocks));
        exec_conf.gpu[0]->call(bind(gpu_nvt_reduce_sum2K, d_sumW.data, d_partial_sumW.data, m_full_num_blocks));
        }
    
    // now, access the data on the host
    ArrayHandle<float> h_sum2K(m_sum2K, access_location::host, access_mode::read);
    float ke_total = h_sum2K.data[0];
    ArrayHandle<float> h_sumW(m_sumW, access_location::host, access_mode::read);
    float W = h_sumW.data[0];
    
    ke_total *= 0.5;
    Scalar T = Scalar(2.0 * ke_total / m_dof);

    // volume/area & other 2D stuff needed
    BoxDim box = m_pdata->getBox();
    Scalar volume;
    unsigned int D = m_sysdef->getNDimensions();
    if (D == 2)
        {
        // "volume" is area in 2D
        volume = (box.xhi - box.xlo)*(box.yhi - box.ylo);
        // W needs to be corrected since the 1/3 factor is built in
        W *= Scalar(3.0/2.0);
        }
    else
        {
        volume = (box.xhi - box.xlo)*(box.yhi - box.ylo)*(box.zhi-box.zlo);
        }
    
    // done!
    m_pdata->release();
    
    // pressure: P = (N * K_B * T + W)/V
    Scalar N = Scalar(m_pdata->getN());
    return (N * T + W) / volume;
    }
        
Scalar TwoStepNPTGPU::computeGroupTemperature(unsigned int timestep)
    {
    // grab access to the particle data
    vector<gpu_pdata_arrays>& d_pdata = m_pdata->acquireReadOnlyGPU();

    // first, run kernels on the GPU to compute the sum2K value
        {
        ArrayHandle<float> d_partial_sum2K(m_partial_sum2K, access_location::device, access_mode::overwrite);
        ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);
        unsigned int group_size = m_group->getIndexArray().getNumElements();
        
        // do the partial sum
        exec_conf.gpu[0]->call(bind(gpu_npt_group_temperature,
                                      d_partial_sum2K.data,
                                      d_pdata[0],
                                      d_index_array.data,
                                      group_size,
                                      m_block_size,
                                      m_group_num_blocks));
        
        ArrayHandle<float> d_sum2K(m_sum2K, access_location::device, access_mode::overwrite);
        // reduce the partial sums
        exec_conf.gpu[0]->call(bind(gpu_nvt_reduce_sum2K, d_sum2K.data, d_partial_sum2K.data, m_group_num_blocks));
        }
    
    // now, access the data on the host
    ArrayHandle<float> h_sum2K(m_sum2K, access_location::host, access_mode::read);
    float ke_total = h_sum2K.data[0];
    
    ke_total *= 0.5;
    
    // done!
    m_pdata->release();
    return 2.0 * ke_total / m_group_dof;
    }

void export_TwoStepNPTGPU()
    {
    class_<TwoStepNPTGPU, boost::shared_ptr<TwoStepNPTGPU>, bases<TwoStepNPT>, boost::noncopyable>
        ("TwoStepNPTGPU", init< boost::shared_ptr<SystemDefinition>,
                          boost::shared_ptr<ParticleGroup>,
                          Scalar,
                          Scalar,
                          boost::shared_ptr<Variant>,
                          boost::shared_ptr<Variant> >())
        ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

