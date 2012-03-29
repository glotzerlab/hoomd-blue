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
    \param thermo_group ComputeThermo to compute thermo properties of the integrated \a group
    \param thermo_all ComputeThermo to compute the pressure of the entire system
    \param tau NPT temperature period
    \param tauP NPT pressure period
    \param T Temperature set point
    \param P Pressure set point
*/
TwoStepNPTGPU::TwoStepNPTGPU(boost::shared_ptr<SystemDefinition> sysdef,
                             boost::shared_ptr<ParticleGroup> group,
                             boost::shared_ptr<ComputeThermo> thermo_group,
                             boost::shared_ptr<ComputeThermo> thermo_all,
                             Scalar tau,
                             Scalar tauP,
                             boost::shared_ptr<Variant> T,
                             boost::shared_ptr<Variant> P)
    : TwoStepNPT(sysdef, group, thermo_group, thermo_all, tau, tauP, T, P)
    {
    // only one GPU is supported
    if (!exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating a TwoStepNPTGPU with CUDA disabled" << endl;
        throw std::runtime_error("Error initializing TwoStepNVEGPU");
        }
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per the NPT method
*/
void TwoStepNPTGPU::integrateStepOne(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();
    if (group_size == 0)
        return;
    
    // profile this step
    if (m_prof)
        m_prof->push(exec_conf, "NPT step 1");

    IntegratorVariables v = getIntegratorVariables();
    Scalar& xi = v.variable[0];
    Scalar& eta = v.variable[1];

    if (!m_state_initialized)
        {
        // compute the current thermodynamic properties
        m_thermo_group->compute(timestep);
        m_thermo_all->compute(timestep);
        
        // compute temperature for the next half time step
        m_curr_group_T = m_thermo_group->getTemperature();
        // compute pressure for the next half time step
        m_curr_P = m_thermo_all->getPressure();
        // if it is not valid, assume that the current pressure is the set pressure (this should only happen in very 
        // rare circumstances, usually at the start of the simulation before things are initialize)
        if (isnan(m_curr_P))
            m_curr_P = m_P->getValue(timestep);
        
        m_state_initialized = true;
        }

    // access all the needed data
    {
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::read);
    ArrayHandle<int3> d_image(m_pdata->getImages(), access_location::device, access_mode::readwrite);

    gpu_boxsize box = m_pdata->getBoxGPU();
    ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);

    // advance thermostat(m_Xi) half a time step
    xi += Scalar(1.0/2.0)/(m_tau*m_tau)*(m_curr_group_T/m_T->getValue(timestep) - Scalar(1.0))*m_deltaT;
    
    // advance barostat (m_Eta) half time step
    Scalar N = Scalar(m_group->getNumMembers());
    eta += Scalar(1.0/2.0)/(m_tauP*m_tauP)*m_V/(N*m_T->getValue(timestep))
            *(m_curr_P - m_P->getValue(timestep))*m_deltaT; 

    // perform the particle update on the GPU
    gpu_npt_step_one(d_pos.data,
                     d_vel.data,
                     d_accel.data,
                     d_index_array.data,
                     group_size,
                     m_partial_scale,
                     xi,
                     eta,
                     m_deltaT);

    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    
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
    gpu_npt_boxscale(m_pdata->getN(),
                     d_pos.data,
                     d_image.data,
                     box,
                     m_partial_scale,
                     eta,
                     m_deltaT);

    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    
    // done profiling
    if (m_prof)
        m_prof->pop(exec_conf);

    } //end of GPUArray scope

    m_pdata->setBox(BoxDim(m_Lx, m_Ly, m_Lz));
    setIntegratorVariables(v);
    }

/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1 on the GPU
*/
void TwoStepNPTGPU::integrateStepTwo(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();
    if (group_size == 0)
        return;
    
    const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();
    
    // compute the current thermodynamic properties
    m_thermo_group->compute(timestep+1);
    m_thermo_all->compute(timestep+1);
    
    // compute temperature for the next half time step
    m_curr_group_T = m_thermo_group->getTemperature();
    // compute pressure for the next half time step
    m_curr_P = m_thermo_all->getPressure();
    // if it is not valid, assume that the current pressure is the set pressure (this should only happen in very 
    // rare circumstances, usually at the start of the simulation before things are initialize)
    if (isnan(m_curr_P))
        m_curr_P = m_P->getValue(timestep);
    
    // profile this step
    if (m_prof)
        m_prof->push(exec_conf, "NPT step 2");

    IntegratorVariables v = getIntegratorVariables();
    Scalar& xi = v.variable[0];
    Scalar& eta = v.variable[1];

    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::readwrite);

    ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::read);
    ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);
    
    // perform the update on the GPU
    gpu_npt_step_two(d_vel.data,
                     d_accel.data,
                     d_index_array.data,
                     group_size,
                     d_net_force.data,
                     xi,
                     eta,
                     m_deltaT);

    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    
    // Update state variables
    Scalar N = Scalar(m_group->getNumMembers());
    eta += Scalar(1.0/2.0)/(m_tauP*m_tauP)*m_V/(N*m_T->getValue(timestep))
                            *(m_curr_P - m_P->getValue(timestep))*m_deltaT;
    xi += Scalar(1.0/2.0)/(m_tau*m_tau)*(m_curr_group_T/m_T->getValue(timestep) - Scalar(1.0))*m_deltaT;

    setIntegratorVariables(v);
    
    // done profiling
    if (m_prof)
        m_prof->pop(exec_conf);
    }

void export_TwoStepNPTGPU()
    {
    class_<TwoStepNPTGPU, boost::shared_ptr<TwoStepNPTGPU>, bases<TwoStepNPT>, boost::noncopyable>
        ("TwoStepNPTGPU", init< boost::shared_ptr<SystemDefinition>,
                          boost::shared_ptr<ParticleGroup>,
                          boost::shared_ptr<ComputeThermo>,
                          boost::shared_ptr<ComputeThermo>,
                          Scalar,
                          Scalar,
                          boost::shared_ptr<Variant>,
                          boost::shared_ptr<Variant> >())
        ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

