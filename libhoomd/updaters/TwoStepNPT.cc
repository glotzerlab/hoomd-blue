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

#include "TwoStepNPT.h"

/*! \file TwoStepNPT.h
    \brief Contains code for the TwoStepNPT class
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
TwoStepNPT::TwoStepNPT(boost::shared_ptr<SystemDefinition> sysdef,
                       boost::shared_ptr<ParticleGroup> group,
                       boost::shared_ptr<ComputeThermo> thermo_group,
                       boost::shared_ptr<ComputeThermo> thermo_all,
                       Scalar tau,
                       Scalar tauP,
                       boost::shared_ptr<Variant> T,
                       boost::shared_ptr<Variant> P)
    : IntegrationMethodTwoStep(sysdef, group), m_thermo_group(thermo_group), m_thermo_all(thermo_all), 
      m_partial_scale(false), m_tau(tau), m_tauP(tauP), m_T(T), m_P(P), m_state_initialized(false)
    {
    if (m_tau <= 0.0)
        cout << "***Warning! tau set less than 0.0 in TwoStepNPT" << endl;
    if (m_tauP <= 0.0)
        cout << "***Warning! tauP set less than 0.0 in TwoStepNPT" << endl;
    
    // precalculate box lengths
    const BoxDim& box = m_pdata->getBox();
    Scalar3 L = box.getL();
    m_Lx = L.x;
    m_Ly = L.y;
    m_Lz = L.z;
    
    m_V = m_Lx*m_Ly*m_Lz;   // volume
    
    // set initial state
    IntegratorVariables v = getIntegratorVariables();

    // choose dummy values for the current temp and pressure
    m_curr_group_T = 0.0;
    m_curr_P = 0.0;

    if (!restartInfoTestValid(v, "npt", 2))
        {
        v.type = "npt";
        v.variable.resize(2);
        v.variable[0] = Scalar(0.0);
        v.variable[1] = Scalar(0.0);
        setValidRestart(false);
        }
    else
        setValidRestart(true);

    setIntegratorVariables(v);
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per the Nose-Hoover
     thermostat and Anderson barostat
*/
void TwoStepNPT::integrateStepOne(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();
    if (group_size == 0)
        return;

    BoxDim box = m_pdata->getBox();
    
    // profile this step
    if (m_prof)
        m_prof->push("NPT step 1");

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

    IntegratorVariables v = getIntegratorVariables();
    Scalar& xi = v.variable[0];
    Scalar& eta = v.variable[1];

    {
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);

    // advance thermostat(xi) half a time step
    xi += Scalar(1.0/2.0)/(m_tau*m_tau)*(m_curr_group_T/m_T->getValue(timestep) - Scalar(1.0))*m_deltaT;
    
    // advance barostat (eta) half time step
    Scalar N = Scalar(m_group->getNumMembers());
    eta += Scalar(1.0/2.0)/(m_tauP*m_tauP)*m_V/(N*m_T->getValue(timestep))
            *(m_curr_P - m_P->getValue(timestep))*m_deltaT;    
    
    // precompute loop invariant quantities
    Scalar exp_v_fac = exp(-Scalar(1.0/4.0)*(eta+xi)*m_deltaT);
    Scalar exp_r_fac = exp(Scalar(1.0/2.0)*eta*m_deltaT);
    Scalar exp_r_fac_inv = Scalar(1.0) / exp_r_fac;
        
    // perform the first half step of NPT
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        
        h_vel.data[j].x = h_vel.data[j].x*exp_v_fac*exp_v_fac + Scalar(1.0/2.0)*m_deltaT*exp_v_fac*h_accel.data[j].x;
        h_pos.data[j].x = h_pos.data[j].x + h_vel.data[j].x*exp_r_fac_inv*m_deltaT;
        
        h_vel.data[j].y = h_vel.data[j].y*exp_v_fac*exp_v_fac + Scalar(1.0/2.0)*m_deltaT*exp_v_fac*h_accel.data[j].y;
        h_pos.data[j].y = h_pos.data[j].y + h_vel.data[j].y*exp_r_fac_inv*m_deltaT;

        h_vel.data[j].z = h_vel.data[j].z*exp_v_fac*exp_v_fac + Scalar(1.0/2.0)*m_deltaT*exp_v_fac*h_accel.data[j].z;
        h_pos.data[j].z = h_pos.data[j].z + h_vel.data[j].z*exp_r_fac_inv*m_deltaT;
        
        if (m_partial_scale)
            {
            h_pos.data[j].x *= exp_r_fac * exp_r_fac;
            h_pos.data[j].y *= exp_r_fac * exp_r_fac;
            h_pos.data[j].z *= exp_r_fac * exp_r_fac;
            }
        }
    
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

    box.setL(make_scalar3(m_Lx, m_Ly, m_Lz));
    
    ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);
    
    for (unsigned int j = 0; j < m_pdata->getN(); j++)
        {
        if (!m_partial_scale)
            {
            h_pos.data[j].x *= box_len_scale;
            h_pos.data[j].y *= box_len_scale;
            h_pos.data[j].z *= box_len_scale;
            }
        box.wrap(h_pos.data[j], h_image.data[j]);
        }

    } // end of GPUArray scope
    setIntegratorVariables(v);
        
    // done profiling
    if (m_prof)
        m_prof->pop();
        
    m_pdata->setGlobalBoxL(box.getL());
    }
        
/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1
*/
void TwoStepNPT::integrateStepTwo(unsigned int timestep)
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
        m_prof->push("NPT step 2");

    IntegratorVariables v = getIntegratorVariables();
    Scalar& xi = v.variable[0];
    Scalar& eta = v.variable[1];

    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::readwrite);

    ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);
    
    // compute loop invariant quantities
    Scalar exp_v_fac = exp(-Scalar(1.0/4.0)*(eta+xi)*m_deltaT);
    
    // perform second half step of NPT integration
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        
        // first, calculate acceleration from the net force
        Scalar minv = Scalar(1.0) / h_vel.data[j].w;
        h_accel.data[j].x = h_net_force.data[j].x*minv;
        h_accel.data[j].y = h_net_force.data[j].y*minv;
        h_accel.data[j].z = h_net_force.data[j].z*minv;
        
        // then, update the velocity
        h_vel.data[j].x = h_vel.data[j].x*exp_v_fac*exp_v_fac + Scalar(1.0/2.0)*m_deltaT*exp_v_fac*h_accel.data[j].x;
        h_vel.data[j].y = h_vel.data[j].y*exp_v_fac*exp_v_fac + Scalar(1.0/2.0)*m_deltaT*exp_v_fac*h_accel.data[j].y;
        h_vel.data[j].z = h_vel.data[j].z*exp_v_fac*exp_v_fac + Scalar(1.0/2.0)*m_deltaT*exp_v_fac*h_accel.data[j].z;
        }
    
    // Update state variables
    Scalar N = Scalar(m_group->getNumMembers());
    eta += Scalar(1.0/2.0)/(m_tauP*m_tauP)*m_V/(N*m_T->getValue(timestep))
                            *(m_curr_P - m_P->getValue(timestep))*m_deltaT;
    xi += Scalar(1.0/2.0)/(m_tau*m_tau)*(m_curr_group_T/m_T->getValue(timestep) - Scalar(1.0))*m_deltaT;
    
    setIntegratorVariables(v);
        
    // done profiling
    if (m_prof)
        m_prof->pop();
    }


void export_TwoStepNPT()
    {
    class_<TwoStepNPT, boost::shared_ptr<TwoStepNPT>, bases<IntegrationMethodTwoStep>, boost::noncopyable>
        ("TwoStepNPT", init< boost::shared_ptr<SystemDefinition>,
                       boost::shared_ptr<ParticleGroup>,
                       boost::shared_ptr<ComputeThermo>,
                       boost::shared_ptr<ComputeThermo>,
                       Scalar,
                       Scalar,
                       boost::shared_ptr<Variant>,
                       boost::shared_ptr<Variant> >())
        .def("setT", &TwoStepNPT::setT)
        .def("setP", &TwoStepNPT::setP)
        .def("setTau", &TwoStepNPT::setTau)
        .def("setTauP", &TwoStepNPT::setTauP)
        .def("setPartialScale", &TwoStepNPT::setPartialScale)
        ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

