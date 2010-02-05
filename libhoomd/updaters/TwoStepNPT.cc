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

#include "TwoStepNPT.h"

/*! \file TwoStepNPT.h
    \brief Contains code for the TwoStepNPT class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param tau NPT temperature period
    \param tauP NPT pressure period
    \param T Temperature set point
    \param P Pressure set point
*/
TwoStepNPT::TwoStepNPT(boost::shared_ptr<SystemDefinition> sysdef,
                       boost::shared_ptr<ParticleGroup> group,
                       Scalar tau,
                       Scalar tauP,
                       boost::shared_ptr<Variant> T,
                       boost::shared_ptr<Variant> P)
    : IntegrationMethodTwoStep(sysdef, group), m_partial_scale(false), m_tau(tau), m_tauP(tauP), m_T(T), m_P(P)
    {
    if (m_tau <= 0.0)
        cout << "***Warning! tau set less than 0.0 in TwoStepNPT" << endl;
    if (m_tauP <= 0.0)
        cout << "***Warning! tauP set less than 0.0 in TwoStepNPT" << endl;
    
    // precalculate box lengths
    const BoxDim& box = m_pdata->getBox();
    
    m_Lx = box.xhi - box.xlo;
    m_Ly = box.yhi - box.ylo;
    m_Lz = box.zhi - box.zlo;
    
    m_V = m_Lx*m_Ly*m_Lz;   // volume
    
    // initialize temperature and pressure computations
    unsigned int dim = m_sysdef->getNDimensions();
    m_group_dof = Scalar(dim*m_group->getNumMembers() - dim);
    m_dof = Scalar(dim*m_pdata->getN() - dim);
    
    // compute the current pressure and temperature on construction
    m_curr_group_T = computeGroupTemperature(0);
    m_curr_P = computePressure(0);
    
    // set initial state
    IntegratorVariables v = getIntegratorVariables();

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

    // profile this step
    if (m_prof)
        m_prof->push("NPT step 1");

    IntegratorVariables v = getIntegratorVariables();
    Scalar& xi = v.variable[0];
    Scalar& eta = v.variable[1];

    const ParticleDataArrays& arrays = m_pdata->acquireReadWrite();
    
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
        
        arrays.vx[j] = arrays.vx[j]*exp_v_fac*exp_v_fac + Scalar(1.0/2.0)*m_deltaT*exp_v_fac*arrays.ax[j];
        arrays.x[j] = arrays.x[j] + arrays.vx[j]*exp_r_fac_inv*m_deltaT;
        
        arrays.vy[j] = arrays.vy[j]*exp_v_fac*exp_v_fac + Scalar(1.0/2.0)*m_deltaT*exp_v_fac*arrays.ay[j];
        arrays.y[j] = arrays.y[j] + arrays.vy[j]*exp_r_fac_inv*m_deltaT;
        
        arrays.vz[j] = arrays.vz[j]*exp_v_fac*exp_v_fac + Scalar(1.0/2.0)*m_deltaT*exp_v_fac*arrays.az[j];
        arrays.z[j] = arrays.z[j] + arrays.vz[j]*exp_r_fac_inv*m_deltaT;
        
        if (m_partial_scale)
            {
            arrays.x[j] *= exp_r_fac * exp_r_fac;
            arrays.y[j] *= exp_r_fac * exp_r_fac;
            arrays.z[j] *= exp_r_fac * exp_r_fac;
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
    
    for (unsigned int j = 0; j < m_pdata->getN(); j++)
        {
        if (!m_partial_scale)
            {
            arrays.x[j] *= box_len_scale;
            arrays.y[j] *= box_len_scale;
            arrays.z[j] *= box_len_scale;
            }
        
        // wrap the particle around the box
        if (arrays.x[j] >= Scalar(m_Lx/2.0))
            {
            arrays.x[j] -= m_Lx;
            arrays.ix[j]++;
            }
        else if (arrays.x[j] < Scalar(-m_Lx/2.0))
            {
            arrays.x[j] += m_Lx;
            arrays.ix[j]--;
            }
            
        if (arrays.y[j] >= Scalar(m_Ly/2.0))
            {
            arrays.y[j] -= m_Ly;
            arrays.iy[j]++;
            }
        else if (arrays.y[j] < Scalar(-m_Ly/2.0))
            {
            arrays.y[j] += m_Ly;
            arrays.iy[j]--;
            }
            
        if (arrays.z[j] >= Scalar(m_Lz/2.0))
            {
            arrays.z[j] -= m_Lz;
            arrays.iz[j]++;
            }
        else if (arrays.z[j] < Scalar(-m_Lz/2.0))
            {
            arrays.z[j] += m_Lz;
            arrays.iz[j]--;
            }
        }
    
    m_pdata->release();
    setIntegratorVariables(v);
        
    // done profiling
    if (m_prof)
        m_prof->pop();
        
    m_pdata->setBox(BoxDim(m_Lx, m_Ly, m_Lz));
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
   
    // compute temperature for the next half time step
    m_curr_group_T = computeGroupTemperature(timestep+1);
    // compute pressure for the next half time step
    m_curr_P = computePressure(timestep+1);

    // profile this step
    if (m_prof)
        m_prof->push("NPT step 2");

    IntegratorVariables v = getIntegratorVariables();
    Scalar& xi = v.variable[0];
    Scalar& eta = v.variable[1];

    const ParticleDataArrays& arrays = m_pdata->acquireReadWrite();
    ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);
    
    // compute loop invariant quantities
    Scalar exp_v_fac = exp(-Scalar(1.0/4.0)*(eta+xi)*m_deltaT);
    
    // perform second half step of NPT integration
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        
        // first, calculate acceleration from the net force
        Scalar minv = Scalar(1.0) / arrays.mass[j];
        arrays.ax[j] = h_net_force.data[j].x*minv;
        arrays.ay[j] = h_net_force.data[j].y*minv;
        arrays.az[j] = h_net_force.data[j].z*minv;
        
        // then, update the velocity
        arrays.vx[j] = arrays.vx[j]*exp_v_fac*exp_v_fac + Scalar(1.0/2.0)*m_deltaT*exp_v_fac*arrays.ax[j];
        arrays.vy[j] = arrays.vy[j]*exp_v_fac*exp_v_fac + Scalar(1.0/2.0)*m_deltaT*exp_v_fac*arrays.ay[j];
        arrays.vz[j] = arrays.vz[j]*exp_v_fac*exp_v_fac + Scalar(1.0/2.0)*m_deltaT*exp_v_fac*arrays.az[j];
        }
    
    // Update state variables
    Scalar N = Scalar(m_group->getNumMembers());
    eta += Scalar(1.0/2.0)/(m_tauP*m_tauP)*m_V/(N*m_T->getValue(timestep))
                            *(m_curr_P - m_P->getValue(timestep))*m_deltaT;
    xi += Scalar(1.0/2.0)/(m_tau*m_tau)*(m_curr_group_T/m_T->getValue(timestep) - Scalar(1.0))*m_deltaT;
    
    m_pdata->release();
    setIntegratorVariables(v);
        
    // done profiling
    if (m_prof)
        m_prof->pop();
    }


Scalar TwoStepNPT::computePressure(unsigned int timestep)
    {
    // grab access to the particle data
    const ParticleDataArraysConst arrays = m_pdata->acquireReadOnly();
    ArrayHandle<Scalar> h_virial(m_pdata->getNetVirial(), access_location::host, access_mode::read);
    
    // sum up the kinetic energy and virial of the whole system
    double ke_total = 0.0;
    double W = 0.0;
    for (unsigned int i=0; i < m_pdata->getN(); i++)
        {
        ke_total += (double)arrays.mass[i]*((double)arrays.vx[i] * (double)arrays.vx[i] + 
                                            (double)arrays.vy[i] * (double)arrays.vy[i] +
                                            (double)arrays.vz[i] * (double)arrays.vz[i]);
        
        W += h_virial.data[i];
        }
        
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
        
Scalar TwoStepNPT::computeGroupTemperature(unsigned int timestep)
    {
    // grab access to the particle data
    const ParticleDataArraysConst arrays = m_pdata->acquireReadOnly();
    
    // sum up the kinetic energy
    double ke_total = 0.0;
    unsigned int group_size = m_group->getNumMembers();
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        ke_total += (double)arrays.mass[j]*((double)arrays.vx[j] * (double)arrays.vx[j] + 
                                            (double)arrays.vy[j] * (double)arrays.vy[j] +
                                            (double)arrays.vz[j] * (double)arrays.vz[j]);
        }
        
    ke_total *= 0.5;
    
    // done!
    m_pdata->release();
    return 2.0 * ke_total / m_group_dof;
    }

void export_TwoStepNPT()
    {
    class_<TwoStepNPT, boost::shared_ptr<TwoStepNPT>, bases<IntegrationMethodTwoStep>, boost::noncopyable>
        ("TwoStepNPT", init< boost::shared_ptr<SystemDefinition>,
                       boost::shared_ptr<ParticleGroup>,
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

