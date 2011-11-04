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

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include "TwoStepNVT.h"

/*! \file TwoStepNVT.h
    \brief Contains code for the TwoStepNVT class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param thermo compute for thermodynamic quantities
    \param tau NVT period
    \param T Temperature set point
    \param suffix Suffix to attach to the end of log quantity names
*/
TwoStepNVT::TwoStepNVT(boost::shared_ptr<SystemDefinition> sysdef,
                       boost::shared_ptr<ParticleGroup> group,
                       boost::shared_ptr<ComputeThermo> thermo,
                       Scalar tau,
                       boost::shared_ptr<Variant> T,
                       const std::string& suffix)
    : IntegrationMethodTwoStep(sysdef, group), m_thermo(thermo), m_tau(tau), m_T(T)
    {
    if (m_tau <= 0.0)
        cout << "***Warning! tau set less than 0.0 in NVTUpdater" << endl;
    
    // set initial state
    IntegratorVariables v = getIntegratorVariables();

    if (!restartInfoTestValid(v, "nvt", 2))
        {
        v.type = "nvt";
        v.variable.resize(2);
        v.variable[0] = Scalar(0.0);
        v.variable[1] = Scalar(0.0);
        setValidRestart(false);
        }
    else
        setValidRestart(true);

    setIntegratorVariables(v);
    m_log_name = string("nvt_reservoir_energy") + suffix;
    }

/*! Returns a list of log quantities this compute calculates
*/
std::vector< std::string > TwoStepNVT::getProvidedLogQuantities()
    {
    vector<string> result;
    result.push_back(m_log_name);
    return result;
    }

/*! \param quantity Name of the log quantity to get
    \param timestep Current time step of the simulation
    \param my_quantity_flag passed as false, changed to true if quanity logged here
*/

Scalar TwoStepNVT::getLogValue(const std::string& quantity, unsigned int timestep, bool &my_quantity_flag)
    {
    if (quantity == m_log_name)  
        {
        my_quantity_flag = true;
        Scalar g = m_thermo->getNDOF();
        IntegratorVariables v = getIntegratorVariables();
        Scalar& xi = v.variable[0];
        Scalar& eta = v.variable[1];
        return g * m_T->getValue(timestep) * (xi*xi*m_tau*m_tau / Scalar(2.0) + eta);
        }
    else
        return Scalar(0);     
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per the velocity verlet
          method.
*/
void TwoStepNVT::integrateStepOne(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();
    if (group_size == 0)
        return;
    
    // profile this step
    if (m_prof)
        m_prof->push("NVT step 1");

    IntegratorVariables v = getIntegratorVariables();
    Scalar& xi = v.variable[0];

    const ParticleDataArrays& arrays = m_pdata->acquireReadWrite();
    
    // precompute loop invariant quantities
    Scalar denominv = Scalar(1.0) / (Scalar(1.0) + m_deltaT/Scalar(2.0) * xi);
    
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        
        arrays.vx[j] = (arrays.vx[j] + Scalar(1.0/2.0)*arrays.ax[j]*m_deltaT) * denominv;
        arrays.x[j] += m_deltaT * arrays.vx[j];
        
        arrays.vy[j] = (arrays.vy[j] + Scalar(1.0/2.0)*arrays.ay[j]*m_deltaT) * denominv;
        arrays.y[j] += m_deltaT * arrays.vy[j];
        
        arrays.vz[j] = (arrays.vz[j] + Scalar(1.0/2.0)*arrays.az[j]*m_deltaT) * denominv;
        arrays.z[j] += m_deltaT * arrays.vz[j];
        }
    
    // particles may have been moved slightly outside the box by the above steps, wrap them back into place
    const BoxDim& box = m_pdata->getBox();
    
    // precalculate box lenghts
    Scalar Lx = box.xhi - box.xlo;
    Scalar Ly = box.yhi - box.ylo;
    Scalar Lz = box.zhi - box.zlo;
    
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        // wrap the particle around the box
        if (arrays.x[j] >= box.xhi)
            {
            arrays.x[j] -= Lx;
            arrays.ix[j]++;
            }
        else if (arrays.x[j] < box.xlo)
            {
            arrays.x[j] += Lx;
            arrays.ix[j]--;
            }
            
        if (arrays.y[j] >= box.yhi)
            {
            arrays.y[j] -= Ly;
            arrays.iy[j]++;
            }
        else if (arrays.y[j] < box.ylo)
            {
            arrays.y[j] += Ly;
            arrays.iy[j]--;
            }
            
        if (arrays.z[j] >= box.zhi)
            {
            arrays.z[j] -= Lz;
            arrays.iz[j]++;
            }
        else if (arrays.z[j] < box.zlo)
            {
            arrays.z[j] += Lz;
            arrays.iz[j]--;
            }
        }
    
    m_pdata->release();
    
    // done profiling
    if (m_prof)
        m_prof->pop();
    }
        
/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1
*/
void TwoStepNVT::integrateStepTwo(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();
    if (group_size == 0)
        return;
    
    IntegratorVariables v = getIntegratorVariables();
    Scalar& xi = v.variable[0];
    Scalar& eta = v.variable[1];
    
    // compute the current thermodynamic properties
    m_thermo->compute(timestep+1);
    
    // next, update the state variables Xi and eta
    Scalar xi_prev = xi;
    Scalar curr_T = m_thermo->getTemperature();
    xi += m_deltaT / (m_tau*m_tau) * (curr_T/m_T->getValue(timestep) - Scalar(1.0));
    eta += m_deltaT / Scalar(2.0) * (xi + xi_prev);
    
    const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();
    
    // profile this step
    if (m_prof)
        m_prof->push("NVT step 2");
    
    const ParticleDataArrays& arrays = m_pdata->acquireReadWrite();
    ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);
    
    // perform second half step of Nose-Hoover integration
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        
        // first, calculate acceleration from the net force
        Scalar minv = Scalar(1.0) / arrays.mass[j];
        arrays.ax[j] = h_net_force.data[j].x*minv;
        arrays.ay[j] = h_net_force.data[j].y*minv;
        arrays.az[j] = h_net_force.data[j].z*minv;
        
        // then, update the velocity
        arrays.vx[j] += Scalar(1.0/2.0) * m_deltaT * (arrays.ax[j] - xi * arrays.vx[j]);
        arrays.vy[j] += Scalar(1.0/2.0) * m_deltaT * (arrays.ay[j] - xi * arrays.vy[j]);
        arrays.vz[j] += Scalar(1.0/2.0) * m_deltaT * (arrays.az[j] - xi * arrays.vz[j]);
        }
    
    m_pdata->release();
    setIntegratorVariables(v);

    
    // done profiling
    if (m_prof)
        m_prof->pop();
    }

void export_TwoStepNVT()
    {
    class_<TwoStepNVT, boost::shared_ptr<TwoStepNVT>, bases<IntegrationMethodTwoStep>, boost::noncopyable>
            ("TwoStepNVT", init< boost::shared_ptr<SystemDefinition>,
                       boost::shared_ptr<ParticleGroup>,
                       boost::shared_ptr<ComputeThermo>,
                       Scalar,
                       boost::shared_ptr<Variant>,
                       const std::string&
                       >())
        .def("setT", &TwoStepNVT::setT)
        .def("setTau", &TwoStepNVT::setTau)
        ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

