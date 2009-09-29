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

/*! \file NPTUpdater.cc
    \brief Defines the NPTUpdater class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include "NPTUpdater.h"
#include <math.h>

using namespace std;

/*! \param sysdef System to update
    \param deltaT Time step to use
    \param tau Nose-Hoover period
    \param tauP barostat period
    \param T Temperature set point
    \param P Pressure set point
*/
NPTUpdater::NPTUpdater(boost::shared_ptr<SystemDefinition> sysdef,
					   Scalar deltaT,
					   Scalar tau,
					   Scalar tauP,
					   boost::shared_ptr<Variant> T, boost::shared_ptr<Variant> P) 
	: Integrator(sysdef, deltaT), m_tau(tau), m_tauP(tauP), m_T(T), m_P(P), m_accel_set(false)
    {
    if (m_tau <= 0.0)
        cout << "***Warning! tau set less than 0.0 in NPTUpdater" << endl;
    if (m_tauP <= 0.0)
        cout << "***Warning! tauP set less than 0.0 in NPTUpdater" << endl;
        
    m_Xi = 0.0;  // Initialize m_Xi and m_Eta
    m_Eta = 0.0;
    const BoxDim& box = m_pdata->getBox();
    // sanity check
    assert(box.xhi > box.xlo && box.yhi > box.ylo && box.zhi > box.zlo);
    
    // precalculate box lenghts
    m_Lx = box.xhi - box.xlo;
    m_Ly = box.yhi - box.ylo;
    m_Lz = box.zhi - box.zlo;
    
    m_V = m_Lx*m_Ly*m_Lz;   // volume
    
    // compute the current pressure and temperature on construction
    m_curr_T = computeTemperature(0);
    m_curr_P = computePressure(0);
    }

Scalar NPTUpdater::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (quantity == string("temperature"))
        {
        return m_curr_T;
        }
    else if (quantity == string("pressure"))
        {
        return m_curr_P;
        }
    else if (quantity == string("conserved_quantity"))
        {
        cout << "**Warning! conserved_quantity not known for NPT integration" << endl;
        return 0.0; // not implemented yet!
        }
    else
        {
        // pass it on up to the base class
        return Integrator::getLogValue(quantity, timestep);
        }
    }


/*! \param timestep Current time step of the simulation
*/
void NPTUpdater::update(unsigned int timestep)
    {
    assert(m_pdata);
    Scalar N = Scalar(m_pdata->getN());
    static bool gave_warning = false;
    
    if (m_forces.size() == 0 && !gave_warning)
        {
        cout << "***Warning! No forces defined in NPTUpdater, continuing anyways" << endl;
        gave_warning = true;
        }
        
    // if we haven't been called before, then the accelerations have not been set and we need to calculate them
    if (!m_accel_set)
        {
        m_accel_set = true;
        computeAccelerations(timestep, "NPT");
        m_curr_T = computeTemperature(timestep);  // Compute temperature and pressure for the first time step
        m_curr_P = computePressure(timestep);
        }
        
    if (m_prof)
        {
        m_prof->push("NPT");
        m_prof->push("Half-step 1");
        }
        
        
    // advance thermostat(m_Xi) half a time step
    
    m_Xi += Scalar(1.0/2.0)/(m_tau*m_tau)*(m_curr_T/m_T->getValue(timestep) - Scalar(1.0))*m_deltaT;
    
    // advance barostat (m_Eta) half time step
    
    m_Eta += Scalar(1.0/2.0)/(m_tauP*m_tauP)*m_V/(N*m_T->getValue(timestep))*(m_curr_P - m_P->getValue(timestep))*m_deltaT;
    
    
    // propagate velocites for a half time step
    // and move coordinates to new positions
    
    // access the particle data arrays
    ParticleDataArrays arrays = m_pdata->acquireReadWrite();
    assert(arrays.x != NULL && arrays.y != NULL && arrays.z != NULL);
    assert(arrays.vx != NULL && arrays.vy != NULL && arrays.vz != NULL);
    assert(arrays.ax != NULL && arrays.ay != NULL && arrays.az != NULL);
    
    Scalar exp_v_fac = exp(-Scalar(1.0/4.0)*(m_Eta+m_Xi)*m_deltaT);
    Scalar exp_r_fac = exp(Scalar(1.0/2.0)*m_Eta*m_deltaT);
    for (unsigned int j = 0; j < arrays.nparticles; j++)
        {
        arrays.vx[j] = arrays.vx[j]*exp_v_fac*exp_v_fac + Scalar(1.0/2.0)*m_deltaT*exp_v_fac*arrays.ax[j]; // update velocity
        arrays.x[j] = arrays.x[j]*exp_r_fac*exp_r_fac + arrays.vx[j]*exp_r_fac*m_deltaT;
        
        arrays.vy[j] = arrays.vy[j]*exp_v_fac*exp_v_fac + Scalar(1.0/2.0)*m_deltaT*exp_v_fac*arrays.ay[j]; // update velocity
        arrays.y[j] = arrays.y[j]*exp_r_fac*exp_r_fac + arrays.vy[j]*exp_r_fac*m_deltaT;
        
        arrays.vz[j] = arrays.vz[j]*exp_v_fac*exp_v_fac + Scalar(1.0/2.0)*m_deltaT*exp_v_fac*arrays.az[j]; // update velocity
        arrays.z[j] = arrays.z[j]*exp_r_fac*exp_r_fac + arrays.vz[j]*exp_r_fac*m_deltaT;
        }
        
    // advance volume
    
    m_V *= exp(Scalar(3.0)*m_Eta*m_deltaT);
    
    // get the scaling factor for the box (V_new/V_old)^(1/3)
    
    Scalar box_len_scale = exp(m_Eta*m_deltaT);
    m_Lx *= box_len_scale;
    m_Ly *= box_len_scale;
    m_Lz *= box_len_scale;
    
    // We aren't done yet! Need to fix the periodic boundary conditions
    // this implementation only works if the particles go a wee bit outside the box, which is all that should ever happen under normal circumstances
    
    for (unsigned int j = 0; j < arrays.nparticles; j++)
        {
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
        
        
    // release the particle data arrays so that they can be accessed to rescale box, add up the accelerations and
    // calculate new pressure and temeprature
    m_pdata->release();
    
    // rescale simulation box
    
    m_pdata->setBox(BoxDim(m_Lx, m_Ly, m_Lz));
    
    
    
    // functions that computeAccelerations calls profile themselves, so suspend
    // the profiling for now
    if (m_prof)
        {
        m_prof->pop();
        m_prof->pop();
        }
        
    // for the next half of the step, we need the accelerations at t+deltaT
    computeAccelerations(timestep+1, "NPT");
    // compute temperature for the next hlf time step
    m_curr_T = computeTemperature(timestep+1);
    // compute pressure for the next half time step
    m_curr_P = computePressure(timestep+1);
    
    
    if (m_prof)
        {
        m_prof->push("NPT");
        m_prof->push("Half-step 2");
        }
        
        
        
    // get the particle data arrays again so we can update the 2nd half of the step
    arrays = m_pdata->acquireReadWrite();
    exp_v_fac = exp(-Scalar(1.0/4.0)*(m_Eta+m_Xi)*m_deltaT);
    
    for (unsigned int j = 0; j < arrays.nparticles; j++)
        {
        arrays.vx[j] = arrays.vx[j]*exp_v_fac*exp_v_fac + Scalar(1.0/2.0)*m_deltaT*exp_v_fac*arrays.ax[j]; // update velocity
        
        arrays.vy[j] = arrays.vy[j]*exp_v_fac*exp_v_fac + Scalar(1.0/2.0)*m_deltaT*exp_v_fac*arrays.ay[j]; // update velocity
        arrays.vz[j] = arrays.vz[j]*exp_v_fac*exp_v_fac + Scalar(1.0/2.0)*m_deltaT*exp_v_fac*arrays.az[j]; // update velocity
        }
        
        
    // release data
    
    m_pdata->release();
    
    
    // Update m_Eta
    
    m_Eta += Scalar(1.0/2.0)/(m_tauP*m_tauP)*m_V/(N*m_T->getValue(timestep))*(m_curr_P - m_P->getValue(timestep))*m_deltaT;
    
    // Update m_Xi
    
    m_Xi += Scalar(1.0/2.0)/(m_tau*m_tau)*(m_curr_T/m_T->getValue(timestep) - Scalar(1.0))*m_deltaT;
    
    
    
    //and we are done!
    
    
    if (m_prof)
        {
        m_prof->pop();
        m_prof->pop();
        }
    }

void export_NPTUpdater()
    {
    class_<NPTUpdater, boost::shared_ptr<NPTUpdater>, bases<Integrator>, boost::noncopyable>
    ("NPTUpdater", init< boost::shared_ptr<SystemDefinition>, Scalar, Scalar, Scalar, boost::shared_ptr<Variant>, boost::shared_ptr<Variant> >())
    .def("setT", &NPTUpdater::setT)
    .def("setTau", &NPTUpdater::setTau)
    .def("setP", &NPTUpdater::setP)
    .def("setTauP", &NPTUpdater::setTauP)
    ;
    
    }

#ifdef WIN32
#pragma warning( pop )
#endif

