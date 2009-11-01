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

/*! \file NVTUpdater.cc
    \brief Defines the NVTUpdater class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include "NVTUpdater.h"
#include <math.h>

using namespace std;

/*! \param sysdef System to update
    \param deltaT Time step to use
    \param tau NVT period
    \param T Temperature set point
*/
NVTUpdater::NVTUpdater(boost::shared_ptr<SystemDefinition> sysdef,
                       Scalar deltaT,
                       Scalar tau,
                       boost::shared_ptr<Variant> T) 
    : Integrator(sysdef, deltaT), m_tau(tau), m_T(T), m_accel_set(false)
    {
    if (m_tau <= 0.0)
        cout << "***Warning! tau set less than 0.0 in NVTUpdater" << endl;
        
    m_curr_T = computeTemperature(0);
    m_dof = Scalar(3*m_pdata->getN() - 3);

    IntegratorVariables v = getIntegratorVariables();

   if (!restartInfoIsGood(v, "nvt", 2)) 
        {
        v.type = "nvt";
        v.variable.resize(2);
        Scalar xi = Scalar(1.0);
        Scalar eta = Scalar(1.0);
        v.variable[0] = xi;
        v.variable[1] = eta;
        }
    setIntegratorVariables(v);
    }

/*! In addition to what Integrator provides, NVTUpdater adds:
        - nvt_xi
        - nvt_eta

    See Logger for more information on what this is about.
*/
std::vector< std::string > NVTUpdater::getProvidedLogQuantities()
    {
    vector<string> result = Integrator::getProvidedLogQuantities();
    result.push_back("nvt_xi");
    result.push_back("nvt_eta");
    return result;
    }

/*! \param quantity Name of the quantity to log
    \param timestep Current time step of the simulation

    NVTUpdater calculates the conserved quantity as the sum of the kinetic and potential energies with the additional
    ficticious energy term. The temperature quantity returns the already computed temperature of the system.

    Working out the math on paper starting from eq. 11 in "The Nose-Poincare Method for Constant Temperature
    Molecular Dynamics" yields the conserved quantity:
    \f[ E_{\mathrm{ext}} = E + gkT_0 \left( \frac{\xi^2\tau^2}{2} + \eta \right) \f]

    All other quantity requests are passed up to Integrator::getLogValue().
*/
Scalar NVTUpdater::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (quantity == string("conserved_quantity"))
        {
        Scalar g = Scalar(3*m_pdata->getN());
        IntegratorVariables v = getIntegratorVariables();
        Scalar xi = v.variable[0];
        Scalar eta = v.variable[1];
        return computeKineticEnergy(timestep) + computePotentialEnergy(timestep) +
               g * m_T->getValue(timestep) * (xi*xi*m_tau*m_tau / Scalar(2.0) + eta);
        }
    else if (quantity == string("temperature"))
        {
        return m_curr_T;
        }
    else if (quantity == string("nvt_xi"))
        {
        Scalar chi = getIntegratorVariables().variable[0];
        return chi;
        }
    else if (quantity == string("nvt_eta"))
        {
        Scalar eta = getIntegratorVariables().variable[1];
        return eta;
        }
    else
        {
        // pass it on up to the base class
        return Integrator::getLogValue(quantity, timestep);
        }
    }

/*! \param timestep Current time step of the simulation
*/
void NVTUpdater::update(unsigned int timestep)
    {
    assert(m_pdata);
    static bool gave_warning = false;
    
    if (m_forces.size() == 0 && !gave_warning)
        {
        cout << "***Warning! No forces defined in NVTUpdater, continuing anyways" << endl;
        gave_warning = true;
        }
        
    // if we haven't been called before, then the accelerations have not been set and we need to calculate them
    if (!m_accel_set)
        {
        m_accel_set = true;
        computeAccelerations(timestep, "NVT");
        }
        
    if (m_prof)
        {
        m_prof->push("NVT");
        m_prof->push("Half-step 1");
        }
        
    // access the particle data arrays
    ParticleDataArrays arrays = m_pdata->acquireReadWrite();
    assert(arrays.x != NULL && arrays.y != NULL && arrays.z != NULL);
    assert(arrays.vx != NULL && arrays.vy != NULL && arrays.vz != NULL);
    assert(arrays.ax != NULL && arrays.ay != NULL && arrays.az != NULL);
    
    IntegratorVariables v = getIntegratorVariables();
    Scalar& xi = v.variable[0];
    Scalar& eta = v.variable[1];

    // now we can get on with the Nose-Hoover
    // first half step: update the velocities and positions
    // sum up K while we are doing the loop
    Scalar Ksum = 0.0;
    for (unsigned int j = 0; j < arrays.nparticles; j++)
        {
        Scalar denominv = Scalar(1.0) / (Scalar(1.0) + m_deltaT/Scalar(2.0) * xi);
        
        arrays.vx[j] = (arrays.vx[j] + Scalar(1.0/2.0)*arrays.ax[j]*m_deltaT) * denominv;
        arrays.x[j] += m_deltaT * arrays.vx[j];
        
        arrays.vy[j] = (arrays.vy[j] + Scalar(1.0/2.0)*arrays.ay[j]*m_deltaT) * denominv;
        arrays.y[j] += m_deltaT * arrays.vy[j];
        
        arrays.vz[j] = (arrays.vz[j] + Scalar(1.0/2.0)*arrays.az[j]*m_deltaT) * denominv;
        arrays.z[j] += m_deltaT * arrays.vz[j];
        
        Ksum += arrays.mass[j] * (arrays.vx[j]*arrays.vx[j] + arrays.vy[j]*arrays.vy[j] + arrays.vz[j]*arrays.vz[j]);
        }
        
    // need previous xi to update eta
    Scalar xi_prev = xi;
    
    // update Xi
    m_curr_T = Ksum / m_dof;
    xi += m_deltaT / (m_tau*m_tau) * (m_curr_T/m_T->getValue(timestep) - Scalar(1.0));
    
    // update eta
    eta += m_deltaT / Scalar(2.0) * (xi + xi_prev);
    
    // We aren't done yet! Need to fix the periodic boundary conditions
    // this implementation only works if the particles go a wee bit outside the box, which is all that should ever happen under normal circumstances
    // get a local copy of the simulation box
    const BoxDim& box = m_pdata->getBox();
    // sanity check
    assert(box.xhi > box.xlo && box.yhi > box.ylo && box.zhi > box.zlo);
    
    // precalculate box lenghts
    Scalar Lx = box.xhi - box.xlo;
    Scalar Ly = box.yhi - box.ylo;
    Scalar Lz = box.zhi - box.zlo;
    
    for (unsigned int j = 0; j < arrays.nparticles; j++)
        {
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
        
        
    // release the particle data arrays so that they can be accessed to add up the accelerations
    m_pdata->release();
    
    // functions that computeAccelerations calls profile themselves, so suspend
    // the profiling for now
    if (m_prof)
        {
        m_prof->pop();
        m_prof->pop();
        }
        
    // for the next half of the step, we need the accelerations at t+deltaT
    computeAccelerations(timestep+1, "NVT");
    
    if (m_prof)
        {
        m_prof->push("NVT");
        m_prof->push("Half-step 2");
        }
        
    // get the particle data arrays again so we can update the 2nd half of the step
    arrays = m_pdata->acquireReadWrite();
    
    // v(t+deltaT) = v(t+deltaT/2) + 1/2 * a(t+deltaT)*deltaT
    for (unsigned int j = 0; j < arrays.nparticles; j++)
        {
        arrays.vx[j] += Scalar(1.0/2.0) * m_deltaT * (arrays.ax[j] - xi * arrays.vx[j]);
        arrays.vy[j] += Scalar(1.0/2.0) * m_deltaT * (arrays.ay[j] - xi * arrays.vy[j]);
        arrays.vz[j] += Scalar(1.0/2.0) * m_deltaT * (arrays.az[j] - xi * arrays.vz[j]);
        }
        
    m_pdata->release();
    setIntegratorVariables(v);
    
    // and now the acceleration at timestep+1 is precalculated for the first half of the next step
    if (m_prof)
        {
        m_prof->pop();
        m_prof->pop();
        }
    }

void export_NVTUpdater()
    {
    class_<NVTUpdater, boost::shared_ptr<NVTUpdater>, bases<Integrator>, boost::noncopyable>
    ("NVTUpdater", init< boost::shared_ptr<SystemDefinition>, Scalar, Scalar, boost::shared_ptr<Variant> >())
    .def("setT", &NVTUpdater::setT)
    .def("setTau", &NVTUpdater::setTau)
    ;
    
    }

#ifdef WIN32
#pragma warning( pop )
#endif

