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

#include <boost/python.hpp>
using namespace boost::python;

#include "TwoStepBerendsen.h"
#ifdef ENABLE_CUDA
#include "TwoStepBerendsenGPU.cuh"
#endif

/*! \file TwoStepBerendsen.cc
    \brief Definition of Berendsen thermostat
*/

// ********************************
// here follows the code for Berendsen on the CPU

/*! \param sysdef System to zero the velocities of
    \param group Group of particles on which this method will act
    \param thermo compute for thermodynamic quantities
    \param tau Berendsen time constant
    \param T Temperature set point
*/
TwoStepBerendsen::TwoStepBerendsen(boost::shared_ptr<SystemDefinition> sysdef,
                                   boost::shared_ptr<ParticleGroup> group,
                                   boost::shared_ptr<ComputeThermo> thermo,
                                   Scalar tau,
                                   boost::shared_ptr<Variant> T)
    : IntegrationMethodTwoStep(sysdef, group), m_thermo(thermo), m_tau(tau), m_T(T)
    {
    if (m_tau <= 0.0)
        cout << "***Warning! tau set less than 0.0 in Berendsen thermostat" << endl;
    }


/*! Perform the needed calculations to zero the system's velocity
    \param timestep Current time step of the simulation
*/
void TwoStepBerendsen::integrateStepOne(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();
    if (group_size == 0)
        return;

    // profile this step
    if (m_prof)
        m_prof->push("Berendsen step 1");

    // compute the current thermodynamic properties and get the temperature
    m_thermo->compute(timestep);
    Scalar curr_T = m_thermo->getTemperature();

    // compute the value of lambda for the current timestep
    Scalar lambda = sqrt(Scalar(1.0) + m_deltaT / m_tau * (m_T->getValue(timestep) / curr_T - Scalar(1.0)));

    // access the particle data for writing on the CPU
    assert(m_pdata);
    const ParticleDataArrays& arrays = m_pdata->acquireReadWrite();

    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);

        // advance velocity forward by half a timestep and position forward by a full timestep
        arrays.vx[j] = lambda * (arrays.vx[j] + arrays.ax[j] * m_deltaT * Scalar(1.0 / 2.0));
        arrays.x[j] += arrays.vx[j] * m_deltaT;

        arrays.vy[j] = lambda * (arrays.vy[j] + arrays.ay[j] * m_deltaT * Scalar(1.0 / 2.0));
        arrays.y[j] += arrays.vy[j] * m_deltaT;

        arrays.vz[j] = lambda * (arrays.vz[j] + arrays.az[j] * m_deltaT * Scalar(1.0 / 2.0));
        arrays.z[j] += arrays.vz[j] * m_deltaT;
        }

    /* particles may have been moved slightly outside the box by the above steps so we should wrap
        them back into place */
    const BoxDim& box = m_pdata->getBox();

    // precalculate box lengths
    Scalar Lx = box.xhi - box.xlo;
    Scalar Ly = box.yhi - box.ylo;
    Scalar Lz = box.zhi - box.zlo;

    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);

        // wrap the particles around the box
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

    if (m_prof)
        m_prof->pop();
    }

/*! \param timestep Current timestep
    \post particle velocities are moved forward to timestep+1
*/
void TwoStepBerendsen::integrateStepTwo(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();
    if (group_size == 0)
        return;

    // access the particle data for writing on the CPU
    assert(m_pdata);
    const ParticleDataArrays& arrays = m_pdata->acquireReadWrite();

    // access the force data
    const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();
    ArrayHandle< Scalar4 > h_net_force(net_force, access_location::host, access_mode::read);

    // profile this step
    if (m_prof)
        m_prof->push("Berendsen step 2");

    // integrate the particle velocities to timestep+1
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);

        // calculate the acceleration from the net force
        Scalar minv = Scalar(1.0) / arrays.mass[j];
        arrays.ax[j] = h_net_force.data[j].x * minv;
        arrays.ay[j] = h_net_force.data[j].y * minv;
        arrays.az[j] = h_net_force.data[j].z * minv;

        // update the velocity
        arrays.vx[j] += arrays.ax[j] * m_deltaT / Scalar(2.0);
        arrays.vy[j] += arrays.ay[j] * m_deltaT / Scalar(2.0);
        arrays.vz[j] += arrays.az[j] * m_deltaT / Scalar(2.0);
        }

    // release the particle data
    m_pdata->release();
    }

void export_Berendsen()
    {
    class_<TwoStepBerendsen, boost::shared_ptr<TwoStepBerendsen>, bases<IntegrationMethodTwoStep>, boost::noncopyable>
    ("TwoStepBerendsen", init< boost::shared_ptr<SystemDefinition>,
                         boost::shared_ptr<ParticleGroup>,
                         boost::shared_ptr<ComputeThermo>,
                         Scalar,
                         boost::shared_ptr<Variant>
                         >())
        .def("setT", &TwoStepBerendsen::setT)
        .def("setTau", &TwoStepBerendsen::setTau)
        ;
    }

