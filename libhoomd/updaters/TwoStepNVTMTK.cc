/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2014 The Regents of
the University of Michigan All rights reserved.

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

#include "TwoStepNVTMTK.h"

#ifdef ENABLE_MPI
#include "Communicator.h"
#include "HOOMDMPI.h"
#endif

/*! \file TwoStepNVTMTK.h
    \brief Contains code for the TwoStepNVTMTK class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param thermo compute for thermodynamic quantities
    \param tau NVT period
    \param T Temperature set point
    \param suffix Suffix to attach to the end of log quantity names
*/
TwoStepNVTMTK::TwoStepNVTMTK(boost::shared_ptr<SystemDefinition> sysdef,
                       boost::shared_ptr<ParticleGroup> group,
                       boost::shared_ptr<ComputeThermo> thermo,
                       Scalar tau,
                       boost::shared_ptr<Variant> T,
                       const std::string& suffix)
    : IntegrationMethodTwoStep(sysdef, group), m_thermo(thermo), m_tau(tau), m_T(T), m_exp_thermo_fac(0.0), m_curr_T(0.0)
    {
    m_exec_conf->msg->notice(5) << "Constructing TwoStepNVTMTK" << endl;

    if (m_tau <= 0.0)
        m_exec_conf->msg->warning() << "integrate.nvt: tau set less than 0.0 in NVTUpdater" << endl;

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
    m_log_name = string("nvt_mtk_reservoir_energy") + suffix;
    }

TwoStepNVTMTK::~TwoStepNVTMTK()
    {
    m_exec_conf->msg->notice(5) << "Destroying TwoStepNVTMTK" << endl;
    }

/*! Returns a list of log quantities this compute calculates
*/
std::vector< std::string > TwoStepNVTMTK::getProvidedLogQuantities()
    {
    vector<string> result;
    result.push_back(m_log_name);
    return result;
    }

/*! \param quantity Name of the log quantity to get
    \param timestep Current time step of the simulation
    \param my_quantity_flag passed as false, changed to true if quanity logged here
*/

Scalar TwoStepNVTMTK::getLogValue(const std::string& quantity, unsigned int timestep, bool &my_quantity_flag)
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
void TwoStepNVTMTK::integrateStepOne(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();
    if (group_size == 0)
        return;

    // profile this step
    if (m_prof)
        m_prof->push("NVT step 1");

    // compute the current thermodynamic properties
    m_thermo->compute(timestep);

    // compute temperature for the next half time step
    m_curr_T = m_thermo->getTemperature();

    // advance thermostat
    advanceThermostat(timestep, false);

    // scope array handles for proper releasing before calling the thermo compute
    {
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);

    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);

        // load variables
        Scalar3 v = make_scalar3(h_vel.data[j].x, h_vel.data[j].y, h_vel.data[j].z);
        Scalar3 pos = make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);
        Scalar3 accel = h_accel.data[j];

        // update velocity and position
        v = m_exp_thermo_fac*v + Scalar(1.0/2.0)*accel*m_deltaT;
        pos += m_deltaT * v;

        // store updated variables
        h_vel.data[j].x = v.x;
        h_vel.data[j].y = v.y;
        h_vel.data[j].z = v.z;

        h_pos.data[j].x = pos.x;
        h_pos.data[j].y = pos.y;
        h_pos.data[j].z = pos.z;
        }

    // particles may have been moved slightly outside the box by the above steps, wrap them back into place
    const BoxDim& box = m_pdata->getBox();

    ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);

    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        // wrap the particles around the box
        box.wrap(h_pos.data[j], h_image.data[j]);
        }
    }

    // done profiling
    if (m_prof)
        m_prof->pop();
    }

/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1
*/
void TwoStepNVTMTK::integrateStepTwo(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();

    const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();

    // profile this step
    if (m_prof)
        m_prof->push("NVT step 2");

    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::readwrite);

    ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);

    // perform second half step of Nose-Hoover integration

    // Kinetic energy * 2
    Scalar v2_sum(0.0);

    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);

        Scalar3 v = make_scalar3(h_vel.data[j].x, h_vel.data[j].y, h_vel.data[j].z);
        Scalar3 accel = h_accel.data[j];
        Scalar3 net_force = make_scalar3(h_net_force.data[j].x,h_net_force.data[j].y,h_net_force.data[j].z);

        // first, calculate acceleration from the net force
        Scalar m = h_vel.data[j].w;
        Scalar minv = Scalar(1.0) / m;
        accel = net_force*minv;

        // then, update the velocity
        v += Scalar(1.0/2.0) * m_deltaT * accel;

        // store velocity
        h_vel.data[j].x = v.x;
        h_vel.data[j].y = v.y;
        h_vel.data[j].z = v.z;

        // store acceleration
        h_accel.data[j] = accel;

        // reduce 2*kinetic energy
        v2_sum += m*dot(v,v);
        }

    #ifdef ENABLE_MPI
    if (m_comm)
        {
        MPI_Allreduce(MPI_IN_PLACE, &v2_sum, 1, MPI_HOOMD_SCALAR, MPI_SUM, m_exec_conf->getMPICommunicator() );
        }
    #endif

    m_curr_T = v2_sum/m_thermo->getNDOF();

    // get temperature and advance thermostat
    advanceThermostat(timestep+1,true);

    // apply the thermostat rescaling
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);

        // load velocity
        Scalar3 v = make_scalar3(h_vel.data[j].x, h_vel.data[j].y, h_vel.data[j].z);

        // rescale
        v *= m_exp_thermo_fac;

        // store velocity
        h_vel.data[j].x = v.x;
        h_vel.data[j].y = v.y;
        h_vel.data[j].z = v.z;
        }

    // done profiling
    if (m_prof)
        m_prof->pop();
    }

void TwoStepNVTMTK::advanceThermostat(unsigned int timestep, bool broadcast)
    {
    IntegratorVariables v = getIntegratorVariables();
    Scalar& xi = v.variable[0];
    Scalar& eta = v.variable[1];

    // update the state variables Xi and eta
    Scalar xi_prime = xi + Scalar(1.0/4.0)*m_deltaT/m_tau/m_tau*(m_curr_T/m_T->getValue(timestep) - Scalar(1.0));
    xi = xi_prime+ Scalar(1.0/4.0)*m_deltaT/(m_tau*m_tau)*(m_curr_T/m_T->getValue(timestep)*
              exp(-xi_prime*m_deltaT) - Scalar(1.0));
    eta += Scalar(1.0/2.0)*xi_prime*m_deltaT;

    // update loop-invariant quantity
    m_exp_thermo_fac = exp(-Scalar(1.0/2.0)*xi_prime*m_deltaT);

    #ifdef ENABLE_MPI
    if (m_comm && broadcast)
        {
        // broadcast integrator variables from rank 0 to other processors
        MPI_Bcast(&xi, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        MPI_Bcast(&eta, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        }
    #endif

    setIntegratorVariables(v);
    }

void export_TwoStepNVTMTK()
    {
    class_<TwoStepNVTMTK, boost::shared_ptr<TwoStepNVTMTK>, bases<IntegrationMethodTwoStep>, boost::noncopyable>
            ("TwoStepNVTMTK", init< boost::shared_ptr<SystemDefinition>,
                       boost::shared_ptr<ParticleGroup>,
                       boost::shared_ptr<ComputeThermo>,
                       Scalar,
                       boost::shared_ptr<Variant>,
                       const std::string&
                       >())
        .def("setT", &TwoStepNVTMTK::setT)
        .def("setTau", &TwoStepNVTMTK::setTau)
        ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif
