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

#include "TwoStepBDNVT.h"

#ifdef ENABLE_MPI
#include "HOOMDMPI.h"
#endif

/*! \file TwoStepBDNVT.h
    \brief Contains code for the TwoStepBDNVT class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param T Temperature set point as a function of time
    \param seed Random seed to use in generating random numbers
    \param gamma_diam Set gamma to the particle diameter of each particle if true, otherwise use a per-type
                      gamma via setGamma()
    \param suffix Suffix to attach to the end of log quantity names
*/
TwoStepBDNVT::TwoStepBDNVT(boost::shared_ptr<SystemDefinition> sysdef,
                           boost::shared_ptr<ParticleGroup> group,
                           boost::shared_ptr<Variant> T,
                           unsigned int seed,
                           bool gamma_diam,
                           const std::string& suffix)
    : TwoStepNVE(sysdef, group, true), m_T(T), m_seed(seed), m_gamma_diam(gamma_diam), m_reservoir_energy(0),  m_extra_energy_overdeltaT(0), m_tally(false)
    {
    m_exec_conf->msg->notice(5) << "Constructing TwoStepBDNVT" << endl;

    // Hash the User's Seed to make it less likely to be a low positive integer
    m_seed = m_seed*0x12345677 + 0x12345 ; m_seed^=(m_seed>>16); m_seed*= 0x45679;

    // set a named, but otherwise blank set of integrator variables
    IntegratorVariables v = getIntegratorVariables();

    if (!restartInfoTestValid(v, "bdnvt", 0))
        {
        v.type = "bdnvt";
        v.variable.resize(0);
        setValidRestart(false);
        }
    else
        setValidRestart(true);

    setIntegratorVariables(v);

    // allocate memory for the per-type gamma storage and initialize them to 1.0
    GPUArray<Scalar> gamma(m_pdata->getNTypes(), exec_conf);
    m_gamma.swap(gamma);
    ArrayHandle<Scalar> h_gamma(m_gamma, access_location::host, access_mode::overwrite);
    for (unsigned int i = 0; i < m_gamma.getNumElements(); i++)
        h_gamma.data[i] = Scalar(1.0);

    m_log_name = string("bdnvt_reservoir_energy") + suffix;
    }

TwoStepBDNVT::~TwoStepBDNVT()
    {
    m_exec_conf->msg->notice(5) << "Destroying TwoStepBDNVT" << endl;
    }

/*! \param typ Particle type to set gamma for
    \param gamma The gamma value to set
*/
void TwoStepBDNVT::setGamma(unsigned int typ, Scalar gamma)
    {
    // check for user errors
    if (m_gamma_diam)
        {
        m_exec_conf->msg->error() << "intergae.bdnvt: Trying to set gamma when it is set to be the diameter! " << typ << endl;
        throw runtime_error("Error setting params in TwoStepBDNVT");
        }
    if (typ >= m_pdata->getNTypes())
        {
        m_exec_conf->msg->error() << "intergae.bdnvt: Trying to set gamma for a non existant type! " << typ << endl;
        throw runtime_error("Error setting params in TwoStepBDNVT");
        }

    ArrayHandle<Scalar> h_gamma(m_gamma, access_location::host, access_mode::readwrite);
    h_gamma.data[typ] = gamma;
    }

/*! Returns a list of log quantities this compute calculates
*/
std::vector< std::string > TwoStepBDNVT::getProvidedLogQuantities()
    {
    vector<string> result;
    if (m_tally)
        result.push_back(m_log_name);
    return result;
    }

/*! \param quantity Name of the log quantity to get
    \param timestep Current time step of the simulation
    \param my_quantity_flag passed as false, changed to true if quanity logged here
*/

Scalar TwoStepBDNVT::getLogValue(const std::string& quantity, unsigned int timestep, bool &my_quantity_flag)
    {
    if (m_tally && quantity == m_log_name)
        {
        my_quantity_flag = true;
        return m_reservoir_energy+m_extra_energy_overdeltaT*m_deltaT;
        }
    else
        return Scalar(0);
    }

/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1
*/
void TwoStepBDNVT::integrateStepTwo(unsigned int timestep)
    {
#ifdef ENABLE_MPI
    unsigned int group_size = m_group->getNumMembers();
#else
    unsigned int group_size = m_group->getNumMembers();
#endif
    if (group_size == 0)
        return;

    const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();


    // profile this step
    if (m_prof)
        m_prof->push("NVE step 2");

    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::readwrite);

    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_gamma(m_gamma, access_location::host, access_mode::read);

    // grab some initial variables
    const Scalar currentTemp = m_T->getValue(timestep);
    const Scalar D = Scalar(m_sysdef->getNDimensions());

    // initialize the RNG
    Saru saru(m_seed, timestep);

    // energy transferred over this time step
    Scalar bd_energy_transfer = 0;

    // a(t+deltaT) gets modified with the bd forces
    // v(t+deltaT) = v(t+deltaT/2) + 1/2 * a(t+deltaT)*deltaT
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);

        // first, calculate the BD forces
        // Generate three random numbers
        Scalar rx = saru.d(-1,1);
        Scalar ry = saru.d(-1,1);
        Scalar rz =  saru.d(-1,1);

        Scalar gamma;
        if (m_gamma_diam)
            gamma = h_diameter.data[j];
        else
            {
            unsigned int type = __scalar_as_int(h_pos.data[j].w);
            gamma = h_gamma.data[type];
            }

        // compute the bd force
        Scalar coeff = sqrt(Scalar(6.0) *gamma*currentTemp/m_deltaT);
        Scalar bd_fx = rx*coeff - gamma*h_vel.data[j].x;
        Scalar bd_fy = ry*coeff - gamma*h_vel.data[j].y;
        Scalar bd_fz = rz*coeff - gamma*h_vel.data[j].z;

        if (D < 3.0)
            bd_fz = Scalar(0.0);

        // then, calculate acceleration from the net force
        Scalar minv = Scalar(1.0) / h_vel.data[j].w;
        h_accel.data[j].x = (h_net_force.data[j].x + bd_fx)*minv;
        h_accel.data[j].y = (h_net_force.data[j].y + bd_fy)*minv;
        h_accel.data[j].z = (h_net_force.data[j].z + bd_fz)*minv;

        // then, update the velocity
        h_vel.data[j].x += Scalar(1.0/2.0)*h_accel.data[j].x*m_deltaT;
        h_vel.data[j].y += Scalar(1.0/2.0)*h_accel.data[j].y*m_deltaT;
        h_vel.data[j].z += Scalar(1.0/2.0)*h_accel.data[j].z*m_deltaT;

        // tally the energy transfer from the bd thermal reservor to the particles
        if (m_tally) bd_energy_transfer += bd_fx * h_vel.data[j].x + bd_fy * h_vel.data[j].y + bd_fz * h_vel.data[j].z;

        // limit the movement of the particles
        if (m_limit)
            {
            Scalar vel = sqrt(h_vel.data[j].x*h_vel.data[j].x + h_vel.data[j].y*h_vel.data[j].y + h_vel.data[j].z*h_vel.data[j].z );
            if ( (vel*m_deltaT) > m_limit_val)
                {
                h_vel.data[j].x = h_vel.data[j].x / vel * m_limit_val / m_deltaT;
                h_vel.data[j].y = h_vel.data[j].y / vel * m_limit_val / m_deltaT;
                h_vel.data[j].z = h_vel.data[j].z / vel * m_limit_val / m_deltaT;
                }
            }
        }

    // update energy reservoir
    if (m_tally) {
        #ifdef ENABLE_MPI
        if (m_comm)
            {
            MPI_Allreduce(MPI_IN_PLACE, &bd_energy_transfer, 1, MPI_HOOMD_SCALAR, MPI_SUM, m_exec_conf->getMPICommunicator());
            }
        #endif
        m_reservoir_energy -= bd_energy_transfer*m_deltaT;
        m_extra_energy_overdeltaT = 0.5*bd_energy_transfer;

       }

    // done profiling
    if (m_prof)
        m_prof->pop();
    }

void export_TwoStepBDNVT()
    {
    class_<TwoStepBDNVT, boost::shared_ptr<TwoStepBDNVT>, bases<TwoStepNVE>, boost::noncopyable>
        ("TwoStepBDNVT", init< boost::shared_ptr<SystemDefinition>,
                         boost::shared_ptr<ParticleGroup>,
                         boost::shared_ptr<Variant>,
                         unsigned int,
                         bool,
                         const std::string&
                         >())
        .def("setT", &TwoStepBDNVT::setT)
        .def("setGamma", &TwoStepBDNVT::setGamma)
        .def("setTally", &TwoStepBDNVT::setTally)
        ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif
