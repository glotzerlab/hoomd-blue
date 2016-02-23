/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2015 The Regents of
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

#include "TwoStepLangevin.h"
#include "saruprng.h"

#ifdef ENABLE_MPI
#include "HOOMDMPI.h"
#endif

#include <boost/python.hpp>
using namespace boost::python;
using namespace std;

/*! \file TwoStepLangevin.h
    \brief Contains code for the TwoStepLangevin class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param T Temperature set point as a function of time
    \param seed Random seed to use in generating random numbers
    \param use_lambda If true, gamma=lambda*diameter, otherwise use a per-type gamma via setGamma()
    \param lambda Scale factor to convert diameter to gamma
    \param suffix Suffix to attach to the end of log quantity names
*/
TwoStepLangevin::TwoStepLangevin(boost::shared_ptr<SystemDefinition> sysdef,
                           boost::shared_ptr<ParticleGroup> group,
                           boost::shared_ptr<Variant> T,
                           unsigned int seed,
                           bool use_lambda,
                           Scalar lambda,
                           const std::string& suffix,                     
                           bool noiseless_t,
                           bool noiseless_r)
    : TwoStepLangevinBase(sysdef, group, T, seed, use_lambda, lambda), m_reservoir_energy(0),  m_extra_energy_overdeltaT(0),
      m_tally(false), m_noiseless_t(noiseless_t), m_noiseless_r(noiseless_r)
    {
    m_exec_conf->msg->notice(5) << "Constructing TwoStepLangevin" << endl;

    m_log_name = string("langevin_reservoir_energy") + suffix;
    }

TwoStepLangevin::~TwoStepLangevin()
    {
    m_exec_conf->msg->notice(5) << "Destroying TwoStepLangevin" << endl;
    }

/*! Returns a list of log quantities this compute calculates
*/
std::vector< std::string > TwoStepLangevin::getProvidedLogQuantities()
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

Scalar TwoStepLangevin::getLogValue(const std::string& quantity, unsigned int timestep, bool &my_quantity_flag)
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
    \post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per the velocity verlet
          method.
*/
void TwoStepLangevin::integrateStepOne(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();

    // profile this step
    if (m_prof)
        m_prof->push("Langevin step 1");

    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);

    const unsigned int D = Scalar(m_sysdef->getNDimensions());
    ArrayHandle<Scalar> h_gamma_r(m_gamma_r, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_torque(m_pdata->getNetTorqueArray(), access_location::host, access_mode::readwrite);
    
    const BoxDim& box = m_pdata->getBox();

    // perform the first half step of velocity verlet
    // r(t+deltaT) = r(t) + v(t)*deltaT + (1/2)a(t)*deltaT^2
    // v(t+deltaT/2) = v(t) + (1/2)a*deltaT
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);

        Scalar dx = h_vel.data[j].x*m_deltaT + Scalar(1.0/2.0)*h_accel.data[j].x*m_deltaT*m_deltaT;
        Scalar dy = h_vel.data[j].y*m_deltaT + Scalar(1.0/2.0)*h_accel.data[j].y*m_deltaT*m_deltaT;
        Scalar dz = h_vel.data[j].z*m_deltaT + Scalar(1.0/2.0)*h_accel.data[j].z*m_deltaT*m_deltaT;

        h_pos.data[j].x += dx;
        h_pos.data[j].y += dy;
        h_pos.data[j].z += dz;
        // particles may have been moved slightly outside the box by the above steps, wrap them back into place
        box.wrap(h_pos.data[j], h_image.data[j]);

        h_vel.data[j].x += Scalar(1.0/2.0)*h_accel.data[j].x*m_deltaT;
        h_vel.data[j].y += Scalar(1.0/2.0)*h_accel.data[j].y*m_deltaT;
        h_vel.data[j].z += Scalar(1.0/2.0)*h_accel.data[j].z*m_deltaT;
        
        // if (D < 3 && m_aniso)
        // {
        //     unsigned int type_r = __scalar_as_int(h_pos.data[j].w);
        //     Scalar gamma_r = h_gamma_r.data[type_r];
            
        //     if (gamma_r)
        //         {
        //         // original Gaussian random torque
        //         Scalar sigma_r = fast::sqrt(Scalar(2.0)*gamma_r*currentTemp/m_deltaT);
        //         Scalar tau_r = gaussian_rng(saru, sigma_r); 
        //         if (m_noiseless_r) tau_r = 0.0;
                
                
        //         vec3<Scalar> axis (0.0, 0.0, 1.0);
        //         Scalar a = (h_torque.data[j].z + tau_r) / gamma_r;
        //         // quat<Scalar> omega = quat<Scalar>::fromAxisAngle(axis, theta);
        //         quat<Scalar> omega (make_scalar4(0,0,0, theta));
        //         quat<Scalar> q (h_orientation.data[j]);
        //         q += Scalar(0.5) * m_deltaT * omega * q ;               
                
        //         // re-normalize (improves stability)
        //         q = q*(Scalar(1.0)/slow::sqrt(norm2(q)));
        //         h_orientation.data[j] = quat_to_scalar4(q);
        //         }
        //     }
                // {
                // Scalar rrx = saru.d(-1,1);
                // Scalar rry = saru.d(-1,1);
                // Scalar rrz = saru.d(-1,1);
                // Scalar coeff_r = sqrt(Scalar(6.0)*h_gamma_r*currentTemp/m_deltaT);
                // if (m_noiseless_r)  coeff_r = Scalar(0.0);
                
                // torque_handle.data[body].x -= h_gamma_r * angvel_handle.data[body].x;
                // torque_handle.data[body].y -= h_gamma_r * angvel_handle.data[body].y;
                // // torque_handle.data[body].x += coeff_r * rrx - h_gamma_r * angvel_handle.data[body].x;
                // // torque_handle.data[body].y += coeff_r * rry - h_gamma_r * angvel_handle.data[body].y;
                // torque_handle.data[body].z += coeff_r * rrz - h_gamma_r * angvel_handle.data[body].z;
                // }
        // }
        }
        
        
        
        }

    // done profiling
    if (m_prof)
        m_prof->pop();
    }

/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1
*/
void TwoStepLangevin::integrateStepTwo(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();

    if (m_aniso && !m_warned_aniso)
        {
        m_exec_conf->msg->warning() << "integrate.langevin: this thermostat "
            "does not operate on rotational degrees of freedom" << endl;
        m_warned_aniso = true;
        }

    const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();

    // profile this step
    if (m_prof)
        m_prof->push("Langevin step 2");

    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_gamma(m_gamma, access_location::host, access_mode::read);

    // grab some initial variables
    const Scalar currentTemp = m_T->getValue(timestep);
    const unsigned int D = Scalar(m_sysdef->getNDimensions());
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
        Scalar rx = saru.s<Scalar>(-1,1);
        Scalar ry = saru.s<Scalar>(-1,1);
        Scalar rz =  saru.s<Scalar>(-1,1);

        Scalar gamma;
        if (m_use_lambda)
            gamma = m_lambda*h_diameter.data[j];
        else
            {
            unsigned int type = __scalar_as_int(h_pos.data[j].w);
            gamma = h_gamma.data[type];
            }

        // compute the bd force
        Scalar coeff = fast::sqrt(Scalar(6.0) *gamma*currentTemp/m_deltaT);
        Scalar bd_fx = rx*coeff - gamma*h_vel.data[j].x;
        Scalar bd_fy = ry*coeff - gamma*h_vel.data[j].y;
        Scalar bd_fz = rz*coeff - gamma*h_vel.data[j].z;

        if (D < 3)
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
        }

    // update energy reservoir
    if (m_tally)
        {
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

void export_TwoStepLangevin()
    {
    class_<TwoStepLangevin, boost::shared_ptr<TwoStepLangevin>, bases<TwoStepLangevinBase>, boost::noncopyable>
        ("TwoStepLangevin", init< boost::shared_ptr<SystemDefinition>,
                            boost::shared_ptr<ParticleGroup>,
                            boost::shared_ptr<Variant>,
                            unsigned int,
                            bool,
                            Scalar,
                            const std::string&,
                            bool,
                            bool
                            >())
        .def("setTally", &TwoStepLangevin::setTally)
        ;
    }
