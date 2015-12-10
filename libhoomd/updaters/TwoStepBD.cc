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

#include "TwoStepBD.h"
#include "saruprng.h"

#ifdef ENABLE_MPI
#include "HOOMDMPI.h"
#endif

#include <boost/python.hpp>
using namespace boost::python;
using namespace std;

/*! \file TwoStepBD.h
    \brief Contains code for the TwoStepBD class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param T Temperature set point as a function of time
    \param seed Random seed to use in generating random numbers
    \param use_lambda If true, gamma=lambda*diameter, otherwise use a per-type gamma via setGamma()
    \param lambda Scale factor to convert diameter to gamma
*/
TwoStepBD::TwoStepBD(boost::shared_ptr<SystemDefinition> sysdef,
                           boost::shared_ptr<ParticleGroup> group,
                           boost::shared_ptr<Variant> T,
                           unsigned int seed,
                           bool use_lambda,
                           Scalar lambda)
    : TwoStepLangevinBase(sysdef, group, T, seed, use_lambda, lambda)
    {
    m_exec_conf->msg->notice(5) << "Constructing TwoStepBD" << endl;
    }

TwoStepBD::~TwoStepBD()
    {
    m_exec_conf->msg->notice(5) << "Destroying TwoStepBD" << endl;
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1

    The integration method here is from the book "The Langevin and Generalised Langevin Approach to the Dynamics of
    Atomic, Polymeric and Colloidal Systems", chapter 6.
*/
void TwoStepBD::integrateStepOne(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();

    // profile this step
    if (m_prof)
        m_prof->push("BD step 1");

    // grab some initial variables
    const Scalar currentTemp = m_T->getValue(timestep);
    const unsigned int D = Scalar(m_sysdef->getNDimensions());

    const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);

    ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_gamma(m_gamma, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
    
    
    ///////////////
    if (m_aniso)
        {
        ArrayHandle<Scalar4> h_orien(m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> h_torque(m_pdata->getNetTorqueArray(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar3> h_inertia(m_pdata->getMomentsOfInertiaArray(), access_location::host, access_mode::read);
        ArrayHandle<Scalar3> h_angmom(m_pdata->getAltAngularMomentumArray(), access_location::host, access_mode::read);
        }
    ///////////////

    const BoxDim& box = m_pdata->getBox();

    // initialize the RNG
    Saru saru(m_seed, timestep, 0xffaabb);

    // perform the first half step
    // r(t+deltaT) = r(t) + (Fc(t) + Fr)*deltaT/gamma
    // v(t+deltaT) = random distribution consistent with T
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);

        // compute the random force
        Scalar rx = saru.s<Scalar>(-1,1);
        Scalar ry = saru.s<Scalar>(-1,1);
        Scalar rz = saru.s<Scalar>(-1,1);

        Scalar gamma;
        if (m_use_lambda)
            gamma = m_lambda*h_diameter.data[j];
        else
            {
            unsigned int type = __scalar_as_int(h_pos.data[j].w);
            gamma = h_gamma.data[type];
            }

        // compute the bd force (the extra factor of 3 is because <rx^2> is 1/3 in the uniform -1,1 distribution
        // it is not the dimensionality of the system
        Scalar coeff = fast::sqrt(Scalar(3.0)*Scalar(2.0)*gamma*currentTemp/m_deltaT);
        Scalar Fr_x = rx*coeff;
        Scalar Fr_y = ry*coeff;
        Scalar Fr_z = rz*coeff;

        if (D < 3)
            Fr_z = Scalar(0.0);

        // update position
        h_pos.data[j].x += (h_net_force.data[j].x + Fr_x) * m_deltaT / gamma;
        h_pos.data[j].y += (h_net_force.data[j].y + Fr_y) * m_deltaT / gamma;
        h_pos.data[j].z += (h_net_force.data[j].z + Fr_z) * m_deltaT / gamma;

        // particles may have been moved slightly outside the box by the above steps, wrap them back into place
        box.wrap(h_pos.data[j], h_image.data[j]);

        // draw a new random velocity for particle j
        Scalar mass =  h_vel.data[j].w;
        Scalar sigma = fast::sqrt(currentTemp/mass);
        h_vel.data[j].x = gaussian_rng(saru, sigma);
        h_vel.data[j].y = gaussian_rng(saru, sigma);
        if (D > 2)
            h_vel.data[j].z = gaussian_rng(saru, sigma);
        else
            h_vel.data[j].z = 0;
        
        
        ///////////////
        Scalar coeff_r = fast::sqrt(Scalar(2.0)*gamma_r*currentTemp/m_deltaT);
        Scalar tau_r = gaussian_rng(saru, sigma);
        Scalar gamma_r = 1.0;
	// for testing. This is to be passed in from boost python
	
        // gamma_r needs initialization
        
        // if (m_use_lambda)
        //     gamma = m_lambda*h_diameter.data[j];
        // else
        //     {
        //     unsigned int type = __scalar_as_int(h_pos.data[j].w);
        //     gamma = h_gamma.data[type];
        //     }

        if (D < 3)
            {
            // h_orien.data[j].x += Scalar(1.0 / 2.0) * m_deltaT / gamma_r * (h_torque.data[j].x + tau_r) ;
            // h_orien.data[j].y += Scalar(1.0 / 2.0) * m_deltaT / gamma_r * (h_torque.data[j].y + tau_r) ;
            // h_orien.data[j].z += Scalar(1.0 / 2.0) * m_deltaT / gamma_r * (h_torque.data[j].z + tau_r) ;
            vec3<Scalar> axis (0.0, 0.0, 1.0);
            Scalar theta = (h_torque.data[j].z + tau_r) / gamma_r;
            quat<Scalar> omega = fromAxisAngle(axis, theta);
            quat<Scalar> q (h_orien.data[j]);
            q += Scalar(0.5) * m_deltaT  * q * omega;
            // renormalize (improves stability)
            q = q*(Scalar(1.0)/slow::sqrt(norm2(q)));
            h_orien.data[j].x = q.x;
            h_orien.data[j].y = q.y;
            h_orien.data[j].z = q.z;
            h_orien.data[j].w = q.w;
            }

        ///////////////
        }

    // done profiling
    if (m_prof)
        m_prof->pop();
    }

/*! \param timestep Current time step
*/
void TwoStepBD::integrateStepTwo(unsigned int timestep)
    {
    // there is no step 2 in Brownian dynamics.
    }

void export_TwoStepBD()
    {
    class_<TwoStepBD, boost::shared_ptr<TwoStepBD>, bases<TwoStepLangevinBase>, boost::noncopyable>
        ("TwoStepBD", init< boost::shared_ptr<SystemDefinition>,
                            boost::shared_ptr<ParticleGroup>,
                            boost::shared_ptr<Variant>,
                            unsigned int,
                            bool,
                            Scalar
                            >())
        ;
    }
