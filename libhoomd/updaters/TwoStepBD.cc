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
#include "VectorMath.h"
#include "saruprng.h"
#include "QuaternionMath.h"
#include "HOOMDMath.h"


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
                           Scalar lambda,
                           bool noiseless_t,
                           bool noiseless_r
                           )
  : TwoStepLangevinBase(sysdef, group, T, seed, use_lambda, lambda), 
    m_noiseless_t(noiseless_t), m_noiseless_r(noiseless_r)
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
    
    ArrayHandle<Scalar> h_gamma_r(m_gamma_r, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_torque(m_pdata->getNetTorqueArray(), access_location::host, access_mode::readwrite);
    
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
        if (m_noiseless_t) coeff = 0.0;
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
        // for testing rotational noise in rotational Brownian dynamics (2D only!)

        if (D < 3 && m_aniso)
        {
            unsigned int type_r = __scalar_as_int(h_pos.data[j].w);
            Scalar gamma_r = h_gamma_r.data[type_r];
            
            if (gamma_r)
                {
                // original Gaussian random torque
                Scalar sigma_r = fast::sqrt(Scalar(2.0)*gamma_r*currentTemp/m_deltaT);
                Scalar tau_r = gaussian_rng(saru, sigma_r); 
                if (m_noiseless_r) tau_r = 0.0;
                
                vec3<Scalar> axis (0.0, 0.0, 1.0);
                Scalar theta = (h_torque.data[j].z + tau_r) / gamma_r;
                // quat<Scalar> omega = quat<Scalar>::fromAxisAngle(axis, theta);
                quat<Scalar> omega (make_scalar4(0,0,0, theta));
                quat<Scalar> q (h_orientation.data[j]);
                q += Scalar(0.5) * m_deltaT * omega * q ;               
                
                // re-normalize (improves stability)
                q = q*(Scalar(1.0)/slow::sqrt(norm2(q)));
                h_orientation.data[j] = quat_to_scalar4(q);
                }
            }
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
                            Scalar,
                            bool, 
                            bool>())
        ;
    }
