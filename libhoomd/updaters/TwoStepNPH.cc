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

// Maintainer: jglaser

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include "TwoStepNPH.h"

/*! \file TwoStepNPH.h
    \brief Contains code for the TwoStepNPH class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param thermo ComputeThermo to use
    \param W piston mass
    \param P pressure set point
    \param mode integration mode (cubic, orthorhombic or tetragonal)
    \param suffix suffix for log quantity
*/
TwoStepNPH::TwoStepNPH(boost::shared_ptr<SystemDefinition> sysdef,
                       boost::shared_ptr<ParticleGroup> group,
                       boost::shared_ptr<ComputeThermo> thermo,
                       Scalar W,
                       boost::shared_ptr<Variant> P,
                       integrationMode mode,
                       const std::string& suffix)
    : IntegrationMethodTwoStep(sysdef, group), m_thermo(thermo), m_W(W), m_P(P), m_mode(mode), m_state_initialized(false)
    {
    // set a named, but otherwise blank set of integrator variables
    IntegratorVariables v = getIntegratorVariables();

    if (!restartInfoTestValid(v, "nph", 3))
            {
            v.type = "nph";
            v.variable.resize(3);

            // store initial box length conjugate
            v.variable[0] = 0.0;               // etax
            v.variable[1] = 0.0;               // etay
            v.variable[2] = 0.0;               // etaz

            setValidRestart(false);
            }
        else
            setValidRestart(true);

    setIntegratorVariables(v);
    m_log_name = string("nph_barostat_energy") + suffix;

    m_curr_P_diag = make_scalar3(0.0,0.0,0.0);
    }

/*! \param timestep Current time step
    \post Particle positions and box dimensions are moved forward to timestep+1, all velocities quantities to
          timestep + dt/2
*/
void TwoStepNPH::integrateStepOne(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();
    if (group_size == 0)
        return;

    // profile this step
    if (m_prof)
        m_prof->push("NPH step 1");

    if (!m_state_initialized)
        {
        //! compute the current pressure tensor
        m_thermo->compute(timestep);

        // compute pressure tensor for next half time step
        PressureTensor P;
        P = m_thermo->getPressureTensor();

        // If for some reason the pressure is not valid, assume internal pressure = external pressure
        if ( isnan(P.xx) || isnan(P.xy) || isnan(P.xz) || isnan(P.yy) || isnan(P.yz) || isnan(P.zz) )
            {
            Scalar extP = m_P->getValue(timestep);
            m_curr_P_diag = make_scalar3(extP,extP,extP);
            }
        else
            {
            // store diagonal elements of pressure tensor
            m_curr_P_diag = make_scalar3(P.xx,P.yy,P.zz);
            }

        m_state_initialized = true;
        }

    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::readwrite);

    /* perform the first half step of the explicitly reversible NPH integration scheme.

       This follows from operator factorization

       - orthorhombic case:

         1) eta_alpha(t+deltaT/2) = eta_alpha(t) + deltaT/2 * V/L_alpha * ( P_{alpha,alpha}(t) - P_ext)
         2) v' = v(t) + (1/2)a(t)*deltaT
         3) L(t+deltaT/2) = L(t) + deltaT*eta(t+deltaT/2)/(2*W)
         4) r'_alpha = r(t) + v'*deltaT* (L_{alpha}^2(t)/L_{alpha}^2(t+deltaT/2))
         5a) L(t+deltaT) = L(t+deltaT/2) + deltaT*eta(t+deltaT/2)/2/W
         5b) r_alpha(t+deltaT) = L_alpha(t+deltaT)/L_alpha(t)*r'_alpha
         5c) v''_alpha = L_alpha(t)/L_alpha(t+deltaT) * v'_alpha

       alpha denotes a cartesian index.

       - isotropic case:

         only eta_x := eta is used and instead of step 1) we have

         1') eta(t+deltaT/2) = eta(t) + deltaT/2 * (1/3*Tr P(t) - P_ext)

         furthermore, in step 3) and 5a) L is replaced with V=L^3

       - tetragonal case:

         Lx := L_perp, Ly = Lz := L_par
         eta_x := eta_perp, etay := eta_par, etaz unused

         instead of step 1) we have

         1'a) eta_perp(t+deltaT/2) = eta_perp + deltaT/2 * V/L_perp * ( P_xx(t) - P_ext)
         1'b) eta_par(t+deltaT/2) = eta_par + deltaT/2 * V/L_par * ( P_yy(t) + P_zz(t) - 2*P_ext)

         steps 3) and 5a) are split into two sub-steps

         L_perp(i+1) = L_perp(i) + deltaT/(2*W)*eta_perp
         L_par(i+1) = L_par(i) + deltaT/(4*W)*eta_par
    */

    IntegratorVariables v = getIntegratorVariables();
    Scalar &etax = v.variable[0];
    Scalar &etay = v.variable[1];
    Scalar &etaz = v.variable[2];

    // obtain box lengths
    Scalar Lx = Scalar(0.0);
    Scalar Ly = Scalar(0.0);
    Scalar Lz = Scalar(0.0);
    Scalar volume = Scalar(0.0);

    BoxDim box = m_pdata->getBox();
    Lx = box.xhi - box.xlo;
    Ly = box.yhi - box.ylo;
    Lz = box.zhi - box.zlo;
    volume = Lx*Ly*Lz;

    Scalar extP = m_P->getValue(timestep);

    // advance eta(t)->eta(t+deltaT/2) (step one)
    if (m_mode == orthorhombic)
        {
        Scalar VdeltaThalf = Scalar(1./2.)*volume*m_deltaT;
        etax += VdeltaThalf/Lx * (m_curr_P_diag.x - extP);
        etay += VdeltaThalf/Ly * (m_curr_P_diag.y - extP);
        etaz += VdeltaThalf/Lz * (m_curr_P_diag.z - extP);
        }
    else if (m_mode == tetragonal)
       {
       Scalar VdeltaThalf = Scalar(1./2.)*volume*m_deltaT;
       etax += VdeltaThalf/Lx * (m_curr_P_diag.x - extP);
       etay += VdeltaThalf/Ly * (m_curr_P_diag.y + m_curr_P_diag.z - Scalar(2.0)*extP);
       }
    else if (m_mode == cubic)
        {
        etax += Scalar(1./2.)*m_deltaT*(Scalar(1./3.)*(m_curr_P_diag.x + m_curr_P_diag.y + m_curr_P_diag.z) - extP);
        }

    // update the box length L(t) -> L(t+deltaT/2) (step three)
    // (since we still keep the accelerations a(t) computed for box length L_alpha(t) in memory,
    // needed in step two, we can exchange the order of the two steps)
    // also pre-calculate L(t+deltaT) (step 5a, only depends on eta(t) of step one)
    Scalar Lx_old = Lx;
    Scalar Ly_old = Ly;
    Scalar Lz_old = Lz;

    Scalar Lx_final = Scalar(0.0);
    Scalar Ly_final = Scalar(0.0);
    Scalar Lz_final = Scalar(0.0);

    Scalar deltaThalfoverW = Scalar(1./2.)*m_deltaT/m_W;

    if (m_mode == orthorhombic)
        {
        Lx += deltaThalfoverW*etax;
        Ly += deltaThalfoverW*etay;
        Lz += deltaThalfoverW*etaz;
        Lx_final = Lx + deltaThalfoverW*etax;
        Ly_final = Ly + deltaThalfoverW*etay;
        Lz_final = Lz + deltaThalfoverW*etaz;
        }
    else if (m_mode == tetragonal)
        {
        Lx += deltaThalfoverW*etax;
        Ly += Scalar(1./2.)*deltaThalfoverW*etay;
        Lz = Ly;
        Lx_final = Lx + deltaThalfoverW*etax;
        Ly_final = Ly + Scalar(1./2.)*deltaThalfoverW*etay;
        Lz_final = Ly_final;
        }
    else if (m_mode == cubic)
        {
        volume += deltaThalfoverW*etax;
        Lx = pow(volume,Scalar(1./3.)); // Lx = Ly = Lz = V^(1/3)
        Ly = Lx;
        Lz = Lx;
        Scalar volume_final = volume + deltaThalfoverW*etax;
        Lx_final = pow(volume_final,Scalar(1./3.)); // Lx = Ly = Lz = V^(1/3)
        Ly_final = Lx_final;
        Lz_final = Lx_final;
        }

    {
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);

    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        Scalar vxtmp = h_vel.data[j].x + Scalar(1.0/2.0)*h_accel.data[j].x*m_deltaT;
        Scalar vytmp = h_vel.data[j].y + Scalar(1.0/2.0)*h_accel.data[j].y*m_deltaT;
        Scalar vztmp = h_vel.data[j].z + Scalar(1.0/2.0)*h_accel.data[j].z*m_deltaT;

        // update positions using result of step two implicitly in the (combined) steps four and 5b
        h_pos.data[j].x = Lx_final/Lx_old*(h_pos.data[j].x+ vxtmp*m_deltaT*Lx_old*Lx_old/Lx/Lx);
        h_pos.data[j].y = Ly_final/Ly_old*(h_pos.data[j].y+ vytmp*m_deltaT*Ly_old*Ly_old/Ly/Ly);
        h_pos.data[j].z = Lz_final/Lz_old*(h_pos.data[j].z+ vztmp*m_deltaT*Lz_old*Lz_old/Lz/Lz);

        // update velocities (step two and step 5c combined)
        h_vel.data[j].x = Lx_old/Lx_final*vxtmp;
        h_vel.data[j].y = Ly_old/Ly_final*vytmp;
        h_vel.data[j].z = Lz_old/Lz_final*vztmp;
        }

    // wrap the particle around the box
    Lx = Lx_final;
    Ly = Ly_final;
    Lz = Lz_final;
    box = BoxDim(Lx_final, Ly_final, Lz_final);
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        // wrap the particles around the box
        if (h_pos.data[j].x >= box.xhi)
            {
            h_pos.data[j].x -= Lx;
            h_image.data[j].x++;
            }
        else if (h_pos.data[j].x < box.xlo)
            {
            h_pos.data[j].x += Lx;
            h_image.data[j].x--;
            }

        if (h_pos.data[j].y >= box.yhi)
            {
            h_pos.data[j].y -= Ly;
            h_image.data[j].y++;
            }
        else if (h_pos.data[j].y < box.ylo)
            {
            h_pos.data[j].y += Ly;
            h_image.data[j].y--;
            }

        if (h_pos.data[j].z >= box.zhi)
            {
            h_pos.data[j].z -= Lz;
            h_image.data[j].z++;
            }
        else if (h_pos.data[j].z < box.zlo)
            {
            h_pos.data[j].z += Lz;
            h_image.data[j].z--;
            }
        }
    }

    // update the simulation box
    m_pdata->setBox(box);

    setIntegratorVariables(v);

    // done profiling
    if (m_prof)
        m_prof->pop();
    }

/*! \param timestep Current time step
    \post particle velocities and box momentum are moved forward to t+deltaT
*/
void TwoStepNPH::integrateStepTwo(unsigned int timestep)
    {
    /* the second step of the explicitly reversible integrator consists of the following to sub-steps

       6) v(t+deltaT) = v'' + 1/2 * a(t+deltaT)*deltaT
       7) eta(t+deltaT/2) -> eta(t+deltaT)
     */
    unsigned int group_size = m_group->getNumMembers();
    if (group_size == 0)
        return;

    const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();

    // profile this step
    if (m_prof)
        m_prof->push("NPH step 2");

    IntegratorVariables v = getIntegratorVariables();
    Scalar &etax = v.variable[0];
    Scalar &etay = v.variable[1];
    Scalar &etaz = v.variable[2];

    {
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);

    // v(t+deltaT) = v'' + 1/2 * a(t+deltaT)*deltaT
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);

        // first, calculate acceleration from the net force
        Scalar minv = Scalar(1.0) / h_vel.data[j].w;
        h_accel.data[j].x = h_net_force.data[j].x*minv;
        h_accel.data[j].y = h_net_force.data[j].y*minv;
        h_accel.data[j].z = h_net_force.data[j].z*minv;

        // then, update the velocity
        h_vel.data[j].x += Scalar(1.0/2.0)*h_accel.data[j].x*m_deltaT;
        h_vel.data[j].y += Scalar(1.0/2.0)*h_accel.data[j].y*m_deltaT;
        h_vel.data[j].z += Scalar(1.0/2.0)*h_accel.data[j].z*m_deltaT;
        }
    }
    // now compute pressure tensor with updated virial and velocities
    m_thermo->compute(timestep+1);

    PressureTensor P;
    P = m_thermo->getPressureTensor();

    // If for some reason the pressure is not valid, assume internal pressure = external pressure
    if ( isnan(P.xx) || isnan(P.xy) || isnan(P.xz) || isnan(P.yy) || isnan(P.yz) || isnan(P.zz) )
        {
        Scalar extP = m_P->getValue(timestep);
        m_curr_P_diag = make_scalar3(extP,extP,extP);
        }
    else
        {
        // store diagonal elements of pressure tensor
        m_curr_P_diag = make_scalar3(P.xx,P.yy,P.zz);
        }

    // advance eta(t+deltaT/2) -> eta(t+deltaT)
    Scalar extP = m_P->getValue(timestep);

    if (m_mode == orthorhombic || m_mode == tetragonal)
        {
        const BoxDim &box = m_pdata->getBox();
        Scalar Lx = box.xhi - box.xlo;
        Scalar Ly = box.yhi - box.ylo;
        Scalar Lz = box.zhi - box.zlo;
        Scalar volume = Lx*Ly*Lz;
        Scalar VdeltaThalf = Scalar(1./2.)*volume*m_deltaT;

        // eta_alpha(t+deltaT) = eta_alpha(t+deltaT/2) + deltaT/2 * V/L_alpha * ( P_{alpha,alpha}(t) - P )
        etax += VdeltaThalf/Lx * (m_curr_P_diag.x - extP);

        if (m_mode == orthorhombic)
            {
            etay += VdeltaThalf/Ly * (m_curr_P_diag.y - extP);
            etaz += VdeltaThalf/Lz * (m_curr_P_diag.z - extP);
            }
        else
            {
            //tetragonal
            etay += VdeltaThalf/Ly * (m_curr_P_diag.y + m_curr_P_diag.z - Scalar(2.0)*extP);
            }
        }
    else if (m_mode == cubic)
        {
        etax += Scalar(1./2.)*m_deltaT * (Scalar(1./3.)*(m_curr_P_diag.x + m_curr_P_diag.y + m_curr_P_diag.z) - extP);
        }

    setIntegratorVariables(v);

    // done profiling
    if (m_prof)
        m_prof->pop();
    }

/*! Returns a list of log quantities this compute calculates
*/
std::vector< std::string > TwoStepNPH::getProvidedLogQuantities()
    {
    vector<string> result;
    result.push_back(m_log_name);
    return result;
    }

/*! \param quantity Name of the log quantity to get
    \param timestep Current time step of the simulation
    \param my_quantity_flag passed as false, changed to true if quantity logged here
*/
Scalar TwoStepNPH::getLogValue(const std::string& quantity, unsigned int timestep, bool &my_quantity_flag)
    {
    if (quantity == m_log_name)
        {
        my_quantity_flag = true;
        IntegratorVariables v = getIntegratorVariables();
        Scalar& etax = v.variable[0];
        Scalar& etay = v.variable[1];
        Scalar& etaz = v.variable[2];
        Scalar barostat_energy = Scalar(0.0);
        if (m_mode == orthorhombic)
            barostat_energy = (etax*etax+etay*etay+etaz*etaz) / Scalar(2.0) / m_W;
        else if (m_mode == tetragonal)
            barostat_energy = (etax*etax+etay*etay/Scalar(2.0)) / Scalar(2.0) / m_W;
        else if (m_mode == cubic)
            barostat_energy = etax*etax/Scalar(2.0)/m_W;

        return barostat_energy;
        }
    else
        return Scalar(0);
    }

void export_TwoStepNPH()
    {
    scope in_nph = class_<TwoStepNPH, boost::shared_ptr<TwoStepNPH>, bases<IntegrationMethodTwoStep>, boost::noncopyable>
        ("TwoStepNPH", init< boost::shared_ptr<SystemDefinition>, boost::shared_ptr<ParticleGroup>, boost::shared_ptr<ComputeThermo>, Scalar, boost::shared_ptr<Variant>, TwoStepNPH::integrationMode, const std::string& >())
        .def("setIntegrationMode", &TwoStepNPH::setIntegrationMode)
        .def("setW", &TwoStepNPH::setW)
        .def("setP", &TwoStepNPH::setP)
        ;

    enum_<TwoStepNPH::integrationMode>("integrationMode")
    .value("cubic", TwoStepNPH::cubic)
    .value("orthorhombic", TwoStepNPH::orthorhombic)
    .value("tetragonal", TwoStepNPH::tetragonal)
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

