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

#include "TwoStepNPTMTK.h"

/*! \file TwoStepNPTMTK.cc
    \brief Contains code for the TwoStepNPTMTK class
*/

//! Coefficients of f(x) = sinh(x)/x = a_0 + a_2 * x^2 + a_4 * x^4 + a_6 * x^6 + a_8 * x^8 + a_10 * x^10
const Scalar f_coeff[] = {Scalar(1.0), Scalar(1.0/6.0), Scalar(1.0/120.0), Scalar(1.0/5040.0),
                          Scalar(1.0/362880.0), Scalar(1.0/39916800.0)};

//! Coefficients of g(x) = coth(x) - 1/x =  a_1 * x + a_3 * x^3 + a_5 * x^5 + a_7 * x^7 + a_9 * x^9
const Scalar g_coeff[] = {Scalar(1.0/3.0), Scalar(-1.0/45.0), Scalar(2.0/945.0),
                          Scalar(-1.0/4725.0), Scalar(1.0/93555.0)};

//! Coefficients of h(x) = (-1/sinh^2(x)+1/x^2) = a_0 + a_2 * x^2 + a_4 * x^4 + a_6 * x^6 + a_8 * x^8 + a_10 * x^10
const Scalar h_coeff[] = {Scalar(1.0/3.0), Scalar(-1.0/15.0), Scalar(2.0/189.0), Scalar(-1.0/675.0),
                          Scalar(2.0/10395.0), Scalar(-1382.0/58046625.0)};
 
/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param thermo_group ComputeThermo to compute thermo properties of the integrated \a group
    \param tau NPT temperature period
    \param tauP NPT pressure period
    \param T Temperature set point
    \param P Pressure set point
    \param mode Mode of integration
*/
TwoStepNPTMTK::TwoStepNPTMTK(boost::shared_ptr<SystemDefinition> sysdef,
                       boost::shared_ptr<ParticleGroup> group,
                       boost::shared_ptr<ComputeThermo> thermo_group,
                       Scalar tau,
                       Scalar tauP,
                       boost::shared_ptr<Variant> T,
                       boost::shared_ptr<Variant> P,
                       integrationMode mode)
    : IntegrationMethodTwoStep(sysdef, group), m_thermo_group(thermo_group),
      m_tau(tau), m_tauP(tauP), m_T(T), m_P(P), m_mode(mode)
    {
    m_exec_conf->msg->notice(5) << "Constructing TwoStepNPTMTK" << endl;

    if (m_tau <= 0.0)
        m_exec_conf->msg->warning() << "integrate.npt: tau set less than 0.0" << endl;
    if (m_tauP <= 0.0)
        m_exec_conf->msg->warning() << "integrate.npt: tauP set less than 0.0" << endl;

    m_V = m_pdata->getGlobalBox().getVolume();  // volume

    // set initial state
    IntegratorVariables v = getIntegratorVariables();

    // choose dummy values for the current temp and pressure
    m_curr_group_T = 0.0;

    if (!restartInfoTestValid(v, "npt_mtk", 8))
        {
        v.type = "npt_mtk";
        v.variable.resize(8,Scalar(0.0));
        setValidRestart(false);
        }
    else
        setValidRestart(true);

    setIntegratorVariables(v);

    m_log_names.resize(2);
    m_log_names[0] = "npt_mtk_thermostat_energy";
    m_log_names[1] = "npt_mtk_barostat_energy";

    for (unsigned int i = 0; i < 9; ++i)
        m_evec_arr[i] = Scalar(0.0);
    }

TwoStepNPTMTK::~TwoStepNPTMTK()
    {
    m_exec_conf->msg->notice(5) << "Destroying TwoStepNPTMTK" << endl;
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per the Martyna-Tobias-Klein barostat and thermostat
*/
void TwoStepNPTMTK::integrateStepOne(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();

    if (group_size == 0)
        return;

    // compute the current thermodynamic properties
    m_thermo_group->compute(timestep);

    // compute temperature for the next half time step
    m_curr_group_T = m_thermo_group->getTemperature();

    // compute pressure for the next half time step
    assert(m_mode == cubic || m_mode == orthorhombic || m_mode == tetragonal || m_mode == triclinic);

    PressureTensor P = m_thermo_group->getPressureTensor();

    if ( isnan(P.xx) || isnan(P.xy) || isnan(P.xz) || isnan(P.yy) || isnan(P.yz) || isnan(P.zz) )
        {
        Scalar extP = m_P->getValue(timestep);
        P.xx = P.yy = P.zz = extP;
        P.xy = P.xz = P.yz = Scalar(0.0);
        }

    // profile this step
    if (m_prof)
        m_prof->push("NPT step 1");

    IntegratorVariables v = getIntegratorVariables();
    Scalar& eta = v.variable[0];  // Thermostat variable
    Scalar& xi = v.variable[1];   // Thermostat velocity
    Scalar& nuxx = v.variable[2];  // Barostat tensor, xx component
    Scalar& nuxy = v.variable[3];  // Barostat tensor, xy component
    Scalar& nuxz = v.variable[4];  // Barostat tensor, xz component
    Scalar& nuyy = v.variable[5];  // Barostat tensor, yy component
    Scalar& nuyz = v.variable[6];  // Barostat tensor, yz component
    Scalar& nuzz = v.variable[7];  // Barostat tensor, zz component

    // advance barostat (nuxx, nuyy, nuzz) half a time step
    Scalar W = m_thermo_group->getNDOF()*m_T->getValue(timestep)*m_tauP*m_tauP;
    Scalar mtk_term = Scalar(1.0/2.0)*m_deltaT*m_curr_group_T/W;
    if (m_mode == cubic)
        {
        Scalar P_iso = Scalar(1.0/3.0)*(P.xx + P.yy + P.zz);
        nuxx += Scalar(1.0/2.0)*m_deltaT*m_V/W*(P_iso - m_P->getValue(timestep)) + mtk_term;
        nuyy = nuzz = nuxx;
        }
    else if (m_mode == tetragonal)
        {
        nuxx += Scalar(1.0/2.0)*m_deltaT*m_V/W*(P.xx - m_P->getValue(timestep)) + mtk_term;
        nuyy += Scalar(1.0/2.0)*m_deltaT*m_V/W*((P.yy + P.zz)/Scalar(2.0) - m_P->getValue(timestep)) + mtk_term;
        nuzz = nuyy;
        }
    else if (m_mode == orthorhombic)
        {
        nuxx += Scalar(1.0/2.0)*m_deltaT*m_V/W*(P.xx - m_P->getValue(timestep)) + mtk_term;
        nuyy += Scalar(1.0/2.0)*m_deltaT*m_V/W*(P.yy - m_P->getValue(timestep)) + mtk_term;
        nuzz += Scalar(1.0/2.0)*m_deltaT*m_V/W*(P.zz - m_P->getValue(timestep)) + mtk_term;
        }
    else if (m_mode == triclinic)
        {
        nuxx += Scalar(1.0/2.0)*m_deltaT*m_V/W*(P.xx - m_P->getValue(timestep)) + mtk_term;
        nuxy += Scalar(1.0/2.0)*m_deltaT*m_V/W*P.xy;
        nuxz += Scalar(1.0/2.0)*m_deltaT*m_V/W*P.xz;
        nuyy += Scalar(1.0/2.0)*m_deltaT*m_V/W*(P.yy - m_P->getValue(timestep)) + mtk_term;
        nuyz += Scalar(1.0/2.0)*m_deltaT*m_V/W*P.yz;
        nuzz += Scalar(1.0/2.0)*m_deltaT*m_V/W*(P.zz - m_P->getValue(timestep)) + mtk_term;
        }

    // advance thermostat (xi, eta) half a time step
    Scalar xi_prime = xi + Scalar(1.0/4.0)*m_deltaT/m_tau/m_tau*(m_curr_group_T/m_T->getValue(timestep) - Scalar(1.0));
    xi = xi_prime+ Scalar(1.0/4.0)*m_deltaT/(m_tau*m_tau)*(m_curr_group_T/m_T->getValue(timestep)*
          exp(-xi_prime*m_deltaT) - Scalar(1.0));

    eta += Scalar(1.0/2.0)*xi_prime*m_deltaT;

    // precompute loop invariant quantities
    Scalar exp_thermo_fac = exp(-Scalar(1.0/2.0)*xi_prime*m_deltaT);

    // update the propagator matrix
    m_ndof = m_thermo_group->getNDOF();
    updatePropagator(nuxx, nuxy, nuxz, nuyy, nuyz, nuzz);

       {
        ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);

        for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
            {
            unsigned int j = m_group->getMemberIndex(group_idx);

            Scalar3 v = make_scalar3(h_vel.data[j].x, h_vel.data[j].y, h_vel.data[j].z);
            Scalar3 accel = h_accel.data[j];
            Scalar3 r = make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);

            // apply thermostat update of velocity
            v *= exp_thermo_fac;

            // update velocity and position by multiplication with upper triangular matrices
            v.x = m_mat_exp_v[0] * v.x + m_mat_exp_v[1] * v.y + m_mat_exp_v[2] * v.z;
            v.y = m_mat_exp_v[3] * v.y + m_mat_exp_v[4] * v.z;
            v.z = m_mat_exp_v[5] * v.z;

            v.x += m_mat_exp_v_int[0] * accel.x + m_mat_exp_v_int[1] * accel.y + m_mat_exp_v_int[2] * accel.z;
            v.y += m_mat_exp_v_int[3] * accel.y + m_mat_exp_v_int[4] * accel.z;
            v.z += m_mat_exp_v_int[5] * accel.z;

            r.x = m_mat_exp_r[0] * r.x + m_mat_exp_r[1] * r.y + m_mat_exp_r[2] * r.z;
            r.y = m_mat_exp_r[3] * r.y + m_mat_exp_r[4] * r.z;
            r.z = m_mat_exp_r[5] * r.z;

            r.x += m_mat_exp_r_int[0] * v.x + m_mat_exp_r_int[1] * v.y + m_mat_exp_r_int[2] * v.z;
            r.y += m_mat_exp_r_int[3] * v.y + m_mat_exp_r_int[4] * v.z;
            r.z += m_mat_exp_r_int[5] * v.z;

            // store velocity
            h_vel.data[j].x = v.x;
            h_vel.data[j].y = v.y;
            h_vel.data[j].z = v.z;

            // store position
            h_pos.data[j].x = r.x;
            h_pos.data[j].y = r.y;
            h_pos.data[j].z = r.z;
            }
        } // end of GPUArray scope

    // advance box lengths
    BoxDim global_box = m_pdata->getGlobalBox();
    Scalar3 a = global_box.getLatticeVector(0);
    Scalar3 b = global_box.getLatticeVector(1);
    Scalar3 c = global_box.getLatticeVector(2);

    // (a,b,c) are the columns of the (upper triangular) cell parameter matrix
    // multiply with upper triangular matrix
    a.x = m_mat_exp_r[0] * a.x;
    b.x = m_mat_exp_r[0] * b.x + m_mat_exp_r[1] * b.y;
    b.y = m_mat_exp_r[3] * b.y;
    c.x = m_mat_exp_r[0] * c.x + m_mat_exp_r[1] * c.y + m_mat_exp_r[2] * c.z;
    c.y = m_mat_exp_r[3] * c.y + m_mat_exp_r[4] * c.z;
    c.z = m_mat_exp_r[5] * c.z;

#ifdef ENABLE_MPI
    if (m_comm)
        {
        // broadcast integrator variables from rank 0 to other processors
        MPI_Bcast(&eta, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        MPI_Bcast(&xi, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        MPI_Bcast(&nuxx, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        MPI_Bcast(&nuxy, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        MPI_Bcast(&nuyz, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        MPI_Bcast(&nuyy, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        MPI_Bcast(&nuyz, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        MPI_Bcast(&nuzz, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());

        MPI_Bcast(&a,sizeof(Scalar3), MPI_BYTE, 0, m_exec_conf->getMPICommunicator());
        MPI_Bcast(&b,sizeof(Scalar3), MPI_BYTE, 0, m_exec_conf->getMPICommunicator());
        MPI_Bcast(&c,sizeof(Scalar3), MPI_BYTE, 0, m_exec_conf->getMPICommunicator());
        }
#endif

    // update box dimensions
    global_box.setL(make_scalar3(a.x,b.y,c.z));
    Scalar xy = b.x/b.y;
    Scalar xz = c.x/c.z;
    Scalar yz = c.y/c.z;
    global_box.setTiltFactors(xy, xz, yz);
  
    // set global box
    m_pdata->setGlobalBox(global_box);

    m_V = global_box.getVolume();  // volume

    // Get new (local) box 
    BoxDim box = m_pdata->getBox();

        {
        ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
        ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);

        // Wrap particles
        for (unsigned int j = 0; j < m_pdata->getN(); j++)
            box.wrap(h_pos.data[j], h_image.data[j]);
        }

   setIntegratorVariables(v);

    // done profiling
    if (m_prof)
        m_prof->pop();
    }

/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1
*/
void TwoStepNPTMTK::integrateStepTwo(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();

    if (group_size == 0)
        return;

    const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();

   // profile this step
    if (m_prof)
        m_prof->push("NPT step 2");

    IntegratorVariables v = getIntegratorVariables();
    Scalar& eta = v.variable[0];  // Thermostat variable
    Scalar& xi = v.variable[1];   // Thermostat velocity
    Scalar& nuxx = v.variable[2];  // Barostat tensor, xx component
    Scalar& nuxy = v.variable[3];  // Barostat tensor, xy component
    Scalar& nuxz = v.variable[4];  // Barostat tensor, xz component
    Scalar& nuyy = v.variable[5];  // Barostat tensor, yy component
    Scalar& nuyz = v.variable[6];  // Barostat tensor, yz component
    Scalar& nuzz = v.variable[7];  // Barostat tensor, zz component

    // precalculate loop-invariant quantities
    {
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::readwrite);

    ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);

    // Kinetic energy * 2
    Scalar m_v2_sum(0.0);

    // perform second half step of NPT integration
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);

        // first, calculate acceleration from the net force
        Scalar m = h_vel.data[j].w;
        Scalar minv = Scalar(1.0) / m;
        h_accel.data[j].x = h_net_force.data[j].x*minv;
        h_accel.data[j].y = h_net_force.data[j].y*minv;
        h_accel.data[j].z = h_net_force.data[j].z*minv;

        Scalar3 accel = make_scalar3(h_accel.data[j].x, h_accel.data[j].y, h_accel.data[j].z);

        // update velocity by multiplication with upper triangular matrix
        Scalar3 v = make_scalar3(h_vel.data[j].x, h_vel.data[j].y, h_vel.data[j].z);
        v.x = m_mat_exp_v[0] * v.x + m_mat_exp_v[1] * v.y + m_mat_exp_v[2] * v.z;
        v.y = m_mat_exp_v[3] * v.y + m_mat_exp_v[4] * v.z;
        v.z = m_mat_exp_v[5] * v.z;

        v.x += m_mat_exp_v_int[0] * accel.x + m_mat_exp_v_int[1] * accel.y + m_mat_exp_v_int[2] * accel.z;
        v.y += m_mat_exp_v_int[3] * accel.y + m_mat_exp_v_int[4] * accel.z;
        v.z += m_mat_exp_v_int[5] * accel.z;

        // store velocity
        h_vel.data[j].x = v.x; h_vel.data[j].y = v.y; h_vel.data[j].z = v.z;

        // reduce E_kin
        m_v2_sum += m*(v.x*v.x + v.y*v.y + v.z*v.z);
        }

#ifdef ENABLE_MPI
    if (m_comm)
        MPI_Allreduce(MPI_IN_PLACE, &m_v2_sum, 1, MPI_HOOMD_SCALAR, MPI_SUM, m_exec_conf->getMPICommunicator() );
#endif

    // Advance thermostat half a time step
    Scalar T_prime =  m_v2_sum/m_thermo_group->getNDOF();
    Scalar xi_prime = xi + Scalar(1.0/4.0)*m_deltaT/m_tau/m_tau*(T_prime/m_T->getValue(timestep) - Scalar(1.0));
    xi = xi_prime+ Scalar(1.0/4.0)*m_deltaT/(m_tau*m_tau)*(T_prime/m_T->getValue(timestep) *
          exp(-xi_prime*m_deltaT) - Scalar(1.0));

    eta += Scalar(1.0/2.0)*xi_prime*m_deltaT;


    // rescale velocities
    Scalar exp_v_fac_thermo = exp(-Scalar(1.0/2.0)*xi_prime*m_deltaT);

    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);

        Scalar3 vel = make_scalar3(h_vel.data[j].x, h_vel.data[j].y, h_vel.data[j].z);
        vel = vel*exp_v_fac_thermo;
        h_vel.data[j].x = vel.x; h_vel.data[j].y = vel.y; h_vel.data[j].z = vel.z;
        }

    } // end GPUArray scope

    if (m_prof)
        m_prof->pop();

    // compute the current thermodynamic properties
    m_thermo_group->compute(timestep+1);

    if (m_prof)
        m_prof->push("NPT step 2");

    // compute temperature for the next half time step
    m_curr_group_T = m_thermo_group->getTemperature();

    // compute pressure for the next half time step
    PressureTensor P = m_thermo_group->getPressureTensor();

    if ( isnan(P.xx) || isnan(P.xy) || isnan(P.xz) || isnan(P.yy) || isnan(P.yz) || isnan(P.zz) )
        {
        Scalar extP = m_P->getValue(timestep);
        P.xx = P.yy = P.zz = extP;
        P.xy = P.xz = P.yz = Scalar(0.0);
        }

    // advance barostat (nuxx, nuyy, nuzz) half a time step
    Scalar W = m_thermo_group->getNDOF()*m_T->getValue(timestep)*m_tauP*m_tauP;
    Scalar mtk_term = Scalar(1.0/2.0)*m_deltaT*m_curr_group_T/W;
    if (m_mode == cubic)
        {
        Scalar P_iso = Scalar(1.0/3.0)*(P.xx + P.yy + P.zz);
        nuxx += Scalar(1.0/2.0)*m_deltaT*m_V/W*(P_iso - m_P->getValue(timestep)) + mtk_term;
        nuyy = nuzz = nuxx;
        }
    else if (m_mode == tetragonal)
        {
        nuxx += Scalar(1.0/2.0)*m_deltaT*m_V/W*(P.xx - m_P->getValue(timestep)) + mtk_term;
        nuyy += Scalar(1.0/2.0)*m_deltaT*m_V/W*((P.yy + P.zz)/Scalar(2.0) - m_P->getValue(timestep)) + mtk_term;
        nuzz = nuyy;
        }
    else if (m_mode == orthorhombic)
        {
        nuxx += Scalar(1.0/2.0)*m_deltaT*m_V/W*(P.xx - m_P->getValue(timestep)) + mtk_term;
        nuyy += Scalar(1.0/2.0)*m_deltaT*m_V/W*(P.yy - m_P->getValue(timestep)) + mtk_term;
        nuzz += Scalar(1.0/2.0)*m_deltaT*m_V/W*(P.zz - m_P->getValue(timestep)) + mtk_term;
        }
    else if (m_mode == triclinic)
        {
        nuxx += Scalar(1.0/2.0)*m_deltaT*m_V/W*(P.xx - m_P->getValue(timestep)) + mtk_term;
        nuxy += Scalar(1.0/2.0)*m_deltaT*m_V/W*P.xy;
        nuxz += Scalar(1.0/2.0)*m_deltaT*m_V/W*P.xz;
        nuyy += Scalar(1.0/2.0)*m_deltaT*m_V/W*(P.yy - m_P->getValue(timestep)) + mtk_term;
        nuyz += Scalar(1.0/2.0)*m_deltaT*m_V/W*P.yz;
        nuzz += Scalar(1.0/2.0)*m_deltaT*m_V/W*(P.zz - m_P->getValue(timestep)) + mtk_term;
        }

#ifdef ENABLE_MPI
    if (m_comm)
        {
        // broadcast integrator variables from rank 0 to other processors
        MPI_Bcast(&eta, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        MPI_Bcast(&xi, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        MPI_Bcast(&nuxx, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        MPI_Bcast(&nuxy, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        MPI_Bcast(&nuyz, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        MPI_Bcast(&nuyy, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        MPI_Bcast(&nuyz, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        MPI_Bcast(&nuzz, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        }
#endif

    setIntegratorVariables(v);

    // done profiling
    if (m_prof)
        m_prof->pop();
    }

/*! Returns a list of log quantities this compute calculates
*/
std::vector< std::string > TwoStepNPTMTK::getProvidedLogQuantities()
    {
    return m_log_names;
    }

/*! \param quantity Name of the log quantity to get
    \param timestep Current time step of the simulation
    \param my_quantity_flag passed as false, changed to true if quantity logged here
*/
Scalar TwoStepNPTMTK::getLogValue(const std::string& quantity, unsigned int timestep, bool &my_quantity_flag)
    {
    if (quantity == m_log_names[0])
        {
        my_quantity_flag = true;
        IntegratorVariables v = getIntegratorVariables();
        Scalar& eta = v.variable[0];
        Scalar& xi = v.variable[1];

        Scalar thermostat_energy = m_thermo_group->getNDOF()*m_T->getValue(timestep)
                                   *(eta + m_tau*m_tau*xi*xi/Scalar(2.0));
        return thermostat_energy;
        }
    else if (quantity == m_log_names[1])
        {
        my_quantity_flag = true;
        IntegratorVariables v = getIntegratorVariables();

        Scalar& nuxx = v.variable[2];  // Barostat tensor, xx component
        Scalar& nuxy = v.variable[3];  // Barostat tensor, xy component
        Scalar& nuxz = v.variable[4];  // Barostat tensor, xz component
        Scalar& nuyy = v.variable[5];  // Barostat tensor, yy component
        Scalar& nuyz = v.variable[6];  // Barostat tensor, yz component
        Scalar& nuzz = v.variable[7];  // Barostat tensor, zz component

        Scalar W = m_thermo_group->getNDOF()*m_T->getValue(timestep)*m_tauP*m_tauP;
        Scalar barostat_energy = Scalar(0.0);
        barostat_energy = W*(nuxx*nuxx+nuyy*nuyy+nuzz*nuzz+nuxy*nuxy+nuxz*nuxz+nuyz*nuyz) / Scalar(2.0);

        return barostat_energy;
        }
    else
        return Scalar(0);
    }

/*! \param nuxx Barostat matrix, xx element
    \param nuxy xy element
    \param nuxz xz element
    \param nuyy yy element
    \param nuyz yz element
    \param nuzz zz element
*/
void TwoStepNPTMTK::updatePropagator(Scalar nuxx, Scalar nuxy, Scalar nuxz, Scalar nuyy, Scalar nuyz, Scalar nuzz)
    {
    // calculate some factors needed for the update matrix
    Scalar mtk_term_2 = (nuxx+nuyy+nuzz)/(Scalar)m_ndof;
    Scalar3 v_fac = make_scalar3(-Scalar(1.0/4.0)*(nuxx+mtk_term_2),
                           -Scalar(1.0/4.0)*(nuyy+mtk_term_2),
                           -Scalar(1.0/4.0)*(nuzz+mtk_term_2));
    Scalar3 exp_v_fac = make_scalar3(exp(v_fac.x*m_deltaT),
                               exp(v_fac.y*m_deltaT),
                               exp(v_fac.z*m_deltaT));
    Scalar3 exp_v_fac_2 = make_scalar3(exp(Scalar(2.0)*v_fac.x*m_deltaT),
                               exp(Scalar(2.0)*v_fac.y*m_deltaT),
                               exp(Scalar(2.0)*v_fac.z*m_deltaT));

    Scalar3 r_fac = make_scalar3(Scalar(1.0/2.0)*nuxx,
                                 Scalar(1.0/2.0)*nuyy,
                                 Scalar(1.0/2.0)*nuzz);
    Scalar3 exp_r_fac = make_scalar3(exp(Scalar(1.0/2.0)*nuxx*m_deltaT),
                                     exp(Scalar(1.0/2.0)*nuyy*m_deltaT),
                                     exp(Scalar(1.0/2.0)*nuzz*m_deltaT));
    Scalar3 exp_r_fac_2 = make_scalar3(exp(nuxx*m_deltaT),
                                       exp(nuyy*m_deltaT),
                                       exp(nuzz*m_deltaT));

    // Calculate power series approximations of analytical functions entering the update equations

    Scalar3 arg_v = v_fac*m_deltaT;
    Scalar3 arg_r = r_fac*m_deltaT;

    // Calculate function f = sinh(x)/x
    Scalar3 f_v = make_scalar3(0.0,0.0,0.0);
    Scalar3 f_r = make_scalar3(0.0,0.0,0.0);
    Scalar3 term_v = make_scalar3(1.0,1.0,1.0);
    Scalar3 term_r = make_scalar3(1.0,1.0,1.0);

    for (unsigned int i = 0; i < 6; i++)
        {
        f_v += f_coeff[i] * term_v;
        f_r += f_coeff[i] * term_r;
        term_v = term_v * arg_v * arg_v;
        term_r = term_r * arg_r * arg_r;
        }

    // Calculate function g = cth(x) - 1/x
    Scalar3 g_v = make_scalar3(0.0,0.0,0.0);
    Scalar3 g_r = make_scalar3(0.0,0.0,0.0);

    term_v = arg_v;
    term_r = arg_r;

    for (unsigned int i = 0; i < 5; i++)
        {
        g_v += g_coeff[i] * term_v;
        g_r += g_coeff[i] * term_r;
        term_v = term_v * arg_v * arg_v;
        term_r = term_r * arg_r * arg_r;
        }

    // Calculate function h = -1/sinh^2(x) + 1/x^2
    Scalar3 h_v = make_scalar3(0.0,0.0,0.0);
    Scalar3 h_r = make_scalar3(0.0,0.0,0.0);

    term_v = term_r = make_scalar3(1.0,1.0,1.0);

    for (unsigned int i = 0; i < 6; i++)
        {
        h_v += h_coeff[i] * term_v;
        h_r += h_coeff[i] * term_r;
        
        term_v = term_v * arg_v * arg_v;
        term_r = term_r * arg_r * arg_r;
        }

    // Calculate matrix exponentials for upper triangular barostat matrix
    /* These are approximations accurate up to and including delta_t^2.
       They are fully time reversible  */   

    // Matrix exp. for velocity update
    m_mat_exp_v[0] = exp_v_fac_2.x;                                                  // xx
    m_mat_exp_v[1] = -m_deltaT*Scalar(1.0/4.0)*nuxy*(exp_v_fac_2.x + exp_v_fac_2.y); // xy
    m_mat_exp_v[2] = -m_deltaT*Scalar(1.0/4.0)*nuxz*(exp_v_fac_2.x + exp_v_fac_2.z)
                     +m_deltaT*m_deltaT*Scalar(1.0/32.0)*nuxy*nuyz
                      *(exp_v_fac_2.x + Scalar(2.0)*exp_v_fac_2.y + exp_v_fac_2.z);  // xz
    m_mat_exp_v[3] = exp_v_fac_2.y;                                                  // yy
    m_mat_exp_v[4] = -m_deltaT*Scalar(1.0/4.0)*nuyz*(exp_v_fac_2.y + exp_v_fac_2.z); // yz
    m_mat_exp_v[5] = exp_v_fac_2.z;                                                  // zz

    // integrated matrix exponential over deltaT
    Scalar3 xz_fac_v = make_scalar3((Scalar(1.0)+g_v.x)*(Scalar(1.0)+g_v.x)+h_v.x,
                                    (Scalar(1.0)+g_v.y)*(Scalar(1.0)+g_v.y)+h_v.y,
                                    (Scalar(1.0)+g_v.z)*(Scalar(1.0)+g_v.z)+h_v.z);

    m_mat_exp_v_int[0] = m_deltaT*Scalar(1.0/2.0)*exp_v_fac.x*f_v.x;                   // xx
    m_mat_exp_v_int[1] = -nuxy*m_deltaT*m_deltaT*Scalar(1.0/16.0)
                         *(exp_v_fac.x*f_v.x*(Scalar(1.0)+g_v.x)
                          +exp_v_fac.y*f_v.y*(Scalar(1.0)+g_v.y));                     // xy
    m_mat_exp_v_int[2] = -nuxz*m_deltaT*m_deltaT*Scalar(1.0/16.0)
                         *(exp_v_fac.x*f_v.x*(Scalar(1.0)+g_v.x)
                          +exp_v_fac.z*f_v.z*(Scalar(1.0)+g_v.z))
                         +m_deltaT*m_deltaT*m_deltaT*nuxy*nuyz*Scalar(1.0/256.0)
                          *(exp_v_fac.x*f_v.x*xz_fac_v.x
                           +Scalar(2.0)*exp_v_fac.y*f_v.y*xz_fac_v.y
                           +exp_v_fac.z*f_v.z*xz_fac_v.z);                             // xz
    m_mat_exp_v_int[3] = m_deltaT*Scalar(1.0/2.0)*exp_v_fac.y*f_v.y;                   // yy
    m_mat_exp_v_int[4] = -nuyz*m_deltaT*m_deltaT*Scalar(1.0/16.0)
                         *(exp_v_fac.x*f_v.y*(Scalar(1.0)+g_v.y)
                          +exp_v_fac.z*f_v.z*(Scalar(1.0)+g_v.z));                     // yz
    m_mat_exp_v_int[5] = m_deltaT*Scalar(1.0/2.0)*exp_v_fac.z*f_v.z;                   // zz

    // Matrix exp. for position update
    m_mat_exp_r[0] = exp_r_fac_2.x;                                                    // xx
    m_mat_exp_r[1] = m_deltaT*Scalar(1.0/2.0)*nuxy*(exp_r_fac_2.x + exp_r_fac_2.y);    // xy
    m_mat_exp_r[2] = m_deltaT*Scalar(1.0/2.0)*nuxz*(exp_r_fac_2.x + exp_r_fac_2.z)  
                     +m_deltaT*m_deltaT*Scalar(1.0/8.0)*nuxy*nuyz
                      *(exp_r_fac_2.x + Scalar(2.0)*exp_r_fac_2.y + exp_r_fac_2.z);    // xz
    m_mat_exp_r[3] = exp_r_fac_2.y;                                                    // yy
    m_mat_exp_r[4] = m_deltaT*Scalar(1.0/2.0)*nuyz*(exp_r_fac_2.y + exp_r_fac_2.z);    // yz
    m_mat_exp_r[5] = exp_r_fac_2.z;                                                    // zz

    // integrated matrix exp. for position update
    Scalar3 xz_fac_r = make_scalar3((Scalar(1.0)+g_r.x)*(Scalar(1.0)+g_r.x)+h_r.x,
                                    (Scalar(1.0)+g_r.y)*(Scalar(1.0)+g_r.y)+h_r.y,
                                    (Scalar(1.0)+g_r.z)*(Scalar(1.0)+g_r.z)+h_r.z);

    m_mat_exp_r_int[0] = m_deltaT*exp_r_fac.x*f_r.x;                                   // xx
    m_mat_exp_r_int[1] = m_deltaT*m_deltaT*nuxy*Scalar(1.0/4.0)
                         *(exp_r_fac.x*f_r.x*(Scalar(1.0)+g_r.x)
                          +exp_r_fac.y*f_r.y*(Scalar(1.0)+g_r.y));                      // xy  
    m_mat_exp_r_int[2] = m_deltaT*m_deltaT*nuxz*Scalar(1.0/4.0)
                         *(exp_r_fac.x*f_r.x*(Scalar(1.0)+g_r.x)
                          +exp_r_fac.z*f_r.z*(Scalar(1.0)+g_r.z))
                        +m_deltaT*m_deltaT*m_deltaT*nuxy*nuyz*Scalar(1.0/32.0)
                         *(exp_r_fac.x*f_r.x*xz_fac_r.x
                          +Scalar(2.0)*exp_r_fac.y*f_r.y*xz_fac_r.y
                          +exp_r_fac.z*f_r.z*xz_fac_r.z);                              // xz
    m_mat_exp_r_int[3] = m_deltaT*exp_r_fac.y*f_r.y;                                   // yy
    m_mat_exp_r_int[4] = m_deltaT*m_deltaT*nuyz*Scalar(1.0/4.0)
                         *(exp_r_fac.y*f_r.y*(Scalar(1.0)+g_r.y)
                          +exp_r_fac.z*f_r.z*(Scalar(1.0)+g_r.z));                     // yz
    m_mat_exp_r_int[5] = m_deltaT*exp_r_fac.z*f_r.z;                                   // zz
    }

void export_TwoStepNPTMTK()
    {
    scope in_npt_mtk = class_<TwoStepNPTMTK, boost::shared_ptr<TwoStepNPTMTK>, bases<IntegrationMethodTwoStep>, boost::noncopyable>
        ("TwoStepNPTMTK", init< boost::shared_ptr<SystemDefinition>,
                       boost::shared_ptr<ParticleGroup>,
                       boost::shared_ptr<ComputeThermo>,
                       Scalar,
                       Scalar,
                       boost::shared_ptr<Variant>,
                       boost::shared_ptr<Variant>,
                       TwoStepNPTMTK::integrationMode>())
        .def("setT", &TwoStepNPTMTK::setT)
        .def("setP", &TwoStepNPTMTK::setP)
        .def("setTau", &TwoStepNPTMTK::setTau)
        .def("setTauP", &TwoStepNPTMTK::setTauP)
        .def("setPartialScale", &TwoStepNPTMTK::setPartialScale)
        ;

    enum_<TwoStepNPTMTK::integrationMode>("integrationMode")
    .value("cubic", TwoStepNPTMTK::cubic)
    .value("orthorhombic", TwoStepNPTMTK::orthorhombic)
    .value("tetragonal", TwoStepNPTMTK::tetragonal)
    .value("triclinic", TwoStepNPTMTK::triclinic)
    ;

    }

#ifdef WIN32
#pragma warning( pop )
#endif
