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

#include "TwoStepNPTMTK.h"

#ifdef ENABLE_MPI
#include "Communicator.h"
#endif 

/*! \file TwoStepNPTMTK.h
    \brief Contains code for the TwoStepNPTMTK class
*/

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

    // precalculate box lengths
    const BoxDim& global_box = m_pdata->getGlobalBox();
    m_L = global_box.getL();

    m_V = m_L.x*m_L.y*m_L.z;   // volume

    // set initial state
    IntegratorVariables v = getIntegratorVariables();

    // choose dummy values for the current temp and pressure
    m_curr_group_T = 0.0;
    m_curr_P = 0.0;

    if (!restartInfoTestValid(v, "npt_mtk", 2))
        {
        v.type = "npt_mtk";
        v.variable.resize(5,Scalar(0.0));
        setValidRestart(false);
        }
    else
        setValidRestart(true);

    setIntegratorVariables(v);

    m_log_names.resize(2);
    m_log_names[0] = "npt_mtk_thermostat_energy";
    m_log_names[1] = "npt_mtk_barostat_energy";
    }

TwoStepNPTMTK::~TwoStepNPTMTK()
    {
    m_exec_conf->msg->notice(5) << "Destroying TwoStepNPTMTK" << endl;
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per the Nose-Hoover
     thermostat and Anderson barostat
*/
void TwoStepNPTMTK::integrateStepOne(unsigned int timestep)
    {
#ifdef ENABLE_MPI
    unsigned int group_size = m_group->getNumLocalMembers();
#else
    unsigned int group_size = m_group->getNumMembers();
#endif

    if (group_size == 0)
        return;

    // compute the current thermodynamic properties
    m_thermo_group->compute(timestep);

    // compute temperature for the next half time step
    m_curr_group_T = m_thermo_group->getTemperature();

    // compute pressure for the next half time step
    assert(m_mode == cubic || m_mode == orthorhombic || m_mode == tetragonal);
    if (m_mode == cubic)
        {
        m_curr_P = m_thermo_group->getPressure();

        // if it is not valid, assume that the current pressure is the set pressure (this should only happen in very
        // rare circumstances, usually at the start of the simulation before things are initialize)
        if (isnan(m_curr_P))
            m_curr_P = m_P->getValue(timestep);
        }
    else if (m_mode == orthorhombic || m_mode == tetragonal)
        {
        PressureTensor P;
        P = m_thermo_group->getPressureTensor();

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
        }

    // profile this step
    if (m_prof)
        m_prof->push("NPT MTK step 1");

    IntegratorVariables v = getIntegratorVariables();
    Scalar& eta = v.variable[0];  // Thermostat variable
    Scalar& xi = v.variable[1];   // Thermostat velocity
    Scalar& nux = v.variable[2];  // Barostat variable for x-direction
    Scalar& nuy = v.variable[3];  // Barostat variable for y-direction
    Scalar& nuz = v.variable[4];  // Barostat variable for z-direction

    {
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);

    // advance barostat (nux, nuy, nuz) half a time step
    Scalar W = m_thermo_group->getNDOF()*m_T->getValue(timestep)*m_tauP*m_tauP;
    Scalar mtk_term = Scalar(1.0/2.0)*m_deltaT*m_curr_group_T/W;
    if (m_mode == cubic)
        {
        nux += Scalar(1.0/2.0)*m_deltaT*m_V/W*(m_curr_P - m_P->getValue(timestep)) + mtk_term;
        nuy = nuz = nux;
        }
    else if (m_mode == tetragonal)
        {
        nux += Scalar(1.0/2.0)*m_deltaT*m_V/W*(m_curr_P_diag.x - m_P->getValue(timestep)) + mtk_term;
        nuy += Scalar(1.0/2.0)*m_deltaT*m_V/W*((m_curr_P_diag.y+m_curr_P_diag.z)/Scalar(2.0) - m_P->getValue(timestep)) + mtk_term;
        nuz = nuy;
        }
    else if (m_mode == orthorhombic)
        {
        nux += Scalar(1.0/2.0)*m_deltaT*m_V/W*(m_curr_P_diag.x - m_P->getValue(timestep)) + mtk_term;
        nuy += Scalar(1.0/2.0)*m_deltaT*m_V/W*(m_curr_P_diag.y - m_P->getValue(timestep)) + mtk_term;
        nuz += Scalar(1.0/2.0)*m_deltaT*m_V/W*(m_curr_P_diag.z - m_P->getValue(timestep)) + mtk_term;
        }

    // advance thermostat (xi, eta) half a time step
    Scalar xi_prime = xi + Scalar(1.0/4.0)*m_deltaT/m_tau/m_tau*(m_curr_group_T/m_T->getValue(timestep) - Scalar(1.0));
    xi = xi_prime+ Scalar(1.0/4.0)*m_deltaT/(m_tau*m_tau)*(m_curr_group_T/m_T->getValue(timestep)*
          exp(-xi_prime*m_deltaT) - Scalar(1.0));

    eta += Scalar(1.0/2.0)*xi_prime*m_deltaT;

    // precompute loop invariant quantities
    Scalar mtk_term_2 = (nux+nuy+nuz)/m_thermo_group->getNDOF();
    Scalar3 v_fac = make_scalar3(Scalar(1.0/4.0)*(nux+mtk_term_2),
                                 Scalar(1.0/4.0)*(nuy+mtk_term_2),
                                 Scalar(1.0/4.0)*(nuz+mtk_term_2));
    m_exp_v_fac = make_scalar3(exp(-v_fac.x*m_deltaT),
                               exp(-v_fac.y*m_deltaT),
                               exp(-v_fac.z*m_deltaT));
    Scalar3 exp_v_fac_2 = make_scalar3(exp(-(Scalar(2.0)*v_fac.x+Scalar(1.0/2.0)*xi_prime)*m_deltaT),
                               exp(-(Scalar(2.0)*v_fac.y+Scalar(1.0/2.0)*xi_prime)*m_deltaT),
                               exp(-(Scalar(2.0)*v_fac.z+Scalar(1.0/2.0)*xi_prime)*m_deltaT));

    Scalar3 r_fac = make_scalar3(Scalar(1.0/2.0)*nux,
                                 Scalar(1.0/2.0)*nuy,
                                 Scalar(1.0/2.0)*nuz);
    Scalar3 exp_r_fac = make_scalar3(exp(r_fac.x*m_deltaT),
                                     exp(r_fac.y*m_deltaT),
                                     exp(r_fac.z*m_deltaT));

    // Coefficients of sinh(x)/x = a_0 + a_2 * x^2 + a_4 * x^4 + a_6 * x^6 + a_8 * x^8 + a_10 * x^10
    const Scalar a[] = {Scalar(1.0), Scalar(1.0/6.0), Scalar(1.0/120.0), Scalar(1.0/5040.0), Scalar(1.0/362880.0), Scalar(1.0/39916800.0)};

    Scalar3 arg_v = v_fac*m_deltaT;
    Scalar3 arg_r = r_fac*m_deltaT;

    m_sinhx_fac_v = make_scalar3(0.0,0.0,0.0);
    Scalar3 sinhx_fac_r = make_scalar3(0.0,0.0,0.0);
    Scalar3 term_v = make_scalar3(1.0,1.0,1.0);
    Scalar3 term_r = make_scalar3(1.0,1.0,1.0);

    for (unsigned int i = 0; i < 6; i++)
        {
        m_sinhx_fac_v += a[i] * term_v;
        sinhx_fac_r += a[i] * term_r;
        term_v = term_v * arg_v * arg_v;
        term_r = term_r * arg_r * arg_r;
        }

    // perform the first half step of NPT
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);

        Scalar3 vel = make_scalar3(h_vel.data[j].x, h_vel.data[j].y, h_vel.data[j].z);
        vel = vel*exp_v_fac_2 + Scalar(1.0/2.0)*m_deltaT*h_accel.data[j]*m_exp_v_fac*m_sinhx_fac_v;
        h_vel.data[j].x = vel.x; h_vel.data[j].y = vel.y; h_vel.data[j].z = vel.z;

        Scalar3 r = make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);
        r = r*exp_r_fac*exp_r_fac + vel*exp_r_fac*sinhx_fac_r*m_deltaT;
        h_pos.data[j].x = r.x; h_pos.data[j].y = r.y; h_pos.data[j].z = r.z;
        }
    } // end of GPUArray scope

    // advance box lengths
    Scalar3 box_len_scale = make_scalar3(exp(nux*m_deltaT), exp(nuy*m_deltaT), exp(nuz*m_deltaT));
    m_L = m_L*box_len_scale;

#ifdef ENABLE_MPI
    if (m_comm)
        {
        assert(m_comm->getMPICommunicator()->size());
        // broadcast integrator variables from rank 0 to other processors
        broadcast(*m_comm->getMPICommunicator(), eta, 0);
        broadcast(*m_comm->getMPICommunicator(), xi, 0);
        broadcast(*m_comm->getMPICommunicator(), nux, 0);
        broadcast(*m_comm->getMPICommunicator(), nuy, 0);
        broadcast(*m_comm->getMPICommunicator(), nuz, 0);

        // broadcast box dimensions
        broadcast(*m_comm->getMPICommunicator(), m_L, 0);
        }
#endif

    // calculate volume
    m_V = m_L.x*m_L.y*m_L.z;

    m_pdata->setGlobalBoxL(m_L);

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
#ifdef ENABLE_MPI
    unsigned int group_size = m_group->getNumLocalMembers();
#else
    unsigned int group_size = m_group->getNumMembers();
#endif
    if (group_size == 0)
        return;

    const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();

   // profile this step
    if (m_prof)
        m_prof->push("NPT MTK step 2");

    IntegratorVariables v = getIntegratorVariables();
    Scalar& eta = v.variable[0];  // Thermostat variable
    Scalar& xi = v.variable[1];   // Thermostat velocity
    Scalar& nux = v.variable[2];  // Barostat variable for x-direction
    Scalar& nuy = v.variable[3];  // Barostat variable for y-direction
    Scalar& nuz = v.variable[4];  // Barostat variable for z-direction

    Scalar mtk_term_2 = (nux+nuy+nuz)/m_thermo_group->getNDOF();
    Scalar3 v_fac_2 = make_scalar3(Scalar(1.0/2.0)*(nux+mtk_term_2),
                                 Scalar(1.0/2.0)*(nuy+mtk_term_2),
                                 Scalar(1.0/2.0)*(nuz+mtk_term_2));
 
    Scalar3 exp_v_fac_2 = make_scalar3(exp(-v_fac_2.x*m_deltaT),
                               exp(-v_fac_2.y*m_deltaT),
                               exp(-v_fac_2.z*m_deltaT));

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

        // then, update the velocity
        Scalar3 vel = make_scalar3(h_vel.data[j].x, h_vel.data[j].y, h_vel.data[j].z);
        vel = vel*exp_v_fac_2 + Scalar(1.0/2.0)*m_deltaT*m_exp_v_fac*h_accel.data[j]*m_sinhx_fac_v;
        h_vel.data[j].x = vel.x; h_vel.data[j].y = vel.y; h_vel.data[j].z = vel.z;

        // reduce E_kin
        m_v2_sum += m*(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z);
        }

#ifdef ENABLE_MPI
    if (m_comm)
        {
        assert(m_comm->getMPICommunicator()->size());
        // broadcast integrator variables from rank 0 to other processors
        m_v2_sum = all_reduce(*m_comm->getMPICommunicator(), m_v2_sum, std::plus<double>());
        } 
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
        m_prof->push("NPT MTK step 2");

    // compute temperature for the next half time step
    m_curr_group_T = m_thermo_group->getTemperature();

    // compute pressure for the next half time step
    assert(m_mode == cubic || m_mode == orthorhombic || m_mode == tetragonal);
    if (m_mode == cubic)
        {
        m_curr_P = m_thermo_group->getPressure();

        // if it is not valid, assume that the current pressure is the set pressure (this should only happen in very
        // rare circumstances, usually at the start of the simulation before things are initialize)
        if (isnan(m_curr_P))
            m_curr_P = m_P->getValue(timestep);
        }
    else if (m_mode == orthorhombic || m_mode == tetragonal)
        {
        PressureTensor P;
        P = m_thermo_group->getPressureTensor();

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
        }

    // advance barostat (nux, nuy, nuz) half a time step
    Scalar W = m_thermo_group->getNDOF()*m_T->getValue(timestep)*m_tauP*m_tauP;
    Scalar mtk_term = Scalar(1.0/2.0)*m_deltaT*m_curr_group_T/W;
    if (m_mode == cubic)
        {
        nux += Scalar(1.0/2.0)*m_deltaT*m_V/W*(m_curr_P - m_P->getValue(timestep)) + mtk_term;
        nuy = nuz = nux;
        }
    else if (m_mode == tetragonal)
        {
        nux += Scalar(1.0/2.0)*m_deltaT*m_V/W*(m_curr_P_diag.x - m_P->getValue(timestep)) + mtk_term;
        nuy += Scalar(1.0/2.0)*m_deltaT*m_V/W*((m_curr_P_diag.y+m_curr_P_diag.z)/Scalar(2.0) - m_P->getValue(timestep)) + mtk_term;
        nuz = nuy;
        }
    else if (m_mode == orthorhombic)
        {
        nux += Scalar(1.0/2.0)*m_deltaT*m_V/W*(m_curr_P_diag.x - m_P->getValue(timestep)) + mtk_term;
        nuy += Scalar(1.0/2.0)*m_deltaT*m_V/W*(m_curr_P_diag.y - m_P->getValue(timestep)) + mtk_term;
        nuz += Scalar(1.0/2.0)*m_deltaT*m_V/W*(m_curr_P_diag.z - m_P->getValue(timestep)) + mtk_term;
        }

#ifdef ENABLE_MPI
    if (m_comm)
        {
        assert(m_comm->getMPICommunicator()->size());
        // broadcast integrator variables from rank 0 to other processors
        broadcast(*m_comm->getMPICommunicator(), eta, 0);
        broadcast(*m_comm->getMPICommunicator(), xi, 0);
        broadcast(*m_comm->getMPICommunicator(), nux, 0);
        broadcast(*m_comm->getMPICommunicator(), nuy, 0);
        broadcast(*m_comm->getMPICommunicator(), nuz, 0);
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
        Scalar& nux = v.variable[2];
        Scalar& nuy = v.variable[3];
        Scalar& nuz = v.variable[4];

        Scalar W = m_thermo_group->getNDOF()*m_T->getValue(timestep)*m_tauP*m_tauP;
        Scalar barostat_energy = Scalar(0.0);
        barostat_energy = W*(nux*nux+nuy*nuy+nuz*nuz) / Scalar(2.0);

        return barostat_energy;
        }
    else
        return Scalar(0);
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
    ;

    }

#ifdef WIN32
#pragma warning( pop )
#endif
