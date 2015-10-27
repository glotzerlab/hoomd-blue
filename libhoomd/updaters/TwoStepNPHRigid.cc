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

// Maintainer: ndtrung

#include "QuaternionMath.h"
#include "TwoStepNPHRigid.h"
#include <boost/python.hpp>
#include <math.h>

using namespace std;
using namespace boost::python;


/*! \file TwoStepNPHRigid.cc
    \brief Contains code for the TwoStepNPHRigid class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param thermo_group ComputeThermo to compute thermo properties of the integrated \a group
    \param thermo_all ComputeThermo to compute the pressure of the entire system
    \param suffix Suffix to attach to the end of log quantity names
    \param tauP NPH pressure period
    \param P Pressure set point
    \param couple Coupling mode
    \param flags Barostatted simulation box degrees of freedom
    \param pchain Number of thermostats coupled with the barostat
    \param iter Number of inner iterations to update the thermostats
*/
TwoStepNPHRigid::TwoStepNPHRigid(boost::shared_ptr<SystemDefinition> sysdef,
                                 boost::shared_ptr<ParticleGroup> group,
                                 boost::shared_ptr<ComputeThermo> thermo_group,
                                 boost::shared_ptr<ComputeThermo> thermo_all,
                                 const std::string& suffix,
                                 Scalar tauP,
                                 boost::shared_ptr<Variant> P,
                                 couplingMode couple,
                                 unsigned int flags,
                                 unsigned int pchain,
                                 unsigned int iter)
    : TwoStepNHRigid(sysdef, group, suffix, 0, pchain, iter)
    {
    m_exec_conf->msg->notice(5) << "Constructing TwoStepNPHRigid" << endl;

    m_thermo_group = thermo_group;
    m_thermo_all = thermo_all;
    m_partial_scale = false;
    m_pressure = P;

    m_pstat = true;
    if (tauP <= 0.0)
        m_exec_conf->msg->warning() << "integrate.nph_rigid: tauP set less than or equal to 0.0" << endl;
    m_pfreq = 1.0 / tauP;

    m_couple = couple;
    m_flags = flags;

    // set initial state
    IntegratorVariables v = getIntegratorVariables();

    if (!restartInfoTestValid(v, "nph_rigid", 3))
        {
        // reset the integrator variable
        v.type = "nph_rigid";
        v.variable.resize(3, Scalar(0.0));
        setValidRestart(false);
        }
    else
        setValidRestart(true);

    setIntegratorVariables(v);

    m_log_names.push_back(string("nph_rigid_xi_b") + suffix);
    }

TwoStepNPHRigid::~TwoStepNPHRigid()
    {
    m_exec_conf->msg->notice(5) << "Destroying TwoStepNPHRigid" << endl;
    }

/*!
*/
void TwoStepNPHRigid::setup()
    {
    TwoStepNHRigid::setup();

    // retrieve integrator variables from restart files
    IntegratorVariables v = getIntegratorVariables();
    m_eta_b[0] = v.variable[0];
    m_eta_dot_b[0] = v.variable[1];
    m_f_eta_b[0] = v.variable[2];

    // initialize thermostat chain positions, velocites, forces
    Scalar kt = m_boltz;
    Scalar p_mass = kt / (m_pfreq * m_pfreq);
    m_q_b[0] = m_dimension * m_dimension * p_mass;
    for (unsigned int i = 1; i < m_pchain; i++)
        {
        m_q_b[i] = p_mass;
        m_f_eta_b[i] = (m_q_b[i] * m_eta_dot_b[i-1] * m_eta_dot_b[i-1] - kt)/m_q_b[i];
        }

    for (unsigned int i = 0; i < 6; i++)
        {
        m_epsilon_dot[i] = m_epsilon[i] = Scalar(0.0);
        m_epsilon_mass[i] = Scalar(1.0);
        }

    // compute current pressure
    compute_current_pressure(0);

    // compute target pressure
    compute_target_pressure(0);

    // initialize epsilon dot, mtk_term1, mtk_term2
    update_nh_barostat(m_akin_t, m_akin_r);
    }

Scalar TwoStepNPHRigid::getLogValue(const std::string& quantity, unsigned int timestep, bool &my_quantity_flag)
    {
    if (quantity == m_log_names[0])
        {
        my_quantity_flag = true;
        if (m_eta_dot_b)
            return m_eta_dot_b[0];
        else
            return Scalar(0);
        }
    else
        return Scalar(0);
    }

void export_TwoStepNPHRigid()
    {
    class_<TwoStepNPHRigid, boost::shared_ptr<TwoStepNPHRigid>, bases<TwoStepNHRigid>, boost::noncopyable>
        ("TwoStepNPHRigid", init< boost::shared_ptr<SystemDefinition>,
                       boost::shared_ptr<ParticleGroup>,
                       boost::shared_ptr<ComputeThermo>,
                       boost::shared_ptr<ComputeThermo>,
                       const std::string&,
                       Scalar,
                       boost::shared_ptr<Variant>,
                       TwoStepNHRigid::couplingMode,
                       unsigned int,
                       unsigned int,
                       unsigned int >());
    }

