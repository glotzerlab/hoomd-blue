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
#include "TwoStepNVTRigid.h"
#include <boost/python.hpp>

using namespace std;
using namespace boost::python;

/*! \file TwoStepNVTRigid.cc
    \brief Contains code for the TwoStepNVTRigid class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
 \param group The group of particles this integration method is to work on
 \param suffix Suffix to attach to the end of log quantity names
 \param thermo compute for thermodynamic quantities
 \param T Controlled temperature
 \param tau Time constant
 \param tchain Number of thermostats in the thermostat chain
 \param iter Number of inner iterations to update the thermostats
 \param suffix Suffix to attach to the end of log quantity names
*/
TwoStepNVTRigid::TwoStepNVTRigid(boost::shared_ptr<SystemDefinition> sysdef,
                                 boost::shared_ptr<ParticleGroup> group,
                                 boost::shared_ptr<ComputeThermo> thermo,
                                 const std::string& suffix,
                                 boost::shared_ptr<Variant> T,
                                 Scalar tau,
                                 unsigned int tchain,
                                 unsigned int iter)
    : TwoStepNHRigid(sysdef, group, suffix, tchain, 0, iter)
    {
    m_exec_conf->msg->notice(5) << "Constructing TwoStepNVTRigid" << endl;

    m_thermo_group = thermo;
    m_temperature = T;

    m_tstat = true;
    if (tau <= 0.0)
        m_exec_conf->msg->warning() << "integrate.nvt_rigid: tau set less than or equal to 0.0" << endl;
    m_tfreq = 1.0 / tau;

    // set initial state
    IntegratorVariables v = getIntegratorVariables();

    if (!restartInfoTestValid(v, "nvt_rigid", 6))
        {
        // reset the integrator variable
        v.type = "nvt_rigid";
        v.variable.resize(6, Scalar(0.0));
        setValidRestart(false);
        }
    else
        setValidRestart(true);

    setIntegratorVariables(v);

    m_log_names.push_back(string("nvt_rigid_xi_t") + suffix);
    m_log_names.push_back(string("nvt_rigid_xi_r") + suffix);
    }

TwoStepNVTRigid::~TwoStepNVTRigid()
    {
    m_exec_conf->msg->notice(5) << "Destroying TwoStepNVTRigid" << endl;
    }

/* Compute the initial forces/torques
*/
void TwoStepNVTRigid::setup()
    {
    TwoStepNHRigid::setup();

    if (m_n_bodies <= 0)
        return;

    // retrieve integrator variables from restart files
    IntegratorVariables v = getIntegratorVariables();
    m_eta_t[0] = v.variable[0];
    m_eta_r[0] = v.variable[1];
    m_eta_dot_r[0] = v.variable[2];
    m_eta_dot_t[0] = v.variable[3];
    m_f_eta_r[0] = v.variable[4];
    m_f_eta_t[0] = v.variable[5];

    // initialize thermostat chain positions, velocites, forces
    Scalar kt = m_boltz * m_temperature->getValue(0);
    Scalar t_mass = kt / (m_tfreq * m_tfreq);
    m_q_t[0] = m_nf_t * t_mass;
    m_q_r[0] = m_nf_r * t_mass;
    for (unsigned int i = 1; i < m_tchain; i++)
        {
        m_q_t[i] = m_q_r[i] = t_mass;
        m_f_eta_t[i] = (m_q_t[i-1] * m_eta_dot_t[i-1] * m_eta_dot_t[i-1] - kt)/m_q_t[i];
        m_f_eta_r[i] = (m_q_r[i-1] * m_eta_dot_r[i-1] * m_eta_dot_r[i-1] - kt)/m_q_r[i];
        }
    }

/*! \param quantity Name of the log quantity to get
    \param timestep Current time step of the simulation
    \param my_quantity_flag passed as false, changed to true if quanity logged here
*/

Scalar TwoStepNVTRigid::getLogValue(const std::string& quantity, unsigned int timestep, bool &my_quantity_flag)
    {
    if (quantity == m_log_names[0])
        {
        my_quantity_flag = true;
        if (m_eta_dot_t)
            return m_eta_dot_t[0];
        else
            return Scalar(0);
        }
    else if (quantity == m_log_names[1])
        {
        my_quantity_flag = true;
        if (m_eta_dot_r)
            return m_eta_dot_r[0];
        else
            return Scalar(0);
        }
    else
        return Scalar(0);
    }

void export_TwoStepNVTRigid()
    {
    class_<TwoStepNVTRigid, boost::shared_ptr<TwoStepNVTRigid>, bases<TwoStepNHRigid>, boost::noncopyable>
        ("TwoStepNVTRigid", init< boost::shared_ptr<SystemDefinition>,
        boost::shared_ptr<ParticleGroup>,
        boost::shared_ptr<ComputeThermo>,
        const std::string&,
        boost::shared_ptr<Variant>,
        Scalar, unsigned int, unsigned int >());
    }

