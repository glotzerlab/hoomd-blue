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

// Maintainer: askeys

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include "FIREEnergyMinimizer.h"
#include "TwoStepNVE.h"

// windows feels the need to #define min and max
#ifdef WIN32
#undef min
#undef max
#endif


/*! \file FIREEnergyMinimizer.h
    \brief Contains code for the FIREEnergyMinimizer class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param dt maximum step size
    \param reset_and_create_integrator Set to true to completely initialize this class

    \post The method is constructed with the given particle data and a NULL profiler.
*/
FIREEnergyMinimizer::FIREEnergyMinimizer(boost::shared_ptr<SystemDefinition> sysdef,
                                         boost::shared_ptr<ParticleGroup> group,
                                         Scalar dt,
                                         bool reset_and_create_integrator)
    :   IntegratorTwoStep(sysdef, dt),
        m_group(group),
        m_nmin(5),
        m_finc(Scalar(1.1)),
        m_fdec(Scalar(0.5)),
        m_alpha_start(Scalar(0.1)),
        m_falpha(Scalar(0.99)),
        m_ftol(Scalar(1e-1)),
        m_etol(Scalar(1e-3)),
        m_deltaT_max(dt),
        m_deltaT_set(dt/Scalar(10.0)),
        m_run_minsteps(10)
    {
    m_exec_conf->msg->notice(5) << "Constructing FIREEnergyMinimizer" << endl;

    // sanity check
    assert(m_sysdef);
    assert(m_pdata);
    if (reset_and_create_integrator)
        {
        reset();
        //createIntegrator();
        boost::shared_ptr<TwoStepNVE> integrator(new TwoStepNVE(sysdef, group));
        addIntegrationMethod(integrator);
        setDeltaT(m_deltaT_set);
        }
    }

FIREEnergyMinimizer::~FIREEnergyMinimizer()
    {
    m_exec_conf->msg->notice(5) << "Destroying FIREEnergyMinimizer" << endl;
    }

//void FIREEnergyMinimizer::createIntegrator()
//    {
//    boost::shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(m_sysdef, 0, m_pdata->getN()-1));
//    boost::shared_ptr<ParticleGroup> group_all(new ParticleGroup(m_sysdef, selector_all));
//    boost::shared_ptr<TwoStepNVE> integrator(new TwoStepNVE(m_sysdef, group_all));
//    addIntegrationMethod(integrator);
//    setDeltaT(m_deltaT);
//    }

/*! \param dt is the new timestep to set

The timestep is used by the underlying NVE integrator to advance the particles.
*/
void FIREEnergyMinimizer::setDeltaT(Scalar dt)
    {
    IntegratorTwoStep::setDeltaT(dt);
    }


/*! \param finc is the new fractional increase to set
*/
void FIREEnergyMinimizer::setFinc(Scalar finc)
    {
    if (!(finc > 1.0))
        {
        m_exec_conf->msg->error() << "integrate.mode_minimize_fire: fractional increase in timestep should be > 1" << endl;
        throw runtime_error("Error setting parameters for FIREEnergyMinimizer");
        }
        m_finc = finc;
    }

/*! \param fdec is the new fractional decrease to set
*/
void FIREEnergyMinimizer::setFdec(Scalar fdec)
    {
    if (!(fdec < 1.0 && fdec >= 0.0))
        {
        m_exec_conf->msg->error() << "integrate.mode_minimize_fire: fractional decrease in timestep should be between 0 and 1" << endl;
        throw runtime_error("Error setting parameters for FIREEnergyMinimizer");
        }
        m_fdec = fdec;
    }

/*! \param alpha_start is the new initial coupling parameter to set

The coupling parameter "alpha" enters into the equations of motion as
v = v*(1-alpha) + alpha*(f_unit*|v|).  Thus, the stronger the coupling, the
more important the "f dot v" term.  When the search direction is successful
for > Nmin steps alpha is decreased by falpha.
*/
void FIREEnergyMinimizer::setAlphaStart(Scalar alpha_start)
    {
    if (!(alpha_start < 1.0 && alpha_start > 0.0))
        {
        m_exec_conf->msg->error() << "integrate.mode_minimize_fire: alpha_start should be between 0 and 1" << endl;
        throw runtime_error("Error setting parameters for FIREEnergyMinimizer");
        }
        m_alpha_start = alpha_start;
    }

/*! \param falpha is the fractional decrease in alpha upon finding a valid search direction

The coupling parameter "alpha" enters into the equations of motion as
v = v*(1-alpha) + alpha*(f_unit*|v|).  Thus, the stronger the coupling, the
more important the "f dot v" term.  When the search direction is successful
for > Nmin steps alpha is decreased by falpha.
*/
void FIREEnergyMinimizer::setFalpha(Scalar falpha)
    {
    if (!(falpha < 1.0 && falpha > 0.0))
        {
        m_exec_conf->msg->error() << "integrate.mode_minimize_fire: falpha should be between 0 and 1" << endl;
        throw runtime_error("Error setting parameters for FIREEnergyMinimizer");
        }
        m_falpha = falpha;
    }

void FIREEnergyMinimizer::reset()
    {
    m_converged = false;
    m_n_since_negative = m_nmin+1;
    m_n_since_start = 0;
    m_alpha = m_alpha_start;
    m_was_reset = true;

    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);

    unsigned int n = m_pdata->getN();
    for (unsigned int i=0; i<n; i++)
        {
        h_vel.data[i].x = Scalar(0.0);
        h_vel.data[i].y = Scalar(0.0);
        h_vel.data[i].z = Scalar(0.0);
        }
    setDeltaT(m_deltaT_set);
    m_pdata->notifyParticleSort();
    }

/*! \param timesteps is the current timestep
*/
void FIREEnergyMinimizer::update(unsigned int timesteps)
    {
    if (m_converged)
        return;

    unsigned int group_size = m_group->getNumMembers();
    if (group_size == 0)
        return;

    IntegratorTwoStep::update(timesteps);

    Scalar P(0.0);
    Scalar vnorm(0.0);
    Scalar fnorm(0.0);

    // Calculate the per-particle potential energy over particles in the group
    Scalar energy(0.0);

    {
    const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();
    ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);

    // total potential energy
    double pe_total = 0.0;
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        pe_total += (double)h_net_force.data[j].w;
        }
    energy = pe_total/Scalar(group_size);
    }


    if (m_was_reset)
        {
        m_was_reset = false;
        m_old_energy = energy + Scalar(100000)*m_etol;
        }

    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_accel(m_pdata->getAccelerations(), access_location::host, access_mode::readwrite);

    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        P += h_accel.data[j].x*h_vel.data[j].x + h_accel.data[j].y*h_vel.data[j].y + h_accel.data[j].z*h_vel.data[j].z;
        fnorm += h_accel.data[j].x*h_accel.data[j].x+h_accel.data[j].y*h_accel.data[j].y+h_accel.data[j].z*h_accel.data[j].z;
        vnorm += h_vel.data[j].x*h_vel.data[j].x+ h_vel.data[j].y*h_vel.data[j].y + h_vel.data[j].z*h_vel.data[j].z;
        }

    fnorm = sqrt(fnorm);
    vnorm = sqrt(vnorm);

    if ((fnorm/sqrt(Scalar(m_sysdef->getNDimensions()*group_size)) < m_ftol && fabs(energy-m_old_energy) < m_etol) && m_n_since_start >= m_run_minsteps)
        {
        m_converged = true;
        return;
        }

    Scalar invfnorm = 1.0/fnorm;
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        h_vel.data[j].x = h_vel.data[j].x*(1.0-m_alpha) + m_alpha*h_accel.data[j].x*invfnorm*vnorm;
        h_vel.data[j].y = h_vel.data[j].y*(1.0-m_alpha) + m_alpha*h_accel.data[j].y*invfnorm*vnorm;
        h_vel.data[j].z = h_vel.data[j].z*(1.0-m_alpha) + m_alpha*h_accel.data[j].z*invfnorm*vnorm;
        }

    if (P > Scalar(0.0))
        {
        m_n_since_negative++;
        if (m_n_since_negative > m_nmin)
            {
            IntegratorTwoStep::setDeltaT(std::min(m_deltaT*m_finc, m_deltaT_max));
            m_alpha *= m_falpha;
            }
        }
    else if (P <= Scalar(0.0))
        {
        IntegratorTwoStep::setDeltaT(m_deltaT*m_fdec);
        m_alpha = m_alpha_start;
        m_n_since_negative = 0;
        for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
            {
            unsigned int j = m_group->getMemberIndex(group_idx);
            h_vel.data[j].x = Scalar(0.0);
            h_vel.data[j].y = Scalar(0.0);
            h_vel.data[j].z = Scalar(0.0);
            }
        }
    m_n_since_start++;
    m_old_energy = energy;

    }


void export_FIREEnergyMinimizer()
    {
    class_<FIREEnergyMinimizer, boost::shared_ptr<FIREEnergyMinimizer>, bases<IntegratorTwoStep>, boost::noncopyable>
        ("FIREEnergyMinimizer", init< boost::shared_ptr<SystemDefinition>, boost::shared_ptr<ParticleGroup>, Scalar >())
        .def("reset", &FIREEnergyMinimizer::reset)
        .def("setDeltaT", &FIREEnergyMinimizer::setDeltaT)
        .def("hasConverged", &FIREEnergyMinimizer::hasConverged)
        .def("setNmin", &FIREEnergyMinimizer::setNmin)
        .def("setFinc", &FIREEnergyMinimizer::setFinc)
        .def("setFdec", &FIREEnergyMinimizer::setFdec)
        .def("setAlphaStart", &FIREEnergyMinimizer::setAlphaStart)
        .def("setFalpha", &FIREEnergyMinimizer::setFalpha)
        .def("setFtol", &FIREEnergyMinimizer::setFtol)
        .def("setEtol", &FIREEnergyMinimizer::setEtol)
        .def("setMinSteps", &FIREEnergyMinimizer::setMinSteps)
        ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif
