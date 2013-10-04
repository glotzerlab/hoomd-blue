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

// Maintainer: ndtrung

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;
#include <boost/bind.hpp>
using namespace boost;

#include "FIREEnergyMinimizerRigidGPU.h"
#include "FIREEnergyMinimizerRigidGPU.cuh"
#include "FIREEnergyMinimizerGPU.cuh"
#include "TwoStepNVERigidGPU.h"

/*! \file FIREEnergyMinimizerRigidGPU.h
    \brief Contains code for the FIREEnergyMinimizerRigidGPU class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on (group is a placeholder for now)
    \param dt Time step for MD integrator
    \param reset_and_create_integrator Flag to indicate if resetting and creating integrator are needed
    \post The method is constructed with the given particle data and a NULL profiler.
*/
FIREEnergyMinimizerRigidGPU::FIREEnergyMinimizerRigidGPU(boost::shared_ptr<SystemDefinition> sysdef,
                                                        boost::shared_ptr<ParticleGroup> group,
                                                        Scalar dt,
                                                        bool reset_and_create_integrator)
    :   FIREEnergyMinimizerRigid(sysdef, group, dt, false)
    {
    // only one GPU is supported
    if (!exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating a FIREEnergyMinimizerRigidGPU with no GPUs in the execution configuration" << endl;
        throw std::runtime_error("Error initializing FIREEnergyMinimizerRigidGPU");
        }

    // allocate the sum arrays
    GPUArray<Scalar> sum_pe(1, m_pdata->getExecConf());
    m_sum_pe.swap(sum_pe);
    GPUArray<Scalar> sum_Pt(3, m_pdata->getExecConf());
    m_sum_Pt.swap(sum_Pt);
    GPUArray<Scalar> sum_Pr(3, m_pdata->getExecConf());
    m_sum_Pr.swap(sum_Pr);

    m_block_size = 256;
    m_num_blocks = m_nparticles / m_block_size + 1;
    GPUArray<Scalar> partial_sum_pe(m_num_blocks, m_pdata->getExecConf());
    m_partial_sum_pe.swap(partial_sum_pe);

    if (reset_and_create_integrator)
        {
        reset();

        boost::shared_ptr<TwoStepNVERigidGPU> integrator(new TwoStepNVERigidGPU(sysdef, group));
        addIntegrationMethod(integrator);
        setDeltaT(m_deltaT);
        }
    }

/*! Reset minimizer parameters and zero velocities
*/

void FIREEnergyMinimizerRigidGPU::reset()
    {
    m_converged = false;
    m_n_since_negative =  m_nmin+1;
    m_n_since_start = 0;
    m_alpha = m_alpha_start;
    m_was_reset = true;

    shared_ptr<RigidData> rigid_data = m_sysdef->getRigidData();
    ArrayHandle<Scalar4> vel_handle(rigid_data->getVel(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> angmom_handle(rigid_data->getAngMom(), access_location::device, access_mode::readwrite);
    ArrayHandle<unsigned int> d_body_index_array(m_body_group->getIndexArray(), access_location::device, access_mode::read);

    gpu_rigid_data_arrays d_rdata;
    d_rdata.n_bodies = rigid_data->getNumBodies();
    d_rdata.n_group_bodies = m_n_bodies;
    d_rdata.local_beg = 0;
    d_rdata.local_num = m_n_bodies;

    d_rdata.body_indices = d_body_index_array.data;
    d_rdata.vel = vel_handle.data;
    d_rdata.angmom = angmom_handle.data;

    gpu_fire_rigid_zero_v(d_rdata);
    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    setDeltaT(m_deltaT_set);
    }

/*! \param timestep is the iteration number
*/
void FIREEnergyMinimizerRigidGPU::update(unsigned int timestep)
    {
    if (m_converged)
        return;

    IntegratorTwoStep::update(timestep);

    if (timestep % m_nevery != 0)
        return;

    if (m_n_bodies <= 0)
        {
        m_exec_conf->msg->error() << "FIREENergyMinimizerRigid: There is no rigid body for this integrator" << endl;
        throw runtime_error("Error update for FIREEnergyMinimizerRigid (no rigid body)");
        return;
        }

    Scalar Pt(0.0), Pr(0.0);
    Scalar vnorm(0.0), wnorm(0.0);
    Scalar fnorm(0.0), tnorm(0.0);
    Scalar energy(0.0);

    // compute the total energy on the GPU
    // CPU version is Scalar energy = computePotentialEnergy(timesteps) / Scalar(nparticles);
    {
    if (m_prof)
        m_prof->push(exec_conf, "FIRE rigid compute total energy");

    ArrayHandle<Scalar4> d_net_force(m_pdata->getNetForce(), access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_partial_sum_pe(m_partial_sum_pe, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_sum_pe(m_sum_pe, access_location::device, access_mode::overwrite);
    ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);
    unsigned int group_size = m_group->getIndexArray().getNumElements();

    gpu_fire_compute_sum_pe(d_index_array.data,
                            group_size,
                            d_net_force.data,
                            d_sum_pe.data,
                            d_partial_sum_pe.data,
                            m_block_size,
                            m_num_blocks);

    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    if (m_prof)
        m_prof->pop(exec_conf);
    }

    {
    ArrayHandle<Scalar> h_sum_pe(m_sum_pe, access_location::host, access_mode::read);
    energy = h_sum_pe.data[0] / Scalar(m_nparticles);
    }

    if (m_was_reset)
        {
        m_was_reset = false;
        m_old_energy = energy + Scalar(100000)*m_etol;
        }

    // sum P, vnorm, fnorm
    {
    if (m_prof)
        m_prof->push(exec_conf, "FIRE rigid P, vnorm, fnorm");

    ArrayHandle<unsigned int> d_body_index_array(m_body_group->getIndexArray(), access_location::device, access_mode::read);

    ArrayHandle<Scalar4> vel_handle(m_rigid_data->getVel(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> angvel_handle(m_rigid_data->getAngVel(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> force_handle(m_rigid_data->getForce(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> torque_handle(m_rigid_data->getTorque(), access_location::device, access_mode::read);

    gpu_rigid_data_arrays d_rdata;
    d_rdata.n_bodies = m_rigid_data->getNumBodies();
    d_rdata.n_group_bodies = m_n_bodies;
    d_rdata.local_beg = 0;
    d_rdata.local_num = m_n_bodies;

    d_rdata.body_indices = d_body_index_array.data;
    d_rdata.vel = vel_handle.data;
    d_rdata.angvel = angvel_handle.data;
    d_rdata.force = force_handle.data;
    d_rdata.torque = torque_handle.data;

    ArrayHandle<Scalar> d_sum_Pt(m_sum_Pt, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_sum_Pr(m_sum_Pr, access_location::device, access_mode::overwrite);
    gpu_fire_rigid_compute_sum_all(d_rdata,
                                   d_sum_Pt.data,
                                   d_sum_Pr.data);

    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    if (m_prof)
        m_prof->pop(exec_conf);
    }

    {

    ArrayHandle<Scalar> h_sum_Pt(m_sum_Pt, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_sum_Pr(m_sum_Pr, access_location::host, access_mode::read);

    Pt = h_sum_Pt.data[0];
    vnorm = sqrt(h_sum_Pt.data[1]);
    fnorm = sqrt(h_sum_Pt.data[2]);

    Pr = h_sum_Pr.data[0];
    wnorm = sqrt(h_sum_Pr.data[1]);
    tnorm = sqrt(h_sum_Pr.data[2]);

    }

    //printf("f = %g (%g); w = %g (%g); e = %g (%g); min_steps: %d (%d) \n", fnorm/sqrt(m_sysdef->getNDimensions() * m_n_bodies), m_ftol, wnorm/sqrt(m_sysdef->getNDimensions() * m_n_bodies), m_wtol, fabs(energy-m_old_energy), m_etol, m_n_since_start, m_run_minsteps);

    if ((fnorm/sqrt(m_sysdef->getNDimensions() * m_n_bodies) < m_ftol && wnorm/sqrt(m_sysdef->getNDimensions() * m_n_bodies) < m_wtol  && fabs(energy-m_old_energy) < m_etol) && m_n_since_start >= m_run_minsteps)
        {
        printf("Converged: f = %g (ftol = %g); w= %g (wtol = %g); e = %g (etol = %g)\n", fnorm/sqrt(m_sysdef->getNDimensions() * m_n_bodies), m_ftol,wnorm/sqrt(m_sysdef->getNDimensions() * m_n_bodies), m_wtol, fabs(energy-m_old_energy), m_etol);
        m_converged = true;
        return;
        }


    // Update velocities
    {
    if (m_prof)
        m_prof->push(exec_conf, "FIRE rigid update velocities and angular momenta");

    ArrayHandle<unsigned int> d_body_index_array(m_body_group->getIndexArray(), access_location::device, access_mode::read);

    ArrayHandle<Scalar4> vel_handle(m_rigid_data->getVel(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> angmom_handle(m_rigid_data->getAngMom(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> force_handle(m_rigid_data->getForce(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> torque_handle(m_rigid_data->getTorque(), access_location::device, access_mode::read);

    gpu_rigid_data_arrays d_rdata;
    d_rdata.n_bodies = m_rigid_data->getNumBodies();
    d_rdata.n_group_bodies = m_n_bodies;
    d_rdata.local_beg = 0;
    d_rdata.local_num = m_n_bodies;

    d_rdata.body_indices = d_body_index_array.data;
    d_rdata.vel = vel_handle.data;
    d_rdata.angmom = angmom_handle.data;
    d_rdata.force = force_handle.data;
    d_rdata.torque = torque_handle.data;

    // Scales velocities and angular momenta
    Scalar factor_t, factor_r;
    if (fabs(fnorm) > EPSILON)
        factor_t = Scalar(m_alpha * vnorm / fnorm);
    else
        factor_t = Scalar(1.0);

    if (fabs(tnorm) > EPSILON)
        factor_r = Scalar(m_alpha * wnorm / tnorm);
    else
        factor_r = Scalar(1.0);


    gpu_fire_rigid_update_v(d_rdata,
                            m_alpha,
                            factor_t,
                            factor_r);

    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();


    if (m_prof)
        m_prof->pop(exec_conf);
    }

    Scalar P = Pt + Pr;
    if (P > Scalar(0.0))
        {
        m_n_since_negative++;
        if (m_n_since_negative > m_nmin)
            {
            IntegratorTwoStep::setDeltaT(std::min(m_deltaT * m_finc, m_deltaT_max));
            m_alpha *= m_falpha;
            }
        }
    else if (P <= Scalar(0.0))
        {
        IntegratorTwoStep::setDeltaT(m_deltaT * m_fdec);
        m_alpha = m_alpha_start;
        m_n_since_negative = 0;
        if (m_prof)
            m_prof->push(exec_conf, "FIRE rigid zero velocities");

        ArrayHandle<unsigned int> d_body_index_array(m_body_group->getIndexArray(), access_location::device, access_mode::read);

        ArrayHandle<Scalar4> vel_handle(m_rigid_data->getVel(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> angmom_handle(m_rigid_data->getAngMom(), access_location::device, access_mode::readwrite);

        gpu_rigid_data_arrays d_rdata;
        d_rdata.n_bodies = m_rigid_data->getNumBodies();
        d_rdata.n_group_bodies = m_n_bodies;
        d_rdata.local_beg = 0;
        d_rdata.local_num = m_n_bodies;

        d_rdata.body_indices = d_body_index_array.data;
        d_rdata.vel = vel_handle.data;
        d_rdata.angmom = angmom_handle.data;

        gpu_fire_rigid_zero_v(d_rdata);
        if (exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        if (m_prof)
            m_prof->pop(exec_conf);
        }
    m_n_since_start++;
    m_old_energy = energy;

    }


void export_FIREEnergyMinimizerRigidGPU()
    {
    class_<FIREEnergyMinimizerRigidGPU, boost::shared_ptr<FIREEnergyMinimizerRigidGPU>, bases<FIREEnergyMinimizerRigid>, boost::noncopyable>
        ("FIREEnergyMinimizerRigidGPU", init< boost::shared_ptr<SystemDefinition>, boost::shared_ptr<ParticleGroup>, Scalar >())
        ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif
