// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: askeys



#include "FIREEnergyMinimizerGPU.h"
#include "FIREEnergyMinimizerGPU.cuh"
#include "TwoStepNVEGPU.h"

namespace py = pybind11;
using namespace std;

/*! \file FIREEnergyMinimizerGPU.h
    \brief Contains code for the FIREEnergyMinimizerGPU class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param dt Default step size

    \post The method is constructed with the given particle data and a NULL profiler.
*/
FIREEnergyMinimizerGPU::FIREEnergyMinimizerGPU(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<ParticleGroup> group, Scalar dt)
    :   FIREEnergyMinimizer(sysdef, group, dt, false)
    {

    // only one GPU is supported
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating a FIREEnergyMinimizer with CUDA disabled" << endl;
        throw std::runtime_error("Error initializing FIREEnergyMinimizer");
        }

    // allocate the sum arrays
    GPUArray<Scalar> sum(1, m_exec_conf);
    m_sum.swap(sum);
    GPUArray<Scalar> sum3(3, m_exec_conf);
    m_sum3.swap(sum3);

    // initialize the partial sum arrays
    m_block_size = 256; //128;
    unsigned int group_size = m_group->getIndexArray().getNumElements();
    m_num_blocks = group_size / m_block_size + 1;
    GPUArray<Scalar> partial_sum1(m_num_blocks, m_exec_conf);
    m_partial_sum1.swap(partial_sum1);
    GPUArray<Scalar> partial_sum2(m_num_blocks, m_exec_conf);
    m_partial_sum2.swap(partial_sum2);
    GPUArray<Scalar> partial_sum3(m_num_blocks, m_exec_conf);
    m_partial_sum3.swap(partial_sum3);

    reset();
    createIntegrator();
    }

void FIREEnergyMinimizerGPU::createIntegrator()
    {
//   std::shared_ptr<ParticleSelector> selector_all(new ParticleSelectorTag(m_sysdef, 0, m_pdata->getN()-1));
//    std::shared_ptr<ParticleGroup> group_all(new ParticleGroup(m_sysdef, selector_all));
    std::shared_ptr<TwoStepNVEGPU> integrator(new TwoStepNVEGPU(m_sysdef, m_group));
    addIntegrationMethod(integrator);
    setDeltaT(m_deltaT);
    }

void FIREEnergyMinimizerGPU::reset()
    {
    m_converged = false;
    m_n_since_negative =  m_nmin+1;
    m_n_since_start = 0;
    m_alpha = m_alpha_start;
    m_was_reset = true;

    {
        ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::readwrite);

        ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);
        unsigned int group_size = m_group->getIndexArray().getNumElements();
        gpu_fire_zero_v( d_vel.data,
                    d_index_array.data,
                    group_size);


        if(m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

    }

    setDeltaT(m_deltaT_set);
    m_pdata->notifyParticleSort();
    }

/*! \param timesteps is the iteration number
*/
void FIREEnergyMinimizerGPU::update(unsigned int timesteps)
    {

    if (m_converged)
        return;

    IntegratorTwoStep::update(timesteps);

    Scalar P(0.0);
    Scalar vnorm(0.0);
    Scalar fnorm(0.0);
    Scalar energy(0.0);

    // compute the total energy on the GPU
    // CPU version is Scalar energy = computePotentialEnergy(timesteps)/Scalar(group_size);

    if (m_prof)
        m_prof->push(m_exec_conf, "FIRE compute total energy");

    unsigned int group_size = m_group->getIndexArray().getNumElements();
    ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);

        {
        ArrayHandle<Scalar4> d_net_force(m_pdata->getNetForce(), access_location::device, access_mode::read);
        ArrayHandle<Scalar> d_partial_sumE(m_partial_sum1, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar> d_sumE(m_sum, access_location::device, access_mode::overwrite);

        gpu_fire_compute_sum_pe(d_index_array.data,
                                group_size,
                                d_net_force.data,
                                d_sumE.data,
                                d_partial_sumE.data,
                                m_block_size,
                                m_num_blocks);

        if(m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    ArrayHandle<Scalar> h_sumE(m_sum, access_location::host, access_mode::read);
    energy = h_sumE.data[0]/Scalar(group_size);

    if (m_prof)
        m_prof->pop(m_exec_conf);



    if (m_was_reset)
        {
        m_was_reset = false;
        m_old_energy = energy + Scalar(100000)*m_etol;
        }

    //sum P, vnorm, fnorm

    if (m_prof)
        m_prof->push(m_exec_conf, "FIRE P, vnorm, fnorm");

        {
        ArrayHandle<Scalar> d_partial_sum_P(m_partial_sum1, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar> d_partial_sum_vsq(m_partial_sum2, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar> d_partial_sum_fsq(m_partial_sum3, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar> d_sum(m_sum3, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::read);
        ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::read);


        gpu_fire_compute_sum_all(m_pdata->getN(),
                                 d_vel.data,
                                 d_accel.data,
                                 d_index_array.data,
                                 group_size,
                                 d_sum.data,
                                 d_partial_sum_P.data,
                                 d_partial_sum_vsq.data,
                                 d_partial_sum_fsq.data,
                                 m_block_size,
                                 m_num_blocks);

        if(m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    ArrayHandle<Scalar> h_sum(m_sum3, access_location::host, access_mode::read);
    P = h_sum.data[0];
    vnorm = sqrt(h_sum.data[1]);
    fnorm = sqrt(h_sum.data[2]);

    if (m_prof)
        m_prof->pop(m_exec_conf);


    if ((fnorm/sqrt(Scalar(m_sysdef->getNDimensions()*group_size)) < m_ftol && fabs(energy-m_old_energy) < m_etol) && m_n_since_start >= m_run_minsteps)
        {
        m_converged = true;
        return;
        }

    //update velocities

    if (m_prof)
        m_prof->push(m_exec_conf, "FIRE update velocities");

    Scalar invfnorm = 1.0/fnorm;


    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::read);

    gpu_fire_update_v(d_vel.data,
                      d_accel.data,
                      d_index_array.data,
                      group_size,
                      m_alpha,
                      vnorm,
                      invfnorm);

    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    if (m_prof)
        m_prof->pop(m_exec_conf);


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
        if (m_prof)
            m_prof->push(m_exec_conf, "FIRE zero velocities");

        gpu_fire_zero_v(d_vel.data,
                        d_index_array.data,
                        group_size);

        if(m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        if (m_prof)
            m_prof->pop(m_exec_conf);
        }

    m_n_since_start++;
    m_old_energy = energy;
    }


void export_FIREEnergyMinimizerGPU(py::module& m)
    {
    py::class_<FIREEnergyMinimizerGPU, std::shared_ptr<FIREEnergyMinimizerGPU> >(m, "FIREEnergyMinimizerGPU", py::base<FIREEnergyMinimizer>())
        .def(py::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>, Scalar >())
        ;
    }
