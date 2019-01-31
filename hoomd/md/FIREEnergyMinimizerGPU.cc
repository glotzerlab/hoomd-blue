// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: askeys



#include "FIREEnergyMinimizerGPU.h"
#include "FIREEnergyMinimizerGPU.cuh"

namespace py = pybind11;
using namespace std;

/*! \file FIREEnergyMinimizerGPU.h
    \brief Contains code for the FIREEnergyMinimizerGPU class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param dt Default step size
*/
FIREEnergyMinimizerGPU::FIREEnergyMinimizerGPU(std::shared_ptr<SystemDefinition> sysdef, Scalar dt)
    :   FIREEnergyMinimizer(sysdef, dt)
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

    unsigned int num_blocks = 0;
    for (auto method = m_methods.begin(); method != m_methods.end(); ++method)
        {
        std::shared_ptr<ParticleGroup> current_group = (*method)->getGroup();

        unsigned int group_size = current_group->getIndexArray().getNumElements();
        num_blocks = std::max(num_blocks, group_size);
        }

    num_blocks = num_blocks/m_block_size + 1;
    GPUArray<Scalar> partial_sum1(num_blocks, m_exec_conf);
    m_partial_sum1.swap(partial_sum1);
    GPUArray<Scalar> partial_sum2(num_blocks, m_exec_conf);
    m_partial_sum2.swap(partial_sum2);
    GPUArray<Scalar> partial_sum3(num_blocks, m_exec_conf);
    m_partial_sum3.swap(partial_sum3);

    reset();
    }

/*! \param timesteps is the iteration number
*/
void FIREEnergyMinimizerGPU::update(unsigned int timestep)
    {

    if (m_converged)
        return;

    IntegratorTwoStep::update(timestep);

    Scalar Pt(0.0);  //translational power
    Scalar Pr(0.0); //rotational power
    Scalar vnorm(0.0);
    Scalar fnorm(0.0);
    Scalar energy(0.0);
    Scalar tnorm(0.0);
    Scalar wnorm(0.0);

    // compute the total energy on the GPU
    // CPU version is Scalar energy = computePotentialEnergy(timesteps)/Scalar(group_size);

    if (m_prof)
        m_prof->push(m_exec_conf, "FIRE compute total energy");

    unsigned int total_group_size = 0;

    for (auto method = m_methods.begin(); method != m_methods.end(); ++method)
        {
        std::shared_ptr<ParticleGroup> current_group = (*method)->getGroup();

        unsigned int group_size = current_group->getIndexArray().getNumElements();
        total_group_size += group_size;

        ArrayHandle< unsigned int > d_index_array(current_group->getIndexArray(), access_location::device, access_mode::read);

            {
            ArrayHandle<Scalar4> d_net_force(m_pdata->getNetForce(), access_location::device, access_mode::read);
            ArrayHandle<Scalar> d_partial_sumE(m_partial_sum1, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar> d_sumE(m_sum, access_location::device, access_mode::overwrite);

            unsigned int num_blocks = group_size/m_block_size + 1;
            gpu_fire_compute_sum_pe(d_index_array.data,
                                    group_size,
                                    d_net_force.data,
                                    d_sumE.data,
                                    d_partial_sumE.data,
                                    m_block_size,
                                    num_blocks);

            if(m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }

        ArrayHandle<Scalar> h_sumE(m_sum, access_location::host, access_mode::read);
        energy += h_sumE.data[0];
        }

    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        MPI_Allreduce(MPI_IN_PLACE, &energy, 1, MPI_HOOMD_SCALAR, MPI_SUM, m_exec_conf->getMPICommunicator());
        MPI_Allreduce(MPI_IN_PLACE, &total_group_size, 1, MPI_INT, MPI_SUM, m_exec_conf->getMPICommunicator());
        }
    #endif

    m_energy_total = energy;
    energy /= (Scalar) total_group_size;

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

    bool aniso = false;

    for (auto method = m_methods.begin(); method != m_methods.end(); ++method)
        {
        std::shared_ptr<ParticleGroup> current_group = (*method)->getGroup();

        unsigned int group_size = current_group->getIndexArray().getNumElements();
        ArrayHandle< unsigned int > d_index_array(current_group->getIndexArray(), access_location::device, access_mode::read);

            {
            ArrayHandle<Scalar> d_partial_sum_P(m_partial_sum1, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar> d_partial_sum_vsq(m_partial_sum2, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar> d_partial_sum_fsq(m_partial_sum3, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar> d_sum(m_sum3, access_location::device, access_mode::overwrite);
            ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::read);
            ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::read);

            unsigned int num_blocks = group_size/m_block_size + 1;

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
                                     num_blocks);

            if(m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }

        ArrayHandle<Scalar> h_sum(m_sum3, access_location::host, access_mode::read);
        Pt += h_sum.data[0];
        vnorm += h_sum.data[1];
        fnorm += h_sum.data[2];

        if ((*method)->getAnisotropic())
            {
            aniso = true;

                {
                ArrayHandle<Scalar> d_partial_sum_Pr(m_partial_sum1, access_location::device, access_mode::overwrite);
                ArrayHandle<Scalar> d_partial_sum_wnorm(m_partial_sum2, access_location::device, access_mode::overwrite);
                ArrayHandle<Scalar> d_partial_sum_tsq(m_partial_sum3, access_location::device, access_mode::overwrite);
                ArrayHandle<Scalar> d_sum(m_sum3, access_location::device, access_mode::overwrite);
                ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::read);
                ArrayHandle<Scalar4> d_angmom(m_pdata->getAngularMomentumArray(), access_location::device, access_mode::read);
                ArrayHandle<Scalar4> d_net_torque(m_pdata->getNetTorqueArray(), access_location::device, access_mode::read);
                ArrayHandle<Scalar3> d_inertia(m_pdata->getMomentsOfInertiaArray(), access_location::device, access_mode::read);

                unsigned int num_blocks = group_size/m_block_size + 1;

                gpu_fire_compute_sum_all_angular(m_pdata->getN(),
                                         d_orientation.data,
                                         d_inertia.data,
                                         d_angmom.data,
                                         d_net_torque.data,
                                         d_index_array.data,
                                         group_size,
                                         d_sum.data,
                                         d_partial_sum_Pr.data,
                                         d_partial_sum_wnorm.data,
                                         d_partial_sum_tsq.data,
                                         m_block_size,
                                         num_blocks);

                if(m_exec_conf->isCUDAErrorCheckingEnabled())
                    CHECK_CUDA_ERROR();
                }
            ArrayHandle<Scalar> h_sum(m_sum3, access_location::host, access_mode::read);
            Pr += h_sum.data[0];
            wnorm += h_sum.data[1];
            tnorm += h_sum.data[2];
            }

        }

    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        MPI_Allreduce(MPI_IN_PLACE, &fnorm, 1, MPI_HOOMD_SCALAR, MPI_SUM, m_exec_conf->getMPICommunicator());
        MPI_Allreduce(MPI_IN_PLACE, &vnorm, 1, MPI_HOOMD_SCALAR, MPI_SUM, m_exec_conf->getMPICommunicator());
        MPI_Allreduce(MPI_IN_PLACE, &Pt, 1, MPI_HOOMD_SCALAR, MPI_SUM, m_exec_conf->getMPICommunicator());

        if (aniso)
            {
            MPI_Allreduce(MPI_IN_PLACE, &tnorm, 1, MPI_HOOMD_SCALAR, MPI_SUM, m_exec_conf->getMPICommunicator());
            MPI_Allreduce(MPI_IN_PLACE, &wnorm, 1, MPI_HOOMD_SCALAR, MPI_SUM, m_exec_conf->getMPICommunicator());
            MPI_Allreduce(MPI_IN_PLACE, &Pr, 1, MPI_HOOMD_SCALAR, MPI_SUM, m_exec_conf->getMPICommunicator());
            }
        }
    #endif

    vnorm = sqrt(vnorm);
    fnorm = sqrt(fnorm);
    wnorm = sqrt(wnorm);
    tnorm = sqrt(tnorm);

    if (m_prof)
        m_prof->pop(m_exec_conf);

    unsigned int ndof = m_sysdef->getNDimensions()*total_group_size;
    m_exec_conf->msg->notice(10) << "FIRE fnorm " << fnorm << " tnorm " << tnorm << " delta_E " << energy-m_old_energy << std::endl;
    m_exec_conf->msg->notice(10) << "FIRE vnorm " << vnorm << " tnorm " << wnorm << std::endl;
    m_exec_conf->msg->notice(10) << "FIRE Pt " << Pt << " Pr " << Pr << std::endl;

    if ((fnorm/sqrt(Scalar(ndof)) < m_ftol && wnorm/sqrt(Scalar(ndof)) < m_wtol  && fabs(energy-m_old_energy) < m_etol) && m_n_since_start >= m_run_minsteps)
        {
        m_converged = true;
        m_exec_conf->msg->notice(4) << "FIRE converged in timestep " << timestep << std::endl;
        return;
        }

    //update velocities

    if (m_prof)
        m_prof->push(m_exec_conf, "FIRE update velocities");

    Scalar factor_t;
    if (fabs(fnorm) > EPSILON)
        factor_t = m_alpha*vnorm/fnorm;
    else
        factor_t = 1.0;

    Scalar factor_r = 0.0;

    if (fabs(tnorm) > EPSILON)
        factor_r = m_alpha * wnorm / tnorm;
    else
        factor_r = 1.0;

    for (auto method = m_methods.begin(); method != m_methods.end(); ++method)
        {
        std::shared_ptr<ParticleGroup> current_group = (*method)->getGroup();

        unsigned int group_size = current_group->getIndexArray().getNumElements();
        ArrayHandle< unsigned int > d_index_array(current_group->getIndexArray(), access_location::device, access_mode::read);

        ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::read);

        gpu_fire_update_v(d_vel.data,
                          d_accel.data,
                          d_index_array.data,
                          group_size,
                          m_alpha,
                          factor_t);

        if(m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        if ((*method)->getAnisotropic())
            {
            ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::read);
            ArrayHandle<Scalar4> d_net_torque(m_pdata->getNetTorqueArray(), access_location::device, access_mode::read);
            ArrayHandle<Scalar4> d_angmom(m_pdata->getAngularMomentumArray(), access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar3> d_inertia(m_pdata->getMomentsOfInertiaArray(), access_location::device, access_mode::read);

            gpu_fire_update_angmom(d_net_torque.data,
                          d_orientation.data,
                          d_inertia.data,
                          d_angmom.data,
                          d_index_array.data,
                          group_size,
                          m_alpha,
                          factor_r);

            if(m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }
        }

    if (m_prof)
        m_prof->pop(m_exec_conf);

    Scalar P = Pt + Pr;

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

        m_exec_conf->msg->notice(6) << "FIRE zero velocities" << std::endl;

        for (auto method = m_methods.begin(); method != m_methods.end(); ++method)
            {
            std::shared_ptr<ParticleGroup> current_group = (*method)->getGroup();
            unsigned int group_size = current_group->getIndexArray().getNumElements();
            ArrayHandle< unsigned int > d_index_array(current_group->getIndexArray(), access_location::device, access_mode::read);

            ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);

            gpu_fire_zero_v(d_vel.data,
                            d_index_array.data,
                            group_size);
            if(m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();

            if ((*method)->getAnisotropic())
                {
                ArrayHandle<Scalar4> d_angmom(m_pdata->getAngularMomentumArray(), access_location::device, access_mode::readwrite);
                gpu_fire_zero_angmom(d_angmom.data,
                                d_index_array.data,
                                group_size);
                if(m_exec_conf->isCUDAErrorCheckingEnabled())
                    CHECK_CUDA_ERROR();

                }
            }

        if (m_prof)
            m_prof->pop(m_exec_conf);
        }

    m_n_since_start++;
    m_old_energy = energy;
    }


void export_FIREEnergyMinimizerGPU(py::module& m)
    {
    py::class_<FIREEnergyMinimizerGPU, std::shared_ptr<FIREEnergyMinimizerGPU> >(m, "FIREEnergyMinimizerGPU", py::base<FIREEnergyMinimizer>())
        .def(py::init< std::shared_ptr<SystemDefinition>, Scalar >())
        ;
    }
