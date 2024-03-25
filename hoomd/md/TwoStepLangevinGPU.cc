// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "TwoStepLangevinGPU.h"
#include "TwoStepLangevinGPU.cuh"
#include "TwoStepNVEGPU.cuh"

#ifdef ENABLE_MPI
#include "hoomd/HOOMDMPI.h"
#endif

using namespace std;

namespace hoomd
    {
namespace md
    {
/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param T Temperature set point as a function of time
*/
TwoStepLangevinGPU::TwoStepLangevinGPU(std::shared_ptr<SystemDefinition> sysdef,
                                       std::shared_ptr<ParticleGroup> group,
                                       std::shared_ptr<Variant> T)
    : TwoStepLangevin(sysdef, group, T)
    {
    if (!m_exec_conf->isCUDAEnabled())
        {
        throw std::runtime_error("Cannot create TwoStepLangevinGPU on a CPU device.");
        }

    // allocate the sum arrays
    GPUArray<Scalar> sum(1, m_exec_conf);
    m_sum.swap(sum);

    // initialize the partial sum array
    m_block_size = 256;
    unsigned int group_size = m_group->getNumMembers();
    m_num_blocks = group_size / m_block_size + 1;
    GPUArray<Scalar> partial_sum1(m_num_blocks, m_exec_conf);
    m_partial_sum1.swap(partial_sum1);

    m_tuner_one.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                       m_exec_conf,
                                       "langevin_nve"));
    m_tuner_angular_one.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                               m_exec_conf,
                                               "langevin_angular",
                                               5,
                                               true));
    m_autotuners.insert(m_autotuners.end(), {m_tuner_one, m_tuner_angular_one});
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per the
   velocity verlet method.

    This method is copied directly from TwoStepNVEGPU::integrateStepOne() and reimplemented here to
   avoid multiple.
*/
void TwoStepLangevinGPU::integrateStepOne(uint64_t timestep)
    {
    // access all the needed data
    BoxDim box = m_pdata->getBox();
    ArrayHandle<unsigned int> d_index_array(m_group->getIndexArray(),
                                            access_location::device,
                                            access_mode::read);

    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(),
                               access_location::device,
                               access_mode::readwrite);
    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(),
                               access_location::device,
                               access_mode::readwrite);
    ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(),
                                 access_location::device,
                                 access_mode::readwrite);
    ArrayHandle<int3> d_image(m_pdata->getImages(),
                              access_location::device,
                              access_mode::readwrite);

    m_exec_conf->beginMultiGPU();
    m_tuner_one->begin();
    // perform the update on the GPU
    kernel::gpu_nve_step_one(d_pos.data,
                             d_vel.data,
                             d_accel.data,
                             d_image.data,
                             d_index_array.data,
                             m_group->getGPUPartition(),
                             box,
                             m_deltaT,
                             false,
                             0,
                             false,
                             m_tuner_one->getParam()[0]);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_one->end();
    m_exec_conf->endMultiGPU();

    if (m_aniso)
        {
        // first part of angular update
        ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(),
                                           access_location::device,
                                           access_mode::readwrite);
        ArrayHandle<Scalar4> d_angmom(m_pdata->getAngularMomentumArray(),
                                      access_location::device,
                                      access_mode::readwrite);
        ArrayHandle<Scalar4> d_net_torque(m_pdata->getNetTorqueArray(),
                                          access_location::device,
                                          access_mode::read);
        ArrayHandle<Scalar3> d_inertia(m_pdata->getMomentsOfInertiaArray(),
                                       access_location::device,
                                       access_mode::read);

        m_exec_conf->beginMultiGPU();
        m_tuner_angular_one->begin();

        kernel::gpu_nve_angular_step_one(d_orientation.data,
                                         d_angmom.data,
                                         d_inertia.data,
                                         d_net_torque.data,
                                         d_index_array.data,
                                         m_group->getGPUPartition(),
                                         m_deltaT,
                                         1.0,
                                         m_tuner_angular_one->getParam()[0]);

        m_tuner_angular_one->end();
        m_exec_conf->endMultiGPU();

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }
    }

/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1 on the GPU
*/
void TwoStepLangevinGPU::integrateStepTwo(uint64_t timestep)
    {
    const GlobalArray<Scalar4>& net_force = m_pdata->getNetForce();

    // get the dimensionality of the system
    const unsigned int D = m_sysdef->getNDimensions();

    ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_gamma(m_gamma, access_location::device, access_mode::read);
    ArrayHandle<Scalar3> d_gamma_r(m_gamma_r, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_index_array(m_group->getIndexArray(),
                                            access_location::device,
                                            access_mode::read);

        {
        ArrayHandle<Scalar> d_partial_sumBD(m_partial_sum1,
                                            access_location::device,
                                            access_mode::overwrite);
        ArrayHandle<Scalar> d_sumBD(m_sum, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(),
                                   access_location::device,
                                   access_mode::read);
        ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(),
                                   access_location::device,
                                   access_mode::readwrite);
        ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(),
                                     access_location::device,
                                     access_mode::readwrite);
        ArrayHandle<unsigned int> d_tag(m_pdata->getTags(),
                                        access_location::device,
                                        access_mode::read);

        unsigned int group_size = m_group->getNumMembers();
        m_num_blocks = group_size / m_block_size + 1;

        // perform the update on the GPU
        kernel::langevin_step_two_args args(d_gamma.data,
                                            (unsigned int)m_gamma.getNumElements(),
                                            m_T->operator()(timestep),
                                            timestep,
                                            m_sysdef->getSeed(),
                                            d_sumBD.data,
                                            d_partial_sumBD.data,
                                            m_block_size,
                                            m_num_blocks,
                                            m_noiseless_t,
                                            m_noiseless_r,
                                            m_tally,
                                            m_exec_conf->dev_prop);

        kernel::gpu_langevin_step_two(d_pos.data,
                                      d_vel.data,
                                      d_accel.data,
                                      d_tag.data,
                                      d_index_array.data,
                                      group_size,
                                      d_net_force.data,
                                      args,
                                      m_deltaT,
                                      D);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        if (m_aniso)
            {
            // second part of angular update
            ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(),
                                               access_location::device,
                                               access_mode::read);
            ArrayHandle<Scalar4> d_angmom(m_pdata->getAngularMomentumArray(),
                                          access_location::device,
                                          access_mode::readwrite);
            ArrayHandle<Scalar4> d_net_torque(m_pdata->getNetTorqueArray(),
                                              access_location::device,
                                              access_mode::read);
            ArrayHandle<Scalar3> d_inertia(m_pdata->getMomentsOfInertiaArray(),
                                           access_location::device,
                                           access_mode::read);

            unsigned int group_size = m_group->getNumMembers();
            gpu_langevin_angular_step_two(d_pos.data,
                                          d_orientation.data,
                                          d_angmom.data,
                                          d_inertia.data,
                                          d_net_torque.data,
                                          d_index_array.data,
                                          d_gamma_r.data,
                                          d_tag.data,
                                          group_size,
                                          args,
                                          m_deltaT,
                                          D,
                                          1.0);

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }
        }

    if (m_tally)
        {
        ArrayHandle<Scalar> h_sumBD(m_sum, access_location::host, access_mode::read);
#ifdef ENABLE_MPI
        if (m_sysdef->isDomainDecomposed())
            {
            MPI_Allreduce(MPI_IN_PLACE,
                          &h_sumBD.data[0],
                          1,
                          MPI_HOOMD_SCALAR,
                          MPI_SUM,
                          m_exec_conf->getMPICommunicator());
            }
#endif
        m_reservoir_energy -= h_sumBD.data[0] * m_deltaT;
        m_extra_energy_overdeltaT = 0.5 * h_sumBD.data[0];
        }
    }

namespace detail
    {
void export_TwoStepLangevinGPU(pybind11::module& m)
    {
    pybind11::class_<TwoStepLangevinGPU, TwoStepLangevin, std::shared_ptr<TwoStepLangevinGPU>>(
        m,
        "TwoStepLangevinGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ParticleGroup>,
                            std::shared_ptr<Variant>>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
