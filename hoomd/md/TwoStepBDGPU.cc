// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "TwoStepBDGPU.h"
#include "TwoStepBDGPU.cuh"

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
TwoStepBDGPU::TwoStepBDGPU(std::shared_ptr<SystemDefinition> sysdef,
                           std::shared_ptr<ParticleGroup> group,
                           std::shared_ptr<Variant> T,
                           bool noiseless_t,
                           bool noiseless_r)
    : TwoStepBD(sysdef, group, T, noiseless_t, noiseless_r)
    {
    if (!m_exec_conf->isCUDAEnabled())
        {
        throw std::runtime_error("Cannot create TwoStepBDGPU on a CPU device.");
        }

    m_tuner.reset(
        new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)}, m_exec_conf, "bd"));
    m_autotuners.push_back(m_tuner);
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward a full time step and velocities are redrawn from the
   proper distribution.
*/
void TwoStepBDGPU::integrateStepOne(uint64_t timestep)
    {
    // access all the needed data
    BoxDim box = m_pdata->getBox();
    ArrayHandle<unsigned int> d_index_array(m_group->getIndexArray(),
                                            access_location::device,
                                            access_mode::read);
    unsigned int group_size = m_group->getNumMembers();
    const unsigned int D = m_sysdef->getNDimensions();
    const GlobalArray<Scalar4>& net_force = m_pdata->getNetForce();

    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(),
                               access_location::device,
                               access_mode::readwrite);
    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(),
                               access_location::device,
                               access_mode::readwrite);
    ArrayHandle<int3> d_image(m_pdata->getImages(),
                              access_location::device,
                              access_mode::readwrite);

    ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_gamma(m_gamma, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);

    // for rotational noise
    ArrayHandle<Scalar3> d_gamma_r(m_gamma_r, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(),
                                       access_location::device,
                                       access_mode::readwrite);
    ArrayHandle<Scalar4> d_torque(m_pdata->getNetTorqueArray(),
                                  access_location::device,
                                  access_mode::readwrite);
    ArrayHandle<Scalar3> d_inertia(m_pdata->getMomentsOfInertiaArray(),
                                   access_location::device,
                                   access_mode::read);
    ArrayHandle<Scalar4> d_angmom(m_pdata->getAngularMomentumArray(),
                                  access_location::device,
                                  access_mode::readwrite);

    kernel::langevin_step_two_args args(d_gamma.data,
                                        static_cast<unsigned int>(m_gamma.getNumElements()),
                                        m_T->operator()(timestep),
                                        timestep,
                                        m_sysdef->getSeed(),
                                        NULL,
                                        NULL,
                                        0,
                                        0,
                                        false,
                                        false,
                                        false,
                                        m_exec_conf->dev_prop);

    bool aniso = m_aniso;

#ifdef __HIP_PLATFORM_NVCC__
    if (m_exec_conf->allConcurrentManagedAccess())
        {
        // prefetch gammas
        auto& gpu_map = m_exec_conf->getGPUIds();
        for (unsigned int idev = 0; idev < m_exec_conf->getNumActiveGPUs(); ++idev)
            {
            cudaMemPrefetchAsync(m_gamma.get(),
                                 sizeof(Scalar) * m_gamma.getNumElements(),
                                 gpu_map[idev]);
            cudaMemPrefetchAsync(m_gamma_r.get(),
                                 sizeof(Scalar) * m_gamma_r.getNumElements(),
                                 gpu_map[idev]);
            }
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }
#endif

    m_exec_conf->beginMultiGPU();
    m_tuner->begin();
    args.block_size = m_tuner->getParam()[0];

    // perform the update on the GPU
    gpu_brownian_step_one(d_pos.data,
                          d_vel.data,
                          d_image.data,
                          box,
                          d_tag.data,
                          d_index_array.data,
                          group_size,
                          d_net_force.data,
                          d_gamma_r.data,
                          d_orientation.data,
                          d_torque.data,
                          d_inertia.data,
                          d_angmom.data,
                          args,
                          aniso,
                          m_deltaT,
                          D,
                          m_noiseless_t,
                          m_noiseless_r,
                          m_group->getGPUPartition());

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    m_tuner->end();
    m_exec_conf->endMultiGPU();
    }

/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1 on the GPU
*/
void TwoStepBDGPU::integrateStepTwo(uint64_t timestep)
    {
    // there is no step 2
    }

namespace detail
    {
void export_TwoStepBDGPU(pybind11::module& m)
    {
    pybind11::class_<TwoStepBDGPU, TwoStepBD, std::shared_ptr<TwoStepBDGPU>>(m, "TwoStepBDGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ParticleGroup>,
                            std::shared_ptr<Variant>,
                            bool,
                            bool>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
