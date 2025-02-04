// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

//
#ifdef ENABLE_HIP

#include "TwoStepRATTLELangevin.h"
#include "TwoStepRATTLELangevinGPU.cuh"
#include "TwoStepRATTLENVEGPU.cuh"

#include "hoomd/Autotuner.h"

#ifdef ENABLE_MPI
#include "hoomd/HOOMDMPI.h"
#endif

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace md
    {
//! Implements Langevin dynamics on the GPU
/*! GPU accelerated version of TwoStepLangevin

    \ingroup updaters
*/
template<class Manifold>
class PYBIND11_EXPORT TwoStepRATTLELangevinGPU : public TwoStepRATTLELangevin<Manifold>
    {
    public:
    //! Constructs the integration method and associates it with the system
    TwoStepRATTLELangevinGPU(std::shared_ptr<SystemDefinition> sysdef,
                             std::shared_ptr<ParticleGroup> group,
                             Manifold manifold,
                             std::shared_ptr<Variant> T,
                             Scalar tolerance);

    virtual ~TwoStepRATTLELangevinGPU() { };

    //! Performs the first step of the integration
    virtual void integrateStepOne(uint64_t timestep);

    //! Performs the second step of the integration
    virtual void integrateStepTwo(uint64_t timestep);

    //! Includes the RATTLE forces to the virial/net force
    virtual void includeRATTLEForce(uint64_t timestep);

    protected:
    unsigned int m_block_size;       //!< block size for partial sum memory
    unsigned int m_num_blocks;       //!< number of memory blocks reserved for partial sum memory
    GPUArray<Scalar> m_partial_sum1; //!< memory space for partial sum over bd energy transfers
    GPUArray<Scalar> m_sum;          //!< memory space for sum over bd energy transfers

    /// Autotuner for block size (step one kernel).
    std::shared_ptr<Autotuner<1>> m_tuner_one;

    /// Autotuner for block size (force kernel).
    std::shared_ptr<Autotuner<1>> m_tuner_force;

    /// Autotuner for block size (angular step one kernel).
    std::shared_ptr<Autotuner<1>> m_tuner_angular_one;
    };

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per the
   velocity verlet method.

    This method is copied directly from TwoStepNVEGPU::integrateStepOne() and reimplemented here to
   avoid multiple.
*/

template<class Manifold>
TwoStepRATTLELangevinGPU<Manifold>::TwoStepRATTLELangevinGPU(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<ParticleGroup> group,
    Manifold manifold,
    std::shared_ptr<Variant> T,
    Scalar tolerance)
    : TwoStepRATTLELangevin<Manifold>(sysdef, group, manifold, T, tolerance)
    {
    if (!this->m_exec_conf->isCUDAEnabled())
        {
        this->m_exec_conf->msg->error()
            << "Creating a TwoStepRATTLELangevinGPU while CUDA is disabled" << std::endl;
        throw std::runtime_error("Error initializing TwoStepRATTLELangevinGPU");
        }

    // allocate the sum arrays
    GPUArray<Scalar> sum(1, this->m_exec_conf);
    m_sum.swap(sum);

    // initialize the partial sum array
    m_block_size = 256;
    unsigned int group_size = this->m_group->getNumMembers();
    m_num_blocks = group_size / m_block_size + 1;
    GPUArray<Scalar> partial_sum1(m_num_blocks, this->m_exec_conf);
    m_partial_sum1.swap(partial_sum1);

    m_tuner_one.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf)},
                                       this->m_exec_conf,
                                       "rattle_langevin_nve"));
    m_tuner_force.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf)},
                                         this->m_exec_conf,
                                         "rattle_langevin_force"));
    m_tuner_angular_one.reset(
        new Autotuner<1>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf)},
                         this->m_exec_conf,
                         "rattle_langevin_angular",
                         5,
                         true));
    this->m_autotuners.insert(this->m_autotuners.end(),
                              {m_tuner_one, m_tuner_force, m_tuner_angular_one});
    }
template<class Manifold>
void TwoStepRATTLELangevinGPU<Manifold>::integrateStepOne(uint64_t timestep)
    {
    if (this->m_box_changed)
        {
        if (!this->m_manifold.fitsInsideBox(this->m_pdata->getGlobalBox()))
            {
            throw std::runtime_error("Parts of the manifold are outside the box");
            }
        this->m_box_changed = false;
        }

    // access all the needed data
    ArrayHandle<unsigned int> d_index_array(this->m_group->getIndexArray(),
                                            access_location::device,
                                            access_mode::read);

    ArrayHandle<Scalar4> d_pos(this->m_pdata->getPositions(),
                               access_location::device,
                               access_mode::readwrite);
    ArrayHandle<Scalar4> d_vel(this->m_pdata->getVelocities(),
                               access_location::device,
                               access_mode::readwrite);
    ArrayHandle<Scalar3> d_accel(this->m_pdata->getAccelerations(),
                                 access_location::device,
                                 access_mode::read);
    ArrayHandle<int3> d_image(this->m_pdata->getImages(),
                              access_location::device,
                              access_mode::readwrite);

    this->m_exec_conf->beginMultiGPU();
    m_tuner_one->begin();
    // perform the update on the GPU
    kernel::gpu_rattle_nve_step_one(d_pos.data,
                                    d_vel.data,
                                    d_accel.data,
                                    d_image.data,
                                    d_index_array.data,
                                    this->m_group->getGPUPartition(),
                                    this->m_pdata->getBox(),
                                    this->m_deltaT,
                                    false,
                                    0,
                                    this->m_tuner_one->getParam()[0]);

    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_one->end();
    this->m_exec_conf->endMultiGPU();

    if (this->m_aniso)
        {
        // first part of angular update
        ArrayHandle<Scalar4> d_orientation(this->m_pdata->getOrientationArray(),
                                           access_location::device,
                                           access_mode::readwrite);
        ArrayHandle<Scalar4> d_angmom(this->m_pdata->getAngularMomentumArray(),
                                      access_location::device,
                                      access_mode::readwrite);
        ArrayHandle<Scalar4> d_net_torque(this->m_pdata->getNetTorqueArray(),
                                          access_location::device,
                                          access_mode::read);
        ArrayHandle<Scalar3> d_inertia(this->m_pdata->getMomentsOfInertiaArray(),
                                       access_location::device,
                                       access_mode::read);

        this->m_exec_conf->beginMultiGPU();
        m_tuner_angular_one->begin();

        kernel::gpu_rattle_nve_angular_step_one(d_orientation.data,
                                                d_angmom.data,
                                                d_inertia.data,
                                                d_net_torque.data,
                                                d_index_array.data,
                                                this->m_group->getGPUPartition(),
                                                this->m_deltaT,
                                                1.0,
                                                m_tuner_angular_one->getParam()[0]);

        m_tuner_angular_one->end();
        this->m_exec_conf->endMultiGPU();

        if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }
    }

/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1 on the GPU
*/
template<class Manifold>
void TwoStepRATTLELangevinGPU<Manifold>::integrateStepTwo(uint64_t timestep)
    {
    const GlobalArray<Scalar4>& net_force = this->m_pdata->getNetForce();

    // get the dimensionality of the system
    const unsigned int D = this->m_sysdef->getNDimensions();

    ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_gamma(this->m_gamma, access_location::device, access_mode::read);
    ArrayHandle<Scalar3> d_gamma_r(this->m_gamma_r, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_index_array(this->m_group->getIndexArray(),
                                            access_location::device,
                                            access_mode::read);

        {
        ArrayHandle<Scalar> d_partial_sumBD(m_partial_sum1,
                                            access_location::device,
                                            access_mode::overwrite);
        ArrayHandle<Scalar> d_sumBD(this->m_sum, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar4> d_pos(this->m_pdata->getPositions(),
                                   access_location::device,
                                   access_mode::read);
        ArrayHandle<Scalar4> d_vel(this->m_pdata->getVelocities(),
                                   access_location::device,
                                   access_mode::readwrite);
        ArrayHandle<Scalar3> d_accel(this->m_pdata->getAccelerations(),
                                     access_location::device,
                                     access_mode::readwrite);
        ArrayHandle<unsigned int> d_tag(this->m_pdata->getTags(),
                                        access_location::device,
                                        access_mode::read);

        unsigned int group_size = this->m_group->getNumMembers();
        m_num_blocks = group_size / m_block_size + 1;

        // perform the update on the GPU
        kernel::rattle_langevin_step_two_args args(d_gamma.data,
                                                   this->m_gamma.getNumElements(),
                                                   (*this->m_T)(timestep),
                                                   this->m_tolerance,
                                                   timestep,
                                                   this->m_sysdef->getSeed(),
                                                   d_sumBD.data,
                                                   d_partial_sumBD.data,
                                                   m_block_size,
                                                   m_num_blocks,
                                                   this->m_noiseless_t,
                                                   this->m_noiseless_r,
                                                   this->m_tally,
                                                   this->m_exec_conf->dev_prop);

        kernel::gpu_rattle_langevin_step_two<Manifold>(d_pos.data,
                                                       d_vel.data,
                                                       d_accel.data,
                                                       d_tag.data,
                                                       d_index_array.data,
                                                       group_size,
                                                       d_net_force.data,
                                                       args,
                                                       this->m_manifold,
                                                       this->m_deltaT,
                                                       D);

        if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        if (this->m_aniso)
            {
            // second part of angular update
            ArrayHandle<Scalar4> d_orientation(this->m_pdata->getOrientationArray(),
                                               access_location::device,
                                               access_mode::read);
            ArrayHandle<Scalar4> d_angmom(this->m_pdata->getAngularMomentumArray(),
                                          access_location::device,
                                          access_mode::readwrite);
            ArrayHandle<Scalar4> d_net_torque(this->m_pdata->getNetTorqueArray(),
                                              access_location::device,
                                              access_mode::read);
            ArrayHandle<Scalar3> d_inertia(this->m_pdata->getMomentsOfInertiaArray(),
                                           access_location::device,
                                           access_mode::read);

            unsigned int group_size = this->m_group->getNumMembers();
            kernel::gpu_rattle_langevin_angular_step_two(d_pos.data,
                                                         d_orientation.data,
                                                         d_angmom.data,
                                                         d_inertia.data,
                                                         d_net_torque.data,
                                                         d_index_array.data,
                                                         d_gamma_r.data,
                                                         d_tag.data,
                                                         group_size,
                                                         args,
                                                         this->m_deltaT,
                                                         D,
                                                         1.0);

            if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }
        }

    if (this->m_tally)
        {
        ArrayHandle<Scalar> h_sumBD(m_sum, access_location::host, access_mode::read);
#ifdef ENABLE_MPI
        if (this->m_sysdef->isDomainDecomposed())
            {
            MPI_Allreduce(MPI_IN_PLACE,
                          &h_sumBD.data[0],
                          1,
                          MPI_HOOMD_SCALAR,
                          MPI_SUM,
                          this->m_exec_conf->getMPICommunicator());
            }
#endif
        this->m_reservoir_energy -= h_sumBD.data[0] * this->m_deltaT;
        this->m_extra_energy_overdeltaT = 0.5 * h_sumBD.data[0];
        }
    }

template<class Manifold>
void TwoStepRATTLELangevinGPU<Manifold>::includeRATTLEForce(uint64_t timestep)
    {
    // access all the needed data
    const GlobalArray<Scalar4>& net_force = this->m_pdata->getNetForce();
    const GlobalArray<Scalar>& net_virial = this->m_pdata->getNetVirial();
    ArrayHandle<Scalar4> d_pos(this->m_pdata->getPositions(),
                               access_location::device,
                               access_mode::read);
    ArrayHandle<Scalar4> d_vel(this->m_pdata->getVelocities(),
                               access_location::device,
                               access_mode::read);
    ArrayHandle<Scalar3> d_accel(this->m_pdata->getAccelerations(),
                                 access_location::device,
                                 access_mode::readwrite);

    ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> d_net_virial(net_virial, access_location::device, access_mode::readwrite);

    ArrayHandle<unsigned int> d_index_array(this->m_group->getIndexArray(),
                                            access_location::device,
                                            access_mode::read);

    size_t net_virial_pitch = net_virial.getPitch();

    // perform the update on the GPU
    this->m_exec_conf->beginMultiGPU();
    m_tuner_force->begin();
    kernel::gpu_include_rattle_force_nve<Manifold>(d_pos.data,
                                                   d_vel.data,
                                                   d_accel.data,
                                                   d_net_force.data,
                                                   d_net_virial.data,
                                                   d_index_array.data,
                                                   this->m_group->getGPUPartition(),
                                                   net_virial_pitch,
                                                   this->m_manifold,
                                                   this->m_tolerance,
                                                   this->m_deltaT,
                                                   false,
                                                   m_tuner_force->getParam()[0]);

    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    m_tuner_force->end();
    this->m_exec_conf->endMultiGPU();
    }

namespace detail
    {
template<class Manifold>
void export_TwoStepRATTLELangevinGPU(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<TwoStepRATTLELangevinGPU<Manifold>,
                     TwoStepRATTLELangevin<Manifold>,
                     std::shared_ptr<TwoStepRATTLELangevinGPU<Manifold>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ParticleGroup>,
                            Manifold,
                            std::shared_ptr<Variant>,
                            Scalar>());
    }
    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif // ENABLE_HIP
