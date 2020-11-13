// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "TwoStepRATTLELangevinGPU.h"
#include "TwoStepRATTLENVEGPU.cuh"
#include "TwoStepRATTLELangevinGPU.cuh"

#ifdef ENABLE_MPI
#include "hoomd/HOOMDMPI.h"
#endif

namespace py = pybind11;
using namespace std;

/*! \file TwoStepRATTLELangevinGPU.h
    \brief Contains code for the TwoStepLangevinGPU class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param T Temperature set point as a function of time
    \param seed Random seed to use in generating random numbers
    \param use_alpha If true, gamma=alpha*diameter, otherwise use a per-type gamma via setGamma()
    \param alpha Scale factor to convert diameter to gamma
*/
TwoStepRATTLELangevinGPU::TwoStepRATTLELangevinGPU(std::shared_ptr<SystemDefinition> sysdef,
                                       std::shared_ptr<ParticleGroup> group,
                       		           std::shared_ptr<Manifold> manifold,
                                       std::shared_ptr<Variant> T,
                                       unsigned int seed,
                           	           Scalar eta)
    : TwoStepRATTLELangevin(sysdef, group, manifold, T, seed, eta), m_manifoldGPU( manifold->returnL(), manifold->returnR(), manifold->returnSurf() )
    {
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating a TwoStepRATTLELangevinGPU while CUDA is disabled" << endl;
        throw std::runtime_error("Error initializing TwoStepRATTLELangevinGPU");
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

    hipDeviceProp_t dev_prop = m_exec_conf->dev_prop;
    m_tuner_one.reset(new Autotuner(dev_prop.warpSize, dev_prop.maxThreadsPerBlock, dev_prop.warpSize, 5, 100000, "rattle_langevin_nve", this->m_exec_conf));
    m_tuner_angular_one.reset(new Autotuner(dev_prop.warpSize, dev_prop.maxThreadsPerBlock, dev_prop.warpSize, 5, 100000, "rattle_langevin_angular", this->m_exec_conf));
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per the velocity verlet
          method.

    This method is copied directly from TwoStepNVEGPU::integrateStepOne() and reimplemented here to avoid multiple.
*/
void TwoStepRATTLELangevinGPU::integrateStepOne(unsigned int timestep)
    {
    // profile this step
    if (m_prof)
        m_prof->push(m_exec_conf, "RATTLELangevin step 1");

    // access all the needed data
    BoxDim box = m_pdata->getBox();
    ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);

    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::read);
    ArrayHandle<int3> d_image(m_pdata->getImages(), access_location::device, access_mode::readwrite);

    m_exec_conf->beginMultiGPU();
    m_tuner_one->begin();
    // perform the update on the GPU
    gpu_rattle_nve_step_one(d_pos.data,
                     d_vel.data,
                     d_accel.data,
                     d_image.data,
                     d_index_array.data,
                     m_group->getGPUPartition(),
                     box,
                     m_deltaT,
                     false,
                     0,
                     m_tuner_one->getParam());

    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_one->end();
    m_exec_conf->endMultiGPU();

    if (m_aniso)
        {
        // first part of angular update
        ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> d_angmom(m_pdata->getAngularMomentumArray(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> d_net_torque(m_pdata->getNetTorqueArray(), access_location::device, access_mode::read);
        ArrayHandle<Scalar3> d_inertia(m_pdata->getMomentsOfInertiaArray(), access_location::device, access_mode::read);

        m_exec_conf->beginMultiGPU();
        m_tuner_angular_one->begin();

        gpu_rattle_nve_angular_step_one(d_orientation.data,
                                 d_angmom.data,
                                 d_inertia.data,
                                 d_net_torque.data,
                                 d_index_array.data,
                                 m_group->getGPUPartition(),
                                 m_deltaT,
                                 1.0,
                                 m_tuner_angular_one->getParam());

        m_tuner_angular_one->end();
        m_exec_conf->endMultiGPU();

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    }

    // done profiling
    if (m_prof)
        m_prof->pop(m_exec_conf);
    }

/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1 on the GPU
*/
void TwoStepRATTLELangevinGPU::integrateStepTwo(unsigned int timestep)
    {
    const GlobalArray< Scalar4 >& net_force = m_pdata->getNetForce();

    // profile this step
    if (m_prof)
        m_prof->push(m_exec_conf, "RATTLELangevin step 2");

    // get the dimensionality of the system
    const unsigned int D = m_sysdef->getNDimensions();

    ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_gamma(m_gamma, access_location::device, access_mode::read);
    ArrayHandle<Scalar3> d_gamma_r(m_gamma_r, access_location::device, access_mode::read);
    ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);

        {
        ArrayHandle<Scalar> d_partial_sumBD(m_partial_sum1, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar> d_sumBD(m_sum, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);

        unsigned int group_size = m_group->getNumMembers();
        m_num_blocks = group_size / m_block_size + 1;

        // perform the update on the GPU
        rattle_langevin_step_two_args args;
        args.d_gamma = d_gamma.data;
        args.n_types = m_gamma.getNumElements();
        args.use_alpha = m_use_alpha;
        args.alpha = m_alpha;
        args.T = (*m_T)(timestep);
        args.eta = m_eta;
        args.timestep = timestep;
        args.seed = m_seed;
        args.d_sum_bdenergy = d_sumBD.data;
        args.d_partial_sum_bdenergy = d_partial_sumBD.data;
        args.block_size = m_block_size;
        args.num_blocks = m_num_blocks;
        args.noiseless_t = m_noiseless_t;
        args.noiseless_r = m_noiseless_r;
        args.tally = m_tally;

        gpu_rattle_langevin_step_two(d_pos.data,
                              d_vel.data,
                              d_accel.data,
                              d_diameter.data,
                              d_tag.data,
                              d_index_array.data,
                              group_size,
                              d_net_force.data,
                              args,
                              m_manifoldGPU,
                              m_deltaT,
                              D);

        if(m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        if (m_aniso)
            {
            // second part of angular update
            ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::read);
            ArrayHandle<Scalar4> d_angmom(m_pdata->getAngularMomentumArray(), access_location::device, access_mode::readwrite);
            ArrayHandle<Scalar4> d_net_torque(m_pdata->getNetTorqueArray(), access_location::device, access_mode::read);
            ArrayHandle<Scalar3> d_inertia(m_pdata->getMomentsOfInertiaArray(), access_location::device, access_mode::read);

            unsigned int group_size = m_group->getNumMembers();
            gpu_rattle_langevin_angular_step_two(d_pos.data,
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
                                     1.0
                                     );

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }

        }



    if (m_tally)
        {
        ArrayHandle<Scalar> h_sumBD(m_sum, access_location::host, access_mode::read);
        #ifdef ENABLE_MPI
        if (m_comm)
            {
            MPI_Allreduce(MPI_IN_PLACE, &h_sumBD.data[0], 1, MPI_HOOMD_SCALAR, MPI_SUM, m_exec_conf->getMPICommunicator());
            }
        #endif
        m_reservoir_energy -= h_sumBD.data[0]*m_deltaT;
        m_extra_energy_overdeltaT= 0.5*h_sumBD.data[0];
        }
    // done profiling
    if (m_prof)
        m_prof->pop(m_exec_conf);
    }

void TwoStepRATTLELangevinGPU::IncludeRATTLEForce(unsigned int timestep)
    {

    // access all the needed data
    const GlobalArray< Scalar4 >& net_force = m_pdata->getNetForce();
    const GlobalArray<Scalar>&  net_virial = m_pdata->getNetVirial();
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::read);
    ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::readwrite);

    ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> d_net_virial(net_virial, access_location::device, access_mode::readwrite);

    ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);

    unsigned int net_virial_pitch = net_virial.getPitch();

    // perform the update on the GPU
    m_exec_conf->beginMultiGPU();
    m_tuner_one->begin();
    gpu_include_rattle_force_nve(d_pos.data,
                     d_vel.data,
                     d_accel.data,
                     d_net_force.data,
                     d_net_virial.data,
                     d_index_array.data,
                     m_group->getGPUPartition(),
                     net_virial_pitch,
                     m_manifoldGPU,
                     m_eta,
                     m_deltaT,
                     false,
                     m_tuner_one->getParam());

    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    m_tuner_one->end();
    m_exec_conf->endMultiGPU();

    }

void export_TwoStepRATTLELangevinGPU(py::module& m)
    {
    py::class_<TwoStepRATTLELangevinGPU, TwoStepRATTLELangevin, std::shared_ptr<TwoStepRATTLELangevinGPU> >(m, "TwoStepRATTLELangevinGPU")
        .def(py::init< std::shared_ptr<SystemDefinition>,
                               std::shared_ptr<ParticleGroup>,
                               std::shared_ptr<Manifold>,
                               std::shared_ptr<Variant>,
                               unsigned int,
                               Scalar
                               >())
        ;
    }
