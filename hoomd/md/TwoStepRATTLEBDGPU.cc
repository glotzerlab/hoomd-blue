// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "TwoStepRATTLEBDGPU.h"
#include "TwoStepRATTLEBDGPU.cuh"

#ifdef ENABLE_MPI
#include "hoomd/HOOMDMPI.h"
#endif

namespace py = pybind11;

using namespace std;

/*! \file TwoStepRATTLEBDGPU.h
    \brief Contains code for the TwoStepRATTLEBDGPU class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param T Temperature set point as a function of time
    \param seed Random seed to use in generating random numbers
    \param use_lambda If true, gamma=lambda*diameter, otherwise use a per-type gamma via setGamma()
    \param lambda Scale factor to convert diameter to gamma
*/
TwoStepRATTLEBDGPU::TwoStepRATTLEBDGPU(std::shared_ptr<SystemDefinition> sysdef,
                           std::shared_ptr<ParticleGroup> group,
                       	   std::shared_ptr<Manifold> manifold,
                           std::shared_ptr<Variant> T,
                           unsigned int seed,
                           bool use_lambda,
                           Scalar lambda,
                           bool noiseless_t,
                           bool noiseless_r,
                           Scalar eta)
    : TwoStepRATTLEBD(sysdef, group, manifold,T, seed, use_lambda, lambda, noiseless_t, noiseless_r, eta), m_manifoldGPU( manifold->returnL(), manifold->returnR(), manifold->returnSurf() )
    {
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating a TwoStepRATTLEBDGPU while CUDA is disabled" << endl;
        throw std::runtime_error("Error initializing TwoStepRATTLEBDGPU");
        }

    unsigned int group_size = m_group->getNumMembersGlobal();
    GPUArray<unsigned int> tmp_groupTags(group_size, m_exec_conf);
    ArrayHandle<unsigned int> groupTags(tmp_groupTags, access_location::host);

    for (unsigned int i = 0; i < group_size; i++)
        {
        unsigned int tag = m_group->getMemberTag(i);
        groupTags.data[i] = tag;
        }

    m_groupTags.swap(tmp_groupTags);

    m_block_size = 256;
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward a full time step and velocities are redrawn from the proper distribution.
*/
void TwoStepRATTLEBDGPU::integrateStepOne(unsigned int timestep)
    {
    // profile this step
    if (m_prof)
        m_prof->push(m_exec_conf, "BD step 1");

    // access all the needed data
    BoxDim box = m_pdata->getBox();
    ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);
    unsigned int group_size = m_group->getNumMembers();
    const unsigned int D = m_sysdef->getNDimensions();
    const GlobalArray< Scalar4 >& net_force = m_pdata->getNetForce();

    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);
    ArrayHandle<int3> d_image(m_pdata->getImages(), access_location::device, access_mode::readwrite);

    ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::read);
    ArrayHandle<Scalar3> d_f_brownian(m_f_brownian, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_gamma(m_gamma, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_groupTags(m_groupTags, access_location::device, access_mode::read);

    // for rotational noise
    ArrayHandle<Scalar3> d_gamma_r(m_gamma_r, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> d_torque(m_pdata->getNetTorqueArray(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar3> d_inertia(m_pdata->getMomentsOfInertiaArray(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_angmom(m_pdata->getAngularMomentumArray(), access_location::device, access_mode::readwrite);

    
    rattle_bd_step_one_args args;
    args.d_gamma = d_gamma.data;
    args.n_types = m_gamma.getNumElements();
    args.use_lambda = m_use_lambda;
    args.lambda = m_lambda;
    args.T = m_T->getValue(timestep);
    args.eta = m_eta;
    args.timestep = timestep;
    args.seed = m_seed;


    bool aniso = m_aniso;

    if (m_exec_conf->allConcurrentManagedAccess())
        {
        // prefetch gammas
        auto& gpu_map = m_exec_conf->getGPUIds();
        for (unsigned int idev = 0; idev < m_exec_conf->getNumActiveGPUs(); ++idev)
            {
            cudaMemPrefetchAsync(m_gamma.get(), sizeof(Scalar)*m_gamma.getNumElements(), gpu_map[idev]);
            cudaMemPrefetchAsync(m_gamma_r.get(), sizeof(Scalar)*m_gamma_r.getNumElements(), gpu_map[idev]);
            }
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

   
    m_exec_conf->beginMultiGPU();

    // perform the update on the GPU
    gpu_rattle_brownian_step_one(d_pos.data,
                          d_image.data,
                          box,
                          d_diameter.data,
                          d_rtag.data,
                          d_groupTags.data,
                          group_size,
                          d_net_force.data,
                          d_f_brownian.data,
                          d_gamma_r.data,
                          d_orientation.data,
                          d_torque.data,
                          d_inertia.data,
                          d_angmom.data,
                          args,
                          aniso,
                          m_deltaT,
                          D,
                          m_noiseless_r,
                          m_group->getGPUPartition());

    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    m_exec_conf->endMultiGPU();

    // done profiling
    if (m_prof)
        m_prof->pop(m_exec_conf);
    }

/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1 on the GPU
*/
void TwoStepRATTLEBDGPU::integrateStepTwo(unsigned int timestep)
    {
    // there is no step 2
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward a full time step and velocities are redrawn from the proper distribution.
*/
void TwoStepRATTLEBDGPU::IncludeRATTLEForce(unsigned int timestep)
    {

    // access all the needed data
    ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);
    unsigned int group_size = m_group->getNumMembers();
    const GlobalArray< Scalar4 >& net_force = m_pdata->getNetForce();
    const GlobalArray<Scalar>&  net_virial = m_pdata->getNetVirial();

    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar3> d_f_brownian(m_f_brownian, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> d_net_virial(net_virial, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> d_gamma(m_gamma, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_groupTags(m_groupTags, access_location::device, access_mode::read);

    unsigned int net_virial_pitch = net_virial.getPitch();

    
    rattle_bd_step_one_args args;
    args.d_gamma = d_gamma.data;
    args.n_types = m_gamma.getNumElements();
    args.use_lambda = m_use_lambda;
    args.lambda = m_lambda;
    args.T = m_T->getValue(timestep);
    args.eta = m_eta;
    args.timestep = timestep;
    args.seed = m_seed;


    if (m_exec_conf->allConcurrentManagedAccess())
        {
        // prefetch gammas
        auto& gpu_map = m_exec_conf->getGPUIds();
        for (unsigned int idev = 0; idev < m_exec_conf->getNumActiveGPUs(); ++idev)
            {
            cudaMemPrefetchAsync(m_gamma.get(), sizeof(Scalar)*m_gamma.getNumElements(), gpu_map[idev]);
            }
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

   
    m_exec_conf->beginMultiGPU();

    // perform the update on the GPU
    gpu_include_rattle_force_bd(d_pos.data,
                          d_vel.data,
                          d_net_force.data,
                          d_f_brownian.data,
                          d_net_virial.data,
                          d_diameter.data,
                          d_rtag.data,
                          d_groupTags.data,
                          group_size,
                          args,
                          m_manifoldGPU,
                          net_virial_pitch,
                          m_deltaT,
                          m_noiseless_t,
                          m_group->getGPUPartition());

    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    m_exec_conf->endMultiGPU();

    // done profiling
    if (m_prof)
        m_prof->pop(m_exec_conf);
    }

void export_TwoStepRATTLEBDGPU(py::module& m)
    {
    py::class_<TwoStepRATTLEBDGPU, std::shared_ptr<TwoStepRATTLEBDGPU> >(m, "TwoStepRATTLEBDGPU", py::base<TwoStepRATTLEBD>())
        .def(py::init< std::shared_ptr<SystemDefinition>,
                               std::shared_ptr<ParticleGroup>,
                               std::shared_ptr<Manifold>,
                               std::shared_ptr<Variant>,
                               unsigned int,
                               bool,
                               Scalar,
                               bool,
                               bool,
			       Scalar>())
        ;
    }
