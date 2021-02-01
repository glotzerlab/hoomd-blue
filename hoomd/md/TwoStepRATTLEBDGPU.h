// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "TwoStepRATTLEBD.h"
#include "TwoStepRATTLEBDGPU.cuh"

#ifdef ENABLE_MPI
#include "hoomd/HOOMDMPI.h"
#endif

#pragma once

#include <pybind11/pybind11.h>

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

namespace py = pybind11;

using namespace std;

//! Implements Brownian dynamics on the GPU
/*! GPU accelerated version of TwoStepBD

    \ingroup updaters
*/
template<class Manifold>
class PYBIND11_EXPORT TwoStepRATTLEBDGPU : public TwoStepRATTLEBD<Manifold>
    {
    public:
        //! Constructs the integration method and associates it with the system
        TwoStepRATTLEBDGPU(std::shared_ptr<SystemDefinition> sysdef,
                     std::shared_ptr<ParticleGroup> group,
                     Manifold manifold,
                     std::shared_ptr<Variant> T,
                     unsigned int seed,
                     Scalar eta = 0.000001)
    : TwoStepRATTLEBD<Manifold>(sysdef, group, manifold,T, seed, eta)
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

        virtual ~TwoStepRATTLEBDGPU() {};

        //! Performs the first step of the integration
        virtual void integrateStepOne(unsigned int timestep);

        //! Performs the second step of the integration
        virtual void integrateStepTwo(unsigned int timestep){};

        //! Includes the RATTLE forces to the virial/net force
        virtual void IncludeRATTLEForce(unsigned int timestep);

    protected:
        unsigned int m_block_size;               //!< block size
        GPUArray<unsigned int>  m_groupTags; //! Stores list converting group index to global tag
    };

/*! \param timestep Current time step
    \post Particle positions are moved forward a full time step and velocities are redrawn from the proper distribution.
*/
template<class Manifold>
void TwoStepRATTLEBDGPU<Manifold>::integrateStepOne(unsigned int timestep)
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
    args.use_alpha = m_use_alpha;
    args.alpha = m_alpha;
    args.T = (*m_T)(timestep);
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
    \post Particle positions are moved forward a full time step and velocities are redrawn from the proper distribution.
*/
template<class Manifold>
void TwoStepRATTLEBDGPU<Manifold>::IncludeRATTLEForce(unsigned int timestep)
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
    args.use_alpha = m_use_alpha;
    args.alpha = m_alpha;
    args.T = (*m_T)(timestep);
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
    gpu_include_rattle_force_bd<Manifold>(d_pos.data,
                          d_vel.data,
                          d_net_force.data,
                          d_f_brownian.data,
                          d_net_virial.data,
                          d_diameter.data,
                          d_rtag.data,
                          d_groupTags.data,
                          group_size,
                          args,
                          m_manifold,
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


//! Exports the TwoStepRATTLEBDGPU class to python
template<class Manifold>
void export_TwoStepRATTLEBDGPU(py::module& m, const std::string& name)
    {
    py::class_<TwoStepRATTLEBDGPU<Manifold>, TwoStepRATTLEBD<Manifold>, std::shared_ptr<TwoStepRATTLEBDGPU<Manifold> > >(m, name.c_str())
        .def(py::init< std::shared_ptr<SystemDefinition>,
                               std::shared_ptr<ParticleGroup>,
                               Manifold,
                               std::shared_ptr<Variant>,
                               unsigned int,
			                   Scalar>())
        ;
    }
