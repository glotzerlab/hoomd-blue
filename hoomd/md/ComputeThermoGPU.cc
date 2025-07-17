// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file ComputeThermoGPU.cc
    \brief Contains code for the ComputeThermoGPU class
*/

#include "ComputeThermoGPU.h"
#include "ComputeThermoGPU.cuh"

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#include "hoomd/HOOMDMPI.h"
#endif

#include <iostream>
using namespace std;

namespace hoomd
    {
namespace md
    {
/*! \param sysdef System for which to compute thermodynamic properties
    \param group Subset of the system over which properties are calculated
*/
ComputeThermoGPU::ComputeThermoGPU(std::shared_ptr<SystemDefinition> sysdef,
                                   std::shared_ptr<ParticleGroup> group)
    : ComputeThermo(sysdef, group), m_scratch(m_exec_conf), m_scratch_pressure_tensor(m_exec_conf),
      m_scratch_rot(m_exec_conf)
    {
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error()
            << "Creating a ComputeThermoGPU with no GPU in the execution configuration" << endl;
        throw std::runtime_error("Error initializing ComputeThermoGPU");
        }

    m_block_size = 256;
    }

//! Destructor
ComputeThermoGPU::~ComputeThermoGPU() { }

/*! Computes all thermodynamic properties of the system in one fell swoop, on the GPU.
 */
void ComputeThermoGPU::computeProperties()
    {
    // just drop out if the group is an empty group
    if (m_group->getNumMembersGlobal() == 0)
        return;

    unsigned int group_size = m_group->getNumMembers();

    assert(m_pdata);

    // number of blocks in reduction (round up)
    unsigned int num_blocks = m_group->getNumMembers() / m_block_size + 1;

    // resize work space
    size_t old_size = m_scratch.size();

    m_scratch.resize(num_blocks);
    m_scratch_pressure_tensor.resize(num_blocks * 6);
    m_scratch_rot.resize(num_blocks);

    if (m_scratch.size() != old_size)
        {
        // reset to zero
        ArrayHandle<Scalar4> d_scratch(m_scratch, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar> d_scratch_pressure_tensor(m_scratch_pressure_tensor,
                                                      access_location::device,
                                                      access_mode::overwrite);
        ArrayHandle<Scalar> d_scratch_rot(m_scratch_rot,
                                          access_location::device,
                                          access_mode::overwrite);

        hipMemset(d_scratch.data, 0, sizeof(Scalar4) * m_scratch.size());
        hipMemset(d_scratch_pressure_tensor.data,
                  0,
                  sizeof(Scalar) * m_scratch_pressure_tensor.size());
        hipMemset(d_scratch_rot.data, 0, sizeof(Scalar) * m_scratch_rot.size());
        }

    // access the particle data
    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(),
                               access_location::device,
                               access_mode::read);
    ArrayHandle<unsigned int> d_body(m_pdata->getBodies(),
                                     access_location::device,
                                     access_mode::read);
    ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);
    BoxDim box = m_pdata->getGlobalBox();

    PDataFlags flags = m_pdata->getFlags();

        { // scope these array handles so they are released before the additional terms are added
        // access the net force, pe, and virial
        const GPUArray<Scalar4>& net_force = m_pdata->getNetForce();
        const GPUArray<Scalar>& net_virial = m_pdata->getNetVirial();
        ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::read);
        ArrayHandle<Scalar> d_net_virial(net_virial, access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_orientation(m_pdata->getOrientationArray(),
                                           access_location::device,
                                           access_mode::read);
        ArrayHandle<Scalar4> d_angmom(m_pdata->getAngularMomentumArray(),
                                      access_location::device,
                                      access_mode::read);
        ArrayHandle<Scalar3> d_inertia(m_pdata->getMomentsOfInertiaArray(),
                                       access_location::device,
                                       access_mode::read);
        ArrayHandle<Scalar4> d_scratch(m_scratch, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar> d_scratch_pressure_tensor(m_scratch_pressure_tensor,
                                                      access_location::device,
                                                      access_mode::overwrite);
        ArrayHandle<Scalar> d_scratch_rot(m_scratch_rot,
                                          access_location::device,
                                          access_mode::overwrite);
        ArrayHandle<Scalar> d_properties(m_properties,
                                         access_location::device,
                                         access_mode::overwrite);

        // access the group
        ArrayHandle<unsigned int> d_index_array(m_group->getIndexArray(),
                                                access_location::device,
                                                access_mode::read);

        m_exec_conf->setDevice();

        // build up args list
        kernel::compute_thermo_args args;
        args.n_blocks = num_blocks;
        args.d_net_force = d_net_force.data;
        args.d_net_virial = d_net_virial.data;
        args.d_orientation = d_orientation.data;
        args.d_angmom = d_angmom.data;
        args.d_inertia = d_inertia.data;
        args.virial_pitch = net_virial.getPitch();
        args.ndof = m_group->getTranslationalDOF();
        args.D = m_sysdef->getNDimensions();
        args.d_scratch = d_scratch.data;
        args.d_scratch_pressure_tensor = d_scratch_pressure_tensor.data;
        args.d_scratch_rot = d_scratch_rot.data;
        args.block_size = m_block_size;
        args.external_virial_xx = m_pdata->getExternalVirial(0);
        args.external_virial_xy = m_pdata->getExternalVirial(1);
        args.external_virial_xz = m_pdata->getExternalVirial(2);
        args.external_virial_yy = m_pdata->getExternalVirial(3);
        args.external_virial_yz = m_pdata->getExternalVirial(4);
        args.external_virial_zz = m_pdata->getExternalVirial(5);
        args.external_energy = m_pdata->getExternalEnergy();

        // perform the computation on the GPU(s)
        gpu_compute_thermo_partial(d_properties.data,
                                   d_vel.data,
                                   d_body.data,
                                   d_tag.data,
                                   d_index_array.data,
                                   group_size,
                                   box,
                                   args,
                                   flags[pdata_flag::pressure_tensor],
                                   flags[pdata_flag::rotational_kinetic_energy]);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        // perform the computation on GPU 0
        gpu_compute_thermo_final(d_properties.data,
                                 d_vel.data,
                                 d_body.data,
                                 d_tag.data,
                                 d_index_array.data,
                                 group_size,
                                 box,
                                 args,
                                 flags[pdata_flag::pressure_tensor],
                                 flags[pdata_flag::rotational_kinetic_energy]);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

#ifdef ENABLE_MPI
    // in MPI, reduce extensive quantities only when they're needed
    m_properties_reduced = !m_pdata->getDomainDecomposition();
#endif // ENABLE_MPI
    }

namespace detail
    {
void export_ComputeThermoGPU(pybind11::module& m)
    {
    pybind11::class_<ComputeThermoGPU, ComputeThermo, std::shared_ptr<ComputeThermoGPU>>(
        m,
        "ComputeThermoGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
