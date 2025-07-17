// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ShearForceComputeGPU.h"
#include "ShearForceComputeGPU.cuh"

#include <vector>

using namespace std;

/*! \file ShearForceComputeGPU.cc
    \brief Contains code for the ShearForceComputeGPU class
*/

namespace hoomd
    {
namespace md
    {
ShearForceComputeGPU::ShearForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                                                 std::shared_ptr<ParticleGroup> group,
						 Scalar shear_force)
    : ShearForceCompute(sysdef, group, shear_force)
    {
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error()
            << "Creating a ShearForceComputeGPU with no GPU in the execution configuration"
            << endl;
        throw std::runtime_error("Error initializing ShearForceComputeGPU");
        }

    // initialize autotuner
    m_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                   m_exec_conf,
                                   "shear_force"));
    m_autotuners.push_back(m_tuner);
    }

/*! This function sets appropriate active forces and torques on all active particles.
 */
void ShearForceComputeGPU::setForces()
    {
    ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::overwrite);

    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_index_array(m_group->getIndexArray(),
                                            access_location::device,
                                            access_mode::read);

    // sanity check
    assert(d_force.data != NULL);
    assert(d_pos.data != NULL);
    assert(d_index_array.data != NULL);
    unsigned int group_size = m_group->getNumMembers();
    unsigned int N = m_pdata->getN();
    const Scalar s_f = 2/m_pdata->getBox().getL().y*m_shear_force;

    m_force.zeroFill();
    // compute the forces on the GPU
    m_tuner->begin();

    kernel::gpu_compute_shear_force_set_forces(group_size,
                                                  d_index_array.data,
                                                  d_force.data,
                                                  d_pos.data,
                                                  s_f,
                                                  N,
                                                  m_tuner->getParam()[0]);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    m_tuner->end();
    }

namespace detail
    {
void export_ShearForceComputeGPU(pybind11::module& m)
    {
    pybind11::class_<ShearForceComputeGPU,
                     ShearForceCompute,
                     std::shared_ptr<ShearForceComputeGPU>>(m, "ShearForceComputeGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>, Scalar>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
