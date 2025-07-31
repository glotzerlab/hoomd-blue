// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "WallCouplingForceComputeGPU.h"
#include "WallCouplingForceComputeGPU.cuh"

#include <vector>

using namespace std;

/*! \file WallCouplingForceComputeGPU.cc
    \brief Contains code for the WallCouplingForceComputeGPU class
*/

namespace hoomd
    {
namespace md
    {
WallCouplingForceComputeGPU::WallCouplingForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                                                 std::shared_ptr<ParticleGroup> group,
						 Scalar radial_force,
						 Scalar tangential_force,
						 Scalar R)
    : WallCouplingForceCompute(sysdef, group, radial_force, tangential_force, R)
    {
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error()
            << "Creating a WallCouplingForceComputeGPU with no GPU in the execution configuration"
            << endl;
        throw std::runtime_error("Error initializing WallCouplingForceComputeGPU");
        }

    // initialize autotuner
    m_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                   m_exec_conf,
                                   "shear_force"));
    m_autotuners.push_back(m_tuner);
    }

/*! This function sets appropriate active forces and torques on all active particles.
 */
void WallCouplingForceComputeGPU::setForces()
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

    m_force.zeroFill();
    // compute the forces on the GPU
    m_tuner->begin();

    kernel::gpu_compute_wall_coupling_force_set_forces(group_size,
                                                  d_index_array.data,
                                                  d_force.data,
                                                  d_pos.data,
                                                  m_radial_force,
                                                  m_tangential_force,
                                                  m_R,
                                                  N,
                                                  m_tuner->getParam()[0]);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    m_tuner->end();
    }

namespace detail
    {
void export_WallCouplingForceComputeGPU(pybind11::module& m)
    {
    pybind11::class_<WallCouplingForceComputeGPU,
                     WallCouplingForceCompute,
                     std::shared_ptr<WallCouplingForceComputeGPU>>(m, "WallCouplingForceComputeGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>, Scalar, Scalar, Scalar>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
