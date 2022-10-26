// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ConstantForceComputeGPU.h"
#include "ConstantForceComputeGPU.cuh"

#include <vector>

using namespace std;

/*! \file ConstantForceComputeGPU.cc
    \brief Contains code for the ConstantForceComputeGPU class
*/

namespace hoomd
    {
namespace md
    {
ConstantForceComputeGPU::ConstantForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                                                 std::shared_ptr<ParticleGroup> group)
    : ConstantForceCompute(sysdef, group)
    {
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error()
            << "Creating a ConstantForceComputeGPU with no GPU in the execution configuration"
            << endl;
        throw std::runtime_error("Error initializing ConstantForceComputeGPU");
        }

    // initialize autotuner
    m_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                   m_exec_conf,
                                   "constant_force"));
    m_autotuners.push_back(m_tuner);

    // unsigned int N = m_pdata->getNGlobal();
    // unsigned int group_size = m_group->getNumMembersGlobal();
    unsigned int type = m_pdata->getNTypes();
    GlobalVector<Scalar3> tmp_f_constantVec(type, m_exec_conf);
    GlobalVector<Scalar3> tmp_t_constantVec(type, m_exec_conf);

        {
        ArrayHandle<Scalar3> old_f_constantVec(m_f_constantVec, access_location::host);
        ArrayHandle<Scalar3> old_t_constantVec(m_t_constantVec, access_location::host);

        ArrayHandle<Scalar3> f_constantVec(tmp_f_constantVec, access_location::host);
        ArrayHandle<Scalar3> t_constantVec(tmp_t_constantVec, access_location::host);

        // for each type of the particles in the group
        for (unsigned int i = 0; i < type; i++)
            {
            f_constantVec.data[i] = old_f_constantVec.data[i];

            t_constantVec.data[i] = old_t_constantVec.data[i];
            }
        }

    m_f_constantVec.swap(tmp_f_constantVec);
    m_t_constantVec.swap(tmp_t_constantVec);
    }

/*! This function sets appropriate active forces and torques on all active particles.
 */
void ConstantForceComputeGPU::setForces()
    {
    //  array handles
    ArrayHandle<Scalar3> d_f_constVec(m_f_constantVec, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::overwrite);

    ArrayHandle<Scalar3> d_t_constVec(m_t_constantVec, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_torque(m_torque, access_location::device, access_mode::overwrite);

    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_index_array(m_group->getIndexArray(),
                                            access_location::device,
                                            access_mode::read);

    // sanity check
    assert(d_force.data != NULL);
    assert(d_f_constVec.data != NULL);
    assert(d_t_constVec.data != NULL);
    assert(d_pos.data != NULL);
    assert(d_index_array.data != NULL);
    unsigned int group_size = m_group->getNumMembers();
    unsigned int N = m_pdata->getN();

    // compute the forces on the GPU
    m_tuner->begin();

    kernel::gpu_compute_constant_force_set_forces(group_size,
                                                  d_index_array.data,
                                                  d_force.data,
                                                  d_torque.data,
                                                  d_pos.data,
                                                  d_f_constVec.data,
                                                  d_t_constVec.data,
                                                  N,
                                                  m_tuner->getParam()[0]);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    m_tuner->end();
    }

namespace detail
    {
void export_ConstantForceComputeGPU(pybind11::module& m)
    {
    pybind11::class_<ConstantForceComputeGPU,
                     ConstantForceCompute,
                     std::shared_ptr<ConstantForceComputeGPU>>(m, "ConstantForceComputeGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
