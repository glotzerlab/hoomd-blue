// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "WallForceConstraintCompute.h"
#include "WallForceConstraintComputeGPU.cuh"
#include "hoomd/Autotuner.h"

/*! \file WallForceConstraintComputeGPU.h
    \brief Declares a class for computing active forces on the GPU
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __WALLFORCECONSTRAINTCOMPUTE_GPU_H__
#define __WALLFORCECONSTRAINTCOMPUTE_GPU_H__

#include <vector>

namespace hoomd
    {
namespace md
    {
//! Adds an active force to a number of particles with confinement on the GPU
/*! \ingroup computes
 */
template<class Manifold>
class PYBIND11_EXPORT WallForceConstraintComputeGPU
    : public WallForceConstraintCompute<Manifold>
    {
    public:
    //! Constructs the compute
    WallForceConstraintComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                                    std::shared_ptr<ParticleGroup> group,
                                    Manifold manifold);

    protected:
    std::shared_ptr<Autotuner<1>> m_tuner_friction;  //!< Autotuner for block size (diff kernel)
    std::shared_ptr<Autotuner<1>> m_tuner_constraint; //!< Autotuner for block size (constr kernel)

    //! Compute constraint forces
    virtual void computeConstraintForces();

    //! Compute friction forces
    virtual void computeFrictionForces();
    };

/*! \file WallForceConstraintComputeGPU.cc
    \brief Contains code for the WallForceConstraintComputeGPU class
*/

/*! \param f_list An array of (x,y,z) tuples for the active force vector for each
           individual particle.
    \param orientation_link if True then forces and torques are applied in the
           particle's reference frame. If false, then the box reference frame is
       used. Only relevant for non-point-like anisotropic particles.
    \param orientation_reverse_link When True, the particle's orientation is set
           to match the active force vector. Useful for using a particle's
       orientation to log the active force vector. Not recommended for
       anisotropic particles
    \param rotation_diff rotational diffusion constant for all particles.
    \param manifold specifies a manfold surface, to which particles are confined.
*/
template<class Manifold>
WallForceConstraintComputeGPU<Manifold>::WallForceConstraintComputeGPU(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<ParticleGroup> group,
    Manifold manifold)
    : WallForceConstraintCompute<Manifold>(sysdef, group, manifold)
    {
    if (!this->m_exec_conf->isCUDAEnabled())
        {
        throw std::runtime_error("WallForceConstraintComputeGPU requires a GPU device.");
        }

    m_tuner_friction.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf)},
                                             this->m_exec_conf,
                                             "wall_constraint_diffusion"));
    m_tuner_constraint.reset(
        new Autotuner<1>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf)},
                         this->m_exec_conf,
                         "wall_constraint_constraint"));
    this->m_autotuners.insert(this->m_autotuners.end(),
                              {m_tuner_friction, m_tuner_constraint});
    }

/*! This function sets appropriate active forces and torques on all active particles.
 */
template<class Manifold> void WallForceConstraintComputeGPU<Manifold>::computeFrictionForces()
    {
    //  array handles
    ArrayHandle<Scalar4> d_force(this->m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(this->m_virial, access_location::device, access_mode::overwrite);

    ArrayHandle<Scalar4> d_pos(this->m_pdata->getPositions(),
                               access_location::device,
                               access_mode::read);
    ArrayHandle<Scalar4> d_net_force(this->m_pdata->getNetForce(),
                                       access_location::device,
                                       access_mode::read);
    ArrayHandle<unsigned int> d_index_array(this->m_group->getIndexArray(),
                                            access_location::device,
                                            access_mode::read);

    // sanity check
    assert(d_force.data != NULL);
    assert(d_virial.data != NULL);
    assert(d_pos.data != NULL);
    assert(d_net_force.data != NULL);
    assert(d_index_array.data != NULL);
    unsigned int group_size = this->m_group->getNumMembers();

    // compute the forces on the GPU
    this->m_tuner_friction->begin();
    kernel::gpu_compute_wall_friction<Manifold>(group_size,
				      d_index_array.data,
				      d_force.data,
				      d_virial.data,
				      d_pos.data,
				      d_net_force.data,
        			      this->m_manifold,
				      this->m_tuner_friction->getParam()[0]);

    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    this->m_tuner_friction->end();
    }

/*! This function applies rotational diffusion to all active particles. The angle between the torque
 vector and
 * force vector does not change
    \param timestep Current timestep
*/
template<class Manifold> void WallForceConstraintComputeGPU<Manifold>::computeConstraintForces()
    {
    //  array handles
    ArrayHandle<Scalar4> d_force(this->m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(this->m_virial, access_location::device, access_mode::overwrite);

    ArrayHandle<Scalar4> d_pos(this->m_pdata->getPositions(),
                               access_location::device,
                               access_mode::read);
    ArrayHandle<unsigned int> d_index_array(this->m_group->getIndexArray(),
                                            access_location::device,
                                            access_mode::read);

    // sanity check
    assert(d_force.data != NULL);
    assert(d_virial.data != NULL);
    assert(d_pos.data != NULL);
    assert(d_index_array.data != NULL);
    unsigned int group_size = this->m_group->getNumMembers();

    // compute the forces on the GPU
    this->m_tuner_constraint->begin();
    kernel::gpu_compute_wall_constraint<Manifold>(group_size,
				      d_index_array.data,
				      d_force.data,
				      d_virial.data,
				      d_pos.data,
        			      this->m_manifold,
				      this->m_tuner_constraint->getParam()[0]);

    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    this->m_tuner_constraint->end();
    }

namespace detail
    {
template<class Manifold>
void export_WallForceConstraintComputeGPU(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<WallForceConstraintComputeGPU<Manifold>,
                     WallForceConstraintCompute<Manifold>,
                     std::shared_ptr<WallForceConstraintComputeGPU<Manifold>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ParticleGroup>,
                            Manifold>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
