// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ActiveForceComputeGPU.cuh"
#include "ActiveForceConstraintCompute.h"
#include "ActiveForceConstraintComputeGPU.cuh"
#include "hoomd/Autotuner.h"

/*! \file ActiveForceConstraintComputeGPU.h
    \brief Declares a class for computing active forces on the GPU
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __ACTIVEFORCECONSTRAINTCOMPUTE_GPU_H__
#define __ACTIVEFORCECONSTRAINTCOMPUTE_GPU_H__

#include <vector>

namespace hoomd
    {
namespace md
    {
//! Adds an active force to a number of particles with confinement on the GPU
/*! \ingroup computes
 */
template<class Manifold>
class PYBIND11_EXPORT ActiveForceConstraintComputeGPU
    : public ActiveForceConstraintCompute<Manifold>
    {
    public:
    //! Constructs the compute
    ActiveForceConstraintComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                                    std::shared_ptr<ParticleGroup> group,
                                    Manifold manifold);

    protected:
    std::shared_ptr<Autotuner<1>> m_tuner_force;      //!< Autotuner for block size (force kernel)
    std::shared_ptr<Autotuner<1>> m_tuner_diffusion;  //!< Autotuner for block size (diff kernel)
    std::shared_ptr<Autotuner<1>> m_tuner_constraint; //!< Autotuner for block size (constr kernel)

    //! Set forces for particles
    virtual void setForces();

    //! Orientational diffusion for spherical particles
    virtual void rotationalDiffusion(Scalar rotational_diffusion, uint64_t timestep);

    //! Set constraints if particles confined to a surface
    virtual void setConstraint();
    };

/*! \file ActiveForceConstraintComputeGPU.cc
    \brief Contains code for the ActiveForceConstraintComputeGPU class
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
ActiveForceConstraintComputeGPU<Manifold>::ActiveForceConstraintComputeGPU(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<ParticleGroup> group,
    Manifold manifold)
    : ActiveForceConstraintCompute<Manifold>(sysdef, group, manifold)
    {
    if (!this->m_exec_conf->isCUDAEnabled())
        {
        throw std::runtime_error("ActiveForceConstraintComputeGPU requires a GPU device.");
        }

    // Initialize autotuners.
    m_tuner_force.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf)},
                                         this->m_exec_conf,
                                         "active_constraint_force"));
    m_tuner_diffusion.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf)},
                                             this->m_exec_conf,
                                             "active_constraint_diffusion"));
    m_tuner_constraint.reset(
        new Autotuner<1>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf)},
                         this->m_exec_conf,
                         "active_constraint_constraint"));
    this->m_autotuners.insert(this->m_autotuners.end(),
                              {m_tuner_force, m_tuner_diffusion, m_tuner_constraint});

    unsigned int type = this->m_pdata->getNTypes();
    GlobalVector<Scalar4> tmp_f_activeVec(type, this->m_exec_conf);
    GlobalVector<Scalar4> tmp_t_activeVec(type, this->m_exec_conf);

        {
        ArrayHandle<Scalar4> old_f_activeVec(this->m_f_activeVec, access_location::host);
        ArrayHandle<Scalar4> old_t_activeVec(this->m_t_activeVec, access_location::host);

        ArrayHandle<Scalar4> f_activeVec(tmp_f_activeVec, access_location::host);
        ArrayHandle<Scalar4> t_activeVec(tmp_t_activeVec, access_location::host);

        // for each type of the particles in the group
        for (unsigned int i = 0; i < type; i++)
            {
            f_activeVec.data[i] = old_f_activeVec.data[i];

            t_activeVec.data[i] = old_t_activeVec.data[i];
            }
        }

    this->m_f_activeVec.swap(tmp_f_activeVec);
    this->m_t_activeVec.swap(tmp_t_activeVec);
    }

/*! This function sets appropriate active forces and torques on all active particles.
 */
template<class Manifold> void ActiveForceConstraintComputeGPU<Manifold>::setForces()
    {
    //  array handles
    ArrayHandle<Scalar4> d_f_actVec(this->m_f_activeVec,
                                    access_location::device,
                                    access_mode::read);
    ArrayHandle<Scalar4> d_force(this->m_force, access_location::device, access_mode::overwrite);

    ArrayHandle<Scalar4> d_t_actVec(this->m_t_activeVec,
                                    access_location::device,
                                    access_mode::read);
    ArrayHandle<Scalar4> d_torque(this->m_torque, access_location::device, access_mode::overwrite);

    ArrayHandle<Scalar4> d_pos(this->m_pdata->getPositions(),
                               access_location::device,
                               access_mode::read);
    ArrayHandle<Scalar4> d_orientation(this->m_pdata->getOrientationArray(),
                                       access_location::device,
                                       access_mode::read);
    ArrayHandle<unsigned int> d_index_array(this->m_group->getIndexArray(),
                                            access_location::device,
                                            access_mode::read);

    // sanity check
    assert(d_force.data != NULL);
    assert(d_f_actVec.data != NULL);
    assert(d_t_actVec.data != NULL);
    assert(d_pos.data != NULL);
    assert(d_orientation.data != NULL);
    assert(d_index_array.data != NULL);
    unsigned int group_size = this->m_group->getNumMembers();
    unsigned int N = this->m_pdata->getN();

    // compute the forces on the GPU
    this->m_tuner_force->begin();
    kernel::gpu_compute_active_force_set_forces(group_size,
                                                d_index_array.data,
                                                d_force.data,
                                                d_torque.data,
                                                d_pos.data,
                                                d_orientation.data,
                                                d_f_actVec.data,
                                                d_t_actVec.data,
                                                N,
                                                this->m_tuner_force->getParam()[0]);

    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    this->m_tuner_force->end();
    }

/*! This function applies rotational diffusion to all active particles. The angle between the torque
 vector and
 * force vector does not change
    \param timestep Current timestep
*/
template<class Manifold>
void ActiveForceConstraintComputeGPU<Manifold>::rotationalDiffusion(Scalar rotational_diffusion,
                                                                    uint64_t timestep)
    {
    //  array handles
    ArrayHandle<Scalar4> d_pos(this->m_pdata->getPositions(),
                               access_location::device,
                               access_mode::read);
    ArrayHandle<Scalar4> d_orientation(this->m_pdata->getOrientationArray(),
                                       access_location::device,
                                       access_mode::readwrite);
    ArrayHandle<unsigned int> d_index_array(this->m_group->getIndexArray(),
                                            access_location::device,
                                            access_mode::read);
    ArrayHandle<unsigned int> d_tag(this->m_pdata->getTags(),
                                    access_location::device,
                                    access_mode::read);

    assert(d_pos.data != NULL);

    bool is2D = (this->m_sysdef->getNDimensions() == 2);
    unsigned int group_size = this->m_group->getNumMembers();

    const auto rotation_constant = slow::sqrt(2.0 * rotational_diffusion * this->m_deltaT);
    // perform the update on the GPU
    this->m_tuner_diffusion->begin();

    kernel::gpu_compute_active_force_constraint_rotational_diffusion<Manifold>(
        group_size,
        d_tag.data,
        d_index_array.data,
        d_pos.data,
        d_orientation.data,
        this->m_manifold,
        is2D,
        rotation_constant,
        timestep,
        this->m_sysdef->getSeed(),
        this->m_tuner_diffusion->getParam()[0]);

    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    this->m_tuner_diffusion->end();
    }

/*! This function sets an ellipsoid surface constraint for all active particles
 */
template<class Manifold> void ActiveForceConstraintComputeGPU<Manifold>::setConstraint()
    {
    //  array handles
    ArrayHandle<Scalar4> d_f_actVec(this->m_f_activeVec,
                                    access_location::device,
                                    access_mode::readwrite);
    ArrayHandle<Scalar4> d_pos(this->m_pdata->getPositions(),
                               access_location::device,
                               access_mode::read);
    ArrayHandle<Scalar4> d_orientation(this->m_pdata->getOrientationArray(),
                                       access_location::device,
                                       access_mode::readwrite);
    ArrayHandle<unsigned int> d_index_array(this->m_group->getIndexArray(),
                                            access_location::device,
                                            access_mode::read);

    assert(d_pos.data != NULL);

    unsigned int group_size = this->m_group->getNumMembers();

    // perform the update on the GPU
    this->m_tuner_constraint->begin();

    kernel::gpu_compute_active_force_set_constraints<Manifold>(
        group_size,
        d_index_array.data,
        d_pos.data,
        d_orientation.data,
        d_f_actVec.data,
        this->m_manifold,
        this->m_tuner_constraint->getParam()[0]);

    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    this->m_tuner_constraint->end();
    }

namespace detail
    {
template<class Manifold>
void export_ActiveForceConstraintComputeGPU(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<ActiveForceConstraintComputeGPU<Manifold>,
                     ActiveForceConstraintCompute<Manifold>,
                     std::shared_ptr<ActiveForceConstraintComputeGPU<Manifold>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ParticleGroup>,
                            Manifold>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
