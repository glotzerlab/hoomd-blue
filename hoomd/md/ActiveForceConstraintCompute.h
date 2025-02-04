// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ActiveForceCompute.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"

/*! \file ActiveForceConstraintCompute.h
    \brief Declares a class for computing active forces and torques
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __ACTIVEFORCECONSTRAINTCOMPUTE_H__
#define __ACTIVEFORCECONSTRAINTCOMPUTE_H__

namespace hoomd
    {
namespace md
    {
//! Adds an active force to a number of particles
/*! \ingroup computes
 */
template<class Manifold>
class PYBIND11_EXPORT ActiveForceConstraintCompute : public ActiveForceCompute
    {
    public:
    //! Constructs the compute
    ActiveForceConstraintCompute(std::shared_ptr<SystemDefinition> sysdef,
                                 std::shared_ptr<ParticleGroup> group,
                                 Manifold manifold);
    //
    //! Destructor
    ~ActiveForceConstraintCompute();

    protected:
    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);

    //! Orientational diffusion for spherical particles
    virtual void rotationalDiffusion(Scalar rotational_diffusion, uint64_t timestep);

    //! Set constraints if particles confined to a surface
    virtual void setConstraint();

    //! Helper function to be called when box changes
    void setBoxChange()
        {
        m_box_changed = true;
        }

    Manifold m_manifold; //!< Constraining Manifold
    bool m_box_changed;
    };

/*! \param sysdef The system definition
    \param group Particle group
    \param rotation_diff Rotational diffusion coefficient
    \param manifold Manifold constraint
 */
template<class Manifold>
ActiveForceConstraintCompute<Manifold>::ActiveForceConstraintCompute(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<ParticleGroup> group,
    Manifold manifold)
    : ActiveForceCompute(sysdef, group), m_manifold(manifold), m_box_changed(true)
    {
    m_pdata->getBoxChangeSignal()
        .template connect<ActiveForceConstraintCompute<Manifold>,
                          &ActiveForceConstraintCompute<Manifold>::setBoxChange>(this);
    }

template<class Manifold> ActiveForceConstraintCompute<Manifold>::~ActiveForceConstraintCompute()
    {
    m_pdata->getBoxChangeSignal()
        .template disconnect<ActiveForceConstraintCompute<Manifold>,
                             &ActiveForceConstraintCompute<Manifold>::setBoxChange>(this);
    m_exec_conf->msg->notice(5) << "Destroying ActiveForceConstraintCompute" << std::endl;
    }

/*! This function applies rotational diffusion to the orientations of all active particles. The
 orientation of any torque vector
 * relative to the force vector is preserved
    \param timestep Current timestep
*/
template<class Manifold>
void ActiveForceConstraintCompute<Manifold>::rotationalDiffusion(Scalar rotational_diffusion,
                                                                 uint64_t timestep)
    {
    //  array handles
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                       access_location::host,
                                       access_mode::readwrite);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

    assert(h_pos.data != NULL);
    assert(h_orientation.data != NULL);
    assert(h_tag.data != NULL);

    const auto rotation_constant = slow::sqrt(2.0 * rotational_diffusion * m_deltaT);
    for (unsigned int i = 0; i < m_group->getNumMembers(); i++)
        {
        unsigned int idx = m_group->getMemberIndex(i);
        unsigned int ptag = h_tag.data[idx];
        hoomd::RandomGenerator rng(
            hoomd::Seed(hoomd::RNGIdentifier::ActiveForceCompute, timestep, m_sysdef->getSeed()),
            hoomd::Counter(ptag));

        quat<Scalar> quati(h_orientation.data[idx]);

        Scalar3 current_pos = make_scalar3(h_pos.data[idx].x, h_pos.data[idx].y, h_pos.data[idx].z);

        vec3<Scalar> norm = normalize(vec3<Scalar>(m_manifold.derivative(current_pos)));

        Scalar delta_theta = hoomd::NormalDistribution<Scalar>(rotation_constant)(rng);

        quat<Scalar> rot_quat = quat<Scalar>::fromAxisAngle(norm, delta_theta);

        quati = rot_quat * quati; // rotational diffusion quaternion applied to orientation
        quati = quati * (Scalar(1.0) / slow::sqrt(norm2(quati)));
        h_orientation.data[idx] = quat_to_scalar4(quati);
        }
    }

/*! This function sets a manifold constraint for all active particles. Torque is not considered here
 */
template<class Manifold> void ActiveForceConstraintCompute<Manifold>::setConstraint()
    {
    //  array handles
    ArrayHandle<Scalar4> h_f_actVec(m_f_activeVec, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                       access_location::host,
                                       access_mode::readwrite);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);

    assert(h_f_actVec.data != NULL);
    assert(h_pos.data != NULL);
    assert(h_orientation.data != NULL);

    for (unsigned int i = 0; i < m_group->getNumMembers(); i++)
        {
        unsigned int idx = m_group->getMemberIndex(i);
        unsigned int type = __scalar_as_int(h_pos.data[idx].w);

        if (h_f_actVec.data[type].w != 0)
            {
            Scalar3 current_pos
                = make_scalar3(h_pos.data[idx].x, h_pos.data[idx].y, h_pos.data[idx].z);

            vec3<Scalar> norm = normalize(vec3<Scalar>(m_manifold.derivative(current_pos)));

            vec3<Scalar> f(h_f_actVec.data[type].x,
                           h_f_actVec.data[type].y,
                           h_f_actVec.data[type].z);
            quat<Scalar> quati(h_orientation.data[idx]);
            vec3<Scalar> fi = rotate(quati,
                                     f); // rotate active force vector from local to global frame

            Scalar dot_prod = dot(fi, norm);

            Scalar dot_perp_prod = slow::rsqrt(1 - dot_prod * dot_prod);

            Scalar phi = slow::atan(dot_prod * dot_perp_prod);

            fi.x -= norm.x * dot_prod;
            fi.y -= norm.y * dot_prod;
            fi.z -= norm.z * dot_prod;

            Scalar new_norm = slow::rsqrt(dot(fi, fi));
            fi *= new_norm;

            vec3<Scalar> rot_vec = cross(norm, fi);
            quat<Scalar> rot_quat = quat<Scalar>::fromAxisAngle(rot_vec, phi);

            quati = rot_quat * quati;
            quati = quati * (Scalar(1.0) / slow::sqrt(norm2(quati)));
            h_orientation.data[idx] = quat_to_scalar4(quati);
            }
        }
    }

/*! This function applies constraints, rotational diffusion, and sets forces for all active
   particles \param timestep Current timestep
*/
template<class Manifold>
void ActiveForceConstraintCompute<Manifold>::computeForces(uint64_t timestep)
    {
    if (m_box_changed)
        {
        if (!m_manifold.fitsInsideBox(m_pdata->getGlobalBox()))
            {
            throw std::runtime_error("Parts of the manifold are outside the box");
            }
        m_box_changed = false;
        }

    setConstraint(); // apply manifold constraints to active particles active force vectors

    setForces(); // set forces for particles

#ifdef ENABLE_HIP
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
#endif
    }

namespace detail
    {
template<class Manifold>
void export_ActiveForceConstraintCompute(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<ActiveForceConstraintCompute<Manifold>,
                     ActiveForceCompute,
                     std::shared_ptr<ActiveForceConstraintCompute<Manifold>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ParticleGroup>,
                            Manifold>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
