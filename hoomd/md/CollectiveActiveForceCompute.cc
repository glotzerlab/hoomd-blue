// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "CollectiveActiveForceCompute.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"

namespace hoomd
    {
namespace md
    {
/*! \file CollectiveActiveForceCompute.cc
    \brief Contains code for the CollectiveActiveForceCompute class
*/

/*! \param sysdef The system definition
    \param group Particle group the force is applied to

    The per-type active force vectors are initialized by ActiveForceCompute
 */
CollectiveActiveForceCompute::CollectiveActiveForceCompute(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<ParticleGroup> group)
    : ActiveForceCompute(sysdef, group)
    {
    m_exec_conf->msg->notice(5) << "Constructing CollectiveActiveForceCompute" << std::endl;
    }

CollectiveActiveForceCompute::~CollectiveActiveForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying CollectiveActiveForceCompute" << std::endl;
    }

/*! Applies the active force vector of each particle type in the box frame, to
    every member of the group. Particle orientations are not read, and no torque is applied.
 */
void CollectiveActiveForceCompute::setForces()
    {
    ArrayHandle<Scalar4> h_f_actVec(m_f_activeVec, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);

    assert(h_f_actVec.data != NULL);
    assert(h_pos.data != NULL);

    // zero forces so we do not leave any forces set for indices that are no longer part of our group
    m_force.zeroFill();

    for (unsigned int i = 0; i < num_members; i++)
        {
        unsigned int idx = m_group->getMemberIndex(i);
        unsigned int type = __scalar_as_int(h_pos.data[idx].w);

        const Scalar4 f_activeVec = h_f_actVec.data[type];
        const vec3<Scalar> f(f_activeVec.w * f_activeVec.x,
                             f_activeVec.w * f_activeVec.y,
                             f_activeVec.w * f_activeVec.z);
        h_force.data[idx] = vec_to_scalar4(f, 0);
        }
    }

/*! Applies rotational diffusion to the active force vector of each particle type. The direction
    is rotated and the magnitude is preserved. Every type with a nonzero active force is diffused,
    whether or not this rank owns any of its particles, and the random numbers depend only on the
    timestep, the seed and the type id. All MPI ranks therefore keep identical vectors without
    communication.

    \param rotational_diffusion Rotational diffusion constant
    \param timestep Current timestep
*/
void CollectiveActiveForceCompute::rotationalDiffusion(Scalar rotational_diffusion,
                                                       uint64_t timestep)
    {
    ArrayHandle<Scalar4> h_f_actVec(m_f_activeVec, access_location::host, access_mode::readwrite);
    assert(h_f_actVec.data != NULL);

    const auto rotation_constant = slow::sqrt(2.0 * rotational_diffusion * m_deltaT);
    const unsigned int n_types = m_pdata->getNTypes();
    const bool is_2d = m_sysdef->getNDimensions() == 2;

    for (unsigned int type = 0; type < n_types; type++)
        {
        if (h_f_actVec.data[type].w == 0)
            {
            continue;
            }

        hoomd::RandomGenerator rng(hoomd::Seed(hoomd::RNGIdentifier::CollectiveActiveForceCompute,
                                               timestep,
                                               m_sysdef->getSeed()),
					       hoomd::Counter(type));

        vec3<Scalar> f(h_f_actVec.data[type].x,
                       h_f_actVec.data[type].y,
                       h_f_actVec.data[type].z);
        quat<Scalar> rot_quat;

        if (is_2d)
            {
            // rotate about the out of plane axis
            Scalar delta_theta = hoomd::NormalDistribution<Scalar>(rotation_constant)(rng);
            rot_quat = quat<Scalar>::fromAxisAngle(vec3<Scalar>(0, 0, 1.0), delta_theta);
            }
        else // 3D: Following Stenhammar, Soft Matter, 2014
            {
            hoomd::SpherePointGenerator<Scalar> unit_vec;
            vec3<Scalar> rand_vec;
            unit_vec(rng, rand_vec);

            vec3<Scalar> aux_vec = cross(f, rand_vec); // rotation axis
            aux_vec *= slow::rsqrt(dot(aux_vec, aux_vec));

            Scalar delta_theta = hoomd::NormalDistribution<Scalar>(rotation_constant)(rng);
            rot_quat = quat<Scalar>::fromAxisAngle(aux_vec, delta_theta);
            }

        f = rotate(rot_quat, f);
        f *= slow::rsqrt(dot(f, f)); // remove accumulated round off from the unit vector

        h_f_actVec.data[type].x = f.x;
        h_f_actVec.data[type].y = f.y;
        h_f_actVec.data[type].z = f.z;
        }
    }

    } // end namespace md
    } // end namespace hoomd
