// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "hoomd/RandomNumbers.h"
#include "hoomd/RNGIdentifiers.h"
#include "ActiveForceCompute.h"

/*! \file ActiveForceConstraintCompute.h
    \brief Declares a class for computing active forces and torques
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __ACTIVEFORCECONSTRAINTCOMPUTE_H__
#define __ACTIVEFORCECONSTRAINTCOMPUTE_H__

using namespace std;
namespace py = pybind11;

//! Adds an active force to a number of particles
/*! \ingroup computes
*/
template < class Manifold>
class PYBIND11_EXPORT ActiveForceConstraintCompute : public ActiveForceCompute
    {
    public:
        //! Constructs the compute
        ActiveForceConstraintCompute(std::shared_ptr<SystemDefinition> sysdef,
                             std::shared_ptr<ParticleGroup> group,
                             Scalar rotation_diff,
                             Manifold manifold)
        : ActiveForceCompute(sysdef,group,rotation_diff), m_manifold(manifold){}

        //! Destructor
        ~ActiveForceConstraintCompute()
        {
          m_exec_conf->msg->notice(5) << "Destroying ActiveForceConstraintCompute" << endl;
        }

    protected:
        //! Actually compute the forces
        virtual void computeForces(uint64_t timestep);

        //! Orientational diffusion for spherical particles
        virtual void rotationalDiffusion(uint64_t timestep);

        //! Set constraints if particles confined to a surface
        virtual void setConstraint();

        Manifold m_manifold;          //!< Constraining Manifold
    };



/*! This function applies rotational diffusion to the orientations of all active particles. The orientation of any torque vector
 * relative to the force vector is preserved
    \param timestep Current timestep
*/
template < class Manifold>
void ActiveForceConstraintCompute<Manifold>::rotationalDiffusion(uint64_t timestep)
    {
    //  array handles
    ArrayHandle<Scalar4> h_pos(m_pdata -> getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

    assert(h_pos.data != NULL);
    assert(h_orientation.data != NULL);
    assert(h_tag.data != NULL);

    for (unsigned int i = 0; i < m_group->getNumMembers(); i++)
        {
        unsigned int idx = m_group->getMemberIndex(i);
        unsigned int ptag = h_tag.data[idx];
        hoomd::RandomGenerator rng(hoomd::Seed(hoomd::RNGIdentifier::ActiveForceCompute,
                                               timestep,
                                               m_sysdef->getSeed()),
                                               hoomd::Counter(ptag));

        quat<Scalar> quati(h_orientation.data[idx]);

        Scalar3 current_pos = make_scalar3(h_pos.data[idx].x, h_pos.data[idx].y, h_pos.data[idx].z);
        Scalar3 norm_scalar3 = m_manifold.derivative(current_pos);
        Scalar norm_normal = fast::rsqrt(dot(norm_scalar3,norm_scalar3));
        norm_scalar3 *= norm_normal;
        vec3<Scalar> norm = vec3<Scalar> (norm_scalar3);

        Scalar delta_theta = hoomd::NormalDistribution<Scalar>(m_rotationConst)(rng);
        Scalar theta = delta_theta/2.0; // half angle to calculate the quaternion which represents the rotation
        quat<Scalar> rot_quat(slow::cos(theta),slow::sin(theta)*norm);//rotational diffusion quaternion

        quati = rot_quat*quati; //rotational diffusion quaternion applied to orientation
        h_orientation.data[idx] = quat_to_scalar4(quati);
        }
    }

/*! This function sets a manifold constraint for all active particles. Torque is not considered here
*/
template < class Manifold>
void ActiveForceConstraintCompute<Manifold>::setConstraint()
    {
    //  array handles
    ArrayHandle<Scalar4> h_f_actVec(m_f_activeVec, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_pos(m_pdata -> getPositions(), access_location::host, access_mode::read);

    assert(h_f_actVec.data != NULL);
    assert(h_pos.data != NULL);
    assert(h_orientation.data != NULL);

    if(!m_manifold.fitsInsideBox(m_pdata->getGlobalBox()))
        {
        throw std::runtime_error("Parts of the manifold are outside the box");
        }


    for (unsigned int i = 0; i < m_group->getNumMembers(); i++)
        {
        unsigned int idx = m_group->getMemberIndex(i);
        unsigned int type = __scalar_as_int(h_pos.data[idx].w);

	if( h_f_actVec.data[type].w != 0){

            Scalar3 current_pos = make_scalar3(h_pos.data[idx].x, h_pos.data[idx].y, h_pos.data[idx].z);

            Scalar3 norm_scalar3 = m_manifold.derivative(current_pos);
            Scalar norm_normal = fast::rsqrt(dot(norm_scalar3,norm_scalar3));
            norm_scalar3 *= norm_normal;
            vec3<Scalar> norm = vec3<Scalar> (norm_scalar3);

            Scalar3 f = make_scalar3(h_f_actVec.data[type].x, h_f_actVec.data[type].y, h_f_actVec.data[type].z);
            quat<Scalar> quati(h_orientation.data[idx]);
            vec3<Scalar> fi = rotate(quati, vec3<Scalar>(f));//rotate active force vector from local to global frame

            Scalar dot_prod = dot(fi,norm);

            Scalar dot_perp_prod = slow::rsqrt(1-dot_prod*dot_prod);

            Scalar phi_half = slow::atan(dot_prod*dot_perp_prod)/2.0;

            fi.x -= norm.x * dot_prod;
            fi.y -= norm.y * dot_prod;
            fi.z -= norm.z * dot_prod;

            Scalar new_norm = slow::rsqrt(dot(fi,fi));
            fi *= new_norm;

            vec3<Scalar> rot_vec = cross(norm,fi);
            rot_vec *= fast::sin(phi_half);

            quat<Scalar> rot_quat(cos(phi_half),rot_vec);

            quati = rot_quat*quati;

            h_orientation.data[idx] = quat_to_scalar4(quati);
	    }
        }
    }

/*! This function applies constraints, rotational diffusion, and sets forces for all active particles
    \param timestep Current timestep
*/
template < class Manifold>
void ActiveForceConstraintCompute<Manifold>::computeForces(uint64_t timestep)
    {
    if (m_prof) m_prof->push(m_exec_conf, "ActiveForceConstraintCompute");

    if (last_computed != timestep)
        {
        m_rotationConst = slow::sqrt(2.0 * m_rotationDiff * m_deltaT);

        last_computed = timestep;

        setConstraint(); // apply manifold constraints to active particles active force vectors

        if (m_rotationDiff != 0)
            {
            rotationalDiffusion(timestep); // apply rotational diffusion to active particles
            }
        setForces(); // set forces for particles
        }

    #ifdef ENABLE_HIP
    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    #endif

    if (m_prof)
        m_prof->pop(m_exec_conf);
    }


template < class Manifold>
void export_ActiveForceConstraintCompute(py::module& m, const std::string& name)
    {
    py::class_< ActiveForceConstraintCompute<Manifold>, ActiveForceCompute, std::shared_ptr<ActiveForceConstraintCompute<Manifold> > >(m, name.c_str() )
    .def(py::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>, Scalar,
                    Manifold >())
    ;
    }

#endif
