// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ShearForceCompute.h"

#include <vector>

namespace hoomd
    {
namespace md
    {
/*! \file ShearForceCompute.cc
    \brief Contains code for the ShearForceCompute class
*/

/*! \param Constant force applied on a group of particles.
 */
ShearForceCompute::ShearForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                                           std::shared_ptr<ParticleGroup> group,
					   Scalar shear_force)

    : ForceCompute(sysdef), m_group(group), m_shear_force(2/m_pdata->getBox().getL().y*shear_force)
    {
    }

ShearForceCompute::~ShearForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying ShearForceCompute" << std::endl;
    }

/*! This function sets appropriate constant forces on all constant particles.
 */
void ShearForceCompute::setForces()
    {
    //  array handles
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    m_force.zeroFill();


    for (unsigned int i = 0; i < m_group->getNumMembers(); i++)
        {
        unsigned int idx = m_group->getMemberIndex(i);
        vec3<Scalar> fi(m_shear_force*h_pos.data[idx].y, 0, 0);
        h_force.data[idx] = vec_to_scalar4(fi, 0);
        }
    }

/*! This function applies rotational diffusion and sets forces for all constant particles
    \param timestep Current timestep
*/
void ShearForceCompute::computeForces(uint64_t timestep)
    {
    setForces(); // set forces for particles

#ifdef ENABLE_HIP
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
#endif
    }

namespace detail
    {
void export_ShearForceCompute(pybind11::module& m)
    {
    pybind11::class_<ShearForceCompute, ForceCompute, std::shared_ptr<ShearForceCompute>>(
        m,
        "ShearForceCompute")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>, Scalar>())
        .def("setShearForce", &ShearForceCompute::setShearForce)
        .def("getShearForce", &ShearForceCompute::getShearForce)
        .def_property_readonly("filter",
                               [](ShearForceCompute& force)
                               { return force.getGroup()->getFilter(); });
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
