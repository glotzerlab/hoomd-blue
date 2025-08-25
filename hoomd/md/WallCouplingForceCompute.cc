// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "WallCouplingForceCompute.h"

#include <vector>

namespace hoomd
    {
namespace md
    {
/*! \file WallCouplingForceCompute.cc
    \brief Contains code for the WallCouplingForceCompute class
*/

/*! \param Constant force applied on a group of particles.
 */
WallCouplingForceCompute::WallCouplingForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                                           std::shared_ptr<ParticleGroup> group,
					   Scalar radial_force,
					   Scalar tangential_force,
					   Scalar lift,
					   Scalar R)

    : ForceCompute(sysdef), m_group(group), m_radial_force(radial_force), m_tangential_force(tangential_force), m_lift_force(lift), m_R(R)
    {
    }

WallCouplingForceCompute::~WallCouplingForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying WallCouplingForceCompute" << std::endl;
    }

/*! This function sets appropriate constant forces on all constant particles.
 */
void WallCouplingForceCompute::setForces()
    {
    //  array handles
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    m_force.zeroFill();


    for (unsigned int i = 0; i < m_group->getNumMembers(); i++)
        {
        unsigned int idx = m_group->getMemberIndex(i);

        Scalar3 pi = make_scalar3(h_pos.data[idx].x, h_pos.data[idx].y, 0);

	Scalar norm = fast::sqrt(pi.x*pi.x+pi.y*pi.y);

	Scalar dist1 = m_R - norm;

	Scalar dist2 = m_R + dist1;

	dist1 = 1/(dist1*dist1*dist1);
	dist2 = 1/(dist2*dist2*dist2);

	pi.x /= norm;
	pi.y /= norm;

        vec3<Scalar> fi(0, 0, 0);

	fi.x = (dist2-dist1)*(pi.x*m_radial_force + pi.y*m_tangential_force) + pi.x*m_lift_force;
	fi.y = (dist2-dist1)*(pi.y*m_radial_force - pi.x*m_tangential_force) + pi.y*m_lift_force;
        h_force.data[idx] = vec_to_scalar4(fi, 0);
        }
    }

/*! This function applies rotational diffusion and sets forces for all constant particles
    \param timestep Current timestep
*/
void WallCouplingForceCompute::computeForces(uint64_t timestep)
    {
    setForces(); // set forces for particles

#ifdef ENABLE_HIP
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
#endif
    }

namespace detail
    {
void export_WallCouplingForceCompute(pybind11::module& m)
    {
    pybind11::class_<WallCouplingForceCompute, ForceCompute, std::shared_ptr<WallCouplingForceCompute>>(
        m,
        "WallCouplingForceCompute")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>, Scalar, Scalar, Scalar, Scalar>())
        .def("setWallCouplingRadialForce", &WallCouplingForceCompute::setWallCouplingRadialForce)
        .def("getWallCouplingRadialForce", &WallCouplingForceCompute::getWallCouplingRadialForce)
        .def("setWallCouplingTangentialForce", &WallCouplingForceCompute::setWallCouplingTangentialForce)
        .def("getWallCouplingTangentialForce", &WallCouplingForceCompute::getWallCouplingTangentialForce)
        .def("setWallCouplingLiftForce", &WallCouplingForceCompute::setWallCouplingLiftForce)
        .def("getWallCouplingLiftForce", &WallCouplingForceCompute::getWallCouplingLiftForce)
        .def("setR", &WallCouplingForceCompute::setR)
        .def("getR", &WallCouplingForceCompute::getR)
        .def_property_readonly("filter",
                               [](WallCouplingForceCompute& force)
                               { return force.getGroup()->getFilter(); });
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
