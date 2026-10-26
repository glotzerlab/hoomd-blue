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
					   Scalar epsilon,
					   Scalar R)

    : ForceCompute(sysdef), m_group(group), m_epsilon(epsilon), m_R(R)
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

	Scalar dist = 2*(m_R - norm);

	dist = 1/(dist*dist*norm);

        vec3<Scalar> fi(0, 0, 0);

	fi.x = -m_epsilon*pi.y*dist;
	fi.y = m_epsilon*pi.x*dist;
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
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>, Scalar, Scalar>())
        .def("setEpsilon", &WallCouplingForceCompute::setEpsilon)
        .def("getEpsilon", &WallCouplingForceCompute::getEpsilon)
        .def("setR", &WallCouplingForceCompute::setR)
        .def("getR", &WallCouplingForceCompute::getR)
        .def_property_readonly("filter",
                               [](WallCouplingForceCompute& force)
                               { return force.getGroup()->getFilter(); });
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
