// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "WallDistanceForceCompute.h"

#include <vector>

namespace hoomd
    {
namespace md
    {
/*! \file WallDistanceForceCompute.cc
    \brief Contains code for the WallDistanceForceCompute class
*/

/*! \param Constant force applied on a group of particles.
 */
WallDistanceForceCompute::WallDistanceForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                                           std::shared_ptr<ParticleGroup> group,
					   Scalar k,
					   Scalar R,
					   bool inverse)

    : ForceCompute(sysdef), m_group(group), m_k(k), m_R(R), m_inverse(inverse)
    {
    }

WallDistanceForceCompute::~WallDistanceForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying WallDistanceForceCompute" << std::endl;
    }

/*! This function sets appropriate constant forces on all constant particles.
 */
void WallDistanceForceCompute::setForces()
    {
    //  array handles
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    m_force.zeroFill();


    for (unsigned int i = 0; i < m_group->getNumMembers(); i++)
        {
        unsigned int idx = m_group->getMemberIndex(i);

        Scalar3 pi = make_scalar3(h_pos.data[idx].x, h_pos.data[idx].y, 0);

        vec3<Scalar> fi(0, 0, 0);

	if(m_inverse)
	{
		Scalar norm = fast::sqrt(pi.x*pi.x+pi.y*pi.y);
		if( norm > 0)
		{
			Scalar dist = 1/(m_R - norm);
			dist = (dist*dist);
			fi.x = m_k*dist*pi.x/norm;
			fi.y = m_k*dist*pi.y/norm;
		}
	}
	else{
		fi.x = m_k*pi.x;
		fi.y = m_k*pi.y;
	}

        h_force.data[idx] = vec_to_scalar4(fi, 0);
        }
    }

/*! This function applies rotational diffusion and sets forces for all constant particles
    \param timestep Current timestep
*/
void WallDistanceForceCompute::computeForces(uint64_t timestep)
    {
    setForces(); // set forces for particles

#ifdef ENABLE_HIP
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
#endif
    }

namespace detail
    {
void export_WallDistanceForceCompute(pybind11::module& m)
    {
    pybind11::class_<WallDistanceForceCompute, ForceCompute, std::shared_ptr<WallDistanceForceCompute>>(
        m,
        "WallDistanceForceCompute")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>, Scalar, Scalar, bool>())
        .def("setK", &WallDistanceForceCompute::setK)
        .def("getK", &WallDistanceForceCompute::getK)
        .def("setR", &WallDistanceForceCompute::setR)
        .def("getR", &WallDistanceForceCompute::getR)
        .def_property_readonly("filter",
                               [](WallDistanceForceCompute& force)
                               { return force.getGroup()->getFilter(); });
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
