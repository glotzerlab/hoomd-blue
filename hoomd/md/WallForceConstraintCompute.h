// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "NeighborList.h"
#include "hoomd/ForceConstraint.h"
#include "hoomd/ParticleGroup.h"

#ifdef ENABLE_HIP
#include "hoomd/Autotuner.h"
#endif

/*! \file WallForceConstraintCompute.h
    \brief Declares a class for computing wall constraint and friction forces
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __WALLFORCECONSTRAINTCOMPUTE_H__
#define __WALLFORCECONSTRAINTCOMPUTE_H__

namespace hoomd
    {
namespace md
    {
struct wall_constraint_params
    {
    Scalar k;
    Scalar mus;
    Scalar muk;
    Scalar Tb;
    Scalar nus;


#ifndef __HIPCC__
    wall_constraint_params() : k(0), mus(0), muk(0), Tb(0), nus(0) { }

    wall_constraint_params(pybind11::dict params)
        : k(params["k"].cast<Scalar>()), mus(params["mu_s"].cast<Scalar>()), muk(params["mu_k"].cast<Scalar>()), Tb(params["T_b"].cast<Scalar>()), nus(params["nu_s"].cast<Scalar>())
        {
        }

    pybind11::dict asDict()
        {
        pybind11::dict v;
        v["k"] = k;
        v["mu_s"] = mus;
        v["mu_k"] = muk;
        v["T_b"] = Tb;
        v["nu_s"] = nus;
        return v;
        }
#endif
    }
#if HOOMD_LONGREAL_SIZE == 32
    __attribute__((aligned(8)));
#else
    __attribute__((aligned(16)));
#endif

//! Adds an active force to a number of particles
/*! \ingroup computes
 */
template<class Manifold>
class PYBIND11_EXPORT WallForceConstraintCompute : public ForceConstraint
    {
    public:
    //! Constructs the compute
    WallForceConstraintCompute(std::shared_ptr<SystemDefinition> sysdef,
                                 std::shared_ptr<ParticleGroup> group,
                                 Manifold manifold,
				 bool brownian);
    //
    //! Destructor
    ~WallForceConstraintCompute();

    //! Set the parameters
    virtual void setParams(unsigned int type, Scalar k, Scalar mus, Scalar muk, Scalar Tb, Scalar nus);

    virtual void setParamsPython(std::string type, pybind11::dict params);

    /// Get the parameters for a type
    pybind11::dict getParams(std::string type);

    std::shared_ptr<ParticleGroup>& getGroup()
        {
        return m_group;
        }

    protected:
    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);

    //! Compute constraint forces
    virtual void computeConstraintForces();

    //! Compute friction forces
    virtual void computeFrictionForces();

    std::shared_ptr<ParticleGroup> m_group; //!< Group of particles on which this force is applied
    Manifold m_manifold; //!< Constraining Manifold
    bool m_brownian;
    Scalar* m_k;   //!< k harmonic spring constant
    Scalar* m_mus; //!< mus stick friction coefficient
    Scalar* m_muk; //!< mus kinetic friction coefficient
    Scalar* m_Tb; //!< mus stick friction coefficient
    Scalar* m_nus; //!< mus stick friction coefficient
    };

/*! \param sysdef The system definition
    \param manifold Manifold constraint
 */
template<class Manifold>
WallForceConstraintCompute<Manifold>::WallForceConstraintCompute(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<ParticleGroup> group,
    Manifold manifold,
    bool brownian)
    : ForceConstraint(sysdef), m_group(group), m_manifold(manifold), m_brownian(brownian), m_k(NULL), m_mus(NULL), m_muk(NULL), m_Tb(NULL), m_nus(NULL)
    {
    m_exec_conf->msg->notice(5) << "Constructing WallForceConstraintCompute" << std::endl;
    m_k = new Scalar[m_pdata->getNTypes()];
    m_mus = new Scalar[m_pdata->getNTypes()];
    m_muk = new Scalar[m_pdata->getNTypes()];
    m_Tb = new Scalar[m_pdata->getNTypes()];
    m_nus = new Scalar[m_pdata->getNTypes()];
    }

template<class Manifold> WallForceConstraintCompute<Manifold>::~WallForceConstraintCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying WallForceConstraintCompute" << std::endl;

    delete[] m_k;
    delete[] m_mus;
    delete[] m_muk;
    delete[] m_Tb;
    delete[] m_nus;
    m_k = NULL;
    m_mus = NULL;
    m_muk = NULL;
    m_Tb = NULL;
    m_nus = NULL;
    }

/*! \param type Type of the wall constraint to set parameters for
    \param k Spring constant for the force computation between wall and particles
    \param mus Stick friction constant of the wall 
    \param muk Kinetic friction constant of the wall 

    Sets parameters for the potential of a particular particle type
*/
template<class Manifold>
void WallForceConstraintCompute<Manifold>::setParams(unsigned int type, Scalar k, Scalar mus, Scalar muk, Scalar Tb, Scalar nus)
    {
    // make sure the type is valid
    if (type >= m_pdata->getNTypes())
        {
        throw std::runtime_error("Invalid particle type.");
        }

    m_k[type] = k;
    m_mus[type] = mus;
    m_muk[type] = muk;
    m_Tb[type] = Tb;
    m_nus[type] = nus;
    }

template<class Manifold>
void WallForceConstraintCompute<Manifold>::setParamsPython(std::string type, pybind11::dict params)
    {
    auto typ = m_pdata->getTypeByName(type);
    auto _params = wall_constraint_params(params);
    setParams(typ, _params.k, _params.mus, _params.muk, _params.Tb, _params.nus);
    }

template<class Manifold>
pybind11::dict WallForceConstraintCompute<Manifold>::getParams(std::string type)
    {
    auto typ = m_pdata->getTypeByName(type);
    if (typ >= m_pdata->getNTypes())
        {
        throw std::runtime_error("Invalid particle type.");
        }
    pybind11::dict params;
    params["k"] = m_k[typ];
    params["mu_s"] = m_mus[typ];
    params["mu_k"] = m_muk[typ];
    params["T_b"] = m_Tb[typ];
    params["nu_s"] = m_nus[typ];
    return params;
    }


/*! This function calculates and applies a friction force and hookian wall constraint forces for all
   particles \param timestep Current timestep
*/
template<class Manifold>
void WallForceConstraintCompute<Manifold>::computeForces(uint64_t timestep)
    {

    computeFrictionForces(); // calculate friction forces

    computeConstraintForces(); // calculate forces 

#ifdef ENABLE_HIP
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
#endif
    }

template<class Manifold>
void WallForceConstraintCompute<Manifold>::computeFrictionForces()
    {
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar4> h_torque(m_torque, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);

    ArrayHandle<Scalar4> h_net_force(m_pdata->getNetForce(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_net_torque(m_pdata->getNetTorqueArray(),access_location::host,access_mode::read);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_angmom(m_pdata->getAngularMomentumArray(), access_location::host, access_mode::read);

    // sanity check
    assert(h_pos.data != NULL);
    assert(h_orientation.data != NULL);
    assert(h_vel.data != NULL);
    assert(h_angmom.data != NULL);
    assert(h_net_force.data != NULL);
    assert(h_net_torque.data != NULL);

    // zero forces so we don't leave any forces set for indices that are no longer part of our group
    memset(h_force.data, 0, sizeof(Scalar4) * m_force.getNumElements());
    memset(h_torque.data, 0, sizeof(Scalar4) * m_force.getNumElements());
    memset((void*)h_virial.data, 0, sizeof(Scalar) * m_virial.getNumElements());

    assert(h_force.data);
    assert(h_torque.data);
    assert(h_virial.data);

    // for each particle
    for (unsigned int i = 0; i < m_group->getNumMembers(); i++)
        {
        unsigned int idx = m_group->getMemberIndex(i);
        unsigned int typei = __scalar_as_int(h_pos.data[idx].w);
        // sanity check
        assert(typei < m_pdata->getNTypes());

        // access the particle's position and type (MEM TRANSFER: 4 scalars)
        Scalar3 pi = make_scalar3(h_pos.data[idx].x, h_pos.data[idx].y, h_pos.data[idx].z);
	if(m_manifold.implicitFunction(pi) >= 0 )
		continue;

	vec3<Scalar> norm = -normalize(vec3<Scalar>(m_manifold.derivative(pi))); 
	vec3<Scalar> net_force(h_net_force.data[idx].x,h_net_force.data[idx].y,h_net_force.data[idx].z);
	Scalar normal_magnitude = dot(norm,net_force);

	if(normal_magnitude < 0)
		continue;

	 vec3<Scalar> perp_vel, perp_force, para_ang_mom, para_torque;
	 Scalar norm_perp_vel = -1;
	 Scalar norm_para_angmom = 0;

	 vec3<Scalar> net_tor(h_net_torque.data[idx].x,h_net_torque.data[idx].y,h_net_torque.data[idx].z);
	 Scalar normal_tor = dot(norm,net_tor);

	 if(!m_brownian)
	 	{
            	quat<Scalar> q(h_orientation.data[idx]);
            	quat<Scalar> p(h_angmom.data[idx]);
            	vec3<Scalar> net_angmom = (Scalar(1. / 2.) * conj(q) * p).v;
		norm_para_angmom = dot(norm,net_angmom);

		vec3<Scalar> net_vel(h_vel.data[idx].x,h_vel.data[idx].y,h_vel.data[idx].z);
		Scalar normal_vel = dot(norm,net_vel);
		perp_vel = -net_vel + normal_vel*norm;
		norm_perp_vel = fast::sqrt(dot(perp_vel,perp_vel));
		}

	 Scalar Tb = m_Tb[typei];
	 Scalar Tbn = m_nus[typei]*Tb;

	if( norm_para_angmom > 1e-6 || norm_para_angmom < -1e-6 )
		{
		if( norm_para_angmom > 0) 
			Tbn = -Tbn;
		para_torque = norm*Tbn;
	   	}
	else
		{
		 if( normal_tor > Tb) 
			 para_torque = -norm*Tbn;
		 else
			{
			if(normal_tor < -Tb) 
				para_torque = norm*Tbn;
			else
				para_torque = -norm*normal_tor;
			}
		}


	if( norm_perp_vel > 1e-6)
		{
		perp_force = perp_vel*normal_magnitude*m_muk[typei]/norm_perp_vel;
	   	}
	else
		{
		perp_force =  -net_force + normal_magnitude*norm;
		Scalar perp_magnitude = fast::sqrt(dot(perp_force,perp_force));

		if( perp_magnitude > m_mus[typei]*normal_magnitude)
			perp_force *= (normal_magnitude*m_muk[typei]/perp_magnitude);
		}


        h_force.data[idx].x = perp_force.x;
        h_force.data[idx].y = perp_force.y;
        h_force.data[idx].z = perp_force.z;

        h_torque.data[idx].x = para_torque.x;
        h_torque.data[idx].y = para_torque.y;
        h_torque.data[idx].z = para_torque.z;

        h_virial.data[0 * m_virial_pitch + idx] = perp_force.x * pi.x;
        h_virial.data[1 * m_virial_pitch + idx] = perp_force.x * pi.y;
        h_virial.data[2 * m_virial_pitch + idx] = perp_force.x * pi.z;
        h_virial.data[3 * m_virial_pitch + idx] = perp_force.y * pi.y;
        h_virial.data[4 * m_virial_pitch + idx] = perp_force.y * pi.z;
        h_virial.data[5 * m_virial_pitch + idx] = perp_force.z * pi.z;

	}

    }

template<class Manifold>
void WallForceConstraintCompute<Manifold>::computeConstraintForces()
    {
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::readwrite);

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);

    // sanity check
    assert(h_pos.data != NULL);

    // for each particle
    for (unsigned int i = 0; i < m_group->getNumMembers(); i++)
        {
        unsigned int idx = m_group->getMemberIndex(i);
        unsigned int typei = __scalar_as_int(h_pos.data[idx].w);
        // sanity check
        assert(typei < m_pdata->getNTypes());

        // access the particle's position and type (MEM TRANSFER: 4 scalars)
        Scalar3 pi = make_scalar3(h_pos.data[idx].x, h_pos.data[idx].y, h_pos.data[idx].z);
	Scalar distance = m_manifold.implicitFunction(pi) ;

        if(distance >= 0)
		continue;

	vec3<Scalar> norm = normalize(vec3<Scalar>(m_manifold.derivative(pi))); 

	norm *= (-m_k[typei]*distance);

        h_force.data[idx].x += norm.x;
        h_force.data[idx].y += norm.y;
        h_force.data[idx].z += norm.z;

        h_virial.data[0 * m_virial_pitch + idx] += norm.x * pi.x;
        h_virial.data[1 * m_virial_pitch + idx] += norm.x * pi.y;
        h_virial.data[2 * m_virial_pitch + idx] += norm.x * pi.z;
        h_virial.data[3 * m_virial_pitch + idx] += norm.y * pi.y;
        h_virial.data[4 * m_virial_pitch + idx] += norm.y * pi.z;
        h_virial.data[5 * m_virial_pitch + idx] += norm.z * pi.z;
	}
    }

namespace detail
    {
template<class Manifold>
void export_WallForceConstraintCompute(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<WallForceConstraintCompute<Manifold>,
                     ForceConstraint,
                     std::shared_ptr<WallForceConstraintCompute<Manifold>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ParticleGroup>,
                            Manifold, bool>())
        .def("setParams", &WallForceConstraintCompute<Manifold>::setParamsPython)
        .def("getParams", &WallForceConstraintCompute<Manifold>::getParams);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
