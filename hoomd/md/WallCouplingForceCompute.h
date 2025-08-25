// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/ForceCompute.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleGroup.h"
#include "hoomd/VectorMath.h"
#include <memory>

/*! \file WallCouplingForceCompute.h
    \brief Declares a class for computing shear forces
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __WALLCOUPLINGFORCECOMPUTE_H__
#define __WALLCOUPLINGFORCECOMPUTE_H__

namespace hoomd
    {
namespace md
    {
//! Adds an shear force to the particles
/*! \ingroup computes
 */
class PYBIND11_EXPORT WallCouplingForceCompute : public ForceCompute
    {
    public:
    //! Constructs the compute
    WallCouplingForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                         std::shared_ptr<ParticleGroup> group,
			 Scalar radial_factor,
			 Scalar tangential_factor,
			 Scalar lift,
			 Scalar R);

    //! Destructor
    ~WallCouplingForceCompute();

    /** Sets max force for a given particle type
        @param typ Particle type to set constant force vector
        @param v The constant force vector value to set (a 3-tuple)
    */
    void setWallCouplingRadialForce(Scalar radial_force)
	{
	m_radial_force = radial_force;
	};

    /// Gets constant force vector for a given particle type
    Scalar getWallCouplingRadialForce()
   	{
	return m_radial_force;
    	}

    void setWallCouplingTangentialForce(Scalar tangential_force)
	{
	m_tangential_force = tangential_force;
	};

    /// Gets constant force vector for a given particle type
    Scalar getWallCouplingTangentialForce()
   	{
	return m_tangential_force;
    	}

    void setWallCouplingLiftForce(Scalar lift_force)
	{
	m_lift_force = lift_force;
	};

    /// Gets constant force vector for a given particle type
    Scalar getWallCouplingLiftForce()
   	{
	return m_lift_force;
    	}

    void setR(Scalar R)
	{
	m_R = R;
	};

    /// Gets constant force vector for a given particle type
    Scalar getR()
   	{
	return m_R;
    	}

    std::shared_ptr<ParticleGroup>& getGroup()
        {
        return m_group;
        }

    protected:
    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);

    //! Set forces for particles
    virtual void setForces();

    std::shared_ptr<ParticleGroup> m_group; //!< Group of particles on which this force is applied
    Scalar m_radial_force; //! constant force unit vectors and magnitudes for each particle type
    Scalar m_tangential_force; //! constant force unit vectors and magnitudes for each particle type
    Scalar m_lift_force; //! constant force unit vectors and magnitudes for each particle type
    Scalar m_R;
    };

    } // end namespace md
    } // end namespace hoomd
#endif
