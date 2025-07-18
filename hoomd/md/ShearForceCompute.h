// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/ForceCompute.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleGroup.h"
#include "hoomd/VectorMath.h"
#include <memory>

/*! \file ShearForceCompute.h
    \brief Declares a class for computing shear forces
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __SHEARFORCECOMPUTE_H__
#define __SHEARFORCECOMPUTE_H__

namespace hoomd
    {
namespace md
    {
//! Adds an shear force to the particles
/*! \ingroup computes
 */
class PYBIND11_EXPORT ShearForceCompute : public ForceCompute
    {
    public:
    //! Constructs the compute
    ShearForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                         std::shared_ptr<ParticleGroup> group,
			 Scalar shear_force);

    //! Destructor
    ~ShearForceCompute();

    /** Sets max force for a given particle type
        @param typ Particle type to set constant force vector
        @param v The constant force vector value to set (a 3-tuple)
    */
    void setShearForce(Scalar shear_force)
	{
	m_shear_force = 2/m_pdata->getBox().getL().y*shear_force;
	};

    /// Gets constant force vector for a given particle type
    Scalar getShearForce()
   	{
	return m_pdata->getBox().getL().y*m_shear_force/2;
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
    Scalar m_shear_force; //! constant force unit vectors and magnitudes for each particle type
    };

    } // end namespace md
    } // end namespace hoomd
#endif
