// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/ForceCompute.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleGroup.h"
#include "hoomd/VectorMath.h"
#include <memory>

/*! \file ConstantForceCompute.h
    \brief Declares a class for computing constant forces and torques
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __CONSTANTFORCECOMPUTE_H__
#define __CONSTANTFORCECOMPUTE_H__

namespace hoomd
    {
namespace md
    {
//! Adds an constant force to a number of particles
/*! \ingroup computes
 */
class PYBIND11_EXPORT ConstantForceCompute : public ForceCompute
    {
    public:
    //! Constructs the compute
    ConstantForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                         std::shared_ptr<ParticleGroup> group);

    //! Destructor
    ~ConstantForceCompute();

    /** Sets constant force vector for a given particle type
        @param typ Particle type to set constant force vector
        @param v The constant force vector value to set (a 3-tuple)
    */
    void setConstantForce(const std::string& type_name, pybind11::tuple v);

    /// Gets constant force vector for a given particle type
    pybind11::tuple getConstantForce(const std::string& type_name);

    /** Sets constant torque vector for a given particle type
        @param typ Particle type to set constant torque vector
        @param v The constant torque vector value to set (a 3-tuple)
    */
    void setConstantTorque(const std::string& type_name, pybind11::tuple v);

    /// Gets constant torque vector for a given particle type
    pybind11::tuple getConstantTorque(const std::string& type_name);

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
    GlobalVector<Scalar3>
        m_constant_force; //! constant force unit vectors and magnitudes for each particle type

    GlobalVector<Scalar3>
        m_constant_torque; //! constant torque unit vectors and magnitudes for each particle type

    bool m_parameters_updated; //!< True if forces need to be rearranged
    };

    } // end namespace md
    } // end namespace hoomd
#endif
