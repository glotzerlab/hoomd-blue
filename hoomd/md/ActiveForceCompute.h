// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/ForceCompute.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/ParticleGroup.h"
#include "hoomd/VectorMath.h"
#include <memory>

/*! \file ActiveForceCompute.h
    \brief Declares a class for computing active forces and torques
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __ACTIVEFORCECOMPUTE_H__
#define __ACTIVEFORCECOMPUTE_H__

namespace hoomd
    {
namespace md
    {
// Forward declaration is necessary to avoid circular includes between this and
// ActiveRotationalDiffusionUpdater.h while making ActiveRotationalDiffusionUpdater a friend class
// of ActiveForceCompute.
class ActiveRotationalDiffusionUpdater;

//! Adds an active force to a number of particles
/*! \ingroup computes
 */
class PYBIND11_EXPORT ActiveForceCompute : public ForceCompute
    {
    public:
    //! Constructs the compute
    ActiveForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                       std::shared_ptr<ParticleGroup> group);

    //! Destructor
    ~ActiveForceCompute();

    /** Sets active force vector for a given particle type
        @param typ Particle type to set active force vector
        @param v The active force vector value to set (a 3-tuple)
    */
    void setActiveForce(const std::string& type_name, pybind11::tuple v);

    /// Gets active force vector for a given particle type
    pybind11::tuple getActiveForce(const std::string& type_name);

    /** Sets active torque vector for a given particle type
        @param typ Particle type to set active torque vector
        @param v The active torque vector value to set (a 3-tuple)
    */
    void setActiveTorque(const std::string& type_name, pybind11::tuple v);

    /// Gets active torque vector for a given particle type
    pybind11::tuple getActiveTorque(const std::string& type_name);

    std::shared_ptr<ParticleGroup>& getGroup()
        {
        return m_group;
        }

    protected:
    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);

    //! Set forces for particles
    virtual void setForces();

    //! Orientational diffusion for spherical particles
    virtual void rotationalDiffusion(Scalar rotational_diffusion, uint64_t timestep);

    std::shared_ptr<ParticleGroup> m_group; //!< Group of particles on which this force is applied
    GlobalVector<Scalar4>
        m_f_activeVec; //! active force unit vectors and magnitudes for each particle type

    GlobalVector<Scalar4>
        m_t_activeVec; //! active torque unit vectors and magnitudes for each particle type

    private:
    // Allow ActiveRotationalDiffusionUpdater to access internal methods and members of
    // ActiveForceCompute classes/subclasses. This is necessary to allow
    // ActiveRotationalDiffusionUpdater to call rotationalDiffusion.
    friend class ActiveRotationalDiffusionUpdater;
    };

    } // end namespace md
    } // end namespace hoomd
#endif
