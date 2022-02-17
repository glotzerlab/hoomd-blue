// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/CollisionMethod.h
 * \brief Declaration of mpcd::CollisionMethod
 */

#ifndef MPCD_COLLISION_METHOD_H_
#define MPCD_COLLISION_METHOD_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "SystemData.h"
#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace mpcd
    {
//! MPCD collision method
/*!
 * This class forms the generic base for an MPCD collision method. It handles the boiler plate of
 * setting up the method and implementing the collision. Each deriving class should implement a
 * rule() that gives the physics of the collision.
 */
class PYBIND11_EXPORT CollisionMethod
    {
    public:
    //! Constructor
    CollisionMethod(std::shared_ptr<mpcd::SystemData> sysdata,
                    uint64_t cur_timestep,
                    uint64_t period,
                    int phase);
    //! Destructor
    virtual ~CollisionMethod() { }

    //! Implementation of the collision rule
    void collide(uint64_t timestep);

    //! Peek if a collision will occur on this timestep
    virtual bool peekCollide(uint64_t timestep) const;

    //! Set autotuner parameters
    /*!
     * \param enable Enable/disable autotuning
     * \param period period (approximate) in time steps when returning occurs
     *
     * Derived classes should override this to set the parameters of their autotuners.
     */
    virtual void setAutotunerParams(bool enable, unsigned int period) { }

    //! Toggle the grid shifting on or off
    /*!
     * \param enable_grid_shift Flag to enable grid shifting if true
     */
    void enableGridShifting(bool enable_grid_shift)
        {
        m_enable_grid_shift = enable_grid_shift;
        }

    //! Generates the random grid shift vector
    void drawGridShift(uint64_t timestep);

    //! Sets a group of particles that is coupled to the MPCD solvent through the collision step
    /*!
     * \param embed_group Group to embed
     */
    void setEmbeddedGroup(std::shared_ptr<ParticleGroup> embed_group)
        {
        m_embed_group = embed_group;
        m_cl->setEmbeddedGroup(m_embed_group);
        }

    //! Set the period of the collision method
    void setPeriod(unsigned int cur_timestep, unsigned int period);

    /// Set the RNG instance
    void setInstance(unsigned int instance)
        {
        m_instance = instance;
        }

    /// Get the RNG instance
    unsigned int getInstance()
        {
        return m_instance;
        }

    protected:
    std::shared_ptr<mpcd::SystemData> m_mpcd_sys;              //!< MPCD system data
    std::shared_ptr<SystemDefinition> m_sysdef;                //!< HOOMD system definition
    std::shared_ptr<hoomd::ParticleData> m_pdata;              //!< HOOMD particle data
    std::shared_ptr<mpcd::ParticleData> m_mpcd_pdata;          //!< MPCD particle data
    std::shared_ptr<const ExecutionConfiguration> m_exec_conf; //!< Execution configuration

    std::shared_ptr<mpcd::CellList> m_cl;         //!< MPCD cell list
    std::shared_ptr<ParticleGroup> m_embed_group; //!< Embedded particles

    uint64_t m_period;        //!< Number of timesteps between collisions
    uint64_t m_next_timestep; //!< Timestep next collision should be performed

    unsigned int m_instance = 0; //!< Unique ID for RNG seeding

    //! Check if a collision should occur and advance the timestep counter
    virtual bool shouldCollide(uint64_t timestep);

    //! Call the collision rule
    virtual void rule(uint64_t timestep) { }

    bool m_enable_grid_shift; //!< Flag to enable grid shifting
    };

namespace detail
    {
//! Export the MPCDCollisionMethod class to python
void export_CollisionMethod(pybind11::module& m);
    }  // end namespace detail
    }  // end namespace mpcd
    }  // end namespace hoomd
#endif // MPCD_COLLISION_METHOD_H_
