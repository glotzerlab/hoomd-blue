// Copyright (c) 2009-2024 The Regents of the University of Michigan.
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

#include "CellList.h"

#include "hoomd/Autotuned.h"
#include "hoomd/ParticleGroup.h"
#include "hoomd/SystemDefinition.h"
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
class PYBIND11_EXPORT CollisionMethod : public Autotuned
    {
    public:
    //! Constructor
    CollisionMethod(std::shared_ptr<SystemDefinition> sysdef,
                    uint64_t cur_timestep,
                    uint64_t period,
                    int phase);
    //! Destructor
    virtual ~CollisionMethod() { }

    //! Implementation of the collision rule
    void collide(uint64_t timestep);

    //! Peek if a collision will occur on this timestep
    virtual bool peekCollide(uint64_t timestep) const;

    //! Get the particle group that is coupled to the MPCD solvent through the collision step.
    std::shared_ptr<ParticleGroup> getEmbeddedGroup()
        {
        return m_embed_group;
        }

    //! Sets a group of particles that is coupled to the MPCD solvent through the collision step
    /*!
     * \param embed_group Group to embed
     */
    void setEmbeddedGroup(std::shared_ptr<ParticleGroup> embed_group)
        {
        m_embed_group = embed_group;
        if (m_cl)
            {
            m_cl->setEmbeddedGroup(embed_group);
            }
        }

    //! Get the period of the collision method
    uint64_t getPeriod() const
        {
        return m_period;
        }

    //! Set the period of the collision method
    void setPeriod(uint64_t cur_timestep, uint64_t period);

    //! Get the cell list used for collisions
    std::shared_ptr<mpcd::CellList> getCellList() const
        {
        return m_cl;
        }

    //! Set the cell list used for collisions
    virtual void setCellList(std::shared_ptr<mpcd::CellList> cl)
        {
        m_cl = cl;
        if (m_cl)
            {
            m_cl->setEmbeddedGroup(m_embed_group);
            }
        }

    protected:
    std::shared_ptr<SystemDefinition> m_sysdef;                //!< HOOMD system definition
    std::shared_ptr<hoomd::ParticleData> m_pdata;              //!< HOOMD particle data
    std::shared_ptr<mpcd::ParticleData> m_mpcd_pdata;          //!< MPCD particle data
    std::shared_ptr<const ExecutionConfiguration> m_exec_conf; //!< Execution configuration

    std::shared_ptr<mpcd::CellList> m_cl;         //!< MPCD cell list
    std::shared_ptr<ParticleGroup> m_embed_group; //!< Embedded particles

    uint64_t m_period;        //!< Number of timesteps between collisions
    uint64_t m_next_timestep; //!< Timestep next collision should be performed

    //! Check if a collision should occur and advance the timestep counter
    virtual bool shouldCollide(uint64_t timestep);

    //! Call the collision rule
    virtual void rule(uint64_t timestep) { }
    };
    } // end namespace mpcd
    } // end namespace hoomd
#endif // MPCD_COLLISION_METHOD_H_
