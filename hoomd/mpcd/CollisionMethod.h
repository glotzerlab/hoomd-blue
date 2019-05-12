// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/CollisionMethod.h
 * \brief Declaration of mpcd::CollisionMethod
 */

#ifndef MPCD_COLLISION_METHOD_H_
#define MPCD_COLLISION_METHOD_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "SystemData.h"
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"

namespace mpcd
{

//! MPCD collision method
/*!
 * This class forms the generic base for an MPCD collision method. It handles the boiler plate of setting up the method
 * and implementing the collision. Each deriving class should implement a rule() that gives the physics of the collision.
 */
class PYBIND11_EXPORT CollisionMethod
    {
    public:
        //! Constructor
        CollisionMethod(std::shared_ptr<mpcd::SystemData> sysdata,
                        unsigned int cur_timestep,
                        unsigned int period,
                        int phase,
                        unsigned int seed);
        //! Destructor
        virtual ~CollisionMethod() { }

        //! Implementation of the collision rule
        void collide(unsigned int timestep);

        //! Peek if a collision will occur on this timestep
        virtual bool peekCollide(unsigned int timestep) const;

        //! Sets the profiler for the integration method to use
        void setProfiler(std::shared_ptr<Profiler> prof)
            {
            m_prof = prof;
            }

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
        void drawGridShift(unsigned int timestep);

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

    protected:
        std::shared_ptr<mpcd::SystemData> m_mpcd_sys;                   //!< MPCD system data
        std::shared_ptr<SystemDefinition> m_sysdef;                     //!< HOOMD system definition
        std::shared_ptr<::ParticleData> m_pdata;                        //!< HOOMD particle data
        std::shared_ptr<mpcd::ParticleData> m_mpcd_pdata;               //!< MPCD particle data
        std::shared_ptr<const ExecutionConfiguration> m_exec_conf;      //!< Execution configuration
        std::shared_ptr<Profiler> m_prof;                               //!< System profiler

        std::shared_ptr<mpcd::CellList> m_cl;          //!< MPCD cell list
        std::shared_ptr<ParticleGroup> m_embed_group;  //!< Embedded particles

        unsigned int m_period;                  //!< Number of timesteps between collisions
        unsigned int m_next_timestep;           //!< Timestep next collision should be performed
        unsigned int m_seed;        //!< Random number seed

        //! Check if a collision should occur and advance the timestep counter
        virtual bool shouldCollide(unsigned int timestep);

        //! Call the collision rule
        virtual void rule(unsigned int timestep) {}

        bool m_enable_grid_shift;   //!< Flag to enable grid shifting
    };

namespace detail
{
//! Export the MPCDCollisionMethod class to python
void export_CollisionMethod(pybind11::module& m);
} // end namespace detail
} // end namespace mpcd
#endif // MPCD_COLLISION_METHOD_H_
