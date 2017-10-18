// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/VirtualParticleFiller.h
 * \brief Definition of class for backfilling solid boundaries with virtual particles.
 */

#ifndef MPCD_VIRTUAL_PARTICLE_FILLER_H_
#define MPCD_VIRTUAL_PARTICLE_FILLER_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "SystemData.h"
#include "hoomd/Variant.h"
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"

namespace mpcd
{

//! Adds virtual particles to the MPCD particle data
class VirtualParticleFiller
    {
    public:
        VirtualParticleFiller(std::shared_ptr<mpcd::SystemData> sysdata,
                              Scalar density,
                              unsigned int type,
                              std::shared_ptr<::Variant> T,
                              unsigned int seed);

        virtual ~VirtualParticleFiller() {}

        //! Fill up virtual particles
        void fill(unsigned int timestep);

        unsigned int getNFill() const
            {
            return m_N_fill;
            }

        unsigned int getNFillGlobal() const
            {
            return m_N_fill_global;
            }

        //! Sets the profiler for the integration method to use
        virtual void setProfiler(std::shared_ptr<Profiler> prof)
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
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {}

    protected:
        std::shared_ptr<::SystemDefinition> m_sysdef;                   //!< HOOMD system definition
        std::shared_ptr<::ParticleData> m_pdata;                        //!< HOOMD particle data
        std::shared_ptr<const ExecutionConfiguration> m_exec_conf;      //!< Execution configuration
        std::shared_ptr<mpcd::ParticleData> m_mpcd_pdata;               //!< MPCD particle data
        std::shared_ptr<mpcd::CellList> m_cl;                           //!< MPCD cell list
        std::shared_ptr<Profiler> m_prof;                               //!< System profiler;

        Scalar m_density;               //!< Fill density
        unsigned int m_type;            //!< Fill type
        std::shared_ptr<::Variant> m_T; //!< Temperature for filled particles
        unsigned int m_seed;            //!< Seed for PRNG

        unsigned int m_N_fill;      //!< Number of particles to fill locally
        unsigned int m_first_tag;   //!< First tag of locally held particles
        unsigned int m_N_fill_global;   //!< Global number of fill particles

        //! Compute the total number of particles to fill
        virtual void computeNumFill() {}

        //! Draw particles within the fill volume
        virtual void drawParticles(unsigned int timestep) {}
    };

namespace detail
{
//! Export the VirtualParticleFiller to python
void export_VirtualParticleFiller(pybind11::module& m);
} // end namespace detail
} // end namespace mpcd

#endif // MPCD_VIRTUAL_PARTICLE_FILLER_H_
