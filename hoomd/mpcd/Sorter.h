// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/Sorter.h
 * \brief Declares mpcd::Sorter, which sorts particles in the cell list
 */

#ifndef MPCD_SORTER_H_
#define MPCD_SORTER_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "SystemData.h"
#include "hoomd/extern/pybind/include/pybind11/pybind11.h"

namespace mpcd
{

//! Sorts MPCD particles
/*!
 * The Sorter puts MPCD particles into an order that is more cache-friendly.
 * The natural way to do this for the algorithm is cell list order. The base
 * Sorter implements applyOrder() for applying the reordering map to the
 * mpcd::ParticleData. Specific sorting algorithms can be implemented by
 * deriving from Sorter and implementing computeOrder(). Any computeOrder()
 * must set the map from old particle index to new particle index, and the
 * reverse mapping.
 *
 * When there are virtual particles in the mpcd::ParticleData, the Sorter will ignore
 * the virtual particles and leave them in place at the end of the arrays. This is
 * because they cannot be removed easily if they are sorted with the rest of the particles,
 * and the performance gains from doing a separate (segmented) sort on them is probably small.
 */
class PYBIND11_EXPORT Sorter
    {
    public:
        //! Constructor
        Sorter(std::shared_ptr<mpcd::SystemData> sysdata,
               unsigned int cur_timestep,
               unsigned int period);

        //! Destructor
        virtual ~Sorter();

        //! Update the particle data order
        virtual void update(unsigned int timestep);

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

        bool peekSort(unsigned int timestep) const;

        //! Change the period
        void setPeriod(unsigned int cur_timestep, unsigned int period)
            {
            m_period = period;
            const unsigned int multiple = cur_timestep / m_period + (cur_timestep % m_period != 0);
            m_next_timestep = multiple * m_period;
            }

    protected:
        std::shared_ptr<mpcd::SystemData> m_mpcd_sys;       //!< MPCD system data
        std::shared_ptr<SystemDefinition> m_sysdef;         //!< HOOMD system definition
        std::shared_ptr<::ParticleData> m_pdata;            //!< HOOMD particle data
        std::shared_ptr<const ExecutionConfiguration> m_exec_conf; //!< Execution configuration
        std::shared_ptr<Profiler> m_prof;   //!< System profiler

        std::shared_ptr<mpcd::ParticleData> m_mpcd_pdata;   //!< MPCD particle data
        std::shared_ptr<mpcd::CellList> m_cl;               //!< MPCD cell list

        GPUVector<unsigned int> m_order;    //!< Maps new sorted index onto old particle indexes
        GPUVector<unsigned int> m_rorder;   //!< Maps old particle indexes onto new sorted indexes

        unsigned int m_period;          //!< Sorting period
        unsigned int m_next_timestep;   //!< Next step to apply sorting

        //! Compute the sorting order at the current timestep
        virtual void computeOrder(unsigned int timestep);

        //! Apply the sorting order
        virtual void applyOrder() const;

    private:
        bool shouldSort(unsigned int timestep);
    };

namespace detail
{
//! Exports the mpcd::Sorter to python
void export_Sorter(pybind11::module& m);
} // end namespace detail
} // end namespace mpcd

#endif // MPCD_SORTER_H_
