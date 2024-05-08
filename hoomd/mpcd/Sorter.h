// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/Sorter.h
 * \brief Declares mpcd::Sorter, which sorts particles in the cell list
 */

#ifndef MPCD_SORTER_H_
#define MPCD_SORTER_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "CellList.h"

#include "hoomd/Autotuned.h"
#include "hoomd/SystemDefinition.h"
#include <pybind11/pybind11.h>

namespace hoomd
    {
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
class PYBIND11_EXPORT Sorter : public Autotuned
    {
    public:
    //! Constructor
    Sorter(std::shared_ptr<SystemDefinition> sysdef,
           unsigned int cur_timestep,
           unsigned int period);

    //! Destructor
    virtual ~Sorter();

    //! Update the particle data order
    virtual void update(uint64_t timestep);

    bool peekSort(uint64_t timestep) const;

    //! Change the period
    void setPeriod(unsigned int cur_timestep, unsigned int period)
        {
        m_period = period;
        const unsigned int multiple = cur_timestep / m_period + (cur_timestep % m_period != 0);
        m_next_timestep = multiple * m_period;
        }

    //! Set the cell list used for sorting
    virtual void setCellList(std::shared_ptr<mpcd::CellList> cl)
        {
        if (cl != m_cl)
            {
            m_cl = cl;
            }
        }

    protected:
    std::shared_ptr<SystemDefinition> m_sysdef;                //!< HOOMD system definition
    std::shared_ptr<hoomd::ParticleData> m_pdata;              //!< HOOMD particle data
    std::shared_ptr<const ExecutionConfiguration> m_exec_conf; //!< Execution configuration

    std::shared_ptr<mpcd::ParticleData> m_mpcd_pdata; //!< MPCD particle data
    std::shared_ptr<mpcd::CellList> m_cl;             //!< MPCD cell list

    GPUVector<unsigned int> m_order;  //!< Maps new sorted index onto old particle indexes
    GPUVector<unsigned int> m_rorder; //!< Maps old particle indexes onto new sorted indexes

    unsigned int m_period;    //!< Sorting period
    uint64_t m_next_timestep; //!< Next step to apply sorting

    //! Compute the sorting order at the current timestep
    virtual void computeOrder(uint64_t timestep);

    //! Apply the sorting order
    virtual void applyOrder() const;

    private:
    bool shouldSort(uint64_t timestep);
    };

namespace detail
    {
//! Exports the mpcd::Sorter to python
void export_Sorter(pybind11::module& m);
    } // end namespace detail
    } // end namespace mpcd
    } // end namespace hoomd
#endif // MPCD_SORTER_H_
