// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: joaander

#include "NeighborList.h"
#include "NeighborListGPU.cuh"
#include "hoomd/Autotuner.h"
#include "hoomd/GlobalArray.h"

/*! \file NeighborListGPU.h
    \brief Declares the NeighborListGPU class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __NEIGHBORLISTGPU_H__
#define __NEIGHBORLISTGPU_H__

namespace hoomd
    {
namespace md
    {
//! Neighbor list build on the GPU
/*! Implements common functions (like distance check)
    on the GPU for use by other GPU nlist classes derived from NeighborListGPU.

    GPU kernel methods are defined in NeighborListGPU.cuh and defined in NeighborListGPU.cu.

    \ingroup computes
*/
class PYBIND11_EXPORT NeighborListGPU : public NeighborList
    {
    public:
    //! Constructs the compute
    NeighborListGPU(std::shared_ptr<SystemDefinition> sysdef, Scalar r_buff)
        : NeighborList(sysdef, r_buff)
        {
        m_exec_conf->msg->notice(5) << "Constructing NeighborlistGPU" << std::endl;

        GlobalArray<unsigned int> flags(1, m_exec_conf);
        std::swap(m_flags, flags);
        TAG_ALLOCATION(m_flags);

#if defined(ENABLE_HIP) && defined(__HIP_PLATFORM_NVCC__)
        if (m_exec_conf->allConcurrentManagedAccess())
            {
            cudaMemAdvise(m_flags.get(),
                          m_flags.getNumElements() * sizeof(unsigned int),
                          cudaMemAdviseSetPreferredLocation,
                          cudaCpuDeviceId);
            CHECK_CUDA_ERROR();
            }
#endif

            {
            ArrayHandle<unsigned int> h_flags(m_flags,
                                              access_location::host,
                                              access_mode::overwrite);
            *h_flags.data = 0;
            }

        // default to full mode
        m_storage_mode = full;
        m_checkn = 1;

        // flag to say how big to resize
        GlobalArray<unsigned int> req_size_nlist(1, m_exec_conf);
        std::swap(m_req_size_nlist, req_size_nlist);
        TAG_ALLOCATION(m_req_size_nlist);

#if defined(ENABLE_HIP) && defined(__HIP_PLATFORM_NVCC__)
        if (m_exec_conf->allConcurrentManagedAccess())
            {
            cudaMemAdvise(m_req_size_nlist.get(),
                          m_req_size_nlist.getNumElements() * sizeof(unsigned int),
                          cudaMemAdviseSetPreferredLocation,
                          cudaCpuDeviceId);
            CHECK_CUDA_ERROR();
            }
#endif

        // create cuda event
        unsigned int warp_size = m_exec_conf->dev_prop.warpSize;
        m_tuner_filter.reset(new Autotuner(warp_size,
                                           1024,
                                           warp_size,
                                           5,
                                           100000,
                                           "nlist_filter",
                                           this->m_exec_conf));
        m_tuner_head_list.reset(new Autotuner(warp_size,
                                              1024,
                                              warp_size,
                                              5,
                                              100000,
                                              "nlist_head_list",
                                              this->m_exec_conf));
        }

    //! Destructor
    virtual ~NeighborListGPU() { }

    //! Set autotuner parameters
    /*! \param enable Enable/disable autotuning
        \param period period (approximate) in time steps when returning occurs
    */
    virtual void setAutotunerParams(bool enable, unsigned int period)
        {
        NeighborList::setAutotunerParams(enable, period);
        m_tuner_filter->setPeriod(period / 10);
        m_tuner_filter->setEnabled(enable);

        m_tuner_head_list->setPeriod(period / 10);
        m_tuner_head_list->setEnabled(enable);
        }

    //! Benchmark the filter kernel
    double benchmarkFilter(unsigned int num_iters);

    //! Update the exclusion list on the GPU
    virtual void updateExListIdx();

    protected:
    GlobalArray<unsigned int> m_flags; //!< Storage for device flags on the GPU

    GlobalArray<unsigned int>
        m_req_size_nlist; //!< Flag to hold the required size of the neighborlist

    //! Builds the neighbor list
    virtual void buildNlist(uint64_t timestep);

    //! Perform the nlist distance check on the GPU
    virtual bool distanceCheck(uint64_t timestep);

    //! GPU nlists set their last updated pos in the compute kernel, this call only resets the last
    //! box length
    virtual void setLastUpdatedPos()
        {
        m_last_L = m_pdata->getGlobalBox().getNearestPlaneDistance();
        m_last_L_local = m_pdata->getBox().getNearestPlaneDistance();
        }

    //! Filter the neighbor list of excluded particles
    virtual void filterNlist();

    //! Build the head list for neighbor list indexing on the GPU
    virtual void buildHeadList();

    //! Schedule the distance check kernel
    /*! \param timestep Current time step
     */
    unsigned int
        m_checkn; //!< Internal counter to assign when checking if the nlist needs an update

    private:
    std::unique_ptr<Autotuner> m_tuner_filter;    //!< Autotuner for filter block size
    std::unique_ptr<Autotuner> m_tuner_head_list; //!< Autotuner for the head list block size

    GlobalArray<unsigned int>
        m_alt_head_list; //!< Alternate array to hold the head list from prefix sum
    };

namespace detail
    {
//! Exports NeighborListGPU to python
void export_NeighborListGPU(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
