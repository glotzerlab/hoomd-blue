// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "NeighborList.h"
#include "NeighborListGPU.cuh"
#include "hoomd/GPUFlags.h"
#include "hoomd/Autotuner.h"

/*! \file NeighborListGPU.h
    \brief Declares the NeighborListGPU class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __NEIGHBORLISTGPU_H__
#define __NEIGHBORLISTGPU_H__

//! Neighbor list build on the GPU
/*! Implements common functions (like distance check)
    on the GPU for use by other GPU nlist classes derived from NeighborListGPU.

    GPU kernel methods are defined in NeighborListGPU.cuh and defined in NeighborListGPU.cu.

    \ingroup computes
*/
class NeighborListGPU : public NeighborList
    {
    public:
        //! Constructs the compute
        NeighborListGPU(std::shared_ptr<SystemDefinition> sysdef, Scalar r_cut, Scalar r_buff)
            : NeighborList(sysdef, r_cut, r_buff)
            {
            GPUArray<unsigned int> flags(1,exec_conf,true);
            m_flags.swap(flags);
            ArrayHandle<unsigned int> h_flags(m_flags,access_location::host, access_mode::overwrite);
            *h_flags.data = 0;

            // default to full mode
            m_storage_mode = full;
            m_checkn = 1;
            m_distcheck_scheduled = false;
            m_last_schedule_tstep = 0;

            // flag to say how big to resize
            GPUFlags<unsigned int> req_size_nlist(exec_conf);
            m_req_size_nlist.swap(req_size_nlist);

            // create cuda event
            cudaEventCreate(&m_event,cudaEventDisableTiming);

            m_tuner_filter.reset(new Autotuner(32, 1024, 32, 5, 100000, "nlist_filter", this->m_exec_conf));
            m_tuner_head_list.reset(new Autotuner(32, 1024, 32, 5, 100000, "nlist_head_list", this->m_exec_conf));
            }

        //! Destructor
        virtual ~NeighborListGPU()
            {
            #ifdef ENABLE_MPI
            if (m_callback_connection.connected())
                m_callback_connection.disconnect();
            #endif

            cudaEventDestroy(m_event);
            }

        //! Set autotuner parameters
        /*! \param enable Enable/disable autotuning
            \param period period (approximate) in time steps when returning occurs
        */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            NeighborList::setAutotunerParams(enable, period);
            m_tuner_filter->setPeriod(period/10);
            m_tuner_filter->setEnabled(enable);

            m_tuner_head_list->setPeriod(period/10);
            m_tuner_head_list->setEnabled(enable);
            }

        //! Benchmark the filter kernel
        double benchmarkFilter(unsigned int num_iters);

        //! Update the exclusion list on the GPU
        virtual void updateExListIdx();

        #ifdef ENABLE_MPI
        //! Set the communicator to use
        /*! \param comm MPI communication class
         */
        virtual void setCommunicator(std::shared_ptr<Communicator> comm)
            {
            // upon first call, register with Communicator
            if (comm && !m_comm)
                comm->addLocalComputeCallback(bind(&NeighborListGPU::scheduleDistanceCheck, this, _1));

            // call base class method
            NeighborList::setCommunicator(comm);
            }
        #endif

    protected:
        GPUArray<unsigned int> m_flags;   //!< Storage for device flags on the GPU

        GPUFlags<unsigned int> m_req_size_nlist;    //!< Flag to hold the required size of the neighborlist

        //! Builds the neighbor list
        virtual void buildNlist(unsigned int timestep);

        //! Perform the nlist distance check on the GPU
        virtual bool distanceCheck(unsigned int timestep);

        //! GPU nlists set their last updated pos in the compute kernel, this call only resets the last box length
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
        void scheduleDistanceCheck(unsigned int timestep);
        unsigned int m_checkn;              //!< Internal counter to assign when checking if the nlist needs an update
        bool m_distcheck_scheduled;         //!< True if a distance check kernel has been queued
        unsigned int m_last_schedule_tstep; //!< Time step of last kernel schedule

        cudaEvent_t m_event;                //!< Event signalling completion of distcheck kernel
        #ifdef ENABLE_MPI
        boost::signals2::connection m_callback_connection; //!< Connection to Communicator
        #endif

    private:
        boost::scoped_ptr<Autotuner> m_tuner_filter; //!< Autotuner for filter block size
        boost::scoped_ptr<Autotuner> m_tuner_head_list; //!< Autotuner for the head list block size

        GPUArray<unsigned int> m_alt_head_list; //!< Alternate array to hold the head list from prefix sum
    };

//! Exports NeighborListGPU to python
void export_NeighborListGPU();

#endif
