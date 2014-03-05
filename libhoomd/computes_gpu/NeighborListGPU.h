/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: joaander

#include "NeighborList.h"
#include "GPUFlags.h"
#include "Autotuner.h"

/*! \file NeighborListGPU.h
    \brief Declares the NeighborListGPU class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __NEIGHBORLISTGPU_H__
#define __NEIGHBORLISTGPU_H__

//! Neighbor list build on the GPU
/*! Implements the O(N^2) neighbor list build on the GPU. Also implements common functions (like distance check)
    on the GPU for use by other GPU nlist classes derived from NeighborListGPU.

    GPU kernel methods are defined in NeighborListGPU.cuh and defined in NeighborListGPU.cu.

    \ingroup computes
*/
class NeighborListGPU : public NeighborList
    {
    public:
        //! Constructs the compute
        NeighborListGPU(boost::shared_ptr<SystemDefinition> sysdef, Scalar r_cut, Scalar r_buff)
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

            // create cuda event
            cudaEventCreate(&m_event,cudaEventDisableTiming);

            m_tuner_filter.reset(new Autotuner(32, 1024, 32, 5, 100000, "nlist_filter", this->m_exec_conf));
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
            }

        //! Benchmark the filter kernel
        double benchmarkFilter(unsigned int num_iters);

        //! Update the exclusion list on the GPU
        virtual void updateExListIdx();

        #ifdef ENABLE_MPI
        //! Set the communicator to use
        /*! \param comm MPI communication class
         */
        virtual void setCommunicator(boost::shared_ptr<Communicator> comm)
            {
            // upon first call, register with Communicator
            if (comm && !m_comm)
                comm->addLocalComputeCallback(bind(&NeighborListGPU::scheduleDistanceCheck, this, _1));

            // call base class method
            NeighborList::setCommunicator(comm);
            }
        #endif

        //! Schedule the distance check kernel
        /*! \param timestep Current time step
         */
        void scheduleDistanceCheck(unsigned int timestep);

    protected:
        GPUArray<unsigned int> m_flags;   //!< Storage for device flags on the GPU

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

    private:
        boost::scoped_ptr<Autotuner> m_tuner_filter; //!< Autotuner for filter block size

        unsigned int m_checkn;              //!< Internal counter to assign when checking if the nlist needs an update
        bool m_distcheck_scheduled;         //!< True if a distance check kernel has been queued
        unsigned int m_last_schedule_tstep; //!< Time step of last kernel schedule

        cudaEvent_t m_event;                //!< Event signalling completion of distcheck kernel
        #ifdef ENABLE_MPI
        boost::signals2::connection m_callback_connection; //!< Connection to Communicator
        #endif
    };

//! Exports NeighborListGPU to python
void export_NeighborListGPU();

#endif
