// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/CellCommunicator.h
 * \brief Declares and defines the mpcd::CellCommunicator class
 */

#ifdef ENABLE_MPI

#ifndef MPCD_CELL_COMMUNICATOR_H_
#define MPCD_CELL_COMMUNICATOR_H_

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "CommunicatorUtilities.h"
#include "CellList.h"

#include "hoomd/DomainDecomposition.h"
#include "hoomd/GPUArray.h"
#include "hoomd/GPUVector.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/SystemDefinition.h"

#include <map>
#include <set>

namespace mpcd
{

//! Communicates properties across the MPCD cell list
class CellCommunicator
    {
    public:
        //! Constructor
        CellCommunicator(std::shared_ptr<SystemDefinition> sysdef,
                         std::shared_ptr<mpcd::CellList> cl);

        //! Destructor
        virtual ~CellCommunicator();

        //! Reduce cell list properties
        /*!
         * \param props Properties to reduce
         * \param op Binary reduction operator
         *
         * Properties are reduced across domain boundaries, as marked in the cell list.
         */
        template<typename T, class PackOpT>
        void communicate(const GPUArray<T>& props, PackOpT op)
            {
            begin(props, op);
            finalize(props, op);
            }

        //! Begin communication of the grid
        template<typename T, class PackOpT>
        void begin(const GPUArray<T>& props, PackOpT op);

        //! Finalize communication of the grid
        template<typename T, class PackOpT>
        void finalize(const GPUArray<T>& props, PackOpT op);

        //! Set autotuner parameters
        /*!
         * \param enable Enable / disable autotuning
         * \param period period (approximate) in time steps when retuning occurs
         */
        void setAutotunerParams(bool enable, unsigned int period)
            {
            }

        //! Set the profiler used by this compute
        /*!
         * \param prof Profiler to use (if null, do not profile)
         */
        void setProfiler(std::shared_ptr<Profiler> prof)
            {
            m_prof = prof;
            }

    private:
        std::shared_ptr<SystemDefinition> m_sysdef;                 //!< System definition
        std::shared_ptr<::ParticleData> m_pdata;                    //!< HOOMD particle data
        std::shared_ptr<const ExecutionConfiguration> m_exec_conf;  //!< Execution configuration
        const MPI_Comm m_mpi_comm;                                  //!< MPI Communicator
        std::shared_ptr<DomainDecomposition> m_decomposition;       //!< Domain decomposition
        std::shared_ptr<Profiler> m_prof;                           //!< Profiler

        std::shared_ptr<mpcd::CellList> m_cl;   //!< MPCD cell list

        bool m_communicating;   //!< Flag if communication is occurring
        GPUVector<unsigned char> m_send_buf;    //!< Send buffer
        GPUVector<unsigned char> m_recv_buf;    //!< Receive buffer
        GPUArray<unsigned int> m_send_idx;      //!< Indexes of cells in send buffer
        GPUArray<unsigned int> m_recv_idx;      //!< Indexes of cells in receive buffer
        std::vector<MPI_Request> m_reqs;        //!< MPI request objects

        std::vector<unsigned int> m_neighbors;  //!< Unique neighbor ranks
        std::vector<unsigned int> m_begin;      //!< Begin offset of every neighbor
        std::vector<unsigned int> m_num_send;   //!< Number of cells to send to every neighbor

        unsigned int m_num_unique_cells;        //!< Number of unique cells to receive
        GPUArray<unsigned int> m_recv_cells;    //!< Reordered mapping of buffer from ranks to group received cells together
        GPUArray<unsigned int> m_recv_cells_begin;  //!< Begin offset of every unique cell
        GPUArray<unsigned int> m_recv_cells_end;    //!< End offset of every unique cell

        bool m_needs_init;      //!< Flag if grid needs to be initialized
        //! Slot that communicator needs to be reinitialized
        void slotInit()
            {
            m_needs_init = true;
            }

        //! Initialize the grid
        void initialize();
    };

} // end namespace mpcd

template<typename T, class PackOpT>
void mpcd::CellCommunicator::begin(const GPUArray<T>& props, PackOpT op)
    {
    if (m_needs_init)
        {
        initialize();
        m_needs_init = false;
        }
    }

//! Finalize communication of the grid
template<typename T, class PackOpT>
void mpcd::CellCommunicator::finalize(const GPUArray<T>& props, PackOpT op)
    {
    }

#endif // MPCD_CELL_COMMUNICATOR_H_

#endif // ENABLE_MPI