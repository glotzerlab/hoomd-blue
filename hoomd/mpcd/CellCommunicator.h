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
        std::vector<MPI_Request> m_reqs;        //!< MPI request objects

        std::vector<unsigned int> m_neighbors;  //!< Unique neighbor ranks
        std::vector<unsigned int> m_begin;      //!< Begin offset of every neighbor
        std::vector<unsigned int> m_num_send;   //!< Number of cells to send to every neighbor

        unsigned int m_num_unique_cells;        //!< Number of unique cells to receive
        GPUArray<unsigned int> m_unique_cells;  //!< Unique cells to receive
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
    // check if communication is occurring and place a lock on
    if (m_communicating) return;
    m_communicating = true;

    // initialize grid if required
    if (m_needs_init)
        {
        initialize();
        m_needs_init = false;
        }

    // resize send / receive buffers for this comm element
    m_send_buf.resize(m_send_idx.getNumElements() * sizeof(typename PackOpT::element));
    m_recv_buf.resize(m_send_idx.getNumElements() * sizeof(typename PackOpT::element));

    // pack the buffers
        {
        ArrayHandle<T> h_props(props, access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_send_idx(m_send_idx, access_location::host, access_mode::read);
        ArrayHandle<unsigned char> h_send_buf(m_send_buf, access_location::host, access_mode::overwrite);
        typename PackOpT::element* send_buf = reinterpret_cast<typename PackOpT::element*>(h_send_buf.data);

        for (unsigned int idx=0; idx < m_send_idx.getNumElements(); ++idx)
            {
            send_buf[idx] = op.pack(h_props.data[h_send_idx.data[idx]]);
            }
        }

    // make the MPI calls
        {
        ArrayHandle<unsigned char> h_send_buf(m_send_buf, access_location::host, access_mode::read);
        typename PackOpT::element* send_buf = reinterpret_cast<typename PackOpT::element*>(h_send_buf.data);

        ArrayHandle<unsigned char> h_recv_buf(m_recv_buf, access_location::host, access_mode::overwrite);
        typename PackOpT::element* recv_buf = reinterpret_cast<typename PackOpT::element*>(h_recv_buf.data);

        m_reqs.resize(2*m_neighbors.size());
        for (unsigned int idx=0; idx < m_neighbors.size(); ++idx)
            {
            const unsigned int neigh = m_neighbors[idx];
            const unsigned int offset = m_begin[idx];
            const size_t num_bytes = sizeof(typename PackOpT::element) * m_num_send[idx];
            MPI_Isend(send_buf + offset, num_bytes, MPI_BYTE, neigh, 0, m_mpi_comm, &m_reqs[2*idx]);
            MPI_Irecv(recv_buf + offset, num_bytes, MPI_BYTE, neigh, 0, m_mpi_comm, &m_reqs[2*idx+1]);
            }
        }
    }

//! Finalize communication of the grid
template<typename T, class PackOpT>
void mpcd::CellCommunicator::finalize(const GPUArray<T>& props, PackOpT op)
    {
    if (!m_communicating) return;

    // finish all MPI requests
    MPI_Waitall(m_reqs.size(), m_reqs.data(), MPI_STATUSES_IGNORE);

    // fold the receive buffer back into the grid
    ArrayHandle<unsigned int> h_recv_cells(m_recv_cells, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_unique_cells(m_unique_cells, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_recv_cells_begin(m_recv_cells_begin, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_recv_cells_end(m_recv_cells_end, access_location::host, access_mode::read);

    ArrayHandle<unsigned char> h_recv_buf(m_recv_buf, access_location::host, access_mode::read);
    typename PackOpT::element* recv_buf = reinterpret_cast<typename PackOpT::element*>(h_recv_buf.data);

    ArrayHandle<T> h_props(props, access_location::host, access_mode::readwrite);

    for (unsigned int idx=0; idx < m_num_unique_cells; ++idx)
        {
        const unsigned int cell_idx = h_unique_cells.data[idx];
        const unsigned int begin = h_recv_cells_begin.data[idx];
        const unsigned int end = h_recv_cells_end.data[idx];

        // loop through all received data for this cell, and unpack it iteratively
        T val = h_props.data[cell_idx];
        for (unsigned int i = begin; i < end; ++i)
            {
            val = op.unpack(recv_buf[h_recv_cells.data[i]], val);
            }

        // save the accumulated unpacked value
        h_props.data[cell_idx] = val;
        }

    m_communicating = false;
    }

#endif // MPCD_CELL_COMMUNICATOR_H_

#endif // ENABLE_MPI