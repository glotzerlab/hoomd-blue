// Copyright (c) 2009-2019 The Regents of the University of Michigan
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

#ifdef ENABLE_CUDA
#include "CellCommunicator.cuh"
#endif // ENABLE_CUDA

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
class PYBIND11_EXPORT CellCommunicator
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
        void communicate(const GPUArray<T>& props, const PackOpT op)
            {
            begin(props, op);
            finalize(props, op);
            }

        //! Begin communication of the grid
        template<typename T, class PackOpT>
        void begin(const GPUArray<T>& props, const PackOpT op);

        //! Finalize communication of the grid
        template<typename T, class PackOpT>
        void finalize(const GPUArray<T>& props, const PackOpT op);

        //! Get the number of unique cells with communication
        unsigned int getNCells()
            {
            if (m_needs_init)
                {
                initialize();
                m_needs_init = false;
                }
            return m_num_cells;
            }

        //! Get the list of unique cells with communication
        const GPUArray<unsigned int>& getCells()
            {
            if (m_needs_init)
                {
                initialize();
                m_needs_init = false;
                }
            return m_cells;
            }

        //! Set autotuner parameters
        /*!
         * \param enable Enable / disable autotuning
         * \param period period (approximate) in time steps when retuning occurs
         */
        void setAutotunerParams(bool enable, unsigned int period)
            {
            #ifdef ENABLE_CUDA
            if (m_tuner_pack)
                {
                m_tuner_pack->setEnabled(enable);
                m_tuner_pack->setPeriod(period);
                }
            if (m_tuner_unpack)
                {
                m_tuner_unpack->setEnabled(enable);
                m_tuner_unpack->setPeriod(period);
                }
            #endif // ENABLE_CUDA
            }

    private:
        static unsigned int num_instances;      //!< Number of communicator instances
        const unsigned int m_id;                //!< Id for this communicator to use in tags

        std::shared_ptr<SystemDefinition> m_sysdef;                 //!< System definition
        std::shared_ptr<::ParticleData> m_pdata;                    //!< HOOMD particle data
        std::shared_ptr<const ExecutionConfiguration> m_exec_conf;  //!< Execution configuration
        const MPI_Comm m_mpi_comm;                                  //!< MPI Communicator
        std::shared_ptr<DomainDecomposition> m_decomposition;       //!< Domain decomposition

        std::shared_ptr<mpcd::CellList> m_cl;   //!< MPCD cell list

        bool m_communicating;   //!< Flag if communication is occurring
        GPUVector<unsigned char> m_send_buf;    //!< Send buffer
        GPUVector<unsigned char> m_recv_buf;    //!< Receive buffer
        GPUArray<unsigned int> m_send_idx;      //!< Indexes of cells in send buffer
        std::vector<MPI_Request> m_reqs;        //!< MPI request objects

        std::vector<unsigned int> m_neighbors;  //!< Unique neighbor ranks
        std::vector<unsigned int> m_begin;      //!< Begin offset of every neighbor
        std::vector<unsigned int> m_num_send;   //!< Number of cells to send to every neighbor

        unsigned int m_num_cells;               //!< Number of unique cells to receive
        GPUArray<unsigned int> m_cells;         //!< Unique cells to receive
        GPUArray<unsigned int> m_recv;          //!< Reordered mapping of buffer from ranks to group received cells together
        GPUArray<unsigned int> m_recv_begin;    //!< Begin offset of every unique cell
        GPUArray<unsigned int> m_recv_end;      //!< End offset of every unique cell

        bool m_needs_init;      //!< Flag if grid needs to be initialized
        //! Slot that communicator needs to be reinitialized
        void slotInit()
            {
            m_needs_init = true;
            }

        //! Initialize the grid
        void initialize();

        //! Packs the property buffer
        template<typename T, class PackOpT>
        void packBuffer(const GPUArray<T>& props, const PackOpT op);

        //! Unpacks the property buffer
        template<typename T, class PackOpT>
        void unpackBuffer(const GPUArray<T>& props, const PackOpT op);

        #ifdef ENABLE_CUDA
        std::unique_ptr<Autotuner> m_tuner_pack;    //!< Tuner for pack kernel
        std::unique_ptr<Autotuner> m_tuner_unpack;  //!< Tuner for unpack kernel

        //! Packs the property buffer on the GPU
        template<typename T, class PackOpT>
        void packBufferGPU(const GPUArray<T>& props, const PackOpT op);

        //! Unpacks the property buffer on the GPU
        template<typename T, class PackOpT>
        void unpackBufferGPU(const GPUArray<T>& props, const PackOpT op);
        #endif // ENABLE_CUDA
    };

} // end namespace mpcd

/*!
 * \param props Property buffer to pack
 * \param op Packing operator to apply to the send buffer
 *
 * \tparam T Type of buffer to pack (inferred)
 * \tparam PackOpT Packing operator type (inferred)
 *
 * The data in \a props is packed into the send buffers, and nonblocking MPI
 * send / receive operations are initiated. If communication is already occurring,
 * the method returns immediately and no action is taken.
 */
template<typename T, class PackOpT>
void mpcd::CellCommunicator::begin(const GPUArray<T>& props, const PackOpT op)
    {
    // check if communication is occurring and place a lock on
    if (m_communicating) return;
    m_communicating = true;

    // ensure that the property grid is sufficiently sized compared to the cell list
    if (props.getNumElements() < m_cl->getNCells())
        {
        m_exec_conf->msg->error() << "mpcd: cell property to be reduced is smaller than cell list dimensions" << std::endl;
        throw std::runtime_error("MPCD cell property has insufficient dimensions");
        }

    // initialize grid if required
    if (m_needs_init)
        {
        initialize();
        m_needs_init = false;
        }

    // resize send / receive buffers for this comm element
    m_send_buf.resize(m_send_idx.getNumElements() * sizeof(typename PackOpT::element));
    m_recv_buf.resize(m_send_idx.getNumElements() * sizeof(typename PackOpT::element));

    #ifdef ENABLE_CUDA
    if (m_exec_conf->isCUDAEnabled())
        {
        packBufferGPU(props, op);
        }
    else
    #endif // ENABLE_CUDA
        {
        packBuffer(props, op);
        }

    // make the MPI calls
        {
        // determine whether to use CPU or GPU CUDA buffers
        access_location::Enum mpi_loc;
        #ifdef ENABLE_MPI_CUDA
        if (m_exec_conf->isCUDAEnabled())
            {
            mpi_loc = access_location::device;
            }
        else
        #endif // ENABLE_MPI_CUDA
            {
            mpi_loc = access_location::host;
            }

        ArrayHandle<unsigned char> h_send_buf(m_send_buf, mpi_loc, access_mode::read);
        ArrayHandle<unsigned char> h_recv_buf(m_recv_buf, mpi_loc, access_mode::overwrite);
        typename PackOpT::element* send_buf = reinterpret_cast<typename PackOpT::element*>(h_send_buf.data);
        typename PackOpT::element* recv_buf = reinterpret_cast<typename PackOpT::element*>(h_recv_buf.data);
        #ifdef ENABLE_MPI_CUDA
        if (mpi_loc == access_location::device) cudaDeviceSynchronize();
        #endif // ENABLE_MPI_CUDA

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
/*!
 * \param props Property buffer to unpack
 * \param op Packing operator to apply to the receive buffer
 *
 * \tparam T Type of buffer to pack (inferred)
 * \tparam PackOpT Packing operator type (inferred)
 *
 * Existing MPI requests are finalized, and then the received buffers are
 * unpacked into \a props. If no communication is ongoing, the method immediately
 * returns and takes no action.
 */
template<typename T, class PackOpT>
void mpcd::CellCommunicator::finalize(const GPUArray<T>& props, const PackOpT op)
    {
    if (!m_communicating) return;

    // finish all MPI requests
    MPI_Waitall(m_reqs.size(), m_reqs.data(), MPI_STATUSES_IGNORE);
    #ifdef ENABLE_MPI_CUDA
    // MPI calls can execute in multiple streams, so force a synchronization before we move on
    if (m_exec_conf->isCUDAEnabled()) cudaDeviceSynchronize();
    #endif // ENABLE_MPI_CUDA

    // unpack the buffer
    #ifdef ENABLE_CUDA
    if (m_exec_conf->isCUDAEnabled())
        {
        unpackBufferGPU(props, op);
        }
    else
    #endif // ENABLE_CUDA
        {
        unpackBuffer(props, op);
        }

    m_communicating = false;
    }

/*!
 * \param props Property buffer to pack
 * \param op Packing operator to apply to the send buffer
 *
 * \tparam T Type of buffer to pack (inferred)
 * \tparam PackOpT Packing operator type (inferred)
 *
 * The packing operator applies a unary pack functor to \a props, which permits
 * a reduction or transformation of the data to be sent. See mpcd::detail::CellEnergyPackOp
 * for an example.
 *
 * \post Communicated cells in \a props are packed into \a m_send_buf.
 */
template<typename T, class PackOpT>
void mpcd::CellCommunicator::packBuffer(const GPUArray<T>& props, const PackOpT op)
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

/*!
 * \param props Property buffer to pack
 * \param op Packing operator to apply to the received buffer
 *
 * \tparam T Type of buffer to pack (inferred)
 * \tparam PackOpT Packing operator type (inferred)
 *
 * The packing operator must be associative so that the order it is applied
 * to the received cells does not matter, e.g., addition.
 *
 * \post The bytes from \a m_recv_buf are unpacked into \a props.
 */
template<typename T, class PackOpT>
void mpcd::CellCommunicator::unpackBuffer(const GPUArray<T>& props, const PackOpT op)
    {
    ArrayHandle<unsigned int> h_recv(m_recv, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_cells(m_cells, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_recv_begin(m_recv_begin, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_recv_end(m_recv_end, access_location::host, access_mode::read);

    ArrayHandle<unsigned char> h_recv_buf(m_recv_buf, access_location::host, access_mode::read);
    typename PackOpT::element* recv_buf = reinterpret_cast<typename PackOpT::element*>(h_recv_buf.data);

    ArrayHandle<T> h_props(props, access_location::host, access_mode::readwrite);

    for (unsigned int idx=0; idx < m_num_cells; ++idx)
        {
        const unsigned int cell_idx = h_cells.data[idx];
        const unsigned int begin = h_recv_begin.data[idx];
        const unsigned int end = h_recv_end.data[idx];

        // loop through all received data for this cell, and unpack it iteratively
        T val = h_props.data[cell_idx];
        for (unsigned int i = begin; i < end; ++i)
            {
            val = op.unpack(recv_buf[h_recv.data[i]], val);
            }

        // save the accumulated unpacked value
        h_props.data[cell_idx] = val;
        }
    }

#ifdef ENABLE_CUDA
/*!
 * \param props Property buffer to pack
 * \param op Packing operator to apply to the send buffer
 *
 * \tparam T Type of buffer to pack (inferred)
 * \tparam PackOpT Packing operator type (inferred)
 *
 * The packing operator applies a unary pack functor to \a props, which permits
 * a reduction or transformation of the data to be sent. See mpcd::detail::CellEnergyPackOp
 * for an example.
 *
 * \post Communicated cells in \a props are packed into \a m_send_buf.
 */
template<typename T, class PackOpT>
void mpcd::CellCommunicator::packBufferGPU(const GPUArray<T>& props, const PackOpT op)
    {
    ArrayHandle<T> d_props(props, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_send_idx(m_send_idx, access_location::device, access_mode::read);
    ArrayHandle<unsigned char> d_send_buf(m_send_buf, access_location::device, access_mode::overwrite);
    typename PackOpT::element* send_buf = reinterpret_cast<typename PackOpT::element*>(d_send_buf.data);

    m_tuner_pack->begin();
    mpcd::gpu::pack_cell_buffer(send_buf,
                                d_props.data,
                                d_send_idx.data,
                                op,
                                m_send_idx.getNumElements(),
                                m_tuner_pack->getParam());
    if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
    m_tuner_pack->end();
    }

/*!
 * \param props Property buffer to pack
 * \param op Packing operator to apply to the received buffer
 *
 * \tparam T Type of buffer to pack (inferred)
 * \tparam PackOpT Packing operator type (inferred)
 *
 * The packing operator must be associative so that the order it is applied
 * to the received cells does not matter, e.g., addition.
 *
 * \post The bytes from \a m_recv_buf are unpacked into \a props.
 */
template<typename T, class PackOpT>
void mpcd::CellCommunicator::unpackBufferGPU(const GPUArray<T>& props, const PackOpT op)
    {
    ArrayHandle<unsigned int> d_recv(m_recv, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_cells(m_cells, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_recv_begin(m_recv_begin, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_recv_end(m_recv_end, access_location::device, access_mode::read);

    ArrayHandle<unsigned char> d_recv_buf(m_recv_buf, access_location::device, access_mode::read);
    typename PackOpT::element* recv_buf = reinterpret_cast<typename PackOpT::element*>(d_recv_buf.data);

    ArrayHandle<T> d_props(props, access_location::device, access_mode::readwrite);

    m_tuner_unpack->begin();
    mpcd::gpu::unpack_cell_buffer(d_props.data,
                                  d_cells.data,
                                  d_recv.data,
                                  d_recv_begin.data,
                                  d_recv_end.data,
                                  recv_buf,
                                  op,
                                  m_num_cells,
                                  m_tuner_unpack->getParam());
    if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
    m_tuner_unpack->end();
    }
#endif // ENABLE_CUDA

#endif // MPCD_CELL_COMMUNICATOR_H_

#endif // ENABLE_MPI
