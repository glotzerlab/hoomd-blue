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

#ifdef ENABLE_CUDA
#include "CellCommunicator.cuh"
#endif // ENABLE_CUDA

#include "CommunicatorUtilities.h"

#ifdef ENABLE_CUDA
#include "hoomd/Autotuner.h"
#endif // ENABLE_CUDA
#include "hoomd/DomainDecomposition.h"
#include "hoomd/GPUArray.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/Index1D.h"
#include "hoomd/SystemDefinition.h"

namespace mpcd
{

//! Reduces properties across the MPCD cell list
class CellCommunicator
    {
    public:
        //! Constructor
        /*!
         * \param sysdef System definition
         * \param cl MPCD cell list
         */
        CellCommunicator(std::shared_ptr<SystemDefinition> sysdef,
                         std::shared_ptr<mpcd::CellList> cl)
            : m_sysdef(sysdef),
              m_pdata(sysdef->getParticleData()),
              m_exec_conf(m_pdata->getExecConf()),
              m_mpi_comm(m_exec_conf->getMPICommunicator()),
              m_decomposition(m_pdata->getDomainDecomposition()),
              m_cl(cl),
              m_send_buf(m_exec_conf),
              m_recv_buf(m_exec_conf),
              m_needs_init(true)
            {
            m_exec_conf->msg->notice(5) << "Constructing MPCD CellCommunicator" << std::endl;
            #ifdef ENABLE_CUDA
            if (m_exec_conf->isCUDAEnabled())
                {
                m_tuner_pack.reset(new Autotuner(32, 1024, 32, 5, 100000, "mpcd_cell_comm_pack", m_exec_conf));
                m_tuner_unpack.reset(new Autotuner(32, 1024, 32, 5, 100000, "mpcd_cell_comm_unpack", m_exec_conf));
                }
            #endif // ENABLE_CUDA
            }

        //! Destructor
        ~CellCommunicator()
            {
            m_exec_conf->msg->notice(5) << "Destroying MPCD CellCommunicator" << std::endl;
            }

        //! Reduce cell list properties
        /*!
         * \param props Properties to reduce
         * \param reduction_op Binary reduction operator
         *
         * Properties are reduced across domain boundaries, as marked in the cell list.
         */
        template<typename T, class ReductionOpT>
        void reduce(const GPUArray<T>& props, ReductionOpT reduction_op);

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

        //! Cartesian coordinate axes for communicating
        enum struct axis : unsigned char
            {
            x=0,
            y,
            z
            };
        unsigned int m_neigh[6];                //!< Neighbor ranks
        Index3D m_idx[6];                       //!< Communication indexers for buffers
        uint3 m_right_offset[3];                //!< Offsets for reading out of right faces
        unsigned int m_max_cells;               //!< Maximum number of cells in any face
        GPUVector<unsigned char> m_send_buf;    //!< MPI buffer for sending
        GPUVector<unsigned char> m_recv_buf;    //!< MPI buffer for receiving

        //! Packs the property buffer
        template<typename T>
        void packBuffer(const GPUArray<T>& props, axis dim);

        //! Unpacks the property buffer
        template<typename T, class ReductionOpT>
        void unpackBuffer(const GPUArray<T>& props, ReductionOpT reduction_op, axis dim);

        #ifdef ENABLE_CUDA
        std::unique_ptr<Autotuner> m_tuner_pack;    //!< Tuner for pack kernel
        std::unique_ptr<Autotuner> m_tuner_unpack;  //!< Tuner for unpack kernel

        //! Packs the property buffer on the GPU
        template<typename T>
        void packBufferGPU(const GPUArray<T>& props, axis dim);

        //! Unpacks the property buffer on the GPU
        template<typename T, class ReductionOpT>
        void unpackBufferGPU(const GPUArray<T>& props, ReductionOpT reduction_op, axis dim);
        #endif // ENABLE_CUDA

        bool m_needs_init;                      //!< Flag if initialization is required
        //! Initialize the communication internals
        void initialize()
            {
            if (!m_needs_init) return;

            // setup the comm dimensions and determine maximum number of cells to communicate
            m_max_cells = 0;
            if (m_cl->isCommunicating(mpcd::detail::face::east))
                {
                setupCommDim(axis::x);
                unsigned int num_cells = m_idx[static_cast<unsigned int>(mpcd::detail::face::east)].getNumElements();
                num_cells += m_idx[static_cast<unsigned int>(mpcd::detail::face::west)].getNumElements();
                if (num_cells > m_max_cells) m_max_cells = num_cells;
                }
            if (m_cl->isCommunicating(mpcd::detail::face::north))
                {
                setupCommDim(axis::y);
                unsigned int num_cells = m_idx[static_cast<unsigned int>(mpcd::detail::face::north)].getNumElements();
                num_cells += m_idx[static_cast<unsigned int>(mpcd::detail::face::south)].getNumElements();
                if (num_cells > m_max_cells) m_max_cells = num_cells;
                }
            if (m_cl->isCommunicating(mpcd::detail::face::up))
                {
                setupCommDim(axis::z);
                unsigned int num_cells = m_idx[static_cast<unsigned int>(mpcd::detail::face::up)].getNumElements();
                num_cells += m_idx[static_cast<unsigned int>(mpcd::detail::face::down)].getNumElements();
                if (num_cells > m_max_cells) m_max_cells = num_cells;
                }

            // initialization succeeded, flip flag
            m_needs_init = false;
            }

        //! Size buffers large enough to hold all send elements
        void sizeBuffers(size_t elem_bytes)
            {
            m_send_buf.resize(elem_bytes * m_max_cells);
            m_recv_buf.resize(elem_bytes * m_max_cells);
            }

        //! Setup buffers for communication
        /*!
         * \param dim Dimension (x,y,z) along which communication is occurring
         */
        void setupCommDim(axis dim)
            {
            // determine the "right" and "left" faces from the dimension
            unsigned int right_face,left_face;
            if (dim == axis::x)
                {
                right_face = static_cast<unsigned int>(mpcd::detail::face::east);
                left_face = static_cast<unsigned int>(mpcd::detail::face::west);
                }
            else if (dim == axis::y)
                {
                right_face = static_cast<unsigned int>(mpcd::detail::face::north);
                left_face = static_cast<unsigned int>(mpcd::detail::face::south);
                }
            else // m_comm_dim == axis::z
                {
                right_face = static_cast<unsigned int>(mpcd::detail::face::up);
                left_face = static_cast<unsigned int>(mpcd::detail::face::down);
                }

            // get the communication neighbors
            m_neigh[right_face] = m_decomposition->getNeighborRank(right_face);
            m_neigh[left_face] = m_decomposition->getNeighborRank(left_face);

            // get the number of cells being sent / received
            auto num_comm_cells = m_cl->getNComm();
            const unsigned int num_right = num_comm_cells[right_face];
            const unsigned int num_left = num_comm_cells[left_face];

            // setup the buffer indexers and the offset into the cell list
            const Index3D& ci = m_cl->getCellIndexer();
            Index3D left_idx, right_idx;
            uint3 right_offset = make_uint3(0,0,0);
            if (dim == axis::x)
                {
                left_idx = Index3D(num_left, ci.getH(), ci.getD());
                right_idx = Index3D(num_right, ci.getH(), ci.getD());
                right_offset.x = ci.getW() - right_idx.getW();
                }
            else if (dim == axis::y)
                {
                left_idx = Index3D(ci.getW(), num_left, ci.getD());
                right_idx = Index3D(ci.getW(), num_right, ci.getD());
                right_offset.y = ci.getH() - right_idx.getH();
                }
            else // m_comm_dim == axis::z
                {
                left_idx = Index3D(ci.getW(), ci.getH(), num_left);
                right_idx = Index3D(ci.getW(), ci.getH(), num_right);
                right_offset.z = ci.getD() - right_idx.getD();
                }
            m_idx[right_face] = right_idx;
            m_idx[left_face] = left_idx;
            m_right_offset[static_cast<unsigned int>(dim)] = right_offset;

            // validate the box size
            unsigned int err = 0;
            const unsigned int nextra = m_cl->getNExtraCells();
            if (dim == axis::x && (left_idx.getW() + nextra) > right_offset.x)
                {
                err = std::max(err, left_idx.getW() + nextra - right_offset.x);
                }
            if (dim == axis::y && (left_idx.getH() + nextra) > right_offset.y)
                {
                err = std::max(err, left_idx.getH() + nextra - right_offset.y);
                }
            if (dim == axis::z && (left_idx.getD() + nextra) > right_offset.z)
                {
                err = std::max(err, left_idx.getD() + nextra - right_offset.z);
                }

            MPI_Allreduce(MPI_IN_PLACE, &err, 1, MPI_UNSIGNED, MPI_MAX, m_mpi_comm);
            if (err)
                {
                if (nextra >= err)
                    {
                    m_exec_conf->msg->error() << "mpcd: Simulation box is overdecomposed by " << err << " cells. " << std::endl;
                    m_exec_conf->msg->error() << "Reduce the number of extra communication cells, or decrease the number of ranks." << std::endl;
                    }
                else
                    {
                    m_exec_conf->msg->error() << "mpcd: Simulation box is overdecomposed by " << err << " cells. " << std::endl;
                    m_exec_conf->msg->error() << "Decrease the number of ranks." << std::endl;
                    }
                throw std::runtime_error("Simulation box is overdecomposed for MPCD");
                }
            }
    };

} // end namespace mpcd

/*!
 * \tparam T Type of data to reduce (inferred)
 * \tparam ReductionOpT Binary reduction operator (inferred)
 */
template<typename T, class ReductionOpT>
void mpcd::CellCommunicator::reduce(const GPUArray<T>& props, ReductionOpT reduction_op)
    {
    if (m_prof) m_prof->push("MPCD cell comm");
    if (props.getNumElements() < m_cl->getNCells())
        {
        m_exec_conf->msg->error() << "mpcd: cell property to be reduced is smaller than cell list dimensions" << std::endl;
        throw std::runtime_error("MPCD cell property has insufficient dimensions");
        }

    if (m_prof) m_prof->push("init");
    if (m_needs_init) initialize();
    sizeBuffers(sizeof(T));
    if (m_prof) m_prof->pop();

    // Communicate along each dimensions
    for (unsigned int dim = 0; dim < m_sysdef->getNDimensions(); ++dim)
        {
        if (!m_cl->isCommunicating(static_cast<mpcd::detail::face>(2*dim))) continue;

        // TODO: decide which pathway to take to pack / unpack the buffers
        #ifdef ENABLE_CUDA
        if (m_exec_conf->isCUDAEnabled())
            {
            packBufferGPU(props, static_cast<axis>(dim));
            }
        else
        #endif // ENABLE_CUDA
            {
            packBuffer(props, static_cast<axis>(dim));
            }

        // send along dir, and receive along the opposite direction from sending
        // TODO: decide whether to try to use CUDA-aware MPI, or just pass over CPU
            {
            #ifdef ENABLE_MPI_CUDA
            access_location::Enum mpi_loc = (m_exec_conf->isCUDAEnabled()) ? access_location::device : access_location::host;
            ArrayHandle<unsigned char> h_send_buf(m_send_buf, mpi_loc, access_mode::read);
            ArrayHandle<unsigned char> h_recv_buf(m_recv_buf, mpi_loc, access_mode::overwrite);
            if (mpi_loc == access_location::device) cudaDeviceSynchronize();
            #else
            if (m_prof) m_prof->push("copy");
            ArrayHandle<unsigned char> h_send_buf(m_send_buf, access_location::host, access_mode::read);
            ArrayHandle<unsigned char> h_recv_buf(m_recv_buf, access_location::host, access_mode::overwrite);
            if (m_prof) m_prof->pop();
            #endif // ENABLE_MPI_CUDA

            // determine face for operations
            unsigned int right_face,left_face;
            if (static_cast<axis>(dim) == axis::x)
                {
                right_face = static_cast<unsigned int>(mpcd::detail::face::east);
                left_face = static_cast<unsigned int>(mpcd::detail::face::west);
                }
            else if (static_cast<axis>(dim) == axis::y)
                {
                right_face = static_cast<unsigned int>(mpcd::detail::face::north);
                left_face = static_cast<unsigned int>(mpcd::detail::face::south);
                }
            else // m_comm_dim == axis::z
                {
                right_face = static_cast<unsigned int>(mpcd::detail::face::up);
                left_face = static_cast<unsigned int>(mpcd::detail::face::down);
                }

            const unsigned int num_right_bytes = m_idx[right_face].getNumElements()*sizeof(T);
            const unsigned int num_left_bytes = m_idx[left_face].getNumElements()*sizeof(T);
            if (m_prof) m_prof->push("MPI send/recv");
            if (m_neigh[left_face] != m_neigh[right_face])
                {
                std::vector<MPI_Request> reqs(4);
                std::vector<MPI_Status> stats(reqs.size());
                // send left, receive right
                /*
                 * the send buffer is packed left face-right face, but the
                 * receive buffer is packed right face-left face so that if
                 * the left/right neighbors are the same, the send buffer
                 * can be sent in one call rather than two
                 */
                MPI_Isend(h_send_buf.data, num_left_bytes, MPI_BYTE, m_neigh[left_face], 0, m_mpi_comm, &reqs[0]);
                MPI_Irecv(h_recv_buf.data, num_right_bytes, MPI_BYTE, m_neigh[right_face], 0, m_mpi_comm, &reqs[1]);

                // send right, receive left
                MPI_Isend(h_send_buf.data + num_left_bytes, num_right_bytes, MPI_BYTE, m_neigh[right_face], 1, m_mpi_comm, &reqs[2]);
                MPI_Irecv(h_recv_buf.data + num_right_bytes, num_left_bytes, MPI_BYTE, m_neigh[left_face], 1, m_mpi_comm, &reqs[3]);

                MPI_Waitall(reqs.size(), &reqs.front(), &stats.front());
                }
            else
                {
                std::vector<MPI_Request> reqs(2);
                std::vector<MPI_Status> stats(reqs.size());
                MPI_Isend(h_send_buf.data, num_left_bytes + num_right_bytes, MPI_BYTE, m_neigh[left_face], 0, m_mpi_comm, &reqs[0]);
                MPI_Irecv(h_recv_buf.data, num_left_bytes + num_right_bytes, MPI_BYTE, m_neigh[left_face], 0, m_mpi_comm, &reqs[1]);

                MPI_Waitall(reqs.size(), &reqs.front(), &stats.front());
                }
            #ifdef ENABLE_MPI_CUDA
            // MPI calls can execute in multiple streams, so force a synchronization before we move on
            if (mpi_loc == access_location::device) cudaDeviceSynchronize();
            #endif // ENABLE_MPI_CUDA
            if (m_prof) m_prof->pop(0, 2*(num_right_bytes + num_left_bytes));
            }

        #ifdef ENABLE_CUDA
        if (m_exec_conf->isCUDAEnabled())
            {
            unpackBufferGPU(props, reduction_op, static_cast<axis>(dim));
            }
        else
        #endif // ENABLE_CUDA
            {
            unpackBuffer(props, reduction_op, static_cast<axis>(dim));
            }
        }
    if (m_prof) m_prof->pop();
    }

/*!
 * \param props Property buffer to pack
 *
 * \tparam T Type of buffer to pack, implicit
 *
 * \post Communicated cells in \a props are packed into \a m_send_buf
 *
 * The send buffer, which is assumed to already be properly sized, is
 * reinterpreted as a pointer to type \a T. The send buffer is packed from
 * \a props using \a m_send_offset and the communication indexer, \a m_comm_idx.
 *
 * \note This method of reading and writing is not cache-friendly, but this is unavoidable
 * to some extent due to 3d indexing. It would be possible to establish a more
 * cache-friendly pathway by rearranging indexes, but the performance gains would
 * probably not be very much compared to the complexity and (lack of) code readability.
 */
template<typename T>
void mpcd::CellCommunicator::packBuffer(const GPUArray<T>& props, axis dim)
    {
    assert(m_send_buf.getNumElements() >= sizeof(T) * (m_left_idx.getNumElements() + m_right_idx.getNumElements()));

    if (m_prof) m_prof->push("pack");
    const Index3D& ci = m_cl->getCellIndexer();
    ArrayHandle<T> h_props(props, access_location::host, access_mode::read);
    ArrayHandle<unsigned char> h_send_buf(m_send_buf, access_location::host, access_mode::overwrite);
    T* send_buf = reinterpret_cast<T*>(h_send_buf.data);

    // get the offsets based on the comm dimension
    uint3 right_offset = m_right_offset[static_cast<unsigned int>(dim)];
    Index3D left_idx, right_idx;
    if (dim == axis::x)
        {
        left_idx = m_idx[static_cast<unsigned int>(mpcd::detail::face::west)];
        right_idx = m_idx[static_cast<unsigned int>(mpcd::detail::face::east)];
        }
    else if (dim == axis::y)
        {
        left_idx = m_idx[static_cast<unsigned int>(mpcd::detail::face::south)];
        right_idx = m_idx[static_cast<unsigned int>(mpcd::detail::face::north)];
        }
    else
        {
        left_idx = m_idx[static_cast<unsigned int>(mpcd::detail::face::down)];
        right_idx = m_idx[static_cast<unsigned int>(mpcd::detail::face::up)];
        }

    // pack for sending through left face
    for (unsigned int k=0; k < left_idx.getD(); ++k)
        {
        for (unsigned int j=0; j < left_idx.getH(); ++j)
            {
            for (unsigned int i=0; i < left_idx.getW(); ++i)
                {
                send_buf[left_idx(i,j,k)] = h_props.data[ci(i,j,k)];
                }
            }
        }

    // pack for sending through right face
    send_buf += left_idx.getNumElements();
    for (unsigned int k=0; k < right_idx.getD(); ++k)
        {
        for (unsigned int j=0; j < right_idx.getH(); ++j)
            {
            for (unsigned int i=0; i < right_idx.getW(); ++i)
                {
                send_buf[right_idx(i,j,k)] = h_props.data[ci(right_offset.x+i,right_offset.y+j,right_offset.z+k)];
                }
            }
        }
    if (m_prof) m_prof->pop();
    }

/*!
 * \param props Property buffer to pack
 * \param reduction_op Reduction operator to apply when unpacking buffers
 *
 * \tparam T Type of buffer to pack, implicit
 * \tparam ReductionOpT Reduction operator functor type, implicit
 *
 * \post The bytes from \a m_recv_buf are unpacked into \a props.
 */
template<typename T, class ReductionOpT>
void mpcd::CellCommunicator::unpackBuffer(const GPUArray<T>& props, ReductionOpT reduction_op, axis dim)
    {
    if (m_prof) m_prof->push("unpack");
    const Index3D& ci = m_cl->getCellIndexer();
    ArrayHandle<T> h_props(props, access_location::host, access_mode::readwrite);

    // get the offsets based on the comm dimension
    uint3 right_offset = m_right_offset[static_cast<unsigned int>(dim)];
    Index3D left_idx, right_idx;
    if (dim == axis::x)
        {
        left_idx = m_idx[static_cast<unsigned int>(mpcd::detail::face::west)];
        right_idx = m_idx[static_cast<unsigned int>(mpcd::detail::face::east)];
        }
    else if (dim == axis::y)
        {
        left_idx = m_idx[static_cast<unsigned int>(mpcd::detail::face::south)];
        right_idx = m_idx[static_cast<unsigned int>(mpcd::detail::face::north)];
        }
    else
        {
        left_idx = m_idx[static_cast<unsigned int>(mpcd::detail::face::down)];
        right_idx = m_idx[static_cast<unsigned int>(mpcd::detail::face::up)];
        }

    // unpack the right buffer
    ArrayHandle<unsigned char> h_recv_buf(m_recv_buf, access_location::host, access_mode::read);
    T* recv_buf = reinterpret_cast<T*>(h_recv_buf.data);
    for (unsigned int k=0; k < right_idx.getD(); ++k)
        {
        for (unsigned int j=0; j < right_idx.getH(); ++j)
            {
            for (unsigned int i=0; i < right_idx.getW(); ++i)
                {
                const unsigned int target = ci(right_offset.x+i,right_offset.y+j,right_offset.z+k);
                const T cur_val = h_props.data[target];
                h_props.data[target] = reduction_op(recv_buf[right_idx(i,j,k)], cur_val);
                }
            }
        }

    // unpack the left buffer
    recv_buf += right_idx.getNumElements();
    for (unsigned int k=0; k < left_idx.getD(); ++k)
        {
        for (unsigned int j=0; j < left_idx.getH(); ++j)
            {
            for (unsigned int i=0; i < left_idx.getW(); ++i)
                {
                const unsigned int target = ci(i,j,k);
                const T cur_val = h_props.data[target];
                h_props.data[target] = reduction_op(recv_buf[left_idx(i,j,k)], cur_val);
                }
            }
        }


    if (m_prof) m_prof->pop();
    }

#ifdef ENABLE_CUDA
/*!
 * \param props Property buffer to pack
 *
 * \tparam T Type of buffer to pack, implicit
 *
 * \post Communicated cells in \a props are packed into \a m_send_buf
 */
template<typename T>
void mpcd::CellCommunicator::packBufferGPU(const GPUArray<T>& props, axis dim)
    {
    assert(m_send_buf.getNumElements() >= sizeof(T) * (m_left_idx.getNumElements() + m_right_idx.getNumElements()));
    if (m_prof) m_prof->push(m_exec_conf,"pack");

    ArrayHandle<T> d_props(props, access_location::device, access_mode::read);
    ArrayHandle<unsigned char> d_send_buf(m_send_buf, access_location::device, access_mode::overwrite);
    T* send_buf = reinterpret_cast<T*>(d_send_buf.data);

    // get the offsets based on the comm dimension
    uint3 right_offset = m_right_offset[static_cast<unsigned int>(dim)];
    Index3D left_idx, right_idx;
    if (dim == axis::x)
        {
        left_idx = m_idx[static_cast<unsigned int>(mpcd::detail::face::west)];
        right_idx = m_idx[static_cast<unsigned int>(mpcd::detail::face::east)];
        }
    else if (dim == axis::y)
        {
        left_idx = m_idx[static_cast<unsigned int>(mpcd::detail::face::south)];
        right_idx = m_idx[static_cast<unsigned int>(mpcd::detail::face::north)];
        }
    else
        {
        left_idx = m_idx[static_cast<unsigned int>(mpcd::detail::face::down)];
        right_idx = m_idx[static_cast<unsigned int>(mpcd::detail::face::up)];
        }

    m_tuner_pack->begin();
    mpcd::gpu::pack_cell_buffer(send_buf,
                                send_buf + left_idx.getNumElements(),
                                left_idx,
                                right_idx,
                                right_offset,
                                d_props.data,
                                m_cl->getCellIndexer(),
                                m_tuner_pack->getParam());
    if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
    m_tuner_pack->end();
    if (m_prof) m_prof->pop(m_exec_conf);
    }

/*!
 * \param props Property buffer to pack
 * \param reduction_op Reduction operator to apply when unpacking buffers
 *
 * \tparam T Type of buffer to pack, implicit
 * \tparam ReductionOpT Reduction operator functor type, implicit
 *
 * \post The bytes from \a m_recv_buf are unpacked into \a props.
 */
template<typename T, class ReductionOpT>
void mpcd::CellCommunicator::unpackBufferGPU(const GPUArray<T>& props, ReductionOpT reduction_op, axis dim)
    {
    if (m_prof) m_prof->push("copy");
    ArrayHandle<unsigned char> d_recv_buf(m_recv_buf, access_location::device, access_mode::read);
    T* recv_buf = reinterpret_cast<T*>(d_recv_buf.data);
    if (m_prof) m_prof->pop();

    if (m_prof) m_prof->push(m_exec_conf, "unpack");
    ArrayHandle<T> d_props(props, access_location::device, access_mode::readwrite);

    // get the offsets based on the comm dimension
    uint3 right_offset = m_right_offset[static_cast<unsigned int>(dim)];
    Index3D left_idx, right_idx;
    if (dim == axis::x)
        {
        left_idx = m_idx[static_cast<unsigned int>(mpcd::detail::face::west)];
        right_idx = m_idx[static_cast<unsigned int>(mpcd::detail::face::east)];
        }
    else if (dim == axis::y)
        {
        left_idx = m_idx[static_cast<unsigned int>(mpcd::detail::face::south)];
        right_idx = m_idx[static_cast<unsigned int>(mpcd::detail::face::north)];
        }
    else
        {
        left_idx = m_idx[static_cast<unsigned int>(mpcd::detail::face::down)];
        right_idx = m_idx[static_cast<unsigned int>(mpcd::detail::face::up)];
        }

    m_tuner_unpack->begin();
    mpcd::gpu::unpack_cell_buffer(d_props.data,
                                  reduction_op,
                                  m_cl->getCellIndexer(),
                                  recv_buf + right_idx.getNumElements(),
                                  recv_buf,
                                  left_idx,
                                  right_idx,
                                  right_offset,
                                  m_tuner_unpack->getParam());
    if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
    m_tuner_unpack->end();
    if (m_prof) m_prof->pop(m_exec_conf);
    }
#endif // ENABLE_CUDA

#endif // MPCD_CELL_COMMUNICATOR_H_

#endif // ENABLE_MPI
