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
              m_left_buf(m_exec_conf),
              m_right_buf(m_exec_conf)
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
        axis m_comm_dim;                   //!< Current dimension for communication
        unsigned int m_left_neigh;              //!< Neighbor rank to the left
        unsigned int m_right_neigh;             //!< Neighbor rank to the right
        Index3D m_left_idx;                     //!< Communication indexer for receive buffers
        Index3D m_right_idx;                    //!< Communication indexer for send buffers
        GPUVector<unsigned char> m_left_buf;    //!< MPI buffer for left face
        GPUVector<unsigned char> m_right_buf;   //!< MPI buffer for right face
        uint3 m_right_offset;                   //!< Offset for reading out of properties on the face

        //! Packs the property buffer
        template<typename T>
        void packBuffer(const GPUArray<T>& props);

        //! Unpacks the property buffer
        template<typename T, class ReductionOpT>
        void unpackBuffer(const GPUArray<T>& props, ReductionOpT reduction_op);

        #ifdef ENABLE_CUDA
        std::unique_ptr<Autotuner> m_tuner_pack;    //!< Tuner for pack kernel
        std::unique_ptr<Autotuner> m_tuner_unpack;  //!< Tuner for unpack kernel

        //! Packs the property buffer on the GPU
        template<typename T>
        void packBufferGPU(const GPUArray<T>& props);

        //! Unpacks the property buffer on the GPU
        template<typename T, class ReductionOpT>
        void unpackBufferGPU(const GPUArray<T>& props, ReductionOpT reduction_op);
        #endif // ENABLE_CUDA

        //! Setup buffers for communication
        /*!
         * \param dim Dimension (x,y,z) along which communication is occurring
         * \param elem_bytes Size of data element being sent, per-cell
         */
        void setupBuffers(axis dim, size_t elem_bytes)
            {
            m_comm_dim = dim;

            // determine the "right" and "left" faces from the dimension
            unsigned int left_face, right_face;
            if (m_comm_dim == axis::x)
                {
                left_face = static_cast<unsigned int>(mpcd::detail::face::west);
                right_face = static_cast<unsigned int>(mpcd::detail::face::east);
                }
            else if (m_comm_dim == axis::y)
                {
                left_face = static_cast<unsigned int>(mpcd::detail::face::south);
                right_face = static_cast<unsigned int>(mpcd::detail::face::north);
                }
            else // m_comm_dim == axis::z
                {
                left_face = static_cast<unsigned int>(mpcd::detail::face::down);
                right_face = static_cast<unsigned int>(mpcd::detail::face::up);
                }

            // get the communication neighbors
            m_left_neigh = m_decomposition->getNeighborRank(left_face);
            m_right_neigh = m_decomposition->getNeighborRank(right_face);

            // get the number of cells being sent / received
            auto num_comm_cells = m_cl->getNComm();
            const unsigned int num_left = num_comm_cells[left_face];
            const unsigned int num_right = num_comm_cells[right_face];

            // setup the buffer indexers and the offset into the cell list
            const Index3D& ci = m_cl->getCellIndexer();
            m_right_offset = make_uint3(0,0,0);
            if (m_comm_dim == axis::x)
                {
                m_left_idx = Index3D(num_left, ci.getH(), ci.getD());
                m_right_idx = Index3D(num_right, ci.getH(), ci.getD());
                m_right_offset.x = ci.getW() - m_right_idx.getW();
                }
            else if (m_comm_dim == axis::y)
                {
                m_left_idx = Index3D(ci.getW(), num_left, ci.getD());
                m_right_idx = Index3D(ci.getW(), num_right, ci.getD());
                m_right_offset.y = ci.getH() - m_right_idx.getH();
                }
            else // m_comm_dim == axis::z
                {
                m_left_idx = Index3D(ci.getW(), ci.getH(), num_left);
                m_right_idx = Index3D(ci.getW(), ci.getH(), num_right);
                m_right_offset.z = ci.getD() - m_right_idx.getD();
                }

            // resize buffer memory
            m_left_buf.resize(2 * elem_bytes * m_left_idx.getNumElements());
            m_right_buf.resize(2 * elem_bytes * m_right_idx.getNumElements());

            // validate that the box size is OK (not overdecomposed)
            checkBoxSize();
            }

        //! Validate the local domain size is large enough for sending
        /*!
         * The communication cells are only permitted to be sent one domain
         * in either direction. This sets a minimum size on the neighboring
         * domains of num_comm_cells * cell_size. This requirement could be
         * relaxed by sending in multiple halos.
         */
        void checkBoxSize() const
            {
            unsigned int err = 0;
            const unsigned int nextra = m_cl->getNExtraCells();
            if (m_comm_dim == axis::x && (m_left_idx.getW() + nextra) > m_right_offset.x)
                {
                err = std::max(err, m_left_idx.getW() + nextra - m_right_offset.x);
                }
            if (m_comm_dim == axis::y && (m_left_idx.getH() + nextra) > m_right_offset.y)
                {
                err = std::max(err, m_left_idx.getH() + nextra - m_right_offset.y);
                }
            if (m_comm_dim == axis::z && (m_left_idx.getD() + nextra) > m_right_offset.z)
                {
                err = std::max(err, m_left_idx.getD() + nextra - m_right_offset.z);
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
    if (m_prof) m_prof->push(m_exec_conf, "MPCD cell comm");
    if (props.getNumElements() < m_cl->getNCells())
        {
        m_exec_conf->msg->error() << "mpcd: cell property to be reduced is smaller than cell list dimensions" << std::endl;
        throw std::runtime_error("MPCD cell property has insufficient dimensions");
        }

    // Communicate along each dimensions
    for (unsigned int dim = 0; dim < m_sysdef->getNDimensions(); ++dim)
        {
        if (!m_cl->isCommunicating(static_cast<mpcd::detail::face>(2*dim))) continue;

        setupBuffers(static_cast<axis>(dim), sizeof(T));

        // TODO: decide which pathway to take to pack / unpack the buffers
        #ifdef ENABLE_CUDA
        if (m_exec_conf->isCUDAEnabled())
            {
            packBufferGPU(props);
            }
        else
        #endif // ENABLE_CUDA
            {
            packBuffer(props);
            }

        // send along dir, and receive along the opposite direction from sending
        // TODO: decide whether to try to use CUDA-aware MPI, or just pass over CPU
            {
            if (m_prof) m_prof->push(m_exec_conf, "copy");
            ArrayHandle<unsigned char> h_right_buf(m_right_buf, access_location::host, access_mode::readwrite);
            ArrayHandle<unsigned char> h_left_buf(m_left_buf, access_location::host, access_mode::readwrite);
            if (m_prof) m_prof->pop(m_exec_conf);

            const unsigned int num_right_bytes = m_right_idx.getNumElements()*sizeof(T);
            const unsigned int num_left_bytes = m_left_idx.getNumElements()*sizeof(T);
            if (m_prof) m_prof->push("MPI send/recv");
            std::vector<MPI_Request> reqs(4);
            std::vector<MPI_Status> stats(reqs.size());

            // send through right face and receive through left face
            MPI_Isend(h_right_buf.data, num_right_bytes, MPI_BYTE, m_right_neigh, 0, m_mpi_comm, &reqs[0]);
            MPI_Irecv(h_left_buf.data+num_left_bytes, num_left_bytes, MPI_BYTE, m_left_neigh, 0, m_mpi_comm, &reqs[1]);

            // send through left face and receive through right face
            MPI_Isend(h_left_buf.data, num_left_bytes, MPI_BYTE, m_left_neigh, 1, m_mpi_comm, &reqs[2]);
            MPI_Irecv(h_right_buf.data+num_right_bytes, num_right_bytes, MPI_BYTE, m_right_neigh, 1, m_mpi_comm, &reqs[3]);

            MPI_Waitall(reqs.size(), &reqs.front(), &stats.front());
            if (m_prof) m_prof->pop(0, 2*(num_right_bytes + num_left_bytes));
            }

        #ifdef ENABLE_CUDA
        if (m_exec_conf->isCUDAEnabled())
            {
            unpackBufferGPU(props, reduction_op);
            }
        else
        #endif // ENABLE_CUDA
            {
            unpackBuffer(props, reduction_op);
            }
        }
    if (m_prof) m_prof->pop(m_exec_conf);
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
void mpcd::CellCommunicator::packBuffer(const GPUArray<T>& props)
    {
    assert(m_right_buf.getNumElements() >= 2 * sizeof(T) * m_right_idx.getNumElements());
    assert(m_left_buf.getNumElements() >= 2 * sizeof(T) * m_left_idx.getNumElements());

    if (m_prof) m_prof->push("pack");
    const Index3D& ci = m_cl->getCellIndexer();
    ArrayHandle<T> h_props(props, access_location::host, access_mode::read);

    // pack the left buffer
        {
        ArrayHandle<unsigned char> h_left_buf(m_left_buf, access_location::host, access_mode::overwrite);
        T* left_buf = reinterpret_cast<T*>(h_left_buf.data);
        for (unsigned int k=0; k < m_left_idx.getD(); ++k)
            {
            for (unsigned int j=0; j < m_left_idx.getH(); ++j)
                {
                for (unsigned int i=0; i < m_left_idx.getW(); ++i)
                    {
                    left_buf[m_left_idx(i,j,k)] = h_props.data[ci(i,j,k)];
                    }
                }
            }
        }

    // pack the right buffer
        {
        ArrayHandle<unsigned char> h_right_buf(m_right_buf, access_location::host, access_mode::overwrite);
        T* right_buf = reinterpret_cast<T*>(h_right_buf.data);
        for (unsigned int k=0; k < m_right_idx.getD(); ++k)
            {
            for (unsigned int j=0; j < m_right_idx.getH(); ++j)
                {
                for (unsigned int i=0; i < m_right_idx.getW(); ++i)
                    {
                    right_buf[m_right_idx(i,j,k)] = h_props.data[ci(m_right_offset.x+i,m_right_offset.y+j,m_right_offset.z+k)];
                    }
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
void mpcd::CellCommunicator::unpackBuffer(const GPUArray<T>& props, ReductionOpT reduction_op)
    {
    if (m_prof) m_prof->push("unpack");
    const Index3D& ci = m_cl->getCellIndexer();
    ArrayHandle<T> h_props(props, access_location::host, access_mode::readwrite);

    // unpack the left buffer
        {
        ArrayHandle<unsigned char> h_left_buf(m_left_buf, access_location::host, access_mode::read);
        T* left_recv_buf = reinterpret_cast<T*>(h_left_buf.data) + m_left_idx.getNumElements();
        for (unsigned int k=0; k < m_left_idx.getD(); ++k)
            {
            for (unsigned int j=0; j < m_left_idx.getH(); ++j)
                {
                for (unsigned int i=0; i < m_left_idx.getW(); ++i)
                    {
                    const unsigned int target = ci(i,j,k);
                    const T cur_val = h_props.data[target];
                    h_props.data[target] = reduction_op(left_recv_buf[m_left_idx(i,j,k)], cur_val);
                    }
                }
            }
        }

    // unpack the right buffer
        {
        ArrayHandle<unsigned char> h_right_buf(m_right_buf, access_location::host, access_mode::read);
        T* right_recv_buf = reinterpret_cast<T*>(h_right_buf.data) + m_right_idx.getNumElements();
        for (unsigned int k=0; k < m_right_idx.getD(); ++k)
            {
            for (unsigned int j=0; j < m_right_idx.getH(); ++j)
                {
                for (unsigned int i=0; i < m_right_idx.getW(); ++i)
                    {
                    const unsigned int target = ci(m_right_offset.x+i,m_right_offset.y+j,m_right_offset.z+k);
                    const T cur_val = h_props.data[target];
                    h_props.data[target] = reduction_op(right_recv_buf[m_right_idx(i,j,k)], cur_val);
                    }
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
void mpcd::CellCommunicator::packBufferGPU(const GPUArray<T>& props)
    {
    assert(m_right_buf.getNumElements() >= 2 * sizeof(T) * m_right_idx.getNumElements());
    assert(m_left_buf.getNumElements() >= 2 * sizeof(T) * m_left_idx.getNumElements());

    ArrayHandle<T> d_props(props, access_location::device, access_mode::read);

    if (m_prof) m_prof->push(m_exec_conf, "copy");
    ArrayHandle<unsigned char> d_left_buf(m_left_buf, access_location::device, access_mode::overwrite);
    T* left_buf = reinterpret_cast<T*>(d_left_buf.data);

    ArrayHandle<unsigned char> d_right_buf(m_right_buf, access_location::device, access_mode::overwrite);
    T* right_buf = reinterpret_cast<T*>(d_right_buf.data);
    if (m_prof) m_prof->pop(m_exec_conf);

    if (m_prof) m_prof->push(m_exec_conf,"pack");
    m_tuner_pack->begin();
    mpcd::gpu::pack_cell_buffer(left_buf,
                                right_buf,
                                m_left_idx,
                                m_right_idx,
                                m_right_offset,
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
void mpcd::CellCommunicator::unpackBufferGPU(const GPUArray<T>& props, ReductionOpT reduction_op)
    {
    ArrayHandle<T> d_props(props, access_location::device, access_mode::readwrite);

    if (m_prof) m_prof->push(m_exec_conf, "copy");
    ArrayHandle<unsigned char> d_left_buf(m_left_buf, access_location::device, access_mode::read);
    T* left_buf = reinterpret_cast<T*>(d_left_buf.data) + m_left_idx.getNumElements();;

    ArrayHandle<unsigned char> d_right_buf(m_right_buf, access_location::device, access_mode::read);
    T* right_buf = reinterpret_cast<T*>(d_right_buf.data) + m_right_idx.getNumElements();
    if (m_prof) m_prof->pop(m_exec_conf);


    if (m_prof) m_prof->push(m_exec_conf, "unpack");
    m_tuner_unpack->begin();
    mpcd::gpu::unpack_cell_buffer(d_props.data,
                                  reduction_op,
                                  m_cl->getCellIndexer(),
                                  left_buf,
                                  right_buf,
                                  m_left_idx,
                                  m_right_idx,
                                  m_right_offset,
                                  m_tuner_unpack->getParam());
    if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
    m_tuner_unpack->end();
    if (m_prof) m_prof->pop(m_exec_conf);
    }
#endif // ENABLE_CUDA

#endif // MPCD_CELL_COMMUNICATOR_H_

#endif // ENABLE_MPI
