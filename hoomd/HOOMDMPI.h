// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __HOOMD_MPI_H__
#define __HOOMD_MPI_H__

/*! \file HOOMDMPI.h
    \brief Defines common functions for MPI operations

    The functions provided here imitate some basic boost.MPI functionality.

    Usage of boost.Serialization is made as described in
    https://stackoverflow.com/questions/3015582/
*/

#ifdef ENABLE_MPI

#include "HOOMDMath.h"
#include "VectorMath.h"

#include <mpi.h>

#include <algorithm>
#include <numeric>
#include <queue>
#include <sstream>
#include <tuple>
#include <vector>

#include <cereal/archives/binary.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/set.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/utility.hpp> // std::pair
#include <cereal/types/vector.hpp>

#ifdef ENABLE_TBB
// https://www.threadingbuildingblocks.org/docs/help/reference/appendices/known_issues/linux_os.html
#include <tbb/concurrent_unordered_map.h>
#include <tbb/concurrent_unordered_set.h>
#include <tbb/concurrent_vector.h>
#endif

namespace cereal
    {
//! Serialization functions for some of our data types
//! Serialization of Scalar4
template<class Archive> void serialize(Archive& ar, hoomd::Scalar4& s, const unsigned int version)
    {
    ar&(hoomd::Scalar&)s.x;
    ar&(hoomd::Scalar&)s.y;
    ar&(hoomd::Scalar&)s.z;
    ar&(hoomd::Scalar&)s.w;
    }

//! Serialization of Scalar3
template<class Archive> void serialize(Archive& ar, hoomd::Scalar3& s, const unsigned int version)
    {
    ar & s.x;
    ar & s.y;
    ar & s.z;
    }

//! Serialization of int3
template<class Archive> void serialize(Archive& ar, int3& i, const unsigned int version)
    {
    ar & i.x;
    ar & i.y;
    ar & i.z;
    }

//! serialization of uint2
template<class Archive> void serialize(Archive& ar, uint2& u, const unsigned int version)
    {
    ar & u.x;
    ar & u.y;
    }

//! serialization of uint3
template<class Archive> void serialize(Archive& ar, uint3& u, const unsigned int version)
    {
    ar & u.x;
    ar & u.y;
    ar & u.z;
    }

//! serialization of uchar3
template<class Archive> void serialize(Archive& ar, uchar3& u, const unsigned int version)
    {
    ar & u.x;
    ar & u.y;
    ar & u.z;
    }

//! Serialization of vec3<Real>
template<class Archive, class Real>
void serialize(Archive& ar, hoomd::vec3<Real>& v, const unsigned int version)
    {
    ar & v.x;
    ar & v.y;
    ar & v.z;
    }

//! Serialization of quat<Real>
template<class Archive, class Real>
void serialize(Archive& ar, hoomd::quat<Real>& q, const unsigned int version)
    {
    // serialize both members
    ar & q.s;
    ar & q.v;
    }

#ifdef ENABLE_TBB
//! Serialization for tbb::concurrent_vector
template<class Archive, class T, class A>
inline void save(Archive& ar, tbb::concurrent_vector<T, A> const& vector)
    {
    ar(make_size_tag(static_cast<size_type>(vector.size()))); // number of elements
    for (auto&& v : vector)
        ar(v);
    }

template<class Archive, class T, class A>
inline void load(Archive& ar, tbb::concurrent_vector<T, A>& vector)
    {
    size_type size;
    ar(make_size_tag(size));

    vector.resize(static_cast<std::size_t>(size));
    for (auto&& v : vector)
        ar(v);
    }

//! Serialization of tbb::concurrent_unordered_set
namespace tbb_unordered_set_detail
    {
//! @internal
template<class Archive, class SetT> inline void save(Archive& ar, SetT const& set)
    {
    ar(make_size_tag(static_cast<size_type>(set.size())));

    for (const auto& i : set)
        ar(i);
    }

//! @internal
template<class Archive, class SetT> inline void load(Archive& ar, SetT& set)
    {
    size_type size;
    ar(make_size_tag(size));

    set.clear();

    for (size_type i = 0; i < size; ++i)
        {
        typename SetT::key_type key;

        ar(key);
        set.emplace(std::move(key));
        }
    }
    } // namespace tbb_unordered_set_detail

//! Saving for tbb::concurrent_unordered_set
template<class Archive, class K, class H, class KE, class A>
inline void save(Archive& ar, tbb::concurrent_unordered_set<K, H, KE, A> const& unordered_set)
    {
    tbb_unordered_set_detail::save(ar, unordered_set);
    }

//! Loading for tbb::concurrent_unordered_set
template<class Archive, class K, class H, class KE, class A>
inline void load(Archive& ar, tbb::concurrent_unordered_set<K, H, KE, A>& unordered_set)
    {
    tbb_unordered_set_detail::load(ar, unordered_set);
    }
#endif
    } // namespace cereal

namespace hoomd
    {
#if HOOMD_LONGREAL_SIZE == 32
//! Define MPI_FLOAT as Scalar MPI data type
const MPI_Datatype MPI_HOOMD_SCALAR = MPI_FLOAT;
const MPI_Datatype MPI_HOOMD_SCALAR_INT = MPI_FLOAT_INT;
#else
//! Define MPI_DOUBLE as Scalar MPI data type
const MPI_Datatype MPI_HOOMD_SCALAR = MPI_DOUBLE;
const MPI_Datatype MPI_HOOMD_SCALAR_INT = MPI_DOUBLE_INT;
#endif

typedef struct
    {
    Scalar s;
    int i;
    } Scalar_Int;

//! Wrapper around MPI_Bcast that handles any serializable object
template<typename T> void bcast(T& val, unsigned int root, const MPI_Comm mpi_comm)
    {
    int rank;
    MPI_Comm_rank(mpi_comm, &rank);

    char* buf = NULL;
    unsigned int recv_count;
    if (rank == (int)root)
        {
        std::stringstream s(std::ios_base::out | std::ios_base::binary);
        cereal::BinaryOutputArchive ar(s);

        // serialize object
        ar << val;

        // do not forget to flush stream
        s.flush();

        // copy string to send buffer
        std::string str = s.str();
        recv_count = (unsigned int)str.size();
        buf = new char[recv_count];
        str.copy(buf, recv_count);
        }

    MPI_Bcast(&recv_count, 1, MPI_INT, root, mpi_comm);
    if (rank != (int)root)
        buf = new char[recv_count];

    MPI_Bcast(buf, recv_count, MPI_BYTE, root, mpi_comm);

    if (rank != (int)root)
        {
        // de-serialize
        std::stringstream s(std::string(buf, recv_count),
                            std::ios_base::in | std::ios_base::binary);
        cereal::BinaryInputArchive ar(s);

        ar >> val;
        }

    delete[] buf;
    }

//! Wrapper around MPI_Scatterv that scatters a vector of serializable objects
template<typename T>
void scatter_v(const std::vector<T>& in_values,
               T& out_value,
               unsigned int root,
               const MPI_Comm mpi_comm)
    {
    int rank;
    int size;
    MPI_Comm_rank(mpi_comm, &rank);
    MPI_Comm_size(mpi_comm, &size);

    assert(in_values.size() == (unsigned int)size);

    unsigned int recv_count;
    int* send_counts = NULL;
    int* displs = NULL;

    char* sbuf = NULL;
    if (rank == (int)root)
        {
        send_counts = new int[size];
        displs = new int[size];
        // construct a vector of serialized objects
        typename std::vector<T>::const_iterator it;
        std::vector<std::string> str;
        unsigned int len = 0;
        for (it = in_values.begin(); it != in_values.end(); ++it)
            {
            unsigned int idx = (unsigned int)(it - in_values.begin());
            std::stringstream s(std::ios_base::out | std::ios_base::binary);
            cereal::BinaryOutputArchive ar(s);

            // serialize object
            ar << *it;
            s.flush();
            str.push_back(s.str());

            displs[idx] = (idx > 0) ? displs[idx - 1] + send_counts[idx - 1] : 0;
            send_counts[idx] = (unsigned int)(str[idx].length());
            len += send_counts[idx];
            }

        // pack vector into send buffer
        sbuf = new char[len];
        for (unsigned int idx = 0; idx < in_values.size(); idx++)
            str[idx].copy(sbuf + displs[idx], send_counts[idx]);
        }

    // scatter sizes of serialized vector elements
    MPI_Scatter(send_counts, 1, MPI_INT, &recv_count, 1, MPI_INT, root, mpi_comm);

    // allocate receive buffer
    char* rbuf = new char[recv_count];

    // scatter actual data
    MPI_Scatterv(sbuf, send_counts, displs, MPI_BYTE, rbuf, recv_count, MPI_BYTE, root, mpi_comm);

    // de-serialize data
    std::stringstream s(std::string(rbuf, recv_count), std::ios_base::in | std::ios_base::binary);
    cereal::BinaryInputArchive ar(s);

    ar >> out_value;

    if (rank == (int)root)
        {
        delete[] send_counts;
        delete[] displs;
        delete[] sbuf;
        }
    delete[] rbuf;
    }

//! Wrapper around MPI_Gatherv
template<typename T>
void gather_v(const T& in_value,
              std::vector<T>& out_values,
              unsigned int root,
              const MPI_Comm mpi_comm)
    {
    int rank;
    int size;
    MPI_Comm_rank(mpi_comm, &rank);
    MPI_Comm_size(mpi_comm, &size);

    // serialize in_value
    std::stringstream s(std::ios_base::out | std::ios_base::binary);
    cereal::BinaryOutputArchive ar(s);

    ar << in_value;
    s.flush();

    // copy into send buffer
    std::string str = s.str();
    unsigned int send_count = (unsigned int)str.length();

    int* recv_counts = NULL;
    int* displs = NULL;
    if (rank == (int)root)
        {
        out_values.resize(size);
        recv_counts = new int[size];
        displs = new int[size];
        }

    // gather lengths of buffers
    MPI_Gather(&send_count, 1, MPI_INT, recv_counts, 1, MPI_INT, root, mpi_comm);

    char* rbuf = NULL;
    if (rank == (int)root)
        {
        unsigned int len = 0;
        for (unsigned int i = 0; i < (unsigned int)size; i++)
            {
            displs[i] = (i > 0) ? displs[i - 1] + recv_counts[i - 1] : 0;
            len += recv_counts[i];
            }
        rbuf = new char[len];
        }

    // now gather actual objects
    MPI_Gatherv((void*)str.data(),
                send_count,
                MPI_BYTE,
                rbuf,
                recv_counts,
                displs,
                MPI_BYTE,
                root,
                mpi_comm);

    // on root processor, de-serialize data
    if (rank == (int)root)
        {
        for (unsigned int i = 0; i < out_values.size(); i++)
            {
            std::stringstream s(std::string(rbuf + displs[i], recv_counts[i]),
                                std::ios_base::in | std::ios_base::binary);
            cereal::BinaryInputArchive ar(s);

            ar >> out_values[i];
            }

        delete[] displs;
        delete[] recv_counts;
        delete[] rbuf;
        }
    }

//! Wrapper around MPI_Allgatherv
template<typename T>
void all_gather_v(const T& in_value, std::vector<T>& out_values, const MPI_Comm mpi_comm)
    {
    int rank;
    int size;
    MPI_Comm_rank(mpi_comm, &rank);
    MPI_Comm_size(mpi_comm, &size);

    // serialize in_value
    std::stringstream s(std::ios_base::out | std::ios_base::binary);
    cereal::BinaryOutputArchive ar(s);

    ar << in_value;
    s.flush();

    // copy into send buffer
    std::string str = s.str();
    unsigned int send_count = (unsigned int)str.length();

    // allocate memory for buffer lengths
    out_values.resize(size);
    int* recv_counts = new int[size];
    int* displs = new int[size];

    // gather lengths of buffers
    MPI_Allgather(&send_count, 1, MPI_INT, recv_counts, 1, MPI_INT, mpi_comm);

    // allocate receiver buffer
    unsigned int len = 0;
    for (unsigned int i = 0; i < (unsigned int)size; i++)
        {
        displs[i] = (i > 0) ? displs[i - 1] + recv_counts[i - 1] : 0;
        len += recv_counts[i];
        }
    char* rbuf = new char[len];

    // now gather actual objects
    MPI_Allgatherv((void*)str.data(),
                   send_count,
                   MPI_BYTE,
                   rbuf,
                   recv_counts,
                   displs,
                   MPI_BYTE,
                   mpi_comm);

    // de-serialize data
    for (unsigned int i = 0; i < out_values.size(); i++)
        {
        std::stringstream s(std::string(rbuf + displs[i], recv_counts[i]),
                            std::ios_base::in | std::ios_base::binary);
        cereal::BinaryInputArchive ar(s);

        ar >> out_values[i];
        }

    delete[] displs;
    delete[] recv_counts;
    delete[] rbuf;
    }

//! Wrapper around MPI_Send that handles any serializable object
template<typename T> void send(const T& val, const unsigned int dest, const MPI_Comm mpi_comm)
    {
    int rank;
    MPI_Comm_rank(mpi_comm, &rank);
    if (rank == static_cast<int>(dest)) // Quick exit, if dest is src
        return;
    char* buf = NULL;
    int recv_count;

    std::stringstream s(std::ios_base::out | std::ios_base::binary);
    cereal::BinaryOutputArchive ar(s);

    // serialize object
    ar << val;

    // do not forget to flush stream
    s.flush();

    // copy string to send buffer
    std::string str = s.str();
    recv_count = (unsigned int)str.size();
    buf = new char[recv_count];
    str.copy(buf, recv_count);

    MPI_Send(&recv_count, 1, MPI_INT, dest, 0, mpi_comm);

    MPI_Send(buf, recv_count, MPI_BYTE, dest, 0, mpi_comm);

    delete[] buf;
    }

//! Wrapper around MPI_Recv that handles any serializable object
template<typename T> void recv(T& val, const unsigned int src, const MPI_Comm mpi_comm)
    {
    int rank;
    MPI_Comm_rank(mpi_comm, &rank);
    if (rank == static_cast<int>(src)) // Quick exit if src is dest.
        return;

    int recv_count;

    MPI_Recv(&recv_count, 1, MPI_INT, src, 0, mpi_comm, MPI_STATUS_IGNORE);

    char* buf = NULL;
    buf = new char[recv_count];

    MPI_Recv(buf, recv_count, MPI_BYTE, src, 0, mpi_comm, MPI_STATUS_IGNORE);

    // de-serialize
    std::stringstream s(std::string(buf, recv_count), std::ios_base::in | std::ios_base::binary);
    cereal::BinaryInputArchive ar(s);
    ar >> val;

    delete[] buf;
    }

/// Helper class that gathers local values from ranks and orders them in ascending tag order.
/**
    To use:
    1) Call setLocalTagsSorted() to establish the sizes and orders of the arrays to be gathered.
    2) Call gatherArray() to combine the per-rank local arrays into a global array in ascending tag
       order.
    3) Repeat step 2 for as many arrays as needed (all must be in the same tag order).

    GatherTagOrder maintains internal cache variables. Reuse an existing GatherTagOrder instances
    to avoid costly memory reallocations.

    Future expansion: Provide an alternate entry point setLocalTagsUnsorted() that allows callers
    to provide arrays directly in index order.
**/
class GatherTagOrder
    {
    public:
    /// Construct GatherTagOrder
    /** \param mpi_comm MPI Communicator.
        \param root Rank to gather on.
    */

    GatherTagOrder(const MPI_Comm mpi_comm = MPI_COMM_WORLD, int root = 0)
        : m_mpi_communicator(mpi_comm), m_root(root)
        {
        int rank;
        int n_ranks;
        MPI_Comm_rank(m_mpi_communicator, &rank);
        MPI_Comm_size(m_mpi_communicator, &n_ranks);

        if (rank == m_root)
            {
            m_recv_counts.resize(n_ranks);
            m_recv_bytes.resize(n_ranks);
            m_displacements.resize(n_ranks);
            m_displacement_bytes.resize(n_ranks);
            }
        }

    ~GatherTagOrder()
        {
        if (m_gather_buffer != nullptr)
            {
            free(m_gather_buffer);
            }
        }

    /// Provide the tag order for the local arrays to be gathered.
    void setLocalTagsSorted(std::vector<unsigned int> local_tag)
        {
        setLocalTagsSorted(local_tag.data(), static_cast<int>(local_tag.size()));
        }

    /// Provide the tag order for the local arrays to be gathered.
    void setLocalTagsSorted(unsigned int* local_tag, int n_local_tags)
        {
        // First, gather all the rank local tags onto the root processor.
        int rank, n_ranks;
        MPI_Comm_rank(m_mpi_communicator, &rank);
        MPI_Comm_size(m_mpi_communicator, &n_ranks);

        MPI_Gather(&n_local_tags,
                   1,
                   MPI_INT,
                   m_recv_counts.data(),
                   1,
                   MPI_INT,
                   m_root,
                   m_mpi_communicator);

        unsigned int* gathered_tags = nullptr;
        if (rank == m_root)
            {
            // std::exclusive requires gcc10+:
            // https://stackoverflow.com/questions/55771604/g-with-stdexclusive-scan-c17
            m_displacements[0] = 0;
            for (size_t i = 1; i < m_displacements.size(); i++)
                {
                m_displacements[i] = m_displacements[i - 1] + m_recv_counts[i - 1];
                }
            m_n_global_tags = m_displacements[n_ranks - 1] + m_recv_counts[n_ranks - 1];
            gathered_tags = allocateGatherBuffer<unsigned int>(m_n_global_tags);
            }

        MPI_Gatherv(static_cast<void*>(local_tag),
                    n_local_tags,
                    MPI_UNSIGNED,
                    static_cast<void*>(gathered_tags),
                    m_recv_counts.data(),
                    m_displacements.data(),
                    MPI_UNSIGNED,
                    m_root,
                    m_mpi_communicator);

        // Next, perform a k-way mergesort to sort the tags. The output of this sort is a
        // list that indicates the global index order in which to read gathered arrays.
        if (rank == m_root)
            {
            m_order.resize(0);
            assert(m_sort_queue.empty());

            for (int i = 0; i < n_ranks; i++)
                {
                if (m_recv_counts[i] > 0)
                    {
                    m_sort_queue.push(std::make_tuple(gathered_tags[m_displacements[i]], i, 0));
                    }
                }

            while (!m_sort_queue.empty())
                {
                unsigned int item_rank = std::get<1>(m_sort_queue.top());
                unsigned int local_index = std::get<2>(m_sort_queue.top());
                unsigned int global_index = m_displacements[item_rank] + local_index;
                m_order.push_back(global_index);

                local_index++;
                if (local_index < static_cast<unsigned int>(m_recv_counts[item_rank]))
                    {
                    m_sort_queue.push(
                        std::make_tuple(gathered_tags[global_index + 1], item_rank, local_index));
                    }

                m_sort_queue.pop();
                }

            assert(m_order.size() == static_cast<size_t>(m_n_global_tags));
            }
        }

    /// Gather a given local array into a global array in tag order
    template<class T>
    void gatherArray(std::vector<T>& global_array, const std::vector<T>& local_array)
        {
        gatherArray(global_array, local_array.data(), static_cast<int>(local_array.size()));
        }

    /// Gather a given local array into a global array in tag order
    template<class T>
    void gatherArray(std::vector<T>& global_array, const T* local_array, int local_size)
        {
        int rank;

        MPI_Comm_rank(m_mpi_communicator, &rank);

        T* gathered_array = nullptr;
        if (rank == m_root)
            {
            gathered_array = allocateGatherBuffer<T>(m_n_global_tags);
            std::transform(m_recv_counts.begin(),
                           m_recv_counts.end(),
                           m_recv_bytes.begin(),
                           [](int v) { return v * sizeof(T); });
            std::transform(m_displacements.begin(),
                           m_displacements.end(),
                           m_displacement_bytes.begin(),
                           [](int v) { return v * sizeof(T); });
            }

        MPI_Gatherv((void*)(local_array),
                    static_cast<int>(local_size * sizeof(T)),
                    MPI_BYTE,
                    (void*)(gathered_array),
                    m_recv_bytes.data(),
                    m_displacement_bytes.data(),
                    MPI_BYTE,
                    m_root,
                    m_mpi_communicator);

        global_array.resize(0);
        if (rank == m_root)
            {
            for (unsigned int i : m_order)
                {
                global_array.push_back(gathered_array[i]);
                }
            }
        }

    private:
    MPI_Comm m_mpi_communicator;
    int m_root;

    std::vector<int> m_recv_counts, m_recv_bytes;
    std::vector<int> m_displacements, m_displacement_bytes;
    std::vector<unsigned int> m_order;

    /// Number of global tags (only set on root rank).
    unsigned int m_n_global_tags;

    /// Priority queue for k-way merge sort: tuple of tag, rank, local index
    typedef std::tuple<unsigned int, unsigned int, unsigned int> queue_item;
    std::priority_queue<queue_item, std::vector<queue_item>, std::greater<queue_item>> m_sort_queue;

    void* m_gather_buffer = nullptr;
    size_t m_gather_buffer_size = 0;

    /// Allocate or return an existing pointer to the gather buffer with space for n T objects.
    template<class T> T* allocateGatherBuffer(int n)
        {
        size_t needed_bytes = sizeof(T) * n;
        if (needed_bytes > m_gather_buffer_size)
            {
            if (m_gather_buffer != nullptr)
                {
                free(m_gather_buffer);
                }

            m_gather_buffer = malloc(needed_bytes);

            if (m_gather_buffer == nullptr)
                {
                throw std::bad_alloc();
                }
            }

        return static_cast<T*>(m_gather_buffer);
        }
    };
    } // namespace hoomd

#endif // ENABLE_MPI
#endif // __HOOMD_MPI_H__
