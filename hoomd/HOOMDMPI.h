// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

#ifndef __HOOMD_MPI_H__
#define __HOOMD_MPI_H__

/*! \file HOOMDMPI.h
    \brief Defines common functions for MPI operations

    The functions provided here imitate some basic boost.MPI functionality.

    Usage of boost.Serialization is made as described in
    http://stackoverflow.com/questions/3015582/
*/

#ifdef ENABLE_MPI

#include "HOOMDMath.h"

#include <mpi.h>

#include <sstream>
#include <vector>

#include <cereal/types/set.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/utility.hpp> // std::pair
#include <cereal/archives/binary.hpp>

#ifdef SINGLE_PRECISION
//! Define MPI_FLOAT as Scalar MPI data type
const MPI_Datatype MPI_HOOMD_SCALAR = MPI_FLOAT;
const MPI_Datatype MPI_HOOMD_SCALAR_INT = MPI_FLOAT_INT;
#else
//! Define MPI_DOUBLE as Scalar MPI data type
const MPI_Datatype MPI_HOOMD_SCALAR = MPI_DOUBLE;
const MPI_Datatype MPI_HOOMD_SCALAR_INT = MPI_DOUBLE_INT;
#endif

#ifdef ENABLE_TBB
// https://www.threadingbuildingblocks.org/docs/help/reference/appendices/known_issues/linux_os.html
#include <tbb/concurrent_vector.h>
#include <tbb/concurrent_unordered_set.h>
#include <tbb/concurrent_unordered_map.h>
#endif

typedef struct{
    Scalar s;
    int i;
    }Scalar_Int;


namespace cereal
   {
   //! Serialization functions for some of our data types
   //! Serialization of Scalar4
   template<class Archive>
   void serialize(Archive & ar, Scalar4 & s, const unsigned int version)
       {
       ar & s.x;
       ar & s.y;
       ar & s.z;
       ar & s.w;
       }

    //! Serialization of Scalar3
    template<class Archive>
    void serialize(Archive & ar, Scalar3 & s, const unsigned int version)
        {
        ar & s.x;
        ar & s.y;
        ar & s.z;
        }


    //! Serialization of int3
    template<class Archive>
    void serialize(Archive & ar, int3 & i, const unsigned int version)
        {
        ar & i.x;
        ar & i.y;
        ar & i.z;
        }

    //! serialization of uint2
    template<class Archive>
    void serialize(Archive & ar, uint2 & u, const unsigned int version)
        {
        ar & u.x;
        ar & u.y;
        }

    //! serialization of uint3
    template<class Archive>
    void serialize(Archive & ar, uint3 & u, const unsigned int version)
        {
        ar & u.x;
        ar & u.y;
        ar & u.z;
        }

    //! serialization of uchar3
    template<class Archive>
    void serialize(Archive & ar, uchar3 & u, const unsigned int version)
        {
        ar & u.x;
        ar & u.y;
        ar & u.z;
        }

      #ifdef ENABLE_TBB
      //! Serialization for tbb::concurrent_vector
      template <class Archive, class T, class A> inline
      void save( Archive & ar, tbb::concurrent_vector<T, A> const & vector )
        {
        ar( make_size_tag( static_cast<size_type>(vector.size()) ) ); // number of elements
        for(auto && v : vector)
            ar( v );
        }

      template <class Archive, class T, class A> inline
      void load( Archive & ar, tbb::concurrent_vector<T, A> & vector )
        {
        size_type size;
        ar( make_size_tag( size ) );

        vector.resize( static_cast<std::size_t>( size ) );
        for(auto && v : vector)
            ar( v );
        }

    //! Serialization of tbb::concurrent_unordered_set
    namespace tbb_unordered_set_detail
        {
        //! @internal
        template <class Archive, class SetT> inline
        void save( Archive & ar, SetT const & set )
            {
            ar( make_size_tag( static_cast<size_type>(set.size()) ) );

            for( const auto & i : set )
                ar( i );
            }

        //! @internal
        template <class Archive, class SetT> inline
        void load( Archive & ar, SetT & set )
            {
            size_type size;
            ar( make_size_tag( size ) );

            set.clear();

            for( size_type i = 0; i < size; ++i )
                {
                typename SetT::key_type key;

                ar( key );
                set.emplace( std::move( key ) );
                }
            }
        }

    //! Saving for tbb::concurrent_unordered_set
    template <class Archive, class K, class H, class KE, class A> inline
    void save(Archive & ar, tbb::concurrent_unordered_set<K, H, KE, A> const & unordered_set )
        {
        tbb_unordered_set_detail::save( ar, unordered_set );
        }

    //! Loading for tbb::concurrent_unordered_set
    template <class Archive, class K, class H, class KE, class A> inline
    void load( Archive & ar, tbb::concurrent_unordered_set<K, H, KE, A> & unordered_set )
        {
        tbb_unordered_set_detail::load( ar, unordered_set );
        }
    #endif
    }


//! Wrapper around MPI_Bcast that handles any serializable object
template<typename T>
void bcast(T& val, unsigned int root, const MPI_Comm mpi_comm)
    {
    int rank;
    MPI_Comm_rank(mpi_comm, &rank);

    char *buf = NULL;
    int recv_count;
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
        recv_count = str.size();
        buf = new char[recv_count];
        str.copy(buf, recv_count);
        }

    MPI_Bcast(&recv_count, 1, MPI_INT, root, mpi_comm);
    if (rank != (int) root)
        buf = new char[recv_count];

    MPI_Bcast(buf, recv_count, MPI_BYTE, root, mpi_comm);

    if (rank != (int)root)
        {
        // de-serialize
        std::stringstream s(std::string(buf, recv_count), std::ios_base::in | std::ios_base::binary);
        cereal::BinaryInputArchive ar(s);

        ar >> val;
        }

    delete[] buf;
    }

//! Wrapper around MPI_Scatterv that scatters a vector of serializable objects
template<typename T>
void scatter_v(const std::vector<T>& in_values, T& out_value, unsigned int root, const MPI_Comm mpi_comm)
    {
    int rank;
    int size;
    MPI_Comm_rank(mpi_comm, &rank);
    MPI_Comm_size(mpi_comm, &size);

    assert(in_values.size() == (unsigned int) size);

    unsigned int recv_count;
    int *send_counts = NULL;
    int *displs = NULL;

    char *sbuf = NULL;
    if (rank == (int)root)
        {
        send_counts = new int[size];
        displs = new int[size];
        // construct a vector of serialized objects
        typename std::vector<T>::const_iterator it;
        std::vector<std::string> str;
        unsigned int len = 0;
        for (it = in_values.begin(); it!= in_values.end(); ++it)
            {
            unsigned int idx = it - in_values.begin();
            std::stringstream s(std::ios_base::out | std::ios_base::binary);
            cereal::BinaryOutputArchive ar(s);

            // serialize object
            ar << *it;
            s.flush();
            str.push_back(s.str());

            displs[idx] = (idx > 0) ? displs[idx-1]+send_counts[idx-1] : 0;
            send_counts[idx] = str[idx].length();
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
    char *rbuf = new char[recv_count];

    // scatter actual data
    MPI_Scatterv(sbuf, send_counts, displs, MPI_BYTE, rbuf, recv_count, MPI_BYTE, root, mpi_comm);

    // de-serialize data
    std::stringstream s(std::string(rbuf, recv_count), std::ios_base::in | std::ios_base::binary);
    cereal::BinaryInputArchive ar(s);

    ar >> out_value;

    if (rank == (int) root)
        {
        delete[] send_counts;
        delete[] displs;
        delete[] sbuf;
        }
    delete[] rbuf;
    }

//! Wrapper around MPI_Gatherv
template<typename T>
void gather_v(const T& in_value, std::vector<T> & out_values, unsigned int root, const MPI_Comm mpi_comm)
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
    unsigned int send_count = str.length();

    int *recv_counts = NULL;
    int *displs = NULL;
    if (rank == (int) root)
        {
        out_values.resize(size);
        recv_counts = new int[size];
        displs = new int[size];
        }

    // gather lengths of buffers
    MPI_Gather(&send_count, 1, MPI_INT, recv_counts, 1, MPI_INT, root, mpi_comm);

    char *rbuf = NULL;
    if (rank == (int) root)
        {
        unsigned int len = 0;
        for (unsigned int i = 0; i < (unsigned int) size; i++)
            {
            displs[i] = (i > 0) ? displs[i-1] + recv_counts[i-1] : 0;
            len += recv_counts[i];
            }
        rbuf = new char[len];
        }

    // now gather actual objects
    MPI_Gatherv((void *)str.data(), send_count, MPI_BYTE, rbuf, recv_counts, displs, MPI_BYTE, root, mpi_comm);

    // on root processor, de-serialize data
    if (rank == (int) root)
        {
        for (unsigned int i = 0; i < out_values.size(); i++)
            {
            std::stringstream s(std::string(rbuf + displs[i], recv_counts[i]), std::ios_base::in | std::ios_base::binary);
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
void all_gather_v(const T& in_value, std::vector<T> & out_values, const MPI_Comm mpi_comm)
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
    unsigned int send_count = str.length();

    // allocate memory for buffer lengths
    out_values.resize(size);
    int *recv_counts = new int[size];
    int *displs = new int[size];

    // gather lengths of buffers
    MPI_Allgather(&send_count, 1, MPI_INT, recv_counts, 1, MPI_INT, mpi_comm);

    // allocate receiver buffer
    unsigned int len = 0;
    for (unsigned int i = 0; i < (unsigned int) size; i++)
        {
        displs[i] = (i > 0) ? displs[i-1] + recv_counts[i-1] : 0;
        len += recv_counts[i];
        }
    char *rbuf = new char[len];

    // now gather actual objects
    MPI_Allgatherv((void *)str.data(), send_count, MPI_BYTE, rbuf, recv_counts, displs, MPI_BYTE, mpi_comm);

    // de-serialize data
    for (unsigned int i = 0; i < out_values.size(); i++)
        {
        std::stringstream s(std::string(rbuf + displs[i], recv_counts[i]), std::ios_base::in | std::ios_base::binary);
        cereal::BinaryInputArchive ar(s);

        ar >> out_values[i];
        }

    delete[] displs;
    delete[] recv_counts;
    delete[] rbuf;
    }

//! Wrapper around MPI_Send that handles any serializable object
template<typename T>
void send(const T& val,const unsigned int dest, const MPI_Comm mpi_comm)
    {
    int rank;
    MPI_Comm_rank(mpi_comm, &rank);
    if(rank == static_cast<int>(dest) ) //Quick exit, if dest is src
      return;
    char *buf = NULL;
    int recv_count;

    std::stringstream s(std::ios_base::out | std::ios_base::binary);
    cereal::BinaryOutputArchive ar(s);

    // serialize object
    ar << val;

    // do not forget to flush stream
    s.flush();

    // copy string to send buffer
    std::string str = s.str();
    recv_count = str.size();
    buf = new char[recv_count];
    str.copy(buf, recv_count);

    MPI_Send(&recv_count, 1 , MPI_INT, dest, 0, mpi_comm);

    MPI_Send(buf, recv_count, MPI_BYTE, dest, 0 , mpi_comm);

    delete[] buf;
    }

//! Wrapper around MPI_Recv that handles any serializable object
template<typename T>
void recv(T& val,const unsigned int src, const MPI_Comm mpi_comm)
    {
    int rank;
    MPI_Comm_rank(mpi_comm, &rank);
    if( rank == static_cast<int>(src) ) //Quick exit if src is dest.
      return;

    int recv_count;

    MPI_Recv(&recv_count, 1, MPI_INT, src, 0 , mpi_comm, MPI_STATUS_IGNORE);

    char *buf = NULL;
    buf = new char[recv_count];

    MPI_Recv(buf, recv_count, MPI_BYTE, src, 0 , mpi_comm, MPI_STATUS_IGNORE);

    // de-serialize
    std::stringstream s(std::string(buf, recv_count), std::ios_base::in | std::ios_base::binary);
    cereal::BinaryInputArchive ar(s);
    ar >> val;

    delete[] buf;
    }


#endif // ENABLE_MPI
#endif // __HOOMD_MPI_H__
