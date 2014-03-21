/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2014 The Regents of
the University of Michigan All rights reserved.

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

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/device/back_inserter.hpp>

#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>

#ifdef SINGLE_PRECISION
//! Define MPI_FLOAT as Scalar MPI data type
const MPI_Datatype MPI_HOOMD_SCALAR = MPI_FLOAT;
#else
//! Define MPI_DOUBLE as Scalar MPI data type
const MPI_Datatype MPI_HOOMD_SCALAR = MPI_DOUBLE;
#endif

namespace boost
   {
    //! Serialization functions for some of our data types
    namespace serialization
        {
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
        }
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
        std::string str;
        boost::iostreams::back_insert_device <std::string> inserter(str);
        boost::iostreams::stream<boost::iostreams::back_insert_device< std::string> > s(inserter);
        boost::archive::binary_oarchive ar(s);

        // serialize object
        ar << val;

        // do not forget to flush stream
        s.flush();

        // copy string to send buffer
        recv_count = str.size();
        buf = new char[recv_count];
        str.copy(buf, recv_count);
        }

    MPI_Bcast(&recv_count, 1, MPI_INT, root, mpi_comm);
    if (rank != (int) root)
        buf = new char[recv_count];

    MPI_Bcast(buf, recv_count, MPI_BYTE, root, mpi_comm);

    // de-serialize
    std::string str(buf,recv_count);
    boost::iostreams::basic_array_source<char> dev(str.data(), str.size());
    boost::iostreams::stream<boost::iostreams::basic_array_source<char> > s(dev);
    boost::archive::binary_iarchive ar(s);

    ar >> val;

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
        std::vector<std::string> str(size);
        unsigned int len = 0;
        for (it = in_values.begin(); it!= in_values.end(); ++it)
            {
            unsigned int idx = it - in_values.begin();
            boost::iostreams::back_insert_device <std::string> inserter(str[idx]);
            boost::iostreams::stream<boost::iostreams::back_insert_device< std::string> > s(inserter);
            boost::archive::binary_oarchive ar(s);

            // serialize object
            ar << *it;
            s.flush();

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
    std::string str(rbuf, recv_count);
    boost::iostreams::basic_array_source<char> dev(str.data(), str.size());
    boost::iostreams::stream<boost::iostreams::basic_array_source<char> > s(dev);
    boost::archive::binary_iarchive ar(s);

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
    std::string str;
    boost::iostreams::back_insert_device <std::string> inserter(str);
    boost::iostreams::stream<boost::iostreams::back_insert_device< std::string> > s(inserter);
    boost::archive::binary_oarchive ar(s);

    ar << in_value;
    s.flush();

    // copy into send buffer
    unsigned int send_count = str.length();
    char *sbuf = new char[send_count];
    str.copy(sbuf, send_count);

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
    MPI_Gatherv(sbuf, send_count, MPI_BYTE, rbuf, recv_counts, displs, MPI_BYTE, root, mpi_comm);

    // on root processor, de-serialize data
    if (rank == (int) root)
        {
        for (unsigned int i = 0; i < out_values.size(); i++)
            {
            std::string str(rbuf + displs[i], recv_counts[i]);
            boost::iostreams::basic_array_source<char> dev(str.data(), str.size());
            boost::iostreams::stream<boost::iostreams::basic_array_source<char> > s(dev);
            boost::archive::binary_iarchive ar(s);

            ar >> out_values[i];
            }

        delete[] displs;
        delete[] recv_counts;
        delete[] rbuf;
        }

    delete[] sbuf;
    }

#endif // ENABLE_MPI
#endif // __HOOMD_MATH_H__
