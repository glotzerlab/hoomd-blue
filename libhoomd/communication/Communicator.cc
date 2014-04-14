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

/*! \file Communicator.cc
    \brief Implements the Communicator class
*/

#ifdef ENABLE_MPI

#include "Communicator.h"
#include "System.h"

#include <boost/bind.hpp>
#include <boost/python.hpp>
#include <algorithm>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

using namespace boost::python;

template<class group_data>
Communicator::GroupCommunicator<group_data>::GroupCommunicator(Communicator& comm, boost::shared_ptr<group_data> gdata)
    : m_comm(comm), m_exec_conf(comm.m_exec_conf), m_gdata(gdata)
    {
    // the size of the bit field must be larger or equal the group size
    assert(sizeof(unsigned int)*8 >= group_data::size);
    }

template<class group_data>
void Communicator::GroupCommunicator<group_data>::migrateGroups(bool incomplete)
    {
    if (m_gdata->getNGlobal())
        {
        if (m_comm.m_prof) m_comm.m_prof->push(m_exec_conf, m_gdata->getName());

        // send map for rank updates
        typedef std::multimap<unsigned int, rank_element_t> map_t;
        map_t send_map;

            {
            ArrayHandle<unsigned int> h_comm_flags(m_comm.m_pdata->getCommFlags(), access_location::host, access_mode::read);
            ArrayHandle<typename group_data::members_t> h_members(m_gdata->getMembersArray(), access_location::host, access_mode::read);
            ArrayHandle<typename group_data::ranks_t> h_group_ranks(m_gdata->getRanksArray(), access_location::host, access_mode::readwrite);
            ArrayHandle<unsigned int> h_group_tag(m_gdata->getTags(), access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_rtag(m_comm.m_pdata->getRTags(), access_location::host, access_mode::read);

            ArrayHandle<unsigned int> h_unique_neighbors(m_comm.m_unique_neighbors, access_location:: host, access_mode::read);

            ArrayHandle<unsigned int> h_cart_ranks(m_comm.m_pdata->getDomainDecomposition()->getCartRanks(), access_location::host, access_mode::read);

            Index3D di = m_comm.m_pdata->getDomainDecomposition()->getDomainIndexer();
            uint3 my_pos = m_comm.m_pdata->getDomainDecomposition()->getGridPos();
            unsigned int my_rank = m_exec_conf->getRank();

            // mark groups whose member ranks need to be updated
            unsigned int n_groups = m_gdata->getN();
            for (unsigned int group_idx = 0; group_idx < n_groups; group_idx++)
                {
                typename group_data::members_t g = h_members.data[group_idx];
                typename group_data::ranks_t r = h_group_ranks.data[group_idx];

                // initialize bit field
                unsigned int mask = 0;

                bool update = false;

                // iterate over group members
                for (unsigned int i = 0; i < group_data::size; i++)
                    {
                    unsigned int tag = g.tag[i];
                    unsigned int pidx = h_rtag.data[tag];

                    if (pidx == NOT_LOCAL)
                        {
                        // if any ptl is non-local, send
                        update = true;
                        }
                    else
                        {
                        if (incomplete)
                            {
                            // initially, update rank information
                            r.idx[i] = my_rank;
                            mask |= (1 << i);
                            }

                        unsigned int flags = h_comm_flags.data[pidx];

                        if (flags)
                            {
                            // particle is sent to a different domain
                            mask |= (1 << i);

                            int ix, iy, iz;
                            ix = iy = iz = 0;

                            if (flags & send_east)
                                ix = 1;
                            else if (flags & send_west)
                                ix = -1;

                            if (flags & send_north)
                                iy = 1;
                            else if (flags & send_south)
                                iy = -1;

                            if (flags & send_up)
                                iz = 1;
                            else if (flags & send_down)
                                iz = -1;

                            int ni = my_pos.x;
                            int nj = my_pos.y;
                            int nk = my_pos.z;

                            ni += ix;
                            if (ni == (int)di.getW())
                                ni = 0;
                            else if (ni < 0)
                                ni += di.getW();

                            nj += iy;
                            if (nj == (int) di.getH())
                                nj = 0;
                            else if (nj < 0)
                                nj += di.getH();

                            nk += iz;
                            if (nk == (int) di.getD())
                                nk = 0;
                            else if (nk < 0)
                                nk += di.getD();

                            // update ranks
                            r.idx[i] = h_cart_ranks.data[di(ni,nj,nk)];

                            update = true;
                            }
                        }
                    } // end loop over group members

                h_group_ranks.data[group_idx] = r;

                // a group that is purely local is not sent
                if (!update) mask = 0;

                if (mask)
                    {
                    // add to sorted output buffer
                    rank_element_t el;
                    el.ranks = r;
                    el.mask = mask;
                    el.tag = h_group_tag.data[group_idx];
                    if (incomplete)
                        // in initialization, send to all neighbors
                        for(unsigned int ineigh = 0; ineigh < m_comm.m_n_unique_neigh; ineigh++)
                            send_map.insert(std::make_pair(h_unique_neighbors.data[ineigh], el));
                    else
                        // send to other ranks owning the bonded group
                        for (unsigned int j = 0; j < group_data::size; ++j)
                            {
                            unsigned int rank = r.idx[j];
                            bool rank_updated = mask & (1 << j);
                            // send out to ranks different from ours
                            if (rank != my_rank && !rank_updated)
                                send_map.insert(std::make_pair(rank, el));
                            }
                    }
                } // end loop over groups
            } // end ArrayHandle scope

        // clear send buffer
        m_ranks_sendbuf.clear();

            {
            // output send data sorted by rank
            for (typename map_t::iterator it = send_map.begin(); it != send_map.end(); ++it)
                {
                m_ranks_sendbuf.push_back(it->second);
                }

            ArrayHandle<unsigned int> h_unique_neighbors(m_comm.m_unique_neighbors, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_begin(m_comm.m_begin, access_location::host, access_mode::overwrite);
            ArrayHandle<unsigned int> h_end(m_comm.m_end, access_location::host, access_mode::overwrite);

            // Find start and end indices
            for (unsigned int i = 0; i < m_comm.m_n_unique_neigh; ++i)
                {
                typename map_t::iterator lower = send_map.lower_bound(h_unique_neighbors.data[i]);
                typename map_t::iterator upper = send_map.upper_bound(h_unique_neighbors.data[i]);
                h_begin.data[i] = std::distance(send_map.begin(),lower);
                h_end.data[i] = std::distance(send_map.begin(),upper);
                }
            }

        /*
         * communicate rank information (phase 1)
         */
        unsigned int n_send_groups[m_comm.m_n_unique_neigh];
        unsigned int n_recv_groups[m_comm.m_n_unique_neigh];
        unsigned int offs[m_comm.m_n_unique_neigh];
        unsigned int n_recv_tot = 0;

            {
            ArrayHandle<unsigned int> h_begin(m_comm.m_begin, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_end(m_comm.m_end, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_unique_neighbors(m_comm.m_unique_neighbors, access_location::host, access_mode::read);

            unsigned int send_bytes = 0;
            unsigned int recv_bytes = 0;
            if (m_comm.m_prof) m_comm.m_prof->push("MPI send/recv");

            // compute send counts
            for (unsigned int ineigh = 0; ineigh < m_comm.m_n_unique_neigh; ineigh++)
                n_send_groups[ineigh] = h_end.data[ineigh] - h_begin.data[ineigh];

            MPI_Request req[2*m_comm.m_n_unique_neigh];
            MPI_Status stat[2*m_comm.m_n_unique_neigh];

            unsigned int nreq = 0;

            // loop over neighbors
            for (unsigned int ineigh = 0; ineigh < m_comm.m_n_unique_neigh; ineigh++)
                {
                // rank of neighbor processor
                unsigned int neighbor = h_unique_neighbors.data[ineigh];

                MPI_Isend(&n_send_groups[ineigh], 1, MPI_UNSIGNED, neighbor, 0, m_comm.m_mpi_comm, & req[nreq++]);
                MPI_Irecv(&n_recv_groups[ineigh], 1, MPI_UNSIGNED, neighbor, 0, m_comm.m_mpi_comm, & req[nreq++]);
                send_bytes += sizeof(unsigned int);
                recv_bytes += sizeof(unsigned int);
                } // end neighbor loop

            MPI_Waitall(nreq, req, stat);

            // sum up receive counts
            for (unsigned int ineigh = 0; ineigh < m_comm.m_n_unique_neigh; ineigh++)
                {
                if (ineigh == 0)
                    offs[ineigh] = 0;
                else
                    offs[ineigh] = offs[ineigh-1] + n_recv_groups[ineigh-1];

                n_recv_tot += n_recv_groups[ineigh];
                }

            if (m_comm.m_prof) m_comm.m_prof->pop(0,send_bytes+recv_bytes);
            }

        // Resize receive buffer
        m_ranks_recvbuf.resize(n_recv_tot);

            {
            ArrayHandle<unsigned int> h_begin(m_comm.m_begin, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_end(m_comm.m_end, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_unique_neighbors(m_comm.m_unique_neighbors, access_location::host, access_mode::read);

            if (m_comm.m_prof) m_comm.m_prof->push("MPI send/recv");

            std::vector<MPI_Request> reqs;
            MPI_Request req;

            unsigned int send_bytes = 0;
            unsigned int recv_bytes = 0;

            // loop over neighbors
            for (unsigned int ineigh = 0; ineigh < m_comm.m_n_unique_neigh; ineigh++)
                {
                // rank of neighbor processor
                unsigned int neighbor = h_unique_neighbors.data[ineigh];

                // exchange particle data
                if (n_send_groups[ineigh])
                    {
                    MPI_Isend(&m_ranks_sendbuf.front()+h_begin.data[ineigh],
                        n_send_groups[ineigh]*sizeof(rank_element_t),
                        MPI_BYTE,
                        neighbor,
                        1,
                        m_comm.m_mpi_comm,
                        &req);
                    reqs.push_back(req);
                    }
                send_bytes+= n_send_groups[ineigh]*sizeof(rank_element_t);

                if (n_recv_groups[ineigh])
                    {
                    MPI_Irecv(&m_ranks_recvbuf.front()+offs[ineigh],
                        n_recv_groups[ineigh]*sizeof(rank_element_t),
                        MPI_BYTE,
                        neighbor,
                        1,
                        m_comm.m_mpi_comm,
                        &req);
                    reqs.push_back(req);
                    }
                recv_bytes += n_recv_groups[ineigh]*sizeof(rank_element_t);
                }

            std::vector<MPI_Status> stats(reqs.size());
            MPI_Waitall(reqs.size(), &reqs.front(), &stats.front());

            if (m_comm.m_prof) m_comm.m_prof->pop(0,send_bytes+recv_bytes);
            }

            {
            // access receive buffers
            ArrayHandle<typename group_data::ranks_t> h_group_ranks(m_gdata->getRanksArray(), access_location::host, access_mode::readwrite);
            ArrayHandle<unsigned int> h_group_rtag(m_gdata->getRTags(), access_location::host, access_mode::read);

            for (unsigned int recv_idx = 0; recv_idx < n_recv_tot; ++recv_idx)
                {
                rank_element_t el = m_ranks_recvbuf[recv_idx];
                unsigned int tag = el.tag;
                unsigned int gidx = h_group_rtag.data[tag];

                if (gidx != GROUP_NOT_LOCAL)
                    {
                    typename group_data::ranks_t new_ranks = el.ranks;
                    unsigned int mask = el.mask;

                    for (unsigned int i = 0; i < group_data::size; ++i)
                        {
                        bool update = mask & (1 << i);

                        if (update)
                            h_group_ranks.data[gidx].idx[i] = new_ranks.idx[i];
                        }
                    }
                }
            }

        // send map for groups
        typedef std::multimap<unsigned int, group_element_t> group_map_t;
        group_map_t group_send_map;

            {
            ArrayHandle<typename group_data::members_t> h_groups(m_gdata->getMembersArray(), access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_group_type(m_gdata->getTypesArray(), access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_group_tag(m_gdata->getTags(), access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_group_rtag(m_gdata->getRTags(), access_location::host, access_mode::readwrite);
            ArrayHandle<typename group_data::ranks_t> h_group_ranks(m_gdata->getRanksArray(), access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_rtag(m_comm.m_pdata->getRTags(), access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_comm_flags(m_comm.m_pdata->getCommFlags(), access_location::host, access_mode::read);

            unsigned int ngroups = m_gdata->getN();

            for (unsigned int group_idx = 0; group_idx < ngroups; group_idx++)
                {
                unsigned int mask = 0;

                typename group_data::members_t members = h_groups.data[group_idx];

                bool send = false;
                for (unsigned int i = 0; i < group_data::size; ++i)
                    {
                    unsigned int tag = members.tag[i];
                    unsigned int pidx = h_rtag.data[tag];

                    if (pidx != NOT_LOCAL && h_comm_flags.data[pidx])
                        {
                        mask |= (1 << i);
                        send = true;
                        }
                    }

                if (send)
                    {
                    // insert into send map
                    typename group_data::packed_t el;
                    el.tags = h_groups.data[group_idx];
                    el.type = h_group_type.data[group_idx];
                    el.group_tag = h_group_tag.data[group_idx];
                    el.ranks = h_group_ranks.data[group_idx];

                    for (unsigned int i = 0; i < group_data::size; ++i)
                        // are we sending to this rank?
                        if (mask & (1 << i))
                            group_send_map.insert(std::make_pair(el.ranks.idx[i], el));

                    // does this group still have local members
                    bool is_local = false;

                    for (unsigned int i = 0; i < group_data::size; ++i)
                        {
                        unsigned int tag = members.tag[i];
                        unsigned int pidx = h_rtag.data[tag];

                        if (pidx != NOT_LOCAL && !h_comm_flags.data[pidx])
                            is_local = true;
                        }

                    // if group is no longer local, flag for removal
                    if (!is_local)
                        h_group_rtag.data[el.group_tag] = GROUP_NOT_LOCAL;
                    }
                } // end loop over groups
            }

        unsigned int new_ngroups;
            {
            ArrayHandle<typename group_data::members_t> h_groups(m_gdata->getMembersArray(), access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_group_type(m_gdata->getTypesArray(), access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_group_tag(m_gdata->getTags(), access_location::host, access_mode::read);
            ArrayHandle<typename group_data::ranks_t> h_group_ranks(m_gdata->getRanksArray(), access_location::host, access_mode::read);

            // access alternate arrays to write to
            ArrayHandle<typename group_data::members_t> h_groups_alt(m_gdata->getAltMembersArray(), access_location::host, access_mode::overwrite);
            ArrayHandle<unsigned int> h_group_type_alt(m_gdata->getAltTypesArray(), access_location::host, access_mode::overwrite);
            ArrayHandle<unsigned int> h_group_tag_alt(m_gdata->getAltTags(), access_location::host, access_mode::overwrite);
            ArrayHandle<typename group_data::ranks_t> h_group_ranks_alt(m_gdata->getAltRanksArray(), access_location::host, access_mode::overwrite);

            // access rtags
            ArrayHandle<unsigned int> h_group_rtag(m_gdata->getRTags(), access_location::host, access_mode::readwrite);

            unsigned int ngroups = m_gdata->getN();
            unsigned int n = 0;
            for (unsigned int group_idx = 0; group_idx < ngroups; group_idx++)
                {
                unsigned int group_tag = h_group_tag.data[group_idx];
                bool keep = h_group_rtag.data[group_tag] != GROUP_NOT_LOCAL;

                if (keep)
                    {
                    h_groups_alt.data[n] = h_groups.data[group_idx];
                    h_group_type_alt.data[n] = h_group_type.data[group_idx];
                    h_group_tag_alt.data[n] = group_tag;
                    h_group_ranks_alt.data[n] = h_group_ranks.data[group_idx];

                    // rebuild rtags
                    h_group_rtag.data[group_tag] = n++;
                    }
                }

                new_ngroups = n;
            }

        // resize alternate arrays to number of groups
        GPUVector<typename group_data::members_t>& alt_groups_array = m_gdata->getAltMembersArray();
        GPUVector<unsigned int>& alt_group_type_array = m_gdata->getAltTypesArray();
        GPUVector<unsigned int>& alt_group_tag_array = m_gdata->getAltTags();
        GPUVector<typename group_data::ranks_t>& alt_group_ranks_array = m_gdata->getAltRanksArray();

        assert(new_ngroups <= m_gdata->getN());
        alt_groups_array.resize(new_ngroups);
        alt_group_type_array.resize(new_ngroups);
        alt_group_tag_array.resize(new_ngroups);
        alt_group_ranks_array.resize(new_ngroups);

        // make alternate arrays current
        m_gdata->swapMemberArrays();
        m_gdata->swapTypeArrays();
        m_gdata->swapTagArrays();
        m_gdata->swapRankArrays();

        assert(m_gdata->getN() == new_ngroups);

        // reset send buf
        m_groups_sendbuf.clear();

        // output groups to send buffer in rank-sorted order
        for (typename group_map_t::iterator it = group_send_map.begin(); it != group_send_map.end(); ++it)
            m_groups_sendbuf.push_back(it->second);

            {
            ArrayHandle<unsigned int> h_unique_neighbors(m_comm.m_unique_neighbors, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_begin(m_comm.m_begin, access_location::host, access_mode::overwrite);
            ArrayHandle<unsigned int> h_end(m_comm.m_end, access_location::host, access_mode::overwrite);

            // Find start and end indices
            for (unsigned int i = 0; i < m_comm.m_n_unique_neigh; ++i)
                {
                typename group_map_t::iterator lower = group_send_map.lower_bound(h_unique_neighbors.data[i]);
                typename group_map_t::iterator upper = group_send_map.upper_bound(h_unique_neighbors.data[i]);
                h_begin.data[i] = std::distance(group_send_map.begin(),lower);
                h_end.data[i] = std::distance(group_send_map.begin(),upper);
                }
            }

        /*
         * communicate groups (phase 2)
         */

       n_recv_tot = 0;
            {
            ArrayHandle<unsigned int> h_begin(m_comm.m_begin, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_end(m_comm.m_end, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_unique_neighbors(m_comm.m_unique_neighbors, access_location::host, access_mode::read);

            unsigned int send_bytes = 0;
            unsigned int recv_bytes = 0;
            if (m_comm.m_prof) m_comm.m_prof->push("MPI send/recv");

            // compute send counts
            for (unsigned int ineigh = 0; ineigh < m_comm.m_n_unique_neigh; ineigh++)
                n_send_groups[ineigh] = h_end.data[ineigh] - h_begin.data[ineigh];

            MPI_Request req[2*m_comm.m_n_unique_neigh];
            MPI_Status stat[2*m_comm.m_n_unique_neigh];

            unsigned int nreq = 0;

            // loop over neighbors
            for (unsigned int ineigh = 0; ineigh < m_comm.m_n_unique_neigh; ineigh++)
                {
                // rank of neighbor processor
                unsigned int neighbor = h_unique_neighbors.data[ineigh];

                MPI_Isend(&n_send_groups[ineigh], 1, MPI_UNSIGNED, neighbor, 0, m_comm.m_mpi_comm, & req[nreq++]);
                MPI_Irecv(&n_recv_groups[ineigh], 1, MPI_UNSIGNED, neighbor, 0, m_comm.m_mpi_comm, & req[nreq++]);
                send_bytes += sizeof(unsigned int);
                recv_bytes += sizeof(unsigned int);
                } // end neighbor loop

            MPI_Waitall(nreq, req, stat);

            // sum up receive counts
            for (unsigned int ineigh = 0; ineigh < m_comm.m_n_unique_neigh; ineigh++)
                {
                if (ineigh == 0)
                    offs[ineigh] = 0;
                else
                    offs[ineigh] = offs[ineigh-1] + n_recv_groups[ineigh-1];

                n_recv_tot += n_recv_groups[ineigh];
                }

            if (m_comm.m_prof) m_comm.m_prof->pop(0,send_bytes+recv_bytes);
            }

        // Resize receive buffer
        m_groups_recvbuf.resize(n_recv_tot);

            {
            ArrayHandle<unsigned int> h_begin(m_comm.m_begin, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_end(m_comm.m_end, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_unique_neighbors(m_comm.m_unique_neighbors, access_location::host, access_mode::read);

            if (m_comm.m_prof) m_comm.m_prof->push("MPI send/recv");

            std::vector<MPI_Request> reqs;
            MPI_Request req;

            unsigned int send_bytes = 0;
            unsigned int recv_bytes = 0;

            // loop over neighbors
            for (unsigned int ineigh = 0; ineigh < m_comm.m_n_unique_neigh; ineigh++)
                {
                // rank of neighbor processor
                unsigned int neighbor = h_unique_neighbors.data[ineigh];

                // exchange particle data
                if (n_send_groups[ineigh])
                    {
                    MPI_Isend(&m_groups_sendbuf.front()+h_begin.data[ineigh],
                        n_send_groups[ineigh]*sizeof(group_element_t),
                        MPI_BYTE,
                        neighbor,
                        1,
                        m_comm.m_mpi_comm,
                        &req);
                    reqs.push_back(req);
                    }
                send_bytes+= n_send_groups[ineigh]*sizeof(group_element_t);

                if (n_recv_groups[ineigh])
                    {
                    MPI_Irecv(&m_groups_recvbuf.front()+offs[ineigh],
                        n_recv_groups[ineigh]*sizeof(group_element_t),
                        MPI_BYTE,
                        neighbor,
                        1,
                        m_comm.m_mpi_comm,
                        &req);
                    reqs.push_back(req);
                    }
                recv_bytes += n_recv_groups[ineigh]*sizeof(group_element_t);
                }

            std::vector<MPI_Status> stats(reqs.size());
            MPI_Waitall(reqs.size(), &reqs.front(), &stats.front());

            if (m_comm.m_prof) m_comm.m_prof->pop(0,send_bytes+recv_bytes);
            }

        // use a std::map, i.e. single-key, to filter out duplicate groups in input buffer
        typedef std::map<unsigned int, group_element_t> recv_map_t;
        recv_map_t recv_map;

        for (unsigned int recv_idx = 0; recv_idx < n_recv_tot; recv_idx++)
            {
            group_element_t el = m_groups_recvbuf[recv_idx];
            unsigned int tag= el.group_tag;
            recv_map.insert(std::make_pair(tag, el));
            }

        unsigned int n_recv_unique = recv_map.size();

        unsigned int old_ngroups = m_gdata->getN();
        new_ngroups = old_ngroups + n_recv_unique;

        // resize group arrays to accomodate additional groups (there can still be duplicates with local groups)
        GPUVector<typename group_data::members_t>& groups_array = m_gdata->getMembersArray();
        GPUVector<unsigned int>& group_type_array = m_gdata->getTypesArray();
        GPUVector<unsigned int>& group_tag_array = m_gdata->getTags();
        GPUVector<typename group_data::ranks_t>& group_ranks_array = m_gdata->getRanksArray();

        groups_array.resize(new_ngroups);
        group_type_array.resize(new_ngroups);
        group_tag_array.resize(new_ngroups);
        group_ranks_array.resize(new_ngroups);

            {
            ArrayHandle<unsigned int> h_group_rtag(m_gdata->getRTags(), access_location::host, access_mode::readwrite);
            ArrayHandle<typename group_data::members_t> h_groups(groups_array, access_location::host, access_mode::readwrite);
            ArrayHandle<unsigned int> h_group_type(group_type_array, access_location::host, access_mode::readwrite);
            ArrayHandle<unsigned int> h_group_tag(group_tag_array, access_location::host, access_mode::readwrite);
            ArrayHandle<typename group_data::ranks_t> h_group_ranks(group_ranks_array, access_location::host, access_mode::readwrite);

            // add non-duplicate groups to group data
            unsigned int add_idx = old_ngroups;
            for (typename recv_map_t::iterator it = recv_map.begin(); it != recv_map.end(); ++it)
                {
                typename group_data::packed_t el = it->second;

                unsigned int tag = el.group_tag;
                unsigned int group_rtag = h_group_rtag.data[tag];

                if (group_rtag == GROUP_NOT_LOCAL)
                    {
                    h_groups.data[add_idx] = el.tags;
                    h_group_type.data[add_idx] = el.type;
                    h_group_tag.data[add_idx] = tag;
                    h_group_ranks.data[add_idx] = el.ranks;

                    // update reverse-lookup table
                    h_group_rtag.data[tag] = add_idx++;
                    }
                }
            new_ngroups = add_idx;
            }

        // resize arrays to final size
        groups_array.resize(new_ngroups);
        group_type_array.resize(new_ngroups);
        group_tag_array.resize(new_ngroups);
        group_ranks_array.resize(new_ngroups);

        // indicate that group table has changed
        m_gdata->setDirty();

        if (m_comm.m_prof) m_comm.m_prof->pop();
        }
    }

//! Mark ghost particles
template<class group_data>
void Communicator::GroupCommunicator<group_data>::markGhostParticles(
    const GPUArray<unsigned int>& plans,
    unsigned int mask)
    {
    if (m_gdata->getNGlobal())
        {
        ArrayHandle<typename group_data::members_t> h_groups(m_gdata->getMembersArray(), access_location::host, access_mode::read);
        ArrayHandle<typename group_data::ranks_t> h_group_ranks(m_gdata->getRanksArray(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_rtag(m_comm.m_pdata->getRTags(), access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_postype(m_comm.m_pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_plan(plans, access_location::host, access_mode::readwrite);

        ArrayHandle<unsigned int> h_cart_ranks_inv(m_comm.m_pdata->getDomainDecomposition()->getInverseCartRanks(),
            access_location::host, access_mode::read);

        Index3D di = m_comm.m_pdata->getDomainDecomposition()->getDomainIndexer();
        unsigned int my_rank = m_exec_conf->getRank();
        uint3 my_pos = m_comm.m_pdata->getDomainDecomposition()->getGridPos();
        const BoxDim& box = m_comm.m_pdata->getBox();

        unsigned int ngroups = m_gdata->getN();

        for (unsigned int group_idx = 0; group_idx < ngroups; ++group_idx)
            {
            typename group_data::members_t g = h_groups.data[group_idx];
            typename group_data::ranks_t r = h_group_ranks.data[group_idx];

            // iterate over group members
            for (unsigned int i = 0; i < group_data::size; ++i)
                {
                unsigned int rank = r.idx[i];

                if (rank != my_rank)
                    {
                    // incomplete group

                    // send group to neighbor rank stored for that member
                    uint3 neigh_pos = di.getTriple(h_cart_ranks_inv.data[rank]);

                    // only neighbors are considered for communication
                    unsigned int flags = 0;
                    if (neigh_pos.x == my_pos.x + 1 || (my_pos.x == di.getW()-1 && neigh_pos.x == 0))
                        flags |= send_east;
                    if (neigh_pos.x == my_pos.x - 1 || (my_pos.x == 0 && neigh_pos.x == di.getW()-1))
                        flags |= send_west;
                    if (neigh_pos.y == my_pos.y + 1 || (my_pos.y == di.getH()-1 && neigh_pos.y == 0))
                        flags |= send_north;
                    if (neigh_pos.y == my_pos.y - 1 || (my_pos.y == 0 && neigh_pos.y == di.getH()-1))
                        flags |= send_south;
                    if (neigh_pos.z == my_pos.z + 1 || (my_pos.z == di.getD()-1 && neigh_pos.z == 0))
                        flags |= send_up;
                    if (neigh_pos.z == my_pos.z - 1 || (my_pos.z == 0 && neigh_pos.z == di.getD()-1))
                        flags |= send_down;

                    flags &= mask;

                    // Send all local members of the group to this neighbor
                    for (unsigned int j = 0; j < group_data::size; ++j)
                        {
                        unsigned int tag_j = g.tag[j];
                        unsigned int rtag_j = h_rtag.data[tag_j];

                        if (rtag_j != NOT_LOCAL)
                            {
                            // disambiguate between positive and negative directions
                            // based on position (this is necessary for boundary conditions
                            // to be applied correctly)
                            if (flags & send_east && flags & send_west)
                                {
                                Scalar4 postype = h_postype.data[rtag_j];
                                Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
                                Scalar3 f = box.makeFraction(pos);
                                // remove one of the flags
                                flags &= ~(f.x > Scalar(0.5) ? send_west : send_east);
                                }
                            if (flags & send_north && flags & send_south)
                                {
                                Scalar4 postype = h_postype.data[rtag_j];
                                Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
                                Scalar3 f = box.makeFraction(pos);
                                // remove one of the flags
                                flags &= ~(f.y > Scalar(0.5) ? send_south : send_north);
                                }
                            if (flags & send_up && flags & send_down)
                                {
                                Scalar4 postype = h_postype.data[rtag_j];
                                Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
                                Scalar3 f = box.makeFraction(pos);
                                // remove one of the flags
                                flags &= ~(f.z > Scalar(0.5) ? send_down : send_up);
                                }

                            h_plan.data[rtag_j] |= flags;
                            }
                        } // end inner loop over group members
                    }
                } // end outer loop over group members
            } // end loop over groups
        }
    }

//! Constructor
Communicator::Communicator(boost::shared_ptr<SystemDefinition> sysdef,
                           boost::shared_ptr<DomainDecomposition> decomposition)
          : m_sysdef(sysdef),
            m_pdata(sysdef->getParticleData()),
            m_exec_conf(m_pdata->getExecConf()),
            m_mpi_comm(m_exec_conf->getMPICommunicator()),
            m_decomposition(decomposition),
            m_is_communicating(false),
            m_force_migrate(false),
            m_nneigh(0),
            m_n_unique_neigh(0),
            m_pos_copybuf(m_exec_conf),
            m_charge_copybuf(m_exec_conf),
            m_diameter_copybuf(m_exec_conf),
            m_velocity_copybuf(m_exec_conf),
            m_orientation_copybuf(m_exec_conf),
            m_plan_copybuf(m_exec_conf),
            m_tag_copybuf(m_exec_conf),
            m_r_ghost(Scalar(0.0)),
            m_r_buff(Scalar(0.0)),
            m_plan(m_exec_conf),
            m_last_flags(0),
            m_comm_pending(false),
            m_bond_comm(*this, m_sysdef->getBondData()),
            m_angle_comm(*this, m_sysdef->getAngleData()),
            m_dihedral_comm(*this, m_sysdef->getDihedralData()),
            m_improper_comm(*this, m_sysdef->getImproperData()),
            m_is_first_step(true)
    {
    // initialize array of neighbor processor ids
    assert(m_mpi_comm);
    assert(m_decomposition);

    m_exec_conf->msg->notice(5) << "Constructing Communicator" << endl;

    for (unsigned int dir = 0; dir < 6; dir ++)
        {
        m_is_at_boundary[dir] = m_decomposition->isAtBoundary(dir) ? 1 : 0;
        }

    for (unsigned int dir = 0; dir < 6; dir ++)
        {
        GPUVector<unsigned int> copy_ghosts(m_exec_conf);
        m_copy_ghosts[dir].swap(copy_ghosts);
        m_num_copy_ghosts[dir] = 0;
        m_num_recv_ghosts[dir] = 0;
        }

    // connect to particle sort signal
    m_sort_connection = m_pdata->connectParticleSort(boost::bind(&Communicator::forceMigrate, this));

    /*
     * Bonded group communication
     */
    m_bonds_changed = true;
    m_bond_connection = m_sysdef->getBondData()->connectGroupNumChange(boost::bind(&Communicator::setBondsChanged, this));

    m_angles_changed = true;
    m_angle_connection = m_sysdef->getAngleData()->connectGroupNumChange(boost::bind(&Communicator::setAnglesChanged, this));

    m_dihedrals_changed = true;
    m_dihedral_connection = m_sysdef->getDihedralData()->connectGroupNumChange(boost::bind(&Communicator::setDihedralsChanged, this));

    m_impropers_changed = true;
    m_improper_connection = m_sysdef->getImproperData()->connectGroupNumChange(boost::bind(&Communicator::setImpropersChanged, this));

    // allocate memory
    GPUArray<unsigned int> neighbors(NEIGH_MAX,m_exec_conf);
    m_neighbors.swap(neighbors);

    GPUArray<unsigned int> unique_neighbors(NEIGH_MAX,m_exec_conf);
    m_unique_neighbors.swap(unique_neighbors);

    // neighbor masks
    GPUArray<unsigned int> adj_mask(NEIGH_MAX, m_exec_conf);
    m_adj_mask.swap(adj_mask);

    GPUArray<unsigned int> begin(NEIGH_MAX,m_exec_conf);
    m_begin.swap(begin);

    GPUArray<unsigned int> end(NEIGH_MAX,m_exec_conf);
    m_end.swap(end);

    initializeNeighborArrays();

    #ifdef ENABLE_CUDA
    if (m_exec_conf->isCUDAEnabled())
        {
        // set up autotuners to determine whether to use mapped memory (boolean values)
        std::vector<unsigned int> valid_params(2);
        valid_params[0] = 0; valid_params[1]  = 1;

        // use a sufficiently long measurement period to average over
        unsigned int nsteps = 100;
        m_tuner_precompute.reset(new Autotuner(valid_params, nsteps, 100000, "comm_precompute", this->m_exec_conf));

        // average execution times instead of median
        m_tuner_precompute->setAverage(true);

        // we require syncing for aligned execution streams
        m_tuner_precompute->setSync(true);
        }
    #endif
    }

//! Destructor
Communicator::~Communicator()
    {
    m_exec_conf->msg->notice(5) << "Destroying Communicator";
    m_sort_connection.disconnect();
    m_bond_connection.disconnect();
    m_angle_connection.disconnect();
    m_dihedral_connection.disconnect();
    m_improper_connection.disconnect();
    }

void Communicator::initializeNeighborArrays()
    {
    Index3D di= m_decomposition->getDomainIndexer();

    uint3 mypos = m_decomposition->getGridPos();
    int l = mypos.x;
    int m = mypos.y;
    int n = mypos.z;

    ArrayHandle<unsigned int> h_neighbors(m_neighbors, access_location::host, access_mode::overwrite);
    ArrayHandle<unsigned int> h_adj_mask(m_adj_mask, access_location::host, access_mode::overwrite);
    ArrayHandle<unsigned int> h_cart_ranks(m_decomposition->getCartRanks(), access_location::host, access_mode::read);

    m_nneigh = 0;

    // loop over neighbors
    for (int ix=-1; ix <= 1; ix++)
        {
        int i = ix + l;
        if (i == (int)di.getW())
            i = 0;
        else if (i < 0)
            i += di.getW();

        // only if communicating along x-direction
        if (ix && di.getW() == 1) continue;

        for (int iy=-1; iy <= 1; iy++)
            {
            int j = iy + m;

            if (j == (int)di.getH())
                j = 0;
            else if (j < 0)
                j += di.getH();

            // only if communicating along y-direction
            if (iy && di.getH() == 1) continue;

            for (int iz=-1; iz <= 1; iz++)
                {
                int k = iz + n;

                if (k == (int)di.getD())
                    k = 0;
                else if (k < 0)
                    k += di.getD();

                // only if communicating along z-direction
                if (iz && di.getD() == 1) continue;

                // exclude ourselves
                if (!ix && !iy && !iz) continue;

                unsigned int dir = ((iz+1)*3+(iy+1))*3+(ix + 1);
                unsigned int mask = 1 << dir;

                unsigned int neighbor = h_cart_ranks.data[di(i,j,k)];
                h_neighbors.data[m_nneigh] = neighbor;
                h_adj_mask.data[m_nneigh] = mask;
                m_nneigh++;
                }
            }
        }

    ArrayHandle<unsigned int> h_unique_neighbors(m_unique_neighbors, access_location::host, access_mode::overwrite);

    // filter neighbors, combining adjacency masks
    std::map<unsigned int, unsigned int> neigh_map;
    for (unsigned int i = 0; i < m_nneigh; ++i)
        {
        unsigned int m = 0;

        for (unsigned int j = 0; j < m_nneigh; ++j)
            if (h_neighbors.data[j] == h_neighbors.data[i])
                m |= h_adj_mask.data[j];

        // std::map inserts the same key only once
        neigh_map.insert(std::make_pair(h_neighbors.data[i], m));
        }

    m_n_unique_neigh = neigh_map.size();

    n = 0;
    for (std::map<unsigned int, unsigned int>::iterator it = neigh_map.begin(); it != neigh_map.end(); ++it)
        {
        h_unique_neighbors.data[n] = it->first;
        h_adj_mask.data[n] = it->second;
        n++;
        }
    }

//! Interface to the communication methods.
void Communicator::communicate(unsigned int timestep)
    {
    // Guard to prevent recursive triggering of migration
    m_is_communicating = true;

    // update ghost communication flags
    m_flags = m_requested_flags(timestep);

    /*
     * Always update ghosts - even if not required, i.e. if the neighbor list
     * needs to be rebuilt. Exceptions are when we have not previously
     * exchanged ghosts, i.e. on the first step or when ghosts have
     * potentially been invalidated, i.e. upon reordering of particles.
     */

    bool update = !m_is_first_step && !m_force_migrate;

    bool precompute = m_tuner_precompute ? m_tuner_precompute->getParam() : false;

    update &= precompute;

    if (m_tuner_precompute) m_tuner_precompute->begin();

    if (update)
        beginUpdateGhosts(timestep);

    // call computation that can be overlapped with communication
    m_local_compute_callbacks(timestep);

    if (update)
        finishUpdateGhosts(timestep);

    if (precompute && update)
        {
        // call subscribers *before* MPI synchronization, but after ghost update
        m_compute_callbacks(timestep);
        }

    // other functions involving syncing
    m_comm_callbacks(timestep);

    // distance check (synchronizes the GPU execution stream)
    bool migrate = m_force_migrate || m_migrate_requests(timestep) || m_is_first_step;

    if (!precompute && !migrate)
        {
        // *after* synchronization, but only if particles do not migrate
        beginUpdateGhosts(timestep);
        finishUpdateGhosts(timestep);

        m_compute_callbacks(timestep);
        }

    if (m_tuner_precompute) m_tuner_precompute->end();

    // Check if migration of particles is requested
    if (migrate)
        {
        m_force_migrate = false;
        m_is_first_step = false;

        // If so, migrate atoms
        migrateParticles();

        // Construct ghost send lists, exchange ghost atom data
        exchangeGhosts();
        }

    m_is_communicating = false;
    }

//! Transfer particles between neighboring domains
void Communicator::migrateParticles()
    {
    if (m_prof)
        m_prof->push("comm_migrate");

    m_exec_conf->msg->notice(7) << "Communicator: migrate particles" << std::endl;

        {
        // wipe out reverse-lookup tag -> idx for old ghost atoms
        ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::readwrite);
        for (unsigned int i = 0; i < m_pdata->getNGhosts(); i++)
            {
            unsigned int idx = m_pdata->getN() + i;
            h_rtag.data[h_tag.data[idx]] = NOT_LOCAL;
            }
        }

    //  reset ghost particle number
    m_pdata->removeAllGhostParticles();

    // get box dimensions
    const BoxDim& box = m_pdata->getBox();

    // determine local particles that are to be sent to neighboring processors and fill send buffer
    for (unsigned int dir=0; dir < 6; dir++)
        {
        if (! isCommunicating(dir) ) continue;

            {
            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_comm_flag(m_pdata->getCommFlags(), access_location::host, access_mode::readwrite);

            // mark all particles which have left the box for sending (rtag=NOT_LOCAL)
            unsigned int N = m_pdata->getN();

            for (unsigned int idx = 0; idx < N; ++idx)
                {
                const Scalar4& postype = h_pos.data[idx];
                Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
                Scalar3 f = box.makeFraction(pos);

                // return true if the particle stays leaves the box
                unsigned int flags = 0;
                if (dir == 0 && f.x >= Scalar(1.0)) flags |= send_east;
                else if (dir == 1 && f.x < Scalar(0.0)) flags |= send_west;
                else if (dir == 2 && f.y >= Scalar(1.0)) flags |= send_north;
                else if (dir == 3 && f.y < Scalar(0.0)) flags |= send_south;
                else if (dir == 4 && f.z >= Scalar(1.0)) flags |= send_up;
                else if (dir == 5 && f.z < Scalar(0.0)) flags |= send_down;

                h_comm_flag.data[idx] = flags;
                }
            }

        /*
         * Bonded group communication, determine groups to be sent
         */
        // Bonds
        m_bond_comm.migrateGroups(m_bonds_changed);
        m_bonds_changed = false;

        // Angles
        m_angle_comm.migrateGroups(m_angles_changed);
        m_angles_changed = false;

        // Dihedrals
        m_dihedral_comm.migrateGroups(m_dihedrals_changed);
        m_dihedrals_changed = false;

        // Dihedrals
        m_improper_comm.migrateGroups(m_impropers_changed);
        m_impropers_changed = false;

        // fill send buffer
        std::vector<unsigned int> comm_flag_out; // not currently used
        m_pdata->removeParticles(m_sendbuf, comm_flag_out);

        unsigned int send_neighbor = m_decomposition->getNeighborRank(dir);

        // we receive from the direction opposite to the one we send to
        unsigned int recv_neighbor;
        if (dir % 2 == 0)
            recv_neighbor = m_decomposition->getNeighborRank(dir+1);
        else
            recv_neighbor = m_decomposition->getNeighborRank(dir-1);

        if (m_prof)
            m_prof->push("MPI send/recv");

        unsigned int n_recv_ptls;

        // communicate size of the message that will contain the particle data
        MPI_Request reqs[2];
        MPI_Status status[2];

        unsigned int n_send_ptls = m_sendbuf.size();

        MPI_Isend(&n_send_ptls, 1, MPI_UNSIGNED, send_neighbor, 0, m_mpi_comm, & reqs[0]);
        MPI_Irecv(&n_recv_ptls, 1, MPI_UNSIGNED, recv_neighbor, 0, m_mpi_comm, & reqs[1]);
        MPI_Waitall(2, reqs, status);

        // Resize receive buffer
        m_recvbuf.resize(n_recv_ptls);

        // exchange particle data
        MPI_Isend(&m_sendbuf.front(), n_send_ptls*sizeof(pdata_element), MPI_BYTE, send_neighbor, 1, m_mpi_comm, & reqs[0]);
        MPI_Irecv(&m_recvbuf.front(), n_recv_ptls*sizeof(pdata_element), MPI_BYTE, recv_neighbor, 1, m_mpi_comm, & reqs[1]);
        MPI_Waitall(2, reqs, status);

        if (m_prof)
            m_prof->pop();

        const BoxDim shifted_box = getShiftedBox();

        // wrap received particles across a global boundary back into global box
        for (unsigned int idx = 0; idx < n_recv_ptls; idx++)
            {
            pdata_element& p = m_recvbuf[idx];
            Scalar4& postype = p.pos;
            int3& image = p.image;

            shifted_box.wrap(postype, image);
            }

        // remove particles that were sent and fill particle data with received particles
        m_pdata->addParticles(m_recvbuf);
        } // end dir loop

    if (m_prof)
        m_prof->pop();
    }

//! Build ghost particle list, exchange ghost particle data
void Communicator::exchangeGhosts()
    {
    if (m_prof)
        m_prof->push("comm_ghost_exch");

    m_exec_conf->msg->notice(7) << "Communicator: exchange ghosts" << std::endl;

    const BoxDim& box = m_pdata->getBox();

    // Sending ghosts proceeds in two stages:
    // Stage 1: mark ghost atoms for sending (for covalently bonded particles, and non-bonded interactions)
    //          construct plans (= itineraries for ghost particles)
    // Stage 2: fill send buffers, exchange ghosts according to plans (sending the plan along with the particle)

    // resize and reset plans
    m_plan.resize(m_pdata->getN());

        {
        ArrayHandle<unsigned int> h_plan(m_plan, access_location::host, access_mode::readwrite);

        for (unsigned int i = 0; i < m_pdata->getN(); ++i)
            h_plan.data[i] = 0;
        }

    /*
     * Mark particles that are part of incomplete bonds for sending
     */
    boost::shared_ptr<BondData> bdata = m_sysdef->getBondData();

    /*
     * Mark non-bonded atoms for sending
     */

    // the ghost layer must be at_least m_r_ghost wide along every lattice direction
    Scalar3 ghost_fraction = m_r_ghost/box.getNearestPlaneDistance();
        {
        // scan all local atom positions if they are within r_ghost from a neighbor
        ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_plan(m_plan, access_location::host, access_mode::readwrite);

        for (unsigned int idx = 0; idx < m_pdata->getN(); idx++)
            {
            Scalar4 postype = h_pos.data[idx];
            Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);

            Scalar3 f = box.makeFraction(pos);
            if (f.x >= Scalar(1.0) - ghost_fraction.x)
                h_plan.data[idx] |= send_east;

            if (f.x < ghost_fraction.x)
                h_plan.data[idx] |= send_west;

            if (f.y >= Scalar(1.0) - ghost_fraction.y)
                h_plan.data[idx] |= send_north;

            if (f.y < ghost_fraction.y)
                h_plan.data[idx] |= send_south;

            if (f.z >= Scalar(1.0) - ghost_fraction.z)
                h_plan.data[idx] |= send_up;

            if (f.z < ghost_fraction.z)
                h_plan.data[idx] |= send_down;
            }
        }

    unsigned int mask = 0;
    Index3D di = m_decomposition->getDomainIndexer();
    if (di.getW() > 1) mask |= (send_east | send_west);
    if (di.getH() > 1) mask |= (send_north| send_south);
    if (di.getD() > 1) mask |= (send_up | send_down);

    // bonds
    m_bond_comm.markGhostParticles(m_plan, mask);

    // angles
    m_angle_comm.markGhostParticles(m_plan,mask);

    // dihedrals
    m_dihedral_comm.markGhostParticles(m_plan,mask);

    // impropers
    m_improper_comm.markGhostParticles(m_plan,mask);

    /*
     * Fill send buffers, exchange particles according to plans
     */

    // resize buffers
    m_plan_copybuf.resize(m_pdata->getN());
    m_pos_copybuf.resize(m_pdata->getN());
    m_charge_copybuf.resize(m_pdata->getN());
    m_diameter_copybuf.resize(m_pdata->getN());
    m_velocity_copybuf.resize(m_pdata->getN());
    m_orientation_copybuf.resize(m_pdata->getN());

    // ghost particle flags
    CommFlags flags = getFlags();

    for (unsigned int dir = 0; dir < 6; dir ++)
        {
        if (! isCommunicating(dir) ) continue;

        m_num_copy_ghosts[dir] = 0;

        // resize array of ghost particle tags
        unsigned int max_copy_ghosts = m_pdata->getN() + m_pdata->getNGhosts();
        m_copy_ghosts[dir].resize(max_copy_ghosts);

        // resize buffers
        m_plan_copybuf.resize(max_copy_ghosts);
        m_pos_copybuf.resize(max_copy_ghosts);
        m_charge_copybuf.resize(max_copy_ghosts);
        m_diameter_copybuf.resize(max_copy_ghosts);
        m_velocity_copybuf.resize(max_copy_ghosts);
        m_orientation_copybuf.resize(max_copy_ghosts);


            {
            // we fill all fields, but send only those that are requested by the CommFlags bitset
            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
            ArrayHandle<unsigned int>  h_plan(m_plan, access_location::host, access_mode::read);

            ArrayHandle<unsigned int> h_copy_ghosts(m_copy_ghosts[dir], access_location::host, access_mode::overwrite);
            ArrayHandle<unsigned int> h_plan_copybuf(m_plan_copybuf, access_location::host, access_mode::overwrite);
            ArrayHandle<Scalar4> h_pos_copybuf(m_pos_copybuf, access_location::host, access_mode::overwrite);
            ArrayHandle<Scalar> h_charge_copybuf(m_charge_copybuf, access_location::host, access_mode::overwrite);
            ArrayHandle<Scalar> h_diameter_copybuf(m_diameter_copybuf, access_location::host, access_mode::overwrite);
            ArrayHandle<Scalar4> h_velocity_copybuf(m_velocity_copybuf, access_location::host, access_mode::overwrite);
            ArrayHandle<Scalar4> h_orientation_copybuf(m_orientation_copybuf, access_location::host, access_mode::overwrite);

            for (unsigned int idx = 0; idx < m_pdata->getN() + m_pdata->getNGhosts(); idx++)
                {

                if (h_plan.data[idx] & (1 << dir))
                    {
                    // send with next message
                    h_pos_copybuf.data[m_num_copy_ghosts[dir]] = h_pos.data[idx];
                    h_charge_copybuf.data[m_num_copy_ghosts[dir]] = h_charge.data[idx];
                    h_diameter_copybuf.data[m_num_copy_ghosts[dir]] = h_diameter.data[idx];
                    h_velocity_copybuf.data[m_num_copy_ghosts[dir]] = h_vel.data[idx];
                    h_orientation_copybuf.data[m_num_copy_ghosts[dir]] = h_orientation.data[idx];
                    h_plan_copybuf.data[m_num_copy_ghosts[dir]] = h_plan.data[idx];

                    h_copy_ghosts.data[m_num_copy_ghosts[dir]] = h_tag.data[idx];
                    m_num_copy_ghosts[dir]++;
                    }
                }
            }
        unsigned int send_neighbor = m_decomposition->getNeighborRank(dir);

        // we receive from the direction opposite to the one we send to
        unsigned int recv_neighbor;
        if (dir % 2 == 0)
            recv_neighbor = m_decomposition->getNeighborRank(dir+1);
        else
            recv_neighbor = m_decomposition->getNeighborRank(dir-1);

        if (m_prof)
            m_prof->push("MPI send/recv");

        // communicate size of the message that will contain the particle data
        MPI_Request reqs[14];
        MPI_Status status[14];

        MPI_Isend(&m_num_copy_ghosts[dir],
            sizeof(unsigned int),
            MPI_BYTE,
            send_neighbor,
            0,
            m_mpi_comm,
            &reqs[0]);
        MPI_Irecv(&m_num_recv_ghosts[dir],
            sizeof(unsigned int),
            MPI_BYTE,
            recv_neighbor,
            0,
            m_mpi_comm,
            &reqs[1]);
        MPI_Waitall(2, reqs, status);

        if (m_prof)
            m_prof->pop();

        // append ghosts at the end of particle data array
        unsigned int start_idx = m_pdata->getN() + m_pdata->getNGhosts();

        // accommodate new ghost particles
        m_pdata->addGhostParticles(m_num_recv_ghosts[dir]);

        // resize plan array
        m_plan.resize(m_pdata->getN() + m_pdata->getNGhosts());

        // exchange particle data, write directly to the particle data arrays
        if (m_prof)
            m_prof->push("MPI send/recv");

            {
            ArrayHandle<unsigned int> h_copy_ghosts(m_copy_ghosts[dir], access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_plan_copybuf(m_plan_copybuf, access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_pos_copybuf(m_pos_copybuf, access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_charge_copybuf(m_charge_copybuf, access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_diameter_copybuf(m_diameter_copybuf, access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_velocity_copybuf(m_velocity_copybuf, access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_orientation_copybuf(m_orientation_copybuf, access_location::host, access_mode::read);

            ArrayHandle<unsigned int> h_plan(m_plan, access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);
            ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::readwrite);

            unsigned int nreq = 0;

            MPI_Isend(h_plan_copybuf.data,
                m_num_copy_ghosts[dir]*sizeof(unsigned int),
                MPI_BYTE,
                send_neighbor,
                1,
                m_mpi_comm,
                &reqs[nreq++]);
            MPI_Irecv(h_plan.data + start_idx,
                m_num_recv_ghosts[dir]*sizeof(unsigned int),
                MPI_BYTE,
                recv_neighbor,
                1,
                m_mpi_comm,
                &reqs[nreq++]);

            MPI_Isend(h_copy_ghosts.data,
                m_num_copy_ghosts[dir]*sizeof(unsigned int),
                MPI_BYTE,
                send_neighbor,
                2,
                m_mpi_comm,
                &reqs[nreq++]);
            MPI_Irecv(h_tag.data + start_idx,
                m_num_recv_ghosts[dir]*sizeof(unsigned int),
                MPI_BYTE,
                recv_neighbor,
                2,
                m_mpi_comm,
                &reqs[nreq++]);

            if (flags[comm_flag::position])
                {
                MPI_Isend(h_pos_copybuf.data,
                    m_num_copy_ghosts[dir]*sizeof(Scalar4),
                    MPI_BYTE,
                    send_neighbor,
                    3,
                    m_mpi_comm,
                    &reqs[nreq++]);
                MPI_Irecv(h_pos.data + start_idx,
                    m_num_recv_ghosts[dir]*sizeof(Scalar4),
                    MPI_BYTE,
                    recv_neighbor,
                    3,
                    m_mpi_comm,
                    &reqs[nreq++]);
                }

            if (flags[comm_flag::charge])
                {
                MPI_Isend(h_charge_copybuf.data,
                    m_num_copy_ghosts[dir]*sizeof(Scalar),
                    MPI_BYTE,
                    send_neighbor,
                    4,
                    m_mpi_comm,
                    &reqs[nreq++]);
                MPI_Irecv(h_charge.data + start_idx,
                    m_num_recv_ghosts[dir]*sizeof(Scalar),
                    MPI_BYTE,
                    recv_neighbor,
                    4,
                    m_mpi_comm,
                    &reqs[nreq++]);
                }

            if (flags[comm_flag::diameter])
                {
                MPI_Isend(h_diameter_copybuf.data,
                    m_num_copy_ghosts[dir]*sizeof(Scalar),
                    MPI_BYTE,
                    send_neighbor,
                    5,
                    m_mpi_comm,
                    &reqs[nreq++]);
                MPI_Irecv(h_diameter.data + start_idx,
                    m_num_recv_ghosts[dir]*sizeof(Scalar),
                    MPI_BYTE,
                    recv_neighbor,
                    5,
                    m_mpi_comm,
                    &reqs[nreq++]);
                }

            if (flags[comm_flag::velocity])
                {
                MPI_Isend(h_velocity_copybuf.data,
                    m_num_copy_ghosts[dir]*sizeof(Scalar4),
                    MPI_BYTE,
                    send_neighbor,
                    6,
                    m_mpi_comm,
                    &reqs[nreq++]);
                MPI_Irecv(h_vel.data + start_idx,
                    m_num_recv_ghosts[dir]*sizeof(Scalar4),
                    MPI_BYTE,
                    recv_neighbor,
                    6,
                    m_mpi_comm,
                    &reqs[nreq++]);
                }


            if (flags[comm_flag::orientation])
                {
                MPI_Isend(h_orientation_copybuf.data,
                    m_num_copy_ghosts[dir]*sizeof(Scalar4),
                    MPI_BYTE,
                    send_neighbor,
                    7,
                    m_mpi_comm,
                    &reqs[nreq++]);
                MPI_Irecv(h_orientation.data + start_idx,
                    m_num_recv_ghosts[dir]*sizeof(Scalar4),
                    MPI_BYTE,
                    recv_neighbor,
                    7,
                    m_mpi_comm,
                    &reqs[nreq++]);
                }

            MPI_Waitall(nreq, reqs, status);
            }

        if (m_prof)
            m_prof->pop();

        // wrap particle positions
        if (flags[comm_flag::position])
            {
            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);

            const BoxDim shifted_box = getShiftedBox();

            for (unsigned int idx = start_idx; idx < start_idx + m_num_recv_ghosts[dir]; idx++)
                {
                Scalar4& pos = h_pos.data[idx];

                // wrap particles received across a global boundary
                int3 img = make_int3(0,0,0);
                shifted_box.wrap(pos,img);
                }
            }

            {
            // set reverse-lookup tag -> idx
            ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::readwrite);

            for (unsigned int idx = start_idx; idx < start_idx + m_num_recv_ghosts[dir]; idx++)
                {
                assert(h_tag.data[idx] <= m_pdata->getNGlobal());
                assert(h_rtag.data[h_tag.data[idx]] == NOT_LOCAL);
                h_rtag.data[h_tag.data[idx]] = idx;
                }
            }
        } // end dir loop

    // we have updated ghost particles, so inform ParticleData about this
    m_pdata->notifyGhostParticleNumberChange();

    m_last_flags = flags;

    if (m_prof)
        m_prof->pop();
    }

//! update positions of ghost particles
void Communicator::beginUpdateGhosts(unsigned int timestep)
    {
    // we have a current m_copy_ghosts liss which contain the indices of particles
    // to send to neighboring processors
    if (m_prof)
        m_prof->push("comm_ghost_update");

    m_exec_conf->msg->notice(7) << "Communicator: update ghosts" << std::endl;

    // update data in these arrays

    unsigned int num_tot_recv_ghosts = 0; // total number of ghosts received

    for (unsigned int dir = 0; dir < 6; dir ++)
        {
        if (! isCommunicating(dir) ) continue;

        CommFlags flags = getFlags();

        if (flags[comm_flag::position])
            {
            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_pos_copybuf(m_pos_copybuf, access_location::host, access_mode::overwrite);
            ArrayHandle<unsigned int> h_copy_ghosts(m_copy_ghosts[dir], access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

            // copy positions of ghost particles
            for (unsigned int ghost_idx = 0; ghost_idx < m_num_copy_ghosts[dir]; ghost_idx++)
                {
                unsigned int idx = h_rtag.data[h_copy_ghosts.data[ghost_idx]];

                assert(idx < m_pdata->getN() + m_pdata->getNGhosts());

                // copy position into send buffer
                h_pos_copybuf.data[ghost_idx] = h_pos.data[idx];
                }
            }

        if (flags[comm_flag::velocity])
            {
            ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_velocity_copybuf(m_velocity_copybuf, access_location::host, access_mode::overwrite);
            ArrayHandle<unsigned int> h_copy_ghosts(m_copy_ghosts[dir], access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

            // copy velocity of ghost particles
            for (unsigned int ghost_idx = 0; ghost_idx < m_num_copy_ghosts[dir]; ghost_idx++)
                {
                unsigned int idx = h_rtag.data[h_copy_ghosts.data[ghost_idx]];

                assert(idx < m_pdata->getN() + m_pdata->getNGhosts());

                // copy velocityition into send buffer
                h_velocity_copybuf.data[ghost_idx] = h_vel.data[idx];
                }
            }

        if (flags[comm_flag::orientation])
            {
            ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_orientation_copybuf(m_orientation_copybuf, access_location::host, access_mode::overwrite);
            ArrayHandle<unsigned int> h_copy_ghosts(m_copy_ghosts[dir], access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

            // copy orientation of ghost particles
            for (unsigned int ghost_idx = 0; ghost_idx < m_num_copy_ghosts[dir]; ghost_idx++)
                {
                unsigned int idx = h_rtag.data[h_copy_ghosts.data[ghost_idx]];

                assert(idx < m_pdata->getN() + m_pdata->getNGhosts());

                // copy orientation into send buffer
                h_orientation_copybuf.data[ghost_idx] = h_orientation.data[idx];
                }
            }


        unsigned int send_neighbor = m_decomposition->getNeighborRank(dir);

        // we receive from the direction opposite to the one we send to
        unsigned int recv_neighbor;
        if (dir % 2 == 0)
            recv_neighbor = m_decomposition->getNeighborRank(dir+1);
        else
            recv_neighbor = m_decomposition->getNeighborRank(dir-1);


        unsigned int start_idx;

        if (m_prof)
            m_prof->push("MPI send/recv");

        start_idx = m_pdata->getN() + num_tot_recv_ghosts;

        num_tot_recv_ghosts += m_num_recv_ghosts[dir];

        size_t sz = 0;
        // only non-permanent fields (position, velocity, orientation) need to be considered here
        // charge and diameter are not updated during a run
        if (flags[comm_flag::position])
            {
            MPI_Request reqs[2];
            MPI_Status status[2];

            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_pos_copybuf(m_pos_copybuf, access_location::host, access_mode::read);

            // exchange particle data, write directly to the particle data arrays
            MPI_Isend(h_pos_copybuf.data, m_num_copy_ghosts[dir]*sizeof(Scalar4), MPI_BYTE, send_neighbor, 1, m_mpi_comm, &reqs[0]);
            MPI_Irecv(h_pos.data + start_idx, m_num_recv_ghosts[dir]*sizeof(Scalar4), MPI_BYTE, recv_neighbor, 1, m_mpi_comm, &reqs[1]);
            MPI_Waitall(2, reqs, status);

            sz += sizeof(Scalar4);
            }

        if (flags[comm_flag::velocity])
            {
            MPI_Request reqs[2];
            MPI_Status status[2];

            ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_vel_copybuf(m_velocity_copybuf, access_location::host, access_mode::read);

            // exchange particle data, write directly to the particle data arrays
            MPI_Isend(h_vel_copybuf.data, m_num_copy_ghosts[dir]*sizeof(Scalar4), MPI_BYTE, send_neighbor, 2, m_mpi_comm, &reqs[0]);
            MPI_Irecv(h_vel.data + start_idx, m_num_recv_ghosts[dir]*sizeof(Scalar4), MPI_BYTE, recv_neighbor, 2, m_mpi_comm, &reqs[1]);
            MPI_Waitall(2, reqs, status);

            sz += sizeof(Scalar4);
            }

        if (flags[comm_flag::orientation])
            {
            MPI_Request reqs[2];
            MPI_Status status[2];

            ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_orientation_copybuf(m_orientation_copybuf, access_location::host, access_mode::read);

            // exchange particle data, write directly to the particle data arrays
            MPI_Isend(h_orientation_copybuf.data, m_num_copy_ghosts[dir]*sizeof(Scalar4), MPI_BYTE, send_neighbor, 3, m_mpi_comm, &reqs[0]);
            MPI_Irecv(h_orientation.data + start_idx, m_num_recv_ghosts[dir]*sizeof(Scalar4), MPI_BYTE, recv_neighbor, 3, m_mpi_comm, &reqs[1]);
            MPI_Waitall(2, reqs, status);

            sz += sizeof(Scalar4);
            }

        if (m_prof)
            m_prof->pop(0, (m_num_recv_ghosts[dir]+m_num_copy_ghosts[dir])*sz);


        // wrap particle positions (only if copying positions)
        if (flags[comm_flag::position])
            {
            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);

            const BoxDim shifted_box = getShiftedBox();
            for (unsigned int idx = start_idx; idx < start_idx + m_num_recv_ghosts[dir]; idx++)
                {
                Scalar4& pos = h_pos.data[idx];

                // wrap particles received across a global boundary
                int3 img = make_int3(0,0,0);
                shifted_box.wrap(pos, img);
                }
            }

        } // end dir loop

        if (m_prof)
            m_prof->pop();
    }

const BoxDim Communicator::getShiftedBox() const
    {
    // construct the shifted global box for applying global boundary conditions
    BoxDim shifted_box = m_pdata->getGlobalBox();
    Scalar3 f= make_scalar3(0.5,0.5,0.5);

    Scalar3 shift = m_pdata->getBox().getNearestPlaneDistance()/
        shifted_box.getNearestPlaneDistance()/2.0;

    Scalar tol = 0.0001;
    shift += tol*make_scalar3(1.0,1.0,1.0);
    for (unsigned int dir = 0; dir < 6; dir ++)
        {
        if (m_decomposition->isAtBoundary(dir) &&  isCommunicating(dir))
            {
            if (dir == face_east)
                f.x += shift.x;
            else if (dir == face_west)
                f.x -= shift.x;
            else if (dir == face_north)
                f.y += shift.y;
            else if (dir == face_south)
                f.y -= shift.y;
            else if (dir == face_up)
                f.z += shift.z;
            else if (dir == face_down)
                f.z -= shift.z;
            }
        }
    Scalar3 dx = shifted_box.makeCoordinates(f);
    Scalar3 lo = shifted_box.getLo();
    Scalar3 hi = shifted_box.getHi();
    lo += dx;
    hi += dx;
    shifted_box.setLoHi(lo, hi);

    // only apply global boundary conditions along the communication directions
    uchar3 periodic = make_uchar3(0,0,0);

    periodic.x = isCommunicating(face_east) ? 1 : 0;
    periodic.y = isCommunicating(face_north) ? 1 : 0;
    periodic.z = isCommunicating(face_up) ? 1 : 0;

    shifted_box.setPeriodic(periodic);

    return shifted_box;
    }

//! Export Communicator class to python
void export_Communicator()
    {
     class_< std::vector<bool> >("std_vector_bool")
    .def(vector_indexing_suite<std::vector<bool> >());

    class_<Communicator, boost::shared_ptr<Communicator>, boost::noncopyable>("Communicator",
           init<boost::shared_ptr<SystemDefinition>,
                boost::shared_ptr<DomainDecomposition> >())
    ;
    }
#endif // ENABLE_MPI
