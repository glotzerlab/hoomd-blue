// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

/*! \file Communicator.cc
    \brief Implements the Communicator class
*/

#ifdef ENABLE_MPI

#include "Communicator.h"
#include "System.h"
#include "HOOMDMPI.h"

#include <algorithm>
#include <hoomd/extern/pybind/include/pybind11/stl.h>


using namespace std;
namespace py = pybind11;

#include <vector>

template<class group_data>
Communicator::GroupCommunicator<group_data>::GroupCommunicator(Communicator& comm, std::shared_ptr<group_data> gdata)
    : m_comm(comm), m_exec_conf(comm.m_exec_conf), m_gdata(gdata)
    {
    // the size of the bit field must be larger or equal the group size
    assert(sizeof(unsigned int)*8 >= group_data::size);
    }

template<class group_data>
void Communicator::GroupCommunicator<group_data>::migrateGroups(bool incomplete, bool local_multiple)
    {
    if (m_gdata->getNGlobal())
        {
        if (m_comm.m_prof) m_comm.m_prof->push(m_exec_conf, m_gdata->getName());

            {
            // wipe out reverse-lookup tag -> idx for old ghost groups
            ArrayHandle<unsigned int> h_group_tag(m_gdata->getTags(), access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_group_rtag(m_gdata->getRTags(), access_location::host, access_mode::readwrite);
            for (unsigned int i = 0; i < m_gdata->getNGhosts(); i++)
                {
                unsigned int idx = m_gdata->getN() + i;
                h_group_rtag.data[h_group_tag.data[idx]] = GROUP_NOT_LOCAL;
                }
            }


        // remove ghost groups
        m_gdata->removeAllGhostGroups();

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
            ArrayHandle<typeval_t> h_group_typeval(m_gdata->getTypeValArray(), access_location::host, access_mode::read);
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
                    el.typeval = h_group_typeval.data[group_idx];
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
                            {
                            if (local_multiple || i == 0)
                                {
                                is_local = true;
                                }
                            }
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
            ArrayHandle<typeval_t> h_group_typeval(m_gdata->getTypeValArray(), access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_group_tag(m_gdata->getTags(), access_location::host, access_mode::read);
            ArrayHandle<typename group_data::ranks_t> h_group_ranks(m_gdata->getRanksArray(), access_location::host, access_mode::read);

            // access alternate arrays to write to
            ArrayHandle<typename group_data::members_t> h_groups_alt(m_gdata->getAltMembersArray(), access_location::host, access_mode::overwrite);
            ArrayHandle<typeval_t> h_group_typeval_alt(m_gdata->getAltTypeValArray(), access_location::host, access_mode::overwrite);
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
                    h_group_typeval_alt.data[n] = h_group_typeval.data[group_idx];
                    h_group_tag_alt.data[n] = group_tag;
                    h_group_ranks_alt.data[n] = h_group_ranks.data[group_idx];

                    // rebuild rtags
                    h_group_rtag.data[group_tag] = n++;
                    }
                }

                new_ngroups = n;
            }


        // make alternate arrays current
        m_gdata->swapMemberArrays();
        m_gdata->swapTypeArrays();
        m_gdata->swapTagArrays();
        m_gdata->swapRankArrays();


        assert(new_ngroups <= m_gdata->getN());

        // resize group arrays
        m_gdata->removeGroups(m_gdata->getN() - new_ngroups);

        assert(m_gdata->getN() == new_ngroups);

        // reset send buf
        m_groups_sendbuf.clear();

        // output groups to send buffer in rank-sorted order
        for (typename group_map_t::iterator it = group_send_map.begin(); it != group_send_map.end(); ++it)
            {
            m_groups_sendbuf.push_back(it->second);
            }

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

        // resize group arrays to accommodate additional groups (there can still be duplicates with local groups)
        m_gdata->addGroups(n_recv_unique);

        auto& groups_array = m_gdata->getMembersArray();
        auto& group_typeval_array = m_gdata->getTypeValArray();
        auto& group_tag_array = m_gdata->getTags();
        auto& group_ranks_array = m_gdata->getRanksArray();

        unsigned int nremove = 0;

        unsigned int myrank = m_exec_conf->getRank();

            {
            ArrayHandle<unsigned int> h_group_rtag(m_gdata->getRTags(), access_location::host, access_mode::readwrite);
            ArrayHandle<typename group_data::members_t> h_groups(groups_array, access_location::host, access_mode::readwrite);
            ArrayHandle<typeval_t> h_group_typeval(group_typeval_array, access_location::host, access_mode::readwrite);
            ArrayHandle<unsigned int> h_group_tag(group_tag_array, access_location::host, access_mode::readwrite);
            ArrayHandle<typename group_data::ranks_t> h_group_ranks(group_ranks_array, access_location::host, access_mode::readwrite);

            // add non-duplicate groups to group data
            unsigned int add_idx = old_ngroups;
            for (typename recv_map_t::iterator it = recv_map.begin(); it != recv_map.end(); ++it)
                {
                typename group_data::packed_t el = it->second;

                unsigned int tag = el.group_tag;
                unsigned int group_rtag = h_group_rtag.data[tag];

                bool remove = false;
                if (! local_multiple)
                    {
                    // only add if we own the first particle
                    assert(group_data::size);
                    if (el.ranks.idx[0] != myrank)
                        {
                        remove = true;
                        }
                    }

                if (!remove)
                    {
                    if (group_rtag == GROUP_NOT_LOCAL)
                        {
                        h_groups.data[add_idx] = el.tags;
                        h_group_typeval.data[add_idx] = el.typeval;
                        h_group_tag.data[add_idx] = tag;
                        h_group_ranks.data[add_idx] = el.ranks;

                        // update reverse-lookup table
                        h_group_rtag.data[tag] = add_idx++;
                        }
                    else
                        {
                        remove = true;
                        }
                    }

                if (remove)
                    {
                    nremove++;
                    }
                }
            }

        // resize arrays to final size
        m_gdata->removeGroups(nremove);

        if (m_comm.m_prof) m_comm.m_prof->pop();
        }
    }

//! Mark ghost particles
template<class group_data>
void Communicator::GroupCommunicator<group_data>::markGhostParticles(
    const GlobalVector<unsigned int>& plans,
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

template<class group_data>
void Communicator::GroupCommunicator<group_data>::exchangeGhostGroups(
    const GlobalArray<unsigned int>& plans, unsigned int mask)
    {
    if (m_gdata->getNGlobal())
        {
        if (m_comm.m_prof) m_comm.m_prof->push(m_exec_conf, m_gdata->getName());

        // send plan for groups
        std::vector<unsigned int> group_plan(m_gdata->getN(), 0);

            {
            ArrayHandle<typename group_data::members_t> h_groups(m_gdata->getMembersArray(), access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_rtag(m_comm.m_pdata->getRTags(), access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_plan(plans, access_location::host, access_mode::read);

            unsigned int ngroups_local = m_gdata->getN();

            unsigned int n_local = m_comm.m_pdata->getN();
            unsigned int max_local = n_local + m_comm.m_pdata->getNGhosts();

            for (unsigned int group_idx = 0; group_idx < ngroups_local; group_idx++)
                {
                typename group_data::members_t members = h_groups.data[group_idx];

                assert(group_data::size);
                unsigned int plan = 0;
                for (unsigned int i = 0; i < group_data::size; ++i)
                    {
                    unsigned int tag = members.tag[i];
                    unsigned int pidx = h_rtag.data[tag];

                    if (i==0 && pidx >= n_local)
                        {
                        // only the rank that owns the first ptl of a group sends it as a ghost
                        plan = 0;
                        break;
                        }

                    assert(pidx != NOT_LOCAL);
                    assert(pidx <= m_comm.m_pdata->getN() + m_comm.m_pdata->getNGhosts());

                    if (pidx >= max_local)
                        {
                        this->m_exec_conf->msg->error() << "comm.*: encountered incomplete " << group_data::getName() << std::endl;
                        throw std::runtime_error("Error during communication");
                        }

                    plan |= h_plan.data[pidx];
                    }

                group_plan[group_idx] = plan;
                } // end loop over groups
            }


        for (unsigned int dir = 0; dir < 6; dir++)
            {
            if (! m_comm.isCommunicating(dir) ) continue;

            unsigned int send_neighbor = m_comm.m_decomposition->getNeighborRank(dir);

            // we receive from the direction opposite to the one we send to
            unsigned int recv_neighbor;
            if (dir % 2 == 0)
                recv_neighbor = m_comm.m_decomposition->getNeighborRank(dir+1);
            else
                recv_neighbor = m_comm.m_decomposition->getNeighborRank(dir-1);

            /*
             * Fill send buffers, exchange groups according to plans
             */

            // resize buffers
            std::vector<unsigned int> plan_copybuf(m_gdata->getN(),0);
            m_groups_sendbuf.resize(m_gdata->getN());
            unsigned int num_copy_ghosts;
            unsigned int num_recv_ghosts;

            num_copy_ghosts = 0;

            // resize array of ghost particle tags
            unsigned int max_copy_ghosts = m_gdata->getN() + m_gdata->getNGhosts();

            // resize buffers
            plan_copybuf.resize(max_copy_ghosts);
            m_groups_sendbuf.resize(max_copy_ghosts);


                {
                ArrayHandle<typename group_data::members_t> h_groups(m_gdata->getMembersArray(), access_location::host, access_mode::read);
                ArrayHandle<typeval_t> h_group_typeval(m_gdata->getTypeValArray(), access_location::host, access_mode::read);
                ArrayHandle<unsigned int> h_group_tag(m_gdata->getTags(), access_location::host, access_mode::read);
                ArrayHandle<typename group_data::ranks_t> h_group_ranks(m_gdata->getRanksArray(), access_location::host, access_mode::read);

                for (unsigned int idx = 0; idx < m_gdata->getN() + m_gdata->getNGhosts(); idx++)
                    {
                    if (group_plan[idx] & (1 << dir))
                        {
                        // send plan
                        plan_copybuf[num_copy_ghosts] = group_plan[idx];

                        // pack group
                        typename group_data::packed_t el;
                        el.tags = h_groups.data[idx];
                        el.typeval = h_group_typeval.data[idx];
                        el.group_tag = h_group_tag.data[idx];
                        el.ranks = h_group_ranks.data[idx];

                        m_groups_sendbuf[num_copy_ghosts] = el;
                        num_copy_ghosts++;
                        }
                    }
                }

            if (m_comm.m_prof)
                m_comm.m_prof->push("MPI send/recv");

            // communicate size of the message that will contain the particle data
            MPI_Request reqs[4];
            MPI_Status status[4];

            MPI_Isend(&num_copy_ghosts,
                sizeof(unsigned int),
                MPI_BYTE,
                send_neighbor,
                0,
                m_comm.m_mpi_comm,
                &reqs[0]);
            MPI_Irecv(&num_recv_ghosts,
                sizeof(unsigned int),
                MPI_BYTE,
                recv_neighbor,
                0,
                m_comm.m_mpi_comm,
                &reqs[1]);
            MPI_Waitall(2, reqs, status);

            if (m_comm.m_prof)
                m_comm.m_prof->pop();

            // append ghosts at the end of particle data array
            unsigned int start_idx = m_gdata->getN() + m_gdata->getNGhosts();

            // resize plan array
            group_plan.resize(m_gdata->getN() + m_gdata->getNGhosts()+ num_recv_ghosts);

            // resize recv buf
            m_groups_recvbuf.resize(num_recv_ghosts);

            // exchange group data, write directly to the particle data arrays
            if (m_comm.m_prof)
                {
                m_comm.m_prof->push("MPI send/recv");
                }

                {
                MPI_Isend(&plan_copybuf.front(),
                    num_copy_ghosts*sizeof(unsigned int),
                    MPI_BYTE,
                    send_neighbor,
                    1,
                    m_comm.m_mpi_comm,
                    &reqs[0]);
                MPI_Irecv(&group_plan.front()+ start_idx,
                    num_recv_ghosts*sizeof(unsigned int),
                    MPI_BYTE,
                    recv_neighbor,
                    1,
                    m_comm.m_mpi_comm,
                    &reqs[1]);

                MPI_Isend(&m_groups_sendbuf.front(),
                    num_copy_ghosts*sizeof(typename group_data::packed_t),
                    MPI_BYTE,
                    send_neighbor,
                    2,
                    m_comm.m_mpi_comm,
                    &reqs[2]);
                MPI_Irecv(&m_groups_recvbuf.front(),
                    num_recv_ghosts*sizeof(typename group_data::packed_t),
                    MPI_BYTE,
                    recv_neighbor,
                    2,
                    m_comm.m_mpi_comm,
                    &reqs[3]);
                MPI_Waitall(4, reqs, status);
                }

            if (m_comm.m_prof)
                m_comm.m_prof->pop();

            unsigned int old_n_ghost = m_gdata->getNGhosts();

            // accommodate new ghost particles
            m_gdata->addGhostGroups(num_recv_ghosts);

            unsigned int added_groups = 0;
                {
                // access group data
                ArrayHandle<typename group_data::members_t> h_groups(m_gdata->getMembersArray(), access_location::host, access_mode::readwrite);
                ArrayHandle<typeval_t> h_group_typeval(m_gdata->getTypeValArray(), access_location::host, access_mode::readwrite);
                ArrayHandle<unsigned int> h_group_tag(m_gdata->getTags(), access_location::host, access_mode::readwrite);
                ArrayHandle<typename group_data::ranks_t> h_group_ranks(m_gdata->getRanksArray(), access_location::host, access_mode::readwrite);
                ArrayHandle<unsigned int> h_group_rtag(m_gdata->getRTags(), access_location::host, access_mode::readwrite);

                // access particle data
                ArrayHandle<unsigned int> h_rtag(m_comm.m_pdata->getRTags(), access_location::host, access_mode::read);

                unsigned int max_local = m_comm.m_pdata->getN() + m_comm.m_pdata->getNGhosts();

                // unpack group buffer
                for (unsigned int i = 0; i < num_recv_ghosts; i++)
                    {
                    typename group_data::packed_t el = m_groups_recvbuf[i];
                    if (h_group_rtag.data[el.group_tag] != GROUP_NOT_LOCAL)
                        continue;

                    bool has_nonlocal_members = false;
                    for (unsigned int j = 0; j < group_data::size; ++j)
                        {
                        unsigned int tag = el.tags.tag[j];
                        assert(tag <= m_comm.m_pdata->getMaximumTag());
                        if (h_rtag.data[tag] >= max_local)
                            {
                            has_nonlocal_members = true;
                            break;
                            }
                        }

                    // omit nonlocal groups
                    if (has_nonlocal_members)
                        continue;

                    h_groups.data[start_idx + added_groups] = el.tags;
                    h_group_typeval.data[start_idx + added_groups] = el.typeval;
                    h_group_tag.data[start_idx + added_groups] = el.group_tag;
                    h_group_ranks.data[start_idx + added_groups] = el.ranks;
                    h_group_rtag.data[el.group_tag] = start_idx+added_groups;

                    added_groups++;
                    }
                }

            // update ghost group number
            m_gdata->removeAllGhostGroups();
            m_gdata->addGhostGroups(old_n_ghost+added_groups);
            } // end loop over direction

        if (m_comm.m_prof)
            m_comm.m_prof->pop();

        // notify subscribers that group order has changed
        m_gdata->notifyGroupReorder();

        } // end if groups exist
    }


//! Constructor
Communicator::Communicator(std::shared_ptr<SystemDefinition> sysdef,
                           std::shared_ptr<DomainDecomposition> decomposition)
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
            m_body_copybuf(m_exec_conf),
            m_image_copybuf(m_exec_conf),
            m_velocity_copybuf(m_exec_conf),
            m_orientation_copybuf(m_exec_conf),
            m_plan_copybuf(m_exec_conf),
            m_tag_copybuf(m_exec_conf),
            m_netforce_copybuf(m_exec_conf),
            m_nettorque_copybuf(m_exec_conf),
            m_netvirial_copybuf(m_exec_conf),
            m_netvirial_recvbuf(m_exec_conf),
            m_plan(m_exec_conf),
            m_plan_reverse(m_exec_conf),
            m_tag_reverse(m_exec_conf),
            m_netforce_reverse_copybuf(m_exec_conf),
            m_netforce_reverse_recvbuf(m_exec_conf),
            m_r_ghost_max(Scalar(0.0)),
            m_r_extra_ghost_max(Scalar(0.0)),
            m_ghosts_added(0),
            m_has_ghost_particles(false),
            m_last_flags(0),
            m_comm_pending(false),
            m_bond_comm(*this, m_sysdef->getBondData()),
            m_angle_comm(*this, m_sysdef->getAngleData()),
            m_dihedral_comm(*this, m_sysdef->getDihedralData()),
            m_improper_comm(*this, m_sysdef->getImproperData()),
            m_constraint_comm(*this, m_sysdef->getConstraintData()),
            m_pair_comm(*this, m_sysdef->getPairData())
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
        GlobalVector<unsigned int> copy_ghosts(m_exec_conf);
        m_copy_ghosts[dir].swap(copy_ghosts);
        m_num_copy_ghosts[dir] = 0;
        m_num_recv_ghosts[dir] = 0;
        }

    // All buffers corresponding to sending ghosts in reverse
    for (unsigned int dir = 0; dir < 6; dir ++)
        {
        GlobalVector<unsigned int> copy_ghosts_reverse(m_exec_conf);
        m_copy_ghosts_reverse[dir].swap(copy_ghosts_reverse);
        GlobalVector<unsigned int> plan_reverse_copybuf(m_exec_conf);
        m_plan_reverse_copybuf[dir].swap(plan_reverse_copybuf);
        m_num_copy_local_ghosts_reverse[dir] = 0;
        m_num_recv_local_ghosts_reverse[dir] = 0;

        GlobalVector<unsigned int> forward_ghosts_reverse(m_exec_conf);
        m_forward_ghosts_reverse[dir].swap(forward_ghosts_reverse);
        m_num_forward_ghosts_reverse[dir] = 0;
        m_num_recv_forward_ghosts_reverse[dir] = 0;
        }

    // connect to particle sort signal
    m_pdata->getParticleSortSignal().connect<Communicator, &Communicator::forceMigrate>(this);

    // connect to particle sort signal
    m_pdata->getGhostParticlesRemovedSignal().connect<Communicator, &Communicator::slotGhostParticlesRemoved>(this);

    // connect to type change signal
    m_pdata->getNumTypesChangeSignal().connect<Communicator, &Communicator::slotNumTypesChanged>(this);

    // allocate per type ghost width
    GlobalArray<Scalar> r_ghost(m_pdata->getNTypes(), m_exec_conf);
    m_r_ghost.swap(r_ghost);

    GlobalArray<Scalar> r_ghost_body(m_pdata->getNTypes(), m_exec_conf);
    m_r_ghost_body.swap(r_ghost_body);

    /*
     * Bonded group communication
     */
    m_bonds_changed = true;
    m_sysdef->getBondData()->getGroupNumChangeSignal().connect<Communicator, &Communicator::setBondsChanged>(this);

    m_angles_changed = true;
    m_sysdef->getAngleData()->getGroupNumChangeSignal().connect<Communicator, &Communicator::setAnglesChanged>(this);

    m_dihedrals_changed = true;
    m_sysdef->getDihedralData()->getGroupNumChangeSignal().connect<Communicator, &Communicator::setDihedralsChanged>(this);

    m_impropers_changed = true;
    m_sysdef->getImproperData()->getGroupNumChangeSignal().connect<Communicator, &Communicator::setImpropersChanged>(this);

    m_constraints_changed = true;
    m_sysdef->getConstraintData()->getGroupNumChangeSignal().connect<Communicator, &Communicator::setConstraintsChanged>(this);

    m_pairs_changed = true;
    m_sysdef->getPairData()->getGroupNumChangeSignal().connect<Communicator, &Communicator::setPairsChanged>(this);

    // allocate memory
    GlobalArray<unsigned int> neighbors(NEIGH_MAX,m_exec_conf);
    m_neighbors.swap(neighbors);

    GlobalArray<unsigned int> unique_neighbors(NEIGH_MAX,m_exec_conf);
    m_unique_neighbors.swap(unique_neighbors);

    // neighbor masks
    GlobalArray<unsigned int> adj_mask(NEIGH_MAX, m_exec_conf);
    m_adj_mask.swap(adj_mask);

    GlobalArray<unsigned int> begin(NEIGH_MAX,m_exec_conf);
    m_begin.swap(begin);

    GlobalArray<unsigned int> end(NEIGH_MAX,m_exec_conf);
    m_end.swap(end);

    initializeNeighborArrays();

    /* create a type for pdata_element */
    const int nitems=14;
    int blocklengths[14] = {4,4,3,1,1,3,1,4,4,3,1,4,4,6};
    MPI_Datatype types[14] = {MPI_HOOMD_SCALAR, MPI_HOOMD_SCALAR, MPI_HOOMD_SCALAR, MPI_HOOMD_SCALAR,
        MPI_HOOMD_SCALAR, MPI_INT, MPI_UNSIGNED, MPI_HOOMD_SCALAR, MPI_HOOMD_SCALAR, MPI_HOOMD_SCALAR,
        MPI_UNSIGNED, MPI_HOOMD_SCALAR, MPI_HOOMD_SCALAR, MPI_HOOMD_SCALAR};
    MPI_Aint offsets[14];

    offsets[0] = offsetof(pdata_element, pos);
    offsets[1] = offsetof(pdata_element, vel);
    offsets[2] = offsetof(pdata_element, accel);
    offsets[3] = offsetof(pdata_element, charge);
    offsets[4] = offsetof(pdata_element, diameter);
    offsets[5] = offsetof(pdata_element, image);
    offsets[6] = offsetof(pdata_element, body);
    offsets[7] = offsetof(pdata_element, orientation);
    offsets[8] = offsetof(pdata_element, angmom);
    offsets[9] = offsetof(pdata_element, inertia);
    offsets[10] = offsetof(pdata_element, tag);
    offsets[11] = offsetof(pdata_element, net_force);
    offsets[12] = offsetof(pdata_element, net_torque);
    offsets[13] = offsetof(pdata_element, net_virial);

    MPI_Datatype tmp;
    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &tmp);
    MPI_Type_commit(&tmp);

    MPI_Type_create_resized(tmp, 0, sizeof(pdata_element), &m_mpi_pdata_element);
    MPI_Type_commit(&m_mpi_pdata_element);
    MPI_Type_free(&tmp);
    }

//! Destructor
Communicator::~Communicator()
    {
    m_exec_conf->msg->notice(5) << "Destroying Communicator" << std::endl;
    m_pdata->getParticleSortSignal().disconnect<Communicator, &Communicator::forceMigrate>(this);
    m_pdata->getGhostParticlesRemovedSignal().disconnect<Communicator, &Communicator::slotGhostParticlesRemoved>(this);
    m_pdata->getNumTypesChangeSignal().disconnect<Communicator, &Communicator::slotNumTypesChanged>(this);

    m_sysdef->getBondData()->getGroupNumChangeSignal().disconnect<Communicator, &Communicator::setBondsChanged>(this);
    m_sysdef->getAngleData()->getGroupNumChangeSignal().disconnect<Communicator, &Communicator::setAnglesChanged>(this);
    m_sysdef->getDihedralData()->getGroupNumChangeSignal().disconnect<Communicator, &Communicator::setDihedralsChanged>(this);
    m_sysdef->getImproperData()->getGroupNumChangeSignal().disconnect<Communicator, &Communicator::setImpropersChanged>(this);
    m_sysdef->getConstraintData()->getGroupNumChangeSignal().disconnect<Communicator, &Communicator::setConstraintsChanged>(this);
    m_sysdef->getPairData()->getGroupNumChangeSignal().disconnect<Communicator, &Communicator::setPairsChanged>(this);

    MPI_Type_free(&m_mpi_pdata_element);
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
    m_flags = CommFlags(0);
    m_requested_flags.emit_accumulate( [&](CommFlags f)
                                        {
                                        m_flags |= f;
                                        }
                                      , timestep);

    if (!m_force_migrate && !m_compute_callbacks.empty() && m_has_ghost_particles)
        {
        // do an obligatory update before determining whether to migrate
        beginUpdateGhosts(timestep);
        finishUpdateGhosts(timestep);

        // call subscribers after ghost update, but before distance check
        m_compute_callbacks.emit(timestep);

        // by now, local particles may have moved outside the box due to the rigid body update
        // we will make sure that they are inside by doing a second migrate if necessary
        }

    bool migrate_request = false;
    if (! m_force_migrate)
        {
        // distance check, may not be called directly after particle reorder (such as
        // due to SFCPackUpdater running before)
        m_migrate_requests.emit_accumulate( [&](bool r)
                                                {
                                                migrate_request = migrate_request || r;
                                                },
                                            timestep);
        }

    bool migrate = migrate_request || m_force_migrate || !m_has_ghost_particles;

    // Update ghosts if we are not migrating
    if (!migrate && m_compute_callbacks.empty())
        {
        beginUpdateGhosts(timestep);

        finishUpdateGhosts(timestep);
        }

    // Check if migration of particles is requested
    if (migrate)
        {
        m_force_migrate = false;

        // If so, migrate atoms
        migrateParticles();

        // Construct ghost send lists, exchange ghost atom data
        exchangeGhosts();

        // update particle data now that ghosts are available
        m_compute_callbacks.emit(timestep);

        m_has_ghost_particles = true;
        }

    m_is_communicating = false;
    }

//! Transfer particles between neighboring domains
void Communicator::migrateParticles()
    {
    m_exec_conf->msg->notice(7) << "Communicator: migrate particles" << std::endl;

    updateGhostWidth();

    // check if simulation box is sufficiently large for domain decomposition
    checkBoxSize();

    if (m_prof)
        m_prof->push("comm_migrate");

    // remove ghost particles from system
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
        m_bond_comm.migrateGroups(m_bonds_changed, true);
        m_bonds_changed = false;

        // Special pairs
        m_pair_comm.migrateGroups(m_pairs_changed, true);
        m_pairs_changed = false;

        // Angles
        m_angle_comm.migrateGroups(m_angles_changed, true);
        m_angles_changed = false;

        // Dihedrals
        m_dihedral_comm.migrateGroups(m_dihedrals_changed, true);
        m_dihedrals_changed = false;

        // Dihedrals
        m_improper_comm.migrateGroups(m_impropers_changed, true);
        m_impropers_changed = false;

        // Constraints
        m_constraint_comm.migrateGroups(m_constraints_changed, true);
        m_constraints_changed = false;

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
        m_reqs.resize(2);
        m_stats.resize(2);

        unsigned int n_send_ptls = m_sendbuf.size();

        MPI_Isend(&n_send_ptls, 1, MPI_UNSIGNED, send_neighbor, 0, m_mpi_comm, & m_reqs[0]);
        MPI_Irecv(&n_recv_ptls, 1, MPI_UNSIGNED, recv_neighbor, 0, m_mpi_comm, & m_reqs[1]);
        MPI_Waitall(2, &m_reqs.front(), &m_stats.front());

        // Resize receive buffer
        m_recvbuf.resize(n_recv_ptls);

        // exchange particle data
        m_reqs.resize(2);
        m_stats.resize(2);
        MPI_Isend(&m_sendbuf.front(), n_send_ptls, m_mpi_pdata_element, send_neighbor, 1, m_mpi_comm, & m_reqs[0]);
        MPI_Irecv(&m_recvbuf.front(), n_recv_ptls, m_mpi_pdata_element, recv_neighbor, 1, m_mpi_comm, & m_reqs[1]);
        MPI_Waitall(2, &m_reqs.front(), &m_stats.front());

        if (m_prof)
            m_prof->pop();

        // wrap received particles across a global boundary back into global box
        const BoxDim shifted_box = getShiftedBox();
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

void Communicator::updateGhostWidth()
    {
        {
        // reset values (this may not be needed in most cases, but it doesn't harm to be safe
        ArrayHandle<Scalar> h_r_ghost(m_r_ghost, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar> h_r_ghost_body(m_r_ghost_body, access_location::host, access_mode::overwrite);
        for (unsigned int cur_type = 0; cur_type < m_pdata->getNTypes(); ++cur_type)
            {
            h_r_ghost.data[cur_type] = Scalar(0.0);
            h_r_ghost_body.data[cur_type] = Scalar(0.0);
            }
        }

    if (!m_ghost_layer_width_requests.empty())
        {
        // update the ghost layer width only if subscribers are available
        ArrayHandle<Scalar> h_r_ghost(m_r_ghost, access_location::host, access_mode::readwrite);

        // reduce per type using the signals, and then overall
        Scalar r_ghost_max = 0.0;
        for (unsigned int cur_type = 0; cur_type < m_pdata->getNTypes(); ++cur_type)
            {
            Scalar r_ghost_i = 0.0;
            m_ghost_layer_width_requests.emit_accumulate([&](Scalar r)
                                                            {
                                                            if (r > r_ghost_i) r_ghost_i = r;
                                                            }
                                                            ,cur_type);
            h_r_ghost.data[cur_type] = r_ghost_i;
            if (r_ghost_i > r_ghost_max) r_ghost_max = r_ghost_i;
            }
        m_r_ghost_max = r_ghost_max;
        }
    if (!m_extra_ghost_layer_width_requests.empty())
        {
        // update the ghost layer width only if subscribers are available
        ArrayHandle<Scalar> h_r_ghost_body(m_r_ghost_body, access_location::host, access_mode::readwrite);

        Scalar r_extra_ghost_max = 0.0;
        for (unsigned int cur_type = 0; cur_type < m_pdata->getNTypes(); ++cur_type)
            {
            Scalar r_extra_ghost_i = 0.0;
            m_extra_ghost_layer_width_requests.emit_accumulate([&](Scalar r)
                                                            {
                                                            if (r > r_extra_ghost_i) r_extra_ghost_i = r;
                                                            }
                                                            ,cur_type);

            h_r_ghost_body.data[cur_type] = r_extra_ghost_i;
            if (r_extra_ghost_i  > r_extra_ghost_max) r_extra_ghost_max = r_extra_ghost_i;
            }
        m_r_extra_ghost_max = r_extra_ghost_max;
        }
    }

//! Build ghost particle list, exchange ghost particle data
void Communicator::exchangeGhosts()
    {
    // check if simulation box is sufficiently large for domain decomposition
    checkBoxSize();

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
     * Mark non-bonded atoms for sending
     */
    updateGhostWidth();

    // compute the ghost layer widths as fractions
    ArrayHandle<Scalar> h_r_ghost(m_r_ghost, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_r_ghost_body(m_r_ghost_body, access_location::host, access_mode::read);
    const Scalar3 box_dist = box.getNearestPlaneDistance();
    std::vector<Scalar3> ghost_fractions(m_pdata->getNTypes());
    std::vector<Scalar3> ghost_fractions_body(m_pdata->getNTypes());
    for (unsigned int cur_type = 0; cur_type < m_pdata->getNTypes(); ++cur_type)
        {
        ghost_fractions[cur_type] = h_r_ghost.data[cur_type] / box_dist;
        ghost_fractions_body[cur_type] = h_r_ghost_body.data[cur_type] / box_dist;
        }

        {
        // scan all local atom positions if they are within r_ghost from a neighbor
        ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_body(m_pdata->getBodies(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_plan(m_plan, access_location::host, access_mode::readwrite);

        for (unsigned int idx = 0; idx < m_pdata->getN(); idx++)
            {
            Scalar4 postype = h_pos.data[idx];
            Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);

            // get the ghost fraction for this particle type
            const unsigned int type = __scalar_as_int(postype.w);
            Scalar3 ghost_fraction = ghost_fractions[type];

            if (h_body.data[idx] < MIN_FLOPPY)
                {
                ghost_fraction += ghost_fractions_body[type];
                }

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

    // special pairs
    m_pair_comm.markGhostParticles(m_plan, mask);

    // angles
    m_angle_comm.markGhostParticles(m_plan,mask);

    // dihedrals
    m_dihedral_comm.markGhostParticles(m_plan,mask);

    // impropers
    m_improper_comm.markGhostParticles(m_plan,mask);

    // constraints
    m_constraint_comm.markGhostParticles(m_plan, mask);

    /*
     * Fill send buffers, exchange particles according to plans
     */

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

        if (flags[comm_flag::position])
            m_pos_copybuf.resize(max_copy_ghosts);

        if (flags[comm_flag::charge])
            m_charge_copybuf.resize(max_copy_ghosts);

        if (flags[comm_flag::body])
            m_body_copybuf.resize(max_copy_ghosts);

        if (flags[comm_flag::image])
            m_image_copybuf.resize(max_copy_ghosts);

        if (flags[comm_flag::diameter])
            m_diameter_copybuf.resize(max_copy_ghosts);

        if (flags[comm_flag::velocity])
            m_velocity_copybuf.resize(max_copy_ghosts);

        if (flags[comm_flag::orientation])
            {
            m_orientation_copybuf.resize(max_copy_ghosts);
            }

            {
            // we fill all fields, but send only those that are requested by the CommFlags bitset
            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_body(m_pdata->getBodies(), access_location::host, access_mode::read);
            ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
            ArrayHandle<unsigned int>  h_plan(m_plan, access_location::host, access_mode::readwrite);

            ArrayHandle<unsigned int> h_copy_ghosts(m_copy_ghosts[dir], access_location::host, access_mode::overwrite);
            ArrayHandle<unsigned int> h_plan_copybuf(m_plan_copybuf, access_location::host, access_mode::overwrite);
            ArrayHandle<Scalar4> h_pos_copybuf(m_pos_copybuf, access_location::host, access_mode::overwrite);
            ArrayHandle<Scalar> h_charge_copybuf(m_charge_copybuf, access_location::host, access_mode::overwrite);
            ArrayHandle<Scalar> h_diameter_copybuf(m_diameter_copybuf, access_location::host, access_mode::overwrite);
            ArrayHandle<unsigned int> h_body_copybuf(m_body_copybuf, access_location::host, access_mode::overwrite);
            ArrayHandle<int3> h_image_copybuf(m_image_copybuf, access_location::host, access_mode::overwrite);
            ArrayHandle<Scalar4> h_velocity_copybuf(m_velocity_copybuf, access_location::host, access_mode::overwrite);
            ArrayHandle<Scalar4> h_orientation_copybuf(m_orientation_copybuf, access_location::host, access_mode::overwrite);

            for (unsigned int idx = 0; idx < m_pdata->getN() + m_pdata->getNGhosts(); idx++)
                {

                if (h_plan.data[idx] & (1 << dir))
                    {
                    // send with next message
                    if (flags[comm_flag::position]) h_pos_copybuf.data[m_num_copy_ghosts[dir]] = h_pos.data[idx];
                    if (flags[comm_flag::charge]) h_charge_copybuf.data[m_num_copy_ghosts[dir]] = h_charge.data[idx];
                    if (flags[comm_flag::diameter]) h_diameter_copybuf.data[m_num_copy_ghosts[dir]] = h_diameter.data[idx];
                    if (flags[comm_flag::body]) h_body_copybuf.data[m_num_copy_ghosts[dir]] = h_body.data[idx];
                    if (flags[comm_flag::image]) h_image_copybuf.data[m_num_copy_ghosts[dir]] = h_image.data[idx];
                    if (flags[comm_flag::velocity]) h_velocity_copybuf.data[m_num_copy_ghosts[dir]] = h_vel.data[idx];
                    if (flags[comm_flag::orientation]) h_orientation_copybuf.data[m_num_copy_ghosts[dir]] = h_orientation.data[idx];
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

        m_reqs.clear();
        m_stats.clear();
        MPI_Request req;

        MPI_Isend(&m_num_copy_ghosts[dir],
            sizeof(unsigned int),
            MPI_BYTE,
            send_neighbor,
            0,
            m_mpi_comm,
            &req);
        m_reqs.push_back(req);
        MPI_Irecv(&m_num_recv_ghosts[dir],
            sizeof(unsigned int),
            MPI_BYTE,
            recv_neighbor,
            0,
            m_mpi_comm,
            &req);
        m_reqs.push_back(req);

        m_stats.resize(2);
        MPI_Waitall(m_reqs.size(), &m_reqs.front(), &m_stats.front());

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
            {
            m_prof->push("MPI send/recv");
            }

            {
            ArrayHandle<unsigned int> h_copy_ghosts(m_copy_ghosts[dir], access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_plan_copybuf(m_plan_copybuf, access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_pos_copybuf(m_pos_copybuf, access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_charge_copybuf(m_charge_copybuf, access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_diameter_copybuf(m_diameter_copybuf, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_body_copybuf(m_body_copybuf, access_location::host, access_mode::read);
            ArrayHandle<int3> h_image_copybuf(m_image_copybuf, access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_velocity_copybuf(m_velocity_copybuf, access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_orientation_copybuf(m_orientation_copybuf, access_location::host, access_mode::read);

            ArrayHandle<unsigned int> h_plan(m_plan, access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar> h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::readwrite);
            ArrayHandle<unsigned int> h_body(m_pdata->getBodies(), access_location::host, access_mode::readwrite);
            ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);
            ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::readwrite);

            // Clear out the mpi variables for new statuses and requests
            m_reqs.clear();
            m_stats.clear();

            MPI_Isend(h_plan_copybuf.data,
                m_num_copy_ghosts[dir]*sizeof(unsigned int),
                MPI_BYTE,
                send_neighbor,
                1,
                m_mpi_comm,
                &req);
            m_reqs.push_back(req);
            MPI_Irecv(h_plan.data + start_idx,
                m_num_recv_ghosts[dir]*sizeof(unsigned int),
                MPI_BYTE,
                recv_neighbor,
                1,
                m_mpi_comm,
                &req);
            m_reqs.push_back(req);

            MPI_Isend(h_copy_ghosts.data,
                m_num_copy_ghosts[dir]*sizeof(unsigned int),
                MPI_BYTE,
                send_neighbor,
                2,
                m_mpi_comm,
                &req);
            m_reqs.push_back(req);
            MPI_Irecv(h_tag.data + start_idx,
                m_num_recv_ghosts[dir]*sizeof(unsigned int),
                MPI_BYTE,
                recv_neighbor,
                2,
                m_mpi_comm,
                &req);
            m_reqs.push_back(req);

            if (flags[comm_flag::position])
                {
                MPI_Isend(h_pos_copybuf.data,
                    m_num_copy_ghosts[dir]*sizeof(Scalar4),
                    MPI_BYTE,
                    send_neighbor,
                    3,
                    m_mpi_comm,
                    &req);
                m_reqs.push_back(req);
                MPI_Irecv(h_pos.data + start_idx,
                    m_num_recv_ghosts[dir]*sizeof(Scalar4),
                    MPI_BYTE,
                    recv_neighbor,
                    3,
                    m_mpi_comm,
                    &req);
                m_reqs.push_back(req);
                }

            if (flags[comm_flag::charge])
                {
                MPI_Isend(h_charge_copybuf.data,
                    m_num_copy_ghosts[dir]*sizeof(Scalar),
                    MPI_BYTE,
                    send_neighbor,
                    4,
                    m_mpi_comm,
                    &req);
                m_reqs.push_back(req);
                MPI_Irecv(h_charge.data + start_idx,
                    m_num_recv_ghosts[dir]*sizeof(Scalar),
                    MPI_BYTE,
                    recv_neighbor,
                    4,
                    m_mpi_comm,
                    &req);
                m_reqs.push_back(req);
                }

            if (flags[comm_flag::diameter])
                {
                MPI_Isend(h_diameter_copybuf.data,
                    m_num_copy_ghosts[dir]*sizeof(Scalar),
                    MPI_BYTE,
                    send_neighbor,
                    5,
                    m_mpi_comm,
                    &req);
                m_reqs.push_back(req);
                MPI_Irecv(h_diameter.data + start_idx,
                    m_num_recv_ghosts[dir]*sizeof(Scalar),
                    MPI_BYTE,
                    recv_neighbor,
                    5,
                    m_mpi_comm,
                    &req);
                m_reqs.push_back(req);
                }

            if (flags[comm_flag::velocity])
                {
                MPI_Isend(h_velocity_copybuf.data,
                    m_num_copy_ghosts[dir]*sizeof(Scalar4),
                    MPI_BYTE,
                    send_neighbor,
                    6,
                    m_mpi_comm,
                    &req);
                m_reqs.push_back(req);
                MPI_Irecv(h_vel.data + start_idx,
                    m_num_recv_ghosts[dir]*sizeof(Scalar4),
                    MPI_BYTE,
                    recv_neighbor,
                    6,
                    m_mpi_comm,
                    &req);
                m_reqs.push_back(req);
                }


            if (flags[comm_flag::orientation])
                {
                MPI_Isend(h_orientation_copybuf.data,
                    m_num_copy_ghosts[dir]*sizeof(Scalar4),
                    MPI_BYTE,
                    send_neighbor,
                    7,
                    m_mpi_comm,
                    &req);
                m_reqs.push_back(req);
                MPI_Irecv(h_orientation.data + start_idx,
                    m_num_recv_ghosts[dir]*sizeof(Scalar4),
                    MPI_BYTE,
                    recv_neighbor,
                    7,
                    m_mpi_comm,
                    &req);
                m_reqs.push_back(req);
                }

            if (flags[comm_flag::body])
                {
                MPI_Isend(h_body_copybuf.data,
                    m_num_copy_ghosts[dir]*sizeof(unsigned int),
                    MPI_BYTE,
                    send_neighbor,
                    8,
                    m_mpi_comm,
                    &req);
                m_reqs.push_back(req);
                MPI_Irecv(h_body.data + start_idx,
                    m_num_recv_ghosts[dir]*sizeof(unsigned int),
                    MPI_BYTE,
                    recv_neighbor,
                    8,
                    m_mpi_comm,
                    &req);
                m_reqs.push_back(req);
                }

            if (flags[comm_flag::image])
                {
                MPI_Isend(h_image_copybuf.data,
                    m_num_copy_ghosts[dir]*sizeof(int3),
                    MPI_BYTE,
                    send_neighbor,
                    9,
                    m_mpi_comm,
                    &req);
                m_reqs.push_back(req);
                MPI_Irecv(h_image.data + start_idx,
                    m_num_recv_ghosts[dir]*sizeof(int3),
                    MPI_BYTE,
                    recv_neighbor,
                    9,
                    m_mpi_comm,
                    &req);
                m_reqs.push_back(req);
                }

            m_stats.resize(m_reqs.size());
            MPI_Waitall(m_reqs.size(), &m_reqs.front(), &m_stats.front());
            }

        if (m_prof)
            m_prof->pop();

        // wrap particle positions
        if (flags[comm_flag::position])
            {
            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
            ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);

            const BoxDim shifted_box = getShiftedBox();

            for (unsigned int idx = start_idx; idx < start_idx + m_num_recv_ghosts[dir]; idx++)
                {
                Scalar4& pos = h_pos.data[idx];

                // wrap particles received across a global boundary
                int3& img = h_image.data[idx];
                shifted_box.wrap(pos,img);
                }
            }

            {
            // set reverse-lookup tag -> idx
            ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::readwrite);

            for (unsigned int idx = start_idx; idx < start_idx + m_num_recv_ghosts[dir]; idx++)
                {
                assert(h_tag.data[idx] <= m_pdata->getMaximumTag());
                assert(h_rtag.data[h_tag.data[idx]] == NOT_LOCAL);
                h_rtag.data[h_tag.data[idx]] = idx;
                }

            }
        } // end dir loop

    m_ghosts_added = m_pdata->getNGhosts();

    // exchange ghost constraints along with ghost particles
    m_constraint_comm.exchangeGhostGroups(m_plan, mask);

    m_last_flags = flags;

    /***********************************************************************************************************************************************************
     * For multi-body force fields we must allow particles to send information back through their ghosts.
     * For this purpose, we implement a system for ghosts to be sent back to their original domain with forces on them that can then be added back to the original local particle.
     * In exchangeGhosts, this involves the construction of reverse plans and then the sending of ghosts back to the domains in which they originated
     **********************************************************************************************************************************************************/

    // Need to check the flag that determines whether reverse net forces are set
    if (flags[comm_flag::reverse_net_force])
        {
        // Set some initial constants that won't change and can be used in all scopes
        unsigned int n_reverse_ghosts_recv = 0;
        unsigned int n_ghosts_init = m_pdata->getNGhosts();
        unsigned int n_local = m_pdata->getN();

        // resize and reset plans
        m_plan_reverse.resize(n_ghosts_init);
        m_tag_reverse.resize(n_ghosts_init);

            {
            ArrayHandle<unsigned int> h_plan_reverse(m_plan_reverse, access_location::host, access_mode::readwrite);

            for (unsigned int i = 0; i < n_ghosts_init; ++i)
                {
                // Unnecessary for tags since those will be overwritten rather than added to
                h_plan_reverse.data[i] = 0;
                }
            }

            // Invert the plans to construct the reverse plans
            {
            ArrayHandle<unsigned int> h_plan_reverse(m_plan_reverse, access_location::host, access_mode::readwrite);
            ArrayHandle<unsigned int> h_plan(m_plan, access_location::host, access_mode::read);

            // Determine the plans for each ghost
            for (unsigned int idx = 0; idx < n_ghosts_init; idx++)
                {
                // Invert the plans directly rather than computing anything about ghost layers
                unsigned int ghost_idx = n_local + idx;
                if (h_plan.data[ghost_idx] & send_east)
                    h_plan_reverse.data[idx] |= send_west;

                if (h_plan.data[ghost_idx] & send_west)
                    h_plan_reverse.data[idx] |= send_east;

                if (h_plan.data[ghost_idx] & send_north)
                    h_plan_reverse.data[idx] |= send_south;

                if (h_plan.data[ghost_idx] & send_south)
                    h_plan_reverse.data[idx] |= send_north;

                if (h_plan.data[ghost_idx] & send_up)
                    h_plan_reverse.data[idx] |= send_down;

                if (h_plan.data[ghost_idx] & send_down)
                    h_plan_reverse.data[idx] |= send_up;
                }
            }

        // Loop over all directions and send ghosts back if that direction is part of their plan
        for (unsigned int dir = 0; dir < 6; dir++)
        {
            if (! isCommunicating(dir) ) continue;

            m_num_copy_local_ghosts_reverse[dir] = 0;
            m_num_forward_ghosts_reverse[dir] = 0;

            // resize buffers
            unsigned int max_copy_ghosts = n_ghosts_init + n_reverse_ghosts_recv;
            m_copy_ghosts_reverse[dir].resize(max_copy_ghosts);
            m_plan_reverse_copybuf[dir].resize(max_copy_ghosts);
            m_forward_ghosts_reverse[dir].resize(max_copy_ghosts);

            // Determine which ghosts need to be forwarded
                {
                ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
                ArrayHandle<unsigned int> h_plan_reverse(m_plan_reverse, access_location::host, access_mode::read);
                ArrayHandle<unsigned int> h_copy_ghosts_reverse(m_copy_ghosts_reverse[dir], access_location::host, access_mode::overwrite);
                ArrayHandle<unsigned int> h_plan_reverse_copybuf(m_plan_reverse_copybuf[dir], access_location::host, access_mode::overwrite);

                // First check the set of local ghost particles to see if any of those need to get sent back
                for (unsigned int idx = 0; idx < m_pdata->getNGhosts(); idx++)
                    {
                    // only send once, namely in the direction with lowest index (by convention)
                    if ((h_plan_reverse.data[idx] & (1 << dir)))
                        {
                        h_plan_reverse_copybuf.data[m_num_copy_local_ghosts_reverse[dir]] = h_plan_reverse.data[idx];
                        h_copy_ghosts_reverse.data[m_num_copy_local_ghosts_reverse[dir]] = h_tag.data[n_local + idx];
                        m_num_copy_local_ghosts_reverse[dir]++;
                        }
                    }

                // Now check if any particles that were sent as reverse ghosts to this need to be sent back further
                ArrayHandle<unsigned int> h_forward_ghosts_reverse(m_forward_ghosts_reverse[dir], access_location::host, access_mode::overwrite);
                ArrayHandle<unsigned int> h_tag_reverse(m_tag_reverse, access_location::host, access_mode::read);

                unsigned int add_idx = m_num_copy_local_ghosts_reverse[dir];
                unsigned int num_ghosts_local = n_ghosts_init;
                for (unsigned int idx = 0; idx < n_reverse_ghosts_recv; ++idx)
                    {
                    if (h_plan_reverse.data[num_ghosts_local + idx] & (1 << dir))
                        {
                        h_plan_reverse_copybuf.data[add_idx + m_num_forward_ghosts_reverse[dir]] = h_plan_reverse.data[num_ghosts_local + idx];
                        h_copy_ghosts_reverse.data[add_idx + m_num_forward_ghosts_reverse[dir]] = h_tag_reverse.data[idx];
                        h_forward_ghosts_reverse.data[m_num_forward_ghosts_reverse[dir]] = idx;
                        m_num_forward_ghosts_reverse[dir]++;
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
            m_reqs.clear();
            m_stats.clear();
            MPI_Request req;

            // Communicate the number of ghosts to forward. We keep separate counts of the local ghosts we are forwarding for the first time and the ghosts that were forwarded to this domain that are being forwarded further
            MPI_Isend(&m_num_forward_ghosts_reverse[dir],
                    sizeof(unsigned int),
                    MPI_BYTE,
                    send_neighbor,
                    0,
                    m_mpi_comm,
                    &req);
            m_reqs.push_back(req);
            MPI_Irecv(&m_num_recv_forward_ghosts_reverse[dir],
                    sizeof(unsigned int),
                    MPI_BYTE,
                    recv_neighbor,
                    0,
                    m_mpi_comm,
                    &req);
            m_reqs.push_back(req);
            MPI_Isend(&m_num_copy_local_ghosts_reverse[dir],
                    sizeof(unsigned int),
                    MPI_BYTE,
                    send_neighbor,
                    1,
                    m_mpi_comm,
                    &req);
            m_reqs.push_back(req);
            MPI_Irecv(&m_num_recv_local_ghosts_reverse[dir],
                    sizeof(unsigned int),
                    MPI_BYTE,
                    recv_neighbor,
                    1,
                    m_mpi_comm,
                    &req);
            m_reqs.push_back(req);

            m_stats.resize(m_reqs.size());
            MPI_Waitall(m_reqs.size(), &m_reqs.front(), &m_stats.front());

            if (m_prof)
                m_prof->pop();

            // append ghosts at the end of particle data array
            unsigned int start_idx_plan = n_ghosts_init + n_reverse_ghosts_recv;
            unsigned int start_idx_tag = n_reverse_ghosts_recv;

            // resize arrays
            unsigned int n_new_ghosts_send = m_num_copy_local_ghosts_reverse[dir]+m_num_forward_ghosts_reverse[dir];
            unsigned int n_new_ghosts_recv = m_num_recv_local_ghosts_reverse[dir]+m_num_recv_forward_ghosts_reverse[dir];
            n_reverse_ghosts_recv += n_new_ghosts_recv;
            m_plan_reverse.resize(n_ghosts_init + n_reverse_ghosts_recv);
            m_tag_reverse.resize(n_reverse_ghosts_recv);

            // exchange particle data, write directly to the particle data arrays
            if (m_prof)
                m_prof->push("MPI send/recv");

            // Now forward the ghosts
            {
                ArrayHandle<unsigned int> h_plan_reverse_copybuf(m_plan_reverse_copybuf[dir], access_location::host, access_mode::read);
                ArrayHandle<unsigned int> h_plan_reverse(m_plan_reverse, access_location::host, access_mode::read);

                ArrayHandle<unsigned int> h_copy_ghosts_reverse(m_copy_ghosts_reverse[dir], access_location::host, access_mode::read);
                ArrayHandle<unsigned int> h_tag_reverse(m_tag_reverse, access_location::host, access_mode::readwrite);

                // Clear out the mpi variables for new statuses and requests
                m_reqs.clear();
                m_stats.clear();

                MPI_Isend(h_plan_reverse_copybuf.data,
                        (n_new_ghosts_send )*sizeof(unsigned int),
                        MPI_BYTE,
                        send_neighbor,
                        2,
                        m_mpi_comm,
                        &req);
                m_reqs.push_back(req);
                MPI_Irecv(h_plan_reverse.data + start_idx_plan,
                        (n_new_ghosts_recv)*sizeof(unsigned int),
                        MPI_BYTE,
                        recv_neighbor,
                        2,
                        m_mpi_comm,
                        &req);
                m_reqs.push_back(req);

                MPI_Isend(h_copy_ghosts_reverse.data,
                        (n_new_ghosts_send)*sizeof(unsigned int),
                        MPI_BYTE,
                        send_neighbor,
                        3,
                        m_mpi_comm,
                        &req);
                m_reqs.push_back(req);
                MPI_Irecv(h_tag_reverse.data + start_idx_tag,
                        (n_new_ghosts_recv)*sizeof(unsigned int),
                        MPI_BYTE,
                        recv_neighbor,
                        3,
                        m_mpi_comm,
                        &req);
                m_reqs.push_back(req);

                m_stats.resize(m_reqs.size());
                MPI_Waitall(m_reqs.size(), &m_reqs.front(), &m_stats.front());
            }

            if (m_prof)
                m_prof->pop();

        } // end dir loop
    }

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

                // copy velocity into send buffer
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
        // charge, body, image and diameter are not updated between neighbor list builds
        if (flags[comm_flag::position])
            {
            m_reqs.resize(2);
            m_stats.resize(2);

            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_pos_copybuf(m_pos_copybuf, access_location::host, access_mode::read);

            // exchange particle data, write directly to the particle data arrays
            MPI_Isend(h_pos_copybuf.data, m_num_copy_ghosts[dir]*sizeof(Scalar4), MPI_BYTE, send_neighbor, 1, m_mpi_comm, &m_reqs[0]);
            MPI_Irecv(h_pos.data + start_idx, m_num_recv_ghosts[dir]*sizeof(Scalar4), MPI_BYTE, recv_neighbor, 1, m_mpi_comm, &m_reqs[1]);
            MPI_Waitall(2, &m_reqs.front(), &m_stats.front());

            sz += sizeof(Scalar4);
            }

        if (flags[comm_flag::velocity])
            {
            m_reqs.resize(2);
            m_stats.resize(2);

            ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_vel_copybuf(m_velocity_copybuf, access_location::host, access_mode::read);

            // exchange particle data, write directly to the particle data arrays
            MPI_Isend(h_vel_copybuf.data, m_num_copy_ghosts[dir]*sizeof(Scalar4), MPI_BYTE, send_neighbor, 2, m_mpi_comm, &m_reqs[0]);
            MPI_Irecv(h_vel.data + start_idx, m_num_recv_ghosts[dir]*sizeof(Scalar4), MPI_BYTE, recv_neighbor, 2, m_mpi_comm, &m_reqs[1]);
            MPI_Waitall(2, &m_reqs.front(), &m_stats.front());

            sz += sizeof(Scalar4);
            }

        if (flags[comm_flag::orientation])
            {
            m_reqs.resize(2);
            m_stats.resize(2);

            ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_orientation_copybuf(m_orientation_copybuf, access_location::host, access_mode::read);

            // exchange particle data, write directly to the particle data arrays
            MPI_Isend(h_orientation_copybuf.data, m_num_copy_ghosts[dir]*sizeof(Scalar4), MPI_BYTE, send_neighbor, 3, m_mpi_comm, &m_reqs[0]);
            MPI_Irecv(h_orientation.data + start_idx, m_num_recv_ghosts[dir]*sizeof(Scalar4), MPI_BYTE, recv_neighbor, 3, m_mpi_comm, &m_reqs[1]);
            MPI_Waitall(2, &m_reqs.front(), &m_stats.front());

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

void Communicator::updateNetForce(unsigned int timestep)
    {
    CommFlags flags = getFlags();
    if (! flags[comm_flag::net_force] && ! flags[comm_flag::reverse_net_force] && ! flags[comm_flag::net_torque] && ! flags[comm_flag::net_virial])
        return;

    // we have a current m_copy_ghosts list which contain the indices of particles
    // to send to neighboring processors
    if (m_prof)
        m_prof->push("comm_ghost_net_force");

    std::ostringstream oss;
    oss << "Communicator: update net ";
    if (flags[comm_flag::net_force])
        {
        oss << "force ";
        }
    if (flags[comm_flag::reverse_net_force])
        {
        oss << "reverse force ";
        }
    if (flags[comm_flag::net_torque])
        {
        oss << "torque ";
        }
    if (flags[comm_flag::net_virial])
        {
        oss << "virial";
        }

    m_exec_conf->msg->notice(7) << oss.str() << std::endl;

    // Set some global counters
    unsigned int num_tot_recv_ghosts = 0; // total number of ghosts received
    unsigned int num_tot_recv_ghosts_reverse = 0; // total number of ghosts received in reverse direction

    // clear data in these arrays
    if (flags[comm_flag::net_force])
        m_netforce_copybuf.clear();

    if (flags[comm_flag::reverse_net_force])
        {
        m_netforce_reverse_copybuf.clear();
        }

    if (flags[comm_flag::net_torque])
        {
        m_nettorque_copybuf.clear();
        }

    if (flags[comm_flag::net_virial])
        {
        m_netvirial_copybuf.clear();
        }

    // update data in these arrays

    for (unsigned int dir = 0; dir < 6; dir ++)
        {
        if (! isCommunicating(dir) ) continue;

        // resize send buffers as needed
        unsigned int old_size;
        if (flags[comm_flag::net_force])
            {
            old_size = m_netforce_copybuf.size();
            m_netforce_copybuf.resize(old_size+m_num_copy_ghosts[dir]);
            }

        if (flags[comm_flag::reverse_net_force])
            {
            old_size = m_netforce_reverse_copybuf.size();
            m_netforce_reverse_copybuf.resize(old_size + m_num_forward_ghosts_reverse[dir] + m_num_copy_local_ghosts_reverse[dir]);
            }

        if (flags[comm_flag::net_torque])
            {
            old_size = m_nettorque_copybuf.size();
            m_nettorque_copybuf.resize(old_size+m_num_copy_ghosts[dir]);
            }

        if (flags[comm_flag::net_virial])
            {
            old_size = m_netvirial_copybuf.size();
            m_netvirial_copybuf.resize(old_size+6*m_num_copy_ghosts[dir]);
            }

        // Copy data into send buffers
        if (flags[comm_flag::net_force])
            {
            ArrayHandle<Scalar4> h_netforce(m_pdata->getNetForce(), access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_netforce_copybuf(m_netforce_copybuf, access_location::host, access_mode::overwrite);
            ArrayHandle<unsigned int> h_copy_ghosts(m_copy_ghosts[dir], access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

            // copy net forces of ghost particles
            for (unsigned int ghost_idx = 0; ghost_idx < m_num_copy_ghosts[dir]; ghost_idx++)
                {
                unsigned int idx = h_rtag.data[h_copy_ghosts.data[ghost_idx]];

                assert(idx < m_pdata->getN() + m_pdata->getNGhosts());

                // copy net force into send buffer
                h_netforce_copybuf.data[ghost_idx] = h_netforce.data[idx];
                }
            }

        if (flags[comm_flag::reverse_net_force])
            {
            ArrayHandle<unsigned int> h_copy_ghosts_reverse(m_copy_ghosts_reverse[dir], access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_netforce_reverse_copybuf(m_netforce_reverse_copybuf, access_location::host, access_mode::overwrite);
            ArrayHandle<Scalar4> h_netforce_reverse_recvbuf(m_netforce_reverse_recvbuf, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_forward_ghosts_reverse(m_forward_ghosts_reverse[dir], access_location::host, access_mode::overwrite);
            ArrayHandle<Scalar4> h_netforce(m_pdata->getNetForce(), access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

            // copy reverse net force of ghost particles
            for (unsigned int ghost_idx = 0; ghost_idx < m_num_copy_local_ghosts_reverse[dir]; ghost_idx++)
                {
                unsigned int idx = h_rtag.data[h_copy_ghosts_reverse.data[ghost_idx]];

                assert(idx < m_pdata->getN() + m_pdata->getNGhosts());

                // copy reverse net force into send buffer
                h_netforce_reverse_copybuf.data[ghost_idx] = h_netforce.data[idx];
                }

            // Scan the entire recv buf for additional particles. These are forces corresponding to ghosts forwarded to this domain
            for (unsigned int i = 0; i < m_num_forward_ghosts_reverse[dir]; ++i)
                {
                unsigned int idx = h_forward_ghosts_reverse.data[i];
                h_netforce_reverse_copybuf.data[m_num_copy_local_ghosts_reverse[dir]+i] = h_netforce_reverse_recvbuf.data[idx];
                }
            }

        if (flags[comm_flag::net_torque])
            {
            ArrayHandle<Scalar4> h_nettorque(m_pdata->getNetTorqueArray(), access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_nettorque_copybuf(m_nettorque_copybuf, access_location::host, access_mode::overwrite);
            ArrayHandle<unsigned int> h_copy_ghosts(m_copy_ghosts[dir], access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

            // copy net torques of ghost particles
            for (unsigned int ghost_idx = 0; ghost_idx < m_num_copy_ghosts[dir]; ghost_idx++)
                {
                unsigned int idx = h_rtag.data[h_copy_ghosts.data[ghost_idx]];

                assert(idx < m_pdata->getN() + m_pdata->getNGhosts());

                // copy net force into send buffer
                h_nettorque_copybuf.data[ghost_idx] = h_nettorque.data[idx];
                }
            }
        if (flags[comm_flag::net_virial])
            {
            ArrayHandle<Scalar> h_netvirial(m_pdata->getNetVirial(), access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_netvirial_copybuf(m_netvirial_copybuf, access_location::host, access_mode::overwrite);
            ArrayHandle<unsigned int> h_copy_ghosts(m_copy_ghosts[dir], access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

            unsigned int pitch = m_pdata->getNetVirial().getPitch();

            // copy net torques of ghost particles
            for (unsigned int ghost_idx = 0; ghost_idx < m_num_copy_ghosts[dir]; ghost_idx++)
                {
                unsigned int idx = h_rtag.data[h_copy_ghosts.data[ghost_idx]];

                assert(idx < m_pdata->getN() + m_pdata->getNGhosts());

                // copy net force into send buffer, transposing
                h_netvirial_copybuf.data[6*ghost_idx+0] = h_netvirial.data[0*pitch+idx];
                h_netvirial_copybuf.data[6*ghost_idx+1] = h_netvirial.data[1*pitch+idx];
                h_netvirial_copybuf.data[6*ghost_idx+2] = h_netvirial.data[2*pitch+idx];
                h_netvirial_copybuf.data[6*ghost_idx+3] = h_netvirial.data[3*pitch+idx];
                h_netvirial_copybuf.data[6*ghost_idx+4] = h_netvirial.data[4*pitch+idx];
                h_netvirial_copybuf.data[6*ghost_idx+5] = h_netvirial.data[5*pitch+idx];
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
        if (flags[comm_flag::net_force])
            {
            m_reqs.clear();
            m_stats.clear();

            ArrayHandle<Scalar4> h_netforce(m_pdata->getNetForce(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_netforce_copybuf(m_netforce_copybuf, access_location::host, access_mode::read);

            // exchange particle data, write directly to the particle data arrays
            MPI_Isend(h_netforce_copybuf.data, m_num_copy_ghosts[dir]*sizeof(Scalar4), MPI_BYTE, send_neighbor, 1, m_mpi_comm, &m_reqs[0]);
            MPI_Irecv(h_netforce.data + start_idx, m_num_recv_ghosts[dir]*sizeof(Scalar4), MPI_BYTE, recv_neighbor, 1, m_mpi_comm, &m_reqs[1]);
            MPI_Waitall(2, &m_reqs.front(), &m_stats.front());

            sz += sizeof(Scalar4);
            }

        // We add new particle data for reverse ghosts after the particle data already received, so save the count and then update it with the new receive count
        unsigned int start_idx_reverse = num_tot_recv_ghosts_reverse;
        num_tot_recv_ghosts_reverse += m_num_recv_local_ghosts_reverse[dir] + m_num_recv_forward_ghosts_reverse[dir];

        // Send the net forces from ghosts traveling back and then add them to the original local particles the ghosts corresponded to
        if (flags[comm_flag::reverse_net_force])
            {
            m_netforce_reverse_recvbuf.resize(num_tot_recv_ghosts_reverse);
                {
                m_reqs.resize(2);
                m_stats.resize(2);

                // use separate recv buffer
                ArrayHandle<Scalar4> h_netforce_reverse_copybuf(m_netforce_reverse_copybuf, access_location::host, access_mode::read);
                ArrayHandle<Scalar4> h_netforce_reverse_recvbuf(m_netforce_reverse_recvbuf, access_location::host, access_mode::readwrite);

                MPI_Isend(h_netforce_reverse_copybuf.data, (m_num_copy_local_ghosts_reverse[dir] + m_num_forward_ghosts_reverse[dir])*sizeof(Scalar4), MPI_BYTE, send_neighbor, 2, m_mpi_comm, &m_reqs[0]);
                MPI_Irecv(h_netforce_reverse_recvbuf.data + start_idx_reverse, (m_num_recv_local_ghosts_reverse[dir] + m_num_recv_forward_ghosts_reverse[dir])*sizeof(Scalar4), MPI_BYTE, recv_neighbor, 2, m_mpi_comm, &m_reqs[1]);
                MPI_Waitall(2, &m_reqs.front(), &m_stats.front());

                sz += sizeof(Scalar4);
                }

            // Add forces
            ArrayHandle<Scalar4> h_netforce(m_pdata->getNetForce(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_netforce_reverse_recvbuf(m_netforce_reverse_recvbuf, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_tag_reverse(m_tag_reverse, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

            unsigned int n_local_particles = m_pdata->getN();
            for(unsigned int i = 0; i < m_num_recv_forward_ghosts_reverse[dir] + m_num_recv_local_ghosts_reverse[dir]; i++)
                {
                unsigned int idx = h_rtag.data[h_tag_reverse.data[start_idx_reverse + i]];
                if (idx < n_local_particles)
                    {
                    Scalar4 f = h_netforce_reverse_recvbuf.data[start_idx_reverse + i];
                    Scalar4 cur_F = h_netforce.data[idx];

                    // add net force to particle data
                    cur_F.x += f.x;
                    cur_F.y += f.y;
                    cur_F.z += f.z;
                    cur_F.w += f.w;
                    h_netforce.data[idx] = cur_F;
                    }
                }
            }

        if (flags[comm_flag::net_torque])
            {
            m_reqs.resize(2);
            m_stats.resize(2);

            ArrayHandle<Scalar4> h_nettorque(m_pdata->getNetTorqueArray(), access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_nettorque_copybuf(m_nettorque_copybuf, access_location::host, access_mode::read);

            MPI_Isend(h_nettorque_copybuf.data, m_num_copy_ghosts[dir]*sizeof(Scalar4), MPI_BYTE, send_neighbor, 2, m_mpi_comm, &m_reqs[0]);
            MPI_Irecv(h_nettorque.data + start_idx, m_num_recv_ghosts[dir]*sizeof(Scalar4), MPI_BYTE, recv_neighbor, 2, m_mpi_comm, &m_reqs[1]);
            MPI_Waitall(2, &m_reqs.front(), &m_stats.front());

            sz += sizeof(Scalar4);
            }

        if (flags[comm_flag::net_virial])
            {
            m_netvirial_recvbuf.resize(6*m_num_recv_ghosts[dir]);
            m_reqs.resize(2);
            m_stats.resize(2);

            ArrayHandle<Scalar> h_netvirial_recvbuf(m_netvirial_recvbuf, access_location::host, access_mode::overwrite);
            ArrayHandle<Scalar> h_netvirial_copybuf(m_netvirial_copybuf, access_location::host, access_mode::read);

            MPI_Isend(h_netvirial_copybuf.data, 6*m_num_copy_ghosts[dir]*sizeof(Scalar), MPI_BYTE, send_neighbor, 3, m_mpi_comm, &m_reqs[0]);
            MPI_Irecv(h_netvirial_recvbuf.data, 6*m_num_recv_ghosts[dir]*sizeof(Scalar), MPI_BYTE, recv_neighbor, 3, m_mpi_comm, &m_reqs[1]);
            MPI_Waitall(2, &m_reqs.front(), &m_stats.front());

            sz += 6*sizeof(Scalar);
            }

        if (m_prof)
            m_prof->pop(0, (m_num_recv_ghosts[dir]+m_num_copy_ghosts[dir])*sz);

        if (flags[comm_flag::net_virial])
            {
            unsigned int pitch = m_pdata->getNetVirial().getPitch();

            // unpack virial
            ArrayHandle<Scalar> h_netvirial_recvbuf(m_netvirial_recvbuf, access_location::host, access_mode::read);
            ArrayHandle<Scalar> h_netvirial(m_pdata->getNetVirial(), access_location::host, access_mode::read);

            for (unsigned int i = 0; i < m_num_recv_ghosts[dir]; ++i)
                {
                h_netvirial.data[0*pitch+start_idx+i] = h_netvirial_recvbuf.data[6*i+0];
                h_netvirial.data[1*pitch+start_idx+i] = h_netvirial_recvbuf.data[6*i+1];
                h_netvirial.data[2*pitch+start_idx+i] = h_netvirial_recvbuf.data[6*i+2];
                h_netvirial.data[3*pitch+start_idx+i] = h_netvirial_recvbuf.data[6*i+3];
                h_netvirial.data[4*pitch+start_idx+i] = h_netvirial_recvbuf.data[6*i+4];
                h_netvirial.data[5*pitch+start_idx+i] = h_netvirial_recvbuf.data[6*i+5];
                }
            }
        } // end dir loop

        if (m_prof)
            m_prof->pop();
    }


void Communicator::removeGhostParticleTags()
    {
    // wipe out reverse-lookup tag -> idx for old ghost atoms
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::readwrite);

    m_exec_conf->msg->notice(9) << "Communicator: removing " << m_ghosts_added << " ghost particles " << std::endl;

    for (unsigned int i = 0; i < m_ghosts_added; i++)
        {
        unsigned int idx = m_pdata->getN() + i;
        h_rtag.data[h_tag.data[idx]] = NOT_LOCAL;
        }

    m_ghosts_added = 0;
    }

const BoxDim Communicator::getShiftedBox() const
    {
    // construct the shifted global box for applying global boundary conditions
    BoxDim shifted_box = m_pdata->getGlobalBox();
    Scalar3 f= make_scalar3(0.5,0.5,0.5);

    /* As was done before, shift the global box by half the size of the domain that you received from.
     * This guarantees that any ghosts that could have been sent are wrapped back in because the smallest size a domain
     * can be is 2*getGhostLayerMaxWidth(). Because the domains and the global box have the same triclinic skew, we can
     * just use the fractional widths rather than using getNearestPlaneDistance().
     */
    uint3 grid_pos = m_decomposition->getGridPos();
    const Index3D& di = m_decomposition->getDomainIndexer();
    Scalar tol = 0.0001;
    for (unsigned int dir = 0; dir < 6; dir ++)
        {
        if (m_decomposition->isAtBoundary(dir) &&  isCommunicating(dir))
            {
            if (dir == face_east)
                {
                int neigh = (int)grid_pos.x + 1;
                if (neigh == (int)di.getW()) neigh = 0;
                Scalar shift = Scalar(0.5)*(m_decomposition->getCumulativeFraction(0, neigh+1) - m_decomposition->getCumulativeFraction(0, neigh));
                f.x += (shift + tol);
                }
            else if (dir == face_west)
                {
                int neigh = (int)grid_pos.x - 1;
                if (neigh == -1) neigh = di.getW() - 1;
                Scalar shift = Scalar(0.5)*(m_decomposition->getCumulativeFraction(0, neigh+1) - m_decomposition->getCumulativeFraction(0, neigh));
                f.x -= (shift + tol);
                }
            else if (dir == face_north)
                {
                int neigh = (int)grid_pos.y + 1;
                if (neigh == (int)di.getH()) neigh = 0;
                Scalar shift = Scalar(0.5)*(m_decomposition->getCumulativeFraction(1, neigh+1) - m_decomposition->getCumulativeFraction(1, neigh));
                f.y += (shift + tol);
                }
            else if (dir == face_south)
                {
                int neigh = (int)grid_pos.y - 1;
                if (neigh == -1) neigh = di.getH() - 1;
                Scalar shift = Scalar(0.5)*(m_decomposition->getCumulativeFraction(1, neigh+1) - m_decomposition->getCumulativeFraction(1, neigh));
                f.y -= (shift + tol);
                }
            else if (dir == face_up)
                {
                int neigh = (int)grid_pos.z + 1;
                if (neigh == (int)di.getD()) neigh = 0;
                Scalar shift = Scalar(0.5)*(m_decomposition->getCumulativeFraction(2, neigh+1) - m_decomposition->getCumulativeFraction(2, neigh));
                f.z += (shift + tol);
                }
            else if (dir == face_down)
                {
                int neigh = (int)grid_pos.z - 1;
                if (neigh == -1) neigh = di.getD() - 1;
                Scalar shift = Scalar(0.5)*(m_decomposition->getCumulativeFraction(2, neigh+1) - m_decomposition->getCumulativeFraction(2, neigh));
                f.z -= (shift + tol);
                }
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
void export_Communicator(py::module& m)
    {
    py::class_<Communicator, std::shared_ptr<Communicator> >(m,"Communicator")
    .def(py::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<DomainDecomposition> >());
    }
#endif // ENABLE_MPI
