// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file MeshGroupData.h
    \brief Defines implementation of MeshGroupData
 */

#include "MeshGroupData.h"
#include "BondedGroupData.h"
#include "Index1D.h"
#include "ParticleData.h"

#include <pybind11/numpy.h>

#ifdef ENABLE_HIP
#include "CachedAllocator.h"
#include "MeshGroupData.cuh"
#endif

using namespace std;

namespace hoomd
    {
/*
 * Implementation of MeshGroupData methods
 */

/*! \param pdata The particle data to associate with
    \param n_group_types Number of bonded group types to initialize
 */
template<unsigned int group_size, typename Group, const char* name, typename snap, bool bond>
MeshGroupData<group_size, Group, name, snap, bond>::MeshGroupData(
    std::shared_ptr<ParticleData> pdata,
    unsigned int n_group_types)
    : BondedGroupData<group_size, Group, name, true>(pdata, n_group_types) {};

/*! \param pdata The particle data to associate with
    \param snapshot Snapshot to initialize from
 */
template<unsigned int group_size, typename Group, const char* name, typename snap, bool bond>
MeshGroupData<group_size, Group, name, snap, bond>::MeshGroupData(
    std::shared_ptr<ParticleData> pdata,
    const TriangleData::Snapshot& snapshot)
    : BondedGroupData<group_size, Group, name, true>(
        pdata,
        static_cast<unsigned int>(snapshot.type_mapping.size()))
    {
    // connect to particle sort signal
    this->m_pdata->getParticleSortSignal()
        .template connect<BondedGroupData<group_size, Group, name, true>,
                          &BondedGroupData<group_size, Group, name, true>::setDirty>(this);

    // initialize from snapshot
    initializeFromSnapshot(snapshot);

#ifdef ENABLE_MPI
    if (this->m_pdata->getDomainDecomposition())
        {
        this->m_pdata->getSingleParticleMoveSignal()
            .template connect<BondedGroupData<group_size, Group, name, true>,
                              &BondedGroupData<group_size, Group, name, true>::moveParticleGroups>(
                this);
        }
#endif
    }

//! Destructor
template<unsigned int group_size, typename Group, const char* name, typename snap, bool bond>
MeshGroupData<group_size, Group, name, snap, bond>::~MeshGroupData()
    {
    this->m_pdata->getParticleSortSignal()
        .template disconnect<BondedGroupData<group_size, Group, name, true>,
                             &BondedGroupData<group_size, Group, name, true>::setDirty>(this);
#ifdef ENABLE_MPI
    this->m_pdata->getSingleParticleMoveSignal()
        .template disconnect<BondedGroupData<group_size, Group, name, true>,
                             &BondedGroupData<group_size, Group, name, true>::moveParticleGroups>(
            this);
#endif
    }

//! Initialize from a snapshot
template<unsigned int group_size, typename Group, const char* name, typename snap, bool bond>
void MeshGroupData<group_size, Group, name, snap, bond>::initializeFromSnapshot(
    const TriangleData::Snapshot& snapshot)
    {
    // check that all fields in the snapshot have correct length
    if (this->m_exec_conf->getRank() == 0 && !snapshot.validate())
        {
        this->m_exec_conf->msg->error()
            << "init.*: invalid " << name << " data snapshot." << std::endl
            << std::endl;
        throw std::runtime_error(std::string("Error initializing ") + name + std::string(" data."));
        }

    // re-initialize data structures
    this->initialize();

    std::vector<typename BondedGroupData<group_size, Group, name, true>::members_t> all_groups;

    if (bond)
        {
        for (unsigned group_idx = 0; group_idx < snapshot.groups.size(); group_idx++)
            {
            typename BondedGroupData<group_size, Group, name, true>::members_t triag;
            std::vector<typename BondedGroupData<group_size, Group, name, true>::members_t> bonds(
                3);
            triag.tag[0] = snapshot.groups[group_idx].tag[0];
            triag.tag[1] = snapshot.groups[group_idx].tag[1];
            triag.tag[2] = snapshot.groups[group_idx].tag[2];

            bonds[0].tag[0] = triag.tag[0];
            bonds[0].tag[1] = triag.tag[1];
            bonds[0].tag[2] = group_idx;
            bonds[0].tag[3] = group_idx;
            bonds[1].tag[0] = triag.tag[1];
            bonds[1].tag[1] = triag.tag[2];
            bonds[1].tag[2] = group_idx;
            bonds[1].tag[3] = group_idx;
            bonds[2].tag[0] = triag.tag[2];
            bonds[2].tag[1] = triag.tag[0];
            bonds[2].tag[2] = group_idx;
            bonds[2].tag[3] = group_idx;

            for (unsigned int i = 0; i < all_groups.size(); ++i)
                {
                for (unsigned int j = 0; j < bonds.size(); ++j)
                    {
                    if ((bonds[j].tag[0] == all_groups[i].tag[0]
                         && bonds[j].tag[1] == all_groups[i].tag[1])
                        || (bonds[j].tag[0] == all_groups[i].tag[1]
                            && bonds[j].tag[1] == all_groups[i].tag[0]))
                        {
                        all_groups[i].tag[3] = group_idx;
                        bonds.erase(bonds.begin() + j);
                        break;
                        }
                    }
                }
            for (unsigned int i = 0; i < bonds.size(); ++i)
                {
                all_groups.push_back(bonds[i]);
                }
            }
        }
    else
        {
        std::vector<typename BondedGroupData<group_size, Group, name, true>::members_t> all_helper;
        all_groups.resize(snapshot.groups.size());
        for (unsigned group_idx = 0; group_idx < snapshot.groups.size(); group_idx++)
            {
            typename BondedGroupData<group_size, Group, name, true>::members_t triag;
            std::vector<typename BondedGroupData<group_size, Group, name, true>::members_t> bonds(
                3);
            std::vector<int> bond_id;
            triag.tag[0] = snapshot.groups[group_idx].tag[0];
            triag.tag[1] = snapshot.groups[group_idx].tag[1];
            triag.tag[2] = snapshot.groups[group_idx].tag[2];

            bonds[0].tag[0] = triag.tag[0];
            bonds[0].tag[1] = triag.tag[1];
            bonds[1].tag[0] = triag.tag[1];
            bonds[1].tag[1] = triag.tag[2];
            bonds[2].tag[0] = triag.tag[2];
            bonds[2].tag[1] = triag.tag[0];

            for (unsigned int i = 0; i < all_helper.size(); ++i)
                {
                for (unsigned int j = 0; j < bonds.size(); ++j)
                    {
                    if ((bonds[j].tag[0] == all_helper[i].tag[0]
                         && bonds[j].tag[1] == all_helper[i].tag[1])
                        || (bonds[j].tag[0] == all_helper[i].tag[1]
                            && bonds[j].tag[1] == all_helper[i].tag[0]))
                        {
                        bond_id.push_back(i);
                        bonds.erase(bonds.begin() + j);
                        break;
                        }
                    }
                }
            unsigned int j = 0;
            for (j = 0; j < bond_id.size(); ++j)
                {
                triag.tag[3 + j] = bond_id[j];
                }

            for (unsigned int i = 0; i < bonds.size(); ++i)
                {
                triag.tag[3 + j] = (unsigned int)all_helper.size();
                all_helper.push_back(bonds[i]);
                j++;
                }
            all_groups[group_idx] = triag;
            }
        }

#ifdef ENABLE_MPI
    if (this->m_pdata->getDomainDecomposition())
        {
        // broadcast to all processors (temporarily)
        std::vector<typeval_t> all_typeval;

        if (this->m_exec_conf->getRank() == 0)
            {
            all_typeval.resize(snapshot.type_id.size());
            // fill in types
            for (unsigned int i = 0; i < snapshot.type_id.size(); ++i)
                {
                typeval_t t;
                t.type = snapshot.type_id[i];
                all_typeval[i] = t;
                }
            this->m_type_mapping = snapshot.type_mapping;
            }

        bcast(all_groups, 0, this->m_exec_conf->getMPICommunicator());
        bcast(all_typeval, 0, this->m_exec_conf->getMPICommunicator());
        bcast(this->m_type_mapping, 0, this->m_exec_conf->getMPICommunicator());

        if (bond)
            {
            for (unsigned int group_tag = 0; group_tag < all_groups.size(); ++group_tag)
                addBondedGroup(Group(all_typeval[0], all_groups[group_tag]));
            }
        else
            {
            for (unsigned int group_tag = 0; group_tag < all_groups.size(); ++group_tag)
                addBondedGroup(Group(all_typeval[group_tag], all_groups[group_tag]));
            }
        }
    else
#endif
        {
        this->m_type_mapping = snapshot.type_mapping;

        typeval_t t;
        t.type = 0;
        // t.type = snapshot.type_id[0];
        for (unsigned group_idx = 0; group_idx < all_groups.size(); group_idx++)
            {
            addBondedGroup(Group(t, all_groups[group_idx]));
            }
        }
    }

template<unsigned int group_size, typename Group, const char* name, typename snap, bool bond>
unsigned int MeshGroupData<group_size, Group, name, snap, bond>::addBondedGroup(Group g)
    {
    this->removeAllGhostGroups();

    typeval_t typeval = g.get_typeval();
    typename BondedGroupData<group_size, Group, name, true>::members_t members_tags
        = g.get_members();

    unsigned int max_tag = this->m_pdata->getMaximumTag();

    unsigned int group_size_half = group_size / 2;
    // validate user input
    for (unsigned int i = 0; i < group_size_half; ++i)
        {
        if (members_tags.tag[i] > max_tag)
            {
            std::ostringstream oss;
            oss << "Particle tag out of bounds when attempting to add " << name << ": ";
            for (unsigned int j = 0; j < group_size; ++j)
                oss << members_tags.tag[j] << ((j != group_size - 1) ? "," : "");
            oss << std::endl;
            throw runtime_error(oss.str());
            }

        for (unsigned int j = 0; j < group_size_half; ++j)
            {
            if (i != j && members_tags.tag[i] == members_tags.tag[j])
                {
                std::ostringstream oss;
                oss << "The same particle can only occur once in a " << name << ": ";
                for (unsigned int k = 0; k < group_size; ++k)
                    oss << members_tags.tag[k] << ((k != group_size - 1) ? "," : "");
                oss << std::endl;
                throw runtime_error(oss.str());
                }
            }
        }

    unsigned int type = typeval.type;
    if (type >= this->m_type_mapping.size())
        {
        std::ostringstream s;
        s << "Invalid " << name << " typeid " << type << ". The number of types is "
          << this->m_type_mapping.size() << ".";
        throw std::runtime_error(s.str());
        }

    // determine if bonded group needs to be added to local data
    bool is_local = true;
#ifdef ENABLE_MPI
    if (this->m_pdata->getDomainDecomposition())
        {
        is_local = false;
        // if any of the member tags is local, store this bond
        for (unsigned int i = 0; i < group_size; ++i)
            if (this->m_pdata->isParticleLocal(members_tags.tag[i]))
                {
                is_local = true;
                break;
                }
        }
#endif

    unsigned int tag = 0;
    // first check if we can recycle a deleted tag
    if (this->m_recycled_tags.size())
        {
        tag = this->m_recycled_tags.top();
        this->m_recycled_tags.pop();

        // update reverse-lookup tag to point to end of local group data
        if (is_local)
            this->m_group_rtag[tag] = this->getN();

        assert(is_local || this->m_group_rtag[tag] == GROUP_NOT_LOCAL);
        }
    else
        {
        // Otherwise, generate a new tag
        tag = this->getNGlobal();

        // add new reverse-lookup tag
        assert(this->m_group_rtag.size() == this->getNGlobal());
        if (is_local)
            {
            this->m_group_rtag.push_back(this->getN());
            }
        else
            {
            this->m_group_rtag.push_back(GROUP_NOT_LOCAL);
            }
        }

    assert(tag <= this->m_recycled_tags.size() + this->getNGlobal());

    if (is_local)
        {
        this->m_groups.push_back(members_tags);
        this->m_group_typeval.push_back(typeval);
        this->m_group_tag.push_back(tag);
#ifdef ENABLE_MPI
        if (this->m_pdata->getDomainDecomposition())
            {
            typename BondedGroupData<group_size, Group, name, true>::ranks_t r;
            // initialize with zero
            for (unsigned int i = 0; i < group_size; ++i)
                r.idx[i] = 0;

            this->m_group_ranks.push_back(r);
            }
#endif

        this->m_n_groups++;
        }

    // add to set of active tags
    this->m_tag_set.insert(tag);
    this->m_invalid_cached_tags = true;

    // increment number of bonded groups
    this->m_nglobal++;

    // notify observers
    this->m_group_num_change_signal.emit();
    this->notifyGroupReorder();

    return tag;
    }

template<unsigned int group_size, typename Group, const char* name, typename snap, bool bond>
void MeshGroupData<group_size, Group, name, snap, bond>::rebuildGPUTable()
    {
#ifdef ENABLE_HIP
    if (this->m_exec_conf->isCUDAEnabled())
        rebuildGPUTableGPU();
    else
#endif
        {
        ArrayHandle<unsigned int> h_rtag(this->m_pdata->getRTags(),
                                         access_location::host,
                                         access_mode::read);

        this->m_gpu_n_groups.resize(this->m_pdata->getN() + this->m_pdata->getNGhosts());

        unsigned int num_groups_max = 0;

        unsigned int group_size_half = group_size / 2;

        unsigned int ngroups_tot = this->m_n_groups + this->m_n_ghost;
            {
            ArrayHandle<unsigned int> h_n_groups(this->m_gpu_n_groups,
                                                 access_location::host,
                                                 access_mode::overwrite);

            unsigned int N = this->m_pdata->getN() + this->m_pdata->getNGhosts();

            // count the number of bonded groups per particle
            // start by initializing the n_groups values to 0
            memset(h_n_groups.data, 0, sizeof(unsigned int) * N);

            // loop through the particles and count the number of groups based on each particle
            // index
            for (unsigned int cur_group = 0; cur_group < ngroups_tot; cur_group++)
                {
                typename BondedGroupData<group_size, Group, name, true>::members_t g
                    = this->m_groups[cur_group];
                for (unsigned int i = 0; i < group_size_half; ++i)
                    {
                    unsigned int tag = g.tag[i];
                    unsigned int idx = h_rtag.data[tag];

                    if (idx == NOT_LOCAL)
                        {
                        throw std::runtime_error("Error building GPU group table.");
                        }

                    h_n_groups.data[idx]++;
                    }
                }

            // find the maximum number of groups
            num_groups_max = *std::max_element(h_n_groups.data, h_n_groups.data + N);
            }

        // resize lookup table
        this->m_gpu_table_indexer
            = Index2D(this->m_pdata->getN() + this->m_pdata->getNGhosts(), num_groups_max);
        this->m_gpu_table.resize(this->m_gpu_table_indexer.getNumElements());

            {
            ArrayHandle<unsigned int> h_n_groups(this->m_gpu_n_groups,
                                                 access_location::host,
                                                 access_mode::overwrite);
            ArrayHandle<typename BondedGroupData<group_size, Group, name, true>::members_t>
                h_gpu_table(this->m_gpu_table, access_location::host, access_mode::overwrite);

            // now, update the actual table
            // zero the number of bonded groups counter (again)
            memset(h_n_groups.data,
                   0,
                   sizeof(unsigned int) * (this->m_pdata->getN() + this->m_pdata->getNGhosts()));

            // loop through all group and add them to each column in the list
            for (unsigned int cur_group = 0; cur_group < ngroups_tot; cur_group++)
                {
                typename BondedGroupData<group_size, Group, name, true>::members_t g
                    = this->m_groups[cur_group];

                for (unsigned int i = 0; i < group_size_half; ++i)
                    {
                    unsigned int tag1 = g.tag[i];
                    unsigned int idx1 = h_rtag.data[tag1];
                    unsigned int num = h_n_groups.data[idx1]++;

                    typename BondedGroupData<group_size, Group, name, true>::members_t h;

                    h.idx[group_size - 1]
                        = static_cast<typeval_t>(this->m_group_typeval[cur_group]).type;
                    for (unsigned int j = group_size_half; j < group_size; ++j)
                        {
                        h.idx[j - 1] = g.tag[j];
                        }

                    // list all group members j!=i in p.idx
                    unsigned int n = 0;
                    for (unsigned int j = 0; j < group_size_half; ++j)
                        {
                        if (j == i)
                            {
                            continue;
                            }
                        unsigned int tag2 = g.tag[j];
                        unsigned int idx2 = h_rtag.data[tag2];
                        h.idx[n++] = idx2;
                        }

                    h_gpu_table.data[this->m_gpu_table_indexer(idx1, num)] = h;
                    }
                }
            }
        }
    }

#ifdef ENABLE_HIP
template<unsigned int group_size, typename Group, const char* name, typename snap, bool bond>
void MeshGroupData<group_size, Group, name, snap, bond>::rebuildGPUTableGPU()
    {
    // resize groups counter
    this->m_gpu_n_groups.resize(this->m_pdata->getN() + this->m_pdata->getNGhosts());

    // resize GPU table to current number of particles
    this->m_gpu_table_indexer = Index2D(this->m_pdata->getN() + this->m_pdata->getNGhosts(),
                                        this->m_gpu_table_indexer.getH());
    this->m_gpu_table.resize(this->m_gpu_table_indexer.getNumElements());
    this->m_gpu_pos_table.resize(this->m_gpu_table_indexer.getNumElements());

    unsigned int group_size_half = group_size / 2;

    bool done = false;
    while (!done)
        {
        unsigned int flag = 0;

            {
            ArrayHandle<typename BondedGroupData<group_size, Group, name, true>::members_t>
                d_groups(this->m_groups, access_location::device, access_mode::read);
            ArrayHandle<typeval_t> d_group_typeval(this->m_group_typeval,
                                                   access_location::device,
                                                   access_mode::read);
            ArrayHandle<unsigned int> d_rtag(this->m_pdata->getRTags(),
                                             access_location::device,
                                             access_mode::read);
            ArrayHandle<unsigned int> d_n_groups(this->m_gpu_n_groups,
                                                 access_location::device,
                                                 access_mode::overwrite);
            ArrayHandle<typename BondedGroupData<group_size, Group, name, true>::members_t>
                d_gpu_table(this->m_gpu_table, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_condition(this->m_condition,
                                                  access_location::device,
                                                  access_mode::readwrite);

            // allocate scratch buffers
            CachedAllocator& alloc = this->m_exec_conf->getCachedAllocator();
            size_t tmp_size = this->m_groups.size() * group_size_half;
            unsigned int nptl = this->m_pdata->getN() + this->m_pdata->getNGhosts();
            ScopedAllocation<unsigned int> d_scratch_g(alloc, tmp_size);
            ScopedAllocation<unsigned int> d_scratch_idx(alloc, tmp_size);
            ScopedAllocation<unsigned int> d_offsets(alloc, tmp_size);

            // fill group table on GPU
            gpu_update_mesh_table<
                group_size,
                typename BondedGroupData<group_size, Group, name, true>::members_t>(
                this->getN() + this->getNGhosts(),
                nptl,
                d_groups.data,
                d_group_typeval.data,
                d_rtag.data,
                d_n_groups.data,
                this->m_gpu_table_indexer.getH(),
                d_condition.data,
                this->m_next_flag,
                flag,
                d_gpu_table.data,
                this->m_gpu_table_indexer.getW(),
                d_scratch_g.data,
                d_scratch_idx.data,
                d_offsets.data,
                this->m_exec_conf->getCachedAllocator());
            }
        if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        if (flag >= this->m_next_flag + 1)
            {
            throw std::runtime_error("Error building GPU group table.");
            }

        if (flag == this->m_next_flag)
            {
            // grow array by incrementing groups per particle
            this->m_gpu_table_indexer = Index2D(this->m_pdata->getN() + this->m_pdata->getNGhosts(),
                                                this->m_gpu_table_indexer.getH() + 1);
            this->m_gpu_table.resize(this->m_gpu_table_indexer.getNumElements());
            this->m_next_flag++;
            }
        else
            done = true;
        }
    }
#endif

/*! \param snapshot Snapshot that will contain the group data
 * \returns a map to lookup snapshot index by tag
 *
 *  Data in the snapshot is in tag order, where non-existent tags are skipped
 */
template<unsigned int group_size, typename Group, const char* name, typename snap, bool bond>
std::map<unsigned int, unsigned int>
MeshGroupData<group_size, Group, name, snap, bond>::takeSnapshot(snap& snapshot) const
    {
    // map to lookup snapshot index by tag
    std::map<unsigned int, unsigned int> index;

    unsigned int group_size_half = group_size / 2;

    std::map<unsigned int, unsigned int> rtag_map;
    for (unsigned int group_idx = 0; group_idx < this->getN(); group_idx++)
        {
        unsigned int tag = this->m_group_tag[group_idx];
        assert(this->m_group_rtag[tag] == group_idx);

        rtag_map.insert(std::pair<unsigned int, unsigned int>(tag, group_idx));
        }

#ifdef ENABLE_MPI
    if (this->m_pdata->getDomainDecomposition())
        {
        // gather local data
        std::vector<typeval_t> typevals; // Group types or constraint values
        std::vector<typename BondedGroupData<group_size, Group, name, true>::members_t>
            members; // Group members

        for (unsigned int group_idx = 0; group_idx < this->getN(); ++group_idx)
            {
            typevals.push_back(this->m_group_typeval[group_idx]);
            members.push_back(this->m_groups[group_idx]);
            }

        std::vector<std::vector<typeval_t>> typevals_proc; // Group types of every processor
        std::vector<std::vector<typename BondedGroupData<group_size, Group, name, true>::members_t>>
            members_proc; // Group members of every processor

        std::vector<std::map<unsigned int, unsigned int>>
            rtag_map_proc; // List of reverse-lookup maps

        unsigned int size = this->m_exec_conf->getNRanks();

        // resize arrays to accumulate group data of all ranks
        typevals_proc.resize(size);
        members_proc.resize(size);
        rtag_map_proc.resize(size);

        // gather all processors' data
        gather_v(typevals, typevals_proc, 0, this->m_exec_conf->getMPICommunicator());
        gather_v(members, members_proc, 0, this->m_exec_conf->getMPICommunicator());
        gather_v(rtag_map, rtag_map_proc, 0, this->m_exec_conf->getMPICommunicator());

        if (this->m_exec_conf->getRank() == 0)
            {
            // allocate memory in snapshot
            snapshot.resize(this->getNGlobal());

            assert(rtag_map_proc.size() == size);

            // create single map of all group ranks and indices
            // groups present on more than one processor will count as one group
            std::map<unsigned int, std::pair<unsigned int, unsigned int>> rank_rtag_map;
            std::map<unsigned int, unsigned int>::iterator it;
            for (unsigned int irank = 0; irank < size; ++irank)
                for (it = rtag_map_proc[irank].begin(); it != rtag_map_proc[irank].end(); ++it)
                    rank_rtag_map.insert(
                        std::make_pair(it->first, std::make_pair(irank, it->second)));

            // add groups to snapshot
            std::map<unsigned int, std::pair<unsigned int, unsigned int>>::iterator rank_rtag_it;

            // index in snapshot
            unsigned int snap_id = 0;

            // loop through active tags
            std::set<unsigned int>::iterator active_tag_it;
            for (active_tag_it = this->m_tag_set.begin(); active_tag_it != this->m_tag_set.end();
                 ++active_tag_it)
                {
                unsigned int group_tag = *active_tag_it;
                rank_rtag_it = rank_rtag_map.find(group_tag);
                if (rank_rtag_it == rank_rtag_map.end())
                    {
                    this->m_exec_conf->msg->error() << endl
                                                    << "Could not find " << name << " " << group_tag
                                                    << " on any processor. " << endl
                                                    << endl;
                    throw std::runtime_error("Error gathering " + std::string(name) + "s");
                    }

                // store tag in index
                index.insert(std::make_pair(group_tag, snap_id));

                // rank contains the processor rank on which the particle was found
                std::pair<unsigned int, unsigned int> rank_idx = rank_rtag_it->second;
                unsigned int rank = rank_idx.first;
                unsigned int idx = rank_idx.second;

                snapshot.type_id[snap_id] = typevals_proc[rank][idx].type;
                for (unsigned int i = 0; i < group_size_half; i++)
                    {
                    snapshot.groups[snap_id].tag[i] = members_proc[rank][idx].tag[i];
                    }
                snap_id++;
                }
            }
        }
    else
#endif
        {
        // allocate memory in snapshot
        snapshot.resize(this->getNGlobal());

        assert(this->getN() == this->getNGlobal());
        std::map<unsigned int, unsigned int>::iterator rtag_it;
        // index in snapshot
        unsigned int snap_id = 0;

        // loop through active tags
        std::set<unsigned int>::iterator active_tag_it;
        for (active_tag_it = this->m_tag_set.begin(); active_tag_it != this->m_tag_set.end();
             ++active_tag_it)
            {
            unsigned int group_tag = *active_tag_it;
            rtag_it = rtag_map.find(group_tag);
            if (rtag_it == rtag_map.end())
                {
                throw std::runtime_error("Could not find " + std::string(name) + " "
                                         + std::to_string(group_tag) + ". Error gathering "
                                         + std::string(name) + "s");
                }

            // store tag in index
            index.insert(std::make_pair(group_tag, snap_id));

            unsigned int group_idx = rtag_it->second;
            typename BondedGroupData<group_size, Group, name, true>::members_t member
                = this->m_groups[group_idx];
            for (unsigned int i = 0; i < group_size_half; i++)
                {
                snapshot.groups[snap_id].tag[i] = member.tag[i];
                }
            snapshot.type_id[snap_id] = ((typeval_t)this->m_group_typeval[group_idx]).type;
            snap_id++;
            }
        }

    snapshot.type_mapping = this->m_type_mapping;

    return index;
    }

namespace detail
    {
template<class T, class Group>
void export_MeshGroupData(pybind11::module& m,
                          std::string name,
                          std::string snapshot_name,
                          bool export_struct)
    {
    // export group structure
    if (export_struct)
        Group::export_to_python(m);

    pybind11::class_<T, std::shared_ptr<T>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<ParticleData>, unsigned int>())
        .def(
            pybind11::init<std::shared_ptr<ParticleData>, const typename TriangleData::Snapshot&>())
        .def("initializeFromSnapshot", &T::initializeFromSnapshot)
        .def("takeSnapshot", &T::takeSnapshot)
        .def("getN", &T::getN)
        .def("getNGlobal", &T::getNGlobal)
        .def("getNTypes", &T::getNTypes)
        .def("getNthTag", &T::getNthTag)
        .def("getMaximumTag", &T::getMaximumTag)
        .def("getGroupByTag", &T::getGroupByTag)
        .def("getTypeByName", &T::getTypeByName)
        .def("setTypeName", &T::setTypeName)
        .def("getNameByType", &T::getNameByType)
        .def("addBondedGroup", &T::addBondedGroup)
        .def("removeBondedGroup", &T::removeBondedGroup)
        .def("getTypes", &T::getTypesPy);
    }

    } // end namespace detail

template class PYBIND11_EXPORT
    MeshGroupData<6, MeshTriangle, name_meshtriangle_data, TriangleData::Snapshot, false>;
template class PYBIND11_EXPORT
    MeshGroupData<4, MeshBond, name_meshbond_data, BondData::Snapshot, true>;

namespace detail
    {
template void export_MeshGroupData<MeshTriangleData, MeshTriangle>(pybind11::module& m,
                                                                   std::string name,
                                                                   std::string snapshot_name,
                                                                   bool export_struct);

template void export_MeshGroupData<MeshBondData, MeshBond>(pybind11::module& m,
                                                           std::string name,
                                                           std::string snapshot_name,
                                                           bool export_struct);
    } // end namespace detail
    } // end namespace hoomd
