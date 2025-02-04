// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file BondedGroupData.h
    \brief Defines implementation of BondedGroupData
 */

#include "BondedGroupData.h"
#include "Index1D.h"
#include "ParticleData.h"

#include <pybind11/numpy.h>

#ifdef ENABLE_HIP
#include "BondedGroupData.cuh"
#include "CachedAllocator.h"
#endif

using namespace std;

namespace hoomd
    {
//! Names of bonded groups
char name_bond_data[] = "bond";
char name_angle_data[] = "angle";
char name_triangle_data[] = "triangle";
char name_dihedral_data[] = "dihedral";
char name_improper_data[] = "improper";
char name_meshbond_data[] = "meshbond";
char name_constraint_data[] = "constraint";
char name_pair_data[] = "pair";

/*
 * Implementation of BondedGroupData methods
 */

/*! \param exec_conf Execution configuration
    \param pdata The particle data to associate with
    \param n_group_types Number of bonded group types to initialize
 */
template<unsigned int group_size, typename Group, const char* name, bool has_type_mapping>
BondedGroupData<group_size, Group, name, has_type_mapping>::BondedGroupData(
    std::shared_ptr<ParticleData> pdata)
    : m_exec_conf(pdata->getExecConf()), m_pdata(pdata), m_n_groups(0), m_n_ghost(0), m_nglobal(0),
      m_groups_dirty(true)
    {
    }

/*! \param exec_conf Execution configuration
    \param pdata The particle data to associate with
    \param n_group_types Number of bonded group types to initialize
 */
template<unsigned int group_size, typename Group, const char* name, bool has_type_mapping>
BondedGroupData<group_size, Group, name, has_type_mapping>::BondedGroupData(
    std::shared_ptr<ParticleData> pdata,
    unsigned int n_group_types)
    : m_exec_conf(pdata->getExecConf()), m_pdata(pdata), m_n_groups(0), m_n_ghost(0), m_nglobal(0),
      m_groups_dirty(true)
    {
    m_exec_conf->msg->notice(5) << "Constructing BondedGroupData (" << name << "s, n=" << group_size
                                << ") " << endl;

    // connect to particle sort signal
    m_pdata->getParticleSortSignal()
        .template connect<BondedGroupData<group_size, Group, name, has_type_mapping>,
                          &BondedGroupData<group_size, Group, name, has_type_mapping>::setDirty>(
            this);
#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        m_pdata->getSingleParticleMoveSignal()
            .template connect<
                BondedGroupData<group_size, Group, name, has_type_mapping>,
                &BondedGroupData<group_size, Group, name, has_type_mapping>::moveParticleGroups>(
                this);
        }
#endif

    if (has_type_mapping)
        {
        // offer a default type mapping
        for (unsigned int i = 0; i < n_group_types; i++)
            {
            m_type_mapping.push_back(detail::getDefaultTypeName(i));
            }
        }

    // initialize data structures
    initialize();
    }

/*! \param exec_conf Execution configuration
    \param pdata The particle data to associate with
    \param snapshot Snapshot to initialize from
 */
template<unsigned int group_size, typename Group, const char* name, bool has_type_mapping>
BondedGroupData<group_size, Group, name, has_type_mapping>::BondedGroupData(
    std::shared_ptr<ParticleData> pdata,
    const Snapshot& snapshot)
    : m_exec_conf(pdata->getExecConf()), m_pdata(pdata), m_n_groups(0), m_n_ghost(0), m_nglobal(0),
      m_groups_dirty(true)
    {
    m_exec_conf->msg->notice(5) << "Constructing BondedGroupData (" << name << ") " << endl;

    // connect to particle sort signal
    m_pdata->getParticleSortSignal()
        .template connect<BondedGroupData<group_size, Group, name, has_type_mapping>,
                          &BondedGroupData<group_size, Group, name, has_type_mapping>::setDirty>(
            this);

    // initialize from snapshot
    initializeFromSnapshot(snapshot);

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        m_pdata->getSingleParticleMoveSignal()
            .template connect<
                BondedGroupData<group_size, Group, name, has_type_mapping>,
                &BondedGroupData<group_size, Group, name, has_type_mapping>::moveParticleGroups>(
                this);
        }
#endif
    }

//! Destructor
template<unsigned int group_size, typename Group, const char* name, bool has_type_mapping>
BondedGroupData<group_size, Group, name, has_type_mapping>::~BondedGroupData()
    {
    m_pdata->getParticleSortSignal()
        .template disconnect<BondedGroupData<group_size, Group, name, has_type_mapping>,
                             &BondedGroupData<group_size, Group, name, has_type_mapping>::setDirty>(
            this);
#ifdef ENABLE_MPI
    m_pdata->getSingleParticleMoveSignal()
        .template disconnect<
            BondedGroupData<group_size, Group, name, has_type_mapping>,
            &BondedGroupData<group_size, Group, name, has_type_mapping>::moveParticleGroups>(this);
#endif
    }

template<unsigned int group_size, typename Group, const char* name, bool has_type_mapping>
void BondedGroupData<group_size, Group, name, has_type_mapping>::initialize()
    {
    // reset global number of groups
    m_nglobal = 0;

    // reset local number of groups
    m_n_groups = 0;

    // reset local number of ghost groups
    m_n_ghost = 0;

    // clear set of active tags
    m_tag_set.clear();
    m_invalid_cached_tags = true;

    // clear reservoir of recycled tags
    while (!m_recycled_tags.empty())
        m_recycled_tags.pop();

    // allocate arrays
    GPUVector<members_t> groups(m_exec_conf);
    m_groups.swap(groups);

    GPUVector<typeval_t> typeval(m_exec_conf);
    m_group_typeval.swap(typeval);

    GPUVector<unsigned int> group_tag(m_exec_conf);
    m_group_tag.swap(group_tag);

    GPUVector<unsigned int> group_rtag(m_exec_conf);
    m_group_rtag.swap(group_rtag);

    // Lookup by particle index table
    GPUVector<members_t> gpu_table(m_exec_conf);
    m_gpu_table.swap(gpu_table);

    GPUVector<unsigned int> gpu_pos_table(m_exec_conf);
    m_gpu_pos_table.swap(gpu_pos_table);

    GPUVector<unsigned int> n_groups(m_exec_conf);
    m_gpu_n_groups.swap(n_groups);

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        GPUVector<ranks_t> group_ranks(m_exec_conf);
        m_group_ranks.swap(group_ranks);
        }
#endif

    // allocate stand-by arrays
    GPUVector<typeval_t> typeval_alt(m_exec_conf);
    m_group_typeval_alt.swap(typeval_alt);

    GPUVector<unsigned int> group_tag_alt(m_exec_conf);
    m_group_tag_alt.swap(group_tag_alt);

    GPUVector<members_t> groups_alt(m_exec_conf);
    m_groups_alt.swap(groups_alt);

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        GPUVector<ranks_t> group_ranks_alt(m_exec_conf);
        m_group_ranks_alt.swap(group_ranks_alt);
        }
#endif

#ifdef ENABLE_HIP
    // allocate condition variable
    GPUArray<unsigned int> condition(1, m_exec_conf);
    m_condition.swap(condition);

    ArrayHandle<unsigned int> h_condition(m_condition,
                                          access_location::host,
                                          access_mode::overwrite);
    *h_condition.data = 0;
    m_next_flag = 1;
#endif

    m_tag_set.clear();
    }

//! Initialize from a snapshot
template<unsigned int group_size, typename Group, const char* name, bool has_type_mapping>
void BondedGroupData<group_size, Group, name, has_type_mapping>::initializeFromSnapshot(
    const Snapshot& snapshot)
    {
    // check that all fields in the snapshot have correct length
    if (m_exec_conf->getRank() == 0)
        {
        snapshot.validate();
        }

    if (snapshot.type_mapping.size() >= 40)
        {
        std::ostringstream s;
        s << "Systems with many " << name
          << " types perform poorly or result "
             "in shared memory errors on the GPU.";
        m_exec_conf->msg->warning() << s.str() << std::endl;
        }

    // re-initialize data structures
    initialize();

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        // broadcast to all processors (temporarily)
        std::vector<members_t> all_groups;
        std::vector<typeval_t> all_typeval;

        if (m_exec_conf->getRank() == 0)
            {
            all_groups = snapshot.groups;
            if (has_type_mapping)
                {
                all_typeval.resize(snapshot.type_id.size());
                // fill in types
                for (unsigned int i = 0; i < snapshot.type_id.size(); ++i)
                    {
                    typeval_t t;
                    t.type = snapshot.type_id[i];
                    all_typeval[i] = t;
                    }
                }
            else
                {
                all_typeval.resize(snapshot.val.size());
                // fill in constraint values
                for (unsigned int i = 0; i < snapshot.val.size(); ++i)
                    {
                    typeval_t t;
                    t.val = snapshot.val[i];
                    all_typeval[i] = t;
                    }
                }

            m_type_mapping = snapshot.type_mapping;
            }

        bcast(all_groups, 0, m_exec_conf->getMPICommunicator());
        bcast(all_typeval, 0, m_exec_conf->getMPICommunicator());
        bcast(m_type_mapping, 0, m_exec_conf->getMPICommunicator());

        // iterate over groups and add those that have local particles
        for (unsigned int group_tag = 0; group_tag < all_groups.size(); ++group_tag)
            addBondedGroup(Group(all_typeval[group_tag], all_groups[group_tag]));
        }
    else
#endif
        {
        m_type_mapping = snapshot.type_mapping;

        if (has_type_mapping)
            {
            // create bonded groups with types
            for (unsigned group_idx = 0; group_idx < snapshot.groups.size(); group_idx++)
                {
                typeval_t t;
                t.type = snapshot.type_id[group_idx];
                addBondedGroup(Group(t, snapshot.groups[group_idx]));
                }
            }
        else
            {
            // create constraints
            for (unsigned group_idx = 0; group_idx < snapshot.groups.size(); group_idx++)
                {
                typeval_t t;
                t.val = snapshot.val[group_idx];
                addBondedGroup(Group(t, snapshot.groups[group_idx]));
                }
            }
        }
    }

template<unsigned int group_size, typename Group, const char* name, bool has_type_mapping>
unsigned int BondedGroupData<group_size, Group, name, has_type_mapping>::addBondedGroup(Group g)
    {
    // we are changing the local number of groups, so remove ghosts
    removeAllGhostGroups();

    typeval_t typeval = g.get_typeval();
    members_t member_tags = g.get_members();

    unsigned int max_tag = m_pdata->getMaximumTag();

    // validate user input
    for (unsigned int i = 0; i < group_size; ++i)
        if (member_tags.tag[i] > max_tag)
            {
            std::ostringstream oss;
            oss << "Particle tag out of bounds when attempting to add " << name << ": ";
            for (unsigned int j = 0; j < group_size; ++j)
                oss << member_tags.tag[j] << ((j != group_size - 1) ? "," : "");
            oss << std::endl;
            throw runtime_error(oss.str());
            }

    for (unsigned int i = 0; i < group_size; ++i)
        for (unsigned int j = 0; j < group_size; ++j)
            if (i != j && member_tags.tag[i] == member_tags.tag[j])
                {
                std::ostringstream oss;
                oss << "The same particle can only occur once in a " << name << ": ";
                for (unsigned int k = 0; k < group_size; ++k)
                    oss << member_tags.tag[k] << ((k != group_size - 1) ? "," : "");
                oss << std::endl;
                throw runtime_error(oss.str());
                }

    if (has_type_mapping)
        {
        unsigned int type = typeval.type;
        if (type >= m_type_mapping.size() && has_type_mapping)
            {
            std::ostringstream s;
            s << "Invalid " << name << " typeid " << type << ". The number of types is "
              << m_type_mapping.size() << ".";
            throw std::runtime_error(s.str());
            }
        }

    unsigned int tag = 0;

    // determine if bonded group needs to be added to local data
    bool is_local = true;
#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        is_local = false;
        // if any of the member tags is local, store this bond
        for (unsigned int i = 0; i < group_size; ++i)
            if (m_pdata->isParticleLocal(member_tags.tag[i]))
                {
                is_local = true;
                break;
                }
        }
#endif

    // first check if we can recycle a deleted tag
    if (m_recycled_tags.size())
        {
        tag = m_recycled_tags.top();
        m_recycled_tags.pop();

        // update reverse-lookup tag to point to end of local group data
        if (is_local)
            m_group_rtag[tag] = getN();

        assert(is_local || m_group_rtag[tag] == GROUP_NOT_LOCAL);
        }
    else
        {
        // Otherwise, generate a new tag
        tag = getNGlobal();

        // add new reverse-lookup tag
        assert(m_group_rtag.size() == getNGlobal());
        if (is_local)
            m_group_rtag.push_back(getN());
        else
            m_group_rtag.push_back(GROUP_NOT_LOCAL);
        }

    assert(tag <= m_recycled_tags.size() + getNGlobal());

    if (is_local)
        {
        m_groups.push_back(member_tags);
        m_group_typeval.push_back(typeval);
        m_group_tag.push_back(tag);
#ifdef ENABLE_MPI
        if (m_pdata->getDomainDecomposition())
            {
            ranks_t r;
            // initialize with zero
            for (unsigned int i = 0; i < group_size; ++i)
                r.idx[i] = 0;

            m_group_ranks.push_back(r);
            }
#endif

        m_n_groups++;
        }

    // add to set of active tags
    m_tag_set.insert(tag);
    m_invalid_cached_tags = true;

    // increment number of bonded groups
    m_nglobal++;

    // notify observers
    m_group_num_change_signal.emit();
    notifyGroupReorder();

    return tag;
    }

//! Return the nth active global tag
/*! \param n Index of bond in global bond table
 */
template<unsigned int group_size, typename Group, const char* name, bool has_type_mapping>
unsigned int BondedGroupData<group_size, Group, name, has_type_mapping>::getNthTag(unsigned int n)
    {
    if (n >= getNGlobal())
        {
        std::ostringstream s;
        s << name << " index " << n << " out of bounds!" << "The number of " << name << "s is "
          << getNGlobal();
        throw std::runtime_error(s.str());
        }

    assert(m_tag_set.size() == getNGlobal());

    // maybe_rebuild_tag_cache only rebuilds if necessary
    maybe_rebuild_tag_cache();
    return m_cached_tag_set[n];
    }

/*! \param tag Tag of bonded group
 * \returns the group
 */
template<unsigned int group_size, typename Group, const char* name, bool has_type_mapping>
const Group
BondedGroupData<group_size, Group, name, has_type_mapping>::getGroupByTag(unsigned int tag) const
    {
    // Find position of bonded group in list
    unsigned int group_idx = m_group_rtag[tag];

    typeval_t typeval;
    members_t members;

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        int my_rank = m_exec_conf->getRank();
        // set local to rank if the bond is local, -1 if not
        int rank = group_idx < m_n_groups ? my_rank : -1;

        // the highest rank owning the group sends it to the others
        MPI_Allreduce(MPI_IN_PLACE, &rank, 1, MPI_INT, MPI_MAX, m_exec_conf->getMPICommunicator());

        if (rank == -1)
            {
            m_exec_conf->msg->error()
                << "Trying to find " << name << " " << tag << " which does not exist!" << endl;
            throw runtime_error(std::string("Error getting ") + name);
            }

        if (rank == (int)my_rank)
            {
            typeval = m_group_typeval[group_idx];
            members = m_groups[group_idx];
            }

        bcast(typeval, rank, m_exec_conf->getMPICommunicator());
        bcast(members, rank, m_exec_conf->getMPICommunicator());
        }
    else
#endif
        {
        if (group_idx == GROUP_NOT_LOCAL)
            {
            std::ostringstream s;
            s << "Trying to get type or constraint value of " << name << " " << tag
              << " which does not exist!";
            throw runtime_error(s.str());
            }

        typeval = m_group_typeval[group_idx];
        members = m_groups[group_idx];
        }

    // perform a final sanity check that the group is valid
    for (unsigned int j = 0; j < group_size; ++j)
        {
        unsigned int ptag = members.tag[j];

        if (!m_pdata->isTagActive(ptag))
            {
            std::ostringstream s;
            s << "Member tag " << ptag << " of " << name << " " << tag << " does not exist!"
              << endl;
            throw runtime_error(s.str());
            }
        }

    return Group(typeval, members);
    }

/*! \param idx Index of bonded group
 * \return Member tags of bonded group
 */
template<unsigned int group_size, typename Group, const char* name, bool has_type_mapping>
unsigned int BondedGroupData<group_size, Group, name, has_type_mapping>::getTypeByIndex(
    unsigned int group_idx) const
    {
    assert(group_idx < getN());
    assert(has_type_mapping);
    return ((typeval_t)m_group_typeval[group_idx]).type;
    }

/*! \param idx Index of bonded group
 * \return Member tags of bonded group
 */
template<unsigned int group_size, typename Group, const char* name, bool has_type_mapping>
Scalar BondedGroupData<group_size, Group, name, has_type_mapping>::getValueByIndex(
    unsigned int group_idx) const
    {
    assert(group_idx < getN() + getNGhosts());
    assert(!has_type_mapping);
    return ((typeval_t)m_group_typeval[group_idx]).val;
    }

/*! \param idx Tag of bonded group
 * \return Type of bonded group
 */
template<unsigned int group_size, typename Group, const char* name, bool has_type_mapping>
const typename BondedGroupData<group_size, Group, name, has_type_mapping>::members_t
BondedGroupData<group_size, Group, name, has_type_mapping>::getMembersByIndex(
    unsigned int group_idx) const
    {
    assert(group_idx < getN() + getNGhosts());
    return m_groups[group_idx];
    }

/*! \param idx Tag of bonded group
 *  \param member new member value
 */
template<unsigned int group_size, typename Group, const char* name, bool has_type_mapping>
void BondedGroupData<group_size, Group, name, has_type_mapping>::setMemberByIndex(
    unsigned int group_idx,
    typename BondedGroupData<group_size, Group, name, has_type_mapping>::members_t member)
    {
    assert(group_idx < getN() + getNGhosts());
    m_groups[group_idx] = member;
    }

/*! \param tag Tag of bonded group to remove
 */
template<unsigned int group_size, typename Group, const char* name, bool has_type_mapping>
void BondedGroupData<group_size, Group, name, has_type_mapping>::removeBondedGroup(unsigned int tag)
    {
    // we are changing the local particle number, remove ghost groups
    removeAllGhostGroups();

    // sanity check
    if (tag >= m_group_rtag.size())
        {
        std::ostringstream s;
        s << "Trying to remove " << name << " " << tag << " which does not exist!";
        throw runtime_error(s.str());
        }

    // Find position of bonded group in list
    unsigned int id = m_group_rtag[tag];

    bool is_local = id < getN();
    assert(is_local || id == GROUP_NOT_LOCAL);

    bool is_available = is_local;

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        int res = is_local ? 1 : 0;

        // check that group is local on some processors
        MPI_Allreduce(MPI_IN_PLACE, &res, 1, MPI_INT, MPI_SUM, m_exec_conf->getMPICommunicator());

        assert((unsigned int)res <= group_size);
        is_available = res;
        }
#endif

    if (!is_available)
        {
        std::ostringstream s;
        s << "Trying to remove " << name << " " << tag << " which has been previously removed!";
        throw runtime_error(s.str());
        }

    // delete from map
    m_group_rtag[tag] = GROUP_NOT_LOCAL;

    if (is_local)
        {
        unsigned int size = m_n_groups;
        // If the bonded group is in the middle of the list, move the last element to
        // to the position of the removed element
        if (id < (size - 1))
            {
            m_groups[id] = (members_t)m_groups[size - 1];
            m_group_typeval[id] = (typeval_t)m_group_typeval[size - 1];
#ifdef ENABLE_MPI
            if (m_pdata->getDomainDecomposition())
                m_group_ranks[id] = (ranks_t)m_group_ranks[size - 1];
#endif
            unsigned int last_tag = m_group_tag[size - 1];
            m_group_rtag[last_tag] = id;
            m_group_tag[id] = last_tag;
            }

        // delete last element
        m_groups.pop_back();
        m_group_typeval.pop_back();
        m_group_tag.pop_back();
#ifdef ENABLE_MPI
        if (m_pdata->getDomainDecomposition())
            m_group_ranks.pop_back();
#endif
        m_n_groups--;
        }

    // remove from set of active tags
    m_tag_set.erase(tag);
    m_invalid_cached_tags = true;

    // maintain a stack of deleted group tags for future recycling
    m_recycled_tags.push(tag);
    m_nglobal--;

    // notify observers
    m_group_num_change_signal.emit();
    notifyGroupReorder();
    }

/*! \param name Type name
 */
template<unsigned int group_size, typename Group, const char* name, bool has_type_mapping>
unsigned int BondedGroupData<group_size, Group, name, has_type_mapping>::getTypeByName(
    const std::string& type_name) const
    {
    // search for the name
    for (unsigned int i = 0; i < m_type_mapping.size(); i++)
        {
        if (m_type_mapping[i] == type_name)
            return i;
        }

    std::ostringstream s;
    s << name << " type " << type_name << " not found!" << endl;
    throw runtime_error(s.str());

    // silence compiler warning
    return 0;
    }

template<unsigned int group_size, typename Group, const char* name, bool has_type_mapping>
const std::string
BondedGroupData<group_size, Group, name, has_type_mapping>::getNameByType(unsigned int type) const
    {
    // check for an invalid request
    if (type >= m_type_mapping.size())
        {
        std::ostringstream s;
        s << "Requesting type name for non-existent type " << type << endl;
        throw runtime_error(s.str());
        }

    // return the name
    return m_type_mapping[type];
    }

template<unsigned int group_size, typename Group, const char* name, bool has_type_mapping>
void BondedGroupData<group_size, Group, name, has_type_mapping>::setTypeName(
    unsigned int type,
    const std::string& new_name)
    {
    // check for an invalid request
    if (type >= this->m_type_mapping.size())
        {
        std::ostringstream s;
        s << "Setting type name for non-existent type " << type;
        throw runtime_error(s.str());
        }

    m_type_mapping[type] = new_name;
    }

/*! Rebuild the cached vector of active tags, if necessary
 */
template<unsigned int group_size, typename Group, const char* name, bool has_type_mapping>
void BondedGroupData<group_size, Group, name, has_type_mapping>::maybe_rebuild_tag_cache()
    {
    if (!m_invalid_cached_tags)
        return;

    // GPUVector checks if the resize is necessary
    m_cached_tag_set.resize(m_tag_set.size());

    ArrayHandle<unsigned int> h_active_tag(m_cached_tag_set,
                                           access_location::host,
                                           access_mode::overwrite);

    // iterate over each element in the set, building a mapping
    // from dense array indices to sparse particle tag indices
    unsigned int i(0);
    for (std::set<unsigned int>::const_iterator it(m_tag_set.begin()); it != m_tag_set.end();
         ++it, ++i)
        {
        h_active_tag.data[i] = *it;
        }

    m_invalid_cached_tags = false;
    }

template<unsigned int group_size, typename Group, const char* name, bool has_type_mapping>
void BondedGroupData<group_size, Group, name, has_type_mapping>::rebuildGPUTable()
    {
#ifdef ENABLE_HIP
    if (m_exec_conf->isCUDAEnabled())
        rebuildGPUTableGPU();
    else
#endif
        {
        ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(),
                                         access_location::host,
                                         access_mode::read);

        m_gpu_n_groups.resize(m_pdata->getN() + m_pdata->getNGhosts());

        unsigned int num_groups_max = 0;

        unsigned int ngroups_tot = m_n_groups + m_n_ghost;
            {
            ArrayHandle<unsigned int> h_n_groups(m_gpu_n_groups,
                                                 access_location::host,
                                                 access_mode::overwrite);

            unsigned int N = m_pdata->getN() + m_pdata->getNGhosts();

            // count the number of bonded groups per particle
            // start by initializing the n_groups values to 0
            memset(h_n_groups.data, 0, sizeof(unsigned int) * N);

            // loop through the particles and count the number of groups based on each particle
            // index
            for (unsigned int cur_group = 0; cur_group < ngroups_tot; cur_group++)
                {
                members_t g = m_groups[cur_group];
                for (unsigned int i = 0; i < group_size; ++i)
                    {
                    unsigned int tag = g.tag[i];
                    unsigned int idx = h_rtag.data[tag];

                    if (idx == NOT_LOCAL)
                        {
                        // incomplete group
                        std::ostringstream oss;
                        oss << name << " ";
                        for (unsigned int k = 0; k < group_size; ++k)
                            oss << g.tag[k] << ((k != group_size - 1) ? ", " : " ");
                        oss << "incomplete!";
                        throw std::runtime_error(oss.str());
                        }

                    h_n_groups.data[idx]++;
                    }
                }

            // find the maximum number of groups
            num_groups_max = *std::max_element(h_n_groups.data, h_n_groups.data + N);
            }

        // resize lookup table
        m_gpu_table_indexer = Index2D(m_pdata->getN() + m_pdata->getNGhosts(), num_groups_max);
        m_gpu_table.resize(m_gpu_table_indexer.getNumElements());
        m_gpu_pos_table.resize(m_gpu_table_indexer.getNumElements());

            {
            ArrayHandle<unsigned int> h_n_groups(m_gpu_n_groups,
                                                 access_location::host,
                                                 access_mode::overwrite);
            ArrayHandle<members_t> h_gpu_table(m_gpu_table,
                                               access_location::host,
                                               access_mode::overwrite);
            ArrayHandle<unsigned int> h_gpu_pos_table(m_gpu_pos_table,
                                                      access_location::host,
                                                      access_mode::overwrite);

            // now, update the actual table
            // zero the number of bonded groups counter (again)
            memset(h_n_groups.data,
                   0,
                   sizeof(unsigned int) * (m_pdata->getN() + m_pdata->getNGhosts()));

            // loop through all group and add them to each column in the list
            for (unsigned int cur_group = 0; cur_group < ngroups_tot; cur_group++)
                {
                members_t g = m_groups[cur_group];

                for (unsigned int i = 0; i < group_size; ++i)
                    {
                    unsigned int tag1 = g.tag[i];
                    unsigned int idx1 = h_rtag.data[tag1];
                    unsigned int num = h_n_groups.data[idx1]++;

                    members_t h;

                    if (has_type_mapping)
                        {
                        // last element = type
                        h.idx[group_size - 1] = ((typeval_t)m_group_typeval[cur_group]).type;
                        }
                    else
                        {
                        // last element = local group idx
                        h.idx[group_size - 1] = cur_group;
                        }

                    // list all group members j!=i in p.idx
                    unsigned int n = 0;
                    unsigned int gpos = 0;
                    for (unsigned int j = 0; j < group_size; ++j)
                        {
                        if (j == i)
                            {
                            gpos = j;
                            continue;
                            }
                        unsigned int tag2 = g.tag[j];
                        unsigned int idx2 = h_rtag.data[tag2];
                        h.idx[n++] = idx2;
                        }

                    h_gpu_table.data[m_gpu_table_indexer(idx1, num)] = h;
                    h_gpu_pos_table.data[m_gpu_table_indexer(idx1, num)] = gpos;
                    }
                }
            }
        }
    }

#ifdef ENABLE_HIP
template<unsigned int group_size, typename Group, const char* name, bool has_type_mapping>
void BondedGroupData<group_size, Group, name, has_type_mapping>::rebuildGPUTableGPU()
    {
    // resize groups counter
    m_gpu_n_groups.resize(m_pdata->getN() + m_pdata->getNGhosts());

    // resize GPU table to current number of particles
    m_gpu_table_indexer
        = Index2D(m_pdata->getN() + m_pdata->getNGhosts(), m_gpu_table_indexer.getH());
    m_gpu_table.resize(m_gpu_table_indexer.getNumElements());
    m_gpu_pos_table.resize(m_gpu_table_indexer.getNumElements());

    bool done = false;
    while (!done)
        {
        unsigned int flag = 0;

            {
            ArrayHandle<members_t> d_groups(m_groups, access_location::device, access_mode::read);
            ArrayHandle<typeval_t> d_group_typeval(m_group_typeval,
                                                   access_location::device,
                                                   access_mode::read);
            ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(),
                                             access_location::device,
                                             access_mode::read);
            ArrayHandle<unsigned int> d_n_groups(m_gpu_n_groups,
                                                 access_location::device,
                                                 access_mode::overwrite);
            ArrayHandle<members_t> d_gpu_table(m_gpu_table,
                                               access_location::device,
                                               access_mode::overwrite);
            ArrayHandle<unsigned int> d_gpu_pos_table(m_gpu_pos_table,
                                                      access_location::device,
                                                      access_mode::overwrite);
            ArrayHandle<unsigned int> d_condition(m_condition,
                                                  access_location::device,
                                                  access_mode::readwrite);

            // allocate scratch buffers
            CachedAllocator& alloc = m_exec_conf->getCachedAllocator();
            size_t tmp_size = m_groups.size() * group_size;
            unsigned int nptl = m_pdata->getN() + m_pdata->getNGhosts();
            ScopedAllocation<unsigned int> d_scratch_g(alloc, tmp_size);
            ScopedAllocation<unsigned int> d_scratch_idx(alloc, tmp_size);
            ScopedAllocation<unsigned int> d_offsets(alloc, tmp_size);

            // fill group table on GPU
            gpu_update_group_table<group_size, members_t>(getN() + getNGhosts(),
                                                          nptl,
                                                          d_groups.data,
                                                          d_group_typeval.data,
                                                          d_rtag.data,
                                                          d_n_groups.data,
                                                          m_gpu_table_indexer.getH(),
                                                          d_condition.data,
                                                          m_next_flag,
                                                          flag,
                                                          d_gpu_table.data,
                                                          d_gpu_pos_table.data,
                                                          m_gpu_table_indexer.getW(),
                                                          d_scratch_g.data,
                                                          d_scratch_idx.data,
                                                          d_offsets.data,
                                                          has_type_mapping,
                                                          m_exec_conf->getCachedAllocator());
            }
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        if (flag >= m_next_flag + 1)
            {
            // incomplete group detected
            unsigned int group_idx = flag - m_next_flag - 1;
            members_t g = m_groups[group_idx];

            std::ostringstream oss;
            oss << name << " ";
            for (unsigned int k = 0; k < group_size; ++k)
                oss << g.tag[k] << ((k != group_size - 1) ? ", " : " ");
            oss << "incomplete!";
            throw std::runtime_error(oss.str());
            }

        if (flag == m_next_flag)
            {
            // grow array by incrementing groups per particle
            m_gpu_table_indexer
                = Index2D(m_pdata->getN() + m_pdata->getNGhosts(), m_gpu_table_indexer.getH() + 1);
            m_gpu_table.resize(m_gpu_table_indexer.getNumElements());
            m_gpu_pos_table.resize(m_gpu_table_indexer.getNumElements());
            m_next_flag++;
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
template<unsigned int group_size, typename Group, const char* name, bool has_type_mapping>
std::map<unsigned int, unsigned int>
BondedGroupData<group_size, Group, name, has_type_mapping>::takeSnapshot(Snapshot& snapshot) const
    {
    // map to lookup snapshot index by tag
    std::map<unsigned int, unsigned int> index;

    std::map<unsigned int, unsigned int> rtag_map;
    for (unsigned int group_idx = 0; group_idx < getN(); group_idx++)
        {
        unsigned int tag = m_group_tag[group_idx];
        assert(m_group_rtag[tag] == group_idx);

        rtag_map.insert(std::pair<unsigned int, unsigned int>(tag, group_idx));
        }

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        // gather local data
        std::vector<typeval_t> typevals; // Group types or constraint values
        std::vector<members_t> members;  // Group members

        for (unsigned int group_idx = 0; group_idx < getN(); ++group_idx)
            {
            typevals.push_back(m_group_typeval[group_idx]);
            members.push_back(m_groups[group_idx]);
            }

        std::vector<std::vector<typeval_t>> typevals_proc; // Group types of every processor
        std::vector<std::vector<members_t>> members_proc;  // Group members of every processor

        std::vector<std::map<unsigned int, unsigned int>>
            rtag_map_proc; // List of reverse-lookup maps

        unsigned int size = m_exec_conf->getNRanks();

        // resize arrays to accumulate group data of all ranks
        typevals_proc.resize(size);
        members_proc.resize(size);
        rtag_map_proc.resize(size);

        // gather all processors' data
        gather_v(typevals, typevals_proc, 0, m_exec_conf->getMPICommunicator());
        gather_v(members, members_proc, 0, m_exec_conf->getMPICommunicator());
        gather_v(rtag_map, rtag_map_proc, 0, m_exec_conf->getMPICommunicator());

        if (m_exec_conf->getRank() == 0)
            {
            // allocate memory in snapshot
            snapshot.resize(getNGlobal());

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
            for (active_tag_it = m_tag_set.begin(); active_tag_it != m_tag_set.end();
                 ++active_tag_it)
                {
                unsigned int group_tag = *active_tag_it;
                rank_rtag_it = rank_rtag_map.find(group_tag);
                if (rank_rtag_it == rank_rtag_map.end())
                    {
                    std::ostringstream s;
                    s << "Could not find " << name << " " << group_tag << " on any processor.";
                    throw std::runtime_error(s.str());
                    }

                // store tag in index
                index.insert(std::make_pair(group_tag, snap_id));

                // rank contains the processor rank on which the particle was found
                std::pair<unsigned int, unsigned int> rank_idx = rank_rtag_it->second;
                unsigned int rank = rank_idx.first;
                unsigned int idx = rank_idx.second;

                if (has_type_mapping)
                    {
                    snapshot.type_id[snap_id] = typevals_proc[rank][idx].type;
                    }
                else
                    {
                    snapshot.val[snap_id] = typevals_proc[rank][idx].val;
                    }
                snapshot.groups[snap_id] = members_proc[rank][idx];
                snap_id++;
                }
            }
        }
    else
#endif
        {
        // allocate memory in snapshot
        snapshot.resize(getNGlobal());

        assert(getN() == getNGlobal());
        std::map<unsigned int, unsigned int>::iterator rtag_it;
        // index in snapshot
        unsigned int snap_id = 0;

        // loop through active tags
        std::set<unsigned int>::iterator active_tag_it;
        for (active_tag_it = m_tag_set.begin(); active_tag_it != m_tag_set.end(); ++active_tag_it)
            {
            unsigned int group_tag = *active_tag_it;
            rtag_it = rtag_map.find(group_tag);
            if (rtag_it == rtag_map.end())
                {
                std::ostringstream s;
                s << "Could not find " << name << " " << group_tag;
                throw std::runtime_error(s.str());
                }

            // store tag in index
            index.insert(std::make_pair(group_tag, snap_id));

            unsigned int group_idx = rtag_it->second;
            snapshot.groups[snap_id] = m_groups[group_idx];
            if (has_type_mapping)
                {
                snapshot.type_id[snap_id] = ((typeval_t)m_group_typeval[group_idx]).type;
                }
            else
                {
                snapshot.val[snap_id] = ((typeval_t)m_group_typeval[group_idx]).val;
                }
            snap_id++;
            }
        }

    snapshot.type_mapping = m_type_mapping;

    return index;
    }

#ifdef ENABLE_MPI
template<unsigned int group_size, typename Group, const char* name, bool has_type_mapping>
void BondedGroupData<group_size, Group, name, has_type_mapping>::moveParticleGroups(
    unsigned int tag,
    unsigned int old_rank,
    unsigned int new_rank)
    {
    unsigned int my_rank = m_exec_conf->getRank();

    // first remove any ghost groups
    removeAllGhostGroups();

    // move groups connected to a particle
    if (my_rank == old_rank)
        {
        std::vector<unsigned int> send_groups;

        // create a list of groups connected to the particle
        for (unsigned int group_idx = 0; group_idx < m_n_groups; ++group_idx)
            {
            members_t members = m_groups[group_idx];
            bool send = false;
            for (unsigned int i = 0; i < group_size; ++i)
                if (members.tag[i] == tag)
                    send = true;
            unsigned int group_tag = m_group_tag[group_idx];
            if (send)
                send_groups.push_back(group_tag);
            }

        MPI_Status stat;
        MPI_Request req;
        unsigned int num = (unsigned int)send_groups.size();

        MPI_Isend(&num, 1, MPI_UNSIGNED, new_rank, 0, m_exec_conf->getMPICommunicator(), &req);
        MPI_Wait(&req, &stat);

        for (std::vector<unsigned int>::iterator it = send_groups.begin(); it != send_groups.end();
             ++it)
            {
            // send group properties to other rank
            unsigned int group_tag = *it;
            unsigned int group_idx = m_group_rtag[group_tag];
            assert(group_idx != GROUP_NOT_LOCAL);

            MPI_Isend(&group_tag,
                      1,
                      MPI_UNSIGNED,
                      new_rank,
                      0,
                      m_exec_conf->getMPICommunicator(),
                      &req);
            MPI_Wait(&req, &stat);
            members_t members = m_groups[group_idx];
            MPI_Isend(&members,
                      sizeof(members_t),
                      MPI_BYTE,
                      new_rank,
                      0,
                      m_exec_conf->getMPICommunicator(),
                      &req);
            MPI_Wait(&req, &stat);
            typeval_t typeval = m_group_typeval[group_idx];
            MPI_Isend(&typeval,
                      sizeof(typeval_t),
                      MPI_BYTE,
                      new_rank,
                      0,
                      m_exec_conf->getMPICommunicator(),
                      &req);
            MPI_Wait(&req, &stat);
            }
        // remove groups that are no longer local
        for (std::vector<unsigned int>::iterator it = send_groups.begin(); it != send_groups.end();
             ++it)
            {
            unsigned int group_tag = *it;
            unsigned int group_idx = m_group_rtag[group_tag];
            members_t members = m_groups[group_idx];
            bool is_local = false;
            for (unsigned int i = 0; i < group_size; ++i)
                if (m_pdata->isParticleLocal(members.tag[i]))
                    is_local = true;

            if (!is_local)
                {
                m_group_rtag[group_tag] = GROUP_NOT_LOCAL;

                m_groups.erase(group_idx);
                m_group_typeval.erase(group_idx);
                m_group_ranks.erase(group_idx);
                m_group_tag.erase(group_idx);

                m_n_groups--;

                // reindex rtags
                ArrayHandle<unsigned int> h_group_rtag(m_group_rtag,
                                                       access_location::host,
                                                       access_mode::readwrite);
                ArrayHandle<unsigned int> h_group_tag(m_group_tag,
                                                      access_location::host,
                                                      access_mode::read);
                for (unsigned int i = 0; i < m_n_groups; ++i)
                    h_group_rtag.data[h_group_tag.data[i]] = i;
                }
            }
        }
    else if (my_rank == new_rank)
        {
        MPI_Status stat;
        MPI_Request req;

        // receive number of groups
        unsigned int num;
        MPI_Irecv(&num, 1, MPI_UNSIGNED, old_rank, 0, m_exec_conf->getMPICommunicator(), &req);
        MPI_Wait(&req, &stat);

        for (unsigned int i = 0; i < num; ++i)
            {
            unsigned int tag;
            MPI_Irecv(&tag, 1, MPI_UNSIGNED, old_rank, 0, m_exec_conf->getMPICommunicator(), &req);
            MPI_Wait(&req, &stat);

            members_t members;
            MPI_Irecv(&members,
                      sizeof(members_t),
                      MPI_BYTE,
                      old_rank,
                      0,
                      m_exec_conf->getMPICommunicator(),
                      &req);
            MPI_Wait(&req, &stat);

            typeval_t typeval;
            MPI_Irecv(&typeval,
                      sizeof(typeval_t),
                      MPI_BYTE,
                      old_rank,
                      0,
                      m_exec_conf->getMPICommunicator(),
                      &req);
            MPI_Wait(&req, &stat);

            bool is_local = m_group_rtag[tag] != NOT_LOCAL;

            // if not already local
            if (!is_local)
                {
                // append to end of group data
                unsigned int n = (unsigned int)m_groups.size();
                m_group_tag.push_back(tag);
                m_groups.push_back(members);
                m_group_typeval.push_back(typeval);
                ranks_t r;
                for (unsigned int j = 0; j < group_size; j++)
                    // initialize to zero
                    r.idx[j] = 0;

                m_n_groups++;

                m_group_ranks.push_back(r);
                m_group_rtag[tag] = n;
                }
            }
        }

    // notify observers
    m_group_num_change_signal.emit();
    notifyGroupReorder();
    }
#endif

namespace detail
    {
template<class T, typename Group>
void export_BondedGroupData(pybind11::module& m,
                            std::string name,
                            std::string snapshot_name,
                            bool export_struct)
    {
    // export group structure
    if (export_struct)
        Group::export_to_python(m);

    pybind11::class_<T, std::shared_ptr<T>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<ParticleData>>())
        .def(pybind11::init<std::shared_ptr<ParticleData>, unsigned int>())
        .def(pybind11::init<std::shared_ptr<ParticleData>, const typename T::Snapshot&>())
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

    if (T::typemap_val)
        {
        // has a type mapping
        typedef typename T::Snapshot Snapshot;
        pybind11::class_<Snapshot, std::shared_ptr<Snapshot>>(m, snapshot_name.c_str())
            .def(pybind11::init<unsigned int>())
            .def_property_readonly("typeid", &Snapshot::getTypeNP)
            .def_property_readonly("group", &Snapshot::getBondedTagsNP)
            .def_property("types", &Snapshot::getTypes, &Snapshot::setTypes)
            .def_property("N", &Snapshot::getSize, &Snapshot::resize);
        }
    else
        {
        // has Scalar values
        typedef typename T::Snapshot Snapshot;
        pybind11::class_<Snapshot, std::shared_ptr<Snapshot>>(m, snapshot_name.c_str())
            .def(pybind11::init<unsigned int>())
            .def_property_readonly("value", &Snapshot::getValueNP)
            .def_property_readonly("group", &Snapshot::getBondedTagsNP)
            .def_property("N", &Snapshot::getSize, &Snapshot::resize);
        }
    }

    } // end namespace detail

template<unsigned int group_size, typename Group, const char* name, bool has_type_mapping>
void BondedGroupData<group_size, Group, name, has_type_mapping>::Snapshot::replicate(
    unsigned int n,
    unsigned int old_n_particles)
    {
    unsigned int old_size = size;
    groups.resize(n * old_size);
    if (has_type_mapping)
        {
        type_id.resize(n * old_size);
        }
    else
        {
        val.resize(n * old_size);
        }

    for (unsigned int i = 0; i < old_size; ++i)
        {
        typename BondedGroupData<group_size, Group, name, has_type_mapping>::members_t g;
        g = groups[i];

        // replicate bonded group
        for (unsigned int j = 0; j < n; ++j)
            {
            typename BondedGroupData<group_size, Group, name, has_type_mapping>::members_t h;

            // update particle tags
            for (unsigned int k = 0; k < group_size; ++k)
                h.tag[k] = g.tag[k] + old_n_particles * j;

            groups[old_size * j + i] = h;
            if (has_type_mapping)
                {
                type_id[old_size * j + i] = type_id[i];
                }
            else
                {
                val[old_size * j + i] = val[i];
                }
            }
        }

    size = n * old_size;
    }

/*! \returns a numpy array that wraps the type_id data element.
    The raw data is referenced by the numpy array, modifications to the numpy array will modify the
   snapshot
*/
template<unsigned int group_size, typename Group, const char* name, bool has_type_mapping>
pybind11::object BondedGroupData<group_size, Group, name, has_type_mapping>::Snapshot::getTypeNP(
    pybind11::object self)
    {
    assert(has_type_mapping);
    auto self_cpp
        = self.cast<BondedGroupData<group_size, Group, name, has_type_mapping>::Snapshot*>();

    if (self_cpp->type_id.size() == 0)
        {
        return pybind11::array(pybind11::dtype::of<unsigned int>(), 0, nullptr);
        }
    return pybind11::array(self_cpp->type_id.size(), self_cpp->type_id.data(), self);
    }

/*! \returns a numpy array that wraps the value data element.
    The raw data is referenced by the numpy array, modifications to the numpy array will modify the
   snapshot
*/
template<unsigned int group_size, typename Group, const char* name, bool has_type_mapping>
pybind11::object BondedGroupData<group_size, Group, name, has_type_mapping>::Snapshot::getValueNP(
    pybind11::object self)
    {
    assert(!has_type_mapping);
    auto self_cpp
        = self.cast<BondedGroupData<group_size, Group, name, has_type_mapping>::Snapshot*>();

    if (self_cpp->val.size() == 0)
        {
        return pybind11::array(pybind11::dtype::of<Scalar>(), 0, nullptr);
        }
    return pybind11::array(self_cpp->val.size(), self_cpp->val.data(), self);
    }

/*! \returns a numpy array that wraps the groups data element.
    The raw data is referenced by the numpy array, modifications to the numpy array will modify the
   snapshot
*/
template<unsigned int group_size, typename Group, const char* name, bool has_type_mapping>
pybind11::object
BondedGroupData<group_size, Group, name, has_type_mapping>::Snapshot::getBondedTagsNP(
    pybind11::object self)
    {
    auto self_cpp
        = self.cast<BondedGroupData<group_size, Group, name, has_type_mapping>::Snapshot*>();
    std::vector<size_t> dims(2);
    dims[0] = self_cpp->groups.size();
    dims[1] = group_size;

    if (dims[0] == 0)
        {
        return pybind11::array(pybind11::dtype::of<unsigned int>(), dims, nullptr);
        }
    return pybind11::array(dims, (unsigned int*)self_cpp->groups.data(), self);
    }

/*! \returns A python list of type names
 */
template<unsigned int group_size, typename Group, const char* name, bool has_type_mapping>
pybind11::list BondedGroupData<group_size, Group, name, has_type_mapping>::Snapshot::getTypes()
    {
    pybind11::list types;

    for (unsigned int i = 0; i < this->type_mapping.size(); i++)
        types.append(pybind11::str(this->type_mapping[i]));

    return types;
    }

/*! \param types Python list of type names to set
 */
template<unsigned int group_size, typename Group, const char* name, bool has_type_mapping>
void BondedGroupData<group_size, Group, name, has_type_mapping>::Snapshot::setTypes(
    pybind11::list types)
    {
    type_mapping.resize(len(types));

    for (unsigned int i = 0; i < len(types); i++)
        this->type_mapping[i] = pybind11::cast<string>(types[i]);
    }

template class PYBIND11_EXPORT BondedGroupData<2, Bond, name_bond_data>;
template class PYBIND11_EXPORT BondedGroupData<4, MeshBond, name_meshbond_data>;
template class PYBIND11_EXPORT BondedGroupData<3, Angle, name_angle_data>;
template class PYBIND11_EXPORT BondedGroupData<3, Angle, name_triangle_data>;
template class PYBIND11_EXPORT BondedGroupData<4, Dihedral, name_dihedral_data>;
template class PYBIND11_EXPORT BondedGroupData<4, Dihedral, name_improper_data>;
template class PYBIND11_EXPORT BondedGroupData<2, Constraint, name_constraint_data, false>;
template class PYBIND11_EXPORT BondedGroupData<2, Bond, name_pair_data>;

namespace detail
    {
template void export_BondedGroupData<BondData, Bond>(pybind11::module& m,
                                                     std::string name,
                                                     std::string snapshot_name,
                                                     bool export_struct);

template void export_BondedGroupData<AngleData, Angle>(pybind11::module& m,
                                                       std::string name,
                                                       std::string snapshot_name,
                                                       bool export_struct);

template void export_BondedGroupData<TriangleData, Angle>(pybind11::module& m,
                                                          std::string name,
                                                          std::string snapshot_name,
                                                          bool export_struct);

template void export_BondedGroupData<DihedralData, Dihedral>(pybind11::module& m,
                                                             std::string name,
                                                             std::string snapshot_name,
                                                             bool export_struct);

template void export_BondedGroupData<ImproperData, Dihedral>(pybind11::module& m,
                                                             std::string name,
                                                             std::string snapshot_name,
                                                             bool export_struct);

template void export_BondedGroupData<ConstraintData, Constraint>(pybind11::module& m,
                                                                 std::string name,
                                                                 std::string snapshot_name,
                                                                 bool export_struct);

template void export_BondedGroupData<PairData, Bond>(pybind11::module& m,
                                                     std::string name,
                                                     std::string snapshot_name,
                                                     bool export_struct);

    } // end namespace detail

    } // end namespace hoomd
