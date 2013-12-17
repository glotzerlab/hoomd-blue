/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

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

/*! \file BondedGroupData.h
    \brief Defines implementation of BondedGroupData
 */


#include "BondedGroupData.h"
#include "ParticleData.h"
#include "Index1D.h"

#ifdef ENABLE_CUDA
#include "BondedGroupData.cuh"
#include "CachedAllocator.h"
#endif

//! Names of bonded groups
char name_bond_data[] = "bond";
char name_angle_data[] = "angle";
char name_dihedral_data[] = "dihedral";
char name_improper_data[] = "improper";

/*
 * Implementation of BondedGroupData methods
 */

/*! \param exec_conf Execution configuration
    \param pdata The particle data to associate with
    \param n_group_types Number of bonded group types to initialize
 */
template<unsigned int group_size, typename Group, const char *name>
BondedGroupData<group_size, Group, name>::BondedGroupData(
    boost::shared_ptr<ParticleData> pdata,
    unsigned int n_group_types)
    : m_exec_conf(pdata->getExecConf()), m_pdata(pdata), m_nglobal(0), m_groups_dirty(true)
    {
    m_exec_conf->msg->notice(5) << "Constructing BondedGroupData (" << name<< "s, n=" << group_size << ") "
        << endl;

    // connect to particle sort signal
    m_sort_connection = m_pdata->connectParticleSort(boost::bind(&BondedGroupData<group_size, Group, name>::setDirty, this));
    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        m_pdata->connectSingleParticleMove(
            boost::bind(&BondedGroupData<group_size, Group, name>::moveParticleGroups, this, _1, _2, _3));
        }
    #endif

    // offer a default type mapping
    for (unsigned int i = 0; i < n_group_types; i++)
        {
        char suffix[2];
        suffix[0] = 'A' + i;
        suffix[1] = '\0';

        std::string type_name = std::string(name) + std::string(suffix);
        m_type_mapping.push_back(type_name);
        }

    // initialize data structures
    initialize();

    #ifdef ENABLE_CUDA
    if (m_exec_conf->isCUDAEnabled())
        {
        // create a ModernGPU context
        m_mgpu_context = mgpu::CreateCudaDeviceAttachStream(0);
        }
    #endif
    }

/*! \param exec_conf Execution configuration
    \param pdata The particle data to associate with
    \param snapshot Snapshot to initialize from
 */
template<unsigned int group_size, typename Group, const char *name>
BondedGroupData<group_size, Group, name>::BondedGroupData(
    boost::shared_ptr<ParticleData> pdata,
    const Snapshot& snapshot)
    : m_exec_conf(pdata->getExecConf()), m_pdata(pdata), m_nglobal(0), m_groups_dirty(true)
    {
    m_exec_conf->msg->notice(5) << "Constructing BondedGroupData (" << name << ") " << endl;

    // connect to particle sort signal
    m_sort_connection = m_pdata->connectParticleSort(boost::bind(&BondedGroupData<group_size, Group, name>::setDirty,
        this));

    #ifdef ENABLE_CUDA
    if (m_exec_conf->isCUDAEnabled())
        {
        // create a ModernGPU context
        m_mgpu_context = mgpu::CreateCudaDeviceAttachStream(0);
        }
    #endif

    // initialize from snapshot
    initializeFromSnapshot(snapshot);

    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        m_pdata->connectSingleParticleMove(
            boost::bind(&BondedGroupData<group_size, Group, name>::moveParticleGroups, this, _1, _2, _3));
        }
    #endif
    }

//! Destructor
template<unsigned int group_size, typename Group, const char *name>
BondedGroupData<group_size, Group, name>::~BondedGroupData()
    {
    m_sort_connection.disconnect();
    #ifdef ENABLE_MPI
    if (m_particle_move_connection.connected())
        m_particle_move_connection.disconnect();
    #endif
    }

template<unsigned int group_size, typename Group, const char *name>
void BondedGroupData<group_size, Group, name>::initialize()
    {
    m_nglobal = 0;

    // clear set of active tags
    m_tag_set.clear();

    // clear reservoir of recycled tags
    while (! m_recycled_tags.empty())
        m_recycled_tags.pop();

    // allocate arrays
    GPUVector<members_t> groups(m_exec_conf);
    m_groups.swap(groups);

    GPUVector<unsigned int> type_id(m_exec_conf);
    m_group_type.swap(type_id);

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
    m_n_groups.swap(n_groups);

    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        GPUVector<ranks_t> group_ranks(m_exec_conf);
        m_group_ranks.swap(group_ranks);
        }
    #endif

    // allocate stand-by arrays
    GPUVector<unsigned int> type_id_alt(m_exec_conf);
    m_group_type_alt.swap(type_id_alt);

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

    #ifdef ENABLE_CUDA
    // allocate condition variable
    GPUArray<unsigned int> condition(1, m_exec_conf);
    m_condition.swap(condition);

    ArrayHandle<unsigned int> h_condition(m_condition, access_location::host, access_mode::overwrite);
    *h_condition.data = 0;
    m_next_flag = 1;
    #endif

    m_tag_set.clear();
    }

//! Initialize from a snapshot
template<unsigned int group_size, typename Group, const char *name>
void BondedGroupData<group_size, Group, name>::initializeFromSnapshot(const Snapshot& snapshot)
    {
    // check that all fields in the snapshot have correct length
    if (m_exec_conf->getRank() == 0 && ! snapshot.validate())
        {
        m_exec_conf->msg->error() << "init.*: invalid " << name << " data snapshot."
                                << std::endl << std::endl;
        throw std::runtime_error(std::string("Error initializing ") + name + std::string(" data."));
        }

    // re-initialize data structures
    initialize();

    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        // broadcast to all processors (temporarily)
        std::vector<members_t> all_groups;
        std::vector<unsigned int> all_type;

        if (m_exec_conf->getRank() == 0)
            {
            all_groups = snapshot.groups;
            all_type = snapshot.type_id;
            m_type_mapping = snapshot.type_mapping;
            }

        bcast(all_groups, 0, m_exec_conf->getMPICommunicator());
        bcast(all_type, 0, m_exec_conf->getMPICommunicator());
        bcast(m_type_mapping, 0, m_exec_conf->getMPICommunicator());

        // iterate over groups and add those that have local particles
        for (unsigned int group_tag = 0; group_tag < all_groups.size(); ++group_tag)
            addBondedGroup(Group(all_type[group_tag], all_groups[group_tag]));
        }
    else
    #endif
        {
        m_type_mapping = snapshot.type_mapping;

        for (unsigned group_idx = 0; group_idx < snapshot.groups.size(); group_idx++)
            addBondedGroup(Group(snapshot.type_id[group_idx], snapshot.groups[group_idx]));
        }
    }

/*! \param type_id Type of bonded group to add
    \param member_tags Particle members of group
 */
template<unsigned int group_size, typename Group, const char *name>
unsigned int BondedGroupData<group_size, Group, name>::addBondedGroup(Group g)
    {
    unsigned int type = g.get_type();
    members_t member_tags = g.get_members();

    // check for some silly errors a user could make
    for (unsigned int i = 0; i < group_size; ++i)
        if (member_tags.tag[i] >= m_pdata->getNGlobal())
            {
            std::ostringstream oss;
            oss << name << ".*: Particle tag out of bounds when attempting to add " << name << ": ";
            for (unsigned int j = 0; j < group_size; ++j)
                oss << member_tags.tag[j] << ((j != group_size - 1) ? "," : "");
            oss << std::endl;
            m_exec_conf->msg->error() << oss.str();
            throw runtime_error(std::string("Error adding ") + name);
            }

    for (unsigned int i = 0; i < group_size; ++i)
        for (unsigned int j = 0; j < group_size; ++j)
            if (i != j && member_tags.tag[i] == member_tags.tag[j])
                {
                std::ostringstream oss;
                oss << name << ".*: The same particle can only occur once in a " << name << ": ";
                for (unsigned int k = 0; k < group_size; ++k)
                    oss << member_tags.tag[k] << ((k != group_size - 1) ? "," : "");
                oss << std::endl;
                m_exec_conf->msg->error() << oss.str();
                throw runtime_error(std::string("Error adding ") + name);
                }

    if (type >= m_type_mapping.size())
        {
        m_exec_conf->msg->error() << name << ".*: Invalid " << name << " type " << type
            << "! The  number of types is " << m_type_mapping.size() << std::endl;
        throw std::runtime_error(std::string("Error adding ") + name);
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
        m_group_type.push_back(type);
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
        }

    // add to set of active tags
    m_tag_set.insert(tag);

    // increment number of bonded groups
    m_nglobal++;

    // set flag to rebuild GPU table
    m_groups_dirty = true;

    // notifiy observers
    m_group_num_change_signal();

    return tag;
    }

//! Return the nth active global tag
/*! \param n Index of bond in global bond table
 */
template<unsigned int group_size, typename Group, const char *name>
unsigned int BondedGroupData<group_size, Group, name>::getNthTag(unsigned int n) const
    {
   if (n >= getNGlobal())
        {
        m_exec_conf->msg->error() << name << ".*: " << name << " index " << n << " out of bounds!"
            << "The number of " << name << "s is " << getNGlobal() << std::endl;
        throw std::runtime_error(std::string("Error getting ") + name);
        }

    assert(m_tag_set.size() == getNGlobal());
    std::set<unsigned int>::const_iterator it = m_tag_set.begin();
    std::advance(it, n);
    return *it;
    }


/*! \param tag Tag of bonded group
 * \returns the group
 */
template<unsigned int group_size, typename Group, const char *name>
const Group BondedGroupData<group_size, Group, name>::getGroupByTag(unsigned int tag) const
    {
    // Find position of bonded group in list
    unsigned int group_idx = m_group_rtag[tag];

    unsigned int type;
    members_t members;

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        int my_rank = m_exec_conf->getRank();
        // set local to rank if the bond is local, -1 if not
        int rank = group_idx < m_groups.size() ? my_rank : -1;

        // the highest rank owning the group sends it to the others
        MPI_Allreduce(MPI_IN_PLACE,
                      &rank,
                      1,
                      MPI_INT,
                      MPI_MAX,
                      m_exec_conf->getMPICommunicator());

        if (rank == -1)
            {
            m_exec_conf->msg->error() << "Trying to find " << name << " " << tag
                 << " which does not exist!" << endl;
            throw runtime_error(std::string("Error getting ") + name);
            }

        if (rank == (int)my_rank)
            {
            type = m_group_type[group_idx];
            members = m_groups[group_idx];
            }

        bcast(type, rank, m_exec_conf->getMPICommunicator());
        bcast(members, rank, m_exec_conf->getMPICommunicator());
        }
    #endif
    else
        {
        if (group_idx == GROUP_NOT_LOCAL)
            {
            m_exec_conf->msg->error() << "Trying to get type of " << name << " " << tag
                 << " which does not exist!" << endl;
            throw runtime_error(std::string("Error getting ") + name);
            }

        type = m_group_type[group_idx];
        members = m_groups[group_idx];
        }

    return Group(type,members);
    }

/*! \param idx Tag of bonded group
 * \return Member tags of bonded group
 */
template<unsigned int group_size, typename Group, const char *name>
unsigned int BondedGroupData<group_size, Group, name>::getTypeByIndex(unsigned int group_idx) const
    {
    assert (group_idx < getN());
    return m_group_type[group_idx];
    }

/*! \param idx Tag of bonded group
 * \return Type of bonded group
 */
template<unsigned int group_size, typename Group, const char *name>
const typename BondedGroupData<group_size, Group, name>::members_t BondedGroupData<group_size, Group, name>::getMembersByIndex(unsigned int group_idx) const
    {
    assert (group_idx < getN());
    return m_groups[group_idx];
    }

/*! \param tag Tag of bonded group to remove
 */
template<unsigned int group_size, typename Group, const char *name>
void BondedGroupData<group_size, Group, name>::removeBondedGroup(unsigned int tag)
    {
    // sanity check
    if (tag >= m_group_rtag.size())
        {
        m_exec_conf->msg->error() << "Trying to remove " << name << " " << tag << " which does not exist!" << endl;
        throw runtime_error(std::string("Error removing ") + name);
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
        MPI_Allreduce(MPI_IN_PLACE,
                      &res,
                      1,
                      MPI_INT,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());

        assert((unsigned int) res <= group_size);
        is_available = res;
        }
    #endif

    if (! is_available)
        {
        m_exec_conf->msg->error() << "Trying to remove " << name << " " << tag
             << " which has been previously removed!" << endl;
        throw runtime_error(std::string("Error removing ") + name);
        }

    // delete from map
    m_group_rtag[tag] = GROUP_NOT_LOCAL;

    if (is_local)
        {
        unsigned int size = m_groups.size();
        // If the bonded group is in the middle of the list, move the last element to
        // to the position of the removed element
        if (id < (size-1))
            {
            m_groups[id] = (members_t) m_groups[size-1];
            m_group_type[id] = (unsigned int) m_group_type[size-1];
            unsigned int last_tag = m_group_tag[size-1];
            m_group_rtag[last_tag] = id;
            m_group_tag[id] = last_tag;
            }

        // delete last element
        m_groups.pop_back();
        m_group_type.pop_back();
        m_group_tag.pop_back();
        #ifdef ENABLE_MPI
        if (m_pdata->getDomainDecomposition())
            m_group_ranks.pop_back();
        #endif
        }

    // remove from set of active tags
    m_tag_set.erase(tag);

    // maintain a stack of deleted group tags for future recycling
    m_recycled_tags.push(tag);
    m_nglobal--;

    // set flag to trigger rebuild of GPU table
    m_groups_dirty = true;

    // notifiy observers
    m_group_num_change_signal();
    }

/*! \param name Type name
 */
template<unsigned int group_size, typename Group, const char *name>
unsigned int BondedGroupData<group_size, Group, name>::getTypeByName(const std::string &type_name) const
    {
    // search for the name
    for (unsigned int i = 0; i < m_type_mapping.size(); i++)
        {
        if (m_type_mapping[i] == type_name)
            return i;
        }

    m_exec_conf->msg->error() << name << " type " << type_name << " not found!" << endl;
    throw runtime_error("Error mapping type name");

    // silence compiler warning
    return 0;
    }

template<unsigned int group_size, typename Group, const char *name>
const std::string BondedGroupData<group_size, Group, name>::getNameByType(unsigned int type) const
    {
    // check for an invalid request
    if (type >= m_type_mapping.size())
        {
        m_exec_conf->msg->error() << "Requesting type name for non-existant type " << type << endl;
        throw runtime_error("Error mapping type name");
        }

    // return the name
    return m_type_mapping[type];
    }

template<unsigned int group_size, typename Group, const char *name>
void BondedGroupData<group_size, Group, name>::rebuildGPUTable()
    {
    #ifdef ENABLE_CUDA
    if (m_exec_conf->isCUDAEnabled())
        rebuildGPUTableGPU();
    else
    #endif
        {
        if (m_prof) m_prof->push("update " + std::string(name) + " table");

        ArrayHandle< unsigned int > h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

        m_n_groups.resize(m_pdata->getN()+m_pdata->getNGhosts());

        unsigned int num_groups_max = 0;
            {
            ArrayHandle<unsigned int> h_n_groups(m_n_groups, access_location::host, access_mode::overwrite);

            unsigned int N = m_pdata->getN()+m_pdata->getNGhosts();
            // count the number of bonded groups per particle
            // start by initializing the n_groups values to 0
            memset(h_n_groups.data, 0, sizeof(unsigned int) * N);

            // loop through the particles and count the number of groups based on each particle index
            for (unsigned int cur_group = 0; cur_group < getN(); cur_group++)
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
                        oss << name << ".*: " << name << " ";
                        for (unsigned int k = 0; k < group_size; ++k)
                            oss << g.tag[k] << ((k != group_size - 1) ? "," : "");
                        oss << " incomplete!" << std::endl;
                        m_exec_conf->msg->error() << oss.str();
                        throw std::runtime_error("Error building GPU group table.");
                        }

                    h_n_groups.data[idx]++;
                    }
                }

            // find the maximum number of groups
            for (unsigned int i = 0; i < N; i++)
                if (h_n_groups.data[i] > num_groups_max)
                    num_groups_max = h_n_groups.data[i];
            }

        // resize lookup table
        m_gpu_table_indexer = Index2D(m_pdata->getN()+m_pdata->getNGhosts(), num_groups_max);
        m_gpu_table.resize(m_gpu_table_indexer.getNumElements());
        m_gpu_pos_table.resize(m_gpu_table_indexer.getNumElements());

            {
            ArrayHandle<unsigned int> h_n_groups(m_n_groups, access_location::host, access_mode::overwrite);
            ArrayHandle<members_t> h_gpu_table(m_gpu_table, access_location::host, access_mode::overwrite);
            ArrayHandle<unsigned int> h_gpu_pos_table(m_gpu_pos_table, access_location::host, access_mode::overwrite);

            // now, update the actual table
            // zero the number of bonded groups counter (again)
            memset(h_n_groups.data, 0, sizeof(unsigned int) * (m_pdata->getN()+m_pdata->getNGhosts()));

            // loop through all group and add them to each column in the list
            for (unsigned int cur_group = 0; cur_group < getN(); cur_group++)
                {
                members_t g = m_groups[cur_group];

                for (unsigned int i = 0; i < group_size; ++i)
                    {
                    unsigned int tag1 = g.tag[i];
                    unsigned int idx1 = h_rtag.data[tag1];
                    unsigned int num = h_n_groups.data[idx1]++;

                    members_t h;
                    // last element = type
                    h.idx[group_size-1] = m_group_type[cur_group];

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

        if (m_prof) m_prof->pop();
        }
    }

#ifdef ENABLE_CUDA
template<unsigned int group_size, typename Group, const char *name>
void BondedGroupData<group_size, Group, name>::rebuildGPUTableGPU()
    {
    if (m_prof) m_prof->push(m_exec_conf, "update " + std::string(name) + " table");

    // resize groups counter
    m_n_groups.resize(m_pdata->getN()+m_pdata->getNGhosts());

    // resize GPU table to current number of particles
    m_gpu_table_indexer = Index2D(m_pdata->getN()+m_pdata->getNGhosts(), m_gpu_table_indexer.getH());
    m_gpu_table.resize(m_gpu_table_indexer.getNumElements());
    m_gpu_pos_table.resize(m_gpu_table_indexer.getNumElements());

    bool done = false;
    while (!done)
        {
        unsigned int flag = 0;

            {
            ArrayHandle<members_t> d_groups(m_groups, access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_group_type(m_group_type, access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_n_groups(m_n_groups, access_location::device, access_mode::overwrite);
            ArrayHandle<members_t> d_gpu_table(m_gpu_table, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_gpu_pos_table(m_gpu_pos_table, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_condition(m_condition, access_location::device, access_mode::readwrite);

            // allocate scratch buffers
            const CachedAllocator& alloc = m_exec_conf->getCachedAllocator();
            unsigned int tmp_size = m_groups.size()*group_size;
            unsigned int nptl = m_pdata->getN()+m_pdata->getNGhosts();
            ScopedAllocation<unsigned int> d_scratch_g(alloc, tmp_size);
            ScopedAllocation<unsigned int> d_scratch_idx(alloc, tmp_size);
            ScopedAllocation<unsigned int> d_offsets(alloc, tmp_size);
            ScopedAllocation<unsigned int> d_seg_offsets(alloc, nptl);

            // fill group table on GPU
            gpu_update_group_table<group_size, members_t>(
                m_groups.size(),
                nptl,
                d_groups.data,
                d_group_type.data,
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
                d_seg_offsets.data,
                m_mgpu_context);
            }
        if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();

        if (flag >= m_next_flag+1)
            {
            // incomplete group detected
            unsigned int group_idx = flag - m_next_flag - 1;
            members_t g = m_groups[group_idx];

            std::ostringstream oss;
            oss << name << ".*: " << name << " ";
            for (unsigned int k = 0; k < group_size; ++k)
                oss << g.tag[k] << ((k != group_size - 1) ? "," : "");
            oss << " incomplete!" << std::endl;
            m_exec_conf->msg->error() << oss.str();
            throw std::runtime_error("Error building GPU group table.");
            }

        if (flag == m_next_flag)
            {
            // grow array by incrementing groups per particle
            m_gpu_table_indexer = Index2D(m_pdata->getN()+m_pdata->getNGhosts(), m_gpu_table_indexer.getH()+1);
            m_gpu_table.resize(m_gpu_table_indexer.getNumElements());
            m_gpu_pos_table.resize(m_gpu_table_indexer.getNumElements());
            m_next_flag++;
            }
        else
            done = true;
        }

    if (m_prof) m_prof->pop(m_exec_conf);
    }
#endif

/*! \param snapshot Snapshot that will contain the group data
 *
 *  Data in the snapshot is in tag order, where non-existant tags are skipped
 */
template<unsigned int group_size, typename Group, const char *name>
void BondedGroupData<group_size, Group, name>::takeSnapshot(Snapshot& snapshot) const
    {
    // allocate memory in snapshot
    snapshot.resize(getNGlobal());

    std::map<unsigned int, unsigned int> rtag_map;

    for (unsigned int group_idx = 0; group_idx < getN(); group_idx++)
        {
        unsigned int tag = m_group_tag[group_idx];
        assert(m_group_rtag[tag] == group_idx);

        rtag_map.insert(std::pair<unsigned int,unsigned int>(tag, group_idx));
        }

    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        // gather local data
        std::vector<unsigned int> types; // Group types
        std::vector<members_t> members;  // Group members

        for (unsigned int group_idx  = 0; group_idx < getN(); ++group_idx)
            {
            types.push_back(m_group_type[group_idx]);
            members.push_back(m_groups[group_idx]);
            }

        std::vector< std::vector<unsigned int> > types_proc;     // Group types of every processor
        std::vector< std::vector<members_t> > members_proc;      // Group members of every processor

        std::vector< std::map<unsigned int, unsigned int> > rtag_map_proc; // List of reverse-lookup maps

        unsigned int size = m_exec_conf->getNRanks();

        // resize arrays to accumulate group data of all ranks
        types_proc.resize(size);
        members_proc.resize(size);
        rtag_map_proc.resize(size);

        // gather all processors' data
        gather_v(types, types_proc, 0, m_exec_conf->getMPICommunicator());
        gather_v(members, members_proc, 0, m_exec_conf->getMPICommunicator());
        gather_v(rtag_map, rtag_map_proc, 0, m_exec_conf->getMPICommunicator());

        if (m_exec_conf->getRank() == 0)
            {
            assert(rtag_map_proc.size() == size);

            // create single map of all group ranks and indices
            // groups present on more than one processor will count as one group
            std::map<unsigned int, std::pair<unsigned int, unsigned int> > rank_rtag_map;
            std::map<unsigned int, unsigned int>::iterator it;
            for (unsigned int irank = 0; irank < size; ++irank)
                for (it = rtag_map_proc[irank].begin(); it != rtag_map_proc[irank].end(); ++it)
                    rank_rtag_map.insert(std::make_pair(it->first, std::make_pair(irank, it->second)));

            // add groups to snapshot
            std::map<unsigned int, std::pair<unsigned int, unsigned int> >::iterator rank_rtag_it;
            for (unsigned int group_tag = 0; group_tag < getNGlobal(); group_tag++)
                {
                rank_rtag_it = rank_rtag_map.find(group_tag);
                if (rank_rtag_it == rank_rtag_map.end())
                    {
                    m_exec_conf->msg->error()
                        << endl << "Could not find " << name << " " << group_tag << " on any processor. "
                        << endl << endl;
                    throw std::runtime_error("Error gathering "+std::string(name)+"s");
                    }

                // rank contains the processor rank on which the particle was found
                std::pair<unsigned int, unsigned int> rank_idx = rank_rtag_it->second;
                unsigned int rank = rank_idx.first;
                unsigned int idx = rank_idx.second;

                snapshot.type_id[group_tag] = types_proc[rank][idx];
                snapshot.groups[group_tag] = members_proc[rank][idx];
                }
            }
        }
    else
    #endif
        {
        assert(getN() == getNGlobal());
        std::map<unsigned int, unsigned int>::iterator rtag_it;
        for (rtag_it = rtag_map.begin(); rtag_it != rtag_map.end(); ++rtag_it)
            {
            unsigned int group_idx = rtag_it->second;
            snapshot.groups[group_idx] = m_groups[group_idx];
            snapshot.type_id[group_idx] = m_group_type[group_idx];
            }
        }

    snapshot.type_mapping = m_type_mapping;
    }

#ifdef ENABLE_MPI
template<unsigned int group_size, typename Group, const char *name>
void BondedGroupData<group_size, Group, name>::moveParticleGroups(unsigned int tag, unsigned int old_rank, unsigned int new_rank)
    {
    unsigned int my_rank = m_exec_conf->getRank();

    // move groups connected to a particle
    if (my_rank == old_rank)
        {
        std::vector<unsigned int> send_groups;

        // create a list of groups connected to the particle
        for (unsigned int group_idx = 0; group_idx < m_groups.size(); ++group_idx)
            {
            members_t members = m_groups[group_idx];
            bool send = false;
            for (unsigned int i = 0; i < group_size; ++i)
                if (members.tag[i] == tag) send = true;
            if (send) send_groups.push_back(group_idx);
            }

        MPI_Status stat;
        MPI_Request req;
        unsigned int num = send_groups.size();

        MPI_Isend(&num, 1, MPI_UNSIGNED, new_rank, 0, m_exec_conf->getMPICommunicator(), &req);
        MPI_Wait(&req, &stat);

        for (std::vector<unsigned int>::iterator it = send_groups.begin(); it != send_groups.end(); ++it)
            {
            // send group properties to other rank
            unsigned int group_tag = m_group_tag[*it];

            MPI_Isend(&group_tag, 1, MPI_UNSIGNED, new_rank, 0, m_exec_conf->getMPICommunicator(), &req);
            MPI_Wait(&req, &stat);
            members_t members = m_groups[*it];
            MPI_Isend(&members, sizeof(members_t), MPI_BYTE, new_rank, 0, m_exec_conf->getMPICommunicator(), &req);
            MPI_Wait(&req, &stat);
            unsigned int type = m_group_type[*it];
            MPI_Isend(&type, 1, MPI_UNSIGNED, new_rank, 0, m_exec_conf->getMPICommunicator(), &req);
            MPI_Wait(&req, &stat);
            }
        // remove groups that are no longer local
        for (std::vector<unsigned int>::iterator it = send_groups.begin(); it != send_groups.end(); ++it)
            {
            members_t members = m_groups[*it];
            bool is_local = false;
            for (unsigned int i = 0; i < group_size; ++i)
                if (m_pdata->isParticleLocal(members.tag[i])) is_local = true;

            if (!is_local)
                {
                unsigned int tag = m_group_tag[*it];
                m_group_rtag[tag] = GROUP_NOT_LOCAL;

                m_groups.erase(*it);
                m_group_type.erase(*it);
                m_group_ranks.erase(*it);
                m_group_tag.erase(*it);
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

        for (unsigned int i =0; i < num; ++i)
            {
            unsigned int tag;
            MPI_Irecv(&tag, 1, MPI_UNSIGNED, old_rank, 0, m_exec_conf->getMPICommunicator(), &req);
            MPI_Wait(&req, &stat);

            members_t members;
            MPI_Irecv(&members, sizeof(members_t), MPI_BYTE, old_rank, 0, m_exec_conf->getMPICommunicator(), &req);
            MPI_Wait(&req, &stat);

            unsigned int type;
            MPI_Irecv(&type, 1, MPI_UNSIGNED, old_rank, 0, m_exec_conf->getMPICommunicator(), &req);
            MPI_Wait(&req, &stat);

            bool is_local = m_group_rtag[tag] != NOT_LOCAL;

            // if not already local
            if (! is_local)
                {
                // append to end of group data
                unsigned int n = m_groups.size();
                m_group_tag.push_back(tag);
                m_groups.push_back(members);
                m_group_type.push_back(type);
                ranks_t r;
                for (unsigned int j = 0; j < group_size; j++)
                    // initialize to zero
                    r.idx[j] = 0;

                m_group_ranks.push_back(r);
                m_group_rtag[tag] = n;
                }
            }
        }

    // notify observers
    m_group_num_change_signal();
    m_groups_dirty = true;
    }

#endif
template<class T, typename Group>
void export_BondedGroupData(std::string name, std::string snapshot_name, bool export_struct)
    {
    // export group structure
    if (export_struct)
        Group::export_to_python();

    scope outer = class_<T, boost::shared_ptr<T> , boost::noncopyable>(name.c_str(),
        init<boost::shared_ptr<ParticleData>, unsigned int>())
        .def(init<boost::shared_ptr<ParticleData>, const typename T::Snapshot& >())
        .def("initializeFromSnapshot", &T::initializeFromSnapshot)
        .def("takeSnapshot", &T::takeSnapshot)
        .def("getN", &T::getN)
        .def("getNGlobal", &T::getNGlobal)
        .def("getNTypes", &T::getNTypes)
        .def("getNthTag", &T::getNthTag)
        .def("getGroupByTag", &T::getGroupByTag)
        .def("getTypeByName", &T::getTypeByName)
        .def("getNameByType", &T::getNameByType)
        .def("addBondedGroup", &T::addBondedGroup)
        .def("removeBondedGroup", &T::removeBondedGroup)
        .def("setProfiler", &T::setProfiler)
        ;

    typedef typename T::Snapshot Snapshot;
    class_<Snapshot, boost::shared_ptr<Snapshot> >
        (snapshot_name.c_str(), init<unsigned int>())
        .def_readwrite("groups", &Snapshot::groups)
        .def_readwrite("type_id", &Snapshot::type_id)
        .def_readwrite("type_mapping", &Snapshot::type_mapping)
        ;
   }

//! Explicit template instantiations
template class BondedGroupData<2, Bond, name_bond_data>;
template void export_BondedGroupData<BondData,Bond>(std::string name,std::string snapshot_name, bool export_struct);

template class BondedGroupData<3, Angle, name_angle_data>;
template void export_BondedGroupData<AngleData,Angle>(std::string name,std::string snapshot_name, bool export_struct);

template class BondedGroupData<4, Dihedral, name_dihedral_data>;
template void export_BondedGroupData<DihedralData,Dihedral>(std::string name,std::string snapshot_name, bool export_struct);

template class BondedGroupData<4, Dihedral, name_improper_data>;
template void export_BondedGroupData<ImproperData,Dihedral>(std::string name,std::string snapshot_name, bool export_struct);
