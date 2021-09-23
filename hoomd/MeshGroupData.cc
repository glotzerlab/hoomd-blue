// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: jglaser

/*! \file BondedGroupData.h
    \brief Defines implementation of BondedGroupData
 */

#include "MeshGroupData.h"
#include "Index1D.h"
#include "ParticleData.h"

#include <pybind11/numpy.h>

#ifdef ENABLE_HIP
#include "BondedGroupData.cuh"
#include "CachedAllocator.h"
#endif

using namespace std;
namespace py = pybind11;

//! Names of bonded groups
char name_bond_data[] = "bond";
char name_meshbond_data[] = "meshbond";
char name_angle_data[] = "angle";
char name_triangle_data[] = "triangle";
char name_meshtriangle_data[] = "meshtriangle";
char name_dihedral_data[] = "dihedral";
char name_improper_data[] = "improper";
char name_constraint_data[] = "constraint";
char name_pair_data[] = "pair";

/*
 * Implementation of BondedGroupData methods
 */

/*! \param exec_conf Execution configuration
    \param pdata The particle data to associate with
    \param n_group_types Number of bonded group types to initialize
 */
template<unsigned int group_size, typename Group, const char* name, bool bond>
MeshGroupData<group_size, Group, name, bond>::MeshGroupData(
    std::shared_ptr<ParticleData> pdata,
    unsigned int n_group_types)
    : BondedGroupData<group_size,Group,name,true>(pdata,n_group_types)
    {
    }

/*! \param exec_conf Execution configuration
    \param pdata The particle data to associate with
    \param snapshot Snapshot to initialize from
 */
template<unsigned int group_size, typename Group, const char* name, bool bond>
MeshGroupData<group_size, Group, name, bond>::MeshGroupData(
    std::shared_ptr<ParticleData> pdata,
    const TriangleData::Snapshot& snapshot)
    : m_exec_conf(pdata->getExecConf()), m_pdata(pdata), m_n_groups(0), m_n_ghost(0), m_nglobal(0),
      m_groups_dirty(true)
    {
    m_exec_conf->msg->notice(5) << "Constructing MeshGroupData (" << name << ") " << endl;

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
template<unsigned int group_size, typename Group, const char* name, bool bond>
MeshGroupData<group_size, Group, name, bond>::~MeshGroupData()
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

//! Initialize from a snapshot
template<unsigned int group_size, typename Group, const char* name, bool bond>
void MeshGroupData<group_size, Group, name, bond>::initializeFromSnapshot(
    const TriangleData::Snapshot& snapshot)
    {
    // check that all fields in the snapshot have correct length
    if (m_exec_conf->getRank() == 0 && !snapshot.validate())
        {
        m_exec_conf->msg->error() << "init.*: invalid " << name << " data snapshot." << std::endl
                                  << std::endl;
        throw std::runtime_error(std::string("Error initializing ") + name + std::string(" data."));
        }

    // re-initialize data structures
    initialize();

    std::vector<members_t> all_groups;

    if(bond)
	{
        for (unsigned group_idx = 0; group_idx < snapshot.groups.size(); group_idx++)
            {
            member_t triag;
	    std::vector<member_t> bonds(3);
            triag.tag[0] = snapshot.groups[group_idx].tag[0];
            triag.tag[1] = snapshot.groups[group_idx].tag[1];
            triag.tag[2] = snapshot.groups[group_idx].tag[2];

	    bonds[0].tag[0] = triag.tag[0];
	    bonds[0].tag[1] = triag.tag[1];
	    bonds[0].tag[2] = group_idx;
	    bonds[1].tag[0] = triag.tag[1];
	    bonds[1].tag[1] = triag.tag[2];
	    bonds[1].tag[2] = group_idx;
	    bonds[2].tag[0] = triag.tag[2];
	    bonds[2].tag[1] = triag.tag[0];
	    bonds[2].tag[2] = group_idx;

	    for (unsigned int i = 0; i < all_groups.size(); ++i)
		{
                for (unsigned int j = 0; j < bonds.size(); ++j)
		   {
                   if( (bonds[j].tag[0] == all_helper[i].tag[0] && bonds[j].tag[1] == all_helper[i].tag[1]) || (bonds[j].tag[0] == all_helper[i].tag[1] && bonds[j].tag[0] == all_helper[i].tag[1]) )
		       {
		       all_groups[i].tag[3]=i;
		       bonds.erase(bonds.begin()+j);
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
        std::vector<members_t> all_helper;
        all_groups.resize(snapshot.groups.size());
        for (unsigned group_idx = 0; group_idx < snapshot.groups.size(); group_idx++)
            {
            member_t triag;
	    std::vector<member_t> bonds(3);
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
                   if( (bonds[j].tag[0] == all_helper[i].tag[0] && bonds[j].tag[1] == all_helper[i].tag[1]) || (bonds[j].tag[0] == all_helper[i].tag[1] && bonds[j].tag[0] == all_helper[i].tag[1]) )
		       {
		       bond_id.push_back(i);
		       bonds.erase(bonds.begin()+j);
		       break;
		       }
	           }
	        }
            unsigned int j = 0
            for (j = 0; j < bond_id.size(); ++j)
		{
	        triag.tag[3+j]=bond_id[j];
	        }

            for (unsigned int i = 0; i < bonds.size(); ++i)
		{
	        triag.tag[3+j]=all_helper.size();
		all_helper.push_back(bonds[i]);
		j++;
	        }
	    }
            all_groups[group_idx] = triag;
	}

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        // broadcast to all processors (temporarily)
        std::vector<typeval_t> all_typeval;

        if (m_exec_conf->getRank() == 0)
            {
            all_typeval.resize(snapshot.type_id.size());
            // fill in types
            for (unsigned int i = 0; i < snapshot.type_id.size(); ++i)
                {
                typeval_t t;
                t.type = snapshot.type_id[i];
                all_typeval[i] = t;
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

        // create bonded groups with types
        for (unsigned group_idx = 0; group_idx < snapshot.groups.size(); group_idx++)
            {
            typeval_t t;
            t.type = snapshot.type_id[group_idx];
            addBondedGroup(Group(t, all_groups[group_idx]));
            }
        }
    }

template<unsigned int group_size, typename Group, const char* name, bool bonding>
unsigned int MeshGroupData<group_size, Group, name, bond>::addBondedGroup(Group g)
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

    for (unsigned int i = 0; i < group_size/2; ++i)
        for (unsigned int j = 0; j < group_size/2; ++j)
            if (i != j && member_tags.tag[i] == member_tags.tag[j])
                {
                std::ostringstream oss;
                oss << "The same particle can only occur once in a " << name << ": ";
                for (unsigned int k = 0; k < group_size; ++k)
                    oss << member_tags.tag[k] << ((k != group_size - 1) ? "," : "");
                oss << std::endl;
                throw runtime_error(oss.str());
                }

    for (unsigned int i = group_size/2; i < group_size; ++i)
        for (unsigned int j = group_size/2; j < group_size; ++j)
            if (i != j && member_tags.tag[i] == member_tags.tag[j])
                {
                std::ostringstream oss;
                oss << "The same simplex can only occur once in a " << name << ": ";
                for (unsigned int k = 0; k < group_size; ++k)
                    oss << member_tags.tag[k] << ((k != group_size - 1) ? "," : "");
                oss << std::endl;
                throw runtime_error(oss.str());
                }

    unsigned int type = typeval.type;
    if (type >= m_type_mapping.size() && has_type_mapping)
        {
        std::ostringstream s;
        s << "Invalid " << name << " typeid " << type << ". The number of types is "
          << m_type_mapping.size() << ".";
        throw std::runtime_error(s.str());
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


/*! \param snapshot Snapshot that will contain the group data
 * \returns a map to lookup snapshot index by tag
 *
 *  Data in the snapshot is in tag order, where non-existent tags are skipped
 */
template<unsigned int group_size, typename Group, const char* name, bool bond>
std::map<unsigned int, unsigned int>
MeshGroupData<group_size, Group, name, bond>::takeSnapshot(TriangleData::Snapshot& snapshot) const
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
                    m_exec_conf->msg->error() << endl
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
                snapshot.groups[snap_id].tag[0] = members_proc[rank][idx].tag[0];
                snapshot.groups[snap_id].tag[1] = members_proc[rank][idx].tag[1];
                snapshot.groups[snap_id].tag[2] = members_proc[rank][idx].tag[2];
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
                m_exec_conf->msg->error() << endl
                                          << "Could not find " << name << " " << group_tag
                                          << ". Possible internal error?" << endl
                                          << endl;
                throw std::runtime_error("Error gathering " + std::string(name) + "s");
                }

            // store tag in index
            index.insert(std::make_pair(group_tag, snap_id));

            unsigned int group_idx = rtag_it->second;
            snapshot.groups[snap_id].tag[0] = m_groups[group_idx].tag[0];
            snapshot.type_id[snap_id] = ((typeval_t)m_group_typeval[group_idx]).type;
            snap_id++;
            }
        }

    snapshot.type_mapping = m_type_mapping;

    return index;
    }


template<class T, typename Group>
void export_BondedGroupData(py::module& m,
                            std::string name,
                            std::string snapshot_name,
                            bool export_struct)
    {
    // export group structure
    if (export_struct)
        Group::export_to_python(m);

    py::class_<T, std::shared_ptr<T>>(m, name.c_str())
        .def(py::init<std::shared_ptr<ParticleData>, unsigned int>())
        .def(py::init<std::shared_ptr<ParticleData>, const typename TriangleData::Snapshot&>())
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
        .def("setProfiler", &T::setProfiler)
        .def("getTypes", &T::getTypesPy);

    }


template class PYBIND11_EXPORT MeshGroupData<4, MeshBond, name_meshbond_data, true>;
template void export_MeshGroupData<MeshBondData, MeshBond>(py::module& m,
                                                     std::string name,
                                                     std::string snapshot_name,
                                                     bool export_struct);

template class PYBIND11_EXPORT BondedGroupData<6, MeshTriangle, name_meshtriangle_data, false>;
template void export_BondedGroupData<MeshTriangleData, MeshTriangle>(py::module& m,
                                                       std::string name,
                                                       std::string snapshot_name,
                                                       bool export_struct);
