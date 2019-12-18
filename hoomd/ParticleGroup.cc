// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander
#include "ParticleGroup.h"

#ifdef ENABLE_HIP
#include "ParticleGroup.cuh"
#include "CachedAllocator.h"

#include <hip/hip_runtime.h>
#endif

#include <algorithm>
#include <iostream>
using namespace std;
namespace py = pybind11;

/*! \param sysdef System the particles are to be selected from
*/
ParticleFilter::ParticleFilter()
    {
    }

/*! \param tag Tag of the particle to check
    \returns true if the particle is selected
    \returns false if it is not
*/
std::vector<unsigned int> ParticleFilter::getSelectedTags(std::shared_ptr<SystemDefinition> sysdef) const
    {
    // base class doesn't do anything useful
    return std::vector<unsigned int>();
    }

ParticleFilterAll::ParticleFilterAll()
    : ParticleFilter()
    { }

/*! \param tag All of the particle to check
    \returns true if particle is local
*/
std::vector<unsigned int> ParticleFilterAll::getSelectedTags(std::shared_ptr<SystemDefinition> sysdef) const
    {
    std::vector<unsigned int> member_tags;
    auto pdata = sysdef->getParticleData();

    // loop through local particles and select those that match selection criterion
    ArrayHandle<unsigned int> h_tag(pdata->getTags(), access_location::host, access_mode::read);
    for (unsigned int idx = 0; idx < pdata->getN(); ++idx)
        {
        unsigned int tag = h_tag.data[idx];
        member_tags.push_back(tag);
        }

    return member_tags;
    }

//////////////////////////////////////////////////////////////////////////////
// ParticleFilterTags

/** Args:
        tags: numpy array of tags to select
*/
ParticleFilterTags::ParticleFilterTags(std::vector<unsigned int> tags)
    : ParticleFilter(), m_tags(tags)
    {
    }

/** Args:
        tags: numpy array of tags to select
*/
ParticleFilterTags::ParticleFilterTags(pybind11::array_t<unsigned int, pybind11::array::c_style> tags)
    : ParticleFilter()
    {
	unsigned int* tags_ptr = (unsigned int*)m_tags.data();
    m_tags.assign(tags_ptr, tags_ptr+m_tags.size());
    }

/*! \param tag Tag of the particle to check
    \returns true if \a m_tag_min <= \a tag <= \a m_tag_max
*/
std::vector<unsigned int> ParticleFilterTags::getSelectedTags(std::shared_ptr<SystemDefinition> sysdef) const
    {
	return m_tags;
    }

/*! \param typ_min Minimum type id to select (inclusive)
    \param typ_max Maximum type id to select (inclusive)
*/
ParticleFilterType::ParticleFilterType(unsigned int typ_min,
                                           unsigned int typ_max)
    : ParticleFilter(), m_typ_min(typ_min), m_typ_max(typ_max)
    {
    }

/*! \param tag Tag of the particle to check
    \returns true if the type of particle \a tag is in the inclusive range [ \a m_typ_min, \a m_typ_max ]
*/
std::vector<unsigned int> ParticleFilterType::getSelectedTags(std::shared_ptr<SystemDefinition> sysdef) const
    {
    std::vector<unsigned int> member_tags;
    auto pdata = sysdef->getParticleData();

    // loop through local particles and select those that match selection criterion
    ArrayHandle<unsigned int> h_tag(pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_postype(pdata->getPositions(), access_location::host, access_mode::read);
    for (unsigned int idx = 0; idx < pdata->getN(); ++idx)
        {
        unsigned int tag = h_tag.data[idx];
        unsigned int typ = __scalar_as_int(h_postype.data[idx].w);
        if (m_typ_min <= typ && typ <= m_typ_max)
            member_tags.push_back(tag);
        }
    return member_tags;
    }

/*! \param sysdef System the particles are to be selected from
    \param rigid true selects particles that are in bodies, false selects particles that are not part of a body
*/
ParticleFilterBody::ParticleFilterBody(bool body)
    : ParticleFilter(), m_body(body)
    {
    }

/*! \param tag Tag of the particle to check
    \returns true if the type of particle \a tag meets the body criteria selected
*/
std::vector<unsigned int> ParticleFilterBody::getSelectedTags(std::shared_ptr<SystemDefinition> sysdef) const
    {
    std::vector<unsigned int> member_tags;
    auto pdata = sysdef->getParticleData();

    // loop through local particles and select those that match selection criterion
    ArrayHandle<unsigned int> h_tag(pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_body(pdata->getBodies(), access_location::host, access_mode::read);
    for (unsigned int idx = 0; idx < pdata->getN(); ++idx)
        {
        unsigned int tag = h_tag.data[idx];

        // get position of particle
        unsigned int body = h_body.data[idx];

        // see if it matches the criteria
        bool result = false;
        if (m_body && body != NO_BODY)
            result = true;
        if (!m_body && body == NO_BODY)
            result = true;

        if (result)
            member_tags.push_back(tag);
        }
    return member_tags;
    }


//////////////////////////////////////////////////////////////////////////////
// ParticleFilterRigid

/*! \param rigid true selects particles that are in rigid bodies, false selects particles that are not part of a rigid body
*/
ParticleFilterRigid::ParticleFilterRigid(bool rigid)
    : ParticleFilter(), m_rigid(rigid)
    {
    }

/*! \param tag Tag of the particle to check
    \returns true if the type of particle \a tag meets the rigid criteria selected
*/
std::vector<unsigned int> ParticleFilterRigid::getSelectedTags(std::shared_ptr<SystemDefinition> sysdef) const
    {
    std::vector<unsigned int> member_tags;
    auto pdata = sysdef->getParticleData();

    // loop through local particles and select those that match selection criterion
    ArrayHandle<unsigned int> h_tag(pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_body(pdata->getBodies(), access_location::host, access_mode::read);
    for (unsigned int idx = 0; idx < pdata->getN(); ++idx)
        {
        unsigned int tag = h_tag.data[idx];

        // get position of particle
        unsigned int body = h_body.data[idx];

        // see if it matches the criteria
        bool result = false;
        if (m_rigid && body < MIN_FLOPPY)
            result = true;
        if (!m_rigid && body >= MIN_FLOPPY)
            result = true;

        if (result)
            member_tags.push_back(tag);
        }
    return member_tags;
    }


//////////////////////////////////////////////////////////////////////////////
// ParticleFilterFloppy

/*! \param floppy true selects particles that are in floppy bodies, false selects particles that are not part of a floppy (non-rigid body)
*/
ParticleFilterFloppy::ParticleFilterFloppy(bool floppy)
    : ParticleFilter(), m_floppy(floppy)
    {
    }

/*! \param tag Tag of the particle to check
    \returns true if the type of particle \a tag meets the rigid criteria selected
*/
std::vector<unsigned int> ParticleFilterFloppy::getSelectedTags(std::shared_ptr<SystemDefinition> sysdef) const
    {
    std::vector<unsigned int> member_tags;
    auto pdata = sysdef->getParticleData();

    // loop through local particles and select those that match selection criterion
    ArrayHandle<unsigned int> h_tag(pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_body(pdata->getBodies(), access_location::host, access_mode::read);
    for (unsigned int idx = 0; idx < pdata->getN(); ++idx)
        {
        unsigned int tag = h_tag.data[idx];

        // get position of particle
        unsigned int body = h_body.data[idx];

        // see if it matches the criteria
        bool result = false;
        if (m_floppy && body >= MIN_FLOPPY && body != NO_BODY)
            result = true;
        if (!m_floppy && (body < MIN_FLOPPY || body == NO_BODY))
            result = true;

        if (result)
            member_tags.push_back(tag);
        }
    return member_tags;
    }

ParticleFilterRigidCenter::ParticleFilterRigidCenter()
    :ParticleFilter()
    {
    }

/*! \returns true if the type of particle \a tag is a center particle of a rigid body
*/
std::vector<unsigned int> ParticleFilterRigidCenter::getSelectedTags(std::shared_ptr<SystemDefinition> sysdef) const
    {
    std::vector<unsigned int> member_tags;
    auto pdata = sysdef->getParticleData();

    // loop through local particles and select those that match selection criterion
    ArrayHandle<unsigned int> h_tag(pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_body(pdata->getBodies(), access_location::host, access_mode::read);
    for (unsigned int idx = 0; idx < pdata->getN(); ++idx)
        {
        unsigned int tag = h_tag.data[idx];

        // get position of particle
        unsigned int body = h_body.data[idx];

        if (body==tag)
            member_tags.push_back(tag);
        }
    return member_tags;
    }

ParticleFilterCuboid::ParticleFilterCuboid(Scalar3 min, Scalar3 max)
    : ParticleFilter(), m_min(min), m_max(max)
    {
    }

/*! \param tag Tag of the particle to check
    \returns true if the type of particle \a tag is in the cuboid

    Evaluation is performed by \a m_min.x <= x < \a m_max.x so that multiple cuboids stacked next to each other
    do not have overlapping sets of particles.
*/
std::vector<unsigned int> ParticleFilterCuboid::getSelectedTags(std::shared_ptr<SystemDefinition> sysdef) const
    {
    std::vector<unsigned int> member_tags;
    auto pdata = sysdef->getParticleData();

    // loop through local particles and select those that match selection criterion
    ArrayHandle<unsigned int> h_tag(pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_postype(pdata->getPositions(), access_location::host, access_mode::read);
    for (unsigned int idx = 0; idx < pdata->getN(); ++idx)
        {
        unsigned int tag = h_tag.data[idx];
        // get position of particle
        Scalar4 postype = h_postype.data[idx];

        // see if it matches the criteria
        bool result = (m_min.x <= postype.x && postype.x < m_max.x &&
                       m_min.y <= postype.y && postype.y < m_max.y &&
                       m_min.z <= postype.z && postype.z < m_max.z);
        if (result)
            member_tags.push_back(tag);
        }

    return member_tags;
    }

//////////////////////////////////////////////////////////////////////////////
// ParticleGroup

/*! \param sysdef System definition to build the group from
    \param selector ParticleFilter used to choose the group members
    \param update_tags If true, update tags whenever global particle number changes

    Particles where criteria falls within the range [min,max] (inclusive) are added to the group.
*/
ParticleGroup::ParticleGroup(std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<ParticleFilter> selector,
    bool update_tags)
    : m_sysdef(sysdef),
      m_pdata(sysdef->getParticleData()),
      m_exec_conf(m_pdata->getExecConf()),
      m_num_local_members(0),
      m_particles_sorted(true),
      m_reallocated(false),
      m_global_ptl_num_change(false),
      m_selector(selector),
      m_update_tags(update_tags),
      m_warning_printed(false)
    {
    #ifdef ENABLE_HIP
    if (m_pdata->getExecConf()->isCUDAEnabled())
        m_gpu_partition = GPUPartition(m_exec_conf->getGPUIds());
    #endif

    // update member tag arrays
    updateMemberTags(true);

    // connect to the particle sort signal
    m_pdata->getParticleSortSignal().connect<ParticleGroup, &ParticleGroup::slotParticleSort>(this);

    // connect reallocate() method to maximum particle number change signal
    m_pdata->getMaxParticleNumberChangeSignal().connect<ParticleGroup, &ParticleGroup::slotReallocate>(this);

    // connect updateMemberTags() method to maximum particle number change signal
    m_pdata->getGlobalParticleNumberChangeSignal().connect<ParticleGroup, &ParticleGroup::slotGlobalParticleNumChange>(this);

    // update GPU memory hints
    updateGPUAdvice();
    }

/*! \param sysdef System definition to build the group from
    \param member_tags List of particle tags that belong to the group

    All particles specified in \a member_tags will be added to the group.
*/
ParticleGroup::ParticleGroup(std::shared_ptr<SystemDefinition> sysdef, const std::vector<unsigned int>& member_tags)
    : m_sysdef(sysdef),
      m_pdata(sysdef->getParticleData()),
      m_exec_conf(m_pdata->getExecConf()),
      m_num_local_members(0),
      m_particles_sorted(true),
      m_reallocated(false),
      m_global_ptl_num_change(false),
      m_update_tags(false),
      m_warning_printed(false)
    {
    // check input
    unsigned int max_tag = m_pdata->getMaximumTag();
    for (std::vector<unsigned int>::const_iterator it = member_tags.begin(); it != member_tags.end(); ++it)
        {
        if (*it > max_tag)
            {
            m_exec_conf->msg->error() << "group.*: Member " << *it << " does not exist in particle data." << std::endl;
            throw std::runtime_error("Error creating ParticleGroup\n");
            }
        }

    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        // do a simple sanity check
        unsigned int nptl = member_tags.size();
        bcast(nptl, 0, m_exec_conf->getMPICommunicator());

        if (nptl != member_tags.size())
            {
            m_exec_conf->msg->error() << "group.*: Member tag list is inconsistent among MPI ranks." << std::endl;
            throw std::runtime_error("Error creating ParticleGroup\n");
            }
        }
    #endif

    // let's make absolutely sure that the tag order given from outside is sorted
    std::vector<unsigned int> sorted_member_tags =  member_tags;
    sort(sorted_member_tags.begin(), sorted_member_tags.end());

    // store member tags
    GlobalArray<unsigned int> member_tags_array(member_tags.size(), m_exec_conf);
    m_member_tags.swap(member_tags_array);
    TAG_ALLOCATION(m_member_tags);

        {
        ArrayHandle<unsigned int> h_member_tags(m_member_tags, access_location::host, access_mode::overwrite);
        std::copy(sorted_member_tags.begin(), sorted_member_tags.end(), h_member_tags.data);
        }

    // one byte per particle to indicate membership in the group, initialize with current number of local particles
    GlobalArray<unsigned int> is_member(m_pdata->getMaxN(), m_pdata->getExecConf());
    m_is_member.swap(is_member);
    TAG_ALLOCATION(m_is_member);

    GlobalArray<unsigned int> is_member_tag(m_pdata->getRTags().size(), m_pdata->getExecConf());
    m_is_member_tag.swap(is_member_tag);
    TAG_ALLOCATION(m_is_member_tag);

    // build the reverse lookup table for tags
    buildTagHash();

    GlobalArray<unsigned int> member_idx(member_tags.size(), m_pdata->getExecConf());
    m_member_idx.swap(member_idx);
    TAG_ALLOCATION(m_member_idx);

    #ifdef ENABLE_HIP
    if (m_pdata->getExecConf()->isCUDAEnabled())
        m_gpu_partition = GPUPartition(m_exec_conf->getGPUIds());
    #endif

    // now that the tag list is completely set up and all memory is allocated, rebuild the index list
    rebuildIndexList();

    // connect to the particle sort signal
    m_pdata->getParticleSortSignal().connect<ParticleGroup, &ParticleGroup::slotParticleSort>(this);

    // connect reallocate() method to maximum particle number change signal
    m_pdata->getMaxParticleNumberChangeSignal().connect<ParticleGroup, &ParticleGroup::slotReallocate>(this);

    // connect updateMemberTags() method to maximum particle number change signal
    m_pdata->getGlobalParticleNumberChangeSignal().connect<ParticleGroup, &ParticleGroup::slotGlobalParticleNumChange>(this);

    // update GPU memory hints
    updateGPUAdvice();
    }

ParticleGroup::~ParticleGroup()
    {
    // disconnect the sort connection, but only if there was a particle data to connect it to in the first place
    if (m_pdata)
        {
        m_pdata->getParticleSortSignal().disconnect<ParticleGroup, &ParticleGroup::slotParticleSort>(this);
        m_pdata->getMaxParticleNumberChangeSignal().disconnect<ParticleGroup, &ParticleGroup::slotReallocate>(this);
        m_pdata->getGlobalParticleNumberChangeSignal().disconnect<ParticleGroup, &ParticleGroup::slotGlobalParticleNumChange>(this);
        }
    }

/*! \param force_update If true, always update member tags
 */
void ParticleGroup::updateMemberTags(bool force_update) const
    {
    if (m_selector && !(m_update_tags || force_update) && ! m_warning_printed)
        {
        m_pdata->getExecConf()->msg->warning()
            << "Particle number change but group is static. Create group with update=True if it should be updated."
            << std::endl << "This warning is printed only once." << std::endl;
        m_warning_printed = true;
        }

    if (m_selector && (m_update_tags || force_update))
        {
        // notice message
        m_pdata->getExecConf()->msg->notice(7) << "ParticleGroup: rebuilding tags" << std::endl;

        // assign all of the particles that belong to the group
        // for each particle in the (global) data
        vector<unsigned int> member_tags = m_selector->getSelectedTags(m_sysdef);

        #ifdef ENABLE_MPI
        if (m_pdata->getDomainDecomposition())
            {
            // combine lists from all processors
            std::vector< std::vector<unsigned int> > member_tags_proc(m_exec_conf->getNRanks());
            all_gather_v(member_tags, member_tags_proc, m_exec_conf->getMPICommunicator());

            assert(member_tags_proc.size() == m_exec_conf->getNRanks());

            // combine all tags into an ordered set
            unsigned int n_ranks = m_exec_conf->getNRanks();
            std::set<unsigned int> tag_set;
            for (unsigned int irank = 0; irank < n_ranks; ++irank)
                {
                tag_set.insert(member_tags_proc[irank].begin(), member_tags_proc[irank].end());
                }

            // construct list
            member_tags.clear();
            member_tags.insert(member_tags.begin(), tag_set.begin(), tag_set.end());
            }
        #endif

        // store member tags in GlobalArray
        GlobalArray<unsigned int> member_tags_array(member_tags.size(), m_pdata->getExecConf());
        m_member_tags.swap(member_tags_array);
        TAG_ALLOCATION(m_member_tags);

        // sort member tags
        std::sort(member_tags.begin(), member_tags.end());

            {
            ArrayHandle<unsigned int> h_member_tags(m_member_tags, access_location::host, access_mode::overwrite);
            std::copy(member_tags.begin(), member_tags.end(), h_member_tags.data);
            }

        GlobalArray<unsigned int> member_idx(member_tags.size(), m_pdata->getExecConf());
        m_member_idx.swap(member_idx);
        TAG_ALLOCATION(m_member_idx);
        }

    // one byte per particle to indicate membership in the group, initialize with current number of local particles
    GlobalArray<unsigned int> is_member(m_pdata->getMaxN(), m_pdata->getExecConf());
    m_is_member.swap(is_member);
    TAG_ALLOCATION(m_is_member);

    GlobalArray<unsigned int> is_member_tag(m_pdata->getRTags().size(), m_pdata->getExecConf());
    m_is_member_tag.swap(is_member_tag);
    TAG_ALLOCATION(m_is_member_tag);

    // build the reverse lookup table for tags
    buildTagHash();

    // now that the tag list is completely set up and all memory is allocated, rebuild the index list
    rebuildIndexList();
    }

void ParticleGroup::reallocate() const
    {
    m_is_member.resize(m_pdata->getMaxN());

    if (m_is_member_tag.getNumElements() != m_pdata->getRTags().size())
        {
        // reallocate if necessary
        GlobalArray<unsigned int> is_member_tag(m_pdata->getRTags().size(), m_exec_conf);
        m_is_member_tag.swap(is_member_tag);
        TAG_ALLOCATION(m_is_member_tag);

        buildTagHash();
        }
    }

/*! \returns Total mass of all particles in the group
    \note This method acquires the ParticleData internally
*/
Scalar ParticleGroup::getTotalMass() const
    {
    // grab the particle data
    ArrayHandle< Scalar4 > h_vel(m_pdata->getVelocities(), access_location::host, access_mode::read);

    // loop  through all indices in the group and total the mass
    Scalar total_mass = 0.0;
    for (unsigned int i = 0; i < getNumMembers(); i++)
        {
        unsigned int idx = getMemberIndex(i);
        total_mass += h_vel.data[idx].w;
        }
    return total_mass;
    }

/*! \returns The center of mass of the group, in unwrapped coordinates
    \note This method acquires the ParticleData internally
*/
Scalar3 ParticleGroup::getCenterOfMass() const
    {
    // grab the particle data
    ArrayHandle< Scalar4 > h_vel(m_pdata->getVelocities(), access_location::host, access_mode::read);
    ArrayHandle< Scalar4 > h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle< int3 > h_image(m_pdata->getImages(), access_location::host, access_mode::read);

    // grab the box dimensions
    BoxDim box = m_pdata->getBox();

    // loop  through all indices in the group and compute the weighted average of the positions
    Scalar total_mass = 0.0;
    Scalar3 center_of_mass = make_scalar3(Scalar(0.0), Scalar(0.0), Scalar(0.0));
    for (unsigned int i = 0; i < getNumMembers(); i++)
        {
        unsigned int idx = getMemberIndex(i);
        Scalar mass = h_vel.data[idx].w;
        total_mass += mass;
        Scalar3 pos = make_scalar3(h_pos.data[idx].x, h_pos.data[idx].y, h_pos.data[idx].z);
        Scalar3 unwrapped = box.shift(pos, h_image.data[idx]);
        center_of_mass.x += mass * unwrapped.x;
        center_of_mass.y += mass * unwrapped.y;
        center_of_mass.z += mass * unwrapped.z;
        }
    center_of_mass.x /= total_mass;
    center_of_mass.y /= total_mass;
    center_of_mass.z /= total_mass;

    return center_of_mass;
    }

/*! \param a First particle group
    \param b Second particle group

    \returns A shared pointer to a newly created particle group that contains all the elements present in \a a and
    \a b
*/
std::shared_ptr<ParticleGroup> ParticleGroup::groupUnion(std::shared_ptr<ParticleGroup> a,
                                                           std::shared_ptr<ParticleGroup> b)
    {
    // vector to store the new list of tags
    vector<unsigned int> member_tags;

    if (a != b)
        {
        unsigned int n_a = a->getNumMembersGlobal();
        unsigned int n_b = b->getNumMembersGlobal();

        // make the union
        ArrayHandle<unsigned int> h_members_a(a->m_member_tags, access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_members_b(b->m_member_tags, access_location::host, access_mode::read);

        insert_iterator< vector<unsigned int> > ii(member_tags, member_tags.begin());
        set_union(h_members_a.data,
                  h_members_a.data + n_a,
                  h_members_b.data,
                  h_members_b.data + n_b,
                  ii);
        }
    else
        {
        unsigned int n_a = a->getNumMembersGlobal();

        // If the two arguments are the same, just return a copy of the whole group (we cannot
        // acquire the member_tags array twice)
        ArrayHandle<unsigned int> h_members_a(a->m_member_tags, access_location::host, access_mode::read);

        insert_iterator< vector<unsigned int> > ii(member_tags, member_tags.begin());
        std::copy(h_members_a.data,
                  h_members_a.data + n_a,
                  ii);
        }



    // create the new particle group
    std::shared_ptr<ParticleGroup> new_group(new ParticleGroup(a->m_sysdef, member_tags));

    // return the newly created group
    return new_group;
    }

/*! \param a First particle group
    \param b Second particle group

    \returns A shared pointer to a newly created particle group that contains only the elements present in both \a a and
    \a b
*/
std::shared_ptr<ParticleGroup> ParticleGroup::groupIntersection(std::shared_ptr<ParticleGroup> a,
                                                                  std::shared_ptr<ParticleGroup> b)
    {
    // vector to store the new list of tags
    vector<unsigned int> member_tags;

    if (a != b)
        {
        unsigned int n_a = a->getNumMembersGlobal();
        unsigned int n_b = b->getNumMembersGlobal();

        // make the intersection
        ArrayHandle<unsigned int> h_members_a(a->m_member_tags, access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_members_b(b->m_member_tags, access_location::host, access_mode::read);

        insert_iterator< vector<unsigned int> > ii(member_tags, member_tags.begin());
        set_intersection(h_members_a.data,
                         h_members_a.data + n_a,
                         h_members_b.data,
                         h_members_b.data + n_b,
                         ii);
        }
    else
        {
        unsigned int n_a = a->getNumMembersGlobal();
        // If the two arguments are the same, just return a copy of the whole group (we cannot
        // acquire the member_tags array twice)
        ArrayHandle<unsigned int> h_members_a(a->m_member_tags, access_location::host, access_mode::read);

        insert_iterator< vector<unsigned int> > ii(member_tags, member_tags.begin());
        std::copy(h_members_a.data,
                  h_members_a.data + n_a,
                  ii);
        }

    // create the new particle group
    std::shared_ptr<ParticleGroup> new_group(new ParticleGroup(a->m_sysdef, member_tags));

    // return the newly created group
    return new_group;
    }

/*! \param a First particle group
    \param b Second particle group

    \returns A shared pointer to a newly created particle group that contains only the elements present in \a a, and
    not any present in \a b
*/
std::shared_ptr<ParticleGroup> ParticleGroup::groupDifference(std::shared_ptr<ParticleGroup> a,
                                                                std::shared_ptr<ParticleGroup> b)
    {
    // vector to store the new list of tags
    vector<unsigned int> member_tags;

    if (a != b)
        {
        unsigned int n_a = a->getNumMembersGlobal();
        unsigned int n_b = b->getNumMembersGlobal();
        // make the difference
        ArrayHandle<unsigned int> h_members_a(a->m_member_tags, access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_members_b(b->m_member_tags, access_location::host, access_mode::read);

        insert_iterator< vector<unsigned int> > ii(member_tags, member_tags.begin());
        set_difference(h_members_a.data,
                  h_members_a.data + n_a,
                  h_members_b.data,
                  h_members_b.data + n_b,
                  ii);
        }
    else
        {
        // If the two arguments are the same, just return an empty group (we cannot
        // acquire the member_tags array twice)
        }


    // create the new particle group
    std::shared_ptr<ParticleGroup> new_group(new ParticleGroup(a->m_sysdef, member_tags));

    // return the newly created group
    return new_group;
    }

/*! Builds the by-tag-lookup table for group membership
 */
void ParticleGroup::buildTagHash() const
    {
    ArrayHandle<unsigned int> h_is_member_tag(m_is_member_tag, access_location::host, access_mode::overwrite);
    ArrayHandle<unsigned int> h_member_tags(m_member_tags, access_location::host, access_mode::read);

    // reset member ship flags
    memset(h_is_member_tag.data, 0, sizeof(unsigned int)*(m_pdata->getRTags().size()));

    unsigned int num_members = m_member_tags.getNumElements();
    for (unsigned int member = 0; member < num_members; member++)
        {
        h_is_member_tag.data[h_member_tags.data[member]] = 1;
        }
    }

/*! \pre m_member_tags has been filled out, listing all particle tags in the group
    \pre memory has been allocated for m_is_member and m_member_idx
    \post m_is_member is updated so that it reflects the current indices of the particles in the group
    \post m_member_idx is updated listing all particle indices belonging to the group, in index order
*/
void ParticleGroup::rebuildIndexList() const
    {
    // notice message
    m_pdata->getExecConf()->msg->notice(10) << "ParticleGroup: rebuilding index" << std::endl;

    #ifdef ENABLE_HIP
    if (m_pdata->getExecConf()->isCUDAEnabled() )
        {
        rebuildIndexListGPU();
        }
    else
    #endif
        {

        // rebuild the membership flags for the  indices in the group and construct member list
        ArrayHandle<unsigned int> h_is_member(m_is_member, access_location::host, access_mode::readwrite);
        ArrayHandle<unsigned int> h_is_member_tag(m_is_member_tag, access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_member_idx(m_member_idx, access_location::host, access_mode::readwrite);
        unsigned int nparticles = m_pdata->getN();
        unsigned int cur_member = 0;
        for (unsigned int idx = 0; idx < nparticles; idx ++)
            {
            assert(h_tag.data[idx] <= m_pdata->getMaximumTag());
            unsigned int is_member = h_is_member_tag.data[h_tag.data[idx]];
            h_is_member.data[idx] =  is_member;
            if (is_member)
                {
                h_member_idx.data[cur_member] = idx;
                cur_member++;
                }
            }

        m_num_local_members = cur_member;
        assert(m_num_local_members <= m_member_tags.getNumElements());
        }

    // index has been rebuilt
    m_particles_sorted = false;

    #ifdef ENABLE_HIP
    if (m_pdata->getExecConf()->isCUDAEnabled())
        {
        // Update GPU load balancing info
        m_gpu_partition.setN(m_num_local_members);
        }
    #endif
    }

void ParticleGroup::updateGPUAdvice() const
    {
    #if defined(ENABLE_HIP) && defined(__HIP_PLATFORM_NVCC__)
    if (m_exec_conf->isCUDAEnabled() && m_exec_conf->allConcurrentManagedAccess())
        {
        // split preferred location of group indices across GPUs
        auto gpu_map = m_exec_conf->getGPUIds();
        for (unsigned int idev = 0; idev < m_exec_conf->getNumActiveGPUs(); ++idev)
            {
            auto range = m_gpu_partition.getRange(idev);
            unsigned int nelem =  range.second - range.first;

            if (!nelem)
                continue;

            cudaMemAdvise(m_member_idx.get()+range.first, sizeof(unsigned int)*nelem, cudaMemAdviseSetPreferredLocation, gpu_map[idev]);
            cudaMemAdvise(m_is_member.get()+range.first, sizeof(unsigned int)*nelem, cudaMemAdviseSetPreferredLocation, gpu_map[idev]);

            // migrate data to preferred location
            cudaMemPrefetchAsync(m_member_idx.get()+range.first, sizeof(unsigned int)*nelem, gpu_map[idev]);
            cudaMemPrefetchAsync(m_is_member.get()+range.first, sizeof(unsigned int)*nelem, gpu_map[idev]);
            }
        CHECK_CUDA_ERROR();
        }
    #endif
    }

#ifdef ENABLE_HIP
//! rebuild index list on the GPU
void ParticleGroup::rebuildIndexListGPU() const
    {
    ArrayHandle<unsigned int> d_is_member(m_is_member, access_location::device, access_mode::overwrite);
    ArrayHandle<unsigned int> d_is_member_tag(m_is_member_tag, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_member_idx(m_member_idx, access_location::device, access_mode::overwrite);
    ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);

    // get temporary buffer
    ScopedAllocation<unsigned int> d_tmp(m_pdata->getExecConf()->getCachedAllocator(), m_pdata->getN());

    // reset membership properties
    if (m_member_tags.getNumElements() > 0)
        {
        gpu_rebuild_index_list(m_pdata->getN(),
                           d_is_member_tag.data,
                           d_is_member.data,
                           d_tag.data);
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        gpu_compact_index_list(m_pdata->getN(),
                           d_is_member.data,
                           d_member_idx.data,
                           m_num_local_members,
                           d_tmp.data,
                           m_pdata->getExecConf()->getCachedAllocator());
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }
    else
        m_num_local_members = 0;
    }
#endif

void export_ParticleGroup(py::module& m)
    {
    py::class_<ParticleGroup, std::shared_ptr<ParticleGroup> >(m,"ParticleGroup")
            .def(py::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleFilter>, bool >())
            .def(py::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleFilter> >())
            .def(py::init<std::shared_ptr<SystemDefinition>, const std::vector<unsigned int>& >())
            .def(py::init<>())
            .def("getNumMembersGlobal", &ParticleGroup::getNumMembersGlobal)
            .def("getMemberTag", &ParticleGroup::getMemberTag)
            .def("getTotalMass", &ParticleGroup::getTotalMass)
            .def("getCenterOfMass", &ParticleGroup::getCenterOfMass)
            .def("groupUnion", &ParticleGroup::groupUnion)
            .def("groupIntersection", &ParticleGroup::groupIntersection)
            .def("groupDifference", &ParticleGroup::groupDifference)
            .def("updateMemberTags", &ParticleGroup::updateMemberTags)
            ;

    py::class_<ParticleFilter, std::shared_ptr<ParticleFilter> >(m,"ParticleFilter")
            .def(py::init< >())
            .def("_get_selected_tags", &ParticleFilter::getSelectedTags)
            ;

    py::class_<ParticleFilterAll, ParticleFilter, std::shared_ptr<ParticleFilterAll> >(m,"ParticleFilterAll")
            .def(py::init< >())
        ;

    py::class_<ParticleFilterTags, ParticleFilter, std::shared_ptr<ParticleFilterTags> >(m,"ParticleFilterTags")
			.def(py::init<pybind11::array_t<unsigned int, pybind11::array::c_style> >())
        ;

    py::class_<ParticleFilterType, ParticleFilter, std::shared_ptr<ParticleFilterType> >(m,"ParticleFilterType")
            .def(py::init< unsigned int, unsigned int >())
        ;

    py::class_<ParticleFilterRigid, ParticleFilter, std::shared_ptr<ParticleFilterRigid> >(m,"ParticleFilterRigid")
            .def(py::init< bool >())
        ;

    py::class_<ParticleFilterBody, ParticleFilter, std::shared_ptr<ParticleFilterBody> >(m,"ParticleFilterBody")

            .def(py::init< bool >())
        ;

    py::class_<ParticleFilterFloppy, ParticleFilter, std::shared_ptr<ParticleFilterFloppy> >(m,"ParticleFilterFloppy")
            .def(py::init< bool >())
        ;

    py::class_<ParticleFilterCuboid, ParticleFilter, std::shared_ptr<ParticleFilterCuboid> >(m,"ParticleFilterCuboid")
            .def(py::init< Scalar3, Scalar3 >())
        ;

    py::class_<ParticleFilterRigidCenter, ParticleFilter, std::shared_ptr<ParticleFilterRigidCenter> >(m,"ParticleFilterRigidCenter")
            .def(py::init< >())
        ;
    }
