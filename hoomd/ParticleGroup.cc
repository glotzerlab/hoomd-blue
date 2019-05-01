// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander
#include "ParticleGroup.h"

#ifdef ENABLE_CUDA
#include "ParticleGroup.cuh"
#include "CachedAllocator.h"

#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <iostream>
using namespace std;
namespace py = pybind11;

/*! \file ParticleGroup.cc
    \brief Defines the ParticleGroup and related classes
*/

//////////////////////////////////////////////////////////////////////////////
// ParticleSelector

/*! \param sysdef System the particles are to be selected from
*/
ParticleSelector::ParticleSelector(std::shared_ptr<SystemDefinition> sysdef)
    : m_sysdef(sysdef), m_pdata(sysdef->getParticleData())
    {
    assert(m_sysdef);
    assert(m_pdata);
    m_exec_conf = m_pdata->getExecConf();
    }

/*! \param tag Tag of the particle to check
    \returns true if the particle is selected
    \returns false if it is not
*/
std::vector<unsigned int> ParticleSelector::getSelectedTags() const
    {
    // base class doesn't do anything useful
    return std::vector<unsigned int>();
    }

//////////////////////////////////////////////////////////////////////////////
// ParticleSelectorAll

/*! \param sysdef System the particles are to be selected from
*/
ParticleSelectorAll::ParticleSelectorAll(std::shared_ptr<SystemDefinition> sysdef)
    : ParticleSelector(sysdef)
    { }

/*! \param tag All of the particle to check
    \returns true if particle is local
*/
std::vector<unsigned int> ParticleSelectorAll::getSelectedTags() const
    {
    std::vector<unsigned int> member_tags;

    // loop through local particles and select those that match selection criterion
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    for (unsigned int idx = 0; idx < m_pdata->getN(); ++idx)
        {
        unsigned int tag = h_tag.data[idx];
        member_tags.push_back(tag);
        }

    return member_tags;
    }

//////////////////////////////////////////////////////////////////////////////
// ParticleSelectorTag

/*! \param sysdef System the particles are to be selected from
    \param tag_min Minimum tag to select (inclusive)
    \param tag_max Maximum tag to select (inclusive)
*/
ParticleSelectorTag::ParticleSelectorTag(std::shared_ptr<SystemDefinition> sysdef,
                                         unsigned int tag_min,
                                         unsigned int tag_max)
    : ParticleSelector(sysdef), m_tag_min(tag_min), m_tag_max(tag_max)
    {
    // make a quick check on the sanity of the input data
    if (m_tag_max < m_tag_min)
        m_exec_conf->msg->warning() << "group: max < min specified when selecting particle tags" << endl;

    if (!m_pdata->getNGlobal() || m_tag_max > m_pdata->getMaximumTag())
        {
        m_exec_conf->msg->error() << "Cannot select particles with tags larger than the number of particles "
             << endl;
        throw runtime_error("Error selecting particles");
        }
    }

/*! \param tag Tag of the particle to check
    \returns true if \a m_tag_min <= \a tag <= \a m_tag_max
*/
std::vector<unsigned int> ParticleSelectorTag::getSelectedTags() const
    {
    std::vector<unsigned int> member_tags;

    // loop through local particles and select those that match selection criterion
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    for (unsigned int idx = 0; idx < m_pdata->getN(); ++idx)
        {
        unsigned int tag = h_tag.data[idx];
        if (tag >= m_tag_min && tag <= m_tag_max)
            member_tags.push_back(tag);
        }

    return member_tags;
    }

//////////////////////////////////////////////////////////////////////////////
// ParticleSelectorType

/*! \param sysdef System the particles are to be selected from
    \param typ_min Minimum type id to select (inclusive)
    \param typ_max Maximum type id to select (inclusive)
*/
ParticleSelectorType::ParticleSelectorType(std::shared_ptr<SystemDefinition> sysdef,
                                           unsigned int typ_min,
                                           unsigned int typ_max)
    : ParticleSelector(sysdef), m_typ_min(typ_min), m_typ_max(typ_max)
    {
    // make a quick check on the sanity of the input data
    if (m_typ_max < m_typ_min)
        m_exec_conf->msg->warning() << "group: max < min specified when selecting particle types" << endl;

    if (m_typ_max >= m_pdata->getNTypes())
        m_exec_conf->msg->warning() << "group: Requesting the selection of a non-existent particle type" << endl;
    }

/*! \param tag Tag of the particle to check
    \returns true if the type of particle \a tag is in the inclusive range [ \a m_typ_min, \a m_typ_max ]
*/
std::vector<unsigned int> ParticleSelectorType::getSelectedTags() const
    {
    std::vector<unsigned int> member_tags;

    // loop through local particles and select those that match selection criterion
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
    for (unsigned int idx = 0; idx < m_pdata->getN(); ++idx)
        {
        unsigned int tag = h_tag.data[idx];
        unsigned int typ = __scalar_as_int(h_postype.data[idx].w);
        if (m_typ_min <= typ && typ <= m_typ_max)
            member_tags.push_back(tag);
        }
    return member_tags;
    }


//////////////////////////////////////////////////////////////////////////////
// ParticleSelectorBody

/*! \param sysdef System the particles are to be selected from
    \param rigid true selects particles that are in bodies, false selects particles that are not part of a body
*/
ParticleSelectorBody::ParticleSelectorBody(std::shared_ptr<SystemDefinition> sysdef,
                                             bool body)
    : ParticleSelector(sysdef), m_body(body)
    {
    }

/*! \param tag Tag of the particle to check
    \returns true if the type of particle \a tag meets the body criteria selected
*/
std::vector<unsigned int> ParticleSelectorBody::getSelectedTags() const
    {
    std::vector<unsigned int> member_tags;

    // loop through local particles and select those that match selection criterion
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_body(m_pdata->getBodies(), access_location::host, access_mode::read);
    for (unsigned int idx = 0; idx < m_pdata->getN(); ++idx)
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
// ParticleSelectorRigid

/*! \param sysdef System the particles are to be selected from
    \param rigid true selects particles that are in rigid bodies, false selects particles that are not part of a rigid body
*/
ParticleSelectorRigid::ParticleSelectorRigid(std::shared_ptr<SystemDefinition> sysdef,
                                             bool rigid)
    : ParticleSelector(sysdef), m_rigid(rigid)
    {
    }

/*! \param tag Tag of the particle to check
    \returns true if the type of particle \a tag meets the rigid criteria selected
*/
std::vector<unsigned int> ParticleSelectorRigid::getSelectedTags() const
    {
    std::vector<unsigned int> member_tags;

    // loop through local particles and select those that match selection criterion
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_body(m_pdata->getBodies(), access_location::host, access_mode::read);
    for (unsigned int idx = 0; idx < m_pdata->getN(); ++idx)
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
// ParticleSelectorFloppy

/*! \param sysdef System the particles are to be selected from
    \param floppy true selects particles that are in floppy bodies, false selects particles that are not part of a floppy (non-rigid body)
*/
ParticleSelectorFloppy::ParticleSelectorFloppy(std::shared_ptr<SystemDefinition> sysdef,
                                             bool floppy)
    : ParticleSelector(sysdef), m_floppy(floppy)
    {
    }

/*! \param tag Tag of the particle to check
    \returns true if the type of particle \a tag meets the rigid criteria selected
*/
std::vector<unsigned int> ParticleSelectorFloppy::getSelectedTags() const
    {
    std::vector<unsigned int> member_tags;

    // loop through local particles and select those that match selection criterion
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_body(m_pdata->getBodies(), access_location::host, access_mode::read);
    for (unsigned int idx = 0; idx < m_pdata->getN(); ++idx)
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


//////////////////////////////////////////////////////////////////////////////
// ParticleSelectorRigidCenter

ParticleSelectorRigidCenter::ParticleSelectorRigidCenter(std::shared_ptr<SystemDefinition> sysdef)
    :ParticleSelector(sysdef)
    {
    }

/*! \param tag Tag of the particle to check
    \returns true if the type of particle \a tag is a center particle of a rigid body
*/
std::vector<unsigned int> ParticleSelectorRigidCenter::getSelectedTags() const
    {
    std::vector<unsigned int> member_tags;

    // loop through local particles and select those that match selection criterion
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_body(m_pdata->getBodies(), access_location::host, access_mode::read);
    for (unsigned int idx = 0; idx < m_pdata->getN(); ++idx)
        {
        unsigned int tag = h_tag.data[idx];

        // get position of particle
        unsigned int body = h_body.data[idx];

        if (body==tag)
            member_tags.push_back(tag);
        }
    return member_tags;
    }


//////////////////////////////////////////////////////////////////////////////
// ParticleSelectorCuboid

ParticleSelectorCuboid::ParticleSelectorCuboid(std::shared_ptr<SystemDefinition> sysdef, Scalar3 min, Scalar3 max)
    :ParticleSelector(sysdef), m_min(min), m_max(max)
    {
    // make a quick check on the sanity of the input data
    if (m_min.x >= m_max.x || m_min.y >= m_max.y || m_min.z >= m_max.z)
        m_exec_conf->msg->warning() << "group: max < min specified when selecting particle in a cuboid" << endl;
    }

/*! \param tag Tag of the particle to check
    \returns true if the type of particle \a tag is in the cuboid

    Evaluation is performed by \a m_min.x <= x < \a m_max.x so that multiple cuboids stacked next to each other
    do not have overlapping sets of particles.
*/
std::vector<unsigned int> ParticleSelectorCuboid::getSelectedTags() const
    {
    std::vector<unsigned int> member_tags;

    // loop through local particles and select those that match selection criterion
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
    for (unsigned int idx = 0; idx < m_pdata->getN(); ++idx)
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
    \param selector ParticleSelector used to choose the group members
    \param update_tags If true, update tags whenever global particle number changes

    Particles where criteria falls within the range [min,max] (inclusive) are added to the group.
*/
ParticleGroup::ParticleGroup(std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<ParticleSelector> selector,
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
    #ifdef ENABLE_CUDA
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

    #ifdef ENABLE_CUDA
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
        vector<unsigned int> member_tags = m_selector->getSelectedTags();

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

    #ifdef ENABLE_CUDA
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

    #ifdef ENABLE_CUDA
    if (m_pdata->getExecConf()->isCUDAEnabled())
        {
        // Update GPU load balancing info
        m_gpu_partition.setN(m_num_local_members);
        }
    #endif
    }

void ParticleGroup::updateGPUAdvice() const
    {
    #ifdef ENABLE_CUDA
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

#ifdef ENABLE_CUDA
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
            .def(py::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleSelector>, bool >())
            .def(py::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleSelector> >())
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

    py::class_<ParticleSelector, std::shared_ptr<ParticleSelector> >(m,"ParticleSelector")
            .def(py::init< std::shared_ptr<SystemDefinition> >())
            .def("getSelectedTags", &ParticleSelector::getSelectedTags)
            ;

    py::class_<ParticleSelectorAll, std::shared_ptr<ParticleSelectorAll> >(m,"ParticleSelectorAll",py::base<ParticleSelector>())
            .def(py::init< std::shared_ptr<SystemDefinition> >())
        ;

    py::class_<ParticleSelectorTag, std::shared_ptr<ParticleSelectorTag> >(m,"ParticleSelectorTag",py::base<ParticleSelector>())
            .def(py::init< std::shared_ptr<SystemDefinition>, unsigned int, unsigned int >())
        ;

    py::class_<ParticleSelectorType, std::shared_ptr<ParticleSelectorType> >(m,"ParticleSelectorType",py::base<ParticleSelector>())
            .def(py::init< std::shared_ptr<SystemDefinition>, unsigned int, unsigned int >())
        ;

    py::class_<ParticleSelectorRigid, std::shared_ptr<ParticleSelectorRigid> >(m,"ParticleSelectorRigid",py::base<ParticleSelector>())
            .def(py::init< std::shared_ptr<SystemDefinition>, bool >())
        ;

    py::class_<ParticleSelectorBody, std::shared_ptr<ParticleSelectorBody> >(m,"ParticleSelectorBody",py::base<ParticleSelector>())

            .def(py::init< std::shared_ptr<SystemDefinition>, bool >())
        ;

    py::class_<ParticleSelectorFloppy, std::shared_ptr<ParticleSelectorFloppy> >(m,"ParticleSelectorFloppy",py::base<ParticleSelector>())
            .def(py::init< std::shared_ptr<SystemDefinition>, bool >())
        ;

    py::class_<ParticleSelectorCuboid, std::shared_ptr<ParticleSelectorCuboid> >(m,"ParticleSelectorCuboid",py::base<ParticleSelector>())
            .def(py::init< std::shared_ptr<SystemDefinition>, Scalar3, Scalar3 >())
        ;

    py::class_<ParticleSelectorRigidCenter, std::shared_ptr<ParticleSelectorRigidCenter> >(m,"ParticleSelectorRigidCenter",py::base<ParticleSelector>())
            .def(py::init< std::shared_ptr<SystemDefinition> >())
        ;
    }
