// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ParticleGroup.h"

#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"

#ifdef ENABLE_HIP
#include "CachedAllocator.h"
#include "ParticleGroup.cuh"

#include <hip/hip_runtime.h>
#endif

#include <algorithm>
#include <iostream>
using namespace std;

namespace hoomd
    {
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
    : m_sysdef(sysdef), m_pdata(sysdef->getParticleData()), m_exec_conf(m_pdata->getExecConf()),
      m_num_local_members(0), m_particles_sorted(true), m_reallocated(false),
      m_global_ptl_num_change(false), m_selector(selector), m_update_tags(update_tags),
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
    m_pdata->getMaxParticleNumberChangeSignal()
        .connect<ParticleGroup, &ParticleGroup::slotReallocate>(this);

    // connect updateMemberTags() method to maximum particle number change signal
    m_pdata->getGlobalParticleNumberChangeSignal()
        .connect<ParticleGroup, &ParticleGroup::slotGlobalParticleNumChange>(this);

    // update GPU memory hints
    updateGPUAdvice();
    }

/*! \param sysdef System definition to build the group from
    \param member_tags List of particle tags that belong to the group

    All particles specified in \a member_tags will be added to the group.
*/
ParticleGroup::ParticleGroup(std::shared_ptr<SystemDefinition> sysdef,
                             const std::vector<unsigned int>& member_tags)
    : m_sysdef(sysdef), m_pdata(sysdef->getParticleData()), m_exec_conf(m_pdata->getExecConf()),
      m_num_local_members(0), m_particles_sorted(true), m_reallocated(false),
      m_global_ptl_num_change(false), m_update_tags(false), m_warning_printed(false)
    {
    // check input
    unsigned int max_tag = m_pdata->getMaximumTag();
    for (std::vector<unsigned int>::const_iterator it = member_tags.begin();
         it != member_tags.end();
         ++it)
        {
        if (*it > max_tag)
            {
            m_exec_conf->msg->error()
                << "group.*: Member " << *it << " does not exist in particle data." << std::endl;
            throw std::runtime_error("Error creating ParticleGroup\n");
            }
        }

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        // do a simple sanity check
        unsigned int nptl = (unsigned int)member_tags.size();
        bcast(nptl, 0, m_exec_conf->getMPICommunicator());

        if (nptl != member_tags.size())
            {
            m_exec_conf->msg->error()
                << "group.*: Member tag list is inconsistent among MPI ranks." << std::endl;
            throw std::runtime_error("Error creating ParticleGroup\n");
            }
        }
#endif

    // let's make absolutely sure that the tag order given from outside is sorted
    std::vector<unsigned int> sorted_member_tags = member_tags;
    sort(sorted_member_tags.begin(), sorted_member_tags.end());

    // store member tags
    GlobalArray<unsigned int> member_tags_array(member_tags.size(), m_exec_conf);
    m_member_tags.swap(member_tags_array);
    TAG_ALLOCATION(m_member_tags);

        {
        ArrayHandle<unsigned int> h_member_tags(m_member_tags,
                                                access_location::host,
                                                access_mode::overwrite);
        std::copy(sorted_member_tags.begin(), sorted_member_tags.end(), h_member_tags.data);
        }

    // one byte per particle to indicate membership in the group, initialize with current number of
    // local particles
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

    // now that the tag list is completely set up and all memory is allocated, rebuild the index
    // list
    rebuildIndexList();

    // connect to the particle sort signal
    m_pdata->getParticleSortSignal().connect<ParticleGroup, &ParticleGroup::slotParticleSort>(this);

    // connect reallocate() method to maximum particle number change signal
    m_pdata->getMaxParticleNumberChangeSignal()
        .connect<ParticleGroup, &ParticleGroup::slotReallocate>(this);

    // connect updateMemberTags() method to maximum particle number change signal
    m_pdata->getGlobalParticleNumberChangeSignal()
        .connect<ParticleGroup, &ParticleGroup::slotGlobalParticleNumChange>(this);

    // update GPU memory hints
    updateGPUAdvice();
    }

ParticleGroup::~ParticleGroup()
    {
    // disconnect the sort connection, but only if there was a particle data to connect it to in the
    // first place
    if (m_pdata)
        {
        m_pdata->getParticleSortSignal()
            .disconnect<ParticleGroup, &ParticleGroup::slotParticleSort>(this);
        m_pdata->getMaxParticleNumberChangeSignal()
            .disconnect<ParticleGroup, &ParticleGroup::slotReallocate>(this);
        m_pdata->getGlobalParticleNumberChangeSignal()
            .disconnect<ParticleGroup, &ParticleGroup::slotGlobalParticleNumChange>(this);
        }
    }

/*! \param force_update If true, always update member tags
 */
void ParticleGroup::updateMemberTags(bool force_update)
    {
    if (m_selector && !(m_update_tags || force_update) && !m_warning_printed)
        {
        m_pdata->getExecConf()->msg->warning()
            << "Particle number change but group is static. Create group with update=True if it "
               "should be updated."
            << std::endl
            << "This warning is printed only once." << std::endl;
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
            std::vector<std::vector<unsigned int>> member_tags_proc(m_exec_conf->getNRanks());
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
            ArrayHandle<unsigned int> h_member_tags(m_member_tags,
                                                    access_location::host,
                                                    access_mode::overwrite);
            std::copy(member_tags.begin(), member_tags.end(), h_member_tags.data);
            }

        GlobalArray<unsigned int> member_idx(member_tags.size(), m_pdata->getExecConf());
        m_member_idx.swap(member_idx);
        TAG_ALLOCATION(m_member_idx);
        }

    // one byte per particle to indicate membership in the group, initialize with current number of
    // local particles
    GlobalArray<unsigned int> is_member(m_pdata->getMaxN(), m_pdata->getExecConf());
    m_is_member.swap(is_member);
    TAG_ALLOCATION(m_is_member);

    GlobalArray<unsigned int> is_member_tag(m_pdata->getRTags().size(), m_pdata->getExecConf());
    m_is_member_tag.swap(is_member_tag);
    TAG_ALLOCATION(m_is_member_tag);

    // build the reverse lookup table for tags
    buildTagHash();

    // now that the tag list is completely set up and all memory is allocated, rebuild the index
    // list
    rebuildIndexList();

    // count the number of central and free particles in the group
    // updateMemberTags cannot call any member function that would result in a checkRebuild() call
    m_n_central_and_free_global = 0;

    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_body(m_pdata->getBodies(),
                                     access_location::host,
                                     access_mode::read);
    ArrayHandle<unsigned int> h_is_member_tag(m_is_member_tag,
                                              access_location::host,
                                              access_mode::read);
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        unsigned int tag = h_tag.data[i];
        unsigned int body = h_body.data[i];

        if (h_is_member_tag.data[tag] && (body == tag || body > MIN_FLOPPY))
            {
            m_n_central_and_free_global++;
            }
        }

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        MPI_Allreduce(MPI_IN_PLACE,
                      &m_n_central_and_free_global,
                      1,
                      MPI_UNSIGNED,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());
        }
#endif
    }

void ParticleGroup::reallocate()
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
Scalar ParticleGroup::getTotalMass()
    {
    // grab the particle data
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::read);

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
Scalar3 ParticleGroup::getCenterOfMass()
    {
    // grab the particle data
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::read);

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

    \returns A shared pointer to a newly created particle group that contains all the elements
   present in \a a and \a b
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
        ArrayHandle<unsigned int> h_members_a(a->m_member_tags,
                                              access_location::host,
                                              access_mode::read);
        ArrayHandle<unsigned int> h_members_b(b->m_member_tags,
                                              access_location::host,
                                              access_mode::read);

        insert_iterator<vector<unsigned int>> ii(member_tags, member_tags.begin());
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
        ArrayHandle<unsigned int> h_members_a(a->m_member_tags,
                                              access_location::host,
                                              access_mode::read);

        insert_iterator<vector<unsigned int>> ii(member_tags, member_tags.begin());
        std::copy(h_members_a.data, h_members_a.data + n_a, ii);
        }

    // create the new particle group
    std::shared_ptr<ParticleGroup> new_group(new ParticleGroup(a->m_sysdef, member_tags));

    // return the newly created group
    return new_group;
    }

/*! \param a First particle group
    \param b Second particle group

    \returns A shared pointer to a newly created particle group that contains only the elements
   present in both \a a and \a b
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
        ArrayHandle<unsigned int> h_members_a(a->m_member_tags,
                                              access_location::host,
                                              access_mode::read);
        ArrayHandle<unsigned int> h_members_b(b->m_member_tags,
                                              access_location::host,
                                              access_mode::read);

        insert_iterator<vector<unsigned int>> ii(member_tags, member_tags.begin());
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
        ArrayHandle<unsigned int> h_members_a(a->m_member_tags,
                                              access_location::host,
                                              access_mode::read);

        insert_iterator<vector<unsigned int>> ii(member_tags, member_tags.begin());
        std::copy(h_members_a.data, h_members_a.data + n_a, ii);
        }

    // create the new particle group
    std::shared_ptr<ParticleGroup> new_group(new ParticleGroup(a->m_sysdef, member_tags));

    // return the newly created group
    return new_group;
    }

/*! \param a First particle group
    \param b Second particle group

    \returns A shared pointer to a newly created particle group that contains only the elements
   present in \a a, and not any present in \a b
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
        ArrayHandle<unsigned int> h_members_a(a->m_member_tags,
                                              access_location::host,
                                              access_mode::read);
        ArrayHandle<unsigned int> h_members_b(b->m_member_tags,
                                              access_location::host,
                                              access_mode::read);

        insert_iterator<vector<unsigned int>> ii(member_tags, member_tags.begin());
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
void ParticleGroup::buildTagHash()
    {
    ArrayHandle<unsigned int> h_is_member_tag(m_is_member_tag,
                                              access_location::host,
                                              access_mode::overwrite);
    ArrayHandle<unsigned int> h_member_tags(m_member_tags,
                                            access_location::host,
                                            access_mode::read);

    // reset member ship flags
    memset(h_is_member_tag.data, 0, sizeof(unsigned int) * (m_pdata->getRTags().size()));

    size_t num_members = m_member_tags.getNumElements();
    for (size_t member = 0; member < num_members; member++)
        {
        h_is_member_tag.data[h_member_tags.data[member]] = 1;
        }
    }

/*! \pre m_member_tags has been filled out, listing all particle tags in the group
    \pre memory has been allocated for m_is_member and m_member_idx
    \post m_is_member is updated so that it reflects the current indices of the particles in the
   group \post m_member_idx is updated listing all particle indices belonging to the group, in index
   order
*/
void ParticleGroup::rebuildIndexList()
    {
    // notice message
    m_pdata->getExecConf()->msg->notice(10) << "ParticleGroup: rebuilding index" << std::endl;

#ifdef ENABLE_HIP
    if (m_pdata->getExecConf()->isCUDAEnabled())
        {
        rebuildIndexListGPU();
        }
    else
#endif
        {
        // rebuild the membership flags for the  indices in the group and construct member list
        ArrayHandle<unsigned int> h_is_member(m_is_member,
                                              access_location::host,
                                              access_mode::readwrite);
        ArrayHandle<unsigned int> h_is_member_tag(m_is_member_tag,
                                                  access_location::host,
                                                  access_mode::read);
        ArrayHandle<unsigned int> h_tag(m_pdata->getTags(),
                                        access_location::host,
                                        access_mode::read);
        ArrayHandle<unsigned int> h_member_idx(m_member_idx,
                                               access_location::host,
                                               access_mode::readwrite);
        unsigned int nparticles = m_pdata->getN();
        unsigned int cur_member = 0;
        for (unsigned int idx = 0; idx < nparticles; idx++)
            {
            assert(h_tag.data[idx] <= m_pdata->getMaximumTag());
            unsigned int is_member = h_is_member_tag.data[h_tag.data[idx]];
            h_is_member.data[idx] = is_member;
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

void ParticleGroup::updateGPUAdvice()
    {
#if defined(ENABLE_HIP) && defined(__HIP_PLATFORM_NVCC__)
    if (m_exec_conf->isCUDAEnabled() && m_exec_conf->allConcurrentManagedAccess())
        {
        // split preferred location of group indices across GPUs
        auto gpu_map = m_exec_conf->getGPUIds();
        for (unsigned int idev = 0; idev < m_exec_conf->getNumActiveGPUs(); ++idev)
            {
            auto range = m_gpu_partition.getRange(idev);
            unsigned int nelem = range.second - range.first;

            if (!nelem)
                continue;

            cudaMemAdvise(m_member_idx.get() + range.first,
                          sizeof(unsigned int) * nelem,
                          cudaMemAdviseSetPreferredLocation,
                          gpu_map[idev]);
            cudaMemAdvise(m_is_member.get() + range.first,
                          sizeof(unsigned int) * nelem,
                          cudaMemAdviseSetPreferredLocation,
                          gpu_map[idev]);

            // migrate data to preferred location
            cudaMemPrefetchAsync(m_member_idx.get() + range.first,
                                 sizeof(unsigned int) * nelem,
                                 gpu_map[idev]);
            cudaMemPrefetchAsync(m_is_member.get() + range.first,
                                 sizeof(unsigned int) * nelem,
                                 gpu_map[idev]);
            }
        CHECK_CUDA_ERROR();
        }
#endif
    }

#ifdef ENABLE_HIP
//! rebuild index list on the GPU
void ParticleGroup::rebuildIndexListGPU()
    {
    ArrayHandle<unsigned int> d_is_member(m_is_member,
                                          access_location::device,
                                          access_mode::overwrite);
    ArrayHandle<unsigned int> d_is_member_tag(m_is_member_tag,
                                              access_location::device,
                                              access_mode::read);
    ArrayHandle<unsigned int> d_member_idx(m_member_idx,
                                           access_location::device,
                                           access_mode::overwrite);
    ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);

    // get temporary buffer
    ScopedAllocation<unsigned int> d_tmp(m_pdata->getExecConf()->getCachedAllocator(),
                                         m_pdata->getN());

    // reset membership properties
    if (m_member_tags.getNumElements() > 0)
        {
        kernel::gpu_rebuild_index_list(m_pdata->getN(),
                                       d_is_member_tag.data,
                                       d_is_member.data,
                                       d_tag.data);
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        kernel::gpu_compact_index_list(m_pdata->getN(),
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

unsigned int ParticleGroup::intersectionSize(std::shared_ptr<ParticleGroup> other)
    {
    unsigned int n = 0;

    for (unsigned int i = 0; i < this->getNumMembers(); i++)
        {
        unsigned int query_idx = this->getMemberIndex(i);
        if (other->isMember(query_idx))
            n++;
        }

#ifdef ENABLE_MPI
    MPI_Allreduce(MPI_IN_PLACE, &n, 1, MPI_UNSIGNED, MPI_SUM, m_exec_conf->getMPICommunicator());
#endif

    return n;
    }

void ParticleGroup::thermalizeParticleMomenta(Scalar kT, uint64_t timestep)
    {
    unsigned int group_size = this->getNumMembers();

    const unsigned int n_dimensions = m_sysdef->getNDimensions();

    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(),
                               access_location::host,
                               access_mode::readwrite);

    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                       access_location::host,
                                       access_mode::read);

    ArrayHandle<Scalar4> h_angmom(m_pdata->getAngularMomentumArray(),
                                  access_location::host,
                                  access_mode::readwrite);

    ArrayHandle<Scalar3> h_inertia(m_pdata->getMomentsOfInertiaArray(),
                                   access_location::host,
                                   access_mode::read);

    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_body(m_pdata->getBodies(),
                                     access_location::host,
                                     access_mode::read);

    // Total the system's linear momentum
    vec3<Scalar> tot_momentum(0, 0, 0);

    // Loop over all particles in the group
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)

        {
        unsigned int j = this->getMemberIndex(group_idx);
        unsigned int ptag = h_tag.data[j];

        // Seed the RNG
        hoomd::RandomGenerator rng(hoomd::Seed(hoomd::RNGIdentifier::ParticleGroupThermalize,
                                               timestep,
                                               m_sysdef->getSeed()),
                                   hoomd::Counter(ptag));

        // Generate a random velocity, excluding constituent particles of rigid bodies
        Scalar mass = h_vel.data[j].w;
        Scalar sigma = slow::sqrt(kT / mass);
        hoomd::NormalDistribution<Scalar> normal(sigma);
        // check if particles are constituent particles of a rigid body
        if (h_tag.data[j] != h_body.data[j] && h_body.data[j] < MIN_FLOPPY)
            {
            h_vel.data[j].x = 0;
            h_vel.data[j].y = 0;
            h_vel.data[j].z = 0;
            }
        else
            {
            h_vel.data[j].x = normal(rng);
            h_vel.data[j].y = normal(rng);
            if (n_dimensions > 2)
                h_vel.data[j].z = normal(rng);
            else
                h_vel.data[j].z = 0; // For 2D systems
            }
        tot_momentum += mass * vec3<Scalar>(h_vel.data[j]);

        // Generate random angular momentum if the particle has rotational degrees of freedom.
        vec3<Scalar> p_vec(0, 0, 0);
        quat<Scalar> q(h_orientation.data[j]);
        vec3<Scalar> I(h_inertia.data[j]);

        if (h_tag.data[j] == h_body.data[j] || h_body.data[j] >= MIN_FLOPPY)
            {
            if (I.x > 0)
                p_vec.x = hoomd::NormalDistribution<Scalar>(slow::sqrt(kT * I.x))(rng);
            if (I.y > 0)
                p_vec.y = hoomd::NormalDistribution<Scalar>(slow::sqrt(kT * I.y))(rng);
            if (I.z > 0)
                p_vec.z = hoomd::NormalDistribution<Scalar>(slow::sqrt(kT * I.z))(rng);
            }

        // Store the angular momentum quaternion
        quat<Scalar> p = Scalar(2.0) * q * p_vec;
        h_angmom.data[j] = quat_to_scalar4(p);
        }

    // Remove the center of mass momentum

#ifdef ENABLE_MPI
    // Reduce the total momentum from all MPI ranks
    if (m_pdata->getDomainDecomposition())
        {
        MPI_Allreduce(MPI_IN_PLACE,
                      &tot_momentum,
                      3,
                      MPI_HOOMD_SCALAR,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());
        }
#endif

    vec3<Scalar> com_momentum(tot_momentum / Scalar(this->getNumMembersGlobal()));

    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = this->getMemberIndex(group_idx);
        Scalar mass = h_vel.data[j].w;
        if (h_tag.data[j] == h_body.data[j] || h_body.data[j] == NO_BODY)
            {
            h_vel.data[j].x = h_vel.data[j].x - com_momentum.x / mass;
            h_vel.data[j].y = h_vel.data[j].y - com_momentum.y / mass;
            if (n_dimensions > 2)
                h_vel.data[j].z = h_vel.data[j].z - com_momentum.z / mass;
            else
                h_vel.data[j].z = 0; // For 2D systems
            }
        }
    }

namespace detail
    {
void export_ParticleGroup(pybind11::module& m)
    {
    pybind11::class_<ParticleGroup, std::shared_ptr<ParticleGroup>>(m, "ParticleGroup")
        .def(pybind11::
                 init<std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleFilter>, bool>())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleFilter>>())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, const std::vector<unsigned int>&>())
        .def("getNumMembersGlobal", &ParticleGroup::getNumMembersGlobal)
        .def("getMemberTag", &ParticleGroup::getMemberTag)
        .def("getTotalMass", &ParticleGroup::getTotalMass)
        .def("getCenterOfMass", &ParticleGroup::getCenterOfMass)
        .def("groupUnion", &ParticleGroup::groupUnion)
        .def("groupIntersection", &ParticleGroup::groupIntersection)
        .def("groupDifference", &ParticleGroup::groupDifference)
        .def("updateMemberTags", &ParticleGroup::updateMemberTags)
        .def("setTranslationalDOF", &ParticleGroup::setTranslationalDOF)
        .def("getTranslationalDOF", &ParticleGroup::getTranslationalDOF)
        .def("setRotationalDOF", &ParticleGroup::setRotationalDOF)
        .def("getRotationalDOF", &ParticleGroup::getRotationalDOF)
        .def("thermalizeParticleMomenta", &ParticleGroup::thermalizeParticleMomenta)
        .def_property_readonly("member_tags", &ParticleGroup::getMemberTags);
    }

    } // end namespace detail

    } // end namespace hoomd
