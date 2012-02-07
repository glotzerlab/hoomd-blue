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

// Maintainer: joaander

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4267 4244 )
#endif

#include "ParticleGroup.h"

#include <boost/python.hpp>
#include <boost/bind.hpp>

using namespace boost::python;
using namespace boost;

#include <algorithm>
#include <iostream>
using namespace std;

#ifdef ENABLE_CUDA
#include "ParticleGroup.cuh"
#endif

/*! \file ParticleGroup.cc
    \brief Defines the ParticleGroup and related classes
*/

//////////////////////////////////////////////////////////////////////////////
// ParticleSelectorTag

/*! \param sysdef System the particles are to be selected from
    \param tag_min Minimum tag to select (inclusive)
    \param tag_max Maximum tag to select (inclusive)
*/
ParticleSelectorTag::ParticleSelectorTag(boost::shared_ptr<SystemDefinition> sysdef,
                                         unsigned int tag_min,
                                         unsigned int tag_max)
    : ParticleSelectorRule<GlobalTagRule>(sysdef)
    {
    // make a quick check on the sanity of the input data
    if (tag_max < tag_min)
        cout << "***Warning! max < min specified when selecting particle tags" << endl;
    
    if (tag_max >= m_pdata->getNGlobal())
        {
        cerr << endl << "***Error! Cannot select particles with tags larger than the number of particles " 
             << endl << endl;
        throw runtime_error("Error selecting particles");
        }

    setParams(make_uint2(tag_min, tag_max));
    }

//////////////////////////////////////////////////////////////////////////////
// ParticleSelectorType

/*! \param sysdef System the particles are to be selected from
    \param typ_min Minimum type id to select (inclusive)
    \param typ_max Maximum type id to select (inclusive)
    */
    ParticleSelectorType::ParticleSelectorType(boost::shared_ptr<SystemDefinition> sysdef,
                                           unsigned int typ_min,
                                           unsigned int typ_max)
    : ParticleSelectorRule<TypeRule>(sysdef)
    {
    // make a quick check on the sanity of the input data
    if (typ_max < typ_min)
        cout << "***Warning! max < min specified when selecting particle types" << endl;
    
    if (typ_max >= m_pdata->getNTypes())
        cout << "***Warning! Requesting for the selection of a non-existant particle type" << endl;

    setParams(make_uint2(typ_min, typ_max));
    }

//////////////////////////////////////////////////////////////////////////////
// ParticleSelectorRigid

/*! \param sysdef System the particles are to be selected from
    \param rigid true selects particles that are in rigid bodies, false selects particles that are not part of a body
*/
ParticleSelectorRigid::ParticleSelectorRigid(boost::shared_ptr<SystemDefinition> sysdef,
                                             bool rigid)
    : ParticleSelectorRule<RigidRule>(sysdef)
    {
    this->setParams(rigid);
    }

//////////////////////////////////////////////////////////////////////////////
// ParticleSelectorCuboid

#if 0
ParticleSelectorCuboid::ParticleSelectorCuboid(boost::shared_ptr<SystemDefinition> sysdef, Scalar3 min, Scalar3 max)
    :ParticleSelector(sysdef), m_min(min), m_max(max)
    {
    // make a quick check on the sanity of the input data
    if (m_min.x >= m_max.x || m_min.y >= m_max.y || m_min.z >= m_max.z)
        cout << "***Warning! max < min specified when selecting particle in a cuboid" << endl;

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_global_tag(m_pdata->getGlobalTags(), access_location::host, access_mode::read);

    // scan all local particles based on their position
    for (unsigned int idx = 0; idx < m_pdata->getN(); idx++)
        {
        Scalar4 pos = h_pos.data[idx];
        if (m_min.x <= pos.x && pos.x < m_max.x &&
            m_min.y <= pos.y && pos.y < m_max.y &&
            m_min.z <= pos.z && pos.z < m_max.z)
            {
            unsigned int global_tag = h_global_tag.data[idx];


        }
    }

/*! \param tag Tag of the particle to check
    \returns true if the type of particle \a tag is in the cuboid
    
    Evaluation is performed by \a m_min.x <= x < \a m_max.x so that multiple cuboids stacked next to each other
    do not have overlapping sets of particles.
*/

#endif

//////////////////////////////////////////////////////////////////////////////
// ParticleSelectorEmpty
//
//////////////////////////////////////////////////////////////////////////////
// ParticleSelectorGlobalTagList

//! Constructor
ParticleSelectorGlobalTagList::ParticleSelectorGlobalTagList(boost::shared_ptr<SystemDefinition> sysdef, const std::vector<unsigned int>& global_tag_list)
    : m_sysdef(sysdef), m_pdata(sysdef->getParticleData())
    {

    // copy the tag list into a GPU array for efficient access on the GPU
    GPUArray<unsigned int> global_tags(global_tag_list.size(), m_pdata->getExecConf());
    m_global_member_tags.swap(global_tags);

    ArrayHandle<unsigned int> h_global_member_tags(m_global_member_tags, access_location::host, access_mode::overwrite);
    std::copy(global_tag_list.begin(), global_tag_list.end(), h_global_member_tags.data);

    }

//! rebuild list of particle global tags that are members of the group and which are owned by the ParticleData
unsigned int ParticleSelectorGlobalTagList::getMemberTags(const GPUArray<unsigned int> &member_tags)
    {
    assert(member_tags.getNumElements() >= m_pdata->getN());

    ArrayHandle<unsigned int> h_tag(m_pdata->getGlobalTags(), access_location::host, access_mode::read);

    // global tags of local members of the group
    ArrayHandle<unsigned int> h_member_tags(member_tags, access_location::host, access_mode::overwrite);

    // all global tags that are members of the group
    ArrayHandle<unsigned int> h_global_member_tags(m_global_member_tags, access_location::host, access_mode::read);

    unsigned int num_members;
    // find intersection of global tag list with global tags of local particles
    num_members =  std::set_intersection(h_global_member_tags.data,
                                         h_global_member_tags.data + m_global_member_tags.getNumElements(),
                                         h_tag.data,
                                         h_tag.data + m_pdata->getN(),
                                         h_member_tags.data) - h_member_tags.data;
    return num_members;
    }

//////////////////////////////////////////////////////////////////////////////
// ParticleSelector set operations

//! Constructor for ParticleSelectorSetOperation
/*! We construct the union of two ParticleSelectors by storing copies of them in class member variables (instead
    of references) and taking the union of their member tags. In this way cyclic references between ParticleSelectors are prevented.
 */
ParticleSelectorSetOperation::ParticleSelectorSetOperation(boost::shared_ptr<SystemDefinition> sysdef,
                                                           boost::shared_ptr<ParticleSelector> a,
                                                           boost::shared_ptr<ParticleSelector> b)
    : ParticleSelector(),
      m_sysdef(sysdef),
      m_pdata(sysdef->getParticleData()),
      m_selector_a(a),
      m_selector_b(b)
    {
    // initialize arrays to hold tags of local group members as returned by both selectors
    GPUArray<unsigned int> member_tags_a(m_pdata->getN(), m_pdata->getExecConf());
    GPUArray<unsigned int> member_tags_b(m_pdata->getN(), m_pdata->getExecConf());
    m_member_tags_a.swap(member_tags_a);
    m_member_tags_b.swap(member_tags_b);
    }

//! rebuild internal list of included tags
unsigned int ParticleSelectorSetOperation::getMemberTags(const GPUArray<unsigned int>& member_tags)
    {
    // resize arrays if necessary
    if (m_pdata->getN() > m_member_tags_a.getNumElements())
        {
        unsigned int new_size = m_member_tags_a.getNumElements();
        while (m_pdata->getN() > new_size) new_size *= 2;
        m_member_tags_a.resize(new_size);
        }

    if (m_pdata->getN() > m_member_tags_b.getNumElements())
        {
        unsigned int new_size = m_member_tags_b.getNumElements();
        while (m_pdata->getN() > new_size) new_size *= 2;
        m_member_tags_b.resize(new_size);
        }

    // get local members from both ParticleSelectors
    unsigned int num_members_a = m_selector_a->getMemberTags(m_member_tags_a);
    unsigned int num_members_b = m_selector_b->getMemberTags(m_member_tags_b);

    assert(num_members_a <= m_pdata->getN());
    assert(num_members_b <= m_pdata->getN());
    assert(member_tags.getNumElements() >= m_pdata->getN());

    // perform the set operation
    return operation(m_member_tags_a, m_member_tags_b, num_members_a, num_members_b, member_tags);
    }


//! Constructor for ParticleSelectorUnion
ParticleSelectorUnion::ParticleSelectorUnion(boost::shared_ptr<SystemDefinition> sysdef,
                                             boost::shared_ptr<ParticleSelector> a,
                                             boost::shared_ptr<ParticleSelector> b)
    : ParticleSelectorSetOperation(sysdef, a, b)
    { }

//! Implementation of a union of two particle selectors
unsigned int ParticleSelectorUnion::operation(const GPUArray<unsigned int>& member_tags_a,
                                 const GPUArray<unsigned int>& member_tags_b,
                                 const unsigned int num_members_a,
                                 const unsigned int num_members_b,
                                 const GPUArray<unsigned int>& member_tags)
    {
    ArrayHandle<unsigned int> h_member_tags_a(member_tags_a,access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_member_tags_b(member_tags_b,access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_member_tags(member_tags,access_location::host, access_mode::readwrite);

    return std::set_union(h_member_tags_a.data,
                          h_member_tags_a.data + num_members_a,
                          h_member_tags_b.data,
                          h_member_tags_b.data + num_members_b,
                          h_member_tags.data) - h_member_tags.data;
    }

//! Constructor for ParticleSelectorDifference
ParticleSelectorDifference::ParticleSelectorDifference(boost::shared_ptr<SystemDefinition> sysdef,
                                             boost::shared_ptr<ParticleSelector> a,
                                             boost::shared_ptr<ParticleSelector> b)
    : ParticleSelectorSetOperation(sysdef, a, b)
    { }

//! Implementation of a difference between two particle selectors
unsigned int ParticleSelectorDifference::operation(const GPUArray<unsigned int>& member_tags_a,
                                 const GPUArray<unsigned int>& member_tags_b,
                                 const unsigned int num_members_a,
                                 const unsigned int num_members_b,
                                 const GPUArray<unsigned int>& member_tags)
    {
    ArrayHandle<unsigned int> h_member_tags_a(member_tags_a,access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_member_tags_b(member_tags_b,access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_member_tags(member_tags,access_location::host, access_mode::readwrite);

    return std::set_difference(h_member_tags_a.data,
                          h_member_tags_a.data + num_members_a,
                          h_member_tags_b.data,
                          h_member_tags_b.data + num_members_b,
                          h_member_tags.data) - h_member_tags.data;
    }

//! Constructor for ParticleSelectorIntersection
ParticleSelectorIntersection::ParticleSelectorIntersection(boost::shared_ptr<SystemDefinition> sysdef,
                                             boost::shared_ptr<ParticleSelector> a,
                                             boost::shared_ptr<ParticleSelector> b)
    : ParticleSelectorSetOperation(sysdef, a, b)
    { }

//! Implementation of a union of two particle selectors
unsigned int ParticleSelectorIntersection::operation(const GPUArray<unsigned int>& member_tags_a,
                                 const GPUArray<unsigned int>& member_tags_b,
                                 const unsigned int num_members_a,
                                 const unsigned int num_members_b,
                                 const GPUArray<unsigned int>& member_tags)
    {
    ArrayHandle<unsigned int> h_member_tags_a(member_tags_a,access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_member_tags_b(member_tags_b,access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_member_tags(member_tags,access_location::host, access_mode::readwrite);

    return std::set_intersection(h_member_tags_a.data,
                          h_member_tags_a.data + num_members_a,
                          h_member_tags_b.data,
                          h_member_tags_b.data + num_members_b,
                          h_member_tags.data) - h_member_tags.data;
    }

//////////////////////////////////////////////////////////////////////////////
// ParticleGroup

//! Empty particle group
ParticleGroup::ParticleGroup()
    : m_selector(boost::shared_ptr<ParticleSelector>(new ParticleSelectorEmptySet())),
      m_num_members(0),
      m_is_empty(true)
    {
    }

/*! \param sysdef System definition to build the group from
    \param selector ParticleSelector used to choose the group members

    Particles where criteria falls within the range [min,max] (inclusive) are added to the group.
*/
ParticleGroup::ParticleGroup(boost::shared_ptr<SystemDefinition> sysdef, boost::shared_ptr<ParticleSelector> selector)
    : m_sysdef(sysdef),
      m_pdata(sysdef->getParticleData()),
      m_selector(selector),
      m_is_empty(false)
    {
    // we use the number of loal particles as the maximum size for the member tags array
    GPUArray<unsigned int> member_tags(m_pdata->getN(), m_pdata->getExecConf());
    m_member_tags.swap(member_tags);

    m_num_members = m_selector->getMemberTags(m_member_tags);

    // one byte per particle to indicate membership in the group
    GPUArray<unsigned char> is_member(m_pdata->getN(), m_pdata->getExecConf());
    m_is_member.swap(is_member);

    GPUArray<unsigned int> member_idx(m_num_members, m_pdata->getExecConf());
    m_member_idx.swap(member_idx);

    // now that the tag list is completely set up and all memory is allocated, rebuild the index list
    rebuildIndexList();
    
    // connect the rebuildTagList method to be called whenever the particles are sorted
    // we could have equivalently used rebuildIndexList, but we have to keep in mind
    // that after sorting the order of global tags changes and that performance is better
    // if the member_tags are also updated to reflect the new ordering
    m_sort_connection = m_pdata->connectParticleSort(bind(&ParticleGroup::rebuildTagList, this));

    // connect the rebuildTagList() method to be called whenever particles are inserted or deleted
    m_particle_num_change_connection = m_pdata->connectParticleNumberChange(bind(&ParticleGroup::rebuildTagList, this));

    //! connect reallocate() method to maximum particle number change signal
    m_max_particle_num_change_connection = m_pdata->connectMaxParticleNumberChange(bind(&ParticleGroup::reallocate, this));
    }

/*! \param sysdef System definition to build the group from
    \param global_tag_list list of global tags to include in the group
 */
ParticleGroup::ParticleGroup(boost::shared_ptr<SystemDefinition> sysdef, const std::vector<unsigned int>& global_tag_list)
    : m_sysdef(sysdef),
      m_pdata(sysdef->getParticleData()),
      m_selector(boost::shared_ptr<ParticleSelectorGlobalTagList>(new ParticleSelectorGlobalTagList(sysdef, global_tag_list))),
      m_is_empty(false)
    {
    // we use the number of loal particles as the maximum size for the member tags array
    GPUArray<unsigned int> member_tags(m_pdata->getN(), m_pdata->getExecConf());
    m_member_tags.swap(member_tags);

    // one byte per particle to indicate membership in the group
    GPUArray<unsigned char> is_member(m_pdata->getN(), m_pdata->getExecConf());
    m_is_member.swap(is_member);

    m_num_members = m_selector->getMemberTags(m_member_tags);

    GPUArray<unsigned int> member_idx(m_num_members, m_pdata->getExecConf());
    m_member_idx.swap(member_idx);

    // now that the tag list is completely set up and all memory is allocated, rebuild the index list
    rebuildIndexList();

    // connect the rebuildTagList method to be called whenever the particles are sorted
    m_sort_connection = m_pdata->connectParticleSort(bind(&ParticleGroup::rebuildTagList, this));

    // connect the rebuildTagList() method to be called whenever particles are inserted or deleted
    m_particle_num_change_connection = m_pdata->connectParticleNumberChange(bind(&ParticleGroup::rebuildTagList, this));

    // connect reallocate() method to maximum particle number change signal
    m_max_particle_num_change_connection = m_pdata->connectMaxParticleNumberChange(bind(&ParticleGroup::reallocate, this));
    }

ParticleGroup::~ParticleGroup()
    {
    if (m_sort_connection.connected())
        // disconnect the sort connection
        m_sort_connection.disconnect();
    if (m_particle_num_change_connection.connected())
        // disconnect the particle number change connection
        m_particle_num_change_connection.disconnect();
    if (m_max_particle_num_change_connection.connected())
        // discconnect the max particle num change connection
        m_max_particle_num_change_connection.disconnect();
    }

void ParticleGroup::reallocate()
    {
    unsigned int max_particle_num = m_member_tags.getNumElements();

    if (max_particle_num < m_pdata->getMaxN())
        {
        // only resize if needed
        while (max_particle_num < m_pdata->getMaxN()) max_particle_num *= 2;
        m_member_tags.resize(max_particle_num);
        m_is_member.resize(max_particle_num);
        }
    }

//! Rebuild the tag list using the ParticleSelector
void ParticleGroup::rebuildTagList()
    {
    // first update our reference of the member tags list
    m_num_members = m_selector->getMemberTags(m_member_tags);

    // then rebuild the index array
    rebuildIndexList();
    }

/*! \returns Total mass of all particles in the group
    \note This method aquires the ParticleData internally

    FIXME: this will not work with global groups (yet)
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
    \note This method aquires the ParticleData internally

    FIXME: This will not work with global groups, as it returns the local center of mass
*/
Scalar3 ParticleGroup::getCenterOfMass() const
    {
    if (! m_pdata) return make_scalar3(0.0,0.0,0.0); // assume this is the global center of mass

    // grab the particle data
    ArrayHandle< Scalar4 > h_vel(m_pdata->getVelocities(), access_location::host, access_mode::read);
    ArrayHandle< Scalar4 > h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle< int3 > h_image(m_pdata->getImages(), access_location::host, access_mode::read);
    
    // grab the box dimensions
    BoxDim box = m_pdata->getBox();
    Scalar Lx = box.xhi - box.xlo;
    Scalar Ly = box.yhi - box.ylo;
    Scalar Lz = box.zhi - box.zlo;
    
    // loop  through all indices in the group and compute the weighted average of the positions
    Scalar total_mass = 0.0;
    Scalar3 center_of_mass = make_scalar3(Scalar(0.0), Scalar(0.0), Scalar(0.0));
    for (unsigned int i = 0; i < getNumMembers(); i++)
        {
        unsigned int idx = getMemberIndex(i);
        Scalar mass = h_vel.data[idx].w;
        total_mass += mass;
        center_of_mass.x += mass * (h_pos.data[idx].x + Scalar(h_image.data[idx].x) * Lx);
        center_of_mass.y += mass * (h_pos.data[idx].y + Scalar(h_image.data[idx].y) * Ly);
        center_of_mass.z += mass * (h_pos.data[idx].z + Scalar(h_image.data[idx].z) * Lz);
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
boost::shared_ptr<ParticleGroup> ParticleGroup::groupUnion(boost::shared_ptr<ParticleGroup> a,
                                                           boost::shared_ptr<ParticleGroup> b)
    {
    // create the union of ParticleSelectors

    // get a valid system definition
    boost::shared_ptr<SystemDefinition> sysdef;
    if (a->m_sysdef)
        sysdef = a->m_sysdef;
    else if (b->m_sysdef)
        sysdef = b->m_sysdef;

    // check if this is a union of two empty groups
    if (! sysdef)
        return boost::shared_ptr<ParticleGroup>(new ParticleGroup());

    boost::shared_ptr<ParticleSelector> p_sel_union(new ParticleSelectorUnion(sysdef, a->m_selector, b->m_selector));

    // create the new particle group
    boost::shared_ptr<ParticleGroup> new_group(new ParticleGroup(sysdef, p_sel_union));
    
    // return the newly created group
    return new_group;
    }

/*! \param a First particle group
    \param b Second particle group

    \returns A shared pointer to a newly created particle group that contains only the elements present in both \a a and
    \a b
*/
boost::shared_ptr<ParticleGroup> ParticleGroup::groupIntersection(boost::shared_ptr<ParticleGroup> a,
                                                                  boost::shared_ptr<ParticleGroup> b)
    {
    // get a valid system definition
    boost::shared_ptr<SystemDefinition> sysdef;
    if (a->m_sysdef)
        sysdef = a->m_sysdef;
    else if (b->m_sysdef)
        sysdef = b->m_sysdef;

    // check if this is an intersection of two empty groups
    if (! sysdef)
        return boost::shared_ptr<ParticleGroup>(new ParticleGroup());

    // create the intersection of ParticleSelectors
    boost::shared_ptr<ParticleSelector> p_sel_intersection(new ParticleSelectorIntersection(sysdef, a->m_selector, b->m_selector));

    // create the new particle group
    boost::shared_ptr<ParticleGroup> new_group(new ParticleGroup(sysdef, p_sel_intersection));
    
    // return the newly created group
    return new_group;
    }

/*! \param a First particle group
    \param b Second particle group

    \returns A shared pointer to a newly created particle group that contains only the elements present in \a a, and
    not any present in \a b
*/
boost::shared_ptr<ParticleGroup> ParticleGroup::groupDifference(boost::shared_ptr<ParticleGroup> a,
                                                                boost::shared_ptr<ParticleGroup> b)
    {
    // get a valid system definition
    boost::shared_ptr<SystemDefinition> sysdef;
    if (a->m_sysdef)
        sysdef = a->m_sysdef;
    else if (b->m_sysdef)
        sysdef = b->m_sysdef;

    // check if this is a difference of two empty groups
    if (! sysdef)
        return boost::shared_ptr<ParticleGroup>(new ParticleGroup());

    // create the intersection of ParticleSelectors
    boost::shared_ptr<ParticleSelector> p_sel_difference(new ParticleSelectorDifference(sysdef, a->m_selector, b->m_selector));

    // create the new particle group
    boost::shared_ptr<ParticleGroup> new_group(new ParticleGroup(sysdef, p_sel_difference));
    
    // return the newly created group
    return new_group;
    }

/*! \pre m_member_tags has been filled out, listing all particle tags in the group
    \pre memory has been allocated for m_is_member and m_member_idx
    \post m_is_member is updated so that it reflects the current indices of the particles in the group
    \post m_member_idx is updated listing all particle indices belonging to the group, in index order
*/
void ParticleGroup::rebuildIndexList()
    {
    // start by rebuilding the bitset of member indices in the group

    // resize indices array if necessary
    if (m_num_members > m_member_idx.getNumElements())
        {
        unsigned int new_size = m_member_idx.getNumElements() ? m_member_idx.getNumElements() : m_num_members;
        while (m_num_members > new_size) new_size*=2;
        m_member_idx.resize(2*m_member_idx.getNumElements());
        }

#ifdef ENABLE_CUDA
    if (m_pdata->getExecConf()->isCUDAEnabled() )
        {
        rebuildIndexListGPU();
        }
    else
#endif
        {

        // clear the flag array
        ArrayHandle<unsigned char> h_is_member(m_is_member, access_location::host, access_mode::readwrite);
        for (unsigned int idx = 0; idx < m_pdata->getN(); idx ++)
            h_is_member.data[idx] = 0;


        // then loop through every particle in the group and set its bit
            {
            ArrayHandle<unsigned int> h_member_tags(m_member_tags, access_location::host, access_mode::read);
            for (unsigned int member_idx = 0; member_idx < m_num_members; member_idx++)
                {
                unsigned int idx = m_pdata->getGlobalRTag(h_member_tags.data[member_idx]);
                assert(idx < m_pdata->getN());
                h_is_member.data[idx] = 1;
                }
            }

        // then loop through the bitset and add indices to the index list
        ArrayHandle<unsigned int> h_handle(m_member_idx, access_location::host, access_mode::readwrite);
        unsigned int cur_member = 0;
        unsigned int nparticles = m_pdata->getN();
        for (unsigned int idx = 0; idx < nparticles; idx++)
            {
            if (h_is_member.data[idx])
                {
                h_handle.data[cur_member] = idx;
                cur_member++;
                }
            }

        // sanity check, the number of indices added to m_member_idx must be the same as the number of members in the group
        assert(cur_member == m_num_members);
        }
    }

#ifdef ENABLE_CUDA
//! rebuild index list on the GPU
void ParticleGroup::rebuildIndexListGPU()
    {
    // do nothing if we have zero members
    if  (!m_num_members) return;

    ArrayHandle<unsigned char> d_is_member(m_is_member, access_location::device, access_mode::overwrite);
    ArrayHandle<unsigned int> d_member_tags(m_member_tags, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_member_idx(m_member_idx, access_location::device, access_mode::overwrite);
    ArrayHandle<unsigned int> d_global_rtag(m_pdata->getGlobalRTags(), access_location::device, access_mode::read);

    gpu_rebuild_index_list(m_pdata->getN(),
                           m_num_members,
                           d_member_tags.data,
                           d_is_member.data,
                           d_member_idx.data,
                           d_global_rtag.data);
    }
#endif

//! Wrapper class for exposing abstract ParticleSelector base class to python
class ParticleSelectorWrap : public ParticleSelector, public wrapper<ParticleSelector>
    {
    public:
        //! Calls the overridden ParticleSelector::getMemberTags()
        unsigned int getMemberTags(const GPUArray<unsigned int>& member_tags)
            {
            return this->get_override("getMemberTags")(member_tags);
            }
    };

void export_ParticleGroup()
    {
    class_<ParticleGroup, boost::shared_ptr<ParticleGroup>, boost::noncopyable>
            ("ParticleGroup", init< boost::shared_ptr<SystemDefinition>, boost::shared_ptr<ParticleSelector> >())
            .def(init<boost::shared_ptr<SystemDefinition>, const std::vector<unsigned int>& >())
            .def("getNumMembers", &ParticleGroup::getNumMembers)
            .def("getMemberTag", &ParticleGroup::getMemberTag)
            .def("getTotalMass", &ParticleGroup::getTotalMass)
            .def("getCenterOfMass", &ParticleGroup::getCenterOfMass)
            .def("groupUnion", &ParticleGroup::groupUnion)
            .def("groupIntersection", &ParticleGroup::groupIntersection)
            .def("groupDifference", &ParticleGroup::groupDifference)
            ;

    class_<ParticleSelectorWrap, boost::noncopyable>
            ("ParticleSelector")
            ;
    
    class_<ParticleSelectorTag, boost::shared_ptr<ParticleSelectorTag>, bases<ParticleSelector>, boost::noncopyable>
        ("ParticleSelectorTag", init< boost::shared_ptr<SystemDefinition>, unsigned int, unsigned int >())
        ;
        
    class_<ParticleSelectorType, boost::shared_ptr<ParticleSelectorType>, bases<ParticleSelector>, boost::noncopyable>
        ("ParticleSelectorType", init< boost::shared_ptr<SystemDefinition>, unsigned int, unsigned int >())
        ;

    class_<ParticleSelectorRigid, boost::shared_ptr<ParticleSelectorRigid>, bases<ParticleSelector>, boost::noncopyable>
        ("ParticleSelectorRigid", init< boost::shared_ptr<SystemDefinition>, bool >())
        ;    

#if 0
    class_<ParticleSelectorCuboid, boost::shared_ptr<ParticleSelectorCuboid>, bases<ParticleSelector>, boost::noncopyable>
       ("ParticleSelectorCuboid", init< boost::shared_ptr<SystemDefinition>, Scalar3, Scalar3 >())
        ;
#endif
    }

#ifdef WIN32
#pragma warning( pop )
#endif

