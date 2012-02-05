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

void ParticleSelectorGlobalTagList::rebuildTagsList()
    {
    // reset tags list
    m_member_tags.clear();

    ArrayHandle<unsigned int> h_tag(m_pdata->getRTags(), access_location::host, access_mode::read);

    std::vector<unsigned int>::const_iterator it;

    for (it = m_global_member_tags.begin(); it != m_global_member_tags.end(); it++)
        {
        if (m_pdata->isLocal(*it))
            {
            // if the particle is present in the local simulation box, add its tag to the local group members
            unsigned int idx = m_pdata->getGlobalRTag(*it);
            m_member_tags.push_back(h_tag.data[idx]);
            }
        }
    }

bool ParticleSelectorCuboid::isSelected(unsigned int tag) const
    {
    assert(tag < m_pdata->getN());

    // identify the index of the current particle tag
    Scalar3 pos = m_pdata->getPosition(tag);
    
    // see if it matches the criteria
    bool result = (m_min.x <= pos.x && pos.x < m_max.x &&
                   m_min.y <= pos.y && pos.y < m_max.y &&
                   m_min.z <= pos.z && pos.z < m_max.z);
    
    return result;
    }
#endif


//////////////////////////////////////////////////////////////////////////////
// ParticleSelectorGlobalTagList

//! Constructor
ParticleSelectorGlobalTagList::ParticleSelectorGlobalTagList(boost::shared_ptr<SystemDefinition> sysdef, const std::vector<unsigned int>& global_tag_list)
    : ParticleSelector(), m_sysdef(sysdef), m_pdata(sysdef->getParticleData())
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

    ArrayHandle<unsigned int> h_tag(m_pdata->getRTags(), access_location::host, access_mode::read);

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
// ParticleSelectorUnion

//! constructor
/*! We construct the union of two ParticleSelectors by storing copies of them in class member variables (instead
    of references) and taking the union of their member tags. In this way cyclic references between ParticleSelectors are prevented.
 */
ParticleSelectorUnion::ParticleSelectorUnion(boost::shared_ptr<ParticleSelector> a, boost::shared_ptr<ParticleSelector> b)
    : ParticleSelector(),
      m_selector_a(a),
      m_selector_b(b)
    {
    }

//! rebuild internal list of included tags
unsigned int ParticleSelectorUnion::getMemberTags(const GPUArray<unsigned int>& member_tags)
    {
#if 0
    // clear list of members
    this->m_member_tags.clear();

    // first rebuild the tag lists of the arguments
    m_selector_a->rebuildTagsList();
    m_selector_b->rebuildTagsList();

    // get the tag lists
    std::vector<unsigned int>& member_tags_a = m_selector_a->getMemberTags();
    std::vector<unsigned int>& member_tags_b = m_selector_b->getMemberTags();

    // make the union
    insert_iterator< vector<unsigned int> > ii(m_member_tags, this->m_member_tags.begin());
    set_union(member_tags_a.begin(), member_tags_a.end(), member_tags_a.begin(), member_tags_b.end(), ii);
#endif
    }


//////////////////////////////////////////////////////////////////////////////
// ParticleSelectorIntersection

//! constructor
/*! We construct the intersection of two ParticleSelectors by storing copies of them in class member variables (instead
    of references) and taking the intersection of their member tags. In this way cyclic references between ParticleSelectors are prevented.
 */
ParticleSelectorIntersection::ParticleSelectorIntersection(boost::shared_ptr<ParticleSelector> a, boost::shared_ptr<ParticleSelector> b)
    : ParticleSelector(),
      m_selector_a(a),
      m_selector_b(b)
    {
    }

//! rebuild internal list of included tags
unsigned int ParticleSelectorIntersection::getMemberTags(const GPUArray<unsigned int> & member_tags)
    {
#if 0
    // clear list of members
    this->m_member_tags.clear();

    // first rebuild the tag lists of the arguments
    m_selector_a->rebuildTagsList();
    m_selector_b->rebuildTagsList();

    // get the tag lists
    std::vector<unsigned int>& member_tags_a = m_selector_a->getMemberTags();
    std::vector<unsigned int>& member_tags_b = m_selector_b->getMemberTags();

    // make the intersection
    insert_iterator< vector<unsigned int> > ii(m_member_tags, this->m_member_tags.begin());
    set_intersection(member_tags_a.begin(), member_tags_a.end(), member_tags_a.begin(), member_tags_b.end(), ii);
#endif
    }

//////////////////////////////////////////////////////////////////////////////
// ParticleSelectorDifference

//! constructor
/*! We construct the union of two ParticleSelectors by storing copies of them in class member variables (instead
    of references) and taking the difference of their member tags. In this way cyclic references between ParticleSelectors are prevented.
 */
ParticleSelectorDifference::ParticleSelectorDifference(boost::shared_ptr<ParticleSelector> a, boost::shared_ptr<ParticleSelector> b)
    : ParticleSelector(),
      m_selector_a(a),
      m_selector_b(b)
    {
    }

//! rebuild internal list of included tags
unsigned int ParticleSelectorDifference::getMemberTags(const GPUArray<unsigned int> & member_tags)
    {
#if 0
    // clear list of members
    this->m_member_tags.clear();

    // first rebuild the tag lists of the arguments
    m_selector_a->rebuildTagsList();
    m_selector_b->rebuildTagsList();

    // get the tag lists
    std::vector<unsigned int>& member_tags_a = m_selector_a->getMemberTags();
    std::vector<unsigned int>& member_tags_b = m_selector_b->getMemberTags();

    // make the difference
    insert_iterator< vector<unsigned int> > ii(m_member_tags, this->m_member_tags.begin());
    set_difference(member_tags_a.begin(), member_tags_a.end(), member_tags_a.begin(), member_tags_b.end(), ii);
#endif
    }

//////////////////////////////////////////////////////////////////////////////
// ParticleGroup
/*! \param sysdef System definition to build the group from
    \param selector ParticleSelector used to choose the group members

    Particles where criteria falls within the range [min,max] (inclusive) are added to the group.
*/
ParticleGroup::ParticleGroup(boost::shared_ptr<SystemDefinition> sysdef, boost::shared_ptr<ParticleSelector> selector)
    : m_sysdef(sysdef),
      m_pdata(sysdef->getParticleData()),
      m_selector(selector)
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
      m_selector(boost::shared_ptr<ParticleSelectorGlobalTagList>(new ParticleSelectorGlobalTagList(sysdef, global_tag_list)))
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

    // connect the rebuildIndexList method to be called whenever the particles are sorted
    m_sort_connection = m_pdata->connectParticleSort(bind(&ParticleGroup::rebuildIndexList, this));

    // connect the rebuildTagList() method to be called whenever particles are inserted or deleted
    m_particle_num_change_connection = m_pdata->connectParticleNumberChange(bind(&ParticleGroup::rebuildTagList, this));

    //! connect reallocate() method to maximum particle number change signal
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
*/
Scalar3 ParticleGroup::getCenterOfMass() const
    {
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
    boost::shared_ptr<ParticleSelector> p_sel_union(new ParticleSelectorUnion(a->m_selector, b->m_selector));

    // create the new particle group
    boost::shared_ptr<ParticleGroup> new_group(new ParticleGroup(a->m_sysdef, p_sel_union));
    
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
    // create the intersection of ParticleSelectors
    boost::shared_ptr<ParticleSelector> p_sel_intersection(new ParticleSelectorIntersection(a->m_selector, b->m_selector));

    // create the new particle group
    boost::shared_ptr<ParticleGroup> new_group(new ParticleGroup(a->m_sysdef, p_sel_intersection));
    
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
    // create the intersection of ParticleSelectors
    boost::shared_ptr<ParticleSelector> p_sel_difference(new ParticleSelectorDifference(a->m_selector, b->m_selector));

    // create the new particle group
    boost::shared_ptr<ParticleGroup> new_group(new ParticleGroup(a->m_sysdef, p_sel_difference));
    
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
    while (m_num_members > m_member_idx.getNumElements())
        m_member_idx.resize(2*m_member_idx.getNumElements());

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
            if (isMember(idx))
                {
                h_handle.data[cur_member] = idx;
                cur_member++;
                }
            }

        // sanity check, the number of indices added to m_member_idx must be the same as the number of members in the group
        assert(cur_member == m_num_elements);
        }
    }

#ifdef ENABLE_CUDA
//! rebuild index list on the GPU
void ParticleGroup::rebuildIndexListGPU()
    {
    ArrayHandle<unsigned char> d_is_member(m_is_member, access_location::device, access_mode::readwrite);
    ArrayHandle<unsigned int> d_member_tags(m_member_tags, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_member_idx(m_member_idx, access_location::device, access_mode::readwrite);
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

