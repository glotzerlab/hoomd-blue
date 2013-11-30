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
    \brief Declares BondedGroupData
 */

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __BONDED_GROUP_DATA_H__
#define __BONDED_GROUP_DATA_H__

//! Sentinel value to indicate group is not present on this processor
const unsigned int GROUP_NOT_LOCAL ((unsigned int) 0xffffffff);

#include "GPUVector.h"
#include "ExecutionConfiguration.h"
#include "Profiler.h"

#include <boost/shared_ptr.hpp>
#include <boost/python.hpp>
using namespace boost::python;
#include <boost/bind.hpp>

#include <stack>
#include <string>
#include <sstream>

/*! BondedGroupData is a generic storage class for small particle groups of fixed
 *  size N=2,3,4..., such as bonds, angles or dihedrals, which form part of a molecule.
 *
 *  \tpp group_size Size of groups
 *  \tpp tags_element Compact storage for n group member tags (e.g. uint2,...)
 *  \tpp name Name of element (bond, angle, dihedral, ..)
 */
template<unsigned int group_size, typename tags_element, const char *name>
class BondedGroupData : boost::noncopyable
    {
    public:
        #ifdef ENABLE_MPI
        //! Type for storing per-member ranks
        typedef struct { unsigned int rank[group_size];} ranks_t;
        #endif

        #ifdef ENABLE_MPI
        //! Packed group entry for communication
        typedef struct
            {
            tags_element tags; //!< Member tags
            unsigned int type;          //!< Type of bonded group
            ranks_t ranks;              //!< Current list of member ranks
            } packed_t;
        #endif

        //! Handy structure for passing around and initializing the group data
        struct Snapshot
            {
            //! Default constructor
            Snapshot()
                { }

            //! Constructor
            /*! \param n_groups Number of groups contained in the snapshot
             */
            Snapshot(unsigned int n_groups)
                {
                resize(n_groups);

                // provide default type mapping
                type_mapping.push_back(std::string(name) + "A");
                }

            //! Resize the snapshot
            /*! \param n_groups Number of groups in the snapshot
             */
            void resize(unsigned int n_groups)
                {
                type_id.resize(n_groups);
                groups.resize(n_groups);
                }

            //! Validate the snapshot
            /* \returns true if number of elements in snapshot is consistent
             */
            bool validate() const
                {
                if (! groups.size() == type_id.size()) return false;
                return true;
                }

            std::vector<unsigned int> type_id;             //!< Stores type for each group
            std::vector<tags_element> groups;     //!< Stores the data for each group
            std::vector<std::string> type_mapping;         //!< Names of group types
            };

        //! Constructor for empty BondedGroupData
        BondedGroupData(boost::shared_ptr<ParticleData> pdata,
            unsigned int n_group_types);

        //! Constructor to initialize from a snapshot
        BondedGroupData(boost::shared_ptr<ParticleData> pdata,
            const Snapshot& snapshot);

        virtual ~BondedGroupData();

        //! Initialize from a snapshot
        virtual void initializeFromSnapshot(const Snapshot& snapshot);

        //! Take a snapshot
        virtual void takeSnapshot(Snapshot& snapshot) const;

        //! Get local number of bonded groups
        unsigned int getN() const
            {
            return m_groups.size();
            }

        //! Get global number of bonded groups
        unsigned int getNGlobal() const
            {
            return m_nglobal;
            }

        //! Get the type of a bonded group by tag
        unsigned int getTypeByTag(unsigned int tag) const;

        //! Get the number of group types
        unsigned int getNTypes() const
            {
            return m_type_mapping.size();
            }

        //! Get the type id by type name
        unsigned int getTypeByName(const std::string &type_name) const;

        //! Get the type name by id
        const std::string getNameByType(unsigned int type) const;

        //! Get the members of a bonded group by tag
        const tags_element getMembersByIndex(unsigned int group_idx) const;

        //! Get the members of a bonded group by tag
        const tags_element getMembersByTag(unsigned int tag) const;

        //! Get the members of a bonded group by tag
        unsigned int getTypeByIndex(unsigned int group_idx) const;

        //! Return group table
        const GPUArray<tags_element>& getMembersArray() const
            {
            return m_groups;
            }

        //! Return group table
        const GPUArray<unsigned int>& getTypesArray() const
            {
            return m_group_type;
            }

        //! Return two-dimensional group-by-ptl-index lookup table
        const GPUArray<unsigned int>& getGroupsByParticleArray()
            {
            // rebuild lookup table if necessary
            if (m_groups_dirty)
                {
                rebuildIndexLookupTable();
                m_groups_dirty = false;
                }

            return m_idx_lookup;
            }

        //! Return two-dimensional group-by-ptl-index lookup table
        const Index2D& getGroupsByParticleIndexer()
            {
            // rebuild lookup table if necessary
            if (m_groups_dirty)
                {
                rebuildIndexLookupTable();
                m_groups_dirty = false;
                }

            return m_idx_lookup_indexer;
            }

        //! Return list of number of groups per particle
        const GPUArray<unsigned int>& getNGroupsArray()
            {
            return m_n_groups;
            }

        //! Return list of group tags
        const GPUArray<unsigned int>& getTags()
            {
            return m_group_tag;
            }

        //! Return reverse-lookup table (group tag-> group index)
        const GPUArray<unsigned int>& getRTags() const
            {
            return m_group_rtag;
            }

        #ifdef ENABLE_MPI
        //! Add new groups to local processor
        /*! \param in Buffer containing the group entries to be added
         *
         *  \pre The elements of the input buffer need to contain up-to-date rank information
         *  There must not be any duplicates in the input buffer of groups that already exist locally,
         *  a condition which can be assured by checking the ranks field of the group entry prior
         *  to adding/communicating.
         *
         *  \post The local group table contains the unpacked groups in addition
         */
        void addGroups(const std::vector<packed_t>& in);

        //! Pack groups into buffer which have non-local particle members and remove left over groups
        /*! Groups packed into the output buffer are those for which any particle rtag is NOT_LOCAL.
         * Any groups for which ALL particle rtags are NOT_LOCAL are removed.
         *
         * \post the out array is resized as necessary
         */
        void removeGroups(std::vector<packed_t>& out);

        #ifdef ENABLE_CUDA
        //! Add new groups to local processor
        /*! \param in Buffer containing the group entries to be added
         *
         *  \pre The elements of the input buffer need to contain up-to-date information on member ranks.
         *  There must not be any duplicates in the input buffer of groups that already exist locally,
         *  a condition that can be assured by checking the ranks field of the group for the local/destination rank
         *  entry prior to adding/communicating.
         *
         *  \post The local group table contains the unpacked groups in addition
         */
        void addGroupsGPU(GPUArray<packed_t>& in);

        //! Pack groups into buffer which have non-local particle members and remove left over groups
        void removeGroupsGPU(GPUArray<packed_t>& out);
        #endif
        #endif

        //! Add a single bonded group on all processors
        /*! \param type_id Type of group to add
         * \param member_tags All particle tag that are members of this bonded group
         * \returns Tag of newly added bond
         */
        unsigned int addBondedGroup(unsigned int type_id, tags_element member_tags);

        //! Remove a single bonded group from all processors
        /*! \param tag Tag of bonded group to remove
         */
        void removeBondedGroup(unsigned int group_tag);

        //! Set the profiler
        /*! \param prof The profiler
         */
        void setProfiler(boost::shared_ptr<Profiler> prof)
            {
            m_prof = prof;
            }

    protected:
        //! A union to access the data elements' components by index
        typedef union
            {
            tags_element data;
            unsigned int tags[group_size];
            } tags_t;

        //! Helper function to transfer bonded groups connected to a single particle
        /*! \param tag Tag of particle that moves between domains
            \param old_rank Old MPI rank for particle
            \param new_rank New MPI rank
         */
        void moveParticleBonds(unsigned int tag, unsigned int old_rank, unsigned int new_rank);

        boost::shared_ptr<const ExecutionConfiguration> m_exec_conf;  //!< Execution configuration for CUDA context
        boost::shared_ptr<ParticleData> m_pdata;        //!< Particle Data these bonds belong to

        GPUVector<tags_element> m_groups;               //!< List of groups
        GPUVector<unsigned int> m_group_type;           //!< List of group types
        GPUVector<unsigned int> m_group_tag;         //!< List of group tags
        GPUVector<unsigned int> m_group_rtag;        //!< Global reverse-lookup table for group tags
        GPUVector<unsigned int> m_idx_lookup;        //!< Lookup table from particle index to group indices (2D)
        Index2D m_idx_lookup_indexer;                //!< Indexer for 2D lookup table
        GPUVector<unsigned int> m_n_groups;          //!< Number of entries in lookup table per particle
        std::vector<std::string> m_type_mapping;     //!< Mapping of types of bonded groups

        #ifdef ENABLE_MPI
        GPUVector<unsigned int> m_group_ranks;       //!< 2D list of group member ranks
        #endif

        /* alternate (stand-by) arrays for swapping in reordered groups */
        GPUVector<tags_element> m_groups_alt;           //!< List of groups (swap-in)
        GPUVector<unsigned int> m_group_type_alt;       //!< List of group types (swap-in)
        GPUVector<unsigned int> m_group_tag_alt;     //!< List of group tags (swap-in)
        #ifdef ENABLE_MPI
        GPUVector<unsigned int> m_group_ranks_alt;   //!< 2D list of group member ranks (swap-in)
        #endif

        unsigned int m_nglobal;                      //!< Global number of groups
        std::stack<unsigned int> m_recycled_tags;    //!< Global tags of removed groups
        boost::shared_ptr<Profiler> m_prof;          //!< Profiler

        //! Initialize internal memory
        void initialize();

    private:
        bool m_groups_dirty;                         //!< Is it necessary to rebuild the lookup-by-index table?
        boost::signals2::connection m_sort_connection;   //!< Connection to the resort signal from ParticleData

        //! Set a flag to trigger rebuild of index table
        void setDirty()
            {
            m_groups_dirty = true;
            }

        //! Helper function to rebuild lookup by index table
        void rebuildIndexLookupTable();

        #ifdef ENABLE_CUDA
        //! Helper function to rebuild lookup by index table on the GPU
        void rebuildIndexLookupTableGPU();
        #endif
    };

//! Exports BondData to python
template<class T>
void export_BondedGroupData();

/*
 * Implementation of BondedGroupData methods
 */

/*! \param exec_conf Execution configuration
    \param pdata The particle data to associate with
    \param n_group_types Number of bonded group types to initialize
 */
template<unsigned int group_size,typename tags_element, const char *name>
BondedGroupData<group_size, tags_element, name>::BondedGroupData(
    boost::shared_ptr<ParticleData> pdata,
    unsigned int n_group_types)
    : m_exec_conf(pdata->getExecConf()), m_pdata(pdata), m_nglobal(0), m_groups_dirty(true)
    {
    m_exec_conf->msg->notice(5) << "Constructing BondedGroupData (" << name<< ") " << endl;

    // connect to particle sort signal
    m_sort_connection = m_pdata->connectParticleSort(boost::bind(&BondedGroupData<group_size, tags_element, name>::setDirty, this));

    // offer a default type mapping
    for (unsigned int i = 0; i < n_group_types; i++)
        {
        char suffix[2];
        suffix[0] = 'A' + i;
        suffix[1] = '\0';

        std::string type_name = std::string(name) + std::string(suffix);
        m_type_mapping.push_back(type_name);
        }
    }

/*! \param exec_conf Execution configuration
    \param pdata The particle data to associate with
    \param snapshot Snapshot to initialize from
 */
template<unsigned int group_size,typename tags_element, const char *name>
BondedGroupData<group_size, tags_element, name>::BondedGroupData(
    boost::shared_ptr<ParticleData> pdata,
    const Snapshot& snapshot)
    : m_exec_conf(pdata->getExecConf()), m_pdata(pdata), m_nglobal(0), m_groups_dirty(true)
    {
    m_exec_conf->msg->notice(5) << "Constructing BondedGroupData (" << name << ") " << endl;

    // connect to particle sort signal
    m_sort_connection = m_pdata->connectParticleSort(boost::bind(&BondedGroupData<group_size, tags_element, name>::setDirty,
        this));

    // initialize data structures
    initialize();

    // initialize from snapshot
    initializeFromSnapshot(snapshot);
    }

//! Destructor
template<unsigned int group_size,typename tags_element, const char *name>
BondedGroupData<group_size, tags_element, name>::~BondedGroupData()
    {
    m_sort_connection.disconnect();
    }

template<unsigned int group_size,typename tags_element, const char *name>
void BondedGroupData<group_size, tags_element, name>::initialize()
    {
    // allocate arrays
    GPUVector<tags_element> groups(m_exec_conf);
    m_groups.swap(groups);

    GPUVector<unsigned int> type_id(m_exec_conf);
    m_group_type.swap(type_id);

    GPUVector<unsigned int> group_tag(m_exec_conf);
    m_group_tag.swap(group_tag);

    GPUVector<unsigned int> group_rtag(m_exec_conf);
    m_group_rtag.swap(group_rtag);

    // Lookup table per particle index
    GPUVector<unsigned int> idx_lookup(m_exec_conf);
    m_idx_lookup.swap(idx_lookup);

    GPUVector<unsigned int> n_groups(m_exec_conf);
    m_n_groups.swap(n_groups);

    #ifdef ENABLE_MPI
    GPUVector<unsigned int> group_ranks(m_exec_conf);
    m_group_ranks.swap(group_ranks);
    #endif

    // allocate stand-by arrays
    GPUVector<unsigned int> type_id_alt(m_exec_conf);
    m_group_type_alt.swap(type_id_alt);

    GPUVector<unsigned int> group_tag_alt(m_exec_conf);
    m_group_tag_alt.swap(group_tag_alt);

    GPUVector<tags_element> groups_alt(m_exec_conf);
    m_groups_alt.swap(groups_alt);

    #ifdef ENABLE_MPI
    GPUVector<unsigned int> group_ranks_alt(m_exec_conf);
    m_group_ranks_alt.swap(group_ranks_alt);
    #endif
    }

//! Initialize from a snapshot
template<unsigned int group_size,typename tags_element, const char *name>
void BondedGroupData<group_size, tags_element, name>::initializeFromSnapshot(const Snapshot& snapshot)
    {
    // check that all fields in the snapshot have correct length
    if (m_exec_conf->getRank() == 0 && ! snapshot.validate())
        {
        m_exec_conf->msg->error() << "init.*: invalid " << name << " data snapshot."
                                << std::endl << std::endl;
        throw std::runtime_error(std::string("Error initializing ") + name + std::string(" data."));
        }

    // initialize data structures
    initialize();

    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        // ...
        m_groups_dirty = true;
        }
    else
    #endif
        {
        m_type_mapping = snapshot.type_mapping;

        for (unsigned group_idx = 0; group_idx < snapshot.groups.size(); group_idx++)
            addBondedGroup(snapshot.type_id[group_idx], snapshot.groups[group_idx]);
        }
    }

/*! \param type_id Type of bonded group to add
    \param member_tags Particle members of group
 */
template<unsigned int group_size,typename tags_element, const char *name>
unsigned int BondedGroupData<group_size, tags_element, name>::addBondedGroup(
    unsigned int type_id,
    tags_element member_tags)
    {
    tags_t t;
    t.data = member_tags;

    // check for some silly errors a user could make
    for (unsigned int i = 0; i < group_size; ++i)
        if (t.tags[i] >= m_pdata->getNGlobal())
            {
            std::ostringstream oss;
            oss << "Particle tag out of bounds when attempting to add " << name << ": ";
            for (unsigned int j = 0; j < group_size; ++j)
                oss << t.tags[j] << ((j != group_size - 1) ? "," : "");
            oss << std::endl;
            m_exec_conf->msg->error() << oss.str();
            throw runtime_error(std::string("Error adding ") + name);
            }

    for (unsigned int i = 0; i < group_size; ++i)
        for (unsigned int j = 0; j < group_size; ++j)
            if (i != j && t.tags[i] == t.tags[j])
                {
                std::ostringstream oss;
                oss << "The same particle can only occur once in a " << name << ": ";
                for (unsigned int k = 0; k < group_size; ++k)
                    oss << t.tags[k] << ((k != group_size - 1) ? "," : "");
                oss << std::endl;
                m_exec_conf->msg->error() << oss.str();
                throw runtime_error(std::string("Error adding ") + name);
                }

    if (type_id >= m_type_mapping.size())
        {
        m_exec_conf->msg->error() << "Invalid " << name << " type! " << type_id << ", the  number of types is "
            << m_type_mapping.size() << std::endl;
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
            if (m_pdata->isParticleLocal(t.tags[i]))
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
        if (is_local) m_group_rtag[tag] = getN();
        }
    else
        {
        // Otherwise, generate a new tag
        tag = getNGlobal();

        // add new reverse-lookup tag
        assert(m_bond_rtag.size() == m_bonds.size());
        if (is_local)
            m_group_rtag.push_back(getN());
        else
            m_group_rtag.push_back(GROUP_NOT_LOCAL);
        }

    assert(tag <= m_recycled_tags.size() + getNGlobal());

    m_groups.push_back(member_tags);
    m_group_type.push_back(type_id);

    m_group_tag.push_back(tag);

    // increment number of bonded groups
    m_nglobal++;

    m_groups_dirty = true;

    return tag;
    }

/*! \param tag Tag of bonded group
 * \returns Type id of bonded group
 */
template<unsigned int group_size,typename tags_element, const char *name>
unsigned int BondedGroupData<group_size, tags_element, name>::getTypeByTag(unsigned int tag) const
    {
    // Find position of bond in bonds list
    unsigned int group_idx = m_group_rtag[tag];

    unsigned int type;

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
            m_exec_conf->msg->error() << "Trying to get type of " << name << " " << tag
                 << " which does not exist!" << endl;
            throw runtime_error(std::string("Error getting ") + name);
            }

        if (rank == (int)my_rank)
            type = m_group_type[group_idx];

        bcast(type, rank, m_exec_conf->getMPICommunicator());
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
        }

    return type;
    }

/*! \param tag Tag of bonded group
 * \returns Member tags of bonded group
 */
template<unsigned int group_size,typename tags_element, const char *name>
const tags_element BondedGroupData<group_size, tags_element, name>::getMembersByTag(unsigned int tag) const
    {
    // Find position of bond in bonds list
    unsigned int group_idx = m_group_rtag[tag];

    tags_element g;

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
            m_exec_conf->msg->error() << "Trying to get members of " << name << " " << tag
                 << " which does not exist!" << endl;
            throw runtime_error(std::string("Error getting ") + name);
            }

        if (rank == (int)my_rank)
            g = m_groups[group_idx];

        bcast(g, rank, m_exec_conf->getMPICommunicator());
        }
    #endif
    else
        {
        if (group_idx == GROUP_NOT_LOCAL)
            {
            m_exec_conf->msg->error() << "Trying to get members of " << name << " " << tag
                 << " which does not exist!" << endl;
            throw runtime_error(std::string("Error getting ") + name);
            }

            g = m_groups[group_idx];
        }

    return g;
    }

/*! \param idx Tag of bonded group
 * \return Member tags of bonded group
 */
template<unsigned int group_size,typename tags_element, const char *name>
unsigned int BondedGroupData<group_size, tags_element, name>::getTypeByIndex(unsigned int group_idx) const
    {
    assert (group_idx < getN());
    return m_group_type[group_idx];
    }

/*! \param idx Tag of bonded group
 * \return Type of bonded group
 */
template<unsigned int group_size,typename tags_element, const char *name>
const tags_element BondedGroupData<group_size, tags_element, name>::getMembersByIndex(unsigned int group_idx) const
    {
    assert (group_idx < getN());
    return m_groups[group_idx];
    }

/*! \param tag Tag of bonded group to remove
 */
template<unsigned int group_size,typename tags_element, const char *name>
void BondedGroupData<group_size, tags_element, name>::removeBondedGroup(unsigned int tag)
    {
    // sanity check
    if (tag >= getNGlobal())
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

        assert(res <= group_size);
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
            m_groups[id] = (tags_element) m_groups[size-1];
            m_group_type[id] = (unsigned int) m_group_type[size-1];
            unsigned int last_tag = m_group_tag[size-1];
            m_group_rtag[last_tag] = id;
            m_group_tag[id] = last_tag;
            }

        // delete last element
        m_groups.pop_back();
        m_group_type.pop_back();
        m_group_tag.pop_back();
        }

    // maintain a stack of deleted group tags for future recycling
    m_recycled_tags.push(tag);
    m_nglobal--;

    m_groups_dirty = true;
    }

/*! \param name Type name
 */
template<unsigned int group_size,typename tags_element, const char *name>
unsigned int BondedGroupData<group_size, tags_element, name>::getTypeByName(const std::string &type_name) const
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

template<unsigned int group_size,typename tags_element, const char *name>
const std::string BondedGroupData<group_size, tags_element, name>::getNameByType(unsigned int type) const
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

template<unsigned int group_size,typename tags_element, const char *name>
void BondedGroupData<group_size, tags_element, name>::rebuildIndexLookupTable()
    {
    #ifdef ENABLE_CUDA
    //if (m_exec_conf->isCUDAEnabled())
    //    rebuildIndexLookupTableGPU();
    //else
    #endif
        {
        if (m_prof) m_prof->push("update " + std::string(name) + " table");

        ArrayHandle< unsigned int > h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

        m_n_groups.resize(m_pdata->getN());

        unsigned int num_groups_max = 0;
            {
            ArrayHandle<unsigned int> h_n_groups(m_n_groups, access_location::host, access_mode::overwrite);

            // count the number of bonds per particle
            // start by initializing the n_bonds values to 0
            memset(h_n_groups.data, 0, sizeof(unsigned int) * m_pdata->getN());

            unsigned int N = m_pdata->getN();

            // loop through the particles and count the number of bonds based on each particle index
            for (unsigned int cur_group = 0; cur_group < getN(); cur_group++)
                {
                tags_t t;
                tags_element g = m_groups[cur_group];
                t.data = g;
                for (unsigned int i = 0; i < group_size; ++i)
                    {
                    unsigned int tag = t.tags[i];
                    unsigned int idx = h_rtag.data[tag];

                    // count only local bond members
                    if (idx < N) h_n_groups.data[idx]++;
                    }
                }

            // find the maximum number of bonds
            for (unsigned int i = 0; i < N; i++)
                if (h_n_groups.data[i] > num_groups_max)
                    num_groups_max = h_n_groups.data[i];
            }

        // resize lookup table
        m_idx_lookup_indexer = Index2D(m_pdata->getN(), num_groups_max);
        m_idx_lookup.resize(m_idx_lookup_indexer.getNumElements());

            {
            ArrayHandle<unsigned int> h_n_groups(m_n_groups, access_location::host, access_mode::overwrite);
            ArrayHandle<unsigned int> h_idx_lookup(m_idx_lookup, access_location::host, access_mode::overwrite);

            // now, update the actual table
            // zero the number of bonds counter (again)
            memset(h_n_groups.data, 0, sizeof(unsigned int) * m_pdata->getN());

            // loop through all group and add them to each column in the list
            unsigned int N = m_pdata->getN();

            for (unsigned int cur_group = 0; cur_group < getN(); cur_group++)
                {
                tags_t t;
                tags_element g = m_groups[cur_group];
                t.data = g;
                for (unsigned int i = 0; i < group_size; ++i)
                    {
                    unsigned int tag = t.tags[i];
                    unsigned int idx = h_rtag.data[tag];

                    // get the number of bonds for each particle
                    // add the new bonds to the table
                    if (idx < N)
                        {
                        unsigned int num = h_n_groups.data[idx]++;
                        h_idx_lookup.data[m_idx_lookup_indexer(idx, num)] = cur_group;
                        }
                    }
                }
            }

        if (m_prof) m_prof->pop();
        }
    }

/*! \param snapshot Snapshot that will contain the group data
 *
 *  Data in the snapshot is in tag order, where non-existant tags are skipped
 */
template<unsigned int group_size,typename tags_element, const char *name>
void BondedGroupData<group_size, tags_element, name>::takeSnapshot(Snapshot& snapshot) const
    {
    // allocate memory in snapshot
    snapshot.resize(getNGlobal());

    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        //..
        }
    else
    #endif
        {
        std::map<unsigned int, unsigned int> rtag_map;

        assert(getN() == getNGlobal());
        for (unsigned int group_idx = 0; group_idx < getN(); group_idx++)
            {
            unsigned int tag = m_group_tag[group_idx];
            assert(m_group_rtag[tag] == group_idx);

            rtag_map.insert(std::pair<unsigned int,unsigned int>(tag, group_idx));
            }
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

template<class T>
void export_BondedGroupData(std::string name)
    {
    scope outer = class_<T, boost::shared_ptr<T> , boost::noncopyable>(name.c_str(),
        init<boost::shared_ptr<ParticleData>, unsigned int>())
        .def(init<boost::shared_ptr<ParticleData>, const typename T::Snapshot& >())
        .def("initializeFromSnapshot", &T::initializeFromSnapshot)
        .def("takeSnapshot", &T::takeSnapshot)
        .def("getN", &T::getN)
        .def("getNGlobal", &T::getNGlobal)
        .def("getNTypes", &T::getNTypes)
        .def("getTypeByTag", &T::getTypeByTag)
        .def("getTypeByName", &T::getTypeByName)
        .def("getNameByType", &T::getNameByType)
        .def("addBondedGroup", &T::addBondedGroup)
        .def("removeBondedGroup", &T::removeBondedGroup)
        .def("setProfiler", &T::setProfiler)
        ;

    typedef typename T::Snapshot Snapshot;
    class_<Snapshot, boost::shared_ptr<Snapshot> >
        ("Snapshot", init<unsigned int>())
        .def_readwrite("groups", &Snapshot::groups)
        .def_readwrite("type_id", &Snapshot::type_id)
        .def_readwrite("type_mapping", &Snapshot::type_mapping)
        ;
    }
#endif
