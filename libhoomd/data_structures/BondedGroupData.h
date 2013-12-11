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
#include "Index1D.h"

#include <boost/signals2.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/python.hpp>
using namespace boost::python;
#include <boost/bind.hpp>

#include <stack>
#include <string>
#include <sstream>

#ifdef ENABLE_CUDA
#include "cached_allocator.h"
#endif

//! Forward declarations
class ParticleData;

//! Storage data type for group members
/*! We use a union to emphasize it that can contain either particle
 * tags or particle indices or other information */
template<unsigned int group_size>
union group_storage
    {
    unsigned int tag[group_size];
    unsigned int idx[group_size];
    };

#ifdef ENABLE_MPI
//! Packed group entry for communication
template<unsigned int group_size>
struct packed_storage
    {
    group_storage<group_size> tags;  //!< Member tags
    unsigned int type;               //!< Type of bonded group
    unsigned int group_tag;          //!< Tag of this group
    group_storage<group_size> ranks; //!< Current list of member ranks
    };
#endif

#ifdef ENABLE_MPI
BOOST_CLASS_IMPLEMENTATION(group_storage<2>,boost::serialization::object_serializable)
namespace boost
   {
   //! Serialization functions for group data types
   namespace serialization
        {
        //! Serialization of group_storage<2> (bonds)
        template<class Archive>
        void serialize(Archive & ar, group_storage<2> & s, const unsigned int version)
            {
            ar & s.tag[0];
            ar & s.tag[1];
            }
        }
    }
#endif

/*! BondedGroupData is a generic storage class for small particle groups of fixed
 *  size N=2,3,4..., such as bonds, angles or dihedrals, which form part of a molecule.
 *
 *  \tpp group_size Size of groups
 *  \tpp group_t Compact storage for n group member tags or indices, e.g. uint2,...
 *  \tpp name Name of element, i.e. bond, angle, dihedral, ..
 */
template<unsigned int group_size, const char *name>
class BondedGroupData : boost::noncopyable
    {
    public:
        //! Group size
        enum { size = group_size } Enum;

        //! Group data element type
        typedef union group_storage<group_size> group_t;

        #ifdef ENABLE_MPI
        //! Type for storing per-member ranks
        typedef group_t ranks_t;
        typedef packed_storage<group_size> packed_t;
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
            std::vector<group_t> groups;     //!< Stores the data for each group
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

        //! Return name of this template
        static std::string getName()
            {
            return std::string(name);
            }

        //! Get the type id by type name
        unsigned int getTypeByName(const std::string &type_name) const;

        //! Get the type name by id
        const std::string getNameByType(unsigned int type) const;

        //! Get the members of a bonded group by tag
        const group_t getMembersByIndex(unsigned int group_idx) const;

        //! Get the members of a bonded group by tag
        const group_t getMembersByTag(unsigned int tag) const;

        //! Get the members of a bonded group by tag
        unsigned int getTypeByIndex(unsigned int group_idx) const;

        /*
         * Access to data structures
         */

        //! Return group table (const)
        const GPUVector<group_t>& getMembersArray() const
            {
            return m_groups;
            }

        //! Return group table (const)
        const GPUVector<unsigned int>& getTypesArray() const
            {
            return m_group_type;
            }

        //! Return list of group tags (const)
        const GPUVector<unsigned int>& getTags() const
            {
            return m_group_tag;
            }

        //! Return reverse-lookup table (group tag-> group index) (const)
        const GPUVector<unsigned int>& getRTags() const
            {
            return m_group_rtag;
            }

        #ifdef ENABLE_MPI
        //! Return auxillary array of member particle ranks (const)
        const GPUVector<ranks_t>& getRanksArray() const
            {
            return m_group_ranks;
            }
        #endif

        /* 
         * Alternate arrays used for reordering data
         * \sa ParticleData
         *
         * \note The only method to modify the size of the group table is by
         * swapping in a new version of the group data.
         *
         * The alternate arrays are returned as non-const versions
         * to enable resizing the underlying GPUVectors before swapping.
         */

        //! Return group table (swap-in)
        GPUVector<group_t>& getAltMembersArray()
            {
            // resize to size of primary groups array
            m_groups_alt.resize(m_groups.size());
            return m_groups_alt;
            }

        //! Return group table (swap-in)
        GPUVector<unsigned int>& getAltTypesArray()
            {
            // resize to size of primary group types array
            m_group_type_alt.resize(m_group_type.size());
            return m_group_type_alt;
            }

        //! Return list of group tags (swap-in)
        GPUVector<unsigned int>& getAltTags()
            {
            // resize to size of primary group tags array
            m_group_tag_alt.resize(m_group_tag.size());
            return m_group_tag_alt;
            }

        #ifdef ENABLE_MPI
        //! Return auxillary array of member particle ranks
        GPUVector<ranks_t>& getAltRanksArray()
            {
            m_group_ranks_alt.resize(m_group_ranks.size());
            return m_group_ranks_alt;
            }
        #endif

        //! Swap group member arrays
        void swapMemberArrays()
            {
            assert(!m_groups_alt.isNull());
            m_groups.swap(m_groups_alt);
            }

        //! Swap group type arrays
        void swapTypeArrays()
            {
            assert(!m_group_type_alt.isNull());
            m_group_type.swap(m_group_type_alt);
            }

        //! Swap group tag arrays
        void swapTagArrrays()
            {
            assert(!m_group_tag_alt.isNull());
            m_group_tag.swap(m_group_tag_alt);
            }

        #ifdef ENABLE_MPI
        //! Swap group ranks arrays
        void swapRankArrays()
            {
            assert(!m_group_ranks_alt.isNull());
            m_group_ranks.swap(m_group_ranks_alt);
            }
        #endif

        /* 
         * GPU group table
         */

        //! Return GPU bonded groups list
        const GPUArray<group_t>& getGPUTable()
            {
            // rebuild lookup table if necessary
            if (m_groups_dirty)
                {
                rebuildGPUTable();
                m_groups_dirty = false;
                }

            return m_gpu_table;
            }

        //! Return two-dimensional group-by-ptl-index lookup table
        const Index2D& getGPUTableIndexer()
            {
            // rebuild lookup table if necessary
            if (m_groups_dirty)
                {
                rebuildGPUTable();
                m_groups_dirty = false;
                }

            return m_gpu_table_indexer;
            }

        //! Return list of number of groups per particle
        const GPUArray<unsigned int>& getNGroupsArray() const
            {
            return m_n_groups;
            }

        /*
         * add/remove groups globally
         */

        //! Add a single bonded group on all processors
        /*! \param type_id Type of group to add
         * \param member_tags All particle tag that are members of this bonded group
         * \returns Tag of newly added bond
         */
        unsigned int addBondedGroup(unsigned int type_id, group_t member_tags);

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

        //! Connects a function to be called every time the global number of bonded groups changes
        boost::signals2::connection connectGroupNumChange(
            const boost::function<void ()> &func)
            {
            return m_group_num_change_signal.connect(func);
            }

    protected:
        #ifdef ENABLE_MPI
        //! Helper function to transfer bonded groups connected to a single particle
        /*! \param tag Tag of particle that moves between domains
            \param old_rank Old MPI rank for particle
            \param new_rank New MPI rank
         */
        void moveParticleBonds(unsigned int tag, unsigned int old_rank, unsigned int new_rank);
        #endif

        boost::shared_ptr<const ExecutionConfiguration> m_exec_conf;  //!< Execution configuration for CUDA context
        boost::shared_ptr<ParticleData> m_pdata;        //!< Particle Data these bonds belong to

        GPUVector<group_t> m_groups;            //!< List of groups
        GPUVector<unsigned int> m_group_type;        //!< List of group types
        GPUVector<unsigned int> m_group_tag;         //!< List of group tags
        GPUVector<unsigned int> m_group_rtag;        //!< Global reverse-lookup table for group tags
        GPUVector<group_t> m_gpu_table;              //!< Storage for groups by particle index for access on the GPU
        Index2D m_gpu_table_indexer;                 //!< Indexer for GPU table
        GPUVector<unsigned int> m_n_groups;          //!< Number of entries in lookup table per particle
        std::vector<std::string> m_type_mapping;     //!< Mapping of types of bonded groups

        #ifdef ENABLE_MPI
        GPUVector<ranks_t> m_group_ranks;       //!< 2D list of group member ranks
        #endif

        /* alternate (stand-by) arrays for swapping in reordered groups */
        GPUVector<group_t> m_groups_alt;           //!< List of groups (swap-in)
        GPUVector<unsigned int> m_group_type_alt;       //!< List of group types (swap-in)
        GPUVector<unsigned int> m_group_tag_alt;     //!< List of group tags (swap-in)
        #ifdef ENABLE_MPI
        GPUVector<ranks_t> m_group_ranks_alt;   //!< 2D list of group member ranks (swap-in)
        #endif

        unsigned int m_nglobal;                      //!< Global number of groups
        std::stack<unsigned int> m_recycled_tags;    //!< Global tags of removed groups
        boost::shared_ptr<Profiler> m_prof;          //!< Profiler

    private:
        bool m_groups_dirty;                         //!< Is it necessary to rebuild the lookup-by-index table?
        boost::signals2::connection m_sort_connection;   //!< Connection to the resort signal from ParticleData
        boost::signals2::signal<void ()> m_group_num_change_signal; //!< Signal that is triggered when groups are added or deleted (globally)

        //! Initialize internal memory
        void initialize();

        //! Set a flag to trigger rebuild of index table
        void setDirty()
            {
            m_groups_dirty = true;
            }

        //! Helper function to rebuild lookup by index table
        void rebuildGPUTable();

        #ifdef ENABLE_CUDA
        //! Helper function to rebuild lookup by index table on the GPU
        void rebuildGPUTableGPU();

        GPUArray<unsigned int> m_condition;          //!< Condition variable for rebuilding GPU table on the GPU
        unsigned int m_next_flag;                    //!< Next flag value for GPU table rebuild
        cached_allocator m_cached_alloc;             //!< Cached allocator for temporary arrays on the GPU
        #endif
    };

//! Exports BondData to python
template<class T>
void export_BondedGroupData(std::string name);

/*!
 * Typedefs for template instantiations
 */

//! Define BondData
extern char name_bond_data[];
typedef BondedGroupData<2, name_bond_data> BondData;

#if 0
//! Define AngleData
extern char name_angle_data[];
typedef BondedGroupData<3, uint3, name_angle_data> AngleData;
template<> void export_BondedGroupData<AngleData>();

//! Define DihedralData
extern char name_dihedral_data[];
typedef BondedGroupData<4, uint4, name_dihedral_data> DihedralData;
template<> void export_BondedGroupData<DihedralData>();

//! Define ImproperData
extern char name_improper_data[];
typedef BondedGroupData<4, uint4, name_improper_data> ImproperData;
template<> void export_BondedGroupData<ImproperData>();
#endif

#endif
