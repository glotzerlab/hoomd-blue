// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


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

#include "ExecutionConfiguration.h"
#include "GPUVector.h"
#include "Profiler.h"
#include "Index1D.h"
#include "HOOMDMath.h"
#include "HOOMDMPI.h"
#include "ParticleData.h"

#ifdef ENABLE_CUDA
#include "CachedAllocator.h"
#include "BondedGroupData.cuh"
#endif

#include <hoomd/extern/nano-signal-slot/nano_signal_slot.hpp>
#include <memory>
#ifndef NVCC
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#endif

#include <stack>
#include <string>
#include <sstream>
#include <set>
#include <vector>
#include <map>

//! Storage data type for group members
/*! We use a union to emphasize it that can contain either particle
 * tags or particle indices or other information */
template<unsigned int group_size>
union group_storage
    {
    unsigned int tag[group_size];
    unsigned int idx[group_size];
    };

//! A union to allow storing a scalar constraint value or a type integer
union typeval_union
    {
    unsigned int type;
    Scalar val;
    };

typedef typeval_union typeval_t;

#ifdef ENABLE_MPI
//! Packed group entry for communication
template<unsigned int group_size>
struct packed_storage
    {
    group_storage<group_size> tags;  //!< Member tags
    typeval_t typeval;               //!< Type of bonded group or constraint value
    unsigned int group_tag;          //!< Tag of this group
    group_storage<group_size> ranks; //!< Current list of member ranks
    };
#endif

#ifdef ENABLE_MPI
namespace cereal
   {
   //! Serialization functions for group data types
        //! Serialization of typeval_union
        template<class Archive>
        void serialize(Archive & ar, typeval_t & t, const unsigned int version)
            {
            // serialize both members
            ar & t.val;
            ar & t.type;
            }

        //! Serialization of group_storage<2> (bonds)
        template<class Archive>
        void serialize(Archive & ar, group_storage<2> & s, const unsigned int version)
            {
            ar & s.tag[0];
            ar & s.tag[1];
            }
        //! Serialization of group_storage<3> (angles)
        template<class Archive>
        void serialize(Archive & ar, group_storage<3> & s, const unsigned int version)
            {
            ar & s.tag[0];
            ar & s.tag[1];
            ar & s.tag[2];
            }
        //! Serialization of group_storage<4> (dihedrals and impropers)
        template<class Archive>
        void serialize(Archive & ar, group_storage<4> & s, const unsigned int version)
            {
            ar & s.tag[0];
            ar & s.tag[1];
            ar & s.tag[2];
            ar & s.tag[3];
            }
    }
#endif

/*! BondedGroupData is a generic storage class for small particle groups of fixed
 *  size N=2,3,4..., such as bonds, angles or dihedrals, which form part of a molecule.
 *
 *  \tpp group_size Size of groups
 *  \tpp name Name of element, i.e. bond, angle, dihedral, ..
 */
template<unsigned int group_size, typename Group, const char *name, bool has_type_mapping = true>
class BondedGroupData
    {
    public:
        //! Group size
        enum { size = group_size } Enum;

        //! Group data element type
        typedef union group_storage<group_size> members_t;

        //! True if typeval is an integer
        static const bool typemap_val = has_type_mapping;

        #ifdef ENABLE_MPI
        //! Type for storing per-member ranks
        typedef members_t ranks_t;
        typedef packed_storage<group_size> packed_t;
        #endif

        //! Handy structure for passing around and initializing the group data
        /*!
         * Bonds in a snapshot are stored with reference to (non-contiguous) particle tags.
         * This implies that if bonds are re-initialized after particle tags have changed,
         * they first need to be updated to point to the correct particle tags.
         */
        struct Snapshot
            {
            //! Default constructor
            Snapshot()
                {
                size = 0;
                }

            //! Constructor
            /*! \param n_groups Number of groups contained in the snapshot
             */
            Snapshot(unsigned int n_groups)
                {
                resize(n_groups);
                }

            //! Resize the snapshot
            /*! \param n_groups Number of groups in the snapshot
             */
            void resize(unsigned int n_groups)
                {
                // zero the newly created bonds
                group_storage<group_size> def;
                memset(&def, 0, sizeof(def));

                if (has_type_mapping)
                    {
                    type_id.resize(n_groups, 0);
                    }
                else
                    {
                    val.resize(n_groups, 0);
                    }
                groups.resize(n_groups, def);
                size = n_groups;
                }

            //! Validate the snapshot
            /* \returns true if number of elements in snapshot is consistent
             */
            bool validate() const
                {
                if (has_type_mapping && groups.size() != type_id.size()) return false;
                if (!has_type_mapping && groups.size() != val.size()) return false;
                return true;
                }

            //! Replicate this snapshot
            /*! \param n Number of times to replicate each particle and its bonded groups
             *  \param old_n_particles Number of particles in system to be replicated
             */
            void replicate(unsigned int n, unsigned int old_n_particles);

            #ifdef ENABLE_MPI
            //! Broadcast the snapshot
            /*! \param root the processor to send from
             *  \param mpi_comm The MPI communicator
             */
            void bcast(unsigned int root, MPI_Comm mpi_comm)
                {
                ::bcast(type_id, root, mpi_comm);
                ::bcast(val, root, mpi_comm);
                ::bcast(groups, root, mpi_comm);
                ::bcast(type_mapping, root, mpi_comm);
                ::bcast(size, root, mpi_comm);
                }
            #endif

            //! Get type as a numpy array
            static pybind11::object getTypeNP(pybind11::object self);
            //! Get value as a numpy array
            static pybind11::object getValueNP(pybind11::object self);

            //! Get bonded tags as a numpy array
            static pybind11::object getBondedTagsNP(pybind11::object self);
            //! Get the type names for python
            pybind11::list getTypes();
            //! Set the type names from python
            void setTypes(pybind11::list types);

            std::vector<unsigned int> type_id;             //!< Stores type for each group
            std::vector<Scalar> val;                       //!< Stores constraint value for each group
            std::vector<members_t> groups;                 //!< Stores the data for each group
            std::vector<std::string> type_mapping;         //!< Names of group types
            unsigned int size;                             //!< Number of bonds in the snapshot
            };

        //! Constructor for empty BondedGroupData
        BondedGroupData(std::shared_ptr<ParticleData> pdata,
            unsigned int n_group_types);

        //! Constructor to initialize from a snapshot
        BondedGroupData(std::shared_ptr<ParticleData> pdata,
            const Snapshot& snapshot);

        virtual ~BondedGroupData();

        //! Initialize from a snapshot
        virtual void initializeFromSnapshot(const Snapshot& snapshot);

        //! Take a snapshot
        virtual std::map<unsigned int, unsigned int> takeSnapshot(Snapshot& snapshot) const;

        //! Get local number of bonded groups
        unsigned int getN() const
            {
            return m_n_groups;
            }

        //! Remove all ghost groups
        /*! This method does not actually operate on the group data, it just ensures
            that the internal counters are reset.
         */
        void removeAllGhostGroups()
            {
            unsigned int new_size = m_groups.size() - m_n_ghost;
            reallocate(new_size);
            m_n_ghost = 0;
            }

        //! Add ghost groups
        /*! \param nghost The number of ghost groups to add
         *
         * This method does not modify any actual group data.
         */
        void addGhostGroups(unsigned int ngroup)
            {
            unsigned int new_size = m_groups.size()+ngroup;
            reallocate(new_size);
            m_n_ghost += ngroup;
            }

        //! Get local number of bonded groups
        unsigned int getNGhosts() const
            {
            return m_n_ghost;
            }


        //! Get global number of bonded groups
        unsigned int getNGlobal() const
            {
            return m_nglobal;
            }

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

        //! Rename a type
        void setTypeName(unsigned int type, const std::string& new_name);

        //! Return the nth active global tag
        unsigned int getNthTag(unsigned int n);

        //! Return the maximum particle tag in the simulation
        unsigned int getMaximumTag() const
            {
            assert(!m_tag_set.empty());
            return *m_tag_set.rbegin();
            }

        //! Return a bonded group by tag
        const Group getGroupByTag(unsigned int tag) const;

        //! Get the members of a bonded group by index
        const members_t getMembersByIndex(unsigned int group_idx) const;

        //! Get the type of a bonded group by index
        unsigned int getTypeByIndex(unsigned int group_idx) const;

        //! Get the constraint value of a bonded group by index
        Scalar getValueByIndex(unsigned int group_idx) const;

        /*
         * Access to data structures
         */

        //! Add local groups
        /*! \note It is assumed that there are no ghost groups present
            at the time this method is called
         */
        void addGroups(unsigned int ngroup)
            {
            assert(m_n_ghost == 0);
            unsigned int new_size = m_n_groups + ngroup;
            reallocate(new_size);
            m_n_groups += ngroup;
            }

        //! Remove local groups
        /*! \note It is assumed that there are no ghost groups present
            at the time this method is called
         */
        void removeGroups(unsigned int nremove)
            {
            assert(m_n_ghost == 0);
            assert(m_n_groups >= nremove);
            unsigned int new_size = m_n_groups - nremove;
            reallocate(new_size);
            m_n_groups -= nremove;
            }

        //! Return group table (const)
        const GPUVector<members_t>& getMembersArray() const
            {
            return m_groups;
            }

        //! Return group table (const)
        const GPUVector<typeval_t>& getTypeValArray() const
            {
            return m_group_typeval;
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
        //! Return auxiliary array of member particle ranks (const)
        const GPUVector<ranks_t>& getRanksArray() const
            {
            return m_group_ranks;
            }
        #endif

        //! Return group table
        GPUVector<members_t>& getMembersArray()
            {
            return m_groups;
            }

        //! Return group table
        GPUVector<typeval_t>& getTypeValArray()
            {
            return m_group_typeval;
            }

        //! Return list of group tags
        GPUVector<unsigned int>& getTags()
            {
            return m_group_tag;
            }

        //! Return reverse-lookup table (group tag-> group index)
        GPUVector<unsigned int>& getRTags()
            {
            return m_group_rtag;
            }

        #ifdef ENABLE_MPI
        //! Return auxiliary array of member particle ranks
        GPUVector<ranks_t>& getRanksArray()
            {
            return m_group_ranks;
            }
        #endif

        /*
         * Alternate arrays used for reordering data
         * \sa ParticleData
         *
         * The alternate arrays are returned as non-const versions
         * to enable resizing the underlying GPUVectors before swapping.
         */

        //! Return group table (swap-in)
        GPUVector<members_t>& getAltMembersArray()
            {
            // resize to size of primary groups array
            m_groups_alt.resize(m_groups.size());
            return m_groups_alt;
            }

        //! Return group table (swap-in)
        GPUVector<typeval_t>& getAltTypeValArray()
            {
            // resize to size of primary group types array
            m_group_typeval_alt.resize(m_group_typeval.size());
            return m_group_typeval_alt;
            }

        //! Return list of group tags (swap-in)
        GPUVector<unsigned int>& getAltTags()
            {
            // resize to size of primary group tags array
            m_group_tag_alt.resize(m_group_tag.size());
            return m_group_tag_alt;
            }

        #ifdef ENABLE_MPI
        //! Return auxiliary array of member particle ranks
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

        //! Swap group type/value arrays
        void swapTypeArrays()
            {
            assert(!m_group_typeval_alt.isNull());
            m_group_typeval.swap(m_group_typeval_alt);
            }

        //! Swap group tag arrays
        void swapTagArrays()
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
        const GPUVector<members_t>& getGPUTable()
            {
            // rebuild lookup table if necessary
            if (m_groups_dirty)
                {
                rebuildGPUTable();
                m_groups_dirty = false;
                }

            return m_gpu_table;
            }

        //! Return GPU list of particle in group position
        const GPUArray<unsigned >& getGPUPosTable()
            {
            // rebuild lookup table if necessary
            if (m_groups_dirty)
                {
                rebuildGPUTable();
                m_groups_dirty = false;
                }

            return m_gpu_pos_table;
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
            return m_gpu_n_groups;
            }

        /*
         * add/remove groups globally
         */

        //! Add a single bonded group on all processors
        /*! \param g Definition of group to add
         */
        unsigned int addBondedGroup(Group g);

        //! Remove a single bonded group from all processors
        /*! \param tag Tag of bonded group to remove
         */
        void removeBondedGroup(unsigned int group_tag);

        //! Set the profiler
        /*! \param prof The profiler
         */
        void setProfiler(std::shared_ptr<Profiler> prof)
            {
            m_prof = prof;
            }

        //! Connects a function to be called every time the global number of bonded groups changes
        Nano::Signal<void()>& getGroupNumChangeSignal()
            {
            return m_group_num_change_signal;
            }

        //! Connects a function to be called every time the local number of bonded groups changes
        Nano::Signal<void()>& getGroupReorderSignal()
            {
            return m_group_reorder_signal;
            }


        //! Notify subscribers that groups have been reordered
        void notifyGroupReorder()
            {
            // set flag to trigger rebuild of GPU table
            m_groups_dirty = true;

            // notify subscribers
            m_group_reorder_signal.emit();
            }

        //! Indicate that GPU table needs to be rebuilt
        void setDirty()
            {
            m_groups_dirty = true;
            }

    protected:
        #ifdef ENABLE_MPI
        //! Helper function to transfer bonded groups connected to a single particle
        /*! \param tag Tag of particle that moves between domains
            \param old_rank Old MPI rank for particle
            \param new_rank New MPI rank
         */
        void moveParticleGroups(unsigned int tag, unsigned int old_rank, unsigned int new_rank);
        #endif

        std::shared_ptr<const ExecutionConfiguration> m_exec_conf;  //!< Execution configuration for CUDA context
        std::shared_ptr<ParticleData> m_pdata;        //!< Particle Data these bonds belong to

        GPUVector<members_t> m_groups;               //!< List of groups
        GPUVector<typeval_t> m_group_typeval;        //!< List of group types/constraint values
        GPUVector<unsigned int> m_group_tag;         //!< List of group tags
        GPUVector<unsigned int> m_group_rtag;        //!< Global reverse-lookup table for group tags
        GPUVector<members_t> m_gpu_table;            //!< Storage for groups by particle index for access on the GPU
        GPUVector<unsigned int> m_gpu_pos_table;     //!< Position of particle idx in group table
        Index2D m_gpu_table_indexer;                 //!< Indexer for GPU table
        GPUVector<unsigned int> m_gpu_n_groups;      //!< Number of entries in lookup table per particle
        std::vector<std::string> m_type_mapping;     //!< Mapping of types of bonded groups

        unsigned int m_n_groups;                     //!< Number of local groups
        unsigned int m_n_ghost;                      //!< Number of ghost groups with no local ptl

        #ifdef ENABLE_MPI
        GPUVector<ranks_t> m_group_ranks;       //!< 2D list of group member ranks
        #endif

        /* alternate (stand-by) arrays for swapping in reordered groups */
        GPUVector<members_t> m_groups_alt;           //!< List of groups (swap-in)
        GPUVector<typeval_t> m_group_typeval_alt;    //!< List of group types/constraint values (swap-in)
        GPUVector<unsigned int> m_group_tag_alt;     //!< List of group tags (swap-in)
        #ifdef ENABLE_MPI
        GPUVector<ranks_t> m_group_ranks_alt;   //!< 2D list of group member ranks (swap-in)
        #endif

        unsigned int m_nglobal;                      //!< Global number of groups
        std::stack<unsigned int> m_recycled_tags;    //!< Global tags of removed groups
        std::set<unsigned int> m_tag_set;            //!< Lookup table for tags by active index
        GPUVector<unsigned int> m_cached_tag_set;    //!< Cached constant-time lookup table for tags by active index
        bool m_invalid_cached_tags;                  //!< true if m_cached_tag_set needs to be rebuilt
        std::shared_ptr<Profiler> m_prof;          //!< Profiler

    private:
        bool m_groups_dirty;                         //!< Is it necessary to rebuild the lookup-by-index table?

        Nano::Signal<void ()> m_group_num_change_signal; //!< Signal that is triggered when groups are added or deleted (globally)
        Nano::Signal<void ()> m_group_reorder_signal;    //!< Signal that is triggered when groups are added or deleted locally

        //! Initialize internal memory
        void initialize();

        //! Helper function to rebuild the active tag cache if necessary
        void maybe_rebuild_tag_cache();

        //! Helper function to rebuild lookup by index table
        void rebuildGPUTable();

        //! Resize internal tables
        /*! \param new_size New size of local group tables, new_size = n_local + n_ghost
         */
        void reallocate(unsigned int new_size)
            {
            m_groups.resize(new_size);
            m_group_typeval.resize(new_size);
            m_group_tag.resize(new_size);
            #ifdef ENABLE_MPI
            if (m_pdata->getDomainDecomposition())
                {
                m_group_ranks.resize(new_size);
                }
            #endif
            }

        #ifdef ENABLE_CUDA
        //! Helper function to rebuild lookup by index table on the GPU
        void rebuildGPUTableGPU();

        GPUArray<unsigned int> m_condition;          //!< Condition variable for rebuilding GPU table on the GPU
        unsigned int m_next_flag;                    //!< Next flag value for GPU table rebuild
        mgpu::ContextPtr m_mgpu_context;                   //!< moderngpu context
        #endif
    };

//! Exports BondData to python
template<class T, class Group>
void export_BondedGroupData(pybind11::module& m, std::string name, std::string snapshot_name, bool export_struct=true);

/*!
 * Typedefs for template instantiations
 */

/*
 * BondData
 */
extern char name_bond_data[];

// Definition of a bond
struct Bond {
    typedef group_storage<2> members_t;

    //! Constructor
    /*! \param type Type of bond
     * \param _a First bond member
     * \param _b Second bond member
     */
    Bond(unsigned int _type, unsigned int _a, unsigned int _b)
        : type(_type), a(_a), b(_b)
        { }

    //! Constructor that takes a members_t (used internally by BondData)
    /*! \param type
     *  \param members group members
     */
    Bond(typeval_t _typeval, members_t _members)
        : type(_typeval.type), a(_members.tag[0]), b(_members.tag[1])
        { }


    //! This helper function needs to be provided for the templated BondData to work correctly
    members_t get_members() const
        {
        members_t m;
        m.tag[0] = a;
        m.tag[1] = b;
        return m;
        }

    //! This helper function needs to be provided for the templated BondData to work correctly
    typeval_t get_typeval() const
        {
        typeval_t t;
        t.type = type;
        return t;
        }

    //! This helper function needs to be provided for the templated BondData to work correctly
    static void export_to_python(pybind11::module& m)
        {
        pybind11::class_<Bond>(m,"Bond")
            .def(pybind11::init<unsigned int, unsigned int, unsigned int>())
            .def_readonly("type", &Bond::type)
            .def_readonly("a", &Bond::a)
            .def_readonly("b", &Bond::b)
        ;
        }

    unsigned int type;  //!< Group type
    unsigned int a;     //!< First bond member
    unsigned int b;     //!< Second bond member
    };

//! Definition of BondData
typedef BondedGroupData<2, Bond, name_bond_data> BondData;

/*
 * AngleData
 */
extern char name_angle_data[];

// Definition of an angle
struct Angle {
    typedef group_storage<3> members_t;

    //! Constructor
    /*! \param type Type of angle
     * \param _a First angle member
     * \param _b Second angle member
     */
    Angle(unsigned int _type, unsigned int _a, unsigned int _b, unsigned int _c)
        : type(_type), a(_a), b(_b), c(_c)
        { }

    //! Constructor that takes a members_t (used internally by AngleData)
    /*! \param type
     *  \param members group members
     */
    Angle(typeval_t _typeval, members_t _members)
        : type(_typeval.type), a(_members.tag[0]), b(_members.tag[1]), c(_members.tag[2])
        { }


    //! This helper function needs to be provided for the templated AngleData to work correctly
    members_t get_members() const
        {
        members_t m;
        m.tag[0] = a;
        m.tag[1] = b;
        m.tag[2] = c;
        return m;
        }

    //! This helper function needs to be provided for the templated AngleData to work correctly
    typeval_t get_typeval() const
        {
        typeval_t t;
        t.type = type;
        return t;
        }

    //! This helper function needs to be provided for the templated AngleData to work correctly
    static void export_to_python(pybind11::module& m)
        {
        pybind11::class_<Angle>(m,"Angle")
            .def(pybind11::init<unsigned int, unsigned int, unsigned int, unsigned int>())
            .def_readonly("type", &Angle::type)
            .def_readonly("a", &Angle::a)
            .def_readonly("b", &Angle::b)
            .def_readonly("c", &Angle::c)
        ;
        }

    unsigned int type;  //!< Group type
    unsigned int a;     //!< First angle member
    unsigned int b;     //!< Second angle member
    unsigned int c;     //!< Third angle member
    };

//! Definition of AngleData
typedef BondedGroupData<3, Angle, name_angle_data> AngleData;

/*
 * DihedralData
 */
extern char name_dihedral_data[];

// Definition of an dihedral
struct Dihedral {
    typedef group_storage<4> members_t;

    //! Constructor
    /*! \param type Type of dihedral
     * \param _a First dihedral member
     * \param _b Second dihedral member
     */
    Dihedral(unsigned int _type, unsigned int _a, unsigned int _b, unsigned int _c, unsigned int _d)
        : type(_type), a(_a), b(_b), c(_c), d(_d)
        { }

    //! Constructor that takes a members_t (used internally by DihedralData)
    /*! \param type
     *  \param members group members
     */
    Dihedral(typeval_t _typeval, members_t _members)
        : type(_typeval.type), a(_members.tag[0]), b(_members.tag[1]), c(_members.tag[2]), d(_members.tag[3])
        { }


    //! This helper function needs to be provided for the templated DihedralData to work correctly
    members_t get_members() const
        {
        members_t m;
        m.tag[0] = a;
        m.tag[1] = b;
        m.tag[2] = c;
        m.tag[3] = d;
        return m;
        }

    //! This helper function needs to be provided for the templated DihedralData to work correctly
    typeval_t get_typeval() const
        {
        typeval_t t;
        t.type = type;
        return t;
        }

    //! This helper function needs to be provided for the templated DihedralData to work correctly
    static void export_to_python(pybind11::module& m)
        {
        pybind11::class_<Dihedral>(m,"Dihedral")
            .def(pybind11::init<unsigned int, unsigned int, unsigned int, unsigned int, unsigned int>())
            .def_readonly("type", &Dihedral::type)
            .def_readonly("a", &Dihedral::a)
            .def_readonly("b", &Dihedral::b)
            .def_readonly("c", &Dihedral::c)
            .def_readonly("d", &Dihedral::d)
        ;
        }

    unsigned int type;  //!< Group type
    unsigned int a;     //!< First dihedral member
    unsigned int b;     //!< Second dihedral member
    unsigned int c;     //!< Third dihedral member
    unsigned int d;     //!< Fourth dihedral member
    };

//! Definition of DihedralData
typedef BondedGroupData<4, Dihedral, name_dihedral_data> DihedralData;

/*
 * ImproperData
 */
extern char name_improper_data[];

//! Definition of ImproperData
typedef BondedGroupData<4, Dihedral, name_improper_data> ImproperData;

/*
 * ConstraintData
 *
 * constraints use the same data type as bonds
 */
extern char name_constraint_data[];

//! Definition of a constraint
/*! Constraints are essentially bonds, but of a single type
    The type information stores the constraint distance
 */
struct Constraint {
    typedef group_storage<2> members_t;

    //! Constructor
    /*! \param d Constraint distance
     * \param _a First bond member
     * \param _b Second bond member
     */
    Constraint(Scalar _d, unsigned int _a, unsigned int _b)
        : d(_d), a(_a), b(_b)
        { }

    //! Constructor that takes a members_t (used internally by BondData)
    /*! \param type
     *  \param members group members
     */
    Constraint(typeval_t _typeval, members_t _members)
        : d(_typeval.val), a(_members.tag[0]), b(_members.tag[1])
        { }

    //! This helper function needs to be provided for the templated BondData to work correctly
    members_t get_members() const
        {
        members_t m;
        m.tag[0] = a;
        m.tag[1] = b;
        return m;
        }

    //! This helper function needs to be provided for the templated BondData to work correctly
    typeval_t get_typeval() const
        {
        typeval_t t;
        t.val = d;
        return t;
        }

    //! This helper function needs to be provided for the templated ConstraintData to work correctly
    static void export_to_python(pybind11::module& m)
        {
        pybind11::class_<Constraint>(m,"Constraint")
            .def(pybind11::init<Scalar, unsigned int, unsigned int>())
            .def_readonly("d", &Constraint::d)
            .def_readonly("a", &Constraint::a)
            .def_readonly("b", &Constraint::b)
        ;
        }

    Scalar d;           //!< Constraint distance
    unsigned int a;     //!< First constraint member
    unsigned int b;     //!< Second constraint member
    };


//! Definition of ConstraintData
typedef BondedGroupData<2, Constraint, name_constraint_data, false> ConstraintData;

/*
 * PairData
 *
 * used for pair potentials between special particle pairs
 */
extern char name_pair_data[];

//! Definition of PairData
typedef BondedGroupData<2, Bond, name_pair_data> PairData;

#endif
