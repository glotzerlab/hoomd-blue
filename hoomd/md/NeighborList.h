// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hoomd/Compute.h"
#include "hoomd/GPUFlags.h"
#include "hoomd/GPUVector.h"
#include "hoomd/GlobalArray.h"
#include "hoomd/Index1D.h"
#include "hoomd/MeshDefinition.h"
#include "hoomd/PythonLocalDataAccess.h"

#include <hoomd/extern/nano-signal-slot/nano_signal_slot.hpp>
#include <memory>
#include <set>
#include <vector>

/*! \file NeighborList.h
    \brief Declares the NeighborList class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __NEIGHBORLIST_H__
#define __NEIGHBORLIST_H__

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#endif

namespace hoomd
    {
namespace md
    {
//! Computes a Neighborlist from the particles
/*! \b Overview:

    A particle \c i is a neighbor of particle \c j if the distance between
    particle them is less than or equal to \c r_cut(i,j). The neighborlist for a given particle
    \c i includes all of these neighbors at a minimum. Other particles are included
    in the list: those up to \c r_list(i,j) which includes a buffer distance so that the neighbor
   list doesn't need to be updated every step.

    There are two ways of storing this information. One is to store only half of the
    neighbors (only those with i < j), and the other is to store all neighbors. There are
    potential tradeoffs between number of computations and memory access complexity for
    each method. NeighborList supports both of these modes via a switch: setStorageMode();

    Some classes with either setting, full or half, but they are faster with the half setting.
   However, others may require that the neighbor list storage mode is set to full.

    <b>Data access:</b>

    Up to Nmax neighbors can be stored for each particle. Data is stored in a flat array in memory.
   A secondary flat list is supplied for each particle which specifies where to start reading
   neighbors from the list (a "head" list). Each element in the list stores the index of the
   neighbor with the highest bits reserved for flags. The head list for accessing elements can be
   gotten with getHeadList() and the array itself can be accessed with getNlistArray().

    The number of neighbors for each particle is stored in an auxiliary array accessed with
   getNNeighArray().

     - <code>jf = nlist[head_list[i] + n]</code> is the index of neighbor \a n of particle \a i,
   where \a n can vary from 0 to <code>n_neigh[i] - 1</code>

    \a jf includes flags in the highest bits. The format and use of these flags are yet to be
   determined.

    \b Filtering:

    By default, a neighbor list includes all particles within a single cutoff distance r_cut.
   Various filters can be applied to remove unwanted neighbors from the list.
     - setFilterBody() prevents two particles of the same body from being neighbors

    \b Algorithms:

    This base class supplies no build algorithm for generating this list, it must be overridden by
   deriving classes. Derived classes implement O(N) efficient strategies using a CellList or a BVH
   tree.

    <b>Needs update check:</b>

    When compute() is called, the neighbor list is updated, but only if it needs to be. Checks
    are performed to see if any particle has moved more than half of the buffer distance, and
    only then is the list actually updated. This check can even be avoided for a number of time
    steps by calling setEvery(). If the caller wants to force a full update, forceUpdate()
    can be called before compute() to do so. Note that if the particle data is resorted,
    an update is automatically forced.

    The CUDA profiler expects the exact same sequence of kernels on every run. Due to the
   non-deterministic cell list, a different sequence of calls may be generated with nlist builds at
   different times. To work around this problem setEvery takes a dist_check parameter. When
   dist_check=True, the above described behavior is followed. When dist_check is false, the nlist is
   built exactly m_rebuild_check_delay steps. This is intended for use in profiling only.

    \b Exclusions:

    Exclusions are stored in \a ex_list, a data structure similar in structure to \a nlist, except
   this time exclusions are stored. User-specified exclusions are stored by tag and translated to
   indices whenever a particle sort occurs (updateExListIdx()). If any exclusions are set,
   filterNlist() is called after buildNlist(). filterNlist() loops through the neighbor list and
   removes any particles that are excluded. This allows an arbitrary number of exclusions to be
   processed without slowing the performance of the buildNlist() step itself.

    <b>Overflow handling:</b>
    For easy support of derived GPU classes to implement overflow detection the overflow condition
   is stored in the GlobalArray \a d_conditions.

     - 0: Maximum nlist size (implementations are free to write to this element only in overflow
   conditions if they choose.)
     - Further indices may be added to handle other conditions at a later time.

    Condition flags are to be set during the buildNlist() call and will be checked by compute()
   which will then take the appropriate action.

    \ingroup computes
*/
class PYBIND11_EXPORT NeighborList : public Compute
    {
    public:
    //! Simple enum for the storage modes
    enum storageMode
        {
        half, //!< Only neighbors i,j are stored where i < j
        full  //!< All neighbors are stored
        };

    //! Constructs the compute
    NeighborList(std::shared_ptr<SystemDefinition> sysdef, Scalar r_buff);

    //! Destructor
    virtual ~NeighborList();

    //! \name Set parameters
    // @{

    /** Add a r_cut matrix

    @param r_cut_matrix Matrix to add

    NeighborList consumers must provide a per type pair r_cut matrix to the neighbor list so
    that when one consumer changes a r_cut value, NeighborList can compute the needed minimum
    and maximum r_cut values across all types.

    Consumers should call notifyRCutMatrixChange() when they any element of of their matrix.
    They should call removeRCutMatrix() when they no longer need to use the neighbor list.
    */
    void addRCutMatrix(const std::shared_ptr<GlobalArray<Scalar>>& r_cut_matrix)
        {
        if (r_cut_matrix->getNumElements() != m_r_cut.getNumElements())
            {
            throw std::invalid_argument("given r_cut_matrix is not the right size");
            }

        m_consumer_r_cut.push_back(r_cut_matrix);
        notifyRCutMatrixChange();
        forceUpdate();
        }

    /// Notify NeighborList that a r_cut matrix value has changed
    virtual void notifyRCutMatrixChange()
        {
        m_rcut_signal.emit();
        forceUpdate();
        }

    /// Add Mesh for meshbond_data
    virtual void AddMesh(std::shared_ptr<MeshDefinition> meshdef)
        {
        m_meshbond_data = meshdef->getMeshBondData();
        m_meshbond_data->getGroupNumChangeSignal()
            .connect<NeighborList, &NeighborList::slotGlobalTopologyNumberChange>(this);
        }

    /** Remove a r_cut matrix

    @param r_cut_matrix Matrix to remove

    Remove a r_cut matrix from the neighbor list. The given matrix will no longer be included
    in the min & max r_cuts when computing the neighbor list.
    */
    void removeRCutMatrix(const std::shared_ptr<GlobalArray<Scalar>>& r_cut_matrix)
        {
        auto p = std::find(m_consumer_r_cut.begin(), m_consumer_r_cut.end(), r_cut_matrix);
        if (p == m_consumer_r_cut.end())
            {
            throw std::invalid_argument("r_cut_matrix not found in neighbor list");
            }
        m_consumer_r_cut.erase(p);
        }

    //! Change the global buffer radius
    virtual void setRBuff(Scalar r_buff);

    //! Change how many timesteps before checking to see if the list should be rebuilt
    /*! \param every Number of time steps to wait before beginning to check if particles have moved
       a sufficient distance to require a neighbor list update. \param dist_check Set to false to
       enforce nlist builds exactly \a every steps
    */
    void setRebuildCheckDelay(uint64_t every)
        {
        m_rebuild_check_delay = every;
        forceUpdate();
        }

    uint64_t getRebuildCheckDelay()
        {
        return m_rebuild_check_delay;
        }

    void setDistCheck(bool dist_check)
        {
        m_dist_check = dist_check;
        }

    bool getDistCheck()
        {
        return m_dist_check;
        }

    //! Set the storage mode
    /*! \param mode Storage mode to set
        - half only stores neighbors where i < j
        - full stores all neighbors

        The neighborlist is not immediately updated to reflect this change. It will take effect
        when compute is called for the next timestep.
    */
    void setStorageMode(storageMode mode)
        {
        m_storage_mode = mode;
        forceUpdate();
        }

    // @}
    //! \name Get properties
    // @{

    //! Get the storage mode
    storageMode getStorageMode()
        {
        return m_storage_mode;
        }

    //! Get the maximum of all rcut
    Scalar getMaxRCut()
        {
        if (m_rcut_changed)
            updateRList();
        return m_rcut_max_max;
        }

    //! Get the maximum of all the rlist
    Scalar getMaxRList()
        {
        return getMaxRCut() + m_r_buff;
        }

    //! Get the minimum of all rcut
    Scalar getMinRCut()
        {
        if (m_rcut_changed)
            updateRList();
        return m_rcut_min;
        }

    //! Get the minimum of all rlist
    Scalar getMinRList()
        {
        return getMinRCut() + m_r_buff;
        }

    //! Get the buffering (skin) length
    Scalar getRBuff()
        {
        return m_r_buff;
        }

    // @}
    //! \name Statistics
    // @{

    //! Clear the count of updates the neighborlist has performed
    virtual void resetStats();

    //! Gets the shortest rebuild period this nlist has experienced since a call to resetStats
    unsigned int getSmallestRebuild();

    // @}
    //! \name Get data
    // @{

    //! Get the number of neighbors array
    const GlobalArray<unsigned int>& getNNeighArray() const
        {
        return m_n_neigh;
        }

    //! Get the neighbor list
    const GlobalArray<unsigned int>& getNListArray() const
        {
        return m_nlist;
        }

    //! Get the head list
    const GlobalArray<size_t>& getHeadList() const
        {
        return m_head_list;
        }

    //! Get the number of exclusions array
    const GlobalArray<unsigned int>& getNExArray()
        {
        return m_n_ex_idx;
        }

    //! Get the exclusion list
    const GlobalArray<unsigned int>& getExListArray()
        {
        return m_ex_list_idx;
        }

    //! Get the neighbor list indexer
    /*! \note Do not save indexers across calls. Get a new indexer after every call to compute() -
       they will change.
    */
    const Index2D& getExListIndexer()
        {
        return m_ex_list_indexer;
        }

    void setExclusions(pybind11::list exclusions);

    void setSingleExclusion(std::string exclusion);

    pybind11::list getExclusions();

    bool getExclusionsSet()
        {
        return m_exclusions_set;
        }

    //! Exclude a pair of particles from being added to the neighbor list
    void addExclusion(unsigned int tag1, unsigned int tag2);

    //! Enable/disable body filtering
    virtual void setFilterBody(bool filter_body)
        {
        // detect if there are any rigid bodies in the system
        bool has_bodies = m_pdata->hasBodies();
        if (has_bodies || !filter_body)
            {
            // only activate body filtering if bodies are present,
            // otherwise it is waste of cycles
            m_filter_body = filter_body;
            }
        forceUpdate();
        }

    //! Collect some statistics on exclusions.
    void countExclusions();

    //! Get number of exclusions involving n particles
    /*! \param n Size of the exclusion
     * \returns Number of excluded particles
     */
    unsigned int getNumExclusions(unsigned int size);

    //! Test if body filtering is set
    virtual bool getFilterBody()
        {
        return m_filter_body;
        }

    //! Return the requested ghost layer width
    virtual Scalar getGhostLayerWidth(unsigned int type)
        {
        if (m_rcut_changed)
            {
            updateRList();
            }

        ArrayHandle<Scalar> h_rcut_max(m_rcut_max, access_location::host, access_mode::read);
        const Scalar rcut_max_i = h_rcut_max.data[type];

        if (rcut_max_i > Scalar(0.0)) // ensure communication is required
            {
            return rcut_max_i + m_r_buff;
            }
        else
            {
            return Scalar(0.0);
            }
        }

    // @}

    //! Computes the NeighborList if it needs updating
    void compute(uint64_t timestep);

    //! Forces a full update of the list on the next call to compute()
    void forceUpdate()
        {
        m_force_update = true;
        }

    //! Get the number of updates
    virtual uint64_t getNumUpdates()
        {
        return m_updates + m_forced_updates;
        }

#ifdef ENABLE_MPI
    //! Returns true if the particle migration criterion is fulfilled
    /*! \param timestep The current timestep
     */
    bool peekUpdate(uint64_t timestep);
#endif

    //! Return true if the neighbor list has been updated this time step
    /*! \param timestep Current time step
     *
     *  This is supposed to be called after a call to compute().
     */
    bool hasBeenUpdated(uint64_t timestep)
        {
        return m_last_updated_tstep == timestep && m_has_been_updated_once;
        }

    Nano::Signal<void()>& getRCutChangeSignal()
        {
        return m_rcut_signal;
        }

    /// Index type pairs
    const Index2D& getTypePairIndexer()
        {
        return m_typpair_idx;
        }

    unsigned int getN() const
        {
        return m_pdata->getN();
        }

    unsigned int getNGhosts() const
        {
        return m_pdata->getNGhosts();
        }

    /// Get the local pair list from Python
    pybind11::array_t<uint32_t> getLocalPairListPython(uint64_t timestep);

    /// Get the global pair list from Python
    pybind11::object getPairListPython(uint64_t timestep);

    /// Validate that types are within Ntypes
    void validateTypes(unsigned int typ1, unsigned int typ2, std::string action);

    /// Set the rcut for a single type pair
    virtual void setRcut(unsigned int typ1, unsigned int typ2, Scalar rcut);

    /// Get the r_cut for a single type pair
    Scalar getRCut(pybind11::tuple types);

    /// Set the rcut for a single type pair using a tuple of strings
    virtual void setRCutPython(pybind11::tuple types, Scalar r_cut);

    protected:
    Index2D m_typpair_idx;           //!< Indexer for full type pair storage
    GlobalArray<Scalar> m_r_cut;     //!< The potential cutoffs stored by pair type
    GlobalArray<Scalar> m_r_listsq;  //!< The neighborlist cutoff radius squared stored by pair type
    GlobalArray<Scalar> m_rcut_max;  //!< The maximum value of rcut per particle type
    GlobalArray<Scalar> m_rcut_base; //!< The base rcut values

    /// List of r_cut matrices from neighborlist consumers
    std::vector<std::shared_ptr<GlobalArray<Scalar>>> m_consumer_r_cut;

    Scalar m_rcut_max_max;      //!< The maximum cutoff radius of any pair
    Scalar m_rcut_min;          //!< The smallest cutoff radius of any pair (that is > 0)
    Scalar m_r_buff;            //!< The buffer around the cutoff
    bool m_filter_body;         //!< Set to true if particles in the same body are to be filtered
    storageMode m_storage_mode; //!< The storage mode

    GlobalArray<unsigned int> m_nlist;   //!< Neighbor list data
    GlobalArray<unsigned int> m_n_neigh; //!< Number of neighbors for each particle
    GlobalArray<Scalar4> m_last_pos;     //!< coordinates of last updated particle positions
    Scalar3 m_last_L;                    //!< Box lengths at last update
    Scalar3 m_last_L_local;              //!< Local Box lengths at last update

    GlobalArray<size_t> m_head_list; //!< Indexes for particles to read from the neighbor list
    GlobalArray<unsigned int>
        m_Nmax; //!< Holds the maximum number of neighbors for each particle type
    GlobalArray<unsigned int>
        m_conditions; //!< Holds the max number of computed particles by type for resizing

    GlobalArray<unsigned int> m_ex_list_tag; //!< List of excluded particles referenced by tag
    GlobalArray<unsigned int> m_ex_list_idx; //!< List of excluded particles referenced by index
    GlobalVector<unsigned int> m_n_ex_tag;   //!< Number of exclusions for a given particle tag
    GlobalArray<unsigned int> m_n_ex_idx;    //!< Number of exclusions for a given particle index
    Index2D m_ex_list_indexer;               //!< Indexer for accessing the exclusion list
    Index2D m_ex_list_indexer_tag;           //!< Indexer for accessing the by-tag exclusion list
    bool m_exclusions_set;                   //!< True if any exclusions have been set

    std::shared_ptr<MeshBondData> m_meshbond_data;

    /// True if the number of particles has changed.
    bool m_n_particles_changed = false;

    /// True if the number of bonds/angles/dihedrals/impropers/pairs has changed.
    bool m_topology_changed = false;

#ifdef ENABLE_MPI
    /// The system's communicator.
    std::shared_ptr<Communicator> m_comm;
#endif

    //! Return true if we are supposed to do a distance check in this time step
    bool shouldCheckDistance(uint64_t timestep);

    //! Performs the distance check
    virtual bool distanceCheck(uint64_t timestep);

    //! Updates the previous position table for use in the next distance check
    virtual void setLastUpdatedPos();

    //! Builds the neighbor list
    virtual void buildNlist(uint64_t timestep);

    //! Updates the idx exclusion list
    virtual void updateExListIdx();

    //! Loops through all pairs, and updates the r_list(i,j)
    void updateRList();

    //! Checks that box is big enough for neighbor list cutoff
    void checkBoxSize();

    //! Filter the neighbor list of excluded particles
    virtual void filterNlist();

    //! Build the head list to allocated memory
    virtual void buildHeadList();

    //! Amortized resizing of the neighborlist
    void resizeNlist(size_t size);

#ifdef ENABLE_MPI
    CommFlags getRequestedCommFlags(uint64_t timestep)
        {
        CommFlags flags(0);
        flags[comm_flag::position] = 1;
        flags[comm_flag::tag] = 1;

        if (m_filter_body)
            flags[comm_flag::body] = 1;

        return flags;
        }
#endif

#ifdef ENABLE_HIP
    //! Reset memory usage hints
    void unsetMemoryMapping();

    //! Update memory usage hints
    void updateMemoryMapping();
#endif

    private:
    Nano::Signal<void()> m_rcut_signal; //!< Signal that is triggered when the cutoff radius changes

    bool m_rcut_changed; //!< Flag if the rcut array has changed
    //! Notify the NeighborList that the rcut has changed for delayed updating
    void slotRCutChange()
        {
        m_rcut_changed = true;
        }

    uint64_t m_updates;           //!< Number of times the neighbor list has been updated
    uint64_t m_forced_updates;    //!< Number of times the neighbor list has been forcibly updated
    uint64_t m_dangerous_updates; //!< Number of dangerous builds counted
    bool m_force_update;          //!< Flag to handle the forcing of neighborlist updates
    bool m_dist_check;            //!< Set to false to disable distance checks (nlist always built
                                  //!< m_rebuild_check_delay steps)
    bool m_has_been_updated_once; //!< True if the neighbor list has been updated at least once

    uint64_t m_last_updated_tstep;          //!< Track the last time step we were updated
    uint64_t m_last_checked_tstep;          //!< Track the last time step we have checked
    bool m_last_check_result;               //!< Last result of rebuild check
    uint64_t m_rebuild_check_delay;         //!< No update checks will be performed until
                                            //!< m_rebuild_check_delay steps after the last one
    std::vector<uint64_t> m_update_periods; //!< Steps between updates
    std::set<std::string> m_exclusions;     //!< Exclusions that have been set

    //! Test if the list needs updating
    bool needsUpdating(uint64_t timestep);

    //! Reallocate internal neighbor list data structures
    void reallocate();

    //! Check the status of the conditions
    bool checkConditions();

    //! Resets the condition status to all zeroes
    virtual void resetConditions();

    //! Grow the exclusions list memory capacity by one row
    void growExclusionList();

    //! Method to be called when the global particle number changes
    void slotGlobalParticleNumberChange()
        {
        m_n_particles_changed = true;
        }

    //! Method to be called when the global bond/angle/dihedral/improper/pair number changes
    void slotGlobalTopologyNumberChange()
        {
        m_topology_changed = true;
        }

    //! Clear all existing exclusions
    void resizeAndClearExclusions();

    //! Add an exclusion for every bond in the ParticleData
    void addExclusionsFromBonds();

    //! Add an exclusion for every bond in the ParticleData
    void addExclusionsFromMeshBonds();

    //! Add exclusions from angles
    void addExclusionsFromAngles();

    //! Add exclusions from dihedrals
    void addExclusionsFromDihedrals();

    //! Add an exclusion for every bond in the ConstraintData
    void addExclusionsFromConstraints();

    //! Add an exclusion for every pair in the ParticleData
    void addExclusionsFromPairs();

    //! Test if an exclusion has been made
    bool isExcluded(unsigned int tag1, unsigned int tag2);

    //! Add an exclusion for every 1,3 pair
    void addOneThreeExclusionsFromTopology();

    //! Add an exclusion for every 1,4 pair
    void addOneFourExclusionsFromTopology();

#ifdef ENABLE_HIP
    GPUPartition m_last_gpu_partition; //!< The partition at the time of the last memory hints
#endif
    };

/// Make the local particle data available to python via zero-copy access.
template<class Output>
class PYBIND11_EXPORT LocalNeighborListData : public LocalDataAccess<Output, NeighborList>
    {
    public:
    LocalNeighborListData(NeighborList& data, size_t N_particles)
        : LocalDataAccess<Output, NeighborList>(data), m_nlist_handle(), m_head_list_handle(),
          m_n_neigh_handle(), m_N_particles(N_particles)
        {
        }

    virtual ~LocalNeighborListData() = default;

    Output getHeadList()
        {
        // This can cause errors when sizeof(unsigned long) != sizeof(size_t)
        return this->template getBuffer<size_t, unsigned long>(m_head_list_handle,
                                                               &NeighborList::getHeadList,
                                                               {m_N_particles},
                                                               false,
                                                               0);
        }

    Output getNNeigh()
        {
        return this->template getBuffer<unsigned int, unsigned int>(m_n_neigh_handle,
                                                                    &NeighborList::getNNeighArray,
                                                                    {m_N_particles},
                                                                    false,
                                                                    0);
        }

    // GhostDataFlag is ignored by this method, but required by the python interface
    Output getNList()
        {
        auto shape
            = std::vector {static_cast<size_t>(this->m_data.getNListArray().getNumElements())};
        return this->template getBuffer<unsigned int, unsigned int>(m_nlist_handle,
                                                                    &NeighborList::getNListArray,
                                                                    shape,
                                                                    false);
        }

    bool isHalfNlist()
        {
        return this->m_data.getStorageMode() == NeighborList::storageMode::half;
        }

    protected:
    void clear()
        {
        m_head_list_handle.reset(nullptr);
        m_nlist_handle.reset(nullptr);
        m_n_neigh_handle.reset(nullptr);
        }

    private:
    std::unique_ptr<ArrayHandle<unsigned int>> m_nlist_handle;
    std::unique_ptr<ArrayHandle<size_t>> m_head_list_handle;
    std::unique_ptr<ArrayHandle<unsigned int>> m_n_neigh_handle;
    size_t m_N_particles;
    };

namespace detail
    {

template<class Output> void export_LocalNeighborListData(pybind11::module& m, std::string name)
    {
    pybind11::class_<LocalNeighborListData<Output>, std::shared_ptr<LocalNeighborListData<Output>>>(
        m,
        name.c_str())
        .def(pybind11::init<NeighborList&, size_t>())
        .def("getNList", &LocalNeighborListData<Output>::getNList)
        .def("getHeadList", &LocalNeighborListData<Output>::getHeadList)
        .def("getNNeigh", &LocalNeighborListData<Output>::getNNeigh)
        .def("isHalfNlist", &LocalNeighborListData<Output>::isHalfNlist)
        .def("enter", &LocalNeighborListData<Output>::enter)
        .def("exit", &LocalNeighborListData<Output>::exit);
    };
    } // end namespace detail

    } // end namespace md
    } // end namespace hoomd

#endif
