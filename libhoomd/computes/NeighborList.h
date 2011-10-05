/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: joaander

#include <boost/shared_ptr.hpp>
#include <boost/signals.hpp>
#include <vector>

#include "Compute.h"
#include "GPUArray.h"
#include "Index1D.h"

/*! \file NeighborList.h
    \brief Declares the NeighborList class
*/

#ifndef __NEIGHBORLIST_H__
#define __NEIGHBORLIST_H__

//! Computes a Neibhorlist from the particles
/*! \b Overview:

    A particle \c i is a neighbor of particle \c j if the distance between
    particle them is less than or equal to \c r_cut. The neighborlist for a given particle
    \c i includes all of these neighbors at a minimum. Other particles particles are included
    in the list: those up to \c r_max which includes a buffer distance so that the neighbor list
    doesn't need to be updated every step.

    There are two ways of storing this information. One is to store only half of the
    neighbors (only those with i < j), and the other is to store all neighbors. There are
    potential tradeoffs between number of computations and memory access complexity for
    each method. NeighborList supports both of these modes via a switch: setStorageMode();

    Some classes with either setting, full or half, but they are faster with the half setting. However,
    others may require that the neighbor list storage mode is set to full.
    
    <b>Data access:</b>
    
    Up to Nmax neighbors can be stored for each particle. Data is stored in a 2D matrix array in memory. Each element
    in the matrix stores the index of the neighbor with the highest bits reserved for flags. An indexer for accessing
    elements can be gotten with getNlistIndexer() and the array itself can be accessed with getNlistArray().

    The number of neighbors for each particle is stored in an auxilliary array accessed with getNNeighArray().
    
     - <code>jf = nlist[nlist_indexer(i,n)]</code> is the index of neighbor \a n of particle \a i, where \a n can vary from
       0 to <code>n_neigh[i] - 1</code>
    
    \a jf includes flags in the highest bits. The format and use of these flags are yet to be determined.
    
    \b Filtering:
    
    By default, a neighbor list includes all particles within a single cutoff distance r_cut. Various filters can be
    applied to remove unwanted neighbors from the list.
     - setFilterBody() prevents two particles of the same body from being neighbors
     - setFilterRcutType() enables individual r_cut values for each pair of particle types
     - setFilterDiameter() enables slj type diameter filtering (TODO: need to specify exactly what this does)
    
    \b Algorithms:

    This base class supplys a dumb O(N^2) algorithm for generating this list. It is very
    slow, but functional. Derived classes implement O(N) efficient straetegies using the CellList.

    <b>Needs updage check:</b>

    When compute() is called, the neighbor list is updated, but only if it needs to be. Checks
    are performed to see if any particle has moved more than half of the buffer distance, and
    only then is the list actually updated. This check can even be avoided for a number of time
    steps by calling setEvery(). If the caller wants to forces a full update, forceUpdate()
    can be called before compute() to do so. Note that if the particle data is resorted,
    an update is automatically forced.
    
    \b Exclusions:
    
    Exclusions are stored in \a ex_list, a data structure similar in structure to \a nlist, except this time exclusions
    are stored. User-specified exclusions are stored by tag and translated to indices whenever a particle sort occurs
    (updateExListIdx()). If any exclusions are set, filterNlist() is called after buildNlist(). filterNlist() loops
    through the neighbor list and removes any particles that are excluded. This allows an arbitrary number of exclusions
    to be processed without slowing the performance of the buildNlist() step itself.
    
    <b>Overvlow handling:</b>
    For easy support of derived GPU classes to implement overvlow detectio the overflow condition is storeed in the
    GPUArray \a d_conditions.
    
     - 0: Maximum nlist size (implementations are free to write to this element only in overflow conditions if they
          choose.)
     - Further indices may be added to handle other conditions at a later time.

    Condition flags are to be set during the buildNlist() call and will be checked by compute() which will then 
    take the appropriate action.
    
    \ingroup computes
*/
class NeighborList : public Compute
    {
    public:
        //! Simple enum for the storage modes
        enum storageMode
            {
            half,   //!< Only neighbors i,j are stored where i < j
            full    //!< All neighbors are stored
            };
        
        //! Constructs the compute
        NeighborList(boost::shared_ptr<SystemDefinition> sysdef, Scalar r_cut, Scalar r_buff);
        
        //! Destructor
        virtual ~NeighborList();
        
        //! \name Set parameters
        // @{
        
        //! Change the cuttoff radius
        virtual void setRCut(Scalar r_cut, Scalar r_buff);
        
        //! Change how many timesteps before checking to see if the list should be rebuilt
        /*! \param every Number of time steps to wait before beignning to check if particles have moved a sufficient distance
                   to require a neighbor list upate.
        */
        void setEvery(unsigned int every)
            {
            m_every = every;
            forceUpdate();
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
        
        // @}
        //! \name Statistics
        // @{
        
        //! Print statistics on the neighborlist
        virtual void printStats();
        
        //! Clear the count of updates the neighborlist has performed
        virtual void resetStats();

        //! Gets the shortest rebuild period this nlist has experienced since a call to resetStats
        unsigned int getSmallestRebuild();
        
        // @}
        //! \name Get data
        // @{
        
        //! Get the number of neighbors array
        const GPUArray<unsigned int>& getNNeighArray()
            {
            return m_n_neigh;
            }
        
        //! Get the neighbor list
        const GPUArray<unsigned int>& getNListArray()
            {
            return m_nlist;
            }
        
        //! Get the number of exclusions array
        const GPUArray<unsigned int>& getNExArray()
            {
            return m_n_ex_idx;
            }
        
         //! Get the exclusion list
         const GPUArray<unsigned int>& getExListArray()
            {
            return m_ex_list_idx;
            }

        //! Get the neighbor list indexer
        /*! \note Do not save indexers across calls. Get a new indexer after every call to compute() - they will
            change.
        */
        const Index2D& getNListIndexer()
            {
            return m_nlist_indexer;
            }
        
        const Index2D& getExListIndexer()
            {
            return m_ex_list_indexer;
            }

        bool getExclusionsSet()
            {
            return m_exclusions_set;
            }

        //! Gives an estimate of the number of nearest neighbors per particle
        virtual Scalar estimateNNeigh();
        
        // @}
        //! \name Handle exclusions
        // @{
        
        //! Exclude a pair of particles from being added to the neighbor list
        void addExclusion(unsigned int tag1, unsigned int tag2);
        
        //! Clear all existing exclusions
        void clearExclusions();
        
        //! Collect some statistics on exclusions.
        void countExclusions();
        
        //! Add an exclusion for every bond in the ParticleData
        void addExclusionsFromBonds();
        
        //! Add exclusions from angles
        void addExclusionsFromAngles();
        
        //! Add exclusions from dihedrals
        void addExclusionsFromDihedrals();
        
        //! Test if an exclusion has been made
        bool isExcluded(unsigned int tag1, unsigned int tag2);
        
        //! Add an exclusion for every 1,3 pair
        void addOneThreeExclusionsFromTopology();
        
        //! Add an exclusion for every 1,4 pair
        void addOneFourExclusionsFromTopology();
        
        //! Enable/disable body filtering
        virtual void setFilterBody(bool filter_body)
            {
            // only set the body exclusions if there are bodies in the rigid data, otherwise it just wastes time
            if (m_sysdef->getRigidData()->getNumBodies() > 0)
                {
                m_filter_body = filter_body;
                forceUpdate();
                }
            }
        
        //! Enable/disable diameter filtering
        virtual void setFilterDiameter(bool filter_diameter)
            {
            m_filter_diameter = filter_diameter;
            forceUpdate();
            }
        
        //! Set the maximum diameter to use in computing neighbor lists
        virtual void setMaximumDiameter(Scalar d_max)
            {
            m_d_max = d_max;
            forceUpdate();
            }
        
        // @}
        
        //! Computes the NeighborList if it needs updating
        void compute(unsigned int timestep);
        
        //! Benchmark the neighbor list
        virtual double benchmark(unsigned int num_iters);
        
        //! Forces a full update of the list on the next call to compute()
        void forceUpdate()
            {
            m_force_update = true;
            }
            
    protected:
        Scalar m_r_cut;             //!< The cuttoff radius
        Scalar m_r_buff;            //!< The buffer around the cuttoff
        Scalar m_d_max;             //!< The maximum diameter of any particle in the system (or greater)
        bool m_filter_body;         //!< Set to true if particles in the same body are to be filtered
        bool m_filter_diameter;     //!< Set to true if particles are to be filtered by diameter (slj style)
        storageMode m_storage_mode; //!< The storage mode
        
        Index2D m_nlist_indexer;             //!< Indexer for accessing the neighbor list
        GPUArray<unsigned int> m_nlist;      //!< Neighbor list data
        GPUArray<unsigned int> m_n_neigh;    //!< Number of neighbors for each particle
        GPUArray<Scalar4> m_last_pos;        //!< coordinates of last updated particle positions
        unsigned int m_Nmax;                 //!< Maximum number of neighbors that can be held in m_nlist
        GPUArray<unsigned int> m_conditions; //!< Condition flags set during the buildNlist() call
        
        GPUArray<unsigned int> m_ex_list_tag;  //!< List of excluded particles referenced by tag
        GPUArray<unsigned int> m_ex_list_idx;  //!< List of excluded particles referenced by index
        GPUArray<unsigned int> m_n_ex_tag;     //!< Number of exclusions for a given particle tag
        GPUArray<unsigned int> m_n_ex_idx;     //!< Number of exclusions for a given particle index
        Index2D m_ex_list_indexer;             //!< Indexer for accessing the exclusion list
        bool m_exclusions_set;                 //!< True if any exclusions have been set

        boost::signals::connection m_sort_connection;   //!< Connection to the ParticleData sort signal
        
        //! Performs the distance check
        virtual bool distanceCheck();
        
        //! Updates the previous position table for use in the next distance check
        virtual void setLastUpdatedPos();
        
        //! Builds the neighbor list
        virtual void buildNlist(unsigned int timestep);
        
        //! Updates the idx exlcusion list
        virtual void updateExListIdx();
        
        //! Filter the neighbor list of excluded particles
        virtual void filterNlist();
        
    private:
        int64_t m_updates;              //!< Number of times the neighbor list has been updated
        int64_t m_forced_updates;       //!< Number of times the neighbor list has been foribly updated
        int64_t m_dangerous_updates;    //!< Number of dangerous builds counted
        bool m_force_update;            //!< Flag to handle the forcing of neighborlist updates
        
        unsigned int m_last_updated_tstep; //!< Track the last time step we were updated
        unsigned int m_every; //!< No update checks will be performed until m_every steps after the last one
        vector<unsigned int> m_update_periods;    //!< Steps between updates
        
        //! Test if the list needs updating
        bool needsUpdating(unsigned int timestep);
        
        //! Allocate the nlist array
        void allocateNlist();
        
        //! Check the status of the conditions
        bool checkConditions();

        //! Resets the condition status
        void resetConditions();
        
        //! Grow the exclusions list memory capacity by one row
        void growExclusionList();
    };

//! Exports NeighborList to python
void export_NeighborList();

#endif

