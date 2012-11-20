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

/*! \file BondData.h
    \brief Declares BondData and related classes
 */

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __BONDDATA_H__
#define __BONDDATA_H__

#include <vector>
#include <stack>

// fall back on compiler tr1/unordered_map if boost doesn't have it
#include <boost/version.hpp>
#if (BOOST_VERSION <= 103600)
#include <tr1/unordered_map>
#else
#include <boost/tr1/unordered_map.hpp>
#endif

#include <boost/shared_ptr.hpp>
#include <boost/signal.hpp>
#include <boost/utility.hpp>

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include "BondData.cuh"
#endif

#include "GPUVector.h"
#include "GPUFlags.h"
#include "ExecutionConfiguration.h"
#include "HOOMDMath.h"

// Sentinel value in bond reverse-lookup map for unassigned bond tags
#define BOND_NOT_LOCAL 0xffffffff

// forward declaration of ParticleData to avoid circular references
class ParticleData;
// forward declaration of Profiler
class Profiler;

//! Stores a bond between two particles
/*! Each bond is given an integer \c type from 0 to \c NBondTypes-1 and the \em tags
    of the two bonded particles.
    \ingroup data_structs
*/
struct Bond
    {
    //! Constructs a bond
    /*! \param bond_type Type index of the bond
        \param tag_a Tag of the first particle in the bond
        \param tag_b Tag of the second particle in the bond
    */
    Bond(unsigned int bond_type, unsigned int tag_a, unsigned int tag_b) : type(bond_type), a(tag_a), b(tag_b) { }
    unsigned int type;  //!< The type index of the bond
    unsigned int a;     //!< The tag of the first particle in the bond
    unsigned int b;     //!< The tag of the second particle in the bond
    };

//! Handy structure for passing around and initializing the bond data
struct SnapshotBondData
    {
    //! Constructor
    /*! \param n_bonds Number of bonds contained in the snapshot
     */
    SnapshotBondData(unsigned int n_bonds)
        {
        type_id.resize(n_bonds);
        bonds.resize(n_bonds);

        // provide default type mapping
        type_mapping.push_back("polymer");
        }

    std::vector<unsigned int> type_id;             //!< Stores type for each bond
    std::vector<uint2> bonds;                      //!< .x and .y are tags of the two particles in the bond
    std::vector<std::string> type_mapping;         //!< Names of bond types
    };

//! Definition of a buffer element
struct bond_element
    {
    uint2 bond;                //!< Member tags of the bond
    unsigned int type;         //!< Type of the bond
    unsigned int tag;          //!< Unique bond identifier
    };


//! Stores all bonds in the simulation and mangages the GPU bond data structure
/*! BondData tracks every bond defined in the simulation. On the CPU, bonds are stored just
    as a simple vector of Bond structs. On the GPU, the list of bonds is decomposed into a
    table with every column listing the bonds of a single particle: see
    gpu_bondtable_array for more info.

    Bonds can be dynamically added, although doing this on a per-timestep basis can
    slow performance significantly. For simplicity and convinence, however, the number
    of bond types cannot change after initialization.
    \ingroup data_structs
*/
class BondData : boost::noncopyable
    {
    public:
        //! Constructs an empty list with no bonds
        BondData(boost::shared_ptr<ParticleData> pdata, unsigned int n_bond_types);
        
        //! Destructor
        ~BondData();
        
        //! Add a bond to the list
        unsigned int addBond(const Bond& bond);

        //! Remove a bond identified by its unique tag from the list
        void removeBond(unsigned int tag);

        //! Get the number of bonds
        /*! \return Number of bonds present
        */
        unsigned int getNumBonds() const
            {
            return (unsigned int)m_bonds.size();
            }

        //! Get the global number of bonds
        /*! \return Global number of bonds
        */
        unsigned int getNumBondsGlobal() const
            {
            return m_num_bonds_global;
            }
            
        //! Get a given bond
        /*! \param i Bond to access
        */
        const Bond getBond(unsigned int i) const
            {
            assert(i < m_bonds.size());
            assert(i < m_bond_type.size());
            uint2 bond = m_bonds[i];
            return Bond(m_bond_type[i], bond.x, bond.y);
            }

        //! Get bond by tag value
        const Bond getBondByTag(unsigned int tag) const;

        //! Get tag given an id
        unsigned int getBondTag(unsigned int id) const;

        //! Get the number of bond types
        /*! \return Number of bond types in the list of bonds
        */
        unsigned int getNBondTypes() const
            {
            return m_n_bond_types;
            }
            
        //! Set the type mapping
        void setBondTypeMapping(const std::vector<std::string>& bond_type_mapping);
        
        //! Gets the particle type index given a name
        unsigned int getTypeByName(const std::string &name);
        
        //! Gets the name of a given particle type index
        std::string getNameByType(unsigned int type);

        //! Unpack a buffer with new bonds to be added, and remove bonds according to a mask
        /*! \param num_bonds Number of bonds in the buffer
         *  \param num_remove_bonds Number of bonds to be removed
         *  \param buf The buffer containing the bond data
         *  \param remove_mask A mask that indicates whether the bond needs to be removed
         */
        void unpackRemoveBonds(unsigned int num_add_bonds,
                               unsigned int num_remove_bonds,
                               const GPUArray<bond_element>& buf,
                               const GPUArray<unsigned int>& remove_mask);

        //! Requests bonds to be removed from the bond table
        /*! \param num_bonds The number of empty bonds to be removed
            \post The internal data structures are resized to reflect the new number of bonds.
                  No memory is usually released. 
            \warning It is the responsibility of the caller t
            to the GPUArrays.
         */
        void shrinkBondTable(unsigned int num_bonds)
            {
            assert(m_bonds.size() == m_bond_type.size());
            assert(m_bonds.size() == m_tags.size());

            unsigned int new_size = m_bonds.size() + num_bonds;
            m_bonds.resize(new_size);
            m_bond_type.resize(new_size);
            m_tags.resize(new_size);
            }

        
        //! Gets the bond table
        const GPUVector<uint2>& getBondTable()
            {
            return m_bonds;
            }

        //! Gets the bond types
        const GPUVector<unsigned int>& getBondTypes()
            {
            return m_bond_type;
            }


        //! Gets the list of bond tags
        const GPUVector<unsigned int>& getBondTags() const
            {
            return m_tags;
            }

        //! Gets the list of bond reverse-lookup tags
        const GPUVector<unsigned int>& getBondRTags() const
            {
            return m_bond_rtag;
            }

        //! Gets the number of bonds array
        const GPUArray<unsigned int>& getNBondsArray() const
            {
            return m_n_bonds;
            }

        //! Gets the number of ghost bonds array
        const GPUArray<unsigned int>& getNGhostBondsArray() const
            {
            return m_n_ghost_bonds;
            }

        //! Access the bonds on the GPU
        /*! \param ghost True if the ghost bond list should be precomputed
         */
        const GPUArray<uint2>& getGPUBondList(bool ghost=false)
            {
            checkUpdateBondList(ghost);
            return m_gpu_bondlist;
            }
       
        //! Access the ghost bond list on the GPU
        const GPUArray<uint2> & getGPUGhostBondList()
            {
            checkUpdateBondList(true);
            return m_gpu_ghost_bondlist;
            }

        //! Takes a snapshot of the current bond data
        void takeSnapshot(SnapshotBondData& snapshot);

        //! Initialize the bond data from a snapshot
        void initializeFromSnapshot(const SnapshotBondData& snapshot);

        //! Set the profiler
        /*! \param prof The profiler
         */
        void setProfiler(boost::shared_ptr<Profiler> prof)
            {
            m_prof = prof;
            }

        //! Helper function to reallocate the GPU bond table
        void reallocate();

    private:
        const unsigned int m_n_bond_types;              //!< Number of bond types
        bool m_bonds_dirty;                             //!< True if the bond list has been changed
        bool m_ghost_bonds_dirty;                       //!< True if the ghost bond list has been changed
        boost::shared_ptr<ParticleData> m_pdata;        //!< Particle Data these bonds belong to
        boost::shared_ptr<const ExecutionConfiguration> exec_conf;  //!< Execution configuration for CUDA context
        GPUVector<uint2> m_bonds;                       //!< List of bonds (x: tag a, y: tag b)
        GPUVector<unsigned int> m_bond_type;            //!< List of bond types
        GPUVector<unsigned int> m_tags;                 //!< Bond tags
        std::stack<unsigned int> m_deleted_tags;        //!< Stack for deleted bond tags
        GPUVector<unsigned int> m_bond_rtag;            //!< Map to support lookup of bonds by tag
        std::vector<std::string> m_bond_type_mapping;   //!< Mapping between bond type indices and names
        
        boost::signals::connection m_sort_connection;   //!< Connection to the resort signal from ParticleData
        boost::signals::connection m_max_particle_num_change_connection; //!< Connection to maximum particle number change signal
        boost::signals::connection m_ghost_particle_num_change_connection; //!< Connection to ghost particle number change signal

    
        boost::shared_ptr<const ExecutionConfiguration> m_exec_conf;    //!< execution configuration for working with CUDA

        //! Helper function to set the dirty flag when particles are resorted
        /*! setDirty() just sets the \c m_bonds_dirty flag when partciles are sorted or a bond is added.
            The flag is used to test if the data structure needs updating on the GPU.
        */
        void setDirty()
            {
            m_bonds_dirty = true;
            m_ghost_bonds_dirty = true;
            }
            
        GPUArray<uint2> m_gpu_bondlist;         //!< List of bonds on the GPU
        GPUArray<uint2> m_gpu_ghost_bondlist;   //!< List of ghost bonds on the GPU
        GPUArray<unsigned int> m_n_bonds;       //!< Array of the number of bonds
        GPUArray<unsigned int> m_n_ghost_bonds; //!< Array of the number of ghost bonds
        bool m_ghost_bond_table_allocated;      //!< True if ghost bond table has been allocated
#ifdef ENABLE_CUDA
        unsigned int m_max_bond_num;            //!< Maximum bond number
        unsigned int m_max_ghost_bond_num;      //!< Maximum ghost bond number
        GPUFlags<unsigned int> m_condition;     //!< Condition variable for bond counting
#endif
        unsigned int m_num_bonds_global;        //!< Total number of bonds on all processors

#ifdef ENABLE_CUDA
        GPUFlags<unsigned int> m_duplicate_recv_bonds; //!< Number of duplicate bonds received
        GPUArray<unsigned int> m_n_fetch_bond;  //!< Temporary counter for filling the bond table
        GPUVector<unsigned char> m_recv_bond_active;   //!< Per-bond flag for buffers (1= bond is retained, 0 = duplicate)
        bool m_buffers_initialized;             //!< True if internal buffers have been initialized
#endif 

        boost::shared_ptr<Profiler> m_prof; //!< The profiler to use
#ifdef ENABLE_CUDA
        //! Helper function to update the bond table on the device
        void updateBondTableGPU(bool ghost);
#endif

        //! Helper function to check and update the GPU bondlist
        void checkUpdateBondList(bool ghost=false);

        //! Helper function to update the GPU bond table
        void updateBondTable(bool ghost);

        //! Helper function to allocate the bond table
        void allocateBondTable(int height);

        //! Helper function to allocate the ghost bond table
        void allocateGhostBondTable(int height);

#ifdef ENABLE_CUDA
        //! Helper function to unpack and remove bonds on the GPU
        void unpackRemoveBondsGPU(unsigned int num_add_bonds,
                               unsigned int num_remove_bonds,
                               const GPUArray<bond_element>& buf,
                               const GPUArray<unsigned int>& remove_mask);
#endif
    };

//! Exports BondData to python
void export_BondData();

#endif

