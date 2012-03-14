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

// Maintainer: dnlebard

/*! \file DihedralData.h
    \brief Declares DihedralData and related classes
 */

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __DIHEDRALDATA_H__
#define __DIHEDRALDATA_H__

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
#include "DihedralData.cuh"
#endif

#include "GPUVector.h"
#include "ExecutionConfiguration.h"
#include "HOOMDMath.h"

// Sentinel value in dihedral reverse-lookup map for unassigned dihedral type
#define NO_DIHEDRAL 0xffffffff

// forward declaration of ParticleData to avoid circular references
class ParticleData;

//! Handy structure for passing around and initializing the dihedral data
struct SnapshotDihedralData
    {
    //! Constructor
    /*! \param n_dihedrals Number of dihedrals contained in the snapshot
     */
    SnapshotDihedralData(unsigned int n_dihedrals)
        {
        type_id.resize(n_dihedrals);
        dihedrals.resize(n_dihedrals);
        }

    std::vector<unsigned int> type_id;                 //!< Stores type for each bo
    std::vector<uint4> dihedrals;                      //!< .x and .y are tags of t
    std::vector<std::string> type_mapping;             //!< Names of dihedral types
    };

//! Stores a dihedral between four particles
/*! Each dihedral is given an integer \c type from 0 to \c NDihedralTypes-1 and the \em tags
    of the three dihedrald particles.
    \ingroup data_structs
*/
struct Dihedral
    {
    //! Constructs a dihedral
    /*! \param dihedral_type Type index of the dihedral
        \param tag_a Tag of the first particle in the dihedral
        \param tag_b Tag of the second particle in the dihedral
        \param tag_c Tag of the third particle in the dihedral
        \param tag_d Tag of the forth particle in the dihedral
    */
    Dihedral(unsigned int dihedral_type, unsigned int tag_a, unsigned int tag_b, unsigned int tag_c, unsigned int tag_d) : type(dihedral_type), a(tag_a), b(tag_b), c(tag_c), d(tag_d) { }
    unsigned int type;  //!< The type index of the dihedral
    unsigned int a;     //!< The tag of the first particle in the dihedral
    unsigned int b;     //!< The tag of the second particle in the dihedral
    unsigned int c;     //!< The tag of the third particle in the dihedral
    unsigned int d;     //!< The tag of the forth particle in the dihedral
    };

//! Stores all dihedrals in the simulation and mangages the GPU dihedral data structure
/*! DihedralData tracks every dihedral defined in the simulation. On the CPU, dihedrals are stored just
    as a simple vector of Dihedral structs. On the GPU, the list of dihedrals is decomposed into a
    table with every column listing the dihedrals of a single particle: see
    gpu_dihedraltable_array for more info.

    A ParticleData instance owns a single DihedralData which classes such as DihedralForceCompute
    can access for their needs.

    Dihedrals can be dynamically added, although doing this on a per-timestep basis can
    slow performance significantly. For simplicity and convinence, however, the number
    of dihedral types cannot change after initialization.
    \ingroup data_structs
*/
class DihedralData : boost::noncopyable
    {
    public:
        //! Constructs an empty list with no dihedrals
        DihedralData(boost::shared_ptr<ParticleData> pdata, unsigned int n_dihedral_types = 0);
        
        //! Destructor
        ~DihedralData();
        
        //! Add a dihedral to the list
        unsigned int addDihedral(const Dihedral& dihedral);

        //! Remove a dihedral identified by its unique tag from the list
        void removeDihedral(unsigned int tag);
        
        //! Get the number of dihedrals
        /*! \return Number of dihedrals present
        */
        unsigned int getNumDihedrals() const
            {
            return (unsigned int)m_dihedrals.size();
            }
            
        //! Get access to a dihedral
        /*! \param i Dihedral to access
        */
        const Dihedral getDihedral(unsigned int i) const
            {
            assert(i < m_dihedrals.size());
            assert(i < m_dihedral_type.size());
            uint4 dihedral = m_dihedrals[i];
            return Dihedral(m_dihedral_type[i], dihedral.x, dihedral.y, dihedral.z, dihedral.w);
            }

        //! Get dihedral by tag value
        const Dihedral getDihedralByTag(unsigned int tag) const;

        //! Get tag given an id
        unsigned int getDihedralTag(unsigned int id) const;

            
        //! Get the number of dihedral types
        /*! \return Number of dihedral types in the list of dihedrals
        */
        unsigned int getNDihedralTypes() const
            {
            return m_n_dihedral_types;
            }
            
        //! Set the type mapping
        void setDihedralTypeMapping(const std::vector<std::string>& dihedral_type_mapping);
        
        //! Gets the particle type index given a name
        unsigned int getTypeByName(const std::string &name);
        
        //! Gets the name of a given particle type index
        std::string getNameByType(unsigned int type);

        //! Gets the dihedral table
        const GPUVector<uint4>& getDihedralTable()
            {
            return m_dihedrals;
            }

        //! Gets the dihedral types
        const GPUVector<unsigned int>& getDihedralTypes()
            {
            return m_dihedral_type;
            }

        //! Gets the list of dihedral tags
        const GPUVector<unsigned int>& getDihedralTags() const
            {
           return m_tags;
            }

        //! Gets the list of dihedral reverse-lookup tags
        const GPUVector<unsigned int>& getDihedralRTags() const
            {
            return m_dihedral_rtag;
            }

        //! Gets the number of dihedrals array
        const GPUArray<unsigned int>& getNDihedralsArray() const
           {
           return m_n_dihedrals;
           }

        //! Access the dihedrals on the GPU
        const GPUArray<uint4>& getGPUDihedralList();

        //! Access the dihedral atom position list on the GPU
        const GPUArray<uint1>& getDihedralABCD();

        //! Takes a snapshot of the current angle data
        void takeSnapshot(SnapshotDihedralData& snapshot);

        //! Initialize the angle data from a snapshot
        void initializeFromSnapshot(const SnapshotDihedralData& snapshot);
        
    private:
        const unsigned int m_n_dihedral_types;              //!< Number of dihedral types
        bool m_dihedrals_dirty;                             //!< True if the dihedral list has been changed
        boost::shared_ptr<ParticleData> m_pdata;            //!< Particle Data these dihedrals belong to
        boost::shared_ptr<const ExecutionConfiguration> exec_conf;  //!< Execution configuration for CUDA context
        GPUVector<uint4> m_dihedrals;                       //!< List of dihedrals
        GPUVector<unsigned int> m_dihedral_type;            //!< List of dihedral types
        GPUVector<unsigned int> m_tags;                     //!< Reverse lookup table for tags
        std::stack<unsigned int> m_deleted_tags;            //!< Stack for deleted dihedral tags
        GPUVector<unsigned int> m_dihedral_rtag;            //!< Map to support lookup of dihedrals by tag
        std::vector<std::string> m_dihedral_type_mapping;   //!< Mapping between dihedral type indices and names
        
        boost::signals::connection m_sort_connection;       //!< Connection to the resort signal from ParticleData
        
        
        //! Helper function to set the dirty flag when particles are resorted
        /*! setDirty() just sets the \c m_dihedrals_dirty flag when partciles are sorted or a dihedral is added.
            The flag is used to test if the data structure needs updating on the GPU.
        */
        void setDirty()
            {
            m_dihedrals_dirty = true;
            }
            
        GPUArray<uint4> m_gpu_dihedral_list;                    //!< List of dihedrals on the GPU (3atoms of a,b,c, or d, plus the type)
        GPUArray<uint1> m_dihedrals_ABCD;                        //!< List of atom positions in the dihedral
        GPUArray<unsigned int> m_n_dihedrals;                    //!< Number of dihedrals

#ifdef ENABLE_CUDA
        //! Helper function to update the dihedral table on the device
        void updateDihedralTableGPU();
#endif

        //! Helper function to update the GPU dihedral table
        void updateDihedralTable();

        //! Helper function to allocate the dihedral table
        void allocateDihedralTable(int height);
        
    };

//! Exports DihedralData to python
void export_DihedralData();

#endif

