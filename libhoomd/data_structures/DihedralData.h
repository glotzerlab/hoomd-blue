/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
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

// $Id$
// $URL$
// Maintainer: akohlmey

/*! \file DihedralData.h
    \brief Declares DihedralData and related classes
 */

#ifndef __DIHEDRALDATA_H__
#define __DIHEDRALDATA_H__

#include <vector>
#include <boost/shared_ptr.hpp>
#include <boost/signal.hpp>
#include <boost/utility.hpp>

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include "DihedralData.cuh"
#endif

#include "ExecutionConfiguration.h"

// forward declaration of ParticleData to avoid circular references
class ParticleData;

//! Stores an dihedral between four particles
/*! Each dihedral is given an integer \c type from 0 to \c NDihedralTypes-1 and the \em tags
    of the three dihedrald particles.
    \ingroup data_structs
*/
struct Dihedral
    {
    //! Constructs an dihedral
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
        
        //! Add an dihedral to the list
        void addDihedral(const Dihedral& dihedral);
        
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
        const Dihedral& getDihedral(unsigned int i) const
            {
            assert(i < m_dihedrals.size()); return m_dihedrals[i];
            }
            
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
        
# ifdef ENABLE_CUDA
        //! Access the dihedrals on the GPU
        gpu_dihedraltable_array& acquireGPU();
        
#endif
        
        
    private:
        const unsigned int m_n_dihedral_types;              //!< Number of dihedral types
        bool m_dihedrals_dirty;                             //!< True if the dihedral list has been changed
        boost::shared_ptr<ParticleData> m_pdata;            //!< Particle Data these dihedrals belong to
        std::vector<Dihedral> m_dihedrals;                  //!< List of dihedrals on the CPU
        std::vector<std::string> m_dihedral_type_mapping;   //!< Mapping between dihedral type indices and names
        
        boost::signals::connection m_sort_connection;       //!< Connection to the resort signal from ParticleData
        
        boost::shared_ptr<const ExecutionConfiguration> exec_conf;  //!< Execution configuration for CUDA context

        
        //! Helper function to set the dirty flag when particles are resorted
        /*! setDirty() just sets the \c m_dihedrals_dirty flag when partciles are sorted or an dihedral is added.
            The flag is used to test if the data structure needs updating on the GPU.
        */
        void setDirty()
            {
            m_dihedrals_dirty = true;
            }
            
#ifdef ENABLE_CUDA
        gpu_dihedraltable_array m_gpu_dihedraldata;    //!< List of dihedrals on the GPU
        uint4 *m_host_dihedrals;             //!< Host copy of the dihedral list (3atoms of a,b,c, or d, plus the type)
        uint1 *m_host_dihedralsABCD;         //!< Host copy of the dihedralABCD list
        unsigned int *m_host_n_dihedrals;    //!< Host copy of the number of dihedrals
        
        /*! \enum dihedralABCD tells if the Dihedral is on the a,b,c, or d atom
        */
        enum dihedralABCD
            {
            a_atom = 0, //!< atom is the a particle in an a-b-c-d quartet
            b_atom = 1, //!< atom is the b particle in an a-b-c-d quartet
            c_atom = 2, //!< atom is the c particle in an a-b-c-d quartet
            d_atom = 3  //!< atom is the d particle in an a-b-c-d quartet
            };
            
        //! Helper function to update the dihedral table on the device
        void updateDihedralTable();
        
        //! Helper function to reallocate the dihedral table on the device
        void reallocateDihedralTable(int height);
        
        //! Helper function to allocate the dihedral table
        void allocateDihedralTable(int height);
        
        //! Helper function to free the dihedral table
        void freeDihedralTable();
        
        //! Copies the dihedral table to the device
        void copyDihedralTable();
        
#endif
    };

//! Exports DihedralData to python
void export_DihedralData();

#endif

