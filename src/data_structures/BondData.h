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
// Maintainer: joaander

/*! \file BondData.h
    \brief Declares BondData and related classes
 */

#ifndef __BONDDATA_H__
#define __BONDDATA_H__

#include <vector>
#include <boost/shared_ptr.hpp>
#include <boost/signal.hpp>
#include <boost/utility.hpp>

#ifdef ENABLE_CUDA
#include <cuda_runtime_api.h>
#include "BondData.cuh"
#endif

// forward declaration of ParticleData to avoid circular references
class ParticleData;

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
        void addBond(const Bond& bond);
        
        //! Get the number of bonds
        /*! \return Number of bonds present
        */
        const unsigned int getNumBonds() const
            {
            return (unsigned int)m_bonds.size();
            }
            
        //! Get a given bond
        /*! \param i Bond to access
        */
        const Bond& getBond(unsigned int i) const
            {
            assert(i < m_bonds.size()); return m_bonds[i];
            }
            
        //! Get the number of bond types
        /*! \return Number of bond types in the list of bonds
        */
        const unsigned int getNBondTypes() const
            {
            return m_n_bond_types;
            }
            
        //! Set the type mapping
        void setBondTypeMapping(const std::vector<std::string>& bond_type_mapping);
        
        //! Gets the particle type index given a name
        unsigned int getTypeByName(const std::string &name);
        
        //! Gets the name of a given particle type index
        std::string getNameByType(unsigned int type);
        
# ifdef ENABLE_CUDA
        //! Access the bonds on the GPU
        std::vector<gpu_bondtable_array>& acquireGPU();
#endif
        
    private:
        const unsigned int m_n_bond_types;              //!< Number of bond types
        bool m_bonds_dirty;                             //!< True if the bond list has been changed
        boost::shared_ptr<ParticleData> m_pdata;        //!< Particle Data these bonds belong to
        std::vector<Bond> m_bonds;                      //!< List of bonds on the CPU
        std::vector<std::string> m_bond_type_mapping;   //!< Mapping between bond type indices and names
        
        boost::signals::connection m_sort_connection;   //!< Connection to the resort signal from ParticleData
        
        //! Helper function to set the dirty flag when particles are resorted
        /*! setDirty() just sets the \c m_bonds_dirty flag when partciles are sorted or a bond is added.
            The flag is used to test if the data structure needs updating on the GPU.
        */
        void setDirty()
            {
            m_bonds_dirty = true;
            }
            
#ifdef ENABLE_CUDA
        std::vector<gpu_bondtable_array> m_gpu_bonddata;    //!< List of bonds on the GPU
        uint2 *m_host_bonds;                //!< Host copy of the bond list
        unsigned int *m_host_n_bonds;       //!< Host copy of the number of bonds
        
        //! Helper function to update the bond table on the device
        void updateBondTable();
        
        //! Helper function to reallocate the bond table on the device
        void reallocateBondTable(int height);
        
        //! Helper function to allocate the bond table
        void allocateBondTable(int height);
        
        //! Helper function to free the bond table
        void freeBondTable();
        
        //! Copies the bond table to the device
        void copyBondTable();
        
#endif
    };

//! Exports BondData to python
void export_BondData();

#endif

