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

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include "BondData.h"
#include "ParticleData.h"

#include <boost/python.hpp>
using namespace boost::python;

#include <boost/bind.hpp>
using namespace boost;

#include <iostream>
#include <stdexcept>
using namespace std;

/*! \file BondData.cc
    \brief Defines BondData.
 */

/*! \param pdata ParticleData these bonds refer into
    \param n_bond_types Number of bond types in the list
*/
BondData::BondData(boost::shared_ptr<ParticleData> pdata, unsigned int n_bond_types) 
    : m_n_bond_types(n_bond_types), m_bonds_dirty(false), m_pdata(pdata), m_last_added_tag(NO_BOND), exec_conf(m_pdata->getExecConf())
    {
    assert(pdata);
    
    // attach to the signal for notifications of particle sorts
    m_sort_connection = m_pdata->connectParticleSort(bind(&BondData::setDirty, this));
    
    // offer a default type mapping
    for (unsigned int i = 0; i < n_bond_types; i++)
        {
        char suffix[2];
        suffix[0] = 'A' + i;
        suffix[1] = '\0';
        
        string name = string("bond") + string(suffix);
        m_bond_type_mapping.push_back(name);
        }
        
#ifdef ENABLE_CUDA
    // get the execution configuration
    boost::shared_ptr<const ExecutionConfiguration> exec_conf = m_pdata->getExecConf();
    
    // init pointers
    m_host_bonds = NULL;
    m_host_n_bonds = NULL;
    m_gpu_bonddata.bonds = NULL;
    m_gpu_bonddata.n_bonds = NULL;
    m_gpu_bonddata.height = 0;
    m_gpu_bonddata.pitch = 0;
        
    // allocate memory on the GPU if there is a GPU in the execution configuration
    if (exec_conf->isCUDAEnabled())
        {
        allocateBondTable(1);
        }
#endif
    }

BondData::~BondData()
    {
    m_sort_connection.disconnect();
    
#ifdef ENABLE_CUDA
    if (exec_conf->isCUDAEnabled())
        {
        freeBondTable();
        }
#endif
    }

/*! \post A bond between particles specified in \a bond is created.

    \note Each bond should only be specified once! There are no checks to prevent one from being
    specified more than once, and doing so would result in twice the force and twice the energy.
    For a bond between \c i and \c j, only call \c addBond(Bond(type,i,j)). Do NOT additionally call
    \c addBond(Bond(type,j,i)). The first call is sufficient to include the forces on both particle
    \c i and \c j.

    \note If a bond is added with \c type=49, then there must be at least 50 bond types (0-49) total,
    even if not all values are used. So bonds should be added with small contiguous types.
    \param bond The Bond to add to the list
 */
void BondData::addBond(const Bond& bond)
    {
    // check for some silly errors a user could make
    if (bond.a >= m_pdata->getN() || bond.b >= m_pdata->getN())
        {
        cerr << endl << "***Error! Particle tag out of bounds when attempting to add bond: " << bond.a << "," << bond.b << endl << endl;
        throw runtime_error("Error adding bond");
        }
        
    if (bond.a == bond.b)
        {
        cerr << endl << "***Error! Particle cannot be bonded to itself! " << bond.a << "," << bond.b << endl << endl;
        throw runtime_error("Error adding bond");
        }
        
    // check that the type is within bouds
    if (bond.type+1 > m_n_bond_types)
        {
        cerr << endl << "***Error! Invalid bond type! " << bond.type << ", the number of types is " << m_n_bond_types << endl << endl;
        throw runtime_error("Error adding bond");
        }

    // first check if we can recycle a deleted tag
    unsigned int tag = 0;
    if (m_deleted_tags.size())
        {
        tag = m_deleted_tags.top();
        m_deleted_tags.pop();
        }
    // Otherwise, generate a new tag
    else tag = m_bonds.size();

    assert(tag <= m_deleted_tags.size() + m_bonds.size());

    m_last_added_tag = tag;

    // add mapping pointing to last element in m_bonds
    m_bond_map.insert(std::pair<unsigned int, unsigned int>(tag,m_bonds.size()));

    m_bonds.push_back(bond);
    m_tags.push_back(tag);

    m_bonds_dirty = true;
    }

//! Get a bond by tag value
/*! \param tag tag of the bond to access
 */
const Bond& BondData::getBondByTag(unsigned int tag) const
    {
    // Find position of bond in bonds list
    unsigned int id;
    boost::unordered_map<unsigned int, unsigned int>::const_iterator it;
    it = m_bond_map.find(tag);
    if (it == m_bond_map.end())
        {
        cerr << endl << "***Error! Trying to get bond tag " << tag << " which does not exist!" << endl << endl;
       throw runtime_error("Error getting bond");
       }
    id = it->second;
    return m_bonds[id];
    }

//! Remove a bond identified by unique tag value
/*! \param tag tag of bond to remove
 * \note Bond removal changes the order of m_bonds. If a hole in the bond list
 * is generated, the last bond in the list is moved up to fill that hole.
 */
void BondData::removeBond(unsigned int tag)
    {
    // Find position of bond in bonds list
    unsigned int id;
    boost::unordered_map<unsigned int, unsigned int>::iterator it;
    it = m_bond_map.find(tag);
    if (it == m_bond_map.end())
        {
        cerr << endl << "***Error! Trying to remove bond tag " << tag << " which does not exist!" << endl << endl;
       throw runtime_error("Error removing bond");
       }
    id = it->second;

    // delete from map
    m_bond_map.erase(it);

    // If the bond is in the middle of the list, move the last element to
    // to the position of the removed element
    if (id < (m_bonds.size()-1))
        {
        m_bonds[id] = m_bonds[m_bonds.size()-1];
        unsigned int last_tag = m_tags[m_bonds.size()-1];
        m_bond_map[last_tag] = id;
        m_tags[id] = last_tag;
        }
    // delete last element
    m_bonds.pop_back();
    m_tags.pop_back();

    // maintain a stack of deleted bond tags for future recycling
    m_deleted_tags.push(tag);

    m_bonds_dirty = true;
    }

/*! \param bond_type_mapping Mapping array to set
    \c bond_type_mapping[type] should be set to the name of the bond type with index \c type.
    The vector \b must have \c n_bond_types elements in it.
*/
void BondData::setBondTypeMapping(const std::vector<std::string>& bond_type_mapping)
    {
    assert(bond_type_mapping.size() == m_n_bond_types);
    m_bond_type_mapping = bond_type_mapping;
    }


/*! \param name Type name to get the index of
    \return Type index of the corresponding type name
    \note Throws an exception if the type name is not found
*/
unsigned int BondData::getTypeByName(const std::string &name)
    {
    // search for the name
    for (unsigned int i = 0; i < m_bond_type_mapping.size(); i++)
        {
        if (m_bond_type_mapping[i] == name)
            return i;
        }
        
    cerr << endl << "***Error! Bond type " << name << " not found!" << endl;
    throw runtime_error("Error mapping type name");
    return 0;
    }

/*! \param type Type index to get the name of
    \returns Type name of the requested type
    \note Type indices must range from 0 to getNTypes or this method throws an exception.
*/
std::string BondData::getNameByType(unsigned int type)
    {
    // check for an invalid request
    if (type >= m_n_bond_types)
        {
        cerr << endl << "***Error! Requesting type name for non-existant type " << type << endl << endl;
        throw runtime_error("Error mapping type name");
        }
        
    // return the name
    return m_bond_type_mapping[type];
    }

#ifdef ENABLE_CUDA

/*! Updates the bond data on the GPU if needed and returns the data structure needed to access it.
*/
gpu_bondtable_array& BondData::acquireGPU()
    {
    if (m_bonds_dirty)
        {
        updateBondTable();
        m_bonds_dirty = false;
        }
    return m_gpu_bonddata;
    }


/*! \post The bond tag data added via addBond() is translated to bonds based
    on particle index for use in the GPU kernel. This new bond table is then uploaded
    to the device.
*/
void BondData::updateBondTable()
    {
    assert(m_host_n_bonds);
    assert(m_host_bonds);
    
    // count the number of bonds per particle
    // start by initializing the host n_bonds values to 0
    memset(m_host_n_bonds, 0, sizeof(unsigned int) * m_pdata->getN());
    
    // loop through the particles and count the number of bonds based on each particle index
    ParticleDataArraysConst arrays = m_pdata->acquireReadOnly();
    for (unsigned int cur_bond = 0; cur_bond < m_bonds.size(); cur_bond++)
        {
        unsigned int tag1 = m_bonds[cur_bond].a;
        unsigned int tag2 = m_bonds[cur_bond].b;
        int idx1 = arrays.rtag[tag1];
        int idx2 = arrays.rtag[tag2];
        
        m_host_n_bonds[idx1]++;
        m_host_n_bonds[idx2]++;
        }
        
    // find the maximum number of bonds
    unsigned int num_bonds_max = 0;
    for (unsigned int i = 0; i < arrays.nparticles; i++)
        {
        if (m_host_n_bonds[i] > num_bonds_max)
            num_bonds_max = m_host_n_bonds[i];
        }
        
    // re allocate memory if needed
    if (num_bonds_max > m_gpu_bonddata.height)
        {
        reallocateBondTable(num_bonds_max);
        }
        
    // now, update the actual table
    // zero the number of bonds counter (again)
    memset(m_host_n_bonds, 0, sizeof(unsigned int) * m_pdata->getN());
    
    // loop through all bonds and add them to each column in the list
    int pitch = m_pdata->getN();
    for (unsigned int cur_bond = 0; cur_bond < m_bonds.size(); cur_bond++)
        {
        unsigned int tag1 = m_bonds[cur_bond].a;
        unsigned int tag2 = m_bonds[cur_bond].b;
        unsigned int type = m_bonds[cur_bond].type;
        int idx1 = arrays.rtag[tag1];
        int idx2 = arrays.rtag[tag2];
        
        // get the number of bonds for each particle
        int num1 = m_host_n_bonds[idx1];
        int num2 = m_host_n_bonds[idx2];
        
        // add the new bonds to the table
        m_host_bonds[num1*pitch + idx1] = make_uint2(idx2, type);
        m_host_bonds[num2*pitch + idx2] = make_uint2(idx1, type);
        
        // increment the number of bonds
        m_host_n_bonds[idx1]++;
        m_host_n_bonds[idx2]++;
        }
        
    m_pdata->release();
    
    // copy the bond table to the device
    copyBondTable();
    }

/*! \param height New height for the bond table
    
    \post Reallocates memory on the device making room for up to
    \a height bonds per particle.
    
    \note updateBondTable() needs to be called after so that the
    data in the bond table will be correct.
*/
void BondData::reallocateBondTable(int height)
    {
    freeBondTable();
    allocateBondTable(height);
    }

/*! \param height Height for the bond table
*/
void BondData::allocateBondTable(int height)
    {
    // make sure the arrays have been deallocated
    assert(m_host_bonds == NULL);
    assert(m_host_n_bonds == NULL);
    
    unsigned int N = m_pdata->getN();
    
    // allocate device memory
    m_gpu_bonddata.allocate(m_pdata->getN(), height);
    CHECK_CUDA_ERROR();        
        
    // allocate and zero host memory
    cudaHostAlloc(&m_host_n_bonds, (size_t)N*sizeof(int), (unsigned int)cudaHostAllocPortable);
    memset((void*)m_host_n_bonds, 0, N*sizeof(int));
    
    cudaHostAlloc(&m_host_bonds, (size_t)N * height * sizeof(uint2), (unsigned int)cudaHostAllocPortable);
    memset((void*)m_host_bonds, 0, N*height*sizeof(uint2));
    CHECK_CUDA_ERROR();
    }

void BondData::freeBondTable()
    {
    // free device memory
    m_gpu_bonddata.deallocate();
    CHECK_CUDA_ERROR();
    
    // free host memory
    cudaFreeHost(m_host_bonds);
    m_host_bonds = NULL;
    cudaFreeHost(m_host_n_bonds);
    m_host_n_bonds = NULL;
    CHECK_CUDA_ERROR();
    }

//! Copies the bond table to the device
void BondData::copyBondTable()
    {
    // we need to copy the table row by row since cudaMemcpy2D has severe pitch limitations
    for (unsigned int row = 0; row < m_gpu_bonddata.height; row++)
        {
        // copy only the portion of the data to each GPU with particles local to that GPU
        cudaMemcpy(m_gpu_bonddata.bonds + m_gpu_bonddata.pitch*row,
                   m_host_bonds + row * m_pdata->getN(),
                   sizeof(uint2) * m_pdata->getN(),
                   cudaMemcpyHostToDevice);
        }
            
    cudaMemcpy(m_gpu_bonddata.n_bonds,
               m_host_n_bonds,
               sizeof(unsigned int) * m_pdata->getN(),
               cudaMemcpyHostToDevice);
    
    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    }
#endif

void export_BondData()
    {
    class_<Bond>("Bond", init<unsigned int, unsigned int, unsigned int>())
        .def_readonly("type", &Bond::type)
        .def_readonly("a", &Bond::a)
        .def_readonly("b", &Bond::b)
        ;
    
    class_<BondData, boost::shared_ptr<BondData>, boost::noncopyable>("BondData", init<shared_ptr<ParticleData>, unsigned int>())
    .def("getNumBonds", &BondData::getNumBonds)
    .def("getNBondTypes", &BondData::getNBondTypes)
    .def("getTypeByName", &BondData::getTypeByName)
    .def("getNameByType", &BondData::getNameByType)
    .def("addBond", &BondData::addBond)
    ;
    
    }

#ifdef WIN32
#pragma warning( pop )
#endif

