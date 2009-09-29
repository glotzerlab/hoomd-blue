/*
Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
Source Software License
Copyright (c) 2008 Ames Laboratory Iowa State University
All rights reserved.

Redistribution and use of HOOMD, in source and binary forms, with or
without modification, are permitted, provided that the following
conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names HOOMD's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
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

#ifdef ENABLE_CUDA
#include "gpu_settings.h"
#endif

/*! \file BondData.cc
    \brief Defines BondData.
 */

/*! \param pdata ParticleData these bonds refer into
    \param n_bond_types Number of bond types in the list
*/
BondData::BondData(boost::shared_ptr<ParticleData> pdata, unsigned int n_bond_types) 
	: m_n_bond_types(n_bond_types), m_bonds_dirty(false), m_pdata(pdata)
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
    const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
    
    // init pointers
    m_host_bonds = NULL;
    m_host_n_bonds = NULL;
    m_gpu_bonddata.resize(exec_conf.gpu.size());
    for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
        {
        m_gpu_bonddata[cur_gpu].bonds = NULL;
        m_gpu_bonddata[cur_gpu].n_bonds = NULL;
        m_gpu_bonddata[cur_gpu].height = 0;
        m_gpu_bonddata[cur_gpu].pitch = 0;
        }
        
    // allocate memory on the GPU if there is a GPU in the execution configuration
    if (exec_conf.gpu.size() >= 1)
        {
        allocateBondTable(1);
        }
#endif
    }

BondData::~BondData()
    {
    m_sort_connection.disconnect();
    
#ifdef ENABLE_CUDA
    const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
    if (!exec_conf.gpu.empty())
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
        
    m_bonds.push_back(bond);
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
std::vector<gpu_bondtable_array>& BondData::acquireGPU()
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
    if (num_bonds_max > m_gpu_bonddata[0].height)
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
    
    // get the execution configuration
    const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
    
    // allocate device memory
    exec_conf.tagAll(__FILE__, __LINE__);
    for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
        exec_conf.gpu[cur_gpu]->call(bind(&gpu_bondtable_array::allocate, &m_gpu_bonddata[cur_gpu], m_pdata->getLocalNum(cur_gpu), height));
        
        
    // allocate and zero host memory
    exec_conf.gpu[0]->call(bind(cudaHostAllocHack, (void**)((void*)&m_host_n_bonds), (size_t)N*sizeof(int), (unsigned int)cudaHostAllocPortable));
    memset((void*)m_host_n_bonds, 0, N*sizeof(int));
    
    exec_conf.gpu[0]->call(bind(cudaHostAllocHack, (void**)((void*)&m_host_bonds), (size_t)N * height * sizeof(uint2), (unsigned int)cudaHostAllocPortable));
    memset((void*)m_host_bonds, 0, N*height*sizeof(uint2));
    }

void BondData::freeBondTable()
    {
    // get the execution configuration
    const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
    
    // free device memory
    exec_conf.tagAll(__FILE__, __LINE__);
    for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
        exec_conf.gpu[cur_gpu]->call(bind(&gpu_bondtable_array::deallocate, &m_gpu_bonddata[cur_gpu]));
        
    // free host memory
    exec_conf.gpu[0]->call(bind(cudaFreeHost, m_host_bonds));
    m_host_bonds = NULL;
    exec_conf.gpu[0]->call(bind(cudaFreeHost, m_host_n_bonds));
    m_host_n_bonds = NULL;
    }

//! Copies the bond table to the device
void BondData::copyBondTable()
    {
    // get the execution configuration
    const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
    
    exec_conf.tagAll(__FILE__, __LINE__);
    for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
        {
        // we need to copy the table row by row since cudaMemcpy2D has severe pitch limitations
        for (unsigned int row = 0; row < m_gpu_bonddata[0].height; row++)
            {
            // copy only the portion of the data to each GPU with particles local to that GPU
            exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, m_gpu_bonddata[cur_gpu].bonds + m_gpu_bonddata[cur_gpu].pitch*row,
                                              m_host_bonds + row * m_pdata->getN() + m_pdata->getLocalBeg(cur_gpu),
                                              sizeof(uint2) * m_pdata->getLocalNum(cur_gpu),
                                              cudaMemcpyHostToDevice));
            }
            
        exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, m_gpu_bonddata[cur_gpu].n_bonds,
                                          m_host_n_bonds + m_pdata->getLocalBeg(cur_gpu),
                                          sizeof(unsigned int) * m_pdata->getLocalNum(cur_gpu),
                                          cudaMemcpyHostToDevice));
        }
    }
#endif

void export_BondData()
    {
    class_<BondData, boost::shared_ptr<BondData>, boost::noncopyable>("BondData", init<shared_ptr<ParticleData>, unsigned int>())
    .def("getNumBonds", &BondData::getNumBonds)
    .def("getNBondTypes", &BondData::getNBondTypes)
    .def("getTypeByName", &BondData::getTypeByName)
    .def("getNameByType", &BondData::getNameByType)
    ;
    
    }

#ifdef WIN32
#pragma warning( pop )
#endif

