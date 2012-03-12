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
    : m_n_bond_types(n_bond_types), m_bonds_dirty(false), m_pdata(pdata), exec_conf(m_pdata->getExecConf()),
      m_bonds(exec_conf), m_bond_type(exec_conf), m_tags(exec_conf), m_bond_rtag(exec_conf)
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
        
    // allocate memory for the GPU bond table
    allocateBondTable(1);
    }

BondData::~BondData()
    {
    m_sort_connection.disconnect();
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
    \returns The unique tag identifying the added bond
 */
unsigned int BondData::addBond(const Bond& bond)
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

        // update reverse-lookup tag
        m_bond_rtag[tag] = m_bonds.size();
        }
    else
        {
        // Otherwise, generate a new tag
        tag = m_bonds.size();

        // add new reverse-lookup tag
        assert(m_bond_rtag.size() == m_bonds.size());
        m_bond_rtag.push_back(m_bonds.size());
        }

    assert(tag <= m_deleted_tags.size() + m_bonds.size());

    m_bonds.push_back(make_uint2(bond.a, bond.b));
    m_bond_type.push_back(bond.type);

    m_tags.push_back(tag);

    m_bonds_dirty = true;
    return tag;
    }

/*! \param tag tag of the bond to access
 */
const Bond BondData::getBondByTag(unsigned int tag) const
    {
    // Find position of bond in bonds list
    unsigned int bond_idx = m_bond_rtag[tag];
    if (bond_idx == NO_BOND)
        {
        cerr << endl << "***Error! Trying to get bond tag " << tag << " which does not exist!" << endl << endl;
        throw runtime_error("Error getting bond");
        }

    uint2 b = m_bonds[bond_idx];
    Bond bond(m_bond_type[bond_idx], b.x, b.y);
    return bond;
    }

/*! \param id Index of bond (0 to N-1)
    \returns Unique tag of bond (for use when calling removeBond())
*/
unsigned int BondData::getBondTag(unsigned int id) const
    {
    if (id >= getNumBonds())
        {
        cerr << endl << "***Error! Trying to get bond tag from id " << id << " which does not exist!" << endl << endl;
        throw runtime_error("Error getting bond tag");
        }
    return m_tags[id];
    }

/*! \param tag tag of bond to remove
 * \note Bond removal changes the order of m_bonds. If a hole in the bond list
 * is generated, the last bond in the list is moved up to fill that hole.
 */
void BondData::removeBond(unsigned int tag)
    {
    // Find position of bond in bonds list
    unsigned int id = m_bond_rtag[tag];
    if (id == NO_BOND)
        {
        cerr << endl << "***Error! Trying to remove bond tag " << tag << " which does not exist!" << endl << endl;
        throw runtime_error("Error removing bond");
        }

    // delete from map
    m_bond_rtag[tag] = NO_BOND;

    unsigned int size = m_bonds.size();
    // If the bond is in the middle of the list, move the last element to
    // to the position of the removed element
    if (id < (size-1))
        {
        m_bonds[id] =(uint2) m_bonds[size-1];
        m_bond_type[id] = (unsigned int) m_bond_type[size-1];
        unsigned int last_tag = m_tags[size-1];
        m_bond_rtag[last_tag] = id;
        m_tags[id] = last_tag;
        }
    // delete last element
    m_bonds.pop_back();
    m_bond_type.pop_back();
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


/*! Updates the bond data on the GPU if needed and returns the data structure needed to access it.
 *
 * \warning The order of bonds in the GPU bond table differs whether it is constructed using
         the GPU implementation of updateBondTableGPU() and the CPU implementation updateBondTable().
         It is therefore unspecified (but in, in any case, deterministic).
*/
const GPUArray<uint2>& BondData::getGPUBondList()
    {
    if (m_bonds_dirty)
        {
#ifdef ENABLE_CUDA
        // update bond table
        if (exec_conf->isCUDAEnabled())
            updateBondTableGPU();
        else
            updateBondTable();
#else
        updateBondTable();
#endif
        m_bonds_dirty = false;
        }
    return m_gpu_bondlist;
    }


#ifdef ENABLE_CUDA
/*! Update GPU bond table (GPU version)

    \post The bond tag data added via addBond() is translated to bonds based
    on particle index for use in the GPU kernel. This new bond table is then uploaded
    to the device.
*/
void BondData::updateBondTableGPU()
    {
    unsigned int max_bond_num;

        {
        ArrayHandle<uint2> d_bonds(m_bonds, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_n_bonds(m_n_bonds, access_location::device, access_mode::overwrite);
        gpu_find_max_bond_number(max_bond_num,
                                 d_n_bonds.data,
                                 d_bonds.data,
                                 m_bonds.size(),
                                 m_pdata->getN(),
                                 d_rtag.data);
        }

    // re allocate memory if needed
    if (max_bond_num > m_gpu_bondlist.getHeight())
        {
        m_gpu_bondlist.resize(m_pdata->getN(), max_bond_num);
        }

        {
        ArrayHandle<uint2> d_bonds(m_bonds, access_location::device, access_mode::read);
        ArrayHandle<uint2> d_gpu_bondlist(m_gpu_bondlist, access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> d_n_bonds(m_n_bonds, access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> d_bond_type(m_bond_type, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::read);
        gpu_create_bondtable(d_gpu_bondlist.data,
                             d_n_bonds.data,
                             d_bonds.data,
                             d_bond_type.data,
                             d_rtag.data,
                             m_bonds.size(),
                             m_gpu_bondlist.getPitch(),
                             m_pdata->getN());
        }
    }
#endif

/*! Update the GPU bond table (CPU version)

    \post The bond tag data added via addBond() is translated to bonds based
    on particle index for use in the GPU kernel. This new bond table is then uploaded
    to the device.
*/
void BondData::updateBondTable()
    {

    ArrayHandle< unsigned int > h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    unsigned int num_bonds_max = 0;
        {
        ArrayHandle<unsigned int> h_n_bonds(m_n_bonds, access_location::host, access_mode::overwrite);

        // count the number of bonds per particle
        // start by initializing the n_bonds values to 0
        memset(h_n_bonds.data, 0, sizeof(unsigned int) * m_pdata->getN());

        // loop through the particles and count the number of bonds based on each particle index
        for (unsigned int cur_bond = 0; cur_bond < m_bonds.size(); cur_bond++)
            {
            unsigned int tag1 = ((uint2) m_bonds[cur_bond]).x;
            unsigned int tag2 = ((uint2) m_bonds[cur_bond]).y;
            unsigned int idx1 = h_rtag.data[tag1];
            unsigned int idx2 = h_rtag.data[tag2];

            // only local particles are considered
            if (idx1 < m_pdata->getN())
                h_n_bonds.data[idx1]++;
            if (idx2 < m_pdata->getN())
                h_n_bonds.data[idx2]++;
            }

        // find the maximum number of bonds
        unsigned int nparticles = m_pdata->getN();
        for (unsigned int i = 0; i < nparticles; i++)
            {
            if (h_n_bonds.data[i] > num_bonds_max)
                num_bonds_max = h_n_bonds.data[i];
            }
        }

    // re allocate memory if needed
    if (num_bonds_max > m_gpu_bondlist.getHeight())
        {
        m_gpu_bondlist.resize(m_pdata->getN(), num_bonds_max);
        }

        {
        ArrayHandle<unsigned int> h_n_bonds(m_n_bonds, access_location::host, access_mode::overwrite);
        ArrayHandle<uint2> h_gpu_bondlist(m_gpu_bondlist, access_location::host, access_mode::overwrite);

        // now, update the actual table
        // zero the number of bonds counter (again)
        memset(h_n_bonds.data, 0, sizeof(unsigned int) * m_pdata->getN());

        // loop through all bonds and add them to each column in the list
        int pitch = m_gpu_bondlist.getPitch();
        for (unsigned int cur_bond = 0; cur_bond < m_bonds.size(); cur_bond++)
            {
            unsigned int tag1 = ((uint2)m_bonds[cur_bond]).x;
            unsigned int tag2 = ((uint2)m_bonds[cur_bond]).y;
            unsigned int type = m_bond_type[cur_bond];
            unsigned int idx1 = h_rtag.data[tag1];
            unsigned int idx2 = h_rtag.data[tag2];

            // get the number of bonds for each particle
            // add the new bonds to the table
            // increment the number of bonds
            if (idx1 < m_pdata->getN())
                {
                unsigned int num1 = h_n_bonds.data[idx1];
                h_gpu_bondlist.data[num1*pitch + idx1] = make_uint2(idx2, type);
                h_n_bonds.data[idx1]++;
                }
            if (idx2 < m_pdata->getN())
                {
                unsigned int num2 = h_n_bonds.data[idx2];
                h_gpu_bondlist.data[num2*pitch + idx2] = make_uint2(idx1, type);
                h_n_bonds.data[idx2]++;
                }
            }
        }
    }


/*! \param height Height for the bond table
*/
void BondData::allocateBondTable(int height)
    {
    // make sure the arrays have been deallocated
    assert(m_n_bonds.isNull());
    
    GPUArray<uint2> gpu_bondlist(m_pdata->getN(), height, exec_conf);
    m_gpu_bondlist.swap(gpu_bondlist);
        
    GPUArray<unsigned int> n_bonds(m_pdata->getN(), exec_conf);
    m_n_bonds.swap(n_bonds);

    }

//! Takes a snapshot of the current bond data
/*! \param snapshot The snapshot that will contain the bond data
*/
void BondData::takeSnapshot(SnapshotBondData& snapshot)
    {
    // check for an invalid request
    if (snapshot.bonds.size() != getNumBonds())
        {
        cerr << endl << "***Error! BondData is being asked to initizalize a snapshot of the wrong size."
             << endl << endl;
        throw runtime_error("Error taking snapshot.");
        }

    assert(snapshot.type_id.size() == getNumBonds());
    assert(snapshot.type_mapping.size() == 0);

    for (unsigned int bond_idx = 0; bond_idx < getNumBonds(); bond_idx++)
        {
        snapshot.bonds[bond_idx] = m_bonds[bond_idx];
        snapshot.type_id[bond_idx] = m_bond_type[bond_idx];
        }

    for (unsigned int i = 0; i < m_n_bond_types; i++)
        snapshot.type_mapping.push_back(m_bond_type_mapping[i]);

    }

//! Initialize the bond data from a snapshot
/*! \param snapshot The snapshot to initialize the bonds from
    Before initialization, the current bond data is cleared.
 */
void BondData::initializeFromSnapshot(const SnapshotBondData& snapshot)
    {
    m_bonds.clear();
    m_bond_type.clear();
    m_tags.clear();
    while (! m_deleted_tags.empty())
        m_deleted_tags.pop();
    m_bond_rtag.clear();

    for (unsigned int bond_idx = 0; bond_idx < snapshot.bonds.size(); bond_idx++)
        {
        Bond bond(snapshot.type_id[bond_idx], snapshot.bonds[bond_idx].x, snapshot.bonds[bond_idx].y);
        addBond(bond);
        }

    setBondTypeMapping(snapshot.type_mapping);
    }

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
    .def("removeBond", &BondData::removeBond)
    .def("getBond", &BondData::getBond)
    .def("getBondByTag", &BondData::getBondByTag)
    .def("getBondTag", &BondData::getBondTag)
    .def("takeSnapshot", &BondData::takeSnapshot)
    .def("initializeFromSnapshot", &BondData::initializeFromSnapshot)
    ;
    
    }

#ifdef WIN32
#pragma warning( pop )
#endif

