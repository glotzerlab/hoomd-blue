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
#include "Profiler.h"

#include <boost/python.hpp>
using namespace boost::python;

#include <boost/bind.hpp>
using namespace boost;

#include <iostream>
#include <stdexcept>

#include <algorithm>
#include <boost/iterator/zip_iterator.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

using namespace std;

/*! \file BondData.cc
    \brief Defines BondData.
 */

/*! \param pdata ParticleData these bonds refer into
    \param n_bond_types Number of bond types in the list
*/
BondData::BondData(boost::shared_ptr<ParticleData> pdata, unsigned int n_bond_types)
    : m_bonds_dirty(false),
      m_pdata(pdata), exec_conf(m_pdata->getExecConf()),
      m_bonds(exec_conf), m_bond_type(exec_conf), m_tags(exec_conf), m_bond_rtag(exec_conf)
#ifdef ENABLE_CUDA
      , m_max_bond_num(0),
      m_buffers_initialized(false)
#endif
    {
    assert(pdata);
    m_exec_conf = m_pdata->getExecConf();
    m_exec_conf->msg->notice(5) << "Constructing BondData" << endl;

    // attach to the signal for notifications of particle sorts
    m_sort_connection = m_pdata->connectParticleSort(bind(&BondData::setDirty, this));

    // attach to max particle num change connection
    m_max_particle_num_change_connection = m_pdata->connectMaxParticleNumberChange(bind(&BondData::reallocate, this));

    // attach to ghost particle number change connection
    m_ghost_particle_num_change_connection = m_pdata->connectGhostParticleNumberChange(bind(&BondData::setDirty, this));

    #ifdef ENABLE_MPI
    // attach to signal when a single particle is moved between domains
    m_ptl_move_connection = m_pdata->connectSingleParticleMove(bind(&BondData::moveParticleBonds, this, _1, _2, _3));
    #endif

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

    #ifdef ENABLE_CUDA
    GPUFlags<unsigned int> condition(exec_conf);
    m_condition.swap(condition);
    #endif

    m_num_bonds_global = 0;
    }

/*! \param pdata ParticleData these bonds refer into
 * \param snapshot SnapshotBondData that contains the bond information
*/
BondData::BondData(boost::shared_ptr<ParticleData> pdata, const SnapshotBondData& snapshot)
    : m_bonds_dirty(false),
      m_pdata(pdata), exec_conf(m_pdata->getExecConf()),
      m_bonds(exec_conf), m_bond_type(exec_conf), m_tags(exec_conf), m_bond_rtag(exec_conf),
#ifdef ENABLE_CUDA
      m_max_bond_num(0),
      m_buffers_initialized(false),
#endif
      m_num_bonds_global(0)
    {
    assert(pdata);
    m_exec_conf = m_pdata->getExecConf();
    m_exec_conf->msg->notice(5) << "Constructing BondData" << endl;

    // attach to the signal for notifications of particle sorts
    m_sort_connection = m_pdata->connectParticleSort(bind(&BondData::setDirty, this));

    // attach to max particle num change connection
    m_max_particle_num_change_connection = m_pdata->connectMaxParticleNumberChange(bind(&BondData::reallocate, this));

    // attach to ghost particle number change connection
    m_ghost_particle_num_change_connection = m_pdata->connectGhostParticleNumberChange(bind(&BondData::setDirty, this));

    #ifdef ENABLE_MPI
    // attach to signal when a single particle is moved between domains
    m_ptl_move_connection = m_pdata->connectSingleParticleMove(bind(&BondData::moveParticleBonds, this, _1, _2, _3));
    #endif

    // allocate memory for the GPU bond table
    allocateBondTable(1);

    #ifdef ENABLE_CUDA
    GPUFlags<unsigned int> condition(exec_conf);
    m_condition.swap(condition);
    #endif

    // initialize bond data from snapshot
    initializeFromSnapshot(snapshot);
    }


BondData::~BondData()
    {
    m_exec_conf->msg->notice(5) << "Destroying BondData" << endl;
    m_sort_connection.disconnect();
    m_max_particle_num_change_connection.disconnect();
    m_ghost_particle_num_change_connection.disconnect();

    #ifdef ENABLE_MPI
    m_ptl_move_connection.disconnect();
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
    \returns The unique tag identifying the added bond
 */
unsigned int BondData::addBond(const Bond& bond)
    {
#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        m_exec_conf->msg->error() << "Dynamically adding bonds in simulations with domain decomposition is not currently supported." << std::endl;
        throw runtime_error("Error adding bond");
        }
#endif
    // check for some silly errors a user could make
    if (bond.a >= m_pdata->getNGlobal() || bond.b >= m_pdata->getNGlobal())
        {
        m_exec_conf->msg->error() << "Particle tag out of bounds when attempting to add bond: " << bond.a << "," << bond.b << endl;
        throw runtime_error("Error adding bond");
        }

    if (bond.a == bond.b)
        {
        m_exec_conf->msg->error() << "Particle cannot be bonded to itself! " << bond.a << "," << bond.b << endl;
        throw runtime_error("Error adding bond");
        }

    // check that the type is within bouds
    if (bond.type+1 > m_bond_type_mapping.size())
        {
        m_exec_conf->msg->error() << "Invalid bond type! " << bond.type << ", the number of types is " << m_bond_type_mapping.size() << endl;
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
    m_num_bonds_global++;

    m_bonds_dirty = true;

    return tag;
    }

/*! \param tag tag of the bond to access
 */
const Bond BondData::getBondByTag(unsigned int tag) const
    {
    // Find position of bond in bonds list
    unsigned int bond_idx = m_bond_rtag[tag];

    uint2 b;
    unsigned int bond_type;

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        // set local to rank if the bond is local, -1 if not
        int rank = bond_idx < m_bonds.size() ? (int) m_exec_conf->getRank() : -1;

        // the largest rank owning the bond sends it to the others
        MPI_Allreduce(MPI_IN_PLACE,
                      &rank,
                      1,
                      MPI_INT,
                      MPI_MAX,
                      m_exec_conf->getMPICommunicator());

        if (rank == -1)
            {
            m_exec_conf->msg->error() << "Trying to get bond tag " << tag << " which does not exist!" << endl;
            throw runtime_error("Error getting bond");
            }

        if (rank == (int)m_exec_conf->getRank())
            {
            b = m_bonds[bond_idx];
            bond_type = m_bond_type[bond_idx];
            }

        bcast(b, rank, m_exec_conf->getMPICommunicator());
        bcast(bond_type, rank, m_exec_conf->getMPICommunicator());
        }
    else
#endif
        {
        if (bond_idx == BOND_NOT_LOCAL)
            {
            m_exec_conf->msg->error() << "Trying to get bond tag " << tag << " which does not exist!" << endl;
            throw runtime_error("Error getting bond");
            }

        b = m_bonds[bond_idx];
        bond_type = m_bond_type[bond_idx];
        }

    Bond bond(bond_type, b.x, b.y);
    return bond;
    }

/*! \param id Index of bond (0 to N-1)
    \returns Unique tag of bond (for use when calling removeBond())
*/
unsigned int BondData::getBondTag(unsigned int id) const
    {
    if (id >= getNumBonds())
        {
        m_exec_conf->msg->error() << "Trying to get bond tag from id " << id << " which does not exist!" << endl;
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
    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        m_exec_conf->msg->error() << "This feature is unsupported in multi-GPU mode." << std::endl;
        throw runtime_error("Error removing bond");
        }
    #endif

    // Find position of bond in bonds list
    unsigned int id = m_bond_rtag[tag];
    if (id == BOND_NOT_LOCAL)
        {
        m_exec_conf->msg->error() << "Trying to remove bond tag " << tag << " which does not exist!" << endl;
        throw runtime_error("Error removing bond");
        }

    // delete from map
    m_bond_rtag[tag] = BOND_NOT_LOCAL;

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
    m_num_bonds_global--;

    m_bonds_dirty = true;
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

    m_exec_conf->msg->error() << "Bond type " << name << " not found!" << endl;
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
    if (type >= m_bond_type_mapping.size())
        {
        m_exec_conf->msg->error() << "Requesting type name for non-existant type " << type << endl;
        throw runtime_error("Error mapping type name");
        }

    // return the name
    return m_bond_type_mapping[type];
    }


/*! Updates the bond data on the GPU if needed and returns the data structure needed to access it.
 */
void BondData::checkUpdateBondList()
    {
    if (m_bonds_dirty)
        {
        if (m_prof)
            m_prof->push("update btable");

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

        if (m_prof)
            m_prof->pop();
        }
    }


#ifdef ENABLE_CUDA
/*! Update GPU bond table (GPU version)

    \post The bond tag data added via addBond() is translated to bonds based
    on particle index for use in the GPU kernel. This new bond table is then uploaded
    to the device.
*/
void BondData::updateBondTableGPU()
    {
    unsigned int condition = 0;

    do
        {
        m_condition.resetFlags(0);

            {
            ArrayHandle<uint2> d_bonds(m_bonds, access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::read);
            ArrayHandle<unsigned int> d_n_bonds(m_n_bonds, access_location::device, access_mode::overwrite);

            gpu_find_max_bond_number(d_n_bonds.data,
                                     d_bonds.data,
                                     m_bonds.size(),
                                     m_pdata->getN(),
                                     m_pdata->getNGhosts(),
                                     d_rtag.data,
                                     m_max_bond_num,
                                     m_condition.getDeviceFlags());

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            }

        condition = m_condition.readFlags();

        if (condition & 1)
            {
            // reallocate bond list
            m_gpu_bondlist.resize(m_pdata->getMaxN(), ++m_max_bond_num);
            }
        if (condition & 2)
            {
            // incomplete bond
            m_exec_conf->msg->error() << "bond.*: Invalid bond." << std::endl << std::endl;
            throw std::runtime_error("Error updating bond list.");
            }
        }
    while (condition);

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

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
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

        unsigned int N = m_pdata->getN();
        // loop through the particles and count the number of bonds based on each particle index
        for (unsigned int cur_bond = 0; cur_bond < m_bonds.size(); cur_bond++)
            {
            unsigned int tag1 = ((uint2) m_bonds[cur_bond]).x;
            unsigned int tag2 = ((uint2) m_bonds[cur_bond]).y;
            unsigned int idx1 = h_rtag.data[tag1];
            unsigned int idx2 = h_rtag.data[tag2];

            // first count only local bond members
            if (idx1 < N)
                h_n_bonds.data[idx1]++;
            if (idx2 < N)
                h_n_bonds.data[idx2]++;
            }

        // find the maximum number of bonds
        for (unsigned int i = 0; i < N; i++)
            if (h_n_bonds.data[i] > num_bonds_max)
                num_bonds_max = h_n_bonds.data[i];
        }

    // re allocate memory if needed
    if (num_bonds_max > m_gpu_bondlist.getHeight())
        {
        m_gpu_bondlist.resize(m_pdata->getMaxN(), num_bonds_max);
        }


        {
        ArrayHandle<unsigned int> h_n_bonds(m_n_bonds, access_location::host, access_mode::overwrite);
        ArrayHandle<uint2> h_gpu_bondlist(m_gpu_bondlist, access_location::host, access_mode::overwrite);

        // now, update the actual table
        // zero the number of bonds counter (again)
        memset(h_n_bonds.data, 0, sizeof(unsigned int) * m_pdata->getN());

        // loop through all bonds and add them to each column in the list
        unsigned int pitch = m_gpu_bondlist.getPitch();

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
            unsigned int N = m_pdata->getN();
            if (idx1 < N)
                {
                unsigned int num1 = h_n_bonds.data[idx1];
                h_gpu_bondlist.data[num1*pitch + idx1] = make_uint2(idx2, type);
                h_n_bonds.data[idx1]++;
                }
            if (idx2 < N)
                {
                unsigned int num2 = h_n_bonds.data[idx2];
                h_gpu_bondlist.data[num2*pitch + idx2] = make_uint2(idx1, type);
                h_n_bonds.data[idx2]++;
                }
            }
        }
    }

//! Helper function to reallocate the GPU bond table
void BondData::reallocate()
    {
    m_gpu_bondlist.resize(m_pdata->getMaxN(), m_gpu_bondlist.getHeight());
    m_n_bonds.resize(m_pdata->getMaxN());
   }

/*! \param height Height for the bond table
*/
void BondData::allocateBondTable(int height)
    {
    // make sure the arrays have been deallocated
    assert(m_n_bonds.isNull());

    GPUArray<uint2> gpu_bondlist(m_pdata->getMaxN(), height, exec_conf);
    m_gpu_bondlist.swap(gpu_bondlist);

    GPUArray<unsigned int> n_bonds(m_pdata->getMaxN(), exec_conf);
    m_n_bonds.swap(n_bonds);
    }

//! Takes a snapshot of the current bond data
/*! \param snapshot The snapshot that will contain the bond data
*/
void BondData::takeSnapshot(SnapshotBondData& snapshot)
    {
    // allocate memory in snapshot
    snapshot.resize(getNumBondsGlobal());

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        std::vector<uint2> local_bonds;
        std::vector<unsigned int> local_typeid;
        std::map<unsigned int, unsigned int> bond_rtag_map;

        // construct a local list of bonds
        for (unsigned int bond_idx = 0; bond_idx < m_bonds.size(); ++bond_idx)
            {
            local_bonds.push_back(m_bonds[bond_idx]);
            local_typeid.push_back(m_bond_type[bond_idx]);

            bond_rtag_map.insert(std::pair<unsigned int, unsigned int>(m_tags[bond_idx], bond_idx));
            }

        // gather lists from all processors
        unsigned int root = 0;

        std::vector< std::vector<uint2> > bond_proc;
        std::vector< std::vector<unsigned int> > typeid_proc;
        std::vector< std::map<unsigned int, unsigned int> > rtag_map_proc;

        gather_v(local_bonds, bond_proc, root, m_exec_conf->getMPICommunicator());
        gather_v(local_typeid, typeid_proc, root, m_exec_conf->getMPICommunicator());
        gather_v(bond_rtag_map, rtag_map_proc, root, m_exec_conf->getMPICommunicator());

        if (m_exec_conf->getRank() == root)
            {
            std::map<unsigned int, unsigned int>::iterator it;

            for (unsigned int bond_tag = 0; bond_tag < getNumBondsGlobal(); ++bond_tag)
                {
                bool found = false;
                unsigned int rank;
                for (rank = 0; rank < m_exec_conf->getNRanks(); ++rank)
                    {
                    it = rtag_map_proc[rank].find(bond_tag);
                    if (it != rtag_map_proc[rank].end())
                        {
                        found = true;
                        break;
                        }
                    }
                if (! found)
                    {
                    cerr << endl << "***Error! Could not find bond " << bond_tag << " on any processor. " << endl << endl;
                    throw std::runtime_error("Error gathering BondData");
                    }

                // store bond in snapshot
                unsigned int bond_idx = it->second;

                snapshot.bonds[bond_tag] = bond_proc[rank][bond_idx];
                snapshot.type_id[bond_tag] = typeid_proc[rank][bond_idx];
                }
            }
        }
    else
#endif
        {
        for (unsigned int bond_idx = 0; bond_idx < getNumBonds(); bond_idx++)
            {
            snapshot.bonds[bond_idx] = m_bonds[bond_idx];
            snapshot.type_id[bond_idx] = m_bond_type[bond_idx];
            }
        }

    snapshot.type_mapping = m_bond_type_mapping;

    }

//! Initialize the bond data from a snapshot
/*! \param snapshot The snapshot to initialize the bonds from
    Before initialization, the current bond data is cleared.

    \pre Particle data must have been initialized on all processors using ParticleData::initializeFromSnapshot()
 */
void BondData::initializeFromSnapshot(const SnapshotBondData& snapshot)
    {
    // check that all fields in the snapshot have correct length
    if (m_exec_conf->getRank() == 0 && ! snapshot.validate())
        {
        m_exec_conf->msg->error() << "init.*: invalid bond data snapshot."
                                << std::endl << std::endl;
        throw std::runtime_error("Error initializing bond data.");
        }

    m_bonds.clear();
    m_bond_type.clear();
    m_tags.clear();
    while (! m_deleted_tags.empty())
        m_deleted_tags.pop();
    m_bond_rtag.clear();

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        // first, broadcast bonds to all processors
        std::vector<uint2> all_bonds;
        std::vector<unsigned int> all_typeid;
        std::vector<std::string> type_mapping;

        unsigned int root = 0;
        if (m_exec_conf->getRank() == root)
            {
            all_bonds = snapshot.bonds;
            all_typeid = snapshot.type_id;
            type_mapping = snapshot.type_mapping;
            }

        bcast(all_bonds, root, m_exec_conf->getMPICommunicator());
        bcast(all_typeid, root, m_exec_conf->getMPICommunicator());
        bcast(type_mapping, root, m_exec_conf->getMPICommunicator());

        // set global number of bonds
        m_num_bonds_global = all_bonds.size();
        m_bond_rtag.resize(m_num_bonds_global);

            {
            // reset reverse lookup tags
            ArrayHandle<unsigned int> h_bond_rtag(m_bond_rtag, access_location::host, access_mode::overwrite);

            for (unsigned int tag = 0; tag < m_num_bonds_global; ++tag)
                h_bond_rtag.data[tag] = BOND_NOT_LOCAL;
            }

        // now iterate over bonds and retain only bonds with local members
        ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

        std::vector<uint2>::iterator it;
        unsigned int nparticles = m_pdata->getN();
        unsigned int bond_idx = 0;
        for (unsigned int bond_tag = 0; bond_tag < m_num_bonds_global; ++bond_tag)
            if ((h_rtag.data[all_bonds[bond_tag].x] < nparticles)
                || (h_rtag.data[all_bonds[bond_tag].y] < nparticles))
                {
                m_bonds.push_back(all_bonds[bond_tag]);
                m_bond_type.push_back(all_typeid[bond_tag]);
                m_tags.push_back(bond_tag);
                m_bond_rtag[bond_tag] = bond_idx++;
                }

        m_bond_type_mapping = type_mapping;
        }
    else
#endif
        {
        m_bond_type_mapping = snapshot.type_mapping;

        for (unsigned int bond_idx = 0; bond_idx < snapshot.bonds.size(); bond_idx++)
            {
            Bond bond(snapshot.type_id[bond_idx], snapshot.bonds[bond_idx].x, snapshot.bonds[bond_idx].y);
            addBond(bond);
            }
        }

    m_bonds_dirty = true;
    }

#ifdef ENABLE_MPI
//! Pack bond data into a buffer
void BondData::retrieveBonds(std::vector<bond_element>& out)
    {
    // access bond data arrays
    ArrayHandle<uint2> h_bonds(m_bonds, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_bond_type(getBondTypes(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_bond_tag(getBondTags(), access_location::host, access_mode::read);

    // access reverse-lookup table
    ArrayHandle<unsigned int> h_bond_rtag(getBondRTags(), access_location::host, access_mode::readwrite);

    unsigned int n_bonds = getNumBonds();

    // reset out vector
    out.clear();

    // pack bonds
    for (unsigned int bond_idx = 0; bond_idx < n_bonds; bond_idx++)
        {
        unsigned int bond_tag = h_bond_tag.data[bond_idx];

        assert(bond_tag < getNumBondsGlobal());

        unsigned int bond_rtag = h_bond_rtag.data[bond_tag];
        if (bond_rtag == BOND_STAGED || bond_rtag == BOND_SPLIT)
            {
            bond_element b;
            b.bond = h_bonds.data[bond_idx];
            b.type = h_bond_type.data[bond_idx];
            b.tag = bond_tag;
            out.push_back(b);
            }

        // mark staged bonds for removal
        if (bond_rtag == BOND_STAGED)
            h_bond_rtag.data[bond_tag] = BOND_NOT_LOCAL;
        }
    }

//! A tuple of pdata pointers
typedef boost::tuple <
    unsigned int *,  // tag
    uint2 *,         // bond
    unsigned int *   // type
    > bdata_it_tuple;

//! A zip iterator for filtering particle data
typedef boost::zip_iterator<bdata_it_tuple> bdata_zip;

//! A tuple of pdata fields
typedef boost::tuple <
    const unsigned int,  // tag
    const uint2,         // bond
    const unsigned int   // type
    > bdata_tuple;

//! A predicate to select particles by rtag
struct bdata_select
    {
    //! Constructor
    bdata_select(const unsigned int *_rtag, const unsigned int _compare)
        : rtag(_rtag), compare(_compare)
        { }

    //! Returns true if the remove flag is set for a particle
    bool operator() (bdata_tuple const x) const
        {
        return rtag[x.get<0>()] == compare;
        }

    const unsigned int *rtag;    //!< The reverse-lookup tag array
    const unsigned int compare;  //!< The value to compare the rtag to
    };

//! A converter from bond_element to a tuple of bond data entries
struct to_bdata_tuple : public std::unary_function<const bond_element, const bdata_tuple>
    {
    const bdata_tuple operator() (const bond_element b) const
        {
        return boost::make_tuple(
            b.tag,
            b.bond,
            b.type
            );
        }
    };

//! A predicate to check if the bond doesn't already exist
struct bond_duplicate
    {
    const unsigned int *h_bond_rtag;      //!< Bond reverse-lookup table
    unsigned int num_bonds_global; //!< Global number of bonds
    unsigned int num_bonds;        //!< Number of local bonds

    //! Constructor
    /*! \param _h_bond_rtag Pointer to reverse-lookup table
     *  \param _num_bonds_global Global number of bonds
     *  \param _num_bonds Number of local bonds
     */
    bond_duplicate(const unsigned int *_h_bond_rtag, unsigned int _num_bonds_global, unsigned int _num_bonds)
        : h_bond_rtag(_h_bond_rtag), num_bonds_global(_num_bonds_global), num_bonds(_num_bonds)
        { }

    //! Return true f bond is duplicate
    const bool operator() (const bdata_tuple t) const
        {
        unsigned int bond_tag = t.get<0>();
        assert(bond_tag < num_bonds_global);

        unsigned int bond_rtag = h_bond_rtag[bond_tag];
        if (bond_rtag != BOND_NOT_LOCAL)
            {
            assert(bond_rtag < num_bonds || bond_rtag == BOND_SPLIT);
            return true;
            }

        return false;
        }
    };

//! Unpack a buffer with new bonds to be added, and remove obsolete bonds
void BondData::addRemoveBonds(const std::vector<bond_element>& in)
    {
    if (m_prof) m_prof->push("unpack/remove bonds");

    unsigned int num_add_bonds = in.size();

    // old number of bonds
    unsigned int old_n_bonds = getNumBonds();

    // new number of bonds, before removal
    unsigned int new_n_bonds = old_n_bonds + num_add_bonds;

    // resize internal data structures to fit the new bonds
    m_bonds.resize(new_n_bonds);
    m_bond_type.resize(new_n_bonds);
    m_tags.resize(new_n_bonds);

        {
        // access bond data arrays
        ArrayHandle<uint2> h_bonds(getBondTable(), access_location::host, access_mode::readwrite);
        ArrayHandle<unsigned int> h_bond_tag(getBondTags(), access_location::host, access_mode::readwrite);
        ArrayHandle<unsigned int> h_bond_type(getBondTypes(), access_location::host, access_mode::readwrite);

        // access reverse-lookup table
        ArrayHandle<unsigned int> h_bond_rtag(getBondRTags(), access_location::host, access_mode::readwrite);

        bdata_zip bdata_begin = boost::make_tuple(
            h_bond_tag.data,
            h_bonds.data,
            h_bond_type.data
            );
        bdata_zip bdata_end = bdata_begin + old_n_bonds;

        // erase all elements for which rtag == BOND_NOT_LOCAL
        // the array remains contiguous
        bdata_zip new_bdata_end;
        new_bdata_end = std::remove_if(bdata_begin, bdata_end, bdata_select(h_bond_rtag.data, BOND_NOT_LOCAL));

        // set up a transform iterator from bond_element to bond data tuple
        boost::transform_iterator<to_bdata_tuple, std::vector<bond_element>::const_iterator> in_transform(
            in.begin(),
            to_bdata_tuple());

        // add new bonds at the end, omitting duplicates
        new_bdata_end = std::remove_copy_if(in_transform, in_transform + num_add_bonds, new_bdata_end,
             bond_duplicate(h_bond_rtag.data, getNumBondsGlobal(), old_n_bonds));

        // compute new size of bond data arrays
        new_n_bonds = new_bdata_end - bdata_begin;

        // recompute rtags
        for (unsigned int bond_idx = 0; bond_idx < new_n_bonds; ++bond_idx)
            {
            // reset rtag of this bond
            unsigned int bond_tag = h_bond_tag.data[bond_idx];
            assert(bond_tag < getNumBondsGlobal());
            h_bond_rtag.data[bond_tag] = bond_idx;
            }
        }

    // due to removed bonds and duplicates, the new array size may be smaller
    m_bonds.resize(new_n_bonds);
    m_bond_type.resize(new_n_bonds);
    m_tags.resize(new_n_bonds);

    // set flag to indicate we have changed the bond table
    setDirty();

    if (m_prof) m_prof->pop();
    }

#ifdef ENABLE_CUDA
//! Pack bond data into a buffer (GPU version)
void BondData::retrieveBondsGPU(GPUVector<bond_element>& out)
    {
    if (m_prof) m_prof->push(m_exec_conf, "pack bonds");

    // access bond data arrays
    ArrayHandle<uint2> d_bonds(m_bonds, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_bond_type(getBondTypes(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_bond_tag(getBondTags(), access_location::device, access_mode::read);

    // access reverse-lookup table
    ArrayHandle<unsigned int> d_bond_rtag(getBondRTags(), access_location::device, access_mode::readwrite);

    unsigned int n_bonds = getNumBonds();

    // reset out vector
    out.clear();

    // count number of bonds with rtag==BOND_STAGED or rtag==BOND_SPLIT
    unsigned int n_pack_bonds = gpu_bdata_count_rtag_staged(
        n_bonds,
        d_bond_tag.data,
        d_bond_rtag.data,
        m_cached_alloc);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    // resize out vector
    out.resize(n_pack_bonds);

    // access output vector
    ArrayHandle<bond_element> d_out(out, access_location::device, access_mode::overwrite);

    // pack bonds on GPU
    gpu_pack_bonds(n_bonds, d_bond_tag.data, d_bonds.data, d_bond_type.data, d_bond_rtag.data, d_out.data, m_cached_alloc);
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    if (m_prof) m_prof->pop(m_exec_conf);
    }

//! Unpack a buffer with new bonds to be added, and remove obsolete bonds (GPU version)
void BondData::addRemoveBondsGPU(const GPUVector<bond_element>& in)
    {
    if (m_prof) m_prof->push(m_exec_conf, "unpack/remove bonds");

    unsigned int num_add_bonds = in.size();

    // old number of bonds
    unsigned int old_n_bonds = getNumBonds();

    // new number of bonds, before removal
    unsigned int new_n_bonds = old_n_bonds + num_add_bonds;

    // resize internal data structures to fit the new bonds
    m_bonds.resize(new_n_bonds);
    m_bond_type.resize(new_n_bonds);
    m_tags.resize(new_n_bonds);

        {
        // access bond data arrays
        ArrayHandle<uint2> d_bonds(getBondTable(), access_location::device, access_mode::readwrite);
        ArrayHandle<unsigned int> d_bond_tag(getBondTags(), access_location::device, access_mode::readwrite);
        ArrayHandle<unsigned int> d_bond_type(getBondTypes(), access_location::device, access_mode::readwrite);

        // access reverse-lookup table
        ArrayHandle<unsigned int> d_bond_rtag(getBondRTags(), access_location::device, access_mode::readwrite);

        // access input array
        ArrayHandle<bond_element> d_in(in, access_location::device, access_mode::read);

        // add new bonds, omitting duplicates, and remove bonds marked for deletion
        new_n_bonds = gpu_bdata_add_remove_bonds(
            old_n_bonds,
            num_add_bonds,
            d_bond_tag.data,
            d_bonds.data,
            d_bond_type.data,
            d_bond_rtag.data,
            d_in.data,
            m_cached_alloc);
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    // due to removed bonds and duplicates, the new array size may be smaller
    m_bonds.resize(new_n_bonds);
    m_bond_type.resize(new_n_bonds);
    m_tags.resize(new_n_bonds);

    // set flag to indicate we have changed the bond table
    setDirty();

    if (m_prof) m_prof->pop(m_exec_conf);
    }
#endif // ENABLE_CUDA

void BondData::moveParticleBonds(unsigned int tag, unsigned int old_rank, unsigned int new_rank)
    {
    unsigned int my_rank = m_exec_conf->getRank();

    // Number of bonds moved
    unsigned int n_bonds;

    std::vector<bond_element> buf;

    if (my_rank == old_rank)
        {
            {
            // access particle data reverse-lookup table
            ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

            // access bond tables
            ArrayHandle<uint2> h_bonds(getBondTable(), access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_bond_tag(getBondTags(), access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_bond_rtag(getBondRTags(), access_location::host, access_mode::readwrite);

            // mark all bonds that involve the particle
            unsigned int num_bonds = getNumBonds();
            for (unsigned int bond_idx = 0; bond_idx < num_bonds; ++bond_idx)
                {
                uint2 bond = h_bonds.data[bond_idx];

                // send bond only if it involves particle
                if (!(bond.x == tag || bond.y == tag))
                    continue;

                assert(bond.x < m_pdata->getNGlobal());
                assert(bond.y < m_pdata->getNGlobal());

                unsigned int rtag_a = h_rtag.data[bond.x];
                unsigned int rtag_b = h_rtag.data[bond.y];

                bool local = true;
                // if bond doesn't have any local members, discard it
                if ( !(rtag_a < m_pdata->getN()) && !(rtag_b < m_pdata->getN()))
                    local = false;

                unsigned int bond_tag = h_bond_tag.data[bond_idx];
                assert(bond_tag < getNumBondsGlobal());
                if (!local)
                    h_bond_rtag.data[bond_tag] = BOND_STAGED;
                else
                    h_bond_rtag.data[bond_tag] = BOND_SPLIT;
                } // end loop over bonds
            }

        // retrieve all bonds for particle being moved
        retrieveBonds(buf);

        // number of bonds being moved
        n_bonds = buf.size();

        // prune bond data
        addRemoveBonds(std::vector<bond_element>());
        }

    // Broadcast number of bonds moved
    bcast(n_bonds, old_rank, m_exec_conf->getMPICommunicator());

    // inform user
    m_exec_conf->msg->notice(6) << "Moving " << n_bonds << " bond(s) from rank " << old_rank << " to " << new_rank << std::endl;

    if (my_rank == old_rank)
        {
        MPI_Status stat;
        MPI_Request req;

        // send bond data
        MPI_Isend(&buf.front(),
            sizeof(bond_element)*n_bonds,
            MPI_BYTE,
            new_rank,
            1,
            m_exec_conf->getMPICommunicator(),
            &req);
        MPI_Waitall(1,&req,&stat);
        }
    else if (my_rank == new_rank)
        {
        MPI_Status stat;
        MPI_Request req;

        std::vector<bond_element> recv_buf(n_bonds);

        // receive bond data
        MPI_Irecv(&recv_buf.front(),
            sizeof(bond_element)*n_bonds,
            MPI_BYTE,
            old_rank,
            1,
            m_exec_conf->getMPICommunicator(),
            &req);
        MPI_Waitall(1,&req,&stat);

        // load bond data
        addRemoveBonds(recv_buf);
        }
    // done
    }
#endif

void export_BondData()
    {
    class_<Bond>("Bond", init<unsigned int, unsigned int, unsigned int>())
        .def_readonly("type", &Bond::type)
        .def_readonly("a", &Bond::a)
        .def_readonly("b", &Bond::b)
        ;

    class_< std::vector<uint2> >("std_vector_uint2")
    .def(vector_indexing_suite<std::vector<uint2> >())
    ;

    class_<BondData, boost::shared_ptr<BondData>, boost::noncopyable>("BondData", init<boost::shared_ptr<ParticleData>, unsigned int>())
    .def("getNumBonds", &BondData::getNumBonds)
    .def("getNumBondsGlobal", &BondData::getNumBondsGlobal)
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
    .def("addBondType", &BondData::addBondType)
    ;

    class_<SnapshotBondData, boost::shared_ptr<SnapshotBondData> >
    ("SnapshotBondData", init<unsigned int>())
    .def_readwrite("bonds", &SnapshotBondData::bonds)
    .def_readwrite("type_id", &SnapshotBondData::type_id)
    .def_readwrite("type_mapping", &SnapshotBondData::type_mapping)
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif
