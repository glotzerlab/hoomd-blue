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

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include "DihedralData.h"
#include "ParticleData.h"

#include <boost/python.hpp>
using namespace boost::python;

#include <boost/bind.hpp>
using namespace boost;

#include <iostream>
#include <stdexcept>
using namespace std;

/*! \file DihedralData.cc
    \brief Defines DihedralData.
 */

/*! \param pdata ParticleData these dihedrals refer into
    \param n_dihedral_types Number of dihedral types in the list
*/
DihedralData::DihedralData(boost::shared_ptr<ParticleData> pdata, unsigned int n_dihedral_types) 
    :  m_dihedrals_dirty(false),
       m_pdata(pdata),
       exec_conf(m_pdata->getExecConf()),
       m_dihedrals(exec_conf),
       m_dihedral_type(exec_conf),
       m_tags(exec_conf),
       m_dihedral_rtag(exec_conf)
    {
    assert(pdata);
    m_exec_conf = m_pdata->getExecConf();
    m_exec_conf->msg->notice(5) << "Constructing DihedralData" << endl;

    // attach to the signal for notifications of particle sorts
    m_sort_connection = m_pdata->connectParticleSort(bind(&DihedralData::setDirty, this));
    
    // offer a default type mapping
    for (unsigned int i = 0; i < n_dihedral_types; i++)
        {
        char suffix[2];
        suffix[0] = 'A' + i;
        suffix[1] = '\0';
        
        string name = string("dihedral") + string(suffix);
        m_dihedral_type_mapping.push_back(name);
        }
        
    // allocate memory for the GPU dihedral table
    allocateDihedralTable(1);
    }

/*! \param pdata ParticleData these dihedrals refer into
    \param snapshot Snapshot to initialize DihedralData from
*/
DihedralData::DihedralData(boost::shared_ptr<ParticleData> pdata, const SnapshotDihedralData& snapshot) 
    :  m_dihedrals_dirty(false),
       m_pdata(pdata),
       exec_conf(m_pdata->getExecConf()),
       m_dihedrals(exec_conf),
       m_dihedral_type(exec_conf),
       m_tags(exec_conf),
       m_dihedral_rtag(exec_conf)
    {
    assert(pdata);
    m_exec_conf = m_pdata->getExecConf();
    m_exec_conf->msg->notice(5) << "Constructing DihedralData" << endl;

    // attach to the signal for notifications of particle sorts
    m_sort_connection = m_pdata->connectParticleSort(bind(&DihedralData::setDirty, this));
    
    // allocate memory for the GPU dihedral table
    allocateDihedralTable(1);

    // initialize from snapshot
    initializeFromSnapshot(snapshot);
    }


DihedralData::~DihedralData()
    {
    m_exec_conf->msg->notice(5) << "Destroying DihedralData" << endl;
    m_sort_connection.disconnect();
    }

/*! \post A dihedral between particles specified in \a dihedral is created.

    \note Each dihedral should only be specified once! There are no checks to prevent one from being
    specified more than once, and doing so would result in twice the force and twice the energy.

    \note If an dihedral is added with \c type=49, then there must be at least 50 dihedral types (0-49) total,
    even if not all values are used. So dihedrals should be added with small contiguous types.
    \param dihedral The Dihedral to add to the list
 */
unsigned int DihedralData::addDihedral(const Dihedral& dihedral)
    {
    #ifdef ENABLE_MPI
    // error out in multi-GPU
    if (m_pdata->getDomainDecomposition())
        {
        m_exec_conf->msg->error() << "Dihedrals/impropers are not currently supported in multi-GPU mode." << std::endl;
        throw std::runtime_error("Error adding dihedral/improper");
        }
    #endif

    // check for some silly errors a user could make
    if (dihedral.a >= m_pdata->getN() || dihedral.b >= m_pdata->getN() || dihedral.c >= m_pdata->getN()  || dihedral.d >= m_pdata->getN())
        {
        m_exec_conf->msg->error() << "Particle tag out of bounds when attempting to add dihedral/improper: " << dihedral.a << "," << dihedral.b << "," << dihedral.c << endl;
        throw runtime_error("Error adding dihedral/improper");
        }
        
    if (dihedral.a == dihedral.b || dihedral.a == dihedral.c || dihedral.b == dihedral.c || dihedral.a == dihedral.d || dihedral.b == dihedral.d || dihedral.c == dihedral.d )
        {
        m_exec_conf->msg->error() << "Particle cannot included in an dihedral/improper twice! " << dihedral.a << "," << dihedral.b << "," << dihedral.c << endl;
        throw runtime_error("Error adding dihedral/improper");
        }
        
    // check that the type is within bouds
    if (dihedral.type+1 > getNDihedralTypes())
        {
        m_exec_conf->msg->error() << "Invalid dihedral/improper type! " << dihedral.type << ", the number of types is " << getNDihedralTypes() << endl;
        throw runtime_error("Error adding dihedral/improper");
        }

    // first check if we can recycle a deleted tag
    unsigned int tag = 0;
    if (m_deleted_tags.size())
        {
        tag = m_deleted_tags.top();
        m_deleted_tags.pop();

        // update reverse-lookup tag
        m_dihedral_rtag[tag] = m_dihedrals.size();
        }
    else
        {
        // Otherwise, generate a new tag
        tag = m_dihedrals.size();

        // add a new reverse-lookup tag
        assert(m_dihedral_rtag.size() == m_dihedrals.size());
        m_dihedral_rtag.push_back(m_dihedrals.size());
        }

    assert(tag <= m_deleted_tags.size() + m_dihedrals.size());

    m_dihedrals.push_back(make_uint4(dihedral.a, dihedral.b, dihedral.c, dihedral.d));
    m_dihedral_type.push_back(dihedral.type);
    m_tags.push_back(tag);

    m_dihedrals_dirty = true;
    return tag;
    }

/*! \param tag tag of the dihedral to access
 */
const Dihedral DihedralData::getDihedralByTag(unsigned int tag) const
    {
    // Find position of dihedral in dihedralss list
    unsigned int dihedral_idx = m_dihedral_rtag[tag];
    if (dihedral_idx == NO_DIHEDRAL)
        {
        m_exec_conf->msg->error() << "Trying to get dihedral tag " << tag << " which does not exist!" << endl;
        throw runtime_error("Error getting dihedral");
        }
    uint4 dihedral = m_dihedrals[dihedral_idx];
    return Dihedral(m_dihedral_type[dihedral_idx], dihedral.x, dihedral.y, dihedral.z, dihedral.w);
    }

/*! \param id Index of dihedral (0 to N-1)
    \returns Unique tag of dihedral (for use when calling removeDihedral())
*/
unsigned int DihedralData::getDihedralTag(unsigned int id) const
    {
    if (id >= getNumDihedrals())
        {
        m_exec_conf->msg->error() << "Trying to get dihedral tag from id " << id << " which does not exist!" << endl;
        throw runtime_error("Error getting dihedral tag");
        }
    return m_tags[id];
    }

/*! \param tag tag of dihedral to remove
 * \note Dihedral removal changes the order of m_dihedrals. If a hole in the dihedral list
 * is generated, the last dihedral in the list is moved up to fill that hole.
 */
void DihedralData::removeDihedral(unsigned int tag)
    {
    // Find position of bond in bonds list
    unsigned int id = m_dihedral_rtag[tag];
    if (id == NO_DIHEDRAL)
        {
        m_exec_conf->msg->error() << "Trying to remove dihedral tag " << tag << " which does not exist!" << endl;
        throw runtime_error("Error removing dihedral");
        }

    // delete from map
    m_dihedral_rtag[tag] = NO_DIHEDRAL;

    unsigned int size = m_dihedrals.size();
    // If the bond is in the middle of the list, move the last element to
    // to the position of the removed element
    if (id < size-1)
        {
        m_dihedrals[id] = (uint4) m_dihedrals[size-1];
        m_dihedral_type[id] = (unsigned int) m_dihedral_type[size-1];
        unsigned int last_tag = m_tags[size-1];
        m_dihedral_rtag[last_tag] = id;
        m_tags[id] = last_tag;
        }
    // delete last element
    m_dihedrals.pop_back();
    m_dihedral_type.pop_back();
    m_tags.pop_back();

    // maintain a stack of deleted bond tags for future recycling
    m_deleted_tags.push(tag);

    m_dihedrals_dirty = true;
    }

/*! \param name Type name to get the index of
    \return Type index of the corresponding type name
    \note Throws an exception if the type name is not found
*/
unsigned int DihedralData::getTypeByName(const std::string &name)
    {
    // search for the name
    for (unsigned int i = 0; i < m_dihedral_type_mapping.size(); i++)
        {
        if (m_dihedral_type_mapping[i] == name)
            return i;
        }
        
    m_exec_conf->msg->error() << "Dihedral/Improper type " << name << " not found!" << endl;
    throw runtime_error("Error mapping type name");
    return 0;
    }

/*! \param type Type index to get the name of
    \returns Type name of the requested type
    \note Type indices must range from 0 to getNTypes or this method throws an exception.
*/
std::string DihedralData::getNameByType(unsigned int type)
    {
    // check for an invalid request
    if (type >= getNDihedralTypes()) 
        {
        m_exec_conf->msg->error() << "Requesting type name for non-existant type " << type << endl;
        throw runtime_error("Error mapping type name");
        }
        
    // return the name
    return m_dihedral_type_mapping[type];
    }


/*! Updates the dihedral data on the GPU if needed and returns the data structure needed to access it.
*/
const GPUArray<uint4>& DihedralData::getGPUDihedralList()
    {
    if (m_dihedrals_dirty)
        {
#ifdef ENABLE_CUDA
        if (exec_conf->isCUDAEnabled())
            updateDihedralTableGPU();
        else
            updateDihedralTable();
#else
        updateDihedralTable();
#endif
        m_dihedrals_dirty = false;
        }
    return m_gpu_dihedral_list;
    }


const GPUArray<uint1>& DihedralData::getDihedralABCD()
    {
    if (m_dihedrals_dirty)
        {
#ifdef ENABLE_CUDA
        if (exec_conf->isCUDAEnabled())
            updateDihedralTableGPU();
        else
            updateDihedralTable();
#else
        updateDihedralTable();
#endif
        m_dihedrals_dirty = false;
        }
    return m_dihedrals_ABCD;
    }

#ifdef ENABLE_CUDA
/*! Update GPU dihedral table (GPU version)

    \post The dihedral tag data added via addDihedral() is translated to dihedrals based
    on particle index for use in the GPU kernel. This new dihedral table is then uploaded
    to the device.
*/
void DihedralData::updateDihedralTableGPU()
    {
    unsigned int max_dihedral_num;

        {
        ArrayHandle<uint4> d_dihedrals(m_dihedrals, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_n_dihedrals(m_n_dihedrals, access_location::device, access_mode::overwrite);
        gpu_find_max_dihedral_number(max_dihedral_num,
                                     d_n_dihedrals.data,
                                     d_dihedrals.data,
                                     m_dihedrals.size(),
                                     m_pdata->getN(),
                                     d_rtag.data);
        }

    if (max_dihedral_num > m_gpu_dihedral_list.getHeight())
        {
        m_gpu_dihedral_list.resize(m_pdata->getN(), max_dihedral_num);
        m_dihedrals_ABCD.resize(m_pdata->getN(), max_dihedral_num);
        }

        {
        ArrayHandle<uint4> d_dihedrals(m_dihedrals, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_n_dihedrals(m_n_dihedrals, access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> d_dihedral_type (m_dihedral_type, access_location::device, access_mode::read);
        ArrayHandle<uint4> d_gpu_dihedral_list(m_gpu_dihedral_list, access_location::device, access_mode::overwrite);
        ArrayHandle<uint1> d_gpu_dihedral_ABCD(m_dihedrals_ABCD, access_location::device, access_mode::overwrite);
        gpu_create_dihedraltable(d_gpu_dihedral_list.data,
                                 d_gpu_dihedral_ABCD.data,
                                 d_n_dihedrals.data,
                                 d_dihedrals.data,
                                 d_dihedral_type.data,
                                 d_rtag.data,
                                 m_dihedrals.size(),
                                 m_gpu_dihedral_list.getPitch(),
                                 m_pdata->getN());
        }
    }
#endif

/*! Update GPU dihedral table (CPU version)

    \post The dihedral tag data added via addDihedral() is translated to dihedrals based
    on particle index for use in the GPU kernel. This new dihedral table is then uploaded
    to the device.
*/
void DihedralData::updateDihedralTable()
    {

    unsigned int num_dihedrals_max = 0;
    ArrayHandle< unsigned int > h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

        {
        ArrayHandle<unsigned int> h_n_dihedrals(m_n_dihedrals, access_location::host, access_mode::overwrite);

        // count the number of dihedrals per particle
        // start by initializing the host n_dihedrals values to 0
        memset(h_n_dihedrals.data, 0, sizeof(unsigned int) * m_pdata->getN());

        // loop through the particles and count the number of dihedrals based on each particle index
        // however, only the b atom in the a-b-c dihedral is included in the count.

        for (unsigned int cur_dihedral = 0; cur_dihedral < m_dihedrals.size(); cur_dihedral++)
            {
            unsigned int tag1 = ((uint4) m_dihedrals[cur_dihedral]).x;
            unsigned int tag2 = ((uint4) m_dihedrals[cur_dihedral]).y;
            unsigned int tag3 = ((uint4) m_dihedrals[cur_dihedral]).z;
            unsigned int tag4 = ((uint4) m_dihedrals[cur_dihedral]).w;
            int idx1 = h_rtag.data[tag1];
            int idx2 = h_rtag.data[tag2];
            int idx3 = h_rtag.data[tag3];
            int idx4 = h_rtag.data[tag4];

            h_n_dihedrals.data[idx1]++;
            h_n_dihedrals.data[idx2]++;
            h_n_dihedrals.data[idx3]++;
            h_n_dihedrals.data[idx4]++;
            }

        // find the maximum number of dihedrals
        unsigned int nparticles = m_pdata->getN();
        for (unsigned int i = 0; i < nparticles; i++)
            {
            if (h_n_dihedrals.data[i] > num_dihedrals_max)
                num_dihedrals_max = h_n_dihedrals.data[i];
            }
        }

    // re allocate memory if needed
    if (num_dihedrals_max > m_gpu_dihedral_list.getHeight())
        {
        m_gpu_dihedral_list.resize(m_pdata->getN(), num_dihedrals_max);
        m_dihedrals_ABCD.resize(m_pdata->getN(), num_dihedrals_max);
        }

        {
        ArrayHandle<unsigned int> h_n_dihedrals(m_n_dihedrals, access_location::host, access_mode::overwrite);
        ArrayHandle<uint4> h_gpu_dihedral_list(m_gpu_dihedral_list, access_location::host, access_mode::overwrite);
        ArrayHandle<uint1> h_dihedral_ABCD(m_dihedrals_ABCD, access_location::host, access_mode::overwrite);
        // now, update the actual table
        // zero the number of dihedrals counter (again)
        memset(h_n_dihedrals.data, 0, sizeof(unsigned int) * m_pdata->getN());

        // loop through all dihedrals and add them to each column in the list
        unsigned int pitch = m_gpu_dihedral_list.getPitch();
        for (unsigned int cur_dihedral = 0; cur_dihedral < m_dihedrals.size(); cur_dihedral++)
            {
            unsigned int tag1 = ((uint4) m_dihedrals[cur_dihedral]).x;
            unsigned int tag2 = ((uint4) m_dihedrals[cur_dihedral]).y;
            unsigned int tag3 = ((uint4) m_dihedrals[cur_dihedral]).z;
            unsigned int tag4 = ((uint4) m_dihedrals[cur_dihedral]).w;
            unsigned int type = m_dihedral_type[cur_dihedral];
            int idx1 = h_rtag.data[tag1];
            int idx2 = h_rtag.data[tag2];
            int idx3 = h_rtag.data[tag3];
            int idx4 = h_rtag.data[tag4];
            unsigned int dihedral_type_abcd;

            // get the number of dihedrals for the b in a-b-c triplet
            int num1 = h_n_dihedrals.data[idx1]; //
            int num2 = h_n_dihedrals.data[idx2];
            int num3 = h_n_dihedrals.data[idx3]; //
            int num4 = h_n_dihedrals.data[idx4]; //

            // add a new dihedral to the table, provided each one is a "b" from an a-b-c triplet
            // store in the texture as .x=a=idx1, .y=c=idx2, and b comes from the gpu
            // or the cpu, generally from the idx2 index
            dihedral_type_abcd = 0; // a atom
            h_gpu_dihedral_list.data[num1*pitch + idx1] = make_uint4(idx2, idx3, idx4, type); //
            h_dihedral_ABCD.data[num1*pitch + idx1] = make_uint1(dihedral_type_abcd);

            dihedral_type_abcd = 1; // b atom
            h_gpu_dihedral_list.data[num2*pitch + idx2] = make_uint4(idx1, idx3, idx4, type);
            h_dihedral_ABCD.data[num2*pitch + idx2] = make_uint1(dihedral_type_abcd);

            dihedral_type_abcd = 2; // c atom
            h_gpu_dihedral_list.data[num3*pitch + idx3] = make_uint4(idx1, idx2, idx4, type); //
            h_dihedral_ABCD.data[num3*pitch + idx3] = make_uint1(dihedral_type_abcd);

            dihedral_type_abcd = 3; // d atom
            h_gpu_dihedral_list.data[num4*pitch + idx4] = make_uint4(idx1, idx2, idx3, type); //
            h_dihedral_ABCD.data[num4*pitch + idx4] = make_uint1(dihedral_type_abcd);

            // increment the number of dihedrals
            h_n_dihedrals.data[idx1]++;
            h_n_dihedrals.data[idx2]++;
            h_n_dihedrals.data[idx3]++;
            h_n_dihedrals.data[idx4]++;
            }
        }
    }


/*! \param height Height for the dihedral table
*/
void DihedralData::allocateDihedralTable(int height)
    {
    assert(m_gpu_dihedral_list.isNull());
    assert(m_dihedrals_ABCD.isNull());
    assert(m_n_dihedrals.isNull());
    
    GPUArray<uint4> gpu_dihedral_list(m_pdata->getN(), height, exec_conf);
    m_gpu_dihedral_list.swap(gpu_dihedral_list);

    GPUArray<uint1> dihedrals_ABCD(m_pdata->getN(), height, exec_conf);
    m_dihedrals_ABCD.swap(dihedrals_ABCD);

    GPUArray<unsigned int> n_dihedrals(m_pdata->getN(), exec_conf);
    m_n_dihedrals.swap(n_dihedrals);
    }

//! Takes a snapshot of the current dihedral data
/*! \param snapshot THe snapshot that will contain the dihedral data
*/
void DihedralData::takeSnapshot(SnapshotDihedralData& snapshot)
    {
    // allocate memory
    snapshot.resize(getNumDihedrals());

    // check for an invalid request
    if (snapshot.dihedrals.size() != getNumDihedrals())
        {
        m_exec_conf->msg->error() << "DihedralData is being asked to initizalize a snapshot of the wrong size."
             << endl;
       throw runtime_error("Error taking snapshot.");
        }

    assert(snapshot.type_id.size() == getNumDihedrals());
    assert(snapshot.type_mapping.size() == 0);

    for (unsigned int dihedral_idx = 0; dihedral_idx < getNumDihedrals(); dihedral_idx++)
        {
        snapshot.dihedrals[dihedral_idx] = m_dihedrals[dihedral_idx];
        snapshot.type_id[dihedral_idx] = m_dihedral_type[dihedral_idx];
        }

    for (unsigned int i = 0; i < getNDihedralTypes(); i++)
        snapshot.type_mapping.push_back(m_dihedral_type_mapping[i]);
    }

//! Initialize the dihedral data from a snapshot
/*! \param snapshot The snapshot to initialize the dihedrals from
    Before initialization, the current angle data is cleared.
 */
void DihedralData::initializeFromSnapshot(const SnapshotDihedralData& snapshot)
    {
    // check that all fields in the snapshot have correct length
    if (m_exec_conf->getRank() == 0 && !snapshot.validate())
        {
        m_exec_conf->msg->error() << "init.*: inconsistent size of dihedral/improper data snapshot."
                                << std::endl << std::endl;
        throw std::runtime_error("Error initializing dihedral/improper data.");
        }

    m_dihedrals.clear();
    m_dihedral_type.clear();
    m_tags.clear();
    while (! m_deleted_tags.empty())
        m_deleted_tags.pop();
    m_dihedral_rtag.clear();

    m_dihedral_type_mapping = snapshot.type_mapping;

    for (unsigned int dihedral_idx = 0; dihedral_idx < snapshot.dihedrals.size(); dihedral_idx++)
        {
        Dihedral dihedral(snapshot.type_id[dihedral_idx],
                          snapshot.dihedrals[dihedral_idx].x,
                          snapshot.dihedrals[dihedral_idx].y,
                          snapshot.dihedrals[dihedral_idx].z,
                          snapshot.dihedrals[dihedral_idx].w);
        addDihedral(dihedral);
        }
    }

void export_DihedralData()
    {
    class_<DihedralData, boost::shared_ptr<DihedralData>, boost::noncopyable>("DihedralData", init<boost::shared_ptr<ParticleData>, unsigned int>())
    .def("addDihedral", &DihedralData::addDihedral)
    .def("getNumDihedrals", &DihedralData::getNumDihedrals)
    .def("getNDihedralTypes", &DihedralData::getNDihedralTypes)
    .def("getTypeByName", &DihedralData::getTypeByName)
    .def("getNameByType", &DihedralData::getNameByType)
    .def("removeDihedral", &DihedralData::removeDihedral)
    .def("getDihedral", &DihedralData::getDihedral)
    .def("getDihedralByTag", &DihedralData::getDihedralByTag)
    .def("getDihedralTag", &DihedralData::getDihedralTag)
    .def("takeSnapshot", &DihedralData::takeSnapshot)
    .def("initializeFromSnapshot", &DihedralData::initializeFromSnapshot)
    .def("addDihedralType", &DihedralData::addDihedralType)
    ;
    
    class_<Dihedral>("Dihedral", init<unsigned int, unsigned int, unsigned int, unsigned int, unsigned int>())
    .def_readwrite("a", &Dihedral::a)
    .def_readwrite("b", &Dihedral::b)
    .def_readwrite("c", &Dihedral::c)
    .def_readwrite("d", &Dihedral::d)
    .def_readwrite("type", &Dihedral::type)
    ;

    class_<SnapshotDihedralData, boost::shared_ptr<SnapshotDihedralData> >
        ("SnapshotDihedralData", init<unsigned int>())
        .def_readwrite("angles", &SnapshotDihedralData::dihedrals)
        .def_readwrite("type_id", &SnapshotDihedralData::type_id)
        .def_readwrite("type_mapping", &SnapshotDihedralData::type_mapping)
        ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

