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

#include "AngleData.h"
#include "ParticleData.h"

#include <boost/python.hpp>
using namespace boost::python;

#include <boost/bind.hpp>
using namespace boost;

#include <iostream>
#include <stdexcept>
using namespace std;

/*! \file AngleData.cc
    \brief Defines AngleData.
 */

/*! \param pdata ParticleData these angles refer into
    \param n_angle_types Number of angle types in the list
*/
AngleData::AngleData(boost::shared_ptr<ParticleData> pdata, unsigned int n_angle_types)
        : m_angles_dirty(false),
          m_pdata(pdata),
          exec_conf(m_pdata->getExecConf()),
          m_angles(exec_conf),
          m_angle_type(exec_conf),
          m_tags(exec_conf),
          m_angle_rtag(exec_conf)
    {
    assert(pdata);
    m_exec_conf = m_pdata->getExecConf();
    m_exec_conf->msg->notice(5) << "Constructing AngleData" << endl;

    // attach to the signal for notifications of particle sorts
    m_sort_connection = m_pdata->connectParticleSort(bind(&AngleData::setDirty, this));
    
    // offer a default type mapping
    for (unsigned int i = 0; i < n_angle_types; i++)
        {
        char suffix[2];
        suffix[0] = 'A' + i;
        suffix[1] = '\0';
        
        string name = string("angle") + string(suffix);
        m_angle_type_mapping.push_back(name);
        }
        
    // allocate memory for the GPU angle table
    allocateAngleTable(1);
    }

/*! \param pdata ParticleData these angles refer into
    \param snapshot Snapshot containing the angle data to initialize this class with
*/
AngleData::AngleData(boost::shared_ptr<ParticleData> pdata, const SnapshotAngleData& snapshot)
    : m_angles_dirty(false),
      m_pdata(pdata),
      exec_conf(m_pdata->getExecConf()),
      m_angles(exec_conf),
      m_angle_type(exec_conf),
      m_tags(exec_conf),
      m_angle_rtag(exec_conf)
   {
    assert(pdata);
    m_exec_conf = m_pdata->getExecConf();
    m_exec_conf->msg->notice(5) << "Constructing AngleData" << endl;

    // attach to the signal for notifications of particle sorts
    m_sort_connection = m_pdata->connectParticleSort(bind(&AngleData::setDirty, this));
    
    // allocate memory for the GPU angle table
    allocateAngleTable(1);

    // initialize from snapshot
    initializeFromSnapshot(snapshot);
    } 

AngleData::~AngleData()
    {
    m_exec_conf->msg->notice(5) << "Destroying AngleData" << endl;
    m_sort_connection.disconnect();
    }

/*! \post An angle between particles specified in \a angle is created.

    \note Each angle should only be specified once! There are no checks to prevent one from being
    specified more than once, and doing so would result in twice the force and twice the energy.

    \note If an angle is added with \c type=49, then there must be at least 50 angle types (0-49) total,
    even if not all values are used. So angles should be added with small contiguous types.
    \param angle The Angle to add to the list
 */
unsigned int AngleData::addAngle(const Angle& angle)
    {
    #ifdef ENABLE_MPI
    // error out in multi-GPU
    if (m_pdata->getDomainDecomposition())
        {
        m_exec_conf->msg->error() << "Angles are not currently supported in multi-processor simulations." << std::endl;
        throw std::runtime_error("Error adding angle");
        }
    #endif

    // check for some silly errors a user could make
    if (angle.a >= m_pdata->getN() || angle.b >= m_pdata->getN() || angle.c >= m_pdata->getN())
        {
        m_exec_conf->msg->error() << "Particle tag out of bounds when attempting to add angle: "
             << angle.a << ","
             << angle.b << ","
             << angle.c << endl;
        throw runtime_error("Error adding angle");
        }
        
    if (angle.a == angle.b || angle.a == angle.c || angle.b == angle.c )
        {
        m_exec_conf->msg->error() << "Particle cannot included in an angle twice! "
             << angle.a << ","
             << angle.b << ","
             << angle.c << endl;
        throw runtime_error("Error adding angle");
        }
        
    // check that the type is within bouds
    if (angle.type+1 > m_angle_type_mapping.size())
        {
        m_exec_conf->msg->error() << "Invalid angle type! "
             << angle.type << ", the number of types is " << m_angle_type_mapping.size() << endl;
        throw runtime_error("Error adding angle");
        }

    // first check if we can recycle a deleted tag
    unsigned int tag = 0;
    if (m_deleted_tags.size())
        {
        tag = m_deleted_tags.top();
        m_deleted_tags.pop();

        // update reverse-lookup tag
        m_angle_rtag[tag] = m_angles.size();
        }
    else
        {
        // Otherwise, generate a new tag
        tag = m_angles.size();

        // add new reverse-lookup tag
        assert(m_angle_rtag.size() == m_angles.size());
        m_angle_rtag.push_back(m_angles.size());
        }

    assert(tag <= m_deleted_tags.size() + m_angles.size());

    m_angles.push_back(make_uint3(angle.a,angle.b,angle.c));
    m_angle_type.push_back(angle.type);
    m_tags.push_back(tag);

    m_angles_dirty = true;
    return tag;
    }

/*! \param tag tag of the angle to access
 */
const Angle AngleData::getAngleByTag(unsigned int tag) const
    {
    // Find position of angle in angles list
    unsigned int angle_idx = m_angle_rtag[tag];
    if (angle_idx == NO_ANGLE)
        {
        m_exec_conf->msg->error() << "Trying to get angle tag " << tag << " which does not exist!" << endl;
        throw runtime_error("Error getting angle");
        }
    uint3 angle = m_angles[angle_idx];
    return Angle(m_angle_type[angle_idx], angle.x, angle.y, angle.z);
    }

/*! \param id Index of angle (0 to N-1)
    \returns Unique tag of angle (for use when calling removeAngle())
*/
unsigned int AngleData::getAngleTag(unsigned int id) const
    {
    if (id >= getNumAngles())
        {
        m_exec_conf->msg->error() << "Trying to get angle tag from id " << id << " which does not exist!" << endl;
        throw runtime_error("Error getting angle tag");
        }
    return m_tags[id];
    }

/*! \param tag tag of angle to remove
 * \note Angle removal changes the order of m_angles. If a hole in the angle list
 * is generated, the last angle in the list is moved up to fill that hole.
 */
void AngleData::removeAngle(unsigned int tag)
    {
    // Find position of angle in angles list
    unsigned int id = m_angle_rtag[tag];
    if (id == NO_ANGLE)
        {
        m_exec_conf->msg->error() << "Trying to remove angle tag " << tag << " which does not exist!" << endl;
        throw runtime_error("Error removing angle");
        }

    // delete from map
    m_angle_rtag[tag] = NO_ANGLE;

    unsigned int size = m_angles.size();
    // If the angle is in the middle of the list, move the last element to
    // to the position of the removed element
    if (id < (size-1))
        {
        m_angles[id] = (uint3) m_angles[size-1];
        m_angle_type[id] = (unsigned int) m_angle_type[size-1];
        unsigned int last_tag = m_tags[size-1];
        m_angle_rtag[last_tag] = id;
        m_tags[id] = last_tag;
        }
    // delete last element
    m_angles.pop_back();
    m_angle_type.pop_back();
    m_tags.pop_back();

    // maintain a stack of deleted angle tags for future recycling
    m_deleted_tags.push(tag);

    m_angles_dirty = true;
    }

/*! \param name Type name to get the index of
    \return Type index of the corresponding type name
    \note Throws an exception if the type name is not found
*/
unsigned int AngleData::getTypeByName(const std::string &name)
    {
    // search for the name
    for (unsigned int i = 0; i < m_angle_type_mapping.size(); i++)
        {
        if (m_angle_type_mapping[i] == name)
            return i;
        }
        
    m_exec_conf->msg->error() << "Angle type " << name << " not found!" << endl;
    throw runtime_error("Error mapping type name");
    return 0;
    }

/*! \param type Type index to get the name of
    \returns Type name of the requested type
    \note Type indices must range from 0 to getNTypes or this method throws an exception.
*/
std::string AngleData::getNameByType(unsigned int type)
    {
    // check for an invalid request
    if (type >= getNAngleTypes())
        {
        m_exec_conf->msg->error() << "Requesting type name for non-existant type " << type << endl;
        throw runtime_error("Error mapping type name");
        }
        
    // return the name
    return m_angle_type_mapping[type];
    }


/*! Updates the angle data on the GPU if needed and returns the data structure needed to access it.
*/
const GPUArray<uint4>& AngleData::getGPUAngleList()
    {
    if (m_angles_dirty)
        {
#ifdef ENABLE_CUDA
        // update angle table
        if (exec_conf->isCUDAEnabled())
            updateAngleTableGPU();
        else
            updateAngleTable();
#else
        updateAngleTable();
#endif
        m_angles_dirty = false;
        }
    return m_gpu_anglelist;
    }


#ifdef ENABLE_CUDA
    
/*! Update GPU angle table (GPU version)

    \post The angle tag data added via addAngle() is translated to angles based
    on particle index for use in the GPU kernel. This new angle table is then uploaded
    to the device.
*/
void AngleData::updateAngleTableGPU()
    {
    unsigned int max_angle_num;
    
        {
        ArrayHandle<uint3> d_angles(m_angles, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_n_angles(m_n_angles, access_location::device, access_mode::overwrite);
        gpu_find_max_angle_number(max_angle_num,
                                 d_n_angles.data,
                                 d_angles.data,
                                 m_angles.size(),
                                 m_pdata->getN(),
                                 d_rtag.data);
        }

    if (max_angle_num > m_gpu_anglelist.getHeight())
        {
        m_gpu_anglelist.resize(m_pdata->getN(), max_angle_num);
        }

        {
        ArrayHandle<uint4> d_gpu_anglelist(m_gpu_anglelist, access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> d_angle_type(m_angle_type, access_location::device, access_mode::read);
        ArrayHandle<uint3> d_angles(m_angles, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_n_angles(m_n_angles, access_location::device, access_mode::overwrite);

        gpu_create_angletable(d_gpu_anglelist.data,
                              d_n_angles.data,
                              d_angles.data,
                              d_angle_type.data,
                              d_rtag.data,
                              m_angles.size(),
                              m_gpu_anglelist.getPitch(),
                              m_pdata->getN() );
        }
    }

#endif

/*! \post The angle tag data added via addAngle() is translated to angles based
    on particle index for use in the GPU kernel. This new angle table is then uploaded
    to the device.
*/
void AngleData::updateAngleTable()
    {
    ArrayHandle< unsigned int > h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
    unsigned int num_angles_max = 0;

        {
        ArrayHandle<unsigned int> h_n_angles(m_n_angles, access_location::host, access_mode::overwrite);

        // count the number of angles per particle
        // start by initializing the host n_angles values to 0
        memset(h_n_angles.data, 0, sizeof(unsigned int) * m_pdata->getN());

        // loop through the particles and count the number of angles based on each particle index
        // however, only the b atom in the a-b-c angle is included in the count.

        for (unsigned int cur_angle = 0; cur_angle < m_angles.size(); cur_angle++)
            {
            unsigned int tag1 = ((uint3)m_angles[cur_angle]).x; //
            unsigned int tagb = ((uint3)m_angles[cur_angle]).y;
            unsigned int tag3 = ((uint3)m_angles[cur_angle]).z; //
            unsigned int idx1 = h_rtag.data[tag1]; //
            unsigned int idxb = h_rtag.data[tagb];
            unsigned int idx3 = h_rtag.data[tag3]; //

            h_n_angles.data[idx1]++; //
            h_n_angles.data[idxb]++;
            h_n_angles.data[idx3]++; //
            }

        // find the maximum number of angles
        unsigned int nparticles = m_pdata->getN();
        for (unsigned int i = 0; i < nparticles; i++)
            {
            if (h_n_angles.data[i] > num_angles_max)
                num_angles_max = h_n_angles.data[i];
            }
        }

    // re allocate memory if needed
    if (num_angles_max > m_gpu_anglelist.getHeight())
        {
        m_gpu_anglelist.resize(m_pdata->getN(), num_angles_max);
        }

        {
        ArrayHandle<unsigned int> h_n_angles(m_n_angles, access_location::host, access_mode::overwrite);
        ArrayHandle<uint4> h_gpu_anglelist(m_gpu_anglelist, access_location::host, access_mode::overwrite);

        // now, update the actual table
        // zero the number of angles counter (again)
        memset(h_n_angles.data, 0, sizeof(unsigned int) * m_pdata->getN());

        // loop through all angles and add them to each column in the list
        unsigned int pitch = m_gpu_anglelist.getPitch();
        for (unsigned int cur_angle = 0; cur_angle < m_angles.size(); cur_angle++)
            {
            unsigned int tag1 = ((uint3)m_angles[cur_angle]).x;
            unsigned int tag2 = ((uint3)m_angles[cur_angle]).y;
            unsigned int tag3 = ((uint3)m_angles[cur_angle]).z;
            unsigned int type = m_angle_type[cur_angle];
            unsigned int idx1 = h_rtag.data[tag1];
            unsigned int idx2 = h_rtag.data[tag2];
            unsigned int idx3 = h_rtag.data[tag3];
            unsigned int angle_type_abc;

            // get the number of angles for the b in a-b-c triplet
            unsigned int num1 = h_n_angles.data[idx1]; //
            unsigned int num2 = h_n_angles.data[idx2];
            unsigned int num3 = h_n_angles.data[idx3]; //

            // add a new angle to the table, provided each one is a "b" from an a-b-c triplet
            // store in the texture as .x=a=idx1, .y=c=idx2, and b comes from the gpu
            // or the cpu, generally from the idx2 index
            angle_type_abc = 0; // a atom
            h_gpu_anglelist.data[num1*pitch + idx1] = make_uint4(idx2, idx3, type, angle_type_abc);

            angle_type_abc = 1; // b atom
            h_gpu_anglelist.data[num2*pitch + idx2] = make_uint4(idx1, idx3, type, angle_type_abc);

            angle_type_abc = 2; // c atom
            h_gpu_anglelist.data[num3*pitch + idx3] = make_uint4(idx1, idx2, type, angle_type_abc);

            // increment the number of angles
            h_n_angles.data[idx1]++;
            h_n_angles.data[idx2]++;
            h_n_angles.data[idx3]++;
            }
        }
    }

/*! \param height Height for the angle table
*/
void AngleData::allocateAngleTable(int height)
    {
    // make sure the arrays have been deallocated
    assert(m_n_angles.isNull());

    
    // allocate device memory
    GPUArray<uint4> gpu_anglelist(m_pdata->getN(), height, exec_conf);
    m_gpu_anglelist.swap(gpu_anglelist);

    GPUArray<unsigned int> n_angles(m_pdata->getN(), exec_conf);
    m_n_angles.swap(n_angles);
    }

//! Takes a snapshot of the current angle data
/*! \param snapshot The snapshot that will contain the angle data
*/
void AngleData::takeSnapshot(SnapshotAngleData& snapshot)
    {
    // check for an invalid request
    if (snapshot.angles.size() != getNumAngles())
        {
        m_exec_conf->msg->error() << "AngleData is being asked to initizalize a snapshot of the wrong size."
             << endl;
        throw runtime_error("Error taking snapshot.");
        }

    assert(snapshot.type_id.size() == getNumAngles());
    assert(snapshot.type_mapping.size() == 0);

    for (unsigned int angle_idx = 0; angle_idx < getNumAngles(); angle_idx++)
        {
        snapshot.angles[angle_idx] = m_angles[angle_idx];
        snapshot.type_id[angle_idx] = m_angle_type[angle_idx];
        }

    for (unsigned int i = 0; i < getNAngleTypes(); i++)
        snapshot.type_mapping.push_back(m_angle_type_mapping[i]);
    }

//! Initialize the angle data from a snapshot
/*! \param snapshot The snapshot to initialize the angles from
    Before initialization, the current angle data is cleared.
 */
void AngleData::initializeFromSnapshot(const SnapshotAngleData& snapshot)
    {
    m_angles.clear();
    m_angle_type.clear();
    m_tags.clear();
    while (! m_deleted_tags.empty())
        m_deleted_tags.pop();
    m_angle_rtag.clear();

    m_angle_type_mapping =  snapshot.type_mapping;
    
    for (unsigned int angle_idx = 0; angle_idx < snapshot.angles.size(); angle_idx++)
        {
        Angle angle(snapshot.type_id[angle_idx], snapshot.angles[angle_idx].x, snapshot.angles[angle_idx].y, snapshot.angles[angle_idx].z);
        addAngle(angle);
        }
    }

void export_AngleData()
    {
    class_<AngleData, boost::shared_ptr<AngleData>, boost::noncopyable>
    ("AngleData", init<boost::shared_ptr<ParticleData>, unsigned int>())
    .def("addAngle", &AngleData::addAngle)
    .def("getNumAngles", &AngleData::getNumAngles)
    .def("getNAngleTypes", &AngleData::getNAngleTypes)
    .def("getTypeByName", &AngleData::getTypeByName)
    .def("getNameByType", &AngleData::getNameByType)
    .def("removeAngle", &AngleData::removeAngle)
    .def("getAngle", &AngleData::getAngle)
    .def("getAngleByTag", &AngleData::getAngleByTag)
    .def("getAngleTag", &AngleData::getAngleTag)
    .def("takeSnapshot", &AngleData::takeSnapshot)
    .def("initializeFromSnapshot", &AngleData::initializeFromSnapshot)
    .def("addAngleType", &AngleData::addAngleType)
    ;
    
    class_<Angle>("Angle", init<unsigned int, unsigned int, unsigned int, unsigned int>())
    .def_readwrite("a", &Angle::a)
    .def_readwrite("b", &Angle::b)
    .def_readwrite("c", &Angle::c)
    .def_readwrite("type", &Angle::type)
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

