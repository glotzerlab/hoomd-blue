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

#ifdef ENABLE_CUDA
#include "gpu_settings.h"
#endif

/*! \file AngleData.cc
    \brief Defines AngleData.
 */

/*! \param pdata ParticleData these angles refer into
    \param n_angle_types Number of angle types in the list

    Taking in pdata as a pointer instead of a shared pointer is sloppy, but there really isn't an alternative
    due to the way ParticleData is constructed. Things will be fixed in a later version with a reorganization
    of the various data structures. For now, be careful not to destroy the ParticleData and keep the AngleData hanging
    around.
*/
AngleData::AngleData(boost::shared_ptr<ParticleData> pdata, unsigned int n_angle_types)
        : m_n_angle_types(n_angle_types), m_angles_dirty(false), m_pdata(pdata)
    {
    assert(pdata);
    
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
        
#ifdef ENABLE_CUDA
    // get the execution configuration
    const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
    
    // init pointers
    m_host_angles = NULL;
    m_host_n_angles = NULL;
    m_gpu_angledata.resize(exec_conf.gpu.size());
    for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
        {
        m_gpu_angledata[cur_gpu].angles = NULL;
        m_gpu_angledata[cur_gpu].n_angles = NULL;
        m_gpu_angledata[cur_gpu].height = 0;
        m_gpu_angledata[cur_gpu].pitch = 0;
        }
        
    // allocate memory on the GPU if there is a GPU in the execution configuration
    if (exec_conf.gpu.size() >= 1)
        {
        allocateAngleTable(1);
        }
#endif
    }

AngleData::~AngleData()
    {
    m_sort_connection.disconnect();
    
#ifdef ENABLE_CUDA
    const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
    if (!exec_conf.gpu.empty())
        {
        freeAngleTable();
        }
#endif
    }

/*! \post An angle between particles specified in \a angle is created.

    \note Each angle should only be specified once! There are no checks to prevent one from being
    specified more than once, and doing so would result in twice the force and twice the energy.

    \note If an angle is added with \c type=49, then there must be at least 50 angle types (0-49) total,
    even if not all values are used. So angles should be added with small contiguous types.
    \param angle The Angle to add to the list
 */
void AngleData::addAngle(const Angle& angle)
    {
    
    // check for some silly errors a user could make
    if (angle.a >= m_pdata->getN() || angle.b >= m_pdata->getN() || angle.c >= m_pdata->getN())
        {
        cerr << endl << "***Error! Particle tag out of bounds when attempting to add angle: "
             << angle.a << ","
             << angle.b << ","
             << angle.c << endl << endl;
        throw runtime_error("Error adding angle");
        }
        
    if (angle.a == angle.b || angle.a == angle.c || angle.b == angle.c )
        {
        cerr << endl << "***Error! Particle cannot included in an angle twice! "
             << angle.a << ","
             << angle.b << ","
             << angle.c << endl << endl;
        throw runtime_error("Error adding angle");
        }
        
    // check that the type is within bouds
    if (angle.type+1 > m_n_angle_types)
        {
        cerr << endl << "***Error! Invalid angle type! "
             << angle.type << ", the number of types is " << m_n_angle_types << endl << endl;
        throw runtime_error("Error adding angle");
        }
        
    m_angles.push_back(angle);
    m_angles_dirty = true;
    }

/*! \param angle_type_mapping Mapping array to set
    \c angle_type_mapping[type] should be set to the name of the angle type with index \c type.
    The vector \b must have \c n_angle_types elements in it.
*/
void AngleData::setAngleTypeMapping(const std::vector<std::string>& angle_type_mapping)
    {
    assert(angle_type_mapping.size() == m_n_angle_types);
    m_angle_type_mapping = angle_type_mapping;
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
        
    cerr << endl << "***Error! Angle type " << name << " not found!" << endl;
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
    if (type >= m_n_angle_types)
        {
        cerr << endl << "***Error! Requesting type name for non-existant type " << type << endl << endl;
        throw runtime_error("Error mapping type name");
        }
        
    // return the name
    return m_angle_type_mapping[type];
    }

#ifdef ENABLE_CUDA

/*! Updates the angle data on the GPU if needed and returns the data structure needed to access it.
*/
std::vector<gpu_angletable_array>& AngleData::acquireGPU()
    {
    if (m_angles_dirty)
        {
        updateAngleTable();
        m_angles_dirty = false;
        }
    return m_gpu_angledata;
    }


/*! \post The angle tag data added via addAngle() is translated to angles based
    on particle index for use in the GPU kernel. This new angle table is then uploaded
    to the device.
*/
void AngleData::updateAngleTable()
    {
    
    assert(m_host_n_angles);
    assert(m_host_angles);
    
    // count the number of angles per particle
    // start by initializing the host n_angles values to 0
    memset(m_host_n_angles, 0, sizeof(unsigned int) * m_pdata->getN());
    
    // loop through the particles and count the number of angles based on each particle index
    // however, only the b atom in the a-b-c angle is included in the count.
    ParticleDataArraysConst arrays = m_pdata->acquireReadOnly();
    for (unsigned int cur_angle = 0; cur_angle < m_angles.size(); cur_angle++)
        {
        unsigned int tag1 = m_angles[cur_angle].a; //
        unsigned int tagb = m_angles[cur_angle].b;
        unsigned int tag3 = m_angles[cur_angle].c; //
        int idx1 = arrays.rtag[tag1]; //
        int idxb = arrays.rtag[tagb];
        int idx3 = arrays.rtag[tag3]; //
        
        m_host_n_angles[idx1]++; //
        m_host_n_angles[idxb]++;
        m_host_n_angles[idx3]++; //
        }
        
    // find the maximum number of angles
    unsigned int num_angles_max = 0;
    for (unsigned int i = 0; i < arrays.nparticles; i++)
        {
        if (m_host_n_angles[i] > num_angles_max)
            num_angles_max = m_host_n_angles[i];
        }
        
    // re allocate memory if needed
    if (num_angles_max > m_gpu_angledata[0].height)
        {
        reallocateAngleTable(num_angles_max);
        }
        
    // now, update the actual table
    // zero the number of angles counter (again)
    memset(m_host_n_angles, 0, sizeof(unsigned int) * m_pdata->getN());
    
    // loop through all angles and add them to each column in the list
    unsigned int pitch = m_pdata->getN(); //removed 'unsigned'
    for (unsigned int cur_angle = 0; cur_angle < m_angles.size(); cur_angle++)
        {
        unsigned int tag1 = m_angles[cur_angle].a;
        unsigned int tag2 = m_angles[cur_angle].b;
        unsigned int tag3 = m_angles[cur_angle].c;
        unsigned int type = m_angles[cur_angle].type;
        int idx1 = arrays.rtag[tag1];
        int idx2 = arrays.rtag[tag2];
        int idx3 = arrays.rtag[tag3];
        angleABC angle_type_abc;
        
        // get the number of angles for the b in a-b-c triplet
        int num1 = m_host_n_angles[idx1]; //
        int num2 = m_host_n_angles[idx2];
        int num3 = m_host_n_angles[idx3]; //
        
        // add a new angle to the table, provided each one is a "b" from an a-b-c triplet
        // store in the texture as .x=a=idx1, .y=c=idx2, and b comes from the gpu
        // or the cpu, generally from the idx2 index
        angle_type_abc = a_atom;
        m_host_angles[num1*pitch + idx1] = make_uint4(idx2, idx3, type, angle_type_abc); //
        
        angle_type_abc = b_atom;
        m_host_angles[num2*pitch + idx2] = make_uint4(idx1, idx3, type, angle_type_abc); // <-- WORKING LINE
        //m_host_angles[idx2] = make_uint4(idx1, idx3, type);  // <-- Probably not WORKING LINE
        
        angle_type_abc = c_atom;
        m_host_angles[num3*pitch + idx3] = make_uint4(idx1, idx2, type, angle_type_abc); //
        
        // increment the number of angles
        m_host_n_angles[idx1]++; //
        m_host_n_angles[idx2]++;
        m_host_n_angles[idx3]++; //
        }
        
    m_pdata->release();
    
    // copy the angle table to the device
    copyAngleTable();
    }

/*! \param height New height for the angle table
    \post Reallocates memory on the device making room for up to
        \a height angles per particle.
    \note updateAngleTable() needs to be called after so that the
        data in the angle table will be correct.
*/
void AngleData::reallocateAngleTable(int height)
    {
    freeAngleTable();
    allocateAngleTable(height);
    }

/*! \param height Height for the angle table
*/
void AngleData::allocateAngleTable(int height)
    {
    // make sure the arrays have been deallocated
    assert(m_host_angles == NULL);
    assert(m_host_n_angles == NULL);
    
    unsigned int N = m_pdata->getN();
    
    // get the execution configuration
    const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
    
    // allocate device memory
    exec_conf.tagAll(__FILE__, __LINE__);
    for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
        exec_conf.gpu[cur_gpu]->call(bind(&gpu_angletable_array::allocate, &m_gpu_angledata[cur_gpu], m_pdata->getLocalNum(cur_gpu), height));
        
        
    // allocate and zero host memory
    exec_conf.gpu[0]->call(bind(cudaHostAllocHack, (void**)((void*)&m_host_n_angles), N*sizeof(int), cudaHostAllocPortable));
    memset((void*)m_host_n_angles, 0, N*sizeof(int));
    
    exec_conf.gpu[0]->call(bind(cudaHostAllocHack, (void**)((void*)&m_host_angles), N * height * sizeof(uint4), cudaHostAllocPortable));
    memset((void*)m_host_angles, 0, N*height*sizeof(uint4));
    }

void AngleData::freeAngleTable()
    {
    // get the execution configuration
    const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
    
    // free device memory
    exec_conf.tagAll(__FILE__, __LINE__);
    for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
        exec_conf.gpu[cur_gpu]->call(bind(&gpu_angletable_array::deallocate, &m_gpu_angledata[cur_gpu]));
        
    // free host memory
    exec_conf.gpu[0]->call(bind(cudaFreeHost, m_host_angles));
    m_host_angles = NULL;
    exec_conf.gpu[0]->call(bind(cudaFreeHost, m_host_n_angles));
    m_host_n_angles = NULL;
    }

//! Copies the angle table to the device
void AngleData::copyAngleTable()
    {
    // get the execution configuration
    const ExecutionConfiguration& exec_conf = m_pdata->getExecConf();
    
    exec_conf.tagAll(__FILE__, __LINE__);
    for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
        {
        // we need to copy the table row by row since cudaMemcpy2D has severe pitch limitations
        for (unsigned int row = 0; row < m_gpu_angledata[0].height; row++)
            {
            exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, m_gpu_angledata[cur_gpu].angles + m_gpu_angledata[cur_gpu].pitch*row,
                                              m_host_angles + row * m_pdata->getN() + m_pdata->getLocalBeg(cur_gpu),
                                              sizeof(uint4) * m_pdata->getLocalNum(cur_gpu),
                                              cudaMemcpyHostToDevice));
                                              
            }
            
        exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, m_gpu_angledata[cur_gpu].n_angles,
                                          m_host_n_angles + m_pdata->getLocalBeg(cur_gpu),
                                          sizeof(unsigned int) * m_pdata->getLocalNum(cur_gpu),
                                          cudaMemcpyHostToDevice));
        }
    }
#endif

void export_AngleData()
    {
    class_<AngleData, boost::shared_ptr<AngleData>, boost::noncopyable>
    ("AngleData", init<boost::shared_ptr<ParticleData>, unsigned int>())
    .def("addAngle", &AngleData::addAngle)
    .def("getNumAngles", &AngleData::getNumAngles)
    .def("getNAngleTypes", &AngleData::getNAngleTypes)
    .def("getTypeByName", &AngleData::getTypeByName)
    .def("getNameByType", &AngleData::getNameByType)
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

