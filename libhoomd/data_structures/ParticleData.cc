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

/*! \file ParticleData.cc
    \brief Contains all code for BoxDim, ParticleData, and SnapshotParticleData.
 */

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 4267 )
#endif

#include <iostream>
#include <cassert>
#include <stdlib.h>
#include <stdexcept>
#include <sstream>
#include <iomanip>

using namespace std;

#include <boost/python.hpp>
using namespace boost::python;

#include "ParticleData.h"
#include "Profiler.h"
#include "AngleData.h"
#include "DihedralData.h"

#include <boost/bind.hpp>

using namespace boost::signals;
using namespace boost;

///////////////////////////////////////////////////////////////////////////
// BoxDim constructors

/*! \post All dimensions are 0.0
*/
BoxDim::BoxDim()
    {
    xlo = xhi = ylo = yhi = zlo = zhi = 0.0;
    }

/*! \param Len Length of one side of the box
    \post Box ranges from \c -Len/2 to \c +Len/2 in all 3 dimensions
 */
BoxDim::BoxDim(Scalar Len)
    {
    // sanity check
    assert(Len > 0);
    
    // assign values
    xlo = ylo = zlo = -Len/Scalar(2.0);
    xhi = zhi = yhi = Len/Scalar(2.0);
    }

/*! \param Len_x Length of the x dimension of the box
    \param Len_y Length of the x dimension of the box
    \param Len_z Length of the x dimension of the box
 */
BoxDim::BoxDim(Scalar Len_x, Scalar Len_y, Scalar Len_z)
    {
    // sanity check
    assert(Len_x > 0 && Len_y > 0 && Len_z > 0);
    
    // assign values
    xlo = -Len_x/Scalar(2.0);
    xhi = Len_x/Scalar(2.0);
    
    ylo = -Len_y/Scalar(2.0);
    yhi = Len_y/Scalar(2.0);
    
    zlo = -Len_z/Scalar(2.0);
    zhi = Len_z/Scalar(2.0);
    }

////////////////////////////////////////////////////////////////////////////
// ParticleData members

/*! \param N Number of particles to allocate memory for
    \param n_types Number of particle types that will exist in the data arrays
    \param box Box the particles live in
    \param exec_conf ExecutionConfiguration to use when executing code on the GPU

    \post \c x,\c y,\c z,\c vx,\c vy,\c vz,\c ax,\c ay, and \c az are allocated and initialized to 0.0
    \post \c charge is allocated and initialized to a value of 0.0
    \post \c diameter is allocated and initialized to a value of 1.0
    \post \c mass is allocated and initialized to a value of 1.0
    \post \c ix, \c iy, \c iz are allocated and initialized to values of 0.0
    \post \c rtag is allocated and given the default initialization rtag[i] = i
    \post \c tag is allocated and given the default initialization tag[i] = i
    \post \c type is allocated and given the default value of type[i] = 0
    \post \c body is allocated and given the devault value of type[i] = NO_BODY
    \post Arrays are not currently acquired

    Type mappings assign particle types "A", "B", "C", ....
*/
ParticleData::ParticleData(unsigned int N, const BoxDim &box, unsigned int n_types, boost::shared_ptr<ExecutionConfiguration> exec_conf)
        : m_box(box), m_exec_conf(exec_conf), m_data(NULL), m_nbytes(0), m_ntypes(n_types)
    {
    m_exec_conf->msg->notice(5) << "Constructing ParticleData" << endl;

    // check the input for errors
    if (m_ntypes == 0)
        {
        m_exec_conf->msg->error() << "Number of particle types must be greater than 0." << endl;
        throw std::runtime_error("Error initializing ParticleData");
        }
        
    // allocate memory
    allocate(N);

    ArrayHandle< Scalar4 > h_vel(getVelocities(), access_location::host, access_mode::overwrite);
    ArrayHandle< Scalar > h_diameter(getDiameters(), access_location::host, access_mode::overwrite);
    ArrayHandle< unsigned int > h_tag(getTags(), access_location::host, access_mode::overwrite);
    ArrayHandle< unsigned int > h_rtag(getRTags(), access_location::host, access_mode::overwrite);
    ArrayHandle< unsigned int > h_body(getBodies(), access_location::host, access_mode::overwrite);
    ArrayHandle< Scalar4 > h_orientation(m_orientation, access_location::host, access_mode::readwrite);
    
    // set default values
    // all values not explicitly set here have been initialized to zero upon allocation
    for (unsigned int i = 0; i < N; i++)
        {
        h_vel.data[i].w = 1.0; // mass

        h_diameter.data[i] = 1.0;
        
        h_body.data[i] = NO_BODY;
        h_rtag.data[i] = i;
        h_tag.data[i] = i;
        h_orientation.data[i] = make_scalar4(1.0, 0.0, 0.0, 0.0);
        }
        
    // default constructed shared ptr is null as desired
    m_prof = boost::shared_ptr<Profiler>();
    
    // setup the type mappings
    for (unsigned int i = 0; i < m_ntypes; i++)
        {
        char name[2];
        name[0] = 'A' + i;
        name[1] = '\0';
        m_type_mapping.push_back(string(name));
        }
        
    // if this is a GPU build, initialize the graphics card mirror data structures
#ifdef ENABLE_CUDA
    if (m_exec_conf->isCUDAEnabled())
        {
        // setup the box
        m_gpu_box.Lx = m_box.xhi - m_box.xlo;
        m_gpu_box.Ly = m_box.yhi - m_box.ylo;
        m_gpu_box.Lz = m_box.zhi - m_box.zlo;
        m_gpu_box.Lxinv = 1.0f / m_gpu_box.Lx;
        m_gpu_box.Lyinv = 1.0f / m_gpu_box.Ly;
        m_gpu_box.Lzinv = 1.0f / m_gpu_box.Lz;
        }
#endif
    }

/*! Calls the initializer's members to determine the number of particles, box size and then
    uses it to fill out the position and velocity data.
    \param init Initializer to use
    \param exec_conf Execution configuration to run on
*/
ParticleData::ParticleData(const ParticleDataInitializer& init, boost::shared_ptr<ExecutionConfiguration> exec_conf) : m_exec_conf(exec_conf), m_data(NULL), m_nbytes(0), m_ntypes(0)
    {
    m_exec_conf->msg->notice(5) << "Constructing ParticleData" << endl;

    m_ntypes = init.getNumParticleTypes();
    // check the input for errors
    if (m_ntypes == 0)
        {
        m_exec_conf->msg->error() << "Number of particle types must be greater than 0." << endl;
        throw std::runtime_error("Error initializing ParticleData");
        }
        
    // allocate memory
    allocate(init.getNumParticles());
    
        {
        ArrayHandle< Scalar4 > h_orientation(m_orientation, access_location::host, access_mode::readwrite);
        
        ArrayHandle< Scalar4 > h_vel(getVelocities(), access_location::host, access_mode::overwrite);
        ArrayHandle< Scalar > h_diameter(getDiameters(), access_location::host, access_mode::overwrite);
        ArrayHandle< unsigned int > h_tag(getTags(), access_location::host, access_mode::overwrite);
        ArrayHandle< unsigned int > h_rtag(getRTags(), access_location::host, access_mode::overwrite);
        ArrayHandle< unsigned int > h_body(getBodies(), access_location::host, access_mode::overwrite);

        // set default values
        // all values not explicitly set here have been initialized to zero upon allocation
        for (unsigned int i = 0; i < getN(); i++)
            {
            h_vel.data[i].w = 1.0; // mass

            h_diameter.data[i] = 1.0;
            
            h_body.data[i] = NO_BODY;
            h_rtag.data[i] = i;
            h_tag.data[i] = i;
            h_orientation.data[i] = make_scalar4(1.0, 0.0, 0.0, 0.0);
            }
        }
        
    setBox(init.getBox());        
    SnapshotParticleData snapshot(getN());
    // initialize the snapshot with default values
    takeSnapshot(snapshot);
    // pass snapshot to initializer
    init.initSnapshot(snapshot);
    // initialize particle data with updated values
    initializeFromSnapshot(snapshot);

        {
        ArrayHandle<Scalar4> h_orientation(getOrientationArray(), access_location::host, access_mode::overwrite);
        init.initOrientation(h_orientation.data);
        init.initMomentInertia(&m_inertia_tensor[0]);
        }
            
    // it is an error for particles to be initialized outside of their box
    if (!inBox())
        {
        m_exec_conf->msg->error() << "Not all particles were found inside the given box" << endl;
        throw runtime_error("Error initializing ParticleData");
        }
        
    // assign the type mapping
    m_type_mapping  = init.getTypeMapping();
    
    // default constructed shared ptr is null as desired
    m_prof = boost::shared_ptr<Profiler>();
    }

ParticleData::~ParticleData()
    {
    m_exec_conf->msg->notice(5) << "Destroying ParticleData" << endl;
    }

/*! \return Simulation box dimensions
 */
const BoxDim & ParticleData::getBox() const
    {
    return m_box;
    }

/*! \param box New box to set
    \note ParticleData does NOT enforce any boundary conditions. When a new box is set,
        it is the responsibility of the caller to ensure that all particles lie within
        the new box.
*/
void ParticleData::setBox(const BoxDim &box)
    {
    m_box = box;
    assert(inBox());
    
#ifdef ENABLE_CUDA
    // setup the box
    m_gpu_box.Lx = m_box.xhi - m_box.xlo;
    m_gpu_box.Ly = m_box.yhi - m_box.ylo;
    m_gpu_box.Lz = m_box.zhi - m_box.zlo;
    m_gpu_box.Lxinv = 1.0f / m_gpu_box.Lx;
    m_gpu_box.Lyinv = 1.0f / m_gpu_box.Ly;
    m_gpu_box.Lzinv = 1.0f / m_gpu_box.Lz;
#endif
    
    m_boxchange_signal();
    }

/*! \param func Function to call when the particles are resorted
    \return Connection to manage the signal/slot connection
    Calls are performed by using boost::signals. The function passed in
    \a func will be called every time the ParticleData is notified of a particle
    sort via notifyParticleSort().
    \note If the caller class is destroyed, it needs to disconnect the signal connection
    via \b con.disconnect where \b con is the return value of this function.
*/
boost::signals::connection ParticleData::connectParticleSort(const boost::function<void ()> &func)
    {
    return m_sort_signal.connect(func);
    }

/*! \b ANY time particles are rearranged in memory, this function must be called.
    \note The call must be made after calling release()
*/
void ParticleData::notifyParticleSort()
    {
    m_sort_signal();
    }

/*! \param func Function to call when the box size changes
    \return Connection to manage the signal/slot connection
    Calls are performed by using boost::signals. The function passed in
    \a func will be called every time the the box size is changed via setBox()
    \note If the caller class is destroyed, it needs to disconnect the signal connection
    via \b con.disconnect where \b con is the return value of this function.
*/
boost::signals::connection ParticleData::connectBoxChange(const boost::function<void ()> &func)
    {
    return m_boxchange_signal.connect(func);
    }

/*! \param name Type name to get the index of
    \return Type index of the corresponding type name
    \note Throws an exception if the type name is not found
*/
unsigned int ParticleData::getTypeByName(const std::string &name) const
    {
    assert(m_type_mapping.size() == m_ntypes);
    // search for the name
    for (unsigned int i = 0; i < m_type_mapping.size(); i++)
        {
        if (m_type_mapping[i] == name)
            return i;
        }
        
    m_exec_conf->msg->error() << "Type " << name << " not found!" << endl;
    throw runtime_error("Error mapping type name");
    return 0;
    }

/*! \param type Type index to get the name of
    \returns Type name of the requested type
    \note Type indices must range from 0 to getNTypes or this method throws an exception.
*/
std::string ParticleData::getNameByType(unsigned int type) const
    {
    assert(m_type_mapping.size() == m_ntypes);
    // check for an invalid request
    if (type >= m_ntypes)
        {
        m_exec_conf->msg->error() << "Requesting type name for non-existant type " << type << endl;
        throw runtime_error("Error mapping type name");
        }
        
    // return the name
    return m_type_mapping[type];
    }


/*! \param N Number of particles to allocate memory for
    \pre No memory is allocated and the per-particle GPUArrays are unitialized
    \post All per-perticle GPUArrays are allocated
*/
void ParticleData::allocate(unsigned int N)
    {
    // check the input
    if (N == 0)
        {
        m_exec_conf->msg->error() << "ParticleData is being asked to allocate 0 particles.... this makes no sense whatsoever" << endl;
        throw runtime_error("Error allocating ParticleData");
        }

    // set particle number
    m_nparticles = N;

    // positions
    GPUArray< Scalar4 > pos(getN(), m_exec_conf);
    m_pos.swap(pos);

    // velocities
    GPUArray< Scalar4 > vel(getN(), m_exec_conf);
    m_vel.swap(vel);

    // accelerations
    GPUArray< Scalar3 > accel(getN(), m_exec_conf);
    m_accel.swap(accel);

    // charge
    GPUArray< Scalar > charge(getN(), m_exec_conf);
    m_charge.swap(charge);

    // diameter
    GPUArray< Scalar > diameter(getN(), m_exec_conf);
    m_diameter.swap(diameter);

    // image
    GPUArray< int3 > image(getN(), m_exec_conf);
    m_image.swap(image);

    // tag
    GPUArray< unsigned int > tag(getN(), m_exec_conf);
    m_tag.swap(tag);

    // reverse-lookup tag
    GPUArray< unsigned int > rtag(getN(), m_exec_conf);
    m_rtag.swap(rtag);

    // body ID
    GPUArray< unsigned int > body(getN(), m_exec_conf);
    m_body.swap(body);

    GPUArray< Scalar4 > net_force(getN(), m_exec_conf);
    m_net_force.swap(net_force);
    GPUArray< Scalar > net_virial(getN(),6, m_exec_conf);
    m_net_virial.swap(net_virial);
    GPUArray< Scalar4 > net_torque(getN(), m_exec_conf);
    m_net_torque.swap(net_torque);
    GPUArray< Scalar4 > orientation(getN(), m_exec_conf);
    m_orientation.swap(orientation);
    m_inertia_tensor.resize(getN());
    }

/*! \return true If and only if all particles are in the simulation box
    \note This function is only called in debug builds
*/
bool ParticleData::inBox()
    {

    ArrayHandle<Scalar4> h_pos(getPositions(), access_location::host, access_mode::read);
    for (unsigned int i = 0; i < getN(); i++)
        {
        if (h_pos.data[i].x < m_box.xlo-Scalar(1e-5) || h_pos.data[i].x > m_box.xhi+Scalar(1e-5))
            {
            m_exec_conf->msg->notice(1) << "pos " << i << ":" << setprecision(12) << h_pos.data[i].x << " " << h_pos.data[i].y << " " << h_pos.data[i].z << endl;
            m_exec_conf->msg->notice(1) << "lo: " << m_box.xlo << " " << m_box.ylo << " " << m_box.zlo << endl;
            m_exec_conf->msg->notice(1) << "hi: " << m_box.xhi << " " << m_box.yhi << " " << m_box.zhi << endl;
            return false;
            }
        if (h_pos.data[i].y < m_box.ylo-Scalar(1e-5) || h_pos.data[i].y > m_box.yhi+Scalar(1e-5))
            {
            m_exec_conf->msg->notice(1) << "pos " << i << ":" << setprecision(12) << h_pos.data[i].x << " " << h_pos.data[i].y << " " << h_pos.data[i].z << endl;
            m_exec_conf->msg->notice(1) << "lo: " << m_box.xlo << " " << m_box.ylo << " " << m_box.zlo << endl;
            m_exec_conf->msg->notice(1) << "hi: " << m_box.xhi << " " << m_box.yhi << " " << m_box.zhi << endl;
            return false;
            }
        if (h_pos.data[i].z < m_box.zlo-Scalar(1e-5) || h_pos.data[i].z > m_box.zhi+Scalar(1e-5))
            {
            m_exec_conf->msg->notice(1) << "pos " << i << ":" << setprecision(12) << h_pos.data[i].x << " " << h_pos.data[i].y << " " << h_pos.data[i].z << endl;
            m_exec_conf->msg->notice(1) << "lo: " << m_box.xlo << " " << m_box.ylo << " " << m_box.zlo << endl;
            m_exec_conf->msg->notice(1) << "hi: " << m_box.xhi << " " << m_box.yhi << " " << m_box.zhi << endl;
            return false;
            }
        }
    return true;
    }

//! Initialize from a snapshot
//! \param snapshot the initial particle data
//! \post the particle data arrays are initialized from the snapshot, in sorted order
void ParticleData::initializeFromSnapshot(const SnapshotParticleData& snapshot)
    {
    ArrayHandle< Scalar4 > h_pos(m_pos, access_location::host, access_mode::overwrite);
    ArrayHandle< Scalar4 > h_vel(m_vel, access_location::host, access_mode::overwrite);
    ArrayHandle< Scalar3 > h_accel(m_accel, access_location::host, access_mode::overwrite);
    ArrayHandle< int3 > h_image(m_image, access_location::host, access_mode::overwrite);
    ArrayHandle< Scalar > h_charge(m_charge, access_location::host, access_mode::overwrite);
    ArrayHandle< Scalar > h_diameter(m_diameter, access_location::host, access_mode::overwrite);
    ArrayHandle< unsigned int > h_body(m_body, access_location::host, access_mode::overwrite);
    ArrayHandle< unsigned int > h_tag(m_tag, access_location::host, access_mode::overwrite);
    ArrayHandle< unsigned int > h_rtag(m_rtag, access_location::host, access_mode::overwrite);

    // make sure the snapshot has the right size
    if (snapshot.size != m_nparticles)
        {
        m_exec_conf->msg->error() << "Snapshot size (" << snapshot.size << " particles) not equal"
             << endl << "          not equal number of particles in system." << endl;
        throw runtime_error("Error initializing ParticleData");
        }

    for (unsigned int tag = 0; tag < m_nparticles; tag++)
        {
        // particle index in sorted order
        unsigned int idx = snapshot.rtag[tag];

        h_pos.data[idx].x = snapshot.pos[tag].x;
        h_pos.data[idx].y = snapshot.pos[tag].y;
        h_pos.data[idx].z = snapshot.pos[tag].z;
        h_pos.data[idx].w = __int_as_scalar(snapshot.type[tag]);

        h_vel.data[idx].x = snapshot.vel[tag].x;
        h_vel.data[idx].y = snapshot.vel[tag].y;
        h_vel.data[idx].z = snapshot.vel[tag].z;
        h_vel.data[idx].w = snapshot.mass[tag];

        h_accel.data[idx] = snapshot.accel[tag];
        
        h_charge.data[idx] = snapshot.charge[tag];

        h_diameter.data[idx] = snapshot.diameter[tag];

        h_image.data[idx] = snapshot.image[tag];

        h_tag.data[idx] = tag;
        h_rtag.data[idx] = idx;

        h_body.data[idx] = snapshot.body[tag];
        }
    }

//! take a particle data snapshot
/* \param snapshot the snapshot to write to
 * \pre snapshot has to be allocated with a number of elements equal or greater than the number of particles)
*/
void ParticleData::takeSnapshot(SnapshotParticleData &snapshot)
    {
    assert(snapshot.size >= getN());

    ArrayHandle< Scalar4 > h_pos(m_pos, access_location::host, access_mode::read);
    ArrayHandle< Scalar4 > h_vel(m_vel, access_location::host, access_mode::read);
    ArrayHandle< Scalar3 > h_accel(m_accel, access_location::host, access_mode::read);
    ArrayHandle< int3 > h_image(m_image, access_location::host, access_mode::read);
    ArrayHandle< Scalar > h_charge(m_charge, access_location::host, access_mode::read);
    ArrayHandle< Scalar > h_diameter(m_diameter, access_location::host, access_mode::read);
    ArrayHandle< unsigned int > h_body(m_body, access_location::host, access_mode::read);
    ArrayHandle< unsigned int > h_tag(m_tag, access_location::host, access_mode::read);
    ArrayHandle< unsigned int > h_rtag(m_rtag, access_location::host, access_mode::read);

    for (unsigned int idx = 0; idx < m_nparticles; idx++)
        {
        unsigned int tag = h_tag.data[idx];

        snapshot.pos[tag] = make_scalar3(h_pos.data[idx].x, h_pos.data[idx].y, h_pos.data[idx].z);
        snapshot.vel[tag] = make_scalar3(h_vel.data[idx].x, h_vel.data[idx].y, h_vel.data[idx].z);
        snapshot.accel[tag] = h_accel.data[idx];
        snapshot.type[tag] = __scalar_as_int(h_pos.data[idx].w);
        snapshot.mass[tag] = h_vel.data[idx].w;
        snapshot.charge[tag] = h_charge.data[idx];
        snapshot.diameter[tag] = h_diameter.data[idx];
        snapshot.image[tag] = h_image.data[idx];
        snapshot.rtag[tag] = idx;
        snapshot.body[tag] = h_body.data[idx];
        }

    }


//! Helper for python __str__ for BoxDim
/*! Formats the box dim into a nice string
    \param box Box to format
*/
string print_boxdim(BoxDim *box)
    {
    assert(box);
    // turn the box dim into a nicely formatted string
    ostringstream s;
    s << "x: (" << box->xlo << "," << box->xhi << ") / y: (" << box->ylo << "," << box->yhi << ") / z: ("
    << box->zlo << "," << box->zhi << ")";
    return s.str();
    }

void export_BoxDim()
    {
    class_<BoxDim>("BoxDim")
    .def(init<Scalar>())
    .def(init<Scalar, Scalar, Scalar>())
    .def_readwrite("xlo", &BoxDim::xlo)
    .def_readwrite("xhi", &BoxDim::xhi)
    .def_readwrite("ylo", &BoxDim::ylo)
    .def_readwrite("yhi", &BoxDim::yhi)
    .def_readwrite("zlo", &BoxDim::zlo)
    .def_readwrite("zhi", &BoxDim::zhi)
    .def("__str__", &print_boxdim)
    ;
    }

//! Wrapper class needed for exposing virtual functions to python
class ParticleDataInitializerWrap : public ParticleDataInitializer, public wrapper<ParticleDataInitializer>
    {
    public:
        //! Calls the overidden ParticleDataInitializer::getNumParticles()
        unsigned int getNumParticles() const
            {
            return this->get_override("getNumParticles")();
            }
            
        //! Calls the overidden ParticleDataInitializer::getNumParticleTypes()
        unsigned int getNumParticleTypes() const
            {
            return this->get_override("getNumParticleTypes")();
            }
            
        //! Calls the overidden ParticleDataInitializer::getBox()
        BoxDim getBox() const
            {
            return this->get_override("getBox")();
            }
            
        //! Calls the overidden ParticleDataInitializer::initSnapshot()
        void initSnapshot(SnapshotParticleData& snapshot) const
            {
            this->get_override("initSnapshot")(snapshot);
            }
            
        //! Calls the overidden ParticleDataInitializer::getTypeMapping()
        std::vector<std::string> getTypeMapping() const
            {
            return this->get_override("getTypeMapping")();
            }
    };


void export_ParticleDataInitializer()
    {
    class_<ParticleDataInitializerWrap, boost::noncopyable>("ParticleDataInitializer")
    .def("getNumParticles", pure_virtual(&ParticleDataInitializer::getNumParticles))
    .def("getNumParticleTypes", pure_virtual(&ParticleDataInitializer::getNumParticleTypes))
    .def("getBox", pure_virtual(&ParticleDataInitializer::getBox))
    .def("initSnapshot", pure_virtual(&ParticleDataInitializer::initSnapshot))
    ;
    }


//! Helper for python __str__ for ParticleData
/*! Gives a synopsis of a ParticleData in a string
    \param pdata Particle data to format parameters from
*/
string print_ParticleData(ParticleData *pdata)
    {
    assert(pdata);
    ostringstream s;
    s << "ParticleData: " << pdata->getN() << " particles";
    return s.str();
    }

void export_ParticleData()
    {
    class_<ParticleData, boost::shared_ptr<ParticleData>, boost::noncopyable>("ParticleData", init<unsigned int, const BoxDim&, unsigned int, boost::shared_ptr<ExecutionConfiguration> >())
    .def(init<const ParticleDataInitializer&, boost::shared_ptr<ExecutionConfiguration> >())
    .def("getBox", &ParticleData::getBox, return_value_policy<copy_const_reference>())
    .def("setBox", &ParticleData::setBox)
    .def("getN", &ParticleData::getN)
    .def("getNTypes", &ParticleData::getNTypes)
    .def("getMaximumDiameter", &ParticleData::getMaxDiameter)
    .def("getNameByType", &ParticleData::getNameByType)
    .def("getTypeByName", &ParticleData::getTypeByName)
    .def("setProfiler", &ParticleData::setProfiler)
    .def("getExecConf", &ParticleData::getExecConf)
    .def("__str__", &print_ParticleData)
    .def("getPosition", &ParticleData::getPosition)
    .def("getVelocity", &ParticleData::getVelocity)
    .def("getAcceleration", &ParticleData::getAcceleration)
    .def("getImage", &ParticleData::getImage)
    .def("getCharge", &ParticleData::getCharge)
    .def("getMass", &ParticleData::getMass)
    .def("getDiameter", &ParticleData::getDiameter)
    .def("getBody", &ParticleData::getBody)
    .def("getType", &ParticleData::getType)
    .def("getOrientation", &ParticleData::getOrientation)
    .def("getPNetForce", &ParticleData::getPNetForce)
    .def("getInertiaTensor", &ParticleData::getInertiaTensor, return_value_policy<copy_const_reference>())
    .def("setPosition", &ParticleData::setPosition)
    .def("setVelocity", &ParticleData::setVelocity)
    .def("setImage", &ParticleData::setImage)
    .def("setCharge", &ParticleData::setCharge)
    .def("setMass", &ParticleData::setMass)
    .def("setDiameter", &ParticleData::setDiameter)
    .def("setBody", &ParticleData::setBody)
    .def("setType", &ParticleData::setType)
    .def("setOrientation", &ParticleData::setOrientation)
    .def("setInertiaTensor", &ParticleData::setInertiaTensor)
    ;
    }

void export_SnapshotParticleData()
    {
    class_<SnapshotParticleData, boost::shared_ptr<SnapshotParticleData>, boost::noncopyable>("SnapshotParticleData", init<unsigned int>())
    .def_readwrite("pos", &SnapshotParticleData::pos)
    .def_readwrite("vel", &SnapshotParticleData::vel)
    .def_readwrite("accel", &SnapshotParticleData::accel)
    .def_readwrite("type", &SnapshotParticleData::type)
    .def_readwrite("mass", &SnapshotParticleData::mass)
    .def_readwrite("charge", &SnapshotParticleData::charge)
    .def_readwrite("diameter", &SnapshotParticleData::diameter)
    .def_readwrite("image", &SnapshotParticleData::image)
    .def_readwrite("rtag", &SnapshotParticleData::rtag)
    .def_readwrite("body", &SnapshotParticleData::body)
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

