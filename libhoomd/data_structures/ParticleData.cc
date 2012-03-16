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

#ifdef ENABLE_MPI
#include "Communicator.h"

#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
using namespace boost::mpi;
#endif

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

    \post \c pos,\c vel,\c accel are allocated and initialized to 0.0
    \post \c charge is allocated and initialized to a value of 0.0
    \post \c diameter is allocated and initialized to a value of 1.0
    \post \c mass is allocated and initialized to a value of 1.0
    \post \c image is allocated and initialized to values of 0.0
    \post \c global_tag is allocated and given the default initialization global_tag[i] = i
    \post \c the reverse lookup map global_rtag is initialized with the identity mapping
    \post \c type is allocated and given the default value of type[i] = 0
    \post \c body is allocated and given the devault value of type[i] = NO_BODY
    \post Arrays are not currently acquired

    Type mappings assign particle types "A", "B", "C", ....
*/
ParticleData::ParticleData(unsigned int N, const BoxDim &box, unsigned int n_types, boost::shared_ptr<ExecutionConfiguration> exec_conf)
        : m_box(box), m_exec_conf(exec_conf), m_ntypes(n_types), m_nghosts(0), m_nglobal(0), m_resize_factor(9./8.)
    {
    // check the input for errors
    if (m_ntypes == 0)
        {
        cerr << endl << "***Error! Number of particle types must be greater than 0." << endl << endl;
        throw std::runtime_error("Error initializing ParticleData");
        }
        
    // allocate memory
    // we are allocating for the number of global particles equal to the number of local particles,
    // since this constructor is only called for initializing a single-processor simulation
    allocate(N);

    // default: number of global particles = number of local particles
    setNGlobal(getN());

    ArrayHandle< Scalar4 > h_vel(getVelocities(), access_location::host, access_mode::overwrite);
    ArrayHandle< Scalar > h_diameter(getDiameters(), access_location::host, access_mode::overwrite);
    ArrayHandle< unsigned int > h_tag(getTags(), access_location::host, access_mode::overwrite);
    ArrayHandle< unsigned int > h_rtag(getRTags(), access_location::host, access_mode::overwrite);
    ArrayHandle< unsigned int > h_global_tag(getGlobalTags(), access_location::host, access_mode::overwrite);
    ArrayHandle< unsigned int > h_global_rtag(getGlobalRTags(), access_location::host, access_mode::overwrite);
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

        h_global_tag.data[i] = i;
        h_global_rtag.data[i] = i;
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
        m_gpu_box.xlo = m_box.xlo;
        m_gpu_box.ylo = m_box.ylo;
        m_gpu_box.zlo = m_box.zlo;
        m_gpu_box.xhi = m_box.xhi;
        m_gpu_box.yhi = m_box.yhi;
        m_gpu_box.zhi = m_box.zhi;
        }
#endif
    
    // initially, global box = local box
    setGlobalBox(m_box);
    }

/*! Calls the initializer's members to determine the number of particles, box size and then
    uses it to fill out the position and velocity data.
    \param init Initializer to use
    \param exec_conf Execution configuration to run on
*/
ParticleData::ParticleData(const ParticleDataInitializer& init, boost::shared_ptr<ExecutionConfiguration> exec_conf) : m_exec_conf(exec_conf), m_ntypes(0), m_nghosts(0), m_nglobal(0), m_resize_factor(9./8.)
    {
    // allocate memory
    allocate(init.getNumParticles());

    // default: number of global particles = number of local particles
    setNGlobal(getN());
    
        {
        ArrayHandle< Scalar4 > h_orientation(m_orientation, access_location::host, access_mode::readwrite);
        
        ArrayHandle< Scalar4 > h_vel(getVelocities(), access_location::host, access_mode::overwrite);
        ArrayHandle< Scalar > h_diameter(getDiameters(), access_location::host, access_mode::overwrite);
        ArrayHandle< unsigned int > h_tag(getTags(), access_location::host, access_mode::overwrite);
        ArrayHandle< unsigned int > h_rtag(getRTags(), access_location::host, access_mode::overwrite);
        ArrayHandle< unsigned int > h_global_tag(getGlobalTags(), access_location::host, access_mode::overwrite);
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
            h_global_tag.data[i] = i;
            h_orientation.data[i] = make_scalar4(1.0, 0.0, 0.0, 0.0);
            }
        }

    SnapshotParticleData snapshot(getN());

    // initialize the snapshot with default values
    takeSnapshot(snapshot);

    // pass snapshot to initializer
    init.initSnapshot(snapshot);

    // initialize particle data with updated values
    initializeFromSnapshot(snapshot);

    setBox(init.getBox());

    // initially, global simulation box = local simulation box
    setGlobalBox(init.getBox());

        {
        ArrayHandle<Scalar4> h_orientation(getOrientationArray(), access_location::host, access_mode::overwrite);
        init.initOrientation(h_orientation.data);
        init.initMomentInertia(&m_inertia_tensor[0]);
        }
            
    // it is an error for particles to be initialized outside of their box
    if (!inBox())
        {
        cerr << endl << "***Error! Not all particles were found inside the given box" << endl << endl;
        throw runtime_error("Error initializing ParticleData");
        }
        
    // default constructed shared ptr is null as desired
    m_prof = boost::shared_ptr<Profiler>();
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
    m_gpu_box.xlo = m_box.xlo;
    m_gpu_box.ylo = m_box.ylo;
    m_gpu_box.zlo = m_box.zlo;
    m_gpu_box.xhi = m_box.xhi;
    m_gpu_box.yhi = m_box.yhi;
    m_gpu_box.zhi = m_box.zhi;
#endif
    
    m_boxchange_signal();
    }

/*! \return Global simulation box dimensions
 */
const BoxDim & ParticleData::getGlobalBox() const
    {
    return m_global_box;
    }

/*! \param box New global box to set
    \note ParticleData does NOT enforce any boundary conditions. When a new box is set,
        it is the responsibility of the caller to ensure that all particles lie within
        the new box.
*/
void ParticleData::setGlobalBox(const BoxDim &global_box)
    {
    m_global_box = global_box;
    assert(inBox());
#ifdef ENABLE_CUDA
    // setup the box
    m_gpu_global_box.Lx = m_global_box.xhi - m_global_box.xlo;
    m_gpu_global_box.Ly = m_global_box.yhi - m_global_box.ylo;
    m_gpu_global_box.Lz = m_global_box.zhi - m_global_box.zlo;
    m_gpu_global_box.Lxinv = 1.0f / m_gpu_global_box.Lx;
    m_gpu_global_box.Lyinv = 1.0f / m_gpu_global_box.Ly;
    m_gpu_global_box.Lzinv = 1.0f / m_gpu_global_box.Lz;
    m_gpu_global_box.xlo = m_global_box.xlo;
    m_gpu_global_box.ylo = m_global_box.ylo;
    m_gpu_global_box.zlo = m_global_box.zlo;
    m_gpu_global_box.xhi = m_global_box.xhi;
    m_gpu_global_box.yhi = m_global_box.yhi;
    m_gpu_global_box.zhi = m_global_box.zhi;
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

/*! \param func Function to be called when the particle data arrays are resized
    \return Connection to manage the signal

    The maximum particle number is the size of the particle data arrays in memory. This
    can be larger than the current local particle number. The arrays are infrequently
    resized (e.g. by doubling the size if necessary), to keep the amount of data
    copied to a minimum.

    \note If the caller class is destroyed, it needs to disconnect the signal connection
    via \b con.disconnect where \b con is the return value of this function.

    \note A change in maximum particle number does not necessarily imply a change in sort order,
          and no extra notifyParticleSort() is called.
*/
boost::signals::connection ParticleData::connectMaxParticleNumberChange(const boost::function<void ()> &func)
    {
    return m_max_particle_num_signal.connect(func);
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
        
    cerr << endl << "***Error! Type " << name << " not found!" << endl;
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
        cerr << endl << "***Error! Requesting type name for non-existant type " << type << endl << endl;
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
        cerr << endl << "***Error! ParticleData is being asked to allocate 0 particles.... this makes no sense whatsoever" << endl << endl;
        throw runtime_error("Error allocating ParticleData");
        }

    // set particle number
    m_nparticles = N;

    // maximum number is the current particle number
    m_max_nparticles = N;

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

    // global tag
    GPUArray< unsigned int> global_tag(getN(), m_exec_conf);
    m_global_tag.swap(global_tag);

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

//! Set global number of particles
/*! \param nglobal Global number of particles
 */
void ParticleData::setNGlobal(unsigned int nglobal)
    {
    if (m_nparticles > nglobal)
        {
        cerr << endl << "***Error! ParticleData is being asked to allocate memory for a global number of particles smaller"
             << endl << "          than the local number of particles. This does not make any sense.";
        throw runtime_error("Error initializing ParticleData");
        }
    if (m_nglobal)
        {
        if (nglobal > m_nglobal)
            {
            // resize array of global reverse lookup tags
            m_global_rtag.resize(nglobal);
            }
        }
    else
        {
        // allocate array
        GPUArray< unsigned int> global_rtag(nglobal, m_exec_conf);
        m_global_rtag.swap(global_rtag);
       }

    // Set global particle number
    m_nglobal = nglobal;

    }

/*! \param max_n new maximum size of particle data arrays (can be greater or smaller than the current maxium size)
 *  To inform classes that allocate arrays for per-particle information of the change of the particle data size,
 *  this method issues a m_max_particle_num_signal().
 *
 *  \note To keep unnecessary data copying to a minimum, arrays are not reallocated with every change of the
 *  particle number, rather an amortized array expanding strategy is used.
 */
void ParticleData::reallocate(unsigned int max_n)
    {

    m_max_nparticles = max_n;

    m_pos.resize(max_n);
    m_vel.resize(max_n);
    m_accel.resize(max_n);
    m_charge.resize(max_n);
    m_diameter.resize(max_n);
    m_image.resize(max_n);
    m_tag.resize(max_n);
    m_rtag.resize(max_n);
    m_global_tag.resize(max_n);
    m_body.resize(max_n);

    m_net_force.resize(max_n);
    m_net_virial.resize(max_n,6);
    m_net_torque.resize(max_n);
    m_orientation.resize(max_n);
    m_inertia_tensor.resize(max_n);

    // notify observers
    m_max_particle_num_signal();
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
            cout << "pos " << i << ":" << setprecision(12) << h_pos.data[i].x << " " << h_pos.data[i].y << " " << h_pos.data[i].z << endl;
            cout << "lo: " << m_box.xlo << " " << m_box.ylo << " " << m_box.zlo << endl;
            cout << "hi: " << m_box.xhi << " " << m_box.yhi << " " << m_box.zhi << endl;
            return false;
            }
        if (h_pos.data[i].y < m_box.ylo-Scalar(1e-5) || h_pos.data[i].y > m_box.yhi+Scalar(1e-5))
            {
            cout << "pos " << i << ":" << setprecision(12) << h_pos.data[i].x << " " << h_pos.data[i].y << " " << h_pos.data[i].z << endl;
            cout << "lo: " << m_box.xlo << " " << m_box.ylo << " " << m_box.zlo << endl;
            cout << "hi: " << m_box.xhi << " " << m_box.yhi << " " << m_box.zhi << endl;
            return false;
            }
        if (h_pos.data[i].z < m_box.zlo-Scalar(1e-5) || h_pos.data[i].z > m_box.zhi+Scalar(1e-5))
            {
            cout << "pos " << i << ":" << setprecision(12) << h_pos.data[i].x << " " << h_pos.data[i].y << " " << h_pos.data[i].z << endl;
            cout << "lo: " << m_box.xlo << " " << m_box.ylo << " " << m_box.zlo << endl;
            cout << "hi: " << m_box.xhi << " " << m_box.yhi << " " << m_box.zhi << endl;
            return false;
            }
        }
    return true;
    }

//! Initialize from a snapshot
//! \param snapshot the initial particle data
//! \post the particle data arrays are initialized from the snapshot, in index order
void ParticleData::initializeFromSnapshot(const SnapshotParticleData& snapshot)
    {
    m_nparticles = snapshot.size;

    // reallocate particle data such that we can accomodate the particles
    reallocate(snapshot.size);

    if (getNGlobal() < snapshot.size)
        {
        cerr << endl << "***Error! Global number of particles must be greater or equal the number "
             << " of local particles." << endl << endl;
        throw std::runtime_error("Error initializing ParticleData");
        }

    ArrayHandle< Scalar4 > h_pos(m_pos, access_location::host, access_mode::overwrite);
    ArrayHandle< Scalar4 > h_vel(m_vel, access_location::host, access_mode::overwrite);
    ArrayHandle< Scalar3 > h_accel(m_accel, access_location::host, access_mode::overwrite);
    ArrayHandle< int3 > h_image(m_image, access_location::host, access_mode::overwrite);
    ArrayHandle< Scalar > h_charge(m_charge, access_location::host, access_mode::overwrite);
    ArrayHandle< Scalar > h_diameter(m_diameter, access_location::host, access_mode::overwrite);
    ArrayHandle< unsigned int > h_body(m_body, access_location::host, access_mode::overwrite);
    ArrayHandle< unsigned int > h_tag(m_tag, access_location::host, access_mode::overwrite);
    ArrayHandle< unsigned int > h_rtag(m_rtag, access_location::host, access_mode::overwrite);
    ArrayHandle< unsigned int > h_global_tag(m_global_tag, access_location::host, access_mode::overwrite);
    ArrayHandle< unsigned int > h_global_rtag(m_global_rtag, access_location::host, access_mode::readwrite);

    // first reset all reverse look-up tags
    for (unsigned int idx = 0; idx < m_nparticles; idx++)
        {
        h_pos.data[idx].x = snapshot.pos[idx].x;
        h_pos.data[idx].y = snapshot.pos[idx].y;
        h_pos.data[idx].z = snapshot.pos[idx].z;
        h_pos.data[idx].w = __int_as_scalar(snapshot.type[idx]);

        h_vel.data[idx].x = snapshot.vel[idx].x;
        h_vel.data[idx].y = snapshot.vel[idx].y;
        h_vel.data[idx].z = snapshot.vel[idx].z;
        h_vel.data[idx].w = snapshot.mass[idx];

        h_accel.data[idx] = snapshot.accel[idx];

        h_charge.data[idx] = snapshot.charge[idx];

        h_diameter.data[idx] = snapshot.diameter[idx];

        h_image.data[idx] = snapshot.image[idx];

        h_global_tag.data[idx] = snapshot.global_tag[idx];
        h_global_rtag.data[snapshot.global_tag[idx]] = idx;

        h_body.data[idx] = snapshot.body[idx];
        }

    // initialize number of particle types
    m_ntypes = snapshot.num_particle_types;

    if (m_ntypes == 0)
        {
        cerr << endl << "***Error! Number of particle types must be greater than 0." << endl << endl;
        throw std::runtime_error("Error initializing ParticleData");
        }

    // initialize type mapping
    m_type_mapping = snapshot.type_mapping;
    assert(m_type_mapping.size() == m_ntypes);
    }

//! take a particle data snapshot
/* \param snapshot the snapshot to write to
 * \pre snapshot has to be allocated with a number of elements equal or greater than the number of particles)
*/
void ParticleData::takeSnapshot(SnapshotParticleData &snapshot)
    {
    assert(snapshot.size >= getN());
    assert(inBox());

    ArrayHandle< Scalar4 > h_pos(m_pos, access_location::host, access_mode::read);
    ArrayHandle< Scalar4 > h_vel(m_vel, access_location::host, access_mode::read);
    ArrayHandle< Scalar3 > h_accel(m_accel, access_location::host, access_mode::read);
    ArrayHandle< int3 > h_image(m_image, access_location::host, access_mode::read);
    ArrayHandle< Scalar > h_charge(m_charge, access_location::host, access_mode::read);
    ArrayHandle< Scalar > h_diameter(m_diameter, access_location::host, access_mode::read);
    ArrayHandle< unsigned int > h_body(m_body, access_location::host, access_mode::read);
    ArrayHandle< unsigned int > h_global_tag(m_global_tag, access_location::host, access_mode::read);

    for (unsigned int idx = 0; idx < m_nparticles; idx++)
        {
        snapshot.pos[idx] = make_scalar3(h_pos.data[idx].x, h_pos.data[idx].y, h_pos.data[idx].z);
        snapshot.vel[idx] = make_scalar3(h_vel.data[idx].x, h_vel.data[idx].y, h_vel.data[idx].z);
        snapshot.accel[idx] = h_accel.data[idx];
        snapshot.type[idx] = __scalar_as_int(h_pos.data[idx].w);
        snapshot.mass[idx] = h_vel.data[idx].w;
        snapshot.charge[idx] = h_charge.data[idx];
        snapshot.diameter[idx] = h_diameter.data[idx];
        snapshot.image[idx] = h_image.data[idx];
        snapshot.global_tag[idx] = h_global_tag.data[idx];
        snapshot.body[idx] = h_body.data[idx];

        // insert reverse lookup global tag -> idx
        snapshot.global_rtag.insert(std::pair<unsigned int, unsigned int>(h_global_tag.data[idx], idx));
        }

    snapshot.num_particle_types = m_ntypes;
    snapshot.type_mapping = m_type_mapping;
    }

//! Remove particles from the local particle data
/*! \param n number of particles to remove
 *
 * This method just decreases the number of particles in the system. The caller
 * has to make sure that the change is reflected in the particle data arrays (i.e. compacting
 * the particle data).
 */
void ParticleData::removeParticles(const unsigned int n)
    {
    assert(n <= getN());
    m_nparticles -= n;
    }

//! Add a number of particles to the local particle data
/*! This function uses amortized array resizing of the particle data structures in the system
    to accomodate the new partices.

    The arrays are resized to the (rounded integer value) of the closes power of m_resize_factor

    \param n number of particles to add
    \post The maximum size of the particle data arrays (accessible via getMaxN()) is
         increased to hold the new particles.
*/
void ParticleData::addParticles(const unsigned int n)
    {
    unsigned int max_nparticles = m_max_nparticles;
    if (m_nparticles + n > max_nparticles)
        {
        while (m_nparticles + n > max_nparticles)
            max_nparticles = (unsigned int) (((float) max_nparticles) * m_resize_factor) + 1 ;

        // actually reallocate particle data arrays
        reallocate(max_nparticles);
        }

    m_nparticles += n;
    }

//! Add ghost particles at the end of the local particle data
/*! Ghost ptls are appended at the end of the particle data.
  Ghost particles have only incomplete particle information (position, charge, diameter) and
  don't need tags.

  \param nghosts number of ghost particles to add
  \post the particle data arrays are resized if necessary to accomodate the ghost particles,
        the number of ghost particles is updated
*/
void ParticleData::addGhostParticles(const unsigned int nghosts)
    {
    assert(nghosts >= 0);

    unsigned int max_nparticles = m_max_nparticles;

    m_nghosts += nghosts;

    if (m_nparticles + m_nghosts > max_nparticles)
        {
        while (m_nparticles + m_nghosts > max_nparticles)
            max_nparticles = (unsigned int) (((float) max_nparticles) * m_resize_factor) + 1 ;

        // reallocate particle data arrays
        reallocate(max_nparticles);
        }

    }

#ifdef ENABLE_MPI
//! Find the processor that owns a particle
/*! \param tag Tag of the particle to search
 * \param is_local True if the particle is local
 */
unsigned int ParticleData::getOwnerRank(unsigned int tag) const
    {
    assert(m_mpi_comm);
    bool is_local = (getGlobalRTag(tag) < getN());
    int n_found;

    // First check that the particle is on exactly one processor
    all_reduce(*m_mpi_comm, is_local ? 1 : 0, n_found, std::plus<int>());

    if (n_found == 0)
        {
        cerr << endl << "***Error! Could not find particle " << tag << " on any processor." << endl << endl;
        throw std::runtime_error("Error accessing particle data.");
        }
    else if (n_found > 1)
       {
        cerr << endl << "***Error! Found particle " << tag << " on multiple processors." << endl << endl;
        throw std::runtime_error("Error accessing particle data.");
       }

    // Now find the processor that owns it
    int owner_rank;
    all_reduce(*m_mpi_comm, is_local ? m_mpi_comm->rank() : -1, owner_rank, boost::mpi::maximum<int>());
    assert (owner_rank >= 0);
    assert (owner_rank < m_mpi_comm->size());

    return (unsigned int) owner_rank;
    }
#endif

///////////////////////////////////////////////////////////
// get accessors

//! Get the current position of a particle
Scalar3 ParticleData::getPosition(unsigned int global_tag) const
    {
    unsigned int idx = getGlobalRTag(global_tag);
    bool found = (idx < getN());
    Scalar3 result;
    if (found)
        {
        ArrayHandle< Scalar4 > h_pos(m_pos, access_location::host, access_mode::read);
        result = make_scalar3(h_pos.data[idx].x, h_pos.data[idx].y, h_pos.data[idx].z);
        }
#ifdef ENABLE_MPI
    if (m_mpi_comm)
        {
        unsigned int owner_rank = getOwnerRank(global_tag);
        broadcast(*m_mpi_comm, result, owner_rank);
        found = true;
        }
#endif
    assert(found);
    return result;
    }

//! Get the current velocity of a particle
Scalar3 ParticleData::getVelocity(unsigned int global_tag) const
    {
    unsigned int idx = getGlobalRTag(global_tag);
    bool found = (idx < getN());
    Scalar3 result;
    if (found)
        {
        ArrayHandle< Scalar4 > h_vel(m_vel, access_location::host, access_mode::read);
        result = make_scalar3(h_vel.data[idx].x, h_vel.data[idx].y, h_vel.data[idx].z);
        }
#ifdef ENABLE_MPI
    if (m_mpi_comm)
        {
        unsigned int owner_rank = getOwnerRank(global_tag);
        broadcast(*m_mpi_comm, result, owner_rank);
        found = true;
        }
#endif
    assert(found);
    return result;
    }

//! Get the current acceleration of a particle
Scalar3 ParticleData::getAcceleration(unsigned int global_tag) const
    {
    unsigned int idx = getGlobalRTag(global_tag);
    bool found = (idx < getN());
    Scalar3 result;
    if (found)
        {
        ArrayHandle< Scalar3 > h_accel(m_accel, access_location::host, access_mode::read);
        result = make_scalar3(h_accel.data[idx].x, h_accel.data[idx].y, h_accel.data[idx].z);
        }
#ifdef ENABLE_MPI
    if (m_mpi_comm)
        {
        unsigned int owner_rank = getOwnerRank(global_tag);
        broadcast(*m_mpi_comm, result, owner_rank);
        found = true;
        }
#endif
    assert(found);
    return result;
    }

//! Get the current image flags of a particle
int3 ParticleData::getImage(unsigned int global_tag) const
    {
    unsigned int idx = getGlobalRTag(global_tag);
    bool found = (idx < getN());
    int3 result;
    if (found)
        {
        ArrayHandle< int3 > h_image(m_image, access_location::host, access_mode::read);
        result = make_int3(h_image.data[idx].x, h_image.data[idx].y, h_image.data[idx].z);
        }
#ifdef ENABLE_MPI
    if (m_mpi_comm)
        {
        unsigned int owner_rank = getOwnerRank(global_tag);
        broadcast(*m_mpi_comm, result, owner_rank);
        found = true;
        }
#endif
    assert(found);
    return result;
    }

//! Get the current charge of a particle
Scalar ParticleData::getCharge(unsigned int global_tag) const
    {
    unsigned int idx = getGlobalRTag(global_tag);
    bool found = (idx < getN());
    Scalar result;
    if (found)
        {
        ArrayHandle< Scalar > h_charge(m_charge, access_location::host, access_mode::read);
        result = h_charge.data[idx];
        }
#ifdef ENABLE_MPI
    if (m_mpi_comm)
        {
        unsigned int owner_rank = getOwnerRank(global_tag);
        broadcast(*m_mpi_comm, result, owner_rank);
        found = true;
        }
#endif
    assert(found);
    return result;
    }

//! Get the current mass of a particle
Scalar ParticleData::getMass(unsigned int global_tag) const
    {
    unsigned int idx = getGlobalRTag(global_tag);
    bool found = (idx < getN());
    Scalar result;
    if (found)
        {
        ArrayHandle< Scalar4 > h_vel(m_vel, access_location::host, access_mode::read);
        result = h_vel.data[idx].w;
        }
#ifdef ENABLE_MPI
    if (m_mpi_comm)
        {
        unsigned int owner_rank = getOwnerRank(global_tag);
        broadcast(*m_mpi_comm, result, owner_rank);
        found = true;
        }
#endif
    assert(found);
    return result;
    }

//! Get the current diameter of a particle
Scalar ParticleData::getDiameter(unsigned int global_tag) const
    {
    unsigned int idx = getGlobalRTag(global_tag);
    bool found = (idx < getN());
    Scalar result;
    if (found)
        {
        ArrayHandle< Scalar > h_diameter(m_diameter, access_location::host, access_mode::read);
        result = h_diameter.data[idx];
        }
#ifdef ENABLE_MPI
    if (m_mpi_comm)
        {
        unsigned int owner_rank = getOwnerRank(global_tag);
        broadcast(*m_mpi_comm, result, owner_rank);
        found = true;
        }
#endif
    assert(found);
    return result;
    }

//! Get the body id of a particle
unsigned int ParticleData::getBody(unsigned int global_tag) const
    {
    unsigned int idx = getGlobalRTag(global_tag);
    bool found = (idx < getN());
    unsigned int result;
    if (found)
        {
        ArrayHandle< unsigned int > h_body(m_body, access_location::host, access_mode::read);
        result = h_body.data[idx];
        }
#ifdef ENABLE_MPI
    if (m_mpi_comm)
        {
        unsigned int owner_rank = getOwnerRank(global_tag);
        broadcast(*m_mpi_comm, result, owner_rank);
        found = true;
        }
#endif
    assert(found);
    return result;
    }

//! Get the current type of a particle
unsigned int ParticleData::getType(unsigned int global_tag) const
    {
    unsigned int idx = getGlobalRTag(global_tag);
    bool found = (idx < getN());
    unsigned int result;
    if (found)
        {
        ArrayHandle< Scalar4 > h_pos(m_pos, access_location::host, access_mode::read);
        result = __scalar_as_int(h_pos.data[idx].w);
        }
#ifdef ENABLE_MPI
    if (m_mpi_comm)
        {
        unsigned int owner_rank = getOwnerRank(global_tag);
        broadcast(*m_mpi_comm, result, owner_rank);
        found = true;
        }
#endif
    assert(found);
    return result;
    }

//! Get the orientation of a particle with a given tag
Scalar4 ParticleData::getOrientation(unsigned int global_tag) const
    {
    unsigned int idx = getGlobalRTag(global_tag);
    bool found = (idx < getN());
    Scalar4 result;
    if (found)
        {
        ArrayHandle< Scalar4 > h_orientation(m_orientation, access_location::host, access_mode::read);
        result = h_orientation.data[idx];
        }
#ifdef ENABLE_MPI
    if (m_mpi_comm)
        {
        unsigned int owner_rank = getOwnerRank(global_tag);
        broadcast(*m_mpi_comm, result, owner_rank);
        found = true;
        }
#endif
    assert(found);
    return result;
    }

//! Get the net force / energy on a given particle
Scalar4 ParticleData::getPNetForce(unsigned int global_tag) const
    {
    unsigned int idx = getGlobalRTag(global_tag);
    bool found = (idx < getN());
    Scalar4 result;
    if (found)
        {
        ArrayHandle< Scalar4 > h_net_force(m_net_force, access_location::host, access_mode::read);
        result = h_net_force.data[idx];
        }
#ifdef ENABLE_MPI
    if (m_mpi_comm)
        {
        unsigned int owner_rank = getOwnerRank(global_tag);
        broadcast(*m_mpi_comm, result, owner_rank);
        found = true;
        }
#endif
    assert(found);
    return result;

    }

//! Set the current position of a particle
void ParticleData::setPosition(unsigned int global_tag, const Scalar3& pos)
    {
    unsigned int idx = getGlobalRTag(global_tag);
    bool found = (idx < getN());

#ifdef ENABLE_MPI
    // make sure the particle is somewhere
    if (m_mpi_comm)
        getOwnerRank(global_tag);
#endif
    if (found)
        {
        ArrayHandle< Scalar4 > h_pos(m_pos, access_location::host, access_mode::readwrite);
        h_pos.data[idx].x = pos.x; h_pos.data[idx].y = pos.y; h_pos.data[idx].z = pos.z;
        }
    }

//! Set the current velocity of a particle
void ParticleData::setVelocity(unsigned int global_tag, const Scalar3& vel)
    {
    unsigned int idx = getGlobalRTag(global_tag);
    bool found = (idx < getN());

#ifdef ENABLE_MPI
    // make sure the particle is somewhere
    if (m_mpi_comm)
        getOwnerRank(global_tag);
#endif
    if (found)
        {
        ArrayHandle< Scalar4 > h_vel(m_vel, access_location::host, access_mode::readwrite);
        h_vel.data[idx].x = vel.x; h_vel.data[idx].y = vel.y; h_vel.data[idx].z = vel.z;
        }
    }

//! Set the current image flags of a particle
void ParticleData::setImage(unsigned int global_tag, const int3& image)
    {
    unsigned int idx = getGlobalRTag(global_tag);
    bool found = (idx < getN());

#ifdef ENABLE_MPI
    // make sure the particle is somewhere
    if (m_mpi_comm)
        getOwnerRank(global_tag);
#endif
    if (found)
        {
        ArrayHandle< int3 > h_image(m_image, access_location::host, access_mode::readwrite);
        h_image.data[idx].x = image.x; h_image.data[idx].y = image.y; h_image.data[idx].z = image.z;
        }
    }

//! Set the current charge of a particle
void ParticleData::setCharge(unsigned int global_tag, Scalar charge)
    {
    unsigned int idx = getGlobalRTag(global_tag);
    bool found = (idx < getN());

#ifdef ENABLE_MPI
    // make sure the particle is somewhere
    if (m_mpi_comm)
        getOwnerRank(global_tag);
#endif
    if (found)
        {
        ArrayHandle< Scalar > h_charge(m_charge, access_location::host, access_mode::readwrite);
        h_charge.data[idx] = charge;
        }
    }

//! Set the current mass of a particle
void ParticleData::setMass(unsigned int global_tag, Scalar mass)
    {
    unsigned int idx = getGlobalRTag(global_tag);
    bool found = (idx < getN());

#ifdef ENABLE_MPI
    // make sure the particle is somewhere
    if (m_mpi_comm)
        getOwnerRank(global_tag);
#endif
    if (found)
        {
        ArrayHandle< Scalar4 > h_vel(m_vel, access_location::host, access_mode::readwrite);
        h_vel.data[idx].w = mass;
        }
    }


//! Set the current diameter of a particle
void ParticleData::setDiameter(unsigned int global_tag, Scalar diameter)
    {
    unsigned int idx = getGlobalRTag(global_tag);
    bool found = (idx < getN());

#ifdef ENABLE_MPI
    // make sure the particle is somewhere
    if (m_mpi_comm)
        getOwnerRank(global_tag);
#endif
    if (found)
        {
        ArrayHandle< Scalar > h_diameter(m_diameter, access_location::host, access_mode::readwrite);
        h_diameter.data[idx] = diameter;
        }
    }

//! Set the body id of a particle
void ParticleData::setBody(unsigned int global_tag, int body)
    {
    unsigned int idx = getGlobalRTag(global_tag);
    bool found = (idx < getN());

#ifdef ENABLE_MPI
    // make sure the particle is somewhere
    if (m_mpi_comm)
        getOwnerRank(global_tag);
#endif
    if (found)
        {
        ArrayHandle< unsigned int > h_body(m_body, access_location::host, access_mode::readwrite);
        h_body.data[idx] = body;
        }
    }

//! Set the current type of a particle
void ParticleData::setType(unsigned int global_tag, unsigned int typ)
    {
    assert(typ < m_ntypes);
    unsigned int idx = getGlobalRTag(global_tag);
    bool found = (idx < getN());

#ifdef ENABLE_MPI
    // make sure the particle is somewhere
    if (m_mpi_comm)
        getOwnerRank(global_tag);
#endif
    if (found)
        {
        ArrayHandle< Scalar4 > h_pos(m_pos, access_location::host, access_mode::readwrite);
        h_pos.data[idx].w = __int_as_scalar(typ);
        }
    }

//! Set the orientation of a particle with a given tag
void ParticleData::setOrientation(unsigned int global_tag, const Scalar4& orientation)
    {
    unsigned int idx = getGlobalRTag(global_tag);
    bool found = (idx < getN());

#ifdef ENABLE_MPI
    // make sure the particle is somewhere
    if (m_mpi_comm)
        getOwnerRank(global_tag);
#endif
    if (found)
        {
        ArrayHandle< Scalar4 > h_orientation(m_orientation, access_location::host, access_mode::readwrite);
        h_orientation.data[idx] = orientation;
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
            
    };


void export_ParticleDataInitializer()
    {
    class_<ParticleDataInitializerWrap, boost::noncopyable>("ParticleDataInitializer")
    .def("getNumParticles", pure_virtual(&ParticleDataInitializer::getNumParticles))
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
    .def("setGlobalBox", &ParticleData::setGlobalBox)
    .def("getGlobalBox", &ParticleData::getGlobalBox, return_value_policy<copy_const_reference>())
    .def("getN", &ParticleData::getN)
    .def("getNGlobal", &ParticleData::getNGlobal)
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
    .def_readwrite("body", &SnapshotParticleData::body)
    .def_readwrite("global_tag", &SnapshotParticleData::global_tag)
    .def_readwrite("num_particle_types", &SnapshotParticleData::num_particle_types)
    .def_readwrite("type_mapping", &SnapshotParticleData::type_mapping)
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

