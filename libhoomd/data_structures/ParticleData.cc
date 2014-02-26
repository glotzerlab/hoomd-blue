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
    \brief Contains all code for ParticleData, and SnapshotParticleData.
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

#ifdef ENABLE_MPI
#include "HOOMDMPI.h"
#endif

#ifdef ENABLE_CUDA
#include "CachedAllocator.h"
#endif

#include <boost/bind.hpp>
#include <boost/iterator/zip_iterator.hpp>
#include <boost/iterator/transform_iterator.hpp>

using namespace boost::signals2;
using namespace boost;

////////////////////////////////////////////////////////////////////////////
// ParticleData members

/*! \param N Number of particles to allocate memory for
    \param n_types Number of particle types that will exist in the data arrays
    \param global_box Box the particles live in
    \param exec_conf ExecutionConfiguration to use when executing code on the GPU
    \param decomposition (optional) Domain decomposition layout

    \post \c pos,\c vel,\c accel are allocated and initialized to 0.0
    \post \c charge is allocated and initialized to a value of 0.0
    \post \c diameter is allocated and initialized to a value of 1.0
    \post \c mass is allocated and initialized to a value of 1.0
    \post \c image is allocated and initialized to values of 0.0
    \post \c tag is allocated and given the default initialization tag[i] = i
    \post \c the reverse lookup map rtag is initialized with the identity mapping
    \post \c type is allocated and given the default value of type[i] = 0
    \post \c body is allocated and given the devault value of type[i] = NO_BODY
    \post Arrays are not currently acquired

    Type mappings assign particle types "A", "B", "C", ....
*/
ParticleData::ParticleData(unsigned int N, const BoxDim &global_box, unsigned int n_types, boost::shared_ptr<ExecutionConfiguration> exec_conf, boost::shared_ptr<DomainDecomposition> decomposition)
        : m_exec_conf(exec_conf),
          m_nparticles(0),
          m_nghosts(0),
          m_max_nparticles(0),
          m_nglobal(0),
          m_resize_factor(9./8.)
    {
    m_exec_conf->msg->notice(5) << "Constructing ParticleData" << endl;

    // check the input for errors
    if (n_types == 0)
        {
        m_exec_conf->msg->error() << "Number of particle types must be greater than 0." << endl;
        throw std::runtime_error("Error initializing ParticleData");
        }

    // initialize snapshot with default values
    SnapshotParticleData snap(N);

    snap.type_mapping.clear();

    // setup the type mappings
    for (unsigned int i = 0; i < n_types; i++)
        {
        char name[2];
        name[0] = 'A' + i;
        name[1] = '\0';
        snap.type_mapping.push_back(string(name));
        }

    #ifdef ENABLE_MPI
    // Set up domain decomposition information
    if (decomposition) setDomainDecomposition(decomposition);
    #endif

    // initialize box dimensions on all procesors
    setGlobalBox(global_box);

    // initialize all processors
    initializeFromSnapshot(snap);

    // default constructed shared ptr is null as desired
    m_prof = boost::shared_ptr<Profiler>();

    // reset external virial
    for (unsigned int i = 0; i < 6; i++)
        m_external_virial[i] = Scalar(0.0);

    // zero the origin
    m_origin = make_scalar3(0,0,0);
    m_o_image = make_int3(0,0,0);

    #ifdef ENABLE_CUDA
    if (m_exec_conf->isCUDAEnabled())
        {
        // create a ModernGPU context
        m_mgpu_context = mgpu::CreateCudaDeviceAttachStream(0);
        }
    #endif
    }

/*! Loads particle data from the snapshot into the internal arrays.
 * \param snapshot The particle data snapshot
 * \param global_box The dimensions of the global simulation box
 * \param exec_conf The execution configuration
 * \param decomposition (optional) Domain decomposition layout
 */
ParticleData::ParticleData(const SnapshotParticleData& snapshot,
                           const BoxDim& global_box,
                           boost::shared_ptr<ExecutionConfiguration> exec_conf,
                           boost::shared_ptr<DomainDecomposition> decomposition
                          )
    : m_exec_conf(exec_conf),
      m_nparticles(0),
      m_nghosts(0),
      m_max_nparticles(0),
      m_nglobal(0),
      m_resize_factor(9./8.)
    {
    m_exec_conf->msg->notice(5) << "Constructing ParticleData" << endl;

    // initialize number of particles
    setNGlobal(snapshot.size);

    #ifdef ENABLE_MPI
    // Set up domain decomposition information
    if (decomposition) setDomainDecomposition(decomposition);
    #endif

    // initialize box dimensions on all procesors
    setGlobalBox(global_box);

    // initialize particle data with snapshot contents
    initializeFromSnapshot(snapshot);

    // it is an error for particles to be initialized outside of their box
    if (!inBox())
        {
        m_exec_conf->msg->error() << "Not all particles were found inside the given box" << endl;
        throw runtime_error("Error initializing ParticleData");
        }

    // reset external virial
    for (unsigned int i = 0; i < 6; i++)
        m_external_virial[i] = Scalar(0.0);

    // default constructed shared ptr is null as desired
    m_prof = boost::shared_ptr<Profiler>();

    // zero the origin
    m_origin = make_scalar3(0,0,0);
    m_o_image = make_int3(0,0,0);

    #ifdef ENABLE_CUDA
    if (m_exec_conf->isCUDAEnabled())
        {
        // create a ModernGPU context
        m_mgpu_context = mgpu::CreateCudaDeviceAttachStream(0);
        }
    #endif
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

/*! \param box New box dimensions to set
    \note ParticleData does NOT enforce any boundary conditions. When a new box is set,
        it is the responsibility of the caller to ensure that all particles lie within
        the new box.
*/
void ParticleData::setGlobalBox(const BoxDim& box)
    {
    assert(box.getPeriodic().x);
    assert(box.getPeriodic().y);
    assert(box.getPeriodic().z);
    m_global_box = box;

#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        bcast(m_global_box, 0, m_exec_conf->getMPICommunicator());
        m_box = m_decomposition->calculateLocalBox(m_global_box);
        }
    else
#endif
        {
        // local box = global box
        m_box = box;
        }

    m_boxchange_signal();
    }

/*! \return Global simulation box dimensions
 */
const BoxDim & ParticleData::getGlobalBox() const
    {
    return m_global_box;
    }

/*! \param func Function to call when the particles are resorted
    \return Connection to manage the signal/slot connection
    Calls are performed by using boost::signals2. The function passed in
    \a func will be called every time the ParticleData is notified of a particle
    sort via notifyParticleSort().
    \note If the caller class is destroyed, it needs to disconnect the signal connection
    via \b con.disconnect where \b con is the return value of this function.
*/
boost::signals2::connection ParticleData::connectParticleSort(const boost::function<void ()> &func)
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
    Calls are performed by using boost::signals2. The function passed in
    \a func will be called every time the the box size is changed via setGlobalBoxL()
    \note If the caller class is destroyed, it needs to disconnect the signal connection
    via \b con.disconnect where \b con is the return value of this function.
*/
boost::signals2::connection ParticleData::connectBoxChange(const boost::function<void ()> &func)
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
          and notifyParticleSort() needs to be called separately after all particle data is available
          on the local processor.
*/
boost::signals2::connection ParticleData::connectMaxParticleNumberChange(const boost::function<void ()> &func)
    {
    return m_max_particle_num_signal.connect(func);
    }

/*! \param func Function to be called when the number of ghost particles changes
    \return Connection to manage the signal
 */
boost::signals2::connection ParticleData::connectGhostParticleNumberChange(const boost::function<void ()> &func)
    {
    return m_ghost_particle_num_signal.connect(func);
    }

/*! This function must be called any time the ghost particles are updated.
 */
void ParticleData::notifyGhostParticleNumberChange()
    {
    m_ghost_particle_num_signal();
    }

#ifdef ENABLE_MPI
/*! \param func Function to be called when a single particle moves between domains
    \return Connection to manage the signal
 */
boost::signals2::connection ParticleData::connectSingleParticleMove(
    const boost::function<void(unsigned int, unsigned int, unsigned int)> &func)
    {
    return m_ptl_move_signal.connect(func);
    }
#endif

/*! \param name Type name to get the index of
    \return Type index of the corresponding type name
    \note Throws an exception if the type name is not found
*/
unsigned int ParticleData::getTypeByName(const std::string &name) const
    {
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
    // check for an invalid request
    if (type >= getNTypes())
        {
        m_exec_conf->msg->error() << "Requesting type name for non-existant type " << type << endl;
        throw runtime_error("Error mapping type name");
        }

    // return the name
    return m_type_mapping[type];
    }


/*! \param N Number of particles to allocate memory for
    \pre No memory is allocated and the per-particle GPUArrays are unitialized
    \post All per-particle GPUArrays are allocated
*/
void ParticleData::allocate(unsigned int N)
    {
    // check the input
    if (N == 0)
        {
        m_exec_conf->msg->error() << "ParticleData is being asked to allocate 0 particles.... this makes no sense whatsoever" << endl;
        throw runtime_error("Error allocating ParticleData");
        }

    // maximum number is the current particle number
    m_max_nparticles = N;

    // positions
    GPUArray< Scalar4 > pos(N, m_exec_conf);
    m_pos.swap(pos);

    // velocities
    GPUArray< Scalar4 > vel(N, m_exec_conf);
    m_vel.swap(vel);

    // accelerations
    GPUArray< Scalar3 > accel(N, m_exec_conf);
    m_accel.swap(accel);

    // charge
    GPUArray< Scalar > charge(N, m_exec_conf);
    m_charge.swap(charge);

    // diameter
    GPUArray< Scalar > diameter(N, m_exec_conf);
    m_diameter.swap(diameter);

    // image
    GPUArray< int3 > image(N, m_exec_conf);
    m_image.swap(image);

    // global tag
    GPUArray< unsigned int> tag(N, m_exec_conf);
    m_tag.swap(tag);

    // body ID
    GPUArray< unsigned int > body(N, m_exec_conf);
    m_body.swap(body);

    GPUArray< Scalar4 > net_force(N, m_exec_conf);
    m_net_force.swap(net_force);
    GPUArray< Scalar > net_virial(N,6, m_exec_conf);
    m_net_virial.swap(net_virial);
    GPUArray< Scalar4 > net_torque(N, m_exec_conf);
    m_net_torque.swap(net_torque);
    GPUArray< Scalar4 > orientation(N, m_exec_conf);
    m_orientation.swap(orientation);
    GPUArray< Scalar4 > angmom(N, m_exec_conf);
    m_angmom.swap(angmom);
    GPUArray< Scalar3 > inertia(N, m_exec_conf);
    m_inertia.swap(inertia);

    #ifdef ENABLE_MPI
    if (m_decomposition)
        {
        GPUArray< unsigned int > comm_flags(N, m_exec_conf);
        m_comm_flags.swap(comm_flags);
        }
    #endif

    // allocate alternate particle data arrays (for swapping in-out)
    allocateAlternateArrays(N);

    // notify observers
    m_max_particle_num_signal();
    }

/*! \param N Number of particles to allocate memory for
    \pre No memory is allocated and the alternate per-particle GPUArrays are unitialized
    \post All alternate per-particle GPUArrays are allocated
*/
void ParticleData::allocateAlternateArrays(unsigned int N)
    {
    assert(N>0);

    // positions
    GPUArray< Scalar4 > pos_alt(N, m_exec_conf);
    m_pos_alt.swap(pos_alt);

    // velocities
    GPUArray< Scalar4 > vel_alt(N, m_exec_conf);
    m_vel_alt.swap(vel_alt);

    // accelerations
    GPUArray< Scalar3 > accel_alt(N, m_exec_conf);
    m_accel_alt.swap(accel_alt);

    // charge
    GPUArray< Scalar > charge_alt(N, m_exec_conf);
    m_charge_alt.swap(charge_alt);

    // diameter
    GPUArray< Scalar > diameter_alt(N, m_exec_conf);
    m_diameter_alt.swap(diameter_alt);

    // image
    GPUArray< int3 > image_alt(N, m_exec_conf);
    m_image_alt.swap(image_alt);

    // global tag
    GPUArray< unsigned int> tag_alt(N, m_exec_conf);
    m_tag_alt.swap(tag_alt);

    // body ID
    GPUArray< unsigned int > body_alt(N, m_exec_conf);
    m_body_alt.swap(body_alt);

    // orientation
    GPUArray< Scalar4 > orientation_alt(N, m_exec_conf);
    m_orientation_alt.swap(orientation_alt);

    // angular momentum
    GPUArray< Scalar4 > angmom_alt(N, m_exec_conf);
    m_angmom_alt.swap(angmom_alt);

    // moments of inertia
    GPUArray< Scalar3 > inertia_alt(N, m_exec_conf);
    m_inertia_alt.swap(inertia_alt);

    // Net force
    GPUArray< Scalar4 > net_force_alt(N, m_exec_conf);
    m_net_force_alt.swap(net_force_alt);

    // Net virial
    GPUArray< Scalar > net_virial_alt(N,6, m_exec_conf);
    m_net_virial_alt.swap(net_virial_alt);

    // Net torque
    GPUArray< Scalar4 > net_torque_alt(N, m_exec_conf);
    m_net_torque_alt.swap(net_torque_alt);
    }


//! Set global number of particles
/*! \param nglobal Global number of particles
 */
void ParticleData::setNGlobal(unsigned int nglobal)
    {
    if (m_nparticles > nglobal)
        {
        m_exec_conf->msg->error() << "ParticleData is being asked to allocate memory for a global number"
                                  << "   of particles smaller than the local number of particles." << std::endl;
        throw runtime_error("Error initializing ParticleData");
        }
    if (m_nglobal)
        {
        if (nglobal > m_nglobal)
            {
            // resize array of global reverse lookup tags
            m_rtag.resize(nglobal);
            }
        }
    else
        {
        // allocate array
        GPUArray< unsigned int> rtag(nglobal, m_exec_conf);
        m_rtag.swap(rtag);
        }

    // Set global particle number
    m_nglobal = nglobal;

    }

/*! \param new_nparticles New particle number
 */
void ParticleData::resize(unsigned int new_nparticles)
    {
    // resize pdata arrays as necessary
    unsigned int max_nparticles = m_max_nparticles;
    if (new_nparticles > max_nparticles)
        {
        // use amortized array resizing
        while (new_nparticles > max_nparticles)
            max_nparticles = ((unsigned int) (((float) max_nparticles) * m_resize_factor)) + 1 ;

        // reallocate particle data arrays
        reallocate(max_nparticles);
        }

    m_nparticles = new_nparticles;
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
    m_exec_conf->msg->notice(7) << "Resizing particle data arrays "
        << m_max_nparticles << " -> " << max_n << " ptls" << std::endl;
    m_max_nparticles = max_n;

    m_pos.resize(max_n);
    m_vel.resize(max_n);
    m_accel.resize(max_n);
    m_charge.resize(max_n);
    m_diameter.resize(max_n);
    m_image.resize(max_n);
    m_tag.resize(max_n);
    m_body.resize(max_n);

    m_net_force.resize(max_n);
    m_net_virial.resize(max_n,6);
    m_net_torque.resize(max_n);
    m_orientation.resize(max_n);
    m_angmom.resize(max_n);
    m_inertia.resize(max_n);

    #ifdef ENABLE_MPI
    if (m_decomposition) m_comm_flags.resize(max_n);
    #endif

    if (! m_pos_alt.isNull())
        {
        // reallocate alternate arrays
        m_pos_alt.resize(max_n);
        m_vel_alt.resize(max_n);
        m_accel_alt.resize(max_n);
        m_charge_alt.resize(max_n);
        m_diameter_alt.resize(max_n);
        m_image_alt.resize(max_n);
        m_tag_alt.resize(max_n);
        m_body_alt.resize(max_n);
        m_orientation_alt.resize(max_n);
        m_angmom_alt.resize(max_n);
        m_inertia_alt.resize(max_n);
        m_net_force_alt.resize(max_n);
        m_net_torque_alt.resize(max_n);
        m_net_virial_alt.resize(max_n, 6);
        }

    // notify observers
    m_max_particle_num_signal();
    }

/*! \return true If and only if all particles are in the simulation box
*/
bool ParticleData::inBox()
    {
    Scalar3 lo = m_box.getLo();
    Scalar3 hi = m_box.getHi();

    const Scalar tol = Scalar(1e-5);

    ArrayHandle<Scalar4> h_pos(getPositions(), access_location::host, access_mode::read);
    for (unsigned int i = 0; i < getN(); i++)
        {
        Scalar3 pos = make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z);
        Scalar3 f = m_box.makeFraction(pos);
        if (f.x < -tol || f.x > Scalar(1.0)+tol)
            {
            m_exec_conf->msg->warning() << "pos " << i << ":" << setprecision(12) << h_pos.data[i].x << " " << h_pos.data[i].y << " " << h_pos.data[i].z << endl;
            m_exec_conf->msg->warning() << "fractional pos :" << setprecision(12) << f.x << " " << f.y << " " << f.z << endl;
            m_exec_conf->msg->warning() << "lo: " << lo.x << " " << lo.y << " " << lo.z << endl;
            m_exec_conf->msg->warning() << "hi: " << hi.x << " " << hi.y << " " << hi.z << endl;
            return false;
            }
        if (f.y < -tol || f.y > Scalar(1.0)+tol)
            {
            m_exec_conf->msg->warning() << "pos " << i << ":" << setprecision(12) << h_pos.data[i].x << " " << h_pos.data[i].y << " " << h_pos.data[i].z << endl;
            m_exec_conf->msg->warning() << "fractional pos :" << setprecision(12) << f.x << " " << f.y << " " << f.z << endl;
            m_exec_conf->msg->warning() << "lo: " << lo.x << " " << lo.y << " " << lo.z << endl;
            m_exec_conf->msg->warning() << "hi: " << hi.x << " " << hi.y << " " << hi.z << endl;
            return false;
            }
        if (f.z < -tol || f.z > Scalar(1.0)+tol)
            {
            m_exec_conf->msg->warning() << "pos " << i << ":" << setprecision(12) << h_pos.data[i].x << " " << h_pos.data[i].y << " " << h_pos.data[i].z << endl;
            m_exec_conf->msg->warning() << "fractional pos :" << setprecision(12) << f.x << " " << f.y << " " << f.z << endl;
            m_exec_conf->msg->warning() << "lo: " << lo.x << " " << lo.y << " " << lo.z << endl;
            m_exec_conf->msg->warning() << "hi: " << hi.x << " " << hi.y << " " << hi.z << endl;
            return false;
            }
        }
    return true;
    }

//! Initialize from a snapshot
/*! \param snapshot the initial particle data

    \post the particle data arrays are initialized from the snapshot, in index order

    \pre In parallel simulations, the local box size must be set before a call to initializeFromSnapshot().
 */
void ParticleData::initializeFromSnapshot(const SnapshotParticleData& snapshot)
    {
    m_exec_conf->msg->notice(4) << "ParticleData: initializing from snapshot" << std::endl;

    // check that all fields in the snapshot have correct length
    if (m_exec_conf->getRank() == 0 && ! snapshot.validate())
        {
        m_exec_conf->msg->error() << "init.*: invalid particle data snapshot."
                                << std::endl << std::endl;
        throw std::runtime_error("Error initializing particle data.");
        }

#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        // gather box information from all processors
        unsigned int root = 0;

        // Define per-processor particle data
        std::vector< std::vector<Scalar3> > pos_proc;              // Position array of every processor
        std::vector< std::vector<Scalar3> > vel_proc;              // Velocities array of every processor
        std::vector< std::vector<Scalar3> > accel_proc;            // Accelerations array of every processor
        std::vector< std::vector<unsigned int> > type_proc;        // Particle types array of every processor
        std::vector< std::vector<Scalar > > mass_proc;             // Particle masses array of every processor
        std::vector< std::vector<Scalar > > charge_proc;           // Particle charges array of every processor
        std::vector< std::vector<Scalar > > diameter_proc;         // Particle diameters array of every processor
        std::vector< std::vector<int3 > > image_proc;              // Particle images array of every processor
        std::vector< std::vector<unsigned int > > body_proc;       // Body ids of every processor
        std::vector< std::vector<Scalar4> > orientation_proc;      // Orientations of every processor
        std::vector< std::vector<Scalar4> > angmom_proc;           // Angular momenta of every processor
        std::vector< std::vector<Scalar3> > inertia_proc;           // Angular momenta of every processor
        std::vector< std::vector<unsigned int > > tag_proc;         // Global tags of every processor
        std::vector< unsigned int > N_proc;                        // Number of particles on every processor


        // resize to number of ranks in communicator
        const MPI_Comm mpi_comm = m_exec_conf->getMPICommunicator();
        unsigned int size = m_exec_conf->getNRanks();
        unsigned int my_rank = m_exec_conf->getRank();

        pos_proc.resize(size);
        vel_proc.resize(size);
        accel_proc.resize(size);
        type_proc.resize(size);
        mass_proc.resize(size);
        charge_proc.resize(size);
        diameter_proc.resize(size);
        image_proc.resize(size);
        body_proc.resize(size);
        orientation_proc.resize(size);
        angmom_proc.resize(size);
        inertia_proc.resize(size);
        tag_proc.resize(size);
        N_proc.resize(size,0);

        if (my_rank == 0)
            {
            // check the input for errors
            if (snapshot.type_mapping.size() == 0)
                {
                m_exec_conf->msg->error() << "Number of particle types must be greater than 0." << endl;
                throw std::runtime_error("Error initializing ParticleData");
                }

            const Index3D& di = m_decomposition->getDomainIndexer();
            unsigned int n_ranks = m_exec_conf->getNRanks();
            ArrayHandle<unsigned int> h_cart_ranks(m_decomposition->getCartRanks(), access_location::host, access_mode::read);

            BoxDim global_box = m_global_box;

            // loop over particles in snapshot, place them into domains
            for (std::vector<Scalar3>::const_iterator it=snapshot.pos.begin(); it != snapshot.pos.end(); it++)
                {
                // determine domain the particle is placed into
                Scalar3 pos = *it;
                Scalar3 f = m_global_box.makeFraction(pos);
                int i= f.x * ((Scalar)di.getW());
                int j= f.y * ((Scalar)di.getH());
                int k= f.z * ((Scalar)di.getD());

                // wrap particles that are exactly on a boundary
                // we only need to wrap in the negative direction, since
                // processor ids are rounded toward zero
                char3 flags = make_char3(0,0,0);
                if (i == (int) di.getW())
                    {
                    i = 0;
                    flags.x = 1;
                    }

                if (j == (int) di.getH())
                    {
                    j = 0;
                    flags.y = 1;
                    }

                if (k == (int) di.getD())
                    {
                    k = 0;
                    flags.z = 1;
                    }

                int3 img = snapshot.image[tag];

                // only wrap if the particles is on one of the boundaries
                uchar3 periodic = make_uchar3(flags.x,flags.y,flags.z);
                global_box.setPeriodic(periodic);
                global_box.wrap(pos, img, flags);

                unsigned int rank = h_cart_ranks.data[di(i,j,k)];
                unsigned int tag = it - snapshot.pos.begin();

                if (rank >= n_ranks)
                    {
                    m_exec_conf->msg->error() << "init.*: Particle " << tag << " out of bounds." << std::endl;
                    m_exec_conf->msg->error() << "Cartesian coordinates: " << std::endl;
                    m_exec_conf->msg->error() << "x: " << pos.x << " y: " << pos.y << " z: " << pos.z << std::endl;
                    m_exec_conf->msg->error() << "Fractional coordinates: " << std::endl;
                    m_exec_conf->msg->error() << "f.x: " << f.x << " f.y: " << f.y << " f.z: " << f.z << std::endl;
                    Scalar3 lo = m_global_box.getLo();
                    Scalar3 hi = m_global_box.getHi();
                    m_exec_conf->msg->error() << "Global box lo: (" << lo.x << ", " << lo.y << ", " << lo.z << ")" << std::endl;
                    m_exec_conf->msg->error() << "           hi: (" << hi.x << ", " << hi.y << ", " << hi.z << ")" << std::endl;

                    throw std::runtime_error("Error initializing from snapshot.");
                    }

                // fill up per-processor data structures
                pos_proc[rank].push_back(pos);
                image_proc[rank].push_back(img);
                vel_proc[rank].push_back(snapshot.vel[tag]);
                accel_proc[rank].push_back(snapshot.accel[tag]);
                type_proc[rank].push_back(snapshot.type[tag]);
                mass_proc[rank].push_back(snapshot.mass[tag]);
                charge_proc[rank].push_back(snapshot.charge[tag]);
                diameter_proc[rank].push_back(snapshot.diameter[tag]);
                body_proc[rank].push_back(snapshot.body[tag]);
                orientation_proc[rank].push_back(snapshot.orientation[tag]);
                angmom_proc[rank].push_back(snapshot.angmom[tag]);
                inertia_proc[rank].push_back(snapshot.inertia[tag]);
                tag_proc[rank].push_back(tag);
                N_proc[rank]++;
                }

            }

        // get type mapping
        m_type_mapping = snapshot.type_mapping;

        // broadcast type mapping
        bcast(m_type_mapping, root, mpi_comm);

        // broadcast global number of particles
        unsigned int nglobal = snapshot.size;
        bcast(nglobal, root, mpi_comm);

        setNGlobal(nglobal);

        // Local particle data
        std::vector<Scalar3> pos;
        std::vector<Scalar3> vel;
        std::vector<Scalar3> accel;
        std::vector<unsigned int> type;
        std::vector<Scalar> mass;
        std::vector<Scalar> charge;
        std::vector<Scalar> diameter;
        std::vector<int3> image;
        std::vector<unsigned int> body;
        std::vector<Scalar4> orientation;
        std::vector<Scalar4> angmom;
        std::vector<Scalar3> inertia;
        std::vector<unsigned int> tag;

        // distribute particle data
        scatter_v(pos_proc,pos,root, mpi_comm);
        scatter_v(vel_proc,vel,root, mpi_comm);
        scatter_v(accel_proc, accel, root, mpi_comm);
        scatter_v(type_proc, type, root, mpi_comm);
        scatter_v(mass_proc, mass, root, mpi_comm);
        scatter_v(charge_proc, charge, root, mpi_comm);
        scatter_v(diameter_proc, diameter, root, mpi_comm);
        scatter_v(image_proc, image, root, mpi_comm);
        scatter_v(body_proc, body, root, mpi_comm);
        scatter_v(orientation_proc, orientation, root, mpi_comm);
        scatter_v(angmom_proc, angmom, root, mpi_comm);
        scatter_v(inertia_proc, inertia, root, mpi_comm);
        scatter_v(tag_proc, tag, root, mpi_comm);

        // distribute number of particles
        scatter_v(N_proc, m_nparticles, root, mpi_comm);

        // reset all reverse lookup tags to NOT_LOCAL flag
            {
            ArrayHandle<unsigned int> h_rtag(getRTags(), access_location::host, access_mode::overwrite);
            for (unsigned int tag = 0; tag < m_nglobal; tag++)
                h_rtag.data[tag] = NOT_LOCAL;
            }

        // we have to allocate even if the number of particles on a processor
        // is zero, so that the arrays can be resized later
        if (m_nparticles == 0)
            allocate(1);
        else
            allocate(m_nparticles);

        // Load particle data
        ArrayHandle< Scalar4 > h_pos(m_pos, access_location::host, access_mode::overwrite);
        ArrayHandle< Scalar4 > h_vel(m_vel, access_location::host, access_mode::overwrite);
        ArrayHandle< Scalar3 > h_accel(m_accel, access_location::host, access_mode::overwrite);
        ArrayHandle< int3 > h_image(m_image, access_location::host, access_mode::overwrite);
        ArrayHandle< Scalar > h_charge(m_charge, access_location::host, access_mode::overwrite);
        ArrayHandle< Scalar > h_diameter(m_diameter, access_location::host, access_mode::overwrite);
        ArrayHandle< unsigned int > h_body(m_body, access_location::host, access_mode::overwrite);
        ArrayHandle< Scalar4 > h_orientation(m_orientation, access_location::host, access_mode::overwrite);
        ArrayHandle< Scalar4 > h_angmom(m_angmom, access_location::host, access_mode::overwrite);
        ArrayHandle< Scalar3 > h_inertia(m_inertia, access_location::host, access_mode::overwrite);
        ArrayHandle< unsigned int > h_tag(m_tag, access_location::host, access_mode::overwrite);
        ArrayHandle< unsigned int > h_comm_flag(m_comm_flags, access_location::host, access_mode::overwrite);
        ArrayHandle< unsigned int > h_rtag(m_rtag, access_location::host, access_mode::readwrite);

        for (unsigned int idx = 0; idx < m_nparticles; idx++)
            {
            h_pos.data[idx] = make_scalar4(pos[idx].x,pos[idx].y, pos[idx].z, __int_as_scalar(type[idx]));
            h_vel.data[idx] = make_scalar4(vel[idx].x, vel[idx].y, vel[idx].z, mass[idx]);
            h_accel.data[idx] = accel[idx];
            h_charge.data[idx] = charge[idx];
            h_diameter.data[idx] = diameter[idx];
            h_image.data[idx] = image[idx];
            h_tag.data[idx] = tag[idx];
            h_rtag.data[tag[idx]] = idx;
            h_body.data[idx] = body[idx];
            h_orientation.data[idx] = orientation[idx];
            h_angmom.data[idx] = angmom[idx];
            h_inertia.data[idx] = inertia[idx];

            h_comm_flag.data[idx] = 0; // initialize with zero
            }

        // reset ghost particle number
        m_nghosts = 0;

        // notify about change in ghost particle number
        notifyGhostParticleNumberChange();
        }
    else
#endif
        {
        // check the input for errors
        if (snapshot.type_mapping.size() == 0)
            {
            m_exec_conf->msg->error() << "Number of particle types must be greater than 0." << endl;
            throw std::runtime_error("Error initializing ParticleData");
            }

        // Initialize number of particles
        setNGlobal(snapshot.size);
        m_nparticles = snapshot.size;

        // allocate particle data such that we can accomodate the particles
        allocate(snapshot.size);

        ArrayHandle< Scalar4 > h_pos(m_pos, access_location::host, access_mode::overwrite);
        ArrayHandle< Scalar4 > h_vel(m_vel, access_location::host, access_mode::overwrite);
        ArrayHandle< Scalar3 > h_accel(m_accel, access_location::host, access_mode::overwrite);
        ArrayHandle< int3 > h_image(m_image, access_location::host, access_mode::overwrite);
        ArrayHandle< Scalar > h_charge(m_charge, access_location::host, access_mode::overwrite);
        ArrayHandle< Scalar > h_diameter(m_diameter, access_location::host, access_mode::overwrite);
        ArrayHandle< unsigned int > h_body(m_body, access_location::host, access_mode::overwrite);
        ArrayHandle< Scalar4 > h_orientation(m_orientation, access_location::host, access_mode::overwrite);
        ArrayHandle< Scalar4 > h_angmom(m_angmom, access_location::host, access_mode::overwrite);
        ArrayHandle< Scalar3 > h_inertia(m_inertia, access_location::host, access_mode::overwrite);
        ArrayHandle< unsigned int > h_tag(m_tag, access_location::host, access_mode::overwrite);
        ArrayHandle< unsigned int > h_rtag(m_rtag, access_location::host, access_mode::readwrite);

        for (unsigned int tag = 0; tag < m_nparticles; tag++)
            {
            h_pos.data[tag] = make_scalar4(snapshot.pos[tag].x,
                                           snapshot.pos[tag].y,
                                           snapshot.pos[tag].z,
                                           __int_as_scalar(snapshot.type[tag]));
            h_vel.data[tag] = make_scalar4(snapshot.vel[tag].x,
                                             snapshot.vel[tag].y,
                                             snapshot.vel[tag].z,
                                             snapshot.mass[tag]);
            h_accel.data[tag] = snapshot.accel[tag];
            h_charge.data[tag] = snapshot.charge[tag];
            h_diameter.data[tag] = snapshot.diameter[tag];
            h_image.data[tag] = snapshot.image[tag];
            h_tag.data[tag] = tag;
            h_rtag.data[tag] = tag;
            h_body.data[tag] = snapshot.body[tag];
            h_orientation.data[tag] = snapshot.orientation[tag];
            h_angmom.data[tag] = snapshot.angmom[tag];
            h_inertia.data[tag] = snapshot.inertia[tag];
            }

        // initialize type mapping
        m_type_mapping = snapshot.type_mapping;
        }

    notifyParticleSort();

    // zero the origin
    m_origin = make_scalar3(0,0,0);
    m_o_image = make_int3(0,0,0);

    m_exec_conf->msg->notice(4) << "ParticleData: finished initializing from snapshot" << std::endl;
    }

//! take a particle data snapshot
/* \param snapshot The snapshot to write to

   \pre snapshot has to be allocated with a number of elements equal to the global number of particles)
*/
void ParticleData::takeSnapshot(SnapshotParticleData &snapshot)
    {
    m_exec_conf->msg->notice(4) << "ParticleData: taking snapshot" << std::endl;
    // allocate memory in snapshot
    snapshot.resize(getNGlobal());

    ArrayHandle< Scalar4 > h_pos(m_pos, access_location::host, access_mode::read);
    ArrayHandle< Scalar4 > h_vel(m_vel, access_location::host, access_mode::read);
    ArrayHandle< Scalar3 > h_accel(m_accel, access_location::host, access_mode::read);
    ArrayHandle< int3 > h_image(m_image, access_location::host, access_mode::read);
    ArrayHandle< Scalar > h_charge(m_charge, access_location::host, access_mode::read);
    ArrayHandle< Scalar > h_diameter(m_diameter, access_location::host, access_mode::read);
    ArrayHandle< unsigned int > h_body(m_body, access_location::host, access_mode::read);
    ArrayHandle< Scalar4 >  h_orientation(m_orientation, access_location::host, access_mode::read);
    ArrayHandle< Scalar4 >  h_angmom(m_angmom, access_location::host, access_mode::read);
    ArrayHandle< Scalar3 >  h_inertia(m_inertia, access_location::host, access_mode::read);
    ArrayHandle< unsigned int > h_tag(m_tag, access_location::host, access_mode::read);

#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        // gather a global snapshot
        std::vector<Scalar3> pos(m_nparticles);
        std::vector<Scalar3> vel(m_nparticles);
        std::vector<Scalar3> accel(m_nparticles);
        std::vector<unsigned int> type(m_nparticles);
        std::vector<Scalar> mass(m_nparticles);
        std::vector<Scalar> charge(m_nparticles);
        std::vector<Scalar> diameter(m_nparticles);
        std::vector<int3> image(m_nparticles);
        std::vector<unsigned int> body(m_nparticles);
        std::vector<Scalar4> orientation(m_nparticles);
        std::vector<Scalar4> angmom(m_nparticles);
        std::vector<Scalar3> inertia(m_nparticles);
        std::vector<unsigned int> tag(m_nparticles);
        std::map<unsigned int, unsigned int> rtag_map;
        for (unsigned int idx = 0; idx < m_nparticles; idx++)
            {
            pos[idx] = make_scalar3(h_pos.data[idx].x, h_pos.data[idx].y, h_pos.data[idx].z) - m_origin;
            vel[idx] = make_scalar3(h_vel.data[idx].x, h_vel.data[idx].y, h_vel.data[idx].z);
            accel[idx] = h_accel.data[idx];
            type[idx] = __scalar_as_int(h_pos.data[idx].w);
            mass[idx] = h_vel.data[idx].w;
            charge[idx] = h_charge.data[idx];
            diameter[idx] = h_diameter.data[idx];
            image[idx] = h_image.data[idx];
            image[idx].x -= m_o_image.x;
            image[idx].y -= m_o_image.y;
            image[idx].z -= m_o_image.z;
            body[idx] = h_body.data[idx];
            orientation[idx] = h_orientation.data[idx];
            angmom[idx] = h_angmom.data[idx];
            inertia[idx] = h_inertia.data[idx];

            // insert reverse lookup global tag -> idx
            rtag_map.insert(std::pair<unsigned int, unsigned int>(h_tag.data[idx], idx));
            }

        std::vector< std::vector<Scalar3> > pos_proc;              // Position array of every processor
        std::vector< std::vector<Scalar3> > vel_proc;              // Velocities array of every processor
        std::vector< std::vector<Scalar3> > accel_proc;            // Accelerations array of every processor
        std::vector< std::vector<unsigned int> > type_proc;        // Particle types array of every processor
        std::vector< std::vector<Scalar > > mass_proc;             // Particle masses array of every processor
        std::vector< std::vector<Scalar > > charge_proc;           // Particle charges array of every processor
        std::vector< std::vector<Scalar > > diameter_proc;         // Particle diameters array of every processor
        std::vector< std::vector<int3 > > image_proc;              // Particle images array of every processor
        std::vector< std::vector<unsigned int > > body_proc;       // Body ids of every processor
        std::vector< std::vector<Scalar4 > > orientation_proc;     // Orientations of every processor
        std::vector< std::vector<Scalar4 > > angmom_proc;          // Angular momenta of every processor
        std::vector< std::vector<Scalar3 > > inertia_proc;         // Moments of inertia of every processor

        std::vector< std::map<unsigned int, unsigned int> > rtag_map_proc; // List of reverse-lookup maps

        const MPI_Comm mpi_comm = m_exec_conf->getMPICommunicator();
        unsigned int size = m_exec_conf->getNRanks();
        unsigned int rank = m_exec_conf->getRank();

        // resize to number of ranks in communicator
        pos_proc.resize(size);
        vel_proc.resize(size);
        accel_proc.resize(size);
        type_proc.resize(size);
        mass_proc.resize(size);
        charge_proc.resize(size);
        diameter_proc.resize(size);
        image_proc.resize(size);
        body_proc.resize(size);
        orientation_proc.resize(size);
        angmom_proc.resize(size);
        inertia_proc.resize(size);
        rtag_map_proc.resize(size);

        unsigned int root = 0;

        // collect all particle data on the root processor
        gather_v(pos, pos_proc, root,mpi_comm);
        gather_v(vel, vel_proc, root, mpi_comm);
        gather_v(accel, accel_proc, root, mpi_comm);
        gather_v(type, type_proc, root, mpi_comm);
        gather_v(mass, mass_proc, root, mpi_comm);
        gather_v(charge, charge_proc, root, mpi_comm);
        gather_v(diameter, diameter_proc, root, mpi_comm);
        gather_v(image, image_proc, root, mpi_comm);
        gather_v(body, body_proc, root, mpi_comm);
        gather_v(orientation, orientation_proc, root, mpi_comm);
        gather_v(angmom, angmom_proc, root, mpi_comm);
        gather_v(inertia, inertia_proc, root, mpi_comm);

        // gather the reverse-lookup maps
        gather_v(rtag_map, rtag_map_proc, root, mpi_comm);

        if (rank == root)
            {

            unsigned int n_ranks = m_exec_conf->getNRanks();
            assert(rtag_map_proc.size() == n_ranks);

            // create single map of all particle ranks and indices
            std::map<unsigned int, std::pair<unsigned int, unsigned int> > rank_rtag_map;
            std::map<unsigned int, unsigned int>::iterator it;
            for (unsigned int irank = 0; irank < n_ranks; ++irank)
                for (it = rtag_map_proc[irank].begin(); it != rtag_map_proc[irank].end(); ++it)
                    rank_rtag_map.insert(std::pair<unsigned int, std::pair<unsigned int, unsigned int> >(
                        it->first, std::pair<unsigned int, unsigned int>(irank, it->second)));

            // add particles to snapshot
            std::map<unsigned int, std::pair<unsigned int, unsigned int> >::iterator rank_rtag_it;
            for (unsigned int tag = 0; tag < getNGlobal(); tag++)
                {
                rank_rtag_it = rank_rtag_map.find(tag);
                if (rank_rtag_it == rank_rtag_map.end())
                    {
                    m_exec_conf->msg->error()
                        << endl << "Could not find particle " << tag << " on any processor. "
                        << endl << endl;
                    throw std::runtime_error("Error gathering ParticleData");
                    }

                // rank contains the processor rank on which the particle was found
                std::pair<unsigned int, unsigned int> rank_idx = rank_rtag_it->second;
                unsigned int rank = rank_idx.first;
                unsigned int idx = rank_idx.second;

                snapshot.pos[tag] = pos_proc[rank][idx];
                snapshot.vel[tag] = vel_proc[rank][idx];
                snapshot.accel[tag] = accel_proc[rank][idx];
                snapshot.type[tag] = type_proc[rank][idx];
                snapshot.mass[tag] = mass_proc[rank][idx];
                snapshot.charge[tag] = charge_proc[rank][idx];
                snapshot.diameter[tag] = diameter_proc[rank][idx];
                snapshot.image[tag] = image_proc[rank][idx];
                snapshot.body[tag] = body_proc[rank][idx];
                snapshot.orientation[tag] = orientation_proc[rank][idx];
                snapshot.angmom[tag] = angmom_proc[rank][idx];
                snapshot.inertia[tag] = inertia_proc[rank][idx];

                // make sure the position stored in the snapshot is within the boundaries
                m_global_box.wrap(snapshot.pos[tag], snapshot.image[tag]);
                }
            }
        }
    else
#endif
        {
        for (unsigned int idx = 0; idx < m_nparticles; idx++)
            {
            unsigned int tag = h_tag.data[idx];
            assert(tag < m_nglobal);
            snapshot.pos[tag] = make_scalar3(h_pos.data[idx].x, h_pos.data[idx].y, h_pos.data[idx].z) - m_origin;
            snapshot.vel[tag] = make_scalar3(h_vel.data[idx].x, h_vel.data[idx].y, h_vel.data[idx].z);
            snapshot.accel[tag] = h_accel.data[idx];
            snapshot.type[tag] = __scalar_as_int(h_pos.data[idx].w);
            snapshot.mass[tag] = h_vel.data[idx].w;
            snapshot.charge[tag] = h_charge.data[idx];
            snapshot.diameter[tag] = h_diameter.data[idx];
            snapshot.image[tag] = h_image.data[idx];
            snapshot.image[tag].x -= m_o_image.x;
            snapshot.image[tag].y -= m_o_image.y;
            snapshot.image[tag].z -= m_o_image.z;
            snapshot.body[tag] = h_body.data[idx];
            snapshot.orientation[tag] = h_orientation.data[idx];
            snapshot.angmom[tag] = h_angmom.data[idx];
            snapshot.inertia[tag] = h_inertia.data[idx];

            // make sure the position stored in the snapshot is within the boundaries
            m_global_box.wrap(snapshot.pos[tag], snapshot.image[tag]);
            }
        }

    snapshot.type_mapping = m_type_mapping;

    m_exec_conf->msg->notice(4) << "ParticleData: finished taking snapshot" << std::endl;
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
            max_nparticles = ((unsigned int) (((float) max_nparticles) * m_resize_factor)) + 1 ;

        // reallocate particle data arrays
        reallocate(max_nparticles);
        }

    }

#ifdef ENABLE_MPI
//! Find the processor that owns a particle
/*! \param tag Tag of the particle to search
 */
unsigned int ParticleData::getOwnerRank(unsigned int tag) const
    {
    assert(m_decomposition);
    int is_local = (getRTag(tag) < getN()) ? 1 : 0;
    int n_found;

    const MPI_Comm mpi_comm = m_exec_conf->getMPICommunicator();
    // First check that the particle is on exactly one processor
    MPI_Allreduce(&is_local, &n_found, 1, MPI_INT, MPI_SUM, mpi_comm);

    if (n_found == 0)
        {
        m_exec_conf->msg->error() << "Could not find particle " << tag << " on any processor." << endl << endl;
        throw std::runtime_error("Error accessing particle data.");
        }
    else if (n_found > 1)
       {
        m_exec_conf->msg->error() << "Found particle " << tag << " on multiple processors." << endl << endl;
        throw std::runtime_error("Error accessing particle data.");
       }

    // Now find the processor that owns it
    int owner_rank;
    int flag =  is_local ? m_exec_conf->getRank() : -1;
    MPI_Allreduce(&flag, &owner_rank, 1, MPI_INT, MPI_MAX, mpi_comm);

    assert (owner_rank >= 0);
    assert ((unsigned int) owner_rank < m_exec_conf->getNRanks());

    return (unsigned int) owner_rank;
    }
#endif

///////////////////////////////////////////////////////////
// get accessors

//! Get the current position of a particle
Scalar3 ParticleData::getPosition(unsigned int tag) const
    {
    unsigned int idx = getRTag(tag);
    bool found = (idx < getN());
    Scalar3 result = make_scalar3(0.0,0.0,0.0);
    if (found)
        {
        ArrayHandle< Scalar4 > h_pos(m_pos, access_location::host, access_mode::read);
        result = make_scalar3(h_pos.data[idx].x, h_pos.data[idx].y, h_pos.data[idx].z);
        }
#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        unsigned int owner_rank = getOwnerRank(tag);
        bcast(result, owner_rank, m_exec_conf->getMPICommunicator());
        found = true;
        }
#endif
    assert(found);
    return result;
    }

//! Get the current velocity of a particle
Scalar3 ParticleData::getVelocity(unsigned int tag) const
    {
    unsigned int idx = getRTag(tag);
    bool found = (idx < getN());
    Scalar3 result = make_scalar3(0.0,0.0,0.0);
    if (found)
        {
        ArrayHandle< Scalar4 > h_vel(m_vel, access_location::host, access_mode::read);
        result = make_scalar3(h_vel.data[idx].x, h_vel.data[idx].y, h_vel.data[idx].z);
        }
#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        unsigned int owner_rank = getOwnerRank(tag);
        bcast(result, owner_rank, m_exec_conf->getMPICommunicator());
        found = true;
        }
#endif
    assert(found);
    return result;
    }

//! Get the current acceleration of a particle
Scalar3 ParticleData::getAcceleration(unsigned int tag) const
    {
    unsigned int idx = getRTag(tag);
    bool found = (idx < getN());
    Scalar3 result = make_scalar3(0.0,0.0,0.0);
    if (found)
        {
        ArrayHandle< Scalar3 > h_accel(m_accel, access_location::host, access_mode::read);
        result = make_scalar3(h_accel.data[idx].x, h_accel.data[idx].y, h_accel.data[idx].z);
        }
#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        unsigned int owner_rank = getOwnerRank(tag);
        bcast(result, owner_rank, m_exec_conf->getMPICommunicator());
        found = true;
        }
#endif
    assert(found);
    return result;
    }

//! Get the current image flags of a particle
int3 ParticleData::getImage(unsigned int tag) const
    {
    unsigned int idx = getRTag(tag);
    bool found = (idx < getN());
    int3 result = make_int3(0,0,0);
    if (found)
        {
        ArrayHandle< int3 > h_image(m_image, access_location::host, access_mode::read);
        result = make_int3(h_image.data[idx].x, h_image.data[idx].y, h_image.data[idx].z);
        }
#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        unsigned int owner_rank = getOwnerRank(tag);
        bcast(result, owner_rank, m_exec_conf->getMPICommunicator());
        found = true;
        }
#endif
    assert(found);
    return result;
    }

//! Get the current charge of a particle
Scalar ParticleData::getCharge(unsigned int tag) const
    {
    unsigned int idx = getRTag(tag);
    bool found = (idx < getN());
    Scalar result = 0.0;
    if (found)
        {
        ArrayHandle< Scalar > h_charge(m_charge, access_location::host, access_mode::read);
        result = h_charge.data[idx];
        }
#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        unsigned int owner_rank = getOwnerRank(tag);
        bcast(result, owner_rank, m_exec_conf->getMPICommunicator());
        found = true;
        }
#endif
    assert(found);
    return result;
    }

//! Get the current mass of a particle
Scalar ParticleData::getMass(unsigned int tag) const
    {
    unsigned int idx = getRTag(tag);
    bool found = (idx < getN());
    Scalar result = 0.0;
    if (found)
        {
        ArrayHandle< Scalar4 > h_vel(m_vel, access_location::host, access_mode::read);
        result = h_vel.data[idx].w;
        }
#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        unsigned int owner_rank = getOwnerRank(tag);
        bcast(result, owner_rank, m_exec_conf->getMPICommunicator());
        found = true;
        }
#endif
    assert(found);
    return result;
    }

//! Get the current diameter of a particle
Scalar ParticleData::getDiameter(unsigned int tag) const
    {
    unsigned int idx = getRTag(tag);
    bool found = (idx < getN());
    Scalar result = 0.0;
    if (found)
        {
        ArrayHandle< Scalar > h_diameter(m_diameter, access_location::host, access_mode::read);
        result = h_diameter.data[idx];
        }
#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        unsigned int owner_rank = getOwnerRank(tag);
        bcast(result, owner_rank, m_exec_conf->getMPICommunicator());
        found = true;
        }
#endif
    assert(found);
    return result;
    }

//! Get the body id of a particle
unsigned int ParticleData::getBody(unsigned int tag) const
    {
    unsigned int idx = getRTag(tag);
    bool found = (idx < getN());
    unsigned int result = 0;
    if (found)
        {
        ArrayHandle< unsigned int > h_body(m_body, access_location::host, access_mode::read);
        result = h_body.data[idx];
        }
#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        unsigned int owner_rank = getOwnerRank(tag);
        bcast(result, owner_rank, m_exec_conf->getMPICommunicator());
        found = true;
        }
#endif
    assert(found);
    return result;
    }

//! Get the current type of a particle
unsigned int ParticleData::getType(unsigned int tag) const
    {
    unsigned int idx = getRTag(tag);
    bool found = (idx < getN());
    unsigned int result = 0;
    if (found)
        {
        ArrayHandle< Scalar4 > h_pos(m_pos, access_location::host, access_mode::read);
        result = __scalar_as_int(h_pos.data[idx].w);
        }
#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        unsigned int owner_rank = getOwnerRank(tag);
        bcast(result, owner_rank, m_exec_conf->getMPICommunicator());
        found = true;
        }
#endif
    assert(found);
    return result;
    }

//! Get the orientation of a particle with a given tag
Scalar4 ParticleData::getOrientation(unsigned int tag) const
    {
    unsigned int idx = getRTag(tag);
    bool found = (idx < getN());
    Scalar4 result = make_scalar4(0.0,0.0,0.0,0.0);
    if (found)
        {
        ArrayHandle< Scalar4 > h_orientation(m_orientation, access_location::host, access_mode::read);
        result = h_orientation.data[idx];
        }
#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        unsigned int owner_rank = getOwnerRank(tag);
        bcast(result, owner_rank, m_exec_conf->getMPICommunicator());
        found = true;
        }
#endif
    assert(found);
    return result;
    }

//! Get the angular momentum of a particle with a given tag
Scalar4 ParticleData::getAngularMomentum(unsigned int tag) const
    {
    unsigned int idx = getRTag(tag);
    bool found = (idx < getN());
    Scalar4 result = make_scalar4(0.0,0.0,0.0,0.0);
    if (found)
        {
        ArrayHandle< Scalar4 > h_angmom(m_angmom, access_location::host, access_mode::read);
        result = h_angmom.data[idx];
        }
#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        unsigned int owner_rank = getOwnerRank(tag);
        bcast(result, owner_rank, m_exec_conf->getMPICommunicator());
        found = true;
        }
#endif
    assert(found);
    return result;
    }

//! Get the moment of inertia of a particle with a given tag
Scalar3 ParticleData::getMomentsOfInertia(unsigned int tag) const
    {
    unsigned int idx = getRTag(tag);
    bool found = (idx < getN());
    Scalar3 result = make_scalar3(0.0,0.0,0.0);
    if (found)
        {
        ArrayHandle< Scalar3 > h_inertia(m_inertia, access_location::host, access_mode::read);
        result = h_inertia.data[idx];
        }
#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        unsigned int owner_rank = getOwnerRank(tag);
        bcast(result, owner_rank, m_exec_conf->getMPICommunicator());
        found = true;
        }
#endif
    assert(found);
    return result;
    }

//! Get the net force / energy on a given particle
Scalar4 ParticleData::getPNetForce(unsigned int tag) const
    {
    unsigned int idx = getRTag(tag);
    bool found = (idx < getN());
    Scalar4 result = make_scalar4(0.0,0.0,0.0,0.0);
    if (found)
        {
        ArrayHandle< Scalar4 > h_net_force(m_net_force, access_location::host, access_mode::read);
        result = h_net_force.data[idx];
        }
#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        unsigned int owner_rank = getOwnerRank(tag);
        bcast(result, owner_rank, m_exec_conf->getMPICommunicator());
        found = true;
        }
#endif
    assert(found);
    return result;

    }

//! Get the net torque a given particle
Scalar4 ParticleData::getNetTorque(unsigned int tag) const
    {
    unsigned int idx = getRTag(tag);
    bool found = (idx < getN());
    Scalar4 result = make_scalar4(0.0,0.0,0.0,0.0);
    if (found)
        {
        ArrayHandle< Scalar4 > h_net_torque(m_net_torque, access_location::host, access_mode::read);
        result = h_net_torque.data[idx];
        }
#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        unsigned int owner_rank = getOwnerRank(tag);
        bcast(result, owner_rank, m_exec_conf->getMPICommunicator());
        found = true;
        }
#endif
    assert(found);
    return result;
}

//! Set the current position of a particle
/* \post In parallel simulations, the particle is moved to a new domain if necessary.
 * \warning Do not call during a simulation (method can overwrite ghost particle data)
 */
void ParticleData::setPosition(unsigned int tag, const Scalar3& pos, bool move)
    {
    unsigned int idx = getRTag(tag);
    bool ptl_local = (idx < getN());

    #ifdef ENABLE_MPI
    // get the owner rank
    unsigned int owner_rank = 0;
    if (m_decomposition) owner_rank = getOwnerRank(tag);
    #endif

    if (ptl_local)
        {
        ArrayHandle< Scalar4 > h_pos(m_pos, access_location::host, access_mode::readwrite);
        h_pos.data[idx].x = pos.x; h_pos.data[idx].y = pos.y; h_pos.data[idx].z = pos.z;
        }

    #ifdef ENABLE_MPI
    if (m_decomposition && move)
        {
        /*
         * migrate particle if necessary
         */
        unsigned my_rank = m_exec_conf->getRank();

        assert(!ptl_local || owner_rank == my_rank);

        // get rank where the particle should be according to new position
        unsigned int new_rank = m_decomposition->placeParticle(m_global_box, pos);

        // should the particle migrate?
        if (new_rank != owner_rank)
            {
            m_exec_conf->msg->notice(6) << "Moving particle " << tag << " from rank " << owner_rank << " to " << new_rank << std::endl;

            if (ptl_local)
                {
                    {
                    // mark for sending
                    ArrayHandle<unsigned int> h_comm_flag(getCommFlags(), access_location::host, access_mode::readwrite);
                    h_comm_flag.data[idx] = 1;
                    }

                std::vector<pdata_element> buf;

                // retrieve particle data
                std::vector<unsigned int> comm_flags; // not used here
                removeParticles(buf,comm_flags);

                assert(buf.size() >= 1);

                // check for particle data consistency
                if (buf.size() != 1)
                    {
                    m_exec_conf->msg->error() << "More than one (" << buf.size() << ") particle marked for sending." << endl << endl;
                    throw std::runtime_error("Error moving particle.");
                    }

                MPI_Request req;
                MPI_Status stat;

                // send particle data to new domain
                MPI_Isend(&buf.front(),
                    sizeof(pdata_element),
                    MPI_BYTE,
                    new_rank,
                    0,
                    m_exec_conf->getMPICommunicator(),
                    &req);
                MPI_Waitall(1,&req,&stat);
                }
            else if (new_rank == my_rank)
                {
                std::vector<pdata_element> buf(1);

                MPI_Request req;
                MPI_Status stat;

                // receive particle data
                MPI_Irecv(&buf.front(),
                    sizeof(pdata_element),
                    MPI_BYTE,
                    owner_rank,
                    0,
                    m_exec_conf->getMPICommunicator(),
                    &req);
                MPI_Waitall(1, &req, &stat);

                // add particle back to local data
                addParticles(buf);
                }

            // Notify observers
            m_ptl_move_signal(tag, owner_rank, new_rank);
            }
        }
    #endif // ENABLE_MPI
    }

//! Set the current velocity of a particle
void ParticleData::setVelocity(unsigned int tag, const Scalar3& vel)
    {
    unsigned int idx = getRTag(tag);
    bool found = (idx < getN());

#ifdef ENABLE_MPI
    // make sure the particle is somewhere
    if (m_decomposition)
        getOwnerRank(tag);
#endif
    if (found)
        {
        ArrayHandle< Scalar4 > h_vel(m_vel, access_location::host, access_mode::readwrite);
        h_vel.data[idx].x = vel.x; h_vel.data[idx].y = vel.y; h_vel.data[idx].z = vel.z;
        }
    }

//! Set the current image flags of a particle
void ParticleData::setImage(unsigned int tag, const int3& image)
    {
    unsigned int idx = getRTag(tag);
    bool found = (idx < getN());

#ifdef ENABLE_MPI
    // make sure the particle is somewhere
    if (m_decomposition)
        getOwnerRank(tag);
#endif
    if (found)
        {
        ArrayHandle< int3 > h_image(m_image, access_location::host, access_mode::readwrite);
        h_image.data[idx].x = image.x; h_image.data[idx].y = image.y; h_image.data[idx].z = image.z;
        }
    }

//! Set the current charge of a particle
void ParticleData::setCharge(unsigned int tag, Scalar charge)
    {
    unsigned int idx = getRTag(tag);
    bool found = (idx < getN());

#ifdef ENABLE_MPI
    // make sure the particle is somewhere
    if (m_decomposition)
        getOwnerRank(tag);
#endif
    if (found)
        {
        ArrayHandle< Scalar > h_charge(m_charge, access_location::host, access_mode::readwrite);
        h_charge.data[idx] = charge;
        }
    }

//! Set the current mass of a particle
void ParticleData::setMass(unsigned int tag, Scalar mass)
    {
    unsigned int idx = getRTag(tag);
    bool found = (idx < getN());

#ifdef ENABLE_MPI
    // make sure the particle is somewhere
    if (m_decomposition)
        getOwnerRank(tag);
#endif
    if (found)
        {
        ArrayHandle< Scalar4 > h_vel(m_vel, access_location::host, access_mode::readwrite);
        h_vel.data[idx].w = mass;
        }
    }


//! Set the current diameter of a particle
void ParticleData::setDiameter(unsigned int tag, Scalar diameter)
    {
    unsigned int idx = getRTag(tag);
    bool found = (idx < getN());

#ifdef ENABLE_MPI
    // make sure the particle is somewhere
    if (m_decomposition)
        getOwnerRank(tag);
#endif
    if (found)
        {
        ArrayHandle< Scalar > h_diameter(m_diameter, access_location::host, access_mode::readwrite);
        h_diameter.data[idx] = diameter;
        }
    }

//! Set the body id of a particle
void ParticleData::setBody(unsigned int tag, int body)
    {
    unsigned int idx = getRTag(tag);
    bool found = (idx < getN());

#ifdef ENABLE_MPI
    // make sure the particle is somewhere
    if (m_decomposition)
        getOwnerRank(tag);
#endif
    if (found)
        {
        ArrayHandle< unsigned int > h_body(m_body, access_location::host, access_mode::readwrite);
        h_body.data[idx] = body;
        }
    }

//! Set the current type of a particle
void ParticleData::setType(unsigned int tag, unsigned int typ)
    {
    assert(typ < getNTypes());
    unsigned int idx = getRTag(tag);
    bool found = (idx < getN());

#ifdef ENABLE_MPI
    // make sure the particle is somewhere
    if (m_decomposition)
        getOwnerRank(tag);
#endif
    if (found)
        {
        ArrayHandle< Scalar4 > h_pos(m_pos, access_location::host, access_mode::readwrite);
        h_pos.data[idx].w = __int_as_scalar(typ);
        }
    }

//! Set the orientation of a particle with a given tag
void ParticleData::setOrientation(unsigned int tag, const Scalar4& orientation)
    {
    unsigned int idx = getRTag(tag);
    bool found = (idx < getN());

#ifdef ENABLE_MPI
    // make sure the particle is somewhere
    if (m_decomposition)
        getOwnerRank(tag);
#endif
    if (found)
        {
        ArrayHandle< Scalar4 > h_orientation(m_orientation, access_location::host, access_mode::readwrite);
        h_orientation.data[idx] = orientation;
        }
    }

//! Set the angular momentum quaternion of a particle with a given tag
void ParticleData::setAngularMomentum(unsigned int tag, const Scalar4& angmom)
    {
    unsigned int idx = getRTag(tag);
    bool found = (idx < getN());

#ifdef ENABLE_MPI
    // make sure the particle is somewhere
    if (m_decomposition)
        getOwnerRank(tag);
#endif
    if (found)
        {
        ArrayHandle< Scalar4 > h_angmom(m_angmom, access_location::host, access_mode::readwrite);
        h_angmom.data[idx] = angmom;
        }
    }

//! Set the angular momentum quaternion of a particle with a given tag
void ParticleData::setMomentsOfInertia(unsigned int tag, const Scalar3& inertia)
    {
    unsigned int idx = getRTag(tag);
    bool found = (idx < getN());

#ifdef ENABLE_MPI
    // make sure the particle is somewhere
    if (m_decomposition)
        getOwnerRank(tag);
#endif
    if (found)
        {
        ArrayHandle< Scalar3 > h_inertia(m_inertia, access_location::host, access_mode::readwrite);
        h_inertia.data[idx] = inertia;
        }
    }

void export_BoxDim()
    {
    class_<BoxDim>("BoxDim")
    .def(init<Scalar>())
    .def(init<Scalar, Scalar, Scalar>())
    .def(init<Scalar3>())
    .def(init<Scalar3, Scalar3, uchar3>())
    .def(init<Scalar, Scalar, Scalar, Scalar>())
    .def("getPeriodic", &BoxDim::getPeriodic)
    .def("setPeriodic", &BoxDim::setPeriodic)
    .def("getL", &BoxDim::getL)
    .def("setL", &BoxDim::setL)
    .def("getLo", &BoxDim::getLo)
    .def("getHi", &BoxDim::getHi)
    .def("setLoHi", &BoxDim::setLoHi)
    .def("getTiltFactorXY", &BoxDim::getTiltFactorXY)
    .def("getTiltFactorXZ", &BoxDim::getTiltFactorXZ)
    .def("getTiltFactorYZ", &BoxDim::getTiltFactorYZ)
    .def("makeFraction", &BoxDim::makeFraction)
    .def("minImage", &BoxDim::minImage)
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
    .def(init<unsigned int, const BoxDim&, unsigned int, boost::shared_ptr<ExecutionConfiguration>, boost::shared_ptr<DomainDecomposition> >())
    .def(init<const SnapshotParticleData&, const BoxDim&, boost::shared_ptr<ExecutionConfiguration> >())
    .def(init<const SnapshotParticleData&, const BoxDim&, boost::shared_ptr<ExecutionConfiguration>, boost::shared_ptr<DomainDecomposition> >())
    .def("getGlobalBox", &ParticleData::getGlobalBox, return_value_policy<copy_const_reference>())
    .def("getBox", &ParticleData::getBox, return_value_policy<copy_const_reference>())
    .def("setGlobalBoxL", &ParticleData::setGlobalBoxL)
    .def("setGlobalBox", &ParticleData::setGlobalBox)
    .def("getN", &ParticleData::getN)
    .def("getNGhosts", &ParticleData::getNGhosts)
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
    .def("getAngularMomentum", &ParticleData::getAngularMomentum)
    .def("getPNetForce", &ParticleData::getPNetForce)
    .def("getNetTorque", &ParticleData::getNetTorque)
    .def("getMomentsOfInertia", &ParticleData::getMomentsOfInertia)
    .def("setPosition", &ParticleData::setPosition)
    .def("setVelocity", &ParticleData::setVelocity)
    .def("setImage", &ParticleData::setImage)
    .def("setCharge", &ParticleData::setCharge)
    .def("setMass", &ParticleData::setMass)
    .def("setDiameter", &ParticleData::setDiameter)
    .def("setBody", &ParticleData::setBody)
    .def("setType", &ParticleData::setType)
    .def("setOrientation", &ParticleData::setOrientation)
    .def("setAngularMomentum", &ParticleData::setAngularMomentum)
    .def("setMomentsOfInertia", &ParticleData::setMomentsOfInertia)
    .def("takeSnapshot", &ParticleData::takeSnapshot)
    .def("initializeFromSnapshot", &ParticleData::initializeFromSnapshot)
#ifdef ENABLE_MPI
    .def("setDomainDecomposition", &ParticleData::setDomainDecomposition)
    .def("getDomainDecomposition", &ParticleData::getDomainDecomposition)
#endif
    ;
    }

//! Constructor for SnapshotParticleData
SnapshotParticleData::SnapshotParticleData(unsigned int N)
       : size(N)
    {
    resize(N);
    }

void SnapshotParticleData::resize(unsigned int N)
    {
    pos.resize(N,make_scalar3(0.0,0.0,0.0));
    vel.resize(N,make_scalar3(0.0,0.0,0.0));
    accel.resize(N,make_scalar3(0.0,0.0,0.0));
    type.resize(N,0);
    mass.resize(N,Scalar(1.0));
    charge.resize(N,Scalar(0.0));
    diameter.resize(N,Scalar(1.0));
    image.resize(N,make_int3(0,0,0));
    body.resize(N,NO_BODY);
    orientation.resize(N,make_scalar4(1.0,0.0,0.0,0.0));
    angmom.resize(N,make_scalar4(1.0,0.0,0.0,0.0));
    inertia.resize(N,make_scalar3(0.0,0.0,0.0));
    size = N;
    }

bool SnapshotParticleData::validate() const
    {
    // Check that a type mapping exists
    if (type_mapping.size() == 0) return false;

    // Check if all other fields are of equal length==size
    if (pos.size() != size || vel.size() != size || accel.size() != size || type.size() != size ||
        mass.size() != size || charge.size() != size || diameter.size() != size ||
        image.size() != size || body.size() != size || orientation.size() != size || angmom.size() != size ||
        inertia.size() != size)
        return false;

    return true;
    }

#ifdef ENABLE_MPI
//! Select non-zero communication lags
struct comm_flag_select : std::unary_function<const unsigned int, bool>
    {
    bool operator() (const unsigned int comm_flag) const
        {
        return comm_flag;
        }
    };

/*! \note This method may only be used during communication or when
 *        no ghost particles are present, because ghost particle values
 *        are undefined after calling this method.
 */
void ParticleData::removeParticles(std::vector<pdata_element>& out, std::vector<unsigned int>& comm_flags)
    {
    if (m_prof) m_prof->push("pack");

    unsigned int num_remove_ptls = 0;

        {
        // access particle data tags and rtags
        ArrayHandle<unsigned int> h_tag(getTags(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_rtag(getRTags(), access_location::host, access_mode::readwrite);
        ArrayHandle<unsigned int> h_comm_flags(getCommFlags(), access_location::host, access_mode::read);

        // set all rtags of ptls with comm_flag != 0 to NOT_LOCAL and count removed particles
        unsigned int N = getN();
        for (unsigned int i = 0; i < N; ++i)
            if (h_comm_flags.data[i])
                {
                unsigned int tag = h_tag.data[i];
                assert(tag < getNGlobal());
                h_rtag.data[tag] = NOT_LOCAL;
                num_remove_ptls++;
                }
        }

    unsigned int old_nparticles = getN();
    unsigned int new_nparticles = m_nparticles - num_remove_ptls;

    // resize output buffers
    out.resize(num_remove_ptls);
    comm_flags.resize(num_remove_ptls);

    // resize particle data using amortized O(1) array resizing
    resize(new_nparticles);

        {
        // access particle data arrays
        ArrayHandle<Scalar4> h_pos(getPositions(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> h_vel(getVelocities(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar3> h_accel(getAccelerations(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> h_charge(getCharges(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> h_diameter(getDiameters(), access_location::host, access_mode::readwrite);
        ArrayHandle<int3> h_image(getImages(), access_location::host, access_mode::readwrite);
        ArrayHandle<unsigned int> h_body(getBodies(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> h_orientation(getOrientationArray(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> h_angmom(getAngularMomentumArray(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar3> h_inertia(getMomentsOfInertiaArray(), access_location::host, access_mode::readwrite);
        ArrayHandle<unsigned int> h_tag(getTags(), access_location::host, access_mode::readwrite);

        ArrayHandle<unsigned int> h_rtag(getRTags(), access_location::host, access_mode::read);

        ArrayHandle<unsigned int> h_comm_flags(getCommFlags(), access_location::host, access_mode::readwrite);

        ArrayHandle<Scalar4> h_pos_alt(m_pos_alt, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar4> h_vel_alt(m_vel_alt, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar3> h_accel_alt(m_accel_alt, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar> h_charge_alt(m_charge_alt, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar> h_diameter_alt(m_diameter_alt, access_location::host, access_mode::overwrite);
        ArrayHandle<int3> h_image_alt(m_image_alt, access_location::host, access_mode::overwrite);
        ArrayHandle<unsigned int> h_body_alt(m_body_alt, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar4> h_orientation_alt(m_orientation_alt, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar4> h_angmom_alt(m_angmom_alt, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar3> h_inertia_alt(m_inertia_alt, access_location::host, access_mode::overwrite);
        ArrayHandle<unsigned int> h_tag_alt(m_tag_alt, access_location::host, access_mode::overwrite);

        unsigned int n =0;
        unsigned int m = 0;
        for (unsigned int i = 0; i < old_nparticles; ++i)
            {
            unsigned int tag = h_tag.data[i];
            if (h_rtag.data[tag] != NOT_LOCAL)
                {
                // copy over to alternate pdata arrays
                h_pos_alt.data[n] = h_pos.data[i];
                h_vel_alt.data[n] = h_vel.data[i];
                h_accel_alt.data[n] = h_accel.data[i];
                h_charge_alt.data[n] = h_charge.data[i];
                h_diameter_alt.data[n] = h_diameter.data[i];
                h_image_alt.data[n] = h_image.data[i];
                h_body_alt.data[n] = h_body.data[i];
                h_orientation_alt.data[n] = h_orientation.data[i];
                h_angmom_alt.data[n] = h_angmom.data[i];
                h_inertia_alt.data[n] = h_inertia.data[i];
                h_tag_alt.data[n] = h_tag.data[i];
                ++n;
                }
            else
                {
                // write to packed array
                pdata_element p;
                p.pos = h_pos.data[i];
                p.vel = h_vel.data[i];
                p.accel = h_accel.data[i];
                p.charge = h_charge.data[i];
                p.diameter = h_diameter.data[i];
                p.image = h_image.data[i];
                p.body = h_body.data[i];
                p.orientation = h_orientation.data[i];
                p.angmom = h_angmom.data[i];
                p.inertia = h_inertia.data[i];
                p.tag = h_tag.data[i];
                out[m++] = p;
                }
            }

        // write out non-zero communication flags
        std::remove_copy_if(h_comm_flags.data, h_comm_flags.data + old_nparticles, comm_flags.begin(),
            std::not1(comm_flag_select()));

        // reset communication flags to zero
        std::fill(h_comm_flags.data, h_comm_flags.data + new_nparticles, 0);
        }

    // swap particle data arrays
    swapPositions();
    swapVelocities();
    swapAccelerations();
    swapCharges();
    swapDiameters();
    swapImages();
    swapBodies();
    swapOrientations();
    swapAngularMomenta();
    swapTags();

        {
        ArrayHandle<unsigned int> h_rtag(getRTags(), access_location::host, access_mode::readwrite);
        ArrayHandle<unsigned int> h_tag(getTags(), access_location::host, access_mode::read);

        // recompute rtags (particles have moved)
        for (unsigned int idx = 0; idx < m_nparticles; ++idx)
            {
            // reset rtag of this ptl
            unsigned int tag = h_tag.data[idx];
            assert(tag < getNGlobal());
            h_rtag.data[tag] = idx;
            }
        }

    if (m_prof) m_prof->pop();

    // notify subscribers that particle data order has been changed
    notifyParticleSort();
    }

//! Remove particles from local domain and append new particle data
void ParticleData::addParticles(const std::vector<pdata_element>& in)
    {
    if (m_prof) m_prof->push("unpack");

    unsigned int num_add_ptls = in.size();

    unsigned int old_nparticles = getN();
    unsigned int new_nparticles = m_nparticles + num_add_ptls;

    // resize particle data using amortized O(1) array resizing
    resize(new_nparticles);

        {
        // access particle data arrays
        ArrayHandle<Scalar4> h_pos(getPositions(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> h_vel(getVelocities(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar3> h_accel(getAccelerations(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> h_charge(getCharges(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> h_diameter(getDiameters(), access_location::host, access_mode::readwrite);
        ArrayHandle<int3> h_image(getImages(), access_location::host, access_mode::readwrite);
        ArrayHandle<unsigned int> h_body(getBodies(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> h_orientation(getOrientationArray(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> h_angmom(getAngularMomentumArray(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar3> h_inertia(getMomentsOfInertiaArray(), access_location::host, access_mode::readwrite);
        ArrayHandle<unsigned int> h_tag(getTags(), access_location::host, access_mode::readwrite);
        ArrayHandle<unsigned int> h_rtag(getRTags(), access_location::host, access_mode::readwrite);
        ArrayHandle<unsigned int> h_comm_flags(m_comm_flags, access_location::host, access_mode::readwrite);

        // add new particles at the end
        unsigned int n = old_nparticles;
        for (std::vector<pdata_element>::const_iterator it = in.begin(); it != in.end(); ++it)
            {
            pdata_element p = *it;
            h_pos.data[n] = p.pos;
            h_vel.data[n] = p.vel;
            h_accel.data[n] = p.accel;
            h_charge.data[n] = p.charge;
            h_diameter.data[n] = p.diameter;
            h_image.data[n] = p.image;
            h_body.data[n] = p.body;
            h_orientation.data[n] = p.orientation;
            h_angmom.data[n] = p.angmom;
            h_inertia.data[n] = p.inertia;
            h_tag.data[n] = p.tag;
            n++;
            }

        // reset communication flags
        std::fill(h_comm_flags.data + old_nparticles, h_comm_flags.data + new_nparticles, 0);

        // recompute rtags
        for (unsigned int idx = 0; idx < m_nparticles; ++idx)
            {
            // reset rtag of this ptl
            unsigned int tag = h_tag.data[idx];
            assert(tag < getNGlobal());
            h_rtag.data[tag] = idx;
            }
        }

    if (m_prof) m_prof->pop();

    // notify subscribers that particle data order has been changed
    notifyParticleSort();
    }

#ifdef ENABLE_CUDA
//! Pack particle data into a buffer (GPU version)
/*! \note This method may only be used during communication or when
 *        no ghost particles are present, because ghost particle values
 *        are undefined after calling this method.
 */
void ParticleData::removeParticlesGPU(GPUVector<pdata_element>& out, GPUVector<unsigned int> &comm_flags)
    {
    if (m_prof) m_prof->push(m_exec_conf, "pack");

    // this is the maximum number of elements we can possibly write to out
    unsigned int max_n_out = out.getNumElements();
    if (comm_flags.getNumElements() < max_n_out)
        max_n_out = comm_flags.getNumElements();

    // allocate array if necessary
    if (! max_n_out)
        {
        out.resize(1);
        comm_flags.resize(1);
        max_n_out = out.getNumElements();
        if (comm_flags.getNumElements() < max_n_out) max_n_out = comm_flags.getNumElements();
        }

    // number of particles that are to be written out
    unsigned int n_out = 0;

    bool done = false;

    // copy without writing past the end of the output array, resizing it as needed
    while (! done)
        {
        // access particle data arrays to read from
        ArrayHandle<Scalar4> d_pos(getPositions(), access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_vel(getVelocities(), access_location::device, access_mode::read);
        ArrayHandle<Scalar3> d_accel(getAccelerations(), access_location::device, access_mode::read);
        ArrayHandle<Scalar> d_charge(getCharges(), access_location::device, access_mode::read);
        ArrayHandle<Scalar> d_diameter(getDiameters(), access_location::device, access_mode::read);
        ArrayHandle<int3> d_image(getImages(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_body(getBodies(), access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_orientation(getOrientationArray(), access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_angmom(getAngularMomentumArray(), access_location::device, access_mode::read);
        ArrayHandle<Scalar3> d_inertia(getMomentsOfInertiaArray(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_tag(getTags(), access_location::device, access_mode::read);

        // access alternate particle data arrays to write to
        ArrayHandle<Scalar4> d_pos_alt(m_pos_alt, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar4> d_vel_alt(m_vel_alt, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar3> d_accel_alt(m_accel_alt, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar> d_charge_alt(m_charge_alt, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar> d_diameter_alt(m_diameter_alt, access_location::device, access_mode::overwrite);
        ArrayHandle<int3> d_image_alt(m_image_alt, access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> d_body_alt(m_body_alt, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar4> d_orientation_alt(m_orientation_alt, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar4> d_angmom_alt(m_angmom_alt, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar3> d_inertia_alt(m_inertia_alt, access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> d_tag_alt(m_tag_alt, access_location::device, access_mode::overwrite);

        ArrayHandle<unsigned int> d_comm_flags(getCommFlags(), access_location::device, access_mode::readwrite);

        // Access reverse-lookup table
        ArrayHandle<unsigned int> d_rtag(getRTags(), access_location::device, access_mode::readwrite);

            {
            // Access output array
            ArrayHandle<pdata_element> d_out(out, access_location::device, access_mode::overwrite);
            ArrayHandle<unsigned int> d_comm_flags_out(comm_flags, access_location::device, access_mode::overwrite);

            // get temporary buffer
            ScopedAllocation<unsigned int> d_tmp(m_exec_conf->getCachedAllocator(), getN());

            n_out = gpu_pdata_remove(getN(),
                           d_pos.data,
                           d_vel.data,
                           d_accel.data,
                           d_charge.data,
                           d_diameter.data,
                           d_image.data,
                           d_body.data,
                           d_orientation.data,
                           d_angmom.data,
                           d_inertia.data,
                           d_tag.data,
                           d_rtag.data,
                           d_pos_alt.data,
                           d_vel_alt.data,
                           d_accel_alt.data,
                           d_charge_alt.data,
                           d_diameter_alt.data,
                           d_image_alt.data,
                           d_body_alt.data,
                           d_orientation_alt.data,
                           d_angmom_alt.data,
                           d_inertia_alt.data,
                           d_tag_alt.data,
                           d_out.data,
                           d_comm_flags.data,
                           d_comm_flags_out.data,
                           max_n_out,
                           d_tmp.data,
                           m_mgpu_context);
           }
        if (m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();

        // resize output vector
        out.resize(n_out);
        comm_flags.resize(n_out);

        // was the array large enough?
        if (n_out <= max_n_out) done = true;

        max_n_out = out.getNumElements();
        if (comm_flags.getNumElements() < max_n_out) max_n_out = comm_flags.getNumElements();
        }

    // update particle number (no need to shrink arrays)
    m_nparticles -= n_out;

    // swap particle data arrays
    swapPositions();
    swapVelocities();
    swapAccelerations();
    swapCharges();
    swapDiameters();
    swapImages();
    swapBodies();
    swapOrientations();
    swapAngularMomenta();
    swapTags();

    // notify subscribers
    notifyParticleSort();

    if (m_prof) m_prof->pop(m_exec_conf);
    }

//! Add new particle data (GPU version)
void ParticleData::addParticlesGPU(const GPUVector<pdata_element>& in)
    {
    if (m_prof) m_prof->push(m_exec_conf, "unpack");

    unsigned int old_nparticles = getN();
    unsigned int num_add_ptls = in.size();
    unsigned int new_nparticles = old_nparticles + num_add_ptls;

    // amortized resizing of particle data
    resize(new_nparticles);

        {
        // access particle data arrays
        ArrayHandle<Scalar4> d_pos(getPositions(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> d_vel(getVelocities(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar3> d_accel(getAccelerations(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar> d_charge(getCharges(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar> d_diameter(getDiameters(), access_location::device, access_mode::readwrite);
        ArrayHandle<int3> d_image(getImages(), access_location::device, access_mode::readwrite);
        ArrayHandle<unsigned int> d_body(getBodies(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> d_orientation(getOrientationArray(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> d_angmom(getAngularMomentumArray(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar3> d_inertia(getMomentsOfInertiaArray(), access_location::device, access_mode::readwrite);
        ArrayHandle<unsigned int> d_tag(getTags(), access_location::device, access_mode::readwrite);
        ArrayHandle<unsigned int> d_rtag(getRTags(), access_location::device, access_mode::readwrite);
        ArrayHandle<unsigned int> d_comm_flags(getCommFlags(), access_location::device, access_mode::readwrite);

        // Access input array
        ArrayHandle<pdata_element> d_in(in, access_location::device, access_mode::read);

        // add new particles on GPU
        gpu_pdata_add_particles(
            old_nparticles,
            num_add_ptls,
            d_pos.data,
            d_vel.data,
            d_accel.data,
            d_charge.data,
            d_diameter.data,
            d_image.data,
            d_body.data,
            d_orientation.data,
            d_angmom.data,
            d_inertia.data,
            d_tag.data,
            d_rtag.data,
            d_in.data,
            d_comm_flags.data);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    // notify subscribers
    notifyParticleSort();

    if (m_prof) m_prof->pop(m_exec_conf);
    }

#endif // ENABLE_CUDA
#endif // ENABLE_MPI

void export_SnapshotParticleData()
    {
    class_<SnapshotParticleData, boost::shared_ptr<SnapshotParticleData> >("SnapshotParticleData", init<unsigned int>())
    .def_readwrite("pos", &SnapshotParticleData::pos)
    .def_readwrite("vel", &SnapshotParticleData::vel)
    .def_readwrite("accel", &SnapshotParticleData::accel)
    .def_readwrite("type", &SnapshotParticleData::type)
    .def_readwrite("mass", &SnapshotParticleData::mass)
    .def_readwrite("charge", &SnapshotParticleData::charge)
    .def_readwrite("diameter", &SnapshotParticleData::diameter)
    .def_readwrite("image", &SnapshotParticleData::image)
    .def_readwrite("body", &SnapshotParticleData::body)
    .def_readwrite("type_mapping", &SnapshotParticleData::type_mapping)
    .def_readwrite("size", &SnapshotParticleData::size)
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif
