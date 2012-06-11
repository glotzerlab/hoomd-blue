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
#include "AngleData.h"
#include "DihedralData.h"

#ifdef ENABLE_MPI
#include "Communicator.h"

#include <boost/mpi.hpp>
#include <boost/serialization/map.hpp>
using namespace boost::mpi;

// Define some of our types as fixed-size MPI datatypes for performance optimization
BOOST_IS_MPI_DATATYPE(Scalar4)
BOOST_IS_MPI_DATATYPE(Scalar3)
BOOST_IS_MPI_DATATYPE(Scalar)
BOOST_IS_MPI_DATATYPE(uint3)
BOOST_IS_MPI_DATATYPE(int3)

#endif

#include <boost/bind.hpp>

using namespace boost::signals;
using namespace boost;

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
    m_exec_conf->msg->notice(5) << "Constructing ParticleData" << endl;

    // check the input for errors
    if (m_ntypes == 0)
        {
        m_exec_conf->msg->error() << "Number of particle types must be greater than 0." << endl;
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
    
    // initially, global box = local box
    m_global_box = box;
    }

/*! Calls the initializer's members to determine the number of particles, box size and then
    uses it to fill out the position and velocity data.
    \param init Initializer to use
    \param exec_conf Execution configuration to run on
*/
ParticleData::ParticleData(const ParticleDataInitializer& init, boost::shared_ptr<ExecutionConfiguration> exec_conf) : m_exec_conf(exec_conf), m_ntypes(0), m_nghosts(0), m_nglobal(0), m_resize_factor(9./8.)
    {
    m_exec_conf->msg->notice(5) << "Constructing ParticleData" << endl;

    // allocate memory
    allocate(init.getNumParticles());

    // default: number of global particles = number of local particles
    setNGlobal(getN());
    
        {
        ArrayHandle< Scalar4 > h_orientation(m_orientation, access_location::host, access_mode::readwrite);
        
        ArrayHandle< Scalar4 > h_vel(getVelocities(), access_location::host, access_mode::overwrite);
        ArrayHandle< Scalar > h_diameter(getDiameters(), access_location::host, access_mode::overwrite);
        ArrayHandle< unsigned int > h_global_tag(getGlobalTags(), access_location::host, access_mode::overwrite);
        ArrayHandle< unsigned int > h_global_rtag(getGlobalRTags(), access_location::host, access_mode::overwrite);
        ArrayHandle< unsigned int > h_body(getBodies(), access_location::host, access_mode::overwrite);

        // set default values
        // all values not explicitly set here have been initialized to zero upon allocation
        for (unsigned int i = 0; i < getN(); i++)
            {
            h_vel.data[i].w = 1.0; // mass

            h_diameter.data[i] = 1.0;
            
            h_body.data[i] = NO_BODY;
            h_global_tag.data[i] = i;
            h_global_rtag.data[i] = i;
            h_orientation.data[i] = make_scalar4(1.0, 0.0, 0.0, 0.0);
            }
        }

    // initialize box dimensions
    setGlobalBoxL(init.getBox().getL());

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

/*! \param L New box lengths to set
    \note ParticleData does NOT enforce any boundary conditions. When a new box is set,
        it is the responsibility of the caller to ensure that all particles lie within
        the new box.
*/
void ParticleData::setGlobalBoxL(const Scalar3 &L)
    {
    m_global_box.setL(L);

#ifdef ENABLE_MPI
    if (m_decomposition)
        m_box = m_decomposition->calculateLocalBox(m_global_box);
    else
#endif
        {
        // local box = global box
        m_box.setL(L);
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
    \a func will be called every time the the box size is changed via setGlobalBoxL()
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
          and notifyParticleSort() needs to be called separately after all particle data is available
          on the local processor.
*/
boost::signals::connection ParticleData::connectMaxParticleNumberChange(const boost::function<void ()> &func)
    {
    return m_max_particle_num_signal.connect(func);
    }

/*! \param func Function to be called when the number of ghost particles changes
    \return Connection to manage the signal
 */
boost::signals::connection ParticleData::connectGhostParticleNumberChange(const boost::function<void ()> &func)
    {
    return m_ghost_particle_num_signal.connect(func);
    }

/*! This function must be called any time the ghost particles are updated.
 */
void ParticleData::notifyGhostParticleNumberChange()
    {
    m_ghost_particle_num_signal();
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
    Scalar3 lo = m_box.getLo();
    Scalar3 hi = m_box.getHi();

    ArrayHandle<Scalar4> h_pos(getPositions(), access_location::host, access_mode::read);
    for (unsigned int i = 0; i < getN(); i++)
        {
        if (h_pos.data[i].x < lo.x-Scalar(1e-5) || h_pos.data[i].x > hi.x+Scalar(1e-5))
            {
            m_exec_conf->msg->notice(1) << "pos " << i << ":" << setprecision(12) << h_pos.data[i].x << " " << h_pos.data[i].y << " " << h_pos.data[i].z << endl;
            m_exec_conf->msg->notice(1) << "lo: " << lo.x << " " << lo.y << " " << lo.z << endl;
            m_exec_conf->msg->notice(1) << "hi: " << hi.x << " " << hi.y << " " << hi.z << endl;
            return false;
            }
        if (h_pos.data[i].y < lo.y-Scalar(1e-5) || h_pos.data[i].y > hi.y+Scalar(1e-5))
            {
            m_exec_conf->msg->notice(1) << "pos " << i << ":" << setprecision(12) << h_pos.data[i].x << " " << h_pos.data[i].y << " " << h_pos.data[i].z << endl;
            m_exec_conf->msg->notice(1) << "lo: " << lo.x << " " << lo.y << " " << lo.z << endl;
            m_exec_conf->msg->notice(1) << "hi: " << hi.x << " " << hi.y << " " << hi.z << endl;
            return false;
            }
        if (h_pos.data[i].z < lo.z-Scalar(1e-5) || h_pos.data[i].z > hi.z+Scalar(1e-5))
            {
            m_exec_conf->msg->notice(1) << "pos " << i << ":" << setprecision(12) << h_pos.data[i].x << " " << h_pos.data[i].y << " " << h_pos.data[i].z << endl;
            m_exec_conf->msg->notice(1) << "lo: " << lo.x << " " << lo.y << " " << lo.z << endl;
            m_exec_conf->msg->notice(1) << "hi: " << hi.x << " " << hi.y << " " << hi.z << endl;
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
    // check the input for errors
    if (snapshot.type_mapping.size() == 0)
        {
        m_exec_conf->msg->error() << "Number of particle types must be greater than 0." << endl;
        throw std::runtime_error("Error initializing ParticleData");
        }
        
#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        // gather box information from all processors
        unsigned int root = m_decomposition->getRoot();

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
        std::vector< std::vector<unsigned int > > global_tag_proc; // Global tags of every processor
        std::vector< unsigned int > N_proc;                        // Number of particles on every processor
 
        // resize to number of ranks in communicator
        boost::shared_ptr<boost::mpi::communicator> mpi_comm = m_decomposition->getMPICommunicator();
        unsigned int size = mpi_comm->size();
        unsigned int my_rank = mpi_comm->rank();

        pos_proc.resize(size);
        vel_proc.resize(size);
        accel_proc.resize(size);
        type_proc.resize(size);
        mass_proc.resize(size);
        charge_proc.resize(size);
        diameter_proc.resize(size);
        image_proc.resize(size);
        body_proc.resize(size);
        global_tag_proc.resize(size);
        N_proc.resize(size);

        if (my_rank == m_decomposition->getRoot())
            {
            Scalar3 scale = m_global_box.getL() / m_box.getL();
            const Index3D& di = m_decomposition->getDomainIndexer();

            // loop over particles in snapshot, place them into domains
            for (std::vector<Scalar3>::const_iterator it=snapshot.pos.begin(); it != snapshot.pos.end(); it++)
                {

                // determine domain the particle is placed into
                Scalar3 f = m_global_box.makeFraction(*it);
                int i= (unsigned int) (f.x * scale.x);
                int j= (unsigned int) (f.y * scale.y);
                int k= (unsigned int) (f.z * scale.z);

                // treat particles lying exactly on the boundary
                if (i == (int) di.getW()) 
                    i--;
                   
                if (j == (int) di.getH())
                    j--;

                if (k == (int) di.getD())
                    k--;
            
                unsigned int rank = di(i,j,k);

                unsigned int tag = it - snapshot.pos.begin() ;
                // fill up per-processor data structures
                pos_proc[rank].push_back(snapshot.pos[tag]);
                vel_proc[rank].push_back(snapshot.vel[tag]);
                accel_proc[rank].push_back(snapshot.accel[tag]);
                type_proc[rank].push_back(snapshot.type[tag]);
                mass_proc[rank].push_back(snapshot.mass[tag]);
                charge_proc[rank].push_back(snapshot.charge[tag]);
                diameter_proc[rank].push_back(snapshot.diameter[tag]);
                image_proc[rank].push_back(snapshot.image[tag]);
                body_proc[rank].push_back(snapshot.body[tag]);
                global_tag_proc[rank].push_back(tag);
                N_proc[rank]++;
                }

            }

        // get number of particle types
        m_ntypes = snapshot.type_mapping.size();

        // get type mapping
        m_type_mapping = snapshot.type_mapping;

        // broadcast number of particle types
        boost::mpi::broadcast(*mpi_comm, m_ntypes, root);

        // broadcast type mapping
        boost::mpi::broadcast(*mpi_comm, m_type_mapping, root);

        // broadcast global number of particles
        unsigned int nglobal = snapshot.size;
        boost::mpi::broadcast(*mpi_comm, nglobal, root);

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
        std::vector<unsigned int> global_tag;
 
        // distribute particle data
        boost::mpi::scatter(*mpi_comm, pos_proc,pos,root);
        boost::mpi::scatter(*mpi_comm, vel_proc,vel,root);
        boost::mpi::scatter(*mpi_comm, accel_proc, accel, root);
        boost::mpi::scatter(*mpi_comm, type_proc, type, root);
        boost::mpi::scatter(*mpi_comm, mass_proc, mass, root);
        boost::mpi::scatter(*mpi_comm, charge_proc, charge, root);
        boost::mpi::scatter(*mpi_comm, diameter_proc, diameter, root);
        boost::mpi::scatter(*mpi_comm, image_proc, image, root);
        boost::mpi::scatter(*mpi_comm, body_proc, body, root);
        boost::mpi::scatter(*mpi_comm, global_tag_proc, global_tag, root);

        // distribute number of particles
        boost::mpi::scatter(*mpi_comm, N_proc, m_nparticles, root);

        // reset all reverse lookup tags to NOT_LOCAL flag
            {
            ArrayHandle<unsigned int> h_global_rtag(getGlobalRTags(), access_location::host, access_mode::overwrite);
            for (unsigned int tag = 0; tag < m_nglobal; tag++)
                h_global_rtag.data[tag] = NOT_LOCAL;
            }

        // reallocate particle data such that we can accomodate the particles (only if necessary)
        /* Note: this reallocates also if m_max_nparticles > m_nparticles, which means
                 unnecessary overhead in a MPI simulation. But currently, the system
                 is first initialized with the global number of particles
                 and this number is reduced to the local number of particles, here, to
                 reduce memory footprint. So until the initialization changes,
                 we will reallocate with the current number of particles.
        */
        if (m_max_nparticles != m_nparticles)
            reallocate(m_nparticles);

        // Load particle data
        ArrayHandle< Scalar4 > h_pos(m_pos, access_location::host, access_mode::overwrite);
        ArrayHandle< Scalar4 > h_vel(m_vel, access_location::host, access_mode::overwrite);
        ArrayHandle< Scalar3 > h_accel(m_accel, access_location::host, access_mode::overwrite);
        ArrayHandle< int3 > h_image(m_image, access_location::host, access_mode::overwrite);
        ArrayHandle< Scalar > h_charge(m_charge, access_location::host, access_mode::overwrite);
        ArrayHandle< Scalar > h_diameter(m_diameter, access_location::host, access_mode::overwrite);
        ArrayHandle< unsigned int > h_body(m_body, access_location::host, access_mode::overwrite);
        ArrayHandle< unsigned int > h_global_tag(m_global_tag, access_location::host, access_mode::overwrite);
        ArrayHandle< unsigned int > h_global_rtag(m_global_rtag, access_location::host, access_mode::readwrite);

        for (unsigned int idx = 0; idx < m_nparticles; idx++)
            {
            h_pos.data[idx] = make_scalar4(pos[idx].x,pos[idx].y, pos[idx].z, __int_as_scalar(type[idx]));
            h_vel.data[idx] = make_scalar4(vel[idx].x, vel[idx].y, vel[idx].z, mass[idx]);
            h_accel.data[idx] = accel[idx];
            h_charge.data[idx] = snapshot.charge[idx];
            h_diameter.data[idx] = snapshot.diameter[idx];
            h_image.data[idx] = snapshot.image[idx];
            h_global_tag.data[idx] = global_tag[idx];
            h_global_rtag.data[global_tag[idx]] = idx;
            h_body.data[idx] = snapshot.body[idx];
            }

        // reset ghost particle number
        m_nghosts = 0;

        // notify about change in ghost particle number
        notifyGhostParticleNumberChange();
        }
    else
#endif
        {
        // Initialize number of particles
        setNGlobal(snapshot.size);
        m_nparticles = snapshot.size;

        // reallocate particle data such that we can accomodate the particles
        reallocate(snapshot.size);

        ArrayHandle< Scalar4 > h_pos(m_pos, access_location::host, access_mode::overwrite);
        ArrayHandle< Scalar4 > h_vel(m_vel, access_location::host, access_mode::overwrite);
        ArrayHandle< Scalar3 > h_accel(m_accel, access_location::host, access_mode::overwrite);
        ArrayHandle< int3 > h_image(m_image, access_location::host, access_mode::overwrite);
        ArrayHandle< Scalar > h_charge(m_charge, access_location::host, access_mode::overwrite);
        ArrayHandle< Scalar > h_diameter(m_diameter, access_location::host, access_mode::overwrite);
        ArrayHandle< unsigned int > h_body(m_body, access_location::host, access_mode::overwrite);
        ArrayHandle< unsigned int > h_global_tag(m_global_tag, access_location::host, access_mode::overwrite);
        ArrayHandle< unsigned int > h_global_rtag(m_global_rtag, access_location::host, access_mode::readwrite);

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
            h_global_tag.data[tag] = tag;
            h_global_rtag.data[tag] = tag;
            h_body.data[tag] = snapshot.body[tag];
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
=======

    for (unsigned int i = 0; i < m_nparticles; i++)
        {
        // particle index in sorted order
        h_pos.data[i].x = snapshot.pos[i].x;
        h_pos.data[i].y = snapshot.pos[i].y;
        h_pos.data[i].z = snapshot.pos[i].z;
        h_pos.data[i].w = __int_as_scalar(snapshot.type[i]);

        h_vel.data[i].x = snapshot.vel[i].x;
        h_vel.data[i].y = snapshot.vel[i].y;
        h_vel.data[i].z = snapshot.vel[i].z;
        h_vel.data[i].w = snapshot.mass[i];

        h_accel.data[i] = snapshot.accel[i];
        
        h_charge.data[i] = snapshot.charge[i];

        h_diameter.data[i] = snapshot.diameter[i];

        h_image.data[i] = snapshot.image[i];

        h_tag.data[i] = i;
        h_rtag.data[i] = i;

        h_body.data[i] = snapshot.body[i];
>>>>>>> d8dcf65... Remove use of snapshot rtag
        }

    notifyParticleSort();

    }

//! take a particle data snapshot
/* \param snapshot The snapshot to write to

   \pre snapshot has to be allocated with a number of elements equal to the global number of particles)
*/
void ParticleData::takeSnapshot(SnapshotParticleData &snapshot)
    {
    // construct global snapshot
    if (snapshot.size != getNGlobal())
        {
        cerr << endl << "***Error! Number of particles in snapshot must be equal to global number of particles." << endl << endl;
        throw std::runtime_error("Error taking ParticleDataSnapshot");
        }

    ArrayHandle< Scalar4 > h_pos(m_pos, access_location::host, access_mode::read);
    ArrayHandle< Scalar4 > h_vel(m_vel, access_location::host, access_mode::read);
    ArrayHandle< Scalar3 > h_accel(m_accel, access_location::host, access_mode::read);
    ArrayHandle< int3 > h_image(m_image, access_location::host, access_mode::read);
    ArrayHandle< Scalar > h_charge(m_charge, access_location::host, access_mode::read);
    ArrayHandle< Scalar > h_diameter(m_diameter, access_location::host, access_mode::read);
    ArrayHandle< unsigned int > h_body(m_body, access_location::host, access_mode::read);
    ArrayHandle< unsigned int > h_global_tag(m_global_tag, access_location::host, access_mode::read);

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
        std::vector<unsigned int> global_tag(m_nparticles);
        std::map<unsigned int, unsigned int> rtag_map;

        for (unsigned int idx = 0; idx < m_nparticles; idx++)
            {
            pos[idx] = make_scalar3(h_pos.data[idx].x, h_pos.data[idx].y, h_pos.data[idx].z);
            vel[idx] = make_scalar3(h_vel.data[idx].x, h_vel.data[idx].y, h_vel.data[idx].z);
            accel[idx] = h_accel.data[idx];
            type[idx] = __scalar_as_int(h_pos.data[idx].w);
            mass[idx] = h_vel.data[idx].w;
            charge[idx] = h_charge.data[idx];
            diameter[idx] = h_diameter.data[idx];
            image[idx] = h_image.data[idx];
            body[idx] = h_body.data[idx];

            // insert reverse lookup global tag -> idx
            rtag_map.insert(std::pair<unsigned int, unsigned int>(h_global_tag.data[idx], idx));
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

        std::vector< std::map<unsigned int, unsigned int> > rtag_map_proc; // List of reverse-lookup maps

        boost::shared_ptr<boost::mpi::communicator> mpi_comm = m_decomposition->getMPICommunicator();
        unsigned int size = mpi_comm->size();
        unsigned int rank = mpi_comm->rank();

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
        rtag_map_proc.resize(size);

        unsigned int root = m_decomposition->getRoot();

        // collect all particle data on the root processor
        boost::mpi::gather(*mpi_comm, pos, pos_proc, root);
        boost::mpi::gather(*mpi_comm, vel, vel_proc, root);
        boost::mpi::gather(*mpi_comm, accel, accel_proc, root);
        boost::mpi::gather(*mpi_comm, type, type_proc, root);
        boost::mpi::gather(*mpi_comm, mass, mass_proc, root);
        boost::mpi::gather(*mpi_comm, charge, charge_proc, root);
        boost::mpi::gather(*mpi_comm, diameter, diameter_proc, root);
        boost::mpi::gather(*mpi_comm, image, image_proc, root);
        boost::mpi::gather(*mpi_comm, body, body_proc, root);

        // gather the reverse-lookup maps
        boost::mpi::gather(*mpi_comm, rtag_map, rtag_map_proc, root);

        if (rank == root)
            {
            std::map<unsigned int, unsigned int>::iterator it;

            for (unsigned int tag = 0; tag < getNGlobal(); tag++)
                {
                bool found = false;
                unsigned int rank;
                for (rank = 0; rank < (unsigned int) size; rank ++)
                    {
                    it = rtag_map_proc[rank].find(tag);
                    if (it != rtag_map_proc[rank].end())
                        {
                        found = true;
                        break;
                        }
                    }
                if (! found)
                    {
                    cerr << endl << "***Error! Could not find particle " << tag << " on any processor. " << endl << endl;
                    throw std::runtime_error("Error gathering ParticleData");
                    }

                // rank contains the processor rank on which the particle was found
                unsigned int idx = it->second;
                snapshot.pos[tag] = pos_proc[rank][idx];
                snapshot.vel[tag] = vel_proc[rank][idx];
                snapshot.accel[tag] = accel_proc[rank][idx];
                snapshot.type[tag] = type_proc[rank][idx];
                snapshot.mass[tag] = mass_proc[rank][idx];
                snapshot.charge[tag] = charge_proc[rank][idx];
                snapshot.diameter[tag] = diameter_proc[rank][idx];
                snapshot.image[tag] = image_proc[rank][idx];
                snapshot.body[tag] = body_proc[rank][idx];
                }
            } 
        }
    else
#endif
        {
        for (unsigned int idx = 0; idx < m_nparticles; idx++)
            {
            unsigned int tag = h_global_tag.data[idx];
            assert(tag < m_nglobal);
            snapshot.pos[tag] = make_scalar3(h_pos.data[idx].x, h_pos.data[idx].y, h_pos.data[idx].z);
            snapshot.vel[tag] = make_scalar3(h_vel.data[idx].x, h_vel.data[idx].y, h_vel.data[idx].z);
            snapshot.accel[tag] = h_accel.data[idx];
            snapshot.type[tag] = __scalar_as_int(h_pos.data[idx].w);
            snapshot.mass[tag] = h_vel.data[idx].w;
            snapshot.charge[tag] = h_charge.data[idx];
            snapshot.diameter[tag] = h_diameter.data[idx];
            snapshot.image[tag] = h_image.data[idx];
            snapshot.body[tag] = h_body.data[idx];
            }
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
            max_nparticles = ((unsigned int) (((float) max_nparticles) * m_resize_factor)) + 1 ;

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
            max_nparticles = ((unsigned int) (((float) max_nparticles) * m_resize_factor)) + 1 ;

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
    assert(m_decomposition);
    bool is_local = (getGlobalRTag(tag) < getN());
    int n_found;

    boost::shared_ptr<boost::mpi::communicator> mpi_comm = m_decomposition->getMPICommunicator();
    // First check that the particle is on exactly one processor
    all_reduce(*mpi_comm, is_local ? 1 : 0, n_found, std::plus<int>());

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
    all_reduce(*mpi_comm, is_local ? mpi_comm->rank() : -1, owner_rank, boost::mpi::maximum<int>());
    assert (owner_rank >= 0);
    assert (owner_rank < mpi_comm->size());

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
    if (m_decomposition)
        {
        unsigned int owner_rank = getOwnerRank(global_tag);
        broadcast(*m_decomposition->getMPICommunicator(), result, owner_rank);
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
    if (m_decomposition)
        {
        unsigned int owner_rank = getOwnerRank(global_tag);
        broadcast(*m_decomposition->getMPICommunicator(), result, owner_rank);
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
    if (m_decomposition)
        {
        unsigned int owner_rank = getOwnerRank(global_tag);
        broadcast(*m_decomposition->getMPICommunicator(), result, owner_rank);
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
    if (m_decomposition)
        {
        unsigned int owner_rank = getOwnerRank(global_tag);
        broadcast(*m_decomposition->getMPICommunicator(), result, owner_rank);
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
    if (m_decomposition)
        {
        unsigned int owner_rank = getOwnerRank(global_tag);
        broadcast(*m_decomposition->getMPICommunicator(), result, owner_rank);
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
    if (m_decomposition)
        {
        unsigned int owner_rank = getOwnerRank(global_tag);
        broadcast(*m_decomposition->getMPICommunicator(), result, owner_rank);
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
    if (m_decomposition)
        {
        unsigned int owner_rank = getOwnerRank(global_tag);
        broadcast(*m_decomposition->getMPICommunicator(), result, owner_rank);
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
    if (m_decomposition)
        {
        unsigned int owner_rank = getOwnerRank(global_tag);
        broadcast(*m_decomposition->getMPICommunicator(), result, owner_rank);
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
    if (m_decomposition)
        {
        unsigned int owner_rank = getOwnerRank(global_tag);
        broadcast(*m_decomposition->getMPICommunicator(), result, owner_rank);
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
    if (m_decomposition)
        {
        unsigned int owner_rank = getOwnerRank(global_tag);
        broadcast(*m_decomposition->getMPICommunicator(), result, owner_rank);
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
    if (m_decomposition)
        {
        unsigned int owner_rank = getOwnerRank(global_tag);
        broadcast(*m_decomposition->getMPICommunicator(), result, owner_rank);
        found = true;
        }
#endif
    assert(found);
    return result;

    }

//! Get the net torque a given particle
Scalar4 ParticleData::getNetTorque(unsigned int global_tag) const
    {
    unsigned int idx = getGlobalRTag(global_tag);
    bool found = (idx < getN());
    Scalar4 result;
    if (found)
        {
        ArrayHandle< Scalar4 > h_net_torque(m_net_torque, access_location::host, access_mode::read);
        result = h_net_torque.data[idx];
        }
#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        unsigned int owner_rank = getOwnerRank(global_tag);
        broadcast(*m_decomposition->getMPICommunicator(), result, owner_rank);
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
    if (m_decomposition)
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
    if (m_decomposition)
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
    if (m_decomposition)
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
    if (m_decomposition)
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
    if (m_decomposition)
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
    if (m_decomposition)
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
    if (m_decomposition)
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
    if (m_decomposition)
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
    if (m_decomposition)
        getOwnerRank(global_tag);
#endif
    if (found)
        {
        ArrayHandle< Scalar4 > h_orientation(m_orientation, access_location::host, access_mode::readwrite);
        h_orientation.data[idx] = orientation;
        }
    }

void export_BoxDim()
    {
    class_<BoxDim>("BoxDim")
    .def(init<Scalar>())
    .def(init<Scalar, Scalar, Scalar>())
    .def(init<Scalar3>())
    .def(init<Scalar3, Scalar3, uchar3>())
    .def("getPeriodic", &BoxDim::getPeriodic)
    .def("setPeriodic", &BoxDim::setPeriodic)
    .def("getL", &BoxDim::getL)
    .def("setL", &BoxDim::setL)
    .def("getLo", &BoxDim::getLo)
    .def("getHi", &BoxDim::getHi)
    .def("setLoHi", &BoxDim::setLoHi)
    .def("makeFraction", &BoxDim::makeFraction)
    .def("minImage", &BoxDim::minImage)
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
    .def("getGlobalBox", &ParticleData::getGlobalBox, return_value_policy<copy_const_reference>())
    .def("getBox", &ParticleData::getBox, return_value_policy<copy_const_reference>())
    .def("setGlobalBoxL", &ParticleData::setGlobalBoxL)
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
    .def("getNetTorque", &ParticleData::getNetTorque)
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
    .def("takeSnapshot", &ParticleData::takeSnapshot)
    .def("initializeFromSnapshot", &ParticleData::initializeFromSnapshot)
    .def("setDomainDecomposition", &ParticleData::setDomainDecomposition)
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
    .def_readwrite("num_particle_types", &SnapshotParticleData::num_particle_types)
    .def_readwrite("type_mapping", &SnapshotParticleData::type_mapping)
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

