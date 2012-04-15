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

/*! \file ParticleData.h
    \brief Defines the ParticleData class and associated utilities
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 )
#endif

#ifndef __PARTICLE_DATA_H__
#define __PARTICLE_DATA_H__

#include "HOOMDMath.h"
#include "GPUArray.h"

#ifdef ENABLE_CUDA
#include "ParticleData.cuh"
#endif

#include "ExecutionConfiguration.h"
#include "BoxDim.h"

#include <boost/shared_ptr.hpp>
#include <boost/signals.hpp>
#include <boost/function.hpp>
#include <boost/utility.hpp>
#include <boost/dynamic_bitset.hpp>

#ifdef ENABLE_MPI
#include "Index1D.h"
#include <boost/mpi.hpp>
#endif

#include <stdlib.h>
#include <vector>
#include <string>
#include <bitset>

using namespace std;

// windows doesn't understand __restrict__, it is __restrict instead
#ifdef WIN32
#define __restrict__ __restrict
#endif

/*! \ingroup hoomd_lib
    @{
*/

/*! \defgroup data_structs Data structures
    \brief All classes that are related to the fundamental data
        structures for storing particles.

    \details See \ref page_dev_info for more information
*/

/*! @}
*/

// Forward declaration of Profiler
class Profiler;

class BondData;

class WallData;

// Forward declaration of AngleData
class AngleData;

// Forward declaration of DihedralData
class DihedralData;

// Forward declaration of RigidData
class RigidData;

// Forward declaration of IntegratorData
class IntegratorData;

//! List of optional fields that can be enabled in ParticleData
struct pdata_flag
    {
    //! The enum
    enum Enum
        {
        isotropic_virial=0,  //!< Bit id in PDataFlags for the isotropic virial
        potential_energy,    //!< Bit id in PDataFlags for the potential energy
        pressure_tensor,     //!< Bit id in PDataFlags for the full virial
        };
    };

//! flags determines which optional fields in in the particle data arrays are to be computed / are valid
typedef std::bitset<32> PDataFlags;

//! Defines a simple structure to deal with complex numbers
/*! This structure is useful to deal with complex numbers for such situations
    as Fourier transforms. Note that we do not need any to define any operations and the
    default constructor is good enough
*/
struct CScalar
    {
    Scalar r; //!< Real part
    Scalar i; //!< Imaginary part
    };

//! Defines a simple moment of inertia structure
/*! This moment of interia is stored per particle. Because there are no per-particle body update steps in the
    design of hoomd, these values are never read or used except at initialization. Thus, a simple descriptive
    structure is used instead of an advanced and complicated GPUArray strided data array.
    
    The member variable components stores the 6 components of an upper-trianglar moment of inertia tensor.
    The components are, in order, Ixx, Ixy, Ixz, Iyy, Iyz, Izz.
    
    They are initialized to 0 and left that way if not specified in an initialization file.
*/
struct InertiaTensor
    {
    InertiaTensor()
        {
        for (unsigned int i = 0; i < 6; i++)
            components[i] = Scalar(0.0);
        }
    
    //! Set the components of the tensor
    void set(Scalar c0, Scalar c1, Scalar c2, Scalar c3, Scalar c4, Scalar c5)
        {
        components[0] = c0;
        components[1] = c1;
        components[2] = c2;
        components[3] = c3;
        components[4] = c4;
        components[5] = c5;
        }
    
    Scalar components[6];   //!< Stores the components of the inertia tensor
    };

//! Sentinel value in \a body to signify that this particle does not belong to a rigid body
const unsigned int NO_BODY = 0xffffffff;

//! Sentinal value in \a r_tag to signify that this particle is not currently present on the local processor
const unsigned int NOT_LOCAL = 0xffffffff;

//! Handy structure for passing around per-particle data
/*! A snapshot is used for two purposes:
 * - Initializing the ParticleData from a ParticleDataInitializer
 * - inside an Analyzer to iterate over the current ParticleData
 *
 * Initializing the ParticleData is accomplished by first filling the particle data arrays with default values
 * (such as type, mass, diameter). Then a snapshot of this initial state is taken and pased to the
 * ParticleDataInitializer, which may modify any of the fields of the snapshot. It then returns it to
 * ParticleData, which in turn initializes its internal arrays from the snapshot using ParticleData::initializeFromSnapshot().
 *
 * To support the second scenerio it is necessary that particles can be accessed in global tag order. Therefore,
 * the data in a snapshot is stored in global tag order.
 * \ingroup data_structs
 */
struct SnapshotParticleData {
    //! Empty snapshot
    SnapshotParticleData()
        : size(0), num_particle_types(0)
        { }

    //! constructor
    /*! \param N number of particles to allocate memory for
     */
    SnapshotParticleData(unsigned int N)
       : size(N), num_particle_types(0)
       {
       pos.resize(N);
       vel.resize(N);
       accel.resize(N);
       type.resize(N);
       mass.resize(N);
       charge.resize(N);
       diameter.resize(N);
       image.resize(N);
       body.resize(N);
       size = N;
       }

    std::vector<Scalar3> pos;       //!< positions
    std::vector<Scalar3> vel;       //!< velocities
    std::vector<Scalar3> accel;     //!< accelerations
    std::vector<unsigned int> type; //!< types
    std::vector<Scalar> mass;       //!< masses
    std::vector<Scalar> charge;     //!< charges
    std::vector<Scalar> diameter;   //!< diameters
    std::vector<int3> image;        //!< images
    std::vector<unsigned int> body; //!< body ids
    unsigned int size;              //!< number of particles in this snapshot
    unsigned int num_particle_types;//!< Number of particle types defined
    std::vector<std::string> type_mapping; //!< Mapping between particle type ids and names
    };

#ifdef ENABLE_MPI
//! Structure that holds information about the domain decomposition
/*! This structure contains data about the global grid and about the role of the local processor in the grid.
 *  Specifically, it contains information about the neighbors and boundaries
 *  of the local simulation box, for all six spatial directions.
 *  A direction is referred to by an integer value \b dir, where
 *  dir =<br>
 *        0 <-> east <br>
 *        1 <-> west <br>
 *        2 <-> north <br>
 *        3 <-> south <br>
 *        4 <-> up <br>
 *        5 <-> down <br>
 */
struct DomainDecomposition
    {
    unsigned int nx;           //!< Number of processors along the x-axis
    unsigned int ny;           //!< Number of processors along the y-axis
    unsigned int nz;           //!< Number of processors along the z-axis

    uint3 grid_pos;            //!< Grid position of this processor
    Index3D index;             //!< Index to the 3D processor grid
    unsigned int neighbors[6]; //!< Rank of neighbors in every spatial direction
    bool is_at_boundary[6];    //!< whether this box shares a boundary with the global simulation box (per direction)
    unsigned int root;         //!< Rank of the root processor
    };
#endif

//! Abstract interface for initializing a ParticleData
/*! A ParticleDataInitializer should only be used with the appropriate constructor
    of ParticleData(). That constructure calls the methods of this class to determine
    the number of particles, number of particle types, the simulation box, and then
    initializes itself. Then initSnapshot() is called to fill out the ParticleDataSnapshot
    to be used to initalize the particle data arrays

    \note This class is an abstract interface with pure virtual functions. Derived
    classes must implement these methods.
    \ingroup data_structs
    */
class ParticleDataInitializer
    {
    public:
        //! Empty constructor
        ParticleDataInitializer() { }
        //! Empty Destructor
        virtual ~ParticleDataInitializer() { }
        
        //! Returns the number of local particles to be initialized
        virtual unsigned int getNumParticles() const = 0;
        
        //! Returns the box the particles will sit in
        virtual BoxDim getBox() const = 0;
        
        //! Initializes the snapshot of the particle data arrays
        /*! \param snapshot snapshot to initialize
        */
        virtual void initSnapshot(SnapshotParticleData& snapshot) const = 0;
        
        //! Initialize the simulation walls
        /*! \param wall_data Shared pointer to the WallData to initialize
            This base class defines an empty method, as walls are optional
        */
        virtual void initWallData(boost::shared_ptr<WallData> wall_data) const {}

        //! Initialize the integrator variables
        /*! \param integrator_data Shared pointer to the IntegratorData to initialize
            This base class defines an empty method, since initializing the 
            integrator variables is optional
        */
        virtual void initIntegratorData(boost::shared_ptr<IntegratorData> integrator_data) const {}
        
        //! Returns the number of dimensions
        /*! The base class returns 3 */
        virtual unsigned int getNumDimensions() const
            {
            return 3;
            }
        
        //! Returns the number of bond types to be created
        /*! Bonds are optional: the base class returns 1 */
        virtual unsigned int getNumBondTypes() const
            {
            return 1;
            }
            
        /*! Angles are optional: the base class returns 1 */
        virtual unsigned int getNumAngleTypes() const
            {
            return 1;
            }
            
        /*! Dihedrals are optional: the base class returns 1 */
        virtual unsigned int getNumDihedralTypes() const
            {
            return 1;
            }
            
        /*! Impropers are optional: the base class returns 1 */
        virtual unsigned int getNumImproperTypes() const
            {
            return 1;
            }
            
        //! Initialize the bond data
        /*! \param bond_data Shared pointer to the BondData to be initialized
            Bonds are optional: the base class does nothing
        */
        virtual void initBondData(boost::shared_ptr<BondData> bond_data) const {}
        
        //! Initialize the angle data
        /*! \param angle_data Shared pointer to the AngleData to be initialized
            Angles are optional: the base class does nothing
        */
        virtual void initAngleData(boost::shared_ptr<AngleData> angle_data) const {}
        
        //! Initialize the dihedral data
        /*! \param dihedral_data Shared pointer to the DihedralData to be initialized
            Dihedrals are optional: the base class does nothing
        */
        virtual void initDihedralData(boost::shared_ptr<DihedralData> dihedral_data) const {}
        
        //! Initialize the improper data
        /*! \param improper_data Shared pointer to the ImproperData to be initialized
            Impropers are optional: the base class does nothing
        */
        virtual void initImproperData(boost::shared_ptr<DihedralData> improper_data) const {}
        
        //! Initialize the rigid data
        /*! \param rigid_data Shared pointer to the RigidData to be initialized
            Rigid bodies are optional: the base class does nothing
        */
        virtual void initRigidData(boost::shared_ptr<RigidData> rigid_data) const {}
        
        //! Initialize the orientation data
        /*! \param orientation Pointer to one orientation per particle to be initialized
        */
        virtual void initOrientation(Scalar4 *orientation) const {}
        
        //! Initialize the inertia tensor data
        /*! \param moment_inertia Pointer to one inertia tensor per particle to be initialize (in tag order!)
        */
        virtual void initMomentInertia(InertiaTensor *moment_inertia) const {}
            
    };

//! Manages all of the data arrays for the particles
/*! <h1> General </h1>
    ParticleData stores and manages particle coordinates, velocities, accelerations, type,
    and tag information. This data must be available both via the CPU and GPU memories.
    All copying of data back and forth from the GPU is accomplished transparently by GPUArray.

    For performance reasons, data is stored as simple arrays. Once a handle to the particle data
    GPUArrays has been acquired, the coordinates of the particle with
    <em>index</em> \c i can be accessed with <code>pos_array_handle.data[i].x</code>,
    <code>pos_array_handle.data[i].y</code>, and <code>pos_array_handle.data[i].z</code>
    where \c i runs from 0 to <code>getN()</code>.

    Velocities and other properties can be accessed in a similar manner.
    
    \note Position and type are combined into a single Scalar4 quantity. x,y,z specifies the position and w specifies
    the type. Use __scalar_as_int() / __int_as_scalar() (or __int_as_float() / __float_as_int()) to extract / set
    this integer that is masquerading as a scalar.
    
    \note Velocity and mass are combined into a single Scalar4 quantity. x,y,z specifies the velocity and w specifies
    the mass.

    \warning Local particles can and will be rearranged in the arrays throughout a simulation.
    So, a particle that was once at index 5 may be at index 123 the next time the data
    is acquired. Individual particles can be tracked through all these changes by their (global) tag.
    The tag of a particle is stored in the \c m_global_tag array, and the ith element contains the tag of the particle
    with index i. Conversely, the the index of a particle with tag \c global_tag can be read from
    the element at position \c global_tag in the a \c m_global_rtag array.

    In a parallel simulation, the global tag is unique among all processors.

    In order to help other classes deal with particles changing indices, any class that
    changes the order must call notifyParticleSort(). Any class interested in being notified
    can subscribe to the signal by calling connectParticleSort().

    Some fields in ParticleData are not computed and assigned by default because they require additional processing
    time. PDataFlags is a bitset that lists which flags (enumerated in pdata_flag) are enable/disabled. Computes should
    call getFlags() and compute the requested quantities whenever the corresponding flag is set. Updaters and Analyzers
    can request flags be computed via their getRequestedPDataFlags() methods. A particular updater or analyzer should 
    return a bitset PDataFlags with only the bits set for the flags that it needs. During a run, System will query
    the updaters and analyzers that are to be executed on the current step. All of the flag requests are combined
    with the binary or operation into a single set of flag requests. System::run() then sets the flags by calling
    setPDataFlags so that the computes produce the requested values during that step.
    
    These fields are:
     - pdata_flag::isotropic_virial - specify that the net_virial should be/is computed (getNetVirial)
     - pdata_flag::potential_energy - specify that the potential energy .w component stored in the net force array 
       (getNetForce) is valid
     - pdata_flag::pressure_tensor - specify that the full virial tensor is valid
       
    If these flags are not set, these arrays can still be read but their values may be incorrect.
    
    If any computation is unable to supply the appropriate values (i.e. rigid body virial can not be computed
    until the second step of the simulation), then it should remove the flag to signify that the values are not valid.
    Any analyzer/updater that expects the value to be set should check the flags that are actually set.
    
    \note Particles are not checked if their position is actually inside the local box. In fact, when using spatial domain decomposition,
    particles may temporarily move outside the boundaries.

    \ingroup data_structs
    
    <h1>Parallel simulations</h1>

    In a parallel (or domain decompositon) simulation, the ParticleData may either correspond to the global state of the
    simulation (e.g. before and after a simulation run), or to the local particles only (e.g. during a simulation run).
    In the latter case, getN() returns the current number of \a local particles. The method getNGlobal() can be used to query the \a global number
    of particles on all processors.

    During the simulation particles may enter or leave the box, therefore the number of \a local particles may change.
    To account for this, the size of the particle data arrays is dynamically updated using amortized doubling of the array sizes. To add particles to
    the domain, the addParticles() method is called, and the arrays are resized if necessary. Conversely, if particles are removed,
    the removeParticles() method is called.

    In addition, since many other classes maintain internal arrays with data for every particle (such as neighbor lists etc.), these
    arrays need to be resized, too, if the particle number changes. Everytime the particle data arrays are reallocated, a
    maximum particle number change signal is triggered. Other classes can subscribe to this signal using connectMaxParticleNumberChange().
    They may use the current maxium size of the particle arrays, which is returned by getMaxN().
    This size changes only infrequently (it is doubled when necessary, see above). Note that getMaxN() can return a higher number
    than the actual number of particles.

    \note addParticles() and removeParticles() only change the particle number counters and the allocated memory size (if necessary).
    They do not actually change any data in the particle arrays. The caller is responsible for (re-)organizing the particle data when particles
    are added or deleted.

    If, after insertion or deletion of particles, the reorganisation of the particle data is complete, i.e. all the particle data
    fields are filled, the class that has modified the ParticleData must inform other classes about the new particle data
    using notifyParticleSort().

    Particle data also stores temporary particles ('ghost atoms'). These are added after the local particle data (i.e. with indices
    starting at getN()). It keeps track of those particles using the addGhostParticles() and removeAllGhostParticles() methods.
    The caller is responsible for updating the particle data arrays with the ghost particle information.

    <h1> Anisotropic particles</h1>

    Anisotropic particles are handled by storing an orientation quaternion for every particle in the simulation.
    Similarly, a net torque is computed and stored for each particle. The design decision made is to not
    duplicate efforts already made to enable composite bodies of anisotropic particles. So the particle orientation
    is a read only quantity when used by most of HOOMD. To integrate this degree of freedom forward, the particle
    must be part of a composite body (stored and handled by RigidData) (there can be single-particle bodies,
    of course) where integration methods like NVERigid will handle updating the degrees of freedom of the composite
    body and then set the constrained position, velocity, and orientation of the constituent particles.
    
    To enable correct initialization of the composite body moment of inertia, each particle is also assigned
    an individual moment of inertia which is summed up correctly to determine the composite body's total moment of
    inertia. As such, the initial particle moment of inertias are only ever used during initialization and do not
    need to be stored in an efficient GPU data structure. Nor does the inertia tensor data need to be resorted,
    so it will always remain in tag order.
    
    Access the orientation quaternion of each particle with the GPUArray gotten from getOrientationArray(), the net
    torque with getTorqueArray(). Individual inertia tensor values can be accessed with getInertiaTensor() and
    setInertiaTensor()
*/
class ParticleData : boost::noncopyable
    {
    public:
        //! Construct with N particles in the given box
        ParticleData(unsigned int N,
                     const BoxDim &box,
                     unsigned int n_types,
                     boost::shared_ptr<ExecutionConfiguration> exec_conf);
        
        //! Construct from an initializer
        ParticleData(const ParticleDataInitializer& init,
                     boost::shared_ptr<ExecutionConfiguration> exec_conf);
        
        //! Destructor
        virtual ~ParticleData();
        
        //! Get the simulation box
        const BoxDim& getBox() const;
        //! Set the simulation box Lengths
        void setGlobalBoxL(const Scalar3 &L);

        //! Get the global simulation box
        const BoxDim& getGlobalBox() const;
        //! Set the global simulation box
        void setGlobalBox(const BoxDim &global_box);
         
        //! Access the execution configuration
        boost::shared_ptr<const ExecutionConfiguration> getExecConf() const
            {
            return m_exec_conf;
            }
            
        //! Get the number of particles
        /*! \return Number of particles in the box
        */
        inline unsigned int getN() const
            {
            return m_nparticles;
            }

        //! Get the currrent maximum number of particles
        /*\ return Maximum number of particles that can be stored in the particle array
        * this number has to be larger than getN() + getNGhosts()
        */
        inline unsigned int getMaxN() const
            {
            return m_max_nparticles;
            }

        //! Get current number of ghost particles
        /*\ return Number of ghost particles
        */
        inline unsigned int getNGhosts() const
            {
            return m_nghosts;
            }

        //! Get the global number of particles in the simulation
        /*!\ return Global number of particles
         */
        inline unsigned int getNGlobal() const
            {
            return m_nglobal;
            }

        //! Set global number of particles
        /*! \param nglobal Global number of particles
         */
        void setNGlobal(unsigned int nglobal);

        //! Get the number of particle types
        /*! \return Number of particle types
            \note Particle types are indexed from 0 to NTypes-1
        */
        unsigned int getNTypes() const
            {
            return m_ntypes;
            }
            
        //! Get the maximum diameter of the particle set
        /*! \return Maximum Diameter Value
        */
        Scalar getMaxDiameter() const
            {
            Scalar maxdiam = 0;
            ArrayHandle< Scalar > h_diameter(getDiameters(), access_location::host, access_mode::read);
            for (unsigned int i = 0; i < m_nparticles; i++) if (h_diameter.data[i] > maxdiam) maxdiam = h_diameter.data[i];
            return maxdiam;
            }
            
        //! Return positions and types
        const GPUArray< Scalar4 >& getPositions() const { return m_pos; }

        //! Return velocities and masses
        const GPUArray< Scalar4 >& getVelocities() const { return m_vel; }
        
        //! Return accelerations
        const GPUArray< Scalar3 >& getAccelerations() const { return m_accel; }

        //! Return charges
        const GPUArray< Scalar >& getCharges() const { return m_charge; }

        //! Return diameters
        const GPUArray< Scalar >& getDiameters() const { return m_diameter; }

        //! Return images
        const GPUArray< int3 >& getImages() const { return m_image; }

        //! Return tags
        const GPUArray< unsigned int >& getTags() const { return m_tag; }

        //! Return reverse-lookup tags
        const GPUArray< unsigned int >& getRTags() const { return m_rtag; }

        //! Return body ids
        const GPUArray< unsigned int >& getBodies() const { return m_body; }

        //! Return global tags
        const GPUArray< unsigned int >& getGlobalTags() const { return m_global_tag; }

        //! Return array of global reverse lookup tags
        const GPUArray< unsigned int >& getGlobalRTags() { return m_global_rtag; }

        
        //! Set the profiler to profile CPU<-->GPU memory copies
        /*! \param prof Pointer to the profiler to use. Set to NULL to deactivate profiling
        */
        void setProfiler(boost::shared_ptr<Profiler> prof)
            {
            m_prof=prof;
            }
            
        //! Connects a function to be called every time the particles are rearranged in memory
        boost::signals::connection connectParticleSort(const boost::function<void ()> &func);
        
        //! Notify listeners that the particles have been rearranged in memory
        void notifyParticleSort();
        
        //! Connects a function to be called every time the box size is changed
        boost::signals::connection connectBoxChange(const boost::function<void ()> &func);

        //! Connects a function to be called every time the maximum particle number changes
        boost::signals::connection connectMaxParticleNumberChange(const boost::function< void()> &func);

        //! Connects a function to be called every time the ghost particles are updated
        boost::signals::connection connectGhostParticleNumberChange(const boost::function< void()> &func);

        //! Notify listeners that the number of ghost particles has changed
        void notifyGhostParticleNumberChange();

        //! Gets the particle type index given a name
        unsigned int getTypeByName(const std::string &name) const;

        //! Gets the name of a given particle type index
        std::string getNameByType(unsigned int type) const;
        
        //! Get the net force array
        const GPUArray< Scalar4 >& getNetForce() const { return m_net_force; }
        
        //! Get the net virial array
        const GPUArray< Scalar >& getNetVirial() const { return m_net_virial; }
        
        //! Get the net torque array
        const GPUArray< Scalar4 >& getNetTorqueArray() const { return m_net_torque; }
        
        //! Get the orientation array
        const GPUArray< Scalar4 >& getOrientationArray() const { return m_orientation; }

#ifdef ENABLE_MPI
        //! Find the processor that owns a particle
        unsigned int getOwnerRank(unsigned int tag) const;
#endif

        //! Get the current position of a particle
        Scalar3 getPosition(unsigned int global_tag) const;

        //! Get the current velocity of a particle
        Scalar3 getVelocity(unsigned int global_tag) const;

        //! Get the current acceleration of a particle
        Scalar3 getAcceleration(unsigned int global_tag) const;

        //! Get the current image flags of a particle
        int3 getImage(unsigned int global_tag) const;

        //! Get the current mass of a particle
        Scalar getMass(unsigned int global_tag) const;

        //! Get the current diameter of a particle
        Scalar getDiameter(unsigned int global_tag) const;

        //! Get the current charge of a particle
        Scalar getCharge(unsigned int global_tag) const;

        //! Get the body id of a particle
        unsigned int getBody(unsigned int global_tag) const;

        //! Get the current type of a particle
        unsigned int getType(unsigned int global_tag) const;

        //! Get the current index of a particle with a given local tag
        unsigned int getRTag(unsigned int tag) const
            {
            assert(tag < getN());
            ArrayHandle< unsigned int> h_rtag(m_rtag, access_location::host, access_mode::read);
            unsigned int idx = h_rtag.data[tag];
            return idx;
            }

        //! Get the current index of a particle with a given global tag
        inline unsigned int getGlobalRTag(unsigned int global_tag) const
            {
            assert(global_tag < m_nglobal);
            ArrayHandle< unsigned int> h_global_rtag(m_global_rtag,access_location::host, access_mode::read);
            unsigned int idx = h_global_rtag.data[global_tag];
#ifdef ENABLE_MPI
            if ((! m_mpi_comm) && !(idx < getN()))
                {
                cerr << endl << "***Error! Possible internal error detected. Could not find particle "
                     << global_tag << "." << endl << endl;
                throw std::runtime_error("Error accessing particle data.");
                }
#endif
            assert(idx < getN() + getNGhosts() || idx == NOT_LOCAL);
            return idx;
            }

        //! Return true if particle is local (= owned by this processor)
        bool isParticleLocal(unsigned int global_tag) const
             {
             assert(global_tag < m_nglobal);
             ArrayHandle< unsigned int> h_global_rtag(m_global_rtag,access_location::host, access_mode::read);
             return h_global_rtag.data[global_tag] < getN();
             }

        //! Get the orientation of a particle with a given tag
        Scalar4 getOrientation(unsigned int global_tag) const;

        //! Get the inertia tensor of a particle with a given tag
        const InertiaTensor& getInertiaTensor(unsigned int tag) const
            {
            return m_inertia_tensor[tag];
            }

        //! Get the net force / energy on a given particle
        Scalar4 getPNetForce(unsigned int global_tag) const;

        //! Set the current position of a particle
        void setPosition(unsigned int global_tag, const Scalar3& pos);

        //! Set the current velocity of a particle
        void setVelocity(unsigned int global_tag, const Scalar3& vel);

        //! Set the current image flags of a particle
        void setImage(unsigned int global_tag, const int3& image);

        //! Set the current charge of a particle
        void setCharge(unsigned int global_tag, Scalar charge);

        //! Set the current mass of a particle
        void setMass(unsigned int global_tag, Scalar mass);

        //! Set the current diameter of a particle
        void setDiameter(unsigned int global_tag, Scalar diameter);

        //! Set the body id of a particle
        void setBody(unsigned int global_tag, int body);

        //! Set the current type of a particle
        void setType(unsigned int global_tag, unsigned int typ);

        //! Set the orientation of a particle with a given tag
        void setOrientation(unsigned int global_tag, const Scalar4& orientation);

        //! Get the inertia tensor of a particle with a given tag
        void setInertiaTensor(unsigned int global_tag, const InertiaTensor& tensor)
            {
            m_inertia_tensor[global_tag] = tensor;
            }
            
        //! Get the particle data flags
        PDataFlags getFlags() { return m_flags; }
        
        //! Set the particle data flags
        /*! \note Setting the flags does not make the requested quantities immediately available. Only after the next
            set of compute() calls will the requested values be computed. The System class talks to the various
            analyzers and updaters to determine the value of the flags for any given time step.
        */
        void setFlags(const PDataFlags& flags) { m_flags = flags; }
        
        //! Remove the given flag
        void removeFlag(pdata_flag::Enum flag) { m_flags[flag] = false; }

        //! Initialize from a snapshot
        void initializeFromSnapshot(const SnapshotParticleData & snapshot);

        //! Take a snapshot
        void takeSnapshot(SnapshotParticleData &snapshot);

        //! Remove particles from the local particle data
        void removeParticles(const unsigned int n);

        //! Add a number of particles to the local particle data
        void addParticles(const unsigned int n);

        //! Add ghost particles at the end of the local particle data
        void addGhostParticles(const unsigned int nghosts);

        //! Remove all ghost particles from system
        void removeAllGhostParticles()
            {
            m_nghosts = 0;
            }
#ifdef ENABLE_MPI
        //! Set the communicator
        void setMPICommunicator(boost::shared_ptr<const boost::mpi::communicator> mpi_comm)
            {
            m_mpi_comm = mpi_comm;
            }

        //! Set domain decomposition information
        void setDomainDecomposition(const DomainDecomposition decomposition)
            {
            m_decomposition = decomposition;
            }

        //! Get the communicator
        boost::shared_ptr<const boost::mpi::communicator> getMPICommunicator()
            {
            return m_mpi_comm;
            }
#endif

    private:
        BoxDim m_box;                               //!< The simulation box
        BoxDim m_global_box;                        //!< Global simulation box
        boost::shared_ptr<ExecutionConfiguration> m_exec_conf; //!< The execution configuration
#ifdef ENABLE_MPI
        boost::shared_ptr<const boost::mpi::communicator> m_mpi_comm; //!< MPI communicator
        DomainDecomposition m_decomposition;        //!< Domain decomposition data
#endif
        unsigned int m_ntypes;                      //!< Number of particle types
        
        std::vector<std::string> m_type_mapping;    //!< Mapping between particle type indices and names
        
        boost::signal<void ()> m_sort_signal;       //!< Signal that is triggered when particles are sorted in memory
        boost::signal<void ()> m_boxchange_signal;  //!< Signal that is triggered when the box size changes
        boost::signal<void ()> m_max_particle_num_signal; //!< Signal that is triggered when the maximum particle number changes
        boost::signal<void ()> m_ghost_particle_num_signal; //!< Signal that is triggered when ghost particles are added to or deleted

        unsigned int m_nparticles;                  //!< number of particles
        unsigned int m_nghosts;                     //!< number of ghost particles
        unsigned int m_max_nparticles;              //!< maximum number of particles
        unsigned int m_nglobal;                     //!< global number of particles

        // per-particle data
        GPUArray<Scalar4> m_pos;                    //!< particle positions and types
        GPUArray<Scalar4> m_vel;                    //!< particle velocities and masses
        GPUArray<Scalar3> m_accel;                  //!< particle accelerations
        GPUArray<Scalar> m_charge;                  //!< particle charges
        GPUArray<Scalar> m_diameter;                //!< particle diameters
        GPUArray<int3> m_image;                     //!< particle images
        GPUArray<unsigned int> m_tag;               //!< particle tags
        GPUArray<unsigned int> m_rtag;              //!< reverse lookup tags
        GPUArray<unsigned int> m_global_tag;        //!< global particle tags
        GPUArray<unsigned int> m_global_rtag;       //!< reverse lookup of local particle indices from global tags
        GPUArray<unsigned int> m_body;              //!< rigid body ids

        boost::shared_ptr<Profiler> m_prof;         //!< Pointer to the profiler. NULL if there is no profiler.
        
        GPUArray< Scalar4 > m_net_force;             //!< Net force calculated for each particle
        GPUArray< Scalar > m_net_virial;             //!< Net virial calculated for each particle (2D GPU array of dimensions 6*number of particles)
        GPUArray< Scalar4 > m_net_torque;            //!< Net torque calculated for each particle
        GPUArray< Scalar4 > m_orientation;           //!< Orientation quaternion for each particle (ignored if not anisotropic)
        std::vector< InertiaTensor > m_inertia_tensor; //!< Inertia tensor for each particle

        const float m_resize_factor;                 //!< The numerical factor with which the particle data arrays are resized
        PDataFlags m_flags;                          //!< Flags identifying which optional fields are valid
        
        
        //! Helper function to allocate particle data
        void allocate(unsigned int N);

        //! Helper function to reallocate particle data
        void reallocate(unsigned int max_n);

        //! Helper function to check that particles are in the box
        bool inBox();
    };


//! Exports the BoxDim class to python
void export_BoxDim();
//! Exports ParticleDataInitializer to python
void export_ParticleDataInitializer();
//! Exports ParticleData to python
void export_ParticleData();
//! Export SnapshotParticleData to python
void export_SnapshotParticleData();

#endif

#ifdef WIN32
#pragma warning( pop )
#endif

