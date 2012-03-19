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

//! Handy structure for passing around per-particle data
/* TODO: document me
*/
struct SnapshotParticleData {
    //! constructor
    //! \param N number of particles to allocate memory for
    SnapshotParticleData(unsigned int N)
       {
       pos.resize(N);
       vel.resize(N);
       accel.resize(N);
       type.resize(N);
       mass.resize(N);
       charge.resize(N);
       diameter.resize(N);
       image.resize(N);
       rtag.resize(N);
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
    std::vector<unsigned int> rtag; //!< reverse-lookup tags
    std::vector<unsigned int> body; //!< body ids
    unsigned int size;              //!< number of particles in this snapshot
    };

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
        
        //! Returns the number of particles to be initialized
        virtual unsigned int getNumParticles() const = 0;
        
        //! Returns the number of particles types to be initialized
        virtual unsigned int getNumParticleTypes() const = 0;
        
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
        
        //! Intialize the type mapping
        virtual std::vector<std::string> getTypeMapping() const = 0;

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
/*! ParticleData stores and manages particle coordinates, velocities, accelerations, type,
    and tag information. This data must be available both via the CPU and GPU memories.
    All copying of data back and forth from the GPU is accomplished transparently by GPUArray.

    For performance reasons, data is stored as simple arrays. Once a handle to the particle data
    GPUArrays has been acquired, the coordinates of the particle with
    <em>index</em> \c i can be accessed with <code>pos_array_handle.data[i].x</code>,
    <code>pos_array_handle.data[i].y</code>, and <code>pos_array_handle.data[i].z</code>
    where \c i runs from 0 to <code>getN()</code>.

    Velocities and other propertys can be accessed in a similar manner.
    
    \note Position and type are combined into a single Scalar4 quantity. x,y,z specifies the position and w specifies
    the type. Use __scalar_as_int() / __int_as_scalar() (or __int_as_float() / __float_as_int()) to extract / set
    this integer that is masquerading as a scalar.
    
    \note Velocity and mass are combined into a single Scalar4 quantity. x,y,z specifies the velocity and w specifies
    the mass.

    \warning Particles can and will be rearranged in the arrays throughout a simulation.
    So, a particle that was once at index 5 may be at index 123 the next time the data
    is acquired. Individual particles can be tracked through all these changes by their tag.
    <code>tag[i]</code> identifies the tag of the particle that currently has index
    \c i, and the index of a particle with tag \c tag can be read from <code>rtag[tag]</code>.

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
    
    \note When writing to the particle data, particles must not be moved outside the box.
    In debug builds, any aquire will fail an assertion if this is done.
    \ingroup data_structs
    
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
        virtual ~ParticleData() {}
        
        //! Get the simulation box
        const BoxDim& getBox() const;
        //! Set the simulation box
        void setBox(const BoxDim &box);

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
            
        //! return positions and types
        const GPUArray< Scalar4 >& getPositions() const { return m_pos; }

        //! return velocities and masses
        const GPUArray< Scalar4 >& getVelocities() const { return m_vel; }
        
        //! return accelerations
        const GPUArray< Scalar3 >& getAccelerations() const { return m_accel; }

        //! return charges
        const GPUArray< Scalar >& getCharges() const { return m_charge; }

        //! return diameters
        const GPUArray< Scalar >& getDiameters() const { return m_diameter; }

        //! return images
        const GPUArray< int3 >& getImages() const { return m_image; }

        //! return tags
        const GPUArray< unsigned int >& getTags() const { return m_tag; }

        //! return reverse-lookup tags
        const GPUArray< unsigned int >& getRTags() const { return m_rtag; }

        //! return body ids
        const GPUArray< unsigned int >& getBodies() const { return m_body; }

#ifdef ENABLE_CUDA
        //! Get the box for the GPU
        /*! \returns Box dimensions suitable for passing to the GPU code
        */
        const gpu_boxsize& getBoxGPU() const
            {
            return m_gpu_box;
            }
            
#endif
        
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
        
        //! Get the current position of a particle
        Scalar3 getPosition(unsigned int tag) const
            {
            assert(tag < getN());
            ArrayHandle< Scalar4 > h_pos(m_pos, access_location::host, access_mode::read);
            ArrayHandle< unsigned int> h_rtag(m_rtag, access_location::host, access_mode::read);
            unsigned int idx = h_rtag.data[tag];
            Scalar3 result = make_scalar3(h_pos.data[idx].x, h_pos.data[idx].y, h_pos.data[idx].z);
            return result;
            }
        //! Get the current velocity of a particle
        Scalar3 getVelocity(unsigned int tag) const
            {
            assert(tag < getN());
            ArrayHandle< Scalar4 > h_vel(m_vel, access_location::host, access_mode::read);
            ArrayHandle< unsigned int> h_rtag(m_rtag, access_location::host, access_mode::read);
            unsigned int idx = h_rtag.data[tag];
            Scalar3 result = make_scalar3(h_vel.data[idx].x, h_vel.data[idx].y, h_vel.data[idx].z);
            return result;
            }
        //! Get the current acceleration of a particle
        Scalar3 getAcceleration(unsigned int tag) const
            {
            assert(tag < getN());
            ArrayHandle< Scalar3 > h_accel(m_accel, access_location::host, access_mode::read);
            ArrayHandle< unsigned int> h_rtag(m_rtag, access_location::host, access_mode::read);
            unsigned int idx = h_rtag.data[tag];
            Scalar3 result = make_scalar3(h_accel.data[idx].x, h_accel.data[idx].y, h_accel.data[idx].z);
            return result;
            }
        //! Get the current image flags of a particle
        int3 getImage(unsigned int tag) const
            {
            assert(tag < getN());
            ArrayHandle< int3 > h_image(m_image, access_location::host, access_mode::read);
            ArrayHandle< unsigned int> h_rtag(m_rtag, access_location::host, access_mode::read);
            unsigned int idx = h_rtag.data[tag];
            int3 result = make_int3(h_image.data[idx].x, h_image.data[idx].y, h_image.data[idx].z);
            return result;
            }
        //! Get the current charge of a particle
        Scalar getCharge(unsigned int tag) const
            {
            assert(tag < getN());
            ArrayHandle< Scalar > h_charge(m_charge, access_location::host, access_mode::read);
            ArrayHandle< unsigned int> h_rtag(m_rtag, access_location::host, access_mode::read);
            unsigned int idx = h_rtag.data[tag];
            Scalar result = h_charge.data[idx];
            return result;
            }
        //! Get the current mass of a particle
        Scalar getMass(unsigned int tag) const
            {
            ArrayHandle< Scalar4 > h_vel(m_vel, access_location::host, access_mode::read);
            ArrayHandle< unsigned int> h_rtag(m_rtag, access_location::host, access_mode::read);
            unsigned int idx = h_rtag.data[tag];
            assert(tag < getN());
            Scalar result = h_vel.data[idx].w;
            return result;
            }
        //! Get the current diameter of a particle
        Scalar getDiameter(unsigned int tag) const
            {
            assert(tag < getN());
            ArrayHandle< Scalar > h_diameter(m_diameter, access_location::host, access_mode::read);
            ArrayHandle< unsigned int> h_rtag(m_rtag, access_location::host, access_mode::read);
            unsigned int idx = h_rtag.data[tag];
            Scalar result = h_diameter.data[idx];
            return result;
            }
        //! Get the current diameter of a particle
        unsigned int getBody(unsigned int tag) const
            {
            assert(tag < getN());
            ArrayHandle< unsigned int > h_body(m_body, access_location::host, access_mode::read);
            ArrayHandle< unsigned int> h_rtag(m_rtag, access_location::host, access_mode::read);
            unsigned int idx = h_rtag.data[tag];
            unsigned int result = h_body.data[idx];
            return result;
            }
        //! Get the current type of a particle
        unsigned int getType(unsigned int tag) const
            {
            assert(tag < getN());
            ArrayHandle< Scalar4 > h_pos(m_pos, access_location::host, access_mode::read);
            ArrayHandle< unsigned int> h_rtag(m_rtag, access_location::host, access_mode::read);
            unsigned int idx = h_rtag.data[tag];
            unsigned int result = __scalar_as_int(h_pos.data[idx].w);
            return result;
            }

        //! Get the current index of a particle with a given tag
        unsigned int getRTag(unsigned int tag) const
            {
            assert(tag < getN());
            ArrayHandle< unsigned int> h_rtag(m_rtag, access_location::host, access_mode::read);
            unsigned int idx = h_rtag.data[tag];
            return idx;
            }
        //! Get the orientation of a particle with a given tag
        Scalar4 getOrientation(unsigned int tag) const
            {
            assert(tag < getN());
            ArrayHandle< Scalar4 > h_orientation(m_orientation, access_location::host, access_mode::read);
            ArrayHandle< unsigned int> h_rtag(m_rtag, access_location::host, access_mode::read);
            unsigned int idx = h_rtag.data[tag];
            return h_orientation.data[idx];
            }
        //! Get the inertia tensor of a particle with a given tag
        const InertiaTensor& getInertiaTensor(unsigned int tag) const
            {
            return m_inertia_tensor[tag];
            }
        //! Get the net force / energy on a given particle
        Scalar4 getPNetForce(unsigned int tag) const
            {
            assert(tag < getN());
            ArrayHandle< Scalar4 > h_net_force(m_net_force, access_location::host, access_mode::read);
            ArrayHandle< unsigned int> h_rtag(m_rtag, access_location::host, access_mode::read);
            unsigned int idx = h_rtag.data[tag];
            return h_net_force.data[idx];
            }
        //! Get the net torque on a given particle
        Scalar4 getNetTorque(unsigned int tag)
            {
            assert(tag < getN());
            ArrayHandle< Scalar4 > h_net_torque(m_net_force, access_location::host, access_mode::read);
            ArrayHandle< unsigned int> h_rtag(m_rtag, access_location::host, access_mode::read);
            unsigned int idx = h_rtag.data[tag];
            return h_net_torque.data[idx];
            }

        //! Set the current position of a particle
        void setPosition(unsigned int tag, const Scalar3& pos)
            {
            assert(tag < getN());
            ArrayHandle< Scalar4 > h_pos(m_pos, access_location::host, access_mode::readwrite);
            ArrayHandle< unsigned int> h_rtag(m_rtag, access_location::host, access_mode::read);
            unsigned int idx = h_rtag.data[tag];
            h_pos.data[idx].x = pos.x; h_pos.data[idx].y = pos.y; h_pos.data[idx].z = pos.z;
            }
        //! Set the current velocity of a particle
        void setVelocity(unsigned int tag, const Scalar3& vel)
            {
            assert(tag < getN());
            ArrayHandle< Scalar4 > h_vel(m_vel, access_location::host, access_mode::readwrite);
            ArrayHandle< unsigned int> h_rtag(m_rtag, access_location::host, access_mode::read);
            unsigned int idx = h_rtag.data[tag];
            h_vel.data[idx].x = vel.x; h_vel.data[idx].y = vel.y; h_vel.data[idx].z = vel.z;
            }
        //! Set the current image flags of a particle
        void setImage(unsigned int tag, const int3& image)
            {
            assert(tag < getN());
            ArrayHandle< int3 > h_image(m_image, access_location::host, access_mode::readwrite);
            ArrayHandle< unsigned int> h_rtag(m_rtag, access_location::host, access_mode::read);
            unsigned int idx = h_rtag.data[tag];
            h_image.data[idx] = image;
            }
        //! Set the current charge of a particle
        void setCharge(unsigned int tag, Scalar charge)
            {
            assert(tag < getN());
            ArrayHandle< Scalar > h_charge(m_charge, access_location::host, access_mode::readwrite);
            ArrayHandle< unsigned int> h_rtag(m_rtag, access_location::host, access_mode::read);
            unsigned int idx = h_rtag.data[tag];
            h_charge.data[idx] = charge;
            }
        //! Set the current mass of a particle
        void setMass(unsigned int tag, Scalar mass)
            {
            assert(tag < getN());
            ArrayHandle< Scalar4 > h_vel(m_vel, access_location::host, access_mode::readwrite);
            ArrayHandle< unsigned int> h_rtag(m_rtag, access_location::host, access_mode::read);
            unsigned int idx = h_rtag.data[tag];
            h_vel.data[idx].w = mass;
            }
        //! Set the current diameter of a particle
        void setDiameter(unsigned int tag, Scalar diameter)
            {
            assert(tag < getN());
            ArrayHandle< Scalar > h_diameter(m_diameter, access_location::host, access_mode::readwrite);
            ArrayHandle< unsigned int> h_rtag(m_rtag, access_location::host, access_mode::read);
            unsigned int idx = h_rtag.data[tag];
            h_diameter.data[idx] = diameter;
            }
        //! Set the current diameter of a particle
        void setBody(unsigned int tag, int body)
            {
            assert(tag < getN());
            ArrayHandle< unsigned int > h_body(m_body, access_location::host, access_mode::readwrite);
            ArrayHandle< unsigned int> h_rtag(m_rtag, access_location::host, access_mode::read);
            unsigned int idx = h_rtag.data[tag];
            h_body.data[idx] = body;
            }
        //! Set the current type of a particle
        void setType(unsigned int tag, unsigned int typ)
            {
            assert(tag < getN());
            assert(typ < getNTypes());
            ArrayHandle< Scalar4 > h_pos(m_pos, access_location::host, access_mode::readwrite);
            ArrayHandle< unsigned int> h_rtag(m_rtag, access_location::host, access_mode::read);
            unsigned int idx = h_rtag.data[tag];
            h_pos.data[idx].w = __int_as_scalar(typ);
            }
        //! Set the orientation of a particle with a given tag
        void setOrientation(unsigned int tag, const Scalar4& orientation)
            {
            assert(tag < getN());
            ArrayHandle< Scalar4 > h_orientation(m_orientation, access_location::host, access_mode::readwrite);
            ArrayHandle< unsigned int> h_rtag(m_rtag, access_location::host, access_mode::read);
            unsigned int idx = h_rtag.data[tag];
            h_orientation.data[idx] = orientation;
            }
        //! Get the inertia tensor of a particle with a given tag
        void setInertiaTensor(unsigned int tag, const InertiaTensor& tensor)
            {
            m_inertia_tensor[tag] = tensor;
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

    private:
        BoxDim m_box;                               //!< The simulation box
        boost::shared_ptr<ExecutionConfiguration> m_exec_conf; //!< The execution configuration
        void *m_data;                               //!< Raw data allocated
        size_t m_nbytes;                            //!< Number of bytes allocated
        unsigned int m_ntypes;                      //!< Number of particle types
        
        std::vector<std::string> m_type_mapping;    //!< Mapping between particle type indices and names
        
        boost::signal<void ()> m_sort_signal;       //!< Signal that is triggered when particles are sorted in memory
        boost::signal<void ()> m_boxchange_signal;  //!< Signal that is triggered when the box size changes

        unsigned int m_nparticles;                  //!< number of particles

        // per-particle data
        GPUArray<Scalar4> m_pos;                    //!< particle positions and types
        GPUArray<Scalar4> m_vel;                    //!< particle velocities and masses
        GPUArray<Scalar3> m_accel;                  //!<  particle accelerations
        GPUArray<Scalar> m_charge;                  //!<  particle charges
        GPUArray<Scalar> m_diameter;                //!< particle diameters
        GPUArray<int3> m_image;                     //!< particle images
        GPUArray<unsigned int> m_tag;               //!< particle tags
        GPUArray<unsigned int> m_rtag;              //!< reverse lookup tags
        GPUArray<unsigned int> m_body;              //!< rigid body ids

        boost::shared_ptr<Profiler> m_prof;         //!< Pointer to the profiler. NULL if there is no profiler.
        
        GPUArray< Scalar4 > m_net_force;             //!< Net force calculated for each particle
        GPUArray< Scalar > m_net_virial;             //!< Net virial calculated for each particle (2D GPU array of dimensions 6*number of particles)
        GPUArray< Scalar4 > m_net_torque;            //!< Net torque calculated for each particle
        GPUArray< Scalar4 > m_orientation;           //!< Orientation quaternion for each particle (ignored if not anisotropic)
        std::vector< InertiaTensor > m_inertia_tensor; //!< Inertia tensor for each particle
        
        PDataFlags m_flags;                          //!< Flags identifying which optional fields are valid
        
#ifdef ENABLE_CUDA
        //! Simple type for identifying where the most up to date particle data is
        gpu_boxsize m_gpu_box;              //!< Mirror structure of m_box for the GPU
#endif
        
        //! Helper function to allocate CPU data
        void allocate(unsigned int N);

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

