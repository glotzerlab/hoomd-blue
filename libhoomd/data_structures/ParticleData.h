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
// Maintainer: joaander

/*! \file ParticleData.h
    \brief Defines the ParticleData class and associated utilities
*/

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
#include "gpu_settings.h"
#endif

#include "ExecutionConfiguration.h"

#include <boost/shared_ptr.hpp>
#include <boost/signals.hpp>
#include <boost/function.hpp>
#include <boost/utility.hpp>

#include <stdlib.h>
#include <vector>
#include <string>

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

// Forward declaration of IntegratorData
class IntegratorData;

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

//! Stores box dimensions
/*! All particles in the ParticleData structure are inside of a box. This struct defines
    that box. Inside is defined as x >= xlo && x < xhi, and similarly for y and z.
    \note Requirements state that xhi = -xlo, and the same goes for y and z
    \ingroup data_structs
*/
struct BoxDim
    {
    Scalar xlo; //!< Minimum x coord of the box
    Scalar xhi; //!< Maximum x coord of the box
    Scalar ylo; //!< Minimum y coord of the box
    Scalar yhi; //!< Maximum y coord of the box
    Scalar zlo; //!< Minimum z coord of the box
    Scalar zhi; //!< Maximum z coord of the box
    
    //! Constructs a useless box
    BoxDim();
    //! Constructs a box from -Len/2 to Len/2
    BoxDim(Scalar Len);
    //! Constructs a box from -Len_x/2 to Len_x/2 for each dimension x
    BoxDim(Scalar Len_x, Scalar Len_y, Scalar Len_z);
    };

//! Sentinel value in \a body to signify that this particle does not belong to a rigid body
const unsigned int NO_BODY = 0xffffffff;

//! Structure of arrays containing the particle data
/*! Once acquired, the user of the ParticleData gets access to the data arrays
    through this structure.
    Assumptions that the user of this data structure can make:
        - None of the arrays alias each other
        - No other code will be accessing these arrays in parallel (data arrays can
            only be acquired once at a time).
        - All particles are inside the box (see BoxDim)
        - Each rtag element refers uniquely to a single particle (care should be taken
            by the user not to to clobber this)

    More importantly, there are some assumptions that cannot be made:
        - Data may be in a different order next time the arrays are acquired
        - Pointers may be in a different location in memory next time the arrays are acquired
        - Anything updating these arrays CANNOT move particles outside the box

    \note Most of the data structures store properties for each particle. For example
          x[i] is the c-coordinate of particle i. The one exception is the rtag array.
          Instead of rtag[i] being the tag of particle i, rtag[tag] is the index i itself.

    Values in the type array can range from 0 to ParticleData->getNTypes()-1.
    \ingroup data_structs
*/
struct ParticleDataArrays
    {
    //! Zeros pointers
    ParticleDataArrays();
    
    unsigned int nparticles;    //!< Number of particles in the arrays
    Scalar * __restrict__ x;    //!< array of x-coordinates
    Scalar * __restrict__ y;    //!< array of y-coordinates
    Scalar * __restrict__ z;    //!< array of z-coordinates
    Scalar * __restrict__ vx;   //!< array of x-component of velocities
    Scalar * __restrict__ vy;   //!< array of y-component of velocities
    Scalar * __restrict__ vz;   //!< array of z-component of velocities
    Scalar * __restrict__ ax;   //!< array of x-component of acceleration
    Scalar * __restrict__ ay;   //!< array of y-component of acceleration
    Scalar * __restrict__ az;   //!< array of z-component of acceleration
    Scalar * __restrict__ charge;   //!< array of charges
    Scalar * __restrict__ mass; //!< array of particle masses
    Scalar * __restrict__ diameter; //!< array of particle diameters
    int * __restrict__ ix;  //!< array of x-component of images
    int * __restrict__ iy;  //!< array of x-component of images
    int * __restrict__ iz;  //!< array of x-component of images
    
    unsigned int * __restrict__ body; //!< Rigid body index this particle belongs to (NO_BODY if not in a rigid body)
    unsigned int * __restrict__ type; //!< Type index of each particle
    unsigned int * __restrict__ rtag; //!< Reverse-lookup tag.
    unsigned int * __restrict__ tag;  //!< Forward-lookup tag.
    };

//! Read only arrays
/*! This is the same as ParticleDataArrays, but has const pointers to prevent
    code with read-only access to write to the data arrays
    \ingroup data_structs
 */
struct ParticleDataArraysConst
    {
    //! Zeros pointers
    ParticleDataArraysConst();
    
    unsigned int nparticles;    //!< Number of particles in the arrays
    Scalar const * __restrict__ x;  //!< array of x-coordinates
    Scalar const * __restrict__ y;  //!< array of y-coordinates
    Scalar const * __restrict__ z;  //!< array of z-coordinates
    Scalar const * __restrict__ vx; //!< array of x-component of velocities
    Scalar const * __restrict__ vy; //!< array of y-component of velocities
    Scalar const * __restrict__ vz; //!< array of z-component of velocities
    Scalar const * __restrict__ ax; //!< array of x-component of acceleration
    Scalar const * __restrict__ ay; //!< array of y-component of acceleration
    Scalar const * __restrict__ az; //!< array of z-component of acceleration
    Scalar const * __restrict__ charge; //!< array of charges
    Scalar const * __restrict__ mass;   //!< array of particle masses
    Scalar const * __restrict__ diameter;   //!< array of particle diameters
    int const * __restrict__ ix;    //!< array of x-component of images
    int const * __restrict__ iy;    //!< array of x-component of images
    int const * __restrict__ iz;    //!< array of x-component of images
    
    unsigned int const * __restrict__ body; //!< Rigid body index this particle belongs to (NO_BODY if not in a rigid body)
    unsigned int const * __restrict__ type; //!< Type index of each particle
    unsigned int const * __restrict__ rtag; //!< Reverse-lookup tag.
    unsigned int const * __restrict__ tag;  //!< Forward-lookup tag.
    };

//! Abstract interface for initializing a ParticleData
/*! A ParticleDataInitializer should only be used with the appropriate constructor
    of ParticleData(). That constructure calls the methods of this class to determine
    the number of particles, number of particle types, the simulation box, and then
    initializes itself. Then initArrays() is called on a set of acquired
    ParticleDataArrays which the initializer is to fill out.

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
        
        //! Initializes the particle data arrays
        virtual void initArrays(const ParticleDataArrays &pdata) const = 0;
        
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
        
    };

//! Manages all of the data arrays for the particles
/*! ParticleData stores and manages particle coordinates, velocities, accelerations, type,
    and tag information. This data must be available both via the CPU and GPU memories.
    All copying of data back and forth from the GPU is accomplished transparently. To access
    the particle data for read-only purposes: call acquireReadOnly() for CPU access or
    acquireReadOnlyGPU() for GPU access. Similarly, if any values in the data are to be
    changed, data pointers can be gotten with: acquireReadWrite() for the CPU and
    acquireReadWriteGPU() for the GPU. A single ParticleData cannot be acquired multiple
    times without releasing it. Call release() to do so. An assert() will fail in debug
    builds if the ParticleData is acquired more than once without being released.

    For performance reasons, data is stored as simple arrays. Once ParticleDataArrays
    (or the const counterpart) has been acquired, the coordinates of the particle with
    <em>index</em> \c i can be accessed with <code>arrays.x[i]</code>, <code>arrays.y[i]</code>,
    and <code>arrays.z[i]</code> where \c i runs from 0 to <code>arrays.nparticles</code>.

    Velocities can similarly be accessed through the members vx,vy, and vz

    \warning Particles can and will be rearranged in the arrays throughout a simulation.
    So, a particle that was once at index 5 may be at index 123 the next time the data
    is acquired. Individual particles can be tracked through all these changes by their tag.
    <code>arrays.tag[i]</code> identifies the tag of the particle that currently has index
    \c i, and the index of a particle with tag \c tag can be read from <code>arrays.rtag[tag]</code>.

    In order to help other classes deal with particles changing indices, any class that
    changes the order must call notifyParticleSort(). Any class interested in being notified
    can subscribe to the signal by calling connectParticleSort().

    \note When writing to the particle data, particles must not be moved outside the box.
    In debug builds, any aquire will fail an assertion if this is done.
    \ingroup data_structs
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
        //! Set the simulation box
        void setBox(const BoxDim &box);

        //! Access the execution configuration
        boost::shared_ptr<const ExecutionConfiguration> getExecConf()
            {
            return m_exec_conf;
            }
            
        //! Get the number of particles
        /*! \return Number of particles in the box
        */
        unsigned int getN() const
            {
            return m_arrays.nparticles;
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
            for (unsigned int i = 0; i < m_arrays.nparticles; i++) if (m_arrays.diameter[i] > maxdiam) maxdiam = m_arrays.diameter[i];
            return maxdiam;
            }
            
        //! Acquire read access to the particle data
        const ParticleDataArraysConst& acquireReadOnly();
        //! Acquire read/write access to the particle data
        const ParticleDataArrays& acquireReadWrite();
                
#ifdef ENABLE_CUDA
        //! Acquire read access to the particle data on the GPU
        gpu_pdata_arrays& acquireReadOnlyGPU();
        //! Acquire read/write access to the particle data on the GPU
        gpu_pdata_arrays& acquireReadWriteGPU();
        
        //! Get the box for the GPU
        /*! \returns Box dimensions suitable for passing to the GPU code
        */
        const gpu_boxsize& getBoxGPU()
            {
            return m_gpu_box;
            }
            
#endif
        
        //! Release the acquired data
        void release();
        
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
        unsigned int getTypeByName(const std::string &name);
        
        //! Gets the name of a given particle type index
        std::string getNameByType(unsigned int type);
        
        //! Get the net force array
        const GPUArray< Scalar4 >& getNetForce() const { return m_net_force; }
        
        //! Get the net virial array
        const GPUArray< Scalar >& getNetVirial() const { return m_net_virial; }
        
        //! Get the current position of a particle
        Scalar3 getPosition(unsigned int tag)
            {
            assert(tag < getN());
            acquireReadOnly();
            unsigned int idx = m_arrays.rtag[tag];
            Scalar3 result = make_scalar3(m_arrays.x[idx], m_arrays.y[idx], m_arrays.z[idx]);
            release();
            return result;
            }
        //! Get the current velocity of a particle
        Scalar3 getVelocity(unsigned int tag)
            {
            assert(tag < getN());
            acquireReadOnly();
            unsigned int idx = m_arrays.rtag[tag];
            Scalar3 result = make_scalar3(m_arrays.vx[idx], m_arrays.vy[idx], m_arrays.vz[idx]);
            release();
            return result;
            }
        //! Get the current acceleration of a particle
        Scalar3 getAcceleration(unsigned int tag)
            {
            assert(tag < getN());
            acquireReadOnly();
            unsigned int idx = m_arrays.rtag[tag];
            Scalar3 result = make_scalar3(m_arrays.ax[idx], m_arrays.ay[idx], m_arrays.az[idx]);
            release();
            return result;
            }
        //! Get the current image flags of a particle
        uint3 getImage(unsigned int tag)
            {
            assert(tag < getN());
            acquireReadOnly();
            unsigned int idx = m_arrays.rtag[tag];
            uint3 result = make_uint3(m_arrays.ix[idx], m_arrays.iy[idx], m_arrays.iz[idx]);
            release();
            return result;
            }
        //! Get the current charge of a particle
        Scalar getCharge(unsigned int tag)
            {
            assert(tag < getN());
            acquireReadOnly();
            unsigned int idx = m_arrays.rtag[tag];
            Scalar result = m_arrays.charge[idx];
            release();
            return result;
            }
        //! Get the current mass of a particle
        Scalar getMass(unsigned int tag)
            {
            assert(tag < getN());
            acquireReadOnly();
            unsigned int idx = m_arrays.rtag[tag];
            Scalar result = m_arrays.mass[idx];
            release();
            return result;
            }
        //! Get the current diameter of a particle
        Scalar getDiameter(unsigned int tag)
            {
            assert(tag < getN());
            acquireReadOnly();
            unsigned int idx = m_arrays.rtag[tag];
            Scalar result = m_arrays.diameter[idx];
            release();
            return result;
            }
        //! Get the current diameter of a particle
        unsigned int getBody(unsigned int tag)
            {
            assert(tag < getN());
            acquireReadOnly();
            unsigned int idx = m_arrays.rtag[tag];
            unsigned int result = m_arrays.body[idx];
            release();
            return result;
            }
        //! Get the current type of a particle
        unsigned int getType(unsigned int tag)
            {
            assert(tag < getN());
            acquireReadOnly();
            unsigned int idx = m_arrays.rtag[tag];
            unsigned int result = m_arrays.type[idx];
            release();
            return result;
            }

        //! Get the current index of a particle with a given tag
        unsigned int getRTag(unsigned int tag)
            {
            assert(tag < getN());
            acquireReadOnly();
            unsigned int idx = m_arrays.rtag[tag];
            release();
            return idx;
            }

        //! Set the current position of a particle
        void setPosition(unsigned int tag, const Scalar3& pos)
            {
            assert(tag < getN());
            acquireReadWrite();
            unsigned int idx = m_arrays.rtag[tag];
            m_arrays.x[idx] = pos.x; m_arrays.y[idx] = pos.y; m_arrays.z[idx] = pos.z;
            release();
            }
        //! Set the current velocity of a particle
        void setVelocity(unsigned int tag, const Scalar3& vel)
            {
            assert(tag < getN());
            acquireReadWrite();
            unsigned int idx = m_arrays.rtag[tag];
            m_arrays.vx[idx] = vel.x; m_arrays.vy[idx] = vel.y; m_arrays.vz[idx] = vel.z;
            release();
            }
        //! Set the current image flags of a particle
        void setImage(unsigned int tag, const uint3& image)
            {
            assert(tag < getN());
            acquireReadWrite();
            unsigned int idx = m_arrays.rtag[tag];
            m_arrays.ix[idx] = image.x; m_arrays.iy[idx] = image.y; m_arrays.iz[idx] = image.z;
            release();
            }
        //! Set the current charge of a particle
        void setCharge(unsigned int tag, Scalar charge)
            {
            assert(tag < getN());
            acquireReadWrite();
            unsigned int idx = m_arrays.rtag[tag];
            m_arrays.charge[idx] = charge;
            release();
            }
        //! Set the current mass of a particle
        void setMass(unsigned int tag, Scalar mass)
            {
            assert(tag < getN());
            acquireReadWrite();
            unsigned int idx = m_arrays.rtag[tag];
            m_arrays.mass[idx] = mass;
            release();
            }
        //! Set the current diameter of a particle
        void setDiameter(unsigned int tag, Scalar diameter)
            {
            assert(tag < getN());
            acquireReadWrite();
            unsigned int idx = m_arrays.rtag[tag];
            m_arrays.diameter[idx] = diameter;
            release();
            }
        //! Set the current diameter of a particle
        void setBody(unsigned int tag, unsigned int body)
            {
            assert(tag < getN());
            acquireReadWrite();
            unsigned int idx = m_arrays.rtag[tag];
            m_arrays.body[idx] = body;
            release();
            }
        //! Get the current type of a particle
        void setType(unsigned int tag, unsigned int typ)
            {
            assert(tag < getN());
            assert(typ < getNTypes());
            acquireReadWrite();
            unsigned int idx = m_arrays.rtag[tag];
            m_arrays.type[idx] = typ;
            release();
            }

    private:
        BoxDim m_box;                               //!< The simulation box
        boost::shared_ptr<ExecutionConfiguration> m_exec_conf; //!< The execution configuration
        void *m_data;                               //!< Raw data allocated
        size_t m_nbytes;                            //!< Number of bytes allocated
        unsigned int m_ntypes;                      //!< Number of particle types
        
        bool m_acquired;                            //!< Flag to track if data has been acquired
        std::vector<std::string> m_type_mapping;    //!< Mapping between particle type indices and names
        
        boost::signal<void ()> m_sort_signal;       //!< Signal that is triggered when particles are sorted in memory
        boost::signal<void ()> m_boxchange_signal;  //!< Signal that is triggered when the box size changes
        
        ParticleDataArrays m_arrays;                //!< Pointers into m_data for particle access
        ParticleDataArraysConst m_arrays_const;     //!< Pointers into m_data for const particle access
        boost::shared_ptr<Profiler> m_prof;         //!< Pointer to the profiler. NULL if there is no profiler.
        
        GPUArray< Scalar4 > m_net_force;             //!< Net force calculated for each particle
        GPUArray< Scalar > m_net_virial;             //!< Net virial calculated for each particle
        
#ifdef ENABLE_CUDA
        
        //! Simple type for identifying where the most up to date particle data is
        enum DataLocation
            {
            cpu,    //!< Particle data was last modified on the CPU
            cpugpu, //!< CPU and GPU contain identical data
            gpu     //!< Particle data was last modified on the GPU
            };
            
        DataLocation m_data_location;       //!< Where the most recently modified particle data lives
        bool m_readwrite_gpu;               //!< Flag to indicate the last acquire was readwriteGPU
        gpu_pdata_arrays m_gpu_pdata;       //!< Stores the pointers to memory on the GPU
        gpu_boxsize m_gpu_box;              //!< Mirror structure of m_box for the GPU
        float * m_d_staging;                //!< Staging array (device memory) where uninterleaved data is copied to/from.
        float4 *m_h_staging;                //!< Staging array (host memory) to copy interleaved data to
        unsigned int m_uninterleave_pitch;  //!< Remember the pitch between x,y,z,type in the uninterleaved data
        unsigned int m_single_xarray_bytes; //!< Remember the number of bytes allocated for a single float array
        
        //! Helper function to move data from the host to the device
        void hostToDeviceCopy();
        //! Helper function to move data from the device to the host
        void deviceToHostCopy();
        
#endif
        
        //! Helper function to allocate CPU data
        void allocate(unsigned int N);
        //! Deallocates data
        void deallocate();
        //! Helper function to check that particles are in the box
        bool inBox(bool need_aquire);
    };

//! Exports the BoxDim class to python
void export_BoxDim();
//! Exports ParticleDataInitializer to python
void export_ParticleDataInitializer();
//! Exports ParticleData to python
void export_ParticleData();

#endif

#ifdef WIN32
#pragma warning( pop )
#endif

