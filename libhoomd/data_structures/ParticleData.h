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
#include "GPUVector.h"

#ifdef ENABLE_CUDA
#include "ParticleData.cuh"
#endif

#include "ExecutionConfiguration.h"
#include "BoxDim.h"

#include <boost/shared_ptr.hpp>
#include <boost/signals2.hpp>
#include <boost/function.hpp>
#include <boost/utility.hpp>
#include <boost/dynamic_bitset.hpp>

#ifdef ENABLE_MPI
#include "Index1D.h"
#endif

#include "DomainDecomposition.h"

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

class WallData;

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

//! Sentinel value in \a r_tag to signify that this particle is not currently present on the local processor
const unsigned int NOT_LOCAL = 0xffffffff;

//! Handy structure for passing around per-particle data
/*! A snapshot is used for two purposes:
 * - Initializing the ParticleData
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
        : size(0)
        {
        }

    //! constructor
    /*! \param N number of particles to allocate memory for
     */
    SnapshotParticleData(unsigned int N);

    //! Resize the snapshot
    /*! \param N number of particles in snapshot
     */
    void resize(unsigned int N);

    //! Validate the snapshot
    /*! \returns true if the number of elements is consistent
     */
    bool validate() const;

    //! Replicate this snapshot
    /*! \param nx Number of times to replicate the system along the x direction
     *  \param ny Number of times to replicate the system along the y direction
     *  \param nz Number of times to replicate the system along the z direction
     *  \param old_box Old box dimensions
     *  \param new_box Dimensions of replicated box
     */
    void replicate(unsigned int nx, unsigned int ny, unsigned int nz,
        const BoxDim& old_box, const BoxDim& new_box);

    std::vector<Scalar3> pos;       //!< positions
    std::vector<Scalar3> vel;       //!< velocities
    std::vector<Scalar3> accel;     //!< accelerations
    std::vector<unsigned int> type; //!< types
    std::vector<Scalar> mass;       //!< masses
    std::vector<Scalar> charge;     //!< charges
    std::vector<Scalar> diameter;   //!< diameters
    std::vector<int3> image;        //!< images
    std::vector<unsigned int> body; //!< body ids
    std::vector<Scalar4> orientation; //!< orientations
    std::vector<InertiaTensor> inertia_tensor; //!< Moments of inertia

    unsigned int size;              //!< number of particles in this snapshot
    std::vector<std::string> type_mapping; //!< Mapping between particle type ids and names
    };

//! Structure to store packed particle data
/* pdata_element is used for compact storage of particle data, mainly for communication.
 */
struct pdata_element
    {
    Scalar4 pos;               //!< Position
    Scalar4 vel;               //!< Velocity
    Scalar3 accel;             //!< Acceleration
    Scalar charge;             //!< Charge
    Scalar diameter;           //!< Diameter
    int3 image;                //!< Image
    unsigned int body;         //!< Body id
    Scalar4 orientation;       //!< Orientation
    unsigned int tag;          //!< global tag
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
    The tag of a particle is stored in the \c m_tag array, and the ith element contains the tag of the particle
    with index i. Conversely, the the index of a particle with tag \c tag can be read from
    the element at position \c tag in the a \c m_rtag array.

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

    ## Parallel simulations

    In a parallel simulation, the ParticleData contains he local particles only, and getN() returns the current number of
    \a local particles. The method getNGlobal() can be used to query the \a global number of particles on all processors.

    During the simulation particles may enter or leave the box, therefore the number of \a local particles may change.
    To account for this, the size of the particle data arrays is dynamically updated using amortized doubling of the array sizes. To add particles to
    the domain, the addParticles() method is called, and the arrays are resized if necessary. Particles are retrieved
    and removed from the local particle data arrays using removeParticles(). To flag particles for removal, set the
    communication flag (m_comm_flags) for that particle to a non-zero value.

    In addition, since many other classes maintain internal arrays holding data for every particle (such as neighbor lists etc.), these
    arrays need to be resized, too, if the particle number changes. Everytime the particle data arrays are reallocated, a
    maximum particle number change signal is triggered. Other classes can subscribe to this signal using connectMaxParticleNumberChange().
    They may use the current maxium size of the particle arrays, which is returned by getMaxN().  This size changes only infrequently
    (by amortized array resizing). Note that getMaxN() can return a higher number
    than the actual number of particles.

    Particle data also stores temporary particles ('ghost atoms'). These are added after the local particle data (i.e. with indices
    starting at getN()). It keeps track of those particles using the addGhostParticles() and removeAllGhostParticles() methods.
    The caller is responsible for updating the particle data arrays with the ghost particle information.

    ## Anisotropic particles

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

    ## Origin shifting

    Parallel MC simulations randomly translate all particles by a fixed vector at periodic intervals. This motion
    is not physical, it is merely the easiest way to shift the origin of the cell list / domain decomposition
    boundaries. Analysis routines (i.e. MSD) and movies are complicated by the random motion of all particles.

    ParticleData can track this origin and subtract it from all particles. This subtraction is done when taking a
    snapshot. Putting the subtraction there naturally puts the correction there for all analysis routines and file I/O
    while leaving the shifted particles in place for computes, updaters, and integrators. On the restoration from
    a snapshot, the origin needs to be cleared.

    Two routines support this: translateOrigin() and resetOrigin(). The position of the origin is tracked by
    ParticleData internally. translateOrigin() moves it by a given vector. resetOrigin() zeroes it. TODO: This might
    not be sufficient for simulations where the box size changes. We'll see in testing.
*/
class ParticleData : boost::noncopyable
    {
    public:
        //! Construct with N particles in the given box
        ParticleData(unsigned int N,
                     const BoxDim &global_box,
                     unsigned int n_types,
                     boost::shared_ptr<ExecutionConfiguration> exec_conf,
                     boost::shared_ptr<DomainDecomposition> decomposition
                        = boost::shared_ptr<DomainDecomposition>()
                     );

        //! Construct using a ParticleDataSnapshot
        ParticleData(const SnapshotParticleData& snapshot,
                     const BoxDim& global_box,
                     boost::shared_ptr<ExecutionConfiguration> exec_conf,
                     boost::shared_ptr<DomainDecomposition> decomposition
                        = boost::shared_ptr<DomainDecomposition>()
                     );

        //! Destructor
        virtual ~ParticleData();

        //! Get the simulation box
        const BoxDim& getBox() const;

        //! Set the global simulation box
        void setGlobalBox(const BoxDim &box);

        //! Set the global simulation box Lengths
        void setGlobalBoxL(const Scalar3 &L)
            {
            BoxDim box(L);
            setGlobalBox(box);
            }

        //! Get the global simulation box
        const BoxDim& getGlobalBox() const;

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
            return m_type_mapping.size();
            }

        //! Get the origin for the particle system
        /*! \return origin of the system
        */
        Scalar3 getOrigin()
            {
            return m_origin;
            }

        //! Get the origin image for the particle system
        /*! \return image of the origin of the system
        */
        int3 getOriginImage()
            {
            return m_o_image;
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

        /*!
         * Access methods to stand-by arrays for fast swapping in of reordered particle data
         *
         * \warning An array that is swapped in has to be completely initialized.
         *          In parallel simulations, the ghost data needs to be initalized as well,
         *          or all ghosts need to be removed and re-initialized before and after reordering.
         *
         * USAGE EXAMPLE:
         * \code
         * m_comm->migrateParticles(); // migrate particles and remove all ghosts
         *     {
         *      ArrayHandle<Scalar4> h_pos_alt(m_pdata->getAltPositions(), access_location::host, access_mode::overwrite)
         *      ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
         *      for (int i=0; i < getN(); ++i)
         *          h_pos_alt.data[i] = h_pos.data[permutation[i]]; // apply some permutation
         *     }
         * m_pdata->swapPositions(); // swap in reordered data at no extra cost
         * notifyParticleSort();     // ensures that ghosts will be restored at next communication step
         * \endcode
         */

        //! Return positions and types (alternate array)
        const GPUArray< Scalar4 >& getAltPositions() const { return m_pos_alt; }

        //! Swap in positions
        inline void swapPositions() { m_pos.swap(m_pos_alt); }

        //! Return velocities and masses (alternate array)
        const GPUArray< Scalar4 >& getAltVelocities() const { return m_vel_alt; }

        //! Swap in velocities
        inline void swapVelocities() { m_vel.swap(m_vel_alt); }

        //! Return accelerations (alternate array)
        const GPUArray< Scalar3 >& getAltAccelerations() const { return m_accel_alt; }

        //! Swap in accelerations
        inline void swapAccelerations() { m_accel.swap(m_accel_alt); }

        //! Return charges (alternate array)
        const GPUArray< Scalar >& getAltCharges() const { return m_charge_alt; }

        //! Swap in accelerations
        inline void swapCharges() { m_charge.swap(m_charge_alt); }

        //! Return diameters (alternate array)
        const GPUArray< Scalar >& getAltDiameters() const { return m_diameter_alt; }

        //! Swap in diameters
        inline void swapDiameters() { m_diameter.swap(m_diameter_alt); }

        //! Return images (alternate array)
        const GPUArray< int3 >& getAltImages() const { return m_image_alt; }

        //! Swap in images
        inline void swapImages() { m_image.swap(m_image_alt); }

        //! Return tags (alternate array)
        const GPUArray< unsigned int >& getAltTags() const { return m_tag_alt; }

        //! Swap in tags
        inline void swapTags() { m_tag.swap(m_tag_alt); }

        //! Return body ids (alternate array)
        const GPUArray< unsigned int >& getAltBodies() const { return m_body_alt; }

        //! Swap in bodies
        inline void swapBodies() { m_body.swap(m_body_alt); }

        //! Get the net force array (alternate array)
        const GPUArray< Scalar4 >& getAltNetForce() const { return m_net_force_alt; }

        //! Swap in net force
        inline void swapNetForce() { m_net_force.swap(m_net_force_alt); }

        //! Get the net virial array (alternate array)
        const GPUArray< Scalar >& getAltNetVirial() const { return m_net_virial_alt; }

        //! Swap in net virial
        inline void swapNetVirial() { m_net_virial.swap(m_net_virial_alt); }

        //! Get the net torque array (alternate array)
        const GPUArray< Scalar4 >& getAltNetTorqueArray() const { return m_net_torque_alt; }

        //! Swap in net torque
        inline void swapNetTorque() { m_net_torque.swap(m_net_torque_alt); }

        //! Get the orientations (alternate array)
        const GPUArray< Scalar4 >& getAltOrientationArray() const { return m_orientation_alt; }

        //! Swap in orientations
        inline void swapOrientations() { m_orientation.swap(m_orientation_alt); }

        //! Set the profiler to profile CPU<-->GPU memory copies
        /*! \param prof Pointer to the profiler to use. Set to NULL to deactivate profiling
        */
        void setProfiler(boost::shared_ptr<Profiler> prof)
            {
            m_prof=prof;
            }

        //! Connects a function to be called every time the particles are rearranged in memory
        boost::signals2::connection connectParticleSort(const boost::function<void ()> &func);

        //! Notify listeners that the particles have been rearranged in memory
        void notifyParticleSort();

        //! Connects a function to be called every time the box size is changed
        boost::signals2::connection connectBoxChange(const boost::function<void ()> &func);

        //! Connects a function to be called every time the maximum particle number changes
        boost::signals2::connection connectMaxParticleNumberChange(const boost::function< void()> &func);

        //! Connects a function to be called every time the ghost particles are updated
        boost::signals2::connection connectGhostParticleNumberChange(const boost::function< void()> &func);

        #ifdef ENABLE_MPI
        //! Connects a function to be called every time a single particle migration is requested
        boost::signals2::connection connectSingleParticleMove(
            const boost::function<void (unsigned int, unsigned int, unsigned int)> &func);
        #endif

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
        //! Get the communication flags array
        const GPUArray< unsigned int >& getCommFlags() const { return m_comm_flags; }
        #endif

        #ifdef ENABLE_MPI
        //! Find the processor that owns a particle
        unsigned int getOwnerRank(unsigned int tag) const;
        #endif

        //! Get the current position of a particle
        Scalar3 getPosition(unsigned int tag) const;

        //! Get the current velocity of a particle
        Scalar3 getVelocity(unsigned int tag) const;

        //! Get the current acceleration of a particle
        Scalar3 getAcceleration(unsigned int tag) const;

        //! Get the current image flags of a particle
        int3 getImage(unsigned int tag) const;

        //! Get the current mass of a particle
        Scalar getMass(unsigned int tag) const;

        //! Get the current diameter of a particle
        Scalar getDiameter(unsigned int tag) const;

        //! Get the current charge of a particle
        Scalar getCharge(unsigned int tag) const;

        //! Get the body id of a particle
        unsigned int getBody(unsigned int tag) const;

        //! Get the current type of a particle
        unsigned int getType(unsigned int tag) const;

        //! Get the current index of a particle with a given global tag
        inline unsigned int getRTag(unsigned int tag) const
            {
            assert(tag < m_nglobal);
            ArrayHandle< unsigned int> h_rtag(m_rtag,access_location::host, access_mode::read);
            unsigned int idx = h_rtag.data[tag];
#ifdef ENABLE_MPI
            assert(m_decomposition || idx < getN());
#endif
            assert(idx < getN() + getNGhosts() || idx == NOT_LOCAL);
            return idx;
            }

        //! Return true if particle is local (= owned by this processor)
        bool isParticleLocal(unsigned int tag) const
             {
             assert(tag < m_nglobal);
             ArrayHandle< unsigned int> h_rtag(m_rtag,access_location::host, access_mode::read);
             return h_rtag.data[tag] < getN();
             }

        //! Get the orientation of a particle with a given tag
        Scalar4 getOrientation(unsigned int tag) const;

        //! Get the inertia tensor of a particle with a given tag
        const InertiaTensor& getInertiaTensor(unsigned int tag) const
            {
            return m_inertia_tensor[tag];
            }

        //! Get the net force / energy on a given particle
        Scalar4 getPNetForce(unsigned int tag) const;

        //! Get the net torque on a given particle
        Scalar4 getNetTorque(unsigned int tag) const;

        //! Set the current position of a particle
        /*! \param move If true, particle is automatically placed into correct domain
         */
        void setPosition(unsigned int tag, const Scalar3& pos, bool move=true);

        //! Set the current velocity of a particle
        void setVelocity(unsigned int tag, const Scalar3& vel);

        //! Set the current image flags of a particle
        void setImage(unsigned int tag, const int3& image);

        //! Set the current charge of a particle
        void setCharge(unsigned int tag, Scalar charge);

        //! Set the current mass of a particle
        void setMass(unsigned int tag, Scalar mass);

        //! Set the current diameter of a particle
        void setDiameter(unsigned int tag, Scalar diameter);

        //! Set the body id of a particle
        void setBody(unsigned int tag, int body);

        //! Set the current type of a particle
        void setType(unsigned int tag, unsigned int typ);

        //! Set the orientation of a particle with a given tag
        void setOrientation(unsigned int tag, const Scalar4& orientation);

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

        //! Set the external contribution to the virial
        void setExternalVirial(unsigned int i, Scalar v)
            {
            assert(i<6);
            m_external_virial[i] = v;
            };

        //! Get the external contribution to the virial
        Scalar getExternalVirial(unsigned int i)
            {
            assert(i<6);
            return m_external_virial[i];
            }

        //! Remove the given flag
        void removeFlag(pdata_flag::Enum flag) { m_flags[flag] = false; }

        //! Initialize from a snapshot
        void initializeFromSnapshot(const SnapshotParticleData & snapshot);

        //! Take a snapshot
        void takeSnapshot(SnapshotParticleData &snapshot);

        //! Add ghost particles at the end of the local particle data
        void addGhostParticles(const unsigned int nghosts);

        //! Remove all ghost particles from system
        void removeAllGhostParticles()
            {
            m_nghosts = 0;
            }

#ifdef ENABLE_MPI
        //! Set domain decomposition information
        void setDomainDecomposition(boost::shared_ptr<DomainDecomposition> decomposition)
            {
            assert(decomposition);
            m_decomposition = decomposition;
            m_box = m_decomposition->calculateLocalBox(m_global_box);
            m_boxchange_signal();
            }

        //! Returns the domain decomin decomposition information
        boost::shared_ptr<DomainDecomposition> getDomainDecomposition()
            {
            return m_decomposition;
            }

        //! Pack particle data into a buffer
        /*! \param out Buffer into which particle data is packed
         *  \param comm_flags Buffer into which communication flags is packed
         *
         *  Packs all particles for which comm_flag>0 into a buffer
         *  and remove them from the particle data
         *
         *  The output buffers are automatically resized to accomodate the data.
         *
         *  \post The particle data arrays remain compact. Any ghost atoms
         *        are invalidated. (call removeAllGhostAtoms() before or after
         *        this method).
         */
        void removeParticles(std::vector<pdata_element>& out, std::vector<unsigned int>& comm_flags);

        //! Remove particles from local domain and add new particle data
        /*! \param in List of particle data elements to fill the particle data with
         */
        void addParticles(const std::vector<pdata_element>& in);

        #ifdef ENABLE_CUDA
        //! Pack particle data into a buffer (GPU version)
        /*! \param out Buffer into which particle data is packed
         *  \param comm_flags Buffer into which communication flags is packed
         *
         *  Pack all particles for which comm_flag >0 into a buffer
         *  and remove them from the particle data
         *
         *  The output buffers are automatically resized to accomodate the data.
         *
         *  \post The particle data arrays remain compact. Any ghost atoms
         *        are invalidated. (call removeAllGhostAtoms() before or after
         *        this method).
         */
        void removeParticlesGPU(GPUVector<pdata_element>& out, GPUVector<unsigned int>& comm_flags);

        //! Remove particles from local domain and add new particle data (GPU version)
        /*! \param in List of particle data elements to fill the particle data with
         */
        void addParticlesGPU(const GPUVector<pdata_element>& in);
        #endif // ENABLE_CUDA

#endif // ENABLE_MPI

        //! Translate the box origin
        /*! \param a vector to apply in the translation
        */
        void translateOrigin(const Scalar3& a)
            {
            m_origin += a;
            // wrap the origin back into the box to prevent it from getting too large
            m_global_box.wrap(m_origin, m_o_image);
            }

        //! Rest the box origin
        /*! \post The origin is 0,0,0
        */
        void resetOrigin()
            {
            m_origin = make_scalar3(0,0,0);
            m_o_image = make_int3(0,0,0);
            }

    private:
        BoxDim m_box;                               //!< The simulation box
        BoxDim m_global_box;                        //!< Global simulation box
        boost::shared_ptr<ExecutionConfiguration> m_exec_conf; //!< The execution configuration
#ifdef ENABLE_MPI
        boost::shared_ptr<DomainDecomposition> m_decomposition;       //!< Domain decomposition data
#endif

        std::vector<std::string> m_type_mapping;    //!< Mapping between particle type indices and names

        boost::signals2::signal<void ()> m_sort_signal;       //!< Signal that is triggered when particles are sorted in memory
        boost::signals2::signal<void ()> m_boxchange_signal;  //!< Signal that is triggered when the box size changes
        boost::signals2::signal<void ()> m_max_particle_num_signal; //!< Signal that is triggered when the maximum particle number changes
        boost::signals2::signal<void ()> m_ghost_particle_num_signal; //!< Signal that is triggered when ghost particles are added to or deleted
        boost::signals2::signal<void ()> m_global_particle_num_signal; //!< Signal that is triggered when the global number of particles changes

        #ifdef ENABLE_MPI
        boost::signals2::signal<void (unsigned int, unsigned int, unsigned int)> m_ptl_move_signal; //!< Signal when particle moves between domains
        #endif

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
        GPUArray<unsigned int> m_body;              //!< rigid body ids
        GPUArray< Scalar4 > m_orientation;          //!< Orientation quaternion for each particle (ignored if not anisotropic)
        #ifdef ENABLE_MPI
        GPUArray<unsigned int> m_comm_flags;        //!< Array of communication flags
        #endif


        /* Alternate particle data arrays are provided for fast swapping in and out of particle data
           The size of these arrays is updated in sync with the main particle data arrays.

           The primary use case is when particle data has to be re-ordered in-place, i.e.
           a temporary array would otherwise be required. Instead of writing to a temporary
           array and copying to the main particle data subsequently, the re-ordered particle
           data can be written to the alternate arrays, which are then swapped in for
           the real particle data at effectively zero cost.
         */
        GPUArray<Scalar4> m_pos_alt;                //!< particle positions and type (swap-in)
        GPUArray<Scalar4> m_vel_alt;                //!< particle velocities and masses (swap-in)
        GPUArray<Scalar3> m_accel_alt;              //!< particle accelerations (swap-in)
        GPUArray<Scalar> m_charge_alt;              //!< particle charges (swap-in)
        GPUArray<Scalar> m_diameter_alt;            //!< particle diameters (swap-in)
        GPUArray<int3> m_image_alt;                 //!< particle images (swap-in)
        GPUArray<unsigned int> m_tag_alt;           //!< particle tags (swap-in)
        GPUArray<unsigned int> m_body_alt;          //!< rigid body ids (swap-in)
        GPUArray<Scalar4> m_orientation_alt;        //!< orientations (swap-in)
        GPUArray<Scalar4> m_net_force_alt;          //!< Net force (swap-in)
        GPUArray<Scalar> m_net_virial_alt;          //!< Net virial (swap-in)
        GPUArray<Scalar4> m_net_torque_alt;         //!< Net torque (swap-in)

        boost::shared_ptr<Profiler> m_prof;         //!< Pointer to the profiler. NULL if there is no profiler.

        GPUArray< Scalar4 > m_net_force;             //!< Net force calculated for each particle
        GPUArray< Scalar > m_net_virial;             //!< Net virial calculated for each particle (2D GPU array of dimensions 6*number of particles)
        GPUArray< Scalar4 > m_net_torque;            //!< Net torque calculated for each particle
        std::vector< InertiaTensor > m_inertia_tensor; //!< Inertia tensor for each particle

        Scalar m_external_virial[6];                 //!< External potential contribution to the virial
        const float m_resize_factor;                 //!< The numerical factor with which the particle data arrays are resized
        PDataFlags m_flags;                          //!< Flags identifying which optional fields are valid

        Scalar3 m_origin;                            //!< Tracks the position of the origin of the coordinate system
        int3 m_o_image;                              //!< Tracks the origin image

        #ifdef ENABLE_CUDA
        mgpu::ContextPtr m_mgpu_context;             //!< moderngpu context
        #endif

        //! Helper function to allocate particle data
        void allocate(unsigned int N);

        //! Helper function to allocate alternate particle data
        void allocateAlternateArrays(unsigned int N);

        //! Helper function for amortized array resizing
        void resize(unsigned int new_nparticles);

        //! Helper function to reallocate particle data
        void reallocate(unsigned int max_n);

        //! Helper function to check that particles are in the box
        bool inBox();
    };


//! Exports the BoxDim class to python
void export_BoxDim();
//! Exports ParticleData to python
void export_ParticleData();
//! Export SnapshotParticleData to python
void export_SnapshotParticleData();

#endif

#ifdef WIN32
#pragma warning( pop )
#endif
