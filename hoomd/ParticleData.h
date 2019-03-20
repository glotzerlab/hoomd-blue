// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file ParticleData.h
    \brief Defines the ParticleData class and associated utilities
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __PARTICLE_DATA_H__
#define __PARTICLE_DATA_H__

#include "HOOMDMath.h"
#include "GlobalArray.h"
#include "GPUVector.h"
#include "GlobalArray.h"

#ifdef ENABLE_CUDA
#include "ParticleData.cuh"
#include "GPUPartition.cuh"
#endif

#include "ExecutionConfiguration.h"
#include "BoxDim.h"

#include "HOOMDMPI.h"

#include <memory>
#include <hoomd/extern/nano-signal-slot/nano_signal_slot.hpp>

#ifndef NVCC
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>
#endif

#ifdef ENABLE_MPI
#include "Index1D.h"
#endif

#include "DomainDecomposition.h"

#include <stdlib.h>
#include <vector>
#include <map>
#include <string>
#include <bitset>
#include <stack>

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

//! Feature-define for HOOMD API
#define HOOMD_SUPPORTS_ADD_REMOVE_PARTICLES

// Forward declaration of Profiler
class Profiler;

// Forward declaration of IntegratorData
class IntegratorData;

//! List of optional fields that can be enabled in ParticleData
struct pdata_flag
    {
    //! The enum
    enum Enum
        {
        isotropic_virial=0,        //!< Bit id in PDataFlags for the isotropic virial
        potential_energy,          //!< Bit id in PDataFlags for the potential energy
        pressure_tensor,           //!< Bit id in PDataFlags for the full virial
        rotational_kinetic_energy,  //!< Bit id in PDataFlags for the rotational kinetic energy
        external_field_virial       //!< Bit id in PDataFlags for the external virial contribution of volume change
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

//! Sentinel value in \a body to signify that this particle does not belong to a body
const unsigned int NO_BODY = 0xffffffff;

//! Unsigned value equivalent to a sign flip in a signed int. All larger values of the \a body flag indicate a floppy body (forces between are ignored, but they are integrated independently).
const unsigned int MIN_FLOPPY = 0x80000000;

//! Sentinel value in \a r_tag to signify that this particle is not currently present on the local processor
const unsigned int NOT_LOCAL = 0xffffffff;

#ifdef ENABLE_MPI
namespace cereal
    {
    //! Serialization of vec3<Real>
    template<class Archive, class Real>
    void serialize(Archive & ar, vec3<Real> & v, const unsigned int version)
        {
        ar & v.x;
        ar & v.y;
        ar & v.z;
        }

    //! Serialization of quat<Real>
    template<class Archive, class Real>
    void serialize(Archive & ar, quat<Real> & q, const unsigned int version)
        {
        // serialize both members
        ar & q.s;
        ar & q.v;
        }
    }
#endif

//! Handy structure for passing around per-particle data
/*! A snapshot is used for two purposes:
 * - Initializing the ParticleData
 * - inside an Analyzer to iterate over the current ParticleData
 *
 * Initializing the ParticleData is accomplished by first filling the particle data arrays with default values
 * (such as type, mass, diameter). Then a snapshot of this initial state is taken and passed to the
 * ParticleDataInitializer, which may modify any of the fields of the snapshot. It then returns it to
 * ParticleData, which in turn initializes its internal arrays from the snapshot using ParticleData::initializeFromSnapshot().
 *
 * To support the second scenario it is necessary that particles can be accessed in global tag order. Therefore,
 * the data in a snapshot is stored in global tag order.
 * \ingroup data_structs
 */
template <class Real>
struct PYBIND11_EXPORT SnapshotParticleData {
    //! Empty snapshot
    SnapshotParticleData()
        : size(0), is_accel_set(false)
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

    //! Insert n elements at position i
    void insert(unsigned int i, unsigned int n);

    //! Validate the snapshot
    /*! \returns true if the number of elements is consistent
     */
    bool validate() const;

    #ifdef ENABLE_MPI
    //! Broadcast the snapshot using MPI
    /*! \param root the processor to send from
        \param mpi_comm The MPI communicator
     */
    void bcast(unsigned int root, MPI_Comm mpi_comm);
    #endif

    //! Replicate this snapshot
    /*! \param nx Number of times to replicate the system along the x direction
     *  \param ny Number of times to replicate the system along the y direction
     *  \param nz Number of times to replicate the system along the z direction
     *  \param old_box Old box dimensions
     *  \param new_box Dimensions of replicated box
     */
    void replicate(unsigned int nx, unsigned int ny, unsigned int nz,
        const BoxDim& old_box, const BoxDim& new_box);

    //! Get pos as a Python object
    static pybind11::object getPosNP(pybind11::object self);
    //! Get vel as a Python object
    static pybind11::object getVelNP(pybind11::object self);
    //! Get accel as a Python object
    static pybind11::object getAccelNP(pybind11::object self);
    //! Get type as a Python object
    static pybind11::object getTypeNP(pybind11::object self);
    //! Get mass as a Python object
    static pybind11::object getMassNP(pybind11::object self);
    //! Get charge as a Python object
    static pybind11::object getChargeNP(pybind11::object self);
    //! Get diameter as a Python object
    static pybind11::object getDiameterNP(pybind11::object self);
    //! Get image as a Python object
    static pybind11::object getImageNP(pybind11::object self);
    //! Get body as a Python object
    static pybind11::object getBodyNP(pybind11::object self);
    //! Get orientation as a Python object
    static pybind11::object getOrientationNP(pybind11::object self);
    //! Get moment of inertia as a numpy array
    static pybind11::object getMomentInertiaNP(pybind11::object self);
    //! Get angular momentum as a numpy array
    static pybind11::object getAngmomNP(pybind11::object self);

    //! Get the type names for python
    pybind11::list getTypes();
    //! Set the type names from python
    void setTypes(pybind11::list types);

    std::vector< vec3<Real> > pos;             //!< positions
    std::vector< vec3<Real> > vel;             //!< velocities
    std::vector< vec3<Real> > accel;           //!< accelerations
    std::vector<unsigned int> type;            //!< types
    std::vector<Real> mass;                    //!< masses
    std::vector<Real> charge;                  //!< charges
    std::vector<Real> diameter;                //!< diameters
    std::vector<int3> image;                   //!< images
    std::vector<unsigned int> body;            //!< body ids
    std::vector< quat<Real> > orientation;     //!< orientations
    std::vector< quat<Real> > angmom;          //!< angular momentum quaternion
    std::vector< vec3<Real> > inertia;         //!< principal moments of inertia

    unsigned int size;                         //!< number of particles in this snapshot
    std::vector<std::string> type_mapping;     //!< Mapping between particle type ids and names

    bool is_accel_set;                         //!< Flag indicating if accel is set
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
    Scalar4 angmom;            //!< Angular momentum
    Scalar3 inertia;           //!< Principal moments of inertia
    unsigned int tag;          //!< global tag
    Scalar4 net_force;         //!< net force
    Scalar4 net_torque;        //!< net torque
    Scalar net_virial[6];      //!< net virial
    };

//! Manages all of the data arrays for the particles
/*! <h1> General </h1>
    ParticleData stores and manages particle coordinates, velocities, accelerations, type,
    and tag information. This data must be available both via the CPU and GPU memories.
    All copying of data back and forth from the GPU is accomplished transparently by GlobalArray.

    For performance reasons, data is stored as simple arrays. Once a handle to the particle data
    GlobalArrays has been acquired, the coordinates of the particle with
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
     - pdata_flag::external_field_virial - specify that an external virial contribution is valid

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
    arrays need to be resized, too, if the particle number changes. Every time the particle data arrays are reallocated, a
    maximum particle number change signal is triggered. Other classes can subscribe to this signal using connectMaxParticleNumberChange().
    They may use the current maximum size of the particle arrays, which is returned by getMaxN().  This size changes only infrequently
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
    must be part of a composite body (there can be single-particle bodies,
    of course) where integration methods like NVERigid will handle updating the degrees of freedom of the composite
    body and then set the constrained position, velocity, and orientation of the constituent particles.

    Particles that are part of a floppy body will have the same value of the body flag, but that value must be a
    negative number less than -1 (which is reserved as NO_BODY). Such particles do not need to be treated specially by the integrator;
    they are integrated independently of one another, but they do not interact. This lack of interaction is enforced through the neighbor
    list, in which particles that belong to the same body are excluded by default.

    To enable correct initialization of the composite body moment of inertia, each particle is also assigned
    an individual moment of inertia which is summed up correctly to determine the composite body's total moment of
    inertia.

    Access the orientation quaternion of each particle with the GlobalArray gotten from getOrientationArray(), the net
    torque with getTorqueArray(). Individual inertia tensor values can be accessed with getMomentsOfInertia() and
    setMomentsOfInertia()

    The current maximum diameter of all composite particles is stored in ParticleData and can be requested
    by the NeighborList or other classes to compute rigid body interactions correctly. The maximum value
    is updated by querying all classes that compute rigid body forces for updated values whenever needed.

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

    ## Acceleration data

    Most initialization routines do not provide acceleration data. In this case, the integrator needs to compute
    appropriate acceleration data before time step 0 for integration to be correct.

    However, the acceleration data is valid on taking/restoring a snapshot or executing additional run() commands
    and there is no need for the integrator to provide acceleration. Doing so produces incorrect results
    with some integrators (see issue #252). Future updates to gsd may enable restarting with acceleration data from
    a file.

    The solution is to store a flag in the particle data (and in the snapshot) indicating if the acceleration data
    is valid. When it is not valid, the integrator will compute accelerations and make it valid in prepRun(). When it
    is valid, the integrator will do nothing. On initialization from a snapshot, ParticleData will inherit its
    valid flag.
*/
class PYBIND11_EXPORT ParticleData
    {
    public:
        //! Construct with N particles in the given box
        ParticleData(unsigned int N,
                     const BoxDim &global_box,
                     unsigned int n_types,
                     std::shared_ptr<ExecutionConfiguration> exec_conf,
                     std::shared_ptr<DomainDecomposition> decomposition
                        = std::shared_ptr<DomainDecomposition>()
                     );

        //! Construct using a ParticleDataSnapshot
        template<class Real>
        ParticleData(const SnapshotParticleData<Real>& snapshot,
                     const BoxDim& global_box,
                     std::shared_ptr<ExecutionConfiguration> exec_conf,
                     std::shared_ptr<DomainDecomposition> decomposition
                        = std::shared_ptr<DomainDecomposition>()
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
        std::shared_ptr<const ExecutionConfiguration> getExecConf() const
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

        //! Get the current maximum number of particles
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

        //! Get the accel set flag
        /*! \returns true if the acceleration has already been set
        */
        inline bool isAccelSet()
            {
            return m_accel_set;
            }

        //! Set the accel set flag to true
        inline void notifyAccelSet()
            {
            m_accel_set = true;
            }

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
            #ifdef ENABLE_MPI
            if (m_decomposition)
                {
                MPI_Allreduce(MPI_IN_PLACE,&maxdiam, 1, MPI_HOOMD_SCALAR, MPI_MAX, m_exec_conf->getMPICommunicator());
                }
            #endif
            return maxdiam;
            }

        /*! Returns true if there are bodies in the system
         */
        bool hasBodies() const
            {
            unsigned int has_bodies = 0;
            ArrayHandle<unsigned int> h_body(getBodies(), access_location::host, access_mode::read);
            for (unsigned int i = 0; i < getN(); ++i)
                {
                if (h_body.data[i] != NO_BODY)
                    {
                    has_bodies = 1;
                    break;
                    }
                }
            #ifdef ENABLE_MPI
            if (m_decomposition)
                {
                MPI_Allreduce(MPI_IN_PLACE,
                    &has_bodies,
                    1,
                    MPI_UNSIGNED,
                    MPI_MAX,
                    m_exec_conf->getMPICommunicator());
                }
            #endif
            return has_bodies;
            }

        //! Return the maximum diameter of all registered composite particles
        Scalar getMaxCompositeParticleDiameter()
            {
            Scalar d_max = 0.0;
            m_composite_particles_signal.emit_accumulate([&](Scalar d)
                                                            {
                                                            if (d > d_max) d_max = d;
                                                            }
                                                        );

            return d_max;
            }

        //! Return positions and types
        const GlobalArray< Scalar4 >& getPositions() const { return m_pos; }

        //! Return velocities and masses
        const GlobalArray< Scalar4 >& getVelocities() const { return m_vel; }

        //! Return accelerations
        const GlobalArray< Scalar3 >& getAccelerations() const { return m_accel; }

        //! Return charges
        const GlobalArray< Scalar >& getCharges() const { return m_charge; }

        //! Return diameters
        const GlobalArray< Scalar >& getDiameters() const { return m_diameter; }

        //! Return images
        const GlobalArray< int3 >& getImages() const { return m_image; }

        //! Return tags
        const GlobalArray< unsigned int >& getTags() const { return m_tag; }

        //! Return reverse-lookup tags
        const GlobalVector< unsigned int >& getRTags() const { return m_rtag; }

        //! Return body ids
        const GlobalArray< unsigned int >& getBodies() const { return m_body; }

        /*!
         * Access methods to stand-by arrays for fast swapping in of reordered particle data
         *
         * \warning An array that is swapped in has to be completely initialized.
         *          In parallel simulations, the ghost data needs to be initialized as well,
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
        const GlobalArray< Scalar4 >& getAltPositions() const { return m_pos_alt; }

        //! Swap in positions
        inline void swapPositions() { m_pos.swap(m_pos_alt); }

        //! Return velocities and masses (alternate array)
        const GlobalArray< Scalar4 >& getAltVelocities() const { return m_vel_alt; }

        //! Swap in velocities
        inline void swapVelocities() { m_vel.swap(m_vel_alt); }

        //! Return accelerations (alternate array)
        const GlobalArray< Scalar3 >& getAltAccelerations() const { return m_accel_alt; }

        //! Swap in accelerations
        inline void swapAccelerations() { m_accel.swap(m_accel_alt); }

        //! Return charges (alternate array)
        const GlobalArray< Scalar >& getAltCharges() const { return m_charge_alt; }

        //! Swap in accelerations
        inline void swapCharges() { m_charge.swap(m_charge_alt); }

        //! Return diameters (alternate array)
        const GlobalArray< Scalar >& getAltDiameters() const { return m_diameter_alt; }

        //! Swap in diameters
        inline void swapDiameters() { m_diameter.swap(m_diameter_alt); }

        //! Return images (alternate array)
        const GlobalArray< int3 >& getAltImages() const { return m_image_alt; }

        //! Swap in images
        inline void swapImages() { m_image.swap(m_image_alt); }

        //! Return tags (alternate array)
        const GlobalArray< unsigned int >& getAltTags() const { return m_tag_alt; }

        //! Swap in tags
        inline void swapTags() { m_tag.swap(m_tag_alt); }

        //! Return body ids (alternate array)
        const GlobalArray< unsigned int >& getAltBodies() const { return m_body_alt; }

        //! Swap in bodies
        inline void swapBodies() { m_body.swap(m_body_alt); }

        //! Get the net force array (alternate array)
        const GlobalArray< Scalar4 >& getAltNetForce() const { return m_net_force_alt; }

        //! Swap in net force
        inline void swapNetForce() { m_net_force.swap(m_net_force_alt); }

        //! Get the net virial array (alternate array)
        const GlobalArray< Scalar >& getAltNetVirial() const { return m_net_virial_alt; }

        //! Swap in net virial
        inline void swapNetVirial() { m_net_virial.swap(m_net_virial_alt); }

        //! Get the net torque array (alternate array)
        const GlobalArray< Scalar4 >& getAltNetTorqueArray() const { return m_net_torque_alt; }

        //! Swap in net torque
        inline void swapNetTorque() { m_net_torque.swap(m_net_torque_alt); }

        //! Get the orientations (alternate array)
        const GlobalArray< Scalar4 >& getAltOrientationArray() const { return m_orientation_alt; }

        //! Swap in orientations
        inline void swapOrientations() { m_orientation.swap(m_orientation_alt); }

        //! Get the angular momenta (alternate array)
        const GlobalArray< Scalar4 >& getAltAngularMomentumArray() const { return m_angmom_alt; }

        //! Get the moments of inertia array (alternate array)
        const GlobalArray< Scalar3 >& getAltMomentsOfInertiaArray() const { return m_inertia_alt; }

        //! Swap in angular momenta
        inline void swapAngularMomenta() { m_angmom.swap(m_angmom_alt); }

        //! Swap in moments of inertia
        inline void swapMomentsOfInertia() { m_inertia.swap(m_inertia_alt); }

        //! Set the profiler to profile CPU<-->GPU memory copies
        /*! \param prof Pointer to the profiler to use. Set to NULL to deactivate profiling
        */
        void setProfiler(std::shared_ptr<Profiler> prof)
            {
            m_prof=prof;
            }

        //! Connects a function to be called every time the particles are rearranged in memory
        Nano::Signal<void ()>& getParticleSortSignal()
            {
            return m_sort_signal;
            }

        //! Notify listeners that the particles have been rearranged in memory
        void notifyParticleSort();

        //! Connects a function to be called every time the box size is changed
        Nano::Signal<void ()>& getBoxChangeSignal()
            {
            return m_boxchange_signal;
            }

        //! Connects a function to be called every time the global number of particles changes
        Nano::Signal< void()>& getGlobalParticleNumberChangeSignal()
            {
            return m_global_particle_num_signal;
            }

        //! Connects a function to be called every time the local maximum particle number changes
        Nano::Signal< void()>& getMaxParticleNumberChangeSignal()
            {
            return m_max_particle_num_signal;
            }

        //! Connects a function to be called every time the ghost particles become invalid
        Nano::Signal< void()>& getGhostParticlesRemovedSignal()
            {
            return m_ghost_particles_removed_signal;
            }

        #ifdef ENABLE_MPI
        //! Connects a function to be called every time a single particle migration is requested
        Nano::Signal<void (unsigned int, unsigned int, unsigned int)>& getSingleParticleMoveSignal()
            {
            return m_ptl_move_signal;
            }
        #endif

        //! Notify listeners that ghost particles have been removed
        void notifyGhostParticlesRemoved();

        //! Connects a function to be called every time the number of types changes
        Nano::Signal< void()>& getNumTypesChangeSignal()
            {
            return m_num_types_signal;
            }

        //! Connects a function to be called every time the maximum diameter of composite particles is needed
        /*! The signal slot returns the maximum diameter
         */
        Nano::Signal< Scalar()>& getCompositeParticlesSignal()
            {
            return m_composite_particles_signal;
            }

        //! Gets the particle type index given a name
        unsigned int getTypeByName(const std::string &name) const;

        //! Gets the name of a given particle type index
        std::string getNameByType(unsigned int type) const;

        //! Rename a type
        void setTypeName(unsigned int type, const std::string& name);

        //! Get the net force array
        const GlobalArray< Scalar4 >& getNetForce() const { return m_net_force; }

        //! Get the net virial array
        const GlobalArray< Scalar >& getNetVirial() const { return m_net_virial; }

        //! Get the net torque array
        const GlobalArray< Scalar4 >& getNetTorqueArray() const { return m_net_torque; }

        //! Get the orientation array
        const GlobalArray< Scalar4 >& getOrientationArray() const { return m_orientation; }

        //! Get the angular momentum array
        const GlobalArray< Scalar4 >& getAngularMomentumArray() const { return m_angmom; }

        //! Get the angular momentum array
        const GlobalArray< Scalar3 >& getMomentsOfInertiaArray() const { return m_inertia; }

        //! Get the communication flags array
        const GlobalArray< unsigned int >& getCommFlags() const { return m_comm_flags; }

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
            assert(tag < m_rtag.size());
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
             assert(tag < m_rtag.size());
             ArrayHandle< unsigned int> h_rtag(m_rtag,access_location::host, access_mode::read);
             return h_rtag.data[tag] < getN();
             }

        //! Return true if the tag is active
        bool isTagActive(unsigned int tag) const
            {
            std::set<unsigned int>::const_iterator it = m_tag_set.find(tag);
            return it != m_tag_set.end();
            }

        /*! Return the maximum particle tag in the simulation
         * \note If there are zero particles in the simulation, returns UINT_MAX
         */
        unsigned int getMaximumTag() const
            {
            if (m_tag_set.empty())
                return UINT_MAX;
            else
                return *m_tag_set.rbegin();
            }

        //! Get the orientation of a particle with a given tag
        Scalar4 getOrientation(unsigned int tag) const;

        //! Get the angular momentum of a particle with a given tag
        Scalar4 getAngularMomentum(unsigned int tag) const;

        //! Get the moment of inertia of a particle with a given tag
        Scalar3 getMomentsOfInertia(unsigned int tag) const;

        //! Get the net force / energy on a given particle
        Scalar4 getPNetForce(unsigned int tag) const;

        //! Get the net torque on a given particle
        Scalar4 getNetTorque(unsigned int tag) const;

        //! Get the net virial for a given particle
        Scalar getPNetVirial(unsigned int tag, unsigned int component) const;

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

        //! Set the orientation of a particle with a given tag
        void setAngularMomentum(unsigned int tag, const Scalar4& angmom);

        //! Set the orientation of a particle with a given tag
        void setMomentsOfInertia(unsigned int tag, const Scalar3& mom_inertia);

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

        //! Set the external contribution to the potential energy
        void setExternalEnergy(Scalar e)
            {
            m_external_energy = e;
            };

        //! Get the external contribution to the virial
        Scalar getExternalEnergy()
            {
            return m_external_energy;
            }

        //! Remove the given flag
        void removeFlag(pdata_flag::Enum flag) { m_flags[flag] = false; }

        //! Initialize from a snapshot
        template <class Real>
        void initializeFromSnapshot(const SnapshotParticleData<Real> & snapshot, bool ignore_bodies=false);

        //! Take a snapshot
        template <class Real>
        std::map<unsigned int, unsigned int> takeSnapshot(SnapshotParticleData<Real> &snapshot);

        //! Add ghost particles at the end of the local particle data
        void addGhostParticles(const unsigned int nghosts);

        //! Remove all ghost particles from system
        void removeAllGhostParticles()
            {
            // reset ghost particle number
            m_nghosts = 0;

            notifyGhostParticlesRemoved();
            }

#ifdef ENABLE_MPI
        //! Set domain decomposition information
        void setDomainDecomposition(std::shared_ptr<DomainDecomposition> decomposition)
            {
            assert(decomposition);
            m_decomposition = decomposition;
            m_box = m_decomposition->calculateLocalBox(m_global_box);
            m_boxchange_signal.emit();
            }

        //! Returns the domain decomin decomposition information
        std::shared_ptr<DomainDecomposition> getDomainDecomposition()
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
         *  The output buffers are automatically resized to accommodate the data.
         *
         *  \post The particle data arrays remain compact. Any ghost atoms
         *        are invalidated. (call removeAllGhostAtoms() before or after
         *        this method).
         */
        void removeParticles(std::vector<pdata_element>& out, std::vector<unsigned int>& comm_flags);

        //! Add new local particles
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
         *  The output buffers are automatically resized to accommodate the data.
         *
         *  \post The particle data arrays remain compact. Any ghost atoms
         *        are invalidated. (call removeAllGhostAtoms() before or after
         *        this method).
         */
        void removeParticlesGPU(GlobalVector<pdata_element>& out, GlobalVector<unsigned int>& comm_flags);

        //! Remove particles from local domain and add new particle data (GPU version)
        /*! \param in List of particle data elements to fill the particle data with
         */
        void addParticlesGPU(const GlobalVector<pdata_element>& in);
        #endif // ENABLE_CUDA

#endif // ENABLE_MPI

        //! Add a single particle to the simulation
        unsigned int addParticle(unsigned int type);

        //! Remove a particle from the simulation
        void removeParticle(unsigned int tag);

        //! Return the nth active global tag
        unsigned int getNthTag(unsigned int n);

        //! Add particle types
        /*! \param Name of type to add
         *
         * Adds the name to the list of types and
         * returns the id of the newly added type
         */
        unsigned int addType(const std::string& type_name);

        //! Translate the box origin
        /*! \param a vector to apply in the translation
        */
        void translateOrigin(const Scalar3& a)
            {
            m_origin += a;
            // wrap the origin back into the box to prevent it from getting too large
            m_global_box.wrap(m_origin, m_o_image);
            }

        //! Set the origin and its image
        void setOrigin(const Scalar3& origin, int3& img)
            {
            m_origin = origin;
            m_o_image = img;
            }

        //! Rest the box origin
        /*! \post The origin is 0,0,0
        */
        void resetOrigin()
            {
            m_origin = make_scalar3(0,0,0);
            m_o_image = make_int3(0,0,0);
            }

        #ifdef ENABLE_CUDA
        //! Return the load balancing GPU partition
        const GPUPartition& getGPUPartition() const
            {
            return m_gpu_partition;
            }
        #endif

    private:
        BoxDim m_box;                               //!< The simulation box
        BoxDim m_global_box;                        //!< Global simulation box
        std::shared_ptr<ExecutionConfiguration> m_exec_conf; //!< The execution configuration
#ifdef ENABLE_MPI
        std::shared_ptr<DomainDecomposition> m_decomposition;       //!< Domain decomposition data
#endif

        std::vector<std::string> m_type_mapping;    //!< Mapping between particle type indices and names

        Nano::Signal<void ()> m_sort_signal;       //!< Signal that is triggered when particles are sorted in memory
        Nano::Signal<void ()> m_boxchange_signal;  //!< Signal that is triggered when the box size changes
        Nano::Signal<void ()> m_max_particle_num_signal; //!< Signal that is triggered when the maximum particle number changes
        Nano::Signal<void ()> m_ghost_particles_removed_signal; //!< Signal that is triggered when ghost particles are removed
        Nano::Signal<void ()> m_global_particle_num_signal; //!< Signal that is triggered when the global number of particles changes
        Nano::Signal<void ()> m_num_types_signal;  //!< Signal that is triggered when the number of types changes
        Nano::Signal<Scalar ()> m_composite_particles_signal;  //!< Signal that is triggered when the maximum diameter of a composite particle is needed

        #ifdef ENABLE_MPI
        Nano::Signal<void (unsigned int, unsigned int, unsigned int)> m_ptl_move_signal; //!< Signal when particle moves between domains
        #endif

        unsigned int m_nparticles;                  //!< number of particles
        unsigned int m_nghosts;                     //!< number of ghost particles
        unsigned int m_max_nparticles;              //!< maximum number of particles
        unsigned int m_nglobal;                     //!< global number of particles
        bool m_accel_set;                           //!< Flag to tell if acceleration data has been set

        // per-particle data
        GlobalArray<Scalar4> m_pos;                    //!< particle positions and types
        GlobalArray<Scalar4> m_vel;                    //!< particle velocities and masses
        GlobalArray<Scalar3> m_accel;                  //!< particle accelerations
        GlobalArray<Scalar> m_charge;                  //!< particle charges
        GlobalArray<Scalar> m_diameter;                //!< particle diameters
        GlobalArray<int3> m_image;                     //!< particle images
        GlobalArray<unsigned int> m_tag;               //!< particle tags
        GlobalVector<unsigned int> m_rtag;             //!< reverse lookup tags
        GlobalArray<unsigned int> m_body;              //!< rigid body ids
        GlobalArray< Scalar4 > m_orientation;          //!< Orientation quaternion for each particle (ignored if not anisotropic)
        GlobalArray< Scalar4 > m_angmom;               //!< Angular momementum quaternion for each particle
        GlobalArray< Scalar3 > m_inertia;              //!< Principal moments of inertia for each particle
        GlobalArray<unsigned int> m_comm_flags;        //!< Array of communication flags

        std::stack<unsigned int> m_recycled_tags;    //!< Global tags of removed particles
        std::set<unsigned int> m_tag_set;            //!< Lookup table for tags by active index
        std::vector<unsigned int> m_cached_tag_set;   //!< Cached constant-time lookup table for tags by active index
        bool m_invalid_cached_tags;                  //!< true if m_cached_tag_set needs to be rebuilt

        /* Alternate particle data arrays are provided for fast swapping in and out of particle data
           The size of these arrays is updated in sync with the main particle data arrays.

           The primary use case is when particle data has to be re-ordered in-place, i.e.
           a temporary array would otherwise be required. Instead of writing to a temporary
           array and copying to the main particle data subsequently, the re-ordered particle
           data can be written to the alternate arrays, which are then swapped in for
           the real particle data at effectively zero cost.
         */
        GlobalArray<Scalar4> m_pos_alt;                //!< particle positions and type (swap-in)
        GlobalArray<Scalar4> m_vel_alt;                //!< particle velocities and masses (swap-in)
        GlobalArray<Scalar3> m_accel_alt;              //!< particle accelerations (swap-in)
        GlobalArray<Scalar> m_charge_alt;              //!< particle charges (swap-in)
        GlobalArray<Scalar> m_diameter_alt;            //!< particle diameters (swap-in)
        GlobalArray<int3> m_image_alt;                 //!< particle images (swap-in)
        GlobalArray<unsigned int> m_tag_alt;           //!< particle tags (swap-in)
        GlobalArray<unsigned int> m_body_alt;          //!< rigid body ids (swap-in)
        GlobalArray<Scalar4> m_orientation_alt;        //!< orientations (swap-in)
        GlobalArray<Scalar4> m_angmom_alt;             //!< angular momenta (swap-in)
        GlobalArray<Scalar3> m_inertia_alt;             //!< Principal moments of inertia for each particle (swap-in)
        GlobalArray<Scalar4> m_net_force_alt;          //!< Net force (swap-in)
        GlobalArray<Scalar> m_net_virial_alt;             //!< Net virial (swap-in)
        GlobalArray<Scalar4> m_net_torque_alt;         //!< Net torque (swap-in)

        std::shared_ptr<Profiler> m_prof;         //!< Pointer to the profiler. NULL if there is no profiler.

        GlobalArray< Scalar4 > m_net_force;             //!< Net force calculated for each particle
        GlobalArray< Scalar > m_net_virial;             //!< Net virial calculated for each particle (2D GPU array of dimensions 6*number of particles)
        GlobalArray< Scalar4 > m_net_torque;            //!< Net torque calculated for each particle

        Scalar m_external_virial[6];                 //!< External potential contribution to the virial
        Scalar m_external_energy;                    //!< External potential energy
        const float m_resize_factor;                 //!< The numerical factor with which the particle data arrays are resized
        PDataFlags m_flags;                          //!< Flags identifying which optional fields are valid

        Scalar3 m_origin;                            //!< Tracks the position of the origin of the coordinate system
        int3 m_o_image;                              //!< Tracks the origin image

        bool m_arrays_allocated;                     //!< True if arrays have been initialized

        #ifdef ENABLE_CUDA
        mgpu::ContextPtr m_mgpu_context;             //!< moderngpu context

        GPUPartition m_gpu_partition;                //!< The partition of the local number of particles across GPUs
        unsigned int m_memory_advice_last_Nmax;      //!< Nmax at which memory hints were last set
        #endif

        //! Helper function to allocate particle data
        void allocate(unsigned int N);

        //! Helper function to allocate alternate particle data
        void allocateAlternateArrays(unsigned int N);

        //! Helper function for amortized array resizing
        void resize(unsigned int new_nparticles);

        //! Helper function to reallocate particle data
        void reallocate(unsigned int max_n);

        //! Helper function to rebuild the active tag cache if necessary
        void maybe_rebuild_tag_cache();

        //! Helper function to check that particles of a snapshot are in the box
        /*! \return true If and only if all particles are in the simulation box
         * \param Snapshot to check
         */
        template <class Real>
        bool inBox(const SnapshotParticleData<Real>& snap);

        //! Update the CUDA memory hints
        void setGPUAdvice();
    };

#ifndef NVCC
//! Exports the BoxDim class to python
void export_BoxDim(pybind11::module& m);
//! Exports ParticleData to python
void export_ParticleData(pybind11::module& m);
//! Export SnapshotParticleData to python
void export_SnapshotParticleData(pybind11::module& m);
#endif


#endif
