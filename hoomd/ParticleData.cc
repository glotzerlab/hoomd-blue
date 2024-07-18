// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file ParticleData.cc
    \brief Contains all code for ParticleData, and SnapshotParticleData.
 */

#include "ParticleData.h"

#ifdef ENABLE_MPI
#include "HOOMDMPI.h"
#endif

#ifdef ENABLE_HIP
#include "CachedAllocator.h"
#include "GPUPartition.cuh"
#endif

#include <pybind11/numpy.h>
#include <pybind11/operators.h>

#include <cassert>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <stdlib.h>

using namespace std;

namespace hoomd
    {
namespace detail
    {
std::string getDefaultTypeName(unsigned int id)
    {
    const char default_name[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    unsigned n = (unsigned int)(sizeof(default_name) / sizeof(char) - 1);
    std::string result(1, default_name[id % n]);

    while (id >= n)
        {
        id = id / n - 1;
        result = std::string(1, default_name[id % n]) + result;
        }

    return result;
    }

    } // end namespace detail

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
ParticleData::ParticleData(unsigned int N,
                           const std::shared_ptr<const BoxDim> global_box,
                           unsigned int n_types,
                           std::shared_ptr<ExecutionConfiguration> exec_conf,
                           std::shared_ptr<DomainDecomposition> decomposition)
    : m_exec_conf(exec_conf), m_nparticles(0), m_nghosts(0), m_max_nparticles(0), m_nglobal(0),
      m_accel_set(false), m_resize_factor(9. / 8.), m_arrays_allocated(false)
    {
    m_exec_conf->msg->notice(5) << "Constructing ParticleData" << endl;

    // initialize snapshot with default values
    SnapshotParticleData<Scalar> snap(N);

    snap.type_mapping.clear();

    // setup the type mappings
    for (unsigned int i = 0; i < n_types; i++)
        {
        snap.type_mapping.push_back(detail::getDefaultTypeName(i));
        }

#ifdef ENABLE_MPI
    // Set up domain decomposition information
    if (decomposition)
        setDomainDecomposition(decomposition);
#endif

    // initialize box dimensions on all processors
    setGlobalBox(global_box);

#ifdef ENABLE_HIP
    if (m_exec_conf->isCUDAEnabled())
        {
        m_gpu_partition = GPUPartition(m_exec_conf->getGPUIds());
        m_memory_advice_last_Nmax = UINT_MAX;
        }
#endif

    // initialize rtag array
    GlobalVector<unsigned int>(exec_conf).swap(m_rtag);
    TAG_ALLOCATION(m_rtag);

    // initialize all processors
    initializeFromSnapshot(snap);

    // reset external virial
    for (unsigned int i = 0; i < 6; i++)
        m_external_virial[i] = Scalar(0.0);

    m_external_energy = Scalar(0.0);

    // zero the origin
    m_origin = make_scalar3(0, 0, 0);
    m_o_image = make_int3(0, 0, 0);
    }

/*! Loads particle data from the snapshot into the internal arrays.
 * \param snapshot The particle data snapshot
 * \param global_box The dimensions of the global simulation box
 * \param exec_conf The execution configuration
 * \param decomposition (optional) Domain decomposition layout
 */
template<class Real>
ParticleData::ParticleData(const SnapshotParticleData<Real>& snapshot,
                           const std::shared_ptr<const BoxDim> global_box,
                           std::shared_ptr<ExecutionConfiguration> exec_conf,
                           std::shared_ptr<DomainDecomposition> decomposition)
    : m_exec_conf(exec_conf), m_nparticles(0), m_nghosts(0), m_max_nparticles(0), m_nglobal(0),
      m_accel_set(false), m_resize_factor(9. / 8.), m_arrays_allocated(false)
    {
    m_exec_conf->msg->notice(5) << "Constructing ParticleData" << endl;

#ifdef ENABLE_MPI
    // Set up domain decomposition information
    if (decomposition)
        setDomainDecomposition(decomposition);
#endif

    // initialize box dimensions on all processors
    setGlobalBox(global_box);

    // it is an error for particles to be initialized outside of their box
    if (!inBox(snapshot))
        {
        m_exec_conf->msg->warning() << "Not all particles were found inside the given box" << endl;
        throw runtime_error("Error initializing ParticleData");
        }

#ifdef ENABLE_HIP
    if (m_exec_conf->isCUDAEnabled())
        {
        m_gpu_partition = GPUPartition(m_exec_conf->getGPUIds());
        m_memory_advice_last_Nmax = UINT_MAX;
        }
#endif

    // initialize rtag array
    GlobalVector<unsigned int>(exec_conf).swap(m_rtag);
    TAG_ALLOCATION(m_rtag);

    // initialize particle data with snapshot contents
    initializeFromSnapshot(snapshot);

    // reset external virial
    for (unsigned int i = 0; i < 6; i++)
        m_external_virial[i] = Scalar(0.0);

    m_external_energy = Scalar(0.0);

    // zero the origin
    m_origin = make_scalar3(0, 0, 0);
    m_o_image = make_int3(0, 0, 0);
    }

ParticleData::~ParticleData()
    {
    m_exec_conf->msg->notice(5) << "Destroying ParticleData" << endl;
    }

/*! \return Simulation box dimensions
 */
const BoxDim ParticleData::getBox() const
    {
    return *m_box;
    }

/*! \param box New box dimensions to set
    \note ParticleData does NOT enforce any boundary conditions. When a new box is set,
        it is the responsibility of the caller to ensure that all particles lie within
        the new box.
*/
void ParticleData::setGlobalBox(const std::shared_ptr<const BoxDim> box)
    {
    assert(box->getPeriodic().x);
    assert(box->getPeriodic().y);
    assert(box->getPeriodic().z);
    if (!box)
        {
        throw std::runtime_error("Expected non-null shared pointer to a box.");
        }
    BoxDim global_box = *box;

#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        bcast(global_box, 0, m_exec_conf->getMPICommunicator());
        m_global_box = std::make_shared<const BoxDim>(global_box);
        m_box = std::make_shared<const BoxDim>(m_decomposition->calculateLocalBox(global_box));
        }
    else
#endif
        {
        // local box = global box
        m_global_box = std::make_shared<const BoxDim>(global_box);
        m_box = m_global_box;
        }

    m_boxchange_signal.emit();
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
#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        auto global_box = box;
        bcast(global_box, 0, m_exec_conf->getMPICommunicator());
        m_global_box = std::make_shared<const BoxDim>(box);
        m_box = std::make_shared<const BoxDim>(m_decomposition->calculateLocalBox(box));
        }
    else
#endif
        {
        // local box = global box
        m_global_box = std::make_shared<const BoxDim>(box);
        m_box = m_global_box;
        }

    m_boxchange_signal.emit();
    }

/*! \return Global simulation box dimensions
 */
const BoxDim ParticleData::getGlobalBox() const
    {
    if (m_global_box)
        {
        return *m_global_box;
        }
    return BoxDim();
    }

/*! \b ANY time particles are rearranged in memory, this function must be called.
    \note The call must be made after calling release()
*/
void ParticleData::notifyParticleSort()
    {
#ifdef ENABLE_HIP
    if (m_exec_conf->isCUDAEnabled())
        {
        // need to update GPUPartition if particle number changes
        m_gpu_partition.setN(getN());

        // update our CUDA hints
        setGPUAdvice();
        }
#endif

    m_sort_signal.emit();
    }

/*! This function is called any time the ghost particles are removed
 *
 * The rationale is that a subscriber (i.e. the Communicator) can perform clean-up for ghost
 * particles it has created. The current ghost particle number is still available through
 * getNGhosts() at the time the signal is triggered.
 *
 */
void ParticleData::notifyGhostParticlesRemoved()
    {
    m_ghost_particles_removed_signal.emit();
    }

/*! \param name Type name to get the index of
    \return Type index of the corresponding type name
    \note Throws an exception if the type name is not found
*/
unsigned int ParticleData::getTypeByName(const std::string& name) const
    {
    // search for the name
    for (unsigned int i = 0; i < m_type_mapping.size(); i++)
        {
        if (m_type_mapping[i] == name)
            return i;
        }

    std::ostringstream s;
    s << "Type " << name << " not found!";
    throw runtime_error(s.str());
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
        ostringstream s;
        s << "Requesting type name for non-existent type " << type << ".";
        throw runtime_error(s.str());
        }

    // return the name
    return m_type_mapping[type];
    }

/*! \param type Type index to get the name of
    \param name New name for this type

    This is designed for use by init.create_empty. It probably will cause things to fail in strange
   ways if called after initialization.
*/
void ParticleData::setTypeName(unsigned int type, const std::string& name)
    {
    // check for an invalid request
    if (type >= getNTypes())
        {
        ostringstream s;
        s << "Setting name for non-existent type " << type << ".";
        throw runtime_error(s.str());
        }

    m_type_mapping[type] = name;
    }

/*! \param N Number of particles to allocate memory for
    \pre No memory is allocated and the per-particle GPUArrays are uninitialized
    \post All per-particle GPUArrays are allocated
*/
void ParticleData::allocate(unsigned int N)
    {
    // maximum number is the current particle number
    m_max_nparticles = N;

    // positions
    GlobalArray<Scalar4> pos(N, m_exec_conf);
    m_pos.swap(pos);
    TAG_ALLOCATION(m_pos);

    // velocities
    GlobalArray<Scalar4> vel(N, m_exec_conf);
    m_vel.swap(vel);
    TAG_ALLOCATION(m_vel);

    // accelerations
    GlobalArray<Scalar3> accel(N, m_exec_conf);
    m_accel.swap(accel);
    TAG_ALLOCATION(m_accel);

    // charge
    GlobalArray<Scalar> charge(N, m_exec_conf);
    m_charge.swap(charge);
    TAG_ALLOCATION(m_charge);

    // diameter
    GlobalArray<Scalar> diameter(N, m_exec_conf);
    m_diameter.swap(diameter);
    TAG_ALLOCATION(m_diameter);

    // image
    GlobalArray<int3> image(N, m_exec_conf);
    m_image.swap(image);
    TAG_ALLOCATION(m_image);

    // global tag
    GlobalArray<unsigned int> tag(N, m_exec_conf);
    m_tag.swap(tag);
    TAG_ALLOCATION(m_tag);

    // body ID
    GlobalArray<unsigned int> body(N, m_exec_conf);
    m_body.swap(body);
    TAG_ALLOCATION(m_body);

    GlobalArray<Scalar4> net_force(N, m_exec_conf);
    m_net_force.swap(net_force);
    TAG_ALLOCATION(m_net_force);
    GlobalArray<Scalar> net_virial(N, 6, m_exec_conf);
    m_net_virial.swap(net_virial);
    TAG_ALLOCATION(m_net_virial);
    GlobalArray<Scalar4> net_torque(N, m_exec_conf);
    m_net_torque.swap(net_torque);
    TAG_ALLOCATION(m_net_torque);

        {
        ArrayHandle<Scalar4> h_net_force(m_net_force,
                                         access_location::host,
                                         access_mode::overwrite);
        ArrayHandle<Scalar4> h_net_torque(m_net_torque,
                                          access_location::host,
                                          access_mode::overwrite);
        ArrayHandle<Scalar> h_net_virial(m_net_virial,
                                         access_location::host,
                                         access_mode::overwrite);
        memset(h_net_force.data, 0, sizeof(Scalar4) * m_net_force.getNumElements());
        memset(h_net_torque.data, 0, sizeof(Scalar4) * m_net_torque.getNumElements());
        memset(h_net_virial.data, 0, sizeof(Scalar) * m_net_virial.getNumElements());
        }

    GlobalArray<Scalar4> orientation(N, m_exec_conf);
    m_orientation.swap(orientation);
    TAG_ALLOCATION(m_orientation);
    GlobalArray<Scalar4> angmom(N, m_exec_conf);
    m_angmom.swap(angmom);
    TAG_ALLOCATION(m_angmom);
    GlobalArray<Scalar3> inertia(N, m_exec_conf);
    m_inertia.swap(inertia);
    TAG_ALLOCATION(m_inertia);

    GlobalArray<unsigned int> comm_flags(N, m_exec_conf);
    m_comm_flags.swap(comm_flags);
    TAG_ALLOCATION(m_comm_flags);

#if defined(ENABLE_HIP) && defined(__HIP_PLATFORM_NVCC__)
    if (m_exec_conf->isCUDAEnabled() && m_exec_conf->allConcurrentManagedAccess())
        {
        auto gpu_map = m_exec_conf->getGPUIds();

        // set up GPU memory mappings
        for (unsigned int idev = 0; idev < m_exec_conf->getNumActiveGPUs(); ++idev)
            {
            // only optimize access for those fields used in force computation
            // (i.e. no net_force/virial/torque, also angmom and inertia are only used by the
            // integrator)
            cudaMemAdvise(m_pos.get(),
                          sizeof(Scalar4) * m_pos.getNumElements(),
                          cudaMemAdviseSetAccessedBy,
                          gpu_map[idev]);
            cudaMemAdvise(m_vel.get(),
                          sizeof(Scalar4) * m_vel.getNumElements(),
                          cudaMemAdviseSetAccessedBy,
                          gpu_map[idev]);
            cudaMemAdvise(m_accel.get(),
                          sizeof(Scalar3) * m_accel.getNumElements(),
                          cudaMemAdviseSetAccessedBy,
                          gpu_map[idev]);
            cudaMemAdvise(m_charge.get(),
                          sizeof(Scalar) * m_charge.getNumElements(),
                          cudaMemAdviseSetAccessedBy,
                          gpu_map[idev]);
            cudaMemAdvise(m_diameter.get(),
                          sizeof(Scalar) * m_diameter.getNumElements(),
                          cudaMemAdviseSetAccessedBy,
                          gpu_map[idev]);
            cudaMemAdvise(m_image.get(),
                          sizeof(int3) * m_image.getNumElements(),
                          cudaMemAdviseSetAccessedBy,
                          gpu_map[idev]);
            cudaMemAdvise(m_tag.get(),
                          sizeof(unsigned int) * m_tag.getNumElements(),
                          cudaMemAdviseSetAccessedBy,
                          gpu_map[idev]);
            cudaMemAdvise(m_body.get(),
                          sizeof(unsigned int) * m_body.getNumElements(),
                          cudaMemAdviseSetAccessedBy,
                          gpu_map[idev]);
            cudaMemAdvise(m_orientation.get(),
                          sizeof(Scalar4) * m_orientation.getNumElements(),
                          cudaMemAdviseSetAccessedBy,
                          gpu_map[idev]);
            }
        CHECK_CUDA_ERROR();
        }
#endif

    // allocate alternate particle data arrays (for swapping in-out)
    allocateAlternateArrays(N);

    // notify observers
    m_max_particle_num_signal.emit();

    m_arrays_allocated = true;
    }

/*! \param N Number of particles to allocate memory for
    \pre No memory is allocated and the alternate per-particle GPUArrays are uninitialized
    \post All alternate per-particle GPUArrays are allocated
*/
void ParticleData::allocateAlternateArrays(unsigned int N)
    {
    // positions
    GlobalArray<Scalar4> pos_alt(N, m_exec_conf);
    m_pos_alt.swap(pos_alt);
    TAG_ALLOCATION(m_pos_alt);

    // velocities
    GlobalArray<Scalar4> vel_alt(N, m_exec_conf);
    m_vel_alt.swap(vel_alt);
    TAG_ALLOCATION(m_vel_alt);

    // accelerations
    GlobalArray<Scalar3> accel_alt(N, m_exec_conf);
    m_accel_alt.swap(accel_alt);
    TAG_ALLOCATION(m_accel_alt);

    // charge
    GlobalArray<Scalar> charge_alt(N, m_exec_conf);
    m_charge_alt.swap(charge_alt);
    TAG_ALLOCATION(m_charge_alt);

    // diameter
    GlobalArray<Scalar> diameter_alt(N, m_exec_conf);
    m_diameter_alt.swap(diameter_alt);
    TAG_ALLOCATION(m_diameter_alt);

    // image
    GlobalArray<int3> image_alt(N, m_exec_conf);
    m_image_alt.swap(image_alt);
    TAG_ALLOCATION(m_image_alt);

    // global tag
    GlobalArray<unsigned int> tag_alt(N, m_exec_conf);
    m_tag_alt.swap(tag_alt);
    TAG_ALLOCATION(m_tag_alt);

    // body ID
    GlobalArray<unsigned int> body_alt(N, m_exec_conf);
    m_body_alt.swap(body_alt);
    TAG_ALLOCATION(m_body_alt);

    // orientation
    GlobalArray<Scalar4> orientation_alt(N, m_exec_conf);
    m_orientation_alt.swap(orientation_alt);
    TAG_ALLOCATION(m_orientation_alt);

    // angular momentum
    GlobalArray<Scalar4> angmom_alt(N, m_exec_conf);
    m_angmom_alt.swap(angmom_alt);
    TAG_ALLOCATION(m_angmom_alt);

    // moments of inertia
    GlobalArray<Scalar3> inertia_alt(N, m_exec_conf);
    m_inertia_alt.swap(inertia_alt);
    TAG_ALLOCATION(m_inertia_alt);

    // Net force
    GlobalArray<Scalar4> net_force_alt(N, m_exec_conf);
    m_net_force_alt.swap(net_force_alt);
    TAG_ALLOCATION(m_net_force_alt);

    // Net virial
    GlobalArray<Scalar> net_virial_alt(N, 6, m_exec_conf);
    m_net_virial_alt.swap(net_virial_alt);
    TAG_ALLOCATION(m_net_virial_alt);

    // Net torque
    GlobalArray<Scalar4> net_torque_alt(N, m_exec_conf);
    m_net_torque_alt.swap(net_torque_alt);
    TAG_ALLOCATION(m_net_torque_alt);

        {
        ArrayHandle<Scalar4> h_net_force_alt(m_net_force_alt,
                                             access_location::host,
                                             access_mode::overwrite);
        ArrayHandle<Scalar4> h_net_torque_alt(m_net_torque_alt,
                                              access_location::host,
                                              access_mode::overwrite);
        ArrayHandle<Scalar> h_net_virial_alt(m_net_virial_alt,
                                             access_location::host,
                                             access_mode::overwrite);
        memset(h_net_force_alt.data, 0, sizeof(Scalar4) * m_net_force_alt.getNumElements());
        memset(h_net_torque_alt.data, 0, sizeof(Scalar4) * m_net_torque_alt.getNumElements());
        memset(h_net_virial_alt.data, 0, sizeof(Scalar) * m_net_virial_alt.getNumElements());
        }

#if defined(ENABLE_HIP) && defined(__HIP_PLATFORM_NVCC__)
    if (m_exec_conf->isCUDAEnabled() && m_exec_conf->allConcurrentManagedAccess())
        {
        auto gpu_map = m_exec_conf->getGPUIds();

        // set up GPU memory mappings
        for (unsigned int idev = 0; idev < m_exec_conf->getNumActiveGPUs(); ++idev)
            {
            cudaMemAdvise(m_pos_alt.get(),
                          sizeof(Scalar4) * m_pos_alt.getNumElements(),
                          cudaMemAdviseSetAccessedBy,
                          gpu_map[idev]);
            cudaMemAdvise(m_vel_alt.get(),
                          sizeof(Scalar4) * m_vel_alt.getNumElements(),
                          cudaMemAdviseSetAccessedBy,
                          gpu_map[idev]);
            cudaMemAdvise(m_accel_alt.get(),
                          sizeof(Scalar3) * m_accel_alt.getNumElements(),
                          cudaMemAdviseSetAccessedBy,
                          gpu_map[idev]);
            cudaMemAdvise(m_charge_alt.get(),
                          sizeof(Scalar) * m_charge_alt.getNumElements(),
                          cudaMemAdviseSetAccessedBy,
                          gpu_map[idev]);
            cudaMemAdvise(m_diameter_alt.get(),
                          sizeof(Scalar) * m_diameter_alt.getNumElements(),
                          cudaMemAdviseSetAccessedBy,
                          gpu_map[idev]);
            cudaMemAdvise(m_image_alt.get(),
                          sizeof(int3) * m_image_alt.getNumElements(),
                          cudaMemAdviseSetAccessedBy,
                          gpu_map[idev]);
            cudaMemAdvise(m_tag_alt.get(),
                          sizeof(unsigned int) * m_tag_alt.getNumElements(),
                          cudaMemAdviseSetAccessedBy,
                          gpu_map[idev]);
            cudaMemAdvise(m_body_alt.get(),
                          sizeof(unsigned int) * m_body_alt.getNumElements(),
                          cudaMemAdviseSetAccessedBy,
                          gpu_map[idev]);
            cudaMemAdvise(m_orientation_alt.get(),
                          sizeof(Scalar4) * m_orientation_alt.getNumElements(),
                          cudaMemAdviseSetAccessedBy,
                          gpu_map[idev]);
            }
        CHECK_CUDA_ERROR();
        }
#endif
    }

//! Set global number of particles
/*! \param nglobal Global number of particles
 */
void ParticleData::setNGlobal(unsigned int nglobal)
    {
    assert(m_nparticles <= nglobal);

    // Set global particle number
    m_nglobal = nglobal;

    // we have changed the global particle number, notify subscribers
    m_global_particle_num_signal.emit();

#if defined(ENABLE_HIP) && defined(__HIP_PLATFORM_NVCC__)
    if (m_exec_conf->isCUDAEnabled() && m_exec_conf->allConcurrentManagedAccess())
        {
        auto gpu_map = m_exec_conf->getGPUIds();

        // set up GPU memory mappings
        for (unsigned int idev = 0; idev < m_exec_conf->getNumActiveGPUs(); ++idev)
            {
            cudaMemAdvise(m_rtag.get(),
                          sizeof(unsigned int) * m_rtag.getNumElements(),
                          cudaMemAdviseSetAccessedBy,
                          gpu_map[idev]);
            }
        CHECK_CUDA_ERROR();
        }
#endif
    }

/*! \param new_nparticles New particle number
 */
void ParticleData::resize(unsigned int new_nparticles)
    {
// update the partition information, so it is available to subscribers of various signals early
#ifdef ENABLE_HIP
    if (m_exec_conf->isCUDAEnabled())
        m_gpu_partition.setN(new_nparticles);
#endif

    if (new_nparticles == 0)
        {
        // guarantee that arrays are allocated
        if (!m_arrays_allocated)
            allocate(1);
        m_nparticles = new_nparticles;

        return;
        }

    // allocate as necessary
    if (!m_arrays_allocated)
        allocate(new_nparticles);

    // resize pdata arrays as necessary
    unsigned int max_nparticles = m_max_nparticles;
    if (new_nparticles > max_nparticles)
        {
        // use amortized array resizing
        while (new_nparticles > max_nparticles)
            max_nparticles = ((unsigned int)(((float)max_nparticles) * m_resize_factor)) + 1;

        // reallocate particle data arrays
        reallocate(max_nparticles);
        }

    m_nparticles = new_nparticles;
    }

/*! \param max_n new maximum size of particle data arrays (can be greater or smaller than the
 * current maximum size) To inform classes that allocate arrays for per-particle information of the
 * change of the particle data size, this method issues a m_max_particle_num_signal.emit().
 *
 *  \note To keep unnecessary data copying to a minimum, arrays are not reallocated with every
 * change of the particle number, rather an amortized array expanding strategy is used.
 */
void ParticleData::reallocate(unsigned int max_n)
    {
    if (!m_arrays_allocated)
        {
        // allocate instead
        allocate(max_n);
        return;
        }

    m_exec_conf->msg->notice(7) << "Resizing particle data arrays " << m_max_nparticles << " -> "
                                << max_n << " ptls" << std::endl;
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
    m_net_virial.resize(max_n, 6);
    m_net_torque.resize(max_n);
        {
        ArrayHandle<Scalar4> h_net_force(m_net_force,
                                         access_location::host,
                                         access_mode::readwrite);
        ArrayHandle<Scalar4> h_net_torque(m_net_torque,
                                          access_location::host,
                                          access_mode::readwrite);
        ArrayHandle<Scalar> h_net_virial(m_net_virial,
                                         access_location::host,
                                         access_mode::readwrite);
        memset(h_net_force.data, 0, sizeof(Scalar4) * m_net_force.getNumElements());
        memset(h_net_torque.data, 0, sizeof(Scalar4) * m_net_torque.getNumElements());
        memset(h_net_virial.data, 0, sizeof(Scalar) * m_net_virial.getNumElements());
        }

    m_orientation.resize(max_n);
    m_angmom.resize(max_n);
    m_inertia.resize(max_n);

    m_comm_flags.resize(max_n);

#if defined(ENABLE_HIP) && defined(__HIP_PLATFORM_NVCC__)
    if (m_exec_conf->isCUDAEnabled() && m_exec_conf->allConcurrentManagedAccess())
        {
        auto gpu_map = m_exec_conf->getGPUIds();

        // set up GPU memory mappings
        for (unsigned int idev = 0; idev < m_exec_conf->getNumActiveGPUs(); ++idev)
            {
            cudaMemAdvise(m_pos.get(),
                          sizeof(Scalar4) * m_pos.getNumElements(),
                          cudaMemAdviseSetAccessedBy,
                          gpu_map[idev]);
            cudaMemAdvise(m_vel.get(),
                          sizeof(Scalar4) * m_vel.getNumElements(),
                          cudaMemAdviseSetAccessedBy,
                          gpu_map[idev]);
            cudaMemAdvise(m_accel.get(),
                          sizeof(Scalar3) * m_accel.getNumElements(),
                          cudaMemAdviseSetAccessedBy,
                          gpu_map[idev]);
            cudaMemAdvise(m_charge.get(),
                          sizeof(Scalar) * m_charge.getNumElements(),
                          cudaMemAdviseSetAccessedBy,
                          gpu_map[idev]);
            cudaMemAdvise(m_diameter.get(),
                          sizeof(Scalar) * m_diameter.getNumElements(),
                          cudaMemAdviseSetAccessedBy,
                          gpu_map[idev]);
            cudaMemAdvise(m_image.get(),
                          sizeof(int3) * m_image.getNumElements(),
                          cudaMemAdviseSetAccessedBy,
                          gpu_map[idev]);
            cudaMemAdvise(m_tag.get(),
                          sizeof(unsigned int) * m_tag.getNumElements(),
                          cudaMemAdviseSetAccessedBy,
                          gpu_map[idev]);
            cudaMemAdvise(m_body.get(),
                          sizeof(unsigned int) * m_body.getNumElements(),
                          cudaMemAdviseSetAccessedBy,
                          gpu_map[idev]);
            cudaMemAdvise(m_orientation.get(),
                          sizeof(Scalar4) * m_orientation.getNumElements(),
                          cudaMemAdviseSetAccessedBy,
                          gpu_map[idev]);
            }
        CHECK_CUDA_ERROR();
        }
#endif

    if (!m_pos_alt.isNull())
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

            {
            ArrayHandle<Scalar4> h_net_force_alt(m_net_force_alt,
                                                 access_location::host,
                                                 access_mode::overwrite);
            ArrayHandle<Scalar4> h_net_torque_alt(m_net_torque_alt,
                                                  access_location::host,
                                                  access_mode::overwrite);
            ArrayHandle<Scalar> h_net_virial_alt(m_net_virial_alt,
                                                 access_location::host,
                                                 access_mode::overwrite);
            memset(h_net_force_alt.data, 0, sizeof(Scalar4) * m_net_force_alt.getNumElements());
            memset(h_net_torque_alt.data, 0, sizeof(Scalar4) * m_net_torque_alt.getNumElements());
            memset(h_net_virial_alt.data, 0, sizeof(Scalar) * m_net_virial_alt.getNumElements());
            }

#if defined(ENABLE_HIP) && defined(__HIP_PLATFORM_NVCC__)
        if (m_exec_conf->isCUDAEnabled() && m_exec_conf->allConcurrentManagedAccess())
            {
            auto gpu_map = m_exec_conf->getGPUIds();

            // set up GPU memory mappings
            for (unsigned int idev = 0; idev < m_exec_conf->getNumActiveGPUs(); ++idev)
                {
                cudaMemAdvise(m_pos_alt.get(),
                              sizeof(Scalar4) * m_pos_alt.getNumElements(),
                              cudaMemAdviseSetAccessedBy,
                              gpu_map[idev]);
                cudaMemAdvise(m_vel_alt.get(),
                              sizeof(Scalar4) * m_vel_alt.getNumElements(),
                              cudaMemAdviseSetAccessedBy,
                              gpu_map[idev]);
                cudaMemAdvise(m_accel_alt.get(),
                              sizeof(Scalar3) * m_accel_alt.getNumElements(),
                              cudaMemAdviseSetAccessedBy,
                              gpu_map[idev]);
                cudaMemAdvise(m_charge_alt.get(),
                              sizeof(Scalar) * m_charge_alt.getNumElements(),
                              cudaMemAdviseSetAccessedBy,
                              gpu_map[idev]);
                cudaMemAdvise(m_diameter_alt.get(),
                              sizeof(Scalar) * m_diameter_alt.getNumElements(),
                              cudaMemAdviseSetAccessedBy,
                              gpu_map[idev]);
                cudaMemAdvise(m_image_alt.get(),
                              sizeof(int3) * m_image_alt.getNumElements(),
                              cudaMemAdviseSetAccessedBy,
                              gpu_map[idev]);
                cudaMemAdvise(m_tag_alt.get(),
                              sizeof(unsigned int) * m_tag_alt.getNumElements(),
                              cudaMemAdviseSetAccessedBy,
                              gpu_map[idev]);
                cudaMemAdvise(m_body_alt.get(),
                              sizeof(unsigned int) * m_body_alt.getNumElements(),
                              cudaMemAdviseSetAccessedBy,
                              gpu_map[idev]);
                cudaMemAdvise(m_orientation_alt.get(),
                              sizeof(Scalar4) * m_orientation_alt.getNumElements(),
                              cudaMemAdviseSetAccessedBy,
                              gpu_map[idev]);
                }
            CHECK_CUDA_ERROR();
            }
#endif
        }

    // notify observers
    m_max_particle_num_signal.emit();
    }

/*! Rebuild the cached vector of active tags, if necessary
 */
void ParticleData::maybe_rebuild_tag_cache()
    {
    if (!m_invalid_cached_tags)
        return;

    // GlobalVector checks if the resize is necessary
    m_cached_tag_set.resize(m_tag_set.size());

    // iterate over each element in the set, building a mapping
    // from dense array indices to sparse particle tag indices
    unsigned int i(0);
    for (std::set<unsigned int>::const_iterator it(m_tag_set.begin()); it != m_tag_set.end();
         ++it, ++i)
        {
        m_cached_tag_set[i] = *it;
        }

    m_invalid_cached_tags = false;
    }

/*! \return true If and only if all particles are in the simulation box
 */
template<class Real> bool ParticleData::inBox(const SnapshotParticleData<Real>& snap)
    {
    bool in_box = true;
    if (m_exec_conf->getRank() == 0)
        {
        Scalar3 lo = m_global_box->getLo();
        Scalar3 hi = m_global_box->getHi();

        const Scalar tol = Scalar(1e-5);

        for (unsigned int i = 0; i < snap.size; i++)
            {
            Scalar3 f = m_global_box->makeFraction(vec_to_scalar3(snap.pos[i]));
            if (f.x < -tol || f.x > Scalar(1.0) + tol || f.y < -tol || f.y > Scalar(1.0) + tol
                || f.z < -tol || f.z > Scalar(1.0) + tol)
                {
                m_exec_conf->msg->warning()
                    << "pos " << i << ":" << setprecision(12) << snap.pos[i].x << " "
                    << snap.pos[i].y << " " << snap.pos[i].z << endl;
                m_exec_conf->msg->warning() << "fractional pos :" << setprecision(12) << f.x << " "
                                            << f.y << " " << f.z << endl;
                m_exec_conf->msg->warning() << "lo: " << lo.x << " " << lo.y << " " << lo.z << endl;
                m_exec_conf->msg->warning() << "hi: " << hi.x << " " << hi.y << " " << hi.z << endl;
                in_box = false;
                break;
                }
            }
        }
#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        bcast(in_box, 0, m_exec_conf->getMPICommunicator());
        }
#endif
    return in_box;
    }

//! Initialize from a snapshot
/*! \param snapshot the initial particle data
    \param ignore_bodies If True, ignore particles that have a body flag set

    \post the particle data arrays are initialized from the snapshot, in index order

    \pre In parallel simulations, the local box size must be set before a call to
   initializeFromSnapshot().
 */
template<class Real>
void ParticleData::initializeFromSnapshot(const SnapshotParticleData<Real>& snapshot,
                                          bool ignore_bodies)
    {
    m_exec_conf->msg->notice(4) << "ParticleData: initializing from snapshot" << std::endl;
    if (snapshot.type_mapping.size() >= 40)
        {
        m_exec_conf->msg->warning() << "Systems with many particle types perform poorly or result "
                                       "in shared memory errors on the GPU."
                                    << std::endl;
        }

    // remove all ghost particles
    removeAllGhostParticles();

    // check that all fields in the snapshot have correct length
    if (m_exec_conf->getRank() == 0)
        {
        snapshot.validate();
        }

    // clear set of active tags
    m_tag_set.clear();

    // clear reservoir of recycled tags
    while (!m_recycled_tags.empty())
        m_recycled_tags.pop();

    // global number of particles
    unsigned int nglobal = 0;
    unsigned int max_typeid = 0;

#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        // gather box information from all processors
        unsigned int root = 0;

        // Define per-processor particle data
        std::vector<std::vector<Scalar3>> pos_proc;       // Position array of every processor
        std::vector<std::vector<Scalar3>> vel_proc;       // Velocities array of every processor
        std::vector<std::vector<Scalar3>> accel_proc;     // Accelerations array of every processor
        std::vector<std::vector<unsigned int>> type_proc; // Particle types array of every processor
        std::vector<std::vector<Scalar>> mass_proc;   // Particle masses array of every processor
        std::vector<std::vector<Scalar>> charge_proc; // Particle charges array of every processor
        std::vector<std::vector<Scalar>>
            diameter_proc;                         // Particle diameters array of every processor
        std::vector<std::vector<int3>> image_proc; // Particle images array of every processor
        std::vector<std::vector<unsigned int>> body_proc;   // Body ids of every processor
        std::vector<std::vector<Scalar4>> orientation_proc; // Orientations of every processor
        std::vector<std::vector<Scalar4>> angmom_proc;      // Angular momenta of every processor
        std::vector<std::vector<Scalar3>> inertia_proc;     // Angular momenta of every processor
        std::vector<std::vector<unsigned int>> tag_proc;    // Global tags of every processor
        std::vector<unsigned int> N_proc; // Number of particles on every processor

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
        N_proc.resize(size, 0);

        if (my_rank == 0)
            {
            ArrayHandle<unsigned int> h_cart_ranks(m_decomposition->getCartRanks(),
                                                   access_location::host,
                                                   access_mode::read);

            const Index3D& di = m_decomposition->getDomainIndexer();
            unsigned int n_ranks = m_exec_conf->getNRanks();

            BoxDim global_box = *m_global_box;

            // loop over particles in snapshot, place them into domains
            for (typename std::vector<vec3<Real>>::const_iterator it = snapshot.pos.begin();
                 it != snapshot.pos.end();
                 it++)
                {
                unsigned int snap_idx = (unsigned int)(it - snapshot.pos.begin());

                // if requested, do not initialize constituent particles of bodies
                if (ignore_bodies && snapshot.body[snap_idx] < MIN_FLOPPY
                    && snapshot.body[snap_idx] != snap_idx)
                    {
                    continue;
                    }

                // determine domain the particle is placed into
                Scalar3 pos = vec_to_scalar3(*it);
                Scalar3 f = m_global_box->makeFraction(pos);
                int i = int(f.x * ((Scalar)di.getW()));
                int j = int(f.y * ((Scalar)di.getH()));
                int k = int(f.z * ((Scalar)di.getD()));

                // wrap particles that are exactly on a boundary
                // we only need to wrap in the negative direction, since
                // processor ids are rounded toward zero
                char3 flags = make_char3(0, 0, 0);
                if (i == (int)di.getW())
                    {
                    i = 0;
                    flags.x = 1;
                    }

                if (j == (int)di.getH())
                    {
                    j = 0;
                    flags.y = 1;
                    }

                if (k == (int)di.getD())
                    {
                    k = 0;
                    flags.z = 1;
                    }

                int3 img = snapshot.image[snap_idx];

                // only wrap if the particles is on one of the boundaries
                uchar3 periodic = make_uchar3(flags.x, flags.y, flags.z);
                global_box.setPeriodic(periodic);
                global_box.wrap(pos, img, flags);

                // place particle using actual domain fractions, not global box fraction
                unsigned int rank
                    = m_decomposition->placeParticle(global_box, pos, h_cart_ranks.data);

                if (rank >= n_ranks)
                    {
                    ostringstream s;
                    s << "init.*: Particle " << snap_idx << " out of bounds." << std::endl;
                    s << "Cartesian coordinates: " << std::endl;
                    s << "x: " << pos.x << " y: " << pos.y << " z: " << pos.z << std::endl;
                    s << "Fractional coordinates: " << std::endl;
                    s << "f.x: " << f.x << " f.y: " << f.y << " f.z: " << f.z << std::endl;
                    Scalar3 lo = m_global_box->getLo();
                    Scalar3 hi = m_global_box->getHi();
                    s << "Global box lo: (" << lo.x << ", " << lo.y << ", " << lo.z << ")"
                      << std::endl;
                    s << "           hi: (" << hi.x << ", " << hi.y << ", " << hi.z << ")"
                      << std::endl;

                    throw std::runtime_error(s.str());
                    }

                // fill up per-processor data structures
                pos_proc[rank].push_back(pos);
                image_proc[rank].push_back(img);
                vel_proc[rank].push_back(vec_to_scalar3(snapshot.vel[snap_idx]));
                accel_proc[rank].push_back(vec_to_scalar3(snapshot.accel[snap_idx]));
                type_proc[rank].push_back(snapshot.type[snap_idx]);
                mass_proc[rank].push_back(snapshot.mass[snap_idx]);
                charge_proc[rank].push_back(snapshot.charge[snap_idx]);
                diameter_proc[rank].push_back(snapshot.diameter[snap_idx]);
                body_proc[rank].push_back(snapshot.body[snap_idx]);
                orientation_proc[rank].push_back(quat_to_scalar4(snapshot.orientation[snap_idx]));
                angmom_proc[rank].push_back(quat_to_scalar4(snapshot.angmom[snap_idx]));
                inertia_proc[rank].push_back(vec_to_scalar3(snapshot.inertia[snap_idx]));
                tag_proc[rank].push_back(nglobal++);
                N_proc[rank]++;

                // determine max typeid on root rank
                max_typeid = std::max(max_typeid, snapshot.type[snap_idx]);
                }
            }

        // get type mapping
        m_type_mapping = snapshot.type_mapping;

        if (my_rank != root)
            {
            m_type_mapping.clear();
            }

        // broadcast type mapping
        bcast(m_type_mapping, root, mpi_comm);

        // broadcast global number of particles
        bcast(nglobal, root, mpi_comm);

        // resize array for reverse-lookup tags
        m_rtag.resize(nglobal);

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
        scatter_v(pos_proc, pos, root, mpi_comm);
        scatter_v(vel_proc, vel, root, mpi_comm);
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

            {
            // reset all reverse lookup tags to NOT_LOCAL flag
            ArrayHandle<unsigned int> h_rtag(getRTags(),
                                             access_location::host,
                                             access_mode::overwrite);

            // we have to reset all previous rtags, to remove 'leftover' ghosts
            unsigned int max_tag = (unsigned int)m_rtag.size();
            for (unsigned int tag = 0; tag < max_tag; tag++)
                h_rtag.data[tag] = NOT_LOCAL;
            }

        // update list of active tags
        for (unsigned int tag = 0; tag < nglobal; tag++)
            {
            m_tag_set.insert(tag);
            }

        // Now that active tag list has changed, invalidate the cache
        m_invalid_cached_tags = true;

        // resize particle data
        resize(m_nparticles);

        // Load particle data
        ArrayHandle<Scalar4> h_pos(m_pos, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar4> h_vel(m_vel, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar3> h_accel(m_accel, access_location::host, access_mode::overwrite);
        ArrayHandle<int3> h_image(m_image, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar> h_charge(m_charge, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar> h_diameter(m_diameter, access_location::host, access_mode::overwrite);
        ArrayHandle<unsigned int> h_body(m_body, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar4> h_orientation(m_orientation,
                                           access_location::host,
                                           access_mode::overwrite);
        ArrayHandle<Scalar4> h_angmom(m_angmom, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar3> h_inertia(m_inertia, access_location::host, access_mode::overwrite);
        ArrayHandle<unsigned int> h_tag(m_tag, access_location::host, access_mode::overwrite);
        ArrayHandle<unsigned int> h_comm_flag(m_comm_flags,
                                              access_location::host,
                                              access_mode::overwrite);
        ArrayHandle<unsigned int> h_rtag(m_rtag, access_location::host, access_mode::readwrite);

        for (unsigned int idx = 0; idx < m_nparticles; idx++)
            {
            h_pos.data[idx]
                = make_scalar4(pos[idx].x, pos[idx].y, pos[idx].z, __int_as_scalar(type[idx]));
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
        }
    else
#endif
        {
        // allocate array for reverse lookup tags
        m_rtag.resize(snapshot.size);

        // Now that active tag list has changed, invalidate the cache
        m_invalid_cached_tags = true;

        // allocate particle data such that we can accommodate the particles
        resize(snapshot.size);

        ArrayHandle<Scalar4> h_pos(m_pos, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar4> h_vel(m_vel, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar3> h_accel(m_accel, access_location::host, access_mode::overwrite);
        ArrayHandle<int3> h_image(m_image, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar> h_charge(m_charge, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar> h_diameter(m_diameter, access_location::host, access_mode::overwrite);
        ArrayHandle<unsigned int> h_body(m_body, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar4> h_orientation(m_orientation,
                                           access_location::host,
                                           access_mode::overwrite);
        ArrayHandle<Scalar4> h_angmom(m_angmom, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar3> h_inertia(m_inertia, access_location::host, access_mode::overwrite);
        ArrayHandle<unsigned int> h_tag(m_tag, access_location::host, access_mode::overwrite);
        ArrayHandle<unsigned int> h_rtag(m_rtag, access_location::host, access_mode::readwrite);

        for (unsigned int snap_idx = 0; snap_idx < snapshot.size; snap_idx++)
            {
            // if requested, do not initialize constituent particles of rigid bodies
            if (ignore_bodies && snapshot.body[snap_idx] != NO_BODY)
                {
                continue;
                }

            max_typeid = std::max(max_typeid, snapshot.type[snap_idx]);

            h_pos.data[nglobal] = make_scalar4(snapshot.pos[snap_idx].x,
                                               snapshot.pos[snap_idx].y,
                                               snapshot.pos[snap_idx].z,
                                               __int_as_scalar(snapshot.type[snap_idx]));
            h_vel.data[nglobal] = make_scalar4(snapshot.vel[snap_idx].x,
                                               snapshot.vel[snap_idx].y,
                                               snapshot.vel[snap_idx].z,
                                               snapshot.mass[snap_idx]);
            h_accel.data[nglobal] = vec_to_scalar3(snapshot.accel[snap_idx]);
            h_charge.data[nglobal] = snapshot.charge[snap_idx];
            h_diameter.data[nglobal] = snapshot.diameter[snap_idx];
            h_image.data[nglobal] = snapshot.image[snap_idx];
            h_tag.data[nglobal] = nglobal;
            h_rtag.data[nglobal] = nglobal;
            h_body.data[nglobal] = snapshot.body[snap_idx];
            h_orientation.data[nglobal] = quat_to_scalar4(snapshot.orientation[snap_idx]);
            h_angmom.data[nglobal] = quat_to_scalar4(snapshot.angmom[snap_idx]);
            h_inertia.data[nglobal] = vec_to_scalar3(snapshot.inertia[snap_idx]);
            nglobal++;
            }

        m_nparticles = nglobal;

        // update list of active tags
        for (unsigned int tag = 0; tag < nglobal; tag++)
            {
            m_tag_set.insert(tag);
            }

        // rtag size reflects actual number of tags
        m_rtag.resize(nglobal);

        // initialize type mapping
        m_type_mapping = snapshot.type_mapping;
        }

    // copy over accel_set flag from snapshot
    m_accel_set = snapshot.is_accel_set;

    // set global number of particles
    setNGlobal(nglobal);

    // notify listeners about resorting of local particles
    notifyParticleSort();

    // zero the origin
    m_origin = make_scalar3(0, 0, 0);
    m_o_image = make_int3(0, 0, 0);

    unsigned int snapshot_size = snapshot.size;

// Raise an exception if there are any invalid type ids. This is done here (instead of in the
// loops above) to avoid MPI communication deadlocks when only some ranks have invalid types.
// As a convenience, broadcast the values needed to evaluate the condition the same on all
// ranks.
#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        bcast(max_typeid, 0, m_exec_conf->getMPICommunicator());
        bcast(snapshot_size, 0, m_exec_conf->getMPICommunicator());
        }
#endif

    if (snapshot_size != 0 && max_typeid >= m_type_mapping.size())
        {
        std::ostringstream s;
        s << "Particle typeid " << max_typeid << " is invalid in a system with "
          << m_type_mapping.size() << " types.";
        throw std::runtime_error(s.str());
        }
    }

//! take a particle data snapshot
/* \param snapshot The snapshot to write to
   \returns a map to lookup the snapshot index from a particle tag

   \pre snapshot has to be allocated with a number of elements equal to the global number of
   particles)
*/
template<class Real> void ParticleData::takeSnapshot(SnapshotParticleData<Real>& snapshot)
    {
    m_exec_conf->msg->notice(4) << "ParticleData: taking snapshot" << std::endl;

    ArrayHandle<Scalar4> h_pos(m_pos, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_vel(m_vel, access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_accel(m_accel, access_location::host, access_mode::read);
    ArrayHandle<int3> h_image(m_image, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_charge(m_charge, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_diameter(m_diameter, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_body(m_body, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_orientation, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_angmom(m_angmom, access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_inertia(m_inertia, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_tag, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_rtag, access_location::host, access_mode::read);

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
            pos[idx]
                = make_scalar3(h_pos.data[idx].x, h_pos.data[idx].y, h_pos.data[idx].z) - m_origin;
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

        std::vector<std::vector<Scalar3>> pos_proc;       // Position array of every processor
        std::vector<std::vector<Scalar3>> vel_proc;       // Velocities array of every processor
        std::vector<std::vector<Scalar3>> accel_proc;     // Accelerations array of every processor
        std::vector<std::vector<unsigned int>> type_proc; // Particle types array of every processor
        std::vector<std::vector<Scalar>> mass_proc;   // Particle masses array of every processor
        std::vector<std::vector<Scalar>> charge_proc; // Particle charges array of every processor
        std::vector<std::vector<Scalar>>
            diameter_proc;                         // Particle diameters array of every processor
        std::vector<std::vector<int3>> image_proc; // Particle images array of every processor
        std::vector<std::vector<unsigned int>> body_proc;   // Body ids of every processor
        std::vector<std::vector<Scalar4>> orientation_proc; // Orientations of every processor
        std::vector<std::vector<Scalar4>> angmom_proc;      // Angular momenta of every processor
        std::vector<std::vector<Scalar3>> inertia_proc;     // Moments of inertia of every processor

        std::vector<std::map<unsigned int, unsigned int>>
            rtag_map_proc; // List of reverse-lookup maps

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
        gather_v(pos, pos_proc, root, mpi_comm);
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
            // allocate memory in snapshot
            snapshot.resize(getNGlobal());

            unsigned int n_ranks = m_exec_conf->getNRanks();
            assert(rtag_map_proc.size() == n_ranks);

            // create single map of all particle ranks and indices
            std::map<unsigned int, std::pair<unsigned int, unsigned int>> rank_rtag_map;
            std::map<unsigned int, unsigned int>::iterator it;
            for (unsigned int irank = 0; irank < n_ranks; ++irank)
                for (it = rtag_map_proc[irank].begin(); it != rtag_map_proc[irank].end(); ++it)
                    rank_rtag_map.insert(
                        std::pair<unsigned int, std::pair<unsigned int, unsigned int>>(
                            it->first,
                            std::pair<unsigned int, unsigned int>(irank, it->second)));

            // add particles to snapshot
            assert(m_tag_set.size() == getNGlobal());
            std::set<unsigned int>::const_iterator tag_set_it = m_tag_set.begin();

            std::map<unsigned int, std::pair<unsigned int, unsigned int>>::iterator rank_rtag_it;
            for (unsigned int snap_id = 0; snap_id < getNGlobal(); snap_id++)
                {
                unsigned int tag = *tag_set_it;
                assert(tag <= getMaximumTag());
                rank_rtag_it = rank_rtag_map.find(tag);

                if (rank_rtag_it == rank_rtag_map.end())
                    {
                    ostringstream o;
                    o << "Error gathering ParticleData: Could not find particle " << tag
                      << " on any processor.";
                    throw std::runtime_error(o.str());
                    }

                // rank contains the processor rank on which the particle was found
                std::pair<unsigned int, unsigned int> rank_idx = rank_rtag_it->second;
                unsigned int rank = rank_idx.first;
                unsigned int idx = rank_idx.second;

                snapshot.pos[snap_id] = vec3<Real>(pos_proc[rank][idx]);
                snapshot.vel[snap_id] = vec3<Real>(vel_proc[rank][idx]);
                snapshot.accel[snap_id] = vec3<Real>(accel_proc[rank][idx]);
                snapshot.type[snap_id] = type_proc[rank][idx];
                snapshot.mass[snap_id] = Real(mass_proc[rank][idx]);
                snapshot.charge[snap_id] = Real(charge_proc[rank][idx]);
                snapshot.diameter[snap_id] = Real(diameter_proc[rank][idx]);
                snapshot.image[snap_id] = image_proc[rank][idx];
                snapshot.body[snap_id] = body_proc[rank][idx];
                snapshot.orientation[snap_id] = quat<Real>(orientation_proc[rank][idx]);
                snapshot.angmom[snap_id] = quat<Real>(angmom_proc[rank][idx]);
                snapshot.inertia[snap_id] = vec3<Real>(inertia_proc[rank][idx]);

                // make sure the position stored in the snapshot is within the boundaries
                Scalar3 tmp = vec_to_scalar3(snapshot.pos[snap_id]);
                m_global_box->wrap(tmp, snapshot.image[snap_id]);
                snapshot.pos[snap_id] = vec3<Real>(tmp);

                std::advance(tag_set_it, 1);
                }
            }
        }
    else
#endif
        {
        snapshot.resize(getNGlobal());

        // populate particles in snapshot order
        // Note, for large N, it is more cache efficient to read particles in the local index order,
        // however one needs to prepare an additional auxiliary array that maps from local index to
        // the monotonically increasing tag order in the snapshot.
        unsigned int tag = 0;
        for (unsigned int snap_id = 0; snap_id < m_nparticles; snap_id++, tag++)
            {
            unsigned int idx = h_rtag.data[tag];

            // skip ahead to the next tag when particles have been removed by removeParticle()
            while (idx >= m_nparticles)
                {
                tag++;
                idx = h_rtag.data[tag];
                }

            assert(idx < m_nparticles);

            snapshot.pos[snap_id] = vec3<Real>(
                make_scalar3(h_pos.data[idx].x, h_pos.data[idx].y, h_pos.data[idx].z) - m_origin);
            snapshot.vel[snap_id]
                = vec3<Real>(make_scalar3(h_vel.data[idx].x, h_vel.data[idx].y, h_vel.data[idx].z));
            snapshot.accel[snap_id] = vec3<Real>(h_accel.data[idx]);
            snapshot.type[snap_id] = __scalar_as_int(h_pos.data[idx].w);
            snapshot.mass[snap_id] = Real(h_vel.data[idx].w);
            snapshot.charge[snap_id] = Real(h_charge.data[idx]);
            snapshot.diameter[snap_id] = Real(h_diameter.data[idx]);
            snapshot.image[snap_id] = h_image.data[idx];
            snapshot.image[snap_id].x -= m_o_image.x;
            snapshot.image[snap_id].y -= m_o_image.y;
            snapshot.image[snap_id].z -= m_o_image.z;
            snapshot.body[snap_id] = h_body.data[idx];
            snapshot.orientation[snap_id] = quat<Real>(h_orientation.data[idx]);
            snapshot.angmom[snap_id] = quat<Real>(h_angmom.data[idx]);
            snapshot.inertia[snap_id] = vec3<Real>(h_inertia.data[idx]);

            // make sure the position stored in the snapshot is within the boundaries
            Scalar3 tmp = vec_to_scalar3(snapshot.pos[snap_id]);
            m_global_box->wrap(tmp, snapshot.image[snap_id]);
            snapshot.pos[snap_id] = vec3<Real>(tmp);
            }
        }

    snapshot.type_mapping = m_type_mapping;

    // copy over acceleration set flag (this is a copy in case users take a snapshot before running)
    snapshot.is_accel_set = m_accel_set;
    }

//! Add ghost particles at the end of the local particle data
/*! Ghost ptls are appended at the end of the particle data.
  Ghost particles have only incomplete particle information (position, charge, diameter) and
  don't need tags.

  \param nghosts number of ghost particles to add
  \post the particle data arrays are resized if necessary to accommodate the ghost particles,
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
            max_nparticles = ((unsigned int)(((float)max_nparticles) * m_resize_factor)) + 1;

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
        ostringstream s;
        s << "Could not find particle " << tag << " on any processor.";
        throw std::runtime_error(s.str());
        }
    else if (n_found > 1)
        {
        ostringstream s;
        s << "Found particle " << tag << " on multiple processors.";
        throw std::runtime_error(s.str());
        }

    // Now find the processor that owns it
    int owner_rank;
    int flag = is_local ? m_exec_conf->getRank() : -1;
    MPI_Allreduce(&flag, &owner_rank, 1, MPI_INT, MPI_MAX, mpi_comm);

    assert(owner_rank >= 0);
    assert((unsigned int)owner_rank < m_exec_conf->getNRanks());

    return (unsigned int)owner_rank;
    }
#endif

///////////////////////////////////////////////////////////
// get accessors

//! Get the current position of a particle
Scalar3 ParticleData::getPosition(unsigned int tag) const
    {
    unsigned int idx = getRTag(tag);
    bool found = (idx < getN());
    Scalar3 result = make_scalar3(0.0, 0.0, 0.0);
    int3 img = make_int3(0, 0, 0);
    if (found)
        {
        ArrayHandle<Scalar4> h_pos(m_pos, access_location::host, access_mode::read);
        result = make_scalar3(h_pos.data[idx].x, h_pos.data[idx].y, h_pos.data[idx].z);
        result = result - m_origin;

        ArrayHandle<int3> h_img(m_image, access_location::host, access_mode::read);
        img = make_int3(h_img.data[idx].x, h_img.data[idx].y, h_img.data[idx].z);
        img.x -= m_o_image.x;
        img.y -= m_o_image.y;
        img.z -= m_o_image.z;
        }
#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        unsigned int owner_rank = getOwnerRank(tag);
        bcast(result.x, owner_rank, m_exec_conf->getMPICommunicator());
        bcast(result.y, owner_rank, m_exec_conf->getMPICommunicator());
        bcast(result.z, owner_rank, m_exec_conf->getMPICommunicator());
        bcast(img.x, owner_rank, m_exec_conf->getMPICommunicator());
        bcast(img.y, owner_rank, m_exec_conf->getMPICommunicator());
        bcast(img.z, owner_rank, m_exec_conf->getMPICommunicator());
        found = true;
        }
#endif
    assert(found);

    m_global_box->wrap(result, img);
    return result;
    }

//! Get the current velocity of a particle
Scalar3 ParticleData::getVelocity(unsigned int tag) const
    {
    unsigned int idx = getRTag(tag);
    bool found = (idx < getN());
    Scalar3 result = make_scalar3(0.0, 0.0, 0.0);
    if (found)
        {
        ArrayHandle<Scalar4> h_vel(m_vel, access_location::host, access_mode::read);
        result = make_scalar3(h_vel.data[idx].x, h_vel.data[idx].y, h_vel.data[idx].z);
        }
#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        unsigned int owner_rank = getOwnerRank(tag);
        bcast(result.x, owner_rank, m_exec_conf->getMPICommunicator());
        bcast(result.y, owner_rank, m_exec_conf->getMPICommunicator());
        bcast(result.z, owner_rank, m_exec_conf->getMPICommunicator());
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
    Scalar3 result = make_scalar3(0.0, 0.0, 0.0);
    if (found)
        {
        ArrayHandle<Scalar3> h_accel(m_accel, access_location::host, access_mode::read);
        result = make_scalar3(h_accel.data[idx].x, h_accel.data[idx].y, h_accel.data[idx].z);
        }
#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        unsigned int owner_rank = getOwnerRank(tag);
        bcast(result.x, owner_rank, m_exec_conf->getMPICommunicator());
        bcast(result.y, owner_rank, m_exec_conf->getMPICommunicator());
        bcast(result.z, owner_rank, m_exec_conf->getMPICommunicator());
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
    int3 result = make_int3(0, 0, 0);
    Scalar3 pos = make_scalar3(0, 0, 0);
    if (found)
        {
        ArrayHandle<int3> h_image(m_image, access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_postype(m_pos, access_location::host, access_mode::read);
        result = make_int3(h_image.data[idx].x, h_image.data[idx].y, h_image.data[idx].z);
        pos = make_scalar3(h_postype.data[idx].x, h_postype.data[idx].y, h_postype.data[idx].z);
        pos = pos - m_origin;
        }
#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        unsigned int owner_rank = getOwnerRank(tag);
        bcast((int&)result.x, owner_rank, m_exec_conf->getMPICommunicator());
        bcast((int&)result.y, owner_rank, m_exec_conf->getMPICommunicator());
        bcast((int&)result.z, owner_rank, m_exec_conf->getMPICommunicator());
        bcast((Scalar&)pos.x, owner_rank, m_exec_conf->getMPICommunicator());
        bcast((Scalar&)pos.y, owner_rank, m_exec_conf->getMPICommunicator());
        bcast((Scalar&)pos.z, owner_rank, m_exec_conf->getMPICommunicator());
        found = true;
        }
#endif
    assert(found);

    // correct for origin shift
    result.x -= m_o_image.x;
    result.y -= m_o_image.y;
    result.z -= m_o_image.z;

    // wrap into correct image
    m_global_box->wrap(pos, result);

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
        ArrayHandle<Scalar> h_charge(m_charge, access_location::host, access_mode::read);
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
        ArrayHandle<Scalar4> h_vel(m_vel, access_location::host, access_mode::read);
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
        ArrayHandle<Scalar> h_diameter(m_diameter, access_location::host, access_mode::read);
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
        ArrayHandle<unsigned int> h_body(m_body, access_location::host, access_mode::read);
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
        ArrayHandle<Scalar4> h_pos(m_pos, access_location::host, access_mode::read);
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
    Scalar4 result = make_scalar4(0.0, 0.0, 0.0, 0.0);
    if (found)
        {
        ArrayHandle<Scalar4> h_orientation(m_orientation, access_location::host, access_mode::read);
        result = h_orientation.data[idx];
        }
#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        unsigned int owner_rank = getOwnerRank(tag);
        bcast((Scalar&)result.x, owner_rank, m_exec_conf->getMPICommunicator());
        bcast((Scalar&)result.y, owner_rank, m_exec_conf->getMPICommunicator());
        bcast((Scalar&)result.z, owner_rank, m_exec_conf->getMPICommunicator());
        bcast((Scalar&)result.w, owner_rank, m_exec_conf->getMPICommunicator());
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
    Scalar4 result = make_scalar4(0.0, 0.0, 0.0, 0.0);
    if (found)
        {
        ArrayHandle<Scalar4> h_angmom(m_angmom, access_location::host, access_mode::read);
        result = h_angmom.data[idx];
        }
#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        unsigned int owner_rank = getOwnerRank(tag);
        bcast((Scalar&)result.x, owner_rank, m_exec_conf->getMPICommunicator());
        bcast((Scalar&)result.y, owner_rank, m_exec_conf->getMPICommunicator());
        bcast((Scalar&)result.z, owner_rank, m_exec_conf->getMPICommunicator());
        bcast((Scalar&)result.w, owner_rank, m_exec_conf->getMPICommunicator());
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
    Scalar3 result = make_scalar3(0.0, 0.0, 0.0);
    if (found)
        {
        ArrayHandle<Scalar3> h_inertia(m_inertia, access_location::host, access_mode::read);
        result = h_inertia.data[idx];
        }
#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        unsigned int owner_rank = getOwnerRank(tag);
        bcast((Scalar&)result.x, owner_rank, m_exec_conf->getMPICommunicator());
        bcast((Scalar&)result.y, owner_rank, m_exec_conf->getMPICommunicator());
        bcast((Scalar&)result.z, owner_rank, m_exec_conf->getMPICommunicator());
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
    Scalar4 result = make_scalar4(0.0, 0.0, 0.0, 0.0);
    if (found)
        {
        ArrayHandle<Scalar4> h_net_force(m_net_force, access_location::host, access_mode::read);
        result = h_net_force.data[idx];
        }
#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        unsigned int owner_rank = getOwnerRank(tag);
        bcast((Scalar&)result.x, owner_rank, m_exec_conf->getMPICommunicator());
        bcast((Scalar&)result.y, owner_rank, m_exec_conf->getMPICommunicator());
        bcast((Scalar&)result.z, owner_rank, m_exec_conf->getMPICommunicator());
        bcast((Scalar&)result.w, owner_rank, m_exec_conf->getMPICommunicator());
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
    Scalar4 result = make_scalar4(0.0, 0.0, 0.0, 0.0);
    if (found)
        {
        ArrayHandle<Scalar4> h_net_torque(m_net_torque, access_location::host, access_mode::read);
        result = h_net_torque.data[idx];
        }
#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        unsigned int owner_rank = getOwnerRank(tag);
        bcast((Scalar&)result.x, owner_rank, m_exec_conf->getMPICommunicator());
        bcast((Scalar&)result.y, owner_rank, m_exec_conf->getMPICommunicator());
        bcast((Scalar&)result.z, owner_rank, m_exec_conf->getMPICommunicator());
        bcast((Scalar&)result.w, owner_rank, m_exec_conf->getMPICommunicator());
        found = true;
        }
#endif
    assert(found);
    return result;
    }

/*! \param tag Global particle tag
    \param component Virial component (0=xx, 1=xy, 2=xz, 3=yy, 4=yz, 5=zz)
    \returns Force of particle referenced by tag
 */
Scalar ParticleData::getPNetVirial(unsigned int tag, unsigned int component) const
    {
    unsigned int i = getRTag(tag);
    bool found = (i < getN());
    Scalar result = Scalar(0.0);
    if (found)
        {
        ArrayHandle<Scalar> h_net_virial(m_net_virial, access_location::host, access_mode::read);
        result = h_net_virial.data[m_net_virial.getPitch() * component + i];
        }
#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        unsigned int owner_rank = getOwnerRank(tag);
        MPI_Bcast(&result, sizeof(Scalar), MPI_BYTE, owner_rank, m_exec_conf->getMPICommunicator());
        }
#endif
    return result;
    }

//! Set the current position of a particle
/* \post In parallel simulations, the particle is moved to a new domain if necessary.
 * \warning Do not call during a simulation (method can overwrite ghost particle data)
 */
void ParticleData::setPosition(unsigned int tag, const Scalar3& pos, bool move)
    {
    // shift using gridtshift origin
    Scalar3 tmp_pos = pos + m_origin;

    unsigned int idx = getRTag(tag);
    bool ptl_local = (idx < getN());

#ifdef ENABLE_MPI
    // get the owner rank
    unsigned int owner_rank = 0;
    if (m_decomposition)
        owner_rank = getOwnerRank(tag);
#endif

    // load current particle image
    int3 img;
    if (ptl_local)
        {
        ArrayHandle<int3> h_image(m_image, access_location::host, access_mode::read);
        img = h_image.data[idx];
        }
    else
        {
        // if we don't own the particle, we are not going to use image flags
        img = make_int3(0, 0, 0);
        }

    // wrap into box and update image
    m_global_box->wrap(tmp_pos, img);

    // store position and image
    if (ptl_local)
        {
        ArrayHandle<Scalar4> h_pos(m_pos, access_location::host, access_mode::readwrite);
        ArrayHandle<int3> h_image(m_image, access_location::host, access_mode::readwrite);

        h_pos.data[idx].x = tmp_pos.x;
        h_pos.data[idx].y = tmp_pos.y;
        h_pos.data[idx].z = tmp_pos.z;
        h_image.data[idx] = img;
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
        ArrayHandle<unsigned int> h_cart_ranks(m_decomposition->getCartRanks(),
                                               access_location::host,
                                               access_mode::read);
        unsigned int new_rank
            = m_decomposition->placeParticle(getGlobalBox(), tmp_pos, h_cart_ranks.data);
        bcast(new_rank, 0, m_exec_conf->getMPICommunicator());

        // should the particle migrate?
        if (new_rank != owner_rank)
            {
            // we are changing the local particle number, so remove ghost particles
            removeAllGhostParticles();

            m_exec_conf->msg->notice(6) << "Moving particle " << tag << " from rank " << owner_rank
                                        << " to " << new_rank << std::endl;

            if (ptl_local)
                {
                    {
                    // mark for sending
                    ArrayHandle<unsigned int> h_comm_flag(getCommFlags(),
                                                          access_location::host,
                                                          access_mode::readwrite);
                    h_comm_flag.data[idx] = 1;
                    }

                std::vector<detail::pdata_element> buf;

                // retrieve particle data
                std::vector<unsigned int> comm_flags; // not used here
                removeParticles(buf, comm_flags);

                assert(buf.size() >= 1);

                // check for particle data consistency
                if (buf.size() != 1)
                    {
                    std::ostringstream s;
                    s << "More than one (" << buf.size() << ") particle marked for sending.";
                    throw std::runtime_error(s.str());
                    }

                MPI_Request req;
                MPI_Status stat;

                // send particle data to new domain
                MPI_Isend(&buf.front(),
                          sizeof(detail::pdata_element),
                          MPI_BYTE,
                          new_rank,
                          0,
                          m_exec_conf->getMPICommunicator(),
                          &req);
                MPI_Waitall(1, &req, &stat);
                }
            else if (new_rank == my_rank)
                {
                std::vector<detail::pdata_element> buf(1);

                MPI_Request req;
                MPI_Status stat;

                // receive particle data
                MPI_Irecv(&buf.front(),
                          sizeof(detail::pdata_element),
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
            m_ptl_move_signal.emit(tag, owner_rank, new_rank);
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
        ArrayHandle<Scalar4> h_vel(m_vel, access_location::host, access_mode::readwrite);
        h_vel.data[idx].x = vel.x;
        h_vel.data[idx].y = vel.y;
        h_vel.data[idx].z = vel.z;
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
        ArrayHandle<int3> h_image(m_image, access_location::host, access_mode::readwrite);
        h_image.data[idx].x = image.x + m_o_image.x;
        h_image.data[idx].y = image.y + m_o_image.y;
        h_image.data[idx].z = image.z + m_o_image.z;
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
        ArrayHandle<Scalar> h_charge(m_charge, access_location::host, access_mode::readwrite);
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
        ArrayHandle<Scalar4> h_vel(m_vel, access_location::host, access_mode::readwrite);
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
        ArrayHandle<Scalar> h_diameter(m_diameter, access_location::host, access_mode::readwrite);
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
        ArrayHandle<unsigned int> h_body(m_body, access_location::host, access_mode::readwrite);
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
        ArrayHandle<Scalar4> h_pos(m_pos, access_location::host, access_mode::readwrite);
        h_pos.data[idx].w = __int_as_scalar(typ);
        // signal that the types have changed
        notifyParticleSort();
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
        ArrayHandle<Scalar4> h_orientation(m_orientation,
                                           access_location::host,
                                           access_mode::readwrite);
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
        ArrayHandle<Scalar4> h_angmom(m_angmom, access_location::host, access_mode::readwrite);
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
        ArrayHandle<Scalar3> h_inertia(m_inertia, access_location::host, access_mode::readwrite);
        h_inertia.data[idx] = inertia;
        }
    }

/*!
 * Initialize the particle data with a new particle of given type.
 *
 * While the design allows for adding particles on a per time step basis,
 * this is slow and should not be performed too often. In particular, if the same
 * particle is to be inserted multiple times at different positions, instead of
 * adding and removing the particle it may be faster to use the setXXX() methods
 * on the already inserted particle.
 *
 * In **MPI simulations**, particles are added to processor rank 0 by default,
 * and its position has to be updated using setPosition(), which also ensures
 * that it is moved to the correct rank.
 *
 * \param type Type of particle to add
 * \returns the unique tag of the newly added particle
 */
unsigned int ParticleData::addParticle(unsigned int type)
    {
    // we are changing the local number of particles, so remove ghosts
    removeAllGhostParticles();

    // the global tag of the newly created particle
    unsigned int tag;

    // first check if we can recycle a deleted tag
    if (m_recycled_tags.size())
        {
        tag = m_recycled_tags.top();
        m_recycled_tags.pop();
        }
    else
        {
        // Otherwise, generate a new tag
        tag = getNGlobal();

        assert(m_rtag.size() == getNGlobal());
        }

    // add to set of active tags
    m_tag_set.insert(tag);

    // invalidate the active tag cache
    m_invalid_cached_tags = true;

    // resize array of global reverse lookup tags
    m_rtag.resize(getMaximumTag() + 1);

        {
        // update reverse-lookup table
        ArrayHandle<unsigned int> h_rtag(m_rtag, access_location::host, access_mode::readwrite);
        assert((h_rtag.data[tag] = NOT_LOCAL));
        if (m_exec_conf->getRank() == 0)
            {
            // we add the particle at the end
            h_rtag.data[tag] = getN();
            }
        else
            {
            // not on this processor
            h_rtag.data[tag] = NOT_LOCAL;
            }
        }

    assert(tag <= m_recycled_tags.size() + getNGlobal());

    if (m_exec_conf->getRank() == 0)
        {
        // resize particle data using amortized O(1) array resizing
        // and update particle number
        unsigned int old_nparticles = getN();
        resize(old_nparticles + 1);

        // access particle data arrays
        ArrayHandle<Scalar4> h_pos(getPositions(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> h_vel(getVelocities(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar3> h_accel(getAccelerations(),
                                     access_location::host,
                                     access_mode::readwrite);
        ArrayHandle<Scalar> h_charge(getCharges(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> h_diameter(getDiameters(),
                                       access_location::host,
                                       access_mode::readwrite);
        ArrayHandle<int3> h_image(getImages(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> h_angmom(getAngularMomentumArray(),
                                      access_location::host,
                                      access_mode::readwrite);
        ArrayHandle<Scalar3> h_inertia(getMomentsOfInertiaArray(),
                                       access_location::host,
                                       access_mode::readwrite);
        ArrayHandle<unsigned int> h_body(getBodies(),
                                         access_location::host,
                                         access_mode::readwrite);
        ArrayHandle<Scalar4> h_orientation(getOrientationArray(),
                                           access_location::host,
                                           access_mode::readwrite);
        ArrayHandle<unsigned int> h_tag(getTags(), access_location::host, access_mode::readwrite);
        ArrayHandle<unsigned int> h_comm_flag(m_comm_flags,
                                              access_location::host,
                                              access_mode::readwrite);

        unsigned int idx = old_nparticles;

        // initialize to some sensible default values
        h_pos.data[idx] = make_scalar4(0, 0, 0, __int_as_scalar(type));
        h_vel.data[idx] = make_scalar4(0, 0, 0, 1.0);
        h_accel.data[idx] = make_scalar3(0, 0, 0);
        h_charge.data[idx] = 0.0;
        h_diameter.data[idx] = 1.0;
        h_image.data[idx] = make_int3(0, 0, 0);
        h_angmom.data[idx] = make_scalar4(0, 0, 0, 0);
        h_inertia.data[idx] = make_scalar3(0, 0, 0);
        h_body.data[idx] = NO_BODY;
        h_orientation.data[idx] = make_scalar4(1.0, 0.0, 0.0, 0.0);
        h_tag.data[idx] = tag;
        h_comm_flag.data[idx] = 0;
        }

    // update global number of particles
    setNGlobal(getNGlobal() + 1);

    // we have added a particle, notify listeners
    notifyParticleSort();

    return tag;
    }

/*! \param tag Tag of particle to remove
 */
void ParticleData::removeParticle(unsigned int tag)
    {
    if (getNGlobal() == 0)
        {
        throw runtime_error("Trying to remove particle when there are zero particles!");
        }

    // we are changing the local number of particles, so remove ghosts
    removeAllGhostParticles();

    // sanity check
    if (tag >= m_rtag.size())
        {
        std::ostringstream s;
        s << "Trying to remove particle " << tag << " which does not exist.";
        throw runtime_error(s.str());
        }

    // Local particle index
    unsigned int idx = m_rtag[tag];

    bool is_local = idx < getN();
    assert(is_local || idx == NOT_LOCAL);

    bool is_available = is_local;

#ifdef ENABLE_MPI
    if (getDomainDecomposition())
        {
        int res = is_local ? 1 : 0;

        // check that particle is local on some processor
        MPI_Allreduce(MPI_IN_PLACE, &res, 1, MPI_INT, MPI_SUM, m_exec_conf->getMPICommunicator());

        assert((unsigned int)res <= 1);
        is_available = res;
        }
#endif

    if (!is_available)
        {
        std::ostringstream s;
        s << "Trying to remove particle " << tag << " which has been previously removed!";
        throw runtime_error(s.str());
        }

    // delete from map
    m_rtag[tag] = NOT_LOCAL;

    if (is_local)
        {
        unsigned int size = getN();

        // If the particle is not the last element of the particle data, move the last element to
        // to the position of the removed element
        if (idx < (size - 1))
            {
            // access particle data arrays
            ArrayHandle<Scalar4> h_pos(getPositions(),
                                       access_location::host,
                                       access_mode::readwrite);
            ArrayHandle<Scalar4> h_vel(getVelocities(),
                                       access_location::host,
                                       access_mode::readwrite);
            ArrayHandle<Scalar3> h_accel(getAccelerations(),
                                         access_location::host,
                                         access_mode::readwrite);
            ArrayHandle<Scalar> h_charge(getCharges(),
                                         access_location::host,
                                         access_mode::readwrite);
            ArrayHandle<Scalar> h_diameter(getDiameters(),
                                           access_location::host,
                                           access_mode::readwrite);
            ArrayHandle<int3> h_image(getImages(), access_location::host, access_mode::readwrite);
            ArrayHandle<unsigned int> h_body(getBodies(),
                                             access_location::host,
                                             access_mode::readwrite);
            ArrayHandle<Scalar4> h_orientation(getOrientationArray(),
                                               access_location::host,
                                               access_mode::readwrite);
            ArrayHandle<unsigned int> h_tag(getTags(),
                                            access_location::host,
                                            access_mode::readwrite);
            ArrayHandle<unsigned int> h_rtag(getRTags(),
                                             access_location::host,
                                             access_mode::readwrite);
            ArrayHandle<unsigned int> h_comm_flag(m_comm_flags,
                                                  access_location::host,
                                                  access_mode::readwrite);

            h_pos.data[idx] = h_pos.data[size - 1];
            h_vel.data[idx] = h_vel.data[size - 1];
            h_accel.data[idx] = h_accel.data[size - 1];
            h_charge.data[idx] = h_charge.data[size - 1];
            h_diameter.data[idx] = h_diameter.data[size - 1];
            h_image.data[idx] = h_image.data[size - 1];
            h_body.data[idx] = h_body.data[size - 1];
            h_orientation.data[idx] = h_orientation.data[size - 1];
            h_tag.data[idx] = h_tag.data[size - 1];
            h_comm_flag.data[idx] = h_comm_flag.data[size - 1];

            unsigned int last_tag = h_tag.data[size - 1];
            h_rtag.data[last_tag] = idx;
            }

        // update particle number
        resize(getN() - 1);
        }

    // remove from set of active tags
    m_tag_set.erase(tag);

    // maintain a stack of deleted group tags for future recycling
    m_recycled_tags.push(tag);

    // invalidate active tag cache
    m_invalid_cached_tags = true;

    // update global particle number
    setNGlobal(getNGlobal() - 1);

    // local particle number may have changed
    notifyParticleSort();
    }

//! Return the nth active global tag
/*! \param n Index of bond in global bond table
 */
unsigned int ParticleData::getNthTag(unsigned int n)
    {
    if (n >= getNGlobal())
        {
        std::ostringstream s;
        s << "Particle id " << n << "does not exist!";
        throw std::runtime_error(s.str());
        }

    assert(m_tag_set.size() == getNGlobal());

    // maybe_rebuild_tag_cache only rebuilds if necessary
    maybe_rebuild_tag_cache();
    return m_cached_tag_set[n];
    }

namespace detail
    {
void export_BoxDim(pybind11::module& m)
    {
    void (BoxDim::*wrap_overload)(Scalar3&, int3&, char3) const = &BoxDim::wrap;
    Scalar3 (BoxDim::*minImage_overload)(const Scalar3&) const = &BoxDim::minImage;
    Scalar3 (BoxDim::*makeFraction_overload)(const Scalar3&, const Scalar3&) const
        = &BoxDim::makeFraction;

    pybind11::class_<BoxDim, std::shared_ptr<BoxDim>>(m, "BoxDim")
        .def(pybind11::init<Scalar>())
        .def(pybind11::init<Scalar, Scalar, Scalar>())
        .def(pybind11::init<Scalar3>())
        .def(pybind11::init<Scalar3, Scalar3, uchar3>())
        .def(pybind11::init<Scalar, Scalar, Scalar, Scalar>())
        .def(pybind11::self == pybind11::self)
        .def(pybind11::self != pybind11::self)
        .def("getPeriodic",
             [](std::shared_ptr<const BoxDim> box)
             {
                 auto periodic = box->getPeriodic();
                 return make_uint3(periodic.x, periodic.y, periodic.z);
             })
        .def("getL", &BoxDim::getL)
        .def("setL", &BoxDim::setL)
        .def("getLo", &BoxDim::getLo)
        .def("getHi", &BoxDim::getHi)
        .def("setLoHi", &BoxDim::setLoHi)
        .def("setTiltFactors", &BoxDim::setTiltFactors)
        .def("getTiltFactorXY", &BoxDim::getTiltFactorXY)
        .def("getTiltFactorXZ", &BoxDim::getTiltFactorXZ)
        .def("getTiltFactorYZ", &BoxDim::getTiltFactorYZ)
        .def("getLatticeVector", &BoxDim::getLatticeVector)
        .def("wrap", wrap_overload)
        .def("minImage", minImage_overload)
        .def("makeFraction", makeFraction_overload)
        .def("getVolume", &BoxDim::getVolume);
    }

//! Helper for python __str__ for ParticleData
/*! Gives a synopsis of a ParticleData in a string
    \param pdata Particle data to format parameters from
*/
string print_ParticleData(ParticleData* pdata)
    {
    assert(pdata);
    ostringstream s;
    s << "ParticleData: " << pdata->getN() << " particles";
    return s.str();
    }

    } // end namespace detail

// instantiate both float and double methods for snapshots
template ParticleData::ParticleData(const SnapshotParticleData<double>& snapshot,
                                    const std::shared_ptr<const BoxDim> global_box,
                                    std::shared_ptr<ExecutionConfiguration> exec_conf,
                                    std::shared_ptr<DomainDecomposition> decomposition);
template void
ParticleData::initializeFromSnapshot<double>(const SnapshotParticleData<double>& snapshot,
                                             bool ignore_bodies);
template void ParticleData::takeSnapshot<double>(SnapshotParticleData<double>& snapshot);

template ParticleData::ParticleData(const SnapshotParticleData<float>& snapshot,
                                    const std::shared_ptr<const BoxDim> global_box,
                                    std::shared_ptr<ExecutionConfiguration> exec_conf,
                                    std::shared_ptr<DomainDecomposition> decomposition);
template void
ParticleData::initializeFromSnapshot<float>(const SnapshotParticleData<float>& snapshot,
                                            bool ignore_bodies);
template void ParticleData::takeSnapshot<float>(SnapshotParticleData<float>& snapshot);

namespace detail
    {
void export_ParticleData(pybind11::module& m)
    {
    void (ParticleData::*setGlobalBox_overload)(const std::shared_ptr<const BoxDim>)
        = &ParticleData::setGlobalBox;
    pybind11::class_<ParticleData, std::shared_ptr<ParticleData>>(m, "ParticleData")
        .def(pybind11::init<unsigned int,
                            const std::shared_ptr<const BoxDim>,
                            unsigned int,
                            std::shared_ptr<ExecutionConfiguration>>())
        .def("getGlobalBox",
             &ParticleData::getGlobalBox,
             pybind11::return_value_policy::reference_internal)
        .def("getBox", &ParticleData::getBox, pybind11::return_value_policy::reference_internal)
        .def("setGlobalBoxL", &ParticleData::setGlobalBoxL)
        .def("setGlobalBox", setGlobalBox_overload)
        .def("getN", &ParticleData::getN)
        .def("getNGhosts", &ParticleData::getNGhosts)
        .def("getNGlobal", &ParticleData::getNGlobal)
        .def("getNTypes", &ParticleData::getNTypes)
        .def("getMaxDiameter", &ParticleData::getMaxDiameter)
        .def("getNameByType", &ParticleData::getNameByType)
        .def("getTypeByName", &ParticleData::getTypeByName)
        .def("setTypeName", &ParticleData::setTypeName)
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
        .def("getPNetVirial", &ParticleData::getPNetVirial)
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
        .def("setPressureFlag", &ParticleData::setPressureFlag)
        .def("getMaximumTag", &ParticleData::getMaximumTag)
        .def("addParticle", &ParticleData::addParticle)
        .def("removeParticle", &ParticleData::removeParticle)
        .def("getNthTag", &ParticleData::getNthTag)
#ifdef ENABLE_MPI
        .def("setDomainDecomposition", &ParticleData::setDomainDecomposition)
        .def("getDomainDecomposition", &ParticleData::getDomainDecomposition)
#endif
        .def("getTypes", &ParticleData::getTypesPy);
    }

    } // end namespace detail

//! Constructor for SnapshotParticleData
template<class Real>
SnapshotParticleData<Real>::SnapshotParticleData(unsigned int N) : size(N), is_accel_set(false)
    {
    resize(N);
    }

template<class Real> void SnapshotParticleData<Real>::resize(unsigned int N)
    {
    pos.resize(N, vec3<Real>(0.0, 0.0, 0.0));
    vel.resize(N, vec3<Real>(0.0, 0.0, 0.0));
    accel.resize(N, vec3<Real>(0.0, 0.0, 0.0));
    type.resize(N, 0);
    mass.resize(N, Scalar(1.0));
    charge.resize(N, Scalar(0.0));
    diameter.resize(N, Scalar(1.0));
    image.resize(N, make_int3(0, 0, 0));
    body.resize(N, NO_BODY);
    orientation.resize(N, quat<Real>(1.0, vec3<Real>(0.0, 0.0, 0.0)));
    angmom.resize(N, quat<Real>(0.0, vec3<Real>(0.0, 0.0, 0.0)));
    inertia.resize(N, vec3<Real>(0.0, 0.0, 0.0));
    size = N;
    is_accel_set = false;
    }

template<class Real> void SnapshotParticleData<Real>::insert(unsigned int i, unsigned int n)
    {
    assert(i <= size);
    pos.insert(pos.begin() + i, n, vec3<Real>(0.0, 0.0, 0.0));
    vel.insert(vel.begin() + i, n, vec3<Real>(0.0, 0.0, 0.0));
    accel.insert(accel.begin() + i, n, vec3<Real>(0.0, 0.0, 0.0));
    type.insert(type.begin() + i, n, 0);
    mass.insert(mass.begin() + i, n, Scalar(1.0));
    charge.insert(charge.begin() + i, n, Scalar(0.0));
    diameter.insert(diameter.begin() + i, n, Scalar(1.0));
    image.insert(image.begin() + i, n, make_int3(0, 0, 0));
    body.insert(body.begin() + i, n, NO_BODY);
    orientation.insert(orientation.begin() + i, n, quat<Real>(1.0, vec3<Real>(0.0, 0.0, 0.0)));
    angmom.insert(angmom.begin() + i, n, quat<Real>(0.0, vec3<Real>(0.0, 0.0, 0.0)));
    inertia.insert(inertia.begin() + i, n, vec3<Real>(0.0, 0.0, 0.0));
    size += n;
    is_accel_set = false;
    }

template<class Real> void SnapshotParticleData<Real>::validate() const
    {
    // Check if all other fields are of equal length==size
    if (pos.size() != size || vel.size() != size || accel.size() != size || type.size() != size
        || mass.size() != size || charge.size() != size || diameter.size() != size
        || image.size() != size || body.size() != size || orientation.size() != size
        || angmom.size() != size || inertia.size() != size)
        {
        throw std::runtime_error("All array sizes must match.");
        }

    // Check that the user provided unique type names.
    std::vector<std::string> types_copy = type_mapping;
    std::sort(types_copy.begin(), types_copy.end());
    auto last = std::unique(types_copy.begin(), types_copy.end());
    if (static_cast<size_t>(last - types_copy.begin()) != type_mapping.size())
        {
        throw std::runtime_error("Type names must be unique.");
        }
    }

#ifdef ENABLE_MPI
//! Select non-zero communication lags
struct comm_flag_select
    {
    bool operator()(const unsigned int comm_flag) const
        {
        return comm_flag;
        }
    };

/*! \note This method may only be used during communication or when
 *        no ghost particles are present, because ghost particle values
 *        are undefined after calling this method.
 */
void ParticleData::removeParticles(std::vector<detail::pdata_element>& out,
                                   std::vector<unsigned int>& comm_flags)
    {
    unsigned int num_remove_ptls = 0;

        {
        // access particle data tags and rtags
        ArrayHandle<unsigned int> h_tag(getTags(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_rtag(getRTags(), access_location::host, access_mode::readwrite);
        ArrayHandle<unsigned int> h_comm_flags(getCommFlags(),
                                               access_location::host,
                                               access_mode::read);

        // set all rtags of ptls with comm_flag != 0 to NOT_LOCAL and count removed particles
        unsigned int N = getN();
        for (unsigned int i = 0; i < N; ++i)
            if (h_comm_flags.data[i])
                {
                unsigned int tag = h_tag.data[i];
                assert(tag <= getMaximumTag());
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
        ArrayHandle<Scalar3> h_accel(getAccelerations(),
                                     access_location::host,
                                     access_mode::readwrite);
        ArrayHandle<Scalar> h_charge(getCharges(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> h_diameter(getDiameters(),
                                       access_location::host,
                                       access_mode::readwrite);
        ArrayHandle<int3> h_image(getImages(), access_location::host, access_mode::readwrite);
        ArrayHandle<unsigned int> h_body(getBodies(),
                                         access_location::host,
                                         access_mode::readwrite);
        ArrayHandle<Scalar4> h_orientation(getOrientationArray(),
                                           access_location::host,
                                           access_mode::readwrite);
        ArrayHandle<Scalar4> h_angmom(getAngularMomentumArray(),
                                      access_location::host,
                                      access_mode::readwrite);
        ArrayHandle<Scalar3> h_inertia(getMomentsOfInertiaArray(),
                                       access_location::host,
                                       access_mode::readwrite);
        ArrayHandle<Scalar4> h_net_force(getNetForce(),
                                         access_location::host,
                                         access_mode::readwrite);
        ArrayHandle<Scalar4> h_net_torque(getNetTorqueArray(),
                                          access_location::host,
                                          access_mode::readwrite);
        ArrayHandle<Scalar> h_net_virial(getNetVirial(),
                                         access_location::host,
                                         access_mode::readwrite);

        ArrayHandle<unsigned int> h_tag(getTags(), access_location::host, access_mode::readwrite);

        ArrayHandle<unsigned int> h_rtag(getRTags(), access_location::host, access_mode::read);

        ArrayHandle<unsigned int> h_comm_flags(getCommFlags(),
                                               access_location::host,
                                               access_mode::readwrite);

        ArrayHandle<Scalar4> h_pos_alt(m_pos_alt, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar4> h_vel_alt(m_vel_alt, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar3> h_accel_alt(m_accel_alt,
                                         access_location::host,
                                         access_mode::overwrite);
        ArrayHandle<Scalar> h_charge_alt(m_charge_alt,
                                         access_location::host,
                                         access_mode::overwrite);
        ArrayHandle<Scalar> h_diameter_alt(m_diameter_alt,
                                           access_location::host,
                                           access_mode::overwrite);
        ArrayHandle<int3> h_image_alt(m_image_alt, access_location::host, access_mode::overwrite);
        ArrayHandle<unsigned int> h_body_alt(m_body_alt,
                                             access_location::host,
                                             access_mode::overwrite);
        ArrayHandle<Scalar4> h_orientation_alt(m_orientation_alt,
                                               access_location::host,
                                               access_mode::overwrite);
        ArrayHandle<Scalar4> h_angmom_alt(m_angmom_alt,
                                          access_location::host,
                                          access_mode::overwrite);
        ArrayHandle<Scalar3> h_inertia_alt(m_inertia_alt,
                                           access_location::host,
                                           access_mode::overwrite);
        ArrayHandle<Scalar4> h_net_force_alt(m_net_force_alt,
                                             access_location::host,
                                             access_mode::overwrite);
        ArrayHandle<Scalar4> h_net_torque_alt(m_net_torque_alt,
                                              access_location::host,
                                              access_mode::overwrite);
        ArrayHandle<Scalar> h_net_virial_alt(m_net_virial_alt,
                                             access_location::host,
                                             access_mode::overwrite);
        ArrayHandle<unsigned int> h_tag_alt(m_tag_alt,
                                            access_location::host,
                                            access_mode::overwrite);

        unsigned int n = 0;
        unsigned int m = 0;
        unsigned int net_virial_pitch = (unsigned int)m_net_virial.getPitch();
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
                h_net_force_alt.data[n] = h_net_force.data[i];
                h_net_torque_alt.data[n] = h_net_torque.data[i];
                for (unsigned int j = 0; j < 6; ++j)
                    h_net_virial_alt.data[net_virial_pitch * j + n]
                        = h_net_virial.data[net_virial_pitch * j + i];
                h_tag_alt.data[n] = h_tag.data[i];
                ++n;
                }
            else
                {
                // write to packed array
                detail::pdata_element p;
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
                p.net_force = h_net_force.data[i];
                p.net_torque = h_net_torque.data[i];
                for (unsigned int j = 0; j < 6; ++j)
                    p.net_virial[j] = h_net_virial.data[net_virial_pitch * j + i];
                p.tag = h_tag.data[i];
                out[m++] = p;
                }
            }

        // write out non-zero communication flags
        std::remove_copy_if(h_comm_flags.data,
                            h_comm_flags.data + old_nparticles,
                            comm_flags.begin(),
                            std::not_fn(comm_flag_select()));

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
    swapMomentsOfInertia();
    swapNetForce();
    swapNetTorque();
    swapNetVirial();
    swapTags();

        {
        ArrayHandle<unsigned int> h_rtag(getRTags(), access_location::host, access_mode::readwrite);
        ArrayHandle<unsigned int> h_tag(getTags(), access_location::host, access_mode::read);

        // recompute rtags (particles have moved)
        for (unsigned int idx = 0; idx < m_nparticles; ++idx)
            {
            // reset rtag of this ptl
            unsigned int tag = h_tag.data[idx];
            assert(tag <= getMaximumTag());
            h_rtag.data[tag] = idx;
            }
        }

    // notify subscribers that particle data order has been changed
    notifyParticleSort();
    }

//! Remove particles from local domain and append new particle data
void ParticleData::addParticles(const std::vector<detail::pdata_element>& in)
    {
    unsigned int num_add_ptls = (unsigned int)in.size();

    unsigned int old_nparticles = getN();
    unsigned int new_nparticles = m_nparticles + num_add_ptls;

    // resize particle data using amortized O(1) array resizing
    resize(new_nparticles);

        {
        // access particle data arrays
        ArrayHandle<Scalar4> h_pos(getPositions(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> h_vel(getVelocities(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar3> h_accel(getAccelerations(),
                                     access_location::host,
                                     access_mode::readwrite);
        ArrayHandle<Scalar> h_charge(getCharges(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar> h_diameter(getDiameters(),
                                       access_location::host,
                                       access_mode::readwrite);
        ArrayHandle<int3> h_image(getImages(), access_location::host, access_mode::readwrite);
        ArrayHandle<unsigned int> h_body(getBodies(),
                                         access_location::host,
                                         access_mode::readwrite);
        ArrayHandle<Scalar4> h_orientation(getOrientationArray(),
                                           access_location::host,
                                           access_mode::readwrite);
        ArrayHandle<Scalar4> h_angmom(getAngularMomentumArray(),
                                      access_location::host,
                                      access_mode::readwrite);
        ArrayHandle<Scalar3> h_inertia(getMomentsOfInertiaArray(),
                                       access_location::host,
                                       access_mode::readwrite);
        ArrayHandle<Scalar4> h_net_force(getNetForce(),
                                         access_location::host,
                                         access_mode::readwrite);
        ArrayHandle<Scalar4> h_net_torque(getNetTorqueArray(),
                                          access_location::host,
                                          access_mode::readwrite);
        ArrayHandle<Scalar> h_net_virial(getNetVirial(),
                                         access_location::host,
                                         access_mode::readwrite);
        ArrayHandle<unsigned int> h_tag(getTags(), access_location::host, access_mode::readwrite);
        ArrayHandle<unsigned int> h_rtag(getRTags(), access_location::host, access_mode::readwrite);
        ArrayHandle<unsigned int> h_comm_flags(m_comm_flags,
                                               access_location::host,
                                               access_mode::readwrite);

        unsigned int net_virial_pitch = (unsigned int)m_net_virial.getPitch();
        // add new particles at the end
        unsigned int n = old_nparticles;
        for (std::vector<detail::pdata_element>::const_iterator it = in.begin(); it != in.end();
             ++it)
            {
            detail::pdata_element p = *it;
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
            h_net_force.data[n] = p.net_force;
            h_net_torque.data[n] = p.net_torque;
            for (unsigned int j = 0; j < 6; ++j)
                h_net_virial.data[net_virial_pitch * j + n] = p.net_virial[j];
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
            assert(tag <= getMaximumTag());
            h_rtag.data[tag] = idx;
            }
        }

    // notify subscribers that particle data order has been changed
    notifyParticleSort();
    }

#ifdef ENABLE_HIP
//! Pack particle data into a buffer (GPU version)
/*! \note This method may only be used during communication or when
 *        no ghost particles are present, because ghost particle values
 *        are undefined after calling this method.
 */
void ParticleData::removeParticlesGPU(GlobalVector<detail::pdata_element>& out,
                                      GlobalVector<unsigned int>& comm_flags)
    {
    // this is the maximum number of elements we can possibly write to out
    unsigned int max_n_out = (unsigned int)out.getNumElements();
    if (comm_flags.getNumElements() < max_n_out)
        max_n_out = (unsigned int)comm_flags.getNumElements();

    // allocate array if necessary
    if (!max_n_out)
        {
        out.resize(1);
        comm_flags.resize(1);
        max_n_out = (unsigned int)out.getNumElements();
        if (comm_flags.getNumElements() < max_n_out)
            max_n_out = (unsigned int)comm_flags.getNumElements();
        }

    // number of particles that are to be written out
    unsigned int n_out = 0;

    bool done = false;

    // copy without writing past the end of the output array, resizing it as needed
    while (!done)
        {
        // access particle data arrays to read from
        ArrayHandle<Scalar4> d_pos(getPositions(), access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_vel(getVelocities(), access_location::device, access_mode::read);
        ArrayHandle<Scalar3> d_accel(getAccelerations(),
                                     access_location::device,
                                     access_mode::read);
        ArrayHandle<Scalar> d_charge(getCharges(), access_location::device, access_mode::read);
        ArrayHandle<Scalar> d_diameter(getDiameters(), access_location::device, access_mode::read);
        ArrayHandle<int3> d_image(getImages(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_body(getBodies(), access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_orientation(getOrientationArray(),
                                           access_location::device,
                                           access_mode::read);
        ArrayHandle<Scalar4> d_angmom(getAngularMomentumArray(),
                                      access_location::device,
                                      access_mode::read);
        ArrayHandle<Scalar3> d_inertia(getMomentsOfInertiaArray(),
                                       access_location::device,
                                       access_mode::read);
        ArrayHandle<Scalar4> d_net_force(getNetForce(), access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_net_torque(getNetTorqueArray(),
                                          access_location::device,
                                          access_mode::read);
        ArrayHandle<Scalar> d_net_virial(getNetVirial(),
                                         access_location::device,
                                         access_mode::read);
        ArrayHandle<unsigned int> d_tag(getTags(), access_location::device, access_mode::read);

        // access alternate particle data arrays to write to
        ArrayHandle<Scalar4> d_pos_alt(m_pos_alt, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar4> d_vel_alt(m_vel_alt, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar3> d_accel_alt(m_accel_alt,
                                         access_location::device,
                                         access_mode::overwrite);
        ArrayHandle<Scalar> d_charge_alt(m_charge_alt,
                                         access_location::device,
                                         access_mode::overwrite);
        ArrayHandle<Scalar> d_diameter_alt(m_diameter_alt,
                                           access_location::device,
                                           access_mode::overwrite);
        ArrayHandle<int3> d_image_alt(m_image_alt, access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> d_body_alt(m_body_alt,
                                             access_location::device,
                                             access_mode::overwrite);
        ArrayHandle<Scalar4> d_orientation_alt(m_orientation_alt,
                                               access_location::device,
                                               access_mode::overwrite);
        ArrayHandle<Scalar4> d_angmom_alt(m_angmom_alt,
                                          access_location::device,
                                          access_mode::overwrite);
        ArrayHandle<Scalar3> d_inertia_alt(m_inertia_alt,
                                           access_location::device,
                                           access_mode::overwrite);
        ArrayHandle<Scalar4> d_net_force_alt(m_net_force_alt,
                                             access_location::device,
                                             access_mode::overwrite);
        ArrayHandle<Scalar4> d_net_torque_alt(m_net_torque_alt,
                                              access_location::device,
                                              access_mode::overwrite);
        ArrayHandle<Scalar> d_net_virial_alt(m_net_virial_alt,
                                             access_location::device,
                                             access_mode::overwrite);
        ArrayHandle<unsigned int> d_tag_alt(m_tag_alt,
                                            access_location::device,
                                            access_mode::overwrite);

        ArrayHandle<unsigned int> d_comm_flags(getCommFlags(),
                                               access_location::device,
                                               access_mode::readwrite);

        // Access reverse-lookup table
        ArrayHandle<unsigned int> d_rtag(getRTags(),
                                         access_location::device,
                                         access_mode::readwrite);

            {
            // Access output array
            ArrayHandle<detail::pdata_element> d_out(out,
                                                     access_location::device,
                                                     access_mode::overwrite);
            ArrayHandle<unsigned int> d_comm_flags_out(comm_flags,
                                                       access_location::device,
                                                       access_mode::overwrite);

            // get temporary buffer
            ScopedAllocation<unsigned int> d_tmp(m_exec_conf->getCachedAllocatorManaged(), getN());

            m_exec_conf->beginMultiGPU();

            n_out = kernel::gpu_pdata_remove(getN(),
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
                                             d_net_force.data,
                                             d_net_torque.data,
                                             d_net_virial.data,
                                             (unsigned int)getNetVirial().getPitch(),
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
                                             d_net_force_alt.data,
                                             d_net_torque_alt.data,
                                             d_net_virial_alt.data,
                                             d_tag_alt.data,
                                             d_out.data,
                                             d_comm_flags.data,
                                             d_comm_flags_out.data,
                                             max_n_out,
                                             d_tmp.data,
                                             m_exec_conf->getCachedAllocatorManaged(),
                                             m_gpu_partition);

            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();

            m_exec_conf->endMultiGPU();
            }

        // resize output vector
        out.resize(n_out);
        comm_flags.resize(n_out);

        // was the array large enough?
        if (n_out <= max_n_out)
            done = true;

        max_n_out = (unsigned int)out.getNumElements();
        if (comm_flags.getNumElements() < max_n_out)
            max_n_out = (unsigned int)comm_flags.getNumElements();
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
    swapMomentsOfInertia();
    swapNetForce();
    swapNetTorque();
    swapNetVirial();
    swapTags();

    // notify subscribers
    notifyParticleSort();
    }

//! Add new particle data (GPU version)
void ParticleData::addParticlesGPU(const GlobalVector<detail::pdata_element>& in)
    {
    unsigned int old_nparticles = getN();
    unsigned int num_add_ptls = (unsigned int)in.size();
    unsigned int new_nparticles = old_nparticles + num_add_ptls;

    // amortized resizing of particle data
    resize(new_nparticles);

        {
        // access particle data arrays
        ArrayHandle<Scalar4> d_pos(getPositions(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> d_vel(getVelocities(),
                                   access_location::device,
                                   access_mode::readwrite);
        ArrayHandle<Scalar3> d_accel(getAccelerations(),
                                     access_location::device,
                                     access_mode::readwrite);
        ArrayHandle<Scalar> d_charge(getCharges(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar> d_diameter(getDiameters(),
                                       access_location::device,
                                       access_mode::readwrite);
        ArrayHandle<int3> d_image(getImages(), access_location::device, access_mode::readwrite);
        ArrayHandle<unsigned int> d_body(getBodies(),
                                         access_location::device,
                                         access_mode::readwrite);
        ArrayHandle<Scalar4> d_orientation(getOrientationArray(),
                                           access_location::device,
                                           access_mode::readwrite);
        ArrayHandle<Scalar4> d_angmom(getAngularMomentumArray(),
                                      access_location::device,
                                      access_mode::readwrite);
        ArrayHandle<Scalar3> d_inertia(getMomentsOfInertiaArray(),
                                       access_location::device,
                                       access_mode::readwrite);
        ArrayHandle<Scalar4> d_net_force(getNetForce(),
                                         access_location::device,
                                         access_mode::readwrite);
        ArrayHandle<Scalar4> d_net_torque(getNetTorqueArray(),
                                          access_location::device,
                                          access_mode::readwrite);
        ArrayHandle<Scalar> d_net_virial(getNetVirial(),
                                         access_location::device,
                                         access_mode::readwrite);
        ArrayHandle<unsigned int> d_tag(getTags(), access_location::device, access_mode::readwrite);
        ArrayHandle<unsigned int> d_rtag(getRTags(),
                                         access_location::device,
                                         access_mode::readwrite);
        ArrayHandle<unsigned int> d_comm_flags(getCommFlags(),
                                               access_location::device,
                                               access_mode::readwrite);

        // Access input array
        ArrayHandle<detail::pdata_element> d_in(in, access_location::device, access_mode::read);

        // add new particles on GPU
        kernel::gpu_pdata_add_particles(old_nparticles,
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
                                        d_net_force.data,
                                        d_net_torque.data,
                                        d_net_virial.data,
                                        (unsigned int)getNetVirial().getPitch(),
                                        d_tag.data,
                                        d_rtag.data,
                                        d_in.data,
                                        d_comm_flags.data);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    // notify subscribers
    notifyParticleSort();
    }

#endif // ENABLE_HIP
#endif // ENABLE_MPI

void ParticleData::setGPUAdvice()
    {
#if defined(ENABLE_HIP) && defined(__HIP_PLATFORM_NVCC__)
    if (m_exec_conf->isCUDAEnabled())
        {
        // only call CUDA API when necessary
        if (m_memory_advice_last_Nmax == m_max_nparticles)
            return;

        m_memory_advice_last_Nmax = m_max_nparticles;

        auto gpu_map = m_exec_conf->getGPUIds();

        if (!m_exec_conf->allConcurrentManagedAccess())
            return;

        // split preferred location of particle data across GPUs
        for (unsigned int idev = 0; idev < m_exec_conf->getNumActiveGPUs(); ++idev)
            {
            auto range = m_gpu_partition.getRange(idev);
            unsigned int nelem = range.second - range.first;

            if (!nelem)
                continue;

            cudaMemAdvise(m_pos.get() + range.first,
                          sizeof(Scalar4) * nelem,
                          cudaMemAdviseSetPreferredLocation,
                          gpu_map[idev]);
            cudaMemAdvise(m_vel.get() + range.first,
                          sizeof(Scalar4) * nelem,
                          cudaMemAdviseSetPreferredLocation,
                          gpu_map[idev]);
            cudaMemAdvise(m_accel.get() + range.first,
                          sizeof(Scalar3) * nelem,
                          cudaMemAdviseSetPreferredLocation,
                          gpu_map[idev]);
            cudaMemAdvise(m_charge.get() + range.first,
                          sizeof(Scalar) * nelem,
                          cudaMemAdviseSetPreferredLocation,
                          gpu_map[idev]);
            cudaMemAdvise(m_diameter.get() + range.first,
                          sizeof(Scalar) * nelem,
                          cudaMemAdviseSetPreferredLocation,
                          gpu_map[idev]);
            cudaMemAdvise(m_image.get() + range.first,
                          sizeof(int3) * nelem,
                          cudaMemAdviseSetPreferredLocation,
                          gpu_map[idev]);
            cudaMemAdvise(m_tag.get() + range.first,
                          sizeof(unsigned int) * nelem,
                          cudaMemAdviseSetPreferredLocation,
                          gpu_map[idev]);
            cudaMemAdvise(m_body.get() + range.first,
                          sizeof(unsigned int) * nelem,
                          cudaMemAdviseSetPreferredLocation,
                          gpu_map[idev]);
            cudaMemAdvise(m_orientation.get() + range.first,
                          sizeof(Scalar4) * nelem,
                          cudaMemAdviseSetPreferredLocation,
                          gpu_map[idev]);
            cudaMemAdvise(m_angmom.get() + range.first,
                          sizeof(Scalar4) * nelem,
                          cudaMemAdviseSetPreferredLocation,
                          gpu_map[idev]);
            cudaMemAdvise(m_inertia.get() + range.first,
                          sizeof(Scalar3) * nelem,
                          cudaMemAdviseSetPreferredLocation,
                          gpu_map[idev]);
            cudaMemAdvise(m_net_force.get() + range.first,
                          sizeof(Scalar4) * nelem,
                          cudaMemAdviseSetPreferredLocation,
                          gpu_map[idev]);
            for (unsigned int i = 0; i < 6; ++i)
                cudaMemAdvise(m_net_virial.get() + i * m_net_virial.getPitch() + range.first,
                              sizeof(Scalar) * nelem,
                              cudaMemAdviseSetPreferredLocation,
                              gpu_map[idev]);
            cudaMemAdvise(m_net_torque.get() + range.first,
                          sizeof(Scalar4) * nelem,
                          cudaMemAdviseSetPreferredLocation,
                          gpu_map[idev]);

            // migrate data to preferred location
            cudaMemPrefetchAsync(m_pos.get() + range.first, sizeof(Scalar4) * nelem, gpu_map[idev]);
            cudaMemPrefetchAsync(m_vel.get() + range.first, sizeof(Scalar4) * nelem, gpu_map[idev]);
            cudaMemPrefetchAsync(m_accel.get() + range.first,
                                 sizeof(Scalar3) * nelem,
                                 gpu_map[idev]);
            cudaMemPrefetchAsync(m_charge.get() + range.first,
                                 sizeof(Scalar) * nelem,
                                 gpu_map[idev]);
            cudaMemPrefetchAsync(m_diameter.get() + range.first,
                                 sizeof(Scalar) * nelem,
                                 gpu_map[idev]);
            cudaMemPrefetchAsync(m_image.get() + range.first, sizeof(int3) * nelem, gpu_map[idev]);
            cudaMemPrefetchAsync(m_tag.get() + range.first,
                                 sizeof(unsigned int) * nelem,
                                 gpu_map[idev]);
            cudaMemPrefetchAsync(m_body.get() + range.first,
                                 sizeof(unsigned int) * nelem,
                                 gpu_map[idev]);
            cudaMemPrefetchAsync(m_orientation.get() + range.first,
                                 sizeof(Scalar4) * nelem,
                                 gpu_map[idev]);
            cudaMemPrefetchAsync(m_angmom.get() + range.first,
                                 sizeof(Scalar4) * nelem,
                                 gpu_map[idev]);
            cudaMemPrefetchAsync(m_inertia.get() + range.first,
                                 sizeof(Scalar3) * nelem,
                                 gpu_map[idev]);
            cudaMemPrefetchAsync(m_net_force.get() + range.first,
                                 sizeof(Scalar4) * nelem,
                                 gpu_map[idev]);
            for (unsigned int i = 0; i < 6; ++i)
                cudaMemPrefetchAsync(m_net_virial.get() + i * m_net_virial.getPitch() + range.first,
                                     sizeof(Scalar) * nelem,
                                     gpu_map[idev]);
            cudaMemPrefetchAsync(m_net_torque.get() + range.first,
                                 sizeof(Scalar4) * nelem,
                                 gpu_map[idev]);
            }
        CHECK_CUDA_ERROR();

        if (!m_pos_alt.isNull())
            {
            for (unsigned int idev = 0; idev < m_exec_conf->getNumActiveGPUs(); ++idev)
                {
                auto range = m_gpu_partition.getRange(idev);
                unsigned int nelem = range.second - range.first;

                if (!nelem)
                    continue;

                cudaMemAdvise(m_pos_alt.get() + range.first,
                              sizeof(Scalar4) * nelem,
                              cudaMemAdviseSetPreferredLocation,
                              gpu_map[idev]);
                cudaMemAdvise(m_vel_alt.get() + range.first,
                              sizeof(Scalar4) * nelem,
                              cudaMemAdviseSetPreferredLocation,
                              gpu_map[idev]);
                cudaMemAdvise(m_accel_alt.get() + range.first,
                              sizeof(Scalar3) * nelem,
                              cudaMemAdviseSetPreferredLocation,
                              gpu_map[idev]);
                cudaMemAdvise(m_charge_alt.get() + range.first,
                              sizeof(Scalar) * nelem,
                              cudaMemAdviseSetPreferredLocation,
                              gpu_map[idev]);
                cudaMemAdvise(m_diameter_alt.get() + range.first,
                              sizeof(Scalar) * nelem,
                              cudaMemAdviseSetPreferredLocation,
                              gpu_map[idev]);
                cudaMemAdvise(m_image_alt.get() + range.first,
                              sizeof(int3) * nelem,
                              cudaMemAdviseSetPreferredLocation,
                              gpu_map[idev]);
                cudaMemAdvise(m_tag_alt.get() + range.first,
                              sizeof(unsigned int) * nelem,
                              cudaMemAdviseSetPreferredLocation,
                              gpu_map[idev]);
                cudaMemAdvise(m_body_alt.get() + range.first,
                              sizeof(unsigned int) * nelem,
                              cudaMemAdviseSetPreferredLocation,
                              gpu_map[idev]);
                cudaMemAdvise(m_orientation_alt.get() + range.first,
                              sizeof(Scalar4) * nelem,
                              cudaMemAdviseSetPreferredLocation,
                              gpu_map[idev]);
                cudaMemAdvise(m_angmom_alt.get() + range.first,
                              sizeof(Scalar4) * nelem,
                              cudaMemAdviseSetPreferredLocation,
                              gpu_map[idev]);
                cudaMemAdvise(m_inertia_alt.get() + range.first,
                              sizeof(Scalar3) * nelem,
                              cudaMemAdviseSetPreferredLocation,
                              gpu_map[idev]);
                cudaMemAdvise(m_net_force_alt.get() + range.first,
                              sizeof(Scalar4) * nelem,
                              cudaMemAdviseSetPreferredLocation,
                              gpu_map[idev]);
                for (unsigned int i = 0; i < 6; ++i)
                    cudaMemAdvise(m_net_virial_alt.get() + i * m_net_virial_alt.getPitch()
                                      + range.first,
                                  sizeof(Scalar) * nelem,
                                  cudaMemAdviseSetPreferredLocation,
                                  gpu_map[idev]);
                cudaMemAdvise(m_net_torque_alt.get() + range.first,
                              sizeof(Scalar4) * nelem,
                              cudaMemAdviseSetPreferredLocation,
                              gpu_map[idev]);

                cudaMemPrefetchAsync(m_pos_alt.get() + range.first,
                                     sizeof(Scalar4) * nelem,
                                     gpu_map[idev]);
                cudaMemPrefetchAsync(m_vel_alt.get() + range.first,
                                     sizeof(Scalar4) * nelem,
                                     gpu_map[idev]);
                cudaMemPrefetchAsync(m_accel_alt.get() + range.first,
                                     sizeof(Scalar3) * nelem,
                                     gpu_map[idev]);
                cudaMemPrefetchAsync(m_charge_alt.get() + range.first,
                                     sizeof(Scalar) * nelem,
                                     gpu_map[idev]);
                cudaMemPrefetchAsync(m_diameter_alt.get() + range.first,
                                     sizeof(Scalar) * nelem,
                                     gpu_map[idev]);
                cudaMemPrefetchAsync(m_image_alt.get() + range.first,
                                     sizeof(int3) * nelem,
                                     gpu_map[idev]);
                cudaMemPrefetchAsync(m_tag_alt.get() + range.first,
                                     sizeof(unsigned int) * nelem,
                                     gpu_map[idev]);
                cudaMemPrefetchAsync(m_body_alt.get() + range.first,
                                     sizeof(unsigned int) * nelem,
                                     gpu_map[idev]);
                cudaMemPrefetchAsync(m_orientation_alt.get() + range.first,
                                     sizeof(Scalar4) * nelem,
                                     gpu_map[idev]);
                cudaMemPrefetchAsync(m_angmom_alt.get() + range.first,
                                     sizeof(Scalar4) * nelem,
                                     gpu_map[idev]);
                cudaMemPrefetchAsync(m_inertia_alt.get() + range.first,
                                     sizeof(Scalar3) * nelem,
                                     gpu_map[idev]);
                cudaMemPrefetchAsync(m_net_force_alt.get() + range.first,
                                     sizeof(Scalar4) * nelem,
                                     gpu_map[idev]);
                for (unsigned int i = 0; i < 6; ++i)
                    cudaMemPrefetchAsync(m_net_virial_alt.get() + i * m_net_virial_alt.getPitch()
                                             + range.first,
                                         sizeof(Scalar) * nelem,
                                         gpu_map[idev]);
                cudaMemPrefetchAsync(m_net_torque_alt.get() + range.first,
                                     sizeof(Scalar4) * nelem,
                                     gpu_map[idev]);
                }

            CHECK_CUDA_ERROR();
            }
        }
#endif
    }

template<class Real>
void SnapshotParticleData<Real>::replicate(unsigned int nx,
                                           unsigned int ny,
                                           unsigned int nz,
                                           const BoxDim& old_box,
                                           const BoxDim& new_box)
    {
    unsigned int old_size = size;

    // resize snapshot
    resize(old_size * nx * ny * nz);

    for (unsigned int i = 0; i < old_size; ++i)
        {
        // unwrap position of particle i in old box using image flags
        vec3<Real> p = pos[i];
        int3 img = image[i];

        // need to cast to a scalar and back because the Box is in Scalars, but we might be in a
        // different type
        p = vec3<Real>(old_box.shift(vec3<Scalar>(p), img));
        vec3<Real> f = old_box.makeFraction(p);

        unsigned int j = 0;
        for (unsigned int l = 0; l < nx; l++)
            for (unsigned int m = 0; m < ny; m++)
                for (unsigned int n = 0; n < nz; n++)
                    {
                    Scalar3 f_new;
                    // replicate particle
                    f_new.x = f.x / (Real)nx + (Real)l / (Real)nx;
                    f_new.y = f.y / (Real)ny + (Real)m / (Real)ny;
                    f_new.z = f.z / (Real)nz + (Real)n / (Real)nz;

                    unsigned int k = j * old_size + i;

                    // coordinates in new box
                    Scalar3 q = new_box.makeCoordinates(f_new);

                    // wrap by multiple box vectors if necessary
                    image[k] = new_box.getImage(q);
                    int3 negimg = make_int3(-image[k].x, -image[k].y, -image[k].z);
                    q = new_box.shift(q, negimg);

                    // rewrap using wrap so that rounding is consistent
                    new_box.wrap(q, image[k]);

                    pos[k] = vec3<Real>(q);
                    vel[k] = vel[i];
                    accel[k] = accel[i];
                    type[k] = type[i];
                    mass[k] = mass[i];
                    charge[k] = charge[i];
                    diameter[k] = diameter[i];
                    // This math also accounts for floppy bodies since body[i]
                    // is already greater than MIN_FLOPPY, so the new body id
                    // body[k] is guaranteed to be so as well. However, we
                    // check to ensure that something that wasn't originally a
                    // floppy body doesn't overflow into the floppy body tags.
                    body[k] = (body[i] != NO_BODY ? j * old_size + body[i] : NO_BODY);
                    if (body[i] < MIN_FLOPPY && body[k] >= MIN_FLOPPY)
                        throw std::runtime_error("Replication would create more distinct rigid "
                                                 "bodies than HOOMD supports!");
                    orientation[k] = orientation[i];
                    angmom[k] = angmom[i];
                    inertia[k] = inertia[i];
                    j++;
                    }
        }
    }

template<class Real>
void SnapshotParticleData<Real>::replicate(unsigned int nx,
                                           unsigned int ny,
                                           unsigned int nz,
                                           std::shared_ptr<const BoxDim> old_box,
                                           std::shared_ptr<const BoxDim> new_box)
    {
    replicate(nx, ny, nz, *old_box, *new_box);
    }

/*! \returns a numpy array that wraps the pos data element.
    The raw data is referenced by the numpy array, modifications to the numpy array will modify the
   snapshot
*/
template<class Real> pybind11::object SnapshotParticleData<Real>::getPosNP(pybind11::object self)
    {
    auto self_cpp = self.cast<SnapshotParticleData<Real>*>();
    // mark as dirty when accessing internal data
    self_cpp->is_accel_set = false;

    std::vector<size_t> dims(2);
    dims[0] = self_cpp->pos.size();
    dims[1] = 3;
    if (dims[0] == 0)
        {
        return pybind11::array(pybind11::dtype::of<Real>(), dims, nullptr);
        }
    return pybind11::array(dims, (Real*)self_cpp->pos.data(), self);
    }

/*! \returns a numpy array that wraps the pos data element.
    The raw data is referenced by the numpy array, modifications to the numpy array will modify the
   snapshot
*/
template<class Real> pybind11::object SnapshotParticleData<Real>::getVelNP(pybind11::object self)
    {
    auto self_cpp = self.cast<SnapshotParticleData<Real>*>();
    // mark as dirty when accessing internal data
    self_cpp->is_accel_set = false;

    std::vector<size_t> dims(2);
    dims[0] = self_cpp->pos.size();
    dims[1] = 3;
    if (dims[0] == 0)
        {
        return pybind11::array(pybind11::dtype::of<Real>(), dims, nullptr);
        }
    return pybind11::array(dims, (Real*)self_cpp->vel.data(), self);
    }

/*! \returns a numpy array that wraps the pos data element.
    The raw data is referenced by the numpy array, modifications to the numpy array will modify the
   snapshot
*/
template<class Real> pybind11::object SnapshotParticleData<Real>::getAccelNP(pybind11::object self)
    {
    auto self_cpp = self.cast<SnapshotParticleData<Real>*>();
    // mark as dirty when accessing internal data
    self_cpp->is_accel_set = false;

    std::vector<size_t> dims(2);
    dims[0] = self_cpp->pos.size();
    dims[1] = 3;
    if (dims[0] == 0)
        {
        return pybind11::array(pybind11::dtype::of<Real>(), dims, nullptr);
        }
    return pybind11::array(dims, (Real*)self_cpp->accel.data(), self);
    }

/*! \returns a numpy array that wraps the type data element.
    The raw data is referenced by the numpy array, modifications to the numpy array will modify the
   snapshot
*/
template<class Real> pybind11::object SnapshotParticleData<Real>::getTypeNP(pybind11::object self)
    {
    auto self_cpp = self.cast<SnapshotParticleData<Real>*>();
    // mark as dirty when accessing internal data
    self_cpp->is_accel_set = false;

    if (self_cpp->type.size() == 0)
        {
        return pybind11::array(pybind11::dtype::of<unsigned int>(), 0, nullptr);
        }
    return pybind11::array(self_cpp->type.size(), self_cpp->type.data(), self);
    }

/*! \returns a numpy array that wraps the mass data element.
    The raw data is referenced by the numpy array, modifications to the numpy array will modify the
   snapshot
*/
template<class Real> pybind11::object SnapshotParticleData<Real>::getMassNP(pybind11::object self)
    {
    auto self_cpp = self.cast<SnapshotParticleData<Real>*>();
    // mark as dirty when accessing internal data
    self_cpp->is_accel_set = false;

    if (self_cpp->mass.size() == 0)
        {
        return pybind11::array(pybind11::dtype::of<Real>(), 0, nullptr);
        }
    return pybind11::array(self_cpp->mass.size(), self_cpp->mass.data(), self);
    }

/*! \returns a numpy array that wraps the charge data element.
    The raw data is referenced by the numpy array, modifications to the numpy array will modify the
   snapshot
*/
template<class Real> pybind11::object SnapshotParticleData<Real>::getChargeNP(pybind11::object self)
    {
    auto self_cpp = self.cast<SnapshotParticleData<Real>*>();
    // mark as dirty when accessing internal data
    self_cpp->is_accel_set = false;

    if (self_cpp->charge.size() == 0)
        {
        return pybind11::array(pybind11::dtype::of<Real>(), 0, nullptr);
        }
    return pybind11::array(self_cpp->charge.size(), self_cpp->charge.data(), self);
    }

/*! \returns a numpy array that wraps the diameter data element.
    The raw data is referenced by the numpy array, modifications to the numpy array will modify the
   snapshot
*/
template<class Real>
pybind11::object SnapshotParticleData<Real>::getDiameterNP(pybind11::object self)
    {
    auto self_cpp = self.cast<SnapshotParticleData<Real>*>();
    // mark as dirty when accessing internal data
    self_cpp->is_accel_set = false;

    if (self_cpp->diameter.size() == 0)
        {
        return pybind11::array(pybind11::dtype::of<Real>(), 0, nullptr);
        }
    return pybind11::array(self_cpp->diameter.size(), self_cpp->diameter.data(), self);
    }

/*! \returns a numpy array that wraps the image data element.
    The raw data is referenced by the numpy array, modifications to the numpy array will modify the
   snapshot
*/
template<class Real> pybind11::object SnapshotParticleData<Real>::getImageNP(pybind11::object self)
    {
    auto self_cpp = self.cast<SnapshotParticleData<Real>*>();
    // mark as dirty when accessing internal data
    self_cpp->is_accel_set = false;

    std::vector<size_t> dims(2);
    dims[0] = self_cpp->pos.size();
    dims[1] = 3;

    if (dims[0] == 0)
        {
        return pybind11::array(pybind11::dtype::of<int>(), dims, nullptr);
        }
    return pybind11::array(dims, (int*)self_cpp->image.data(), self);
    }

/*! \returns a numpy array that wraps the body data element.
    The raw data is referenced by the numpy array, modifications to the numpy array will modify the
   snapshot
*/
template<class Real> pybind11::object SnapshotParticleData<Real>::getBodyNP(pybind11::object self)
    {
    auto self_cpp = self.cast<SnapshotParticleData<Real>*>();
    // mark as dirty when accessing internal data
    self_cpp->is_accel_set = false;

    if (self_cpp->body.size() == 0)
        {
        return pybind11::array(pybind11::dtype::of<Real>(), 0, nullptr);
        }
    return pybind11::array(self_cpp->body.size(), (int*)self_cpp->body.data(), self);
    }

/*! \returns a numpy array that wraps the orientation data element.
    The raw data is referenced by the numpy array, modifications to the numpy array will modify the
   snapshot
*/
template<class Real>
pybind11::object SnapshotParticleData<Real>::getOrientationNP(pybind11::object self)
    {
    auto self_cpp = self.cast<SnapshotParticleData<Real>*>();
    // mark as dirty when accessing internal data
    self_cpp->is_accel_set = false;

    std::vector<size_t> dims(2);
    dims[0] = self_cpp->pos.size();
    dims[1] = 4;

    if (dims[0] == 0)
        {
        return pybind11::array(pybind11::dtype::of<Real>(), dims, nullptr);
        }
    return pybind11::array(dims, (Real*)self_cpp->orientation.data(), self);
    }

/*! \returns a numpy array that wraps the moment of inertia data element.
    The raw data is referenced by the numpy array, modifications to the numpy array will modify the
   snapshot
*/
template<class Real>
pybind11::object SnapshotParticleData<Real>::getMomentInertiaNP(pybind11::object self)
    {
    auto self_cpp = self.cast<SnapshotParticleData<Real>*>();
    // mark as dirty when accessing internal data
    self_cpp->is_accel_set = false;

    std::vector<size_t> dims(2);
    dims[0] = self_cpp->inertia.size();
    dims[1] = 3;

    if (dims[0] == 0)
        {
        return pybind11::array(pybind11::dtype::of<Real>(), dims, nullptr);
        }
    return pybind11::array(dims, (Real*)self_cpp->inertia.data(), self);
    }

/*! \returns a numpy array that wraps the angular momentum data element.
    The raw data is referenced by the numpy array, modifications to the numpy array will modify the
   snapshot
*/
template<class Real> pybind11::object SnapshotParticleData<Real>::getAngmomNP(pybind11::object self)
    {
    auto self_cpp = self.cast<SnapshotParticleData<Real>*>();
    // mark as dirty when accessing internal data
    self_cpp->is_accel_set = false;

    std::vector<size_t> dims(2);
    dims[0] = self_cpp->angmom.size();
    dims[1] = 4;

    if (dims[0] == 0)
        {
        return pybind11::array(pybind11::dtype::of<Real>(), dims, nullptr);
        }
    return pybind11::array(dims, (Real*)self_cpp->angmom.data(), self);
    }

/*! \returns A python list of type names
 */
template<class Real> pybind11::list SnapshotParticleData<Real>::getTypes()
    {
    pybind11::list types;

    for (unsigned int i = 0; i < type_mapping.size(); i++)
        types.append(pybind11::str(type_mapping[i]));

    return types;
    }

/*! \param types Python list of type names to set
 */
template<class Real> void SnapshotParticleData<Real>::setTypes(pybind11::list types)
    {
    // set dirty
    is_accel_set = false;

    type_mapping.resize(len(types));

    for (unsigned int i = 0; i < len(types); i++)
        type_mapping[i] = pybind11::cast<string>(types[i]);
    }

#ifdef ENABLE_MPI
template<class Real> void SnapshotParticleData<Real>::bcast(unsigned int root, MPI_Comm mpi_comm)
    {
    // broadcast all member quantities
    hoomd::bcast(pos, root, mpi_comm);
    hoomd::bcast(vel, root, mpi_comm);
    hoomd::bcast(accel, root, mpi_comm);
    hoomd::bcast(type, root, mpi_comm);
    hoomd::bcast(mass, root, mpi_comm);
    hoomd::bcast(charge, root, mpi_comm);
    hoomd::bcast(diameter, root, mpi_comm);
    hoomd::bcast(image, root, mpi_comm);
    hoomd::bcast(body, root, mpi_comm);
    hoomd::bcast(orientation, root, mpi_comm);
    hoomd::bcast(angmom, root, mpi_comm);
    hoomd::bcast(inertia, root, mpi_comm);

    hoomd::bcast(size, root, mpi_comm);
    hoomd::bcast(type_mapping, root, mpi_comm);
    hoomd::bcast(is_accel_set, root, mpi_comm);
    }
#endif

// instantiate both float and double snapshots
template struct SnapshotParticleData<float>;
template struct SnapshotParticleData<double>;

namespace detail
    {
void export_SnapshotParticleData(pybind11::module& m)
    {
    pybind11::class_<SnapshotParticleData<float>, std::shared_ptr<SnapshotParticleData<float>>>(
        m,
        "SnapshotParticleData_float")
        .def(pybind11::init<unsigned int>())
        .def_property_readonly("position", &SnapshotParticleData<float>::getPosNP)
        .def_property_readonly("velocity", &SnapshotParticleData<float>::getVelNP)
        .def_property_readonly("acceleration", &SnapshotParticleData<float>::getAccelNP)
        .def_property_readonly("typeid", &SnapshotParticleData<float>::getTypeNP)
        .def_property_readonly("mass", &SnapshotParticleData<float>::getMassNP)
        .def_property_readonly("charge", &SnapshotParticleData<float>::getChargeNP)
        .def_property_readonly("diameter", &SnapshotParticleData<float>::getDiameterNP)
        .def_property_readonly("image", &SnapshotParticleData<float>::getImageNP)
        .def_property_readonly("body", &SnapshotParticleData<float>::getBodyNP)
        .def_property_readonly("orientation", &SnapshotParticleData<float>::getOrientationNP)
        .def_property_readonly("moment_inertia", &SnapshotParticleData<float>::getMomentInertiaNP)
        .def_property_readonly("angmom", &SnapshotParticleData<float>::getAngmomNP)
        .def_property("types",
                      &SnapshotParticleData<float>::getTypes,
                      &SnapshotParticleData<float>::setTypes)
        .def_property("N",
                      &SnapshotParticleData<float>::getSize,
                      &SnapshotParticleData<float>::resize)
        .def_readonly("is_accel_set", &SnapshotParticleData<float>::is_accel_set);

    pybind11::class_<SnapshotParticleData<double>, std::shared_ptr<SnapshotParticleData<double>>>(
        m,
        "SnapshotParticleData_double")
        .def(pybind11::init<unsigned int>())
        .def_property_readonly("position", &SnapshotParticleData<double>::getPosNP)
        .def_property_readonly("velocity", &SnapshotParticleData<double>::getVelNP)
        .def_property_readonly("acceleration", &SnapshotParticleData<double>::getAccelNP)
        .def_property_readonly("typeid", &SnapshotParticleData<double>::getTypeNP)
        .def_property_readonly("mass", &SnapshotParticleData<double>::getMassNP)
        .def_property_readonly("charge", &SnapshotParticleData<double>::getChargeNP)
        .def_property_readonly("diameter", &SnapshotParticleData<double>::getDiameterNP)
        .def_property_readonly("image", &SnapshotParticleData<double>::getImageNP)
        .def_property_readonly("body", &SnapshotParticleData<double>::getBodyNP)
        .def_property_readonly("orientation", &SnapshotParticleData<double>::getOrientationNP)
        .def_property_readonly("moment_inertia", &SnapshotParticleData<double>::getMomentInertiaNP)
        .def_property_readonly("angmom", &SnapshotParticleData<double>::getAngmomNP)
        .def_property("types",
                      &SnapshotParticleData<double>::getTypes,
                      &SnapshotParticleData<double>::setTypes)
        .def_property("N",
                      &SnapshotParticleData<double>::getSize,
                      &SnapshotParticleData<double>::resize)
        .def_readonly("is_accel_set", &SnapshotParticleData<double>::is_accel_set);
    }

    } // end namespace detail

    } // end namespace hoomd
