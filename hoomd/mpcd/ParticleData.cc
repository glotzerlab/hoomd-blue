// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: mphoward

/*!
 * \file mpcd/ParticleData.h
 * \brief Definition of mpcd::ParticleData
 */

#include "ParticleData.h"

#ifdef ENABLE_CUDA
#include "hoomd/CachedAllocator.h"
#endif

#ifdef ENABLE_MPI
#include "hoomd/HOOMDMPI.h"
#endif // ENABLE_MPI

#include <iomanip>
using namespace std;

// 9/8 factor for amortized growth
const float mpcd::ParticleData::resize_factor = 9./8.;

/*!
 * \param N Number of MPCD particles
 * \param n_types Number of MPCD particle types
 * \param global_box Global simulation box
 * \param exec_conf Execution configuration
 * \param decomposition Domain decomposition
 *
 * If \a n_types is zero, it is automatically incremented to 1. Default type
 * names A, B, C, ... are created. The simulation is initialized from a default
 * snapshot.
 */
mpcd::ParticleData::ParticleData(unsigned int N,
                                 unsigned int n_types,
                                 const BoxDim& global_box,
                                 std::shared_ptr<ExecutionConfiguration> exec_conf,
                                 std::shared_ptr<DomainDecomposition> decomposition)
    : m_N(0), m_N_global(0), m_N_max(0), m_exec_conf(exec_conf), m_mass(1.0), m_valid_cell_cache(false)
    {
    m_exec_conf->msg->notice(5) << "Constructing MPCD ParticleData" << endl;

    // set domain decomposition
    #ifdef ENABLE_MPI
    if (decomposition)
        m_decomposition = decomposition;
    #endif // ENABLE_MPI

    // construct snapshot with default type mapping (A, B, C, ...)
    mpcd::ParticleDataSnapshot snapshot(N);
    unsigned int eff_n_types = n_types;
    if (m_exec_conf->getRank() == 0 && eff_n_types == 0)
        {
        m_exec_conf->msg->warning() << "Number of MPCD types is 0, incrementing to 1" << std::endl;
        ++eff_n_types;
        }
    for (unsigned int i=0; i < eff_n_types; ++i)
        {
        char name[2];
        name[0] = 'A' + i;
        name[1] = '\0';
        snapshot.type_mapping.push_back(string(name));
        }

    initializeFromSnapshot(snapshot, global_box);

    #ifdef ENABLE_CUDA
    if (m_exec_conf->isCUDAEnabled())
        {
        // create a ModernGPU context
        m_mgpu_context = mgpu::CreateCudaDeviceAttachStream(0);
        }
    #endif
    }

/*!
 * \param snapshot Snapshot of the MPCD particle data
 * \param global_box Global simulation box
 * \param exec_conf Execution configuration
 * \param decomposition Domain decomposition
 */
mpcd::ParticleData::ParticleData(mpcd::ParticleDataSnapshot& snapshot,
                                 const BoxDim& global_box,
                                 std::shared_ptr<const ExecutionConfiguration> exec_conf,
                                 std::shared_ptr<DomainDecomposition> decomposition)
    : m_N(0), m_N_global(0), m_N_max(0), m_exec_conf(exec_conf), m_mass(1.0), m_valid_cell_cache(false)
    {
    m_exec_conf->msg->notice(5) << "Constructing MPCD ParticleData" << endl;

    // set domain decomposition
    #ifdef ENABLE_MPI
    if (decomposition)
        m_decomposition = decomposition;
    #endif

    if (m_exec_conf->getRank() == 0 && snapshot.type_mapping.size() == 0)
        {
        m_exec_conf->msg->warning() << "Number of MPCD types in snapshot is 0, incrementing to 1" << std::endl;
        snapshot.type_mapping.push_back("A");
        }

    initializeFromSnapshot(snapshot, global_box);

    #ifdef ENABLE_CUDA
    if (m_exec_conf->isCUDAEnabled())
        {
        // create a ModernGPU context
        m_mgpu_context = mgpu::CreateCudaDeviceAttachStream(0);
        }
    #endif
    }

mpcd::ParticleData::~ParticleData()
    {
    m_exec_conf->msg->notice(5) << "Destroying MPCD ParticleData" << endl;
    }

/*!
 * \param snapshot mpcd::ParticleDataSnapshot to read on the root (0) rank
 * \param global_box Current global box
 * \post The MPCD particle data has been initialized across all ranks with
 *       the contents of \a snapshot.
 *
 * This is a collective call, requiring participation from all ranks even if
 * the snapshot is only valid on the root rank.
 */
void mpcd::ParticleData::initializeFromSnapshot(const mpcd::ParticleDataSnapshot& snapshot,
                                                const BoxDim& global_box)
    {
    m_exec_conf->msg->notice(4) << "MPCD ParticleData: initializing from snapshot" << std::endl;

    if (!checkSnapshot(snapshot))
        {
        m_exec_conf->msg->error() << "Invalid MPCD particle data snapshot" << std::endl;
        throw std::runtime_error("Error initializing MPCD particle data.");
        }

    if (!checkInBox(snapshot, global_box))
        {
        m_exec_conf->msg->error() << "Not all MPCD particles were found inside the box" << endl;
        throw runtime_error("Error initializing MPCD particle data");
        }

    // global number of particles
    unsigned int nglobal(0);

    #ifdef ENABLE_MPI
    if (m_decomposition)
        {
        // Define per-processor particle data
        std::vector< std::vector<Scalar3> > pos_proc;              // Position array of every processor
        std::vector< std::vector<Scalar3> > vel_proc;              // Velocities array of every processor
        std::vector< std::vector<unsigned int> > type_proc;        // Particle types array of every processor
        std::vector< std::vector<unsigned int> > tag_proc;         // Global tags array of every processor
        std::vector<unsigned int> N_proc;                          // Number of particles on every processor

        // resize to number of ranks in communicator
        const MPI_Comm mpi_comm = m_exec_conf->getMPICommunicator();
        unsigned int n_ranks = m_exec_conf->getNRanks();
        unsigned int rank = m_exec_conf->getRank();
        pos_proc.resize(n_ranks);
        vel_proc.resize(n_ranks);
        type_proc.resize(n_ranks);
        tag_proc.resize(n_ranks);
        N_proc.resize(n_ranks,0);

        // scatter information to all processors from rank 0 (root)
        const unsigned int root = 0;
        if (rank == root)
            {
            const Index3D& di = m_decomposition->getDomainIndexer();
            unsigned int n_ranks = m_exec_conf->getNRanks();

            // loop over particles in snapshot, place them into domains
            for (auto it = snapshot.position.begin(); it != snapshot.position.end(); ++it)
                {
                unsigned int snap_idx = it - snapshot.position.begin();

                // determine domain the particle is placed into
                Scalar3 pos = vec_to_scalar3(*it);
                Scalar3 f = global_box.makeFraction(pos);
                int i= f.x * ((Scalar)di.getW());
                int j= f.y * ((Scalar)di.getH());
                int k= f.z * ((Scalar)di.getD());

                // wrap particles that are exactly on a boundary
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
                uchar3 periodic = make_uchar3(flags.x,flags.y,flags.z);
                BoxDim wrapping_box = global_box;
                wrapping_box.setPeriodic(periodic);
                int3 img = make_int3(0,0,0);
                wrapping_box.wrap(pos, img, flags);

                // place particle into the computational domain
                unsigned int rank = m_decomposition->placeParticle(global_box, pos);
                if (rank >= n_ranks)
                    {
                    m_exec_conf->msg->error() << "init.*: Particle " << snap_idx << " out of bounds." << std::endl;
                    m_exec_conf->msg->error() << "Cartesian coordinates: " << std::endl;
                    m_exec_conf->msg->error() << "x: " << pos.x << " y: " << pos.y << " z: " << pos.z << std::endl;
                    m_exec_conf->msg->error() << "Fractional coordinates: " << std::endl;
                    m_exec_conf->msg->error() << "f.x: " << f.x << " f.y: " << f.y << " f.z: " << f.z << std::endl;
                    Scalar3 lo = global_box.getLo();
                    Scalar3 hi = global_box.getHi();
                    m_exec_conf->msg->error() << "Global box lo: (" << lo.x << ", " << lo.y << ", " << lo.z << ")" << std::endl;
                    m_exec_conf->msg->error() << "           hi: (" << hi.x << ", " << hi.y << ", " << hi.z << ")" << std::endl;

                    throw std::runtime_error("Error initializing from snapshot.");
                    }

                // fill up per-processor data structures
                pos_proc[rank].push_back(pos);
                vel_proc[rank].push_back(vec_to_scalar3(snapshot.velocity[snap_idx]));
                type_proc[rank].push_back(snapshot.type[snap_idx]);
                tag_proc[rank].push_back(nglobal++);
                ++N_proc[rank];
                }

            // mass is set equal for all particles
            m_mass = snapshot.mass;
            }

        // get type mapping
        m_type_mapping = snapshot.type_mapping;

        if (rank != root)
            {
            m_type_mapping.clear();
            }

        // broadcast the particle mass
        bcast(m_mass, root, mpi_comm);

        // broadcast type mapping
        bcast(m_type_mapping, root, mpi_comm);

        // broadcast global number of particles
        bcast(nglobal, root, mpi_comm);

        // Local particle data
        std::vector<Scalar3> pos;
        std::vector<Scalar3> vel;
        std::vector<unsigned int> type;
        std::vector<unsigned int> tag;

        // distribute particle data to processors
        scatter_v(pos_proc,pos, root, mpi_comm);
        scatter_v(vel_proc,vel, root, mpi_comm);
        scatter_v(type_proc, type, root, mpi_comm);
        scatter_v(tag_proc, tag, root, mpi_comm);
        scatter_v(N_proc, m_N, root, mpi_comm);

        // we have to allocate even if the number of particles on a processor
        // is zero, so that the arrays can be resized later
        if (m_N == 0)
            allocate(1);
        else
            allocate(m_N);

        // Fill-up particle data arrays
        ArrayHandle<Scalar4> h_pos(m_pos, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar4> h_vel(m_vel, access_location::host, access_mode::overwrite);
        ArrayHandle<unsigned int> h_tag(m_tag, access_location::host, access_mode::overwrite);
        ArrayHandle<unsigned int> h_comm_flag(m_comm_flags, access_location::host, access_mode::overwrite);
        for (unsigned int idx = 0; idx < m_N; idx++)
            {
            h_pos.data[idx] = make_scalar4(pos[idx].x,pos[idx].y, pos[idx].z, __int_as_scalar(type[idx]));
            h_vel.data[idx] = make_scalar4(vel[idx].x, vel[idx].y, vel[idx].z, __int_as_scalar(mpcd::detail::NO_CELL));
            h_tag.data[idx] = tag[idx];
            h_comm_flag.data[idx] = 0; // initialize with zero by default
            }
        }
    else
    #endif // ENABLE_MPI
        {
        allocate(snapshot.size);

        ArrayHandle<Scalar4> h_pos(m_pos, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar4> h_vel(m_vel, access_location::host, access_mode::overwrite);
        ArrayHandle<unsigned int> h_tag(m_tag, access_location::host, access_mode::overwrite);

        for (unsigned int snap_idx = 0; snap_idx < snapshot.size; ++snap_idx)
            {
            h_pos.data[nglobal] = make_scalar4(snapshot.position[snap_idx].x,
                                               snapshot.position[snap_idx].y,
                                               snapshot.position[snap_idx].z,
                                               __int_as_scalar(snapshot.type[snap_idx]));
            h_vel.data[nglobal] = make_scalar4(snapshot.velocity[snap_idx].x,
                                               snapshot.velocity[snap_idx].y,
                                               snapshot.velocity[snap_idx].z,
                                               __int_as_scalar(mpcd::detail::NO_CELL));
            h_tag.data[nglobal] = nglobal;
            nglobal++;
            }

        // mass is equal for all particles
        m_mass = snapshot.mass;

        // number of local particles is the global number
        m_N = nglobal;

        // initialize type mapping
        m_type_mapping = snapshot.type_mapping;
        }

    setNGlobal(nglobal);

    // TODO: any particle data signaling to subscribers
    }

/*!
 * \param snapshot mpcd::ParticleDataSnapshot to fill
 * \param global_box Current global box
 * \post \a snapshot holds the current MPCD particle data on the root (0) rank
 */
void mpcd::ParticleData::takeSnapshot(mpcd::ParticleDataSnapshot& snapshot, const BoxDim& global_box) const
    {
    m_exec_conf->msg->notice(4) << "MPCD ParticleData: taking snapshot" << std::endl;

    ArrayHandle<Scalar4> h_pos(m_pos, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_vel(m_vel, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_tag, access_location::host, access_mode::read);

#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        // gather a global snapshot
        std::vector<Scalar3> pos(m_N);
        std::vector<Scalar3> vel(m_N);
        std::vector<unsigned int> type(m_N);
        std::vector<unsigned int> tag(m_N);
        for (unsigned int idx = 0; idx < m_N; ++idx)
            {
            pos[idx] = make_scalar3(h_pos.data[idx].x, h_pos.data[idx].y, h_pos.data[idx].z);
            vel[idx] = make_scalar3(h_vel.data[idx].x, h_vel.data[idx].y, h_vel.data[idx].z);
            type[idx] = __scalar_as_int(h_pos.data[idx].w);
            tag[idx] = h_tag.data[idx];
            }

        // Create per-processor arrays to gather the data back to root
        std::vector< std::vector<Scalar3> > pos_proc;              // Position array of every processor
        std::vector< std::vector<Scalar3> > vel_proc;              // Velocities array of every processor
        std::vector< std::vector<unsigned int> > type_proc;        // Particle types array of every processor
        std::vector< std::vector<unsigned int> > tag_proc;         // Tag array of every processor

        // resize to number of ranks in communicator
        const MPI_Comm mpi_comm = m_exec_conf->getMPICommunicator();
        const unsigned int n_ranks = m_exec_conf->getNRanks();
        const unsigned int rank = m_exec_conf->getRank();
        pos_proc.resize(n_ranks);
        vel_proc.resize(n_ranks);
        type_proc.resize(n_ranks);
        tag_proc.resize(n_ranks);

        // collect all particle data on the root processor
        const unsigned int root = 0;
        gather_v(pos, pos_proc, root,mpi_comm);
        gather_v(vel, vel_proc, root, mpi_comm);
        gather_v(type, type_proc, root, mpi_comm);
        gather_v(tag, tag_proc, root, mpi_comm);

        if (rank == root)
            {
            // allocate memory in snapshot
            snapshot.resize(getNGlobal());

            // write back into the snapshot in tag order, don't really care about cache coherency
            for (unsigned int rank_idx = 0; rank_idx < n_ranks; ++rank_idx)
                {
                const unsigned int N = pos_proc[rank_idx].size();
                for (unsigned int idx = 0; idx < N; ++idx)
                    {
                    const unsigned int snap_idx = tag_proc[rank_idx][idx];

                    // make sure the position stored in the snapshot is within the boundaries
                    Scalar3 pos_i = pos_proc[rank_idx][idx];
                    int3 img = make_int3(0,0,0);
                    global_box.wrap(pos_i,img);

                    // push particle into the snapshot
                    snapshot.position[snap_idx] = vec3<Scalar>(pos_i);
                    snapshot.velocity[snap_idx] = vec3<Scalar>(vel_proc[rank_idx][idx]);
                    snapshot.type[snap_idx] = type_proc[rank_idx][idx];
                    }
                }
            }
        }
    else
#endif
        {
        // allocate memory in snapshot
        snapshot.resize(getNGlobal());

        // iterate through particles
        for (unsigned int idx = 0; idx < m_N; ++idx)
            {
            const unsigned int snap_idx = h_tag.data[idx];

            // make sure the position stored in the snapshot is within the boundaries
            Scalar4 postype = h_pos.data[idx];
            Scalar3 pos_i = make_scalar3(postype.x, postype.y, postype.z);
            const unsigned int type_i = __scalar_as_int(postype.w);
            int3 img = make_int3(0,0,0);
            global_box.wrap(pos_i,img);

            // push particle into the snapshot
            snapshot.position[snap_idx] = vec3<Scalar>(pos_i);
            snapshot.velocity[snap_idx] = vec3<Scalar>(h_vel.data[idx]);
            snapshot.type[snap_idx] = type_i;
            }
        }

    // set the type mapping and mass, which is synced across processors
    snapshot.type_mapping = m_type_mapping;
    snapshot.mass = m_mass;
    }

/*!
 * \param snapshot Snapshot to validate
 *
 * This is a convenience wrapper to broadcast the result of snapshot validation
 * from the root rank. Note that this is a collective call that must be made
 * from all ranks.
 *
 * \sa mpcd::ParticleDataSnapshot::validate
 */
bool mpcd::ParticleData::checkSnapshot(const mpcd::ParticleDataSnapshot& snapshot)
    {
    bool valid_snapshot = true;
    if (m_exec_conf->getRank() == 0)
        {
        valid_snapshot = snapshot.validate();
        }
    #ifdef ENABLE_MPI
    if (m_decomposition)
        {
        bcast(valid_snapshot, 0, m_exec_conf->getMPICommunicator());
        }
    #endif

    return valid_snapshot;
    }

/*!
 * \param snapshot Snapshot to validate
 * \param box Global simulation box
 *
 * Checks that all particles lie within the global simulation box on initialization
 * on the root rank, and broadcasts the results to the other ranks. Note that
 * this is a collective call that must be made from all ranks.
 */
bool mpcd::ParticleData::checkInBox(const mpcd::ParticleDataSnapshot& snapshot,
                                    const BoxDim& box)
    {
    bool in_box = true;
    if (m_exec_conf->getRank() == 0)
        {
        Scalar3 lo = box.getLo();
        Scalar3 hi = box.getHi();

        const Scalar tol = Scalar(1e-5);

        for (unsigned int i = 0; i < snapshot.size; ++i)
            {
            Scalar3 f = box.makeFraction(vec_to_scalar3(snapshot.position[i]));
            if (f.x < -tol || f.x > Scalar(1.0)+tol ||
                f.y < -tol || f.y > Scalar(1.0)+tol ||
                f.z < -tol || f.z > Scalar(1.0)+tol)
                {
                m_exec_conf->msg->warning() << "pos " << i << ":" << setprecision(12) << snapshot.position[i].x << " " << snapshot.position[i].y << " " << snapshot.position[i].z << endl;
                m_exec_conf->msg->warning() << "fractional pos :" << setprecision(12) << f.x << " " << f.y << " " << f.z << endl;
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

/*!
 * \param nglobal Global number of particles
 */
void mpcd::ParticleData::setNGlobal(unsigned int nglobal)
    {
    assert(m_N <= nglobal);
    m_N_global = nglobal;

    // TODO: any signaling if needed to subscribers
    }

/*!
 * \param N_max maximum number of particles that can be held in allocation
 *
 * A clean allocation of all particle data arrays is made to accommodate \a N_max
 * total particles.
 */
void mpcd::ParticleData::allocate(unsigned int N_max)
    {
    m_N_max = N_max;

    //! Allocate the particle data
    GPUArray<Scalar4> pos(N_max, m_exec_conf);
    m_pos.swap(pos);

    GPUArray<Scalar4> vel(N_max, m_exec_conf);
    m_vel.swap(vel);

    GPUArray<unsigned int> tag(N_max, m_exec_conf);
    m_tag.swap(tag);

    #ifdef ENABLE_MPI
    if (m_decomposition)
        {
        GPUArray<unsigned int> comm_flags(N_max, m_exec_conf);
        m_comm_flags.swap(comm_flags);
        }
    #endif // ENABLE_MPI

    // Allocate the alternate data
    GPUArray<Scalar4> pos_alt(N_max, m_exec_conf);
    m_pos_alt.swap(pos_alt);

    GPUArray<Scalar4> vel_alt(N_max, m_exec_conf);
    m_vel_alt.swap(vel_alt);

    GPUArray<unsigned int> tag_alt(N_max, m_exec_conf);
    m_tag_alt.swap(tag_alt);

    #ifdef ENABLE_MPI
    if (m_decomposition)
        {
        GPUArray<unsigned int> comm_flags_alt(N_max, m_exec_conf);
        m_comm_flags_alt.swap(comm_flags_alt);
        }
    #endif // ENABLE_MPI
    }

/*!
 * \param N_max maximum number of particles that can be held in allocation
 *
 * A resize of all particle data arrays is made to accommodate \a N_max particles.
 * This has the behavior defined by GPUArray for resizing.
 */
void mpcd::ParticleData::reallocate(unsigned int N_max)
    {
    m_N_max = N_max;

    //! Reallocate the particle data
    m_pos.resize(N_max);
    m_vel.resize(N_max);
    m_tag.resize(N_max);

    #ifdef ENABLE_MPI
    if (m_decomposition)
        {
        m_comm_flags.resize(N_max);
        }
    #endif // ENABLE_MPI

    // Reallocate the alternate data
    m_pos_alt.resize(N_max);
    m_vel_alt.resize(N_max);
    m_tag_alt.resize(N_max);
    #ifdef ENABLE_MPI
    if (m_decomposition)
        {
        m_comm_flags_alt.resize(N_max);
        }
    #endif // ENABLE_MPI
    }

/*!
 * \param N New number of particles held by the data
 *
 * A new size for the particle data arrays is chosen using amortized growth
 * (set by growth_factor), and the particle data is resized. If N is less than
 * m_N_max, then nothing needs to be reallocated, and the current number of
 * owned particles is simply changed.
 */
void mpcd::ParticleData::resize(unsigned int N)
    {
    // compute the new size of the array using amortized growth
    unsigned int N_max = m_N_max;
    if (N > N_max)
        {
        while (N > N_max)
            {
            N_max = ((unsigned int) (((float) N_max) * resize_factor)) + 1;
            }
        reallocate(N_max);
        }
    m_N = N;
    }

/*!
 * \param mass New particle mass
 *
 * This is a collective call requiring the participation of all ranks. The mass
 * will be broadcast from rank 0 (root) to ensure it is synced across all ranks
 * in MPI. The mass cannot be set independently between ranks because this is not meaningful.
 */
void mpcd::ParticleData::setMass(Scalar mass)
    {
    assert(mass > Scalar(0.));
    m_mass = mass;

    #ifdef ENABLE_MPI
    // in mpi, the mass must be synced between all ranks
    if (m_decomposition)
        {
        bcast(m_mass, 0, m_exec_conf->getMPICommunicator());
        }
    #endif // ENABLE_MPI
    }

/*!
 * \param name Type name to get the index of
 * \return Type index of the corresponding type name
 * \throw runtime_error if the type name is not found
 */
unsigned int mpcd::ParticleData::getTypeByName(const std::string &name) const
    {
    // search for the name
    for (unsigned int i = 0; i < m_type_mapping.size(); ++i)
        {
        if (m_type_mapping[i] == name)
            return i;
        }

    m_exec_conf->msg->error() << "MPCD particle type " << name << " not found!" << endl;
    throw runtime_error("Error mapping MPCD type name");
    }

/*!
 * \param type Type index to get the name of
 * \returns Type name of the requested type
 * \throw runtime_error if the requested type index is out of bounds
 *
 * Type indices must range from 0 to getNTypes.
 */
std::string mpcd::ParticleData::getNameByType(unsigned int type) const
    {
    // check for an invalid request
    if (type >= m_type_mapping.size())
        {
        m_exec_conf->msg->error() << "Requesting name for non-existant MPCD particle type " << type << endl;
        throw runtime_error("Error mapping MPCD type name");
        }

    // return the name
    return m_type_mapping[type];
    }

/*!
 * \param idx Local particle index
 * \returns Position of particle with local index \a idx
 * \throw runtime_error if the requested local particle index is out of bounds
 */
Scalar3 mpcd::ParticleData::getPosition(unsigned int idx) const
    {
    if (idx >= m_N)
        {
        m_exec_conf->msg->error() << "Requested MPCD particle local index " << idx << " is out of range" << endl;
        throw std::runtime_error("Error accessing MPCD particle data.");
        }
    ArrayHandle<Scalar4> h_pos(m_pos, access_location::host, access_mode::read);
    const Scalar4 postype = h_pos.data[idx];
    return make_scalar3(postype.x, postype.y, postype.z);
    }

/*!
 * \param idx Local particle index
 * \returns Type of particle with local index \a idx
 * \throw runtime_error if the requested local particle index is out of bounds
 */
unsigned int mpcd::ParticleData::getType(unsigned int idx) const
    {
    if (idx >= m_N)
        {
        m_exec_conf->msg->error() << "Requested MPCD particle local index " << idx << " is out of range" << endl;
        throw std::runtime_error("Error accessing MPCD particle data.");
        }
    ArrayHandle<Scalar4> h_pos(m_pos, access_location::host, access_mode::read);
    const Scalar4 postype = h_pos.data[idx];
    return __scalar_as_int(postype.w);
    }

/*!
 * \param idx Local particle index
 * \returns Velocity of particle with local index \a idx
 * \throw runtime_error if the requested local particle index is out of bounds
 */
Scalar3 mpcd::ParticleData::getVelocity(unsigned int idx) const
    {
    if (idx >= m_N)
        {
        m_exec_conf->msg->error() << "Requested MPCD particle local index " << idx << " is out of range" << endl;
        throw std::runtime_error("Error accessing MPCD particle data.");
        }
    ArrayHandle<Scalar4> h_vel(m_vel, access_location::host, access_mode::read);
    const Scalar4 velcell = h_vel.data[idx];
    return make_scalar3(velcell.x, velcell.y, velcell.z);
    }

/*!
 * \param idx Local particle index
 * \returns Tag of particle with local index \a idx
 * \throw runtime_error if the requested local particle index is out of bounds
 */
unsigned int mpcd::ParticleData::getTag(unsigned int idx) const
    {
    if (idx >= m_N)
        {
        m_exec_conf->msg->error() << "Requested MPCD particle local index " << idx << " is out of range" << endl;
        throw std::runtime_error("Error accessing MPCD particle data.");
        }
    ArrayHandle<unsigned int> h_tag(m_tag, access_location::host, access_mode::read);
    return h_tag.data[idx];
    }

#ifdef ENABLE_MPI
/*!
 * \param out Buffer into which particle data is packed
 * \param mask Mask for \a m_comm_flags to determine if communication is necessary
 *
 * Packs all particles where the communication flags are bitwise AND against \a mask
 * into a buffer and removes them from the particle data arrays. The output buffer
 * is automatically resized to accommodate the data.
 *
 * \post The particle data arrays remain compact.
 */
void mpcd::ParticleData::removeParticles(GPUVector<mpcd::detail::pdata_element>& out,
                                         unsigned int mask)
    {
    // count the number of particles to remove
    const unsigned int old_nparticles = m_N;
    unsigned int num_remove_ptls = 0;
        {
        // access particle data tags and rtags
        ArrayHandle<unsigned int> h_comm_flags(m_comm_flags, access_location::host, access_mode::read);

        // count removed particles
        for (unsigned int i = 0; i < old_nparticles; ++i)
            {
            if (h_comm_flags.data[i] & mask)
                {
                ++num_remove_ptls;
                }
            }
        }
    const unsigned int new_nparticles = old_nparticles - num_remove_ptls;

    // resize output buffers
    out.resize(num_remove_ptls);

    // pack the output buffer
        {
        // access particle data arrays
        ArrayHandle<Scalar4> h_pos(m_pos, access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_vel(m_vel, access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_tag(m_tag, access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_comm_flags(m_comm_flags, access_location::host, access_mode::read);

        ArrayHandle<Scalar4> h_pos_alt(m_pos_alt, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar4> h_vel_alt(m_vel_alt, access_location::host, access_mode::overwrite);
        ArrayHandle<unsigned int> h_tag_alt(m_tag_alt, access_location::host, access_mode::overwrite);
        ArrayHandle<unsigned int> h_comm_flags_alt(m_comm_flags_alt, access_location::host, access_mode::overwrite);

        ArrayHandle<mpcd::detail::pdata_element> h_out(out, access_location::host, access_mode::overwrite);
        unsigned int n(0), m(0);
        for (unsigned int i = 0; i < old_nparticles; ++i)
            {
            if (h_comm_flags.data[i] & mask)
                {
                // write to packed array
                mpcd::detail::pdata_element p;
                p.pos = h_pos.data[i];
                p.vel = h_vel.data[i];
                p.tag = h_tag.data[i];
                p.comm_flag = h_comm_flags.data[i];
                h_out.data[m++] = p;
                }
            else
                {
                // copy over to alternate pdata arrays
                h_pos_alt.data[n] = h_pos.data[i];
                h_vel_alt.data[n] = h_vel.data[i];
                h_tag_alt.data[n] = h_tag.data[i];
                h_comm_flags_alt.data[n] = h_comm_flags.data[i];
                ++n;
                }
            }
        }

    // swap particle data arrays to update local particle data
    swapPositions();
    swapVelocities();
    swapTags();
    swapCommFlags();

    // resize self down (just changes value of m_N since removing)
    resize(new_nparticles);

    // TODO: signal particle data has changed
    invalidateCellCache();
    }

/*!
 * \param in List of particle data elements to fill the particle data with
 * \param mask Bitmask for direction send occurred
 */
void mpcd::ParticleData::addParticles(const GPUVector<mpcd::detail::pdata_element>& in,
                                      unsigned int mask)
    {
    if (m_prof) m_prof->push("unpack");

    unsigned int num_add_ptls = in.size();

    unsigned int old_nparticles = m_N;
    unsigned int new_nparticles = old_nparticles + num_add_ptls;

    // resize particle data using amortized O(1) array resizing
    resize(new_nparticles);

        {
        // access particle data arrays
        ArrayHandle<Scalar4> h_pos(getPositions(), access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> h_vel(getVelocities(), access_location::host, access_mode::readwrite);
        ArrayHandle<unsigned int> h_tag(getTags(), access_location::host, access_mode::readwrite);
        ArrayHandle<unsigned int> h_comm_flags(m_comm_flags, access_location::host, access_mode::readwrite);

        // add new particles at the end
        ArrayHandle<mpcd::detail::pdata_element> h_in(in, access_location::host, access_mode::read);
        unsigned int n = old_nparticles;
        for (unsigned int i = 0 ; i < num_add_ptls; ++i)
            {
            const mpcd::detail::pdata_element& p = h_in.data[i];
            h_pos.data[n] = p.pos;
            h_vel.data[n] = p.vel;
            h_tag.data[n] = p.tag;
            h_comm_flags.data[n] = p.comm_flag & ~mask; // unset the bitmask after communication
            n++;
            }
        }

    if (m_prof) m_prof->pop();

    // TODO: signal particle data has changed
    invalidateCellCache();
    }

#ifdef ENABLE_CUDA
/*!
 * \param out Buffer into which particle data is packed
 * \param mask Mask for \a m_comm_flags to determine if communication is necessary
 *
 * Packs all particles where the communication flags are bitwise AND against \a mask
 * into a buffer and removes them from the particle data arrays using the GPU.
 * The output buffer is automatically resized to accommodate the data.
 *
 * \post The particle data arrays remain compact.
 */
void mpcd::ParticleData::removeParticlesGPU(GPUVector<mpcd::detail::pdata_element>& out,
                                            unsigned int mask)
    {
    if (m_prof) m_prof->push(m_exec_conf, "pack");

    // this is the maximum number of elements we can possibly write to out
    // (use getNumElements rather than size, since this gives the capacity of the vector)
    unsigned int max_n_out = out.getNumElements();

    // allocate array if necessary
    if (!max_n_out)
        {
        max_n_out = 1;
        out.resize(max_n_out);
        }

    // number of particles that are to be written out
    unsigned int n_out = 0;

    bool done = false;

    // copy without writing past the end of the output array, resizing it as needed
    while (!done)
        {
        // access particle data arrays to read from
        ArrayHandle<Scalar4> d_pos(m_pos, access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_vel(m_vel, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_tag(m_tag, access_location::device, access_mode::read);
        ArrayHandle<unsigned int> d_comm_flags(m_comm_flags, access_location::device, access_mode::read);

        // access alternate particle data arrays to write to
        ArrayHandle<Scalar4> d_pos_alt(m_pos_alt, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar4> d_vel_alt(m_vel_alt, access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> d_tag_alt(m_tag_alt, access_location::device, access_mode::overwrite);
        ArrayHandle<unsigned int> d_comm_flags_alt(m_comm_flags_alt, access_location::device, access_mode::overwrite);

            {
            // Access output array
            ArrayHandle<mpcd::detail::pdata_element> d_out(out, access_location::device, access_mode::overwrite);

            // get temporary buffer
            // TODO: FIX THIS, CACHED ALLOCATOR LEAKS MEMORY!
            ScopedAllocation<unsigned int> d_tmp(m_exec_conf->getCachedAllocator(), getN());

            n_out = mpcd::gpu::remove_particles(d_out.data,
                                                mask,
                                                m_N,
                                                d_pos.data,
                                                d_vel.data,
                                                d_tag.data,
                                                d_comm_flags.data,
                                                d_pos_alt.data,
                                                d_vel_alt.data,
                                                d_tag_alt.data,
                                                d_comm_flags_alt.data,
                                                max_n_out,
                                                d_tmp.data,
                                                m_mgpu_context);
           }
        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        // resize output vector
        out.resize(n_out);

        // was the array large enough?
        if (n_out <= max_n_out) done = true;

        max_n_out = out.getNumElements();
        }

    // update particle number
    unsigned int new_nparticles = m_N - n_out;
    resize(new_nparticles);

    // swap particle data arrays
    swapPositions();
    swapVelocities();
    swapTags();
    swapCommFlags();

    // TODO: notify particle data has changed
    invalidateCellCache();

    if (m_prof) m_prof->pop(m_exec_conf);
    }

/*!
 * \param in List of particle data elements to fill the particle data with
 */
void mpcd::ParticleData::addParticlesGPU(const GPUVector<mpcd::detail::pdata_element>& in,
                                         unsigned int mask)
    {
    if (m_prof) m_prof->push(m_exec_conf, "unpack");

    unsigned int old_nparticles = m_N;
    unsigned int num_add_ptls = in.size();
    unsigned int new_nparticles = old_nparticles + num_add_ptls;

    // amortized resizing of particle data
    resize(new_nparticles);

        {
        // access particle data arrays
        ArrayHandle<Scalar4> d_pos(m_pos, access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar4> d_vel(m_vel, access_location::device, access_mode::readwrite);
        ArrayHandle<unsigned int> d_tag(m_tag, access_location::device, access_mode::readwrite);
        ArrayHandle<unsigned int> d_comm_flags(m_comm_flags, access_location::device, access_mode::readwrite);

        // Access input array
        ArrayHandle<mpcd::detail::pdata_element> d_in(in, access_location::device, access_mode::read);

        // add new particles on GPU
        mpcd::gpu::add_particles(old_nparticles,
                                 num_add_ptls,
                                 d_pos.data,
                                 d_vel.data,
                                 d_tag.data,
                                 d_comm_flags.data,
                                 d_in.data,
                                 mask);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    // TODO: notify particle data has changed
    invalidateCellCache();

    if (m_prof) m_prof->pop(m_exec_conf);
    }
#endif // ENABLE_CUDA
#endif // ENABLE_MPI

/*!
 * \param m Python module to export to
 */
void mpcd::detail::export_ParticleData(pybind11::module& m)
    {
    pybind11::class_< mpcd::ParticleData, std::shared_ptr<mpcd::ParticleData> >(m, "MPCDParticleData")
    .def(pybind11::init< unsigned int, unsigned int, const BoxDim&, std::shared_ptr<ExecutionConfiguration> >())
    .def(pybind11::init< unsigned int, unsigned int, const BoxDim&, std::shared_ptr<ExecutionConfiguration>, std::shared_ptr<DomainDecomposition> >())
    .def(pybind11::init< mpcd::ParticleDataSnapshot&, const BoxDim&, std::shared_ptr<ExecutionConfiguration> >())
    .def(pybind11::init< mpcd::ParticleDataSnapshot&, const BoxDim&, std::shared_ptr<ExecutionConfiguration>, std::shared_ptr<DomainDecomposition> >())
    .def_property_readonly("N", &mpcd::ParticleData::getN)
    .def_property_readonly("N_global", &mpcd::ParticleData::getNGlobal)
    .def("getPosition", &mpcd::ParticleData::getPosition)
    .def("getType", &mpcd::ParticleData::getType)
    .def("getVelocity", &mpcd::ParticleData::getVelocity)
    .def("getTag", &mpcd::ParticleData::getTag)
    .def_property_readonly("n_types", &mpcd::ParticleData::getNTypes)
    .def_property_readonly("types", &mpcd::ParticleData::getTypeNames)
    .def("getNameByType", &mpcd::ParticleData::getNameByType)
    .def("getTypeByName", &mpcd::ParticleData::getTypeByName)
    .def_property("mass", &mpcd::ParticleData::getMass, &mpcd::ParticleData::setMass)
    ;
    }
