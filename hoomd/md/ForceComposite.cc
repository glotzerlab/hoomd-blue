// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ForceComposite.h"
#include "hoomd/VectorMath.h"

#include <map>
#include <sstream>
#include <string.h>

#include <pybind11/stl.h>

/*! \file ForceComposite.cc
    \brief Contains code for the ForceComposite class
*/

namespace hoomd
    {
namespace md
    {
/*! \param sysdef SystemDefinition containing the ParticleData to compute forces on
 */
ForceComposite::ForceComposite(std::shared_ptr<SystemDefinition> sysdef)
    : MolecularForceCompute(sysdef), m_bodies_changed(false), m_particles_added_removed(false)
    {
    m_pdata->getGlobalParticleNumberChangeSignal()
        .connect<ForceComposite, &ForceComposite::slotPtlsAddedRemoved>(this);

    m_exec_conf->msg->notice(7) << "ForceComposite initialize memory" << std::endl;

    GlobalArray<unsigned int> body_types(m_pdata->getNTypes(), 1, m_exec_conf);
    m_body_types.swap(body_types);
    TAG_ALLOCATION(m_body_types);

    GlobalArray<Scalar3> body_pos(m_pdata->getNTypes(), 1, m_exec_conf);
    m_body_pos.swap(body_pos);
    TAG_ALLOCATION(m_body_pos);

    GlobalArray<Scalar4> body_orientation(m_pdata->getNTypes(), 1, m_exec_conf);
    m_body_orientation.swap(body_orientation);
    TAG_ALLOCATION(m_body_orientation);

    GlobalArray<unsigned int> body_len(m_pdata->getNTypes(), m_exec_conf);
    m_body_len.swap(body_len);
    TAG_ALLOCATION(m_body_len);

    // reset elements to zero
    ArrayHandle<unsigned int> h_body_len(m_body_len, access_location::host, access_mode::readwrite);
    for (unsigned int i = 0; i < this->m_pdata->getNTypes(); ++i)
        {
        h_body_len.data[i] = 0;
        }

    m_d_max.resize(m_pdata->getNTypes(), Scalar(0.0));
    m_d_max_changed.resize(m_pdata->getNTypes(), false);

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        auto comm_weak = m_sysdef->getCommunicator();
        assert(comm_weak.lock());
        m_comm = comm_weak.lock();

        // register this class with the communicator
        m_comm->getBodyGhostLayerWidthRequestSignal()
            .connect<ForceComposite, &ForceComposite::requestBodyGhostLayerWidth>(this);
        }
#endif
    }

//! Destructor
ForceComposite::~ForceComposite()
    {
    // disconnect from signal in ParticleData;
    m_pdata->getGlobalParticleNumberChangeSignal()
        .disconnect<ForceComposite, &ForceComposite::slotPtlsAddedRemoved>(this);
#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        m_comm->getBodyGhostLayerWidthRequestSignal()
            .disconnect<ForceComposite, &ForceComposite::requestBodyGhostLayerWidth>(this);
        }
#endif
    }

void ForceComposite::setParam(unsigned int body_typeid,
                              std::vector<unsigned int>& type,
                              std::vector<Scalar3>& pos,
                              std::vector<Scalar4>& orientation)
    {
    assert(m_body_types.getPitch() >= m_pdata->getNTypes());
    assert(m_body_pos.getPitch() >= m_pdata->getNTypes());
    assert(m_body_orientation.getPitch() >= m_pdata->getNTypes());

    if (body_typeid >= m_pdata->getNTypes())
        {
        throw std::runtime_error("Error initializing ForceComposite: Invalid rigid body type.");
        }

    if (type.size() != pos.size() || orientation.size() != pos.size())
        {
        std::ostringstream error_msg;
        error_msg << "Error initializing ForceComposite: Constituent particle lists"
                  << " (position, orientation, type) are of unequal length.";
        throw std::runtime_error(error_msg.str());
        }
    bool body_updated = false;

    bool body_len_changed = false;

        // detect if bodies have changed

        {
        ArrayHandle<unsigned int> h_body_type(m_body_types,
                                              access_location::host,
                                              access_mode::read);
        ArrayHandle<Scalar3> h_body_pos(m_body_pos, access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_body_orientation(m_body_orientation,
                                                access_location::host,
                                                access_mode::read);
        ArrayHandle<unsigned int> h_body_len(m_body_len,
                                             access_location::host,
                                             access_mode::readwrite);

        assert(body_typeid < m_body_len.getNumElements());
        if (type.size() != h_body_len.data[body_typeid])
            {
            body_updated = true;

            h_body_len.data[body_typeid] = (unsigned int)type.size();
            body_len_changed = true;
            }
        else
            {
            for (unsigned int i = 0; i < type.size(); ++i)
                {
                auto body_index = m_body_idx(body_typeid, i);
                if (type[i] != h_body_type.data[body_index] || pos[i] != h_body_pos.data[body_index]
                    || orientation[i] != h_body_orientation.data[body_index])
                    {
                    body_updated = true;
                    }
                }
            }
        }

    if (body_len_changed)
        {
        if (type.size() > m_body_types.getHeight())
            {
            // resize per-type arrays
            m_body_types.resize(m_pdata->getNTypes(), type.size());
            m_body_pos.resize(m_pdata->getNTypes(), type.size());
            m_body_orientation.resize(m_pdata->getNTypes(), type.size());

            m_body_idx = Index2D((unsigned int)m_body_types.getPitch(),
                                 (unsigned int)m_body_types.getHeight());
            }
        }

    if (body_updated)
        {
            {
            ArrayHandle<unsigned int> h_body_type(m_body_types,
                                                  access_location::host,
                                                  access_mode::readwrite);
            ArrayHandle<Scalar3> h_body_pos(m_body_pos,
                                            access_location::host,
                                            access_mode::readwrite);
            ArrayHandle<Scalar4> h_body_orientation(m_body_orientation,
                                                    access_location::host,
                                                    access_mode::readwrite);

            // store body data in GlobalArray
            for (unsigned int i = 0; i < type.size(); ++i)
                {
                h_body_type.data[m_body_idx(body_typeid, i)] = type[i];
                h_body_pos.data[m_body_idx(body_typeid, i)] = pos[i];
                h_body_orientation.data[m_body_idx(body_typeid, i)] = orientation[i];
                }
            }
        m_bodies_changed = true;
        assert(m_d_max_changed.size() > body_typeid);

        // make sure central particle will be communicated
        m_d_max_changed[body_typeid] = true;

        // also update diameter on constituent particles
        for (unsigned int i = 0; i < type.size(); ++i)
            {
            m_d_max_changed[type[i]] = true;
            }
        }
    }

/** Compute the needed body ghost layer width.

    For central particles, the body ghost layer width is the maximum of [d_i + r_ghost_i] where d_i
    is distance of particle i from the center of the body and r_ghost_i is the ghost width for
    particle i determined by cutoff and other interactions.

    The body ghost layer width for constituent particles *should be* the maximum diameter d_i among
    all rigid body types that have this particle as a consituent. However, this must be larger due
    to limitations in the way that individual rigid body particles are indexed relative to the
    molecules in MolecularForceCompute. In the worst case, for a ghost particle within the
    interaction ghost width r_ghost_i of a boundary, *ALL* other particles in that body must be
    included. The ghost layer width needed to satisfy this condition is the maximum of [2*d_i +
    r_ghost_i], allowing for enough distance to communicate another particle placed at -r_i.
*/
Scalar ForceComposite::requestBodyGhostLayerWidth(unsigned int type, Scalar* h_r_ghost)
    {
    assert(m_body_len.getNumElements() > type);
    ArrayHandle<unsigned int> h_body_len(m_body_len, access_location::host, access_mode::read);

    if (m_d_max_changed[type])
        {
        m_d_max[type] = Scalar(0.0);
        ArrayHandle<Scalar3> h_body_pos(m_body_pos, access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_body_type(m_body_types,
                                              access_location::host,
                                              access_mode::read);

        if (h_body_len.data[type] != 0)
            {
            // central particles
            for (unsigned int body_particle = 0; body_particle < h_body_len.data[type];
                 body_particle++)
                {
                unsigned int constituent_typeid = h_body_type.data[m_body_idx(type, body_particle)];
                vec3<Scalar> constituent_position(h_body_pos.data[m_body_idx(type, body_particle)]);
                Scalar d = slow::sqrt(dot(constituent_position, constituent_position));
                m_d_max[type] = std::max(m_d_max[type], d + h_r_ghost[constituent_typeid]);
                }
            }
        else
            {
            // constituent particles
            for (unsigned int body_type = 0; body_type < m_pdata->getNTypes(); body_type++)
                {
                if (h_body_len.data[body_type] == 0)
                    {
                    continue;
                    }

                for (unsigned int body_particle = 0; body_particle < h_body_len.data[body_type];
                     body_particle++)
                    {
                    unsigned int constituent_typeid
                        = h_body_type.data[m_body_idx(body_type, body_particle)];

                    if (constituent_typeid == type)
                        {
                        vec3<Scalar> constituent_position(
                            h_body_pos.data[m_body_idx(body_type, body_particle)]);
                        Scalar d = slow::sqrt(dot(constituent_position, constituent_position));
                        m_d_max[type] = std::max(m_d_max[type],
                                                 d * Scalar(2.0) + h_r_ghost[constituent_typeid]);
                        }
                    }
                }
            }
        }

    m_d_max_changed[type] = false;
    m_exec_conf->msg->notice(7) << "ForceComposite: requesting ghost layer for type "
                                << m_pdata->getNameByType(type) << ": " << m_d_max[type]
                                << std::endl;

    return m_d_max[type];
    }

void ForceComposite::validateRigidBodies()
    {
    if (!(m_bodies_changed || m_particles_added_removed))
        {
        return;
        }

    // check validity of rigid body types: no nested rigid bodies
    unsigned int ntypes = m_pdata->getNTypes();
    assert(m_body_types.getPitch() >= ntypes);
        {
        ArrayHandle<unsigned int> h_body_type(m_body_types,
                                              access_location::host,
                                              access_mode::read);
        ArrayHandle<unsigned int> h_body_len(m_body_len, access_location::host, access_mode::read);
        for (unsigned int itype = 0; itype < ntypes; ++itype)
            {
            for (unsigned int j = 0; j < h_body_len.data[itype]; ++j)
                {
                assert(h_body_type.data[m_body_idx(itype, j)] <= ntypes);
                if (h_body_len.data[h_body_type.data[m_body_idx(itype, j)]] != 0)
                    {
                    throw std::runtime_error(
                        "Error initializing ForceComposite: A rigid body type "
                        "may not contain constituent particles that are also rigid bodies.");
                    }
                }
            }
        }

    SnapshotParticleData<Scalar> snap;

    // take a snapshot on rank 0
    m_pdata->takeSnapshot(snap);

    std::vector<unsigned int> molecule_tag;

    // number of bodies in system
    unsigned int nbodies = 0;

    // number of free particles in the system
    m_n_free_particles_global = 0;

    // Validate the body tags in the system and assign molecules into molecule tag
    if (m_exec_conf->getRank() == 0)
        {
        // access body data
        ArrayHandle<unsigned int> h_body_len(m_body_len, access_location::host, access_mode::read);
        typedef std::map<unsigned int, unsigned int> map_t;
        // This will count the length of all molecules (realized rigid bodies) in the system.
        map_t body_particle_count;

        molecule_tag.resize(snap.size, NO_MOLECULE);

        // count number of constituent particles to add
        for (unsigned i = 0; i < snap.size; ++i)
            {
            assert(snap.type[i] < ntypes);

            // If a particle is of a type with a non-zero body definition it should be a central
            // particle and the body value should equal its particle tag.
            if (h_body_len.data[snap.type[i]] != 0)
                {
                if (snap.body[i] != i)
                    {
                    throw std::runtime_error(
                        "Error validating rigid bodies: Particles of types defining rigid bodies "
                        "must have a body tag identical to their particle tag to be considered a "
                        "central particle.");
                    }

                // Create a new molecule count for central particle i.
                body_particle_count.insert(std::make_pair(i, 0));
                molecule_tag[i] = nbodies++;
                }
            // validate constituent particles. MIN_FLOPPY defines the maximum tag for a particle
            // that is in a rigid body. Tags higher than this can be in a floppy body.
            else if (snap.body[i] < MIN_FLOPPY)
                {
                // check if particle body tag correctly points to the central particle and less than
                // the number of particles in the system. This first check is to ensure that no
                // unallocated memory is attempted to be accessed in the second check.
                if (snap.body[i] >= snap.size || snap.body[snap.body[i]] != snap.body[i])
                    {
                    throw std::runtime_error(
                        "Error validating rigid bodies: Constituent particle body tags must "
                        "be the tag of their central particles.");
                    }

                unsigned int central_particle_index = snap.body[i];
                map_t::iterator it = body_particle_count.find(central_particle_index);
                // If find returns the end of the map, then the central particle for this
                // particular composite particle has not been seen yet. Since we are iterating
                // over the snapshot, this means that the tag for the central particle is higher
                // than the composite particles which is not allowed.
                if (it == body_particle_count.end())
                    {
                    throw std::runtime_error(
                        "Error validating rigid bodies: Central particle must have a lower "
                        "tag than all constituent particles.");
                    }

                unsigned int current_molecule_size = it->second;
                unsigned int body_type = snap.type[central_particle_index];
                if (current_molecule_size == h_body_len.data[body_type])
                    {
                    throw std::runtime_error(
                        "Error validating rigid bodies: Too many constituent particles for "
                        "rigid body.");
                    }
                // increase molecule size by one as particle is validated
                it->second++;
                // Mark consistent particle in molecule as belonging to its central particle.
                molecule_tag[i] = molecule_tag[snap.body[i]];
                }
            else
                {
                m_n_free_particles_global++;
                }
            }
        for (auto it = body_particle_count.begin(); it != body_particle_count.end(); ++it)
            {
            const auto central_particle_tag = it->first;
            const auto molecule_size = it->second;
            unsigned int central_particle_type = snap.type[central_particle_tag];
            if (molecule_size != h_body_len.data[central_particle_type])
                {
                std::ostringstream error_msg;
                error_msg << "Error validating rigid bodies: Incomplete rigid body with only "
                          << molecule_size << " constituent particles " << "instead of "
                          << h_body_len.data[central_particle_type] << " for body "
                          << central_particle_tag;
                throw std::runtime_error(error_msg.str());
                }
            }
        }

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        bcast(molecule_tag, 0, m_exec_conf->getMPICommunicator());
        bcast(nbodies, 0, m_exec_conf->getMPICommunicator());
        bcast(m_n_free_particles_global, 0, m_exec_conf->getMPICommunicator());
        }
#endif

    // resize Molecular tag member array
    m_molecule_tag.resize(molecule_tag.size());
        {
        ArrayHandle<unsigned int> h_molecule_tag(m_molecule_tag,
                                                 access_location::host,
                                                 access_mode::overwrite);
        std::copy(molecule_tag.begin(), molecule_tag.end(), h_molecule_tag.data);
        }

    // store number of molecules in all ranks
    m_n_molecules_global = nbodies;

    // reset flags
    m_bodies_changed = false;
    m_particles_added_removed = false;
    }

void ForceComposite::pyCreateRigidBodies(pybind11::dict charges)
    {
    if (pybind11::len(charges) == 0)
        {
        createRigidBodies(std::unordered_map<unsigned int, std::vector<Scalar>>());
        return;
        }
    ArrayHandle<unsigned int> h_body_len(m_body_len, access_location::host, access_mode::read);
    std::unordered_map<unsigned int, std::vector<Scalar>> charges_map;
    for (const auto& item : charges)
        {
        const auto type = m_pdata->getTypeByName(item.first.cast<std::string>());
        if (h_body_len.data[type] == 0)
            {
            throw std::runtime_error("Charge provided for non-central particle type.");
            }
        const auto charges_list = item.second.cast<pybind11::list>();
        if (pybind11::len(charges_list) != h_body_len.data[type])
            {
            throw std::runtime_error("Charges provided not consistent with rigid body size.");
            }
        std::vector<Scalar> charges_vector;
        for (auto& charge : charges_list)
            {
            charges_vector.emplace_back(charge.cast<Scalar>());
            }
        charges_map.insert({type, charges_vector});
        }
    createRigidBodies(charges_map);
    }

void ForceComposite::createRigidBodies(
    const std::unordered_map<unsigned int, std::vector<Scalar>> charges)
    {
    SnapshotParticleData<Scalar> snap;
    const BoxDim& global_box = m_pdata->getGlobalBox();

    // take a snapshot on rank 0
    m_pdata->takeSnapshot(snap);
    bool remove_existing_constituents = false;
    unsigned int n_constituent_particles_to_add = 0;
    unsigned int n_free_particles = 0;

    if (m_exec_conf->getRank() == 0)
        {
        ArrayHandle<unsigned int> h_body_len(m_body_len, access_location::host, access_mode::read);
        for (unsigned int particle_tag = 0; particle_tag < snap.size; ++particle_tag)
            {
            // Determine whether each particle is rigid or free based on the particle type and the
            // rigid body definition.
            if (h_body_len.data[snap.type[particle_tag]] == 0)
                {
                n_free_particles++;
                }
            else
                {
                // Determine whether we need to remove existing constituent particles. These will
                // be recreated below.
                if (snap.body[particle_tag] != particle_tag)
                    {
                    remove_existing_constituents = true;
                    }

                // Increase the number of particles we need to add by the number of constituent
                // particles this rigid body center has based on its type.
                n_constituent_particles_to_add += h_body_len.data[snap.type[particle_tag]];
                }
            }
        }

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        bcast(remove_existing_constituents, 0, m_exec_conf->getMPICommunicator());
        bcast(n_free_particles, 0, m_exec_conf->getMPICommunicator());
        }
#endif

    if (remove_existing_constituents)
        {
        m_exec_conf->msg->notice(7)
            << "ForceComposite reinitialize particle data without rigid bodies" << std::endl;
        m_pdata->initializeFromSnapshot(snap, true);
        m_pdata->takeSnapshot(snap);
        }

    std::vector<unsigned int> molecule_tag;
    unsigned int n_central_particles = snap.size - n_free_particles;

    if (m_exec_conf->getRank() == 0)
        {
        unsigned int initial_snapshot_size = snap.size;
        snap.insert(snap.size, n_constituent_particles_to_add);

        ArrayHandle<unsigned int> h_body_len(m_body_len, access_location::host, access_mode::read);
        ArrayHandle<Scalar3> h_body_pos(m_body_pos, access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_body_orientation(m_body_orientation,
                                                access_location::host,
                                                access_mode::read);
        ArrayHandle<unsigned int> h_body_type(m_body_types,
                                              access_location::host,
                                              access_mode::read);
        molecule_tag.resize(snap.size, NO_MOLECULE);

        unsigned int constituent_particle_tag = initial_snapshot_size;
        for (unsigned int particle_tag = 0; particle_tag < initial_snapshot_size; ++particle_tag)
            {
            assert(snap.type[particle_tag] < m_pdata->getNTypes());

            // If the length of the body definition is zero it must be a free particle because all
            // constituent particles have been removed.
            if (h_body_len.data[snap.type[particle_tag]] == 0)
                {
                snap.body[particle_tag] = NO_BODY;
                continue;
                }
            snap.body[particle_tag] = particle_tag;
            molecule_tag[particle_tag] = particle_tag;

            unsigned int body_type = snap.type[particle_tag];
            unsigned int n_body_particles = h_body_len.data[body_type];

            for (unsigned int current_body_index = 0; current_body_index < n_body_particles;
                 ++current_body_index)
                {
                size_t body_idx = m_body_idx(body_type, current_body_index);

                // Update constituent particle snapshot properties from default.
                snap.type[constituent_particle_tag] = h_body_type.data[body_idx];
                snap.body[constituent_particle_tag] = particle_tag;
                if (!charges.empty())
                    {
                    snap.charge[constituent_particle_tag]
                        = charges.at(body_type)[current_body_index];
                    }

                // Set position and orientation of constituents
                vec3<Scalar> body_position(snap.pos[particle_tag]);
                quat<Scalar> body_orientation(snap.orientation[particle_tag]);
                vec3<Scalar> local_position(h_body_pos.data[body_idx]);
                quat<Scalar> local_orientation(h_body_orientation.data[body_idx]);

                vec3<Scalar> constituent_position
                    = body_position + rotate(body_orientation, local_position);
                quat<Scalar> constituent_orientation = body_orientation * local_orientation;

                snap.pos[constituent_particle_tag] = constituent_position;
                snap.image[constituent_particle_tag] = snap.image[particle_tag];
                snap.orientation[constituent_particle_tag] = constituent_orientation;

                // wrap back into the box
                global_box.wrap(snap.pos[constituent_particle_tag],
                                snap.image[constituent_particle_tag]);

                // Since the central particle tags here will be [0, n_central_particles), we know
                // that the molecule number will be the same as the central particle tag.
                molecule_tag[constituent_particle_tag] = particle_tag;

                ++constituent_particle_tag;
                }
            }
        }

    // Keep rigid bodies this time when initializing.
    m_pdata->initializeFromSnapshot(snap, false);

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        bcast(molecule_tag, 0, m_exec_conf->getMPICommunicator());
        }
#endif

    m_molecule_tag.resize(molecule_tag.size());
        {
        // store global molecule information in GlobalArray
        ArrayHandle<unsigned int> h_molecule_tag(m_molecule_tag,
                                                 access_location::host,
                                                 access_mode::overwrite);
        std::copy(molecule_tag.begin(), molecule_tag.end(), h_molecule_tag.data);
        }
    m_n_molecules_global = n_central_particles;
    m_n_free_particles_global = n_free_particles;

    m_bodies_changed = false;
    m_particles_added_removed = false;
    }

#ifdef ENABLE_MPI
/*! \param timestep Current time step
 */
CommFlags ForceComposite::getRequestedCommFlags(uint64_t timestep)
    {
    CommFlags flags = CommFlags(0);

    // request orientations
    flags[comm_flag::orientation] = 1;

    // only communicate net virial if needed
    PDataFlags pdata_flags = this->m_pdata->getFlags();
    if (pdata_flags[pdata_flag::pressure_tensor])
        {
        flags[comm_flag::net_virial] = 1;
        }

    // request body ids
    flags[comm_flag::body] = 1;

    // we need central particle images
    flags[comm_flag::image] = 1;

    flags |= MolecularForceCompute::getRequestedCommFlags(timestep);

    return flags;
    }
#endif

//! Compute the forces and torques on the central particle
void ForceComposite::computeForces(uint64_t timestep)
    {
    // If no rigid bodies exist return early. This also prevents accessing arrays assuming that this
    // is non-zero.
    if (m_n_molecules_global == 0)
        {
        return;
        }

    // access local molecule data
    // need to move this on top because of scoping issues
    Index2D molecule_indexer = getMoleculeIndexer();
    unsigned int nmol = molecule_indexer.getH();

    ArrayHandle<unsigned int> h_molecule_length(getMoleculeLengths(),
                                                access_location::host,
                                                access_mode::read);
    ArrayHandle<unsigned int> h_molecule_list(getMoleculeList(),
                                              access_location::host,
                                              access_mode::read);

    // access particle data
    ArrayHandle<unsigned int> h_body(m_pdata->getBodies(),
                                     access_location::host,
                                     access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                       access_location::host,
                                       access_mode::read);

    // access net force and torque acting on constituent particles
    ArrayHandle<Scalar4> h_net_force(m_pdata->getNetForce(),
                                     access_location::host,
                                     access_mode::readwrite);
    ArrayHandle<Scalar4> h_net_torque(m_pdata->getNetTorqueArray(),
                                      access_location::host,
                                      access_mode::readwrite);
    ArrayHandle<Scalar> h_net_virial(m_pdata->getNetVirial(),
                                     access_location::host,
                                     access_mode::readwrite);

    // access the force and torque array for the central particle
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar4> h_torque(m_torque, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);

    // access rigid body definition
    ArrayHandle<Scalar3> h_body_pos(m_body_pos, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_body_len(m_body_len, access_location::host, access_mode::read);

    // reset constraint forces and torques
    memset(h_force.data, 0, sizeof(Scalar4) * m_pdata->getN());
    memset(h_torque.data, 0, sizeof(Scalar4) * m_pdata->getN());
    memset(h_virial.data, 0, sizeof(Scalar) * m_virial.getNumElements());

    unsigned int n_particles_local = m_pdata->getN() + m_pdata->getNGhosts();
    size_t net_virial_pitch = m_pdata->getNetVirial().getPitch();

    PDataFlags flags = m_pdata->getFlags();
    bool compute_virial = false;
    if (flags[pdata_flag::pressure_tensor])
        {
        compute_virial = true;
        }

    // loop over all molecules, also incomplete ones
    for (unsigned int ibody = 0; ibody < nmol; ibody++)
        {
        // get central particle tag from first particle in molecule
        assert(h_molecule_length.data[ibody] > 0);
        unsigned int first_idx = h_molecule_list.data[molecule_indexer(0, ibody)];

        assert(first_idx < m_pdata->getN() + m_pdata->getNGhosts());
        unsigned int central_tag = h_body.data[first_idx];

        assert(central_tag <= m_pdata->getMaximumTag());
        unsigned int central_idx = h_rtag.data[central_tag];

        if (central_idx >= n_particles_local)
            continue;

        // the central particle must be present
        assert(central_tag == h_tag.data[first_idx]);

        // central particle position and orientation
        Scalar4 postype = h_postype.data[central_idx];
        quat<Scalar> orientation(h_orientation.data[central_idx]);

        // body type
        unsigned int type = __scalar_as_int(postype.w);

        // sum up forces and torques from constituent particles
        for (unsigned int constituent_index = 0; constituent_index < h_molecule_length.data[ibody];
             ++constituent_index)
            {
            unsigned int idxj = h_molecule_list.data[molecule_indexer(constituent_index, ibody)];
            assert(idxj < m_pdata->getN() + m_pdata->getNGhosts());

            assert(idxj == central_idx || constituent_index > 0);
            if (idxj == central_idx)
                continue;

            // force and torque on particle
            Scalar4 net_force = h_net_force.data[idxj];
            Scalar4 net_torque = h_net_torque.data[idxj];
            vec3<Scalar> f(net_force);

            // zero net energy on constituent particles to avoid double counting
            // also zero net force and torque for consistency
            h_net_force.data[idxj] = make_scalar4(0.0, 0.0, 0.0, 0.0);
            h_net_torque.data[idxj] = make_scalar4(0.0, 0.0, 0.0, 0.0);

            // only add forces for local central particles
            if (central_idx < m_pdata->getN())
                {
                // if the central particle is local, the molecule should be complete
                if (h_molecule_length.data[ibody] != h_body_len.data[type] + 1)
                    {
                    std::ostringstream error_msg;
                    error_msg << "Composite particle with body tag " << central_tag
                              << " is incomplete.";
                    throw std::runtime_error(error_msg.str());
                    }

                // sum up center of mass force
                h_force.data[central_idx].x += f.x;
                h_force.data[central_idx].y += f.y;
                h_force.data[central_idx].z += f.z;

                // sum up energy
                h_force.data[central_idx].w += net_force.w;

                // fetch relative position from rigid body definition
                vec3<Scalar> dr(h_body_pos.data[m_body_idx(type, constituent_index - 1)]);

                // rotate into space frame
                vec3<Scalar> dr_space = rotate(orientation, dr);

                // torque = r x f
                vec3<Scalar> delta_torque(cross(dr_space, f));
                h_torque.data[central_idx].x += delta_torque.x;
                h_torque.data[central_idx].y += delta_torque.y;
                h_torque.data[central_idx].z += delta_torque.z;

                /* from previous rigid body implementation: Access Torque elements from a single
                   particle. Right now I will am assuming that the particle and rigid body reference
                   frames are the same. Probably have to rotate first.
                 */
                h_torque.data[central_idx].x += net_torque.x;
                h_torque.data[central_idx].y += net_torque.y;
                h_torque.data[central_idx].z += net_torque.z;

                if (compute_virial)
                    {
                    // sum up virial
                    Scalar virialxx = h_net_virial.data[0 * net_virial_pitch + idxj];
                    Scalar virialxy = h_net_virial.data[1 * net_virial_pitch + idxj];
                    Scalar virialxz = h_net_virial.data[2 * net_virial_pitch + idxj];
                    Scalar virialyy = h_net_virial.data[3 * net_virial_pitch + idxj];
                    Scalar virialyz = h_net_virial.data[4 * net_virial_pitch + idxj];
                    Scalar virialzz = h_net_virial.data[5 * net_virial_pitch + idxj];

                    // subtract intra-body virial prt
                    h_virial.data[0 * m_virial_pitch + central_idx] += virialxx - f.x * dr_space.x;
                    h_virial.data[1 * m_virial_pitch + central_idx] += virialxy - f.x * dr_space.y;
                    h_virial.data[2 * m_virial_pitch + central_idx] += virialxz - f.x * dr_space.z;
                    h_virial.data[3 * m_virial_pitch + central_idx] += virialyy - f.y * dr_space.y;
                    h_virial.data[4 * m_virial_pitch + central_idx] += virialyz - f.y * dr_space.z;
                    h_virial.data[5 * m_virial_pitch + central_idx] += virialzz - f.z * dr_space.z;
                    }
                }

            // zero net virial
            h_net_virial.data[0 * net_virial_pitch + idxj] = 0.0;
            h_net_virial.data[1 * net_virial_pitch + idxj] = 0.0;
            h_net_virial.data[2 * net_virial_pitch + idxj] = 0.0;
            h_net_virial.data[3 * net_virial_pitch + idxj] = 0.0;
            h_net_virial.data[4 * net_virial_pitch + idxj] = 0.0;
            h_net_virial.data[5 * net_virial_pitch + idxj] = 0.0;
            }
        }
    }

/* Set position, velocity, and type of constituent particles in rigid bodies in the 1st or second
 * half of integration on the CPU based on the body center of mass and particle relative position in
 * each body frame.
 */

void ForceComposite::updateCompositeParticles(uint64_t timestep)
    {
    // If no rigid bodies exist return early. This also prevents accessing arrays assuming that this
    // is non-zero.
    if (m_n_molecules_global == 0)
        {
        return;
        }

    // access molecule order (this needs to be on top because of ArrayHandle scope) and its
    // pervasive use across this function.
    ArrayHandle<unsigned int> h_molecule_order(getMoleculeOrder(),
                                               access_location::host,
                                               access_mode::read);
    ArrayHandle<unsigned int> h_molecule_len(getMoleculeLengths(),
                                             access_location::host,
                                             access_mode::read);
    ArrayHandle<unsigned int> h_molecule_idx(getMoleculeIndex(),
                                             access_location::host,
                                             access_mode::read);

    // access the particle data arrays
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::readwrite);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                       access_location::host,
                                       access_mode::readwrite);
    ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);

    ArrayHandle<unsigned int> h_body(m_pdata->getBodies(),
                                     access_location::host,
                                     access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

    // access body positions and orientations
    ArrayHandle<Scalar3> h_body_pos(m_body_pos, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_body_orientation(m_body_orientation,
                                            access_location::host,
                                            access_mode::read);
    ArrayHandle<unsigned int> h_body_types(m_body_types, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_body_len(m_body_len, access_location::host, access_mode::read);

    const BoxDim& box = m_pdata->getBox();
    const BoxDim& global_box = m_pdata->getGlobalBox();

    // we need to update both local and ghost particles
    unsigned int n_particles_local = m_pdata->getN() + m_pdata->getNGhosts();
    for (unsigned int particle_index = 0; particle_index < n_particles_local; particle_index++)
        {
        unsigned int central_tag = h_body.data[particle_index];

        // Do nothing with floppy bodies, since we don't need to update their positions or
        // orientations here.
        if (central_tag >= MIN_FLOPPY)
            {
            continue;
            }

        // body tag equals tag for central particle
        assert(central_tag <= m_pdata->getMaximumTag());
        unsigned int central_idx = h_rtag.data[central_tag];

        // If this is a rigid body center continue, since we do not need to update its position or
        // orientation (the integrator methods do this).
        if (particle_index == central_idx)
            {
            continue;
            }

        // If the central particle is not local, then we cannot update the position and orientation
        // of this particle. Ideally, this would perform an error check. However, that is not
        // feasible as ForceComposite does not have knowledge of which ghost particles are within
        // the interaction ghost width (and need therefore need to be updated) vs those that are
        // communicated to make bodies whole.
        if (central_idx == NOT_LOCAL)
            {
            continue;
            }

        // central particle position and orientation
        assert(central_idx <= m_pdata->getN() + m_pdata->getNGhosts());

        Scalar4 postype = h_postype.data[central_idx];
        vec3<Scalar> pos(postype);
        quat<Scalar> orientation(h_orientation.data[central_idx]);

        // body type
        unsigned int type = __scalar_as_int(postype.w);

        unsigned int body_len = h_body_len.data[type];
        unsigned int mol_idx = h_molecule_idx.data[particle_index];
        // Checks if the number of local particle in a molecule denoted by
        // h_molecule_len.data[particle_index] is equal to the number of particles in the rigid body
        // definition `body_len`. As above, this error check *should* be performed for all local and
        // ghost particles within the interaction ghost width. However, that check is not feasible
        // here. At least catch this error for particles local to this rank.
        if (body_len != h_molecule_len.data[mol_idx] - 1)
            {
            if (particle_index < m_pdata->getN())
                {
                // if the molecule is incomplete and has local members, this is an error
                std::ostringstream error_msg;
                error_msg << "Error while updating constituent particles:"
                          << "Composite particle with body tag " << central_tag
                          << " incomplete: " << "body_len=" << body_len
                          << ", molecule_len=" << h_molecule_len.data[mol_idx] - 1;
                throw std::runtime_error(error_msg.str());
                }

            // otherwise we must ignore it
            continue;
            }

        int3 img = h_image.data[central_idx];

        // fetch relative index in body from molecule list
        assert(h_molecule_order.data[particle_index] > 0);
        unsigned int idx_in_body = h_molecule_order.data[particle_index] - 1;

        vec3<Scalar> local_pos(h_body_pos.data[m_body_idx(type, idx_in_body)]);
        vec3<Scalar> dr_space = rotate(orientation, local_pos);

        // update position and orientation
        vec3<Scalar> updated_pos(pos);
        quat<Scalar> local_orientation(h_body_orientation.data[m_body_idx(type, idx_in_body)]);

        updated_pos += dr_space;
        quat<Scalar> updated_orientation = orientation * local_orientation;

        // this runs before the ForceComputes,
        // wrap into box, allowing rigid bodies to span multiple images
        int3 imgi = box.getImage(vec_to_scalar3(updated_pos));
        int3 negimgi = make_int3(-imgi.x, -imgi.y, -imgi.z);
        updated_pos = global_box.shift(updated_pos, negimgi);

        h_postype.data[particle_index]
            = make_scalar4(updated_pos.x,
                           updated_pos.y,
                           updated_pos.z,
                           __int_as_scalar(h_body_types.data[m_body_idx(type, idx_in_body)]));
        h_orientation.data[particle_index] = quat_to_scalar4(updated_orientation);
        h_image.data[particle_index] = img + imgi;
        }
    }

namespace detail
    {
void export_ForceComposite(pybind11::module& m)
    {
    pybind11::class_<ForceComposite, MolecularForceCompute, std::shared_ptr<ForceComposite>>(
        m,
        "ForceComposite")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>())
        .def("setBody", &ForceComposite::setBody)
        .def("getBody", &ForceComposite::getBody)
        .def("validateRigidBodies", &ForceComposite::validateRigidBodies)
        .def("createRigidBodies", &ForceComposite::pyCreateRigidBodies)
        .def("updateCompositeParticles", &ForceComposite::updateCompositeParticles);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
