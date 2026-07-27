// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "MolecularForceCompute.h"
#include "NeighborList.h"

/*! \file ForceComposite.h
    \brief Implementation of a rigid body force compute

    Rigid body data is stored per type. Every rigid body is defined by a unique central particle of
    the rigid body type. A rigid body can only have one particle of that type.

    Nested rigid bodies are not supported, i.e. when a rigid body contains a rigid body particle of
    another type.

    The particle data body tag is equal to the tag of central particle, and therefore
    not-contiguous.  The molecule/body id can therefore be used to look up the central particle
    easily.

    Notes:
        - All functions that expect molecules must check m_n_molecules_global first to see if any
        molecules exist. If none exist then, we cannot trust the arrays of MolecularForceCompute to
        be allocated, and should short-circuit the functions with an early return.
        - The split between validation, creation, and constituent particle placement is intentional
        even if it isn't "optimal". Since creation and validation are only called infrequently and
        updating is efficient, this preserves the most readability without sacrificing meaningfully
        performance.

    Communication:

    The communication scheme for ForceComposite is split between ForceComposite, Communicator, and
    IntegratorTwoStep. ForceComposite works with Communicator to adjust the ghost communication
    width for the rigid body types. For every ghost consitutent particle, the corresponding central
    particle must be present to compute the most up to date constituent position / orientation (as
    well as all other particles in the body due to indexing limitations). To implement this,
    Communicator requests a special body ghost width that ForceComposite computes. See
    requestBodyGhostLayerWidth for details on the body ghost width. Communicator then takes the
    maximum of the existing interaction ghost width and the body ghost width to determine the final
    ghost width for central particles.

    To ensure that constituent particles are synchronized between their home and neighboring ranks,
    IntegratorTwoStep updates the central particles, then updates the constituents
    (updateCompositeParticles), then communicates, and Communicator updates the constituents again.
    The first update is needed so that the constituent particles are migrated and added to ghost
    layers when needed. The update after communication is needed to ensure that the ghost
    constituents are placed correctly according to the ghost central particles after communicating.

    Working within the above framework, the home rank for the central particle must also be able
    to access all ghost constituents when summing the net force and torque. The worst case here
    is a central particle right on the domain boundary and the constituent particle at a distance
    equal to the ghost width into the ghost layer. Therefore, the minimum ghost width for a
    constituent is the maximum distance for any particle of that type to its central particle.
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __ForceComposite_H__
#define __ForceComposite_H__

namespace hoomd
    {
namespace md
    {
class PYBIND11_EXPORT ForceComposite : public MolecularForceCompute
    {
    public:
    //! Constructs the compute
    ForceComposite(std::shared_ptr<SystemDefinition> sysdef);

    //! Destructor
    virtual ~ForceComposite();

    //! Set the coordinates for the template for a rigid body of type typeid
    /*! \param body_type The type of rigid body
     * \param type Types of the constituent particles
     * \param pos Relative positions of the constituent particles
     * \param orientation Orientations of the constituent particles
     */
    virtual void setParam(unsigned int body_typeid,
                          std::vector<unsigned int>& type,
                          std::vector<Scalar3>& pos,
                          std::vector<Scalar4>& orientation);

#ifdef ENABLE_MPI
    //! Get ghost particle fields requested by this pair potential
    CommFlags getRequestedCommFlags(uint64_t timestep) override;
#endif

    /// Update the constituent particles of a composite particle using the position, velocity
    /// and orientation of the central particle.
    virtual void updateCompositeParticles(uint64_t timestep);

    /// Validate rigid body constituent particles. The method purposely does not check
    /// positions or orientation.
    virtual void validateRigidBodies();

    //! Create rigid body constituent particles
    void pyCreateRigidBodies(pybind11::dict charges, pybind11::dict masses);

    //! Create rigid body constituent particles
    virtual void
    createRigidBodies(const std::unordered_map<unsigned int, std::vector<Scalar>> charges,
                      const std::unordered_map<unsigned int, std::vector<Scalar>> masses);

    /// Construct from a Python dictionary
    void setBody(std::string typ, pybind11::object v)
        {
        if (v.is_none())
            {
            return;
            }
        pybind11::list types = v["constituent_types"];
        pybind11::list positions = v["positions"];
        pybind11::list orientations = v["orientations"];
        auto N = pybind11::len(positions);
        // Ensure proper list lengths
        for (const auto& list : {types, orientations})
            {
            if (pybind11::len(list) != N)
                {
                throw std::runtime_error("All attributes of a rigid body must be the same length.");
                }
            }

        // extract the data from the python lists
        std::vector<Scalar3> pos_vector;
        std::vector<Scalar4> orientation_vector;
        std::vector<unsigned int> type_vector;

        for (size_t i(0); i < N; ++i)
            {
            pybind11::tuple position_i(positions[i]);
            pos_vector.emplace_back(make_scalar3(position_i[0].cast<Scalar>(),
                                                 position_i[1].cast<Scalar>(),
                                                 position_i[2].cast<Scalar>()));

            pybind11::tuple orientation_i(orientations[i]);
            orientation_vector.emplace_back(make_scalar4(orientation_i[0].cast<Scalar>(),
                                                         orientation_i[1].cast<Scalar>(),
                                                         orientation_i[2].cast<Scalar>(),
                                                         orientation_i[3].cast<Scalar>()));

            type_vector.emplace_back(m_pdata->getTypeByName(types[i].cast<std::string>()));
            }

        setParam(m_pdata->getTypeByName(typ), type_vector, pos_vector, orientation_vector);
        }

    /// Convert parameters to a python dictionary
    pybind11::object getBody(std::string body_type)
        {
        auto body_type_id = m_pdata->getTypeByName(body_type);
        ArrayHandle<unsigned int> h_body_len(m_body_len,
                                             access_location::host,
                                             access_mode::readwrite);
        unsigned int N = h_body_len.data[body_type_id];
        if (N == 0)
            {
            return pybind11::none();
            }
        ArrayHandle<Scalar3> h_body_pos(m_body_pos, access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_body_orientation(m_body_orientation,
                                                access_location::host,
                                                access_mode::read);
        ArrayHandle<unsigned int> h_body_types(m_body_types,
                                               access_location::host,
                                               access_mode::read);

        pybind11::list positions;
        pybind11::list orientations;
        pybind11::list types;

        for (unsigned int i = 0; i < N; i++)
            {
            auto index = m_body_idx(body_type_id, i);
            positions.append(pybind11::make_tuple(static_cast<Scalar>(h_body_pos.data[index].x),
                                                  static_cast<Scalar>(h_body_pos.data[index].y),
                                                  static_cast<Scalar>(h_body_pos.data[index].z)));
            orientations.append(
                pybind11::make_tuple(static_cast<Scalar>(h_body_orientation.data[index].x),
                                     static_cast<Scalar>(h_body_orientation.data[index].y),
                                     static_cast<Scalar>(h_body_orientation.data[index].z),
                                     static_cast<Scalar>(h_body_orientation.data[index].w)));
            types.append(m_pdata->getNameByType(h_body_types.data[index]));
            }
        pybind11::dict v;
        v["constituent_types"] = types;
        v["positions"] = positions;
        v["orientations"] = orientations;
        return v;
        }

    /// Get the number of free particles (global)
    unsigned int getNFreeParticlesGlobal()
        {
        return m_n_free_particles_global;
        }

    /// Get constituent particle types per type id
    const GPUArray<unsigned int>& getBodyTypes() const
        {
        return m_body_types;
        }

    /// Get constituent particle offsets per type id
    const GPUArray<Scalar3>& getBodyOffsets() const
        {
        return m_body_pos;
        }

    /// Get constituent particle orientations per type id
    const GPUArray<Scalar4>& getBodyOrientations() const
        {
        return m_body_orientation;
        }

    /// Get length of body per type id
    const GPUArray<unsigned int>& getBodyLengths() const
        {
        return m_body_len;
        }

    /// Get body parameter indexer
    const Index2D& getBodyIndexer()
        {
        return m_body_idx;
        }

    //! Get rigid centers
    GPUVector<unsigned int>& getRigidCenters()
        {
        checkParticlesSorted();
        return m_rigid_center;
        }

    //! Get lookup centers
    GPUVector<unsigned int>& getLookupCenters()
        {
        checkParticlesSorted();
        return m_lookup_center;
        }

    //! Get number of local rigid bodies
    const unsigned int getNLocal() const
        {
        return m_n_rigid;
        }

    protected:
    bool m_bodies_changed;          //!< True if constituent particles have changed
    bool m_particles_added_removed; //!< True if particles have been added or removed

    /// The number of free particles in the simulation box.
    unsigned int m_n_free_particles_global;

    GPUArray<unsigned int> m_body_types;  //!< Constituent particle types per type id (2D)
    GPUArray<Scalar3> m_body_pos;         //!< Constituent particle offsets per type id (2D)
    GPUArray<Scalar4> m_body_orientation; //!< Constituent particle orientations per type id (2D)
    GPUArray<unsigned int> m_body_len;    //!< Length of body per type id

    Index2D m_body_idx; //!< Indexer for body parameters

    std::vector<Scalar> m_d_max;       //!< Maximum body diameter per constituent particle type
    std::vector<bool> m_d_max_changed; //!< True if maximum body diameter changed (per type)

    unsigned int m_n_rigid;                  //!< Number of rigid bodies on the local rank.
    GPUVector<unsigned int> m_rigid_center;  //!< Local particle indices of all central particles
    GPUVector<unsigned int> m_lookup_center; //!< Lookup particle index -> central particle index

#ifdef ENABLE_MPI
    /// The system's communicator.
    std::shared_ptr<Communicator> m_comm;
#endif

    //! Method to be called when particles are added or removed
    void slotPtlsAddedRemoved()
        {
        m_particles_added_removed = true;
        }

    //! Helper function to check if particles have been sorted and rebuild indices if necessary
    void checkParticlesSorted() override
        {
        if (m_rebuild_molecules)
            // identify center particles for use in GPU kernel
            findRigidCenters();

        // Must be called second since the method sets m_rebuild_molecules
        // to false if it is true.
        MolecularForceCompute::checkParticlesSorted();
        }

    //! Helper kernel to sort rigid bodies by their center particles
    virtual void findRigidCenters();

    /// Return the requested minimum ghost layer width for a body's central particle.
    virtual Scalar requestBodyGhostLayerWidth(unsigned int type, Scalar* h_r_ghost);

    //! Compute the forces and torques on the central particle
    void computeForces(uint64_t timestep) override;
    };

    } // end namespace md
    } // end namespace hoomd

#endif
