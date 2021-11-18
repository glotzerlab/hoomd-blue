// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: jglaser

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
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#ifdef ENABLE_HIP
#include "hoomd/GPUPartition.cuh"
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
                          std::vector<Scalar4>& orientation,
                          std::vector<Scalar>& charge,
                          std::vector<Scalar>& diameter);

    //! Returns true because we compute the torque on the central particle
    virtual bool isAnisotropic()
        {
        return true;
        }

#ifdef ENABLE_MPI
    //! Get ghost particle fields requested by this pair potential
    virtual CommFlags getRequestedCommFlags(uint64_t timestep);
#endif

    /// Update the constituent particles of a composite particle using the position, velocity
    /// and orientation of the central particle.
    virtual void updateCompositeParticles(uint64_t timestep);

    /// Validate rigid body constituent particles. The method purposely does not check
    /// positions or orientation.
    virtual void validateRigidBodies();

    //! Create rigid body constituent particles
    virtual void createRigidBodies();

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
        pybind11::list charges = v["charges"];
        pybind11::list diameters = v["diameters"];
        auto N = pybind11::len(positions);
        // Ensure proper list lengths
        for (const auto& list : {types, orientations, charges, diameters})
            {
            if (pybind11::len(list) != N)
                {
                throw std::runtime_error("All attributes of a rigid body must be the same length.");
                }
            }

        // extract the data from the python lists
        std::vector<Scalar3> pos_vector;
        std::vector<Scalar4> orientation_vector;
        std::vector<Scalar> charge_vector;
        std::vector<Scalar> diameter_vector;
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

            charge_vector.emplace_back(charges[i].cast<Scalar>());
            diameter_vector.emplace_back(diameters[i].cast<Scalar>());
            type_vector.emplace_back(m_pdata->getTypeByName(types[i].cast<std::string>()));
            }

        setParam(m_pdata->getTypeByName(typ),
                 type_vector,
                 pos_vector,
                 orientation_vector,
                 charge_vector,
                 diameter_vector);
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
        pybind11::list charges;
        pybind11::list diameters;

        for (unsigned int i = 0; i < N; i++)
            {
            auto index = m_body_idx(body_type_id, i);
            positions.append(pybind11::make_tuple(h_body_pos.data[index].x,
                                                  h_body_pos.data[index].y,
                                                  h_body_pos.data[index].z));
            orientations.append(pybind11::make_tuple(h_body_orientation.data[index].x,
                                                     h_body_orientation.data[index].y,
                                                     h_body_orientation.data[index].z,
                                                     h_body_orientation.data[index].w));
            types.append(m_pdata->getNameByType(h_body_types.data[index]));
            charges.append(m_body_charge[body_type_id][i]);
            diameters.append(m_body_diameter[body_type_id][i]);
            }
        pybind11::dict v;
        v["constituent_types"] = types;
        v["positions"] = positions;
        v["orientations"] = orientations;
        v["charges"] = charges;
        v["diameters"] = diameters;
        return std::move(v);
        }

    protected:
    bool m_bodies_changed;          //!< True if constituent particles have changed
    bool m_particles_added_removed; //!< True if particles have been added or removed

    GlobalArray<unsigned int> m_body_types;  //!< Constituent particle types per type id (2D)
    GlobalArray<Scalar3> m_body_pos;         //!< Constituent particle offsets per type id (2D)
    GlobalArray<Scalar4> m_body_orientation; //!< Constituent particle orientations per type id (2D)
    GlobalArray<unsigned int> m_body_len;    //!< Length of body per type id

    std::vector<std::vector<Scalar>> m_body_charge;   //!< Constituent particle charges
    std::vector<std::vector<Scalar>> m_body_diameter; //!< Constituent particle diameters
    Index2D m_body_idx;                               //!< Indexer for body parameters

    std::vector<Scalar> m_d_max;       //!< Maximum body diameter per constituent particle type
    std::vector<bool> m_d_max_changed; //!< True if maximum body diameter changed (per type)
    std::vector<Scalar> m_body_max_diameter; //!< List of diameters for all body types
    Scalar m_global_max_d;                   //!< Maximum over all body diameters

#ifdef ENABLE_MPI
    /// The system's communicator.
    std::shared_ptr<Communicator> m_comm;
#endif

    //! Method to be called when particles are added or removed
    void slotPtlsAddedRemoved()
        {
        m_particles_added_removed = true;
        }

    //! Returns the maximum diameter over all rigid bodies
    Scalar getMaxBodyDiameter()
        {
        if (m_global_max_d_changed)
            {
            // find maximum diameter over all bodies
            Scalar d_max(0.0);
            ArrayHandle<unsigned int> h_body_len(m_body_len,
                                                 access_location::host,
                                                 access_mode::read);
            for (unsigned int i = 0; i < m_pdata->getNTypes(); ++i)
                {
                if (h_body_len.data[i] != 0 && m_body_max_diameter[i] > d_max)
                    d_max = m_body_max_diameter[i];
                }

            // cache value
            m_global_max_d = d_max;
            m_global_max_d_changed = false;

            m_exec_conf->msg->notice(7)
                << "ForceComposite: Maximum body diameter is " << m_global_max_d << std::endl;
            }

        return m_global_max_d;
        }

    //! Return the requested minimum ghost layer width
    virtual Scalar requestExtraGhostLayerWidth(unsigned int type);

    //! Compute the forces and torques on the central particle
    virtual void computeForces(uint64_t timestep);

    //! Helper method to calculate the body diameter
    Scalar getBodyDiameter(unsigned int body_type);

    private:
    bool m_global_max_d_changed; //!< True if we updated any rigid body
    };

namespace detail
    {
//! Exports the ForceComposite to python
void export_ForceComposite(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
