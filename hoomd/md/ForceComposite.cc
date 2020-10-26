// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

#include "ForceComposite.h"
#include "hoomd/VectorMath.h"

#include <map>
#include <string.h>
namespace py = pybind11;

/*! \file ForceComposite.cc
    \brief Contains code for the ForceComposite class
*/

/*! \param sysdef SystemDefinition containing the ParticleData to compute forces on
*/
ForceComposite::ForceComposite(std::shared_ptr<SystemDefinition> sysdef)
        : MolecularForceCompute(sysdef), m_bodies_changed(false), m_ptls_added_removed(false),
         m_global_max_d(0.0),
         m_memory_initialized(false),
         #ifdef ENABLE_MPI
         m_comm_ghost_layer_connected(false),
         #endif
         m_global_max_d_changed(true)
    {
    // connect to the ParticleData to receive notifications when the number of types changes
    m_pdata->getNumTypesChangeSignal().connect<ForceComposite, &ForceComposite::slotNumTypesChange>(this);

    m_pdata->getGlobalParticleNumberChangeSignal().connect<ForceComposite, &ForceComposite::slotPtlsAddedRemoved>(this);

    // connect to box change signal
    m_pdata->getCompositeParticlesSignal().connect<ForceComposite, &ForceComposite::getMaxBodyDiameter>(this);
    }

//! Destructor
ForceComposite::~ForceComposite()
    {
    // disconnect from signal in ParticleData;
    m_pdata->getNumTypesChangeSignal().disconnect<ForceComposite, &ForceComposite::slotNumTypesChange>(this);
    m_pdata->getGlobalParticleNumberChangeSignal().disconnect<ForceComposite, &ForceComposite::slotPtlsAddedRemoved>(this);
    m_pdata->getCompositeParticlesSignal().disconnect<ForceComposite, &ForceComposite::getMaxBodyDiameter>(this);
    #ifdef ENABLE_MPI
    if (m_comm_ghost_layer_connected)
        m_comm->getExtraGhostLayerWidthRequestSignal().disconnect<ForceComposite, &ForceComposite::requestExtraGhostLayerWidth>(this);
    #endif
    }

void ForceComposite::lazyInitMem()
    {
    if (m_memory_initialized)
        return;

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

    m_body_charge.resize(m_pdata->getNTypes());
    m_body_diameter.resize(m_pdata->getNTypes());

    m_d_max.resize(m_pdata->getNTypes(), Scalar(0.0));
    m_d_max_changed.resize(m_pdata->getNTypes(), false);

    m_body_max_diameter.resize(m_pdata->getNTypes(), Scalar(0.0));

    m_memory_initialized = true;
    }

void ForceComposite::setParam(unsigned int body_typeid,
    std::vector<unsigned int>& type,
    std::vector<Scalar3>& pos,
    std::vector<Scalar4>& orientation,
    std::vector<Scalar>& charge,
    std::vector<Scalar>& diameter)
    {
    lazyInitMem();

    assert(m_body_types.getPitch() >= m_pdata->getNTypes());
    assert(m_body_pos.getPitch() >= m_pdata->getNTypes());
    assert(m_body_orientation.getPitch() >= m_pdata->getNTypes());
    assert(m_body_charge.size() >= m_pdata->getNTypes());
    assert(m_body_diameter.size() >= m_pdata->getNTypes());

    if (body_typeid >= m_pdata->getNTypes())
        {
        m_exec_conf->msg->error() << "constrain.rigid(): Invalid rigid body type." << std::endl;
        throw std::runtime_error("Error initializing ForceComposite");
        }

    if (type.size() != pos.size() || orientation.size() != pos.size())
        {
        m_exec_conf->msg->error() << "constrain.rigid(): Constituent particle lists"
            <<" (position, orientation, type) are of unequal length." << std::endl;
        throw std::runtime_error("Error initializing ForceComposite");
        }
    if (charge.size() && charge.size() != pos.size())
        {
        m_exec_conf->msg->error() << "constrain.rigid(): Charges are non-empty but of different"
            <<" length than the positions." << std::endl;
        throw std::runtime_error("Error initializing ForceComposite");
        }
    if (diameter.size() && diameter.size() != pos.size())
        {
        m_exec_conf->msg->error() << "constrain.rigid(): Diameters are non-empty but of different"
            <<" length than the positions." << std::endl;
        throw std::runtime_error("Error initializing ForceComposite");
        }

    bool body_updated = false;

    bool body_len_changed = false;

    // detect if bodies have changed

        {
        ArrayHandle<unsigned int> h_body_type(m_body_types, access_location::host, access_mode::read);
        ArrayHandle<Scalar3> h_body_pos(m_body_pos, access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_body_orientation(m_body_orientation, access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_body_len(m_body_len, access_location::host, access_mode::readwrite);

        assert(body_typeid < m_body_len.getNumElements());
        if (type.size() != h_body_len.data[body_typeid])
            {
            body_updated = true;

            h_body_len.data[body_typeid] = type.size();
            body_len_changed = true;
            }
        else
            {
            for (unsigned int i = 0; i < type.size(); ++i)
                {
                if (type[i] != h_body_type.data[m_body_idx(body_typeid,i)] ||
                    pos[i].x != h_body_pos.data[m_body_idx(body_typeid,i)].x ||
                    pos[i].y != h_body_pos.data[m_body_idx(body_typeid,i)].y ||
                    pos[i].x != h_body_pos.data[m_body_idx(body_typeid,i)].z ||
                    orientation[i].x != h_body_orientation.data[m_body_idx(body_typeid,i)].x ||
                    orientation[i].y != h_body_orientation.data[m_body_idx(body_typeid,i)].y ||
                    orientation[i].z != h_body_orientation.data[m_body_idx(body_typeid,i)].z ||
                    orientation[i].w != h_body_orientation.data[m_body_idx(body_typeid,i)].w)
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

            m_body_idx = Index2D(m_body_types.getPitch(), m_body_types.getHeight());

            // set memory hints
            lazyInitMem();
            }
        }

    if (body_updated)
        {
            {
            ArrayHandle<unsigned int> h_body_type(m_body_types, access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar3> h_body_pos(m_body_pos, access_location::host, access_mode::readwrite);
            ArrayHandle<Scalar4> h_body_orientation(m_body_orientation, access_location::host, access_mode::readwrite);

            m_body_charge[body_typeid].resize(type.size());
            m_body_diameter[body_typeid].resize(type.size());

            // store body data in GlobalArray
            for (unsigned int i = 0; i < type.size(); ++i)
                {
                h_body_type.data[m_body_idx(body_typeid,i)] = type[i];
                h_body_pos.data[m_body_idx(body_typeid,i)] = pos[i];
                h_body_orientation.data[m_body_idx(body_typeid,i)] = orientation[i];

                m_body_charge[body_typeid][i] = charge[i];
                m_body_diameter[body_typeid][i] = diameter[i];
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

        // story body diameter
        m_body_max_diameter[body_typeid] = getBodyDiameter(body_typeid);

        // indicate that the maximum diameter may have changed
        m_global_max_d_changed = true;
        }
   }

Scalar ForceComposite::getBodyDiameter(unsigned int body_type)
    {
    lazyInitMem();

    m_exec_conf->msg->notice(7) << "ForceComposite: calculating body diameter for type " << m_pdata->getNameByType(body_type) << std::endl;

    // get maximum pairwise distance
    ArrayHandle<unsigned int> h_body_len(m_body_len, access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_body_pos(m_body_pos, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_body_type(m_body_types, access_location::host, access_mode::read);

    Scalar d_max(0.0);

    for (unsigned int i = 0; i < h_body_len.data[body_type]; ++i)
        {
        // distance to central particle
        Scalar3 dr = h_body_pos.data[m_body_idx(body_type,i)];
        Scalar d = sqrt(dot(dr,dr));
        if (d > d_max)
            {
            d_max = d;
            }

        // distance to every other particle
        for (unsigned int j = 0; j < h_body_len.data[body_type]; ++j)
            {
            dr = h_body_pos.data[m_body_idx(body_type,i)]-h_body_pos.data[m_body_idx(body_type,j)];
            d = sqrt(dot(dr,dr));

            if (d > d_max)
                {
                d_max = d;
                }
            }
        }

    return d_max;
    }

void ForceComposite::slotNumTypesChange()
    {
    //! initial allocation if necessary
    lazyInitMem();

    unsigned int old_ntypes = m_body_len.getNumElements();
    unsigned int new_ntypes = m_pdata->getNTypes();

    unsigned int height = m_body_pos.getHeight();

    // resize per-type arrays (2D)
    m_body_types.resize(new_ntypes, height);
    m_body_pos.resize(new_ntypes, height);
    m_body_orientation.resize(new_ntypes, height);

    m_body_charge.resize(new_ntypes);
    m_body_diameter.resize(new_ntypes);

    m_body_idx = Index2D(m_body_pos.getPitch(), height);

    m_body_len.resize(new_ntypes);

    // reset newly added elements to zero
    ArrayHandle<unsigned int> h_body_len(m_body_len, access_location::host, access_mode::readwrite);
    for (unsigned int i = old_ntypes; i < new_ntypes; ++i)
        {
        h_body_len.data[i] = 0;
        }

    m_d_max.resize(new_ntypes, Scalar(0.0));
    m_d_max_changed.resize(new_ntypes, false);

    m_body_max_diameter.resize(new_ntypes,0.0);

    //! update memory hints, after re-allocation
    lazyInitMem();
    }

Scalar ForceComposite::requestExtraGhostLayerWidth(unsigned int type)
    {
    lazyInitMem();

    ArrayHandle<unsigned int> h_body_len(m_body_len, access_location::host, access_mode::read);

    if (m_d_max_changed[type])
        {
        m_d_max[type] = Scalar(0.0);

        assert(m_body_len.getNumElements() > type);

        ArrayHandle<Scalar3> h_body_pos(m_body_pos, access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_body_type(m_body_types, access_location::host, access_mode::read);

        unsigned int ntypes = m_pdata->getNTypes();

        // find maximum body radius over all bodies this type participates in
        for (unsigned int body_type = 0; body_type < ntypes; ++body_type)
            {
            bool is_part_of_body = body_type == type;
            for (unsigned int i = 0; i < h_body_len.data[body_type]; ++i)
                {
                if (h_body_type.data[m_body_idx(body_type,i)] == type)
                    {
                    is_part_of_body = true;
                    }
                }

            if (is_part_of_body)
                {
                for (unsigned int i = 0; i < h_body_len.data[body_type]; ++i)
                    {
                    if (body_type != type && h_body_type.data[m_body_idx(body_type,i)] != type) continue;

                    // distance to central particle
                    Scalar3 dr = h_body_pos.data[m_body_idx(body_type,i)];
                    Scalar d = sqrt(dot(dr,dr));
                    if (d > m_d_max[type])
                        {
                        m_d_max[type] = d;
                        }

                    if (body_type != type)
                        {
                        // for non-central particles, distance to every other particle
                        for (unsigned int j = 0; j < h_body_len.data[body_type]; ++j)
                            {
                            dr = h_body_pos.data[m_body_idx(body_type,i)]-h_body_pos.data[m_body_idx(body_type,j)];
                            d = sqrt(dot(dr,dr));

                            if (d > m_d_max[type])
                                {
                                m_d_max[type] = d;
                                }
                            }
                        }
                    }
                }
            }

        m_d_max_changed[type] = false;

        m_exec_conf->msg->notice(7) << "ForceComposite: requesting ghost layer for type "
            << m_pdata->getNameByType(type) << ": " << m_d_max[type] << std::endl;
        }

    return m_d_max[type];
    }

void ForceComposite::validateRigidBodies(bool create)
    {
    lazyInitMem();

    if (m_bodies_changed || m_ptls_added_removed)
        {
        // check validity of rigid body types: no nested rigid bodies
        unsigned int ntypes = m_pdata->getNTypes();
        assert(m_body_types.getPitch() >= ntypes);

            {
            ArrayHandle<unsigned int> h_body_type(m_body_types, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_body_len(m_body_len, access_location::host, access_mode::read);
            for (unsigned int itype = 0; itype < ntypes; ++itype)
                {
                for (unsigned int j= 0; j < h_body_len.data[itype]; ++j)
                    {
                    assert(h_body_type.data[m_body_idx(itype,j)] <= ntypes);
                    if (h_body_len.data[h_body_type.data[m_body_idx(itype,j)]] != 0)
                        {
                        m_exec_conf->msg->error() << "constrain.rigid(): A rigid body type may not contain constituent particles "
                            << "that are also rigid bodies!" << std::endl;
                        throw std::runtime_error("Error initializing ForceComposite");
                        }
                    }
                }
            }

        SnapshotParticleData<Scalar> snap;

        // take a snapshot on rank 0
        m_pdata->takeSnapshot(snap);

        // constituent particles added as rigid body copies
        unsigned int n_add_ptls = 0;

        // True if we need to remove all constituent particles from the system first
        bool need_remove_bodies = false;

        if (m_exec_conf->getRank() == 0)
            {
            // access body data
            ArrayHandle<unsigned int> h_body_len(m_body_len, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_body_type(m_body_types, access_location::host, access_mode::read);

            typedef std::map<unsigned int, unsigned int> map_t;
            map_t count_body_ptls;

            // count number of constituent particles to add
            for (unsigned i = 0; i < snap.size; ++i)
                {
                assert(snap.type[i] < ntypes);

                bool is_central_ptl = h_body_len.data[snap.type[i]] != 0;

                if (create)
                    {
                    if (snap.body[i] < MIN_FLOPPY)
                        {
                        if (!is_central_ptl)
                            {
                            need_remove_bodies = true;
                            }
                        else
                            {
                            // wipe out body flag
                            assert(snap.body[i] == i);
                            snap.body[i] = NO_BODY;
                            }
                        }

                    // for each particle of a body type, add a copy of the constituent particles
                    n_add_ptls += h_body_len.data[snap.type[i]];
                    }
                else
                    {
                    // validate constituent particle
                    if (is_central_ptl)
                        {
                        if (snap.body[i] != i)
                            {
                            m_exec_conf->msg->error() << "constrain.rigid(): Central particles must have a body tag identical to their contiguous tag." << std::endl;
                            throw std::runtime_error("Error validating rigid bodies\n");
                            }

                        count_body_ptls.insert(std::make_pair(i,0));
                        }
                    if (snap.body[i] < MIN_FLOPPY)
                        {
                        // check if ptl body tag correctly points to the central particle
                        if (snap.body[i] >= snap.size || snap.body[snap.body[i]] != snap.body[i])
                            {
                            m_exec_conf->msg->error() << "constrain.rigid(): Constituent particle body tags must point to the center particle." << std::endl;
                            throw std::runtime_error("Error validating rigid bodies\n");
                            }

                        if (! is_central_ptl)
                            {
                            unsigned int central_ptl = snap.body[i];
                            unsigned int body_type = snap.type[central_ptl];

                            map_t::iterator it = count_body_ptls.find(central_ptl);
                            if (it == count_body_ptls.end())
                                {
                                m_exec_conf->msg->error() << "constrain.rigid(): Central particle " << snap.body[i]
                                    << " does not precede particle with tag " << i << std::endl;
                                throw std::runtime_error("Error validating rigid bodies\n");
                                }

                            unsigned int n = it->second;
                            if (n == h_body_len.data[body_type])
                                {
                                m_exec_conf->msg->error() << "constrain.rigid(): Number of constituent particles for body " << snap.body[i] << " exceeds definition"
                                     << std::endl;
                                throw std::runtime_error("Error validating rigid bodies\n");
                                }

                            if (h_body_type.data[m_body_idx(body_type, n)] != snap.type[i])
                                {
                                m_exec_conf->msg->error() << "constrain.rigid(): Constituent particle types must be consistent with rigid body parameters." << std::endl;
                                throw std::runtime_error("Error validating rigid bodies\n");
                                }

                            // increase count
                            it->second++;
                            }
                        }
                    }
                }

            if (! create)
                {
                for (map_t::iterator it = count_body_ptls.begin(); it != count_body_ptls.end();++it)
                    {
                    unsigned int central_ptl_type = snap.type[it->first];
                    if (it->second != h_body_len.data[central_ptl_type])
                        {
                        m_exec_conf->msg->error() << "constrain.rigid(): Incomplete rigid body with only " << it->second << " constituent particles "
                            << "instead of " << h_body_len.data[central_ptl_type] << " for body " << it->first << std::endl;
                        throw std::runtime_error("Error validating rigid bodies\n");
                        }
                    }
                }
            }

        #ifdef ENABLE_MPI
        if (m_pdata->getDomainDecomposition())
            {
            bcast(need_remove_bodies, 0, m_exec_conf->getMPICommunicator());
            bcast(n_add_ptls, 0, m_exec_conf->getMPICommunicator());
            }
        #endif

        if (need_remove_bodies)
            {
            m_exec_conf->msg->notice(2)
                << "constrain.rigid(): Removing all particles part of rigid bodies (except central particles)."
                << "Particle tags may change."
                << std::endl;

            // re-initialize, removing rigid bodies
            m_pdata->initializeFromSnapshot(snap, true);

            // update snapshot
            m_pdata->takeSnapshot(snap);
            }

        std::vector<unsigned int> molecule_tag;

        // number of bodies in system
        unsigned int nbodies = 0;

        const BoxDim& global_box = m_pdata->getGlobalBox();
        SnapshotParticleData<Scalar> snap_out = snap;

        if (create)
            {
            if (m_exec_conf->getRank() == 0)
                {
                unsigned int old_size = snap.size;

                // resize and reset global molecule table
                molecule_tag.resize(old_size+n_add_ptls, NO_MOLECULE);

                // access body data
                ArrayHandle<unsigned int> h_body_type(m_body_types, access_location::host, access_mode::read);
                ArrayHandle<Scalar3> h_body_pos(m_body_pos, access_location::host, access_mode::read);
                ArrayHandle<Scalar4> h_body_orientation(m_body_orientation, access_location::host, access_mode::read);
                ArrayHandle<unsigned int> h_body_len(m_body_len, access_location::host, access_mode::read);

                unsigned int snap_idx_out = old_size;

                // create copies
                for (unsigned i = 0; i < old_size; ++i)
                    {
                    assert(snap.type[i] < ntypes);

                    bool is_central_ptl = h_body_len.data[snap.type[i]] != 0;

                    assert(snap.body[i] == NO_BODY);

                    if (is_central_ptl)
                        {
                        unsigned int body_type = snap.type[i];

                        unsigned body_tag = i;

                        // set body id to tag of central ptl
                        snap_out.body[i] = body_tag;

                        // set molecule tag
                        molecule_tag[i] = nbodies;

                        vec3<Scalar> central_pos(snap.pos[i]);
                        quat<Scalar> central_orientation(snap.orientation[i]);
                        int3 central_img = snap.image[i];

                        // insert elements into snapshot
                        unsigned int n= h_body_len.data[body_type];
                        snap_out.insert(snap_idx_out, n);

                        for (unsigned int j = 0; j < n; ++j)
                            {
                            // set type
                            snap_out.type[snap_idx_out] = h_body_type.data[m_body_idx(body_type,j)];

                            // set body index on constituent particle
                            snap_out.body[snap_idx_out] = body_tag;

                            // use contiguous molecule tag
                            molecule_tag[snap_idx_out] = nbodies;

                            // update position and orientation to ensure particles end up in correct domain
                            vec3<Scalar> pos(central_pos);

                            pos += rotate(central_orientation, vec3<Scalar>(h_body_pos.data[m_body_idx(body_type,j)]));
                            quat<Scalar> orientation = central_orientation*quat<Scalar>(h_body_orientation.data[m_body_idx(body_type,j)]);

                            // wrap into box, allowing rigid bodies to span multiple images
                            int3 img = global_box.getImage(vec_to_scalar3(pos));
                            int3 negimg = make_int3(-img.x, -img.y, -img.z);
                            pos = global_box.shift(pos, negimg);

                            snap_out.pos[snap_idx_out] = pos;
                            snap_out.image[snap_idx_out] = central_img + img;
                            snap_out.orientation[snap_idx_out] = orientation;

                            // set charge and diameter
                            snap_out.charge[snap_idx_out] = m_body_charge[body_type][j];
                            snap_out.diameter[snap_idx_out] = m_body_diameter[body_type][j];
                            snap_idx_out++;
                            }

                        nbodies++;
                        }

                    }
                }

            m_exec_conf->msg->notice(2) << "constrain.rigid(): Creating " << nbodies << " rigid bodies (adding "
                << n_add_ptls << " particles)" << std::endl;
            }
        else
            {
            if (m_exec_conf->getRank() == 0)
                {
                molecule_tag.resize(snap.size, NO_MOLECULE);

                typedef std::map<unsigned int, unsigned int> map_t;
                map_t count_body_ptls;

                // access body data
                ArrayHandle<Scalar3> h_body_pos(m_body_pos, access_location::host, access_mode::read);
                ArrayHandle<Scalar4> h_body_orientation(m_body_orientation, access_location::host, access_mode::read);

                // assign contiguous molecule tags and update constituent particle positions and orientations
                for (unsigned i = 0; i < snap_out.size; ++i)
                    {
                    assert(snap_out.type[i] < ntypes);

                    if (snap_out.body[i] < MIN_FLOPPY)
                        {
                        if (snap_out.body[i] == i)
                            {
                            // central particle
                            molecule_tag[i] = nbodies++;
                            count_body_ptls.insert(std::make_pair(snap_out.body[i],0));
                            }
                        else
                            {
                            molecule_tag[i] = molecule_tag[snap_out.body[i]];

                            // update position and orientation to ensure particles end up in correct domain
                            vec3<Scalar> pos(snap_out.pos[snap_out.body[i]]);
                            quat<Scalar> central_orientation(snap_out.orientation[snap_out.body[i]]);
                            int3 central_img = snap_out.image[snap_out.body[i]];

                            map_t::iterator it = count_body_ptls.find(snap_out.body[i]);
                            unsigned int j = it->second;
                            unsigned int body_type = snap_out.type[snap_out.body[i]];
                            pos += rotate(central_orientation, vec3<Scalar>(h_body_pos.data[m_body_idx(body_type,j)]));
                            quat<Scalar> orientation = central_orientation*quat<Scalar>(h_body_orientation.data[m_body_idx(body_type,j)]);

                            // wrap into box, allowing rigid bodies to span multiple images
                            int3 img = global_box.getImage(vec_to_scalar3(pos));
                            int3 negimg = make_int3(-img.x, -img.y, -img.z);
                            pos = global_box.shift(pos, negimg);

                            snap_out.pos[i] = pos;
                            snap_out.image[i] = central_img + img;
                            snap_out.orientation[i] = orientation;

                            it->second++;
                            }
                        }
                    }
                }

           }

        // re-initialize, keeping particles with body != NO_BODY at this time
        m_pdata->initializeFromSnapshot(snap_out, false);

        #ifdef ENABLE_MPI
        if (m_pdata->getDomainDecomposition())
            {
            bcast(molecule_tag, 0, m_exec_conf->getMPICommunicator());
            }
        #endif

        // resize GPU table
        m_molecule_tag.resize(molecule_tag.size());
            {
            // store global molecule information in GlobalArray
            ArrayHandle<unsigned int> h_molecule_tag(m_molecule_tag, access_location::host, access_mode::overwrite);
            std::copy(molecule_tag.begin(), molecule_tag.end(), h_molecule_tag.data);
            }

        // store number of molecules
        m_n_molecules_global = nbodies;

        #ifdef ENABLE_MPI
        if (m_pdata->getDomainDecomposition())
            {
            bcast(m_n_molecules_global, 0, m_exec_conf->getMPICommunicator());
            }
        #endif

        // reset flags
        m_bodies_changed = false;
        m_ptls_added_removed = false;
        }
    }

#ifdef ENABLE_MPI
/*! \param timestep Current time step
 */
CommFlags ForceComposite::getRequestedCommFlags(unsigned int timestep)
    {
    CommFlags flags = CommFlags(0);

    // request orientations
    flags[comm_flag::orientation] = 1;

    // only communicate net virial if needed
    PDataFlags pdata_flags = this->m_pdata->getFlags();
    if (pdata_flags[pdata_flag::isotropic_virial] || pdata_flags[pdata_flag::pressure_tensor])
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
void ForceComposite::computeForces(unsigned int timestep)
    {
    // access local molecule data
    // need to move this on top because of scoping issues
    Index2D molecule_indexer = getMoleculeIndexer();
    unsigned int nmol = molecule_indexer.getH();

    ArrayHandle<unsigned int> h_molecule_length(getMoleculeLengths(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_molecule_list(getMoleculeList(), access_location::host, access_mode::read);

    // access particle data
    ArrayHandle<unsigned int> h_body(m_pdata->getBodies(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);

    // access net force and torque acting on constituent particles
    ArrayHandle<Scalar4> h_net_force(m_pdata->getNetForce(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_net_torque(m_pdata->getNetTorqueArray(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_net_virial(m_pdata->getNetVirial(), access_location::host, access_mode::readwrite);

    // access the force and torque array for the central ptl
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar4> h_torque(m_torque, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);

    // access rigid body definition
    ArrayHandle<Scalar3> h_body_pos(m_body_pos, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_body_len(m_body_len, access_location::host, access_mode::read);

    // reset constraint forces and torques
    memset(h_force.data,0, sizeof(Scalar4)*m_pdata->getN());
    memset(h_torque.data,0, sizeof(Scalar4)*m_pdata->getN());
    memset(h_virial.data,0, sizeof(Scalar)*m_virial.getNumElements());

    unsigned int nptl_local = m_pdata->getN() + m_pdata->getNGhosts();
    unsigned int net_virial_pitch = m_pdata->getNetVirial().getPitch();

    PDataFlags flags = m_pdata->getFlags();
    bool compute_virial = false;
    if (flags[pdata_flag::isotropic_virial] || flags[pdata_flag::pressure_tensor])
        {
        compute_virial = true;
        }

    // loop over all molecules, also incomplete ones
    for (unsigned int ibody = 0; ibody < nmol; ibody++)
        {
        unsigned int len = h_molecule_length.data[ibody];

        // get central ptl tag from first ptl in molecule
        assert(len>0);
        unsigned int first_idx = h_molecule_list.data[molecule_indexer(0,ibody)];

        assert(first_idx < m_pdata->getN() + m_pdata->getNGhosts());
        unsigned int central_tag = h_body.data[first_idx];

        assert(central_tag <= m_pdata->getMaximumTag());
        unsigned int central_idx = h_rtag.data[central_tag];

        if (central_idx >= nptl_local) continue;

        // the central ptl must be present
        assert(central_tag == h_tag.data[first_idx]);

        // central ptl position and orientation
        Scalar4 postype = h_postype.data[central_idx];
        quat<Scalar> orientation(h_orientation.data[central_idx]);

        // body type
        unsigned int type = __scalar_as_int(postype.w);

        // sum up forces and torques from constituent particles
        for (unsigned int jptl = 0; jptl < len; ++jptl)
            {
            unsigned int idxj = h_molecule_list.data[molecule_indexer(jptl,ibody)];
            assert(idxj < m_pdata->getN() + m_pdata->getNGhosts());

            assert(idxj == central_idx || jptl > 0);
            if (idxj == central_idx) continue;

            // force and torque on particle
            Scalar4 net_force = h_net_force.data[idxj];
            Scalar4 net_torque = h_net_torque.data[idxj];
            vec3<Scalar> f(net_force);

            // zero net energy on constituent ptls to avoid double counting
            // also zero net force and torque for consistency
            h_net_force.data[idxj] = make_scalar4(0.0,0.0,0.0,0.0);
            h_net_torque.data[idxj] = make_scalar4(0.0,0.0,0.0,0.0);

            // only add forces for local central particles
            if (central_idx < m_pdata->getN())
                {
                // if the central particle is local, the molecule should be complete
                if (len != h_body_len.data[type] + 1)
                    {
                    m_exec_conf->msg->errorAllRanks() << "constrain.rigid(): Composite particle with body tag "
                                                      << central_tag << " incomplete" << std::endl << std::endl;
                    throw std::runtime_error("Error computing composite particle forces.\n");
                    }

                // sum up center of mass force
                h_force.data[central_idx].x += f.x;
                h_force.data[central_idx].y += f.y;
                h_force.data[central_idx].z += f.z;

                // sum up energy
                h_force.data[central_idx].w += net_force.w;

                // fetch relative position from rigid body definition
                vec3<Scalar> dr(h_body_pos.data[m_body_idx(type, jptl - 1)]);

                // rotate into space frame
                vec3<Scalar> dr_space = rotate(orientation, dr);

                // torque = r x f
                vec3<Scalar> delta_torque(cross(dr_space,f));
                h_torque.data[central_idx].x += delta_torque.x;
                h_torque.data[central_idx].y += delta_torque.y;
                h_torque.data[central_idx].z += delta_torque.z;

                /* from previous rigid body implementation: Access Torque elements from a single particle. Right now I will am assuming that the particle
                    and rigid body reference frames are the same. Probably have to rotate first.
                 */
                h_torque.data[central_idx].x += net_torque.x;
                h_torque.data[central_idx].y += net_torque.y;
                h_torque.data[central_idx].z += net_torque.z;

                if (compute_virial)
                    {
                    // sum up virial
                    Scalar virialxx = h_net_virial.data[0*net_virial_pitch+idxj];
                    Scalar virialxy = h_net_virial.data[1*net_virial_pitch+idxj];
                    Scalar virialxz = h_net_virial.data[2*net_virial_pitch+idxj];
                    Scalar virialyy = h_net_virial.data[3*net_virial_pitch+idxj];
                    Scalar virialyz = h_net_virial.data[4*net_virial_pitch+idxj];
                    Scalar virialzz = h_net_virial.data[5*net_virial_pitch+idxj];

                    // subtract intra-body virial prt
                    h_virial.data[0*m_virial_pitch+central_idx] += virialxx - f.x*dr_space.x;
                    h_virial.data[1*m_virial_pitch+central_idx] += virialxy - f.x*dr_space.y;
                    h_virial.data[2*m_virial_pitch+central_idx] += virialxz - f.x*dr_space.z;
                    h_virial.data[3*m_virial_pitch+central_idx] += virialyy - f.y*dr_space.y;
                    h_virial.data[4*m_virial_pitch+central_idx] += virialyz - f.y*dr_space.z;
                    h_virial.data[5*m_virial_pitch+central_idx] += virialzz - f.z*dr_space.z;
                    }
                }

            // zero net virial
            h_net_virial.data[0*net_virial_pitch+idxj] = 0.0;
            h_net_virial.data[1*net_virial_pitch+idxj] = 0.0;
            h_net_virial.data[2*net_virial_pitch+idxj] = 0.0;
            h_net_virial.data[3*net_virial_pitch+idxj] = 0.0;
            h_net_virial.data[4*net_virial_pitch+idxj] = 0.0;
            h_net_virial.data[5*net_virial_pitch+idxj] = 0.0;
            }
        }
    }

/* Set position and velocity of constituent particles in rigid bodies in the 1st or second half of integration on the CPU
    based on the body center of mass and particle relative position in each body frame.
*/

void ForceComposite::updateCompositeParticles(unsigned int timestep)
    {
    // access molecule order (this needs to be on top because of ArrayHandle scope)
    ArrayHandle<unsigned int> h_molecule_order(getMoleculeOrder(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_molecule_len(getMoleculeLengths(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_molecule_idx(getMoleculeIndex(), access_location::host, access_mode::read);

    // access the particle data arrays
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);
    ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);

    /*
    ArrayHandle<Scalar4> h_angmom(m_pdata->getAngularMomentumArray(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    */

    ArrayHandle<unsigned int> h_body(m_pdata->getBodies(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

    // access body positions and orientations
    ArrayHandle<Scalar3> h_body_pos(m_body_pos, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_body_orientation(m_body_orientation, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_body_len(m_body_len, access_location::host, access_mode::read);

    const BoxDim& box = m_pdata->getBox();
    const BoxDim& global_box = m_pdata->getGlobalBox();

    // we need to update both local and ghost particles
    unsigned int nptl = m_pdata->getN() + m_pdata->getNGhosts();

    for (unsigned int iptl = 0; iptl < nptl; iptl++)
        {
        unsigned int central_tag = h_body.data[iptl];

        if (central_tag >= MIN_FLOPPY)
            continue;

        // body tag equals tag for central ptl
        assert(central_tag <= m_pdata->getMaximumTag());
        unsigned int central_idx = h_rtag.data[central_tag];

        if (central_idx == NOT_LOCAL && iptl >= m_pdata->getN())
            continue;

        if (central_idx == NOT_LOCAL)
            {
            m_exec_conf->msg->errorAllRanks() << "constrain.rigid(): Missing central particle tag " << central_tag
                                              << "!" << std::endl << std::endl;
            throw std::runtime_error("Error updating composite particles.\n");
            }

        // central ptl position and orientation
        assert(central_idx <= m_pdata->getN() + m_pdata->getNGhosts());

        // do not overwrite the central ptl
        if (iptl == central_idx) continue;

        Scalar4 postype = h_postype.data[central_idx];
        vec3<Scalar> pos(postype);
        quat<Scalar> orientation(h_orientation.data[central_idx]);

        // body type
        unsigned int type = __scalar_as_int(postype.w);

        unsigned int body_len = h_body_len.data[type];
        unsigned int mol_idx = h_molecule_idx.data[iptl];
        if (body_len != h_molecule_len.data[mol_idx] - 1)
            {
            if (iptl < m_pdata->getN())
                {
                // if the molecule is incomplete and has local members, this is an error
                m_exec_conf->msg->errorAllRanks() << "constrain.rigid(): Composite particle with body tag "
                                                  << central_tag << " incomplete" << std::endl << std::endl;
                throw std::runtime_error("Error while updating constituent particles.\n");
                }

            // otherwise we must ignore it
            continue;
            }

        int3 img = h_image.data[central_idx];

        // fetch relative index in body from molecule list
        assert(h_molecule_order.data[iptl] > 0);
        unsigned int idx_in_body = h_molecule_order.data[iptl] - 1;

        vec3<Scalar> local_pos(h_body_pos.data[m_body_idx(type,idx_in_body)]);
        vec3<Scalar> dr_space = rotate(orientation, local_pos);

        // update position and orientation
        vec3<Scalar> updated_pos(pos);
        quat<Scalar> local_orientation(h_body_orientation.data[m_body_idx(type, idx_in_body)]);

        updated_pos += dr_space;
        quat<Scalar> updated_orientation = orientation*local_orientation;

        // this runs before the ForceComputes,
        // wrap into box, allowing rigid bodies to span multiple images
        int3 imgi = box.getImage(vec_to_scalar3(updated_pos));
        int3 negimgi = make_int3(-imgi.x,-imgi.y,-imgi.z);
        updated_pos = global_box.shift(updated_pos, negimgi);

        h_postype.data[iptl] = make_scalar4(updated_pos.x, updated_pos.y, updated_pos.z, h_postype.data[iptl].w);
        h_orientation.data[iptl] = quat_to_scalar4(updated_orientation);
        h_image.data[iptl] = img+imgi;
        }
    }

void export_ForceComposite(py::module& m)
    {
    py::class_< ForceComposite, std::shared_ptr<ForceComposite> >(m, "ForceComposite", py::base<MolecularForceCompute>())
        .def(py::init< std::shared_ptr<SystemDefinition> >())
        .def("setParam", &ForceComposite::setParam)
        .def("validateRigidBodies", &ForceComposite::validateRigidBodies)
    ;
    }
