/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2015 The Regents of
the University of Michigan All rights reserved.

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

// Maintainer: jglaser

#include "ForceComposite.h"
#include "VectorMath.h"

#include <string.h>

#include <boost/python.hpp>

/*! \file ForceComposite.cc
    \brief Contains code for the ForceComposite class
*/

/*! \param sysdef SystemDefinition containing the ParticleData to compute forces on
*/
ForceComposite::ForceComposite(boost::shared_ptr<SystemDefinition> sysdef)
        : MolecularForceCompute(sysdef), m_bodies_changed(false), m_ptls_added_removed(false)
    {
    // connect to the ParticleData to receive notifications when the number of types changes
    m_num_type_change_connection = m_pdata->connectNumTypesChange(boost::bind(&ForceComposite::slotNumTypesChange, this));

    m_global_ptl_num_change_connection = m_pdata->connectGlobalParticleNumberChange(boost::bind(&ForceComposite::slotPtlsAddedRemoved, this));

    GPUArray<unsigned int> body_types(m_pdata->getNTypes(), 1, m_exec_conf);
    m_body_types.swap(body_types);

    GPUArray<Scalar3> body_pos(m_pdata->getNTypes(), 1, m_exec_conf);
    m_body_pos.swap(body_pos);

    GPUArray<Scalar4> body_orientation(m_pdata->getNTypes(), 1, m_exec_conf);
    m_body_orientation.swap(body_orientation);

    GPUArray<unsigned int> body_len(m_pdata->getNTypes(), m_exec_conf);
    m_body_len.swap(body_len);

    m_d_max_changed.resize(m_pdata->getNTypes(), false);
    }

//! Destructor
ForceComposite::~ForceComposite()
    {
    // disconnect from signal in ParticleData;
    m_num_type_change_connection.disconnect();

    m_global_ptl_num_change_connection.disconnect();

    if (m_comm_ghost_layer_connection.connected())
        m_comm_ghost_layer_connection.disconnect();
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
        m_exec_conf->msg->error() << "constrain.rigid(): Invalid rigid body type." << std::endl;
        throw std::runtime_error("Error initializing ForceComposite");
        }

    if (type.size() != pos.size() || orientation.size() != pos.size())
        {
        m_exec_conf->msg->error() << "constrain.rigid(): Constituent particle lists"
            <<" (position, orientation, type) are of unequal length." << std::endl;
        throw std::runtime_error("Error initializing ForceComposite");
        }

    bool body_updated = false;

    bool body_len_changed = false;

    // detect if bodies have changed

    ArrayHandle<unsigned int> h_body_len(m_body_len, access_location::host, access_mode::readwrite);

        {
        ArrayHandle<unsigned int> h_body_type(m_body_types, access_location::host, access_mode::read);
        ArrayHandle<Scalar3> h_body_pos(m_body_pos, access_location::host, access_mode::read);
        ArrayHandle<Scalar4> h_body_orientation(m_body_orientation, access_location::host, access_mode::read);

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
            }
        }

    if (body_updated)
        {
        ArrayHandle<unsigned int> h_body_type(m_body_types, access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar3> h_body_pos(m_body_pos, access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> h_body_orientation(m_body_orientation, access_location::host, access_mode::readwrite);

        // store body data in GPUArray
        for (unsigned int i = 0; i < type.size(); ++i)
            {
            h_body_type.data[m_body_idx(body_typeid,i)] = type[i];
            h_body_pos.data[m_body_idx(body_typeid,i)] = pos[i];
            h_body_orientation.data[m_body_idx(body_typeid,i)] = orientation[i];
            }

        m_bodies_changed = true;
        assert(m_d_max_changed.size() > body_typeid);
        m_d_max_changed[body_typeid] = true;

        // also update diameter on constituent particles
        for (unsigned int i = 0; i < type.size(); ++i)
            {
            m_d_max_changed[type[i]] = true;
            }
        }
   }

void ForceComposite::slotNumTypesChange()
    {
    unsigned int old_ntypes = m_body_len.getNumElements();
    unsigned int new_ntypes = m_pdata->getNTypes();

    unsigned int height = m_body_pos.getHeight();

    // resize per-type arrays (2D)
    m_body_types.resize(new_ntypes, height);
    m_body_pos.resize(new_ntypes, height);
    m_body_orientation.resize(new_ntypes, height);

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
    }

Scalar ForceComposite::requestGhostLayerWidth(unsigned int type)
    {
    // the default ghost layer is there to ensure that constituent particles are always
    // communicated for every central particle
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
            bool is_part_of_body = (body_type == type);
            for (unsigned int i = 0; i < h_body_len.data[body_type]; ++i)
                {
                if (h_body_type.data[i] == type)
                    {
                    is_part_of_body = true;
                    break;
                    }
                }

            if (! is_part_of_body) continue;

            // compute maximum distance to central particle
            for (unsigned int i = 0; i < h_body_len.data[body_type]; ++i)
                {
                Scalar3 r = h_body_pos.data[m_body_idx(body_type,i)];
                Scalar d = sqrt(dot(r,r));

                if (d > m_d_max[type])
                    {
                    m_d_max[type] = d;
                    }
                }
            }

        m_exec_conf->msg->notice(7) << "ForceComposite: requesting ghost layer for type "
            << m_pdata->getNameByType(type) << ": " << m_d_max[type] << std::endl;

        m_d_max_changed[type] = false;
        }

    return m_d_max[type];
    }

void ForceComposite::createRigidBodies()
    {
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
                        throw std::runtime_error("Error intializing ForceComposite");
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

            // count number of constituent particls to add
            for (unsigned i = 0; i < snap.size; ++i)
                {
                assert(snap.type[i] < ntypes);

                bool is_central_ptl = h_body_len.data[snap.type[i]] != 0;

                if (snap.body[i] != NO_BODY)
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

        if (m_exec_conf->getRank() == 0)
            {
            unsigned int old_size = snap.size;

            // resize and reset global molecule table
            molecule_tag.resize(old_size+n_add_ptls, NO_MOLECULE);

            // acces body data
            ArrayHandle<unsigned int> h_body_type(m_body_types, access_location::host, access_mode::read);
            ArrayHandle<Scalar3> h_body_pos(m_body_pos, access_location::host, access_mode::read);
            ArrayHandle<Scalar4> h_body_orientation(m_body_orientation, access_location::host, access_mode::read);
            ArrayHandle<unsigned int> h_body_len(m_body_len, access_location::host, access_mode::read);

            unsigned int snap_idx_out = 0;

            // create copies
            for (unsigned i = 0; i < old_size; ++i)
                {
                assert(snap.type[i] < ntypes);

                bool is_central_ptl = h_body_len.data[snap.type[i]] != 0;

                assert(snap.body[i] == NO_BODY);

                if (is_central_ptl)
                    {
                    unsigned int body_type = snap.type[i];

                    unsigned body_tag = snap_idx_out;

                    // set body id to tag of central ptl
                    snap_out.body[snap_idx_out] = body_tag;

                    // set contiguous molecule tag
                    molecule_tag[snap_idx_out] = nbodies;

                    vec3<Scalar> central_pos(snap.pos[i]);
                    quat<Scalar> central_orientation(snap.orientation[i]);
                    int3 central_img = snap.image[i];

                    snap_idx_out++;

                    // insert elements into snapshot
                    unsigned int n= h_body_len.data[body_type];
                    snap_out.insert(snap_idx_out, n);

                    for (unsigned int j = 0; j < n; ++j)
                        {
                        // position and orientation will be overwritten during integration, no need to set here

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

                        // wrap into box
                        int3 img = central_img;
                        global_box.wrap(pos, img);

                        snap_out.pos[snap_idx_out] = pos;
                        snap_out.image[snap_idx_out] = img;
                        snap_out.orientation[snap_idx_out] = orientation;

                        snap_idx_out++;
                        }

                    nbodies++;
                    }

                }
            }

        m_exec_conf->msg->notice(2) << "constrain.rigid(): Creating " << nbodies << " rigid bodies (adding "
            << n_add_ptls << " particles)" << std::endl;

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
            // store global molecule information in GPUArray
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

    // request communication of particle forces
    flags[comm_flag::net_force] = 1;

    // request communication of particle torques
    flags[comm_flag::net_torque] = 1;

    // only communicate net virial if needed
    PDataFlags pdata_flags = this->m_pdata->getFlags();
    if (pdata_flags[pdata_flag::isotropic_virial] || pdata_flags[pdata_flag::pressure_tensor])
        {
        flags[comm_flag::net_virial] = 1;
        }

    // request body ids
    flags[comm_flag::body] = 1;

    flags |= MolecularForceCompute::getRequestedCommFlags(timestep);

    return flags;
    }
#endif

//! Compute the forces and torques on the central particle
void ForceComposite::computeForces(unsigned int timestep)
    {
    // at this point, all constituent particles of a local rigid body (i.e. one for which the central particle
    // is local) need to be present to correctly compute forces
    if (m_particles_sorted)
        {
        // initialize molecule table
        initMolecules();
        }

    // access particle data
    ArrayHandle<unsigned int> h_body(m_pdata->getBodies(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);

    // access net force and torque acting on constituent particles
    ArrayHandle<Scalar4> h_net_force(m_pdata->getNetForce(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_net_torque(m_pdata->getNetTorqueArray(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_net_virial(m_pdata->getNetVirial(), access_location::host, access_mode::read);

    // access the force and torque array for the central ptl
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar4> h_torque(m_torque, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);

    // for each local body
    unsigned int nmol = m_molecule_indexer.getW();

    ArrayHandle<unsigned int> h_molecule_length(m_molecule_length, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_molecule_list(m_molecule_list, access_location::host, access_mode::read);

    ArrayHandle<Scalar3> h_body_pos(m_body_pos, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_body_len(m_body_len, access_location::host, access_mode::read);

    // reset constraint forces and torques
    memset(h_force.data,0, sizeof(Scalar4)*m_pdata->getN());
    memset(h_torque.data,0, sizeof(Scalar4)*m_pdata->getN());
    memset(h_virial.data,0, sizeof(Scalar)*m_virial.getNumElements());

    unsigned int nptl_local = m_pdata->getN();
    unsigned int net_virial_pitch = m_pdata->getNetVirial().getPitch();

    PDataFlags flags = m_pdata->getFlags();
    bool compute_virial = false;
    if (flags[pdata_flag::isotropic_virial] || flags[pdata_flag::pressure_tensor])
        {
        compute_virial = true;
        }

    for (unsigned int ibody = 0; ibody < nmol; ibody++)
        {
        unsigned int len = h_molecule_length.data[ibody];

        // get central ptl tag from first ptl in molecule
        assert(len>0);
        unsigned int first_idx = h_molecule_list.data[m_molecule_indexer(ibody,0)];

        assert(first_idx < m_pdata->getN() + m_pdata->getNGhosts());
        unsigned int central_tag = h_body.data[first_idx];

        assert(central_tag <= m_pdata->getMaximumTag());
        unsigned int central_idx = h_rtag.data[central_tag];

        // do not compute force on ghost particles
        if (central_idx >= nptl_local) continue;

        // the central ptl must be present
        assert(central_tag == h_tag.data[first_idx]);

        // central ptl position and orientation
        Scalar4 postype = h_postype.data[central_idx];
        quat<Scalar> orientation(h_orientation.data[central_idx]);

        // body type
        unsigned int type = __scalar_as_int(postype.w);

        if (len != h_body_len.data[type] + 1)
            {
            m_exec_conf->msg->error() << "constrain.rigd(): Composite particle with body tag " << central_tag << " incomplete"
                << std::endl << std::endl;
            throw std::runtime_error("Error computing composite particle forces.\n");
            }

        // sum up forces and torques from constituent particles
        for (unsigned int jptl = 0; jptl < len; ++jptl)
            {
            unsigned int idxj = h_molecule_list.data[m_molecule_indexer(ibody, jptl)];
            assert(idxj < m_pdata->getN() + m_pdata->getNGhosts());

            if (idxj == central_idx) continue;

            // force and torque on particle
            Scalar4 net_force = h_net_force.data[idxj];
            vec3<Scalar> f(net_force);

            // sum up center of mass force
            h_force.data[central_idx].x += f.x;
            h_force.data[central_idx].y += f.y;
            h_force.data[central_idx].z += f.z;

            unsigned int tagj = h_tag.data[idxj];
            assert(tagj <= m_pdata->getMaximumTag());

            // we know that the tag of the constituent ptl relative to the central ptl indicates
            // the position in the rigid body
            vec3<Scalar> dr(h_body_pos.data[m_body_idx(type, tagj - central_tag - 1)]);

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
            Scalar4 net_torque = h_net_torque.data[idxj];
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
        }
    }

/* Set position and velocity of constituent particles in rigid bodies in the 1st or second half of integration on the CPU
    based on the body center of mass and particle relative position in each body frame.
*/

void ForceComposite::updateCompositeParticles(unsigned int timestep, bool remote)
    {
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

    const BoxDim& box = m_pdata->getBox();

    // we need to update both local and ghost particles
    unsigned int nptl = m_pdata->getN() + m_pdata->getNGhosts();

    for (unsigned int iptl = 0; iptl < nptl; iptl++)
        {
        unsigned int central_tag = h_body.data[iptl];

        if (central_tag == NO_BODY)
            continue;

        // body tag equals tag for central ptl
        assert(central_tag <= m_pdata->getMaximumTag());
        unsigned int central_idx = h_rtag.data[central_tag];

        if ((!remote && central_idx >= m_pdata->getN()))
            {
            // only update local composite particles
            continue;
            }

        if (central_idx == NOT_LOCAL && iptl >= m_pdata->getN())
            continue;

        if (central_idx == NOT_LOCAL)
            {
            m_exec_conf->msg->error() << "constrain.rigd(): Missing central particle tag " << central_tag << "!"
                << std::endl << std::endl;
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

        int3 img = h_image.data[central_idx];

        unsigned int tag = h_tag.data[iptl];

        vec3<Scalar> local_pos(h_body_pos.data[m_body_idx(type, tag - central_tag - 1)]);
        vec3<Scalar> dr_space = rotate(orientation, local_pos);

        assert(tag-central_tag >= 1);

        // update position and orientation
        vec3<Scalar> updated_pos(pos);

        // we know that the tag of the constituent ptl relative to the central ptl indicates
        // the position in the rigid body
        quat<Scalar> local_orientation(h_body_orientation.data[m_body_idx(type, tag - central_tag - 1)]);

        updated_pos += dr_space;
        quat<Scalar> updated_orientation = orientation*local_orientation;

        // this runs before the ForceComputes, wrap particle into box
        int3 imgi = img;
        box.wrap(updated_pos, imgi);

        h_postype.data[iptl] = make_scalar4(updated_pos.x, updated_pos.y, updated_pos.z, h_postype.data[iptl].w);
        h_orientation.data[iptl] = quat_to_scalar4(updated_orientation);
        h_image.data[iptl] = imgi;
        }
    }

void export_ForceComposite()
    {
    class_< ForceComposite, boost::shared_ptr<ForceComposite>, bases<MolecularForceCompute>, boost::noncopyable >
    ("ForceComposite", init< boost::shared_ptr<SystemDefinition> >())
        .def("setParam", &ForceComposite::setParam)
        .def("createRigidBodies", &ForceComposite::createRigidBodies)
    ;
    }
