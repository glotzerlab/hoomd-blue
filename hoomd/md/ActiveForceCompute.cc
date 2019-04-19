// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander


#include "ActiveForceCompute.h"
#include "hoomd/RandomNumbers.h"
#include "hoomd/RNGIdentifiers.h"

#include <vector>

using namespace std;
using namespace hoomd;
namespace py = pybind11;

/*! \file ActiveForceCompute.cc
    \brief Contains code for the ActiveForceCompute class
*/

/*! \param seed required user-specified seed number for random number generator.
    \param f_lst An array of (x,y,z) tuples for the active force vector for each particle.
    \param t_lst An array of (xyz) tuples for the active torque vector for each particle
    \param orientation_link if True then forces and torques are applied in the particle's reference frame. If false, then the box reference fra    me is used. Only relevant for non-point-like anisotropic particles.
    /param orientation_reverse_link When True, the particle's orientation is set to match the active force vector. Useful for
    for using a particle's orientation to log the active force vector. Not recommended for anisotropic particles
    \param rotation_diff rotational diffusion constant for all particles.
    \param constraint specifies a constraint surface, to which particles are confined,
    such as update.constraint_ellipsoid.
*/
ActiveForceCompute::ActiveForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                                        std::shared_ptr<ParticleGroup> group,
                                        int seed,
                                        py::list f_lst,
                                        py::list t_lst,
                                        bool orientation_link,
                                        bool orientation_reverse_link,
                                        Scalar rotation_diff,
                                        Scalar3 P,
                                        Scalar rx,
                                        Scalar ry,
                                        Scalar rz)
        : ForceCompute(sysdef), m_group(group), m_orientationLink(orientation_link), m_orientationReverseLink(orientation_reverse_link),
            m_rotationDiff(rotation_diff), m_P(P), m_rx(rx), m_ry(ry), m_rz(rz)
    {

    unsigned int group_size = m_group->getNumMembersGlobal();
    if (group_size == 0)
        {
        m_exec_conf->msg->error() << "Creating a ActiveForceCompute with an empty group" << endl;
        throw std::runtime_error("Error initializing ActiveForceCompute");
        }

    vector<Scalar3> c_f_lst;
    py::tuple tmp_force;
    for (unsigned int i = 0; i < len(f_lst); i++)
        {
        tmp_force = py::cast<py::tuple>(f_lst[i]);
        if (len(tmp_force) !=3)
            throw runtime_error("Non-3D force given for ActiveForceCompute");
        c_f_lst.push_back( make_scalar3(py::cast<Scalar>(tmp_force[0]), py::cast<Scalar>(tmp_force[1]), py::cast<Scalar>(tmp_force[2])));
        }

    vector<Scalar3> c_t_lst;
    py::tuple tmp_torque;
    for (unsigned int i = 0; i < len(t_lst); i++)
        {
        tmp_torque = py::cast<py::tuple>(t_lst[i]);
        if (len(tmp_torque) !=3)
            throw runtime_error("Non-3D torque given for ActiveForceCompute");
        c_t_lst.push_back( make_scalar3(py::cast<Scalar>(tmp_torque[0]), py::cast<Scalar>(tmp_torque[1]), py::cast<Scalar>(tmp_torque[2])));
        }


    if (c_f_lst.size() != group_size) { throw runtime_error("Force given for ActiveForceCompute doesn't match particle number."); }
    if (c_t_lst.size() != group_size) { throw runtime_error("Torque given for ActiveForceCompute doesn't match particle number."); }
    if (m_orientationLink == true && m_rotationDiff != 0)
        {
        throw runtime_error("Non-spherical particles and rotational diffusion is ill defined. Instead implement rotational diffusion through the integrator, or if you are working with point particles set orientation_link=False.");
        }

    GPUArray<Scalar3> tmp_f_activeVec(group_size, m_exec_conf);
    GPUArray<Scalar> tmp_f_activeMag(group_size, m_exec_conf);

    GPUArray<Scalar3> tmp_t_activeVec(group_size, m_exec_conf);
    GPUArray<Scalar> tmp_t_activeMag(group_size, m_exec_conf);


    m_f_activeVec.swap(tmp_f_activeVec);
    m_f_activeMag.swap(tmp_f_activeMag);

    m_t_activeVec.swap(tmp_t_activeVec);
    m_t_activeMag.swap(tmp_t_activeMag);

    ArrayHandle<Scalar3> h_f_activeVec(m_f_activeVec, access_location::host);
    ArrayHandle<Scalar> h_f_activeMag(m_f_activeMag, access_location::host);

    ArrayHandle<Scalar3> h_t_activeVec(m_t_activeVec, access_location::host);
    ArrayHandle<Scalar> h_t_activeMag(m_t_activeMag, access_location::host);


    // for each of the particles in the group
    for (unsigned int i = 0; i < group_size; i++)
        {
        h_f_activeMag.data[i] = slow::sqrt(c_f_lst[i].x*c_f_lst[i].x + c_f_lst[i].y*c_f_lst[i].y + c_f_lst[i].z*c_f_lst[i].z);
        h_t_activeMag.data[i] = slow::sqrt(c_t_lst[i].x*c_t_lst[i].x + c_t_lst[i].y*c_t_lst[i].y + c_t_lst[i].z*c_t_lst[i].z);
        if(h_f_activeMag.data[i] == 0.0) // fixes divide by 0 case if magnitude of active force vector is 0
            {
            h_f_activeMag.data[i] = 0.000000000001;
            }
        if(h_t_activeMag.data[i] == 0.0) // fixes divide by 0 case if magnitude of active torque vector is 0
            {
            h_t_activeMag.data[i] = 0.000000000001;
            }

        h_f_activeVec.data[i] = make_scalar3(0, 0, 0);
        h_f_activeVec.data[i].x = c_f_lst[i].x / h_f_activeMag.data[i];
        h_f_activeVec.data[i].y = c_f_lst[i].y / h_f_activeMag.data[i];
        h_f_activeVec.data[i].z = c_f_lst[i].z / h_f_activeMag.data[i];

        h_t_activeVec.data[i] = make_scalar3(0, 0, 0);
        h_t_activeVec.data[i].x = c_t_lst[i].x / h_t_activeMag.data[i];
        h_t_activeVec.data[i].y = c_t_lst[i].y / h_t_activeMag.data[i];
        h_t_activeVec.data[i].z = c_t_lst[i].z / h_t_activeMag.data[i];

        }

    last_computed = 10;

    // Hash the User's Seed to make it less likely to be a low positive integer
    m_seed = seed*0x12345677 + 0x12345; seed^=(seed>>16); seed*= 0x45679;

    // broadcast the seed from rank 0 to all other ranks.
    #ifdef ENABLE_MPI
        if(this->m_pdata->getDomainDecomposition())
            bcast(m_seed, 0, this->m_exec_conf->getMPICommunicator());
    #endif
    }

ActiveForceCompute::~ActiveForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying ActiveForceCompute" << endl;
    }

/*! This function sets appropriate active forces on all active particles.
*/
void ActiveForceCompute::setForces()
    {

    //  array handles
    ArrayHandle<Scalar3> h_f_actVec(m_f_activeVec, access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_t_actVec(m_t_activeVec, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_f_actMag(m_f_activeMag, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_t_actMag(m_t_activeMag, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_force(m_force,access_location::host,access_mode::overwrite);
    ArrayHandle<Scalar4> h_torque(m_torque,access_location::host,access_mode::overwrite);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    // sanity check
    assert(h_f_actVec.data != NULL);
    assert(h_t_actVec.data != NULL);
    assert(h_f_actMag.data != NULL);
    assert(h_t_actMag.data != NULL);
    assert(h_orientation.data != NULL);

    // zero forces so we don't leave any forces set for indices that are no longer part of our group
    memset(h_force.data, 0, sizeof(Scalar4) * m_force.getNumElements());
    memset(h_torque.data, 0, sizeof(Scalar4) * m_force.getNumElements());

    for (unsigned int i = 0; i < m_group->getNumMembers(); i++)
        {
        unsigned int tag = m_group->getMemberTag(i);
        unsigned int idx = h_rtag.data[tag];

        Scalar3 f;
        Scalar3 t;
        // rotate force according to particle orientation only if orientation is linked to active force vector
        if (m_orientationLink == true)
            {
            vec3<Scalar> fi;
            f = make_scalar3(h_f_actMag.data[i]*h_f_actVec.data[i].x, h_f_actMag.data[i]*h_f_actVec.data[i].y, h_f_actMag.data[i]*h_f_actVec.data[i].z);
            quat<Scalar> quati(h_orientation.data[idx]);
            fi = rotate(quati, vec3<Scalar>(f));
            h_force.data[idx].x = fi.x;
            h_force.data[idx].y = fi.y;
            h_force.data[idx].z = fi.z;

            vec3<Scalar> ti;
            t = make_scalar3(h_t_actMag.data[i]*h_t_actVec.data[i].x, h_t_actMag.data[i]*h_t_actVec.data[i].y, h_t_actMag.data[i]*h_t_actVec.data[i].z);
            ti = rotate(quati, vec3<Scalar>(t));
            h_torque.data[idx].x = ti.x;
            h_torque.data[idx].y = ti.y;
            h_torque.data[idx].z = ti.z;
            }
        else // no orientation link
            {
            f = make_scalar3(h_f_actMag.data[i]*h_f_actVec.data[i].x, h_f_actMag.data[i]*h_f_actVec.data[i].y, h_f_actMag.data[i]*h_f_actVec.data[i].z);
            h_force.data[idx].x = f.x;
            h_force.data[idx].y = f.y;
            h_force.data[idx].z = f.z;

            t = make_scalar3(h_t_actMag.data[i]*h_t_actVec.data[i].x, h_t_actMag.data[i]*h_t_actVec.data[i].y, h_t_actMag.data[i]*h_t_actVec.data[i].z);
            h_torque.data[idx].x = t.x;
            h_torque.data[idx].y = t.y;
            h_torque.data[idx].z = t.z;
            }
        // rotate particle orientation only if orientation is reverse linked to active force vector. Does not operate on torque vector
        if (m_orientationReverseLink == true)
            {
            vec3<Scalar> f(h_f_actMag.data[i]*h_f_actVec.data[i].x, h_f_actMag.data[i]*h_f_actVec.data[i].y, h_f_actMag.data[i]*h_f_actVec.data[i].z);
            vec3<Scalar> vecZ(0.0, 0.0, 1.0);
            vec3<Scalar> quatVec = cross(vecZ, f);
            Scalar quatScal = slow::sqrt(h_f_actMag.data[i]*h_f_actMag.data[i]) + dot(f, vecZ);
            quat<Scalar> quati(quatScal, quatVec);
            quati = quati * (Scalar(1.0) / slow::sqrt(norm2(quati)));
            h_orientation.data[idx] = quat_to_scalar4(quati);
            }
        }
    }

/*! This function applies rotational diffusion to all active particles. The orientation of any torque vector
 * relative to the force vector is preserved
    \param timestep Current timestep
*/
void ActiveForceCompute::rotationalDiffusion(unsigned int timestep)
    {
    //  array handles
    ArrayHandle<Scalar3> h_f_actVec(m_f_activeVec, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> h_t_actVec(m_t_activeVec, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_pos(m_pdata -> getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
    assert(h_pos.data != NULL);

    for (unsigned int i = 0; i < m_group->getNumMembers(); i++)
        {
        unsigned int tag = m_group->getMemberTag(i);
        unsigned int idx = h_rtag.data[tag];
        hoomd::RandomGenerator rng(hoomd::RNGIdentifier::ActiveForceCompute, m_seed, tag, timestep);

        if (m_sysdef->getNDimensions() == 2) // 2D
            {
            Scalar delta_theta; // rotational diffusion angle
            delta_theta = hoomd::NormalDistribution<Scalar>(m_rotationConst)(rng);
            Scalar theta; // angle on plane defining orientation of active force vector
            theta = atan2(h_f_actVec.data[i].y, h_f_actVec.data[i].x);
            theta += delta_theta;
            h_f_actVec.data[i].x = slow::cos(theta);
            h_f_actVec.data[i].y = slow::sin(theta);
            // In 2D, the only meaningful torque vector is out of plane and should not change
            }
        else // 3D: Following Stenhammar, Soft Matter, 2014
            {
            if (m_rx == 0) // if no constraint
                {
                hoomd::SpherePointGenerator<Scalar> unit_vec;
                vec3<Scalar> rand_vec;
                unit_vec(rng, rand_vec);

                vec3<Scalar> aux_vec;
                aux_vec.x = h_f_actVec.data[i].y * rand_vec.z - h_f_actVec.data[i].z * rand_vec.y;
                aux_vec.y = h_f_actVec.data[i].z * rand_vec.x - h_f_actVec.data[i].x * rand_vec.z;
                aux_vec.z = h_f_actVec.data[i].x * rand_vec.y - h_f_actVec.data[i].y * rand_vec.x;
                Scalar aux_vec_mag = slow::sqrt(aux_vec.x*aux_vec.x + aux_vec.y*aux_vec.y + aux_vec.z*aux_vec.z);
                aux_vec.x /= aux_vec_mag;
                aux_vec.y /= aux_vec_mag;
                aux_vec.z /= aux_vec_mag;

                vec3<Scalar> current_f_vec;
                current_f_vec.x = h_f_actVec.data[i].x;
                current_f_vec.y = h_f_actVec.data[i].y;
                current_f_vec.z = h_f_actVec.data[i].z;

                vec3<Scalar> current_t_vec;
                current_t_vec.x = h_t_actVec.data[i].x;
                current_t_vec.y = h_t_actVec.data[i].y;
                current_t_vec.z = h_t_actVec.data[i].z;

                Scalar delta_theta = hoomd::NormalDistribution<Scalar>(m_rotationConst)(rng);
                h_f_actVec.data[i].x = slow::cos(delta_theta)*current_f_vec.x + slow::sin(delta_theta)*aux_vec.x;
                h_f_actVec.data[i].y = slow::cos(delta_theta)*current_f_vec.y + slow::sin(delta_theta)*aux_vec.y;
                h_f_actVec.data[i].z = slow::cos(delta_theta)*current_f_vec.z + slow::sin(delta_theta)*aux_vec.z;

                h_t_actVec.data[i].x = slow::cos(delta_theta)*current_t_vec.x + slow::sin(delta_theta)*aux_vec.x;
                h_t_actVec.data[i].y = slow::cos(delta_theta)*current_t_vec.y + slow::sin(delta_theta)*aux_vec.y;
                h_t_actVec.data[i].z = slow::cos(delta_theta)*current_t_vec.z + slow::sin(delta_theta)*aux_vec.z;

                }
            else // if constraint exists
                {
                EvaluatorConstraintEllipsoid Ellipsoid(m_P, m_rx, m_ry, m_rz);

                Scalar3 current_pos = make_scalar3(h_pos.data[idx].x, h_pos.data[idx].y, h_pos.data[idx].z);
                Scalar3 norm_scalar3 = Ellipsoid.evalNormal(current_pos); // the normal vector to which the particles are confined.

                vec3<Scalar> norm;
                norm = vec3<Scalar> (norm_scalar3);

                vec3<Scalar> current_f_vec;
                current_f_vec.x = h_f_actVec.data[i].x;
                current_f_vec.y = h_f_actVec.data[i].y;
                current_f_vec.z = h_f_actVec.data[i].z;

                vec3<Scalar> current_t_vec;
                current_t_vec.x = h_t_actVec.data[i].x;
                current_t_vec.y = h_t_actVec.data[i].y;
                current_t_vec.z = h_t_actVec.data[i].z;

                vec3<Scalar> aux_vec = cross(current_f_vec, norm); // aux vec for defining direction that active force vector rotates towards. Torque ignored

                Scalar delta_theta; // rotational diffusion angle
                delta_theta = hoomd::NormalDistribution<Scalar>(m_rotationConst)(rng);

                h_f_actVec.data[i].x = slow::cos(delta_theta)*current_f_vec.x + slow::sin(delta_theta)*aux_vec.x;
                h_f_actVec.data[i].y = slow::cos(delta_theta)*current_f_vec.y + slow::sin(delta_theta)*aux_vec.y;
                h_f_actVec.data[i].z = slow::cos(delta_theta)*current_f_vec.z + slow::sin(delta_theta)*aux_vec.z;

                h_t_actVec.data[i].x = slow::cos(delta_theta)*current_t_vec.x + slow::sin(delta_theta)*aux_vec.x;
                h_t_actVec.data[i].y = slow::cos(delta_theta)*current_t_vec.y + slow::sin(delta_theta)*aux_vec.y;
                h_t_actVec.data[i].z = slow::cos(delta_theta)*current_t_vec.z + slow::sin(delta_theta)*aux_vec.z;

                }
            }
        }
    }

/*! This function sets an ellipsoid surface constraint for all active particles. Torque is not considered here
*/
void ActiveForceCompute::setConstraint()
    {
    EvaluatorConstraintEllipsoid Ellipsoid(m_P, m_rx, m_ry, m_rz);

    //  array handles
    ArrayHandle<Scalar3> h_f_actVec(m_f_activeVec, access_location::host, access_mode::readwrite);
    ArrayHandle <Scalar4> h_pos(m_pdata -> getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
    assert(h_pos.data != NULL);

    for (unsigned int i = 0; i < m_group->getNumMembers(); i++)
        {
        unsigned int tag = m_group->getMemberTag(i);
        unsigned int idx = h_rtag.data[tag];

        Scalar3 current_pos = make_scalar3(h_pos.data[idx].x, h_pos.data[idx].y, h_pos.data[idx].z);

        Scalar3 norm_scalar3 = Ellipsoid.evalNormal(current_pos); // the normal vector to which the particles are confined.
        vec3<Scalar> norm;
        norm = vec3<Scalar>(norm_scalar3);
        Scalar dot_prod = h_f_actVec.data[i].x * norm.x + h_f_actVec.data[i].y * norm.y + h_f_actVec.data[i].z * norm.z;

        h_f_actVec.data[i].x -= norm.x * dot_prod;
        h_f_actVec.data[i].y -= norm.y * dot_prod;
        h_f_actVec.data[i].z -= norm.z * dot_prod;

        Scalar new_norm = slow::sqrt(h_f_actVec.data[i].x*h_f_actVec.data[i].x
                                     + h_f_actVec.data[i].y*h_f_actVec.data[i].y
                                     + h_f_actVec.data[i].z*h_f_actVec.data[i].z);

        h_f_actVec.data[i].x /= new_norm;
        h_f_actVec.data[i].y /= new_norm;
        h_f_actVec.data[i].z /= new_norm;
        }
    }

/*! This function applies constraints, rotational diffusion, and sets forces for all active particles
    \param timestep Current timestep
*/
void ActiveForceCompute::computeForces(unsigned int timestep)
    {
    if (m_prof) m_prof->push(m_exec_conf, "ActiveForceCompute");

    if (last_computed != timestep)
        {
        m_rotationConst = slow::sqrt(2.0 * m_rotationDiff * m_deltaT);

        last_computed = timestep;

        if (m_rx != 0)
            {
            setConstraint(); // apply surface constraints to active particles active force vectors
            }
        if (m_rotationDiff != 0)
            {
            rotationalDiffusion(timestep); // apply rotational diffusion to active particles
            }
        setForces(); // set forces for particles
        }

    #ifdef ENABLE_CUDA
    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    #endif

    if (m_prof)
        m_prof->pop(m_exec_conf);
    }


void export_ActiveForceCompute(py::module& m)
    {
    py::class_< ActiveForceCompute, std::shared_ptr<ActiveForceCompute> >(m, "ActiveForceCompute", py::base<ForceCompute>())
    .def(py::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>, int, py::list, py::list,  bool, bool, Scalar,
                    Scalar3, Scalar, Scalar, Scalar >())
    ;
    }
