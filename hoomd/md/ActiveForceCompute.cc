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
                                        Scalar rotation_diff,
                                        Scalar3 P,
                                        Scalar rx,
                                        Scalar ry,
                                        Scalar rz)
        : ForceCompute(sysdef), m_group(group),
            m_rotationDiff(rotation_diff), m_P(P), m_rx(rx), m_ry(ry), m_rz(rz)
    {

    // In case of MPI run, every rank should be initialized with the same seed.
    // For simplicity we broadcast the seed of rank 0 to all ranks.

    m_seed = seed*0x12345677 + 0x12345; seed^=(seed>>16); seed*= 0x45679;

    #ifdef ENABLE_MPI
    if( this->m_pdata->getDomainDecomposition() )
        bcast(m_seed,0,this->m_exec_conf->getMPICommunicator());
    #endif


    // allocate memory for the per-type active_force storage and initialize them to (1.0,0,0)
    GlobalVector<Scalar3> tmp_f_activeVec(m_pdata->getNTypes(), m_exec_conf);
    GlobalVector<Scalar> tmp_f_activeMag(m_pdata->getNTypes(), m_exec_conf);

    m_f_activeVec.swap(tmp_f_activeVec);
    m_f_activeMag.swap(tmp_f_activeMag);
    TAG_ALLOCATION(m_f_activeVec);
    TAG_ALLOCATION(m_f_activeMag);

    ArrayHandle<Scalar3> h_f_activeVec(m_f_activeVec, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_f_activeMag(m_f_activeMag, access_location::host, access_mode::overwrite);
    for (unsigned int i = 0; i < m_f_activeMag.size(); i++){
        h_f_activeMag.data[i] = 1;
        h_f_activeVec.data[i] = make_scalar3(1.0,0.0,0.0);
    }

    // allocate memory for the per-type active_force storage and initialize them to (0,0,0)
    GlobalVector<Scalar3> tmp_t_activeVec(m_pdata->getNTypes(), m_exec_conf);
    GlobalVector<Scalar> tmp_t_activeMag(m_pdata->getNTypes(), m_exec_conf);

    m_t_activeVec.swap(tmp_t_activeVec);
    m_t_activeMag.swap(tmp_t_activeMag);
    TAG_ALLOCATION(m_t_activeVec);
    TAG_ALLOCATION(m_t_activeMag);

    ArrayHandle<Scalar3> h_t_activeVec(m_t_activeVec, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_t_activeMag(m_t_activeMag, access_location::host, access_mode::overwrite);
    for (unsigned int i = 0; i < m_t_activeMag.size(); i++){
        h_t_activeMag.data[i] = 0;
        h_t_activeVec.data[i] = make_scalar3(0.0,0.0,0.0);
    }


    #if defined(ENABLE_HIP) && defined(__HIP_PLATFORM_NVCC__)
    if (m_exec_conf->isCUDAEnabled() && m_exec_conf->allConcurrentManagedAccess())
        {
        cudaMemAdvise(m_f_activeVec.get(), sizeof(Scalar3)*m_f_activeVec.getNumElements(), cudaMemAdviseSetReadMostly, 0);
        cudaMemAdvise(m_f_activeMag.get(), sizeof(Scalar)*m_f_activeMag.getNumElements(), cudaMemAdviseSetReadMostly, 0);

        cudaMemAdvise(m_t_activeVec.get(), sizeof(Scalar3)*m_t_activeVec.getNumElements(), cudaMemAdviseSetReadMostly, 0);
        cudaMemAdvise(m_t_activeMag.get(), sizeof(Scalar)*m_t_activeMag.getNumElements(), cudaMemAdviseSetReadMostly, 0);
        }
    #endif

    }

ActiveForceCompute::~ActiveForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying ActiveForceCompute" << endl;
    }



void ActiveForceCompute::setActiveForce(const std::string& type_name, pybind11::tuple v)
    {
    unsigned int typ = this->m_pdata->getTypeByName(type_name);

    if (pybind11::len(v) != 3)
        {
        throw invalid_argument("gamma_r values must be 3-tuples");
        }

    // check for user errors
    if (typ >= m_pdata->getNTypes())
        {
        throw invalid_argument("Type does not exist");
        }

    Scalar3 f_activeVec;
    f_activeVec.x = pybind11::cast<Scalar>(v[0]);
    f_activeVec.y = pybind11::cast<Scalar>(v[1]);
    f_activeVec.z = pybind11::cast<Scalar>(v[2]);

    Scalar f_activeMag = slow::sqrt(f_activeVec.x*f_activeVec.x+f_activeVec.y*f_activeVec.y+f_activeVec.z*f_activeVec.z);

    f_activeVec.x /= f_activeMag;
    f_activeVec.y /= f_activeMag;
    f_activeVec.z /= f_activeMag;

    ArrayHandle<Scalar3> h_f_activeVec(m_f_activeVec, access_location::host, access_mode::readwrite);
    h_f_activeVec.data[typ] = f_activeVec;

    ArrayHandle<Scalar> h_f_activeMag(m_f_activeMag, access_location::host, access_mode::readwrite);
    h_f_activeMag.data[typ] = f_activeMag;

    std::cout << f_activeVec.x << " " << f_activeVec.y << " " << f_activeVec.z << std::endl;
    }

pybind11::tuple ActiveForceCompute::getActiveForce(const std::string& type_name)
    {
    pybind11::list v;
    unsigned int typ = this->m_pdata->getTypeByName(type_name);

    ArrayHandle<Scalar3> h_f_activeVec(m_f_activeVec, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_f_activeMag(m_f_activeMag, access_location::host, access_mode::readwrite);
    Scalar3 f_activeVec = h_f_activeVec.data[typ];
    Scalar f_activeMag = h_f_activeMag.data[typ];
    v.append(f_activeMag*f_activeVec.x);
    v.append(f_activeMag*f_activeVec.y);
    v.append(f_activeMag*f_activeVec.z);
    return pybind11::tuple(v);
    }

void ActiveForceCompute::setActiveTorque(const std::string& type_name, pybind11::tuple v)
    {
    unsigned int typ = this->m_pdata->getTypeByName(type_name);

    if (pybind11::len(v) != 3)
        {
        throw invalid_argument("gamma_r values must be 3-tuples");
        }

    // check for user errors
    if (typ >= m_pdata->getNTypes())
        {
        throw invalid_argument("Type does not exist");
        }

    Scalar3 t_activeVec;
    t_activeVec.x = pybind11::cast<Scalar>(v[0]);
    t_activeVec.y = pybind11::cast<Scalar>(v[1]);
    t_activeVec.z = pybind11::cast<Scalar>(v[2]);

    Scalar t_activeMag = slow::sqrt(t_activeVec.x*t_activeVec.x+t_activeVec.y*t_activeVec.y+t_activeVec.z*t_activeVec.z);

    if(t_activeMag > 0)
        {
        t_activeVec.x /= t_activeMag;
        t_activeVec.y /= t_activeMag;
        t_activeVec.z /= t_activeMag;
        }
    else
       {
        t_activeVec.x = 0;
        t_activeVec.y = 0;
        t_activeVec.z = 0;
        }

    ArrayHandle<Scalar3> h_t_activeVec(m_t_activeVec, access_location::host, access_mode::readwrite);
    h_t_activeVec.data[typ] = t_activeVec;

    ArrayHandle<Scalar> h_t_activeMag(m_t_activeMag, access_location::host, access_mode::readwrite);
    h_t_activeMag.data[typ] = t_activeMag;

    std::cout << t_activeVec.x << " " << t_activeVec.y << " " << t_activeVec.z << std::endl;
    }

pybind11::tuple ActiveForceCompute::getActiveTorque(const std::string& type_name)
    {
    pybind11::list v;
    unsigned int typ = this->m_pdata->getTypeByName(type_name);

    ArrayHandle<Scalar3> h_t_activeVec(m_t_activeVec, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_t_activeMag(m_t_activeMag, access_location::host, access_mode::readwrite);
    Scalar3 t_activeVec = h_t_activeVec.data[typ];
    Scalar t_activeMag = h_t_activeMag.data[typ];
    v.append(t_activeMag*t_activeVec.x);
    v.append(t_activeMag*t_activeVec.y);
    v.append(t_activeMag*t_activeVec.z);
    return pybind11::tuple(v);
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
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    // sanity check
    assert(h_f_actVec.data != NULL);
    assert(h_t_actVec.data != NULL);
    assert(h_f_actMag.data != NULL);
    assert(h_t_actMag.data != NULL);
    assert(h_orientation.data != NULL);
    assert(h_pos.data != NULL);

    // zero forces so we don't leave any forces set for indices that are no longer part of our group
    memset(h_force.data, 0, sizeof(Scalar4) * m_force.getNumElements());
    memset(h_torque.data, 0, sizeof(Scalar4) * m_force.getNumElements());

    for (unsigned int i = 0; i < m_group->getNumMembers(); i++)
        {
        unsigned int tag = m_group->getMemberTag(i);
        unsigned int idx = h_rtag.data[tag];
        unsigned int j = m_group->getMemberIndex(i);
        unsigned int type = __scalar_as_int(h_pos.data[j].w);

        Scalar3 f = make_scalar3(h_f_actMag.data[type]*h_f_actVec.data[type].x, h_f_actMag.data[type]*h_f_actVec.data[type].y, h_f_actMag.data[type]*h_f_actVec.data[type].z);
        quat<Scalar> quati(h_orientation.data[idx]);
        vec3<Scalar> fi = rotate(quati, vec3<Scalar>(f));
        h_force.data[idx].x = fi.x;
        h_force.data[idx].y = fi.y;
        h_force.data[idx].z = fi.z;

        Scalar3 t = make_scalar3(h_t_actMag.data[type]*h_t_actVec.data[type].x, h_t_actMag.data[type]*h_t_actVec.data[type].y, h_t_actMag.data[type]*h_t_actVec.data[type].z);
        vec3<Scalar> ti = rotate(quati, vec3<Scalar>(t));
        h_torque.data[idx].x = ti.x;
        h_torque.data[idx].y = ti.y;
        h_torque.data[idx].z = ti.z;
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
    ArrayHandle<Scalar3> h_f_actVec(m_f_activeVec, access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_t_actVec(m_t_activeVec, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_pos(m_pdata -> getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
    assert(h_pos.data != NULL);

    for (unsigned int i = 0; i < m_group->getNumMembers(); i++)
        {
        unsigned int tag = m_group->getMemberTag(i);
        unsigned int idx = h_rtag.data[tag];
        unsigned int j = m_group->getMemberIndex(i);
        unsigned int type = __scalar_as_int(h_pos.data[j].w);

        Scalar3 current_pos = make_scalar3(h_pos.data[idx].x, h_pos.data[idx].y, h_pos.data[idx].z);

        Scalar3 norm_scalar3 = Ellipsoid.evalNormal(current_pos); // the normal vector to which the particles are confined.
        vec3<Scalar> norm = vec3<Scalar>(norm_scalar3);

        Scalar new_norm = slow::sqrt(norm.x*norm.x + norm.y*norm.y + norm.z*norm.z);

        norm.x /= new_norm;
        norm.y /= new_norm;
        norm.z /= new_norm;

        Scalar3 f = make_scalar3(h_f_actVec.data[type].x, h_f_actVec.data[type].y, h_f_actVec.data[type].z);
        quat<Scalar> quati(h_orientation.data[idx]);
        vec3<Scalar> fi = rotate(quati, vec3<Scalar>(f));


        Scalar dot_prod = fi.x * norm.x + fi.y * norm.y + fi.z * norm.z;

        Scalar dot_perp_prod = slow::sqrt(1-dot_prod*dot_prod);

        Scalar phi_half = slow::atan(dot_perp/dot_perp_prod)/2.0


        fi.x -= norm.x * dot_prod;
        fi.y -= norm.y * dot_prod;
        fi.z -= norm.z * dot_prod;

        new_norm = slow::sqrt(fi.x*fi.x + fi.y*fi.y + fi.z*fi.z);

        fi.x /= new_norm;
        fi.y /= new_norm;
        fi.z /= new_norm;

        vec3<Scalar> rot_vec = cross(norm,fi) 
        rot_vec.x *= slow::sin(phi_half)
        rot_vec.y *= slow::sin(phi_half)
        rot_vec.z *= slow::sin(phi_half)

        quat<Scalar> rot_quat(cos(phi_half),rot_vec);

        quati = rot_quat*quati

        h_orientation.data[idx].x = quati.s;
        h_orientation.data[idx].y = quati.v.x;
        h_orientation.data[idx].z = quati.v.y;
        h_orientation.data[idx].w = quati.v.z;

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

    #ifdef ENABLE_HIP
    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    #endif

    if (m_prof)
        m_prof->pop(m_exec_conf);
    }


void export_ActiveForceCompute(py::module& m)
    {
    py::class_< ActiveForceCompute, ForceCompute, std::shared_ptr<ActiveForceCompute> >(m, "ActiveForceCompute")
    .def(py::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>, int, Scalar,
                    Scalar3, Scalar, Scalar, Scalar >())
    .def("setActiveForce", &ActiveForceCompute::setActiveForce)
    .def("getActiveForce", &ActiveForceCompute::getActiveForce)
    .def("setActiveTorque", &ActiveForceCompute::setActiveTorque)
    .def("getActiveTorque", &ActiveForceCompute::getActiveTorque)
    ;
    }
