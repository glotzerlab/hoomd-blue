// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ActiveForceCompute.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"

#include <vector>

namespace hoomd
    {
namespace md
    {
/*! \file ActiveForceCompute.cc
    \brief Contains code for the ActiveForceCompute class
*/

/*! \param rotation_diff rotational diffusion constant for all particles.
 */
ActiveForceCompute::ActiveForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                                       std::shared_ptr<ParticleGroup> group)

    : ForceCompute(sysdef), m_group(group)
    {
    // allocate memory for the per-type active_force storage and initialize them to (1.0,0,0)
    GlobalVector<Scalar4> tmp_f_activeVec(m_pdata->getNTypes(), m_exec_conf);

    m_f_activeVec.swap(tmp_f_activeVec);
    TAG_ALLOCATION(m_f_activeVec);

    ArrayHandle<Scalar4> h_f_activeVec(m_f_activeVec,
                                       access_location::host,
                                       access_mode::overwrite);
    for (unsigned int i = 0; i < m_f_activeVec.size(); i++)
        h_f_activeVec.data[i] = make_scalar4(1.0, 0.0, 0.0, 1.0);

    // allocate memory for the per-type active_force storage and initialize them to (0,0,0)
    GlobalVector<Scalar4> tmp_t_activeVec(m_pdata->getNTypes(), m_exec_conf);

    m_t_activeVec.swap(tmp_t_activeVec);
    TAG_ALLOCATION(m_t_activeVec);

    ArrayHandle<Scalar4> h_t_activeVec(m_t_activeVec,
                                       access_location::host,
                                       access_mode::overwrite);
    for (unsigned int i = 0; i < m_t_activeVec.size(); i++)
        h_t_activeVec.data[i] = make_scalar4(1.0, 0.0, 0.0, 0.0);

#if defined(ENABLE_HIP) && defined(__HIP_PLATFORM_NVCC__)
    if (m_exec_conf->isCUDAEnabled() && m_exec_conf->allConcurrentManagedAccess())
        {
        cudaMemAdvise(m_f_activeVec.get(),
                      sizeof(Scalar4) * m_f_activeVec.getNumElements(),
                      cudaMemAdviseSetReadMostly,
                      0);

        cudaMemAdvise(m_t_activeVec.get(),
                      sizeof(Scalar4) * m_t_activeVec.getNumElements(),
                      cudaMemAdviseSetReadMostly,
                      0);
        }
#endif
    }

ActiveForceCompute::~ActiveForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying ActiveForceCompute" << std::endl;
    }

void ActiveForceCompute::setActiveForce(const std::string& type_name, pybind11::tuple v)
    {
    unsigned int typ = this->m_pdata->getTypeByName(type_name);

    if (pybind11::len(v) != 3)
        {
        throw std::invalid_argument("gamma_r values must be 3-tuples");
        }

    // check for user errors
    if (typ >= m_pdata->getNTypes())
        {
        throw std::invalid_argument("Type does not exist");
        }

    Scalar4 f_activeVec;
    f_activeVec.x = pybind11::cast<Scalar>(v[0]);
    f_activeVec.y = pybind11::cast<Scalar>(v[1]);
    f_activeVec.z = pybind11::cast<Scalar>(v[2]);

    Scalar f_activeMag = slow::sqrt(f_activeVec.x * f_activeVec.x + f_activeVec.y * f_activeVec.y
                                    + f_activeVec.z * f_activeVec.z);

    if (f_activeMag > 0)
        {
        f_activeVec.x /= f_activeMag;
        f_activeVec.y /= f_activeMag;
        f_activeVec.z /= f_activeMag;
        f_activeVec.w = f_activeMag;
        }
    else
        {
        f_activeVec.x = 1;
        f_activeVec.y = 0;
        f_activeVec.z = 0;
        f_activeVec.w = 0;
        }

    ArrayHandle<Scalar4> h_f_activeVec(m_f_activeVec,
                                       access_location::host,
                                       access_mode::readwrite);
    h_f_activeVec.data[typ] = f_activeVec;
    }

pybind11::tuple ActiveForceCompute::getActiveForce(const std::string& type_name)
    {
    pybind11::list v;
    unsigned int typ = this->m_pdata->getTypeByName(type_name);

    ArrayHandle<Scalar4> h_f_activeVec(m_f_activeVec, access_location::host, access_mode::read);

    Scalar4 f_activeVec = h_f_activeVec.data[typ];
    v.append(f_activeVec.w * f_activeVec.x);
    v.append(f_activeVec.w * f_activeVec.y);
    v.append(f_activeVec.w * f_activeVec.z);
    return pybind11::tuple(v);
    }

void ActiveForceCompute::setActiveTorque(const std::string& type_name, pybind11::tuple v)
    {
    unsigned int typ = this->m_pdata->getTypeByName(type_name);

    if (pybind11::len(v) != 3)
        {
        throw std::invalid_argument("gamma_r values must be 3-tuples");
        }

    // check for user errors
    if (typ >= m_pdata->getNTypes())
        {
        throw std::invalid_argument("Type does not exist");
        }

    Scalar4 t_activeVec;
    t_activeVec.x = pybind11::cast<Scalar>(v[0]);
    t_activeVec.y = pybind11::cast<Scalar>(v[1]);
    t_activeVec.z = pybind11::cast<Scalar>(v[2]);

    Scalar t_activeMag = slow::sqrt(t_activeVec.x * t_activeVec.x + t_activeVec.y * t_activeVec.y
                                    + t_activeVec.z * t_activeVec.z);

    if (t_activeMag > 0)
        {
        t_activeVec.x /= t_activeMag;
        t_activeVec.y /= t_activeMag;
        t_activeVec.z /= t_activeMag;
        t_activeVec.w = t_activeMag;
        }
    else
        {
        t_activeVec.x = 0;
        t_activeVec.y = 0;
        t_activeVec.z = 0;
        t_activeVec.w = 0;
        }

    ArrayHandle<Scalar4> h_t_activeVec(m_t_activeVec,
                                       access_location::host,
                                       access_mode::readwrite);
    h_t_activeVec.data[typ] = t_activeVec;
    }

pybind11::tuple ActiveForceCompute::getActiveTorque(const std::string& type_name)
    {
    pybind11::list v;
    unsigned int typ = this->m_pdata->getTypeByName(type_name);

    ArrayHandle<Scalar4> h_t_activeVec(m_t_activeVec, access_location::host, access_mode::read);
    Scalar4 t_activeVec = h_t_activeVec.data[typ];
    v.append(t_activeVec.w * t_activeVec.x);
    v.append(t_activeVec.w * t_activeVec.y);
    v.append(t_activeVec.w * t_activeVec.z);
    return pybind11::tuple(v);
    }

/*! This function sets appropriate active forces on all active particles.
 */
void ActiveForceCompute::setForces()
    {
    //  array handles
    ArrayHandle<Scalar4> h_f_actVec(m_f_activeVec, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_t_actVec(m_t_activeVec, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar4> h_torque(m_torque, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                       access_location::host,
                                       access_mode::read);

    // sanity check
    assert(h_f_actVec.data != NULL);
    assert(h_t_actVec.data != NULL);
    assert(h_orientation.data != NULL);
    assert(h_pos.data != NULL);

    // zero forces so we don't leave any forces set for indices that are no longer part of our group
    memset(h_force.data, 0, sizeof(Scalar4) * m_force.getNumElements());
    memset(h_torque.data, 0, sizeof(Scalar4) * m_force.getNumElements());

    for (unsigned int i = 0; i < m_group->getNumMembers(); i++)
        {
        unsigned int idx = m_group->getMemberIndex(i);
        unsigned int type = __scalar_as_int(h_pos.data[idx].w);

        vec3<Scalar> f(h_f_actVec.data[type].w * h_f_actVec.data[type].x,
                       h_f_actVec.data[type].w * h_f_actVec.data[type].y,
                       h_f_actVec.data[type].w * h_f_actVec.data[type].z);
        quat<Scalar> quati(h_orientation.data[idx]);
        vec3<Scalar> fi = rotate(quati, f);
        h_force.data[idx] = vec_to_scalar4(fi, 0);

        vec3<Scalar> t(h_t_actVec.data[type].w * h_t_actVec.data[type].x,
                       h_t_actVec.data[type].w * h_t_actVec.data[type].y,
                       h_t_actVec.data[type].w * h_t_actVec.data[type].z);
        vec3<Scalar> ti = rotate(quati, t);
        h_torque.data[idx] = vec_to_scalar4(ti, 0);
        }
    }

/*! This function applies rotational diffusion to the orientations of all active particles. The
 orientation of any torque vector
 * relative to the force vector is preserved
    \param timestep Current timestep
*/
void ActiveForceCompute::rotationalDiffusion(Scalar rotational_diffusion, uint64_t timestep)
    {
    //  array handles
    ArrayHandle<Scalar4> h_f_actVec(m_f_activeVec, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                       access_location::host,
                                       access_mode::readwrite);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

    assert(h_f_actVec.data != NULL);
    assert(h_pos.data != NULL);
    assert(h_orientation.data != NULL);
    assert(h_tag.data != NULL);

    const auto rotation_constant = slow::sqrt(2.0 * rotational_diffusion * m_deltaT);
    for (unsigned int i = 0; i < m_group->getNumMembers(); i++)
        {
        unsigned int idx = m_group->getMemberIndex(i);
        unsigned int type = __scalar_as_int(h_pos.data[idx].w);

        if (h_f_actVec.data[type].w != 0)
            {
            unsigned int ptag = h_tag.data[idx];
            hoomd::RandomGenerator rng(hoomd::Seed(hoomd::RNGIdentifier::ActiveForceCompute,
                                                   timestep,
                                                   m_sysdef->getSeed()),
                                       hoomd::Counter(ptag));

            quat<Scalar> quati(h_orientation.data[idx]);

            if (m_sysdef->getNDimensions() == 2) // 2D
                {
                Scalar delta_theta = hoomd::NormalDistribution<Scalar>(rotation_constant)(rng);

                vec3<Scalar> b(0, 0, 1.0);
                quat<Scalar> rot_quat = quat<Scalar>::fromAxisAngle(b, delta_theta);

                quati = rot_quat * quati; // rotational diffusion quaternion applied to orientation
                quati = quati * (Scalar(1.0) / slow::sqrt(norm2(quati)));
                h_orientation.data[idx] = quat_to_scalar4(quati);
                // In 2D, the only meaningful torque vector is out of plane and should not change
                }
            else // 3D: Following Stenhammar, Soft Matter, 2014
                {
                hoomd::SpherePointGenerator<Scalar> unit_vec;
                vec3<Scalar> rand_vec;
                unit_vec(rng, rand_vec);

                vec3<Scalar> f(h_f_actVec.data[type].x,
                               h_f_actVec.data[type].y,
                               h_f_actVec.data[type].z);
                vec3<Scalar> fi
                    = rotate(quati, f); // rotate active force vector from local to global frame

                vec3<Scalar> aux_vec = cross(fi, rand_vec); // rotation axis
                Scalar aux_vec_mag = slow::rsqrt(dot(aux_vec, aux_vec));
                aux_vec *= aux_vec_mag;

                Scalar delta_theta = hoomd::NormalDistribution<Scalar>(rotation_constant)(rng);
                quat<Scalar> rot_quat = quat<Scalar>::fromAxisAngle(aux_vec, delta_theta);

                quati = rot_quat * quati; // rotational diffusion quaternion applied to orientation
                quati = quati * (Scalar(1.0) / slow::sqrt(norm2(quati)));
                h_orientation.data[idx] = quat_to_scalar4(quati);
                }
            }
        }
    }

/*! This function applies rotational diffusion and sets forces for all active particles
    \param timestep Current timestep
*/
void ActiveForceCompute::computeForces(uint64_t timestep)
    {
    setForces(); // set forces for particles

#ifdef ENABLE_HIP
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
#endif
    }

namespace detail
    {
void export_ActiveForceCompute(pybind11::module& m)
    {
    pybind11::class_<ActiveForceCompute, ForceCompute, std::shared_ptr<ActiveForceCompute>>(
        m,
        "ActiveForceCompute")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>>())
        .def("setActiveForce", &ActiveForceCompute::setActiveForce)
        .def("getActiveForce", &ActiveForceCompute::getActiveForce)
        .def("setActiveTorque", &ActiveForceCompute::setActiveTorque)
        .def("getActiveTorque", &ActiveForceCompute::getActiveTorque)
        .def_property_readonly("filter",
                               [](ActiveForceCompute& force)
                               { return force.getGroup()->getFilter(); });
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
