// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ConstantForceCompute.h"

#include <vector>

namespace hoomd
    {
namespace md
    {
/*! \file ConstantForceCompute.cc
    \brief Contains code for the ConstantForceCompute class
*/

/*! \param Constant force applied on a group of particles.
 */
ConstantForceCompute::ConstantForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                                           std::shared_ptr<ParticleGroup> group)

    : ForceCompute(sysdef), m_group(group), m_parameters_updated(false)
    {
    // allocate memory for the per-type constant_force storage and initialize them to (1.0,0,0)
    GlobalVector<Scalar3> tmp_f(m_pdata->getNTypes(), m_exec_conf);

    m_constant_force.swap(tmp_f);
    TAG_ALLOCATION(m_constant_force);

    ArrayHandle<Scalar3> h_constant_force(m_constant_force,
                                          access_location::host,
                                          access_mode::overwrite);
    for (unsigned int i = 0; i < m_constant_force.size(); i++)
        h_constant_force.data[i] = make_scalar3(0.0, 0.0, 0.0);

    // allocate memory for the per-type constant_torque storage and initialize them to (0,0,0)
    GlobalVector<Scalar3> tmp_t(m_pdata->getNTypes(), m_exec_conf);

    m_constant_torque.swap(tmp_t);
    TAG_ALLOCATION(m_constant_torque);

    ArrayHandle<Scalar3> h_constant_torque(m_constant_torque,
                                           access_location::host,
                                           access_mode::overwrite);
    for (unsigned int i = 0; i < m_constant_torque.size(); i++)
        h_constant_torque.data[i] = make_scalar3(0.0, 0.0, 0.0);

#if defined(ENABLE_HIP) && defined(__HIP_PLATFORM_NVCC__)
    if (m_exec_conf->isCUDAEnabled() && m_exec_conf->allConcurrentManagedAccess())
        {
        cudaMemAdvise(m_constant_force.get(),
                      sizeof(Scalar3) * m_constant_force.getNumElements(),
                      cudaMemAdviseSetReadMostly,
                      0);

        cudaMemAdvise(m_constant_torque.get(),
                      sizeof(Scalar3) * m_constant_torque.getNumElements(),
                      cudaMemAdviseSetReadMostly,
                      0);
        }
#endif
    }

ConstantForceCompute::~ConstantForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying ConstantForceCompute" << std::endl;
    }

void ConstantForceCompute::setConstantForce(const std::string& type_name, pybind11::tuple v)
    {
    unsigned int typ = this->m_pdata->getTypeByName(type_name);

    if (pybind11::len(v) != 3)
        {
        throw std::invalid_argument("force values must be 3-tuples");
        }

    // check for user errors
    if (typ >= m_pdata->getNTypes())
        {
        throw std::invalid_argument("Type does not exist");
        }

    Scalar3 force;
    force.x = pybind11::cast<Scalar>(v[0]);
    force.y = pybind11::cast<Scalar>(v[1]);
    force.z = pybind11::cast<Scalar>(v[2]);

    ArrayHandle<Scalar3> h_constant_force(m_constant_force,
                                          access_location::host,
                                          access_mode::readwrite);
    h_constant_force.data[typ] = force;

    m_parameters_updated = true;
    }

pybind11::tuple ConstantForceCompute::getConstantForce(const std::string& type_name)
    {
    pybind11::list v;
    unsigned int typ = this->m_pdata->getTypeByName(type_name);

    ArrayHandle<Scalar3> h_constant_force(m_constant_force,
                                          access_location::host,
                                          access_mode::read);

    Scalar3 f_constantVec = h_constant_force.data[typ];
    v.append(f_constantVec.x);
    v.append(f_constantVec.y);
    v.append(f_constantVec.z);
    return pybind11::tuple(v);
    }

void ConstantForceCompute::setConstantTorque(const std::string& type_name, pybind11::tuple v)
    {
    unsigned int typ = this->m_pdata->getTypeByName(type_name);

    if (pybind11::len(v) != 3)
        {
        throw std::invalid_argument("torque values must be 3-tuples");
        }

    // check for user errors
    if (typ >= m_pdata->getNTypes())
        {
        throw std::invalid_argument("Type does not exist");
        }

    Scalar3 torque;
    torque.x = pybind11::cast<Scalar>(v[0]);
    torque.y = pybind11::cast<Scalar>(v[1]);
    torque.z = pybind11::cast<Scalar>(v[2]);

    ArrayHandle<Scalar3> h_constant_torque(m_constant_torque,
                                           access_location::host,
                                           access_mode::readwrite);
    h_constant_torque.data[typ] = torque;

    m_parameters_updated = true;
    }

pybind11::tuple ConstantForceCompute::getConstantTorque(const std::string& type_name)
    {
    pybind11::list v;
    unsigned int typ = this->m_pdata->getTypeByName(type_name);

    ArrayHandle<Scalar3> h_constant_torque(m_constant_torque,
                                           access_location::host,
                                           access_mode::read);
    Scalar3 t_constantVec = h_constant_torque.data[typ];
    v.append(t_constantVec.x);
    v.append(t_constantVec.y);
    v.append(t_constantVec.z);
    return pybind11::tuple(v);
    }

/*! This function sets appropriate constant forces on all constant particles.
 */
void ConstantForceCompute::setForces()
    {
    //  array handles
    ArrayHandle<Scalar3> h_f_actVec(m_constant_force, access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_t_actVec(m_constant_torque, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar4> h_torque(m_torque, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);

    // sanity check
    assert(h_f_actVec.data != NULL);
    assert(h_t_actVec.data != NULL);

    // zero forces so we don't leave any forces set for indices that are no longer part of our group
    memset(h_force.data, 0, sizeof(Scalar4) * m_force.getNumElements());
    memset(h_torque.data, 0, sizeof(Scalar4) * m_force.getNumElements());

    for (unsigned int i = 0; i < m_group->getNumMembers(); i++)
        {
        unsigned int idx = m_group->getMemberIndex(i);
        unsigned int type = __scalar_as_int(h_pos.data[idx].w);

        vec3<Scalar> fi(h_f_actVec.data[type].x, h_f_actVec.data[type].y, h_f_actVec.data[type].z);
        h_force.data[idx] = vec_to_scalar4(fi, 0);

        vec3<Scalar> ti(h_t_actVec.data[type].x, h_t_actVec.data[type].y, h_t_actVec.data[type].z);
        h_torque.data[idx] = vec_to_scalar4(ti, 0);
        }
    }

/*! This function applies rotational diffusion and sets forces for all constant particles
    \param timestep Current timestep
*/
void ConstantForceCompute::computeForces(uint64_t timestep)
    {
    if (m_particles_sorted || m_parameters_updated)
        {
        setForces(); // set forces for particles
        m_parameters_updated = false;
        }

#ifdef ENABLE_HIP
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
#endif
    }

namespace detail
    {
void export_ConstantForceCompute(pybind11::module& m)
    {
    pybind11::class_<ConstantForceCompute, ForceCompute, std::shared_ptr<ConstantForceCompute>>(
        m,
        "ConstantForceCompute")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>>())
        .def("setConstantForce", &ConstantForceCompute::setConstantForce)
        .def("getConstantForce", &ConstantForceCompute::getConstantForce)
        .def("setConstantTorque", &ConstantForceCompute::setConstantTorque)
        .def("getConstantTorque", &ConstantForceCompute::getConstantTorque)
        .def_property_readonly("filter",
                               [](ConstantForceCompute& force)
                               { return force.getGroup()->getFilter(); });
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
