// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file ComputeThermo.cc
    \brief Contains code for the ComputeThermo class
*/

#include "ComputeThermo.h"
#include "hoomd/VectorMath.h"

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#include "hoomd/HOOMDMPI.h"
#endif

#include <iostream>
using namespace std;

namespace hoomd
    {
namespace md
    {
/*! \param sysdef System for which to compute thermodynamic properties
    \param group Subset of the system over which properties are calculated
*/
ComputeThermo::ComputeThermo(std::shared_ptr<SystemDefinition> sysdef,
                             std::shared_ptr<ParticleGroup> group)
    : Compute(sysdef), m_group(group)
    {
    m_exec_conf->msg->notice(5) << "Constructing ComputeThermo" << endl;

    assert(m_pdata);
    GlobalArray<Scalar> properties(thermo_index::num_quantities, m_exec_conf);
    m_properties.swap(properties);
    TAG_ALLOCATION(m_properties);

#if defined(ENABLE_HIP) && defined(__HIP_PLATFORM_NVCC__)
    if (m_exec_conf->isCUDAEnabled() && m_exec_conf->allConcurrentManagedAccess())
        {
        // store in host memory for faster access from CPU
        cudaMemAdvise(m_properties.get(),
                      m_properties.getNumElements() * sizeof(Scalar),
                      cudaMemAdviseSetPreferredLocation,
                      cudaCpuDeviceId);
        CHECK_CUDA_ERROR();
        }
#endif

    m_computed_flags.reset();

#ifdef ENABLE_MPI
    m_properties_reduced = true;
#endif
    }

ComputeThermo::~ComputeThermo()
    {
    m_exec_conf->msg->notice(5) << "Destroying ComputeThermo" << endl;
    }

/*! Calls computeProperties if the properties need updating
    \param timestep Current time step of the simulation
*/
void ComputeThermo::compute(uint64_t timestep)
    {
    Compute::compute(timestep);
    if (shouldCompute(timestep))
        {
        computeProperties();
        m_computed_flags = m_pdata->getFlags();
        }
    }

/*! Computes all thermodynamic properties of the system in one fell swoop.
 */
void ComputeThermo::computeProperties()
    {
    // just drop out if the group is an empty group
    if (m_group->getNumMembersGlobal() == 0)
        return;

    unsigned int group_size = m_group->getNumMembers();

    assert(m_pdata);

    // access the particle data
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_body(m_pdata->getBodies(),
                                     access_location::host,
                                     access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

    // access the net force, pe, and virial
    const GlobalArray<Scalar4>& net_force = m_pdata->getNetForce();
    const GlobalArray<Scalar>& net_virial = m_pdata->getNetVirial();
    ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_net_virial(net_virial, access_location::host, access_mode::read);

    // total kinetic energy
    double ke_trans_total = 0.0;

    PDataFlags flags = m_pdata->getFlags();

    double pressure_kinetic_xx = 0.0;
    double pressure_kinetic_xy = 0.0;
    double pressure_kinetic_xz = 0.0;
    double pressure_kinetic_yy = 0.0;
    double pressure_kinetic_yz = 0.0;
    double pressure_kinetic_zz = 0.0;

    if (flags[pdata_flag::pressure_tensor])
        {
        // Calculate kinetic part of pressure tensor
        for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
            {
            unsigned int j = m_group->getMemberIndex(group_idx);
            // ignore rigid body constituent particles in the sum
            if (h_body.data[j] >= MIN_FLOPPY || h_body.data[j] == h_tag.data[j])
                {
                double mass = h_vel.data[j].w;
                pressure_kinetic_xx += mass * ((double)h_vel.data[j].x * (double)h_vel.data[j].x);
                pressure_kinetic_xy += mass * ((double)h_vel.data[j].x * (double)h_vel.data[j].y);
                pressure_kinetic_xz += mass * ((double)h_vel.data[j].x * (double)h_vel.data[j].z);
                pressure_kinetic_yy += mass * ((double)h_vel.data[j].y * (double)h_vel.data[j].y);
                pressure_kinetic_yz += mass * ((double)h_vel.data[j].y * (double)h_vel.data[j].z);
                pressure_kinetic_zz += mass * ((double)h_vel.data[j].z * (double)h_vel.data[j].z);
                }
            }
        // kinetic energy = 1/2 trace of kinetic part of pressure tensor
        ke_trans_total
            = Scalar(0.5) * (pressure_kinetic_xx + pressure_kinetic_yy + pressure_kinetic_zz);
        }
    else
        {
        // total kinetic energy
        for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
            {
            unsigned int j = m_group->getMemberIndex(group_idx);
            // ignore rigid body constituent particles in the sum
            if (h_body.data[j] >= MIN_FLOPPY || h_body.data[j] == h_tag.data[j])
                {
                ke_trans_total += (double)h_vel.data[j].w
                                  * ((double)h_vel.data[j].x * (double)h_vel.data[j].x
                                     + (double)h_vel.data[j].y * (double)h_vel.data[j].y
                                     + (double)h_vel.data[j].z * (double)h_vel.data[j].z);
                }
            }

        ke_trans_total *= Scalar(0.5);
        }

    // total rotational kinetic energy
    double ke_rot_total = 0.0;

    if (flags[pdata_flag::rotational_kinetic_energy])
        {
        // Calculate rotational part of kinetic energy
        ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                           access_location::host,
                                           access_mode::read);
        ArrayHandle<Scalar4> h_angmom(m_pdata->getAngularMomentumArray(),
                                      access_location::host,
                                      access_mode::read);
        ArrayHandle<Scalar3> h_inertia(m_pdata->getMomentsOfInertiaArray(),
                                       access_location::host,
                                       access_mode::read);

        for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
            {
            unsigned int j = m_group->getMemberIndex(group_idx);
            // ignore rigid body constituent particles in the sum
            if (h_body.data[j] >= MIN_FLOPPY || h_body.data[j] == h_tag.data[j])
                {
                Scalar3 I = h_inertia.data[j];
                quat<Scalar> q(h_orientation.data[j]);
                quat<Scalar> p(h_angmom.data[j]);
                quat<Scalar> s(Scalar(0.5) * conj(q) * p);

                // only if the moment of inertia along one principal axis is non-zero, that axis
                // carries angular momentum
                if (I.x > 0)
                    {
                    ke_rot_total += s.v.x * s.v.x / I.x;
                    }
                if (I.y > 0)
                    {
                    ke_rot_total += s.v.y * s.v.y / I.y;
                    }
                if (I.z > 0)
                    {
                    ke_rot_total += s.v.z * s.v.z / I.z;
                    }
                }
            }

        ke_rot_total /= Scalar(2.0);
        }

    // total potential energy
    double pe_total = 0.0;
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);

        // ignore rigid body constituent particles in the sum
        if (h_body.data[j] >= MIN_FLOPPY || h_body.data[j] == h_tag.data[j])
            {
            pe_total += (double)h_net_force.data[j].w;
            }
        }

    pe_total += m_pdata->getExternalEnergy();

    double W = 0.0;
    double virial_xx = m_pdata->getExternalVirial(0);
    double virial_xy = m_pdata->getExternalVirial(1);
    double virial_xz = m_pdata->getExternalVirial(2);
    double virial_yy = m_pdata->getExternalVirial(3);
    double virial_yz = m_pdata->getExternalVirial(4);
    double virial_zz = m_pdata->getExternalVirial(5);

    if (flags[pdata_flag::pressure_tensor])
        {
        // Calculate upper triangular virial tensor
        size_t virial_pitch = net_virial.getPitch();
        for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
            {
            unsigned int j = m_group->getMemberIndex(group_idx);
            // ignore rigid body constituent particles in the sum
            if (h_body.data[j] >= MIN_FLOPPY || h_body.data[j] == h_tag.data[j])
                {
                virial_xx += (double)h_net_virial.data[j + 0 * virial_pitch];
                virial_xy += (double)h_net_virial.data[j + 1 * virial_pitch];
                virial_xz += (double)h_net_virial.data[j + 2 * virial_pitch];
                virial_yy += (double)h_net_virial.data[j + 3 * virial_pitch];
                virial_yz += (double)h_net_virial.data[j + 4 * virial_pitch];
                virial_zz += (double)h_net_virial.data[j + 5 * virial_pitch];
                }
            }

        // isotropic virial = 1/3 trace of virial tensor
        W = Scalar(1. / 3.) * (virial_xx + virial_yy + virial_zz);
        }

    // compute the pressure
    // volume/area & other 2D stuff needed
    BoxDim global_box = m_pdata->getGlobalBox();

    Scalar3 L = global_box.getL();
    Scalar volume;
    unsigned int D = m_sysdef->getNDimensions();
    if (D == 2)
        {
        // "volume" is area in 2D
        volume = L.x * L.y;
        // W needs to be corrected since the 1/3 factor is built in
        W *= Scalar(3.0 / 2.0);
        }
    else
        {
        volume = L.x * L.y * L.z;
        }

    // pressure: P = (N * K_B * T + W)/V
    Scalar pressure = (2.0 * ke_trans_total / Scalar(D) + W) / volume;

    // pressure tensor = (kinetic part + virial) / V
    Scalar pressure_xx = (pressure_kinetic_xx + virial_xx) / volume;
    Scalar pressure_xy = (pressure_kinetic_xy + virial_xy) / volume;
    Scalar pressure_xz = (pressure_kinetic_xz + virial_xz) / volume;
    Scalar pressure_yy = (pressure_kinetic_yy + virial_yy) / volume;
    Scalar pressure_yz = (pressure_kinetic_yz + virial_yz) / volume;
    Scalar pressure_zz = (pressure_kinetic_zz + virial_zz) / volume;

    // fill out the GlobalArray
    ArrayHandle<Scalar> h_properties(m_properties, access_location::host, access_mode::overwrite);
    h_properties.data[thermo_index::translational_kinetic_energy] = Scalar(ke_trans_total);
    h_properties.data[thermo_index::rotational_kinetic_energy] = Scalar(ke_rot_total);
    h_properties.data[thermo_index::potential_energy] = Scalar(pe_total);
    h_properties.data[thermo_index::pressure] = pressure;
    h_properties.data[thermo_index::pressure_xx] = pressure_xx;
    h_properties.data[thermo_index::pressure_xy] = pressure_xy;
    h_properties.data[thermo_index::pressure_xz] = pressure_xz;
    h_properties.data[thermo_index::pressure_yy] = pressure_yy;
    h_properties.data[thermo_index::pressure_yz] = pressure_yz;
    h_properties.data[thermo_index::pressure_zz] = pressure_zz;

#ifdef ENABLE_MPI
    // in MPI, reduce extensive quantities only when they're needed
    m_properties_reduced = !m_pdata->getDomainDecomposition();
#endif // ENABLE_MPI
    }

#ifdef ENABLE_MPI
void ComputeThermo::reduceProperties()
    {
    if (m_properties_reduced)
        return;

    // reduce properties
    ArrayHandle<Scalar> h_properties(m_properties, access_location::host, access_mode::readwrite);
    MPI_Allreduce(MPI_IN_PLACE,
                  h_properties.data,
                  thermo_index::num_quantities,
                  MPI_HOOMD_SCALAR,
                  MPI_SUM,
                  m_exec_conf->getMPICommunicator());

    m_properties_reduced = true;
    }
#endif

namespace detail
    {
void export_ComputeThermo(pybind11::module& m)
    {
    pybind11::class_<ComputeThermo, Compute, std::shared_ptr<ComputeThermo>>(m, "ComputeThermo")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>>())
        .def_property_readonly("kinetic_temperature", &ComputeThermo::getTemperature)
        .def_property_readonly("pressure", &ComputeThermo::getPressure)
        .def_property_readonly("pressure_tensor", &ComputeThermo::getPressureTensorPython)
        .def_property_readonly("degrees_of_freedom", &ComputeThermo::getNDOF)
        .def_property_readonly("translational_degrees_of_freedom",
                               &ComputeThermo::getTranslationalDOF)
        .def_property_readonly("rotational_degrees_of_freedom", &ComputeThermo::getRotationalDOF)
        .def_property_readonly("num_particles", &ComputeThermo::getNumParticles)
        .def_property_readonly("kinetic_energy", &ComputeThermo::getKineticEnergy)
        .def_property_readonly("translational_kinetic_energy",
                               &ComputeThermo::getTranslationalKineticEnergy)
        .def_property_readonly("rotational_kinetic_energy",
                               &ComputeThermo::getRotationalKineticEnergy)
        .def_property_readonly("potential_energy", &ComputeThermo::getPotentialEnergy)
        .def_property_readonly("volume", &ComputeThermo::getVolume);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
