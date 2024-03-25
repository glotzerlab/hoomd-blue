// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file ComputeThermoHMA.cc
    \brief Contains code for the ComputeThermoHMA class
*/

#include "ComputeThermoHMA.h"
#include "hoomd/VectorMath.h"

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#include "hoomd/HOOMDMPI.h"
#endif

#include <iomanip>
#include <iostream>
using namespace std;

namespace hoomd
    {
namespace md
    {
/*! \param sysdef System for which to compute thermodynamic properties
    \param group Subset of the system over which properties are calculated
    \param temperature The temperature that governs sampling of the integrator
    \param harmonicPressure The contribution to the pressure from harmonic fluctuations
*/
ComputeThermoHMA::ComputeThermoHMA(std::shared_ptr<SystemDefinition> sysdef,
                                   std::shared_ptr<ParticleGroup> group,
                                   const double temperature,
                                   const double harmonicPressure)
    : Compute(sysdef), m_group(group), m_harmonicPressure(harmonicPressure)
    {
    m_exec_conf->msg->notice(5) << "Constructing ComputeThermoHMA" << endl;

    assert(m_pdata);
    GPUArray<Scalar> properties(thermoHMA_index::num_quantities, m_exec_conf);
    m_properties.swap(properties);

#ifdef ENABLE_MPI
    m_properties_reduced = true;
#endif

    m_temperature = temperature;

    BoxDim box = m_pdata->getGlobalBox();

    SnapshotParticleData<Scalar> snapshot;

    m_pdata->takeSnapshot(snapshot);

#ifdef ENABLE_MPI
    // when the simulation is decomposed on multiple ranks
    if (m_pdata->getDomainDecomposition())
        {
        // broadcast the snapshot so the particle positions are available on all ranks
        snapshot.bcast(0, m_exec_conf->getMPICommunicator());
        }
#endif

    GlobalArray<Scalar3> lat(snapshot.size, m_exec_conf);
    m_lattice_site.swap(lat);
    TAG_ALLOCATION(m_lattice_site);
    ArrayHandle<Scalar3> h_lattice_site(m_lattice_site,
                                        access_location::host,
                                        access_mode::overwrite);

    // for each particle in the data
    for (unsigned int tag = 0; tag < snapshot.size; tag++)
        {
        // save its initial position
        vec3<Scalar> pos = snapshot.pos[tag];
        vec3<Scalar> unwrapped = box.shift(pos, snapshot.image[tag]);
        h_lattice_site.data[tag] = make_scalar3(unwrapped.x, unwrapped.y, unwrapped.z);
        }
    }

ComputeThermoHMA::~ComputeThermoHMA()
    {
    m_exec_conf->msg->notice(5) << "Destroying ComputeThermoHMA" << endl;
    }

/*! Calls computeProperties if the properties need updating
    \param timestep Current time step of the simulation
*/
void ComputeThermoHMA::compute(uint64_t timestep)
    {
    Compute::compute(timestep);
    if (!shouldCompute(timestep))
        return;

    computeProperties();
    }

/*! Computes all thermodynamic properties of the system in one fell swoop.
 */
void ComputeThermoHMA::computeProperties()
    {
    // just drop out if the group is an empty group
    if (m_group->getNumMembersGlobal() == 0)
        return;

    unsigned int group_size = m_group->getNumMembers();

    assert(m_pdata);

    // access the net force, pe, and virial
    const GlobalArray<Scalar>& net_virial = m_pdata->getNetVirial();
    ArrayHandle<Scalar4> h_net_force(m_pdata->getNetForce(),
                                     access_location::host,
                                     access_mode::read);
    ArrayHandle<Scalar> h_net_virial(net_virial, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_lattice_site(m_lattice_site, access_location::host, access_mode::read);
    ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::read);

    // total potential energy
    const BoxDim& box = m_pdata->getGlobalBox();
    Scalar volume;
    Scalar3 L = box.getL();
    unsigned int D = m_sysdef->getNDimensions();
    if (D == 2)
        {
        // "volume" is area in 2D
        volume = L.x * L.y;
        }
    else
        {
        volume = L.x * L.y * L.z;
        }
    double pe_total = 0.0, p_HMA = 0.0;
    double fV = (m_harmonicPressure / m_temperature - group_size / box.getVolume())
                / (D * (group_size - 1));
    double W = 0;
    size_t virial_pitch = net_virial.getPitch();
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        unsigned int tag = h_tag.data[group_idx];
        pe_total += (double)h_net_force.data[j].w;
        W += Scalar(1. / D)
             * ((double)h_net_virial.data[j + 0 * virial_pitch]
                + (double)h_net_virial.data[j + 3 * virial_pitch]
                + (double)h_net_virial.data[j + 5 * virial_pitch]);

        Scalar4 pos4 = h_pos.data[group_idx];
        Scalar3 pos3 = make_scalar3(pos4.x, pos4.y, pos4.z);
        Scalar3 dr = box.shift(pos3, h_image.data[group_idx]) - h_lattice_site.data[tag];
        double fdr = 0;
        fdr += (double)h_net_force.data[group_idx].x * dr.x;
        fdr += (double)h_net_force.data[group_idx].y * dr.y;
        fdr += (double)h_net_force.data[group_idx].z * dr.z;
        pe_total += 0.5 * fdr;
        p_HMA += fV * fdr;
        }
    pe_total += 1.5 * (group_size - 1) * m_temperature;
    pe_total += m_pdata->getExternalEnergy();

    Scalar p_total = m_harmonicPressure + W / volume + p_HMA;
    ArrayHandle<Scalar> h_properties(m_properties, access_location::host, access_mode::overwrite);
    h_properties.data[thermoHMA_index::potential_energyHMA] = Scalar(pe_total);
    h_properties.data[thermoHMA_index::pressureHMA] = p_total;

#ifdef ENABLE_MPI
    // in MPI, reduce extensive quantities only when they're needed
    m_properties_reduced = !m_pdata->getDomainDecomposition();
#endif // ENABLE_MPI
    }

#ifdef ENABLE_MPI
void ComputeThermoHMA::reduceProperties()
    {
    if (m_properties_reduced)
        return;

    // reduce properties
    ArrayHandle<Scalar> h_properties(m_properties, access_location::host, access_mode::readwrite);
    MPI_Allreduce(MPI_IN_PLACE,
                  h_properties.data,
                  thermoHMA_index::num_quantities,
                  MPI_HOOMD_SCALAR,
                  MPI_SUM,
                  m_exec_conf->getMPICommunicator());

    m_properties_reduced = true;
    }
#endif

namespace detail
    {
void export_ComputeThermoHMA(pybind11::module& m)
    {
    pybind11::class_<ComputeThermoHMA, Compute, std::shared_ptr<ComputeThermoHMA>>(
        m,
        "ComputeThermoHMA")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ParticleGroup>,
                            const double,
                            const double>())
        .def_property("kT", &ComputeThermoHMA::getTemperature, &ComputeThermoHMA::setTemperature)
        .def_property("harmonic_pressure",
                      &ComputeThermoHMA::getHarmonicPressure,
                      &ComputeThermoHMA::setHarmonicPressure)
        .def_property_readonly("potential_energy", &ComputeThermoHMA::getPotentialEnergyHMA)
        .def_property_readonly("pressure", &ComputeThermoHMA::getPressureHMA);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
