// Copyright (c) 2009-2018 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: ajs42

/*! \file ComputeThermoHMA.cc
    \brief Contains code for the ComputeThermoHMA class
*/

#include "ComputeThermoHMA.h"
#include "VectorMath.h"

#ifdef ENABLE_MPI
#include "Communicator.h"
#include "HOOMDMPI.h"
#endif

namespace py = pybind11;

#include <iostream>
#include <iomanip>
using namespace std;

/*! \param sysdef System for which to compute thermodynamic properties
    \param group Subset of the system over which properties are calculated
    \param temperature The temperature that governs sampling of the integrator
    \param suffix Suffix to append to all logged quantity names
*/
ComputeThermoHMA::ComputeThermoHMA(std::shared_ptr<SystemDefinition> sysdef,
                             std::shared_ptr<ParticleGroup> group, const double temperature,
                             const std::string& suffix)
    : Compute(sysdef), m_group(group), m_logging_enabled(true)
    {
    m_exec_conf->msg->notice(5) << "Constructing ComputeThermoHMA" << endl;

    assert(m_pdata);
    GPUArray< Scalar > properties(thermoHMA_index::num_quantities, m_exec_conf);
    m_properties.swap(properties);

    m_logname_list.push_back(string("potential_energyHMA") + suffix);
    m_logname_list.push_back(string("pressureHMA") + suffix);

    #ifdef ENABLE_MPI
    m_properties_reduced = true;
    #endif

    T = temperature;

    m_lattice_x.resize(m_pdata->getNGlobal());
    m_lattice_y.resize(m_pdata->getNGlobal());
    m_lattice_z.resize(m_pdata->getNGlobal());

    BoxDim box = m_pdata->getGlobalBox();

    SnapshotParticleData<Scalar> snapshot;

    m_pdata->takeSnapshot(snapshot);

    // for each particle in the data
    for (unsigned int tag = 0; tag < snapshot.size; tag++)
        {
        // save its initial position
        vec3<Scalar> pos = snapshot.pos[tag];
        vec3<Scalar> unwrapped = box.shift(pos, snapshot.image[tag]);
        m_lattice_x[tag] = unwrapped.x;
        m_lattice_y[tag] = unwrapped.y;
        m_lattice_z[tag] = unwrapped.z;
        }
    }

ComputeThermoHMA::~ComputeThermoHMA()
    {
    m_exec_conf->msg->notice(5) << "Destroying ComputeThermoHMA" << endl;

    //m_pdata->getParticleSortSignal().disconnect<ComputeThermoHMA, &ComputeThermoHMA::slotParticleSort>(this);
    }

/*! \param ndof Number of degrees of freedom to set
*/
void ComputeThermoHMA::setHarmonicPressure(double harmonicPressure)
    {
    pHarmonic = harmonicPressure;
    }

/*! Calls computeProperties if the properties need updating
    \param timestep Current time step of the simulation
*/
void ComputeThermoHMA::compute(unsigned int timestep)
    {
    if (!shouldCompute(timestep))
        return;

    computeProperties();
    }

std::vector< std::string > ComputeThermoHMA::getProvidedLogQuantities()
    {
    if (m_logging_enabled)
        {
        return m_logname_list;
        }
    else
        {
        return std::vector< std::string >();
        }
    }

Scalar ComputeThermoHMA::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    compute(timestep);
    if (quantity == m_logname_list[0])
        {
        return getPotentialEnergyHMA();
        }
    else if (quantity == m_logname_list[1])
        {
        return getPressureHMA();
        }
    else
        {
        m_exec_conf->msg->error() << "compute.thermoHMA: " << quantity << " is not a valid log quantity" << endl;
        throw runtime_error("Error getting log value");
        }
    }

/*! Computes all thermodynamic properties of the system in one fell swoop.
*/
void ComputeThermoHMA::computeProperties()
    {
    // just drop out if the group is an empty group
    if (m_group->getNumMembersGlobal() == 0)
        return;

    unsigned int group_size = m_group->getNumMembers();

    if (m_prof) m_prof->push("ThermoHMA");

    assert(m_pdata);

    // access the net force, pe, and virial
    const GlobalArray< Scalar >& net_virial = m_pdata->getNetVirial();
    ArrayHandle<Scalar4> h_net_force(m_pdata->getNetForce(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_net_virial(net_virial, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

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
    double fV = (pHarmonic/T - group_size/box.getVolume())/(D*(group_size-1));
    Scalar3 dr0;
    double W = 0;
    unsigned int virial_pitch = net_virial.getPitch();
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        //unsigned int tag = m_group->getMemberTag(group_idx);
        unsigned int tag = h_tag.data[group_idx];
        if (tag!=0) continue;
        Scalar dx = h_pos.data[group_idx].x - m_lattice_x[tag];
        Scalar dy = h_pos.data[group_idx].y - m_lattice_y[tag];
        Scalar dz = h_pos.data[group_idx].z - m_lattice_z[tag];
        dr0 = make_scalar3(dx, dy, dz);
        break;
        }
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        //unsigned int tag = m_group->getMemberTag(group_idx);
        unsigned int j = m_group->getMemberIndex(group_idx);
        unsigned int tag = h_tag.data[group_idx];
        pe_total += (double)h_net_force.data[j].w;
        W += Scalar(1./D)* ((double)h_net_virial.data[j+0*virial_pitch] +
                            (double)h_net_virial.data[j+3*virial_pitch] +
                            (double)h_net_virial.data[j+5*virial_pitch] );

        Scalar dx = h_pos.data[group_idx].x - m_lattice_x[tag] - dr0.x;
        Scalar dy = h_pos.data[group_idx].y - m_lattice_y[tag] - dr0.y;
        Scalar dz = h_pos.data[group_idx].z - m_lattice_z[tag] - dr0.z;
        Scalar3 dr = make_scalar3(dx, dy, dz);
        dr = box.minImage(dr);
        double fdr = 0;
        fdr += (double)h_net_force.data[group_idx].x * dr.x;
        fdr += (double)h_net_force.data[group_idx].y * dr.y;
        fdr += (double)h_net_force.data[group_idx].z * dr.z;
        pe_total += 0.5*fdr;
        p_HMA += fV*fdr;
        }
    pe_total += 1.5*(group_size-1)*T;
    pe_total += m_pdata->getExternalEnergy();

    Scalar p_total = pHarmonic + W / volume + p_HMA;
    ArrayHandle<Scalar> h_properties(m_properties, access_location::host, access_mode::overwrite);
    h_properties.data[thermoHMA_index::potential_energyHMA] = Scalar(pe_total);
    h_properties.data[thermoHMA_index::pressureHMA] = p_total;

    #ifdef ENABLE_MPI
    // in MPI, reduce extensive quantities only when they're needed
    m_properties_reduced = !m_pdata->getDomainDecomposition();
    #endif // ENABLE_MPI

    if (m_prof) m_prof->pop();
    }

#ifdef ENABLE_MPI
void ComputeThermoHMA::reduceProperties()
    {
    if (m_properties_reduced) return;

    // reduce properties
    ArrayHandle<Scalar> h_properties(m_properties, access_location::host, access_mode::readwrite);
    MPI_Allreduce(MPI_IN_PLACE, h_properties.data, thermoHMA_index::num_quantities, MPI_HOOMD_SCALAR,
            MPI_SUM, m_exec_conf->getMPICommunicator());

    m_properties_reduced = true;
    }
#endif

void export_ComputeThermoHMA(py::module& m)
    {
    py::class_<ComputeThermoHMA, std::shared_ptr<ComputeThermoHMA> >(m,"ComputeThermoHMA",py::base<Compute>())
    .def(py::init< std::shared_ptr<SystemDefinition>,std::shared_ptr<ParticleGroup>,const double,const std::string& >())
    .def("getPotentialEnergyHMA", &ComputeThermoHMA::getPotentialEnergyHMA)
    .def("getPressureHMA", &ComputeThermoHMA::getPressureHMA)
    .def("setHarmonicPressure", &ComputeThermoHMA::setHarmonicPressure)
    .def("setLoggingEnabled", &ComputeThermoHMA::setLoggingEnabled)
    ;
    }
