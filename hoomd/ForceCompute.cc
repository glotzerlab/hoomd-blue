// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file ForceCompute.cc
    \brief Defines the ForceCompute class
*/

#include "ForceCompute.h"

#ifdef ENABLE_MPI
#include "Communicator.h"
#endif

#include <iostream>
using namespace std;

#include <pybind11/numpy.h>

#include <memory>

namespace hoomd
    {
/*! \param sysdef System to compute forces on
    \post The Compute is initialized and all memory needed for the forces is allocated
    \post \c force and \c virial GPUarrays are initialized
    \post All forces are initialized to 0
*/
ForceCompute::ForceCompute(std::shared_ptr<SystemDefinition> sysdef)
    : Compute(sysdef), m_particles_sorted(false), m_buffers_writeable(false)
    {
    assert(m_pdata);
    assert(m_pdata->getMaxN() > 0);

    // allocate data on the host
    unsigned int max_num_particles = m_pdata->getMaxN();
    GlobalArray<Scalar4> force(max_num_particles, m_exec_conf);
    GlobalArray<Scalar> virial(max_num_particles, 6, m_exec_conf);
    GlobalArray<Scalar4> torque(max_num_particles, m_exec_conf);
    m_force.swap(force);
    TAG_ALLOCATION(m_force);
    m_virial.swap(virial);
    TAG_ALLOCATION(m_virial);
    m_torque.swap(torque);
    TAG_ALLOCATION(m_torque);

        {
        ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar4> h_torque(m_torque, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);
        memset(h_force.data, 0, sizeof(Scalar4) * m_force.getNumElements());
        memset(h_torque.data, 0, sizeof(Scalar4) * m_torque.getNumElements());
        memset(h_virial.data, 0, sizeof(Scalar) * m_virial.getNumElements());
        }

#if defined(ENABLE_HIP) && defined(__HIP_PLATFORM_NVCC__)
    if (m_exec_conf->isCUDAEnabled() && m_exec_conf->allConcurrentManagedAccess())
        {
        auto gpu_map = m_exec_conf->getGPUIds();

        // set up GPU memory mappings
        for (unsigned int idev = 0; idev < m_exec_conf->getNumActiveGPUs(); ++idev)
            {
            cudaMemAdvise(m_force.get(),
                          sizeof(Scalar4) * m_force.getNumElements(),
                          cudaMemAdviseSetAccessedBy,
                          gpu_map[idev]);
            cudaMemAdvise(m_virial.get(),
                          sizeof(Scalar) * m_virial.getNumElements(),
                          cudaMemAdviseSetAccessedBy,
                          gpu_map[idev]);
            cudaMemAdvise(m_torque.get(),
                          sizeof(Scalar4) * m_torque.getNumElements(),
                          cudaMemAdviseSetAccessedBy,
                          gpu_map[idev]);
            }
        CHECK_CUDA_ERROR();
        }
#endif

    m_virial_pitch = m_virial.getPitch();

    // connect to the ParticleData to receive notifications when particles change order in memory
    m_pdata->getParticleSortSignal().connect<ForceCompute, &ForceCompute::setParticlesSorted>(this);

    // connect to the ParticleData to receive notifications when the maximum number of particles
    // changes
    m_pdata->getMaxParticleNumberChangeSignal().connect<ForceCompute, &ForceCompute::reallocate>(
        this);

    // reset external virial
    for (unsigned int i = 0; i < 6; ++i)
        m_external_virial[i] = Scalar(0.0);

    m_external_energy = Scalar(0.0);

    // initialize GPU memory hints
    updateGPUAdvice();

    // start with no flags computed
    m_computed_flags.reset();

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        m_gather_tag_order = GatherTagOrder(m_exec_conf->getMPICommunicator());
        }
#endif
    }

/*! \post m_force, m_virial and m_torque are resized to the current maximum particle number
 */
void ForceCompute::reallocate()
    {
    m_force.resize(m_pdata->getMaxN());
    m_virial.resize(m_pdata->getMaxN(), 6);
    m_torque.resize(m_pdata->getMaxN());

        {
        ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar4> h_torque(m_torque, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);
        memset(h_force.data, 0, sizeof(Scalar4) * m_force.getNumElements());
        memset(h_torque.data, 0, sizeof(Scalar4) * m_torque.getNumElements());
        memset(h_virial.data, 0, sizeof(Scalar) * m_virial.getNumElements());
        }

    // the pitch of the virial array may have changed
    m_virial_pitch = m_virial.getPitch();

    // update memory hints
    updateGPUAdvice();
    }

void ForceCompute::updateGPUAdvice()
    {
#if defined(ENABLE_HIP) && defined(__HIP_PLATFORM_NVCC__)
    if (m_exec_conf->isCUDAEnabled() && m_exec_conf->allConcurrentManagedAccess())
        {
        auto gpu_map = m_exec_conf->getGPUIds();

        // split preferred location of particle data across GPUs
        const GPUPartition& gpu_partition = m_pdata->getGPUPartition();

        for (unsigned int idev = 0; idev < m_exec_conf->getNumActiveGPUs(); ++idev)
            {
            // set preferred location
            auto range = gpu_partition.getRange(idev);
            unsigned int nelem = range.second - range.first;

            if (!nelem)
                continue;

            cudaMemAdvise(m_force.get() + range.first,
                          sizeof(Scalar4) * nelem,
                          cudaMemAdviseSetPreferredLocation,
                          gpu_map[idev]);
            for (unsigned int i = 0; i < 6; ++i)
                cudaMemAdvise(m_virial.get() + i * m_virial.getPitch() + range.first,
                              sizeof(Scalar) * nelem,
                              cudaMemAdviseSetPreferredLocation,
                              gpu_map[idev]);
            cudaMemAdvise(m_torque.get() + range.first,
                          sizeof(Scalar4) * nelem,
                          cudaMemAdviseSetPreferredLocation,
                          gpu_map[idev]);

            cudaMemPrefetchAsync(m_force.get() + range.first,
                                 sizeof(Scalar4) * nelem,
                                 gpu_map[idev]);
            for (unsigned int i = 0; i < 6; ++i)
                cudaMemPrefetchAsync(m_virial.get() + i * m_virial.getPitch() + range.first,
                                     sizeof(Scalar) * nelem,
                                     gpu_map[idev]);
            cudaMemPrefetchAsync(m_torque.get() + range.first,
                                 sizeof(Scalar4) * nelem,
                                 gpu_map[idev]);
            }
        CHECK_CUDA_ERROR();

        // set up GPU memory mappings
        for (unsigned int idev = 0; idev < m_exec_conf->getNumActiveGPUs(); ++idev)
            {
            cudaMemAdvise(m_force.get(),
                          sizeof(Scalar4) * m_force.getNumElements(),
                          cudaMemAdviseSetAccessedBy,
                          gpu_map[idev]);
            cudaMemAdvise(m_virial.get(),
                          sizeof(Scalar) * m_virial.getNumElements(),
                          cudaMemAdviseSetAccessedBy,
                          gpu_map[idev]);
            cudaMemAdvise(m_torque.get(),
                          sizeof(Scalar4) * m_torque.getNumElements(),
                          cudaMemAdviseSetAccessedBy,
                          gpu_map[idev]);
            }
        CHECK_CUDA_ERROR();
        }
#endif
    }

/*! Frees allocated memory
 */
ForceCompute::~ForceCompute()
    {
    m_pdata->getParticleSortSignal().disconnect<ForceCompute, &ForceCompute::setParticlesSorted>(
        this);
    m_pdata->getMaxParticleNumberChangeSignal().disconnect<ForceCompute, &ForceCompute::reallocate>(
        this);
    }

/*! Sums the total potential energy calculated by the last call to compute() and returns it.
 */
Scalar ForceCompute::calcEnergySum()
    {
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::read);
    double pe_total = m_external_energy;
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        pe_total += (double)h_force.data[i].w;
        }
#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        // reduce potential energy on all processors
        MPI_Allreduce(MPI_IN_PLACE,
                      &pe_total,
                      1,
                      MPI_DOUBLE,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());
        }
#endif
    return Scalar(pe_total);
    }

/*! Sums the potential energy of a particle group calculated by the last call to compute() and
 * returns it.
 */
Scalar ForceCompute::calcEnergyGroup(std::shared_ptr<ParticleGroup> group)
    {
    unsigned int group_size = group->getNumMembers();
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::read);

    double pe_total = 0.0;

    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = group->getMemberIndex(group_idx);

        pe_total += (double)h_force.data[j].w;
        }
#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        // reduce potential energy on all processors
        MPI_Allreduce(MPI_IN_PLACE,
                      &pe_total,
                      1,
                      MPI_DOUBLE,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());
        }
#endif
    return Scalar(pe_total);
    }
/*! Sums the force of a particle group calculated by the last call to compute() and returns it.
 */

vec3<double> ForceCompute::calcForceGroup(std::shared_ptr<ParticleGroup> group)
    {
    unsigned int group_size = group->getNumMembers();
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::read);

    vec3<double> f_total = vec3<double>();

    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = group->getMemberIndex(group_idx);

        f_total += (vec3<double>)h_force.data[j];
        }
#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        // reduce potential energy on all processors
        MPI_Allreduce(MPI_IN_PLACE,
                      &f_total,
                      3,
                      MPI_DOUBLE,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());
        }
#endif
    return vec3<double>(f_total);
    }

/*! Sums the virial contributions of a particle group calculated by the last call to compute() and
 * returns it.
 */
std::vector<Scalar> ForceCompute::calcVirialGroup(std::shared_ptr<ParticleGroup> group)
    {
    const unsigned int group_size = group->getNumMembers();
    const ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::read);

    std::vector<Scalar> total_virial(6, 0.);

    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        const unsigned int j = group->getMemberIndex(group_idx);

        for (int i = 0; i < 6; i++)
            total_virial[i] += h_virial.data[m_virial_pitch * i + j];
        }
#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        // reduce potential energy on all processors
        MPI_Allreduce(MPI_IN_PLACE,
                      total_virial.data(),
                      6,
                      MPI_HOOMD_SCALAR,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());
        }
#endif
    return total_virial;
    }

pybind11::object ForceCompute::getEnergiesPython()
    {
    bool root = true;
#ifdef ENABLE_MPI
    // if we are not the root processor, return None
    root = m_exec_conf->isRoot();
#endif

    std::vector<size_t> dims(1);
    if (root)
        {
        dims[0] = m_pdata->getNGlobal();
        }
    else
        {
        dims[0] = 0;
        }
    std::vector<double> global_energy(dims[0]);

    // sort energies by particle tag
    std::vector<double> local_energy;
    local_energy.reserve(m_pdata->getN());
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::read);
    sortLocalTags();
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        local_energy.push_back(h_force.data[h_rtag.data[m_local_tag[i]]].w);
        }

    if (m_sysdef->isDomainDecomposed())
        {
#ifdef ENABLE_MPI
        m_gather_tag_order.setLocalTagsSorted(m_local_tag);
        m_gather_tag_order.gatherArray(global_energy, local_energy);
#endif
        }
    else
        {
        global_energy = std::move(local_energy);
        }

    if (root)
        {
        return pybind11::array(dims, global_energy.data());
        }
    return pybind11::none();
    }

pybind11::object ForceCompute::getForcesPython()
    {
    bool root = true;
#ifdef ENABLE_MPI
    // if we are not the root processor, return None
    root = m_exec_conf->isRoot();
#endif

    std::vector<size_t> dims(2);
    if (root)
        {
        dims[0] = m_pdata->getNGlobal();
        dims[1] = 3;
        }
    else
        {
        dims[0] = 0;
        dims[1] = 0;
        }
    std::vector<vec3<double>> global_force(dims[0]);

    // sort forces by particle tag
    std::vector<vec3<double>> local_force(m_pdata->getN());
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::read);
    sortLocalTags();
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        local_force[i].x = h_force.data[h_rtag.data[m_local_tag[i]]].x;
        local_force[i].y = h_force.data[h_rtag.data[m_local_tag[i]]].y;
        local_force[i].z = h_force.data[h_rtag.data[m_local_tag[i]]].z;
        }

    if (m_sysdef->isDomainDecomposed())
        {
#ifdef ENABLE_MPI
        m_gather_tag_order.setLocalTagsSorted(m_local_tag);
        m_gather_tag_order.gatherArray(global_force, local_force);
#endif
        }
    else
        {
        global_force = std::move(local_force);
        }

    if (root)
        {
        return pybind11::array(dims, (double*)global_force.data());
        }
    else
        {
        return pybind11::none();
        }
    }

pybind11::object ForceCompute::getTorquesPython()
    {
    bool root = true;
#ifdef ENABLE_MPI
    // if we are not the root processor, return None
    root = m_exec_conf->isRoot();
#endif

    std::vector<size_t> dims(2);
    if (root)
        {
        dims[0] = m_pdata->getNGlobal();
        dims[1] = 3;
        }
    else
        {
        dims[0] = 0;
        dims[1] = 0;
        }
    std::vector<vec3<double>> global_torque(dims[0]);

    // sort torques by particle tag
    std::vector<vec3<double>> local_torque(m_pdata->getN());
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_torque(m_torque, access_location::host, access_mode::read);
    sortLocalTags();
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        local_torque[i].x = h_torque.data[h_rtag.data[m_local_tag[i]]].x;
        local_torque[i].y = h_torque.data[h_rtag.data[m_local_tag[i]]].y;
        local_torque[i].z = h_torque.data[h_rtag.data[m_local_tag[i]]].z;
        }

    if (m_sysdef->isDomainDecomposed())
        {
#ifdef ENABLE_MPI
        m_gather_tag_order.setLocalTagsSorted(m_local_tag);
        m_gather_tag_order.gatherArray(global_torque, local_torque);
#endif
        }
    else
        {
        global_torque = std::move(local_torque);
        }

    if (root)
        {
        return pybind11::array(dims, (double*)global_torque.data());
        }
    else
        {
        return pybind11::none();
        }
    }

pybind11::object ForceCompute::getVirialsPython()
    {
    if (!m_computed_flags[pdata_flag::pressure_tensor])
        {
        return pybind11::none();
        }

    bool root = true;
#ifdef ENABLE_MPI
    // if we are not the root processor, return None
    root = m_exec_conf->isRoot();
#endif

    std::vector<size_t> dims(2);
    if (root)
        {
        dims[0] = m_pdata->getNGlobal();
        dims[1] = 6;
        }
    else
        {
        dims[0] = 0;
        dims[1] = 0;
        }
    std::vector<hoomd::detail::vec6<double>> global_virial(dims[0]);

    // sort virials by particle tag
    std::vector<hoomd::detail::vec6<double>> local_virial(m_pdata->getN());
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::read);
    sortLocalTags();
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        local_virial[i].xx = h_virial.data[m_virial_pitch * 0 + h_rtag.data[m_local_tag[i]]];
        local_virial[i].xy = h_virial.data[m_virial_pitch * 1 + h_rtag.data[m_local_tag[i]]];
        local_virial[i].xz = h_virial.data[m_virial_pitch * 2 + h_rtag.data[m_local_tag[i]]];
        local_virial[i].yy = h_virial.data[m_virial_pitch * 3 + h_rtag.data[m_local_tag[i]]];
        local_virial[i].yz = h_virial.data[m_virial_pitch * 4 + h_rtag.data[m_local_tag[i]]];
        local_virial[i].zz = h_virial.data[m_virial_pitch * 5 + h_rtag.data[m_local_tag[i]]];
        }

    if (m_sysdef->isDomainDecomposed())
        {
#ifdef ENABLE_MPI
        m_gather_tag_order.setLocalTagsSorted(m_local_tag);
        m_gather_tag_order.gatherArray(global_virial, local_virial);
#endif
        }
    else
        {
        global_virial = std::move(local_virial);
        }

    if (root)
        {
        return pybind11::array(dims, (double*)global_virial.data());
        }
    else
        {
        return pybind11::none();
        }
    }

/*! Performs the force computation.
    \param timestep Current Timestep
    \note If compute() has previously been called with a value of timestep equal to
        the current value, the forces are assumed to already have been computed and nothing will
        be done
*/

void ForceCompute::compute(uint64_t timestep)
    {
    Compute::compute(timestep);
    // recompute forces if the particles were sorted, this is a new timestep, or the particle data
    // flags do not match
    if (m_particles_sorted || shouldCompute(timestep) || m_pdata->getFlags() != m_computed_flags)
        {
        computeForces(timestep);
        }

    m_particles_sorted = false;
    m_computed_flags = m_pdata->getFlags();
    }

/*! \param tag Global particle tag
    \returns Torque of particle referenced by tag
 */
Scalar4 ForceCompute::getTorque(unsigned int tag)
    {
    unsigned int i = m_pdata->getRTag(tag);
    bool found = (i < m_pdata->getN());
    Scalar4 result = make_scalar4(0.0, 0.0, 0.0, 0.0);
    if (found)
        {
        ArrayHandle<Scalar4> h_torque(m_torque, access_location::host, access_mode::read);
        result = h_torque.data[i];
        }
#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        unsigned int owner_rank = m_pdata->getOwnerRank(tag);
        MPI_Bcast(&result,
                  sizeof(Scalar4),
                  MPI_BYTE,
                  owner_rank,
                  m_exec_conf->getMPICommunicator());
        }
#endif
    return result;
    }

/*! \param tag Global particle tag
    \returns Force of particle referenced by tag
 */
Scalar3 ForceCompute::getForce(unsigned int tag)
    {
    unsigned int i = m_pdata->getRTag(tag);
    bool found = (i < m_pdata->getN());
    Scalar3 result = make_scalar3(0.0, 0.0, 0.0);
    if (found)
        {
        ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::read);
        result = make_scalar3(h_force.data[i].x, h_force.data[i].y, h_force.data[i].z);
        }
#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        unsigned int owner_rank = m_pdata->getOwnerRank(tag);
        MPI_Bcast(&result,
                  sizeof(Scalar3),
                  MPI_BYTE,
                  owner_rank,
                  m_exec_conf->getMPICommunicator());
        }
#endif
    return result;
    }

/*! \param tag Global particle tag
    \param component Virial component (0=xx, 1=xy, 2=xz, 3=yy, 4=yz, 5=zz)
    \returns Virial of particle referenced by tag
 */
Scalar ForceCompute::getVirial(unsigned int tag, unsigned int component)
    {
    unsigned int i = m_pdata->getRTag(tag);
    bool found = (i < m_pdata->getN());
    Scalar result = Scalar(0.0);
    if (found)
        {
        ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::read);
        result = h_virial.data[m_virial_pitch * component + i];
        }
#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        unsigned int owner_rank = m_pdata->getOwnerRank(tag);
        MPI_Bcast(&result, sizeof(Scalar), MPI_BYTE, owner_rank, m_exec_conf->getMPICommunicator());
        }
#endif
    return result;
    }

/*! \param tag Global particle tag
    \returns Energy of particle referenced by tag
 */
Scalar ForceCompute::getEnergy(unsigned int tag)
    {
    unsigned int i = m_pdata->getRTag(tag);
    bool found = (i < m_pdata->getN());
    Scalar result = Scalar(0.0);
    if (found)
        {
        ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::read);
        result = h_force.data[i].w;
        }
#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        unsigned int owner_rank = m_pdata->getOwnerRank(tag);
        MPI_Bcast(&result, sizeof(Scalar), MPI_BYTE, owner_rank, m_exec_conf->getMPICommunicator());
        }
#endif
    return result;
    }

namespace detail
    {
void export_ForceCompute(pybind11::module& m)
    {
    pybind11::class_<ForceCompute, Compute, std::shared_ptr<ForceCompute>>(m, "ForceCompute")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>())
        .def("getForce", &ForceCompute::getForce)
        .def("getTorque", &ForceCompute::getTorque)
        .def("getVirial", &ForceCompute::getVirial)
        .def("getEnergy", &ForceCompute::getEnergy)
        .def("getExternalEnergy", &ForceCompute::getExternalEnergy)
        .def("getExternalVirial", &ForceCompute::getExternalVirial)
        .def("calcEnergySum", &ForceCompute::calcEnergySum)
        .def("getEnergies", &ForceCompute::getEnergiesPython)
        .def("getForces", &ForceCompute::getForcesPython)
        .def("getTorques", &ForceCompute::getTorquesPython)
        .def("getVirials", &ForceCompute::getVirialsPython);
    }
    } // end namespace detail

    } // end namespace hoomd
