// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file ComputeThermoHMAGPU.cc
    \brief Contains code for the ComputeThermoHMAGPU class
*/

#include "ComputeThermoHMAGPU.h"
#include "ComputeThermoHMAGPU.cuh"
#include "hoomd/GPUPartition.cuh"

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
    \param temperature The temperature that governs sampling of the integrator
    \param harmonicPressure The contribution to the pressure from harmonic fluctuations
*/

ComputeThermoHMAGPU::ComputeThermoHMAGPU(std::shared_ptr<SystemDefinition> sysdef,
                                         std::shared_ptr<ParticleGroup> group,
                                         const double temperature,
                                         const double harmonicPressure)
    : ComputeThermoHMA(sysdef, group, temperature, harmonicPressure), m_scratch(m_exec_conf)
    {
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error()
            << "Creating a ComputeThermoHMAGPU with no GPU in the execution configuration" << endl;
        throw std::runtime_error("Error initializing ComputeThermoHMAGPU");
        }

    m_block_size = 512;

#ifdef __HIP_PLATFORM_NVCC__
    if (m_exec_conf->allConcurrentManagedAccess())
        {
        auto gpu_map = m_exec_conf->getGPUIds();

        // set up GPU memory mappings
        for (unsigned int idev = 0; idev < m_exec_conf->getNumActiveGPUs(); ++idev)
            {
            // only optimize access for those fields used in force computation
            // (i.e. no net_force/virial/torque, also angmom and inertia are only used by the
            // integrator)
            cudaMemAdvise(m_lattice_site.get(),
                          sizeof(Scalar3) * m_lattice_site.getNumElements(),
                          cudaMemAdviseSetAccessedBy,
                          gpu_map[idev]);
            }
        CHECK_CUDA_ERROR();
        }
#endif

    hipEventCreateWithFlags(&m_event, hipEventDisableTiming);
    }

//! Destructor
ComputeThermoHMAGPU::~ComputeThermoHMAGPU()
    {
    hipEventDestroy(m_event);
    }

/*! Computes all thermodynamic properties of the system in one fell swoop, on the GPU.
 */
void ComputeThermoHMAGPU::computeProperties()
    {
    // just drop out if the group is an empty group
    if (m_group->getNumMembersGlobal() == 0)
        return;

    unsigned int group_size = m_group->getNumMembers();

    assert(m_pdata);

    // number of blocks in reduction (round up for every GPU)
    unsigned int num_blocks
        = m_group->getNumMembers() / m_block_size + m_exec_conf->getNumActiveGPUs();

    // resize work space
    size_t old_size = m_scratch.size();

    m_scratch.resize(num_blocks);

    if (m_scratch.size() != old_size)
        {
#ifdef __HIP_PLATFORM_NVCC__
        if (m_exec_conf->allConcurrentManagedAccess())
            {
            auto& gpu_map = m_exec_conf->getGPUIds();

            // map scratch array into memory of all GPUs
            for (unsigned int idev = 0; idev < m_exec_conf->getNumActiveGPUs(); ++idev)
                {
                cudaMemAdvise(m_scratch.get(),
                              sizeof(Scalar3) * m_scratch.getNumElements(),
                              cudaMemAdviseSetAccessedBy,
                              gpu_map[idev]);
                }
            CHECK_CUDA_ERROR();
            }
#endif

        // reset to zero, to be on the safe side
        ArrayHandle<Scalar3> d_scratch(m_scratch, access_location::device, access_mode::overwrite);

        hipMemset(d_scratch.data, 0, sizeof(Scalar3) * m_scratch.size());
        }

    // access the particle data
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_body(m_pdata->getBodies(),
                                     access_location::device,
                                     access_mode::read);
    ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);
    ArrayHandle<Scalar3> d_lattice_site(m_lattice_site, access_location::device, access_mode::read);
    ArrayHandle<int3> d_image(m_pdata->getImages(), access_location::device, access_mode::read);
    BoxDim box = m_pdata->getGlobalBox();

        { // scope these array handles so they are released before the additional terms are added
        // access the net force, pe, and virial
        const GlobalArray<Scalar4>& net_force = m_pdata->getNetForce();
        const GlobalArray<Scalar>& net_virial = m_pdata->getNetVirial();
        ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::read);
        ArrayHandle<Scalar> d_net_virial(net_virial, access_location::device, access_mode::read);
        ArrayHandle<Scalar3> d_scratch(m_scratch, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar> d_properties(m_properties,
                                         access_location::device,
                                         access_mode::overwrite);

        // access the group
        ArrayHandle<unsigned int> d_index_array(m_group->getIndexArray(),
                                                access_location::device,
                                                access_mode::read);

        m_exec_conf->beginMultiGPU();

        // build up args list
        kernel::compute_thermo_hma_args args;
        args.n_blocks = num_blocks;
        args.d_net_force = d_net_force.data;
        args.d_net_virial = d_net_virial.data;
        args.virial_pitch = net_virial.getPitch();
        args.D = m_sysdef->getNDimensions();
        args.d_scratch = d_scratch.data;
        args.block_size = m_block_size;
        args.external_virial_xx = m_pdata->getExternalVirial(0);
        args.external_virial_yy = m_pdata->getExternalVirial(3);
        args.external_virial_zz = m_pdata->getExternalVirial(5);
        args.external_energy = m_pdata->getExternalEnergy();
        args.temperature = m_temperature;
        args.harmonicPressure = m_harmonicPressure;

        // perform the computation on the GPU(s)
        gpu_compute_thermo_hma_partial(d_pos.data,
                                       d_lattice_site.data,
                                       d_image.data,
                                       d_body.data,
                                       d_tag.data,
                                       d_index_array.data,
                                       group_size,
                                       box,
                                       args,
                                       m_group->getGPUPartition());

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();

        // converge GPUs
        m_exec_conf->endMultiGPU();

        // perform the computation on GPU 0
        gpu_compute_thermo_hma_final(d_properties.data,
                                     d_body.data,
                                     d_tag.data,
                                     d_index_array.data,
                                     group_size,
                                     box,
                                     args);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

#ifdef ENABLE_MPI
    // in MPI, reduce extensive quantities only when they're needed
    m_properties_reduced = !m_pdata->getDomainDecomposition();
#endif // ENABLE_MPI
    }

namespace detail
    {
void export_ComputeThermoHMAGPU(pybind11::module& m)
    {
    pybind11::class_<ComputeThermoHMAGPU, ComputeThermoHMA, std::shared_ptr<ComputeThermoHMAGPU>>(
        m,
        "ComputeThermoHMAGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ParticleGroup>,
                            const double,
                            const double>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
