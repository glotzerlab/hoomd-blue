// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/ReverseNonequilibriumShearFlowGPU.h
 * \brief Definition of mpcd::ReverseNonequilibriumShearFlowGPU
 */

#include "ReverseNonequilibriumShearFlowGPU.cuh"
#include "ReverseNonequilibriumShearFlowGPU.h"
#include "ReverseNonequilibriumShearFlowUtilities.h"

namespace hoomd
    {

mpcd::ReverseNonequilibriumShearFlowGPU::ReverseNonequilibriumShearFlowGPU(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<Trigger> trigger,
    unsigned int num_swap,
    Scalar slab_width,
    Scalar target_momentum)
    : mpcd::ReverseNonequilibriumShearFlow(sysdef, trigger, num_swap, slab_width, target_momentum),
      m_num_lo_hi(2, m_exec_conf), m_momentum_sum(m_exec_conf)
    {
    m_tuner_filter.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                          m_exec_conf,
                                          "mpcd_rnes_filter"));
    m_tuner_swap.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                        m_exec_conf,
                                        "mpcd_rnes_swap"));
    m_autotuners.insert(m_autotuners.end(), {m_tuner_filter, m_tuner_swap});
    }

void mpcd::ReverseNonequilibriumShearFlowGPU::findSwapParticles()
    {
    // fill the layers with (unsorted) particles. this uses do loop for reallocation
    bool filled = false;
    do
        {
        const unsigned int num_lo_alloc
            = static_cast<unsigned int>(m_particles_lo.getNumElements());
        const unsigned int num_hi_alloc
            = static_cast<unsigned int>(m_particles_hi.getNumElements());
            // filter particles into their slab in y-direction and record momentum in x-direction
            {
            ArrayHandle<Scalar4> d_pos(m_mpcd_pdata->getPositions(),
                                       access_location::device,
                                       access_mode::read);
            ArrayHandle<Scalar4> d_vel(m_mpcd_pdata->getVelocities(),
                                       access_location::device,
                                       access_mode::read);

            ArrayHandle<Scalar2> d_particles_lo(m_particles_lo,
                                                access_location::device,
                                                access_mode::overwrite);
            ArrayHandle<Scalar2> d_particles_hi(m_particles_hi,
                                                access_location::device,
                                                access_mode::overwrite);
            ArrayHandle<unsigned int> d_num_lo_hi(m_num_lo_hi,
                                                  access_location::device,
                                                  access_mode::overwrite);

            m_tuner_filter->begin();
            mpcd::gpu::rnes_filter_particles(d_particles_lo.data,
                                             d_particles_hi.data,
                                             d_num_lo_hi.data,
                                             d_pos.data,
                                             d_vel.data,
                                             m_mpcd_pdata->getMass(),
                                             m_pos_lo,
                                             m_pos_hi,
                                             num_lo_alloc,
                                             num_hi_alloc,
                                             m_mpcd_pdata->getN(),
                                             m_tuner_filter->getParam()[0]);
            if (m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            m_tuner_filter->end();
            }

            // read number in each slab
            {
            ArrayHandle<unsigned int> h_num_lo_hi(m_num_lo_hi,
                                                  access_location::host,
                                                  access_mode::read);
            m_num_lo = h_num_lo_hi.data[0];
            m_num_hi = h_num_lo_hi.data[1];
            }

        // reallocate if required and go again, this won't happen too much
        filled = true;
        if (m_num_lo > num_lo_alloc)
            {
            GPUArray<Scalar2> particles_lo(m_num_lo, m_exec_conf);
            m_particles_lo.swap(particles_lo);
            filled = false;
            }
        if (m_num_hi > num_hi_alloc)
            {
            GPUArray<Scalar2> particles_hi(m_num_hi, m_exec_conf);
            m_particles_hi.swap(particles_hi);
            filled = false;
            }
        } while (!filled);
    }

void mpcd::ReverseNonequilibriumShearFlowGPU::sortOutSwapParticles()
    {
    ArrayHandle<Scalar2> d_particles_lo(m_particles_lo,
                                        access_location::device,
                                        access_mode::readwrite);
    ArrayHandle<Scalar2> d_particles_hi(m_particles_hi,
                                        access_location::device,
                                        access_mode::readwrite);
    ArrayHandle<unsigned int> d_tag(m_mpcd_pdata->getTags(),
                                    access_location::device,
                                    access_mode::read);

    if (std::isinf(m_target_momentum))
        {
        mpcd::gpu::rnes_sort(d_particles_lo.data, m_num_lo, detail::MaximumMomentum(d_tag.data));
        mpcd::gpu::rnes_sort(d_particles_hi.data, m_num_hi, detail::MinimumMomentum(d_tag.data));
        }
    else
        {
        mpcd::gpu::rnes_sort(d_particles_lo.data,
                             m_num_lo,
                             detail::CompareMomentumToTarget(m_target_momentum, d_tag.data));
        mpcd::gpu::rnes_sort(d_particles_hi.data,
                             m_num_hi,
                             detail::CompareMomentumToTarget(-m_target_momentum, d_tag.data));
        }

    // copy only the top particles to the host memory
    const unsigned int num_top_lo = std::min(m_num_swap, m_num_lo);
    m_top_particles_lo.resize(num_top_lo);
    mpcd::gpu::rnes_copy_top_particles(m_top_particles_lo.data(), d_particles_lo.data, num_top_lo);
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    const unsigned int num_top_hi = std::min(m_num_swap, m_num_hi);
    m_top_particles_hi.resize(num_top_hi);
    mpcd::gpu::rnes_copy_top_particles(m_top_particles_hi.data(), d_particles_hi.data, num_top_hi);
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    }

void mpcd::ReverseNonequilibriumShearFlowGPU::swapParticleMomentum()
    {
    ArrayHandle<Scalar4> d_vel(m_mpcd_pdata->getVelocities(),
                               access_location::device,
                               access_mode::readwrite);
    ArrayHandle<Scalar2> d_particles_staged(m_particles_staged,
                                            access_location::device,
                                            access_mode::read);

    // perform swap
    m_momentum_sum.resetFlags(0);
    m_tuner_swap->begin();
    mpcd::gpu::rnes_swap_particles(d_vel.data,
                                   m_momentum_sum.getDeviceFlags(),
                                   d_particles_staged.data,
                                   m_mpcd_pdata->getMass(),
                                   m_num_staged,
                                   m_tuner_swap->getParam()[0]);
    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_swap->end();

    // sum momentum exchange from this swap
    Scalar momentum_sum = m_momentum_sum.readFlags();
#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        auto comm = m_exec_conf->getMPICommunicator();
        MPI_Allreduce(MPI_IN_PLACE, &momentum_sum, 1, MPI_HOOMD_SCALAR, MPI_SUM, comm);
        }
#endif

    // absolute value gives extra factor of 2, divide out when accumulating
    m_summed_momentum_exchange += Scalar(0.5) * momentum_sum;
    }

namespace mpcd
    {
namespace detail
    {
/*!
 * \param m Python module to export to
 */
void export_ReverseNonequilibriumShearFlowGPU(pybind11::module& m)
    {
    pybind11::class_<mpcd::ReverseNonequilibriumShearFlowGPU,
                     mpcd::ReverseNonequilibriumShearFlow,
                     std::shared_ptr<mpcd::ReverseNonequilibriumShearFlowGPU>>(
        m,
        "ReverseNonequilibriumShearFlowGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<Trigger>,
                            unsigned int,
                            Scalar,
                            Scalar>());
    }
    } // namespace detail
    } // namespace mpcd

    } // end namespace hoomd
