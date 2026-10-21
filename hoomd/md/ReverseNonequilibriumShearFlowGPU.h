// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file md/ReverseNonequilibriumShearFlowGPU.h
 * \brief Declaration of reverse nonequilibrium shear flow updater on the GPU.
 */

#ifndef MD_REVERSE_NONEQUILIBRIUM_SHEAR_FLOW_GPU_H_
#define MD_REVERSE_NONEQUILIBRIUM_SHEAR_FLOW_GPU_H_

#ifdef __NVCC__
#error This header cannot be compiled by nvcc
#endif

#include "ReverseNonequilibriumShearFlow.h"
#include "ReverseNonequilibriumShearFlowGPU.cuh"
#include "hoomd/Autotuner.h"
#include "hoomd/GPUFlags.h"
#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace md
    {
//! Reverse nonequilibrium shear flow updater on GPU
template<class ParticleLoaderT>
class PYBIND11_EXPORT ReverseNonequilibriumShearFlowGPU
    : public ReverseNonequilibriumShearFlow<ParticleLoaderT>
    {
    public:
    //! Constructor
    template<class T = ParticleLoaderT, std::enable_if_t<!T::use_particle_group, bool> = true>
    ReverseNonequilibriumShearFlowGPU(std::shared_ptr<SystemDefinition> sysdef,
                                      std::shared_ptr<Trigger> trigger,
                                      unsigned int num_swap,
                                      Scalar slab_width,
                                      Scalar target_momentum);

    //! Constructor with ParticleGroup
    template<class T = ParticleLoaderT, std::enable_if_t<T::use_particle_group, bool> = true>
    ReverseNonequilibriumShearFlowGPU(std::shared_ptr<SystemDefinition> sysdef,
                                      std::shared_ptr<Trigger> trigger,
                                      unsigned int num_swap,
                                      Scalar slab_width,
                                      Scalar target_momentum,
                                      std::shared_ptr<ParticleGroup> group);

    protected:
    //! Find candidate particles for swapping
    void findSwapParticles() override;

    //! Sort and copy only the top candidates for swapping
    void sortOutSwapParticles() override;

    //! Swap particle momenta
    void swapParticleMomentum() override;

    private:
    GPUArray<unsigned int> m_num_lo_hi;
    GPUFlags<Scalar> m_momentum_sum;
    std::shared_ptr<hoomd::Autotuner<1>> m_tuner_filter; //!< Autotuner for filtering particles
    std::shared_ptr<hoomd::Autotuner<1>> m_tuner_swap;   //!< Autotuner for swapping particles
    };

template<class ParticleLoaderT>
template<class T, std::enable_if_t<!T::use_particle_group, bool>>
ReverseNonequilibriumShearFlowGPU<ParticleLoaderT>::ReverseNonequilibriumShearFlowGPU(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<Trigger> trigger,
    unsigned int num_swap,
    Scalar slab_width,
    Scalar target_momentum)
    : ReverseNonequilibriumShearFlow<ParticleLoaderT>(sysdef,
                                                      trigger,
                                                      num_swap,
                                                      slab_width,
                                                      target_momentum),
      m_num_lo_hi(2, this->m_exec_conf), m_momentum_sum(this->m_exec_conf)
    {
    m_tuner_filter.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf)},
                                          this->m_exec_conf,
                                          "md_rnes_filter"));
    m_tuner_swap.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf)},
                                        this->m_exec_conf,
                                        "md_rnes_swap"));
    this->m_autotuners.insert(this->m_autotuners.end(), {m_tuner_filter, m_tuner_swap});
    }

template<class ParticleLoaderT>
template<class T, std::enable_if_t<T::use_particle_group, bool>>
ReverseNonequilibriumShearFlowGPU<ParticleLoaderT>::ReverseNonequilibriumShearFlowGPU(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<Trigger> trigger,
    unsigned int num_swap,
    Scalar slab_width,
    Scalar target_momentum,
    std::shared_ptr<ParticleGroup> group)
    : ReverseNonequilibriumShearFlow<ParticleLoaderT>(sysdef,
                                                      trigger,
                                                      num_swap,
                                                      slab_width,
                                                      target_momentum,
                                                      group),
      m_num_lo_hi(2, this->m_exec_conf), m_momentum_sum(this->m_exec_conf)
    {
    m_tuner_filter.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf)},
                                          this->m_exec_conf,
                                          "md_rnes_filter"));
    m_tuner_swap.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf)},
                                        this->m_exec_conf,
                                        "md_rnes_swap"));
    this->m_autotuners.insert(this->m_autotuners.end(), {m_tuner_filter, m_tuner_swap});
    }

template<class ParticleLoaderT>
void ReverseNonequilibriumShearFlowGPU<ParticleLoaderT>::findSwapParticles()
    {
    // fill the layers with (unsorted) particles. this uses do loop for reallocation
    bool filled = false;
    do
        {
        const unsigned int num_lo_alloc
            = static_cast<unsigned int>(this->m_particles_lo.getNumElements());
        const unsigned int num_hi_alloc
            = static_cast<unsigned int>(this->m_particles_hi.getNumElements());
            // filter particles into their slab in y-direction and record momentum in x-direction
            {
            ArrayHandle<Scalar4> d_pos(this->m_particle_loader.getPositions(),
                                       access_location::device,
                                       access_mode::read);
            ArrayHandle<Scalar4> d_vel(this->m_particle_loader.getVelocities(),
                                       access_location::device,
                                       access_mode::read);

            ArrayHandle<Scalar2> d_particles_lo(this->m_particles_lo,
                                                access_location::device,
                                                access_mode::overwrite);
            ArrayHandle<Scalar2> d_particles_hi(this->m_particles_hi,
                                                access_location::device,
                                                access_mode::overwrite);
            ArrayHandle<unsigned int> d_num_lo_hi(m_num_lo_hi,
                                                  access_location::device,
                                                  access_mode::overwrite);

            std::unique_ptr<ArrayHandle<unsigned int>> group_members;
            typename ParticleLoaderT::IndexReader index_reader;
            if constexpr (ParticleLoaderT::use_particle_group)
                {
                group_members.reset(new ArrayHandle<unsigned int>(
                    this->m_particle_loader.getGroup()->getIndexArray(),
                    access_location::device,
                    access_mode::read));
                index_reader = typename ParticleLoaderT::IndexReader(group_members->data);
                }

            m_tuner_filter->begin();
            md::gpu::rnes_filter_particles(
                d_particles_lo.data,
                d_particles_hi.data,
                d_num_lo_hi.data,
                d_pos.data,
                d_vel.data,
                index_reader,
                this->m_particle_loader.makeVelocityMassReader(d_vel.data),
                this->m_pos_lo,
                this->m_pos_hi,
                num_lo_alloc,
                num_hi_alloc,
                this->m_particle_loader.getN(),
                m_tuner_filter->getParam()[0]);
            if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
                CHECK_CUDA_ERROR();
            m_tuner_filter->end();
            }

            // read number in each slab
            {
            ArrayHandle<unsigned int> h_num_lo_hi(m_num_lo_hi,
                                                  access_location::host,
                                                  access_mode::read);
            this->m_num_lo = h_num_lo_hi.data[0];
            this->m_num_hi = h_num_lo_hi.data[1];
            }

        // reallocate if required and go again, this won't happen too much
        filled = true;
        if (this->m_num_lo > num_lo_alloc)
            {
            GPUArray<Scalar2> particles_lo(this->m_num_lo, this->m_exec_conf);
            this->m_particles_lo.swap(particles_lo);
            filled = false;
            }
        if (this->m_num_hi > num_hi_alloc)
            {
            GPUArray<Scalar2> particles_hi(this->m_num_hi, this->m_exec_conf);
            this->m_particles_hi.swap(particles_hi);
            filled = false;
            }
        } while (!filled);
    }

template<class ParticleLoaderT>
void ReverseNonequilibriumShearFlowGPU<ParticleLoaderT>::sortOutSwapParticles()
    {
    ArrayHandle<Scalar2> d_particles_lo(this->m_particles_lo,
                                        access_location::device,
                                        access_mode::readwrite);
    ArrayHandle<Scalar2> d_particles_hi(this->m_particles_hi,
                                        access_location::device,
                                        access_mode::readwrite);
    ArrayHandle<unsigned int> d_tag(this->m_particle_loader.getTags(),
                                    access_location::device,
                                    access_mode::read);

    if (std::isinf(this->m_target_momentum))
        {
        gpu::rnes_sort(d_particles_lo.data, this->m_num_lo, detail::MaximumMomentum(d_tag.data));
        gpu::rnes_sort(d_particles_hi.data, this->m_num_hi, detail::MinimumMomentum(d_tag.data));
        }
    else
        {
        gpu::rnes_sort(d_particles_lo.data,
                       this->m_num_lo,
                       detail::CompareMomentumToTarget(this->m_target_momentum, d_tag.data));
        gpu::rnes_sort(d_particles_hi.data,
                       this->m_num_hi,
                       detail::CompareMomentumToTarget(-this->m_target_momentum, d_tag.data));
        }

    // copy only the top particles to the host memory
    const unsigned int num_top_lo = std::min(this->m_num_swap, this->m_num_lo);
    this->m_top_particles_lo.resize(num_top_lo);
    gpu::rnes_copy_top_particles(this->m_top_particles_lo.data(), d_particles_lo.data, num_top_lo);
    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    const unsigned int num_top_hi = std::min(this->m_num_swap, this->m_num_hi);
    this->m_top_particles_hi.resize(num_top_hi);
    gpu::rnes_copy_top_particles(this->m_top_particles_hi.data(), d_particles_hi.data, num_top_hi);
    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    }

template<class ParticleLoaderT>
void ReverseNonequilibriumShearFlowGPU<ParticleLoaderT>::swapParticleMomentum()
    {
    ArrayHandle<Scalar4> d_vel(this->m_particle_loader.getVelocities(),
                               access_location::device,
                               access_mode::readwrite);
    ArrayHandle<Scalar2> d_particles_staged(this->m_particles_staged,
                                            access_location::device,
                                            access_mode::read);

    // perform swap
    m_momentum_sum.resetFlags(0);
    m_tuner_swap->begin();
    md::gpu::rnes_swap_particles(d_vel.data,
                                 m_momentum_sum.getDeviceFlags(),
                                 d_particles_staged.data,
                                 this->m_particle_loader.makeVelocityMassReader(d_vel.data),
                                 this->m_num_staged,
                                 m_tuner_swap->getParam()[0]);
    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_swap->end();

    // sum momentum exchange from this swap
    Scalar momentum_sum = m_momentum_sum.readFlags();
#ifdef ENABLE_MPI
    if (this->m_sysdef->isDomainDecomposed())
        {
        auto comm = this->m_exec_conf->getMPICommunicator();
        MPI_Allreduce(MPI_IN_PLACE, &momentum_sum, 1, MPI_HOOMD_SCALAR, MPI_SUM, comm);
        }
#endif

    // absolute value gives extra factor of 2, divide out when accumulating
    this->m_summed_momentum_exchange += Scalar(0.5) * momentum_sum;
    }

namespace detail
    {
/*!
 * \param m Python module to export to
 */
template<class ParticleLoaderT>
void export_ReverseNonequilibriumShearFlowGPU(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<md::ReverseNonequilibriumShearFlowGPU<ParticleLoaderT>,
                     md::ReverseNonequilibriumShearFlow<ParticleLoaderT>,
                     std::shared_ptr<md::ReverseNonequilibriumShearFlowGPU<ParticleLoaderT>>>
        cls(m, name.c_str());

    if constexpr (ParticleLoaderT::use_particle_group)
        {
        cls.def(pybind11::init<std::shared_ptr<SystemDefinition>,
                               std::shared_ptr<Trigger>,
                               unsigned int,
                               Scalar,
                               Scalar,
                               std::shared_ptr<ParticleGroup>>());
        }
    else
        {
        cls.def(pybind11::init<std::shared_ptr<SystemDefinition>,
                               std::shared_ptr<Trigger>,
                               unsigned int,
                               Scalar,
                               Scalar>());
        }
    }
    } // namespace detail

    } // end namespace md
    } // end namespace hoomd
#endif // MD_REVERSE_NONEQUILIBRIUM_SHEAR_FLOW_GPU_H_
