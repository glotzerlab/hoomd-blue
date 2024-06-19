// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/RejectionVirtualParticleFillerGPU.cuh
 * \brief Declaration and definition of CUDA kernels for RejectionVirtualParticleFillerGPU
 */

#ifndef MPCD_REJECTION_VIRTUAL_PARTICLE_FILLER_GPU_H_
#define MPCD_REJECTION_VIRTUAL_PARTICLE_FILLER_GPU_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "RejectionVirtualParticleFiller.h"
#include "RejectionVirtualParticleFillerGPU.cuh"

#include "hoomd/Autotuner.h"
#include "hoomd/CachedAllocator.h"
#include <pybind11/pybind11.h>

#include <iterator>

namespace hoomd
    {
namespace mpcd
    {

//! Adds virtual particles to the MPCD particle data for various confining geometries using the GPU
template<class Geometry>
class PYBIND11_EXPORT RejectionVirtualParticleFillerGPU
    : public mpcd::RejectionVirtualParticleFiller<Geometry>
    {
    public:
    //! Constructor
    RejectionVirtualParticleFillerGPU(std::shared_ptr<SystemDefinition> sysdef,
                                      const std::string& type,
                                      Scalar density,
                                      std::shared_ptr<Variant> T,
                                      std::shared_ptr<const Geometry> geom)
        : mpcd::RejectionVirtualParticleFiller<Geometry>(sysdef, type, density, T, geom),
          m_keep_particles(this->m_exec_conf), m_keep_indices(this->m_exec_conf),
          m_num_keep(this->m_exec_conf)
        {
        m_tuner1.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf)},
                                        this->m_exec_conf,
                                        "mpcd_rejection_filler_draw_particles"));
        m_tuner2.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf)},
                                        this->m_exec_conf,
                                        "mpcd_rejection_filler_tag_particles"));
        this->m_autotuners.insert(this->m_autotuners.end(), {m_tuner1, m_tuner2});
        }

    protected:
    //! Fill the volume outside the confinement
    virtual void fill(unsigned int timestep);

    private:
    GPUArray<bool> m_keep_particles; // Track whether particles are in/out of bounds for geometry
    GPUArray<unsigned int> m_keep_indices;  // Indices for particles out of bound for geometry
    GPUFlags<unsigned int> m_num_keep;      // Number of particles to keep
    std::shared_ptr<Autotuner<1>> m_tuner1; //!< Autotuner for drawing particles
    std::shared_ptr<Autotuner<1>> m_tuner2; //!< Autotuner for particle tagging
    };

template<class Geometry>
void RejectionVirtualParticleFillerGPU<Geometry>::fill(unsigned int timestep)
    {
    // Number of particles that we need to draw (constant)
    const BoxDim& box = this->m_pdata->getBox();
    const Scalar3 lo = box.getLo();
    const Scalar3 hi = box.getHi();
    const unsigned int num_virtual_max
        = static_cast<unsigned int>(std::round(this->m_density * box.getVolume()));

    // Step 1
    if (num_virtual_max > this->m_tmp_pos.getNumElements())
        {
        GPUArray<Scalar4> tmp_pos(num_virtual_max, this->m_exec_conf);
        this->m_tmp_pos.swap(tmp_pos);
        GPUArray<Scalar4> tmp_vel(num_virtual_max, this->m_exec_conf);
        this->m_tmp_vel.swap(tmp_vel);
        GPUArray<bool> keep_particles(num_virtual_max, this->m_exec_conf);
        m_keep_particles.swap(keep_particles);
        GPUArray<unsigned int> keep_indices(num_virtual_max, this->m_exec_conf);
        m_keep_indices.swap(keep_indices);
        }

    // Step 2
    unsigned int first_tag = this->computeFirstTag(num_virtual_max);
    const Scalar vel_factor = fast::sqrt((*this->m_T)(timestep) / this->m_mpcd_pdata->getMass());
    ArrayHandle<Scalar4> d_tmp_pos(this->m_tmp_pos,
                                   access_location::device,
                                   access_mode::overwrite);
    ArrayHandle<Scalar4> d_tmp_vel(this->m_tmp_vel,
                                   access_location::device,
                                   access_mode::overwrite);
    ArrayHandle<bool> d_keep_particles(m_keep_particles,
                                       access_location::device,
                                       access_mode::overwrite);
    ArrayHandle<unsigned int> d_keep_indices(m_keep_indices,
                                             access_location::device,
                                             access_mode::overwrite);
    mpcd::gpu::draw_virtual_particles_args_t args(d_tmp_pos.data,
                                                  d_tmp_vel.data,
                                                  d_keep_particles.data,
                                                  lo,
                                                  hi,
                                                  first_tag,
                                                  vel_factor,
                                                  this->m_type,
                                                  num_virtual_max,
                                                  timestep,
                                                  this->m_sysdef->getSeed(),
                                                  this->m_filler_id,
                                                  m_tuner1->getParam()[0]);
    m_tuner1->begin();
    mpcd::gpu::draw_virtual_particles<Geometry>(args, *(this->m_geom));
    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner1->end();
        {
        // on GPU, we need to compact the selected particles down with CUB
        // size storage
        void* d_tmp_storage = NULL;
        size_t tmp_storage_bytes = 0;
        mpcd::gpu::compact_virtual_particle_indices(d_tmp_storage,
                                                    tmp_storage_bytes,
                                                    d_keep_particles.data,
                                                    num_virtual_max,
                                                    d_keep_indices.data,
                                                    m_num_keep.getDeviceFlags());
        ScopedAllocation<unsigned char> d_tmp_alloc(this->m_exec_conf->getCachedAllocator(),
                                                    (tmp_storage_bytes > 0) ? tmp_storage_bytes
                                                                            : 1);
        d_tmp_storage = (void*)d_tmp_alloc();

        // run selection
        mpcd::gpu::compact_virtual_particle_indices(d_tmp_storage,
                                                    tmp_storage_bytes,
                                                    d_keep_particles.data,
                                                    num_virtual_max,
                                                    d_keep_indices.data,
                                                    m_num_keep.getDeviceFlags());
        }
    const unsigned int num_selected = m_num_keep.readFlags();

    // Step 3
    first_tag = this->computeFirstTag(num_selected);
    const unsigned int first_idx = this->m_mpcd_pdata->addVirtualParticles(num_selected);
    ArrayHandle<Scalar4> d_pos(this->m_mpcd_pdata->getPositions(),
                               access_location::device,
                               access_mode::readwrite);
    ArrayHandle<Scalar4> d_vel(this->m_mpcd_pdata->getVelocities(),
                               access_location::device,
                               access_mode::readwrite);
    ArrayHandle<unsigned int> d_tag(this->m_mpcd_pdata->getTags(),
                                    access_location::device,
                                    access_mode::readwrite);
    m_tuner2->begin();
    mpcd::gpu::copy_virtual_particles(d_keep_indices.data,
                                      d_pos.data,
                                      d_vel.data,
                                      d_tag.data,
                                      d_tmp_pos.data,
                                      d_tmp_vel.data,
                                      first_idx,
                                      first_tag,
                                      num_selected,
                                      m_tuner2->getParam()[0]);
    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner2->end();
    }

namespace detail
    {
//! Export RejectionVirtualParticleFillerGPU to python
template<class Geometry> void export_RejectionVirtualParticleFillerGPU(pybind11::module& m)
    {
    namespace py = pybind11;
    const std::string name = Geometry::getName() + "FillerGPU";
    py::class_<mpcd::RejectionVirtualParticleFillerGPU<Geometry>,
               std::shared_ptr<mpcd::RejectionVirtualParticleFillerGPU<Geometry>>>(
        m,
        name.c_str(),
        py::base<mpcd::RejectionVirtualParticleFiller<Geometry>>())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            const std::string&,
                            Scalar,
                            std::shared_ptr<Variant>,
                            std::shared_ptr<const Geometry>>());
    }
    } // end namespace detail
    } // end namespace mpcd
    } // namespace hoomd
#endif // MPCD_REJECTION_VIRTUAL_PARTICLE_FILLER_GPU_H_
