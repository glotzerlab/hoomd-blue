// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/RejectionVirtualParticleFiller.h
 * \brief Declaration and definition of RejectionVirtualParticleFiller
 */

#ifndef MPCD_REJECTION_VIRTUAL_PARTICLE_FILLER_H_
#define MPCD_REJECTION_VIRTUAL_PARTICLE_FILLER_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "VirtualParticleFiller.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace mpcd
    {

//! Adds virtual particles to MPCD particle data for a given geometry.
/*!
 * Here we implement virtual particle filler using rejection sampling method. It is a simple method
 * where we draw positions in 3 dimensions from an uniform distribution and accept particles that
 * lie outside the confinement defined by template geometry.
 * A merit of this method comes from the fact that it works on any geometry that is being used.
 * However, this method degrades in performance as simulation box size increases in size.
 */
template<class Geometry>
class PYBIND11_EXPORT RejectionVirtualParticleFiller : public mpcd::VirtualParticleFiller
    {
    public:
    //! Constructor
    RejectionVirtualParticleFiller(std::shared_ptr<SystemDefinition> sysdef,
                                   const std::string& type,
                                   Scalar density,
                                   std::shared_ptr<Variant> T,
                                   std::shared_ptr<const Geometry> geom)
        : mpcd::VirtualParticleFiller(sysdef, type, density, T), m_geom(geom),
          m_tmp_pos(m_exec_conf), m_tmp_vel(m_exec_conf)
        {
        m_exec_conf->msg->notice(5)
            << "Constructing MPCD RejectionVirtualParticleFiller : " + Geometry::getName()
            << std::endl;
        }

    //! Destructor
    virtual ~RejectionVirtualParticleFiller()
        {
        m_exec_conf->msg->notice(5)
            << "Destroying MPCD RejectionVirtualParticleFiller" << std::endl;
        }

    //! Get the streaming geometry
    std::shared_ptr<const Geometry> getGeometry() const
        {
        return m_geom;
        }

    //! Set the streaming geometry
    void setGeometry(std::shared_ptr<const Geometry> geom)
        {
        m_geom = geom;
        }

    //! Fill the particles outside the confinement
    virtual void fill(uint64_t timestep);

    protected:
    std::shared_ptr<const Geometry> m_geom;
    GPUArray<Scalar4> m_tmp_pos;
    GPUArray<Scalar4> m_tmp_vel;
    };

template<class Geometry> void RejectionVirtualParticleFiller<Geometry>::fill(uint64_t timestep)
    {
    // Number of particles that we need to draw (constant)
    const BoxDim& box = m_pdata->getBox();
    const Scalar3 lo = box.getLo();
    const Scalar3 hi = box.getHi();
    const unsigned int num_virtual_max
        = static_cast<unsigned int>(std::round(m_density * box.getVolume()));

    // Step 1: Create temporary GPUArrays to draw Particles locally using the worst case estimate
    // for number of particles.
    if (num_virtual_max > m_tmp_pos.getNumElements())
        {
        GPUArray<Scalar4> tmp_pos(num_virtual_max, m_exec_conf);
        GPUArray<Scalar4> tmp_vel(num_virtual_max, m_exec_conf);
        m_tmp_pos.swap(tmp_pos);
        m_tmp_vel.swap(tmp_vel);
        }

    // Step 2: Draw the particles and assign velocities simultaneously by using temporary memory.
    // Only keep the ones that are outside the geometry.
    unsigned int num_selected = 0;
    unsigned int first_tag = computeFirstTag(num_virtual_max);
    uint16_t seed = m_sysdef->getSeed();
    const Scalar vel_factor = fast::sqrt((*m_T)(timestep) / m_mpcd_pdata->getMass());
    ArrayHandle<Scalar4> h_tmp_pos(m_tmp_pos, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar4> h_tmp_vel(m_tmp_vel, access_location::host, access_mode::overwrite);
    for (unsigned int i = 0; i < num_virtual_max; ++i)
        {
        const unsigned int tag = first_tag + i;
        hoomd::RandomGenerator rng(
            hoomd::Seed(hoomd::RNGIdentifier::VirtualParticleFiller, timestep, seed),
            hoomd::Counter(tag, m_filler_id));

        Scalar3 particle = make_scalar3(hoomd::UniformDistribution<Scalar>(lo.x, hi.x)(rng),
                                        hoomd::UniformDistribution<Scalar>(lo.y, hi.y)(rng),
                                        hoomd::UniformDistribution<Scalar>(lo.z, hi.z)(rng));

        if (m_geom->isOutside(particle))
            {
            h_tmp_pos.data[num_selected]
                = make_scalar4(particle.x, particle.y, particle.z, __int_as_scalar(m_type));

            hoomd::NormalDistribution<Scalar> gen(vel_factor, 0.0);
            Scalar3 vel;
            gen(vel.x, vel.y, rng);
            vel.z = gen(rng);
            h_tmp_vel.data[num_selected]
                = make_scalar4(vel.x, vel.y, vel.z, __int_as_scalar(mpcd::detail::NO_CELL));
            ++num_selected;
            }
        }

    // Step 3: Allocate memory for the new virtual particles, and copy. Also recompute tags based
    // on actual number selected
    first_tag = computeFirstTag(num_selected);
    const unsigned int first_idx = m_mpcd_pdata->addVirtualParticles(num_selected);
    ArrayHandle<Scalar4> h_pos(m_mpcd_pdata->getPositions(),
                               access_location::host,
                               access_mode::readwrite);
    ArrayHandle<Scalar4> h_vel(m_mpcd_pdata->getVelocities(),
                               access_location::host,
                               access_mode::readwrite);
    ArrayHandle<unsigned int> h_tag(m_mpcd_pdata->getTags(),
                                    access_location::host,
                                    access_mode::readwrite);
    for (unsigned int i = 0; i < num_selected; ++i)
        {
        const unsigned int idx = first_idx + i;
        h_pos.data[idx] = h_tmp_pos.data[i];
        h_vel.data[idx] = h_tmp_vel.data[i];
        h_tag.data[idx] = first_tag + i;
        }
    }

namespace detail
    {
//! Export RejectionVirtualParticleFiller to python
template<class Geometry> void export_RejectionVirtualParticleFiller(pybind11::module& m)
    {
    namespace py = pybind11;
    const std::string name = Geometry::getName() + "Filler";
    py::class_<mpcd::RejectionVirtualParticleFiller<Geometry>,
               std::shared_ptr<mpcd::RejectionVirtualParticleFiller<Geometry>>>(
        m,
        name.c_str(),
        py::base<mpcd::VirtualParticleFiller>())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            const std::string&,
                            Scalar,
                            std::shared_ptr<Variant>,
                            std::shared_ptr<const Geometry>>())
        .def_property_readonly("geometry",
                               &mpcd::RejectionVirtualParticleFiller<Geometry>::getGeometry);
    }
    } // end namespace detail
    } // end namespace mpcd
    } // namespace hoomd
#endif // MPCD_REJECTION_FILLER_H_
