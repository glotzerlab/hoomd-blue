// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/ReverseNonequilibriumShearFlow.h
 * \brief Declaration of reverse nonequilibrium shear flow updater.
 */

#ifndef MPCD_REVERSE_NONEQUILIBRIUM_SHEAR_FLOW_H_
#define MPCD_REVERSE_NONEQUILIBRIUM_SHEAR_FLOW_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "hoomd/ParticleGroup.h"
#include "hoomd/Updater.h"
#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace mpcd
    {
//! Reverse nonequilibrium shear flow updater
/*!
 * A flow is induced by swapping velocities in x direction based on particle position in
 * y-direction.
 */
class PYBIND11_EXPORT ReverseNonequilibriumShearFlow : public Updater
    {
    public:
    //! Constructor
    ReverseNonequilibriumShearFlow(std::shared_ptr<SystemDefinition> sysdef,
                                   std::shared_ptr<Trigger> trigger,
                                   unsigned int num_swap,
                                   Scalar slab_width,
                                   Scalar target_momentum);

    //! Destructor
    virtual ~ReverseNonequilibriumShearFlow();

    //! Apply velocity swaps
    virtual void update(uint64_t timestep);

    //! Get the maximum number of swapped pairs
    Scalar getNumSwap() const
        {
        return m_num_swap;
        }

    //! Set the maximum number of swapped pairs
    void setNumSwap(unsigned int num_swap);

    //! Get the slab width
    Scalar getSlabWidth() const
        {
        return m_slab_width;
        }

    //! Set the slab width
    void setSlabWidth(Scalar slab_width);

    //! Get the target momentum
    Scalar getTargetMomentum() const
        {
        return m_target_momentum;
        }

    //! Set the target momentum
    void setTargetMomentum(Scalar target_momentum);

    //! Get summed exchanged momentum
    Scalar getSummedExchangedMomentum() const
        {
        return m_summed_momentum_exchange;
        }

    protected:
    std::shared_ptr<mpcd::ParticleData> m_mpcd_pdata; //!< MPCD particle data

    unsigned int m_num_swap;  //!< Maximum number of swaps
    Scalar m_slab_width;      //!< Width of slabs
    Scalar m_target_momentum; //!< Target momentum

    Scalar m_summed_momentum_exchange; //!< Total momentum exchanged so far
    Scalar2 m_pos_lo;                  //!< y bounds of lower slab
    Scalar2 m_pos_hi;                  //!< y bounds of upper slab
    unsigned int m_num_lo;             //!< Number of particles in lower slab
    GPUArray<Scalar2> m_particles_lo;  //!< Sorted particle indexes and momenta in lower slab
    unsigned int m_num_hi;             //!< Number of particles in upper slab
    GPUArray<Scalar2> m_particles_hi;  //!< Sorted particle indexes and momenta in upper slab

    std::vector<Scalar2> m_top_particles_lo; //!< Top candidates for swapping in lower slab
    std::vector<Scalar2> m_top_particles_hi; //!< Top candidates for swapping in upper slab
    unsigned int m_num_staged;               //!< Number of particles staged for swapping
    GPUArray<Scalar2> m_particles_staged;    //!< Particle indexes and momenta staged for swapping

    //! Find candidate particles for swapping
    virtual void findSwapParticles();

    //! Sort and copy only the top candidates for swapping
    virtual void sortOutSwapParticles();

    //! Stage particles for swapping
    void stageSwapParticles();

    //! Swap particle momenta
    virtual void swapParticleMomentum();

    private:
    bool m_update_slabs; //!< If true, update the slab positions

    //! Request to check box on next update
    void requestUpdateSlabs()
        {
        m_update_slabs = true;
        }

    //! Sets the slab positions in the box and validates them
    void setSlabs();
    };

    } // end namespace mpcd
    } // end namespace hoomd
#endif // MPCD_REVERSE_NONEQUILIBRIUM_SHEAR_FLOW_H_
