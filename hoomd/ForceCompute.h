// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander
#include "Compute.h"
#include "Index1D.h"
#include "ParticleGroup.h"

#include "GlobalArray.h"
#include "GlobalArray.h"

#ifdef ENABLE_CUDA
#include "ParticleData.cuh"
#endif

#ifdef ENABLE_MPI
#include "Communicator.h"
#endif

#include <memory>
#include <hoomd/extern/nano-signal-slot/nano_signal_slot.hpp>

/*! \file ForceCompute.h
    \brief Declares the ForceCompute class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __FORCECOMPUTE_H__
#define __FORCECOMPUTE_H__

//! Handy structure for passing the force arrays around
/*! \c fx, \c fy, \c fz have length equal to the number of particles and store the x,y,z
    components of the force on that particle. \a pe is also included as the potential energy
    for each particle, if it can be defined for the force. \a virial is the per particle virial.

    The per particle potential energy is defined such that \f$ \sum_i^N \mathrm{pe}_i = V_{\mathrm{total}} \f$

    The per particle virial is a upper triangular 3x3 matrix that is defined such
    that
    \f$ \sum_k^N \left(\mathrm{virial}_{ij}\right)_k = \sum_k^N \sum_{l>k} \frac{1}{2} \left( \vec{f}_{kl,i} \vec{r}_{kl,j} \right) \f$

    \ingroup data_structs
*/

class PYBIND11_EXPORT ForceCompute : public Compute
    {
    public:
        //! Constructs the compute
        ForceCompute(std::shared_ptr<SystemDefinition> sysdef);

        //! Destructor
        virtual ~ForceCompute();

        //! Store the timestep size
        virtual void setDeltaT(Scalar dt)
            {
            m_deltaT = dt;
            }

        #ifdef ENABLE_MPI
        //! Pre-compute the forces
        /*! This method is called in MPI simulations BEFORE the particles are migrated
         * and can be used to overlap computation with communication
         */
        virtual void preCompute(unsigned int timestep){}
        #endif

        //! Computes the forces
        virtual void compute(unsigned int timestep);

        //! Benchmark the force compute
        virtual double benchmark(unsigned int num_iters);

        //! Total the potential energy
        Scalar calcEnergySum();

        //! Sum the potential energy of a group
        Scalar calcEnergyGroup(std::shared_ptr<ParticleGroup> group);

        //! Sum the all forces for a group
        vec3<double> calcForceGroup(std::shared_ptr<ParticleGroup> group);

        //! Sum all virial terms for a group
        std::vector<Scalar> calcVirialGroup(std::shared_ptr<ParticleGroup> group);

        //! Easy access to the torque on a single particle
        Scalar4 getTorque(unsigned int tag);

        //! Easy access to the force on a single particle
        Scalar3 getForce(unsigned int tag);

        //! Easy access to the virial on a single particle
        Scalar getVirial(unsigned int tag, unsigned int component);

        //! Easy access to the energy on a single particle
        Scalar getEnergy(unsigned int tag);

        //! Get the array of computed forces
        GlobalArray<Scalar4>& getForceArray()
            {
            return m_force;
            }

        //! Get the array of computed virials
        GlobalArray<Scalar>& getVirialArray()
            {
            return m_virial;
            }

        //! Get the array of computed torques
        GlobalArray<Scalar4>& getTorqueArray()
            {
            return m_torque;
            }

        //! Get the contribution to the external virial
        Scalar getExternalVirial(unsigned int dir)
            {
            assert(dir<6);
            return m_external_virial[dir];
            }

        //! Get the contribution to the external potential energy
        Scalar getExternalEnergy()
            {
            return m_external_energy;
            }

        #ifdef ENABLE_MPI
        //! Get requested ghost communication flags
        virtual CommFlags getRequestedCommFlags(unsigned int timestep)
            {
            // by default, only request positions
            CommFlags flags(0);
            flags[comm_flag::position] = 1;
            flags[comm_flag::net_force] = 1; // only used if constraints are present
            return flags;
            }
        #endif

        //! Returns true if this ForceCompute requires anisotropic integration
        virtual bool isAnisotropic()
            {
            // by default, only translational degrees of freedom are integrated
            return false;
            }

    protected:
        bool m_particles_sorted;    //!< Flag set to true when particles are resorted in memory

        //! Helper function called when particles are sorted
        /*! setParticlesSorted() is passed as a slot to the particle sort signal.
            It is used to flag \c m_particles_sorted so that a second call to compute
            with the same timestep can properly recalculate the forces, which are stored
            by index.
        */
        void setParticlesSorted()
            {
            m_particles_sorted = true;
            }

        //! Reallocate internal arrays
        void reallocate();

        //! Update GPU memory hints
        void updateGPUAdvice();

        Scalar m_deltaT;  //!< timestep size (required for some types of non-conservative forces)

        GlobalArray<Scalar4> m_force;            //!< m_force.x,m_force.y,m_force.z are the x,y,z components of the force, m_force.u is the PE

        /*! per-particle virial, a 2D array with width=number
            of particles and height=6. The elements of the (upper triangular)
            3x3 virial matrix \f$ \left(\mathrm{virial}_{ij}\right),k \f$ for
            particle \f$k\f$ are stored in the rows and are indexed in the
            order xx, xy, xz, yy, yz, zz
         */
        GlobalArray<Scalar>  m_virial;
        unsigned int m_virial_pitch;    //!< The pitch of the 2D virial array
        GlobalArray<Scalar4> m_torque;    //!< per-particle torque
        int m_nbytes;                   //!< stores the number of bytes of memory allocated

        Scalar m_external_virial[6]; //!< Stores external contribution to virial
        Scalar m_external_energy;    //!< Stores external contribution to potential energy

        //! Actually perform the computation of the forces
        /*! This is pure virtual here. Sub-classes must implement this function. It will be called by
            the base class compute() when the forces need to be computed.
            \param timestep Current time step
        */
        virtual void computeForces(unsigned int timestep){}
    };

//! Exports the ForceCompute class to python
#ifndef NVCC
void export_ForceCompute(pybind11::module& m);
#endif

#endif
