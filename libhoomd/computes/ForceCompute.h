/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2014 The Regents of
the University of Michigan All rights reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: joaander

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 )
#endif

#include <boost/shared_ptr.hpp>
#include <boost/signals2.hpp>

#include "Compute.h"
#include "Index1D.h"

#ifdef ENABLE_CUDA
#include "ParticleData.cuh"
#endif

#ifdef ENABLE_MPI
#include "Communicator.h"
#endif

/*! \file ForceCompute.h
    \brief Declares the ForceCompute class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

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

class ForceCompute : public Compute
    {
    public:
        //! Constructs the compute
        ForceCompute(boost::shared_ptr<SystemDefinition> sysdef);

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
        virtual void preCompute(unsigned int timestep) { }
        #endif

        //! Computes the forces
        virtual void compute(unsigned int timestep);

        //! Benchmark the force compute
        virtual double benchmark(unsigned int num_iters);

        //! Total the potential energy
        Scalar calcEnergySum();

        //! Easy access to the torque on a single particle
        Scalar4 getTorque(unsigned int tag);

        //! Easy access to the force on a single particle
        Scalar3 getForce(unsigned int tag);

        //! Easy access to the virial on a single particle
        Scalar getVirial(unsigned int tag, unsigned int component);

        //! Easy access to the energy on a single particle
        Scalar getEnergy(unsigned int tag);

        //! Get the array of computed forces
        GPUArray<Scalar4>& getForceArray()
            {
            return m_force;
            }

        //! Get the array of computed virials
        GPUArray<Scalar>& getVirialArray()
            {
            return m_virial;
            }

        //! Get the array of computed torques
        GPUArray<Scalar4>& getTorqueArray()
            {
            return m_torque;
            }

        //! Get the contribution to the external virial
        Scalar getExternalVirial(unsigned int dir)
            {
            assert(dir<6);
            return m_external_virial[dir];
            }

        #ifdef ENABLE_MPI
        //! Get requested ghost communication flags
        virtual CommFlags getRequestedCommFlags(unsigned int timestep)
            {
            // by default, only request positions
            CommFlags flags(0);
            flags[comm_flag::position] = 1;
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
            with the same timestep can properly recaculate the forces, which are stored
            by index.
        */
        void setParticlesSorted()
            {
            m_particles_sorted = true;
            }

        //! Reallocate internal arrays
        void reallocate();

        Scalar m_deltaT;  //!< timestep size (required for some types of non-conservative forces)

        GPUArray<Scalar4> m_force;            //!< m_force.x,m_force.y,m_force.z are the x,y,z components of the force, m_force.u is the PE

        /*! per-particle virial, a 2D GPUArray with width=number
            of particles and height=6. The elements of the (upper triangular)
            3x3 virial matrix \f$ \left(\mathrm{virial}_{ij}\right),k \f$ for
            particle \f$k\f$ are stored in the rows and are indexed in the
            order xx, xy, xz, yy, yz, zz
         */
        GPUArray<Scalar>  m_virial;
        unsigned int m_virial_pitch;    //!< The pitch of the 2D virial array
        GPUArray<Scalar4> m_torque;    //!< per-particle torque
        int m_nbytes;                   //!< stores the number of bytes of memory allocated

        Scalar m_external_virial[6]; //!< Stores external contribution to virial

        //! Connection to the signal notifying when particles are resorted
        boost::signals2::connection m_sort_connection;

        //! Connection to the signal notifying when maximum number of particles changes
        boost::signals2::connection m_max_particle_num_change_connection;

        //! Actually perform the computation of the forces
        /*! This is pure virtual here. Sub-classes must implement this function. It will be called by
            the base class compute() when the forces need to be computed.
            \param timestep Current time step
        */
        virtual void computeForces(unsigned int timestep)=0;
    };

//! Exports the ForceCompute class to python
void export_ForceCompute();

#endif

#ifdef WIN32
#pragma warning( pop )
#endif
