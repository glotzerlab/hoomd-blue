/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
// Maintainer: joaander

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 )
#endif

#include <boost/shared_ptr.hpp>
#include <boost/signals.hpp>

#include "Compute.h"
#include "Index1D.h"

#ifdef ENABLE_CUDA
#include "ParticleData.cuh"
#include "ForceCompute.cuh"
#endif

/*! \file ForceCompute.h
    \brief Declares the ForceCompute class
*/

#ifndef __FORCECOMPUTE_H__
#define __FORCECOMPUTE_H__

//! Handy structure for passing the force arrays around
/*! \c fx, \c fy, \c fz have length equal to the number of particles and store the x,y,z
    components of the force on that particle. \a pe is also included as the potential energy
    for each particle, if it can be defined for the force. \a virial is the per particle virial.

    The per particle potential energy is defined such that \f$ \sum_i^N \mathrm{pe}_i = V_{\mathrm{total}} \f$

    The per particle virial is defined such that 
    \f$ \sum_i^N \mathrm{virial}_i = \frac{1}{3} \sum_i^N \sum_{j>i} \vec{r}_{ij} \cdot \vec{f}_{ij} \f$

    \ingroup data_structs
*/
struct ForceDataArrays
    {
    //! Zeroes pointers
    ForceDataArrays();
    
<<<<<<< .mine
		GPUArray<Scalar4> force;
		GPUArray<Scalar> virial;
		Scalar const * __restrict__ fx; //!< x-component of the force
		Scalar const * __restrict__ fy; //!< y-component of the force
    Scalar const * __restrict__ fz; //!< z-component of the force
    Scalar const * __restrict__ pe; //!< per-particle potential energy
    Scalar const * __restrict__ virial; //!< per-particle virial
=======
    GPUArray<Scalar4> f;
    GPUArray<Scalar> virial; //!< per-particle virial
>>>>>>> .r3651
    };

#ifdef ENABLE_CUDA
//! Structure for managing force data on the GPU
/*! ForceDataArraysGPU is very closely tied in with ForceCompute and neither can exist without
    the other (at least when compiling for the GPU). ForceCompute uses ForceDataArraysGPU
    to store the force data on each GPU in the execution configuration. ForceDataArraysGPU does
    all the dirty work of allocating memory and performing the host to/from device transfers.

    Internal storage of the pointers to device memory where the data is stored can be accessed
    via the gpu_force_data_arrays member \c d_data.

    \ingroup data_structs
*/
<<<<<<< .mine
/*
	struct ForceDataArraysGPU
			{
			//! Zeros pointers
			ForceDataArraysGPU();
			
			gpu_force_data_arrays d_data;       //!< Data stored on the GPU
			
	private:
			//! Allocates memory
			cudaError_t allocate(unsigned int num);
			//! Frees memory
			cudaError_t deallocate();
			//! Copies from the host to the device
			cudaError_t hostToDeviceCopy(Scalar *fx, Scalar *fy, Scalar *fz, Scalar *pe, Scalar *virial);
			//! Copies from the device to the host
			cudaError_t deviceToHostCopy(Scalar *fx, Scalar *fy, Scalar *fz, Scalar *pe, Scalar *virial);
			
			unsigned int m_num;                 //!< Number of particles in the simulation
			float4 *h_staging;                  //!< Host memory array for staging interleaved data
			
			friend class ForceCompute;
			};
=======
struct ForceDataArraysGPU
    {
    //! Zeros pointers
    ForceDataArraysGPU();
    
    gpu_force_data_arrays d_data;       //!< Data stored on the GPU
    
private:
    //! Allocates memory
    cudaError_t allocate(unsigned int num);
    //! Frees memory
    cudaError_t deallocate();
    //! Copies from the host to the device
    cudaError_t hostToDeviceCopy(Scalar *fx, Scalar *fy, Scalar *fz, Scalar *pe, Scalar *virial);
    //! Copies from the device to the host
    cudaError_t deviceToHostCopy(Scalar *fx, Scalar *fy, Scalar *fz, Scalar *pe, Scalar *virial);
    
    unsigned int m_num;                 //!< Number of particles in the simulation
    GPUArray<float4> h_staging;                  //!< Host memory array for staging interleaved data
    
    friend class ForceCompute;
    };
>>>>>>> .r3651
#endif
*/
//! Defines an interface for computing forces on each particle
/*! Derived classes actually provide the implementation that computes the forces.
    This base class exists so that some other part of the code can have a list of
    ForceComputes without needing to know what types of forces are being calculated.
    The base class also implements the data structures both on the CPU and GPU
    and handles the CPU <-> GPU copies of the force data.

    Like with ParticleData forces are stored with contiguous x,y,z components on the CPU
    and interleaved ones on the GPU.

    \b OpenMP <br>
    To aid in OpenMP force computations, the base class ForceCompute will optionally allocate a memory area to hold
    partial force sums for the forces computed by each CPU thread. The partial arrays will be indexed by the 
    Index2D m_index_thread_partial created in the allocateThreadPartial() function. 

    \ingroup computes
*/
class ForceCompute : public Compute
    {
    public:
        //! Constructs the compute
        ForceCompute(boost::shared_ptr<SystemDefinition> sysdef);
        
        //! Destructor
        virtual ~ForceCompute();
        
        //! Access the computed force data
        const ForceDataArrays& acquire();
        
#ifdef ENABLE_CUDA
        //! Access the computed force data on the GPU
        ForceDataArraysGPU& acquireGPU();
#endif

        //! Store the timestep size
        virtual void setDeltaT(Scalar dt)
            {
            m_deltaT = dt;
            }
        
        //! Computes the forces
        virtual void compute(unsigned int timestep);
        
        //! Benchmark the force compute
        virtual double benchmark(unsigned int num_iters);
        
        //! Total the potential energy
        Scalar calcEnergySum();

        //! Easy access to the force on a single particle
        Scalar3 getForce(unsigned int tag)
            {
						ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::read);
/*
#ifdef ENABLE_CUDA
            if (m_data_location == gpu)
                deviceToHostCopy();
#endif
*/
            unsigned int i = m_pdata->getRTag(tag);
            return make_scalar3(h_force.data[i].x,h_force.data[i].y,h_force.data[i].z);
            }
        //! Easy access to the virial on a single particle
        Scalar getVirial(unsigned int tag)
            {
						ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::read);
/*
#ifdef ENABLE_CUDA
            if (m_data_location == gpu)
                deviceToHostCopy();
#endif
*/
            unsigned int i = m_pdata->getRTag(tag);
            return h_virial.data[i];
            }
        //! Easy access to the energy on a single particle
        Scalar getEnergy(unsigned int tag)
            {
						ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::read);
/*
#ifdef ENABLE_CUDA
            if (m_data_location == gpu)
                deviceToHostCopy();
#endif
*/
            unsigned int i = m_pdata->getRTag(tag);
            return h_force.data[i].w;
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

        //! Allocates the force and virial partial data
        void allocateThreadPartial();
        
        Scalar m_deltaT;  //!< timestep size (required for some types of non-conservative forces)
				            
				GPUArray<Scalar4> m_force;			//!< m_force.x,m_force.y,m_force.z are the x,y,z components of the force, m_force.u is the PE
				GPUArray<Scalar>  m_virial;			//!< per-particle virial (see ForceDataArrays for definition)

        GPUArray<Scalar>  m_fx;     //!< x-component of the force
        GPUArray<Scalar>  m_fy;     //!< y-component of the force
        GPUArray<Scalar>  m_fz;     //!< z-component of the force
        GPUArray<Scalar>  m_pe;     //!< per-particle potential energy (see ForceDataArrays for definition)
        GPUArray<Scalar>  m_virial; //!< per-particle virial (see ForceDataArrays for definition)

        int m_nbytes;                   //!< stores the number of bytes of memory allocated
				
        GPUArray<Scalar4>  m_fdata_partial; //!< Stores partial force/pe for each CPU thread
        GPUArray<Scalar>  m_virial_partial; //!< Stores partial virial data summed for each CPU thread
        Index2D m_index_thread_partial;         //!< Indexer to index the above 2 arrays by (particle, thread)

        //! Connection to the signal notifying when particles are resorted
        boost::signals::connection m_sort_connection;   
        
        ForceDataArrays m_arrays;       //!< Structure-of-arrays for quick returning via acquire
        
        //! Simple type for identifying where the most up to date particle data is
        enum DataLocation
            {
            cpu,    //!< Particle data was last modified on the CPU
            cpugpu, //!< CPU and GPU contain identical data
            gpu     //!< Particle data was last modified on the GPU
            };
            
#ifdef ENABLE_CUDA
        DataLocation m_data_location;               //!< Where the neighborlist data currently lives

        ForceDataArraysGPU m_gpu_forces;            //!< Storage location for forces on the device
        
        //! Helper function to move data from the host to the device
        void hostToDeviceCopy();
        //! Helper function to move data from the device to the host
        void deviceToHostCopy();
#endif
        
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

