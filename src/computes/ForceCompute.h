/*
Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
Source Software License
Copyright (c) 2008 Ames Laboratory Iowa State University
All rights reserved.

Redistribution and use of HOOMD, in source and binary forms, with or
without modification, are permitted, provided that the following
conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names HOOMD's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
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
    \f$ \sum_i^N \mathrm{virial}_i = -\frac{1}{3} \sum_i^N \sum_{j>i} \vec{r}_{ij} \cdot \vec{f}_{ij} \f$

    \ingroup data_structs
*/
struct ForceDataArrays
    {
    //! Zeroes pointers
    ForceDataArrays();
    
    Scalar const * __restrict__ fx; //!< x-component of the force
    Scalar const * __restrict__ fy; //!< y-component of the force
    Scalar const * __restrict__ fz; //!< z-component of the force
    Scalar const * __restrict__ pe; //!< per-particle potential energy
    Scalar const * __restrict__ virial; //!< per-particle virial
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
struct ForceDataArraysGPU
    {
    //! Zeros pointers
    ForceDataArraysGPU();
    
    gpu_force_data_arrays d_data;       //!< Data stored on the GPU
    
private:
    //! Allocates memory
    cudaError_t allocate(unsigned int num_local, unsigned int local_start);
    //! Frees memory
    cudaError_t deallocate();
    //! Copies from the host to the device
    cudaError_t hostToDeviceCopy(Scalar *fx, Scalar *fy, Scalar *fz, Scalar *pe, Scalar *virial);
    //! Copies from the device to the host
    cudaError_t deviceToHostCopy(Scalar *fx, Scalar *fy, Scalar *fz, Scalar *pe, Scalar *virial);
    
    unsigned int m_num_local;           //!< Number of particles local to this GPU
    unsigned int m_local_start;         //!< Starting index of local data in global array
    float4 *h_staging;                  //!< Host memory array for staging interleaved data
    
    friend class ForceCompute;
    };
#endif

//! Defines an interface for computing forces on each particle
/*! Derived classes actually provide the implementation that computes the forces.
    This base class exists so that some other part of the code can have a list of
    ForceComputes without needing to know what types of forces are being calculated.
    The base class also implements the data structures both on the CPU and GPU
    and handles the CPU <-> GPU copies of the force data.

    Like with ParticleData forces are stored with contiguous x,y,z components on the CPU
    and interleaved ones on the GPU.
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
        vector<ForceDataArraysGPU>& acquireGPU();
#endif
        
        //! Computes the forces
        virtual void compute(unsigned int timestep);
        
        //! Benchmark the force compute
        virtual double benchmark(unsigned int num_iters);
        
        //! Total the potential energy
        Scalar calcEnergySum();
        
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
            
        Scalar * __restrict__ m_fx;     //!< x-component of the force
        Scalar * __restrict__ m_fy;     //!< y-component of the force
        Scalar * __restrict__ m_fz;     //!< z-component of the force
        Scalar * __restrict__ m_pe;     //!< per-particle potential energy (see ForceDataArrays for definition)
        Scalar * __restrict__ m_virial; //!< per-particle virial (see ForceDataArrays for definition)
        int m_nbytes;                   //!< stores the number of bytes of memory allocated
        
        //! Connection to the signal notifying when particles are resorted
        boost::signals::connection m_sort_connection;   
        
        ForceDataArrays m_arrays;       //!< Structure-of-arrays for quick returning via acquire
        
#ifdef ENABLE_CUDA
        //! Simple type for identifying where the most up to date particle data is
        enum DataLocation
            {
            cpu,    //!< Particle data was last modified on the CPU
            cpugpu, //!< CPU and GPU contain identical data
            gpu     //!< Particle data was last modified on the GPU
            };
            
        DataLocation m_data_location;               //!< Where the neighborlist data currently lives
        vector<ForceDataArraysGPU> m_gpu_forces;    //!< Storage location for forces on the device
        
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

