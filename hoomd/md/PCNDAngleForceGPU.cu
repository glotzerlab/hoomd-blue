// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "hip/hip_runtime.h"
#include "PCNDAngleForceGPU.cuh"

#include "hoomd/TextureTools.h"
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"

using namespace hoomd;

#include <assert.h>

// small number. cutoff for ignoring the angle as being ill defined.
#define SMALL Scalar(0.001)

/*! \file PCNDAngleForceGPU.cu
    \brief Defines GPU kernel code for calculating the PCND angle forces. Used by
    PCNDAngleForceComputeGPU.
*/

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Kernel for caculating PCND angle forces on the GPU
/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch pitch of 2D virial array
    \param N number of particles
    \param d_pos particle positions on the device
    \param box Box dimensions for periodic boundary condition handling
    \param alist Angle data to use in calculating the forces
    \param pitch Pitch of 2D angles list
    \param n_angles_list List of numbers of angles stored on the GPU
*/
__global__ void gpu_compute_PCND_angle_forces_kernel(Scalar4* d_force,
                                                     Scalar* d_virial,
						     const unsigned int* d_tag,
                                                     const size_t virial_pitch,
                                                     const unsigned int N,
                                                     const Scalar4* d_pos,
                                                     BoxDim box,
                                                     const group_storage<3>* alist,
                                                     const unsigned int* apos_list,
                                                     const unsigned int pitch,
                                                     const unsigned int* n_angles_list,
                                                     Scalar2* d_params,
                                                     uint64_t timestep,
                                                     uint64_t PCNDtimestep)
    {
    // start by identifying which particle we are to handle
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= N)
        return;
	
    // load in the length of the list for this thread (MEM TRANSFER: 4 bytes)
    int n_angles = n_angles_list[idx];

    // initialize the force to 0
    Scalar4 force_idx = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));
     
    // loop over all angles
    for (int angle_idx = 0; angle_idx < n_angles; angle_idx++)
        {		
        group_storage<3> cur_angle = alist[pitch * angle_idx + idx];
        
	int cur_angle_type = cur_angle.idx[2];
	int cur_angle_abc = apos_list[pitch * angle_idx + idx];
        // get the angle parameters (MEM TRANSFER: 8 bytes)
	Scalar2 params = __ldg(d_params + cur_angle_type);
	Scalar Xi = params.x;
	Scalar Tau = params.y;
				
        uint16_t seed = N;
	
	// read in the tag of our particle.
	unsigned int ptag = d_tag[idx];

	// Initialize the Random Number Generator and generate the 6 random numbers
	RandomGenerator rng(hoomd::Seed(RNGIdentifier::PCNDAngleForceCompute, timestep, seed),
			    hoomd::Counter(ptag));
	UniformDistribution<Scalar> uniform(Scalar(0), Scalar(1));

	Scalar a_x = uniform(rng);
	Scalar b_x = uniform(rng);
	Scalar a_y = uniform(rng);
	Scalar b_y = uniform(rng);
	Scalar a_z = uniform(rng);
	Scalar b_z = uniform(rng);
						
	if (cur_angle_abc == 1 && PCNDtimestep == 0)
	   {
           force_idx.x = Xi * sqrt(-2 * log(a_x)) * cosf(2 * 3.1415926535897 * b_x);
	   force_idx.y = Xi * sqrt(-2 * log(a_y)) * cosf(2 * 3.1415926535897 * b_y);
	   force_idx.z = Xi * sqrt(-2 * log(a_z)) * cosf(2 * 3.1415926535897 * b_z);

           force_idx.w = sqrt(force_idx.x * force_idx.x + force_idx.y * force_idx.y + force_idx.z * force_idx.z);
           d_force[idx] = force_idx;
	   }
        else if (cur_angle_abc == 1 && PCNDtimestep != 0)
	   {
           Scalar magx = d_force[idx].x;
	   Scalar magy = d_force[idx].y;
           Scalar magz = d_force[idx].z;

	   Scalar E = exp(-1 / Tau);
           Scalar hx = Xi * sqrt(-2 * (1 - E * E) * log(a_x)) * cosf(2 * 3.1415926535897 * b_x);
	   Scalar hy = Xi * sqrt(-2 * (1 - E * E) * log(a_y)) * cosf(2 * 3.1415926535897 * b_y);
	   Scalar hz = Xi * sqrt(-2 * (1 - E * E) * log(a_z)) * cosf(2 * 3.1415926535897 * b_z);
		
	   if (hx > Xi * sqrt(-2 * log(0.001)))
	      {
              hx = Xi * sqrt(-2 * log(0.001));
	      }
	   else if (hx <- Xi * sqrt(-2 * log(0.001)))
	      {
	      hx = -Xi * sqrt(-2 * log(0.001));
	      }
	   if (hy > Xi * sqrt(-2 * log(0.001)))
	      {
	      hy = Xi * sqrt(-2 * log(0.001));
	      }
           else if (hy <- Xi * sqrt(-2 * log(0.001)))
	      {
	      hy = -Xi * sqrt(-2 * log(0.001));
	      }
           if (hz > Xi * sqrt(-2 * log(0.001)))
	      {
	      hz = Xi * sqrt(-2 * log(0.001));
	      }
           else if (hz <- Xi * sqrt(-2 * log(0.001)))
	      {
	      hz= -Xi * sqrt(-2 * log(0.001));
	      }

	    force_idx.x = E * magx + hx;
	    force_idx.y = E * magy + hy;
	    force_idx.z = E * magz + hz;
        
	    force_idx.w = sqrt(force_idx.x * force_idx.x + force_idx.y * force_idx.y + force_idx.z * force_idx.z);
            d_force[idx] = force_idx;
	   }
        }
    }

/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch pitch of 2D virial array
    \param N number of particles
    \param d_pos device array of particle positions
    \param box Box dimensions (in GPU format) to use for periodic boundary conditions
    \param atable List of angles stored on the GPU
    \param pitch Pitch of 2D angles list
    \param n_angles_list List of numbers of angles stored on the GPU
    \param d_params Xi and Tau params packed as Scalar2 variables
    \param n_angle_types Number of angle types in d_params
    \param block_size Block size to use when performing calculations

    \returns Any error code resulting from the kernel launch
    \note Always returns hipSuccess in release builds to avoid the hipDeviceSynchronize()

    \a d_params should include one Scalar2 element per angle type. The x component contains Xi
    the RMS force magnitude and the y component contains Tau the correlation time.
*/
hipError_t gpu_compute_PCND_angle_forces(Scalar4* d_force,
                                         Scalar* d_virial,
                                         const size_t virial_pitch,
					 const unsigned int* d_tag,
                                         const unsigned int N,
                                         const Scalar4* d_pos,
                                         const BoxDim& box,
                                         const group_storage<3>* atable,
                                         const unsigned int* apos_list,
                                         const unsigned int pitch,
                                         const unsigned int* n_angles_list,
                                         Scalar2* d_params,
                                         unsigned int n_angle_types,
                                         int block_size,
                                         uint64_t timestep,
                                         uint64_t PCNDtimestep)
    {
    assert(d_params);
    assert(d_PCNDsr);
    assert(d_PCNDepow);
    
    static unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)gpu_compute_PCND_angle_forces_kernel);
    max_block_size = attr.maxThreadsPerBlock;
    
    unsigned int run_block_size = min(block_size, max_block_size);

    // setup the grid to run the kernel
    dim3 grid(N / run_block_size + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    // run the kernel
    hipLaunchKernelGGL((gpu_compute_PCND_angle_forces_kernel),
		        dim3(grid),
			dim3(threads),
			0,
			0,
			d_force,
                        d_virial,
			d_tag,
                        virial_pitch,
                        N,
                        d_pos,
                        box,
                        atable,
                        apos_list,
                        pitch,
                        n_angles_list,
                        d_params,
                        timestep,
                        PCNDtimestep);

    return hipSuccess;
    }

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
