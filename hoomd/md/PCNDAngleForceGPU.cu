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
    \brief Defines GPU kernel code for calculating the PCND angle forces. Used by PCNDAngleForceComputeGPU.
*/

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Texture for reading angle parameters
// scalar2_tex_t angle_params_tex;

//! Texture for reading angle CGCMM S-R parameters
// scalar2_tex_t angle_CGCMMsr_tex; // MISSING EPSILON!!! sigma=.x, rcut=.y

//! Texture for reading angle CGCMM Epsilon-pow/pref parameters
// scalar4_tex_t angle_CGCMMepow_tex; // now with EPSILON=.x, pow1=.y, pow2=.z, pref=.w

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
                                                     Scalar2* d_PCNDsr,
                                                     uint16_t* d_PCNDepow,
                                                     uint64_t timestep,
                                                     //float* devData,
                                                     uint64_t PCNDtimestep,
                                                     float* devCarryover)
    {
    // start by identifying which particle we are to handle
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= N)
        return;
    //curandState localState = state[idx];
    //curandState localState;  local state of generator
	
    // load in the length of the list for this thread (MEM TRANSFER: 4 bytes)
    int n_angles = n_angles_list[idx];

    // initialize the force to 0
    Scalar4 force_idx = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));
     
    //make counter for number of loops
    int counter = 0;
		
    // loop over all angles
    for (int angle_idx = 0; angle_idx < n_angles; angle_idx++)
	{
	int cur_angle_abc = apos_list[pitch * angle_idx + idx];
			
	if (cur_angle_abc == 1)
	    {
	    counter += 1;
	    //printf("forcew=%f\n",force_idx.w);
	    ////////////////////////////////////////////////////////////Get params
            group_storage<3> cur_angle = alist[pitch * angle_idx + idx];

	    //int cur_angle_x_idx = cur_angle.idx[0];
	    //int cur_angle_y_idx = cur_angle.idx[1];
	    int cur_angle_type = cur_angle.idx[2];
				
	    // get the angle parameters (MEM TRANSFER: 8 bytes)
	    //Scalar2 params = texFetchScalar2(d_params, angle_params_tex, cur_angle_type);
	    Scalar2 params = __ldg(d_params + cur_angle_type);
	    Scalar Xi = params.x; //K
	    Scalar Tau = params.y;//t_0
				
	    ////////////////// get sig params
	    //const Scalar2 cgSR = texFetchScalar2(d_CGCMMsr, angle_CGCMMsr_tex, cur_angle_type);
            const Scalar2 cgSR = __ldg(d_PCNDsr + cur_angle_type);

	    int number = cgSR.x;//sigma//number
	    //Scalar cgrcut = cgSR.y;
				
            /////////////get eps param
            //const Scalar4 cgEPOW = texFetchScalar4(d_CGCMMepow, angle_CGCMMepow_tex, cur_angle_type);
            const uint16_t cgEPOW = __ldg(d_PCNDepow + cur_angle_type);
	    uint16_t seed = cgEPOW;

	    // read in the tag of our particle.
	    unsigned int ptag = d_tag[idx];
	    // get the angle pow/pref parameters (MEM TRANSFER: 12 bytes)
	    //int seed = cgEPOW.x; //get parameter seed (epsilon)
            //Scalar cgpow1 = cgEPOW.y;
            //Scalar cgpow2 = cgEPOW.z;
            //Scalar cgpref = cgEPOW.w;
            //////////////////////////////////////////////////////////////

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
						
	    if (PCNDtimestep == 0)
	        {
		devCarryover[(number) * 6 + counter * 3] = Xi * sqrt(-2 * log(a_x)) * cosf(2 * 3.1415926535897 * b_x);
		devCarryover[(number) * 6 + 1 + counter * 3] = Xi * sqrt(-2 * log(a_y)) * cosf(2 * 3.1415926535897 * b_y);
		devCarryover[(number) * 6 + 2 + counter * 3] = Xi * sqrt(-2 * log(a_z)) * cosf(2 * 3.1415926535897 * b_z);				
		force_idx.x += devCarryover[(number) * 6 + counter * 3];
		force_idx.y += devCarryover[(number) * 6 + 1 + counter * 3];
		force_idx.z += devCarryover[(number) * 6 + 2 + counter * 3];
	        }
	    else if (PCNDtimestep != 0)
	        {
		Scalar magx = devCarryover[(number) * 6 + counter * 3];
		Scalar magy = devCarryover[(number) * 6 + 1 + counter * 3];
		Scalar magz = devCarryover[(number) * 6 + 2 + counter * 3];
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
	        if (idx == 70 && timestep <10)
		    {
		    Scalar carryx = devCarryover[(number) * 6 + 0 + counter * 3];
		    Scalar carryy = devCarryover[(number) * 6 + 1 + counter * 3];
		    Scalar carryz = devCarryover[(number) * 6 + 2 + counter * 3];
		    //printf("timestep = %i magx=%f carryover=%f counter=%i, hx=%f, R1=%f, R2=%f\n",timestep,magx,carryx,counter,hx,R1,R2);
		    //printf("timestep = %i magy=%f carryover=%f counter=%i, hy=%f, R3=%f, R4=%f\n",timestep,magy,carryy,counter,hy,R3,R4);
		    //printf("timestep = %i magz=%f carryover=%f counter=%i, hz=%f, R5=%f, R6=%f\n",timestep,magz,carryz,counter,hz,R5,R6);
		    printf("forcex=%f forcey=%f forcez=%f counter=%i\n",force_idx.x,force_idx.y,force_idx.z,counter);
		    } 
					
		devCarryover[(number) * 6 + counter * 3] = E * magx + hx;
		devCarryover[(number) * 6 + 1 + counter * 3] = E * magy + hy;
		devCarryover[(number) * 6 + 2 + counter * 3] = E * magz + hz;
		force_idx.x += devCarryover[(number) * 6 + counter * 3];
		force_idx.y += devCarryover[(number) * 6 + 1 + counter * 3];
		force_idx.z += devCarryover[(number) * 6 + 2 + counter * 3];
		}
	    }
        }
        force_idx.w += sqrt(force_idx.x * force_idx.x + force_idx.y * force_idx.y + force_idx.z * force_idx.z);
        d_force[idx] = force_idx;
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
    \param d_params K and t_0 params packed as Scalar2 variables
    \param d_PCNDsr sigma, and rcut packed as Scalar2 variables
    \param d_PCNDepow epsilon, pow1, pow2, and prefactor packed as Scalar4 variables
    \param n_angle_types Number of angle types in d_params
    \param block_size Block size to use when performing calculations
    \param compute_capability Compute capability of the device (200, 300, 350, ...)

    \returns Any error code resulting from the kernel launch
    \note Always returns hipSuccess in release builds to avoid the hipDeviceSynchronize()

    \a d_params should include one Scalar2 element per angle type. The x component contains K
    the spring constant and the y component contains t_0 the equilibrium angle.
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
                                         Scalar2* d_PCNDsr,
                                         uint16_t* d_PCNDepow,
                                         unsigned int n_angle_types,
                                         int block_size,
                                         //const unsigned int compute_capability,
                                         uint64_t timestep,
                                         //float* devData,
                                         uint64_t PCNDtimestep,
                                         float* devCarryover)
    {
    assert(d_params);
    assert(d_PCNDsr);
    assert(d_PCNDepow);
    
    //float *lookupArray;
    //cudaMemcpyToSymbol(lookupArray,(void*)hostData,100*sizeof(float),0);
    
    static unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)gpu_compute_PCND_angle_forces_kernel);
    max_block_size = attr.maxThreadsPerBlock;
    
    unsigned int run_block_size = min(block_size, max_block_size);

    // setup the grid to run the kernel
    dim3 grid(N / run_block_size + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    // bind the textures on pre sm 35 arches
    /* if (compute_capability < 350)
        {
        cudaError_t error = cudaBindTexture(0, angle_params_tex, d_params, sizeof(Scalar2) * n_angle_types);
        if (error != cudaSuccess)
            return error;

        error = cudaBindTexture(0, angle_CGCMMsr_tex, d_CGCMMsr, sizeof(Scalar2) * n_angle_types);
        if (error != cudaSuccess)
            return error;

        error = cudaBindTexture(0, angle_CGCMMepow_tex, d_CGCMMepow, sizeof(Scalar4) * n_angle_types);
        if (error != cudaSuccess)
            return error;
        }
    */
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
                        d_PCNDsr,
                        d_PCNDepow,
                        timestep,
                        //devData,
                        PCNDtimestep,
                        devCarryover);

    return hipSuccess;
    }

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
