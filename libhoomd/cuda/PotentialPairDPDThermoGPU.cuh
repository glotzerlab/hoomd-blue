/*! \file AllDriverPotentialPairExtGPU.cuh
    \brief Declares driver functions for computing all types of pair forces on the GPU from glotzer-hoomd-plugins
*/

#ifndef __POTENTIAL_PAIR_DPDTHERMO_CUH__
#define __POTENTIAL_PAIR_DPDTHERMO_CUH__

#include "ForceCompute.cuh"
#include "ParticleData.cuh"
#include "NeighborList.cuh"
#include "EvaluatorPairDPDThermo.h"
#include "Index1D.h"
#include <cassert>
#include "gpu_settings.h"


//! args struct for passing additional options to gpu_compute_dpd_forces
struct dpd_pair_args
    {
    int block_size;         //!< block size to execute on
    unsigned int seed;
    unsigned int timestep;
    float deltaT;
    float T;    
    };

#ifdef NVCC
//! Texture for reading particle positions
texture<float4, 1, cudaReadModeElementType> pdata_dpd_pos_tex;

//! Texture for reading particle velocities
texture<float4, 1, cudaReadModeElementType> pdata_dpd_vel_tex;

//! Kernel for calculating pair forces
/*! This kernel is called to calculate the pair forces on all N particles. Actual evaluation of the potentials and 
    forces for each pair is handled via the template class \a evaluator.

    \param force_data Device memory array to write calculated forces to
    \param pdata Particle data on the GPU to calculate forces on
    \param box Box dimensions used to implement periodic boundary conditions
    \param nlist Neigbhor list data on the GPU to use to calculate the forces
    \param d_params Parameters for the potential, stored per type pair
    \param d_rcutsq rcut squared, stored per type pair
    \param ntypes Number of types in the simulation
    
    \a d_params, and \a d_rcutsq must be indexed with an Index2DUpperTriangler(typei, typej) to access the
    unique value for that type pair. These values are all cached into shared memory for quick access, so a dynamic
    amount of shared memory must be allocatd for this kernel launch. The amount is
    (2*sizeof(float) + sizeof(typename evaluator::param_type)) * typpair_idx.getNumElements()
    
    Certain options are controlled via template parameters to avoid the performance hit when they are not enabled.
    \tparam evaluator EvaluatorPair class to evualuate V(r) and -delta V(r)/r
    \tparam shift_mode 0: No energy shifting is done. 1: V(r) is shifted to be 0 at rcut. 2: XPLOR switching is enabled
                       (See PotentialPair for a discussion on what that entails)
    
    <b>Implementation details</b>
    Each block will calculate the forces on a block of particles.
    Each thread will calculate the total force on one particle.
    The neighborlist is arranged in columns so that reads are fully coalesced when doing this.
*/
template< class evaluator >
__global__ void gpu_compute_dpd_forces_kernel(gpu_force_data_arrays force_data,
                                               gpu_pdata_arrays pdata,
                                               gpu_boxsize box,
                                               gpu_nlist_array nlist,
                                               typename evaluator::param_type *d_params,
                                               float *d_rcutsq,
                                               unsigned int d_seed,
                                               unsigned int d_timestep,
                                               float d_deltaT,
                                               float d_T,
                                               int ntypes)
    {
    Index2D typpair_idx(ntypes);
    const unsigned int num_typ_parameters = typpair_idx.getNumElements();

    // shared arrays for per type pair parameters
    extern __shared__ char s_data[];
    typename evaluator::param_type *s_params = 
        (typename evaluator::param_type *)(&s_data[0]);
    float *s_rcutsq = (float *)(&s_data[num_typ_parameters*sizeof(evaluator::param_type)]);
    
    // load in the per type pair parameters
    for (unsigned int cur_offset = 0; cur_offset < num_typ_parameters; cur_offset += blockDim.x)
        {
        if (cur_offset + threadIdx.x < num_typ_parameters)
            {
            s_rcutsq[cur_offset + threadIdx.x] = d_rcutsq[cur_offset + threadIdx.x];
            s_params[cur_offset + threadIdx.x] = d_params[cur_offset + threadIdx.x];
            }
        }
    __syncthreads();
    
    // start by identifying which particle we are to handle
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= pdata.local_num)
        return;
        
    // load in the length of the neighbor list (MEM_TRANSFER: 4 bytes)
    unsigned int n_neigh = nlist.n_neigh[idx];
    
    // read in the position of our particle. Texture reads of float4's are faster than global reads on compute 1.0 hardware
    // (MEM TRANSFER: 16 bytes)
    float4 posi = tex1Dfetch(pdata_dpd_pos_tex, idx);
    
    // read in the velocity of our particle. Texture reads of float4's are faster than global reads on compute 1.0 hardware
    // (MEM TRANSFER: 16 bytes)
    float4 veli = tex1Dfetch(pdata_dpd_vel_tex, idx);    
    
    // initialize the force to 0
    float4 force = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float virial = 0.0f;
    
    // prefetch neighbor index
    unsigned int cur_j = 0;
    unsigned int next_j = nlist.list[idx];
    
    // loop over neighbors
    // on pre Fermi hardware, there is a bug that causes rare and random ULFs when simply looping over n_neigh
    // the workaround (activated via the template paramter) is to loop over nlist.height and put an if (i < n_neigh)
    // inside the loop
    #if (__CUDA_ARCH__ < 200)
    for (int neigh_idx = 0; neigh_idx < nlist.height; neigh_idx++)
    #else
    for (int neigh_idx = 0; neigh_idx < n_neigh; neigh_idx++)
    #endif
        {
        #if (__CUDA_ARCH__ < 200)
        if (neigh_idx < n_neigh)
        #endif
            {
            // read the current neighbor index (MEM TRANSFER: 4 bytes)
            // prefetch the next value and set the current one
            cur_j = next_j;
            next_j = nlist.list[nlist.pitch*(neigh_idx+1) + idx];
            
            // get the neighbor's position (MEM TRANSFER: 16 bytes)
            float4 posj = tex1Dfetch(pdata_dpd_pos_tex, cur_j);

            // get the neighbor's position (MEM TRANSFER: 16 bytes)
            float4 velj = tex1Dfetch(pdata_dpd_vel_tex, cur_j);
                        
            // calculate dr (with periodic boundary conditions) (FLOPS: 3)
            float dx = posi.x - posj.x;
            float dy = posi.y - posj.y;
            float dz = posi.z - posj.z;
            
            // apply periodic boundary conditions: (FLOPS 12)
            dx -= box.Lx * rintf(dx * box.Lxinv);
            dy -= box.Ly * rintf(dy * box.Lyinv);
            dz -= box.Lz * rintf(dz * box.Lzinv);
            
            // calculate r squard (FLOPS: 5)
            float rsq = dx*dx + dy*dy + dz*dz;
            
            // calculate dv (FLOPS: 3)
            float dvx = veli.x - velj.x;
            float dvy = veli.y - velj.y;
            float dvz = veli.z - velj.z;            
            
            float dot = dx*dvx + dy*dvy + dz*dvz;
            
            // access the per type pair parameters
            unsigned int typpair = typpair_idx(__float_as_int(posi.w), __float_as_int(posj.w));
            float rcutsq = s_rcutsq[typpair];
            typename evaluator::param_type param = s_params[typpair];

            // 
            evaluator eval(rsq, rcutsq, param);
            
            // evaluate the potential
            float force_divr = 0.0f;
            float pair_eng = 0.0f;

            // Special Potential Pair DPD Requirements
            eval.set_seed_ij_timestep(d_seed,idx,cur_j,d_timestep);  
            eval.setDeltaT(d_deltaT);  
            eval.setRDotV(dot);
            eval.setT(d_T);            
            
            eval.evalForceEnergyThermo(force_divr, pair_eng);

            // calculate the virial (FLOPS: 3)
            virial += float(1.0/6.0) * rsq * force_divr;
            
            // add up the force vector components (FLOPS: 6)
            force.x += __fmul_rn(dx, force_divr);
            force.y += __fmul_rn(dy, force_divr);
            force.z += __fmul_rn(dz, force_divr);
            force.w += pair_eng;
            }
        }
        
    // potential energy per particle must be halved
    force.w *= 0.5f;
    // now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes)
    force_data.force[idx] = force;
    force_data.virial[idx] = virial;
    }

//! Kernel driver that computes lj forces on the GPU for LJForceComputeGPU
/*! \param force_data Device memory array to write calculated forces to
    \param pdata Particle data on the GPU to calculate forces on
    \param box Box dimensions used to implement periodic boundary conditions
    \param nlist Neigbhor list data on the GPU to use to calculate the forces
    \param d_params Parameters for the potential, stored per type pair
    \param d_rcutsq rcut squared, stored per type pair
    \param ntypes Number of types in the simulation
    \param args Additional options
    
    This is just a driver function for gpu_compute_dpd_forces_kernel(), see it for details.
*/
template< class evaluator >
cudaError_t gpu_compute_dpd_forces(const gpu_force_data_arrays& force_data,
                                    const gpu_pdata_arrays &pdata,
                                    const gpu_boxsize &box,
                                    const gpu_nlist_array &nlist,
                                    typename evaluator::param_type *d_params,
                                    float *d_rcutsq,
                                    int ntypes,
                                    const dpd_pair_args& args)
    {
    assert(d_params);
    assert(d_rcutsq);
    assert(ntypes > 0);
    
    // setup the grid to run the kernel
    dim3 grid( pdata.local_num / args.block_size + 1, 1, 1);
    dim3 threads(args.block_size, 1, 1);
    
    // bind the position texture
    pdata_dpd_pos_tex.normalized = false;
    pdata_dpd_pos_tex.filterMode = cudaFilterModePoint;
    cudaError_t error = cudaBindTexture(0, pdata_dpd_pos_tex, pdata.pos, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;

    // bind the velocity texture
    pdata_dpd_vel_tex.normalized = false;
    pdata_dpd_vel_tex.filterMode = cudaFilterModePoint;
    error = cudaBindTexture(0, pdata_dpd_vel_tex, pdata.vel, sizeof(float4) * pdata.N);
    if (error != cudaSuccess)
        return error;

    
    Index2D typpair_idx(ntypes);
    unsigned int shared_bytes = (2*sizeof(float) + sizeof(typename evaluator::param_type)) 
                                * typpair_idx.getNumElements();
    
    // run the kernel
    gpu_compute_dpd_forces_kernel<evaluator>
              <<<grid, threads, shared_bytes>>>(force_data, pdata, box, nlist, d_params, d_rcutsq, args.seed, args.timestep, args.deltaT, args.T, ntypes);

        
    if (!g_gpu_error_checking)
        {
        return cudaSuccess;
        }
    else
        {
        cudaThreadSynchronize();
        return cudaGetLastError();
        }
    }

#endif

#endif // __POTENTIAL_PAIR_DPDTHERMO_CUH__

