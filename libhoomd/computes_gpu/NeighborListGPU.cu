#include "NeighborListGPUBinned.cuh"

#include "NeighborListGPU.cuh"
#include <stdio.h>

__global__ void gpu_nlist_needs_update_check_new_kernel(unsigned int *d_result,
                                                        const float4 *d_last_pos,
                                                        const float4 *d_pos,
                                                        const unsigned int N,
                                                        const gpu_boxsize box,
                                                        const float maxshiftsq)
    {
    // each thread will compare vs it's old position to see if the list needs updating
    // if that is true, write a 1 to nlist_needs_updating
    // it is possible that writes will collide, but at least one will succeed and that is all that matters
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N)
        {
        float4 cur_pos = d_pos[idx];
        float4 last_pos = d_last_pos[idx];
        float dx = cur_pos.x - last_pos.x;
        float dy = cur_pos.y - last_pos.y;
        float dz = cur_pos.z - last_pos.z;

        dx = dx - box.Lx * rintf(dx * box.Lxinv);
        dy = dy - box.Ly * rintf(dy * box.Lyinv);
        dz = dz - box.Lz * rintf(dz * box.Lzinv);

        float drsq = dx*dx + dy*dy + dz*dz;
        
        if (drsq >= maxshiftsq)
            {
            *d_result = 1;
            }
        }
    }

cudaError_t gpu_nlist_needs_update_check_new(unsigned int *d_result,
                                             const float4 *d_last_pos,
                                             const float4 *d_pos,
                                             const unsigned int N,
                                             const gpu_boxsize& box,
                                             const float maxshiftsq)
    {
    int zero = 0;
    cudaMemcpy(d_result, &zero, sizeof(unsigned int), cudaMemcpyHostToDevice);
    
    unsigned int block_size = 128;
    int n_blocks = (int)ceil(float(N)/(float)block_size);
    gpu_nlist_needs_update_check_new_kernel<<<n_blocks, block_size>>>(d_result,
                                                                      d_last_pos,
                                                                      d_pos,
                                                                      N,
                                                                      box,
                                                                      maxshiftsq);
    
    return cudaSuccess;
    }

