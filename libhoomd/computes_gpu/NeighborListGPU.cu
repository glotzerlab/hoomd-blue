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

const unsigned int FILTER_BATCH_SIZE = 4;

__global__ void gpu_nlist_filter_kernel(unsigned int *d_n_neigh,
                                        unsigned int *d_nlist,
                                        const Index2D nli,
                                        const unsigned int *d_n_ex,
                                        const unsigned int *d_ex_list,
                                        const Index2D exli,
                                        const unsigned int N,
                                        const unsigned int ex_start)
    {
    // compute the particle index this thread operates on
    const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    // quit now if this thread is processing past the end of the particle list
    if (idx >= N)
        return;
    
    const unsigned int n_neigh = d_n_neigh[idx];
    const unsigned int n_ex = d_n_ex[idx];
    unsigned int new_n_neigh = 0;
    
    // quit now if the ex_start flag is past the end of n_ex
    if (ex_start >= n_ex)
        return;
    
    // count the number of exclusions to process in this thread
    const unsigned int n_ex_process = n_ex - ex_start;
    
    // load the exclusion list into "local" memory - fully unrolled loops should dump this into registers
    unsigned int l_ex_list[FILTER_BATCH_SIZE];
    #pragma unroll
    for (unsigned int cur_ex_idx = 0; cur_ex_idx < FILTER_BATCH_SIZE; cur_ex_idx++)
        {
        if (cur_ex_idx < n_ex_process)
            l_ex_list[cur_ex_idx] = d_ex_list[exli(idx, cur_ex_idx + ex_start)];
        else
            l_ex_list[cur_ex_idx] = 0xffffffff;
        }
    
    // loop over the list, regenerating it as we go
    for (unsigned int cur_neigh_idx = 0; cur_neigh_idx < n_neigh; cur_neigh_idx++)
        {
        unsigned int cur_neigh = d_nlist[nli(idx, cur_neigh_idx)];
        
        // test if excluded
        bool excluded = false;
        #pragma unroll
        for (unsigned int cur_ex_idx = 0; cur_ex_idx < FILTER_BATCH_SIZE; cur_ex_idx++)
            {
            if (cur_neigh == l_ex_list[cur_ex_idx])
                excluded = true;
            }
        
        // add it back to the list if it is not excluded
        if (!excluded)
            {
            if (new_n_neigh != cur_neigh_idx)
                d_nlist[nli(idx, new_n_neigh)] = cur_neigh;
            new_n_neigh++;
            }
        }
    
    // update the number of neighbors
    d_n_neigh[idx] = new_n_neigh;
    }

cudaError_t gpu_nlist_filter(unsigned int *d_n_neigh,
                             unsigned int *d_nlist,
                             const Index2D& nli,
                             const unsigned int *d_n_ex,
                             const unsigned int *d_ex_list,
                             const Index2D& exli,
                             const unsigned int N)
    {
    // determine parameters for kernel launch
    unsigned int block_size = 192;
    int n_blocks = (int)ceil(float(N)/(float)block_size);
    
    // split the processing of the full exclusion list up into a number of batches
    unsigned int n_batches = (unsigned int)ceil(float(exli.getH())/(float)FILTER_BATCH_SIZE);
    unsigned int ex_start = 0;
    for (unsigned int batch = 0; batch < n_batches; batch++)
        {
        gpu_nlist_filter_kernel<<<n_blocks, block_size>>>(d_n_neigh,
                                                          d_nlist,
                                                          nli,
                                                          d_n_ex,
                                                          d_ex_list,
                                                          exli,
                                                          N,
                                                          ex_start);
        
        ex_start += FILTER_BATCH_SIZE;
        }
    
    return cudaSuccess;
    }
