#include <cufft.h>

#include "CommunicatorGridGPU.cuh"
//! Define plus operator for complex data type (needed by CommunicatorMesh)
__device__ inline cufftComplex operator + (cufftComplex& lhs, const cufftComplex& rhs)
    {
    cufftComplex res;
    res.x = lhs.x + rhs.x;
    res.y = lhs.y + rhs.y;
    return res;
    }

template<typename T>
__global__ void gpu_gridcomm_scatter_send_cells_kernel (
    unsigned int n_send_cells,
    unsigned int *d_send_idx,
    const T *d_grid,
    T *d_send_buf)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_send_cells) return;
    d_send_buf[idx] = d_grid[d_send_idx[idx]];
    }

template<typename T, bool add_outer>
__global__ void gpu_gridcomm_scatter_add_recv_cells_kernel (
    unsigned int n_unique_recv_cells,
    const T *d_recv_buf,
    T *d_grid,
    const unsigned int *d_cell_recv,
    const unsigned int *d_cell_recv_begin,
    const unsigned int *d_cell_recv_end,
    const unsigned int *d_recv_idx)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_unique_recv_cells) return;

    unsigned int begin = d_cell_recv_begin[idx];
    unsigned int end = d_cell_recv_end[idx];

    T val = d_recv_buf[d_cell_recv[begin]];

    // add together multiple received cells
    for (unsigned int i = begin+1; i < end; i++)
        val = val + d_recv_buf[d_cell_recv[i]];

    unsigned int recv_cell = d_recv_idx[d_cell_recv[begin]];
    if (add_outer)
        {
        // add to grid
        d_grid[recv_cell] = d_grid[recv_cell] + val;
        }
    else
        {
        // write out to grid
        d_grid[recv_cell] = val;
        }
    }

template<typename T>
void gpu_gridcomm_scatter_send_cells(
    unsigned int n_send_cells,
    unsigned int *d_send_idx,
    const T *d_grid,
    T *d_send_buf)
    {
    unsigned int block_size = 512;
    unsigned int n_blocks = n_send_cells/block_size + 1;

    gpu_gridcomm_scatter_send_cells_kernel<T><<<n_blocks, block_size>>>(
        n_send_cells,
        d_send_idx,
        d_grid,
        d_send_buf);
    }

template<typename T>
void gpu_gridcomm_scatter_add_recv_cells(
    unsigned int n_unique_recv_cells,
    const T *d_recv_buf,
    T *d_grid,
    const unsigned int *d_cell_recv,
    const unsigned int *d_cell_recv_begin,
    const unsigned int *d_cell_recv_end,
    const unsigned int *d_recv_idx,
    bool add_outer)
    {
    unsigned int block_size = 512;
    unsigned int n_blocks = n_unique_recv_cells/block_size + 1;

    if (add_outer)
        {
        gpu_gridcomm_scatter_add_recv_cells_kernel<T,true><<<n_blocks,block_size>>>(
            n_unique_recv_cells,
            d_recv_buf,
            d_grid,
            d_cell_recv,
            d_cell_recv_begin,
            d_cell_recv_end,
            d_recv_idx);
        }
    else
        {
        gpu_gridcomm_scatter_add_recv_cells_kernel<T,false><<<n_blocks,block_size>>>(
            n_unique_recv_cells,
            d_recv_buf,
            d_grid,
            d_cell_recv,
            d_cell_recv_begin,
            d_cell_recv_end,
            d_recv_idx);
        }
    }

//! Template instantiation for cufftComplex
template void gpu_gridcomm_scatter_send_cells<cufftComplex>(
    unsigned int n_send_cells,
    unsigned int *d_send_idx,
    const cufftComplex *d_grid,
    cufftComplex *d_send_buf);

template void gpu_gridcomm_scatter_add_recv_cells<cufftComplex>(
    unsigned int n_unique_recv_cells,
    const cufftComplex *d_recv_buf,
    cufftComplex *d_grid,
    const unsigned int *d_cell_recv,
    const unsigned int *d_cell_recv_begin,
    const unsigned int *d_cell_recv_end,
    const unsigned int *d_recv_idx,
    bool add_outer);

//! Template instantiation for Scalar
template void gpu_gridcomm_scatter_send_cells<Scalar>(
    unsigned int n_send_cells,
    unsigned int *d_send_idx,
    const Scalar *d_grid,
    Scalar *d_send_buf);

template void gpu_gridcomm_scatter_add_recv_cells<Scalar>(
    unsigned int n_unique_recv_cells,
    const Scalar *d_recv_buf,
    Scalar *d_grid,
    const unsigned int *d_cell_recv,
    const unsigned int *d_cell_recv_begin,
    const unsigned int *d_cell_recv_end,
    const unsigned int *d_recv_idx,
    bool add_outer);

//! Template instantiation for unsigned int
template void gpu_gridcomm_scatter_send_cells<unsigned int>(
    unsigned int n_send_cells,
    unsigned int *d_send_idx,
    const unsigned int *d_grid,
    unsigned int *d_send_buf);

template void gpu_gridcomm_scatter_add_recv_cells<unsigned int>(
    unsigned int n_unique_recv_cells,
    const unsigned int *d_recv_buf,
    unsigned int *d_grid,
    const unsigned int *d_cell_recv,
    const unsigned int *d_cell_recv_begin,
    const unsigned int *d_cell_recv_end,
    const unsigned int *d_recv_idx,
    bool add_outer);
