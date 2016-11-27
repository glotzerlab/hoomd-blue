#include <cuda.h>
#include "dfft_cuda.cuh"

// redistribute between group-cyclic distributions with different cycles 
// c0 <= c1, n-dimensional version
__global__ void gpu_b2c_pack_kernel_nd(unsigned int local_size,
                                    int *d_c0,
                                    int *d_c1,
                                    int ndim,
                                    int *d_embed,
                                    int *d_length,
                                    int row_m,
                                    const cuda_cpx_t *local_data,
                                    cuda_cpx_t *send_data
                                    )
    {
    extern __shared__ int nidx_shared[];
    // index of local component
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // do not read beyond end of array
    if (idx >= local_size) return;

    // determine local coordinate tuple
    int *nidx = &nidx_shared[threadIdx.x*ndim];

    int tmp = idx;
    for (int i = ndim-1; i >= 0; --i)
        {
        int embed = d_embed[i];
        nidx[i] = tmp % embed;
        tmp /= embed;
        int l = d_length[i];
        if (nidx[i] >= l) return;
        }


    // determine index of packet and index in packet
    int lidx = 0;
    int size_tot = 1;
    tmp = 1;
    int packet_idx = 0; 
    for (int i = 0; i <ndim; ++i)
        {
        int c0 = d_c0[i];
        int c1 = d_c1[i];
        int ratio = c1/c0;
        int l = d_length[i];
        int size = ((l/ratio > 1) ? (l/ratio) : 1);

        lidx *= size;
        lidx += nidx[i]/ratio; // index in packet in column-major
        int num_packets = l/size;
        if (!row_m)
            {
            packet_idx *= num_packets;
            packet_idx += (nidx[i] % ratio);
            }
        else
            {
            packet_idx += tmp*(nidx[i] % ratio);
            tmp *= num_packets;
            }
        size_tot *= size;
        }

    send_data[packet_idx*size_tot+lidx] = local_data[idx];
    }

// redistribute between group-cyclic distributions with different cycles
// c0 <= c1, n-dimensional version, unpack kernel
__global__ void gpu_b2c_unpack_kernel_nd(unsigned int local_size,
                                    int *d_c0,
                                    int *d_c1,
                                    int ndim,
                                    int *d_embed,
                                    int *d_length,
                                    int row_m,
                                    const cuda_cpx_t *recv_data,
                                    cuda_cpx_t *local_data
                                    )
    {
    // index of local component
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // do not read beyond end of array
    if (idx >= local_size) return;

    int packet_idx = 0;

    int tmp = 1; 
    int tmp_packet = 1;

    // index in packet
    int lidx = 0;

    int size = 1; // packet size
    int j = idx;
    for (int i = ndim-1; i>= 0; --i)
        {
        int l = d_length[i];
        int c0 = d_c0[i];
        int c1 = d_c1[i];
        int ratio = c1/c0;
        int embed = d_embed[i];
        
        // determine local index in current dimension
        int j1 = j % embed;
        j /= embed;
        if (j1 >= l) return; // do not fill outer embedding layer
        
        // determine packet idx along current dimension
        // and index in packet
        int lpidx;
        int num_packets;
        int lidxi; // index in packet
        int sizei;
        if (l >= ratio)
            {
            num_packets = ratio;
            sizei = l/ratio;
            lidxi = j1 % sizei;
            lpidx = j1 / sizei;
            }
        else
            {
            lpidx = j1;
            num_packets = l;
            sizei = 1;
            lidxi = 0;
            }
        
        if (!row_m)
            {
            /* packets in column major order */
            packet_idx += tmp*lpidx;
            tmp *= num_packets;
            }
        else
            {
            /* packets in row-major order */
            packet_idx *= num_packets;
            packet_idx += lpidx;
            }

        // inside packet: column-major
        lidx += tmp_packet*lidxi;
        tmp_packet *= sizei;
        size *= sizei;
        }

    local_data[idx] = recv_data[packet_idx*size + lidx];
    }

void gpu_b2c_pack_nd(unsigned int local_size,
                     int *d_c0,
                     int *d_c1,
                     int ndim,
                     int *d_embed,
                     int *d_length,
                     int row_m,
                     const cuda_cpx_t *local_data,
                     cuda_cpx_t *send_data
                     )
    {
    unsigned int block_size =512;
    unsigned int n_blocks = local_size/block_size;
    if (local_size % block_size) n_blocks++;

    int shared_size = ndim*block_size*sizeof(int);
    gpu_b2c_pack_kernel_nd<<<n_blocks, block_size,shared_size>>>(local_size,
                                                  d_c0,
                                                  d_c1,
                                                  ndim,
                                                  d_embed,
                                                  d_length,
                                                  row_m,
                                                  local_data,
                                                  send_data);
    }

void gpu_b2c_unpack_nd(unsigned int local_size,
                     int *d_c0,
                     int *d_c1,
                     int ndim,
                     int *d_embed,
                     int *d_length,
                     int row_m,
                     const cuda_cpx_t *recv_data,
                     cuda_cpx_t *local_data
                     )
    {
    unsigned int block_size =512;
    unsigned int n_blocks = local_size/block_size;
    if (local_size % block_size) n_blocks++;

    gpu_b2c_unpack_kernel_nd<<<n_blocks, block_size>>>(local_size,
                             d_c0,
                             d_c1,
                             ndim,
                             d_embed,
                             d_length,
                             row_m,
                             recv_data,
                             local_data);
    } 

__device__ unsigned int bit_reverse(unsigned int in,
                                    unsigned int pow_of_two)
    {
    unsigned int rev = 0;
    unsigned int pow = pow_of_two;
    for (unsigned int i = 0; pow > 1; i++)
        {
         pow /= 2;
         rev *= 2;
         rev += ((in & (1 << i)) ? 1 : 0);
        }
    return rev;
    }

// redistribute between group-cyclic distributions with different cycles 
// c0 >= c1, n-dimensional version
__global__ void gpu_c2b_pack_kernel_nd(unsigned int local_size,
                                    int *d_c0,
                                    int *d_c1,
                                    int ndim,
                                    int *d_embed,
                                    int *d_length,
                                    int row_m,
                                    int *d_pdim,
                                    int *d_rev_j1,
                                    int *d_rev,
                                    const cuda_cpx_t *local_data,
                                    cuda_cpx_t *send_data
                                    )
    {
    extern __shared__ int nidx_shared[];
    // index of local component
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // do not read beyond end of array
    if (idx >= local_size) return;

    // determine local coordinate tuple
    int *nidx = &nidx_shared[threadIdx.x*ndim];

    int tmp = idx;
    for (int i = ndim-1; i >= 0; --i)
        {
        int embed = d_embed[i];
        nidx[i] = tmp % embed;
        tmp /= embed;
        int l = d_length[i];
        if (nidx[i] >= l) return;

       }


    // determine index of packet and index in packet
    int lidx = 0;
    int size_tot = 1;
    tmp = 1;
    int packet_idx = 0; 
    for (int i = 0; i <ndim; ++i)
        {
        int c0 = d_c0[i];
        int c1 = d_c1[i];
        int ratio = c0/c1;
        int l = d_length[i];
        int size;
        int j1 = nidx[i];
        int lpidx;
        int num_packets;
        int rev_j1 = d_rev_j1[i];
        int rev_global = d_rev[i];
        if (rev_j1) j1= bit_reverse(j1, l);
        if (! rev_global)
            {
            size = ((l/ratio > 1) ? (l/ratio) : 1);
            lidx *= size;
            lidx += (j1%size); // index in packet in column-major
            num_packets = l/size;
            lpidx = j1/size;
            }
        else
            {
            // global bitreversal
            int p = d_pdim[i];
            if (p/c1 > c0)
                {
                size = ((p/c1 <= l*c0) ? (l*c0*c1/p) : 1);
                num_packets = l/size;

                // inside packet: column major
                lidx *= size;
                int lidxi = bit_reverse(j1/num_packets,size);
                lidx += lidxi;
                lpidx = bit_reverse(j1 %num_packets,num_packets);
                }
            else
                {
                size = ((l*p/c1 >= c0) ? l*p/c1/c0 : 1);
                num_packets = l/size;
                int lidxi = bit_reverse(j1%size,size);
                lidx *= size;
                lidx += lidxi;
                lpidx = bit_reverse(j1 / size,num_packets);
                }
            }

        if (!row_m)
            {
            packet_idx *= num_packets;
            packet_idx += lpidx;
            }
        else
            {
            packet_idx += tmp*lpidx;
            tmp *= num_packets;
            }

        size_tot *= size;
        }

    send_data[packet_idx*size_tot+lidx] = local_data[idx];
    }

// redistribute between group-cyclic distributions with different cycles
// c0 >= c1, n-dimensional version, unpack kernel
__global__ void gpu_c2b_unpack_kernel_nd(unsigned int local_size,
                                    int *d_c0,
                                    int *d_c1,
                                    int ndim,
                                    int *d_embed,
                                    int *d_length,
                                    int row_m,
                                    int *d_pdim,
                                    int *d_rev,
                                    int *d_rev_partial,
                                    const cuda_cpx_t *recv_data,
                                    cuda_cpx_t *local_data
                                    )
    {
    // index of local component
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // do not read beyond end of array
    if (idx >= local_size) return;

    int packet_idx = 0;

    int tmp = 1; 
    int tmp_packet = 1;

    // index in packet
    int lidx = 0;

    int size = 1; // packet size
    int j = idx;
    for (int i = ndim-1; i>= 0; --i)
        {
        int l = d_length[i];
        int c0 = d_c0[i];
        int c1 = d_c1[i];
        int ratio = c0/c1;
        int embed = d_embed[i];
        
        // determine local index in current dimension
        int j1 = j % embed;
        j /= embed;
        if (j1 >= l) return; // do not fill outer embedding layer
        
        // determine packet idx along current dimension
        // and index in packet
        int lpidx;
        int num_packets;
        int lidxi; // index in packet
        int sizei;
        int rev = d_rev[i];
        if (!rev)
            {
            if (l >= ratio)
                {
                num_packets = ratio;
                sizei = l/ratio;
                lidxi = j1 /ratio;
                lpidx = (j1 % ratio);
                }
            else
                {
                lpidx = j1;
                num_packets = l;
                sizei = 1;
                lidxi = 0;
                }
            }
        else
            {
            // global bit reversal
            int p = d_pdim[i];
            if (c0 < p/c1)
                {
                // this section is usually not called during a dfft
                sizei = ((p/c1 <= l*c0) ? (l*c0*c1/p) : 1);
                num_packets = l/sizei;
                lidxi = j1 / num_packets;
                lpidx = bit_reverse(j1 % num_packets,num_packets);
                }
            else
                {
                sizei = ((l*p/c1 >= c0) ? l*p/c1/c0 : 1);
                num_packets = l/sizei;
                lidxi = j1 % sizei;
                int rev_partial = d_rev_partial[i];
                if (rev_partial)
                    lpidx = j1 / sizei;
                else
                    lpidx = bit_reverse(j1 / sizei, num_packets);
                }
            }

        if (!row_m)
            {
            /* packets in column major order */
            packet_idx += tmp*lpidx;
            tmp *= num_packets;
            }
        else
            {
            /* packets in row-major order */
            packet_idx *= num_packets;
            packet_idx += lpidx;
            }

        // inside packet: column-major
        lidx += tmp_packet*lidxi;
        tmp_packet *= sizei;
        size *= sizei;
        }

    local_data[idx] = recv_data[packet_idx*size + lidx];
    }

void gpu_c2b_pack_nd(unsigned int local_size,
                     int *d_c0,
                     int *d_c1,
                     int ndim,
                     int *d_embed,
                     int *d_length,
                     int row_m,
                     int *d_pdim,
                     int *d_rev_j1,
                     int *d_rev,
                     const cuda_cpx_t *local_data,
                     cuda_cpx_t *send_data
                     )
    {
    unsigned int block_size =512;
    unsigned int n_blocks = local_size/block_size;
    if (local_size % block_size) n_blocks++;

    int shared_size = ndim*block_size*sizeof(int);
    gpu_c2b_pack_kernel_nd<<<n_blocks, block_size,shared_size>>>(local_size,
                                                  d_c0,
                                                  d_c1,
                                                  ndim,
                                                  d_embed,
                                                  d_length,
                                                  row_m,
                                                  d_pdim,
                                                  d_rev_j1,
                                                  d_rev,
                                                  local_data,
                                                  send_data);
    }

void gpu_c2b_unpack_nd(unsigned int local_size,
                     int *d_c0,
                     int *d_c1,
                     int ndim,
                     int *d_embed,
                     int *d_length,
                     int row_m,
                     int *d_pdim,
                     int *d_rev,
                     int *d_rev_partial,
                     const cuda_cpx_t *recv_data,
                     cuda_cpx_t *local_data
                     )
    {
    unsigned int block_size =512;
    unsigned int n_blocks = local_size/block_size;
    if (local_size % block_size) n_blocks++;

    gpu_c2b_unpack_kernel_nd<<<n_blocks, block_size>>>(local_size,
                             d_c0,
                             d_c1,
                             ndim,
                             d_embed,
                             d_length,
                             row_m,
                             d_pdim,
                             d_rev,
                             d_rev_partial,
                             recv_data,
                             local_data);
    } 


// redistribute between group-cyclic distributions with different cycles
// c0 <= c1
__global__ void gpu_b2c_pack_kernel(unsigned int local_size,
                                    unsigned int ratio,
                                    unsigned int size,
                                    unsigned int npackets,
                                    unsigned int stride,
                                    cuda_cpx_t *local_data,
                                    cuda_cpx_t *send_data
                                    )
    {
    // index of local component
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // do not read beyond end of array
    if (idx >= local_size) return;

    unsigned int j = (idx/stride) % npackets; // packet number
    unsigned int r = (idx/stride - j)/ratio; // index in packet

    unsigned int offset = j*size;
    send_data[offset + r*stride + (idx%stride)] = local_data[idx];
    }

void gpu_b2c_pack(unsigned int local_size,
                  unsigned int ratio,
                  unsigned int size,
                  unsigned int npackets,
                  unsigned int stride,
                  cuda_cpx_t *local_data,
                  cuda_cpx_t *send_data)
    {
    unsigned int block_size =512;
    unsigned int n_blocks = local_size/block_size;
    if (local_size % block_size) n_blocks++;

    gpu_b2c_pack_kernel<<<n_blocks, block_size>>>(local_size,
                                                  ratio,
                                                  size,
                                                  npackets,
                                                  stride,
                                                  local_data,
                                                  send_data);
    }

// apply twiddle factors
__global__ void gpu_twiddle_kernel(unsigned int local_size,
                                   const unsigned int length,
                                   const unsigned int stride,
                                   float alpha,
                                   cuda_cpx_t *d_in,
                                   cuda_cpx_t *d_out,
                                   int inv)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= local_size) return;

    int j = idx/stride;
    if (j >= length) return;
    float theta = -2.0f * float(M_PI) * alpha/(float) length;
    cuda_cpx_t w;
    CUDA_RE(w) = cosf((float)j*theta);
    CUDA_IM(w) = sinf((float)j*theta);

    cuda_cpx_t in = d_in[idx];
    cuda_cpx_t out;
    float sign = inv ? -1.0f : 1.0f;

    w.y *= sign;

    CUDA_RE(out) = CUDA_RE(in) * CUDA_RE(w) - CUDA_IM(in) * CUDA_IM(w);
    CUDA_IM(out) = CUDA_RE(in) * CUDA_IM(w) + CUDA_IM(in) * CUDA_RE(w); 

    d_out[idx] = out;
    }

// apply twiddle factors (n-dimensional version)
__global__ void gpu_twiddle_kernel_nd(unsigned int local_size,
                                   int ndim,
                                   int *d_embed,
                                   int *d_length,
                                   float *d_alpha,
                                   cuda_cpx_t *d_in,
                                   cuda_cpx_t *d_out,
                                   int inv)
    {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= local_size) return;

    // complex-multiply twiddle factors for all dimensions
    int tmp = idx;

    float theta = 0.0f;
    for (int i = ndim-1; i>=0;--i)
        {
        int embed = d_embed[i];
        int length = d_length[i];

        int j = tmp % embed;
        if (j >= length) return;
        tmp /= embed;

        float alpha = d_alpha[i];
        theta -= (float)j*2.0f * float(M_PI) * alpha/(float) length;
        }

    cuda_cpx_t w;
    CUDA_RE(w) = cosf(theta);
    CUDA_IM(w) = sinf(theta);

    cuda_cpx_t out;
    float sign = inv ? -1.0f : 1.0f;

    w.y *= sign;

    cuda_cpx_t in = d_in[idx];
    CUDA_RE(out) = CUDA_RE(in) * CUDA_RE(w) - CUDA_IM(in) * CUDA_IM(w);
    CUDA_IM(out) = CUDA_RE(in) * CUDA_IM(w) + CUDA_IM(in) * CUDA_RE(w); 

    d_out[idx] = out;        
    }

void gpu_twiddle(unsigned int local_size,
                 const unsigned int length,
                 const unsigned int stride,
                 float alpha,
                 cuda_cpx_t *d_in,
                 cuda_cpx_t *d_out,
                 int inv)
    {
    unsigned int block_size =512;
    unsigned int n_block = local_size/block_size;
    if (local_size % block_size ) n_block++;

    gpu_twiddle_kernel<<<n_block, block_size>>>(local_size,
                                                length,
                                                stride,
                                                alpha,
                                                d_in,
                                                d_out,
                                                inv);
}

void gpu_twiddle_nd(unsigned int local_size,
                 int ndim,
                 int *d_embed,
                 int *d_length,
                 float *d_alpha,
                 cuda_cpx_t *d_in,
                 cuda_cpx_t *d_out,
                 int inv)
    {
    unsigned int block_size =512;
    unsigned int n_block = local_size/block_size;
    if (local_size % block_size ) n_block++;

    gpu_twiddle_kernel_nd<<<n_block, block_size>>>(local_size, ndim, d_embed,
        d_length, d_alpha, d_in, d_out, inv);
    }
 
__global__ void gpu_c2b_unpack_kernel(const unsigned int local_size,
                                      const unsigned int length,
                                      const unsigned int c0,
                                      const unsigned int c1, 
                                      const unsigned int size,
                                      const unsigned int j0,
                                      const unsigned int stride,
                                      int rev,
                                      cuda_cpx_t *d_local_data,
                                      const cuda_cpx_t *d_scratch)
    {
    unsigned int idx = blockDim.x*blockIdx.x+threadIdx.x;

    if (idx >= local_size) return;

    // source processor
    int r = idx/size; // packet index
    int j1, j1_offset, del;
    int j0_remote = j0 + r*c1;
    if (rev && (length >= c0))
        {
        j1_offset = j0_remote*length/c0;
        del = 1;
        }
    else
        {
        j1_offset = j0_remote/c1;
        del = c0/c1;
        }

    // local index
    j1 = j1_offset + ((idx%size)/stride)*del;
    
    d_local_data[j1*stride+idx%stride] = d_scratch[idx];
    }

void gpu_c2b_unpack(const unsigned int local_size,
                    const unsigned int length,
                    const unsigned int c0,
                    const unsigned int c1, 
                    const unsigned int size,
                    const unsigned int j0,
                    const unsigned int stride,
                    const int rev,
                    cuda_cpx_t *d_local_data,
                    const cuda_cpx_t *d_scratch)
    {
    unsigned int block_size =512;
    unsigned int n_block = local_size/block_size;
    if (local_size % block_size ) n_block++;

    gpu_c2b_unpack_kernel<<<n_block, block_size>>>(local_size,
                                                   length,
                                                   c0,
                                                   c1,
                                                   size,
                                                   j0,
                                                   stride,
                                                   rev,
                                                   d_local_data,
                                                   d_scratch);
    }

__global__ void gpu_transpose_kernel(const unsigned int size,
                                     const unsigned int length,
                                     const unsigned int stride,
                                     const unsigned int embed,
                                     const cuda_cpx_t *in,
                                     cuda_cpx_t *out)
    {
    int idx = blockIdx.x*blockDim.x+threadIdx.x;
    if (idx >= size) return;

    int i = idx / stride;
    if (i >= length) return;

    int j = idx % stride;

    out[j*embed + i] = in[idx];
    }

#define TILE_DIM 16
#define BLOCK_ROWS 16

__global__ void transpose_sdk(cuda_cpx_t *odata, const cuda_cpx_t *idata, int width, int height, int embed)
{
    __shared__ cuda_cpx_t tile[TILE_DIM][TILE_DIM+1];

    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
    int index_in = xIndex + (yIndex)*width;

    int xIndex_new = blockIdx.y * TILE_DIM + threadIdx.x;
    int yIndex_new = blockIdx.x * TILE_DIM + threadIdx.y;
    int index_out = xIndex_new + (yIndex_new)*embed;

    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
    {
        if ((xIndex < width) && ((i+yIndex) <height))
            tile[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];
    }

    __syncthreads();

    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
    {
        if (xIndex_new< height && ((yIndex_new+i) <width))
            odata[index_out+i*embed] = tile[threadIdx.x][threadIdx.y+i];
    }
}

void gpu_transpose(const unsigned int size,
                   const unsigned int length,
                   const unsigned int stride,
                   const unsigned int embed,
                   const cuda_cpx_t *in,
                   cuda_cpx_t *out)
    {
    unsigned int block_size =512;
    unsigned int n_block = size/block_size;
    if (size % block_size ) n_block++;
    
//    gpu_transpose_kernel<<<n_block, block_size>>>(size, length, stride, embed, in, out);
    int size_x = stride;
    int size_y = length;
    int nblocks_x = size_x/TILE_DIM;
    if (size_x%TILE_DIM) nblocks_x++;
    int nblocks_y = size_y/TILE_DIM;
    if (size_y%TILE_DIM) nblocks_y++;
    dim3 grid(nblocks_x, nblocks_y), threads(TILE_DIM,BLOCK_ROWS);
    if (stride == 1 || length ==1 )
        cudaMemcpy(out,in,sizeof(cuda_cpx_t)*stride*length,cudaMemcpyDefault);
    else
        transpose_sdk<<<grid, threads>>>(out,in, size_x, size_y,embed);
    }
