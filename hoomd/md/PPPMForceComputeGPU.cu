// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "PPPMForceComputeGPU.cuh"
#include "hoomd/TextureTools.h"

// __scalar2int_rd is __float2int_rd in single, __double2int_rd in double
#if HOOMD_LONGREAL_SIZE == 32
#define __scalar2int_rd __float2int_rd
#else
#define __scalar2int_rd __double2int_rd
#endif

#define GPU_PPPM_MAX_ORDER 7

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
// workaround for HIP bug
#ifdef __HIP_PLATFORM_HCC__
inline __device__ float myAtomicAdd(float* address, float val)
    {
    unsigned int* address_as_uint = (unsigned int*)address;
    unsigned int old = *address_as_uint, assumed;

    do
        {
        assumed = old;
        old = atomicCAS(address_as_uint, assumed, __float_as_uint(val + __uint_as_float(assumed)));
        } while (assumed != old);

    return __uint_as_float(old);
    }
#else
inline __device__ float myAtomicAdd(float* address, float val)
    {
    return atomicAdd(address, val);
    }
#endif

//! GPU implementation of sinc(x)==sin(x)/x
__device__ Scalar gpu_sinc(Scalar x)
    {
    Scalar sinc = 0;

    //! Coefficients of a power expansion of sin(x)/x
    const Scalar sinc_coeff[] = {Scalar(1.0),
                                 Scalar(-1.0 / 6.0),
                                 Scalar(1.0 / 120.0),
                                 Scalar(-1.0 / 5040.0),
                                 Scalar(1.0 / 362880.0),
                                 Scalar(-1.0 / 39916800.0)};

    if (x * x <= Scalar(1.0))
        {
        Scalar term = Scalar(1.0);
        for (unsigned int i = 0; i < 6; ++i)
            {
            sinc += sinc_coeff[i] * term;
            term *= x * x;
            }
        }
    else
        {
        sinc = fast::sin(x) / x;
        }

    return sinc;
    }

__device__ int3 find_cell(const Scalar3& pos,
                          const unsigned int& inner_nx,
                          const unsigned int& inner_ny,
                          const unsigned int& inner_nz,
                          const uint3& n_ghost_cells,
                          const BoxDim& box,
                          int order,
                          Scalar3& dr)
    {
    // compute coordinates in units of the mesh size
    Scalar3 f = box.makeFraction(pos);
    uchar3 periodic = box.getPeriodic();

    Scalar3 reduced_pos
        = make_scalar3(f.x * (Scalar)inner_nx, f.y * (Scalar)inner_ny, f.z * (Scalar)inner_nz);

    reduced_pos += make_scalar3(n_ghost_cells.x, n_ghost_cells.y, n_ghost_cells.z);

    Scalar shift, shiftone;
    if (order % 2)
        {
        shift = Scalar(0.5);
        shiftone = Scalar(0.0);
        }
    else
        {
        shift = Scalar(0.0);
        shiftone = Scalar(0.5);
        }

    int ix = __scalar2int_rd(reduced_pos.x + shift);
    int iy = __scalar2int_rd(reduced_pos.y + shift);
    int iz = __scalar2int_rd(reduced_pos.z + shift);

    // set distance to cell center
    dr.x = shiftone + (Scalar)ix - reduced_pos.x;
    dr.y = shiftone + (Scalar)iy - reduced_pos.y;
    dr.z = shiftone + (Scalar)iz - reduced_pos.z;

    // handle particles on the boundary
    if (periodic.x && ix == (int)inner_nx)
        ix = 0;
    if (periodic.y && iy == (int)inner_ny)
        iy = 0;
    if (periodic.z && iz == (int)inner_nz)
        iz = 0;

    return make_int3(ix, iy, iz);
    }

__global__ void gpu_assign_particles_kernel(const uint3 mesh_dim,
                                            const uint3 n_ghost_bins,
                                            unsigned int work_size,
                                            const unsigned int* d_index_array,
                                            const Scalar4* d_postype,
                                            const Scalar* d_charge,
                                            hipfftComplex* d_mesh,
                                            Scalar V_cell,
                                            int order,
                                            unsigned int offset,
                                            BoxDim box,
                                            const Scalar* d_rho_coeff)
    {
    extern __shared__ Scalar s_coeff[];

    // load in interpolation coefficients
    unsigned int ncoeffs = order * (2 * order + 1);
    for (unsigned int cur_offset = 0; cur_offset < ncoeffs; cur_offset += blockDim.x)
        {
        if (cur_offset + threadIdx.x < ncoeffs)
            {
            s_coeff[cur_offset + threadIdx.x] = d_rho_coeff[cur_offset + threadIdx.x];
            }
        }
    __syncthreads();

    unsigned int work_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (work_idx >= work_size)
        return;

    unsigned int group_idx = work_idx + offset;

    int3 bin_dim = make_int3(mesh_dim.x + 2 * n_ghost_bins.x,
                             mesh_dim.y + 2 * n_ghost_bins.y,
                             mesh_dim.z + 2 * n_ghost_bins.z);

    // grid coordinates of bin (column-major)
    unsigned int idx = d_index_array[group_idx];

    Scalar4 postype = d_postype[idx];

    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
    Scalar qi = d_charge[idx];

    // compute coordinates in units of the cell size
    Scalar3 dr = make_scalar3(0, 0, 0);
    int3 bin_coord
        = find_cell(pos, mesh_dim.x, mesh_dim.y, mesh_dim.z, n_ghost_bins, box, order, dr);

    // ignore particles that are not within our domain (the error should be caught by HOOMD's cell
    // list)
    if (bin_coord.x < 0 || bin_coord.x >= bin_dim.x || bin_coord.y < 0 || bin_coord.y >= bin_dim.y
        || bin_coord.z < 0 || bin_coord.z >= bin_dim.z)
        {
        return;
        }

    int i = bin_coord.x;
    int j = bin_coord.y;
    int k = bin_coord.z;

    int nlower = -(order - 1) / 2;
    int nupper = order / 2;

    Scalar result;

    int mult_fact = 2 * order + 1;

    Scalar x0 = qi;

    bool ignore_x = false;
    bool ignore_y = false;
    bool ignore_z = false;

    // loop over neighboring bins
    for (int l = nlower; l <= nupper; ++l)
        {
        // precalculate assignment factor
        result = Scalar(0.0);
        for (int iorder = order - 1; iorder >= 0; iorder--)
            {
            result = s_coeff[l - nlower + iorder * mult_fact] + result * dr.x;
            }
        Scalar y0 = x0 * result;

        int neighi = i + l;
        if (neighi >= (int)bin_dim.x)
            {
            if (!n_ghost_bins.x)
                neighi -= (int)bin_dim.x;
            else
                ignore_x = true;
            }
        else if (neighi < 0)
            {
            if (!n_ghost_bins.x)
                neighi += (int)bin_dim.x;
            else
                ignore_x = true;
            }

        for (int m = nlower; m <= nupper; ++m)
            {
            result = Scalar(0.0);
            for (int iorder = order - 1; iorder >= 0; iorder--)
                {
                result = s_coeff[m - nlower + iorder * mult_fact] + result * dr.y;
                }
            Scalar z0 = y0 * result;

            int neighj = j + m;
            if (neighj >= (int)bin_dim.y)
                {
                if (!n_ghost_bins.y)
                    neighj -= (int)bin_dim.y;
                else
                    ignore_y = true;
                }
            else if (neighj < 0)
                {
                if (!n_ghost_bins.y)
                    neighj += (int)bin_dim.y;
                else
                    ignore_y = true;
                }

            for (int n = nlower; n <= nupper; ++n)
                {
                result = Scalar(0.0);
                for (int iorder = order - 1; iorder >= 0; iorder--)
                    {
                    result = s_coeff[n - nlower + iorder * mult_fact] + result * dr.z;
                    }

                int neighk = k + n;

                if (neighk >= (int)bin_dim.z)
                    {
                    if (!n_ghost_bins.z)
                        neighk -= (int)bin_dim.z;
                    else
                        ignore_z = true;
                    }
                else if (neighk < 0)
                    {
                    if (!n_ghost_bins.z)
                        neighk += (int)bin_dim.z;
                    else
                        ignore_z = true;
                    }

                if (!ignore_x && !ignore_y && !ignore_z)
                    {
                    // write out to global memory using row-major
                    unsigned int cell_idx = neighi + bin_dim.x * (neighj + bin_dim.y * neighk);

                    // compute fraction of particle density assigned to cell
                    // from particles in this bin
                    myAtomicAdd(&d_mesh[cell_idx].x, z0 * result / V_cell);
                    }

                ignore_z = false;
                }
            ignore_y = false;
            }
        ignore_x = false;
        } // end of loop over neighboring bins
    }

__global__ void gpu_reduce_meshes(const unsigned int mesh_elements,
                                  const hipfftComplex* d_mesh_scratch,
                                  hipfftComplex* d_mesh,
                                  unsigned int ngpu)
    {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= mesh_elements)
        return;

    hipfftComplex res;
    res.x = 0;
    res.y = 0;

    // reduce over all temporary meshes
    for (unsigned int igpu = 0; igpu < ngpu; ++igpu)
        {
        hipfftComplex m = d_mesh_scratch[idx + igpu * mesh_elements];
        res.x += m.x;
        res.y += m.y;
        }
    d_mesh[idx] = res;
    }

void gpu_assign_particles(const uint3 mesh_dim,
                          const uint3 n_ghost_bins,
                          const uint3 grid_dim,
                          unsigned int group_size,
                          const unsigned int* d_index_array,
                          const Scalar4* d_postype,
                          const Scalar* d_charge,
                          hipfftComplex* d_mesh,
                          hipfftComplex* d_mesh_scratch,
                          const unsigned int mesh_elements,
                          int order,
                          const BoxDim& box,
                          unsigned int block_size,
                          const Scalar* d_rho_coeff,
                          const hipDeviceProp_t& dev_prop,
                          const GPUPartition& gpu_partition)
    {
    hipMemsetAsync(d_mesh, 0, sizeof(hipfftComplex) * grid_dim.x * grid_dim.y * grid_dim.z);
    Scalar V_cell = box.getVolume() / (Scalar)(mesh_dim.x * mesh_dim.y * mesh_dim.z);

    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)gpu_assign_particles_kernel);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(max_block_size, block_size);

    while (attr.sharedSizeBytes >= dev_prop.sharedMemPerBlock)
        {
        run_block_size -= dev_prop.warpSize;
        }

    // iterate over active GPUs in reverse, to end up on first GPU when returning from this function
    unsigned int ngpu = gpu_partition.getNumActiveGPUs();
    for (int idev = ngpu - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);

        if (ngpu > 1)
            {
            // zero the temporary mesh array
            hipMemsetAsync(d_mesh_scratch + idev * mesh_elements,
                           0,
                           sizeof(hipfftComplex) * mesh_elements);
            }

        unsigned int nwork = range.second - range.first;
        unsigned int n_blocks = nwork / run_block_size + 1;
        const size_t shared_bytes = order * (2 * order + 1) * sizeof(Scalar);

        hipLaunchKernelGGL((gpu_assign_particles_kernel),
                           dim3(n_blocks),
                           dim3(run_block_size),
                           shared_bytes,
                           0,
                           mesh_dim,
                           n_ghost_bins,
                           nwork,
                           d_index_array,
                           d_postype,
                           d_charge,
                           ngpu > 1 ? d_mesh_scratch + idev * mesh_elements : d_mesh,
                           V_cell,
                           order,
                           range.first,
                           box,
                           d_rho_coeff);
        }
    }

//! Reduce temporary arrays for every GPU
void gpu_reduce_meshes(const unsigned int mesh_elements,
                       const hipfftComplex* d_mesh_scratch,
                       hipfftComplex* d_mesh,
                       const unsigned int ngpu,
                       const unsigned int block_size)
    {
    // reduce meshes on GPU 0
    hipLaunchKernelGGL((gpu_reduce_meshes),
                       dim3(mesh_elements / block_size + 1),
                       dim3(block_size),
                       0,
                       0,
                       mesh_elements,
                       d_mesh_scratch,
                       d_mesh,
                       ngpu);
    }

__global__ void gpu_compute_mesh_virial_kernel(const unsigned int n_wave_vectors,
                                               hipfftComplex* d_fourier_mesh,
                                               Scalar* d_inf_f,
                                               Scalar* d_virial_mesh,
                                               const Scalar3* d_k,
                                               const bool exclude_dc,
                                               Scalar kappa)
    {
    unsigned int idx;

    idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= n_wave_vectors)
        return;

    if (!exclude_dc || idx != 0)
        {
        // non-zero wave vector
        hipfftComplex fourier = d_fourier_mesh[idx];

        Scalar3 k = d_k[idx];

        Scalar rhog = (fourier.x * fourier.x + fourier.y * fourier.y) * d_inf_f[idx];
        Scalar vterm = -Scalar(2.0) * (Scalar(1.0) / dot(k, k) + Scalar(0.25) / (kappa * kappa));

        d_virial_mesh[0 * n_wave_vectors + idx] = rhog * (Scalar(1.0) + vterm * k.x * k.x); // xx
        d_virial_mesh[1 * n_wave_vectors + idx] = rhog * (vterm * k.x * k.y);               // xy
        d_virial_mesh[2 * n_wave_vectors + idx] = rhog * (vterm * k.x * k.z);               // xz
        d_virial_mesh[3 * n_wave_vectors + idx] = rhog * (Scalar(1.0) + vterm * k.y * k.y); // yy
        d_virial_mesh[4 * n_wave_vectors + idx] = rhog * (vterm * k.y * k.z);               // yz
        d_virial_mesh[5 * n_wave_vectors + idx] = rhog * (Scalar(1.0) + vterm * k.z * k.z); // zz
        }
    else
        {
        d_virial_mesh[0 * n_wave_vectors + idx] = Scalar(0.0);
        d_virial_mesh[1 * n_wave_vectors + idx] = Scalar(0.0);
        d_virial_mesh[2 * n_wave_vectors + idx] = Scalar(0.0);
        d_virial_mesh[3 * n_wave_vectors + idx] = Scalar(0.0);
        d_virial_mesh[4 * n_wave_vectors + idx] = Scalar(0.0);
        d_virial_mesh[5 * n_wave_vectors + idx] = Scalar(0.0);
        }
    }

void gpu_compute_mesh_virial(const unsigned int n_wave_vectors,
                             hipfftComplex* d_fourier_mesh,
                             Scalar* d_inf_f,
                             Scalar* d_virial_mesh,
                             const Scalar3* d_k,
                             const bool exclude_dc,
                             Scalar kappa)

    {
    const unsigned int block_size = 256;

    dim3 grid(n_wave_vectors / block_size + 1, 1, 1);

    hipLaunchKernelGGL((gpu_compute_mesh_virial_kernel),
                       dim3(grid),
                       dim3(block_size),
                       0,
                       0,
                       n_wave_vectors,
                       d_fourier_mesh,
                       d_inf_f,
                       d_virial_mesh,
                       d_k,
                       exclude_dc,
                       kappa);
    }

__global__ void gpu_update_meshes_kernel(const unsigned int n_wave_vectors,
                                         hipfftComplex* d_fourier_mesh,
                                         hipfftComplex* d_fourier_mesh_G_x,
                                         hipfftComplex* d_fourier_mesh_G_y,
                                         hipfftComplex* d_fourier_mesh_G_z,
                                         const Scalar* d_inf_f,
                                         const Scalar3* d_k,
                                         unsigned int NNN)
    {
    unsigned int k;

    k = blockDim.x * blockIdx.x + threadIdx.x;

    if (k >= n_wave_vectors)
        return;

    hipfftComplex f = d_fourier_mesh[k];

    Scalar scaled_inf_f = d_inf_f[k] / ((Scalar)NNN);

    Scalar3 kvec = d_k[k];

    // Normalization
    hipfftComplex fourier_G_x;
    fourier_G_x.x = f.y * kvec.x * scaled_inf_f;
    fourier_G_x.y = -f.x * kvec.x * scaled_inf_f;

    hipfftComplex fourier_G_y;
    fourier_G_y.x = f.y * kvec.y * scaled_inf_f;
    fourier_G_y.y = -f.x * kvec.y * scaled_inf_f;

    hipfftComplex fourier_G_z;
    fourier_G_z.x = f.y * kvec.z * scaled_inf_f;
    fourier_G_z.y = -f.x * kvec.z * scaled_inf_f;

    // store in global memory
    d_fourier_mesh_G_x[k] = fourier_G_x;
    d_fourier_mesh_G_y[k] = fourier_G_y;
    d_fourier_mesh_G_z[k] = fourier_G_z;
    }

void gpu_update_meshes(const unsigned int n_wave_vectors,
                       hipfftComplex* d_fourier_mesh,
                       hipfftComplex* d_fourier_mesh_G_x,
                       hipfftComplex* d_fourier_mesh_G_y,
                       hipfftComplex* d_fourier_mesh_G_z,
                       const Scalar* d_inf_f,
                       const Scalar3* d_k,
                       unsigned int NNN,
                       unsigned int block_size)

    {
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)gpu_update_meshes_kernel);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(max_block_size, block_size);
    dim3 grid(n_wave_vectors / run_block_size + 1, 1, 1);

    hipLaunchKernelGGL((gpu_update_meshes_kernel),
                       dim3(grid),
                       dim3(run_block_size),
                       0,
                       0,
                       n_wave_vectors,
                       d_fourier_mesh,
                       d_fourier_mesh_G_x,
                       d_fourier_mesh_G_y,
                       d_fourier_mesh_G_z,
                       d_inf_f,
                       d_k,
                       NNN);
    }

__global__ void gpu_compute_forces_kernel(const unsigned int work_size,
                                          const Scalar4* d_postype,
                                          Scalar4* d_force,
                                          const uint3 grid_dim,
                                          const uint3 n_ghost_cells,
                                          const Scalar* d_charge,
                                          const BoxDim box,
                                          int order,
                                          const unsigned int* d_index_array,
                                          const hipfftComplex* inv_fourier_mesh_x,
                                          const hipfftComplex* inv_fourier_mesh_y,
                                          const hipfftComplex* inv_fourier_mesh_z,
                                          const Scalar* d_rho_coeff,
                                          const unsigned int offset)
    {
    extern __shared__ Scalar s_coeff[];

    // load in interpolation coefficients
    unsigned int ncoeffs = order * (2 * order + 1);
    for (unsigned int cur_offset = 0; cur_offset < ncoeffs; cur_offset += blockDim.x)
        {
        if (cur_offset + threadIdx.x < ncoeffs)
            {
            s_coeff[cur_offset + threadIdx.x] = d_rho_coeff[cur_offset + threadIdx.x];
            }
        }
    __syncthreads();

    unsigned int work_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (work_idx >= work_size)
        return;

    unsigned int group_idx = work_idx + offset;

    unsigned int idx = d_index_array[group_idx];

    int3 inner_dim = make_int3(grid_dim.x - 2 * n_ghost_cells.x,
                               grid_dim.y - 2 * n_ghost_cells.y,
                               grid_dim.z - 2 * n_ghost_cells.z);

    Scalar4 postype = d_postype[idx];

    Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
    unsigned int type = __scalar_as_int(postype.w);
    Scalar qi = d_charge[idx];

    Scalar3 dr = make_scalar3(0, 0, 0);

    // find cell the particle is in
    int3 cell_coord
        = find_cell(pos, inner_dim.x, inner_dim.y, inner_dim.z, n_ghost_cells, box, order, dr);

    // ignore particles that are not within our domain (the error should be caught by HOOMD's cell
    // list)
    if (cell_coord.x < 0 || cell_coord.x >= (int)grid_dim.x || cell_coord.y < 0
        || cell_coord.y >= (int)grid_dim.y || cell_coord.z < 0 || cell_coord.z >= (int)grid_dim.z)
        {
        return;
        }

    Scalar3 force = make_scalar3(0.0, 0.0, 0.0);

    int nlower = -(order - 1) / 2;
    int nupper = order / 2;

    Scalar result;
    int mult_fact = 2 * order + 1;

    // back-interpolate forces from neighboring mesh points
    for (int l = nlower; l <= nupper; ++l)
        {
        result = Scalar(0.0);
        for (int k = order - 1; k >= 0; k--)
            {
            result = s_coeff[l - nlower + k * mult_fact] + result * dr.x;
            }
        Scalar x0 = result;

        for (int m = nlower; m <= nupper; ++m)
            {
            result = Scalar(0.0);
            for (int k = order - 1; k >= 0; k--)
                {
                result = s_coeff[m - nlower + k * mult_fact] + result * dr.y;
                }
            Scalar y0 = x0 * result;

            for (int n = nlower; n <= nupper; ++n)
                {
                result = Scalar(0.0);
                for (int k = order - 1; k >= 0; k--)
                    {
                    result = s_coeff[n - nlower + k * mult_fact] + result * dr.z;
                    }
                Scalar z0 = y0 * result;

                int neighl = (int)cell_coord.x + l;
                int neighm = (int)cell_coord.y + m;
                int neighn = (int)cell_coord.z + n;

                if (!n_ghost_cells.x)
                    {
                    if (neighl >= (int)grid_dim.x)
                        neighl -= grid_dim.x;
                    else if (neighl < 0)
                        neighl += grid_dim.x;
                    }

                if (!n_ghost_cells.y)
                    {
                    if (neighm >= (int)grid_dim.y)
                        neighm -= grid_dim.y;
                    else if (neighm < 0)
                        neighm += grid_dim.y;
                    }

                if (!n_ghost_cells.z)
                    {
                    if (neighn >= (int)grid_dim.z)
                        neighn -= grid_dim.z;
                    else if (neighn < 0)
                        neighn += grid_dim.z;
                    }

                // use column-major layout
                unsigned int cell_idx = neighl + grid_dim.x * (neighm + grid_dim.y * neighn);

                hipfftComplex inv_mesh_x = inv_fourier_mesh_x[cell_idx];
                hipfftComplex inv_mesh_y = inv_fourier_mesh_y[cell_idx];
                hipfftComplex inv_mesh_z = inv_fourier_mesh_z[cell_idx];

                force.x += qi * z0 * inv_mesh_x.x;
                force.y += qi * z0 * inv_mesh_y.x;
                force.z += qi * z0 * inv_mesh_z.x;
                }
            }
        } // end neighbor cells loop

    d_force[idx] = make_scalar4(force.x, force.y, force.z, 0.0);
    }

void gpu_compute_forces(const unsigned int N,
                        const Scalar4* d_postype,
                        Scalar4* d_force,
                        const hipfftComplex* d_inv_fourier_mesh_x,
                        const hipfftComplex* d_inv_fourier_mesh_y,
                        const hipfftComplex* d_inv_fourier_mesh_z,
                        const uint3 grid_dim,
                        const uint3 n_ghost_cells,
                        const Scalar* d_charge,
                        const BoxDim& box,
                        int order,
                        const unsigned int* d_index_array,
                        const GPUPartition& gpu_partition,
                        const GPUPartition& all_gpu_partition,
                        const Scalar* d_rho_coeff,
                        unsigned int block_size,
                        bool local_fft,
                        unsigned int inv_mesh_elements)
    {
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)gpu_compute_forces_kernel);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(max_block_size, block_size);

    // iterate over active GPUs in reverse, to end up on first GPU when returning from this function
    for (int idev = all_gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = all_gpu_partition.getRangeAndSetGPU(idev);

        // reset force array for ALL particles
        hipMemsetAsync(d_force + range.first, 0, sizeof(Scalar4) * (range.second - range.first));
        }

    // iterate over active GPUs in reverse, to end up on first GPU when returning from this function
    for (int idev = gpu_partition.getNumActiveGPUs() - 1; idev >= 0; --idev)
        {
        auto range = gpu_partition.getRangeAndSetGPU(idev);

        unsigned int nwork = range.second - range.first;
        unsigned int n_blocks = nwork / run_block_size + 1;
        const size_t shared_bytes = order * (2 * order + 1) * sizeof(Scalar);

        hipLaunchKernelGGL(
            (gpu_compute_forces_kernel),
            dim3(n_blocks),
            dim3(run_block_size),
            shared_bytes,
            0,
            nwork,
            d_postype,
            d_force,
            grid_dim,
            n_ghost_cells,
            d_charge,
            box,
            order,
            d_index_array,
            local_fft ? d_inv_fourier_mesh_x + idev * inv_mesh_elements : d_inv_fourier_mesh_x,
            local_fft ? d_inv_fourier_mesh_y + idev * inv_mesh_elements : d_inv_fourier_mesh_y,
            local_fft ? d_inv_fourier_mesh_z + idev * inv_mesh_elements : d_inv_fourier_mesh_z,
            d_rho_coeff,
            range.first);
        }
    }

__global__ void kernel_calculate_pe_partial(int n_wave_vectors,
                                            Scalar* sum_partial,
                                            const hipfftComplex* d_fourier_mesh,
                                            const Scalar* d_inf_f,
                                            const bool exclude_dc)
    {
    HIP_DYNAMIC_SHARED(Scalar, sdata)

    unsigned int tidx = threadIdx.x;

    unsigned int j;

    j = blockDim.x * blockIdx.x + threadIdx.x;

    Scalar mySum = Scalar(0.0);

    if (j < n_wave_vectors)
        {
        if (!exclude_dc || j != 0)
            {
            mySum = d_fourier_mesh[j].x * d_fourier_mesh[j].x
                    + d_fourier_mesh[j].y * d_fourier_mesh[j].y;
            mySum *= d_inf_f[j];
            }
        }

    sdata[tidx] = mySum;

    __syncthreads();

    // reduce the sum
    int offs = blockDim.x >> 1;
    while (offs > 0)
        {
        if (tidx < offs)
            {
            sdata[tidx] += sdata[tidx + offs];
            }
        offs >>= 1;
        __syncthreads();
        }

    // write result to global memory
    if (tidx == 0)
        sum_partial[blockIdx.x] = sdata[0];
    }

__global__ void kernel_final_reduce_pe(Scalar* sum_partial, unsigned int nblocks, Scalar* sum)
    {
    HIP_DYNAMIC_SHARED(Scalar, smem)

    if (threadIdx.x == 0)
        *sum = Scalar(0.0);

    for (int start = 0; start < nblocks; start += blockDim.x)
        {
        __syncthreads();
        if (start + threadIdx.x < nblocks)
            smem[threadIdx.x] = sum_partial[start + threadIdx.x];
        else
            smem[threadIdx.x] = Scalar(0.0);

        __syncthreads();

        // reduce the sum
        int offs = blockDim.x >> 1;
        while (offs > 0)
            {
            if (threadIdx.x < offs)
                smem[threadIdx.x] += smem[threadIdx.x + offs];
            offs >>= 1;
            __syncthreads();
            }

        if (threadIdx.x == 0)
            {
            *sum += smem[0];
            }
        }
    }

void gpu_compute_pe(unsigned int n_wave_vectors,
                    Scalar* d_sum_partial,
                    Scalar* d_sum,
                    const hipfftComplex* d_fourier_mesh,
                    const Scalar* d_inf_f,
                    const unsigned int block_size,
                    const uint3 mesh_dim,
                    const bool exclude_dc)
    {
    unsigned int n_blocks = n_wave_vectors / block_size + 1;

    unsigned int shared_size = (unsigned int)(block_size * sizeof(Scalar));

    dim3 grid(n_blocks, 1, 1);

    hipLaunchKernelGGL((kernel_calculate_pe_partial),
                       dim3(grid),
                       dim3(block_size),
                       shared_size,
                       0,
                       n_wave_vectors,
                       d_sum_partial,
                       d_fourier_mesh,
                       d_inf_f,
                       exclude_dc);

    // calculate final sum of mesh values
    const unsigned int final_block_size = 256;
    shared_size = final_block_size * sizeof(Scalar);
    hipLaunchKernelGGL((kernel_final_reduce_pe),
                       dim3(1),
                       dim3(final_block_size),
                       shared_size,
                       0,
                       d_sum_partial,
                       n_blocks,
                       d_sum);
    }

__global__ void kernel_calculate_virial_partial(int n_wave_vectors,
                                                Scalar* sum_virial_partial,
                                                const Scalar* d_mesh_virial)
    {
    HIP_DYNAMIC_SHARED(Scalar, sdata)

    unsigned int j;

    j = blockDim.x * blockIdx.x + threadIdx.x;

    unsigned int tidx = threadIdx.x;

    Scalar mySum_xx = Scalar(0.0);
    Scalar mySum_xy = Scalar(0.0);
    Scalar mySum_xz = Scalar(0.0);
    Scalar mySum_yy = Scalar(0.0);
    Scalar mySum_yz = Scalar(0.0);
    Scalar mySum_zz = Scalar(0.0);

    if (j < n_wave_vectors)
        {
        mySum_xx = d_mesh_virial[0 * n_wave_vectors + j];
        mySum_xy = d_mesh_virial[1 * n_wave_vectors + j];
        mySum_xz = d_mesh_virial[2 * n_wave_vectors + j];
        mySum_yy = d_mesh_virial[3 * n_wave_vectors + j];
        mySum_yz = d_mesh_virial[4 * n_wave_vectors + j];
        mySum_zz = d_mesh_virial[5 * n_wave_vectors + j];
        }

    sdata[0 * blockDim.x + tidx] = mySum_xx;
    sdata[1 * blockDim.x + tidx] = mySum_xy;
    sdata[2 * blockDim.x + tidx] = mySum_xz;
    sdata[3 * blockDim.x + tidx] = mySum_yy;
    sdata[4 * blockDim.x + tidx] = mySum_yz;
    sdata[5 * blockDim.x + tidx] = mySum_zz;

    __syncthreads();

    // reduce the sum
    int offs = blockDim.x >> 1;
    while (offs > 0)
        {
        if (tidx < offs)
            {
            sdata[0 * blockDim.x + tidx] += sdata[0 * blockDim.x + tidx + offs];
            sdata[1 * blockDim.x + tidx] += sdata[1 * blockDim.x + tidx + offs];
            sdata[2 * blockDim.x + tidx] += sdata[2 * blockDim.x + tidx + offs];
            sdata[3 * blockDim.x + tidx] += sdata[3 * blockDim.x + tidx + offs];
            sdata[4 * blockDim.x + tidx] += sdata[4 * blockDim.x + tidx + offs];
            sdata[5 * blockDim.x + tidx] += sdata[5 * blockDim.x + tidx + offs];
            }
        offs >>= 1;
        __syncthreads();
        }

    // write result to global memory
    if (tidx == 0)
        {
        sum_virial_partial[0 * gridDim.x + blockIdx.x] = sdata[0 * blockDim.x];
        sum_virial_partial[1 * gridDim.x + blockIdx.x] = sdata[1 * blockDim.x];
        sum_virial_partial[2 * gridDim.x + blockIdx.x] = sdata[2 * blockDim.x];
        sum_virial_partial[3 * gridDim.x + blockIdx.x] = sdata[3 * blockDim.x];
        sum_virial_partial[4 * gridDim.x + blockIdx.x] = sdata[4 * blockDim.x];
        sum_virial_partial[5 * gridDim.x + blockIdx.x] = sdata[5 * blockDim.x];
        }
    }

__global__ void
kernel_final_reduce_virial(Scalar* sum_virial_partial, unsigned int nblocks, Scalar* sum_virial)
    {
    HIP_DYNAMIC_SHARED(Scalar, smem)

    if (threadIdx.x == 0)
        {
        sum_virial[0] = Scalar(0.0);
        sum_virial[1] = Scalar(0.0);
        sum_virial[2] = Scalar(0.0);
        sum_virial[3] = Scalar(0.0);
        sum_virial[4] = Scalar(0.0);
        sum_virial[5] = Scalar(0.0);
        }

    for (int start = 0; start < nblocks; start += blockDim.x)
        {
        __syncthreads();
        if (start + threadIdx.x < nblocks)
            {
            smem[0 * blockDim.x + threadIdx.x]
                = sum_virial_partial[0 * nblocks + start + threadIdx.x];
            smem[1 * blockDim.x + threadIdx.x]
                = sum_virial_partial[1 * nblocks + start + threadIdx.x];
            smem[2 * blockDim.x + threadIdx.x]
                = sum_virial_partial[2 * nblocks + start + threadIdx.x];
            smem[3 * blockDim.x + threadIdx.x]
                = sum_virial_partial[3 * nblocks + start + threadIdx.x];
            smem[4 * blockDim.x + threadIdx.x]
                = sum_virial_partial[4 * nblocks + start + threadIdx.x];
            smem[5 * blockDim.x + threadIdx.x]
                = sum_virial_partial[5 * nblocks + start + threadIdx.x];
            }
        else
            {
            smem[0 * blockDim.x + threadIdx.x] = Scalar(0.0);
            smem[1 * blockDim.x + threadIdx.x] = Scalar(0.0);
            smem[2 * blockDim.x + threadIdx.x] = Scalar(0.0);
            smem[3 * blockDim.x + threadIdx.x] = Scalar(0.0);
            smem[4 * blockDim.x + threadIdx.x] = Scalar(0.0);
            smem[5 * blockDim.x + threadIdx.x] = Scalar(0.0);
            }

        __syncthreads();

        // reduce the sum
        int offs = blockDim.x >> 1;
        while (offs > 0)
            {
            if (threadIdx.x < offs)
                {
                smem[0 * blockDim.x + threadIdx.x] += smem[0 * blockDim.x + threadIdx.x + offs];
                smem[1 * blockDim.x + threadIdx.x] += smem[1 * blockDim.x + threadIdx.x + offs];
                smem[2 * blockDim.x + threadIdx.x] += smem[2 * blockDim.x + threadIdx.x + offs];
                smem[3 * blockDim.x + threadIdx.x] += smem[3 * blockDim.x + threadIdx.x + offs];
                smem[4 * blockDim.x + threadIdx.x] += smem[4 * blockDim.x + threadIdx.x + offs];
                smem[5 * blockDim.x + threadIdx.x] += smem[5 * blockDim.x + threadIdx.x + offs];
                }
            offs >>= 1;
            __syncthreads();
            }

        if (threadIdx.x == 0)
            {
            sum_virial[0] += smem[0 * blockDim.x];
            sum_virial[1] += smem[1 * blockDim.x];
            sum_virial[2] += smem[2 * blockDim.x];
            sum_virial[3] += smem[3 * blockDim.x];
            sum_virial[4] += smem[4 * blockDim.x];
            sum_virial[5] += smem[5 * blockDim.x];
            }
        }
    }

void gpu_compute_virial(unsigned int n_wave_vectors,
                        Scalar* d_sum_virial_partial,
                        Scalar* d_sum_virial,
                        const Scalar* d_mesh_virial,
                        const unsigned int block_size)
    {
    unsigned int n_blocks = n_wave_vectors / block_size + 1;

    unsigned int shared_size = (unsigned int)(6 * block_size * sizeof(Scalar));

    dim3 grid(n_blocks, 1, 1);

    hipLaunchKernelGGL((kernel_calculate_virial_partial),
                       dim3(grid),
                       dim3(block_size),
                       shared_size,
                       0,
                       n_wave_vectors,
                       d_sum_virial_partial,
                       d_mesh_virial);

    // calculate final virial values
    const unsigned int final_block_size = 256;
    shared_size = 6 * final_block_size * sizeof(Scalar);
    hipLaunchKernelGGL((kernel_final_reduce_virial),
                       dim3(1),
                       dim3(final_block_size),
                       shared_size,
                       0,
                       d_sum_virial_partial,
                       n_blocks,
                       d_sum_virial);
    }

template<bool local_fft>
__global__ void gpu_compute_influence_function_kernel(const uint3 mesh_dim,
                                                      const unsigned int n_wave_vectors,
                                                      const uint3 global_dim,
                                                      Scalar* d_inf_f,
                                                      Scalar3* d_k,
                                                      const Scalar3 b1,
                                                      const Scalar3 b2,
                                                      const Scalar3 b3,
                                                      const uint3 pidx,
                                                      const uint3 pdim,
                                                      int nbx,
                                                      int nby,
                                                      int nbz,
                                                      const Scalar* gf_b,
                                                      int order,
                                                      Scalar kappa,
                                                      Scalar alpha)
    {
    unsigned int kidx;

    kidx = blockDim.x * blockIdx.x + threadIdx.x;

    if (kidx >= n_wave_vectors)
        return;

    int l, m, n;
    if (local_fft)
        {
        // use row-major layout
        int ny = mesh_dim.y;
        int nx = mesh_dim.x;
        n = kidx / ny / nx;
        m = (kidx - n * ny * nx) / nx;
        l = kidx % nx;
        }
#ifdef ENABLE_MPI
    else
        {
        // local layout: row-major
        int ny = mesh_dim.y;
        int nx = mesh_dim.x;
        int n_local = kidx / ny / nx;
        int m_local = (kidx - n_local * ny * nx) / nx;
        int l_local = kidx % nx;

        // cyclic distribution
        l = l_local * pdim.x + pidx.x;
        m = m_local * pdim.y + pidx.y;
        n = n_local * pdim.z + pidx.z;
        }
#endif

    // compute Miller indices
    if (l >= (int)(global_dim.x / 2 + global_dim.x % 2))
        l -= (int)global_dim.x;
    if (m >= (int)(global_dim.y / 2 + global_dim.y % 2))
        m -= (int)global_dim.y;
    if (n >= (int)(global_dim.z / 2 + global_dim.z % 2))
        n -= (int)global_dim.z;

    Scalar val;
    Scalar3 kval = (Scalar)l * b1 + (Scalar)m * b2 + (Scalar)n * b3;

    Scalar3 kH = Scalar(2.0 * M_PI)
                 * make_scalar3(Scalar(1.0) / (Scalar)global_dim.x,
                                Scalar(1.0) / (Scalar)global_dim.y,
                                Scalar(1.0) / (Scalar)global_dim.z);

    Scalar snx = fast::sin(Scalar(0.5) * l * kH.x);
    Scalar snx2 = snx * snx;

    Scalar sny = fast::sin(Scalar(0.5) * m * kH.y);
    Scalar sny2 = sny * sny;

    Scalar snz = fast::sin(Scalar(0.5) * n * kH.z);
    Scalar snz2 = snz * snz;

    Scalar sx(0.0), sy(0.0), sz(0.0);
    for (int iorder = order - 1; iorder >= 0; iorder--)
        {
        sx = gf_b[iorder] + sx * snx2;
        sy = gf_b[iorder] + sy * sny2;
        sz = gf_b[iorder] + sz * snz2;
        }
    Scalar denominator = sx * sy * sz;
    denominator *= denominator;

    if (l != 0 || m != 0 || n != 0)
        {
        Scalar sum1(0.0);
        Scalar numerator = Scalar(4.0 * M_PI) / dot(kval, kval);

        for (int ix = -nbx; ix <= nbx; ix++)
            {
            Scalar qx = ((Scalar)l + (Scalar)ix * global_dim.x);
            Scalar3 knx = qx * b1;

            Scalar argx = Scalar(0.5) * qx * kH.x;
            Scalar wxs = gpu_sinc(argx);
            Scalar wx(1.0);
            for (int iorder = 0; iorder < order; ++iorder)
                {
                wx *= wxs;
                }

            for (int iy = -nby; iy <= nby; iy++)
                {
                Scalar qy = ((Scalar)m + (Scalar)iy * global_dim.y);
                Scalar3 kny = qy * b2;

                Scalar argy = Scalar(0.5) * qy * kH.y;
                Scalar wys = gpu_sinc(argy);
                Scalar wy(1.0);
                for (int iorder = 0; iorder < order; ++iorder)
                    {
                    wy *= wys;
                    }

                for (int iz = -nbz; iz <= nbz; iz++)
                    {
                    Scalar qz = ((Scalar)n + (Scalar)iz * global_dim.z);
                    Scalar3 knz = qz * b3;

                    Scalar argz = Scalar(0.5) * qz * kH.z;
                    Scalar wzs = gpu_sinc(argz);
                    Scalar wz(1.0);
                    for (int iorder = 0; iorder < order; ++iorder)
                        {
                        wz *= wzs;
                        }

                    Scalar3 kn = knx + kny + knz;
                    Scalar dot1 = dot(kn, kval);
                    Scalar dot2 = dot(kn, kn) + alpha * alpha;

                    Scalar arg_gauss = Scalar(0.25) * dot2 / kappa / kappa;
                    Scalar gauss = exp(-arg_gauss);

                    sum1 += (dot1 / dot2) * gauss * wx * wx * wy * wy * wz * wz;
                    }
                }
            }
        val = numerator * sum1 / denominator;
        }
    else
        {
        val = Scalar(0.0);
        }

    // write out result
    d_inf_f[kidx] = val;
    d_k[kidx] = kval;
    }

void gpu_compute_influence_function(const uint3 mesh_dim,
                                    const uint3 global_dim,
                                    Scalar* d_inf_f,
                                    Scalar3* d_k,
                                    const BoxDim& global_box,
                                    const bool local_fft,
                                    const uint3 pidx,
                                    const uint3 pdim,
                                    const Scalar EPS_HOC,
                                    Scalar kappa,
                                    Scalar alpha,
                                    const Scalar* d_gf_b,
                                    int order,
                                    unsigned int block_size)
    {
    // compute reciprocal lattice vectors
    Scalar3 a1 = global_box.getLatticeVector(0);
    Scalar3 a2 = global_box.getLatticeVector(1);
    Scalar3 a3 = global_box.getLatticeVector(2);

    Scalar V_box = global_box.getVolume();
    Scalar3 b1 = Scalar(2.0 * M_PI)
                 * make_scalar3(a2.y * a3.z - a2.z * a3.y,
                                a2.z * a3.x - a2.x * a3.z,
                                a2.x * a3.y - a2.y * a3.x)
                 / V_box;
    Scalar3 b2 = Scalar(2.0 * M_PI)
                 * make_scalar3(a3.y * a1.z - a3.z * a1.y,
                                a3.z * a1.x - a3.x * a1.z,
                                a3.x * a1.y - a3.y * a1.x)
                 / V_box;
    Scalar3 b3 = Scalar(2.0 * M_PI)
                 * make_scalar3(a1.y * a2.z - a1.z * a2.y,
                                a1.z * a2.x - a1.x * a2.z,
                                a1.x * a2.y - a1.y * a2.x)
                 / V_box;

    unsigned int num_wave_vectors = mesh_dim.x * mesh_dim.y * mesh_dim.z;

    Scalar3 L = global_box.getL();
    Scalar temp = floor(((kappa * L.x / (M_PI * global_dim.x)) * pow(-log(EPS_HOC), 0.25)));
    int nbx = (int)temp;
    temp = floor(((kappa * L.y / (M_PI * global_dim.y)) * pow(-log(EPS_HOC), 0.25)));
    int nby = (int)temp;
    temp = floor(((kappa * L.z / (M_PI * global_dim.z)) * pow(-log(EPS_HOC), 0.25)));
    int nbz = (int)temp;

    if (local_fft)
        {
        unsigned int max_block_size;
        hipFuncAttributes attr;
        hipFuncGetAttributes(&attr, (const void*)gpu_compute_influence_function_kernel<true>);
        max_block_size = attr.maxThreadsPerBlock;

        unsigned int run_block_size = min(max_block_size, block_size);

        unsigned int n_blocks = num_wave_vectors / run_block_size;
        if (num_wave_vectors % run_block_size)
            n_blocks += 1;

        dim3 grid(n_blocks, 1, 1);

        hipLaunchKernelGGL((gpu_compute_influence_function_kernel<true>),
                           dim3(grid),
                           dim3(run_block_size),
                           0,
                           0,
                           mesh_dim,
                           num_wave_vectors,
                           global_dim,
                           d_inf_f,
                           d_k,
                           b1,
                           b2,
                           b3,
                           pidx,
                           pdim,
                           nbx,
                           nby,
                           nbz,
                           d_gf_b,
                           order,
                           kappa,
                           alpha);
        }
#ifdef ENABLE_MPI
    else
        {
        unsigned int max_block_size;
        hipFuncAttributes attr;
        hipFuncGetAttributes(&attr, (const void*)gpu_compute_influence_function_kernel<false>);
        max_block_size = attr.maxThreadsPerBlock;

        unsigned int run_block_size = min(max_block_size, block_size);

        unsigned int n_blocks = num_wave_vectors / run_block_size;
        if (num_wave_vectors % run_block_size)
            n_blocks += 1;

        dim3 grid(n_blocks, 1, 1);

        hipLaunchKernelGGL((gpu_compute_influence_function_kernel<false>),
                           dim3(grid),
                           dim3(run_block_size),
                           0,
                           0,
                           mesh_dim,
                           num_wave_vectors,
                           global_dim,
                           d_inf_f,
                           d_k,
                           b1,
                           b2,
                           b3,
                           pidx,
                           pdim,
                           nbx,
                           nby,
                           nbz,
                           d_gf_b,
                           order,
                           kappa,
                           alpha);
        }
#endif
    }

//! The developer has chosen not to document this function
__global__ void gpu_fix_exclusions_kernel(Scalar4* d_force,
                                          Scalar* d_virial,
                                          const size_t virial_pitch,
                                          const Scalar4* d_pos,
                                          const Scalar* d_charge,
                                          const BoxDim box,
                                          const unsigned int* d_n_neigh,
                                          const unsigned int* d_nlist,
                                          const Index2D nli,
                                          Scalar kappa,
                                          Scalar alpha,
                                          unsigned int* d_group_members,
                                          unsigned int group_size)
    {
    // start by identifying which particle we are to handle
    int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (group_idx < group_size)
        {
        unsigned int idx = d_group_members[group_idx];
        const Scalar sqrtpi = sqrtf(M_PI);
        unsigned int n_neigh = d_n_neigh[idx];
        Scalar4 postypei = __ldg(d_pos + idx);
        Scalar3 posi = make_scalar3(postypei.x, postypei.y, postypei.z);

        Scalar qi = __ldg(d_charge + idx);
        // initialize the force to 0
        Scalar4 force = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));
        Scalar virial[6];
        for (unsigned int i = 0; i < 6; i++)
            virial[i] = Scalar(0.0);
        unsigned int cur_j = 0;
        // prefetch neighbor index
        unsigned int next_j = d_nlist[nli(idx, 0)];

        for (int neigh_idx = 0; neigh_idx < n_neigh; neigh_idx++)
            {
                {
                // read the current neighbor index (MEM TRANSFER: 4 bytes)
                // prefetch the next value and set the current one
                cur_j = next_j;
                if (neigh_idx + 1 < n_neigh)
                    next_j = d_nlist[nli(idx, neigh_idx + 1)];

                // get the neighbor's position (MEM TRANSFER: 16 bytes)
                Scalar4 postypej = __ldg(d_pos + cur_j);
                Scalar3 posj = make_scalar3(postypej.x, postypej.y, postypej.z);

                Scalar qj = __ldg(d_charge + cur_j);

                // calculate dr (with periodic boundary conditions) (FLOPS: 3)
                Scalar3 dx = posi - posj;

                // apply periodic boundary conditions: (FLOPS 12)
                dx = box.minImage(dx);

                // calculate r squared (FLOPS: 5)
                Scalar rsq = dot(dx, dx);
                Scalar r = sqrtf(rsq);
                Scalar qiqj = qi * qj;
                Scalar expfac = fast::exp(-alpha * r);
                Scalar arg1 = kappa * r - alpha / Scalar(2.0) / kappa;
                Scalar arg2 = kappa * r + alpha / Scalar(2.0) / kappa;
                Scalar erffac = (::erf(arg1) * expfac + expfac - fast::erfc(arg2) * exp(alpha * r))
                                / (Scalar(2.0) * r);

                Scalar force_divr
                    = qiqj
                      * (expfac * Scalar(2.0) * kappa / sqrtpi * fast::exp(-arg1 * arg1)
                         - Scalar(0.5) * alpha
                               * (expfac * ::erfc(arg1) + fast::exp(alpha * r) * fast::erfc(arg2))
                         - erffac)
                      / rsq;

                // subtract long-range part of pair-interaction
                Scalar pair_eng = -qiqj * erffac;

                Scalar force_div2r = Scalar(0.5) * force_divr;
                virial[0] += dx.x * dx.x * force_div2r;
                virial[1] += dx.x * dx.y * force_div2r;
                virial[2] += dx.x * dx.z * force_div2r;
                virial[3] += dx.y * dx.y * force_div2r;
                virial[4] += dx.y * dx.z * force_div2r;
                virial[5] += dx.z * dx.z * force_div2r;

                force.x += dx.x * force_divr;
                force.y += dx.y * force_divr;
                force.z += dx.z * force_divr;

                force.w += pair_eng;
                }
            }
        force.w *= Scalar(0.5);
        d_force[idx].x += force.x;
        d_force[idx].y += force.y;
        d_force[idx].z += force.z;
        d_force[idx].w += force.w;
        for (unsigned int i = 0; i < 6; i++)
            d_virial[i * virial_pitch + idx] += virial[i];
        }
    }

//! The developer has chosen not to document this function
hipError_t gpu_fix_exclusions(Scalar4* d_force,
                              Scalar* d_virial,
                              const size_t virial_pitch,
                              const unsigned int Nmax,
                              const Scalar4* d_pos,
                              const Scalar* d_charge,
                              const BoxDim& box,
                              const unsigned int* d_n_ex,
                              const unsigned int* d_exlist,
                              const Index2D nex,
                              Scalar kappa,
                              Scalar alpha,
                              unsigned int* d_group_members,
                              unsigned int group_size,
                              int block_size)
    {
    dim3 grid(group_size / block_size + 1, 1, 1);
    dim3 threads(block_size, 1, 1);

    hipLaunchKernelGGL((gpu_fix_exclusions_kernel),
                       dim3(grid),
                       dim3(threads),
                       0,
                       0,
                       d_force,
                       d_virial,
                       virial_pitch,
                       d_pos,
                       d_charge,
                       box,
                       d_n_ex,
                       d_exlist,
                       nex,
                       kappa,
                       alpha,
                       d_group_members,
                       group_size);
    return hipSuccess;
    }

    } // namespace kernel
    } // end namespace md
    } // end namespace hoomd
