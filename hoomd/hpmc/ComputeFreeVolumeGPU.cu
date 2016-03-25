#include "ComputeFreeVolumeGPU.cuh"

#include "Moves.h"
#include "ShapeSphere.h"
#include "ShapeConvexPolygon.h"
#include "ShapeSpheropolygon.h"
#include "ShapePolyhedron.h"
#include "ShapeConvexPolyhedron.h"
#include "ShapeSpheropolyhedron.h"
#include "ShapeSimplePolygon.h"
#include "ShapeEllipsoid.h"
#include "ShapeFacetedSphere.h"
#include "ShapeSphinx.h"
#include "ShapeUnion.h"
#include "hoomd/TextureTools.h"

namespace hpmc
{

namespace detail
{

/*! \file IntegratorHPMCMonoImplicitGPU.cu
    \brief Definition of CUDA kernels and drivers for IntegratorHPMCMonoImplicit
*/

//! Texture for reading postype
scalar4_tex_t postype_tex;
//! Texture for reading orientation
scalar4_tex_t orientation_tex;

//! Compute the cell that a particle sits in
__device__ inline unsigned int compute_cell_idx(const Scalar3 p,
                                               const BoxDim& box,
                                               const Scalar3& ghost_width,
                                               const uint3& cell_dim,
                                               const Index3D& ci)
    {
    // find the bin each particle belongs in
    Scalar3 f = box.makeFraction(p,ghost_width);
    uchar3 periodic = box.getPeriodic();
    int ib = (unsigned int)(f.x * cell_dim.x);
    int jb = (unsigned int)(f.y * cell_dim.y);
    int kb = (unsigned int)(f.z * cell_dim.z);

    // need to handle the case where the particle is exactly at the box hi
    if (ib == (int)cell_dim.x && periodic.x)
        ib = 0;
    if (jb == (int)cell_dim.y && periodic.y)
        jb = 0;
    if (kb == (int)cell_dim.z && periodic.z)
        kb = 0;

    // identify the bin
    return ci(ib,jb,kb);
    }


//! Kernel to estimate the colloid overlap volume and the depletant free volume
/*! \param n_sample Number of probe depletant particles to generate
    \param type Type of depletant particle
    \param d_postype Particle positions and types by index
    \param d_orientation Particle orientation
    \param d_cell_size The size of each cell
    \param ci Cell indexer
    \param cli Cell list indexer
    \param d_cell_adj List of adjacent cells
    \param cadji Cell adjacency indexer
    \param cell_dim Dimensions of the cell list
    \param N number of particles
    \param num_types Number of particle types
    \param seed User chosen random number seed
    \param a Size of rotation move (per type)
    \param timestep Current timestep of the simulation
    \param dim Dimension of the simulation box
    \param box Simulation box
    \param d_n_overlap_all Total overlap counter (output value)
    \param ghost_width Width of ghost layer
    \param d_params Per-type shape parameters
*/
template< class Shape >
__global__ void gpu_hpmc_free_volume_kernel(unsigned int n_sample,
                                     unsigned int type,
                                     Scalar4 *d_postype,
                                     Scalar4 *d_orientation,
                                     const unsigned int *d_cell_size,
                                     const Index3D ci,
                                     const Index2D cli,
                                     const unsigned int *d_excell_idx,
                                     const unsigned int *d_excell_size,
                                     const Index2D excli,
                                     const uint3 cell_dim,
                                     const unsigned int N,
                                     const unsigned int num_types,
                                     const unsigned int seed,
                                     const unsigned int select,
                                     const unsigned int timestep,
                                     const unsigned int dim,
                                     const BoxDim box,
                                     unsigned int *d_n_overlap_all,
                                     Scalar3 ghost_width,
                                     const typename Shape::param_type *d_params)
    {
    unsigned int group = threadIdx.z;
    unsigned int offset = threadIdx.y;
    unsigned int group_size = blockDim.y;
    bool master = (offset == 0 && threadIdx.x == 0);
    unsigned int n_groups = blockDim.z;

    // determine sample idx
    unsigned int i;
    if (gridDim.y > 1)
        {
        // if gridDim.y > 1, then the fermi workaround is in place, index blocks on a 2D grid
        i = (blockIdx.x + blockIdx.y * 65535) * n_groups + group;
        }
    else
        {
        i = blockIdx.x * n_groups + group;
        }


    // load the per type pair parameters into shared memory
    extern __shared__ char s_data[];
    typename Shape::param_type *s_params = (typename Shape::param_type *)(&s_data[0]);

    // copy over parameters one int per thread for fast loads
        {
        unsigned int tidx = threadIdx.x+blockDim.x*threadIdx.y + blockDim.x*blockDim.y*threadIdx.z;
        unsigned int block_size = blockDim.x*blockDim.y*blockDim.z;
        unsigned int param_size = num_types*sizeof(typename Shape::param_type) / sizeof(int);

        for (unsigned int cur_offset = 0; cur_offset < param_size; cur_offset += block_size)
            {
            if (cur_offset + tidx < param_size)
                {
                ((int *)s_params)[cur_offset + tidx] = ((int *)d_params)[cur_offset + tidx];
                }
            }
        }

    unsigned int *s_overlap = (unsigned int *)(&s_params[num_types]);
    __shared__ unsigned int s_n_overlap;

    if (master)
        {
        s_overlap[group] = 0;
        }

    if (master && group == 0)
        {
        s_n_overlap = 0;
        }


    __syncthreads();

    bool active = true;

    if (i >= n_sample)
        {
        active = false;
        }

    // one RNG per particle
    SaruGPU rng(i, seed+select, timestep);

    unsigned int my_cell;

    // test depletant position
    vec3<Scalar> pos_i;
    quat<Scalar> orientation_i;
    Shape shape_i(orientation_i, s_params[type]);

    if (active)
        {
        // select a random particle coordinate in the box
        Scalar xrand = rng.template s<Scalar>();
        Scalar yrand = rng.template s<Scalar>();
        Scalar zrand = rng.template s<Scalar>();

        Scalar3 f = make_scalar3(xrand, yrand, zrand);
        pos_i = vec3<Scalar>(box.makeCoordinates(f));

        if (shape_i.hasOrientation())
            {
            shape_i.orientation = generateRandomOrientation(rng);
            }

        // find cell the particle is in
        Scalar3 p = vec_to_scalar3(pos_i);
        my_cell = compute_cell_idx(p, box, ghost_width, cell_dim, ci);
        }

    if (active)
        {
        // loop over neighboring cells and check for overlaps
        unsigned int excell_size = d_excell_size[my_cell];

        for (unsigned int k = 0; k < excell_size; k += group_size)
            {
            unsigned int local_k = k + offset;
            if (local_k < excell_size)
                {
                // read in position, and orientation of neighboring particle
                #if ( __CUDA_ARCH__ > 300)
                unsigned int j = __ldg(&d_excell_idx[excli(local_k, my_cell)]);
                #else
                unsigned int j = d_excell_idx[excli(local_k, my_cell)];
                #endif

                Scalar4 postype_j = texFetchScalar4(d_postype, postype_tex, j);
                Scalar4 orientation_j = make_scalar4(1,0,0,0);
                Shape shape_j(quat<Scalar>(orientation_j), s_params[__scalar_as_int(postype_j.w)]);
                if (shape_j.hasOrientation())
                    shape_j.orientation = quat<Scalar>(texFetchScalar4(d_orientation, orientation_tex, j));

                // put particle j into the coordinate system of particle i
                vec3<Scalar> r_ij = vec3<Scalar>(postype_j) - pos_i;
                r_ij = vec3<Scalar>(box.minImage(vec_to_scalar3(r_ij)));

                // check for overlaps
                OverlapReal rsq = dot(r_ij,r_ij);
                OverlapReal DaDb = shape_i.getCircumsphereDiameter() + shape_j.getCircumsphereDiameter();

                if (rsq*OverlapReal(4.0) <= DaDb * DaDb)
                    {
                    // circumsphere overlap
                    unsigned int err_count;
                    if (test_overlap(r_ij, shape_i, shape_j, err_count))
                        {
                        atomicAdd(&s_overlap[group],1);
                        break;
                        }
                    }
                }
            }
        }

    __syncthreads();

    unsigned int overlap = s_overlap[group];

    if (master)
        {
        // this thread counts towards the total overlap volume
        if (overlap)
            {
            atomicAdd(&s_n_overlap, 1);
            }
        }

    __syncthreads();

    if (master && group == 0)
        {
        // final tally into global mem
        atomicAdd(d_n_overlap_all, s_n_overlap);
        }
    }

//! Kernel driver for gpu_hpmc_free_volume_kernel()
/*! \param args Bundled arguments
    \param d_params Per-type shape parameters
    \returns Error codes generated by any CUDA calls, or cudaSuccess when there is no error

    This templatized method is the kernel driver for parallel update of any shape. It is instantiated for every shape at the
    bottom of this file.

    \ingroup hpmc_kernels
*/
template< class Shape >
cudaError_t gpu_hpmc_free_volume(const hpmc_free_volume_args_t& args, const typename Shape::param_type *d_params)
    {
    assert(args.d_postype);
    assert(args.d_orientation);
    assert(args.d_cell_size);
    assert(args.group_size >= 1);
    assert(args.group_size <= 32);  // note, really should be warp size of the device
    assert(args.block_size%(args.stride*args.group_size)==0);


    // bind the textures
    postype_tex.normalized = false;
    postype_tex.filterMode = cudaFilterModePoint;
    cudaError_t error = cudaBindTexture(0, postype_tex, args.d_postype, sizeof(Scalar4)*args.max_n);
    if (error != cudaSuccess)
        return error;

    orientation_tex.normalized = false;
    orientation_tex.filterMode = cudaFilterModePoint;
    error = cudaBindTexture(0, orientation_tex, args.d_orientation, sizeof(Scalar4)*args.max_n);
    if (error != cudaSuccess)
        return error;

    // reset counters
    cudaMemsetAsync(args.d_n_overlap_all,0, sizeof(unsigned int));

    // determine the maximum block size and clamp the input block size down
    static int max_block_size = -1;
    static int sm = -1;
    if (max_block_size == -1)
        {
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, gpu_hpmc_free_volume_kernel<Shape>);
        max_block_size = attr.maxThreadsPerBlock;
        sm = attr.binaryVersion;
        }

    // setup the grid to run the kernel
    unsigned int n_groups = min(args.block_size, (unsigned int)max_block_size) / args.group_size / args.stride;

    dim3 threads(args.stride, args.group_size, n_groups);
    dim3 grid( args.n_sample / n_groups + 1, 1, 1);

    // hack to enable grids of more than 65k blocks
    if (sm < 30 && grid.x > 65535)
        {
        grid.y = grid.x / 65535 + 1;
        grid.x = 65535;
        }

    unsigned int shared_bytes = args.num_types * sizeof(typename Shape::param_type) + n_groups*sizeof(unsigned int);

    gpu_hpmc_free_volume_kernel<Shape><<<grid, threads, shared_bytes>>>(
                                                     args.n_sample,
                                                     args.type,
                                                     args.d_postype,
                                                     args.d_orientation,
                                                     args.d_cell_size,
                                                     args.ci,
                                                     args.cli,
                                                     args.d_excell_idx,
                                                     args.d_excell_size,
                                                     args.excli,
                                                     args.cell_dim,
                                                     args.N,
                                                     args.num_types,
                                                     args.seed,
                                                     args.select,
                                                     args.timestep,
                                                     args.dim,
                                                     args.box,
                                                     args.d_n_overlap_all,
                                                     args.ghost_width,
                                                     d_params);

    return cudaSuccess;
    }

//! Template instantiations

//! Overlap volume count for ShapeSphere
template cudaError_t gpu_hpmc_free_volume<ShapeSphere>(const hpmc_free_volume_args_t& args,
                                                  const typename ShapeSphere::param_type *d_params);

//! Overlap volume count for ShapeConvexPolygon
template cudaError_t gpu_hpmc_free_volume<ShapeConvexPolygon>(const hpmc_free_volume_args_t& args,
                                                         const typename ShapeConvexPolygon::param_type *d_params);

//! Overlap volume count for ShapePolyhedron
template cudaError_t gpu_hpmc_free_volume<ShapePolyhedron>(const hpmc_free_volume_args_t& args,
                                                      const typename ShapePolyhedron::param_type *d_params);

//! Overlap volume count for ShapeConvexPolyhedron
template cudaError_t gpu_hpmc_free_volume<ShapeConvexPolyhedron<8> >(const hpmc_free_volume_args_t& args,
                                                            const typename ShapeConvexPolyhedron<8>::param_type *d_params);
template cudaError_t gpu_hpmc_free_volume<ShapeConvexPolyhedron<16> >(const hpmc_free_volume_args_t& args,
                                                            const typename ShapeConvexPolyhedron<16>::param_type *d_params);
template cudaError_t gpu_hpmc_free_volume<ShapeConvexPolyhedron<32> >(const hpmc_free_volume_args_t& args,
                                                            const typename ShapeConvexPolyhedron<32>::param_type *d_params);
template cudaError_t gpu_hpmc_free_volume<ShapeConvexPolyhedron<64> >(const hpmc_free_volume_args_t& args,
                                                            const typename ShapeConvexPolyhedron<64>::param_type *d_params);
template cudaError_t gpu_hpmc_free_volume<ShapeConvexPolyhedron<128> >(const hpmc_free_volume_args_t& args,
                                                            const typename ShapeConvexPolyhedron<128>::param_type *d_params);

//! Overlap volume count for ShapeSpheropolyhedron
template cudaError_t gpu_hpmc_free_volume<ShapeSpheropolyhedron<8> >(const hpmc_free_volume_args_t& args,
                                                            const typename ShapeSpheropolyhedron<8>::param_type *d_params);
template cudaError_t gpu_hpmc_free_volume<ShapeSpheropolyhedron<16> >(const hpmc_free_volume_args_t& args,
                                                            const typename ShapeSpheropolyhedron<16>::param_type *d_params);
template cudaError_t gpu_hpmc_free_volume<ShapeSpheropolyhedron<32> >(const hpmc_free_volume_args_t& args,
                                                            const typename ShapeSpheropolyhedron<32>::param_type *d_params);
template cudaError_t gpu_hpmc_free_volume<ShapeSpheropolyhedron<64> >(const hpmc_free_volume_args_t& args,
                                                            const typename ShapeSpheropolyhedron<64>::param_type *d_params);
template cudaError_t gpu_hpmc_free_volume<ShapeSpheropolyhedron<128> >(const hpmc_free_volume_args_t& args,
                                                            const typename ShapeSpheropolyhedron<128>::param_type *d_params);

//! Overlap volume count for ShapeSimplePolygon
template cudaError_t gpu_hpmc_free_volume<ShapeSimplePolygon>(const hpmc_free_volume_args_t& args,
                                                         const typename ShapeSimplePolygon::param_type *d_params);

//! Overlap volume count for ShapeEllipsoid
template cudaError_t gpu_hpmc_free_volume<ShapeEllipsoid>(const hpmc_free_volume_args_t& args,
                                                     const typename ShapeEllipsoid::param_type *d_params);

//! Overlap volume count for ShapeSpheropolygon
template cudaError_t gpu_hpmc_free_volume<ShapeSpheropolygon>(const hpmc_free_volume_args_t& args,
                                                         const typename ShapeSpheropolygon::param_type *d_params);

//! Overlap volume count for ShapeFacetedSphere
template cudaError_t gpu_hpmc_free_volume<ShapeFacetedSphere>(const hpmc_free_volume_args_t& args,
                                                         const typename ShapeFacetedSphere::param_type *d_params);
#ifdef ENABLE_SPHINX_GPU
//! Overlap volume count for ShapeSphinx
template cudaError_t gpu_hpmc_free_volume<ShapeSphinx>(const hpmc_free_volume_args_t& args,
                                                       const typename ShapeSphinx::param_type *d_params);
#endif

//! Overlap volume count for ShapeUnion<ShapeSphere>
template cudaError_t gpu_hpmc_free_volume<ShapeUnion<ShapeSphere> >(const hpmc_free_volume_args_t& args,
                                                         const typename ShapeUnion<ShapeSphere>::param_type *d_params);

} // detail

} // hpmc
